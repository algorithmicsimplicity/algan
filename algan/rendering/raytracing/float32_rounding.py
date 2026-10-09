"""How far float32 rounding moves geometry on screen, and the warning when it shows.

Algan is float32 end to end, and there is no wider mode to switch to: the
timeline stores world positions as float32 rows, the projection casts every
primitive to float32 in world space, the kernels are float32 (compiled with
``fast_math``), and Apple's Metal backend has no float64 at all. Three roundings
follow, each a few ulps of the *largest coordinate involved*, not of the
geometry's own size:

* **Stored positions.** A point ``p`` is held to about ``eps * |p_i|`` per
  axis, so geometry far from the origin keeps only as much shape as that
  spacing allows.
* **The kernels' world-space arithmetic.** A hit is formed as
  ``eye + t * dir - center`` (``raster_taichi.py``'s ``_bez_pixel_hit``), so
  it is rounded at the eye's magnitude as well as the point's.
* **The ray directions.** ``_generate_ray`` forms each pixel's point on the
  screen plane in world space and subtracts the eye from it; both round at
  their own magnitudes. That error is the same for every scene the camera
  sees, and with a near-orthographic camera's huge focal length it is the
  largest of the three.

Only rounding *across* the view direction moves anything on screen, which is
why an axis-aligned camera a thousand units out is still exact: its large
coordinate lies along the view.

The estimate, per visible point ``p`` at depth ``z`` in front of an eye ``o``
looking along unit ``n``, with a focal length of ``K`` output pixels per
radian, ``k0`` pixels per world unit on the screen plane and screen point ``s``::

    error(p) = eps * sqrt( K^2 * across(p^2 + o^2) / z^2
                           + k0^2 * across(s^2) + K^2 * across(n^2) )

where ``across(v^2) = sum_i v_i^2 (1 - n_i^2)`` keeps the part of a per-axis
rounding vector that lies across the view. The first term is the point's own
rounding and the hit arithmetic (the two added in quadrature), the other two
the ray direction's (against a float64 emulation of ``_generate_ray``, the
worst pixel's error is 0.7-1.1x of it). It is the *spacing* of representable
positions, not an expected error, so a value of 1 means shapes are drawn from
a lattice one pixel across.

Calibration (1280x720, camera turned 35 degrees, text brought a unit in front
of it against the same text left at the origin): estimates of 0.16 and 0.37 px
render indistinguishably, 0.66 px steps the glyph edges, 1.0-1.6 px makes the
strokes uneven, 2.6 px and up doubles and dashes them. Text at the origin seen
through ``set_near_orthographic`` behaves the same way: clean to 0.35 px
(distance 2e4), uneven from 0.53 (3e4), a doubled stroke at 1.75 (1e5).

Each projection (``RayTracedTrianglePrimitive``, its logical-PN subclass and
``RayTracedBezierCircuitPrimitive``) reduces its own control points or vertices
to one record on its device -- no sync -- and the render loop reads the
batch's records once, after the batch has rendered, keeping the render's
worst. :class:`Float32RoundingJob` is the per-render scope; it warns once, at
the end of the render, with :class:`~algan.errors.Float32PrecisionWarning`.
"""

from __future__ import annotations

import bisect
import math
import warnings
from typing import NamedTuple

import torch
import torch.nn.functional as F

from algan.errors import Float32PrecisionWarning, _user_stacklevel
from algan.logging.logger import get_logger
from algan.rendering.mps_compat import clamp_floor
from algan.rendering.raytracing.utils import _expand_frames, _flat_frames

logger = get_logger("raytracing")

#: float32's machine epsilon, 2**-23: the spacing of floats in [1, 2). A value
#: ``x`` is represented to within half of ``EPS32 * |x|``; the estimate uses
#: the whole spacing.
EPS32 = float(torch.finfo(torch.float32).eps)

#: Estimated on-screen rounding, in output pixels, from which it is visible:
#: edges step and glyphs grow uneven strokes. Calibrated by rendering text
#: brought in front of a distant camera against the same text left at the
#: origin (the two are identical in exact arithmetic).
VISIBLE_ERROR_PX = 0.5
#: From here shapes break up: doubled strokes, dashed outlines, gaps.
BROKEN_ERROR_PX = 2.0

#: Points this far outside the frame (as a fraction of its half-size) still
#: count as visible: a stroke or a glow reaches in from outside.
_SCREEN_MARGIN = 1.1
#: Upper bound on ``frames * points`` evaluated at once, which bounds the
#: estimate's temporaries to about 60 MB however large the batch.
_CHUNK_ELEMENTS = 1 << 20

#: A finding this close in front of the eye, as a fraction of the eye's own
#: distance from the origin, is something placed in front of a distant camera
#: rather than the camera's focal length alone; it changes the advice.
_CLOSE_IN_FRONT = 0.1

#: Field order of a record's ``values``.
_ERROR, _DEPTH, _DISTANCE, _EYE_DISTANCE = range(4)


class CameraRounding(NamedTuple):
    """The per-frame camera terms of the estimate, for one projection."""

    #: [T, 3] eye positions, and one of them as the reference the points are
    #: centred on before the frame dot products (keeps a HUD's depth exact).
    eye: torch.Tensor
    reference: torch.Tensor
    #: [T, 4, 6] rows applied to ``[p - reference, p^2]``: forward, the two
    #: screen rows, and the across-the-view weights.
    rows: torch.Tensor
    #: [T, 4] what to subtract from those products: ``(o - reference) . n``,
    #: ``(o - reference) . b0``, ``(o - reference) . b1`` and ``-across(o^2)``.
    offsets: torch.Tensor
    #: [T, 2] ``(o - s) . b0`` and ``(o - s) . b1``: where the optical axis
    #: lands on the screen (zero for a camera whose screen is centred on it).
    centre: torch.Tensor
    #: [T] eye-to-screen distance along the forward axis.
    screen_distance: torch.Tensor
    #: [T] ``K^2``, the focal length in output pixels, squared.
    focal_sq: torch.Tensor
    #: [T] the ray-direction terms, ``k0^2 across(s^2) + K^2 across(n^2)``.
    ray_sq: torch.Tensor
    #: The visible half-extent of the screen in its own coordinates.
    limit_u: float
    limit_v: float


def camera_rounding(camera, num_frames, device) -> CameraRounding:
    """The camera terms for a projection over ``num_frames`` frames.

    Reads the snapshot the projection reads (``ray_origin``, ``screen_point``,
    ``screen_basis``) and the output resolution, so the estimate is in the
    pixels of the delivered video whatever the supersampling.
    """
    eye = _expand_frames(_flat_frames(camera.ray_origin, (3,)), num_frames).to(device)
    screen = _expand_frames(_flat_frames(camera.screen_point, (3,)), num_frames).to(
        device
    )
    basis = _expand_frames(_flat_frames(camera.screen_basis, (3, 3)), num_frames).to(
        device
    )
    height = float(getattr(camera, "output_screen_height", camera.screen_height))
    width = float(getattr(camera, "output_screen_width", camera.screen_width))
    # The camera projects onto the plane whose normal is the basis' third row,
    # then reads screen coordinates as dot products with the first two (see
    # ``utils._pixel_bases``); the rows need not be orthogonal or unit.
    forward = F.normalize(basis[:, 2], p=2, dim=-1)
    across = 1.0 - forward.square()
    screen_distance = ((screen - eye) * forward).sum(-1)
    pixels_per_unit = 0.5 * height * basis[:, :2].norm(p=2, dim=-1).amax(-1)
    focal_sq = (pixels_per_unit * screen_distance).square()
    ray_sq = pixels_per_unit.square() * (screen.square() * across).sum(-1) + (
        focal_sq * (forward.square() * across).sum(-1)
    )
    reference = eye[0]
    zeros = torch.zeros_like(forward)
    rows = torch.stack(
        (
            torch.cat((forward, zeros), -1),
            torch.cat((basis[:, 0], zeros), -1),
            torch.cat((basis[:, 1], zeros), -1),
            torch.cat((zeros, across), -1),
        ),
        dim=1,
    )
    eye_offset = eye - reference
    offsets = torch.stack(
        (
            (eye_offset * forward).sum(-1),
            (eye_offset * basis[:, 0]).sum(-1),
            (eye_offset * basis[:, 1]).sum(-1),
            -(eye.square() * across).sum(-1),
        ),
        dim=-1,
    )
    centre = torch.stack(
        (
            ((eye - screen) * basis[:, 0]).sum(-1),
            ((eye - screen) * basis[:, 1]).sum(-1),
        ),
        dim=-1,
    )
    return CameraRounding(
        eye=eye,
        reference=reference,
        rows=rows,
        offsets=offsets,
        centre=centre,
        screen_distance=screen_distance,
        focal_sq=focal_sq,
        ray_sq=ray_sq,
        limit_u=_SCREEN_MARGIN * width / max(height, 1.0),
        limit_v=_SCREEN_MARGIN,
    )


def worst_point_rounding(points, terms: CameraRounding, *, skip_below=None):
    """Reduce ``points`` to the record of the one whose rounding shows most.

    ``points`` is ``[Tc, M, 3]`` world positions, ``Tc`` either 1 (static) or
    the camera's frame count. Returns ``(values, index)`` on the points'
    device without synchronizing it: ``values`` is float32 ``[4]`` -- the
    estimated error in output pixels, the point's depth in front of the eye,
    its distance from the origin and the eye's -- and ``index`` is int64
    ``[2]``, the frame within the batch and the point. A batch with nothing
    on screen records an error of 0.

    ``skip_below`` (pixels; default ``VISIBLE_ERROR_PX``, 0 to always
    evaluate) lets a CPU projection skip the per-point pass when a bound from
    the points' bounding box shows none of them can reach it, recording an
    error of 0 instead. On another device the bound would cost a sync, and
    the pass is cheap there, so it always runs.
    """
    points = points.float()
    device = points.device
    num_points = int(points.shape[1])
    num_frames = int(terms.eye.shape[0])
    best = torch.zeros((), dtype=torch.float32, device=device)
    best_at = torch.zeros((), dtype=torch.int64, device=device)
    if num_points == 0 or points.shape[0] not in (1, num_frames):
        return _record(points, terms, best, best_at, max(num_points, 1))
    if skip_below is None:
        skip_below = VISIBLE_ERROR_PX
    if (
        skip_below > 0
        and device.type == "cpu"
        and not _may_reach(points, terms, skip_below)
    ):
        return _record(points, terms, best, best_at, num_points)
    static = points.shape[0] == 1
    point_step = min(num_points, _CHUNK_ELEMENTS)
    frame_step = max(1, _CHUNK_ELEMENTS // point_step)
    for first in range(0, num_points, point_step):
        last = min(num_points, first + point_step)
        lifted = _lifted(points[:, first:last], terms) if static else None
        for start in range(0, num_frames, frame_step):
            stop = min(num_frames, start + frame_step)
            if not static:
                lifted = _lifted(points[start:stop, first:last], terms)
            score = _chunk_scores(lifted, terms, start, stop)
            chunk_best, chunk_at = score.reshape(-1).max(0)
            # The flat index into [frames, points] of the whole batch.
            frame = start + torch.div(chunk_at, last - first, rounding_mode="floor")
            at = frame * num_points + first + chunk_at % (last - first)
            better = chunk_best > best
            best = torch.where(better, chunk_best, best)
            best_at = torch.where(better, at, best_at)
    return _record(points, terms, best, best_at, num_points)


def _may_reach(points, terms, threshold_px):
    """Whether any of ``points`` could round by ``threshold_px`` on screen.

    Conservative, from the points' bounding box alone: no point's rounding
    term exceeds what the box's farthest corner gives, so a point can only
    reach the threshold at a depth below some ``z_crit`` per frame, and an
    on-screen point that shallow lies within ``z_crit * sqrt(1 + L^2)`` of the
    eye (``L`` bounding the tangent of the frame's half-diagonal, generously).
    A box farther than that from every frame's eye cannot reach it. A frame
    whose ray directions alone reach it needs only something on screen, which
    the box cannot rule out. Two reductions over the points; everything else
    is per frame.
    """
    lo, hi = _bounding_box(points)
    farthest_sq = torch.maximum(lo.square(), hi.square())
    across = terms.rows[:, 3, 3:]
    bound_sq = across @ farthest_sq - terms.offsets[:, 3]
    margin_sq = (threshold_px / EPS32) ** 2 - terms.ray_sq
    z_crit_sq = terms.focal_sq * bound_sq / clamp_floor(margin_sq, 1e-30)
    scale = terms.rows[:, 1:3, :3].norm(p=2, dim=-1).amin(-1)
    reach = terms.centre.abs() + torch.tensor(
        (terms.limit_u, terms.limit_v), dtype=terms.centre.dtype
    )
    tangent_sq = (2.0 * reach.norm(p=2, dim=-1)) ** 2 / clamp_floor(
        (terms.screen_distance * scale).square(), 1e-30
    )
    radius_sq = z_crit_sq * (1.0 + tangent_sq)
    gap = (lo - terms.eye).clamp_min(0) + (terms.eye - hi).clamp_min(0)
    near = gap.square().sum(-1) <= radius_sq
    return bool(((margin_sq <= 0) | near).any())


def _bounding_box(points):
    """Per-axis min and max of ``[..., 3]`` points, in one pass.

    Read as rows of 32 points, not of one: a reduction over an outer axis
    with only three lanes left is ~10x slower on the CPU (measured 220 ms
    against 18 ms on a 38-frame, 290k-point logical-PN mesh).
    """
    flat = points.reshape(-1)
    whole = flat.numel() // 96 * 96
    parts = []
    if whole:
        lo, hi = torch.aminmax(flat[:whole].view(-1, 96), dim=0)
        parts.append((lo.view(32, 3).amin(0), hi.view(32, 3).amax(0)))
    if whole < flat.numel():
        parts.append(torch.aminmax(flat[whole:].view(-1, 3), dim=0))
    lo = torch.stack([part[0] for part in parts]).amin(0)
    hi = torch.stack([part[1] for part in parts]).amax(0)
    return lo, hi


def _lifted(points, terms):
    """``[p - reference, p^2]``, what the frame rows are applied to.

    Depth and screen position come from points centred on one eye: a point a
    unit in front of an eye a thousand units out keeps its depth exactly
    through the subtraction, where the difference of two absolute dot
    products would keep only a few digits of it.
    """
    return torch.cat((points - terms.reference, points.square()), -1)


def _chunk_scores(lifted, terms, start, stop):
    """``(error / EPS32)^2`` per frame in ``[start, stop)`` and point, 0 off screen."""
    count = lifted.shape[1]
    rows = terms.rows[start:stop]
    if lifted.shape[0] == 1:
        products = (lifted[0] @ rows.reshape(-1, 6).T).view(count, stop - start, 4)
        products = products.transpose(0, 1)
    else:
        products = torch.bmm(lifted, rows.transpose(1, 2))
    products = products - terms.offsets[start:stop, None, :]
    depth, gu, gv, across_sq = products.unbind(-1)
    distance = terms.screen_distance[start:stop, None]
    centre = terms.centre[start:stop]
    u = centre[:, None, 0] * depth + distance * gu
    v = centre[:, None, 1] * depth + distance * gv
    visible = (
        (depth > 0)
        & (u.abs() <= terms.limit_u * depth)
        & (v.abs() <= terms.limit_v * depth)
    )
    score = terms.focal_sq[start:stop, None] * across_sq / depth.square()
    score = score + terms.ray_sq[start:stop, None]
    return torch.where(visible, score, torch.zeros_like(score))


def record_projection(
    primitive, camera, points, points_per_element=1, group_sizes=None
):
    """Attach a projection's worst-rounding record to ``primitive``.

    ``points`` is the primitive's world geometry, ``[Tc, ..., 3]``; each run of
    ``points_per_element`` consecutive points is one element (a triangle's
    corners, a cubic's controls), and ``group_sizes`` -- element counts, a
    tensor -- optionally groups consecutive elements further (cubics into
    circuits). The record names the frame and the (grouped) element, which the
    render loop resolves to a Mob through ``_member_owners``.

    Never raises: the estimate is advice, and a render must not fail for it.
    """
    try:
        num_frames = int(camera.ray_origin.shape[0])
        flat = points.reshape(points.shape[0], -1, 3)
        terms = camera_rounding(camera, num_frames, flat.device)
        values, index = worst_point_rounding(flat, terms)
        element = torch.div(index[1], points_per_element, rounding_mode="floor")
        if group_sizes is not None:
            ends = group_sizes.reshape(-1).to(element.device).long().cumsum(0)
            element = torch.searchsorted(ends, element.view(1), right=True)[0]
        primitive._rt_float32_record = (values, torch.stack((index[0], element)))
    except Exception:  # noqa: BLE001
        logger.debug("Skipped the float32 rounding estimate.", exc_info=True)
        primitive._rt_float32_record = None


def _record(points, terms, best, best_at, num_points):
    frame = torch.div(best_at, num_points, rounding_mode="floor")
    point = best_at - frame * num_points
    if int(points.shape[1]) == 0:
        position = torch.zeros(3, dtype=torch.float32, device=points.device)
    else:
        # Point first, then frame: two small gathers, no copy of the batch
        # and no sync (the indices stay on the device).
        source_frame = frame if points.shape[0] > 1 else torch.zeros_like(frame)
        position = points.index_select(1, point.view(1))
        position = position.index_select(0, source_frame.view(1)).reshape(3)
    eye = terms.eye.index_select(0, frame.view(1))[0]
    forward = terms.rows.index_select(0, frame.view(1))[0, 0, :3]
    values = torch.stack(
        (
            EPS32 * best.sqrt(),
            ((position - eye) * forward).sum(),
            position.norm(),
            eye.norm(),
        )
    )
    return values, torch.stack((frame, point))


class Float32Finding(NamedTuple):
    """The worst rounding a render job has seen, resolved to its Mob."""

    error_px: float
    mob: object
    distance: float
    depth: float
    eye_distance: float
    time: float


def describe_owner(mob) -> str:
    """``Tex``, ``Square 'axis'``, or ``a BezierCircuitCubic in Text 'title'``.

    A primitive is built by whichever Mob draws it, which for text and most
    composite shapes is a part nobody named; the part's outermost ancestor is
    what the author wrote, so it is named alongside.
    """
    if mob is None:
        return "a Mob"
    own = mob._describe() if hasattr(mob, "_describe") else type(mob).__name__
    root = mob
    seen = set()
    while getattr(root, "parents", None) and id(root) not in seen:
        seen.add(id(root))
        root = root.parents[0]
    if root is mob:
        return own
    outer = root._describe() if hasattr(root, "_describe") else type(root).__name__
    text = _quoted_text(root)
    if text:
        outer = f"{outer} {text}"
    return f"a {own} in {outer}"


def _quoted_text(mob):
    """A Text/Tex root's own words, shortened, so a page of labels names one."""
    for name in ("text", "original_text", "tex_strings"):
        value = mob.__dict__.get(name)
        if value is None:
            continue
        if isinstance(value, (tuple, list)):
            value = " ".join(str(part) for part in value)
        value = " ".join(str(value).split())
        if not value:
            return ""
        # Quoted by hand: repr would double every backslash of a TeX string.
        return "'" + (value if len(value) <= 30 else value[:27] + "...") + "'"
    return ""


def _owner_of(primitive, element):
    """The Mob that built ``element`` of a projected primitive, or ``None``.

    ``_member_owners`` / ``_member_ends`` are host lists the render loop
    attaches when it assembles the primitive from per-Mob parts: owner ``i``
    built elements ``[_member_ends[i - 1], _member_ends[i])``. Without
    ``_member_ends`` each owner built one element (the batched circuit build,
    one circuit per Mob), or a lone owner built them all.
    """
    owners = getattr(primitive, "_member_owners", None)
    if not owners:
        return None
    ends = getattr(primitive, "_member_ends", None)
    member = int(element) if ends is None else bisect.bisect_right(ends, int(element))
    return owners[min(max(member, 0), len(owners) - 1)]


class Float32RoundingJob:
    """One render job's worst on-screen float32 rounding.

    Lives on the Scene for the duration of ``get_frames`` (and is shared by a
    camera-view pass, which copies the Scene's attributes), so a nested pass
    adds to the job that started it instead of starting -- and warning for --
    one of its own.
    """

    def __init__(self):
        self.worst = None
        self.batches = 0

    def note_batch(self, primitives, frame_time):
        """Read one rendered batch's records and keep the worst.

        ``frame_time`` maps a frame offset within the batch to seconds. The
        records are read in one transfer per batch, after the batch has
        rendered, so the read never waits on anything still queued.
        """
        with_records = [
            primitive
            for primitive in primitives
            if getattr(primitive, "_rt_float32_record", None) is not None
        ]
        if not with_records:
            return
        self.batches += 1
        # One transfer per device the batch projected on (normally one).
        records = [primitive._rt_float32_record for primitive in with_records]
        by_device = {}
        for k, record in enumerate(records):
            by_device.setdefault(record[0].device, []).append(k)
        order = [k for ks in by_device.values() for k in ks]
        values = torch.cat(
            [
                torch.stack([records[k][0] for k in ks]).cpu()
                for ks in by_device.values()
            ]
        )
        index = torch.cat(
            [
                torch.stack([records[k][1] for k in ks]).cpu()
                for ks in by_device.values()
            ]
        )
        with_records = [with_records[k] for k in order]
        errors = values[:, _ERROR]
        k = int(torch.argmax(errors))
        error = float(errors[k])
        if not math.isfinite(error) or error <= 0.0:
            return
        if self.worst is not None and error <= self.worst.error_px:
            return
        frame, element = (int(v) for v in index[k].tolist())
        row = values[k].tolist()
        self.worst = Float32Finding(
            error_px=error,
            mob=_owner_of(with_records[k], element),
            distance=row[_DISTANCE],
            depth=row[_DEPTH],
            eye_distance=row[_EYE_DISTANCE],
            time=float(frame_time(frame)),
        )

    def report(self):
        """Warn once if the job's worst rounding reached the visible level."""
        finding = self.worst
        if finding is None or finding.error_px < VISIBLE_ERROR_PX:
            return
        warnings.warn(
            precision_message(finding),
            Float32PrecisionWarning,
            stacklevel=_user_stacklevel(),
        )


def precision_message(finding: Float32Finding) -> str:
    """The warning's text: what, how bad, why, and what to do about it."""
    if finding.error_px >= BROKEN_ERROR_PX:
        verdict = "look broken (doubled or dashed strokes, gaps)"
    else:
        verdict = "look grainy (uneven strokes, stepped edges)"
    where = (
        f"this point is {_units(finding.distance, 4)} from the origin, "
        f"{_units(finding.depth, 3)} in front of a camera "
        f"{_units(finding.eye_distance, 4)} from the origin"
    )
    if finding.depth < _CLOSE_IN_FRONT * finding.eye_distance:
        # Something parked close in front of a distant eye: a HUD.
        advice = (
            "Keep the camera, and anything placed just in front of it, within "
            "a few hundred units of the origin -- for a flat look, a narrow "
            "view such as camera.set_near_orthographic(distance=80) rather "
            "than a large distance; place it deeper in front of the camera, "
            "since the rounding shrinks as 1/depth; or turn the scene instead "
            "of the camera"
        )
    else:
        # A distant camera's own long focal length.
        advice = (
            "Bring the camera closer to the origin -- for a flat look, "
            "camera.set_near_orthographic() at its default distance rather "
            "than a larger one -- or turn the scene instead of the camera"
        )
    return (
        f"float32 rounding will make {describe_owner(finding.mob)} {verdict}: "
        f"it moves its points by up to about {finding.error_px:.2g} pixels on "
        f"screen (worst at t={finding.time:.2f}s). Algan stores and renders "
        f"positions as float32, about 7 significant digits, so a point is "
        f"only as exact as its largest coordinate allows; {where}, and the "
        f"camera magnifies that rounding by its focal length over the depth. "
        f"{advice}: a camera looking along an axis loses no precision across "
        f"its view."
    )


def _units(value, digits):
    """``1 unit``, ``0.25 units``, ``1000 units``."""
    text = f"{value:.{digits}g}"
    return f"{text} unit" if text == "1" else f"{text} units"
