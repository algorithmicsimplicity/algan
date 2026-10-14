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

The job keeps a second finding: **depth order**. A primary hit's depth is
rounded at the eye's distance from the origin, so two flat shapes that overlap
on screen trade places when their depths differ by less than a few spacings of
it (:data:`DEPTH_ORDER_SPACINGS`). Near the origin that is inside the
renderer's coplanar tie bins, where draw order decides; with the eye past about
210 units it is not, and layers stacked at one depth -- ordinary practice, and
what the backpropagation video's network panel did under a camera 10,000 units
out -- flicker through each other. When the eye is that far out the raster
pipeline scans its sorted fragment stream for such pairs
(:func:`depth_order_record`, no sync), the tracer attaches the batch's records
to its first primitive (:func:`attach_depth_order`), and the job reads them
with the batch's projection records and warns once, naming both Mobs.
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
#: From here shapes break up: doubled strokes, dashed outlines, gaps. The
#: calibration saw uneven strokes up to 1.6 px and a doubled one by 1.75 (the
#: old 1e5 ``set_near_orthographic`` default, turned).
BROKEN_ERROR_PX = 1.5

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
    what the author wrote, so it is named alongside -- short of the camera,
    which carries every panel riding it and so would name them all alike.
    """
    from algan.rendering.camera import Camera

    if mob is None:
        return "a Mob"
    own = mob._describe() if hasattr(mob, "_describe") else type(mob).__name__
    root = mob
    seen = set()
    while getattr(root, "parents", None) and id(root) not in seen:
        seen.add(id(root))
        if isinstance(root.parents[0], Camera):
            break
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
        # The depth-order check (depth_order_record): the render's closest
        # call, its share of the resolution, and how many pixel overlaps the
        # render saw.
        self.depth = None
        self._depth_ratio = math.inf
        self.depth_pairs = 0

    def _read_depth_order(self, primitives, frame_time):
        """Fold the batch's :func:`attach_depth_order` records into the job's."""
        held = []
        for primitive in primitives:
            held.extend(getattr(primitive, DEPTH_ORDER_HELD, None) or ())
            if hasattr(primitive, DEPTH_ORDER_HELD):
                setattr(primitive, DEPTH_ORDER_HELD, None)
        for records, owners, eye in held:
            rows = torch.stack([record.cpu() for record in records])
            pairs = int(rows[:, _D_COUNT].sum())
            if pairs == 0 or owners is None:
                continue
            self.depth_pairs += pairs
            k = int(torch.argmin(rows[:, _D_RATIO]))
            row = rows[k].tolist()
            if row[_D_RATIO] >= self._depth_ratio:
                continue
            self._depth_ratio = row[_D_RATIO]
            self.depth = DepthOrderFinding(
                first=_circuit_mob(owners, row[_D_CIRCUIT_A]),
                second=_circuit_mob(owners, row[_D_CIRCUIT_B]),
                gap=row[_D_GAP],
                resolution=row[_D_RESOLUTION],
                depth=row[_D_DEPTH],
                eye_distance=eye,
                time=float(frame_time(int(row[_D_FRAME]))),
            )

    def note_batch(self, primitives, frame_time):
        """Read one rendered batch's records and keep the worst.

        ``frame_time`` maps a frame offset within the batch to seconds. The
        records are read in one transfer per batch, after the batch has
        rendered, so the read never waits on anything still queued.
        """
        self._read_depth_order(primitives, frame_time)
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
        """Warn once for each finding that shows.

        Once if the job's worst rounding reached the visible level, and once
        if two overlapping flat shapes came closer in depth than the camera
        resolves.
        """
        finding = self.worst
        if finding is not None and finding.error_px >= VISIBLE_ERROR_PX:
            warnings.warn(
                precision_message(finding),
                Float32PrecisionWarning,
                stacklevel=_user_stacklevel(),
            )
        if self.depth is not None:
            warnings.warn(
                depth_order_message(self.depth, self.depth_pairs),
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
    """``1 unit``, ``0.25 units``, ``1000 units``, ``16000 units``."""
    # Whole units once ``g`` would switch to an exponent: a distance of
    # 16000 reads better than 1.6e+04.
    text = f"{value:.0f}" if abs(value) >= 10**digits else f"{value:.{digits}g}"
    return f"{text} unit" if text == "1" else f"{text} units"


# ---------------------------------------------------------------------------
# Depth order: which of two overlapping flat shapes is drawn in front
# ---------------------------------------------------------------------------

#: Two overlapping flat shapes keep their order only when their depths differ
#: by more than about this many float32 spacings of the larger of the eye's
#: distance from the origin and the depth. A circuit's depth along a pixel's
#: ray is ``(centre - eye) . n / (dir . n)``, every coordinate of the
#: difference rounded at the eye's magnitude, and differently for two layers
#: with different centres. Measured: two squares riding a camera whose eye is
#: 16,000 units out traded places across their whole overlap from frame to
#: frame at depth gaps of 0 and 0.001, and kept their order from 0.003; four
#: spacings is 0.0076 there; at gap 0 they traded places with the eye 1000
#: units out too. The same squares fixed in the world, the camera moving past
#: them, kept their order even when coplanar: their depths came out
#: bit-identical, a tie the depth bins hand to draw order alike every frame,
#: and the scan skips exact ties. Riding the camera, a third of the pairs came
#: out a spacing apart, either way round. A gap that rounds to a spacing or
#: two still counts, so for geometry that does not move with the camera the
#: warning errs on the safe side.
DEPTH_ORDER_SPACINGS = 4.0

#: Key in a merged scene: per bezier primitive, in merge order, ``(circuits,
#: _member_owners, _member_ends, _owner_mob)`` -- who drew each merged
#: circuit, resolved to ids only when a distant camera asks
#: (:func:`circuit_owner_ids`).
CIRCUIT_OWNER_PARTS = "_rt_circuit_owner_parts"
_CIRCUIT_OWNER_IDS = "_rt_circuit_owner_ids"
_CIRCUIT_SHOWS = "_rt_circuit_shows"
#: Key in a merged scene: the records (:func:`depth_order_record`) a
#: wavefront chunk leaves for ``render_batch_raytraced``, which keeps the ones
#: whose chunk rendered and attaches them to the batch
#: (:func:`attach_depth_order`).
DEPTH_ORDER_RECORDS = "_rt_depth_order_records"
#: Attribute of a batch's first primitive: ``(records, owner ids, eye)`` per
#: render call, until the render job reads them with the batch.
DEPTH_ORDER_HELD = "_rt_depth_order"
#: Fragments per pass of the check, which bounds its temporaries (about 40
#: bytes a fragment) however long the stream.
_DEPTH_CHUNK = 1 << 20
#: Field order of a depth-order record.
(
    _D_COUNT,
    _D_RATIO,
    _D_GAP,
    _D_RESOLUTION,
    _D_CIRCUIT_A,
    _D_CIRCUIT_B,
    _D_FRAME,
    _D_DEPTH,
) = range(8)


def depth_order_eye(eyes, tie_step):
    """The eye distance the depth-order check works at, or ``None`` to skip it.

    ``eyes`` are a batch's eye positions (``[T, 3]``, host). Two depths that
    rounding could swap differ by less than ``DEPTH_ORDER_SPACINGS`` spacings;
    while that is below the renderer's coplanar tie step (``tie_step``, the
    1e-4 depth bins; the eye within about 210 units of the origin) such a
    pair lands in one bin, where draw order decides, so an ordinary render
    never runs the check. (Two exactly coplanar shapes riding a camera can
    still land either side of a bin edge, frame by frame; that is the tie
    bins' own flaw, and not one a depth gap can be advised for.)
    """
    if eyes is None or eyes.numel() == 0:
        return None
    eye = float(eyes.reshape(-1, 3).double().norm(dim=-1).max())
    if not math.isfinite(eye) or DEPTH_ORDER_SPACINGS * EPS32 * eye <= tie_step:
        return None
    return eye


def circuit_owner_ids(merged, device):
    """``(ids, mobs, host_ids)``: the index into ``mobs`` of each merged circuit's Mob.

    ``ids`` is int32 on ``device`` (``host_ids`` its CPU copy), -1 where the
    owner is unknown. Built from :data:`CIRCUIT_OWNER_PARTS` once per merged
    scene, with the same rules as ``_owner_of``, and cached on it.
    """
    cached = merged.get(_CIRCUIT_OWNER_IDS)
    if cached is not None and cached[0].device == device:
        return cached
    mobs, slots, lanes = [], {}, []

    def slot(mob):
        if mob is None:
            return -1
        if id(mob) not in slots:
            slots[id(mob)] = len(mobs)
            mobs.append(mob)
        return slots[id(mob)]

    for count, owners, ends, lone in merged.get(CIRCUIT_OWNER_PARTS) or ():
        lane = torch.full((int(count),), -1, dtype=torch.int32)
        if owners and ends is not None and len(ends) == len(owners):
            start = 0
            for owner, end in zip(owners, ends):
                lane[start : min(int(end), lane.shape[0])] = slot(owner)
                start = int(end)
        elif owners and len(owners) == lane.shape[0]:
            lane = torch.tensor([slot(owner) for owner in owners], dtype=torch.int32)
        elif owners and len(owners) == 1:
            lane.fill_(slot(owners[0]))
        elif lone is not None:
            lane.fill_(slot(lone))
        lanes.append(lane)
    host_ids = torch.cat(lanes) if lanes else torch.zeros(0, dtype=torch.int32)
    cached = (host_ids.to(device), mobs, host_ids)
    merged[_CIRCUIT_OWNER_IDS] = cached
    return cached


def circuit_shows(merged, device):
    """Whether each merged circuit draws anything, per frame.

    Four ``[T, circuits]`` bool tables on ``device``, each with its own frame
    count ``T`` and read at ``frame % T`` as the kernels read it: whether the
    fill grid (``circuit_colors``) has a texel with alpha or glow, whether the
    circuit is filled at all, whether the border grid
    (``circuit_border_colors``) has one, and whether it has a border width. A
    fragment draws its circuit's fill, its border or a blend of the two; one
    that draws neither -- inside an outline with a transparent fill, or of a
    Mob waiting at opacity 0 to fade in -- cannot flicker through anything.
    Cached on the merged scene.
    """
    from algan.rendering.raytracing.raytrace_kernels_taichi import (
        _M_BORDER_W,
        _M_FILLED,
    )

    cached = merged.get(_CIRCUIT_SHOWS)
    if cached is not None and cached[0].device == device:
        return cached
    meta = merged["circuit_meta"]

    def texels(name):
        # Glow (channel 3) is additive and shows at zero alpha (channel 4).
        return merged[name][..., 3:5].amax(dim=2).amax(dim=-1).gt(0)

    tables = tuple(
        table.to(device)
        for table in (
            texels("circuit_colors"),
            meta[..., _M_FILLED].gt(0.5),
            texels("circuit_border_colors"),
            meta[..., _M_BORDER_W].gt(0.0),
        )
    )
    merged[_CIRCUIT_SHOWS] = tables
    return tables


def _drawn(shows, frame, circuit, border_share, full_share):
    """Whether circuit fragments draw anything, from :func:`circuit_shows`.

    ``border_share`` is each fragment's quantized share of its pixel in the
    border band (``raster_taichi._pack_bez_ref``), ``full_share`` all of it:
    the fragment draws its fill where the share is short of full, its border
    where it is above zero.
    """

    def at(table):
        frames, circuits = table.shape
        index = (frame % frames) * circuits + circuit.clamp_max(circuits - 1)
        return table.reshape(-1).index_select(0, index)

    fill_texels, filled, border_texels, bordered = shows
    fill = (border_share < full_share) & at(fill_texels) & at(filled)
    border = (border_share > 0) & at(border_texels) & at(bordered)
    return fill | border


def depth_order_record(
    key_s,
    ref_s,
    cov_s,
    owner_ids,
    shows,
    eye,
    frame_start,
    pixels_per_frame,
    border_bits,
    full_coverage,
):
    """Overlapping flat shapes whose order rounding decides, in one sorted stream.

    ``key_s`` / ``ref_s`` / ``cov_s`` are the raster pipeline's fragment
    stream, sorted by pixel and then depth: ``(pixel << 32) | depth bits``,
    the fragment's primitive reference (a circuit's is negative) and its
    coverage of the pixel. A pair is two fragments adjacent in one pixel's
    list that are both circuits covering the WHOLE pixel, both drawing
    something at that frame (``shows``, :func:`circuit_shows`), drawn by
    different Mobs (``owner_ids``, per merged circuit), less than
    ``DEPTH_ORDER_SPACINGS`` float32 spacings of the larger of ``eye`` and
    the depth apart -- but not at bit-identical depths. Whole pixels, because
    touching shapes -- a word's neighbouring glyphs -- share only antialiased
    edge pixels, where either order composites alike; drawing something,
    because a shape held at opacity 0 to fade in later hides nothing whichever
    side it is on; different Mobs, because the strokes inside one pack (a
    net's synapses) cross everywhere, coplanar, and show nothing for it; not
    identical depths, because those tie, and the tie goes to draw order the
    same way every frame (coplanar shapes fixed in the world).

    Returns a float64 record on the stream's device, without synchronizing:
    the pair count, then for the pair whose gap is the smallest share of its
    resolution, that share, the gap, the resolution, the two circuits, the
    frame (``frame_start`` + the pixel's frame in the window) and the depth.
    """
    n = int(key_s.shape[0])
    device = key_s.device
    count = torch.zeros((), dtype=torch.float64, device=device)
    best = torch.full((7,), math.inf, dtype=torch.float64, device=device)
    last_circuit = int(owner_ids.shape[0]) - 1
    if n < 2 or last_circuit < 0:
        return torch.cat((count.view(1), best))
    for lo in range(0, n - 1, _DEPTH_CHUNK):
        hi = min(n, lo + _DEPTH_CHUNK + 1)
        key = key_s[lo:hi]
        ref = ref_s[lo:hi]
        pixel = key >> 32
        pair = pixel[1:] == pixel[:-1]
        flat = (ref < 0) & (cov_s[lo:hi] >= full_coverage)
        pair &= flat[1:] & flat[:-1]
        code = (-ref.to(torch.int64) - 1).clamp_min(0)
        circuit = (code >> border_bits).clamp_max(last_circuit)
        owner = owner_ids.index_select(0, circuit)
        pair &= (owner[1:] != owner[:-1]) & (owner[1:] >= 0) & (owner[:-1] >= 0)
        frame = frame_start + torch.div(pixel, pixels_per_frame, rounding_mode="floor")
        full_share = (1 << border_bits) - 1
        drawn = _drawn(shows, frame, circuit, code & full_share, full_share)
        pair &= drawn[1:] & drawn[:-1]
        depth = (key & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
        gap = (depth[1:] - depth[:-1]).abs()
        resolution = (DEPTH_ORDER_SPACINGS * EPS32) * torch.clamp_min(
            torch.maximum(depth[1:].abs(), depth[:-1].abs()), eye
        )
        pair &= (gap > 0) & (gap < resolution)
        count = count + pair.sum()
        ratio = torch.where(pair, gap / resolution, torch.full_like(gap, math.inf))
        k = ratio.argmin()
        row = torch.stack(
            (
                ratio[k].double(),
                gap[k].double(),
                resolution[k].double(),
                circuit[k].double(),
                circuit[k + 1].double(),
                frame[k + 1].double(),
                depth[k + 1].double(),
            )
        )
        best = torch.where(row[0] < best[0], row, best)
    return torch.cat((count.view(1), best))


def attach_depth_order(primitive, records, merged, eye):
    """Leave a rendered batch's depth-order records on its first primitive.

    Beside the projection's ``_rt_float32_record``, for the same reasons:
    :meth:`Float32RoundingJob.note_batch` reads them with the batch, after it
    has rendered, and a batch that fails is dropped with its primitives, so
    frames rendered again are never counted twice. ``records`` are
    :func:`depth_order_record` results (device tensors, unread), ``merged``
    the scene whose circuits they index, ``eye`` the eye distance they were
    measured at.
    """
    held = getattr(primitive, DEPTH_ORDER_HELD, None) or []
    held.append((list(records), merged.get(_CIRCUIT_OWNER_IDS), eye))
    setattr(primitive, DEPTH_ORDER_HELD, held)


def _circuit_mob(owners, circuit):
    """The Mob that drew merged ``circuit``, from :func:`circuit_owner_ids`'s result."""
    _ids, mobs, host_ids = owners
    circuit = int(circuit)
    if not 0 <= circuit < host_ids.shape[0]:
        return None
    slot = int(host_ids[circuit])
    return mobs[slot] if 0 <= slot < len(mobs) else None


class DepthOrderFinding(NamedTuple):
    """The render's closest call between two overlapping flat shapes."""

    first: object
    second: object
    gap: float
    resolution: float
    depth: float
    eye_distance: float
    time: float


def depth_order_message(finding: DepthOrderFinding, pairs: int) -> str:
    """The depth-order warning's text: what, where, why, and what to do."""
    return (
        f"float32 rounding can swap which of two overlapping flat shapes is in "
        f"front: {describe_owner(finding.first)} and "
        f"{describe_owner(finding.second)} cover the same pixels "
        f"{_units(finding.gap, 2)} apart in depth, "
        f"{_units(finding.depth, 4)} from a camera "
        f"{_units(finding.eye_distance, 4)} from the origin, where depths are "
        f"only told apart {_units(finding.resolution, 2)} or more apart (worst "
        f"at t={finding.time:.2f}s; {pairs} overlapping pixel(s) in this "
        f"render). Shapes that close -- including shapes at one depth, which "
        f"draw order decides near the origin -- can trade places from frame to "
        f"frame, one flickering through the other. Separate them along the "
        f"view by at least {_units(2 * finding.resolution, 2)} (a "
        f"near-orthographic camera shows no parallax for it), or bring the "
        f"camera closer to the origin: the resolution shrinks with its "
        f"distance."
    )
