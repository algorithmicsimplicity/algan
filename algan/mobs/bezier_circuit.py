"""Cubic bezier circuits -- the geometry behind every 2-D shape.

:class:`BezierCircuitCubic` is a closed loop of cubic bezier curves, stored as
control points. It is what :class:`~algan.mobs.shapes_2d.Circle`,
:class:`~algan.mobs.shapes_2d.Square`, :class:`~algan.mobs.text.Text` and
:class:`~algan.mobs.text.Tex` are made of, which is why a circle is a true circle
at any zoom rather than a many-sided polygon, and why any of them can morph into
any other.

The class owns the fill/border model as well as the geometry: on a filled shape
the border is drawn *inside* the outline, so raising ``stroke_width`` eats into
the fill instead of growing the silhouette -- which keeps bordered text legible
and stops neighbouring glyphs fusing. An unfilled circuit has no interior to eat
into, so its stroke stays centred on the path.

It owns color *across* the shape too, through a ``grid_width`` x
``grid_height`` grid of colored points laid over the circuit's own frame, which
the renderer samples bilinearly per fragment. This is the same thing
:class:`~algan.mobs.surfaces.surface.Surface` calls its grid, over the same
``(u, v)`` domain, and it is named to match: a circuit is always planar, so
there is no reason to give it separate vertex colors and a texture map the way
a curved surface needs -- one grid is both.
:meth:`BezierCircuitCubic.set_color_by_function` fills it in from ``(u, v)``,
:meth:`BezierCircuitCubic.set_color_by_image` from a picture, and
:meth:`~algan.mobs.shapes_2d.Line.set_color_by_function` from a single ``t``
along the path. The grid is one point, i.e. one flat color, unless it was asked
for.

``build_render_primitives_batched`` packs many circuits into one
:class:`~algan.rendering.primitives.bezier_circuit_primitive.BezierCircuitPrimitive`
for the renderer, which evaluates the curves analytically rather than
tessellating them.
"""

from __future__ import annotations

import math
import threading
import typing as _typing
from functools import lru_cache

import numpy as np
import torch.nn.functional as F

from algan.animatable_base.animatable import animated_function
from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import Off, Sync
from algan.constants.color import *
from algan.constants.spatial import OUTWARD, RIGHT, UP
from algan.errors import AlganConfigurationError
from algan.mobs.nonplanar_circuit import (
    build_render_primitives as build_nonplanar_render_primitives,
)
from algan.mobs.nonplanar_circuit import classify_circuit
from algan.mobs.stroke_style import _stroke_style
from algan.rendering.mps_compat import cummax_values
from algan.rendering.raytracing.utils import _unify_time
from algan.settings.renderer_settings import RENDERER_REGISTRY
from algan.settings.video_settings import PREVIEW
from algan.utils.mob_utils import pack_animatable_rows, pack_member_rows
from algan.utils.tensor_utils import *

# Three.js's fixed dielectric F0 = 0.04 corresponds to IOR 1.5; MeshStandard
# has no ``ior`` of its own, so that is the default a circuit falls back to.
DIELECTRIC_IOR = 1.5

# Ceiling on the texture-grid size ``wave_color`` will refine a circuit to. A
# wave tighter than this can afford is drawn as smoothly as the budget allows.
_MAX_WAVE_TEXTURE_RESOLUTION = 64


def _stroke_width_in_render_pixels(stroke_width, video_settings):
    """Convert an authored ``stroke_width`` to the renderer's stroke width.

    ``stroke_width`` is authored against PREVIEW's frame height so a border keeps
    its apparent weight at any resolution.  The renderer wants the FULL stroke
    width in pixels; where it lays that width is
    ``SETTINGS.style.border_placement``'s business, not this function's (a
    filled circuit lays it inside the outline by default, an unfilled one
    centres it on the path -- ``_circuit_point_region``).
    """
    return stroke_width * video_settings.resolution[1] / PREVIEW.resolution[1]


def _circuit_ior(ior, metalness):
    """Pack a material's IOR into a circuit's transport channel.

    Mirrors the triangle primitive's ``_derive_material_surface_params``: an
    unsigned magnitude feeding dielectric F0. Whether the circuit transmits is
    carried by the separate ``transmission`` channel, not by this one's sign.
    Non-PBR circuits (metalness < 0) get 0: inert, since their reflectance is 0
    anyway.
    """
    return torch.where(metalness >= 0.0, ior.abs(), torch.zeros_like(ior))


def _resample_texture_grid(value, old_size, new_size):
    """Resample per-texel circuit values between two ``(width, height)`` texture
    grids, for every circuit packed into ``value`` ``[..., N * old, C]``.

    Texels are stored with the width (first basis) axis outermost, so the packed
    rows reshape straight into a ``[width, height]`` image.
    """
    leading = value.shape[:-2]
    channels = value.shape[-1]
    image = value.reshape(-1, *old_size, channels).permute(0, 3, 1, 2)
    resized = F.interpolate(
        image, size=tuple(new_size), mode="bilinear", align_corners=True
    )
    return resized.permute(0, 2, 3, 1).reshape(*leading, -1, channels)


#: Relative band within which control points count as equally far out, from
#: a circuit's centre or from its first axis. Symmetric shapes tie exactly in
#: exact arithmetic -- every corner of a square, every point of a circle, the
#: two ends of a line -- and in floating point come out a few ulps apart, in
#: whichever direction the centroid's rounding happened to fall. A band a
#: hundred times wider than float32 rounding keeps those ties ties, and the
#: lowest index wins them, so rounding never picks the winner.
_TIE_TOLERANCE = 1e-5

#: How far a unit plane normal must lean along an axis before that axis decides
#: which way the plane faces (see :func:`_face_outward`).
_FACING_TOLERANCE = 1e-4


def _dot3(a, b):
    """Dot product over a last axis of size 3, as three products and two sums.

    The frame computation below runs in NumPy and is written out elementwise,
    never through a library reduction (``norm``, ``cross``, ``sum``): every
    operation is then one correctly rounded IEEE operation per element, so a
    circuit's result does not depend on what else shares its array. That is what
    lets :func:`_circuit_frames` frame a whole ``Text`` at once and still give
    each glyph the bits it gets framed alone -- the pack must land on exactly
    what its members would have. (NumPy rather than torch because a circuit is
    a few dozen points: torch's per-operation overhead was the whole cost.)
    """
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _norm3(v):
    """Euclidean length over a last axis of size 3 (see :func:`_dot3`)."""
    return np.sqrt(_dot3(v, v))


def _cross3(a, b):
    """Cross product over a last axis of size 3 (see :func:`_dot3`)."""
    return np.stack(
        (
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ),
        -1,
    )


def _halving_sum(values):
    """Sum over axis 1 by pairwise halving, zero-padded to a power of two.

    Each circuit's terms sit at the start of its row with zeros after them, so
    halving a row longer than its own power of two first adds only zeros to
    them, exactly: the total is the same to the last bit whether a circuit is
    summed alone or padded out beside a longer one. A library ``sum`` promises
    no such thing; its reduction order moves with the array's shape.
    """
    size = values.shape[1]
    padded = 1 if size <= 1 else 1 << (size - 1).bit_length()
    if padded != size:
        pad = np.zeros(
            (values.shape[0], padded - size, *values.shape[2:]), values.dtype
        )
        values = np.concatenate((values, pad), 1)
    while padded > 1:
        padded //= 2
        values = values[:, :padded] + values[:, padded:]
    return values[:, 0]


def _lowest_within_band(scores, present):
    """Per row, the lowest index whose score is within the tie band of the top.

    ``argmax`` promises nothing about equal maxima, and a circuit's control
    points tie constantly (see ``_TIE_TOLERANCE``); the winner sets the
    circuit's first axis, so an unspecified winner is an unspecified frame.
    Returns ``(index, top)`` per row.
    """
    top = np.where(present, scores, -1.0).max(1)
    count = scores.shape[1]
    tied = present & (scores >= (top * (1.0 - _TIE_TOLERANCE))[:, None])
    return np.where(tied, np.arange(count), count).min(1), top


def _face_outward(normal):
    """Sign unit plane normals so the plane faces OUTWARD.

    A plane has two faces, and which one a circuit presents used to follow
    from which two control points won its tie-breaks -- about half of all
    glyphs faced away from the camera, and a translation of a few ulps could
    turn one over. The sign is now the plane's own: the normal points along
    +z unless the plane is edge-on to it, then along +y, then along +x. Each
    axis decides only when the normal leans along it by more than
    ``_FACING_TOLERANCE``, far beyond rounding, so noise cannot flip a face.
    """
    x, y, z = normal[..., 0], normal[..., 1], normal[..., 2]
    lean = np.where(
        np.abs(z) > _FACING_TOLERANCE,
        z,
        np.where(np.abs(y) > _FACING_TOLERANCE, y, x),
    )
    return np.where((lean < 0)[..., None], -normal, normal)


def _texture_grid_offsets(samples, *, device=None, dtype=None):
    """Sample offsets along one axis of a circuit's frame, spanning -1 to 1.

    A single sample stands for the whole span, so it sits at the **centre** of
    it -- which is where :meth:`~.BezierCircuitCubic.get_base_grid` reports it
    too, at ``0.5``. ``torch.linspace(-1, 1, 1)`` is -1, the low END of the span,
    which put a lone texel in a corner of the frame and left its world position
    depending on the *sign* of the basis rows it is laid out along: re-signing
    row 1 (which :func:`_circuit_location_and_basis` does, so that a flat shape
    faces the viewer) then moved every glyph's texel clear across its own frame.
    The colour is unaffected either way -- one texel is one flat colour, and the
    renderer clamps the axis to it -- but ``wave_color`` reads each part's
    position from here, so at a corner a text fade's per-glyph lag turned on a
    convention, and a shape was ordered by its corner rather than by where it is.

    All three places that lay out a circuit's texels share this: construction,
    :meth:`~.BezierCircuitCubic.from_batches` (whose pack must land on exactly
    the same points -- ``test_batched_bezier_mobs.py`` is what says so) and the
    refinement a colour wave runs.
    """
    if samples < 2:
        return torch.zeros(1, device=device, dtype=dtype)
    return torch.linspace(-1, 1, samples, device=device, dtype=dtype) * (1 + 1e-5)


#: Samples per cubic segment used to measure a circuit's centroid. A symmetric
#: shape comes out exactly centred at any count, so this only bounds how far a
#: *curved, lopsided* one can be off: measured against the closed form for a
#: unit half disc (centroid at 4 / 3pi), 8 samples land within 1.3e-3 of it, 16
#: within 2.8e-4 and 32 within 2.5e-5, which is where the cubic approximation of
#: the arc itself takes over. 32 is the last count that buys anything, and the
#: work is one small matrix multiply either way.
_CENTROID_SAMPLES_PER_SEGMENT = 32

#: Below this, times the circuit's diagonal and the size of its coordinates
#: (the larger of its largest coordinate and its diagonal), an enclosed area is
#: not an area. A straight Line's control points are collinear only up to
#: float32 rounding -- a third of the way along, they are off the line by an
#: ulp of their coordinates -- so the shoelace sum over them is a sliver of
#: rounding noise, which scales with exactly that product. The bound used to be
#: 1e-9 of the diagonal squared, below that noise: a Line from (2, 1) to
#: (-1, 3) enclosed "area", and the centroid divided noise by it, anchoring
#: the line a third of the way off its middle and pinning row 0 to its end
#: instead of its start. Now anything thinner than ~1e-5 of its own size
#: measures as the path it is.
_DEGENERATE_AREA_FRACTION = 1e-5


@lru_cache(maxsize=4)
def _bezier_sample_weights(samples):
    """Bernstein weights that sample one cubic segment, shape ``(samples, 4)``.

    float64, read-only and cached: every circuit wants the identical matrix.
    """
    t = np.linspace(0.0, 1.0, samples + 1)[:-1].reshape(-1, 1)
    s = 1.0 - t
    weights = np.concatenate((s * s * s, 3 * s * s * t, 3 * s * t * t, t * t * t), -1)
    weights.setflags(write=False)
    return weights


def _circuit_centroids(points, counts, origins, extents):
    """Where each circuit balances: the centroid of the region it encloses.

    This is the point a shape turns about, so it is its own centre of area and
    not the middle of the box around it. The two differ for anything not
    point-symmetric -- a ``Triangle``'s box centre sits a quarter of a unit
    above its centroid, which is enough to make a spin look like it is also
    drifting upward, since the shape then orbits a point above itself.

    Measured by shoelace sums over a polyline sampled from the curves
    (``_CENTROID_SAMPLES_PER_SEGMENT`` a segment), taken about ``origins``
    (which lie in each circuit's plane, so the triangle fans those sums
    describe are planar and their signed areas exact), in float64 so a
    symmetric shape comes back at its exact centre. A segment whose P0 is not
    where the previous segment's P3 left off starts a new loop -- which is how
    a circuit carries holes -- and each loop closes on itself, so a hole
    subtracts its own area. A path that encloses no area -- a straight
    :class:`~.Line`, or any open stroke -- falls back to the centroid of the
    path itself, weighted by arc length; the chord that would close it for a
    fill carries no weight there. A circuit of fewer than one segment, or of no
    length, falls back to its origin.

    Every circuit's sums run over its own polyline only (see
    :func:`_halving_sum`), so the batch does not change any circuit's result.

    Parameters
    ----------
    points
        NumPy control points, shape ``(B, N, 3)``: each circuit's own first,
        zero padding after.
    counts
        Each circuit's number of control points, shape ``(B,)``.
    origins
        Where the moments are taken, shape ``(B, 3)``: each circuit's
        bounding-box centre.
    extents
        Each circuit's bounding-box diagonal, shape ``(B,)``.

    Returns
    -------
    tuple
        ``(centroids, area_normals, encloses_area)``: shapes ``(B, 3)``,
        ``(B, 3)`` and ``(B,)``, float64. An area normal is the unit normal of
        the plane a circuit's loops span (Newell's), meaningful only where
        ``encloses_area``.
    """
    batch, width, _ = points.shape
    samples = _CENTROID_SAMPLES_PER_SEGMENT
    segments = counts // 4
    most = width // 4
    origin = origins.astype(np.float64)
    extent = extents.astype(np.float64)
    control = (points[:, : most * 4].astype(np.float64) - origin[:, None]).reshape(
        batch, most, 4, 3
    )
    segment_index = np.arange(most)
    valid = segment_index < segments[:, None]

    weights = _bezier_sample_weights(samples)
    rows = control[:, :, None]

    def term(k):
        return weights[:, k].reshape(1, 1, samples, 1) * rows[..., k, :]

    curve = ((term(0) + term(1)) + term(2)) + term(3)  # (B, M, S, 3)

    # Loops: a segment starts one when it is its circuit's first, or when it
    # does not continue the one before it.
    tolerance = 1e-6 * np.maximum(extent, 1.0)
    starts = np.ones((batch, most), dtype=bool)
    if most > 1:
        gaps = _norm3(control[:, 1:, 0] - control[:, :-1, 3])
        starts[:, 1:] = gaps > tolerance[:, None]
    starts &= valid
    ends = np.zeros_like(starts)
    ends[:, :-1] = starts[:, 1:]
    ends |= segment_index == (segments - 1)[:, None]
    ends &= valid
    loop = np.cumsum(starts, 1) - 1
    first_segment = np.maximum.accumulate(np.where(starts, segment_index, -1), 1).clip(
        min=0
    )

    # Steps within a segment go sample to sample; a segment's last sample steps
    # to the next segment's first while the loop goes on, and to the
    # segment's own end point where the loop closes.
    following = np.concatenate((curve[:, 1:, 0], np.zeros_like(curve[:, :1, 0])), 1)
    last_step = np.where(ends[..., None], control[:, :, 3], following)
    step_end = np.concatenate((curve[:, :, 1:], last_step[:, :, None]), 2)
    keep = valid[:, :, None, None]
    step_start = np.where(keep, curve, 0.0).reshape(batch, most * samples, 3)
    step_end = np.where(keep, step_end, 0.0).reshape(batch, most * samples, 3)

    # Then each loop's closing step, from its end point back to where it began,
    # right after its circuit's own samples.
    length = most * samples + most
    start = np.zeros((batch, length, 3))
    end = np.zeros((batch, length, 3))
    start[:, : most * samples] = step_start
    end[:, : most * samples] = step_end
    wraps = np.zeros((batch, length), dtype=bool)
    closing_row, closing_segment = np.nonzero(ends)
    slot = segments[closing_row] * samples + loop[closing_row, closing_segment]
    start[closing_row, slot] = control[closing_row, closing_segment, 3]
    end[closing_row, slot] = curve[
        closing_row, first_segment[closing_row, closing_segment], 0
    ]
    wraps[closing_row, slot] = True

    cross = _cross3(start, end)
    vector_area = _halving_sum(cross) * 0.5
    area_size = _norm3(vector_area)
    present = np.arange(width) < counts[:, None]
    magnitude = np.where(present, np.abs(points).max(-1), 0).max(1).astype(np.float64)
    encloses_area = (segments > 0) & (
        area_size > _DEGENERATE_AREA_FRACTION * extent * np.maximum(magnitude, extent)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        area_normal = vector_area / np.maximum(area_size, 1e-300)[:, None]
        twice_areas = _dot3(cross, area_normal[:, None])
        area_centroid = (
            _halving_sum((start + end) * twice_areas[..., None])
            / (3 * _halving_sum(twice_areas))[:, None]
        )
        lengths = _norm3(end - start) * (~wraps)
        total = _halving_sum(lengths)
        path_centroid = (
            _halving_sum(((start + end) * 0.5) * lengths[..., None])
            / np.maximum(total, 1e-300)[:, None]
        )
    offset = np.where(
        encloses_area[:, None],
        area_centroid,
        np.where(((segments > 0) & (total > 0))[:, None], path_centroid, 0.0),
    )
    return origin + offset, area_normal, encloses_area


def _grouped_centroids(points, counts, origins, extents):
    """:func:`_circuit_centroids`, run on circuits of similar length together.

    The centroid pass pads every circuit to the longest one in its call, so a
    ``Text`` whose longest glyph has four times the segments of its shortest
    would do four times the work on the short ones. Circuits are grouped by the
    power of two of their segment count, which bounds the padding at half of
    each group. Grouping changes no result: every circuit's sums run over its
    own rows only.
    """
    segments = np.maximum(counts // 4, 1)
    buckets = np.ceil(np.log2(segments)).astype(np.int64)
    unique = np.unique(buckets)
    if unique.size == 1:
        return _circuit_centroids(points, counts, origins, extents)
    batch = points.shape[0]
    centroids = np.zeros((batch, 3))
    normals = np.zeros((batch, 3))
    encloses = np.zeros(batch, dtype=bool)
    for bucket in unique:
        rows = np.nonzero(buckets == bucket)[0]
        width = int(counts[rows].max())
        (centroids[rows], normals[rows], encloses[rows]) = _circuit_centroids(
            points[rows, :width], counts[rows], origins[rows], extents[rows]
        )
    return centroids, normals, encloses


def _circuit_centroid(control_points, bbox_centre):
    """The centroid of one circuit, about ``bbox_centre``, as a torch tensor.

    See :func:`_circuit_centroids`, which this runs for a batch of one.
    """
    points = control_points.reshape(-1, 3).detach().cpu().numpy()
    origin = bbox_centre.reshape(1, 3).detach().cpu().numpy()
    extent = _norm3(points.max(0) - points.min(0)).reshape(1)
    centroid, _, _ = _circuit_centroids(
        points[None], np.array([points.shape[0]]), origin, extent
    )
    return torch.from_numpy(centroid[0].astype(points.dtype)).to(control_points.device)


#: Largest circuit, in control points, whose frame is memoized, and how many
#: frames are kept. A glyph is a few dozen points; the bound keeps a packed
#: whole-scene batch (hashed as one key) out of the memo.
_FRAME_MEMO_MAX_POINTS = 4096
_FRAME_MEMO_SIZE = 32768
_FRAME_MEMO: dict = {}
_FRAME_MEMO_LOCK = threading.Lock()


def _frame_memo_key(points):
    """The memo key of one circuit's control points, or None to skip the memo."""
    if points.device.type != "cpu" or points.shape[0] > _FRAME_MEMO_MAX_POINTS:
        return None
    try:
        raw = points.detach().contiguous().numpy().tobytes()
    except (TypeError, RuntimeError):  # a dtype numpy has no equivalent for
        return None
    return (points.dtype, points.shape[0], raw)


def _remember_frame(key, frame):
    stored = tuple(
        value.clone() if torch.is_tensor(value) else value for value in frame
    )
    # Circuits can be built on the batch-prep worker too (an updater building
    # a Mob during replay), so eviction must not race an insert.
    with _FRAME_MEMO_LOCK:
        if len(_FRAME_MEMO) >= _FRAME_MEMO_SIZE:
            # Drop the oldest half: insertion order is age order in a dict.
            for stale in list(_FRAME_MEMO)[: _FRAME_MEMO_SIZE // 2]:
                del _FRAME_MEMO[stale]
        _FRAME_MEMO[key] = stored


def _circuit_location_and_basis(control_points):
    """Memoized frame of one circuit: see :func:`_circuit_frames`.

    Returns ``(location, basis, second_axis_synthesized)``: shapes ``(3,)`` and
    ``(9,)``, and a bool. A bit-identical input returns a copy of the frame
    computed for it before; framed alone or in a batch, a circuit gets the
    same bits either way.
    """
    points = control_points.reshape(-1, 3)
    key = _frame_memo_key(points)
    cached = _FRAME_MEMO.get(key) if key is not None else None
    if cached is None:
        locations, bases, synthesized = _circuit_frames([points])
        cached = (locations[0], bases[0], bool(synthesized[0]))
        if key is not None:
            _remember_frame(key, cached)
        return cached
    return tuple(value.clone() if torch.is_tensor(value) else value for value in cached)


def _circuit_locations_and_bases(control_point_batches):
    """Memoized frames of many circuits, computed together where not memoized.

    Returns ``(locations, bases)``, shapes ``(count, 3)`` and ``(count, 9)``.
    """
    batches = [points.reshape(-1, 3) for points in control_point_batches]
    keys = [_frame_memo_key(points) for points in batches]
    frames = [(_FRAME_MEMO.get(key) if key is not None else None) for key in keys]
    missing = [index for index, frame in enumerate(frames) if frame is None]
    if missing:
        locations, bases, synthesized = _circuit_frames(
            [batches[index] for index in missing]
        )
        for row, index in enumerate(missing):
            frame = (locations[row], bases[row], bool(synthesized[row]))
            frames[index] = frame
            if keys[index] is not None:
                _remember_frame(keys[index], frame)
    return (
        torch.stack([frame[0] for frame in frames]),
        torch.stack([frame[1] for frame in frames]),
    )


def _circuit_frames(control_point_batches):
    """Return the local frame of each circuit, plus whether its second in-plane
    axis had to be synthesized.

    Row 2 is the circuit's plane normal. Rows 0 and 1 span that plane and always
    share a length, so the frame is orthogonal with a square in-plane footprint
    -- which is what the texture grid, laid out along those two rows, relies on.

    The plane is the one the circuit's loops span (their Newell normal) when
    it encloses area, and otherwise the one through its centre, its furthest
    control point and the control point furthest from that axis. Which of the
    plane's two faces the circuit presents is the plane's own business, not its
    control points': :func:`_face_outward` turns it towards OUTWARD, so every
    flat shape, whatever order its outline was authored in, faces the camera it
    was drawn in front of, as ``DEFAULT_BASIS`` says a Mob does.

    Within the plane the rows are aligned to the world axes, not to the
    circuit's own geometry. They used to point at the control point furthest
    from the centre, i.e. at a *corner*: a ``Rectangle(4, 1)`` came out with row
    0 = ``(-2, 0.5, 0)``. Since a shape-``(*, 3)`` factor to
    :meth:`~.Mob.scale` scales the Mob's own right, up and forward axes, that
    made ``rect.scale([4, 1, 1])`` stretch along the diagonal and render the
    rectangle as a parallelogram. Both rows are as long as the distance from the
    centre to the furthest control point, so ``scale_coefficient`` and anything
    derived from it (``Circle.radius``) follow the shape's extent.

    A straight Line is the exception and keeps the geometry-derived frame: its
    control points are collinear, so there is no plane to align to, and the
    extremal displacement genuinely is the shape's own axis. Row 0 is pinned to
    the START of such a path -- see :class:`~algan.mobs.shapes_2d.Line`, whose
    documented guarantee this is -- and row 1 is a clockwise quarter turn of it
    about OUTWARD.

    Every choice among equally distant control points is made within
    ``_TIE_TOLERANCE`` and every facing decision within
    ``_FACING_TOLERANCE``, so rounding cannot turn a frame round, and every
    reduction runs over a circuit's own rows (see :func:`_dot3`), so framing a
    whole ``Text`` at once gives each glyph exactly the frame it gets alone.

    Parameters
    ----------
    control_point_batches
        One ``(N_i, 3)`` tensor of cubic control points per circuit, all of one
        dtype and device.

    Returns
    -------
    tuple
        ``(locations, bases, synthesized)``: each circuit's centre, shape
        ``(B, 3)``; its flattened 3x3 frame, shape ``(B, 9)``; and whether its
        control points were collinear (or coincident), so that row 1 carries
        no extent of the shape's own, shape ``(B,)``.
    """
    batches = [points.reshape(-1, 3) for points in control_point_batches]
    device = batches[0].device
    arrays = [points.detach().cpu().numpy() for points in batches]
    dtype = arrays[0].dtype
    counts = np.array([array.shape[0] for array in arrays], dtype=np.int64)
    batch = len(arrays)
    width = int(counts.max())
    points = np.zeros((batch, width, 3), dtype=dtype)
    for row, array in enumerate(arrays):
        points[row, : array.shape[0]] = array
    present = np.arange(width) < counts[:, None]
    lowest = np.where(present[..., None], points, np.inf).min(1)
    highest = np.where(present[..., None], points, -np.inf).max(1)
    bbox_centre = (lowest + highest) * 0.5
    diagonal = highest - lowest
    # The frame is anchored at the shape's centroid, not at the middle of its
    # box: ``location`` is what a Mob turns and scales about, and a shape has
    # to turn about itself. See :func:`_circuit_centroids`.
    centroids, area_normals, encloses_area = _grouped_centroids(
        points, counts, bbox_centre, _norm3(diagonal.astype(np.float64))
    )
    location = centroids.astype(dtype)
    coincident = _norm3(diagonal) <= 1e-6
    rows = np.arange(batch)
    right, up, outward = (
        axis.reshape(3).cpu().numpy().astype(dtype) for axis in (RIGHT, UP, OUTWARD)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        disps = points - location[:, None]
        first_index, farthest = _lowest_within_band(_norm3(disps), present)
        first = disps[rows, first_index]
        short = farthest <= 1e-4
        first_axis = np.where(short[:, None], right * 1e-4, first)
        first_unit = first_axis / _norm3(first_axis)[:, None]
        planar_disps = (
            disps - _dot3(disps, first_unit[:, None])[..., None] * first_unit[:, None]
        )
        second_index, widest = _lowest_within_band(_norm3(planar_disps), present)
        collinear = widest <= 1e-4

        # A plane: the loops' own, else the extremal pair's.
        pair = _cross3(first_unit, planar_disps[rows, second_index])
        pair_normal = pair / np.maximum(_norm3(pair), 1e-30)[:, None]
        normal = _face_outward(
            np.where(encloses_area[:, None], area_normals.astype(dtype), pair_normal)
        )
        scale = np.where(short, 1e-4, farthest).astype(dtype)[:, None]
        # Row 0 takes whichever world axis the plane admits, preferring x; row
        # 1 follows from the plane's orientation, so on a plane whose normal
        # faces the camera it comes out along +y -- an upright shape's own up
        # is UP, which is what ``wave_color`` means by "bottom to top". Only one
        # of the three world axes can be parallel to the normal, so one always
        # qualifies.
        along = None
        for axis in (outward, up, right):
            candidate = axis[None] - _dot3(axis[None], normal)[:, None] * normal
            size = _norm3(candidate)[:, None]
            unit = candidate / np.maximum(size, 1e-30)
            along = unit if along is None else np.where(size > 1e-4, unit, along)
        row_0 = along * scale
        row_1 = _cross3(normal, along) * scale

        # Collinear: row 0 from the centre to the start of the path, row 1 a
        # clockwise quarter turn of it about OUTWARD, carrying no extent of the
        # shape's own. The turn is written out, (x, y, z) -> (y, -x, z), so it
        # is exact. Row 2 is the negated cross of the two, which is OUTWARD for
        # a path in the screen plane; a path along z gives no cross at all and
        # takes RIGHT as its second axis instead.
        line_0 = first
        line_size = _norm3(line_0)[:, None]
        line_unit = line_0 / np.maximum(line_size, 1e-30)
        line_1 = np.stack((line_0[:, 1], -line_0[:, 0], line_0[:, 2]), -1)
        line_1 = line_1 * (line_size / np.maximum(_norm3(line_1), 1e-30)[:, None])
        along_z = _norm3(_cross3(line_unit, line_1)) <= 1e-6 * line_size[:, 0]
        line_1 = np.where(along_z[:, None], right * line_size, line_1)
        line_cross = _cross3(line_unit, line_1)
        line_2 = -line_cross / np.maximum(_norm3(line_cross), 1e-30)[:, None]
        turned = _face_outward(line_2)
        line_1 = np.where((_dot3(turned, line_2) < 0)[:, None], -line_1, line_1)

    basis = np.where(
        collinear[:, None],
        np.concatenate((line_0, line_1, turned), -1),
        np.concatenate((row_0, row_1, normal), -1),
    )
    basis = np.where(coincident[:, None], np.eye(3, dtype=dtype).reshape(1, 9), basis)
    return (
        torch.from_numpy(np.ascontiguousarray(location)).to(device),
        torch.from_numpy(np.ascontiguousarray(basis)).to(device),
        torch.from_numpy(collinear | coincident).to(device),
    )


class BezierCircuitCubic(Mob):
    """A closed loop of cubic bezier curves -- the geometry every 2-D shape is
    made of.

    The curves are evaluated analytically by the renderer rather than
    tessellated, so a circuit stays exactly smooth at any zoom, and any circuit
    can :meth:`~algan.animatable_base.mob.Mob.become` any other. On a filled
    circuit the border is drawn *inside* the outline, so raising ``stroke_width``
    eats into the fill instead of growing the silhouette; an unfilled circuit has
    no interior to eat into, so its stroke stays centred on the path.

    **Color across a circuit.** A circuit carries a rectangular grid of colored
    points, laid across its own frame and sampled bilinearly per fragment by the
    renderer. ``grid_width`` x ``grid_height`` is therefore the resolution of
    everything painted on the shape -- a gradient, an image, a color wave -- and
    it defaults to a single point, i.e. one flat color. Raise it and fill it in
    with :meth:`~.BezierCircuitCubic.set_color_by_function` or
    :meth:`~.BezierCircuitCubic.set_color_by_image`.

    This grid is the counterpart of
    :attr:`Surface.grid <algan.mobs.surfaces.surface.Surface.grid>`, and shares
    its name deliberately. A ``Surface`` keeps its vertex colors and its texture
    maps apart because a curved surface wants geometry and image detail at
    different resolutions; a circuit is always flat, so the distinction buys it
    nothing and one grid serves as both.

    The grid's ``(u, v)`` domain is the circuit's own frame, exactly as
    :class:`~algan.mobs.surfaces.surface.Surface`'s is: ``u`` runs from 0 to 1
    along the first basis row and ``v`` along the second, which for an upright
    2-D shape means ``u`` left to right and ``v`` bottom to top. Both rows are as
    long as the distance from the centre to the furthest control point, so the
    frame spans the square that circumscribes the shape and the shape itself
    covers the middle of the domain rather than all of it.

    Animation
    ---------
    Construction is immediate. Spawn the circuit before animating its geometry,
    width or colors (1 second by default, adjustable with ``Seq(runtime=...)``).
    ``cap_style``, ``joint_type`` and ``miter_limit`` are immediate configuration,
    applying to this circuit at every timestamp, before or after spawning.
    These controls support planar paths, including paths rotated in 3-D;
    nonplanar paths and shaded surface boundaries require the default styles.

    Parameters
    ----------
    control_points
        The cubic bezier control points, shape ``(*, 3)`` in world units, in
        groups of four: ``P0, P1, P2, P3`` per segment, with each segment
        starting where the previous one ended. A segment that starts somewhere
        else begins a new sub-circuit, which is how a shape gets holes.
    normals
        Per-control-point normals, shape ``(*, 3)``, used for lighting. Defaults
        to ``None``, meaning the circuit's own plane normal is used.
    stroke_width
        Width of the border stroke, in pixels against ``PREVIEW``'s frame height
        (396): the renderer scales it by its own frame height over that, so a
        border keeps its apparent weight at any resolution. Defaults to ``5``;
        pass ``0`` for no border.
    stroke_color
        Color of the border stroke. Defaults to ``WHITE``. The circuit's
        ``color`` is its *fill* color and does not touch the border; see
        :attr:`~.BezierCircuitCubic.stroke_color`.
    filled
        Whether the interior is painted. Defaults to ``True``; ``False`` leaves
        an outline whose stroke is centred on the path (what
        :class:`~algan.mobs.shapes_2d.Line` uses).
    add_texture_grid
        Whether to build the texture grid at all. Defaults to ``True``. ``False``
        leaves the circuit one color and no per-texel storage, and the
        ``set_color_by_*`` methods then have nothing to write to.
    grid_width
        Number of color samples along the circuit's first basis row -- ``u``,
        left to right on an upright shape. Defaults to ``1``: one flat color,
        which is what a shape wants unless you are painting something across it.
    grid_height
        Number of color samples along the second basis row (``v``). Defaults to
        ``None``, meaning match ``grid_width`` -- except on a circuit
        whose control points are collinear (a straight
        :class:`~algan.mobs.shapes_2d.Line`), where the second row is synthesized
        perpendicular to the path and carries no extent of the shape, so it
        defaults to ``1`` and the grid runs along the line only.
    empty
        Whether the circuit is invisible: fill and border are forced to zero
        opacity. Defaults to ``False``. Used for shapes that exist only to
        position or morph into something else.
    z_index
        Which of two *exactly coplanar* circuits draws in front: the higher
        ``z_index`` wins. Defaults to ``0``, which leaves the shape in author
        order -- coplanar 2-D geometry draws in the order it was created, each
        composite Mob kept whole and drawn parent-first, so an arrow crossing a
        grid authored before it lands on top of that grid without being asked
        to. Raise it to override that: a label over a panel authored after it,
        a highlight over the shape it marks. Setting it propagates to the whole
        sub-hierarchy (see :attr:`~.BezierCircuitCubic.z_index`).

        It is *not* a general depth override. The renderer spends it as a bias
        of a few ten-thousandths of a world unit toward the camera -- enough to
        settle a tie between surfaces at the same depth, far too little to
        reorder anything genuinely in front of or behind. Values are small
        integers; a few hundred would start to shift the shape visibly. Matches
        Manim's attribute of the same name, both in meaning and in being a
        stable sort key over the authored order, and
        :class:`~algan.mobs.manim_mob.ManimMob` carries it across on import.
    shade_in_3d
        Whether to shade the filled circuit as a surface. Defaults to False.
    cap_style
        Open-path endpoint shape: ``"round"``, ``"butt"`` (ends at the endpoint),
        or ``"square"`` (extends half a stroke width). Closed paths have no caps.
        Manim cap enums are also accepted. Defaults to ``"round"``;
        ``"auto"`` and ``None`` select that default.
    joint_type
        Corner shape: ``"round"``, ``"bevel"``, or ``"miter"``. Manim join enums
        are also accepted. Defaults to ``"round"``; ``"auto"`` and ``None``
        select that default. Filled borders retain their configured placement.
    miter_limit
        Maximum miter length divided by the stroke's half-width. Longer miters
        become bevel joins. Must be finite and at least 1. Defaults to ``4``.
    fill_opacity
        Initial fill alpha from 0 to 1, independent of the stroke and multiplied
        by overall ``opacity``. Defaults to None, preserving the fill color's
        alpha. Can be animated later through :attr:`fill_opacity`.
    stroke_opacity
        Initial stroke alpha from 0 to 1, independent of the fill. Defaults to
        None, preserving ``stroke_color``'s alpha. Can be animated later through
        :attr:`stroke_opacity`.
    **kwargs
        Passed to :class:`~algan.animatable_base.mob.Mob` -- notably ``color``,
        which is the fill color. ``location`` is the exception: a circuit's own
        is derived from the control points (the centroid of the region they
        enclose, so the shape turns about itself), so one given here is applied
        as a move onto that point once the frame has been derived, rather than
        replacing it.

    Raises
    ------
    :class:`~.AlganConfigurationError`
        If a stroke style is unknown or the miter limit is not finite and at
        least 1. Rendering also raises for non-default styles on nonplanar
        paths or shaded surface boundaries.

    See Also
    --------
    :meth:`~.BezierCircuitCubic.set_color_by_function` : Color it by a function of ``(u, v)``.
    :meth:`~.BezierCircuitCubic.set_color_by_image` : Paint an image across it.
    :class:`~algan.mobs.surfaces.surface.Surface` : The 3-D counterpart, with the same ``(u, v)`` conventions.

    Examples
    --------
    .. algan:: Example1BezierCircuitCubic
        :save_last_frame:

        from algan import *
        import torch

        square = Square(grid_width=32, grid_height=32, stroke_width=0)
        square.set_color_by_function(
            lambda uv: torch.cat(
                (uv[..., :1], uv[..., 1:], torch.zeros_like(uv[..., :1])), -1
            )
        )
        square.spawn()

        Scene.save_video()
    """

    _morph_family = "bezier"

    # Plain scalar, deliberately not timeline-backed: it selects between
    # coplanar draw orders rather than describing a pose, and animating it
    # would only ever step between discrete orderings. The class default keeps
    # the property readable on a part-built Mob (``ManimCompatMob.__getattr__``
    # would otherwise forward the miss to the backing Manim object).
    _z_index = 0.0

    # Set per batch by ``RenderLoopMixin._authored_draw_order``: the whole draw
    # order resolved to depth bins, of which the authored ``z_index`` is one
    # input. ``None`` means no render has resolved one, and a primitive built
    # directly still honours ``z_index`` on its own.
    _draw_bias = None

    def _render_draw_bias(self):
        """Depth-bin bias this circuit renders with."""
        return self.z_index if self._draw_bias is None else self._draw_bias

    @property
    def z_index(self):
        """Which of two exactly coplanar circuits draws in front (higher wins).

        ``0`` (the default) means author order, which already keeps a composite
        Mob whole and parent-first; this is the override for when that is not
        what you want. Assigning propagates to every circuit below this one in
        the hierarchy, matching Manim's ``set_z_index(..., family=True)``, so a
        composite raises as one thing rather than leaving its parts on opposite
        sides of whatever they cross.

        Animation
        ---------
        Takes effect immediately and is not animated: it selects between
        discrete orderings, so there is nothing to interpolate and no context
        (``Seq``, ``Sync``, ``Off``) changes how it applies. The write reaches
        every circuit in this Mob's sub-hierarchy; plain Mobs in between, such
        as a circuit's texture points, have no draw order and are skipped. It
        may be set before or after :meth:`~.Animatable.spawn` -- the renderer
        reads it afresh for every frame batch.
        """
        return self._z_index

    @z_index.setter
    def z_index(self, value):
        value = float(value)
        self._z_index = value
        # ``children`` is absent while the base Mob is still initializing, and
        # a circuit's texture-point children are plain Mobs with no draw order
        # of their own -- both are skipped rather than special-cased.
        pending = list(getattr(self, "children", None) or ())
        while pending:
            mob = pending.pop()
            if isinstance(mob, BezierCircuitCubic):
                mob._z_index = value
            pending.extend(getattr(mob, "children", None) or ())

    def __init__(
        self,
        control_points: torch.Tensor | list,
        normals: torch.Tensor | None = None,
        stroke_width: float | torch.Tensor = 5,
        stroke_color: Color | torch.Tensor | tuple | list = WHITE,
        filled: bool = True,
        add_texture_grid: bool = True,
        grid_width: int = 1,
        grid_height: int | None = None,
        empty: bool = False,
        z_index: float = 0,
        shade_in_3d: bool = False,
        cap_style: _typing.Any = "round",
        joint_type: _typing.Any = "round",
        miter_limit: float = 4,
        fill_opacity: float | torch.Tensor | None = None,
        stroke_opacity: float | torch.Tensor | None = None,
        **kwargs: _typing.Any,
    ) -> None:
        self.cap_style, self.joint_type, self.miter_limit = _stroke_style(
            cap_style, joint_type, miter_limit
        )
        self.num_bezier_parameters = 4
        self.z_index = z_index
        # Cast first: every other geometry entry point in Algan takes a nested
        # sequence as happily as a tensor, and a bare ``.view`` here made a list
        # of points fail with AttributeError instead.
        control_points = cast_to_tensor(control_points)
        control_points = control_points.reshape(-1, control_points.shape[-1])

        # A circuit's ``location`` is DERIVED below, from the control points --
        # it is the centroid of the region they enclose, not a free parameter --
        # so a caller's own reached ``Mob.__init__`` twice and surfaced as
        # "got multiple values for argument 'location'" from inside the texture
        # grid's construction, naming a line the caller never wrote. It is still
        # a perfectly sensible request (put the finished circuit here), and it
        # is one every shape built on this already answers, so answer it the way
        # ``Circle`` does: derive the frame, then move onto the point asked for.
        requested_location = kwargs.pop("location", None)

        kwargs2 = dict(kwargs.items())

        if "color" in kwargs2:
            # Parsed here rather than in Mob.__init__: this runs first, and it
            # indexes the value's shape, so a hex string or an RGB tuple has to
            # already be a color by now.
            color = to_color(kwargs2["color"])
            kwargs2["color"] = color.reshape(-1, color.shape[-1]).mean(-2)
        if normals is not None:
            normals = normals.reshape(-1, 3)
        (
            kwargs2["location"],
            kwargs2["basis"],
            second_axis_synthesized,
        ) = _circuit_location_and_basis(control_points)

        # Decided once, here, from the authored control points: a circuit whose
        # sub-paths do not lie in planes cannot be rendered by projecting them
        # onto one (see algan.mobs.nonplanar_circuit). The plan is topology
        # only -- the geometry it describes is rebuilt from the live control
        # points every render batch -- but the choice itself is fixed, exactly
        # as the plane in ``basis`` above is.
        #
        # ``shade_in_3d`` also asks for the patch plan on FLAT geometry, which
        # is a lighting decision rather than a geometric one: a patch is 3-D
        # geometry that a material and the scene's lights reach, an analytic
        # circuit is drawn unlit. Stored because ``_after_repack`` has to reach
        # the same verdict.
        self.shade_in_3d = bool(shade_in_3d)
        self._nonplanar_plan = classify_circuit(
            control_points, filled, self.shade_in_3d
        )

        self.grid_width = self.grid_height = 1
        self.num_texture_points = 0
        first_basis = kwargs2["basis"][..., :3]
        second_basis = kwargs2["basis"][..., 3:6]
        self.first_basis = first_basis
        self.second_basis = second_basis

        super().__init__(**kwargs2)
        kwargs["scene"] = self.scene
        self.register_attrs_as_animatable(
            ["stroke_width"],
            BezierCircuitCubic,
        )
        self.filled = filled
        self.empty = empty
        if self.empty:
            self.color = self.color.as_subclass(Color).set_opacity(0)

        texture_triangle_vertices = self.location.squeeze(0)
        if add_texture_grid:
            width = max(int(grid_width), 1)
            if grid_height is None:
                # A collinear circuit's second basis row is synthesized
                # perpendicular to the path, so every point of the shape maps to
                # the same v: sampling it more than once buys nothing.
                height = 1 if second_axis_synthesized else width
            else:
                height = max(int(grid_height), 1)

            a1 = _texture_grid_offsets(width).view(-1, 1, 1)
            a2 = _texture_grid_offsets(height).view(1, -1, 1)
            texture_grid_points = (a1 * first_basis + a2 * second_basis) + self.location
            texture_triangle_vertices = texture_grid_points
            self.grid_width = width
            self.grid_height = height
            texture_triangle_vertices = texture_triangle_vertices.reshape(
                -1, texture_triangle_vertices.shape[-1]
            )
            self.num_texture_points = texture_triangle_vertices.shape[-2]

            # control_points = torch.cat((control_points, texture_triangle_vertices), -2)
        self.stroke_width = cast_to_tensor(stroke_width)
        stroke_color = cast_to_tensor(stroke_color)
        if self.empty:
            stroke_color = stroke_color.as_subclass(Color).set_opacity(0)

        fill_texture_kwargs = dict(kwargs)
        fill_texture_kwargs["color"] = self.color if self.filled else stroke_color
        self.grid = Mob(texture_triangle_vertices, **fill_texture_kwargs)
        self.grid.exclude_from_boundary = True
        self.grid.is_primitive = True
        self.add_children(self.grid)

        border_texture_kwargs = dict(kwargs)
        border_texture_kwargs["color"] = stroke_color
        self.border_grid = Mob(texture_triangle_vertices, **border_texture_kwargs)
        self.border_grid.exclude_from_boundary = True
        self.border_grid.is_primitive = True
        # ``color`` on the circuit means fill color.  The border grid remains a
        # child so it follows transforms and participates in waves/cloning, but
        # it must not inherit ordinary fill-color writes from an ancestor.
        self.border_grid._excluded_from_parent_attrs = frozenset({"color"})
        self.add_children(self.border_grid)

        self.control_points = Mob(control_points, **fill_texture_kwargs)
        self.control_points.is_primitive = True
        self.add_children(self.control_points)
        self.control_points.num_points_per_object = 4
        self.components = [
            self.grid,
            self.border_grid,
            self.control_points,
        ]

        self.normals = normals
        self.is_primitive = True
        self.render_primitive = RENDERER_REGISTRY.bezier_circuit_primitive

        if fill_opacity is not None:
            self.fill_opacity = fill_opacity
        if stroke_opacity is not None:
            self.stroke_opacity = stroke_opacity

        if requested_location is not None:
            # Off, unlike ``Circle``'s: this runs before the circuit exists to
            # the scene, so where it was asked to be built is its starting pose
            # and not a move recorded from somewhere it never was.
            with Off(animation_manager=self.animation_manager):
                self.move_to(requested_location)

    def _after_repack(self):
        """Re-decide the planar/patch/stroke split against the whole pack.

        ``batch_mobs`` clones its first member and then writes every member's
        control points in, so the plan made at construction describes one tile
        of what is now a whole sphere. Classification is per sub-path and the
        pack's sub-paths are its members', so redoing it here reaches the same
        decision the members reached individually -- which is what
        ``ManimMob(..., batch=True)`` relies on.
        """
        self._nonplanar_plan = classify_circuit(
            self.control_points.location.reshape(-1, 3),
            self.filled,
            getattr(self, "shade_in_3d", False),
        )

    @classmethod
    def from_batches(cls, control_point_batches, *args, **kwargs):
        """Build many independently indexable circuits without per-circuit mobs.

        ``control_point_batches`` contains one cubic-bezier tensor per logical
        object.  Geometry is concatenated once while ``parent_batch_sizes``
        retains the control-point boundaries used by rendering and indexed
        views.
        """
        batches = [
            cast_to_tensor(points).reshape(-1, 3) for points in control_point_batches
        ]
        if not batches:
            raise AlganConfigurationError(
                "from_batches requires at least one bezier circuit"
            )
        point_counts = torch.tensor(
            [len(points) for points in batches], dtype=torch.long
        )
        if bool((point_counts % 4 != 0).any()):
            raise AlganConfigurationError(
                "every cubic bezier circuit must contain a multiple of 4 points"
            )

        mob = cls(torch.cat(batches, -2), *args, **kwargs)
        # Every member framed in one pass, each exactly as it would be alone.
        locations, bases = _circuit_locations_and_bases(batches)
        locations = locations.unsqueeze(0)
        bases = bases.unsqueeze(0)
        count = len(batches)

        with Off(
            record_funcs=False,
            record_attr_modifications=False,
            animation_manager=mob.animation_manager,
        ):
            pack_animatable_rows(
                mob, count, overrides={"location": locations, "basis": bases}
            )

            texture_point_count = max(mob.num_texture_points, 1)
            # The same offsets construction laid one member's grid out, so the
            # pack lands on exactly the points the members would have.
            axis_kwargs = {"device": locations.device, "dtype": locations.dtype}
            grid_locations = (
                _texture_grid_offsets(mob.grid_width, **axis_kwargs).view(
                    1, 1, -1, 1, 1
                )
                * bases[..., :3].unsqueeze(-2).unsqueeze(-2)
                + _texture_grid_offsets(mob.grid_height, **axis_kwargs).view(
                    1, 1, 1, -1, 1
                )
                * bases[..., 3:6].unsqueeze(-2).unsqueeze(-2)
                + locations.unsqueeze(-2).unsqueeze(-2)
            ).reshape(1, count * texture_point_count, 3)
            for texture_mob in (
                mob.grid,
                mob.border_grid,
            ):
                pack_member_rows(
                    texture_mob,
                    count,
                    texture_point_count,
                    overrides={"location": grid_locations},
                )

            mob.control_points.parent_batch_sizes = point_counts
        return mob

    def _refine_sampling_for_color_wave(self, direction, max_spacing, pulsed_attrs):
        """Refine the texture grids so a color wave crosses the shape smoothly.

        A circuit's fill and border are colored by bilinearly sampling the
        independent ``grid`` and ``border_grid`` grids laid
        across it. Those grids are a single sample unless
        ``grid_width`` / ``grid_height`` were raised by hand, so
        a shape flashes as one flat color instead of showing the wave
        travelling over it. Lay down a grid fine enough that neighbouring
        samples are no further than ``max_spacing`` apart along the wave (see
        :meth:`~.Mob._refine_sampling_for_color_wave`).

        A circuit whose grid is already rectangular was sized deliberately by
        its author, so it is left exactly as it is rather than being squared up
        for the runtime of a wave.
        """
        if "color" not in pulsed_attrs:
            # Only color is stored per texture point. A circuit's opacity is a
            # single shader parameter for the whole fill, so an opacity wave --
            # the fade Text and Tex spawn with, for one -- cannot be made any
            # smoother by adding texels.
            return None
        if self.empty:
            return None
        size = self.grid_width
        if (
            self.num_texture_points < 1
            or size != self.grid_height
            or self.data_sub_inds is not None
            or self.grid.data_sub_inds is not None
            or self.border_grid.data_sub_inds is not None
        ):
            return None
        objects = self.location.shape[-2]
        expected_points = objects * self.num_texture_points
        if (
            self.grid.location.shape[-2] != expected_points
            or self.border_grid.location.shape[-2] != expected_points
        ):
            return None

        # The grid spans the full width of the circuit's own frame along each of
        # its two basis vectors, so each axis covers twice the projected length
        # of its basis vector, with ``size`` samples over it.
        def projected_span(basis):
            return 2 * dot_product(direction, basis, dim=-1).abs().amax().item()

        span = max(
            projected_span(self.basis[..., :3]), projected_span(self.basis[..., 3:6])
        )
        spacing = span if size < 2 else span / (size - 1)
        if not span > 0 or spacing <= max_spacing:
            return None
        required = int(math.ceil(span / max_spacing)) + 1
        new_size = max(size, min(required, _MAX_WAVE_TEXTURE_RESOLUTION))
        if new_size == size:
            return None
        self._set_texture_grid_resolution(new_size, new_size)

        def restore():
            self._set_texture_grid_resolution(size, size)

        return restore

    def _set_texture_grid_resolution(self, width, height):
        """Internal: rebuild the texture grid at ``width x height`` per circuit.

        The grid is laid out exactly as the constructor lays it out, so the
        renderer's texture lookup is unchanged: ``width`` samples along the first
        basis vector, each holding ``height`` samples along the second. Existing
        per-texel values are resampled onto the new grid. Row counts change, so
        callers must be prepared for the history split this performs.
        """
        old_size = (self.grid_width, self.grid_height)
        old_points = self.num_texture_points
        objects = self.location.shape[-2]
        old_values = {}
        for texture_mob in (self.grid, self.border_grid):
            values = {}
            for attr in dict.fromkeys(texture_mob.animatable_attrs):
                try:
                    values[attr] = getattr(texture_mob, attr).clone()
                except AttributeError:
                    continue
            old_values[texture_mob] = values

        # A different number of texture points cannot be interpolated from the
        # old ones, so the recorded history stays behind on a frozen clone.
        if self.is_spawned():
            self.detach_history()
        self.grid_width, self.grid_height = width, height
        self.num_texture_points = width * height

        location = self.location

        def offsets(size):
            return _texture_grid_offsets(
                size, device=location.device, dtype=location.dtype
            )

        first = self.basis[..., :3].unsqueeze(-2).unsqueeze(-2)
        second = self.basis[..., 3:6].unsqueeze(-2).unsqueeze(-2)
        points = (
            offsets(width).view(-1, 1, 1) * first
            + offsets(height).view(1, -1, 1) * second
            + location.unsqueeze(-2).unsqueeze(-2)
        )
        new_locations = points.reshape(*points.shape[:-4], -1, 3)
        for texture_mob in (self.grid, self.border_grid):
            texture_mob._setattr_and_rebatch_without_record("location", new_locations)

            # Every attribute stored one value per texture point has to follow
            # the new grid; otherwise later writes can no longer broadcast.
            for attr, value in old_values[texture_mob].items():
                if attr == "location" or value.shape[-2] != objects * old_points:
                    continue
                texture_mob._setattr_and_rebatch_without_record(
                    attr, _resample_texture_grid(value, old_size, (width, height))
                )

            if texture_mob.parent_batch_sizes is not None:
                texture_mob.parent_batch_sizes = torch.full(
                    (objects,),
                    self.num_texture_points,
                    dtype=texture_mob.parent_batch_sizes.dtype,
                )
            texture_mob.batch_size = objects * self.num_texture_points
        self._memory_per_timestep_cache = None
        return self

    def get_animatable_attrs(self):
        return {"stroke_width"}.union(super().get_animatable_attrs())

    def _stroke_style_key(self):
        style = _stroke_style(self.cap_style, self.joint_type, self.miter_limit)
        return None if style[:2] == ("round", "round") else style

    #: ``filled`` and ``empty`` decide whether the circuit is a disc or a ring.
    #: ``get_render_primitives`` reads both live and neither is animatable, so a
    #: filled Square becoming an unfilled one used to stay solid -- a full-range
    #: difference over 3.6% of the frame. The colors that go with the fill
    #: arrive separately: ``grid`` and ``border_grid`` are
    #: components, and the morph recurses into them. ``z_index`` is deliberately
    #: absent -- it already reaches the endpoint on its own, and assigning it
    #: here would bypass the setter that propagates it to the sub-hierarchy.
    _MORPH_ADOPTED_ATTRS = (
        *Mob._MORPH_ADOPTED_ATTRS,
        "filled",
        "empty",
        "cap_style",
        "joint_type",
        "miter_limit",
    )

    #: Both of them are also untravellable, and ``filled`` is the sharpest case
    #: of it in the package: the flag does not merely hide the interior, it
    #: decides where the stroke goes (a filled circuit lays its border INWARD
    #: from the outline by default, an unfilled one centres it on the path --
    #: see ``_circuit_point_region``), so no value of anything animatable
    #: interpolates between the two. A pair that crosses it cross-fades.
    _MORPH_UNTRAVELLABLE_ATTRS = (
        *Mob._MORPH_UNTRAVELLABLE_ATTRS,
        "filled",
        "empty",
        "cap_style",
        "joint_type",
        "miter_limit",
    )

    @property
    def fill_opacity(self) -> torch.Tensor:
        """Read or animate the fill color's alpha independently of the stroke.

        Values range from 0 (transparent) to 1 (opaque), with shape ``(*, 1)``
        per fill texel. This is the alpha component of ``color``. Once set, it
        survives later ``color`` assignments whose alpha is 1, which includes
        every named color, so ``box.color = RED`` recolors a translucent box
        without making it opaque. A color with its own alpha below 1 replaces
        it. The Mob's overall ``opacity`` multiplies both fill and stroke at
        render time.

        Animation
        ---------
        Assignment fades this circuit and descendant circuit fills over the
        current context's runtime (1 second by default). RGB, glow and stroke
        alpha are preserved. Use ``with Off(): ...`` for an immediate change.
        Unfilled paths stay unfilled; choose ``filled=True`` before spawning
        to animate a fill from zero alpha.

        Examples
        --------
        .. algan:: Example1CircuitFillOpacity

            from algan import *

            square = Square(fill_opacity=0.2, stroke_opacity=1).spawn()
            square.fill_opacity = 0.8
            square.color = RED  # keeps the 0.8 fill opacity
            Scene.save_video()
        """
        return (self.grid.color if self.filled else self.color)[..., -1:]

    @fill_opacity.setter
    def fill_opacity(self, value: float | torch.Tensor) -> None:
        self._set_component_alpha(value, stroke=False)

    @property
    def stroke_opacity(self) -> torch.Tensor:
        """Read or animate stroke alpha without changing the fill.

        Values range from 0 to 1, shape ``(*, 1)`` per stroke texel. This is the
        alpha component of ``stroke_color``. Once set, it survives later
        ``stroke_color`` assignments whose alpha is 1, such as a named color; a
        color with its own alpha below 1 replaces it. On an unfilled path it
        also survives ``color`` assignments, which recolor that path's stroke.
        Overall Mob ``opacity`` multiplies it at render time.

        Animation
        ---------
        Assignment fades this circuit and descendant circuit strokes over the
        current context's runtime (1 second by default), preserving their RGB
        and glow. Use ``with Off(): ...`` for an immediate change. May be set
        before or after spawning.

        Examples
        --------
        .. algan:: Example1CircuitStrokeOpacity

            from algan import *

            square = Square(fill_opacity=0.5, stroke_opacity=1).spawn()
            square.stroke_opacity = 0
            Scene.save_video()
        """
        return self.border_grid.color[..., -1:]

    @stroke_opacity.setter
    def stroke_opacity(self, value: float | torch.Tensor) -> None:
        self._set_component_alpha(value, stroke=True)

    def _component_alpha_parts(self, *, stroke):
        """The Mobs whose color alpha is this circuit's fill or stroke opacity.

        An unfilled circuit draws its path from ``grid`` as well as
        ``border_grid``, so both carry the stroke there.
        """
        if stroke:
            return [self.border_grid] + ([] if self.filled else [self.grid])
        return [self] + ([self.grid, self.control_points] if self.filled else [])

    def _mark_explicit_alpha(self, *, stroke):
        """Make later default-alpha color writes keep this circuit's alpha.

        See :meth:`~algan.animatable_base.mob.Mob._keep_explicit_alpha`.
        """
        for part in self._component_alpha_parts(stroke=stroke):
            part._explicit_color_alpha = True
        self.scene._has_explicit_color_alpha = True

    def _set_component_alpha(self, value, *, stroke):
        from algan.animatable_base.mob import _validate_opacity

        value = _validate_opacity(cast_to_tensor(value))
        scene = self.scene
        # These writes set alpha deliberately, including back to 1, so they
        # must not be mistaken for a color write that keeps the old alpha.
        previous = getattr(scene, "_writing_component_alpha", False)
        scene._writing_component_alpha = True
        try:
            with Sync(animation_manager=self.animation_manager):
                for circuit in self.get_descendants():
                    if not isinstance(circuit, BezierCircuitCubic):
                        continue
                    for part in circuit._component_alpha_parts(stroke=stroke):
                        colors = part.color.clone()
                        colors[..., -1:] = value
                        part.set_non_recursive(color=colors)
                    circuit._mark_explicit_alpha(stroke=stroke)
        finally:
            scene._writing_component_alpha = previous

    @property
    def stroke_color(self):
        """The color of the circuit's border stroke.

        Separate from :attr:`~algan.animatable_base.mob.Mob.color`, which is the
        *fill*: setting one never changes the other, and on a filled circuit the
        border is drawn inside the outline unless
        ``SETTINGS.style.border_placement`` is ``"centered"``. Accepts anything a color attribute
        does -- a :class:`~algan.constants.color.Color`, a named constant, a hex
        string.

        Reading it back gives the border's per-texel colors as a ``(N, 5)``
        tensor (RGB, glow, alpha), one row per texel of the circuit's texture
        grid, rather than the single value that was assigned -- so compare
        against ``stroke_color[0]`` rather than against a ``Color`` directly.

        Animation
        ---------
        Recorded like any other color attribute: the border cross-fades to the
        new color over the current context's runtime (1 second by default).
        Wrap the write in ``Off()`` to change it instantly.
        """
        return self.border_grid.color

    @stroke_color.setter
    def stroke_color(self, value):
        self.border_grid.color = value

    def get_base_grid(self) -> torch.Tensor:
        """Get the circuit's texture grid, the ``(u, v)`` domain it is colored
        over.

        Values run from 0 to 1 along both axes: ``u`` along the circuit's first
        basis row, ``v`` along its second, which on an upright 2-D shape means
        ``u`` left to right and ``v`` bottom to top. Both rows are as long as the
        distance from the circuit's centre to its furthest control point, so the
        domain covers the square that circumscribes the shape and the shape sits
        in the middle of it. An axis with a single sample carries one color for the whole span
        and is evaluated at its centre, ``0.5``.

        This is the input the ``set_color_by_*`` methods evaluate their
        functions over, so it is what to write those functions in terms of.

        Returns
        -------
        torch.Tensor
            The ``(u, v)`` coordinates, shape
            ``[grid_width, grid_height, 2]``.

        See Also
        --------
        :meth:`~.BezierCircuitCubic.set_color_by_function` : Color the circuit over this grid.
        """
        device = self.grid.location.device

        def axis(size):
            if size < 2:
                return torch.full((1,), 0.5, device=device)
            return torch.linspace(0, 1, size, device=device)

        width, height = self.grid_width, self.grid_height
        return torch.stack(
            (
                axis(width).view(-1, 1).expand(-1, height),
                axis(height).view(1, -1).expand(width, -1),
            ),
            -1,
        )

    def _apply_texture_grid_colors(self, colors, what):
        """Internal: write one color per texel onto the grids that are visible.

        ``colors`` holds one entry per texel of :meth:`get_base_grid`, in that
        grid's own layout. The fill grid always takes them; an unfilled circuit
        has no interior to show them in, so its border grid takes them too --
        the same pairing the constructor makes when it hands an unfilled
        circuit's texture grids the border color.
        """
        colors = Color.add_defaults(cast_to_tensor(colors))
        colors = colors.reshape(-1, colors.shape[-1])
        if colors.shape[-2] != self.num_texture_points:
            raise AlganConfigurationError(
                f"{what} must return one color per texel: expected "
                f"{self.num_texture_points} "
                f"({self.grid_width} x {self.grid_height}), got {colors.shape[-2]}"
            )
        objects = self.location.shape[-2]
        if objects > 1:
            # ``from_batches`` mobs (Text, Tex) pack every circuit's texels into
            # one row block each, and every circuit is colored over its own
            # frame, so the same grid of colors repeats per circuit.
            colors = colors.repeat(objects, 1)
        targets = [self.grid]
        if not self.filled:
            targets.append(self.border_grid)
        with Sync(animation_manager=self.animation_manager):
            for target in targets:
                target.color = colors.unsqueeze(0)
        return self

    def _require_texture_grid(self, method):
        """Internal: refuse to paint a circuit that has nowhere to paint."""
        if self.num_texture_points < 2:
            raise AlganConfigurationError(
                f"{type(self).__name__}.{method} needs a texture grid with more "
                "than one texel, but this circuit has "
                f"{self.num_texture_points}. The grid is the resolution of "
                "anything painted across the shape, and it is one flat color "
                "by default -- construct the shape with e.g. "
                "grid_width=64, grid_height=64."
            )

    def set_color_by_function(self, function):
        """Color the circuit by a function of its ``(u, v)`` parameters.

        Gives each texel of the circuit's texture grid its own color, for
        gradients, heat maps or anything where color carries data, and the
        renderer interpolates between them across the shape. The colors travel
        with the circuit as it moves and morphs.

        The grid is the resolution of the result, and it is a single flat color
        unless you asked for more: build the shape with ``grid_width`` /
        ``grid_height`` (see :class:`~.BezierCircuitCubic`). On a filled
        circuit this colors the fill, leaving ``stroke_color`` alone; on an
        unfilled one, where the stroke is all there is, it colors the stroke.
        A multi-circuit mob (a :class:`~algan.mobs.text.Text`, a
        :class:`~algan.mobs.text.Tex`) colors every circuit over its own frame,
        so the pattern repeats per glyph.

        Animation
        ---------
        Recorded as an animation over the current context's runtime (1 second
        by default), so the colors cross-fade smoothly. Wrap the call in
        ``Off()`` to apply it instantly.

        Parameters
        ----------
        function
            Callable taking a ``(u, v)`` tensor of shape ``[..., 2]``, with both
            coordinates in ``[0, 1]``, and returning colors of shape
            ``[..., 3]`` (RGB), ``[..., 4]`` (RGBA) or ``[..., 5]`` (RGB, glow,
            alpha -- Algan's internal channel order). Channels are in ``[0, 1]``;
            a missing alpha defaults to 1 and a missing glow to 0. Must be
            vectorized -- it is called once on the whole grid, not per texel.

        Returns
        -------
        :class:`~.BezierCircuitCubic`
            This circuit, so calls can be chained.

        Raises
        ------
        ValueError
            If the circuit has a single-texel texture grid, or if ``function``
            returns the wrong number of colors.

        See Also
        --------
        :meth:`~.BezierCircuitCubic.set_color_by_image` : Paint an image on instead.
        :meth:`~.BezierCircuitCubic.get_base_grid` : The ``(u, v)`` grid this evaluates over.

        Examples
        --------
        .. algan:: Example1BezierCircuitCubicSetColorByFunction
            :save_last_frame:

            from algan import *
            import torch

            circle = Circle(grid_width=48, grid_height=48)
            circle.set_color_by_function(
                lambda uv: torch.cat(
                    (uv[..., :1], torch.zeros_like(uv[..., :1]), uv[..., 1:]), -1
                )
            )
            circle.spawn()

            Scene.save_video()
        """
        self._require_texture_grid("set_color_by_function")
        return self._apply_texture_grid_colors(
            function(self.get_base_grid().clone()), "set_color_by_function's function"
        )

    def set_color_by_image(self, rgba_array_or_file_path):
        """Paint an image across the circuit.

        The image is resampled onto the circuit's texture grid and interpolated
        across the shape by the renderer, and it follows the shape as it moves
        and morphs. The image's top-left corner lands at the top left of the
        frame, which on an upright 2-D shape is ``(u, v) == (0, 1)``: ``v`` runs
        up the frame, as it does on a
        :class:`~algan.mobs.surfaces.surface.Surface`, while an image's rows run
        down the picture.

        Unlike :meth:`~algan.mobs.surfaces.surface.Surface.set_color_by_image`,
        which keeps the image at its own resolution, a circuit has no separate
        texture map: the texture grid *is* the resolution, so build the shape
        with a ``grid_width`` / ``grid_height`` matching the
        detail you need. Remember too that the grid spans the square
        circumscribing the shape, so the shape shows the middle of the picture.

        Animation
        ---------
        Recorded as an animation over the current context's runtime (1 second
        by default): the circuit cross-fades, texel by texel, to the image. Wrap
        the call in ``Off()`` to apply it instantly.

        Parameters
        ----------
        rgba_array_or_file_path
            Path to an image file, or an RGBA array of shape ``[H, W, 4]`` or
            ``[H, W, 5]`` with channels in ``[0, 1]``. Paths resolve relative to
            the working directory and then the main script's directory, so an
            image beside your script is found either way.

        Returns
        -------
        :class:`~.BezierCircuitCubic`
            This circuit, so calls can be chained.

        Raises
        ------
        ValueError
            If the circuit has a single-texel texture grid.

        See Also
        --------
        :meth:`~.BezierCircuitCubic.set_color_by_function` : Color it by a function instead.
        :class:`~algan.mobs.image_mob.ImageMob` : An image as a Mob of its own, at full resolution.
        """
        self._require_texture_grid("set_color_by_image")
        from algan.utils.file_utils import get_image

        image = get_image(rgba_array_or_file_path)
        # ``image`` is [row, column, channel] with rows running DOWN the
        # picture; the grid is [u, v] with v running UP the circuit's frame, so
        # the resample lands on (v, u), transposes back, and flips v -- which is
        # the same flip ``mesh.image_to_texture_map`` does for a surface, for the
        # same reason. Without it the picture arrives upside down: the contract
        # is that its top-left corner lands at the top left of the frame, not
        # that its first row lands at v == 0.
        resized = F.interpolate(
            image.permute(2, 0, 1).unsqueeze(0),
            (self.grid_height, self.grid_width),
            mode="bilinear",
            antialias=True,
        ).squeeze(0)
        return self._apply_texture_grid_colors(
            resized.permute(2, 1, 0).flip(1), "set_color_by_image's image"
        )

    def get_default_color(self):
        """Get the color a circuit uses when none was given.

        Returns
        -------
        :class:`~algan.constants.color.Color`
            ``PURPLE``.
        """
        return PURPLE

    def _get_memory_used_per_timestep(self):
        # Called for every circuit every render batch just to size batches;
        # the shape reads below go through the animated-attribute machinery,
        # so cache the result against the global structure version (row
        # re-allocation bumps it).
        from algan.animation_timeline.timeline import STRUCTURE_VERSION

        cache = getattr(self, "_memory_per_timestep_cache", None)
        if cache is not None and cache[0] == STRUCTURE_VERSION[0]:
            return cache[1]
        n_ctrl = self.control_points.location.shape[-2]
        n_tex = self.grid.location.shape[-2]
        n_border_tex = self.border_grid.location.shape[-2]
        n_loc = self.location.shape[-2]
        n_segments = max(n_ctrl // 4, 1)  # cubic beziers have 4 control points each
        # Animation state: control points (3 floats), two color textures (5
        # each), location/basis (6). Texture positions are structural sampling
        # data used by wave animation and are charged alongside their colors.
        animation_bytes = (
            n_ctrl * 3 + (n_tex + n_border_tex) * (3 + 5) + n_loc * 6
        ) * 4
        # Primitive output: control points, fill texture, border texture, and
        # per-circuit normals/border data.
        primitive_bytes = (
            n_segments * 4 * 3 * 4 + (n_tex + n_border_tex) * 5 * 4 + n_loc * 12
        )
        # Sampled edges, metadata and the content-dependent STBVH are charged
        # exactly by the final scene upload instead of guessed here (the old
        # fixed 100-sample estimate was wrong for the actual 1..512 range).
        result = int(animation_bytes + primitive_bytes)
        self._memory_per_timestep_cache = (STRUCTURE_VERSION[0], result)
        return result

    def get_render_primitives(self):
        if self.empty:
            return None
        # Derive transport directly from the material shader parameters. A
        # negative metalness sentinel marks non-PBR materials; Standard and
        # Physical materials expose metalness/roughness as animatable attrs.
        surface_template = self.opacity[..., :1]

        def material_param(name, default):
            if name in self.animatable_attrs:
                return getattr(self, name)
            return torch.full_like(surface_template, default)

        metalness = material_param("metalness", -1.0)
        roughness = material_param("roughness", 0.0)
        # Opacity is coverage and transmission is transparency: independent
        # channels, never folded together (see _derive_material_surface_params).
        transmission = material_param("transmission", 0.0).clamp(0.0, 1.0)
        ior = _circuit_ior(material_param("ior", DIELECTRIC_IOR), metalness)

        shader_vars = broadcast_all(
            [
                self.opacity,
                self.basis,
                self.glow,
                _stroke_width_in_render_pixels(
                    self.stroke_width,
                    getattr(self.scene, "_geometry_view", self.scene).video_settings,
                ),
                metalness,
                roughness,
                ior,
                transmission,
            ],
            ignored_dims=[-1],
        )
        num_control_points = 4  # cubic beziers
        if self._nonplanar_plan is not None:
            if self._stroke_style_key() is not None:
                raise AlganConfigurationError(
                    "Stroke cap/join controls require an unshaded planar path; "
                    "nonplanar paths and shaded boundaries currently require round styles"
                )
            # Not projectable onto one plane: this circuit renders as PN patches
            # and/or per-run circuits built from the same live control points.
            return build_nonplanar_render_primitives(
                self,
                unsquish(self.control_points.location, -2, num_control_points),
                self.grid.get_animated_attribute("color"),
                self.border_grid.get_animated_attribute("color"),
                *shader_vars[:1],
                *shader_vars[2:],
            )
        # Read the color rows as plain tensors. ``mob.color`` hands back a
        # :class:`~algan.constants.color.Color` so callers get its rgb / glow /
        # opacity views, but a Tensor subclass routes *every* subsequent
        # operation through ``__torch_function__``, and building one batch's
        # circuits performs tens of thousands of them. Nothing below this point
        # wants the views -- only the numbers.
        if self.control_points.parent_batch_sizes is None:
            return self._get_render_primitives(
                unsquish(self.control_points.location, -2, num_control_points),
                self.grid.get_animated_attribute("color"),
                self.border_grid.get_animated_attribute("color"),
                self.location,
                self.basis,
                *shader_vars,
            )
        x = self.control_points.location
        tpc = self.grid.get_animated_attribute("color")
        border_tpc = self.border_grid.get_animated_attribute("color")
        num_segments_per_circuit = (
            self.control_points.parent_batch_sizes // num_control_points
        )
        return self._get_render_primitives(
            unsquish((x), -2, num_control_points),
            (tpc),
            (border_tpc),
            self.location,
            self.basis,
            *shader_vars,
            num_segments_per_circuit,
        )

    def _get_render_primitives(
        self,
        x,
        tpc,
        border_tpc,
        loc,
        basis,
        o,
        n,
        g,
        bw,
        reflectivity,
        roughness,
        refractive_index,
        transmission,
        num_segments_per_circuit=None,
    ):
        # x = unsquish(x, -2, num_control_points)
        # assert x.shape == [*, N, num_control_points, 3], where N is number of bezier segments.
        start_points = x[..., :1, :]
        end_points = x[..., -1:, :]

        # We allow for rendering circuits with holes,
        # we treat beziers which don't start at the previous one's end as marking the start of a new circuit (i.e. a hole).
        circuit_start_mask = (start_points - end_points.roll(1, -3)).norm(
            p=2, dim=-1, keepdim=True
        ) > 1e-5
        circuit_end_mask = (end_points - start_points.roll(-1, -3)).norm(
            p=2, dim=-1, keepdim=True
        ) > 1e-5

        if num_segments_per_circuit is not None:
            # Packed members are independent even when their endpoints touch.
            # Restart the subpath scan at every member and compare its final
            # endpoint with its OWN start, as a standalone circuit would.
            counts = num_segments_per_circuit.to(device=x.device)
            member_ends = counts.cumsum(0) - 1
            member_starts = member_ends + 1 - counts
            circuit_start_mask[..., member_starts, :, :] = True
            circuit_end_mask[..., member_ends, :, :] = (
                end_points.index_select(-3, member_ends)
                - start_points.index_select(-3, member_starts)
            ).norm(p=2, dim=-1, keepdim=True) > 1e-5

        inds = torch.arange(x.shape[-3], device=x.device).view(-1, 1, 1)
        circuit_start_inds = torch.where(circuit_start_mask, inds, 0)
        circuit_start_inds = cummax_values(circuit_start_inds, -3)
        # circuit_start_inds now contains the index of the start of the current index's circuit.

        next_segment_inds = (inds + 1) % x.shape[-3]
        if num_segments_per_circuit is not None:
            next_segment_inds[member_ends] = member_starts.view(-1, 1, 1)
        # If the current ind is the end of the circuit, then the next segment is the first ind of this circuit, otherwise it is the next ind.
        next_segment_inds = torch.where(
            circuit_end_mask, circuit_start_inds, next_segment_inds
        )
        # We subtract inds so that each ind is represented as an offset from the current ind.
        # This way, we can concatenate together offsets from different objects, and then just add a torch.arange during rendering
        # to recover the index in the new concatenated tensor.
        next_segment_inds_offset = next_segment_inds - inds

        if num_segments_per_circuit is None:
            starting_inds = circuit_start_mask[0, :, 0, 0].nonzero()[:, 0]
            num_segments_per_circuit = []
            if len(starting_inds) == 0:
                num_segments_per_circuit.append(
                    torch.tensor(
                        (circuit_start_mask.shape[-3],),
                        device=next_segment_inds.device,
                        dtype=next_segment_inds.dtype,
                    ).squeeze()
                )
            else:
                for i in range(len(starting_inds)):
                    num_segments_per_circuit.append(
                        (
                            starting_inds[(i + 1)]
                            if (i + 1) < len(starting_inds)
                            else circuit_start_mask.shape[-3]
                        )
                        - starting_inds[i]
                    )
            # num_segments_per_circuit = torch.stack(num_segments_per_circuit, 0)
            num_segments_per_circuit = torch.tensor(
                [x.shape[-3]], device=x.device, dtype=torch.long
            )
            c = tpc.unsqueeze(-3)
            border_c = border_tpc.unsqueeze(-3)
            texture_point_count = max(self.num_texture_points, 1)
            if texture_point_count > c.shape[-2]:
                c = c.expand([-1, -1, texture_point_count, -1])
            if texture_point_count > border_c.shape[-2]:
                border_c = border_c.expand([-1, -1, texture_point_count, -1])
        else:
            texture_point_count = max(self.num_texture_points, 1)
            c = unsquish(tpc, -2, texture_point_count)
            border_c = unsquish(border_tpc, -2, texture_point_count)

        prim = self.render_primitive(
            x,
            next_segment_inds_offset,
            num_segments_per_circuit,
            c,
            o,
            basis[..., -3:],
            bw,
            border_c,
            loc,
            cast_to_tensor(self.grid_width).expand(-1, loc.shape[1], -1),
            cast_to_tensor(self.grid_height).expand(-1, loc.shape[1], -1),
            basis[..., :3],
            basis[..., 3:6],
            glow=g,
            num_texture_points=self.num_texture_points,
            filled=self.filled,
            stroke_style=self._stroke_style_key(),
            reflectivity=reflectivity,
            roughness=roughness,
            refractive_index=refractive_index,
            transmission=transmission,
            z_index=(
                None
                if (bias := self._render_draw_bias()) == 0.0
                else torch.full(
                    (1, bw.shape[-2], 1), bias, dtype=bw.dtype, device=bw.device
                )
            ),
        )
        prim.num_texture_points = self.num_texture_points
        # A circuit casts a shadow like anything else, so it honours
        # Mob.casts_shadows; receives_shadows is accepted and inert here
        # (2-D geometry renders unlit and receives no shadow to begin
        # with) -- see the primitive's declare_shadow_flags.
        prim.declare_shadow_flags(*self._resolved_shadow_flags())
        return prim

    @animated_function(animated_args={"t": 0.0})
    def draw(self, t: float = 1.0) -> BezierCircuitCubic:
        """Draw the circuit on, as though traced by a pen.

        The path is revealed from its start point to a fraction ``t`` of the way
        round, so animating it is what makes a shape appear stroke by stroke
        rather than fading in. A shape drawn this way ends up with exactly the
        geometry it started with, so it can be moved and morphed afterwards as
        usual.

        Animation
        ---------
        Recorded as an animation: the drawn portion sweeps from 0 to ``t`` over
        the current context's runtime (1 second by default). Wrap the call to
        change that -- ``with Seq(runtime=3): shape.draw()`` -- or in ``Off()``
        to jump straight to the end state. Applies to this circuit only, not to
        its descendants, so a composite Mob draws each of its parts separately.

        Parameters
        ----------
        t
            How much of the path is drawn by the end, from ``0`` (nothing) to
            ``1`` (the whole circuit). Defaults to ``1.0``.

        Returns
        -------
        :class:`~.BezierCircuitCubic`
            This circuit, so calls can be chained.

        Examples
        --------
        .. algan:: Example1BezierCircuitCubicDraw

            from algan import *

            square = Square().spawn()
            square.draw()

            Scene.save_video()
        """
        self._original_control_points = self.control_points.location.clone()
        num_frames = self.control_points.location.shape[0]
        total_control_points = self._original_control_points.shape[-2]
        points = self._original_control_points.expand(num_frames, -1, -1)

        if self.control_points.parent_batch_sizes is not None:
            num_mobs = len(self.control_points.parent_batch_sizes)
        else:
            num_mobs = 1

        num_control_points_per_mob = total_control_points // num_mobs
        N_per_mob = num_control_points_per_mob // 4

        # Reshape points to (num_frames, num_mobs, N_per_mob, 4, 3)
        points_reshaped = points.view(num_frames, num_mobs, N_per_mob, 4, 3)

        # Ensure t is a tensor and has shape (num_frames, num_mobs, 1, 1)
        t = cast_to_tensor(t).to(points.device)
        while t.dim() < 3:
            t = t.unsqueeze(0)
        if t.shape[1] != num_mobs:
            t = t.expand(-1, num_mobs, -1)
        t = t.unsqueeze(-1)  # (num_frames, num_mobs, 1, 1)

        # Calculate local b parameters
        inds_local = torch.arange(N_per_mob, device=points.device, dtype=points.dtype)
        b = (N_per_mob * t - inds_local.view(1, 1, N_per_mob, 1)).clamp(
            0.0, 1.0
        )  # (num_frames, num_mobs, N_per_mob, 1, 1)

        # Portion matrix coefficients for each segment
        mb = 1.0 - b
        b2 = b * b
        mb2 = mb * mb
        b3 = b2 * b
        mb3 = mb2 * mb

        # Construct portion_matrix of shape (num_frames, num_mobs, N_per_mob, 4, 4)
        portion_matrix = torch.zeros(
            (num_frames, num_mobs, N_per_mob, 4, 4),
            device=points.device,
            dtype=points.dtype,
        )
        portion_matrix[..., 0, 0] = 1.0

        portion_matrix[..., 1, 0] = mb.squeeze(-1)
        portion_matrix[..., 1, 1] = b.squeeze(-1)

        portion_matrix[..., 2, 0] = mb2.squeeze(-1)
        portion_matrix[..., 2, 1] = 2.0 * mb.squeeze(-1) * b.squeeze(-1)
        portion_matrix[..., 2, 2] = b2.squeeze(-1)

        portion_matrix[..., 3, 0] = mb3.squeeze(-1)
        portion_matrix[..., 3, 1] = 3.0 * mb2.squeeze(-1) * b.squeeze(-1)
        portion_matrix[..., 3, 2] = 3.0 * mb.squeeze(-1) * b2.squeeze(-1)
        portion_matrix[..., 3, 3] = b3.squeeze(-1)

        # Compute new control points
        new_points = torch.matmul(portion_matrix, points_reshaped)

        # Reshape back to (num_frames, total_control_points, 3)
        new_points = new_points.view(num_frames, total_control_points, 3)

        # Set the control points location absolute
        self.control_points.location = new_points
        return self

    def _set_control_points_to_partial(self, full_control_points, start_t, end_t):
        full_control_points = cast_to_tensor(full_control_points)
        start_t = cast_to_tensor(start_t).to(full_control_points)
        end_t = cast_to_tensor(end_t).to(full_control_points)

        def frame_values(value, name):
            if value.numel() == 1:
                return value.reshape(1)
            values = value.reshape(value.shape[0], -1)
            if values.shape[1] != 1:
                raise AlganConfigurationError(
                    f"{name} must contain one value per frame"
                )
            return values[:, 0]

        start_t = frame_values(start_t, "start_t")
        end_t = frame_values(end_t, "end_t")
        num_frames = max(full_control_points.shape[0], start_t.numel(), end_t.numel())
        if full_control_points.shape[0] == 1:
            full_control_points = full_control_points.expand(num_frames, -1, -1)
        elif full_control_points.shape[0] != num_frames:
            raise AlganConfigurationError(
                "full_control_points must have one row or one row per frame"
            )
        if start_t.numel() == 1:
            start_t = start_t.expand(num_frames)
        elif start_t.numel() != num_frames:
            raise AlganConfigurationError("start_t must have one value per frame")
        if end_t.numel() == 1:
            end_t = end_t.expand(num_frames)
        elif end_t.numel() != num_frames:
            raise AlganConfigurationError("end_t must have one value per frame")

        total_control_points = full_control_points.shape[-2]

        if self.control_points.parent_batch_sizes is not None:
            num_mobs = len(self.control_points.parent_batch_sizes)
        else:
            num_mobs = 1

        num_control_points_per_mob = total_control_points // num_mobs
        N_per_mob = num_control_points_per_mob // 4

        points_reshaped = full_control_points.view(
            num_frames, num_mobs, N_per_mob, 4, 3
        )

        j = torch.arange(
            N_per_mob,
            device=full_control_points.device,
            dtype=full_control_points.dtype,
        ).view(1, 1, N_per_mob, 1, 1)
        s_start = j / N_per_mob
        s_end = (j + 1) / N_per_mob

        a = torch.clamp(start_t.view(-1, 1, 1, 1, 1), min=s_start, max=s_end)
        b = torch.clamp(end_t.view(-1, 1, 1, 1, 1), min=s_start, max=s_end)

        local_a = (a - s_start) * N_per_mob
        local_b = (b - s_start) * N_per_mob

        P0 = points_reshaped[..., 0, :]
        P1 = points_reshaped[..., 1, :]
        P2 = points_reshaped[..., 2, :]
        P3 = points_reshaped[..., 3, :]

        b_t = local_b.squeeze(-1)
        mb_t = 1.0 - b_t

        Q0 = P0
        Q1 = mb_t * P0 + b_t * P1
        Q2 = mb_t**2 * P0 + 2.0 * mb_t * b_t * P1 + b_t**2 * P2
        Q3 = (
            mb_t**3 * P0
            + 3.0 * mb_t**2 * b_t * P1
            + 3.0 * mb_t * b_t**2 * P2
            + b_t**3 * P3
        )

        u = torch.where(b_t > 1e-6, local_a.squeeze(-1) / b_t, torch.zeros_like(b_t))
        u = torch.clamp(u, 0.0, 1.0)
        mu = 1.0 - u

        R3 = Q3
        R2 = u * Q3 + mu * Q2
        R1 = u**2 * Q3 + 2.0 * u * mu * Q2 + mu**2 * Q1
        R0 = u**3 * Q3 + 3.0 * u**2 * mu * Q2 + 3.0 * u * mu**2 * Q1 + mu**3 * Q0

        new_points = torch.stack([R0, R1, R2, R3], -2).view(
            num_frames, total_control_points, 3
        )
        self.control_points.location = new_points
        return self


class BezierCurveCubic(BezierCircuitCubic):
    """An open path of cubic bezier curves -- a stroke with no interior.

    A :class:`BezierCircuitCubic` with ``filled=False``, which is the whole
    difference: there is no fill to paint, so the color you set is the stroke's,
    :attr:`~.BezierCircuitCubic.stroke_color`. The stroke stays centred on the
    path rather than being laid inward from an outline, which is what
    :class:`~algan.mobs.shapes_2d.Line` is built on.

    The control points need not describe a closed loop, and need not lie in a
    plane either: an open path bounds no surface, so a 3-D curve keeps its true
    position in space and is drawn by splitting it into near-straight runs, each
    turned to face the camera so the stroke keeps a constant width on screen.
    That decision is made once, from the control points you construct it with
    (see :doc:`the tutorial </advanced_user_tutorials/bezier_curves>`).
    Nothing stops the path from closing -- a closed path drawn as a curve is an
    outline.

    Parameters
    ----------
    *args, **kwargs
        Passed to :class:`~.BezierCircuitCubic`, except ``filled``, which is
        always ``False``. ``color`` stands in for ``stroke_color`` when that was
        not given, as it does on a :class:`~algan.mobs.shapes_2d.Line`, since an
        unfilled path has no fill for it to act on otherwise.

    See Also
    --------
    :class:`~.BezierCircuitCubic` : The filled counterpart, and where the parameters are documented.

    Examples
    --------
    .. algan:: Example1BezierCurveCubic
        :save_last_frame:

        from algan import *
        import torch

        BezierCurveCubic(
            torch.tensor([[-2.0, -1.0, 0.0], [-1.0, 2.0, 0.0],
                          [1.0, -2.0, 0.0], [2.0, 1.0, 0.0]]),
            stroke_color=YELLOW,
        ).spawn()

        Scene.save_video()
    """

    def __init__(self, *args, filled=None, **kwargs):
        if filled:
            raise AlganConfigurationError(
                "BezierCurveCubic is the unfilled circuit, so it cannot be "
                "filled. Use BezierCircuitCubic(..., filled=True) for a path "
                "with an interior."
            )
        # ``color`` is the FILL color, and this circuit has no fill: left alone
        # it would silently do nothing, which is the one way a curve comes out
        # white after being asked for a color. ``Line`` -- the other unfilled
        # circuit a user reaches for -- already reads it as the stroke's
        # (``_translate_vector_style_kwargs``, ``line=True``), so this reads it
        # the same way. An explicit ``stroke_color`` still wins; so does one
        # passed positionally, which is ``BezierCircuitCubic``'s fourth
        # parameter.
        if "color" in kwargs and "stroke_color" not in kwargs and len(args) < 4:
            kwargs["stroke_color"] = kwargs["color"]
        super().__init__(*args, filled=False, **kwargs)


def build_render_primitives_batched(actors, scene):
    """Build the merged (collection-level) bezier render primitive for
    ``actors`` in one vectorized pass.

    Byte-identical replacement for calling ``get_render_primitives()`` on
    every actor and concatenating the per-actor primitives through
    ``BezierCircuitPrimitive(triangle_collection=...)``: each attribute is
    read from its timeline once for the whole group (contiguous rows read as
    a single slice), and the per-segment circuit topology (subpath start/end
    masks, next-segment indices) is computed with per-actor index maps that
    reproduce each actor's local ``roll``/``cummax`` wrap-around semantics.

    Callers must guarantee (see ``RenderLoopMixin._is_batchable_bezier`` and
    ``_build_deferred_beziers``): stock ``BezierCircuitCubic`` build methods,
    not ``empty``, un-batched control points, singleton rows for the scalar
    attributes, and uniform ``num_texture_points`` / ``filled`` / fill and
    border texture-color row counts / primitive class across the group.
    """
    from algan.animation_timeline.timeline import RowRanges
    from algan.rendering.primitives.bezier_circuit_primitive import (
        chord_tolerance_pixels,
    )

    timeline = scene.timeline_manager
    first = actors[0]
    ntp = first.num_texture_points
    M = len(actors)

    def read(attr, mobs):
        tl = timeline.attr_to_timeline[attr]
        # Merge the per-mob cached [begin, end) runs (ranges_for) instead of
        # rebuilding them from the index tensors: this is called every frame
        # batch, and tensor->int conversion per mob dominates otherwise.
        pairs = []
        for m in mobs:
            r = tl.ranges_for(m.id)
            if r.pairs is None:  # non-contiguous rows (defensive)
                return tl.get(
                    RowRanges(
                        None,
                        tensor=torch.cat([tl.mob_id_to_inds[mm.id] for mm in mobs]),
                    )
                )
            for b, e in r.pairs:
                if pairs and pairs[-1][1] == b:
                    pairs[-1] = (pairs[-1][0], e)
                else:
                    pairs.append((b, e))
        return tl.get(RowRanges(pairs))

    # --- batched attribute reads (mirrors the per-actor property reads and
    # the ``vars`` broadcast in get_render_primitives) ---
    o = read("opacity", actors)

    def read_optional_material(attr, default):
        values = []
        for actor in actors:
            if attr in actor.animatable_attrs:
                tl = timeline.attr_to_timeline[attr]
                values.append(tl.get(tl.ranges_for(actor.id)))
            else:
                values.append(torch.full_like(o[:, :1, :1], default))
        values, _ = _unify_time(values, f"bezier {attr} merge")
        return torch.cat(values, 1)

    reflectivity = read_optional_material("metalness", -1.0)
    roughness = read_optional_material("roughness", 0.0)
    # Opacity is coverage, transmission is transparency: independent channels
    # (see _derive_material_surface_params). ``o`` is left alone.
    transmission = read_optional_material("transmission", 0.0).clamp(0.0, 1.0)
    refractive_index = _circuit_ior(
        read_optional_material("ior", DIELECTRIC_IOR), reflectivity
    )
    basis = read("basis", actors)
    g = read("glow", actors)
    bw = _stroke_width_in_render_pixels(
        read("stroke_width", actors),
        getattr(scene, "_geometry_view", scene).video_settings,
    )
    loc = read("location", actors)
    o, basis, g, bw = broadcast_all([o, basis, g, bw], ignored_dims=[-1])
    cp = read("location", [a.control_points for a in actors])
    tpc = read("color", [a.grid for a in actors])
    border_tpc = read("color", [a.border_grid for a in actors])

    # --- circuit topology (mirrors _get_render_primitives) ---
    loc_inds = timeline.attr_to_timeline["location"].mob_id_to_inds
    seg_counts = torch.tensor(
        [loc_inds[a.control_points.id].numel() // 4 for a in actors], dtype=torch.long
    )
    x = unsquish(cp, -2, 4)  # [T, S_total, 4, 3]
    S_tot = x.shape[-3]
    seg_offsets = seg_counts.cumsum(0) - seg_counts
    mob_of_seg = torch.repeat_interleave(torch.arange(M), seg_counts)
    off_of_seg = seg_offsets[mob_of_seg]
    gidx = torch.arange(S_tot)
    local = gidx - off_of_seg
    last_local = seg_counts[mob_of_seg] - 1

    start_points = x[..., :1, :]
    end_points = x[..., -1:, :]
    # Per-actor wrap-around neighbours: each actor's own roll(+-1, -3).
    prev_idx = torch.where(local == 0, off_of_seg + last_local, gidx - 1)
    next_idx = torch.where(local == last_local, off_of_seg, gidx + 1)
    circuit_start_mask = (start_points - end_points.index_select(-3, prev_idx)).norm(
        p=2, dim=-1, keepdim=True
    ) > 1e-5
    circuit_end_mask = (end_points - start_points.index_select(-3, next_idx)).norm(
        p=2, dim=-1, keepdim=True
    ) > 1e-5

    local_col = local.view(-1, 1, 1)
    off_col = off_of_seg.view(-1, 1, 1)
    # The per-actor where(mask, local_ind, 0) + cummax scan, run in global
    # index space: candidate values are per-actor monotone blocks (every
    # actor's candidates are >= its offset and below the next actor's), so
    # one global cummax restarts cleanly at every actor boundary.
    circuit_start_inds = torch.where(circuit_start_mask, local_col + off_col, off_col)
    circuit_start_inds = cummax_values(circuit_start_inds, -3) - off_col
    next_segment_inds = torch.where(
        local == last_local, torch.zeros_like(local), local + 1
    ).view(-1, 1, 1)
    next_segment_inds = torch.where(
        circuit_end_mask, circuit_start_inds, next_segment_inds
    )
    next_segment_inds_offset = next_segment_inds - local_col  # [T, S, 1, 1]

    # --- texture colors (mirrors the ``c`` construction) ---
    texture_point_count = max(ntp, 1)

    def texture_colors(values):
        colors = unsquish(values, -2, values.shape[-2] // M)
        if texture_point_count > colors.shape[-2]:
            colors = colors.expand([-1, -1, texture_point_count, -1])
        return colors

    c = texture_colors(tpc)  # [T, M, P, 5]
    bc = texture_colors(border_tpc)

    # --- per-primitive color/border math (mirrors
    # BezierCircuitPrimitive.__init__'s scalar path) ---
    normals = basis[..., -3:]
    colors, fill_opacity, fill_glow = broadcast_all(
        [c, o.unsqueeze(-2), g.unsqueeze(-2)], ignored_dims=[-1]
    )
    colors = colors.clone()
    colors[..., -2:-1] += fill_glow
    colors[..., -1:] *= fill_opacity
    bc, border_opacity, border_glow = broadcast_all(
        [bc, o.unsqueeze(-2), g.unsqueeze(-2)], ignored_dims=[-1]
    )
    bc = bc.clone()
    bc[..., -2:-1] += border_glow
    bc[..., -1:] *= border_opacity

    # --- collection-level assembly (mirrors the triangle_collection branch
    # of BezierCircuitPrimitive.__init__) ---
    # Keep the deferred mega-primitive on the materialized animation/source
    # device.  The prefetch worker must not upload the next batch while the
    # current one occupies the render device; upload happens at the managed
    # render-memory boundary.
    device = x.device
    cls = first.render_primitive
    mega = cls.__new__(cls)
    # The ray tracer interprets this legacy density setting as the maximum
    # screen-space curve-to-chord error in pixels. It must be the value
    # BezierCircuitPrimitive's constructor defaults to, because this builder's
    # whole contract is to be a byte-identical replacement for that
    # constructor: it stood at 1 against the per-actor path's 0.5, which the
    # default analytic-AA route hides (it clamps the tolerance to
    # analytic_aa_chord_tolerance = 0.25, so 0.5 and 1 both land on 0.25) and
    # the classic supersampled route does not -- there every batched circuit
    # was flattened to twice the per-actor path's chord error. Harmless while
    # the batched build reached a fifth of a scene's circuits; not harmless now
    # that a group clash no longer sends the rest down the other path (P9).
    mega.num_pixels_per_sample = chord_tolerance_pixels
    mega.num_bezier_parameters = 4
    mega.num_texture_points = ntp
    mega.filled = first.filled
    mega.num_segments_per_object = seg_counts.to(device)
    mega.corners = x.to(device)
    cols = colors.to(device)
    if ntp == 0:
        cols = cols.squeeze(-2)
    mega.next_segment_inds = next_segment_inds_offset.to(device) + torch.arange(
        S_tot, device=device
    ).view(-1, 1, 1)
    mega.normals = normals.to(device)
    mega.stroke_width = bw.to(device)
    mega.stroke_color = bc.to(device)

    T = loc.shape[0]

    def per_actor_int(vals):
        return (
            torch.tensor([float(v) for v in vals]).view(1, M, 1).int().expand(T, -1, -1)
        )

    # ``_is_batchable_bezier`` guarantees one attribute row -- and therefore one
    # circuit -- per actor here, so the lane is simply the actors' scalars.
    zs = [float(a._render_draw_bias()) for a in actors]
    mega._has_z_index = any(z != 0.0 for z in zs)
    mega.z_index = (
        torch.tensor(zs, dtype=bw.dtype).view(1, M, 1).to(device)
        if mega._has_z_index
        else torch.zeros((1, M, 1), dtype=bw.dtype, device=device)
    )

    mega.mob_center = loc.to(device)
    mega.grid_width = per_actor_int([a.grid_width for a in actors]).to(device)
    mega.grid_height = per_actor_int([a.grid_height for a in actors]).to(device)
    mega.basis1 = basis[..., :3].to(device)
    mega.basis2 = basis[..., 3:6].to(device)
    mega.reflectivity = reflectivity.to(device)
    mega.roughness = roughness.to(device)
    mega.refractive_index = refractive_index.to(device)
    mega.transmission = transmission.to(device)
    if ntp > 0:
        cols = cols[..., -ntp:, :]
    mega.colors = cols
    return mega
