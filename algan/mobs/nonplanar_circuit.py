"""Rendering bezier circuits whose control points do not lie in one plane.

A :class:`~algan.mobs.bezier_circuit.BezierCircuitCubic` is normally a *planar*
outline: the renderer intersects a camera ray with the circuit's own plane and
decides coverage analytically in that plane's ``(u, v)`` coordinates, which is
what keeps a circle exactly round and a glyph crisp at any zoom.  Getting there
costs an orthogonal projection of the control points onto that plane
(``_build_circuit_geometry``), and for a genuinely planar shape that projection
is the identity.

Manim's outlines carry no such restriction.  Its 3-D objects are built from
bezier geometry with control points anywhere in space: a ``Surface`` -- and every
shape built on it, ``Sphere``, ``Torus``, ``Cone`` -- is a grid of curved quad
tiles, and a 3-D ``ParametricFunction`` is a single stroked path that leaves its
own plane entirely.  Projecting those onto one plane is not an approximation, it
is a different shape.  Measured on stock Manim geometry: per-tile flattening of a
``Sphere()`` moves a control point *shared by two neighbouring tiles* 0.017 world
units apart, which is an open seam, and a helix ``ParametricFunction`` loses its
entire radius and renders as a flat sinusoid.

This module is the front end's answer.  Every circuit is classified once, at
construction, into one of three cases:

``planar``
    Every sub-path lies in a plane to within :data:`PLANARITY_TOLERANCE`.  This
    is every 2-D shape, every glyph of every ``Text`` and ``Tex``, every SVG and
    every planar Manim mobject -- the overwhelming majority.  Nothing about them
    changes: they keep the analytic bezier path exactly as before, and this
    module never runs again for them.

``patch`` (non-planar and filled)
    Each closed sub-path -- one Manim tile -- becomes **logical PN triangles**,
    the same curved-patch primitive an Algan
    :class:`~algan.mobs.surfaces.surface.Surface` is made of, diced to flat
    triangles per patch per frame.  A tile's corners are its cubics' shared
    endpoints and their normals come from the two boundary tangents meeting
    there, so the patch reproduces the tile's curvature instead of collapsing
    it, and two tiles sharing an edge build that edge's PN curve from the same
    endpoints.

``stroke`` (non-planar and unfilled)
    An open path bounds no surface, so there is no interior to define and no
    depth to invent.  It stays an analytic circuit and the plane problem is
    solved by *splitting*: consecutive segments are grouped into maximal
    near-straight runs, and each run becomes its own circuit whose plane is
    turned to face the camera about the run's own axis (:func:`run_planes`).
    A helix then keeps its true position in space while its stroke keeps the
    constant screen-space width the renderer gives every other circuit -- which
    is what Manim draws, and what a swept 3-D tube would not.

The classification is fixed at construction, which is the commitment the
circuit's plane already makes (``basis`` is derived once and does not track
control-point animation).  The *geometry* is rebuilt from the live control
points on every render batch, so transforms, animation and ``become`` all follow
without the plan going stale.

Everything a non-planar circuit produces is ordinary geometry -- PN patches and
planar circuits -- so shadow, reflection and refraction rays intersect exactly
what the camera sees.  There is no second, screen-space description of these
surfaces for the ray-traced side to disagree with.

Set ``ALGAN_NONPLANAR_CIRCUITS=0`` to classify everything as planar, i.e. to
restore the flattening this module replaces.

Known limits, all of them exotic in Manim-imported geometry:

* A filled non-planar circuit's *holes* are filled.  Each closed sub-path
  becomes its own patch group, so the even-odd fill rule that carves a hole out
  of a planar glyph has no equivalent here.  Manim's 3-D tiles have no holes.
* Corner normals are estimated from one tile's own boundary tangents, which is a
  one-sided difference of the underlying surface.  Two tiles sharing a corner
  therefore agree on its position exactly but on its normal only to the tiling's
  own accuracy (~2.5 degrees on a stock ``Sphere()``, a sub-pixel seam at 1080p,
  against the 0.017-world-unit gap flattening opens there).
* A non-planar circuit's texture grid is collapsed to one color per member.
  The grid is laid out across a circuit's plane frame, which a patch group and a
  split stroke no longer share.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from algan.environment import env_flag
from algan.rendering.mps_compat import reduction_index_dtype

#: A sub-path counts as planar when ``sqrt(lambda_min / lambda_max)`` of its
#: control points' covariance -- RMS spread off the best-fit plane as a fraction
#: of RMS spread along the shape, and so invariant to scale and rotation -- is at
#: most this.  The measure separates the two populations by orders of magnitude
#: rather than by a margin: geometry that is planar by construction lands at
#: float noise (~1e-7 for a rotated ``Square`` built in float32), while a stock
#: ``manim.Sphere()`` tile measures ~0.03 and a 3-D helix ~0.39.
PLANARITY_TOLERANCE = 1e-3

#: How far a stroke run may bend before it is cut in two:
#: ``sqrt(lambda_mid / lambda_max)`` of its control points' covariance, i.e. its
#: RMS extent across itself as a fraction of its extent along itself.  A run's
#: plane is turned to face the camera (see :func:`run_planes`), which discards
#: exactly that across-itself extent, so this is the bound on how far a stroke
#: can be displaced from the path it draws.  At 0.02 a curve is cut roughly
#: every 9 degrees of turn.
STRAIGHTNESS_TOLERANCE = 0.02

#: Longest run of segments the stroke split will join into one circuit.  Only a
#: bound on the cost of the greedy search, which refits the run per extension;
#: a curve is cut by :data:`STRAIGHTNESS_TOLERANCE` long before this, and a
#: dead-straight path is not distorted by any run length.
MAX_STROKE_RUN_SEGMENTS = 64

#: Render tolerance for the PN patches a filled non-planar circuit becomes, in
#: output pixels at the renderer's reference frame height. Twice a Surface's,
#: because a circuit's patches carry no curvature of their own to resolve.
PATCH_RENDER_TOLERANCE_PIXELS = 1.0

#: How far in front of its own patch, in depth-tie bins per world unit of patch
#: bulge, a border stroke is placed.  The displacement runs along the view axis,
#: so it changes only the depth ordering and never where the stroke lands on
#: screen (see ``_apply_z_index_bias``); it reproduces Manim's fill-then-stroke
#: order for a tile whose fill is now curved and whose stroke is not.
BORDER_DEPTH_BIAS_BINS_PER_UNIT = 2.0e4


def nonplanar_circuits_enabled():
    """Whether non-planar circuits are given 3-D geometry (default True)."""
    return env_flag("ALGAN_NONPLANAR_CIRCUITS", True)


class NonPlanarPlan:
    """The construction-time decision for one non-planar circuit.

    Topology only -- segment groupings and vertex index lists.  Every position,
    normal and color is read from the live timeline at render time, so the plan
    survives any amount of animation.
    """

    __slots__ = (
        "mode",
        "run_starts",
        "run_counts",
        "corner_seg",
        "corner_prev_seg",
        "corner_next",
        "corner_subpath",
        "subpath_corner_counts",
        "tri_index",
        "sagitta",
    )

    def __init__(
        self,
        mode,
        run_starts,
        run_counts,
        corner_seg=None,
        corner_prev_seg=None,
        corner_next=None,
        corner_subpath=None,
        subpath_corner_counts=None,
        tri_index=None,
        sagitta=0.0,
    ):
        self.mode = mode
        self.run_starts = run_starts
        self.run_counts = run_counts
        self.corner_seg = corner_seg
        self.corner_prev_seg = corner_prev_seg
        self.corner_next = corner_next
        self.corner_subpath = corner_subpath
        self.subpath_corner_counts = subpath_corner_counts
        self.tri_index = tri_index
        self.sagitta = float(sagitta)

    @property
    def num_runs(self):
        return int(self.run_counts.numel())

    @property
    def num_subpaths(self):
        return int(self.subpath_corner_counts.numel())


def _window_covariances(points, starts, ends):
    """Covariance of every window ``points[starts[k]:ends[k])``, as ``[K, 3, 3]``.

    Prefix sums over the point array, so K windows of any lengths cost two
    cumulative sums and K gathers rather than K reductions.  That matters
    because this runs at the construction of *every* circuit: a page of ``Text``
    is one circuit with a thousand sub-paths, and measuring them one at a time
    cost 0.35s of pure Python-loop overhead before this was batched.

    Double precision throughout, and the points are centred on their global mean
    first: geometry that is planar by construction has to come out at float
    noise for :data:`PLANARITY_TOLERANCE` to mean anything, and differencing two
    large prefix sums is exactly how that precision gets lost.
    """
    points = points.reshape(-1, 3).double()
    if starts.shape[0] == 1:
        # One window is the overwhelmingly common shape -- a Square, a Circle,
        # any glyph without a counter -- and it is reached once per constructed
        # circuit, so it skips the prefix machinery for four ops.
        selected = points[int(starts[0]) : int(ends[0])]
        centred = selected - selected.mean(0, keepdim=True)
        return (centred.T @ centred / max(selected.shape[0], 1)).unsqueeze(0)
    points = points - points.mean(0, keepdim=True)
    zero1 = points.new_zeros((1, 3))
    zero2 = points.new_zeros((1, 3, 3))
    prefix1 = torch.cat((zero1, points.cumsum(0)), 0)
    prefix2 = torch.cat(
        (zero2, (points.unsqueeze(-1) * points.unsqueeze(-2)).cumsum(0)), 0
    )
    counts = (ends - starts).clamp_min(1).double().view(-1, 1)
    total = prefix1[ends] - prefix1[starts]
    second = prefix2[ends] - prefix2[starts]
    mean = total / counts
    return second / counts.unsqueeze(-1) - mean.unsqueeze(-1) * mean.unsqueeze(-2)


def _eigen_ratios(covariances, index):
    """``sqrt(lambda[index] / lambda[2])`` per covariance, zero where degenerate.

    ``index`` 0 is the spread *off* the best-fit plane and 1 the spread across
    the shape, both as a fraction of the spread along it -- dimensionless, and
    invariant to scale and rotation, which is what lets one threshold serve a
    glyph and a sphere tile alike.
    """
    eigenvalues = torch.linalg.eigvalsh(covariances).clamp_min(0.0)
    largest = eigenvalues[..., 2]
    ratio = (eigenvalues[..., index] / largest.clamp_min(1e-30)).sqrt()
    return torch.where(largest <= 1e-30, torch.zeros_like(ratio), ratio)


def _whole_range(points):
    starts = torch.zeros(1, dtype=torch.long)
    ends = torch.tensor([points.reshape(-1, 3).shape[0]], dtype=torch.long)
    return starts, ends


def plane_residual_ratio(points):
    """``sqrt(lambda_min / lambda_max)`` of ``points``' covariance, or 0.

    The eigenvalues of a point set's covariance are its squared RMS spreads along
    the principal axes, so the smallest is the spread *off* the best-fit plane
    and the largest the spread along the shape.

    Geometry that is planar by construction has a residual at float noise, and
    telling that apart from a real bulge is the whole job.  Collinear points (a
    straight 3-D line) give zero -- degenerately planar, which is exactly how the
    renderer already treats them.
    """
    points = points.reshape(-1, 3)
    if points.shape[0] < 3:
        return 0.0
    return float(
        _eigen_ratios(_window_covariances(points, *_whole_range(points)), 0)[0]
    )


def straightness_ratio(points):
    """``sqrt(lambda_mid / lambda_max)`` of ``points``' covariance, or 0.

    The middle eigenvalue of a point set's covariance is its squared RMS spread
    along the widest direction *across* the shape, so this is how far the set
    departs from a straight line, as a fraction of its length.  Collinear points
    give zero at any length; a shallow arc grows it with the arc's turn.
    """
    points = points.reshape(-1, 3)
    if points.shape[0] < 3:
        return 0.0
    return float(
        _eigen_ratios(_window_covariances(points, *_whole_range(points)), 1)[0]
    )


def _expand_ranges(starts, ends):
    """Contiguous ranges as ``(flat index, owning range)``, both ``[sum lengths]``.

    The one shape reduction over ranges of different lengths needs, and the
    counterpart to :func:`_window_covariances`: build it once here rather than
    re-deriving the offsets at each use.
    """
    lengths = (ends - starts).clamp_min(0)
    owner = torch.repeat_interleave(torch.arange(starts.shape[0]), lengths)
    offsets = torch.repeat_interleave(lengths.cumsum(0) - lengths, lengths)
    local = torch.arange(int(lengths.sum())) - offsets
    return starts[owner] + local, owner


def _window_sagittae(points, starts, ends, covariances):
    """Greatest distance from each window's points to its own best-fit plane.

    ``[K]``, in world units.  Sizes the depth bias that keeps a border stroke in
    front of the curved patch it outlines: the two are coincident by
    construction, and this is how far the patch bulges away from the flat
    outline.
    """
    points = points.reshape(-1, 3).double()
    normals = torch.linalg.eigh(covariances).eigenvectors[..., :, 0]  # [K, 3]
    index, owner = _expand_ranges(starts, ends)
    selected = points[index]
    counts = (ends - starts).clamp_min(1).double().view(-1, 1)
    centres = torch.zeros_like(normals).index_add_(0, owner, selected) / counts
    offset = ((selected - centres[owner]) * normals[owner]).sum(-1).abs()
    return torch.zeros(starts.shape[0], dtype=offset.dtype).scatter_reduce_(
        0, owner, offset, "amax", include_self=True
    )


def plane_sagitta(points):
    """Greatest distance from ``points`` to their best-fit plane, in world units."""
    points = points.reshape(-1, 3)
    if points.shape[0] < 3:
        return 0.0
    starts, ends = _whole_range(points)
    return float(
        _window_sagittae(
            points, starts, ends, _window_covariances(points, starts, ends)
        )[0]
    )


def subpath_bounds(corners):
    """Split ``[S, 4, 3]`` cubic segments into contiguous sub-paths.

    A segment starts a new sub-path when it does not begin where the previous one
    ended -- the rule the renderer applies to separate a glyph's outline from its
    holes (``BezierCircuitCubic._get_render_primitives``), so the two agree on
    what a sub-path is.

    Returns
    -------
    list[tuple[int, int]]
        ``(first segment, segment count)`` per sub-path, in order.
    """
    num_segments = corners.shape[0]
    if num_segments == 0:
        return []
    starts = [0]
    gaps = (corners[1:, 0, :] - corners[:-1, 3, :]).norm(p=2, dim=-1) > 1e-5
    starts.extend(int(index) + 1 for index in gaps.nonzero().reshape(-1))
    starts.append(num_segments)
    return [(starts[i], starts[i + 1] - starts[i]) for i in range(len(starts) - 1)]


def _straight_run_capacity(points, subpath_point_bounds):
    """For every segment, the longest near-straight run that may start there.

    Lengths come from a power-of-two ladder up to
    :data:`MAX_STROKE_RUN_SEGMENTS`, so the whole split costs one batched
    eigenvalue solve per rung rather than one per segment -- the greedy
    refit-per-extension this replaces spent 0.98s on a stock helix.  A run
    shorter than the exact greedy answer only means the path is cut slightly
    more often; the tolerance it is cut on is satisfied either way.

    Windows never cross a sub-path boundary: ``subpath_point_bounds`` gives each
    segment the last point of its own sub-path.
    """
    num_segments = points.shape[0] // 4
    starts = torch.arange(num_segments) * 4
    capacity = torch.ones(num_segments, dtype=torch.long)
    length = 2
    while length <= MAX_STROKE_RUN_SEGMENTS:
        ends = torch.minimum(starts + length * 4, subpath_point_bounds)
        # Only rungs that actually fit inside the sub-path can be taken.
        fits = (ends - starts) == length * 4
        ratios = _eigen_ratios(_window_covariances(points, starts, ends), 1)
        capacity = torch.where(
            fits & (ratios <= STRAIGHTNESS_TOLERANCE),
            torch.full_like(capacity, length),
            capacity,
        )
        length *= 2
    return capacity


def _runs_from_capacity(capacity, start, count):
    """Walk one sub-path's segments, taking the longest run offered at each."""
    runs = []
    offset = 0
    while offset < count:
        length = min(int(capacity[start + offset]), count - offset)
        runs.append((start + offset, max(length, 1)))
        offset += max(length, 1)
    return runs


def _subpath_corners(corners, start, count):
    """The distinct corner vertices of one closed sub-path.

    Returns ``(outgoing segment, incoming segment)`` index lists.  A corner's
    outgoing tangent is read from its outgoing segment's first handle and its
    incoming tangent from its incoming segment's last handle, so the pair spans
    the surface there.

    Coincident corners and zero-length segments are skipped rather than kept as
    degenerate vertices: a sphere's pole tile collapses one of its sides to a
    point, and a triangle with two identical corners has neither normal nor area.
    """
    points = corners[start : start + count, 0, :]
    kept = []
    for index in range(count):
        if kept and bool((points[index] - points[kept[-1]]).norm(p=2, dim=-1) <= 1e-6):
            continue
        kept.append(index)
    # The loop closes, so drop a trailing corner that has walked back onto the
    # first one.
    while len(kept) > 1 and bool(
        (points[kept[-1]] - points[kept[0]]).norm(p=2, dim=-1) <= 1e-6
    ):
        kept.pop()
    if len(kept) < 3:
        return [], []

    def outgoing(index):
        for step in range(count):
            segment = start + (index + step) % count
            if bool((corners[segment, 1] - corners[segment, 0]).norm() > 1e-9):
                return segment
        return start + index

    def incoming(index):
        for step in range(1, count + 1):
            segment = start + (index - step) % count
            if bool((corners[segment, 2] - corners[segment, 3]).norm() > 1e-9):
                return segment
        return start + (index - 1) % count

    return [outgoing(i) for i in kept], [incoming(i) for i in kept]


def classify_circuit(control_points, filled, shade_in_3d=False):
    """Decide how a circuit's control points have to be rendered.

    ``control_points`` is the construction pose, shape ``[4 * S, 3]``.  Returns
    ``None`` when every sub-path is planar, which leaves the circuit on the
    analytic bezier path with nothing changed.

    Classification is per sub-path so that a *packed* circuit -- one Mob standing
    for many shapes, as ``batch_mobs`` and ``from_batches`` build -- is judged on
    its members rather than on their union, which is non-planar the moment two
    planar members face different ways.

    ``shade_in_3d`` asks for the ``patch`` plan even where the geometry is
    planar, and is named after the Manim attribute it carries
    (``VMobject.shade_in_3d``, which ``ThreeDVMobject`` and ``Surface`` set).
    It exists because the two plans differ in more than geometry: an analytic
    circuit is drawn UNLIT, while a PN patch is ordinary 3-D geometry that
    reaches ``SETTINGS.style.default_material`` and the scene's lights. A Manim
    ``Cube`` is six *flat* ``Square`` faces with ``shade_in_3d=True``, so
    planarity alone would leave them unlit where Manim shades them. Only a
    FILLED circuit can take the patch plan -- an open path bounds no surface --
    so an unfilled one is classified on planarity as usual.
    """
    if not nonplanar_circuits_enabled():
        return None
    points = control_points.reshape(-1, 3)
    if points.shape[0] < 4 or points.shape[0] % 4:
        return None
    corners = points.detach().to(torch.float32).cpu().reshape(-1, 4, 3)
    subpaths = subpath_bounds(corners)
    if not subpaths:
        return None

    # One batched eigenvalue solve for every sub-path at once. This is the whole
    # cost that planar geometry pays -- which is to say every 2-D shape and every
    # glyph in the package -- so it is measured, not looped.
    flat = corners.reshape(-1, 3)
    subpath_starts = torch.tensor([start * 4 for start, _ in subpaths])
    subpath_ends = torch.tensor([(start + count) * 4 for start, count in subpaths])
    covariances = _window_covariances(flat, subpath_starts, subpath_ends)
    if bool((_eigen_ratios(covariances, 0) <= PLANARITY_TOLERANCE).all()) and not (
        shade_in_3d and filled
    ):
        return None

    sagitta = float(
        _window_sagittae(flat, subpath_starts, subpath_ends, covariances).max()
    )

    # Every sub-path is split, planar ones included: the runs' planes are all
    # turned toward the camera at render time, and a planar sub-path kept whole
    # would be the one thing that turn could visibly distort.
    segment_subpath = torch.repeat_interleave(
        torch.arange(len(subpaths)), torch.tensor([count for _, count in subpaths])
    )
    capacity = _straight_run_capacity(flat, subpath_ends[segment_subpath])
    runs = []
    for start, count in subpaths:
        runs.extend(_runs_from_capacity(capacity, start, count))
    run_starts = torch.tensor([run[0] for run in runs], dtype=torch.long)
    run_counts = torch.tensor([run[1] for run in runs], dtype=torch.long)

    if not filled:
        return NonPlanarPlan("stroke", run_starts, run_counts, sagitta=sagitta)

    corner_seg = []
    corner_prev = []
    corner_next = []
    corner_subpath = []
    subpath_corner_counts = []
    tri_index = []
    for start, count in subpaths:
        outgoing, incoming = _subpath_corners(corners, start, count)
        if len(outgoing) < 3:
            continue
        base = len(corner_seg)
        subpath_index = len(subpath_corner_counts)
        corner_seg.extend(outgoing)
        corner_prev.extend(incoming)
        corner_next.extend(base + (i + 1) % len(outgoing) for i in range(len(outgoing)))
        corner_subpath.extend([subpath_index] * len(outgoing))
        subpath_corner_counts.append(len(outgoing))
        # Fan from the sub-path's first corner. Every vertex is an authored
        # corner carrying its own normal, so two tiles sharing an edge derive
        # that edge's PN curve from the same two endpoints and the seam stays
        # closed however finely either side is diced.
        for i in range(1, len(outgoing) - 1):
            tri_index.extend((base, base + i, base + i + 1))
    if not tri_index:
        # Nothing fillable survived (every sub-path degenerate). Draw the outline
        # instead of dropping the Mob.
        return NonPlanarPlan("stroke", run_starts, run_counts, sagitta=sagitta)

    return NonPlanarPlan(
        "patch",
        run_starts,
        run_counts,
        corner_seg=torch.tensor(corner_seg, dtype=torch.long),
        corner_prev_seg=torch.tensor(corner_prev, dtype=torch.long),
        corner_next=torch.tensor(corner_next, dtype=torch.long),
        corner_subpath=torch.tensor(corner_subpath, dtype=torch.long),
        subpath_corner_counts=torch.tensor(subpath_corner_counts, dtype=torch.long),
        tri_index=torch.tensor(tri_index, dtype=torch.long),
        sagitta=sagitta,
    )


def patch_corner_normals(x, plan):
    """Live positions and unit normals of a ``patch``-mode circuit's corners.

    ``x`` is ``[T, S, 4, 3]``.  A corner's normal is the cross product of the two
    boundary tangents meeting there, taken in path order so it agrees with the
    sub-path's winding.  Where those tangents are parallel -- a collapsed pole
    tile, a straight-sided quad -- the sub-path's own area normal stands in, and
    every corner is flipped to agree with that area normal so a reflex corner
    cannot invert one vertex of an otherwise consistent patch.
    """
    device = x.device
    corner_seg = plan.corner_seg.to(device)
    corner_prev_seg = plan.corner_prev_seg.to(device)
    corner_next = plan.corner_next.to(device)
    corner_subpath = plan.corner_subpath.to(device)
    counts = plan.subpath_corner_counts.to(device).to(x.dtype)

    position = x[:, corner_seg, 0, :]  # [T, V, 3]
    outgoing = x[:, corner_seg, 1, :] - position
    incoming = position - x[:, corner_prev_seg, 2, :]
    raw = torch.cross(incoming, outgoing, dim=-1)

    frames = position.shape[0]
    num_subpaths = plan.num_subpaths
    totals = torch.zeros(
        frames, num_subpaths, 3, device=device, dtype=position.dtype
    ).index_add_(1, corner_subpath, position)
    centroid = totals / counts.clamp_min(1.0).view(1, -1, 1)
    relative = position - centroid[:, corner_subpath, :]
    # Newell area normal per sub-path, about its own centroid: one orientation
    # for the whole patch group.
    area = torch.zeros(
        frames, num_subpaths, 3, device=device, dtype=position.dtype
    ).index_add_(
        1, corner_subpath, torch.cross(relative, relative[:, corner_next, :], dim=-1)
    )
    area = F.normalize(area, p=2, dim=-1)[:, corner_subpath, :]

    parallel = raw.norm(p=2, dim=-1, keepdim=True) <= 1e-6 * (
        incoming.norm(p=2, dim=-1, keepdim=True)
        * outgoing.norm(p=2, dim=-1, keepdim=True)
    )
    normals = torch.where(parallel, area, raw)
    normals = torch.where((normals * area).sum(-1, keepdim=True) < 0, -normals, normals)
    return position, F.normalize(normals, p=2, dim=-1)


def _extremal_within_group(values, scores, groups, num_groups):
    """The row of ``values`` scoring highest within each group.

    Ties break toward the lowest index, for the reason a whole circuit's frame
    does (``_extremal_control_point_index``): the winner sets the basis, and an
    unspecified winner is an unspecified basis -- which is what the border width
    and the coverage filter are measured in.
    """
    frames = values.shape[0]
    limit = values.shape[1]
    spread = groups.unsqueeze(0).expand(frames, -1)
    best = torch.full(
        (frames, num_groups), -1.0, device=values.device, dtype=scores.dtype
    ).scatter_reduce_(1, spread, scores, "amax", include_self=True)
    # int32 in MPS-friendly mode, where an int64 amin ``scatter_reduce_`` is
    # unimplemented; a control-point index is bounded by ``limit``, so the
    # narrow tie-break picks the same winner.
    idx_dtype = reduction_index_dtype()
    index = torch.arange(limit, device=values.device, dtype=idx_dtype).expand_as(scores)
    chosen = torch.full(
        (frames, num_groups), limit, device=values.device, dtype=idx_dtype
    ).scatter_reduce_(
        1,
        spread,
        torch.where(scores >= best[:, groups], index, limit),
        "amin",
        include_self=True,
    )
    chosen = chosen.to(torch.long).clamp_max(limit - 1)
    return torch.gather(values, 1, chosen.unsqueeze(-1).expand(-1, -1, 3))


def run_segment_topology(plan, device):
    """``(segment order, next-segment offsets, segment run)`` for stroke runs.

    Each run renders as its own circuit, so its last segment links back to its
    own first.  Where the run is open that link is discontinuous, which is
    exactly the fill-closure the renderer already recognises: it draws the run's
    real end and suppresses the synthesized closing edge (``Line`` is the
    one-segment case of it).
    """
    run_starts = plan.run_starts.to(device)
    run_counts = plan.run_counts.to(device)
    total = int(run_counts.sum())
    segment_run = torch.repeat_interleave(
        torch.arange(run_counts.numel(), device=device), run_counts
    )
    local = torch.arange(total, device=device) - torch.repeat_interleave(
        run_counts.cumsum(0) - run_counts, run_counts
    )
    segments = torch.repeat_interleave(run_starts, run_counts) + local
    next_local = (local + 1) % run_counts[segment_run]
    return segments, next_local - local, segment_run


def run_planes(x, plan, eye=None):
    """Centre and frame of each stroke run's plane, per frame.

    ``x`` is ``[T, S, 4, 3]``; ``eye`` is the camera position ``[T, 1, 3]``.
    Returns ``(centre, first, second, normal)``, each ``[T, R, 3]``, rebuilt
    every batch so an animated path keeps a plane that follows it.

    **The plane is turned to face the camera about the run's own axis.**  A
    circuit's stroke is a band lying *in* its plane, so a plane seen edge-on
    draws a band of no width -- which is what a 3-D path does to its own
    osculating plane every time it curves toward the viewer, and it is why the
    obvious choice here (each run's best-fit plane) draws a helix as a dashed
    line.  Instead the normal is the view direction with the run's principal
    axis projected out.  That plane still *contains* the run's axis, so the path
    keeps its length and its depth along itself exactly; and the band direction
    ``cross(normal, axis)`` is then perpendicular to the view, so the stroke is
    at full width however the path is turned.  What it gives up is the run's
    extent *across* itself, which :data:`STRAIGHTNESS_TOLERANCE` is what bounds.

    Without a camera -- a primitive built outside a render loop -- each run
    falls back to its own best-fit plane.
    """
    device = x.device
    segments, _, segment_run = run_segment_topology(plan, device)
    points = x[:, segments].reshape(x.shape[0], -1, 3)  # [T, 4 * S_run, 3]
    point_run = segment_run.repeat_interleave(4)
    num_runs = plan.num_runs

    weights = (plan.run_counts.to(device) * 4).to(points.dtype).clamp_min(1.0)
    centre = torch.zeros(
        points.shape[0], num_runs, 3, device=device, dtype=points.dtype
    ).index_add_(1, point_run, points) / weights.view(1, -1, 1)

    relative = points - centre[:, point_run, :]
    first = _extremal_within_group(
        relative, relative.norm(p=2, dim=-1), point_run, num_runs
    )
    scale = first.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
    first_unit = F.normalize(first, p=2, dim=-1)
    planar = (
        relative
        - (relative * first_unit[:, point_run, :]).sum(-1, keepdim=True)
        * first_unit[:, point_run, :]
    )
    second = _extremal_within_group(
        planar, planar.norm(p=2, dim=-1), point_run, num_runs
    )
    normal = torch.cross(first_unit, F.normalize(second, p=2, dim=-1), dim=-1)

    # A run whose control points are collinear spans no plane of its own. Pick
    # any perpendicular, which is what a straight Line already does -- its stroke
    # is a band whose orientation about the path is arbitrary either way.
    fallback = F.normalize(
        torch.cross(first_unit, first_unit.roll(1, -1) + 1e-3, dim=-1), p=2, dim=-1
    )
    normal = F.normalize(
        torch.where(normal.norm(p=2, dim=-1, keepdim=True) < 1e-4, fallback, normal),
        p=2,
        dim=-1,
    )

    if eye is not None:
        view = F.normalize(centre - eye.to(centre), p=2, dim=-1)
        facing = view - (view * first_unit).sum(-1, keepdim=True) * first_unit
        # A run pointing straight at the camera leaves nothing to face it with,
        # and covers almost no pixels either way; keep its own plane there.
        normal = torch.where(
            facing.norm(p=2, dim=-1, keepdim=True) > 1e-4,
            F.normalize(facing, p=2, dim=-1),
            normal,
        )
        # A static path in front of a MOVING camera has a per-frame plane and
        # single-frame everything else. Carry the frame count across the whole
        # frame rather than leaving the primitive's rows disagreeing about how
        # long the batch is.
        if normal.shape[0] != centre.shape[0]:
            frames = normal.shape[0]
            centre = centre.expand(frames, -1, -1)
            first_unit = first_unit.expand(frames, -1, -1)
            scale = scale.expand(frames, -1, -1)

    second_unit = F.normalize(torch.cross(normal, first_unit, dim=-1), p=2, dim=-1)
    return centre, first_unit * scale, second_unit * scale, normal


def camera_eye(circuit):
    """The camera position over this render batch, ``[T, 1, 3]``, or ``None``.

    Read from the *materialized* camera state, which batch preparation sets for
    exactly this batch's frames before it collects primitives -- the same state
    ``RenderLoopMixin._materialize_render_state`` snapshots a few lines later in
    the same pass, on the same thread.  This is not the live camera: the render
    thread never sees it, so a prefetch worker preparing the next batch cannot
    hand a stroke the wrong frame's viewpoint.

    ``None`` when there is no scene or no camera -- a primitive built outside a
    render loop, which the callers fall back to plane-fitting for.
    """
    scene = getattr(circuit, "scene", None)
    camera = getattr(scene, "camera", None)
    if camera is None:
        return None
    location = getattr(camera, "location", None)
    if location is None or location.numel() % 3:
        return None
    return location.reshape(location.shape[0], -1, 3)[:, :1, :]


def member_of_segment(circuit, num_segments, device):
    """Which packed member each segment belongs to (all zeros when unpacked)."""
    sizes = circuit.control_points.parent_batch_sizes
    if sizes is None:
        return torch.zeros(num_segments, dtype=torch.long, device=device)
    counts = (sizes.to(device) // 4).long()
    return torch.repeat_interleave(torch.arange(counts.numel(), device=device), counts)


def _per_item(value, member_of, frames):
    """Broadcast a per-member attribute ``[T, M, C]`` onto a per-item axis."""
    while value.dim() < 3:
        value = value.unsqueeze(0)
    if value.shape[0] == 1 and frames > 1:
        value = value.expand(frames, -1, -1)
    if value.shape[-2] == 1:
        value = value.expand(-1, int(member_of.max()) + 1, -1)
    return value[:, member_of, :]


def _member_colors(values, num_members, channels):
    """One color per packed member, averaging a member's texture grid.

    A circuit's texture grid is laid out across its plane frame, and neither a
    patch group nor a split stroke has that frame any more, so the grid collapses
    to its mean rather than being resampled onto geometry it does not describe.
    """
    values = values.reshape(values.shape[0], num_members, -1, channels)
    return values.mean(-2)


def build_patch_primitive(circuit, x, colors, opacity, glow, shader_params):
    """The logical PN triangle primitive a filled non-planar circuit renders as.

    One PN patch per authored corner triangle, carrying the tile's own curvature
    in its corner normals: the same primitive, dice and tolerances an Algan
    :class:`~algan.mobs.surfaces.surface.Surface` produces, so a converted Manim
    tile and a native curved surface are the same thing to the renderer.
    """
    from algan.constants.color import Color
    from algan.rendering.raytracing.primitives import LogicalPNTrianglePrimitive

    plan = circuit._nonplanar_plan
    device = x.device
    frames = x.shape[0]
    tri_index = plan.tri_index.to(device)
    position, normals = patch_corner_normals(x, plan)
    members = member_of_segment(circuit, x.shape[1], device)
    vertex_member = members[plan.corner_seg.to(device)][tri_index]

    vertex_colors = _per_item(colors, vertex_member, frames).clone()
    vertex_colors[..., -1:] *= _per_item(opacity, vertex_member, frames)
    vertex_colors[..., -2:-1] += _per_item(glow, vertex_member, frames)

    primitive = LogicalPNTrianglePrimitive(
        corners=position[:, tri_index, :],
        colors=vertex_colors.as_subclass(Color),
        normals=normals[:, tri_index, :],
        glow=vertex_colors[..., -2:-1].as_subclass(torch.Tensor),
        shader=circuit.shader,
        render_tolerance_pixels=PATCH_RENDER_TOLERANCE_PIXELS,
        # The patches ARE the surface here: there is no analytic shape behind
        # them whose own accuracy could excuse a coarser dice.
        geometry_slack_ratio=0.0,
        **{
            name: _per_item(value, vertex_member, frames)
            for name, value in shader_params.items()
        },
    )
    # A converted tile is a sheet, not the shell of a solid -- a Manim Sphere is
    # 288 independent quads, not a closed orientable mesh -- so by default a
    # back-facing hit keeps the viewer-facing flip rather than being shaded as
    # an inside.
    #
    # A tile Manim itself flagged ``shade_in_3d`` is the exception, and it is
    # ONE line rather than a normal fix-up because
    # :func:`patch_corner_normals` already computes exactly Manim's winding
    # normal: both are ``cross(incoming, outgoing)`` at the corner (Manim spells
    # it ``cross(points[i+3] - points[i], points[i-3] - points[i])``, which is
    # the same product). Verified per tile against
    # ``get_3d_vmob_start_corner_unit_normal``: Cube 6/6, Cylinder 578/578,
    # Cone 1024/1024 and Torus 576/576 agree in sign, the mixed-winding Torus
    # included. So the ONLY thing standing between the two engines was this
    # flip; declaring the patch one-sided hands the authored winding straight
    # to the shader, which is the whole content of Manim's convention.
    primitive.declare_one_sided(bool(getattr(circuit, "shade_in_3d", False)))
    primitive.declare_shadow_flags(*circuit._resolved_shadow_flags())
    # One shell per sub-path: a tile's own coverage must not be summed with the
    # neighbour it merely touches.
    primitive.mesh_ids = (
        plan.corner_subpath.to(device)[tri_index].reshape(-1, 3)[:, 0].to(torch.int32)
    )
    return primitive


def _scene_point_lights(circuit):
    """This batch's point lights, as ``(origin, color)`` rows, or ``[]``.

    Read from the materialized light Mobs for the same reason
    :func:`camera_eye` reads the materialized camera: batch preparation sets
    their state for exactly this batch's frames before it collects primitives.

    ``color`` is built the way ``RenderLoopMixin._materialize_render_state``
    builds the row it packs for the kernel -- decode to the working space,
    then alpha, then opacity, then intensity, in that order, since the scalars
    do not commute through the decode. That also carries the LIFESPAN for
    free: a frame outside a light's lifespan materializes at opacity 0, so its
    row is all-zero and contributes nothing, which is the same "zero color is
    not live" gate ``RayTracedTrianglePrimitive._shade_vertex_colors`` uses.

    Extended lights (directional, spot, area, hemisphere) are skipped: they
    force the per-fragment lighting path, and the per-vertex shader convention
    every material here is written against knows point lights only.
    """
    from algan.rendering.raytracing import settings as rt_settings
    from algan.utils.color_space import srgb_to_linear

    scene = getattr(circuit, "scene", None)
    rows = []
    for light in getattr(scene, "light_sources", None) or ():
        is_extended = getattr(light, "_is_extended", None)
        if is_extended is not None and is_extended():
            continue
        origin = getattr(light, "location", None)
        rgba = getattr(light, "color", None)
        if origin is None or rgba is None:
            continue
        if rt_settings.linear_color_space:
            rgba = torch.cat((srgb_to_linear(rgba[..., :3]), rgba[..., 3:]), -1)
        color = rgba[..., :-1] * rgba[..., -1:] * light.opacity
        intensity = getattr(light, "intensity", None)
        if intensity is not None:
            color = color * intensity
        if not bool((color != 0).any()):
            continue
        rows.append((origin, color))
    return rows


def _shaded_border_colors(circuit, x, plan, border, frames):
    """A patch border's color with the patch's own shading baked into it.

    The border of a curved tile is drawn as flat stroke runs
    (:func:`build_stroke_primitive`), and the renderer draws circuits UNLIT --
    so an imported Manim ``Surface``'s grid lines came out at full
    ``LIGHT_GREY`` everywhere while its fill shaded, where Manim shades the two
    together (``ThreeDCamera.get_stroke_rgbas`` runs the stroke rgbas through
    the same ``modified_rgbas`` as the fill). Baking is the only way to shade
    them: a circuit carries no material, and the ribbon's own normal faces the
    camera, so lighting it as geometry would light every grid line head-on.

    The normal used is the TILE's, at the corner each run starts from --
    :func:`patch_corner_normals` computes exactly Manim's winding normal, and
    Manim shades a tile's stroke from its start and end corner normals. One
    color per run rather than Manim's 2-stop gradient along the tile: at the
    0.30 px a ``Surface``'s default ``stroke_width`` comes to, the difference
    is well under the width of the line it colors.

    Returns ``[T, R, C]``, one color per run, or ``None`` when there is
    nothing to bake -- no material, or no light for it to answer.
    """
    from algan.settings import SETTINGS

    shader = getattr(circuit, "shader", None)
    material_params = {}
    if shader is not None and hasattr(circuit, "get_shader_params"):
        material_params = dict(circuit.get_shader_params())
    if shader is None:
        # The same fallback the patch's own primitive makes: a mob that set no
        # material renders as the process default (``TrianglePrimitive``'s
        # ``shader is None`` branch). Resolving it here rather than reading
        # ``circuit.shader`` alone is what makes the border track the patch --
        # a converted Manim tile carries no material of its own, so the fill
        # shades through this fallback and the border has to find it too.
        material = SETTINGS.style.default_material
        shader = None if material is None else material.shader
        if material is not None:
            material_params = dict(material.get_shader_param_values())
    lights = _scene_point_lights(circuit)
    if shader is None or not lights:
        return None

    device = x.device
    position, normals = patch_corner_normals(x, plan)
    # Corner index per segment, to look up the corner each run starts at.
    corner_seg = plan.corner_seg.to(device)
    seg_to_corner = torch.zeros(
        int(x.shape[1]), dtype=torch.long, device=device
    ).index_copy_(
        0, corner_seg, torch.arange(corner_seg.numel(), device=device, dtype=torch.long)
    )
    run_corner = seg_to_corner[plan.run_starts.to(device)]
    point = position[:, run_corner, :]
    normal = normals[:, run_corner, :]

    members = member_of_segment(circuit, x.shape[1], device)
    colors = _per_item(border, members[plan.run_starts.to(device)], frames).clone()
    # The shader convention takes rgb plus the glow tail and leaves opacity
    # alone -- the same split ``_shade_vertex_colors`` makes with ``[..., :-1]``.
    albedo = colors[..., :-1]
    scene = getattr(circuit, "scene", None)
    eye = camera_eye(circuit)
    # The material's extra parameters, in the shader's own signature order --
    # the same rebuild ``RayTracedTrianglePrimitive._ordered_shader_param_values``
    # does, and for the same reason. The signature defaults are NOT a safe
    # stand-in: ``lambert_shader``'s ``emissive`` defaults to a plain tuple,
    # which the shader then multiplies by a float, so a scene on the stock
    # DiffuseMaterial raises rather than shading.
    import inspect

    from algan.rendering.raytracing.primitives import SHADER_FIXED_PARAM_COUNT

    signature = inspect.signature(shader).parameters
    params = []
    for name in list(signature.keys())[SHADER_FIXED_PARAM_COUNT:]:
        if name in material_params:
            params.append(material_params[name])
            continue
        default = signature[name].default
        params.append(default if default is not inspect._empty else 0)
    for origin, light_color in lights:
        albedo = shader(
            getattr(scene, "memory", None),
            point,
            normal,
            albedo,
            eye if eye is not None else point,
            origin.reshape(origin.shape[0], -1, 3)[:, :1, :],
            light_color.reshape(light_color.shape[0], -1, light_color.shape[-1])[
                :, :1, :
            ],
            1,
            1,
            *params,
        )
    colors[..., :-1] = albedo
    return colors


def build_stroke_primitive(
    circuit,
    x,
    colors,
    border_colors,
    opacity,
    glow,
    stroke_width,
    reflectivity,
    roughness,
    refractive_index,
    transmission,
    depth_bias=0.0,
    eye=None,
    run_colors=None,
):
    """The analytic circuit primitive a non-planar path's stroke renders as.

    One circuit per near-straight run, each with its own camera-facing plane
    (:func:`run_planes`), so the path keeps its true position in space while its
    stroke keeps the constant screen-space width the renderer gives every other
    circuit.  Runs meet at shared endpoints, so a bend between two runs shows as
    a slight angular joint rather than a gap.

    ``depth_bias`` moves every run toward the camera by that many depth-tie bins.
    The patch border path spends it to sit in front of the curved patch it
    outlines; along the view axis, so it reorders without moving anything on
    screen.
    """
    plan = circuit._nonplanar_plan
    device = x.device
    frames = x.shape[0]
    segments, next_offsets, _ = run_segment_topology(plan, device)
    centre, first, second, normal = run_planes(x, plan, eye)

    num_runs = plan.num_runs
    members = member_of_segment(circuit, x.shape[1], device)
    run_member = members[plan.run_starts.to(device)]

    def per_run(value):
        return _per_item(value, run_member, frames)

    # Already one row per run when the caller baked shading into it.
    fill_rows = per_run(colors) if run_colors is None else run_colors
    border_rows = per_run(border_colors) if run_colors is None else run_colors

    bias = float(depth_bias) + circuit._render_draw_bias()
    grid = torch.ones((1, num_runs, 1), dtype=torch.int32, device=device)
    primitive = circuit.render_primitive(
        x[:, segments],
        # [1, S, 1, 1] like the stock builder's: the merge cats on the segment
        # axis and adds an arange, and the geometry build reads the leading
        # frame axis off this tensor.
        next_offsets.view(1, -1, 1, 1),
        plan.run_counts.to(device),
        fill_rows.unsqueeze(-2),
        per_run(opacity),
        normal,
        per_run(stroke_width),
        border_rows.unsqueeze(-2),
        centre,
        grid,
        grid,
        first,
        second,
        glow=per_run(glow),
        num_texture_points=0,
        filled=False,
        reflectivity=per_run(reflectivity),
        roughness=per_run(roughness),
        refractive_index=per_run(refractive_index),
        transmission=per_run(transmission),
        z_index=(
            None
            if bias == 0.0
            else torch.full((1, num_runs, 1), bias, dtype=x.dtype, device=device)
        ),
    )
    primitive.num_texture_points = 0
    primitive.declare_shadow_flags(*circuit._resolved_shadow_flags())
    return primitive


def build_render_primitives(
    circuit,
    x,
    fill_colors,
    border_colors,
    opacity,
    glow,
    stroke_width,
    reflectivity,
    roughness,
    refractive_index,
    transmission,
):
    """Every primitive a non-planar circuit contributes to a render batch.

    A ``stroke`` circuit is one primitive.  A ``patch`` circuit is its PN patches
    plus, when its border is visible, the boundary drawn as stroke runs in front
    of them -- which is the order Manim draws a filled tile with a stroke in.
    """
    plan = circuit._nonplanar_plan
    sizes = circuit.control_points.parent_batch_sizes
    num_members = 1 if sizes is None else int(sizes.numel())
    channels = fill_colors.shape[-1]
    fill = _member_colors(fill_colors, num_members, channels)
    border = _member_colors(border_colors, num_members, channels)
    eye = camera_eye(circuit)

    if plan.mode == "stroke":
        return [
            build_stroke_primitive(
                circuit,
                x,
                border,
                border,
                opacity,
                glow,
                stroke_width,
                reflectivity,
                roughness,
                refractive_index,
                transmission,
                eye=eye,
            )
        ]

    primitives = [
        build_patch_primitive(
            circuit, x, fill, opacity, glow, circuit.get_shader_params()
        )
    ]
    if bool((stroke_width.abs() > 1e-6).any()) and bool(
        (border[..., -1:] > 1e-5).any()
    ):
        primitives.append(
            build_stroke_primitive(
                circuit,
                x,
                border,
                border,
                opacity,
                glow,
                stroke_width,
                reflectivity,
                roughness,
                refractive_index,
                transmission,
                depth_bias=plan.sagitta * BORDER_DEPTH_BIAS_BINS_PER_UNIT,
                eye=eye,
                # A patch shades and its border is a circuit, which the
                # renderer draws unlit -- so the two have to be brought
                # together on the host. Manim runs a tile's stroke through the
                # same shading as its fill, and a grid line that stays at full
                # brightness over a shaded solid is what that difference looks
                # like.
                run_colors=_shaded_border_colors(circuit, x, plan, border, x.shape[0]),
            )
        )
    return primitives
