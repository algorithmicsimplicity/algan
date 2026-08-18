"""Logical point-normal triangle construction and adaptive dicing helpers.

These patches are renderer-independent geometry. A :class:`Surface` chooses
its logical PN topology once, at construction time. The render pipeline later
dices the cubic position and quadratic normal patches into ordinary flat
triangles for the materialized camera frames; no PN primitive reaches the ray
tracer or STBVH.

Dicing is *per patch*: every logical patch picks its own subdivision in every
frame, so a patch that fills the screen does not force the rest of the mesh (nor
the same mesh in other frames) to its tessellation.  Independent dices would
crack a mesh open along the seams between patches, so the level of each of a
patch's three boundary curves is decided by a function of that curve alone --
the two endpoints and their normals, which adjacent patches share exactly -- and
both neighbours therefore compute the same boundary independently, with no
adjacency information.  A patch's own dice is free to be finer than its boundary
levels; its boundary vertices are then snapped onto the coarser boundary
polyline (:func:`snap_boundary_values`), which keeps the shared seam identical
on both sides.  Restricting the levels to powers of two is what makes that snap
exact: the coarse polyline's knots are always vertices of the finer dice, so no
boundary segment can cut a corner off the coarse polyline.

A dice is also *per direction* (:func:`dice_pattern`): ``2 ** along`` rows
fanning from one corner, each cut into at most ``2 ** across`` columns.  Equal
levels reproduce the uniform barycentric grid exactly, and a smaller ``across``
buys back the microtriangles a patch would otherwise spend resolving a direction
its surface is flat along -- a cylinder's length, a cone's slant, an extruded
profile.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

_SUBDIVISION_UV_CACHE = {}
_VERTEX_UV_CACHE = {}
_TRIANGLE_INDEX_CACHE = {}
_BOUNDARY_CACHE = {}

# Patch-local corner indices of the three boundary curves, in the order the
# subdivision grid numbers them: edge 0 runs P0 -> P1, edge 1 runs P1 -> P2 and
# edge 2 runs P0 -> P2.
EDGE_CORNERS = ((0, 1), (1, 2), (0, 2))


def logical_pn_control_points(corners, normals):
    """Return the ten control points of a standard cubic PN triangle.

    ``corners`` and ``normals`` have shape ``[..., 3, 3]``. Every edge
    control depends only on that edge's endpoints and endpoint normals, so
    adjacent logical patches with shared vertex normals have the same curved
    boundary.
    """
    p = corners.float()
    n = F.normalize(normals.float(), p=2, dim=-1)
    p0, p1, p2 = p.unbind(-2)
    n0, n1, n2 = n.unbind(-2)

    b210 = _edge_control(p0, p1, n0)
    b120 = _edge_control(p1, p0, n1)
    b021 = _edge_control(p1, p2, n1)
    b012 = _edge_control(p2, p1, n2)
    b102 = _edge_control(p2, p0, n2)
    b201 = _edge_control(p0, p2, n0)
    edge_average = (b210 + b120 + b021 + b012 + b102 + b201) / 6.0
    vertex_average = (p0 + p1 + p2) / 3.0
    b111 = edge_average + 0.5 * (edge_average - vertex_average)

    return torch.stack(
        (
            p0,
            p1,
            p2,
            b210,
            b120,
            b021,
            b012,
            b102,
            b201,
            b111,
        ),
        dim=-2,
    )


def _edge_control(pi, pj, ni):
    """The cubic control one third of the way from ``pi`` towards ``pj``,
    pulled into the tangent plane at ``pi``.
    """
    direction = pj - pi
    projected = (direction * ni).sum(-1, keepdim=True) * ni
    return (2.0 * pi + pj - projected) / 3.0


def logical_pn_normal_control_points(corners, normals):
    """Return the six control vectors of the quadratic PN normal patch."""
    p = corners.float()
    n = F.normalize(normals.float(), p=2, dim=-1)
    p0, p1, p2 = p.unbind(-2)
    n0, n1, n2 = n.unbind(-2)

    def edge(pi, pj, ni, nj):
        direction = pj - pi
        length_squared = (direction * direction).sum(-1, keepdim=True)
        safe_length_squared = length_squared.clamp_min(torch.finfo(direction.dtype).eps)
        projection = (
            2.0 * (direction * (ni + nj)).sum(-1, keepdim=True) / safe_length_squared
        )
        candidate = ni + nj - projection * direction
        fallback = ni + nj
        candidate = torch.where(
            length_squared > torch.finfo(direction.dtype).eps,
            candidate,
            fallback,
        )
        return F.normalize(candidate, p=2, dim=-1)

    return torch.stack(
        (
            n0,
            n1,
            n2,
            edge(p0, p1, n0, n1),
            edge(p1, p2, n1, n2),
            edge(p0, p2, n0, n2),
        ),
        dim=-2,
    )


def mean_patch_edge_length(corners):
    """Mean patch edge length of ``[T, P, 3, 3]`` corners, one value per frame.

    The scale a logical PN mesh's own accuracy is quoted against. A mob that has
    been scaled since construction carries patches that are bigger or smaller in
    proportion, and so is the distance from its PN patches to the surface they
    approximate; quoting that distance as a fraction of this carries it forward
    (see ``Surface._geometry_slack_ratio``).
    """
    return (corners - corners.roll(1, dims=-2)).norm(dim=-1).mean(dim=(1, 2))


def _lexicographically_greater(a, b):
    """Elementwise ``a > b`` under lexicographic order on the last axis."""
    greater = a > b
    differs = greater | (a < b)
    # ``argmax`` on the float cast picks the first differing component, and 0
    # when the keys are equal -- where ``greater`` is False, which is the
    # answer we want for equal keys.
    first = differs.float().argmax(-1, keepdim=True)
    return greater.gather(-1, first).squeeze(-1)


def logical_pn_edge_control_points(corners, normals):
    """Return each patch edge's cubic controls in a canonical orientation.

    ``corners``/``normals`` have shape ``[..., 3, 3]``; the result is
    ``[..., 3, 4, 3]``, indexed by :data:`EDGE_CORNERS`.

    The boundary curve of a logical PN patch is the cubic through the edge's two
    endpoints with the two edge controls that :func:`logical_pn_control_points`
    builds, so it depends on nothing but the endpoints and their normals -- data
    two adjacent patches share exactly. Both patches must therefore agree
    *bit for bit* on any tessellation decision taken from it, and they see the
    edge in opposite orientations. The endpoints are ordered here by a
    lexicographic comparison of their ``(position, normal)`` keys, which makes
    the control tuple, and hence every float operation downstream of it,
    orientation-independent.
    """
    p = corners.float()
    n = F.normalize(normals.float(), p=2, dim=-1)
    first = [i for i, _ in EDGE_CORNERS]
    second = [j for _, j in EDGE_CORNERS]
    pa = p[..., first, :]
    pb = p[..., second, :]
    na = n[..., first, :]
    nb = n[..., second, :]

    swap = _lexicographically_greater(
        torch.cat((pa, na), -1), torch.cat((pb, nb), -1)
    ).unsqueeze(-1)
    start = torch.where(swap, pb, pa)
    end = torch.where(swap, pa, pb)
    start_normal = torch.where(swap, nb, na)
    end_normal = torch.where(swap, na, nb)

    return torch.stack(
        (
            start,
            _edge_control(start, end, start_normal),
            _edge_control(end, start, end_normal),
            end,
        ),
        dim=-2,
    )


def evaluate_cubic_curve(control_points, t):
    """Evaluate cubic curves ``[K, 4, 3]`` at parameters ``t``.

    Returns ``[K, *t.shape, 3]``.  Accumulated term by term rather than as a
    weighted sum over a stacked control axis: the level searches call this on
    every chord of every patch edge, where the four-wide intermediate is the
    whole cost.
    """
    controls = control_points.view(control_points.shape[0], *((1,) * t.ndim), 4, 3)
    s = 1.0 - t
    weights = (s * s * s, 3.0 * s * s * t, 3.0 * s * t * t, t * t * t)
    total = None
    for index, weight in enumerate(weights):
        term = weight.unsqueeze(0).unsqueeze(-1) * controls[..., index, :]
        total = term if total is None else total + term
    return total


def evaluate_logical_pn(control_points, uv):
    """Evaluate cubic logical PN position patches at coordinates ``uv``.

    Parameters
    ----------
    control_points
        Tensor ``[T, P, 10, 3]``.
    uv
        Tensor ``[..., 2]`` in the barycentric ``(u, v)`` domain.

    Returns
    -------
    Tensor
        Shape ``[T, P, *uv.shape[:-1], 3]``.
    """
    extra_dims = uv.ndim - 1
    controls = control_points.view(
        *control_points.shape[:-2],
        *((1,) * extra_dims),
        10,
        3,
    )
    uv_shape = (1,) * (control_points.ndim - 2) + uv.shape[:-1] + (1,)
    u = uv[..., 0].view(*uv_shape)
    v = uv[..., 1].view(*uv_shape)
    w = 1.0 - u - v
    return (
        (w * w * w) * controls[..., 0, :]
        + (u * u * u) * controls[..., 1, :]
        + (v * v * v) * controls[..., 2, :]
        + (3.0 * w * w * u) * controls[..., 3, :]
        + (3.0 * w * u * u) * controls[..., 4, :]
        + (3.0 * u * u * v) * controls[..., 5, :]
        + (3.0 * u * v * v) * controls[..., 6, :]
        + (3.0 * w * v * v) * controls[..., 7, :]
        + (3.0 * w * w * v) * controls[..., 8, :]
        + (6.0 * w * u * v) * controls[..., 9, :]
    )


def evaluate_logical_pn_per_patch(control_points, uv):
    """Evaluate cubic logical PN position patches, one ``uv`` per patch.

    :func:`evaluate_logical_pn` evaluates every patch at every coordinate,
    which is what dicing wants: one subdivision pattern shared by the whole
    mesh. This instead pairs each patch with its own coordinate, for callers
    that already know which patch a point falls in -- looking up where a texel
    sits on the surface, say.

    Parameters
    ----------
    control_points
        Tensor ``[..., 10, 3]``.
    uv
        Tensor ``[..., 2]`` in the barycentric ``(u, v)`` domain, broadcasting
        against ``control_points``' leading dimensions.

    Returns
    -------
    Tensor
        Shape ``[..., 3]``.
    """
    u = uv[..., :1]
    v = uv[..., 1:]
    w = 1.0 - u - v
    return (
        (w * w * w) * control_points[..., 0, :]
        + (u * u * u) * control_points[..., 1, :]
        + (v * v * v) * control_points[..., 2, :]
        + (3.0 * w * w * u) * control_points[..., 3, :]
        + (3.0 * w * u * u) * control_points[..., 4, :]
        + (3.0 * u * u * v) * control_points[..., 5, :]
        + (3.0 * u * v * v) * control_points[..., 6, :]
        + (3.0 * w * v * v) * control_points[..., 7, :]
        + (3.0 * w * w * v) * control_points[..., 8, :]
        + (6.0 * w * u * v) * control_points[..., 9, :]
    )


def evaluate_logical_pn_normals(control_points, uv):
    """Evaluate and normalize quadratic logical PN normal patches."""
    extra_dims = uv.ndim - 1
    controls = control_points.view(
        *control_points.shape[:-2],
        *((1,) * extra_dims),
        6,
        3,
    )
    uv_shape = (1,) * (control_points.ndim - 2) + uv.shape[:-1] + (1,)
    u = uv[..., 0].view(*uv_shape)
    v = uv[..., 1].view(*uv_shape)
    w = 1.0 - u - v
    normals = (
        (w * w) * controls[..., 0, :]
        + (u * u) * controls[..., 1, :]
        + (v * v) * controls[..., 2, :]
        + (2.0 * w * u) * controls[..., 3, :]
        + (2.0 * u * v) * controls[..., 4, :]
        + (2.0 * w * v) * controls[..., 5, :]
    )
    return F.normalize(normals, p=2, dim=-1)


def _vertex_id_table(level, device):
    """``[n + 1, n + 1]`` map from grid coordinates ``(i, j)`` to vertex id.

    Only entries with ``i + j <= n`` are meaningful; the vertices are numbered
    row-major in ``i``, which is the order :func:`subdivision_vertex_uvs`
    produces.
    """
    n = 1 << level
    steps = torch.arange(n + 1, device=device)
    row_offsets = steps * (n + 1) - (steps * (steps - 1)) // 2
    return row_offsets.view(-1, 1) + steps.view(1, -1)


def _grid_coordinates(level, device):
    """Integer ``(i, j)`` grid coordinates of every subdivision vertex."""
    n = 1 << level
    steps = torch.arange(n + 1, device=device)
    i = steps.view(-1, 1).expand(n + 1, n + 1)
    j = steps.view(1, -1).expand(n + 1, n + 1)
    inside = (i + j) <= n
    return i[inside], j[inside]


def subdivision_vertex_uvs(level, *, device, dtype):
    """Return the shared vertices of the uniform level-``level`` dicing.

    Shape ``[V, 2]`` with ``V = (n + 1)(n + 2) / 2`` for ``n = 2 ** level``,
    in the same ``(u, v)`` domain as :func:`evaluate_logical_pn`.  Evaluating
    the patch on these shared vertices, rather than once per microtriangle
    corner, is what makes an adaptive dice affordable: each vertex is visited
    by up to six microtriangles.
    """
    level = _checked_level(level)
    key = (level, device.type, device.index, dtype)
    cached = _VERTEX_UV_CACHE.get(key)
    if cached is not None:
        return cached
    n = 1 << level
    i, j = _grid_coordinates(level, device)
    result = torch.stack((i.to(dtype), j.to(dtype)), dim=-1) / n
    _VERTEX_UV_CACHE[key] = result
    return result


def subdivision_triangle_indices(level, *, device):
    """Return the ``[4 ** level, 3]`` microtriangle index buffer.

    Indices refer to :func:`subdivision_vertex_uvs`.  Upward-pointing
    microtriangles come first, then the downward-pointing ones.
    """
    level = _checked_level(level)
    key = (level, device.type, device.index)
    cached = _TRIANGLE_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    n = 1 << level
    table = _vertex_id_table(level, device)
    steps = torch.arange(n, device=device)
    i = steps.view(-1, 1).expand(n, n)
    j = steps.view(1, -1).expand(n, n)
    upward = (i + j) < n
    downward = (i + j) < (n - 1)
    result = torch.cat(
        (
            torch.stack((table[i, j], table[i + 1, j], table[i, j + 1]), dim=-1)[
                upward
            ],
            torch.stack(
                (table[i + 1, j], table[i + 1, j + 1], table[i, j + 1]), dim=-1
            )[downward],
        ),
        dim=0,
    )
    _TRIANGLE_INDEX_CACHE[key] = result
    return result


class SubdivisionBoundary(NamedTuple):
    """Where each subdivision vertex sits on the patch boundary.

    ``edge_of_vertex`` and ``index_on_edge`` are meaningless (and unused)
    wherever ``is_interior`` is set.  Patch corners belong to two edges; each is
    assigned to one of them, which is harmless because a corner is a knot of
    both at every level.
    """

    edge_of_vertex: torch.Tensor  # [V] long, 0..2
    index_on_edge: torch.Tensor  # [V] long, 0..2 ** level
    is_interior: torch.Tensor  # [V] bool
    edge_vertex_ids: torch.Tensor  # [3, 2 ** level + 1] long


def subdivision_boundary_map(level, *, device):
    """Return the :class:`SubdivisionBoundary` of a level-``level`` dicing."""
    level = _checked_level(level)
    key = (level, device.type, device.index)
    cached = _BOUNDARY_CACHE.get(key)
    if cached is not None:
        return cached

    n = 1 << level
    i, j = _grid_coordinates(level, device)
    edge_of_vertex = torch.zeros_like(i)
    index_on_edge = torch.zeros_like(i)
    is_interior = torch.ones_like(i, dtype=torch.bool)

    on_first = j == 0
    on_third = (~on_first) & (i == 0)
    on_second = (~on_first) & (~on_third) & ((i + j) == n)
    for edge, mask, coordinate in (
        (0, on_first, i),
        (1, on_second, j),
        (2, on_third, j),
    ):
        edge_of_vertex = torch.where(mask, torch.full_like(i, edge), edge_of_vertex)
        index_on_edge = torch.where(mask, coordinate, index_on_edge)
        is_interior = is_interior & ~mask

    table = _vertex_id_table(level, device)
    steps = torch.arange(n + 1, device=device)
    zeros = torch.zeros_like(steps)
    edge_vertex_ids = torch.stack(
        (
            table[steps, zeros],
            table[n - steps, steps],
            table[zeros, steps],
        ),
        dim=0,
    )

    result = SubdivisionBoundary(
        edge_of_vertex, index_on_edge, is_interior, edge_vertex_ids
    )
    _BOUNDARY_CACHE[key] = result
    return result


def snap_boundary_values(values, pattern_edge_levels, edge_levels, boundary):
    """Pull boundary vertices onto the patch's coarser boundary polylines.

    Parameters
    ----------
    values
        Per-vertex values ``[K, V, C]`` evaluated on the true patch at the
        dicing ``boundary`` describes.
    pattern_edge_levels
        The dice's own level on each of the three edges, ``[3]`` -- ``(a, a, b)``
        permuted for the pattern's apex. A single level stands for all three, as
        a uniform dice has.
    edge_levels
        ``[K, 3]`` boundary levels, each at most the dice's level on that edge.
    boundary
        The :class:`SubdivisionBoundary` of the dice.

    A vertex that lies on boundary curve ``e`` at parameter ``t`` is replaced by
    the point at ``t`` along the polyline through the ``2 ** edge_levels[e]``
    boundary knots -- all of which are themselves vertices of this grid, since
    both counts are powers of two and the dice's knots along any one edge are
    evenly spaced in that edge's parameter.  The patch's boundary therefore
    becomes exactly that polyline, which its neighbour (whose own dice may
    differ in both directions) reproduces vertex for vertex.  Interior vertices
    are untouched.
    """
    pattern_edge_levels = (
        torch.as_tensor(
            pattern_edge_levels, dtype=edge_levels.dtype, device=edge_levels.device
        )
        .reshape(-1)
        .expand(3)
    )
    shift = pattern_edge_levels.view(1, 3) - edge_levels
    if bool((shift <= 0).all()):
        return values

    num_patches, num_vertices = values.shape[0], values.shape[1]
    edge_counts = torch.bitwise_left_shift(
        torch.ones_like(pattern_edge_levels), pattern_edge_levels
    )
    edges = boundary.edge_of_vertex.view(1, num_vertices).expand(
        num_patches, num_vertices
    )
    positions = boundary.index_on_edge.view(1, num_vertices).expand(
        num_patches, num_vertices
    )
    vertex_shift = shift.clamp_min(0).gather(1, edges)
    step = torch.bitwise_left_shift(torch.ones_like(vertex_shift), vertex_shift)
    low_position = torch.bitwise_left_shift(
        torch.bitwise_right_shift(positions, vertex_shift), vertex_shift
    )
    # Each edge stops at its own knot count: the two edges the dice's rows run
    # between carry ``2 ** along`` knots, the third ``2 ** across``.
    high_position = torch.minimum(low_position + step, edge_counts[edges])
    blend = (positions - low_position).to(values.dtype) / step.to(values.dtype)

    identity = (
        torch.arange(num_vertices, device=values.device)
        .view(1, num_vertices)
        .expand(num_patches, num_vertices)
    )
    interior = boundary.is_interior.view(1, num_vertices)
    low = torch.where(interior, identity, boundary.edge_vertex_ids[edges, low_position])
    high = torch.where(
        interior, identity, boundary.edge_vertex_ids[edges, high_position]
    )
    blend = torch.where(interior, torch.zeros_like(blend), blend).unsqueeze(-1)

    channels = values.shape[-1]
    low_values = values.gather(1, low.unsqueeze(-1).expand(-1, -1, channels))
    high_values = values.gather(1, high.unsqueeze(-1).expand(-1, -1, channels))
    return low_values + (high_values - low_values) * blend


def subdivision_triangle_uvs(level, *, device, dtype):
    """Return the ``4 ** level`` uniform microtriangles of the unit triangle.

    The result has shape ``[M, 3, 2]`` and uses the same corner convention as
    :func:`evaluate_logical_pn`: ``(w, u, v)`` corresponds to original
    corners ``(P0, P1, P2)``.  This is the per-corner form of
    :func:`subdivision_vertex_uvs` gathered through
    :func:`subdivision_triangle_indices`, so the three share a triangle order.
    """
    level = _checked_level(level)
    key = (level, device.type, device.index, dtype)
    cached = _SUBDIVISION_UV_CACHE.get(key)
    if cached is not None:
        return cached
    result = subdivision_vertex_uvs(level, device=device, dtype=dtype)[
        subdivision_triangle_indices(level, device=device)
    ]
    _SUBDIVISION_UV_CACHE[key] = result
    return result


def _checked_level(level):
    level = int(level)
    if level < 0:
        raise ValueError("logical PN subdivision level must be non-negative")
    return level


# Which patch edge sits opposite each corner, in :data:`EDGE_CORNERS` numbering.
# A dice fans its rows from one corner, so the edge opposite that corner is the
# one the rows run parallel to -- the direction the dice may be coarse in.
OPPOSITE_EDGE = (1, 2, 0)


class DicePattern(NamedTuple):
    """One patch's dicing: ``2 ** along`` rows, up to ``2 ** across`` columns.

    The rows fan out from corner ``apex`` and run parallel to the edge opposite
    it, so the two edges that meet at the apex are each cut into ``2 ** along``
    segments and the opposite edge into ``2 ** across``. Row ``i`` carries
    ``ceil(i * 2 ** across / 2 ** along)`` columns, which is what makes the
    columns thin out towards the apex, and makes ``along == across`` reproduce
    the uniform barycentric grid exactly.

    ``edge_levels`` states the dice's own level on each of the three edges, in
    :data:`EDGE_CORNERS` numbering, which is what :func:`snap_boundary_values`
    needs to pull each boundary back onto its shared polyline.
    """

    along: int
    across: int
    apex: int
    vertex_uv: torch.Tensor  # [V, 2]
    triangle_indices: torch.Tensor  # [M, 3]
    boundary: SubdivisionBoundary
    edge_levels: tuple  # per edge, the dice's own level

    @property
    def triangle_count(self):
        return self.triangle_indices.shape[0]


_DICE_PATTERN_CACHE = {}


def dice_triangle_count(along, across):
    """Microtriangles in a ``(along, across)`` dice: ``n * (m + 1) - m``.

    Reduces to ``n ** 2 == 4 ** along`` when the two levels are equal, and to
    ``2 * n - 1`` when the dice is one column wide.  Accepts tensors, so the
    packing arithmetic can size a whole batch at once.
    """
    n = torch.bitwise_left_shift(torch.ones_like(along), along)
    m = torch.bitwise_left_shift(torch.ones_like(across), across)
    return n * (m + 1) - m


def _row_column_counts(along, across):
    """Columns in each of the ``2 ** along + 1`` rows, coarsest row first."""
    n, m = 1 << along, 1 << across
    steps = torch.arange(n + 1)
    return (-torch.div(-steps * m, n, rounding_mode="floor")).tolist()  # ceil


def dice_pattern(along, across, apex, *, device, dtype):
    """Return the cached :class:`DicePattern` for one dice shape.

    ``along == across`` returns the uniform barycentric grid (ignoring ``apex``,
    which it has no use for), so a patch whose two directions want the same
    detail dices exactly as it always has.
    """
    along = _checked_level(along)
    across = _checked_level(across)
    if across > along:
        raise ValueError("logical PN dice across level cannot exceed the along level")
    apex = 0 if along == across else int(apex)
    if not 0 <= apex <= 2:
        raise ValueError("logical PN dice apex must name a patch corner (0, 1 or 2)")
    key = (along, across, apex, device.type, device.index, dtype)
    cached = _DICE_PATTERN_CACHE.get(key)
    if cached is not None:
        return cached

    if along == across:
        pattern = DicePattern(
            along,
            across,
            apex,
            subdivision_vertex_uvs(along, device=device, dtype=dtype),
            subdivision_triangle_indices(along, device=device),
            subdivision_boundary_map(along, device=device),
            (along, along, along),
        )
    else:
        pattern = _build_anisotropic_pattern(along, across, apex, device, dtype)
    _DICE_PATTERN_CACHE[key] = pattern
    return pattern


def _build_anisotropic_pattern(along, across, apex, device, dtype):
    n = 1 << along
    counts = _row_column_counts(along, across)
    # ``others`` in this order keeps every microtriangle wound the same way as
    # the uniform grid's: (apex, others[0], others[1]) is a rotation of
    # (P0, P1, P2), so walking a row in the +t direction turns the same way
    # around the patch whichever corner the rows fan from.
    others = ((apex + 1) % 3, (apex + 2) % 3)
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count + 1)
    num_vertices = offsets[-1]

    weights = torch.zeros((num_vertices, 3), dtype=torch.float64)
    edge_of_vertex = torch.zeros(num_vertices, dtype=torch.long)
    index_on_edge = torch.zeros(num_vertices, dtype=torch.long)
    is_interior = torch.ones(num_vertices, dtype=torch.bool)
    triangles = []

    apex_edges = tuple(_edge_of_corners(apex, other) for other in others)
    opposite_edge = OPPOSITE_EDGE[apex]

    for i, count in enumerate(counts):
        start = offsets[i]
        rows = torch.arange(count + 1, dtype=torch.float64)
        s = i / n
        t = rows / count if count else rows
        weights[start : start + count + 1, apex] = 1.0 - s
        weights[start : start + count + 1, others[0]] = s * (1.0 - t)
        weights[start : start + count + 1, others[1]] = s * t
        # The two edges meeting at the apex carry one knot per row; the edge
        # opposite it carries the whole of the last row. A corner sits on two
        # edges and is claimed by one, which is safe because it is a knot of
        # both at every level.
        if i == n:
            span = torch.arange(count + 1)
            edge_of_vertex[start : start + count + 1] = opposite_edge
            index_on_edge[start : start + count + 1] = _edge_index(
                span, count, others[0], others[1]
            )
            is_interior[start : start + count + 1] = False
        else:
            for column, (other, edge) in zip((0, count), zip(others, apex_edges)):
                edge_of_vertex[start + column] = edge
                index_on_edge[start + column] = _edge_index(
                    torch.tensor(i), n, apex, other
                )
                is_interior[start + column] = False
        if i:
            triangles.append(
                _strip_triangles(offsets[i - 1], counts[i - 1], start, count)
            )

    edge_vertex_ids = _edge_vertex_ids(
        counts, offsets, n, apex, others, apex_edges, opposite_edge
    )
    uv = (weights[:, 1:] / weights.sum(-1, keepdim=True).clamp_min(1e-30)).to(dtype)
    return DicePattern(
        along,
        across,
        apex,
        uv.to(device),
        torch.cat(triangles).to(device),
        SubdivisionBoundary(
            edge_of_vertex.to(device),
            index_on_edge.to(device),
            is_interior.to(device),
            edge_vertex_ids.to(device),
        ),
        tuple(across if edge == opposite_edge else along for edge in range(3)),
    )


def _edge_of_corners(first, second):
    """The :data:`EDGE_CORNERS` index of the edge joining two corners."""
    pair = (min(first, second), max(first, second))
    return EDGE_CORNERS.index(pair)


def _edge_index(position, count, from_corner, to_corner):
    """Renumber a knot counted from ``from_corner`` into edge-canonical order.

    :data:`EDGE_CORNERS` lists every edge low corner first, and
    :func:`snap_boundary_values` indexes knots in that direction, so a row that
    walks the edge the other way counts down instead of up.
    """
    return position if from_corner < to_corner else count - position


def _strip_triangles(lower_start, lower_count, upper_start, upper_count):
    """Triangulate the band between two rows of ``lower/upper_count`` segments.

    The two rows generally carry different numbers of knots, so the band is a
    triangle strip rather than a row of quads: at each step whichever side is
    behind in normalized position advances, which is the same merge a uniform
    grid's row performs when its two sides differ by exactly one.

    The comparison is done on integer cross-products rather than on the
    normalized positions themselves, so ties are exact whatever the counts are.
    """
    lower_keys = torch.arange(1, lower_count + 1) * upper_count
    upper_keys = torch.arange(1, upper_count + 1) * lower_count
    keys = torch.cat((lower_keys, upper_keys))
    # Stable sort with the lower row's keys first in the concatenation: a tie
    # advances the lower row, which is what reproduces the uniform grid.
    order = torch.argsort(keys, stable=True)
    took_lower = order < lower_count
    steps = took_lower.long()
    lower_before = torch.cumsum(steps, 0) - steps
    upper_before = torch.cumsum(1 - steps, 0) - (1 - steps)
    lower_ids = lower_start + lower_before
    upper_ids = upper_start + upper_before
    third = torch.where(
        took_lower, lower_start + lower_before + 1, upper_start + upper_before + 1
    )
    return torch.stack((lower_ids, upper_ids, third), dim=-1)


def _edge_vertex_ids(counts, offsets, n, apex, others, apex_edges, opposite_edge):
    """``[3, max knots + 1]`` vertex ids along each edge, canonically ordered.

    Rows shorter than the widest edge repeat their last id; those entries are
    never read, because :func:`snap_boundary_values` clamps each edge's
    positions to that edge's own knot count.
    """
    width = max(n, counts[-1]) + 1
    ids = torch.zeros((3, width), dtype=torch.long)
    for other, edge, column in zip(others, apex_edges, (0, -1)):
        along_edge = torch.tensor(
            [offsets[i] + (0 if column == 0 else counts[i]) for i in range(n + 1)]
        )
        if apex > other:
            along_edge = along_edge.flip(0)
        ids[edge, : n + 1] = along_edge
        ids[edge, n + 1 :] = along_edge[-1]
    last = torch.arange(counts[-1] + 1) + offsets[n]
    if others[0] > others[1]:
        last = last.flip(0)
    ids[opposite_edge, : counts[-1] + 1] = last
    ids[opposite_edge, counts[-1] + 1 :] = last[-1]
    return ids


def triangle_uv_to_barycentric(triangle_uv):
    """Convert ``[..., (u, v)]`` to weights for ``(P0, P1, P2)``."""
    u = triangle_uv[..., 0]
    v = triangle_uv[..., 1]
    return torch.stack((1.0 - u - v, u, v), dim=-1)


def interpolate_patch_attribute(values, triangle_uv):
    """Interpolate per-corner patch attributes onto diced microtriangles.

    ``values`` is ``[K, 3, C]`` (one row per selected patch) and
    ``triangle_uv`` is ``[M, 3, 2]``; the result is ``[K, M, 3, C]``.

    Attributes are linear in the barycentric coordinates, so this needs no
    boundary snapping: along a shared edge the interpolant depends only on the
    two endpoint values, which both patches hold in common.

    The dice itself calls :func:`interpolate_patch_vertex_attribute`, which
    evaluates a sixth as many points and gathers.  This is kept as the
    definition that one is stated against: the equivalence test and the dice's
    A/B reference arm (``benchmarks/_pn_dice_ab.py``) both compare to it, so it
    is not dead code.
    """
    weights = triangle_uv_to_barycentric(triangle_uv)
    return torch.einsum("mak,pkc->pmac", weights, values)


def interpolate_patch_vertex_attribute(values, vertex_uv):
    """Interpolate per-corner patch attributes onto the *shared* dice vertices.

    ``values`` is ``[K, 3, C]`` and ``vertex_uv`` is ``[V, 2]``; the result is
    ``[K, V, C]``.  Gathering that through :func:`subdivision_triangle_indices`
    reproduces :func:`interpolate_patch_attribute` exactly -- a microtriangle's
    corners *are* these vertices, and :func:`subdivision_triangle_uvs` is
    literally this vertex list gathered through those indices -- for a sixth of
    the arithmetic, because a vertex is a corner of up to six microtriangles.
    It is the attribute counterpart of what
    :func:`subdivision_vertex_uvs` already does for positions.
    """
    weights = triangle_uv_to_barycentric(vertex_uv)
    return torch.einsum("vk,pkc->pvc", weights, values)


__all__ = [
    "EDGE_CORNERS",
    "OPPOSITE_EDGE",
    "DicePattern",
    "SubdivisionBoundary",
    "dice_pattern",
    "dice_triangle_count",
    "evaluate_cubic_curve",
    "evaluate_logical_pn",
    "evaluate_logical_pn_normals",
    "interpolate_patch_attribute",
    "interpolate_patch_vertex_attribute",
    "logical_pn_control_points",
    "logical_pn_edge_control_points",
    "logical_pn_normal_control_points",
    "mean_patch_edge_length",
    "snap_boundary_values",
    "subdivision_boundary_map",
    "subdivision_triangle_indices",
    "subdivision_triangle_uvs",
    "subdivision_vertex_uvs",
    "triangle_uv_to_barycentric",
]
