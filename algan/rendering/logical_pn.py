"""Logical point-normal triangle construction and adaptive dicing helpers.

These patches are renderer-independent geometry. A :class:`Surface` chooses
its logical PN topology once, at construction time. The render pipeline later
dices the cubic position and quadratic normal patches into ordinary flat
triangles for the materialized camera frames; no PN primitive reaches the ray
tracer or STBVH.

Dicing is *per patch*: every logical patch picks its own subdivision level in
every frame, so a patch that fills the screen does not force the rest of the
mesh (nor the same mesh in other frames) to its tessellation.  Independent
levels would crack a mesh open along the seams between patches, so the level of
each of a patch's three boundary curves is decided by a function of that curve
alone -- the two endpoints and their normals, which adjacent patches share
exactly -- and both neighbours therefore compute the same boundary
independently, with no adjacency information.  A patch's interior level is
free to exceed its boundary levels; the boundary vertices of the finer interior
grid are then snapped onto the coarser boundary polyline
(:func:`snap_boundary_values`), which keeps the shared seam identical on both
sides.  Restricting the levels to powers of two is what makes that snap exact:
the coarse polyline's knots are always vertices of the finer grid, so no
boundary segment can cut a corner off the coarse polyline.
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


def snap_boundary_values(values, level, edge_levels, boundary):
    """Pull boundary vertices onto the patch's coarser boundary polylines.

    Parameters
    ----------
    values
        Per-vertex values ``[K, V, C]`` evaluated on the true patch at
        subdivision ``level``.
    level
        The interior subdivision level the values were evaluated at.
    edge_levels
        ``[K, 3]`` boundary levels, each at most ``level``.
    boundary
        The :class:`SubdivisionBoundary` for ``level``.

    A vertex that lies on boundary curve ``e`` at parameter ``t`` is replaced by
    the point at ``t`` along the polyline through the ``2 ** edge_levels[e]``
    boundary knots -- all of which are themselves vertices of this grid, since
    both counts are powers of two.  The patch's boundary therefore becomes
    exactly that polyline, which its neighbour (whose own interior level may
    differ) reproduces vertex for vertex.  Interior vertices are untouched.
    """
    shift = int(level) - edge_levels
    if bool((shift <= 0).all()):
        return values

    num_patches, num_vertices = values.shape[0], values.shape[1]
    n = 1 << int(level)
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
    high_position = (low_position + step).clamp_max(n)
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
    """
    weights = triangle_uv_to_barycentric(triangle_uv)
    return torch.einsum("mak,pkc->pmac", weights, values)


__all__ = [
    "EDGE_CORNERS",
    "SubdivisionBoundary",
    "evaluate_cubic_curve",
    "evaluate_logical_pn",
    "evaluate_logical_pn_normals",
    "interpolate_patch_attribute",
    "logical_pn_control_points",
    "logical_pn_edge_control_points",
    "logical_pn_normal_control_points",
    "snap_boundary_values",
    "subdivision_boundary_map",
    "subdivision_triangle_indices",
    "subdivision_triangle_uvs",
    "subdivision_vertex_uvs",
    "triangle_uv_to_barycentric",
]
