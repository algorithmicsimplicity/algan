"""Logical point-normal triangle construction and uniform dicing helpers.

These patches are renderer-independent geometry. A :class:`Surface` chooses
its logical PN topology once, at construction time. The render pipeline later
dices the cubic position and quadratic normal patches into ordinary flat
triangles for the materialized camera frames; no PN primitive reaches the ray
tracer or STBVH.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_SUBDIVISION_UV_CACHE = {}


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

    def edge(pi, pj, ni):
        direction = pj - pi
        projected = (direction * ni).sum(-1, keepdim=True) * ni
        return (2.0 * pi + pj - projected) / 3.0

    b210 = edge(p0, p1, n0)
    b120 = edge(p1, p0, n1)
    b021 = edge(p1, p2, n1)
    b012 = edge(p2, p1, n2)
    b102 = edge(p2, p0, n2)
    b201 = edge(p0, p2, n0)
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


def subdivision_triangle_uvs(level, *, device, dtype):
    """Return the ``4**level`` uniform microtriangles of the unit triangle.

    The result has shape ``[M, 3, 2]`` and uses the same corner convention as
    :func:`evaluate_logical_pn`: ``(w, u, v)`` corresponds to original
    corners ``(P0, P1, P2)``.
    """
    level = int(level)
    if level < 0:
        raise ValueError("logical PN subdivision level must be non-negative")
    key = (level, device.type, device.index, dtype)
    cached = _SUBDIVISION_UV_CACHE.get(key)
    if cached is not None:
        return cached

    subdivisions = 1 << level
    triangles = []
    for i in range(subdivisions):
        for j in range(subdivisions - i):
            p00 = (i / subdivisions, j / subdivisions)
            p10 = ((i + 1) / subdivisions, j / subdivisions)
            p01 = (i / subdivisions, (j + 1) / subdivisions)
            triangles.append((p00, p10, p01))
            if i + j < subdivisions - 1:
                p11 = (
                    (i + 1) / subdivisions,
                    (j + 1) / subdivisions,
                )
                triangles.append((p10, p11, p01))

    result = torch.tensor(triangles, device=device, dtype=dtype)
    _SUBDIVISION_UV_CACHE[key] = result
    return result


def triangle_uv_to_barycentric(triangle_uv):
    """Convert ``[..., (u, v)]`` to weights for ``(P0, P1, P2)``."""
    u = triangle_uv[..., 0]
    v = triangle_uv[..., 1]
    return torch.stack((1.0 - u - v, u, v), dim=-1)


def interpolate_triangle_attribute(values, triangle_uv):
    """Interpolate a coarse per-corner attribute onto diced microtriangles.

    ``values`` is ``[T, P, 3, C]`` and the result is
    ``[T, P * M, 3, C]``.
    """
    weights = triangle_uv_to_barycentric(triangle_uv)
    interpolated = torch.einsum("mak,tpkc->tpmac", weights, values)
    return interpolated.reshape(
        values.shape[0],
        values.shape[1] * triangle_uv.shape[0],
        3,
        values.shape[-1],
    )


__all__ = [
    "evaluate_logical_pn",
    "evaluate_logical_pn_normals",
    "interpolate_triangle_attribute",
    "logical_pn_control_points",
    "logical_pn_normal_control_points",
    "subdivision_triangle_uvs",
    "triangle_uv_to_barycentric",
]
