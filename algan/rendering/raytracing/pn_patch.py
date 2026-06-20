"""Quadratic point-normal (PN) triangle patches.

A PN triangle upgrades a flat triangle (three corners with per-vertex
normals) to a curved patch without changing the mesh: each edge midpoint is
displaced so the boundary curves bend away from the flat triangle to respect
the vertex normals, giving coarsely tessellated smooth surfaces smooth
silhouettes. Here the patch is the *quadratic* Bezier (Steiner) triangle

    S(u, v) = w^2 B200 + u^2 B020 + v^2 B002
              + 2wu B110 + 2uv B011 + 2wv B101,      w = 1 - u - v

over the barycentric domain ``u, v >= 0, u + v <= 1``, with corner control
points at the vertices and one control point per edge. Vertex weights are
``(w, u, v)`` for corners ``(P0, P1, P2)``, matching the renderer's flat
triangle convention, and adjacent patches share boundary curves exactly
(each edge control point depends only on that edge's two vertices), so PN
meshes stay watertight.

The trace kernels consume the patch in monomial form

    S(u, v) = K0 + Ku u + Kv v + Kuu u^2 + Kvv v^2 + Kuv uv

which is cheaper to evaluate and differentiate per ray. This module holds
the (pure PyTorch, dependency-free) construction helpers shared by the
production primitive and the unit tests.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pn_control_points(corners, normals):
    """Control points of the quadratic PN patch of each triangle.

    Each mid-edge control point is the edge midpoint projected onto the
    tangent planes of the edge's two endpoints (and averaged) -- the
    standard quadratic point-normal construction. Normals need not be
    normalized; zero normals leave their edges straight, so a triangle with
    all-zero (or face-constant) normals stays exactly flat.

    Parameters
    ----------
    corners, normals : Tensor[..., 3 (corner), 3 (xyz)]

    Returns
    -------
    Tensor[..., 6, 3]
        Control points ordered ``(P0, P1, P2, E01, E12, E02)``.
    """
    n = F.normalize(normals.float(), p=2, dim=-1)
    p = corners.float()
    p0, p1, p2 = p[..., 0, :], p[..., 1, :], p[..., 2, :]
    n0, n1, n2 = n[..., 0, :], n[..., 1, :], n[..., 2, :]

    def edge(pi, pj, ni, nj):
        e = pj - pi
        return ((pi + pj) * 0.5
                - 0.25 * ((e * ni).sum(-1, keepdim=True) * ni
                          - (e * nj).sum(-1, keepdim=True) * nj))

    return torch.stack((p0, p1, p2, edge(p0, p1, n0, n1),
                        edge(p1, p2, n1, n2), edge(p0, p2, n0, n2)), -2)


def pn_patch_coefficients(control_points):
    """Monomial coefficients ``[K0 | Ku | Kv | Kuu | Kvv | Kuv]`` of the
    quadratic Bezier triangle, the form consumed by the trace kernels.

    Parameters
    ----------
    control_points : Tensor[..., 6, 3]
        As returned by :func:`pn_control_points`.

    Returns
    -------
    Tensor[..., 18]
    """
    p0, p1, p2, e01, e12, e02 = control_points.unbind(-2)
    return torch.cat((p0,
                      2.0 * (e01 - p0),
                      2.0 * (e02 - p0),
                      p0 + p1 - 2.0 * e01,
                      p0 + p2 - 2.0 * e02,
                      2.0 * (p0 + e12 - e01 - e02)), -1)


def pn_obb(control_points):
    """Tight oriented bounding box (OBB) of each quadratic PN patch.

    Packed as ``[center(3) | u*hu(3) | v*hv(3) | w*hw(3)]`` (12 floats): the
    three stored vectors are the orthonormal base-triangle frame axes (edge,
    in-plane, normal) scaled by their half-extents over the patch's six control
    points. The patch lies in the convex hull of those control points, which the
    box bounds, so it is a *conservative* culling volume -- a ray that misses the
    OBB provably misses the patch -- letting the trace kernel reject the vast
    majority of false-positive candidates before the (expensive) matrix-pencil
    solve. A degenerate (zero-area) triangle yields an effectively infinite box
    so it never falsely rejects. Half-extents are inflated by a hair so a hit
    exactly on a control-point extreme can't be rounded out.

    Parameters
    ----------
    control_points : Tensor[..., 6, 3]
        As returned by :func:`pn_control_points` (corners then edge points).

    Returns
    -------
    Tensor[..., 12]
    """
    cp = control_points.float()
    p0, p1, p2 = cp[..., 0, :], cp[..., 1, :], cp[..., 2, :]
    e1 = p1 - p0
    e2 = p2 - p0
    w = torch.cross(e1, e2, dim=-1)
    wn = w.norm(p=2, dim=-1, keepdim=True)
    un = e1.norm(p=2, dim=-1, keepdim=True)
    degenerate = (wn.squeeze(-1) < 1e-12) | (un.squeeze(-1) < 1e-12)
    w = w / wn.clamp_min(1e-12)
    u = e1 / un.clamp_min(1e-12)
    v = torch.cross(w, u, dim=-1)

    rel = cp - p0.unsqueeze(-2)  # [..., 6, 3]
    pu = (rel * u.unsqueeze(-2)).sum(-1)  # [..., 6]
    pv = (rel * v.unsqueeze(-2)).sum(-1)
    pw = (rel * w.unsqueeze(-2)).sum(-1)

    def axis(pc, ax):
        lo = pc.amin(-1)
        hi = pc.amax(-1)
        c = (lo + hi) * 0.5
        h = (hi - lo) * 0.5 * 1.01 + 1e-6   # hair of conservative slack
        return c.unsqueeze(-1), ax * h.unsqueeze(-1)

    cu, axu = axis(pu, u)
    cv, axv = axis(pv, v)
    cw, axw = axis(pw, w)
    center = p0 + u * cu + v * cv + w * cw
    obb = torch.cat((center, axu, axv, axw), -1)  # [..., 12]

    if bool(degenerate.any()):
        big = torch.zeros_like(obb)
        big[..., 0:3] = p0
        big[..., 3] = 1e18   # axu = (1e18, 0, 0)
        big[..., 7] = 1e18   # axv = (0, 1e18, 0)
        big[..., 11] = 1e18  # axw = (0, 0, 1e18)
        obb = torch.where(degenerate.unsqueeze(-1), big, obb)
    return obb


def evaluate_pn_patch(coefficients, u, v):
    """Evaluate patches at barycentric parameters (broadcast against the
    coefficient batch): ``S(u, v)`` for monomial rows ``[..., 18]``.
    """
    k = coefficients.unflatten(-1, (6, 3))
    u = u.unsqueeze(-1)
    v = v.unsqueeze(-1)
    return (k[..., 0, :] + u * k[..., 1, :] + v * k[..., 2, :]
            + (u * u) * k[..., 3, :] + (v * v) * k[..., 4, :]
            + (u * v) * k[..., 5, :])
