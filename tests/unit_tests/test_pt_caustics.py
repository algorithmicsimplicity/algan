"""Manifold next-event estimation: the derivatives, the walk and the
generalized geometry term (``pt_caustics_taichi``), checked against finite
differences and against closed forms for flat interfaces.
"""

import math

import numpy as np
import pytest
import torch

from algan.rendering.raytracing.pt_caustics_taichi import (
    MNEE_MAX_VERTICES,
    _pt_mnee_admissible,
    _pt_mnee_barycentric,
    _pt_mnee_constraint,
    _pt_mnee_geometry,
    _pt_mnee_solve,
    _pt_mnee_vertex,
)
from algan.rendering.raytracing.transport_taichi import _offset_ray_origin
from algan.rendering.taichi_runtime import init_taichi
from algan.taichi_compat import ti


@pytest.fixture(scope="module", autouse=True)
def _taichi():
    init_taichi()


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


#: A curved facet: its corner normals disagree, so the shading normal turns
#: across it and the B block's frame terms are exercised.
_CURVED_POS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [0.0, 1.0, -0.05]]
_CURVED_NRM = [_unit([0.0, 0.0, 1.0]), _unit([0.2, 0.0, 1.0]), _unit([0.0, -0.3, 1.0])]


def _tri_tables(positions, normals):
    pos = torch.tensor(np.asarray(positions, dtype=np.float32).reshape(1, -1, 9))
    nrm = torch.tensor(np.asarray(normals, dtype=np.float32).reshape(1, -1, 9))
    return pos, nrm


@ti.kernel
def _vertex_probe(
    tri_pos: ti.types.ndarray(),
    tri_norm: ti.types.ndarray(),
    a: ti.f32,
    b: ti.f32,
    flat: ti.f32,
    eps: ti.f32,
    out: ti.types.ndarray(),
):
    for _ in range(1):
        p, n, ng, du, dv, dndu, dndv, valid = _pt_mnee_vertex(
            0, 0, a, b, flat, tri_pos, tri_norm
        )
        # Central differences of the normal along the frame, through the
        # barycentrics of the displaced points.
        fd = ti.Matrix.zero(ti.f32, 2, 3)
        for k in ti.static(range(2)):
            e = du if k == 0 else dv
            ap, bp, _ip = _pt_mnee_barycentric(0, 0, p + e * eps, tri_pos)
            am, bm, _im = _pt_mnee_barycentric(0, 0, p - e * eps, tri_pos)
            _pp, n_p, _gp, _up, _vp, _dup, _dvp, _okp = _pt_mnee_vertex(
                0, 0, ap, bp, flat, tri_pos, tri_norm
            )
            _pm, n_m, _gm, _um, _vm, _dum, _dvm, _okm = _pt_mnee_vertex(
                0, 0, am, bm, flat, tri_pos, tri_norm
            )
            d = (n_p - n_m) / (2.0 * eps)
            for c in ti.static(range(3)):
                fd[k, c] = d[c]
        for c in ti.static(range(3)):
            out[0, c] = dndu[c]
            out[1, c] = dndv[c]
            out[2, c] = fd[0, c]
            out[3, c] = fd[1, c]
            out[4, c] = n[c]
            out[5, c] = du[c]
            out[6, c] = dv[c]
            out[7, c] = ng[c]
        out[8, 0] = ti.cast(valid, ti.f32)


@pytest.mark.parametrize("flat", [0.0, 0.4])
def test_vertex_normal_derivatives_match_finite_differences(flat):
    pos, nrm = _tri_tables(_CURVED_POS, _CURVED_NRM)
    out = torch.zeros((9, 3), dtype=torch.float32)
    _vertex_probe(pos, nrm, 0.3, 0.25, flat, 1e-3, out)
    out = out.numpy().astype(np.float64)
    assert out[8, 0] == 1.0
    np.testing.assert_allclose(out[0], out[2], atol=2e-3)
    np.testing.assert_allclose(out[1], out[3], atol=2e-3)
    # An orthonormal frame in the facet's plane, the normal on its front.
    du, dv, ng, n = out[5], out[6], out[7], out[4]
    assert abs(np.dot(du, dv)) < 1e-6
    assert abs(np.linalg.norm(du) - 1.0) < 1e-6
    assert abs(np.dot(du, ng)) < 1e-6
    assert abs(np.dot(dv, ng)) < 1e-6
    assert np.dot(n, ng) > 0.0


@ti.kernel
def _constraint_probe(
    tri_pos: ti.types.ndarray(),
    tri_norm: ti.types.ndarray(),
    args: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    for _ in range(1):
        a = args[0]
        b = args[1]
        flat = args[2]
        ior = args[3]
        fixed = ti.cast(args[4], ti.i32)
        xp = ti.math.vec3(args[5], args[6], args[7])
        xn = ti.math.vec3(args[8], args[9], args[10])
        eps = args[11]
        ep_u = ti.math.vec3(args[12], args[13], args[14])
        ep_v = ti.math.vec3(args[15], args[16], args[17])
        en_u = ti.math.vec3(args[18], args[19], args[20])
        en_v = ti.math.vec3(args[21], args[22], args[23])
        p, n, ng, du, dv, dndu, dndv, _ok = _pt_mnee_vertex(
            0, 0, a, b, flat, tri_pos, tri_norm
        )
        c, a_blk, b_blk, c_blk, eta, valid = _pt_mnee_constraint(
            xp, p, xn, fixed, n, ng, du, dv, dndu, dndv, ior, ep_u, ep_v, en_u, en_v
        )
        for k in ti.static(range(4)):
            out[0, k] = a_blk[k]
            out[1, k] = b_blk[k]
            out[2, k] = c_blk[k]
        out[6, 0] = c[0]
        out[6, 1] = c[1]
        out[6, 2] = eta
        out[6, 3] = ti.cast(valid, ti.f32)
        # Finite differences, one column (u, then v) at a time.
        for k in ti.static(range(2)):
            # A: the previous point moves.
            e_p = ep_u if k == 0 else ep_v
            c_p, _a1, _b1, _c1, _e1, _v1 = _pt_mnee_constraint(
                xp + e_p * eps,
                p,
                xn,
                fixed,
                n,
                ng,
                du,
                dv,
                dndu,
                dndv,
                ior,
                ep_u,
                ep_v,
                en_u,
                en_v,
            )
            c_m, _a2, _b2, _c2, _e2, _v2 = _pt_mnee_constraint(
                xp - e_p * eps,
                p,
                xn,
                fixed,
                n,
                ng,
                du,
                dv,
                dndu,
                dndv,
                ior,
                ep_u,
                ep_v,
                en_u,
                en_v,
            )
            d = (c_p - c_m) / (2.0 * eps)
            out[3, k] = d[0]
            out[3, 2 + k] = d[1]
            # B: this vertex slides across its facet (normal and frame too).
            e = du if k == 0 else dv
            ap, bp, _ip = _pt_mnee_barycentric(0, 0, p + e * eps, tri_pos)
            am, bm, _im = _pt_mnee_barycentric(0, 0, p - e * eps, tri_pos)
            pp, n_p, g_p, u_p, v_p, du_p, dv_p, _okp = _pt_mnee_vertex(
                0, 0, ap, bp, flat, tri_pos, tri_norm
            )
            pm, n_m, g_m, u_m, v_m, du_m, dv_m, _okm = _pt_mnee_vertex(
                0, 0, am, bm, flat, tri_pos, tri_norm
            )
            c_bp, _a3, _b3, _c3, _e3, _v3 = _pt_mnee_constraint(
                xp,
                pp,
                xn,
                fixed,
                n_p,
                g_p,
                u_p,
                v_p,
                du_p,
                dv_p,
                ior,
                ep_u,
                ep_v,
                en_u,
                en_v,
            )
            c_bm, _a4, _b4, _c4, _e4, _v4 = _pt_mnee_constraint(
                xp,
                pm,
                xn,
                fixed,
                n_m,
                g_m,
                u_m,
                v_m,
                du_m,
                dv_m,
                ior,
                ep_u,
                ep_v,
                en_u,
                en_v,
            )
            d = (c_bp - c_bm) / (2.0 * eps)
            out[4, k] = d[0]
            out[4, 2 + k] = d[1]
            # C: the next point moves (or the fixed direction turns).
            e_n = en_u if k == 0 else en_v
            xn_p = xn + e_n * eps
            xn_m = xn - e_n * eps
            if fixed != 0:
                xn_p = xn_p.normalized()
                xn_m = xn_m.normalized()
            c_np, _a5, _b5, _c5, _e5, _v5 = _pt_mnee_constraint(
                xp,
                p,
                xn_p,
                fixed,
                n,
                ng,
                du,
                dv,
                dndu,
                dndv,
                ior,
                ep_u,
                ep_v,
                en_u,
                en_v,
            )
            c_nm, _a6, _b6, _c6, _e6, _v6 = _pt_mnee_constraint(
                xp,
                p,
                xn_m,
                fixed,
                n,
                ng,
                du,
                dv,
                dndu,
                dndv,
                ior,
                ep_u,
                ep_v,
                en_u,
                en_v,
            )
            d = (c_np - c_nm) / (2.0 * eps)
            out[5, k] = d[0]
            out[5, 2 + k] = d[1]


@pytest.mark.parametrize("fixed", [0, 1])
@pytest.mark.parametrize("from_inside", [False, True])
def test_constraint_blocks_match_finite_differences(fixed, from_inside):
    pos, nrm = _tri_tables(_CURVED_POS, _CURVED_NRM)
    above = [0.2, 0.3, 2.0]
    below = [0.45, 0.05, -1.5]
    xp, xn = (below, above) if from_inside else (above, below)
    if fixed:
        p_guess = np.array([0.3, 0.25, 0.0])
        xn = list(_unit(np.asarray(xn) - p_guess))
    rng = np.random.default_rng(3)
    frames = [_unit(rng.normal(size=3)) for _ in range(4)]
    if fixed:
        w = np.asarray(xn)
        frames[2] = _unit(np.cross(w, [1.0, 0.0, 0.0]))
        frames[3] = np.cross(w, frames[2])
    args = [0.3, 0.25, 0.2, 1.5, float(fixed), *xp, *xn, 1e-3]
    for e in frames:
        args.extend(e.tolist())
    out = torch.zeros((7, 4), dtype=torch.float32)
    _constraint_probe(pos, nrm, torch.tensor(args, dtype=torch.float32), out)
    out = out.numpy().astype(np.float64)
    assert out[6, 3] == 1.0
    expected_eta = 1.0 / 1.5 if from_inside else 1.5
    assert abs(out[6, 2] - expected_eta) < 1e-6
    for analytic, numeric in ((0, 3), (1, 4), (2, 5)):
        scale = max(1.0, np.abs(out[numeric]).max())
        np.testing.assert_allclose(out[analytic], out[numeric], atol=3e-3 * scale)


def test_same_side_neighbours_are_rejected():
    pos, nrm = _tri_tables(_CURVED_POS, _CURVED_NRM)
    args = [0.3, 0.25, 0.0, 1.5, 0.0, 0.2, 0.3, 2.0, 0.4, 0.1, 1.0, 1e-3]
    args.extend([1.0, 0.0, 0.0] * 4)
    out = torch.zeros((7, 4), dtype=torch.float32)
    _constraint_probe(pos, nrm, torch.tensor(args, dtype=torch.float32), out)
    assert out[6, 3].item() == 0.0


# ---------------------------------------------------------------------------
# The walk and the geometry term on flat interfaces: big facets, so a solve
# never leaves its triangle and the BVH projection is never consulted (the
# traversal arguments are inert placeholders here; renders cover the rest).
# ---------------------------------------------------------------------------

_MAT_WIDTH = 41


@ti.kernel
def _solve_probe(
    tri_pos: ti.types.ndarray(),
    tri_norm: ti.types.ndarray(),
    tri_mat: ti.types.ndarray(),
    tri_extra: ti.types.ndarray(),
    tri_uvs: ti.types.ndarray(),
    tri_tex_meta: ti.types.ndarray(),
    textures: ti.types.ndarray(),
    nodes: ti.types.ndarray(ndim=2),
    miss: ti.types.ndarray(),
    leaf: ti.types.ndarray(),
    span: ti.types.ndarray(),
    shell: ti.types.ndarray(),
    seeds: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    for i in range(seeds.shape[0]):
        x0 = ti.math.vec3(seeds[i, 0], seeds[i, 1], seeds[i, 2])
        y = ti.math.vec3(seeds[i, 3], seeds[i, 4], seeds[i, 5])
        fixed = ti.cast(seeds[i, 6], ti.i32)
        count = ti.cast(seeds[i, 7], ti.i32)
        prims = ti.Vector([-1] * MNEE_MAX_VERTICES)
        bary_a = ti.Vector([0.0] * MNEE_MAX_VERTICES)
        bary_b = ti.Vector([0.0] * MNEE_MAX_VERTICES)
        cids = ti.Vector([0] * MNEE_MAX_VERTICES)
        for q in ti.static(range(MNEE_MAX_VERTICES)):
            prims[q] = ti.cast(seeds[i, 8 + 3 * q], ti.i32)
            bary_a[q] = seeds[i, 9 + 3 * q]
            bary_b[q] = seeds[i, 10 + 3 * q]
        (
            ok,
            capped,
            prims,
            bary_a,
            bary_b,
            ps,
            ngs,
            etas,
            a_blks,
            b_blks,
            c_blks,
            _cf,
        ) = _pt_mnee_solve(
            0,
            x0,
            ti.math.vec3(0.0, 0.0, 0.0),
            y,
            fixed,
            count,
            prims,
            bary_a,
            bary_b,
            cids,
            0,
            0.0,
            nodes,
            miss,
            leaf,
            span,
            1,
            tri_pos,
            tri_norm,
            tri_mat,
            tri_extra,
            tri_uvs,
            tri_tex_meta,
            textures,
            tri_pos.shape[1],
            shell,
            1,
        )
        g, g_ok = _pt_mnee_geometry(x0, count, ps, ngs, a_blks, b_blks, c_blks)
        out[i, 0] = ti.cast(ok, ti.f32)
        out[i, 1] = ti.cast(capped, ti.f32)
        out[i, 2] = g if g_ok == 1 else -1.0
        for q in ti.static(range(MNEE_MAX_VERTICES)):
            for k in ti.static(range(3)):
                out[i, 3 + 3 * q + k] = ps[q, k]
            out[i, 9 + q] = etas[q]


def _plane_tris(z, up, size=60.0):
    """One facet covering the test region of the plane ``z`` well inside its
    edges (normal +z when ``up``).
    """
    tri = [[-size, -size, z], [size, -size, z], [0.0, size, z]]
    if not up:
        tri = [tri[0], tri[2], tri[1]]
    n = [0.0, 0.0, 1.0 if up else -1.0]
    return [tri], [[n, n, n]]


def _material_tables(n_tri, ior):
    mat = torch.zeros((1, n_tri, _MAT_WIDTH), dtype=torch.float32)
    extra = torch.zeros((1, n_tri, 15), dtype=torch.float32)
    extra[..., 6:9] = ior
    extra[..., 9:12] = 1.0
    return mat, extra


def _run_solve(tris, normals, ior, seeds):
    pos, nrm = _tri_tables(tris, normals)
    mat, extra = _material_tables(pos.shape[1], ior)
    out = torch.zeros((len(seeds), 12), dtype=torch.float32)
    _solve_probe(
        pos,
        nrm,
        mat,
        extra,
        torch.zeros((1, 1, 6), dtype=torch.float32),
        torch.zeros((1, 12), dtype=torch.int32),
        torch.zeros((1, 1, 4), dtype=torch.float32),
        torch.zeros((1, 4), dtype=torch.int64),
        torch.zeros((1,), dtype=torch.int32),
        torch.zeros((1,), dtype=torch.int32),
        torch.zeros((1,), dtype=torch.int32),
        torch.zeros((1, 1 + pos.shape[1]), dtype=torch.int32),
        torch.tensor(np.asarray(seeds, dtype=np.float32)),
        out,
    )
    return out.numpy().astype(np.float64)


def _bary(tri, p):
    v0, v1, v2 = (np.asarray(v, dtype=np.float64) for v in tri)
    m = np.stack([v1 - v0, v2 - v0], axis=1)
    sol, *_ = np.linalg.lstsq(m, np.asarray(p, dtype=np.float64) - v0, rcond=None)
    return sol


def _seed_on_plane(tris, x0, toward, z, shift=(0.0, 0.0)):
    """Seed (prim, a, b) where the straight segment crosses the plane,
    optionally moved by ``shift`` across it.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    toward = np.asarray(toward, dtype=np.float64)
    t = (z - x0[2]) / (toward[2] - x0[2])
    p = x0 + t * (toward - x0)
    p[0] += shift[0]
    p[1] += shift[1]
    for prim, tri in enumerate(tris):
        a, b = _bary(tri, p)
        if a >= 0 and b >= 0 and a + b <= 1:
            return prim, a, b
    raise AssertionError("seed off the test plane")


def _snell_plane(x0, y, z, n_lower, n_upper):
    """The refraction point on the plane ``z`` of the path x0 -> y (float64
    bisection on Fermat's optical length, which is convex along the line).
    """
    x0 = np.asarray(x0, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n0 = n_lower if x0[2] < z else n_upper
    n1 = n_upper if y[2] > z else n_lower

    def length(s):
        p = np.array([x0[0] + s * (y[0] - x0[0]), x0[1] + s * (y[1] - x0[1]), z])
        return n0 * np.linalg.norm(p - x0) + n1 * np.linalg.norm(y - p)

    lo, hi = 0.0, 1.0
    for _ in range(200):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if length(m1) < length(m2):
            hi = m2
        else:
            lo = m1
    s = 0.5 * (lo + hi)
    return np.array([x0[0] + s * (y[0] - x0[0]), x0[1] + s * (y[1] - x0[1]), z])


@pytest.mark.parametrize("ior", [1.33, 1.5])
@pytest.mark.parametrize("shift", [(0.03, -0.05), (0.0, -2.4)])
def test_flat_interface_on_axis_matches_closed_form(ior, shift):
    # Water below the plane z = 0 (its outward normal +z); a receiver at
    # depth h2 under a point light at height h1. Reversing the pencil: rays
    # leaving x0 at angle t reach the light plane at t (h2 + n h1), so the
    # geometry term is 1 / (h2 + n h1)^2. The second seed starts the walk at
    # a wide angle, where the half vector has saturated and an unchecked
    # Newton step flies off the facet.
    tris, normals = _plane_tris(0.0, up=True)
    h1, h2 = 2.0, 0.75
    x0 = [0.0, 0.0, -h2]
    y = [0.0, 0.0, h1]
    prim, a, b = _seed_on_plane(tris, x0, y, 0.0, shift)
    seeds = [[*x0, *y, 0.0, 1.0, prim, a, b, -1, 0.0, 0.0]]
    out = _run_solve(tris, normals, ior, seeds)
    assert out[0, 0] == 1.0, out[0]
    np.testing.assert_allclose(out[0, 3:6], [0.0, 0.0, 0.0], atol=1e-4)
    assert abs(out[0, 9] - 1.0 / ior) < 1e-6
    assert out[0, 2] == pytest.approx(1.0 / (h2 + ior * h1) ** 2, rel=2e-3)


def _plane_geometry_fd(x0, y, z, n_lower, n_upper, fixed, eps=1e-4):
    """Finite-difference generalized geometry term of the one-vertex chain
    through the plane ``z``, in the measure ``_pt_mnee_geometry`` uses.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def solve(end):
        if fixed:
            # A fixed direction: the end is a point far along it.
            return _snell_plane(x0, x0 + end * 1e4, z, n_lower, n_upper)
        return _snell_plane(x0, end, z, n_lower, n_upper)

    p1 = solve(y)
    w_l = _unit(y if fixed else y - p1)
    b1 = _unit(
        np.cross(w_l, [1.0, 0.0, 0.0] if abs(w_l[0]) <= 0.9 else [0.0, 1.0, 0.0])
    )
    b2 = np.cross(w_l, b1)
    cols = []
    for e in (b1, b2):
        if fixed:
            d_p = solve(_unit(y + e * eps)) - solve(_unit(y - e * eps))
        else:
            d_p = solve(y + e * eps) - solve(y - e * eps)
        cols.append(d_p[:2] / (2.0 * eps))
    det = abs(np.linalg.det(np.stack(cols, axis=1)))
    leg = p1 - x0
    return abs(leg[2]) / np.linalg.norm(leg) / leg.dot(leg) * det


@pytest.mark.parametrize("fixed", [0, 1])
@pytest.mark.parametrize("receiver_below", [True, False])
def test_flat_interface_off_axis_matches_finite_differences(fixed, receiver_below):
    ior = 1.33
    tris, normals = _plane_tris(0.0, up=True)
    if receiver_below:
        x0 = [0.3, -0.2, -0.8]
        y = [-0.9, 0.7, 1.7]
    else:
        x0 = [0.3, -0.2, 0.8]
        y = [-0.9, 0.7, -1.7]
    if fixed:
        y = list(_unit(np.asarray(y) - np.asarray(x0)) * np.array([1.0, 0.8, 1.0]))
        y = list(_unit(y))
        toward = np.asarray(x0) + np.asarray(y) * 5.0
    else:
        toward = y
    prim, a, b = _seed_on_plane(tris, x0, toward, 0.0)
    seeds = [[*x0, *y, float(fixed), 1.0, prim, a, b, -1, 0.0, 0.0]]
    out = _run_solve(tris, normals, ior, seeds)
    assert out[0, 0] == 1.0, out[0]
    if fixed:
        # The solved vertex refracts the leg from x0 into the fixed direction.
        p1 = out[0, 3:6]
        wi = _unit(np.asarray(x0) - p1)
        wo = np.asarray(y)
        n_i = ior if wi[2] < 0 else 1.0
        n_o = ior if wo[2] < 0 else 1.0
        sin_i = math.hypot(wi[0], wi[1])
        sin_o = math.hypot(wo[0], wo[1])
        assert n_i * sin_i == pytest.approx(n_o * sin_o, rel=1e-3)
    else:
        expected = _snell_plane(x0, y, 0.0, ior, 1.0)
        np.testing.assert_allclose(out[0, 3:6], expected, atol=2e-4)
    g_fd = _plane_geometry_fd(x0, y, 0.0, ior, 1.0, fixed)
    assert out[0, 2] == pytest.approx(g_fd, rel=5e-3)


@pytest.mark.parametrize("depth", [1e-2, 1e-3])
def test_a_receiver_just_under_the_interface_still_converges(depth):
    # Just under a facet the first leg is a difference of f32 points that
    # round at the facet's size, so its direction is uncertain by more than
    # MNEE_THRESHOLD and the walk could never get below it: a scatter point
    # inside a subsurface solid that close to its face lost its light. The
    # walk converges to the chain's own noise floor there instead
    # (MNEE_ROUNDING), which still puts the vertex where Snell's law does and
    # the geometry term within the percent or so such short legs resolve.
    ior = 1.5
    tris, normals = _plane_tris(0.0, up=True, size=4.0)
    x0 = [0.2, -0.1, -depth]
    y = [-1.1, 0.8, 2.5]
    prim, a, b = _seed_on_plane(tris, x0, y, 0.0)
    seeds = [[*x0, *y, 0.0, 1.0, prim, a, b, -1, 0.0, 0.0]]
    out = _run_solve(tris, normals, ior, seeds)
    assert out[0, 0] == 1.0, out[0]
    expected = _snell_plane(x0, y, 0.0, ior, 1.0)
    np.testing.assert_allclose(out[0, 3:6], expected, atol=1e-5)
    g_fd = _plane_geometry_fd(x0, y, 0.0, ior, 1.0, 0)
    assert out[0, 2] == pytest.approx(g_fd, rel=3e-2)


def test_slab_two_vertex_chain_matches_closed_form():
    # A glass slab between z = -d and z = 0, in air: the top facet's outward
    # normal is +z, the bottom's -z. On axis the pencil from x0 spreads by
    # (h_below + d / n + h_above), so the term is its inverse square.
    ior, d, h_below, h_above = 1.5, 0.4, 0.6, 1.3
    top, top_n = _plane_tris(0.0, up=True)
    bottom, bottom_n = _plane_tris(-d, up=False)
    tris = bottom + top
    normals = bottom_n + top_n
    x0 = [0.0, 0.0, -d - h_below]
    y = [0.0, 0.0, h_above]
    p_b, a_b, b_b = _seed_on_plane(bottom, x0, y, -d, (0.05, 0.02))
    p_t, a_t, b_t = _seed_on_plane(top, x0, y, 0.0, (-0.03, 0.06))
    seeds = [[*x0, *y, 0.0, 2.0, p_b, a_b, b_b, 1 + p_t, a_t, b_t]]
    out = _run_solve(tris, normals, ior, seeds)
    assert out[0, 0] == 1.0, out[0]
    np.testing.assert_allclose(out[0, 3:6], [0.0, 0.0, -d], atol=1e-4)
    np.testing.assert_allclose(out[0, 6:9], [0.0, 0.0, 0.0], atol=1e-4)
    assert out[0, 9] == pytest.approx(ior)
    assert out[0, 10] == pytest.approx(1.0 / ior)
    expected = 1.0 / (h_below + d / ior + h_above) ** 2
    assert out[0, 2] == pytest.approx(expected, rel=2e-3)


def test_total_internal_reflection_has_no_solution():
    # From inside water toward a light far off to the side: past the
    # critical angle no refracted chain exists, and the walk must say so.
    tris, normals = _plane_tris(0.0, up=True)
    x0 = [0.0, 0.0, -0.2]
    y = [6.0, 0.0, 0.05]
    prim, a, b = _seed_on_plane(tris, x0, [0.5, 0.0, 0.0], 0.0)
    seeds = [[*x0, *y, 0.0, 1.0, prim, a, b, -1, 0.0, 0.0]]
    out = _run_solve(tris, normals, 1.33, seeds)
    expected = _snell_plane(x0, y, 0.0, 1.33, 1.0)
    # Fermat's stationary point exists for a light above the plane; it is
    # only TIR-bound when the light sits below the critical cone. Here the
    # light is barely above the plane, far away: the stationary point is
    # grazing, so either the walk converges onto it or it fails -- it must
    # never report a point that does not refract.
    if out[0, 0] == 1.0:
        np.testing.assert_allclose(out[0, 3:6], expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Admissibility: the decisions after a walk, which the caustic block and the
# coverage test share
# ---------------------------------------------------------------------------


@ti.kernel
def _admissible_probe(
    tri_pos: ti.types.ndarray(),
    tri_norm: ti.types.ndarray(),
    tri_mat: ti.types.ndarray(),
    tri_extra: ti.types.ndarray(),
    tri_uvs: ti.types.ndarray(),
    tri_tex_meta: ti.types.ndarray(),
    textures: ti.types.ndarray(),
    nodes: ti.types.ndarray(ndim=2),
    miss: ti.types.ndarray(),
    leaf: ti.types.ndarray(),
    span: ti.types.ndarray(),
    first_leaf: ti.i32,
    shell: ti.types.ndarray(),
    seeds: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    for i in range(seeds.shape[0]):
        x0 = ti.math.vec3(seeds[i, 0], seeds[i, 1], seeds[i, 2])
        y = ti.math.vec3(seeds[i, 3], seeds[i, 4], seeds[i, 5])
        fixed = ti.cast(seeds[i, 6], ti.i32)
        count = ti.cast(seeds[i, 7], ti.i32)
        n_recv = ti.math.vec3(seeds[i, 14], seeds[i, 15], seeds[i, 16])
        prims = ti.Vector([-1] * MNEE_MAX_VERTICES)
        bary_a = ti.Vector([0.0] * MNEE_MAX_VERTICES)
        bary_b = ti.Vector([0.0] * MNEE_MAX_VERTICES)
        cids = ti.Vector([0] * MNEE_MAX_VERTICES)
        for q in ti.static(range(MNEE_MAX_VERTICES)):
            prims[q] = ti.cast(seeds[i, 8 + 3 * q], ti.i32)
            bary_a[q] = seeds[i, 9 + 3 * q]
            bary_b[q] = seeds[i, 10 + 3 * q]
        (
            ok,
            _capped,
            prims,
            bary_a,
            bary_b,
            ps,
            ngs,
            _etas,
            _a_blks,
            _b_blks,
            _c_blks,
            _cf,
        ) = _pt_mnee_solve(
            0,
            x0,
            ti.math.vec3(0.0, 0.0, 0.0),
            y,
            fixed,
            count,
            prims,
            bary_a,
            bary_b,
            cids,
            0,
            0.0,
            nodes,
            miss,
            leaf,
            span,
            first_leaf,
            tri_pos,
            tri_norm,
            tri_mat,
            tri_extra,
            tri_uvs,
            tri_tex_meta,
            textures,
            tri_pos.shape[1],
            shell,
            1,
        )
        out[i, 0] = ti.cast(ok, ti.f32)
        if ok == 1:
            sorigin = x0
            if n_recv.norm() > 0.0:
                sorigin = _offset_ray_origin(x0, n_recv)
            out[i, 1] = ti.cast(
                _pt_mnee_admissible(
                    0,
                    x0,
                    sorigin,
                    n_recv,
                    y,
                    fixed,
                    count,
                    prims,
                    bary_a,
                    bary_b,
                    ps,
                    ngs,
                    0,
                    0.0,
                    nodes,
                    miss,
                    leaf,
                    span,
                    first_leaf,
                    tri_pos,
                    tri_norm,
                    tri_mat,
                    shell,
                    1,
                ),
                ti.f32,
            )


def _run_admissible(tris, normals, ior, seeds):
    from algan.rendering.raytracing.stbvh import build_stbvh

    pos, nrm = _tri_tables(tris, normals)
    corners = pos.view(1, -1, 3, 3)
    pad = 1e-3
    bvh = build_stbvh(
        (corners.amin(2) - pad).contiguous(),
        (corners.amax(2) + pad).contiguous(),
        num_frames=1,
        tightness=1.0,
    )
    mat, extra = _material_tables(pos.shape[1], ior)
    out = torch.full((len(seeds), 2), -1.0, dtype=torch.float32)
    _admissible_probe(
        pos,
        nrm,
        mat,
        extra,
        torch.zeros((1, 1, 6), dtype=torch.float32),
        torch.zeros((1, 12), dtype=torch.int32),
        torch.zeros((1, 1, 4), dtype=torch.float32),
        bvh.kernel_blocks,
        bvh.node_miss,
        bvh.leaf_prim,
        bvh.leaf_tspan,
        int(bvh.first_leaf),
        torch.zeros((1, 1 + pos.shape[1]), dtype=torch.int32),
        torch.tensor(np.asarray(seeds, dtype=np.float32)),
        out,
    )
    return out.numpy().astype(np.float64)


def _one_vertex_seed(tris, x0, y, fixed, z, n_recv):
    toward = np.asarray(x0) + 10.0 * np.asarray(y) if fixed else y
    prim, a, b = _seed_on_plane(tris, x0, toward, z)
    return [*x0, *y, float(fixed), 1.0, prim, a, b, -1, 0.0, 0.0, *n_recv]


def test_a_chain_is_admissible_only_where_a_light_sample_delivers_it():
    # One interface at z = 0, its material below (normal +z), and a second
    # caster at z = 1.5 that only the last of these chains has in its way.
    tris, normals = _plane_tris(0.0, up=True, size=8.0)
    up = [0.0, 0.0, 1.0]
    down = [0.0, 0.0, -1.0]
    sky = list(_unit([0.2, 0.1, 1.0]))
    seeds = [
        # A pool floor lit by the sky through the surface.
        _one_vertex_seed(tris, [0.3, -0.2, -1.0], sky, 1, 0.0, up),
        # The same chain from a receiver facing away from the interface.
        _one_vertex_seed(tris, [0.3, -0.2, -1.0], sky, 1, 0.0, down),
        # A light at infinity on the material side: inside an endless solid.
        _one_vertex_seed(tris, [0.3, -0.2, 1.0], [-s for s in sky], 1, 0.0, down),
        # A point light there is merely inside the solid.
        _one_vertex_seed(tris, [0.3, -0.2, 1.0], [0.5, 0.3, -1.5], 0, 0.0, down),
    ]
    out = _run_admissible(tris, normals, 1.33, seeds)
    assert (out[:, 0] == 1.0).all(), out
    assert out[:, 1].tolist() == [1.0, 0.0, 0.0, 1.0]
    above, above_n = _plane_tris(1.5, up=True, size=8.0)
    out = _run_admissible(tris + above, normals + above_n, 1.33, seeds[:1])
    assert out[0].tolist() == [1.0, 0.0]


@pytest.fixture
def caster_merge(monkeypatch):
    """Five triangles of smooth clear glass in one physical material (a test
    changes what it needs), with the arena copy a plain clone.
    """
    from contextlib import nullcontext
    from types import SimpleNamespace

    from algan.rendering.raytracing import settings as rt_settings
    from algan.rendering.raytracing import tracer
    from algan.rendering.raytracing.settings import _MAT_DEFAULTS
    from algan.rendering.raytracing.shading_taichi import _MID_PHYSICAL

    monkeypatch.setattr(tracer, "_arena_copy", lambda _memory, value: value.clone())
    monkeypatch.setattr(rt_settings, "pt_caustics", True)
    memory = SimpleNamespace(scope=lambda *_args, **_kwargs: nullcontext())
    n = 5
    extra = torch.zeros((1, n, 15))
    extra[..., 6:9] = 1.5
    extra[..., 9:12] = 1.0
    merged = {
        "tri_mat": torch.tensor(_MAT_DEFAULTS).reshape(1, 1, -1).repeat(1, n, 1),
        "tri_mat_id": torch.full((1, n), _MID_PHYSICAL, dtype=torch.int32),
        "tri_obj": torch.arange(10, 10 + n, dtype=torch.int32).view(1, n),
        "tri_extra": extra,
        "tri_tex_meta": torch.full((1, 21), -1, dtype=torch.int32),
        "num_colored_triangles": n,
        "num_triangles": n,
    }
    return memory, merged


def test_casters_are_the_smooth_refracting_lit_triangles(caster_merge):
    from algan.rendering.raytracing.pt_caustics import _caster_mask
    from algan.rendering.raytracing.shading_taichi import _MID_UNLIT

    _memory, merged = caster_merge
    extra = merged["tri_extra"]
    extra[0, 1, 1:6:2] = 0.3  # rough
    extra[0, 2, 0:6:2] = 1.0  # wholly metal
    extra[0, 3, 6:9] = 1.0  # index-matched: passes straight
    merged["tri_mat_id"][0, 4] = _MID_UNLIT
    assert _caster_mask(merged).tolist() == [[True, False, False, False, False]]
    extra[0, 0, 9:12] = 0.0  # opaque
    assert not _caster_mask(merged).any()


def test_a_property_anywhere_on_the_triangle_keeps_it(caster_merge):
    # Corner values interpolate: one smooth, transmitting, dielectric,
    # bending corner is enough for some point of the face to refract.
    from algan.rendering.raytracing.pt_caustics import _caster_mask

    _memory, merged = caster_merge
    extra = merged["tri_extra"]
    extra[0, :, 1:6:2] = torch.tensor([0.0, 0.5, 0.5])
    extra[0, :, 9:12] = torch.tensor([0.0, 0.0, 0.2])
    extra[0, :, 0:6:2] = torch.tensor([1.0, 1.0, 0.0])
    extra[0, :, 6:9] = torch.tensor([1.0, 1.0, 1.33])
    assert _caster_mask(merged).all()


def test_a_mob_that_opts_out_casts_no_caustic(caster_merge):
    from algan.rendering.raytracing.pt_caustics import _caster_mask
    from algan.rendering.raytracing.shading_taichi import _MAT_NO_CAUSTIC_CAST

    _memory, merged = caster_merge
    merged["tri_mat"][0, 2, _MAT_NO_CAUSTIC_CAST] = 1.0
    assert _caster_mask(merged).tolist() == [[True, True, False, True, True]]


def test_glass_that_casts_no_shadow_casts_no_caustic(caster_merge):
    # The walks skip a triangle that casts no shadow, so as a caster it would
    # hold back the BSDF paths through it for a chain that never forms.
    from algan.rendering.raytracing.pt_caustics import _caster_mask

    _memory, merged = caster_merge
    merged["tri_frame_casts"] = torch.tensor([[True, False, True, True, True]])
    assert _caster_mask(merged).tolist() == [[True, False, True, True, True]]
    # Per-frame rows broadcast against per-frame materials; rows that cannot
    # keep a triangle only where it casts in every frame.
    merged["tri_extra"] = merged["tri_extra"].repeat(2, 1, 1)
    merged["tri_frame_casts"] = torch.tensor(
        [[True, False, True, True, True], [True, True, True, False, True]]
    )
    assert _caster_mask(merged).tolist() == [
        [True, False, True, True, True],
        [True, True, True, False, True],
    ]
    merged["tri_extra"] = merged["tri_extra"][:1].repeat(3, 1, 1)
    assert _caster_mask(merged).tolist() == [[True, False, True, False, True]] * 3


def test_material_maps_and_promoted_triangles(caster_merge):
    # A transmission map (bit 8) may make an opaque-by-corners triangle
    # transmit; a promoted constant-material triangle never does.
    from algan.rendering.raytracing.pt_caustics import _caster_mask

    _memory, merged = caster_merge
    merged["tri_extra"][0, 3:, 9:12] = 0.0
    merged["num_colored_triangles"] = 3
    meta = torch.full((2, 21), -1, dtype=torch.int32)
    meta[0, 3] = 0
    meta[0, 9] = 8
    meta[1, 3] = 0
    meta[1, 9] = 1 | 2 | 4  # what a promoted map carries
    merged["tri_tex_meta"] = meta
    assert _caster_mask(merged).tolist() == [[True, True, True, True, False]]
    # Past the per-vertex rows a property the map does not drive takes the
    # kernel's default, index 1 included: the map must supply the index too.
    merged["tri_extra"] = merged["tri_extra"][:, :3]
    assert _caster_mask(merged).tolist() == [[True, True, True, False, False]]
    meta[0, 9] = 8 | 4
    assert _caster_mask(merged).tolist() == [[True, True, True, True, False]]


def test_the_caster_section_follows_the_shell_columns(caster_merge):
    from algan.rendering.raytracing import settings as rt_settings
    from algan.rendering.raytracing.pt_caustics import _prepare_casters

    memory, merged = caster_merge
    merged["tri_extra"][0, 1, 9:12] = 0.0
    placeholder = torch.full((1, 1), -1, dtype=torch.int32)
    shells, offset, media_only = _prepare_casters(memory, merged, placeholder)
    assert (offset, media_only) == (5, False)
    assert shells.tolist() == [[-1] * 5 + [10, -1, 12, 13, 14]]
    # Behind the media section (two N-wide sections already): appended after.
    media = torch.arange(10, dtype=torch.int32).view(1, 10)
    shells, offset, media_only = _prepare_casters(memory, merged, media, 5)
    assert (offset, media_only) == (10, False)
    assert shells[:, :10].tolist() == media.tolist()
    assert shells[0, 10:].tolist() == [10, -1, 12, 13, 14]
    # Off, with no scattering medium or with subsurface delta lights off, or
    # nothing that refracts: the table is untouched and the kernel compiles
    # without caustics.
    rt_settings.pt_caustics = False
    assert _prepare_casters(memory, merged, media) == (media, 0, False)
    rt_settings.pt_subsurface_delta_lights = False
    assert _prepare_casters(memory, merged, media, 5) == (media, 0, False)
    rt_settings.pt_subsurface_delta_lights = True
    rt_settings.pt_caustics = True
    merged["tri_extra"][..., 9:12] = 0.0
    assert _prepare_casters(memory, merged, media) == (media, 0, False)


def test_without_caustics_the_section_holds_only_subsurface_boundaries(
    caster_merge, monkeypatch
):
    """With ``pt_caustics`` off a batch with a scattering medium still gets a
    caster section, so delta lights reach the medium's scatter points through
    its boundary (``pt_subsurface_delta_lights``). It names only the smooth
    boundaries of scattering interiors, and says it is for media only.
    """
    from algan.rendering.raytracing import settings as rt_settings
    from algan.rendering.raytracing.pt_caustics import _prepare_casters
    from algan.rendering.raytracing.shading_taichi import (
        _MAT_MEDIUM_CLOSED,
        _MAT_SIGMA_S,
    )

    memory, merged = caster_merge
    monkeypatch.setattr(rt_settings, "pt_caustics", False)
    monkeypatch.setattr(rt_settings, "pt_subsurface_delta_lights", True)
    mat = merged["tri_mat"]
    mat[0, :, _MAT_MEDIUM_CLOSED] = 1.0
    mat[0, 0:2, _MAT_SIGMA_S] = 4.0  # 0 and 1 bound a scattering interior
    mat[0, 3, _MAT_SIGMA_S + 2] = 0.5  # so does 3, scattering blue only
    merged["tri_extra"][0, 1, 1:6:2] = 0.3  # 1 is rough: no caster
    media = torch.arange(10, dtype=torch.int32).view(1, 10)
    shells, offset, media_only = _prepare_casters(memory, merged, media, 5)
    assert (offset, media_only) == (10, True)
    # 2 and 4 are clear glass: no caster without caustics.
    assert shells[0, 10:].tolist() == [10, -1, -1, 13, -1]
    # No media section: nothing to light, nothing built.
    opacity = media[:, :5]
    assert _prepare_casters(memory, merged, opacity, 0) == (opacity, 0, False)
    # A medium whose boundary is open is not closed: not a subsurface solid.
    mat[0, :, _MAT_MEDIUM_CLOSED] = 0.0
    assert _prepare_casters(memory, merged, media, 5) == (media, 0, False)


def test_the_caster_section_broadcasts_collapsed_time_rows(caster_merge):
    memory, merged = caster_merge
    from algan.rendering.raytracing.pt_caustics import _prepare_casters

    merged["tri_extra"] = merged["tri_extra"].repeat(3, 1, 1)
    merged["tri_extra"][1, 0, 9:12] = 0.0
    shells, offset, _media_only = _prepare_casters(
        memory, merged, torch.full((1, 1), -1, dtype=torch.int32)
    )
    assert shells.shape == (3, 10)
    assert shells[:, offset].tolist() == [10, -1, 10]
