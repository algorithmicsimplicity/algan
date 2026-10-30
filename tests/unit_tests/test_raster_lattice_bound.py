"""Projected-path range limits, and the facing bit on both coverage paths.

A triangle's exact coverage (``raster_taichi._ss_pixel``) evaluates its edge
functions on a 1/4096-pixel int64 lattice, and the values that must fit are set
by how far the triangle's projected vertices reach -- unbounded for a vertex
just in front of the camera plane. A 60-wide floor whose near edge sat a few
thousandths in front of the camera overflowed it, came out with the wrong
orientation, and most of the floor vanished for that frame. The projection now
demotes such a triangle to flag 0.5 (``raster_pipeline._lattice_demote``): it
keeps its projected bbox and band scale, but the kernels ray-cast it.

The ray-cast path (``_raycast_pixel``) never used to set the facing bit, so a
back-facing straddler read as front-facing beside its projected neighbour. The
facing tests pin the two paths to the same answer.

The geometry is the probe camera of the review that found both: eye at z = 20
looking down -z, the screen plane through the origin with a half-height of 4,
176 x 99 pixels, so about 247.5 pixels per unit at unit depth.
"""

# No ``from __future__ import annotations``: it would turn the probe kernel's
# ndarray annotations into strings, which Taichi cannot compile.
import functools
import math

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

W, H = 176, 99
HALF_W, HALF_H = W / 2.0, H / 2.0
EYE = (0.0, 0.0, 20.0)
SCREEN_POINT = (0.0, 0.0, 0.0)
BASIS_X = (4.0, 0.0, 0.0)
BASIS_Y = (0.0, 4.0, 0.0)
# The sheet route's geometry-kernel template value: run representation,
# sliver policy pinned to drop (raster_pipeline: 1 + 2 + 4 * 2).
AA = 11


def test_lattice_demotion_keeps_safe_triangles_and_sends_the_rest_to_half():
    """``_lattice_demote`` on hand-placed coordinates either side of each bound.

    ``(max|sx| + W) * (max|sy| + H)`` is put at 0.98 and 1.02 of 2**36 with
    both coordinates under 2**21, and one coordinate past 2**21 with a small
    product. Only exact-path values (above 0.5) move, any of them -- a later
    exact-path flag such as 2 included -- and they move to exactly 0.5.
    """
    from algan.rendering.raytracing.raster_pipeline import (
        FLAG_FRONT_RAYCAST,
        _lattice_demote,
    )
    from algan.rendering.raytracing.raster_taichi import (
        _AA_LATTICE_MAX_COORD,
        _AA_LATTICE_MAX_EXTENT,
    )

    assert FLAG_FRONT_RAYCAST == 0.5
    assert _AA_LATTICE_MAX_EXTENT == 2.0**36
    assert _AA_LATTICE_MAX_COORD == 2.0**21

    my = 2.0**16 - H

    def mx(fraction):
        return fraction * 2.0**20 - W

    rows = [
        # (sx, sy, flag in, flag out)
        ((10.0, 100.0, 50.0), (10.0, 20.0, 80.0), 1.0, 1.0),
        ((-mx(0.98), 10.0, 20.0), (my, 10.0, 20.0), 1.0, 1.0),
        ((-mx(1.02), 10.0, 20.0), (my, 10.0, 20.0), 1.0, 0.5),
        ((-2.2e6, 10.0, 20.0), (1.0, 2.0, 3.0), 1.0, 0.5),
        ((10.0, 100.0, 50.0), (10.0, 20.0, 80.0), 2.0, 2.0),
        ((-mx(1.02), 10.0, 20.0), (my, 10.0, 20.0), 2.0, 0.5),
        ((-1e9, 1e9, 20.0), (1e9, 10.0, 20.0), 0.0, 0.0),
        ((-1e9, 1e9, 20.0), (1e9, 10.0, 20.0), -1.0, -1.0),
        ((float("nan"), 10.0, 20.0), (1.0, 2.0, 3.0), 1.0, 0.5),
    ]
    sx = torch.tensor([r[0] for r in rows], dtype=torch.float32).unsqueeze(0)
    sy = torch.tensor([r[1] for r in rows], dtype=torch.float32).unsqueeze(0)
    flag = torch.tensor([r[2] for r in rows], dtype=torch.float32).unsqueeze(0)
    out = _lattice_demote(flag, sx, sy, HALF_W, HALF_H)
    assert out.dtype == torch.float32
    assert out[0].tolist() == [r[3] for r in rows]


def _floor_triangles(delta):
    """A 60 x 40 floor quad one unit below the eye, near edge ``delta`` in
    front of the camera plane (the review's bound-study geometry), f32.
    """
    z0 = EYE[2] - delta
    a = (-30.0, -1.0, z0)
    b = (30.0, -1.0, z0)
    c = (30.0, -1.0, z0 - 40.0)
    d = (-30.0, -1.0, z0 - 40.0)
    return np.array([(a, b, c), (a, c, d)], dtype=np.float32)


def _camera_tensors(basis_y=BASIS_Y):
    def row(v):
        return torch.tensor(v, dtype=torch.float32).view(1, 3).contiguous()

    return row(EYE), row(SCREEN_POINT), row(BASIS_X), row(basis_y)


def _project(tris, basis_y=BASIS_Y):
    """The projection table the renderer builds, wide (analytic coverage)."""
    from algan.rendering.raytracing import raster_pipeline as rp

    eye, sp, pbx, pby = _camera_tensors(basis_y)
    verts = torch.from_numpy(tris).view(1, -1, 3, 3)
    return rp._triangle_projection_fused(
        verts, eye, sp, pbx, pby, HALF_W, HALF_H, True, None
    ).contiguous()


@functools.lru_cache(maxsize=1)
def _probe_kernel():
    from algan.rendering.raytracing import raster_taichi as rt
    from algan.rendering.raytracing.raytrace_kernels_taichi import CameraProjection
    from algan.rendering.taichi_runtime import init_taichi
    from algan.taichi_compat import ti

    # Algan's own init, never a bare ``ti.init`` (see test_watertight_triangle).
    init_taichi()

    @ti.kernel
    def probe(
        tri_pos: ti.types.ndarray(),
        tri_screen: ti.types.ndarray(),
        cam_origin: ti.types.ndarray(),
        screen_point: ti.types.ndarray(),
        pbx: ti.types.ndarray(),
        pby: ti.types.ndarray(),
        out: ti.types.ndarray(),
        ntri: ti.i32,
        width: ti.i32,
        height: ti.i32,
        hw: ti.f32,
        hh: ti.f32,
    ):
        cam_proj = ti.static(CameraProjection(cam_origin, False))
        for n, py, px in ti.ndrange(ntri, height, width):
            # 0-1: the emission's own dispatch on the projection flag, as the
            # write pass runs it (store_exact = 1).
            use_ss, sm, vm, cam_o, il = rt._ss_setup(
                0, n, 1, tri_pos, tri_screen, cam_origin, AA
            )
            ok, _lp, _t, _w1, _w2, _cov, msk = rt._pair_pixel(
                n,
                0,
                0,
                0,
                width,
                height,
                py * width + px,
                0,
                0,
                width,
                height,
                0,
                width * height,
                hw,
                hh,
                use_ss,
                sm,
                vm,
                cam_o,
                il,
                cam_origin,
                screen_point,
                pbx,
                pby,
                cam_proj,
                AA,
                1,
            )
            out[n, py, px, 0] = ok
            out[n, py, px, 1] = msk
            # 2-3: the projected path forced, whatever the flag says.
            sm2 = ti.Matrix.zero(ti.f32, 3, 3)
            for i in ti.static(range(3)):
                sm2[0, i] = tri_screen[0, n, i]
                sm2[1, i] = tri_screen[0, n, 3 + i]
                sm2[2, i] = tri_screen[0, n, 6 + i]
            il2 = ti.math.vec3(
                tri_screen[0, n, 10], tri_screen[0, n, 11], tri_screen[0, n, 12]
            )
            ok2, _t2, _a2, _b2, _c2, msk2 = rt._ss_pixel(
                px, py, sm2, vm, cam_o, il2, AA, 1, 0, cam_origin, cam_proj
            )
            out[n, py, px, 2] = ok2
            out[n, py, px, 3] = msk2
            # 4-5: the ray-cast path forced.
            ok3, _t3, _a3, _b3, _c3, msk3 = rt._raycast_pixel(
                px,
                py,
                0,
                vm,
                hw,
                hh,
                cam_origin,
                screen_point,
                pbx,
                pby,
                cam_proj,
                AA,
            )
            out[n, py, px, 4] = ok3
            out[n, py, px, 5] = msk3

    return probe


def _run_probe(tris, screen, basis_y=BASIS_Y):
    probe = _probe_kernel()
    eye, sp, pbx, pby = _camera_tensors(basis_y)
    ntri = tris.shape[0]
    out = torch.zeros((ntri, H, W, 6), dtype=torch.int32)
    probe(
        torch.from_numpy(tris).view(1, ntri, 9).contiguous(),
        screen,
        eye,
        sp,
        pbx,
        pby,
        out,
        ntri,
        W,
        H,
        HALF_W,
        HALF_H,
    )
    return out.numpy()


def _sample_offsets():
    from algan.rendering.raytracing.raster_taichi import _AA_FIXED_SCALE, _AA_SAMPLES

    return [(ox / _AA_FIXED_SCALE, oy / _AA_FIXED_SCALE) for ox, oy in _AA_SAMPLES]


def _truth_masks(tri, band=0.05):
    """Per-pixel sample masks of an all-front triangle in float64: its f32
    world vertices projected exactly, each sample tested against the three
    edges. Samples within ``band`` pixels of an edge are left out (``care``),
    so neither path's float rounding can decide the comparison.
    """
    eye = np.array(EYE)
    sp = np.array(SCREEN_POINT)
    pbx, pby = np.array(BASIS_X), np.array(BASIS_Y)
    n = np.cross(pbx, pby)
    big_d = (sp - eye) @ n
    pts = []
    for v in tri.astype(np.float64):
        d = v - eye
        rel = eye + (big_d / (d @ n)) * d - sp
        u = np.cross(rel, pby) @ n / (n @ n)
        w = np.cross(pbx, rel) @ n / (n @ n)
        pts.append((u * HALF_H + HALF_W, w * HALF_H + HALF_H))
    pts = np.array(pts)
    area2 = (pts[1, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1]) - (
        pts[1, 1] - pts[0, 1]
    ) * (pts[2, 0] - pts[0, 0])
    orient = 1.0 if area2 > 0 else -1.0
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    mask = np.zeros((H, W), dtype=np.int64)
    care = np.zeros((H, W), dtype=np.int64)
    for k, (ox, oy) in enumerate(_sample_offsets()):
        qx = xs + 0.5 + ox
        qy = ys + 0.5 + oy
        dmin = np.full((H, W), np.inf)
        for i in range(3):
            p, q = pts[(i + 1) % 3], pts[(i + 2) % 3]
            ex, ey = q[0] - p[0], q[1] - p[1]
            dist = orient * (ex * (qy - p[1]) - ey * (qx - p[0])) / math.hypot(ex, ey)
            dmin = np.minimum(dmin, dist)
        mask |= (dmin > 0).astype(np.int64) << k
        care |= (np.abs(dmin) > band).astype(np.int64) << k
    extent = (np.abs(pts[:, 0]).max() + W) * (np.abs(pts[:, 1]).max() + H)
    return mask, care, extent


@pytest.mark.parametrize(
    ("delta", "expected_flag"),
    [
        # (max|sx| + W)(max|sy| + H) ~ 2**35.8: just inside, exact path.
        (0.0055, 1.0),
        # ~ 2**36.1: just outside. The old exact path still drew this one.
        (0.005, 0.5),
        # ~ 2**38.7: the old exact path lost the whole of the first triangle.
        (0.002, 0.5),
    ],
)
def test_floor_just_in_front_of_the_camera_is_covered_either_side_of_the_bound(
    delta, expected_flag
):
    """Each floor triangle's emitted sample masks equal the float64 answer.

    Whichever path the flag selects, it must be the right answer: below the
    bound the projected lattice path, above it the ray cast. Each triangle is
    hundreds of thousands to millions of pixels across, so most of it is off
    screen; on screen every sample farther than 0.05 px from an edge is
    compared.
    """
    from algan.rendering.raytracing.raster_taichi import _AA_MASK_ALL

    tris = _floor_triangles(delta)
    screen = _project(tris)
    out = _run_probe(tris, screen)
    for k in range(tris.shape[0]):
        flag = float(screen[0, k, 9])
        truth, care, extent = _truth_masks(tris[k])
        # The case sits where it says, on the f64 extents.
        assert (extent <= 2.0**36) == (expected_flag == 1.0), math.log2(extent)
        assert flag == expected_flag
        ok = out[k, :, :, 0] != 0
        emitted = np.where(ok, out[k, :, :, 1] & _AA_MASK_ALL, 0)
        wrong = ((emitted ^ truth) & care) != 0
        covered = (truth & care) != 0
        assert covered.sum() > 1000, "the triangle is meant to be on screen"
        assert int(wrong.sum()) == 0, (
            f"triangle {k} (flag {flag}): {int(wrong.sum())} of "
            f"{int(covered.sum())} covered pixels carry a wrong sample mask"
        )


def _facing_triangles():
    """Two all-front, on-screen triangles: the same corners in both windings."""
    a = (-1.0, -1.0, 10.0)
    b = (2.0, -0.5, 8.0)
    c = (0.5, 1.5, 12.0)
    return np.array([(a, b, c), (a, c, b)], dtype=np.float32)


@pytest.mark.parametrize(
    "basis_y", [BASIS_Y, (0.0, -4.0, 0.0)], ids=["standard", "mirrored"]
)
def test_raycast_facing_bit_matches_the_projected_path(basis_y):
    """The ray-cast path's facing bit equals ``_ss_pixel``'s on every pixel.

    Both windings, so one sets the bit and the other does not; and a mirrored
    pixel basis, which flips the projected orientation of the same triangle --
    the ray-cast sign must follow it rather than the world winding alone.
    """
    from algan.rendering.raytracing.raster_pipeline import AA_BACKFACE_BIT

    tris = _facing_triangles()
    screen = _project(tris, basis_y)
    assert screen[0, :, 9].tolist() == [1.0, 1.0]
    out = _run_probe(tris, screen, basis_y)
    set_by_winding = []
    for k in range(tris.shape[0]):
        both = (out[k, :, :, 2] != 0) & (out[k, :, :, 4] != 0)
        assert both.sum() > 500
        ss_bit = (out[k, :, :, 3] & AA_BACKFACE_BIT) != 0
        rc_bit = (out[k, :, :, 5] & AA_BACKFACE_BIT) != 0
        assert np.array_equal(ss_bit[both], rc_bit[both]), (
            f"winding {k}: the ray-cast facing bit disagrees on "
            f"{int((ss_bit[both] != rc_bit[both]).sum())} of {int(both.sum())} pixels"
        )
        # Constant over the triangle, as facing is.
        assert len(np.unique(rc_bit[both])) == 1
        set_by_winding.append(bool(rc_bit[both][0]))
    assert set_by_winding[0] != set_by_winding[1]
