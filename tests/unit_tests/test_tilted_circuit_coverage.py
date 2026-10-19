"""Analytic coverage of circuits on planes tilted away from the camera.

A circuit's analytic coverage is a box filter of its signed distance ON
SCREEN, but the distance is measured in the circuit's own plane. Converted
with a camera-facing pixel size, which is what the filter did until the
anisotropy caveat of DESIGN_analytic_aa.md ss4 was closed, a plane tilted away
from the camera got a filter narrower than a pixel along its foreshortened
direction, and a thin stroke drawn on it beaded: its brightness rose and fell
each time it crossed a pixel row (the floor grid of the backpropagation
video's error landscape, about +-10 % along a line). ``_bez_pixel_hit`` now
makes its nearest-edge query on screen, through the plane's own projection at
the pixel (``_circuit_screen_map``, ``_bezier_point_metrics_in``).

A shallow stroke on a steeply tilted plane is rendered at a small size and
at four times that size; the larger render, averaged back down, stands for
the true box-filtered line. Each column's ink -- the line's excess over the
background, summed down the column in linear light -- must be the same all
along the line and must match the reference column for column.

The screen query must also keep the plane query's ties. A corner's two walls
tie exactly at their shared vertex in the plane; through the screen map they
differ by rounding, and the nearer wall's repeat from the next grid cell then
took the second slot, so the oriented wedge, handed one wall twice, left the
corner a half-plane.

``analytic_aa_bez_anisotropic`` (``ALGAN_ANALYTIC_AA_BEZ_ANISOTROPIC``) is a
``ti.static`` gate, so one process exercises one arm: the render test asserts
the default arm and is skipped under the other.
"""

# No ``from __future__ import annotations``: it would turn the probe kernel's
# ndarray annotations into strings, which Taichi cannot compile.
import numpy as np
import pytest
import torch
from PIL import Image

from algan.animation_timeline.animation_contexts import Off
from algan.constants.color import BLACK, WHITE
from algan.constants.spatial import LEFT, OUT, RIGHT
from algan.mobs.shapes_2d import Line
from algan.rendering.raytracing import settings as rt_settings
from algan.scene import Scene
from algan.settings.video_settings import LD

SMALL = (320, 180)
SCALE = 4


def _linear(rgb):
    c = rgb.astype(np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _render(tmp_path, resolution, tilt, turn):
    path = tmp_path / f"tilted_{tilt}_{turn}_{resolution[0]}.png"
    with Scene() as scene:
        with Off():
            scene.set_background(BLACK)
            line = Line(LEFT * 5, RIGHT * 5, stroke_width=8, color=WHITE)
            # Turned within its plane, so it crosses pixel rows on screen, then
            # the plane is tipped away from the camera about the screen's x axis:
            # the stroke's width runs along the foreshortened direction.
            line.rotate(turn, OUT).rotate(tilt, RIGHT)
        line.spawn(animate=False)
        scene.save_frame(
            str(path),
            video_settings=LD.set(resolution=resolution, supersampling=1),
            post_processes=(),
        )
    with Image.open(path) as image:
        return _linear(np.asarray(image.convert("RGB")))


def _column_ink(img):
    luma = img @ np.array([0.2126, 0.7152, 0.0722])
    return luma.sum(0)


# Each beaded under the camera-facing filter: 17.6 % and 7.3 % spread, up to
# 26 % and 13 % off the reference in one column (the reference was a little off
# too, at a quarter of the pixel). Facing the camera the same line measured
# 0.1 %.
@pytest.mark.parametrize(("tilt", "turn"), [(-65, 20), (-60, -10)])
def test_a_stroke_on_a_tilted_plane_has_even_ink(tmp_path, tilt, turn):
    if not rt_settings.analytic_aa_bez_anisotropic:
        pytest.skip("ALGAN_ANALYTIC_AA_BEZ_ANISOTROPIC=0 selects the plane filter")
    small = _render(tmp_path, SMALL, tilt, turn)
    big = _render(tmp_path, (SMALL[0] * SCALE, SMALL[1] * SCALE), tilt, turn)
    h, w, c = big.shape
    reference = big.reshape(h // SCALE, SCALE, w // SCALE, SCALE, c).mean(axis=(1, 3))

    ink = _column_ink(small)
    ref_ink = _column_ink(reference)
    covered = np.nonzero(ref_ink > 0.25 * ref_ink.max())[0]
    assert covered.size > 0.5 * SMALL[0], "the line should span most of the frame"
    # Away from the stroke's round caps.
    span = covered[(covered > covered[0] + 12) & (covered < covered[-1] - 12)]
    ink, ref_ink = ink[span], ref_ink[span]

    spread = ink.std() / ink.mean()
    worst = np.abs(ink / ref_ink - 1).max()
    # Measured through the plane's projection: 0.2 % spread and 0.4-0.5 % worst
    # column in both cases, with the stroke's total ink unchanged.
    assert spread < 0.015, f"ink varies by {spread:.1%} along the line"
    assert worst < 0.03, f"a column's ink is {worst:.1%} off the 4x reference"
    assert abs(ink.mean() / ref_ink.mean() - 1) < 0.01


def test_a_corner_keeps_both_walls_through_any_screen_map():
    """At a corner's shared vertex the two walls tie, and both keep a slot.

    A unit square under the 8x8 spatial grid, queried from outside its
    bottom-left corner on the diagonal: both walls are closest at the vertex
    they share, an exact tie in the plane. The corner's cell lists both walls,
    and the next cells along list each again. Through a rotated screen map the
    two distances differ by rounding, and a strict ranking gave the second slot
    to whichever wall's repeat came round after the other -- at 60 of these 61
    rotations, every one but the identity -- so the wedge drew a half-plane.
    """
    from algan.rendering.raytracing import raytrace_kernels_taichi as k
    from algan.rendering.raytracing.bezier_acceleration import (
        build_bezier_edge_acceleration,
    )
    from algan.rendering.raytracing.raster_taichi import _AA_RANK_TIE
    from algan.rendering.taichi_runtime import init_taichi
    from algan.settings._startup import render_device
    from algan.taichi_compat import ti

    # Algan's own init, never a bare ``ti.init`` (test_watertight_triangle.py).
    init_taichi()
    device = render_device()
    # x0, y0, x1, y1, border-visible, inward side; counter-clockwise, so the
    # inward side is every wall's left.
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    rows = [(*corners[i], *corners[(i + 1) % 4], 1.0, 1.0) for i in range(4)]
    edges = torch.tensor([rows], dtype=torch.float32)
    accel = build_bezier_edge_acceleration(edges, torch.tensor([0, 4]))
    edges = edges.to(device).contiguous()
    accel = accel.to(device).contiguous()
    angles = torch.linspace(0.0, 1.5, 61)
    rotations = torch.stack([angles.cos(), angles.sin()], 1)
    rotations = rotations.to(device=device, dtype=torch.float32).contiguous()
    out = torch.zeros((angles.numel(), 3), dtype=torch.float32, device=device)
    # The renderer's tie for a pixel a hundredth of the square across.
    tie = _AA_RANK_TIE * 0.01

    @ti.kernel
    def probe(
        edges_2d: ti.types.ndarray(dtype=ti.f32, ndim=3),
        edge_accel: ti.types.ndarray(dtype=ti.i32, ndim=1),
        rotation: ti.types.ndarray(dtype=ti.f32, ndim=2),
        result: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        for i in range(rotation.shape[0]):
            c = rotation[i, 0]
            s = rotation[i, 1]
            (_cr, md, _cu, _cv, e1x, e1y, _s1, sd, _su, _sv, e2x, e2y, _s2) = (
                k._bezier_point_metrics_in(
                    0,
                    0,
                    -0.03,
                    -0.03,
                    0.2,
                    1,
                    edges_2d,
                    edge_accel,
                    True,
                    c,
                    -s,
                    s,
                    c,
                    0.2,
                    tie,
                )
            )
            result[i, 0] = md
            result[i, 1] = sd
            # The two walls' turn: +-1 for the corner's two, 0 for one twice.
            result[i, 2] = e1x * e2y - e1y * e2x

    probe(edges, accel, rotations, out)
    nearest, second, turn = out.cpu().T
    assert torch.allclose(second, nearest, rtol=1e-4), "both walls at the vertex"
    repeated = angles[turn.abs() < 0.5]
    assert repeated.numel() == 0, (
        f"one wall in both slots at {repeated.numel()} rotations, "
        f"from {float(repeated[0]):.3f} rad"
    )
