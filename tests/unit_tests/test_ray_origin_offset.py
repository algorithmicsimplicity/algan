"""Scale-aware secondary-ray origin placement shared by both renderers."""

import torch

from algan.rendering.raytracing.ray_origin_taichi import _offset_ray_origin
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _offset_transmitted_origin,
    default_scatter,
)
from algan.rendering.taichi_runtime import init_taichi
from algan.settings._startup import render_device
from algan.taichi_compat import ti

DEVICE = render_device().type


@ti.kernel
def _offset_probe(
    points: ti.types.ndarray(),
    sides: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    for i in range(points.shape[0]):
        p = ti.math.vec3(points[i, 0], points[i, 1], points[i, 2])
        side = ti.math.vec3(sides[i, 0], sides[i, 1], sides[i, 2])
        q = _offset_ray_origin(p, side)
        for k in ti.static(range(3)):
            out[i, k] = q[k]


@ti.kernel
def _reflection_probe(
    points: ti.types.ndarray(),
    directions: ti.types.ndarray(),
    normals: ti.types.ndarray(),
    params: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    for i in range(points.shape[0]):
        p = ti.math.vec3(points[i, 0], points[i, 1], points[i, 2])
        rd = ti.math.vec3(directions[i, 0], directions[i, 1], directions[i, 2])
        n = ti.math.vec3(normals[i, 0], normals[i, 1], normals[i, 2])
        one3 = ti.math.vec3(1.0, 1.0, 1.0)
        one4 = ti.math.vec4(1.0, 1.0, 1.0, 1.0)
        (
            _contrib,
            _pass_w,
            q,
            refl,
            _refl_w,
            _trans_orig,
            _trans_dir,
            _trans_w,
        ) = default_scatter(
            rd,
            n,
            n,
            p,
            one4,
            one3,
            1.0,
            1.0,
            1.5,
            0.0,
            params,
            0,
            0,
            1,
            1,
        )
        for k in ti.static(range(3)):
            out[i, k] = q[k]
            out[i, 3 + k] = refl[k]


@ti.kernel
def _transmission_probe(
    points: ti.types.ndarray(),
    directions: ti.types.ndarray(),
    normals: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    for i in range(points.shape[0]):
        p = ti.math.vec3(points[i, 0], points[i, 1], points[i, 2])
        rd = ti.math.vec3(directions[i, 0], directions[i, 1], directions[i, 2])
        n = ti.math.vec3(normals[i, 0], normals[i, 1], normals[i, 2])
        q = _offset_transmitted_origin(p, rd, n, n)
        for k in ti.static(range(3)):
            out[i, k] = q[k]


def _run_offset(points, sides):
    init_taichi()
    points_t = torch.tensor(points, dtype=torch.float32, device=DEVICE)
    sides_t = torch.tensor(sides, dtype=torch.float32, device=DEVICE)
    out = torch.zeros_like(points_t)
    _offset_probe(points_t, sides_t, out)
    return points_t.cpu().double(), out.cpu().double()


def test_shared_offset_scales_across_world_coordinate_orders():
    """The shared helper moves at both tiny and huge f32 coordinates.

    Below the paper's near-origin threshold it uses the small absolute fallback;
    above it the step follows the hit point's local representable spacing.  A
    fixed 1e-3 offset would instead be the same size everywhere and would round
    back to the input at the largest coordinate here.
    """
    magnitudes = [1e-6, 1e-3, 1.0, 1e3, 1e6]
    points, out = _run_offset(
        [[m, m, m] for m in magnitudes],
        [[0.0, 0.0, 1.0]] * len(magnitudes),
    )

    steps = (out[:, 2] - points[:, 2]).tolist()
    assert all(step > 0.0 for step in steps), steps
    assert torch.equal(out[:, :2], points[:, :2])

    # Near the origin the fallback stays tiny enough to be local; far from it
    # the bit-space step grows with f32 spacing instead of rounding to zero.
    assert steps[0] <= 2.0 / 65536.0
    assert steps[1] <= 2.0 / 65536.0
    assert steps[-1] > 1.0
    assert steps[-1] > 1000.0 * steps[2]


def test_large_coordinate_deterministic_reflection_does_not_self_hit():
    """A reflected deterministic ray leaves its source plane at x=1e7.

    At this magnitude f32 spacing is much larger than the old 1e-3 world
    offset, so that expression rounded back onto the hit plane.  The shared
    offset must produce a representably different origin on the reflection
    side, making the plane intersection lie behind the outgoing ray.
    """
    init_taichi()
    points = torch.tensor([[1e7, 0.0, 0.0]], dtype=torch.float32, device=DEVICE)
    directions = torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float32, device=DEVICE)
    normals = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=DEVICE)
    params = torch.zeros((1, 1, 1), dtype=torch.float32, device=DEVICE)
    out = torch.zeros((1, 6), dtype=torch.float32, device=DEVICE)
    _reflection_probe(points, directions, normals, params, out)

    hit_x = float(points[0, 0])
    origin_x = float(out[0, 0])
    reflected_x = float(out[0, 3])
    assert origin_x > hit_x
    assert reflected_x > 0.0
    self_t = (hit_x - origin_x) / reflected_x
    assert self_t < 0.0


def test_small_scale_offset_does_not_jump_past_nearby_geometry():
    """A small scene keeps geometry closer than the old 1e-3 lift visible."""
    points, out = _run_offset([[1e-3, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
    hit_x = float(points[0, 0])
    origin_x = float(out[0, 0])
    nearby_x = hit_x + 5e-5

    assert hit_x < origin_x < nearby_x
    # The historical fixed offset would have started beyond this surface.
    assert hit_x + 1e-3 > nearby_x


def test_refraction_offset_selects_entering_and_exiting_sides():
    """Transmission origins follow the outgoing side for entry and exit."""
    init_taichi()
    points = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=DEVICE,
    )
    directions = torch.tensor(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=DEVICE,
    )
    normals = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=DEVICE,
    )
    out = torch.zeros((2, 3), dtype=torch.float32, device=DEVICE)
    _transmission_probe(points, directions, normals, out)

    assert float(out[0, 0]) < float(points[0, 0])  # entering: inside (-normal)
    assert float(out[1, 0]) > float(points[1, 0])  # exiting: outside (+normal)


def test_shared_offset_moves_toward_requested_side_at_negative_coordinates():
    """Bit-space sign handling works on both sides of the world origin."""
    points, out = _run_offset(
        [[-1000.0, 2.0, 3.0], [-1000.0, 2.0, 3.0]],
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
    )
    assert float(out[0, 0]) > float(points[0, 0])
    assert float(out[1, 0]) < float(points[1, 0])
