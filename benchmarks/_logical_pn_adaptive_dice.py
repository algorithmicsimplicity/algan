"""Adaptive logical PN dicing: triangle counts and dice cost.

Reports what the per-patch dice produces for a batch of camera frames, next to
what the old per-frame uniform level would have cost (``P * 4 ** max level``
across the whole batch, which is what the padded tensor used to be).

    .venv/Scripts/python.exe benchmarks/_logical_pn_adaptive_dice.py
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import torch

import algan  # noqa: F401  -- sets up devices / inference mode
from algan.mobs.shapes_3d import Sphere, Torus


def make_camera(z_positions, *, screen_height=1080, device=None):
    z = torch.as_tensor(z_positions, dtype=torch.float32, device=device)
    origins = torch.zeros((len(z), 1, 3), device=device)
    origins[..., 2] = z.view(-1, 1)
    screen_points = origins.clone()
    screen_points[..., 2] += 1.0
    return SimpleNamespace(
        ray_origin=origins,
        screen_point=screen_points,
        screen_basis=torch.eye(3, device=device).unsqueeze(0).repeat(len(z), 1, 1),
        screen_width=int(screen_height * 16 / 9),
        screen_height=screen_height,
        output_screen_width=int(screen_height * 16 / 9),
        output_screen_height=screen_height,
        analytic_raster=False,
    )


def report(name, mob, z_positions):
    from algan.rendering.raytracing.primitives import LogicalPNTrianglePrimitive

    # The scene batcher always hands the renderer the collection form, whose
    # corners are [frames, patches, 3, 3].
    primitive = LogicalPNTrianglePrimitive(
        triangle_collection=[mob.get_render_primitives()]
    )
    device = primitive.corners.device
    camera = make_camera(z_positions, device=device)
    num_patches = primitive.corners.shape[1]

    start = time.perf_counter()
    primitive._dice_logical_pn(camera)
    elapsed = time.perf_counter() - start

    levels = primitive._logical_pn_subdivision_levels
    edge_levels = primitive._logical_pn_edge_levels
    counts = (4**levels).sum(1)
    width = primitive.corners.shape[1]
    old_width = num_patches * 4 ** int(levels.amax())
    frames = levels.shape[0]

    print(
        f"{name}: {num_patches} patches x {frames} frames, "
        f"render_tolerance_pixels={primitive.render_tolerance_pixels}"
    )
    print(
        f"  patch levels  min/med/max : {int(levels.amin())} / "
        f"{int(levels.float().median())} / {int(levels.amax())}"
    )
    print(
        f"  edge  levels  min/max     : {int(edge_levels.amin())} / "
        f"{int(edge_levels.amax())}"
    )
    print(
        f"  edge below patch level    : "
        f"{float((edge_levels < levels.unsqueeze(-1)).float().mean()) * 100:.1f}%"
    )
    print(f"  per-frame triangles       : {counts.tolist()}")
    print(
        f"  padded width  new / old   : {width} / {old_width} "
        f"({old_width / max(width, 1):.1f}x)"
    )
    print(f"  total stored  new / old   : {width * frames} / {old_width * frames}")
    print(f"  dice time                 : {elapsed * 1000:.0f} ms")
    print()


if __name__ == "__main__":
    # A close-up frame mixed with distant ones: the case where one frame used to
    # set the tessellation for the whole batch.
    report("Sphere", Sphere(), [-3.0, -6.0, -12.0, -25.0, -50.0])
    report("Torus", Torus(), [-3.0, -6.0, -12.0, -25.0, -50.0])
    # A coarse mesh, where render-time dicing (rather than the construction-time
    # grid) is what carries the surface: patches near the silhouette curve away
    # far faster than the ones facing the camera.
    coarse = {"geometry_tolerance": 0.2, "max_grid_resolution": 12}
    report("Sphere (coarse grid)", Sphere(**coarse), [-2.2, -3.0, -6.0, -12.0, -25.0])
    report("Sphere (coarse grid, one close frame)", Sphere(**coarse), [-2.2])
    report("Torus (coarse grid)", Torus(**coarse), [-3.0, -6.0, -12.0, -25.0, -50.0])
