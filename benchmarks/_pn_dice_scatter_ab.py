"""A/B the logical PN dice write-out: index_copy_ vs the advanced-index scatter.

Each selected (frame, patch) writes a contiguous run of columns of a
``[T, max_triangles, 3, D]`` output. Indexing that with a (rows, columns) pair
lowers to ``index_put_``, which cannot see the runs; folding the pair into one
flattened row index makes it ``index_copy_``.

The two write the same rows with the same values, so this checks byte-equality
of every diced array as well as timing them.

    .venv/Scripts/python.exe benchmarks/_pn_dice_scatter_ab.py
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import torch

import algan  # noqa: F401  -- sets up devices / inference mode
from algan.mobs.shapes_3d import Sphere, Torus
from algan.rendering.raytracing import primitives as primitives_module
from algan.rendering.raytracing.primitives import LogicalPNTrianglePrimitive


def _legacy_scatter(output, values, targets):
    """The write this replaced: a two-index advanced-index scatter.

    ``targets`` is the flattened ``frame * max_triangles + column``, so the
    (rows, columns) pair the old code built is recovered by dividing it back
    out. Same destinations, same values, different torch entry point.
    """
    trailing = output.shape[2:]
    columns_per_frame = output.shape[1]
    rows = torch.div(targets, columns_per_frame, rounding_mode="floor")
    columns = targets - rows * columns_per_frame
    output[rows, columns] = values.reshape(-1, *trailing)


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


_CAPTURED = (
    "corners",
    "normals",
    "colors",
    "_logical_pn_padding",
    "_logical_pn_subdivision_levels",
)


def _dice(mob, camera, device, *, legacy, repeats):
    original = primitives_module._scatter_diced_rows
    if legacy:
        primitives_module._scatter_diced_rows = _legacy_scatter
    try:
        best = float("inf")
        captured = None
        for _ in range(repeats):
            primitive = LogicalPNTrianglePrimitive(
                triangle_collection=[mob.get_render_primitives()]
            )
            primitive.corners = primitive.corners.float().to(device)
            primitive.normals = primitive.normals.float().to(device)
            primitive.colors = primitive.colors.float().to(device)
            for name in primitive._surface_params:
                value = getattr(primitive, name, None)
                if torch.is_tensor(value):
                    setattr(primitive, name, value.to(device))
            if primitive.uvs is not None:
                primitive.uvs = primitive.uvs.to(device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            primitive._dice_logical_pn(camera)
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - start)
            captured = {name: getattr(primitive, name) for name in _CAPTURED}
        return captured, best
    finally:
        primitives_module._scatter_diced_rows = original


def report(name, mob, z_positions, *, repeats=3):
    device = torch.device("cuda")
    camera = make_camera(z_positions, device=device)

    _dice(mob, camera, device, legacy=False, repeats=1)  # warm
    legacy, legacy_time = _dice(mob, camera, device, legacy=True, repeats=repeats)
    fused, fused_time = _dice(mob, camera, device, legacy=False, repeats=repeats)

    mismatched = [key for key in _CAPTURED if not torch.equal(legacy[key], fused[key])]
    status = "byte-identical" if not mismatched else f"DIFFERS: {mismatched}"
    triangles = int(legacy["corners"].shape[0] * legacy["corners"].shape[1])
    print(
        f"{name}: {triangles} diced rows -- "
        f"index_put_ {legacy_time * 1000:.1f} ms -> "
        f"index_copy_ {fused_time * 1000:.1f} ms "
        f"({legacy_time / max(fused_time, 1e-9):.2f}x)  [{status}]"
    )
    return legacy_time, fused_time


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA render device")

    totals = [0.0, 0.0]
    coarse = {"geometry_tolerance": 0.2, "max_grid_resolution": 12}
    for name, mob, zs in (
        ("Sphere", Sphere(), [-3.0, -6.0, -12.0, -25.0, -50.0]),
        ("Torus", Torus(), [-3.0, -6.0, -12.0, -25.0, -50.0]),
        ("Sphere (coarse grid)", Sphere(**coarse), [-2.2, -3.0, -6.0, -12.0, -25.0]),
        ("Torus (coarse grid)", Torus(**coarse), [-2.5, -3.0, -6.0, -12.0, -25.0]),
    ):
        a, b = report(name, mob, zs)
        totals[0] += a
        totals[1] += b
    print(
        f"TOTAL dice time: index_put_ {totals[0] * 1000:.1f} ms -> "
        f"index_copy_ {totals[1] * 1000:.1f} ms "
        f"({totals[0] / max(totals[1], 1e-9):.2f}x)"
    )
