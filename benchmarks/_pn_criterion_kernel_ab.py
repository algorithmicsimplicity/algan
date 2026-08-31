"""A/B the fused logical PN subdivision-level criterion against the torch path.

Reports, per scene, how the levels the two paths choose differ and how long each
search takes. The kernels are deliberately *not* byte-identical (Taichi runs
with fast_math), so a handful of borderline patches landing one level apart is
expected; a large or systematic shift is not.

Also checks the property the dice actually depends on: every boundary curve
shared by two patches must be handed the same level by both, or the mesh cracks
open along the seam.

    .venv/Scripts/python.exe benchmarks/_pn_criterion_kernel_ab.py
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import torch

import algan  # noqa: F401  -- sets up devices / inference mode
from algan.mobs.shapes_3d import Sphere, Torus
from algan.rendering.logical_pn import (
    logical_pn_control_points,
    logical_pn_edge_control_points,
)
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.primitives import LogicalPNTrianglePrimitive
from algan.rendering.raytracing.utils import _expand_frames, _flat_frames
from algan.rendering.taichi_runtime import _sync_devices
from algan.settings._startup import render_device


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


def _animate(corners, normals, num_frames):
    """Spin the mesh, giving it genuinely per-frame geometry.

    A static mesh reaches the kernels as a stride-0 broadcast view of one frame;
    a moving one arrives as a real ``[T, P, 3, 3]`` array and takes the other
    branch of ``_frame_broadcast_base``. Both need testing -- every scene that
    matters animates.
    """
    angles = torch.linspace(0.0, 1.2, num_frames, device=corners.device)
    cos = angles.cos().view(-1, 1, 1)
    sin = angles.sin().view(-1, 1, 1)

    def spin(values):
        x, y, z = values[0].unbind(-1)
        return torch.stack(
            (x * cos - z * sin, y.expand_as(cos * x), x * sin + z * cos), dim=-1
        )

    return spin(corners), spin(normals)


def _search_inputs(primitive, camera, device, moving=False):
    """The exact arguments ``_dice_logical_pn`` hands the level search.

    ``device`` is the render device: a real render has already uploaded the
    primitive's source geometry there (``upload_primitive_source``) by the time
    the dice runs, and the kernel path is only taken for CUDA tensors, so a
    harness that leaves the mesh on the animation device measures torch twice.
    """
    num_frames = int(camera.ray_origin.shape[0])
    corners = primitive.corners.float().to(device)
    normals = primitive.normals.float().to(device)
    if moving:
        corners, normals = _animate(corners, normals, num_frames)
    cam_o = _expand_frames(_flat_frames(camera.ray_origin, (3,)), num_frames).to(device)
    sp = _expand_frames(_flat_frames(camera.screen_point, (3,)), num_frames).to(device)
    sb = _expand_frames(_flat_frames(camera.screen_basis, (3, 3)), num_frames).to(
        device
    )
    control_points = primitive._expanded_frames(
        logical_pn_control_points(corners, normals), num_frames, "corners"
    )
    edge_controls = primitive._expanded_frames(
        logical_pn_edge_control_points(corners, normals), num_frames, "edges"
    )
    height = getattr(camera, "output_screen_height", camera.screen_height)
    return control_points, edge_controls, cam_o, sp, sb, height


def _assert_kernel_reachable(inputs):
    """Fail loudly rather than measuring the torch path twice."""
    from algan.rendering.raytracing.primitives import _pn_criterion_inputs

    control_points, edge_controls, cam_o, sp, sb, _ = inputs
    front_sign = torch.sign(((sp - cam_o) * sb[:, 2]).sum(-1))
    rt_settings.set_pn_criterion_kernel(True)
    if (
        _pn_criterion_inputs(control_points, edge_controls, cam_o, sp, sb, front_sign)
        is None
    ):
        raise SystemExit("kernel path unreachable for these inputs; nothing to A/B")


def _timed_search(primitive, inputs, *, use_kernel, repeats):
    rt_settings.set_pn_criterion_kernel(use_kernel)
    levels = edge_levels = None
    best = float("inf")
    for _ in range(repeats):
        # _sync_devices syncs Taichi too: on a CPU arch the kernel is a Taichi
        # launch that torch knows nothing about, and timing it without the
        # Taichi barrier would measure the dispatch rather than the work.
        _sync_devices()
        start = time.perf_counter()
        # Four values since per-dimension dicing landed: the trailing
        # (apex, across) pair is the row direction and the across level. This
        # script unpacked two until 2026-08-26 and had been broken by that
        # change for as long as it took to notice -- it refused to run without
        # CUDA, so nothing ever executed the line.
        levels, edge_levels, _apex, _across = primitive._required_subdivision_levels(
            *inputs
        )
        _sync_devices()
        best = min(best, time.perf_counter() - start)
    return levels, edge_levels, best


def report(name, mob, z_positions, *, repeats=3, moving=False):
    device = render_device()
    primitive = LogicalPNTrianglePrimitive(
        triangle_collection=[mob.get_render_primitives()]
    )
    camera = make_camera(z_positions, device=device)
    inputs = _search_inputs(primitive, camera, device, moving=moving)
    _assert_kernel_reachable(inputs)
    stride = "per-frame" if inputs[0].stride(0) else "broadcast"
    name = f"{name} [{stride}]"

    # Warm the kernels (first launch pays JIT/cache lookup) before timing.
    _timed_search(primitive, inputs, use_kernel=True, repeats=1)

    torch_levels, torch_edges, torch_time = _timed_search(
        primitive, inputs, use_kernel=False, repeats=repeats
    )
    kernel_levels, kernel_edges, kernel_time = _timed_search(
        primitive, inputs, use_kernel=True, repeats=repeats
    )
    rt_settings.set_pn_criterion_kernel(True)

    level_diff = (kernel_levels != torch_levels).sum().item()
    edge_diff = (kernel_edges != torch_edges).sum().item()
    level_span = kernel_levels - torch_levels
    edge_span = kernel_edges - torch_edges
    torch_tris = int((4**torch_levels).sum())
    kernel_tris = int((4**kernel_levels).sum())

    print(f"{name}: {torch_levels.shape[1]} patches x {torch_levels.shape[0]} frames")
    print(
        f"  patch levels differ : {level_diff} / {torch_levels.numel()} "
        f"({100.0 * level_diff / max(torch_levels.numel(), 1):.3f}%), "
        f"range {int(level_span.amin())}..{int(level_span.amax())}"
    )
    print(
        f"  edge  levels differ : {edge_diff} / {torch_edges.numel()} "
        f"({100.0 * edge_diff / max(torch_edges.numel(), 1):.3f}%), "
        f"range {int(edge_span.amin())}..{int(edge_span.amax())}"
    )
    print(
        f"  diced triangles     : torch {torch_tris} -> kernel {kernel_tris} "
        f"({kernel_tris / max(torch_tris, 1):.4f}x)"
    )
    print(
        f"  search time         : torch {torch_time * 1000:.1f} ms -> "
        f"kernel {kernel_time * 1000:.1f} ms "
        f"({torch_time / max(kernel_time, 1e-9):.2f}x)"
    )
    print(
        f"  patch >= edge level : {bool((kernel_levels.unsqueeze(-1) >= kernel_edges).all())}"
    )
    print()
    return torch_time, kernel_time


def shared_edge_check(mob, z_positions):
    """Every boundary curve two patches share must get one level, from both.

    Patches are matched by their canonical edge controls: two patches sharing a
    curve produce a bit-identical control tuple (that is what
    ``logical_pn_edge_control_points`` is for), so grouping every (frame, patch,
    edge) by those twelve floats recovers the adjacency without any topology.
    """
    device = render_device()
    primitive = LogicalPNTrianglePrimitive(
        triangle_collection=[mob.get_render_primitives()]
    )
    camera = make_camera(z_positions, device=device)
    inputs = _search_inputs(primitive, camera, device)
    rt_settings.set_pn_criterion_kernel(True)
    _, edge_levels, _apex, _across = primitive._required_subdivision_levels(*inputs)

    edge_controls = inputs[1]
    frames = edge_levels.shape[0]
    keys = edge_controls.reshape(frames, -1, 12)
    flat_levels = edge_levels.reshape(frames, -1)
    mismatched = 0
    shared = 0
    for frame in range(frames):
        unique, inverse = torch.unique(keys[frame], dim=0, return_inverse=True)
        counts = torch.zeros(
            unique.shape[0], dtype=torch.long, device=inverse.device
        ).index_add_(0, inverse, torch.ones_like(inverse))
        lo = torch.full(
            (unique.shape[0],), 1 << 30, dtype=torch.long, device=inverse.device
        ).scatter_reduce_(0, inverse, flat_levels[frame], reduce="amin")
        hi = torch.full(
            (unique.shape[0],), -1, dtype=torch.long, device=inverse.device
        ).scatter_reduce_(0, inverse, flat_levels[frame], reduce="amax")
        shared += int((counts > 1).sum())
        mismatched += int(((counts > 1) & (lo != hi)).sum())
    status = "OK" if mismatched == 0 else "CRACK RISK"
    print(
        f"shared-edge levels: {shared} shared curves, {mismatched} disagreeing [{status}]"
    )
    return mismatched


if __name__ == "__main__":
    # Runs on whatever device Algan selected. The kernels used to be reachable
    # only on CUDA, so this script refused anything else; since the criterion
    # gate became "the tensors are already on Taichi's arch device" it is also
    # reachable on a CPU render, and the same A/B answers the same questions
    # there. _assert_kernel_reachable is what catches an unreachable kernel now,
    # and it fails loudly rather than measuring torch against itself.

    totals = [0.0, 0.0]
    coarse = {"geometry_tolerance": 0.2, "max_grid_resolution": 12}
    for name, mob, zs, moving in (
        ("Sphere", Sphere(), [-3.0, -6.0, -12.0, -25.0, -50.0], False),
        ("Torus", Torus(), [-3.0, -6.0, -12.0, -25.0, -50.0], False),
        (
            "Sphere (coarse grid)",
            Sphere(**coarse),
            [-2.2, -3.0, -6.0, -12.0, -25.0],
            False,
        ),
        (
            "Torus (coarse grid)",
            Torus(**coarse),
            [-3.0, -6.0, -12.0, -25.0, -50.0],
            False,
        ),
        (
            "Sphere (tight tolerance)",
            Sphere(render_tolerance_pixels=50.0, **coarse),
            [-2.2, -3.0, -6.0],
            False,
        ),
        # The stride-1 branch: real per-frame geometry, as every animated scene
        # produces. A static-only harness never reaches it.
        ("Sphere (moving)", Sphere(), [-3.0, -6.0, -12.0, -25.0, -50.0], True),
        ("Torus (moving)", Torus(**coarse), [-2.5, -3.0, -6.0, -12.0, -25.0], True),
    ):
        a, b = report(name, mob, zs, moving=moving)
        totals[0] += a
        totals[1] += b
    print(
        f"TOTAL search time: torch {totals[0] * 1000:.1f} ms -> "
        f"kernel {totals[1] * 1000:.1f} ms "
        f"({totals[0] / max(totals[1], 1e-9):.2f}x)"
    )
    print()
    shared_edge_check(Sphere(**coarse), [-2.2, -3.0, -6.0, -12.0, -25.0])
    shared_edge_check(Torus(**coarse), [-3.0, -6.0, -12.0])
