"""A/B the fused bezier chord-count search against the torch path.

Same shape as ``_pn_criterion_kernel_ab.py``: the kernel is not byte-identical
(Taichi runs with fast_math), so a handful of borderline segments landing on a
neighbouring chord count is expected; a large or systematic shift is not. The
count is what the packed polyline geometry is built from, so a difference here
is a geometry difference -- inside ``num_pixels_per_sample`` by construction.

    .venv/Scripts/python.exe benchmarks/_bez_chord_kernel_ab.py
"""

from __future__ import annotations

import time

import torch

from algan import *  # noqa: F403
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.primitives import (
    RayTracedBezierCircuitPrimitive,
    _bezier_criterion_inputs,
)
from algan.rendering.raytracing.utils import _expand_frames, _flat_frames
from algan.rendering.taichi_runtime import sync_devices
from algan.settings._startup import _RENDER_DEVICE


def make_camera(z_positions, *, screen_height=1080, device=None):
    z = torch.as_tensor(z_positions, dtype=torch.float32, device=device)
    origins = torch.zeros((len(z), 1, 3), device=device)
    origins[..., 2] = z.view(-1, 1)
    screen_points = origins.clone()
    screen_points[..., 2] += 1.0
    return (
        _expand_frames(_flat_frames(origins, (3,)), len(z)),
        _expand_frames(_flat_frames(screen_points, (3,)), len(z)),
        _expand_frames(
            _flat_frames(
                torch.eye(3, device=device).unsqueeze(0).repeat(len(z), 1, 1), (3, 3)
            ),
            len(z),
        ),
        screen_height,
    )


def _timed(primitive, args, *, use_kernel, repeats):
    rt_settings.set_pn_criterion_kernel(use_kernel)
    counts = None
    best = float("inf")
    for _ in range(repeats):
        sync_devices()
        start = time.perf_counter()
        counts = primitive._compute_samples_per_segment(*args)
        sync_devices()
        best = min(best, time.perf_counter() - start)
    return counts, best


def _circuit_primitives(mob):
    """Every bezier circuit under ``mob``, as the renderer would collect them.

    ``Text``/``Tex`` are groups of per-glyph ``BezierCircuitCubic`` children, so
    the primitive list has to be gathered from the descendants rather than from
    the group itself.
    """
    sources = [mob] if hasattr(mob, "get_render_primitives") else []
    sources += [
        descendant
        for descendant in mob.get_descendants()
        if hasattr(descendant, "get_render_primitives")
    ]
    collected = []
    for source in sources:
        primitive = source.get_render_primitives()
        if primitive is None:
            continue
        collected.extend(primitive if isinstance(primitive, list) else [primitive])
    return collected


def report(name, mob, z_positions, *, repeats=3):
    device = torch.device(_RENDER_DEVICE)
    primitive = RayTracedBezierCircuitPrimitive(
        triangle_collection=_circuit_primitives(mob)
    )
    corners = primitive.corners.float().contiguous().to(device)
    cam_o, sp, sb, screen_h = make_camera(z_positions, device=device)
    args = (corners, cam_o, sp, sb, screen_h, False)

    if _bezier_criterion_inputs(corners, cam_o, sp, sb) is None:
        raise SystemExit("kernel path unreachable for these inputs; nothing to A/B")

    _timed(primitive, args, use_kernel=True, repeats=1)  # warm the kernel
    torch_counts, torch_time = _timed(
        primitive, args, use_kernel=False, repeats=repeats
    )
    kernel_counts, kernel_time = _timed(
        primitive, args, use_kernel=True, repeats=repeats
    )
    rt_settings.set_pn_criterion_kernel(True)

    differ = int((torch_counts != kernel_counts).sum())
    span = kernel_counts.float() / torch_counts.clamp_min(1).float()
    print(f"{name}: {torch_counts.numel()} segments x {len(z_positions)} frames")
    print(
        f"  chord counts differ : {differ} / {torch_counts.numel()} "
        f"({100.0 * differ / max(torch_counts.numel(), 1):.3f}%), "
        f"ratio {float(span.min()):.3f}..{float(span.max()):.3f}"
    )
    print(
        f"  total chords        : torch {int(torch_counts.sum())} -> "
        f"kernel {int(kernel_counts.sum())} "
        f"({int(kernel_counts.sum()) / max(int(torch_counts.sum()), 1):.4f}x)"
    )
    print(
        f"  search time         : torch {torch_time * 1000:.1f} ms -> "
        f"kernel {kernel_time * 1000:.1f} ms "
        f"({torch_time / max(kernel_time, 1e-9):.2f}x)"
    )
    print()
    return torch_time, kernel_time


if __name__ == "__main__":
    # See _pn_criterion_kernel_ab.py: the criterion gate is no longer CUDA-only,
    # so this A/B runs wherever Algan's render device is.

    totals = [0.0, 0.0]
    frames = [-3.0, -4.5, -6.0, -9.0, -14.0, -25.0, -40.0, -60.0]
    for name, mob in (
        ("Text (short)", Text("Chord counts per segment").scale(0.6)),  # noqa: F405
        ("Text (paragraph)", Text(("the quick brown fox " * 4 + "\n") * 8).scale(0.3)),  # noqa: F405
        ("Tex", Tex(r"\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}").scale(1.5)),  # noqa: F405
        (
            "Circle grid",
            Group(
                *[
                    Circle().scale(0.2).move(RIGHT * (i % 12 - 6) + UP * (i // 12 - 4))
                    for i in range(96)
                ]
            ),
        ),  # noqa: F405
    ):
        a, b = report(name, mob, frames)
        totals[0] += a
        totals[1] += b
    print(
        f"TOTAL search time: torch {totals[0] * 1000:.1f} ms -> "
        f"kernel {totals[1] * 1000:.1f} ms "
        f"({totals[0] / max(totals[1], 1e-9):.2f}x)"
    )
