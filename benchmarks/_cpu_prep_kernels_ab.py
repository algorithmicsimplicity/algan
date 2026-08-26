"""A/B the three CPU batch-prep kernels against the torch paths they replace.

Each arm goes through the *shipped* dispatch, flipped with
``ALGAN_OPT_DISABLE=cpukernels``, so this measures what a render actually runs
rather than a re-implementation. Shapes come from
``DESIGN_optimization_targets.md``'s P10/P10b split of
``get_render_primitives_batched``, whose rows these are:

* ``compute_grid_vertex_normals``' sides and crosses -- ~35% of the stage.
* ``grid_to_triangle_vertices`` -- ~20% across its two call sites.
* ``TrianglePrimitive``'s vertex-colour bake -- 13.5%, the per-surface tail's
  largest row.

Two of the three are byte-identical and are asserted so here. The normals
kernel is not, by design (``surface_kernels_taichi`` records why and why it
cannot open a seam); it is compared in ulps instead.

    ALGAN_RENDER_DEVICE=cpu ALGAN_USE_DAEMON=0 uv run python benchmarks/_cpu_prep_kernels_ab.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _memory_cap import cap_process_memory  # noqa: E402

# Shapes are parameters here, not a real scene.
cap_process_memory(float(os.environ.get("ALGAN_PREP_AB_MEM_GB", "10")))

import torch  # noqa: E402

import algan.animation_timeline.timeline as tl  # noqa: E402
import algan.rendering.taichi_runtime as taichi_runtime  # noqa: E402
from algan.mobs.surfaces.surface import (  # noqa: E402
    compute_grid_vertex_normals,
    get_grid_to_triangle_indices,
    grid_to_triangle_vertices,
)
from algan.rendering.primitives.triangle_primitive import (  # noqa: E402
    _bake_glow_and_opacity,
)
from algan.rendering.taichi_runtime import init_taichi, taichi_arch_is_cpu  # noqa: E402
from algan.utils.tensor_utils import broadcast_all  # noqa: E402

init_taichi()

ROUNDS = int(os.environ.get("AB_ROUNDS", "5"))


def _arm(kernels):
    tl._OPT_DISABLED = frozenset(() if kernels else ("cpukernels",))
    # cpugather and cpucolors ship off by default (they measured slower than
    # torch -- which is what this script is for), so the kernel arm has to opt
    # them in explicitly or it would silently measure torch against torch.
    taichi_runtime._OPT_ENABLED = frozenset(("cpugather", "cpucolors"))


def _bits(t):
    return t.detach().contiguous().view(torch.int32)


def _ulps(a, b):
    ia, ib = _bits(a).to(torch.int64), _bits(b).to(torch.int64)
    flip = -(1 << 31)
    ia = torch.where(ia < 0, flip - ia, ia)
    ib = torch.where(ib < 0, flip - ib, ib)
    return (ia - ib).abs()


def _sphere_grid(w, h, frames):
    u = torch.linspace(0, 2 * torch.pi, w)
    v = torch.linspace(0, torch.pi, h)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    g = torch.stack(
        (vv.sin() * uu.cos(), vv.sin() * uu.sin(), vv.cos().expand_as(uu)), -1
    )
    return g.unsqueeze(0).expand(frames, -1, -1, -1).contiguous()


def _stack(w, h, frames, surfaces):
    return torch.stack([_sphere_grid(w, h, frames) for _ in range(surfaces)])


def time_arms(run, iters):
    """Median of ROUNDS alternating measurements per arm.

    Alternating rather than blocking: a few cores and a few hundred MB per
    result make back-to-back blocks drift further than the effect.
    """
    for warm in (False, True):
        _arm(warm)
        run()

    def once(kernels):
        _arm(kernels)
        start = time.perf_counter()
        for _ in range(iters):
            run()
        return (time.perf_counter() - start) / iters * 1000.0

    torch_runs, kernel_runs = [], []
    for _ in range(ROUNDS):
        torch_runs.append(once(False))
        kernel_runs.append(once(True))
    return statistics.median(torch_runs), statistics.median(kernel_runs)


def report(name, run, iters, exact):
    _arm(False)
    reference = run()
    _arm(True)
    fused = run()

    if exact:
        verdict = "identical" if torch.equal(reference, fused) else "DIFFERS"
    else:
        distance = _ulps(reference, fused)
        moved = int((distance != 0).sum())
        verdict = f"{moved / reference.numel():.1%} @ {int(distance.max())} ulp"
    del reference, fused

    t_torch, t_kernel = time_arms(run, iters)
    print(
        f"  {name:<40}{t_torch:>9.1f}{t_kernel:>9.1f}"
        f"{t_torch / max(t_kernel, 1e-9):>8.2f}x   {verdict}"
    )


def main():
    if not taichi_arch_is_cpu():
        print("Taichi is not on a CPU arch; the prep kernels never dispatch here.")
        return 1

    print(f"{'row':<42}{'torch ms':>9}{'ti ms':>9}{'speedup':>8}   parity")

    print("\ncompute_grid_vertex_normals (sides + crosses):")
    for surfaces, frames, w, h in (
        (19, 50, 24, 12),
        (19, 50, 40, 20),
        (120, 50, 24, 12),
    ):
        grid = _stack(w, h, frames, surfaces)
        report(
            f"[{surfaces}, {frames}, {w}, {h}, 3]",
            lambda g=grid: compute_grid_vertex_normals(g),
            3,
            exact=False,
        )
        del grid

    print("\ngrid_to_triangle_vertices:")
    for surfaces, frames, w, h in ((19, 50, 24, 12), (19, 50, 40, 20)):
        grid = _stack(w, h, frames, surfaces)
        for weld in ((False, False, False), (True, True, True)):
            report(
                f"[{surfaces}, {frames}, {w}, {h}, 3] weld={int(any(weld))}",
                lambda g=grid, wd=weld: grid_to_triangle_vertices(g, weld=wd),
                3,
                exact=True,
            )
        del grid

    print("\nTrianglePrimitive vertex-colour bake:")
    for frames, triangles in ((50, 100_000), (50, 500_000)):
        generator = torch.Generator().manual_seed(0)
        raw_colors = torch.rand(frames, triangles, 5, generator=generator)
        raw_opacity = torch.rand(1, 1, 1, generator=generator)
        raw_glow = torch.rand(1, triangles, 1, generator=generator)
        colors, opacity, glow = broadcast_all(
            [raw_colors, raw_opacity, raw_glow], ignored_dims=[-1]
        )
        report(
            f"[{frames}, {triangles}, 5]",
            lambda c=colors, o=opacity, g=glow: _bake_glow_and_opacity(c, o, g),
            3,
            exact=True,
        )
        del raw_colors, raw_opacity, raw_glow, colors, opacity, glow

    # The gather's index tables are cached per (shape, device, weld); drop them
    # so a later run in the same process does not inherit this one's.
    get_grid_to_triangle_indices.__globals__["_grid_triangle_indices_cache"].clear()
    _arm(True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
