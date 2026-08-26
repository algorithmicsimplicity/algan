"""A/B: a fused Taichi kernel vs the shipped torch sides-and-crosses block.

``DESIGN_optimization_targets.md`` P10b ranks that block first among what is
left in ``get_render_primitives_batched`` (~35% of the stage) and notes that the
remaining torch-side win -- collapsing the four cross products with
``cross(xm,ym) + ... = cross(xm - xp, ym - yp)`` -- costs bit-identity. This
measures whether a kernel gets the win *without* that cost.

Two questions, in order:

1. **Is it bit-identical under Algan's own Taichi configuration?** Not under a
   configuration chosen to make the answer yes: ``fast_math=True`` is what
   ``taichi_init_kwargs`` ships and what every render runs, and it is a property
   of the whole ``ti.init``, so it cannot be turned off for one kernel. Compared
   on bit patterns, so a signed-zero flip or a changed NaN payload cannot pass.
2. **How much faster is it**, on the shapes the batched build passes.

Measured answers: **8.4-11.3x** across two runs on the REAL and DOC rows, and **not**
bit-identical -- open and closed-seam grids match exactly, sphere-derived ones
differ on ~4% of elements by 1-2 ulp. The cause is ``torch.cross``'s rounding on
the cross product's cancelling third component, not Taichi's codegen; the kernel
module docstring records how that was isolated. Mismatches are reported in ulps
rather than absolute error, because at this size the absolute number says
nothing.

The torch arm is not a re-implementation: it calls the same ``_wrapped_difference``
and ``broadcast_cross_product`` the shipped function calls, in the same order,
so it tracks whatever the paired arm currently is. Only the block is compared --
the seam merges, pole fans and normalize that follow it are untouched, and P10b
measures them at ~4% of the function combined.

    ALGAN_RENDER_DEVICE=cpu ALGAN_USE_DAEMON=0 uv run python benchmarks/_grid_normals_kernel_ab.py
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _memory_cap import cap_process_memory  # noqa: E402

# The 120-surface row below is sized from a parameter, not from a real scene.
cap_process_memory(float(os.environ.get("ALGAN_GRIDNORM_MEM_GB", "10")))

import torch  # noqa: E402

from algan.rendering.taichi_runtime import init_taichi, sync_devices  # noqa: E402

init_taichi()

from _grid_normals_ab import _bits, _sphere_grid, cases  # noqa: E402
from _grid_normals_kernel_taichi import grid_normals_sides_crosses  # noqa: E402

from algan.mobs.surfaces.surface import _wrapped_difference  # noqa: E402
from algan.utils.tensor_utils import broadcast_cross_product  # noqa: E402

ROUNDS = int(os.environ.get("AB_ROUNDS", "5"))


def _ulp_distance(a, b):
    """Float32 ulps between two tensors, counted across the sign boundary."""
    ia = _bits(a).to(torch.int64)
    ib = _bits(b).to(torch.int64)
    flip = -(1 << 31)
    ia = torch.where(ia < 0, flip - ia, ia)
    ib = torch.where(ib < 0, flip - ib, ib)
    return (ia - ib).abs()


def torch_block(grid):
    """The shipped sides + crosses + accumulate block, verbatim.

    Copied from ``compute_grid_vertex_normals``'s paired arm
    (``surface.py:554-579``) and calling the same helpers, so this arm moves
    when that one does.
    """
    side_x_minus = _wrapped_difference(grid, -3, 1)
    side_y_minus = _wrapped_difference(grid, -2, 1)
    side_x_plus = _wrapped_difference(grid, -3, -1)
    side_y_plus = _wrapped_difference(grid, -2, -1)
    normals_xm_ym = broadcast_cross_product(side_x_minus, side_y_minus)
    normals_ym_xp = broadcast_cross_product(side_y_minus, side_x_plus)
    normals_xp_yp = broadcast_cross_product(side_x_plus, side_y_plus)
    normals_yp_xm = broadcast_cross_product(side_y_plus, side_x_minus)
    normals_xm_ym[..., 0, :, :] = 0
    normals_yp_xm[..., 0, :, :] = 0
    normals_ym_xp[..., -1, :, :] = 0
    normals_xp_yp[..., -1, :, :] = 0
    normals_xm_ym[..., :, 0, :] = 0
    normals_ym_xp[..., :, 0, :] = 0
    normals_xp_yp[..., :, -1, :] = 0
    normals_yp_xm[..., :, -1, :] = 0
    unnormalized = normals_xm_ym + normals_ym_xp
    unnormalized += normals_xp_yp
    unnormalized += normals_yp_xm
    return unnormalized


def kernel_block(grid):
    """Same result, one pass. Leading dims are flattened into the batch axis."""
    W, H = grid.shape[-3], grid.shape[-2]
    flat = grid.reshape(-1, W, H, 3)
    out = torch.empty_like(flat)
    grid_normals_sides_crosses(flat, out)
    return out.reshape(grid.shape)


def bench_cases():
    """The shared fixtures, plus the shape the design doc quotes its ratios at."""
    for name, grid in cases():
        if grid.dtype != torch.float32:
            print(f"{name:<34}{'SKIPPED (float64 fixture)':>10}")
            continue
        yield name, grid
    yield (
        "DOC [120, 50, 24, 12, 3]",
        torch.stack([_sphere_grid(24, 12, 50) for _ in range(120)]),
    )


def main():
    failures = 0
    print(f"{'case':<34}{'bitwise':>10}{'torch ms':>10}{'ti ms':>10}{'speedup':>10}")
    for name, grid in bench_cases():
        grid = grid.contiguous()
        reference = torch_block(grid)
        fused = kernel_block(grid)

        same = reference.shape == fused.shape and bool(
            (_bits(reference) == _bits(fused)).all()
        )
        if not same:
            failures += 1

        def timed(fn, iters, g=grid):
            sync_devices()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn(g)
            sync_devices()
            return (time.perf_counter() - t0) / iters * 1000.0

        iters = max(3, min(30, int(2e7 // max(1, grid.numel()))))
        timed(torch_block, 2)
        timed(kernel_block, 2)
        torch_runs, ti_runs = [], []
        for _ in range(ROUNDS):
            torch_runs.append(timed(torch_block, iters))
            ti_runs.append(timed(kernel_block, iters))
        t_torch = statistics.median(torch_runs)
        t_ti = statistics.median(ti_runs)
        print(
            f"{name:<34}{'OK' if same else 'DIFFER':>10}"
            f"{t_torch:>10.2f}{t_ti:>10.2f}{t_torch / max(t_ti, 1e-9):>9.2f}x"
        )
        if not same:
            distance = _ulp_distance(reference, fused)
            n_diff = int((distance != 0).sum())
            print(
                f"    {n_diff} of {reference.numel()} elements differ "
                f"({n_diff / reference.numel():.2%}), max {int(distance.max())} ulp"
            )
        del reference, fused, grid

    if failures:
        print(f"\n{failures} case(s) are NOT bit-identical")
        return 1
    print("\nevery case is bit-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
