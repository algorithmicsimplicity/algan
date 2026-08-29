"""A/B the paired triangle-side construction in ``compute_grid_vertex_normals``.

That function is **59.8% of `get_render_primitives_batched`**, which the
2026-08-16 profile makes the largest single item in a render (85.35 s, 21.9%) --
see P10 in ``DESIGN_optimization_targets.md``.

It used to build the four triangles around each vertex from an eight-wide stack:
each of the four rolled neighbours appears twice in the pairing, so the stack
materialized ``[..., W, H, 8, 3]`` -- eight copies of the grid -- subtracted the
grid from every one of them, and then sliced two **stride-2** views back out to
cross. The paired form differences each neighbour once and crosses the four
pairs directly.

The paired arm has since taken two further passes over the same block, on the
same terms (P10): each side is written straight into its output buffer instead
of through a materialized ``roll``, and the four crossed pairs are accumulated
in place instead of through three temporaries. Both are covered by the same
comparison -- this script A/Bs *whatever the paired arm currently is* against
the legacy stacked form, so the REAL rows move when either changes.

Every operation involved is elementwise on the same values in the same order, so
the result must be **bit-identical**, not merely close. This script asserts that
on bit patterns rather than on ``==``, so a signed-zero flip or a changed NaN
payload cannot pass, and then times both arms.

Cases cover what the surrounding code branches on: open grids, x-closed
(cylinder) and y-closed seams, both pole configurations (sphere), degenerate and
single-column grids, a batched ``[N, T, W, H, 3]`` stack like the one the batched
build actually passes, plus float64, NaN and signed zeros.

    .venv/Scripts/python.exe benchmarks/_grid_normals_ab.py
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import algan.animation_timeline.timeline as tl  # noqa: E402
from algan.mobs.surfaces.surface import compute_grid_vertex_normals  # noqa: E402
from algan.rendering.taichi_runtime import _sync_devices  # noqa: E402

DEVICE = os.environ.get("ALGAN_AB_DEVICE", "cpu")
ROUNDS = int(os.environ.get("AB_ROUNDS", "5"))


def _arm(paired):
    # ``cpukernels`` is always disabled here. This script compares the two
    # *torch* forms of the block, and on a CPU arch the fused kernel would
    # otherwise serve both arms -- reporting 1.00x and passing the bit-identity
    # assertion vacuously. The kernel has its own A/B in
    # ``benchmarks/_cpu_prep_kernels_ab.py``.
    disabled = {"cpukernels"} if paired else {"cpukernels", "gridnormals"}
    tl._OPT_DISABLED = frozenset(disabled)


def _bits(t):
    """Bit pattern of a float tensor, so -0.0 != 0.0 and NaN payloads compare."""
    if t.dtype == torch.float64:
        return t.detach().contiguous().view(torch.int64)
    if t.dtype == torch.float32:
        return t.detach().contiguous().view(torch.int32)
    return t.detach().contiguous()


def _sphere_grid(w, h, frames=1, dtype=torch.float32):
    u = torch.linspace(0, 1, w, dtype=dtype, device=DEVICE)
    v = torch.linspace(0, 1, h, dtype=dtype, device=DEVICE)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    theta = uu * 2 * torch.pi
    phi = vv * torch.pi
    g = torch.stack((phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos()), -1)
    return g.unsqueeze(0).expand(frames, -1, -1, -1).contiguous()


def _cylinder_grid(w, h, frames=1, dtype=torch.float32):
    u = torch.linspace(0, 1, w, dtype=dtype, device=DEVICE)
    v = torch.linspace(-1, 1, h, dtype=dtype, device=DEVICE)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    theta = uu * 2 * torch.pi
    g = torch.stack((theta.cos(), theta.sin(), vv), -1)
    return g.unsqueeze(0).expand(frames, -1, -1, -1).contiguous()


def _plane_grid(w, h, frames=1, dtype=torch.float32):
    u = torch.linspace(-1, 1, w, dtype=dtype, device=DEVICE)
    v = torch.linspace(-1, 1, h, dtype=dtype, device=DEVICE)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    g = torch.stack((uu, vv, (uu * vv).sin()), -1)
    return g.unsqueeze(0).expand(frames, -1, -1, -1).contiguous()


def cases():
    yield "sphere 32x16, 4 frames", _sphere_grid(32, 16, 4)
    yield "sphere 64x32, 1 frame", _sphere_grid(64, 32, 1)
    yield "cylinder 48x24 (x-closed)", _cylinder_grid(48, 24, 3)
    yield "plane 40x40 (open)", _plane_grid(40, 40, 3)
    yield "single column 8x1", _plane_grid(8, 1, 2)
    yield "two rows 8x2", _plane_grid(8, 2, 2)
    yield "float64 sphere 24x12", _sphere_grid(24, 12, 2, dtype=torch.float64)

    # The shapes the batched build actually passes: a stack of same-shaped
    # surfaces, [N, T, W, H, 3]. The reference scene runs ~18.6 surfaces per
    # call over ~50-frame windows, which is the row to read -- the small cases
    # above are dispatch-bound and say nothing about it.
    yield (
        "REAL [19, 50, 24, 12, 3]",
        torch.stack([_sphere_grid(24, 12, 50) for _ in range(19)]),
    )
    yield (
        "REAL [19, 50, 40, 20, 3]",
        torch.stack([_sphere_grid(40, 20, 50) for _ in range(19)]),
    )
    yield (
        "batched stack [6, 5, 24, 12, 3]",
        torch.stack([_sphere_grid(24, 12, 5) for _ in range(6)]),
    )

    degenerate = _plane_grid(16, 16, 2).clone()
    degenerate[..., 3, :, :] = degenerate[..., 4, :, :]  # collapsed column
    yield "collapsed column", degenerate

    nan = _sphere_grid(16, 8, 2).clone()
    nan[..., 2, 3, :] = float("nan")
    nan[..., 5, 2, 1] = float("inf")
    yield "NaN + inf", nan

    signed = _plane_grid(12, 12, 1).clone()
    signed[..., 0] = signed[..., 0] * 0.0  # produces both +0.0 and -0.0
    yield "signed zeros", signed


def main():
    failures = 0
    print(f"device={DEVICE}\n")
    print(f"{'case':<34}{'bitwise':>10}{'old ms':>10}{'new ms':>10}{'speedup':>10}")
    for name, grid in cases():
        _arm(False)
        old = compute_grid_vertex_normals(grid)
        _arm(True)
        new = compute_grid_vertex_normals(grid)

        same = old.shape == new.shape and bool((_bits(old) == _bits(new)).all())
        if not same:
            failures += 1

        # Alternate the arms round by round and take medians: a single
        # back-to-back pair on this machine swings further than the effect (the
        # design doc's standing warning about wall-clock A/Bs here).
        def timed(paired, iters, g=grid):
            _arm(paired)
            _sync_devices()
            t0 = time.perf_counter()
            for _ in range(iters):
                compute_grid_vertex_normals(g)
            _sync_devices()
            return (time.perf_counter() - t0) / iters * 1000.0

        iters = max(3, min(30, int(2e7 // max(1, grid.numel()))))
        for warm in (False, True):
            timed(warm, 2)
        old_runs, new_runs = [], []
        for _ in range(ROUNDS):
            old_runs.append(timed(False, iters))
            new_runs.append(timed(True, iters))
        t_old = statistics.median(old_runs)
        t_new = statistics.median(new_runs)
        print(
            f"{name:<34}{'OK' if same else 'DIFFER':>10}"
            f"{t_old:>10.2f}{t_new:>10.2f}{t_old / max(t_new, 1e-9):>9.2f}x"
        )
        if not same:
            diff = (old.float() - new.float()).abs()
            print(f"    peak |old-new| = {float(diff.max()):.3e}")

    _arm(True)
    if failures:
        print(f"\nFAIL: {failures} case(s) are not bit-identical")
        return 1
    print("\nOK: every case is bit-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
