"""Phase 0 §8.3: what does the AOT build step cost, cold and warm?

    uv run python benchmarks/_taichi_aot_build_cost.py

§5.3 proposes running the build lazily on first need, and §8.3 asks whether
that policy survives contact with the real number: "If a cold build is minutes
rather than seconds, the lazy-on-first-need policy in §5.3 needs rethinking."

Cold here means an **empty Taichi offline cache**, so the subprocess compiles
the kernel for x64 from scratch. Warm means the same cache populated. Both are
measured as subprocess wall time -- what a render would actually wait for --
and split against the build's own JSON timing, because the two halves have very
different fixes: Taichi compilation is unavoidable, Python import time is not.

The last section is beyond §8.3 and is the question that makes the rest matter:
**does a kernel reached through the AOT/C-API path still deliver §2's speedup?**
A build that is cheap and a kernel that lost its win would be worthless, and
nothing in §8 asks. Measured against torch on the shapes the batched surface
build passes.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_REPO = _HERE.parent

KERNEL = "grid_normals_sides_crosses"
KERNEL_MODULE = "algan.mobs.surfaces.surface_kernels_taichi"

#: Shapes the batched surface build passes, per ``_grid_normals_kernel_ab.py``.
AB_SHAPES = ((4, 64, 64, 3), (4, 256, 256, 3), (4, 512, 1024, 3))


def run_build(out_dir: Path, cache_dir: Path) -> tuple[float, dict]:
    """Run the build subprocess against ``cache_dir`` and time its wall clock."""
    environment = dict(os.environ)
    environment["ALGAN_RENDER_DEVICE"] = "cpu"
    environment["ALGAN_USE_DAEMON"] = "0"
    # Taichi honours this over the ti.init kwarg, which is exactly what a cold
    # measurement needs: an offline cache nothing else has populated.
    environment["TI_OFFLINE_CACHE_FILE_PATH"] = str(cache_dir)
    started = time.perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            str(_HERE / "_taichi_aot_build.py"),
            "--out",
            str(out_dir),
            "--kernels",
            f"{KERNEL_MODULE}:{KERNEL}",
        ],
        cwd=str(_REPO),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    wall = time.perf_counter() - started
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise SystemExit("AOT build subprocess failed")
    return wall, json.loads(completed.stdout.strip().splitlines()[-1])


def measure_build_cost() -> None:
    print("=" * 78)
    print("§8.3  AOT build cost")
    print("=" * 78)

    scratch = Path(tempfile.mkdtemp(prefix="algan_aot_cost_"))
    cache = scratch / "ti_cache"
    rows = []
    try:
        # Cold: no offline cache at all, so Taichi compiles the kernel for x64.
        cold_wall, cold = run_build(scratch / "cold", cache)
        rows.append(("cold (empty offline cache)", cold_wall, cold))

        # Warm: the same cache, so the compile is a cache load.
        warm_wall, warm = run_build(scratch / "warm", cache)
        rows.append(("warm (populated cache)", warm_wall, warm))

        # And again, to see whether "warm" is stable or still settling.
        warm2_wall, warm2 = run_build(scratch / "warm2", cache)
        rows.append(("warm, second time", warm2_wall, warm2))

        print(f"\ncache: {cache}")
        print(
            f"\n{'arm':<28s} {'wall':>8s} {'in-proc':>9s} {'compile':>9s} {'import etc':>11s}"
        )
        for label, wall, record in rows:
            overhead = wall - record["total_seconds"]
            print(
                f"{label:<28s} {wall:7.2f}s {record['total_seconds']:8.2f}s "
                f"{record['compile_seconds']:8.2f}s {overhead:10.2f}s"
            )
        print(
            f"\nartifact: {rows[0][2]['bytes'] / 1024:.1f} KiB for "
            f"{len(rows[0][2]['kernels'])} kernel(s)"
        )

        cold_compile = rows[0][2]["compile_seconds"]
        warm_compile = rows[1][2]["compile_seconds"]
        print(
            f"\nTaichi compilation is {cold_compile:.2f}s cold and "
            f"{warm_compile:.2f}s warm."
        )
        print(
            f"Interpreter + `import algan` is {rows[0][1] - rows[0][2]['total_seconds']:.2f}s "
            "and is the same either way -- it dominates the wall clock."
        )
        verdict = (
            "SECONDS, not minutes: §5.3's lazy-on-first-need policy stands."
            if cold_wall < 60
            else "MINUTES: §5.3's lazy-on-first-need policy needs rethinking."
        )
        print(f"\n§8.3 verdict: cold build is {cold_wall:.1f}s. {verdict}")

        measure_load_cost(scratch / "cold")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def measure_load_cost(module_dir: Path) -> None:
    """Time ``ti_load_aot_module`` from a built directory."""
    from _taichi_c_api_shim import TI_ARCH_X64, CApiRuntime

    print("\n" + "-" * 78)
    print("load cost from a warm artifact (§8.3's second half)")

    samples = []
    for _ in range(5):
        runtime = CApiRuntime(TI_ARCH_X64)
        started = time.perf_counter()
        runtime.load_module(module_dir)
        runtime.kernel(KERNEL)
        samples.append((time.perf_counter() - started) * 1000)
        runtime.destroy()
    print(
        f"  ti_load_aot_module + resolve: median {statistics.median(samples):.1f} ms "
        f"(min {min(samples):.1f}, max {max(samples):.1f}) over {len(samples)} runs"
    )

    started = time.perf_counter()
    runtime = CApiRuntime(TI_ARCH_X64)
    create_ms = (time.perf_counter() - started) * 1000
    runtime.destroy()
    print(f"  ti_create_runtime(x64):       {create_ms:.1f} ms, once per process")


def measure_speedup() -> None:
    """Does the AOT/C-API path keep the win the Python launch measures?"""
    import torch
    from _taichi_arch_coexistence_probe import torch_reference
    from _taichi_c_api_shim import TI_ARCH_X64, CApiRuntime

    from algan.mobs.surfaces.surface_kernels_taichi import (
        grid_normals_sides_crosses as python_kernel,
    )
    from algan.rendering.taichi_runtime import init_taichi

    print("\n" + "=" * 78)
    print("beyond §8.3: does the AOT path keep §2's speedup?")
    print("=" * 78)

    init_taichi()
    scratch = Path(tempfile.mkdtemp(prefix="algan_aot_ab_"))
    try:
        run_build(scratch / "module", scratch / "cache")
        runtime = CApiRuntime(TI_ARCH_X64)
        runtime.load_module(scratch / "module")

        print(
            f"\n{'shape':<20s} {'torch':>10s} {'python-ti':>12s} {'c-api aot':>12s}"
            f" {'py x':>7s} {'aot x':>7s}"
        )

        def timed(call, repeats):
            for _ in range(3):
                call()
            # Median of rounds rather than one mean: this box is a shared
            # 4-vCPU container and a single block reads whatever the
            # scheduler was doing.
            rounds = []
            for _ in range(5):
                started = time.perf_counter()
                for _ in range(repeats):
                    call()
                rounds.append((time.perf_counter() - started) / repeats * 1e3)
            return statistics.median(rounds)

        for shape in AB_SHAPES:
            grid = torch.randn(*shape, dtype=torch.float32)
            out = torch.zeros_like(grid)
            repeats = 30 if grid.numel() < 2e6 else 8

            ms_torch = timed(lambda g=grid: torch_reference(g), repeats)
            ms_python = timed(lambda g=grid, o=out: python_kernel(g, o), repeats)
            ms_capi = timed(lambda g=grid, o=out: runtime.launch(KERNEL, g, o), repeats)
            print(
                f"{str(shape):<20s} {ms_torch:9.2f}ms {ms_python:11.2f}ms "
                f"{ms_capi:11.2f}ms {ms_torch / ms_python:6.2f}x "
                f"{ms_torch / ms_capi:6.2f}x"
            )
        runtime.destroy()
        print(
            "\nIf the last two columns track each other, the AOT path preserves the\n"
            "win and the design's payoff is §2's number. If 'aot x' is materially\n"
            "lower, the mechanism costs part of what it is built to deliver."
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    measure_build_cost()
    measure_speedup()
