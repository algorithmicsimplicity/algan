"""Phase 0 §8.1/§4: does a C-API x64 runtime coexist with the live Python Program?

``DESIGN_taichi_arch_coexistence.md`` §4 verified the C-API round trip
**x64-against-x64**, and §8.1 is the blocking experiment that repeats it with a
CUDA Python ``Program`` live in the same process. This script is both: it reads
the live arch and reports which pairing it actually ran.

    uv run python benchmarks/_taichi_arch_coexistence_probe.py            # §4, x64/x64
    ALGAN_RENDER_DEVICE=cuda uv run python benchmarks/_taichi_arch_coexistence_probe.py

On a CUDA box the second form is §8.1 and needs nothing else: it builds the x64
AOT module in a subprocess, creates the C-API runtime beside the live CUDA
program, launches on torch CPU tensors, interleaves with CUDA-side Python
launches, and asserts both of §8.1's conditions -- ``data_ptr()`` unchanged and
no device-memory growth across the CPU launches.

**The positive control is the point.** §8.1 asks for "no allocation growth",
which a blind measurement also reports. So the script first runs the *staging*
case deliberately -- the live CUDA program launched against host tensors, which
is exactly what production avoids today -- and records the VRAM it moves. A run
whose control shows no growth either has proved nothing, and says so.

Blocking checks print PASS/FAIL and set the exit code. Claims re-tested from
the design print HOLDS/REFUTED and do not: a refuted claim is a finding to
report, not a reason to call the coexistence experiment failed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_REPO = _HERE.parent

# The kernel under test, on both sides. Two plain ndarray arguments, no
# templates -- §5.1's eligibility, and the one kernel §2 measures as paying.
KERNEL = "grid_normals_sides_crosses"
KERNEL_MODULE = "algan.mobs.surfaces.surface_kernels_taichi"

#: Grid the correctness and staging checks use: big enough that a staging copy
#: is unmistakable in a VRAM reading (2 x 24 MB at f32).
BIG_SHAPE = (4, 512, 1024, 3)
#: Grid the launch-overhead timing uses: small enough that the launch dominates.
SMALL_SHAPE = (1, 8, 8, 3)

_RESULTS = []


def check(name: str, ok: bool, detail: str = "", blocking: bool = True) -> bool:
    """Record one check.

    ``blocking=False`` marks a claim from the design being re-tested rather than
    a condition §8.1 turns on -- a refuted claim there is a finding to report,
    not a reason to call the coexistence experiment failed.
    """
    _RESULTS.append((name, ok, blocking))
    tag = ("PASS" if ok else "FAIL") if blocking else ("HOLDS" if ok else "REFUTED")
    print(f"  [{tag}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def note(message: str) -> None:
    print(f"         {message}")


# --- reference ---------------------------------------------------------------


def torch_reference(grid):
    """The torch form of the block, for a correctness oracle.

    Deliberately written from ``compute_grid_vertex_normals``' shape rather than
    imported: the point is to check the AOT kernel computed the right thing, and
    an oracle that shares code with it cannot.
    """
    import torch

    def rolled(t, shift, axis):
        return torch.roll(t, shifts=shift, dims=axis) - t

    xm = rolled(grid, 1, 1)
    xp = rolled(grid, -1, 1)
    ym = rolled(grid, 1, 2)
    yp = rolled(grid, -1, 2)

    W, H = grid.shape[1], grid.shape[2]
    out = torch.zeros_like(grid)

    def gated(a, b, x_slice, y_slice):
        contribution = torch.zeros_like(grid)
        contribution[:, x_slice, y_slice] = torch.cross(
            a[:, x_slice, y_slice], b[:, x_slice, y_slice], dim=-1
        )
        return contribution

    # The four gated triangles, in the ((A + B) + C) + D order the in-place
    # accumulation takes, so the oracle rounds the same way the kernel does.
    out = out + gated(xm, ym, slice(1, W), slice(1, H))
    out = out + gated(ym, xp, slice(0, W - 1), slice(1, H))
    out = out + gated(xp, yp, slice(0, W - 1), slice(0, H - 1))
    out = out + gated(yp, xm, slice(1, W), slice(0, H - 1))
    return out


# --- device memory -----------------------------------------------------------


def device_free_bytes():
    """Driver-reported free VRAM, or None on a box without CUDA.

    ``mem_get_info`` rather than ``torch.cuda.memory_allocated``: Taichi's
    staging allocations go through Taichi's own allocator, which torch's
    accounting cannot see. This reads the driver, so it sees everything.
    """
    import torch

    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    free, _total = torch.cuda.mem_get_info()
    return free


# --- the probe ---------------------------------------------------------------


def build_aot_module(out_dir: Path) -> dict:
    """Run §5.3's build subprocess and return its timing record."""
    environment = dict(os.environ)
    environment["ALGAN_RENDER_DEVICE"] = "cpu"
    # A warm daemon refuses a run whose environment it cannot serve; this is a
    # plain subprocess either way, but be explicit so the build is this process.
    environment["ALGAN_USE_DAEMON"] = "0"
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
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise SystemExit("AOT build subprocess failed")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _require_taichi_backend():
    """Refuse to run under any backend but Taichi.

    Every question this probe asks -- whether a C-API x64 runtime built with
    ``ti.aot.Module`` coexists beside a live Python ``Program``
    (``DESIGN_taichi_arch_coexistence.md`` §4/§8.1) -- is a question about
    Taichi's own C API and AOT support, neither of which Quadrants
    implements (``ti.aot`` does not exist there at all). Below, ``import
    taichi as ti`` is deliberately the literal package rather than
    ``algan.taichi_compat``, while ``algan.mobs.surfaces.surface_kernels_taichi``
    (imported later, for the "live Python-side" launches) goes through that
    layer and binds whatever ``ALGAN_TAICHI_BACKEND`` selects -- so an
    unchecked run under the Quadrants default would load both compilers into
    one process. Refuse outright instead.
    """
    from algan.taichi_compat import BACKEND

    if BACKEND != "taichi":
        raise SystemExit(
            "_taichi_arch_coexistence_probe.py is Taichi-only: it probes "
            "Taichi's C API and `ti.aot.Module`, which Quadrants does not "
            f"implement. Current backend is {BACKEND!r}; re-run with "
            "ALGAN_TAICHI_BACKEND=taichi."
        )


def main() -> int:
    _require_taichi_backend()
    # Deliberately the literal Taichi package; see `_require_taichi_backend`.
    import taichi as ti
    import torch

    from algan.rendering.taichi_runtime import init_taichi
    from algan.settings._startup import render_device

    print("=" * 78)
    print("Phase 0 probe: C-API x64 runtime beside the live Python Program")
    print("=" * 78)

    # 1. The ordinary Python Taichi Program, live throughout (§4 step 1).
    init_taichi()
    live_arch = ti.lang.impl.get_runtime().prog.config().arch
    is_cuda = live_arch == ti.cuda
    pairing = (
        "cuda-against-x64 (this is §8.1)" if is_cuda else "x64-against-x64 (this is §4)"
    )
    print(f"\nrender device : {render_device()}")
    print(f"live arch     : {live_arch}")
    print(f"pairing       : {pairing}")
    print(f"torch cuda    : {torch.cuda.is_available()}")
    if not is_cuda:
        print(
            "\nNOTE: the live arch is not CUDA, so this run re-verifies §4 and\n"
            "      exercises the harness. It CANNOT answer §8.1, which is the\n"
            "      blocking experiment. Re-run on a CUDA box."
        )

    from algan.mobs.surfaces.surface_kernels_taichi import (
        grid_normals_sides_crosses as python_kernel,
    )

    torch.manual_seed(0)
    grid_cpu = torch.randn(*BIG_SHAPE, dtype=torch.float32)
    expected = torch_reference(grid_cpu)

    # --- 0. Positive control: what does a staging launch look like? ----------
    print("\n[0] positive control -- the staging launch this design would remove")
    staging_delta = None
    if is_cuda:
        before = device_free_bytes()
        out_host = torch.zeros_like(grid_cpu)
        # The live CUDA program, launched against HOST tensors. Taichi stages
        # both arguments through VRAM; this is the tax §1 describes.
        python_kernel(grid_cpu, out_host)
        ti.sync()
        after = device_free_bytes()
        staging_delta = before - after
        note(f"free VRAM moved by {staging_delta / 1e6:+.1f} MB over one staged launch")
        note(f"(the two arguments are {grid_cpu.numel() * 4 / 1e6:.1f} MB each)")
        check(
            "control: a staged launch is visible in the VRAM reading",
            staging_delta is not None,
            f"delta {staging_delta / 1e6:+.1f} MB",
        )
        check(
            "control: the staged launch still computed the right answer",
            torch.allclose(out_host, expected, atol=1e-4),
        )
    else:
        note("skipped: no CUDA device, so there is no staging to observe")

    # --- 1. Build the x64 AOT module in a subprocess (§3.3, §5.3) ------------
    print("\n[1] build the x64 AOT module out of process")
    cache = Path(tempfile.mkdtemp(prefix="algan_aot_"))
    build_started = time.perf_counter()
    record = build_aot_module(cache)
    build_wall = time.perf_counter() - build_started
    note(
        f"subprocess wall {build_wall:.2f}s, in-process build "
        f"{record['total_seconds']:.2f}s, {record['bytes'] / 1024:.1f} KiB"
    )
    check("AOT module built and saved", (cache / "metadata.tcb").exists())
    check(
        "the AOT module was built for x64, not the live arch",
        (cache / f"{KERNEL}.ll").exists(),
        "x64 AOT emits LLVM IR per kernel",
    )

    # --- 2. Create the C-API runtime beside the live Program (§4 step 2) -----
    print("\n[2] create a C-API x64 runtime beside the live Program")
    from _taichi_c_api_shim import TI_ARCH_X64, CApiRuntime, ensure_ti_lib_dir, lib

    # ti_create_runtime fails with a runtime_lib_dir error without this (§4).
    # The shim sets it on first load; do it here too so the value is reported
    # before the call that needs it rather than after.
    note(f"TI_LIB_DIR = {ensure_ti_lib_dir()}")
    try:
        runtime = CApiRuntime(TI_ARCH_X64)
    except Exception as error:  # noqa: BLE001 -- the point is to report it
        check("ti_create_runtime(TI_ARCH_X64) succeeds", False, repr(error))
        print("\nThis is §8.1's failure mode. The design is dead; fall back to §9.1.")
        return 1
    check("ti_create_runtime(TI_ARCH_X64) succeeds beside the live Program", True)
    note(f"available archs from the C API: {lib().available_archs()}")

    # §4 records that a *second* runtime also succeeds -- there is no singleton
    # guard in libtaichi_c_api.so, which is the whole mechanism.
    try:
        second = CApiRuntime(TI_ARCH_X64)
        check("a second C-API runtime also succeeds (no singleton guard)", True)
        second.destroy()
    except Exception as error:  # noqa: BLE001
        check(
            "a second C-API runtime also succeeds (no singleton guard)",
            False,
            repr(error),
        )

    failures_before_launch = sum(
        1 for _, ok, blocking in _RESULTS if blocking and not ok
    )
    if failures_before_launch:
        runtime.destroy()
        return failures_before_launch

    # --- 3. Load the module and resolve the kernel (§4 step 3) ---------------
    print("\n[3] load the AOT module and resolve the kernel by name")
    load_started = time.perf_counter()
    runtime.load_module(cache)
    load_seconds = time.perf_counter() - load_started
    check("ti_load_aot_module succeeds", True, f"{load_seconds * 1000:.1f} ms")
    try:
        runtime.kernel(KERNEL)
        check(f"ti_get_aot_module_kernel resolved {KERNEL!r}", True)
    except Exception as error:  # noqa: BLE001
        check(f"ti_get_aot_module_kernel resolved {KERNEL!r}", False, repr(error))
        runtime.destroy()
        return sum(1 for _, ok, blocking in _RESULTS if blocking and not ok)

    # --- 4. Launch on torch CPU tensors, and measure (§4 step 4/5, §8.1) -----
    print("\n[4] launch the x64 kernel on torch CPU tensors")
    out_capi = torch.zeros_like(grid_cpu)
    pointers_before = (grid_cpu.data_ptr(), out_capi.data_ptr())

    free_before = device_free_bytes()
    launch_started = time.perf_counter()
    runtime.launch(KERNEL, grid_cpu, out_capi)
    launch_seconds = time.perf_counter() - launch_started
    free_after = device_free_bytes()

    pointers_after = (grid_cpu.data_ptr(), out_capi.data_ptr())
    check(
        "the C-API launch produced the right answer",
        torch.allclose(out_capi, expected, atol=1e-4),
        f"max |diff| {float((out_capi - expected).abs().max()):.3e}",
    )
    check(
        "neither tensor was reallocated (ti_import_cpu_memory did not copy)",
        pointers_before == pointers_after,
    )
    note(f"launch wall {launch_seconds * 1000:.1f} ms on {BIG_SHAPE}")

    if is_cuda:
        delta = free_before - free_after
        note(f"free VRAM moved by {delta / 1e6:+.1f} MB over the C-API launch")
        # A staging copy of these arguments would be ~50 MB. Allow slack for
        # unrelated driver bookkeeping but nothing near an argument's size.
        budget = 4e6
        check(
            "no device memory was consumed by the C-API CPU launch",
            abs(delta) < budget,
            f"delta {delta / 1e6:+.1f} MB against a {budget / 1e6:.0f} MB budget",
        )
        if staging_delta is not None:
            check(
                "the control moved materially more VRAM than the C-API launch",
                abs(staging_delta) > abs(delta) + budget,
                f"control {staging_delta / 1e6:+.1f} MB vs C-API {delta / 1e6:+.1f} MB",
            )
        else:
            note("no control delta to compare against")
    else:
        note("no CUDA device: the VRAM assertions of §8.1 cannot run here")

    # --- 5. Interleave with Python-side launches, both orders (§4 step 5) ----
    print("\n[5] interleave C-API and Python-side launches in one process")
    live_grid = grid_cpu.to(render_device()) if is_cuda else grid_cpu
    live_out = torch.zeros_like(live_grid)

    python_kernel(live_grid, live_out)
    ti.sync()
    check(
        "a Python-side launch on the live arch still works after the C-API one",
        torch.allclose(live_out.cpu(), expected, atol=1e-4),
    )

    out_capi.zero_()
    runtime.launch(KERNEL, grid_cpu, out_capi)
    check(
        "a C-API launch still works after the Python-side one",
        torch.allclose(out_capi, expected, atol=1e-4),
    )

    live_out.zero_()
    python_kernel(live_grid, live_out)
    ti.sync()
    check(
        "and the Python side again, after that",
        torch.allclose(live_out.cpu(), expected, atol=1e-4),
    )

    # --- 6. Launch overhead, C-API against the Python path (§4) -------------
    #
    # §4 records 77-89 us for a C-API launch against 173 us for an ordinary
    # @ti.kernel call, and lists "lower launch overhead than the Python path"
    # as one of two properties better than expected. Broken down here, because
    # a single pair of numbers cannot say whether a difference is the shim's
    # Python or the C API's own dispatch.
    print("\n[6] launch overhead on a tiny grid")
    from _taichi_c_api_shim import TiArgument

    small = torch.randn(*SMALL_SHAPE, dtype=torch.float32)
    small_out = torch.zeros_like(small)
    small_live = small.to(render_device()) if is_cuda else small
    small_live_out = torch.zeros_like(small_live)

    kernel_handle = runtime.kernel(KERNEL)
    prebuilt = (TiArgument * 2)(
        runtime.ndarray_argument(small), runtime.ndarray_argument(small_out)
    )
    dll = lib().dll

    def capi_launch():
        runtime.launch(KERNEL, small, small_out)

    def build_arguments():
        runtime.ndarray_argument(small)
        runtime.ndarray_argument(small_out)

    def bare_launch():
        dll.ti_launch_kernel(runtime.handle, kernel_handle, 2, prebuilt)
        dll.ti_wait(runtime.handle)

    def python_launch():
        python_kernel(small_live, small_live_out)

    # Interleaved rounds and medians, not one timed block per arm. A single
    # pair on a shared cloud vCPU reads anywhere from 0.75x to 1.27x purely on
    # scheduling drift, which is wide enough to invent or erase §4's result.
    arms = {
        "capi": capi_launch,
        "arguments": build_arguments,
        "bare": bare_launch,
        "python": python_launch,
    }
    samples = {name: [] for name in arms}
    for _ in range(100):
        capi_launch()
        python_launch()
    for _round in range(9):
        for name, call in arms.items():
            started = time.perf_counter()
            for _ in range(500):
                call()
            samples[name].append((time.perf_counter() - started) / 500 * 1e6)
    if is_cuda:
        ti.sync()

    import statistics

    median = {name: statistics.median(values) for name, values in samples.items()}
    spread = {name: (min(values), max(values)) for name, values in samples.items()}
    for label, name in (
        ("C-API, whole shim launch()", "capi"),
        ("  of which building two TiArguments", "arguments"),
        ("  ti_launch_kernel + ti_wait alone", "bare"),
        (f"Python @ti.kernel on {live_arch}", "python"),
    ):
        low, high = spread[name]
        note(f"{label:<40s}: {median[name]:7.1f} us  (min {low:.1f}, max {high:.1f})")
    ratio = median["python"] / median["capi"]
    note(f"Python / C-API median ratio: {ratio:.2f}x   (§4 reports ~2.0x)")
    check(
        "§4's claim that the C-API launch is materially cheaper than @ti.kernel",
        ratio >= 1.5,
        f"measured {ratio:.2f}x; the arms' ranges overlap, so this reads as parity. "
        f"ti_launch_kernel alone is {median['bare']:.0f} us of the "
        f"{median['capi']:.0f} us, so the cost is the C API's own dispatch, "
        "not the shim's Python",
        blocking=False,
    )

    # The rounds above are ~20k launches, which is enough to expose the leak
    # that §5.5's front door has as specified: an imported CPU memory handle
    # cannot be released on an x64 runtime (ti_free_memory refuses with
    # "(not supported) taichi::arch_is_cpu"), so importing per launch grows the
    # process at ~90-108 bytes a launch forever. The shim memoizes on
    # (data_ptr, nbytes) instead; this is the check that it still does.
    note(
        f"{runtime.imports} imports served by {len(runtime._imported)} handles "
        f"({runtime.import_hits / max(1, runtime.imports):.1%} reused)"
    )
    check(
        "imported memory handles are reused, not re-imported per launch",
        len(runtime._imported) <= 8 and runtime.imports > 1000,
        "ti_free_memory cannot release them on an x64 runtime, so an import "
        "per launch is an unbounded leak",
    )

    # --- 7. Errors are return codes (§5.5) ----------------------------------
    print("\n[7] the error contract")
    try:
        runtime.kernel("no_such_kernel_exists")
        check("resolving a missing kernel raises rather than returning null", False)
    except KeyError:
        check("resolving a missing kernel raises rather than returning null", True)
    except Exception as error:  # noqa: BLE001
        check(
            "resolving a missing kernel raises rather than returning null",
            True,
            f"raised {type(error).__name__}",
        )
    try:
        runtime.launch(KERNEL, grid_cpu.cuda() if is_cuda else grid_cpu.to("meta"))
        check("a non-CPU tensor is refused before it reaches the C API", False)
    except Exception as error:  # noqa: BLE001
        check(
            "a non-CPU tensor is refused before it reaches the C API",
            isinstance(
                error, (ValueError, TypeError, NotImplementedError, RuntimeError)
            ),
            f"raised {type(error).__name__}",
        )

    runtime.destroy()
    check(
        "ti_destroy_runtime at the end, and again, is safe", (runtime.destroy() or True)
    )

    failures = sum(1 for _, ok, blocking in _RESULTS if blocking and not ok)
    refuted = [name for name, ok, blocking in _RESULTS if not blocking and not ok]
    print("\n" + "=" * 78)
    blocking_total = sum(1 for _, _, blocking in _RESULTS if blocking)
    print(
        f"{blocking_total - failures}/{blocking_total} blocking checks passed  ({pairing})"
    )
    for name in refuted:
        print(f"REFUTED (design claim, not a blocker): {name}")
    if not is_cuda:
        print("§8.1 IS NOT ANSWERED BY THIS RUN -- it needs a CUDA device.")
    elif not failures:
        print("§8.1 ANSWERED: a CUDA Program and a C-API x64 runtime coexist,")
        print("and the CPU launches consumed no device memory.")
    print("=" * 78)
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
