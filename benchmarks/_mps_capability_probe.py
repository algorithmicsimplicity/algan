"""Is an MPS render device viable? Six questions, answered on real Apple hardware.

Algan pins the macOS CI job to ``ALGAN_RENDER_DEVICE=cpu`` because an MPS render
fails in two families -- torch cannot make the raster pipeline's ``float64``
tensors, and Taichi's SPIR-V codegen (which serves **both** the Metal and the
Vulkan backends) answers "Type f64 not supported" to the same kernels. Removing
f64 is bounded work. Whether it is *worth* doing depends on facts that only an
Apple GPU can supply, and this script is how they get supplied.

    uv run python benchmarks/_mps_capability_probe.py

Every question is decided by measurement or by a compile that either succeeds or
does not. Nothing here is argued from source.

1. **Which Taichi backend does Algan actually get?** ``_taichi_arch()`` returns
   ``ti.gpu`` off CUDA, and ``ti.gpu`` is the preference list
   ``[cuda, metal, vulkan, ...]`` -- metal *ahead* of vulkan, so the workflow
   comment's "resolves through Vulkan" is a guess worth checking.
2. **What can that backend do?** Read from Taichi's own trace log
   (``DeviceCapability::...``), and -- more usefully -- from whether a kernel
   needing each capability compiles.
3. **Do the 64-bit integer kernels compile there?** The raster pipeline is built
   on 64-bit fragment keys (``(depth << 32) | layer``) and
   ``sheet_compact_taichi`` does ``ti.atomic_min`` on i64. Metal has 64-bit
   ints; 64-bit *atomics* are a different question. A no here means the port is
   a raster redesign rather than a dtype change.
4. **Do the wide kernels bind?** ``sheet_resolve_shade`` takes 49 ndarray
   arguments, ``wavefront_shade`` 38. Metal's classic per-stage buffer limit is
   31. A ladder finds where it actually breaks.
5. **What does the interop cost?** Taichi 1.7.4 has no torch-MPS import: an
   ndarray argument that is neither a host tensor nor a CUDA tensor on a CUDA
   arch is copied to the host before the launch and copied back after
   (``taichi/lang/kernel_impl.py``). This times that round trip against the CPU
   arch, on a bandwidth-bound kernel and a compute-bound one -- because the
   answer may differ, and if it does, "MPS for path-traced scenes only" is a
   real option.
6. **Which torch ops does MPS actually implement?** The raster path leans on
   ``scatter_reduce_`` amin/amax over int64, ``unique_consecutive``,
   ``bincount``, ``cummax``, 64-bit shifts and dtype re-views. Run with
   ``PYTORCH_ENABLE_MPS_FALLBACK`` cleared, so a gap is an exception rather than
   a silent host round trip.

A seventh section answers a question the first six raised: **how
non-deterministic is a float32 reduction on MPS?**  ``raster_pipeline``
accumulates coverage in f64 because a float atomic's summation order is not
reproducible and the sum feeds a *threshold* -- two consecutive CUDA renders of
``materials_and_lighting`` differed by up to 28 channel values over 9.6% of a
frame before it. If MPS is to run in a deliberately non-deterministic mode, the
size of that wobble is the thing to know, and it is measured here beside a
fixed-point i64 accumulator, which is the deterministic replacement that needs
no f64 at all.

Each arm runs in its **own subprocess**. A backend that cannot compile a kernel
does not always raise -- it can abort the process -- and an aborted arm has to
be a recorded result rather than the end of the run. Sections print one
``ALGAN_PROBE_JSON <json>`` line; the orchestrator collects them, writes
``mps_probe_results.json`` beside itself, and prints the report.
"""

import argparse
import json
import linecache
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import taichi as ti

_REPO = Path(__file__).resolve().parents[1]

#: Printed by every section, parsed by the orchestrator. A marker rather than
#: bare stdout because Taichi writes its banner and any compile warnings to the
#: same stream.
_MARKER = "ALGAN_PROBE_JSON "


#: Written *before* an arm does the thing that might abort the process, so the
#: orchestrator learns which backend a crashed arm was on. Without it a crash
#: reports only that it crashed, and "Metal refused this" is indistinguishable
#: from "the arch was never moved off the CPU" -- which is exactly the ambiguity
#: the first run of this probe came back with.
_PRE_MARKER = "ALGAN_PROBE_PRE "


def _emit(payload):
    """Hand one section's result back to the orchestrator."""
    sys.stdout.write(_MARKER + json.dumps(payload) + "\n")
    sys.stdout.flush()


def _emit_pre(payload):
    """Report what is already known, before anything that can abort."""
    sys.stdout.write(_PRE_MARKER + json.dumps(payload) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Kernels under test.
#
# Defined at module scope, which is safe before ``ti.init``: the decorator only
# registers a kernel, and nothing is compiled until a launch supplies argument
# types (this is the same property that lets Algan's kernel modules import
# before the render device is known). So a module carrying an f64 kernel still
# imports cleanly on a backend that has no f64 -- the failure lands on the arm
# that launches it, which is where it is wanted.
# ---------------------------------------------------------------------------


@ti.kernel
def _k_f32_basic(src: ti.types.ndarray(), dst: ti.types.ndarray(), n: ti.i32):
    for i in range(n):
        dst[i] = src[i] * 2.0 + 1.0


@ti.kernel
def _k_f32_atomic_add(src: ti.types.ndarray(), acc: ti.types.ndarray(), n: ti.i32):
    for i in range(n):
        ti.atomic_add(acc[0], src[i])


@ti.kernel
def _k_i32_atomic_minmax(src: ti.types.ndarray(), out: ti.types.ndarray(), n: ti.i32):
    for i in range(n):
        ti.atomic_min(out[0], src[i])
        ti.atomic_max(out[1], src[i])


@ti.kernel
def _k_i64_keys(depth: ti.types.ndarray(), out: ti.types.ndarray(), n: ti.i32):
    """The raster pipeline's key format: pack, then unpack (``raster_taichi``)."""
    for i in range(n):
        d = ti.cast(depth[i], ti.u64)
        layer = ti.cast(i, ti.u64)
        key = ti.cast((d << 32) | (ti.u64(0xFFFFFFFF) - layer), ti.i64)
        out[i] = ti.cast(ti.cast(key, ti.u64) >> 32, ti.i64)


@ti.kernel
def _k_i64_atomic_add(src: ti.types.ndarray(), acc: ti.types.ndarray(), n: ti.i32):
    for i in range(n):
        ti.atomic_add(acc[0], ti.cast(src[i], ti.i64))


@ti.kernel
def _k_i64_atomic_min(src: ti.types.ndarray(), out: ti.types.ndarray(), n: ti.i32):
    for i in range(n):
        ti.atomic_min(out[0], ti.cast(src[i], ti.i64))


@ti.kernel
def _k_i64_fixed_point(cov: ti.types.ndarray(), acc: ti.types.ndarray(), n: ti.i32):
    """The deterministic replacement for the f64 coverage sums.

    Coverage is in [0, 1], so a Q32 integer accumulator is exact and
    order-independent -- reproducible on any backend that has i64 atomics, which
    is a stronger guarantee than the f64 it would replace, not a weaker one.
    """
    for i in range(n):
        q = ti.cast(ti.round(cov[i] * 4294967296.0), ti.i64)
        ti.atomic_add(acc[0], q)


@ti.kernel
def _k_f64_ndarray(src: ti.types.ndarray(), dst: ti.types.ndarray(), n: ti.i32):
    for i in range(n):
        dst[i] = ti.cast(src[i], ti.f64) * ti.f64(2.0)


@ti.kernel
def _k_f64_atomic_add(src: ti.types.ndarray(), acc: ti.types.ndarray(), n: ti.i32):
    for i in range(n):
        ti.atomic_add(acc[0], ti.cast(src[i], ti.f64))


@ti.kernel
def _k_bit_cast(src: ti.types.ndarray(), dst: ti.types.ndarray(), n: ti.i32):
    """``raster_taichi._key_payload``'s f32 <-> u32 round trip."""
    for i in range(n):
        bits = ti.bit_cast(src[i], ti.u32)
        dst[i] = ti.bit_cast(bits, ti.f32)


@ti.kernel
def _k_bandwidth(src: ti.types.ndarray(), dst: ti.types.ndarray(), n: ti.i32):
    """Memory-bound: one load, a little arithmetic, one store. The raster and
    sheet stages are this shape, and they are what a staging copy taxes worst.
    """
    for i in range(n):
        dst[i] = src[i] * 1.0000001 + 0.5


@ti.kernel
def _k_compute(
    src: ti.types.ndarray(), dst: ti.types.ndarray(), n: ti.i32, iters: ti.i32
):
    """Compute-bound: the same two buffers, a lot more arithmetic between them.
    Stands in for the path tracer, whose arithmetic intensity is the one thing
    that could pay for a host round trip.
    """
    for i in range(n):
        x = src[i]
        for _ in range(iters):
            x = x * 1.0000001 + 0.5
            x = ti.sqrt(x * x + 1.0)
        dst[i] = x


# ---------------------------------------------------------------------------
# Section: env
# ---------------------------------------------------------------------------


def _section_env():
    import torch
    from taichi._lib import core as _ti_core

    mps = getattr(torch, "mps", None)
    backends_mps = getattr(torch.backends, "mps", None)
    out = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "taichi": ".".join(str(part) for part in ti.__version__),
        "mps_is_built": bool(backends_mps and backends_mps.is_built()),
        "mps_is_available": bool(backends_mps and backends_mps.is_available()),
    }
    # Recorded because "put both torch and Taichi on Vulkan" is an obvious
    # thing to try. Torch's Vulkan backend is an Android mobile-inference path,
    # not a compute device: the dispatch key exists but the ops are not built,
    # so a tensor cannot even be allocated on it.
    with_suppressed(out, "torch_is_vulkan_available", torch.is_vulkan_available)
    try:
        torch.zeros(4, device="vulkan")
        out["torch_vulkan_tensor"] = "ok"
    except Exception as exc:
        out["torch_vulkan_tensor"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    for name, arch in (("metal", ti.metal), ("vulkan", ti.vulkan), ("cuda", ti.cuda)):
        try:
            out["ti_supports_" + name] = bool(ti.lang.misc.is_arch_supported(arch))
        except Exception as exc:  # pragma: no cover - probe
            out["ti_supports_" + name] = "error: " + repr(exc)
    for name in ("with_metal", "with_vulkan", "with_cuda"):
        try:
            out["ti_" + name] = bool(getattr(_ti_core, name)())
        except Exception as exc:  # pragma: no cover - probe
            out["ti_" + name] = "error: " + repr(exc)
    if out["mps_is_available"] and mps is not None:
        with_suppressed(out, "mps_recommended_max_memory", mps.recommended_max_memory)
        with_suppressed(out, "mps_driver_allocated_memory", mps.driver_allocated_memory)
    return out


def with_suppressed(out, key, fn):
    """Record ``fn()`` under ``key``, or the exception it raised."""
    try:
        out[key] = fn()
    except Exception as exc:  # pragma: no cover - probe
        out[key] = "error: " + repr(exc)


# ---------------------------------------------------------------------------
# Section: arch -- Q1, plus the two device-pairing predicates the engine uses
# ---------------------------------------------------------------------------


def _bring_up(device):
    """Point Algan's render device at ``device`` and start Taichi from it.

    Goes through ``SETTINGS`` and the engine's own entry point rather than
    calling ``ti.init`` with an arch, because *which arch Algan selects* is
    question 1. Forcing a backend is done from the outside with ``TI_ARCH``,
    which ``ti.init`` honours over its own argument.

    ``ensure_taichi_for_render``, not ``init_taichi``, and the difference is
    load-bearing: ``init_taichi`` is a no-op when a program already exists, so
    if anything brought Taichi up before this call -- on whatever
    ``ALGAN_RENDER_DEVICE`` said at import -- the arch would silently stay
    there and every arm would report the wrong backend while claiming the right
    device. ``ensure_taichi_for_render`` re-runs ``ti.init`` when the live arch
    no longer matches the device, which is the behaviour a render gets.

    Returns the diagnostics needed to tell those two cases apart afterwards.
    """
    from algan.rendering import taichi_runtime
    from algan.settings import SETTINGS

    already_up = taichi_runtime._already_initialized()
    SETTINGS.computing.set(render_device=device)
    reinitialized = taichi_runtime.ensure_taichi_for_render()
    return {
        "taichi_was_up_before_bring_up": already_up,
        "taichi_reinitialized": reinitialized,
    }


def _live_arch_name():
    """The running program's arch.

    Read after a launch has materialized a kernel wherever possible: asking
    before anything has been compiled is what made the first run of this probe
    report ``arm64`` for arms whose timings prove they ran on the GPU.
    """
    from taichi._lib import core as _ti_core

    prog = ti.lang.impl.get_runtime().prog
    return _ti_core.arch_name(prog.config().arch)


def _arch_report():
    """Every reading of the arch this process can take, so they can disagree.

    ``prog.config()`` and ``impl.current_cfg()`` are two different paths to the
    same fact, and ``can_use_bloom_taichi`` in the engine trusts the second. A
    probe that reports one number cannot notice them diverging; this one can.
    """
    from taichi._lib import core as _ti_core

    report = {}
    try:
        report["prog_arch"] = _live_arch_name()
    except Exception as exc:  # pragma: no cover - probe
        report["prog_arch"] = "error: " + repr(exc)
    try:
        report["current_cfg_arch"] = _ti_core.arch_name(ti.lang.impl.current_cfg().arch)
    except Exception as exc:  # pragma: no cover - probe
        report["current_cfg_arch"] = "error: " + repr(exc)
    return report


def _warm_up_arch():
    """Launch one trivial kernel so the arch reading is taken from a live
    program rather than from one that has compiled nothing yet.
    """
    import torch

    src = torch.zeros(4, dtype=torch.float32)
    dst = torch.zeros(4, dtype=torch.float32)
    _k_f32_basic(src, dst, 4)
    ti.sync()


def _settled_arch():
    """The arch, read once a kernel has actually been compiled on it.

    Every arm calls this before the launch it exists to test, so the backend is
    known even for an arm that then aborts the process. ``_k_f32_basic`` is the
    warm-up because it is the one kernel already measured to run everywhere.
    """
    out = {"arch_before_launch": _arch_report()}
    try:
        _warm_up_arch()
        out["warm_up"] = "ok"
    except Exception as exc:
        out["warm_up"] = "error: " + repr(exc)[:500]
    out["arch"] = _arch_report()
    out["live_arch"] = out["arch"].get("prog_arch")
    return out


def _section_arch(device):
    import torch

    from algan.rendering import taichi_runtime
    from algan.settings import SETTINGS

    out = dict(_bring_up(device))
    out.update(
        {
            "render_device": str(SETTINGS.computing.render_device),
            "ti_arch_env_override": os.environ.get("TI_ARCH"),
            "arch_before_launch": _arch_report(),
        }
    )
    try:
        _warm_up_arch()
        out["warm_up"] = "ok"
    except Exception as exc:
        out["warm_up"] = "error: " + repr(exc)[:500]
    out["arch_after_launch"] = _arch_report()
    out["live_arch"] = out["arch_after_launch"].get("prog_arch")
    out["taichi_arch_is_cpu"] = taichi_runtime.taichi_arch_is_cpu()
    # ``taichi_launch_is_local`` is the engine's own answer to "does this launch
    # avoid a staging copy". It compares device *types*, so on MPS it says yes
    # while Taichi copies through the host -- record what it claims here so the
    # claim can be compared against section 5's measurement.
    for name in ("cpu", "mps"):
        try:
            out["launch_is_local_" + name] = taichi_runtime.taichi_launch_is_local(
                torch.device(name)
            )
        except Exception as exc:  # pragma: no cover - probe
            out["launch_is_local_" + name] = "error: " + repr(exc)
    return out


# ---------------------------------------------------------------------------
# Section: feature -- Q2 (empirically) and Q3
# ---------------------------------------------------------------------------


def _feature_cases():
    """Each case: build args on the device, launch, and check the answer.

    Checking the *value* matters as much as the compile. A backend that lowers
    i64 to something narrower would compile and return the wrong key, and a
    probe that only asked "did it raise" would report that as support.
    """
    import torch

    def f32_basic(dev):
        src = torch.arange(64, dtype=torch.float32, device=dev)
        dst = torch.zeros(64, dtype=torch.float32, device=dev)
        _k_f32_basic(src, dst, 64)
        return bool(torch.allclose(dst.cpu(), src.cpu() * 2 + 1))

    def f32_atomic_add(dev):
        src = torch.ones(4096, dtype=torch.float32, device=dev)
        acc = torch.zeros(1, dtype=torch.float32, device=dev)
        _k_f32_atomic_add(src, acc, 4096)
        return abs(float(acc.cpu()[0]) - 4096.0) < 1e-3

    def i32_atomic_minmax(dev):
        src = torch.arange(-500, 500, dtype=torch.int32, device=dev)
        out = torch.tensor([1 << 30, -(1 << 30)], dtype=torch.int32, device=dev)
        _k_i32_atomic_minmax(src, out, 1000)
        got = out.cpu().tolist()
        return got == [-500, 499]

    def i64_keys(dev):
        depth = torch.arange(1, 65, dtype=torch.int64, device=dev)
        out = torch.zeros(64, dtype=torch.int64, device=dev)
        _k_i64_keys(depth, out, 64)
        return out.cpu().tolist() == list(range(1, 65))

    def i64_atomic_add(dev):
        src = torch.full((4096,), 3, dtype=torch.int64, device=dev)
        acc = torch.zeros(1, dtype=torch.int64, device=dev)
        _k_i64_atomic_add(src, acc, 4096)
        return int(acc.cpu()[0]) == 3 * 4096

    def i64_atomic_min(dev):
        src = torch.arange(1000, 2000, dtype=torch.int64, device=dev)
        out = torch.full((1,), 1 << 40, dtype=torch.int64, device=dev)
        _k_i64_atomic_min(src, out, 1000)
        return int(out.cpu()[0]) == 1000

    def i64_fixed_point(dev):
        cov = torch.full((1024,), 0.25, dtype=torch.float32, device=dev)
        acc = torch.zeros(1, dtype=torch.int64, device=dev)
        _k_i64_fixed_point(cov, acc, 1024)
        return int(acc.cpu()[0]) == 1024 * (1 << 30)

    def f64_ndarray(dev):
        src = torch.arange(64, dtype=torch.float32, device=dev)
        # The destination has to be f64 for the kernel to mean anything, and on
        # MPS torch refuses to make one -- which is itself the answer, recorded
        # as the exception this arm reports.
        dst = torch.zeros(64, dtype=torch.float64, device=dev)
        _k_f64_ndarray(src, dst, 64)
        return bool(torch.allclose(dst.cpu(), src.cpu().double() * 2))

    def f64_atomic_add(dev):
        src = torch.ones(1024, dtype=torch.float32, device=dev)
        acc = torch.zeros(1, dtype=torch.float64, device=dev)
        _k_f64_atomic_add(src, acc, 1024)
        return abs(float(acc.cpu()[0]) - 1024.0) < 1e-9

    def bit_cast(dev):
        src = torch.linspace(0.5, 2.0, 64, dtype=torch.float32, device=dev)
        dst = torch.zeros(64, dtype=torch.float32, device=dev)
        _k_bit_cast(src, dst, 64)
        return bool(torch.equal(dst.cpu(), src.cpu()))

    return {
        "f32_basic": f32_basic,
        "f32_atomic_add": f32_atomic_add,
        "i32_atomic_minmax": i32_atomic_minmax,
        "i64_keys": i64_keys,
        "i64_atomic_add": i64_atomic_add,
        "i64_atomic_min": i64_atomic_min,
        "i64_fixed_point": i64_fixed_point,
        "f64_ndarray": f64_ndarray,
        "f64_atomic_add": f64_atomic_add,
        "bit_cast": bit_cast,
    }


def _section_feature(device, feature):
    brought_up = _bring_up(device)
    case = _feature_cases()[feature]
    out = {"feature": feature, "device": device, **brought_up}
    out.update(_settled_arch())
    _emit_pre(out)
    started = time.perf_counter()
    try:
        correct = case(device)
        ti.sync()
        out["status"] = "ok" if correct else "wrong_result"
    except Exception as exc:
        out["status"] = "error"
        out["error_type"] = type(exc).__name__
        out["error"] = str(exc)[:2000]
    out["seconds"] = round(time.perf_counter() - started, 3)
    return out


# ---------------------------------------------------------------------------
# Section: args -- Q4
# ---------------------------------------------------------------------------


def _build_wide_kernel(nargs):
    """A kernel taking ``nargs`` ndarray arguments, built at runtime.

    Generated rather than written out because the question is where the binding
    limit is, and that wants a ladder. Compiled with ``exec`` in a namespace of
    its own: this module has no ``from __future__ import annotations`` (see the
    ``benchmarks/*`` I002 ignore in pyproject.toml), so the ``ti.types``
    annotations stay live objects here as they must.
    """
    params = ", ".join(f"a{i}: ti.types.ndarray()" for i in range(nargs))
    body = "\n".join(f"        a{i}[i] = a{i}[i] + {i}.0" for i in range(nargs))
    source = (
        f"@ti.kernel\ndef wide({params}, n: ti.i32):\n"
        + "    for i in range(n):\n"
        + body
        + "\n"
    )
    # Taichi reads a kernel's *source* back with ``inspect`` when it
    # materializes, so a function that exists only as a code object raises
    # "Cannot find source code for Object". Seeding linecache under the same
    # pseudo-filename is what makes ``inspect.getsourcefile`` willing to answer
    # for it, and is enough: nothing here needs the file to exist on disk.
    filename = f"<wide_kernel_{nargs}>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = {"ti": ti}
    exec(compile(source, filename, "exec"), namespace)
    return namespace["wide"]


def _section_args(device, nargs):
    import torch

    brought_up = _bring_up(device)
    out = {"nargs": nargs, "device": device, **brought_up}
    out.update(_settled_arch())
    _emit_pre(out)
    started = time.perf_counter()
    try:
        kernel = _build_wide_kernel(nargs)
        tensors = [
            torch.zeros(16, dtype=torch.float32, device=device) for _ in range(nargs)
        ]
        kernel(*tensors, 16)
        ti.sync()
        expected = [float(i) for i in range(nargs)]
        got = [float(t.cpu()[0]) for t in tensors]
        out["status"] = "ok" if got == expected else "wrong_result"
    except Exception as exc:
        out["status"] = "error"
        out["error_type"] = type(exc).__name__
        out["error"] = str(exc)[:2000]
    out["seconds"] = round(time.perf_counter() - started, 3)
    return out


# ---------------------------------------------------------------------------
# Section: staging -- Q5
# ---------------------------------------------------------------------------

#: 4M f32 in and 4M f32 out: 32 MB of traffic per launch, which is the scale a
#: real chunk of the fragment stream moves and far past any per-launch noise.
_BANDWIDTH_N = 4 << 20
#: Small enough that the inner loop, not the transfer, is the work.
_COMPUTE_N = 1 << 18
_COMPUTE_ITERS = 256
_TIMED_LAUNCHES = 20
_WARMUP_LAUNCHES = 3


def _sync(device):
    import torch

    ti.sync()
    if device == "mps":
        torch.mps.synchronize()


def _time_launches(fn, device):
    """Median wall time of one launch, with the device drained around each."""
    for _ in range(_WARMUP_LAUNCHES):
        fn()
    _sync(device)
    samples = []
    for _ in range(_TIMED_LAUNCHES):
        started = time.perf_counter()
        fn()
        _sync(device)
        samples.append(time.perf_counter() - started)
    samples.sort()
    return {
        "median_ms": round(1000 * samples[len(samples) // 2], 4),
        "min_ms": round(1000 * samples[0], 4),
        "max_ms": round(1000 * samples[-1], 4),
    }


def _section_staging(device, tensor_device, workload):
    import torch

    brought_up = _bring_up(device)
    out = {
        "render_device": device,
        "tensor_device": tensor_device,
        "workload": workload,
        **brought_up,
    }
    try:
        if workload == "bandwidth":
            n = _BANDWIDTH_N
            src = torch.rand(n, dtype=torch.float32, device=tensor_device)
            dst = torch.zeros(n, dtype=torch.float32, device=tensor_device)
            out["bytes_per_launch"] = 2 * 4 * n
            out.update(_time_launches(lambda: _k_bandwidth(src, dst, n), tensor_device))
        elif workload == "compute":
            n = _COMPUTE_N
            src = torch.rand(n, dtype=torch.float32, device=tensor_device)
            dst = torch.zeros(n, dtype=torch.float32, device=tensor_device)
            out["bytes_per_launch"] = 2 * 4 * n
            out["iters"] = _COMPUTE_ITERS
            out.update(
                _time_launches(
                    lambda: _k_compute(src, dst, n, _COMPUTE_ITERS), tensor_device
                )
            )
        elif workload == "torch_only":
            # The control: the same bandwidth-bound arithmetic expressed in
            # torch, so "what would this device do if Taichi were not in the
            # way" is on the same page as the numbers above.
            n = _BANDWIDTH_N
            src = torch.rand(n, dtype=torch.float32, device=tensor_device)
            out["bytes_per_launch"] = 2 * 4 * n
            out.update(
                _time_launches(lambda: src.mul(1.0000001).add_(0.5), tensor_device)
            )
        out["status"] = "ok"
    except Exception as exc:
        out["status"] = "error"
        out["error_type"] = type(exc).__name__
        out["error"] = str(exc)[:2000]
    # Read AFTER the launches. The torch_only arm compiles nothing, so its arch
    # reading is the one to distrust, and it is labelled as such rather than
    # quietly reported beside the others.
    out["arch"] = _arch_report()
    out["live_arch"] = out["arch"].get("prog_arch")
    out["arch_reading_is_post_launch"] = workload != "torch_only"
    return out


# ---------------------------------------------------------------------------
# Section: native -- would Taichi-OWNED ndarrays avoid the round trip?
#
# Algan already annotates every kernel argument ``ti.types.ndarray()`` and holds
# no ``ti.field`` anywhere, so "switch to ndarrays" is not a change that can be
# made -- it is the current state. The real question underneath it is different:
# what a kernel gets today is a *torch tensor* (an external array), and Taichi
# has two entirely separate binding paths for those.
#
#   * ``set_arg_ext_array`` (torch tensor): passes a raw pointer, and for
#     anything that is neither a host tensor nor a CUDA tensor on a CUDA arch,
#     copies to the host before the launch and back after
#     (``kernel_impl.py``). Per launch, both directions, inputs included.
#   * ``set_arg_ndarray`` (a Taichi-owned ``ti.ndarray``): binds the device
#     allocation itself, and registers **no copy-back callback** at all.
#
# So Taichi-owned ndarrays do remove the per-launch copy. What they cannot
# remove is the copy at the boundary: Taichi 1.7.4 gives ``ScalarNdarray`` only
# ``from_numpy``/``to_numpy`` -- host memory, no ``to_torch``, no DLPack -- so
# data arriving from torch still crosses once. That relocates the cost from
# per-launch to per-crossing, which is a win exactly when a sub-pipeline runs
# many launches between handoffs. This section measures both halves so the
# break-even launch count can be computed rather than guessed.
# ---------------------------------------------------------------------------


def _section_native(device, workload):
    import numpy as np
    import torch

    brought_up = _bring_up(device)
    out = {"device": device, "workload": workload, **brought_up}
    out.update(_settled_arch())
    _emit_pre(out)
    n = _BANDWIDTH_N
    try:
        if workload == "launch":
            # The same kernel and the same element count as the Q5 bandwidth
            # arm, so the two numbers are directly comparable.
            src = ti.ndarray(ti.f32, shape=n)
            dst = ti.ndarray(ti.f32, shape=n)
            out["bytes_per_launch"] = 2 * 4 * n
            out.update(_time_launches(lambda: _k_bandwidth(src, dst, n), "cpu"))
        elif workload == "crossing":
            # What one handoff costs: host array in, host array out. This is
            # the price of every torch <-> Taichi boundary if the pipeline
            # holds its buffers as Taichi ndarrays.
            arr = ti.ndarray(ti.f32, shape=n)
            host = np.zeros(n, dtype=np.float32)
            out["bytes_per_crossing"] = 4 * n
            out["from_numpy"] = _time_launches(lambda: arr.from_numpy(host), "cpu")
            out["to_numpy"] = _time_launches(arr.to_numpy, "cpu")
        elif workload == "crossing_from_torch":
            # The realistic version of the same handoff: the data starts in a
            # torch tensor on the render device, which is where the raster
            # pipeline actually produces it.
            arr = ti.ndarray(ti.f32, shape=n)
            tensor = torch.zeros(n, dtype=torch.float32, device=device)
            out["bytes_per_crossing"] = 4 * n
            out.update(
                _time_launches(
                    lambda: arr.from_numpy(tensor.cpu().numpy()), tensor.device.type
                )
            )
        out["status"] = "ok"
    except Exception as exc:
        out["status"] = "error"
        out["error_type"] = type(exc).__name__
        out["error"] = str(exc)[:2000]
    return out


# ---------------------------------------------------------------------------
# Section: torchops -- Q6
# ---------------------------------------------------------------------------


def _torch_op_cases():
    """The ops the raster path actually calls, one closure each.

    Taken from what ``raster_pipeline.py``, ``sheets.py`` and ``primitives.py``
    use, with the dtypes they use them at -- a ``scatter_reduce_`` that works on
    f32 and not on int64 is a gap, and asking at the wrong dtype would miss it.
    """
    import torch

    def make(dev):
        return {
            "float64_tensor": lambda: torch.zeros(8, dtype=torch.float64, device=dev),
            "cumsum_i64": lambda: torch.arange(64, device=dev).cumsum(0),
            "cumsum_f32": lambda: torch.rand(64, device=dev).cumsum(0),
            "scatter_add_f32": lambda: torch.zeros(8, device=dev).scatter_add_(
                0, torch.randint(0, 8, (64,), device=dev), torch.rand(64, device=dev)
            ),
            "scatter_add_i64": lambda: torch.zeros(
                8, dtype=torch.int64, device=dev
            ).scatter_add_(
                0,
                torch.randint(0, 8, (64,), device=dev),
                torch.ones(64, dtype=torch.int64, device=dev),
            ),
            "scatter_reduce_amin_i64": lambda: torch.full(
                (8,), 1 << 40, dtype=torch.int64, device=dev
            ).scatter_reduce_(
                0,
                torch.randint(0, 8, (64,), device=dev),
                torch.randint(0, 1000, (64,), dtype=torch.int64, device=dev),
                reduce="amin",
                include_self=True,
            ),
            "scatter_reduce_amax_i64": lambda: torch.full(
                (8,), -1, dtype=torch.int64, device=dev
            ).scatter_reduce_(
                0,
                torch.randint(0, 8, (64,), device=dev),
                torch.randint(0, 1000, (64,), dtype=torch.int64, device=dev),
                reduce="amax",
                include_self=True,
            ),
            "scatter_reduce_sum_f32": lambda: torch.zeros(
                8, device=dev
            ).scatter_reduce_(
                0,
                torch.randint(0, 8, (64,), device=dev),
                torch.rand(64, device=dev),
                reduce="sum",
                include_self=True,
            ),
            "unique_consecutive_i64": lambda: torch.unique_consecutive(
                torch.tensor([1, 1, 2, 2, 3], dtype=torch.int64, device=dev),
                return_counts=True,
            ),
            "bincount_i64": lambda: torch.bincount(
                torch.randint(0, 8, (64,), device=dev)
            ),
            "cummax_f32": lambda: torch.cummax(torch.rand(64, device=dev), 0),
            "searchsorted_f32": lambda: torch.searchsorted(
                torch.linspace(0, 1, 64, device=dev), torch.rand(8, device=dev)
            ),
            "argsort_i64": lambda: torch.randint(
                0, 1 << 40, (64,), dtype=torch.int64, device=dev
            ).argsort(),
            "sort_i64": lambda: torch.randint(
                0, 1 << 40, (64,), dtype=torch.int64, device=dev
            ).sort(),
            "view_i64_as_u8": lambda: torch.arange(
                8, dtype=torch.int64, device=dev
            ).view(torch.uint8),
            "view_u8_as_bool": lambda: torch.zeros(
                8, dtype=torch.uint8, device=dev
            ).view(torch.bool),
            "view_i32_as_f32": lambda: torch.arange(
                8, dtype=torch.int32, device=dev
            ).view(torch.float32),
            "bitshift_i64": lambda: (
                torch.arange(8, dtype=torch.int64, device=dev) << 32
            )
            >> 32,
            "bitwise_and_i64": lambda: torch.arange(8, dtype=torch.int64, device=dev)
            & 0xFFFFFFFF,
            "repeat_interleave_i64": lambda: torch.repeat_interleave(
                torch.arange(8, dtype=torch.int64, device=dev),
                torch.ones(8, dtype=torch.int64, device=dev) * 3,
            ),
            "nonzero_bool": lambda: torch.zeros(
                64, dtype=torch.bool, device=dev
            ).nonzero(),
            "masked_select_f32": lambda: torch.rand(64, device=dev).masked_select(
                torch.rand(64, device=dev) > 0.5
            ),
            "index_select_i64": lambda: torch.arange(
                64, dtype=torch.int64, device=dev
            ).index_select(0, torch.randint(0, 64, (8,), device=dev)),
            "maximum_i64": lambda: torch.maximum(
                torch.arange(8, dtype=torch.int64, device=dev),
                torch.zeros(8, dtype=torch.int64, device=dev),
            ),
            "index_put_accumulate_f32": lambda: torch.zeros(8, device=dev).index_put_(
                (torch.randint(0, 8, (64,), device=dev),),
                torch.rand(64, device=dev),
                accumulate=True,
            ),
        }

    return make


def _section_torchops(device):
    import torch

    # A silent CPU fallback would answer "supported" to a question about what
    # the GPU can do, so make sure it is off no matter what the runner set.
    os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    cases = _torch_op_cases()(device)
    results = {}
    for name, fn in cases.items():
        try:
            value = fn()
            if device == "mps":
                torch.mps.synchronize()
            del value
            results[name] = {"status": "ok"}
        except Exception as exc:
            results[name] = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }
    return {
        "device": device,
        "fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "ops": results,
    }


# ---------------------------------------------------------------------------
# Section: determinism -- what a non-deterministic mode would actually cost
# ---------------------------------------------------------------------------

#: Shaped like the reduction ``_one_mesh_pixel_caps`` performs: a long fragment
#: stream folded into a much smaller set of pixels, which is where a float
#: atomic's ordering shows up.
_DET_FRAGMENTS = 400_000
_DET_SEGMENTS = 5_000
_DET_RUNS = 6


def _section_determinism(device):
    import torch

    out = {"device": device, "fragments": _DET_FRAGMENTS, "segments": _DET_SEGMENTS}
    try:
        generator = torch.Generator(device="cpu").manual_seed(20260828)
        cov = torch.rand(_DET_FRAGMENTS, generator=generator).to(device)
        seg = torch.randint(
            0, _DET_SEGMENTS, (_DET_FRAGMENTS,), generator=generator
        ).to(device)

        f32_runs = []
        for _ in range(_DET_RUNS):
            acc = torch.zeros(_DET_SEGMENTS, dtype=torch.float32, device=device)
            acc.scatter_add_(0, seg, cov)
            f32_runs.append(acc.cpu())
        spread = max(float((run - f32_runs[0]).abs().max()) for run in f32_runs[1:])
        out["f32_scatter_add_identical"] = all(
            torch.equal(run, f32_runs[0]) for run in f32_runs[1:]
        )
        out["f32_scatter_add_max_spread"] = spread
        # What that spread is worth in the units the renderer cares about: the
        # sum feeds a coverage threshold, and coverage is compared at f32.
        out["f32_spread_in_f32_ulps_at_1"] = spread / 5.960464477539063e-08

        # The fixed-point form, which is what a deterministic mode would use on
        # a device with no f64. Rounded in f32 because that is what a kernel on
        # this device could do; the point is the *accumulation*, and an integer
        # add has no ordering to be wrong about. Expected bit-identical every
        # time -- and if it is not, that is the more interesting finding.
        q = torch.round(cov * 4294967296.0).to(torch.int64)
        i64_runs = []
        for _ in range(_DET_RUNS):
            acc = torch.zeros(_DET_SEGMENTS, dtype=torch.int64, device=device)
            acc.scatter_add_(0, seg, q)
            i64_runs.append(acc.cpu())
        out["i64_fixed_point_identical"] = all(
            torch.equal(run, i64_runs[0]) for run in i64_runs[1:]
        )
        out["status"] = "ok"
    except Exception as exc:
        out["status"] = "error"
        out["error_type"] = type(exc).__name__
        out["error"] = str(exc)[:2000]
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

#: Probed in this order, one subprocess each. ``f32_basic`` first so a backend
#: that cannot run anything at all is distinguishable from one that merely
#: lacks a type.
_FEATURES = [
    "f32_basic",
    "f32_atomic_add",
    "i32_atomic_minmax",
    "i64_keys",
    "i64_atomic_add",
    "i64_atomic_min",
    "i64_fixed_point",
    "f64_ndarray",
    "f64_atomic_add",
    "bit_cast",
]

#: 31 is Metal's classic per-stage buffer limit and 49 is what
#: ``sheet_resolve_shade`` actually asks for; the rest bracket them. 63 and 64
#: are the control: Taichi caps kernel arguments *per backend* ("Do not exceed
#: 64 on x64 backend"), so the top of the ladder tells a backend-specific
#: binding limit apart from Taichi's own ceiling. Each entry passes ``nargs``
#: ndarrays plus one scalar, so 63 is the last that fits under a cap of 64.
_ARG_LADDER = [8, 16, 24, 31, 32, 40, 49, 56, 63, 64]


def _run_arm(args, extra_env=None, timeout=900):
    """Run one section in a fresh interpreter and return its JSON result."""
    env = dict(os.environ)
    env.setdefault("ALGAN_RENDER_DEVICE", "cpu")
    if extra_env:
        env.update(extra_env)
    command = [sys.executable, str(Path(__file__).resolve()), *args]
    try:
        completed = subprocess.run(
            command,
            cwd=str(_REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "argv": args}
    payload = None
    pre = None
    for line in completed.stdout.splitlines():
        if line.startswith(_MARKER):
            payload = json.loads(line[len(_MARKER) :])
        elif line.startswith(_PRE_MARKER):
            pre = json.loads(line[len(_PRE_MARKER) :])
    if payload is None:
        # An arm that produced no result crashed the interpreter, which is a
        # result: a backend rejecting a kernel does not always raise. What it
        # was running when it died comes from the pre-launch line, which is
        # written before the risky launch for exactly this case.
        crashed = {
            "status": "crashed",
            "argv": args,
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr.strip().splitlines()[-25:],
            "stdout_tail": [
                line
                for line in completed.stdout.strip().splitlines()[-10:]
                if not line.startswith(_PRE_MARKER)
            ],
        }
        if pre:
            crashed["pre_launch"] = pre
            crashed["live_arch"] = pre.get("live_arch")
        return crashed
    payload.setdefault("status", "ok")
    payload["returncode"] = completed.returncode
    # Taichi logs its device capabilities at trace level and nowhere else -- the
    # pybind class exposes no members -- so the caps arm is read out of the log.
    caps = [
        line.strip()
        for line in (completed.stderr + completed.stdout).splitlines()
        if "DeviceCapability::" in line
    ]
    if caps:
        payload["device_capabilities"] = caps
    return payload


def _why(record):
    """One line saying why an arm did not come back ``ok``.

    A bare "crashed" is the least useful thing this script can print: a Metal
    capability rejection, a binding-limit abort and a broken harness all look
    identical under it. The reason lives in the exception or in the last lines
    the dying process wrote, so put it on the summary line.
    """
    status = record.get("status")
    if status == "ok":
        return ""
    if status == "error":
        return "{}: {}".format(
            record.get("error_type", "?"),
            " ".join(str(record.get("error", "")).split())[:160],
        )
    if status == "crashed":
        interesting = [
            line
            for line in record.get("stderr_tail", [])
            if line.strip() and "Taichi] version" not in line
        ]
        detail = interesting[-1] if interesting else ""
        return "rc={} {}".format(record.get("returncode", "?"), detail[:160])
    return str(status)


def _devices_to_probe():
    """``cpu`` always -- it is the control every GPU number is read against."""
    import torch

    devices = ["cpu"]
    backends_mps = getattr(torch.backends, "mps", None)
    if backends_mps is not None and backends_mps.is_available():
        devices.append("mps")
    return devices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tensor-device", default=None)
    parser.add_argument("--feature")
    parser.add_argument("--nargs", type=int)
    parser.add_argument("--workload")
    parser.add_argument("--out", default=str(_REPO / "mps_probe_results.json"))
    args = parser.parse_args()

    if args.section:
        _emit(_dispatch(args))
        return 0
    return _orchestrate(args)


def _dispatch(args):
    section = args.section
    if section == "env":
        return _section_env()
    if section == "arch":
        return _section_arch(args.device)
    if section == "feature":
        return _section_feature(args.device, args.feature)
    if section == "args":
        return _section_args(args.device, args.nargs)
    if section == "staging":
        return _section_staging(
            args.device, args.tensor_device or args.device, args.workload
        )
    if section == "native":
        return _section_native(args.device, args.workload)
    if section == "torchops":
        return _section_torchops(args.device)
    if section == "determinism":
        return _section_determinism(args.device)
    raise SystemExit("unknown section: " + str(section))


def _orchestrate(args):
    devices = _devices_to_probe()
    results = {"devices_probed": devices, "sections": {}}
    sections = results["sections"]

    print("== environment ==", flush=True)
    sections["env"] = _run_arm(["--section", "env"])
    print(json.dumps(sections["env"], indent=2), flush=True)

    # Q1/Q2. The trace log is the only place Taichi says what the backend can
    # do, so this arm is run with it turned up; every other arm stays quiet.
    print("\n== Q1/Q2: arch and device capabilities ==", flush=True)
    sections["arch"] = {}
    for device in devices:
        sections["arch"][device] = _run_arm(
            ["--section", "arch", "--device", device],
            extra_env={"TI_LOG_LEVEL": "trace"},
        )
        print(device, "->", json.dumps(sections["arch"][device])[:400], flush=True)

    print("\n== Q2/Q3: which types and atomics compile ==", flush=True)
    sections["features"] = {}
    for device in devices:
        sections["features"][device] = {}
        for feature in _FEATURES:
            record = _run_arm(
                ["--section", "feature", "--device", device, "--feature", feature]
            )
            sections["features"][device][feature] = record
            print(
                "  {:>6}  {:<20} {:<6} {:<9} {}".format(
                    device,
                    feature,
                    record.get("live_arch", "?"),
                    record["status"],
                    _why(record),
                ),
                flush=True,
            )

    print("\n== Q4: how many ndarray arguments bind ==", flush=True)
    sections["args"] = {}
    for device in devices:
        sections["args"][device] = {}
        for nargs in _ARG_LADDER:
            record = _run_arm(
                ["--section", "args", "--device", device, "--nargs", str(nargs)]
            )
            sections["args"][device][str(nargs)] = record
            print(
                "  {:>6}  {:>3} args  {:<6} {:<9} {}".format(
                    device,
                    nargs,
                    record.get("live_arch", "?"),
                    record["status"],
                    _why(record),
                ),
                flush=True,
            )
            if record["status"] not in ("ok",):
                # Past the first failure the ladder has told us what we came
                # for; keep going anyway, because a limit that is not monotone
                # would be worth knowing about.
                pass

    print("\n== Q5: what the interop costs ==", flush=True)
    sections["staging"] = {}
    arms = [("cpu", "cpu")]
    if "mps" in devices:
        arms += [("mps", "mps"), ("mps", "cpu")]
    for workload in ("bandwidth", "compute", "torch_only"):
        for render_device, tensor_device in arms:
            if workload == "torch_only" and render_device != tensor_device:
                continue
            key = f"{workload}|{render_device}|{tensor_device}"
            record = _run_arm(
                [
                    "--section",
                    "staging",
                    "--device",
                    render_device,
                    "--tensor-device",
                    tensor_device,
                    "--workload",
                    workload,
                ]
            )
            sections["staging"][key] = record
            print(
                "  {:<11} device={:<4} tensors={:<4} arch={:<6} {:<7}"
                " median {} ms  {}".format(
                    workload,
                    render_device,
                    tensor_device,
                    record.get("live_arch", "?"),
                    record.get("status", "?"),
                    record.get("median_ms", "-"),
                    _why(record),
                ),
                flush=True,
            )

    # Vulkan is the other backend a Mac offers, and "run both halves on Vulkan"
    # is the natural workaround to try. Forcing it here settles the Taichi half
    # by measurement: if a Vulkan arm stages a torch tensor exactly as the Metal
    # arm does, the backend was never what made the copy happen.
    if "mps" in devices:
        print("\n== workaround check: Taichi forced onto Vulkan ==", flush=True)
        sections["vulkan"] = {}
        sections["vulkan"]["arch"] = _run_arm(
            ["--section", "arch", "--device", "mps"],
            extra_env={"TI_ARCH": "vulkan", "TI_LOG_LEVEL": "trace"},
        )
        print(
            "  arch arm -> {} ({})".format(
                sections["vulkan"]["arch"].get("live_arch", "?"),
                sections["vulkan"]["arch"].get("status", "?"),
            ),
            flush=True,
        )
        for tensor_device in ("mps", "cpu"):
            record = _run_arm(
                [
                    "--section",
                    "staging",
                    "--device",
                    "mps",
                    "--tensor-device",
                    tensor_device,
                    "--workload",
                    "bandwidth",
                ],
                extra_env={"TI_ARCH": "vulkan"},
            )
            sections["vulkan"][f"bandwidth|{tensor_device}"] = record
            print(
                "  bandwidth  tensors={:<4} arch={:<6} {:<7} median {} ms  {}".format(
                    tensor_device,
                    record.get("live_arch", "?"),
                    record.get("status", "?"),
                    record.get("median_ms", "-"),
                    _why(record),
                ),
                flush=True,
            )

    print(
        "\n== native ndarrays: does Taichi-owned memory dodge the copy? ==", flush=True
    )
    sections["native"] = {}
    for device in devices:
        for workload in ("launch", "crossing", "crossing_from_torch"):
            record = _run_arm(
                ["--section", "native", "--device", device, "--workload", workload]
            )
            sections["native"][f"{workload}|{device}"] = record
            timing = record.get("median_ms")
            if timing is None and "from_numpy" in record:
                timing = "in {} / out {}".format(
                    record["from_numpy"].get("median_ms"),
                    record["to_numpy"].get("median_ms"),
                )
            print(
                "  {:<20} device={:<4} arch={:<6} {:<7} median {} ms  {}".format(
                    workload,
                    device,
                    record.get("live_arch", "?"),
                    record.get("status", "?"),
                    timing if timing is not None else "-",
                    _why(record),
                ),
                flush=True,
            )

    print("\n== Q6: torch op coverage ==", flush=True)
    sections["torchops"] = {}
    for device in devices:
        record = _run_arm(["--section", "torchops", "--device", device])
        sections["torchops"][device] = record
        missing = [
            name
            for name, value in record.get("ops", {}).items()
            if value["status"] != "ok"
        ]
        print(
            "  {}: {} of {} ops failed".format(
                device, len(missing), len(record.get("ops", {}))
            ),
            flush=True,
        )
        for name in missing:
            print(
                "    - {}: {}".format(name, record["ops"][name]["error_type"]),
                flush=True,
            )

    print("\n== determinism of a float reduction ==", flush=True)
    sections["determinism"] = {}
    for device in devices:
        record = _run_arm(["--section", "determinism", "--device", device])
        sections["determinism"][device] = record
        print(f"  {device}: {json.dumps(record)}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nwrote " + args.out, flush=True)
    _print_verdict(results)
    return 0


def _print_verdict(results):
    """The four facts that decide whether the port is worth starting."""
    print("\n" + "=" * 72, flush=True)
    print("VERDICT INPUTS", flush=True)
    print("=" * 72, flush=True)

    arch = results["sections"].get("arch", {}).get("mps", {})
    print(
        "Q1 backend Algan selects for MPS: {}".format(arch.get("live_arch", "n/a")),
        flush=True,
    )
    print(
        "   engine believes launch is staging-free: {}".format(
            arch.get("launch_is_local_mps", "n/a")
        ),
        flush=True,
    )

    features = results["sections"].get("features", {}).get("mps", {})
    if features:
        blocking = ["i64_keys", "i64_atomic_min", "i64_atomic_add", "i64_fixed_point"]
        print("Q3 64-bit integer kernels:", flush=True)
        for name in blocking:
            print(
                "     {:<18} {}".format(
                    name, features.get(name, {}).get("status", "n/a")
                ),
                flush=True,
            )
        print(
            "   f32 atomic add (a non-deterministic mode's floor): {}".format(
                features.get("f32_atomic_add", {}).get("status", "n/a")
            ),
            flush=True,
        )

    ladder = results["sections"].get("args", {}).get("mps", {})
    if ladder:
        ok = [int(k) for k, v in ladder.items() if v.get("status") == "ok"]
        print(
            "Q4 widest kernel that bound: {} ndarray args (sheet_resolve_shade needs 49)".format(
                max(ok) if ok else "none"
            ),
            flush=True,
        )

    staging = results["sections"].get("staging", {})
    base = staging.get("bandwidth|cpu|cpu", {}).get("median_ms")
    mps = staging.get("bandwidth|mps|mps", {}).get("median_ms")
    if base and mps:
        print(
            f"Q5 bandwidth-bound launch: cpu arch {base} ms vs mps {mps} ms ({mps / base:.2f}x)",
            flush=True,
        )
    base_c = staging.get("compute|cpu|cpu", {}).get("median_ms")
    mps_c = staging.get("compute|mps|mps", {}).get("median_ms")
    if base_c and mps_c:
        print(
            f"   compute-bound launch:   cpu arch {base_c} ms vs mps {mps_c} ms ({mps_c / base_c:.2f}x)",
            flush=True,
        )
    print("=" * 72, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
