"""Single entry point for initializing the Taichi CUDA runtime.

The rasterizer, the fragment blender and the ray tracer each need Taichi up
before they define their kernels, and they may be imported in any order.
Routing every one of them through :func:`init_taichi` guarantees the runtime
is initialized exactly once, with a consistent performance config -- calling
``ti.init`` again would discard already-compiled kernels (previously the
rasterizer modules re-initialized each other at import).

Performance config (most are Taichi's own defaults, set explicitly so they
cannot silently regress, plus two empirically-tuned register knobs):

* ``fast_math`` / ``advanced_optimization`` / ``offline_cache`` -- on by
  default in Taichi 1.7.4; pinned here.
* ``offline_cache_file_path`` -- Algan's dedicated kernel cache,
  ``_TAICHI_CACHE_DIRECTORY`` (unless the standard
  ``TI_OFFLINE_CACHE_FILE_PATH`` env var is set, which then wins).
* ``ALGAN_GPU_MAX_REG`` (env int) -> ``gpu_max_reg``: cap on registers per
  thread for CUDA codegen (ptxas ``-maxrregcount``). 0/unset leaves it to
  ptxas, which for the big deterministic ray-trace kernel settles on 128 and
  spills heavily to local memory. Raising the cap keeps more values in
  registers (fewer spills) at the cost of occupancy; the sweet spot is tuned
  empirically per kernel/GPU.
* ``ALGAN_OPT_LEVEL`` (env int) -> ``opt_level`` (Taichi default 1). Higher is
  more aggressive but can *increase* register pressure, so it is opt-in.
"""

from __future__ import annotations

import contextlib
import datetime as _datetime
import json
import os
import threading
import time

import taichi as ti
import torch

from algan.environment import env_flag, env_int, env_str
from algan.settings._startup import _TAICHI_CACHE_DIRECTORY, render_device

_COMPILE_LOG_LOCK = threading.Lock()
_COMPILE_FRONTEND = {}
_COMPILE_NOTICE_CALLBACK = None
_COMPILE_NOTICE_THREAD_ID = None


def _compile_logging_enabled():
    return env_flag("ALGAN_LOG_TAICHI_COMPILES", False)


def _compile_log_path():
    path = env_str("ALGAN_TAICHI_COMPILE_LOG", "").strip()
    return os.path.abspath(os.path.expanduser(path)) if path else ""


def _emit_compile_record(record):
    """Print and optionally persist one Taichi compilation timing record."""
    if not _compile_logging_enabled():
        return
    phase = record["phase"]
    name = record["kernel"]
    status = record["status"]
    stamp = record["timestamp"]
    if phase == "start":
        message = f"[Taichi compile] started {name} at {stamp}"
    elif phase == "complete":
        message = (
            f"[Taichi compile] completed {name} at {stamp}: "
            f"frontend={record['frontend_seconds']:.3f}s, "
            f"backend={record['backend_seconds']:.3f}s, "
            f"total={record['total_seconds']:.3f}s"
        )
    else:
        message = (
            f"[Taichi compile] {status} {name} at {stamp} after "
            f"{record.get('total_seconds', 0.0):.3f}s"
        )
    with _COMPILE_LOG_LOCK:
        print(message, flush=True)
        path = _compile_log_path()
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")


def _kernel_timing_name(kernel, key):
    func = kernel.func
    module = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", "kernel"))
    base = f"{module}.{qualname}" if module else qualname
    instance = key[1] if len(key) > 1 else 0
    mode = getattr(kernel.autodiff_mode, "name", str(kernel.autodiff_mode))
    return f"{base}[specialization={instance}, autodiff={mode}]"


def _set_compile_notice_callback(callback):
    """Register the one-shot callback used for a real kernel compilation."""
    global _COMPILE_NOTICE_CALLBACK, _COMPILE_NOTICE_THREAD_ID
    with _COMPILE_LOG_LOCK:
        _COMPILE_NOTICE_CALLBACK = callback
        _COMPILE_NOTICE_THREAD_ID = threading.get_ident() if callback else None


def _compile_notice_active():
    with _COMPILE_LOG_LOCK:
        return (
            _COMPILE_NOTICE_CALLBACK is not None
            and threading.get_ident() == _COMPILE_NOTICE_THREAD_ID
        )


def _notify_compile_notice():
    global _COMPILE_NOTICE_CALLBACK, _COMPILE_NOTICE_THREAD_ID
    with _COMPILE_LOG_LOCK:
        if threading.get_ident() != _COMPILE_NOTICE_THREAD_ID:
            return
        callback = _COMPILE_NOTICE_CALLBACK
        _COMPILE_NOTICE_CALLBACK = None
        _COMPILE_NOTICE_THREAD_ID = None
    if callback is not None:
        callback()


def _taichi_log_level():
    """Return Taichi's current native logging threshold, when available."""
    is_effective = getattr(ti, "is_logging_effective", None)
    if is_effective is None:
        return None
    try:
        for level in ("trace", "debug", "info", "warn", "error", "critical"):
            if is_effective(level):
                return level
    except Exception:
        return None
    return None


def _capture_native_stderr(call):
    """Run ``call`` while collecting native writes to file descriptor 2."""
    read_fd, write_fd = os.pipe()
    chunks = []

    def drain():
        try:
            while True:
                chunk = os.read(read_fd, 4096)
                if not chunk:
                    return
                chunks.append(chunk)
        except OSError:
            return

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    saved_fd = os.dup(2)
    write_fd_open = True
    try:
        os.dup2(write_fd, 2)
        os.close(write_fd)
        write_fd_open = False
        result = call()
    finally:
        if write_fd_open:
            os.close(write_fd)
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        reader.join()
        os.close(read_fd)
    return result, b"".join(chunks)


def _compile_with_cache_observation(call):
    """Run one backend compile and return its native cache diagnostics.

    Taichi 1.7.4 reports an offline-cache hit as ``Create kernel ... from
    cache`` and a miss as ``Cache kernel ...``. The messages are native
    debug-level output, so temporarily enable and capture that stream while
    preserving the caller's logging level and output.
    """
    previous_level = _taichi_log_level()
    if previous_level is None:
        return call(), b""
    try:
        if previous_level != "trace":
            ti.set_logging_level("trace")
    except Exception:
        return call(), b""
    try:
        result, native_log = _capture_native_stderr(call)
    finally:
        if previous_level != "trace":
            with contextlib.suppress(Exception):
                ti.set_logging_level(previous_level)
    if previous_level in ("trace", "debug") and native_log:
        with contextlib.suppress(OSError):
            os.write(2, native_log)
    return result, native_log


def _loaded_from_offline_cache(native_log):
    return b"from cache" in native_log.lower()


def _install_taichi_compile_logger():
    """Instrument Taichi's Python front-end and backend compiler boundary.

    Taichi 1.7.4 performs compilation in two distinct stages.
    ``Kernel.materialize`` transforms the Python AST and creates a C++ kernel;
    ``Program.compile_kernel`` then lowers it for the selected backend or loads
    the matching offline-cache artifact. We time both and report their sum when
    the specialization becomes launchable. With an empty cache the backend time
    is cold compilation time; with a populated cache it is cache lookup/load time.
    """
    from taichi.lang import impl as _ti_impl
    from taichi.lang.kernel_impl import Kernel as _TaichiKernel

    if not getattr(_TaichiKernel, "_algan_compile_timing_wrapped", False):
        original_materialize = _TaichiKernel.materialize

        def timed_materialize(self, key=None, args=None, arg_features=None):
            if key is None:
                key = (self.func, 0, self.autodiff_mode)
            if key in self.compiled_kernels:
                return original_materialize(
                    self, key=key, args=args, arg_features=arg_features
                )

            name = _kernel_timing_name(self, key)
            started_wall = (
                _datetime.datetime.now(_datetime.timezone.utc)
                .astimezone()
                .isoformat(timespec="milliseconds")
            )
            started = time.perf_counter()
            _emit_compile_record(
                {
                    "phase": "start",
                    "status": "started",
                    "kernel": name,
                    "timestamp": started_wall,
                }
            )
            try:
                result = original_materialize(
                    self, key=key, args=args, arg_features=arg_features
                )
            except Exception:
                elapsed = time.perf_counter() - started
                _emit_compile_record(
                    {
                        "phase": "failed",
                        "status": "frontend_failed",
                        "kernel": name,
                        "timestamp": _datetime.datetime.now(_datetime.timezone.utc)
                        .astimezone()
                        .isoformat(timespec="milliseconds"),
                        "frontend_seconds": elapsed,
                        "backend_seconds": 0.0,
                        "total_seconds": elapsed,
                    }
                )
                raise

            frontend_seconds = time.perf_counter() - started
            compiled = self.compiled_kernels.get(key)
            if compiled is not None:
                with _COMPILE_LOG_LOCK:
                    _COMPILE_FRONTEND[id(compiled)] = {
                        "kernel": name,
                        "frontend_seconds": frontend_seconds,
                        "started_perf": started,
                    }
            return result

        _TaichiKernel.materialize = timed_materialize
        _TaichiKernel._algan_compile_timing_wrapped = True

    program = _ti_impl.get_runtime().prog
    if program is None:
        return
    program_type = type(program)
    if getattr(program_type, "_algan_compile_timing_wrapped", False):
        return
    original_compile_kernel = program_type.compile_kernel

    def timed_compile_kernel(self, *args, **kwargs):
        cpp_kernel = args[-1] if args else kwargs.get("kernel")
        with _COMPILE_LOG_LOCK:
            meta = _COMPILE_FRONTEND.pop(id(cpp_kernel), None)
        # ``Program.compile_kernel`` is also called on ordinary launches after
        # a specialization is already ready. Only a preceding materialize call
        # places metadata here, so skip those zero-cost cache lookups entirely.
        if meta is None:
            return original_compile_kernel(self, *args, **kwargs)
        name = meta["kernel"]
        frontend_seconds = float(meta["frontend_seconds"])
        backend_started = time.perf_counter()
        observe_cache = _compile_notice_active()
        try:
            if observe_cache:
                result, native_log = _compile_with_cache_observation(
                    lambda: original_compile_kernel(self, *args, **kwargs)
                )
            else:
                result = original_compile_kernel(self, *args, **kwargs)
        except Exception:
            backend_seconds = time.perf_counter() - backend_started
            total_seconds = frontend_seconds + backend_seconds
            _emit_compile_record(
                {
                    "phase": "failed",
                    "status": "backend_failed",
                    "kernel": name,
                    "timestamp": _datetime.datetime.now(_datetime.timezone.utc)
                    .astimezone()
                    .isoformat(timespec="milliseconds"),
                    "frontend_seconds": frontend_seconds,
                    "backend_seconds": backend_seconds,
                    "total_seconds": total_seconds,
                }
            )
            raise
        if observe_cache and not _loaded_from_offline_cache(native_log):
            _notify_compile_notice()
        backend_seconds = time.perf_counter() - backend_started
        total_seconds = frontend_seconds + backend_seconds
        _emit_compile_record(
            {
                "phase": "complete",
                "status": "complete",
                "kernel": name,
                "timestamp": _datetime.datetime.now(_datetime.timezone.utc)
                .astimezone()
                .isoformat(timespec="milliseconds"),
                "frontend_seconds": frontend_seconds,
                "backend_seconds": backend_seconds,
                "total_seconds": total_seconds,
            }
        )
        return result

    program_type.compile_kernel = timed_compile_kernel
    program_type._algan_compile_timing_wrapped = True


def _sync_devices():
    """Block until all pending torch and Taichi device work has finished.

    The one device-sync helper in the package: nothing else calls
    ``torch.cuda.synchronize``, ``torch.mps.synchronize`` or ``ti.sync``
    directly. Syncs whichever torch backend is present (CUDA, MPS) and then
    Taichi; on a CPU-only build every arm is a no-op.

    Only the main thread syncs. Stage boundaries can run on the batch-prep
    worker thread (scene prefetch), where syncing would serialize against the
    render thread's GPU work — misattributing it in a profile — and where
    ``ti.sync()`` is not safe to call at all.
    """
    if threading.current_thread() is not threading.main_thread():
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.mps.is_available():
        torch.mps.synchronize()
    # Guarded: several call sites sync before any kernel has run, and ti.sync()
    # on an uninitialized runtime raises "Please call init() first".
    if _already_initialized():
        ti.sync()


def _already_initialized():
    try:
        return ti.lang.impl.get_runtime().prog is not None
    except Exception:
        return False


def _taichi_arch():
    """Return the explicit Taichi backend for Algan's render device.

    ``ti.gpu`` is a backend preference list, not a CUDA-only alias.  On a
    machine without CUDA it falls through to Vulkan, and some headless Vulkan
    configurations crash inside Taichi instead of returning an error.  Torch
    has already probed the usable render device, so select the matching Taichi
    backend directly and never trigger that fallback chain.
    """
    if render_device().type == "cpu":
        return ti.cpu
    return ti.gpu


def _live_arch():
    """The arch of the running Taichi program, or ``None`` if there is none."""
    if not _already_initialized():
        return None
    with contextlib.suppress(Exception):
        return ti.lang.impl.get_runtime().prog.config().arch
    return None


#: Which Taichi backends actually serve each render-device type.
#:
#: Written out rather than resolved through ``adaptive_arch_select`` on purpose.
#: Resolving ``ti.gpu`` means probing every backend in the list, which is the
#: fallback chain :func:`_taichi_arch` exists to avoid -- it reaches Vulkan and
#: OpenGL, and some headless configurations crash inside Taichi rather than
#: reporting that they are unavailable. This comparison runs once per render
#: job and must not be able to take the process down.
#:
#: ``mps`` lists both SPIR-V backends because either really does serve an Apple
#: GPU: ``ti.gpu`` selects metal, ``TI_ARCH=vulkan`` selects vulkan, and both
#: run on the same physical device.
_ARCHS_SERVING_DEVICE = {
    "cpu": (ti.cpu,),
    "cuda": (ti.cuda,),
    "mps": (ti.metal, ti.vulkan),
}


def _arch_matches_render_device():
    """Whether the live program's arch is one that serves the render device.

    Compares against the **live** arch rather than against the last value
    :func:`_taichi_arch` returned, because ``ti.gpu`` is a preference list:
    Taichi resolves it to cuda, metal or vulkan at ``ti.init``, so two different
    render devices can both ask for ``ti.gpu`` and get different programs.

    And it compares against the arch that serves *this* device, not merely
    against "some GPU". Testing ``live != ti.cpu`` made every GPU backend
    interchangeable, so a render device moving between two of them -- cuda to
    mps, or back -- kept whichever program was already up and launched every
    kernel on the wrong device with no re-init and no error. The docstring here
    claimed to rule that out while the code was what allowed it; this is the
    comparison it described.

    An unrecognised device type keeps the old, coarse rule. It is the honest
    answer for a backend this mapping has never seen, and it does not force a
    re-initialization on every render of a device that may well be fine.
    """
    live = _live_arch()
    if live is None:
        return False
    serving = _ARCHS_SERVING_DEVICE.get(render_device().type)
    if serving is None:
        return live != ti.cpu
    return live in serving


def taichi_arch_is_cpu():
    """Whether Taichi runs kernels on the CPU.

    Every Algan kernel takes its arguments as torch tensors, and Taichi stages
    any argument that does not already live on its arch's device. Launching a
    kernel with the CPU batch-prep tensors while the arch is CUDA therefore
    copies every argument -- inputs *and* the whole result -- through VRAM, on
    the worker thread that is deliberately kept off the GPU; that is what made
    the timeline's own kernels a liability (see
    ``timeline._generate_array_states_taichi``). So a prep-stage kernel is worth
    dispatching only when the arch is the CPU, which is exactly when the render
    device is.

    Reads the live program's arch when Taichi is already up and Algan's selected
    backend otherwise, so asking never forces initialization.
    """
    live = _live_arch()
    if live is not None:
        return live == ti.cpu
    return _taichi_arch() == ti.cpu


def taichi_arch_is_cuda():
    """Whether Taichi runs kernels on CUDA.

    The companion to :func:`taichi_arch_is_cpu`, and read for the same reason:
    CUDA is the only GPU backend that can adopt a torch allocation instead of
    copying it (see :func:`taichi_launch_is_local`).

    Reads the live program's arch when Taichi is already up. Otherwise it
    answers from the render device, because :func:`_taichi_arch` returns the
    ``ti.gpu`` *preference list* off the CPU and that list is headed by cuda --
    so a CUDA render device is exactly the case where the arch will come up
    cuda. Asking never forces initialization.
    """
    live = _live_arch()
    if live is not None:
        return live == ti.cuda
    return render_device().type == "cuda"


def taichi_launch_is_local(device):
    """Whether a kernel launched against a tensor on ``device`` avoids staging.

    The inverse of the hazard :func:`taichi_arch_is_cpu` describes. A launch is
    free of the copy exactly when Taichi can bind the torch allocation itself,
    and only two pairings can:

    * a **host** tensor on a **CPU** arch, and
    * a **CUDA** tensor on a **CUDA** arch.

    Everything else stages, including a pairing whose two halves name the same
    physical device. That is not obvious and it is the whole reason this is not
    ``device.type == render_device().type``: Taichi implements
    ``Device::import_memory`` for ``CpuDevice`` and ``CudaDevice`` and for
    nothing else, so its Metal and Vulkan backends cannot take a pointer they
    did not allocate. An MPS tensor on a Metal arch is therefore copied to the
    host before the launch and copied back after (``kernel_impl.py``), even
    though both sides are the same Apple GPU -- measured at 53x the cost of the
    same kernel on the CPU arch, against a device-equality test that called it
    free (``DESIGN_mps_support.md`` §1.3).

    Phrased as a property of the pairing rather than as ``device.type ==
    "cuda"``, because those are not the same question either. A host tensor on
    a CUDA arch stages, and must take the torch path; a host tensor on a **CPU**
    arch does not stage, and a call site that tests for CUDA turns the kernel
    off in exactly the case where it is free.
    """
    if taichi_arch_is_cpu():
        return device.type == "cpu"
    return device.type == "cuda" and taichi_arch_is_cuda()


#: CPU batch-prep kernels that are dispatched by default.
#:
#: Only ``cpunormals`` pays. ``benchmarks/_cpu_prep_kernels_ab.py`` measures the
#: other two at **0.69-1.03x** (the gather) and **0.79-0.81x** (the color bake)
#: -- both byte-identical, both slower than the torch call they replace, on the
#: shapes the batched build passes. They are the same shape of work the timeline
#: query turned out to be: a memory-bound copy with nothing to fuse, where
#: torch's vectorized ``index_select``/``clone`` already saturates the load and a
#: kernel only adds launch overhead. Making the channel loop a compile-time
#: unroll (``channels: ti.template()``) moved them barely at all, which is the
#: confirmation that they are bandwidth-bound rather than codegen-bound.
#:
#: They are kept, tested and opt-in rather than deleted so the measurement is
#: reproducible and so a machine with different memory bandwidth can be checked
#: with ``ALGAN_OPT_ENABLE=cpugather,cpucolors`` instead of a rebuild.
_CPU_PREP_KERNELS_ON_BY_DEFAULT = frozenset(("cpunormals",))

_OPT_ENABLED = None


def _opt_enabled(name):
    """Whether ``ALGAN_OPT_ENABLE`` names this off-by-default optimization."""
    global _OPT_ENABLED
    if _OPT_ENABLED is None:
        _OPT_ENABLED = frozenset(env_str("ALGAN_OPT_ENABLE", "").split(","))
    return name in _OPT_ENABLED


def cpu_prep_kernel_enabled(name):
    """Whether the CPU batch-prep kernel called ``name`` should be dispatched.

    False on a GPU arch (see :func:`taichi_arch_is_cpu`), and false when either
    ``ALGAN_OPT_DISABLE=cpukernels`` (all of them) or ``ALGAN_OPT_DISABLE=<name>``
    is set, so each kernel keeps the bisect switch the other prep optimizations
    use and an A/B can run the torch arm without a rebuild.

    A kernel outside :data:`_CPU_PREP_KERNELS_ON_BY_DEFAULT` additionally needs
    naming in ``ALGAN_OPT_ENABLE``, because it measured slower than torch.
    """
    from algan.animation_timeline.timeline import _opt_disabled

    if _opt_disabled("cpukernels") or _opt_disabled(name):
        return False
    if name not in _CPU_PREP_KERNELS_ON_BY_DEFAULT and not _opt_enabled(name):
        return False
    return taichi_arch_is_cpu()


def taichi_init_kwargs():
    """Algan's Taichi runtime config, as a kwargs dict.

    Exposed so the profiler (:mod:`algan.utils.profiling_utils`) can launch
    Taichi with the *exact same* performance config production uses, just
    adding ``kernel_profiler=True`` -- otherwise the benchmark silently
    measures a different (much faster) config than real renders. See
    [[algan-render-benchmarking]].
    """
    kwargs = {
        "arch": _taichi_arch(),
        "fast_math": True,
        # advanced_optimization defaults off (it raised register
        # pressure on the big megakernels); env ALGAN_ADV_OPT=1 to A/B.
        "advanced_optimization": env_flag("ALGAN_ADV_OPT", False),
        # debug=True inserts a bounds-check on *every* ndarray access;
        # the ray-trace megakernels do millions of array reads per ray
        # (BVH nodes, packed geometry), so it ran them ~11x slower with
        # no benefit to released renders. Keep it off (env ALGAN_TI_DEBUG=1
        # re-enables it for kernel development).
        "debug": env_flag("ALGAN_TI_DEBUG", False),
        "offline_cache": True,
        # The default 100 MB cache LRU-evicts large megakernel
        # artifacts once several variants (general / no-PN / lean /
        # path-trace / wavefront, plus per-config rebuilds) are
        # compiled, forcing repeated ~minutes-long recompiles. Raise
        # it (disk-backed) so every kernel stays cached.
        #
        # 2 GB, and the ceiling is not a guess: the pybind setter for
        # this field takes a *signed 32-bit* int, so 2_147_483_647 is
        # accepted and 2**31 raises TypeError out of ti.init (checked
        # on taichi 1.7.4 -- it fails loudly rather than wrapping, so
        # a too-large value here could never become a silently tiny
        # cache). 2_000_000_000 keeps a margin under that and is 20x
        # the Taichi default.
        #
        # Worth the headroom because the eviction is not free and not
        # visible: Taichi prunes to 75% of this on program *exit*, so
        # a working set over the cap is re-compiled by the next run,
        # every run, with nothing in the log to say so. A single CUDA
        # megakernel artifact runs 30-45 MB, and one machine's cache
        # was measured sitting at 751 MiB of the old 1 GB.
        "offline_cache_max_size_of_files": 2_000_000_000,
    }
    # Keep Algan's compiled kernels in a dedicated directory under Algan's
    # cache dir instead of Taichi's global default, so they never contend
    # with other Taichi programs for the LRU budget. A ti.init kwarg beats
    # the TI_OFFLINE_CACHE_FILE_PATH env var (Taichi warns and ignores the
    # env), so only pass it when the env var is unset to keep that standard
    # escape hatch working.
    if not env_str("TI_OFFLINE_CACHE_FILE_PATH"):
        kwargs["offline_cache_file_path"] = str(_TAICHI_CACHE_DIRECTORY)
    max_reg = env_int("ALGAN_GPU_MAX_REG", 0)
    if max_reg > 0:
        kwargs["gpu_max_reg"] = max_reg
    opt_level = env_int("ALGAN_OPT_LEVEL", 0)
    if opt_level > 0:
        kwargs["opt_level"] = opt_level
    return kwargs


def init_taichi():
    """Initialize Taichi on Algan's selected backend, once.

    Never re-initializes: a second ``ti.init`` discards every kernel compiled so
    far, so anything that just needs Taichi up (a kernel module at import, a
    benchmark, a test) can say so without paying for it. The one caller that
    *does* need the arch re-selected is :func:`ensure_taichi_for_render`.
    """
    if _already_initialized():
        _install_taichi_compile_logger()
        return
    ti.init(**taichi_init_kwargs())
    _install_taichi_compile_logger()


def ensure_taichi_for_render():
    """Bring Taichi up on the arch the current render device needs.

    Called once at the start of a render job, and the only place a running
    Taichi program is ever replaced. Three cases:

    * **No program** -- ordinary first init.
    * **Program on the right arch** -- the overwhelmingly common case, and free.
    * **Program on the wrong arch** -- ``SETTINGS.computing.render_device``
      moved across the CPU/GPU line since the last render, so ``ti.init`` runs
      again on the new arch.

    The third case is not cheap and is not meant to be hidden. ``ti.init``
    itself is 0.2 s on the CPU and ~0.9 s on CUDA, but it calls ``impl.reset()``,
    which clears ``compiled_kernels`` on every registered kernel -- so the next
    render re-materializes each kernel it launches and re-reads them from the
    offline cache. On a trivial scene (one ``Square``, warm offline cache) that
    costs the next render **+4 s on the CPU and +24 s on CUDA**, against a
    steady-state render of 0.13 s and 0.27 s respectively; on CUDA that is
    within ~10% of a full cold start, so essentially the whole
    "Preparing render kernels" pass is paid again. On a real scene it is that
    pass in full. That is why the device is a top-of-script setting and why this
    compares arches instead of just calling ``ti.init`` each time.

    Safe because Algan holds no ``ti.field`` or ``ti.Ndarray`` anywhere -- every
    kernel argument is a torch tensor. Reading a Taichi field created before a
    re-init segfaults the process with no Python exception, so if that ever
    stops being true, this function stops being safe.

    Returns whether Taichi was re-initialized.
    """
    global _ARCH_READY_FOR
    wanted = render_device()
    if not _already_initialized():
        init_taichi()
        _ARCH_READY_FOR = wanted
        return False
    if _arch_matches_render_device():
        _install_taichi_compile_logger()
        _ARCH_READY_FOR = wanted
        return False
    ti.init(**taichi_init_kwargs())
    _install_taichi_compile_logger()
    _ARCH_READY_FOR = wanted
    return True


#: The device object :func:`ensure_taichi_for_render` last brought the runtime
#: in line with. Compared by *identity*: ``SETTINGS.computing.set`` deep-copies
#: every field, so any write to the section produces a new ``torch.device`` and
#: re-arms the check even when the value is unchanged. That costs one extra
#: arch comparison per ``set`` call and needs no notification protocol between
#: the settings section and this module.
_ARCH_READY_FOR = None


#: How many render jobs are between :func:`ensure_taichi_for_render` and their
#: last frame. A counter rather than a flag: ``save_frame`` inside a
#: ``save_video`` post-process, or a nested preview, would otherwise clear it
#: early.
_RENDER_JOBS_ACTIVE = 0


@contextlib.contextmanager
def render_job_holding_the_arch():
    """Mark the arch as in use for the duration of one render job.

    A device change *during* a render is the one way this design can corrupt
    something rather than merely be slow: the batch-prep worker launches kernels
    on its own thread, and a change would arm the arch guard there, so the next
    prep launch could run ``ti.init`` -- discarding every compiled kernel -- while
    the render thread is inside one. ``SETTINGS.computing.set(render_device=...)``
    consults :func:`render_is_active` and refuses instead.
    """
    global _RENDER_JOBS_ACTIVE
    _RENDER_JOBS_ACTIVE += 1
    try:
        yield
    finally:
        _RENDER_JOBS_ACTIVE -= 1
        if _RENDER_JOBS_ACTIVE == 0:
            # Release the MPS zero-copy import cache with the job that filled
            # it. Each entry holds a torch storage alive so Taichi cannot be
            # reading a buffer the caching allocator has recycled, and the
            # biggest of those storages is the render arena -- which the render
            # loop drops on purpose. A cache that outlived the job would pin
            # it. A no-op on every device but MPS, and on MPS without the
            # patched Taichi build.
            from algan.rendering.mps_zero_copy import clear_import_cache

            clear_import_cache()


def render_is_active():
    """Whether a render job currently depends on the live arch."""
    return _RENDER_JOBS_ACTIVE > 0


def install_render_arch_guard():
    """Make every kernel launch check the arch before it reaches Taichi.

    Kernels are launched from outside a render -- ``get_render_primitives()``
    builds them, benchmarks and unit tests call them directly -- and since the
    kernel modules no longer initialize Taichi at import, *something* has to.
    Doing it at the launch is the only placement that cannot be forgotten by a
    future call site, and it covers the second case too: a render device
    changed with no render in between leaves already-compiled kernels bound to
    the old arch, and this catches the next launch of one.

    Costs one :func:`render_device` call and an identity compare per launch --
    0.31 us measured, against ~72 us for the launch itself, and only on the
    outermost wrapper, so a fast-launch hit pays it once too.

    **Install this after** ``taichi_fast_launch.apply()``: that dispatcher goes
    straight to the C++ launch on a cache hit without calling through to the
    wrapper it replaced, so a guard installed *under* it would be skipped on
    exactly the repeat launches that most need checking.
    """
    from taichi.lang.kernel_impl import Kernel

    if getattr(Kernel, "_algan_arch_guard_installed", False):
        return
    previous_call = Kernel.__call__

    def guarded_call(self, *args, **kwargs):
        if _ARCH_READY_FOR is not render_device():
            ensure_taichi_for_render()
        return previous_call(self, *args, **kwargs)

    Kernel.__call__ = guarded_call
    Kernel._algan_arch_guard_installed = True
