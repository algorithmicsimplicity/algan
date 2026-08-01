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

from algan.settings._startup import _RENDER_DEVICE, _TAICHI_CACHE_DIRECTORY

_COMPILE_LOG_LOCK = threading.Lock()
_COMPILE_FRONTEND = {}
_COMPILE_NOTICE_CALLBACK = None
_COMPILE_NOTICE_THREAD_ID = None


def _compile_logging_enabled():
    return os.environ.get("ALGAN_LOG_TAICHI_COMPILES", "0") != "0"


def _compile_log_path():
    path = os.environ.get("ALGAN_TAICHI_COMPILE_LOG", "").strip()
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


def sync_devices():
    """Block until all pending torch-CUDA and Taichi device work has finished."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
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
    render_device = _RENDER_DEVICE
    if render_device.type == "cpu":
        return ti.cpu
    return ti.gpu


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
        "advanced_optimization": os.environ.get("ALGAN_ADV_OPT", "0") == "1",
        # debug=True inserts a bounds-check on *every* ndarray access;
        # the ray-trace megakernels do millions of array reads per ray
        # (BVH nodes, packed geometry), so it ran them ~11x slower with
        # no benefit to released renders. Keep it off (env ALGAN_TI_DEBUG=1
        # re-enables it for kernel development).
        "debug": os.environ.get("ALGAN_TI_DEBUG", "0") == "1",
        "offline_cache": True,
        # The default 100 MB cache LRU-evicts large megakernel
        # artifacts once several variants (general / no-PN / lean /
        # path-trace / wavefront, plus per-config rebuilds) are
        # compiled, forcing repeated ~minutes-long recompiles. Raise
        # it (disk-backed) so every kernel stays cached. The field is
        # a 32-bit int, so stay just under 2^31 bytes (~1.9 GB, still
        # 19x the default).
        "offline_cache_max_size_of_files": 1_000_000_000,
    }
    # Keep Algan's compiled kernels in a dedicated directory under Algan's
    # cache dir instead of Taichi's global default, so they never contend
    # with other Taichi programs for the LRU budget. A ti.init kwarg beats
    # the TI_OFFLINE_CACHE_FILE_PATH env var (Taichi warns and ignores the
    # env), so only pass it when the env var is unset to keep that standard
    # escape hatch working.
    if not os.environ.get("TI_OFFLINE_CACHE_FILE_PATH"):
        kwargs["offline_cache_file_path"] = str(_TAICHI_CACHE_DIRECTORY)
    max_reg = int(os.environ.get("ALGAN_GPU_MAX_REG", "0"))
    if max_reg > 0:
        kwargs["gpu_max_reg"] = max_reg
    opt_level = int(os.environ.get("ALGAN_OPT_LEVEL", "0"))
    if opt_level > 0:
        kwargs["opt_level"] = opt_level
    return kwargs


def init_taichi():
    """Initialize Taichi on Algan's selected backend, once."""
    if _already_initialized():
        _install_taichi_compile_logger()
        return
    ti.init(**taichi_init_kwargs())
    _install_taichi_compile_logger()
