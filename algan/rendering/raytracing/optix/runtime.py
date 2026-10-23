"""The process's OptiX device context, pipeline and launches.

One :class:`OptixRuntime` per process, created on first use on the CUDA
context PyTorch and Quadrants share (the device's primary context), and kept
for the life of the process. Every launch and build goes on PyTorch's current
stream -- the legacy default stream Quadrants launches on too -- so OptiX work
is ordered with the kernels around it and no extra synchronisation exists.

Device programs (``algan_optix_programs.cu``) are compiled by NVRTC once per
source/option/compiler combination and the PTX is kept under
``<cache dir>/optix``; OptiX keeps its own disk cache of the compiled module.
"""

from __future__ import annotations

import atexit
import contextlib
import ctypes
import hashlib
import os
import sys
import threading
from ctypes import byref, c_size_t, c_uint, c_uint64, c_void_p, sizeof
from pathlib import Path

import torch

from algan.logging.logger import PERF, get_logger
from algan.rendering.raytracing.optix import _abi as abi
from algan.rendering.raytracing.optix import _nvrtc as nvrtc

logger = get_logger(__name__)

_HERE = Path(__file__).resolve().parent
_PROGRAMS = "algan_optix_programs.cu"
_DEVICE_HEADER = "algan_optix_device.h"

#: Translucent shadow casters one ray records before the resolve falls back to
#: the software march (``AO_SHADOW_HITS`` in the device programs).
SHADOW_HITS = 4

#: Payload words the deepest trace uses (the gather's 24).
_PAYLOAD_WORDS = 24

#: Ray types (shadow, gather, gather's same-bin pass); a hit group record is
#: ``type + RAY_TYPES * class``.
RAY_TYPES = 3

#: Create the device context with OptiX's validation fully on and log at
#: level 4 (everything), into ``OptixRuntime.logs``. Slow; for a diagnostic
#: run of the tests, set before the runtime is first created.
VALIDATION = False

#: GAS SBT classes: an opaque primitive (any-hit disabled) and a translucent
#: one (any-hit called once per primitive).
CLASS_OPAQUE = 0
CLASS_TRANSLUCENT = 1


class _Params(ctypes.Structure):
    """``struct Params`` of the device programs, field for field."""

    _fields_ = [
        ("handles", c_uint64),
        ("num_handles", c_uint),
        ("pad0", c_uint),
        ("min_hit_distance", ctypes.c_float),
        ("inv_bin", ctypes.c_float),
        ("bin_width", ctypes.c_float),
        ("layer_offset", ctypes.c_float),
        ("queue_data", c_uint64),
        ("queue_valid", c_uint64),
        ("layout", c_uint64),
        ("occ_out", c_uint64),
        ("event_frame", c_uint64),
        ("event_src_prim", c_uint64),
        ("tri_obj", c_uint64),
        ("status", c_uint64),
        ("hits", c_uint64),
        ("num_rays", c_uint),
        ("num_events", c_uint),
        ("event_first", c_uint),
        ("tri_obj_frames", c_uint),
        ("tri_obj_count", c_uint),
        ("phase", c_uint),
        ("identity", c_uint),
        ("pad1", c_uint),
        ("eps_self", ctypes.c_float),
        ("eps_near", ctypes.c_float),
        ("pad2", ctypes.c_float),
        ("pad3", ctypes.c_float),
        ("active", c_uint64),
        ("rs_ro", c_uint64),
        ("rs_rd", c_uint64),
        ("rs_sca", c_uint64),
        ("rs_int", c_uint64),
        ("rs_pix", c_uint64),
        ("hit_f", c_uint64),
        ("hit_i", c_uint64),
        ("num_active", c_uint),
        ("time_start", ctypes.c_int),
        ("ray_offset", ctypes.c_int),
        ("pixels_per_frame", ctypes.c_int),
        ("ro_stride", c_uint),
        ("rd_stride", c_uint),
        ("sca_stride", c_uint),
        ("int_stride", c_uint),
        ("hit_f_q", c_uint),
        ("hit_f_c", c_uint),
        ("hit_i_q", c_uint),
        ("hit_i_c", c_uint),
        ("closest_only", c_uint),
        ("pad4", c_uint),
    ]


def new_params() -> _Params:
    return _Params()


def _cuda_driver():
    if sys.platform == "win32":
        return ctypes.WinDLL("nvcuda.dll")
    return ctypes.CDLL("libcuda.so.1")


def _current_cu_context() -> int:
    """The CUcontext current on this thread (PyTorch made it current)."""
    context = c_void_p()
    result = _cuda_driver().cuCtxGetCurrent(byref(context))
    if result != 0 or not context.value:
        raise abi.OptixError(f"no current CUDA context (cuCtxGetCurrent -> {result})")
    return context.value


def _ptx_architecture(device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    wanted = 10 * major + minor
    supported = [a for a in nvrtc.supported_architectures() if a <= wanted]
    if not supported:
        raise abi.OptixError(f"NVRTC cannot target compute capability {wanted}")
    return f"compute_{max(supported)}"


def _program_options(arch: str, kbuf: int, edge_epsilon: float) -> list[str]:
    return [
        f"--gpu-architecture={arch}",
        "-std=c++17",
        "-default-device",
        "--relocatable-device-code=true",
        f"-DAO_KBUF={int(kbuf)}",
        f"-DAO_SHADOW_HITS={SHADOW_HITS}",
        f"-DAO_TRIANGLE_EDGE_EPSILON={float(edge_epsilon)!r}f",
    ]


def _compile_programs(arch: str, kbuf: int, edge_epsilon: float, use_cache=True):
    """The programs' PTX and its cache file (``None`` when uncached).

    Read from ``<cache dir>/optix`` when this exact source, option and
    compiler combination was compiled before (and ``use_cache``); otherwise
    compiled and written there.
    """
    source = (_HERE / _PROGRAMS).read_text(encoding="utf-8")
    header = (_HERE / _DEVICE_HEADER).read_text(encoding="utf-8")
    options = _program_options(arch, kbuf, edge_epsilon)
    key = hashlib.sha256(
        "\0".join([source, header, *options, repr(nvrtc.version())]).encode()
    ).hexdigest()[:32]
    cache = None
    try:
        from algan.settings._startup import _CACHE_DIRECTORY

        cache = Path(_CACHE_DIRECTORY) / "optix" / f"programs-{key}.ptx"
        if use_cache and cache.exists():
            return cache.read_bytes(), cache
    except OSError:
        cache = None
    ptx = nvrtc.compile_ptx(source, _PROGRAMS, options, {_DEVICE_HEADER: header})
    if cache is not None:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            # Unique per process and thread, then an atomic rename: two
            # processes compiling at once never read a half-written file.
            tmp = cache.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_bytes(ptx)
            tmp.replace(cache)
        except OSError:
            cache = None
    return ptx, cache


class OptixRuntime:
    """Device context, module, pipeline and shader binding table."""

    def __init__(self, device, kbuf: int, edge_epsilon: float):
        self.device = torch.device(device)
        self.context = self.module = self.pipeline = None
        self._groups = None
        torch.cuda.init()
        try:
            self._create(kbuf, edge_epsilon)
        except BaseException:
            self.close()
            raise

    def _create(self, kbuf, edge_epsilon):
        with torch.cuda.device(self.device):
            # Make the primary context current on this thread.
            self._anchor = torch.zeros(1, device=self.device)
            self.api = abi.OptixApi()
            self.logs: list[str] = []

            def _log(level, tag, message, _data):
                self.logs.append(
                    f"[{level}][{tag.decode(errors='replace') if tag else ''}] "
                    f"{message.decode(errors='replace') if message else ''}"
                )

            self._log_callback = abi.OptixLogCallback(_log)
            options = abi.OptixDeviceContextOptions(
                logCallbackFunction=self._log_callback,
                logCallbackData=None,
                logCallbackLevel=4 if VALIDATION else 2,
                validationMode=(
                    abi.OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
                    if VALIDATION
                    else abi.OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF
                ),
            )
            context = c_void_p()
            self.api(
                "optixDeviceContextCreate",
                c_void_p(_current_cu_context()),
                byref(options),
                byref(context),
            )
            self.context = context
            self.rtcore_version = self._property(
                abi.OPTIX_DEVICE_PROPERTY_RTCORE_VERSION
            )
            self.max_primitives = self._property(
                abi.OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS
            )
            self.kbuf = int(kbuf)
            arch = _ptx_architecture(self.device)
            ptx, cache = _compile_programs(arch, kbuf, edge_epsilon)
            try:
                self._create_module(ptx)
            except abi.OptixError:
                if cache is None:
                    raise
                # A cached PTX OptiX refuses (truncated, or from another
                # driver's era) is deleted, and the programs compile afresh.
                cache.unlink(missing_ok=True)
                ptx, cache = _compile_programs(arch, kbuf, edge_epsilon, False)
                try:
                    self._create_module(ptx)
                except abi.OptixError:
                    if cache is not None:
                        cache.unlink(missing_ok=True)
                    raise
            self._build_pipeline()

    def _property(self, which) -> int:
        value = c_uint()
        try:
            self.api(
                "optixDeviceContextGetProperty",
                self.context,
                which,
                ctypes.addressof(value),
                sizeof(value),
            )
        except abi.OptixError:
            return 0
        return int(value.value)

    # ------------------------------------------------------------ pipeline
    def _create_module(self, ptx: bytes):
        self._compile_options = abi.OptixPipelineCompileOptions(
            usesMotionBlur=0,
            traversableGraphFlags=abi.OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
            numPayloadValues=_PAYLOAD_WORDS,
            numAttributeValues=2,
            exceptionFlags=abi.OPTIX_EXCEPTION_FLAG_NONE,
            pipelineLaunchParamsVariableName=b"params",
            pipelineLaunchParamsSizeInBytes=sizeof(_Params),
            usesPrimitiveTypeFlags=abi.OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE,
            allowOpacityMicromaps=0,
            allowClusteredGeometry=0,
        )
        module_options = abi.OptixModuleCompileOptions(
            maxRegisterCount=0,
            optLevel=abi.OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
            debugLevel=abi.OPTIX_COMPILE_DEBUG_LEVEL_NONE,
        )
        log = ctypes.create_string_buffer(16384)
        log_size = c_size_t(sizeof(log))
        module = c_void_p()
        result = self.api.fn["optixModuleCreate"](
            self.context,
            byref(module_options),
            byref(self._compile_options),
            ptx,
            len(ptx),
            log,
            byref(log_size),
            byref(module),
        )
        self.api.check(result, "optixModuleCreate", log.value)
        self.module = module

    def _build_pipeline(self):
        module = self.module.value
        log = ctypes.create_string_buffer(16384)

        def single(kind, field, entry):
            desc = abi.OptixProgramGroupDesc(kind=kind, flags=0)
            setattr(desc, field, abi.OptixProgramGroupSingleModule(module, entry))
            return desc

        def hit(anyhit, closesthit=None):
            desc = abi.OptixProgramGroupDesc(
                kind=abi.OPTIX_PROGRAM_GROUP_KIND_HITGROUP, flags=0
            )
            desc.hitgroup = abi.OptixProgramGroupHitgroup(
                module if closesthit else None,
                closesthit,
                module,
                anyhit,
                None,
                None,
            )
            return desc

        raygen = abi.OPTIX_PROGRAM_GROUP_KIND_RAYGEN
        miss = abi.OPTIX_PROGRAM_GROUP_KIND_MISS
        descs = [
            single(raygen, "raygen", b"__raygen__shadow"),
            single(raygen, "raygen", b"__raygen__gather"),
            single(miss, "miss", b"__miss__shadow"),
            single(miss, "miss", b"__miss__gather"),
            # Hit groups, record ``type + RAY_TYPES * class``.
            hit(b"__anyhit__shadow_opaque"),
            hit(b"__anyhit__gather_opaque", b"__closesthit__gather"),
            hit(b"__anyhit__gather_bin_opaque"),
            hit(b"__anyhit__shadow_translucent"),
            hit(b"__anyhit__gather_translucent", b"__closesthit__gather"),
            hit(b"__anyhit__gather_bin_translucent"),
        ]
        array = (abi.OptixProgramGroupDesc * len(descs))(*descs)
        groups = (c_void_p * len(descs))()
        group_options = abi.OptixProgramGroupOptions()
        log_size = c_size_t(sizeof(log))
        result = self.api.fn["optixProgramGroupCreate"](
            self.context,
            array,
            len(descs),
            byref(group_options),
            log,
            byref(log_size),
            groups,
        )
        self.api.check(result, "optixProgramGroupCreate", log.value)
        self._groups = groups
        link = abi.OptixPipelineLinkOptions(maxTraceDepth=1, maxTraversableGraphDepth=1)
        pipeline = c_void_p()
        log_size = c_size_t(sizeof(log))
        result = self.api.fn["optixPipelineCreate"](
            self.context,
            byref(self._compile_options),
            byref(link),
            groups,
            len(descs),
            log,
            byref(log_size),
            byref(pipeline),
        )
        self.api.check(result, "optixPipelineCreate", log.value)
        self.pipeline = pipeline

        size = abi.OPTIX_SBT_RECORD_HEADER_SIZE

        def record(group) -> bytes:
            buffer = ctypes.create_string_buffer(size)
            self.api(
                "optixSbtRecordPackHeader", c_void_p(group), ctypes.addressof(buffer)
            )
            return buffer.raw

        g = list(groups)
        blob = b"".join(
            [record(g[0]), record(g[1])]  # raygens
            + [record(g[2]), record(g[3]), record(g[3])]  # miss per ray type
            + [record(x) for x in g[4:10]]  # hit groups
        )
        self._sbt_device = torch.frombuffer(bytearray(blob), dtype=torch.uint8).to(
            self.device
        )
        base = self._sbt_device.data_ptr()
        assert base % abi.OPTIX_SBT_RECORD_ALIGNMENT == 0

        def table(raygen_index):
            return abi.OptixShaderBindingTable(
                raygenRecord=base + raygen_index * size,
                exceptionRecord=0,
                missRecordBase=base + 2 * size,
                missRecordStrideInBytes=size,
                missRecordCount=RAY_TYPES,
                hitgroupRecordBase=base + 5 * size,
                hitgroupRecordStrideInBytes=size,
                hitgroupRecordCount=2 * RAY_TYPES,
                callablesRecordBase=0,
                callablesRecordStrideInBytes=0,
                callablesRecordCount=0,
            )

        self._sbt = {"shadow": table(0), "gather": table(1)}

    # ------------------------------------------------------------ launch
    def launch(self, program: str, params: _Params, width: int):
        """Launch ``program`` over ``width`` threads on PyTorch's current stream.

        The parameters go through a pinned staging block and a device block of
        their own, both from PyTorch's caching allocators and both used on the
        launch's stream: the copy and the launch run in stream order, and
        neither block is handed out again before that stream is done with it,
        so launches from different threads or streams never share them.
        """
        if width <= 0:
            return
        if width >= 1 << 30:
            raise abi.OptixError(f"an OptiX launch is limited to 2^30 threads: {width}")
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            raw = bytes(params)
            host = torch.empty(len(raw), dtype=torch.uint8, pin_memory=True)
            ctypes.memmove(host.data_ptr(), raw, len(raw))
            device_params = torch.empty(len(raw), dtype=torch.uint8, device=self.device)
            device_params.copy_(host, non_blocking=True)
            self.api(
                "optixLaunch",
                self.pipeline,
                c_void_p(stream.cuda_stream),
                c_uint64(device_params.data_ptr()),
                len(raw),
                byref(self._sbt[program]),
                int(width),
                1,
                1,
            )

    # ------------------------------------------------------------ GAS
    def accel_sizes(self, build_input, options) -> abi.OptixAccelBufferSizes:
        sizes = abi.OptixAccelBufferSizes()
        self.api(
            "optixAccelComputeMemoryUsage",
            self.context,
            byref(options),
            byref(build_input),
            1,
            byref(sizes),
        )
        return sizes

    def accel_build(
        self, build_input, options, temp_ptr, temp_bytes, out_ptr, out_bytes, emit=None
    ) -> int:
        handle = c_uint64()
        stream = torch.cuda.current_stream(self.device)
        self.api(
            "optixAccelBuild",
            self.context,
            c_void_p(stream.cuda_stream),
            byref(options),
            byref(build_input),
            1,
            c_uint64(temp_ptr),
            temp_bytes,
            c_uint64(out_ptr),
            out_bytes,
            byref(handle),
            byref(emit) if emit is not None else None,
            1 if emit is not None else 0,
        )
        return int(handle.value)

    def accel_compact(self, handle, out_ptr, out_bytes) -> int:
        compacted = c_uint64()
        stream = torch.cuda.current_stream(self.device)
        self.api(
            "optixAccelCompact",
            self.context,
            c_void_p(stream.cuda_stream),
            c_uint64(handle),
            c_uint64(out_ptr),
            out_bytes,
            byref(compacted),
        )
        return int(compacted.value)

    # ------------------------------------------------------------ teardown
    def close(self):
        """Destroy the pipeline, program groups, module and context, in that
        order. Safe to call twice and on a half-built runtime; the ctypes log
        callback stays referenced by ``self`` until the context is gone.
        """
        api = getattr(self, "api", None)
        if api is None:
            return
        # Tear down whatever state CUDA is in.
        with contextlib.suppress(Exception):
            torch.cuda.synchronize(self.device)
        steps = []
        if self.pipeline is not None:
            steps.append(("optixPipelineDestroy", self.pipeline))
        for group in list(self._groups or []):
            if group:
                steps.append(("optixProgramGroupDestroy", c_void_p(group)))
        if self.module is not None:
            steps.append(("optixModuleDestroy", self.module))
        if self.context is not None:
            steps.append(("optixDeviceContextDestroy", self.context))
        self.pipeline = self.module = self.context = None
        self._groups = None
        for name, handle in steps:
            with contextlib.suppress(abi.OptixError):
                api(name, handle)


_RUNTIME = None
_RUNTIME_ERROR = None
_LOCK = threading.Lock()


def _shutdown():
    """Process exit: tear the runtime down.

    Run while the interpreter -- and the ctypes log callback the context calls
    into -- is still alive.
    """
    global _RUNTIME
    runtime, _RUNTIME = _RUNTIME, None
    if runtime is not None:
        runtime.close()


def _transient(exc) -> bool:
    """A creation failure the next batch may not meet again."""
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    if isinstance(exc, abi.OptixError):
        return exc.transient
    from algan.utils.memory_utils import is_cuda_oom

    return isinstance(exc, RuntimeError) and is_cuda_oom(exc)


def _device_key(device):
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def get_runtime(device, kbuf, edge_epsilon):
    """The process's runtime, or ``None``.

    A failure that will recur -- no OptiX runtime, the ABI refused, no RT
    cores, programs that do not compile -- is logged once and remembered for
    the process, the warm daemon included. One that may not -- device or host
    memory, a CUDA error -- is logged and tried again at the next call.
    """
    global _RUNTIME, _RUNTIME_ERROR
    device = _device_key(device)
    with _LOCK:
        if _RUNTIME is not None:
            if _RUNTIME.device == device and _RUNTIME.kbuf == int(kbuf):
                return _RUNTIME
            return None
        if _RUNTIME_ERROR is not None:
            return None
        try:
            runtime = OptixRuntime(device, kbuf, edge_epsilon)
            if runtime.rtcore_version <= 0:
                runtime.close()
                raise abi.OptixError(
                    "this GPU has no RT cores; OptiX would trace on the SMs, which "
                    "is no faster than Algan's own traversal"
                )
        except Exception as exc:  # noqa: BLE001 - any failure means "fall back"
            if _transient(exc):
                logger.log(
                    PERF,
                    f"OptiX could not start this time ({exc}); this batch keeps "
                    "the software traversal and the next one tries again.",
                )
                return None
            _RUNTIME_ERROR = exc
            logger.warning(
                "SETTINGS.raytracing.experimental.optix is on, but OptiX is not "
                "usable here, so the software traversal renders instead: %s",
                exc,
            )
            return None
        logger.log(
            PERF,
            f"OptiX side path ready: RT core version {runtime.rtcore_version}, "
            f"runtime {runtime.api.path}",
        )
        atexit.register(_shutdown)
        _RUNTIME = runtime
        return runtime
