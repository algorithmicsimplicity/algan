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

import ctypes
import hashlib
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

#: Ray types; a hit group record is ``type + RAY_TYPES * class``.
RAY_TYPE_SHADOW = 0
RAY_TYPE_GATHER = 1
RAY_TYPE_GATHER_BIN = 2
RAY_TYPES = 3

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


def _compile_programs(arch: str, kbuf: int, edge_epsilon: float) -> bytes:
    """The programs' PTX, from the disk cache when this exact build exists."""
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
        if cache.exists():
            return cache.read_bytes()
    except OSError:
        cache = None
    ptx = nvrtc.compile_ptx(source, _PROGRAMS, options, {_DEVICE_HEADER: header})
    if cache is not None:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_bytes(ptx)
            tmp.replace(cache)
        except OSError:
            pass
    return ptx


class OptixRuntime:
    """Device context, module, pipeline and shader binding table."""

    def __init__(self, device, kbuf: int, edge_epsilon: float):
        self.device = torch.device(device)
        torch.cuda.init()
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
                logCallbackLevel=2,
                validationMode=abi.OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF,
            )
            self.context = c_void_p()
            self.api(
                "optixDeviceContextCreate",
                c_void_p(_current_cu_context()),
                byref(options),
                byref(self.context),
            )
            self.rtcore_version = self._property(
                abi.OPTIX_DEVICE_PROPERTY_RTCORE_VERSION
            )
            self.max_primitives = self._property(
                abi.OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS
            )
            self.kbuf = int(kbuf)
            self._build_pipeline(
                _compile_programs(_ptx_architecture(self.device), kbuf, edge_epsilon)
            )
            self._params_device = torch.zeros(
                sizeof(_Params), dtype=torch.uint8, device=self.device
            )

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
    def _build_pipeline(self, ptx: bytes):
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
        self.module = c_void_p()
        result = self.api.fn["optixModuleCreate"](
            self.context,
            byref(module_options),
            byref(self._compile_options),
            ptx,
            len(ptx),
            log,
            byref(log_size),
            byref(self.module),
        )
        self.api.check(result, "optixModuleCreate", log.value)
        module = self.module.value

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
        self.pipeline = c_void_p()
        log_size = c_size_t(sizeof(log))
        result = self.api.fn["optixPipelineCreate"](
            self.context,
            byref(self._compile_options),
            byref(link),
            groups,
            len(descs),
            log,
            byref(log_size),
            byref(self.pipeline),
        )
        self.api.check(result, "optixPipelineCreate", log.value)

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
        """Launch ``program`` over ``width`` threads on the current stream."""
        if width <= 0:
            return
        if width >= 1 << 30:
            raise abi.OptixError(f"an OptiX launch is limited to 2^30 threads: {width}")
        stream = torch.cuda.current_stream(self.device)
        raw = bytes(params)
        # Pinned staging, copied in stream order: no host wait. The caching
        # host allocator does not hand the block out again until the copy has
        # run, and the launch reads the device copy in stream order too.
        host = torch.empty(len(raw), dtype=torch.uint8, pin_memory=True)
        ctypes.memmove(host.data_ptr(), raw, len(raw))
        self._params_device.copy_(host, non_blocking=True)
        self.api(
            "optixLaunch",
            self.pipeline,
            c_void_p(stream.cuda_stream),
            c_uint64(self._params_device.data_ptr()),
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


_RUNTIME = None
_RUNTIME_ERROR = None
_LOCK = threading.Lock()


def _device_key(device):
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def get_runtime(device, kbuf, edge_epsilon):
    """The process's runtime, or ``None`` with the reason logged once."""
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
                raise abi.OptixError(
                    "this GPU has no RT cores; OptiX would trace on the SMs, which "
                    "is no faster than Algan's own traversal"
                )
        except Exception as exc:  # noqa: BLE001 - any failure means "fall back"
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
        _RUNTIME = runtime
        return runtime


def runtime_error():
    """Why :func:`get_runtime` failed, or ``None``."""
    return _RUNTIME_ERROR
