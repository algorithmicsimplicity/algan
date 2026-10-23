"""The OptiX 9.1 host ABI subset Algan calls, re-declared for ctypes.

Nothing here comes from NVIDIA's headers, which are proprietary and are not
vendored into this MIT repository: the enum values, struct layouts and the
order of the driver's function table are restated from the published ABI, the
way Mitsuba's BSD ``optix_api.h`` and Dr.Jit do it. The OptiX *runtime* is the
display driver's own ``nvoptix.dll`` / ``libnvoptix.so.1``; the function table
is fetched with ``optixQueryFunctionTable`` for ABI 118 (OptiX 9.1), and a
driver that does not speak that ABI refuses the query, which is how an old
driver falls back to the software walk.

No compiler can check a ``sizeof`` here, so the sizes the x64 C layouts have
are asserted at import (``_LAYOUT_SIZES``), and ``tests/unit_tests/test_optix.py``
exercises every call on a real device.
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    Union,
    byref,
    c_char,
    c_char_p,
    c_float,
    c_int,
    c_size_t,
    c_uint,
    c_uint64,
    c_ushort,
    c_void_p,
    sizeof,
)
from pathlib import Path

#: ``optixQueryFunctionTable`` ABI version of OptiX 9.1.
OPTIX_ABI_VERSION = 118

# ----------------------------------------------------------------- enums
OPTIX_SUCCESS = 0
OPTIX_BUILD_INPUT_TYPE_TRIANGLES = 0x2141
OPTIX_VERTEX_FORMAT_FLOAT3 = 0x2121
OPTIX_INDICES_FORMAT_UNSIGNED_INT3 = 0x2103
OPTIX_BUILD_OPERATION_BUILD = 0x2161
OPTIX_BUILD_FLAG_ALLOW_COMPACTION = 1 << 1
OPTIX_BUILD_FLAG_PREFER_FAST_TRACE = 1 << 2
OPTIX_PROPERTY_TYPE_COMPACTED_SIZE = 0x2181
OPTIX_GEOMETRY_FLAG_NONE = 0
OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT = 1 << 0
OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL = 1 << 1
OPTIX_PROGRAM_GROUP_KIND_RAYGEN = 0x2421
OPTIX_PROGRAM_GROUP_KIND_MISS = 0x2422
OPTIX_PROGRAM_GROUP_KIND_HITGROUP = 0x2424
OPTIX_COMPILE_OPTIMIZATION_DEFAULT = 0
OPTIX_COMPILE_DEBUG_LEVEL_NONE = 0x2350
OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS = 1 << 0
OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE = 1 << 31
OPTIX_EXCEPTION_FLAG_NONE = 0
OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF = 0
OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL = 0xFFFFFFFF
OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS = 0x2003
OPTIX_DEVICE_PROPERTY_RTCORE_VERSION = 0x2005
OPTIX_SBT_RECORD_HEADER_SIZE = 32
OPTIX_SBT_RECORD_ALIGNMENT = 16
OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT = 128

# ----------------------------------------------------------------- structs
OptixLogCallback = CFUNCTYPE(None, c_uint, c_char_p, c_char_p, c_void_p)


class OptixDeviceContextOptions(Structure):
    _fields_ = [
        ("logCallbackFunction", OptixLogCallback),
        ("logCallbackData", c_void_p),
        ("logCallbackLevel", c_int),
        ("validationMode", c_int),
    ]


class OptixBuildInputOpacityMicromap(Structure):
    _fields_ = [
        ("indexingMode", c_int),
        ("opacityMicromapArray", c_uint64),
        ("indexBuffer", c_uint64),
        ("indexSizeInBytes", c_uint),
        ("indexStrideInBytes", c_uint),
        ("indexOffset", c_uint),
        ("numMicromapUsageCounts", c_uint),
        ("micromapUsageCounts", c_void_p),
    ]


class OptixBuildInputTriangleArray(Structure):
    _fields_ = [
        ("vertexBuffers", POINTER(c_uint64)),
        ("numVertices", c_uint),
        ("vertexFormat", c_int),
        ("vertexStrideInBytes", c_uint),
        ("indexBuffer", c_uint64),
        ("numIndexTriplets", c_uint),
        ("indexFormat", c_int),
        ("indexStrideInBytes", c_uint),
        ("preTransform", c_uint64),
        ("flags", POINTER(c_uint)),
        ("numSbtRecords", c_uint),
        ("sbtIndexOffsetBuffer", c_uint64),
        ("sbtIndexOffsetSizeInBytes", c_uint),
        ("sbtIndexOffsetStrideInBytes", c_uint),
        ("primitiveIndexOffset", c_uint),
        ("transformFormat", c_int),
        ("opacityMicromap", OptixBuildInputOpacityMicromap),
    ]


class _BuildInputUnion(Union):
    # The C union is as large as its largest member, padded to 1024 bytes.
    _fields_ = [("pad", c_char * 1024), ("triangleArray", OptixBuildInputTriangleArray)]


class OptixBuildInput(Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", c_int), ("u", _BuildInputUnion)]


class OptixMotionOptions(Structure):
    _fields_ = [
        ("numKeys", c_ushort),
        ("flags", c_ushort),
        ("timeBegin", c_float),
        ("timeEnd", c_float),
    ]


class OptixAccelBuildOptions(Structure):
    _fields_ = [
        ("buildFlags", c_uint),
        ("operation", c_int),
        ("motionOptions", OptixMotionOptions),
    ]


class OptixAccelBufferSizes(Structure):
    _fields_ = [
        ("outputSizeInBytes", c_size_t),
        ("tempSizeInBytes", c_size_t),
        ("tempUpdateSizeInBytes", c_size_t),
    ]


class OptixAccelEmitDesc(Structure):
    _fields_ = [("result", c_uint64), ("type", c_int)]


class OptixModuleCompileOptions(Structure):
    _fields_ = [
        ("maxRegisterCount", c_int),
        ("optLevel", c_int),
        ("debugLevel", c_int),
        ("boundValues", c_void_p),
        ("numBoundValues", c_uint),
        ("numPayloadTypes", c_uint),
        ("payloadTypes", c_void_p),
        ("baseModule", c_void_p),
    ]


class OptixPipelineCompileOptions(Structure):
    _fields_ = [
        ("usesMotionBlur", c_int),
        ("traversableGraphFlags", c_uint),
        ("numPayloadValues", c_int),
        ("numAttributeValues", c_int),
        ("exceptionFlags", c_uint),
        ("pipelineLaunchParamsVariableName", c_char_p),
        ("pipelineLaunchParamsSizeInBytes", c_size_t),
        ("usesPrimitiveTypeFlags", c_uint),
        ("allowOpacityMicromaps", c_int),
        ("allowClusteredGeometry", c_int),
    ]


class OptixProgramGroupSingleModule(Structure):
    _fields_ = [("module", c_void_p), ("entryFunctionName", c_char_p)]


class OptixProgramGroupHitgroup(Structure):
    _fields_ = [
        ("moduleCH", c_void_p),
        ("entryFunctionNameCH", c_char_p),
        ("moduleAH", c_void_p),
        ("entryFunctionNameAH", c_char_p),
        ("moduleIS", c_void_p),
        ("entryFunctionNameIS", c_char_p),
    ]


class _ProgramGroupUnion(Union):
    _fields_ = [
        ("hitgroup", OptixProgramGroupHitgroup),
        ("raygen", OptixProgramGroupSingleModule),
        ("miss", OptixProgramGroupSingleModule),
        ("exception", OptixProgramGroupSingleModule),
    ]


class OptixProgramGroupDesc(Structure):
    _anonymous_ = ("u",)
    _fields_ = [("kind", c_int), ("flags", c_uint), ("u", _ProgramGroupUnion)]


class OptixProgramGroupOptions(Structure):
    _fields_ = [("payloadType", c_void_p)]


class OptixPipelineLinkOptions(Structure):
    _fields_ = [
        ("maxTraceDepth", c_uint),
        ("maxContinuationCallableDepth", c_uint),
        ("maxDirectCallableDepthFromState", c_uint),
        ("maxDirectCallableDepthFromTraversal", c_uint),
        ("maxTraversableGraphDepth", c_uint),
    ]


class OptixShaderBindingTable(Structure):
    _fields_ = [
        ("raygenRecord", c_uint64),
        ("exceptionRecord", c_uint64),
        ("missRecordBase", c_uint64),
        ("missRecordStrideInBytes", c_uint),
        ("missRecordCount", c_uint),
        ("hitgroupRecordBase", c_uint64),
        ("hitgroupRecordStrideInBytes", c_uint),
        ("hitgroupRecordCount", c_uint),
        ("callablesRecordBase", c_uint64),
        ("callablesRecordStrideInBytes", c_uint),
        ("callablesRecordCount", c_uint),
    ]


#: The x64 C sizes of the structs passed by pointer: a field dropped or
#: reordered here would be read by the driver as garbage, not refused.
_LAYOUT_SIZES = {
    OptixBuildInput: 1032,
    OptixBuildInputTriangleArray: 144,
    OptixProgramGroupDesc: 56,
    OptixShaderBindingTable: 64,
    OptixPipelineCompileOptions: 56,
    OptixModuleCompileOptions: 48,
    OptixAccelBuildOptions: 20,
    OptixAccelBufferSizes: 24,
    OptixAccelEmitDesc: 16,
    OptixPipelineLinkOptions: 20,
    OptixDeviceContextOptions: 24,
}
for _struct, _size in _LAYOUT_SIZES.items():
    assert sizeof(_struct) == _size, (_struct.__name__, sizeof(_struct), _size)

# ----------------------------------------------------------------- function table
#: ``OptixFunctionTable`` for ABI 118, every entry in declaration order: the
#: driver fills the whole struct, so its size and order must match exactly,
#: including the entries Algan never calls.
_TABLE_NAMES = (
    "optixGetErrorName",
    "optixGetErrorString",
    "optixDeviceContextCreate",
    "optixDeviceContextDestroy",
    "optixDeviceContextGetProperty",
    "optixDeviceContextSetLogCallback",
    "optixDeviceContextSetCacheEnabled",
    "optixDeviceContextSetCacheLocation",
    "optixDeviceContextSetCacheDatabaseSizes",
    "optixDeviceContextGetCacheEnabled",
    "optixDeviceContextGetCacheLocation",
    "optixDeviceContextGetCacheDatabaseSizes",
    "optixModuleCreate",
    "optixModuleCreateWithTasks",
    "optixModuleGetCompilationState",
    "optixModuleCancelCreation",
    "optixStub",
    "optixDeviceContextCancelCreations",
    "optixModuleDestroy",
    "optixBuiltinISModuleGet",
    "optixTaskExecute",
    "optixTaskGetSerializationKey",
    "optixTaskSerializeOutput",
    "optixTaskDeserializeOutput",
    "optixProgramGroupCreate",
    "optixProgramGroupDestroy",
    "optixProgramGroupGetStackSize",
    "optixPipelineCreate",
    "optixPipelineDestroy",
    "optixPipelineSetStackSizeFromCallDepths",
    "optixPipelineSetStackSize",
    "optixPipelineSymbolMemcpyAsync",
    "optixAccelComputeMemoryUsage",
    "optixAccelBuild",
    "optixAccelGetRelocationInfo",
    "optixCheckRelocationCompatibility",
    "optixAccelRelocate",
    "optixAccelCompact",
    "optixAccelEmitProperty",
    "optixConvertPointerToTraversableHandle",
    "optixOpacityMicromapArrayComputeMemoryUsage",
    "optixOpacityMicromapArrayBuild",
    "optixOpacityMicromapArrayGetRelocationInfo",
    "optixOpacityMicromapArrayRelocate",
    "stub1",
    "stub2",
    "optixClusterAccelComputeMemoryUsage",
    "optixClusterAccelBuild",
    "optixSbtRecordPackHeader",
    "optixLaunch",
    "optixCoopVecMatrixConvert",
    "optixCoopVecMatrixComputeSize",
    "optixDenoiserCreate",
    "optixDenoiserDestroy",
    "optixDenoiserComputeMemoryResources",
    "optixDenoiserSetup",
    "optixDenoiserInvoke",
    "optixDenoiserComputeIntensity",
    "optixDenoiserComputeAverageColor",
    "optixDenoiserCreateWithUserModel",
)


class OptixFunctionTable(Structure):
    _fields_ = [(name, c_void_p) for name in _TABLE_NAMES]


_P = POINTER
#: Prototypes of the entries Algan calls.
_PROTOTYPES = {
    "optixGetErrorName": CFUNCTYPE(c_char_p, c_int),
    "optixGetErrorString": CFUNCTYPE(c_char_p, c_int),
    "optixDeviceContextCreate": CFUNCTYPE(
        c_int, c_void_p, _P(OptixDeviceContextOptions), _P(c_void_p)
    ),
    "optixDeviceContextDestroy": CFUNCTYPE(c_int, c_void_p),
    "optixDeviceContextGetProperty": CFUNCTYPE(
        c_int, c_void_p, c_int, c_void_p, c_size_t
    ),
    "optixModuleCreate": CFUNCTYPE(
        c_int,
        c_void_p,
        _P(OptixModuleCompileOptions),
        _P(OptixPipelineCompileOptions),
        c_char_p,
        c_size_t,
        c_char_p,
        _P(c_size_t),
        _P(c_void_p),
    ),
    "optixModuleDestroy": CFUNCTYPE(c_int, c_void_p),
    "optixProgramGroupCreate": CFUNCTYPE(
        c_int,
        c_void_p,
        _P(OptixProgramGroupDesc),
        c_uint,
        _P(OptixProgramGroupOptions),
        c_char_p,
        _P(c_size_t),
        _P(c_void_p),
    ),
    "optixProgramGroupDestroy": CFUNCTYPE(c_int, c_void_p),
    "optixPipelineCreate": CFUNCTYPE(
        c_int,
        c_void_p,
        _P(OptixPipelineCompileOptions),
        _P(OptixPipelineLinkOptions),
        _P(c_void_p),
        c_uint,
        c_char_p,
        _P(c_size_t),
        _P(c_void_p),
    ),
    "optixPipelineDestroy": CFUNCTYPE(c_int, c_void_p),
    "optixAccelComputeMemoryUsage": CFUNCTYPE(
        c_int,
        c_void_p,
        _P(OptixAccelBuildOptions),
        _P(OptixBuildInput),
        c_uint,
        _P(OptixAccelBufferSizes),
    ),
    "optixAccelBuild": CFUNCTYPE(
        c_int,
        c_void_p,
        c_void_p,
        _P(OptixAccelBuildOptions),
        _P(OptixBuildInput),
        c_uint,
        c_uint64,
        c_size_t,
        c_uint64,
        c_size_t,
        _P(c_uint64),
        _P(OptixAccelEmitDesc),
        c_uint,
    ),
    "optixAccelCompact": CFUNCTYPE(
        c_int, c_void_p, c_void_p, c_uint64, c_uint64, c_size_t, _P(c_uint64)
    ),
    "optixSbtRecordPackHeader": CFUNCTYPE(c_int, c_void_p, c_void_p),
    "optixLaunch": CFUNCTYPE(
        c_int,
        c_void_p,
        c_void_p,
        c_uint64,
        c_size_t,
        _P(OptixShaderBindingTable),
        c_uint,
        c_uint,
        c_uint,
    ),
}


#: Result codes that can clear up on their own -- memory, the CUDA state, the
#: disk cache -- as against a missing runtime, a refused ABI or a program
#: that does not compile, which recur on every attempt.
_TRANSIENT_RESULTS = frozenset(
    {
        7002,  # host out of memory
        7004,  # file I/O
        7010,  # disk cache: invalid path
        7011,  # disk cache: permission
        7012,  # disk cache: database
        7013,  # disk cache: invalid data
        7052,  # CUDA not initialized
        7807,  # device out of memory
        7900,  # CUDA error
    }
)


class OptixError(RuntimeError):
    """An OptiX call, or loading the runtime, failed.

    ``code`` is the call's result code, or ``None`` when no call failed (the
    runtime library is missing, the ABI is refused, the device has no RT
    cores). ``transient`` says whether trying again later could succeed.
    """

    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code

    @property
    def transient(self):
        return self.code in _TRANSIENT_RESULTS


def _windows_file_version(path: Path) -> tuple[int, ...]:
    """A DLL's file version, for picking the newest of several driver copies."""
    ver = ctypes.windll.version
    size = ver.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        return (0,)
    buf = ctypes.create_string_buffer(size)
    if not ver.GetFileVersionInfoW(str(path), 0, size, buf):
        return (0,)
    block = c_void_p()
    length = c_uint()
    if not ver.VerQueryValueW(buf, "\\", byref(block), byref(length)):
        return (0,)
    words = ctypes.cast(block, POINTER(c_uint * 13)).contents
    # VS_FIXEDFILEINFO: signature, struct version, then the file version.
    ms, ls = words[2], words[3]
    return (ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF)


def _windows_runtime_candidates() -> list[Path]:
    """Every ``nvoptix.dll`` of the installed display driver.

    The driver installs it beside its OpenGL ICD, which the display adapter's
    registry key names (``OpenGLDriverName``); ``System32`` is checked first
    for drivers that still copy it there.
    """
    import winreg

    system = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32"
    if (system / "nvoptix.dll").exists():
        return [system / "nvoptix.dll"]
    found = []
    adapters = (
        r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    )
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, adapters) as key:
            index = 0
            while True:
                try:
                    sub = winreg.EnumKey(key, index)
                except OSError:
                    break
                index += 1
                if not re.fullmatch(r"\d{4}", sub):
                    continue
                try:
                    with winreg.OpenKey(key, sub) as adapter:
                        value, _ = winreg.QueryValueEx(adapter, "OpenGLDriverName")
                except OSError:
                    continue
                for entry in value if isinstance(value, list) else [value]:
                    candidate = Path(entry).parent / "nvoptix.dll"
                    if candidate.exists():
                        found.append(candidate)
    except OSError:
        pass
    if not found:
        store = system / "DriverStore" / "FileRepository"
        found = list(store.glob("nv_dispi.inf_*/nvoptix.dll"))
    return sorted(set(found), key=_windows_file_version, reverse=True)


def load_runtime():
    """Open the driver's OptiX runtime library; raise :class:`OptixError`."""
    if sys.platform == "win32":
        candidates = _windows_runtime_candidates()
        if not candidates:
            raise OptixError("nvoptix.dll not found (no NVIDIA display driver?)")
        return ctypes.CDLL(str(candidates[0])), candidates[0]
    if sys.platform.startswith("linux"):
        try:
            return ctypes.CDLL("libnvoptix.so.1"), Path("libnvoptix.so.1")
        except OSError as exc:
            raise OptixError(f"libnvoptix.so.1 not loadable: {exc}") from exc
    raise OptixError(f"OptiX is not available on {sys.platform}")


class OptixApi:
    """The driver's function table for ABI 118, with checked calls."""

    def __init__(self):
        self._dll, self.path = load_runtime()
        query = self._dll.optixQueryFunctionTable
        query.restype = c_int
        query.argtypes = [c_int, c_uint, c_void_p, c_void_p, c_void_p, c_size_t]
        self.table = OptixFunctionTable()
        result = query(
            OPTIX_ABI_VERSION, 0, None, None, byref(self.table), sizeof(self.table)
        )
        if result != OPTIX_SUCCESS:
            raise OptixError(
                f"the driver's OptiX runtime ({self.path}) does not offer function "
                f"table ABI {OPTIX_ABI_VERSION} (OptiX 9.1): error {result}; a newer "
                "NVIDIA driver is needed"
            )
        self.fn = {}
        for name, prototype in _PROTOTYPES.items():
            pointer = getattr(self.table, name)
            if not pointer:
                raise OptixError(f"OptiX function table entry {name} is null")
            self.fn[name] = prototype(pointer)
        # A table read at the wrong offsets fails here rather than at a launch:
        # the first entry must name the success code.
        name = self.fn["optixGetErrorName"](OPTIX_SUCCESS)
        if name != b"OPTIX_SUCCESS":
            raise OptixError(
                f"the OptiX function table from {self.path} does not have the "
                f"expected layout (optixGetErrorName(0) = {name!r})"
            )

    def check(self, result: int, what: str, log: bytes | None = None):
        if result == OPTIX_SUCCESS:
            return
        name = self.fn["optixGetErrorName"](result)
        text = self.fn["optixGetErrorString"](result)
        detail = f"\n{log.decode(errors='replace')}" if log else ""
        raise OptixError(
            f"{what} failed: {result} {name.decode() if name else ''} "
            f"{text.decode() if text else ''}{detail}",
            code=result,
        )

    def __call__(self, name, *args):
        self.check(self.fn[name](*args), name)
