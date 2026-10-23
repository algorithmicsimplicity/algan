"""The re-declared OptiX 9.1 host ABI, held to the x64 C layout.

``optix/_abi.py`` restates the struct layouts and enum values Algan passes to
the driver; nothing checks them at a call -- a struct with a field out of
place is not refused, it is misread. The numbers below are the offsets and
sizes the C rules give the OptiX 9.1.0 declarations (``OPTIX_ABI_VERSION``
118) on x64: natural alignment, pointers and ``size_t`` 8 bytes, enums 4. They
were derived by hand from the published headers and are written here as
numbers only. CPU-only; nothing is loaded.
"""

from __future__ import annotations

from ctypes import sizeof

import pytest

from algan.rendering.raytracing.optix import _abi
from algan.rendering.raytracing.optix import runtime as optix_runtime

#: struct -> (size, {field: offset}) for every field the struct declares.
_LAYOUTS = {
    "OptixDeviceContextOptions": (
        24,
        {
            "logCallbackFunction": 0,
            "logCallbackData": 8,
            "logCallbackLevel": 16,
            "validationMode": 20,
        },
    ),
    "OptixBuildInputOpacityMicromap": (
        48,
        {
            "indexingMode": 0,
            "opacityMicromapArray": 8,
            "indexBuffer": 16,
            "indexSizeInBytes": 24,
            "indexStrideInBytes": 28,
            "indexOffset": 32,
            "numMicromapUsageCounts": 36,
            "micromapUsageCounts": 40,
        },
    ),
    "OptixBuildInputTriangleArray": (
        144,
        {
            "vertexBuffers": 0,
            "numVertices": 8,
            "vertexFormat": 12,
            "vertexStrideInBytes": 16,
            "indexBuffer": 24,
            "numIndexTriplets": 32,
            "indexFormat": 36,
            "indexStrideInBytes": 40,
            "preTransform": 48,
            "flags": 56,
            "numSbtRecords": 64,
            "sbtIndexOffsetBuffer": 72,
            "sbtIndexOffsetSizeInBytes": 80,
            "sbtIndexOffsetStrideInBytes": 84,
            "primitiveIndexOffset": 88,
            "transformFormat": 92,
            "opacityMicromap": 96,
        },
    ),
    # The input union is padded to 1024 bytes and holds pointers.
    "OptixBuildInput": (1032, {"type": 0, "u": 8}),
    "OptixMotionOptions": (
        12,
        {"numKeys": 0, "flags": 2, "timeBegin": 4, "timeEnd": 8},
    ),
    "OptixAccelBuildOptions": (
        20,
        {"buildFlags": 0, "operation": 4, "motionOptions": 8},
    ),
    "OptixAccelBufferSizes": (
        24,
        {"outputSizeInBytes": 0, "tempSizeInBytes": 8, "tempUpdateSizeInBytes": 16},
    ),
    "OptixAccelEmitDesc": (16, {"result": 0, "type": 8}),
    "OptixModuleCompileOptions": (
        48,
        {
            "maxRegisterCount": 0,
            "optLevel": 4,
            "debugLevel": 8,
            "boundValues": 16,
            "numBoundValues": 24,
            "numPayloadTypes": 28,
            "payloadTypes": 32,
            "baseModule": 40,
        },
    ),
    "OptixPipelineCompileOptions": (
        56,
        {
            "usesMotionBlur": 0,
            "traversableGraphFlags": 4,
            "numPayloadValues": 8,
            "numAttributeValues": 12,
            "exceptionFlags": 16,
            "pipelineLaunchParamsVariableName": 24,
            "pipelineLaunchParamsSizeInBytes": 32,
            "usesPrimitiveTypeFlags": 40,
            "allowOpacityMicromaps": 44,
            "allowClusteredGeometry": 48,
        },
    ),
    "OptixProgramGroupSingleModule": (16, {"module": 0, "entryFunctionName": 8}),
    "OptixProgramGroupHitgroup": (
        48,
        {
            "moduleCH": 0,
            "entryFunctionNameCH": 8,
            "moduleAH": 16,
            "entryFunctionNameAH": 24,
            "moduleIS": 32,
            "entryFunctionNameIS": 40,
        },
    ),
    # kind, flags, then the union of the program kinds (the hit group, six
    # pointers, is the largest).
    "OptixProgramGroupDesc": (56, {"kind": 0, "flags": 4, "u": 8}),
    "OptixProgramGroupOptions": (8, {"payloadType": 0}),
    "OptixPipelineLinkOptions": (
        20,
        {
            "maxTraceDepth": 0,
            "maxContinuationCallableDepth": 4,
            "maxDirectCallableDepthFromState": 8,
            "maxDirectCallableDepthFromTraversal": 12,
            "maxTraversableGraphDepth": 16,
        },
    ),
    "OptixShaderBindingTable": (
        64,
        {
            "raygenRecord": 0,
            "exceptionRecord": 8,
            "missRecordBase": 16,
            "missRecordStrideInBytes": 24,
            "missRecordCount": 28,
            "hitgroupRecordBase": 32,
            "hitgroupRecordStrideInBytes": 40,
            "hitgroupRecordCount": 44,
            "callablesRecordBase": 48,
            "callablesRecordStrideInBytes": 56,
            "callablesRecordCount": 60,
        },
    ),
}


@pytest.mark.parametrize("name", sorted(_LAYOUTS))
def test_struct_sizes_and_field_offsets_match_the_c_layout(name):
    struct = getattr(_abi, name)
    size, offsets = _LAYOUTS[name]
    assert sizeof(struct) == size
    declared = [field[0] for field in struct._fields_]
    assert declared == list(offsets), "fields missing, extra or out of order"
    for field, offset in offsets.items():
        assert getattr(struct, field).offset == offset, field


def test_union_members_sit_at_the_union_start():
    assert _abi._BuildInputUnion.triangleArray.offset == 0
    assert sizeof(_abi._BuildInputUnion) == 1024
    for member in ("hitgroup", "raygen", "miss", "exception"):
        assert getattr(_abi._ProgramGroupUnion, member).offset == 0
    assert sizeof(_abi._ProgramGroupUnion) == 48


def test_the_function_table_has_every_abi_118_entry():
    # 60 entries through the denoiser's, pointer-sized, in declaration order;
    # the first and the launch entry anchor the order.
    assert sizeof(_abi.OptixFunctionTable) == 60 * 8
    names = list(_abi._TABLE_NAMES)
    assert names[0] == "optixGetErrorName"
    assert names.index("optixAccelBuild") == 33
    assert names.index("optixSbtRecordPackHeader") == 48
    assert names.index("optixLaunch") == 49
    assert names[-1] == "optixDenoiserCreateWithUserModel"


def test_enum_values_and_constants():
    expected = {
        "OPTIX_ABI_VERSION": 118,
        "OPTIX_SUCCESS": 0,
        "OPTIX_BUILD_INPUT_TYPE_TRIANGLES": 0x2141,
        "OPTIX_VERTEX_FORMAT_FLOAT3": 0x2121,
        "OPTIX_INDICES_FORMAT_UNSIGNED_INT3": 0x2103,
        "OPTIX_BUILD_OPERATION_BUILD": 0x2161,
        "OPTIX_BUILD_FLAG_ALLOW_COMPACTION": 2,
        "OPTIX_BUILD_FLAG_PREFER_FAST_TRACE": 4,
        "OPTIX_PROPERTY_TYPE_COMPACTED_SIZE": 0x2181,
        "OPTIX_GEOMETRY_FLAG_NONE": 0,
        "OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT": 1,
        "OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL": 2,
        "OPTIX_PROGRAM_GROUP_KIND_RAYGEN": 0x2421,
        "OPTIX_PROGRAM_GROUP_KIND_MISS": 0x2422,
        "OPTIX_PROGRAM_GROUP_KIND_HITGROUP": 0x2424,
        "OPTIX_COMPILE_OPTIMIZATION_DEFAULT": 0,
        "OPTIX_COMPILE_DEBUG_LEVEL_NONE": 0x2350,
        "OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS": 1,
        "OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE": 1 << 31,
        "OPTIX_EXCEPTION_FLAG_NONE": 0,
        "OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF": 0,
        "OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL": 0xFFFFFFFF,
        "OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS": 0x2003,
        "OPTIX_DEVICE_PROPERTY_RTCORE_VERSION": 0x2005,
        "OPTIX_SBT_RECORD_HEADER_SIZE": 32,
        "OPTIX_SBT_RECORD_ALIGNMENT": 16,
        "OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT": 128,
    }
    for name, value in expected.items():
        assert getattr(_abi, name) == value, name


def test_launch_params_match_the_device_struct():
    """``runtime._Params`` mirrors ``struct Params`` of the device programs.

    The programs assert the struct's size at compile time; NVRTC has no
    ``offsetof``, so the offsets are held here.
    """
    params = optix_runtime._Params
    assert sizeof(params) == 272
    for field, offset in {
        "handles": 0,
        "num_handles": 8,
        "min_hit_distance": 16,
        "queue_data": 32,
        "hits": 96,
        "num_rays": 104,
        "eps_self": 136,
        "active": 152,
        "num_active": 216,
        "hit_i_c": 260,
        "closest_only": 264,
    }.items():
        assert getattr(params, field).offset == offset, field


def test_transient_failures_are_told_apart_from_permanent_ones():
    assert _abi.OptixError("oom", code=7807).transient
    assert _abi.OptixError("host oom", code=7002).transient
    assert not _abi.OptixError("abi refused").transient
    assert not _abi.OptixError("bad ptx", code=7200).transient
