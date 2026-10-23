// The OptiX device ABI subset Algan's programs use, re-declared.
//
// OptiX device functions are inline PTX calls to `_optix_*` intrinsics that
// the OptiX compiler resolves when it ingests the module. NVIDIA's own
// wrappers for them live in proprietary headers, which are not vendored into
// this MIT repository; these are independent declarations of the same call
// interface (the precedent is Dr.Jit, which emits the same calls from its own
// PTX generator). Only what algan_optix_programs.cu needs is here.
#pragma once

typedef unsigned long long AoHandle;

#define AO_RAY_FLAG_NONE 0u
#define AO_RAY_FLAG_DISABLE_ANYHIT (1u << 0)
#define AO_RAY_FLAG_ENFORCE_ANYHIT (1u << 1)
#define AO_RAY_FLAG_TERMINATE_ON_FIRST_HIT (1u << 2)
#define AO_RAY_FLAG_DISABLE_CLOSESTHIT (1u << 3)

// ------------------------------------------------------------------ trace
// `_optix_trace_typed_32`: payload type, handle, origin, direction, tmin,
// tmax, time, visibility mask, ray flags, SBT offset, SBT stride, miss index,
// the number of payload words in use, then 32 payload words in and out.
static __forceinline__ __device__ void ao_trace(
    AoHandle handle, float ox, float oy, float oz, float dx, float dy, float dz,
    float tmin, float tmax, unsigned int flags, unsigned int sbt_offset,
    unsigned int sbt_stride, unsigned int miss_index, unsigned int used,
    unsigned int (&p)[32])
{
    unsigned int type = 0u;  // the default payload type
    float time = 0.0f;
    unsigned int mask = 0xFFu;
    asm volatile(
        "call"
        "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,"
        "%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31),"
        "_optix_trace_typed_32,"
        "(%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,"
        "%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,"
        "%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80);"
        : "=r"(p[0]), "=r"(p[1]), "=r"(p[2]), "=r"(p[3]), "=r"(p[4]),
          "=r"(p[5]), "=r"(p[6]), "=r"(p[7]), "=r"(p[8]), "=r"(p[9]),
          "=r"(p[10]), "=r"(p[11]), "=r"(p[12]), "=r"(p[13]), "=r"(p[14]),
          "=r"(p[15]), "=r"(p[16]), "=r"(p[17]), "=r"(p[18]), "=r"(p[19]),
          "=r"(p[20]), "=r"(p[21]), "=r"(p[22]), "=r"(p[23]), "=r"(p[24]),
          "=r"(p[25]), "=r"(p[26]), "=r"(p[27]), "=r"(p[28]), "=r"(p[29]),
          "=r"(p[30]), "=r"(p[31])
        : "r"(type), "l"(handle), "f"(ox), "f"(oy), "f"(oz), "f"(dx), "f"(dy),
          "f"(dz), "f"(tmin), "f"(tmax), "f"(time), "r"(mask), "r"(flags),
          "r"(sbt_offset), "r"(sbt_stride), "r"(miss_index), "r"(used),
          "r"(p[0]), "r"(p[1]), "r"(p[2]), "r"(p[3]), "r"(p[4]), "r"(p[5]),
          "r"(p[6]), "r"(p[7]), "r"(p[8]), "r"(p[9]), "r"(p[10]), "r"(p[11]),
          "r"(p[12]), "r"(p[13]), "r"(p[14]), "r"(p[15]), "r"(p[16]),
          "r"(p[17]), "r"(p[18]), "r"(p[19]), "r"(p[20]), "r"(p[21]),
          "r"(p[22]), "r"(p[23]), "r"(p[24]), "r"(p[25]), "r"(p[26]),
          "r"(p[27]), "r"(p[28]), "r"(p[29]), "r"(p[30]), "r"(p[31])
        :);
}

// ------------------------------------------------------------------ payload
template <unsigned int I>
static __forceinline__ __device__ unsigned int ao_get_payload()
{
    unsigned int value;
    asm volatile("call (%0), _optix_get_payload, (%1);" : "=r"(value) : "r"(I) :);
    return value;
}

template <unsigned int I>
static __forceinline__ __device__ void ao_set_payload(unsigned int value)
{
    asm volatile("call _optix_set_payload, (%0, %1);" : : "r"(I), "r"(value) :);
}

// ------------------------------------------------------------------ queries
static __forceinline__ __device__ unsigned int ao_launch_index_x()
{
    unsigned int value;
    asm("call (%0), _optix_get_launch_index_x, ();" : "=r"(value) :);
    return value;
}

// In an any-hit program: the candidate intersection's distance.
static __forceinline__ __device__ float ao_ray_tmax()
{
    float value;
    asm("call (%0), _optix_get_ray_tmax, ();" : "=f"(value) :);
    return value;
}

static __forceinline__ __device__ unsigned int ao_primitive_index()
{
    unsigned int value;
    asm("call (%0), _optix_read_primitive_idx, ();" : "=r"(value) :);
    return value;
}

// Weights of the triangle's second and third vertex.
static __forceinline__ __device__ void ao_triangle_barycentrics(float& b1, float& b2)
{
    asm("call (%0, %1), _optix_get_triangle_barycentrics, ();"
        : "=f"(b1), "=f"(b2) :);
}

static __forceinline__ __device__ void ao_ignore_intersection()
{
    asm volatile("call _optix_ignore_intersection, ();");
}

static __forceinline__ __device__ void ao_terminate_ray()
{
    asm volatile("call _optix_terminate_ray, ();");
}
