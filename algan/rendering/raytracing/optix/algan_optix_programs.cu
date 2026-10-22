// OptiX device programs for Algan's hardware ray-tracing side path.
//
// Compiled at runtime by NVRTC (PyTorch's copy) against the re-declared ABI
// in algan_optix_device.h; no NVIDIA header is involved. The host side is
// algan/rendering/raytracing/optix/runtime.py, which also passes the -D
// constants below so they always match the Taichi kernels' values.
//
// Two queries, mirroring the software walk in raytrace_kernels_taichi.py:
//
// * SHADOW (ray type 0) -- the deterministic renderer's shadow queue
//   (shadow_queue.py). An opaque caster (GAS SBT class 0, any-hit disabled)
//   ends the ray in hardware: the ordered march's answer is then exactly full
//   occlusion. A translucent caster (class 1) is recorded -- the first
//   AO_SHADOW_HITS of them in the march's own order -- and ignored, and a
//   Taichi kernel multiplies the list with the march's own material code
//   (optix_shadow_taichi.py). Identity-aware acceptance
//   (_shadow_identity_t_min) runs in the any-hit programs.
//
// * GATHER (ray types 1 and 2) -- _collect_hits for triangle-only batches:
//   the first AO_KBUF hits strictly after the ray's cursor in _comes_after
//   order, cut after the first opaque one. Type 1 lets the hardware commit
//   the nearest opaque hit and records translucent ones; type 2 re-traces the
//   committed hit's depth bin, where a hit farther along but on a higher layer
//   still comes first (the bins of _comes_after).

#include "algan_optix_device.h"

#ifndef AO_KBUF
#define AO_KBUF 4
#endif
#ifndef AO_SHADOW_HITS
#define AO_SHADOW_HITS 4
#endif
#ifndef AO_TRIANGLE_EDGE_EPSILON
#define AO_TRIANGLE_EDGE_EPSILON 2e-4f
#endif
// Every list lives in payload words, four per entry; ao_list_insert spells
// out four entries.
static_assert(AO_KBUF >= 1 && AO_KBUF <= 4, "the gather lists hold at most 4 hits");
static_assert(AO_SHADOW_HITS >= 1 && AO_SHADOW_HITS <= 4, "at most 4 shadow hits");

// Mirrored by runtime._Params; keep the two in the same order.
struct Params {
    // ---- shared
    const AoHandle* handles;  // one GAS per batch frame (or one for all)
    unsigned int num_handles;
    unsigned int pad0;
    float min_hit_distance;
    float inv_bin;            // 1 / depth_tie_epsilon
    float bin_width;          // depth_tie_epsilon
    float layer_offset;       // layer_offset_triangles
    // ---- shadow queue
    const float* queue_data;  // [R, 7]: origin, direction, tmax
    const int* queue_valid;   // [R]
    const int* layout;        // [slots, 2]: fan sample, first slot
    float* occ_out;           // [R, 3]
    const int* event_frame;   // [events]
    const int* event_src_prim;  // [events] source triangle, or null
    const int* tri_obj;       // [To, N] mesh id per triangle, or null
    int* status;              // [R]: 0 untraced, 1 answered, 2 + n: n hits listed
    float* hits;              // [R, 4 * AO_SHADOW_HITS]: t, prim bits, b1, b2
    unsigned int num_rays;
    unsigned int num_events;
    unsigned int event_first;
    unsigned int tri_obj_frames;
    unsigned int tri_obj_count;
    unsigned int phase;
    unsigned int identity;
    unsigned int pad1;
    float eps_self;
    float eps_near;
    float pad2;
    float pad3;
    // ---- gather
    const int* active;        // [num_active] ray slots
    const float* rs_ro;
    const float* rs_rd;
    const float* rs_sca;
    int* rs_int;
    const int* rs_pix;
    float* hit_f;             // [kbuf, 4, A]
    int* hit_i;               // [kbuf, 2, A]
    unsigned int num_active;
    int time_start;
    int ray_offset;
    int pixels_per_frame;
    unsigned int ro_stride;
    unsigned int rd_stride;
    unsigned int sca_stride;
    unsigned int int_stride;
    unsigned int hit_f_q;     // element strides of hit_f's first two axes
    unsigned int hit_f_c;
    unsigned int hit_i_q;
    unsigned int hit_i_c;
    unsigned int closest_only;  // keep one hit (the opaque_closest variant)
    unsigned int pad4;
};

// runtime._Params mirrors this struct field for field; test_optix_abi.py
// holds its offsets (NVRTC has no offsetof without the standard headers).
static_assert(sizeof(Params) == 272, "Params layout");

extern "C" {
__constant__ Params params;
}

// ------------------------------------------------------------------ order
// _comes_after: depth bins of width depth_tie_epsilon, ties on descending
// layer. The single f32 multiply rounds exactly as Taichi's does.
static __forceinline__ __device__ float ao_bin(float t)
{
    return floorf(t * params.inv_bin);
}

static __forceinline__ __device__ bool ao_comes_after(float t, float layer,
                                                      float t_prev, float layer_prev)
{
    const float bt = ao_bin(t);
    const float bp = ao_bin(t_prev);
    return (bt > bp) || ((bt == bp) && (layer < layer_prev));
}

static __forceinline__ __device__ float ao_layer(unsigned int prim)
{
    return params.layer_offset + (float)prim;
}

// A sorted list of hits in payload words [base, base + 4 * cap): t, prim,
// b1, b2 per entry; ``count`` is how many entries are valid (the caller
// tracks totals beyond ``cap`` itself). Inserts in _comes_after order and
// drops whatever falls past the end.
#define AO_LIST_GET(I) ao_get_payload<(I)>()
#define AO_LIST_SET(I, V) ao_set_payload<(I)>(V)

template <unsigned int BASE, unsigned int CAP>
static __forceinline__ __device__ void ao_list_insert(unsigned int count, float t,
                                                      unsigned int prim, float b1,
                                                      float b2)
{
    float lt[CAP];
    unsigned int lp[CAP];
    float la[CAP];
    float lb[CAP];
#pragma unroll
    for (unsigned int k = 0; k < CAP; ++k) {
        lt[k] = 0.0f;
        lp[k] = 0u;
        la[k] = 0.0f;
        lb[k] = 0.0f;
    }
    // Payload words need compile-time indices, so read them unrolled.
    if (CAP > 0) { lt[0] = __uint_as_float(AO_LIST_GET(BASE + 0)); lp[0] = AO_LIST_GET(BASE + 1); la[0] = __uint_as_float(AO_LIST_GET(BASE + 2)); lb[0] = __uint_as_float(AO_LIST_GET(BASE + 3)); }
    if (CAP > 1) { lt[1] = __uint_as_float(AO_LIST_GET(BASE + 4)); lp[1] = AO_LIST_GET(BASE + 5); la[1] = __uint_as_float(AO_LIST_GET(BASE + 6)); lb[1] = __uint_as_float(AO_LIST_GET(BASE + 7)); }
    if (CAP > 2) { lt[2] = __uint_as_float(AO_LIST_GET(BASE + 8)); lp[2] = AO_LIST_GET(BASE + 9); la[2] = __uint_as_float(AO_LIST_GET(BASE + 10)); lb[2] = __uint_as_float(AO_LIST_GET(BASE + 11)); }
    if (CAP > 3) { lt[3] = __uint_as_float(AO_LIST_GET(BASE + 12)); lp[3] = AO_LIST_GET(BASE + 13); la[3] = __uint_as_float(AO_LIST_GET(BASE + 14)); lb[3] = __uint_as_float(AO_LIST_GET(BASE + 15)); }
    const unsigned int n = count < CAP ? count : CAP;
    // Position: the number of kept entries the new hit comes after.
    const float layer = ao_layer(prim & 0x1FFFFFFFu);
    unsigned int pos = 0;
#pragma unroll
    for (unsigned int k = 0; k < CAP; ++k) {
        if (k < n && ao_comes_after(t, layer, lt[k], ao_layer(lp[k] & 0x1FFFFFFFu)))
            pos = k + 1;
    }
    if (pos >= CAP)
        return;
#pragma unroll
    for (int k = (int)CAP - 1; k > 0; --k) {
        if ((unsigned int)k > pos) {
            lt[k] = lt[k - 1];
            lp[k] = lp[k - 1];
            la[k] = la[k - 1];
            lb[k] = lb[k - 1];
        }
    }
#pragma unroll
    for (unsigned int k = 0; k < CAP; ++k) {
        if (k == pos) {
            lt[k] = t;
            lp[k] = prim;
            la[k] = b1;
            lb[k] = b2;
        }
    }
    if (CAP > 0) { AO_LIST_SET(BASE + 0, __float_as_uint(lt[0])); AO_LIST_SET(BASE + 1, lp[0]); AO_LIST_SET(BASE + 2, __float_as_uint(la[0])); AO_LIST_SET(BASE + 3, __float_as_uint(lb[0])); }
    if (CAP > 1) { AO_LIST_SET(BASE + 4, __float_as_uint(lt[1])); AO_LIST_SET(BASE + 5, lp[1]); AO_LIST_SET(BASE + 6, __float_as_uint(la[1])); AO_LIST_SET(BASE + 7, __float_as_uint(lb[1])); }
    if (CAP > 2) { AO_LIST_SET(BASE + 8, __float_as_uint(lt[2])); AO_LIST_SET(BASE + 9, lp[2]); AO_LIST_SET(BASE + 10, __float_as_uint(la[2])); AO_LIST_SET(BASE + 11, __float_as_uint(lb[2])); }
    if (CAP > 3) { AO_LIST_SET(BASE + 12, __float_as_uint(lt[3])); AO_LIST_SET(BASE + 13, lp[3]); AO_LIST_SET(BASE + 14, __float_as_uint(la[3])); AO_LIST_SET(BASE + 15, __float_as_uint(lb[3])); }
}

// ================================================================== shadow
// Payload: p0 = 1 until the miss program clears it (an accepted opaque hit
// ends the ray without a miss), p1 = translucent hits seen, p2.. the list.

// The acceptance floor of _shadow_identity_t_min, for ray ``r``'s event.
static __forceinline__ __device__ bool ao_shadow_rejects(unsigned int prim, float t)
{
    if (params.identity == 0u)
        return false;
    const unsigned int r = ao_launch_index_x();
    const unsigned int e = params.event_first + r % params.num_events;
    const int src = params.event_src_prim[e];
    if (src < 0)
        return false;  // traced from min_hit_distance instead
    float floor_t = 0.0f;
    if ((int)prim == src) {
        floor_t = params.eps_self;
    } else if (params.eps_near > 0.0f && params.tri_obj != nullptr) {
        const unsigned int f = (unsigned int)params.event_frame[e];
        const unsigned long long row =
            (unsigned long long)(f % params.tri_obj_frames) * params.tri_obj_count;
        if (params.tri_obj[row + prim] == params.tri_obj[row + (unsigned int)src])
            floor_t = params.eps_near;
    }
    return !(t > floor_t);
}

extern "C" __global__ void __raygen__shadow()
{
    const unsigned int r = ao_launch_index_x();
    if (r >= params.num_rays)
        return;
    // Which rays this phase traces: shadow_queue_trace's rule exactly
    // (phase 0: soft fans and the hard fan's diagonal pair; phase 1: the
    // off-diagonal taps of a pair that disagreed) -- or, phase 2, every ray:
    // the adaptive rule is the reduce's to apply, which reads an off-diagonal
    // tap only where the diagonal pair disagreed, so tracing them all changes
    // no answer and saves the second launch pair.
    const int flag = params.queue_valid[r];
    const unsigned int slot = r / params.num_events;
    const int sk = params.layout[2 * slot];
    const unsigned int first =
        (unsigned int)params.layout[2 * slot + 1] * params.num_events + r % params.num_events;
    bool trace = flag != 0;
    if (params.phase == 2u) {
        // every valid ray
    } else if (params.phase == 0u) {
        if (flag == 2 && sk >= 2)
            trace = false;
    } else {
        trace = flag == 2 && sk >= 2;
        if (trace) {
            if (params.queue_valid[first] == 2 &&
                params.queue_valid[first + params.num_events] == 2) {
                const float* a = params.occ_out + 3ull * first;
                const float* b = params.occ_out + 3ull * (first + params.num_events);
                if (a[0] == b[0] && a[1] == b[1] && a[2] == b[2])
                    trace = false;
            }
        }
    }
    if (!trace) {
        params.status[r] = 0;
        return;
    }
    const unsigned int e = params.event_first + r % params.num_events;
    const unsigned int f = (unsigned int)params.event_frame[e];
    const AoHandle handle = params.handles[f % params.num_handles];
    const float* q = params.queue_data + 7ull * r;
    const float ox = q[0], oy = q[1], oz = q[2];
    const float dx = q[3], dy = q[4], dz = q[5];
    const float tmax = q[6];
    int src = -1;
    if (params.identity != 0u)
        src = params.event_src_prim[e];
    // With a source triangle every other mesh is accepted down to t > 0 and
    // the floors are tested per hit; without one, the plain epsilon.
    const float tmin = src >= 0 ? 0.0f : params.min_hit_distance;
    float* occ = params.occ_out + 3ull * r;
    const bool finite = isfinite(ox) && isfinite(oy) && isfinite(oz) && isfinite(dx) &&
                        isfinite(dy) && isfinite(dz) && !isnan(tmax);
    if (!finite || !(tmax > tmin)) {
        // The march finds nothing on such a ray (its slab tests fail).
        occ[0] = 0.0f;
        occ[1] = 0.0f;
        occ[2] = 0.0f;
        params.status[r] = 1;
        return;
    }
    unsigned int flags = AO_RAY_FLAG_TERMINATE_ON_FIRST_HIT | AO_RAY_FLAG_DISABLE_CLOSESTHIT;
    if (src >= 0)
        flags |= AO_RAY_FLAG_ENFORCE_ANYHIT;  // the opaque floor test
    unsigned int p[32];
#pragma unroll
    for (int k = 0; k < 32; ++k)
        p[k] = 0u;
    p[0] = 1u;
    ao_trace(handle, ox, oy, oz, dx, dy, dz, tmin, tmax, flags, 0u, 3u, 0u,
             2u + 4u * AO_SHADOW_HITS, p);
    if (p[0] != 0u) {
        occ[0] = 1.0f;
        occ[1] = 1.0f;
        occ[2] = 1.0f;
        params.status[r] = 1;
        return;
    }
    if (p[1] == 0u) {
        occ[0] = 0.0f;
        occ[1] = 0.0f;
        occ[2] = 0.0f;
        params.status[r] = 1;
        return;
    }
    params.status[r] = 2 + (int)p[1];
    // Stored as words: the prim slots are integers, and nothing here should
    // pass them through a float register.
    unsigned int* out = reinterpret_cast<unsigned int*>(params.hits) +
                        (unsigned long long)(4 * AO_SHADOW_HITS) * r;
#pragma unroll
    for (unsigned int k = 0; k < 4 * AO_SHADOW_HITS; ++k)
        out[k] = p[2 + k];
}

extern "C" __global__ void __miss__shadow()
{
    ao_set_payload<0>(0u);
}

// Opaque casters: only reached on rays that enforce any-hit (a source
// triangle), to apply the identity floor; returning accepts, which ends the
// ray (TERMINATE_ON_FIRST_HIT).
extern "C" __global__ void __anyhit__shadow_opaque()
{
    if (ao_shadow_rejects(ao_primitive_index(), ao_ray_tmax())) {
        ao_ignore_intersection();
        return;
    }
}

extern "C" __global__ void __anyhit__shadow_translucent()
{
    const float t = ao_ray_tmax();
    const unsigned int prim = ao_primitive_index();
    if (ao_shadow_rejects(prim, t)) {
        ao_ignore_intersection();
        return;
    }
    float b1, b2;
    ao_triangle_barycentrics(b1, b2);
    const unsigned int count = ao_get_payload<1>();
    ao_list_insert<2, AO_SHADOW_HITS>(count, t, prim, b1, b2);
    ao_set_payload<1>(count + 1u);
    ao_ignore_intersection();
}

// ================================================================== gather
// Type 1 payload: p0 = translucent entries kept, p1..p16 their list, p17 = 1
// when an opaque hit was committed, p18..p21 that hit (t, prim, b1, b2),
// p22/p23 = the cursor (t_prev, layer_prev).
// Type 2 payload: p0 = entries seen, p1..p16 the list (prim | class << 31),
// p17 = the committed hit's bin, p22/p23 = the cursor.

static __forceinline__ __device__ bool ao_gather_accepts(float t, unsigned int prim)
{
    const float t_prev = __uint_as_float(ao_get_payload<22>());
    const float layer_prev = __uint_as_float(ao_get_payload<23>());
    return (t > params.min_hit_distance) &&
           ao_comes_after(t, ao_layer(prim), t_prev, layer_prev);
}

extern "C" __global__ void __anyhit__gather_opaque()
{
    // Reached only when the ray enforces any-hit (a cursor past t = 0).
    if (!ao_gather_accepts(ao_ray_tmax(), ao_primitive_index())) {
        ao_ignore_intersection();
        return;
    }
}

extern "C" __global__ void __anyhit__gather_translucent()
{
    const float t = ao_ray_tmax();
    const unsigned int prim = ao_primitive_index();
    if (ao_gather_accepts(t, prim)) {
        float b1, b2;
        ao_triangle_barycentrics(b1, b2);
        const unsigned int count = ao_get_payload<0>();
        ao_list_insert<1, AO_KBUF>(count, t, prim, b1, b2);
        ao_set_payload<0>(count + 1u);
    }
    ao_ignore_intersection();
}

extern "C" __global__ void __closesthit__gather()
{
    float b1, b2;
    ao_triangle_barycentrics(b1, b2);
    ao_set_payload<17>(1u);
    ao_set_payload<18>(__float_as_uint(ao_ray_tmax()));
    ao_set_payload<19>(ao_primitive_index());
    ao_set_payload<20>(__float_as_uint(b1));
    ao_set_payload<21>(__float_as_uint(b2));
}

template <unsigned int CLASS>
static __forceinline__ __device__ void ao_gather_bin_hit()
{
    const float t = ao_ray_tmax();
    const unsigned int prim = ao_primitive_index();
    const float bin_o = __uint_as_float(ao_get_payload<17>());
    if (ao_bin(t) == bin_o && ao_gather_accepts(t, prim)) {
        float b1, b2;
        ao_triangle_barycentrics(b1, b2);
        const unsigned int count = ao_get_payload<0>();
        ao_list_insert<1, AO_KBUF>(count, t, prim | (CLASS << 31), b1, b2);
        ao_set_payload<0>(count + 1u);
    }
    ao_ignore_intersection();
}

extern "C" __global__ void __anyhit__gather_bin_opaque() { ao_gather_bin_hit<1u>(); }
extern "C" __global__ void __anyhit__gather_bin_translucent() { ao_gather_bin_hit<0u>(); }

extern "C" __global__ void __miss__gather() {}

extern "C" __global__ void __raygen__gather()
{
    const unsigned int i = ao_launch_index_x();
    if (i >= params.num_active)
        return;
    const unsigned int r = (unsigned int)params.active[i];
    const float* ro = params.rs_ro + (unsigned long long)params.ro_stride * r;
    const float* rd = params.rs_rd + (unsigned long long)params.rd_stride * r;
    const float* sca = params.rs_sca + (unsigned long long)params.sca_stride * r;
    const float t_prev = sca[1];
    const float layer_prev = sca[2];
    const int g = params.ray_offset + params.rs_pix[r];
    const int f = params.time_start + g / params.pixels_per_frame;
    const AoHandle handle = params.handles[(unsigned int)f % params.num_handles];
    const float ox = ro[0], oy = ro[1], oz = ro[2];
    const float dx = rd[0], dy = rd[1], dz = rd[2];

    // Up to AO_KBUF + 1 + AO_KBUF candidates: the translucent list, the
    // committed opaque hit, the committed hit's bin.
    const int CAND = 2 * AO_KBUF + 1;
    float ct[CAND];
    unsigned int cp[CAND];
    float ca[CAND];
    float cb[CAND];
    unsigned int copq[CAND];
    int nc = 0;

    const bool finite = isfinite(ox) && isfinite(oy) && isfinite(oz) && isfinite(dx) &&
                        isfinite(dy) && isfinite(dz);
    if (finite) {
        float tmin = params.min_hit_distance;
        unsigned int flags = AO_RAY_FLAG_NONE;
        if (t_prev > 0.0f || layer_prev < 1e30f) {
            // Hits in the cursor's bin at a smaller t can still come after it
            // (a lower layer): start a bin early and filter every hit,
            // opaque ones included, by the exact order.
            tmin = fmaxf(tmin, (ao_bin(t_prev) - 1.0f) * params.bin_width);
            flags = AO_RAY_FLAG_ENFORCE_ANYHIT;
        }
        unsigned int p[32];
#pragma unroll
        for (int k = 0; k < 32; ++k)
            p[k] = 0u;
        p[22] = __float_as_uint(t_prev);
        p[23] = __float_as_uint(layer_prev);
        ao_trace(handle, ox, oy, oz, dx, dy, dz, tmin, 1e30f, flags, 1u, 3u, 1u, 24u, p);
        const unsigned int kept = p[0] < AO_KBUF ? p[0] : AO_KBUF;
#pragma unroll
        for (unsigned int k = 0; k < AO_KBUF; ++k) {
            if (k < kept) {
                ct[nc] = __uint_as_float(p[1 + 4 * k]);
                cp[nc] = p[2 + 4 * k];
                ca[nc] = __uint_as_float(p[3 + 4 * k]);
                cb[nc] = __uint_as_float(p[4 + 4 * k]);
                copq[nc] = 0u;
                ++nc;
            }
        }
        if (p[17] != 0u) {
            const float t_o = __uint_as_float(p[18]);
            ct[nc] = t_o;
            cp[nc] = p[19];
            ca[nc] = __uint_as_float(p[20]);
            cb[nc] = __uint_as_float(p[21]);
            copq[nc] = 1u;
            ++nc;
            // The committed hit's bin: a hit farther along it on a higher
            // layer comes first, and the hardware cut it off at t_o.
            const float bin_o = ao_bin(t_o);
            const float t_end = (bin_o + 2.0f) * params.bin_width;
            unsigned int s[32];
#pragma unroll
            for (int k = 0; k < 32; ++k)
                s[k] = 0u;
            s[17] = __float_as_uint(bin_o);
            s[22] = __float_as_uint(t_prev);
            s[23] = __float_as_uint(layer_prev);
            // From the bin's start, not t_o: a coplanar twin at exactly t_o
            // on a higher layer comes first, and the hardware's interval
            // need not include its own lower end. Hits seen twice are the
            // same primitive and are merged below.
            const float t_begin = fmaxf(tmin, (bin_o - 1.0f) * params.bin_width);
            ao_trace(handle, ox, oy, oz, dx, dy, dz, t_begin, t_end,
                     AO_RAY_FLAG_ENFORCE_ANYHIT | AO_RAY_FLAG_DISABLE_CLOSESTHIT, 2u, 3u,
                     2u, 24u, s);
            const unsigned int seen = s[0] < AO_KBUF ? s[0] : AO_KBUF;
#pragma unroll
            for (unsigned int k = 0; k < AO_KBUF; ++k) {
                if (k < seen) {
                    const unsigned int word = s[2 + 4 * k];
                    const unsigned int prim = word & 0x7FFFFFFFu;
                    bool dup = false;
                    for (int c = 0; c < nc; ++c)
                        dup = dup || (cp[c] == prim);
                    if (!dup) {
                        ct[nc] = __uint_as_float(s[1 + 4 * k]);
                        cp[nc] = prim;
                        ca[nc] = __uint_as_float(s[3 + 4 * k]);
                        cb[nc] = __uint_as_float(s[4 + 4 * k]);
                        copq[nc] = word >> 31;
                        ++nc;
                    }
                }
            }
        }
    }
    // Sort the candidates into _comes_after order (at most 2 * kbuf + 1).
    for (int a = 1; a < nc; ++a) {
        for (int b = a; b > 0; --b) {
            if (ao_comes_after(ct[b - 1], ao_layer(cp[b - 1]), ct[b], ao_layer(cp[b]))) {
                float tt = ct[b]; ct[b] = ct[b - 1]; ct[b - 1] = tt;
                unsigned int pp = cp[b]; cp[b] = cp[b - 1]; cp[b - 1] = pp;
                float aa = ca[b]; ca[b] = ca[b - 1]; ca[b - 1] = aa;
                float bb = cb[b]; cb[b] = cb[b - 1]; cb[b - 1] = bb;
                unsigned int oo = copq[b]; copq[b] = copq[b - 1]; copq[b - 1] = oo;
            }
        }
    }
    // Everything after the first opaque hit is absorbed; keep kbuf (under
    // opaque_closest, the first hit alone: _nearest_surface_g's answer).
    const int keep = params.closest_only != 0u ? 1 : AO_KBUF;
    int count = 0;
    for (int c = 0; c < nc && count < keep; ++c) {
        ++count;
        if (copq[c] != 0u)
            break;
    }
    params.rs_int[(unsigned long long)params.int_stride * r + 3] = count;
    if (count > 0) {
#pragma unroll
        for (int q = 0; q < AO_KBUF; ++q) {
            float t = 0.0f, layer = 0.0f, a = 0.0f, b = 0.0f;
            int prim = 0, fl = 0;
            if (q < count) {
                t = ct[q];
                layer = ao_layer(cp[q]);
                a = ca[q];
                b = cb[q];
                prim = (int)cp[q];
                const float w0 = 1.0f - a - b;
                const int edge = fminf(w0, fminf(a, b)) < AO_TRIANGLE_EDGE_EPSILON ? 1 : 0;
                fl = 1 | (edge << 2);
            }
            float* hf = params.hit_f + (unsigned long long)params.hit_f_q * q + i;
            hf[0] = t;
            hf[params.hit_f_c] = layer;
            hf[2ull * params.hit_f_c] = a;
            hf[3ull * params.hit_f_c] = b;
            int* hi = params.hit_i + (unsigned long long)params.hit_i_q * q + i;
            hi[0] = prim;
            hi[params.hit_i_c] = fl;
        }
    }
}
