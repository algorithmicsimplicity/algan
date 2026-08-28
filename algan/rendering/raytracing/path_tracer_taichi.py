"""Path-tracer stage kernels + the deterministic Sobol-Owen sampler.

This module is the kernel half of the ``samples_per_pixel > 1`` renderer (the
host orchestration lives in ``path_tracer.py``). The path tracer shares the
deterministic renderer's wavefront shape -- generate, traverse, shade, compact,
with host compaction between iterations -- and shares its *traversal kernel*
outright: paths keep the ``rs_ro/rs_rd/rs_sca/rs_int/rs_pix`` state layout, so
``wavefront_kernels_taichi.wavefront_traverse_events`` gathers their surface
events with no PT-specific traversal variant. What is PT-specific lives here:

``pt_generate``
    One path per (pixel, wave sample): the primary ray jittered inside its
    pixel by the sampler (implicit anti-aliasing at output resolution).
``pt_shade``
    Drains the transient hit-event batch front-to-back in ``(t, layer)``
    order -- the same peel, seam and coplanar-layer rules as the
    deterministic renderer -- compositing transparency *deterministically*
    (throughput-weighted, never stochastic alpha), so stacked vector
    graphics and text match the deterministic composite with zero variance.
``pt_reduce``
    Folds a wave's per-path accumulators into the chunk's per-pixel sample
    sums (``accum``), applying leftover throughput to the background. One
    thread per pixel sums its own wave samples in a fixed order, so
    accumulation uses no atomics and a render is reproducible run-to-run.

``finalize_samples`` (in ``raytrace_kernels_taichi``) then averages ``accum``
into the frame buffer exactly as it always has.

Sampler
-------
Hash-based Owen-scrambled Sobol after Burley, "Practical Hash-based Owen
Scrambling" (JCGT 2020). Only the 2D Sobol (0,2) pair is evaluated directly;
higher dimensions are *padded*: each logical 2D pair reuses the base pair
under an Owen shuffle of the sample index plus per-dimension Owen scrambles,
all seeded by hashes of ``(pt_seed, frame, pixel, pair)``. Every sample is a
pure function of those inputs -- independent of tile, wave, batch and chunk
splits, and of thread scheduling.

Dimension-pair allocation (a fixed table; keep in sync with ``pt_shade``):

===========================  ==================================================
pair                         use
===========================  ==================================================
0                            sub-pixel jitter (2D)
1                            lens (2D) -- reserved for depth of field
2 + 6b + 0                   bounce ``b``: x lobe select, y Russian roulette
2 + 6b + 1                   bounce ``b``: BSDF direction (2D)
2 + 6b + 2                   bounce ``b``: x light select, y spare
2 + 6b + 3                   bounce ``b``: light point (2D)
2 + 6b + 4, 5                bounce ``b``: reserved for volumes
===========================  ==================================================

Transparency continuations are deterministic and consume no dimensions.
"""

import taichi as ti

from algan.rendering.raytracing.raytrace_kernels_taichi import (
    _comes_after,
    _flat_triangle_color,
    _generate_ray,
    _sample_circuit_color,
    depth_tie_epsilon,
    kbuf,
    max_surfaces_per_ray,
    min_weight,
)
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _ACTIVE,
    _DONE,
)

# Sampler dimension pairs (see the module docstring's table).
PAIR_PIXEL = 0
PAIR_LENS = 1
PAIR_BOUNCE_BASE = 2
PAIRS_PER_BOUNCE = 6

# Per-path commit row (``pt_acc``): radiance accumulated so far, the leftover
# throughput the background shows through, and the camera-segment
# alpha transparency (see ``pt_reduce``).
PT_ACC_WIDTH = 9
_PT_ACC_LEFTOVER = 4
_PT_ACC_ALPHA = 8

# Device-side truncation tallies, read back by the host once per wave and fed
# through ``truncation.record_truncation`` (ceilings are counted, not silent).
PT_STATS_WIDTH = 4
PT_STAT_TRUNC_SURFACES = 0


def _sobol_dim1_directions():
    """Direction numbers of the second Sobol dimension (primitive polynomial
    ``x + 1``): ``v[0] = 2^31``, ``v[j] = v[j-1] ^ (v[j-1] >> 1)``.
    """
    dirs = []
    v = 0x80000000
    for _ in range(32):
        dirs.append(v)
        v = v ^ (v >> 1)
    return tuple(dirs)


_SOBOL_DIM1 = _sobol_dim1_directions()


@ti.func
def _pt_hash(x: ti.u32) -> ti.u32:
    # Wellons' lowbias32: a fast, well-mixed 32-bit finalizer.
    x ^= x >> 16
    x = x * ti.u32(0x7FEB352D)
    x ^= x >> 15
    x = x * ti.u32(0x846CA68B)
    x ^= x >> 16
    return x


@ti.func
def _pt_hash_combine(seed: ti.u32, value: ti.u32) -> ti.u32:
    return _pt_hash(seed ^ (value + ti.u32(0x9E3779B9) + (seed << 6) + (seed >> 2)))


@ti.func
def _pt_reverse_bits(x: ti.u32) -> ti.u32:
    x = ((x >> 1) & ti.u32(0x55555555)) | ((x & ti.u32(0x55555555)) << 1)
    x = ((x >> 2) & ti.u32(0x33333333)) | ((x & ti.u32(0x33333333)) << 2)
    x = ((x >> 4) & ti.u32(0x0F0F0F0F)) | ((x & ti.u32(0x0F0F0F0F)) << 4)
    x = ((x >> 8) & ti.u32(0x00FF00FF)) | ((x & ti.u32(0x00FF00FF)) << 8)
    return (x >> 16) | (x << 16)


@ti.func
def _pt_laine_karras(x: ti.u32, seed: ti.u32) -> ti.u32:
    """Laine-Karras-style hash whose per-bit avalanche property makes it a
    valid base-2 Owen scramble of the *reversed* digit string (Burley 2020,
    listing 3).
    """
    x = x + seed
    x ^= x * ti.u32(0x6C50B47C)
    x ^= x * ti.u32(0xB82F1E52)
    x ^= x * ti.u32(0xC7AFE638)
    x ^= x * ti.u32(0x8D22F6E6)
    return x


@ti.func
def _pt_owen_scramble(x: ti.u32, seed: ti.u32) -> ti.u32:
    """Base-2 Owen scramble of a value whose first digit is bit 31."""
    return _pt_reverse_bits(_pt_laine_karras(_pt_reverse_bits(x), seed))


@ti.func
def _pt_sobol_dim1(index: ti.u32) -> ti.u32:
    result = ti.u32(0)
    for j in ti.static(range(32)):
        if (index >> j) & ti.u32(1) != 0:
            result ^= ti.u32(_SOBOL_DIM1[j])
    return result


@ti.func
def _pt_key(f: ti.i32, pixel: ti.i32) -> ti.u32:
    """Per-(frame, pixel) sampler key. ``f`` is the absolute frame and
    ``pixel`` the frame-local pixel index, so the key -- and with it every
    sample -- is independent of how the render was split into chunks.
    """
    return _pt_hash_combine(ti.cast(f, ti.u32), ti.cast(pixel, ti.u32))


@ti.func
def pt_sample_2d(seed_root: ti.u32, key: ti.u32, pair: ti.i32,
                 sample_index: ti.i32) -> ti.math.vec2:
    """Sample ``pair`` of the pixel's padded Sobol sequence at
    ``sample_index``: Owen-shuffled index, Owen-scrambled (0,2) point.

    Any prefix of the returned sequence is well stratified, which is what
    makes progressive rendering (waves) and future adaptive sampling sound;
    distinct ``(key, pair)`` values decorrelate into independent sequences.
    """
    pair_seed = _pt_hash_combine(seed_root, _pt_hash_combine(key, ti.cast(pair, ti.u32)))
    shuffle_seed = _pt_hash(pair_seed ^ ti.u32(0x51633E2D))
    seed_x = _pt_hash(pair_seed ^ ti.u32(0x68BC21EB))
    seed_y = _pt_hash(pair_seed ^ ti.u32(0x02E5BE93))
    index = _pt_owen_scramble(ti.cast(sample_index, ti.u32), shuffle_seed)
    vx = _pt_reverse_bits(_pt_laine_karras(index, seed_x))
    vy = _pt_owen_scramble(_pt_sobol_dim1(index), seed_y)
    # Take the top 24 bits: exactly representable in f32, uniform in [0, 1).
    return ti.math.vec2(
        ti.cast(vx >> 8, ti.f32) * (1.0 / 16777216.0),
        ti.cast(vy >> 8, ti.f32) * (1.0 / 16777216.0),
    )


@ti.kernel
def pt_sampler_probe(seed_root: ti.u32, f: ti.i32, pixel: ti.i32, pair: ti.i32,
                     out: ti.types.ndarray()):
    """Test probe: fill ``out [n, 2]`` with samples 0..n-1 of one pixel/pair.

    Exists so the sampler's stratification, reproducibility and decorrelation
    can be unit-tested without driving the render pipeline.
    """
    key = _pt_key(f, pixel)
    for s in range(out.shape[0]):
        u = pt_sample_2d(seed_root, key, pair, s)
        out[s, 0] = u[0]
        out[s, 1] = u[1]


@ti.kernel
def pt_generate(num_slots: ti.i32, tile_pixels: ti.i32, sample_base: ti.i32,
                seed_root: ti.u32, time_start: ti.i32, width: ti.i32,
                height: ti.i32, tile_start: ti.i32,
                half_screen_w: ti.f32, half_screen_h: ti.f32,
                cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
                pixel_basis_x: ti.types.ndarray(),
                pixel_basis_y: ti.types.ndarray(),
                rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
                rs_pix: ti.types.ndarray()):
    """Write each slot's jittered primary ray.

    Slot layout: ``slot = k * tile_pixels + p_local`` holds wave sample
    ``sample_base + k`` of tile pixel ``p_local``, so one wave puts every tile
    pixel's next ``S`` samples in flight and ``pt_reduce`` can walk a pixel's
    slots at stride ``tile_pixels``. The rest of the per-slot state is
    constant at generation and broadcast-filled by the host (the same
    coalesced-fill reasoning as the deterministic ``const_fill`` path).
    """
    pixels_per_frame = width * height
    for slot in range(num_slots):
        p_local = slot % tile_pixels
        s = sample_base + slot // tile_pixels
        g = tile_start + p_local
        f_rel = g // pixels_per_frame
        p = g - f_rel * pixels_per_frame
        f = time_start + f_rel
        py = p // width
        px = p - py * width
        jitter = pt_sample_2d(seed_root, _pt_key(f, p), PAIR_PIXEL, s)
        ro, rd = _generate_ray(f, px, py, jitter[0], jitter[1],
                               half_screen_w, half_screen_h,
                               cam_origin, screen_point,
                               pixel_basis_x, pixel_basis_y)
        for k in ti.static(range(3)):
            rs_ro[slot, k] = ro[k]
            rs_rd[slot, k] = rd[k]
        rs_pix[slot] = p_local


@ti.kernel
def pt_shade(active: ti.types.ndarray(), num_active: ti.i32,
             tri_colors: ti.types.ndarray(), tri_uvs: ti.types.ndarray(),
             tri_tex_meta: ti.types.ndarray(), textures: ti.types.ndarray(),
             num_colored_triangles: ti.i32,
             circuit_meta: ti.types.ndarray(),
             circuit_colors: ti.types.ndarray(),
             circuit_border_colors: ti.types.ndarray(),
             time_start: ti.i32, width: ti.i32, height: ti.i32,
             ray_offset: ti.i32,
             rs_sca: ti.types.ndarray(), rs_int: ti.types.ndarray(),
             rs_pix: ti.types.ndarray(),
             hit_f: ti.types.ndarray(), hit_i: ti.types.ndarray(),
             pt_thru: ti.types.ndarray(), pt_acc: ti.types.ndarray(),
             pt_stats: ti.types.ndarray()):
    """Consume one traverse's hit-event batch: deterministic alpha compositing.

    Every crossed surface contributes ``throughput * alpha * color`` and
    attenuates the path by ``1 - alpha`` -- the same front-to-back composite
    as the deterministic renderer, in the same ``(t, layer)`` order with the
    same seam rule, so a stack of 2-D vector graphics resolves exactly (zero
    variance; the sampler's only role on such content is the sub-pixel
    jitter). A path retires when its throughput falls under ``min_weight``,
    when its peel completes (the leftover shows the background, applied in
    ``pt_reduce``), or at the counted ``max_surfaces_per_ray`` ceiling.

    PT state beyond the shared wavefront layout: ``rs_sca[r, 0]`` holds the
    camera-segment alpha transparency (the probability-mass the background
    alpha shows through; see ``pt_reduce``), and ``pt_thru`` the RGB + glow
    throughput. Scattering (BSDF bounces, lights) lands here in later stages;
    this kernel is the transport skeleton.
    """
    pixels_per_frame = width * height
    for i in range(num_active):
        r = active[i]
        num_hits = rs_int[r, 3]
        if num_hits > 0:
            g = ray_offset + rs_pix[r]
            f = time_start + g // pixels_per_frame
            thru = ti.math.vec4(pt_thru[r, 0], pt_thru[r, 1], pt_thru[r, 2],
                                pt_thru[r, 3])
            t_alpha = rs_sca[r, 0]
            t_prev = rs_sca[r, 1]
            layer_prev = rs_sca[r, 2]
            seam_t = rs_sca[r, 3]
            processed = rs_int[r, 1]
            acc = ti.math.vec4(0.0, 0.0, 0.0, 0.0)

            kb_t = ti.Vector([0.0] * kbuf)
            kb_layer = ti.Vector([0.0] * kbuf)
            kb_prim = ti.Vector([0] * kbuf)
            kb_flags = ti.Vector([0] * kbuf)
            kb_a = ti.Vector([0.0] * kbuf)
            kb_b = ti.Vector([0.0] * kbuf)
            for q in ti.static(range(kbuf)):
                kb_t[q] = hit_f[i, q, 0]
                kb_layer[q] = hit_f[i, q, 1]
                kb_a[q] = hit_f[i, q, 2]
                kb_b[q] = hit_f[i, q, 3]
                kb_prim[q] = hit_i[i, q, 0]
                kb_flags[q] = hit_i[i, q, 1]

            done = False
            drained = 0
            while drained < num_hits:
                # Nearest unconsumed slot; scalars + ti.static extraction keep
                # the kb_* vectors out of local memory (same pattern as
                # wavefront_shade).
                sel = 0
                sel_found = 0
                t_hit = 0.0
                hit_layer = 0.0
                for q in ti.static(range(kbuf)):
                    if (q < num_hits) and (kb_prim[q] >= 0):
                        if sel_found == 0:
                            sel = q
                            t_hit = kb_t[q]
                            hit_layer = kb_layer[q]
                            sel_found = 1
                        elif _comes_after(t_hit, hit_layer,
                                          kb_t[q], kb_layer[q]):
                            sel = q
                            t_hit = kb_t[q]
                            hit_layer = kb_layer[q]
                prim = 0
                flags = 0
                a = 0.0
                b = 0.0
                for q in ti.static(range(kbuf)):
                    if q == sel:
                        prim = kb_prim[q]
                        flags = kb_flags[q]
                        a = kb_a[q]
                        b = kb_b[q]
                        kb_prim[q] = -1
                drained += 1
                processed += 1
                htype = flags & 3
                edge_hit = (flags >> 2) & 1
                border = (flags >> 3) & 1

                seam_eps = depth_tie_epsilon
                if (edge_hit == 1) and (t_hit - seam_t <= seam_eps):
                    t_prev = t_hit
                    layer_prev = hit_layer
                    continue
                seam_t = t_hit if edge_hit == 1 else -1e30

                color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                alpha = 0.0
                if htype == 1:
                    w0 = 1.0 - a - b
                    color, alpha = _flat_triangle_color(
                        f, prim, w0, a, b, tri_colors, tri_uvs, tri_tex_meta,
                        textures, num_colored_triangles)
                else:
                    color, alpha = _sample_circuit_color(
                        prim, f, a, b, border,
                        circuit_meta, circuit_colors, circuit_border_colors)
                alpha = ti.math.clamp(alpha, 0.0, 1.0)

                acc += thru * color * alpha
                thru *= 1.0 - alpha
                t_alpha *= 1.0 - alpha
                t_prev = t_hit
                layer_prev = hit_layer
                if ti.max(thru[0], ti.max(thru[1], thru[2])) < min_weight:
                    done = True
                    break

            if (not done) and (num_hits < kbuf):
                # Fewer hits than the gather could hold: the peel is complete
                # and the leftover throughput shows the background.
                done = True
            if processed >= max_surfaces_per_ray:
                # Truncation, not completion (see truncation.py): a ray still
                # active here is being cut short by the ceiling.
                if not done:
                    ti.atomic_add(pt_stats[PT_STAT_TRUNC_SURFACES], 1)
                done = True

            for k in ti.static(range(4)):
                pt_thru[r, k] = thru[k]
                pt_acc[r, k] += acc[k]
            rs_sca[r, 0] = t_alpha
            rs_sca[r, 1] = t_prev
            rs_sca[r, 2] = layer_prev
            rs_sca[r, 3] = seam_t
            rs_int[r, 1] = processed
            rs_int[r, 2] = _DONE if done else _ACTIVE
            if done:
                for k in ti.static(range(4)):
                    pt_acc[r, _PT_ACC_LEFTOVER + k] = thru[k]
                pt_acc[r, _PT_ACC_ALPHA] = t_alpha
        else:
            # No surface this segment: the path escapes to the background.
            for k in ti.static(range(4)):
                pt_acc[r, _PT_ACC_LEFTOVER + k] = pt_thru[r, k]
            pt_acc[r, _PT_ACC_ALPHA] = rs_sca[r, 0]
            rs_int[r, 2] = _DONE


@ti.kernel
def pt_reduce(tile_start: ti.i32, tile_pixels: ti.i32, wave_samples: ti.i32,
              transparent: ti.i32, width: ti.i32, height: ti.i32,
              out: ti.types.ndarray(), pt_acc: ti.types.ndarray(),
              accum: ti.types.ndarray()):
    """Fold one wave's per-path rows into the chunk's per-pixel sample sums.

    One thread per tile pixel walks its own wave samples in index order --
    exclusive slots, no atomics, a fixed summation order -- which is what
    makes path-traced output reproducible run-to-run. The background
    (prefilled into ``out`` at byte scale) enters here through each path's
    leftover throughput; a sample's alpha is ``1 - t_a * (1 - bg_alpha)``
    where ``t_a`` is the deterministically-composited camera-segment
    transparency, so alpha matches the deterministic renderer's compositing
    contract in expectation (exactly, on scatter-free content).
    """
    pixels_per_frame = width * height
    for p_local in range(tile_pixels):
        g = tile_start + p_local
        f_rel = g // pixels_per_frame
        p = g - f_rel * pixels_per_frame
        sum_acc = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        sum_leftover = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        sum_t_alpha = 0.0
        for k in range(wave_samples):
            r = k * tile_pixels + p_local
            for c in ti.static(range(4)):
                sum_acc[c] += pt_acc[r, c]
                sum_leftover[c] += pt_acc[r, _PT_ACC_LEFTOVER + c]
            sum_t_alpha += pt_acc[r, _PT_ACC_ALPHA]
        background = ti.math.vec4(
            ti.cast(out[f_rel, p, 0], ti.f32),
            ti.cast(out[f_rel, p, 1], ti.f32),
            ti.cast(out[f_rel, p, 2], ti.f32),
            ti.cast(out[f_rel, p, 3], ti.f32)) / 255.0
        for c in ti.static(range(4)):
            accum[f_rel, p, c] += sum_acc[c] + sum_leftover[c] * background[c]
        if transparent != 0:
            bg_alpha = ti.cast(out[f_rel, p, 4], ti.f32) / 255.0
            accum[f_rel, p, 4] += ti.cast(wave_samples, ti.f32) \
                - sum_t_alpha * (1.0 - bg_alpha)
