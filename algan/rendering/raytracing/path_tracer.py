"""Host orchestration for the ``samples_per_pixel > 1`` path tracer.

``path_trace_render`` runs one render chunk's pixels through the PT stage
kernels (``path_tracer_taichi``) in the deterministic renderer's wavefront
shape: bounded tiles of per-path state in the arena, a
generate -> traverse -> shade -> compact iteration with host compaction, and
pointer rewinds that release each tile's state exactly.

Structure per chunk::

    for tile of pixels:                       # bounded by the arena's free bytes
        allocate per-slot state, pool = tile_pixels * wave_samples
        for wave of samples:                  # slot = (wave sample, tile pixel)
            pt_generate                       # jittered primaries, one per slot
            while any path is active:
                wavefront_traverse_events     # SHARED with the deterministic
                                              # renderer: same state layout, so
                                              # no PT traversal variant exists
                pt_shade                      # deterministic alpha peel (+ the
                                              # scattering stages, as they land)
                compact                       # keep status == _ACTIVE
            pt_reduce                         # exclusive per-pixel sums -> accum

    finalize_samples(accum) -> frame buffer   # caller (tracer.render_chunk)

One wave holds one path per (tile pixel, wave sample) so every path owns an
exclusive accumulator row: accumulation is plain stores in a fixed order (no
atomics), which makes path-traced output reproducible run-to-run for a given
configuration. The tile/wave split itself is sized from the arena's free
bytes, so byte-identity holds per machine + memory budget, like the
deterministic renderer's batch windows.

Paths never split (transparency continues in place; refraction, when it
lands, is a stochastic lobe choice), so there is no shared continuation pool,
no overflow retry, and compaction may always scan just the active list.
"""

from __future__ import annotations

import torch

from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.path_tracer_taichi import (
    PT_ACC_WIDTH,
    PT_STAT_TRUNC_SURFACES,
    PT_STATS_WIDTH,
    pt_generate,
    pt_reduce,
    pt_shade,
)
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    kbuf,
    max_surfaces_per_ray,
)
from algan.rendering.raytracing.refit_bvh import RefitBVH
from algan.rendering.raytracing.truncation import record_truncation
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    wavefront_traverse_events,
)
from algan.utils.memory_utils import InsufficientMemoryException

# Bytes of arena state per path slot: rs_ro/rs_rd (12 + 12), rs_sca (7 f32),
# rs_int (5 i32), rs_pix (i32), pt_thru (4 f32), pt_acc (PT_ACC_WIDTH f32),
# the compactor's ping-pong index pair (2 i32), and the transient
# per-iteration hit-event batch at worst case num_active == pool
# (kbuf * (4 f32 + 2 i32)).
_PT_BYTES_PER_SLOT = (
    12 + 12 + 7 * 4 + 5 * 4 + 4 + 4 * 4 + PT_ACC_WIDTH * 4 + 2 * 4
    + kbuf * (4 * 4 + 2 * 4)
)
# Per-tile fixed words: the compactor's counter, the stats tallies, alignment.
_PT_FIXED_BYTES = 64


def _pt_slots_budget(memory):
    """How many path slots one tile may hold, from the arena's free bytes.

    The same sizing policy as the deterministic wavefront's
    ``_auto_primary_per_tile`` (fewer, bigger tiles amortize launch cost;
    never overrun the arena), with the PT's own per-slot footprint. The
    result only affects how work is grouped, not what any path computes.
    """
    if not rt_settings.wavefront_tile_auto or not getattr(memory, "managed", False):
        return max(1, int(rt_settings.wavefront_tile_rays))
    free = memory.get_num_bytes_remaining()
    alignment = (-memory.current_pointer) % torch.float32.itemsize
    safety = min(1.0, max(0.0, float(rt_settings.wavefront_tile_safety)))
    usable = int(free * safety) - alignment - _PT_FIXED_BYTES
    budget = max(0, usable) // _PT_BYTES_PER_SLOT
    hi = max(1, int(rt_settings.wavefront_tile_max))
    lo = min(hi, max(1, int(rt_settings.wavefront_tile_min)))
    if budget < lo:
        return max(1, budget)
    return min(budget, hi)


def _pt_tile_shape(memory, num_pixels, samples):
    """Choose (tile pixels, wave samples) for the slot budget.

    Prefer covering every pixel in one tile and spending the leftover slots
    on samples in flight (fewer reduce launches, better machine occupancy at
    low resolutions); when the budget cannot hold all pixels, tile the pixels
    at one sample per wave. ``pt_wave_samples`` pins the wave size instead.
    """
    budget = _pt_slots_budget(memory)
    wave_pref = int(rt_settings.pt_wave_samples)
    if wave_pref > 0:
        wave = max(1, min(int(samples), wave_pref))
        tile = max(1, min(int(num_pixels), budget // wave))
        return tile, wave
    if budget >= num_pixels:
        return int(num_pixels), max(1, min(int(samples), budget // num_pixels))
    return max(1, budget), 1


def path_trace_render(
    *,
    memory,
    tri_bvh,
    bez_bvh,
    merged,
    cam_origin,
    screen_point,
    pixel_basis_x,
    pixel_basis_y,
    pixel_world_scale,
    time_start,
    time_end,
    width,
    height,
    half_screen_w,
    half_screen_h,
    layer_offset_triangles,
    has_tri,
    has_bez,
    transparent,
    samples,
    out,
    accum,
):
    """Path trace frames ``[time_start, time_end)`` into ``accum``.

    ``out`` must hold the prefilled background (byte scale) and ``accum`` the
    zeroed ``[frames, pixels, 5]`` sample sums; the caller averages them with
    ``finalize_samples``. Raises the arena's memory exceptions with all tile
    state released, so the chunk-halving retry in ``render_chunk`` works
    unchanged.
    """
    # Local import: tracer imports this module lazily at dispatch, and these
    # helpers live beside the deterministic orchestration it reuses.
    from algan.rendering.raytracing.tracer import _arena_values, _ArenaRayCompactor

    i32 = torch.int32
    f32 = torch.float32
    device = memory.data.device
    num_frames = int(time_end) - int(time_start)
    n = num_frames * int(width) * int(height)
    if n <= 0 or int(samples) <= 0:
        return
    samples = int(samples)
    seed_root = int(rt_settings.pt_seed) & 0xFFFFFFFF
    bvh_refit = 1 if isinstance(tri_bvh, RefitBVH) else 0

    with memory.scope("pt_metadata"):
        # The traverse kernel rebuilds each pixel's primary ray from
        # gen_meta[2:] to convert slant ranges to perpendicular depth (see
        # wavefront_traverse_events); the jitter pair is unused (gen_first=0).
        gen_meta = _arena_values(
            memory, [0.5, 0.5, float(half_screen_w), float(half_screen_h)], f32
        )

    tile_pixels, wave_samples = _pt_tile_shape(memory, n, samples)
    # Per-slot init rows (see path_tracer_taichi's state notes): rs_sca =
    # [t_alpha=1, t_prev=0, layer_prev=1e30, seam_t=-1e30, base_dist=0, 0, 0];
    # rs_int all-zero = [bounces used=0, processed=0, _ACTIVE, no hits, spare].
    sca_init = torch.tensor(
        [1.0, 0.0, 1e30, -1e30, 0.0, 0.0, 0.0], dtype=f32, device=device
    )

    tile_start = 0
    while tile_start < n:
        tp = min(tile_pixels, n - tile_start)
        pool = tp * wave_samples
        state_ptrs = memory.get_pointers()
        try:
            with memory.scope("pt_state", slots=pool):
                rs_ro = memory.get_tensor((pool, 3), f32)
                rs_rd = memory.get_tensor((pool, 3), f32)
                rs_sca = memory.get_tensor((pool, 7), f32)
                rs_int = memory.get_tensor((pool, 5), i32)
                rs_pix = memory.get_tensor((pool,), i32)
                pt_thru = memory.get_tensor((pool, 4), f32)
                pt_acc = memory.get_tensor((pool, PT_ACC_WIDTH), f32)
                pt_stats = memory.get_tensor((PT_STATS_WIDTH,), i32)
            compactor = _ArenaRayCompactor(memory, pool, i32)
            pt_stats.zero_()

            for sample_base in range(0, samples, wave_samples):
                sw = min(wave_samples, samples - sample_base)
                slots = tp * sw
                rs_sca[:slots].copy_(sca_init)
                rs_int[:slots].zero_()
                pt_thru[:slots].fill_(1.0)
                pt_acc[:slots].zero_()
                pt_generate(
                    int(slots),
                    int(tp),
                    int(sample_base),
                    seed_root,
                    int(time_start),
                    int(width),
                    int(height),
                    int(tile_start),
                    float(half_screen_w),
                    float(half_screen_h),
                    cam_origin,
                    screen_point,
                    pixel_basis_x,
                    pixel_basis_y,
                    rs_ro,
                    rs_rd,
                    rs_pix,
                )
                active = compactor.initial(slots)
                it = 0
                max_iters = max_surfaces_per_ray + 4
                while active.numel() > 0 and it < max_iters:
                    na = int(active.numel())
                    with memory.temp():
                        hit_f = memory.get_tensor((na, kbuf, 4), f32)
                        hit_i = memory.get_tensor((na, kbuf, 2), i32)
                        wavefront_traverse_events(
                            active,
                            na,
                            tri_bvh.blocks,
                            tri_bvh.node_miss,
                            tri_bvh.leaf_prim,
                            tri_bvh.leaf_tspan,
                            int(tri_bvh.first_leaf),
                            merged["tri_pos"],
                            bez_bvh.blocks,
                            bez_bvh.node_miss,
                            bez_bvh.leaf_prim,
                            bez_bvh.leaf_tspan,
                            int(bez_bvh.first_leaf),
                            merged["circuit_meta"],
                            merged["edges_2d"],
                            merged["edge_accel"],
                            merged["tri_opaque_bvh"].blocks,
                            merged["tri_opaque_bvh"].node_miss,
                            merged["tri_opaque_bvh"].leaf_prim,
                            merged["tri_opaque_bvh"].leaf_tspan,
                            int(merged["tri_opaque_bvh"].first_leaf),
                            merged["bez_opaque_bvh"].blocks,
                            merged["bez_opaque_bvh"].node_miss,
                            merged["bez_opaque_bvh"].leaf_prim,
                            merged["bez_opaque_bvh"].leaf_tspan,
                            int(merged["bez_opaque_bvh"].first_leaf),
                            pixel_world_scale,
                            float(layer_offset_triangles),
                            bvh_refit,
                            int(has_tri),
                            int(has_bez),
                            0,  # opaque_closest: deterministic-only rollout
                            0,  # opaque_prepass: deterministic-only rollout
                            int(time_start),
                            int(width),
                            int(height),
                            int(tile_start),
                            rs_ro,
                            rs_rd,
                            rs_sca,
                            rs_int,
                            hit_f,
                            hit_i,
                            rs_pix,
                            0,  # gen_first: PT generation is its own kernel
                            cam_origin,
                            screen_point,
                            pixel_basis_x,
                            pixel_basis_y,
                            gen_meta,
                        )
                        pt_shade(
                            active,
                            na,
                            merged["tri_colors"],
                            merged["tri_uvs"],
                            merged["tri_tex_meta"],
                            merged["textures"],
                            int(merged["num_colored_triangles"]),
                            merged["circuit_meta"],
                            merged["circuit_colors"],
                            merged["circuit_border_colors"],
                            int(time_start),
                            int(width),
                            int(height),
                            int(tile_start),
                            rs_sca,
                            rs_int,
                            rs_pix,
                            hit_f,
                            hit_i,
                            pt_thru,
                            pt_acc,
                            pt_stats,
                        )
                    active = compactor.select(rs_int, 0, source=active)
                    it += 1
                pt_reduce(
                    int(tile_start),
                    int(tp),
                    int(sw),
                    1 if transparent else 0,
                    int(width),
                    int(height),
                    out,
                    pt_acc,
                    accum,
                )

            truncated = int(pt_stats[PT_STAT_TRUNC_SURFACES].item())
            if truncated:
                record_truncation(
                    "surfaces_per_ray", truncated, cap=max_surfaces_per_ray
                )
        except (InsufficientMemoryException, RuntimeError):
            # Release the tile's state before the chunk-halving retry in
            # render_chunk sees the exception (mirrors run_tile).
            memory.set_pointers(state_ptrs)
            raise
        memory.set_pointers(state_ptrs)
        tile_start += tp
