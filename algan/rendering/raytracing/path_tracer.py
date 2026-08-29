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

import math

import torch

from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.path_tracer_taichi import (
    _NEE_EMISSIVE_TRI,
    _NEE_ENV,
    _NEE_LIGHT_ROW,
    _NM_COUNT,
    _NM_ENV_CDF_H,
    _NM_ENV_CDF_W,
    _NM_ENV_H,
    _NM_ENV_INTENSITY,
    _NM_ENV_OFF,
    _NM_ENV_SHARE,
    _NM_ENV_W,
    _NM_LIGHT_SAMPLES,
    _SHELL_RING_SLOTS,
    NEE_META_WIDTH,
    PT_ACC_WIDTH,
    PT_INT_WIDTH,
    PT_STAT_SHELL_RING,
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
from algan.rendering.raytracing.shading_taichi import (
    _LT_AREA_SAMPLE,
    _LT_DIRECTIONAL,
    _LT_POINT,
    _LT_SPOT,
    _MID_LAMBERT,
    _MID_PHYSICAL,
    ALL_PIDS,
)
from algan.rendering.raytracing.truncation import record_truncation
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    SCA_WIDTH_NESTED,
    wavefront_traverse_events,
)
from algan.utils.memory_utils import InsufficientMemoryException

# Bytes of arena state per path slot: rs_ro/rs_rd (12 + 12), rs_sca (the
# nested-IOR width -- the path tracer always carries the media stack),
# rs_int (PT_INT_WIDTH i32: the shared 5 plus the closed-shell ring),
# rs_pix (i32), pt_thru (4 f32), pt_acc (PT_ACC_WIDTH f32),
# the compactor's ping-pong index pair (2 i32), and the transient
# per-iteration hit-event batch at worst case num_active == pool
# (kbuf * (4 f32 + 2 i32)).
_PT_BYTES_PER_SLOT = (
    12
    + 12
    + SCA_WIDTH_NESTED * 4
    + PT_INT_WIDTH * 4
    + 4
    + 4 * 4
    + PT_ACC_WIDTH * 4
    + 2 * 4
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


def _build_env_cdf(env_rgb, max_h=128, max_w=256):
    """2D sampling distribution of an equirect environment map.

    Bins the map's peak-channel luminance (the codebase's colour-to-scalar
    convention) times sin(theta) into at most ``max_h x max_w`` cells, with a
    1% uniform floor so the pdf is positive wherever bilinear filtering can
    leak radiance out of a bright texel's cell -- what keeps the estimator
    unbiased rather than merely well-aimed. Returns ``(env_cdf, power)``:
    the ``[H, W + 1]`` float32 tensor the kernels binary-search (row
    conditionals in columns ``0..W-1``, the row marginal in column ``W``)
    and the map's total luminance integral (before ``environment_intensity``)
    for the selection-table power weight.
    """
    h = int(env_rgb.shape[0])
    w = int(env_rgb.shape[1])
    ch = max(1, min(int(max_h), h))
    cw = max(1, min(int(max_w), w))
    lum = env_rgb.amax(-1).clamp_min(0).double()
    if (h, w) != (ch, cw):
        lum = torch.nn.functional.adaptive_avg_pool2d(
            lum.unsqueeze(0).unsqueeze(0), (ch, cw)
        )[0, 0]
    v = (torch.arange(ch, dtype=torch.float64, device=lum.device) + 0.5) / ch
    sin_t = torch.sin(math.pi * v).unsqueeze(1)
    w_bin = (lum + 0.01 * lum.mean() + 1e-12) * sin_t
    row = w_bin.sum(1)
    cond = w_bin.cumsum(1) / row.unsqueeze(1)
    cond[:, -1] = 1.0
    marg = row.cumsum(0) / row.sum()
    marg[-1] = 1.0
    env_cdf = torch.empty((ch, cw + 1), dtype=torch.float64, device=lum.device)
    env_cdf[:, :cw] = cond
    env_cdf[:, cw] = marg
    power = float((lum * sin_t).sum() * (math.pi / ch) * (2.0 * math.pi / cw))
    return env_cdf.float(), power


def _build_nee_tables(memory, merged, light_pos, light_col, num_lights, env_meta):
    """Build one render call's power-weighted next-event table.

    One flat CDF over everything a shadow ray can aim at -- delta and
    area-cell light rows (ambient-like rows are the kernel's deterministic
    fill and never enter), emissive lit triangles (frame-0 peak luminance
    times area; an emitter dark at frame 0 is simply never NEE-sampled and
    reaches the image through BSDF hits at weight 1, which stays unbiased),
    and one environment entry when a map is present and ``pt_env_nee`` is
    on. Light-row weights take the max over frames so a light dark at frame
    0 but lit later is still sampled (rows have no MIS backstop).

    Returns arena tensors ``(nee_cdf [E], nee_ref [E, 2], nee_meta
    [NEE_META_WIDTH], tri_emit_prob [N], env_cdf [H, W + 1])`` -- every
    selection probability the kernels divide by or MIS against comes from
    these, so both ends of each MIS pair see identical numbers.
    """
    from algan.rendering.raytracing.tracer import _arena_copy, _arena_values

    device = memory.data.device
    i64 = torch.int64
    powers = []
    kinds = []
    refs = []
    if num_lights > 0:
        row_power = light_col[..., :3].amax(0).amax(-1).double()
        if light_col.shape[2] > 3:
            ltypes = (light_col[0, :, 3] + 0.5).to(i64)
        else:
            ltypes = torch.zeros(num_lights, dtype=i64, device=device)
        sampled = (
            (ltypes == _LT_POINT)
            | (ltypes == _LT_DIRECTIONAL)
            | (ltypes == _LT_SPOT)
            | (ltypes == _LT_AREA_SAMPLE)
        ) & (row_power > 0)
        idx = sampled.nonzero(as_tuple=False).flatten()
        if idx.numel():
            powers.append(row_power[idx])
            kinds.append(torch.full_like(idx, _NEE_LIGHT_ROW))
            refs.append(idx)

    tri_mat = merged["tri_mat"]
    n_tri = int(merged.get("num_triangles") or 0)
    if n_tri > 0 and int(tri_mat.shape[2]) > 3:
        pid = merged["tri_mat_id"][0].to(i64)
        lit = (pid >= _MID_LAMBERT) & (pid <= _MID_PHYSICAL)
        em = tri_mat[0, :, 0:3].amax(-1).clamp_min(0) * tri_mat[0, :, 3].clamp_min(0)
        p9 = merged["tri_pos"][0].double()
        area = 0.5 * torch.linalg.cross(
            p9[:, 3:6] - p9[:, 0:3], p9[:, 6:9] - p9[:, 0:3], dim=-1
        ).norm(dim=-1)
        p_e = torch.where(lit, em.double() * area * math.pi, torch.zeros_like(area))
        e_idx = (p_e > 0).nonzero(as_tuple=False).flatten()
        if e_idx.numel():
            powers.append(p_e[e_idx])
            kinds.append(torch.full_like(e_idx, _NEE_EMISSIVE_TRI))
            refs.append(e_idx)

    # Environment geometry rides the meta vector whenever a map is packed
    # (the escape fold needs it with or without env NEE); the CDF and the
    # selection entry exist only under ``pt_env_nee``.
    env_off = env_w = env_h = 0
    env_intensity = 0.0
    env_cdf_host = None
    if env_meta is not None:
        env_off = int(env_meta[0])
        env_w = int(env_meta[1])
        env_h = int(env_meta[2])
        env_intensity = float(env_meta[3])
    if env_w > 0 and env_h > 0 and rt_settings.pt_env_nee:
        texels = merged["textures"][0, env_off : env_off + env_w * env_h, 0:3]
        env_rgb = texels.float().reshape(env_w, env_h, 3).permute(1, 0, 2)
        env_cdf_host, env_power = _build_env_cdf(env_rgb)
        env_power *= max(env_intensity, 0.0)
        if env_power > 0:
            powers.append(torch.tensor([env_power], dtype=torch.float64, device=device))
            kinds.append(torch.tensor([_NEE_ENV], dtype=i64, device=device))
            refs.append(torch.tensor([0], dtype=i64, device=device))

    env_share = 0.0
    if powers:
        power = torch.cat(powers)
        prob = power / power.sum()
        cdf = prob.cumsum(0)
        cdf[-1] = 1.0
        kind = torch.cat(kinds)
        ref = torch.cat(refs)
        num_entries = int(cdf.numel())
        if bool((kind[-1] == _NEE_ENV).item()):
            env_share = float(prob[-1].item())
    else:
        num_entries = 0

    with memory.scope(
        "pt_nee_tables", entries=max(num_entries, 1), emitters=max(n_tri, 1)
    ):
        emit_prob = memory.get_tensor((max(n_tri, 1),), torch.float32)
        emit_prob.zero_()
        if num_entries > 0:
            nee_cdf = _arena_copy(memory, cdf.float())
            nee_ref = _arena_copy(
                memory,
                torch.stack((kind, ref), -1).to(torch.int32),
            )
            emissive_rows = kind == _NEE_EMISSIVE_TRI
            if bool(emissive_rows.any().item()):
                emit_prob[ref[emissive_rows]] = prob[emissive_rows].float()
        else:
            nee_cdf = memory.get_tensor((1,), torch.float32)
            nee_cdf.zero_()
            nee_ref = memory.get_tensor((1, 2), torch.int32)
            nee_ref.zero_()
        if env_cdf_host is not None:
            env_cdf = _arena_copy(memory, env_cdf_host)
            cdf_h, cdf_w = int(env_cdf.shape[0]), int(env_cdf.shape[1]) - 1
        else:
            env_cdf = memory.get_tensor((1, 2), torch.float32)
            env_cdf.zero_()
            cdf_h = cdf_w = 1
        meta = [0.0] * NEE_META_WIDTH
        meta[_NM_COUNT] = float(num_entries)
        meta[_NM_ENV_SHARE] = env_share
        meta[_NM_LIGHT_SAMPLES] = float(max(1, int(rt_settings.pt_light_samples)))
        meta[_NM_ENV_OFF] = float(env_off)
        meta[_NM_ENV_W] = float(env_w)
        meta[_NM_ENV_H] = float(env_h)
        meta[_NM_ENV_INTENSITY] = env_intensity
        meta[_NM_ENV_CDF_H] = float(cdf_h)
        meta[_NM_ENV_CDF_W] = float(cdf_w)
        nee_meta = _arena_values(memory, meta, torch.float32)
    return nee_cdf, nee_ref, nee_meta, emit_prob, env_cdf


def _build_shell_table(memory, merged):
    """Per-triangle closed-shell ids for the camera-segment opacity ring.

    ``tri_shell[f % rows, n]`` is the triangle's ``tri_obj`` surface id where
    it belongs to a declared closed shell whose coverage may be ceilinged
    (``tri_closed``, already folded with the transmission exemption at pack
    time), and -1 everywhere else -- one gather per crossing in the kernel
    instead of two.  ``tri_obj`` and ``tri_closed`` collapse independently
    under ``merge_dedup_time``, so the two are broadcast against each other;
    a collapsed row means "the same every frame", which ``f % rows`` indexing
    preserves.  Scenes with nothing declared -- ``solid_shell_alpha`` off, a
    triangle-free merge (which builds no ``tri_closed`` at all), or no
    declaring mob -- share a ``[1, 1]`` placeholder of -1, which the kernel's
    ``>= 0`` gate never acts on.
    """
    from algan.rendering.raytracing.tracer import _arena_copy

    tri_closed = merged.get("tri_closed") if rt_settings.solid_shell_alpha else None
    if tri_closed is not None:
        closed = tri_closed > 0.5
        if bool(closed.any()):
            shell = torch.where(
                closed,
                merged["tri_obj"].to(torch.int32),
                torch.full((1, 1), -1, dtype=torch.int32, device=tri_closed.device),
            )
            with memory.scope("pt_shell_table", rows=int(shell.shape[0])):
                return _arena_copy(memory, shell.contiguous())
    with memory.scope("pt_shell_table", rows=1):
        placeholder = memory.get_tensor((1, 1), torch.int32)
    placeholder.fill_(-1)
    return placeholder


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
    light_pos,
    light_col,
    num_lights,
    frag_pipelines,
    shadows,
    max_bounces,
    transparent,
    samples,
    env_meta=None,
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
    # The power-weighted next-event table + environment CDF for this call
    # (before the tile budget is taken, so their bytes are accounted).
    nee_cdf, nee_ref, nee_meta, tri_emit_prob, env_cdf = _build_nee_tables(
        memory, merged, light_pos, light_col, int(num_lights), env_meta
    )
    tri_shell = _build_shell_table(memory, merged)

    tile_pixels, wave_samples = _pt_tile_shape(memory, n, samples)
    # Per-slot init rows (see path_tracer_taichi's state notes): rs_sca =
    # [t_alpha=1, t_prev=0, layer_prev=1e30, seam_t=-1e30, base_dist=0,
    # prev_pdf=-1 (camera segment; _SCA_PREV_PDF), 0] plus the zeroed
    # nested-IOR stack columns (air outside); rs_int =
    # [bounces_left=max_bounces, processed=0, _ACTIVE, no hits,
    # max_bounces (the bounce ordinal's reference)] plus the empty (-1)
    # closed-shell ring.
    sca_init = torch.tensor(
        [1.0, 0.0, 1e30, -1e30, 0.0, -1.0, 0.0] + [0.0] * (SCA_WIDTH_NESTED - 7),
        dtype=f32,
        device=device,
    )
    int_init = torch.tensor(
        [int(max_bounces), 0, 0, 0, int(max_bounces)] + [-1] * (PT_INT_WIDTH - 5),
        dtype=i32,
        device=device,
    )
    rr_start = max(0, int(rt_settings.pt_rr_start_bounce))
    firefly_clamp = float(rt_settings.pt_firefly_clamp)

    tile_start = 0
    while tile_start < n:
        tp = min(tile_pixels, n - tile_start)
        pool = tp * wave_samples
        state_ptrs = memory.get_pointers()
        try:
            with memory.scope("pt_state", slots=pool):
                rs_ro = memory.get_tensor((pool, 3), f32)
                rs_rd = memory.get_tensor((pool, 3), f32)
                rs_sca = memory.get_tensor((pool, SCA_WIDTH_NESTED), f32)
                rs_int = memory.get_tensor((pool, PT_INT_WIDTH), i32)
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
                rs_int[:slots].copy_(int_init)
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
                            tri_bvh.blocks,
                            tri_bvh.node_miss,
                            tri_bvh.leaf_prim,
                            tri_bvh.leaf_tspan,
                            int(tri_bvh.first_leaf),
                            merged["tri_pos"],
                            merged["tri_norm"],
                            merged["tri_extra"],
                            merged["tri_colors"],
                            merged["tri_uvs"],
                            merged["tri_tex_meta"],
                            merged["textures"],
                            int(merged["num_colored_triangles"]),
                            bez_bvh.blocks,
                            bez_bvh.node_miss,
                            bez_bvh.leaf_prim,
                            bez_bvh.leaf_tspan,
                            int(bez_bvh.first_leaf),
                            merged["circuit_meta"],
                            merged["circuit_colors"],
                            merged["circuit_border_colors"],
                            merged["edges_2d"],
                            merged["edge_accel"],
                            merged["tri_mat_id"],
                            merged["tri_mat"],
                            light_pos,
                            light_col,
                            int(num_lights),
                            pixel_world_scale,
                            float(layer_offset_triangles),
                            cam_origin,
                            bvh_refit,
                            int(has_tri),
                            int(has_bez),
                            int(shadows),
                            frag_pipelines,
                            ALL_PIDS,
                            seed_root,
                            int(sample_base),
                            int(tp),
                            rr_start,
                            firefly_clamp,
                            int(time_start),
                            int(width),
                            int(height),
                            int(tile_start),
                            rs_ro,
                            rs_rd,
                            rs_sca,
                            rs_int,
                            rs_pix,
                            hit_f,
                            hit_i,
                            pt_thru,
                            pt_acc,
                            pt_stats,
                            nee_cdf,
                            nee_ref,
                            nee_meta,
                            tri_emit_prob,
                            env_cdf,
                            tri_shell,
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
            ring_over = int(pt_stats[PT_STAT_SHELL_RING].item())
            if ring_over:
                record_truncation("closed_shell_ring", ring_over, cap=_SHELL_RING_SLOTS)
        except (InsufficientMemoryException, RuntimeError):
            # Release the tile's state before the chunk-halving retry in
            # render_chunk sees the exception (mirrors run_tile).
            memory.set_pointers(state_ptrs)
            raise
        memory.set_pointers(state_ptrs)
        tile_start += tp
