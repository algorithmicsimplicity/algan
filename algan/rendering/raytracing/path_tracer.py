"""Host orchestration for the ``samples_per_pixel > 1`` path tracer.

``path_trace_render`` runs one render chunk's pixels through the PT stage
kernels (``path_tracer_taichi``) in the deterministic renderer's wavefront
shape: bounded tiles of per-path state in the arena, a
generate -> traverse -> shade -> compact iteration with host compaction, and
pointer rewinds that release each tile's state exactly.

Structure per chunk::

    for tile of pixels:                       # bounded by the arena's free bytes
        allocate per-slot state, pool = tile_pixels * wave_samples
        for wave of samples:                  # slot = (wave sample, wave pixel)
            pt_generate                       # jittered primaries, one per slot
            while any path is active:
                wavefront_traverse_events     # SHARED with the deterministic
                                              # renderer: same state layout, so
                                              # no PT traversal variant exists
                pt_shade                      # deterministic alpha peel (+ the
                                              # scattering stages, as they land)
                compact                       # keep status == _ACTIVE
            pt_reduce                         # exclusive per-pixel sums -> accum
            compact the pixel list            # adaptive sampling only

    finalize_samples(accum) -> frame buffer   # caller (tracer.render_chunk)

Every wave runs over an explicit **pixel list** (``pt_generate``'s
``pix_list``), not over a contiguous span: the uniform loop hands it the
tile's identity list, and adaptive sampling (``pt_error_target > 0``,
roadmap section 2) hands it the pixels that have not finished. One code path,
and ``samples_per_pixel`` becomes a ceiling that converged pixels never
reach. A pixel may stop early only if **none of its samples took a random
decision** -- the kernel flags those (``_PT_ACC_STOCH``) and the host refuses
to stop a pixel that has any, whatever its error estimate says, because a
Monte Carlo estimator cannot tell "converged at zero" from "has not found the
light yet". What is left to stop is the zero-variance content section 2 was
aimed at: 2-D interiors, unlit stacks and the background.

One wave holds one path per (tile pixel, wave sample), so every path owns an
exclusive accumulator row and accumulation is plain stores rather than
atomics. That is a property of the current layout, **not** a contract: the
renderer promises convergence, not byte-identical frames, and a future
feature that needs paths to split (see ``DESIGN_path_tracer_roadmap.md``
section 8) is free to trade this layout for a shared pool with atomic
accumulation.

Paths do not split today (transparency continues in place; a dielectric
picks reflect-or-refract stochastically), so there is no shared continuation
pool, no overflow retry, and compaction may always scan just the active
list.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from algan.logging.logger import PERF, get_logger
from algan.rendering.mps_compat import accumulate_dtype
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.area_light_quads import NO_QUAD_BASE
from algan.rendering.raytracing.light_tree import (
    LT_F_WIDTH,
    LT_I_WIDTH,
    LT_LEFT,
    LT_RIGHT,
    build_light_trees,
)
from algan.rendering.raytracing.path_tracer_taichi import (
    _NEE_AMBIENT_ROW,
    _NEE_AUTHORED_ROW,
    _NEE_EMISSIVE_TRI,
    _NEE_ENV,
    _NEE_LIGHT_ROW,
    _NM_AMBIENT_COUNT,
    _NM_AMBIENT_PACKED,
    _NM_ANIM_SEED,
    _NM_AOV,
    _NM_AUTHORED_COUNT,
    _NM_AUTHORED_SAMPLES,
    _NM_COUNT,
    _NM_ENV_CDF_H,
    _NM_ENV_CDF_W,
    _NM_ENV_H,
    _NM_ENV_INTENSITY,
    _NM_ENV_OFF,
    _NM_ENV_SHARE,
    _NM_ENV_W,
    _NM_FAR_CLIP,
    _NM_INF_COUNT,
    _NM_LIGHT_SAMPLES,
    _NM_QUAD_BASE,
    _NM_TREE_MIX,
    _NM_TREE_ON,
    _SHELL_RING_SLOTS,
    NEE_META_WIDTH,
    PT_ACC_WIDTH,
    PT_AOV_WIDTH,
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
    _LT_AMBIENT,
    _LT_AREA_SAMPLE,
    _LT_DIRECTIONAL,
    _LT_HEMISPHERE,
    _LT_POINT,
    _LT_SPOT,
    _MAT_ONE_SIDED,
    _MID_LAMBERT,
    _MID_PHYSICAL,
    ALL_PIDS,
    max_shadow_lights,
    shadow_vis_slots,
)
from algan.rendering.raytracing.truncation import (
    record_path_samples,
    record_truncation,
)
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    SCA_WIDTH_NESTED,
    wavefront_traverse_events,
)
from algan.utils.memory_utils import InsufficientMemoryException

logger = get_logger("raytracing")

# Bytes of arena state per path slot: rs_ro/rs_rd (12 + 12), rs_sca (the
# nested-IOR width -- the path tracer always carries the media stack),
# rs_int (PT_INT_WIDTH i32: the shared 5 plus the closed-shell ring),
# rs_pix (i32), pt_thru (4 f32), pt_acc (PT_ACC_WIDTH f32-- the radiance,
# leftover and alpha columns plus adaptive sampling's stochastic flag), pt_aov
# (PT_AOV_WIDTH f32 -- budgeted whether or not the denoiser's AOVs are on,
# since the budget only shapes work grouping, never results),
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
    + PT_AOV_WIDTH * 4
    + 2 * 4
    + kbuf * (4 * 4 + 2 * 4)
)
# Per-tile fixed words: the compactor's counter, the stats tallies, alignment.
_PT_FIXED_BYTES = 64

#: Absolute floor in the adaptive error metric's denominator, in the frame
#: buffer's linear radiance units (1.0 = display white). Without it a pixel
#: whose radiance is near zero divides a tiny difference by a tinier sum and
#: runs to the ceiling for noise nobody can see; with it, the tolerance at
#: ``pt_error_target = t`` is ``t * _PT_ERR_EPS`` in absolute linear units
#: wherever the signal is smaller than the floor.
#:
#: 0.02 is chosen so the accepted half-buffer difference is roughly CONSTANT
#: in the 8-bit counts the frame is finally written as. Pushing
#: ``t * (value + 0.02)`` through the sRGB OETF's slope at each level, a
#: target of 0.02 accepts about 1.4 counts at linear 0.001, 1.3 at 0.01, 1.9
#: at 0.1 and 2.2 at 1.0 -- flat across four decades, because the encode's
#: log-like shape cancels the metric's proportionality to the signal. That is
#: also why the metric is computed on linear radiance with no perceptual
#: transform: a *relative* metric is invariant under any power law, so sqrt or
#: a PU curve would only rescale it by a constant, and this floor is the only
#: part of the formula that actually decides how darks are treated.
_PT_ERR_EPS = 0.02


def pt_adaptive_active(samples):
    """Whether adaptive sampling runs for a render of ``samples`` spp.

    ``pt_error_target = 0`` is the byte-parity escape hatch: uniform waves,
    no half-sum buffer, no per-pixel rescale. A ceiling below two samples has
    nothing to stop early either, and the estimator needs balanced halves.
    Read live off ``rt_settings`` at call time, like every other toggle.
    """
    return int(samples) >= 2 and float(rt_settings.pt_error_target) > 0.0


def _pt_floor_samples(samples):
    """The floor every pixel is given before any of them may stop.

    Forced even (the estimator splits a pixel's samples into halves, and an
    odd count leaves them unbalanced) and at least 2, then capped by the
    ceiling.
    """
    floor = max(2, int(rt_settings.pt_min_samples))
    floor += floor % 2
    return min(floor, int(samples))


def _log_sample_spread(n_p, samples, num_cells, floor=None):
    """One PERF line per chunk: what the ceiling actually cost.

    Budget events log at PERF rather than WARNING (``truncation.py``'s rule):
    stopping a converged pixel is the sampler working as designed, not a
    degradation of the image. ``n_p`` of ``None`` is the uniform arm, where
    every pixel is at the ceiling by construction and nothing needs reading
    back from the device.
    """
    if not logger.isEnabledFor(PERF):
        return
    if n_p is None:
        logger.log(
            PERF,
            f"path tracer samples/pixel: uniform {samples} "
            f"(pt_error_target = 0) over {num_cells} cells.",
        )
        return
    at_floor = int((n_p <= floor).sum().item())
    at_ceiling = int((n_p >= samples).sum().item())
    mean = float(n_p.to(torch.float32).mean().item())
    logger.log(
        PERF,
        f"path tracer samples/pixel: mean {mean:.2f} of {samples} "
        f"({at_floor} of {num_cells} cells stopped at the floor {floor}, "
        f"{at_ceiling} reached the ceiling).",
    )


def _pt_wave_size(pool, active_pixels, sample_base, floor_samples, remaining):
    """How many samples the next adaptive wave puts in flight, per pixel.

    Three limits, in order of who imposes them:

    * **The pool.** The tile's slots were allocated for the whole tile, so a
      compacted list of ``active_pixels`` may spend ``pool // active_pixels``
      of them per pixel -- exactly the "same budget, fewer pixels" the
      shrinking list buys. ``//`` against the pool that was actually
      allocated (rather than against the arena budget, which the state
      allocation has since eaten into) is what keeps it from overrunning.
    * **The decision schedule.** The first waves take the tile to
      ``floor_samples`` and stop there, because a single wave of
      ``samples_per_pixel`` -- which is what the budget allows at any
      resolution a whole frame fits in one tile -- would finish the render
      before a single pixel could be retired. Above the floor the count at
      most DOUBLES per wave, so the error is re-checked at 2x granularity and
      a pixel overshoots its true stopping point by at most a factor of two.
    * **The ceiling**, and evenness: the estimator splits a pixel's samples
      into halves, so every wave but the last keeps the running total even.
      The last wave may be odd -- no decision is taken after it.
    """
    if sample_base < floor_samples:
        limit = floor_samples - sample_base
    else:
        limit = min(remaining, max(2, sample_base))
    sw = max(1, min(pool // max(1, active_pixels), limit))
    if sw > 1 and sw < remaining:
        sw -= sw % 2
    return max(1, sw)


def _pt_active_pixels(
    accum, accum_odd, pix_list, num_samples, target, keep_buf, tile_start, width
):
    """The subset of ``pix_list`` that may not stop yet.

    **A pixel may stop early only if every one of its samples was
    deterministic given the sub-pixel jitter**, and only then does its error
    estimate get a vote. ``accum_odd[..., 3]`` is the count of samples whose
    path took any random decision (``_PT_ACC_STOCH`` in
    ``path_tracer_taichi``: a lit crossing's next-event estimation, an
    authored crossing or a custom scatter, a lobe pick with more than the
    pass-through branch), and a non-zero count runs the pixel to the ceiling
    unconditionally.

    That gate is the correctness of the whole mechanism, not a refinement of
    it. A half-buffer difference cannot see the estimator's one failure mode:
    a pixel whose first samples ALL return zero has two halves that agree
    exactly, and no choice of ``target`` or ``eps`` distinguishes "converged
    at zero" from "has not found the light yet". It was not hypothetical --
    on ``tests/path_traced/scenes/lit_and_shadowed.py``, whose next-event
    table is dominated by an emissive slab that most surface points cannot
    see, a purely statistical rule left 249 lit pixels of 9216 stuck at pure
    black (255 counts). The kernel knows which paths gambled, so it says so
    and no lit pixel is ever stopped on evidence a Monte Carlo estimator
    cannot supply. What is left to stop is exactly what section 2 was for:
    2-D interiors, unlit stacks, and the background, which are zero-variance
    by construction (roadmap contract 4).

    For those eligible pixels, with ``n = num_samples`` samples so far each
    half holds ``n / 2`` of them: ``O = accum_odd / (n/2)`` is the odd half's
    mean and ``E = (accum_rgb - accum_odd) / (n/2)`` the even half's, so

        err = max_c |E - O| / (max_c (E + O) + _PT_ERR_EPS)

    which in terms of the stored sums is
    ``max_c|rgb - 2*odd| / (max_c(rgb) + (n/2) * eps)``. Every input is a
    deterministic sum, so the decision -- which pixels stop, and after how
    many samples -- is a reproducible function of the rendered data rather
    than of how the render was tiled.

    Only the three colour channels enter. Alpha (column 4, transparent
    output only) rides the same paths and carries no independent variance,
    and the leftover columns are already folded into ``accum``'s RGB by
    ``pt_reduce``.

    **A pixel is also kept alive when a 4-neighbour is.** With the
    stochastic gate in front of it this is no longer load-bearing for
    correctness, but it is still worth its cost: a 2-D edge pixel is
    deterministic given its jitter, and four jittered samples that happen to
    agree would otherwise freeze a coverage value the neighbouring edge
    pixels are visibly still resolving. The dilation is grown from the
    unconverged pixels each round, never from the rescued ones, so the ring
    stays one pixel wide; it runs over the flat cell index with a stride of
    ``width``, so it wraps at row ends (one spurious neighbour per row, which
    can only be conservative); and it never revives a pixel that has already
    stopped, because only pixels still in ``pix_list`` are candidates. That
    is what keeps every live pixel's sample count equal, and its Sobol prefix
    contiguous.

    ``keep_buf`` is a ``[tile_pixels]`` bool scratch buffer, left all-False
    on exit so the next call need not clear it.
    """
    idx = pix_list.long()
    rgb = accum.view(-1, accum.shape[-1])[idx, :3]
    odd_row = accum_odd.view(-1, 4)[idx]
    odd = odd_row[:, :3]
    inv_half = 2.0 / float(num_samples)
    diff = (rgb - 2.0 * odd).abs().amax(-1) * inv_half
    scale = rgb.amax(-1).clamp_min(0.0) * inv_half
    keep = (diff > target * (scale + _PT_ERR_EPS)) | (odd_row[:, 3] > 0.0)
    local = idx - tile_start
    keep_buf[local] = keep
    grown = keep_buf.clone()
    grown[1:] |= keep_buf[:-1]
    grown[:-1] |= keep_buf[1:]
    if keep_buf.numel() > width:
        grown[width:] |= keep_buf[:-width]
        grown[:-width] |= keep_buf[width:]
    keep_buf[local] = False
    return pix_list[grown[local]]


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
    acc = accumulate_dtype()
    lum = env_rgb.amax(-1).clamp_min(0).to(acc)
    if (h, w) != (ch, cw):
        lum = torch.nn.functional.adaptive_avg_pool2d(
            lum.unsqueeze(0).unsqueeze(0), (ch, cw)
        )[0, 0]
    v = (torch.arange(ch, dtype=acc, device=lum.device) + 0.5) / ch
    sin_t = torch.sin(math.pi * v).unsqueeze(1)
    w_bin = (lum + 0.01 * lum.mean() + 1e-12) * sin_t
    row = w_bin.sum(1)
    cond = w_bin.cumsum(1) / row.unsqueeze(1)
    cond[:, -1] = 1.0
    marg = row.cumsum(0) / row.sum()
    marg[-1] = 1.0
    env_cdf = torch.empty((ch, cw + 1), dtype=acc, device=lum.device)
    env_cdf[:, :cw] = cond
    env_cdf[:, cw] = marg
    power = float((lum * sin_t).sum() * (math.pi / ch) * (2.0 * math.pi / cw))
    return env_cdf.float(), power


_HALF_PI = 0.5 * math.pi


def _row_tree_geometry(lc_f, lp_f, ltype):
    """Bounds and orientation cone of one packed light row, per frame.

    ``lp_f [R, 3]`` / ``lc_f [R, C]`` are the row's packed columns at each
    frame of the chunk. A point row is a degenerate box widened by its
    shadow radius (which is where its visibility ray may actually aim) under
    a full cone; a spot keeps its outer cone; an area cell is the rectangle
    its row stands for, emitting one-sided about the rectangle normal.
    """
    rows = lp_f.shape[0]
    axis = np.zeros((rows, 3))
    axis[:, 2] = 1.0
    theta_o = np.full(rows, math.pi)
    theta_e = np.full(rows, _HALF_PI)
    ext = np.zeros((rows, 3))
    if lc_f.shape[1] > 11:
        ext += np.abs(lc_f[:, 11])[:, None]
    if ltype == _LT_SPOT and lc_f.shape[1] > 9:
        axis = lc_f[:, 6:9].copy()
        theta_o = np.zeros(rows)
        theta_e = np.arccos(np.clip(lc_f[:, 9], -1.0, 1.0))
    elif ltype == _LT_AREA_SAMPLE and lc_f.shape[1] > 14:
        axis = lc_f[:, 6:9].copy()
        theta_o = np.zeros(rows)
        b1 = lc_f[:, 12:15]
        b2 = np.cross(axis, b1)
        ext = np.abs(b1) * lc_f[:, 9:10] + np.abs(b2) * lc_f[:, 10:11]
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = np.where(
        norm > 1e-9, axis / np.maximum(norm, 1e-9), np.array([0.0, 0.0, 1.0])
    )
    # The row's authored falloff exponent (``_light_eval``'s column 4), which
    # defaults to 0: the importance must not assume inverse-square.
    decay = np.zeros(rows)
    if lc_f.shape[1] > 4:
        decay = np.maximum(lc_f[:, 4], 0.0)
    return lp_f - ext, lp_f + ext, axis, theta_o, theta_e, decay


def _light_tree_geometry(
    merged, light_pos, light_col, row_ids, row_types, tri_ids, frames, quad_decay=None
):
    """Per-frame bounds and cones of every finite next-event entry.

    Returns ``(bmin, bmax, axis, theta_o, theta_e, decay)``, each
    ``[R, E, ...]`` with the entries in the next-event table's own order
    (light rows, then emissive triangles) so a tree leaf's entry index
    addresses ``nee_ref`` directly. Everything is gathered per frame the way
    the kernels index these tensors (``f % rows``), which is what makes the
    tree follow a moving light.
    """
    fr = np.asarray(frames, dtype=np.int64)
    rows = int(fr.shape[0])
    blocks = []
    if row_ids.numel():
        lp_all = light_pos.detach().cpu().numpy().astype(np.float64)
        lc_all = light_col.detach().cpu().numpy().astype(np.float64)
        tl = fr % lp_all.shape[0]
        ids = row_ids.cpu().numpy()
        types = row_types.cpu().numpy()
        lp_g = lp_all[tl][:, ids, :3]
        lc_g = lc_all[tl][:, ids, :]
        n_rows = int(ids.shape[0])
        bn = np.zeros((rows, n_rows, 3))
        bx = np.zeros((rows, n_rows, 3))
        ax = np.zeros((rows, n_rows, 3))
        to = np.zeros((rows, n_rows))
        te = np.zeros((rows, n_rows))
        dk = np.zeros((rows, n_rows))
        for j in range(n_rows):
            r_bn, r_bx, r_ax, r_to, r_te, r_dk = _row_tree_geometry(
                lc_g[:, j, :], lp_g[:, j, :], int(types[j])
            )
            bn[:, j], bx[:, j], ax[:, j], to[:, j], te[:, j], dk[:, j] = (
                r_bn,
                r_bx,
                r_ax,
                r_to,
                r_te,
                r_dk,
            )
        blocks.append((bn, bx, ax, to, te, dk))
    if tri_ids.numel():
        tri_pos = merged["tri_pos"]
        pos = tri_pos[:, tri_ids, :].detach().cpu().numpy().astype(np.float64)
        tp = fr % pos.shape[0]
        pos = pos[tp]
        v0 = pos[..., 0:3]
        v1 = pos[..., 3:6]
        v2 = pos[..., 6:9]
        bn = np.minimum(np.minimum(v0, v1), v2)
        bx = np.maximum(np.maximum(v0, v1), v2)
        ng = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(ng, axis=-1, keepdims=True)
        ax = np.where(
            norm > 1e-12, ng / np.maximum(norm, 1e-12), np.array([0.0, 0.0, 1.0])
        )
        tri_mat = merged["tri_mat"]
        one_sided = np.zeros(int(tri_ids.numel()), dtype=bool)
        if int(tri_mat.shape[2]) > _MAT_ONE_SIDED:
            one_sided = tri_mat[0, tri_ids, _MAT_ONE_SIDED].detach().cpu().numpy() > 0.5
        # A two-sided emitter's normals span a hemisphere, which is what
        # theta_o = pi/2 (with theta_e = pi/2) says; a one-sided one has the
        # single geometric normal it is packed with.
        to = np.broadcast_to(
            np.where(one_sided, 0.0, _HALF_PI), (rows, one_sided.shape[0])
        )
        te = np.full((rows, one_sided.shape[0]), _HALF_PI)
        # Inverse square, always: this is the area-measure pdf's Jacobian,
        # not something the author chose.
        dk = np.full((rows, one_sided.shape[0]), 2.0)
        if quad_decay is not None:
            # ...except on a RectAreaLight's synthetic quad, whose emitted
            # radiance carries a ``d^(2 - decay)`` multiplier that cancels
            # that Jacobian back to the row model's authored falloff. The
            # importance must read the NET exponent or it aims the sampler
            # at the near light in a scene where every light contributes
            # equally -- measured 1.34x worse than the flat CDF (section 6a).
            tri_np = tri_ids.detach().cpu().numpy()
            dk = np.array(
                [float(quad_decay.get(int(p), 2.0)) for p in tri_np], dtype=np.float64
            )[None, :].repeat(rows, axis=0)
        blocks.append((bn, bx, ax, to, te, dk))
    return tuple(np.concatenate([b[k] for b in blocks], axis=1) for k in range(6))


def _build_light_tree_tables(
    memory,
    merged,
    light_pos,
    light_col,
    kind,
    ref,
    power,
    ltypes,
    n_tri,
    time_start,
    num_frames,
    enabled,
    quad_decay=None,
):
    """Pack this render call's light trees for the kernel.

    Splits the next-event entries into the **finite** ones a tree can
    discriminate between (point / spot / area-cell rows and emissive
    triangles) and the **infinite** ones it cannot (directional rows, the
    environment entry), builds one tree per distinct frame over the finite
    set, and returns the kernel-facing tensors plus
    ``(tree_on, tree_mix, num_inf)`` for ``nee_meta``.

    ``tree_mix`` is ``P_finite / P_total``, so an infinite entry's effective
    selection probability is ``(1 - tree_mix) * power / P_inf = power /
    P_total`` -- exactly the flat CDF's number. Only the split among the
    finite entries changes, which is why ``_NM_ENV_SHARE`` needs no
    adjustment.
    """
    from algan.rendering.raytracing.tracer import _arena_copy

    device = memory.data.device
    i32 = torch.int32
    f32 = torch.float32
    if enabled:
        is_row = kind == _NEE_LIGHT_ROW
        row_types = torch.zeros_like(kind)
        if ltypes is not None and bool(is_row.any().item()):
            row_types[is_row] = ltypes[ref[is_row]].to(kind.dtype)
        finite = (is_row & (row_types != _LT_DIRECTIONAL)) | (kind == _NEE_EMISSIVE_TRI)
        fin_pos = finite.nonzero(as_tuple=False).flatten()
        inf_pos = (~finite).nonzero(as_tuple=False).flatten()
    else:
        fin_pos = torch.zeros(0, dtype=torch.int64, device=device)
        inf_pos = torch.zeros(0, dtype=torch.int64, device=device)
    n_fin = int(fin_pos.numel())
    n_inf = int(inf_pos.numel())
    p_fin = float(power[fin_pos].sum().item()) if n_fin else 0.0
    p_inf = float(power[inf_pos].sum().item()) if n_inf else 0.0
    total = p_fin + p_inf
    tree_mix = (p_fin / total) if total > 0.0 else 0.0

    if n_fin > 0:
        fin_rows = fin_pos[is_row[fin_pos]]
        fin_tris = fin_pos[kind[fin_pos] == _NEE_EMISSIVE_TRI]
        geo = _light_tree_geometry(
            merged,
            light_pos,
            light_col,
            ref[fin_rows],
            row_types[fin_rows],
            ref[fin_tris],
            range(int(time_start), int(time_start) + int(num_frames)),
            quad_decay,
        )
        # geo[5] is the falloff exponent, which is authored rather than
        # animated, so the tree takes frame 0's row for every frame.
        node_f, node_i, entry_leaf, frame_row = build_light_trees(
            power[fin_pos].detach().cpu().numpy().astype(np.float64),
            *geo[:5],
            geo[5][0],
        )
        # A leaf's right link becomes the GLOBAL entry index, so a descent
        # lands on a row of ``nee_ref`` without a second indirection; the
        # entry -> leaf map stays keyed by the tree's own local index, which
        # is what ``tri_emit_entry`` stores.
        leaves = node_i[:, :, LT_LEFT] < 0
        globals_of_local = fin_pos.detach().cpu().numpy().astype(np.int32)
        right = node_i[:, :, LT_RIGHT]
        right[leaves] = globals_of_local[right[leaves]]
    else:
        node_f = np.zeros((1, 1, LT_F_WIDTH), dtype=np.float32)
        node_i = np.full((1, 1, LT_I_WIDTH), -1, dtype=np.int32)
        entry_leaf = np.zeros((1, 1), dtype=np.int32)
        frame_row = np.zeros(max(int(num_frames), 1), dtype=np.int32)

    with memory.scope(
        "pt_light_tree",
        rows=int(node_f.shape[0]),
        nodes=int(node_f.shape[1]),
        infinite=max(n_inf, 1),
    ):
        lt_node_f = _arena_copy(memory, torch.from_numpy(node_f))
        lt_node_i = _arena_copy(memory, torch.from_numpy(node_i))
        lt_entry_leaf = _arena_copy(memory, torch.from_numpy(entry_leaf))
        lt_frame = _arena_copy(memory, torch.from_numpy(frame_row))
        if n_inf > 0:
            inf_power = power[inf_pos]
            inf_cdf = (inf_power / inf_power.sum()).cumsum(0)
            inf_cdf[-1] = 1.0
            nee_inf_cdf = _arena_copy(memory, inf_cdf.float())
            nee_inf_ref = _arena_copy(memory, inf_pos.to(i32))
        else:
            nee_inf_cdf = memory.get_tensor((1,), f32)
            nee_inf_cdf.zero_()
            nee_inf_ref = memory.get_tensor((1,), i32)
            nee_inf_ref.zero_()
        # Triangle -> its LOCAL tree index (-1 = not in the table), the one
        # extra lookup the MIS pdf query needs at a BSDF hit on an emitter.
        emit_entry = memory.get_tensor((max(n_tri, 1),), i32)
        emit_entry.fill_(-1)
        if n_fin > 0:
            local = torch.arange(n_fin, device=device, dtype=torch.int64)
            tri_local = local[kind[fin_pos] == _NEE_EMISSIVE_TRI]
            if tri_local.numel():
                emit_entry[ref[fin_pos[kind[fin_pos] == _NEE_EMISSIVE_TRI]]] = (
                    tri_local.to(i32)
                )
    return (
        lt_node_f,
        lt_node_i,
        lt_entry_leaf,
        lt_frame,
        nee_inf_cdf,
        nee_inf_ref,
        emit_entry,
        tree_mix,
        n_inf,
    )


def _build_nee_tables(
    memory,
    merged,
    light_pos,
    light_col,
    num_lights,
    env_meta,
    far_clip=0.0,
    time_start=0,
    num_frames=1,
):
    """Build one render call's power-weighted next-event table.

    One flat CDF over everything a shadow ray can aim at -- delta and
    area-cell light rows (ambient-like rows are the kernel's deterministic
    fill and never enter), emissive lit triangles (frame-0 peak luminance
    times area; an emitter dark at frame 0 is simply never NEE-sampled and
    reaches the image through BSDF hits at weight 1, which stays unbiased),
    and one environment entry when a map is present and ``pt_env_nee`` is
    on. Light-row weights take the max over frames so a light dark at frame
    0 but lit later is still sampled (rows have no MIS backstop).

    Under ``pt_light_tree`` (the default) that flat CDF stops being how a
    finite entry is *chosen* -- a light tree over the same entries is
    (``_build_light_tree_tables``, ``light_tree.py``) -- but it is still
    built: it is the ``pt_light_tree = False`` arm, and ``tri_emit_prob``
    remains the "this triangle is in the table" predicate either way.

    Returns arena tensors ``(nee_cdf [E + R], nee_ref [E + A + R, 2], nee_meta
    [NEE_META_WIDTH], tri_emit_prob [N], env_cdf [H, W + 1], tri_emit_entry
    [N], lt_node_f [rows, nodes, 14], lt_node_i [rows, nodes, 3],
    lt_entry_leaf [rows, E_finite], lt_frame [frames], nee_inf_cdf [E_inf],
    nee_inf_ref [E_inf], pt_emit_falloff [Q, 2])`` plus the two host-side
    numbers the launch needs, ``(auth_mode, authored_slots)`` -- every
    selection probability the kernels divide by or MIS against comes from
    these, so both ends of each MIS pair see identical numbers. The ``A``
    ambient-like rows sit AFTER the ``E`` sampled entries and are not part of
    the CDF (which the kernel searches at ``num_nee``): they are the
    deterministic fill's row indexes, packed here so ``pt_shade`` need not
    rescan every light row's type column at every lit crossing.

    The ``R`` rows after those are the AUTHORED-appearance branch's own table
    (roadmap section 6a-bis), present only when that branch samples: the light
    rows an authored stage sums, with a self-normalised power CDF of their own
    in the matching span of ``nee_cdf``. It is a separate table rather than a
    subset of the sampled entries because the two disagree about a
    ``RectAreaLight``: the sampled entries hold its emissive quads and NOT its
    cell rows, while an authored material lights from those cell rows, which
    is the model those materials have. Keeping them apart is also what keeps
    every draw useful -- selecting from the sampled entries would have to
    reject emissive triangles and the environment, which do not light an
    authored surface at all.

    ``pt_emit_falloff`` and ``_NM_QUAD_BASE`` are the area-light quads'
    (``area_light_quads``): the ``Q`` synthetic emissive triangles a
    ``RectAreaLight`` was turned into carry ``(2 - decay, distance)`` each, so
    the kernel can reproduce the row model's falloff at both MIS ends, and the
    cell rows those quads replace are withdrawn from the table here so nothing
    is counted twice. A render with no area light gets a ``[1, 2]``
    placeholder and a base past every primitive index.
    """
    from algan.rendering.raytracing.tracer import _arena_copy, _arena_values

    device = memory.data.device
    i64 = torch.int64
    acc = accumulate_dtype()
    powers = []
    kinds = []
    refs = []
    # Row indexes of the direction-less (ambient / hemisphere) rows, packed
    # onto the table's tail for the kernel's deterministic fill -- see
    # ``_NEE_AMBIENT_ROW``. Ascending, which is the order the linear scan they
    # replace visited them in, and read at frame 0 because a row's type column
    # is frame-invariant (``Light._build_aux`` fills it from the class
    # attribute). The compact packing has no type column at all: every row is
    # a point light there, so there is nothing to find.
    amb_rows = None
    ltypes = None
    auth_idx = None
    auth_power = None
    # The area-light quads this render call added, if any (area_light_quads):
    # the first synthetic primitive index, each quad's ``(2 - decay, range)``
    # falloff pair, and the packed rows they replace.
    quad_base = merged.get("pt_quad_base")
    quad_falloff = merged.get("pt_quad_falloff")
    quad_rows = merged.get("pt_quad_rows")
    if num_lights > 0:
        row_power = light_col[..., :3].amax(0).amax(-1).to(acc)
        if light_col.shape[2] > 3:
            ltypes = (light_col[0, :, 3] + 0.5).to(i64)
            if rt_settings.pt_ambient_rows:
                amb = (ltypes == _LT_AMBIENT) | (ltypes == _LT_HEMISPHERE)
                found = amb.nonzero(as_tuple=False).flatten()
                if found.numel():
                    amb_rows = found
        else:
            ltypes = torch.zeros(num_lights, dtype=i64, device=device)
        sampled = (
            (ltypes == _LT_POINT)
            | (ltypes == _LT_DIRECTIONAL)
            | (ltypes == _LT_SPOT)
            | (ltypes == _LT_AREA_SAMPLE)
        ) & (row_power > 0)
        # The authored-appearance branch's own table, taken BEFORE the quad
        # withdrawal below (roadmap 6a-bis). An authored material is defined as
        # a sum over the packed rows and lights from a ``RectAreaLight``'s cell
        # rows whether or not the path tracer replaced that light with geometry
        # for its OWN estimator -- so the two tables agree on which rows exist
        # and disagree, deliberately, about the area light's.
        auth_idx = sampled.nonzero(as_tuple=False).flatten()
        auth_power = row_power[auth_idx]
        if quad_rows:
            # A RectAreaLight the quad path has turned into geometry is in
            # this table twice otherwise -- once as its K cell rows and once
            # as its two emissive triangles -- and the estimator would count
            # both. The rows stay in ``light_col`` because the
            # authored-appearance branch still lights from them; they simply
            # stop being SELECTABLE here.
            sampled[torch.tensor(quad_rows, dtype=i64, device=sampled.device)] = False
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
        if quad_base is not None and tri_mat.shape[0] > 1:
            # A light row's table weight is its MAX over the chunk's frames,
            # so a light dark at frame 0 but lit later is still sampled (a row
            # has no MIS backstop). The quads inherit that rule for the light
            # they stand for -- everything else keeps frame 0's emission, as
            # emissive meshes always have.
            qb = int(quad_base)
            em_q = (
                tri_mat[:, qb:, 0:3].amax(-1).clamp_min(0)
                * tri_mat[:, qb:, 3].clamp_min(0)
            ).amax(0)
            em = torch.cat((em[:qb], em_q))
        p9 = merged["tri_pos"][0].to(acc)
        area = 0.5 * torch.linalg.cross(
            p9[:, 3:6] - p9[:, 0:3], p9[:, 6:9] - p9[:, 0:3], dim=-1
        ).norm(dim=-1)
        p_e = torch.where(lit, em.to(acc) * area * math.pi, torch.zeros_like(area))
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
            powers.append(torch.tensor([env_power], dtype=acc, device=device))
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

    num_ambient = 0 if amb_rows is None else int(amb_rows.numel())

    # ------------------------------------------------------------------
    # The authored-appearance branch's estimator (roadmap section 6a-bis).
    # ------------------------------------------------------------------
    # ``off``  -- the branch sums every row and traces a shadow ray per row up
    #             to the cap, exactly as it always has.
    # ``auto`` -- that sum while it is affordable, this estimator past the cap.
    # ``always`` -- this estimator at every light count (the A/B arm).
    #
    # The estimator fills ``A`` slots with the direction-less rows and ``S``
    # with rows drawn from ``auth_cdf``, so it needs ``A + S`` visibility slots
    # where the sum needs one per light. ``S`` never exceeds
    # ``pt_light_samples``: the kernel spends the crossing's OWN next-event
    # dimension pairs on these draws (a crossing is either lit or authored,
    # never both), and there are ``2 * pt_light_samples`` of them.
    #
    # Forced back to the sum in three cases, each because the estimator would
    # be a strictly worse spelling of the same answer: no sampleable row at all
    # (nothing to draw from), the ambient rows not packed (``pt_ambient_rows``
    # off -- the deterministic slots come from that tail), and no slot left for
    # a sampled row after the ambient ones.
    num_authored = 0 if auth_idx is None else int(auth_idx.numel())
    auth_choice = str(rt_settings.pt_authored_light_sampling).strip().lower()
    auth_mode = int(
        auth_choice == "always"
        or (auth_choice == "auto" and num_lights > max_shadow_lights)
    )
    if num_authored == 0 or not rt_settings.pt_ambient_rows:
        auth_mode = 0
    auth_want = max(1, int(rt_settings.pt_light_samples))
    auth_amb = min(num_ambient, max_shadow_lights)
    auth_samples = max(0, min(auth_want, max_shadow_lights - auth_amb))
    if auth_samples == 0:
        auth_mode = 0
    if auth_mode == 0:
        num_authored = 0
        auth_samples = 0
        authored_slots = int(num_lights)
    else:
        authored_slots = auth_amb + auth_samples
        if auth_samples < auth_want:
            logger.log(
                PERF,
                "path tracer: authored-appearance materials sample %d of the "
                "%d light rows they asked for (%d of the %d shadow slots go "
                "to ambient rows).",
                auth_samples,
                auth_want,
                auth_amb,
                max_shadow_lights,
            )

    with memory.scope(
        "pt_nee_tables",
        entries=max(num_entries + num_ambient + num_authored, 1),
        emitters=max(n_tri, 1),
    ):
        emit_prob = memory.get_tensor((max(n_tri, 1),), torch.float32)
        emit_prob.zero_()
        if num_entries > 0:
            emissive_rows = kind == _NEE_EMISSIVE_TRI
            if bool(emissive_rows.any().item()):
                emit_prob[ref[emissive_rows]] = prob[emissive_rows].float()
        else:
            kind = torch.zeros(0, dtype=i64, device=device)
            ref = torch.zeros(0, dtype=i64, device=device)
        # The authored branch's CDF is SELF-NORMALISED and occupies the span of
        # ``nee_cdf`` matching its rows' span of ``nee_ref``. Nothing else
        # reads it (the sampled search is bounded by ``num_nee``), and it is
        # built only in the sampled mode, so an ``off`` render's tables are the
        # bytes they always were.
        cdf_parts = [cdf.float()] if num_entries > 0 else []
        if num_authored > 0:
            a_prob = auth_power / auth_power.sum()
            a_cdf = a_prob.cumsum(0)
            a_cdf[-1] = 1.0
            cdf_parts.append(a_cdf.float())
        if cdf_parts:
            nee_cdf = _arena_copy(memory, torch.cat(cdf_parts))
        else:
            nee_cdf = memory.get_tensor((1,), torch.float32)
            nee_cdf.zero_()
        if num_entries > 0 or num_ambient > 0 or num_authored > 0:
            # The ambient rows go AFTER the ``num_entries`` sampled ones: the
            # CDF search is bounded by ``num_nee`` and never sees them, and
            # the kernel walks them at ``nee_ref[num_nee + j]``. The authored
            # rows go after those, at ``nee_ref[num_nee + num_ambient + k]``.
            if num_ambient > 0:
                kind = torch.cat((kind, torch.full_like(amb_rows, _NEE_AMBIENT_ROW)))
                ref = torch.cat((ref, amb_rows))
            if num_authored > 0:
                kind = torch.cat((kind, torch.full_like(auth_idx, _NEE_AUTHORED_ROW)))
                ref = torch.cat((ref, auth_idx))
            nee_ref = _arena_copy(
                memory,
                torch.stack((kind, ref), -1).to(torch.int32),
            )
        else:
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
        # The far plane rides the meta vector rather than a new kernel
        # argument: pt_shade is close to Taichi's 64-argument ceiling, and a
        # per-render scalar is exactly what this vector is for.
        meta[_NM_FAR_CLIP] = float(max(0.0, far_clip))
        # ``pt_ambient_rows`` off packs nothing and the kernel keeps its
        # linear scan; the count alone cannot say so (0 packed rows is a real
        # answer -- a scene with no ambient light), hence the separate word.
        meta[_NM_AMBIENT_PACKED] = 1.0 if rt_settings.pt_ambient_rows else 0.0
        meta[_NM_AMBIENT_COUNT] = float(num_ambient)
        meta[_NM_ANIM_SEED] = 1.0 if rt_settings.pt_animated_seed else 0.0
        # Where the synthetic area-light quads start. One compare in the
        # drain loop turns into both the camera-invisibility test and the
        # gate on the falloff multiplier; ``NO_QUAD_BASE`` is past any
        # primitive index a batch can hold, so a render with no area light
        # takes neither branch and is bit-identical.
        meta[_NM_QUAD_BASE] = float(
            NO_QUAD_BASE if quad_base is None else int(quad_base)
        )
        # Both zero unless the authored branch samples (roadmap 6a-bis); the
        # kernel reads them only in the arm compiled for that, and the arm is
        # chosen by ``auth_sampled``, which the caller derives from the same
        # decision that set these.
        meta[_NM_AUTHORED_SAMPLES] = float(auth_samples)
        meta[_NM_AUTHORED_COUNT] = float(num_authored)

    with memory.scope(
        "pt_quad_falloff",
        quads=1 if quad_falloff is None else int(quad_falloff.shape[0]),
    ):
        if quad_falloff is None:
            emit_falloff = memory.get_tensor((1, 2), torch.float32)
            emit_falloff.zero_()
        else:
            emit_falloff = _arena_copy(memory, quad_falloff.float().contiguous())

    # The light tree over the SAMPLED entries only: the ambient rows on
    # nee_ref's tail are the deterministic fill and were never selected.
    tree_on = bool(rt_settings.pt_light_tree) and num_entries > 0
    (
        lt_node_f,
        lt_node_i,
        lt_entry_leaf,
        lt_frame,
        nee_inf_cdf,
        nee_inf_ref,
        emit_entry,
        tree_mix,
        n_inf,
    ) = _build_light_tree_tables(
        memory,
        merged,
        light_pos,
        light_col,
        kind[:num_entries] if num_entries > 0 else kind,
        ref[:num_entries] if num_entries > 0 else ref,
        power if num_entries > 0 else torch.zeros(0, dtype=acc, device=device),
        ltypes,
        n_tri,
        time_start,
        num_frames,
        tree_on,
        None
        if quad_base is None or quad_falloff is None
        else {
            int(quad_base) + j: 2.0 - float(quad_falloff[j, 0].item())
            for j in range(int(quad_falloff.shape[0]))
        },
    )
    with memory.scope("pt_nee_meta"):
        meta[_NM_TREE_ON] = 1.0 if tree_on else 0.0
        meta[_NM_TREE_MIX] = float(tree_mix)
        meta[_NM_INF_COUNT] = float(n_inf)
        nee_meta = _arena_values(memory, meta, torch.float32)
    return (
        nee_cdf,
        nee_ref,
        nee_meta,
        emit_prob,
        env_cdf,
        emit_entry,
        lt_node_f,
        lt_node_i,
        lt_entry_leaf,
        lt_frame,
        nee_inf_cdf,
        nee_inf_ref,
        emit_falloff,
        auth_mode,
        authored_slots,
    )


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
    frag_scatters=(),
    shadows,
    max_bounces,
    near_clip,
    far_clip,
    transparent,
    samples,
    env_meta=None,
    aovs=None,
    out,
    accum,
    accum_odd=None,
):
    """Path trace frames ``[time_start, time_end)`` into ``accum``.

    ``out`` must hold the prefilled background (byte scale) and ``accum`` the
    zeroed ``[frames, pixels, 5]`` sample sums; the caller averages them with
    ``finalize_samples``. ``near_clip`` / ``far_clip`` are the camera's clip
    distances in world units (0 = no plane), applied with the deterministic
    renderer's semantics: the near plane advances the primary origin and
    seeds ``base_dist``, and the far plane retires a path once
    ``base_dist + t`` passes it. Raises the arena's memory exceptions with all tile
    state released, so the chunk-halving retry in ``render_chunk`` works
    unchanged.

    ``frag_scatters`` is the per-pipeline custom ray-continuation tuple, the
    same one the deterministic wavefront takes (empty when no pipeline in
    this batch overrides bouncing). A crossing whose pipeline supplies one
    commits the radiance the scatter returns and continues along one of its
    three branches as a delta lobe -- see ``pt_shade``. Empty compiles the
    kernel exactly as it compiled before custom scatter reached here.

    ``aovs``, when given, is ``(albedo, normal, bg_weight)`` -- three zeroed
    ``[frames, pixels, 3]`` float32 tensors the call fills with per-pixel
    SAMPLE SUMS of the denoiser guides (see ``pt_aov`` in
    ``path_tracer_taichi``): the caller divides by ``samples`` and folds
    ``bg_weight`` with its own background colors (the kernel does not know
    them). ``None`` skips all AOV work.

    ``accum_odd`` is the adaptive sampler's stopping-rule buffer -- a zeroed
    ``[frames, pixels, 4]`` float32 tensor the caller allocates exactly when
    :func:`pt_adaptive_active` says so, into which ``pt_reduce`` sums the RGB
    of the ODD sample indices (columns 0-2) and the count of samples whose
    path took a random decision (column 3). With it, ``samples`` is a
    CEILING: every pixel gets ``pt_min_samples``, and after that a pixel stops
    only if none of its samples was stochastic AND its two halves agree to
    within ``pt_error_target``.
    The per-pixel sums are rescaled to ``samples`` before returning, so the
    caller's one scalar division (``finalize_samples``, and the AOVs' own
    ``1 / samples``) still yields per-pixel means. ``None`` -- which is what
    ``pt_error_target = 0`` produces -- is the uniform loop, byte for byte as
    it was before adaptive sampling existed.
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
    animated_seed = 1 if rt_settings.pt_animated_seed else 0
    bvh_refit = 1 if isinstance(tri_bvh, RefitBVH) else 0
    # Compile-time shadow mode of every visibility ray pt_shade spawns, the
    # deterministic renderer's decision (tracer.py) reduced to the two modes
    # worth having here: 3 = opaque any-hit (the ordered march compiled out)
    # when the batch provably holds nothing a shadow ray could pass partly
    # through, else 1 = the ordered march. Mode 2, the deferred any-hit for
    # mixed batches, is left out on purpose -- it measured as a loss on the
    # deterministic renderer. A transmissive surface is alpha 1 (so it never
    # reads as translucent) and passes light rather than blocking it, and an
    # uncertain texture alpha is attenuation only the march can evaluate;
    # both keep the march. With shadows off the query is compiled out
    # entirely, so the mode is pinned at 1 to avoid a second kernel variant.
    shadow_mode = 1
    if int(shadows) and rt_settings.pt_shadow_anyhit:
        provably_opaque = not (
            merged.get("has_transmissive", True)
            or merged.get("tri_has_translucent", True)
            or merged.get("bez_has_translucent", True)
            or merged.get("has_uncertain_texture_alpha", True)
        )
        if provably_opaque:
            shadow_mode = 3
    # Closest-hit traversal: with every visible primitive opaque, a path's
    # peel ends at its first crossing (there is no pass-through), so the
    # k-buffer's remaining slots are filled and drained for nothing. The
    # shared traverse kernel's ``opaque_closest`` walks the MAIN trees --
    # ``merged["opaque_bvh_skipped"]`` (the merge aliasing the dedicated
    # opaque-only trees, which is the default while no deterministic rollout
    # walks them) gates ``opaque_prepass`` only, and the path tracer never
    # enables that. pt_shade's completion test needs no change: a gather of
    # at most one hit still satisfies ``num_hits < kbuf``, so a path that
    # neither scattered nor retired is finished exactly as the deterministic
    # ``wavefront_shade`` finishes it under the same template.
    opaque_closest = int(
        rt_settings.pt_opaque_closest and merged.get("all_visible_opaque", False)
    )

    with memory.scope("pt_metadata"):
        # The traverse kernel rebuilds each pixel's primary ray from
        # gen_meta[2:] to convert slant ranges to perpendicular depth (see
        # wavefront_traverse_events); the jitter pair is unused (gen_first=0).
        gen_meta = _arena_values(
            memory, [0.5, 0.5, float(half_screen_w), float(half_screen_h)], f32
        )
    # The power-weighted next-event table + environment CDF for this call
    # (before the tile budget is taken, so their bytes are accounted).
    # Timed and logged at PERF: this is host work per chunk (the light tree
    # build, the per-frame emitter geometry, the arena uploads) and it is
    # what a GPU profile cannot see -- a T4 session attributed 430 ms of a
    # 2.1 s many-light render to the host before the tree build was memoized.
    setup_t0 = time.perf_counter()
    (
        nee_cdf,
        nee_ref,
        nee_meta,
        tri_emit_prob,
        env_cdf,
        tri_emit_entry,
        lt_node_f,
        lt_node_i,
        lt_entry_leaf,
        lt_frame,
        nee_inf_cdf,
        nee_inf_ref,
        pt_emit_falloff,
        auth_sampled,
        authored_slots,
    ) = _build_nee_tables(
        memory,
        merged,
        light_pos,
        light_col,
        int(num_lights),
        env_meta,
        far_clip,
        int(time_start),
        num_frames,
    )
    tri_shell = _build_shell_table(memory, merged)
    logger.log(
        PERF,
        "path tracer: next-event setup %.1f ms (%d entries, %d tree nodes, %d frames)",
        1000.0 * (time.perf_counter() - setup_t0),
        int(nee_meta[_NM_COUNT].item()),
        int(lt_node_f.shape[1]) if lt_node_f.dim() == 3 else 0,
        num_frames,
    )
    if aovs is not None:
        nee_meta[_NM_AOV] = 1.0
        aov_albedo_flat = aovs[0].view(-1, 3)
        aov_normal_flat = aovs[1].view(-1, 3)
        aov_bgw_flat = aovs[2].view(-1, 3)

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

    # Adaptive sampling (roadmap section 2). ``accum_odd`` is the caller's
    # signal: allocated exactly when the mechanism is on. ``n_p`` counts what
    # each pixel actually received -- host-side torch, because it is read once
    # per wave to build the next pixel list and once at the end to rescale.
    adaptive = accum_odd is not None
    error_target = float(rt_settings.pt_error_target)
    floor_samples = _pt_floor_samples(samples) if adaptive else samples
    n_p = torch.zeros(n, dtype=i32, device=device) if adaptive else None
    if accum_odd is None:
        # pt_reduce takes the argument either way; with ``adaptive == 0`` it
        # never indexes it, so a one-cell dummy keeps the launch site single.
        accum_odd = torch.zeros((1, 1, 4), dtype=f32, device=device)

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
                pt_aov = memory.get_tensor(
                    (pool if aovs is not None else 1, PT_AOV_WIDTH), f32
                )
                pt_stats = memory.get_tensor((PT_STATS_WIDTH,), i32)
            compactor = _ArenaRayCompactor(memory, pool, i32)
            pt_stats.zero_()

            # The wave's pixel list: the tile's identity list to start with,
            # compacted to the unconverged pixels once the floor is in.
            # ``keep_buf`` is the compaction's dense scratch (see
            # ``_pt_active_pixels``), all-False between calls.
            pix_list = torch.arange(
                tile_start, tile_start + tp, dtype=i32, device=device
            )
            keep_buf = (
                torch.zeros(tp, dtype=torch.bool, device=device) if adaptive else None
            )
            active_pixels = tp
            sample_base = 0
            while sample_base < samples and active_pixels > 0:
                remaining = samples - sample_base
                if adaptive:
                    sw = _pt_wave_size(
                        pool, active_pixels, sample_base, floor_samples, remaining
                    )
                else:
                    sw = min(wave_samples, remaining)
                slots = active_pixels * sw
                rs_sca[:slots].copy_(sca_init)
                rs_int[:slots].copy_(int_init)
                pt_thru[:slots].fill_(1.0)
                pt_acc[:slots].zero_()
                if aovs is not None:
                    pt_aov[:slots].zero_()
                pt_generate(
                    int(slots),
                    int(active_pixels),
                    int(sample_base),
                    seed_root,
                    animated_seed,
                    int(time_start),
                    int(width),
                    int(height),
                    float(half_screen_w),
                    float(half_screen_h),
                    cam_origin,
                    screen_point,
                    pixel_basis_x,
                    pixel_basis_y,
                    float(near_clip),
                    pix_list,
                    rs_ro,
                    rs_rd,
                    rs_sca,
                    rs_pix,
                )
                active = compactor.initial(slots)
                it = 0
                max_iters = max_surfaces_per_ray + 4
                while active.numel() > 0 and it < max_iters:
                    na = int(active.numel())
                    with memory.temp():
                        # [kbuf, channel, num_active]: the ray ordinal is LAST so the
                        # traverse kernel's stores and shade's gathers coalesce.
                        hit_f = memory.get_tensor((kbuf, 4, na), f32)
                        hit_i = memory.get_tensor((kbuf, 2, na), i32)
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
                            opaque_closest,
                            0,  # opaque_prepass: deterministic-only rollout
                            int(time_start),
                            int(width),
                            int(height),
                            # ray_offset: rs_pix holds the GLOBAL flat cell
                            # (pt_generate writes it through the pixel list),
                            # so there is nothing to add. The kernel decodes
                            # ``ray_offset + rs_pix[r]`` on the gen_first == 0
                            # path the path tracer always takes, and reads
                            # rs_pix for nothing else.
                            0,
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
                            shadow_mode,
                            # The visibility payload is sized by what the
                            # authored branch will actually fill: one slot per
                            # light when it sums them, and the far smaller
                            # ``ambient + sampled`` when it samples.
                            shadow_vis_slots(authored_slots),
                            int(auth_sampled),
                            frag_pipelines,
                            frag_scatters,
                            ALL_PIDS,
                            seed_root,
                            int(sample_base),
                            int(active_pixels),
                            rr_start,
                            firefly_clamp,
                            int(time_start),
                            int(width),
                            int(height),
                            0,  # ray_offset: rs_pix is already global
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
                            pt_aov,
                            tri_emit_entry,
                            lt_node_f,
                            lt_node_i,
                            lt_entry_leaf,
                            lt_frame,
                            nee_inf_cdf,
                            nee_inf_ref,
                            pt_emit_falloff,
                        )
                    active = compactor.select(rs_int, 0, source=active)
                    it += 1
                if aovs is not None:
                    # Fold this wave's per-path guide sums into the chunk's
                    # per-pixel sums. Slot r = wave_sample * active_pixels +
                    # wave pixel (pt_generate/pt_reduce's layout), so a view +
                    # sum is the whole reduction: one tensor op per wave,
                    # scattered through the wave's pixel list.
                    wave_sums = (
                        pt_aov[:slots].view(sw, active_pixels, PT_AOV_WIDTH).sum(0)
                    )
                    seg = pix_list.long()
                    aov_albedo_flat.index_add_(0, seg, wave_sums[:, 0:3])
                    aov_normal_flat.index_add_(0, seg, wave_sums[:, 3:6])
                    aov_bgw_flat.index_add_(0, seg, wave_sums[:, 6:9])
                pt_reduce(
                    int(sample_base),
                    int(active_pixels),
                    int(sw),
                    1 if transparent else 0,
                    1 if adaptive else 0,
                    int(width),
                    int(height),
                    pix_list,
                    out,
                    pt_acc,
                    accum,
                    accum_odd,
                )
                sample_base += sw
                if adaptive:
                    n_p[pix_list.long()] += sw
                    # Re-decide only on an even total: the estimator's two
                    # halves are only balanced there, and every pixel still
                    # alive has received exactly ``sample_base`` samples --
                    # which is what keeps each one's sampler prefix the
                    # contiguous 0..n_p the Sobol sequence is stratified over.
                    if (
                        sample_base < samples
                        and sample_base >= floor_samples
                        and sample_base % 2 == 0
                    ):
                        pix_list = _pt_active_pixels(
                            accum,
                            accum_odd,
                            pix_list,
                            sample_base,
                            error_target,
                            keep_buf,
                            tile_start,
                            width,
                        )
                        active_pixels = int(pix_list.numel())

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

    if not adaptive:
        # Uniform: every pixel got the ceiling, and nothing is rescaled --
        # this arm must stay byte-identical to the pre-adaptive renderer.
        record_path_samples(float(samples) * n, n)
        _log_sample_spread(None, samples, n)
        return
    # Turn the per-pixel sums into "as if every pixel had ``samples`` samples",
    # so the caller's single scalar divisions (finalize_samples, and the AOVs'
    # own 1 / samples in tracer.render_chunk) still produce per-pixel means.
    # All five columns: alpha rides the same wave count.
    scale = (float(samples) / n_p.to(f32).clamp_min(1.0)).unsqueeze(-1)
    accum.view(-1, accum.shape[-1]).mul_(scale)
    if aovs is not None:
        aov_albedo_flat.mul_(scale)
        aov_normal_flat.mul_(scale)
        aov_bgw_flat.mul_(scale)
    record_path_samples(float(n_p.sum().item()), n)
    _log_sample_spread(n_p, samples, n, floor=floor_samples)
