"""Ray-traced render orchestration: renderer dispatch and the deterministic
wavefront's host-side tile / iteration loop.

:func:`render_batch_raytraced` is the entry point called by the render loop
for a batch of frames. It obtains the merged per-geometry-type arrays +
STBVHs (``scene_builder``), prepares camera / light / environment tensors in
the render arena, and dispatches on the sample count:

* ``samples_per_pixel == 1`` (default) -- the deterministic *wavefront*
  tracer (:func:`raytrace_render_wavefront`): bounded ray tiles run
  generate -> traverse -> shade -> composite kernel stages
  (``wavefront_kernels_taichi.py``), with per-ray state pool-allocated from
  ``ManualMemory``, Taichi-side compaction of the still-active rays between
  host iterations, and a shared continuation pool for reflective /
  refractive splits (an overflowing tile is discarded and retried with fewer
  primaries, never approximated).
* ``samples_per_pixel > 1`` -- the path tracer (``path_tracer.py`` +
  ``path_tracer_taichi.py``): the same wavefront shape, sharing this
  renderer's traversal kernel outright, run in sample waves of one path per
  pixel at output resolution (jittered samples are the anti-aliasing) into a
  float32 per-pixel buffer that ``finalize_samples`` averages.

Reflections and refraction are inferred from the mob's Three.js-style
material properties. Use ``MeshStandardMaterial(metalness=..., roughness=...)``
or ``MeshPhysicalMaterial`` before spawning; rays bounce up to
``max_bounces`` times.

On out-of-memory the frame window is halved and retried
(``OutOfRenderMemory``); see ``render_batch_raytraced``.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from functools import wraps
from typing import Literal

import torch

from algan.environment import env_float
from algan.rendering import fragment_capture
from algan.rendering.mps_compat import clamp_floor
from algan.rendering.post_processing.post_process import post_process_frames
from algan.rendering.primitives.primitive import OutOfRenderMemory
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    finalize_samples,
    kbuf,
    max_surfaces_per_ray,
)
from algan.rendering.raytracing.render_metadata import (
    GLOSS_BASE,
    allocate_render_metadata,
)
from algan.rendering.raytracing.scene_builder import (
    _downsample_background,
    _merge_scene,
    _pack_lights,
    _prefill_background,
    copy_merged_scene_to_arena,
)

# NOTE: only immutable settings values may be imported by value here; the
# mutable module globals (samples_per_pixel, TONEMAP_*, shadows, ...) must be
# read live as ``rt_settings.X`` or their setters silently stop working.
from algan.rendering.raytracing.settings import (
    _get_tonemap_t_val,
    _scene_has_user_pipeline,
    is_post_process_tonemap_enabled,
)
from algan.rendering.raytracing.truncation import (
    TruncationCounts,
    attach_render_stats,
    attach_truncations,
    record_truncation,
    report_truncations,
    restore_path_samples,
    restore_truncations,
    snapshot_path_samples,
    snapshot_truncations,
)
from algan.rendering.taichi_runtime import (
    _set_compile_notice_callback,
)
from algan.settings import SETTINGS

rt_settings = SETTINGS.raytracing
from algan.rendering.raytracing.shading_taichi import (
    _USER_PIPELINE_BASE,
    ALL_PIDS,
    max_shadow_lights,
    shadow_vis_slots,
)

# Diagnostics: bumped each time the wavefront engages the Family A+B memory-trim
# path (used by benchmarks/_wf_mem_trim_ab.py to confirm the trim actually fired).
_MEM_TRIM_ENGAGED = [0]
# Number of tile attempts discarded and retried after the shared continuation
# allocator reported overflow. Kept as a list for low-overhead in-process tests.
_WAVEFRONT_POOL_RETRIES = [0]
# Diagnostics: the material-pipeline gate masks of the most recently rendered
# batch (ALL_PIDS = ungated), so an A/B can confirm the gate actually engaged
# instead of inferring it from timings (benchmarks/_frag_pid_gate_ab.py).
_FRAG_PID_LAST = {"tri": ALL_PIDS, "pn": ALL_PIDS}
from algan.logging.logger import PERF, get_logger
from algan.rendering.raytracing.batch_policy import (
    BatchExecutionPolicy,
    WavefrontPolicy,
)

# ``build_frag_pipelines`` is imported lazily in the render dispatch to avoid a
# module-load import cycle (fragment_shaders -> shading_taichi -> raytracing
# package __init__ -> primitives).
from algan.rendering.raytracing.glossy_prefilter_taichi import (
    _BOX_SIGMA,
    GL_MAIN_SIGMA,
    GL_MAIN_WIDTH,
    GL_PYR_WIDTH,
    GL_ROW_DIST,
    GL_ROW_WIDTH,
    gloss_composite,
    gloss_pyramid_level,
    gloss_scatter,
)
from algan.rendering.raytracing.scene_bounds import triangle_scene_bounds
from algan.rendering.raytracing.shadow_queue import (
    ShadowTraceContext,
    _gather_shadow_payload,
    _scatter_shadow_visibility,
)
from algan.rendering.raytracing.utils import _expand_frames, _flat_frames, _pixel_bases
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _SCA_IOR_DEPTH,
    ALLOC_NEXT,
    ALLOC_OVERFLOW,
    ALLOC_TRUNC_SURFACES,
    ALLOC_WIDTH,
    SCA_WIDTH_PLAIN,
    compact_ray_slots,
    reorder_ray_slots,
    sca_width,
    wavefront_generate_rays,
    wavefront_ray_sort_keys,
    wavefront_shade,
    wavefront_shadow_events,
    wavefront_traverse_events,
    wf_composite_accum,
    wf_composite_accum_aa,
    wf_composite_accum_sparse,
    wf_finalize_aa,
    wf_finalize_uncovered,
)
from algan.rendering.raytracing.wavefront_state import RayState
from algan.utils.memory_utils import (
    InsufficientMemoryException,
    ensure_render_headroom,
    is_cuda_oom,
    release_torch_memory,
)

logger = get_logger("raytracing")


@dataclass(frozen=True)
class RenderPlan:
    """Resolved renderer route and capability assessment for one batch.

    The plan is attached to ``scene.last_render_plan`` and returned through
    :class:`algan.utils.algan_utils.RenderResult` after a file render.  It is
    intentionally data-only so applications can log or serialize it without
    depending on Taichi/Torch implementation objects.
    """

    backend: Literal["deterministic_wavefront", "path_tracer"]
    samples_per_pixel: int
    requested_features: tuple[str, ...]
    unsupported_features: tuple[str, ...] = ()
    #: How often each of the renderer's fixed ceilings bound, cumulative over
    #: the render job (see
    #: :mod:`algan.rendering.raytracing.truncation`). Zero everywhere means
    #: nothing was silently dropped -- the counters are unconditional, so a
    #: zero is a measurement rather than a missing instrument. Only the plan
    #: of a *finished* batch carries counts; the one built during validation
    #: is necessarily empty.
    truncations: TruncationCounts = TruncationCounts()
    #: Samples per pixel the path tracer actually took, averaged over every
    #: path-traced cell of the render job. With adaptive sampling on
    #: (``pt_error_target > 0``) ``samples_per_pixel`` is only the ceiling and
    #: this is the measurement: converged pixels -- every unlit 2-D pixel --
    #: stop at ``pt_min_samples``. **Zero means the path tracer did not run**,
    #: so it is 0.0 on every deterministic render.
    path_samples_mean: float = 0.0
    primary_route: str | None = None
    effective_anti_alias_level: int = 1
    fallback_reasons: tuple[str, ...] = ()

    @property
    def is_supported(self) -> bool:
        return not self.unsupported_features

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "samples_per_pixel": self.samples_per_pixel,
            "requested_features": list(self.requested_features),
            "unsupported_features": list(self.unsupported_features),
            "truncations": self.truncations.as_dict(),
            "path_samples_mean": self.path_samples_mean,
            "primary_route": self.primary_route,
            "effective_anti_alias_level": self.effective_anti_alias_level,
            "fallback_reasons": list(self.fallback_reasons),
        }


def _host_tensor(value):
    """Return a detached CPU tensor for render-input preparation.

    Camera/environment/light arithmetic is preparation work.  Keeping it on
    the host prevents a caller-provided render-device tensor from turning a
    subtraction, norm, stack, or cast into an allocation beside the arena.
    """
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.as_tensor(value, device="cpu")


def _arena_copy(memory, tensor, dtype=None):
    """Copy ``tensor`` into the render arena, optionally casting on copy.

    The source may live on the animation device (normally CPU); ``copy_``
    performs the transfer directly into the reserved byte buffer instead of
    creating a transient ``tensor.to(render_device)`` allocation.
    """
    dtype = tensor.dtype if dtype is None else dtype
    out = memory.get_tensor(tensor.shape, dtype)
    out.copy_(tensor)
    return out


def _arena_values(memory, values, dtype=torch.float32):
    source = torch.tensor(values, dtype=dtype, device="cpu")
    return _arena_copy(memory, source)


def _alloc_wavefront_state(memory, tn, sca_width, *, global_hits=True) -> RayState:
    """Allocate the wavefront's per-ray global state from the render memory pool
    (a bump allocator) rather than fresh ``torch.empty`` tensors.

    The caller snapshots ``memory.get_pointers()`` before each tile and restores
    them after, so this ~hundreds-of-MB of state is released *deterministically*
    at the end of every tile and the next tile reuses the same arena bytes.
    Previously these were ``torch.empty`` allocations that the CUDA caching
    allocator / Python GC didn't reclaim before the next tile asked for its own,
    so consecutive tiles' state piled up and OOMed the GPU at HD/AA>=2.
    """
    f32 = torch.float32
    i32 = torch.int32
    core = (
        memory.get_tensor((tn, 3), f32),  # rs_ro
        memory.get_tensor((tn, 3), f32),  # rs_rd
        memory.get_tensor((tn, 4), f32),  # rs_acc
        # rs_sca: 0 weight red, 1 t_prev, 2 layer_prev, 3 seam_t, 4 base_dist,
        # 5 weight green, 6 weight blue (color transport).
        memory.get_tensor((tn, sca_width), f32),  # rs_sca (7 general)
        # rs_int: bounces_left, processed, status, num_hits, accumulator_row.
        # Column 4 is live on the sparse route; dense kernels use columns 0-3.
        memory.get_tensor((tn, 5), i32),  # rs_int
    )
    if global_hits:
        return RayState(
            *(
                core
                + (
                    memory.get_tensor((tn, kbuf), f32),  # rs_kt
                    memory.get_tensor((tn, kbuf), f32),  # rs_kl
                    memory.get_tensor((tn, kbuf), f32),  # rs_ka
                    memory.get_tensor((tn, kbuf), f32),  # rs_kb
                    memory.get_tensor((tn, kbuf), i32),  # rs_kp
                    memory.get_tensor((tn, kbuf), i32),  # rs_kf
                )
            )
        )

    # The supported general renderer no longer attaches a K-buffer to every
    # continuation-pool slot. Keep six tiny placeholders solely so the state
    # tuple remains ABI-compatible with the hybrid raster frontend and the
    # unsupported legacy orchestrators. The general traverse/shade pair uses an
    # exact-size transient surface-event batch allocated for the current active
    # queue instead.
    stub_f = memory.get_tensor((1, 1), f32)
    stub_i = memory.get_tensor((1, 1), i32)
    return RayState(*(core + (stub_f, stub_f, stub_f, stub_f, stub_i, stub_i)))


def _gloss_pyramid_levels(width, height, max_levels):
    """Offsets and dimensions of the reflection pyramid's levels.

    Returns ``([(offset, w, h), ...], total_texels)``. Halving stops at 1x1 or
    at ``max_levels``, whichever comes first; the top level is the whole
    frame's glossy pixels averaged into one texel, which is what a lobe wide
    enough to reflect everything in view should read.
    """
    levels = []
    w, h, off = max(1, int(width)), max(1, int(height)), 0
    while True:
        levels.append((off, w, h))
        off += w * h
        if (w == 1 and h == 1) or len(levels) >= max(1, int(max_levels)):
            break
        # CEILING, not floor. A floor-halved odd dimension leaves its last
        # row or column with no destination texel at all, and the reduction's
        # clamp cannot reach it -- 3 columns reduce to 1, whose 2x2 window
        # covers columns 0 and 1 and drops column 2 entirely. Harmless-looking
        # in the middle of the chain and fatal at the top of it: measured on
        # calib_glossy, the 3x2 level held the whole reflection in its last
        # column and the 1x1 above it came out pure black with a validity
        # weight of 1, so every wide-lobe pixel prefiltered to nothing.
        w = max(1, -(-w // 2))
        h = max(1, -(-h // 2))
    return levels, off


def _deferred_shadow_sample_count(samples, light_columns):
    # Deferred events expose only sample zero and have zero footprints.
    # Compact RGB light rows prove every emitter is a hard point light.
    # Extended/soft lights must keep their original masked emitter fan.
    if rt_settings.shadow_deferred_single_sample and light_columns == 3:
        return 1
    return samples


def _ray_sort_key_format(frame_count, device):
    # Sparse slots retain indices into the full frame window, not the ray
    # pool. Bound frame bits from that window even when coverage is tiny.
    if rt_settings.wf_ray_sort_compact and device.type == "cuda" and frame_count <= 16:
        return torch.int32, 2
    return torch.int64, 0


def _gloss_frame_bounds(covered_idx, pixels_per_frame, num_frames):
    """Where each frame's covered pixels start and end in ordinal space.

    ``covered_idx`` is ascending global pixel indices over the whole window,
    and a global index is ``frame * pixels_per_frame + pixel``, so a frame owns
    a contiguous ordinal range and one searchsorted finds every boundary. The
    tile loop clamps to these so a per-frame reflection buffer is complete
    before it is prefiltered.
    """
    edges = torch.arange(
        num_frames + 1, device=covered_idx.device, dtype=covered_idx.dtype
    ) * int(pixels_per_frame)
    return torch.searchsorted(covered_idx, edges).tolist()


def _gloss_clear(gl_main, gl_pyr):
    """Reset the per-frame reflection buffers.

    The blur radius column is initialised NEGATIVE, not zero: it is what marks
    a pixel as having a prefiltered glossy branch at all, and zero is a legal
    radius (a reflection in contact with its reflector).
    """
    gl_main.zero_()
    gl_main[:, GL_MAIN_SIGMA] = -1.0
    gl_pyr.zero_()


def _gloss_finish_frame(
    frame_rel, gl_levels, gl_main, gl_pyr, width, height, tonemapping, out
):
    """Prefilter one frame's reflection buffer and composite it.

    The pyramid is built bottom-up (each level from the one below), then every
    glossy pixel fetches it trilinearly at the level matching its own blur
    radius and overwrites the value the tile composite wrote for it.
    """
    num_levels = int(gl_levels.shape[0])
    for lvl in range(1, num_levels):
        gloss_pyramid_level(lvl - 1, lvl, gl_levels, gl_pyr)
    gloss_composite(
        int(width) * int(height),
        int(width),
        int(height),
        num_levels,
        int(frame_rel),
        int(tonemapping),
        float(rt_settings.tonemap_exposure),
        gl_levels,
        gl_main,
        gl_pyr,
        out,
    )


def _secondary_split_needed(merged, analytic_raster=False):
    """Does analytic AA make this scene's reflectors a SPLITTING path?

    Two independent reasons, both of which need the shared continuation pool that
    only a "splitting" batch gets:

    1. ANALYTIC COVERAGE ITSELF. A reflector's silhouette pixel is only partly
       covered, so its fragment's alpha is partial -- and the resolve sends a
       reflection into the pixel's own ray slot only when the reflected energy
       DOMINATES the pass-through (``refl_max >= cover_pass``). At a silhouette
       it does not, so without the split path compiled in the reflection is
       dropped outright: a dark rim around every mirror, and the more so the
       better the coverage. Splitting is the correct answer there -- the
       reflection goes to a pool slot and the pass-through continues -- and it is
       the same thing a semi-transparent reflector already does.
    2. CONTINUATION-RAY SUPERSAMPLING (``analytic_aa_secondary_samples > 1``),
       which needs N-1 spare slots for every reflective primary at once.

    A plain opaque mirror was never a splitting path before, so such a scene got
    ``pool_ratio == 1`` -- no spare slots at all -- and every attempted
    reservation would fail. Worse, at ratio 1 the host IGNORES the pool's
    overflow flag, so those failures are silent. Joining the existing split flag
    rather than inventing a second notion of splitting also gets the compaction
    and gen-fused decisions right, since both already treat splitting as a
    property of the batch. The refraction-only kernel branches this compiles in
    stay runtime-inert on a mirror-only scene.
    """
    if not analytic_raster:
        return False
    reflective = bool(
        merged.get("tri_has_reflective")
        or merged.get("bez_has_reflective")
        or merged.get("tex_has_reflective")
        or merged.get("has_refl_transparent")
        or merged.get("has_refractive")
    )
    if not reflective:
        return False
    return bool(
        rt_settings.analytic_aa_tri_active()
        or rt_settings.analytic_aa_bez_active()
        or int(rt_settings.effective_analytic_aa_secondary_samples()) > 1
        # 3. THE SPLIT-SUM GLOSSY ROUTE, which always sends its reflection to a
        #    pool slot (it accumulates into a different row than the pixel's
        #    own, so it cannot continue in the primary's slot). One spare slot
        #    per primary is enough and ``_split_pool_ratio``'s weakest arm
        #    already allocates two -- but only if this says the batch splits.
        #    At ratio 1 the host ignores the pool's overflow flag, so getting
        #    this wrong would drop every glossy reflection in silence.
        or int(rt_settings.glossy_reflection_mode()) == 3
    )


def _split_pool_ratio(splitting, merged, analytic_raster=False, custom_scatter=False):
    """Spare pool slots per primary for a splitting batch.

    Physical glass retains the measured ``base * N`` allowance because its
    front/back layers can split concurrently. An opaque analytic mirror needs
    only N sampled reflections plus, at a partially covered silhouette, one
    pass-through slot; allocating ``N + 1`` instead of the glass path's ``2N``
    admits larger primary tiles without weakening the overflow retry. A weak
    dielectric sheen emits only one reflection and needs two total slots.

    **This is an ESTIMATE of the average, not a bound, and the overflow retry
    is load-bearing.** ``N + 1`` is exact for ONE reflective fragment per
    pixel, which is what a non-analytic raster pixel has; analytic triangle
    coverage puts SEVERAL partially covering fragments of the same mesh in one
    pixel by construction and each of them splits. Their masks partition the
    pixel and a fragment spawns at most one continuation per sample it owns, so
    the true per-pixel ceiling is ``_AA_NUM_SAMPLES`` (8), and a dense mesh sits
    near it: measured 6.10 continuations per covered pixel on a smooth-shaded
    metal sphere filling the covered set, against this budget of 5.

    Sizing for that ceiling instead was measured and rejected. The pool is
    ``primaries * ratio`` and the per-tile setup is O(pool) (the full-pool
    ``rs_int[:, 2]`` DONE prefill, the compactor), so pool bytes stay flat while
    the tile COUNT scales with the ratio -- ratio 9 cost 3.7% on a mixed scene
    whose real demand was 1.86, to save 5% on the metal-dominated one. The
    estimate stays where the common case is; ``_overflow_retry_primary`` makes
    being wrong cost one resolve pass rather than a halving cascade.
    """
    physical_split = bool(
        merged.get("has_refractive")
        or merged.get("has_refl_transparent")
        or custom_scatter
    )
    ratio = int(rt_settings.refract_initial_pool_ratio) if physical_split else 1
    if _secondary_split_needed(merged, analytic_raster):
        samples = int(rt_settings.effective_analytic_aa_secondary_samples())
        strong = bool(
            merged.get("has_strong_reflective")
            or merged.get("has_refractive")
            or merged.get("has_refl_transparent")
        )
        if strong and physical_split:
            # Glass can split at several layers, so retain the measured
            # physical-path multiplier.
            ratio = max(ratio, 1) * samples
        elif strong:
            # An opaque mirror needs N continuation slots when fully covered.
            # At a silhouette it can need those N plus the original
            # pass-through branch, but it does not need the glass path's 2N.
            ratio = max(ratio, samples + 1)
        else:
            # A weak dielectric sheen emits one reflection plus, at a
            # silhouette, the original pass-through.
            ratio = max(ratio, 2)
    return ratio


def _shared_pool_slots(
    primary_capacity, memory_primary, pool_ratio, analytic_raster, *, triangle_aa=None
):
    """How many slots to allocate for the shared continuation pool.

    ``pool_ratio`` is an ESTIMATE of the average continuations per primary (see
    :func:`_split_pool_ratio`) and the tile is clamped to the WORK, so deriving
    the pool from the clamped tile too leaves a covered set that fits in one
    tile with exactly zero slack over that estimate. Any batch whose real
    demand exceeds it then overflows its FIRST attempt and throws away a
    finished resolve -- and analytic triangle coverage routinely exceeds it
    (6.10 continuations per covered pixel measured on a metal sphere, against
    an estimate of 5).

    So a work-clamped tile gets the analytic CEILING instead: one continuation
    per sub-pixel coverage sample plus the pass-through, which a coverage
    partition cannot exceed on opaque geometry. It is capped by the slots the
    memory budget actually granted, so a memory-clamped tile is unchanged --
    there is no spare memory to take. The TILE is untouched either way, which
    is what separates this from raising ``pool_ratio``: that divides the tile
    size and pays for the headroom in extra tiles (measured 3.7% on a mixed
    scene), while the only per-tile work that scales with the pool is the DONE
    prefill and the compaction scan, ~3 ms at 5M slots.
    """
    budgeted = max(1, int(memory_primary)) * int(pool_ratio)
    ratio = int(pool_ratio)
    if triangle_aa is None:
        triangle_aa = analytic_raster and rt_settings.analytic_aa_tri_active()
    if analytic_raster and triangle_aa:
        from algan.rendering.raytracing.raster_taichi import _AA_NUM_SAMPLES

        ratio = max(ratio, _AA_NUM_SAMPLES + 1)
    return max(1, min(int(primary_capacity) * ratio, budgeted))


# Fraction of the exactly-measured fit to actually retry with (see
# ``_overflow_retry_primary``). A second overflow costs another discarded
# resolve, so the margin is deliberately generous relative to the ~10% spread
# in per-pixel demand that a coverage partition produces. Lower it on a scene
# that still overflows its retry; raise it towards 1 to trade retry risk for
# tile efficiency.
pool_retry_safety = min(1.0, max(0.0, env_float("ALGAN_POOL_RETRY_SAFETY", 0.85)))

# Bounce iterations that get their own profile label
# (``wavefront:   - bounce <i> <phase>``); iterations past this share the one
# ``bounce 8+`` label, so a pathological scene cannot grow the report's table
# without bound. Purely an instrumentation constant: it names stages, it never
# gates work.
_BOUNCE_STAGE_CAP = 8


def _read_tile_alloc(rs_alloc):
    """Copy one tile's ``rs_alloc`` words to the host, in a single transfer.

    Every word is read on every tile now, including the overflow flag on the
    split-free path that used to short-circuit past it. That short-circuit is
    what made a failed reservation at ``pool_ratio == 1`` invisible: there is
    no retry to make there, so the flag was never even looked at, and the
    dropped branch left no trace (RENDERER_WORK_QUEUE.md item 1). Looking costs
    one device synchronisation per tile, against the one the ray compactor
    already forces per wavefront iteration *inside* the tile.
    """
    return rs_alloc.tolist()


def _record_tile_truncations(alloc, pool):
    """Fold an ACCEPTED tile attempt's truncation counters into the render's.

    Only accepted attempts: an overflowing tile is discarded and re-run, and
    counting the discarded attempt would report truncations for frames that
    were never composited.
    """
    record_truncation(
        "surfaces_per_ray",
        alloc[ALLOC_TRUNC_SURFACES],
        cap=max_surfaces_per_ray,
    )
    # ``rs_alloc[ALLOC_NEXT]`` keeps counting past the capacity (a failed
    # reservation still does its atomic increment), so the surplus over the
    # pool IS the number of continuations that found no slot. A splitting
    # batch retries instead of accepting that, which is why an accepted
    # attempt can only show a surplus at ``pool_ratio == 1``.
    record_truncation("dropped_continuations", alloc[ALLOC_NEXT] - pool)


def _overflow_retry_primary(attempt_primary, slots_wanted, pool):
    """Primary count to retry an overflowing tile with.

    ``rs_alloc[0]`` keeps counting past the capacity -- a failed reservation
    still does its atomic increment -- so an overflowing tile reports EXACTLY
    how many slots it wanted. Scaling the primaries by
    ``pool / slots_wanted`` therefore lands on a tile that fits in one step,
    instead of halving blindly: on the metal sphere the halving overshot to
    1798 primaries where 2654 fit, and because ``learned_primary_cap`` only
    ever shrinks, every later tile in the render inherited the overshoot.

    A safety margin absorbs the fact that demand per pixel is not uniform
    across a tile (the slice that survives the shrink may be denser than the
    average that produced the measurement), and the result is always at least
    one primary smaller than the failed attempt so the retry loop terminates.
    """
    if slots_wanted <= 0 or pool <= 0:
        return max(1, attempt_primary // 2)
    scaled = int(attempt_primary * pool * pool_retry_safety / slots_wanted)
    return max(1, min(scaled, attempt_primary - 1))


def _shrink_sparse_memory_retry(attempt_primary, shared_pool_capacity, pool_ratio):
    """Reduce both primary work and its pool after an arena allocation failure.

    Overflow retries intentionally keep the pool so fewer primaries inherit
    its spare slots. Memory retries cannot: that pool may be the allocation
    preventing even one primary's scratch from fitting. Both the primary
    resolve and secondary drain use this rule, after their one-pixel guard.
    """
    primary = max(1, int(attempt_primary) // 2)
    if pool_ratio > 1:
        shared_pool_capacity = max(primary, int(shared_pool_capacity) // 2)
        pool = shared_pool_capacity
    else:
        pool = primary
    return primary, shared_pool_capacity, pool


def _resolve_wavefront_policy(
    merged,
    *,
    analytic_raster,
    aa_level,
    near_clip,
    refraction,
    ior_stack,
    shadow_mode,
    custom_scatter,
    has_triangles,
):
    """Resolve once; low-level callers can use their already-decided flags."""
    pool_ratio = _split_pool_ratio(refraction, merged, analytic_raster, custom_scatter)
    memory_trim = bool(
        rt_settings.wf_mem_trim
        and merged.get("mem_trim_active")
        and not shadow_mode
        and not custom_scatter
        and not refraction
    )
    opaque_allowed = bool(
        not merged.get("opaque_bvh_skipped", False)
        and not refraction
        and not custom_scatter
        and not memory_trim
    )
    return WavefrontPolicy(
        refraction=bool(refraction),
        ior_stack=bool(ior_stack),
        state_scalar_width=sca_width(ior_stack),
        pool_ratio=pool_ratio,
        tile_rays=int(rt_settings.wavefront_tile_rays),
        memory_trim=memory_trim,
        opaque_closest=bool(
            rt_settings.wf_opaque_closest
            and opaque_allowed
            and merged.get("all_visible_opaque", False)
        ),
        opaque_prepass=bool(
            rt_settings.wf_opaque_prepass
            and opaque_allowed
            and merged.get("has_any_opaque", False)
            and merged.get("has_any_translucent", False)
            and not merged.get("has_uncertain_texture_alpha", False)
        ),
        fused_generation=bool(
            rt_settings.wf_gen_fused_active()
            and pool_ratio == 1
            and near_clip <= 0.0
            and max(1, int(aa_level)) <= 1
            and not analytic_raster
        ),
        triangle_pipeline_mask=_frag_pid_mask(
            merged, "tri", has_triangles, _record=False
        ),
        triangle_aa=bool(rt_settings.analytic_aa_tri_active()),
    )


def resolve_batch_policy(
    merged,
    requested_aa=1,
    *,
    light_sources=(),
    environment_map=None,
    near_clip=0.0,
    far_clip=0.0,
    transparent_background=False,
):
    """Prepare the shared allocation/execution policy from live settings.

    Called once for each newly prepared batch, never at import time. Cheap
    scene facts and capability gates are resolved here rather than independently
    in arena preflight, frame sizing, and wavefront allocation. Fallback reasons
    name the vetoes; they do not silently select a different renderer backend.
    """
    samples = max(1, int(rt_settings.samples_per_pixel))
    deterministic = samples <= 1
    requested_aa = max(1, int(requested_aa))
    near_clip, far_clip = float(near_clip), float(far_clip)
    requested_shadows = bool(rt_settings.shadows)
    shadows = bool(requested_shadows and deterministic)
    extended = bool(
        deterministic
        and any(
            getattr(light, "_render_aux", None) is not None
            for light in (light_sources or ())
        )
    )
    fragment = bool(
        deterministic
        and (
            rt_settings.fragment_shading
            or shadows
            or _scene_has_user_pipeline(merged)
            or extended
            or environment_map is not None
        )
    )
    custom_scatter = bool(
        (fragment or not deterministic) and _scene_has_custom_scatter(merged)
    )
    num_tri, num_bez = (
        int(merged.get("num_triangles", 0)),
        int(merged.get("num_circuits", 0)),
    )
    has_tri = bool(not rt_settings.gate_empty_traversals or num_tri > 0)
    has_bez = bool(not rt_settings.gate_empty_traversals or num_bez > 0)
    analytic_split = bool(deterministic and _secondary_split_needed(merged, True))
    physical_split = bool(
        merged.get("has_refractive") or merged.get("has_refl_transparent")
    )

    reasons = []
    if deterministic:
        for name in (
            "hybrid_raster",
            "analytic_aa",
            "sheet_resolve",
            "analytic_aa_run",
            "raster_sparse_coverage",
            "raster_empty_skip",
            "raster_covered_shade",
        ):
            if not getattr(rt_settings, name):
                reasons.append(f"{name}_disabled")
        if transparent_background and environment_map is not None:
            reasons.append("transparent_environment_background")
        if merged.get("tri_frame_valid") is None:
            reasons.append("analytic_projection_unavailable")
        if near_clip > 0.0:
            reasons.append("near_clipping")
        if num_tri <= 0 and num_bez <= 0:
            reasons.append("no_raster_geometry")
        if num_tri > 0 and not rt_settings.analytic_aa_tri_active():
            reasons.append("triangle_analytic_aa_unavailable")
        if num_bez > 0 and not rt_settings.analytic_aa_bez_active():
            reasons.append("bezier_analytic_aa_unavailable")
        if custom_scatter:
            reasons.append("custom_scatter")
        if (
            rt_settings.wf_mem_trim
            and merged.get("mem_trim_active")
            and not shadows
            and not physical_split
            and not analytic_split
            and not custom_scatter
        ):
            reasons.append("memory_trim")
    analytic = bool(deterministic and not reasons)
    effective_aa = 1 if analytic or not deterministic else requested_aa
    inplace_aa = bool(rt_settings.inplace_aa)
    shadow_mode = int(shadows)
    if shadows and rt_settings.shadow_anyhit:
        if rt_settings.shadow_anyhit == "gather":
            shadow_mode = 4
        elif merged.get("has_transmissive", True):
            shadow_mode = 1
        else:
            translucent = bool(
                merged.get("tri_has_translucent", True)
                or merged.get("bez_has_translucent", True)
                or merged.get("has_uncertain_texture_alpha", True)
            )
            shadow_mode = 2 if translucent else 3
    refraction = bool(
        deterministic
        and (physical_split or custom_scatter or (analytic and analytic_split))
    )
    ior_stack = bool(rt_settings.nested_ior_mode() != 0 and refraction)
    wavefront = None
    if deterministic:
        wavefront = _resolve_wavefront_policy(
            merged,
            analytic_raster=analytic,
            aa_level=effective_aa if inplace_aa else 1,
            near_clip=near_clip,
            refraction=refraction,
            ior_stack=ior_stack,
            shadow_mode=shadow_mode,
            custom_scatter=custom_scatter,
            has_triangles=has_tri,
        )
    return BatchExecutionPolicy(
        primary_route="path_tracer"
        if not deterministic
        else "analytic_sheets"
        if analytic
        else "classic_wavefront",
        fallback_reasons=tuple(reasons),
        samples_per_pixel=samples,
        requested_aa=requested_aa,
        effective_aa=effective_aa,
        inplace_aa=inplace_aa,
        fragment_shading=fragment,
        shadow_mode=shadow_mode,
        shadows=requested_shadows,
        custom_scatter=custom_scatter,
        lights_extended=extended,
        has_triangles=has_tri,
        has_beziers=has_bez,
        max_bounces=int(rt_settings.max_bounces),
        near_clip=near_clip,
        far_clip=far_clip,
        wavefront=wavefront,
    )


def analytic_raster_route_active(merged, **kwargs):
    """Compatibility query for callers that need only the primary-route flag."""
    return resolve_batch_policy(merged, **kwargs).analytic_raster


def effective_anti_alias_level(merged, requested, **kwargs):
    """Sample AA level; use the prepared policy's frame_scale for buffer sizes."""
    return resolve_batch_policy(merged, requested, **kwargs).effective_aa


def _wavefront_state_bytes_per_primary(
    pool_ratio,
    extra_bytes_per_slot=0,
    extra_bytes_per_primary=0,
    state_sca_width=SCA_WIDTH_PLAIN,
):
    """Bytes charged to one initial primary when sizing a wavefront tile.

    Each primary contributes ``pool_ratio`` slots to one *shared* continuation
    pool, rather than owning a private block. ``pix_accum`` remains per primary;
    the two-word shared allocator (next slot + overflow flag) is fixed per tile
    and is accounted separately by the callers. Orchestrator-specific extras
    (for example the sorted path's event record/key arrays) are passed in so
    adaptive tile sizing can account for them.
    """
    coefficients = _wavefront_state_coefficients(state_sca_width)
    per_slot = coefficients["pool"] + extra_bytes_per_slot
    per_primary = coefficients["primary"] + extra_bytes_per_primary
    return pool_ratio * per_slot + per_primary


# Ray-state cost of one pool slot and one primary ray, in bytes. These are
# *measured*, not derived: recording the arena while rendering gives 100 and 28
# for the maintained route. An earlier hand-derived version charged 196 per
# slot because it counted 6*kbuf words of K-buffers that this route does not
# allocate at all -- they are (1,1) stubs, with a transient event batch sized
# to the live queue instead -- which halved every tile for no reason.
#
# Only wavefront *tile* sizing reads these, and the arena bounds the result, so
# an inaccuracy here costs tile efficiency and at worst an out-of-memory retry;
# it is not what sizes a render batch. To re-measure, render with
# ``ALGAN_WAVEFRONT_TILE_AUTO=0`` at two values of ``ALGAN_WAVEFRONT_TILE_RAYS``
# and difference the arena's high-water mark.
_WAVEFRONT_BYTES_PER_POOL_SLOT = 100
_WAVEFRONT_BYTES_PER_PRIMARY = 28
_WAVEFRONT_FIXED_BYTES = 24


def _wavefront_state_coefficients(state_sca_width=SCA_WIDTH_PLAIN):
    """Measured per-slot / per-primary / fixed bytes of the ray-state block,
    for a tile whose ``rs_sca`` rows carry ``state_sca_width`` f32 columns.

    The measured constants describe the classic ``SCA_WIDTH_PLAIN`` layout, so
    a wider row is charged its difference per pool slot. The width is the
    BATCH's (``sca_width(ior_stack_flag)``), not the nested-IOR setting's:
    ``ior_stack_flag`` is the setting AND ``refraction_flag``, and the two part
    company for any batch the flag leaves clear. Charging the setting instead
    -- which is what this did while the nested-IOR gate was off by default,
    where the two agreed -- shrinks every such tile the moment that default
    flips, for state those tiles never allocate.
    """
    coefficients = {
        "pool": _WAVEFRONT_BYTES_PER_POOL_SLOT,
        "primary": _WAVEFRONT_BYTES_PER_PRIMARY,
        "fixed": _WAVEFRONT_FIXED_BYTES,
    }
    extra_columns = int(state_sca_width) - SCA_WIDTH_PLAIN
    if extra_columns > 0:
        # Nested-IOR media stack (DESIGN_mesh_identity_open.md §H): rs_sca rows
        # grow to SCA_WIDTH_NESTED f32 columns per slot in a refracting batch
        # (the depth counter plus IOR_STACK_DEPTH entries), so tile sizing must
        # charge them or every auto tile overruns the arena it was fitted to.
        coefficients["pool"] += extra_columns * torch.float32.itemsize
    return coefficients


def _auto_primary_per_tile(
    memory,
    pool_ratio,
    static_primary,
    extra_bytes_per_slot=0,
    extra_bytes_per_primary=0,
    fixed_bytes=0,
    state_sca_width=SCA_WIDTH_PLAIN,
):
    """Primary rays per wavefront tile, sized from the render pool's free
    bytes when ``settings.wavefront_tile_auto`` is on (see settings.py for the
    rationale: fewer, bigger tiles amortize the fixed host-side kernel-launch
    cost). Falls back to ``static_primary`` (the wavefront_tile_rays-derived
    value) for unmanaged pools or when auto is disabled. Byte-identical to any
    other tile size: tiles partition pixels, and every per-pixel computation
    is independent of its tile.
    """
    if not rt_settings.wavefront_tile_auto or not getattr(memory, "managed", False):
        return static_primary
    bytes_per_primary = _wavefront_state_bytes_per_primary(
        pool_ratio, extra_bytes_per_slot, extra_bytes_per_primary, state_sca_width
    )
    free = memory.get_num_bytes_remaining()
    # Every per-tile allocation is f32/i32.  The output immediately before it
    # can be uint8 (including an odd-sized five-channel transparent frame), so
    # mirror ManualMemory's one initial four-byte alignment exactly.  All
    # subsequent allocations remain aligned because their sizes are multiples
    # of four.
    alignment_bytes = (-memory.current_pointer) % torch.float32.itemsize
    safety = min(1.0, max(0.0, float(rt_settings.wavefront_tile_safety)))
    usable = int(free * safety) - alignment_bytes - int(fixed_bytes)
    budget = max(0, usable) // bytes_per_primary
    hi = max(1, rt_settings.wavefront_tile_max // pool_ratio)
    lo = min(hi, max(1, rt_settings.wavefront_tile_min // pool_ratio))
    if budget < hi:
        # This chunk's peak depends on the capacity supplied by scene prep.
        # It cannot establish a minimum workspace for the next scene batch.
        memory.last_chunk_capacity_limited = True
    # The minimum is a launch-amortisation preference, not permission to
    # overrun the arena.  When less than the preferred floor fits, use the
    # exact smaller value; a one-primary allocation is attempted only when no
    # primary can fit, preserving the normal single-frame OOM diagnostic.
    if budget < lo:
        return max(1, budget)
    return min(budget, hi)


class _ArenaRayCompactor:
    """Stable-lifetime ray-index buffers owned by ``ManualMemory``.

    PyTorch's comparison/advanced-index/nonzero chain created several fresh
    CUDA tensors after the render arena had already been reserved.  A Taichi
    filter kernel now writes directly into these two ping-pong buffers and a
    one-word counter, so compaction cannot exceed the arena allowance.
    """

    def __init__(self, memory, capacity, dtype=torch.int32):
        self.capacity = int(capacity)
        self.a = memory.get_tensor((self.capacity,), dtype)
        self.b = memory.get_tensor((self.capacity,), dtype)
        self.count = memory.get_tensor((1,), dtype)
        self.current = self.a
        self.spare = self.b
        self.size = 0

    def reorder(self, active, perm):
        """Permute ``active`` (the list ``select`` last returned) by ``perm``
        (an int64 index tensor of its length) into the spare buffer and make
        that the current list. Returns the new view, as ``select`` does.
        """
        n = int(active.numel())
        if active.device.type == "mps":
            reorder_ray_slots(active, perm, self.spare, n)
        else:
            torch.index_select(active, 0, perm, out=self.spare[:n])
        self.current, self.spare = self.spare, self.current
        self.size = n
        return self.current[:n]

    def initial(self, size):
        self.size = int(size)
        torch.arange(self.size, out=self.current[: self.size])
        return self.current[: self.size]

    def select(
        self,
        rs_int,
        desired_status,
        *,
        source=None,
        scan_pool=False,
        rs_key=None,
        desired_key=0,
    ):
        if source is None:
            source = self.current[: self.size]
        source_size = self.capacity if scan_pool else int(source.numel())
        self.count.zero_()
        compact_ray_slots(
            source,
            source_size,
            bool(scan_pool),
            int(desired_status),
            # ``rs_key`` is a compile-time-unused argument without a key
            # predicate. Reuse the current index array instead of reserving an
            # otherwise dead placeholder word.
            rs_int,
            self.current if rs_key is None else rs_key,
            rs_key is not None,
            int(desired_key),
            self.spare,
            self.count,
        )
        size = int(self.count.item())
        self.current, self.spare = self.spare, self.current
        self.size = size
        return self.current[:size]


_kernel_compile_notice_shown = False


def _show_kernel_compile_notice():
    global _kernel_compile_notice_shown
    if _kernel_compile_notice_shown:
        return
    _kernel_compile_notice_shown = True
    logger.info(
        "Preparing render kernels. If this is the first render on this machine"
        " (or after an update), compiling the GPU kernels can take several"
        " minutes. Compiled kernels are cached, so subsequent renders start"
        " immediately."
    )


def _observe_render_kernel_compiles(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if not _kernel_compile_notice_shown:
            _set_compile_notice_callback(_show_kernel_compile_notice)
        try:
            return function(*args, **kwargs)
        finally:
            _set_compile_notice_callback(None)

    return wrapped


def _append_env_texture(textures, env, intensity, device):
    """Append an equirect environment map to the shared flat texel buffer.

    Returns the widened buffer and the map's placement meta
    ``(offset, width, height, intensity)`` for the shade kernel (packed into
    the typed render metadata arrays -- the kernel is at the 64-arg ceiling).
    Texels are stored column-major (``offset + x * height + y``) to match
    ``_sample_tex_vec5``.
    """
    # Environment resampling and concatenation are scene preparation, not
    # rendering.  Always perform them on the host even when a direct caller
    # supplied render-device textures; the completed storage is uploaded once
    # through ``copy_merged_scene_to_arena``.
    device = torch.device("cpu")
    textures = textures.detach().cpu()
    env = _host_tensor(env).float()
    max_w = 2048
    if env.shape[1] > max_w:
        scale = max_w / env.shape[1]
        env = torch.nn.functional.interpolate(
            env.permute(2, 0, 1).unsqueeze(0), scale_factor=scale, mode="area"
        )[0].permute(1, 2, 0)
    h, w = int(env.shape[0]), int(env.shape[1])
    texels = torch.zeros((w * h, 5), dtype=torch.float32, device=device)
    texels[:, :3] = env.permute(1, 0, 2).reshape(w * h, 3)
    texels[:, 3] = 1.0
    if textures.dtype != torch.float32:
        texels = texels.to(textures.dtype)
    offset = int(textures.shape[1])
    texels = texels.unsqueeze(0).expand(textures.shape[0], -1, -1)
    textures = torch.cat((textures, texels), 1).contiguous()
    return textures, (offset, w, h, float(intensity))


def _env_sh_coeffs(env, intensity):
    """Order-1 spherical-harmonics irradiance of an equirect environment map,
    as the linear form ``E(n) / pi = A + Bx*nx + By*ny + Bz*nz`` consumed by
    the in-kernel ENV_SH light row. A uniform map of color ``c`` yields
    ``A = c, B = 0`` -- i.e. it lights like an ambient light of color ``c``.
    """
    import math

    e = _host_tensor(env).float()
    if e.shape[0] > 32 or e.shape[1] > 64:
        e = torch.nn.functional.adaptive_avg_pool2d(
            e.permute(2, 0, 1).unsqueeze(0), (16, 32)
        )[0].permute(1, 2, 0)
    H, W = int(e.shape[0]), int(e.shape[1])
    v = (torch.arange(H, dtype=torch.float32) + 0.5) / H
    theta = math.pi * v  # 0 at the top row (y = +1)
    y = torch.cos(theta)
    sin_t = torch.sin(theta)
    u = (torch.arange(W, dtype=torch.float32) + 0.5) / W
    phi = (u - 0.5) * (2.0 * math.pi)  # matches _sample_env_map
    x = torch.cos(phi).unsqueeze(0) * sin_t.unsqueeze(1)
    z = torch.sin(phi).unsqueeze(0) * sin_t.unsqueeze(1)
    yy = y.unsqueeze(1).expand(H, W)
    dw = (sin_t * (math.pi / H) * (2.0 * math.pi / W)).unsqueeze(1)
    L = e * float(intensity)
    w = dw.expand(H, W).unsqueeze(-1)
    Y00 = 0.28209479177387814
    Y1 = 0.4886025119029199
    L00 = (L * (Y00 * w)).sum((0, 1))
    L1x = (L * ((Y1 * x).unsqueeze(-1) * w)).sum((0, 1))
    L1y = (L * ((Y1 * yy).unsqueeze(-1) * w)).sum((0, 1))
    L1z = (L * ((Y1 * z).unsqueeze(-1) * w)).sum((0, 1))
    a0 = math.pi  # irradiance convolution coefficients
    a1 = 2.0 * math.pi / 3.0
    A = a0 * Y00 * L00 / math.pi
    Bx = a1 * Y1 * L1x / math.pi
    By = a1 * Y1 * L1y / math.pi
    Bz = a1 * Y1 * L1z / math.pi
    return A, Bx, By, Bz


def _append_env_sh_light(light_pos, light_col, num_lights, env, intensity, device):
    """Add the environment map's diffuse irradiance as one ENV_SH light row
    (type 6) to the packed lights, widening the color rows to 16 columns if
    they are still in the compact point-light packing.
    """
    A, Bx, By, Bz = _env_sh_coeffs(env, intensity)
    row = torch.zeros(16)
    row[0:3] = A
    row[3] = 6.0  # LIGHT_ENV_SH
    row[6:9] = Bx
    row[9:12] = By
    row[12:15] = Bz
    T = light_pos.shape[0] if num_lights > 0 else 1
    row = row.view(1, 1, 16).expand(T, 1, 16).to(device)
    zero_pos = torch.zeros((T, 1, 3), device=device)
    if num_lights == 0:
        return zero_pos.contiguous(), row.contiguous(), 1
    if light_col.shape[2] < 16:
        pad = torch.zeros(
            (light_col.shape[0], light_col.shape[1], 16 - light_col.shape[2]),
            device=device,
        )
        light_col = torch.cat((light_col, pad), -1)
    light_pos = torch.cat((light_pos, zero_pos), 1).contiguous()
    light_col = torch.cat((light_col, row), 1).contiguous()
    return light_pos, light_col, num_lights + 1


def _build_render_plan(
    samples_per_pixel,
    scene_environment_map,
    merged,
    light_sources=(),
    *,
    execution_policy=None,
):
    """Resolve the renderer route and feature compatibility for a batch."""
    samples_requested = max(1, int(samples_per_pixel))
    backend = "path_tracer" if samples_requested > 1 else "deterministic_wavefront"
    requested = []
    # Renderer selection is explicit; this does not retry on another backend.
    # Homogeneous interiors require stochastic transport, so the deterministic
    # route must name the unsupported feature rather than silently dropping it.
    unsupported = []
    if bool(merged.get("has_scattering_media")):
        requested.append("homogeneous scattering media / random-walk SSS")
        if samples_requested <= 1:
            unsupported.append("homogeneous scattering media / random-walk SSS")
    if scene_environment_map is not None:
        requested.append("environment maps")
    if bool(merged.get("has_refractive")):
        requested.append("refractive materials")
    if _scene_has_user_pipeline(merged):
        requested.append("custom fragment-shader pipelines")
    if any(
        getattr(light, "_render_aux", None) is not None
        for light in (light_sources or ())
    ):
        requested.append("extended lights")
    return RenderPlan(
        backend=backend,
        samples_per_pixel=samples_requested,
        requested_features=tuple(requested),
        unsupported_features=tuple(unsupported),
        primary_route=execution_policy.primary_route
        if execution_policy is not None
        else None,
        effective_anti_alias_level=execution_policy.effective_aa
        if execution_policy is not None
        else 1,
        fallback_reasons=execution_policy.fallback_reasons
        if execution_policy is not None
        else (),
    )


def _validate_render_capabilities(
    samples_per_pixel,
    scene_environment_map,
    merged,
    light_sources=(),
    *,
    execution_policy=None,
):
    """Apply the unsupported-feature policy to the selected renderer.

    Selection is explicit: one sample uses the deterministic renderer; more
    than one uses the path tracer. The capability checks here cover the feature
    metadata collected by ``_build_render_plan``, not every possible geometry,
    user shader or allocation error. Homogeneous scattering currently requires
    the path tracer. The policy raises by default; warning and ignore modes are
    intended for controlled migration and comparisons, not silent fallbacks.
    """
    plan = _build_render_plan(
        samples_per_pixel,
        scene_environment_map,
        merged,
        light_sources,
        execution_policy=execution_policy,
    )
    if plan.unsupported_features:
        feature_list = ", ".join(plan.unsupported_features)
        suggestion = (
            "Set samples_per_pixel > 1 to use the path tracer."
            if plan.backend == "deterministic_wavefront"
            else "Remove the unsupported features or use a compatible renderer."
        )
        rt_settings.report_unsupported_features(
            f"The {plan.backend} renderer cannot honor: {feature_list}. "
            f"{suggestion} To discard these features deliberately, use "
            "set_unsupported_feature_policy('warn'/'ignore') explicitly."
        )

    return plan


#: Merged-scene key holding the arena reverse pointer this batch has allocated
#: down to and must keep: the lowest (deepest) persistent allocation made from
#: inside a chunk that still has to be readable in the next one. Read by
#: ``rewind_to`` and by ``RenderLoopMixin._render_primitive_batch``. Published
#: explicitly by whoever makes such an allocation rather than read off the
#: arena, so an unrelated persistent allocation inside a chunk can never be
#: retained by accident.
ARENA_RETAINED_REVERSE_POINTER = "_arena_retained_reverse_pointer"


def _retain_persistent(merged, memory):
    """Publish the arena's current reverse pointer as batch-lived.

    Lowers ``ARENA_RETAINED_REVERSE_POINTER`` to the pointer reached, so that
    several publishers in one batch (the raster tables, a re-homed deferred
    BVH build) each hold their own range open. A ``None`` arena means nothing
    was allocated from one, so there is nothing to hold.
    """
    if memory is None:
        return
    reverse = memory.get_pointers()[1]
    retained = merged.get(ARENA_RETAINED_REVERSE_POINTER)
    if retained is None or reverse < retained:
        merged[ARENA_RETAINED_REVERSE_POINTER] = reverse


class _DeferredBVHRequired(Exception):
    """Restart a sparse chunk at a boundary that owns no coverage or tile state."""


def _publish_deferred_bvhs(merged, memory):
    """Publish a batch's trees transactionally, then retain only their storage.

    Call before chunk-local reverse allocations. A sparse continuation discovered
    later requests a chunk restart instead of pinning the coverage beneath it.
    Construction and publication have separate state: a failed copy leaves the
    external trees available for a retry, with neither arena end consumed.
    """
    from algan.rendering.raytracing.scene_builder import build_deferred_bvhs

    pointers = memory.get_pointers()
    try:
        build_deferred_bvhs(merged, memory)
    except Exception:
        memory.set_pointers(pointers)
        raise
    _retain_persistent(merged, memory)


def _publish_bvhs_at_chunk_boundary(merged, memory):
    """One reclaim-and-retry for allocator pressure, never retry a real error.

    At this boundary a smaller ray tile cannot free any more arena bytes. If
    publication still fails after allocator cleanup, the prepared scene batch
    must shrink; do not repeatedly rediscover coverage or halve its ray pool.
    """
    for attempt in range(2):
        try:
            _publish_deferred_bvhs(merged, memory)
            return
        except (InsufficientMemoryException, RuntimeError) as exc:
            if not isinstance(exc, InsufficientMemoryException) and not is_cuda_oom(
                exc
            ):
                raise
            if attempt:
                raise OutOfRenderMemory(
                    "Deferred BVHs did not fit at the clean chunk boundary. "
                    "Reduce the prepared scene batch or geometry complexity."
                ) from exc
            traceback.clear_frames(exc.__traceback__)
            release_torch_memory(force_gc=False)


def _build_raster_tables(
    merged,
    memory,
    cam_origin,
    screen_point,
    pixel_basis_x,
    pixel_basis_y,
    half_screen_w,
    half_screen_h,
    width,
):
    """Build the batch-wide raster projection / bounds tables, once per batch.

    Allocated at the arena's persistent end: they cover the whole prepared
    batch, every chunk of it reads the same rows, and the per-chunk reset only
    rewinds the forward pointer. See the call site for the lifetime contract.
    """
    from algan.rendering.raytracing.raster_pipeline import (
        precompute_circuit_screen_bounds,
        precompute_triangle_projection,
        precompute_triangle_screen_bounds,
    )

    tri_bounds = None
    bez_bounds = None
    with memory.scope(
        "raster_precompute", aa_tri=int(rt_settings.analytic_aa_tri_active())
    ):
        tri_screen = precompute_triangle_projection(
            merged,
            cam_origin,
            screen_point,
            pixel_basis_x,
            pixel_basis_y,
            half_screen_w,
            half_screen_h,
            memory,
            persist=True,
        )
        # Live reads (settings convention): each kill-switch falls back to
        # the per-frame pair emission inside prepare_sparse_raster_coverage.
        if (
            rt_settings.raster_tri_precompute
            and int(merged.get("num_triangles", 0)) > 0
        ):
            tri_bounds = precompute_triangle_screen_bounds(
                merged,
                tri_screen,
                cam_origin,
                screen_point,
                pixel_basis_x,
                pixel_basis_y,
                half_screen_w,
                half_screen_h,
                width,
                memory,
                persist=True,
            )
        if rt_settings.raster_bez_precompute and int(merged.get("num_circuits", 0)) > 0:
            bez_bounds = precompute_circuit_screen_bounds(
                merged,
                cam_origin,
                screen_point,
                pixel_basis_x,
                pixel_basis_y,
                half_screen_w,
                half_screen_h,
                width,
                memory,
                persist=True,
            )
    return tri_screen, tri_bounds, bez_bounds


@_observe_render_kernel_compiles
def render_batch_raytraced(
    primitives,
    scene,
    screen_width,
    screen_height,
    time_start,
    time_end,
    background,
    transparent_background,
    ray_origin,
    screen_point,
    screen_basis,
    anti_alias_level=1,
    light_sources=(),
    memory=None,
    post_processes=(),
    **kwargs,
):
    """Render a primitive batch through the selected hybrid or path tracer.

    Consume the batch's immutable scene/camera/light data, validate renderer
    capabilities, reserve scene and transient arena storage, and render frame
    chunks into the output buffer. Memory includes geometry, acceleration
    structures, hit events, path/continuation state and post-processing scratch;
    it is not independent of scene complexity. The enclosing render loop owns
    frame-window retries, while renderer-specific tiling bounds transient work.
    """
    # Core route/allocation choices come from the prepared batch, not module
    # imports or a second, potentially inconsistent settings resolution.
    tonemap_exposure = rt_settings.tonemap_exposure
    scene_env_map = getattr(scene, "environment_map", None)
    env_map = scene_env_map
    env_source = env_map.detach().cpu() if torch.is_tensor(env_map) else env_map
    env_meta = getattr(primitives[0], "_rt_env_meta", None)
    merged = getattr(primitives[0], "_rt_device_scene", None)
    policy = getattr(primitives[0], "_rt_batch_policy", None)
    camera = getattr(scene, "camera", None)
    policy_inputs = {
        "light_sources": light_sources,
        "environment_map": env_map,
        "near_clip": float(getattr(camera, "near", 0.0) or 0.0),
        "far_clip": float(getattr(camera, "far", 0.0) or 0.0),
        "transparent_background": transparent_background,
    }
    if merged is None:
        merged_host = _merge_scene(primitives, light_sources=light_sources)
        if policy is None:
            policy = resolve_batch_policy(
                merged_host, anti_alias_level, **policy_inputs
            )
        # Validate on host metadata before reserving/copying the persistent
        # device scene. Unsupported combinations therefore fail before costly
        # arena allocations or any Taichi kernel compilation.
        plan = _validate_render_capabilities(
            policy.samples_per_pixel,
            scene_env_map,
            merged_host,
            light_sources,
            execution_policy=policy,
        )
        if env_map is not None:
            merged_host = dict(merged_host)
            texture_device = merged_host["textures"].device
            merged_host["textures"], env_meta = _append_env_texture(
                merged_host["textures"],
                env_source,
                float(getattr(scene, "environment_intensity", 1.0)),
                texture_device,
            )
        merged = copy_merged_scene_to_arena(merged_host, memory, persist=True)
    else:
        if policy is None:
            policy = resolve_batch_policy(merged, anti_alias_level, **policy_inputs)
        plan = _validate_render_capabilities(
            policy.samples_per_pixel,
            scene_env_map,
            merged,
            light_sources,
            execution_policy=policy,
        )
    scene.last_render_plan = plan

    samples_per_pixel = policy.samples_per_pixel
    max_bounces = policy.max_bounces
    lights_extended = policy.lights_extended
    near_clip, far_clip = policy.near_clip, policy.far_clip
    analytic_raster = policy.analytic_raster
    aa = policy.effective_aa
    width = screen_width * policy.frame_scale
    height = screen_height * policy.frame_scale
    kernel_aa = policy.kernel_aa
    post_aa = policy.frame_scale

    C_out = 5 if transparent_background else 4
    device = memory.data.device
    num_frames = merged["num_frames"]

    # Camera snapshots stay on the animation/source device during prefetch.
    # Complete their small vector math there, then copy only the kernel-facing
    # results into arena-backed render tensors.
    cam_origin_host = _expand_frames(
        _flat_frames(_host_tensor(ray_origin), (3,)), num_frames
    ).contiguous()
    sp_host = _expand_frames(
        _flat_frames(_host_tensor(screen_point), (3,)), num_frames
    ).contiguous()
    sb_host = _expand_frames(
        _flat_frames(_host_tensor(screen_basis), (3, 3)), num_frames
    )
    pbx_host, pby_host = _pixel_bases(sb_host)
    # World units per screen pixel per unit distance (for border widths). Border
    # widths are authored in *anti-aliased* pixels (see BezierCircuit), so this
    # always uses the super-sampled height (screen_height * aa), whether or not
    # the frame buffer itself is super-sampled.
    b1_norm = sb_host[:, 1].norm(p=2, dim=-1)
    screen_dist = (sp_host - cam_origin_host).norm(p=2, dim=-1)
    pixel_world_scale_host = (
        2.0 / clamp_floor(screen_height * aa * b1_norm * screen_dist, 1e-12)
    ).contiguous()
    # Camera and packed-light inputs cover the whole prepared batch and are
    # paid once, so they are one calibration scope even though the light copies
    # happen further down (nothing else allocates in between).
    with memory.scope("persistent_inputs", cam_frames=int(cam_origin_host.shape[0])):
        cam_origin = _arena_copy(memory, cam_origin_host)
        sp = _arena_copy(memory, sp_host)
        pbx = _arena_copy(memory, pbx_host)
        pby = _arena_copy(memory, pby_host)
        pixel_world_scale = _arena_copy(memory, pixel_world_scale_host)

    # An animated/image background arrives super-sampled at the *requested*
    # anti-alias level (Scene.set_background and
    # _prepare_background_for_chunk both build it at screen * anti_alias_level).
    # This batch's frame buffer is at output resolution whenever the route
    # takes one sample per output pixel: in-place AA, and the analytic raster
    # route, which forces ``aa == 1`` however many samples were requested.
    # Average the background down to match -- a super-sampled background read
    # at output stride silently scrolls a different slice of itself into every
    # frame. (Solid colors are resolution-free and pass through untouched.)
    background_aa = max(1, int(anti_alias_level))
    if background_aa > 1 and width == screen_width and height == screen_height:
        background = _downsample_background(
            background,
            background_aa,
            time_end - time_start,
            screen_height,
            screen_width,
        )

    # A deferred-BVH batch (scene_builder._finalize_bvhs) holds placeholder
    # trees; the Monte Carlo megakernel traverses unconditionally, so build
    # the real trees now if that is where this batch is headed. (The
    # deterministic wavefront has its own later, finer-grained check.)
    if (merged.get("bvh_deferred") or merged.get("bvh_rehome_pending")) and int(
        samples_per_pixel
    ) > 1:
        _publish_deferred_bvhs(merged, memory)
    tri_bvh = merged["tri_bvh"]
    bez_bvh = merged["bez_bvh"]
    has_tri = int(policy.has_triangles)
    has_bez = int(policy.has_beziers)
    t_val = _get_tonemap_t_val()
    # The scene builder has already reduced every geometry type's per-frame
    # bounds/edge geometry to conservative batch-wide coverage-possibility bits.
    # If all three are false, every valid primitive is point-degenerate and
    # exact primary coverage is empty for the entire materialized batch: leave
    # the background prefill untouched and skip even the sparse COUNT discovery
    # pass.  This applies to moving batches too; it is not a static-scene
    # shortcut.
    sparse_batch_empty = bool(
        int(samples_per_pixel) <= 1
        and rt_settings.hybrid_raster
        and rt_settings.raster_sparse_coverage
        and rt_settings.raster_empty_skip
        and rt_settings.raster_covered_shade
        and t_val == 3
        and env_map is None
        and not any(
            (
                merged.get("tri_has_extent", False),
                (
                    merged.get("bez_has_visible", False)
                    and merged.get("bez_has_nondegenerate_edges", False)
                ),
            )
        )
    )

    samples = max(1, int(samples_per_pixel))
    # The path tracer always runs at aa == 1 (jittered samples are the AA), so
    # this is ``samples`` there; on the deterministic in-place-AA route it
    # folds the ``aa^2`` sub-pixel average into the per-pixel sample count.
    samples_eff = samples * (kernel_aa * kernel_aa)

    det_frag = policy.fragment_shading
    frag_flag = int(det_frag)
    shadow_flag = policy.shadow_mode
    sheet_route = analytic_raster
    # Composed custom fragment-shader pipelines injected into the shade kernel as
    # a flat ti.template() tuple; empty () keeps the built-in / vertex-shaded
    # kernel specialization unchanged (see shading_taichi._run_frag_pipeline).
    # A non-empty ``frag_scatters`` tuple switches the monolithic shade kernel's
    # bounce block to per-material scatter dispatch (custom ray bouncing); it is
    # only assembled when a pipeline in *this* scene overrides bouncing, so an
    # ordinary scene keeps the byte-identical built-in bounce block (empty ()).
    if det_frag or samples > 1:
        from algan.rendering.shaders.fragment_shaders import (
            build_frag_pipelines,
            build_frag_scatters,
        )

        # Narrowed to the pipelines THIS batch's primitives carry. The registry
        # behind them is process-global and append-only, so handing over all of
        # it would specialize this render's shade kernel on every pipeline the
        # process ever registered -- a scene with no custom shader at all would
        # compile its own uncached kernel variant just because some earlier
        # scene had one. The path tracer evaluates the same pipelines at its
        # hits and takes the same scatter tuple: a custom scatter is a delta
        # continuation there (pt_shade), not a refusal.
        batch_pids = _batch_user_pipeline_ids(merged)
        frag_pipelines = build_frag_pipelines(batch_pids)
        frag_scatters = build_frag_scatters(batch_pids) if policy.custom_scatter else ()
    else:
        frag_pipelines = ()
        frag_scatters = ()
    wavefront_policy = policy.wavefront
    refraction_flag = (
        int(wavefront_policy.refraction) if wavefront_policy is not None else 0
    )
    ior_stack_flag = (
        int(wavefront_policy.ior_stack) if wavefront_policy is not None else 0
    )
    # Environment map: append its texels to the shared texture buffer (the
    # merged dict is shallow-copied -- it is cached across batches) and, when
    # its ambient lighting is enabled, its SH irradiance as an extra light row.
    if det_frag or samples > 1:
        # The path tracer packs lights exactly as the deterministic
        # per-fragment route does: its next-event estimation reads the same
        # rows through the same ``_light_eval`` radiometry. The env-SH row
        # stays deterministic-only on purpose: the path tracer integrates
        # the map for real (CDF next-event estimation + escaping rays), and
        # the SH irradiance row would light every diffuse vertex a second
        # time.
        light_device = torch.device("cpu")
        light_pos_host, light_col_host, num_lights = _pack_lights(
            light_sources, num_frames, light_device
        )
        if (
            env_map is not None
            and samples <= 1
            and getattr(scene, "environment_ambient", True)
        ):
            light_pos_host, light_col_host, num_lights = _append_env_sh_light(
                light_pos_host,
                light_col_host,
                num_lights,
                env_source,
                float(getattr(scene, "environment_intensity", 1.0)),
                light_device,
            )
        with memory.scope(
            "persistent_inputs",
            light_pos_cells=light_pos_host.numel(),
            light_col_cells=light_col_host.numel(),
        ):
            light_pos = _arena_copy(memory, light_pos_host)
            light_col = _arena_copy(memory, light_col_host)
    else:
        # Deterministic, fragment shading off: tiny placeholders for the
        # (compiled-out) material/light kernel args.
        with memory.scope("persistent_inputs", light_route="placeholder"):
            light_pos = memory.get_tensor((1, 1, 3), torch.float32)
            light_col = memory.get_tensor((1, 1, 3), torch.float32)
        light_pos.zero_()
        light_col.zero_()
        num_lights = 0

    # The shadow-light ceiling is the one truncation the host can see without
    # asking a kernel: max_shadow_lights is the length of a ``ti.Vector`` and
    # therefore compile-time, so every light slot past it is simply never
    # written and the surplus lights render lit-but-shadowless. Counted per
    # batch, since a light spawning mid-scene can push a later batch over a cap
    # the first ones sat under. Each RectAreaLight emitter sample already
    # occupies its own row here (``_pack_lights`` expands them), which is why
    # the count is of light SLOTS rather than of the author's lights.
    #
    # Deterministic renders only. The path tracer does not sum light rows at a
    # lit surface at all -- it samples the next-event table, which has no cap --
    # and since roadmap 6a-bis its authored-appearance branch samples its rows
    # too, so at ``samples > 1`` there is nothing here to truncate. Firing it
    # anyway told a user already rendering with the path tracer to render with
    # the path tracer. (``pt_authored_light_sampling = "off"`` puts the cap back
    # on authored materials there and is deliberately not reported: it is an A/B
    # arm, not a configuration to warn about.)
    if shadow_flag and num_lights > max_shadow_lights and samples <= 1:
        record_truncation(
            "shadow_lights",
            int(num_lights) - max_shadow_lights,
            cap=max_shadow_lights,
        )

    # Frame counts of the windows that were actually launched. They differ from
    # the requested window whenever a chunk had to be sub-divided below, and the
    # batching loop's memory model needs the difference: it measures the arena's
    # high-water mark over this call and would otherwise credit a sub-divided
    # chunk's (smaller) peak to the frame count it planned, under-reading the
    # per-frame cost and planning the same over-large chunk again.
    launched_frames = []

    def rewind_to(pointers):
        """Reclaim chunk state while preserving explicitly published batch data.

        The first chunk may build raster tables before it allocates coverage.
        A deferred BVH is published before that coverage, or after unwinding
        the chunk and requesting a restart. Both cache on ``merged`` and record
        their reverse boundary through ``_retain_persistent``. Retaining that
        boundary does not retain intervening coverage/tile allocations.
        """
        forward, reverse = pointers
        if merged is not None:
            retained = merged.get(ARENA_RETAINED_REVERSE_POINTER)
            if retained is not None:
                reverse = min(reverse, retained)
        memory.set_pointers((forward, reverse))

    def render_chunk(start, end):
        nonlocal tri_bvh, bez_bvh
        # The Monte Carlo kernels launch one thread per (frame, pixel,
        # sample) path; keep the flattened index within int32 range. (The
        # deterministic kernels loop the aa^2 sub-pixels serially per pixel, so
        # only the Monte Carlo path multiplies the thread count by the samples.)
        if samples > 1 and (end - start) * width * height * samples_eff >= 1 << 31:
            logger.log(PERF, f"Reducing the frame batch to fit memory: {start}:{end}")
            if end - start <= 1:
                raise OutOfRenderMemory(
                    "samples_per_pixel * resolution exceeds the ray tracer's "
                    "per-launch path budget (2^31). Please lower the sample "
                    "count, resolution or anti-alias level."
                )
            middle = (start + end) // 2
            return render_chunk(start, middle) + render_chunk(middle, end)
        entry_pointers = memory.get_pointers()
        # Rolled back beside the arena pointers when a chunk is discarded for
        # memory: the failed attempt's truncations describe frames that are
        # about to be re-rendered, and counting both attempts would double them.
        entry_truncations = snapshot_truncations()
        entry_path_samples = snapshot_path_samples()
        try:
            post_tonemap = is_post_process_tonemap_enabled()
            if rt_settings.linear_color_space and not post_tonemap:
                # The in-composite route's frame buffer is uint8, and linear
                # values must never be stored in 8 bits: linear 0.033 -- an
                # ordinary dark grey -- quantises to byte 8, and the darks fall
                # apart. Display encoding is *why* an 8-bit buffer holds
                # encoded values. Rather than silently render a crushed frame,
                # say so; post_process_tonemap is on by default, so reaching
                # this means someone turned it off deliberately.
                rt_settings.report_unsupported_features(
                    "linear_color_space needs the float HDR frame buffer, which "
                    "post_process_tonemap provides; with it off the buffer is "
                    "uint8 and linear values would be quantised to 8 bits, "
                    "crushing the darks. Leave post_process_tonemap on (the "
                    "default), or set "
                    "SETTINGS.raytracing.set(linear_color_space=False)."
                )
            out_dtype = torch.float32 if post_tonemap else torch.uint8
            # The denoiser (public ``denoise``, default on) applies only to
            # path-traced output on the float HDR buffer -- the deterministic
            # renderer has no noise, and the byte buffer holds
            # display-encoded values the filter was not trained on. Resolved
            # here, once per chunk: ``get_denoiser`` memoizes the loaded
            # network per process and degrades to None (denoise off, one
            # warning) when the weights cannot be had.
            denoiser = None
            if samples > 1 and rt_settings.denoise:
                if post_tonemap:
                    from algan.rendering.denoise import get_denoiser

                    denoiser = get_denoiser(device)
                else:
                    logger.log(
                        PERF,
                        "denoise is on but post_process_tonemap is off: the "
                        "frame buffer is uint8, so denoising is skipped.",
                    )
            # Drivers are element counts, not the resolution: the buffers scale
            # linearly, so keying on width/height would make the table useless
            # at any resolution the corpus happened not to cover.
            with memory.scope(
                "frame_buffers",
                out_cells=(end - start) * width * height * C_out,
                dtype=str(out_dtype),
            ):
                out = memory.get_tensor((end - start, width * height, C_out), out_dtype)
                _prefill_background(
                    out,
                    background,
                    start - time_start,
                    device,
                    background_frames=time_end - time_start,
                )
                if env_meta is not None and sheet_route:
                    # Background-as-final-sheet (DESIGN_sheet_resolve.md
                    # §4.5): under the sheet route the frame buffer IS the
                    # background stage, so an env-mapped batch prefills the
                    # map per (frame, pixel) and the resolve hands its
                    # leftover weight to the composite instead of sampling
                    # the map at retire. Empty pixels are then final with no
                    # resolve launch at all.
                    from algan.rendering.raytracing.sheet_resolve_taichi import (
                        env_background_prefill,
                    )

                    eo, ew, eh, ei = env_meta
                    env_background_prefill(
                        int(end - start),
                        int(width),
                        int(height),
                        int(start),
                        0.5,
                        0.5,
                        float(width // 2),
                        float(height // 2),
                        cam_origin,
                        sp,
                        pbx,
                        pby,
                        int(eo),
                        int(ew),
                        int(eh),
                        float(ei),
                        merged["textures"],
                        out,
                    )
                accum = accum_odd = None
                if samples > 1:
                    # f32 per-pixel sample sums, averaged by finalize_samples.
                    # Its own scope: the accumulator is float32 whatever the
                    # frame buffer's dtype is, so charging it under the frame
                    # buffer's dtype key would claim a dependency that does
                    # not exist (and leave it unmeasured on the byte route).
                    with memory.scope(
                        "frame_accum", accum_cells=(end - start) * width * height * 5
                    ):
                        accum = memory.get_tensor(
                            (end - start, width * height, 5), torch.float32
                        )
                    accum.zero_()
                    # Local, like ``path_trace_render`` below: the path
                    # tracer's modules are imported at dispatch so a
                    # deterministic render never pays for them.
                    from algan.rendering.raytracing.path_tracer import (
                        pt_adaptive_active,
                    )

                    if pt_adaptive_active(samples_eff):
                        # Adaptive sampling's stopping-rule buffer: the RGB of
                        # the ODD sample indices plus the count of stochastic
                        # samples, which is what the per-pixel rule needs
                        # (roadmap section 2). Allocated only when the
                        # mechanism runs, so ``pt_error_target = 0`` charges
                        # the memory model exactly what it charged before --
                        # which is what keeps that arm's frame batching, and
                        # therefore its output, identical.
                        with memory.scope(
                            "frame_accum_odd",
                            accum_cells=(end - start) * width * height * 4,
                        ):
                            accum_odd = memory.get_tensor(
                                (end - start, width * height, 4), torch.float32
                            )
                        accum_odd.zero_()
                aovs = aov_bg = None
                if denoiser is not None:
                    # The denoiser's guides: per-pixel sample sums of albedo,
                    # normal and background weight (path_trace_render fills
                    # them), plus a snapshot of the prefilled background the
                    # weight is folded with -- the kernel never knows the
                    # background's colors, only how much of each path reached
                    # it.
                    with memory.scope(
                        "denoise_aovs",
                        aov_cells=(end - start) * width * height * 12,
                    ):
                        aovs = tuple(
                            memory.get_tensor(
                                (end - start, width * height, 3), torch.float32
                            )
                            for _ in range(3)
                        )
                        aov_bg = memory.get_tensor(
                            (end - start, width * height, 3), torch.float32
                        )
                    for tensor in aovs:
                        tensor.zero_()
                    aov_bg.copy_(out[:, :, :3])
                    aov_bg /= 255.0
            # Coplanar layer order: circuits < triangles < PN patches.
            layer_offset_triangles = float(merged["num_circuits"])
            if samples > 1:
                from algan.rendering.raytracing.path_tracer import (
                    path_trace_render,
                )

                with memory.temp():
                    path_trace_render(
                        memory=memory,
                        tri_bvh=tri_bvh,
                        bez_bvh=bez_bvh,
                        merged=merged,
                        cam_origin=cam_origin,
                        screen_point=sp,
                        pixel_basis_x=pbx,
                        pixel_basis_y=pby,
                        pixel_world_scale=pixel_world_scale,
                        time_start=start,
                        time_end=end,
                        width=width,
                        height=height,
                        half_screen_w=float(width // 2),
                        half_screen_h=float(height // 2),
                        layer_offset_triangles=layer_offset_triangles,
                        has_tri=has_tri,
                        has_bez=has_bez,
                        light_pos=light_pos,
                        light_col=light_col,
                        num_lights=num_lights,
                        frag_pipelines=frag_pipelines,
                        frag_scatters=frag_scatters,
                        shadows=int(policy.shadows),
                        max_bounces=int(max_bounces),
                        near_clip=near_clip,
                        far_clip=far_clip,
                        transparent=transparent_background,
                        samples=samples_eff,
                        env_meta=env_meta,
                        aovs=aovs,
                        out=out,
                        accum=accum,
                        accum_odd=accum_odd,
                    )
                finalize_samples(
                    samples_eff,
                    1 if transparent_background else 0,
                    t_val,
                    float(tonemap_exposure),
                    accum,
                    out,
                )
                if denoiser is not None:
                    # Denoise the finalized linear HDR color in place,
                    # between the estimator and everything display-facing
                    # (tonemap, FXAA, user post-processes). The float buffer
                    # holds linear radiance at byte scale; alpha/coverage
                    # channels pass through untouched. The guides: sample
                    # sums divided down, the background weight folded with
                    # the prefill snapshot.
                    inv_spp = 1.0 / float(samples_eff)
                    shape = (end - start, height, width, 3)
                    albedo = ((aovs[0] + aovs[2] * aov_bg) * inv_spp).view(shape)
                    normal = (aovs[1] * inv_spp).view(shape)
                    color = (out[:, :, :3] * (1.0 / 255.0)).view(shape)
                    # Under adaptive sampling the path tracer knows which
                    # pixels took a random decision (the stochastic sample
                    # count in accum_odd's last column); every other pixel
                    # is exact and the filter passes it through untouched,
                    # skipping tiles that hold none (Denoiser.__call__).
                    stochastic = None
                    if accum_odd is not None:
                        stochastic = (accum_odd[:, :, 3] > 0.0).view(shape[:3])
                    denoised = denoiser(color, albedo, normal, stochastic)
                    out[:, :, :3] = (
                        denoised.reshape(end - start, width * height, 3) * 255.0
                    )
            else:
                # col_row/gen/layer metadata, AA accumulation and every tile
                # buffer are wavefront-only. Release them before post
                # processing so the two phases share the same temporary arena
                # range (the batch estimator models max(wavefront, post), not
                # their sum).
                if not sparse_batch_empty:
                    with memory.temp():
                        raytrace_render_wavefront(
                            tri_bvh,
                            bez_bvh,
                            merged,
                            cam_origin,
                            sp,
                            pbx,
                            pby,
                            pixel_world_scale,
                            int(start),
                            int(end),
                            int(width),
                            int(height),
                            float(width // 2),
                            float(height // 2),
                            layer_offset_triangles,
                            has_tri,
                            has_bez,
                            int(max_bounces),
                            light_pos,
                            light_col,
                            int(num_lights),
                            frag_flag,
                            frag_pipelines,
                            frag_scatters,
                            shadow_flag,
                            refraction_flag,
                            ior_stack_flag,
                            1 if transparent_background else 0,
                            memory,
                            out,
                            kernel_aa,
                            lights_extended=lights_extended,
                            env_meta=env_meta,
                            near_clip=near_clip,
                            far_clip=far_clip,
                            analytic_raster=analytic_raster,
                            policy=wavefront_policy,
                        )
            frames = out.view(end - start, height, width, C_out)
            # Post-processing launches Taichi kernels (the tonemap in particular)
            # from Taichi's own CUDA pool. The render just accumulated torch
            # reserved-but-free blocks that Taichi cannot draw on; hand them back
            # to the driver *before* the tonemap when free VRAM is low, so the
            # launch has room instead of OOMing into the split-retry round-trip.
            # Gated internally on free-memory pressure -- a no-op when memory is
            # plentiful (the common case).
            ensure_render_headroom(device)
            frames = post_process_frames(
                memory,
                frames,
                anti_alias_level=post_aa,
                post_processes=list(post_processes),
                apply_fxaa=scene.video_settings.fxaa,
                premultiplied_over=scene.premultiplied_over,
            )
            rewind_to(entry_pointers)
            launched_frames.append(end - start)
            return [frames]
        except _DeferredBVHRequired as exc:
            # The resolve discovered a continuation after merge-time deferral.
            # Unwind coverage and all tile/iteration state before publishing
            # batch-lived trees. Otherwise retaining the trees also pins every
            # reverse allocation made between the batch floor and this tile.
            rewind_to(entry_pointers)
            restore_truncations(entry_truncations)
            restore_path_samples(entry_path_samples)
            traceback.clear_frames(exc.__traceback__)
            _publish_bvhs_at_chunk_boundary(merged, memory)
            tri_bvh, bez_bvh = merged["tri_bvh"], merged["bez_bvh"]
            # Refill the output as well as rediscovering coverage: earlier tiles
            # may already have composited, and uncovered pixels may be tonemapped.
            # Successful publication clears both deferred flags, so this restart
            # can happen only once per prepared batch (not once per ray tile).
            return render_chunk(start, end)
        except (InsufficientMemoryException, RuntimeError) as exc:
            # A Taichi kernel launch (e.g. the post-process tonemap) exhausts
            # VRAM as a plain RuntimeError from its own allocator, not a torch
            # OOM; recognise it so the same rewind + release_torch_memory + split retry
            # recovers it. Any non-OOM RuntimeError is a real error -- re-raise.
            if not isinstance(exc, InsufficientMemoryException) and not is_cuda_oom(
                exc
            ):
                raise
            logger.log(PERF, f"Reducing the frame batch to fit memory: {start}:{end}")
            rewind_to(entry_pointers)
            restore_truncations(entry_truncations)
            restore_path_samples(entry_path_samples)
            # All this stuff is necessary to free local variables assigned during the previous render attempt.
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.clear_frames(exc_traceback)
            # traceback.print_tb(exc_traceback)
            # exc_traceback.tb_next.tb_frame.clear()
            # Release the failed allocation (e.g. the wavefront's large per-ray
            # state) so it doesn't fragment/block the smaller retry.
            release_torch_memory(force_gc=False)
            if end - start <= 1:
                raise OutOfRenderMemory(
                    "Insufficient memory to ray trace a single frame. "
                    "Please lower the resolution or anti-alias level."
                ) from None
            middle = (start + end) // 2
            return render_chunk(start, middle) + render_chunk(middle, end)

    chunks = render_chunk(time_start, time_end)
    # Whatever the batch truncated is reported now -- once the frames exist and
    # before the caller can act on them -- and the running totals are grafted
    # onto the plan the Scene hands back, so a script can assert on them
    # without parsing logs.
    report_truncations()
    scene.last_render_plan = attach_render_stats(attach_truncations(plan))
    if memory is not None and launched_frames:
        memory.last_launch_frames = max(launched_frames)
    if len(chunks) == 1:
        return chunks[0]
    return torch.cat(chunks, 0)


def _run_wavefront_tiles(
    memory,
    out,
    *,
    n,
    width,
    height,
    time_start,
    transparent,
    aa_level,
    pool_ratio,
    primary_per_tile,
    cam_origin,
    screen_point,
    pixel_basis_x,
    pixel_basis_y,
    half_screen_w,
    half_screen_h,
    max_bounces,
    near_clip,
    run_tile,
    auto_extra_slot_bytes=0,
    auto_extra_primary_bytes=0,
    auto_fixed_bytes=0,
    gen_fused=False,
    global_hits=True,
    analytic_raster=False,
    sca_width=SCA_WIDTH_PLAIN,
):
    """Run deterministic-wavefront screen tiles with a shared split pool.

    ``run_tile(tile_start, tn_primary, pool, state, rs_pix, pix_accum,
    rs_alloc)`` supplies the variant-specific traverse/shade iteration. The
    first ``tn_primary`` slots hold primary rays and every spawned continuation
    atomically appends to the shared remainder of the pool. ``rs_alloc`` is a
    two-word counter: next free slot and overflow flag.

    Pool exhaustion is never accepted as a rendering approximation. An
    overflowing attempt is discarded before compositing and retried with fewer
    primaries based on measured demand, retaining the same pool capacity. An
    attempt scope restores both arena ends on every exit, including exceptions
    during allocator readback or compositing.
    """
    t_val = _get_tonemap_t_val()
    i32 = torch.int32
    f32 = torch.float32
    # Placeholder covered list: the composite kernel keeps its covered-compact
    # arguments (the sparse sheet route uses its own composite), but every
    # tile here runs the full non-compacted pass.
    covered_dummy = torch.zeros(1, dtype=i32, device=out.device)
    aa = max(1, int(aa_level))
    do_aa = aa > 1
    inv_aa = 1.0 / aa

    # Constant primary-ray init rows (rs_sca / rs_int). When there is no split
    # pool and no near clip these are identical for every primary, so they are
    # filled with coalesced broadcast copies here rather than by the strided
    # per-ray stores in the memory-bound generate kernel (``write_const == 0``
    # tells the kernel the host already filled them). base_dist (rs_sca[4]) is
    # 0 without a near clip; _ACTIVE == 0 (rs_int cols 1-3 are all zero).
    const_fill = pool_ratio == 1 and near_clip <= 0.0
    if const_fill:
        # Trailing zeros keep the broadcast copy below shape-safe if the
        # nested-IOR stack ever widens rs_sca under a const-fill-eligible
        # batch. Inert today: the stack gate implies refraction_flag, which
        # forces pool_ratio > 1 and therefore const_fill == 0 (see the
        # ior_stack_flag definition in render_batch_raytraced).
        sca_init = torch.tensor(
            [1.0, 0.0, 1e30, -1e30, 0.0, 1.0, 1.0]
            + [0.0] * (sca_width - SCA_WIDTH_PLAIN),
            dtype=f32,
            device=out.device,
        )
        int_init = torch.tensor(
            [int(max_bounces), 0, 0, 0], dtype=i32, device=out.device
        )
    aa_accum = None
    if do_aa:
        aa_accum = memory.get_tensor((n, 5 if transparent else 4), f32)
        aa_accum.zero_()

    # Adaptive tile sizing (after aa_accum so free-bytes accounting sees it).
    # The allocator's two int32 words are fixed per tile rather than per ray.
    primary_per_tile = _auto_primary_per_tile(
        memory,
        pool_ratio,
        primary_per_tile,
        auto_extra_slot_bytes,
        auto_extra_primary_bytes,
        auto_fixed_bytes + ALLOC_WIDTH * torch.int32.itemsize,
        # This batch's own rs_sca width, not the nested-IOR setting's: the
        # rows allocated below are `sca_width` wide, so that is what the tile
        # has to be charged for.
        sca_width,
    )
    primary_capacity = min(max(1, int(primary_per_tile)), max(1, int(n)))
    shared_pool_capacity = _shared_pool_slots(
        primary_capacity, primary_per_tile, pool_ratio, analytic_raster
    )

    # Remember a successful reduced tile size after an overflow so every
    # subsequent tile does not repeat the same failed first attempt. The pool
    # itself remains at ``shared_pool_capacity`` for all splitting tiles, so a
    # smaller final tile automatically receives the otherwise-unused slots.
    learned_primary_cap = primary_capacity

    for si in range(aa):
        for sj in range(aa):
            jx = (si + 0.5) * inv_aa if do_aa else 0.5
            jy = (sj + 0.5) * inv_aa if do_aa else 0.5
            tile_start = 0

            while tile_start < n:
                remaining = n - tile_start
                attempt_primary = min(learned_primary_cap, remaining)
                # Split-free renders do not need a shared reserve; keeping their
                # final tile exact avoids scanning unused slots. Splitting
                # renders retain the full fixed pool across retries and tiles.
                pool = shared_pool_capacity if pool_ratio > 1 else attempt_primary

                while True:
                    with memory.temp(clear_persist=True):
                        # Per-ray state for one tile: ``pool`` slots plus
                        # ``attempt_primary`` per-primary rows. Calibrated as
                        # unit coefficients (bytes per slot, per primary, and
                        # fixed per tile) rather than as a peak -- under
                        # wavefront_tile_auto the tile is sized from whatever
                        # arena is free, so its peak would measure the arena.
                        with memory.scope(
                            "wavefront_state",
                            pool=pool,
                            primary=attempt_primary,
                            global_hits=int(global_hits),
                        ):
                            state = _alloc_wavefront_state(
                                memory, pool, sca_width, global_hits=global_hits
                            )
                            rs_pix = memory.get_tensor((pool,), i32)
                            pix_accum = memory.get_tensor((attempt_primary, 7), f32)
                            # [0] next free shared slot, [1] overflow flag,
                            # [2] the compositing-ceiling truncation counter
                            # (ALLOC_* in wavefront_kernels_taichi). The classic
                            # generation kernel initialises the first two;
                            # everything past them is zeroed below, since only
                            # atomic adds ever touch it.
                            rs_alloc = memory.get_tensor((ALLOC_WIDTH,), i32)
                        (
                            rs_ro,
                            rs_rd,
                            rs_acc,
                            rs_sca,
                            rs_int,
                            rs_kt,
                            rs_kl,
                            rs_ka,
                            rs_kb,
                            rs_kp,
                            rs_kf,
                        ) = state

                        if sca_width != SCA_WIDTH_PLAIN:
                            # Nested-IOR stack columns
                            # (DESIGN_mesh_identity_open.md §H): zeroed once
                            # per tile on the host so any path that forgets to
                            # write a stack degrades to today's air-outside
                            # behaviour instead of reading arena noise. Covers
                            # primaries and pool slots alike, and costs
                            # O(pool) floats against a tile that runs orders of
                            # magnitude more work. The generate kernel is left
                            # alone: its write_const block only runs when
                            # pool_ratio == 1, which the stack gate excludes.
                            rs_sca[:, _SCA_IOR_DEPTH:].zero_()

                        if gen_fused:
                            pix_accum.zero_()
                            rs_alloc.zero_()
                        else:
                            # The generation kernel writes only the allocator's
                            # own two words; the truncation counters past them
                            # are accumulate-only and must start at zero.
                            rs_alloc[ALLOC_OVERFLOW + 1 :].zero_()

                            # rs_acc and pix_accum start all-zero, and the
                            # constant rs_sca / rs_int primary init rows are
                            # filled here for the split-free, near-clip-free
                            # case. Doing this as contiguous memsets /
                            # broadcast copies is far cheaper than the strided
                            # per-ray stores the generate kernel otherwise
                            # does through the AoS [ray, channel] layout
                            # (memory-bound kernel); byte-identical -- same
                            # values, just coalesced.
                            rs_acc.zero_()
                            pix_accum.zero_()
                            if const_fill:
                                rs_sca[:attempt_primary].copy_(sca_init)
                                # rs_int is 5 wide; generate only wrote cols
                                # 0-3 (col 4 is the legacy sorted-path
                                # "drained" field it never touched), so fill
                                # only 0-3 to leave col 4 exactly as before --
                                # byte-identical.
                                rs_int[:attempt_primary, :4].copy_(int_init)
                            wavefront_generate_rays(
                                cam_origin,
                                screen_point,
                                pixel_basis_x,
                                pixel_basis_y,
                                int(time_start),
                                int(width),
                                int(height),
                                float(half_screen_w),
                                float(half_screen_h),
                                int(max_bounces),
                                int(tile_start),
                                int(attempt_primary),
                                float(jx),
                                float(jy),
                                float(near_clip),
                                0 if const_fill else 1,
                                rs_ro,
                                rs_rd,
                                rs_acc,
                                rs_sca,
                                rs_int,
                                rs_pix,
                                pix_accum,
                                rs_alloc,
                            )

                        run_tile(
                            tile_start,
                            attempt_primary,
                            pool,
                            state,
                            rs_pix,
                            pix_accum,
                            rs_alloc,
                        )
                        alloc = _read_tile_alloc(rs_alloc)
                        overflow = pool_ratio > 1 and alloc[ALLOC_OVERFLOW] != 0
                        if overflow:
                            if attempt_primary <= 1:
                                raise OutOfRenderMemory(
                                    "A single pixel's deterministic ray tree "
                                    f"exceeded the shared wavefront pool of {pool} "
                                    "slots. Lower MAX_BOUNCES / transparency "
                                    "complexity, or increase WAVEFRONT_TILE_RAYS."
                                )
                            next_primary = _overflow_retry_primary(
                                attempt_primary, alloc[ALLOC_NEXT], pool
                            )
                            _WAVEFRONT_POOL_RETRIES[0] += 1
                            logger.log(
                                PERF,
                                "Wavefront continuation pool overflowed for tile "
                                f"{tile_start}:{tile_start + attempt_primary}; "
                                f"retrying with {next_primary} primaries and the "
                                f"same {pool}-slot pool",
                            )
                            learned_primary_cap = min(learned_primary_cap, next_primary)
                            attempt_primary = next_primary
                            continue
                        # Past the retry: this attempt is the one that composites,
                        # so its counters are the ones that count.
                        _record_tile_truncations(alloc, pool)

                        if do_aa:
                            wf_composite_accum_aa(
                                int(time_start),
                                int(width),
                                int(height),
                                1 if transparent else 0,
                                int(tile_start),
                                pix_accum,
                                out,
                                aa_accum,
                            )
                        else:
                            wf_composite_accum(
                                int(time_start),
                                int(width),
                                int(height),
                                1 if transparent else 0,
                                int(tile_start),
                                pix_accum,
                                t_val,
                                float(rt_settings.tonemap_exposure),
                                0,
                                0,
                                covered_dummy,
                                0,
                                out,
                            )
                        tile_start += attempt_primary
                        break

    if do_aa:
        wf_finalize_aa(
            int(width),
            int(height),
            1 if transparent else 0,
            float(inv_aa * inv_aa),
            t_val,
            float(rt_settings.tonemap_exposure),
            aa_accum,
            out,
        )


def raytrace_render_wavefront(
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
    max_bounces,
    light_pos,
    light_col,
    num_lights,
    frag_flag,
    frag_pipelines,
    frag_scatters,
    shadow_flag,
    refraction_flag,
    ior_stack_flag,
    transparent,
    memory,
    out,
    aa_level=1,
    lights_extended=False,
    env_meta=None,
    near_clip=0.0,
    far_clip=0.0,
    analytic_raster=False,
    *,
    policy=None,
):
    """Wavefront orchestration for the general triangle/PN/bezier path.

    Persistent continuation state is stage-split in global memory; arena-backed
    index buffers and a filter kernel compact rays between host iterations. Hit records are different:
    traversal writes one exact-size ``[num_active, kbuf]`` transient event
    batch, shade consumes it immediately, and the arena range is then reused.
    No pool-wide K-buffer is attached to secondary radiance ray slots. The
    persistent scalar state carries ``base_dist`` for Bezier border widths
    across bounces.

    ``frag_flag``/``shadow_flag`` select the deterministic per-fragment shading
    and opacity-weighted hard-shadow paths (compile-time templates of the shade
    kernel); ``light_pos``/``light_col`` feed both.

    ``frag_scatters`` is the per-pipeline custom ray-continuation (scatter) tuple
    (empty when no scene pipeline overrides bouncing); a non-empty tuple switches
    the monolithic shade kernel's bounce block from the built-in
    opacity/reflectivity/Fresnel logic to per-material scatter dispatch
    (``_run_frag_scatter``), so users can customise reflection / refraction /
    pass-through. Empty keeps the built-in bounce block byte-identical.

    ``refraction_flag`` enables simultaneous reflection + refraction (glass): the
    shade kernel SPLITS such a ray, continuing the reflected branch in place and
    spawning the refracted branch into a free pool slot. The pool is therefore
    over-allocated by ``pool_ratio`` (only when refraction / custom scatter is on)
    -- it holds ``primary_per_tile`` one-per-pixel rays plus spare slots for
    split branches, at fixed total memory (fewer pixels per tile instead of
    bigger per-ray state). Each ray commits into a shared per-pixel accumulator
    (``pix_accum``) on termination, so a pixel's reflected and refracted branches
    sum.

    When fragment shading is active, the monolithic ``wavefront_shade`` kernel
    below is the shade architecture: it handles custom scatter and
    normal-mapped lighting, and on the built-in materials it drains up to kbuf
    hits per launch. (A Cycles-style *sorted* alternative -- rays suspended at
    their material events and shaded by one kernel per material bucket -- was
    measured slower for exactly that reason and has been removed.) The
    vertex-shaded path below is unaffected.
    """
    rt_settings = SETTINGS.raytracing
    # Opt-in inline stages for this function's loops (the profiler's pipeline
    # hooks cannot reach inside one function). A shared nullcontext unless the
    # profiler is installed; imported here rather than at module level because
    # profiling_utils imports this module to hook it.
    from algan.utils.profiling_utils import stage as _stage

    # rs_sca's row width for this batch. A local, not a call at each use site:
    # ``sca_width`` is also the name of the module-level helper imported from
    # wavefront_kernels_taichi, so a bare ``sca_width`` inside this function is
    # the FUNCTION, not a width (that mistake allocated ray state with a
    # function object as its column count).
    if policy is None:
        policy = _resolve_wavefront_policy(
            merged,
            analytic_raster=analytic_raster,
            aa_level=aa_level,
            near_clip=near_clip,
            refraction=refraction_flag,
            ior_stack=ior_stack_flag,
            shadow_mode=shadow_flag,
            custom_scatter=bool(frag_scatters),
            has_triangles=has_tri,
        )
    elif policy.refraction != bool(refraction_flag) or policy.ior_stack != bool(
        ior_stack_flag
    ):
        raise RuntimeError("prepared wavefront policy disagrees with ray-state flags")
    state_sca_width = policy.state_scalar_width
    shadow_context = ShadowTraceContext(
        merged,
        light_pos,
        light_col,
        num_lights,
        pixel_world_scale,
        layer_offset_triangles,
    )
    i32 = torch.int32
    f32 = torch.float32
    max_iters = max_surfaces_per_ray + max_bounces * 2 + 4
    n = (time_end - time_start) * width * height

    # Compile-time walk selector: the merge builds either all-classic or
    # all-refit trees for a batch (see scene_builder._build_accel), so the
    # tree object's type is the authority -- never the live toggle, which the
    # user may have flipped since this batch was merged/prewarmed.
    from algan.rendering.raytracing.refit_bvh import RefitBVH

    bvh_refit = 1 if isinstance(tri_bvh, RefitBVH) else 0

    pool_ratio = policy.pool_ratio
    primary_per_tile = max(1, policy.tile_rays // pool_ratio)
    mem_trim = int(policy.memory_trim)
    if mem_trim:
        _MEM_TRIM_ENGAGED[0] += 1
        t_bvh = merged["tri_bvh_t"]
        a_pos, a_norm = merged["tri_pos_t"], merged["tri_norm_t"]
        a_mat, a_matid = merged["tri_mat_t"], merged["tri_mat_id_t"]
        a_uvs, a_meta = merged["tri_uvs_t"], merged["tri_tex_meta_t"]
        col_row_arr = merged["tri_col_row"]
    else:
        t_bvh = tri_bvh
        a_pos, a_norm = merged["tri_pos"], merged["tri_norm"]
        a_mat, a_matid = merged["tri_mat"], merged["tri_mat_id"]
        a_uvs, a_meta = merged["tri_uvs"], merged["tri_tex_meta"]
        with memory.scope("batch_metadata", col_row_placeholder=1):
            col_row_arr = memory.get_tensor((1,), i32)
        col_row_arr.zero_()
    opaque_closest = int(policy.opaque_closest)
    opaque_prepass = int(policy.opaque_prepass)
    tri_pids = policy.triangle_pipeline_mask
    _FRAG_PID_LAST["tri"] = tri_pids
    # Settings were resolved during preparation; still reject missing data or
    # incompatible state instead of silently executing a different AA route.
    use_raster = bool(
        analytic_raster
        and merged.get("tri_frame_valid") is not None
        and (merged["num_triangles"] > 0 or merged["num_circuits"] > 0)
        and not mem_trim
        and not frag_scatters
        and near_clip <= 0.0
        and max(1, int(aa_level)) <= 1
    )
    if analytic_raster and not use_raster:
        raise RuntimeError(
            "Analytic raster AA was selected before allocation, but the "
            "wavefront route rejected the batch."
        )
    env_active = env_meta is not None and int(env_meta[1]) > 0
    # The sheet route is the only raster resolve (the dense fragment walk is
    # deleted): env-mapped batches prefill the frame buffer in render_chunk,
    # the resolve stays linear, and an in-kernel tonemap runs in the
    # composite + the uncovered-pixel finalize (DESIGN_sheet_resolve.md §5).
    sparse_coverage = use_raster

    # Fused primary-ray generation (settings.wf_gen_fused): the tile's first
    # traverse generates its rays in-kernel and the first shade uses the
    # implicit initial state, skipping the standalone generate pass. Only for
    # split-free (one slot per pixel, so pix == r), near-clip-free (implicit
    # base_dist == 0) renders on the one-sample-per-pixel AA path (fixed
    # 0.5/0.5 jitter). Everything else keeps the classic generate kernel.
    def _ensure_bvhs():
        # Deferred-BVH batch (scene_builder._finalize_bvhs): build the real
        # trees and rebind everything derived from the placeholders. Deferral
        # implies mem_trim was inactive at merge, so t_bvh is plain tri_bvh.
        nonlocal tri_bvh, bez_bvh, t_bvh, bvh_refit
        _publish_deferred_bvhs(merged, memory)
        tri_bvh = merged["tri_bvh"]
        bez_bvh = merged["bez_bvh"]
        t_bvh = tri_bvh
        bvh_refit = 1 if isinstance(tri_bvh, RefitBVH) else 0

    if (merged.get("bvh_deferred") or merged.get("bvh_rehome_pending")) and (
        shadow_flag != 0 or not use_raster
    ):
        # Runtime routing needs the trees after all: primary shadows trace
        # them from the sheet resolve's event queue (raster_shadow_trace),
        # and a batch that fell back to classic primary traversal (near
        # clip, in-place AA, flipped toggles, ...) walks them for every
        # primary ray.
        _ensure_bvhs()

    gen_fused = policy.fused_generation
    # Fixed render metadata and ray-generation scalars, paid once per batch.
    # Keep these separate from the raster precompute tables so the latter's
    # per-(frame, primitive) coefficient is not fitted through a constant.
    with memory.scope("batch_metadata"):
        # Always the real four values, even where primary generation is not
        # fused into traverse: the traverse kernel rebuilds each pixel's
        # PRIMARY ray from gen_meta[2:] to convert its slant ranges to
        # perpendicular depth (see _axis_cos), and a zeroed placeholder there
        # yields a degenerate ray and a collapsed pixel_size. The jitter pair
        # is unused on that path (a sub-pixel offset does not move the
        # cosine), so the same four values serve both.
        gen_meta = _arena_values(
            memory, [0.5, 0.5, float(half_screen_w), float(half_screen_h)], f32
        )
    with memory.scope("batch_metadata"):
        render_metadata = allocate_render_metadata(
            memory,
            layer_offset_triangles,
            env_meta=env_meta,
            far_clip=far_clip,
            max_bounces=max_bounces,
        )

    tri_screen = None
    tri_bounds = None
    bez_bounds = None
    if use_raster:
        # Screen-space projection and bounds tables, sized
        # [batch_frames, primitives, cols]. ``batch_frames`` is the whole
        # prepared batch's frame count, not the render chunk's, so these tables
        # are the same for every chunk of a batch -- and a batch is routinely
        # rendered in several chunks (four on this project's reference scene,
        # more as the resolution rises and fewer frames fit the arena), which
        # used to rebuild them from scratch every time.
        #
        # They are therefore built once per batch, from the arena's persistent
        # end so the per-chunk forward reset does not reclaim them, and cached
        # on the merged scene (which lives for the batch). The reverse pointer
        # reached here is published alongside them so the render loop can hold
        # the arena open across chunks exactly that far and no further (see
        # RenderLoopMixin._render_primitive_batch).
        cached_tables = merged.get("_raster_tables")
        if cached_tables is not None:
            tri_screen, tri_bounds, bez_bounds = cached_tables
        else:
            tri_screen, tri_bounds, bez_bounds = _build_raster_tables(
                merged,
                memory,
                cam_origin,
                screen_point,
                pixel_basis_x,
                pixel_basis_y,
                half_screen_w,
                half_screen_h,
                width,
            )
            merged["_raster_tables"] = (tri_screen, tri_bounds, bez_bounds)
            _retain_persistent(merged, memory)

    def _scene_sort_bounds():
        """The batch scene box the Morton sort keys quantise over, as
        ``(lo[3], 1023 / extent[3])`` host floats; computed once per merged
        scene (one readback) and cached on it.
        """
        bounds = merged.get("_ray_sort_bounds")
        if bounds is None:
            scene_box = triangle_scene_bounds(merged)
            lo3 = list(scene_box.lower)
            inv3 = [
                1023.0 / max(hi_v - lo_v, 1e-12)
                for lo_v, hi_v in zip(lo3, scene_box.upper)
            ]
            bounds = (lo3, inv3)
            merged["_ray_sort_bounds"] = bounds
        return bounds

    def _sort_active_rays(active, na, rs_ro, rs_rd, rs_pix, compactor):
        """Permute the active list into (frame, octant, origin Morton) order
        (rt_settings.wf_ray_sort); see ``wavefront_ray_sort_keys``.
        """
        lo3, inv3 = _scene_sort_bounds()
        key_dtype, origin_shift = _ray_sort_key_format(
            int(time_end) - int(time_start), rs_ro.device
        )
        keys = memory.get_tensor((na,), key_dtype)
        wavefront_ray_sort_keys(
            active,
            na,
            rs_ro,
            rs_rd,
            rs_pix,
            int(width) * int(height),
            lo3[0],
            lo3[1],
            lo3[2],
            inv3[0],
            inv3[1],
            inv3[2],
            keys,
            origin_shift,
        )
        return compactor.reorder(active, torch.argsort(keys))

    def _deferred_wavefront_shadows(
        active, na, hit_f, hit_i, rs_ro, rs_rd, rs_int, rs_pix
    ):
        """Trace this drain iteration's shadow events; return the visibility
        table the shade kernel's ``deferred_shadows`` arm reads
        (``[na * kbuf, 3 * vis_lights]`` f32, all-lit where no event).
        """
        from algan.rendering.raytracing.raster_pipeline import (
            _shadow_identity_epsilons,
        )

        rows = na * kbuf
        term_mode = int(rt_settings.shadow_terminator_mode())
        ev_accept = memory.get_tensor((rows,), i32)
        ev_pos = memory.get_tensor((rows, 3), f32)
        ev_snrm = memory.get_tensor((rows, 3), f32)
        ev_fnrm = memory.get_tensor((rows, 3), f32)
        ev_frame = memory.get_tensor((rows,), i32)
        ev_msk = memory.get_tensor((rows,), i32)
        ev_toff = memory.get_tensor((rows if term_mode == 1 else 1, 3), f32)
        ev_sort = 1 if rt_settings.wf_shadow_event_sort else 0
        ev_key = memory.get_tensor((rows if ev_sort else 1,), torch.int64)
        lo3, inv3 = _scene_sort_bounds() if ev_sort else ([0.0] * 3, [1.0] * 3)
        wavefront_shadow_events(
            active,
            na,
            rs_ro,
            rs_rd,
            rs_int,
            rs_pix,
            hit_f,
            hit_i,
            a_pos,
            a_norm,
            a_uvs,
            a_meta,
            merged["textures"],
            int(merged["num_colored_triangles"]),
            a_matid,
            a_mat,
            int(mem_trim),
            int(rt_settings.shadow_sided_cull),
            term_mode,
            int(time_start),
            int(width),
            int(height),
            0,
            ev_accept,
            ev_pos,
            ev_snrm,
            ev_fnrm,
            ev_frame,
            ev_msk,
            ev_toff,
            ev_sort,
            ev_key,
            lo3[0],
            lo3[1],
            lo3[2],
            inv3[0],
            inv3[1],
            inv3[2],
        )
        vis_lights = shadow_vis_slots(num_lights)
        vis_tab = memory.get_tensor((rows, 3 * vis_lights), f32)
        vis_tab.fill_(1.0)
        acc_idx = ev_accept.nonzero(as_tuple=True)[0]
        num_events = int(acc_idx.numel())
        if num_events == 0:
            return vis_tab
        if ev_sort:
            # Trace the events in Morton order of (frame, hit position) -- the
            # key the events kernel wrote per accepted row. The ray sort puts
            # the rays' ORIGINS in order; a bounce scatters the hit points,
            # and it is at the hit points that these rays start. The result
            # rows go back through ``acc_idx`` below, so the permutation
            # cannot change a single output value.
            acc_idx = acc_idx.index_select(
                0, torch.argsort(ev_key.index_select(0, acc_idx))
            )
        sec_aa = _deferred_shadow_sample_count(
            rt_settings.effective_analytic_aa_secondary_samples(), light_col.shape[2]
        )
        ev_dp = memory.get_tensor((num_events if sec_aa > 1 else 1, 6), f32)
        ev_dp.zero_()
        shadow_vis = memory.get_tensor((num_events, max(1, int(num_lights)), 3), f32)
        shadow_vis.fill_(1.0)
        dummy_i = memory.get_tensor((1,), i32)
        dummy_i.zero_()
        identity_on = bool(rt_settings.shadow_identity_reject)
        if identity_on:
            # No source identity on this path, exactly as the inline fan had
            # none: a -1 source keeps the plain acceptance epsilon per ray.
            ev_src = memory.get_tensor((num_events,), i32)
            ev_src.fill_(-1)
            eps_self, eps_near = _shadow_identity_epsilons(merged)
        else:
            ev_src = dummy_i
            from algan.rendering.raytracing.raytrace_kernels_taichi import (
                min_hit_distance,
            )

            eps_self, eps_near = float(min_hit_distance), 0.0
        payload = _gather_shadow_payload(
            memory,
            acc_idx,
            ev_pos,
            ev_snrm,
            ev_fnrm,
            ev_frame,
            ev_msk,
            ev_dp,
            ev_toff,
            with_terminator=term_mode == 1,
        )
        shadow_context.trace(
            payload,
            t_bvh,
            bez_bvh,
            shadow_vis,
            samples=sec_aa,
            shadow_mode=shadow_flag,
            has_triangles=has_tri,
            has_beziers=has_bez,
            source_primitives=ev_src,
            identity_enabled=identity_on,
            self_epsilon=eps_self,
            near_epsilon=eps_near,
            terminator_mode=term_mode,
            adaptive_taps=rt_settings.shadow_adaptive_taps,
        )
        _scatter_shadow_visibility(vis_tab, acc_idx, shadow_vis)
        return vis_tab

    def _drain_sparse_secondary(
        active, state, rs_pix, pix_accum, rs_alloc, compactor, rs_vis
    ):
        """Run iterations >= 1 for compact raster primaries.

        ``rs_pix`` contains the real window-local pixel while
        ``rs_int[:, 4]`` contains the compact accumulator row.  A zero
        ``ray_offset`` therefore addresses the full prepared frame window.
        """
        rs_ro, rs_rd = state.origin, state.direction
        rs_acc, rs_sca = state.accumulated, state.scalars
        rs_int = state.integers
        it = 1
        while active.numel() > 0 and it < max_iters:
            with _stage("wavefront:   - drain active count"):
                na = int(active.numel())
            # Per-iteration attribution: the label carries the bounce index
            # (iteration 1 of the loop is bounce 0 -- the primary visible-
            # surface pass is the sheet resolve, not an iteration here) and
            # ``items`` carries the rays entering the iteration, which the
            # report's bounce table prints. ``na`` is host-side already --
            # numel() is shape metadata; the count itself was read back by the
            # compactor (``select`` -> ``count.item()``) or came from
            # ``compactor.initial`` -- so no device sync is added. Iterations
            # past the cap share one label so the table stays small.
            bounce = f"bounce {it - 1}" if it <= _BOUNCE_STAGE_CAP else "bounce 8+"
            with memory.temp():
                if rt_settings.wf_ray_sort and na >= int(rt_settings.wf_ray_sort_min):
                    with _stage(f"wavefront:   - {bounce} ray sort", items=na):
                        active = _sort_active_rays(
                            active, na, rs_ro, rs_rd, rs_pix, compactor
                        )
                with _stage("wavefront:   - drain scratch"):
                    # [kbuf, channel, num_active]: the ray ordinal is LAST so the
                    # traverse kernel's stores and shade's gathers coalesce.
                    hit_f = memory.get_tensor((kbuf, 4, na), f32)
                    hit_i = memory.get_tensor((kbuf, 2, na), i32)
                with _stage(f"wavefront:   - {bounce} traverse", items=na):
                    wavefront_traverse_events(
                        active,
                        na,
                        t_bvh.blocks,
                        t_bvh.node_miss,
                        t_bvh.leaf_prim,
                        t_bvh.leaf_tspan,
                        int(t_bvh.first_leaf),
                        a_pos,
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
                        opaque_prepass,
                        int(time_start),
                        int(width),
                        int(height),
                        0,
                        rs_ro,
                        rs_rd,
                        rs_sca,
                        rs_int,
                        hit_f,
                        hit_i,
                        rs_pix,
                        0,
                        cam_origin,
                        screen_point,
                        pixel_basis_x,
                        pixel_basis_y,
                        gen_meta,
                    )
                # Deferred shadows (rt_settings.wf_deferred_shadows): the lit
                # triangle hits in this iteration's K-buffers become shadow
                # events, traced by the lean raster_shadow_trace kernel exactly
                # as the sheet resolve's are; the shade kernel then reads the
                # traced visibility instead of marching inline.
                deferred = 0
                rs_vis_arg = rs_vis
                if shadow_flag and frag_flag and rt_settings.wf_deferred_shadows:
                    deferred = 1
                    with _stage(f"wavefront:   - {bounce} shadow events", items=na):
                        rs_vis_arg = _deferred_wavefront_shadows(
                            active, na, hit_f, hit_i, rs_ro, rs_rd, rs_int, rs_pix
                        )
                with _stage(f"wavefront:   - {bounce} shade", items=na):
                    wavefront_shade(
                        active,
                        na,
                        t_bvh.blocks,
                        t_bvh.node_miss,
                        t_bvh.leaf_prim,
                        t_bvh.leaf_tspan,
                        int(t_bvh.first_leaf),
                        a_pos,
                        a_norm,
                        merged["tri_extra"],
                        merged["tri_colors"],
                        a_uvs,
                        a_meta,
                        merged["textures"],
                        int(merged["num_colored_triangles"]),
                        col_row_arr,
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
                        pixel_world_scale,
                        render_metadata.floats,
                        render_metadata.ints,
                        int(frag_flag),
                        frag_pipelines,
                        frag_scatters,
                        int(tri_pids),
                        int(shadow_flag),
                        int(refraction_flag),
                        int(ior_stack_flag),
                        bvh_refit,
                        int(has_tri),
                        int(has_bez),
                        deferred,
                        # Shadow terminator gate (RENDERER_WORK_QUEUE.md item 20),
                        # read live per batch like the other shadow toggles.
                        int(rt_settings.shadow_terminator_mode()),
                        int(rt_settings.shadow_sided_cull),
                        int(rt_settings.wf_skip_unlit_normal),
                        int(rt_settings.direct_specular_lobe),
                        int(mem_trim),
                        opaque_closest,
                        0,
                        1,  # compact: rs_int[:, 4] holds the accumulator row
                        # Post-loop weight-floor exit, read live per batch
                        # (a ti.template() gate: flipping it mid-process
                        # compiles the other variant rather than reusing one).
                        int(rt_settings.weight_floor_exit),
                        # Light slots the vis payload carries: what this batch
                        # needs, bucketed, not the 16-light cap.
                        shadow_vis_slots(num_lights),
                        a_matid,
                        a_mat,
                        light_pos,
                        light_col,
                        int(num_lights),
                        int(time_start),
                        int(width),
                        int(height),
                        0,
                        rs_ro,
                        rs_rd,
                        rs_acc,
                        rs_sca,
                        rs_int,
                        hit_f,
                        hit_i,
                        rs_pix,
                        pix_accum,
                        rs_alloc,
                        rs_vis_arg,
                        cam_origin,
                    )
            active = compactor.select(
                rs_int,
                0,
                source=active,
                scan_pool=(pool_ratio != 1 or not rt_settings.wf_compact_active_only),
            )
            it += 1

    if sparse_coverage:
        from algan.rendering.raytracing.raster_pipeline import (
            prepare_sparse_raster_coverage,
            shade_sparse_raster_coverage,
        )

        # Sparse hit records live at the arena's reverse end for the runtime
        # of the window.  Coverage-sized ray pools are allocated/reset from the
        # forward end one compact slice at a time.
        #
        # Batch tables/trees are published before this scope. A newly discovered
        # need for a deferred BVH unwinds the whole chunk before publication, so
        # the batch floor can never pin these window-local reverse allocations.
        with memory.temp(
            clear_persist=True,
            persist_floor=lambda: merged.get(ARENA_RETAINED_REVERSE_POINTER),
        ):
            capture_fragments = fragment_capture.is_armed()
            coverage = prepare_sparse_raster_coverage(
                merged,
                tri_screen,
                tri_bounds,
                bez_bounds,
                memory,
                cam_origin,
                screen_point,
                pixel_basis_x,
                pixel_basis_y,
                pixel_world_scale,
                col_row_arr,
                time_start,
                time_end,
                width,
                height,
                half_screen_w,
                half_screen_h,
                layer_offset_triangles,
                env_in_composite=env_active,
                retain_fragments=capture_fragments,
            )
            # The GUI viewer's per-pixel inspector, when one is waiting for this
            # chunk. Off, it is a module-global read; on, it copies the coverage
            # record to the host, which has to happen HERE -- the arrays are
            # arena tensors and the enclosing ``memory.temp`` reclaims them.
            if capture_fragments:
                fragment_capture.capture(coverage, merged, time_start, width, height)
            t_val_sparse = _get_tonemap_t_val()
            # Display-referred coverage resolve (settings.aa_display_resolve).
            # The clamp it applies is only sound where a pixel's leftover
            # weight is a pure AREA, so every way of making it something else
            # turns it off for the whole batch: any translucent, transmissive
            # or refractive material folds transmittance into that weight, and
            # a texture whose alpha the builder could not settle may do the
            # same. (Reflection does too, but per pixel rather than per batch:
            # a bounced primary deposits no geometric residual, so those
            # pixels opt themselves out.) It also needs the linear frame
            # buffer -- under an in-kernel tonemap the composite's output is
            # already display-referred and the clamp would be applied twice.
            geo_cov_sparse = int(
                bool(rt_settings.aa_display_resolve)
                and t_val_sparse == 3
                and not merged.get("has_any_translucent")
                and not merged.get("has_transmissive")
                and not merged.get("has_refractive")
                and not merged.get("has_refl_transparent")
                and not merged.get("has_uncertain_texture_alpha")
            )
            with _stage("wavefront:   - sparse setup"):
                if t_val_sparse != 3:
                    # In-kernel tonemap on the sparse route (sheet unification,
                    # DESIGN_sheet_resolve.md §4.8): every pixel the covered
                    # composite will not touch owes finalize(bg) -- including the
                    # whole frame when nothing is covered at all. The covered
                    # composite reads its pixels' RAW prefilled background before
                    # writing, so ordering between the two is free.
                    with memory.temp():
                        total_px = (
                            (int(time_end) - int(time_start)) * int(width) * int(height)
                        )
                        covered_mask = memory.get_tensor((total_px,), torch.uint8)
                        covered_mask.zero_()
                        if coverage is not None:
                            covered_mask[coverage["covered_idx"].to(torch.int64)] = 1
                        wf_finalize_uncovered(
                            total_px,
                            int(width),
                            int(height),
                            covered_mask,
                            t_val_sparse,
                            float(rt_settings.tonemap_exposure),
                            out,
                        )
                if coverage is None:
                    return

                num_covered_total = int(coverage["num_covered"])

                # SPLIT-SUM GLOSSY (DESIGN_glossy_prefilter.md). Two buffers per
                # FRAME -- never per batch, which is many frames and would make
                # these the render's dominant allocation. Covered ordinals are
                # ordered by global pixel index, so a frame's covered pixels are a
                # contiguous ordinal range and the tile loop can be clamped to one
                # frame at a time and flushed at the boundary. Both come from the
                # arena, so the runtime memory model measures them like everything
                # else and the batch size adapts on its own.
                #
                # ALLOCATED BEFORE THE TILE IS SIZED, exactly as the classic route
                # allocates ``aa_accum`` first: ``_auto_primary_per_tile`` spends
                # every free arena byte (wavefront_tile_safety is 1.0), so a buffer
                # taken AFTER it is taken out of the tile's own state. At PREVIEW
                # these are only 16 MB, but the tile is sized against a nearly
                # exhausted arena by the last chunk of a batch, and 16 MB there is
                # the whole margin -- solids_and_camera and materials_and_lighting
                # both failed to fit their first attempt and then rode the halving
                # retry down to one covered pixel, which cannot help: a splitting
                # batch holds the pool fixed across retries, so nothing but
                # ``pix_accum`` shrinks.
                gl_active = int(rt_settings.glossy_reflection_mode()) == 3
                gl_main = gl_pyr = gl_levels = None
                gl_sigma_max = 0.0
                gl_bounds = None
                if gl_active:
                    levels, pyr_texels = _gloss_pyramid_levels(
                        width, height, int(rt_settings.glossy_prefilter_max_levels)
                    )
                    gl_main = memory.get_tensor(
                        (int(width) * int(height), GL_MAIN_WIDTH), f32
                    )
                    gl_pyr = memory.get_tensor((pyr_texels, GL_PYR_WIDTH), f32)
                    gl_levels = _arena_values(
                        memory, [c for row in levels for c in row], torch.int32
                    ).view(len(levels), 3)
                    gl_sigma_max = _BOX_SIGMA * float(1 << (len(levels) - 1))
                    gl_bounds = _gloss_frame_bounds(
                        coverage["covered_idx"],
                        int(width) * int(height),
                        int(time_end) - int(time_start),
                    )
                    _gloss_clear(gl_main, gl_pyr)
                gl_frame = 0

                # One 7-column accumulator row per primary on the plain route; the
                # glossy route adds a second row for the reflection alone and
                # widens both (DESIGN_glossy_prefilter.md). The tile is CHARGED and
                # the buffer ALLOCATED from these same two numbers, so the sizing
                # cannot be fitted to a narrower row than the tile goes on to take
                # -- the measured per-primary coefficient describes the plain row,
                # and a tile that spends every free byte on it has nothing left for
                # the wide one.
                pix_accum_rows, pix_accum_cols = (
                    (2, GL_ROW_WIDTH) if gl_active else (1, 8)
                )
                sparse_primary = _auto_primary_per_tile(
                    memory,
                    pool_ratio,
                    primary_per_tile,
                    extra_bytes_per_primary=(
                        (pix_accum_rows * pix_accum_cols - 7) * torch.float32.itemsize
                    ),
                    fixed_bytes=ALLOC_WIDTH * torch.int32.itemsize,
                    # The width this route's own _alloc_wavefront_state uses below.
                    state_sca_width=state_sca_width,
                )
                primary_capacity = min(max(1, int(sparse_primary)), num_covered_total)
                shared_pool_capacity = _shared_pool_slots(
                    primary_capacity,
                    sparse_primary,
                    pool_ratio,
                    analytic_raster,
                    triangle_aa=policy.triangle_aa,
                )
                learned_primary_cap = primary_capacity
                covered_start = 0

            while covered_start < num_covered_total:
                remaining = num_covered_total - covered_start
                attempt_primary = min(learned_primary_cap, remaining)
                if gl_active:
                    # Frames with no covered pixel own an empty ordinal range;
                    # skip to the frame this tile starts in. The tile itself is
                    # NOT clamped to that frame: the per-frame reflection
                    # buffers are filled one frame-part at a time after the
                    # drain (see the scatter loop below), so a tile spans as
                    # many frames as the arena lets it. Clamping used to cost a
                    # whole bounce loop -- traverse, shade and compaction
                    # launches per iteration -- per FRAME rather than per
                    # tile, which at PREVIEW put the glossy route at twice the
                    # render time of the unprefiltered one.
                    while gl_bounds[gl_frame + 1] <= covered_start:
                        gl_frame += 1
                pool = shared_pool_capacity if pool_ratio > 1 else attempt_primary

                while True:
                    # A tile attempt owns both ends through readback/compositing.
                    # A missing BVH requests a restart outside the chunk; no batch
                    # allocation may be published beneath this attempt's scratch.
                    with memory.temp(
                        clear_persist=True,
                        persist_floor=lambda: merged.get(
                            ARENA_RETAINED_REVERSE_POINTER
                        ),
                    ):
                        try:
                            with _stage("wavefront:   - tile state alloc"):
                                # Same unit-coefficient treatment as the dense tile
                                # above; this route additionally holds the visibility
                                # word and both compaction index buffers.
                                with memory.scope(
                                    "wavefront_state",
                                    pool=pool,
                                    primary=attempt_primary,
                                    global_hits=0,
                                    sparse=1,
                                ):
                                    state = _alloc_wavefront_state(
                                        memory, pool, state_sca_width, global_hits=False
                                    )
                                    rs_pix = memory.get_tensor((pool,), i32)
                                    # The glossy route doubles the rows (a second
                                    # accumulator per pixel for the reflection alone)
                                    # and widens them (the reflection's energy, blur
                                    # scale and distances) rather than spending a
                                    # kernel argument on any of it -- both kernels
                                    # involved are at 72 parameters against Taichi's
                                    # 64 runtime ones. See DESIGN_glossy_prefilter.md.
                                    pix_accum = memory.get_tensor(
                                        (
                                            attempt_primary * pix_accum_rows,
                                            pix_accum_cols,
                                        ),
                                        f32,
                                    )
                                    rs_alloc = memory.get_tensor((ALLOC_WIDTH,), i32)
                                    rs_vis = memory.get_tensor((1,), i32)
                                    compactor = _ArenaRayCompactor(memory, pool, i32)
                                rs_int = state.integers
                                if state_sca_width != SCA_WIDTH_PLAIN:
                                    # Same per-tile stack zeroing as the dense tile
                                    # above (DESIGN_mesh_identity_open.md §H).
                                    state.scalars[:, _SCA_IOR_DEPTH:].zero_()
                                pix_accum.zero_()
                                if gl_active:
                                    # A glossy ray that never hits anything writes no
                                    # distance at all, and that has to read as "the
                                    # reflection is infinitely far", i.e. fully
                                    # blurred -- not as the zero a cleared buffer
                                    # would give, which is a contact reflection.
                                    pix_accum[attempt_primary:, GL_ROW_DIST] = float(
                                        "inf"
                                    )
                                    render_metadata.ints[GLOSS_BASE] = attempt_primary
                                rs_int[:, 2].fill_(1)
                                rs_alloc.zero_()
                                rs_alloc[0] = attempt_primary

                            with memory.temp():
                                covered_idx = shade_sparse_raster_coverage(
                                    coverage,
                                    covered_start,
                                    covered_start + attempt_primary,
                                    merged,
                                    tri_screen,
                                    memory,
                                    cam_origin,
                                    screen_point,
                                    pixel_basis_x,
                                    pixel_basis_y,
                                    pixel_world_scale,
                                    render_metadata,
                                    gen_meta,
                                    light_pos,
                                    light_col,
                                    num_lights,
                                    col_row_arr,
                                    frag_flag,
                                    frag_pipelines,
                                    int(tri_pids),
                                    int(rt_settings.wf_skip_unlit_normal),
                                    refraction_flag,
                                    ior_stack_flag,
                                    time_start,
                                    width,
                                    height,
                                    half_screen_w,
                                    half_screen_h,
                                    state,
                                    rs_pix,
                                    pix_accum,
                                    rs_alloc,
                                    shadow_flag,
                                    t_bvh,
                                    bez_bvh,
                                    layer_offset_triangles,
                                    max_bounces,
                                    shadow_context=shadow_context,
                                )
                        except (InsufficientMemoryException, RuntimeError) as exc:
                            # Taichi launches OOM as a bare RuntimeError from their
                            # own allocator; treat those as OOM, re-raise real ones.
                            if not isinstance(
                                exc, InsufficientMemoryException
                            ) and not is_cuda_oom(exc):
                                raise
                            release_torch_memory(force_gc=False)
                            if attempt_primary <= 1:
                                raise OutOfRenderMemory(
                                    "Sparse raster state did not fit for one "
                                    "covered pixel. Lower the resolution or "
                                    "transparency complexity."
                                ) from exc
                            next_primary, shared_pool_capacity, pool = (
                                _shrink_sparse_memory_retry(
                                    attempt_primary, shared_pool_capacity, pool_ratio
                                )
                            )
                            _WAVEFRONT_POOL_RETRIES[0] += 1
                            learned_primary_cap = min(learned_primary_cap, next_primary)
                            attempt_primary = next_primary
                            continue

                        # The resolve's own overflow, checked before the bounce
                        # drain adds to the same counters. Truncations are folded
                        # in at the ACCEPT point below, once, so this stage keeps
                        # the split-free short-circuit it always had.
                        with _stage("wavefront:   - pool overflow poll"):
                            overflow = (
                                pool_ratio > 1
                                and int(rs_alloc[ALLOC_OVERFLOW].item()) != 0
                            )
                        if overflow:
                            if attempt_primary <= 1:
                                raise OutOfRenderMemory(
                                    "A single covered pixel's deterministic ray "
                                    f"tree exceeded the shared pool of {pool} "
                                    "slots."
                                )
                            next_primary = _overflow_retry_primary(
                                attempt_primary, int(rs_alloc[ALLOC_NEXT].item()), pool
                            )
                            _WAVEFRONT_POOL_RETRIES[0] += 1
                            learned_primary_cap = min(learned_primary_cap, next_primary)
                            attempt_primary = next_primary
                            continue

                        try:
                            with _stage("wavefront:   - bounce drain"):
                                active = compactor.select(
                                    rs_int, 0, source=compactor.current, scan_pool=True
                                )
                                if active.numel() > 0 and (
                                    merged.get("bvh_deferred")
                                    or merged.get("bvh_rehome_pending")
                                ):
                                    raise _DeferredBVHRequired
                                _drain_sparse_secondary(
                                    active,
                                    state,
                                    rs_pix,
                                    pix_accum,
                                    rs_alloc,
                                    compactor,
                                    rs_vis,
                                )
                        except (InsufficientMemoryException, RuntimeError) as exc:
                            # Taichi launches OOM as a bare RuntimeError from their
                            # own allocator; treat those as OOM, re-raise real ones.
                            if not isinstance(
                                exc, InsufficientMemoryException
                            ) and not is_cuda_oom(exc):
                                raise
                            release_torch_memory(force_gc=False)
                            if attempt_primary <= 1:
                                raise OutOfRenderMemory(
                                    "Sparse raster bounce scratch did not fit for "
                                    "one covered pixel. Lower the resolution or "
                                    "transparency complexity."
                                ) from exc
                            next_primary, shared_pool_capacity, pool = (
                                _shrink_sparse_memory_retry(
                                    attempt_primary, shared_pool_capacity, pool_ratio
                                )
                            )
                            _WAVEFRONT_POOL_RETRIES[0] += 1
                            learned_primary_cap = min(learned_primary_cap, next_primary)
                            attempt_primary = next_primary
                            continue

                        # Secondary shading can itself split again.  Discard and
                        # retry the whole compact slice before compositing if any
                        # of those later allocations exhausted the shared pool.
                        # This is the slice's accept point, so it is also where the
                        # resolve's and the drain's truncation counters are folded
                        # into the render's totals.
                        with _stage("wavefront:   - alloc readback"):
                            alloc = _read_tile_alloc(rs_alloc)
                        if pool_ratio > 1 and alloc[ALLOC_OVERFLOW] != 0:
                            if attempt_primary <= 1:
                                raise OutOfRenderMemory(
                                    "A single covered pixel's deterministic ray "
                                    f"tree exceeded the shared pool of {pool} "
                                    "slots."
                                )
                            next_primary = _overflow_retry_primary(
                                attempt_primary, alloc[ALLOC_NEXT], pool
                            )
                            _WAVEFRONT_POOL_RETRIES[0] += 1
                            learned_primary_cap = min(learned_primary_cap, next_primary)
                            attempt_primary = next_primary
                            continue
                        _record_tile_truncations(alloc, pool)

                        with _stage("wavefront:   - tile composite"):
                            if gl_active:
                                # The reflection buffers hold ONE frame, so a tile that
                                # spans several is scattered, composited and finished
                                # one frame-part at a time, in frame order: the
                                # scatter reads the raw prefilled background and must
                                # precede the composite of the same pixels, and the
                                # finish overwrites the composite's values for the
                                # glossy pixels, so a frame's part is composited before
                                # it is finished and the buffers are cleared only once
                                # a frame's last pixel is in -- a frame whose pixels
                                # straddle two tiles keeps its buffer across them.
                                tile_start = covered_start
                                tile_end = covered_start + attempt_primary
                                ppf = int(width) * int(height)
                                while covered_start < tile_end:
                                    frame_end = gl_bounds[gl_frame + 1]
                                    if frame_end <= covered_start:
                                        # A frame with no covered pixel: nothing to
                                        # scatter or finish, exactly as the skip at
                                        # the top of the tile treats it.
                                        gl_frame += 1
                                        continue
                                    part_end = min(tile_end, frame_end)
                                    a = covered_start - tile_start
                                    b = part_end - tile_start
                                    gloss_scatter(
                                        int(b - a),
                                        int(attempt_primary),
                                        gl_frame * ppf,
                                        gl_frame,
                                        int(width),
                                        float(gl_sigma_max),
                                        covered_idx[a:b],
                                        pix_accum,
                                        gl_main,
                                        gl_pyr,
                                        out,
                                        int(a),
                                    )
                                    wf_composite_accum_sparse(
                                        int(time_start),
                                        int(width),
                                        int(height),
                                        1 if transparent else 0,
                                        0,
                                        covered_idx[a:b],
                                        pix_accum[a:b],
                                        t_val_sparse,
                                        float(rt_settings.tonemap_exposure),
                                        geo_cov_sparse,
                                        out,
                                    )
                                    covered_start = part_end
                                    if covered_start >= frame_end:
                                        # Frame complete: prefilter its reflection
                                        # buffer and composite the glossy pixels over
                                        # the values just written for them.
                                        _gloss_finish_frame(
                                            gl_frame,
                                            gl_levels,
                                            gl_main,
                                            gl_pyr,
                                            int(width),
                                            int(height),
                                            t_val_sparse,
                                            out,
                                        )
                                        _gloss_clear(gl_main, gl_pyr)
                                        gl_frame += 1
                                break

                            wf_composite_accum_sparse(
                                int(time_start),
                                int(width),
                                int(height),
                                1 if transparent else 0,
                                0,
                                covered_idx,
                                pix_accum,
                                t_val_sparse,
                                float(rt_settings.tonemap_exposure),
                                geo_cov_sparse,
                                out,
                            )
                            covered_start += attempt_primary
                            break
        return

    def run_tile(tile_start, tn_primary, pool, state, rs_pix, pix_accum, rs_alloc):
        (
            rs_ro,
            rs_rd,
            rs_acc,
            rs_sca,
            rs_int,
            rs_kt,
            rs_kl,
            rs_ka,
            rs_kb,
            rs_kp,
            rs_kf,
        ) = state
        # One-element placeholder for the classic shade kernel's legacy
        # deferred-visibility argument.
        rs_vis = memory.get_tensor((1,), i32)
        compactor = _ArenaRayCompactor(memory, pool, i32)
        it = 0
        active = compactor.initial(tn_primary)
        while active.numel() > 0 and it < max_iters:
            na = int(active.numel())
            # Fused generation: the tile's first iteration generates rays in
            # traversal and shades with implicit initial state (separate
            # compile-time instantiations). Later iterations and unfused
            # renders use the same compact event-batch kernels with materialized
            # persistent state.
            first = 1 if (gen_fused and it == 0) else 0
            # The hit batch is phase-local: traversal writes one compact
            # [active ray, kbuf] surface-event record and shade consumes it in
            # the same host iteration. Releasing this arena scope before
            # compaction removes the six permanent [pool, kbuf] arrays from
            # secondary radiance state while preserving the existing four-hit
            # traversal/shading behavior.
            with memory.temp():
                # [kbuf, channel, num_active]: the ray ordinal is LAST so the
                # traverse kernel's stores and shade's gathers coalesce.
                hit_f = memory.get_tensor((kbuf, 4, na), f32)
                hit_i = memory.get_tensor((kbuf, 2, na), i32)
                wavefront_traverse_events(
                    active,
                    na,
                    t_bvh.blocks,
                    t_bvh.node_miss,
                    t_bvh.leaf_prim,
                    t_bvh.leaf_tspan,
                    int(t_bvh.first_leaf),
                    a_pos,
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
                    opaque_prepass,
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
                    first,
                    cam_origin,
                    screen_point,
                    pixel_basis_x,
                    pixel_basis_y,
                    gen_meta,
                )
                wavefront_shade(
                    active,
                    na,
                    t_bvh.blocks,
                    t_bvh.node_miss,
                    t_bvh.leaf_prim,
                    t_bvh.leaf_tspan,
                    int(t_bvh.first_leaf),
                    a_pos,
                    a_norm,
                    merged["tri_extra"],
                    merged["tri_colors"],
                    a_uvs,
                    a_meta,
                    merged["textures"],
                    int(merged["num_colored_triangles"]),
                    col_row_arr,
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
                    pixel_world_scale,
                    render_metadata.floats,
                    render_metadata.ints,
                    int(frag_flag),
                    frag_pipelines,
                    frag_scatters,
                    int(tri_pids),
                    int(shadow_flag),
                    int(refraction_flag),
                    int(ior_stack_flag),
                    bvh_refit,
                    int(has_tri),
                    int(has_bez),
                    0,
                    # Shadow terminator gate (RENDERER_WORK_QUEUE.md item 20),
                    # read live per batch like the other shadow toggles.
                    int(rt_settings.shadow_terminator_mode()),
                    int(rt_settings.shadow_sided_cull),
                    int(rt_settings.wf_skip_unlit_normal),
                    int(rt_settings.direct_specular_lobe),
                    int(mem_trim),
                    opaque_closest,
                    first,
                    0,  # compact: dense tiles accumulate at the ray's pixel
                    # Post-loop weight-floor exit, read live per batch (see
                    # the sparse drain call site).
                    int(rt_settings.weight_floor_exit),
                    # Light slots the vis payload carries: what this batch
                    # needs, bucketed, not the 16-light cap.
                    shadow_vis_slots(num_lights),
                    a_matid,
                    a_mat,
                    light_pos,
                    light_col,
                    int(num_lights),
                    int(time_start),
                    int(width),
                    int(height),
                    int(tile_start),
                    rs_ro,
                    rs_rd,
                    rs_acc,
                    rs_sca,
                    rs_int,
                    hit_f,
                    hit_i,
                    rs_pix,
                    pix_accum,
                    rs_alloc,
                    rs_vis,
                    cam_origin,
                )
            active = compactor.select(
                rs_int,
                0,
                source=active,
                scan_pool=(pool_ratio != 1 or not rt_settings.wf_compact_active_only),
            )
            it += 1

    _run_wavefront_tiles(
        memory,
        out,
        n=n,
        width=width,
        height=height,
        time_start=time_start,
        transparent=transparent,
        aa_level=aa_level,
        pool_ratio=pool_ratio,
        primary_per_tile=primary_per_tile,
        cam_origin=cam_origin,
        screen_point=screen_point,
        pixel_basis_x=pixel_basis_x,
        pixel_basis_y=pixel_basis_y,
        half_screen_w=half_screen_w,
        half_screen_h=half_screen_h,
        max_bounces=max_bounces,
        near_clip=near_clip,
        run_tile=run_tile,
        # rs_vis placeholder + the compactor's output-count word.
        auto_fixed_bytes=2 * torch.int32.itemsize,
        gen_fused=gen_fused,
        global_hits=False,
        analytic_raster=analytic_raster,
        sca_width=state_sca_width,
    )


def _scene_has_custom_scatter(merged):
    """True if any merged primitive's material pipeline carries a custom
    scatter func (user-controlled ray bouncing). The monolithic wavefront
    shade kernel dispatches these directly; this check only decides whether
    the scatter templates get compiled in (scatter-free scenes stay on the
    scatter-free, byte-identical default path). Cheap: exits on the
    tensor-max user-pipeline pre-check for the (overwhelmingly common)
    all-built-in scene.
    """
    cached = merged.get("has_custom_scatter")
    if cached is not None:
        return bool(cached)
    if not _scene_has_user_pipeline(merged):
        merged["has_custom_scatter"] = False
        return False
    from algan.rendering.shaders.fragment_shaders import build_frag_scatters

    # The whole registry, indexed by GLOBAL id -- so this must be asked with
    # global ids, not with the dense ones the merge packs
    # (``_batch_user_pipeline_ids`` returns exactly those globals).
    scatters = build_frag_scatters()
    ids = _batch_user_pipeline_ids(merged)
    if ids is None:
        ids = []
        for prefix in ("tri", "pn"):
            arr = merged.get(f"{prefix}_mat_id")
            if arr is not None and arr.numel():
                ids.extend(torch.unique(arr.detach().cpu()).tolist())
    for pid in ids:
        i = int(pid) - _USER_PIPELINE_BASE
        if 0 <= i < len(scatters) and scatters[i] is not None:
            merged["has_custom_scatter"] = True
            return True
    merged["has_custom_scatter"] = False
    return False


def _batch_user_pipeline_ids(merged):
    """The user fragment-pipeline ids this batch's primitives carry, in the
    slot order the batch packs them, for narrowing the injected pipeline /
    scatter tuples to what the batch can reach
    (``fragment_shaders.build_frag_pipelines``).

    These are GLOBAL registry ids; the merge has renumbered the packed table
    so that slot ``i`` is global id ``result[i]``
    (``scene_builder._densify_frag_pipeline_ids``), which is what keeps the
    injected tuple's shape a function of the batch rather than of the
    process's registration history.

    Returns ``None`` -- "the whole registry, indexed by global id", the safe
    answer -- for a merged scene that carries no such list, whose packed ids
    are therefore still global. That is the same reason ``_frag_pid_mask``
    returns ``ALL_PIDS`` there: dropping a pipeline the kernel can still
    dispatch to would shade that surface with no material at all. Reads only
    the merge-time host-side list, so it costs no device reduction.
    """
    ids = merged.get("frag_pipeline_ids")
    return None if ids is None else tuple(ids)


def _frag_pid_mask(merged, prefix, active, _record=True):
    """Compile-time bitmask of the material pipeline ids ``prefix`` geometry
    carries in this batch (bit ``p`` = pipeline id ``p``), for the shade
    kernels' compile-time material gating (see ``rt_settings.frag_pid_gate``
    and ``shading_taichi._run_frag_pipeline``).

    ``ALL_PIDS`` -- every stage compiled in, i.e. the ungated kernel -- is
    returned whenever the gate is off, the geometry type is absent from the
    kernel (``active``), or the merged scene predates the host-side id list;
    the mask must never *miss* an id the kernel can read, and the merge-time
    list is the only source that costs no device reduction here.

    The ids come from ``scene_builder``'s ``torch.unique`` over the very
    ``{prefix}_mat_id`` table the kernel indexes, so the mask is exact --
    except under the Family-A memory trim, whose compacted table can only
    drop primitives, leaving the mask a (still safe) superset.
    """
    mask = ALL_PIDS
    material_ids = merged.get(f"{prefix}_material_ids")
    if rt_settings.frag_pid_gate and active and material_ids:
        mask = 0
        for pid in material_ids:
            pid = int(pid)
            if pid < 0:
                mask = ALL_PIDS
                break
            mask |= 1 << pid
    if _record:
        _FRAG_PID_LAST[prefix] = mask
    return mask


_originals = {}


def is_ray_tracing_enabled():
    """Vestigial: always False. The ray-traced primitive classes are now the
    engine's only renderer (``RENDERER_REGISTRY`` binds them by default), and
    the ``enable_ray_tracing`` toggle that used to populate ``_originals`` was
    removed with the rasterizer. Kept only because ``post_processing.bloom``
    probes for it defensively.
    """
    return bool(_originals)
