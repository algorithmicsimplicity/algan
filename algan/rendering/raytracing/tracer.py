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
* ``samples_per_pixel > 1`` -- the Monte Carlo path-tracing megakernel
  (``path_trace_scene_stbvh``), one thread per (frame, pixel, sample) path,
  accumulating into a float32 per-pixel buffer that ``finalize_samples``
  averages.

Reflections and refraction are inferred from the mob's Three.js-style
material properties. Use ``MeshStandardMaterial(metalness=..., roughness=...)``
or ``MeshPhysicalMaterial`` before spawning; rays bounce up to
``MAX_BOUNCES`` times.

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

from algan.errors import UnsupportedFeatureError
from algan.rendering.post_processing.post_process import post_process_frames
from algan.rendering.primitives.primitive import OutOfRenderMemory
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    KBUF,
    MAX_SURFACES_PER_RAY,
    finalize_samples,
    path_trace_scene_stbvh,
)
from algan.rendering.raytracing.scene_builder import (
    _downsample_background,
    _merge_scene,
    _pack_lights,
    _prefill_background,
    copy_merged_scene_to_arena,
)

# NOTE: only immutable settings values may be imported by value here; the
# mutable module globals (SAMPLES_PER_PIXEL, TONEMAP_*, SHADOWS, ...) must be
# read live as ``rt_settings.X`` or their setters silently stop working.
from algan.rendering.raytracing.settings import (
    _get_tonemap_t_val,
    _scene_has_user_pipeline,
    is_post_process_tonemap_enabled,
)
from algan.rendering.raytracing.truncation import (
    TruncationCounts,
    attach_truncations,
    record_truncation,
    report_truncations,
    restore_truncations,
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
    MAX_SHADOW_LIGHTS,
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
from algan.rendering.raytracing.utils import _expand_frames, _flat_frames, _pixel_bases

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
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _SCA_IOR_DEPTH,
    SCA_WIDTH_PLAIN,
    ALLOC_NEXT,
    ALLOC_OVERFLOW,
    ALLOC_TRUNC_SURFACES,
    ALLOC_WIDTH,
    compact_ray_slots,
    sca_width,
    wavefront_generate_rays,
    wavefront_shade,
    wavefront_traverse,
    wavefront_traverse_events,
    wf_composite_accum,
    wf_composite_accum_aa,
    wf_composite_accum_sparse,
    wf_finalize_aa,
    wf_finalize_uncovered,
)
from algan.utils.memory_utils import (
    InsufficientMemoryException,
    empty_cache,
    ensure_render_headroom,
    is_cuda_oom,
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

    backend: Literal["deterministic_wavefront", "monte_carlo"]
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


def _alloc_wavefront_state(memory, tn, sca_width, *, global_hits=True):
    """Allocate the wavefront's per-ray global state from the render memory pool
    (a bump allocator) rather than fresh ``torch.empty`` tensors.

    The caller snapshots ``memory.get_pointers()`` before each tile and restores
    them after, so this ~hundreds-of-MB of state is released *deterministically*
    at the end of every iteration and the next tile reuses the same arena bytes.
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
        # 5 weight green, 6 weight blue (colour transport).
        memory.get_tensor((tn, sca_width), f32),  # rs_sca (7 general)
        # rs_int: 0 bounces_left, 1 processed, 2 status, 3 num_hits, 4 drained
        # (column 4 is used only by the legacy sorted-material path
        # (unsupported); the classic kernels index columns 0-3 and never
        # read it).
        memory.get_tensor((tn, 5), i32),  # rs_int
    )
    if global_hits:
        return core + (
            memory.get_tensor((tn, KBUF), f32),  # rs_kt
            memory.get_tensor((tn, KBUF), f32),  # rs_kl
            memory.get_tensor((tn, KBUF), f32),  # rs_ka
            memory.get_tensor((tn, KBUF), f32),  # rs_kb
            memory.get_tensor((tn, KBUF), i32),  # rs_kp
            memory.get_tensor((tn, KBUF), i32),  # rs_kf
        )

    # The supported general renderer no longer attaches a K-buffer to every
    # continuation-pool slot. Keep six tiny placeholders solely so the state
    # tuple remains ABI-compatible with the hybrid raster frontend and the
    # unsupported legacy orchestrators. The general traverse/shade pair uses an
    # exact-size transient surface-event batch allocated for the current active
    # queue instead.
    stub_f = memory.get_tensor((1, 1), f32)
    stub_i = memory.get_tensor((1, 1), i32)
    return core + (stub_f, stub_f, stub_f, stub_f, stub_i, stub_i)


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
        float(rt_settings.TONEMAP_EXPOSURE),
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
    2. CONTINUATION-RAY SUPERSAMPLING (``ANALYTIC_AA_SECONDARY_SAMPLES > 1``),
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


def _shared_pool_slots(primary_capacity, memory_primary, pool_ratio, analytic_raster):
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
    if analytic_raster and rt_settings.analytic_aa_tri_active():
        from algan.rendering.raytracing.raster_taichi import _AA_NUM_SAMPLES

        ratio = max(ratio, _AA_NUM_SAMPLES + 1)
    return max(1, min(int(primary_capacity) * ratio, budgeted))


# Fraction of the exactly-measured fit to actually retry with (see
# ``_overflow_retry_primary``). A second overflow costs another discarded
# resolve, so the margin is deliberately generous relative to the ~10% spread
# in per-pixel demand that a coverage partition produces.
_POOL_RETRY_SAFETY = 0.85


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
        cap=MAX_SURFACES_PER_RAY,
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
    scaled = int(attempt_primary * pool * _POOL_RETRY_SAFETY / slots_wanted)
    return max(1, min(scaled, attempt_primary - 1))


def analytic_raster_route_active(
    merged,
    *,
    light_sources=(),
    environment_map=None,
    near_clip=0.0,
    far_clip=0.0,
    transparent_background=False,
):
    """Whether this batch can use analytic coverage at output resolution.

    This is the single host-side route decision shared by allocation planning
    and rendering.  A requested supersample level is therefore retained for
    every route that the raster frontend cannot honor; only a batch whose
    complete primary geometry has analytic coverage selects AA=1.

    Analytic coverage is resolved by the sheet route and nothing else
    (DESIGN_sheet_resolve.md; the fragment walk that once served it is
    deleted), so the sheet resolve's own preconditions are route vetoes here:
    with any of them off the batch falls back to the classic wavefront at the
    requested supersample level.  A transparent background composites fine
    from sheet leftover weight, but not together with an environment map
    (the env prefill would fill the alpha the background owes), so that one
    combination also falls back.
    """
    if (
        int(rt_settings.SAMPLES_PER_PIXEL) > 1
        or not rt_settings.HYBRID_RASTER
        or not rt_settings.ANALYTIC_AA
        or not rt_settings.SHEET_RESOLVE
        or not rt_settings.ANALYTIC_AA_RUN
        or not rt_settings.RASTER_SPARSE_COVERAGE
        or not rt_settings.RASTER_EMPTY_SKIP
        or not rt_settings.RASTER_COVERED_SHADE
        or (transparent_background and environment_map is not None)
        or merged.get("tri_frame_valid") is None
        or merged.get("textured_active")
        or float(near_clip) > 0.0
    ):
        return False

    num_tri = int(merged.get("num_triangles", 0))
    num_bez = int(merged.get("num_circuits", 0))
    if num_tri <= 0 and num_bez <= 0:
        return False
    if num_tri > 0 and not rt_settings.analytic_aa_tri_active():
        return False
    if num_bez > 0 and not rt_settings.analytic_aa_bez_active():
        return False

    shadow = bool(rt_settings.SHADOWS)
    lights_extended = any(
        getattr(light, "_render_aux", None) is not None
        for light in (light_sources or ())
    )
    has_environment = environment_map is not None
    frag = (
        bool(rt_settings.FRAGMENT_SHADING)
        or shadow
        or _scene_has_user_pipeline(merged)
        or lights_extended
        or has_environment
    )
    custom_scatter = bool(frag and _scene_has_custom_scatter(merged))
    if custom_scatter:
        return False

    extended = (
        lights_extended
        or has_environment
        or float(near_clip) > 0.0
        or float(far_clip) > 0.0
    )
    if (
        frag
        and rt_settings.WAVEFRONT_SORT_MATERIALS is True
        and not extended
        and not merged.get("bez_has_reflective", False)
    ):
        return False

    analytic_split = _secondary_split_needed(merged, True)
    refraction = bool(
        merged.get("has_refractive")
        or merged.get("has_refl_transparent")
        or analytic_split
    )
    mem_trim = bool(
        rt_settings.WF_MEM_TRIM
        and merged.get("mem_trim_active")
        and not shadow
        and not refraction
    )
    return not mem_trim


def effective_anti_alias_level(
    merged,
    requested,
    *,
    light_sources=(),
    environment_map=None,
    near_clip=0.0,
    far_clip=0.0,
    transparent_background=False,
):
    """Return 1 for analytic raster, otherwise the requested AA setting."""
    requested = max(1, int(requested))
    if analytic_raster_route_active(
        merged,
        light_sources=light_sources,
        environment_map=environment_map,
        near_clip=near_clip,
        far_clip=far_clip,
        transparent_background=transparent_background,
    ):
        return 1
    return requested


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
# slot because it counted 6*KBUF words of K-buffers that this route does not
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
    bytes when ``settings.WAVEFRONT_TILE_AUTO`` is on (see settings.py for the
    rationale: fewer, bigger tiles amortize the fixed host-side kernel-launch
    cost). Falls back to ``static_primary`` (the WAVEFRONT_TILE_RAYS-derived
    value) for unmanaged pools or when auto is disabled. Byte-identical to any
    other tile size: tiles partition pixels, and every per-pixel computation
    is independent of its tile.
    """
    if not rt_settings.WAVEFRONT_TILE_AUTO or not getattr(memory, "managed", False):
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
    safety = min(1.0, max(0.0, float(rt_settings.WAVEFRONT_TILE_SAFETY)))
    usable = int(free * safety) - alignment_bytes - int(fixed_bytes)
    budget = max(0, usable) // bytes_per_primary
    hi = max(1, rt_settings.WAVEFRONT_TILE_MAX // pool_ratio)
    lo = min(hi, max(1, rt_settings.WAVEFRONT_TILE_MIN // pool_ratio))
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
    the ``layer_offsets`` ndarray -- the kernel is at the 64-arg ceiling).
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
    the in-kernel ENV_SH light row. A uniform map of colour ``c`` yields
    ``A = c, B = 0`` -- i.e. it lights like an ambient light of colour ``c``.
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
    samples_per_pixel, scene_environment_map, merged, light_sources=()
):
    """Resolve the renderer route and feature compatibility for a batch."""
    samples_requested = max(1, int(samples_per_pixel))
    backend = "monte_carlo" if samples_requested > 1 else "deterministic_wavefront"
    requested = []
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

    unsupported = tuple(requested) if samples_requested > 1 else ()
    return RenderPlan(
        backend=backend,
        samples_per_pixel=samples_requested,
        requested_features=tuple(requested),
        unsupported_features=unsupported,
    )


def _validate_render_capabilities(
    samples_per_pixel, scene_environment_map, merged, light_sources=()
):
    """Validate that the selected renderer can honor the authored scene.

    ``samples_per_pixel > 1`` selects the Monte Carlo megakernel. Several
    features currently exist only in the deterministic wavefront renderer;
    silently discarding them is more dangerous than failing early. The global
    unsupported-feature policy permits an explicit warning/ignore migration
    mode for benchmarks and legacy projects.
    """
    plan = _build_render_plan(
        samples_per_pixel,
        scene_environment_map,
        merged,
        light_sources,
    )
    if plan.unsupported_features:
        feature_list = ", ".join(plan.unsupported_features)
        rt_settings.report_unsupported_features(
            "The Monte Carlo renderer selected by samples_per_pixel > 1 "
            f"cannot honor: {feature_list}. Set samples_per_pixel to 1 to use "
            "the deterministic wavefront renderer, remove those features, or "
            "set_unsupported_feature_policy('warn'/'ignore') explicitly."
        )

    # Keep direct mutation of the legacy globals from bypassing the guarded
    # public setters. These backends are known-broken and must not render a
    # misleading result.
    if bool(getattr(rt_settings, "WF_TEXTURED", False)):
        raise UnsupportedFeatureError(
            "The legacy textured wavefront renderer is unsupported."
        )
    if getattr(rt_settings, "WAVEFRONT_SORT_MATERIALS", "auto") is True:
        raise UnsupportedFeatureError(
            "The legacy sorted-material wavefront renderer is unsupported."
        )
    return plan


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
            rt_settings.RASTER_TRI_PRECOMPUTE
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
        if rt_settings.RASTER_BEZ_PRECOMPUTE and int(merged.get("num_circuits", 0)) > 0:
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
    background_color,
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
    """Render frames [time_start, time_end) of a primitive batch by ray
    tracing into a fixed [frames, pixels, channels] buffer.

    On out-of-memory the time window is halved and retried; per-frame memory
    is just the output buffer (plus post-processing), independent of scene
    depth complexity or bounce count.
    """
    # Read the user-toggleable settings *live* from the settings module.
    # These names used to be imported by value at module-import time, which
    # froze them before user code ran -- silently disabling
    # set_ray_traced_shadows() / set_samples_per_pixel() / etc. for anyone
    # calling the setters after `import algan` (i.e. everyone).
    SAMPLES_PER_PIXEL = rt_settings.SAMPLES_PER_PIXEL
    SHADOWS = rt_settings.SHADOWS
    FRAGMENT_SHADING = rt_settings.FRAGMENT_SHADING
    MAX_BOUNCES = rt_settings.MAX_BOUNCES
    TONEMAP_EXPOSURE = rt_settings.TONEMAP_EXPOSURE
    INDIRECT_BOUNCE_STRENGTH = rt_settings.INDIRECT_BOUNCE_STRENGTH
    scene_env_map = getattr(scene, "environment_map", None)
    env_map = scene_env_map if int(SAMPLES_PER_PIXEL) <= 1 else None
    env_source = env_map.detach().cpu() if torch.is_tensor(env_map) else env_map
    env_meta = getattr(primitives[0], "_rt_env_meta", None)
    merged = getattr(primitives[0], "_rt_device_scene", None)
    if merged is None:
        merged_host = _merge_scene(primitives)
        # Validate on host metadata before reserving/copying the persistent
        # device scene. Unsupported combinations therefore fail before costly
        # arena allocations or any Taichi kernel compilation.
        plan = _validate_render_capabilities(
            SAMPLES_PER_PIXEL,
            scene_env_map,
            merged_host,
            light_sources,
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
        plan = _validate_render_capabilities(
            SAMPLES_PER_PIXEL,
            scene_env_map,
            merged,
            light_sources,
        )
    scene.last_render_plan = plan

    # Refraction is only implemented by the general wavefront tracer, which is
    # already where every deterministic (samples <= 1) batch goes -- so this
    # only gates the refraction template, not routing. The Monte Carlo
    # megakernel (samples > 1) ignores the refractive index.
    refractive_det = bool(merged.get("has_refractive")) and int(SAMPLES_PER_PIXEL) <= 1
    # Semi-transparent PBR surfaces split off a reflection branch, so they need
    # the same pool + split code the refraction path compiles in. No routing
    # implication: the deterministic (samples <= 1) path is already wavefront.
    refl_transparent_det = (
        bool(merged.get("has_refl_transparent")) and int(SAMPLES_PER_PIXEL) <= 1
    )

    # Extended lights (directional / ambient / hemisphere / spot / area /
    # falloff / soft shadows) and environment maps are features of the
    # deterministic general wavefront with per-fragment lighting: their
    # presence forces fragment shading on and routes away from the textured /
    # sorted variants. Plain point-light scenes keep the compact light packing
    # and are untouched.
    lights_extended = int(SAMPLES_PER_PIXEL) <= 1 and any(
        getattr(light, "_render_aux", None) is not None
        for light in (light_sources or ())
    )
    cam = getattr(scene, "camera", None)
    near_clip = float(getattr(cam, "near", 0.0) or 0.0)
    far_clip = float(getattr(cam, "far", 0.0) or 0.0)
    analytic_raster = analytic_raster_route_active(
        merged,
        light_sources=light_sources,
        environment_map=env_map,
        near_clip=near_clip,
        far_clip=far_clip,
        transparent_background=transparent_background,
    )
    aa = 1 if analytic_raster else max(1, int(anti_alias_level))

    # Anti-aliasing strategy. Analytic raster coverage always renders at output
    # resolution (aa == 1). Every route it cannot cover keeps the requested
    # setting and either renders a supersampled buffer or, with INPLACE_AA,
    # averages ``aa^2`` jittered sub-pixel rays in place at the output
    # resolution, so the frame buffer stays ``screen_width x screen_height``
    # regardless of ``aa`` (aa^2x less render memory than super-sampling): the
    # wavefront runs the full gen→traverse→shade→compact→composite pipeline
    # once per sub-pixel sample, accumulating into a float buffer and
    # averaging at the end, while the Monte Carlo megakernel folds the aa^2
    # factor into its per-pixel sample count (see samples_eff below).
    inplace_aa = bool(rt_settings.INPLACE_AA)
    if inplace_aa:
        width = screen_width
        height = screen_height
        kernel_aa = aa  # in-kernel sub-pixel averaging factor
        post_aa = 1  # post-processing does not down-sample
    else:
        width = screen_width * aa
        height = screen_height * aa
        kernel_aa = 1
        post_aa = aa

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
        2.0 / (screen_height * aa * b1_norm * screen_dist).clamp_min(1e-12)
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
    # anti-alias level (Scene.set_background_color and
    # _prepare_background_for_chunk both build it at screen * anti_alias_level).
    # This batch's frame buffer is at output resolution whenever the route
    # takes one sample per output pixel: in-place AA, and the analytic raster
    # route, which forces ``aa == 1`` however many samples were requested.
    # Average the background down to match -- a super-sampled background read
    # at output stride silently scrolls a different slice of itself into every
    # frame. (Solid colors are resolution-free and pass through untouched.)
    background_aa = max(1, int(anti_alias_level))
    if background_aa > 1 and width == screen_width and height == screen_height:
        background_color = _downsample_background(
            background_color,
            background_aa,
            time_end - time_start,
            screen_height,
            screen_width,
        )

    # A deferred-BVH batch (scene_builder._finalize_bvhs) holds placeholder
    # trees; the Monte Carlo megakernel traverses unconditionally, so build
    # the real trees now if that is where this batch is headed. (The
    # deterministic wavefront has its own later, finer-grained check.)
    if merged.get("bvh_deferred") and int(SAMPLES_PER_PIXEL) > 1:
        from algan.rendering.raytracing.scene_builder import build_deferred_bvhs

        build_deferred_bvhs(merged)
    tri_bvh = merged["tri_bvh"]
    bez_bvh = merged["bez_bvh"]
    # A geometry type absent from the whole batch has only a placeholder BVH;
    # tell the deterministic kernel so it skips that empty traversal per ray.
    if rt_settings.gate_empty_traversals:
        has_tri = 1 if merged["num_triangles"] > 0 else 0
        has_bez = 1 if merged["num_circuits"] > 0 else 0
    else:  # benchmarking escape hatch: traverse every (possibly empty) tree
        has_tri = has_bez = 1
    t_val = _get_tonemap_t_val()
    # The scene builder has already reduced every geometry type's per-frame
    # bounds/edge geometry to conservative batch-wide coverage-possibility bits.
    # If all three are false, every valid primitive is point-degenerate and
    # exact primary coverage is empty for the entire materialized batch: leave
    # the background prefill untouched and skip even the sparse COUNT discovery
    # pass.  This applies to moving batches too; it is not a static-scene
    # shortcut.
    sparse_batch_empty = bool(
        int(SAMPLES_PER_PIXEL) <= 1
        and rt_settings.HYBRID_RASTER
        and rt_settings.RASTER_SPARSE_COVERAGE
        and rt_settings.RASTER_EMPTY_SKIP
        and rt_settings.RASTER_COVERED_SHADE
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

    samples = max(1, int(SAMPLES_PER_PIXEL))
    # In-place AA folds the anti-alias super-sampling into the Monte Carlo
    # sample count: each of the ``aa^2`` sub-pixels would have drawn ``samples``
    # random rays jittered over its own cell and then been averaged down, which
    # is equivalent (same total, same expectation) to drawing ``samples * aa^2``
    # rays jittered over the whole output pixel. (The wavefront/super-sample
    # path keeps ``kernel_aa == 1``, so ``samples_eff == samples`` there.)
    samples_eff = samples * (kernel_aa * kernel_aa)

    # Deterministic per-fragment shading is active for a single-sample,
    # non-physical render with the toggle on; it needs the scene's point lights
    # in the kernel. (Physical mode packs the same lights for its own path.)
    # Deterministic hard shadows are evaluated inside the per-fragment lighting
    # model, so enabling them implies fragment shading for this render.
    det_shadows = bool(SHADOWS) and samples <= 1
    # A mob with a custom fragment pipeline (Mob.set_fragment_shader) forces
    # fragment shading on for this render, without a persistent global toggle.
    scene_has_frag_pipeline = _scene_has_user_pipeline(merged)
    det_frag = (
        bool(FRAGMENT_SHADING)
        or det_shadows
        or scene_has_frag_pipeline
        or lights_extended
        or env_map is not None
    ) and samples <= 1
    frag_flag = 1 if det_frag else 0
    shadow_flag = 1 if det_shadows else 0
    # Opaque any-hit shadow early-out (compile-time mode of the shadow
    # query; see rt_settings.SHADOW_ANYHIT): 2 = any-hit pre-pass over the
    # opaque-flagged leaves with the ordered march as fallback; 3 = any-hit
    # only, valid when the batch provably contains no translucent geometry
    # (every visible primitive carries the opaque leaf flag, so a miss proves
    # the ray lit). Uncertain texture alpha keeps the fallback: such
    # primitives are not opaque-flagged, and their shadow attenuation only
    # the march can evaluate. 4 = gather-march: the ordered peel rebuilt on
    # the KBUF gather, valid for any batch (the drain evaluates translucent
    # attenuation exactly like the march), so it needs no translucent gate.
    if shadow_flag and rt_settings.SHADOW_ANYHIT:
        if rt_settings.SHADOW_ANYHIT == "gather":
            shadow_flag = 4
        elif merged.get("has_transmissive", True):
            # Both any-hit modes ask "is anything there", and answer full
            # occlusion when something is. That was equivalent to the march
            # while a covered surface always blocked; it is not once a
            # transmissive one passes light instead (see
            # ``raytrace_kernels_taichi._shadow_pass_through``) -- a glass
            # ball is alpha 1, so it does not even read as translucent below.
            # Such a batch keeps the ordered march, which evaluates the
            # attenuation exactly.
            shadow_flag = 1
        else:
            batch_has_translucent = (
                merged.get("tri_has_translucent", True)
                or merged.get("bez_has_translucent", True)
                or merged.get("has_uncertain_texture_alpha", True)
            )
            shadow_flag = 2 if batch_has_translucent else 3
    # THE SHEET ROUTE (DESIGN_sheet_resolve.md): decided ONCE here so the
    # frame-buffer prefill in render_chunk, the sparse-route gate and the
    # emission's compaction all answer the same question (the host/kernel
    # dual-language rule). Shadows are served by the route itself since
    # Phase 4a — the resolve kernel's event pass builds the shadow queue
    # from sheet records, so no second walk exists. The sheet resolve is
    # the ONLY resolve for analytic coverage (the fragment walk is deleted),
    # so its preconditions — SHEET_RESOLVE, ANALYTIC_AA_RUN, the sparse
    # toggles, samples <= 1, transparent background only without an env map
    # — all live inside analytic_raster_route_active, and the route IS the
    # analytic-raster decision. prepare_sparse_raster_coverage still raises
    # on an emission-side disagreement rather than painting a wrong frame.
    sheet_route = analytic_raster
    # Composed custom fragment-shader pipelines injected into the shade kernel as
    # a flat ti.template() tuple; empty () keeps the built-in / vertex-shaded
    # kernel specialization unchanged (see shading_taichi._run_frag_pipeline).
    # A non-empty ``frag_scatters`` tuple switches the monolithic shade kernel's
    # bounce block to per-material scatter dispatch (custom ray bouncing); it is
    # only assembled when a pipeline in *this* scene overrides bouncing, so an
    # ordinary scene keeps the byte-identical built-in bounce block (empty ()).
    if det_frag:
        from algan.rendering.shaders.fragment_shaders import (
            build_frag_pipelines,
            build_frag_scatters,
        )

        # Narrowed to the pipelines THIS batch's primitives carry. The registry
        # behind them is process-global and append-only, so handing over all of
        # it would specialize this render's shade kernel on every pipeline the
        # process ever registered -- a scene with no custom shader at all would
        # compile its own uncached kernel variant just because some earlier
        # scene had one.
        batch_pids = _batch_user_pipeline_ids(merged)
        frag_pipelines = build_frag_pipelines(batch_pids)
        frag_scatters = (
            build_frag_scatters(batch_pids) if _scene_has_custom_scatter(merged) else ()
        )
    else:
        frag_pipelines = ()
        frag_scatters = ()
    # Refraction (general wavefront only; see refractive_det above). A custom
    # scatter may spawn a transmitted branch, so it needs the same split pool +
    # transmitted-branch code the refraction path compiles in.
    # Continuation-ray supersampling makes every reflector a splitting path, so
    # it needs the same shared pool (see _secondary_split_needed). Deterministic
    # only: the Monte Carlo path tracer antialiases by jittered sampling already.
    refraction_flag = (
        1
        if (
            refractive_det
            or refl_transparent_det
            or frag_scatters
            or (samples <= 1 and _secondary_split_needed(merged, analytic_raster))
        )
        else 0
    )
    # Nested-IOR media stack (DESIGN_mesh_identity_open.md §H). Gated on
    # ``refraction_flag`` as well as the setting, and that conjunction is
    # load-bearing beyond "the stack only makes sense where rays split": the
    # stack rides a WIDER rs_sca and its initialisation story leans on the
    # split pool existing. Every batch with refraction_flag == 1 has
    # pool_ratio > 1 (REFRACT_INITIAL_POOL_RATIO >= 2), which is exactly what
    # excludes the two init paths that cannot write the new columns -- the
    # host const_fill broadcast (requires pool_ratio == 1) and fused
    # generation (same). Key this gate on the raw setting alone and those
    # paths would hand the kernels uninitialised stack columns to read.
    ior_stack_flag = (
        1 if (rt_settings.nested_ior_mode() != 0 and refraction_flag) else 0
    )
    # Environment map: append its texels to the shared texture buffer (the
    # merged dict is shallow-copied -- it is cached across batches) and, when
    # its ambient lighting is enabled, its SH irradiance as an extra light row.
    if det_frag:
        light_device = torch.device("cpu")
        light_pos_host, light_col_host, num_lights = _pack_lights(
            light_sources, num_frames, light_device
        )
        if env_map is not None and getattr(scene, "environment_ambient", True):
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
    elif samples > 1:
        light_pos = light_col = None
        num_lights = 0
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
    # asking a kernel: MAX_SHADOW_LIGHTS is the length of a ``ti.Vector`` and
    # therefore compile-time, so every light slot past it is simply never
    # written and the surplus lights render lit-but-shadowless. Counted per
    # batch, since a light spawning mid-scene can push a later batch over a cap
    # the first ones sat under. Each RectAreaLight emitter sample already
    # occupies its own row here (``_pack_lights`` expands them), which is why
    # the count is of light SLOTS rather than of the author's lights.
    if shadow_flag and num_lights > MAX_SHADOW_LIGHTS:
        record_truncation(
            "shadow_lights",
            int(num_lights) - MAX_SHADOW_LIGHTS,
            cap=MAX_SHADOW_LIGHTS,
        )

    # Frame counts of the windows that were actually launched. They differ from
    # the requested window whenever a chunk had to be sub-divided below, and the
    # batching loop's memory model needs the difference: it measures the arena's
    # high-water mark over this call and would otherwise credit a sub-divided
    # chunk's (smaller) peak to the frame count it planned, under-reading the
    # per-frame cost and planning the same over-large chunk again.
    launched_frames = []

    def rewind_to(pointers):
        """Rewind the arena to ``pointers`` without reclaiming the batch-wide
        raster tables.

        ``_build_raster_tables`` allocates them at the arena's persistent
        (reverse) end from inside the *first* chunk, caches them on ``merged``
        -- which lives for the whole batch -- and publishes the reverse pointer
        it reached. Every later chunk reads that cache instead of rebuilding.
        Restoring the reverse pointer to this chunk's entry value would hand
        their range back to the allocator while the cache still points into it,
        so the next chunk's forward allocations grow straight over the tables
        and ``_window_pairs`` reads garbage bounds (a negative bbox width
        surfaced as ``repeats can not be negative``). Only the split retry and
        the recursive halves could hit that: between the per-chunk rewind and
        the render loop's own ``set_pointers`` nothing else allocates.
        """
        forward, reverse = pointers
        if merged is not None and merged.get("_raster_tables") is not None:
            retained = merged.get("_raster_tables_reverse_pointer")
            if retained is not None:
                reverse = min(reverse, retained)
        memory.set_pointers((forward, reverse))

    def render_chunk(start, end):
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
        try:
            post_tonemap = is_post_process_tonemap_enabled()
            if rt_settings.LINEAR_COLOR_SPACE and not post_tonemap:
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
                    background_color,
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
                accum = None
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
            # Coplanar layer order: circuits < triangles < PN patches.
            layer_offset_triangles = float(merged["num_circuits"])
            shared_args = (
                tri_bvh.blocks,
                tri_bvh.node_miss,
                tri_bvh.leaf_prim,
                tri_bvh.leaf_tspan,
                tri_bvh.first_leaf,
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
                bez_bvh.first_leaf,
                merged["circuit_meta"],
                merged["circuit_colors"],
                merged["circuit_border_colors"],
                merged["edges_2d"],
                merged["edge_accel"],
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
                int(MAX_BOUNCES),
                1 if transparent_background else 0,
            )
            if samples > 1:
                from algan.rendering.raytracing.refit_bvh import RefitBVH

                path_trace_scene_stbvh(
                    1 if isinstance(tri_bvh, RefitBVH) else 0,
                    *shared_args,
                    samples_eff,
                    float(INDIRECT_BOUNCE_STRENGTH),
                    out,
                    accum,
                )
                finalize_samples(
                    samples_eff,
                    1 if transparent_background else 0,
                    t_val,
                    float(TONEMAP_EXPOSURE),
                    accum,
                    out,
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
                            int(MAX_BOUNCES),
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
            )
            rewind_to(entry_pointers)
            launched_frames.append(end - start)
            return [frames]
        except (InsufficientMemoryException, RuntimeError) as exc:
            # A Taichi kernel launch (e.g. the post-process tonemap) exhausts
            # VRAM as a plain RuntimeError from its own allocator, not a torch
            # OOM; recognise it so the same rewind + empty_cache + split retry
            # recovers it. Any non-OOM RuntimeError is a real error -- re-raise.
            if not isinstance(exc, InsufficientMemoryException) and not is_cuda_oom(
                exc
            ):
                raise
            logger.log(PERF, f"Reducing the frame batch to fit memory: {start}:{end}")
            rewind_to(entry_pointers)
            restore_truncations(entry_truncations)
            # All this stuff is necessary to free local variables assigned during the previous render attempt.
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.clear_frames(exc_traceback)
            # traceback.print_tb(exc_traceback)
            # exc_traceback.tb_next.tb_frame.clear()
            # Release the failed allocation (e.g. the wavefront's large per-ray
            # state) so it doesn't fragment/block the smaller retry.
            empty_cache()
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
    scene.last_render_plan = attach_truncations(plan)
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
    overflowing attempt is discarded before compositing and retried with half
    as many primaries while retaining the same pool capacity. Thus the
    continuation headroom doubles on every retry without increasing arena
    memory, and there is no fixed per-pixel split limit.
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
                    state_ptrs = memory.get_pointers()
                    try:
                        # Per-ray state for one tile: ``pool`` slots plus
                        # ``attempt_primary`` per-primary rows. Calibrated as
                        # unit coefficients (bytes per slot, per primary, and
                        # fixed per tile) rather than as a peak -- under
                        # WAVEFRONT_TILE_AUTO the tile is sized from whatever
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
                    except (InsufficientMemoryException, RuntimeError):
                        memory.set_pointers(state_ptrs)
                        raise

                    alloc = _read_tile_alloc(rs_alloc)
                    overflow = pool_ratio > 1 and alloc[ALLOC_OVERFLOW] != 0
                    if overflow:
                        memory.set_pointers(state_ptrs)
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
                            float(rt_settings.TONEMAP_EXPOSURE),
                            0,
                            0,
                            covered_dummy,
                            0,
                            out,
                        )
                    memory.set_pointers(state_ptrs)
                    tile_start += attempt_primary
                    break

    if do_aa:
        wf_finalize_aa(
            int(width),
            int(height),
            1 if transparent else 0,
            float(inv_aa * inv_aa),
            t_val,
            float(rt_settings.TONEMAP_EXPOSURE),
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
):
    """Wavefront orchestration for the general triangle/PN/bezier path.

    Persistent continuation state is stage-split in global memory and PyTorch
    compacts ray indices between host iterations. Hit records are different:
    traversal writes one exact-size ``[num_active, KBUF]`` transient event
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

    When fragment shading is active, ``settings.WAVEFRONT_SORT_MATERIALS``
    selects the shade architecture: the monolithic ``wavefront_shade`` kernel
    below (the default, and the only supported path -- it handles custom
    scatter and normal-mapped lighting, and on the built-in materials it is
    faster because it drains up to KBUF hits per launch while sorting pays
    per-event kernel round trips and host syncs), or the UNSUPPORTED legacy
    Cycles-style *sorted* pipeline (rays suspended at their material events,
    bucketed by (geometry type, material pipeline id) and shaded by dedicated
    per-material kernels -- see ``wavefront_sorted_kernels_taichi``), routed
    only when explicitly forced (``set_material_sorting(True)``) and kept for
    reference only. The vertex-shaded path below is unaffected either way.
    """
    rt_settings = SETTINGS.raytracing
    # rs_sca's row width for this batch. A local, not a call at each use site:
    # ``sca_width`` is also the name of the module-level helper imported from
    # wavefront_kernels_taichi, so a bare ``sca_width`` inside this function is
    # the FUNCTION, not a width (that mistake allocated ray state with a
    # function object as its column count).
    state_sca_width = sca_width(ior_stack_flag)
    # UNSUPPORTED legacy textured-surface shader (Surface / flat-triangle
    # scenes): shades from three per-triangle texture lookups instead of
    # per-vertex arrays. Only reachable via the opt-in WF_TEXTURED toggle;
    # kept for reference. Extended lights, environment maps and near/far
    # clipping live in the monolithic general shade kernel below; a scene
    # using any of them skips the textured / sorted variants.
    uses_extended_features = (
        bool(lights_extended)
        or env_meta is not None
        or near_clip > 0.0
        or far_clip > 0.0
    )
    if merged.get("textured_active") and not uses_extended_features:
        return _raytrace_render_wavefront_textured(
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
            refraction_flag,
            transparent,
            memory,
            out,
            aa_level,
        )
    sort_mode = rt_settings.WAVEFRONT_SORT_MATERIALS
    # The monolith handles custom scatter + normal maps, so it is the default
    # for every fragment-shaded scene; the UNSUPPORTED legacy sorted pipeline
    # runs only when explicitly forced (kept for reference -- see settings).
    use_sorted = (
        bool(frag_flag)
        and (sort_mode is True)
        and not uses_extended_features
        and not merged.get("bez_has_reflective", False)
    )
    if use_sorted:
        return _raytrace_render_wavefront_sorted(
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
            frag_pipelines,
            shadow_flag,
            refraction_flag,
            transparent,
            memory,
            out,
            aa_level,
        )
    i32 = torch.int32
    f32 = torch.float32
    max_iters = MAX_SURFACES_PER_RAY + max_bounces * 2 + 4
    n = (time_end - time_start) * width * height

    # Compile-time walk selector: the merge builds either all-classic or
    # all-refit trees for a batch (see scene_builder._build_accel), so the
    # tree object's type is the authority -- never the live toggle, which the
    # user may have flipped since this batch was merged/prewarmed.
    from algan.rendering.raytracing.refit_bvh import RefitBVH

    bvh_refit = 1 if isinstance(tri_bvh, RefitBVH) else 0

    # Pool over-allocation for ray splitting. Glass (reflective+refractive)
    # surfaces split, and so does any reflector under continuation-ray
    # supersampling; the plain single-ray path keeps pool_ratio == 1 (one slot
    # per pixel, as before).
    pool_ratio = _split_pool_ratio(
        refraction_flag, merged, analytic_raster, bool(frag_scatters)
    )
    # Read live (settings convention): runtime-mutable for tile-size A/B.
    primary_per_tile = max(1, rt_settings.WAVEFRONT_TILE_RAYS // pool_ratio)

    # Family A+B memory-trim: engage only for the no-shadow, non-refractive,
    # scatter-free triangle path (the trim arrays are built by scene_builder
    # only when ALGAN_WF_MEM_TRIM). Rebinds the triangle geometry + BVH to the
    # band-reordered/compacted variants and supplies the col_row remap; PN and
    # bezier are untouched. tri_colors/tri_extra stay in their original order
    # (addressed via col_row). ``mem_trim == 0`` leaves everything byte-identical.
    mem_trim = (
        1
        if (
            rt_settings.WF_MEM_TRIM
            and merged.get("mem_trim_active")
            and shadow_flag == 0
            and len(frag_scatters) == 0
            and refraction_flag == 0
        )
        else 0
    )
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
    # ``opaque_bvh_skipped``: the merge aliased the opaque trees to the main
    # ones because neither rollout was live when it ran -- keep them off for
    # this batch even if a toggle flipped since (the dedicated trees do not
    # exist to walk).
    opaque_closest = int(
        rt_settings.WF_OPAQUE_CLOSEST
        and not merged.get("opaque_bvh_skipped", False)
        and merged.get("all_visible_opaque", False)
        and not refraction_flag
        and len(frag_scatters) == 0
        and not mem_trim
        and not merged.get("textured_active", False)
    )
    opaque_prepass = int(
        rt_settings.WF_OPAQUE_PREPASS
        and not merged.get("opaque_bvh_skipped", False)
        and merged.get("has_any_opaque", False)
        and merged.get("has_any_translucent", False)
        and not merged.get("has_uncertain_texture_alpha", False)
        and not refraction_flag
        and len(frag_scatters) == 0
        and not mem_trim
        and not merged.get("textured_active", False)
    )
    # Compile-time material gating of the shade kernels: the pipeline ids this
    # batch's triangles / PN patches carry, as a bitmask template, so the
    # stages it cannot reach are never compiled in (rt_settings.FRAG_PID_GATE;
    # ALL_PIDS = ungated). Per geometry type, because the two shade sites are
    # separate funcs -- a mesh scene's flat triangles and its PN patches each
    # get their own gate.
    tri_pids = _frag_pid_mask(merged, "tri", has_tri)
    # Hybrid raster front-end: primary visibility resolved by the sheet route
    # (DESIGN_sheet_resolve.md) — the sparse emission compacts exact analytic
    # coverage into per-pixel sheets and one kernel resolves, builds shadow
    # events for, and shades them; only bounced continuations enter the
    # classic wavefront loop. PN patches are conservatively routed to the
    # classic path without altering their geometry. This re-derives the route
    # from the live toggles and the merged batch facts so drift against the
    # allocation-time decision (analytic_raster, which IS the sheet route) is
    # caught below rather than rendered wrong.
    use_raster = (
        rt_settings.HYBRID_RASTER
        and analytic_raster
        and merged.get("tri_frame_valid") is not None
        and (merged["num_triangles"] > 0 or merged["num_circuits"] > 0)
        and not merged.get("textured_active")
        and mem_trim == 0
        and len(frag_scatters) == 0
        and near_clip <= 0.0
        and max(1, int(aa_level)) <= 1
        and rt_settings.RASTER_SPARSE_COVERAGE
        and rt_settings.RASTER_EMPTY_SKIP
        and rt_settings.RASTER_COVERED_SHADE
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

    # Fused primary-ray generation (settings.WF_GEN_FUSED): the tile's first
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
        from algan.rendering.raytracing.scene_builder import build_deferred_bvhs

        build_deferred_bvhs(merged)
        tri_bvh = merged["tri_bvh"]
        bez_bvh = merged["bez_bvh"]
        t_bvh = tri_bvh
        bvh_refit = 1 if isinstance(tri_bvh, RefitBVH) else 0

    if merged.get("bvh_deferred") and (shadow_flag != 0 or not use_raster):
        # Runtime routing needs the trees after all: primary shadows trace
        # them from the sheet resolve's event queue (raster_shadow_trace),
        # and a batch that fell back to classic primary traversal (near
        # clip, in-place AA, flipped toggles, ...) walks them for every
        # primary ray.
        _ensure_bvhs()

    gen_fused = (
        rt_settings.wf_gen_fused_active()
        and pool_ratio == 1
        and near_clip <= 0.0
        and max(1, int(aa_level)) <= 1
        and not use_raster
    )
    # Route metadata words: a handful of floats whose count depends on the
    # selected route, paid once per batch. Separate from the raster precompute
    # tables below so the latter's per-(frame, primitive) coefficient is not
    # fitted through a route-dependent constant.
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
    if gen_fused or use_raster or env_meta is not None or far_clip > 0.0:
        # Extras packed behind the two layer offsets (the shade kernel is at
        # the 64-arg ceiling): env map placement in the shared texel buffer +
        # the camera's far clip distance, and -- read only by the fused first
        # shade iteration -- max_bounces. The kernel detects them by length.
        eo, ew, eh, ei = env_meta if env_meta is not None else (0, 0, 0, 0.0)
        layer_values = [
            float(layer_offset_triangles),
            float(eo),
            float(ew),
            float(eh),
            float(ei),
            float(far_clip),
            float(max_bounces),
            # [7] the split-sum glossy route's first glossy accumulator row,
            # rewritten per tile by the sparse loop below and 0 (inert)
            # everywhere else. It rides here rather than as a kernel argument
            # for the same reason the env placement does -- see the comment
            # above and DESIGN_glossy_prefilter.md §4.3.
            0.0,
        ]
        with memory.scope(
            "batch_metadata",
            gen_fused=int(bool(gen_fused)),
            raster=int(bool(use_raster)),
            extended=int(env_meta is not None or far_clip > 0.0),
        ):
            layer_offsets_t = _arena_values(memory, layer_values, f32)
    else:
        with memory.scope(
            "batch_metadata",
            gen_fused=int(bool(gen_fused)),
            raster=int(bool(use_raster)),
            extended=int(env_meta is not None or far_clip > 0.0),
        ):
            layer_offsets_t = _arena_values(
                memory, [float(layer_offset_triangles)], f32
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
        # RenderLoopMixin.render_primitive_batch).
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
            merged["_raster_tables_reverse_pointer"] = memory.get_pointers()[1]

    def _drain_sparse_secondary(
        active, state, rs_pix, pix_accum, rs_alloc, compactor, rs_vis
    ):
        """Run iterations >= 1 for compact raster primaries.

        ``rs_pix`` contains the real window-local pixel while
        ``rs_int[:, 4]`` contains the compact accumulator row.  A zero
        ``ray_offset`` therefore addresses the full prepared frame window.
        """
        (
            rs_ro,
            rs_rd,
            rs_acc,
            rs_sca,
            rs_int,
            _rs_kt,
            _rs_kl,
            _rs_ka,
            _rs_kb,
            _rs_kp,
            _rs_kf,
        ) = state
        it = 1
        while active.numel() > 0 and it < max_iters:
            na = int(active.numel())
            with memory.temp():
                hit_f = memory.get_tensor((na, KBUF, 4), f32)
                hit_i = memory.get_tensor((na, KBUF, 2), i32)
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
                    layer_offsets_t,
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
                    int(rt_settings.WF_SKIP_UNLIT_NORMAL),
                    int(rt_settings.DIRECT_SPECULAR_LOBE),
                    int(mem_trim),
                    opaque_closest,
                    0,
                    1,  # compact: rs_int[:, 4] holds the accumulator row
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
                    rs_vis,
                    cam_origin,
                )
            active = compactor.select(
                rs_int,
                0,
                source=active,
                scan_pool=(pool_ratio != 1 or not rt_settings.WF_COMPACT_ACTIVE_ONLY),
            )
            it += 1

    if sparse_coverage:
        from algan.rendering.raytracing.raster_pipeline import (
            prepare_sparse_raster_coverage,
            shade_sparse_raster_coverage,
        )

        # Sparse hit records live at the arena's reverse end for the duration
        # of the window.  Coverage-sized ray pools are allocated/reset from the
        # forward end one compact slice at a time.
        with memory.temp(clear_persist=True):
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
            )
            t_val_sparse = _get_tonemap_t_val()
            if t_val_sparse != 3:
                # In-kernel tonemap on the sparse route (sheet unification,
                # DESIGN_sheet_resolve.md §4.8): every pixel the covered
                # composite will not touch owes finalize(bg) — including the
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
                        float(rt_settings.TONEMAP_EXPOSURE),
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
            # every free arena byte (WAVEFRONT_TILE_SAFETY is 1.0), so a buffer
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
                    width, height, int(rt_settings.GLOSSY_PREFILTER_MAX_LEVELS)
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
            pix_accum_rows, pix_accum_cols = (2, GL_ROW_WIDTH) if gl_active else (1, 7)
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
                primary_capacity, sparse_primary, pool_ratio, analytic_raster
            )
            learned_primary_cap = primary_capacity
            covered_start = 0

            while covered_start < num_covered_total:
                remaining = num_covered_total - covered_start
                attempt_primary = min(learned_primary_cap, remaining)
                if gl_active:
                    while gl_bounds[gl_frame + 1] <= covered_start:
                        gl_frame += 1
                    attempt_primary = min(
                        attempt_primary, gl_bounds[gl_frame + 1] - covered_start
                    )
                pool = shared_pool_capacity if pool_ratio > 1 else attempt_primary

                while True:
                    state_ptrs = memory.get_pointers()
                    try:
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
                                (attempt_primary * pix_accum_rows, pix_accum_cols),
                                f32,
                            )
                            rs_alloc = memory.get_tensor((ALLOC_WIDTH,), i32)
                            rs_vis = memory.get_tensor((1,), i32)
                            compactor = _ArenaRayCompactor(memory, pool, i32)
                        rs_int = state[4]
                        if state_sca_width != SCA_WIDTH_PLAIN:
                            # Same per-tile stack zeroing as the dense tile
                            # above (DESIGN_mesh_identity_open.md §H).
                            state[3][:, _SCA_IOR_DEPTH:].zero_()
                        pix_accum.zero_()
                        if gl_active:
                            # A glossy ray that never hits anything writes no
                            # distance at all, and that has to read as "the
                            # reflection is infinitely far", i.e. fully
                            # blurred -- not as the zero a cleared buffer
                            # would give, which is a contact reflection.
                            pix_accum[attempt_primary:, GL_ROW_DIST] = float("inf")
                            layer_offsets_t[7] = float(attempt_primary)
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
                                layer_offsets_t,
                                gen_meta,
                                light_pos,
                                light_col,
                                num_lights,
                                col_row_arr,
                                frag_flag,
                                frag_pipelines,
                                int(tri_pids),
                                int(rt_settings.WF_SKIP_UNLIT_NORMAL),
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
                            )
                    except (InsufficientMemoryException, RuntimeError) as exc:
                        # Taichi launches OOM as a bare RuntimeError from their
                        # own allocator; treat those as OOM, re-raise real ones.
                        if not isinstance(
                            exc, InsufficientMemoryException
                        ) and not is_cuda_oom(exc):
                            raise
                        memory.set_pointers(state_ptrs)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if attempt_primary <= 1:
                            raise OutOfRenderMemory(
                                "Sparse raster state did not fit for one "
                                "covered pixel. Lower the resolution or "
                                "transparency complexity."
                            ) from exc
                        next_primary = max(1, attempt_primary // 2)
                        _WAVEFRONT_POOL_RETRIES[0] += 1
                        learned_primary_cap = min(learned_primary_cap, next_primary)
                        attempt_primary = next_primary
                        # A splitting batch holds the pool fixed across the
                        # OVERFLOW retry on purpose (the shrunken tile inherits
                        # the spare slots), but the pool is also the tile's
                        # dominant allocation -- so halving only the primaries
                        # after a MEMORY failure re-attempts a state block of
                        # very nearly the same size, fails identically, and
                        # rides all the way down to the one-pixel diagnostic
                        # without ever having freed anything. Halve the pool
                        # with it, keeping the primaries' own slots (the shared
                        # allocator starts at ``attempt_primary``); an
                        # under-sized pool overflows, and the overflow retry
                        # above is exact and terminating.
                        if pool_ratio > 1:
                            shared_pool_capacity = max(
                                next_primary, shared_pool_capacity // 2
                            )
                            pool = shared_pool_capacity
                        else:
                            pool = attempt_primary
                        continue

                    # The resolve's own overflow, checked before the bounce
                    # drain adds to the same counters. Truncations are folded
                    # in at the ACCEPT point below, once, so this stage keeps
                    # the split-free short-circuit it always had.
                    overflow = (
                        pool_ratio > 1 and int(rs_alloc[ALLOC_OVERFLOW].item()) != 0
                    )
                    if overflow:
                        memory.set_pointers(state_ptrs)
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
                        active = compactor.select(
                            rs_int, 0, source=compactor.current, scan_pool=True
                        )
                        if active.numel() > 0 and merged.get("bvh_deferred"):
                            _ensure_bvhs()
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
                        memory.set_pointers(state_ptrs)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if attempt_primary <= 1:
                            raise OutOfRenderMemory(
                                "Sparse raster bounce scratch did not fit for "
                                "one covered pixel. Lower the resolution or "
                                "transparency complexity."
                            ) from exc
                        next_primary = max(1, attempt_primary // 2)
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
                    alloc = _read_tile_alloc(rs_alloc)
                    if pool_ratio > 1 and alloc[ALLOC_OVERFLOW] != 0:
                        memory.set_pointers(state_ptrs)
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

                    if gl_active:
                        # BEFORE the composite: both read the frame buffer's
                        # raw prefilled background, and the composite
                        # overwrites it with the finalized pixel.
                        gloss_scatter(
                            int(attempt_primary),
                            int(attempt_primary),
                            gl_frame * int(width) * int(height),
                            gl_frame,
                            int(width),
                            float(gl_sigma_max),
                            covered_idx,
                            pix_accum,
                            gl_main,
                            gl_pyr,
                            out,
                        )

                    wf_composite_accum_sparse(
                        int(time_start),
                        int(width),
                        int(height),
                        1 if transparent else 0,
                        0,
                        covered_idx,
                        pix_accum[:attempt_primary] if gl_active else pix_accum,
                        t_val_sparse,
                        float(rt_settings.TONEMAP_EXPOSURE),
                        out,
                    )
                    memory.set_pointers(state_ptrs)
                    covered_start += attempt_primary
                    if gl_active and covered_start >= gl_bounds[gl_frame + 1]:
                        # Frame complete: prefilter its reflection buffer and
                        # composite the glossy pixels over the values the tile
                        # composites just wrote for them.
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
            # [active ray, KBUF] surface-event record and shade consumes it in
            # the same host iteration. Releasing this arena scope before
            # compaction removes the six permanent [pool, KBUF] arrays from
            # secondary radiance state while preserving the existing four-hit
            # traversal/shading behavior.
            with memory.temp():
                hit_f = memory.get_tensor((na, KBUF, 4), f32)
                hit_i = memory.get_tensor((na, KBUF, 2), i32)
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
                    layer_offsets_t,
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
                    int(rt_settings.WF_SKIP_UNLIT_NORMAL),
                    int(rt_settings.DIRECT_SPECULAR_LOBE),
                    int(mem_trim),
                    opaque_closest,
                    first,
                    0,  # compact: dense tiles accumulate at the ray's pixel
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
                scan_pool=(pool_ratio != 1 or not rt_settings.WF_COMPACT_ACTIVE_ONLY),
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


def _raytrace_render_wavefront_textured(
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
    refraction_flag,
    transparent,
    memory,
    out,
    aa_level=1,
):
    """UNSUPPORTED legacy textured-surface wavefront orchestration (Surface /
    flat-triangle scenes only; no longer maintained, kept for reference --
    see ``settings.WF_TEXTURED``). Same generate -> traverse -> shade ->
    composite tile loop as the monolithic :func:`raytrace_render_wavefront`,
    but the shade stage is ``wf_shade_textured`` reading the three
    per-triangle texture banks built by
    ``scene_builder._build_textured_scene``. PN and bezier traversals gate out
    (the scene is all flat triangles).
    """
    from algan.rendering.raytracing.wavefront_textured_kernels_taichi import (
        wf_shade_textured,
    )

    rt_settings = SETTINGS.raytracing

    i32 = torch.int32
    f32 = torch.float32
    max_iters = MAX_SURFACES_PER_RAY + max_bounces * 2 + 4
    n = (time_end - time_start) * width * height

    # Feature templates (compiled into the shade kernel one at a time to measure
    # each feature's cost -- see settings.WF_TEXTURED_FEATURES).
    feat = int(rt_settings.WF_TEXTURED_FEATURES)
    feat_bez = 1 if (feat & rt_settings.WF_TEX_BEZ) else 0
    feat_scatter = 1 if (feat & rt_settings.WF_TEX_SCATTER) else 0
    feat_shadows = 1 if (feat & rt_settings.WF_TEX_SHADOWS) else 0
    feat_normalmap = 1 if (feat & rt_settings.WF_TEX_NORMALMAP) else 0
    # Compile the bezier traversal into the (shared) traverse kernel when the
    # bezier feature is on, so its cost is included even on a bezier-free scene.
    has_bez_eff = 1 if (feat_bez or has_bez) else 0

    pool_ratio = rt_settings.refract_initial_pool_ratio if refraction_flag else 1
    primary_per_tile = max(1, rt_settings.WAVEFRONT_TILE_RAYS // pool_ratio)
    # The gen block compiles out on this path (the classic generate kernel is
    # kept), but traverse still reads gen_meta[2:] to rebuild each pixel's
    # primary ray for the perpendicular-depth conversion (see _axis_cos), so
    # these carry the real screen half-extents rather than zeros.
    gen_meta = _arena_values(
        memory, [0.5, 0.5, float(half_screen_w), float(half_screen_h)], f32
    )

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
        compactor = _ArenaRayCompactor(memory, pool, i32)
        active = compactor.initial(tn_primary)
        it = 0
        while active.numel() > 0 and it < max_iters:
            na = int(active.numel())
            wavefront_traverse(
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
                0,
                int(has_tri),
                int(has_bez_eff),
                0,
                0,
                int(time_start),
                int(width),
                int(height),
                int(tile_start),
                rs_ro,
                rs_rd,
                rs_sca,
                rs_int,
                rs_kt,
                rs_kl,
                rs_ka,
                rs_kb,
                rs_kp,
                rs_kf,
                rs_pix,
                0,
                cam_origin,
                screen_point,
                pixel_basis_x,
                pixel_basis_y,
                gen_meta,
            )
            wf_shade_textured(
                active,
                na,
                merged["tri_pos"],
                merged["tri_norm"],
                merged["tx_uv"],
                merged["tx_color_idx"],
                merged["tx_mat_idx"],
                merged["tx_surf_idx"],
                merged["tx_color_bank"],
                merged["tx_color_meta"],
                merged["tx_mat_bank"],
                merged["tx_mat_meta"],
                merged["tx_surf_bank"],
                merged["tx_surf_meta"],
                merged["tx_nmap_idx"],
                merged["tx_nmap_bank"],
                merged["tx_nmap_meta"],
                merged["circuit_meta"],
                merged["circuit_colors"],
                merged["circuit_border_colors"],
                tri_bvh.blocks,
                tri_bvh.node_miss,
                tri_bvh.leaf_prim,
                tri_bvh.leaf_tspan,
                int(tri_bvh.first_leaf),
                pixel_world_scale,
                float(layer_offset_triangles),
                light_pos,
                light_col,
                int(num_lights),
                int(refraction_flag),
                int(feat_bez),
                int(feat_scatter),
                int(feat_shadows),
                int(feat_normalmap),
                int(time_start),
                int(width),
                int(height),
                int(tile_start),
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
                rs_pix,
                pix_accum,
                rs_alloc,
            )
            active = compactor.select(
                rs_int,
                0,
                source=active,
                scan_pool=(pool_ratio != 1 or not rt_settings.WF_COMPACT_ACTIVE_ONLY),
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
        near_clip=0.0,
        run_tile=run_tile,
        # Compactor output-count word.
        auto_fixed_bytes=torch.int32.itemsize,
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

    scatters = build_frag_scatters()
    for prefix in ("tri", "pn"):
        material_ids = merged.get(f"{prefix}_material_ids")
        if material_ids is None:
            arr = merged.get(f"{prefix}_mat_id")
            material_ids = (
                ()
                if arr is None or not arr.numel()
                else torch.unique(arr.detach().cpu()).tolist()
            )
        for pid in material_ids:
            i = int(pid) - _USER_PIPELINE_BASE
            if 0 <= i < len(scatters) and scatters[i] is not None:
                merged["has_custom_scatter"] = True
                return True
    merged["has_custom_scatter"] = False
    return False


def _batch_user_pipeline_ids(merged):
    """The user fragment-pipeline ids this batch's primitives carry, for
    narrowing the injected pipeline / scatter tuples to what the batch can
    reach (``fragment_shaders.build_frag_pipelines``).

    Returns ``None`` -- "the whole registry", the safe answer -- when a
    geometry type is present but the merged scene cannot enumerate its
    material ids, for the same reason ``_frag_pid_mask`` returns ``ALL_PIDS``
    there: dropping a pipeline the kernel can still dispatch to would shade
    that surface with no material at all. Reads only the merge-time host-side
    id lists, so it costs no device reduction.
    """
    ids = set()
    for prefix in ("tri", "pn"):
        material_ids = merged.get(f"{prefix}_material_ids")
        if material_ids:
            ids.update(
                pid
                for pid in (int(value) for value in material_ids)
                if pid >= _USER_PIPELINE_BASE
            )
            continue
        arr = merged.get(f"{prefix}_mat_id")
        if arr is not None and arr.numel():
            return None
    return frozenset(ids)


def _frag_pid_mask(merged, prefix, active, _record=True):
    """Compile-time bitmask of the material pipeline ids ``prefix`` geometry
    carries in this batch (bit ``p`` = pipeline id ``p``), for the shade
    kernels' compile-time material gating (see ``rt_settings.FRAG_PID_GATE``
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
    if rt_settings.FRAG_PID_GATE and active and material_ids:
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


def _raytrace_render_wavefront_sorted(
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
    frag_pipelines,
    shadow_flag,
    refraction_flag,
    transparent,
    memory,
    out,
    aa_level=1,
):
    """UNSUPPORTED legacy Cycles-style sorted-material orchestration of the
    fragment-shading wavefront (no longer maintained, kept for reference; only
    reachable via ``set_material_sorting(True)`` -- see
    ``wavefront_sorted_kernels_taichi`` for the kernel split).

    Per host iteration: rays needing a K-buffer refill are traversed
    (``wavefront_traverse``, unchanged), ``wf_peel`` advances every ray with
    unconsumed hits to its next material event (compositing bezier hits and
    background escapes inline), the pending events get their shadow bits from
    one ``wf_shadow_event`` launch, and each material bucket -- rays whose
    event key ``(geometry type << 8) | pipeline id`` matches -- is shaded by a
    dedicated ``wf_shade_event`` instantiation with that material's pipeline
    and scatter funcs as compile-time templates. The bucket table is built
    once per chunk from the merged scene's material ids, so only materials
    actually present cost a kernel instantiation.
    """
    from algan.rendering.raytracing.shading_taichi import (
        _USER_PIPELINE_BASE,
        builtin_pipeline_fn,
    )
    from algan.rendering.raytracing.wavefront_sorted_kernels_taichi import (
        ST_PEEL,
        ST_SHADE,
        ST_TRAVERSE,
        default_scatter,
        wf_peel,
        wf_shade_event,
        wf_shadow_event,
    )
    from algan.rendering.shaders.fragment_shaders import build_frag_scatters

    i32 = torch.int32
    f32 = torch.float32
    n = (time_end - time_start) * width * height

    # Material bucket table: one entry per (geometry type, pipeline id) pair
    # present in the merged scene. Each bucket carries the composed pipeline
    # func + scatter func to inject and the geometry type's parameter block.
    scatters = build_frag_scatters()

    def _resolve(pid):
        if pid < _USER_PIPELINE_BASE:
            return builtin_pipeline_fn(pid), default_scatter
        i = pid - _USER_PIPELINE_BASE
        fn = frag_pipelines[i]
        # Both tuples are trimmed after their last live entry, so a pipeline
        # past the end of the scatter list is simply scatterless.
        sc = scatters[i] if i < len(scatters) else None
        return fn, (sc if sc is not None else default_scatter)

    buckets = []
    has_custom_scatter = False
    if merged["num_triangles"] > 0:
        tri_material_ids = merged.get("tri_material_ids")
        if tri_material_ids is None:
            tri_material_ids = torch.unique(
                merged["tri_mat_id"].detach().cpu()
            ).tolist()
        for pid in tri_material_ids:
            fn, sc = _resolve(int(pid))
            has_custom_scatter |= sc is not default_scatter
            buckets.append(((1 << 8) | int(pid), fn, sc, merged["tri_mat"]))
    # A custom scatter may spawn transmitted branches, which need the glass
    # split pool (and the peel's IOR sampling) even in a scene with no
    # refractive surface.
    refraction_flag = 1 if (refraction_flag or has_custom_scatter) else 0

    # Worst case: every one of MAX_SURFACES_PER_RAY hits is a material event
    # (one peel+shade pass each) plus a traverse per K-buffer refill / bounce.
    max_iters = (
        MAX_SURFACES_PER_RAY + MAX_SURFACES_PER_RAY // KBUF + max_bounces * 2 + 8
    )

    pool_ratio = rt_settings.refract_initial_pool_ratio if refraction_flag else 1
    # The sorted path carries ~1.5x the classic per-ray state (the event
    # record + keys), so tiles hold fewer rays for the same memory envelope.
    primary_per_tile = max(1, (rt_settings.WAVEFRONT_TILE_RAYS * 2) // (3 * pool_ratio))
    # The gen block compiles out on this path (the classic generate kernel is
    # kept), but traverse still reads gen_meta[2:] to rebuild each pixel's
    # primary ray for the perpendicular-depth conversion (see _axis_cos), so
    # these carry the real screen half-extents rather than zeros.
    gen_meta = _arena_values(
        memory, [0.5, 0.5, float(half_screen_w), float(half_screen_h)], f32
    )

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
        # Event state: hit record, sort key, event primitive index and
        # per-event shadow visibility bits (placeholder when unused).
        rs_hit = memory.get_tensor((pool, 16), f32)
        rs_key = memory.get_tensor((pool,), i32)
        rs_eprim = memory.get_tensor((pool,), i32)
        rs_vis = memory.get_tensor((pool,) if shadow_flag else (1,), i32)
        compactor = _ArenaRayCompactor(memory, pool, i32)
        # The drained counter (rs_int col 4, pool garbage after
        # allocation) must be 0 for every ray entering ST_TRAVERSE;
        # the kernels maintain that invariant from here on.
        rs_int[:, 4].zero_()

        it = 0
        while it < max_iters:
            # Build every work list directly in the arena. Pending SHADE
            # events never survive an iteration, so TRAVERSE then PEEL scans
            # are sufficient to decide termination.
            trav = compactor.select(rs_int, ST_TRAVERSE, scan_pool=True)
            if trav.numel():
                wavefront_traverse(
                    trav,
                    int(trav.numel()),
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
                    0,
                    int(has_tri),
                    int(has_bez),
                    0,
                    0,
                    int(time_start),
                    int(width),
                    int(height),
                    int(tile_start),
                    rs_ro,
                    rs_rd,
                    rs_sca,
                    rs_int,
                    rs_kt,
                    rs_kl,
                    rs_ka,
                    rs_kb,
                    rs_kp,
                    rs_kf,
                    rs_pix,
                    0,
                    cam_origin,
                    screen_point,
                    pixel_basis_x,
                    pixel_basis_y,
                    gen_meta,
                )
            # Traversal transitions its rays to PEEL, so one post-traverse
            # scan naturally includes both those rays and previously-peeling
            # rays without a temporary torch.cat.
            peel_idx = compactor.select(rs_int, ST_PEEL, scan_pool=True)
            if trav.numel() == 0 and peel_idx.numel() == 0:
                break
            if peel_idx.numel():
                wf_peel(
                    peel_idx,
                    int(peel_idx.numel()),
                    merged["tri_pos"],
                    merged["tri_norm"],
                    merged["tri_extra"],
                    merged["tri_colors"],
                    merged["tri_uvs"],
                    merged["tri_tex_meta"],
                    merged["textures"],
                    int(merged["num_colored_triangles"]),
                    merged["circuit_meta"],
                    merged["circuit_colors"],
                    merged["circuit_border_colors"],
                    merged["tri_mat_id"],
                    int(refraction_flag),
                    int(has_tri),
                    int(has_bez),
                    int(time_start),
                    int(width),
                    int(height),
                    int(tile_start),
                    rs_acc,
                    rs_sca,
                    rs_int,
                    rs_kt,
                    rs_kl,
                    rs_ka,
                    rs_kb,
                    rs_kp,
                    rs_kf,
                    rs_pix,
                    rs_hit,
                    rs_key,
                    rs_eprim,
                    pix_accum,
                )
            shade_all = compactor.select(rs_int, ST_SHADE, scan_pool=True)
            if shade_all.numel():
                if shadow_flag:
                    wf_shadow_event(
                        shade_all,
                        int(shade_all.numel()),
                        tri_bvh.blocks,
                        tri_bvh.node_miss,
                        tri_bvh.leaf_prim,
                        tri_bvh.leaf_tspan,
                        int(tri_bvh.first_leaf),
                        merged["tri_pos"],
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
                        pixel_world_scale,
                        float(layer_offset_triangles),
                        int(has_tri),
                        int(has_bez),
                        light_pos,
                        int(num_lights),
                        int(time_start),
                        int(width),
                        int(height),
                        int(tile_start),
                        rs_ro,
                        rs_rd,
                        rs_sca,
                        rs_hit,
                        rs_pix,
                        rs_vis,
                    )
                for key_val, fn, sc, mat in buckets:
                    bidx = compactor.select(
                        rs_int,
                        ST_SHADE,
                        scan_pool=True,
                        rs_key=rs_key,
                        desired_key=key_val,
                    )
                    cnt = int(bidx.numel())
                    if cnt == 0:
                        continue
                    wf_shade_event(
                        bidx,
                        cnt,
                        mat,
                        light_pos,
                        light_col,
                        int(num_lights),
                        fn,
                        sc,
                        int(shadow_flag),
                        int(refraction_flag),
                        int(time_start),
                        int(width),
                        int(height),
                        int(tile_start),
                        rs_ro,
                        rs_rd,
                        rs_acc,
                        rs_sca,
                        rs_int,
                        rs_hit,
                        rs_eprim,
                        rs_pix,
                        pix_accum,
                        rs_alloc,
                        rs_vis,
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
        near_clip=0.0,
        run_tile=run_tile,
        # rs_hit(16 f32) + rs_key + rs_eprim (+ rs_vis with shadows) per slot.
        auto_extra_slot_bytes=16 * 4 + 4 + 4 + (4 if shadow_flag else 0),
        # Compactor output-count word, plus the one-word rs_vis placeholder
        # when shadow visibility is not a per-slot array.
        auto_fixed_bytes=(torch.int32.itemsize * (1 if shadow_flag else 2)),
    )


_originals = {}


def is_ray_tracing_enabled():
    """Vestigial: always False. The ray-traced primitive classes are now the
    engine's only renderer (``RENDERER_REGISTRY`` binds them by default), and
    the ``enable_ray_tracing`` toggle that used to populate ``_originals`` was
    removed with the rasterizer. Kept only because ``post_processing.bloom``
    probes for it defensively.
    """
    return bool(_originals)
