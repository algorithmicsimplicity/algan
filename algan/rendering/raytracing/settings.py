import os

from algan.rendering.raytracing.shading_taichi import _USER_PIPELINE_BASE

# Maximum number of ray bounces (mirror reflections / diffuse scatters).
MAX_BOUNCES = 4
# Rays averaged per pixel. 1 renders with the exact deterministic kernel;
# > 1 switches to the Monte Carlo pathtracer (stochastic transparency,
# glossy reflections, optional diffuse indirect lighting).
SAMPLES_PER_PIXEL = 1

TONEMAPPING = True
TONEMAP_EXPOSURE = 1.0
TONEMAP_METHOD = "neutral"
POST_PROCESS_TONEMAP = False

# Strength of diffuse indirect bounces in the Monte Carlo renderer: 0 keeps
# surfaces purely (vertex-shader) lit, > 0 scatters paths on diffuse hits
# with throughput ``albedo * strength`` for color bleeding.
INDIRECT_BOUNCE_STRENGTH = 0.0

# Radiance scale of explicit point lights in physical mode. The default of
# pi makes a white light produce roughly albedo-level Lambertian brightness.
LIGHT_INTENSITY = 3.141592653589793
# Constant ambient term added per diffuse interaction in physical mode.
AMBIENT_LIGHT = 0.0
# When True, the deterministic trace kernel is told which geometry types are
# actually present and skips the per-ray traversal of any type whose tree is
# just the empty placeholder (a launch-uniform branch, no divergence). Set
# False to force all three traversals -- used by the A/B benchmark to measure
# the gain in isolation.
GATE_EMPTY_TRAVERSALS = True

# Wavefront traversal rollouts. Changes to sibling revalidation and child
# ordering are enabled by default after parity validation; the opaque paths
# remain opt-in until their scene classification and shading gates are proven.
WF_REVALIDATE_PENDING = os.environ.get("ALGAN_WF_REVALIDATE_PENDING", "0") == "1"
WF_NEAR_FIRST = os.environ.get("ALGAN_WF_NEAR_FIRST", "0") == "1"
WF_OPAQUE_CLOSEST = os.environ.get("ALGAN_WF_OPAQUE_CLOSEST", "0") == "1"
WF_OPAQUE_PREPASS = os.environ.get("ALGAN_WF_OPAQUE_PREPASS", "0") == "1"

INPLACE_AA = os.environ.get("ALGAN_INPLACE_AA", "0") == "1"
# Rays per wavefront screen tile. The wavefront holds per-ray state for every
# ray it processes at once (~(18 + 6*KBUF) floats/ray); processing the chunk in
# tiles of this many rays bounds that state so it fits at any resolution / chunk
# length (a single HD frame is ~2M rays). ~2M rays * ~168 B ~= 350 MB of state.
WAVEFRONT_TILE_RAYS = int(os.environ.get("ALGAN_WAVEFRONT_TILE", str(1 << 21)))
# Adaptive tile sizing: size wavefront tiles from the render pool's *actual*
# free bytes instead of the fixed WAVEFRONT_TILE_RAYS. The static ~2M-ray
# default keeps tiles small enough for any GPU, but every tile pays a fixed
# host-side cost per kernel launch (the traverse/shade kernels marshal 60+
# ndarray args per launch -- ~7-9 ms each on this project's hardware), so a
# UHD frame split into 16 tiles wastes seconds per render on launch overhead
# alone. Auto mode computes rays-per-tile = free_pool_bytes * SAFETY /
# bytes_per_ray at render time: bigger tiles on cards with headroom (fewer
# launches, same per-ray state), smaller tiles instead of an OOM-retry when
# the pool is nearly full. Byte-identical by construction -- tiling is
# per-pixel independent (validated by benchmarks/_wf_tile_auto_ab.py).
# Setting ALGAN_WAVEFRONT_TILE explicitly disables auto and honors the fixed
# value (for A/B runs and reproduction).
# Default OFF: measured 1.008x (noise-level) on HD frames -- in *unprofiled*
# runs the per-launch host cost overlaps device execution, so fewer launches
# buy nothing (the profiler's per-kernel syncs made launches look expensive).
# Opt in for memory-constrained renders, where shrinking tiles beats the
# window-halving OOM retry.
WAVEFRONT_TILE_AUTO = (
    os.environ.get("ALGAN_WAVEFRONT_TILE_AUTO", "1") == "1"
    and "ALGAN_WAVEFRONT_TILE" not in os.environ)
# Fraction of the pool's free bytes the per-tile ray state may claim.  Every
# built-in per-slot/fixed allocation and ManualMemory's initial alignment are
# now accounted exactly, so the default can use the whole allowance.  Keep the
# override as an opt-in diagnostic/performance headroom control.
WAVEFRONT_TILE_SAFETY = float(
    os.environ.get("ALGAN_WAVEFRONT_TILE_SAFETY", "1.0"))
# Preferred lower bound and hard upper bound for auto tile size (rays). The
# runtime honors the floor when it fits, but deliberately goes below it when
# exact arena headroom requires a smaller tile; the cap bounds active-index
# buffers and launch size on very large pools.
WAVEFRONT_TILE_MIN = int(
    os.environ.get("ALGAN_WAVEFRONT_TILE_MIN", str(1 << 18)))
WAVEFRONT_TILE_MAX = int(
    os.environ.get("ALGAN_WAVEFRONT_TILE_MAX", str(1 << 25)))


def set_wavefront_tile_auto(enabled):
    """Toggle adaptive (pool-sized) wavefront tile sizing (see
    ``WAVEFRONT_TILE_AUTO``). Off falls back to the fixed
    ``WAVEFRONT_TILE_RAYS``."""
    global WAVEFRONT_TILE_AUTO
    WAVEFRONT_TILE_AUTO = bool(enabled)
# On the common non-splitting wavefront path (no refraction/custom scatter), a
# ray that leaves the active set can never become active again.  Compact the
# next iteration from the previous active indexes rather than scanning the
# entire tile-sized ray pool after every traverse/shade pass.  Deep transparent
# scenes benefit most as the active population shrinks over successive passes.
# Splitting paths retain the full-pool scan because a shade pass may activate a
# spare slot that was not in the previous active set.  Runtime-mutable for
# in-process A/B checks; the env var selects the startup default.
WF_COMPACT_ACTIVE_ONLY = os.environ.get(
    "ALGAN_WF_COMPACT_ACTIVE_ONLY", "1") == "1"
# Initial ratio of total shared ray-pool slots to primary rays for a tile that
# may split. This is only a launch-efficiency heuristic, not a per-pixel or
# per-path split limit: all pixels append continuations into one shared pool.
# If that pool overflows, the tile is discarded and retried with half as many
# primary rays while keeping the same pool capacity, doubling the continuation
# headroom until it succeeds. A ratio of two avoids an automatic failed first
# attempt for ordinary glass while using far less memory per primary than the
# old fixed eight-slots-per-pixel layout. The legacy environment variable is
# accepted as an alias so existing tuning scripts remain valid; its value now
# controls only the initial ratio, never a hard maximum number of splits.
REFRACT_INITIAL_POOL_RATIO = max(2, int(os.environ.get(
    "ALGAN_WAVEFRONT_INITIAL_POOL_RATIO",
    os.environ.get("ALGAN_WAVEFRONT_SPLIT", "2"),
)))
# Backwards-compatible name for code that imported the old setting. It now
# denotes the initial shared-pool ratio; it is no longer a per-pixel slot cap.
REFRACT_SPLIT_SLOTS = REFRACT_INITIAL_POOL_RATIO
# When True, the *deterministic* raytracer (SAMPLES_PER_PIXEL == 1, non-physical)
# shades the core lit materials per fragment inside the trace kernel instead of
# baking per-vertex colours (Gouraud). Ignored by the Monte Carlo pathtracer.
FRAGMENT_SHADING = True
# Promote a mob whose colour AND material params (reflectivity/roughness/index
# of refraction) are constant across the whole surface to a 1x1 texture at merge
# time, dropping its per-vertex ``tri_colors``/``tri_extra`` rows, instead of
# broadcasting the constant to every vertex. The shared texel buffer keeps one
# copy per mob (and, when the colour is also constant across frames, one copy
# total) rather than [T, N, 3, 5] / [T, N, 15]. Only applied on the
# deterministic fragment-shading wavefront path -- the only path where a
# "constant colour" mob genuinely has constant per-fragment colour (vertex
# lighting bakes per-vertex variation, so a promoted mob would be wrong there).
# The trace kernels guard every per-vertex read with ``prim < array.shape[1]``,
# so the shrunk arrays are never indexed for a promoted prim and every other
# render path stays byte-identical. Sampling a 1x1 map reduces exactly to the
# stored constant, so a promoted render matches the per-vertex one to <=1 ULP
# (the barycentric sum ``w0+w1+w2`` is not exactly 1.0 in f32). Default on;
# ALGAN_PROMOTE_CONSTANTS=0 disables it (for A/B and validation).
PROMOTE_CONSTANTS = os.environ.get("ALGAN_PROMOTE_CONSTANTS", "1") == "1"

# Skip the up-front per-fragment shading-normal computation for UNLIT hits on
# the fragment-shading wavefront. An UNLIT material passes its colour through
# unchanged (``_run_frag_pipeline`` ignores the shading normal for it), so
# computing the interpolated/normal-mapped normal for such a hit is wasted work.
# Reflective/refractive continuation recomputes its own normal on demand, so
# this is byte-identical. Compile-time template of the shade kernel (no runtime
# arg -- the shade kernel is already at Taichi's 64-arg ceiling); this is the
# speed-relevant core of the "Family A" material-field trim (skipping the
# normal work), decoupled from the memory-side array trimming.
# ALGAN_WF_SKIP_UNLIT_NORMAL=0 disables it (for A/B and validation).
WF_SKIP_UNLIT_NORMAL = os.environ.get("ALGAN_WF_SKIP_UNLIT_NORMAL", "1") == "1"

# Fused primary-ray generation on the general wavefront. The classic pipeline
# opens every tile with a standalone ``wavefront_generate_rays`` pass that
# writes ~104 B of initial per-ray state (ro/rd/acc/sca/int/pix/accum) only
# for the first traverse/shade to read it straight back -- a pure memory
# round-trip costing ~10 ms per 2M-ray tile on a GTX 1050 (~4.5 s of the UHD
# bezier benchmark). When on, a split-free (no refraction / custom scatter),
# near-clip-free, AA-in-super-sample render skips that pass: the tile's first
# traverse generates the rays in-kernel (compile-time ``gen_first`` template,
# persisting only ro/rd) and the first shade treats the initial state as
# constants (``first_iter`` template), with survivors writing their state
# back exactly as before. Byte-identical (same _generate_ray math, same
# order; validated by benchmarks/_wf_gen_fused_ab.py); iterations >= 1 and
# every non-qualifying render use the unchanged classic kernels.
#
# Values: True / False force fused / classic generation; "auto" (default)
# picks adaptively per render job. Fusing costs a SECOND compile-time
# instantiation of the traverse+shade monoliths (the ``first`` variants) on
# top of the classic ones its later iterations share with the unfused path --
# roughly 10 s of warm-start kernel prep per program run (AST re-transform +
# offline-cache load; see utils/taichi_warmstart.py) -- to win ~8.2% of
# steady-state wavefront time. Auto therefore starts every job unfused and
# turns fusing on mid-render only once the measured per-frame render rate
# forecasts a remaining-time saving above WF_GEN_FUSED_MIN_WIN (the decision
# is sticky for the process: once the fused variants are compiled, later jobs
# start fused for free). Byte-identical either way, so the switch cannot
# change output. ALGAN_WF_GEN_FUSED=0/1 forces (for A/B and validation).
def _parse_gen_fused_mode(v):
    v = str(v).strip().lower()
    if v in ("1", "true", "on"):
        return True
    if v in ("0", "false", "off"):
        return False
    return "auto"


WF_GEN_FUSED = _parse_gen_fused_mode(
    os.environ.get("ALGAN_WF_GEN_FUSED", "auto"))

# Fraction of wavefront render time the fused generation saves (the measured
# steady-state win; used only by the "auto" forecast).
WF_GEN_FUSED_GAIN = float(os.environ.get("ALGAN_WF_GEN_FUSED_GAIN", "0.082"))
# Minimum forecasted saving (seconds of remaining render time * GAIN) before
# "auto" pays the fused variants' compile cost. The default covers the
# worst case observed on this project's hardware -- a cold offline cache,
# where the two extra instantiations cost ~25 s -- so a marginal render never
# loses time to the switch.
WF_GEN_FUSED_MIN_WIN = float(
    os.environ.get("ALGAN_WF_GEN_FUSED_MIN_WIN", "30.0"))

# Adaptive state ("auto" mode only). The decision is process-sticky; the
# batch counter restarts per render job so the forecast never uses the
# compile-inflated first batch of a job.
_WF_GEN_FUSED_ON = False
_WF_GEN_FUSED_BATCHES = 0


def set_gen_fused(mode):
    """Set fused primary-ray generation on the deterministic wavefront:
    ``True``/``False`` force it on/off; ``"auto"`` (default) starts unfused
    for fast startup and enables it mid-render when the forecasted remaining
    render time justifies compiling the fused kernel variants. All modes are
    byte-identical (see ``WF_GEN_FUSED``)."""
    global WF_GEN_FUSED
    WF_GEN_FUSED = _parse_gen_fused_mode(mode)


def wf_gen_fused_active():
    """Live effective value of the fused-generation toggle (resolves
    ``"auto"`` to the adaptive decision)."""
    if WF_GEN_FUSED == "auto":
        return _WF_GEN_FUSED_ON
    return bool(WF_GEN_FUSED)


def _begin_render_job():
    """Render-loop hook: a new render job starts (resets the per-job batch
    count; the fused decision itself stays sticky for the process)."""
    global _WF_GEN_FUSED_BATCHES
    _WF_GEN_FUSED_BATCHES = 0


def _note_batch_rendered(frames, seconds, frames_remaining):
    """Render-loop hook: a batch of ``frames`` frames rendered in ``seconds``
    wall seconds with ``frames_remaining`` still to go. Returns True when this
    call switches fused generation on (so the caller can log it). The first
    rendered batch of a job is never used for the forecast -- it typically
    contains the one-off kernel materialization/compile time."""
    global _WF_GEN_FUSED_ON, _WF_GEN_FUSED_BATCHES
    _WF_GEN_FUSED_BATCHES += 1
    if (WF_GEN_FUSED != "auto" or _WF_GEN_FUSED_ON
            or _WF_GEN_FUSED_BATCHES < 2
            or frames <= 0 or seconds <= 0.0 or frames_remaining <= 0):
        return False
    projected_win = frames_remaining * (seconds / frames) * WF_GEN_FUSED_GAIN
    if projected_win <= WF_GEN_FUSED_MIN_WIN:
        return False
    _WF_GEN_FUSED_ON = True
    return True

# "Family A+B" full material-field memory trim for the fragment-shading
# wavefront. When on, triangles are reordered into material-class bands so
# ``tri_norm`` (needs-normal prims) and ``tri_mat`` (lit prims) are stored as
# compacted PREFIXES, and the promotion-compacted ``tri_colors``/``tri_extra``
# are addressed through a per-prim remap ``col_row`` (Family B), with
# ``tex_meta``/``uvs`` widened to full band-order arrays. Saves per-primitive
# memory at the cost of a per-hit indirection gather (``col_row``) -- an
# experimental measurement of that trade-off (expected slightly SLOWER on the
# occupancy-bound kernel; see benchmarks/_wf_mem_trim_ab.py). Byte-identical to
# the baseline. Opt-in; only engaged for a no-shadow, non-refractive,
# scatter-free triangle path (the common case). Default OFF.
WF_MEM_TRIM = os.environ.get("ALGAN_WF_MEM_TRIM", "0") == "1"

# Hybrid raster front-end for primary visibility (raytracer-v2 phase 2).
# When on (and the batch qualifies -- see ``use_raster`` in
# ``tracer.raytrace_render_wavefront``), the deterministic wavefront's first
# iteration is replaced by a raster pipeline: per-frame screen-space binning
# of triangle candidates, an opaque z-prepass (packed atomicMin), an exact
# per-pixel fragment list sorted by raw depth, and a resolve+shade kernel
# that composites the whole straight-line transparency stack in one pass
# (unbounded K) and spawns reflection/refraction continuations into the
# existing wavefront pool. Bounce iterations, compositing and memory tiling
# are unchanged. NOT byte-identical to the classic path: hit ordering is raw
# sorted depth (no DEPTH_TIE_EPSILON binning) and the opaque prepass culls
# strictly. Currently flat-triangle-only scenes, no shadows / custom scatter /
# mem-trim / in-place AA / near-clip. Default OFF while validating.
HYBRID_RASTER = os.environ.get("ALGAN_HYBRID_RASTER", "0") == "1"


def set_hybrid_raster(enabled):
    """Toggle the hybrid raster primary-visibility front-end (see
    ``HYBRID_RASTER``)."""
    global HYBRID_RASTER
    HYBRID_RASTER = bool(enabled)


# Screen-space rasterization inside the hybrid raster front-end. When on
# (default), each (prim, chunk) pair projects its triangle once and tests
# candidate pixels with edge functions + perspective-correct barycentric
# interpolation, instead of generating a world-space ray and running
# Moller-Trumbore per candidate pixel. Numerically equivalent to the ray-cast
# path (verified worst |dt| ~5e-5, |d_bary| ~6e-5 -- benchmarks/
# _rt2_ss_math_check.py); a triangle with any vertex at/behind the camera plane
# falls back to per-pixel ray casting.
#
# The win scales with bbox OVERDRAW (candidate pixels that miss the triangle):
# the edge-function inside-test rejects a miss far more cheaply than a full
# ray-gen + Moller-Trumbore. Kernel-isolated A/B: a dense thin-triangle mesh
# (flat neural_net, ~10x overdraw) runs 1.36x faster with SS; low-overdraw
# large triangles (spheres that fill their bbox) pay ~6% for the per-triangle
# setup with no misses to save on. Default ON because the high-overdraw case is
# both the expensive one and the realistic one; ALGAN_RASTER_SS=0 forces the
# ray-cast path for A/B.
RASTER_SS = os.environ.get("ALGAN_RASTER_SS", "1") == "1"


def set_raster_screen_space(enabled):
    """Toggle screen-space rasterization in the hybrid raster front-end (see
    ``RASTER_SS``)."""
    global RASTER_SS
    RASTER_SS = bool(enabled)


# UNSUPPORTED legacy "textured surface" wavefront (Surface / flat-triangle
# scenes only). This variant is no longer maintained and no longer works; the
# monolithic general wavefront is the only supported deterministic tracer.
# When on, the deterministic wavefront shaded from three per-triangle texture
# lookups instead of per-vertex arrays: a colour texture (RGBA+glow), a
# material texture (the shading parameter block) and a surface texture
# (reflectivity/roughness/index-of-refraction used for scatter); see
# scene_builder._build_textured_scene + wavefront_textured_kernels_taichi. It
# was a proof-of-concept built to benchmark the texture-lookup shading
# architecture, kept for reference only. Default OFF; do not enable.
WF_TEXTURED = os.environ.get("ALGAN_WF_TEXTURED", "0") == "1"


def set_textured_wavefront(enabled):
    """Enable the UNSUPPORTED legacy texture-lookup wavefront shader (kept for
    reference only, no longer works; see ``WF_TEXTURED``)."""
    global WF_TEXTURED
    WF_TEXTURED = bool(enabled)


# --- Scene merge + STBVH build device --------------------------------------
# The per-batch scene prep -- merging every primitive's packed ``_rt_*``
# geometry into one contiguous array per geometry type and building one STBVH
# per type -- is pure vectorised torch. ``project_to_screen`` (the heavy
# per-mob vertex work) always runs on the CPU animation device, hidden on the
# prefetch worker thread; this toggle controls only where the *merge + STBVH
# build* run afterwards.
#
# When on (default), they run on the render device: measured ~10-17x faster
# than the CPU build the previous commit introduced (the STBVH build alone was
# ~6.5s of the UHD bezier benchmark on the CPU). To keep the exact arena
# accounting that CPU build was introduced for, the GPU merge runs on the
# *render thread* (never the prefetch worker, so its transient out-of-place
# peak can be measured/bounded without a concurrent render polluting the
# stats), the finished scene is still uploaded into the render arena through
# ``copy_merged_scene_to_arena`` (so the arena footprint is unchanged and
# subtracted exactly), and the transient build scratch -- which lives in the
# render pool's non-arena headroom -- is (a) proactively bounded against that
# headroom before the merge is attempted and (b) caught by the existing
# OOM -> window-shrink retry if the estimate was low. Off falls back to the
# byte-exact CPU build. ALGAN_MERGE_ON_GPU=0 disables.
MERGE_ON_GPU = os.environ.get("ALGAN_MERGE_ON_GPU", "1") == "1"

# Multiplier turning a batch's packed ``_rt_*`` input bytes into a conservative
# estimate of the GPU merge's transient peak (the out-of-place cat / argsort /
# unique / dyadic-time-pyramid scratch plus the merged output). Measured peaks
# run ~3-6x the packed inputs; the default leaves margin so the proactive
# headroom check rarely lets a batch through that the OOM retry then has to
# shrink. Read live.
MERGE_GPU_PEAK_FACTOR = float(
    os.environ.get("ALGAN_MERGE_GPU_PEAK_FACTOR", "6.0"))

# Opt-in exact measurement of the GPU merge's transient peak, for calibrating
# ``MERGE_GPU_PEAK_FACTOR``. It calls ``torch.cuda.reset_peak_memory_stats``
# around the build, which clobbers the process-wide peak counter that
# ``profiling_utils`` reads for its whole-render peak, so it is OFF by default
# -- the always-on ``MERGE_GPU_PEAK_FACTOR`` estimate is what actually bounds
# the build against the pool headroom. Enable during a calibration run (not a
# profiling run) to log the measured peak alongside the estimate.
MERGE_TRACK_PEAK = os.environ.get("ALGAN_MERGE_TRACK_PEAK", "0") == "1"


def set_merge_on_gpu(enabled):
    """Toggle GPU-side scene merge + STBVH build (see ``MERGE_ON_GPU``)."""
    global MERGE_ON_GPU
    MERGE_ON_GPU = bool(enabled)


def merge_on_gpu_active():
    """True when the scene merge + STBVH build should run on the render device.

    Requires ``MERGE_ON_GPU`` and a CUDA render device -- the offload only pays
    off on a real accelerator, and the transient-peak accounting uses the
    ``torch.cuda`` memory-stats API. A CPU (or MPS) render device keeps the
    merge on the CPU, byte-identically to the pre-toggle path.
    """
    if not MERGE_ON_GPU:
        return False
    from algan.settings.defaults import COMPUTING_DEFAULTS

    return COMPUTING_DEFAULTS.render_device.type == "cuda"


# --- project_to_screen device ----------------------------------------------
# ``project_to_screen`` (per-primitive vertex shading + screen projection +
# geometry packing into the ``_rt_*`` arrays) is the next-largest vectorised
# prep phase after the merge -- measured ~2.6s/batch on the bezier text
# benchmark, larger than the merge. Like the merge it is pure vectorised torch,
# so it runs far faster on the render device. When on (default) it runs there,
# on the *render thread* (deferred off the prefetch worker, same as the merge,
# so its transient peak is measured/bounded without a concurrent render), which
# also leaves the packed ``_rt_*`` already on the device for the GPU merge to
# consume with no upload. The heavy Python-bound timeline materialization
# (``set_state_to_times``) is what still rides the hidden worker. Off keeps
# projection on the CPU source device. ALGAN_PROJECT_ON_GPU=0 disables.
PROJECT_ON_GPU = os.environ.get("ALGAN_PROJECT_ON_GPU", "1") == "1"

# Conservative multiplier from a batch's pre-projection source-geometry bytes
# to the projection's transient device peak (source + shading scratch + packed
# ``_rt_*`` output; the polyline sampling can expand bezier geometry well past
# its control points, hence a larger default than the merge factor). Bounds the
# projection against the pool headroom before it is attempted; the OOM retry is
# the exact fallback. Read live.
PROJECT_GPU_PEAK_FACTOR = float(
    os.environ.get("ALGAN_PROJECT_GPU_PEAK_FACTOR", "8.0"))


def set_project_on_gpu(enabled):
    """Toggle GPU-side ``project_to_screen`` (see ``PROJECT_ON_GPU``)."""
    global PROJECT_ON_GPU
    PROJECT_ON_GPU = bool(enabled)


def project_on_gpu_active():
    """True when ``project_to_screen`` should run on the render device.

    Requires ``PROJECT_ON_GPU`` and a CUDA render device (see
    ``merge_on_gpu_active`` for why CUDA specifically).
    """
    if not PROJECT_ON_GPU:
        return False
    from algan.settings.defaults import COMPUTING_DEFAULTS

    return COMPUTING_DEFAULTS.render_device.type == "cuda"


# Feature bitmask for the UNSUPPORTED legacy textured wavefront (see
# WF_TEXTURED): each bit compiled one of the monolith's features back into the
# (otherwise lean) textured shade kernel, so the marginal occupancy /
# performance cost of each could be measured one at a time (see
# benchmarks/_wf_textured_features_ab.py). The features are added in the order
# beziers -> custom scatter -> shadows -> normal maps.
WF_TEX_BEZ = 1        # bezier-circuit traversal + shading
WF_TEX_SCATTER = 2    # per-material custom scatter dispatch (ray bouncing)
WF_TEX_SHADOWS = 4    # binary hard shadow rays (triangle occluders)
WF_TEX_NORMALMAP = 8  # tangent-space normal-map perturbation of the shading normal
WF_TEXTURED_FEATURES = int(os.environ.get("ALGAN_WF_TEXTURED_FEATURES", "0"))


def set_textured_features(mask):
    """Set which monolith features are compiled into the UNSUPPORTED legacy
    textured wavefront shade kernel (a bitmask of WF_TEX_BEZ / _SCATTER /
    _SHADOWS / _NORMALMAP; see ``WF_TEXTURED``)."""
    global WF_TEXTURED_FEATURES
    WF_TEXTURED_FEATURES = int(mask)


# UNSUPPORTED legacy Cycles-style sorted material dispatch for the
# deterministic wavefront's *fragment-shading* path. The sorted pipeline is no
# longer maintained and no longer works; the monolithic shade kernel is the
# only supported deterministic shade path. When active, the monolith was
# replaced by a peel (surface-eval) kernel that suspends each ray at its next
# material event, a host-side sort of the pending events by (geometry type,
# material pipeline id), and one small geometry-free shade kernel *per
# material bucket* with the material's pipeline + scatter funcs injected at
# compile time -- so a warp never mixes materials and no kernel carries
# another material's code (see wavefront_sorted_kernels_taichi).
#
# Values: "auto" (default) and False/"0" both use the monolithic shade kernel,
# which supports *everything* the sorted path did -- custom ray-bouncing
# (scatter) and normal-mapped lighting -- while staying faster on the built-in
# materials (it drains up to KBUF hits per launch, whereas sorting pays
# per-event kernel round trips + host syncs; see benchmarks/_wf_sorted_ab.py
# and _wf_monolith_scatter_ab.py). True/"1" still routes to the sorted
# pipeline, but that route is unsupported (kept for reference only). "auto" is
# kept as a distinct label so the engine can revisit this heuristic later
# without an API change.
def _parse_sort_mode(v):
    v = str(v).strip().lower()
    if v in ("1", "true", "on"):
        return True
    if v in ("0", "false", "off"):
        return False
    return "auto"


WAVEFRONT_SORT_MATERIALS = _parse_sort_mode(
    os.environ.get("ALGAN_WF_SORT_MATERIALS", "auto"))


def set_material_sorting(enabled):
    """Set Cycles-style sorted per-material shading of the deterministic
    wavefront's fragment-shading path: ``True`` forces the UNSUPPORTED legacy
    sorted pipeline (kept for reference only, no longer works);
    ``False`` / ``"auto"`` (default) use the monolithic kernel, which handles
    custom scatter + normal maps itself and is faster on built-in materials.
    See ``WAVEFRONT_SORT_MATERIALS``."""
    global WAVEFRONT_SORT_MATERIALS
    WAVEFRONT_SORT_MATERIALS = _parse_sort_mode(enabled)


def set_fragment_shading(enabled):
    """Toggle per-fragment shading of the *deterministic* ray tracer.

    When enabled, triangle/PN hits whose material is one of the core lit
    shaders (the legacy diffuse default, ``MeshBasicMaterial``,
    ``MeshLambertMaterial``, ``MeshPhongMaterial``, ``MeshStandardMaterial``,
    ``MeshPhysicalMaterial``) are shaded per fragment in-kernel from the raw
    albedo, a per-primitive material block and the scene's point lights --
    crisper specular highlights and smooth shading on coarse meshes. Other
    materials keep vertex shading.
    Only the deterministic renderer (``set_samples_per_pixel(1)``, non-physical)
    is affected. Set before rendering.
    """
    global FRAGMENT_SHADING
    FRAGMENT_SHADING = bool(enabled)


# When True, the deterministic ray tracer casts binary hard shadows: each
# shaded triangle/PN fragment fires one shadow ray per point light and an
# opaque occluder (alpha >= SHADOW_ALPHA_THRESHOLD) fully blocks that light's
# direct contribution. Implies per-fragment shading (shadows are evaluated in
# the lighting model) and forces the general kernel. Lights with a non-zero
# ``shadow_radius`` / ``shadow_angle`` (and area lights) get *soft* shadows: a
# fixed deterministic fan of SOFT_SHADOW_SAMPLES rays is traced across the
# emitter instead of a single ray. Off by default.
SHADOWS = False

# Number of shadow rays in the deterministic soft-shadow fan (per light with a
# non-zero shadow radius, per shaded fragment). More = smoother penumbras,
# linearly more shadow cost. Baked into the shade kernel at compile time; set
# the env var ALGAN_SOFT_SHADOW_SAMPLES before the first render to change it.
SOFT_SHADOW_SAMPLES = max(2, int(os.environ.get(
    "ALGAN_SOFT_SHADOW_SAMPLES", "8")))


def set_ray_traced_shadows(enabled):
    """Toggle binary hard shadows in the *deterministic* ray tracer.

    When enabled, every shaded triangle/PN fragment traces one shadow ray per
    scene point light; a light is occluded (its direct diffuse/specular term
    dropped, ambient/emissive kept) when an opaque surface lies between the
    fragment and the light. Shadows are evaluated inside the wavefront shade
    kernel's per-fragment lighting model, so this implies
    :func:`set_fragment_shading` for the render. Shadows are binary and ignore
    partial transparency; lights with a non-zero ``shadow_radius`` /
    ``shadow_angle`` (and area lights) get *soft* shadows via a deterministic
    fan of ``SOFT_SHADOW_SAMPLES`` rays, while glass shadows need the physical
    path tracer (``set_samples_per_pixel(n)`` with ``n > 1``). Only the
    deterministic renderer (``set_samples_per_pixel(1)``, non-physical) is
    affected. Set before rendering.
    """
    global SHADOWS
    SHADOWS = bool(enabled)

def set_light_intensity(intensity):
    """Radiance scale applied to explicit point lights in physical mode."""
    global LIGHT_INTENSITY
    LIGHT_INTENSITY = float(intensity)


def set_ambient_light(intensity):
    """Constant ambient lighting term used in physical mode."""
    global AMBIENT_LIGHT
    AMBIENT_LIGHT = float(intensity)


def set_samples_per_pixel(samples):
    """Set how many rays are averaged per pixel. 1 (the default) uses the
    exact deterministic renderer; larger values enable Monte Carlo path
    tracing with that many samples.
    """
    global SAMPLES_PER_PIXEL
    SAMPLES_PER_PIXEL = max(1, int(samples))


def set_indirect_bounce_strength(strength):
    """Set the diffuse indirect lighting strength of the Monte Carlo
    renderer (0 disables diffuse bounces).
    """
    global INDIRECT_BOUNCE_STRENGTH
    INDIRECT_BOUNCE_STRENGTH = float(strength)


def set_tonemapping(enabled):
    """Enable or disable ACES Filmic Tonemapping in the ray-tracing rendering kernels."""
    global TONEMAPPING
    TONEMAPPING = bool(enabled)


def set_tonemap_exposure(exposure):
    """Set the exposure multiplier for the ACES Filmic Tonemapper."""
    global TONEMAP_EXPOSURE
    TONEMAP_EXPOSURE = float(exposure)


def set_tonemap_method(method):
    """Set the tonemapping method ("neutral" or "agx")."""
    global TONEMAP_METHOD
    if method not in ("neutral", "agx"):
        raise ValueError("tonemap_method must be 'neutral' or 'agx'")
    TONEMAP_METHOD = str(method)


def set_post_process_tonemap(enabled):
    """Enable or disable post-process tonemapping instead of in-kernel tonemapping."""
    global POST_PROCESS_TONEMAP
    POST_PROCESS_TONEMAP = bool(enabled)


def is_post_process_tonemap_enabled():
    """Return whether post-process tonemapping is enabled."""
    return POST_PROCESS_TONEMAP


def _get_tonemap_t_val():
    if POST_PROCESS_TONEMAP:
        return 3
    if not TONEMAPPING:
        return 0
    return 2 if TONEMAP_METHOD == "agx" else 1

# --- Core lit material registry (shader function -> in-kernel material id) ----
# Ids must match shading_taichi: 0 default diffuse, 1 basic/unlit/passthrough,
# 2 lambert, 3 phong, 4 standard, 5 physical.
def _build_core_shader_ids():
    from algan.rendering.shaders.material_shaders import (
        basic_material_shader,
        lambert_shader,
        phong_shader,
        physical_shader,
        standard_shader,
    )
    from algan.rendering.shaders.pbr_shaders import default_shader, null_shader

    return {
        default_shader: 0,
        null_shader: 1,
        basic_material_shader: 1,
        lambert_shader: 2,
        phong_shader: 3,
        standard_shader: 4,
        physical_shader: 5,
    }


_CORE_SHADER_IDS = None
# Per-material parameter defaults (canonical 26-slot block; see shading_taichi).
# Slots 12+ are the MeshPhysicalMaterial extension, defaults matching the
# physical_shader signature (ior 1.5, specular_intensity 1, specular_color
# white, clearcoat/sheen off, sheen_roughness 1, transmission/iridescence 0).
_MAT_DEFAULTS = [0.0, 0.0, 0.0, 1.0, 0.0666, 0.0666, 0.0666, 30.0, 1.0, 0.0,
                 0.0, 1.0,
                 1.5, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                 0.0, 0.0]
# Material-property name -> (start slot, width) in the canonical block.
_MAT_SLOTS = {
    "emissive": (0, 3),
    "emissive_intensity": (3, 1),
    "specular": (4, 3),
    "shininess": (7, 1),
    "roughness": (8, 1),
    "metalness": (9, 1),
    "flat_shading": (10, 1),
    "env_map_intensity": (11, 1),
    "ior": (12, 1),
    "specular_intensity": (13, 1),
    "specular_color": (14, 3),
    "clearcoat": (17, 1),
    "clearcoat_roughness": (18, 1),
    "sheen": (19, 1),
    "sheen_roughness": (20, 1),
    "sheen_color": (21, 3),
    "transmission": (24, 1),
    "iridescence": (25, 1),
}


def _core_shader_ids():
    global _CORE_SHADER_IDS
    if _CORE_SHADER_IDS is None:
        _CORE_SHADER_IDS = _build_core_shader_ids()
    return _CORE_SHADER_IDS


def _shader_material_id(shader):
    """In-kernel material id for a shader function. Unknown / non-core shaders
    (and ``None``) map to 1 (unlit passthrough: the kernel returns the colour --
    raw or baked -- unchanged)."""
    if shader is None:
        return 1
    return _core_shader_ids().get(shader, 1)


def _shader_is_core(shader):
    """True if ``shader`` has an in-kernel port (so its hits can be fragment
    shaded rather than baked)."""
    return shader is not None and shader in _core_shader_ids()

def _constant_promotion_active():
    """True when constant-property -> 1x1-texture promotion applies to this
    render: it is enabled, and the batch will render through the deterministic
    fragment-shading general wavefront (the only path where a mob's colours are
    raw albedo, so a "constant colour" is genuinely constant per fragment, and
    the only kernel whose per-vertex reads are guarded for shrunk arrays).
    Every deterministic (samples <= 1) batch renders through that kernel."""
    return PROMOTE_CONSTANTS and FRAGMENT_SHADING and SAMPLES_PER_PIXEL <= 1

def _scene_has_user_pipeline(merged):
    """True if any merged primitive carries a custom fragment-pipeline id
    (``>= _USER_PIPELINE_BASE``), so the render must enable fragment shading."""
    cached = merged.get("has_user_pipeline")
    if cached is not None:
        return bool(cached)
    for key in ("tri_mat_id", "pn_mat_id"):
        arr = merged.get(key)
        # Compatibility for externally assembled scenes that predate the
        # cached flag.  Move tiny ids to the host before reducing so this
        # fallback cannot create a scalar/reduction workspace beside the
        # render arena.
        if (arr is not None and arr.numel()
                and int(arr.detach().cpu().max()) >= _USER_PIPELINE_BASE):
            return True
    return False
