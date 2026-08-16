from __future__ import annotations

import warnings

from algan.environment import env_flag, env_float, env_int, env_is_set, env_str
from algan.errors import UnsupportedFeatureError, UnsupportedFeatureWarning
from algan.rendering.raytracing.shading_taichi import _USER_PIPELINE_BASE
from algan.settings._startup import _HDR_BUFFER_F16, _RENDER_DEVICE

# Maximum number of ray bounces (mirror reflections / diffuse scatters).
MAX_BOUNCES = 8
# Rays averaged per pixel. 1 renders with the exact deterministic kernel;
# > 1 switches to the Monte Carlo pathtracer (stochastic transparency,
# glossy reflections, optional diffuse indirect lighting).
SAMPLES_PER_PIXEL = 1

# Policy for renderer/backend combinations that cannot honor authored scene
# features. "error" is the safe public default; "warn" and "ignore" are
# available for controlled migration and benchmarking.
UNSUPPORTED_FEATURE_POLICY = (
    env_str("ALGAN_UNSUPPORTED_FEATURE_POLICY", "error").strip().lower()
)
if UNSUPPORTED_FEATURE_POLICY not in {"error", "warn", "ignore"}:
    UNSUPPORTED_FEATURE_POLICY = "error"


def set_unsupported_feature_policy(policy):
    """Set unsupported-feature handling to ``error``, ``warn``, or ``ignore``."""
    normalized = str(policy).strip().lower()
    if normalized not in {"error", "warn", "ignore"}:
        raise UnsupportedFeatureError("policy must be 'error', 'warn', or 'ignore'")
    global UNSUPPORTED_FEATURE_POLICY
    UNSUPPORTED_FEATURE_POLICY = normalized


def report_unsupported_features(message):
    """Apply the configured policy to an unsupported render combination."""
    if UNSUPPORTED_FEATURE_POLICY == "ignore":
        return
    if UNSUPPORTED_FEATURE_POLICY == "warn":
        warnings.warn(message, UnsupportedFeatureWarning, stacklevel=3)
        return
    raise UnsupportedFeatureError(message)


TONEMAPPING = True
TONEMAP_EXPOSURE = 1.0
TONEMAP_METHOD = "neutral"
# Tonemap in post-processing (composite writes linear HDR float) rather than
# in the composite kernel. This is the physically-correct order: bloom/glow
# and the supersample downsample run in linear HDR and tonemapping is applied
# last (Unity/Unreal do the same), so HDR highlights keep their chroma
# instead of clipping to white. It also makes the composite a linear blend
# that is identity for empty pixels (enabling the covered-pixel compaction).
# Costs a float32 frame buffer (4x the uint8 one), so fewer frames per batch.
# Env override for A/B and re-baselining.
POST_PROCESS_TONEMAP = env_flag("ALGAN_POST_PROCESS_TONEMAP", True)

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
WF_REVALIDATE_PENDING = env_flag("ALGAN_WF_REVALIDATE_PENDING", False)
WF_NEAR_FIRST = env_flag("ALGAN_WF_NEAR_FIRST", False)
WF_OPAQUE_CLOSEST = env_flag("ALGAN_WF_OPAQUE_CLOSEST", False)
WF_OPAQUE_PREPASS = env_flag("ALGAN_WF_OPAQUE_PREPASS", False)

INPLACE_AA = env_flag("ALGAN_INPLACE_AA", False)
# Rays per wavefront screen tile. The wavefront holds per-ray state for every
# ray it processes at once (~(18 + 6*KBUF) floats/ray); processing the chunk in
# tiles of this many rays bounds that state so it fits at any resolution / chunk
# length (a single HD frame is ~2M rays). ~2M rays * ~168 B ~= 350 MB of state.
WAVEFRONT_TILE_RAYS = env_int("ALGAN_WAVEFRONT_TILE", 1 << 21)
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
    env_flag("ALGAN_WAVEFRONT_TILE_AUTO", True)
    and not env_is_set("ALGAN_WAVEFRONT_TILE")
)
# Fraction of the pool's free bytes the per-tile ray state may claim.  Every
# built-in per-slot/fixed allocation and ManualMemory's initial alignment are
# now accounted exactly, so the default can use the whole allowance.  Keep the
# override as an opt-in diagnostic/performance headroom control.
WAVEFRONT_TILE_SAFETY = env_float("ALGAN_WAVEFRONT_TILE_SAFETY", 1.0)
# Preferred lower bound and hard upper bound for auto tile size (rays). The
# runtime honors the floor when it fits, but deliberately goes below it when
# exact arena headroom requires a smaller tile; the cap bounds active-index
# buffers and launch size on very large pools.
WAVEFRONT_TILE_MIN = env_int("ALGAN_WAVEFRONT_TILE_MIN", 1 << 18)
WAVEFRONT_TILE_MAX = env_int("ALGAN_WAVEFRONT_TILE_MAX", 1 << 25)


def set_wavefront_tile_auto(enabled):
    """Toggle adaptive (pool-sized) wavefront tile sizing (see
    ``WAVEFRONT_TILE_AUTO``). Off falls back to the fixed
    ``WAVEFRONT_TILE_RAYS``.
    """
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
WF_COMPACT_ACTIVE_ONLY = env_flag("ALGAN_WF_COMPACT_ACTIVE_ONLY", True)
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
REFRACT_INITIAL_POOL_RATIO = max(
    2,
    env_int(
        "ALGAN_WAVEFRONT_INITIAL_POOL_RATIO",
        env_int("ALGAN_WAVEFRONT_SPLIT", 2),
    ),
)
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
PROMOTE_CONSTANTS = env_flag("ALGAN_PROMOTE_CONSTANTS", True)

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
WF_SKIP_UNLIT_NORMAL = env_flag("ALGAN_WF_SKIP_UNLIT_NORMAL", True)

# Compile-time material-pipeline gating. The per-hit material dispatch
# (``shading_taichi._run_frag_pipeline``) is inlined into the shade kernels
# with every built-in stage reachable -- including ``_stage_physical``'s
# clearcoat + sheen lobes -- so a scene of plain diffuse triangles still pays
# their register footprint in the occupancy-starved shade kernel. When on, the
# host hands each shade kernel a bitmask of the pipeline ids the batch's
# triangles / PN patches actually carry (built at merge time as
# ``{tri,pn}_material_ids``, no device reduction) and the absent stages are
# never compiled in; a batch that carries exactly one id drops the per-hit id
# fetch and compare as well. Byte-identical by construction: the mask is a
# superset of the ids the kernel can read, so only unreachable branches go.
# Costs one Taichi kernel variant per distinct (tri, pn) material combination.
#
# Measured (kernel-profiler device time, benchmarks/_frag_pid_gate_profile.py):
# 1.4x on ``raster_first_shade`` for a scene mixing several materials, and
# NOTHING for a single-material scene or for ``wavefront_shade`` -- only the
# raster resolve sits close enough to its occupancy cliff for the dropped
# stages to matter. Hence experimental and off by default rather than on:
# ALGAN_FRAG_PID_GATE=1 opts in.
FRAG_PID_GATE = env_flag("ALGAN_FRAG_PID_GATE", False)


def set_frag_pid_gate(enabled):
    """Toggle compile-time material-pipeline gating of the shade kernels (see
    ``FRAG_PID_GATE``). Takes effect at the next render batch.
    """
    global FRAG_PID_GATE
    FRAG_PID_GATE = bool(enabled)


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


WF_GEN_FUSED = _parse_gen_fused_mode(env_str("ALGAN_WF_GEN_FUSED", "auto"))

# Fraction of wavefront render time the fused generation saves (the measured
# steady-state win; used only by the "auto" forecast).
WF_GEN_FUSED_GAIN = env_float("ALGAN_WF_GEN_FUSED_GAIN", 0.082)
# Minimum forecasted saving (seconds of remaining render time * GAIN) before
# "auto" pays the fused variants' compile cost. The default covers the
# worst case observed on this project's hardware -- a cold offline cache,
# where the two extra instantiations cost ~25 s -- so a marginal render never
# loses time to the switch.
WF_GEN_FUSED_MIN_WIN = env_float("ALGAN_WF_GEN_FUSED_MIN_WIN", 30.0)

# Adaptive state ("auto" mode only). The decision is process-sticky; the
# batch counter restarts per render job so the forecast never uses the
# compile-inflated first batch of a job.
_WF_GEN_FUSED_ON = False
_WF_GEN_FUSED_BATCHES = 0

# Learned per-output-frame arena footprint of the sparse-coverage discovery
# (prepare_sparse_raster_coverage): the whole-window exact hit stream + compact
# result is a data-dependent, whole-window allocation the wavefront memory model
# cannot predict a priori (it scales with the covered-fragment count, unknown
# until the COUNT kernel runs). Rather than bound it by a crippling worst-case
# (full-frame coverage), the actual footprint of each rendered chunk is amortized
# per output frame here, and the render-chunk preflight reserves
# ``num_frames * this`` so subsequent chunks are sized to fit the discovery peak
# instead of over-committing and relying on the OOM window-halving. Reset per
# render job; tracked as the running maximum (coverage is temporally coherent, so
# the densest chunk seen is a safe reservation). Bytes per frame, float.
_SPARSE_DISCOVERY_BYTES_PER_FRAME = 0.0
# Safety multiplier on the learned footprint: absorbs the small per-pair count
# arrays, arena alignment, and modest coverage growth between adjacent chunks.
SPARSE_DISCOVERY_SAFETY = env_float("ALGAN_SPARSE_DISCOVERY_SAFETY", 1.25)


def note_sparse_discovery_footprint(arena_bytes, num_frames):
    """Record the arena footprint of one sparse-coverage discovery pass so the
    render-chunk preflight can reserve for the next one (see
    ``_SPARSE_DISCOVERY_BYTES_PER_FRAME``).
    """
    global _SPARSE_DISCOVERY_BYTES_PER_FRAME
    per_frame = float(arena_bytes) / max(1, int(num_frames))
    if per_frame > _SPARSE_DISCOVERY_BYTES_PER_FRAME:
        _SPARSE_DISCOVERY_BYTES_PER_FRAME = per_frame


def sparse_discovery_bytes_for_frames(num_frames):
    """Reserved arena bytes for the sparse-coverage discovery over ``num_frames``
    frames (0 until the first discovery of the job establishes a density).
    """
    return int(
        _SPARSE_DISCOVERY_BYTES_PER_FRAME
        * SPARSE_DISCOVERY_SAFETY
        * max(1, int(num_frames))
    )


def set_gen_fused(mode):
    """Set fused primary-ray generation on the deterministic wavefront:
    ``True``/``False`` force it on/off; ``"auto"`` (default) starts unfused
    for fast startup and enables it mid-render when the forecasted remaining
    render time justifies compiling the fused kernel variants. All modes are
    byte-identical (see ``WF_GEN_FUSED``).
    """
    global WF_GEN_FUSED
    WF_GEN_FUSED = _parse_gen_fused_mode(mode)


def wf_gen_fused_active():
    """Live effective value of the fused-generation toggle (resolves
    ``"auto"`` to the adaptive decision).
    """
    if WF_GEN_FUSED == "auto":
        return _WF_GEN_FUSED_ON
    return bool(WF_GEN_FUSED)


def _begin_render_job():
    """Render-loop hook: a new render job starts (resets the per-job batch
    count; the fused decision itself stays sticky for the process).
    """
    global _WF_GEN_FUSED_BATCHES, _SPARSE_DISCOVERY_BYTES_PER_FRAME
    _WF_GEN_FUSED_BATCHES = 0
    _SPARSE_DISCOVERY_BYTES_PER_FRAME = 0.0


def _note_batch_rendered(frames, seconds, frames_remaining):
    """Render-loop hook: a batch of ``frames`` frames rendered in ``seconds``
    wall seconds with ``frames_remaining`` still to go. Returns True when this
    call switches fused generation on (so the caller can log it). The first
    rendered batch of a job is never used for the forecast -- it typically
    contains the one-off kernel materialization/compile time.
    """
    global _WF_GEN_FUSED_ON, _WF_GEN_FUSED_BATCHES
    _WF_GEN_FUSED_BATCHES += 1
    if (
        WF_GEN_FUSED != "auto"
        or _WF_GEN_FUSED_ON
        or _WF_GEN_FUSED_BATCHES < 2
        or frames <= 0
        or seconds <= 0.0
        or frames_remaining <= 0
    ):
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
WF_MEM_TRIM = env_flag("ALGAN_WF_MEM_TRIM", False)

# Shared-topology binned-SAH refit BVH (raytracer-v2 design doc section 9;
# refit_bvh.py). When on, the per-batch scene merge builds ONE binned-SAH
# topology per geometry type over the batch-union primitive boxes and refits
# its node bounds per frame, instead of the classic spatio-temporal
# instance tree (stbvh.py). Measured headroom (benchmarks/_rt2_refit_sah.py):
# the classic STBVH costs 1.37-2.33x more expected node visits than a refit
# topology, and refit staleness across a batch is <= 1.04 vs a per-frame
# rebuild. The traversal kernels select the walk with a compile-time
# ``refit`` template, so both modes coexist in one process (in-process A/B).
# The set of exact intersections found is identical; discovery order and
# box-face boundary cases differ, so output parity is epsilon-level (the same
# class of deviation as ALGAN_BVH_BLOCK_F16 / tightness changes), bounded by
# benchmarks/_rt2_refit_parity.py. Ignored (classic trees are built) under
# the unsupported legacy textured / sorted-material orchestrators, which are
# not plumbed for the refit walk. Default ON (ALGAN_BVH_REFIT=0 restores the
# classic per-batch STBVH instance trees); note that while on, the triangle
# ``builder`` selection (e.g. "split") applies only where the classic trees
# are still built.
BVH_REFIT = env_flag("ALGAN_BVH_REFIT", True)


def set_refit_bvh(enabled):
    """Toggle the shared-topology binned-SAH refit BVH (see ``BVH_REFIT``).
    Takes effect at the next batch's scene merge.
    """
    global BVH_REFIT
    BVH_REFIT = bool(enabled)


# Skip building the per-batch STBVHs when the batch provably never traverses
# one: the hybrid raster front-end resolves and shades all primary rays
# without trees, so a deterministic, shadow-free batch with no reflective /
# refractive / custom-scatter materials leaves them untouched (the trees were
# ~2.5s of build per batch on the MD bezier profile). Placeholder trees keep
# the kernel ABI; scene_builder.build_deferred_bvhs builds the real trees on
# demand the moment shadows, classic routing, an actually spawned continuation
# ray, or the Monte Carlo path needs them -- so the rendered output is always
# exactly what the eager build produces. ALGAN_BVH_DEFER=0 disables (for A/B
# and validation).
BVH_DEFER = env_flag("ALGAN_BVH_DEFER", True)


def set_bvh_defer(enabled):
    """Toggle deferred (on-demand) STBVH builds for batches that provably do
    not traverse them (see ``BVH_DEFER``). Takes effect at the next batch's
    scene merge.
    """
    global BVH_DEFER
    BVH_DEFER = bool(enabled)


# Collapse temporally-constant merged tables (materials, normals, colours,
# per-vertex extras, UV tables and the 2-D edge tables) to a single frame at
# merge time; every consumer reads their time axis as ``f % shape[0]``. The
# rendered pixels are unchanged for a given batch window, but the collapse
# shrinks the merged scene the arena planner measures, so batch windows can
# differ from an uncollapsed run on memory-tight scenes (re-windowed output
# differs at the epsilon level, the same class as any window change).
# ALGAN_MERGE_DEDUP_TIME=0 restores the full time bands (byte-level A/B
# against pre-collapse baselines).
MERGE_DEDUP_TIME = env_flag("ALGAN_MERGE_DEDUP_TIME", True)


def set_merge_dedup_time(enabled):
    """Toggle the merge-time collapse of temporally-constant tables (see
    ``MERGE_DEDUP_TIME``). Takes effect at the next batch's scene merge.
    """
    global MERGE_DEDUP_TIME
    MERGE_DEDUP_TIME = bool(enabled)


# Opaque any-hit shadow early-out. The deterministic shadow query is an
# ordered closest-hit march that restarts a full three-tree traversal per
# peeled surface; but any interval-opaque blocker (main-tree leaf flag:
# classic ``leaf_tspan`` bit 31 / refit link bit 30) forces the final
# occlusion to exactly 1.0 no matter what lies in front of it. When on, the
# shadow query first runs a cheap unordered any-hit walk over just the
# opaque-flagged leaves and returns full occlusion on the first hit; batches
# that provably contain no translucent geometry skip the march entirely (a
# miss then proves the ray lit). Not strictly byte-identical in two corner
# cases the march itself gets wrong (an opaque edge hit seam-merged into a
# coincident translucent edge within DEPTH_TIE_EPSILON, and an opaque
# blocker past MAX_SURFACES_PER_RAY peels); the any-hit's answer is the
# physically correct one in both. Experimental while the pixel suites
# qualify it; ALGAN_SHADOW_ANYHIT=1 opts in.
#
# ALGAN_SHADOW_ANYHIT=gather selects the gather-march instead: the same
# ordered shadow peel rebuilt on the KBUF gather (_collect_hits), so a
# k-surface translucent stack costs ceil((k+1)/KBUF) traversals instead of
# k+1 while all-opaque rays stay at one. Valid for any batch (the drain
# evaluates translucent attenuation exactly like the march); shares the
# march's output up to the seam-merge corner the camera peel also has.
SHADOW_ANYHIT = (
    "gather"
    if env_str("ALGAN_SHADOW_ANYHIT", "0").strip().lower() == "gather"
    else env_flag("ALGAN_SHADOW_ANYHIT", False)
)


def set_shadow_anyhit(enabled):
    """Select the shadow-query early-out mode (see ``SHADOW_ANYHIT``).

    ``True`` enables the opaque any-hit walks, the string ``"gather"`` the
    KBUF gather-march, ``False`` the classic ordered march. Takes effect at
    the next render batch.
    """
    global SHADOW_ANYHIT
    if isinstance(enabled, str) and enabled.strip().lower() == "gather":
        SHADOW_ANYHIT = "gather"
    else:
        SHADOW_ANYHIT = bool(enabled)


# Build the dedicated opaque-only STBVHs only when a rollout that walks them
# (WF_OPAQUE_CLOSEST / WF_OPAQUE_PREPASS) is live at build time; otherwise
# alias the main tree -- same kernel ABI, and the opaque-tree reads are
# compiled out by the same templates that gate those rollouts. Saves the
# second per-geometry build (~40% of per-batch BVH build time) and its
# arena bytes. ALGAN_OPAQUE_BVH_SKIP_DEAD=0 restores the unconditional
# builds (byte-level A/B: the skip also shrinks the merged scene the arena
# planner measures).
OPAQUE_BVH_SKIP_DEAD = env_flag("ALGAN_OPAQUE_BVH_SKIP_DEAD", True)


def set_opaque_bvh_skip_dead(enabled):
    """Toggle skipping the dedicated opaque-only STBVH builds while no
    rollout consumes them (see ``OPAQUE_BVH_SKIP_DEAD``). Takes effect at
    the next batch's scene merge.
    """
    global OPAQUE_BVH_SKIP_DEAD
    OPAQUE_BVH_SKIP_DEAD = bool(enabled)


def refit_bvh_active():
    """Live effective value of the refit-BVH toggle: the legacy textured /
    sorted-material orchestrators walk the classic tree only.
    """
    return BVH_REFIT and not WF_TEXTURED and WAVEFRONT_SORT_MATERIALS is not True


# Hybrid raster front-end for deterministic primary visibility.
# Eligible flat-triangle/Bezier batches replace iteration zero with a typed
# opaque visibility buffer, alpha-filtered ordered fragment runs, optional exact
# sparse hard-shadow events, and a serial per-pixel resolve. Ordering matches the
# classic transitive depth-bin/descending-layer relation. Surviving reflected or
# refracted branches stay in the shared continuation pool and are traced by the
# classic bounce loop (whose per-iteration surface-event batches replaced the
# old per-slot K-buffers). Primary hard shadows and emitter-radius
# soft shadows use a sparse any-hit event queue. PN geometry is preserved and
# simply falls back to classic primary traversal. The straight-ray safety limit
# is MAX_SURFACES_PER_RAY (currently 256), not literally unbounded. Custom
# scatter, mem-trim, in-place AA, near clipping and legacy routes still fall
# back to classic. Default ON (ALGAN_HYBRID_RASTER=0 restores the classic
# iteration-zero wavefront).
HYBRID_RASTER = env_flag("ALGAN_HYBRID_RASTER", True)


def set_hybrid_raster(enabled):
    """Toggle the hybrid raster primary-visibility front-end (see
    ``HYBRID_RASTER``).
    """
    global HYBRID_RASTER
    HYBRID_RASTER = bool(enabled)


# Screen-space intersection mode inside the hybrid raster frontend. When on
# (default), one projection record is precomputed per (frame, triangle), and
# candidate chunks use edge functions plus perspective-correct barycentrics.
# Invalid/camera-plane-straddling projections fall back to exact per-pixel
# Moller-Trumbore ray casting. ALGAN_RASTER_SS=0 forces ray casting for all
# triangle candidates; the optimal policy may eventually be selected per pair.
RASTER_SS = env_flag("ALGAN_RASTER_SS", True)


def set_raster_screen_space(enabled):
    """Toggle screen-space rasterization in the hybrid raster front-end (see
    ``RASTER_SS``).
    """
    global RASTER_SS
    RASTER_SS = bool(enabled)


# Once-per-window batched circuit screen-bounds precompute inside the hybrid
# raster front-end (the bezier analogue of the per-batch triangle projection
# table). The per-(tile, frame) fallback re-projects every circuit's AABB
# corners with ~130 small tensor dispatches per call, which dominates host
# time on circuit-only scenes (tiny scenes measured ~8s of a ~19s render).
# Byte-identical by construction -- identical elementwise arithmetic, batched
# over the frame dimension; validated by benchmarks/_raster_bez_pre_parity.py.
# The toggle is a kill-switch / A-B hook.
RASTER_BEZ_PRECOMPUTE = env_flag("ALGAN_RASTER_BEZ_PRECOMPUTE", True)


def set_raster_bez_precompute(enabled):
    """Toggle the batched circuit screen-bounds precompute in the hybrid
    raster front-end (see ``RASTER_BEZ_PRECOMPUTE``).
    """
    global RASTER_BEZ_PRECOMPUTE
    RASTER_BEZ_PRECOMPUTE = bool(enabled)


# The flat-triangle companion of RASTER_BEZ_PRECOMPUTE: batches the bbox /
# class-mask derivation and candidate pair emission that ``_frame_pairs``
# performed per (tile, frame) on top of the per-batch projection table.
# Byte-identical by construction; same parity script.
RASTER_TRI_PRECOMPUTE = env_flag("ALGAN_RASTER_TRI_PRECOMPUTE", True)


def set_raster_tri_precompute(enabled):
    """Toggle the batched triangle screen-bounds precompute in the hybrid
    raster front-end (see ``RASTER_TRI_PRECOMPUTE``).
    """
    global RASTER_TRI_PRECOMPUTE
    RASTER_TRI_PRECOMPUTE = bool(enabled)


# Camera-plane clip for candidate bboxes (raster_pipeline._clipped_screen_
# extents).  A primitive straddling the camera plane has no bounded projection,
# so the front-end used to hand it the entire window as its candidate bbox --
# at HD, ~65k candidate chunks for one primitive-frame.  A camera travelling
# past or around the scene puts most of the geometry in that state at once,
# which is how ``camera.orbit`` ran renders out of memory.  With this on, a
# straddler is clipped to the front half-space and gets the real screen extent
# of the part a primary ray can reach (and is culled outright when that lands
# off screen); only primitives passing essentially through the camera origin
# still fall back to the whole window.
#
# Output-neutral: candidate pixels are exact-tested either way, so a tighter
# bbox only skips pixels that would have missed.  Off restores the old
# full-window straddler bbox.  Parity: benchmarks/_raster_straddle_clip_parity.py;
# the conservativeness proof is brute-forced by
# benchmarks/_raster_clip_extents_check.py.
RASTER_STRADDLE_CLIP = env_flag("ALGAN_RASTER_STRADDLE_CLIP", True)


def set_raster_straddle_clip(enabled):
    """Toggle the camera-plane clip of hybrid-raster candidate bboxes (see
    ``RASTER_STRADDLE_CLIP``).
    """
    global RASTER_STRADDLE_CLIP
    RASTER_STRADDLE_CLIP = bool(enabled)


# Empty-pixel fast path of the raster resolve: the host pre-fills every
# primary's committed state with the retired-empty result (pix_accum row
# [0,0,0,0, 1,1,1] -- zero colour, full leftover background weight -- with
# the pool already pre-marked DONE), so ``raster_first_shade`` threads whose
# pixel has no fragments and no z-prepass winner exit before ray generation
# with zero writes, and worked pixels *store* their leftover weight instead
# of atomically accumulating it onto a zero base.  A tile with no candidate
# pairs at all additionally skips the resolve (and shadow-event) launches
# entirely.  Empty screen regions previously paid ~15 ms/tile of per-pixel
# state writes.  Byte-identical (same values, different write path);
# validated by benchmarks/_raster_empty_skip_parity.py.  Kill-switch / A-B
# hook; read once per render batch so the host fill and the kernel template
# always agree.
RASTER_EMPTY_SKIP = env_flag("ALGAN_RASTER_EMPTY_SKIP", True)


def set_raster_empty_skip(enabled):
    """Toggle the empty-pixel fast path of the hybrid raster resolve (see
    ``RASTER_EMPTY_SKIP``).
    """
    global RASTER_EMPTY_SKIP
    RASTER_EMPTY_SKIP = bool(enabled)


# Host-side per-frame candidate-class summary flags for the batched screen-
# bounds tables: one conservative (opaque, translucent) "any candidates"
# bool per frame, computed once per window and moved to the host beside the
# tables.  ``_window_pairs`` then skips its per-tile tensor work -- most
# importantly the synchronizing ``.nonzero()`` inside ``_class_pairs_flat``
# -- for every (tile, class) whose covered frames provably have no
# candidates.  Byte-identical: a skipped class is exactly one whose mask was
# all-false, where ``_class_pairs_flat`` returned None anyway.  Same parity
# script as RASTER_EMPTY_SKIP.
RASTER_PAIR_FLAGS = env_flag("ALGAN_RASTER_PAIR_FLAGS", True)


def set_raster_pair_flags(enabled):
    """Toggle the host-side per-frame candidate-class flags used to skip
    empty per-tile pair emission (see ``RASTER_PAIR_FLAGS``).
    """
    global RASTER_PAIR_FLAGS
    RASTER_PAIR_FLAGS = bool(enabled)


# Covered-pixel-compacted resolve: the rasterizer already visits only the
# pixels a primitive covers (its z-prepass sets a z-winner, its count emits
# surviving fragments), so the set of pixels the resolve must actually shade
# -- ``(fragments > 0) OR (z-winner)`` -- is a compact list built from those
# per-pixel products.  ``raster_first_shade`` then launches one thread per
# COVERED pixel instead of one per tile pixel that early-outs, turning the
# resolve from O(tile pixels) into O(covered pixels).  Empty pixels keep the
# host's retired-empty pre-fill untouched (so this requires RASTER_EMPTY_SKIP
# and is disabled under an environment map, where empty pixels still sample
# the sky in the resolve).  Byte-identical: the covered list is the ascending
# nonzero order, so covered pixels are shaded in their original relative order
# and skipped pixels do exactly what their early-out did (nothing).
RASTER_COVERED_SHADE = env_flag("ALGAN_RASTER_COVERED_SHADE", True)


def set_raster_covered_shade(enabled):
    """Toggle the covered-pixel-compacted raster resolve (see
    ``RASTER_COVERED_SHADE``).
    """
    global RASTER_COVERED_SHADE
    RASTER_COVERED_SHADE = bool(enabled)


# Fully sparse primary-raster lifecycle.  The classic hybrid front-end used
# conservative candidate bboxes for geometry work, but allocated/initialized
# wavefront state, z/run buffers, accumulators, and a compaction scan for every
# pixel in the enclosing linear wavefront tile.  This path first emits exact
# hit records for every candidate, sorts/culls them in sparse hit space, and
# allocates every downstream structure for the unique covered pixels only.
#
# It requires the retired-empty/background identity used by
# RASTER_EMPTY_SKIP, covered-pixel resolve semantics, post-process tonemapping,
# and no environment map.  When an environment map is present every primary
# pixel genuinely samples the sky, so full-screen state is coverage work rather
# than empty overhead and the dense path remains correct.
RASTER_SPARSE_COVERAGE = env_flag("ALGAN_RASTER_SPARSE_COVERAGE", True)


def set_raster_sparse_coverage(enabled):
    """Toggle the exact covered-pixel lifecycle of the hybrid raster path."""
    global RASTER_SPARSE_COVERAGE
    RASTER_SPARSE_COVERAGE = bool(enabled)


# Analytic anti-aliasing (see DESIGN_analytic_aa.md). Instead of rendering at
# ``anti_alias_level`` times the output resolution and box-filtering back down
# (aa^2 work for every stage), each raster fragment carries the FRACTION OF THE
# PIXEL SQUARE its primitive covers, and the resolve folds that into the
# fragment's alpha. One shade per fragment, coverage resolved continuously.
#
# PHASE 1 (implemented): Bezier circuits only -- text and 2D shapes, the bulk of
# real Algan content. ``_bezier_point_metrics`` already returns the distance to
# the nearest outline segment plus a crossing parity, i.e. a signed distance
# field, so circuit coverage is a box filter of an SDF that is already computed.
# Circuits also have no shared-edge seam problem (a glyph or shape is ONE closed
# circuit), which is why they can ship ahead of triangles.
#
# Flat triangles are covered too (see ANALYTIC_AA_TRI below), and the quantities
# coverage cannot express analytically -- shadow-edge visibility and the image
# seen inside a reflection or refraction -- are handled by taking N sub-pixel
# samples of those specific queries (ANALYTIC_AA_SECONDARY_SAMPLES). Measured
# against the supersampled anti_alias_level=2 default across eleven
# feature-specific scenes, analytic AA at aa=1 is better on eight and 7-9% short
# on three (specular highlights, a flat mirror's reflected image, a lens's
# refracted image), where the residual is the CONTENT of a minified secondary
# image. Read DESIGN_analytic_aa.md ss19 before dropping ``anti_alias_level``
# to 1; what is still untouched is texture minification (no mip chain).
ANALYTIC_AA = env_flag("ALGAN_ANALYTIC_AA", True)

# PHASE 2 (implemented): flat triangles. Coverage comes from the screen-space
# edge functions ``_ss_pixel`` already evaluates, normalised by the edge lengths
# in columns 10:12 of the projection table. Triangles need a seam rule that
# circuits do not: two triangles sharing an edge inside a pixel cover it
# completely between them, and plain multiplicative compositing would leave a
# background-coloured lattice on every internal edge. The resolve therefore
# tracks transmittance independently for the fixed sub-pixel samples; disjoint
# masks partition the pixel without a source-object side table.
#
# Subordinate per-geometry switches (only meaningful while ANALYTIC_AA is on).
ANALYTIC_AA_BEZ = env_flag("ALGAN_ANALYTIC_AA_BEZ", True)
#
# Triangle coverage: exact fixed-point rasterization (a 1/4096-pixel integer
# lattice, int64 edge functions and a top-left fill rule) partitions eight
# sub-pixel samples among the triangles covering a pixel, the seam rule sums the
# disjoint sub-areas of one object, and per-sample occlusion keeps a mesh's back
# faces out of its own silhouette. Against an anti_alias_level=4 reference it
# beats the plain aliased render on every config -- a subdivided sphere, a
# translucent one, sub-pixel rods, a slanted quad -- at 40-78% less error, with
# essentially the reference's own edge gradation (588 distinct edge levels
# against 608). See DESIGN_analytic_aa.md ss14-ss16.
ANALYTIC_AA_TRI = env_flag("ALGAN_ANALYTIC_AA_TRI", True)

# The seam rule itself. Off, coverage still scales alpha but consecutive
# fragments of one object composite multiplicatively instead of unioning their
# disjoint sub-areas -- which is the lattice this exists to remove. Kept as a
# toggle purely so the parity script can measure the difference; there is no
# reason to turn it off in a real render.
ANALYTIC_AA_SEAM = env_flag("ALGAN_ANALYTIC_AA_SEAM", True)

# What to do with a triangle that CONTAINS NO SUB-PIXEL SAMPLE. The exact
# fixed-point test answers "does this triangle contain this sample"; a triangle
# narrower than the sample spacing contains none, and a mesh produces a rim of
# exactly those where it turns edge-on at a silhouette. The four policies:
#
#   drop      Contribute nothing, exactly as supersampling does -- a sub-pixel
#             sample either lands on the geometry or it does not. Sound because
#             the fill rule PARTITIONS the samples: any sample a narrow triangle
#             misses is contained by whichever neighbour of the tiling does
#             contain it, so dropping cannot open a hole in a closed surface.
#   exact     Claim the nearest sample with the EXACT area of (triangle n pixel
#             square) as its coverage, but do not occlude it (the sample belongs
#             to whoever contains it). Keeps sub-sample-width geometry visible.
#   exact_occ As ``exact``, but the claimed sample is also occluded. NOTE this
#             now coincides with ``exact``: per-sample transmittance (ss18) has
#             no separate occlusion set to opt into -- attenuating a sample IS
#             occluding it -- and an areal fragment attenuates all of them.
#   area      The pre-exact-area fallback: the continuous product of clamped
#             edge distances. It is a reconstruction filter that deliberately
#             spreads coverage half a pixel past the geometry, so a rim of
#             tiling slivers sums to more than the pixel and dilates every mesh.
#             Kept only so the parity script can measure against it.
#
# See DESIGN_analytic_aa.md ss15/ss16 for the measurements behind the default.
ANALYTIC_AA_SLIVER_MODES = ("area", "exact", "drop", "exact_occ")
ANALYTIC_AA_SLIVER = env_str("ALGAN_ANALYTIC_AA_SLIVER", "drop")

# Exact, angle-aware coverage for a circuit's boundary instead of a box filter
# of its signed distance.
#
# `clamp(d + 0.5, 0, 1)` is the exact area of (half-plane n pixel) for an
# AXIS-ALIGNED boundary and for no other orientation: it is the b == 0 case of
# the general formula (_halfplane_clip_area). At 45 degrees it reports full
# coverage at d = 0.5 where the truth is 0.957, peaking at 0.043 of coverage in
# between -- a systematic error in the edge's ANGLE, so diagonal edges of a glyph
# carry visibly different weight from horizontal ones. The exact form needs the
# boundary's DIRECTION, which _bezier_point_metrics forms as the closest-point
# vector and used to discard.
#
# Rides into the kernels inside the ``aa_bez`` template value (1 box filter,
# 2 exact) rather than as a constant, so each form gets its own compiled variant
# and its own offline-cache entry -- the same trap the sliver policy avoids the
# same way. See DESIGN_analytic_aa.md ss21.
ANALYTIC_AA_EXACT = env_flag("ALGAN_ANALYTIC_AA_EXACT", True)

# Model a circuit's local boundary with the TWO nearest segments (a strip or a
# corner) instead of one half-plane -- THE ORIENTED WEDGE
# (DESIGN_analytic_aa_v2.md ss5). Default ON (2026-08-13): both walls' inward
# sides come from storage (edges_2d column 5, written by the flatten-time
# parity probe), which retires the ss21.6 handedness calibration that flipped
# at exactly the corners the model exists for. Convex vs reflex is which RAY
# of its line each wall segment occupies, read off the closest points against
# the apex. Validated standalone to 0.0017/0.0010 (convex/reflex worst
# coverage error over 600 random corners, benchmarks/_aa_wedge_check.py); at
# matched dilation the wedge beats both the box and the lone-exact arms on
# stem/corner/glyph and improves slant -- the ss21.2 stem failure
# (text -6.8% vs the box filter) is gone.
ANALYTIC_AA_BEZ_WEDGE = env_flag("ALGAN_ANALYTIC_AA_BEZ_WEDGE", True)

# The ss21.3/21.8/21.9 exact-triangle formulations (single exact area vs the
# mask, packed cells, scalar surface accounting) are DELETED, not parked:
# DESIGN_analytic_aa_v2.md's run-corrected representation (ANALYTIC_AA_RUN)
# supersedes them, and its ss8 Phase D note plus DESIGN_analytic_aa.md ss21
# keep the record of what was measured and why they failed. The rule they all
# broke: a fragment's CLAIM and its OCCLUSION must be the same quantity, and
# only atomic sub-pixel ownership guarantees it -- the run rule keeps atomic
# masks for everything contended and corrects magnitude per RUN, where areas
# provably sum.

# RUN-CORRECTED triangle coverage (DESIGN_analytic_aa_v2.md ss4): the shipped
# 8-sample fill-rule masks stay the atomic ownership substrate for everything
# contended, and the exact clipped area is layered on top where nothing is --
# fragments carry ``_pixel_clip_area`` in ``frag_cov``, sample-less slivers are
# emitted as area donors at their clipped centroid, and the resolve corrects
# each uncontended RUN (consecutive same-(surface, facing) fragments over
# uniform per-sample transmittance) by the single scalar ``E / Q``. Every
# contended case falls back to the shipped per-sample behavior bit-for-bit --
# the uniform-svis gate IS the "no overlap" predicate. Subordinate to
# ANALYTIC_AA / ANALYTIC_AA_TRI; the sliver policy knob is inert under it
# (sliver behavior is fixed by the design, not configurable).
# Default ON (2026-08-13) on the v2 ss7.2 ladder: static mesh L1
# 0.0355 -> 0.0292 against the aa=4 reference, tri video 0.119 -> 0.107 at
# edge levels 620/621, seam notches inside the documented band, trans
# improves, thin gains its reachable share (0.857 -> 0.884; the 0.99 target
# was calibrated on the rejected cells accounting -- see the ss8 Phase D
# note). Worst-case cost is +6.6% frame device on sub-pixel-diced meshes;
# RUN=0 is byte-identical to the pre-v2 renderer.
ANALYTIC_AA_RUN = env_flag("ALGAN_ANALYTIC_AA_RUN", True)

# The corr > 1 accounting rule (v2 ss4.4), the design's one open empirical
# question, decided by harness: "clamp" scales the run's per-sample writes by
# corr and clamps each at zero (claim exact, leftover keeps a bounded residual
# of the shed error); "redistribute" additionally pushes the clamped residue
# onto the run's unowned samples (leftover exact, weirder per-sample
# semantics). Compile-time template value; both stay byte-identical while
# ANALYTIC_AA_RUN is off.
# Measured (v2 ss4.4, decided by harness as designed): redistribute wins --
# tri L1 0.107 vs clamp's 0.110 with edge levels 620 against the aa=4
# reference's own 621, seam notches 9 vs 12, trans/thin at parity. Exact
# leftovers cost two registers and a run-end scale.
ANALYTIC_AA_RUN_RULES = ("clamp", "redistribute")
ANALYTIC_AA_RUN_RULE = env_str("ALGAN_ANALYTIC_AA_RUN_RULE", "redistribute")

# Sub-pixel samples for what coverage CANNOT antialias analytically: the image
# seen inside a reflection or through refracting glass. Coverage resolves a
# mirror's own outline exactly, but the reflected scene is sampled by the
# continuation ray, and one continuation per pixel aliases however good the
# primary coverage is (DESIGN_analytic_aa.md ss7).
#
# With this at N, a reflective or refractive hit spawns N continuations instead
# of one: each is the primary ray re-generated through a different sub-pixel
# position and re-intersected with that hit's own plane, so the reflected image
# is sampled at N sub-pixel positions, each carrying 1/N of the throughput. At
# N=4 those positions are the 2x2 grid anti_alias_level=2 supersamples at, which
# is the arm this is meant to match.
#
# The split happens ONCE, at the primary hit; deeper bounces continue as single
# rays, so the cost is N times the secondary traversal, not N^depth. Only the
# reflective/refractive pixels pay it. 1 disables it, and is byte-identical.
ANALYTIC_AA_SECONDARY_SAMPLES = env_int("ALGAN_ANALYTIC_AA_SECONDARY", 4)

# Minimum share of a pixel a REFLECTED or REFRACTED branch must carry before it
# is worth spending N sub-pixel continuations on instead of one.
#
# Without this, a plain glossy sphere -- whose only "reflection" is the ~4%
# Fresnel sheen every PBR dielectric has -- spawns four extra traced rays per
# pixel for a lobe contributing 4% of its colour, and measures both slower and
# slightly worse than plain supersampling. The whole value of coverage is that
# the expensive fallbacks fire only on the pixels that need them.
ANALYTIC_AA_SECONDARY_MIN_ENERGY = env_float("ALGAN_ANALYTIC_AA_SECONDARY_MIN_ENERGY", 0.12)

# Roughness-driven GLOSSY REFLECTION for the deterministic tracer: a rough
# reflector's continuation rays spread over a GGX lobe instead of all taking the
# one mirror direction, so a reflected image blurs as roughness rises. Without
# it a MeshStandardMaterial(roughness=0.18) shows a razor-thin reflection
# beside a broad direct highlight -- the same material described two ways.
#
# The lobe is sampled by the continuations the fragment ALREADY spawns
# (ANALYTIC_AA_SECONDARY_SAMPLES): the taps that vary in sub-pixel position now
# vary in lobe direction too, so this costs no extra rays and no extra pool
# slots. It therefore only reaches a fragment that takes the secondary-sampling
# branch; a fragment with one tap stays specular-perfect (blurring needs more
# than one sample -- a single deterministic tap is not a blur, it is a mirror
# pointing the wrong way).
#
# Deterministic by construction: the taps are a stratified GGX radial CDF plus a
# golden-angle azimuth, indexed by tap number, with an optional per-pixel
# Bayer rotation. No ti.random anywhere, so the same frame renders byte-identical
# every time and an animation cannot hiss. Roughness below
# ``_GLOSSY_MIN_ROUGHNESS`` (raster_taichi) takes the untouched mirror path, so
# a true mirror is byte-identical to the pre-glossy build.
#
# See DESIGN_analytic_aa.md ss20.
GLOSSY_REFLECTION = env_flag("ALGAN_GLOSSY_REFLECTION", True)

# Rotate each pixel's lobe fan by a 4x4 Bayer index (interleaved sampling), so
# four taps read as a smear rather than four ghost copies of the reflected
# image: neighbouring pixels sample different parts of the lobe and the eye
# integrates across them. Fixed in SCREEN space, hence still frame-independent
# -- the pattern does not swim, twinkle or depend on time. Off restores the
# plain per-fragment fan (kept so the parity script can measure the difference).
GLOSSY_INTERLEAVE = env_flag("ALGAN_GLOSSY_INTERLEAVE", True)


def set_glossy_reflection(enabled, *, interleave=None):
    """Toggle roughness-driven glossy reflections (see ``GLOSSY_REFLECTION``)."""
    global GLOSSY_REFLECTION, GLOSSY_INTERLEAVE
    GLOSSY_REFLECTION = bool(enabled)
    if interleave is not None:
        GLOSSY_INTERLEAVE = bool(interleave)


def glossy_reflection_mode():
    """Live glossy-lobe mode: 0 off, 1 fan only, 2 fan + per-pixel rotation.

    Read at call time (never imported by value) and returned as an int, because
    it reaches the resolve as a TEMPLATE value: each mode compiles its own
    kernel variant, so the offline cache -- which does not invalidate on
    ``@ti.func`` edits, let alone on a Python constant -- cannot serve one
    mode's kernel for another.
    """
    if not GLOSSY_REFLECTION:
        return 0
    return 2 if GLOSSY_INTERLEAVE else 1


# Minimum half-width, in pixels, of a filled circuit's drawn region. This
# replaces the classic ``outline_w = 0.6 * pixel_size`` fill dilation, whose
# purpose is to keep sub-pixel features (hairlines, thin glyph stems, degenerate
# zero-area fills) from vanishing entirely. The classic constant is 0.6 of a
# SUPERSAMPLE pixel and is therefore NOT anti-alias-level invariant: at the
# reference AA=2 it dilates by 0.3 output pixels, at AA=1 by 0.6. Analytic AA
# runs at AA=1, so 0.3 reproduces the reference appearance rather than doubling
# every stroke weight. Tune only against rendered Text/Tex.
ANALYTIC_AA_BEZ_MIN_HALF_WIDTH = env_float("ALGAN_ANALYTIC_AA_BEZ_MIN_HALF_WIDTH", 0.3)

# Maximum curve-to-chord flattening error, in pixels, for Bezier circuits under
# analytic AA (overrides the primitive's own ``num_pixels_per_sample`` only when
# it is looser). The classic 0.5 is measured against the SUPERSAMPLED height, so
# at the AA=2 reference it is 0.25 output pixels; at AA=1 it would relax to 0.5
# and a continuous coverage function would expose the flattening facets that box
# filtering currently hides. Costs edges (memory + _bezier_point_metrics work).
ANALYTIC_AA_CHORD_TOLERANCE = env_float("ALGAN_ANALYTIC_AA_CHORD_TOLERANCE", 0.25)


def set_analytic_aa(
    enabled,
    *,
    bezier=None,
    triangles=None,
    seam=None,
    sliver=None,
    secondary=None,
    exact=None,
    run=None,
    run_rule=None,
):
    """Toggle analytic anti-aliasing (see ``ANALYTIC_AA``)."""
    global ANALYTIC_AA, ANALYTIC_AA_BEZ, ANALYTIC_AA_TRI, ANALYTIC_AA_SEAM
    global ANALYTIC_AA_SLIVER, ANALYTIC_AA_SECONDARY_SAMPLES, ANALYTIC_AA_EXACT
    global ANALYTIC_AA_RUN, ANALYTIC_AA_RUN_RULE
    if secondary is not None:
        ANALYTIC_AA_SECONDARY_SAMPLES = int(secondary)
    if exact is not None:
        ANALYTIC_AA_EXACT = bool(exact)
    if run is not None:
        ANALYTIC_AA_RUN = bool(run)
    if run_rule is not None:
        if run_rule not in ANALYTIC_AA_RUN_RULES:
            raise ValueError(f"run_rule must be one of {ANALYTIC_AA_RUN_RULES}")
        ANALYTIC_AA_RUN_RULE = run_rule
    ANALYTIC_AA = bool(enabled)
    if bezier is not None:
        ANALYTIC_AA_BEZ = bool(bezier)
    if triangles is not None:
        ANALYTIC_AA_TRI = bool(triangles)
    if seam is not None:
        ANALYTIC_AA_SEAM = bool(seam)
    if sliver is not None:
        if sliver not in ANALYTIC_AA_SLIVER_MODES:
            raise ValueError(f"sliver must be one of {ANALYTIC_AA_SLIVER_MODES}")
        ANALYTIC_AA_SLIVER = sliver


def analytic_aa_secondary_samples():
    """Live continuation-ray sample count; 1 (off) unless analytic AA is on.

    Snapped to a supported set size, because the sub-pixel positions are a
    compile-time table and N reaches the resolve as a template value.
    """
    if not ANALYTIC_AA:
        return 1
    n = int(ANALYTIC_AA_SECONDARY_SAMPLES)
    for k in (8, 4, 2):
        if n >= k:
            return k
    return 1


def analytic_aa_sliver_mode():
    """Index of the live sample-less-triangle policy in the mode tuple.

    Read at call time (never imported by value) and returned as an int, because
    it reaches the kernels as part of the ``aa`` template value: the geometry
    kernels see ``1 + mode``, so each policy compiles its own variant and the
    offline cache cannot serve one policy's kernel for another.
    """
    try:
        return ANALYTIC_AA_SLIVER_MODES.index(ANALYTIC_AA_SLIVER)
    except ValueError:
        return ANALYTIC_AA_SLIVER_MODES.index("drop")


def analytic_aa_bez_active():
    """Live effective value of circuit analytic coverage.

    Read at call time, never imported by value: settings are module globals with
    env-var defaults and user code flips them after import.
    """
    return ANALYTIC_AA and ANALYTIC_AA_BEZ


def analytic_aa_bez_mode():
    """Circuit coverage as the kernels' ``aa_bez`` template value.

    0 off, 1 the box filter, 2 the exact angle-aware area
    (``ANALYTIC_AA_EXACT``), 3 that plus the two-segment boundary model
    (``ANALYTIC_AA_BEZ_WEDGE``, default off -- see its comment).
    The distinction rides in the template value so the two forms cannot share an
    offline-cache entry; everything downstream that only asks whether circuit
    coverage is on keeps testing it for truth.
    """
    if not analytic_aa_bez_active():
        return 0
    if not ANALYTIC_AA_EXACT:
        return 1
    return 3 if ANALYTIC_AA_BEZ_WEDGE else 2


def analytic_aa_tri_active():
    """Live effective value of flat-triangle analytic coverage."""
    return ANALYTIC_AA and ANALYTIC_AA_TRI


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
WF_TEXTURED = False


def set_textured_wavefront(enabled):
    """Reject the removed legacy texture-lookup wavefront renderer."""
    global WF_TEXTURED
    if bool(enabled):
        WF_TEXTURED = False
        raise UnsupportedFeatureError(
            "The legacy textured wavefront renderer is unsupported and cannot "
            "be enabled. Use the general deterministic wavefront renderer."
        )
    WF_TEXTURED = False


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
MERGE_ON_GPU = env_flag("ALGAN_MERGE_ON_GPU", True)

# Multiplier turning a batch's packed ``_rt_*`` input bytes into a conservative
# estimate of the GPU merge's transient peak (the out-of-place cat / argsort /
# unique / dyadic-time-pyramid scratch plus the merged output). Measured peaks
# run ~3-6x the packed inputs; the default leaves margin so the proactive
# headroom check rarely lets a batch through that the OOM retry then has to
# shrink. Read live.
# The merge and the projection build out of place in *pool headroom*, not in
# the render arena, so the arena's high-water mark cannot see them and the
# runtime chunk model does not cover them. They keep a multiplicative bound on
# their packed inputs. It is deliberately generous: torch's allocator counters
# cannot see Taichi's separate pool at all, and the out-of-memory handler is
# the exact fallback when the estimate is low.
MERGE_GPU_PEAK_FACTOR = env_float("ALGAN_MERGE_GPU_PEAK_FACTOR", 6.0)


def merge_gpu_peak_factor():
    """Live multiplier bounding the GPU merge's transient out-of-arena peak."""
    return MERGE_GPU_PEAK_FACTOR


# Exact measurement of the GPU merge's transient peak, which calibrates
# ``MERGE_GPU_PEAK_FACTOR``. This used to default off because it called
# ``torch.cuda.reset_peak_memory_stats`` directly and so destroyed the
# process-wide peak counter ``profiling_utils`` reports for the whole render.
# It now goes through ``memory_utils.begin_cuda_peak``/``end_cuda_peak``, which
# remember the displaced high-water mark, so measuring costs a pair of cheap
# counter reads and nothing else -- hence on by default. The headroom bound
# itself is still the ``MERGE_GPU_PEAK_FACTOR`` estimate.
MERGE_TRACK_PEAK = env_flag("ALGAN_MERGE_TRACK_PEAK", True)


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

    return _RENDER_DEVICE.type == "cuda"


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
PROJECT_ON_GPU = env_flag("ALGAN_PROJECT_ON_GPU", True)

# Conservative multiplier from a batch's pre-projection source-geometry bytes
# to the projection's transient device peak (source + shading scratch + packed
# ``_rt_*`` output; the polyline sampling can expand bezier geometry well past
# its control points, hence a larger default than the merge factor). Bounds the
# projection against the pool headroom before it is attempted; the OOM retry is
# the exact fallback. Read live.
PROJECT_GPU_PEAK_FACTOR = env_float("ALGAN_PROJECT_GPU_PEAK_FACTOR", 8.0)


def project_gpu_peak_factor():
    """Live multiplier bounding projection's transient out-of-arena peak."""
    return PROJECT_GPU_PEAK_FACTOR


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

    return _RENDER_DEVICE.type == "cuda"


# --- logical PN / bezier subdivision-level criteria -------------------------
# The level searches that decide how finely each logical PN patch and each
# bezier segment is diced are reductions -- a few hundred thousand sample
# points in, one peak-pixel-error scalar per patch out -- written as ~30
# elementwise torch passes over large scratch. They are therefore bound by
# device bandwidth, not arithmetic: measured 67.9s (8.5% of a reference
# save_video) for the PN criterion and 18.4s (2.3%) for the bezier chord
# search. With this on they run as fused Taichi kernels
# (``logical_pn_taichi``) that keep every intermediate in registers.
#
# NOT byte-identical to the torch path: Taichi initialises with fast_math, so
# borderline patches round to a neighbouring level. That moves geometry (inside
# render_tolerance by construction), so a render baseline changes. Crack-
# freeness is preserved -- shared boundary curves still reach a bit-identical
# level, and the cross-thread reduction is order-independent ``max`` -- and is
# regression-tested by benchmarks/_logical_pn_crack_check.py.
#
# ALGAN_PN_CRITERION_KERNEL=0 restores the torch path (for A/B). Only used when
# projection runs on a CUDA render device: elsewhere the criterion's tensors
# live on the CPU, where launching Taichi against them stages every argument
# through VRAM (see generate_array_states' docstring), and projection may run
# on the prefetch worker rather than the render thread.
PN_CRITERION_KERNEL = env_flag("ALGAN_PN_CRITERION_KERNEL", True)


def set_pn_criterion_kernel(enabled):
    """Toggle the fused subdivision-level criterion kernels (see
    ``PN_CRITERION_KERNEL``).
    """
    global PN_CRITERION_KERNEL
    PN_CRITERION_KERNEL = bool(enabled)


def pn_criterion_kernel_active():
    """True when the level searches should use their fused Taichi kernels."""
    return PN_CRITERION_KERNEL and project_on_gpu_active()


# Feature bitmask for the UNSUPPORTED legacy textured wavefront (see
# WF_TEXTURED): each bit compiled one of the monolith's features back into the
# (otherwise lean) textured shade kernel, so the marginal occupancy /
# performance cost of each could be measured one at a time (see
# benchmarks/_wf_textured_features_ab.py). The features are added in the order
# beziers -> custom scatter -> shadows -> normal maps.
WF_TEX_BEZ = 1  # bezier-circuit traversal + shading
WF_TEX_SCATTER = 2  # per-material custom scatter dispatch (ray bouncing)
WF_TEX_SHADOWS = 4  # binary hard shadow rays (triangle occluders)
WF_TEX_NORMALMAP = 8  # tangent-space normal-map perturbation of the shading normal
WF_TEXTURED_FEATURES = env_int("ALGAN_WF_TEXTURED_FEATURES", 0)


def set_textured_features(mask):
    """Reject feature configuration for the removed textured renderer."""
    global WF_TEXTURED_FEATURES
    if int(mask) != 0:
        WF_TEXTURED_FEATURES = 0
        raise UnsupportedFeatureError(
            "Textured-wavefront feature masks are unsupported because that "
            "legacy renderer has been removed from the public execution path."
        )
    WF_TEXTURED_FEATURES = 0


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


WAVEFRONT_SORT_MATERIALS = "0"  # auto"


def set_material_sorting(enabled):
    """Reject the removed legacy sorted-material renderer when forced on."""
    global WAVEFRONT_SORT_MATERIALS
    parsed = _parse_sort_mode(enabled)
    if parsed is True:
        WAVEFRONT_SORT_MATERIALS = "auto"
        raise UnsupportedFeatureError(
            "The legacy sorted-material wavefront renderer is unsupported. "
            "Use the monolithic deterministic shade kernel."
        )
    WAVEFRONT_SORT_MATERIALS = parsed


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


# When True, the deterministic ray tracer casts hard shadows: each shaded
# triangle/PN fragment fires one shadow ray per point light and multiplies the
# light that remains through every occluder's transparency. Fully opaque
# occluders block the direct contribution. Implies per-fragment shading
# (shadows are evaluated in the lighting model) and forces the general kernel.
# Lights with a non-zero ``shadow_radius`` / ``shadow_angle`` (and area lights)
# get *soft* shadows: a fixed deterministic fan of SOFT_SHADOW_SAMPLES rays is
# traced across the emitter instead of a single ray. Off by default.
SHADOWS = False

# Number of shadow rays in the deterministic soft-shadow fan (per light with a
# non-zero shadow radius, per shaded fragment). More = smoother penumbras,
# linearly more shadow cost. Baked into the shade kernel at compile time; set
# the env var ALGAN_SOFT_SHADOW_SAMPLES before the first render to change it.


def set_ray_traced_shadows(enabled):
    """Toggle hard shadows in the *deterministic* ray tracer.

    When enabled, every shaded triangle/PN fragment traces one shadow ray per
    scene point light. Every partially opaque surface between the fragment and
    light attenuates its direct diffuse/specular term by ``1 - opacity``;
    stacked surfaces multiply, while a fully opaque surface blocks it. Ambient
    and emissive terms remain unchanged. Shadows are evaluated inside the
    wavefront shade kernel's per-fragment lighting model, so this implies
    :func:`set_fragment_shading` for the render. Lights with a non-zero
    ``shadow_radius`` / ``shadow_angle`` (and area lights) get *soft* shadows
    via a deterministic fan of ``SOFT_SHADOW_SAMPLES`` rays. Refractive glass
    transport still needs the physical path tracer
    (``set_samples_per_pixel(n)`` with ``n > 1``). Only the deterministic
    renderer (``set_samples_per_pixel(1)``, non-physical) is affected. Set
    before rendering.
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
    """Enable or disable tonemapping of the rendered frame.

    The curve is selected by :func:`set_tonemap_method` ("neutral", the Khronos
    PBR Neutral mapper, or "agx") -- not ACES, whatever the old docstring said.

    Disabling it makes output **linear**: an authored colour lands on the pixel
    it names, which is what you want when matching a reference image. This flag
    is honoured wherever the tonemap actually runs, so it works on its own --
    with ``post_process_tonemap`` on (the default) the composite writes linear
    HDR and the post stage simply clamps instead of applying a curve. There is
    no need to also disable ``post_process_tonemap``, and doing so costs HDR
    headroom (see :func:`set_post_process_tonemap`).
    """
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
    """Enable or disable post-process tonemapping instead of in-kernel tonemapping.

    Disabling it makes the composite write **uint8**, which clamps every channel
    -- including the glow lane -- to 0-255 before bloom runs. A mob with
    ``glow > 1`` therefore saturates and its halo comes out markedly dimmer and
    less saturated than on the default HDR path. Turn this off only for an A/B
    against the legacy in-kernel tonemap; to get linear output, use
    :func:`set_tonemapping` alone, which keeps the HDR buffer.
    """
    global POST_PROCESS_TONEMAP
    POST_PROCESS_TONEMAP = bool(enabled)


def hdr_frame_dtype():
    """dtype of the linear-HDR frame buffer used under post-process
    tonemapping.

    Defaults to float32. float16 (RGBA16F) halves the frame-buffer memory
    (so ~2x more frames per batch), but is opt-in via ``ALGAN_HDR_BUFFER_F16=1``
    because GPUs with poor FP16 throughput -- notably consumer Pascal
    (GTX 10-series) at ~1/64 FP32 -- run the f16 torch post-processing (and
    f16 buffer traffic) far slower than the memory saving is worth (measured
    ~80% slower end-to-end on a GTX 1050). On Turing/Ampere+ (fast f16) it is
    a clear win, so enable it there.
    """
    import torch

    if _HDR_BUFFER_F16:
        return torch.float16
    return torch.float32


POST_TONEMAP_KERNEL = env_flag("ALGAN_POST_TONEMAP_KERNEL", True)


def set_post_tonemap_kernel(enabled):
    """Toggle the standalone Taichi post-process tonemap kernel (vs the torch
    tonemap pipeline). The kernel reuses the in-composite tonemap ti.funcs and
    computes in f32, recovering most of the cost the move to post-process
    tonemapping added (the torch tonemap ran ~20 ops/pixel over every frame).
    Kill-switch / A-B hook.
    """
    global POST_TONEMAP_KERNEL
    POST_TONEMAP_KERNEL = bool(enabled)


def is_post_tonemap_kernel_enabled():
    return POST_TONEMAP_KERNEL


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
_MAT_DEFAULTS = [
    0.0,
    0.0,
    0.0,
    1.0,
    0.0666,
    0.0666,
    0.0666,
    30.0,
    1.0,
    0.0,
    0.0,
    1.0,
    1.5,
    1.0,
    1.0,
    1.0,
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
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
    raw or baked -- unchanged).
    """
    if shader is None:
        return 1
    return _core_shader_ids().get(shader, 1)


def _shader_is_core(shader):
    """True if ``shader`` has an in-kernel port (so its hits can be fragment
    shaded rather than baked).
    """
    return shader is not None and shader in _core_shader_ids()


def _constant_promotion_active():
    """True when constant-property -> 1x1-texture promotion applies to this
    render: it is enabled, and the batch will render through the deterministic
    fragment-shading general wavefront (the only path where a mob's colours are
    raw albedo, so a "constant colour" is genuinely constant per fragment, and
    the only kernel whose per-vertex reads are guarded for shrunk arrays).
    Every deterministic (samples <= 1) batch renders through that kernel.
    """
    return PROMOTE_CONSTANTS and FRAGMENT_SHADING and SAMPLES_PER_PIXEL <= 1


def _scene_has_user_pipeline(merged):
    """True if any merged primitive carries a custom fragment-pipeline id
    (``>= _USER_PIPELINE_BASE``), so the render must enable fragment shading.
    """
    cached = merged.get("has_user_pipeline")
    if cached is not None:
        return bool(cached)
    for key in ("tri_mat_id", "pn_mat_id"):
        arr = merged.get(key)
        # Compatibility for externally assembled scenes that predate the
        # cached flag.  Move tiny ids to the host before reducing so this
        # fallback cannot create a scalar/reduction workspace beside the
        # render arena.
        if (
            arr is not None
            and arr.numel()
            and int(arr.detach().cpu().max()) >= _USER_PIPELINE_BASE
        ):
            return True
    return False
