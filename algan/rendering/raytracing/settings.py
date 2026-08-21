"""The renderer's feature toggles, as module globals with setter functions.

Every ray-tracing switch lives here as a module-level global with an environment
variable default and a setter, and :data:`algan.SETTINGS`'s ``raytracing`` section
is the public face of them. The ones that change what the image looks like are
exposed directly; the kernel and performance switches are reachable through
``SETTINGS.raytracing.experimental``.

**Read these live** -- ``rt_settings.X`` at call time, not ``from ... import X``
at module import. Importing by value freezes a toggle at its import-time state,
before user code has had a chance to set it. That bug has shipped before.

:func:`set_unsupported_feature_policy` and :func:`report_unsupported_features`
decide what happens when a Scene asks for something the selected tracer cannot
do: raise (the default), warn, or ignore.
"""

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
WAVEFRONT_TILE_AUTO = env_flag("ALGAN_WAVEFRONT_TILE_AUTO", True) and not env_is_set(
    "ALGAN_WAVEFRONT_TILE"
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
# 1.4x on the raster resolve for a scene mixing several materials (measured on
# the fragment walk, the sheet resolve's predecessor), and NOTHING for a
# single-material scene or for ``wavefront_shade`` -- only the raster resolve
# sits close enough to its occupancy cliff for the dropped stages to matter.
# Hence experimental and off by default rather than on: ALGAN_FRAG_PID_GATE=1
# opts in.
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


# Per-triangle SURFACE identity at the granularity the mob declares, rather than
# one id per merged COLLECTION MEMBER. The member count is right only when one
# member is one surface, and it is wrong at both ends: ``Polyhedron`` hands the
# batcher one member per TRIANGLE (a Cube is twelve members, so the analytic-AA
# run rule can never span one of its faces), while a packed-grid ``Surface``
# hands it one member covering EVERY packed sphere at once (so distinct spheres
# are unioned into one surface). Mobs that know better stamp ``mesh_key`` (merge
# with the neighbours sharing it) or ``mesh_ids`` (subdivide into per-triangle
# shells) on the primitive they build, and
# ``primitives._mesh_ids_from_collection`` resolves those into explicit ids.
# Off restores the per-member ids exactly, so it is a byte-level A/B switch.
#
# DEFAULT ON since 2026-08. The quality question is settled and the correctness
# argument is the reason. Coarser identity is more correct (a Cube's face diagonal
# becomes an interior edge rather than a boundary between two "surfaces"), and
# it was held back because it also makes the v2 4.2
# ``U == _AA_MASK_ALL -> corr = 1`` short-circuit fire on facet-boundary pixels
# where the facets fill all eight sub-pixel samples without covering the pixel's
# area -- on an Icosahedron, 0.61% of covered pixels move into ``union-full``,
# 125 of them with ``1 - E`` past 0.30.
#
# benchmarks/_aa_run_gate_check.py now settles which of those two effects wins,
# by replaying the resolve's per-sample transmittance and scoring the coverage
# each pixel actually ends up with against an EXACT analytic reference (the
# per-fragment error metric it used to report could not: it did not model the
# transmittance at all). Measured at --res md on CPU, mean coverage error over
# silhouette pixels: Cube 0.0250 -> 0.0248, Icosahedron 0.0258 -> 0.0256, and
# every Surface-backed case unmoved because a Surface is already one merged
# member. Nothing regresses and nothing gains beyond noise, so the coverage
# evidence is NEUTRAL: what argues for this is the correctness case above, not a
# measured quality win. _analytic_aa_fillrule_check and _aa_dump_check both pass
# with it on.
#
# The packed-grid Surface -- the end this fixes in the other direction -- is now
# covered too, by two _aa_run_gate_check cases (a 4x4 pack whose footprints
# overlap, and a spaced-out control). It needed a defect fixed first: a packed
# grid is diced logical PN, and _dice_logical_pn built its patch->surface map
# from per-member counts alone, so the mesh_ids Surface stamps on it were
# resolved and then discarded. With that fixed, the harness's reference-free A/B
# (--mesh-ab, which unlike the scored column also sees the overlapping pixels
# the exact reference has to drop) reports the predicted gain: 18 of 36224
# pixels move on the overlapping pack and MESH_ID=0 is the side that paints
# MORE, while the non-overlapping control moves zero. Small, but it is the win
# this was said to be missing.
#
# Flipping also moves the fast-suite render by up to 49 channel values at solid
# edges, so BOTH device baseline sets have to be regenerated and
# expected_outputs_cuda/ needs a CUDA machine. DESIGN_mesh_identity.md 3.5, 4.5.
MESH_ID = env_flag("ALGAN_MESH_ID", True)


# Orient a closed ``Polyhedron``'s faces outward at construction
# (``shapes_3d.orient_faces_outward``). The face index lists Algan ships for the
# Platonic solids are Manim's and are not consistently oriented: 12 of an
# Icosahedron's 20 faces wind inward, 2 of 4 on a Tetrahedron, 2 of 8 on an
# Octahedron, 3 of 12 on a Dodecahedron, 0 of 6 on a Cube. The projected winding
# sign IS ``_AA_BACKFACE_BIT``, which is what separates a closed mesh's near and
# far sheets for the analytic-AA run rule, so on those solids the bit names
# nothing -- measured, 960 of an Icosahedron's 46220 covered pixels have one
# facing group holding BOTH sheets, against 4 with this on.
#
# DEFAULT ON since 2026-08. Measured, the fast-suite render (which draws a Cube,
# an Icosahedron and an Octahedron) is BYTE-IDENTICAL across this flag while
# MESH_ID is off --
# a per-triangle surface id makes every run one fragment, so the facing bit
# groups nothing. With MESH_ID=1 it does change the render, which is the
# mechanism: one id per solid leaves facing as the only separator between the
# near and far sheets. Read at Polyhedron construction, not at render time.
# IT DOES MOVE A `become` MORPH, which the byte-identical static result above
# does not cover and which cost a full-render investigation to pin down.
# Reversing an inward face reverses the vertex order WITHIN it, and `become`
# pairs primitives corner by corner, so the interpolation path changes: measured,
# Tetrahedron.become(Cube) differs by up to 227 channel values across this flag
# while a STATIC Tetrahedron is byte-identical and Tetrahedron.become(Tetrahedron)
# is too (there the reordering cancels on both sides). The endpoints are the
# correct solids either way; only the in-between path moves. That is what makes
# tests/full_renders' complex_hierarchy_become move by 197, entirely from this
# flag and not at all from MESH_ID.
# DESIGN_mesh_identity.md 3.7 and 6.5.
POLYHEDRON_WINDING = env_flag("ALGAN_POLYHEDRON_WINDING", True)


def set_polyhedron_winding(enabled):
    """Toggle outward face orientation for closed polyhedra (see
    ``POLYHEDRON_WINDING``). Takes effect for the next ``Polyhedron`` built.
    """
    global POLYHEDRON_WINDING
    POLYHEDRON_WINDING = bool(enabled)


def set_mesh_id(enabled):
    """Toggle mob-declared surface identity (see ``MESH_ID``). Takes effect at
    the next batch's primitive build.
    """
    global MESH_ID
    MESH_ID = bool(enabled)


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


# Self-shadow rejection by identity (DESIGN_mesh_identity_open.md ssI). A
# shadow ray currently rejects its own surface with MIN_HIT_DISTANCE plus a
# normal offset of 10 * MIN_HIT_DISTANCE -- absolute world-space constants
# applied to EVERY hit, so a small object resting on a plane loses its contact
# shadow within 1e-3 of the contact and grazing light on small geometry
# produces acne. On the sheet route's shadow queue the event's source surface
# id is available (packed into ``event_msk`` above the material pipeline id),
# so the acceptance test becomes
#
#     accept = (t < max_t) and (hit_mesh != src_mesh ? t > 0 : t > MIN_HIT_DISTANCE)
#
# and the cross-mesh threshold goes to zero while self-rejection stays exactly
# as safe. The rejection is per hit -- "same mesh AND near-zero t", never
# "same mesh": a concave solid legitimately shadows itself. Events without a
# usable source id (bezier-originated, or ids that do not fit the packing) and
# every path outside the sheet route's shadow queue keep today's epsilon.
# DEFAULT ON.
SHADOW_IDENTITY_REJECT = env_flag("ALGAN_SHADOW_IDENTITY_REJECT", True)


def set_shadow_identity_reject(enabled):
    """Toggle self-shadow rejection by identity (see
    ``SHADOW_IDENTITY_REJECT``). Takes effect at the next render batch.
    """
    global SHADOW_IDENTITY_REJECT
    SHADOW_IDENTITY_REJECT = bool(enabled)


# The acceptance floor a shadow ray keeps against its OWN primitive, as a
# fraction of the batch's scene scale (the diagonal of the merged triangle
# bounding box over every frame of the batch). This is what retires the last
# absolute constant on the shadow path: `MIN_HIT_DISTANCE` = 1e-4 is only ever
# right for a scene about ten units across, which is where the default below
# reproduces it exactly (1e-5 * 10). A scene authored at millimetre or
# kilometre scale gets a floor in proportion instead of acne at one end and
# erased contact at the other.
#
# The scale is GEOMETRIC, deliberately not the pixel footprint: the error this
# floor guards against is the positional error of the reconstructed hit point,
# which is a property of the coordinates and the tessellation. Tying it to
# pixels would shrink it as resolution rises and make a 4K render noisier than
# a 720p one from the same geometry.
SHADOW_EPS_RELATIVE = env_float("ALGAN_SHADOW_EPS_RELATIVE", 1e-5)

# What fraction of that floor a hit on the SAME mesh but a DIFFERENT primitive
# keeps. 0.0 is primitive-precise: only the triangle the ray actually started
# from is treated as a possible artifact, so a concave crease and a mesh with
# two separate parts get their contact shadow back. 1.0 restores mesh-wide
# rejection, which is what shipped first and what to compare against. Values
# in between buy back protection at mesh seams, where the reconstructed point
# of one facet can land under its neighbour: raise this if a diced curved
# surface shows seam speckle with the feature on.
SHADOW_NEAR_FRACTION = env_float("ALGAN_SHADOW_NEAR_FRACTION", 0.0)


def set_shadow_eps_relative(value):
    """Set the shadow acceptance floor as a fraction of scene scale (see
    ``SHADOW_EPS_RELATIVE``). Takes effect at the next render batch.
    """
    global SHADOW_EPS_RELATIVE
    SHADOW_EPS_RELATIVE = float(value)


def set_shadow_near_fraction(value):
    """Set the same-mesh share of the shadow acceptance floor (see
    ``SHADOW_NEAR_FRACTION``). Takes effect at the next render batch.
    """
    global SHADOW_NEAR_FRACTION
    SHADOW_NEAR_FRACTION = float(value)


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


# Empty-pixel fast path of the raster resolve: the prefilled frame buffer IS
# the committed state of an uncovered pixel, so the sparse route touches only
# covered pixels and empty screen regions cost nothing.  The sheet route is
# built on that identity, so this flag is one of its preconditions
# (analytic_raster_route_active): switching it off routes the batch to the
# classic supersampled wavefront.  Kill-switch / A-B hook.
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


# Fused permutation gather in the sparse compaction
# (sheet_compact_taichi.gather_fragment_arrays). The sorted fragment stream is
# six arrays produced from ONE permutation, and six index_select calls read
# that permutation six times -- ~106 bytes of traffic per fragment where one
# kernel moves 66. That is DESIGN_optimization_targets.md T5's own proposal.
#
# DEFAULT OFF, on the measurement rather than the argument. Built, and it is
# bit-identical (a gather copies bits, so fast_math has nothing to act on) and
# it is faster -- but only just: 13.7 ms -> 9.6 ms across both gather sites of
# a 3840x2160 frame, which is 4 ms of a 1.3 s frame. Against that it raises the
# frame's peak CUDA allocation by 50-160 MB, because the six outputs must all
# exist before the kernel writes the first, where the sequential form lets the
# allocator hand each one the block the previous stage just freed. This session
# reached here from an out-of-memory failure on this very scene; 4 ms is not
# worth 150 MB. Turn it on for a bandwidth-bound machine with VRAM to spare.
RASTER_FUSED_GATHER = env_flag("ALGAN_RASTER_FUSED_GATHER", False)


def set_raster_fused_gather(enabled):
    """Toggle the fused six-array fragment gather (see ``RASTER_FUSED_GATHER``).
    Takes effect at the next batch's emission.
    """
    global RASTER_FUSED_GATHER
    RASTER_FUSED_GATHER = bool(enabled)


# Covered-pixel-compacted resolve: the emission already knows exactly which
# pixels hold fragments, so the resolve launches one thread per COVERED pixel
# instead of one per screen pixel that early-outs, turning the resolve from
# O(screen pixels) into O(covered pixels).  Empty pixels keep the frame
# buffer's prefill untouched (so this requires RASTER_EMPTY_SKIP; an
# environment map is served by prefilling the map itself per pixel in
# render_chunk).  A precondition of the sheet route
# (analytic_raster_route_active): off routes the batch to the classic
# supersampled wavefront.
RASTER_COVERED_SHADE = env_flag("ALGAN_RASTER_COVERED_SHADE", True)


def set_raster_covered_shade(enabled):
    """Toggle the covered-pixel-compacted raster resolve (see
    ``RASTER_COVERED_SHADE``).
    """
    global RASTER_COVERED_SHADE
    RASTER_COVERED_SHADE = bool(enabled)


# Fully sparse primary-raster lifecycle: emit exact hit records for every
# candidate, sort/cull them in sparse hit space, and allocate every downstream
# structure for the unique covered pixels only.  It requires the
# retired-empty/background identity used by RASTER_EMPTY_SKIP and the
# covered-pixel resolve semantics; environment maps and in-kernel tonemapping
# are served on this route by the env prefill and the composite/uncovered
# finalize (DESIGN_sheet_resolve.md §5).  A precondition of the sheet route
# (analytic_raster_route_active): off routes the batch to the classic
# supersampled wavefront.
RASTER_SPARSE_COVERAGE = env_flag("ALGAN_RASTER_SPARSE_COVERAGE", True)


def set_raster_sparse_coverage(enabled):
    """Toggle the exact covered-pixel lifecycle of the hybrid raster path."""
    global RASTER_SPARSE_COVERAGE
    RASTER_SPARSE_COVERAGE = bool(enabled)


# The sheet resolve (DESIGN_sheet_resolve.md): the sparse emission's fragment
# stream is compacted into per-pixel SHEETS -- maximal same-surface regions,
# keyed (pixel, mesh, facing, depth band, conflict rank), carrying exact area
# and unioned sample masks -- and the resolve composites the few depth-sorted
# sheets per pixel instead of walking a raw fragment list. Shading is
# evaluated once per sheet and aggregation happens before the kernel with no
# lookahead budget.
#
# Default ON (the Phase-4 flip), and the ONLY resolve for analytic coverage:
# the fragment walk it replaced is deleted. It serves every batch the route
# accepts — analytic AA active for the geometry present, deterministic
# single-sample rendering, transparent background only without an env map —
# shadows, env maps and non-default tonemaps included. A batch the route does
# not accept, or this flag OFF, renders through the classic supersampled
# wavefront (DESIGN_sheet_resolve.md §5's stated fallback).
SHEET_RESOLVE = env_flag("ALGAN_SHEET_RESOLVE", True)


def set_sheet_resolve(enabled):
    """Toggle the sheet-compaction resolve (DESIGN_sheet_resolve.md)."""
    global SHEET_RESOLVE
    SHEET_RESOLVE = bool(enabled)


# Shading-discontinuity split in sheet compaction (sheets._shade_class). The
# resolve shades ONCE per sheet at its dominant fragment, which is licensed
# exactly where shading varies smoothly across the sheet -- and a hard crease
# (two flat-shaded faces of one solid meeting inside a pixel: same mesh id,
# same facing, no depth gap) violates that, so the fused sheet takes the
# dominant face's color for the whole pixel and every interior
# (non-silhouette) edge of a lit flat-shaded mesh renders winner-take-all
# jagged where the fragment walk used to blend per fragment. With this on,
# compaction keys triangle groups additionally by a SHADING CLASS -- a
# flat-shaded triangle's quantized unit face normal (declared, or the
# geometric fallback the shade kernel substitutes for all-zero vertex
# normals), class 0 for smooth-shaded triangles -- so crease faces become
# SIBLING sheets of one band (disjoint exact areas, additive compositing,
# DESIGN_sheet_resolve.md §4.4) and each shades with its own normal. That is
# the old per-fragment area-weighted blend across interior edges, paid only
# at crease pixels; smooth (diced PN) geometry compacts exactly as before.
#
# ON by default since the split shipped; the scenes carrying flat-shaded
# solids are baselined with it. Turning it OFF returns every lit crease edge
# to the winner-take-all staircase and is a re-baseline in its own right.
# What a band's sheets COMMIT does not depend on the flag either way -- §4.4
# gives the band one occlusion write -- so flipping it moves colour at crease
# pixels, never coverage.
SHEET_SHADE_SPLIT = env_flag("ALGAN_SHEET_SHADE_SPLIT", True)


def set_sheet_shade_split(enabled):
    """Toggle the crease shading-class split in sheet compaction (see
    ``SHEET_SHADE_SPLIT``). Takes effect at the next batch's emission.
    """
    global SHEET_SHADE_SPLIT
    SHEET_SHADE_SPLIT = bool(enabled)


# Kernel band reductions in the compaction
# (sheet_compact_taichi.sheet_band_reduce / mask_popcount). The mask passes
# were written one SAMPLE LANE at a time because torch has no way to say
# "reduce these eight bits at once": the union and the
# DESIGN_sheet_resolve.md §6.2 fusion detector cost one scatter_add_ per lane
# (eight passes over the fragment stream and eight over the sheet array) to
# learn whether each lane's count is 0, 1 or more, and the popcount cost eight
# shift/and/add passes to count at most eight bits. The exact-area sum walks
# the same stream and could not share it, because scatter_add_ wanted an f64
# copy of the whole fragment array first. One kernel now does all of it.
# Measured at 3840x2160: compact_sheets 445 -> 352 ms, and 27 MB less peak.
#
# Bit-identical. The mask passes are integer reductions, exact under any
# order, and the fusion detector stays order-independent because atomic_or's
# RETURN value tells a fragment whether a lane was already claimed. The area
# sum keeps its float64 accumulator (ss6.6.4) -- widened in a register off an
# f32 read, so the f64 copy of the fragment array is gone but the exactness is
# not -- and agrees with the torch scatter_add_ bitwise. See
# sheet_compact_taichi's module docstring for both arguments and
# benchmarks/_sheet_kernel_check.py for the checks.
SHEET_MASK_KERNEL = env_flag("ALGAN_SHEET_MASK_KERNEL", True)


def set_sheet_mask_kernel(enabled):
    """Toggle the kernel sample-mask reductions in sheet compaction (see
    ``SHEET_MASK_KERNEL``). Takes effect at the next batch's emission.
    """
    global SHEET_MASK_KERNEL
    SHEET_MASK_KERNEL = bool(enabled)


# Kernel conflict-rank scan in the compaction
# (sheet_compact_taichi.sheet_conflict_rank). A fragment's conflict rank --
# the largest, over the sample lanes it claims, of how many EARLIER fragments
# of its band claim the same lane -- keys a band's sub-bands when the fill
# rule's partition is violated (sheets.compact_sheets). Torch walked it lane
# by lane: eight cumsum passes over the stream, an index_select + maximum +
# two wheres each, and five live [n] arrays at the peak -- the compaction's
# one genuine remaining scan (RENDERER_WORK_QUEUE.md item 11;
# DESIGN_sheet_resolve.md §10.4). One kernel now walks each band forward once
# with the eight per-lane counters in registers, gathering msk[order[j]]
# itself instead of materializing the sorted+masked copy.
#
# Bit-identical, and trivially so: both arms are integer and visit the stream
# in the SAME order -- the kernel's serial band walk reads fragments exactly
# as the cumsums do -- so unlike SHEET_MASK_KERNEL above it needs no
# order-independence argument at all. The max=15 clamp stays in
# compact_sheets in both arms.
SHEET_RANK_KERNEL = env_flag("ALGAN_SHEET_RANK_KERNEL", True)


def set_sheet_rank_kernel(enabled):
    """Toggle the kernel conflict-rank scan in sheet compaction (see
    ``SHEET_RANK_KERNEL``). Takes effect at the next batch's emission.
    """
    global SHEET_RANK_KERNEL
    SHEET_RANK_KERNEL = bool(enabled)


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

# Let the RUN rule see FULL-MASK fragments that do not cover the whole pixel
# (DESIGN_mesh_identity.md ss6.3.2). v2 ss4.2 starts the run lookahead only when
# the first fragment's mask is PARTIAL, because an interior pixel is one
# full-mask fragment and must not pay for a scan. But a diced mesh's silhouette
# produces full-mask fragments too -- one triangle owning all eight sub-pixel
# samples while covering a fraction of the pixel's AREA -- and those never enter
# the run rule, never compute E, and are painted at 1.0 with their exact area
# sitting unread in ``frag_cov``. On a fine Sphere that is 52% of the silhouette
# pixels and the single largest contributor to its coverage error.
#
# An interior full-mask fragment has ``cov`` within float dust of 1, so the gate
# relaxes to "partial mask OR (full mask AND cov < 1 - dust)": the hot path is
# untouched and exactly the silhouette pixels are admitted. A full-union
# sheet then takes ``corr = min(area, 1)`` (Q == 1 there) instead of
# short-circuiting to 1, with the same dust band keeping a genuine interior
# tiling bit-identical.
#
# Scoped to the RUN, not the fragment: a full-mask fragment owns every sample,
# so by the fill rule the rest of its sheet in that pixel owns none -- they are
# empty-mask area DONORS whose area is real, and only the run's E counts them.
# Measured both ways on a fine Sphere: fragment scope reaches 0.0255, run scope
# 0.0060. On the two flat solids (no donors) they coincide.
#
# Measured, mean coverage error over silhouette pixels, --res md CPU
# (benchmarks/_aa_run_gate_check.py's |cF-E| column, replayed before it was
# built): flat quad 0.0020 -> 0.0000 (EXACT), cylinder 0.0260 -> 0.0030 (-88%),
# cylinder(256,2) 0.0211 -> 0.0030 (-86%), sphere(192,96) 0.0383 -> 0.0060
# (-84%). Worth more than doubling the sample count, at no sample cost and no
# interior work. The two flat solids barely move (cube 0.0250 -> 0.0214) because
# their error is the far-sheet re-claim, which needs mesh-level identity
# instead -- the two halves of ss6.3 have different owners.
#
# Carried as ``aa_grp = 2`` rather than a new kernel argument: every
# ``ti.static(aa_grp)`` test in the kernels is a truthiness test. Subordinate to
# ANALYTIC_AA_SEAM and to the run rule (aa_tri 3 or 4).
ANALYTIC_AA_RUN_FULL = env_flag("ALGAN_ANALYTIC_AA_RUN_FULL", False)


# THE ONE-MESH RULE (DESIGN_mesh_identity.md ss6.6). Where every fragment in a
# pixel is an OPAQUE triangle of ONE surface, the pixel's coverage is that
# mesh's NEAR SHEET's exact area and nothing else -- both sheets project to the
# same silhouette, so the far sheet must not add coverage on top of it.
#
# What it fixes. The run rule's ``corr < 1`` scales the OCCLUSION write as well
# as the claim, so the samples the near sheet owns keep a residual transmittance
# standing for the part of the pixel the sheet does not cover. That residue lies
# OUTSIDE the mesh, but it carries no position, so when the far sheet of the
# same solid arrives owning the same samples it claims the residue as though it
# were background showing through -- uncorrected, because svis is no longer
# uniform and its own run cannot engage. Measured on one Cylinder pixel: near
# sheet claims 0.2396 (exact, corr 0.9583), far sheet adds 0.0104, pixel lands
# on 0.2500 = 2/8 against a true 0.2394.
#
# This is what mob-declared identity (MESH_ID) was built to enable and what no
# consumer read until now: "these two sheets are ONE mesh" is not a geometric
# question and cannot be answered by an epsilon.
#
# Restricted to ONE-MESH pixels because a facing change across TWO meshes is an
# ordinary occlusion, and to OPAQUE ones because a translucent solid's far sheet
# is genuinely visible through its near sheet. The host marks the pixels (it has
# the CSR to do it as a segment reduction) and carries the flag in a spare
# frag_msk bit, so no kernel argument changes; carried as aa_grp = 3, and every
# ti.static(aa_grp) test in the kernels is a truthiness test.
#
# Measured on benchmarks/_aa_line_check's own thin Cylinder at 33 deg -- the
# geometry its ink-wobble metric actually reads -- mean coverage error over
# silhouette pixels 0.0299 -> 0.0064, which is 79% of the error there. Note that
# is a DIFFERENT mechanism from the one ss6.3.2 chased: on that geometry the
# relaxed run gate is worth only 19%, which is why it moved ink wobble by
# nothing.
#
# IMPLIES ANALYTIC_AA_RUN_FULL, and that implication is wired in exactly one
# place: raster_pipeline._aa_group returns aa_grp = 3, and every reader asks
# _aa_run_full(), which accepts 2 or 3. It was once wired only on the kernel
# side, so the relaxed semantics ran over fragment lists whose area donors the
# truncation had already discarded; that cost a flat quad -8% of ink wobble
# where correct wiring gives -63%.
# tests/unit_tests/test_analytic_aa_gates.py pins it.
#
# DEFAULT ON. Measured on CUDA (DESIGN_mesh_identity.md ss6.6.1-3): coverage error
# against an exact analytic reference falls 70-100% on all eleven harness cases,
# and on-lattice -- the share of silhouette pixels landing on a multiple of 1/8,
# i.e. whether coverage is still sample-quantized -- collapses from 8-91% to
# 0-1.6%. Ink wobble: flat quad 0.0139 -> 0.0052, Cylinder 0.0568 -> 0.0124,
# Cylinder(256,2) 0.0765 -> 0.0429, bezier Line unchanged.
#
# What it costs, so the trade is on the record: ~2-5% render time (the honest
# figure is 1.038x, from a 35 s shadowed scene; the small scenes sit on the
# fixed-overhead floor).
#
# IT DOES NOT COST THE INTERIOR NOTCHES, though two earlier revisions of this
# comment and of ss6.6.2 said it did. Measured by gate (--notch-probe): the
# relaxed run gate this rule IMPLIES takes a pathological 0.045-radius rod diced
# 256x from 50 to 239 notched interior pixels, and the cap then adds 14 -- ~92%
# of the residue is ss6.3.2's gate, not the ceiling. Confirmed per pixel by
# replaying a notched pixel's own fragments with only the clip disabled: it
# recovers those same 14 of 253, and the ceiling is IDENTICAL on notched and
# clean pixels. Do not debug frag_cap for this; see ss6.3.2.
#
# Its occlusion-side defect was real and is fixed: the fragment walk's cap used
# to clip eff while the occlusion write kept the uncapped dens; the sheet
# resolve scales a capped sheet's occlusion with its claim as part of the claim
# arithmetic itself (DESIGN_sheet_resolve.md), so there is no separate toggle.
#
# DETERMINISM IS A REQUIREMENT OF THIS RULE, NOT AN INCIDENTAL PROPERTY. The
# per-pixel ceiling feeds a threshold in the resolve, so the reduction that builds
# it must be reproducible or borderline fragments flip between runs. It is
# accumulated in float64 and rounded to float32 for exactly that reason -- see the
# comment at the reduction in raster_pipeline.prepare_sparse_raster_coverage and
# ss6.6.4. If you change how the ceiling is computed, A/A the render (twice, same
# settings, compare) before trusting any baseline.
ANALYTIC_AA_ONE_MESH = env_flag("ALGAN_ANALYTIC_AA_ONE_MESH", True)


# Build the BEZIER-CIRCUIT STBVH with the median-split instance ordering the
# triangle tree already uses, instead of Morton (DESIGN_mesh_identity.md ss3.4).
# A space-filling curve is cheap but packs spatially distant instances into the
# same balanced subtree at its discontinuities, leaving loose internal boxes; a
# top-down longest-axis median split gives tighter boxes and ~20-25% fewer
# traversal steps. Both are pure REORDERINGS -- same instances, same opaque
# flags, same tree shape -- so no traversal code changes and the set of
# intersections found is unchanged.
#
# Bezier was pinned to Morton because a circuit's seam de-duplication is
# discovery-order sensitive, so the reorder was expected to move output at the
# epsilon level. It does not: measured byte-identical against a zero
# run-to-run floor (``_order_window_check.py``, and independently every one of
# 51,200 rays returns the same primitive under both orderings).
#
# The inherited "~20-25% fewer traversal steps" is now MEASURED rather than
# assumed (``_bvh_steps.py``, DESIGN_mesh_identity_open.md ssE/ssF): 3.300 ->
# 2.302 sibling-block tests per ray on 35 circuits plus Text and Tex, a 30%
# reduction, and 3.159 -> 2.219 on incoherent rays, so it is not an artifact of
# primary-ray coherence. The same instrument reproduces the triangle tree's
# already-known 25% (7.944 -> 5.960), which is what says the instrument is
# measuring the tree rather than itself.
#
# READ THIS BEFORE CONCLUDING ANYTHING FROM THE FLAG. ``BVH_REFIT`` defaults
# ON, and ``_build_accel``'s refit branch ignores ``builder`` outright, so at
# shipped defaults NO STBVH is built for any geometry type and this flag --
# like ``ALGAN_BVH_BUILD`` -- changes nothing at all. It governs the tree you
# get with ``ALGAN_BVH_REFIT=0``, and that is the only configuration in which
# either the win above or any A/B of it exists.
BEZ_BVH_SPLIT = env_flag("ALGAN_BEZ_BVH_SPLIT", True)


# Weld a closed surface grid's u-seam and its collapsed poles into SHARED
# vertices instead of coincident duplicates (DESIGN_mesh_identity.md ss3.1).
#
# get_grid_to_triangle_indices builds two triangles per grid cell and never
# bridges column W-1 back to column 0, so a Sphere's wraparound is a genuine
# two-copy seam -- measured, the two columns differ by up to 1.7e-07 in f32 and
# are NOT bitwise equal, while every interior shared edge IS a bit-identical
# duplicate of the same gather. That asymmetry is the point: a watertight
# intersection test (ss3.2) fixes numerical ambiguity, and at the u-seam it
# would OPEN a crack rather than close one, because the gap is real geometry.
# The poles are collapsed degenerate fans, every point of the row mapping to the
# same position with 4e-08 of jitter.
#
# With this on, the wrap cell indexes column 0 and a pole row is one vertex,
# which also drops the (W-1) degenerate triangles each pole contributes.
#
# NOT what ss3.1 claimed. It said this "retires two authoring-side epsilon
# special-cases (the 1e-4 normal merge and the pole-normal salvage)". It does
# not: compute_grid_vertex_normals accumulates over the GRID, not over the
# welded triangle list, so column 0 still misses the wrap-around neighbourhood
# and a pole row still accumulates from sub-epsilon differences. Both fixups
# stay necessary and stay in place. Retiring them needs the normal accumulation
# to run on the welded topology, which this does not do.
#
# DEFAULT OFF -- but NOT for the reason this comment used to give. "Geometry
# moves, so all pixel baselines move" is measured FALSE: the welded vertices are
# coincident to 1.7e-07 and the dropped triangles have zero area, so the
# rasterizer cannot see the difference. benchmarks/_weld_check.py on CUDA,
# --res md, byte-identical on all three arms while the triangle count
# demonstrably drops:
#
#   plain (Sphere/Cylinder/Cone/Torus)   6668 -> 6572 tris   max|d| 0
#   Sphere + colour checkerboard         4096 -> 3968 tris   max|d| 0
#   Sphere + normal map                  4096 -> 3968 tris   max|d| 0
#
# The two textured arms are the point: the POLE weld changes the triangle list, so
# every per-vertex attribute including uv must go through the same indices, while
# the U-SEAM wrap deliberately does not weld (wrapping it would give the last cell
# column u = 0 where the texture needs u = 1). A checkerboard is used rather than a
# photo because a one-column uv error moves a hard edge and hides in a gradient.
#
# THE BLOCKER THAT HELD IT OFF IS CLOSED. It was that the weld is applied by
# get_grid_to_triangle_indices, which only the RENDER path calls, while the morph
# path built its triangles with grid_to_triangle_vertices
# (morph_conversions._grid_to_pn_soup) and knew nothing about the gate -- so a
# welded Sphere rendered one triangulation and morphed another. Both consumers now
# ask ``surface_weld_flags`` for the same grid, the way ss3.2 routed both
# intersection arms through _tri_hit, and the unit suite is green with the weld on.
#
# What remained was only that it MOVES a moving PN scene, because the dice level is
# chosen per patch per frame from projected size, so a different triangle list can
# land on a different level. Those baselines are regenerated (CUDA), the frames
# reviewed, and the gate is on.
WELD_SURFACE_SEAMS = env_flag("ALGAN_WELD_SURFACE_SEAMS", True)


def set_weld_surface_seams(enabled):
    """Toggle surface seam/pole welding (see ``WELD_SURFACE_SEAMS``).

    Takes effect on the next primitive build.
    """
    global WELD_SURFACE_SEAMS
    WELD_SURFACE_SEAMS = bool(enabled)


def set_bez_bvh_split(enabled):
    """Toggle median-split ordering for the bezier STBVH (see ``BEZ_BVH_SPLIT``).

    Takes effect on the next scene build.
    """
    global BEZ_BVH_SPLIT
    BEZ_BVH_SPLIT = bool(enabled)


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
ANALYTIC_AA_SECONDARY_MIN_ENERGY = env_float(
    "ALGAN_ANALYTIC_AA_SECONDARY_MIN_ENERGY", 0.12
)

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
# DEFAULT OFF, and off is not "roughness is ignored": with the lobe off a
# reflection carries only the share of itself a single ray can stand for
# (``wavefront_kernels_taichi._mirror_share``), which is what keeps a rough
# metal from drawing a sharp mirror image. On is the opt-in that spends rays
# to blur the reflection for real.
#
# It is off because four taps cannot integrate a wide lobe. The two ways to
# spend them both cost something visible, measured on a rough metal wall
# reflecting a small bright source: the Bayer rotation below resolves a glossy
# transition into K+1 levels, i.e. an ordered dither that CRAWLS as geometry
# moves under a screen-fixed pattern (the reason this shipped disabled, in a
# way that also disabled roughness entirely -- see DESIGN_analytic_aa.md ss20),
# and without it the same four taps land as four discrete ghost copies of the
# reflected image. ANALYTIC_AA_SECONDARY_SAMPLES = 8 halves the dither and
# doubles the ghosts; neither makes it clean. Turn this on for a still, or for
# a scene whose reflected content is low-contrast enough not to dither.
#
# See DESIGN_analytic_aa.md ss20.
GLOSSY_REFLECTION = env_flag("ALGAN_GLOSSY_REFLECTION", False)

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


# NESTED DIELECTRIC MEDIA for the deterministic tracer: a ray carries the stack
# of media it is inside (rs_sca columns 7+, see
# ``wavefront_kernels_taichi.IOR_STACK_DEPTH``) and each glass interface takes
# the RELATIVE index n_inside/n_outside instead of assuming air outside. This
# is what makes glass inside glass, a sphere inside a box, or a bubble in a
# liquid bend light correctly at the inner interfaces; without it every
# interface refracts as though the outside were air.
#
# DEFAULT OFF. The stack widens ``rs_sca`` (+4 f32 per ray), forces a cold
# recompile of both shade kernels' new template variants, and changes what a
# nested scene renders -- so it is an opt-in until its output has been lived
# with. Fresnel reflectance keeps the MATERIAL index even when this is on (the
# relative index reaches only Snell's law): ``_material_reflectance``'s
# dielectric branch is itself gated ``ior > 1 + 1e-4``, which a relative index
# below 1 would silently zero. See DESIGN_mesh_identity_open.md §H.
NESTED_IOR = env_flag("ALGAN_NESTED_IOR", False)


def set_nested_ior(enabled):
    """Toggle the nested-dielectric IOR stack (see ``NESTED_IOR``)."""
    global NESTED_IOR
    NESTED_IOR = bool(enabled)


def nested_ior_mode():
    """Live nested-IOR mode: 0 off, 1 media stack maintained.

    Read at call time (never imported by value) and returned as an int,
    because it reaches the shade/resolve kernels as a TEMPLATE value: each
    mode compiles its own kernel variant, so the offline cache cannot serve
    one mode's kernel for another (see ``glossy_reflection_mode``).
    """
    return 1 if NESTED_IOR else 0


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
    run_full=None,
    one_mesh=None,
):
    """Toggle analytic anti-aliasing (see ``ANALYTIC_AA``)."""
    global ANALYTIC_AA, ANALYTIC_AA_BEZ, ANALYTIC_AA_TRI, ANALYTIC_AA_SEAM
    global ANALYTIC_AA_SLIVER, ANALYTIC_AA_SECONDARY_SAMPLES, ANALYTIC_AA_EXACT
    global ANALYTIC_AA_RUN, ANALYTIC_AA_RUN_RULE, ANALYTIC_AA_RUN_FULL
    global ANALYTIC_AA_ONE_MESH
    if secondary is not None:
        ANALYTIC_AA_SECONDARY_SAMPLES = int(secondary)
    if exact is not None:
        ANALYTIC_AA_EXACT = bool(exact)
    if run is not None:
        ANALYTIC_AA_RUN = bool(run)
    if run_full is not None:
        ANALYTIC_AA_RUN_FULL = bool(run_full)
    if one_mesh is not None:
        ANALYTIC_AA_ONE_MESH = bool(one_mesh)
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
    (``ANALYTIC_AA_BEZ_WEDGE``, default on since 2026-08-13).
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


# --- what the level searches are allowed to stop resolving ------------------
# The level searches score the flat dice against the PN patch, which is itself
# only an approximation of the surface the author asked for -- accurate to
# ``geometry_tolerance`` in world units, by construction. Detail finer than that
# is not a feature of the surface, it is the PN patch's own error, and
# subdividing to resolve it buys nothing: measured on a cylinder patch, a strip
# of 31 microtriangles sits 0.000768 world units from the analytic cylinder
# where the uniform level-4 dice's 256 sit 0.000782 from it, both against a PN
# patch that is itself 0.000739 off.
#
# With this on, both criteria subtract the projected size of the surface's own
# accuracy from the deviation they measure (see ``_guarded_pixel_error``). The
# slack is a world-space length projected per sample, so it does nothing for
# distant geometry -- where a whole ``geometry_tolerance`` is a fraction of a
# pixel -- and does its work in the close-ups where the dice would otherwise
# chase sub-surface-accuracy wiggle. The guarantee becomes: the dice lands
# within ``render_tolerance`` of the logical surface, plus the accuracy of that
# surface itself, which is the bound the render already inherits from
# construction.
#
# Moves rendered output (coarser tessellation in close-ups).
# ALGAN_PN_GEOMETRY_SLACK=0 restores the strict PN-patch criterion.
PN_GEOMETRY_SLACK = env_flag("ALGAN_PN_GEOMETRY_SLACK", True)

# --- per-dimension dicing ---------------------------------------------------
# A patch's dice is (level_along, level_across): 2**along rows fanning from one
# corner, each cut into at most 2**across columns. Equal levels reproduce the
# uniform barycentric grid exactly, so this only ever *removes* microtriangles
# from a patch whose two directions need different detail -- anything developable
# (a cylinder's length, a cone's slant, an extruded profile), where the flat
# direction otherwise pays whatever the curved one costs.
#
# The across level is not inferred from the boundary; it is searched and the
# chosen pattern is measured by the same criterion as the isotropic one, so a
# patch can only be coarsened where its own dice still meets the tolerance.
#
# Moves rendered output (different microtriangles under the same tolerance).
# ALGAN_PN_ANISOTROPIC_DICE=0 restores the uniform per-patch grid.
PN_ANISOTROPIC_DICE = env_flag("ALGAN_PN_ANISOTROPIC_DICE", True)


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
    # 26 one_sided: 0.0 is two-sided lighting, what every mob got before the
    # built-in solids started declaring an outside (shading_taichi._MAT_ONE_SIDED).
    0.0,
]
# Material-property name -> (start slot, width) in the canonical block.
# ``one_sided`` (slot 26) is deliberately absent: it is declared by the mob's
# geometry, not by its material, and ``_pack_material`` writes it directly.
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
