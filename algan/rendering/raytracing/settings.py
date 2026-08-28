"""The renderer's feature toggles: the storage behind ``SETTINGS.raytracing``.

Every ray-tracing switch lives here as a module-level value with an environment
variable default, under **the same name** ``SETTINGS.raytracing`` exposes it by.
That section is the public, validated face of this module; the switches that
change what the image looks like are exposed on it directly, and the kernel and
performance switches through ``SETTINGS.raytracing.experimental``.

One name, deliberately. These used to be UPPER_CASE here and lowercase there,
joined by a hand-maintained table in ``algan/settings/raytracing_settings.py``,
and nine switches reached the engine with no way to set them because nobody
added their row. The field set is derived from this module now, so a switch
declared here is reachable by construction. The corollary is that a helper
function may not share a name with a field -- the later ``def`` would take the
name over and the field would vanish silently (see ``_shadowed_fields``).

**Read these live** -- ``rt_settings.x`` at call time, not ``from ... import X``
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
from algan.logging.logger import get_logger
from algan.rendering.raytracing.shading_taichi import _USER_PIPELINE_BASE
from algan.settings._startup import render_device

# Maximum number of ray bounces (mirror reflections / diffuse scatters).
max_bounces = 8
# Rays averaged per pixel. 1 renders with the exact deterministic kernel;
# > 1 switches to the Monte Carlo pathtracer (stochastic transparency,
# glossy reflections, optional diffuse indirect lighting).
samples_per_pixel = 1

# Policy for renderer/backend combinations that cannot honor authored scene
# features. "error" is the safe public default; "warn" and "ignore" are
# available for controlled migration and benchmarking.
unsupported_feature_policy = (
    env_str("ALGAN_UNSUPPORTED_FEATURE_POLICY", "error").strip().lower()
)
if unsupported_feature_policy not in {"error", "warn", "ignore"}:
    unsupported_feature_policy = "error"


def set_unsupported_feature_policy(policy):
    """Set unsupported-feature handling to ``error``, ``warn``, or ``ignore``."""
    normalized = str(policy).strip().lower()
    if normalized not in {"error", "warn", "ignore"}:
        raise UnsupportedFeatureError("policy must be 'error', 'warn', or 'ignore'")
    global unsupported_feature_policy
    unsupported_feature_policy = normalized


def report_unsupported_features(message):
    """Apply the configured policy to an unsupported render combination."""
    if unsupported_feature_policy == "ignore":
        return
    if unsupported_feature_policy == "warn":
        warnings.warn(message, UnsupportedFeatureWarning, stacklevel=3)
        return
    raise UnsupportedFeatureError(message)


# Decode authored color to linear light at the render boundary, do all shading
# and compositing arithmetic there, and apply the sRGB OETF at the byte write.
# This is what three.js does (a LinearSRGBColorSpace working space, then
# `colorspace_fragment` unconditionally at the end of the shader), and it is
# what makes lights additive: sRGB encoding is concave, so summing encoded
# values overshoots badly -- two lights that should land a white surface on byte
# 188 land it on 255 instead. Without it the arithmetic is provably
# display-referred: a swept light intensity fits byte/255 = 0.1009 + 1.0000*i
# with a max residual of 0.0011.
#
# Unlit flat content is untouched, because decode-then-encode with no arithmetic
# between is the identity. Set ALGAN_LINEAR_COLOR=0 to restore the previous
# display-referred pipeline for A/B; that arm is byte-identical to the tree
# before the working space landed. See LINEAR_COLOR_WORK.md.
linear_color_space = env_flag("ALGAN_LINEAR_COLOR", True)

# Base ambient coefficient: the constant fill every lighting model adds on top
# of its direct terms. The renderer always passes ``ambient_light_intensity``
# as 1, so the scale lives here -- without it a single point light washes
# surfaces out to white and unlit sides stop reading as unlit (the reference is
# a Three.js scene lit by one point light with no AmbientLight).
#
# The two numbers are the SAME fill in two unit systems, not two settings.
# 0.1 was chosen as a display-referred coefficient; carrying it unchanged into
# the linear working space would make the ambient nearly nine times brighter,
# because 0.1 of linear light encodes to byte 89 where 0.1 of an encoded value
# is byte 26. srgb_to_linear(0.1) = 0.01003, so 0.01 delivers what the old
# pipeline delivered -- the number changes because the units changed, not
# because the look was retuned. ``_ambient_strength()`` in
# ``shading_taichi`` and ``shaders/material_shaders`` picks the one that
# matches linear_color_space; both used to hold their own copy of the pair,
# which is two sources of truth for one look-defining constant.
#
# Folded into the kernels inside ``ti.static``, so a change takes effect for
# kernels compiled after it (CLAUDE.md's ti.static hazard); set the environment
# variable for a guaranteed one. Distinct from ``ambient_light``, which belongs
# to the unwired physical-mode Monte Carlo kernel and is inert.
ambient_strength = env_float("ALGAN_AMBIENT_STRENGTH", 0.1)
ambient_strength_linear = env_float("ALGAN_AMBIENT_STRENGTH_LINEAR", 0.01)

# Off by default: an authored color lands on the pixel it names. That is now
# true because the working space is linear and the OETF runs at the byte write
# (see linear_color_space above), so this default is the same choice three.js
# makes with NoToneMapping rather than a workaround for a missing conversion.
# The curve (Khronos PBR Neutral) reserves headroom by design: it maps linear
# 1.0 to 0.869, so even with the OETF applied afterwards an authored white
# renders 240 rather than 255. Measured over the six full-render scenes, 98.52%
# of color channels are already inside the display range and 1.45% are above
# it. Bloom runs *before* the tonemap on the unclamped HDR buffer, so over-range
# energy is a visible halo before anything clamps. Turn it on for a filmic look,
# accepting that every SDR value shifts. See TONEMAP_FINDINGS.md.
tonemapping = False
tonemap_exposure = 1.0
tonemap_method = "neutral"
# Tonemap in post-processing (composite writes linear HDR float) rather than
# in the composite kernel. This is the physically-correct order: bloom/glow
# and the supersample downsample run in linear HDR and tonemapping is applied
# last (Unity/Unreal do the same), so HDR highlights keep their chroma
# instead of clipping to white. It also makes the composite a linear blend
# that is identity for empty pixels (enabling the covered-pixel compaction).
# Costs a float32 frame buffer (4x the uint8 one), so fewer frames per batch.
# Env override for A/B and re-baselining.
post_process_tonemap = env_flag("ALGAN_POST_PROCESS_TONEMAP", True)

# Strength of diffuse indirect bounces in the Monte Carlo renderer: 0 keeps
# surfaces purely (vertex-shader) lit, > 0 scatters paths on diffuse hits
# with throughput ``albedo * strength`` for color bleeding.
indirect_bounce_strength = 0.0

# Radiance scale of explicit point lights in physical mode. The default of
# pi makes a white light produce roughly albedo-level Lambertian brightness.
light_intensity = 3.141592653589793
# Constant ambient term added per diffuse interaction in physical mode.
ambient_light = 0.0
# When True, the deterministic trace kernel is told which geometry types are
# actually present and skips the per-ray traversal of any type whose tree is
# just the empty placeholder (a launch-uniform branch, no divergence). Set
# False to force all three traversals -- used by the A/B benchmark to measure
# the gain in isolation.
gate_empty_traversals = True

# Wavefront traversal rollouts. Changes to sibling revalidation and child
# ordering are enabled by default after parity validation; the opaque paths
# remain opt-in until their scene classification and shading gates are proven.
wf_revalidate_pending = env_flag("ALGAN_WF_REVALIDATE_PENDING", False)
wf_near_first = env_flag("ALGAN_WF_NEAR_FIRST", False)
wf_opaque_closest = env_flag("ALGAN_WF_OPAQUE_CLOSEST", False)
wf_opaque_prepass = env_flag("ALGAN_WF_OPAQUE_PREPASS", False)

inplace_aa = env_flag("ALGAN_INPLACE_AA", False)
# Rays per wavefront screen tile. The wavefront holds per-ray state for every
# ray it processes at once (~(18 + 6*kbuf) floats/ray); processing the chunk in
# tiles of this many rays bounds that state so it fits at any resolution / chunk
# length (a single HD frame is ~2M rays). ~2M rays * ~168 B ~= 350 MB of state.
wavefront_tile_rays = env_int("ALGAN_WAVEFRONT_TILE", 1 << 21)
# Adaptive tile sizing: size wavefront tiles from the render pool's *actual*
# free bytes instead of the fixed wavefront_tile_rays. The static ~2M-ray
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
wavefront_tile_auto = env_flag("ALGAN_WAVEFRONT_TILE_AUTO", True) and not env_is_set(
    "ALGAN_WAVEFRONT_TILE"
)
# Fraction of the pool's free bytes the per-tile ray state may claim.  Every
# built-in per-slot/fixed allocation and ManualMemory's initial alignment are
# now accounted exactly, so the default can use the whole allowance.  Keep the
# override as an opt-in diagnostic/performance headroom control.
wavefront_tile_safety = env_float("ALGAN_WAVEFRONT_TILE_SAFETY", 1.0)
# Preferred lower bound and hard upper bound for auto tile size (rays). The
# runtime honors the floor when it fits, but deliberately goes below it when
# exact arena headroom requires a smaller tile; the cap bounds active-index
# buffers and launch size on very large pools.
wavefront_tile_min = env_int("ALGAN_WAVEFRONT_TILE_MIN", 1 << 18)
wavefront_tile_max = env_int("ALGAN_WAVEFRONT_TILE_MAX", 1 << 25)


def set_wavefront_tile_auto(enabled):
    """Toggle adaptive (pool-sized) wavefront tile sizing (see
    ``wavefront_tile_auto``). Off falls back to the fixed
    ``wavefront_tile_rays``.
    """
    global wavefront_tile_auto
    wavefront_tile_auto = bool(enabled)


# On the common non-splitting wavefront path (no refraction/custom scatter), a
# ray that leaves the active set can never become active again.  Compact the
# next iteration from the previous active indexes rather than scanning the
# entire tile-sized ray pool after every traverse/shade pass.  Deep transparent
# scenes benefit most as the active population shrinks over successive passes.
# Splitting paths retain the full-pool scan because a shade pass may activate a
# spare slot that was not in the previous active set.  Runtime-mutable for
# in-process A/B checks; the env var selects the startup default.
wf_compact_active_only = env_flag("ALGAN_WF_COMPACT_ACTIVE_ONLY", True)
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
refract_initial_pool_ratio = max(
    2,
    env_int(
        "ALGAN_WAVEFRONT_INITIAL_POOL_RATIO",
        env_int("ALGAN_WAVEFRONT_SPLIT", 2),
    ),
)
# Backwards-compatible name for code that imported the old setting. It now
# denotes the initial shared-pool ratio; it is no longer a per-pixel slot cap.
REFRACT_SPLIT_SLOTS = refract_initial_pool_ratio
# When True, the *deterministic* raytracer (samples_per_pixel == 1, non-physical)
# shades the core lit materials per fragment inside the trace kernel instead of
# baking per-vertex colors (Gouraud). Ignored by the Monte Carlo pathtracer.
fragment_shading = True
# Promote a mob whose color AND material params (reflectivity/roughness/index
# of refraction) are constant across the whole surface to a 1x1 texture at merge
# time, dropping its per-vertex ``tri_colors``/``tri_extra`` rows, instead of
# broadcasting the constant to every vertex. The shared texel buffer keeps one
# copy per mob (and, when the color is also constant across frames, one copy
# total) rather than [T, N, 3, 5] / [T, N, 15]. Only applied on the
# deterministic fragment-shading wavefront path -- the only path where a
# "constant color" mob genuinely has constant per-fragment color (vertex
# lighting bakes per-vertex variation, so a promoted mob would be wrong there).
# The trace kernels guard every per-vertex read with ``prim < array.shape[1]``,
# so the shrunk arrays are never indexed for a promoted prim and every other
# render path stays byte-identical. Sampling a 1x1 map reduces exactly to the
# stored constant, so a promoted render matches the per-vertex one to <=1 ULP
# (the barycentric sum ``w0+w1+w2`` is not exactly 1.0 in f32). Default on;
# ALGAN_PROMOTE_CONSTANTS=0 disables it (for A/B and validation).
promote_constants = env_flag("ALGAN_PROMOTE_CONSTANTS", True)

# Skip the up-front per-fragment shading-normal computation for UNLIT hits on
# the fragment-shading wavefront. An UNLIT material passes its color through
# unchanged (``_run_frag_pipeline`` ignores the shading normal for it), so
# computing the interpolated/normal-mapped normal for such a hit is wasted work.
# Reflective/refractive continuation recomputes its own normal on demand, so
# this is byte-identical. Compile-time template of the shade kernel (no runtime
# arg -- the shade kernel is already at Taichi's 64-arg ceiling); this is the
# speed-relevant core of the "Family A" material-field trim (skipping the
# normal work), decoupled from the memory-side array trimming.
# ALGAN_WF_SKIP_UNLIT_NORMAL=0 disables it (for A/B and validation).
wf_skip_unlit_normal = env_flag("ALGAN_WF_SKIP_UNLIT_NORMAL", True)

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
frag_pid_gate = env_flag("ALGAN_FRAG_PID_GATE", False)


def set_frag_pid_gate(enabled):
    """Toggle compile-time material-pipeline gating of the shade kernels (see
    ``frag_pid_gate``). Takes effect at the next render batch.
    """
    global frag_pid_gate
    frag_pid_gate = bool(enabled)


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
# forecasts a remaining-time saving above wf_gen_fused_min_win (the decision
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


wf_gen_fused = _parse_gen_fused_mode(env_str("ALGAN_WF_GEN_FUSED", "auto"))

# Fraction of wavefront render time the fused generation saves (the measured
# steady-state win; used only by the "auto" forecast).
wf_gen_fused_gain = env_float("ALGAN_WF_GEN_FUSED_GAIN", 0.082)
# Minimum forecasted saving (seconds of remaining render time * GAIN) before
# "auto" pays the fused variants' compile cost. The default covers the
# worst case observed on this project's hardware -- a cold offline cache,
# where the two extra instantiations cost ~25 s -- so a marginal render never
# loses time to the switch.
wf_gen_fused_min_win = env_float("ALGAN_WF_GEN_FUSED_MIN_WIN", 30.0)

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
sparse_discovery_safety = env_float("ALGAN_SPARSE_DISCOVERY_SAFETY", 1.25)


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
        * sparse_discovery_safety
        * max(1, int(num_frames))
    )


def set_wf_gen_fused(mode):
    """Set fused primary-ray generation on the deterministic wavefront:
    ``True``/``False`` force it on/off; ``"auto"`` (default) starts unfused
    for fast startup and enables it mid-render when the forecasted remaining
    render time justifies compiling the fused kernel variants. All modes are
    byte-identical (see ``wf_gen_fused``).
    """
    global wf_gen_fused
    wf_gen_fused = _parse_gen_fused_mode(mode)


def wf_gen_fused_active():
    """Live effective value of the fused-generation toggle (resolves
    ``"auto"`` to the adaptive decision).
    """
    if wf_gen_fused == "auto":
        return _WF_GEN_FUSED_ON
    return bool(wf_gen_fused)


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
        wf_gen_fused != "auto"
        or _WF_GEN_FUSED_ON
        or _WF_GEN_FUSED_BATCHES < 2
        or frames <= 0
        or seconds <= 0.0
        or frames_remaining <= 0
    ):
        return False
    projected_win = frames_remaining * (seconds / frames) * wf_gen_fused_gain
    if projected_win <= wf_gen_fused_min_win:
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
wf_mem_trim = env_flag("ALGAN_WF_MEM_TRIM", False)

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
bvh_refit = env_flag("ALGAN_BVH_REFIT", True)


def set_bvh_refit(enabled):
    """Toggle the shared-topology binned-SAH refit BVH (see ``bvh_refit``).
    Takes effect at the next batch's scene merge.
    """
    global bvh_refit
    bvh_refit = bool(enabled)


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
bvh_defer = env_flag("ALGAN_BVH_DEFER", True)


def set_bvh_defer(enabled):
    """Toggle deferred (on-demand) STBVH builds for batches that provably do
    not traverse them (see ``bvh_defer``). Takes effect at the next batch's
    scene merge.
    """
    global bvh_defer
    bvh_defer = bool(enabled)


# Collapse temporally-constant merged tables (materials, normals, colors,
# per-vertex extras, UV tables and the 2-D edge tables) to a single frame at
# merge time; every consumer reads their time axis as ``f % shape[0]``. The
# rendered pixels are unchanged for a given batch window, but the collapse
# shrinks the merged scene the arena planner measures, so batch windows can
# differ from an uncollapsed run on memory-tight scenes (re-windowed output
# differs at the epsilon level, the same class as any window change).
# ALGAN_MERGE_DEDUP_TIME=0 restores the full time bands (byte-level A/B
# against pre-collapse baselines).
merge_dedup_time = env_flag("ALGAN_MERGE_DEDUP_TIME", True)


# Extend the merge-time collapse to the per-frame GEOMETRY the list above
# deliberately skipped: ``tri_pos`` / ``tri_obj`` / ``tri_closed`` and the
# per-frame bounds/opacity tables that feed the BVH builds and the raster
# host tables. "Rigid motion lives in tri_pos" is a rationale about the
# moving case that forfeited the static case, where the equality probe is
# one pass and the saving is (T-1)/T of the array
# (DESIGN_renderer_structural_candidates.md item 1). Collapsed bounds also
# reach the BVH builders at ``Tc == 1``, waking their static branches (one
# instance spanning all frames -- ``build_stbvh``/``build_refit_bvh`` both
# accept it) instead of building per-frame structure over byte-identical
# boxes. Requires merge_dedup_time; ALGAN_MERGE_DEDUP_GEOMETRY=0 restores
# the dense tables (byte-level A/B).
merge_dedup_geometry = env_flag("ALGAN_MERGE_DEDUP_GEOMETRY", True)


def set_merge_dedup_geometry(enabled):
    """Toggle the merge-time collapse of temporally-constant geometry tables
    (see ``merge_dedup_geometry``). Takes effect at the next batch's merge.
    """
    global merge_dedup_geometry
    merge_dedup_geometry = bool(enabled)


# Texture banks with a real per-map time length. The shared flat texel buffer
# used to carry one leading time axis for ALL maps, unified to the batch
# maximum at assembly -- so one animated map re-expanded every static map (and
# the environment map) to T copies, and a static image was still stored once
# per materialized frame. With this on, each map's frames are flattened along
# the texel axis (frame f of a map at (offset, w, h) starts at
# ``offset + (f % t) * w * h``) and the map's time length ``t`` rides in the
# texture meta (cols 10-12), so the assembled buffer's leading axis is always
# 1 and every map keeps its own length. Byte-identical: the sampler reads the
# same texel values through ``(f % t)`` that it read through the buffer's
# ``f % shape[0]``. ALGAN_TEXTURE_TIME_FLAT=0 restores the shared time axis.
texture_time_flat = env_flag("ALGAN_TEXTURE_TIME_FLAT", True)


def set_texture_time_flat(enabled):
    """Toggle per-map texture time lengths (see ``texture_time_flat``).
    Takes effect at the next batch's scene merge.
    """
    global texture_time_flat
    texture_time_flat = bool(enabled)


# Content-deduplicate the shared texel buffer: a map whose processed texels
# (post decode/pad/flatten) equal an already-appended map's reuses that map's
# placement instead of appending a second copy. Every textured primitive is a
# singleton collection, so N mobs sharing one image used to store the image N
# times (DESIGN_renderer_structural_candidates.md item 5). Byte-identical by
# construction: equality is exact (``torch.equal`` after a shape prefilter),
# and two prims reading one placement read the same texels they read from two.
# ALGAN_TEXTURE_CONTENT_DEDUP=0 restores per-map appends.
texture_content_dedup = env_flag("ALGAN_TEXTURE_CONTENT_DEDUP", True)


def set_texture_content_dedup(enabled):
    """Toggle texture-bank content dedup (see ``texture_content_dedup``).
    Takes effect at the next batch's scene merge.
    """
    global texture_content_dedup
    texture_content_dedup = bool(enabled)


# Collapse a temporally-constant color-texture window before the premultiply
# / wrap-pad / decode / merge chain runs over it (Surface.get_render_primitives).
# The timeline materializes a wide attribute's window dense -- one image per
# frame whether or not anything edited it -- and every downstream copy used to
# be made per frame. When the window's frames and the surface's opacity are
# byte-identical across the batch, one frame carries the batch (every consumer
# reads texture time as ``f % shape[0]``). ALGAN_TEXTURE_WINDOW_COLLAPSE=0
# restores the dense chain (byte-level A/B).
texture_window_collapse = env_flag("ALGAN_TEXTURE_WINDOW_COLLAPSE", True)


def set_texture_window_collapse(enabled):
    """Toggle the static color-texture window collapse (see
    ``texture_window_collapse``). Takes effect at the next primitive build.
    """
    global texture_window_collapse
    texture_window_collapse = bool(enabled)


# Apply the mob's animated opacity to a color texture IN THE SAMPLER instead
# of premultiplying the map on the host. The premultiply
# (``Color.mult_opacity`` in ``Surface.get_render_primitives`` /
# ``TriangleMesh.get_render_primitives``) scales only the map's coverage
# channel, but it welds the (per-frame) opacity into the (usually static)
# texels -- so a plain fade of a static image voids texture_window_collapse
# and rebuilds, re-decodes and re-uploads the full map once per frame
# (DESIGN_renderer_structural_candidates.md item 5). With this on, the
# primitive carries the opacity as per-frame scalars (``texture_opacity``),
# the merge stores them as a tiny per-map region inside the shared texel
# bank (tex-meta cols 13-14: row offset / frame count -- data, not a new
# kernel argument: the resolve kernel is at Taichi's runtime-argument
# ceiling), and the sampler multiplies the sampled coverage by the frame's
# scalar. The collapse then keys on texel constancy alone.
#
# NOT byte-identical across the flip: the multiply moves from before the
# bilinear filter (per texel, on the host) to after it (per sample, in the
# kernel), which reorders f32 rounding by up to an ulp -- the same class of
# qualified exception as ALGAN_WIDE_ATTR_RENDER_DEVICE. With opacity == 1
# (no fade anywhere) the multiply is exact and the flip IS byte-identical.
# Requires texture_time_flat and is disabled under the legacy wf_textured
# path (which consumes premultiplied maps); see
# ``texture_opacity_in_kernel_active``. ALGAN_TEXTURE_OPACITY_IN_KERNEL=0
# restores the host premultiply byte-identically.
texture_opacity_in_kernel = env_flag("ALGAN_TEXTURE_OPACITY_IN_KERNEL", True)


def set_texture_opacity_in_kernel(enabled):
    """Toggle the in-sampler texture opacity multiply (see
    ``texture_opacity_in_kernel``). Takes effect at the next primitive build.
    """
    global texture_opacity_in_kernel
    texture_opacity_in_kernel = bool(enabled)


def texture_opacity_in_kernel_active():
    """Whether primitive builds should hand opacity to the sampler.

    One predicate for every decision point (Surface / TriangleMesh builds,
    the estimators) so a build cannot half-flip. The merge itself keys off
    the PRIMITIVE (``texture_opacity is not None``), so a setting change
    between a build and its merge stays coherent. Requires texture_time_flat
    (the opacity region rides the flattened bank's row addressing) and is
    off under wf_textured, whose legacy bank builder consumes premultiplied
    maps.
    """
    return texture_opacity_in_kernel and texture_time_flat and not wf_textured


# Store u8-provenance color maps as RGBA bytes instead of five f32 channels:
# x5 fewer bank bytes on the largest array of any textured merge
# (DESIGN_renderer_structural_candidates.md item 5). A map qualifies when the
# authoring side proved every texel is exactly k/255 with zero glow
# (``texture_u8_ok`` -- checked ONCE at authoring, never at the merge, which
# must not add device syncs on the prefetch worker) and its window arrived
# collapsed to one frame (an interpolating window's in-between texels are not
# k/255). Bytes are bit-packed into f32 lanes of the SAME shared bank (one
# RGBA texel per lane, ``ti.bit_cast`` in the sampler) and decoded through a
# per-map 256-entry LUT scattered from the map's OWN direct decode -- the
# exact tensor the f32 arm would have stored -- so the sampler consumes that
# arm's own bits. The one residue: torch-CPU's SIMD body and scalar tail can
# decode the SAME byte to bit patterns one ulp apart within one tensor, so
# on such (straddling) bytes the f32 arm itself stores two patterns and the
# LUT necessarily picks one; <= 1 ulp in linear light, CPU-only, and never
# observed to move a rendered output byte (benchmarks/_texture_opacity_ab.py
# asserts frame bytes). Meta col 15 carries the LUT base row (-1 = plain f32
# map). Requires the in-kernel opacity multiply (a premultiplied map is not
# k/255). ALGAN_TEXTURE_U8_STORAGE=0 restores f32 storage.
texture_u8_storage = env_flag("ALGAN_TEXTURE_U8_STORAGE", True)


def set_texture_u8_storage(enabled):
    """Toggle u8 color-map storage (see ``texture_u8_storage``). Takes
    effect at the next batch's scene merge.
    """
    global texture_u8_storage
    texture_u8_storage = bool(enabled)


# Describe an animated color texture's frame window as ENDPOINT maps plus
# per-frame interpolation weights read off the timeline, instead of
# materializing one full image per frame (stage 4 of the texture line;
# DESIGN_optimization_targets.md). The timeline's conservative gate
# (``AnimationTimeline._describe_segment_windows``) accepts a window only
# when every event touching the map's rows is a plain recorded assignment
# (``Mob._apply_change``, marked ``_algan_replay_is_plain_lerp``) with
# non-overlapping replay windows and no active updater depends on the mob;
# anything else falls back to the dense per-frame materialization
# byte-identically. Accepted windows upload K endpoint images (authored
# texels, so u8-eligible) plus a tiny per-frame (i0, i1, w) region of the
# bank (meta cols 16-17), and the sampler lerps the two endpoint texels in
# AUTHORED space before the linear-light decode -- the same order the dense
# path applies them (timeline lerp, then the merge's decode). The lerp's
# arithmetic is re-associated ((E1 - E0) * w against the dense change * w)
# and the decode runs in-kernel, so the flip is a qualified exception like
# ALGAN_WIDE_ATTR_RENDER_DEVICE: bounded by the render suites' tolerance,
# not byte-identical (benchmarks/_texture_lerp_ab.py asserts the bound).
# Requires the in-kernel opacity multiply (see ``texture_time_lerp_active``).
# ALGAN_TEXTURE_TIME_LERP=0 restores dense windows.
texture_time_lerp = env_flag("ALGAN_TEXTURE_TIME_LERP", True)


def set_texture_time_lerp(enabled):
    """Toggle in-kernel texture time interpolation (see
    ``texture_time_lerp``). Takes effect at the next frame batch.
    """
    global texture_time_lerp
    texture_time_lerp = bool(enabled)


def texture_time_lerp_active():
    """Whether frame batches may describe texture windows as segments.

    One predicate for the timeline gate and the estimators. The primitive
    build and the merge key off the DESCRIPTION instead (a stashed segment
    window / ``texture_lerp`` on the primitive), so a setting change
    mid-batch stays coherent: the batch finishes in whichever mode its
    materialization ran. Requires the in-kernel opacity multiply -- the
    legacy host premultiply folds the animated opacity into the texels,
    which endpoint maps cannot represent.
    """
    return texture_time_lerp and texture_opacity_in_kernel_active()


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
# pixels move on the overlapping pack and mesh_id=0 is the side that paints
# MORE, while the non-overlapping control moves zero. Small, but it is the win
# this was said to be missing.
#
# Flipping also moves the fast-suite render by up to 49 channel values at solid
# edges, so BOTH device baseline sets have to be regenerated and
# expected_outputs_cuda/ needs a CUDA machine. DESIGN_mesh_identity.md 3.5, 4.5.
mesh_id = env_flag("ALGAN_MESH_ID", True)


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
# mesh_id is off --
# a per-triangle surface id makes every run one fragment, so the facing bit
# groups nothing. With mesh_id=1 it does change the render, which is the
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
# flag and not at all from mesh_id.
# DESIGN_mesh_identity.md 3.7 and 6.5.
polyhedron_winding = env_flag("ALGAN_POLYHEDRON_WINDING", True)


def set_polyhedron_winding(enabled):
    """Toggle outward face orientation for closed polyhedra (see
    ``polyhedron_winding``). Takes effect for the next ``Polyhedron`` built.
    """
    global polyhedron_winding
    polyhedron_winding = bool(enabled)


def set_mesh_id(enabled):
    """Toggle mob-declared surface identity (see ``mesh_id``). Takes effect at
    the next batch's primitive build.
    """
    global mesh_id
    mesh_id = bool(enabled)


def set_merge_dedup_time(enabled):
    """Toggle the merge-time collapse of temporally-constant tables (see
    ``merge_dedup_time``). Takes effect at the next batch's scene merge.
    """
    global merge_dedup_time
    merge_dedup_time = bool(enabled)


# Opaque any-hit shadow early-out. The deterministic shadow query is an
# ordered closest-hit march that restarts a full two-tree (triangle + Bezier)
# traversal per peeled surface; but any interval-opaque blocker (main-tree leaf flag:
# classic ``leaf_tspan`` bit 31 / refit link bit 30) forces the final
# occlusion to exactly 1.0 no matter what lies in front of it. When on, the
# shadow query first runs a cheap unordered any-hit walk over just the
# opaque-flagged leaves and returns full occlusion on the first hit; batches
# that provably contain no translucent geometry skip the march entirely (a
# miss then proves the ray lit). Not strictly byte-identical in two corner
# cases the march itself gets wrong (an opaque edge hit seam-merged into a
# coincident translucent edge within depth_tie_epsilon, and an opaque
# blocker past max_surfaces_per_ray peels); the any-hit's answer is the
# physically correct one in both.
#
# QUALIFIED 2026-08-26, and deliberately NOT the default. Correctness: all
# three modes are byte-identical on both purpose-built corner scenes (each
# proven to reach its case) and on materials_and_lighting, on a CPU box and
# on a Tesla T4 (benchmarks/_shadow_anyhit_check.py; the structural round in
# DESIGN_optimization_targets.md). Performance is why the default stays off:
# on the nn UHD benchmark (a batch with translucent geometry, so mode 2)
# the flip measured 29.5 s -> 34.2 s end to end -- raster_shadow_trace
# 3.8 -> 6.6 s because the deferred any-hit pre-pass pays a second full
# traversal on the miss-dominated rays, and wavefront_shade 6.3 -> 8.2 s
# from the wider mode-2 kernel variant -- while the shadowed static-gallery
# scene measured neutral. Flip it per render for translucent-stack or
# proven-all-opaque (mode 3) scenes; a smarter default would engage the
# any-hit only where mode 3 applies. ALGAN_SHADOW_ANYHIT=1 opts in.
#
# ALGAN_SHADOW_ANYHIT=gather selects the gather-march instead: the same
# ordered shadow peel rebuilt on the kbuf gather (_collect_hits), so a
# k-surface translucent stack costs ceil((k+1)/kbuf) traversals instead of
# k+1 while all-opaque rays stay at one. Valid for any batch (the drain
# evaluates translucent attenuation exactly like the march); shares the
# march's output up to the seam-merge corner the camera peel also has.
shadow_anyhit = (
    "gather"
    if env_str("ALGAN_SHADOW_ANYHIT", "0").strip().lower() == "gather"
    else env_flag("ALGAN_SHADOW_ANYHIT", False)
)


def set_shadow_anyhit(enabled):
    """Select the shadow-query early-out mode (see ``shadow_anyhit``).

    ``True`` enables the opaque any-hit walks, the string ``"gather"`` the
    kbuf gather-march, ``False`` the classic ordered march. Takes effect at
    the next render batch.
    """
    global shadow_anyhit
    if isinstance(enabled, str) and enabled.strip().lower() == "gather":
        shadow_anyhit = "gather"
    else:
        shadow_anyhit = bool(enabled)


# Colored shadow payloads. The deterministic shadow query's visibility value
# is an RGB triple end to end; with this on, a transmissive surface TINTS the
# light it passes (by its albedo, matching what the bounce loop does to its
# transmitted share) and absorbs Beer-Lambert over the interior chord of any
# solid it crosses, instead of passing one achromatic fraction -- measured
# against a Three.js path tracer, green glass cast a grey shadow before this
# (renderer_audit REPORT.md ss4.10). DEFAULT ON.
#
# This gates only the tinting and the absorption, NOT the payload width: with
# it off every channel carries today's scalar value unchanged, so renders are
# byte-identical while the plumbing stays exercised. Every kernel use sits
# behind ti.static, which resolves at COMPILE time -- flipping this
# mid-process recompiles nothing, so the two arms must be separate processes.
# Declaring the variable import-time is what makes that honest: a warm daemon
# refuses a client whose value differs rather than serving kernels compiled
# for the other arm.
rgb_shadow_tint = env_flag("ALGAN_RGB_SHADOW_TINT", True)


# Watertight (Woop-Benthin-Wald) ray/triangle intersection, replacing the
# dilated Moller-Trumbore test and the matched epsilon pair that patches its
# cracks. The full derivation, the CUDA FMA hazard it had to survive, and the
# measurements live beside the kernel it gates
# (``raytrace_kernels_taichi._tri_hit``); this is only its storage, so that the
# switch is reachable as SETTINGS.raytracing.experimental.watertight_tri
# instead of an environment variable that has to precede the import.
# DEFAULT ON.
watertight_tri = env_flag("ALGAN_WATERTIGHT_TRI", True)


def set_watertight_tri(enabled):
    """Toggle the watertight ray/triangle intersection (see ``watertight_tri``).

    The gate compiles into the kernels, so this takes effect for kernels
    compiled AFTER the call -- existing variants are reused unchanged. For a
    guaranteed switch, set ``ALGAN_WATERTIGHT_TRI`` before importing algan, or
    run each arm in its own process. (The Taichi *offline* cache is not a
    hazard here: it keys on the compiled IR, so each arm has its own entry.)
    """
    global watertight_tri
    watertight_tri = bool(enabled)


def set_rgb_shadow_tint(enabled):
    """Toggle colored shadow tinting/absorption (see ``rgb_shadow_tint``).

    Because the gate compiles into the kernels, this takes effect for kernels
    compiled AFTER the call -- existing variants are reused. For a guaranteed
    switch, set ``ALGAN_RGB_SHADOW_TINT`` before importing algan or run the
    other arm in its own process.
    """
    global rgb_shadow_tint
    rgb_shadow_tint = bool(enabled)


# Self-shadow rejection by identity (DESIGN_mesh_identity_open.md ssI). A
# shadow ray currently rejects its own surface with min_hit_distance plus a
# normal offset of 10 * min_hit_distance -- absolute world-space constants
# applied to EVERY hit, so a small object resting on a plane loses its contact
# shadow within 1e-3 of the contact and grazing light on small geometry
# produces acne. On the sheet route's shadow queue the event's source surface
# id is available (packed into ``event_msk`` above the material pipeline id),
# so the acceptance test becomes
#
#     accept = (t < max_t) and (hit_mesh != src_mesh ? t > 0 : t > min_hit_distance)
#
# and the cross-mesh threshold goes to zero while self-rejection stays exactly
# as safe. The rejection is per hit -- "same mesh AND near-zero t", never
# "same mesh": a concave solid legitimately shadows itself. Events without a
# usable source id (bezier-originated, or ids that do not fit the packing) and
# every path outside the sheet route's shadow queue keep today's epsilon.
# DEFAULT ON.
shadow_identity_reject = env_flag("ALGAN_SHADOW_IDENTITY_REJECT", True)


def set_shadow_identity_reject(enabled):
    """Toggle self-shadow rejection by identity (see
    ``shadow_identity_reject``). Takes effect at the next render batch.
    """
    global shadow_identity_reject
    shadow_identity_reject = bool(enabled)


# Shadow-terminator offset for diced / smooth-shaded surfaces (Hanika, "A
# Microfacet-Based Shadow Terminator", Ray Tracing Gems II ch. 4). A PN patch
# or any smooth-shaded mesh reaches the renderer as FLAT triangles carrying a
# smooth per-vertex normal field, and every shadow ray starts from the FACE
# normal's fixed lift (``10 * min_hit_distance`` in ``raster_shadow_trace``).
# The facet is a chord BELOW the smooth surface it approximates, so
# neighbouring facets rise above the plane the origin was lifted from: near
# the terminator the shadow ray leaves almost tangentially and strikes a
# neighbouring facet a long way away -- acne no acceptance epsilon can reject
# (RENDERER_WORK_QUEUE.md item 20). With this on, the shadow-event build
# displaces the origin onto the smooth surface implied by the three vertex
# normals, by an amount derived from the hit's barycentrics:
#
#     d_i   = min(0, (p - p_i) . n_i)          for i in 0,1,2   (n_i unit)
#     delta = -(w0 * d_0 * n_0 + a * d_1 * n_1 + b * d_2 * n_2)
#
# On a genuinely FLAT facet ``delta`` is exactly the zero vector BY
# CONSTRUCTION, by either of two guards in the helper. Algan's own flat family
# (``Polyhedron`` and everything built on it) packs no vertex normals at all --
# the corner normals are literally zero -- and the degenerate-normal guard
# (``norm > 1e-9``) returns zero for it. A mesh that DOES carry a duplicated
# face normal at each corner (an import, an authored mesh) instead trips the
# constant-field test: after normalizing, the three normals agree
# (``n0 . n1 > 1 - 1e-6`` and ``n0 . n2 > 1 - 1e-6``) and the formula is
# skipped outright, so float evaluation cannot leave ulp-scale dust on flat
# geometry. Either way today's origin stays bit for bit. ``delta`` is bounded
# by the facet, so it needs no clamp and no epsilon. It therefore cannot help
# a genuinely flat mesh (there delta IS zero); what it buys is that the trace
# side may RELAX its face-normal horizon cull wherever the origin actually
# moved onto the smooth surface: that cull exists to keep terminator-band rays
# off the geometry, and once the origin sits on the smooth surface the face
# normal's horizon is not the surface's.
#
# Tri-state, like shadow_anyhit's "gather" string; ``shadow_terminator_mode``
# is the int the kernels see:
#   0  off -- today's origin, today's guard. The A/B control.
#   1  on (the default) -- Hanika offset AND the relaxed guard.
#   2  ALGAN_SHADOW_TERMINATOR=relax -- DIAGNOSTIC ONLY, not a supported
#      configuration: the guard is relaxed but the origin is NOT offset.
#      This is the arm that makes the acne visible, and therefore the only
#      thing that can prove the offset is what removes it.
shadow_terminator = (
    2
    if env_str("ALGAN_SHADOW_TERMINATOR", "1").strip().lower() in ("relax", "2")
    else env_flag("ALGAN_SHADOW_TERMINATOR", True)
)


def set_shadow_terminator(enabled):
    """Toggle the shadow-terminator offset (see ``shadow_terminator``).

    ``True``/``False`` switch it on/off; the string ``"relax"`` (or a value
    equal to ``2``) selects mode 2, the diagnostic guard-relaxation-only arm.
    Anything else is read for truth: ``None`` and ``0`` are off, other numbers
    are on. Takes effect at the next render batch.

    Selecting mode 2 needs an EXACT 2, and that is deliberate. It is the arm
    whose images are knowingly wrong, so nothing should land on it by rounding:
    an earlier version truncated with ``int(enabled) == 2`` and quietly put
    ``2.5`` there, while routing ``np.int32(2)`` and ``np.float64(2.0)`` --
    the same number in two dtypes -- to different arms, because only the
    latter passes ``isinstance(x, float)``. Comparing by value fixes both.
    """
    global shadow_terminator
    if isinstance(enabled, str):
        shadow_terminator = 2 if enabled.strip().lower() == "relax" else bool(enabled)
    elif enabled is not None and enabled == 2 and not isinstance(enabled, bool):
        shadow_terminator = 2
    else:
        shadow_terminator = bool(enabled)


def shadow_terminator_mode():
    """Live shadow-terminator mode as an int: 0 off, 1 Hanika offset + the
    relaxed guard (the default), 2 the diagnostic relax-only arm.

    Read at call time (never imported by value) and returned as an int,
    because it reaches the resolve/shade kernels as a TEMPLATE value: each
    mode compiles its own kernel variant, so the offline cache cannot serve
    one mode's kernel for another (see ``glossy_reflection_mode``).
    """
    if not shadow_terminator:
        return 0
    return 2 if shadow_terminator == 2 else 1


# The acceptance floor a shadow ray keeps against its OWN primitive, as a
# fraction of the batch's scene scale (the diagonal of the merged triangle
# bounding box over every frame of the batch). This is what retires the last
# absolute constant on the shadow path: `min_hit_distance` = 1e-4 is only ever
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
shadow_eps_relative = env_float("ALGAN_SHADOW_EPS_RELATIVE", 1e-5)

# What fraction of that floor a hit on the SAME mesh but a DIFFERENT primitive
# keeps. 0.0 is primitive-precise: only the triangle the ray actually started
# from is treated as a possible artifact, so a concave crease and a mesh with
# two separate parts get their contact shadow back. 1.0 restores mesh-wide
# rejection, which is what shipped first and what to compare against. Values
# in between buy back protection at mesh seams, where the reconstructed point
# of one facet can land under its neighbour: raise this if a diced curved
# surface shows seam speckle with the feature on.
shadow_near_fraction = env_float("ALGAN_SHADOW_NEAR_FRACTION", 0.0)


def set_shadow_eps_relative(value):
    """Set the shadow acceptance floor as a fraction of scene scale (see
    ``shadow_eps_relative``). Takes effect at the next render batch.
    """
    global shadow_eps_relative
    shadow_eps_relative = float(value)


def set_shadow_near_fraction(value):
    """Set the same-mesh share of the shadow acceptance floor (see
    ``shadow_near_fraction``). Takes effect at the next render batch.
    """
    global shadow_near_fraction
    shadow_near_fraction = float(value)


# Build the dedicated opaque-only STBVHs only when a rollout that walks them
# (wf_opaque_closest / wf_opaque_prepass) is live at build time; otherwise
# alias the main tree -- same kernel ABI, and the opaque-tree reads are
# compiled out by the same templates that gate those rollouts. Saves the
# second per-geometry build (~40% of per-batch BVH build time) and its
# arena bytes. ALGAN_OPAQUE_BVH_SKIP_DEAD=0 restores the unconditional
# builds (byte-level A/B: the skip also shrinks the merged scene the arena
# planner measures).
opaque_bvh_skip_dead = env_flag("ALGAN_OPAQUE_BVH_SKIP_DEAD", True)


def set_opaque_bvh_skip_dead(enabled):
    """Toggle skipping the dedicated opaque-only STBVH builds while no
    rollout consumes them (see ``opaque_bvh_skip_dead``). Takes effect at
    the next batch's scene merge.
    """
    global opaque_bvh_skip_dead
    opaque_bvh_skip_dead = bool(enabled)


def refit_bvh_active():
    """Live effective value of the refit-BVH toggle: the legacy textured /
    sorted-material orchestrators walk the classic tree only.
    """
    return bvh_refit and not wf_textured and wavefront_sort_materials is not True


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
# is max_surfaces_per_ray (currently 256), not literally unbounded. Custom
# scatter, mem-trim, in-place AA, near clipping and legacy routes still fall
# back to classic. Default ON (ALGAN_HYBRID_RASTER=0 restores the classic
# iteration-zero wavefront).
hybrid_raster = env_flag("ALGAN_HYBRID_RASTER", True)


def set_hybrid_raster(enabled):
    """Toggle the hybrid raster primary-visibility front-end (see
    ``hybrid_raster``).
    """
    global hybrid_raster
    hybrid_raster = bool(enabled)


# Screen-space intersection mode inside the hybrid raster frontend. When on
# (default), one projection record is precomputed per (frame, triangle), and
# candidate chunks use edge functions plus perspective-correct barycentrics.
# Invalid/camera-plane-straddling projections fall back to exact per-pixel
# Moller-Trumbore ray casting. ALGAN_RASTER_SS=0 forces ray casting for all
# triangle candidates; the optimal policy may eventually be selected per pair.
raster_ss = env_flag("ALGAN_RASTER_SS", True)


def set_raster_ss(enabled):
    """Toggle screen-space rasterization in the hybrid raster front-end (see
    ``raster_ss``).
    """
    global raster_ss
    raster_ss = bool(enabled)


# Once-per-window batched circuit screen-bounds precompute inside the hybrid
# raster front-end (the bezier analogue of the per-batch triangle projection
# table). The per-(tile, frame) fallback re-projects every circuit's AABB
# corners with ~130 small tensor dispatches per call, which dominates host
# time on circuit-only scenes (tiny scenes measured ~8s of a ~19s render).
# Byte-identical by construction -- identical elementwise arithmetic, batched
# over the frame dimension; validated by benchmarks/_raster_bez_pre_parity.py.
# The toggle is a kill-switch / A-B hook.
raster_bez_precompute = env_flag("ALGAN_RASTER_BEZ_PRECOMPUTE", True)


def set_raster_bez_precompute(enabled):
    """Toggle the batched circuit screen-bounds precompute in the hybrid
    raster front-end (see ``raster_bez_precompute``).
    """
    global raster_bez_precompute
    raster_bez_precompute = bool(enabled)


# The flat-triangle companion of raster_bez_precompute: batches the bbox /
# class-mask derivation and candidate pair emission that ``_frame_pairs``
# performed per (tile, frame) on top of the per-batch projection table.
# Byte-identical by construction; same parity script.
raster_tri_precompute = env_flag("ALGAN_RASTER_TRI_PRECOMPUTE", True)


def set_raster_tri_precompute(enabled):
    """Toggle the batched triangle screen-bounds precompute in the hybrid
    raster front-end (see ``raster_tri_precompute``).
    """
    global raster_tri_precompute
    raster_tri_precompute = bool(enabled)


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
raster_straddle_clip = env_flag("ALGAN_RASTER_STRADDLE_CLIP", True)


def set_raster_straddle_clip(enabled):
    """Toggle the camera-plane clip of hybrid-raster candidate bboxes (see
    ``raster_straddle_clip``).
    """
    global raster_straddle_clip
    raster_straddle_clip = bool(enabled)


# Empty-pixel fast path of the raster resolve: the prefilled frame buffer IS
# the committed state of an uncovered pixel, so the sparse route touches only
# covered pixels and empty screen regions cost nothing.  The sheet route is
# built on that identity, so this flag is one of its preconditions
# (analytic_raster_route_active): switching it off routes the batch to the
# classic supersampled wavefront.  Kill-switch / A-B hook.
raster_empty_skip = env_flag("ALGAN_RASTER_EMPTY_SKIP", True)


def set_raster_empty_skip(enabled):
    """Toggle the empty-pixel fast path of the hybrid raster resolve (see
    ``raster_empty_skip``).
    """
    global raster_empty_skip
    raster_empty_skip = bool(enabled)


# Host-side per-frame candidate-class summary flags for the batched screen-
# bounds tables: one conservative (opaque, translucent) "any candidates"
# bool per frame, computed once per window and moved to the host beside the
# tables.  ``_window_pairs`` then skips its per-tile tensor work -- most
# importantly the synchronizing ``.nonzero()`` inside ``_class_pairs_flat``
# -- for every (tile, class) whose covered frames provably have no
# candidates.  Byte-identical: a skipped class is exactly one whose mask was
# all-false, where ``_class_pairs_flat`` returned None anyway.  Same parity
# script as raster_empty_skip.
raster_pair_flags = env_flag("ALGAN_RASTER_PAIR_FLAGS", True)


def set_raster_pair_flags(enabled):
    """Toggle the host-side per-frame candidate-class flags used to skip
    empty per-tile pair emission (see ``raster_pair_flags``).
    """
    global raster_pair_flags
    raster_pair_flags = bool(enabled)


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
raster_fused_gather = env_flag("ALGAN_RASTER_FUSED_GATHER", False)


def set_raster_fused_gather(enabled):
    """Toggle the fused six-array fragment gather (see ``raster_fused_gather``).
    Takes effect at the next batch's emission.
    """
    global raster_fused_gather
    raster_fused_gather = bool(enabled)


# Kernel opaque-prefix truncation in the sparse emission
# (sheet_compact_taichi.opaque_prefix_keep, behind ``_opaque_prefix_keep``).
# Each covered pixel keeps its fragment prefix through the first proven-opaque
# hit; torch routed an amin scatter per opaque fragment through a whole-stream
# repeat_interleave segment map and then compared two full-length arrays to
# build the keep mask -- several [n] intermediates and two device syncs
# (nonzero) for what is a per-pixel prefix scan over the CSR the host already
# has. One kernel walks each pixel's run instead: find the first opaque
# fragment, write the flags.
#
# Bit-identical by construction: both arms are integer flag comparisons over
# identical ranges.
#
# Measured on the real nn-scene 3840x2160 stream (3.13 M fragments over 756 k
# covered pixels): the keep mask itself goes 2.9 -> 0.4 ms per frame; of the
# 15.3 ms the whole truncation block cost before, the remainder is the
# any/sum/nonzero device syncs both arms still share.
raster_opaque_trunc_kernel = env_flag("ALGAN_RASTER_OPAQUE_TRUNC_KERNEL", True)


def set_raster_opaque_trunc_kernel(enabled):
    """Toggle the kernel opaque-prefix truncation mask (see
    ``raster_opaque_trunc_kernel``). Takes effect at the next batch's emission.
    """
    global raster_opaque_trunc_kernel
    raster_opaque_trunc_kernel = bool(enabled)


# Kernel pair expansion behind ``raster_pipeline._class_pairs_flat``
# (sheet_compact_taichi.pair_expand_count / pair_expand_write). The window
# pairs' chunk expansion built its ``(primitive, frame, bbox, offset)`` rows
# with a ``nonzero``, six gathers, a ``repeat_interleave`` whose OUTPUT is the
# whole row count, an arange/cumsum/subtract chain for the chunk offsets, and
# an eight-column ``stack`` in int64 narrowed to int32 -- measured on the real
# nn-scene 3840x2160 frame (49,307 triangle candidates expanding to 6.34 M
# pair rows), 50 ms of the 57 ms call, almost all of it traffic through
# ~800 MB of int64 intermediates for a 200 MB result. Two kernels replace it:
# one counts chunks per candidate, the host keeps only the cuB prefix sum it
# kept anyway, and one writes each row directly (binary search over the prefix
# for its candidate).
#
# Bit-identical by construction: every column is an integer computed from the
# same inputs, and the row order -- candidates ascending in flattened
# row-major order, chunks ascending within a candidate -- is the order
# ``nonzero`` + ``repeat_interleave`` produced, which downstream fragment
# offsets (and through them the stable sort ties) inherit.
raster_pair_expand_kernel = env_flag("ALGAN_RASTER_PAIR_EXPAND_KERNEL", True)


def set_raster_pair_expand_kernel(enabled):
    """Toggle the kernel pair-row expansion (see ``raster_pair_expand_kernel``).
    Takes effect at the next batch's emission.
    """
    global raster_pair_expand_kernel
    raster_pair_expand_kernel = bool(enabled)


# Covered-pixel-compacted resolve: the emission already knows exactly which
# pixels hold fragments, so the resolve launches one thread per COVERED pixel
# instead of one per screen pixel that early-outs, turning the resolve from
# O(screen pixels) into O(covered pixels).  Empty pixels keep the frame
# buffer's prefill untouched (so this requires raster_empty_skip; an
# environment map is served by prefilling the map itself per pixel in
# render_chunk).  A precondition of the sheet route
# (analytic_raster_route_active): off routes the batch to the classic
# supersampled wavefront.
raster_covered_shade = env_flag("ALGAN_RASTER_COVERED_SHADE", True)


def set_raster_covered_shade(enabled):
    """Toggle the covered-pixel-compacted raster resolve (see
    ``raster_covered_shade``).
    """
    global raster_covered_shade
    raster_covered_shade = bool(enabled)


# Fully sparse primary-raster lifecycle: emit exact hit records for every
# candidate, sort/cull them in sparse hit space, and allocate every downstream
# structure for the unique covered pixels only.  It requires the
# retired-empty/background identity used by raster_empty_skip and the
# covered-pixel resolve semantics; environment maps and in-kernel tonemapping
# are served on this route by the env prefill and the composite/uncovered
# finalize (DESIGN_sheet_resolve.md §5).  A precondition of the sheet route
# (analytic_raster_route_active): off routes the batch to the classic
# supersampled wavefront.
raster_sparse_coverage = env_flag("ALGAN_RASTER_SPARSE_COVERAGE", True)


def set_raster_sparse_coverage(enabled):
    """Toggle the exact covered-pixel lifecycle of the hybrid raster path."""
    global raster_sparse_coverage
    raster_sparse_coverage = bool(enabled)


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
sheet_resolve = env_flag("ALGAN_SHEET_RESOLVE", True)


def set_sheet_resolve(enabled):
    """Toggle the sheet-compaction resolve (DESIGN_sheet_resolve.md)."""
    global sheet_resolve
    sheet_resolve = bool(enabled)


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
# gives the band one occlusion write -- so flipping it moves color at crease
# pixels, never coverage.
sheet_shade_split = env_flag("ALGAN_SHEET_SHADE_SPLIT", True)


# Cross-pass material memoization in the shadowed sheet resolve
# (RENDERER_WORK_QUEUE.md item 9). A shadowed batch launches
# ``sheet_resolve_shade`` TWICE over the same sheets -- mode 1 walks the
# transport and builds the shadow events, mode 2 shades reading the traced
# visibility -- and mode 1 already fetches everything mode 2 re-fetches. The
# obvious saving (skip the fetches in mode 1) is not available: the color's
# alpha, the transmission and the reflectivity all steer the walk itself, so
# cutting them changes which sheets are reached and therefore the shadows.
#
# What IS available is carrying them across. With this on, mode 1 stores each
# processed triangle sheet's color (4), alpha, reflectivity, roughness, IOR,
# transmission and surface point -- twelve floats -- and mode 2 reads them
# back instead of calling _tri_color_g / _tri_extra_g /
# _tri_ior_transmission_g / _tri_surface_point again. The values are copied
# verbatim through f32, so the frame is BYTE-IDENTICAL
# (benchmarks/_sheet_memo_parity.py).
#
# DEFAULT OFF, on measurement -- the candidate is built and correct, and it
# does not pay on a GPU. Measured on a Tesla T4 (2026-08-27, tag memo3),
# warm RUN 2, unsynced profile:
#
#   nn_scene_UHD      sheet_resolve_shade  0.306 s -> 0.304 s   (1.2% of a
#                     22.9 s render; end to end 25.69 -> 25.85 s, neutral)
#   static_gallery    sheet_resolve_shade  0.027 s -> 0.027 s   (0.6% of a
#                     4.5 s render)
#
# The stage this optimises is 0.6-1.2% of a render on that card, and the memo
# moves it by less than a millisecond, because the fetches it removes were
# already cheap next to what the kernel is actually bound by. Against that it
# costs 48 B per sheet of arena, which the runtime memory model prices into
# the next chunk's length -- a real cost for no measured gain.
#
# WHY IT LOOKED PROMISING, because the trap is reusable: the number that
# ranked this work came from benchmarks/_resolve_mode_ratio.py, which brackets
# every launch with a device sync. That sync makes each launch absorb the
# queue it drains, so it reported ~12 s and ~16 s per mode on a render whose
# whole resolve kernel is 0.3 s. The harness says so in its own docstring
# ("read the two modes' TOTALS against each other, not against an unsynced
# profile"); the reading that ranked the memoization went past it and treated
# the ratio as if it sized the stage. It does not. Size a stage from the
# unsynced profile, always.
#
# Kept rather than reverted: it is byte-identical either way
# (benchmarks/_sheet_memo_parity.py), it compiles out entirely when off, and
# a CPU render spends a much larger share in this kernel -- so
# ALGAN_SHEET_RESOLVE_MEMO=1 is worth trying there. It should not be flipped
# on for a GPU render without a fresh unsynced profile of the target scene.
sheet_resolve_memo = env_flag("ALGAN_SHEET_RESOLVE_MEMO", False)


def set_sheet_resolve_memo(enabled):
    """Toggle the shadowed resolve's cross-pass material memo (see
    ``sheet_resolve_memo``). Takes effect at the next resolve launch.
    """
    global sheet_resolve_memo
    sheet_resolve_memo = bool(enabled)


def set_sheet_shade_split(enabled):
    """Toggle the crease shading-class split in sheet compaction (see
    ``sheet_shade_split``). Takes effect at the next batch's emission.
    """
    global sheet_shade_split
    sheet_shade_split = bool(enabled)


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
sheet_mask_kernel = env_flag("ALGAN_SHEET_MASK_KERNEL", True)


def set_sheet_mask_kernel(enabled):
    """Toggle the kernel sample-mask reductions in sheet compaction (see
    ``sheet_mask_kernel``). Takes effect at the next batch's emission.
    """
    global sheet_mask_kernel
    sheet_mask_kernel = bool(enabled)


# Kernel one-mesh reduction in the sparse emission
# (sheet_compact_taichi.one_mesh_pixel_reduce / one_mesh_pixel_apply, behind
# ``raster_pipeline._one_mesh_pixel_caps``). The per-pixel surface-id spread
# and the two facing-split f64 coverage sums behind analytic_aa_one_mesh were
# four scatter reductions routed through a whole-stream repeat_interleave
# segment map, plus two full-length f64 ``where`` temporaries; two kernels now
# walk each covered pixel's CSR run with all four aggregates in registers, and
# the segment map never exists.
#
# The id spread is integer min/max -- exact under any order. The coverage sums
# keep their float64 accumulate / float32 round contract (ss6.6.4): the torch
# arm's atomic ``scatter_add_`` had no summation order at all, the kernel walks
# its pixels serially in stream order, and both are expected bitwise-equal
# after the round for the reason ``sheet_band_reduce``'s area sum is --
# verified by measurement in benchmarks/_sheet_kernel_check.py, not assumed.
#
# Measured on the real nn-scene 3840x2160 stream (3.13 M fragments over 756 k
# covered pixels): 8.5 -> 2.1 ms per frame, and the whole-stream
# repeat_interleave segment map no longer exists.
sheet_one_mesh_kernel = env_flag("ALGAN_SHEET_ONE_MESH_KERNEL", True)


def set_sheet_one_mesh_kernel(enabled):
    """Toggle the kernel one-mesh reduction in the sparse emission (see
    ``sheet_one_mesh_kernel``). Takes effect at the next batch's emission.
    """
    global sheet_one_mesh_kernel
    sheet_one_mesh_kernel = bool(enabled)


# Order a sheet by its nearest POSITIONED fragment -- one that owns at least
# one sub-pixel sample -- instead of by its nearest fragment of any kind
# (sheets.compact_sheets).
#
# The emission also emits AREA DONORS: fragments whose clipped area is real
# but which own no sample, because the triangle slips between the sample
# positions (raster_taichi's sliver policy, "an EMPTY mask is EMITTED as an
# area donor"). A donor carries area and no position, and the design says so
# in as many words -- an all-donor sheet is "areal, position-less" (§4.4).
# The compaction nevertheless let a donor set the sheet's DEPTH, which is a
# position: it is what the pixel's sheets are sorted by, and what the resolve
# reads back as ``t_hit``.
#
# That only matters where two sheets' depth ranges INTERLEAVE -- two opaque
# surfaces crossing inside one pixel -- because there the whole pixel goes to
# whichever sheet sorts first, both of them claiming exact area 1 and the
# full sample union. A donor at the leading corner of the pixel then carries
# its whole surface in front of one that is nearer at every sample.
#
# Measured on ``solids_and_camera``'s axis triad, where each ``Arrow3D``
# shaft starts buried inside the ``Dot3D`` marker and punches out through it.
# At one pixel of the sphere:
#
#     sample:      0        1        2        3        4        5     ...
#     shaft:    7.52637  7.52637  7.51145  7.53989  7.51917  7.53989
#     sphere:   7.50552  7.50552  7.50552  7.50640  7.50411  7.50384
#
# the sphere is nearer at all eight samples, and its area-weighted depth
# (7.5055) is well in front of the shaft's (7.5275). The shaft sorted first
# anyway, on a 0.017-area donor at t=7.49997 that owns no sample at all, and
# took the pixel whole -- a supersampled reference gives that pixel ~100%
# sphere. Those are the stray arrow-colored specks inside the marker sphere,
# and the same mechanism speckles any thin mob running inside a thicker one
# (a ``Line3D`` inside an ``Arrow3D`` shaft).
#
# THE BLAST RADIUS IS THE INTERLEAVING CASE ONLY, and within it only sheets
# that own a sample AND have a donor nearer than their nearest sampled
# fragment. The walk order is a fragment's place in the emission's (depth
# bin, descending layer) stream, so two sheets keep their relative order
# under either rule whenever every fragment of one precedes every fragment of
# the other there -- ordinary nested and stacked geometry, untouched. A sheet
# with no sampled fragment at all keeps its nearest donor's depth: it is
# position-less, there is nothing better, and the resolve already treats it
# as a uniform veil.
#
# What this does NOT do is antialias the interpenetration seam: the pixel
# still goes whole to one surface, as a z-buffer would, and only the choice
# is repaired. Blending it needs per-sample depth in the resolve (a depth
# plane per sheet), which is not built -- DESIGN_sheet_resolve.md §6.1.1 is
# where that limit is declared.
sheet_positioned_depth = env_flag("ALGAN_SHEET_POSITIONED_DEPTH", True)


def set_sheet_positioned_depth(enabled):
    """Toggle positioned-fragment sheet ordering (see
    ``sheet_positioned_depth``). Takes effect at the next batch's emission.
    """
    global sheet_positioned_depth
    sheet_positioned_depth = bool(enabled)


# Per-sample depth gate for interpenetrating sheets (DESIGN_sheet_resolve.md
# ss6.1.1, OX_SHEET_INTERPENETRATION_AUDIT.md ss6). At compaction the host
# computes, per sub-pixel sample, each material-opaque full-coverage triangle
# sheet's exact nearest depth at that sample, and a sheet cedes a sample when
# another SURFACE's enforcer is strictly nearer there beyond depth_tie_epsilon;
# the resolve zeroes those samples' claim/occlusion slots, so two surfaces
# crossing inside one pixel paint winner-per-sample instead of whole-pixel to
# whichever sheet sorted first. Ties and near-ties keep today's walk order.
# Reflective materials veto the pixel: a reflective sheet can break the walk
# before the winner claims, which would turn a mis-gated pixel into an unlit
# one. Multi-sheet bands -- shade-class siblings and conflict-rank splits --
# are exempt on BOTH sides: their band-pooled arithmetic writes occlusion once
# and ignores slots, so gating a sibling would over-occlude.
sheet_sample_depth = env_flag("ALGAN_SHEET_SAMPLE_DEPTH", True)


def set_sheet_sample_depth(enabled):
    """Toggle per-sample depth gating of interpenetrating sheets (see
    ``sheet_sample_depth``). Takes effect at the next batch's compaction.
    """
    global sheet_sample_depth
    sheet_sample_depth = bool(enabled)


# Kernel lane-owner scan behind sheet_sample_depth
# (sheet_compact_taichi.sheet_lane_first_owner, behind
# ``sheets._lane_first_owners``). Building the per-sample nearest-owner depth
# table asked, once per sample lane, "which sorted fragment of each sheet owns
# this lane earliest" -- eight masked full-length ``where`` copies and eight
# amin ``scatter_reduce_`` passes over the stream. One kernel does all eight
# lanes' atomic mins in a single pass into one pre-initialised table.
#
# Bit-identical: integer amin per (sheet, lane) slot, order-independent, and
# every step after the table is identical arithmetic on identical values.
#
# Measured on the real nn-scene 3840x2160 stream (3.13 M fragments): 42.3 ->
# 7.7 ms per call under a synthetic uniform band distribution, and most of the
# 14.5 ms the lane loop measured on the stream's own (skewed, mostly
# single-fragment) bands.
sheet_sample_depth_kernel = env_flag("ALGAN_SHEET_SAMPLE_DEPTH_KERNEL", True)


def set_sheet_sample_depth_kernel(enabled):
    """Toggle the kernel lane-owner scan of sheet_sample_depth (see
    ``sheet_sample_depth_kernel``). Takes effect at the next batch's emission.
    """
    global sheet_sample_depth_kernel
    sheet_sample_depth_kernel = bool(enabled)


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
# as the cumsums do -- so unlike sheet_mask_kernel above it needs no
# order-independence argument at all. The max=15 clamp stays in
# compact_sheets in both arms.
sheet_rank_kernel = env_flag("ALGAN_SHEET_RANK_KERNEL", True)


def set_sheet_rank_kernel(enabled):
    """Toggle the kernel conflict-rank scan in sheet compaction (see
    ``sheet_rank_kernel``). Takes effect at the next batch's emission.
    """
    global sheet_rank_kernel
    sheet_rank_kernel = bool(enabled)


# Kernel per-band order stats + dominant fragment behind ``compact_sheets``
# (sheet_compact_taichi.band_stats_reduce / band_stats_rep_orig). The five
# per-band scatters -- nearest sorted/original position (amin), the same
# restricted to POSITIONED fragments, the largest exact area (amax), and the
# fragment count -- each walked the whole stream once through their own
# scatter_reduce_/scatter_add_, and the positioned restriction masked two of
# them with full-length ``where`` copies; one kernel updates all six tables in
# a single visit per fragment. The dominant fragment's position needs its
# band's completed maximum first, so it stays a second launch.
#
# Bit-identical: integer amin/amax/add are exact under any atomics order, an
# f32 amax has no association at all, and the tie-break (earliest original
# position among fragments AT the max) is the same amin over the same
# candidates both arms compare. The caller's gathers off the filled tables are
# unchanged.
sheet_band_stats_kernel = env_flag("ALGAN_SHEET_BAND_STATS_KERNEL", True)


def set_sheet_band_stats_kernel(enabled):
    """Toggle the fused per-band order stats / dominant fragment scan (see
    ``sheet_band_stats_kernel``). Takes effect at the next batch's compaction.
    """
    global sheet_band_stats_kernel
    sheet_band_stats_kernel = bool(enabled)


# Kernel application of the solid-shell opacity ceiling behind
# ``compact_sheets`` (sheet_compact_taichi.solid_shell_ceiling). After the
# block's own depth sort orders each (pixel, surface) segment -- and after the
# f64 cumsum that prefixes it, which STAYS IN TORCH because a serial register
# walk cannot reproduce a cub scan's reassociation bitwise (measured: 61 of
# 3.13 M spend values move, flipping 10 visible f32 outputs on the real
# nn-scene frame) -- torch walked the stream three more times to spend the
# allowance: nonzero + scatter_ for segment starts, two facing-split f64
# scatter_add_s through a whole-stream segment map, and clone + index_copy_
# for the write-back. One thread per segment (detected in-kernel along the
# permutation) walks its run twice with the cap in registers and the caller's
# prefix as data.
#
# The float contract follows ss6.6.4 as before: front/back sums accumulate in
# f64 registers serially in stream order (the atomic scatter_add_ had no
# order), verified bitwise against the torch arm at 4K shapes and on the real
# captured stream in benchmarks/_sheet_kernel_check.py rather than assumed.
sheet_shell_ceiling_kernel = env_flag("ALGAN_SHEET_SHELL_CEILING_KERNEL", True)


def set_sheet_shell_ceiling_kernel(enabled):
    """Toggle the kernel solid-shell ceiling application (see
    ``sheet_shell_ceiling_kernel``). Takes effect at the next batch's
    compaction.
    """
    global sheet_shell_ceiling_kernel
    sheet_shell_ceiling_kernel = bool(enabled)


# Analytic anti-aliasing (see DESIGN_analytic_aa.md). Instead of rendering at
# ``super_sampling_anti_aliasing`` times the output resolution and box-filtering back down
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
# Flat triangles are covered too (see analytic_aa_tri below), and the quantities
# coverage cannot express analytically -- shadow-edge visibility and the image
# seen inside a reflection or refraction -- are handled by taking N sub-pixel
# samples of those specific queries (analytic_aa_secondary_samples). Measured
# against the supersampled super_sampling_anti_aliasing=2 default across
# eleven feature-specific scenes, analytic AA at aa=1 is better on eight and
# 7-9% short on three (specular highlights, a flat mirror's reflected image, a
# lens's refracted image), where the residual is the CONTENT of a minified
# secondary image. Read DESIGN_analytic_aa.md ss19 before dropping
# ``super_sampling_anti_aliasing`` to 1; what is still untouched is texture
# minification (no mip chain).
analytic_aa = env_flag("ALGAN_ANALYTIC_AA", True)

# PHASE 2 (implemented): flat triangles. Coverage comes from the screen-space
# edge functions ``_ss_pixel`` already evaluates, normalised by the edge lengths
# in columns 10:12 of the projection table. Triangles need a seam rule that
# circuits do not: two triangles sharing an edge inside a pixel cover it
# completely between them, and plain multiplicative compositing would leave a
# background-colored lattice on every internal edge. The resolve therefore
# tracks transmittance independently for the fixed sub-pixel samples; disjoint
# masks partition the pixel without a source-object side table.
#
# Subordinate per-geometry switches (only meaningful while analytic_aa is on).
analytic_aa_bez = env_flag("ALGAN_ANALYTIC_AA_BEZ", True)
#
# Triangle coverage: exact fixed-point rasterization (a 1/4096-pixel integer
# lattice, int64 edge functions and a top-left fill rule) partitions eight
# sub-pixel samples among the triangles covering a pixel, the seam rule sums the
# disjoint sub-areas of one object, and per-sample occlusion keeps a mesh's back
# faces out of its own silhouette. Against a super_sampling_anti_aliasing=4
# reference it beats the plain aliased render on every config -- a subdivided
# sphere, a translucent one, sub-pixel rods, a slanted quad -- at 40-78% less
# error, with essentially the reference's own edge gradation (588 distinct edge
# levels against 608). See DESIGN_analytic_aa.md ss14-ss16.
analytic_aa_tri = env_flag("ALGAN_ANALYTIC_AA_TRI", True)

# The seam rule itself. Off, coverage still scales alpha but consecutive
# fragments of one object composite multiplicatively instead of unioning their
# disjoint sub-areas -- which is the lattice this exists to remove. Kept as a
# toggle purely so the parity script can measure the difference; there is no
# reason to turn it off in a real render.
analytic_aa_seam = env_flag("ALGAN_ANALYTIC_AA_SEAM", True)

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
analytic_aa_sliver = env_str("ALGAN_ANALYTIC_AA_SLIVER", "drop")

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
analytic_aa_exact = env_flag("ALGAN_ANALYTIC_AA_EXACT", True)

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
analytic_aa_bez_wedge = env_flag("ALGAN_ANALYTIC_AA_BEZ_WEDGE", True)

# The ss21.3/21.8/21.9 exact-triangle formulations (single exact area vs the
# mask, packed cells, scalar surface accounting) are DELETED, not parked:
# DESIGN_analytic_aa_v2.md's run-corrected representation (analytic_aa_run)
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
# analytic_aa / analytic_aa_tri; the sliver policy knob is inert under it
# (sliver behavior is fixed by the design, not configurable).
# Default ON (2026-08-13) on the v2 ss7.2 ladder: static mesh L1
# 0.0355 -> 0.0292 against the aa=4 reference, tri video 0.119 -> 0.107 at
# edge levels 620/621, seam notches inside the documented band, trans
# improves, thin gains its reachable share (0.857 -> 0.884; the 0.99 target
# was calibrated on the rejected cells accounting -- see the ss8 Phase D
# note). Worst-case cost is +6.6% frame device on sub-pixel-diced meshes;
# RUN=0 is byte-identical to the pre-v2 renderer.
analytic_aa_run = env_flag("ALGAN_ANALYTIC_AA_RUN", True)

# The corr > 1 accounting rule (v2 ss4.4), the design's one open empirical
# question, decided by harness: "clamp" scales the run's per-sample writes by
# corr and clamps each at zero (claim exact, leftover keeps a bounded residual
# of the shed error); "redistribute" additionally pushes the clamped residue
# onto the run's unowned samples (leftover exact, weirder per-sample
# semantics). Compile-time template value; both stay byte-identical while
# analytic_aa_run is off.
# Measured (v2 ss4.4, decided by harness as designed): redistribute wins --
# tri L1 0.107 vs clamp's 0.110 with edge levels 620 against the aa=4
# reference's own 621, seam notches 9 vs 12, trans/thin at parity. Exact
# leftovers cost two registers and a run-end scale.
ANALYTIC_AA_RUN_RULES = ("clamp", "redistribute")
analytic_aa_run_rule = env_str("ALGAN_ANALYTIC_AA_RUN_RULE", "redistribute")

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
# analytic_aa_seam and to the run rule (aa_tri 3 or 4).
analytic_aa_run_full = env_flag("ALGAN_ANALYTIC_AA_RUN_FULL", False)


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
# This is what mob-declared identity (mesh_id) was built to enable and what no
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
# IMPLIES analytic_aa_run_full, and that implication is wired in exactly one
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
analytic_aa_one_mesh = env_flag("ALGAN_ANALYTIC_AA_ONE_MESH", True)


# What ``Mob.opacity`` MEANS on a closed solid. ``opacity`` is documented as a
# property of the Mob, so rendering at opacity ``a`` over a backdrop must give
# ``a * (the Mob rendered opaque) + (1 - a) * backdrop`` in linear light. A flat
# Circle satisfies it exactly. A closed solid does not: a camera ray crosses its
# shell twice, both crossings composite, and the measured consequence is that
# every built-in solid under-delivers -- an authored 0.55 sphere renders 0.679,
# an authored 0.55 cube renders 0.744 (benchmarks/_opacity_alpha_check.py, the
# harness this switch is verified against).
#
# ON, the sheet compaction caps a DECLARED-closed surface's cumulative exact
# coverage per (pixel, surface) at max(front, back) -- the larger of its two
# shells' own areas -- walking its sheets in depth order and shrinking later
# sheets' area as the allowance is spent. An interior pixel of a convex shell
# holds front = back = 1, so the far sheet is left with nothing regardless of
# runtime sample visibility, which is exactly the one-attenuation composite; at
# the silhouette the cap is the shell's own footprint, so no ink is lost at the
# rim (the harness's ``ink`` column is the instrument for that). The cap is NOT
# clamped to 1: a ray crossing a declared shell MORE than twice (a torus hole,
# a mid-morph self-overlap) legitimately attenuates per crossing -- the
# conflict-rank machinery's measured behaviour (sheets.py) -- and front sums
# past 1 keep it.
#
# Exempt, by construction rather than by special case: surfaces that do not
# declare closed (open cones/cylinders, partial sweeps, unprovable Polyhedra,
# all 2-D shapes), and transmissive materials, whose declaration folds away at
# pack time because refraction visits both shells as physical transport.
#
# KNOWN LIMIT, and it is a CROSS-ROUTE one. This rule lives in the sheet
# compaction, which feeds the resolve that serves PRIMARY visibility.
# Reflection and refraction continuations leave that resolve through
# ``_spawn_pool_ray`` into the classic wavefront bounce loop, and
# ``wavefront_shade`` composites every hit it drains with no ceiling of any
# kind -- it does not even receive ``tri_obj``, so it holds no surface identity
# to key one on (the comment at wavefront_kernels_taichi.py says so in as many
# words). A half-transparent solid therefore composites at its authored opacity
# when the camera looks at it directly, and at the old doubled opacity in a
# MIRROR's image of it. The same gap applies to the Monte Carlo megakernel
# (``samples_per_pixel > 1``), whose stochastic transparency gives each shell an
# independent interaction chance and so reproduces ``(1 - a)^2`` in
# expectation, and to any batch the sheet route rejects and the classic
# wavefront serves instead (``analytic_raster_route_active``). Measured before
# this rule existed, the two primary routes agreed: sphere 0.55 delivered 0.679
# on the sheet route and 0.677 on the wavefront one, so what the fallback still
# does is exactly the old behaviour rather than some third thing. Closing
# either needs surface identity plumbed into ray or path state, which is a
# wider change than this one and is not attempted here.
#
# OFF restores today's behaviour exactly: the ceiling lives entirely in the
# compaction, gated on this flag read at batch time, so no pixel, sheet or
# kernel variant changes.
solid_shell_alpha = env_flag("ALGAN_SOLID_SHELL_ALPHA", True)

# Deliver the DIRECT LIGHTS' share of the reflected specular lobe, which the
# traced continuation cannot: a ray only finds light that has geometry, and a
# directional or point light is a delta.
#
# A hit's energy is partitioned into a reflected share ``R`` (traced), a
# transmitted share, and a remainder that weights the locally shaded color.
# That partition is sound as reflectance, but the shaded color is where the
# analytic GGX highlight lives -- so the lights' own specular reflection is
# weighted by the share that is explicitly NOT reflected, and the materials
# whose reflected or transmitted lobe dominates lose their highlight outright.
# Clear glass is the case that surfaced it: ``trans_share = 1 - R`` eats the
# remainder, leaving the highlight ``R * (1 - _mirror_share(roughness))`` --
# 1.2% of it at roughness 0.05. A perfect mirror is the same defect wearing
# ``R = 1`` instead, shading at weight zero.
#
# The scatter sites therefore add the lobe back at exactly the complement of
# the share the shaded color already carries (``R * _mirror_share +
# trans_share``), which puts the reflected lobe at unit weight overall and
# leaves the diffuse/ambient partition untouched. Not double counting: the
# traced ray carries the environment, this carries the delta lights, and the
# two sources are disjoint.
#
# Measured on ``scenes/matlight_pbr_subset.json`` against three-gpu-pathtracer,
# black background, the transmissive sphere's own disc: (0.034, 0.063, 0.013)
# linear at g/r 1.87 -- doubly tinted, no highlight -- becomes (1.129, 1.159,
# 1.108) at g/r 1.03, against the reference's g/r 1.07. The opaque control and
# the transmission-0.5 sphere move by 0.06% and 0.2%.
#
# Scope: the built-in material arm of both primary routes (the sheet resolve
# and the classic wavefront bounce loop). A custom fragment scatter owns its
# own transport and a bezier circuit is never material-shaded, so neither has
# a GGX highlight to restore. OFF restores the previous weighting exactly.
direct_specular_lobe = env_flag("ALGAN_DIRECT_SPECULAR_LOBE", True)

# Retire a bounce-loop ray whose throughput fell under min_weight even when
# its last processed hit took an in-place reflection branch.
#
# The drain loop already retires any ray whose post-hit weight crosses the
# significance floor (``wavefront_shade``'s in-loop test), but every reflect-
# here branch ends in a ``break`` that jumps past that test, and the
# post-loop peel-complete tests deliberately exclude bounced rays -- so a
# reflecting sub-floor ray rides to ``max_bounces``, several further
# generations of pure waste (diagnosed in scratch_perf/r3/ox/
# REPORT_immortal_rays.md: on the nn PREVIEW scene exactly 30 rays bounce at
# 100% survival through bounces 5-7, all below the floor since their first
# tail iteration). The fix adds the same floor test to the post-loop block,
# between the peel-complete tests and the surface-ceiling test, where the
# bounced rays now arrive. Completion, not truncation: the existing commit
# block deposits accumulated color + leftover throughput (env map included)
# exactly as for any other retirement, and ALLOC_TRUNC_SURFACES is untouched.
#
# Not provably byte-identical: it retires transport the renderer currently
# traces, bounded by the envelope the existing floor already accepts --
# dropped contribution <= w * radiance with w < 1e-3 at cull time and decay
# <= ~0.105/generation afterwards, under half a u8 LSB for scene radiance
# <= ~2. Measured (scratch_perf/r3/ox/REPORT_weight_floor_impl.md, nn scene
# at PREVIEW, lossless libx264rgb encode, benchmarks/_video_diff.py): worst
# channel diff exactly 1 -- 4424 byte-differing pixel-instances over 50
# frames (worst frame 107 of 278784), nothing over the suites' tol-2 gate.
# The default is ON: the project owner reviewed that measurement and accepted
# a 1-LSB maximum variation without visual inspection (2026-08-26), the same
# deliberate posture as the pn_criterion fast_math exception. The committed
# render baselines stay valid because every suite tolerates channel
# deviations up to 2. ALGAN_WEIGHT_FLOOR_EXIT=0 /
# SETTINGS.raytracing.experimental.weight_floor_exit opts out and restores
# the pre-change tree byte-identically (verified against it directly).
#
# Gate: a ``ti.template()`` argument of ``wavefront_shade`` (the kernel is at
# Taichi's runtime-argument ceiling; see the packed-ndarray comments there),
# read live per batch at the call sites -- so flipping it mid-process
# compiles the other variant rather than baking (CLAUDE.md's ti.static
# hazard). OFF compiles today's kernel body exactly.
weight_floor_exit = env_flag("ALGAN_WEIGHT_FLOOR_EXIT", True)


# Per-mob shadow flags (``Mob.casts_shadows`` / ``Mob.receives_shadows``).
#
# OFF restores the pre-flag renderer exactly, and does it entirely on the HOST:
# the caster bit is simply not stamped into the BVH leaf words, and the material
# block's ``no_shadow_receive`` slot is left at its 0.0 default. Both kernel
# tests then read a bit that is never set and a slot that is always zero, which
# is bit-for-bit what they did before the flags existed -- so no kernel variant
# changes and there is no ti.static gate to go stale mid-process (CLAUDE.md's
# hazard). The flags stay settable on a Mob with this off; they just do nothing,
# which is the same thing ``SETTINGS.raytracing.shadows = False`` does to them.
per_mob_shadow_flags = env_flag("ALGAN_PER_MOB_SHADOW_FLAGS", True)


def set_per_mob_shadow_flags(enabled):
    """Toggle the per-mob shadow flags (see ``per_mob_shadow_flags``).
    Takes effect at the next batch's scene merge.
    """
    global per_mob_shadow_flags
    per_mob_shadow_flags = bool(enabled)


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
# READ THIS BEFORE CONCLUDING ANYTHING FROM THE FLAG. ``bvh_refit`` defaults
# ON, and ``_build_accel``'s refit branch ignores ``builder`` outright, so at
# shipped defaults NO STBVH is built for any geometry type and this flag --
# like ``ALGAN_BVH_BUILD`` -- changes nothing at all. It governs the tree you
# get with ``ALGAN_BVH_REFIT=0``, and that is the only configuration in which
# either the win above or any A/B of it exists.
bez_bvh_split = env_flag("ALGAN_BEZ_BVH_SPLIT", True)


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
#   Sphere + color checkerboard         4096 -> 3968 tris   max|d| 0
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
weld_surface_seams = env_flag("ALGAN_WELD_SURFACE_SEAMS", True)


def set_weld_surface_seams(enabled):
    """Toggle surface seam/pole welding (see ``weld_surface_seams``).

    Takes effect on the next primitive build.
    """
    global weld_surface_seams
    weld_surface_seams = bool(enabled)


def set_bez_bvh_split(enabled):
    """Toggle median-split ordering for the bezier STBVH (see ``bez_bvh_split``).

    Takes effect on the next scene build.
    """
    global bez_bvh_split
    bez_bvh_split = bool(enabled)


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
# N=4 those positions are the 2x2 grid super_sampling_anti_aliasing=2
# supersamples at, which is the arm this is meant to match.
#
# The split happens ONCE, at the primary hit; deeper bounces continue as single
# rays, so the cost is N times the secondary traversal, not N^depth. Only the
# reflective/refractive pixels pay it. 1 disables it, and is byte-identical.
analytic_aa_secondary_samples = env_int("ALGAN_ANALYTIC_AA_SECONDARY", 4)

# Minimum share of a pixel a REFLECTED or REFRACTED branch must carry before it
# is worth spending N sub-pixel continuations on instead of one.
#
# Without this, a plain glossy sphere -- whose only "reflection" is the ~4%
# Fresnel sheen every PBR dielectric has -- spawns four extra traced rays per
# pixel for a lobe contributing 4% of its color, and measures both slower and
# slightly worse than plain supersampling. The whole value of coverage is that
# the expensive fallbacks fire only on the pixels that need them.
analytic_aa_secondary_min_energy = env_float(
    "ALGAN_ANALYTIC_AA_SECONDARY_MIN_ENERGY", 0.12
)

glossy_reflection = env_flag("ALGAN_GLOSSY_REFLECTION", True)

# Rotate each pixel's lobe fan by a 4x4 Bayer index (interleaved sampling), so
# four taps read as a smear rather than four ghost copies of the reflected
# image: neighbouring pixels sample different parts of the lobe and the eye
# integrates across them. Fixed in SCREEN space, hence still frame-independent
# -- the pattern does not swim, twinkle or depend on time. Off restores the
# plain per-fragment fan (kept so the parity script can measure the difference).
glossy_interleave = env_flag("ALGAN_GLOSSY_INTERLEAVE", True)


# SPLIT-SUM PREFILTERING for glossy reflections, the answer to why the tap fan
# above ships disabled (renderer audit REPORT.md §4.5.1,
# DESIGN_glossy_prefilter.md). Instead of N taps over the lobe, ONE
# deterministic ray in the mirror direction with throughput 1, accumulated into
# a per-pixel reflection buffer; the lobe's ENERGY comes from the analytic
# split-sum DFG term (which replaces ``_mirror_share``'s throttle outright) and
# its SHAPE from prefiltering that buffer by the lobe's screen footprint before
# compositing.
#
# It fixes both halves of what the fan gets wrong. Nothing crawls, because the
# ray direction is a smooth function of position rather than a screen-fixed
# dither pattern; nothing ghosts, because a wide lobe is a wide filter rather
# than N discrete copies; and it costs FEWER rays than the fan, not more --
# more taps was never the lever, since the artefact the throttle exists to hide
# is minification aliasing and no amount of point sampling fixes that.
#
# DEFAULT ON, gated behind ``glossy_reflection`` -- which is now default on too,
# so this is the route a rough reflector takes unless something says otherwise.
# The old tap fan is still reachable for comparison with
# ``set_glossy_reflection(True, prefilter=False)``.
glossy_prefilter = env_flag("ALGAN_GLOSSY_PREFILTER", True)

# How many mip levels the prefilter's reflection pyramid may have. Each level
# doubles the blur radius it can represent, so 10 covers a sigma of ~148 px --
# past a frame's own height at every preset below UHD, and a lobe wider than
# the frame reads as the average of everything glossy in it either way. Lower
# it only to cap the pyramid's memory; the buffers are per FRAME, not per
# batch, and the runtime memory model measures them like everything else.
glossy_prefilter_max_levels = env_int("ALGAN_GLOSSY_PREFILTER_LEVELS", 10)


def set_glossy_reflection(enabled, *, interleave=None, prefilter=None):
    """Toggle roughness-driven glossy reflections (see ``glossy_reflection``).

    ``prefilter`` selects the split-sum route (default) over the tap fan;
    ``interleave`` only means anything to the fan.
    """
    global glossy_reflection, glossy_interleave, glossy_prefilter
    glossy_reflection = bool(enabled)
    if interleave is not None:
        glossy_interleave = bool(interleave)
    if prefilter is not None:
        glossy_prefilter = bool(prefilter)


def glossy_reflection_mode():
    """Live glossy-lobe mode: 0 off, 1 fan only, 2 fan + per-pixel rotation,
    3 split-sum prefilter (``glossy_prefilter``, the default when glossy
    reflections are on at all).

    Read at call time (never imported by value) and returned as an int, because
    it reaches the resolve as a TEMPLATE value: each mode compiles its own
    kernel variant, so the offline cache -- which does not invalidate on
    ``@ti.func`` edits, let alone on a Python constant -- cannot serve one
    mode's kernel for another.
    """
    if not glossy_reflection:
        return 0
    if glossy_prefilter:
        return 3
    return 2 if glossy_interleave else 1


def glossy_blur_sigma_px(roughness, d_reflected, d_primary, theta_px):
    """The prefilter's blur radius in pixels, in pure Python.

    The kernels compute this inline (``sheet_resolve_taichi`` produces the
    per-pixel scale, ``glossy_prefilter_taichi.gloss_scatter`` applies the cone
    factor); this is the same arithmetic in one place, for tests and for
    anything on the host that needs to predict it. See
    ``DESIGN_glossy_prefilter.md`` §3 for where each term comes from.

    ``d_reflected`` is how far past the primary hit the reflected content sits;
    ``inf`` (a ray that escaped) is a reflection of the sky and blurs by the
    full lobe angle, and 0 (a reflection in contact with its reflector) does
    not blur at all.
    """
    alpha = float(roughness) * float(roughness)
    sigma_angle = 2.0 * alpha
    d_r = float(d_reflected)
    d_p = float(d_primary)
    if d_r == float("inf"):
        cone = 1.0
    else:
        total = d_p + d_r
        cone = 0.0 if total <= 0.0 else max(0.0, min(1.0, d_r / total))
    return cone * sigma_angle / float(theta_px)


# NESTED DIELECTRIC MEDIA for the deterministic tracer: a ray carries the stack
# of media it is inside (rs_sca columns 7+, see
# ``wavefront_kernels_taichi.IOR_STACK_DEPTH``) and each glass interface takes
# the RELATIVE index n_inside/n_outside instead of assuming air outside. This
# is what makes glass inside glass, a sphere inside a box, or a bubble in a
# liquid bend light correctly at the inner interfaces; without it every
# interface refracts as though the outside were air.
#
# DEFAULT ON. It shipped opt-in until its output had been lived with; it has
# been. The stack widens ``rs_sca`` by 5 f32 per ray (``IOR_STACK_DEPTH`` = 4
# entries plus the depth counter) and compiles its own template variants of
# both shade kernels, in any batch whose ``ior_stack_flag`` is set --
# ``nested_ior_mode() != 0 and refraction_flag``. Read ``refraction_flag``
# before assuming that means "a scene with glass in it": it is also set by a
# reflective primitive under analytic AA (``_secondary_split_needed``, which
# is what gives a mirror the split pool it needs), and every PBR triangle is
# reflective. So an ordinary ``MeshStandardMaterial`` scene takes the wider
# state and the new variants with no transmission anywhere.
#
# What that costs such a scene is the state and a cold compile, not its
# pixels: with nothing transmissive, no transmitted child is ever spawned, so
# nothing pushes or pops the stack and every interface still reads air. That
# is measured, not assumed -- `tests/fast` (a `MeshStandardMaterial` scene,
# hence a widened one) and five of the six `tests/full_renders` scenes render
# byte-identically with the gate off and on; only `materials_and_lighting`,
# the one scene carrying transmission, moves.
#
# What it changes there is a nested scene, which was simply wrong before, plus
# a thin edge/silhouette band on an un-nested solid where a ray grazing a
# shared edge used to be classified as ENTERING a solid it never left; the
# stack declines to bend it a second time, which is the physically right
# answer at a hit where there is no interface.
#
# Two limits stand with it on. Fresnel reflectance keeps the MATERIAL index
# (the relative index reaches only Snell's law): ``_material_reflectance``'s
# dielectric branch is itself gated ``ior > 1 + 1e-4``, which a relative index
# below 1 would silently zero. And a scene carrying a custom fragment scatter
# gets no nesting at all. See DESIGN_mesh_identity_open.md §H for both, and
# ``benchmarks/_nested_ior_ab.py`` for the four frames that bound this.
nested_ior = env_flag("ALGAN_NESTED_IOR", True)


def set_nested_ior(enabled):
    """Toggle the nested-dielectric IOR stack (see ``nested_ior``)."""
    global nested_ior
    nested_ior = bool(enabled)


def nested_ior_mode():
    """Live nested-IOR mode: 0 off, 1 media stack maintained.

    Read at call time (never imported by value) and returned as an int,
    because it reaches the shade/resolve kernels as a TEMPLATE value: each
    mode compiles its own kernel variant, so the offline cache cannot serve
    one mode's kernel for another (see ``glossy_reflection_mode``).
    """
    return 1 if nested_ior else 0


# Minimum half-width, in pixels, of a filled circuit's drawn region. This
# replaces the classic ``outline_w = 0.6 * pixel_size`` fill dilation, whose
# purpose is to keep sub-pixel features (hairlines, thin glyph stems, degenerate
# zero-area fills) from vanishing entirely. The classic constant is 0.6 of a
# SUPERSAMPLE pixel and is therefore NOT anti-alias-level invariant: at the
# reference AA=2 it dilates by 0.3 output pixels, at AA=1 by 0.6. Analytic AA
# runs at AA=1, so 0.3 reproduces the reference appearance rather than doubling
# every stroke weight. Tune only against rendered Text/Tex.
analytic_aa_bez_min_half_width = env_float("ALGAN_ANALYTIC_AA_BEZ_MIN_HALF_WIDTH", 0.3)

# Maximum curve-to-chord flattening error, in pixels, for Bezier circuits under
# analytic AA (overrides the primitive's own ``num_pixels_per_sample`` only when
# it is looser). The classic 0.5 is measured against the SUPERSAMPLED height, so
# at the AA=2 reference it is 0.25 output pixels; at AA=1 it would relax to 0.5
# and a continuous coverage function would expose the flattening facets that box
# filtering currently hides. Costs edges (memory + _bezier_point_metrics work).
analytic_aa_chord_tolerance = env_float("ALGAN_ANALYTIC_AA_CHORD_TOLERANCE", 0.25)


def set_analytic_aa(
    enabled,
    *,
    bezier=None,
    triangles=None,
    seam=None,
    sliver=None,
    secondary=None,
    exact=None,
    wedge=None,
    run=None,
    run_rule=None,
    run_full=None,
    one_mesh=None,
):
    """Toggle analytic anti-aliasing (see ``analytic_aa``)."""
    global analytic_aa, analytic_aa_bez, analytic_aa_tri, analytic_aa_seam
    global analytic_aa_sliver, analytic_aa_secondary_samples, analytic_aa_exact
    global analytic_aa_run, analytic_aa_run_rule, analytic_aa_run_full
    global analytic_aa_one_mesh, analytic_aa_bez_wedge
    if secondary is not None:
        analytic_aa_secondary_samples = int(secondary)
    if exact is not None:
        analytic_aa_exact = bool(exact)
    if wedge is not None:
        analytic_aa_bez_wedge = bool(wedge)
    if run is not None:
        analytic_aa_run = bool(run)
    if run_full is not None:
        analytic_aa_run_full = bool(run_full)
    if one_mesh is not None:
        analytic_aa_one_mesh = bool(one_mesh)
    if run_rule is not None:
        if run_rule not in ANALYTIC_AA_RUN_RULES:
            raise ValueError(f"run_rule must be one of {ANALYTIC_AA_RUN_RULES}")
        analytic_aa_run_rule = run_rule
    analytic_aa = bool(enabled)
    if bezier is not None:
        analytic_aa_bez = bool(bezier)
    if triangles is not None:
        analytic_aa_tri = bool(triangles)
    if seam is not None:
        analytic_aa_seam = bool(seam)
    if sliver is not None:
        if sliver not in ANALYTIC_AA_SLIVER_MODES:
            raise ValueError(f"sliver must be one of {ANALYTIC_AA_SLIVER_MODES}")
        analytic_aa_sliver = sliver


def effective_analytic_aa_secondary_samples():
    """Live continuation-ray tap count; 1 (off) unless analytic AA is on.

    Clamped to ``[1, _AA_SEC_MAX]`` and otherwise returned unchanged: every
    count in that range has real sub-pixel positions (hand-written for 1/2/4/8,
    generated for the rest), so there is no supported set left to snap to. N
    still reaches the resolve as a template value, which is why a value above
    the ceiling is clamped with a warning rather than honoured -- the position
    mask is an i32 bitfield, and each extra distinct tap count also pays a
    kernel compile of its own (see the warning text).
    """
    if not analytic_aa:
        return 1
    n = int(analytic_aa_secondary_samples)
    if n > _secondary_tap_ceiling():
        _warn_secondary_clamped(n)
        return _secondary_tap_ceiling()
    return max(1, n)


def _secondary_tap_ceiling():
    """The tap ceiling, from the kernel module that owns the bitfield.

    Imported at call time rather than at module import: importing
    ``raster_taichi`` from here at module level would make this module's
    import depend on the whole geometry-kernel family loading in a fixed
    order, and this function runs once per batch prep, not once per pixel.
    """
    from algan.rendering.raytracing.raster_taichi import _AA_SEC_MAX

    return _AA_SEC_MAX


#: Set by the first clamp warning; a misconfigured tap count clamps every
#: batch of every render, and one notice per process is what a reader needs.
_SECONDARY_CLAMP_WARNED = False


def _warn_secondary_clamped(requested):
    """Warn once per process that an over-ceiling tap count was clamped."""
    global _SECONDARY_CLAMP_WARNED
    if _SECONDARY_CLAMP_WARNED:
        return
    _SECONDARY_CLAMP_WARNED = True
    get_logger("raytracing").warning(
        f"analytic_aa_secondary_samples={requested} exceeds the ceiling of "
        f"{_secondary_tap_ceiling()} and was clamped to it: the "
        f"continuation-position mask is a 32-bit field, one bit per tap, and "
        f"every extra distinct count is paid again in kernel compile time "
        f"because the resolve unrolls ti.static(range(sec_aa)) at four call "
        f"sites."
    )


def analytic_aa_sliver_mode():
    """Index of the live sample-less-triangle policy in the mode tuple.

    Read at call time (never imported by value) and returned as an int, because
    it reaches the kernels as part of the ``aa`` template value: the geometry
    kernels see ``1 + mode``, so each policy compiles its own variant and the
    offline cache cannot serve one policy's kernel for another.
    """
    try:
        return ANALYTIC_AA_SLIVER_MODES.index(analytic_aa_sliver)
    except ValueError:
        return ANALYTIC_AA_SLIVER_MODES.index("drop")


def analytic_aa_bez_active():
    """Live effective value of circuit analytic coverage.

    Read at call time, never imported by value: settings are module globals with
    env-var defaults and user code flips them after import.
    """
    return analytic_aa and analytic_aa_bez


def analytic_aa_bez_mode():
    """Circuit coverage as the kernels' ``aa_bez`` template value.

    0 off, 1 the box filter, 2 the exact angle-aware area
    (``analytic_aa_exact``), 3 that plus the two-segment boundary model
    (``analytic_aa_bez_wedge``, default on since 2026-08-13).
    The distinction rides in the template value so the two forms cannot share an
    offline-cache entry; everything downstream that only asks whether circuit
    coverage is on keeps testing it for truth.
    """
    if not analytic_aa_bez_active():
        return 0
    if not analytic_aa_exact:
        return 1
    return 3 if analytic_aa_bez_wedge else 2


def analytic_aa_tri_active():
    """Live effective value of flat-triangle analytic coverage."""
    return analytic_aa and analytic_aa_tri


# UNSUPPORTED legacy "textured surface" wavefront (Surface / flat-triangle
# scenes only). This variant is no longer maintained and no longer works; the
# monolithic general wavefront is the only supported deterministic tracer.
# When on, the deterministic wavefront shaded from three per-triangle texture
# lookups instead of per-vertex arrays: a color texture (RGBA+glow), a
# material texture (the shading parameter block) and a surface texture
# (reflectivity/roughness/index-of-refraction used for scatter); see
# scene_builder._build_textured_scene + wavefront_textured_kernels_taichi. It
# was a proof-of-concept built to benchmark the texture-lookup shading
# architecture, kept for reference only. Default OFF; do not enable.
wf_textured = False


def set_wf_textured(enabled):
    """Reject the removed legacy texture-lookup wavefront renderer."""
    global wf_textured
    if bool(enabled):
        wf_textured = False
        raise UnsupportedFeatureError(
            "The legacy textured wavefront renderer is unsupported and cannot "
            "be enabled. Use the general deterministic wavefront renderer."
        )
    wf_textured = False


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
merge_on_gpu = env_flag("ALGAN_MERGE_ON_GPU", True)

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
merge_gpu_peak_factor = env_float("ALGAN_MERGE_GPU_PEAK_FACTOR", 6.0)


# Exact measurement of the GPU merge's transient peak, which calibrates
# ``merge_gpu_peak_factor``. This used to default off because it called
# ``torch.cuda.reset_peak_memory_stats`` directly and so destroyed the
# process-wide peak counter ``profiling_utils`` reports for the whole render.
# It now goes through ``memory_utils.begin_cuda_peak``/``end_cuda_peak``, which
# remember the displaced high-water mark, so measuring costs a pair of cheap
# counter reads and nothing else -- hence on by default. The headroom bound
# itself is still the ``merge_gpu_peak_factor`` estimate.
merge_track_peak = env_flag("ALGAN_MERGE_TRACK_PEAK", True)


def set_merge_on_gpu(enabled):
    """Toggle GPU-side scene merge + STBVH build (see ``merge_on_gpu``)."""
    global merge_on_gpu
    merge_on_gpu = bool(enabled)


def merge_on_gpu_active():
    """True when the scene merge + STBVH build should run on the render device.

    Requires ``merge_on_gpu`` and a CUDA render device -- the offload only pays
    off on a real accelerator, and the transient-peak accounting uses the
    ``torch.cuda`` memory-stats API. A CPU (or MPS) render device keeps the
    merge on the CPU, byte-identically to the pre-toggle path.
    """
    if not merge_on_gpu:
        return False

    return render_device().type == "cuda"


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
project_on_gpu = env_flag("ALGAN_PROJECT_ON_GPU", True)

# Conservative multiplier from a batch's pre-projection source-geometry bytes
# to the projection's transient device peak (source + shading scratch + packed
# ``_rt_*`` output; the polyline sampling can expand bezier geometry well past
# its control points, hence a larger default than the merge factor). Bounds the
# projection against the pool headroom before it is attempted; the OOM retry is
# the exact fallback. Read live.
project_gpu_peak_factor = env_float("ALGAN_PROJECT_GPU_PEAK_FACTOR", 8.0)


def set_project_on_gpu(enabled):
    """Toggle GPU-side ``project_to_screen`` (see ``project_on_gpu``)."""
    global project_on_gpu
    project_on_gpu = bool(enabled)


def project_on_gpu_active():
    """True when ``project_to_screen`` should run on the render device.

    Requires ``project_on_gpu`` and a CUDA render device (see
    ``merge_on_gpu_active`` for why CUDA specifically).
    """
    if not project_on_gpu:
        return False

    return render_device().type == "cuda"


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
# ALGAN_PN_CRITERION_KERNEL=0 restores the torch path (for A/B). Used wherever
# the criterion's tensors already sit on Taichi's arch device, which is two
# arrangements: a CUDA render device that projection has uploaded to, and a CPU
# render device, where the arch is x64 and the host tensors never had to move.
#
# Only the first qualified until 2026-08-26. The exclusion was written as
# "requires CUDA" but its reason was staging -- launching Taichi against a
# tensor that is not on its arch's device copies every argument through VRAM
# (see generate_array_states' docstring) -- and that cannot happen when the arch
# IS the CPU. So the rule turned the kernels off in the one case where they cost
# nothing to run. benchmarks/_pn_criterion_cpu_ab.py measures what that was
# worth on a CPU render.
#
# The second reason given for the exclusion -- that projection may run on the
# prefetch worker rather than the render thread -- is not a hazard here: P13
# established that Python-side Taichi launches from that worker are safe
# alongside main-thread launches, and the cpu_prep_kernel_enabled kernels
# already launch from it on every CPU render.
pn_criterion_kernel = env_flag("ALGAN_PN_CRITERION_KERNEL", True)


def set_pn_criterion_kernel(enabled):
    """Toggle the fused subdivision-level criterion kernels (see
    ``pn_criterion_kernel``).
    """
    global pn_criterion_kernel
    pn_criterion_kernel = bool(enabled)


def pn_criterion_kernel_active():
    """True when the level searches should use their fused Taichi kernels.

    Two arrangements put the criterion's tensors on Taichi's arch device, which
    is the whole requirement (see ``taichi_runtime.taichi_launch_is_local``):

    * projection ran on a CUDA render device, so the geometry is already there;
    * the arch **is** the CPU, so the host tensors never had to go anywhere.

    The second was excluded until 2026-08-26 for a reason that only covered the
    first: launching against CPU tensors stages every argument through VRAM.
    That is true when the arch is CUDA and false when it is x64, so the rule as
    written turned the kernels off in the one case where they are free. The
    remaining tensor-by-tensor device check lives in the input builders.
    """
    if not pn_criterion_kernel:
        return False
    if project_on_gpu_active():
        return True
    # Deferred: taichi_runtime imports taichi, and this module is read during
    # settings construction. Asking does not force initialization.
    from algan.rendering.taichi_runtime import taichi_arch_is_cpu

    return taichi_arch_is_cpu()


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
pn_geometry_slack = env_flag("ALGAN_PN_GEOMETRY_SLACK", True)

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
pn_anisotropic_dice = env_flag("ALGAN_PN_ANISOTROPIC_DICE", True)


# Feature bitmask for the UNSUPPORTED legacy textured wavefront (see
# wf_textured): each bit compiled one of the monolith's features back into the
# (otherwise lean) textured shade kernel, so the marginal occupancy /
# performance cost of each could be measured one at a time (see
# benchmarks/_wf_textured_features_ab.py). The features are added in the order
# beziers -> custom scatter -> shadows -> normal maps.
WF_TEX_BEZ = 1  # bezier-circuit traversal + shading
WF_TEX_SCATTER = 2  # per-material custom scatter dispatch (ray bouncing)
WF_TEX_SHADOWS = 4  # binary hard shadow rays (triangle occluders)
WF_TEX_NORMALMAP = 8  # tangent-space normal-map perturbation of the shading normal
wf_textured_features = env_int("ALGAN_WF_TEXTURED_FEATURES", 0)


def set_wf_textured_features(mask):
    """Reject feature configuration for the removed textured renderer."""
    global wf_textured_features
    if int(mask) != 0:
        wf_textured_features = 0
        raise UnsupportedFeatureError(
            "Textured-wavefront feature masks are unsupported because that "
            "legacy renderer has been removed from the public execution path."
        )
    wf_textured_features = 0


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
# materials (it drains up to kbuf hits per launch, whereas sorting pays
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


wavefront_sort_materials = "0"  # auto"


def set_wavefront_sort_materials(enabled):
    """Reject the removed legacy sorted-material renderer when forced on."""
    global wavefront_sort_materials
    parsed = _parse_sort_mode(enabled)
    if parsed is True:
        wavefront_sort_materials = "auto"
        raise UnsupportedFeatureError(
            "The legacy sorted-material wavefront renderer is unsupported. "
            "Use the monolithic deterministic shade kernel."
        )
    wavefront_sort_materials = parsed


def set_fragment_shading(enabled):
    """Toggle per-fragment shading of the *deterministic* ray tracer.

    When enabled, triangle/PN hits whose material is one of the core lit
    shaders (the legacy diffuse default, ``MeshBasicMaterial``,
    ``MeshLambertMaterial``, ``MeshPhongMaterial``, ``MeshStandardMaterial``,
    ``MeshPhysicalMaterial``, ``MeshToonMaterial``, ``MeshNormalMaterial``,
    ``MeshMatcapMaterial`` and ``MeshDepthMaterial``) are shaded per fragment
    in-kernel from the raw albedo, a per-primitive material block and the
    scene's lights -- crisper specular highlights and smooth shading on coarse
    meshes. Other
    materials keep vertex shading.
    Only the deterministic renderer (``set_samples_per_pixel(1)``, non-physical)
    is affected. Set before rendering.
    """
    global fragment_shading
    fragment_shading = bool(enabled)


# What a RECT AREA LIGHT's shadow rays integrate. Each of its packed rows
# stands for one cell of its emitter grid, but nothing downstream knew a cell
# existed: RectAreaLight.build_aux left the shadow-radius column (packed
# column 11) at zero, so every row took the single-hard-ray path in both
# shadow fans and the union of K hard shadows was a staircase with K+1 levels
# (measured [0.01, 0.25, 0.52, 0.74] on a k/4 grid at samples=4) wherever a
# continuous ramp belongs.
#
# ON, build_aux packs each row's CELL half-extents and the rectangle's right
# axis into columns the area type never used, and both shadow fans place
# their SOFT_SHADOW_SAMPLES samples inside that cell, in the light's own
# plane (an R2 low-discrepancy sequence whose s = 0 sample is exactly the
# cell centre). Nothing about radiance, power fractions or ``intensity``
# changes -- this is the visibility term only.
#
# COST, documented rather than hidden: a row with a non-zero emitter extent
# fires SOFT_SHADOW_SAMPLES (default 8) shadow rays instead of 1 -- the same
# rule a PointLight with a non-zero ``shadow_radius`` already obeys. An area
# light has K rows, so its shadow cost goes from K rays to K * 8.
# ``samples`` stays the user's dial for both quality and cost.
#
# KNOWN LIMITS, both deliberate exclusions. The deferred shadow prepass
# (``wavefront_shadow``) reads neither light type nor radius and treats every
# row as a hard point light; it is dead code today (the tracer always
# compiles ``deferred_shadows == 0``) and must learn these columns before it
# is ever revived. And the Monte Carlo megakernel's next-event estimation
# reads packed columns 0-2 only, with extended lights rejected at preflight
# when ``samples_per_pixel > 1`` anyway, so an area row never reaches it:
# SPP > 1 keeps hard per-row rays.
#
# OFF restores today's row bit-for-bit. The flag is read host-side ONLY, in
# build_aux, which packs zeros to the extra columns when it is off -- the
# kernels' ``radius`` stays 0.0 and takes the existing single-ray path. There
# is deliberately no ``ti.static`` gate: a compile-time switch would fork the
# shade kernels into per-arm variants (and need one process per arm, since a
# flipped template gate is resolved when the kernel compiles), while this
# shape needs no recompile and one process can render both arms.
area_light_soft_shadows = env_flag("ALGAN_AREA_LIGHT_SOFT_SHADOWS", True)


# When True, the deterministic ray tracer casts hard shadows: each shaded
# triangle/PN fragment fires one shadow ray per point light and multiplies the
# light that remains through every occluder's transparency. Fully opaque
# occluders block the direct contribution. Implies per-fragment shading
# (shadows are evaluated in the lighting model) and forces the general kernel.
# Soft emitters fire a deterministic fan of SOFT_SHADOW_SAMPLES rays instead
# of a single ray: point/spot lights with a non-zero ``shadow_radius`` spread
# theirs over the emitter disk (directional: over ``shadow_angle``, radius =
# tan(half-angle)), and a RectAreaLight's rows each integrate visibility over
# their own cell of the emitter grid when area_light_soft_shadows is on --
# see that flag for the cost. Off by default.
shadows = False

# Number of shadow rays in the deterministic soft-shadow fan (per light with a
# non-zero shadow radius, per shaded fragment). More = smoother penumbras,
# linearly more shadow cost. Baked into the shade kernel at compile time; set
# the env var ALGAN_SOFT_SHADOW_SAMPLES before the first render to change it.


def set_shadows(enabled):
    """Toggle hard shadows in the *deterministic* ray tracer.

    When enabled, every shaded triangle/PN fragment traces one shadow ray per
    scene point light. Every partially opaque surface between the fragment and
    light attenuates its direct diffuse/specular term by ``1 - opacity``;
    stacked surfaces multiply, while a fully opaque surface blocks it. Ambient
    and emissive terms remain unchanged. Shadows are evaluated inside the
    wavefront shade kernel's per-fragment lighting model, so this implies
    :func:`set_fragment_shading` for the render. Lights with a non-zero
    ``shadow_radius`` / ``shadow_angle`` get *soft* shadows via a
    deterministic fan of ``SOFT_SHADOW_SAMPLES`` rays; a RectAreaLight's rows
    do too -- each integrating visibility over its own cell of the emitter
    grid -- when ``area_light_soft_shadows`` is on (the default). Refractive
    glass transport still needs the physical path tracer
    (``set_samples_per_pixel(n)`` with ``n > 1``). Only the deterministic
    renderer (``set_samples_per_pixel(1)``, non-physical) is affected. Set
    before rendering.
    """
    global shadows
    shadows = bool(enabled)


def set_light_intensity(intensity):
    """Radiance scale applied to explicit point lights in physical mode."""
    global light_intensity
    light_intensity = float(intensity)


def set_ambient_light(intensity):
    """Constant ambient lighting term used in physical mode."""
    global ambient_light
    ambient_light = float(intensity)


def set_samples_per_pixel(samples):
    """Set how many rays are averaged per pixel. 1 (the default) uses the
    exact deterministic renderer; larger values enable Monte Carlo path
    tracing with that many samples.
    """
    global samples_per_pixel
    samples_per_pixel = max(1, int(samples))


def set_indirect_bounce_strength(strength):
    """Set the diffuse indirect lighting strength of the Monte Carlo
    renderer (0 disables diffuse bounces).
    """
    global indirect_bounce_strength
    indirect_bounce_strength = float(strength)


def set_linear_color_space(enabled):
    """Enable or disable the linear working color space.

    **On by default.** Authored color is decoded from sRGB to linear light at
    the render boundary, every shading and compositing operation happens in
    linear, and the sRGB transfer function is applied once at the byte write.
    This is the arrangement three.js uses, and it is what makes lights add:
    sRGB encoding is concave, so adding encoded values overshoots the encoded
    sum -- two lights that should put a white surface on byte 188 put it on 255.

    Unlit flat 2-D content is unaffected either way, because decoding and then
    encoding with no arithmetic in between is the identity. What moves is
    anything the renderer actually computes: lit surfaces (mid-tones lift, a
    surface at half illumination going from byte 128 to 188), antialiased edges,
    alpha compositing and the supersample downsample.

    Turning it off restores the previous display-referred pipeline exactly,
    including the illumination-budget normalisation that had to exist to stop
    gamma-space light sums running away. It is there for A/B comparison and for
    reproducing pre-change output; ``LINEAR_COLOR_WORK.md`` has the
    measurements.
    """
    global linear_color_space
    linear_color_space = bool(enabled)


def set_tonemapping(enabled):
    """Enable or disable tonemapping of the rendered frame.

    **Off by default**, which makes output linear: an authored color lands on
    the pixel it names, white renders as 255, and a primary stays primary.

    Enabling it applies the curve selected by :func:`set_tonemap_method`
    ("neutral", the Khronos PBR Neutral mapper, or "agx") -- not ACES, whatever
    the old docstring said. That buys highlight roll-off, so values above 1.0
    stay distinguishable instead of clipping flat, and it costs a shift on
    *every* value: the curve is not the identity anywhere except at 0, an
    authored 255 renders as 222, and saturated colors desaturate. The two are
    not separable -- a curve that is the identity on ``[0, 1]`` must clamp
    above it -- so this is a choice about which you would rather have.
    ``TONEMAP_FINDINGS.md`` has the measurements.

    This flag is honoured wherever the tonemap actually runs, so it works on
    its own -- with ``post_process_tonemap`` on (the default) the composite
    writes linear HDR and the post stage simply clamps instead of applying a
    curve. There is no need to also disable ``post_process_tonemap``, and doing
    so costs HDR headroom (see :func:`set_post_process_tonemap`).
    """
    global tonemapping
    tonemapping = bool(enabled)


def set_tonemap_exposure(exposure):
    """Set the exposure multiplier applied to the frame before it is encoded.

    The color is multiplied by this before the tonemap curve runs, and --
    since tonemapping is off by default -- before the plain clamp too, so it
    brightens or darkens the whole render either way. Defaults to ``1.0``,
    which is exact: it moves no pixel.

    This is the right control for "the whole scene is too dark". Reach for it
    before raising every light's intensity.
    """
    global tonemap_exposure
    tonemap_exposure = float(exposure)


def set_tonemap_method(method):
    """Set the tonemapping method ("neutral" or "agx")."""
    global tonemap_method
    if method not in ("neutral", "agx"):
        raise ValueError("tonemap_method must be 'neutral' or 'agx'")
    tonemap_method = str(method)


def set_post_process_tonemap(enabled):
    """Enable or disable post-process tonemapping instead of in-kernel tonemapping.

    Disabling it makes the composite write **uint8**, which clamps every channel
    -- including the glow lane -- to 0-255 before bloom runs. A mob with
    ``glow > 1`` therefore saturates and its halo comes out markedly dimmer and
    less saturated than on the default HDR path. Turn this off only for an A/B
    against the legacy in-kernel tonemap; to get linear output, use
    :func:`set_tonemapping` alone, which keeps the HDR buffer.
    """
    global post_process_tonemap
    post_process_tonemap = bool(enabled)


# Store the linear-HDR frame buffer as float16 (RGBA16F) instead of float32.
# Halves the frame buffer (so ~2x more frames per batch), but is off by default
# because GPUs with poor FP16 throughput -- notably consumer Pascal (GTX
# 10-series) at ~1/64 FP32 -- run the f16 torch post-processing (and f16 buffer
# traffic) far slower than the memory saving is worth (measured ~80% slower end
# to end on a GTX 1050). On Turing/Ampere+ (fast f16) it is a clear win, so
# enable it there.
#
# Read at buffer allocation, never baked into a kernel: nothing specializes on
# it, so a script may flip it between renders and the next batch allocates the
# other dtype. That is why it is an ordinary setting rather than one of the
# init-only environment variables it used to sit among -- the environment
# variable only seeds this default.
hdr_buffer_f16 = env_flag("ALGAN_HDR_BUFFER_F16", False)


def set_hdr_buffer_f16(enabled):
    """Toggle the float16 linear-HDR frame buffer (see ``hdr_buffer_f16``).

    Takes effect at the next render batch's buffer allocation.
    """
    global hdr_buffer_f16
    hdr_buffer_f16 = bool(enabled)


def hdr_frame_dtype():
    """dtype of the linear-HDR frame buffer used under post-process
    tonemapping: float32, or float16 when ``hdr_buffer_f16`` is on.
    """
    import torch

    return torch.float16 if hdr_buffer_f16 else torch.float32


post_tonemap_kernel = env_flag("ALGAN_POST_TONEMAP_KERNEL", True)


def set_post_tonemap_kernel(enabled):
    """Toggle the standalone Taichi post-process tonemap kernel (vs the torch
    tonemap pipeline). The kernel reuses the in-composite tonemap ti.funcs and
    computes in f32, recovering most of the cost the move to post-process
    tonemapping added (the torch tonemap ran ~20 ops/pixel over every frame).
    Kill-switch / A-B hook.
    """
    global post_tonemap_kernel
    post_tonemap_kernel = bool(enabled)


def is_post_tonemap_kernel_enabled():
    return post_tonemap_kernel


def is_post_process_tonemap_enabled():
    """Return whether post-process tonemapping is enabled."""
    return post_process_tonemap


def _get_tonemap_t_val():
    if post_process_tonemap:
        return 3
    if not tonemapping:
        return 0
    return 2 if tonemap_method == "agx" else 1


# --- Core lit material registry (shader function -> in-kernel material id) ----
# Ids must match shading_taichi: 0 manim (Manim's default 3-D lighting), 1 basic/unlit/passthrough,
# 2 lambert, 3 phong, 4 standard, 5 physical, 6 toon, 7 normal, 8 matcap,
# 9 depth.
def _build_core_shader_ids():
    from algan.rendering.shaders.material_shaders import (
        basic_material_shader,
        depth_shader,
        lambert_shader,
        manim_shader,
        matcap_shader,
        normal_shader,
        phong_shader,
        physical_shader,
        standard_shader,
        toon_shader,
    )
    from algan.rendering.shaders.pbr_shaders import null_shader

    return {
        manim_shader: 0,
        null_shader: 1,
        basic_material_shader: 1,
        lambert_shader: 2,
        phong_shader: 3,
        standard_shader: 4,
        physical_shader: 5,
        toon_shader: 6,
        normal_shader: 7,
        matcap_shader: 8,
        depth_shader: 9,
    }


_CORE_SHADER_IDS = None
# Per-material parameter defaults (canonical 33-slot block; see shading_taichi).
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
    # 27..29 attenuation_sigma: 0.0 is no Beer-Lambert absorption, what every
    # material did before MeshPhysicalMaterial.attenuation_color /
    # attenuation_distance were plumbed through (shading_taichi's slot map).
    0.0,
    0.0,
    0.0,
    # 30..32 the toon / depth materials' own fields, defaults matching their
    # shader signatures (MeshToonMaterial.bands 3, MeshDepthMaterial
    # near 0.1 / far 100). Read only under those materials' pipeline ids, so
    # the non-zero defaults never reach another stage's block.
    3.0,
    0.1,
    100.0,
    # 33 no_shadow_receive: 0.0 is "this surface is darkened by shadows cast
    # onto it", what every mob did before Mob.receives_shadows existed
    # (shading_taichi._MAT_NO_SHADOW_RECEIVE).
    0.0,
]
# Material-property name -> (start slot, width) in the canonical block.
# ``one_sided`` (slot 26) and ``no_shadow_receive`` (slot 33) are deliberately
# absent: both are declared by the mob's geometry, not by its material, and
# ``_pack_material`` writes them directly.
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
    "attenuation_sigma": (27, 3),
    "num_bands": (30, 1),
    "near": (31, 1),
    "far": (32, 1),
}


def _core_shader_ids():
    global _CORE_SHADER_IDS
    if _CORE_SHADER_IDS is None:
        _CORE_SHADER_IDS = _build_core_shader_ids()
    return _CORE_SHADER_IDS


def _shader_material_id(shader):
    """In-kernel material id for a shader function. Unknown / non-core shaders
    (and ``None``) map to 1 (unlit passthrough: the kernel returns the color --
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
    fragment-shading general wavefront (the only path where a mob's colors are
    raw albedo, so a "constant color" is genuinely constant per fragment, and
    the only kernel whose per-vertex reads are guarded for shrunk arrays).
    Every deterministic (samples <= 1) batch renders through that kernel.
    """
    return promote_constants and fragment_shading and samples_per_pixel <= 1


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
