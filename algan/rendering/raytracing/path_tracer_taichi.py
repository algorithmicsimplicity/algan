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
    deterministic renderer. Transparency composites *deterministically*
    (throughput-weighted, never stochastic alpha), so stacked vector
    graphics and text match the deterministic composite with zero variance.
    Which emitter a next-event sample aims at is decided by descending the
    per-frame **light tree** (``light_tree.py``, Conty Estevez & Kulla
    2018) -- a selection that weighs distance and orientation as well as
    power, so cost stays ``O(log E)`` in emitter count and the shadow ray
    is not spent on something facing away. ``pt_light_tree = False``
    restores the flat power CDF exactly.
    Lit surfaces receive next-event estimation over every packed light row
    (emitter radiometry via the shared ``_light_eval``; the SURFACE response
    is the physical BSDF, not the deterministic stage formula -- see
    ``_pt_lit_f_pdf``), and scatter one importance-sampled continuation:
    cosine-hemisphere diffuse, GGX specular via spherical-cap VNDF sampling
    (Dupuy & Benyoub 2023) with Turquin-style multiple-scattering
    compensation, or a refracted transmission ray through the shared
    nested-IOR stack with Beer-Lambert interior absorption.
``pt_reduce``
    Folds a wave's per-path accumulators into the chunk's per-pixel sample
    sums (``accum``), applying leftover throughput to the background. One
    thread per pixel sums its own wave samples, so accumulation needs no
    atomics -- a property of the no-splitting layout, not a promise about
    frames (see DESIGN_path_tracer_roadmap.md section 8).

``finalize_samples`` (in ``raytrace_kernels_taichi``) then averages ``accum``
into the frame buffer exactly as it always has.

Surface treatment by pipeline id (see ``shading_taichi``):

* lambert/phong/standard/physical (2-5): physically integrated -- NEE direct
  lighting through ONE BSDF (``_pt_lit_f_pdf``: ``albedo/pi`` diffuse, exact
  Smith ``G2``, Fresnel, Turquin compensation) for every emitter kind, plus
  the sampled continuation above. The direction-less ambient / hemisphere
  rows contribute ``e_diff * L``, the physical answer for a constant-radiance
  environment over the diffuse lobe, and no specular fill (indirect transport
  replaces that). Light units are the deterministic renderer's; the response
  is not, so a lit surface is not as bright as its ``spp == 1`` render --
  which is deliberate (roadmap section 5). Phong has no Blinn-Phong highlight
  here: it is GGX like everything else. Emissive surfaces illuminate their
  surroundings through BSDF-sampled paths.
* manim/toon/normal/matcap/depth/user (0, 6-9, >= 10): authored appearance --
  the hit is shaded exactly as the deterministic renderer shades it
  (``_run_frag_pipeline``, shadow visibility included), the result treated
  as emitted radiance, and the path continues as a Lambert bounce on the
  base color so these surfaces send and receive indirect light. Which light
  rows that shading sums over is the one thing this renderer changes about
  them: past ``max_shadow_lights`` (or on demand -- ``auth_sampled``,
  ``rt_settings.pt_authored_light_sampling``) the branch fills the ambient
  rows deterministically and DRAWS the rest, scaling each drawn row's
  radiance by ``1 / (S * p)`` through ``_SampledLightView`` so that neither
  the pipeline nor any stage is touched. Every built-in stage is linear in a
  light's colour, so the estimate is unbiased for the sum (roadmap section
  6a-bis).
* unlit (1) and bezier circuits: emission + deterministic transparency; a
  reflective or transmissive circuit spawns the matching specular / pane
  continuation. Unlit content never diffuse-scatters, which keeps text and
  vector graphics exact.

Sampler
-------
Hash-based Owen-scrambled Sobol after Burley, "Practical Hash-based Owen
Scrambling" (JCGT 2020). Only the 2D Sobol (0,2) pair is evaluated directly;
higher dimensions are *padded*: each logical 2D pair reuses the base pair
under an Owen shuffle of the sample index plus per-dimension Owen scrambles,
all seeded by hashes of ``(pt_seed, frame-or-0, pixel, pair)``. Every sample
is a pure function of ``(pt_seed, frame-or-0, pixel, pair, index)`` --
independent of tile, wave, batch and chunk splits, and of thread scheduling.
"frame-or-0" is the sampler's one animation choice
(``rt_settings.pt_animated_seed``, off by default): with the frame folded out
of the key every frame draws the same sample set, so a static region's error
is a fixed noise texture rather than per-frame shimmer; on, each frame is
decorrelated from its neighbours. Both ends -- ``pt_generate`` and
``pt_shade`` -- take the choice as a runtime value and must agree, so it is
carried to the first as a plain argument and to the second in ``nee_meta``
(``_NM_ANIM_SEED``). The one decision still drawn from a hash RNG
(``_pt_rng_seeded``, keyed on the same inputs plus the peel step) is the
authored-appearance branch's per-light soft-shadow jitter in its SUMMING arm,
whose count per crossing is the light count (and whose salt aliases above 64
lights). Its sampling arm draws none: it spends the crossing's own next-event
pairs, listed below, on both the row pick and the light point -- a crossing is
either lit or authored, never both, so the two arms cannot collide over them.
The pass/scatter choice at each crossed surface has a crossing-indexed pair of
its own (see the table below).

Every draw's seed splits in two: ``_pt_path_seed`` mixes ``(pt_seed,
frame-or-0, pixel)`` once per thread, and ``_pt_pair_seed`` mixes the
dimension pair on top of it per draw. That is a hoist, not a weakening --
the combine avalanches its second argument, so distinct pairs still
decorrelate.

Dimension-pair allocation (a fixed table; keep in sync with ``pt_shade``).
``B`` is the render's ``max_bounces`` and ``L`` is ``pt_light_samples``; the
per-crossing block sits after every bounce pair because it draws per surface
CROSSING ``c`` (a translucent stack visits several lit surfaces per bounce
ordinal), not per bounce. One crossing owns ``2L + 1`` pairs: the ``L``
next-event pairs it may draw, plus the lobe select every crossing draws:

=============================  ================================================
pair                           use
=============================  ================================================
0                              sub-pixel jitter (2D)
1                              lens (2D) -- reserved for depth of field
2 + 6b + 0                     bounce ``b``: y Russian roulette (x unused --
                               the roulette draw keeps one component)
2 + 6b + 1                     bounce ``b``: BSDF direction (2D)
2 + 6b + 2, 3                  bounce ``b``: reserved (legacy light slots)
2 + 6b + 4, 5                  bounce ``b``: reserved for volumes
2 + 6B + (2L+1)c + 2s + 0      crossing ``c``, NEE sample ``s``: x entry
                               select (an authored crossing's sampling arm:
                               x light-row select)
2 + 6B + (2L+1)c + 2s + 1      crossing ``c``, NEE sample ``s``: light point
                               (2D)
2 + 6B + (2L+1)c + 2L          crossing ``c``: x lobe select -- pass / diffuse
                               / specular / transmit, or a custom scatter's
                               branch (y unused)
=============================  ================================================

The lobe select used to draw white noise from the hash RNG, on the argument
that a crossing has no fixed dimension index; indexing on ``processed``, the
trick next-event estimation already used, gives it one (roadmap section 7).
"""

from algan.rendering.raytracing.arena_args_taichi import (
    ArenaView,
    arena_packed,
)
from algan.rendering.raytracing.light_tree import (
    LT_AXIS,
    LT_BMAX,
    LT_BMIN,
    LT_COS_THETA_E,
    LT_COS_THETA_O,
    LT_DECAY,
    LT_LEFT,
    LT_PARENT,
    LT_POWER,
    LT_RIGHT,
    LT_SIN_THETA_O,
)
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    _M_IOR,
    _M_REFLECTIVITY,
    _M_ROUGHNESS,
    _M_TRANSMISSION,
    NODE_ARG,
    _bezier_normal,
    _comes_after,
    _generate_ray,
    _sample_circuit_color,
    _shadow_occluded,
    depth_tie_epsilon,
    kbuf,
    max_surfaces_per_ray,
    min_weight,
)
from algan.rendering.raytracing.shading_taichi import (
    _MAT_ATTENUATION_SIGMA,
    _MAT_NO_SHADOW_RECEIVE,
    _MAT_ONE_SIDED,
    _MID_LAMBERT,
    _MID_PHONG,
    _MID_PHYSICAL,
    _MID_UNLIT,
    _USER_PIPELINE_BASE,
    SHADOW_VIS_CHANNELS,
    _light_eval,
    _prep_normal,
    _run_frag_pipeline,
    _sided_shading_normal,
    light_vis_index,
)
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _ACTIVE,
    _DONE,
    _LT_AMBIENT,
    _LT_AREA_SAMPLE,
    _LT_DIRECTIONAL,
    _LT_ENV_SH,
    _LT_HEMISPHERE,
    _PI,
    _env_brdf_approx,
    _material_reflectance,
    _offset_transmitted_origin,
    _refract_ray,
    _relative_ior,
    _sample_env_map,
    _tri_color_g,
    _tri_extra_g,
    _tri_ior_transmission_g,
    _tri_normal_g,
    _write_ior_stack,
)
from algan.taichi_compat import submodule, ti

#: The compiler's own subscript builder, used by ``_SampledLightView`` to index
#: a ``ti.Vector`` local from Python scope (``ArenaView`` reaches the arena
#: buffer the same way).
_ti_impl = submodule("lang.impl")

# Sampler dimension pairs (see the module docstring's table).
PAIR_PIXEL = 0
PAIR_LENS = 1
PAIR_BOUNCE_BASE = 2
PAIRS_PER_BOUNCE = 6
_PAIR_LOBE = 0
_PAIR_BSDF_DIR = 1

# Per-path commit row (``pt_acc``): radiance accumulated so far, the leftover
# throughput the background shows through, the camera-segment alpha
# transparency, and whether this path ever took a STOCHASTIC decision (see
# ``pt_reduce``).
PT_ACC_WIDTH = 10
_PT_ACC_LEFTOVER = 4
_PT_ACC_ALPHA = 8
#: 1.0 once the path has made any random choice beyond the sub-pixel jitter:
#: a lit crossing (next-event estimation picks an emitter), an authored
#: crossing or a custom scatter, or a lobe pick with more than the
#: pass-through branch available. Sticky -- only ever written 1.0, and
#: ``pt_acc`` is zeroed once per wave, so it survives the several ``pt_shade``
#: launches one path takes.
#:
#: A path that never sets it is deterministic GIVEN ITS JITTER: an unlit
#: transparent stack (pass-through at probability 1), an unlit opaque absorb,
#: and an escape to the background or environment map. Those are the pixels
#: adaptive sampling is allowed to stop early -- the half-buffer error
#: estimate cannot tell "converged at zero" from "has not found the light
#: yet", so the host requires this to be zero as well (see
#: ``path_tracer._pt_active_pixels``).
_PT_ACC_STOCH = 9

# Device-side truncation tallies, read back by the host once per wave and fed
# through ``truncation.record_truncation`` (ceilings are counted, not silent).
PT_STATS_WIDTH = 4
PT_STAT_TRUNC_SURFACES = 0
PT_STAT_SHELL_RING = 1

# Russian-roulette survival floor: a path is never continued with less than
# this probability, bounding the throughput amplification at 1/floor.
_PT_RR_FLOOR = 0.05

_INV_PI = 0.3183098861837907

# rs_sca column 5 (unused by the shared traverse kernel, which reads only
# columns 1/2/4): the solid-angle pdf of the last scatter direction, for the
# power-heuristic MIS weight applied when a BSDF-sampled path finds an
# emitter (an emissive triangle, or the environment map at escape).
#   < 0   camera segment -- the path has never scattered; emission weight 1.
#   == 0  the last scatter was a delta lobe (refraction, a tinted pane) or a
#         vertex that runs no surface NEE (authored appearance, a circuit):
#         emission weight 1 -- next-event estimation never covered it.
#   > 0   the lobe-mixture pdf of the sampled direction at a lit vertex that
#         ran the NEE block: emission there is MIS-weighted against it.
# Pass-through crossings keep the value (the ray, and with it the pdf at its
# origin vertex, continues unchanged).
_SCA_PREV_PDF = 5

# rs_int columns.  The shared traverse kernel touches only columns 0-4
# (bounces_left, processed, status, num_hits, max_bounces); the four columns
# after them belong to ``pt_shade`` alone: the closed-shell opacity ring --
# the ``tri_shell`` surface ids the camera segment is currently INSIDE of,
# -1 marking an empty slot.  Entering a declared closed shell composites the
# crossing and stores the id; the matching exit crossing finds the id,
# removes it, and composites nothing.  That is the per-ray limit of the
# sheet route's coverage ceiling (``solid_shell_alpha``, sheets.py), which
# spends ``max(front, back)`` coverage per (pixel, surface) in depth order
# -- and like it the ring counts CROSSINGS, not containment: a ray crossing
# one shell four times (a torus hole) attenuates twice.  A camera ray inside
# more than four declared shells at once overflows the ring: the surplus
# crossing composites normally (erring toward the doubled attenuation every
# crossing produced before the ceiling existed) and is tallied in
# ``pt_stats[PT_STAT_SHELL_RING]``.
PT_INT_WIDTH = 9
_INT_RING0 = 5
_SHELL_RING_SLOTS = 4

# Next-event table entry kinds (column 0 of ``nee_ref``; the table is built
# per render call by ``path_tracer._build_nee_tables``).
_NEE_LIGHT_ROW = 0
_NEE_EMISSIVE_TRI = 1
_NEE_ENV = 2
# Not a selectable entry: the ambient / hemisphere rows the kernel's
# deterministic fill visits, appended AFTER the ``E`` sampled entries so the
# CDF search (which takes ``num_nee``) never sees them. ``ref`` is the packed
# light row's index; they are stored in ascending row order, which is the
# order the linear scan they replace visited them in.
_NEE_AMBIENT_ROW = 3
# Not a selectable entry either: the AUTHORED-appearance branch's own light-row
# table (roadmap section 6a-bis), appended after the ambient tail with its own
# self-normalised CDF occupying the matching span of ``nee_cdf``. It exists
# only when the host chose the sampled authored mode, and nothing else ever
# reads it -- the sampled-entry search is bounded by ``num_nee`` and the
# ambient fill by ``amb_count``. It holds the light rows an authored stage
# SUMS: every direction-carrying row with power, a ``RectAreaLight``'s cell
# rows included, whether or not the quad path withdrew those rows from the
# sampled table above (an authored material lights from the rows either way,
# so its table must not follow that withdrawal).
_NEE_AUTHORED_ROW = 4

# Word layout of the ``nee_meta`` f32 vector (integer-valued words carry
# exact small ints; decoded with ``+ 0.5`` casts).
NEE_META_WIDTH = 20
_NM_COUNT = 0  # entries in nee_cdf / nee_ref (0 = no next-event sampling)
_NM_ENV_SHARE = 1  # env entry's selection probability (0 = env NEE off)
_NM_LIGHT_SAMPLES = 2  # pt_light_samples
_NM_ENV_OFF = 3  # env map placement in the shared texel buffer ...
_NM_ENV_W = 4  # ... and its dimensions (0 = no environment map)
_NM_ENV_H = 5
_NM_ENV_INTENSITY = 6
_NM_ENV_CDF_H = 7  # env CDF bin-grid dimensions
_NM_ENV_CDF_W = 8
_NM_AOV = 9  # 1 = accumulate the denoiser's albedo/normal AOVs (pt_aov)
_NM_FAR_CLIP = 10  # camera.far in world units (0 = no far plane)
_NM_AMBIENT_PACKED = 11  # 1 = the ambient rows ride nee_ref's tail ...
_NM_AMBIENT_COUNT = 12  # ... and there are this many of them
_NM_ANIM_SEED = 13  # 1 = the sampler key includes the frame (pt_animated_seed)
# Light-tree selection (light_tree.py). Off, the kernel takes the flat CDF
# path byte for byte and the tree tensors are [1, 1, ...] placeholders.
_NM_TREE_ON = 14  # 1 = select finite entries by descending the light tree
_NM_TREE_MIX = 15  # P(tree) = finite power / total power; the rest is the
#                    position-independent infinite list (directional + env),
#                    which is what keeps the mixture unbiased
_NM_INF_COUNT = 16  # entries in nee_inf_cdf / nee_inf_ref
# First primitive index of the synthetic RectAreaLight quads this render call
# appended (``area_light_quads``; 1 << 30 when it appended none, which is past
# any primitive a batch can hold). One compare against it is the whole
# camera-invisibility test AND the gate on the per-emitter falloff multiplier
# in ``pt_emit_falloff``, so an ordinary emissive triangle takes neither branch
# and is bit-identical to what it was before area-light quads existed.
_NM_QUAD_BASE = 17
# The authored-appearance branch's sampled mode (roadmap section 6a-bis). Both
# words are read only inside that branch and only when the kernel was compiled
# with ``auth_sampled`` on, so a lit-only scene never loads them.
_NM_AUTHORED_SAMPLES = 18  # rows the authored branch draws per crossing ...
_NM_AUTHORED_COUNT = 19  # ... out of this many entries in the authored table

# Per-path AOV row (``pt_aov``), accumulated only when ``_NM_AOV`` says so
# (the tensor is a [1, PT_AOV_WIDTH] dummy otherwise -- every access is
# gated). The albedo and normal guides are throughput-weighted composites
# over the path's DELTA PREFIX -- the crossings before its first non-delta
# scatter: pass-through (a straight line) and refraction / the tinted pane
# (delta lobes) keep accumulating, a diffuse or GGX pick closes the prefix
# at its own crossing, and an escape credits the environment map (in-kernel,
# where the texels are) or the background (via the leftover weight, folded
# on the host where the prefill lives). On an opaque single-surface scene
# this IS the standard first-non-delta-vertex convention; on algan's
# translucent stacks it degrades to the flat-shaded composite, which is
# exactly the detail a denoiser should preserve.
PT_AOV_WIDTH = 10
_AOV_ALB = 0  # 3 columns: sum of thru.rgb * alpha * base color
_AOV_NRM = 3  # 3 columns: sum of thru.a * alpha * shading normal
_AOV_BGW = 6  # 3 columns: leftover throughput while open (background credit)
_AOV_CLOSED = 9  # 1 = the delta prefix has ended; accumulate nothing more


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

    Callers that honour ``pt_animated_seed`` pass ``f * animated_seed``, which
    is the frame itself when the flag is 1 and frame 0 for every frame when it
    is 0 (see the module docstring's sampler section).
    """
    return _pt_hash_combine(ti.cast(f, ti.u32), ti.cast(pixel, ti.u32))


@ti.func
def _pt_path_seed(seed_root: ti.u32, key: ti.u32) -> ti.u32:
    """The PATH half of every draw's seed: the part that depends only on
    ``(pt_seed, frame-or-0, pixel)`` and is therefore constant for the whole
    path.

    It is split out so a shading thread hashes it ONCE and every draw the
    path makes only hashes its own ``pair`` on top (``_pt_pair_seed``); the
    combine used to be nested the other way round
    (``combine(seed_root, combine(key, pair))``), which put a ``pair``
    -dependent term inside both hashes and made the whole seed unhoistable.
    Ordering the two combines this way is what makes the per-path half
    loop-invariant -- and it moves every sample, which is why it landed with
    the section 5 re-baseline (roadmap section 0.2).
    """
    return _pt_hash_combine(seed_root, key)


@ti.func
def _pt_pair_seed(path_seed: ti.u32, pair: ti.i32) -> ti.u32:
    """The per-DIMENSION-PAIR half of a draw's seed. ``_pt_hash_combine``
    avalanches its second argument, so distinct pairs still decorrelate into
    independent sequences from one shared ``path_seed``.
    """
    return _pt_hash_combine(path_seed, ti.cast(pair, ti.u32))


@ti.func
def pt_sample_2d_seeded(path_seed: ti.u32, pair: ti.i32,
                        sample_index: ti.i32) -> ti.math.vec2:
    """``pt_sample_2d`` with the per-path seed already hoisted (see
    ``_pt_path_seed``) -- what every draw inside ``pt_shade`` calls.
    """
    pair_seed = _pt_pair_seed(path_seed, pair)
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


@ti.func
def pt_sample_2d(seed_root: ti.u32, key: ti.u32, pair: ti.i32,
                 sample_index: ti.i32) -> ti.math.vec2:
    """Sample ``pair`` of the pixel's padded Sobol sequence at
    ``sample_index``: Owen-shuffled index, Owen-scrambled (0,2) point.

    Any prefix of the returned sequence is well stratified, which is what
    makes progressive rendering (waves) and future adaptive sampling sound;
    distinct ``(key, pair)`` values decorrelate into independent sequences.

    The unhoisted spelling, for callers that draw once (the test probe): the
    kernels hold ``_pt_path_seed``'s result and call
    ``pt_sample_2d_seeded``.
    """
    return pt_sample_2d_seeded(_pt_path_seed(seed_root, key), pair,
                               sample_index)


@ti.func
def _pt_rng_seeded(path_seed: ti.u32, sample_index: ti.i32,
                   salt_a: ti.i32, salt_b: ti.i32) -> ti.f32:
    """White-noise uniform in [0, 1) for the unbounded-count decisions
    (per-light shadow jitter): a pure hash of the path identity plus two
    salts, so it is exactly as reproducible as the Sobol samples without
    consuming a dimension pair. Takes the hoisted per-path seed, which is
    the same value ``pt_sample_2d_seeded`` takes.
    """
    h = _pt_hash_combine(path_seed, ti.cast(sample_index, ti.u32))
    h = _pt_hash_combine(h, ti.cast(salt_a, ti.u32))
    h = _pt_hash_combine(h, ti.cast(salt_b, ti.u32))
    return ti.cast(h >> 8, ti.f32) * (1.0 / 16777216.0)


# Self-intersection offsetting (Wachter & Binder, "A Fast and Robust Method
# for Avoiding Self-Intersection", Ray Tracing Gems 2019 ch. 6). The constants
# are theirs: below ``_OFS_ORIGIN`` in magnitude a coordinate is offset by an
# absolute ``_OFS_FLOAT`` (float spacing near zero is finer than any useful
# world epsilon), above it by ``_OFS_INT`` ULPs, which scales with the point's
# own magnitude exactly as the representable spacing does.
_OFS_ORIGIN = 1.0 / 32.0
_OFS_FLOAT = 1.0 / 65536.0
_OFS_INT = 256.0


@ti.func
def _pt_offset_ray_origin(p, n):
    """Move hit point ``p`` off the surface along ``n`` by a SCALE-AWARE
    epsilon, and return the spawn origin.

    The fixed ``10 * min_hit_distance`` (1e-3 world units) this replaces was
    wrong in both directions: acne on a scene authored at large coordinates,
    where 1e-3 is below the float spacing of the hit point, and light leaking
    through thin geometry on one authored at small coordinates, where 1e-3 is
    a visible distance. Offsetting in INTEGER float space instead ties the
    step to the representable spacing at ``p``, so it is the smallest step
    that provably changes the coordinate whatever the scene's scale.

    ``n`` points to the side the ray leaves from; each call site keeps its own
    convention (the geometric normal flipped toward the outgoing direction,
    or the ray direction itself for a zero-thickness pane).
    """
    out = ti.math.vec3(0.0, 0.0, 0.0)
    for k in ti.static(range(3)):
        off_i = ti.cast(_OFS_INT * n[k], ti.i32)
        if p[k] < 0.0:
            off_i = -off_i
        p_i = ti.bit_cast(ti.bit_cast(p[k], ti.i32) + off_i, ti.f32)
        if ti.abs(p[k]) < _OFS_ORIGIN:
            out[k] = p[k] + _OFS_FLOAT * n[k]
        else:
            out[k] = p_i
    return out


@ti.func
def _pt_shadow_tmax(sorigin, wi, ldist):
    """Shadow-ray max distance: the emitter end pulled back by the SAME
    scale-aware offset ``_pt_offset_ray_origin`` applies at the surface end,
    so a light sitting on geometry is not occluded by its own emitter and the
    pull-back scales with the scene the way the spawn offset does (it was a
    fixed ``20 * min_hit_distance``).

    The pull-back is measured as a difference of two nearby points, so it
    stays exact even when ``ldist`` is the 1e7 sentinel a directional row or
    an environment sample carries.
    """
    lp = sorigin + wi * ldist
    back = (lp - _pt_offset_ray_origin(lp, -wi)).dot(wi)
    return ldist - ti.max(back, 0.0)


@ti.func
def _pt_onb(n):
    """Orthonormal basis around unit ``n`` (Duff et al., JCGT 2017)."""
    sign = 1.0 if n[2] >= 0.0 else -1.0
    a = -1.0 / (sign + n[2])
    b = n[0] * n[1] * a
    t = ti.math.vec3(1.0 + sign * n[0] * n[0] * a, sign * b, -sign * n[0])
    bt = ti.math.vec3(b, sign + n[1] * n[1] * a, -n[1])
    return t, bt


@ti.func
def _pt_cosine_direction(n, u):
    """Cosine-weighted hemisphere direction about unit ``n`` (pdf cos/pi)."""
    r = ti.sqrt(u[0])
    phi = 6.2831853 * u[1]
    x = r * ti.cos(phi)
    y = r * ti.sin(phi)
    z = ti.sqrt(ti.max(1.0 - u[0], 0.0))
    t, bt = _pt_onb(n)
    return (t * x + bt * y + n * z).normalized()


@ti.func
def _pt_vndf_half_vector(wo_local, alpha, u):
    """Sample the GGX visible-normal distribution for local view ``wo_local``
    (z up, ``wo_local.z > 0``): the spherical-cap formulation of Dupuy &
    Benyoub 2023 -- no rejection, exact VNDF.
    """
    v = ti.math.vec3(alpha * wo_local[0], alpha * wo_local[1],
                     wo_local[2]).normalized()
    phi = 6.2831853 * u[0]
    z = (1.0 - u[1]) * (1.0 + v[2]) - v[2]
    s = ti.sqrt(ti.math.clamp(1.0 - z * z, 0.0, 1.0))
    c = ti.math.vec3(s * ti.cos(phi), s * ti.sin(phi), z) + v
    return ti.math.vec3(alpha * c[0], alpha * c[1],
                        ti.max(c[2], 1e-6)).normalized()


@ti.func
def _pt_smith_lambda(cos_theta, alpha):
    """Smith's Lambda for isotropic GGX: the exact G1/G2 the VNDF estimator
    wants, and -- since the section-5 change -- what every direct-lighting
    response in this renderer uses too. The deterministic stages keep their
    own ``k = (r+1)^2/8`` remap (``_smith_geometry``); nothing here calls it.
    """
    c2 = ti.math.clamp(cos_theta * cos_theta, 1e-8, 1.0)
    t2 = (1.0 - c2) / c2
    return 0.5 * (ti.sqrt(1.0 + alpha * alpha * t2) - 1.0)


@ti.func
def _pt_ggx_energy(f0, n_dot_v, roughness):
    """Single-scatter GGX directional albedo with Turquin multiple-scattering
    compensation folded in: ``E_ss(f0) * (1 + f0 (1 - E1)/E1)`` per channel,
    with both terms from the shared Karis split-sum fit (exactly the energy
    the deterministic glossy route uses).
    """
    e_ss = _env_brdf_approx(f0, n_dot_v, roughness)
    e1 = _env_brdf_approx(ti.math.vec3(1.0, 1.0, 1.0), n_dot_v, roughness)
    e1s = ti.math.clamp(e1[0], 1e-3, 1.0)
    return ti.min(e_ss * (1.0 + f0 * ((1.0 - e1s) / e1s)),
                  ti.math.vec3(1.0, 1.0, 1.0))


@ti.func
def _pt_pick_nee_entry(nee_cdf: ti.template(), n, u):
    """Binary-search the combined light table's CDF: returns the entry index
    whose cumulative bracket contains ``u`` and that entry's selection
    probability (the CDF difference).
    """
    lo = 0
    hi = n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if u < nee_cdf[mid]:
            hi = mid
        else:
            lo = mid + 1
    prev = 0.0
    if lo > 0:
        prev = nee_cdf[lo - 1]
    return lo, nee_cdf[lo] - prev


@ti.func
def _pt_pick_authored_row(nee_cdf: ti.template(), base, n, u):
    """Binary-search the AUTHORED branch's own light-row CDF.

    The same search as :func:`_pt_pick_nee_entry` over the span
    ``[base, base + n)`` of ``nee_cdf``, which the host wrote as a
    self-normalised CDF of its own (it ends at 1.0, so ``u`` in ``[0, 1)``
    lands inside it and the first bracket's ``prev`` is 0). Returns the entry
    index RELATIVE to ``base``, and its selection probability.

    ``base`` is the CDF's base, which is NOT the rows' base: the authored table
    follows both the sampled entries and the ambient tail in ``nee_ref``, but
    only the sampled entries in ``nee_cdf`` -- the ambient rows are the
    deterministic fill and have no selection probability at all. The caller
    holds both and adds the right one to ``k``.
    """
    lo = base
    hi = base + n - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if u < nee_cdf[mid]:
            hi = mid
        else:
            lo = mid + 1
    prev = 0.0
    if lo > base:
        prev = nee_cdf[lo - 1]
    return lo - base, nee_cdf[lo] - prev


class _SampledLightView(tuple):
    """``light_pos`` / ``light_col`` re-indexed by a per-thread row map.

    The authored-appearance branch's sampled mode (roadmap section 6a-bis)
    hands ``_run_frag_pipeline`` a few SLOTS where it used to hand it every
    packed light row, and each sampled slot's radiance carries that draw's
    Monte Carlo weight. Neither the pipeline nor any stage is touched: a view
    in ``ArenaView``'s idiom (a tuple subclass, so ``ti.static`` passes it
    through and it can be bound to a name in kernel scope) rewrites
    ``view[tl, slot, c]`` into ``inner[tl, rows[slot], c]``, multiplied by
    ``scale[slot]`` for the three radiance channels only.

    ``rows`` and ``scale`` are matrix-typed ``Expr``s -- ``ti.Vector`` locals
    of the calling kernel -- so they are indexed through the compiler's own
    subscript builder rather than with ``[]``, exactly as ``ArenaView`` indexes
    the arena buffer. They are filled per crossing and read at every use, which
    is why the view is built once and never rebuilt.

    Scaling only channels 0-2 is what makes the weight ride the RADIANCE rather
    than the visibility vector: ``_light_vis`` is compiled out entirely when
    shadows are off, so a weight parked there would be dead-code-eliminated and
    every shadowless path-traced render would be silently wrong. Every other
    packed column -- the type id, the decay and range, the cone axis and
    cosines, the hemisphere ground colour, the power fraction -- passes through
    unscaled, and every light model ``_light_eval`` evaluates is linear in the
    three it scales.

    Read-only, deliberately: a scaled read returns an rvalue, where an
    ``ArenaView`` subscript is an lvalue. Nothing downstream of
    ``_run_frag_pipeline`` writes a light row.
    """

    __slots__ = ()

    def __new__(cls, inner, rows, scale=None):
        return super().__new__(cls, (inner, rows, scale))

    @property
    def inner(self):
        return tuple.__getitem__(self, 0)

    @property
    def rows(self):
        return tuple.__getitem__(self, 1)

    @property
    def scale(self):
        return tuple.__getitem__(self, 2)

    @property
    def shape(self):
        # Forwarded so ``f % light_pos.shape[0]`` and the compact-vs-extended
        # row test ``light_col.shape[2] > 3`` still read the real table's.
        return self.inner.shape

    def __getitem__(self, idx):
        tl, slot, c = idx
        val = self.inner[tl, _ti_impl.subscript(None, self.rows, slot), c]
        if self.scale is None:
            return val
        w = _ti_impl.subscript(None, self.scale, slot)
        # Every channel index in this package is a Python literal, so the
        # gate resolves at build time; a user stage computing one at runtime
        # gets the same rule as a select.
        if isinstance(c, int):
            return val * w if c < 3 else val
        return val * ti.select(c < 3, w, 1.0)


#: Descent / upward-walk step ceiling. The tree is binary over at most a few
#: thousand entries and the SAOH keeps it near-balanced, so this is a
#: watchdog against a malformed tree hanging a GPU, not a working limit.  The
#: two walks share it so that an entry past the ceiling reads as probability
#: zero at BOTH MIS ends rather than at one of them.
_PT_LT_MAX_DEPTH = 1024


@ti.func
def _pt_lt_importance(lt_node_f: ti.template(), row, node, p):
    """Conty-Kulla node importance at shading point ``p``.

    ``power * cos(theta') / d^decay``, where ``theta'`` is how far the node's
    orientation cone still has to turn to face ``p`` once its normal spread
    (``theta_o``) and the angle the node's bounds subtend from ``p``
    (``theta_u``) are both credited to it, and the whole node scores zero once
    that residual passes the emission spread ``theta_e`` -- a back-facing
    subtree is skipped instead of being picked in proportion to its power.
    Every angle is carried as its cosine and sine and every subtraction goes
    through the angle-subtraction identities, so no ``acos``/``asin``/``cos``
    runs per node visit; that alone is half the descent's cost -- 844 ms of
    ``pt_shade`` device time on a bare 32-light ring against 674 ms, over a
    flat-CDF baseline of 504 ms.

    The exponent comes off the node (``LT_DECAY``) rather than being the
    inverse square a physical renderer could assume: Algan's light rows
    default to ``decay = 0`` and genuinely do not fade with distance, so a
    hard-coded ``1/d^2`` would aim the sampler at the near lights while every
    light contributes the same -- measured *worse* than the flat CDF. An
    emissive triangle always carries 2, which is its area-to-solid-angle
    Jacobian and not an authored choice.

    **Position only, deliberately.** PBRT-v4's ``LightBounds::Importance``
    also multiplies by a bound on the receiver's own cosine, and the shading
    normal *is* in registers at the next-event call site -- but the MIS pdf
    query at a BSDF hit has to evaluate this same function at the PREVIOUS
    vertex, and the path state carries that vertex's position (``rs_ro``) and
    not its normal.  Both ends calling one function is what makes the MIS
    weights sum to one, so the normal term is dropped at both rather than
    used at one; carrying it would cost three more ``rs_sca`` columns (only
    one is free) and a wider ``_PT_BYTES_PER_SLOT``. See
    ``DESIGN_path_tracer_roadmap.md`` section 6b.
    """
    power = lt_node_f[row, node, LT_POWER]
    imp = 0.0
    if power > 0.0:
        bmin = ti.math.vec3(lt_node_f[row, node, LT_BMIN + 0],
                            lt_node_f[row, node, LT_BMIN + 1],
                            lt_node_f[row, node, LT_BMIN + 2])
        bmax = ti.math.vec3(lt_node_f[row, node, LT_BMAX + 0],
                            lt_node_f[row, node, LT_BMAX + 1],
                            lt_node_f[row, node, LT_BMAX + 2])
        axis = ti.math.vec3(lt_node_f[row, node, LT_AXIS + 0],
                            lt_node_f[row, node, LT_AXIS + 1],
                            lt_node_f[row, node, LT_AXIS + 2])
        cos_o = lt_node_f[row, node, LT_COS_THETA_O]
        sin_o = lt_node_f[row, node, LT_SIN_THETA_O]
        cos_e = lt_node_f[row, node, LT_COS_THETA_E]
        decay = lt_node_f[row, node, LT_DECAY]
        center = (bmin + bmax) * 0.5
        radius = 0.5 * (bmax - bmin).norm()
        to_p = p - center
        d = to_p.norm()
        # Inside the node's bounding sphere the point could be arbitrarily
        # close to an emitter, so the falloff reverts to the sphere's own
        # scale and the bounds subtend every direction.
        d2 = ti.max(d * d, ti.max(radius * radius, 1e-12))
        cos_u = 0.0
        sin_u = 1.0
        wi = axis
        if d > radius:
            sin_u = ti.math.clamp(radius / d, 0.0, 1.0)
            cos_u = ti.sqrt(ti.max(1.0 - sin_u * sin_u, 0.0))
        if d > 1e-12:
            wi = to_p / d
        cos_t = ti.math.clamp(axis.dot(wi), -1.0, 1.0)
        sin_t = ti.sqrt(ti.max(1.0 - cos_t * cos_t, 0.0))
        # cos(theta - theta_o) and sin(theta - theta_o), both clamped at 0
        # (PBRT-v4's CosSubClamped / SinSubClamped): the angle-subtraction
        # identities, so no inverse trigonometry runs per node visit.
        cos_x = 1.0
        sin_x = 0.0
        if cos_t <= cos_o:
            cos_x = cos_t * cos_o + sin_t * sin_o
            sin_x = sin_t * cos_o - cos_t * sin_o
        cos_p = 1.0
        if cos_x <= cos_u:
            cos_p = cos_x * cos_u + sin_x * sin_u
        if cos_p > cos_e:
            falloff = 1.0
            if decay > 0.0:
                if decay == 2.0:
                    falloff = 1.0 / d2
                else:
                    falloff = ti.pow(d2, -0.5 * decay)
            imp = power * cos_p * falloff
    return imp


@ti.func
def _pt_lt_descend(lt_node_f: ti.template(), lt_node_i: ti.template(),
                   row, p, u):
    """Descend the light tree from the root to a leaf with one random number.

    At each node the two children are scored by ``_pt_lt_importance``, one is
    picked in proportion, and ``u`` is **rescaled** into the chosen child's
    bracket -- so a single stratified draw (the entry-select Sobol dimension)
    stratifies the whole descent instead of one level of it. Returns
    ``(leaf node, entry index, probability)``.

    A node whose two children both score zero -- which happens, because a
    parent's union box and union cone can face ``p`` when neither child does
    -- splits the draw evenly instead of giving up. That spends a shadow ray
    that will return next to nothing, and it buys the property the MIS
    weights rest on: the descent is a genuine probability distribution over
    the leaves, with no mass quietly lost part-way down.
    """
    node = 0
    prob = 1.0
    depth = 0
    while (lt_node_i[row, node, LT_LEFT] >= 0) and (depth < _PT_LT_MAX_DEPTH):
        c0 = lt_node_i[row, node, LT_LEFT]
        c1 = lt_node_i[row, node, LT_RIGHT]
        i0 = _pt_lt_importance(lt_node_f, row, c0, p)
        i1 = _pt_lt_importance(lt_node_f, row, c1, p)
        total = i0 + i1
        # p0 and p1 are formed the SAME way the upward walk forms them
        # (i / total, never 1 - p0): the two must agree bit for bit or the
        # MIS weights stop summing to one.
        p0 = 0.5
        p1 = 0.5
        if total > 0.0:
            p0 = i0 / total
            p1 = i1 / total
        if u < p0:
            node = c0
            prob *= p0
            u = ti.math.clamp(u / ti.max(p0, 1e-12), 0.0, 0.99999994)
        else:
            node = c1
            prob *= p1
            u = ti.math.clamp((u - p0) / ti.max(p1, 1e-12), 0.0, 0.99999994)
        depth += 1
    entry = -1
    if lt_node_i[row, node, LT_LEFT] < 0:
        entry = lt_node_i[row, node, LT_RIGHT]
    else:
        prob = 0.0
    return node, entry, prob


@ti.func
def _pt_lt_pmf(lt_node_f: ti.template(), lt_node_i: ti.template(),
               row, leaf, p):
    """Probability the descent from ``p`` reaches ``leaf``, walked upward.

    The MIS pdf at a BSDF hit needs the probability next-event estimation
    would have had of choosing this emitter FROM THE PREVIOUS VERTEX, which
    a spatially-varying sampler can no longer read out of a table. Walking
    the stored parent chain costs one importance pair per level, against a
    whole re-descent's worth of comparisons -- and it evaluates the same
    expressions ``_pt_lt_descend`` does, in the same order per level.
    """
    node = leaf
    prob = 1.0
    steps = 0
    reached = 0
    while steps <= _PT_LT_MAX_DEPTH:
        par = lt_node_i[row, node, LT_PARENT]
        if par < 0:
            reached = 1
            break
        c0 = lt_node_i[row, par, LT_LEFT]
        c1 = lt_node_i[row, par, LT_RIGHT]
        i0 = _pt_lt_importance(lt_node_f, row, c0, p)
        i1 = _pt_lt_importance(lt_node_f, row, c1, p)
        total = i0 + i1
        p0 = 0.5
        p1 = 0.5
        if total > 0.0:
            p0 = i0 / total
            p1 = i1 / total
        if node == c0:
            prob *= p0
        else:
            prob *= p1
        node = par
        steps += 1
    if reached == 0:
        prob = 0.0
    return prob


@ti.func
def _pt_ggx_ndf(n_dot_h, alpha):
    """Isotropic GGX normal distribution with ``alpha`` = roughness^2 -- the
    same parameterisation the VNDF sampler and ``_pt_smith_lambda`` use, so
    an evaluated pdf matches the sampled one exactly.
    """
    a2 = alpha * alpha
    d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0
    return a2 / ti.max(_PI * d * d, 1e-12)


@ti.func
def _pt_lit_lobes(pid, params: ti.template(), f, prim, albedo3, metalness,
                  rough, ior, T, shade_n, rd):
    """Continuation-lobe energies of a physically-integrated (lit) hit:
    ``(e_diff, e_spec, e_trans, f0, rough_eff)``. The single source for the
    sampled continuation AND for every direct-lighting response (roadmap
    section 5) -- one BSDF, so MIS weights sum to one and a light row and an
    emissive triangle of matched radiance light a surface identically.

    ``rough_eff`` is the GGX roughness the caller must carry downstream: it
    is the crossing's own ``rough`` for every pipeline except phong, whose
    highlight is authored as a Blinn-Phong exponent instead.
    """
    one3 = ti.math.vec3(1.0, 1.0, 1.0)
    e_diff = ti.math.vec3(0.0, 0.0, 0.0)
    e_spec = ti.math.vec3(0.0, 0.0, 0.0)
    e_trans = ti.math.vec3(0.0, 0.0, 0.0)
    f0 = ti.math.vec3(0.0, 0.0, 0.0)
    rough_eff = rough
    # |cos|, not cos. The GGX reflection lobe lives on whichever side of the
    # interface the ray arrived from, and ``shade_n`` is the surface's own
    # declared side: a one-sided solid keeps its OUTWARD normal when a
    # refracted path hits its exit face from inside
    # (``_sided_shading_normal``), so ``shade_n . -rd`` is then negative.
    # Clamping that to 1e-4 read a head-on interior hit as a grazing one and
    # handed the specular lobe a Fresnel of ~1, which is what made the exit
    # face of a glass prism choose a continuation it could never sample (see
    # ``spec_n`` in ``pt_shade``).
    n_dot_v = ti.max(ti.abs(shade_n.dot(-rd)), 1e-4)
    if pid == _MID_LAMBERT:
        # The lambert stage has no specular lobe at all.
        e_diff = albedo3
    elif pid == _MID_PHONG:
        # Phong under the path tracer is GGX. Its Blinn-Phong highlight was
        # the last thing evaluated by a formula the continuation could not
        # sample, and deleting that formula without giving the material a
        # lobe would silently drop ``specular`` / ``shininess`` -- exactly
        # the failure the fallback must not have (roadmap section 9). The
        # exponent maps to the GGX width by the standard
        # ``alpha = sqrt(2 / (s + 2))``, and the authored specular colour is
        # F0; the highlight moves and softens, which is the visible change
        # the limitations page calls out.
        tmp = f % params.shape[0]
        f0 = ti.math.vec3(params[tmp, prim, 4], params[tmp, prim, 5],
                          params[tmp, prim, 6])
        shininess = ti.max(params[tmp, prim, 7], 1e-3)
        rough_eff = ti.math.clamp(
            ti.sqrt(ti.sqrt(2.0 / (shininess + 2.0))), 0.03, 1.0)
        e_spec = _pt_ggx_energy(f0, n_dot_v, rough_eff)
        e_diff = albedo3 * (one3 - e_spec)
    else:
        tm = f % params.shape[0]
        met = ti.math.clamp(ti.max(metalness, 0.0), 0.0, 1.0)
        diel_f0 = ti.math.vec3(0.04, 0.04, 0.04)
        if pid == _MID_PHYSICAL:
            ior_m = params[tm, prim, 12]
            ratio = (ior_m - 1.0) / ti.max(ior_m + 1.0, 1e-4)
            diel_f0 = ti.math.vec3(
                params[tm, prim, 14], params[tm, prim, 15],
                params[tm, prim, 16]) \
                * (ratio * ratio * params[tm, prim, 13])
        f0 = diel_f0 * (1.0 - met) + albedo3 * met
        e_spec = _pt_ggx_energy(f0, n_dot_v, rough)
        _R3, diel_pass = _material_reflectance(
            rd, shade_n, ti.max(metalness, 0.0), ior, albedo3, T)
        e_trans = albedo3 * (diel_pass * T)
        e_diff = albedo3 * ((1.0 - met) * (1.0 - T)) * (one3 - e_spec)
    return e_diff, e_spec, e_trans, f0, rough_eff


@ti.func
def _pt_lit_f_pdf(e_diff, e_spec, f0, rough, shade_n, rd, wi,
                  w_pass, w_diff, w_spec, w_trans):
    """Physical BSDF response of a lit vertex toward ``wi`` and the pdf with
    which its continuation sampler generates ``wi``.

    Returns ``(f_cos, pdf)``: the BRDF times the surface cosine (diffuse
    ``e_diff / pi``, plus exact GGX with the sampled lobe's own alpha,
    Fresnel and Turquin compensation), and the lobe-mixture solid-angle
    density (cosine and VNDF terms weighted by the same lobe-selection
    weights the continuation draws from; the pass/transmission lobes are
    deltas and add no density). Both ends of every MIS pair -- next-event
    samples toward emitters and BSDF paths that find them -- use this one
    function, which is what makes the power-heuristic weights sum to one.
    """
    f_cos = ti.math.vec3(0.0, 0.0, 0.0)
    pdf = 0.0
    w_sum = w_pass + w_diff + w_spec + w_trans
    cos_i = shade_n.dot(wi)
    if (cos_i > 1e-6) and (w_sum > 1e-6):
        f_cos = e_diff * (_INV_PI * cos_i)
        pdf = (w_diff / w_sum) * (_INV_PI * cos_i)
        if w_spec > 0.0:
            one3 = ti.math.vec3(1.0, 1.0, 1.0)
            v = -rd
            h = (v + wi).normalized()
            n_dot_v = ti.max(shade_n.dot(v), 1e-4)
            n_dot_h = ti.math.clamp(shade_n.dot(h), 0.0, 1.0)
            v_dot_h = ti.max(v.dot(h), 1e-4)
            a_g = ti.max(rough * rough, 1e-4)
            d = _pt_ggx_ndf(n_dot_h, a_g)
            lam_v = _pt_smith_lambda(n_dot_v, a_g)
            lam_l = _pt_smith_lambda(cos_i, a_g)
            g1_v = 1.0 / (1.0 + lam_v)
            g2 = 1.0 / (1.0 + lam_v + lam_l)
            fres = f0 + (one3 - f0) * ti.pow(1.0 - v_dot_h, 5.0)
            e1 = _env_brdf_approx(one3, n_dot_v, rough)
            e1s = ti.math.clamp(e1[0], 1e-3, 1.0)
            comp = one3 + f0 * ((1.0 - e1s) / e1s)
            f_cos += fres * comp * (d * g2 / ti.max(4.0 * n_dot_v, 1e-6))
            pdf += (w_spec / w_sum) * (g1_v * d / ti.max(4.0 * n_dot_v, 1e-6))
    return f_cos, pdf


@ti.func
def _pt_env_pdf_sa(rd, env_cdf: ti.template(), cdf_h, cdf_w):
    """Solid-angle density with which ``_pt_env_sample`` generates unit
    direction ``rd``: the direction's bin probability (marginal times
    conditional CDF differences) over the bin's uniform (u, v) footprint,
    through the equirect Jacobian ``2 pi^2 sin(theta)``.
    """
    u = ti.atan2(rd[2], rd[0]) * (0.5 / _PI) + 0.5
    v = 0.5 - ti.asin(ti.math.clamp(rd[1], -1.0, 1.0)) / _PI
    x = ti.math.clamp(ti.cast(u * cdf_w, ti.i32), 0, cdf_w - 1)
    y = ti.math.clamp(ti.cast(v * cdf_h, ti.i32), 0, cdf_h - 1)
    marg_prev = 0.0
    if y > 0:
        marg_prev = env_cdf[y - 1, cdf_w]
    p_y = env_cdf[y, cdf_w] - marg_prev
    cond_prev = 0.0
    if x > 0:
        cond_prev = env_cdf[y, x - 1]
    p_x = env_cdf[y, x] - cond_prev
    sin_t = ti.sqrt(ti.max(1.0 - rd[1] * rd[1], 1e-8))
    return p_y * p_x * ti.cast(cdf_h * cdf_w, ti.f32) \
        / (2.0 * _PI * _PI * ti.max(sin_t, 1e-4))


@ti.func
def _pt_env_sample(env_cdf: ti.template(), cdf_h, cdf_w, u1, u2):
    """Draw an environment direction from the 2D luminance CDF: binary-search
    the marginal (column ``cdf_w``) with ``u1`` for the row, that row's
    conditional with ``u2`` for the column, and reuse each search's bracket
    remainder as the uniform intra-bin jitter (so the pdf above is exact and
    no extra dimensions are consumed). Returns ``(dir, pdf_sa)``.
    """
    lo = 0
    hi = cdf_h - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if u1 < env_cdf[mid, cdf_w]:
            hi = mid
        else:
            lo = mid + 1
    y = lo
    marg_prev = 0.0
    if y > 0:
        marg_prev = env_cdf[y - 1, cdf_w]
    p_y = env_cdf[y, cdf_w] - marg_prev
    ry = ti.math.clamp((u1 - marg_prev) / ti.max(p_y, 1e-12), 0.0, 1.0)
    lo = 0
    hi = cdf_w - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if u2 < env_cdf[y, mid]:
            hi = mid
        else:
            lo = mid + 1
    x = lo
    cond_prev = 0.0
    if x > 0:
        cond_prev = env_cdf[y, x - 1]
    p_x = env_cdf[y, x] - cond_prev
    rx = ti.math.clamp((u2 - cond_prev) / ti.max(p_x, 1e-12), 0.0, 1.0)
    u = (ti.cast(x, ti.f32) + rx) / ti.cast(cdf_w, ti.f32)
    v = (ti.cast(y, ti.f32) + ry) / ti.cast(cdf_h, ti.f32)
    theta = _PI * v
    phi = 2.0 * _PI * (u - 0.5)
    sin_t = ti.sin(theta)
    direction = ti.math.vec3(ti.cos(phi) * sin_t, ti.cos(theta),
                             ti.sin(phi) * sin_t)
    pdf = p_y * p_x * ti.cast(cdf_h * cdf_w, ti.f32) \
        / (2.0 * _PI * _PI * ti.max(sin_t, 1e-4))
    return direction, pdf


@ti.func
def _pt_emissive_sample(tri_pos: ti.template(), f, prim, u1, u2):
    """Uniform point on emissive triangle ``prim`` at frame ``f``: returns
    ``(point, front_normal, area, w0, w1, w2)`` with the frame's own area
    (an animated emitter keeps an exact area-measure pdf) and the geometric
    front normal ``(v1-v0) x (v2-v0)`` that decides the emitting side.
    """
    tp = f % tri_pos.shape[0]
    v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                      tri_pos[tp, prim, 2])
    v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                      tri_pos[tp, prim, 5])
    v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                      tri_pos[tp, prim, 8])
    su = ti.sqrt(ti.max(u1, 0.0))
    w1 = 1.0 - su
    w2 = u2 * su
    w0 = 1.0 - w1 - w2
    point = v0 * w0 + v1 * w1 + v2 * w2
    ng = (v1 - v0).cross(v2 - v0)
    n_len = ng.norm()
    area = 0.5 * n_len
    normal = ti.math.vec3(0.0, 0.0, 0.0)
    if n_len > 1e-12:
        normal = ng / n_len
    return point, normal, area, w0, w1, w2


@ti.func
def _pt_nee_light_row(light_pos: ti.template(), light_col: ti.template(),
                      f, li, spos, n, u1, u2):
    """Radiometry + visibility target of one CDF-selected packed light row.

    Delta rows (point / spot / directional) keep ``_light_eval``'s
    centre-of-emitter radiometry -- exact stage brightness parity -- with the
    visibility ray jittered across the shadow softness by
    ``_pt_light_sample_point`` (deterministic-fan semantics). An area-sample
    row instead evaluates the radiometry AT a uniform point inside its own
    packed cell (falloff distance, range fade and the one-sided cosine all
    from the sampled point), which is what turns the deterministic K-row
    staircase into the continuous area integral in expectation. Returns
    ``(ld, lc, spec_w, wi_vis, ldist, valid)`` -- ``ld`` for the shading
    response, ``wi_vis``/``ldist`` for the visibility trace.
    """
    tl = f % light_pos.shape[0]
    ltype = 0
    if light_col.shape[2] > 3:
        ltype = ti.cast(light_col[tl, li, 3] + 0.5, ti.i32)
    ld = ti.math.vec3(0.0, 0.0, 0.0)
    lc = ti.math.vec3(0.0, 0.0, 0.0)
    spec_w = 1.0
    wi_vis = ti.math.vec3(0.0, 0.0, 0.0)
    ldist = 1e7
    valid = 0
    if ltype == _LT_AREA_SAMPLE:
        lp = ti.math.vec3(light_pos[tl, li, 0], light_pos[tl, li, 1],
                          light_pos[tl, li, 2])
        an = ti.math.vec3(light_col[tl, li, 6], light_col[tl, li, 7],
                          light_col[tl, li, 8])
        hu = light_col[tl, li, 9]
        hv = light_col[tl, li, 10]
        target = lp
        if (hu > 0.0) or (hv > 0.0):
            b1 = ti.math.vec3(light_col[tl, li, 12], light_col[tl, li, 13],
                              light_col[tl, li, 14])
            b2 = an.cross(b1)
            target = lp + b1 * (hu * (2.0 * u1 - 1.0)) \
                + b2 * (hv * (2.0 * u2 - 1.0))
        to_light = target - spos
        d = to_light.norm()
        if d > 1e-5:
            wi = to_light / d
            lc = ti.math.vec3(light_col[tl, li, 0], light_col[tl, li, 1],
                              light_col[tl, li, 2])
            # Falloff exactly as ``_light_eval``'s POINT/SPOT/AREA block,
            # with the sampled point's own distance.
            decay = light_col[tl, li, 4]
            if decay > 0.0:
                lc = lc / ti.pow(ti.max(d, 1e-4), decay)
            rng = light_col[tl, li, 5]
            if rng > 0.0:
                q = ti.math.clamp(d / rng, 0.0, 1.0)
                q2 = q * q
                fade = ti.math.clamp(1.0 - q2 * q2, 0.0, 1.0)
                lc = lc * (fade * fade)
            # One-sided cosine emission of the rectangle, at the sample.
            lc = lc * ti.max((-wi).dot(an), 0.0)
            ld = wi
            wi_vis = wi
            ldist = d
            valid = 1
    else:
        ld, lc, spec_w, _frac = _light_eval(light_pos, light_col, f, li,
                                            spos, n)
        wi_vis, ldist, valid = _pt_light_sample_point(
            light_pos, light_col, f, li, spos, u1, u2)
    return ld, lc, spec_w, wi_vis, ldist, valid


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
def pt_nee_pick_probe(nee_cdf: ti.types.ndarray(), n: ti.i32,
                      u: ti.types.ndarray(), out: ti.types.ndarray()):
    """Test probe: binary-search each ``u`` through a table CDF, writing
    ``out[i] = (entry index, selection probability)``. Exists so the
    next-event table search can be unit-tested without a render.
    """
    for i in range(u.shape[0]):
        entry, p = _pt_pick_nee_entry(nee_cdf, n, u[i])
        out[i, 0] = ti.cast(entry, ti.f32)
        out[i, 1] = p


@ti.kernel
def pt_light_tree_probe(lt_node_f: ti.types.ndarray(),
                        lt_node_i: ti.types.ndarray(), row: ti.i32,
                        pts: ti.types.ndarray(), u: ti.types.ndarray(),
                        out: ti.types.ndarray()):
    """Test probe: descend the tree, then walk back up from where it landed.

    ``out[i] = (entry, descent probability, upward PMF at the same point,
    leaf node)``. Columns 1 and 2 are the two ends of every emissive MIS
    pair -- they must be equal to float precision or the power-heuristic
    weights stop summing to one, which is the single property this whole
    structure has to preserve.
    """
    for i in range(u.shape[0]):
        p = ti.math.vec3(pts[i, 0], pts[i, 1], pts[i, 2])
        leaf, entry, prob = _pt_lt_descend(lt_node_f, lt_node_i, row, p, u[i])
        out[i, 0] = ti.cast(entry, ti.f32)
        out[i, 1] = prob
        out[i, 2] = _pt_lt_pmf(lt_node_f, lt_node_i, row, leaf, p)
        out[i, 3] = ti.cast(leaf, ti.f32)


@ti.kernel
def pt_light_tree_pmf_probe(lt_node_f: ti.types.ndarray(),
                            lt_node_i: ti.types.ndarray(), row: ti.i32,
                            pts: ti.types.ndarray(),
                            leaves: ti.types.ndarray(),
                            out: ti.types.ndarray()):
    """Test probe: ``out[i, j]`` is the selection probability of leaf ``j``
    from point ``i``. Summing a row over every leaf must give 1 -- the tree
    is a probability distribution over the entries, not just a way to find a
    bright one.
    """
    for i, j in ti.ndrange(pts.shape[0], leaves.shape[0]):
        p = ti.math.vec3(pts[i, 0], pts[i, 1], pts[i, 2])
        out[i, j] = _pt_lt_pmf(lt_node_f, lt_node_i, row, leaves[j], p)


@ti.kernel
def pt_env_sample_probe(env_cdf: ti.types.ndarray(), cdf_h: ti.i32,
                        cdf_w: ti.i32, u: ti.types.ndarray(),
                        out: ti.types.ndarray()):
    """Test probe: draw one environment direction per ``(u1, u2)`` row,
    writing ``out[i] = (dir.x, dir.y, dir.z, pdf, pdf re-evaluated from the
    direction)`` -- the sampling/evaluation pair every escape MIS weight
    needs to agree on.
    """
    for i in range(u.shape[0]):
        d, pdf = _pt_env_sample(env_cdf, cdf_h, cdf_w, u[i, 0], u[i, 1])
        out[i, 0] = d[0]
        out[i, 1] = d[1]
        out[i, 2] = d[2]
        out[i, 3] = pdf
        out[i, 4] = _pt_env_pdf_sa(d, env_cdf, cdf_h, cdf_w)


@ti.kernel
def pt_env_pdf_probe(env_cdf: ti.types.ndarray(), cdf_h: ti.i32,
                     cdf_w: ti.i32, dirs: ti.types.ndarray(),
                     out: ti.types.ndarray()):
    """Test probe: the sampler's solid-angle density at given directions,
    for quadrature checks that the pdf integrates to one over the sphere.
    """
    for i in range(dirs.shape[0]):
        d = ti.math.vec3(dirs[i, 0], dirs[i, 1], dirs[i, 2]).normalized()
        out[i] = _pt_env_pdf_sa(d, env_cdf, cdf_h, cdf_w)


@ti.kernel
def pt_offset_probe(points: ti.types.ndarray(), normals: ti.types.ndarray(),
                    out: ti.types.ndarray()):
    """Test probe: the spawn origin ``_pt_offset_ray_origin`` returns for each
    ``(point, normal)`` row. Exists so the scale-aware self-intersection
    offset can be unit-tested at several magnitudes without a render.
    """
    for i in range(points.shape[0]):
        p = ti.math.vec3(points[i, 0], points[i, 1], points[i, 2])
        n = ti.math.vec3(normals[i, 0], normals[i, 1], normals[i, 2])
        q = _pt_offset_ray_origin(p, n)
        for k in ti.static(range(3)):
            out[i, k] = q[k]


@ti.kernel
def pt_generate(num_slots: ti.i32, tile_pixels: ti.i32, sample_base: ti.i32,
                seed_root: ti.u32, animated_seed: ti.i32, time_start: ti.i32,
                width: ti.i32, height: ti.i32,
                half_screen_w: ti.f32, half_screen_h: ti.f32,
                cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
                pixel_basis_x: ti.types.ndarray(),
                pixel_basis_y: ti.types.ndarray(), near_clip: ti.f32,
                pix_list: ti.types.ndarray(),
                rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
                rs_sca: ti.types.ndarray(), rs_pix: ti.types.ndarray()):
    """Write each slot's jittered primary ray.

    Slot layout: ``slot = k * tile_pixels + p_local`` holds wave sample
    ``sample_base + k`` of the wave's ``p_local``-th pixel, so one wave puts
    every one of its pixels' next ``S`` samples in flight and ``pt_reduce``
    can walk a pixel's slots at stride ``tile_pixels``. The rest of the
    per-slot state is constant at generation and broadcast-filled by the host
    (the same coalesced-fill reasoning as the deterministic ``const_fill``
    path) -- except ``base_dist`` under a near clip, which varies per ray and
    is written here over the host's broadcast zero.

    ``pix_list[p_local]`` is the wave's pixel list: the GLOBAL flat cell
    ``(frame - time_start) * width * height + pixel`` each of the wave's
    ``tile_pixels`` columns renders. Adaptive sampling
    (``pt_error_target > 0``) hands a compacted list of the pixels that have
    not converged; the uniform loop hands the tile's identity list
    ``tile_start .. tile_start + tile_pixels``. ``rs_pix`` therefore stores
    the GLOBAL cell, and the traverse and shade kernels take
    ``ray_offset = 0`` -- there is no contiguous tile to offset by any more.

    ``tile_pixels`` is the number of pixels in THIS wave, which is also what
    makes the sampler's per-pixel prefix contiguous: every pixel alive in a
    wave has received exactly the same number of samples so far, so
    ``s_index = sample_base + slot // tile_pixels`` still enumerates
    ``0 .. n_p`` for each of them without gaps or repeats (sampler purity,
    ``DESIGN_path_tracer_roadmap.md`` contract 1).

    ``animated_seed`` is ``rt_settings.pt_animated_seed`` as 1 / 0 and folds
    the frame out of the sampler key when it is 0; ``pt_shade`` reads the same
    value from ``nee_meta[_NM_ANIM_SEED]`` and the two must agree.
    """
    pixels_per_frame = width * height
    for slot in range(num_slots):
        p_local = slot % tile_pixels
        s = sample_base + slot // tile_pixels
        g = pix_list[p_local]
        f_rel = g // pixels_per_frame
        p = g - f_rel * pixels_per_frame
        f = time_start + f_rel
        py = p // width
        px = p - py * width
        # The per-path half of the sampler seed, hoisted out of the draw the
        # same way ``pt_shade`` hoists it (roadmap section 0.2).
        path_seed = _pt_path_seed(seed_root, _pt_key(f * animated_seed, p))
        jitter = pt_sample_2d_seeded(path_seed, PAIR_PIXEL, s)
        ro, rd = _generate_ray(f, px, py, jitter[0], jitter[1],
                               half_screen_w, half_screen_h,
                               cam_origin, screen_point,
                               pixel_basis_x, pixel_basis_y)
        if near_clip > 0.0:
            # Near plane, identical to ``wavefront_generate_rays``: advance
            # the origin to the plane at ``near_clip`` along the camera's
            # forward axis (planar, like Three.js), and seed ``base_dist``
            # with the skipped distance so far-plane and screen-space widths
            # stay camera-relative rather than origin-relative.
            fwd = (ti.math.vec3(screen_point[f, 0], screen_point[f, 1],
                                screen_point[f, 2])
                   - ti.math.vec3(cam_origin[f, 0], cam_origin[f, 1],
                                  cam_origin[f, 2])).normalized()
            t_near = near_clip / ti.max(rd.dot(fwd), 1e-6)
            ro = ro + rd * t_near
            rs_sca[slot, 4] = t_near
        for k in ti.static(range(3)):
            rs_ro[slot, k] = ro[k]
            rs_rd[slot, k] = rd[k]
        rs_pix[slot] = g


@ti.func
def _pt_nee_visibility(refit: ti.template(), anyhit: ti.template(),
                       has_tri: ti.template(),
                       has_bez: ti.template(),
                       sorigin, wi, ldist, f, ff,
                       pixel_size_per_t, base_dist, layer_offset_triangles,
                       t_nodes: ti.template(), t_node_miss: ti.template(),
                       t_leaf_prim: ti.template(), t_leaf_tspan: ti.template(),
                       t_first_leaf, tri_pos: ti.template(),
                       tri_colors: ti.template(), tri_uvs: ti.template(),
                       tri_tex_meta: ti.template(), textures: ti.template(),
                       tri_extra: ti.template(), num_colored_triangles,
                       b_nodes: ti.template(), b_node_miss: ti.template(),
                       b_leaf_prim: ti.template(), b_leaf_tspan: ti.template(),
                       b_first_leaf, circuit_meta: ti.template(),
                       circuit_colors: ti.template(),
                       circuit_border_colors: ti.template(),
                       edges_2d: ti.template(), edge_accel: ti.template()):
    """RGB visibility of one NEE shadow ray: 1 - occlusion via the shared
    shadow query (translucent blockers tint, ``casts_shadows`` is honored by
    the leaf test inside the walk).

    ``anyhit`` is the host's compile-time shadow mode
    (``rt_settings.pt_shadow_anyhit``, decided in ``path_trace_render``): 1 is
    the ordered march, 3 the opaque any-hit walk with the march compiled out
    -- valid only on a batch with no translucent and no transmissive
    geometry, where every blocker occludes fully. See ``_shadow_occluded``.

    The emitter end is pulled back by ``_pt_shadow_tmax``, the matching
    scale-aware counterpart of the spawn offset at the surface end.
    """
    occ = _shadow_occluded(
        refit, anyhit, sorigin, wi, f, ff,
        _pt_shadow_tmax(sorigin, wi, ldist),
        pixel_size_per_t, base_dist, layer_offset_triangles,
        has_tri, has_bez,
        t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
        tri_pos, tri_colors, tri_uvs, tri_tex_meta, textures, tri_extra,
        num_colored_triangles,
        b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan, b_first_leaf,
        circuit_meta, circuit_colors, circuit_border_colors,
        edges_2d, edge_accel,
        -1, -1, 0.0, 0.0, tri_pos, 0)
    return ti.math.vec3(1.0, 1.0, 1.0) - occ


@ti.func
def _pt_light_sample_point(light_pos: ti.template(), light_col: ti.template(),
                           f, li, spos, u1, u2):
    """The visibility-ray target for light ``li``: the packed position,
    jittered across the emitter (soft-shadow disk / cone, or a rect-area
    row's own cell) exactly as the deterministic fan integrates it -- with a
    random point per sample instead of a fixed fan, so the penumbra converges
    continuously over samples. Returns ``(wi, ldist, valid)``.
    """
    tl = f % light_pos.shape[0]
    lp = ti.math.vec3(light_pos[tl, li, 0], light_pos[tl, li, 1],
                      light_pos[tl, li, 2])
    ltype = 0
    radius = 0.0
    hu = 0.0
    hv = 0.0
    if light_col.shape[2] > 3:
        ltype = ti.cast(light_col[tl, li, 3] + 0.5, ti.i32)
        radius = light_col[tl, li, 11]
        if ltype == _LT_AREA_SAMPLE:
            hu = light_col[tl, li, 9]
            hv = light_col[tl, li, 10]
    wi = ti.math.vec3(0.0, 0.0, 0.0)
    ldist = 1e7
    valid = 0
    if ltype == _LT_DIRECTIONAL:
        wi = -ti.math.vec3(light_col[tl, li, 6], light_col[tl, li, 7],
                           light_col[tl, li, 8])
        if radius > 0.0:
            aref = ti.math.vec3(1.0, 0.0, 0.0)
            if ti.abs(wi[0]) > 0.9:
                aref = ti.math.vec3(0.0, 1.0, 0.0)
            b1 = wi.cross(aref).normalized()
            b2 = wi.cross(b1)
            ang = 6.2831853 * u1
            rr = radius * ti.sqrt(u2)
            wi = (wi + (ti.cos(ang) * b1 + ti.sin(ang) * b2) * rr).normalized()
        valid = 1
    elif (ltype != _LT_AMBIENT) and (ltype != _LT_HEMISPHERE) \
            and (ltype != _LT_ENV_SH):
        target = lp
        if radius > 0.0:
            if (hu > 0.0) or (hv > 0.0):
                # Rect-area cell: uniform point inside this row's own cell,
                # in the light's plane (b1 = packed right axis).
                b1 = ti.math.vec3(light_col[tl, li, 12], light_col[tl, li, 13],
                                  light_col[tl, li, 14])
                b2 = ti.math.vec3(light_col[tl, li, 6], light_col[tl, li, 7],
                                  light_col[tl, li, 8]).cross(b1)
                target = lp + b1 * (hu * (2.0 * u1 - 1.0)) \
                    + b2 * (hv * (2.0 * u2 - 1.0))
            else:
                to = lp - spos
                d = to.norm()
                if d > 1e-5:
                    wi0 = to / d
                    aref = ti.math.vec3(1.0, 0.0, 0.0)
                    if ti.abs(wi0[0]) > 0.9:
                        aref = ti.math.vec3(0.0, 1.0, 0.0)
                    b1 = wi0.cross(aref).normalized()
                    b2 = wi0.cross(b1)
                    ang = 6.2831853 * u1
                    rr = radius * ti.sqrt(u2)
                    target = lp + (ti.cos(ang) * b1 + ti.sin(ang) * b2) * rr
        to_light = target - spos
        ldist = to_light.norm()
        if ldist > 1e-5:
            wi = to_light / ldist
            valid = 1
    return wi, ldist, valid


@ti.func
def _pt_meta_escape(nee_meta: ti.template()):
    """The ``nee_meta`` words EVERY path reads, hit or not: the environment
    map's placement and its sampling geometry, the light-sample count and the
    AOV gate. Returns ``(n_ls, env_off, env_w, env_h, env_intensity,
    env_share, cdf_h, cdf_w, aov_on)``.
    """
    return (
        ti.max(ti.cast(nee_meta[_NM_LIGHT_SAMPLES] + 0.5, ti.i32), 1),
        ti.cast(nee_meta[_NM_ENV_OFF] + 0.5, ti.i32),
        ti.cast(nee_meta[_NM_ENV_W] + 0.5, ti.i32),
        ti.cast(nee_meta[_NM_ENV_H] + 0.5, ti.i32),
        nee_meta[_NM_ENV_INTENSITY],
        nee_meta[_NM_ENV_SHARE],
        ti.cast(nee_meta[_NM_ENV_CDF_H] + 0.5, ti.i32),
        ti.cast(nee_meta[_NM_ENV_CDF_W] + 0.5, ti.i32),
        nee_meta[_NM_AOV] > 0.5,
    )


@ti.func
def _pt_meta_hit(nee_meta: ti.template()):
    """The ``nee_meta`` words only a path with a surface event reads: the
    next-event table size, the far plane, the packed ambient-row window, the
    sampler's frame gate, the light tree's selection words and the first
    synthetic area-light quad. Returns
    ``(num_nee, far_clip, amb_packed, amb_count, anim_seed, tree_on,
    tree_mix, num_inf, quad_base)``.
    """
    return (
        ti.cast(nee_meta[_NM_COUNT] + 0.5, ti.i32),
        nee_meta[_NM_FAR_CLIP],
        nee_meta[_NM_AMBIENT_PACKED] > 0.5,
        ti.cast(nee_meta[_NM_AMBIENT_COUNT] + 0.5, ti.i32),
        ti.cast(nee_meta[_NM_ANIM_SEED] + 0.5, ti.i32),
        nee_meta[_NM_TREE_ON] > 0.5,
        nee_meta[_NM_TREE_MIX],
        ti.cast(nee_meta[_NM_INF_COUNT] + 0.5, ti.i32),
        ti.cast(nee_meta[_NM_QUAD_BASE] + 0.5, ti.i32),
    )


@ti.func
def _pt_quad_radiance_scale(pt_emit_falloff: ti.template(), prim, quad_base, d):
    """Per-emitter radiance multiplier of a synthetic RectAreaLight quad.

    A packed area-light row applies ``_light_eval``'s emitter model -- a
    ``d^-decay`` falloff (``decay`` defaults to 0: no falloff at all) and a
    range fade -- while a physical emissive quad has inverse square built into
    transport. The difference is ``d^(2 - decay)`` times that same fade, and it
    rides the EMITTER so both ends of the MIS pair evaluate it from the same
    distance and the power-heuristic weights still sum to one: the next-event
    end knows ``ldist``, the BSDF-hit end knows ``t_hit``.

    An ordinary emissive triangle (``prim < quad_base``) returns exactly 1.0
    without touching the table, so emissive meshes are unchanged.
    """
    m = 1.0
    if prim >= quad_base:
        j = prim - quad_base
        expo = pt_emit_falloff[j, 0]
        rng = pt_emit_falloff[j, 1]
        if expo != 0.0:
            # ``ti.max(d, 1e-4)`` is _light_eval's own clamp, so a shading
            # point on the emitter reads the same number either model.
            m = ti.pow(ti.max(d, 1e-4), expo)
        if rng > 0.0:
            q = ti.math.clamp(d / rng, 0.0, 1.0)
            q2 = q * q
            fade = ti.math.clamp(1.0 - q2 * q2, 0.0, 1.0)
            m = m * (fade * fade)
    return m


@ti.kernel
def pt_shade_arena(active: ti.types.ndarray(), num_active: ti.i32,
             t_nodes: NODE_ARG, t_first_leaf: ti.i32,
             num_colored_triangles: ti.i32,
             b_nodes: NODE_ARG, b_first_leaf: ti.i32,
             num_lights: ti.i32,
             layer_offset_triangles: ti.f32,
             refit: ti.template(), has_tri: ti.template(),
             has_bez: ti.template(), shadows: ti.template(),
             # Compile-time shadow-query mode of every visibility ray this
             # kernel spawns (rt_settings.pt_shadow_anyhit): 1 = the ordered
             # march, 3 = opaque any-hit. Decided host-side per batch, exactly
             # as the deterministic renderer decides its own.
             shadow_mode: ti.template(),
             # Light slots this variant's ``vis`` payload carries -- what the
             # batch needs, not the cap (shading_taichi.shadow_vis_slots).
             vis_lights: ti.template(),
             # 1 = the authored-appearance branch SAMPLES its light rows
             # (roadmap section 6a-bis); 0 = it sums every row and traces a
             # shadow ray per row up to ``vis_lights``, which is what it has
             # always done. Compile-time rather than a ``nee_meta`` word for
             # one reason the runtime spelling cannot give: the summing arm
             # feeds ``_run_frag_pipeline`` a row ordinal that may run PAST
             # ``vis_lights`` (a 40-light rig at the 16-slot cap), so it cannot
             # go through a per-thread slot map at all -- and a runtime choice
             # would make every scene pay for the map whether or not it uses
             # one. Taichi specialises on a ``ti.template()`` argument, so both
             # arms still compile and run in ONE process (unlike a
             # ``ti.static`` gate read off a setting), and off is byte-for-byte
             # the kernel this file produced before sampling existed.
             auth_sampled: ti.template(),
             frag_pipelines: ti.template(), frag_scatters: ti.template(),
             tri_pids: ti.template(),
             seed_root: ti.u32, sample_base: ti.i32, tile_pixels: ti.i32,
             rr_start: ti.i32, firefly_clamp: ti.f32,
             time_start: ti.i32, width: ti.i32, height: ti.i32,
             ray_offset: ti.i32,
             rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
             rs_sca: ti.types.ndarray(), rs_int: ti.types.ndarray(),
             rs_pix: ti.types.ndarray(),
             hit_f: ti.types.ndarray(), hit_i: ti.types.ndarray(),
             pt_thru: ti.types.ndarray(), pt_acc: ti.types.ndarray(),
             pt_stats: ti.types.ndarray(),
             pt_aov: ti.types.ndarray(),
             arena_f32: ti.types.ndarray(),
             arena_i32: ti.types.ndarray(),
             aoff: ti.types.ndarray(),
             ashp: ti.types.ndarray()):
    """Consume one traverse's hit-event batch (see the module docstring).

    Deterministic alpha compositing carries every crossed surface's local
    radiance (NEE-lit, frag-shaded, or raw emission by pipeline); the one
    stochastic continuation per path is chosen among pass-through and the
    material's importance-sampled lobes, with proper reweighting, so an
    unlit-only stack keeps the zero-variance composite while lit content
    gets full transport. The camera-segment alpha transparency
    (``rs_sca[r, 0]``) freezes at the first scatter. On that segment a
    declared closed shell (``tri_shell``: its surface id, or -1) attenuates
    once per entry/exit pair -- the exiting crossing contributes nothing --
    via the per-ray ring in ``rs_int`` (see ``_INT_RING0``), matching the
    deterministic route's ``solid_shell_alpha`` coverage ceiling.

    Direct lighting at a lit vertex is the sum of a deterministic fill from
    the direction-less rows (ambient / hemisphere) and ``pt_light_samples``
    draws from the next-event table (``nee_ref``), each entry chosen either
    by descending the light tree at the shading point (``lt_node_f`` /
    ``lt_node_i``, the finite emitters) or from the position-independent
    infinite list (``nee_inf_cdf``: directional rows and the environment),
    or -- with ``pt_light_tree`` off -- from the one flat power CDF
    (``nee_cdf``): delta and area light rows at stage radiometry, emissive
    triangles and the environment map at physical radiometry with
    power-heuristic MIS against the continuation lobes (the emitters BSDF
    paths can also find; delta lights have no geometry to MIS against).
    Escaping rays sample the environment map in their own direction, so
    mirrors and GI see the sky the deterministic renderer shows.

    Direct lighting at an AUTHORED-appearance vertex is whatever that
    material's stage says it is, over whichever light rows this kernel hands
    it. With ``auth_sampled`` off that is every row, each with its own shadow
    ray up to ``vis_lights`` -- the deterministic renderer's model, cap
    included. With it on the branch fills the direction-less rows and then
    draws ``_NM_AUTHORED_SAMPLES`` rows from the authored table on
    ``nee_ref``'s tail, weighting each by ``1 / (S * p)`` on its radiance;
    the stage still sees a light index, a colour and a visibility triple and
    cannot tell (roadmap section 6a-bis). That table is the authored
    branch's own, not the next-event entries: a ``RectAreaLight`` reaches a
    physically-integrated surface as its emissive quads and an authored one
    as its packed cell rows, which is the model each has.

    ``frag_scatters`` is the per-pipeline custom ray-continuation tuple (the
    same one ``wavefront_shade`` takes, narrowed to the batch): a crossing
    whose material pipeline supplies a scatter has its radiance AND its
    continuation decided by the user's ``@ti.func``, and continues as a
    **delta lobe** -- one of the three returned branches picked
    stochastically, weighted by the branch weights, and ``prev_pdf = 0``
    because no next-event strategy covers where the user's direction lands
    (the contract refraction and the tinted pane already get). An empty tuple
    -- every scene that authors no custom scatter -- compiles this kernel
    exactly as it compiled before the feature existed.
    """
    # Arena-bound parameters (arena_args_taichi): each name is
    # rebound to a window into its dtype's buffer, at the offset
    # the host wrote into aoff. Order is _PT_SHADE_ARENA's.
    t_node_miss = ti.static(ArenaView(arena_i32, aoff[0], (ashp[0],)))
    t_leaf_prim = ti.static(ArenaView(arena_i32, aoff[1], (ashp[1],)))
    t_leaf_tspan = ti.static(ArenaView(arena_i32, aoff[2], (ashp[2],)))
    tri_pos = ti.static(ArenaView(arena_f32, aoff[3], (ashp[3], ashp[4], ashp[5])))
    tri_norm = ti.static(ArenaView(arena_f32, aoff[4], (ashp[6], ashp[7], ashp[8])))
    tri_extra = ti.static(ArenaView(arena_f32, aoff[5], (ashp[9], ashp[10], ashp[11])))
    tri_colors = ti.static(ArenaView(
        arena_f32, aoff[6], (ashp[12], ashp[13], ashp[14], ashp[15])))
    tri_uvs = ti.static(ArenaView(arena_f32, aoff[7], (ashp[16], ashp[17], ashp[18])))
    tri_tex_meta = ti.static(ArenaView(arena_i32, aoff[8], (ashp[19], ashp[20])))
    textures = ti.static(ArenaView(arena_f32, aoff[9], (ashp[21], ashp[22], ashp[23])))
    b_node_miss = ti.static(ArenaView(arena_i32, aoff[10], (ashp[24],)))
    b_leaf_prim = ti.static(ArenaView(arena_i32, aoff[11], (ashp[25],)))
    b_leaf_tspan = ti.static(ArenaView(arena_i32, aoff[12], (ashp[26],)))
    circuit_meta = ti.static(ArenaView(
        arena_f32, aoff[13], (ashp[27], ashp[28], ashp[29])))
    circuit_colors = ti.static(ArenaView(
        arena_f32, aoff[14], (ashp[30], ashp[31], ashp[32], ashp[33])))
    circuit_border_colors = ti.static(ArenaView(
        arena_f32, aoff[15], (ashp[34], ashp[35], ashp[36], ashp[37])))
    edges_2d = ti.static(ArenaView(arena_f32, aoff[16], (ashp[38], ashp[39], ashp[40])))
    edge_accel = ti.static(ArenaView(arena_i32, aoff[17], (ashp[41],)))
    tri_mat_id = ti.static(ArenaView(arena_i32, aoff[18], (ashp[42], ashp[43])))
    tri_mat = ti.static(ArenaView(arena_f32, aoff[19], (ashp[44], ashp[45], ashp[46])))
    light_pos = ti.static(ArenaView(
        arena_f32, aoff[20], (ashp[47], ashp[48], ashp[49])))
    light_col = ti.static(ArenaView(
        arena_f32, aoff[21], (ashp[50], ashp[51], ashp[52])))
    pixel_world_scale = ti.static(ArenaView(arena_f32, aoff[22], (ashp[53],)))
    cam_origin = ti.static(ArenaView(arena_f32, aoff[23], (ashp[54], ashp[55])))
    nee_cdf = ti.static(ArenaView(arena_f32, aoff[24], (ashp[56],)))
    nee_ref = ti.static(ArenaView(arena_i32, aoff[25], (ashp[57], ashp[58])))
    nee_meta = ti.static(ArenaView(arena_f32, aoff[26], (ashp[59],)))
    tri_emit_prob = ti.static(ArenaView(arena_f32, aoff[27], (ashp[60],)))
    env_cdf = ti.static(ArenaView(arena_f32, aoff[28], (ashp[61], ashp[62])))
    tri_shell = ti.static(ArenaView(arena_i32, aoff[29], (ashp[63], ashp[64])))
    tri_emit_entry = ti.static(ArenaView(arena_i32, aoff[30], (ashp[65],)))
    lt_node_f = ti.static(ArenaView(
        arena_f32, aoff[31], (ashp[66], ashp[67], ashp[68])))
    lt_node_i = ti.static(ArenaView(
        arena_i32, aoff[32], (ashp[69], ashp[70], ashp[71])))
    lt_entry_leaf = ti.static(ArenaView(arena_i32, aoff[33], (ashp[72], ashp[73])))
    lt_frame = ti.static(ArenaView(arena_i32, aoff[34], (ashp[74],)))
    nee_inf_cdf = ti.static(ArenaView(arena_f32, aoff[35], (ashp[75],)))
    nee_inf_ref = ti.static(ArenaView(arena_i32, aoff[36], (ashp[76],)))
    pt_emit_falloff = ti.static(ArenaView(arena_f32, aoff[37], (ashp[77], ashp[78])))
    pixels_per_frame = width * height
    for i in range(num_active):
        r = active[i]
        num_hits = rs_int[r, 3]
        # Next-event table + environment metadata (runtime words, so one
        # compiled kernel serves every scene shape). ``aov_on`` gates every
        # pt_aov access: with it off the tensor is a [1, PT_AOV_WIDTH] dummy.
        # The words are decoded inside the branches that read them -- an
        # escaping path reads the environment half only.
        if num_hits > 0:
            (n_ls, env_off, env_w, env_h, env_intensity, env_share,
             cdf_h, cdf_w, aov_on) = _pt_meta_escape(nee_meta)
            (num_nee, far_clip, amb_packed, amb_count, anim_seed,
             tree_on, tree_mix, num_inf, quad_base) = _pt_meta_hit(nee_meta)
            g = ray_offset + rs_pix[r]
            f_rel = g // pixels_per_frame
            f = time_start + f_rel
            p = g - f_rel * pixels_per_frame
            # Which of the chunk's light trees this frame uses: frames whose
            # emitter geometry is identical share one (light_tree.py).
            lt_r = 0
            if tree_on:
                lt_r = lt_frame[f_rel]
            key = _pt_key(f * anim_seed, p)
            # Sampler seeds and pair bases that are constant for the whole
            # path, hoisted out of the drain loop (roadmap section 0.2): the
            # per-path seed half every draw shares, the first per-crossing
            # pair, and the width of one crossing's block (``2L`` next-event
            # pairs plus the lobe select; see the module docstring's table).
            path_seed = _pt_path_seed(seed_root, key)
            s_index = sample_base + r // tile_pixels
            ff = ti.cast(f, ti.f32)
            pixel_size_per_t = pixel_world_scale[f]
            cam_pos = ti.math.vec3(cam_origin[f, 0], cam_origin[f, 1],
                                   cam_origin[f, 2])
            ro = ti.math.vec3(rs_ro[r, 0], rs_ro[r, 1], rs_ro[r, 2])
            rd = ti.math.vec3(rs_rd[r, 0], rs_rd[r, 1], rs_rd[r, 2])
            thru = ti.math.vec4(pt_thru[r, 0], pt_thru[r, 1], pt_thru[r, 2],
                                pt_thru[r, 3])
            t_alpha = rs_sca[r, 0]
            t_prev = rs_sca[r, 1]
            layer_prev = rs_sca[r, 2]
            seam_t = rs_sca[r, 3]
            base_dist = rs_sca[r, 4]
            prev_pdf = rs_sca[r, _SCA_PREV_PDF]
            bounces_left = rs_int[r, 0]
            processed = rs_int[r, 1]
            acc = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
            # Bounce ordinal for the sampler's dimension table; the FIRST
            # value written by the host is max_bounces, so the ordinal is
            # the difference.
            max_b = rs_int[r, 4]
            pair_nee0 = PAIR_BOUNCE_BASE + PAIRS_PER_BOUNCE * max_b
            pairs_per_cross = 2 * n_ls + 1
            ring = ti.Vector([-1, -1, -1, -1])
            for q in ti.static(range(_SHELL_RING_SLOTS)):
                ring[q] = rs_int[r, _INT_RING0 + q]
            aov_open = 0
            aov_alb = ti.math.vec3(0.0, 0.0, 0.0)
            aov_nrm = ti.math.vec3(0.0, 0.0, 0.0)
            if aov_on:
                if pt_aov[r, _AOV_CLOSED] < 0.5:
                    aov_open = 1
            # Set at the first random decision this launch takes (see
            # ``_PT_ACC_STOCH``); folded into the sticky column at write-back.
            stoch = 0

            kb_t = ti.Vector([0.0] * kbuf)
            kb_layer = ti.Vector([0.0] * kbuf)
            kb_prim = ti.Vector([0] * kbuf)
            kb_flags = ti.Vector([0] * kbuf)
            kb_a = ti.Vector([0.0] * kbuf)
            kb_b = ti.Vector([0.0] * kbuf)
            for q in ti.static(range(kbuf)):
                kb_t[q] = hit_f[q, 0, i]
                kb_layer[q] = hit_f[q, 1, i]
                kb_a[q] = hit_f[q, 2, i]
                kb_b[q] = hit_f[q, 3, i]
                kb_prim[q] = hit_i[q, 0, i]
                kb_flags[q] = hit_i[q, 1, i]

            done = False
            bounced = False
            absorbed = False
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
                if (far_clip > 0.0) and (base_dist + t_hit > far_clip):
                    # Past the camera's far distance, the same test and the
                    # same site as ``wavefront_shade``. Hits drain
                    # front-to-back so everything left is farther still;
                    # retire the path (not absorbed, so its leftover
                    # throughput still shows the background or the
                    # environment map). ``base_dist`` accumulates across
                    # scatters, so the plane clips path length from the
                    # camera exactly as it does for the other renderer.
                    done = True
                    break
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
                # This crossing's dimension-pair block (module docstring):
                # ``2L`` next-event pairs then the lobe select.
                pair_cross0 = pair_nee0 + pairs_per_cross * processed
                htype = flags & 3
                edge_hit = (flags >> 2) & 1
                border = (flags >> 3) & 1

                seam_eps = depth_tie_epsilon
                if (edge_hit == 1) and (t_hit - seam_t <= seam_eps):
                    t_prev = t_hit
                    layer_prev = hit_layer
                    continue
                seam_t = t_hit if edge_hit == 1 else -1e30

                # A synthetic RectAreaLight quad is invisible to the CAMERA
                # SEGMENT and to nothing else: ``bounces_left >= max_b`` is
                # "this path has not scattered yet", the same reading the
                # closed-shell ring takes. A primary ray peels straight
                # through (the panel is not drawn, exactly as the
                # deterministic renderer does not draw a light), while a ray
                # that has bounced -- the reflection in a mirror, an indirect
                # diffuse ray -- sees it and collects its emission. The quads
                # are packed non-opaque so nothing behind one is pruned from
                # the gather while it is being skipped.
                if (htype == 1) and (prim >= quad_base) \
                        and (bounces_left >= max_b):
                    t_prev = t_hit
                    layer_prev = hit_layer
                    continue

                w0 = 1.0 - a - b
                color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                alpha = 0.0
                metalness = -1.0
                rough = 0.0
                pid = _MID_UNLIT
                if htype == 1:
                    color, alpha = _tri_color_g(
                        0, f, prim, w0, a, b, tri_colors, tri_colors, tri_uvs,
                        tri_tex_meta, textures, num_colored_triangles)
                    metalness, rough = _tri_extra_g(
                        0, f, prim, w0, a, b, tri_extra, tri_colors, tri_uvs,
                        tri_tex_meta, textures, num_colored_triangles)
                    pid = tri_mat_id[f % tri_mat_id.shape[0], prim]
                else:
                    color, alpha = _sample_circuit_color(
                        prim, f, a, b, border,
                        circuit_meta, circuit_colors, circuit_border_colors)
                    cm = f % circuit_meta.shape[0]
                    metalness = circuit_meta[cm, prim, _M_REFLECTIVITY]
                    rough = circuit_meta[cm, prim, _M_ROUGHNESS]
                alpha = ti.math.clamp(alpha, 0.0, 1.0)

                # Closed-shell opacity ring (``solid_shell_alpha``).  On the
                # camera segment a declared closed shell attenuates ONCE per
                # entry/exit pair: the entering crossing composites and
                # remembers the surface id, the exiting crossing finds the
                # id, removes it, and contributes nothing (alpha 0 makes it
                # a weight-1 pass-through with zero radiance below).  This
                # is the per-ray limit of the sheet route's coverage
                # ceiling; see ``_INT_RING0``.  A post-scatter segment is
                # physical transport and never suppresses; a seam-skipped
                # duplicate never reaches this point, so a shared edge
                # toggles once.
                suppressed = 0
                if (htype == 1) and (bounces_left >= max_b):
                    sid_cs = ti.cast(
                        tri_shell[f % tri_shell.shape[0], prim], ti.i32)
                    if sid_cs >= 0:
                        removed = 0
                        for q in ti.static(range(_SHELL_RING_SLOTS)):
                            if (removed == 0) and (ring[q] == sid_cs):
                                ring[q] = -1
                                removed = 1
                        if removed == 1:
                            suppressed = 1
                        else:
                            inserted = 0
                            for q in ti.static(range(_SHELL_RING_SLOTS)):
                                if (inserted == 0) and (ring[q] < 0):
                                    ring[q] = sid_cs
                                    inserted = 1
                            if inserted == 0:
                                ti.atomic_add(
                                    pt_stats[PT_STAT_SHELL_RING], 1)
                if suppressed == 1:
                    alpha = 0.0
                albedo3 = ti.math.vec3(color[0], color[1], color[2])

                lit = (htype == 1) and (pid >= _MID_LAMBERT) \
                    and (pid <= _MID_PHYSICAL)
                authored = (htype == 1) and (not lit) and (pid != _MID_UNLIT)

                # IOR / transmission of the surface (material or per-texel).
                ior = 0.0
                T = 0.0
                if htype == 1:
                    ior, T = _tri_ior_transmission_g(
                        0, f, prim, w0, a, b, tri_extra, tri_colors, tri_uvs,
                        tri_tex_meta, textures, num_colored_triangles)
                else:
                    cm2 = f % circuit_meta.shape[0]
                    ior = circuit_meta[cm2, prim, _M_IOR]
                    T = circuit_meta[cm2, prim, _M_TRANSMISSION]
                T = ti.math.clamp(T, 0.0, 1.0)

                # Normals: needed by every treatment but pure unlit.
                snrm = ti.math.vec3(0.0, 0.0, 0.0)
                fnrm = ti.math.vec3(0.0, 0.0, 0.0)
                shade_n = ti.math.vec3(0.0, 0.0, 0.0)
                fn_len = 0.0
                needs_normal = lit or authored or (metalness >= 0.0) \
                    or (T > 1e-4)
                if needs_normal:
                    if htype == 1:
                        snrm = _tri_normal_g(
                            0, f, prim, w0, a, b, tri_norm, tri_pos, tri_uvs,
                            tri_tex_meta, textures, num_colored_triangles)
                        tp = f % tri_pos.shape[0]
                        v0 = ti.math.vec3(tri_pos[tp, prim, 0],
                                          tri_pos[tp, prim, 1],
                                          tri_pos[tp, prim, 2])
                        v1 = ti.math.vec3(tri_pos[tp, prim, 3],
                                          tri_pos[tp, prim, 4],
                                          tri_pos[tp, prim, 5])
                        v2 = ti.math.vec3(tri_pos[tp, prim, 6],
                                          tri_pos[tp, prim, 7],
                                          tri_pos[tp, prim, 8])
                        fnrm = (v1 - v0).cross(v2 - v0)
                        fn_len = fnrm.norm()
                        if fn_len > 1e-12:
                            fnrm = fnrm.normalized()
                    else:
                        snrm = _bezier_normal(f, prim, circuit_meta)
                        if snrm.norm() > 1e-9:
                            snrm = snrm.normalized()
                        fnrm = snrm
                    view_dir = -rd
                    if htype == 1:
                        flat = 0.0
                        if pid < _USER_PIPELINE_BASE:
                            shade_n = _sided_shading_normal(
                                snrm, fnrm, view_dir, tri_mat, f, prim)
                            flat = tri_mat[f % tri_mat.shape[0], prim, 10]
                        else:
                            shade_n = snrm
                        shade_n = _prep_normal(shade_n, fnrm, flat, view_dir)
                    else:
                        # Circuits are flat and two-sided.
                        shade_n = snrm
                        if shade_n.dot(rd) > 0.0:
                            shade_n = -shade_n

                # Normal the GGX reflection lobe is built on: ``shade_n``
                # turned to FACE THE RAY. Mirror reflection is invariant to
                # the normal's sign, but the lobe's cosines and its
                # above-the-horizon test are not, and ``shade_n`` is the
                # surface's declared side, not the ray's: a one-sided solid
                # keeps its outward normal when a refracted path hits its
                # exit face from inside. Sampling the lobe about that normal
                # put every VNDF direction below its horizon, so ``ok == 0``
                # absorbed the path -- with a few samples per pixel, all of
                # them absorbed is a pure-black pixel where the sky should
                # show through the glass (roadmap, "Isolated black pixels
                # beside glass"). Total internal reflection is the same hit
                # with the transmission branch closed, and it lands here too.
                # A front-facing hit gets ``spec_n == shade_n`` and is
                # unaffected.
                spec_n = shade_n
                if shade_n.dot(rd) > 0.0:
                    spec_n = -shade_n

                hit_p = ro + t_hit * rd

                # Volumetric absorption on exiting a transmissive interior
                # (Beer-Lambert), mirroring the wavefront's site exactly: the
                # side test uses the RAW interpolated normal (the sided/prepped
                # shading normal is viewer-flipped and would never read
                # "exiting").
                if (T > 1e-4) and (htype == 1) and (pid < _USER_PIPELINE_BASE):
                    if needs_normal and (rd.dot(snrm) > 0.0):
                        tma = f % tri_mat.shape[0]
                        sa = _MAT_ATTENUATION_SIGMA
                        seg = ti.max(t_hit - t_prev, 0.0)
                        for k in ti.static(range(3)):
                            thru[k] *= ti.exp(-tri_mat[tma, prim, sa + k] * seg)

                # Lit-lobe energies + selection weights, hoisted ahead of the
                # radiance block: the emission MIS weight and every NEE
                # response need exactly the lobes the continuation samples.
                e_diff_l = ti.math.vec3(0.0, 0.0, 0.0)
                e_spec_l = ti.math.vec3(0.0, 0.0, 0.0)
                e_trans_l = ti.math.vec3(0.0, 0.0, 0.0)
                f0_l = ti.math.vec3(0.0, 0.0, 0.0)
                wl_pass = 1.0 - alpha
                wl_diff = 0.0
                wl_spec = 0.0
                wl_trans = 0.0
                if lit and (suppressed == 0):
                    # ``rough`` is REPLACED by the lobe set's own GGX width:
                    # phong authors its highlight as a Blinn-Phong exponent,
                    # so its lobes, its NEE responses and its continuation
                    # must all read the converted roughness. Every other
                    # pipeline gets its own value back unchanged.
                    e_diff_l, e_spec_l, e_trans_l, f0_l, rough = \
                        _pt_lit_lobes(
                            pid, tri_mat, f, prim, albedo3, metalness, rough,
                            ior, T, shade_n, rd)
                    wl_diff = alpha * ti.max(e_diff_l[0],
                                             ti.max(e_diff_l[1],
                                                    e_diff_l[2]))
                    wl_spec = alpha * ti.max(e_spec_l[0],
                                             ti.max(e_spec_l[1],
                                                    e_spec_l[2]))
                    wl_trans = alpha * ti.max(e_trans_l[0],
                                              ti.max(e_trans_l[1],
                                                     e_trans_l[2]))

                # Custom-scatter state for this crossing: whether a user
                # scatter owns it, the radiance it committed, and its three
                # continuation branches. Declared unconditionally because a
                # Taichi ``if`` body -- ``ti.static`` included -- is its own
                # variable scope, so a declaration inside the static gate the
                # readers below sit in would not be visible to them. Every
                # READ is gated, so with an empty tuple these are locals
                # nothing loads and the compiled kernel is the one this file
                # produced before custom scatter reached it.
                sc_on = 0
                sc_contrib = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                sc_pass_w = ti.math.vec3(0.0, 0.0, 0.0)
                sc_refl_o = ti.math.vec3(0.0, 0.0, 0.0)
                sc_refl_d = ti.math.vec3(0.0, 0.0, 0.0)
                sc_refl_w = ti.math.vec3(0.0, 0.0, 0.0)
                sc_trans_o = ti.math.vec3(0.0, 0.0, 0.0)
                sc_trans_d = ti.math.vec3(0.0, 0.0, 0.0)
                sc_trans_w = ti.math.vec3(0.0, 0.0, 0.0)

                # ----------------------------------------------------------
                # Local radiance of this crossing (emission semantics).
                # ----------------------------------------------------------
                local = ti.math.vec4(color[0], color[1], color[2], color[3])
                # A suppressed crossing contributes nothing: skip its NEE /
                # frag-pipeline work outright (its ``local`` would be
                # multiplied by the zeroed alpha anyway).
                if lit and (suppressed == 0):
                    # The lit treatment samples ONE emitter out of the
                    # next-event table per light sample, so this crossing's
                    # value is a Monte Carlo estimate however many lobes the
                    # continuation below ends up offering.
                    stoch = 1
                    tm = f % tri_mat.shape[0]
                    emissive = ti.math.vec3(tri_mat[tm, prim, 0],
                                            tri_mat[tm, prim, 1],
                                            tri_mat[tm, prim, 2]) \
                        * tri_mat[tm, prim, 3]
                    # The BSDF end of the falloff pair: ``t_hit`` is measured
                    # from ``ro``, which a pass-through crossing leaves alone,
                    # so it IS the distance from the shading point the
                    # next-event end would have measured ``ldist`` from.
                    emissive = emissive * _pt_quad_radiance_scale(
                        pt_emit_falloff, prim, quad_base, t_hit)
                    # Emission reached through a sampled smooth lobe is
                    # MIS-weighted against the NEE strategy that also covers
                    # this triangle; camera rays, delta continuations and
                    # emitters outside the table (or their un-sampled back
                    # side) keep weight 1.
                    w_emit = 1.0
                    if (prev_pdf > 0.0) and (tri_emit_prob[prim] > 0.0) \
                            and (fn_len > 1e-12):
                        covered = 1
                        cos_l = (-rd).dot(fnrm)
                        if tri_mat[tm, prim, _MAT_ONE_SIDED] > 0.5:
                            if cos_l <= 1e-6:
                                covered = 0
                        area_e = 0.5 * fn_len
                        if covered == 1:
                            c_l = ti.max(ti.abs(cos_l), 1e-4)
                            # The probability next-event estimation would
                            # have had of picking THIS triangle from the
                            # PREVIOUS vertex. Flat CDF: a table lookup.
                            # Light tree: a query, because selection is a
                            # function of the shading point -- walked up
                            # from the triangle's leaf at ``ro``, which is
                            # the previous scatter origin (rs_ro is not
                            # touched by a pass-through crossing).
                            p_ne = tri_emit_prob[prim]
                            if tree_on:
                                p_ne = 0.0
                                ent_m = tri_emit_entry[prim]
                                if ent_m >= 0:
                                    p_ne = tree_mix * _pt_lt_pmf(
                                        lt_node_f, lt_node_i, lt_r,
                                        lt_entry_leaf[lt_r, ent_m], ro)
                            pdf_ne = p_ne * (t_hit * t_hit) \
                                / ti.max(area_e * c_l, 1e-12) \
                                * ti.cast(n_ls, ti.f32)
                            w_emit = prev_pdf * prev_pdf \
                                / ti.max(prev_pdf * prev_pdf
                                         + pdf_ne * pdf_ne, 1e-20)
                    direct = emissive * w_emit
                    recv = 1
                    if tri_mat.shape[2] > _MAT_NO_SHADOW_RECEIVE:
                        if tri_mat[tm, prim, _MAT_NO_SHADOW_RECEIVE] > 0.5:
                            recv = 0
                    sorigin = _pt_offset_ray_origin(
                        hit_p, fnrm if fnrm.dot(-rd) >= 0.0 else -fnrm)
                    # Deterministic fill from the direction-less rows.  A
                    # constant-radiance environment ``L`` over the diffuse
                    # lobe integrates to exactly ``e_diff * L`` -- the
                    # physical answer, and the one consistent with the BSDF
                    # every other emitter kind now goes through (roadmap
                    # section 5).  ``_light_eval`` stays the EMITTER model:
                    # for a hemisphere row it is what blends sky and ground
                    # by the shading normal.  Specular gets no ambient fill;
                    # indirect transport is what replaces it.
                    # Never a visibility ray, never in the sampled table.
                    # Their row indexes ride the tail of ``nee_ref`` when the
                    # host packed them (``pt_ambient_rows``), so this visits
                    # exactly those rows in the same ascending order the scan
                    # below walks them in; the scan is the fallback.
                    if amb_packed:
                        for j in range(amb_count):
                            li = nee_ref[num_nee + j, 1]
                            _ld_a, lc, _sw_a, _frac = _light_eval(
                                light_pos, light_col, f, li, hit_p, shade_n)
                            direct += e_diff_l * lc
                    elif light_col.shape[2] > 3:
                        tl_f = f % light_col.shape[0]
                        for li in range(num_lights):
                            lt_row = ti.cast(light_col[tl_f, li, 3] + 0.5,
                                             ti.i32)
                            if (lt_row == _LT_AMBIENT) \
                                    or (lt_row == _LT_HEMISPHERE):
                                _ld_b, lc, _sw_b, _frac = _light_eval(
                                    light_pos, light_col, f, li, hit_p,
                                    shade_n)
                                direct += e_diff_l * lc
                    # Next-event estimation: ``pt_light_samples`` draws from
                    # the power-weighted table (delta/area light rows,
                    # emissive triangles, the environment map).
                    if num_nee > 0:
                        inv_ls = 1.0 / ti.cast(n_ls, ti.f32)
                        for ls in range(n_ls):
                            pair_sel = pair_cross0 + 2 * ls
                            u_sel = pt_sample_2d_seeded(path_seed, pair_sel,
                                                        s_index)
                            u_pt = pt_sample_2d_seeded(path_seed,
                                                       pair_sel + 1, s_index)
                            entry = 0
                            p_sel = 0.0
                            if tree_on:
                                # One draw, two stages: the position-
                                # independent tree-vs-infinite mixture, then
                                # the descent, each consuming its bracket of
                                # the same stratified number.
                                uu = u_sel[0]
                                if uu < tree_mix:
                                    uu = ti.math.clamp(
                                        uu / ti.max(tree_mix, 1e-12),
                                        0.0, 0.99999994)
                                    _lf, ent_s, p_d = _pt_lt_descend(
                                        lt_node_f, lt_node_i, lt_r,
                                        sorigin, uu)
                                    if ent_s >= 0:
                                        entry = ent_s
                                        p_sel = p_d * tree_mix
                                elif num_inf > 0:
                                    uu = ti.math.clamp(
                                        (uu - tree_mix)
                                        / ti.max(1.0 - tree_mix, 1e-12),
                                        0.0, 0.99999994)
                                    slot, p_i = _pt_pick_nee_entry(
                                        nee_inf_cdf, num_inf, uu)
                                    entry = nee_inf_ref[slot]
                                    p_sel = p_i * (1.0 - tree_mix)
                            else:
                                entry, p_sel = _pt_pick_nee_entry(
                                    nee_cdf, num_nee, u_sel[0])
                            kind = nee_ref[entry, 0]
                            ref = nee_ref[entry, 1]
                            contrib = ti.math.vec3(0.0, 0.0, 0.0)
                            wi_vis = ti.math.vec3(0.0, 0.0, 0.0)
                            ldist = 1e7
                            if kind == _NEE_LIGHT_ROW:
                                if p_sel > 1e-12:
                                    ld, lc, _sw_r, wi_v, ld_d, valid = \
                                        _pt_nee_light_row(
                                            light_pos, light_col, f, ref,
                                            hit_p, shade_n,
                                            u_pt[0], u_pt[1])
                                    if (valid == 1) and (
                                            (lc[0] != 0.0)
                                            or (lc[1] != 0.0)
                                            or (lc[2] != 0.0)):
                                        # One BSDF for every emitter kind
                                        # (roadmap section 5): the estimator
                                        # is ``f_cos * radiance / p_sel``,
                                        # with ``_light_eval``'s radiometry
                                        # untouched -- that is the emitter
                                        # model, and only the surface
                                        # response changed. A light row is
                                        # unhittable by a BSDF ray, so there
                                        # is nothing to MIS against.
                                        f_cos_r, _pdf_r = _pt_lit_f_pdf(
                                            e_diff_l, e_spec_l, f0_l,
                                            rough, shade_n, rd, ld,
                                            wl_pass, wl_diff, wl_spec,
                                            wl_trans)
                                        contrib = f_cos_r * lc \
                                            * (inv_ls / p_sel)
                                        wi_vis = wi_v
                                        ldist = ld_d
                            elif kind == _NEE_EMISSIVE_TRI:
                                # Flat arm: the table's own probability, so
                                # this end and the MIS end read one number.
                                # Tree arm: the descent's probability, which
                                # is what the upward walk reproduces.
                                p_tri = tri_emit_prob[ref]
                                if tree_on:
                                    p_tri = p_sel
                                if p_tri > 1e-12:
                                    pe, ne, area_s, ew0, ew1, ew2 = \
                                        _pt_emissive_sample(
                                            tri_pos, f, ref,
                                            u_pt[0], u_pt[1])
                                    to_e = pe - hit_p
                                    d_e = to_e.norm()
                                    if (d_e > 1e-4) and (area_s > 1e-12):
                                        wi = to_e / d_e
                                        cos_l = (-wi).dot(ne)
                                        tm_e = f % tri_mat.shape[0]
                                        if tri_mat[tm_e, ref,
                                                   _MAT_ONE_SIDED] > 0.5:
                                            if cos_l <= 1e-6:
                                                cos_l = 0.0
                                        c_l = ti.abs(cos_l)
                                        if c_l > 1e-6:
                                            le = ti.math.vec3(
                                                tri_mat[tm_e, ref, 0],
                                                tri_mat[tm_e, ref, 1],
                                                tri_mat[tm_e, ref, 2]) \
                                                * tri_mat[tm_e, ref, 3]
                                            # The next-event end of the
                                            # falloff pair -- the same
                                            # function of the same distance
                                            # the BSDF end applies above.
                                            le = le * _pt_quad_radiance_scale(
                                                pt_emit_falloff, ref,
                                                quad_base, d_e)
                                            _ec, e_alpha = _tri_color_g(
                                                0, f, ref, ew0, ew1, ew2,
                                                tri_colors, tri_colors,
                                                tri_uvs, tri_tex_meta,
                                                textures,
                                                num_colored_triangles)
                                            f_cos, pdf_b = _pt_lit_f_pdf(
                                                e_diff_l, e_spec_l, f0_l,
                                                rough, shade_n, rd, wi,
                                                wl_pass, wl_diff, wl_spec,
                                                wl_trans)
                                            if bounces_left <= 0:
                                                pdf_b = 0.0
                                            pdf_sa = p_tri * (d_e * d_e) \
                                                / ti.max(area_s * c_l,
                                                         1e-12)
                                            pdf_h = pdf_sa \
                                                * ti.cast(n_ls, ti.f32)
                                            w_mis = pdf_h * pdf_h / ti.max(
                                                pdf_h * pdf_h
                                                + pdf_b * pdf_b, 1e-20)
                                            contrib = f_cos * le * (
                                                e_alpha * w_mis * inv_ls
                                                / ti.max(pdf_sa, 1e-12))
                                            wi_vis = wi
                                            ldist = d_e
                            else:  # _NEE_ENV
                                if (env_share > 0.0) and (env_w > 0):
                                    dir_e, pdf_e = _pt_env_sample(
                                        env_cdf, cdf_h, cdf_w,
                                        u_pt[0], u_pt[1])
                                    pdf_sa = env_share * pdf_e
                                    if (pdf_sa > 1e-12) \
                                            and (shade_n.dot(dir_e)
                                                 > 1e-6):
                                        ec = _sample_env_map(
                                            f, dir_e, env_off, env_w,
                                            env_h, env_intensity,
                                            textures)
                                        f_cos, pdf_b = _pt_lit_f_pdf(
                                            e_diff_l, e_spec_l, f0_l,
                                            rough, shade_n, rd, dir_e,
                                            wl_pass, wl_diff, wl_spec,
                                            wl_trans)
                                        if bounces_left <= 0:
                                            pdf_b = 0.0
                                        pdf_h = pdf_sa \
                                            * ti.cast(n_ls, ti.f32)
                                        w_mis = pdf_h * pdf_h / ti.max(
                                            pdf_h * pdf_h
                                            + pdf_b * pdf_b, 1e-20)
                                        contrib = f_cos * ec * (
                                            w_mis * inv_ls / pdf_sa)
                                        wi_vis = dir_e
                                        ldist = 1e7
                            if (contrib[0] != 0.0) or (contrib[1] != 0.0) \
                                    or (contrib[2] != 0.0):
                                vis3 = ti.math.vec3(1.0, 1.0, 1.0)
                                if ti.static(shadows != 0):
                                    if recv == 1:
                                        if shade_n.dot(wi_vis) > 1e-4:
                                            vis3 = _pt_nee_visibility(
                                                refit, shadow_mode,
                                                has_tri, has_bez,
                                                sorigin, wi_vis, ldist,
                                                f, ff,
                                                pixel_size_per_t,
                                                base_dist,
                                                layer_offset_triangles,
                                                t_nodes, t_node_miss,
                                                t_leaf_prim, t_leaf_tspan,
                                                t_first_leaf,
                                                tri_pos, tri_colors,
                                                tri_uvs, tri_tex_meta,
                                                textures, tri_extra,
                                                num_colored_triangles,
                                                b_nodes, b_node_miss,
                                                b_leaf_prim, b_leaf_tspan,
                                                b_first_leaf,
                                                circuit_meta,
                                                circuit_colors,
                                                circuit_border_colors,
                                                edges_2d, edge_accel)
                                        else:
                                            vis3 = ti.math.vec3(0.0, 0.0,
                                                                0.0)
                                direct += contrib * vis3
                    local = ti.math.vec4(direct[0], direct[1], direct[2],
                                         color[3])
                elif authored and (suppressed == 0):
                    # An authored-appearance crossing draws too, and in BOTH
                    # arms: the summing arm jitters each light's soft-shadow
                    # ray, the sampling arm picks which rows to light from at
                    # all, and either way a custom scatter picks a branch
                    # below. Dropping this flag would freeze the pixel on
                    # however few samples it had (see ``_PT_ACC_STOCH``).
                    stoch = 1
                    vis = ti.Vector([1.0] * (SHADOW_VIS_CHANNELS * vis_lights))
                    if ti.static(auth_sampled != 0):
                        # ------------------------------------------------------
                        # Sampled mode (roadmap section 6a-bis). The surface is
                        # lit from ``A`` direction-less rows filled exactly as the
                        # lit branch fills them, plus ``S`` rows DRAWN from the
                        # authored table, each carrying ``1 / (S * p)`` on its
                        # radiance. ``_run_frag_pipeline`` is handed slots instead
                        # of rows -- through a view that re-indexes and scales, so
                        # neither it nor any stage knows the difference.
                        auth_s = ti.max(
                            ti.cast(nee_meta[_NM_AUTHORED_SAMPLES] + 0.5, ti.i32), 0)
                        auth_n = ti.cast(
                            nee_meta[_NM_AUTHORED_COUNT] + 0.5, ti.i32)
                        # The authored table's ROWS sit after the sampled
                        # entries and the ambient tail; its CDF sits directly
                        # after the sampled entries' CDF, because the ambient
                        # rows have no CDF of their own (they are the
                        # deterministic fill). Two bases, deliberately -- one
                        # base for both reads the wrong bracket and runs off
                        # the end of ``nee_cdf``.
                        auth_base = num_nee + amb_count
                        lrow = ti.Vector([0] * vis_lights)
                        lscale = ti.Vector([0.0] * vis_lights)
                        a_use = ti.min(amb_count, vis_lights)
                        n_slots = ti.min(a_use + auth_s, vis_lights)
                        for j in range(a_use):
                            lrow[j] = nee_ref[num_nee + j, 1]
                            lscale[j] = 1.0
                        recv_a = 1
                        if pid < _USER_PIPELINE_BASE:
                            if tri_mat.shape[2] > _MAT_NO_SHADOW_RECEIVE:
                                if tri_mat[f % tri_mat.shape[0], prim,
                                           _MAT_NO_SHADOW_RECEIVE] > 0.5:
                                    recv_a = 0
                        sorigin_a = _pt_offset_ray_origin(
                            hit_p, fnrm if fnrm.dot(-rd) >= 0.0 else -fnrm)
                        inv_s = 1.0 / ti.cast(ti.max(auth_s, 1), ti.f32)
                        for ls in range(auth_s):
                            s = a_use + ls
                            if s < n_slots:
                                # The crossing's OWN next-event pairs: a crossing
                                # is either lit or authored, never both, so this
                                # branch reuses the block the lit branch would
                                # have spent (module docstring's table) rather
                                # than claiming a dimension of its own.
                                pair_sel = pair_cross0 + 2 * ls
                                u_sel = pt_sample_2d_seeded(path_seed, pair_sel,
                                                            s_index)
                                u_pt = pt_sample_2d_seeded(path_seed,
                                                           pair_sel + 1, s_index)
                                k_row, p_sel = _pt_pick_authored_row(
                                    nee_cdf, num_nee, auth_n, u_sel[0])
                                li = nee_ref[auth_base + k_row, 1]
                                w = 0.0
                                if p_sel > 1e-12:
                                    w = inv_s / p_sel
                                lrow[s] = li
                                lscale[s] = w
                                if ti.static(shadows != 0):
                                    if (recv_a == 1) and (w > 0.0):
                                        wi, ldist, valid = _pt_light_sample_point(
                                            light_pos, light_col, f, li, hit_p,
                                            u_pt[0], u_pt[1])
                                        if valid == 1:
                                            v3 = _pt_nee_visibility(
                                                refit, shadow_mode,
                                                has_tri, has_bez,
                                                sorigin_a, wi, ldist, f, ff,
                                                pixel_size_per_t, base_dist,
                                                layer_offset_triangles,
                                                t_nodes, t_node_miss, t_leaf_prim,
                                                t_leaf_tspan, t_first_leaf,
                                                tri_pos, tri_colors, tri_uvs,
                                                tri_tex_meta, textures, tri_extra,
                                                num_colored_triangles,
                                                b_nodes, b_node_miss, b_leaf_prim,
                                                b_leaf_tspan, b_first_leaf,
                                                circuit_meta, circuit_colors,
                                                circuit_border_colors,
                                                edges_2d, edge_accel)
                                            base = light_vis_index(s, 0)
                                            vis[base] = v3[0]
                                            vis[base + 1] = v3[1]
                                            vis[base + 2] = v3[2]
                        # Built AFTER the fill only for readability: the vectors
                        # are mutated in place, so a view built before it would
                        # read the same values.
                        lpos_v = ti.static(_SampledLightView(light_pos, lrow))
                        lcol_v = ti.static(
                            _SampledLightView(light_col, lrow, lscale))
                        local = _run_frag_pipeline(
                            frag_pipelines, tri_pids, prim, f, hit_p, -rd,
                            snrm, fnrm, albedo3, color[3],
                            lpos_v, lcol_v, n_slots, tri_mat_id,
                            tri_mat, shadows, vis, cam_pos)
                    else:
                        if ti.static(shadows != 0):
                            recv_a = 1
                            if pid < _USER_PIPELINE_BASE:
                                if tri_mat.shape[2] > _MAT_NO_SHADOW_RECEIVE:
                                    if tri_mat[f % tri_mat.shape[0], prim,
                                               _MAT_NO_SHADOW_RECEIVE] > 0.5:
                                        recv_a = 0
                            if recv_a == 1:
                                sorigin_a = _pt_offset_ray_origin(
                                    hit_p,
                                    fnrm if fnrm.dot(-rd) >= 0.0 else -fnrm)
                                for li in range(num_lights):
                                    if li < vis_lights:
                                        u1 = _pt_rng_seeded(
                                            path_seed, s_index,
                                            processed * 64 + li, 2)
                                        u2 = _pt_rng_seeded(
                                            path_seed, s_index,
                                            processed * 64 + li, 3)
                                        wi, ldist, valid = _pt_light_sample_point(
                                            light_pos, light_col, f, li, hit_p,
                                            u1, u2)
                                        if valid == 1:
                                            v3 = _pt_nee_visibility(
                                                refit, shadow_mode,
                                                has_tri, has_bez,
                                                sorigin_a, wi, ldist, f, ff,
                                                pixel_size_per_t, base_dist,
                                                layer_offset_triangles,
                                                t_nodes, t_node_miss, t_leaf_prim,
                                                t_leaf_tspan, t_first_leaf,
                                                tri_pos, tri_colors, tri_uvs,
                                                tri_tex_meta, textures, tri_extra,
                                                num_colored_triangles,
                                                b_nodes, b_node_miss, b_leaf_prim,
                                                b_leaf_tspan, b_first_leaf,
                                                circuit_meta, circuit_colors,
                                                circuit_border_colors,
                                                edges_2d, edge_accel)
                                            base = light_vis_index(li, 0)
                                            vis[base] = v3[0]
                                            vis[base + 1] = v3[1]
                                            vis[base + 2] = v3[2]
                        local = _run_frag_pipeline(
                            frag_pipelines, tri_pids, prim, f, hit_p, -rd,
                            snrm, fnrm, albedo3, color[3],
                            light_pos, light_col, num_lights, tri_mat_id,
                            tri_mat, shadows, vis, cam_pos)
                    if ti.static(len(frag_scatters) > 0):
                        # Custom ray continuation. The pid switch mirrors
                        # ``_run_frag_scatter``'s, minus its default-scatter
                        # fallback: a user pipeline that supplies no scatter
                        # keeps the path tracer's own importance-sampled
                        # lobes rather than inheriting the deterministic
                        # renderer's opacity/Fresnel continuation. The
                        # ``shaded`` argument is the pipeline output just
                        # computed, exactly what ``wavefront_shade`` hands
                        # it, and ``refraction`` is 1 because this renderer
                        # can carry a transmitted branch (it samples one of
                        # the three rather than splitting).
                        for pi in ti.static(range(len(frag_scatters))):
                            # ``bool(func)`` rather than an ``is not``
                            # comparison -- see ``_run_frag_scatter``.
                            if ti.static(bool(frag_scatters[pi])):
                                if pid == _USER_PIPELINE_BASE + pi:
                                    (sc_contrib, sc_pass_w, sc_refl_o,
                                     sc_refl_d, sc_refl_w, sc_trans_o,
                                     sc_trans_d, sc_trans_w) = \
                                        frag_scatters[pi](
                                            rd, snrm, fnrm, hit_p, local,
                                            albedo3, alpha, metalness, ior,
                                            T, tri_mat, f, prim,
                                            bounces_left, 1)
                                    sc_on = 1
                                    stoch = 1

                indirect_path = bounces_left < max_b
                add = thru * alpha * local
                if ti.static(len(frag_scatters) > 0):
                    if sc_on == 1:
                        # The scatter committed this crossing's radiance
                        # itself: ``contrib`` already folds in the shaded
                        # colour it was handed and the surface's coverage
                        # (the deterministic renderer likewise adds
                        # ``weight * contrib``), so it REPLACES
                        # ``alpha * local`` rather than scaling it.
                        add = ti.math.vec4(thru[0] * sc_contrib[0],
                                           thru[1] * sc_contrib[1],
                                           thru[2] * sc_contrib[2],
                                           thru[3] * sc_contrib[3])
                if indirect_path and (firefly_clamp > 0.0):
                    add = ti.min(add, ti.math.vec4(firefly_clamp,
                                                   firefly_clamp,
                                                   firefly_clamp,
                                                   firefly_clamp))
                acc += add

                # AOV guides: this crossing's contribution to the delta
                # prefix, at the SAME weights the radiance composite uses
                # (no firefly clamp -- a guide, not radiance). A suppressed
                # closed-shell exit has alpha 0 and adds nothing.
                if aov_open == 1:
                    aov_alb += ti.math.vec3(thru[0], thru[1], thru[2]) \
                        * (alpha * albedo3)
                    aov_nrm += (thru[3] * alpha) * shade_n

                # ----------------------------------------------------------
                # Continuation A: the user's scatter, as a delta lobe.
                # ----------------------------------------------------------
                # The scatter returns three branches (pass-through, reflect,
                # transmit) with vec3 throughput multipliers. Paths never
                # split here (contract 3), so exactly one branch is picked --
                # with the same importance weighting the built-in lobe pick
                # below uses, from the same ``u_lobe`` draw -- and the path
                # continues with ``prev_pdf = 0``: it is the user's direction
                # and the user's density, so nothing MIS-covers what it finds
                # next, which is precisely the contract refraction and the
                # tinted pane already get. A pass-through keeps peeling this
                # batch; a reflect or transmit ends the camera segment and
                # spends a bounce, as any scatter does.
                #
                # ``sc_next`` carries the pass-through out to a RUNTIME ``if``
                # below rather than continuing from inside the static gate: a
                # ``continue`` under a compile-time gate is emitted bare and
                # invalidates the SPIR-V module (test_kernel_control_flow.py).
                sc_next = 0
                if ti.static(len(frag_scatters) > 0):
                    if sc_on == 1:
                        cw_pass = ti.max(sc_pass_w[0],
                                         ti.max(sc_pass_w[1], sc_pass_w[2]))
                        cw_refl = ti.max(sc_refl_w[0],
                                         ti.max(sc_refl_w[1], sc_refl_w[2]))
                        cw_trans = ti.max(sc_trans_w[0],
                                          ti.max(sc_trans_w[1],
                                                 sc_trans_w[2]))
                        if bounces_left <= 0:
                            # Out of bounces: only the pass-through survives,
                            # the rule ``wavefront_shade`` applies to the same
                            # returned branches.
                            cw_refl = 0.0
                            cw_trans = 0.0
                        cw_pass = ti.max(cw_pass, 0.0)
                        cw_refl = ti.max(cw_refl, 0.0)
                        cw_trans = ti.max(cw_trans, 0.0)
                        cw_sum = cw_pass + cw_refl + cw_trans
                        if cw_sum <= 1e-6:
                            # The scatter absorbed the ray: nothing continues
                            # and the background must not show through, in
                            # colour or in coverage (see the built-in
                            # absorption below).
                            t_alpha = 0.0
                            absorbed = True
                            aov_open = 0
                            done = True
                            break
                        # The crossing's stratified lobe-select draw -- the
                        # same pair the built-in lobe pick below uses (the
                        # two are mutually exclusive: a custom scatter ends
                        # this crossing).
                        u_c = pt_sample_2d_seeded(
                            path_seed, pair_cross0 + 2 * n_ls, s_index)[0]
                        pick_c = u_c * cw_sum
                        c_tint = ti.math.vec3(0.0, 0.0, 0.0)
                        c_ro = ti.math.vec3(0.0, 0.0, 0.0)
                        c_rd = ti.math.vec3(0.0, 0.0, 0.0)
                        c_scatter = 0
                        if pick_c < cw_pass:
                            c_tint = sc_pass_w \
                                * (cw_sum / ti.max(cw_pass, 1e-6))
                        elif pick_c < cw_pass + cw_refl:
                            c_tint = sc_refl_w \
                                * (cw_sum / ti.max(cw_refl, 1e-6))
                            c_ro = sc_refl_o
                            c_rd = sc_refl_d
                            c_scatter = 1
                        else:
                            c_tint = sc_trans_w \
                                * (cw_sum / ti.max(cw_trans, 1e-6))
                            c_ro = sc_trans_o
                            c_rd = sc_trans_d
                            c_scatter = 1
                        c_mean = (c_tint[0] + c_tint[1] + c_tint[2]) / 3.0
                        thru = ti.math.vec4(thru[0] * c_tint[0],
                                            thru[1] * c_tint[1],
                                            thru[2] * c_tint[2],
                                            thru[3] * c_mean)
                        if c_scatter == 0:
                            # Pass-through: the camera segment survives and
                            # its transparency takes the same weight the
                            # throughput did. Peel on (via ``sc_next``).
                            t_alpha *= c_mean
                            t_prev = t_hit
                            layer_prev = hit_layer
                            sc_next = 1
                            if ti.max(thru[0],
                                      ti.max(thru[1],
                                             thru[2])) < min_weight:
                                done = True
                                break
                        else:
                            # A scatter: the camera segment ends here, and
                            # what the continuation finds is covered by no
                            # NEE.
                            t_alpha = 0.0
                            prev_pdf = 0.0
                            bounce_ord_c = max_b - bounces_left
                            survived_c = 1
                            if bounce_ord_c >= rr_start:
                                u_rr_c = pt_sample_2d_seeded(
                                    path_seed,
                                    PAIR_BOUNCE_BASE
                                    + PAIRS_PER_BOUNCE * bounce_ord_c
                                    + _PAIR_LOBE, s_index)[1]
                                p_rr_c = ti.math.clamp(
                                    ti.max(thru[0],
                                           ti.max(thru[1], thru[2])),
                                    _PT_RR_FLOOR, 1.0)
                                if u_rr_c >= p_rr_c:
                                    survived_c = 0
                                else:
                                    thru *= 1.0 / p_rr_c
                            if survived_c == 0:
                                thru = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                                absorbed = True
                                done = True
                                break
                            if ti.max(thru[0],
                                      ti.max(thru[1],
                                             thru[2])) < min_weight:
                                done = True
                                break
                            ro = c_ro
                            rd = c_rd
                            base_dist += t_hit
                            t_prev = 0.0
                            layer_prev = 1e30
                            seam_t = -1e30
                            bounces_left -= 1
                            bounced = True
                            break
                if sc_next == 1:
                    # The custom scatter passed this crossing through; the
                    # built-in lobe pick below is not its business.
                    continue

                # ----------------------------------------------------------
                # Continuation: pass-through | diffuse | specular | transmit.
                # ----------------------------------------------------------
                one3 = ti.math.vec3(1.0, 1.0, 1.0)
                w_pass = 1.0 - alpha
                e_diff = ti.math.vec3(0.0, 0.0, 0.0)
                e_spec = ti.math.vec3(0.0, 0.0, 0.0)
                e_trans = ti.math.vec3(0.0, 0.0, 0.0)
                f0 = ti.math.vec3(0.0, 0.0, 0.0)
                diel_pass = 0.0
                if needs_normal and (bounces_left > 0):
                    # ``spec_n``, not ``shade_n``: the reflection lobe's
                    # cosine is measured from the side the ray is on.
                    n_dot_v = ti.max(spec_n.dot(-rd), 1e-4)
                    if lit:
                        # The hoisted lobes (the same values every NEE
                        # response and MIS pdf above used).
                        e_diff = e_diff_l
                        e_spec = e_spec_l
                        e_trans = e_trans_l
                        f0 = f0_l
                    elif authored:
                        e_diff = albedo3
                    elif metalness >= 0.0:
                        # Reflective / transmissive circuit or unlit surface.
                        met = ti.math.clamp(metalness, 0.0, 1.0)
                        f0 = ti.math.vec3(0.04, 0.04, 0.04) * (1.0 - met) \
                            + albedo3 * met
                        if met > 0.0:
                            e_spec = _pt_ggx_energy(f0, n_dot_v, rough)
                        if T > 1e-4:
                            _R3c, diel_pass = _material_reflectance(
                                rd, shade_n, met, ior, albedo3, T)
                            e_trans = albedo3 * (diel_pass * T)
                w_diff = alpha * ti.max(e_diff[0], ti.max(e_diff[1],
                                                          e_diff[2]))
                w_spec = alpha * ti.max(e_spec[0], ti.max(e_spec[1],
                                                          e_spec[2]))
                w_trans = alpha * ti.max(e_trans[0], ti.max(e_trans[1],
                                                            e_trans[2]))
                w_sum = w_pass + w_diff + w_spec + w_trans
                if w_sum <= 1e-6:
                    # Fully absorbed (e.g. an opaque unlit surface): nothing
                    # continues and the background must NOT show through --
                    # in color or in coverage, so the camera-segment
                    # transparency drops to zero exactly as a scatter's
                    # would (the deterministic composite reads such a hit
                    # as fully opaque).
                    t_alpha = 0.0
                    absorbed = True
                    aov_open = 0  # the absorbing surface is the guide
                    done = True
                    break
                if w_sum - w_pass > 1e-9:
                    # More than the pass-through branch is available, so the
                    # pick below is a real draw -- and whichever branch wins,
                    # this sample is one realisation of several. An unlit
                    # transparent stack has ``w_sum == w_pass`` and stays
                    # deterministic; an unlit opaque absorb never gets here.
                    stoch = 1
                # Stratified lobe selection (roadmap section 7): the pick
                # drives which lobe a bounce explores, so it gets its own
                # crossing-indexed Sobol pair rather than the white noise it
                # used to draw. Only the x component is consumed.
                u_lobe = pt_sample_2d_seeded(
                    path_seed, pair_cross0 + 2 * n_ls, s_index)[0]
                pick = u_lobe * w_sum
                if pick < w_pass:
                    # Deterministic in an unlit stack (probability 1 there).
                    scale = (1.0 - alpha) * (w_sum / w_pass)
                    thru *= scale
                    t_alpha *= scale
                    t_prev = t_hit
                    layer_prev = hit_layer
                    if ti.max(thru[0], ti.max(thru[1], thru[2])) < min_weight:
                        done = True
                        break
                else:
                    # A scatter: the camera segment ends here (coverage
                    # freezes; see pt_reduce).
                    t_alpha = 0.0
                    bounce_ord = max_b - bounces_left
                    pair_base = PAIR_BOUNCE_BASE \
                        + PAIRS_PER_BOUNCE * bounce_ord
                    u_dir = pt_sample_2d_seeded(
                        path_seed, pair_base + _PAIR_BSDF_DIR, s_index)
                    new_rd = rd
                    new_ro = hit_p
                    ok = 1
                    if pick < w_pass + w_diff:
                        # A diffuse pick is non-delta: the AOV prefix ends at
                        # this crossing (whose albedo/normal it just added).
                        aov_open = 0
                        p_sel = w_diff / w_sum
                        new_rd = _pt_cosine_direction(shade_n, u_dir)
                        tint = e_diff * (alpha / ti.max(p_sel, 1e-6))
                        thru = ti.math.vec4(
                            thru[0] * tint[0], thru[1] * tint[1],
                            thru[2] * tint[2],
                            thru[3] * (alpha / ti.max(p_sel, 1e-6))
                            * ((e_diff[0] + e_diff[1] + e_diff[2]) / 3.0))
                        new_ro = _pt_offset_ray_origin(
                            hit_p,
                            fnrm if fnrm.dot(new_rd) >= 0.0 else -fnrm)
                        # MIS state: only a lit vertex ran the NEE block, so
                        # only its sampled direction carries a pdf for the
                        # next emitter hit to weight against.
                        prev_pdf = 0.0
                        if lit:
                            _fc_d, prev_pdf = _pt_lit_f_pdf(
                                e_diff_l, e_spec_l, f0_l, rough, shade_n,
                                rd, new_rd, wl_pass, wl_diff, wl_spec,
                                wl_trans)
                    elif pick < w_pass + w_diff + w_spec:
                        # GGX is non-delta (any roughness): close the prefix.
                        aov_open = 0
                        p_sel = w_spec / w_sum
                        # VNDF sample about the RAY-FACING normal (see
                        # ``spec_n``): about ``shade_n`` an interior hit --
                        # the exit face of a refracting solid, total internal
                        # reflection included -- puts every sampled direction
                        # below the horizon and rejects it.
                        t_b, b_b = _pt_onb(spec_n)
                        wo_l = ti.math.vec3(t_b.dot(-rd), b_b.dot(-rd),
                                            ti.max(spec_n.dot(-rd), 1e-4))
                        a_g = ti.max(rough * rough, 1e-4)
                        h_l = _pt_vndf_half_vector(wo_l, a_g, u_dir)
                        h_w = (t_b * h_l[0] + b_b * h_l[1]
                               + spec_n * h_l[2]).normalized()
                        new_rd = (rd - 2.0 * rd.dot(h_w) * h_w).normalized()
                        n_dot_l2 = spec_n.dot(new_rd)
                        if n_dot_l2 <= 1e-5:
                            ok = 0
                        else:
                            v_dot_h = ti.max((-rd).dot(h_w), 1e-4)
                            fres = f0 + (one3 - f0) \
                                * ti.pow(1.0 - v_dot_h, 5.0)
                            lam_v = _pt_smith_lambda(wo_l[2], a_g)
                            lam_l = _pt_smith_lambda(n_dot_l2, a_g)
                            g_ratio = (1.0 + lam_v) \
                                / ti.max(1.0 + lam_v + lam_l, 1e-6)
                            # Turquin compensation on the sampled lobe.
                            e1 = _env_brdf_approx(one3, wo_l[2], rough)
                            e1s = ti.math.clamp(e1[0], 1e-3, 1.0)
                            comp = one3 + f0 * ((1.0 - e1s) / e1s)
                            tint = fres * g_ratio * comp \
                                * (alpha / ti.max(p_sel, 1e-6))
                            gmean = (tint[0] + tint[1] + tint[2]) / 3.0
                            thru = ti.math.vec4(thru[0] * tint[0],
                                                thru[1] * tint[1],
                                                thru[2] * tint[2],
                                                thru[3] * gmean)
                            new_ro = _pt_offset_ray_origin(
                                hit_p,
                                fnrm if fnrm.dot(new_rd) > 0.0 else -fnrm)
                            prev_pdf = 0.0
                            if lit:
                                _fc_s, prev_pdf = _pt_lit_f_pdf(
                                    e_diff_l, e_spec_l, f0_l, rough,
                                    shade_n, rd, new_rd, wl_pass, wl_diff,
                                    wl_spec, wl_trans)
                    else:
                        p_sel = w_trans / w_sum
                        if htype == 1:
                            entering = rd.dot(fnrm) < 0.0
                            rel = _relative_ior(rs_sca, r, ior, entering, 1)
                            new_rd = _refract_ray(rd, shade_n, rel)
                            _write_ior_stack(rs_sca, r, r, ior, entering,
                                             1, 1)
                            new_ro = _offset_transmitted_origin(
                                hit_p, new_rd, fnrm, shade_n)
                        else:
                            # Zero-thickness pane: unbent, tinted.
                            new_rd = rd
                            new_ro = _pt_offset_ray_origin(hit_p, rd)
                        tint = e_trans * (alpha / ti.max(p_sel, 1e-6))
                        gmean = (tint[0] + tint[1] + tint[2]) / 3.0
                        thru = ti.math.vec4(thru[0] * tint[0],
                                            thru[1] * tint[1],
                                            thru[2] * tint[2],
                                            thru[3] * gmean)
                        # Refraction / a tinted pane is a delta lobe: what
                        # it finds next is not MIS-covered by any NEE.
                        prev_pdf = 0.0
                    if ok == 0:
                        # Rejected sample direction: absorbed, not escaped.
                        absorbed = True
                        done = True
                        break
                    # Russian roulette past the configured depth.
                    survived = 1
                    if bounce_ord >= rr_start:
                        u_rr = pt_sample_2d_seeded(
                            path_seed, pair_base + _PAIR_LOBE, s_index)[1]
                        p_rr = ti.math.clamp(
                            ti.max(thru[0], ti.max(thru[1], thru[2])),
                            _PT_RR_FLOOR, 1.0)
                        if u_rr >= p_rr:
                            survived = 0
                        else:
                            thru *= 1.0 / p_rr
                    if survived == 0:
                        thru = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                        absorbed = True
                        done = True
                        break
                    if ti.max(thru[0], ti.max(thru[1], thru[2])) < min_weight:
                        done = True
                        break
                    ro = new_ro
                    rd = new_rd
                    base_dist += t_hit
                    t_prev = 0.0
                    layer_prev = 1e30
                    seam_t = -1e30
                    bounces_left -= 1
                    bounced = True
                    break

            if (not done) and (not bounced) and (num_hits < kbuf):
                # Fewer hits than the gather could hold: the peel is complete
                # and the leftover throughput shows the background.
                done = True
            if processed >= max_surfaces_per_ray:
                # Truncation, not completion (see truncation.py): a ray still
                # active here is being cut short by the ceiling.
                if not done:
                    ti.atomic_add(pt_stats[PT_STAT_TRUNC_SURFACES], 1)
                done = True

            for k in ti.static(range(3)):
                rs_ro[r, k] = ro[k]
                rs_rd[r, k] = rd[k]
            for k in ti.static(range(4)):
                pt_thru[r, k] = thru[k]
                pt_acc[r, k] += acc[k]
            rs_sca[r, 0] = t_alpha
            rs_sca[r, 1] = t_prev
            rs_sca[r, 2] = layer_prev
            rs_sca[r, 3] = seam_t
            rs_sca[r, 4] = base_dist
            rs_sca[r, _SCA_PREV_PDF] = prev_pdf
            rs_int[r, 0] = bounces_left
            rs_int[r, 1] = processed
            rs_int[r, 2] = _DONE if done else _ACTIVE
            if stoch == 1:
                # Sticky: only ever written 1.0, never cleared, so a path that
                # became stochastic on an earlier launch stays flagged.
                pt_acc[r, _PT_ACC_STOCH] = 1.0
            for q in ti.static(range(_SHELL_RING_SLOTS)):
                rs_int[r, _INT_RING0 + q] = ring[q]
            if aov_on:
                for k in ti.static(range(3)):
                    pt_aov[r, _AOV_ALB + k] += aov_alb[k]
                    pt_aov[r, _AOV_NRM + k] += aov_nrm[k]
                pt_aov[r, _AOV_CLOSED] = 1.0 - ti.cast(aov_open, ti.f32)
            if done:
                leftover = thru
                if absorbed:
                    # An absorbed path (opaque surface, RR kill, rejected
                    # sample direction) carries nothing to the background.
                    # A ceiling-truncated path keeps its leftover, matching
                    # the wavefront's documented degrade.
                    leftover = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                if (env_w > 0) and (ti.max(leftover[0],
                                           ti.max(leftover[1],
                                                  leftover[2])) > 0.0):
                    # Environment escape: the leftover throughput samples
                    # the map in the ray's own direction (mirrors and GI
                    # see the sky, exactly the deterministic wavefront's
                    # retire rule) instead of showing the prefilled
                    # background, and the sample reads opaque. A smooth
                    # BSDF continuation MIS-weights against the env NEE
                    # that also covered its direction.
                    ec = _sample_env_map(f, rd, env_off, env_w, env_h,
                                         env_intensity, textures)
                    w_env = 1.0
                    if (env_share > 0.0) and (prev_pdf > 0.0):
                        p_e = env_share * _pt_env_pdf_sa(
                            rd, env_cdf, cdf_h, cdf_w) \
                            * ti.cast(n_ls, ti.f32)
                        w_env = prev_pdf * prev_pdf \
                            / ti.max(prev_pdf * prev_pdf + p_e * p_e,
                                     1e-20)
                    env_add = ti.math.vec3(leftover[0] * ec[0],
                                           leftover[1] * ec[1],
                                           leftover[2] * ec[2]) * w_env
                    if (bounces_left < max_b) and (firefly_clamp > 0.0):
                        env_add = ti.min(env_add,
                                         ti.math.vec3(firefly_clamp,
                                                      firefly_clamp,
                                                      firefly_clamp))
                    for k in ti.static(range(3)):
                        pt_acc[r, k] += env_add[k]
                    if (aov_on) and (aov_open == 1):
                        # An escaping delta prefix sees the map: credit it
                        # to the albedo guide (unweighted -- a guide).
                        for k in ti.static(range(3)):
                            pt_aov[r, _AOV_ALB + k] += leftover[k] * ec[k]
                    leftover = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                    t_alpha = 0.0
                for k in ti.static(range(4)):
                    pt_acc[r, _PT_ACC_LEFTOVER + k] = leftover[k]
                pt_acc[r, _PT_ACC_ALPHA] = t_alpha
                if (aov_on) and (aov_open == 1):
                    # Whatever is left shows the background; the host folds
                    # this weight with the prefilled background color.
                    for k in ti.static(range(3)):
                        pt_aov[r, _AOV_BGW + k] += leftover[k]
        else:
            # No surface this segment: the path escapes to the background
            # (or, with an environment map, to the map in its direction).
            (n_ls, env_off, env_w, env_h, env_intensity, env_share,
             cdf_h, cdf_w, aov_on) = _pt_meta_escape(nee_meta)
            leftover_e = ti.math.vec4(pt_thru[r, 0], pt_thru[r, 1],
                                      pt_thru[r, 2], pt_thru[r, 3])
            t_alpha_e = rs_sca[r, 0]
            aov_open_e = 0
            if aov_on:
                if pt_aov[r, _AOV_CLOSED] < 0.5:
                    aov_open_e = 1
            if (env_w > 0) and (ti.max(leftover_e[0],
                                       ti.max(leftover_e[1],
                                              leftover_e[2])) > 0.0):
                g = ray_offset + rs_pix[r]
                f = time_start + g // pixels_per_frame
                rd = ti.math.vec3(rs_rd[r, 0], rs_rd[r, 1], rs_rd[r, 2])
                prev_pdf_e = rs_sca[r, _SCA_PREV_PDF]
                ec = _sample_env_map(f, rd, env_off, env_w, env_h,
                                     env_intensity, textures)
                w_env = 1.0
                if (env_share > 0.0) and (prev_pdf_e > 0.0):
                    p_e = env_share * _pt_env_pdf_sa(rd, env_cdf,
                                                     cdf_h, cdf_w) \
                        * ti.cast(n_ls, ti.f32)
                    w_env = prev_pdf_e * prev_pdf_e \
                        / ti.max(prev_pdf_e * prev_pdf_e + p_e * p_e,
                                 1e-20)
                env_add = ti.math.vec3(leftover_e[0] * ec[0],
                                       leftover_e[1] * ec[1],
                                       leftover_e[2] * ec[2]) * w_env
                if (rs_int[r, 0] < rs_int[r, 4]) \
                        and (firefly_clamp > 0.0):
                    env_add = ti.min(env_add,
                                     ti.math.vec3(firefly_clamp,
                                                  firefly_clamp,
                                                  firefly_clamp))
                for k in ti.static(range(3)):
                    pt_acc[r, k] += env_add[k]
                if aov_open_e == 1:
                    for k in ti.static(range(3)):
                        pt_aov[r, _AOV_ALB + k] += leftover_e[k] * ec[k]
                leftover_e = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                t_alpha_e = 0.0
            for k in ti.static(range(4)):
                pt_acc[r, _PT_ACC_LEFTOVER + k] = leftover_e[k]
            pt_acc[r, _PT_ACC_ALPHA] = t_alpha_e
            if aov_open_e == 1:
                for k in ti.static(range(3)):
                    pt_aov[r, _AOV_BGW + k] += leftover_e[k]
            rs_int[r, 2] = _DONE


#: What ``pt_shade`` binds through the arena, in offset-table order:
#: ``aoff[i]`` is the i-th entry's element offset into its dtype's
#: buffer and ``ashp`` holds their shapes end to end. The kernel's
#: binding prologue reads those slots by literal index, so the two
#: are one edit apart -- ``tests/unit_tests/test_arena_args.py``
#: fails if they stop agreeing.
_PT_SHADE_ARENA = (
    ("t_node_miss", "i32", 1),
    ("t_leaf_prim", "i32", 1),
    ("t_leaf_tspan", "i32", 1),
    ("tri_pos", "f32", 3),
    ("tri_norm", "f32", 3),
    ("tri_extra", "f32", 3),
    ("tri_colors", "f32", 4),
    ("tri_uvs", "f32", 3),
    ("tri_tex_meta", "i32", 2),
    ("textures", "f32", 3),
    ("b_node_miss", "i32", 1),
    ("b_leaf_prim", "i32", 1),
    ("b_leaf_tspan", "i32", 1),
    ("circuit_meta", "f32", 3),
    ("circuit_colors", "f32", 4),
    ("circuit_border_colors", "f32", 4),
    ("edges_2d", "f32", 3),
    ("edge_accel", "i32", 1),
    ("tri_mat_id", "i32", 2),
    ("tri_mat", "f32", 3),
    ("light_pos", "f32", 3),
    ("light_col", "f32", 3),
    ("pixel_world_scale", "f32", 1),
    ("cam_origin", "f32", 2),
    ("nee_cdf", "f32", 1),
    ("nee_ref", "i32", 2),
    ("nee_meta", "f32", 1),
    ("tri_emit_prob", "f32", 1),
    ("env_cdf", "f32", 2),
    ("tri_shell", "i32", 2),
    ("tri_emit_entry", "i32", 1),
    ("lt_node_f", "f32", 3),
    ("lt_node_i", "i32", 3),
    ("lt_entry_leaf", "i32", 2),
    ("lt_frame", "i32", 1),
    ("nee_inf_cdf", "f32", 1),
    ("nee_inf_ref", "i32", 1),
    ("pt_emit_falloff", "f32", 2),
)

#: The argument list every launch site passes. Unchanged by the
#: conversion -- that is the point of the wrapper below.
_PT_SHADE_PARAMS = (
    "active", "num_active", "t_nodes", "t_node_miss", "t_leaf_prim",
    "t_leaf_tspan", "t_first_leaf", "tri_pos", "tri_norm", "tri_extra",
    "tri_colors", "tri_uvs", "tri_tex_meta", "textures",
    "num_colored_triangles", "b_nodes", "b_node_miss", "b_leaf_prim",
    "b_leaf_tspan", "b_first_leaf", "circuit_meta", "circuit_colors",
    "circuit_border_colors", "edges_2d", "edge_accel", "tri_mat_id",
    "tri_mat", "light_pos", "light_col", "num_lights", "pixel_world_scale",
    "layer_offset_triangles", "cam_origin", "refit", "has_tri", "has_bez",
    "shadows", "shadow_mode",
    "vis_lights", "auth_sampled",
    "frag_pipelines", "frag_scatters", "tri_pids", "seed_root",
    "sample_base",
    "tile_pixels", "rr_start", "firefly_clamp", "time_start", "width",
    "height", "ray_offset", "rs_ro", "rs_rd", "rs_sca", "rs_int", "rs_pix",
    "hit_f", "hit_i", "pt_thru", "pt_acc", "pt_stats", "nee_cdf", "nee_ref",
    "nee_meta", "tri_emit_prob", "env_cdf", "tri_shell", "pt_aov",
    "tri_emit_entry", "lt_node_f", "lt_node_i", "lt_entry_leaf", "lt_frame",
    "nee_inf_cdf", "nee_inf_ref", "pt_emit_falloff",
)

_pt_shade_launch = arena_packed(
    __name__, "pt_shade_arena", _PT_SHADE_PARAMS, _PT_SHADE_ARENA)


def pt_shade(*args):
    """Pack the arena-bound arguments, then launch ``pt_shade_arena``.

    Takes the argument list this kernel had before it was converted to
    the arena calling convention, so no launch site changed; see
    `arena_args_taichi`.
    """
    return _pt_shade_launch(*args)


@ti.kernel
def pt_reduce(sample_base: ti.i32, tile_pixels: ti.i32, wave_samples: ti.i32,
              transparent: ti.i32, adaptive: ti.i32,
              width: ti.i32, height: ti.i32,
              pix_list: ti.types.ndarray(),
              out: ti.types.ndarray(), pt_acc: ti.types.ndarray(),
              accum: ti.types.ndarray(), accum_odd: ti.types.ndarray()):
    """Fold one wave's per-path rows into the chunk's per-pixel sample sums.

    One thread per wave pixel walks its own wave samples in index order --
    exclusive slots, no atomics, a fixed summation order, because no path
    splits today (not because frames are promised to match). The background
    (prefilled into ``out`` at byte scale) enters here through each path's
    leftover throughput; a sample's alpha is ``1 - t_a * (1 - bg_alpha)``
    where ``t_a`` is the deterministically-composited camera-segment
    transparency, so alpha matches the deterministic renderer's compositing
    contract in expectation (exactly, on scatter-free content).

    ``pix_list[p_local]`` is the wave's pixel list (see ``pt_generate``): the
    global flat cell each column belongs to. ``sample_base`` is the sample
    index the wave starts at, so slot ``k`` of a pixel carries sample
    ``sample_base + k``.

    With ``adaptive`` on, ``accum_odd[f_rel, p]`` collects the two things the
    host's stopping rule needs (section 2 of
    ``DESIGN_path_tracer_roadmap.md``): columns 0-2 are the RGB of the ODD
    sample indices, which subtracted from ``accum`` give the two half-sums,
    and column 3 counts this pixel's STOCHASTIC samples -- the paths that took
    any random decision beyond the sub-pixel jitter (``_PT_ACC_STOCH``). A
    pixel with a non-zero count is never stopped early, whatever its halves
    say. ``accum`` itself keeps the full sums and is accumulated in exactly
    the order it always was, so the uniform arm (``adaptive == 0``,
    ``accum_odd`` a dummy) is byte-identical.
    """
    pixels_per_frame = width * height
    for p_local in range(tile_pixels):
        g = pix_list[p_local]
        f_rel = g // pixels_per_frame
        p = g - f_rel * pixels_per_frame
        sum_acc = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        sum_leftover = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        odd_acc = ti.math.vec3(0.0, 0.0, 0.0)
        odd_leftover = ti.math.vec3(0.0, 0.0, 0.0)
        stoch_count = 0.0
        sum_t_alpha = 0.0
        for k in range(wave_samples):
            r = k * tile_pixels + p_local
            for c in ti.static(range(4)):
                sum_acc[c] += pt_acc[r, c]
                sum_leftover[c] += pt_acc[r, _PT_ACC_LEFTOVER + c]
            if adaptive != 0:
                stoch_count += pt_acc[r, _PT_ACC_STOCH]
                if (sample_base + k) % 2 == 1:
                    for c in ti.static(range(3)):
                        odd_acc[c] += pt_acc[r, c]
                        odd_leftover[c] += pt_acc[r, _PT_ACC_LEFTOVER + c]
            sum_t_alpha += pt_acc[r, _PT_ACC_ALPHA]
        background = ti.math.vec4(
            ti.cast(out[f_rel, p, 0], ti.f32),
            ti.cast(out[f_rel, p, 1], ti.f32),
            ti.cast(out[f_rel, p, 2], ti.f32),
            ti.cast(out[f_rel, p, 3], ti.f32)) / 255.0
        for c in ti.static(range(4)):
            accum[f_rel, p, c] += sum_acc[c] + sum_leftover[c] * background[c]
        if adaptive != 0:
            for c in ti.static(range(3)):
                accum_odd[f_rel, p, c] += odd_acc[c] \
                    + odd_leftover[c] * background[c]
            accum_odd[f_rel, p, 3] += stoch_count
        if transparent != 0:
            bg_alpha = ti.cast(out[f_rel, p, 4], ti.f32) / 255.0
            accum[f_rel, p, 4] += ti.cast(wave_samples, ti.f32) \
                - sum_t_alpha * (1.0 - bg_alpha)
