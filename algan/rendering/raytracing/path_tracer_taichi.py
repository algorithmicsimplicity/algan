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
    (throughput-weighted, never stochastic alpha). Unlit interiors therefore
    avoid alpha-selection noise; jittered edges and stochastic illumination
    can still differ from the deterministic renderer.
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
    cosine-hemisphere diffuse, opaque GGX specular with Turquin-style
    multiple-scattering compensation, or one unified GGX dielectric
    interface with coupled, reciprocal energy compensation. Both rough
    reflection and refraction combine spherical-cap VNDF sampling
    (Dupuy & Benyoub 2023) with a broad compensation lobe; smooth glass
    is delta. A MeshPhysicalMaterial's clearcoat (VNDF-sampled GGX) and
    sheen (cosine-sampled Charlie) are two more lobes of the same pick,
    layered over the base as Three.js layers them (``_PL_CC``). The coat
    also attenuates the emission under it, at both MIS ends. Transmitted
    radiance includes the relative index-squared factor, with eta-aware
    roulette and the shared nested-IOR stack and Beer-Lambert absorption.
    Homogeneous interiors sample RGB-mixture free flights, HG phase directions,
    and direct lighting with shell-aware analytic shadow transmittance. The
    same bounded walk implements subsurface scattering inside closed solids.
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
  Smith ``G2``, Fresnel, opaque-reflection Turquin compensation) for every emitter kind, plus
  the sampled continuation above. The direction-less ambient / hemisphere
  rows contribute ``e_diff * L``, the physical answer for a constant-radiance
  environment over the diffuse lobe, and no specular fill (indirect transport
  replaces that). Light units and the diffuse response are the
  deterministic renderer's, so a front-lit matte surface reads the same in
  both (``test_a_lit_lambert_surface_reads_the_same_in_both_renderers``);
  the specular lobes still differ where the deterministic renderer
  approximates (roadmap section 5). Phong has no Blinn-Phong highlight
  here: it is GGX like the other lit pipelines. Emissive triangles illuminate
  their surroundings through next-event estimation and BSDF-sampled paths,
  combined with MIS where both strategies can reach the emitter.
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
  the pipeline nor any stage is touched. This weighting is unbiased for a
  linear light sum, not for an arbitrary authored pipeline: nonlinear stages
  such as the Manim display-range clamp can introduce bias (roadmap section
  6a-bis).
* unlit (1) and bezier circuits: emission + deterministic transparency; a
  reflective or transmissive circuit spawns the matching specular / pane
  continuation. Unlit content never diffuse-scatters, which keeps text and
  vector interiors free of diffuse-scattering noise.

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

Blue noise in screen space
^^^^^^^^^^^^^^^^^^^^^^^^^^

Hashing ``(frame, pixel)`` gives neighbouring pixels independent sequences, so
the per-pixel error is WHITE noise across the image -- the spectrum both the
eye and a denoiser's convolutional prior handle worst. With
``rt_settings.pt_blue_noise`` on -- it ships OFF, and roadmap section 7 has
the measurement that decided that -- the per-pixel half of the seed comes
instead from a shipped 64x64 tile of sampler keys (``blue_noise.py``,
``scripts/generate_blue_noise_tile.py``), optimised by simulated annealing so
that low-sample-count error is distributed as blue noise (Heitz et al. 2019).
``_pt_bn_path_seed`` is the whole of it: ``path_seed =
hash(_PT_BN_SALT, tile[(y + oy) mod 64, (x + ox) mod 64])``, with ``(ox, oy)``
a toroidal shift hashed from ``(pt_seed, frame-or-0)``.

Three properties this keeps. **Purity** -- a draw is still a pure function of
``(pt_seed, frame-or-0, pixel, pair, sample index)``, now plus fixed shipped
data. **Stratification** -- ``pt_sample_2d_seeded`` is untouched, so one pixel
still walks one Owen-scrambled Sobol sequence and every prefix property holds;
what changed is only WHICH pixel gets which sequence. **Convergence** -- every
pixel's estimator is the same estimator, so MSE at equal spp is unchanged;
what improves is where the error sits in the frequency domain.

Two consequences worth knowing. Pixels 64 apart share a sequence (the tile is
tiled, as it is in the paper), and the tile is optimised for the dimension
pairs a low-spp render spends its variance on -- pair 0, bounce 0's, and the
first crossing's at the SHIPPED ``max_bounces`` / ``pt_light_samples``; a
render that moves those pairs keeps a correct but less-optimised assignment.
Off, the key derivation is byte-for-byte what it was.

Dimension-pair allocation (a fixed table; keep in sync with ``pt_shade``).
``B`` is the render's ``max_bounces`` and ``L`` is ``pt_light_samples``; the
per-crossing block sits after every bounce pair because it draws per surface
CROSSING ``c`` (surface peels plus medium events, which consume no surface
hit), not per bounce. One crossing owns ``2L + 1`` pairs: the ``L``
next-event pairs it may draw, plus the lobe select every crossing draws:

=============================  ================================================
pair                           use
=============================  ================================================
0                              sub-pixel jitter (2D)
1                              lens (2D): a point on the thin lens's
                               aperture disk (``pt_generate``; unused while
                               the frame's aperture is 0)
2 + 6b + 0                     bounce ``b``: y Russian roulette (x unused --
                               the roulette draw keeps one component)
2 + 6b + 1                     bounce ``b``: BSDF direction (2D)
2 + 6b + 2, 3                  bounce ``b``: reserved (legacy light slots)
2 + 6b + 4                     medium free flight: x RGB channel, y distance;
                               seed salted by straight-subsegment ordinal. In
                               a density grid: the FIRST tentative collision's
                               x event, y distance; every later one is a hash
                               of that salted seed and the sample index
                               (``pt_volume_taichi``)
2 + 6b + 5                     medium HG continuation direction (2D)
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

A shadow connection through a density grid has no fixed number of draws at
all, so its ratio tracking is hash-only: seeded from ``path_seed``, the
connection's own NEE pair (or, on the authored summing arm, its crossing and
light row) and the sample index -- still a pure function of path identity.
"""

from algan.rendering.raytracing.arena_args_taichi import (
    ArenaView,
    arena_packed,
)
from algan.rendering.raytracing.ggx_albedo import (
    GGX_COSINE_SIZE,
    GGX_ROUGHNESS_SIZE,
)
from algan.rendering.raytracing.glass_energy import GLASS_ENERGY_SHAPE
from algan.rendering.raytracing.glass_energy_taichi import (
    _pt_glass_energy_lookup,
    _pt_glass_ms_parameters,
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
from algan.rendering.raytracing.pt_caustics_taichi import (
    _PT_CAU_CLAMP,
    _PT_CAU_COUNT,
    _PT_CAU_DIRECT,
    _PT_CAU_F,
    _PT_CAU_K,
    _PT_CAU_KIND,
    _PT_CAU_NE,
    _PT_CAU_NSIDE,
    _PT_CAU_ONE_SIDED,
    _PT_CAU_PB,
    _PT_CAU_PY,
    _PT_CAU_SPEC,
    _PT_CAU_VAL,
    _PT_CAU_WB,
    _PT_CAU_WOLD,
    _PT_CAU_X0,
    _PT_CAU_Y,
    MNEE_MAX_CONCENTRATION,
    MNEE_MAX_VERTICES,
    PT_STAT_CAUSTIC_CHAIN,
    PT_STAT_CAUSTIC_SOLVER,
    _pt_caster_id,
    _pt_cau_pend,
    _pt_cau_receiver,
    _pt_caustic_mis,
    _pt_mnee_admissible,
    _pt_mnee_covered,
    _pt_mnee_geometry,
    _pt_mnee_leg,
    _pt_mnee_probe,
    _pt_mnee_solve,
    _pt_mnee_turn,
)
from algan.rendering.raytracing.pt_media_taichi import (
    PT_MEDIA_SLOTS,
    PT_STAT_MEDIA_QUERY,
    PT_STAT_MEDIA_STACK,
    _pt_hg_pdf,
    _pt_hg_sample,
    _pt_initial_media,
    _pt_medium_coeff,
    _pt_medium_cross,
    _pt_medium_id,
    _pt_medium_sample,
    _pt_medium_stops_connection,
    _pt_medium_top,
    _pt_medium_tracked,
    _pt_medium_transmittance,
)
from algan.rendering.raytracing.pt_volume_taichi import (
    PT_STAT_VOLUME_STEPS,
    _pt_medium_volume,
    _pt_vol_seed,
    _pt_vol_track,
)
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    _M_IOR,
    _M_REFLECTIVITY,
    _M_ROUGHNESS,
    _M_TRANSMISSION,
    NODE_ARG,
    CameraProjection,
    _bezier_normal,
    _bezier_plane_point,
    _camera_point,
    _comes_after,
    _generate_ray,
    _pixel_size_base,
    _projected,
    _sample_circuit_color,
    _shadow_occluded,
    _tri_surface_point,
    depth_tie_epsilon,
    kbuf,
    max_surfaces_per_ray,
    megakernel_register_cap,
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
    _d_charlie,
    _ibl_sheen_brdf,
    _light_eval,
    _prep_normal,
    _run_frag_pipeline,
    _sided_shading_normal,
    _v_neubelt,
    light_vis_index,
)
from algan.rendering.raytracing.texture_mips_taichi import _triangle_uv_footprint
from algan.rendering.raytracing.transport_taichi import (
    _offset_ray_origin as _pt_offset_ray_origin,
)
from algan.rendering.raytracing.transport_taichi import (
    _shadow_tmax as _pt_shadow_tmax,
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
PT_ACC_WIDTH = 15
_PT_ACC_LEFTOVER = 4
_PT_ACC_ALPHA = 8
#: 1.0 once the path has made any random choice beyond the sub-pixel jitter:
#: a lens sample (``pt_generate``, whenever the frame's aperture is open), a
#: lit crossing (next-event estimation picks an emitter), an authored
#: crossing or a custom scatter, or a lobe pick with more than the
#: pass-through branch available. Sticky -- only ever written 1.0, and
#: ``pt_acc`` is zeroed once per wave, before ``pt_generate``, so it survives
#: the several ``pt_shade`` launches one path takes.
#:
#: The lens sample needs it as much as a light draw does: a small bright shape
#: out of focus is hit by only a fraction of the lens positions, so the first
#: few samples of a bokeh pixel can all miss it, agree at black, and freeze the
#: pixel there. Flagged, every pixel of a depth-of-field frame runs to the
#: ceiling (and is denoised); no rule narrower than "the aperture is open" is
#: safe, because an escaping lens ray cannot know a neighbouring lens position
#: would have hit something.
#:
#: A path that never sets it is deterministic GIVEN ITS JITTER: an unlit
#: transparent stack (pass-through at probability 1), an unlit opaque absorb,
#: and an escape to the background or environment map. Those are the pixels
#: adaptive sampling is allowed to stop early -- the half-buffer error
#: estimate cannot tell "converged at zero" from "has not found the light
#: yet", so the host requires this to be zero as well (see
#: ``path_tracer._pt_active_pixels``).
_PT_ACC_STOCH = 9
#: The emitter-MIS origin, for a path whose last scatter vertex is behind a
#: smooth transmission (glass, a tinted pane). Next-event estimation at that
#: vertex reached the emitters *through* the transmissive surface (the shadow
#: query tints rather than blocks it), so the BSDF continuation that crosses
#: the surface and finds an emitter has to be MIS-weighted against that NEE,
#: not counted at weight 1 -- weight 1 counted the same light twice, and a
#: white furnace under an index-matched slab came out 16% over unity. The
#: crossing therefore keeps ``prev_pdf``, and the solid-angle pdf NEE would
#: have had for an emissive triangle is measured from the scatter vertex
#: recorded here (columns 10-12, a world point), not from the crossing's exit
#: point. Not where the surface refracts into or out of a scattering interior:
#: NEE stops there (``_pt_medium_stops_connection``), so that crossing drops
#: ``prev_pdf`` as before. Column 13 is 1.0 while the origin is in force: set
#: by the first smooth transmission after a scatter, cleared by the next
#: scatter. Zeroed with the rest of the row before ``pt_generate``, so a
#: camera segment starts clear.
_PT_ACC_MIS_ORIGIN = 10
_PT_ACC_MIS_CARRY = 13
#: 1.0 once the path has bounced off a blurring lobe: diffuse, sheen, a
#: medium's phase function, or a GGX lobe (base, clearcoat, rough glass) whose
#: alpha is off its 1e-4 floor -- the smoothness test that already makes glass
#: delta. Sticky, and zeroed with the row before ``pt_generate``. It decides
#: which background the path's leftover throughput sees in ``pt_reduce``: a
#: path that has only gone straight on, refracted through smooth glass or
#: bounced off a mirror sees the backdrop pixel behind it, as the
#: deterministic renderer's continuation rays do; a scattered one sees the
#: frame's average backdrop. The backdrop is a screen-space picture with no
#: directions, so a per-pixel lookup after a scatter painted an image
#: background's own texels onto every diffuse surface in front of them.
_PT_ACC_SCATTERED = 14

# Device-side truncation tallies, read back by the host once per wave and fed
# through ``truncation.record_truncation`` (ceilings are counted, not silent).
# Columns 2-3 are the media module's, 4 the density-grid tracker's and 5-6
# the caustic module's (``PT_STAT_MEDIA_*``, ``PT_STAT_VOLUME_STEPS``,
# ``PT_STAT_CAUSTIC_*``).
PT_STATS_WIDTH = 7
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
#   == -1 camera segment -- the path has never scattered; emission weight 1.
#   == 0  the last scatter was a delta reflection (a mirror, glass's smooth
#         reflection) or a vertex that runs no surface NEE (authored
#         appearance, a circuit): emission weight 1 -- next-event estimation
#         never covered it.
#   > 0   the lobe-mixture pdf of the sampled direction at a lit vertex that
#         ran the NEE block: emission there is MIS-weighted against it.
# Pass-through crossings keep the value (the ray, and with it the pdf at its
# origin vertex, continues unchanged). So do smooth transmissions (refraction,
# a tinted pane): NEE's shadow ray passes them tinted, so it covered what lies
# beyond them -- see ``_PT_ACC_MIS_CARRY``.
# -2 marks an authored diffuse continuation whose direct term already
# includes area-light rows. Pass-throughs preserve it; another scatter replaces it.
_SCA_PREV_PDF = 5
# Cancels radiance's cumulative eta^2 factor ONLY for path termination.
# Column 6 is unused by the shared traverse; PT owns throughput separately.
_SCA_ETA_SCALE = 6

# rs_int columns.  The shared traverse kernel touches only columns 0-4
# (bounces_left, processed, status, num_hits, max_bounces); the four columns
# after them belong to ``pt_shade`` alone: the closed-shell opacity ring --
# the ``tri_shell`` surface ids the current straight segment has entered,
# -1 marking an empty slot.  Entering a declared closed shell composites the
# crossing and stores the id; the matching exit crossing finds the id,
# removes it, and composites nothing.  That is the per-ray limit of the
# sheet route's coverage ceiling (``solid_shell_alpha``, sheets.py), which
# spends ``max(front, back)`` coverage per (pixel, surface) in depth order
# -- and like it the ring counts CROSSINGS, not containment: a ray crossing
# one shell four times (a torus hole) attenuates twice. A straight segment inside
# more than four declared shells at once overflows the ring: the surplus
# crossing composites normally (erring toward the doubled attenuation every
# crossing produced before the ceiling existed) and is tallied in
# ``pt_stats[PT_STAT_SHELL_RING]``.
PT_INT_WIDTH = 16
_INT_MEDIA0 = 9  # four representative triangle indices, -1 = empty
_INT_MEDIA_INIT = 13  # -1 until near-clipped camera containment is classified
_INT_MEDIA_EVENTS = 14  # independent of the surface-peel budget
_INT_MEDIA_SEGMENTS = 15  # free-flight draw ordinal, stable across hit batches
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
NEE_META_WIDTH = 23
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
# gate on one-sided emission and authored direct-light exclusion, so an
# ordinary emissive triangle takes neither branch
# and is bit-identical to what it was before area-light quads existed.
_NM_QUAD_BASE = 17
# The authored-appearance branch's sampled mode (roadmap section 6a-bis). Both
# words are read only inside that branch and only when the kernel was compiled
# with ``auth_sampled`` on, so a lit-only scene never loads them.
_NM_AUTHORED_SAMPLES = 18  # rows the authored branch draws per crossing ...
_NM_AUTHORED_COUNT = 19  # ... out of this many entries in the authored table
# 1 = the per-pixel sampler seed comes from the shipped blue-noise tile, which
# then occupies ``_NM_BN_BASE ...`` of this same vector (roadmap section 7).
# 0 leaves the vector NEE_META_WIDTH long and the key derivation exactly as it
# was.
_NM_BLUE_NOISE = 20
_NM_MEDIA = 21  # second-half offset in tri_shell, 0 = no volume transport
# Offset of the caustic-caster section in tri_shell (``_pt_caster_id``), 0 =
# no caster; read only by a kernel compiled with ``caustics`` on. Negative
# when only medium vertices receive through it: ``pt_caustics`` is off and
# the section is there to carry delta lights into a subsurface interior
# (``pt_caustics._prepare_casters``).
_NM_CAUSTIC = 22

# Blue-noise screen-space error distribution (roadmap section 7; Heitz et al.
# 2019). The tile is ``PT_BN_TILE x PT_BN_TILE`` per-pixel sampler keys --
# ``blue_noise.py`` loads the shipped permutation -- and rides ``nee_meta``'s
# tail rather than a new kernel argument or a new arena entry: it is per-render
# constant data of exactly the kind that vector already carries, and pt_shade
# is one of the arena-packed kernels where an argument is expensive. Integer
# tile values are exact in f32 (they are far below 2^24), so the f32 arena
# carries them losslessly.
#
# ``_PT_BN_SALT`` is what turns a tile value into a path seed, and it is a
# FIXED constant rather than ``pt_seed``: the map from tile value to sample
# sequence is what ``scripts/generate_blue_noise_tile.py`` optimises, so
# nothing per-render may enter it. The render seed and the frame enter as a
# toroidal SHIFT of the tile lookup instead (``_pt_bn_shift``), which is an
# isometry of the tile's own torus and therefore preserves the optimisation
# while still decorrelating seeds and (under ``pt_animated_seed``) frames. The
# cost of that spelling is that two (seed, frame) pairs landing on the same
# shift render identically -- one chance in ``PT_BN_TILE ** 2``.
PT_BN_TILE = 64
PT_BN_TILE_MASK = PT_BN_TILE - 1
PT_BN_TILE_VALUES = PT_BN_TILE * PT_BN_TILE
_PT_BN_SALT = 0x9E3779B1
#: First word of the tile inside ``nee_meta`` (the header ends there).
_NM_BN_BASE = NEE_META_WIDTH

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
def _pt_bn_shift(seed_root: ti.u32, f: ti.i32):
    """Where this ``(pt_seed, frame-or-0)`` reads the blue-noise tile.

    A toroidal translation, and nothing else, is what the render seed and the
    frame are allowed to do to the tile: it permutes which pixel gets which
    key without touching the key-to-sequence map the tile was optimised
    against (see ``_PT_BN_SALT``).
    """
    o = _pt_hash_combine(seed_root, ti.cast(f, ti.u32))
    return (
        ti.cast(o & ti.u32(PT_BN_TILE_MASK), ti.i32),
        ti.cast((o >> ti.u32(16)) & ti.u32(PT_BN_TILE_MASK), ti.i32),
    )


@ti.func
def _pt_bn_path_seed(nee_meta: ti.template(), seed_root: ti.u32, f: ti.i32,
                     px: ti.i32, py: ti.i32) -> ti.u32:
    """``_pt_path_seed`` for the blue-noise arm: the per-pixel half of every
    draw's seed comes from the shipped tile at ``(x, y)`` mod its edge, shifted
    by ``(pt_seed, frame-or-0)``.

    Purity is unchanged -- the draw is still a pure function of ``(pt_seed,
    frame-or-0, pixel, pair, sample index)`` plus fixed shipped data -- and so
    is stratification: one pixel still walks one Owen-scrambled Sobol
    sequence, just a different one. What changes is WHICH pixel gets which
    sequence, which is the whole method (roadmap section 7).
    """
    ox, oy = _pt_bn_shift(seed_root, f)
    tx = (px + ox) & PT_BN_TILE_MASK
    ty = (py + oy) & PT_BN_TILE_MASK
    v = ti.cast(nee_meta[_NM_BN_BASE + ty * PT_BN_TILE + tx] + 0.5, ti.u32)
    return _pt_hash_combine(ti.u32(_PT_BN_SALT), v)


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
    response in this renderer uses too. The deterministic stages'
    ``_smith_geometry`` is the same ``G2`` in three.js's closed form.
    """
    c2 = ti.math.clamp(cos_theta * cos_theta, 1e-8, 1.0)
    t2 = (1.0 - c2) / c2
    return 0.5 * (ti.sqrt(1.0 + alpha * alpha * t2) - 1.0)


#: Where the opaque GGX albedo table starts inside ``pt_glass_energy``
#: (``ggx_albedo.pt_energy_tables``: the glass table, then this one).
_GGX_TABLE_OFFSET = GLASS_ENERGY_SHAPE[0]


@ti.func
def _pt_ggx_albedo(table: ti.template(), n_dot_v, roughness):
    """``(A, B)`` of the path tracer's own GGX lobe (``ggx_albedo``): with
    Schlick Fresnel it reflects ``f0 * A + B`` of what arrives from ``n_dot_v``,
    and ``A + B`` is the white single-scatter albedo ``E1`` that the Turquin
    compensation divides by. Bilinear over roughness and ``sqrt(n.v)``.
    """
    r = ti.math.clamp(roughness, 0.0, 1.0) * (GGX_ROUGHNESS_SIZE - 1)
    c = ti.sqrt(ti.math.clamp(n_dot_v, 0.0, 1.0)) * (GGX_COSINE_SIZE - 1)
    ir = ti.min(ti.cast(r, ti.i32), GGX_ROUGHNESS_SIZE - 2)
    ic = ti.min(ti.cast(c, ti.i32), GGX_COSINE_SIZE - 2)
    fr, fc = r - ir, c - ic
    out = ti.math.vec2(0.0, 0.0)
    for dr, dc in ti.static(ti.ndrange(2, 2)):
        w = (fr if dr else 1.0 - fr) * (fc if dc else 1.0 - fc)
        row = _GGX_TABLE_OFFSET + (ir + dr) * GGX_COSINE_SIZE + ic + dc
        out += w * ti.math.vec2(table[row, 0], table[row, 1])
    return out


@ti.func
def _pt_ggx_compensation(table: ti.template(), f0, n_dot_v, roughness):
    """Turquin's multiple-scattering factor ``1 + f0 (1 - E1) / E1``.

    ``E1`` has to be the albedo of the lobe being compensated -- the exact
    Smith one this module samples -- or the factor creates energy: with
    Karis's split-sum fit standing in for it, a white rough metal reflected
    up to 27% more than arrived.
    """
    ab = _pt_ggx_albedo(table, n_dot_v, roughness)
    e1s = ti.math.clamp(ab[0] + ab[1], 1e-3, 1.0)
    return 1.0 + f0 * ((1.0 - e1s) / e1s)


@ti.func
def _pt_ggx_energy(table: ti.template(), f0, n_dot_v, roughness):
    """Single-scatter GGX directional albedo with Turquin multiple-scattering
    compensation folded in: ``E_ss(f0) * (1 + f0 (1 - E1)/E1)`` per channel,
    both from the tabulated albedo of the lobe ``_pt_lit_f_pdf`` evaluates.
    A white lobe (``f0 = 1``) therefore reflects exactly what arrives.
    """
    ab = _pt_ggx_albedo(table, n_dot_v, roughness)
    e_ss = f0 * ab[0] + ab[1]
    return ti.min(e_ss * _pt_ggx_compensation(table, f0, n_dot_v, roughness),
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
        # scale and the bounds subtend every direction: theta_u = pi
        # (PBRT-v4's ``EntireSphere``), which makes ``cos_p`` 1 below. It
        # used to be pi/2, which left a residual angle and scored ZERO for a
        # parent of spots whose cones do contain the point -- a delta light
        # with no MIS backstop, so its light was lost.
        d2 = ti.max(d * d, ti.max(radius * radius, 1e-12))
        cos_u = -1.0
        sin_u = 0.0
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
    nh2 = n_dot_h * n_dot_h
    # Avoid cancelling alpha^2 at the peak of a nearly smooth lobe.
    d = ti.max(1.0 - nh2, 0.0) + nh2 * a2
    return a2 / ti.max(_PI * d * d, 1e-30)


@ti.func
def _pt_fresnel(f0, cos_i, eta, metalness, albedo):
    """Schlick with Snell's thin-side angle and total internal reflection.

    ``eta`` is n_incident / n_transmitted; zero keeps the opaque-material
    model. Only the dielectric part changes sides, never the conductor tint.
    Both NEE evaluation and sampled GGX reflection use this function.
    """
    c = ti.math.clamp(cos_i, 0.0, 1.0)
    tail = ti.pow(1.0 - c, 5.0)
    fres = f0 + (1.0 - f0) * tail
    if eta > 0.0:
        m = ti.math.clamp(metalness, 0.0, 1.0)
        metal = m * (albedo + (1.0 - albedo) * tail)
        dielectric_f0 = f0 - m * albedo
        if ti.abs(eta - 1.0) < 1e-4:
            fres = metal
        elif eta > 1.0:
            sin2_t = eta * eta * ti.max(0.0, 1.0 - c * c)
            if sin2_t >= 1.0:
                fres = metal + (1.0 - m)
            else:
                thin_cos = ti.sqrt(ti.max(0.0, 1.0 - sin2_t))
                thin_tail = ti.pow(1.0 - thin_cos, 5.0)
                fres = metal + dielectric_f0 \
                    + ((1.0 - m) - dielectric_f0) * thin_tail
    return ti.math.clamp(fres, 0.0, 1.0)


@ti.func
def _pt_dielectric_fresnel(cos_i, eta):
    """Unpolarised Fresnel, with eta = n_incident / n_transmitted."""
    c = ti.math.clamp(cos_i, 0.0, 1.0)
    F = 0.0
    if ti.abs(eta - 1.0) >= 1e-4:
        sin2_t = eta * eta * ti.max(0.0, 1.0 - c * c)
        F = 1.0
        if sin2_t < 1.0:
            ct = ti.sqrt(ti.max(0.0, 1.0 - sin2_t))
            rs = (eta * c - ct) / ti.max(eta * c + ct, 1e-12)
            rp = (c - eta * ct) / ti.max(c + eta * ct, 1e-12)
            F = 0.5 * (rs * rs + rp * rp)
    return ti.math.clamp(F, 0.0, 1.0)


@ti.func
def _pt_glass_terms(f0, cos_i, eta, metalness, albedo, T):
    """Facet reflection/transmission colours and reflection probability.

    Default physical glass uses exact Fresnel. Authored specular F0/tint
    still controls normal incidence, and the conductor fraction keeps its
    Schlick tint. All three consumers use these same branch probabilities.
    """
    m = ti.math.clamp(metalness, 0.0, 1.0)
    F = _pt_dielectric_fresnel(cos_i, eta)
    ratio = (eta - 1.0) / ti.max(eta + 1.0, 1e-6)
    base = ratio * ratio
    tail = ti.math.clamp((F - base) / ti.max(1.0 - base, 1e-6), 0.0, 1.0)
    diel_f0 = f0 - m * albedo
    conductor = m * (albedo + (1.0 - albedo)
                    * ti.pow(1.0 - ti.math.clamp(cos_i, 0.0, 1.0), 5.0))
    R = ti.math.clamp(conductor + diel_f0
                      + ((1.0 - m) - diel_f0) * tail, 0.0, 1.0)
    trans = albedo * ((1.0 - m) * T * (1.0 - F))
    # ``R`` is the Schlick-remapped authored reflectance while ``trans`` uses
    # exact ``1 - F``, so an authored specular above the interface's own base
    # (``specular_intensity`` > 1, or a specular_color over white) makes
    # R + trans exceed one. Bound the facet budget before adding compensation
    # of its lossless neutral fraction: otherwise repeated internal crossings
    # could amplify energy. The clamp is exact at the defaults,
    # where R == F and trans <= 1 - F already holds.
    trans = ti.min(trans, ti.math.clamp(ti.math.vec3(1.0, 1.0, 1.0) - R,
                                        0.0, 1.0))
    wr = ti.max(R[0], ti.max(R[1], R[2]))
    wt = ti.max(trans[0], ti.max(trans[1], trans[2]))
    pr = wr / ti.max(wr + wt, 1e-12)
    return R, trans, pr


@ti.func
def _pt_glass_f_pdf(glass_energy: ti.template(), f0, rough, n, rd, wi,
                     eta, metalness, albedo, T):
    """Energy-compensated GGX dielectric f*cos and conditional solid-angle PDF.

    The neutral dielectric component closes its lost-power budget through a
    reciprocal two-port lobe, coupled by the two media's squared indices.
    Intentional absorption/tint is not normalised away. The density includes
    BOTH the original VNDF/facet strategy and the broad compensation strategy;
    the same mixture is sampled by _pt_sample_glass.

    PBRT-v4 section 9.7: both outcomes share one visible-normal distribution.
    The transmission Jacobian differs from reflection's and radiance carries
    (n_incident/n_transmitted)^2. Smooth interfaces are delta distributions.
    """
    fc = ti.math.vec3(0.0, 0.0, 0.0)
    pdf = 0.0
    nv = n.dot(-rd)
    ni = n.dot(wi)
    if (rough * rough >= 1e-4) and (nv > 1e-6) and (ti.abs(ni) > 1e-6):
        v = -rd
        h = v + wi
        if ni < 0.0:
            h = v + wi / eta
        h2 = h.dot(h)
        if h2 > 1e-20:
            h = h / ti.sqrt(h2)
            if n.dot(h) < 0.0:
                h = -h
            vh = v.dot(h)
            ih = wi.dot(h)
            if (vh > 1e-6) and (ih * ni > 0.0):
                ag = ti.max(rough * rough, 1e-4)
                D = _pt_ggx_ndf(ti.max(n.dot(h), 0.0), ag)
                lv = _pt_smith_lambda(nv, ag)
                li = _pt_smith_lambda(ti.abs(ni), ag)
                G1 = 1.0 / (1.0 + lv)
                G = 1.0 / (1.0 + lv + li)
                R, trans, pr = _pt_glass_terms(
                    f0, vh, eta, metalness, albedo, T)
                if ni > 0.0:
                    fc = R * (D * G / (4.0 * nv))
                    pdf = pr * D * G1 / (4.0 * nv)
                else:
                    denom = ih + eta * vh
                    denom2 = denom * denom
                    if denom2 > 1e-20:
                        jacobian = ti.abs(ih) / denom2
                        fc = trans * (D * G * vh * jacobian / nv) * (eta * eta)
                        pdf = (1.0 - pr) * D * G1 * vh / nv * jacobian
    if (rough * rough >= 1e-4) and (nv > 1e-6) and (ti.abs(ni) > 1e-6):
        coeff, p_ms, p_reflect = _pt_glass_ms_parameters(
            glass_energy, f0, rough, eta, metalness, albedo, T, nv)
        if p_ms > 0.0:
            destination_eta = eta
            side_probability = p_reflect
            if ni < 0.0:
                destination_eta = 1.0 / eta
                side_probability = 1.0 - p_reflect
            escape = _pt_glass_energy_lookup(glass_energy, rough,
                                               destination_eta, ti.abs(ni))
            fc += coeff * (escape[0] * ti.abs(ni))
            pdf = (1.0 - p_ms) * pdf \
                + p_ms * side_probability * (ti.abs(ni) * _INV_PI)
    return fc, pdf


@ti.func
def _pt_sample_glass_single(rd, n, rough, eta, f0, metalness, albedo, T,
                     u_dir, u_branch):
    """Sample the shared facet, then its Fresnel-weighted outcome.

    Returns direction, delta throughput (rough events use the full BSDF),
    transmitted, delta, valid. Rejected facets are zero-contribution samples;
    no resampling, splitting, or variable-size path state is introduced.
    """
    delta = 1 if rough * rough < 1e-4 else 0
    h = n
    if delta == 0:
        tb, bb = _pt_onb(n)
        wo = ti.math.vec3(tb.dot(-rd), bb.dot(-rd), ti.max(n.dot(-rd), 1e-6))
        hl = _pt_vndf_half_vector(wo, ti.max(rough * rough, 1e-4), u_dir)
        h = (tb * hl[0] + bb * hl[1] + n * hl[2]).normalized()
    ci = ti.math.clamp((-rd).dot(h), 0.0, 1.0)
    R, trans, pr = _pt_glass_terms(f0, ci, eta, metalness, albedo, T)
    wi = rd
    tint = ti.math.vec3(0.0, 0.0, 0.0)
    transmitted = 0
    valid = 1
    if u_branch < pr:
        wi = (rd + 2.0 * ci * h).normalized()
        tint = R / ti.max(pr, 1e-12)
        if n.dot(wi) <= 1e-6:
            valid = 0
    else:
        transmitted = 1
        sin2_t = eta * eta * ti.max(0.0, 1.0 - ci * ci)
        if (sin2_t >= 1.0) or (pr >= 1.0):
            valid = 0
        else:
            wi = (eta * rd + (eta * ci - ti.sqrt(1.0 - sin2_t)) * h).normalized()
            tint = trans * (eta * eta / ti.max(1.0 - pr, 1e-12))
            if n.dot(wi) >= -1e-6:
                valid = 0
    return wi, tint, transmitted, delta, valid


@ti.func
def _pt_sample_glass(glass_energy: ti.template(), rd, n, rough, eta, f0,
                     metalness, albedo, T, u_dir, u_branch):
    """VNDF/smooth glass mixed with the coupled compensation lobe.

    The existing independent branch scalar is partitioned and remapped for
    both choices. The direction pair is never reused as a branch variate.
    Smooth glass remains the original delta sampler, bit for bit.
    """
    _coeff, p_ms, p_reflect = _pt_glass_ms_parameters(
        glass_energy, f0, rough, eta, metalness, albedo, T, n.dot(-rd))
    wi = rd
    tint = ti.math.vec3(0.0, 0.0, 0.0)
    transmitted, delta, valid = 0, 0, 1
    if u_branch < p_ms:
        side = n
        if u_branch / p_ms >= p_reflect:
            side = -n
            transmitted = 1
        wi = _pt_cosine_direction(side, u_dir)
    else:
        branch = (u_branch - p_ms) / ti.max(1.0 - p_ms, 1e-12)
        wi, tint, transmitted, delta, valid = _pt_sample_glass_single(
            rd, n, rough, eta, f0, metalness, albedo, T, u_dir, branch)
    return wi, tint, transmitted, delta, valid


# MeshPhysicalMaterial's two layers over the base BSDF, as Three.js composes
# them (``meshphysical.glsl`` / ``RE_Direct_Physical``):
#
#   f = (1 - cc F_c(n.v)) * [s(v, l) * f_base + f_sheen] + cc * f_coat
#
# ``f_coat`` is a colourless GGX dielectric (F0 0.04, i.e. ior 1.5, F90 1)
# with its own roughness, ``f_sheen`` is Charlie x Neubelt on
# ``sheen_color * sheen``, and ``s(v, l) = 1 - max3(sheen) * max(E(n.v),
# E(n.l))`` is the sheen's albedo scaling of the base. ``_pt_lit_lobes``
# hands the per-crossing values on in ONE vector (slots ``_PL_*``) so that
# every NEE response, every MIS pdf and the continuation read the same layer.
# The coat and the sheen are lobes of their own in the continuation's pick,
# each with its own selection weight; a material with neither gets
# ``_pt_layer_none()``, whose scales are exactly 1 and weights exactly 0, so
# it evaluates and samples bit for bit as before.
#
# Emission sits UNDER the coat, so it leaves through it attenuated like the
# base: ``L_e * (1 - cc F_c(n.v))`` toward the viewer. Three.js scales
# ``totalEmissiveRadiance`` with the rest of ``outgoingLight``, and glTF's
# ``fresnel_coat`` mixes emission and base alike. Both MIS ends apply it: a
# path that hits the emitter reads this crossing's ``_PL_ATT``, and next-event
# estimation rebuilds the same factor at its sampled point
# (``_pt_emitter_coat_att``). Attenuating only one end would bias the result.
# The emitter tables still choose emitters by the raw emissive power. That
# power is an upper bound on what leaves through the coat, and any positive
# selection weight is unbiased when both ends read the same table.
_PL_CC = 0  # clearcoat weight, [0, 1]
_PL_CC_ROUGH = 1  # clearcoat roughness, [0, 1]
_PL_SHEEN_R = 2  # sheen roughness, [1e-4, 1] (as _stage_physical clamps it)
_PL_SHEEN_C = 3  # 3 slots: sheen_color * sheen (WebGLMaterials premultiplies)
_PL_SHEEN_MAX = 6  # max3 of the above
_PL_SHEEN_EV = 7  # E(n.v): Three.js's IBLSheenBRDF fit of the sheen albedo
_PL_ATT = 8  # 1 - cc * F_c(n.v): what the coat passes of base and emission
_PL_W_COAT = 9  # selection weights: the lobes' energies from
_PL_W_SHEEN = 10  # _pt_lit_lobes, times alpha once the caller scales them
_PL_WIDTH = 11


@ti.func
def _pt_layer_none():
    """The layer vector of a surface with neither clearcoat nor sheen."""
    layer = ti.Vector([0.0] * _PL_WIDTH)
    layer[_PL_SHEEN_R] = 1.0
    layer[_PL_ATT] = 1.0
    return layer


@ti.func
def _pt_coat_fresnel(cos_t):
    """Three.js's clearcoat Fresnel: Schlick with F0 0.04 and F90 1."""
    return 0.04 + 0.96 * ti.pow(1.0 - ti.math.clamp(cos_t, 0.0, 1.0), 5.0)


@ti.func
def _pt_physical_layer(params: ti.template(), tm, prim, n_dot_v):
    """Clearcoat and sheen of a MeshPhysicalMaterial crossing, read from the
    same slots (17-23) ``_stage_physical`` reads, with their selection
    energies at ``n_dot_v``. A block narrower than those slots has neither.
    """
    layer = _pt_layer_none()
    if params.shape[2] > 23:
        cc = ti.math.clamp(params[tm, prim, 17], 0.0, 1.0)
        sheen = params[tm, prim, 19]
        sheen_c = ti.max(ti.math.vec3(params[tm, prim, 21],
                                      params[tm, prim, 22],
                                      params[tm, prim, 23]) * sheen,
                         ti.math.vec3(0.0, 0.0, 0.0))
        sheen_max = ti.max(sheen_c[0], ti.max(sheen_c[1], sheen_c[2]))
        sheen_r = ti.math.clamp(params[tm, prim, 20], 1e-4, 1.0)
        layer[_PL_CC] = cc
        layer[_PL_CC_ROUGH] = ti.math.clamp(params[tm, prim, 18], 0.0, 1.0)
        layer[_PL_SHEEN_R] = sheen_r
        for k in ti.static(range(3)):
            layer[_PL_SHEEN_C + k] = sheen_c[k]
        layer[_PL_SHEEN_MAX] = sheen_max
        if cc > 0.0:
            # The coat's albedo is close to its Fresnel at n.v (exactly that
            # when smooth): what it takes from the base is what it reflects.
            fc_v = cc * _pt_coat_fresnel(n_dot_v)
            layer[_PL_ATT] = 1.0 - fc_v
            layer[_PL_W_COAT] = fc_v
        if sheen_max > 0.0:
            e_v = _ibl_sheen_brdf(ti.math.clamp(n_dot_v, 0.0, 1.0), sheen_r)
            layer[_PL_SHEEN_EV] = e_v
            layer[_PL_W_SHEEN] = sheen_max * e_v
    return layer


@ti.func
def _pt_emitter_coat_att(tri_mat: ti.template(), tri_mat_id: ti.template(),
                         tri_norm: ti.template(), tri_pos: ti.template(),
                         tri_uvs: ti.template(), tri_tex_meta: ti.template(),
                         textures: ti.template(), num_colored_triangles,
                         f, prim, w0, w1, w2, face_n, wi):
    """The share of emissive triangle ``prim``'s emission that its clearcoat
    lets out toward ``-wi``, at barycentrics ``(w0, w1, w2)``. This is the
    next-event end of the ``_PL_ATT`` that a hit on the emitter reads.

    A hit takes ``_PL_ATT`` from ``|shade_n . rd|``, and ``shade_n`` is
    ``_tri_normal_g`` passed through ``_prep_normal``. ``_sided_shading_normal``
    may also flip it toward the viewer, which cannot change the magnitude.
    This function rebuilds that normal at the sampled point and takes the same
    ``1 - cc * F_c`` of it, so both ends of the MIS pair weight one integrand.
    One difference remains for a normal-mapped emitter: a hit filters the map
    over its ray cone, while this end reads the finest level. The result is
    exactly 1 for any triangle that is not a coated MeshPhysicalMaterial,
    without reading its normals.
    """
    att = 1.0
    tm = f % tri_mat.shape[0]
    if (tri_mat.shape[2] > 23) \
            and (tri_mat_id[f % tri_mat_id.shape[0], prim] == _MID_PHYSICAL):
        cc = ti.math.clamp(tri_mat[tm, prim, 17], 0.0, 1.0)
        if cc > 0.0:
            n_s = _tri_normal_g(
                0, f, prim, w0, w1, w2, tri_norm, tri_pos, tri_uvs,
                tri_tex_meta, textures, num_colored_triangles)
            n_s = _prep_normal(n_s, face_n, tri_mat[tm, prim, 10], -wi)
            att = 1.0 - cc * _pt_coat_fresnel(
                ti.max(ti.abs(n_s.dot(wi)), 1e-4))
    return att


@ti.func
def _pt_base_scale(layer, cos_l):
    """What the coat and the sheen leave of the base BSDF toward a direction
    of cosine ``cos_l``: ``(att * s(v, l), s(v, l) / s(v, v), att * s(v, v))``.

    The first scales a base lobe evaluated from scratch (GGX); the second
    one whose energy ``_pt_lit_lobes`` already scaled by the third (the
    diffuse, whose ``e_diff`` is its whole response); the third is the
    direction-free part (glass, the selection energies, the ambient fill).
    All three are exactly 1 for ``_pt_layer_none()``.
    """
    att = layer[_PL_ATT]
    sheen_max = layer[_PL_SHEEN_MAX]
    s_l = 1.0
    s_v = 1.0
    if sheen_max > 0.0:
        e_v = layer[_PL_SHEEN_EV]
        e_l = _ibl_sheen_brdf(ti.math.clamp(cos_l, 0.0, 1.0),
                              layer[_PL_SHEEN_R])
        s_v = ti.max(1.0 - sheen_max * e_v, 0.0)
        s_l = ti.max(1.0 - sheen_max * ti.max(e_v, e_l), 0.0)
    return att * s_l, s_l / ti.max(s_v, 1e-6), att * s_v


@ti.func
def _pt_coat_f_pdf(layer, n, rd, wi):
    """The clearcoat lobe toward ``wi``: ``(f_cos, pdf)``, the coat's BSDF
    times cosine (scalar: it is colourless) and the density of its own VNDF
    sampler (``_pt_sample_layer``). ``n`` faces the ray; exact height-
    correlated Smith, like the base GGX here and Three.js's
    ``BRDF_GGX_Clearcoat``. No multiple-scattering compensation, as there.
    """
    f_cos = 0.0
    pdf = 0.0
    cc = layer[_PL_CC]
    cos_l = n.dot(wi)
    if (cc > 0.0) and (cos_l > 1e-6):
        v = -rd
        cos_v = ti.max(n.dot(v), 1e-4)
        h = (v + wi).normalized()
        n_dot_h = ti.math.clamp(n.dot(h), 0.0, 1.0)
        rc = layer[_PL_CC_ROUGH]
        a_c = ti.max(rc * rc, 1e-4)
        d = _pt_ggx_ndf(n_dot_h, a_c)
        lam_v = _pt_smith_lambda(cos_v, a_c)
        lam_l = _pt_smith_lambda(cos_l, a_c)
        q = d / ti.max(4.0 * cos_v, 1e-6)
        f_cos = cc * _pt_coat_fresnel(v.dot(h)) * q / (1.0 + lam_v + lam_l)
        pdf = q / (1.0 + lam_v)
    return f_cos, pdf


@ti.func
def _pt_sheen_f_pdf(layer, n, rd, wi):
    """The sheen lobe toward ``wi``: ``(f_cos, pdf)`` -- Charlie x Neubelt
    (``_stage_physical``'s lobe, term for term) times cosine, and the
    cosine-hemisphere density it is sampled with.
    """
    f_cos = ti.math.vec3(0.0, 0.0, 0.0)
    pdf = 0.0
    cos_l = n.dot(wi)
    if (layer[_PL_SHEEN_MAX] > 0.0) and (cos_l > 1e-6):
        v = -rd
        cos_v = ti.max(n.dot(v), 1e-4)
        n_dot_h = ti.math.clamp(n.dot((v + wi).normalized()), 0.0, 1.0)
        lobe = _d_charlie(n_dot_h, layer[_PL_SHEEN_R]) \
            * _v_neubelt(cos_v, cos_l) * cos_l
        f_cos = ti.math.vec3(layer[_PL_SHEEN_C + 0], layer[_PL_SHEEN_C + 1],
                             layer[_PL_SHEEN_C + 2]) * lobe
        pdf = cos_l * _INV_PI
    return f_cos, pdf


@ti.func
def _pt_sample_layer(layer, n, rd, u, coat):
    """Draw the clearcoat (``coat == 1``: VNDF about ``n``, which faces the
    ray) or the sheen lobe (cosine-weighted). Returns ``(wi, weight, ok)``
    with ``weight`` that lobe's own ``f_cos / pdf`` -- exactly what
    ``_pt_coat_f_pdf`` / ``_pt_sheen_f_pdf`` give for ``wi``.
    """
    wi = rd
    weight = ti.math.vec3(0.0, 0.0, 0.0)
    ok = 0
    if coat == 1:
        t_b, b_b = _pt_onb(n)
        wo_l = ti.math.vec3(t_b.dot(-rd), b_b.dot(-rd),
                            ti.max(n.dot(-rd), 1e-4))
        rc = layer[_PL_CC_ROUGH]
        a_c = ti.max(rc * rc, 1e-4)
        h_l = _pt_vndf_half_vector(wo_l, a_c, u)
        h_w = (t_b * h_l[0] + b_b * h_l[1] + n * h_l[2]).normalized()
        wi = (rd - 2.0 * rd.dot(h_w) * h_w).normalized()
        cos_l = n.dot(wi)
        if cos_l > 1e-6:
            lam_v = _pt_smith_lambda(wo_l[2], a_c)
            lam_l = _pt_smith_lambda(cos_l, a_c)
            w = layer[_PL_CC] * _pt_coat_fresnel((-rd).dot(h_w)) \
                * (1.0 + lam_v) / (1.0 + lam_v + lam_l)
            weight = ti.math.vec3(w, w, w)
            ok = 1
    else:
        wi = _pt_cosine_direction(n, u)
        fc, pdf = _pt_sheen_f_pdf(layer, n, rd, wi)
        if pdf > 0.0:
            weight = fc / pdf
            ok = 1
    return wi, weight, ok


@ti.func
def _pt_lit_lobes(energy: ti.template(), pid, params: ti.template(), f, prim,
                  albedo3, metalness, rough, ior, T, shade_n, rd, eta):
    """Continuation-lobe energies of a physically-integrated (lit) hit:
    ``(e_diff, e_spec, e_trans, f0, rough_eff, layer)``. The single source
    for the sampled continuation AND for every direct-lighting response
    (roadmap section 5) -- one BSDF, so MIS weights sum to one and a light
    row and an emissive triangle of matched radiance light a surface
    identically.

    ``rough_eff`` is the GGX roughness the caller must carry downstream: it
    is the crossing's own ``rough`` for every pipeline except phong, whose
    highlight is authored as a Blinn-Phong exponent instead. ``layer`` is a
    physical material's clearcoat and sheen (see ``_PL_CC``), and
    ``_pt_layer_none()`` for everything else; the three base energies come
    back already scaled by what those layers leave of the base at ``n.v``.
    """
    one3 = ti.math.vec3(1.0, 1.0, 1.0)
    e_diff = ti.math.vec3(0.0, 0.0, 0.0)
    e_spec = ti.math.vec3(0.0, 0.0, 0.0)
    e_trans = ti.math.vec3(0.0, 0.0, 0.0)
    f0 = ti.math.vec3(0.0, 0.0, 0.0)
    rough_eff = rough
    layer = _pt_layer_none()
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
        e_spec = _pt_ggx_energy(energy, f0, n_dot_v, rough_eff)
        e_diff = albedo3 * (one3 - e_spec)
    else:
        tm = f % params.shape[0]
        met = ti.math.clamp(ti.max(metalness, 0.0), 0.0, 1.0)
        diel_f0 = ti.math.vec3(0.04, 0.04, 0.04)
        if eta > 0.0:
            ratio = (eta - 1.0) / ti.max(eta + 1.0, 1e-6)
            diel_f0 = one3 * (ratio * ratio)
        if pid == _MID_PHYSICAL:
            ior_m = params[tm, prim, 12]
            ratio = (ior_m - 1.0) / ti.max(ior_m + 1.0, 1e-4)
            if eta > 0.0:
                ratio = (eta - 1.0) / ti.max(eta + 1.0, 1e-6)
            diel_f0 = ti.math.vec3(
                params[tm, prim, 14], params[tm, prim, 15],
                params[tm, prim, 16]) \
                * (ratio * ratio * params[tm, prim, 13])
        f0 = diel_f0 * (1.0 - met) + albedo3 * met
        e_spec = _pt_ggx_energy(energy, f0, n_dot_v, rough)
        diel_pass = 0.0
        if eta <= 0.0:
            # Only this arm consumes it: the eta > 0 arm below overwrites
            # ``diel_pass`` outright, and ``_material_reflectance`` is the
            # exact-Fresnel routine with its own Snell/TIR branch -- the most
            # expensive call in this innermost per-crossing loop.
            _R3, diel_pass = _material_reflectance(
                rd, shade_n, ti.max(metalness, 0.0), ior, albedo3, T)
        if eta > 0.0:
            # The same enclosing medium that bends the ray supplies the
            # Fresnel interface. Importance weights may be approximate, but
            # must offer the reflected branch when transmission closes.
            ratio = (eta - 1.0) / ti.max(eta + 1.0, 1e-6)
            interface_f0 = ratio * ratio
            diel = _pt_fresnel(one3 * interface_f0, n_dot_v, eta, 0.0, one3)
            diel_pass = (1.0 - met) * (1.0 - diel[0])
            e_spec = ti.max(e_spec, _pt_fresnel(f0, n_dot_v, eta, met, albedo3))
            if ti.abs(eta - 1.0) < 1e-4:
                e_spec = met * _pt_ggx_energy(energy, albedo3, n_dot_v, rough)
        e_trans = albedo3 * (diel_pass * T)
        e_diff = albedo3 * ((1.0 - met) * (1.0 - T)) * (one3 - e_spec)
        if pid == _MID_PHYSICAL:
            layer = _pt_physical_layer(params, tm, prim, n_dot_v)
            _bl, _ratio, base_v = _pt_base_scale(layer, n_dot_v)
            e_diff *= base_v
            e_spec *= base_v
            e_trans *= base_v
    return e_diff, e_spec, e_trans, f0, rough_eff, layer


@ti.func
def _pt_lit_f_pdf(glass_energy: ti.template(), e_diff, e_spec, f0, rough,
                  shade_n, rd, wi,
                  w_pass, w_diff, w_spec, w_trans, eta, metalness, albedo, T,
                  layer):
    """Physical BSDF response of a lit vertex toward ``wi`` and the pdf with
    which its continuation sampler generates ``wi``.

    Returns ``(f_cos, pdf)``: BSDF times absolute surface cosine and the
    continuation's mixture density. Rough glass includes reflection and
    transmission from one GGX distribution; opaque GGX retains Turquin
    compensation. Smooth glass and straight pass-through are deltas and
    add no solid-angle density. A physical material's clearcoat and sheen
    (``layer``, from ``_pt_lit_lobes``) attenuate that base and add two
    lobes of their own, with selection weights ``layer[_PL_W_COAT]`` and
    ``layer[_PL_W_SHEEN]``. Both ends of every MIS pair -- next-event
    samples toward emitters and BSDF paths that find them -- use this one
    function, which is what makes the power-heuristic weights sum to one.
    """
    f_cos = ti.math.vec3(0.0, 0.0, 0.0)
    pdf = 0.0
    w_coat = layer[_PL_W_COAT]
    w_sheen = layer[_PL_W_SHEEN]
    w_sum = w_pass + w_diff + w_spec + w_trans + w_coat + w_sheen
    cos_i = shade_n.dot(wi)
    spec_n = shade_n
    if spec_n.dot(rd) > 0.0:
        spec_n = -spec_n
    cos_s = spec_n.dot(wi)
    base_l, base_ratio, base_v = _pt_base_scale(layer, cos_s)
    # The diffuse lobe REFLECTS, so it needs ``wi`` above the horizon of the
    # declared side (``cos_i``) AND on the same side as the outgoing direction
    # (``cos_s``). The two differ only on a one-sided surface hit from behind,
    # where ``shade_n`` keeps its outward normal -- the shapes_3d solids
    # default to ``two_sided = False``, so that is the ordinary case, not a
    # corner. Testing ``cos_i`` alone reported a diffuse response for light
    # entering the front and leaving the back of an opaque surface; the NEE
    # shadow gate refuses exactly those directions, so the two ends of the MIS
    # pair disagreed and their power-heuristic weights stopped summing to one.
    # Transmitting interfaces are unaffected: the caller sets
    # ``shade_n = spec_n`` for glass, making the second test an identity.
    if (cos_i > 1e-6) and (cos_s > 1e-6) and (w_sum > 1e-6):
        f_cos = e_diff * (_INV_PI * cos_i * base_ratio)
        pdf = (w_diff / w_sum) * (_INV_PI * cos_i)
    glass = (eta > 0.0) and (ti.abs(eta - 1.0) >= 1e-4)
    if glass and (w_sum > 1e-6):
        fg, pg = _pt_glass_f_pdf(glass_energy, f0, rough, spec_n, rd, wi,
                                eta, metalness, albedo, T)
        f_cos += fg * base_v
        pdf += ((w_spec + w_trans) / w_sum) * pg
    elif (cos_s > 1e-6) and (w_sum > 1e-6):
        if w_spec > 0.0:
            v = -rd
            h = (v + wi).normalized()
            n_dot_v = ti.max(spec_n.dot(v), 1e-4)
            n_dot_h = ti.math.clamp(spec_n.dot(h), 0.0, 1.0)
            v_dot_h = ti.max(v.dot(h), 1e-4)
            a_g = ti.max(rough * rough, 1e-4)
            d = _pt_ggx_ndf(n_dot_h, a_g)
            lam_v = _pt_smith_lambda(n_dot_v, a_g)
            lam_l = _pt_smith_lambda(cos_s, a_g)
            g1_v = 1.0 / (1.0 + lam_v)
            g2 = 1.0 / (1.0 + lam_v + lam_l)
            g2 *= base_l  # what the coat and sheen leave of this lobe
            fres = _pt_fresnel(f0, v_dot_h, eta, metalness, albedo)
            comp = _pt_ggx_compensation(glass_energy, f0, n_dot_v, rough)
            f_cos += fres * comp * (d * g2 / ti.max(4.0 * n_dot_v, 1e-6))
            pdf += (w_spec / w_sum) * (g1_v * d / ti.max(4.0 * n_dot_v, 1e-6))
    # The clearcoat and sheen reflect on the ray's side, glass included.
    if (cos_s > 1e-6) and (w_sum > 1e-6) and (w_coat + w_sheen > 0.0):
        fc_c, pc_c = _pt_coat_f_pdf(layer, spec_n, rd, wi)
        fc_s, pc_s = _pt_sheen_f_pdf(layer, spec_n, rd, wi)
        f_cos += fc_s + ti.math.vec3(fc_c, fc_c, fc_c)
        pdf += (w_coat * pc_c + w_sheen * pc_s) / w_sum
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
                # three.js r155+'s floor: a light touching a surface boosts
                # it at most 100x instead of exploding to inf.
                lc = lc / ti.max(ti.pow(d, decay), 0.01)
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
def pt_screen_sampler_probe(nee_meta: ti.types.ndarray(), seed_root: ti.u32,
                            f: ti.i32, blue_noise: ti.i32, width: ti.i32,
                            pair: ti.i32, sample_index: ti.i32,
                            out: ti.types.ndarray()):
    """Test probe: one draw per pixel of an ``out [h, w, 2]`` screen grid.

    The sampler tests probe ONE pixel's sequence; the blue-noise tile is about
    how sequences are laid out ACROSS pixels (roadmap section 7), which needs a
    whole grid. ``blue_noise`` picks the arm, and both branches are written the
    way ``pt_generate`` and ``pt_shade`` write them -- with the probe's own
    ``width``, so a pixel's flat index in the off arm is the one a render of
    that width would give it. Keep the two branches in step with those kernels:
    the off branch's job is to be exactly what they compute.
    """
    for py, px in ti.ndrange(out.shape[0], out.shape[1]):
        path_seed = ti.u32(0)
        if blue_noise != 0:
            path_seed = _pt_bn_path_seed(nee_meta, seed_root, f, px, py)
        else:
            path_seed = _pt_path_seed(seed_root, _pt_key(f, py * width + px))
        u = pt_sample_2d_seeded(path_seed, pair, sample_index)
        out[py, px, 0] = u[0]
        out[py, px, 1] = u[1]


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


@ti.func
def _pt_concentric_disk(u):
    """Map a unit-square sample onto the unit disk (Shirley & Chiu 1997).

    Area-preserving and continuous, so the stratification of the Sobol pair
    that feeds it survives onto the aperture; the polar map ``r = sqrt(u)``
    would tear the square's strata apart at the seam.
    """
    a = 2.0 * u[0] - 1.0
    b = 2.0 * u[1] - 1.0
    r = 0.0
    phi = 0.0
    if a * a > b * b:
        r = a
        phi = 0.7853981633974483 * (b / a)
    elif b != 0.0:
        r = b
        phi = 1.5707963267948966 - 0.7853981633974483 * (a / b)
    return ti.math.vec2(r * ti.cos(phi), r * ti.sin(phi))


@ti.kernel
def pt_generate(num_slots: ti.i32, tile_pixels: ti.i32, sample_base: ti.i32,
                seed_root: ti.u32, animated_seed: ti.i32, time_start: ti.i32,
                width: ti.i32, height: ti.i32,
                half_screen_w: ti.f32, half_screen_h: ti.f32,
                cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
                pixel_basis_x: ti.types.ndarray(),
                pixel_basis_y: ti.types.ndarray(),
                cam_proj_table: ti.types.ndarray(), projected: ti.template(),
                near_clip: ti.f32,
                pix_list: ti.types.ndarray(),
                rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
                rs_sca: ti.types.ndarray(), rs_pix: ti.types.ndarray(),
                nee_meta: ti.types.ndarray(), lens: ti.types.ndarray(),
                pt_acc: ti.types.ndarray()):
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

    ``nee_meta`` is the shade kernel's own metadata vector, passed here for
    the one word (and the one table) both ends of the sampler have to agree
    on: ``_NM_BLUE_NOISE`` and, behind it, the blue-noise tile at
    ``_NM_BN_BASE``. An ordinary ndarray parameter rather than an arena
    binding because this kernel is not arena-packed (it takes a dozen arrays,
    not forty).

    ``lens`` is the thin lens, ``[T_lens, 2]`` rows of (aperture RADIUS,
    focus distance) in world units, indexed ``f % T_lens`` so one row serves a
    static camera and the host's ``[1, 2]`` zero row serves a pinhole. Where a
    frame's radius is positive the ray starts at a point of the aperture disk
    (sampler pair ``PAIR_LENS``, Shirley-Chiu concentric mapping) and is aimed
    at the point its pinhole ray meets the focus plane -- the plane
    ``focus`` in front of the camera along its forward axis, parallel to the
    screen, the same planar depth the near clip uses. The aperture disk lies
    in that plane's parallel through the camera, so the near-clip advance
    below stays exact for a lens origin. Such a path is flagged stochastic in
    ``pt_acc`` (see ``_PT_ACC_STOCH``), which the host zeroes before this
    launch. A zero radius leaves the pinhole arithmetic untouched, bit for
    bit.

    ``cam_proj_table`` is the per-frame projection, read only when
    ``projected`` (``_generate_ray``, ``CameraProjection``). Under an
    orthographic camera every pixel's ray is parallel to the forward axis, so
    the same construction is a telecentric lens: each pixel's own aperture
    disk around its ray origin, aimed at the point straight ahead on the
    focus plane, and a point ``z`` deep blurs over ``aperture * |z - focus|
    / focus`` world units -- in proportion to the defocus alone, with no
    perspective shrink.
    """
    cam_proj = ti.static(CameraProjection(cam_proj_table, projected))
    pixels_per_frame = width * height
    blue_noise = nee_meta[_NM_BLUE_NOISE] > 0.5
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
        # same way ``pt_shade`` hoists it (roadmap section 0.2), and derived
        # the same way it derives it (roadmap section 7).
        path_seed = ti.u32(0)
        if blue_noise:
            path_seed = _pt_bn_path_seed(nee_meta, seed_root,
                                         f * animated_seed, px, py)
        else:
            path_seed = _pt_path_seed(seed_root, _pt_key(f * animated_seed, p))
        jitter = pt_sample_2d_seeded(path_seed, PAIR_PIXEL, s)
        ro, rd = _generate_ray(f, px, py, jitter[0], jitter[1],
                               half_screen_w, half_screen_h,
                               cam_origin, screen_point,
                               pixel_basis_x, pixel_basis_y, cam_proj)
        tl = f % lens.shape[0]
        lens_r = lens[tl, 0]
        if lens_r > 0.0:
            # Thin lens. The lens basis is the camera's own: forward from the
            # screen point (as the near clip builds it), right from the
            # pixel basis, which is screen_half_height * right; the
            # Gram-Schmidt step only guards a hand-built screen. The new
            # direction is formed relative to the ray origin -- the camera
            # ray scaled to the focus plane, minus the lens offset -- rather
            # than from an absolute focus point, which would cost f32 digits
            # under a camera thousands of units from the origin.
            fwd_l = (ti.math.vec3(screen_point[f, 0], screen_point[f, 1],
                                  screen_point[f, 2]) - ro).normalized()
            if ti.static(_projected(cam_proj)):
                # Forward from the eye, as the near clip builds it: an
                # orthographic ray starts off the axis.
                fwd_l = (ti.math.vec3(screen_point[f, 0], screen_point[f, 1],
                                      screen_point[f, 2])
                         - ti.math.vec3(cam_origin[f, 0], cam_origin[f, 1],
                                        cam_origin[f, 2])).normalized()
            bx = ti.math.vec3(pixel_basis_x[f, 0], pixel_basis_x[f, 1],
                              pixel_basis_x[f, 2])
            ex = (bx - fwd_l * bx.dot(fwd_l)).normalized()
            ey = fwd_l.cross(ex)
            disk = _pt_concentric_disk(
                pt_sample_2d_seeded(path_seed, PAIR_LENS, s)) * lens_r
            off = ex * disk[0] + ey * disk[1]
            t_focus = lens[tl, 1] / ti.max(rd.dot(fwd_l), 1e-6)
            rd = (rd * t_focus - off).normalized()
            ro = ro + off
            pt_acc[slot, _PT_ACC_STOCH] = 1.0
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
def _pt_authored_surface(media_enabled, pipelines: ti.template(), pids: ti.template(),
                          prim, f, pos, view, normal, face, albedo, glow,
                          light_pos: ti.template(), light_col: ti.template(),
                          num_lights, pid_arr: ti.template(), params: ti.template(),
                          shadows: ti.template(), vis, camera):
    # A medium's extinction is physical transport even with surface shadows
    # disabled. Keep the original flag in ordinary scenes, and call authored
    # stages exactly once (custom stages need not be safe to evaluate twice).
    result = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    if ti.static(shadows == 0):
        if media_enabled:
            result = _run_frag_pipeline(
                pipelines, pids, prim, f, pos, view, normal, face, albedo, glow,
                light_pos, light_col, num_lights, pid_arr, params, 1, vis, camera)
        else:
            result = _run_frag_pipeline(
                pipelines, pids, prim, f, pos, view, normal, face, albedo, glow,
                light_pos, light_col, num_lights, pid_arr, params, 0, vis, camera)
    else:
        result = _run_frag_pipeline(
            pipelines, pids, prim, f, pos, view, normal, face, albedo, glow,
            light_pos, light_col, num_lights, pid_arr, params, shadows, vis, camera)
    return result


@ti.func
def _pt_vertex_f_pdf(medium_vertex, phase_g, glass_energy: ti.template(),
                     e_diff, e_spec, f0, rough,
                     shade_n, rd, wi, w_pass, w_diff, w_spec, w_trans,
                     eta, metalness, albedo, transmission, layer):
    f_cos = ti.math.vec3(0.0, 0.0, 0.0)
    pdf = 0.0
    if medium_vertex:
        pdf = _pt_hg_pdf(rd.dot(wi), phase_g)
        f_cos = ti.math.vec3(pdf, pdf, pdf)
    else:
        f_cos, pdf = _pt_lit_f_pdf(
            glass_energy, e_diff, e_spec, f0, rough, shade_n, rd, wi,
            w_pass, w_diff, w_spec, w_trans, eta, metalness, albedo, transmission,
            layer)
    return f_cos, pdf


@ti.func
def _pt_nee_visibility(refit: ti.template(), anyhit: ti.template(),
                       has_tri: ti.template(),
                       has_bez: ti.template(),
                       sorigin, wi, ldist, f, ff,
                       pixel_size_per_t, base_dist,
                       pixel_size_base: ti.template(),
                       layer_offset_triangles,
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
        pixel_size_per_t, base_dist, pixel_size_base, layer_offset_triangles,
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
    tree_mix, num_inf, quad_base, blue_noise)``.
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
        nee_meta[_NM_BLUE_NOISE] > 0.5,
    )


@ti.func
def _pt_caustic_chain(refit: ti.template(), shadow_mode: ti.template(),
                      has_tri: ti.template(), has_bez: ti.template(),
                      x0, sorigin, n_recv, wi_vis, ldist, y, fixed, f, ff,
                      pixel_size_per_t, base_dist,
                      pixel_size_base: ti.template(), layer_offset_triangles,
                      t_nodes: ti.template(), t_node_miss: ti.template(),
                      t_leaf_prim: ti.template(), t_leaf_tspan: ti.template(),
                      t_first_leaf, tri_pos: ti.template(),
                      tri_norm: ti.template(), tri_colors: ti.template(),
                      tri_uvs: ti.template(), tri_tex_meta: ti.template(),
                      textures: ti.template(), tri_extra: ti.template(),
                      tri_mat: ti.template(), tri_mat_id: ti.template(),
                      num_colored_triangles,
                      b_nodes: ti.template(), b_node_miss: ti.template(),
                      b_leaf_prim: ti.template(), b_leaf_tspan: ti.template(),
                      b_first_leaf, circuit_meta: ti.template(),
                      circuit_colors: ti.template(),
                      circuit_border_colors: ti.template(),
                      edges_2d: ti.template(), edge_accel: ti.template(),
                      tri_shell: ti.template(), caster_offset, media,
                      media_offset, glass_energy: ti.template(),
                      vol_i: ti.template(), vol_f: ti.template(),
                      vol_seed: ti.u32):
    """One light sample's caustic connection (roadmap section 1).

    ``x0`` is the receiver, ``sorigin`` the shadow ray's spawn point off it
    and ``n_recv`` its geometric normal on the light's side (zero at a medium
    vertex); ``wi_vis`` / ``ldist`` are the straight shadow ray the
    next-event sample traces, and ``y`` its light end -- the sampled point,
    or with ``fixed`` the unit direction toward a directional light or an
    environment sample. The straight segment is probed for caster crossings
    (``_pt_mnee_probe``), which seed a manifold walk (``_pt_mnee_solve``) to
    the bent chain that reaches the same light end by refracting at each of
    them. Returns ``(straight, ok, w0, g, tau, nr2, e, d_path, overflow,
    capped, media_over, media_cut, vol_over, count)``:

    * ``straight`` -- what the STRAIGHT connection keeps through the casters
      (their uncovered fraction), the factor the caller applies to its own
      contribution: a refracting caster bends the light it covers instead of
      tinting it through. It stays 1 when the segment crosses more casters
      than a chain may hold (``overflow``), which keeps the march's old
      tinted reading there rather than dropping the light.
    * ``ok`` -- a chain was found. Then ``w0`` is the receiver's direction
      toward the first vertex (the receiver's BSDF is evaluated toward it),
      ``g`` the generalized geometry term (``_pt_mnee_geometry``), ``tau``
      the chain's RGB throughput, ``nr2`` the radiance change
      ``(n_receiver / n_light)^2`` across it, ``e`` the unit direction the
      light emits into (from ``y`` toward the last vertex) and ``d_path``
      the chain's unfolded length. ``tau`` is each vertex's smooth-glass
      transmission at the camera path's own Fresnel and coverage -- the
      expectation of its delta-transmission throughput without the ``eta^2``
      that ``nr2`` carries -- times every leg's visibility (the ordinary
      shadow query: what is not a caster still blocks and tints), an
      interior leg's Beer-Lambert absorption and, with media on, each leg's
      medium transmittance.
    * ``capped`` -- the walk gave up at ``MNEE_MAX_ITERATIONS`` still
      walking; ``media_over`` / ``media_cut`` -- a leg's medium stack
      overflowed or its walk was truncated (``PT_STAT_MEDIA_*``). Each of
      these leaves ``ok`` at 0. ``vol_over`` counts leg pieces whose
      density-grid tracking exhausted its step budget and was absorbed
      (``PT_STAT_VOLUME_STEPS``); each leg is ratio-tracked from
      ``vol_seed`` and its ordinal.
    * ``count`` -- the caster interfaces the straight segment crosses: the
      chain's vertices, when there is one.
    """
    straight = ti.math.vec3(1.0, 1.0, 1.0)
    ok = 0
    capped = 0
    media_over = 0
    media_cut = 0
    vol_over = 0
    w0 = ti.math.vec3(0.0, 0.0, 0.0)
    g = 0.0
    tau = ti.math.vec3(0.0, 0.0, 0.0)
    nr2 = 1.0
    e = ti.math.vec3(0.0, 0.0, 0.0)
    d_path = 0.0
    count, prims, bary_a, bary_b, straight_p, overflow = _pt_mnee_probe(
        refit, sorigin, wi_vis,
        _pt_shadow_tmax(sorigin, wi_vis, ldist - (sorigin - x0).dot(wi_vis)),
        f, layer_offset_triangles, t_nodes, t_node_miss, t_leaf_prim,
        t_leaf_tspan, t_first_leaf, tri_pos, tri_colors, tri_uvs,
        tri_tex_meta, textures, tri_extra, circuit_meta,
        num_colored_triangles, tri_shell, caster_offset)
    if overflow == 0:
        straight = straight_p
    if (overflow == 0) and (count > 0):
        cids = ti.Vector([-1] * MNEE_MAX_VERTICES)
        for q in ti.static(range(MNEE_MAX_VERTICES)):
            if q < count:
                cids[q] = _pt_caster_id(tri_shell, f, prims[q], caster_offset)
        solved, capped, prims, bary_a, bary_b, ps, ngs, etas, a_blks, \
            b_blks, c_blks, _c_floor = _pt_mnee_solve(
                refit, x0, n_recv, y, fixed, count, prims, bary_a, bary_b,
                cids, f, layer_offset_triangles, t_nodes, t_node_miss,
                t_leaf_prim, t_leaf_tspan, t_first_leaf, tri_pos, tri_norm,
                tri_mat, tri_extra, tri_uvs, tri_tex_meta, textures,
                num_colored_triangles, tri_shell, caster_offset)
        if solved == 1:
            g, ok = _pt_mnee_geometry(x0, count, ps, ngs, a_blks, b_blks,
                                      c_blks)
        if ok == 1:
            w0 = (ti.math.vec3(ps[0, 0], ps[0, 1], ps[0, 2]) - x0).normalized()
            # The decisions the coverage test repeats (``pt_caustic_cover``).
            ok = _pt_mnee_admissible(
                refit, x0, sorigin, n_recv, y, fixed, count, prims, bary_a,
                bary_b, ps, ngs, f, layer_offset_triangles, t_nodes,
                t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf, tri_pos,
                tri_norm, tri_mat, tri_shell, caster_offset)
        # Vertices: the smooth-glass transmission the camera path's own
        # delta lobe would carry across each one, at the same Fresnel,
        # authored tint and coverage, without its ``eta^2`` (``nr2`` below).
        # One dynamic loop with a static extraction keeps a single inlined
        # copy of the material evaluation.
        tau = ti.math.vec3(1.0, 1.0, 1.0)
        eta_prod = 1.0
        if ok == 1:
            for q in range(count):
                prim_q, a_q, b_q, rd_q, spec_q, ci_q, _turn_ok = \
                    _pt_mnee_turn(q, count, x0, y, fixed, prims, bary_a,
                                  bary_b, ps, f, tri_mat, tri_pos, tri_norm)
                eta_q = 1.0
                for qq in ti.static(range(MNEE_MAX_VERTICES)):
                    if qq == q:
                        eta_q = etas[qq]
                w_q = 1.0 - a_q - b_q
                color_q, alpha_q = _tri_color_g(
                    0, f, prim_q, w_q, a_q, b_q, tri_colors, tri_colors,
                    tri_uvs, tri_tex_meta, textures, num_colored_triangles)
                met_q, rough_q = _tri_extra_g(
                    0, f, prim_q, w_q, a_q, b_q, tri_extra, tri_colors,
                    tri_uvs, tri_tex_meta, textures, num_colored_triangles)
                ior_q, t_q = _tri_ior_transmission_g(
                    0, f, prim_q, w_q, a_q, b_q, tri_extra, tri_colors,
                    tri_uvs, tri_tex_meta, textures, num_colored_triangles)
                t_q = ti.math.clamp(t_q, 0.0, 1.0)
                alb_q = ti.math.vec3(color_q[0], color_q[1], color_q[2])
                # The chain's index ratio is the walk's (light side over
                # receiver side); the camera path's ``eta``, incident over
                # transmitted along its own direction, is its inverse.
                eta_cam = 1.0 / ti.max(eta_q, 1e-6)
                _ed_q, _es_q, _et_q, f0_q, _r_q, layer_q = _pt_lit_lobes(
                    glass_energy, tri_mat_id[f % tri_mat_id.shape[0], prim_q],
                    tri_mat, f, prim_q, alb_q, met_q, rough_q, ior_q, t_q,
                    spec_q, rd_q, eta_cam)
                _rf_q, trans_q, _pr_q = _pt_glass_terms(
                    f0_q, ti.math.clamp(ci_q, 0.0, 1.0), eta_cam, met_q, alb_q,
                    t_q)
                _bl_q, _br_q, bv_q = _pt_base_scale(layer_q, 1.0)
                tau *= trans_q * (ti.math.clamp(alpha_q, 0.0, 1.0) * bv_q)
                eta_prod *= eta_q
        # Legs: x0 -> x_1 -> ... -> x_k -> y, each a straight shadow segment
        # that crosses no caster but at its own ends (the admissibility test
        # above) and is otherwise the ordinary shadow query's business.
        if ok == 1:
            m_stack = media
            for q in range(count + 1):
                origin, ray_d, reach, leg_d, leg_len, inner, b_prim, b_ng, \
                    b_fixed = _pt_mnee_leg(q, count, x0, sorigin, n_recv, y,
                                           fixed, prims, ps, ngs)
                if b_fixed == 0:
                    d_path += leg_len
                if ok == 1:
                    tau *= _pt_nee_visibility(
                        refit, shadow_mode, has_tri, has_bez, origin, ray_d,
                        reach, f, ff, pixel_size_per_t, base_dist,
                        pixel_size_base, layer_offset_triangles,
                        t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                        t_first_leaf, tri_pos, tri_colors, tri_uvs,
                        tri_tex_meta, textures, tri_extra,
                        num_colored_triangles,
                        b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                        b_first_leaf, circuit_meta, circuit_colors,
                        circuit_border_colors, edges_2d, edge_accel)
                    # Inside a caster (never toward a light at infinity,
                    # which is not admissible): the camera path's
                    # Beer-Lambert, unless the medium walk below integrates
                    # the solid's interior -- a scattering one, or one a
                    # density grid varies (``_pt_medium_tracked``).
                    if inner >= 0:
                        tracked = 0
                        if media_offset > 0:
                            if _pt_medium_id(tri_shell, f, inner,
                                             media_offset) >= 0:
                                tracked = _pt_medium_tracked(
                                    vol_i, tri_mat, f, inner)
                        if tracked == 0:
                            tma = f % tri_mat.shape[0]
                            for k in ti.static(range(3)):
                                tau[k] *= ti.exp(
                                    -tri_mat[tma, inner,
                                             _MAT_ATTENUATION_SIGMA + k]
                                    * leg_len)
                    if media_offset > 0:
                        m_vis, m_over, m_cut, m_steps = _pt_medium_transmittance(
                            refit, origin, ray_d, f,
                            _pt_shadow_tmax(origin, ray_d, reach), m_stack,
                            layer_offset_triangles, t_nodes, t_node_miss,
                            t_leaf_prim, t_leaf_tspan, t_first_leaf, tri_pos,
                            tri_mat, tri_shell, media_offset, vol_i, vol_f,
                            _pt_vol_seed(vol_seed, q))
                        media_over += m_over
                        media_cut += m_cut
                        vol_over += m_steps
                        tau *= m_vis
                        if b_prim >= 0:
                            m_stack, m_push = _pt_medium_cross(
                                m_stack, tri_shell, f, b_prim, media_offset,
                                leg_d.dot(b_ng) < 0.0)
                            media_over += m_push
                        if (m_over != 0) or (m_cut != 0):
                            ok = 0
            if media_over != 0:
                ok = 0
        if ok == 1:
            nr2 = 1.0 / ti.max(eta_prod * eta_prod, 1e-12)
            if fixed == 0:
                last = ti.math.vec3(ps[0, 0], ps[0, 1], ps[0, 2])
                for qq in ti.static(range(1, MNEE_MAX_VERTICES)):
                    if qq < count:
                        last = ti.math.vec3(ps[qq, 0], ps[qq, 1], ps[qq, 2])
                e = (last - y).normalized()
            else:
                e = -y
    return straight, ok, w0, g, tau, nr2, e, d_path, overflow, capped, \
        media_over, media_cut, vol_over, count


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
             # 1 = manifold next-event estimation (roadmap section 1,
             # ``pt_caustics_taichi``): a lit receiver's light samples are
             # also connected THROUGH the caustic casters ``tri_shell``'s
             # caster section names, and the BSDF paths those connections
             # cover share their light by MIS (``_PT_CAU_COUNT``,
             # ``pt_caustic_cover``). The host compiles it in
             # only for a batch with shadows on and a caster in it; off is
             # the kernel this file produced before caustics existed.
             caustics: ti.template(),
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
             # The coverage test's per-path record (``PT_CAU_WIDTH``); a
             # one-row dummy unless ``caustics`` is on.
             pt_cau: ti.types.ndarray(),
             projected: ti.template(),
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
    (``rs_sca[r, 0]``) freezes at the first scatter. On each straight segment a
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
    pt_glass_energy = ti.static(ArenaView(arena_f32, aoff[37], (ashp[77], ashp[78])))
    cam_proj_table = ti.static(ArenaView(arena_f32, aoff[38], (ashp[79], ashp[80])))
    pt_vol_i = ti.static(ArenaView(arena_i32, aoff[39], (ashp[81],)))
    pt_vol_f = ti.static(ArenaView(arena_f32, aoff[40], (ashp[82],)))
    cam_proj = ti.static(CameraProjection(cam_proj_table, projected))
    pixels_per_frame = width * height
    # Quadrants patch 0005's per-kernel register cap, off unless
    # rt_settings.megakernel_max_reg asks for one (see megakernel_register_cap).
    if ti.static(megakernel_register_cap()):
        ti.loop_config(max_reg=ti.static(megakernel_register_cap()))
    for i in range(num_active):
        r = active[i]
        num_hits = rs_int[r, 3]
        # Next-event table + environment metadata (runtime words, so one
        # compiled kernel serves every scene shape). ``aov_on`` gates every
        # pt_aov access: with it off the tensor is a [1, PT_AOV_WIDTH] dummy.
        # The words are decoded inside the branches that read them -- an
        # escaping path reads the environment half only.
        media_offset = ti.cast(nee_meta[_NM_MEDIA] + 0.5, ti.i32)
        if (num_hits > 0) or (media_offset > 0):
            (n_ls, env_off, env_w, env_h, env_intensity, env_share,
             cdf_h, cdf_w, aov_on) = _pt_meta_escape(nee_meta)
            (num_nee, far_clip, amb_packed, amb_count, anim_seed,
             tree_on, tree_mix, num_inf, quad_base,
             blue_noise) = _pt_meta_hit(nee_meta)
            # Caustic state (``caustics``): the caster section's offset in
            # tri_shell, whether only medium vertices receive through it
            # (``_NM_CAUSTIC``'s sign), and the path's caustic count
            # (``_PT_CAU_COUNT``). Declared either way; nothing reads them in
            # a kernel without.
            caster_offset = 0
            cau_media_only = 0
            cst = 0
            if ti.static(caustics != 0):
                cau_word = nee_meta[_NM_CAUSTIC]
                caster_offset = ti.cast(ti.abs(cau_word) + 0.5, ti.i32)
                if cau_word < 0.0:
                    cau_media_only = 1
                cst = ti.cast(pt_cau[r, _PT_CAU_COUNT] + 0.5, ti.i32)
            g = ray_offset + rs_pix[r]
            f_rel = g // pixels_per_frame
            f = time_start + f_rel
            p = g - f_rel * pixels_per_frame
            # Which of the chunk's light trees this frame uses: frames whose
            # emitter geometry is identical share one (light_tree.py).
            lt_r = 0
            if tree_on:
                lt_r = lt_frame[f_rel]
            # Sampler seeds and pair bases that are constant for the whole
            # path, hoisted out of the drain loop (roadmap section 0.2): the
            # per-path seed half every draw shares, the first per-crossing
            # pair, and the width of one crossing's block (``2L`` next-event
            # pairs plus the lobe select; see the module docstring's table).
            # ``pt_generate`` derives the same seed from the same words and
            # the two must agree.
            path_seed = ti.u32(0)
            if blue_noise:
                py_bn = p // width
                path_seed = _pt_bn_path_seed(nee_meta, seed_root,
                                             f * anim_seed, p - py_bn * width,
                                             py_bn)
            else:
                path_seed = _pt_path_seed(seed_root, _pt_key(f * anim_seed, p))
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
            eta_scale = rs_sca[r, _SCA_ETA_SCALE]
            mis_carry = ti.cast(pt_acc[r, _PT_ACC_MIS_CARRY] > 0.5, ti.i32)
            mis_org = ti.math.vec3(pt_acc[r, _PT_ACC_MIS_ORIGIN],
                                   pt_acc[r, _PT_ACC_MIS_ORIGIN + 1],
                                   pt_acc[r, _PT_ACC_MIS_ORIGIN + 2])
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
            # Set at the first blurring bounce (``_PT_ACC_SCATTERED``),
            # folded into its sticky column the same way.
            scattered = 0
            media = ti.Vector([-1, -1, -1, -1])
            media_events = ti.max(rs_int[r, _INT_MEDIA_EVENTS], 0)
            media_segments = ti.max(rs_int[r, _INT_MEDIA_SEGMENTS], 0)
            media_bad = 0
            if media_offset > 0:
                for q in ti.static(range(PT_MEDIA_SLOTS)):
                    media[q] = rs_int[r, _INT_MEDIA0 + q]
                if rs_int[r, _INT_MEDIA_INIT] < 0:
                    media, overflow_m, truncated_m = _pt_initial_media(
                        refit, ro, f, layer_offset_triangles,
                        t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                        t_first_leaf, tri_pos, tri_shell, media_offset)
                    ti.atomic_add(pt_stats[PT_STAT_MEDIA_STACK], overflow_m)
                    ti.atomic_add(pt_stats[PT_STAT_MEDIA_QUERY], truncated_m)
                    media_bad = overflow_m + truncated_m
                    # The IOR stack and the medium stack describe the same
                    # near-clipped point, including a camera inside glass.
                    depth_m = 0
                    for q in ti.static(range(PT_MEDIA_SLOTS)):
                        if media[q] >= 0:
                            rs_sca[r, 8 + q] = tri_mat[f % tri_mat.shape[0], media[q], 12]
                            depth_m += 1
                    rs_sca[r, 7] = ti.cast(depth_m, ti.f32)
                    rs_int[r, _INT_MEDIA_INIT] = 1

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
            while (drained < num_hits) or ((media_offset > 0) and (num_hits < kbuf)):
                if media_bad != 0:
                    t_alpha = 0.0
                    done = True
                    absorbed = True
                    break
                # Nearest unconsumed slot; scalars + ti.static extraction keep
                # the kb_* vectors out of local memory (same pattern as
                # wavefront_shade).
                sel = 0
                sel_found = 0
                t_hit = 1e30
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
                    t_hit = ti.max(far_clip - base_dist, t_prev)
                    sel_found = 0

                medium_vertex = False
                medium_prim = _pt_medium_top(media)
                phase_g = 0.0
                medium_albedo = ti.math.vec3(0.0, 0.0, 0.0)
                if (media_offset > 0) and (medium_prim >= 0) and (t_hit > t_prev):
                    sigma_a_m, sigma_s_m, phase_g = _pt_medium_coeff(
                        tri_mat, f, medium_prim)
                    sigma_t_m = sigma_a_m + sigma_s_m
                    medium_albedo = sigma_s_m / ti.max(sigma_t_m, 1e-30)
                    seg_m = t_hit - t_prev
                    weight_m = ti.exp(-sigma_t_m * seg_m)
                    vol_m = _pt_medium_volume(pt_vol_i, tri_mat, f, medium_prim)
                    if vol_m >= 0:
                        # A density grid (pt_volume_taichi): spectral tracking
                        # when the flight may scatter, ratio tracking when it
                        # may not. Its first tentative collision takes the
                        # flight's Sobol pair, as the homogeneous flight
                        # below does; later ones hash the flight's seed.
                        flight_seed = _pt_hash_combine(
                            path_seed, ti.u32(0xA511E9B3) + ti.cast(media_segments, ti.u32))
                        pair_m = PAIR_BOUNCE_BASE + PAIRS_PER_BOUNCE * (max_b - bounces_left) + 4
                        u_m = pt_sample_2d_seeded(flight_seed, pair_m, s_index)
                        scatter_m = 0
                        if (bounces_left > 0) and (sigma_s_m.max() > 0.0):
                            scatter_m = 1
                        hit_m, t_m, weight_m, steps_m = _pt_vol_track(
                            pt_vol_i, pt_vol_f, vol_m, f, ro, rd, t_prev, t_hit,
                            sigma_a_m, sigma_s_m,
                            ti.math.vec3(thru[0], thru[1], thru[2]), scatter_m,
                            _pt_vol_seed(flight_seed, s_index), u_m)
                        ti.atomic_add(pt_stats[PT_STAT_VOLUME_STEPS], steps_m)
                        media_segments += 1
                        stoch = 1
                        if hit_m != 0:
                            medium_vertex = True
                            t_hit = t_m
                    elif (bounces_left > 0) and (sigma_s_m.max() > 0.0):
                        # Fresh domain per straight subsegment, not per launch:
                        # unrelated transparent layers may subdivide a flight.
                        # The 2D Sobol prefix remains pure under batch splitting.
                        flight_seed = _pt_hash_combine(
                            path_seed, ti.u32(0xA511E9B3) + ti.cast(media_segments, ti.u32))
                        pair_m = PAIR_BOUNCE_BASE + PAIRS_PER_BOUNCE * (max_b - bounces_left) + 4
                        u_m = pt_sample_2d_seeded(flight_seed, pair_m, s_index)
                        medium_vertex, distance_m, weight_m = _pt_medium_sample(
                            sigma_a_m, sigma_s_m, seg_m, u_m)
                        media_segments += 1
                        stoch = 1
                        if medium_vertex:
                            t_hit = t_prev + distance_m
                    rgb_before_m = thru[0] + thru[1] + thru[2]
                    for k in ti.static(range(3)):
                        thru[k] *= weight_m[k]
                    thru[3] *= (weight_m[0] + weight_m[1] + weight_m[2]) / 3.0
                    # Coverage carries the same RGB-mean survival estimator.
                    # In particular, analytic absorption/depth exhaustion must
                    # not leave a dark volume at alpha zero. The ratio also
                    # preserves colored survival weights across subsegments;
                    # a sampled collision closes coverage below as usual.
                    if rgb_before_m > 0.0:
                        t_alpha *= (thru[0] + thru[1] + thru[2]) / rgb_before_m
                    if thru[0] + thru[1] + thru[2] <= 0.0:
                        t_alpha = 0.0
                        done = True
                        absorbed = True
                        break
                if (sel_found == 0) and (not medium_vertex):
                    done = True
                    break
                prim = 0
                flags = 0
                a = 0.0
                b = 0.0
                for q in ti.static(range(kbuf)):
                    if (q == sel) and (not medium_vertex):
                        prim = kb_prim[q]
                        flags = kb_flags[q]
                        a = kb_a[q]
                        b = kb_b[q]
                        kb_prim[q] = -1
                if medium_vertex:
                    prim = medium_prim
                    flags = 3  # virtual interior vertex, not a triangle/circuit
                    media_events += 1
                else:
                    drained += 1
                    processed += 1
                # This crossing's dimension-pair block (module docstring):
                # ``2L`` next-event pairs then the lobe select.
                pair_cross0 = pair_nee0 + pairs_per_cross * (processed + media_events)
                htype = flags & 3
                edge_hit = (flags >> 2) & 1
                border = (flags >> 3) & 1

                seam_eps = depth_tie_epsilon
                if (edge_hit == 1) and (t_hit - seam_t <= seam_eps):
                    t_prev = t_hit
                    layer_prev = hit_layer
                    continue
                seam_t = t_hit if edge_hit == 1 else -1e30

                w0 = 1.0 - a - b
                color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                alpha = 0.0
                metalness = -1.0
                rough = 0.0
                pid = _MID_UNLIT
                tex_du, tex_dv = 0.0, 0.0
                if htype == 1:
                    footprint = (base_dist + t_hit) * pixel_size_per_t
                    if ti.static(_projected(cam_proj)):
                        # The part of a pixel's world size that does not grow
                        # along the camera path: all of it for an
                        # orthographic camera.
                        footprint = cam_proj[f, 2] + footprint
                    tex_du, tex_dv = _triangle_uv_footprint(
                        0, f, prim, rd, footprint,
                        tri_pos, tri_uvs, tri_tex_meta, num_colored_triangles)
                    color, alpha = _tri_color_g(
                        0, f, prim, w0, a, b, tri_colors, tri_colors, tri_uvs,
                        tri_tex_meta, textures, num_colored_triangles, tex_du, tex_dv)
                    metalness, rough = _tri_extra_g(
                        0, f, prim, w0, a, b, tri_extra, tri_colors, tri_uvs,
                        tri_tex_meta, textures, num_colored_triangles, tex_du, tex_dv)
                    pid = tri_mat_id[f % tri_mat_id.shape[0], prim]
                elif htype == 0:
                    color, alpha = _sample_circuit_color(
                        prim, f, a, b, border,
                        circuit_meta, circuit_colors, circuit_border_colors)
                    cm = f % circuit_meta.shape[0]
                    metalness = circuit_meta[cm, prim, _M_REFLECTIVITY]
                    rough = circuit_meta[cm, prim, _M_ROUGHNESS]
                if medium_vertex:
                    alpha = 1.0
                    color = ti.math.vec4(medium_albedo[0], medium_albedo[1], medium_albedo[2], 0.0)
                    pid = _MID_PHYSICAL
                alpha = ti.math.clamp(alpha, 0.0, 1.0)

                # Closed-shell opacity ring (``solid_shell_alpha``). On
                # each straight segment a declared shell attenuates ONCE per
                # entry/exit pair: the entering crossing composites and
                # remembers the surface id, the exiting crossing finds the
                # id, removes it, and contributes nothing (alpha 0 makes it
                # a weight-1 pass-through with zero radiance below).  This
                # is the per-ray limit of the sheet route's coverage
                # ceiling; see ``_INT_RING0``. Actual scatters reset the ring
                # so refraction still evaluates both interfaces. Pass-throughs
                # retain it, including along a mirror's reflected segment.
                # A seam-skipped duplicate never toggles the ring twice.
                suppressed = 0
                if htype == 1:
                    sid_cs = ti.cast(
                        tri_shell[f % tri_shell.shape[0], prim % tri_shell.shape[1]], ti.i32)
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
                lit = lit or medium_vertex
                authored = (htype == 1) and (not lit) and (pid != _MID_UNLIT)

                # IOR / transmission of the surface (material or per-texel).
                ior = 0.0
                T = 0.0
                if htype == 1:
                    ior, T = _tri_ior_transmission_g(
                        0, f, prim, w0, a, b, tri_extra, tri_colors, tri_uvs,
                        tri_tex_meta, textures, num_colored_triangles, tex_du, tex_dv)
                elif htype == 0:
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
                needs_normal = needs_normal and (not medium_vertex)
                if needs_normal:
                    if htype == 1:
                        snrm = _tri_normal_g(
                            0, f, prim, w0, a, b, tri_norm, tri_pos, tri_uvs,
                            tri_tex_meta, textures, num_colored_triangles, tex_du, tex_dv)
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

                eta = 0.0
                if (T > 1e-4) and (metalness < 1.0 - 1e-6) and (htype == 1):
                    entering_f = rd.dot(fnrm) < 0.0
                    rel_f = _relative_ior(rs_sca, r, ior, entering_f, 1)
                    eta = 1.0 / ti.max(rel_f, 1e-6) if entering_f else rel_f
                glass = lit and (eta > 0.0) and (ti.abs(eta - 1.0) >= 1e-4)
                if glass:
                    # The diffuse part of a partially transmitting material
                    # reflects on the incident side too. Otherwise an inside
                    # diffuse pick could cross without updating the medium.
                    shade_n = spec_n

                # On the surface hit: ``ro + t_hit * rd`` rounds at the ray
                # origin's magnitude, and a distant camera's primary ray (an
                # eye thousands of units out) sinks it
                # below the lift _pt_offset_ray_origin gives the next ray,
                # whose shadow ray or bounce then hits the surface it left --
                # speckle across every lit surface and mirror. A triangle hit
                # is rebuilt from its barycentrics, a circuit's put back on
                # its plane (_tri_surface_point, _bezier_plane_point). A
                # medium vertex (htype 3) is on no surface and stays where it
                # was sampled: its ``prim`` is the shell triangle it entered
                # through, and read as a circuit it projected the vertex onto
                # an unrelated plane -- each entry facet's fog printed with
                # its own brightness.
                hit_p = ro + t_hit * rd
                if htype == 1:
                    hit_p = _tri_surface_point(f, prim, w0, a, b, tri_pos)
                elif htype == 0:
                    hit_p = _bezier_plane_point(f, prim, hit_p, circuit_meta)
                # Where the material stages measure from: the eye, or off the
                # pinhole the start of the camera ray through the hit
                # (_camera_point).
                hit_cam = cam_pos
                if ti.static(_projected(cam_proj)):
                    hit_cam = _camera_point(f, hit_p, cam_origin, cam_proj)

                # Volumetric absorption on exiting a transmissive interior
                # (Beer-Lambert), mirroring the wavefront's site exactly: the
                # side test uses the RAW interpolated normal (the sided/prepped
                # shading normal is viewer-flipped and would never read
                # "exiting").
                medium_boundary = False
                if (media_offset > 0) and (htype == 1):
                    medium_boundary = _pt_medium_id(tri_shell, f, prim, media_offset) >= 0
                if (T > 1e-4) and (htype == 1) and (pid < _USER_PIPELINE_BASE) and (not medium_boundary):
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
                # Clearcoat / sheen of a physical crossing; its two
                # selection weights ride the vector (scaled by alpha below).
                layer_l = _pt_layer_none()
                if lit and (suppressed == 0) and (not medium_vertex):
                    # ``rough`` is REPLACED by the lobe set's own GGX width:
                    # phong authors its highlight as a Blinn-Phong exponent,
                    # so its lobes, its NEE responses and its continuation
                    # must all read the converted roughness. Every other
                    # pipeline gets its own value back unchanged.
                    e_diff_l, e_spec_l, e_trans_l, f0_l, rough, layer_l = \
                        _pt_lit_lobes(
                            pt_glass_energy, pid, tri_mat, f, prim, albedo3,
                            metalness, rough, ior, T, shade_n, rd, eta)
                    wl_diff = alpha * ti.max(e_diff_l[0],
                                             ti.max(e_diff_l[1],
                                                    e_diff_l[2]))
                    wl_spec = alpha * ti.max(e_spec_l[0],
                                             ti.max(e_spec_l[1],
                                                    e_spec_l[2]))
                    wl_trans = alpha * ti.max(e_trans_l[0],
                                              ti.max(e_trans_l[1],
                                                     e_trans_l[2]))
                    layer_l[_PL_W_COAT] = alpha * layer_l[_PL_W_COAT]
                    layer_l[_PL_W_SHEEN] = alpha * layer_l[_PL_W_SHEEN]
                rough_glass = glass and (rough * rough >= 1e-4)
                # Interpolating three identical f32 transmission values can
                # land one ulp below 1. Do not turn those nominally null
                # boundaries into depth-consuming scatters (or black pixels
                # when max_bounces == 0). The tolerance is below min_weight.
                null_boundary = medium_boundary and (not glass) and (alpha > 0.999999) \
                    and (wl_diff <= 1e-6) and (wl_spec <= 1e-6) and (wl_trans > 0.0) \
                    and (layer_l[_PL_W_COAT] + layer_l[_PL_W_SHEEN] <= 1e-6)

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
                # Caustic state of this crossing (``caustics``; declared
                # unconditionally for the same scoping reason): whether it is
                # a receiver that runs the caustic block, whether the path
                # reaches it without having scattered (its caustics are then
                # direct light, which the firefly clamp must not see; see the
                # accumulation below), and that caustic light.
                mnee_vertex = 0
                cau_direct = 0
                caustic = ti.math.vec3(0.0, 0.0, 0.0)

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
                    # Scaled by what a clearcoat lets out of the emission
                    # under it toward the ray (``_PL_ATT``, exactly 1 without
                    # a coat). Next-event estimation applies the same factor
                    # through ``_pt_emitter_coat_att``.
                    emissive = ti.math.vec3(tri_mat[tm, prim, 0],
                                            tri_mat[tm, prim, 1],
                                            tri_mat[tm, prim, 2]) \
                        * (tri_mat[tm, prim, 3] * layer_l[_PL_ATT])
                    # Synthetic panels emit from their front face only. An
                    # authored diffuse vertex already evaluated their rows:
                    # suppress that immediate emission, while still absorbing
                    # the ray at the opaque panel (never see through it).
                    if prim >= quad_base:
                        if (rd.dot(fnrm) >= 0.0) or (prev_pdf == -2.0):
                            emissive = ti.math.vec3(0.0, 0.0, 0.0)
                    # Emission reached through a sampled smooth lobe is
                    # MIS-weighted against the NEE strategy that also covers
                    # this triangle; camera rays, delta continuations and
                    # emitters outside the table (or their un-sampled back
                    # side) keep weight 1.
                    w_emit = 1.0
                    if (prev_pdf > 0.0) and (tri_emit_prob[prim] > 0.0) \
                            and (fn_len > 1e-12):
                        covered = 1
                        # Behind a smooth transmission, NEE connected the
                        # scatter vertex to this point straight through the
                        # surface: measure from there (``_PT_ACC_MIS_CARRY``).
                        ne_org = ro
                        ne_dist2 = t_hit * t_hit
                        ne_dir = rd
                        if mis_carry == 1:
                            ne_org = mis_org
                            ne_vec = hit_p - mis_org
                            ne_dist2 = ne_vec.dot(ne_vec)
                            ne_dir = ne_vec / ti.max(ti.sqrt(ne_dist2), 1e-12)
                        cos_l = (-ne_dir).dot(fnrm)
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
                            # from the triangle's leaf at the previous
                            # scatter origin: ``ro`` (rs_ro is not touched by
                            # a pass-through crossing), or the recorded
                            # origin behind a smooth transmission.
                            p_ne = tri_emit_prob[prim]
                            if tree_on:
                                p_ne = 0.0
                                ent_m = tri_emit_entry[prim]
                                if ent_m >= 0:
                                    p_ne = tree_mix * _pt_lt_pmf(
                                        lt_node_f, lt_node_i, lt_r,
                                        lt_entry_leaf[lt_r, ent_m], ne_org)
                            pdf_ne = p_ne * ne_dist2 \
                                / ti.max(area_e * c_l, 1e-12) \
                                * ti.cast(n_ls, ti.f32)
                            w_emit = prev_pdf * prev_pdf \
                                / ti.max(prev_pdf * prev_pdf
                                         + pdf_ne * pdf_ne, 1e-20)
                        if ti.static(caustics != 0):
                            # Found through casters from a receiver: a light
                            # sample's chain there may have delivered this
                            # very path (``_PT_CAU_COUNT``). Held back for
                            # ``pt_caustic_cover``, which weights it against
                            # that chain where the walk to this point is this
                            # path -- whichever side of the emitter the
                            # straight line from the receiver sees, so the
                            # selection probability is asked for here even
                            # where ``covered`` is 0 -- and adds it at weight
                            # 1 where it is not.
                            p_nc = 0.0
                            if cst >= 2:
                                p_nc = tri_emit_prob[prim]
                                if tree_on:
                                    p_nc = 0.0
                                    ent_c = tri_emit_entry[prim]
                                    if ent_c >= 0:
                                        p_nc = tree_mix * _pt_lt_pmf(
                                            lt_node_f, lt_node_i, lt_r,
                                            lt_entry_leaf[lt_r, ent_c], ne_org)
                            if p_nc > 0.0:
                                held = 0
                                if (not medium_vertex) and (
                                        emissive.max() > 0.0):
                                    clamp_h = 0.0
                                    if bounces_left < max_b:
                                        clamp_h = ti.max(firefly_clamp, 0.0)
                                    held = _pt_cau_pend(
                                        pt_cau, r, 1, hit_p,
                                        ti.math.vec3(thru[0], thru[1],
                                                     thru[2])
                                        * emissive * alpha,
                                        w_emit, cst - 1, f,
                                        p_nc * ti.cast(n_ls, ti.f32)
                                        / ti.max(area_e, 1e-12), fnrm,
                                        ti.cast(tri_mat[tm, prim,
                                                        _MAT_ONE_SIDED]
                                                + 0.5, ti.i32),
                                        clamp_h)
                                if held == 1:
                                    w_emit = 0.0
                    if medium_vertex:
                        emissive = ti.math.vec3(0.0, 0.0, 0.0)
                    direct = emissive * w_emit
                    recv = 1
                    if tri_mat.shape[2] > _MAT_NO_SHADOW_RECEIVE:
                        if tri_mat[tm, prim, _MAT_NO_SHADOW_RECEIVE] > 0.5:
                            recv = 0
                    sorigin = _pt_offset_ray_origin(
                        hit_p, fnrm if fnrm.dot(-rd) >= 0.0 else -fnrm)
                    if medium_vertex:
                        recv = 1
                        sorigin = hit_p
                    # A caustic receiver: a shadow-receiving lit vertex that
                    # does not refract itself (smooth or rough glass is a
                    # caster's business) and is not an index-matched medium
                    # boundary. A medium vertex is one -- which is how light
                    # reaches the inside of a subsurface solid. So is nothing
                    # while this launch holds an emitter path back for the
                    # coverage test (an emitter crossing that is a receiver
                    # itself): its record still describes that path's
                    # receiver, so this vertex lights as with caustics off.
                    # With ``pt_caustics`` off the section is there for the
                    # subsurface case alone (``cau_media_only``), and a
                    # surface keeps its straight shadow.
                    if ti.static((caustics != 0) and (shadows != 0)):
                        if (caster_offset > 0) and (num_nee > 0) \
                                and (recv == 1) and (not glass) \
                                and (not null_boundary) \
                                and (medium_vertex or (cau_media_only == 0)) \
                                and (pt_cau[r, _PT_CAU_KIND] < 0.5):
                            mnee_vertex = 1
                            if (scattered == 0) and (
                                    pt_acc[r, _PT_ACC_SCATTERED] < 0.5):
                                cau_direct = 1
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
                            # The sample's light end for the caustic block
                            # (``caustics``): its straight shadow ray, the
                            # point (or with ``m_fixed`` the direction) a
                            # chain must reach, the part of its estimator
                            # that does not depend on the chain, and the
                            # density it was drawn with (``n_ls`` samples'
                            # worth, in the light end's own measure; none for
                            # a light row, which no BSDF path hits). Recorded
                            # whether or not the straight sample contributes:
                            # a bent chain can reach a light the straight line
                            # misses (a spot cone's edge, a one-sided panel).
                            m_ready = 0
                            m_fixed = 0
                            m_wi = ti.math.vec3(0.0, 0.0, 0.0)
                            m_ld = 1e7
                            m_y = ti.math.vec3(0.0, 0.0, 0.0)
                            m_ne = ti.math.vec3(0.0, 0.0, 0.0)
                            m_one_sided = 0
                            m_base = ti.math.vec3(0.0, 0.0, 0.0)
                            m_py = 0.0
                            if kind == _NEE_LIGHT_ROW:
                                if p_sel > 1e-12:
                                    ld, lc, _sw_r, wi_v, ld_d, valid = \
                                        _pt_nee_light_row(
                                            light_pos, light_col, f, ref,
                                            hit_p, shade_n,
                                            u_pt[0], u_pt[1])
                                    if ti.static(caustics != 0):
                                        if valid == 1:
                                            m_ready = 1
                                            m_wi = wi_v
                                            m_ld = ld_d
                                            m_y = hit_p + wi_v * ld_d
                                            if light_col.shape[2] > 3:
                                                if ti.cast(light_col[
                                                        f % light_col.shape[0],
                                                        ref, 3] + 0.5,
                                                        ti.i32) \
                                                        == _LT_DIRECTIONAL:
                                                    m_fixed = 1
                                                    m_y = wi_v
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
                                        f_cos_r, _pdf_r = _pt_vertex_f_pdf(
                                            medium_vertex, phase_g, pt_glass_energy,
                                            e_diff_l, e_spec_l, f0_l,
                                            rough, shade_n, rd, ld,
                                            wl_pass, wl_diff, wl_spec,
                                            wl_trans, eta, metalness, albedo3, T,
                                            layer_l)
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
                                    if ti.static(caustics != 0):
                                        if (d_e > 1e-4) and (area_s > 1e-12):
                                            m_ready = 1
                                            m_wi = to_e / d_e
                                            m_ld = d_e
                                            m_y = pe
                                            m_ne = ne
                                            tm_m = f % tri_mat.shape[0]
                                            if tri_mat[tm_m, ref,
                                                       _MAT_ONE_SIDED] > 0.5:
                                                m_one_sided = 1
                                            _ec_m, ea_m = _tri_color_g(
                                                0, f, ref, ew0, ew1, ew2,
                                                tri_colors, tri_colors,
                                                tri_uvs, tri_tex_meta,
                                                textures,
                                                num_colored_triangles)
                                            m_base = ti.math.vec3(
                                                tri_mat[tm_m, ref, 0],
                                                tri_mat[tm_m, ref, 1],
                                                tri_mat[tm_m, ref, 2]) \
                                                * (tri_mat[tm_m, ref, 3]
                                                   * ea_m * area_s * inv_ls
                                                   / p_tri)
                                            m_py = p_tri / area_s \
                                                * ti.cast(n_ls, ti.f32)
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
                                            # What the emitter's clearcoat
                                            # lets out toward this vertex:
                                            # the factor a hit on it reads
                                            # as ``_PL_ATT``.
                                            le = ti.math.vec3(
                                                tri_mat[tm_e, ref, 0],
                                                tri_mat[tm_e, ref, 1],
                                                tri_mat[tm_e, ref, 2]) \
                                                * (tri_mat[tm_e, ref, 3]
                                                   * _pt_emitter_coat_att(
                                                       tri_mat, tri_mat_id,
                                                       tri_norm, tri_pos,
                                                       tri_uvs, tri_tex_meta,
                                                       textures,
                                                       num_colored_triangles,
                                                       f, ref, ew0, ew1, ew2,
                                                       ne, wi))
                                            _ec, e_alpha = _tri_color_g(
                                                0, f, ref, ew0, ew1, ew2,
                                                tri_colors, tri_colors,
                                                tri_uvs, tri_tex_meta,
                                                textures,
                                                num_colored_triangles)
                                            f_cos, pdf_b = _pt_vertex_f_pdf(
                                                medium_vertex, phase_g, pt_glass_energy,
                                                e_diff_l, e_spec_l, f0_l,
                                                rough, shade_n, rd, wi,
                                                wl_pass, wl_diff, wl_spec,
                                                wl_trans, eta, metalness, albedo3, T,
                                                layer_l)
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
                                    if ti.static(caustics != 0):
                                        if pdf_sa > 1e-12:
                                            m_ready = 1
                                            m_fixed = 1
                                            m_wi = dir_e
                                            m_y = dir_e
                                            m_base = _sample_env_map(
                                                f, dir_e, env_off, env_w,
                                                env_h, env_intensity,
                                                textures) \
                                                * (inv_ls / pdf_sa)
                                            m_py = pdf_sa \
                                                * ti.cast(n_ls, ti.f32)
                                    if (pdf_sa > 1e-12) \
                                            and ((shade_n.dot(dir_e) > 1e-6)
                                                 or ((wl_spec + layer_l[_PL_W_COAT]
                                                      + layer_l[_PL_W_SHEEN] > 0.0) and
                                                     (spec_n.dot(dir_e) > 1e-6))
                                                 or rough_glass or medium_vertex):
                                        ec = _sample_env_map(
                                            f, dir_e, env_off, env_w,
                                            env_h, env_intensity,
                                            textures)
                                        f_cos, pdf_b = _pt_vertex_f_pdf(
                                            medium_vertex, phase_g, pt_glass_energy,
                                            e_diff_l, e_spec_l, f0_l,
                                            rough, shade_n, rd, dir_e,
                                            wl_pass, wl_diff, wl_spec,
                                            wl_trans, eta, metalness, albedo3, T,
                                            layer_l)
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
                            if ti.static((caustics != 0) and (shadows != 0)):
                                # Manifold next-event estimation (roadmap
                                # section 1): the same light sample, connected
                                # through the casters on its straight segment
                                # by the refracting chain that reaches it.
                                # The straight connection keeps only what the
                                # casters leave uncovered; the chain arrives
                                # MIS-weighted against the BSDF path that
                                # makes the same journey (``_pt_caustic_mis``;
                                # that path's share is ``pt_caustic_cover``'s)
                                # and outside the firefly clamp while the path
                                # is still direct (see ``caustic``).
                                if (mnee_vertex == 1) and (m_ready == 1) \
                                        and ((spec_n.dot(m_wi) > 1e-4)
                                             or medium_vertex):
                                    n_side = ti.math.vec3(0.0, 0.0, 0.0)
                                    if not medium_vertex:
                                        n_side = fnrm if fnrm.dot(-rd) >= 0.0 \
                                            else -fnrm
                                    # A density grid on a leg is ratio-tracked
                                    # from a stream of its own, apart from the
                                    # straight connection's below.
                                    m_vseed = _pt_vol_seed(_pt_vol_seed(
                                        path_seed ^ ti.u32(0x7F4A7C15), pair_sel),
                                        s_index)
                                    (straight_m, m_ok, m_w0, m_g, m_tau,
                                     m_nr2, m_e, m_d, m_over, m_capped,
                                     m_mo, m_mc, m_vo, m_k) = _pt_caustic_chain(
                                        refit, shadow_mode, has_tri, has_bez,
                                        hit_p, sorigin, n_side, m_wi, m_ld,
                                        m_y, m_fixed, f, ff,
                                        pixel_size_per_t, base_dist,
                                        _pixel_size_base(cam_proj, f),
                                        layer_offset_triangles,
                                        t_nodes, t_node_miss, t_leaf_prim,
                                        t_leaf_tspan, t_first_leaf,
                                        tri_pos, tri_norm, tri_colors,
                                        tri_uvs, tri_tex_meta, textures,
                                        tri_extra, tri_mat, tri_mat_id,
                                        num_colored_triangles,
                                        b_nodes, b_node_miss, b_leaf_prim,
                                        b_leaf_tspan, b_first_leaf,
                                        circuit_meta, circuit_colors,
                                        circuit_border_colors, edges_2d,
                                        edge_accel, tri_shell, caster_offset,
                                        media, media_offset, pt_glass_energy,
                                        pt_vol_i, pt_vol_f, m_vseed)
                                    ti.atomic_add(
                                        pt_stats[PT_STAT_VOLUME_STEPS], m_vo)
                                    if m_over != 0:
                                        ti.atomic_add(
                                            pt_stats[PT_STAT_CAUSTIC_CHAIN], 1)
                                    if m_capped != 0:
                                        ti.atomic_add(
                                            pt_stats[PT_STAT_CAUSTIC_SOLVER], 1)
                                    if m_mo != 0:
                                        ti.atomic_add(
                                            pt_stats[PT_STAT_MEDIA_STACK], 1)
                                    if m_mc != 0:
                                        ti.atomic_add(
                                            pt_stats[PT_STAT_MEDIA_QUERY], 1)
                                    contrib *= straight_m
                                    if m_ok == 1:
                                        # Per emitter kind, the estimator the
                                        # straight sample uses with the
                                        # chain's geometry term in place of
                                        # the inverse square law, capped as a
                                        # concentration over the unobstructed
                                        # light at the same path length.
                                        m_light = ti.math.vec3(0.0, 0.0, 0.0)
                                        # The solid angle the chain's first
                                        # leg sweeps per unit light measure.
                                        m_jac = m_g
                                        if kind == _NEE_LIGHT_ROW:
                                            # ``_light_eval`` at a stand-in
                                            # point: the unfolded distance
                                            # along the direction the light
                                            # emits into, so the falloff, the
                                            # range fade, the spot cone and an
                                            # area cell's cosine all read the
                                            # bent path.
                                            tl_m = f % light_pos.shape[0]
                                            lp_m = ti.math.vec3(
                                                light_pos[tl_m, ref, 0],
                                                light_pos[tl_m, ref, 1],
                                                light_pos[tl_m, ref, 2])
                                            _ld_m, lc_m, _sw_m, _fr_m = \
                                                _light_eval(
                                                    light_pos, light_col, f,
                                                    ref, lp_m + m_e * m_d,
                                                    shade_n)
                                            conc = m_g
                                            if m_fixed == 0:
                                                conc = m_g * m_d * m_d
                                            m_light = lc_m * (
                                                ti.min(conc,
                                                       MNEE_MAX_CONCENTRATION)
                                                * inv_ls / p_sel)
                                        elif kind == _NEE_EMISSIVE_TRI:
                                            cos_m = m_e.dot(m_ne)
                                            if (m_one_sided == 1) \
                                                    and (cos_m <= 1e-6):
                                                cos_m = 0.0
                                            d2_m = ti.max(m_d * m_d, 1e-12)
                                            m_light = m_base * (
                                                ti.abs(cos_m)
                                                * ti.min(m_g * d2_m,
                                                         MNEE_MAX_CONCENTRATION)
                                                / d2_m)
                                            m_jac = m_g * ti.abs(cos_m)
                                        else:
                                            m_light = m_base * ti.min(
                                                m_g, MNEE_MAX_CONCENTRATION)
                                        f_cos_m, _pdf_m = _pt_vertex_f_pdf(
                                            medium_vertex, phase_g,
                                            pt_glass_energy,
                                            e_diff_l, e_spec_l, f0_l,
                                            rough, shade_n, rd, m_w0,
                                            wl_pass, wl_diff, wl_spec,
                                            wl_trans, eta, metalness,
                                            albedo3, T, layer_l)
                                        # The BSDF path makes this journey
                                        # only with a bounce left for the
                                        # receiver's scatter and one for each
                                        # caster crossing after it.
                                        p_b_m = _pdf_m
                                        if bounces_left < m_k + 1:
                                            p_b_m = 0.0
                                        m_add = f_cos_m * m_light * m_tau \
                                            * (m_nr2 * _pt_caustic_mis(
                                                m_py, m_jac, p_b_m))
                                        if cau_direct == 1:
                                            caustic += m_add
                                        else:
                                            direct += m_add
                            if (contrib[0] != 0.0) or (contrib[1] != 0.0) \
                                    or (contrib[2] != 0.0):
                                if rough_glass:
                                    # Same geometric support as the sampler:
                                    # a perturbed normal must not turn a
                                    # reflected direction into a crossing.
                                    geom_refl = rd.dot(fnrm) * wi_vis.dot(fnrm) < 0.0
                                    if (spec_n.dot(wi_vis) > 0.0) != geom_refl:
                                        contrib = ti.math.vec3(0.0, 0.0, 0.0)
                                vis3 = ti.math.vec3(1.0, 1.0, 1.0)
                                if ti.static(shadows != 0):
                                    if recv == 1:
                                        if (spec_n.dot(wi_vis) > 1e-4) or rough_glass or medium_vertex:
                                            shadow_origin = sorigin
                                            if rough_glass:
                                                shadow_origin = _pt_offset_ray_origin(
                                                    hit_p, fnrm if fnrm.dot(wi_vis) >= 0.0 else -fnrm)
                                            # ldist was measured at hit_p.
                                            # Pulling the origin toward the
                                            # light must shorten the segment,
                                            # or a near-origin emitter can
                                            # shadow itself after refraction.
                                            shadow_ldist = ldist \
                                                - (shadow_origin - hit_p).dot(wi_vis)
                                            vis3 = _pt_nee_visibility(
                                                refit, shadow_mode,
                                                has_tri, has_bez,
                                                shadow_origin, wi_vis, shadow_ldist,
                                                f, ff,
                                                pixel_size_per_t,
                                                base_dist,
                                                _pixel_size_base(cam_proj, f),
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
                                if media_offset > 0:
                                    m_origin = hit_p
                                    m_stack = media
                                    m_over = 0
                                    if not medium_vertex:
                                        m_origin = _pt_offset_ray_origin(
                                            hit_p, fnrm if fnrm.dot(wi_vis) >= 0.0 else -fnrm)
                                        if medium_boundary and (rd.dot(fnrm) * wi_vis.dot(fnrm) > 0.0):
                                            m_stack, m_over = _pt_medium_cross(
                                                m_stack, tri_shell, f, prim, media_offset,
                                                rd.dot(fnrm) < 0.0)
                                    m_dist = _pt_shadow_tmax(m_origin, wi_vis,
                                        ldist - (m_origin - hit_p).dot(wi_vis))
                                    # A density grid on the way is ratio-tracked
                                    # from this light sample's own stream.
                                    m_seed = _pt_vol_seed(_pt_vol_seed(
                                        path_seed ^ ti.u32(0x3C6EF372), pair_sel), s_index)
                                    m_vis, m_over2, m_cut, m_steps = _pt_medium_transmittance(
                                        refit, m_origin, wi_vis, f, m_dist, m_stack,
                                        layer_offset_triangles,
                                        t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                                        t_first_leaf, tri_pos, tri_mat, tri_shell, media_offset,
                                        pt_vol_i, pt_vol_f, m_seed)
                                    ti.atomic_add(pt_stats[PT_STAT_MEDIA_STACK], m_over + m_over2)
                                    ti.atomic_add(pt_stats[PT_STAT_MEDIA_QUERY], m_cut)
                                    ti.atomic_add(pt_stats[PT_STAT_VOLUME_STEPS], m_steps)
                                    if m_over != 0:
                                        m_vis = ti.math.vec3(0.0, 0.0, 0.0)
                                    vis3 *= m_vis
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
                    # Always pass this visibility vector to the stage. With
                    # surface shadows off it is all ones except for physical
                    # medium extinction; no scene-specific kernel gate is needed.
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
                                                _pixel_size_base(cam_proj, f),
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
                                if (media_offset > 0) and (w > 0.0):
                                    wi_m, distance_m, valid_m = _pt_light_sample_point(
                                        light_pos, light_col, f, li, hit_p, u_pt[0], u_pt[1])
                                    if valid_m == 1:
                                        seed_m = _pt_vol_seed(_pt_vol_seed(
                                            path_seed ^ ti.u32(0x3C6EF372), pair_sel), s_index)
                                        vis_m, over_m, cut_m, steps_m = _pt_medium_transmittance(
                                            refit, sorigin_a, wi_m, f,
                                            _pt_shadow_tmax(sorigin_a, wi_m,
                                                distance_m - (sorigin_a - hit_p).dot(wi_m)),
                                            media, layer_offset_triangles,
                                            t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                                            t_first_leaf, tri_pos, tri_mat, tri_shell, media_offset,
                                            pt_vol_i, pt_vol_f, seed_m)
                                        ti.atomic_add(pt_stats[PT_STAT_MEDIA_STACK], over_m)
                                        ti.atomic_add(pt_stats[PT_STAT_MEDIA_QUERY], cut_m)
                                        ti.atomic_add(pt_stats[PT_STAT_VOLUME_STEPS], steps_m)
                                        base_m = light_vis_index(s, 0)
                                        for channel_m in ti.static(range(3)):
                                            vis[base_m + channel_m] *= vis_m[channel_m]
                        # Built AFTER the fill only for readability: the vectors
                        # are mutated in place, so a view built before it would
                        # read the same values.
                        lpos_v = ti.static(_SampledLightView(light_pos, lrow))
                        lcol_v = ti.static(
                            _SampledLightView(light_col, lrow, lscale))
                        local = _pt_authored_surface(
                            media_offset > 0, frag_pipelines, tri_pids, prim, f, hit_p, -rd,
                            snrm, fnrm, albedo3, color[3],
                            lpos_v, lcol_v, n_slots, tri_mat_id,
                            tri_mat, shadows, vis, hit_cam)
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
                                                _pixel_size_base(cam_proj, f),
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
                        if media_offset > 0:
                            sorigin_m = _pt_offset_ray_origin(
                                hit_p, fnrm if fnrm.dot(-rd) >= 0.0 else -fnrm)
                            for li_m in range(ti.min(num_lights, vis_lights)):
                                u1_m = _pt_rng_seeded(path_seed, s_index, processed * 64 + li_m, 2)
                                u2_m = _pt_rng_seeded(path_seed, s_index, processed * 64 + li_m, 3)
                                wi_m, distance_m, valid_m = _pt_light_sample_point(
                                    light_pos, light_col, f, li_m, hit_p, u1_m, u2_m)
                                if valid_m == 1:
                                    seed_m = _pt_vol_seed(_pt_vol_seed(
                                        path_seed ^ ti.u32(0x7A3D1F55),
                                        processed * 64 + li_m), s_index)
                                    vis_m, over_m, cut_m, steps_m = _pt_medium_transmittance(
                                        refit, sorigin_m, wi_m, f,
                                        _pt_shadow_tmax(sorigin_m, wi_m,
                                            distance_m - (sorigin_m - hit_p).dot(wi_m)),
                                        media, layer_offset_triangles,
                                        t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                                        t_first_leaf, tri_pos, tri_mat, tri_shell, media_offset,
                                        pt_vol_i, pt_vol_f, seed_m)
                                    ti.atomic_add(pt_stats[PT_STAT_MEDIA_STACK], over_m)
                                    ti.atomic_add(pt_stats[PT_STAT_MEDIA_QUERY], cut_m)
                                    ti.atomic_add(pt_stats[PT_STAT_VOLUME_STEPS], steps_m)
                                    base_m = light_vis_index(li_m, 0)
                                    for channel_m in ti.static(range(3)):
                                        vis[base_m + channel_m] *= vis_m[channel_m]
                        local = _pt_authored_surface(
                            media_offset > 0, frag_pipelines, tri_pids, prim, f, hit_p, -rd,
                            snrm, fnrm, albedo3, color[3],
                            light_pos, light_col, num_lights, tri_mat_id,
                            tri_mat, shadows, vis, hit_cam)
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
                if ti.static(caustics != 0):
                    # A caustic received by a path that has not scattered
                    # yet -- a camera ray, or one that only passed mirrors
                    # and smooth glass, which choose no direction -- is
                    # direct light and skips the clamp: a caustic is a
                    # concentrated image of its emitter, and the clamp would
                    # cap every focus at its value. Once the path has
                    # scattered the caustic is indirect light like any other,
                    # and it went through ``direct`` and the clamp above.
                    acc += ti.math.vec4(thru[0] * caustic[0],
                                        thru[1] * caustic[1],
                                        thru[2] * caustic[2], 0.0) * alpha

                # AOV guides: this crossing's contribution to the delta
                # prefix, at the SAME weights the radiance composite uses
                # (no firefly clamp -- a guide, not radiance). A suppressed
                # closed-shell exit has alpha 0 and adds nothing.
                if (aov_open == 1) and (not null_boundary):
                    aov_alb += ti.math.vec3(thru[0], thru[1], thru[2]) \
                        * (alpha * albedo3)
                    aov_nrm += (thru[3] * alpha) * shade_n

                if medium_vertex:
                    t_alpha = 0.0
                    aov_open = 0
                    scattered = 1
                    bounce_ord_m = max_b - bounces_left
                    pair_phase = PAIR_BOUNCE_BASE + PAIRS_PER_BOUNCE * bounce_ord_m
                    u_phase = pt_sample_2d_seeded(path_seed, pair_phase + 5, s_index)
                    new_rd_m, prev_pdf = _pt_hg_sample(rd, phase_g, u_phase)
                    survived_m = 1
                    if bounce_ord_m >= rr_start:
                        u_rr_m = pt_sample_2d_seeded(path_seed, pair_phase, s_index)[1]
                        p_rr_m = ti.math.clamp(
                            ti.max(thru[0], ti.max(thru[1], thru[2])) * eta_scale,
                            _PT_RR_FLOOR, 1.0)
                        if u_rr_m >= p_rr_m:
                            survived_m = 0
                        else:
                            thru /= p_rr_m
                    if survived_m == 0:
                        absorbed = True
                        done = True
                        break
                    if ti.static(caustics != 0):
                        cst = mnee_vertex
                        if mnee_vertex == 1:
                            # The receiver the coverage test walks from: a
                            # medium vertex has no side and no horizon.
                            _pt_cau_receiver(
                                pt_cau, r, hit_p,
                                ti.math.vec3(0.0, 0.0, 0.0),
                                ti.math.vec3(0.0, 0.0, 0.0), new_rd_m,
                                prev_pdf, cau_direct)
                    mis_carry = 0
                    ro = hit_p
                    rd = new_rd_m
                    base_dist += t_hit
                    t_prev = 0.0
                    layer_prev = 1e30
                    seam_t = -1e30
                    ring = ti.Vector([-1, -1, -1, -1])
                    bounces_left -= 1
                    bounced = True
                    break

                if null_boundary:
                    # No change of direction or scattering density. Keep the
                    # previous vertex/pdf for emitter MIS and spend no bounce.
                    media, overflow_m = _pt_medium_cross(
                        media, tri_shell, f, prim, media_offset, rd.dot(fnrm) < 0.0)
                    ti.atomic_add(pt_stats[PT_STAT_MEDIA_STACK], overflow_m)
                    if overflow_m != 0:
                        t_alpha = 0.0
                        done = True
                        absorbed = True
                        break
                    _write_ior_stack(rs_sca, r, r, ior, rd.dot(fnrm) < 0.0, 1, 1)
                    for k in ti.static(range(3)):
                        thru[k] *= e_trans_l[k]
                    thru[3] *= (e_trans_l[0] + e_trans_l[1] + e_trans_l[2]) / 3.0
                    t_prev = t_hit
                    layer_prev = hit_layer
                    continue

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
                                             thru[2])) * eta_scale < min_weight:
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
                                           ti.max(thru[1], thru[2])) * eta_scale,
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
                                             thru[2])) * eta_scale < min_weight:
                                # Retired as absorbed: ``rd`` still points
                                # through the surface, so an escape would
                                # show what lies behind it.
                                absorbed = True
                                done = True
                                break
                            # A new geometric segment needs a fresh shell
                            # pairing, including custom reflection/refraction.
                            ring = ti.Vector([-1, -1, -1, -1])
                            mis_carry = 0
                            if ti.static(caustics != 0):
                                cst = 0
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
                # Continuation: pass-through | diffuse | specular | transmit
                # | clearcoat | sheen.
                # ----------------------------------------------------------
                w_pass = 1.0 - alpha
                e_diff = ti.math.vec3(0.0, 0.0, 0.0)
                e_spec = ti.math.vec3(0.0, 0.0, 0.0)
                e_trans = ti.math.vec3(0.0, 0.0, 0.0)
                f0 = ti.math.vec3(0.0, 0.0, 0.0)
                w_coat = 0.0
                w_sheen = 0.0
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
                        w_coat = layer_l[_PL_W_COAT]
                        w_sheen = layer_l[_PL_W_SHEEN]
                    elif authored:
                        e_diff = albedo3
                    elif metalness >= 0.0:
                        # Reflective / transmissive circuit or unlit surface.
                        met = ti.math.clamp(metalness, 0.0, 1.0)
                        f0 = ti.math.vec3(0.04, 0.04, 0.04) * (1.0 - met) \
                            + albedo3 * met
                        if met > 0.0:
                            e_spec = _pt_ggx_energy(
                                pt_glass_energy, f0, n_dot_v, rough)
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
                # The layers' lobes sit past the base ones in the pick, so a
                # material without them draws exactly as it always did.
                w_base = w_pass + w_diff + w_spec + w_trans
                w_sum = w_base + w_coat + w_sheen
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
                    if medium_boundary:
                        media, overflow_m = _pt_medium_cross(
                            media, tri_shell, f, prim, media_offset, rd.dot(fnrm) < 0.0)
                        ti.atomic_add(pt_stats[PT_STAT_MEDIA_STACK], overflow_m)
                        if overflow_m != 0:
                            t_alpha = 0.0
                            done = True
                            absorbed = True
                            break
                    # Deterministic in an unlit stack (probability 1 there).
                    scale = (1.0 - alpha) * (w_sum / w_pass)
                    thru *= scale
                    t_alpha *= scale
                    t_prev = t_hit
                    layer_prev = hit_layer
                    if ti.max(thru[0], ti.max(thru[1], thru[2])) * eta_scale < min_weight:
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
                    # A smooth transmission keeps the last scatter's MIS
                    # state (``_PT_ACC_MIS_CARRY``); every other pick
                    # replaces it.
                    smooth_cross = 0
                    prev_kept = prev_pdf
                    # The caustic count after this bounce
                    # (``_PT_CAU_COUNT``); each lobe below sets its own.
                    cst_next = 0
                    if pick < w_pass + w_diff:
                        # A diffuse pick is non-delta: the AOV prefix ends at
                        # this crossing (whose albedo/normal it just added).
                        aov_open = 0
                        scattered = 1
                        p_sel = w_diff / w_sum
                        new_rd = _pt_cosine_direction(shade_n, u_dir)
                        if glass and (rd.dot(fnrm) * new_rd.dot(fnrm) >= 0.0):
                            ok = 0
                        # ``e_diff`` carries the sheen's scaling at l = v;
                        # this is the rest of it toward ``new_rd`` (1 without).
                        _bl_d, s_ratio, _bv_d = _pt_base_scale(
                            layer_l, spec_n.dot(new_rd))
                        tint = e_diff * (alpha * s_ratio / ti.max(p_sel, 1e-6))
                        if glass:
                            # A mixed diffuse/glass material must evaluate
                            # the full mixture for either sampling strategy.
                            fc_mix, pdf_mix = _pt_lit_f_pdf(
                                pt_glass_energy,
                                e_diff_l, e_spec_l, f0_l, rough, shade_n,
                                rd, new_rd, wl_pass, wl_diff, wl_spec,
                                wl_trans, eta, metalness, albedo3, T,
                                layer_l)
                            tint = fc_mix * (alpha / ti.max(pdf_mix, 1e-12))
                        diff_mean = (alpha * s_ratio / ti.max(p_sel, 1e-6)) \
                            * ((e_diff[0] + e_diff[1] + e_diff[2]) / 3.0)
                        if glass:
                            diff_mean = (tint[0] + tint[1] + tint[2]) / 3.0
                        thru = ti.math.vec4(
                            thru[0] * tint[0], thru[1] * tint[1],
                            thru[2] * tint[2],
                            thru[3] * diff_mean)
                        new_ro = _pt_offset_ray_origin(
                            hit_p,
                            fnrm if fnrm.dot(new_rd) >= 0.0 else -fnrm)
                        # MIS state: only a lit vertex ran the NEE block, so
                        # only its sampled direction carries a pdf for the
                        # next emitter hit to weight against.
                        prev_pdf = -2.0 if authored else 0.0
                        if lit:
                            _fc_d, prev_pdf = _pt_lit_f_pdf(
                                pt_glass_energy,
                                e_diff_l, e_spec_l, f0_l, rough, shade_n,
                                rd, new_rd, wl_pass, wl_diff, wl_spec,
                                wl_trans, eta, metalness, albedo3, T,
                                layer_l)
                        if ti.static(caustics != 0):
                            cst_next = mnee_vertex
                    elif pick >= w_base:
                        # Clearcoat or sheen (non-delta: close the prefix),
                        # each drawn by its own lobe and weighted by its own
                        # ``f_cos / pdf`` over its selection probability --
                        # the same per-lobe estimator as the base lobes.
                        aov_open = 0
                        is_coat = 0
                        p_sel = w_sheen / w_sum
                        if pick < w_base + w_coat:
                            is_coat = 1
                            p_sel = w_coat / w_sum
                        cc_r = layer_l[_PL_CC_ROUGH]
                        if (is_coat == 0) or (cc_r * cc_r >= 1e-4):
                            scattered = 1
                            if ti.static(caustics != 0):
                                cst_next = mnee_vertex
                        new_rd, tint_l, ok = _pt_sample_layer(
                            layer_l, spec_n, rd, u_dir, is_coat)
                        if glass and (rd.dot(fnrm) * new_rd.dot(fnrm) >= 0.0):
                            ok = 0
                        if ok == 1:
                            tint = tint_l * (alpha / ti.max(p_sel, 1e-12))
                            gmean = (tint[0] + tint[1] + tint[2]) / 3.0
                            thru = ti.math.vec4(thru[0] * tint[0],
                                                thru[1] * tint[1],
                                                thru[2] * tint[2],
                                                thru[3] * gmean)
                            new_ro = _pt_offset_ray_origin(
                                hit_p,
                                fnrm if fnrm.dot(new_rd) > 0.0 else -fnrm)
                            _fc_l, prev_pdf = _pt_lit_f_pdf(
                                pt_glass_energy,
                                e_diff_l, e_spec_l, f0_l, rough, shade_n,
                                rd, new_rd, wl_pass, wl_diff, wl_spec,
                                wl_trans, eta, metalness, albedo3, T,
                                layer_l)
                    elif glass:
                        # Remapping the outer lobe draw leaves a uniform
                        # independent Fresnel draw after choosing the facet.
                        wg = w_spec + w_trans
                        ug = (pick - w_pass - w_diff) / ti.max(wg, 1e-12)
                        new_rd, tint_g, crossed, delta_g, ok = _pt_sample_glass(
                            pt_glass_energy,
                            rd, spec_n, rough, eta, f0, metalness, albedo3, T,
                            u_dir, ug)
                        geometric_cross = rd.dot(fnrm) * new_rd.dot(fnrm) > 0.0
                        if (crossed == 1) != geometric_cross:
                            ok = 0
                        prev_pdf = 0.0
                        if delta_g == 1:
                            _bl_g, _r_g, bv_g = _pt_base_scale(layer_l, 1.0)
                            tint_g *= alpha * w_sum / ti.max(wg, 1e-12) * bv_g
                            if crossed == 1:
                                # NEE from the last scatter vertex already
                                # reached past this surface (its shadow ray
                                # passes transmissive blockers, tinted), so
                                # what the crossing finds is MIS-covered by
                                # it. Only the smooth reflection is not.
                                prev_pdf = prev_kept
                                smooth_cross = 1
                                if ti.static(caustics != 0):
                                    # One more caster on a receiver's chain;
                                    # glass that is not a caster is passed
                                    # tinted by the chain's legs, as by NEE.
                                    cst_next = cst
                                    if (cst >= 1) and (_pt_caster_id(
                                            tri_shell, f, prim,
                                            caster_offset) >= 0):
                                        cst_next = cst + 1
                                        if cst_next > MNEE_MAX_VERTICES + 1:
                                            cst_next = 0
                        else:
                            aov_open = 0
                            scattered = 1
                            fc_g, prev_pdf = _pt_lit_f_pdf(
                                pt_glass_energy,
                                e_diff_l, e_spec_l, f0_l, rough, shade_n,
                                rd, new_rd, wl_pass, wl_diff, wl_spec,
                                wl_trans, eta, metalness, albedo3, T,
                                layer_l)
                            if prev_pdf <= 1e-12:
                                ok = 0
                            tint_g = fc_g * (alpha / ti.max(prev_pdf, 1e-12))
                        if ok == 1:
                            gmean = (tint_g[0] + tint_g[1] + tint_g[2]) / 3.0
                            thru = ti.math.vec4(thru[0] * tint_g[0],
                                                thru[1] * tint_g[1],
                                                thru[2] * tint_g[2],
                                                thru[3] * gmean)
                            if crossed == 1:
                                _write_ior_stack(rs_sca, r, r, ior,
                                                 rd.dot(fnrm) < 0.0, 1, 1)
                                eta_scale /= ti.max(eta * eta, 1e-12)
                            new_ro = _pt_offset_ray_origin(
                                hit_p, fnrm if fnrm.dot(new_rd) >= 0.0 else -fnrm)
                    elif pick < w_pass + w_diff + w_spec:
                        # GGX is non-delta (any roughness): close the prefix.
                        aov_open = 0
                        if rough * rough >= 1e-4:
                            # Its alpha off the 1e-4 floor: a blurring lobe,
                            # not a mirror (``_PT_ACC_SCATTERED``).
                            scattered = 1
                            if ti.static(caustics != 0):
                                cst_next = mnee_vertex
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
                            fres = _pt_fresnel(
                                f0, v_dot_h, eta, metalness, albedo3)
                            # What the clearcoat and sheen leave this lobe.
                            bl_s, _r_s, _bv_s = _pt_base_scale(
                                layer_l, n_dot_l2)
                            fres = fres * bl_s
                            lam_v = _pt_smith_lambda(wo_l[2], a_g)
                            lam_l = _pt_smith_lambda(n_dot_l2, a_g)
                            g_ratio = (1.0 + lam_v) \
                                / ti.max(1.0 + lam_v + lam_l, 1e-6)
                            # Turquin compensation on the sampled lobe.
                            comp = _pt_ggx_compensation(
                                pt_glass_energy, f0, wo_l[2], rough)
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
                                    pt_glass_energy,
                                    e_diff_l, e_spec_l, f0_l, rough,
                                    shade_n, rd, new_rd, wl_pass, wl_diff,
                                    wl_spec, wl_trans, eta, metalness, albedo3, T,
                                    layer_l)
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
                        # Refraction / a tinted pane is a delta lobe, but
                        # NEE from the last scatter vertex passed this
                        # surface too (tinted): keep that vertex's pdf, as
                        # the smooth-glass crossing above does.
                        prev_pdf = prev_kept
                        smooth_cross = 1
                        if ti.static(caustics != 0):
                            cst_next = cst
                    if (media_offset > 0) and medium_boundary and (ok == 1):
                        if rd.dot(fnrm) * new_rd.dot(fnrm) > 0.0:
                            top_before = _pt_medium_top(media)
                            media, overflow_m = _pt_medium_cross(
                                media, tri_shell, f, prim, media_offset, rd.dot(fnrm) < 0.0)
                            ti.atomic_add(pt_stats[PT_STAT_MEDIA_STACK], overflow_m)
                            if overflow_m != 0:
                                ok = 0
                            # A refracting boundary of a scattering interior
                            # is where NEE's straight connection stops
                            # (``_pt_medium_transmittance``), so the smooth
                            # crossing above kept a pdf no light sample
                            # covers: what the path finds past it is its
                            # alone. Weighting it against that NEE darkened
                            # a dense conservative cube in a white furnace
                            # to 57%. A caster crossing on a caustic chain
                            # (``cst_next`` >= 2) keeps it: the coverage
                            # test weighs that path against the chain.
                            if (smooth_cross == 1) and (cst_next < 2):
                                if _pt_medium_stops_connection(
                                        pt_vol_i, tri_mat, f, top_before,
                                        _pt_medium_top(media)) == 1:
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
                            ti.max(thru[0], ti.max(thru[1], thru[2])) * eta_scale,
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
                    if ti.max(thru[0], ti.max(thru[1], thru[2])) * eta_scale < min_weight:
                        # Retired as absorbed: ``rd`` is still the incoming
                        # direction, so an escape would sample the env (or
                        # background) behind the surface.
                        absorbed = True
                        done = True
                        break
                    if smooth_cross == 1:
                        if mis_carry == 0:
                            # ``ro`` is still the last scatter vertex: the
                            # first crossing behind it records it.
                            mis_org = ro
                            mis_carry = 1
                    else:
                        mis_carry = 0
                    if ti.static(caustics != 0):
                        if (cst_next == 1) and (smooth_cross == 0):
                            # A blurring scatter at a receiver (a smooth
                            # crossing only carries the count): what the
                            # coverage test walks from, as the caustic block
                            # above did -- this side, this horizon.
                            _pt_cau_receiver(
                                pt_cau, r, hit_p,
                                fnrm if fnrm.dot(-rd) >= 0.0 else -fnrm,
                                spec_n, new_rd, prev_pdf, cau_direct)
                        cst = cst_next
                    ring = ti.Vector([-1, -1, -1, -1])
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
            pt_acc[r, _PT_ACC_MIS_CARRY] = ti.cast(mis_carry, ti.f32)
            for k in ti.static(range(3)):
                pt_acc[r, _PT_ACC_MIS_ORIGIN + k] = mis_org[k]
            rs_sca[r, 0] = t_alpha
            rs_sca[r, 1] = t_prev
            rs_sca[r, 2] = layer_prev
            rs_sca[r, 3] = seam_t
            rs_sca[r, 4] = base_dist
            rs_sca[r, _SCA_PREV_PDF] = prev_pdf
            rs_sca[r, _SCA_ETA_SCALE] = eta_scale
            rs_int[r, 0] = bounces_left
            rs_int[r, 1] = processed
            rs_int[r, 2] = _DONE if done else _ACTIVE
            if media_offset > 0:
                for q in ti.static(range(PT_MEDIA_SLOTS)):
                    rs_int[r, _INT_MEDIA0 + q] = media[q]
                rs_int[r, _INT_MEDIA_EVENTS] = media_events
                rs_int[r, _INT_MEDIA_SEGMENTS] = media_segments
            if stoch == 1:
                # Sticky: only ever written 1.0, never cleared, so a path that
                # became stochastic on an earlier launch stays flagged.
                pt_acc[r, _PT_ACC_STOCH] = 1.0
            if scattered == 1:
                pt_acc[r, _PT_ACC_SCATTERED] = 1.0
            if ti.static(caustics != 0):
                pt_cau[r, _PT_CAU_COUNT] = ti.cast(cst, ti.f32)
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
                        if ti.static(caustics != 0):
                            # Through casters from a receiver: held back for
                            # the coverage test, as at an emitter hit
                            # (``_PT_CAU_COUNT``).
                            if (cst >= 2) and (p_e > 0.0):
                                clamp_e = 0.0
                                if bounces_left < max_b:
                                    clamp_e = ti.max(firefly_clamp, 0.0)
                                if _pt_cau_pend(
                                        pt_cau, r, 2, rd,
                                        ti.math.vec3(leftover[0] * ec[0],
                                                     leftover[1] * ec[1],
                                                     leftover[2] * ec[2]),
                                        w_env, cst - 1, f, p_e,
                                        ti.math.vec3(0.0, 0.0, 0.0), 0,
                                        clamp_e) == 1:
                                    w_env = 0.0
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
                    if ti.static(caustics != 0):
                        cst_e = ti.cast(pt_cau[r, _PT_CAU_COUNT] + 0.5,
                                        ti.i32)
                        if (cst_e >= 2) and (p_e > 0.0):
                            clamp_x = 0.0
                            if rs_int[r, 0] < rs_int[r, 4]:
                                clamp_x = ti.max(firefly_clamp, 0.0)
                            if _pt_cau_pend(
                                    pt_cau, r, 2, rd,
                                    ti.math.vec3(leftover_e[0] * ec[0],
                                                 leftover_e[1] * ec[1],
                                                 leftover_e[2] * ec[2]),
                                    w_env, cst_e - 1, f, p_e,
                                    ti.math.vec3(0.0, 0.0, 0.0), 0,
                                    clamp_x) == 1:
                                w_env = 0.0
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
    ("pt_glass_energy", "f32", 2),
    ("cam_proj_table", "f32", 2),
    ("pt_vol_i", "i32", 1),
    ("pt_vol_f", "f32", 1),
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
    "vis_lights", "auth_sampled", "caustics",
    "frag_pipelines", "frag_scatters", "tri_pids", "seed_root",
    "sample_base",
    "tile_pixels", "rr_start", "firefly_clamp", "time_start", "width",
    "height", "ray_offset", "rs_ro", "rs_rd", "rs_sca", "rs_int", "rs_pix",
    "hit_f", "hit_i", "pt_thru", "pt_acc", "pt_stats", "nee_cdf", "nee_ref",
    "nee_meta", "tri_emit_prob", "env_cdf", "tri_shell", "pt_aov", "pt_cau",
    "tri_emit_entry", "lt_node_f", "lt_node_i", "lt_entry_leaf", "lt_frame",
    "nee_inf_cdf", "nee_inf_ref", "pt_glass_energy", "cam_proj_table",
    "pt_vol_i", "pt_vol_f", "projected",
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
def pt_caustic_cover_arena(active: ti.types.ndarray(), num_active: ti.i32,
                           t_nodes: NODE_ARG, t_first_leaf: ti.i32,
                           num_colored_triangles: ti.i32,
                           caster_offset: ti.i32,
                           layer_offset_triangles: ti.f32,
                           refit: ti.template(),
                           pt_acc: ti.types.ndarray(),
                           pt_cau: ti.types.ndarray(),
                           arena_f32: ti.types.ndarray(),
                           arena_i32: ti.types.ndarray(),
                           aoff: ti.types.ndarray(),
                           ashp: ti.types.ndarray()):
    """The coverage test of the BSDF paths ``pt_shade`` held back (roadmap
    section 1; launched after each ``pt_shade`` of a batch with ``caustics``
    on, over the same active slots).

    A path that scattered at a caustic receiver and then reached an emitter
    through casters left its radiance in its ``pt_cau`` record instead of
    adding it. Where a light sample at that receiver, aimed at the same light
    end, would have walked to this same chain (``_pt_mnee_covered``), the
    caustic block counted the path too, and the two share it by the power
    heuristic: this adds the BSDF path's share, ``1 - _pt_caustic_mis``,
    from the same densities the caustic block weighed, and leaves a direct
    caustic outside the firefly clamp as the caustic block does. Everywhere
    else the chain cannot deliver the path, and it is added at weight 1.
    Where the straight segment to the light end held more casters than a
    chain may, the caustic block kept the straight connection tinted, as
    with caustics off, and the path gets the MIS weight it had then. The
    record is cleared either way, so the next ``pt_shade`` starts with none.
    """
    # Arena-bound parameters, in _PT_CAUSTIC_COVER_ARENA's order.
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
    circuit_meta = ti.static(ArenaView(
        arena_f32, aoff[10], (ashp[24], ashp[25], ashp[26])))
    tri_mat = ti.static(ArenaView(arena_f32, aoff[11], (ashp[27], ashp[28], ashp[29])))
    tri_shell = ti.static(ArenaView(arena_i32, aoff[12], (ashp[30], ashp[31])))
    for i in range(num_active):
        r = active[i]
        kind = ti.cast(pt_cau[r, _PT_CAU_KIND] + 0.5, ti.i32)
        if kind > 0:
            pt_cau[r, _PT_CAU_KIND] = 0.0
            f = ti.cast(pt_cau[r, _PT_CAU_F] + 0.5, ti.i32)
            x0 = ti.math.vec3(0.0, 0.0, 0.0)
            n_side = ti.math.vec3(0.0, 0.0, 0.0)
            spec = ti.math.vec3(0.0, 0.0, 0.0)
            w_b = ti.math.vec3(0.0, 0.0, 0.0)
            y = ti.math.vec3(0.0, 0.0, 0.0)
            n_e = ti.math.vec3(0.0, 0.0, 0.0)
            value = ti.math.vec3(0.0, 0.0, 0.0)
            for c in ti.static(range(3)):
                x0[c] = pt_cau[r, _PT_CAU_X0 + c]
                n_side[c] = pt_cau[r, _PT_CAU_NSIDE + c]
                spec[c] = pt_cau[r, _PT_CAU_SPEC + c]
                w_b[c] = pt_cau[r, _PT_CAU_WB + c]
                y[c] = pt_cau[r, _PT_CAU_Y + c]
                n_e[c] = pt_cau[r, _PT_CAU_NE + c]
                value[c] = pt_cau[r, _PT_CAU_VAL + c]
            k = ti.cast(pt_cau[r, _PT_CAU_K] + 0.5, ti.i32)
            one_sided = ti.cast(pt_cau[r, _PT_CAU_ONE_SIDED] + 0.5, ti.i32)
            fixed = 0
            if kind == 2:
                fixed = 1
            covered, overflow, jac = _pt_mnee_covered(
                refit, x0, n_side, spec, w_b, y, n_e, one_sided, fixed, k, f,
                layer_offset_triangles, t_nodes, t_node_miss, t_leaf_prim,
                t_leaf_tspan, t_first_leaf, tri_pos, tri_norm, tri_colors,
                tri_uvs, tri_tex_meta, textures, tri_extra, tri_mat,
                circuit_meta, num_colored_triangles, tri_shell, caster_offset)
            clamp = pt_cau[r, _PT_CAU_CLAMP]
            if overflow == 1:
                value *= pt_cau[r, _PT_CAU_WOLD]
            elif covered == 1:
                value *= 1.0 - _pt_caustic_mis(
                    pt_cau[r, _PT_CAU_PY], jac, pt_cau[r, _PT_CAU_PB])
                if pt_cau[r, _PT_CAU_DIRECT] > 0.5:
                    clamp = 0.0
            if clamp > 0.0:
                value = ti.min(value, ti.math.vec3(clamp, clamp, clamp))
            for c in ti.static(range(3)):
                pt_acc[r, c] += value[c]


#: What ``pt_caustic_cover`` binds through the arena (see ``_PT_SHADE_ARENA``).
_PT_CAUSTIC_COVER_ARENA = (
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
    ("circuit_meta", "f32", 3),
    ("tri_mat", "f32", 3),
    ("tri_shell", "i32", 2),
)

_PT_CAUSTIC_COVER_PARAMS = (
    "active", "num_active", "t_nodes", "t_node_miss", "t_leaf_prim",
    "t_leaf_tspan", "t_first_leaf", "tri_pos", "tri_norm", "tri_extra",
    "tri_colors", "tri_uvs", "tri_tex_meta", "textures",
    "num_colored_triangles", "circuit_meta", "tri_mat", "tri_shell",
    "caster_offset", "layer_offset_triangles", "refit", "pt_acc", "pt_cau",
)

_pt_caustic_cover_launch = arena_packed(
    __name__, "pt_caustic_cover_arena", _PT_CAUSTIC_COVER_PARAMS,
    _PT_CAUSTIC_COVER_ARENA)


def pt_caustic_cover(*args):
    """Pack the arena-bound arguments, then launch ``pt_caustic_cover_arena``
    (argument order ``_PT_CAUSTIC_COVER_PARAMS``).
    """
    return _pt_caustic_cover_launch(*args)


@ti.kernel
def pt_reduce(sample_base: ti.i32, tile_pixels: ti.i32, wave_samples: ti.i32,
              transparent: ti.i32, adaptive: ti.i32,
              width: ti.i32, height: ti.i32,
              pix_list: ti.types.ndarray(),
              out: ti.types.ndarray(), pt_acc: ti.types.ndarray(),
              accum: ti.types.ndarray(), accum_odd: ti.types.ndarray(),
              bg_scattered: ti.types.ndarray()):
    """Fold one wave's per-path rows into the chunk's per-pixel sample sums.

    One thread per wave pixel walks its own wave samples in index order --
    exclusive slots, no atomics, a fixed summation order, because no path
    splits today (not because frames are promised to match). The background
    (prefilled into ``out`` at byte scale) enters here through each path's
    leftover throughput; a sample's alpha is ``1 - t_a * (1 - bg_alpha)``
    where ``t_a`` is the deterministically-composited camera-segment
    transparency, so alpha matches the deterministic renderer's compositing
    contract in expectation (exactly, on scatter-free content).

    A path that scattered (``_PT_ACC_SCATTERED``) sees
    ``bg_scattered[f_rel, 0:4]`` instead -- the frame's average backdrop at
    byte scale (``path_tracer.scattered_background``) -- because the pixel
    behind the camera ray says nothing about the direction it left in.
    ``bg_scattered[f_rel, 4]`` is 1.0 when every pixel of the frame holds the
    same backdrop (a solid colour); there the average IS the pixel, and the
    fold below is the single-sum one, byte for byte.

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
        uniform = bg_scattered[f_rel, 4] > 0.5
        sum_acc = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        sum_leftover = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        sum_scat = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        odd_acc = ti.math.vec3(0.0, 0.0, 0.0)
        odd_leftover = ti.math.vec3(0.0, 0.0, 0.0)
        odd_scat = ti.math.vec3(0.0, 0.0, 0.0)
        stoch_count = 0.0
        sum_t_alpha = 0.0
        for k in range(wave_samples):
            r = k * tile_pixels + p_local
            scat = (not uniform) and (pt_acc[r, _PT_ACC_SCATTERED] > 0.5)
            for c in ti.static(range(4)):
                sum_acc[c] += pt_acc[r, c]
                if scat:
                    sum_scat[c] += pt_acc[r, _PT_ACC_LEFTOVER + c]
                else:
                    sum_leftover[c] += pt_acc[r, _PT_ACC_LEFTOVER + c]
            if adaptive != 0:
                stoch_count += pt_acc[r, _PT_ACC_STOCH]
                if (sample_base + k) % 2 == 1:
                    for c in ti.static(range(3)):
                        odd_acc[c] += pt_acc[r, c]
                        if scat:
                            odd_scat[c] += pt_acc[r, _PT_ACC_LEFTOVER + c]
                        else:
                            odd_leftover[c] += pt_acc[r, _PT_ACC_LEFTOVER + c]
            sum_t_alpha += pt_acc[r, _PT_ACC_ALPHA]
        background = ti.math.vec4(
            ti.cast(out[f_rel, p, 0], ti.f32),
            ti.cast(out[f_rel, p, 1], ti.f32),
            ti.cast(out[f_rel, p, 2], ti.f32),
            ti.cast(out[f_rel, p, 3], ti.f32)) / 255.0
        bg_s = ti.math.vec4(bg_scattered[f_rel, 0], bg_scattered[f_rel, 1],
                            bg_scattered[f_rel, 2], bg_scattered[f_rel, 3]) \
            / 255.0
        if uniform:
            for c in ti.static(range(4)):
                accum[f_rel, p, c] += sum_acc[c] \
                    + sum_leftover[c] * background[c]
            if adaptive != 0:
                for c in ti.static(range(3)):
                    accum_odd[f_rel, p, c] += odd_acc[c] \
                        + odd_leftover[c] * background[c]
        else:
            for c in ti.static(range(4)):
                accum[f_rel, p, c] += sum_acc[c] \
                    + sum_leftover[c] * background[c] + sum_scat[c] * bg_s[c]
            if adaptive != 0:
                for c in ti.static(range(3)):
                    accum_odd[f_rel, p, c] += odd_acc[c] \
                        + odd_leftover[c] * background[c] \
                        + odd_scat[c] * bg_s[c]
        if adaptive != 0:
            accum_odd[f_rel, p, 3] += stoch_count
        if transparent != 0:
            bg_alpha = ti.cast(out[f_rel, p, 4], ti.f32) / 255.0
            accum[f_rel, p, 4] += ti.cast(wave_samples, ti.f32) \
                - sum_t_alpha * (1.0 - bg_alpha)
