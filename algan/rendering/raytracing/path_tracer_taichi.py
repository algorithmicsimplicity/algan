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
    Lit surfaces receive next-event estimation over every packed light row
    (radiometry via the shared ``_light_eval``, so brightness matches the
    deterministic stages), and scatter one importance-sampled continuation:
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
  lighting per the stage formulas (same lobes, same light units, minus the
  ambient fill, which real indirect transport replaces) plus the sampled
  continuation above. Emissive surfaces illuminate their surroundings
  through BSDF-sampled paths.
* manim/toon/normal/matcap/depth/user (0, 6-9, >= 10): authored appearance --
  the hit is shaded exactly as the deterministic renderer shades it
  (``_run_frag_pipeline``, shadow visibility included), the result treated
  as emitted radiance, and the path continues as a Lambert bounce on the
  base color so these surfaces send and receive indirect light.
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
all seeded by hashes of ``(pt_seed, frame, pixel, pair)``. Every sample is a
pure function of those inputs -- independent of tile, wave, batch and chunk
splits, and of thread scheduling. Decisions whose count per path is unbounded
(the pass/scatter choice at each crossed surface, per-light soft-shadow
jitter) draw from a hash RNG keyed on the same inputs plus the peel step, so
they stay reproducible without consuming Sobol dimensions.

Dimension-pair allocation (a fixed table; keep in sync with ``pt_shade``).
``B`` is the render's ``max_bounces`` and ``L`` is ``pt_light_samples``; the
next-event block sits after every bounce pair because it draws per surface
CROSSING ``c`` (a translucent stack visits several lit surfaces per bounce
ordinal), not per bounce:

===========================  ==================================================
pair                         use
===========================  ==================================================
0                            sub-pixel jitter (2D)
1                            lens (2D) -- reserved for depth of field
2 + 6b + 0                   bounce ``b``: y Russian roulette (x unused -- the
                             lobe select draws white noise, see the roadmap)
2 + 6b + 1                   bounce ``b``: BSDF direction (2D)
2 + 6b + 2, 3                bounce ``b``: reserved (legacy light slots)
2 + 6b + 4, 5                bounce ``b``: reserved for volumes
2 + 6B + 2(cL + s) + 0       crossing ``c``, NEE sample ``s``: x entry select
2 + 6B + 2(cL + s) + 1       crossing ``c``, NEE sample ``s``: light point (2D)
===========================  ==================================================
"""

import taichi as ti

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
    min_hit_distance,
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
    _d_charlie,
    _ggx_distribution,
    _ibl_sheen_brdf,
    _light_eval,
    _prep_normal,
    _run_frag_pipeline,
    _sided_shading_normal,
    _smith_geometry,
    _v_neubelt,
    light_vis_index,
    max_shadow_lights,
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

# Sampler dimension pairs (see the module docstring's table).
PAIR_PIXEL = 0
PAIR_LENS = 1
PAIR_BOUNCE_BASE = 2
PAIRS_PER_BOUNCE = 6
_PAIR_LOBE = 0
_PAIR_BSDF_DIR = 1

# Per-path commit row (``pt_acc``): radiance accumulated so far, the leftover
# throughput the background shows through, and the camera-segment
# alpha transparency (see ``pt_reduce``).
PT_ACC_WIDTH = 9
_PT_ACC_LEFTOVER = 4
_PT_ACC_ALPHA = 8

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

# Word layout of the ``nee_meta`` f32 vector (integer-valued words carry
# exact small ints; decoded with ``+ 0.5`` casts).
NEE_META_WIDTH = 10
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
    """
    return _pt_hash_combine(ti.cast(f, ti.u32), ti.cast(pixel, ti.u32))


@ti.func
def pt_sample_2d(seed_root: ti.u32, key: ti.u32, pair: ti.i32,
                 sample_index: ti.i32) -> ti.math.vec2:
    """Sample ``pair`` of the pixel's padded Sobol sequence at
    ``sample_index``: Owen-shuffled index, Owen-scrambled (0,2) point.

    Any prefix of the returned sequence is well stratified, which is what
    makes progressive rendering (waves) and future adaptive sampling sound;
    distinct ``(key, pair)`` values decorrelate into independent sequences.
    """
    pair_seed = _pt_hash_combine(seed_root, _pt_hash_combine(key, ti.cast(pair, ti.u32)))
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
def _pt_rng(seed_root: ti.u32, key: ti.u32, sample_index: ti.i32,
            salt_a: ti.i32, salt_b: ti.i32) -> ti.f32:
    """White-noise uniform in [0, 1) for the unbounded-count decisions
    (pass/scatter choice per crossed surface, per-light shadow jitter): a
    pure hash of the path identity plus two salts, so it is exactly as
    reproducible as the Sobol samples without consuming a dimension pair.
    """
    h = _pt_hash_combine(seed_root, key)
    h = _pt_hash_combine(h, ti.cast(sample_index, ti.u32))
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
    """Smith's Lambda for isotropic GGX (for the exact G1/G2 the VNDF
    estimator wants -- distinct from ``_smith_geometry``'s direct-lighting
    remap, which the NEE response keeps for stage parity).
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
def _pt_ggx_ndf(n_dot_h, alpha):
    """Isotropic GGX normal distribution with ``alpha`` = roughness^2 -- the
    same parameterisation the VNDF sampler and ``_pt_smith_lambda`` use, so
    an evaluated pdf matches the sampled one exactly (``_smith_geometry``'s
    direct-lighting remap deliberately does not).
    """
    a2 = alpha * alpha
    d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0
    return a2 / ti.max(_PI * d * d, 1e-12)


@ti.func
def _pt_lit_lobes(pid, params: ti.template(), f, prim, albedo3, metalness,
                  rough, ior, T, shade_n, rd):
    """Continuation-lobe energies of a physically-integrated (lit) hit:
    ``(e_diff, e_spec, e_trans, f0)``. The single source for both the
    sampled continuation and the NEE-side BSDF evaluation -- MIS is only
    correct while the two agree term for term.
    """
    one3 = ti.math.vec3(1.0, 1.0, 1.0)
    e_diff = ti.math.vec3(0.0, 0.0, 0.0)
    e_spec = ti.math.vec3(0.0, 0.0, 0.0)
    e_trans = ti.math.vec3(0.0, 0.0, 0.0)
    f0 = ti.math.vec3(0.0, 0.0, 0.0)
    n_dot_v = ti.max(shade_n.dot(-rd), 1e-4)
    if (pid == _MID_LAMBERT) or (pid == _MID_PHONG):
        # Pure-diffuse indirect transport: the lambert stage has no specular
        # lobe at all, and phong's Blinn highlight responds to delta lights
        # via NEE only (an indirect GGX proxy for it would add energy its
        # stage never had).
        e_diff = albedo3
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
    return e_diff, e_spec, e_trans, f0


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
def pt_generate(num_slots: ti.i32, tile_pixels: ti.i32, sample_base: ti.i32,
                seed_root: ti.u32, time_start: ti.i32, width: ti.i32,
                height: ti.i32, tile_start: ti.i32,
                half_screen_w: ti.f32, half_screen_h: ti.f32,
                cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
                pixel_basis_x: ti.types.ndarray(),
                pixel_basis_y: ti.types.ndarray(),
                rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
                rs_pix: ti.types.ndarray()):
    """Write each slot's jittered primary ray.

    Slot layout: ``slot = k * tile_pixels + p_local`` holds wave sample
    ``sample_base + k`` of tile pixel ``p_local``, so one wave puts every tile
    pixel's next ``S`` samples in flight and ``pt_reduce`` can walk a pixel's
    slots at stride ``tile_pixels``. The rest of the per-slot state is
    constant at generation and broadcast-filled by the host (the same
    coalesced-fill reasoning as the deterministic ``const_fill`` path).
    """
    pixels_per_frame = width * height
    for slot in range(num_slots):
        p_local = slot % tile_pixels
        s = sample_base + slot // tile_pixels
        g = tile_start + p_local
        f_rel = g // pixels_per_frame
        p = g - f_rel * pixels_per_frame
        f = time_start + f_rel
        py = p // width
        px = p - py * width
        jitter = pt_sample_2d(seed_root, _pt_key(f, p), PAIR_PIXEL, s)
        ro, rd = _generate_ray(f, px, py, jitter[0], jitter[1],
                               half_screen_w, half_screen_h,
                               cam_origin, screen_point,
                               pixel_basis_x, pixel_basis_y)
        for k in ti.static(range(3)):
            rs_ro[slot, k] = ro[k]
            rs_rd[slot, k] = rd[k]
        rs_pix[slot] = p_local


@ti.func
def _pt_nee_visibility(refit: ti.template(), has_tri: ti.template(),
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
    ordered shadow march (translucent blockers tint, ``casts_shadows`` is
    honored by the leaf test inside the walk).
    """
    occ = _shadow_occluded(
        refit, 1, sorigin, wi, f, ff,
        ldist - 20.0 * min_hit_distance,
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
def _pt_direct_response(pid, params: ti.template(), f, prim, albedo,
                        n, view_dir, ld, lc, spec_w):
    """One light's direct response of a physically-integrated pipeline
    (lambert/phong/standard/physical), term for term the matching
    ``shading_taichi`` stage minus the ambient fill (real indirect transport
    replaces it) and minus emissive (added once per hit, not per light).
    """
    tm = f % params.shape[0]
    out = ti.math.vec3(0.0, 0.0, 0.0)
    one = ti.math.vec3(1.0, 1.0, 1.0)
    n_dot_l = ti.max(n.dot(ld), 0.0)
    if pid == _MID_LAMBERT:
        out = albedo * lc * n_dot_l
    elif pid == _MID_PHONG:
        specular = ti.math.vec3(params[tm, prim, 4], params[tm, prim, 5],
                                params[tm, prim, 6])
        shininess = params[tm, prim, 7]
        half = (ld + view_dir).normalized()
        v_dot_h = ti.max(view_dir.dot(half), 0.0)
        n_dot_h = ti.max(n.dot(half), 0.0)
        fresnel = specular + (one - specular) \
            * ti.pow(ti.max(1.0 - v_dot_h, 0.0), 5.0)
        d = (shininess * 0.5 + 1.0) * ti.pow(n_dot_h, ti.max(shininess, 1e-3))
        out = (albedo + fresnel * (0.25 * d * spec_w)) * lc * n_dot_l
    else:
        # standard / physical share the Cook-Torrance core.
        roughness = params[tm, prim, 8]
        metalness = params[tm, prim, 9]
        f0 = ti.math.vec3(0.04, 0.04, 0.04) * (1.0 - metalness) \
            + albedo * metalness
        transmission = 0.0
        if pid == _MID_PHYSICAL:
            ior = params[tm, prim, 12]
            specular_intensity = params[tm, prim, 13]
            specular_color = ti.math.vec3(params[tm, prim, 14],
                                          params[tm, prim, 15],
                                          params[tm, prim, 16])
            ratio = (ior - 1.0) / ti.max(ior + 1.0, 1e-4)
            f0 = (specular_color * (ratio * ratio * specular_intensity)
                  * (1.0 - metalness) + albedo * metalness)
            transmission = params[tm, prim, 24]
        half = (ld + view_dir).normalized()
        n_dot_v = ti.max(n.dot(view_dir), 1e-4)
        n_dot_h = ti.max(n.dot(half), 0.0)
        v_dot_h = ti.max(view_dir.dot(half), 0.0)
        fresnel = f0 + (one - f0) * ti.pow(ti.max(1.0 - v_dot_h, 0.0), 5.0)
        ndf = _ggx_distribution(n_dot_h, roughness)
        geom = _smith_geometry(n_dot_v, n_dot_l, roughness)
        spec = (ndf * geom) * fresnel / ti.max(4.0 * n_dot_v * n_dot_l, 1e-4)
        k_d = (one - fresnel) * ((1.0 - metalness) * (1.0 - transmission))
        out = (k_d * albedo + spec * spec_w) * lc * n_dot_l
        if pid == _MID_PHYSICAL:
            clearcoat = params[tm, prim, 17]
            clearcoat_roughness = params[tm, prim, 18]
            sheen = params[tm, prim, 19]
            sheen_roughness = ti.math.clamp(params[tm, prim, 20], 1e-4, 1.0)
            sheen_c = ti.math.vec3(params[tm, prim, 21], params[tm, prim, 22],
                                   params[tm, prim, 23]) * sheen
            sheen_max = ti.max(sheen_c[0], ti.max(sheen_c[1], sheen_c[2]))
            sheen_comp = 1.0 - sheen_max * ti.max(
                _ibl_sheen_brdf(n_dot_v, sheen_roughness),
                _ibl_sheen_brdf(n_dot_l, sheen_roughness))
            out *= sheen_comp
            cc_ndf = _ggx_distribution(n_dot_h, clearcoat_roughness)
            cc_geom = _smith_geometry(n_dot_v, n_dot_l, clearcoat_roughness)
            cc_fresnel = 0.04 + 0.96 * ti.pow(ti.max(1.0 - v_dot_h, 0.0), 5.0)
            out += lc * (clearcoat * cc_ndf * cc_geom * cc_fresnel
                         / ti.max(4.0 * n_dot_v * n_dot_l, 1e-4)
                         * n_dot_l * spec_w)
            sheen_brdf = _d_charlie(n_dot_h, sheen_roughness) \
                * _v_neubelt(n_dot_v, n_dot_l)
            out += sheen_c * lc * (sheen_brdf * n_dot_l * spec_w)
    return out


@ti.kernel
def pt_shade(active: ti.types.ndarray(), num_active: ti.i32,
             t_nodes: NODE_ARG, t_node_miss: ti.types.ndarray(),
             t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
             t_first_leaf: ti.i32,
             tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
             tri_extra: ti.types.ndarray(),
             tri_colors: ti.types.ndarray(), tri_uvs: ti.types.ndarray(),
             tri_tex_meta: ti.types.ndarray(), textures: ti.types.ndarray(),
             num_colored_triangles: ti.i32,
             b_nodes: NODE_ARG, b_node_miss: ti.types.ndarray(),
             b_leaf_prim: ti.types.ndarray(), b_leaf_tspan: ti.types.ndarray(),
             b_first_leaf: ti.i32,
             circuit_meta: ti.types.ndarray(),
             circuit_colors: ti.types.ndarray(),
             circuit_border_colors: ti.types.ndarray(),
             edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
             tri_mat_id: ti.types.ndarray(), tri_mat: ti.types.ndarray(),
             light_pos: ti.types.ndarray(), light_col: ti.types.ndarray(),
             num_lights: ti.i32,
             pixel_world_scale: ti.types.ndarray(),
             layer_offset_triangles: ti.f32,
             cam_origin: ti.types.ndarray(),
             refit: ti.template(), has_tri: ti.template(),
             has_bez: ti.template(), shadows: ti.template(),
             frag_pipelines: ti.template(), tri_pids: ti.template(),
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
             nee_cdf: ti.types.ndarray(), nee_ref: ti.types.ndarray(),
             nee_meta: ti.types.ndarray(),
             tri_emit_prob: ti.types.ndarray(),
             env_cdf: ti.types.ndarray(),
             tri_shell: ti.types.ndarray(),
             pt_aov: ti.types.ndarray()):
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
    draws from the power-weighted next-event table (``nee_cdf`` /
    ``nee_ref``): delta and area light rows at stage radiometry, emissive
    triangles and the environment map at physical radiometry with
    power-heuristic MIS against the continuation lobes (the emitters BSDF
    paths can also find; delta lights have no geometry to MIS against).
    Escaping rays sample the environment map in their own direction, so
    mirrors and GI see the sky the deterministic renderer shows.
    """
    pixels_per_frame = width * height
    for i in range(num_active):
        r = active[i]
        num_hits = rs_int[r, 3]
        # Next-event table + environment metadata (runtime words, so one
        # compiled kernel serves every scene shape).
        num_nee = ti.cast(nee_meta[_NM_COUNT] + 0.5, ti.i32)
        env_share = nee_meta[_NM_ENV_SHARE]
        n_ls = ti.max(ti.cast(nee_meta[_NM_LIGHT_SAMPLES] + 0.5, ti.i32), 1)
        env_off = ti.cast(nee_meta[_NM_ENV_OFF] + 0.5, ti.i32)
        env_w = ti.cast(nee_meta[_NM_ENV_W] + 0.5, ti.i32)
        env_h = ti.cast(nee_meta[_NM_ENV_H] + 0.5, ti.i32)
        env_intensity = nee_meta[_NM_ENV_INTENSITY]
        cdf_h = ti.cast(nee_meta[_NM_ENV_CDF_H] + 0.5, ti.i32)
        cdf_w = ti.cast(nee_meta[_NM_ENV_CDF_W] + 0.5, ti.i32)
        # AOV accumulation for the denoiser (every pt_aov access is gated on
        # this: with it off the tensor is a [1, PT_AOV_WIDTH] dummy).
        aov_on = nee_meta[_NM_AOV] > 0.5
        if num_hits > 0:
            g = ray_offset + rs_pix[r]
            f = time_start + g // pixels_per_frame
            p = g - (g // pixels_per_frame) * pixels_per_frame
            key = _pt_key(f, p)
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
            ring = ti.Vector([-1, -1, -1, -1])
            for q in ti.static(range(_SHELL_RING_SLOTS)):
                ring[q] = rs_int[r, _INT_RING0 + q]
            aov_open = 0
            aov_alb = ti.math.vec3(0.0, 0.0, 0.0)
            aov_nrm = ti.math.vec3(0.0, 0.0, 0.0)
            if aov_on:
                if pt_aov[r, _AOV_CLOSED] < 0.5:
                    aov_open = 1

            kb_t = ti.Vector([0.0] * kbuf)
            kb_layer = ti.Vector([0.0] * kbuf)
            kb_prim = ti.Vector([0] * kbuf)
            kb_flags = ti.Vector([0] * kbuf)
            kb_a = ti.Vector([0.0] * kbuf)
            kb_b = ti.Vector([0.0] * kbuf)
            for q in ti.static(range(kbuf)):
                kb_t[q] = hit_f[i, q, 0]
                kb_layer[q] = hit_f[i, q, 1]
                kb_a[q] = hit_f[i, q, 2]
                kb_b[q] = hit_f[i, q, 3]
                kb_prim[q] = hit_i[i, q, 0]
                kb_flags[q] = hit_i[i, q, 1]

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
                    e_diff_l, e_spec_l, e_trans_l, f0_l = _pt_lit_lobes(
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

                # ----------------------------------------------------------
                # Local radiance of this crossing (emission semantics).
                # ----------------------------------------------------------
                local = ti.math.vec4(color[0], color[1], color[2], color[3])
                # A suppressed crossing contributes nothing: skip its NEE /
                # frag-pipeline work outright (its ``local`` would be
                # multiplied by the zeroed alpha anyway).
                if lit and (suppressed == 0):
                    tm = f % tri_mat.shape[0]
                    emissive = ti.math.vec3(tri_mat[tm, prim, 0],
                                            tri_mat[tm, prim, 1],
                                            tri_mat[tm, prim, 2]) \
                        * tri_mat[tm, prim, 3]
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
                            pdf_ne = tri_emit_prob[prim] * (t_hit * t_hit) \
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
                    offs_s = fnrm
                    if fnrm.dot(-rd) < 0.0:
                        offs_s = -fnrm
                    sorigin = hit_p + offs_s * (10.0 * min_hit_distance)
                    # Deterministic fill from the direction-less rows -- the
                    # literal ambient / hemisphere semantics of the stages,
                    # never a visibility ray, never in the sampled table.
                    if light_col.shape[2] > 3:
                        tl_f = f % light_col.shape[0]
                        for li in range(num_lights):
                            lt_row = ti.cast(light_col[tl_f, li, 3] + 0.5,
                                             ti.i32)
                            if (lt_row == _LT_AMBIENT) \
                                    or (lt_row == _LT_HEMISPHERE):
                                ld, lc, spec_w, _frac = _light_eval(
                                    light_pos, light_col, f, li, hit_p,
                                    shade_n)
                                if (lc[0] != 0.0) or (lc[1] != 0.0) \
                                        or (lc[2] != 0.0):
                                    direct += _pt_direct_response(
                                        pid, tri_mat, f, prim, albedo3,
                                        shade_n, -rd, ld, lc, spec_w)
                    # Next-event estimation: ``pt_light_samples`` draws from
                    # the power-weighted table (delta/area light rows,
                    # emissive triangles, the environment map).
                    if num_nee > 0:
                        inv_ls = 1.0 / ti.cast(n_ls, ti.f32)
                        pair_nee0 = PAIR_BOUNCE_BASE \
                            + PAIRS_PER_BOUNCE * max_b
                        for ls in range(n_ls):
                            pair_sel = pair_nee0 \
                                + 2 * (processed * n_ls + ls)
                            u_sel = pt_sample_2d(seed_root, key, pair_sel,
                                                 s_index)
                            u_pt = pt_sample_2d(seed_root, key,
                                                pair_sel + 1, s_index)
                            entry, p_sel = _pt_pick_nee_entry(
                                nee_cdf, num_nee, u_sel[0])
                            kind = nee_ref[entry, 0]
                            ref = nee_ref[entry, 1]
                            contrib = ti.math.vec3(0.0, 0.0, 0.0)
                            wi_vis = ti.math.vec3(0.0, 0.0, 0.0)
                            ldist = 1e7
                            if kind == _NEE_LIGHT_ROW:
                                if p_sel > 1e-12:
                                    ld, lc, spec_w, wi_v, ld_d, valid = \
                                        _pt_nee_light_row(
                                            light_pos, light_col, f, ref,
                                            hit_p, shade_n,
                                            u_pt[0], u_pt[1])
                                    if (valid == 1) and (
                                            (lc[0] != 0.0)
                                            or (lc[1] != 0.0)
                                            or (lc[2] != 0.0)):
                                        contrib = _pt_direct_response(
                                            pid, tri_mat, f, prim,
                                            albedo3, shade_n, -rd, ld,
                                            lc, spec_w) * (inv_ls / p_sel)
                                        wi_vis = wi_v
                                        ldist = ld_d
                            elif kind == _NEE_EMISSIVE_TRI:
                                p_tri = tri_emit_prob[ref]
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
                                                refit, has_tri, has_bez,
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
                    vis = ti.Vector([1.0] * (3 * max_shadow_lights))
                    if ti.static(shadows != 0):
                        recv_a = 1
                        if pid < _USER_PIPELINE_BASE:
                            if tri_mat.shape[2] > _MAT_NO_SHADOW_RECEIVE:
                                if tri_mat[f % tri_mat.shape[0], prim,
                                           _MAT_NO_SHADOW_RECEIVE] > 0.5:
                                    recv_a = 0
                        if recv_a == 1:
                            offs_a = fnrm
                            if fnrm.dot(-rd) < 0.0:
                                offs_a = -fnrm
                            sorigin_a = hit_p + offs_a \
                                * (10.0 * min_hit_distance)
                            for li in range(num_lights):
                                if li < max_shadow_lights:
                                    u1 = _pt_rng(seed_root, key, s_index,
                                                 processed * 64 + li, 2)
                                    u2 = _pt_rng(seed_root, key, s_index,
                                                 processed * 64 + li, 3)
                                    wi, ldist, valid = _pt_light_sample_point(
                                        light_pos, light_col, f, li, hit_p,
                                        u1, u2)
                                    if valid == 1:
                                        v3 = _pt_nee_visibility(
                                            refit, has_tri, has_bez,
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

                indirect_path = bounces_left < max_b
                add = thru * alpha * local
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
                    n_dot_v = ti.max(shade_n.dot(-rd), 1e-4)
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
                u_lobe = _pt_rng(seed_root, key, s_index, processed, 7)
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
                    u_dir = pt_sample_2d(seed_root, key,
                                         pair_base + _PAIR_BSDF_DIR, s_index)
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
                        offs = fnrm
                        if fnrm.dot(new_rd) < 0.0:
                            offs = -fnrm
                        new_ro = hit_p + offs * (10.0 * min_hit_distance)
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
                        # VNDF sample about the shading normal.
                        t_b, b_b = _pt_onb(shade_n)
                        wo_l = ti.math.vec3(t_b.dot(-rd), b_b.dot(-rd),
                                            ti.max(shade_n.dot(-rd), 1e-4))
                        a_g = ti.max(rough * rough, 1e-4)
                        h_l = _pt_vndf_half_vector(wo_l, a_g, u_dir)
                        h_w = (t_b * h_l[0] + b_b * h_l[1]
                               + shade_n * h_l[2]).normalized()
                        new_rd = (rd - 2.0 * rd.dot(h_w) * h_w).normalized()
                        n_dot_l2 = shade_n.dot(new_rd)
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
                            offs = fnrm if fnrm.dot(new_rd) > 0.0 else -fnrm
                            new_ro = hit_p + offs * (10.0 * min_hit_distance)
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
                            new_ro = hit_p + rd * (10.0 * min_hit_distance)
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
                        u_rr = pt_sample_2d(seed_root, key,
                                            pair_base + _PAIR_LOBE,
                                            s_index)[1]
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


@ti.kernel
def pt_reduce(tile_start: ti.i32, tile_pixels: ti.i32, wave_samples: ti.i32,
              transparent: ti.i32, width: ti.i32, height: ti.i32,
              out: ti.types.ndarray(), pt_acc: ti.types.ndarray(),
              accum: ti.types.ndarray()):
    """Fold one wave's per-path rows into the chunk's per-pixel sample sums.

    One thread per tile pixel walks its own wave samples in index order --
    exclusive slots, no atomics, a fixed summation order, because no path
    splits today (not because frames are promised to match). The background
    (prefilled into ``out`` at byte scale) enters here through each path's
    leftover throughput; a sample's alpha is ``1 - t_a * (1 - bg_alpha)``
    where ``t_a`` is the deterministically-composited camera-segment
    transparency, so alpha matches the deterministic renderer's compositing
    contract in expectation (exactly, on scatter-free content).
    """
    pixels_per_frame = width * height
    for p_local in range(tile_pixels):
        g = tile_start + p_local
        f_rel = g // pixels_per_frame
        p = g - f_rel * pixels_per_frame
        sum_acc = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        sum_leftover = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        sum_t_alpha = 0.0
        for k in range(wave_samples):
            r = k * tile_pixels + p_local
            for c in ti.static(range(4)):
                sum_acc[c] += pt_acc[r, c]
                sum_leftover[c] += pt_acc[r, _PT_ACC_LEFTOVER + c]
            sum_t_alpha += pt_acc[r, _PT_ACC_ALPHA]
        background = ti.math.vec4(
            ti.cast(out[f_rel, p, 0], ti.f32),
            ti.cast(out[f_rel, p, 1], ti.f32),
            ti.cast(out[f_rel, p, 2], ti.f32),
            ti.cast(out[f_rel, p, 3], ti.f32)) / 255.0
        for c in ti.static(range(4)):
            accum[f_rel, p, c] += sum_acc[c] + sum_leftover[c] * background[c]
        if transparent != 0:
            bg_alpha = ti.cast(out[f_rel, p, 4], ti.f32) / 255.0
            accum[f_rel, p, 4] += ti.cast(wave_samples, ti.f32) \
                - sum_t_alpha * (1.0 - bg_alpha)
