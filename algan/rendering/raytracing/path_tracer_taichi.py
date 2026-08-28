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
    thread per pixel sums its own wave samples in a fixed order, so
    accumulation uses no atomics and a render is reproducible run-to-run.

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

Dimension-pair allocation (a fixed table; keep in sync with ``pt_shade``):

===========================  ==================================================
pair                         use
===========================  ==================================================
0                            sub-pixel jitter (2D)
1                            lens (2D) -- reserved for depth of field
2 + 6b + 0                   bounce ``b``: x lobe select, y Russian roulette
2 + 6b + 1                   bounce ``b``: BSDF direction (2D)
2 + 6b + 2                   bounce ``b``: x light select, y spare
2 + 6b + 3                   bounce ``b``: light point (2D)
2 + 6b + 4, 5                bounce ``b``: reserved for volumes
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
    _env_brdf_approx,
    _material_reflectance,
    _offset_transmitted_origin,
    _refract_ray,
    _relative_ior,
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

# Russian-roulette survival floor: a path is never continued with less than
# this probability, bounding the throughput amplification at 1/floor.
_PT_RR_FLOOR = 0.05


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
             pt_stats: ti.types.ndarray()):
    """Consume one traverse's hit-event batch (see the module docstring).

    Deterministic alpha compositing carries every crossed surface's local
    radiance (NEE-lit, frag-shaded, or raw emission by pipeline); the one
    stochastic continuation per path is chosen among pass-through and the
    material's importance-sampled lobes, with proper reweighting, so an
    unlit-only stack keeps the zero-variance composite while lit content
    gets full transport. The camera-segment alpha transparency
    (``rs_sca[r, 0]``) freezes at the first scatter.
    """
    pixels_per_frame = width * height
    for i in range(num_active):
        r = active[i]
        num_hits = rs_int[r, 3]
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
            bounces_left = rs_int[r, 0]
            processed = rs_int[r, 1]
            acc = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
            # Bounce ordinal for the sampler's dimension table; the FIRST
            # value written by the host is max_bounces, so the ordinal is
            # the difference.
            max_b = rs_int[r, 4]

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
                        if fnrm.norm() > 1e-12:
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

                # ----------------------------------------------------------
                # Local radiance of this crossing (emission semantics).
                # ----------------------------------------------------------
                local = ti.math.vec4(color[0], color[1], color[2], color[3])
                if lit:
                    tm = f % tri_mat.shape[0]
                    emissive = ti.math.vec3(tri_mat[tm, prim, 0],
                                            tri_mat[tm, prim, 1],
                                            tri_mat[tm, prim, 2]) \
                        * tri_mat[tm, prim, 3]
                    direct = emissive
                    recv = 1
                    if tri_mat.shape[2] > _MAT_NO_SHADOW_RECEIVE:
                        if tri_mat[tm, prim, _MAT_NO_SHADOW_RECEIVE] > 0.5:
                            recv = 0
                    offs_s = fnrm
                    if fnrm.dot(-rd) < 0.0:
                        offs_s = -fnrm
                    sorigin = hit_p + offs_s * (10.0 * min_hit_distance)
                    for li in range(num_lights):
                        ld, lc, spec_w, _frac = _light_eval(
                            light_pos, light_col, f, li, hit_p, shade_n)
                        if (lc[0] != 0.0) or (lc[1] != 0.0) or (lc[2] != 0.0):
                            resp = _pt_direct_response(
                                pid, tri_mat, f, prim, albedo3, shade_n,
                                -rd, ld, lc, spec_w)
                            vis3 = ti.math.vec3(1.0, 1.0, 1.0)
                            if ti.static(shadows != 0):
                                if (recv == 1) and (spec_w > 0.0):
                                    u1 = _pt_rng(seed_root, key, s_index,
                                                 processed * 64 + li, 0)
                                    u2 = _pt_rng(seed_root, key, s_index,
                                                 processed * 64 + li, 1)
                                    wi, ldist, valid = _pt_light_sample_point(
                                        light_pos, light_col, f, li, hit_p,
                                        u1, u2)
                                    if valid == 1 \
                                            and shade_n.dot(wi) > 1e-4:
                                        vis3 = _pt_nee_visibility(
                                            refit, has_tri, has_bez,
                                            sorigin, wi, ldist, f, ff,
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
                                    elif valid == 1:
                                        vis3 = ti.math.vec3(0.0, 0.0, 0.0)
                            direct += resp * vis3
                    local = ti.math.vec4(direct[0], direct[1], direct[2],
                                         color[3])
                elif authored:
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
                        if (pid == _MID_LAMBERT) or (pid == _MID_PHONG):
                            # Pure-diffuse indirect transport: the lambert
                            # stage has no specular lobe at all, and phong's
                            # Blinn highlight responds to delta lights via
                            # NEE only (an indirect GGX proxy for it would
                            # add energy its stage never had).
                            e_diff = albedo3
                        else:
                            tm2 = f % tri_mat.shape[0]
                            met = ti.math.clamp(ti.max(metalness, 0.0),
                                                0.0, 1.0)
                            diel_f0 = ti.math.vec3(0.04, 0.04, 0.04)
                            if pid == _MID_PHYSICAL:
                                ior_m = tri_mat[tm2, prim, 12]
                                ratio = (ior_m - 1.0) \
                                    / ti.max(ior_m + 1.0, 1e-4)
                                diel_f0 = ti.math.vec3(
                                    tri_mat[tm2, prim, 14],
                                    tri_mat[tm2, prim, 15],
                                    tri_mat[tm2, prim, 16]) \
                                    * (ratio * ratio * tri_mat[tm2, prim, 13])
                            f0 = diel_f0 * (1.0 - met) + albedo3 * met
                            e_spec = _pt_ggx_energy(f0, n_dot_v, rough)
                            _R3, diel_pass = _material_reflectance(
                                rd, shade_n, ti.max(metalness, 0.0), ior,
                                albedo3, T)
                            e_trans = albedo3 * (diel_pass * T)
                            e_diff = albedo3 * ((1.0 - met) * (1.0 - T)) \
                                * (one3 - e_spec)
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
                    # continues and the background must NOT show through.
                    absorbed = True
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
                    elif pick < w_pass + w_diff + w_spec:
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
            rs_int[r, 0] = bounces_left
            rs_int[r, 1] = processed
            rs_int[r, 2] = _DONE if done else _ACTIVE
            if done:
                leftover = thru
                if absorbed:
                    # An absorbed path (opaque surface, RR kill, rejected
                    # sample direction) carries nothing to the background.
                    # A ceiling-truncated path keeps its leftover, matching
                    # the wavefront's documented degrade.
                    leftover = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                for k in ti.static(range(4)):
                    pt_acc[r, _PT_ACC_LEFTOVER + k] = leftover[k]
                pt_acc[r, _PT_ACC_ALPHA] = t_alpha
        else:
            # No surface this segment: the path escapes to the background.
            for k in ti.static(range(4)):
                pt_acc[r, _PT_ACC_LEFTOVER + k] = pt_thru[r, k]
            pt_acc[r, _PT_ACC_ALPHA] = rs_sca[r, 0]
            rs_int[r, 2] = _DONE


@ti.kernel
def pt_reduce(tile_start: ti.i32, tile_pixels: ti.i32, wave_samples: ti.i32,
              transparent: ti.i32, width: ti.i32, height: ti.i32,
              out: ti.types.ndarray(), pt_acc: ti.types.ndarray(),
              accum: ti.types.ndarray()):
    """Fold one wave's per-path rows into the chunk's per-pixel sample sums.

    One thread per tile pixel walks its own wave samples in index order --
    exclusive slots, no atomics, a fixed summation order -- which is what
    makes path-traced output reproducible run-to-run. The background
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
