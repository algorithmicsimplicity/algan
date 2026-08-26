"""Wavefront (stage-split) kernels of the deterministic general ray tracer.

This module splits the tracing process into small per-stage kernels connected
by per-ray state in global memory, driven by a host-side iteration loop
(``tracer.raytrace_render_wavefront``):

* :func:`wavefront_generate_rays` -- initialise per-ray state with primary
  rays (skipped when fused generation folds this into the first traverse; see
  ``settings.WF_GEN_FUSED``).
* :func:`wavefront_traverse_events` -- for each *active* ray, gather the
  KBUF nearest hits (reusing the unchanged ``_collect_hits``) into a transient
  compact event batch indexed by active-queue ordinal.
* :func:`wavefront_shade` -- immediately drain that event batch: built-in /
  custom fragment shading, shadows, and reflection / refraction continuations.

The supported general path does not attach hit arrays to every continuation
pool slot. The legacy textured/material-sorted paths retain
:func:`wavefront_traverse` and its pool-wide K-buffer ABI for reference.
* :func:`wf_composite_accum` (and the AA variants) -- composite each pixel's
  accumulator over the background.

Between iterations the host compacts the still-active rays with the
:func:`compact_ray_slots` kernel into arena-backed ping-pong index buffers
(see ``tracer._ArenaRayCompactor``), so each launch processes only rays that
still have work -- warps refill as rays drop out, which is the divergence fix.
"""
import taichi as ti

from algan.rendering.raytracing.glossy_prefilter_taichi import (
    GL_ROW_DIST as _GL_ROW_DIST,
)
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    _M_IOR,
    _M_REFLECTIVITY,
    _M_TRANSMISSION,
    DEPTH_TIE_EPSILON,
    KBUF,
    MAX_SHADOW_LIGHTS,
    MAX_SURFACES_PER_RAY,
    MIN_ALPHA,
    MIN_HIT_DISTANCE,
    MIN_WEIGHT,
    NODE_ARG,
    _M_IOR,
    _M_REFLECTIVITY,
    _M_ROUGHNESS,
    _M_TRANSMISSION,
    _axis_cos,
    _bezier_normal,
    _collect_hits,
    _comes_after,
    _flat_triangle_color,
    _generate_ray,
    _nearest_surface_g,
    _safe_inverse,
    _sample_circuit_color,
    _shade_tri_hit,
    _shadow_occluded,
    _triangle_color,
    _triangle_extra,
    _triangle_normal,
    finalize_pixel_color,
)
from algan.rendering.raytracing.shading_taichi import (
    _MAT_ATTENUATION_SIGMA,
    _MAT_NO_SHADOW_RECEIVE,
    _MID_UNLIT,
    _USER_PIPELINE_BASE,
    _orient_hit_normals,
    _reflect_frame,
    _shadow_terminator_delta,
    _vis_max_component,
    direct_specular_lobe,
    light_vis_index,
)
from algan.settings._startup import _SOFT_SHADOW_SAMPLES as SOFT_SHADOW_SAMPLES

# Per-ray status codes (rs_int column 2).
_ACTIVE = 0
_DONE = 1

# Nested-dielectric media stack (DESIGN_mesh_identity_open.md §H), carried in
# rs_sca columns beyond the seven the classic layout speaks for. The gate is
# ``rt_settings.nested_ior_mode()`` read live per batch (see
# ``tracer.render_batch_raytraced``'s ``ior_stack_flag``); when it is off the
# state keeps its classic width and every stack line compiles out.
#
# rs_sca[r, _SCA_IOR_DEPTH] is HOW MANY media the ray is inside;
# rs_sca[r, _SCA_IOR_BASE + i] is the i-th medium's index of refraction,
# outermost first. Depth 0 means air.
IOR_STACK_DEPTH = 4          # media a ray may be inside at once
_SCA_IOR_DEPTH = 7           # rs_sca column: stack depth, stored as f32
_SCA_IOR_BASE = 8            # rs_sca column of stack[0]
SCA_WIDTH_PLAIN = 7
SCA_WIDTH_NESTED = _SCA_IOR_BASE + IOR_STACK_DEPTH        # 12


def sca_width(nested_ior):
    """rs_sca row width for a nested-IOR gate value (plain Python)."""
    return SCA_WIDTH_NESTED if nested_ior else SCA_WIDTH_PLAIN


@ti.kernel
def compact_ray_slots(
        source: ti.types.ndarray(), num_source: int,
        scan_pool: ti.template(), desired_status: int,
        rs_int: ti.types.ndarray(), rs_key: ti.types.ndarray(),
        use_key: ti.template(), desired_key: int,
        output: ti.types.ndarray(), output_count: ti.types.ndarray()):
    """Compact matching ray slots into a caller-owned arena buffer.

    This replaces the host-side ``comparison -> nonzero -> index`` PyTorch
    chain.  Those three ordinary CUDA operations each allocated a new tensor
    outside :class:`ManualMemory` on every wavefront iteration, so a nearly
    full render arena could still OOM the device.  The output and its one-word
    counter are supplied by the render arena; the kernel needs no temporary
    device storage.

    ``scan_pool`` selects either every pool slot (needed when ray splitting may
    activate a spare slot) or the previously-active ``source`` list.  The
    optional key predicate is used by the material-sorted path to form one
    bucket at a time without a host sort.
    """
    for i in range(num_source):
        # Define ``r`` in the enclosing Taichi scope. A name first assigned in
        # separate ti.static branches is not visible after the branch during
        # kernel AST transformation.
        r = i
        if ti.static(not scan_pool):
            r = source[i]
        keep = rs_int[r, 2] == desired_status
        if ti.static(use_key):
            keep = keep and rs_key[r] == desired_key
        if keep:
            out_i = ti.atomic_add(output_count[0], 1)
            output[out_i] = r

# Light type ids of the extended packed light rows (see
# algan.rendering.lights and scene_builder._pack_lights). Only the ids the
# shadow code branches on are needed here.
_LT_POINT = 0
_LT_DIRECTIONAL = 1
_LT_AMBIENT = 2
_LT_HEMISPHERE = 3
_LT_SPOT = 4
_LT_AREA_SAMPLE = 5
_LT_ENV_SH = 6


@ti.func
def _light_zero_radiance(light_col: ti.template(), tl, li, ltype, to_light,
                         ldist):
    """1 when light ``li``'s evaluated radiance at this fragment is exactly
    zero from geometry alone: beyond its range fade, outside its spot cone's
    outer angle, or behind its (one-sided) area sample.

    Reproduces ``_light_eval``'s attenuation factors with the same arithmetic
    on the same inputs (``to_light``/``ldist`` are the fan site's
    ``lp - spos`` and its norm, bitwise the stage's ``d``), so "exactly zero"
    here is exactly zero there and the fragment's shadow fan cannot influence
    any stage whose vis-multiplied terms all carry ``lc`` as a factor
    (lambert/phong/toon/standard/physical; normal/matcap/depth consume no
    lights at all, so the cull cannot touch their output either). Callers still gate ``_stage_default``
    out on the hit's pipeline id, which is now **conservative rather than
    load-bearing**: its base-colour fade used to accumulate a vis-weighted
    ``w`` even at ``lc == 0``, and no longer does (the fade share carries
    ``lc``), so admitting it here would be correct -- and would spare those
    fans -- but it has not been measured and is left alone.
    Underflowed (but not bitwise-zero) multipliers are treated as live: that
    only traces a fan whose result multiplies zero, never the reverse.
    """
    zero = 0
    if light_col.shape[2] > 3:
        if (ltype == _LT_POINT) or (ltype == _LT_SPOT) \
                or (ltype == _LT_AREA_SAMPLE):
            rng = light_col[tl, li, 5]
            if rng > 0.0:
                q = ti.math.clamp(ldist / rng, 0.0, 1.0)
                q2 = q * q
                fade = ti.math.clamp(1.0 - q2 * q2, 0.0, 1.0)
                if fade * fade == 0.0:
                    zero = 1
            if (zero == 0) and ((ltype == _LT_SPOT)
                                or (ltype == _LT_AREA_SAMPLE)):
                ld = to_light.normalized()
                if ltype == _LT_SPOT:
                    sd = ti.math.vec3(light_col[tl, li, 6],
                                      light_col[tl, li, 7],
                                      light_col[tl, li, 8])
                    cos_outer = light_col[tl, li, 9]
                    cos_inner = light_col[tl, li, 10]
                    c = (-ld).dot(sd)
                    t = ti.math.clamp(
                        (c - cos_outer)
                        / ti.max(cos_inner - cos_outer, 1e-6), 0.0, 1.0)
                    if t * t * (3.0 - 2.0 * t) == 0.0:
                        zero = 1
                else:
                    an = ti.math.vec3(light_col[tl, li, 6],
                                      light_col[tl, li, 7],
                                      light_col[tl, li, 8])
                    if ti.max((-ld).dot(an), 0.0) == 0.0:
                        zero = 1
    return zero

# Golden-angle increment of the deterministic soft-shadow sample fan.
_GOLDEN_ANGLE = 2.3999632297286533

# R2 low-discrepancy increments for a RECT area light's shadow fan: sample s
# of a cell lands at fractional position (frac(0.5 + a1*s), frac(0.5 +
# a2*s)) scaled to [-1, 1]^2 across it, where a1 = 1/phi2 and a2 = 1/phi2^2
# for the plastic number phi2 = 1.324717957244746 -- the 2-D analogue of the
# golden-angle spiral the disk branch uses. R2 rather than a jittered grid:
# deterministic (no per-cell state to carry or seed), and uniform over any
# sample count S; s = 0 is exactly the cell centre, so a one-sample fan
# degenerates to today's centre ray.
_R2_SEQUENCE_A1 = 0.7548776662466927
_R2_SEQUENCE_A2 = 0.5698402909980532

_PI = 3.141592653589793

# Deferred shadows (the ``deferred_shadows`` compile-time template of the
# shade kernel; currently never enabled -- the tracer always passes 0, the
# separate shadow kernel having measured slower than inline shadows) pack a
# per-(K-buffer hit, light) occlusion bit into a single int32, so at most
# 32 // KBUF lights fit.
# The inline (default) shadow path uses the full ``MAX_SHADOW_LIGHTS``; only
# the deferred bit-packing is bounded here (lights past it stay lit in
# deferred mode -- a no-op unless a user both enables deferred shadows and
# uses more than 32 // KBUF lights).
_DEFERRED_SHADOW_LIGHTS = max(1, min(MAX_SHADOW_LIGHTS, 32 // KBUF))


@ti.func
def _sample_env_map(f, rd, env_off, env_w, env_h, env_intensity,
                    textures: ti.template()):
    """RGB of the equirectangular environment map in unit direction ``rd``.

    The map lives at ``(env_off, env_w, env_h)`` in the shared flat texel
    buffer (appended by the tracer after the material textures). ``u`` wraps
    the azimuth (+x at the image's horizontal center), ``v = 0`` is straight
    up (+y), matching the usual equirect convention of sky at the top row.
    """
    u = ti.atan2(rd[2], rd[0]) * (0.5 / _PI) + 0.5
    v = 0.5 - ti.asin(ti.math.clamp(rd[1], -1.0, 1.0)) / _PI
    smp = _sample_tex_vec5(f, u, v, env_off, env_w, env_h, textures)
    return ti.math.vec3(smp[0], smp[1], smp[2]) * env_intensity


@ti.func
def _corner_ior(f, prim, w0, w1, w2, extra: ti.template()):
    """Barycentric index of refraction of a confirmed triangle/PN hit.
    Columns 6-8 of the surface ``extra`` row hold the per-corner refractive
    index (unsigned magnitude, 0 = non-PBR); columns 0-5 are
    reflectivity/roughness and 9-11 transmission.
    """
    te = f % extra.shape[0]
    return (w0 * extra[te, prim, 6] + w1 * extra[te, prim, 7]
            + w2 * extra[te, prim, 8])


@ti.func
def _corner_transmission(f, prim, w0, w1, w2, extra: ti.template()):
    """Barycentric transmission of a confirmed triangle/PN hit (columns 9-11 of
    the surface ``extra`` row; 0 = lets no light through). Independent of alpha,
    which carries coverage only -- see ``_derive_material_surface_params``.
    """
    te = f % extra.shape[0]
    return (w0 * extra[te, prim, 9] + w1 * extra[te, prim, 10]
            + w2 * extra[te, prim, 11])


@ti.func
def _tri_uv(f, prim_uv_index, w0, w1, w2, tri_uvs: ti.template()):
    """Barycentric UV coordinate of a hit on a textured triangle."""
    tu = f % tri_uvs.shape[0]
    u = (w0 * tri_uvs[tu, prim_uv_index, 0]
         + w1 * tri_uvs[tu, prim_uv_index, 2]
         + w2 * tri_uvs[tu, prim_uv_index, 4])
    v = (w0 * tri_uvs[tu, prim_uv_index, 1]
         + w1 * tri_uvs[tu, prim_uv_index, 3]
         + w2 * tri_uvs[tu, prim_uv_index, 5])
    return u, v


@ti.func
def _sample_tex_vec5(f, u, v, offset, width_i, height_i,
                     textures: ti.template()):
    """Bilinear sample of all 5 channels of a map placed at ``offset`` in the
    shared flat texel buffer (same filtering as ``_sample_texture``, but
    addressed by an explicit (offset, w, h) triplet so material and normal
    maps can share the buffer with the color maps).
    """
    width = ti.cast(width_i, ti.f32)
    height = ti.cast(height_i, ti.f32)

    px = ti.math.clamp(u * (width - 1.0), 0.0, ti.max(width - 1.0, 0.0))
    py = ti.math.clamp(v * (height - 1.0), 0.0, ti.max(height - 1.0, 0.0))

    x_floor = ti.floor(px)
    y_floor = ti.floor(py)
    xr = px - x_floor
    yr = py - y_floor

    out = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
    sum_w = 0.0
    tc = f % textures.shape[0]
    num_points = textures.shape[1]

    for corner in ti.static(range(4)):
        cx = ti.cast(x_floor + (corner % 2), ti.i32)
        cy = ti.cast(y_floor + (corner // 2), ti.i32)
        w = (xr if (corner % 2) == 1 else 1.0 - xr) * (
            yr if (corner // 2) == 1 else 1.0 - yr)

        cx = ti.math.clamp(cx, 0, ti.max(width_i - 1, 0))
        cy = ti.math.clamp(cy, 0, ti.max(height_i - 1, 0))

        abs_idx = ti.math.clamp(offset + cx * height_i + cy, 0,
                                num_points - 1)
        for ci in ti.static(range(5)):
            out[ci] += w * textures[tc, abs_idx, ci]
        sum_w += w

    return out / ti.max(sum_w, 1e-6)


@ti.func
def _flat_triangle_extra(f, prim, w0, w1, w2, tri_extra: ti.template(),
                         tri_uvs: ti.template(), tri_tex_meta: ti.template(),
                         textures: ti.template(),
                         num_colored_triangles: ti.i32):
    """(reflectivity, roughness) of a triangle hit: per-vertex barycentric
    values (``_triangle_extra``) unless the triangle carries a material map
    (meta cols 3-5) whose bitmask (col 9) marks the property texture-driven,
    in which case that property is sampled per fragment instead.
    """
    reflectivity = 0.0
    roughness = 0.0
    # A promoted constant-material triangle (see _merge_scene) carries no
    # per-vertex extra row -- its prim sits past the (shrunk) tri_extra -- so the
    # vertex read is skipped and every property comes from the material map
    # below (its bitmask covers reflectivity+roughness). For every other batch
    # tri_extra spans all prims, so this guard is always true and the result is
    # byte-identical to the plain per-vertex read.
    if prim < tri_extra.shape[1]:
        reflectivity, roughness = _triangle_extra(f, prim, w0, w1, w2, tri_extra)
    if prim >= num_colored_triangles:
        idx = prim - num_colored_triangles
        if tri_tex_meta[idx, 3] >= 0:
            flags = tri_tex_meta[idx, 9]
            u, v = _tri_uv(f, idx, w0, w1, w2, tri_uvs)
            m = _sample_tex_vec5(f, u, v, tri_tex_meta[idx, 3],
                                 tri_tex_meta[idx, 4], tri_tex_meta[idx, 5],
                                 textures)
            if (flags & 1) != 0:
                reflectivity = m[0]
            if (flags & 2) != 0:
                roughness = m[1]
    return reflectivity, roughness


@ti.func
def _flat_corner_ior_transmission(f, prim, w0, w1, w2, extra: ti.template(),
                                  tri_uvs: ti.template(),
                                  tri_tex_meta: ti.template(),
                                  textures: ti.template(),
                                  num_colored_triangles: ti.i32):
    """(IOR, transmission) of a triangle hit: per-vertex (``_corner_ior`` /
    ``_corner_transmission``) unless the material map's bitmask marks the
    property texture-driven (bit 2 / channel 2, bit 3 / channel 3). One
    fused map fetch serves both properties -- the separate per-property
    fetches sampled the same texels of the same map.
    """
    # See _flat_triangle_extra: a promoted constant-material triangle has no
    # per-vertex extra row, so these come from the material map instead; the
    # guard is a no-op (always true) for every non-promoted batch.
    ior = 1.0
    transmission = 0.0
    if prim < extra.shape[1]:
        ior = _corner_ior(f, prim, w0, w1, w2, extra)
        transmission = _corner_transmission(f, prim, w0, w1, w2, extra)
    if prim >= num_colored_triangles:
        idx = prim - num_colored_triangles
        if tri_tex_meta[idx, 3] >= 0:
            flags = tri_tex_meta[idx, 9]
            if (flags & 12) != 0:
                u, v = _tri_uv(f, idx, w0, w1, w2, tri_uvs)
                m = _sample_tex_vec5(f, u, v, tri_tex_meta[idx, 3],
                                     tri_tex_meta[idx, 4],
                                     tri_tex_meta[idx, 5], textures)
                if (flags & 4) != 0:
                    ior = m[2]
                if (flags & 8) != 0:
                    transmission = m[3]
    return ior, transmission


@ti.func
def _flat_triangle_material(f, prim, w0, w1, w2, tri_extra: ti.template(),
                            tri_uvs: ti.template(),
                            tri_tex_meta: ti.template(),
                            textures: ti.template(),
                            num_colored_triangles: ti.i32):
    """(reflectivity, roughness, IOR, transmission) of a triangle hit with a
    single material-map fetch. Exactly ``_flat_triangle_extra`` +
    ``_flat_corner_ior_transmission`` with the redundant repeat samples of
    the same map removed; in a fully constant-promoted batch every hit takes
    the map path, so the separate fetches tripled the texel traffic.
    """
    reflectivity = 0.0
    roughness = 0.0
    ior = 1.0
    transmission = 0.0
    if prim < tri_extra.shape[1]:
        reflectivity, roughness = _triangle_extra(f, prim, w0, w1, w2,
                                                  tri_extra)
        ior = _corner_ior(f, prim, w0, w1, w2, tri_extra)
        transmission = _corner_transmission(f, prim, w0, w1, w2, tri_extra)
    if prim >= num_colored_triangles:
        idx = prim - num_colored_triangles
        if tri_tex_meta[idx, 3] >= 0:
            flags = tri_tex_meta[idx, 9]
            u, v = _tri_uv(f, idx, w0, w1, w2, tri_uvs)
            m = _sample_tex_vec5(f, u, v, tri_tex_meta[idx, 3],
                                 tri_tex_meta[idx, 4], tri_tex_meta[idx, 5],
                                 textures)
            if (flags & 1) != 0:
                reflectivity = m[0]
            if (flags & 2) != 0:
                roughness = m[1]
            if (flags & 4) != 0:
                ior = m[2]
            if (flags & 8) != 0:
                transmission = m[3]
    return reflectivity, roughness, ior, transmission


@ti.func
def _flat_triangle_normal(f, prim, w0, w1, w2, tri_norm: ti.template(),
                          tri_pos: ti.template(), tri_uvs: ti.template(),
                          tri_tex_meta: ti.template(),
                          textures: ti.template(),
                          num_colored_triangles: ti.i32):
    """Shading normal of a triangle hit; when the triangle carries a
    tangent-space normal map (meta cols 6-8) the interpolated vertex normal
    is perturbed by the sampled tangent-space vector. The tangent frame is
    derived per hit from the triangle's positions and UVs (x along
    increasing u, y along increasing v, z along the smooth normal), so no
    extra per-vertex tangent array is needed.
    """
    normal = _triangle_normal(f, prim, w0, w1, w2, tri_norm, tri_pos)
    if prim >= num_colored_triangles:
        idx = prim - num_colored_triangles
        if tri_tex_meta[idx, 6] >= 0:
            u, v = _tri_uv(f, idx, w0, w1, w2, tri_uvs)
            m = _sample_tex_vec5(f, u, v, tri_tex_meta[idx, 6],
                                 tri_tex_meta[idx, 7], tri_tex_meta[idx, 8],
                                 textures)
            tn = ti.math.vec3(m[0], m[1], m[2])
            if tn.norm() > 1e-6 and normal.norm() > 1e-9:
                nb = normal.normalized()
                tp = f % tri_pos.shape[0]
                v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                                  tri_pos[tp, prim, 2])
                v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                                  tri_pos[tp, prim, 5])
                v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                                  tri_pos[tp, prim, 8])
                tu = f % tri_uvs.shape[0]
                du1 = tri_uvs[tu, idx, 2] - tri_uvs[tu, idx, 0]
                dv1 = tri_uvs[tu, idx, 3] - tri_uvs[tu, idx, 1]
                du2 = tri_uvs[tu, idx, 4] - tri_uvs[tu, idx, 0]
                dv2 = tri_uvs[tu, idx, 5] - tri_uvs[tu, idx, 1]
                det = du1 * dv2 - du2 * dv1
                if ti.abs(det) > 1e-12:
                    inv_det = 1.0 / det
                    e1 = v1 - v0
                    e2 = v2 - v0
                    tang = (e1 * dv2 - e2 * dv1) * inv_det
                    tang = tang - nb * nb.dot(tang)  # Gram-Schmidt vs normal
                    if tang.norm() > 1e-9:
                        tang = tang.normalized()
                        bit = (e2 * du1 - e1 * du2) * inv_det
                        bit = bit - nb * nb.dot(bit) - tang * tang.dot(bit)
                        if bit.norm() > 1e-9:
                            bit = bit.normalized()
                            pert = tang * tn[0] + bit * tn[1] + nb * tn[2]
                            if pert.norm() > 1e-9:
                                normal = pert.normalized()
    return normal


# ---------------------------------------------------------------------------
# "Family A+B" memory-trim variants of the flat-triangle samplers. In the trim
# layout the triangles are reordered into material-class bands ([needs-mat |
# needs-norm-only | bare]) so ``tri_norm``/``tri_mat`` are compacted PREFIXES
# (guarded by ``prim < shape[1]``), while the promotion-compacted
# ``tri_colors``/``tri_extra`` are addressed through a per-prim remap
# ``col_row[prim]`` (-1 = promoted -> its colour/material comes from the 1x1
# maps). ``tex_meta``/``uvs`` are full band-order arrays here (one row per prim),
# so texture metadata is indexed directly by ``prim`` (no ``prim-num_colored``).
# These mirror the non-trim samplers 1:1; only the indexing differs, so the trim
# path is byte-identical to the baseline. Opt-in (ALGAN_WF_MEM_TRIM); see
# scene_builder._build_mem_trim.
# ---------------------------------------------------------------------------


@ti.func
def _flat_triangle_color_trim(f, prim, w0, w1, w2, tri_colors: ti.template(),
                              col_row: ti.template(), tri_uvs: ti.template(),
                              tex_meta: ti.template(), textures: ti.template()):
    color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    alpha = 0.0
    coff = tex_meta[prim, 0]
    if coff < 0:  # no colour map -> per-vertex row (never -1 here: promoted
        cr = col_row[prim]                              # prims always get a map)
        color, alpha = _triangle_color(f, cr, w0, w1, w2, tri_colors)
    else:
        u, v = _tri_uv(f, prim, w0, w1, w2, tri_uvs)
        m = _sample_tex_vec5(f, u, v, tex_meta[prim, 0], tex_meta[prim, 1],
                             tex_meta[prim, 2], textures)
        color = ti.math.vec4(m[0], m[1], m[2], m[3])
        alpha = m[4]
    return color, alpha


@ti.func
def _flat_triangle_extra_trim(f, prim, w0, w1, w2, tri_extra: ti.template(),
                              col_row: ti.template(), tri_uvs: ti.template(),
                              tex_meta: ti.template(), textures: ti.template()):
    reflectivity = 0.0
    roughness = 0.0
    cr = col_row[prim]
    if cr >= 0:
        reflectivity, roughness = _triangle_extra(f, cr, w0, w1, w2, tri_extra)
    if tex_meta[prim, 3] >= 0:
        flags = tex_meta[prim, 9]
        u, v = _tri_uv(f, prim, w0, w1, w2, tri_uvs)
        m = _sample_tex_vec5(f, u, v, tex_meta[prim, 3], tex_meta[prim, 4],
                             tex_meta[prim, 5], textures)
        if (flags & 1) != 0:
            reflectivity = m[0]
        if (flags & 2) != 0:
            roughness = m[1]
    return reflectivity, roughness


@ti.func
def _flat_corner_ior_transmission_trim(f, prim, w0, w1, w2,
                                       tri_extra: ti.template(),
                                       col_row: ti.template(),
                                       tri_uvs: ti.template(),
                                       tex_meta: ti.template(),
                                       textures: ti.template()):
    """Trim-layout twin of ``_flat_corner_ior_transmission`` (one fused map
    fetch for both properties).
    """
    ior = 1.0
    transmission = 0.0
    cr = col_row[prim]
    if cr >= 0:
        ior = _corner_ior(f, cr, w0, w1, w2, tri_extra)
        transmission = _corner_transmission(f, cr, w0, w1, w2, tri_extra)
    if tex_meta[prim, 3] >= 0:
        flags = tex_meta[prim, 9]
        if (flags & 12) != 0:
            u, v = _tri_uv(f, prim, w0, w1, w2, tri_uvs)
            m = _sample_tex_vec5(f, u, v, tex_meta[prim, 3], tex_meta[prim, 4],
                                 tex_meta[prim, 5], textures)
            if (flags & 4) != 0:
                ior = m[2]
            if (flags & 8) != 0:
                transmission = m[3]
    return ior, transmission


@ti.func
def _flat_triangle_normal_trim(f, prim, w0, w1, w2, tri_norm: ti.template(),
                               tri_pos: ti.template(), tri_uvs: ti.template(),
                               tex_meta: ti.template(), textures: ti.template()):
    # ``tri_norm`` is the compacted needs-normal prefix; a bare prim (index past
    # the prefix) never consumes the shading normal, so return 0 for it.
    normal = ti.math.vec3(0.0, 0.0, 0.0)
    if prim < tri_norm.shape[1]:
        normal = _triangle_normal(f, prim, w0, w1, w2, tri_norm, tri_pos)
        if tex_meta[prim, 6] >= 0:
            u, v = _tri_uv(f, prim, w0, w1, w2, tri_uvs)
            m = _sample_tex_vec5(f, u, v, tex_meta[prim, 6], tex_meta[prim, 7],
                                 tex_meta[prim, 8], textures)
            tn = ti.math.vec3(m[0], m[1], m[2])
            if tn.norm() > 1e-6 and normal.norm() > 1e-9:
                nb = normal.normalized()
                tp = f % tri_pos.shape[0]
                v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                                  tri_pos[tp, prim, 2])
                v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                                  tri_pos[tp, prim, 5])
                v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                                  tri_pos[tp, prim, 8])
                tu = f % tri_uvs.shape[0]
                du1 = tri_uvs[tu, prim, 2] - tri_uvs[tu, prim, 0]
                dv1 = tri_uvs[tu, prim, 3] - tri_uvs[tu, prim, 1]
                du2 = tri_uvs[tu, prim, 4] - tri_uvs[tu, prim, 0]
                dv2 = tri_uvs[tu, prim, 5] - tri_uvs[tu, prim, 1]
                det = du1 * dv2 - du2 * dv1
                if ti.abs(det) > 1e-12:
                    inv_det = 1.0 / det
                    e1 = v1 - v0
                    e2 = v2 - v0
                    tang = (e1 * dv2 - e2 * dv1) * inv_det
                    tang = tang - nb * nb.dot(tang)
                    if tang.norm() > 1e-9:
                        tang = tang.normalized()
                        bit = (e2 * du1 - e1 * du2) * inv_det
                        bit = bit - nb * nb.dot(bit) - tang * tang.dot(bit)
                        if bit.norm() > 1e-9:
                            bit = bit.normalized()
                            pert = tang * tn[0] + bit * tn[1] + nb * tn[2]
                            if pert.norm() > 1e-9:
                                normal = pert.normalized()
    return normal


# --- mem-trim dispatch wrappers: pick trim vs baseline sampler at compile time
# (``mem_trim`` template), so the shade kernel's hit sites are a 1-line swap and
# the ``mem_trim == 0`` branch inlines to the exact baseline call. ---


@ti.func
def _tri_color_g(mem_trim: ti.template(), f, prim, w0, w1, w2,
                 tri_colors: ti.template(), col_row: ti.template(),
                 tri_uvs: ti.template(), tex_meta: ti.template(),
                 textures: ti.template(), num_colored: ti.template()):
    color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    alpha = 0.0
    if ti.static(mem_trim != 0):
        color, alpha = _flat_triangle_color_trim(
            f, prim, w0, w1, w2, tri_colors, col_row, tri_uvs, tex_meta,
            textures)
    else:
        color, alpha = _flat_triangle_color(
            f, prim, w0, w1, w2, tri_colors, tri_uvs, tex_meta, textures,
            num_colored)
    return color, alpha


@ti.func
def _tri_extra_g(mem_trim: ti.template(), f, prim, w0, w1, w2,
                 tri_extra: ti.template(), col_row: ti.template(),
                 tri_uvs: ti.template(), tex_meta: ti.template(),
                 textures: ti.template(), num_colored: ti.template()):
    reflectivity = 0.0
    roughness = 0.0
    if ti.static(mem_trim != 0):
        reflectivity, roughness = _flat_triangle_extra_trim(
            f, prim, w0, w1, w2, tri_extra, col_row, tri_uvs, tex_meta,
            textures)
    else:
        reflectivity, roughness = _flat_triangle_extra(
            f, prim, w0, w1, w2, tri_extra, tri_uvs, tex_meta, textures,
            num_colored)
    return reflectivity, roughness


@ti.func
def _tri_ior_transmission_g(mem_trim: ti.template(), f, prim, w0, w1, w2,
                            tri_extra: ti.template(), col_row: ti.template(),
                            tri_uvs: ti.template(), tex_meta: ti.template(),
                            textures: ti.template(), num_colored: ti.template()):
    ior = 1.0
    transmission = 0.0
    if ti.static(mem_trim != 0):
        ior, transmission = _flat_corner_ior_transmission_trim(
            f, prim, w0, w1, w2, tri_extra, col_row, tri_uvs, tex_meta,
            textures)
    else:
        ior, transmission = _flat_corner_ior_transmission(
            f, prim, w0, w1, w2, tri_extra, tri_uvs, tex_meta, textures,
            num_colored)
    return ior, transmission


@ti.func
def _tri_material_g(mem_trim: ti.template(), f, prim, w0, w1, w2,
                    tri_extra: ti.template(), col_row: ti.template(),
                    tri_uvs: ti.template(), tex_meta: ti.template(),
                    textures: ti.template(), num_colored: ti.template()):
    """All four material properties of a triangle hit in one call: a single
    map fetch on the baseline path (the adjacent per-property calls it
    replaces fetched the same map up to three times per hit).
    """
    reflectivity = 0.0
    roughness = 0.0
    ior = 1.0
    transmission = 0.0
    if ti.static(mem_trim != 0):
        reflectivity, roughness = _flat_triangle_extra_trim(
            f, prim, w0, w1, w2, tri_extra, col_row, tri_uvs, tex_meta,
            textures)
        ior, transmission = _flat_corner_ior_transmission_trim(
            f, prim, w0, w1, w2, tri_extra, col_row, tri_uvs, tex_meta,
            textures)
    else:
        reflectivity, roughness, ior, transmission = _flat_triangle_material(
            f, prim, w0, w1, w2, tri_extra, tri_uvs, tex_meta, textures,
            num_colored)
    return reflectivity, roughness, ior, transmission


@ti.func
def _tri_normal_g(mem_trim: ti.template(), f, prim, w0, w1, w2,
                  tri_norm: ti.template(), tri_pos: ti.template(),
                  tri_uvs: ti.template(), tex_meta: ti.template(),
                  textures: ti.template(), num_colored: ti.template()):
    normal = ti.math.vec3(0.0, 0.0, 0.0)
    if ti.static(mem_trim != 0):
        normal = _flat_triangle_normal_trim(
            f, prim, w0, w1, w2, tri_norm, tri_pos, tri_uvs, tex_meta, textures)
    else:
        normal = _flat_triangle_normal(
            f, prim, w0, w1, w2, tri_norm, tri_pos, tri_uvs, tex_meta, textures,
            num_colored)
    return normal


@ti.func
def _refract_ray(rd, n_out, ior):
    """Direction of the transmitted ray for incident unit direction ``rd``
    crossing a surface with outward unit normal ``n_out`` and index of
    refraction ``ior`` (relative to air). Snell's law, with the air<->medium
    side chosen from the sign of ``rd . n_out`` (entering when the ray opposes
    the outward normal, exiting otherwise). On total internal reflection the
    ray is mirror-reflected instead, so it always continues sensibly.
    """
    cosi = rd.dot(n_out)
    n = n_out
    eta = 1.0 / ior          # entering: air (1) -> medium (ior)
    if cosi > 0.0:           # exiting: medium (ior) -> air (1)
        n = -n_out
        eta = ior
    # ``n`` now opposes ``rd``; cos of the incidence angle is non-negative.
    cos_i = -rd.dot(n)
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    out = rd
    if sin2_t > 1.0:         # total internal reflection
        out = rd - 2.0 * rd.dot(n) * n
    else:
        cos_t = ti.sqrt(1.0 - sin2_t)
        out = eta * rd + (eta * cos_i - cos_t) * n
    return out.normalized()


@ti.func
def _relative_ior(rs_sca: ti.template(), r, ior, entering,
                  nested: ti.template()):
    """The index ratio ``_refract_ray`` wants at row ``r``'s glass interface.

    Returns n_inside/n_outside, which is exactly the reading ``_refract_ray``
    gives its ``ior`` argument in both directions -- entering ``eta = 1/rel``,
    exiting ``eta = rel`` -- so ``_refract_ray`` is unchanged.

    **The stack supplies only the OUTSIDE index; the inside always comes from
    the hit's own ``ior``.** That is what the feature is for -- today's bug is
    that every interface assumes air outside -- and taking the inside locally
    is the more honest of the two readings. ``ior`` is barycentrically
    interpolated per hit (``_corner_ior``), so the value at THIS point is what
    the material says here; the alternative, reading the inside off the stack,
    would import the entry hit's interpolation into the exit hit. On a
    constant-index material those differ in their last bits (a constant 1.5
    arrives as ``1.5*(w0+w1+w2)``), and on a spatially varying one they differ
    outright.

    ``entering`` says which side the ray crosses from, and is decided by the
    GEOMETRIC face normal, not the shading normal ``_refract_ray`` picks its
    own side from. Inside-ness is a geometric fact; an interpolated shading
    normal is not, and on a diced surface it tips past the silhouette at
    grazing angles, where deciding with it would call an EXIT an entry. The
    two disagreeing costs nothing until something actually encloses the hit,
    because the stack only supplies the outside. This assumes consistent
    outward winding -- the same assumption one-sided shading already makes
    (``Mob.two_sided``).

    With the gate off, or with nothing enclosing this interface, returns
    ``ior`` ITSELF -- no division at all -- so an un-nested interface is
    identical to the pre-feature kernel by construction rather than by IEEE
    argument. That is not the same as an un-nested SCENE being identical: a
    ray grazing a shared edge can be classified as entering a solid it never
    left, which puts something on the stack and does change those pixels. See
    the measurement in DESIGN_mesh_identity_open.md §H, and
    ``benchmarks/_nested_ior_ab.py``, which bounds it.

    The denominator is clamped to a small positive floor so a corrupt material
    index cannot produce an inf/NaN direction.

    Entering past ``IOR_STACK_DEPTH`` still reads s[N-1] as its outside --
    correct at this interface -- and the matching exits pop back onto a stack
    that is still right, so only the interfaces beyond the cap are wrong (see
    ``_write_ior_stack`` for the deliberate deviation from the design doc's
    overflow rule).
    """
    rel = ior
    if ti.static(nested != 0):
        # Depth of the medium ENCLOSING this interface: entering, that is
        # everything the ray is already inside; exiting, everything it is
        # inside except the medium it is leaving.
        d = ti.cast(rs_sca[r, _SCA_IOR_DEPTH] + 0.5, ti.i32)
        if entering != 1:
            d = d - 1
        if d > 0:
            n_out = rs_sca[r, _SCA_IOR_BASE
                           + ti.min(d, IOR_STACK_DEPTH) - 1]
            rel = ior / ti.max(n_out, 1e-6)
    return rel


@ti.func
def _write_ior_stack(rs_sca: ti.template(), src, dst, ior, entering,
                     refracting, nested: ti.template()):
    """Fill child row ``dst``'s stack columns from parent row ``src``'s.

    ``refracting == 0`` copies depth and all N entries: a reflection and a
    coverage pass-through stay in the same medium. Refracting while entering
    pushes the hit's interpolated ``ior`` (depth d+1, entry d stored when
    d < N); refracting while exiting pops (depth max(d-1, 0)).

    DEVIATION from DESIGN_mesh_identity_open.md §H's overflow rule ("do not
    push, do not bump"): the depth counter keeps counting past N and only the
    first N entries are stored. The doc's rule shifts every subsequent pop by
    one, so a nest deeper than N would render wrong at every interface OUTSIDE
    the overflow as well as at the overflowing one; counting past N is wrong
    only at the interfaces past N -- entering at depth == N still reads
    s[N-1] as its outside, which is CORRECT, and the matching exits come back
    onto a stack that is still right. Reads clamp their index to [0, N-1]
    (see ``_relative_ior``), so nothing goes out of bounds.

    Compiles out entirely unless the nested-IOR gate is on.
    """
    if ti.static(nested != 0):
        d = ti.cast(rs_sca[src, _SCA_IOR_DEPTH] + 0.5, ti.i32)
        nd = d
        pushed = 0
        if refracting == 1:
            if entering == 1:
                nd = d + 1
                pushed = 1 if d < IOR_STACK_DEPTH else 0
            else:
                nd = ti.max(d - 1, 0)
        rs_sca[dst, _SCA_IOR_DEPTH] = ti.cast(nd, ti.f32)
        for k in ti.static(range(IOR_STACK_DEPTH)):
            rs_sca[dst, _SCA_IOR_BASE + k] = rs_sca[src, _SCA_IOR_BASE + k]
        if pushed == 1:
            rs_sca[dst, _SCA_IOR_BASE + d] = ior


@ti.func
def _offset_transmitted_origin(hit_point, out_dir, face_n, shade_n):
    """Move a transmitted ray robustly onto the outgoing side of a surface.

    Offsetting along ``out_dir`` is not reliable at grazing angles: its normal
    component can remain inside the intersection tolerance, so the continuation
    immediately re-hits the triangle it just left (or its neighbour across a
    shared edge). Those self-hits show up as mesh-aligned black/green speckles
    on transmissive surfaces. Prefer the geometric normal, whose sign is
    selected from the outgoing direction; fall back to the shading normal and
    finally the direction for degenerate faces. A small forward displacement is
    included as well: at a shared convex edge, crossing only one face plane can
    still leave the origin outside the neighbouring face, while forward motion
    follows the refracted ray into the actual solid interior.
    """
    n = face_n
    if n.dot(n) <= 1e-18:
        n = shade_n
    if n.dot(n) <= 1e-18:
        n = out_dir
    n = n.normalized()
    if n.dot(out_dir) < 0.0:
        n = -n
    return hit_point + (n + out_dir) * (10.0 * MIN_HIT_DISTANCE)


# ---------------------------------------------------------------------------
# ``rs_alloc`` slot layout. One tiny i32 ndarray carries the tile's shared-pool
# allocator AND its truncation counters, rather than each taking a kernel
# argument: the shade kernels are already near Taichi's 64-argument ceiling,
# and an ndarray's SHAPE is not part of a kernel's compiled signature, so
# widening this costs no recompile. The host zeroes it per tile attempt and
# reads it back once, after the attempt is accepted.
# ---------------------------------------------------------------------------
#: Next free slot in the shared continuation pool; starts at ``num_primary``
#: and keeps counting past capacity, so an overflow reports how much it wanted.
ALLOC_NEXT = 0
#: Raised to 1 by any reservation that did not fit.
ALLOC_OVERFLOW = 1
#: Rays retired by the ``MAX_SURFACES_PER_RAY`` compositing ceiling with
#: transport still to carry (``rendering.raytracing.truncation``).
ALLOC_TRUNC_SURFACES = 2
#: Number of ``rs_alloc`` words the host must allocate and zero.
ALLOC_WIDTH = 3


@ti.func
def _reserve_continuation_slot(rs_alloc: ti.template(), capacity):
    """Append one continuation to the tile-wide shared ray pool.

    ``rs_alloc[0]`` is the next free slot and starts at ``num_primary``. When
    the append exceeds ``capacity``, no state is written and ``rs_alloc[1]`` is
    atomically raised. On a *splitting* batch the host discards that tile
    attempt and retries it with fewer primaries, so an overflow can never
    silently remove light transport. A batch at ``pool_ratio == 1`` has no
    spare slots to retry into: there the host counts the failed reservations
    (``rs_alloc[0]`` minus the capacity) as dropped continuations instead.
    """
    slot = ti.atomic_add(rs_alloc[0], 1)
    valid = slot < capacity
    if not valid:
        ti.atomic_max(rs_alloc[1], 1)
    return slot, valid


# ---------------------------------------------------------------------------
# Ray-continuation (scatter) contract + built-in default. Shared by the
# monolithic ``wavefront_shade`` and the sorted ``wf_shade_event`` (which
# imports these), so both express bouncing through the same funcs. See the
# scatter contract in ``shading_taichi`` for the argument/return meaning.
# ---------------------------------------------------------------------------


# How much of a material's specular lobe a SINGLE traced ray is allowed to
# carry, as a function of roughness. See ``_mirror_share`` below; the constant
# is the GGX ``alpha`` at which half the reflection is still traced.
#
# 0.15 is where the lobe stops being resolvable as an image: GGX at
# roughness 0.15 puts its median microfacet at ``atan(alpha) = 1.29`` degrees,
# a reflected deflection of ~2.6 degrees, which is already ~20 px of blur
# across a PREVIEW frame. Below it a mirror ray is a fair stand-in for the
# lobe; above it the lobe is a wide integral one ray cannot estimate, and the
# reflected image it draws instead is a sharp minified alias -- the bright
# hard-edged chunks this replaced.
#
# A module constant for the same reason as ``_GLOSSY_MIN_ROUGHNESS``: it is
# baked into the compiled kernel and is not a template argument, so an env knob
# would let the offline cache serve a kernel built for a different value.
_MIRROR_SHARE_ALPHA = 0.15 * 0.15
_MIRROR_SHARE_A2 = _MIRROR_SHARE_ALPHA * _MIRROR_SHARE_ALPHA


@ti.func
def _mirror_share(roughness):
    """Fraction of the specular lobe one mirror ray may carry, in [0, 1].

    A continuation is a single ray in a single direction, so it can only stand
    for a reflection whose lobe is narrow enough that every direction in it
    sees roughly the same thing. Spending the material's whole Fresnel energy
    on one direction regardless of roughness is what made a
    ``MeshStandardMaterial(roughness=0.35)`` draw a razor-sharp, full-strength
    mirror image: minified 10x or more by the reflection's own geometry, that
    image aliases into hard bright chunks that no amount of sub-pixel sampling
    of the SAME direction can fix (``ANALYTIC_AA_SECONDARY_SAMPLES`` moves the
    ray's origin, not its direction, so on a flat face all four taps agree).

    So the ray carries only the share of the lobe that sits within a cone it
    can honestly represent -- the GGX CDF mass inside half-angle
    ``atan(_MIRROR_SHARE_ALPHA)``::

        F(theta) = tan^2 / (alpha^2 + tan^2),  alpha = roughness^2

    which is ``1`` for a mirror (byte-identical to tracing it outright), ~0.83
    at roughness 0.10, 0.5 at 0.15 and ~0.03 at 0.35.

    The rest is not lost. The caller shades ``alpha * (1 - R - trans_share)``
    locally, so whatever the mirror ray gives up goes back to the material's
    own shading -- which already carries a roughness-correct GGX highlight from
    the direct lights and the ambient/environment term that stands in for the
    reflected surroundings. A rough metal therefore reads as a rough metal
    rather than as a mirror with an aliased picture painted on it.

    This is the stand-in for a properly sampled glossy lobe, not a rival to it:
    with ``GLOSSY_REFLECTION`` on, the raster resolve's continuations spread
    over the real GGX lobe (``raster_taichi._glossy_reflect``) and skip this
    entirely, because the lobe is then being integrated rather than
    approximated by its peak.
    """
    a = roughness * roughness
    return _MIRROR_SHARE_A2 / (_MIRROR_SHARE_A2 + a * a)


@ti.func
def _env_brdf_approx(f0, n_dot_v, roughness):
    """Directional albedo of a GGX specular lobe -- the DFG half of the
    split-sum approximation (Karis 2013), in ``[0, 1]`` per channel.

    ``integral L(l) f(l,v) dl`` factors into a prefiltered radiance and a BRDF
    integral that depends on nothing but ``(n.v, roughness)``. This is that
    second factor: the fraction of arriving light the lobe reflects, summed
    over every direction in it. It is EXACT energy where ``_mirror_share`` is
    a throttle -- the throttle asks "how much of this lobe may one ray stand
    for", which is a question about sampling, and answers it by throwing the
    rest at the material's own shading (for a metal, the ambient fill: a rough
    metal reflecting 4.7% of what it should, measured in the renderer audit
    ss4.5). The split-sum asks "how much does this lobe reflect", which is a
    question about the material, and the answer costs no rays at all.

    The analytic fit rather than the usual 2-D LUT: a LUT is another texture to
    pack, upload and address inside the shade kernel for what is a four-term
    polynomial (Karis 2014, "Mobile"), whose error against the numerically
    integrated table is under 1% over the whole ``(n.v, roughness)`` square.

    It degenerates to Schlick as roughness goes to zero -- ``f0`` at normal
    incidence, 1 at grazing -- which is what lets the mirror path below
    ``_GLOSSY_MIN_ROUGHNESS`` keep using ``_material_reflectance`` unchanged
    while the glossy path above it uses this: the two agree across the
    threshold instead of stepping.
    """
    nv = ti.math.clamp(n_dot_v, 0.0, 1.0)
    r = ti.math.clamp(roughness, 0.0, 1.0)
    cx = r * -1.0 + 1.0
    cy = r * -0.0275 + 0.0425
    cz = r * -0.572 + 1.04
    cw = r * 0.022 - 0.04
    # exp2(-9.28 * nv), spelled in ti.exp: Taichi has no exp2 (checked, 1.7.4),
    # and 9.28 * ln 2 = 6.4324058.
    a004 = ti.min(cx * cx, ti.exp(-6.4324058 * nv)) * cx + cy
    term_a = -1.04 * a004 + cz
    term_b = 1.04 * a004 + cw
    return ti.math.clamp(f0 * term_a + term_b, 0.0, 1.0)


@ti.func
def _material_env_brdf(rd, normal, metalness, packed_ior, albedo, roughness):
    """Split-sum directional albedo of a Three.js-style material's specular
    lobe -- the drop-in for ``_material_reflectance``'s ``R`` on the
    PREFILTERED glossy route (``DESIGN_glossy_prefilter.md`` ss2.1).

    ``_material_reflectance`` evaluates Schlick at the incident angle, which is
    the reflectance of the *mirror direction*; a lobe wide enough to need
    prefiltering does not reflect that. This integrates the lobe instead, from
    the same F0: ``mix(dielectric_f0, albedo, metalness)``, the identical blend
    that function performs before Schlick's tail, so a mirror and a rough metal
    of the same material still describe one material.

    ``metalness < 0`` is the same legacy/unlit sentinel, and returns zero: an
    unlit material has no specular lobe to integrate. An IOR at or below 1 is
    index-matched with the air around it, so its dielectric lobe vanishes and
    only the metal share survives -- the same explicit gate, for the same
    reason (Schlick cannot express that limit; any f0 still reflects fully at
    grazing).

    The two lobes are integrated SEPARATELY and blended after, where
    ``_material_reflectance`` may blend F0 first. ``E`` is affine in F0
    (``F0*A + B``) so the two are algebraically identical wherever both lobes
    exist -- but the index-matched gate has to remove the dielectric lobe
    ENTIRELY, and a blended F0 of zero still carries the ``B`` bias term, i.e.
    a grazing sheen on a surface that has no interface to reflect from.

    TRANSMISSION IS NOT AN ARGUMENT, deliberately. The caller applies this only
    to the opaque reflective branch: glass keeps Schlick and the side-aware
    total-internal-reflection logic ``_material_reflectance`` documents, whose
    correctness depends on evaluating a single interface at a single angle.
    """
    result = ti.math.vec3(0.0, 0.0, 0.0)
    if metalness >= 0.0:
        m = ti.math.clamp(metalness, 0.0, 1.0)
        ior = ti.abs(packed_ior)
        n = normal.normalized()
        nv = ti.abs(rd.dot(n))
        e_diel = ti.math.vec3(0.0, 0.0, 0.0)
        if ior > 1.0 + 1e-4:
            r0 = (1.0 - ior) / (1.0 + ior)
            f0d = r0 * r0
            e_diel = _env_brdf_approx(
                ti.math.vec3(f0d, f0d, f0d), nv, roughness)
        e_metal = _env_brdf_approx(
            ti.math.clamp(albedo, 0.0, 1.0), nv, roughness)
        result = e_diel * (1.0 - m) + e_metal * m
    return result


@ti.func
def _material_reflectance(rd, normal, metalness, packed_ior, albedo,
                          transmission):
    """Per-channel (vec3) Schlick reflectance of a Three.js-style material,
    plus the scalar dielectric pass fraction that gates transmission.

    ``metalness < 0`` is the internal sentinel for legacy/unlit materials that
    have no PBR specular lobe.  ``packed_ior`` is an unsigned magnitude (abs
    guards any legacy sign packing).  An IOR at or below 1 means the medium is
    index-matched with the surrounding air: there is no dielectric interface,
    so the dielectric lobe vanishes entirely.  Schlick's form cannot express
    that limit itself -- any f0 still reflects fully at grazing -- so it is an
    explicit gate rather than an f0 of zero.

    Transport is full-colour: the metal lobe's F0 is the surface ``albedo``
    (conductor tint, whitening to 1 at grazing -- same model as the Monte
    Carlo megakernel's coloured throughput), the dielectric lobe stays
    achromatic.  A white metal reduces exactly to the old scalar lobe (R = 1).
    Blending the two lobes after Schlick is algebraically identical to
    blending f0 first (R is linear in f0).

    **Which side of the interface the ray is on matters, and only for a
    transmissive material.**  Schlick's approximation is written for a ray
    arriving from the *thin* side; a ray already inside the glass reflects far
    more than the same incident angle suggests, and past the critical angle it
    reflects everything.  KHR_materials_volume states the three cases
    normatively: entering, evaluate Schlick at the incident angle; leaving
    without total internal reflection, evaluate it at the angle on the AIR side
    (Snell's partner of the incident one); leaving beyond the critical angle,
    ``F = 1``.  For glass (ior 1.5, critical angle 41.8 deg) an internal ray at
    40 deg has a true reflectance of 0.245, which the air-side Schlick
    reproduces to three digits and the inside-angle one puts at 0.041 -- so
    without this the light leaving a solid is split six-to-one the wrong way,
    and at total internal reflection it leaks through a surface that should be
    a perfect mirror.

    ``transmission`` gates that side test, deliberately.  The renderer does not
    track which medium a ray is in, so "the ray is on the far side of the
    surface" is inferred from the sign of ``rd . normal`` -- sound for a closed
    transmissive solid, wrong for a back-facing hit on an ordinary opaque
    surface (an open mesh seen from behind), which is not inside anything.
    Gating on transmission keeps every non-transmissive material bit-for-bit
    on the path it took before.

    Returns ``(R, diel_pass)``.  ``diel_pass = (1-m) * (1-r_diel)`` is the
    fraction of incident light that enters the dielectric interior -- the
    only share that can transmit.  It must NOT be derived from ``1 - R``: a
    coloured metal has ``R < 1`` in its absorbed channels, and that absorbed
    share would then leak through transmissive surfaces as if dielectric
    (with scalar transport ``m = 1`` forced ``R = 1``, which hid this).
    Total internal reflection arrives here as ``r_diel = 1``, hence
    ``diel_pass = 0``: the transmitted branch is given no energy at all rather
    than being handed the mirror direction with the transmitted weight (which
    also tinted a perfectly achromatic Fresnel reflection by the glass colour).
    """
    result = ti.math.vec3(0.0, 0.0, 0.0)
    diel_pass = 1.0
    if metalness >= 0.0:
        m = ti.math.clamp(metalness, 0.0, 1.0)
        ior = ti.abs(packed_ior)
        n = normal.normalized()
        cos_n = rd.dot(n)  # signed: < 0 arriving from outside, > 0 leaving
        cosi = ti.math.clamp(ti.abs(cos_n), 0.0, 1.0)
        tail = ti.pow(1.0 - cosi, 5.0)
        r_diel = 0.0
        if ior > 1.0 + 1e-4:
            r0 = (1.0 - ior) / (1.0 + ior)
            dielectric_f0 = r0 * r0
            # The cosine Schlick is evaluated at, and whether there is one at
            # all. Unchanged (the incident cosine, no TIR) for everything that
            # does not transmit and for every ray arriving from outside.
            cos_s = cosi
            total_internal = False
            if (transmission > 1e-4) and (cos_n > 0.0):
                sin2_t = ior * ior * (1.0 - cosi * cosi)
                if sin2_t > 1.0:
                    total_internal = True
                else:
                    cos_s = ti.sqrt(1.0 - sin2_t)
            if total_internal:
                r_diel = 1.0
            else:
                tail_s = ti.pow(1.0 - cos_s, 5.0)
                r_diel = dielectric_f0 + (1.0 - dielectric_f0) * tail_s
        f0_metal = ti.math.clamp(albedo, 0.0, 1.0)
        r_metal = f0_metal + (1.0 - f0_metal) * tail
        result = r_diel * (1.0 - m) + m * r_metal
        diel_pass = (1.0 - m) * (1.0 - r_diel)
    return ti.math.clamp(result, 0.0, 1.0), diel_pass


@ti.func
def _scatter_impl(rd, n_interp, face_n, hit_point, shaded, albedo, alpha,
                  reflectivity, ior, transmission, params: ti.template(),
                  f, prim, bounces_left, refraction: ti.template(),
                  pane: ti.template()):
    """Built-in continuation derived from the material's PBR properties.

    Transport is full-colour: the branch weights (``pass_w``, ``refl_w``,
    ``trans_w``) are vec3 throughput multipliers. ``albedo`` (vec3, the raw
    surface colour before lighting) tints the metal share of the Fresnel lobe
    (coloured mirrors) and the transmitted share (coloured glass); the
    dielectric reflection stays achromatic. Branch decisions (which
    continuation is heavier, minimum-weight culls) reduce a colour weight to
    its maximum component, the same convention as the Monte Carlo megakernel's
    coloured throughput -- for a white surface every component is equal and
    all decisions match the old scalar transport exactly.

    Shared body of :func:`default_scatter` (``pane`` 0, solid geometry) and
    :func:`circuit_scatter` (``pane`` 1, zero-thickness planar circuits). The
    two exist as separate wrappers because the scatter contract is a fixed
    injection signature shared with user scatters (see ``shading_taichi``), so
    the geometry distinction cannot be an extra runtime argument.

    The historical ``reflectivity`` argument carries packed ``metalness``
    (negative means no PBR material); ``ior`` is an unsigned magnitude feeding
    dielectric F0 and Snell; ``transmission`` alone says how much light passes
    through. All of them, like roughness, come from the material and nothing
    else -- there is no renderer-side control (see
    ``_derive_material_surface_params``).

    ``alpha`` and ``transmission`` are independent: alpha is coverage (how much
    of the surface is there -- a fade, a spawn), transmission is how much light
    the part that IS there lets through. Folding them together (as this once
    did) makes clear glass indistinguishable from an absent object.

    Transmissive geometry refracts if it is solid (Snell, entering then exiting
    the medium); a circuit is a thin pane, whose entry and exit interfaces
    coincide, so its net bend is zero and its transmitted ray simply continues
    as the pass-through. Both split off the same Fresnel energy; only the
    transmitted direction differs.

    ``roughness`` does not reach the bounce HERE: this continuation is a single
    ray, and one deterministic sample of a glossy lobe is not a blur -- it is a
    mirror pointing the wrong way. Blurring needs several rays, so it lives
    where several already exist: the raster resolve's primary hit spreads its
    ``ANALYTIC_AA_SECONDARY_SAMPLES`` continuations over the material's GGX lobe
    (``raster_taichi._glossy_reflect``, ``GLOSSY_REFLECTION``). The split happens
    once, at that primary hit, so the deeper bounces this func drives stay
    specular-perfect and the cost stays N x the secondary traversal rather than
    N^depth. The Monte Carlo megakernel jitters every bounce, by a wider
    normal-perturbation lobe rather than GGX.

    Roughness does not fade the bounce here either. The callers that spawn a
    PRIMARY continuation scale ``R`` by ``_mirror_share(roughness)`` so a rough
    material cannot draw a full-strength mirror image; this func is the deeper
    bounce, and its signature is the user-facing scatter contract, so it takes
    neither roughness nor the fade. A mirror reflecting a rough metal shows
    that metal's reflection unfaded as well as unblurred -- one scope, stated
    in DESIGN_analytic_aa.md ss20.7.

    A partially covering reflective surface has two continuations: the mirror
    reflection (``alpha * R``) and whatever shows through behind it
    (``1 - alpha``). Every PBR material has a non-zero Fresnel ``R`` (>= 4% at
    normal incidence), so tracing only the reflection would discard everything
    behind any semi-transparent PBR surface -- an opaque black silhouette.

    Both are traced when the split pool is compiled in (``refraction != 0``,
    which ``scene_builder``'s ``has_refl_transparent`` turns on for exactly
    these scenes): the *reflection* goes to the split slot and the
    pass-through continues as the primary ray. That is the mirror image of the
    glass split below, and deliberate -- the pass-through is the depth-layer
    walk, so keeping it primary preserves ``t_prev``/``layer_prev`` and spends
    no bounce. Without the pool only the heavier continuation is traced, so
    the dropped term is always the smaller one. Opaque surfaces have nothing
    behind them (``1 - alpha == 0``) and always reflect: mirrors are unchanged.
    """
    alpha = ti.math.clamp(alpha, 0.0, 1.0)
    T = ti.math.clamp(transmission, 0.0, 1.0)
    tint = ti.math.clamp(albedo, 0.0, 1.0)
    normal = n_interp.normalized()
    R, diel_pass = _material_reflectance(rd, normal, reflectivity, ior,
                                         albedo, T)
    if bounces_left <= 0:
        # Out of bounces: no reflected ray. The transmitted share stays gated
        # by ``diel_pass`` (the metal share never transmits), so zeroing R
        # only stops the bounce; a fully metallic surface shades instead.
        R = ti.math.vec3(0.0, 0.0, 0.0)

    # Transmission alone says whether light passes through; solid geometry
    # refracts (is_glass), a zero-thickness circuit does not bend (is_pane).
    # Reflection uses the metal-blended Fresnel ``R`` above; transmission is
    # gated by ``diel_pass = (1-m)(1-R_diel)`` -- only the non-metallic share
    # that enters the dielectric interior transmits (Three.js semantics), so
    # a fully metallic surface stays a mirror at any transmission and IOR.
    is_glass = False
    is_pane = False
    glass_ior = ior
    if ti.static(refraction != 0):
        if (T > 1e-4) and (bounces_left > 0) \
                and (glass_ior > 1.0 + 1e-4):
            if ti.static(pane != 0):
                is_pane = True
            else:
                is_glass = True

    # The four shares a hit splits into, summing to 1 per channel:
    #   alpha * (1 - R - trans_share)  shaded here      (contrib)
    #   alpha * R                      reflected        (refl_energy)
    #   alpha * trans_share            transmitted      (trans_energy,
    #                                                    albedo-tinted when it
    #                                                    actually transmits)
    #   1 - alpha                      missed entirely  (cover_pass)
    # where trans_share = (1-m)(1-R_diel) * T: only the share entering the
    # dielectric interior can transmit. It is NOT alpha*(1-R)*T -- a coloured
    # metal has R < 1 in its absorbed channels, and deriving transmission
    # from 1-R would leak that absorbed light through the surface (scalar
    # transport hid this because m = 1 forced R = 1). For achromatic
    # surfaces 1 - R - trans_share == (1-R)(1-T) exactly.
    # Coverage and transmission are independent: alpha says how much of the
    # surface is there, T how much light the part that IS there passes.
    # When the transmitted share gets no ray of its own (index-matched
    # ior <= 1 where nothing bends, refraction pool absent, or bounces
    # exhausted) it continues unbent as part of the primary pass-through
    # below instead of being dropped -- dropping it rendered transmissive
    # surfaces as opaque black.
    one3 = ti.math.vec3(1.0, 1.0, 1.0)
    trans_share = diel_pass * T
    # The accumulator's 4th lane (glow) has no colour channel of its own; it
    # takes the max-component reduction of the reflectance (exact when the
    # components are equal, i.e. everywhere the old scalar transport reached).
    r_glow = ti.max(R[0], ti.max(R[1], R[2]))
    share = alpha * (one3 - R - trans_share)
    contrib = ti.math.vec4(share[0], share[1], share[2],
                           alpha * (1.0 - r_glow - trans_share)) * shaded
    refl_energy = alpha * R
    refl_max = ti.max(refl_energy[0],
                      ti.max(refl_energy[1], refl_energy[2]))
    trans_energy = alpha * trans_share
    cover_pass = 1.0 - alpha
    cover3 = ti.math.vec3(cover_pass, cover_pass, cover_pass)

    # A reflective surface with something behind it has two continuations (the
    # reflection and the pass-through). With the split pool compiled in both
    # are traced -- see the branch below.
    split_refl = False
    if ti.static(refraction != 0):
        if (refl_max > MIN_ALPHA) and (cover_pass > MIN_ALPHA) \
                and (bounces_left > 0):
            split_refl = True
    zero3 = ti.math.vec3(0.0, 0.0, 0.0)
    pass_w = zero3
    refl_w = zero3
    trans_w = zero3
    refl_dir = zero3
    refl_orig = zero3
    trans_dir = zero3
    trans_orig = zero3
    if is_glass:
        # Refracted branch takes the split slot; the primary carries whichever
        # of the reflection / coverage-miss is heavier (three continuations, two
        # rays -- so the dropped term is always the smaller one). At full
        # coverage the miss is empty and the primary is always the reflection.
        rdt = _refract_ray(rd, normal, glass_ior)
        trans_dir = rdt
        trans_orig = _offset_transmitted_origin(
            hit_point, rdt, face_n, normal)
        trans_w = trans_energy * tint
        if (refl_max > MIN_ALPHA) and (refl_max >= cover_pass):
            refl_dir, nref = _reflect_frame(rd, normal, face_n)
            refl_orig = hit_point + nref * (10.0 * MIN_HIT_DISTANCE)
            refl_w = refl_energy
        else:
            pass_w = cover3
    elif is_pane:
        # Thin pane: entry and exit interfaces coincide, so the transmitted ray
        # is unbent and merges into the pass-through (the depth-layer walk,
        # which spends no bounce) along with the coverage-miss. Only the
        # reflection needs a ray of its own, so it takes the split slot.
        trans_dir, nref = _reflect_frame(rd, normal, face_n)
        trans_orig = hit_point + nref * (10.0 * MIN_HIT_DISTANCE)
        trans_w = refl_energy
        pass_w = cover3 + trans_energy * tint
    elif split_refl or ((refl_max > MIN_ALPHA)
                        and (refl_max >= cover_pass)):
        rdir, nref = _reflect_frame(rd, normal, face_n)
        rorig = hit_point + nref * (10.0 * MIN_HIT_DISTANCE)
        if split_refl:
            # Reflection into the split slot; the pass-through stays the
            # primary ray. It continues the depth-layer walk (no bounce
            # spent, ``t_prev``/``layer_prev`` preserved by the caller).
            trans_dir = rdir
            trans_orig = rorig
            trans_w = refl_energy
            pass_w = cover3 + trans_energy * tint
        else:
            # No split pool compiled in: trace only the heavier continuation,
            # so the dropped term is always the smaller of the two.
            refl_dir = rdir
            refl_orig = rorig
            refl_w = refl_energy
    else:
        pass_w = cover3 + trans_energy * tint
    return (contrib, pass_w, refl_orig, refl_dir, refl_w,
            trans_orig, trans_dir, trans_w)


@ti.func
def default_scatter(rd, n_interp, face_n, hit_point, shaded, albedo, alpha,
                    reflectivity, ior, transmission, params: ti.template(),
                    f, prim, bounces_left, refraction: ti.template()):
    """Built-in scatter for solid geometry (triangles / PN patches): a
    transmissive surface refracts. This is the signature the scatter contract
    fixes (see ``shading_taichi``), shared with user scatters and injected as
    the sorted pipeline's ``scatter_fn``. See :func:`_scatter_impl`.
    """
    return _scatter_impl(rd, n_interp, face_n, hit_point, shaded, albedo,
                         alpha, reflectivity, ior, transmission, params, f,
                         prim, bounces_left, refraction, 0)


@ti.func
def circuit_scatter(rd, n_interp, face_n, hit_point, shaded, albedo, alpha,
                    reflectivity, ior, transmission, params: ti.template(),
                    f, prim, bounces_left, refraction: ti.template()):
    """Built-in scatter for bezier circuits: a transmissive circuit is a thin
    pane, so it transmits unbent rather than refracting. See
    :func:`_scatter_impl`.
    """
    return _scatter_impl(rd, n_interp, face_n, hit_point, shaded, albedo,
                         alpha, reflectivity, ior, transmission, params, f,
                         prim, bounces_left, refraction, 1)


@ti.func
def _run_frag_scatter(frag_scatters: ti.template(), pid_arr: ti.template(),
                      f, prim, rd, n_interp, face_n, hit_point, shaded,
                      albedo, alpha,
                      reflectivity, ior, transmission, params: ti.template(),
                      bounces_left, refraction: ti.template()):
    """Per-primitive ray-continuation dispatch for the monolithic shade kernel:
    pick the material's scatter func by pipeline id (``pid_arr[f, prim]``) and
    return its 8-tuple. Built-in materials and user pipelines without a custom
    scatter use :func:`default_scatter`; a user pid whose pipeline supplied a
    scatter uses it. The pid switch mirrors ``_run_frag_pipeline``; ``None``
    entries of ``frag_scatters`` compile out -- a slot is None both for a
    scatterless user pipeline and for one this batch does not use (the tuple is
    narrowed to the batch, see ``fragment_shaders.build_frag_scatters``), and
    either way the default scatter is what that id would have got.
    """
    pid = pid_arr[f % pid_arr.shape[0], prim]
    (contrib, pass_w, refl_orig, refl_dir, refl_w,
     trans_orig, trans_dir, trans_w) = default_scatter(
        rd, n_interp, face_n, hit_point, shaded, albedo, alpha, reflectivity,
        ior, transmission, params, f, prim, bounces_left, refraction)
    for pi in ti.static(range(len(frag_scatters))):
        # ``bool(func) is True`` / ``bool(None) is False`` -- avoids an ``is
        # not`` comparison node, which Taichi's AST transformer rejects even
        # inside ``ti.static``. A scatterless pipeline's None entry compiles
        # its branch (and the None "call") out.
        if ti.static(bool(frag_scatters[pi])):
            if pid == _USER_PIPELINE_BASE + pi:
                (contrib, pass_w, refl_orig, refl_dir, refl_w,
                 trans_orig, trans_dir, trans_w) = frag_scatters[pi](
                    rd, n_interp, face_n, hit_point, shaded, albedo, alpha,
                    reflectivity, ior, transmission, params, f, prim,
                    bounces_left, refraction)
    return (contrib, pass_w, refl_orig, refl_dir, refl_w,
            trans_orig, trans_dir, trans_w)


@ti.kernel
def wf_composite(
        time_start: int, width: int, height: int, transparent: int,
        ray_offset: int,
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
        tonemapping: ti.template(), tonemap_exposure: ti.f32,
        out: ti.types.ndarray()):
    """Composite each ray's premultiplied accumulator over the pre-filled
    background. State is indexed tile-locally by ``r``; the global ray is
    ``ray_offset + r``.
    """
    pixels_per_frame = width * height
    num_rays = rs_acc.shape[0]
    for r in range(num_rays):
        g = ray_offset + r
        f_rel = g // pixels_per_frame
        p = g - f_rel * pixels_per_frame
        # Colour transport: leftover throughput is per-channel (columns 0/5/6);
        # the glow lane and coverage alpha take its mean.
        weight = ti.math.vec4(rs_sca[r, 0], rs_sca[r, 5], rs_sca[r, 6], 0.0)
        weight[3] = ti.max(weight[0], ti.max(weight[1], weight[2]))
        csum = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        for ci in ti.static(range(4)):
            bg = ti.cast(out[f_rel, p, ci], ti.f32)
            csum[ci] = rs_acc[r, ci] * 255.0 + weight[ci] * bg
        color_final = finalize_pixel_color(csum, 1.0, tonemapping, tonemap_exposure)
        for ci in ti.static(range(4)):
            if ti.static(tonemapping == 3):
                out[f_rel, p, ci] = color_final[ci]
            else:
                out[f_rel, p, ci] = ti.cast(color_final[ci], ti.u8)
        if transparent != 0:
            bg_a = ti.cast(out[f_rel, p, 4], ti.f32)
            val = (1.0 - weight[3]) * 255.0 + weight[3] * bg_a
            out[f_rel, p, 4] = ti.cast(
                ti.math.clamp(val + 0.5, 0.0, 255.0), ti.u8)


@ti.kernel
def wf_composite_aa(
        time_start: int, width: int, height: int, transparent: int,
        ray_offset: int,
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
        out: ti.types.ndarray(), aa_accum: ti.types.ndarray()):
    """Like ``wf_composite`` but accumulates into a float buffer for in-place
    AA averaging. Each call adds one sub-pixel sample's composited value;
    ``wf_finalize_aa`` averages after all ``aa^2`` passes.
    """
    pixels_per_frame = width * height
    num_rays = rs_acc.shape[0]
    for r in range(num_rays):
        g = ray_offset + r
        f_rel = g // pixels_per_frame
        p = g - f_rel * pixels_per_frame
        idx = f_rel * pixels_per_frame + p
        weight = ti.math.vec4(rs_sca[r, 0], rs_sca[r, 5], rs_sca[r, 6], 0.0)
        weight[3] = ti.max(weight[0], ti.max(weight[1], weight[2]))
        for ci in ti.static(range(4)):
            bg = ti.cast(out[f_rel, p, ci], ti.f32)
            aa_accum[idx, ci] += rs_acc[r, ci] * 255.0 + weight[ci] * bg
        if transparent != 0:
            bg_a = ti.cast(out[f_rel, p, 4], ti.f32)
            aa_accum[idx, 4] += (1.0 - weight[3]) * 255.0 + weight[3] * bg_a


@ti.kernel
def wf_composite_accum(
        time_start: int, width: int, height: int, transparent: int,
        ray_offset: int,
        pix_accum: ti.types.ndarray(),
        tonemapping: ti.template(), tonemap_exposure: ti.f32,
        empty: ti.template(), covered: ti.template(),
        covered_idx: ti.types.ndarray(), num_covered: int,
        out: ti.types.ndarray()):
    """Composite the general path's per-pixel accumulator over the pre-filled
    background. Mirrors ``wf_composite`` arithmetic exactly, but reads the shared
    ``pix_accum`` (premultiplied colour cols 0-3 + summed leftover/background
    weight cols 4-6, deposited by every terminating ray) instead of one ray slot
    -- so a pixel whose ray split into reflected + refracted branches sums both.
    For a non-split pixel ``pix_accum[r] == (acc, weight)`` of its lone ray, so
    the result is byte-identical to ``wf_composite``. Indexed by local pixel
    ``r``; the global cell is ``ray_offset + r``.

    ``empty`` (compile-time): the raster front-end pre-fills ``pix_accum`` with
    the retired-empty constant ``[0,0,0,0, 1,1,1]`` and, when a whole tile has
    no candidate geometry, leaves it untouched (RASTER_EMPTY_SKIP). For that
    tile the accumulator is *known* to be that constant, so the 28-byte-per-
    pixel ``pix_accum`` read -- the kernel's dominant memory traffic -- is
    dropped: ``acc == 0`` and ``weight == 1`` collapse the blend to the bare
    background ``finalize(bg)``, byte-for-byte what the full read produces.

    ``covered`` (compile-time): under post-process tonemapping (``tonemapping
    == 3``) the composite is a pure linear blend, so an empty pixel's result
    ``finalize(bg) == bg`` is exactly what the background pre-fill already
    wrote -- a no-op. The host therefore passes the same compact covered
    list the resolve used and this loop runs one thread per covered pixel
    (``r = covered_idx[t]``); the untouched empty pixels keep their
    pre-filled background. Only valid with ``tonemapping == 3`` (in-kernel
    tonemap would owe every empty pixel ``tonemap(bg) != bg``).
    """
    pixels_per_frame = width * height
    num_primary = pix_accum.shape[0]
    loop_n = num_primary
    if ti.static(covered):
        loop_n = num_covered
    for t in range(loop_n):
        r = t
        if ti.static(covered):
            r = covered_idx[t]
        g = ray_offset + r
        f_rel = g // pixels_per_frame
        p = g - f_rel * pixels_per_frame
        weight = ti.math.vec4(1.0, 1.0, 1.0, 1.0)
        if ti.static(not empty):
            weight = ti.math.vec4(pix_accum[r, 4], pix_accum[r, 5],
                                  pix_accum[r, 6], 0.0)
            weight[3] = ti.max(weight[0], ti.max(weight[1], weight[2]))
        csum = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        for ci in ti.static(range(4)):
            bg = ti.cast(out[f_rel, p, ci], ti.f32)
            if ti.static(empty):
                csum[ci] = bg
            else:
                csum[ci] = pix_accum[r, ci] * 255.0 + weight[ci] * bg
        color_final = finalize_pixel_color(csum, 1.0, tonemapping, tonemap_exposure)
        for ci in ti.static(range(4)):
            if ti.static(tonemapping == 3):
                out[f_rel, p, ci] = color_final[ci]
            else:
                out[f_rel, p, ci] = ti.cast(color_final[ci], ti.u8)
        if transparent != 0:
            bg_a = ti.cast(out[f_rel, p, 4], ti.f32)
            val = (1.0 - weight[3]) * 255.0 + weight[3] * bg_a
            out[f_rel, p, 4] = ti.cast(
                ti.math.clamp(val + 0.5, 0.0, 255.0), ti.u8)


@ti.kernel
def wf_composite_accum_sparse(
        time_start: int, width: int, height: int, transparent: int,
        ray_offset: int, pixel_idx: ti.types.ndarray(),
        pix_accum: ti.types.ndarray(),
        tonemapping: ti.template(), tonemap_exposure: ti.f32,
        out: ti.types.ndarray()):
    """Composite compact accumulator rows at their real local pixels.

    This is the covered-pixel counterpart of :func:`wf_composite_accum`:
    empty pixels never appear in ``pixel_idx``. Row ``r`` of ``pix_accum``
    belongs to local pixel ``pixel_idx[r]``. Under post-process tonemapping
    (``tonemapping == 3``, the historical only mode) the untouched pixels'
    prefilled background IS their final value; under an in-kernel tonemap
    the sheet route pairs this with :func:`wf_finalize_uncovered`, which
    owes every untouched pixel ``finalize(bg)``.
    """
    pixels_per_frame = width * height
    for r in range(pix_accum.shape[0]):
        local_pixel = pixel_idx[r]
        g = ray_offset + local_pixel
        f_rel = g // pixels_per_frame
        p = g - f_rel * pixels_per_frame
        weight = ti.math.vec4(
            pix_accum[r, 4], pix_accum[r, 5], pix_accum[r, 6], 0.0)
        weight[3] = ti.max(weight[0], ti.max(weight[1], weight[2]))
        csum = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        for ci in ti.static(range(4)):
            bg = ti.cast(out[f_rel, p, ci], ti.f32)
            csum[ci] = pix_accum[r, ci] * 255.0 + weight[ci] * bg
        color_final = finalize_pixel_color(
            csum, 1.0, tonemapping, tonemap_exposure)
        for ci in ti.static(range(4)):
            if ti.static(tonemapping == 3):
                out[f_rel, p, ci] = color_final[ci]
            else:
                out[f_rel, p, ci] = ti.cast(color_final[ci], ti.u8)
        if transparent != 0:
            bg_a = ti.cast(out[f_rel, p, 4], ti.f32)
            val = (1.0 - weight[3]) * 255.0 + weight[3] * bg_a
            out[f_rel, p, 4] = ti.cast(
                ti.math.clamp(val + 0.5, 0.0, 255.0), ti.u8)


@ti.kernel
def wf_finalize_uncovered(
        num_pixels: int, width: int, height: int,
        covered_mask: ti.types.ndarray(),
        tonemapping: ti.template(), tonemap_exposure: ti.f32,
        out: ti.types.ndarray()):
    """``finalize(bg)`` for every pixel the sparse resolve did not touch.

    Under an in-kernel tonemap an untouched pixel owes ``tonemap(bg) != bg``,
    which the covered-only composite cannot supply. The mask is the sparse
    coverage scattered to per-pixel bytes; covered pixels were finalized by
    :func:`wf_composite_accum_sparse` (which read their RAW prefilled
    background first, so ordering between the two kernels is free). Only
    launched when ``tonemapping != 3`` — the linear mode's untouched pixels
    are already final by identity.
    """
    pixels_per_frame = width * height
    for g in range(num_pixels):
        if covered_mask[g] != 0:
            continue
        f_rel = g // pixels_per_frame
        p = g - f_rel * pixels_per_frame
        csum = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        for ci in ti.static(range(4)):
            csum[ci] = ti.cast(out[f_rel, p, ci], ti.f32)
        color_final = finalize_pixel_color(
            csum, 1.0, tonemapping, tonemap_exposure)
        for ci in ti.static(range(4)):
            if ti.static(tonemapping == 3):
                out[f_rel, p, ci] = color_final[ci]
            else:
                out[f_rel, p, ci] = ti.cast(color_final[ci], ti.u8)


@ti.kernel
def wf_composite_accum_aa(
        time_start: int, width: int, height: int, transparent: int,
        ray_offset: int,
        pix_accum: ti.types.ndarray(), out: ti.types.ndarray(),
        aa_accum: ti.types.ndarray()):
    """Like ``wf_composite_accum`` but accumulates into a float buffer for
    in-place AA averaging.
    """
    pixels_per_frame = width * height
    num_primary = pix_accum.shape[0]
    for r in range(num_primary):
        g = ray_offset + r
        f_rel = g // pixels_per_frame
        p = g - f_rel * pixels_per_frame
        idx = f_rel * pixels_per_frame + p
        weight = ti.math.vec4(pix_accum[r, 4], pix_accum[r, 5],
                              pix_accum[r, 6], 0.0)
        weight[3] = ti.max(weight[0], ti.max(weight[1], weight[2]))
        for ci in ti.static(range(4)):
            bg = ti.cast(out[f_rel, p, ci], ti.f32)
            aa_accum[idx, ci] += pix_accum[r, ci] * 255.0 + weight[ci] * bg
        if transparent != 0:
            bg_a = ti.cast(out[f_rel, p, 4], ti.f32)
            aa_accum[idx, 4] += (1.0 - weight[3]) * 255.0 + weight[3] * bg_a


@ti.kernel
def wf_finalize_aa(
        width: int, height: int, transparent: int,
        inv_samples: float,
        tonemapping: ti.template(), tonemap_exposure: ti.f32,
        aa_accum: ti.types.ndarray(), out: ti.types.ndarray()):
    """Average the AA float accumulator and write the final uint8 output.
    Called once after all ``aa^2`` sub-pixel passes have been accumulated by
    ``wf_composite_aa`` or ``wf_composite_accum_aa``.
    """
    pixels_per_frame = width * height
    num_pixels = aa_accum.shape[0]
    for idx in range(num_pixels):
        f_rel = idx // pixels_per_frame
        p = idx - f_rel * pixels_per_frame
        csum = ti.math.vec4(aa_accum[idx, 0], aa_accum[idx, 1], aa_accum[idx, 2], aa_accum[idx, 3])
        color_final = finalize_pixel_color(csum, inv_samples, tonemapping, tonemap_exposure)
        for ci in ti.static(range(4)):
            if ti.static(tonemapping == 3):
                out[f_rel, p, ci] = color_final[ci]
            else:
                out[f_rel, p, ci] = ti.cast(color_final[ci], ti.u8)
        if transparent != 0:
            out[f_rel, p, 4] = ti.cast(
                ti.math.clamp(aa_accum[idx, 4] * inv_samples + 0.5,
                              0.0, 255.0), ti.u8)


# ---------------------------------------------------------------------------
# General (triangle + PN patch + bezier circuit) wavefront kernels.
#
# The traverse stage reuses the general ``_collect_hits`` (all three BVHs +
# the Matrix Pencil PN solver) and the shade stage drains the gathered hits
# front-to-back per geometry type. State carries an extra scalar, base_dist
# (rs_sca column 4), used by the bezier screen-constant border width and
# accumulated across mirror bounces.
# ---------------------------------------------------------------------------


@ti.kernel
def wavefront_generate_rays(
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_screen_w: float, half_screen_h: float, max_bounces: int,
        ray_offset: int, num_primary: int, jitter_x: float, jitter_y: float,
        near_clip: ti.f32, write_const: int,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
        rs_int: ti.types.ndarray(),
        rs_pix: ti.types.ndarray(), pix_accum: ti.types.ndarray(),
        rs_alloc: ti.types.ndarray()):
    """Initialise primaries and the tile-wide continuation pool.

    Slots ``[0, num_primary)`` are one-per-pixel primary rays. Every remaining
    slot belongs to one shared append-only pool, so a pixel with a deep ray tree
    can consume capacity left unused by simple pixels. ``rs_alloc[0]`` starts at
    ``num_primary`` and ``rs_alloc[1]`` records overflow. The host discards and
    retries an overflowing tile with fewer primaries; branches are never silently
    dropped. Each ray records its target local pixel in ``rs_pix`` and commits
    premultiplied colour/background weight into that row of ``pix_accum`` when it
    terminates.
    """
    pixels_per_frame = width * height
    num_rays = rs_ro.shape[0]
    for r in range(num_rays):
        if r < num_primary:
            g = ray_offset + r
            f_rel = g // pixels_per_frame
            p = g - f_rel * pixels_per_frame
            f = time_start + f_rel
            py = p // width
            px = p - py * width
            ro, rd = _generate_ray(f, px, py, jitter_x, jitter_y,
                                   half_screen_w, half_screen_h,
                                   cam_origin, screen_point,
                                   pixel_basis_x, pixel_basis_y)
            t_near = 0.0
            if near_clip > 0.0:
                # Near plane: advance the ray origin to the plane at
                # ``near_clip`` along the camera's forward axis, so geometry
                # closer than the plane is skipped (planar, like Three.js).
                # The skipped distance seeds base_dist, keeping distances
                # (far clip, screen-space border widths) camera-relative.
                fwd = (ti.math.vec3(screen_point[f, 0], screen_point[f, 1],
                                    screen_point[f, 2])
                       - ti.math.vec3(cam_origin[f, 0], cam_origin[f, 1],
                                      cam_origin[f, 2])).normalized()
                t_near = near_clip / ti.max(rd.dot(fwd), 1e-6)
                ro = ro + rd * t_near
            for k in ti.static(range(3)):
                rs_ro[r, k] = ro[k]
                rs_rd[r, k] = rd[k]
            # rs_acc and pix_accum are all-zero at generation, and (when
            # ``write_const == 0``) rs_sca / rs_int are the constant primary
            # init rows below. All of that is hoisted to coalesced
            # ``torch.zero_()`` / broadcast fills in the host tile driver (see
            # ``_run_wavefront_tiles``): a contiguous fill moves far more
            # bytes/s than these strided per-ray stores through the AoS
            # [ray, channel] layout, so keeping them here needlessly slows the
            # memory-bound generate kernel. ``write_const != 0`` restores the
            # in-kernel init for the paths the host fill can't cover (a shared
            # split pool, or a near clip whose base_dist varies per ray).
            # Byte-identical either way -- same constants.
            if write_const != 0:
                rs_sca[r, 0] = 1.0     # weight (red channel; green/blue in 5/6)
                rs_sca[r, 1] = 0.0     # t_prev
                rs_sca[r, 2] = 1e30    # layer_prev
                rs_sca[r, 3] = -1e30   # seam_t
                rs_sca[r, 4] = t_near  # base_dist
                rs_sca[r, 5] = 1.0     # weight green
                rs_sca[r, 6] = 1.0     # weight blue
                rs_int[r, 0] = max_bounces
                rs_int[r, 1] = 0
                rs_int[r, 2] = _ACTIVE
                rs_int[r, 3] = 0
            rs_pix[r] = r
        else:
            # Free shared-pool slot: inactive until a continuation append
            # reserves it. Only reached when a split pool is allocated
            # (pool > num_primary), which always uses ``write_const != 0``.
            if write_const != 0:
                rs_int[r, 2] = _DONE
        if r == 0:
            rs_alloc[0] = num_primary
            rs_alloc[1] = 0


@ti.kernel
def wavefront_traverse(
        active: ti.types.ndarray(), num_active: int,
        t_nodes: NODE_ARG, t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int, tri_pos: ti.types.ndarray(),
        b_nodes: NODE_ARG, b_node_miss: ti.types.ndarray(),
        b_leaf_prim: ti.types.ndarray(), b_leaf_tspan: ti.types.ndarray(),
        b_first_leaf: int, circuit_meta: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        # Opaque-only STBVHs used by the optional mixed-scene prepass. They
        # retain the normal primitive index space and are ignored when the
        # compile-time feature is disabled.
        ot_nodes: NODE_ARG, ot_node_miss: ti.types.ndarray(),
        ot_leaf_prim: ti.types.ndarray(), ot_leaf_tspan: ti.types.ndarray(),
        ot_first_leaf: int,
        ob_nodes: NODE_ARG, ob_node_miss: ti.types.ndarray(),
        ob_leaf_prim: ti.types.ndarray(), ob_leaf_tspan: ti.types.ndarray(),
        ob_first_leaf: int,
        pixel_world_scale: ti.types.ndarray(),
        layer_offset_triangles: float,
        refit: ti.template(),
        has_tri: ti.template(), has_bez: ti.template(),
        opaque_closest: ti.template(),
        opaque_prepass: ti.template(),
        time_start: int, width: int, height: int, ray_offset: int,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_sca: ti.types.ndarray(), rs_int: ti.types.ndarray(),
        rs_kt: ti.types.ndarray(), rs_kl: ti.types.ndarray(),
        rs_ka: ti.types.ndarray(), rs_kb: ti.types.ndarray(),
        rs_kp: ti.types.ndarray(),
        rs_kf: ti.types.ndarray(),
        rs_pix: ti.types.ndarray(),
        # Fused primary-ray generation (compile-time; see WF_GEN_FUSED). When
        # ``gen_first`` is on this launch IS the tile's first iteration on a
        # split-free, near-clip-free render: every ray is primary and owns
        # pixel ``r``, so the ray is generated here (writing only ro/rd back)
        # and the standalone wavefront_generate_rays pass is skipped. The
        # remaining initial state is implicit in the matching ``first_iter``
        # shade. ``gen_meta`` packs [jitter_x, jitter_y, half_w, half_h].
        gen_first: ti.template(),
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        gen_meta: ti.types.ndarray()):
    """Gather KBUF nearest hits across all three BVHs for each active ray
    (reuses the unchanged general ``_collect_hits``, Matrix Pencil solver
    included). The frame is taken from the ray's *pixel* (``rs_pix``), not its
    slot index -- a spawned (split) ray lives in a spare slot whose index is not
    its pixel; the global cell is ``ray_offset + rs_pix[r]``.
    """
    pixels_per_frame = width * height
    for i in range(num_active):
        r = active[i]
        ro = ti.math.vec3(0.0, 0.0, 0.0)
        rd = ti.math.vec3(0.0, 0.0, 0.0)
        t_prev = 0.0
        layer_prev = 1e30
        base_dist = 0.0
        f = 0
        px = 0
        py = 0
        if ti.static(gen_first != 0):
            g = ray_offset + r
            f_rel = g // pixels_per_frame
            p = g - f_rel * pixels_per_frame
            f = time_start + f_rel
            py = p // width
            px = p - py * width
            ro, rd = _generate_ray(f, px, py, gen_meta[0], gen_meta[1],
                                   gen_meta[2], gen_meta[3],
                                   cam_origin, screen_point,
                                   pixel_basis_x, pixel_basis_y)
            # Persist for the shade stage + later K-buffer refills; the other
            # initial state (t_prev = 0, layer_prev = 1e30, base_dist = 0,
            # pix = r) stays implicit.
            for k in ti.static(range(3)):
                rs_ro[r, k] = ro[k]
                rs_rd[r, k] = rd[k]
        else:
            ro = ti.math.vec3(rs_ro[r, 0], rs_ro[r, 1], rs_ro[r, 2])
            rd = ti.math.vec3(rs_rd[r, 0], rs_rd[r, 1], rs_rd[r, 2])
            t_prev = rs_sca[r, 1]
            layer_prev = rs_sca[r, 2]
            base_dist = rs_sca[r, 4]
            g = ray_offset + rs_pix[r]
            f_rel = g // pixels_per_frame
            p = g - f_rel * pixels_per_frame
            f = time_start + f_rel
            py = p // width
            px = p - py * width
        inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                              _safe_inverse(rd[2]))
        ff = ti.cast(f, ti.f32)
        # Distances handed to pixel_size_per_t are slant ranges along THIS
        # pixel's primary ray, but pixel_world_scale is per unit perpendicular
        # depth (see _axis_cos). The cosine is a property of the pixel, not of
        # the current ray, so it is rebuilt from the pixel: a continuation ray
        # carries a bounce direction for which the camera axis means nothing,
        # yet the primary segment already folded into its base_dist still
        # needs converting.
        pro, prd = _generate_ray(f, px, py, 0.5, 0.5, gen_meta[2], gen_meta[3],
                                 cam_origin, screen_point,
                                 pixel_basis_x, pixel_basis_y)
        pixel_size_per_t = pixel_world_scale[f] * _axis_cos(f, pro, prd,
                                                            screen_point)

        kb_t = ti.Vector([0.0] * KBUF)
        kb_layer = ti.Vector([0.0] * KBUF)
        kb_prim = ti.Vector([0] * KBUF)
        kb_flags = ti.Vector([0] * KBUF)
        kb_a = ti.Vector([0.0] * KBUF)
        kb_b = ti.Vector([0.0] * KBUF)
        num_hits = 0
        if ti.static(opaque_closest):
            (found, t_hit, hit_layer, hit_prim, hit_type, hit_a, hit_b,
             hit_border, edge_hit) = _nearest_surface_g(
                refit, has_tri, has_bez,
                ro, rd, inv_rd, f, ff, t_prev, layer_prev, 1e30,
                pixel_size_per_t, base_dist, layer_offset_triangles,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                t_first_leaf, tri_pos,
                b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                b_first_leaf, circuit_meta, edges_2d, edge_accel,
                # Camera rays carry no source identity: (-1, _, 0) compiles
                # the identity-aware acceptance floor out entirely. The
                # forwarded tri_pos is never read.
                -1, -1, 0.0, 0.0, tri_pos, 0,
                # Not a shadow ray, so a non-casting primitive stays
                # visible to it and the leaf test compiles out.
                0)
            num_hits = found
            if found != 0:
                kb_t[0] = t_hit
                kb_layer[0] = hit_layer
                kb_prim[0] = hit_prim
                kb_flags[0] = hit_type | (edge_hit << 2) | (hit_border << 3)
                kb_a[0] = hit_a
                kb_b[0] = hit_b
        else:
            initial_opq_t = 1e30
            initial_opq_layer = -1e30
            if ti.static(opaque_prepass):
                (opq_found, initial_opq_t, initial_opq_layer, opq_prim,
                 opq_type, opq_a, opq_b, opq_border, opq_edge) = \
                    _nearest_surface_g(
                        refit, has_tri, has_bez,
                        ro, rd, inv_rd, f, ff, t_prev, layer_prev, 1e30,
                        pixel_size_per_t, base_dist, layer_offset_triangles,
                        ot_nodes, ot_node_miss, ot_leaf_prim,
                        ot_leaf_tspan, ot_first_leaf, tri_pos,
                        ob_nodes, ob_node_miss, ob_leaf_prim,
                        ob_leaf_tspan, ob_first_leaf, circuit_meta,
                        edges_2d, edge_accel,
                        -1, -1, 0.0, 0.0, tri_pos, 0,
                        # Not a shadow ray, so a non-casting primitive stays
                        # visible to it and the leaf test compiles out.
                        0)
                if opq_found == 0:
                    initial_opq_t = 1e30
                    initial_opq_layer = -1e30
            num_hits = _collect_hits(
                refit, ro, rd, inv_rd, f, ff, t_prev, layer_prev,
                pixel_size_per_t, base_dist, layer_offset_triangles,
                kb_t, kb_layer, kb_prim, kb_flags, kb_a, kb_b,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
                tri_pos,
                b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan, b_first_leaf,
                circuit_meta, edges_2d, edge_accel, has_tri, has_bez,
                initial_opq_t, initial_opq_layer,
                -1, -1, 0.0, 0.0, tri_pos, 0,
                # Not a shadow ray, so a non-casting primitive stays
                # visible to it and the leaf test compiles out.
                0)
        rs_int[r, 3] = num_hits
        # num_hits == 0 leaves the ray _ACTIVE (not _DONE) so wavefront_shade
        # commits its accumulated colour + leftover (background) throughput to
        # the per-pixel accumulator before retiring it -- a split branch's
        # background contribution must be summed, not dropped.
        if num_hits > 0:
            for q in ti.static(range(KBUF)):
                rs_kt[r, q] = kb_t[q]
                rs_kl[r, q] = kb_layer[q]
                rs_kp[r, q] = kb_prim[q]
                rs_kf[r, q] = kb_flags[q]
                rs_ka[r, q] = kb_a[q]
                rs_kb[r, q] = kb_b[q]


@ti.kernel
def wavefront_traverse_events(
        active: ti.types.ndarray(), num_active: int,
        t_nodes: NODE_ARG, t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int, tri_pos: ti.types.ndarray(),
        b_nodes: NODE_ARG, b_node_miss: ti.types.ndarray(),
        b_leaf_prim: ti.types.ndarray(), b_leaf_tspan: ti.types.ndarray(),
        b_first_leaf: int, circuit_meta: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        # Opaque-only STBVHs used by the optional mixed-scene prepass. They
        # retain the normal primitive index space and are ignored when the
        # compile-time feature is disabled.
        ot_nodes: NODE_ARG, ot_node_miss: ti.types.ndarray(),
        ot_leaf_prim: ti.types.ndarray(), ot_leaf_tspan: ti.types.ndarray(),
        ot_first_leaf: int,
        ob_nodes: NODE_ARG, ob_node_miss: ti.types.ndarray(),
        ob_leaf_prim: ti.types.ndarray(), ob_leaf_tspan: ti.types.ndarray(),
        ob_first_leaf: int,
        pixel_world_scale: ti.types.ndarray(),
        layer_offset_triangles: float,
        refit: ti.template(),
        has_tri: ti.template(), has_bez: ti.template(),
        opaque_closest: ti.template(),
        opaque_prepass: ti.template(),
        time_start: int, width: int, height: int, ray_offset: int,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_sca: ti.types.ndarray(), rs_int: ti.types.ndarray(),
        hit_f: ti.types.ndarray(), hit_i: ti.types.ndarray(),
        rs_pix: ti.types.ndarray(),
        # Fused primary-ray generation (compile-time; see WF_GEN_FUSED). When
        # ``gen_first`` is on this launch IS the tile's first iteration on a
        # split-free, near-clip-free render: every ray is primary and owns
        # pixel ``r``, so the ray is generated here (writing only ro/rd back)
        # and the standalone wavefront_generate_rays pass is skipped. The
        # remaining initial state is implicit in the matching ``first_iter``
        # shade. ``gen_meta`` packs [jitter_x, jitter_y, half_w, half_h].
        gen_first: ti.template(),
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        gen_meta: ti.types.ndarray()):
    """Gather KBUF nearest hits into a transient compact event batch.

    This reuses the general ``_collect_hits`` path, including the Matrix Pencil
    PN solver. Events are indexed by active-queue ordinal and consumed by the
    immediately following shade launch. The frame is taken from the ray's
    *pixel* (``rs_pix``), not its slot index: a split ray lives in a spare slot
    whose index is not its pixel; the global cell is
    ``ray_offset + rs_pix[r]``.
    """
    pixels_per_frame = width * height
    for i in range(num_active):
        r = active[i]
        ro = ti.math.vec3(0.0, 0.0, 0.0)
        rd = ti.math.vec3(0.0, 0.0, 0.0)
        t_prev = 0.0
        layer_prev = 1e30
        base_dist = 0.0
        f = 0
        px = 0
        py = 0
        if ti.static(gen_first != 0):
            g = ray_offset + r
            f_rel = g // pixels_per_frame
            p = g - f_rel * pixels_per_frame
            f = time_start + f_rel
            py = p // width
            px = p - py * width
            ro, rd = _generate_ray(f, px, py, gen_meta[0], gen_meta[1],
                                   gen_meta[2], gen_meta[3],
                                   cam_origin, screen_point,
                                   pixel_basis_x, pixel_basis_y)
            # Persist for the shade stage + later K-buffer refills; the other
            # initial state (t_prev = 0, layer_prev = 1e30, base_dist = 0,
            # pix = r) stays implicit.
            for k in ti.static(range(3)):
                rs_ro[r, k] = ro[k]
                rs_rd[r, k] = rd[k]
        else:
            ro = ti.math.vec3(rs_ro[r, 0], rs_ro[r, 1], rs_ro[r, 2])
            rd = ti.math.vec3(rs_rd[r, 0], rs_rd[r, 1], rs_rd[r, 2])
            t_prev = rs_sca[r, 1]
            layer_prev = rs_sca[r, 2]
            base_dist = rs_sca[r, 4]
            g = ray_offset + rs_pix[r]
            f_rel = g // pixels_per_frame
            p = g - f_rel * pixels_per_frame
            f = time_start + f_rel
            py = p // width
            px = p - py * width
        inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                              _safe_inverse(rd[2]))
        ff = ti.cast(f, ti.f32)
        # Distances handed to pixel_size_per_t are slant ranges along THIS
        # pixel's primary ray, but pixel_world_scale is per unit perpendicular
        # depth (see _axis_cos). The cosine is a property of the pixel, not of
        # the current ray, so it is rebuilt from the pixel: a continuation ray
        # carries a bounce direction for which the camera axis means nothing,
        # yet the primary segment already folded into its base_dist still
        # needs converting.
        pro, prd = _generate_ray(f, px, py, 0.5, 0.5, gen_meta[2], gen_meta[3],
                                 cam_origin, screen_point,
                                 pixel_basis_x, pixel_basis_y)
        pixel_size_per_t = pixel_world_scale[f] * _axis_cos(f, pro, prd,
                                                            screen_point)

        kb_t = ti.Vector([0.0] * KBUF)
        kb_layer = ti.Vector([0.0] * KBUF)
        kb_prim = ti.Vector([0] * KBUF)
        kb_flags = ti.Vector([0] * KBUF)
        kb_a = ti.Vector([0.0] * KBUF)
        kb_b = ti.Vector([0.0] * KBUF)
        num_hits = 0
        if ti.static(opaque_closest):
            (found, t_hit, hit_layer, hit_prim, hit_type, hit_a, hit_b,
             hit_border, edge_hit) = _nearest_surface_g(
                refit, has_tri, has_bez,
                ro, rd, inv_rd, f, ff, t_prev, layer_prev, 1e30,
                pixel_size_per_t, base_dist, layer_offset_triangles,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                t_first_leaf, tri_pos,
                b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                b_first_leaf, circuit_meta, edges_2d, edge_accel,
                # No source identity on camera rays: (-1, _, 0) compiles the
                # identity-aware acceptance floor out entirely; the forwarded
                # tri_pos is never read.
                -1, -1, 0.0, 0.0, tri_pos, 0,
                # Not a shadow ray, so a non-casting primitive stays
                # visible to it and the leaf test compiles out.
                0)
            num_hits = found
            if found != 0:
                kb_t[0] = t_hit
                kb_layer[0] = hit_layer
                kb_prim[0] = hit_prim
                kb_flags[0] = hit_type | (edge_hit << 2) | (hit_border << 3)
                kb_a[0] = hit_a
                kb_b[0] = hit_b
        else:
            initial_opq_t = 1e30
            initial_opq_layer = -1e30
            if ti.static(opaque_prepass):
                (opq_found, initial_opq_t, initial_opq_layer, opq_prim,
                 opq_type, opq_a, opq_b, opq_border, opq_edge) = \
                    _nearest_surface_g(
                        refit, has_tri, has_bez,
                        ro, rd, inv_rd, f, ff, t_prev, layer_prev, 1e30,
                        pixel_size_per_t, base_dist, layer_offset_triangles,
                        ot_nodes, ot_node_miss, ot_leaf_prim,
                        ot_leaf_tspan, ot_first_leaf, tri_pos,
                        ob_nodes, ob_node_miss, ob_leaf_prim,
                        ob_leaf_tspan, ob_first_leaf, circuit_meta,
                        edges_2d, edge_accel,
                        -1, -1, 0.0, 0.0, tri_pos, 0,
                        # Not a shadow ray, so a non-casting primitive stays
                        # visible to it and the leaf test compiles out.
                        0)
                if opq_found == 0:
                    initial_opq_t = 1e30
                    initial_opq_layer = -1e30
            num_hits = _collect_hits(
                refit, ro, rd, inv_rd, f, ff, t_prev, layer_prev,
                pixel_size_per_t, base_dist, layer_offset_triangles,
                kb_t, kb_layer, kb_prim, kb_flags, kb_a, kb_b,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
                tri_pos,
                b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan, b_first_leaf,
                circuit_meta, edges_2d, edge_accel, has_tri, has_bez,
                initial_opq_t, initial_opq_layer,
                -1, -1, 0.0, 0.0, tri_pos, 0,
                # Not a shadow ray, so a non-casting primitive stays
                # visible to it and the leaf test compiles out.
                0)
        rs_int[r, 3] = num_hits
        if num_hits > 0:
            # Surface events are indexed by compacted active-queue ordinal,
            # not by the sparse ray-pool slot.  The host releases this exact
            # [num_active, KBUF] batch immediately after shade consumes it.
            for q in ti.static(range(KBUF)):
                hit_f[i, q, 0] = kb_t[q]
                hit_f[i, q, 1] = kb_layer[q]
                hit_f[i, q, 2] = kb_a[q]
                hit_f[i, q, 3] = kb_b[q]
                hit_i[i, q, 0] = kb_prim[q]
                hit_i[i, q, 1] = kb_flags[q]


@ti.kernel
def wavefront_shadow(
        active: ti.types.ndarray(), num_active: int,
        t_nodes: NODE_ARG, t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int,
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_colors: ti.types.ndarray(), tri_uvs: ti.types.ndarray(),
        tri_tex_meta: ti.types.ndarray(), textures: ti.types.ndarray(),
        tri_extra: ti.types.ndarray(),
        num_colored_triangles: ti.i32,
        b_nodes: NODE_ARG, b_node_miss: ti.types.ndarray(),
        b_leaf_prim: ti.types.ndarray(), b_leaf_tspan: ti.types.ndarray(),
        b_first_leaf: int,
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(),
        layer_offset_triangles: float,
        refit: ti.template(),
        # Shadow-terminator gate (rt_settings.shadow_terminator_mode()), same
        # semantics as raster_shadow_trace's.
        shadow_term: ti.template(),
        has_tri: ti.template(), has_bez: ti.template(),
        light_pos: ti.types.ndarray(), num_lights: int,
        time_start: int, width: int, height: int, ray_offset: int,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_sca: ti.types.ndarray(), rs_int: ti.types.ndarray(),
        hit_f: ti.types.ndarray(), hit_i: ti.types.ndarray(),
        rs_pix: ti.types.ndarray(), rs_vis: ti.types.ndarray()):
    """Legacy deferred binary shadow stage for the general wavefront (currently
    unused: the tracer always compiles ``wavefront_shade`` with
    ``deferred_shadows == 0`` -- the split measured slower than inline
    shadows; kept for a future occupancy-bound workload): for each active ray,
    precompute per-(K-buffer hit, light) occlusion into a packed int32 (bit
    ``q * MAX_SHADOW_LIGHTS + li``). Run between traverse and shade so the
    shade kernel reads visibility bits instead of inlining the heavy
    ``_shadow_occluded`` -> ``_nearest_surface_g`` -> PN-solver call graph
    (register-pressure relief -> higher shade-kernel occupancy). The per-hit
    shadow geometry mirrors ``wavefront_shade``'s inline block. Before shadows
    accumulated opacity, the bits drove byte-identical shading. Because
    ``_collect_hits`` stops gathering at the first opaque hit, the K-buffer
    holds (almost) exactly the hits shade consumes, so few bits are computed
    and never read.

    Opacity-weighted shadows no longer fit in these bits, so this stage must be
    converted to a floating-point visibility buffer before it can be revived.

    HOST CONTRACT if this kernel is ever revived: like ``hit_f``/``hit_i``,
    ``rs_vis`` is indexed by *active-queue ordinal* (``rs_vis[i]``, not the
    sparse pool slot), so the host must allocate it with ``num_active``
    elements per iteration and launch this kernel inside the same temporary
    arena scope as the surface-event batch, between traverse and shade. The
    tracer's current 1-element ``rs_vis`` placeholder is only valid while
    ``wavefront_shade`` is compiled with ``deferred_shadows == 0``.
    """
    pixels_per_frame = width * height
    for i in range(num_active):
        r = active[i]
        num_hits = rs_int[r, 3]
        bits = 0
        if num_hits > 0:
            pix = rs_pix[r]
            f = time_start + (ray_offset + pix) // pixels_per_frame
            ff = ti.cast(f, ti.f32)
            ro = ti.math.vec3(rs_ro[r, 0], rs_ro[r, 1], rs_ro[r, 2])
            rd = ti.math.vec3(rs_rd[r, 0], rs_rd[r, 1], rs_rd[r, 2])
            base_dist = rs_sca[r, 4]
            pixel_size_per_t = pixel_world_scale[f]
            tl = f % light_pos.shape[0]
            for q in ti.static(range(KBUF)):
                if q < num_hits:
                    prim = hit_i[i, q, 0]
                    if prim >= 0:
                        htype = hit_i[i, q, 1] & 3
                        if htype == 1:
                            a = hit_f[i, q, 2]
                            b = hit_f[i, q, 3]
                            t_hit = hit_f[i, q, 0]
                            snrm = _flat_triangle_normal(
                                f, prim, 1.0 - a - b, a, b, tri_norm,
                                tri_pos, tri_uvs, tri_tex_meta, textures,
                                num_colored_triangles)
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
                            snrm, fnrm = _orient_hit_normals(snrm, fnrm, rd)
                            spos = ro + t_hit * rd
                            # Shadow-terminator origin (Hanika; RTGems II ch.
                            # 4), identical to raster_shadow_trace's: the
                            # face-normal lift stays, the stored/computed
                            # displacement onto the smooth surface rides on
                            # top of it, and ``lifted`` records whether the
                            # origin genuinely moved (a flat facet's delta is
                            # exactly zero by construction:
                            # _shadow_terminator_delta short-circuits a
                            # constant normal field) to license the
                            # horizon-cull relaxation below.
                            # RENDERER_WORK_QUEUE.md item 20.
                            sorigin = spos + fnrm * (10.0 * MIN_HIT_DISTANCE)
                            lifted = 0
                            if ti.static(shadow_term != 0):
                                if ti.static(shadow_term == 1):
                                    delta = _shadow_terminator_delta(
                                        f, prim, 1.0 - a - b, a, b, spos,
                                        snrm, tri_pos, tri_norm)
                                    if (delta[0] != 0.0) or (delta[1] != 0.0) \
                                            or (delta[2] != 0.0):
                                        sorigin = sorigin + delta
                                        lifted = 1
                                else:
                                    lifted = 1
                            for li in range(num_lights):
                                if li < _DEFERRED_SHADOW_LIGHTS:
                                    lp = ti.math.vec3(light_pos[tl, li, 0],
                                                      light_pos[tl, li, 1],
                                                      light_pos[tl, li, 2])
                                    to_light = lp - spos
                                    ldist = to_light.norm()
                                    if ldist > 1e-5:
                                        wi = to_light / ldist
                                        # Horizon cull, relaxed only where the
                                        # origin moved onto the smooth surface
                                        # (see raster_shadow_trace).
                                        horizon_ok = (fnrm.dot(wi) > 1e-3) \
                                            and (snrm.dot(wi) > 1e-4)
                                        if ti.static(shadow_term != 0):
                                            if lifted == 1:
                                                horizon_ok = \
                                                    snrm.dot(wi) > 1e-4
                                        if horizon_ok:
                                            occ = _shadow_occluded(
                                                refit, 1, sorigin, wi, f, ff,
                                                ldist - 20.0 * MIN_HIT_DISTANCE,
                                                pixel_size_per_t, base_dist,
                                                layer_offset_triangles,
                                                has_tri, has_bez,
                                                t_nodes, t_node_miss,
                                                t_leaf_prim, t_leaf_tspan,
                                                t_first_leaf, tri_pos,
                                                tri_colors, tri_uvs,
                                                tri_tex_meta, textures,
                                                tri_extra,
                                                num_colored_triangles,
                                                b_nodes, b_node_miss,
                                                b_leaf_prim, b_leaf_tspan,
                                                b_first_leaf, circuit_meta,
                                                circuit_colors,
                                                circuit_border_colors,
                                                edges_2d, edge_accel,
                                                # No source identity on this
                                                # path (see wavefront_traverse).
                                                -1, -1, 0.0, 0.0, tri_pos, 0)
                                            if _vis_max_component(occ) > 0.5:
                                                # Binary by necessity: the
                                                # bit pack cannot carry colour
                                                # (this mode is documented as
                                                # never enabled above, and is
                                                # now also colour-blind -- the
                                                # max-component reduction is
                                                # what a bit test means for an
                                                # RGB payload).
                                                bits |= (
                                                    1 << (q
                                                          * _DEFERRED_SHADOW_LIGHTS
                                                          + li))
        rs_vis[i] = bits


@ti.kernel
def wavefront_shade(
        active: ti.types.ndarray(), num_active: int,
        # Triangle STBVH (for shadow rays) + geometry/shading data.
        t_nodes: NODE_ARG, t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int,
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_extra: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        # Family A+B memory-trim: reordered/compacted triangle arrays + the
        # per-prim colour/extra remap ``col_row`` (see scene_builder._build_mem_
        # _trim). Unused when ``mem_trim == 0`` (col_row is a 1-elem stub).
        col_row: ti.types.ndarray(),
        # PN patch STBVH + geometry/shading data.
        # Bezier STBVH + geometry/shading data.
        b_nodes: NODE_ARG, b_node_miss: ti.types.ndarray(),
        b_leaf_prim: ti.types.ndarray(), b_leaf_tspan: ti.types.ndarray(),
        b_first_leaf: int,
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(),
        # Two floats packed into one ndarray to free an arg slot for ``col_row``
        # (this kernel is at Taichi's 64 runtime-arg ceiling): [tri, pn].
        layer_offsets: ti.types.ndarray(),
        # Fragment shading + deterministic hard shadows (compile-time
        # templates, both 0 on the default vertex-shaded path so the whole
        # block below compiles out) and their data.
        # ``refraction`` (also compile-time) enables Snell-law bending of the
        # transmitted ray for surfaces with a refractive index (extra cols 6-8).
        frag_shading: ti.template(), frag_pipelines: ti.template(),
        frag_scatters: ti.template(),
        # Compile-time bitmasks of the material pipeline ids the batch's
        # triangles / PN patches carry: the materials a scene does not use are
        # not compiled into this kernel at all, and a single-material batch
        # drops the per-hit id fetch and compare with them (see
        # ``shading_taichi._run_frag_pipeline``). ``ALL_PIDS`` keeps every
        # stage, which is the ungated kernel.
        tri_pids: ti.template(),
        shadows: ti.template(),
        refraction: ti.template(),
        # Nested-IOR media stack gate (module head): 0 compiles every stack
        # read/write out and keeps the classic rs_sca width.
        ior_stack: ti.template(),
        refit: ti.template(),
        has_tri: ti.template(), has_bez: ti.template(),
        deferred_shadows: ti.template(),
        # Shadow-terminator gate (rt_settings.shadow_terminator_mode()):
        # != 0 relaxes the face-normal horizon cull where the origin moved,
        # == 1 additionally applies the Hanika offset inline; 0 keeps
        # today's origin and guard exactly (see raster_shadow_trace).
        shadow_term: ti.template(),
        skip_unlit_normal: ti.template(),
        # Deliver the direct lights' share of the reflected specular lobe,
        # which the continuation this hit spawns cannot: a ray only finds
        # light that has geometry, and a delta light has none
        # (rt_settings.DIRECT_SPECULAR_LOBE). A ti.template(), not a runtime
        # arg, so it costs this kernel nothing against the CUDA argument
        # ceiling noted below. 0 restores the previous weighting exactly.
        direct_spec: ti.template(),
        mem_trim: ti.template(),
        opaque_closest: ti.template(),
        # Fused generation's first host iteration (see wavefront_traverse's
        # ``gen_first``): the initial per-ray state was never materialised, so
        # it is used as compile-time constants here (acc = 0, weight = 1,
        # t_prev = 0, layer_prev = 1e30, seam_t = -1e30, base_dist = 0,
        # processed = 0, pix = r) instead of read from global state;
        # max_bounces rides in layer_offsets[6] (this kernel is at the CUDA
        # 64-arg ceiling). Survivors write their state back below exactly as
        # before (plus rs_pix), so iterations >= 1 run the classic kernel.
        first_iter: ti.template(),
        # Sparse raster coverage: the ray's accumulator row is the compact
        # covered-pixel index in ``rs_int[:, 4]`` while ``rs_pix`` keeps the
        # real window-local pixel for frame/ray addressing (matching the
        # sheet resolve's contract). Compile-time, so the classic
        # dense path compiles the extra state out entirely -- and unlike the
        # ndarray shape it probed before, a template is actually a compile-time
        # constant (``ti.static`` rejects an ndarray ``.shape`` expression).
        compact: ti.template(),
        # Post-loop significance-floor exit (rt_settings.WEIGHT_FLOOR_EXIT,
        # read live per batch like the other template gates): a ray whose
        # throughput fell under MIN_WEIGHT retires even if its last act was an
        # in-place bounce -- every reflect branch ``break``s past the in-loop
        # floor test above and the peel-complete tests exclude bounced rays,
        # so such a ray otherwise rides to the bounce cap. Completion, not
        # truncation: the commit block below deposits its accumulated colour
        # + leftover throughput exactly as for any other retirement.
        weight_floor_exit: ti.template(),
        tri_mat_id: ti.types.ndarray(), tri_mat: ti.types.ndarray(),
        light_pos: ti.types.ndarray(), light_col: ti.types.ndarray(),
        num_lights: int,
        time_start: int, width: int, height: int, ray_offset: int,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
        rs_int: ti.types.ndarray(),
        hit_f: ti.types.ndarray(), hit_i: ti.types.ndarray(),
        rs_pix: ti.types.ndarray(), pix_accum: ti.types.ndarray(),
        rs_alloc: ti.types.ndarray(), rs_vis: ti.types.ndarray(),
        # Per-frame camera world position, handed on to the material stages
        # (``cam_pos``): depth-style shading measures from the CAMERA, and a
        # bounced ray's own origin is not it.
        cam_origin: ti.types.ndarray()):
    """Drain the compact event batch front-to-back.

    Alpha-composite each surface with per-geometry-type shading and mirror
    bounces until the ray's throughput is
    spent or its K-buffer is exhausted.

    When ``frag_shading`` is enabled, triangle/PN hits are material-shaded per
    fragment from the raw albedo (bezier circuits keep their sampled colour),
    and when ``shadows`` is also enabled each such fragment fires one
    opacity-accumulating shadow ray per light through all three BVHs inside the
    per-fragment lighting model.

    When ``refraction`` is enabled, a transparent refractive surface (glass)
    reflects AND refracts at once. The reflected branch continues in this ray
    slot while the transmitted branch appends to the tile-wide shared pool via
    ``rs_alloc``. If the append exceeds pool capacity, the host discards and
    reruns the tile with fewer primaries rather than accepting a missing branch.
    Every ray commits its colour + leftover background weight into ``pix_accum``
    through ``rs_pix``, so all branches of a pixel sum correctly.  The sparse
    raster path stores the compact accumulator row in ``rs_int[:, 4]`` while
    retaining the real local pixel in ``rs_pix`` for frame/ray addressing;
    the ``compact`` template selects that representation without consuming
    another runtime argument.
    """
    pixels_per_frame = width * height
    # Unpack the layer offset (packed into one ndarray to stay within the
    # 64-arg ceiling); the body below references this name unchanged.
    layer_offset_triangles = layer_offsets[0]
    # Optional extras ride behind the layer offset in the same packed
    # ndarray (again: 64-arg ceiling): [1..4] = environment map placement
    # (offset, width, height, intensity) in the shared texel buffer -- rays
    # that retire without consuming all their throughput pick up the
    # environment in their final direction (skybox + correct reflections) --
    # and [5] = the camera's far clip distance (0 = disabled).
    env_off = 0
    env_w = 0
    env_h = 0
    env_intensity = 0.0
    far_clip = 0.0
    if layer_offsets.shape[0] > 5:
        env_off = ti.cast(layer_offsets[1] + 0.5, ti.i32)
        env_w = ti.cast(layer_offsets[2] + 0.5, ti.i32)
        env_h = ti.cast(layer_offsets[3] + 0.5, ti.i32)
        env_intensity = layer_offsets[4]
        far_clip = layer_offsets[5]
    # First accumulator row belonging to the split-sum glossy half of
    # ``pix_accum`` (DESIGN_glossy_prefilter.md §4.3), or 0 when the route is
    # not active -- which is every render but an opt-in glossy one, and the
    # reason this rides in ``layer_offsets`` rather than as a kernel argument:
    # this kernel is at 72 parameters against Taichi's 64 runtime ones, the
    # same ceiling that put the environment map's placement in here.
    gloss_base = 0
    if layer_offsets.shape[0] > 7:
        gloss_base = ti.cast(layer_offsets[7] + 0.5, ti.i32)
    for i in range(num_active):
        r = active[i]
        pix = r
        if ti.static(first_iter == 0):
            pix = rs_pix[r]
        accum_pix = pix
        if ti.static(compact):
            accum_pix = rs_int[r, 4]
        # Per RAY, hoisted out of the hit loop: a glossy reflection ray and
        # everything it goes on to spawn share the glossy accumulator row.
        is_gloss_ray = (gloss_base > 0) and (accum_pix >= gloss_base)
        num_hits = rs_int[r, 3]
        if num_hits > 0:
            f = time_start + (ray_offset + pix) // pixels_per_frame
            ro = ti.math.vec3(rs_ro[r, 0], rs_ro[r, 1], rs_ro[r, 2])
            rd = ti.math.vec3(rs_rd[r, 0], rs_rd[r, 1], rs_rd[r, 2])
            acc = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
            weight = ti.math.vec3(1.0, 1.0, 1.0)
            t_prev = 0.0
            layer_prev = 1e30
            seam_t = -1e30
            base_dist = 0.0
            bounces_left = 0
            processed = 0
            if ti.static(first_iter != 0):
                bounces_left = ti.cast(layer_offsets[6] + 0.5, ti.i32)
            else:
                acc = ti.math.vec4(rs_acc[r, 0], rs_acc[r, 1], rs_acc[r, 2],
                                   rs_acc[r, 3])
                weight = ti.math.vec3(rs_sca[r, 0], rs_sca[r, 5],
                                      rs_sca[r, 6])
                t_prev = rs_sca[r, 1]
                layer_prev = rs_sca[r, 2]
                seam_t = rs_sca[r, 3]
                base_dist = rs_sca[r, 4]
                bounces_left = rs_int[r, 0]
                processed = rs_int[r, 1]

            kb_t = ti.Vector([0.0] * KBUF)
            kb_layer = ti.Vector([0.0] * KBUF)
            kb_prim = ti.Vector([0] * KBUF)
            kb_flags = ti.Vector([0] * KBUF)
            kb_a = ti.Vector([0.0] * KBUF)
            kb_b = ti.Vector([0.0] * KBUF)
            for q in ti.static(range(KBUF)):
                kb_t[q] = hit_f[i, q, 0]
                kb_layer[q] = hit_f[i, q, 1]
                kb_a[q] = hit_f[i, q, 2]
                kb_b[q] = hit_f[i, q, 3]
                kb_prim[q] = hit_i[i, q, 0]
                kb_flags[q] = hit_i[i, q, 1]

            bounced = False
            done = False
            drained = 0
            ff = ti.cast(f, ti.f32)
            pixel_size_per_t = pixel_world_scale[f]
            while drained < num_hits:
                # Nearest unconsumed slot. The best slot's (t, layer) ride in
                # scalars and the extraction below is a ti.static select, so
                # the kb_* vectors are never dynamically indexed (a dynamic
                # vector index spills the whole vector to local memory).
                sel = 0
                sel_found = 0
                t_hit = 0.0
                hit_layer = 0.0
                for q in ti.static(range(KBUF)):
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
                    # Past the camera's far distance. Hits drain front-to-back,
                    # so everything left is farther still -- retire the ray to
                    # the background/environment.
                    done = True
                    break
                prim = 0
                flags = 0
                a = 0.0
                b = 0.0
                for q in ti.static(range(KBUF)):
                    if q == sel:
                        prim = kb_prim[q]
                        flags = kb_flags[q]
                        a = kb_a[q]
                        b = kb_b[q]
                        kb_prim[q] = -1
                drained += 1
                processed += 1
                if is_gloss_ray:
                    # SPLIT-SUM PREFILTER (DESIGN_glossy_prefilter.md §4.3):
                    # how far past its reflector this reflection turned out to
                    # be, which is what sets its blur radius. The TOTAL camera
                    # path length, not the segment: a descendant's is always
                    # larger than its parent's, so the minimum over the whole
                    # sub-tree is exactly the first hit of the reflection ray
                    # itself. A short segment three bounces in would win a
                    # ``t_hit`` minimum and report a distant reflection as
                    # being in contact.
                    ti.atomic_min(pix_accum[accum_pix, _GL_ROW_DIST],
                                  base_dist + t_hit)
                htype = flags & 3
                edge_hit = (flags >> 2) & 1
                border = (flags >> 3) & 1

                seam_eps = DEPTH_TIE_EPSILON
                if (edge_hit == 1) and (t_hit - seam_t <= seam_eps):
                    t_prev = t_hit
                    layer_prev = hit_layer
                    continue
                seam_t = t_hit if edge_hit == 1 else -1e30

                color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                alpha = 0.0
                reflectivity = 0.0
                rough = 0.0
                if htype == 1:
                    w0 = 1.0 - a - b
                    color, alpha = _tri_color_g(mem_trim, f, prim, w0, a, b,
                                                tri_colors, col_row, tri_uvs,
                                                tri_tex_meta, textures,
                                                num_colored_triangles)
                    reflectivity, rough = _tri_extra_g(
                        mem_trim, f, prim, w0, a, b, tri_extra, col_row,
                        tri_uvs, tri_tex_meta, textures, num_colored_triangles)
                else:
                    color, alpha = _sample_circuit_color(
                        prim, f, a, b, border,
                        circuit_meta, circuit_colors, circuit_border_colors)
                    cm = f % circuit_meta.shape[0]
                    reflectivity = circuit_meta[cm, prim, _M_REFLECTIVITY]
                    rough = circuit_meta[cm, prim, _M_ROUGHNESS]

                # Raw surface colour, saved before fragment shading replaces
                # ``color`` with the lit result: the colour transport tints
                # the metal Fresnel lobe and the transmitted share with it.
                # (PBR materials always shade per fragment on this renderer,
                # so whenever the tint matters this is the true albedo.)
                albedo3 = ti.math.vec3(color[0], color[1], color[2])

                # Per-light shadow visibility for this hit (all-lit unless
                # shadow rays accumulate blocker opacity). Compiled out
                # when shadows are off; only triangle/PN hits cast/receive
                # shadows. The light loop below is a *runtime* loop (not
                # ti.static-unrolled) so the heavy ``_shadow_occluded`` ->
                # ``_nearest_surface`` -> PN solver call graph is inlined
                # once, not once per light. The payload is RGB,
                # channel-major per light (see shading_taichi.
                # light_vis_index): with the tint gate off every channel
                # holds the same scalar the old payload did.
                #
                # Declared at the hit's own scope rather than inside the
                # shading branch that fills it: the reflected lobe's
                # direct-light add-back further down is a sibling ti.static
                # block, and Taichi scopes a name to the block it is bound in.
                vis = ti.Vector([1.0]
                                * (3 * MAX_SHADOW_LIGHTS))
                # Fragment shading: ``color`` arrived as the interpolated raw
                # albedo for triangle/PN hits; evaluate the lighting model per
                # fragment. Bezier circuits (htype 0) keep their sampled colour.
                # Compiled out entirely on the default (vertex-shaded) path via
                # ti.static.
                if ti.static(frag_shading != 0):
                    if ti.static((shadows != 0) and (deferred_shadows != 0)):
                        # Legacy deferred shadows: read the per-(hit, light)
                        # binary occlusion bits precomputed by
                        # ``wavefront_shadow`` for this hit's K-buffer slot
                        # (``sel``). This mode is currently never enabled and
                        # must use floats before it can support the active
                        # opacity-weighted shadow contract -- and its bits
                        # cannot carry colour, so it is now also colour-blind:
                        # a blocked light sets all three channels to 0, an
                        # unblocked one keeps them at 1.
                        sbits = rs_vis[i]
                        for li in range(num_lights):
                            if li < _DEFERRED_SHADOW_LIGHTS:
                                if ((sbits
                                     >> (sel * _DEFERRED_SHADOW_LIGHTS + li))
                                        & 1) != 0:
                                    vis[light_vis_index(li, 0)] = 0.0
                                    vis[light_vis_index(li, 1)] = 0.0
                                    vis[light_vis_index(li, 2)] = 0.0
                    if ti.static((shadows != 0) and (deferred_shadows == 0)):
                        # Shadow visibility is skipped exactly where it cannot
                        # reach the output: an UNLIT hit never consumes ``vis``
                        # (passthrough shading; scatters take no ``vis``), and
                        # in every built-in stage a zero-colour light row (not
                        # yet spawned, or despawned) contributes nothing
                        # whatever its visibility -- every lit stage's terms
                        # carry the light colour as a factor. Only user
                        # pipelines, which may read ``vis`` arbitrarily, keep
                        # the exact fan for every light.
                        do_fan = 0
                        fan_exact = 1
                        fan_geom = 0
                        if htype == 1:
                            pid_s = tri_mat_id[f % tri_mat_id.shape[0], prim]
                            # ...and a hit whose mob declared
                            # receives_shadows False never consumes ``vis``
                            # either: it is shaded as though every light
                            # reached it, so the fan is skipped for the same
                            # reason an unlit hit skips it. Asked of BUILT-IN
                            # pipelines only, and behind a width test -- see
                            # sheet_resolve_taichi for why slot 33 belongs to a
                            # custom pipeline whenever its block is wide enough
                            # to have one.
                            recv_s = 1
                            if pid_s < _USER_PIPELINE_BASE:
                                if tri_mat.shape[2] > _MAT_NO_SHADOW_RECEIVE:
                                    if tri_mat[f % tri_mat.shape[0], prim,
                                               _MAT_NO_SHADOW_RECEIVE] > 0.5:
                                        recv_s = 0
                            if (pid_s != _MID_UNLIT) and (recv_s == 1):
                                do_fan = 1
                                if pid_s < _USER_PIPELINE_BASE:
                                    fan_exact = 0
                                    # Geometric zero-radiance culling is
                                    # valid for EVERY built-in stage: each
                                    # one's vis-multiplied terms carry lc,
                                    # so a culled fan's all-lit default
                                    # multiplies zero either way (see
                                    # _light_zero_radiance).
                                    fan_geom = 1
                        if do_fan == 1:
                            # Smooth shading normal and the *geometric* face
                            # normal of the hit facet/patch.
                            snrm = _tri_normal_g(
                                mem_trim, f, prim, 1.0 - a - b, a, b,
                                tri_norm, tri_pos, tri_uvs, tri_tex_meta,
                                textures, num_colored_triangles)
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
                            snrm, fnrm = _orient_hit_normals(snrm, fnrm, rd)
                            spos = ro + t_hit * rd
                            # Shadow-terminator origin (Hanika; RTGems II ch.
                            # 4), identical to raster_shadow_trace's: the
                            # face-normal lift stays, the displacement onto
                            # the smooth surface the vertex normals imply
                            # rides on top of it, and ``lifted`` records
                            # whether the origin genuinely moved (a flat
                            # facet's delta is exactly zero by construction:
                            # _shadow_terminator_delta short-circuits a
                            # constant normal field) to license the
                            # horizon-cull relaxation in the sample loop
                            # below. RENDERER_WORK_QUEUE.md item 20.
                            sorigin = spos + fnrm * (10.0 * MIN_HIT_DISTANCE)
                            lifted = 0
                            if ti.static(shadow_term != 0):
                                if ti.static(shadow_term == 1):
                                    delta = _shadow_terminator_delta(
                                        f, prim, 1.0 - a - b, a, b, spos,
                                        snrm, tri_pos, tri_norm)
                                    if (delta[0] != 0.0) or (delta[1] != 0.0) \
                                            or (delta[2] != 0.0):
                                        sorigin = sorigin + delta
                                        lifted = 1
                                else:
                                    lifted = 1
                            tl = f % light_pos.shape[0]
                            for li in range(num_lights):
                                if (li < MAX_SHADOW_LIGHTS) and (
                                        (fan_exact == 1)
                                        or (light_col[tl, li, 0] != 0.0)
                                        or (light_col[tl, li, 1] != 0.0)
                                        or (light_col[tl, li, 2] != 0.0)):
                                    lp = ti.math.vec3(light_pos[tl, li, 0],
                                                      light_pos[tl, li, 1],
                                                      light_pos[tl, li, 2])
                                    # Extended light rows carry a type id and a
                                    # soft-shadow radius; the compact 3-column
                                    # packing (plain point lights) keeps the
                                    # original single-ray path bit-for-bit.
                                    ltype = 0
                                    radius = 0.0
                                    hu = 0.0
                                    hv = 0.0
                                    if light_col.shape[2] > 3:
                                        ltype = ti.cast(
                                            light_col[tl, li, 3] + 0.5, ti.i32)
                                        radius = light_col[tl, li, 11]
                                        # A rect-area row carries its CELL's
                                        # half-extents along the emitter
                                        # plane's own axes. The ltype guard is
                                        # load-bearing, not defensive:
                                        # columns 9/10 are a spot light's cone
                                        # cosines there, and reading them
                                        # unguarded would turn every spot
                                        # light into a rect emitter.
                                        if ltype == _LT_AREA_SAMPLE:
                                            hu = light_col[tl, li, 9]
                                            hv = light_col[tl, li, 10]
                                    to_light = lp - spos
                                    ldist = to_light.norm()
                                    wi = ti.math.vec3(0.0, 0.0, 0.0)
                                    valid = 0
                                    if ltype == _LT_DIRECTIONAL:
                                        # Parallel rays: occlusion along the
                                        # (reversed) emission direction,
                                        # unbounded range.
                                        wi = -ti.math.vec3(
                                            light_col[tl, li, 6],
                                            light_col[tl, li, 7],
                                            light_col[tl, li, 8])
                                        ldist = 1e7
                                        valid = 1
                                    elif (ltype != _LT_AMBIENT) \
                                            and (ltype != _LT_HEMISPHERE) \
                                            and (ltype != _LT_ENV_SH) \
                                            and (ldist > 1e-5):
                                        wi = to_light / ldist
                                        valid = 1
                                    # A light past its range, a fragment
                                    # outside a spot cone, an area sample's
                                    # backface: exactly zero radiance here,
                                    # so the fan's result multiplies zero.
                                    # Skipping leaves vis[li] at its all-lit
                                    # default, exactly like the zero-colour
                                    # skip above.
                                    if (valid == 1) and (fan_geom == 1):
                                        if _light_zero_radiance(
                                                light_col, tl, li, ltype,
                                                to_light, ldist) == 1:
                                            valid = 0
                                    if valid == 1:
                                        # Soft shadows: a fixed golden-angle fan
                                        # of samples across the emitter disk
                                        # (directional: an angular cone,
                                        # radius = tan(half-angle)). radius 0
                                        # keeps the single hard ray.
                                        ns = 1
                                        b1 = ti.math.vec3(0.0, 0.0, 0.0)
                                        b2 = ti.math.vec3(0.0, 0.0, 0.0)
                                        if radius > 0.0:
                                            ns = SOFT_SHADOW_SAMPLES
                                            if (hu > 0.0) or (hv > 0.0):
                                                # Rect emitter: the fan samples
                                                # INSIDE this row's own cell, in
                                                # the light's own plane -- b1 is
                                                # the packed right axis and b2
                                                # the up axis recovered exactly
                                                # as _rect_axes builds it. The
                                                # offsets do not depend on wi,
                                                # so a moving sub-pixel origin
                                                # needs no basis rebuild.
                                                b1 = ti.math.vec3(
                                                    light_col[tl, li, 12],
                                                    light_col[tl, li, 13],
                                                    light_col[tl, li, 14])
                                                b2 = ti.math.vec3(
                                                    light_col[tl, li, 6],
                                                    light_col[tl, li, 7],
                                                    light_col[tl, li, 8]) \
                                                    .cross(b1)
                                            else:
                                                aref = ti.math.vec3(
                                                    1.0, 0.0, 0.0)
                                                if ti.abs(wi[0]) > 0.9:
                                                    aref = ti.math.vec3(
                                                        0.0, 1.0, 0.0)
                                                b1 = wi.cross(aref).normalized()
                                                b2 = wi.cross(b1)
                                        occ_sum = ti.math.vec3(0.0)
                                        n_valid = 0.0
                                        for s in range(ns):
                                            wis = wi
                                            ldn = ldist
                                            ok = 1
                                            # Declared here, not in the arms
                                            # below: a Taichi local is scoped
                                            # to the block it is FIRST
                                            # assigned in, so assigning it in
                                            # every arm of an if/else does not
                                            # make it readable after the
                                            # if/else (TaichiNameError at
                                            # compile time). raster_shadow_trace
                                            # carries the same initialiser for
                                            # the same reason.
                                            off = ti.math.vec3(0.0, 0.0, 0.0)
                                            if radius > 0.0:
                                                if (hu > 0.0) or (hv > 0.0):
                                                    # R2 sequence across the
                                                    # cell: s = 0 is exactly
                                                    # the cell centre, so a
                                                    # one-sample fan degenerates
                                                    # to today's ray.
                                                    u = 0.5 + _R2_SEQUENCE_A1 \
                                                        * s
                                                    v = 0.5 + _R2_SEQUENCE_A2 \
                                                        * s
                                                    ru = 2.0 * (u - ti.floor(u)) \
                                                        - 1.0
                                                    rv = 2.0 * (v - ti.floor(v)) \
                                                        - 1.0
                                                    off = b1 * (hu * ru) \
                                                        + b2 * (hv * rv)
                                                else:
                                                    ang = _GOLDEN_ANGLE * s
                                                    rr = radius * ti.sqrt(
                                                        (ti.cast(s, ti.f32)
                                                         + 0.5)
                                                        / ti.cast(ns, ti.f32))
                                                    off = (ti.cos(ang) * b1
                                                           + ti.sin(ang) * b2) \
                                                        * rr
                                                if ltype == _LT_DIRECTIONAL:
                                                    wis = (wi + off) \
                                                        .normalized()
                                                else:
                                                    tls = lp + off - spos
                                                    ldn = tls.norm()
                                                    if ldn > 1e-5:
                                                        wis = tls / ldn
                                                    else:
                                                        ok = 0
                                            # Skip samples below the geometric/
                                            # shading horizon (self-shadow acne
                                            # / no direct light to occlude
                                            # anyway). Where the origin moved
                                            # onto the smooth surface its
                                            # vertex normals imply (lifted == 1,
                                            # shadow_term != 0) the FACE
                                            # normal's horizon is not that
                                            # surface's, so its term drops and
                                            # only the shading normal's cull
                                            # remains; flat facets keep delta
                                            # == 0 and both keep the two-sided
                                            # test EXACTLY as written first.
                                            horizon_ok = (
                                                fnrm.dot(wis) > 1e-3) \
                                                and (snrm.dot(wis) > 1e-4)
                                            if ti.static(shadow_term != 0):
                                                if lifted == 1:
                                                    horizon_ok = \
                                                        snrm.dot(wis) > 1e-4
                                            if (ok == 1) and horizon_ok:
                                                n_valid += 1.0
                                                occ_sum += _shadow_occluded(
                                                    refit, shadows,
                                                    sorigin, wis, f, ff,
                                                    ldn - 20.0
                                                    * MIN_HIT_DISTANCE,
                                                    pixel_size_per_t, base_dist,
                                                    layer_offset_triangles,
                                                    has_tri, has_bez,
                                                    t_nodes, t_node_miss,
                                                    t_leaf_prim, t_leaf_tspan,
                                                    t_first_leaf, tri_pos,
                                                    tri_colors, tri_uvs,
                                                    tri_tex_meta, textures,
                                                    tri_extra,
                                                    num_colored_triangles,
                                                    b_nodes, b_node_miss,
                                                    b_leaf_prim, b_leaf_tspan,
                                                    b_first_leaf, circuit_meta,
                                                    circuit_colors,
                                                    circuit_border_colors,
                                                    edges_2d, edge_accel,
                                                    # No source identity on
                                                    # this path (the shade
                                                    # kernel does not receive
                                                    # tri_obj; see
                                                    # wavefront_traverse).
                                                    -1, -1, 0.0, 0.0, tri_pos, 0)
                                        if n_valid > 0.0:
                                            # The soft-shadow fan still
                                            # averages over the SCALAR sample
                                            # count; only the occlusion sum is
                                            # per-channel.
                                            base = light_vis_index(li, 0)
                                            vis[base] = 1.0 - occ_sum[0] \
                                                / n_valid
                                            vis[base + 1] = 1.0 - occ_sum[1] \
                                                / n_valid
                                            vis[base + 2] = 1.0 - occ_sum[2] \
                                                / n_valid
                    if htype == 1:
                        # Light with the *normal-mapped* shading normal (equals
                        # the interpolated vertex normal when the triangle has
                        # no normal map, so unmapped scenes are byte-identical).
                        # UNLIT hits pass their colour through unchanged and
                        # never consume the shading normal (a reflective/glass
                        # continuation recomputes its own normal below), so skip
                        # the normal work for them when the template is on.
                        sn = ti.math.vec3(0.0, 0.0, 0.0)
                        if ti.static(skip_unlit_normal != 0):
                            if tri_mat_id[f % tri_mat_id.shape[0], prim] \
                                    != _MID_UNLIT:
                                sn = _tri_normal_g(
                                    mem_trim, f, prim, 1.0 - a - b, a, b,
                                    tri_norm, tri_pos, tri_uvs, tri_tex_meta,
                                    textures, num_colored_triangles)
                        else:
                            sn = _tri_normal_g(
                                mem_trim, f, prim, 1.0 - a - b, a, b, tri_norm,
                                tri_pos, tri_uvs, tri_tex_meta, textures,
                                num_colored_triangles)
                        # A traversal hit IS the centre ray's intersection, so
                        # the position it always used is passed through
                        # unchanged (see _shade_tri_hit).
                        color = _shade_tri_hit(frag_pipelines, tri_pids,
                                               f, prim, a, b, rd,
                                               ro + t_hit * rd, tri_pos, sn,
                                               tri_mat_id, tri_mat,
                                               light_pos, light_col, num_lights,
                                               color, shadows, vis,
                                               cam_origin)

                if ti.static(len(frag_scatters) == 0):
                    # Built-in continuation.  The packed surface channels carry
                    # material metalness / roughness / IOR / transmission; no
                    # independent mirror/refraction controls remain.
                    alpha = ti.math.clamp(alpha, 0.0, 1.0)

                    ior = 0.0
                    T = 0.0
                    if htype == 1:
                        ior, T = _tri_ior_transmission_g(
                            mem_trim, f, prim, 1.0 - a - b, a, b,
                            tri_extra, col_row, tri_uvs, tri_tex_meta,
                            textures, num_colored_triangles)
                    else:
                        if ti.static(has_bez != 0):
                            cmr = f % circuit_meta.shape[0]
                            ior = circuit_meta[cmr, prim, _M_IOR]
                            T = circuit_meta[cmr, prim, _M_TRANSMISSION]
                    T = ti.math.clamp(T, 0.0, 1.0)

                    # The surface normal is consumed only by the PBR Fresnel
                    # lobe (metalness >= 0) and by a reflective / transmissive
                    # continuation.  For an unlit, non-transmissive hit (the
                    # common vertex-shaded flat/text case) _material_reflectance
                    # ignores the normal and returns R = 0, and every branch
                    # that reads the normal below is gated on R > 0 or T > 0 --
                    # so skip its cross-product + normalize entirely.  Byte-
                    # identical: when the guard is false the normal is dead.
                    normal = ti.math.vec3(0.0, 0.0, 0.0)
                    # The GEOMETRIC normal beside the shading one. Both the
                    # refracted origin offset and the reflection frame need it
                    # (a shading normal tipped past the silhouette aims the
                    # mirror ray into the solid -- see
                    # ``shading_taichi._reflect_frame``), so it is built once
                    # here under the same guard rather than per branch. A
                    # circuit is flat, so its two normals coincide.
                    geo_normal = ti.math.vec3(0.0, 0.0, 0.0)
                    if (reflectivity >= 0.0) or (T > 1e-4):
                        if htype == 1:
                            normal = _tri_normal_g(
                                mem_trim, f, prim, 1.0 - a - b, a, b, tri_norm,
                                tri_pos, tri_uvs, tri_tex_meta, textures,
                                num_colored_triangles)
                            gp = f % tri_pos.shape[0]
                            g0 = ti.math.vec3(tri_pos[gp, prim, 0],
                                              tri_pos[gp, prim, 1],
                                              tri_pos[gp, prim, 2])
                            g1 = ti.math.vec3(tri_pos[gp, prim, 3],
                                              tri_pos[gp, prim, 4],
                                              tri_pos[gp, prim, 5])
                            g2 = ti.math.vec3(tri_pos[gp, prim, 6],
                                              tri_pos[gp, prim, 7],
                                              tri_pos[gp, prim, 8])
                            geo_normal = (g1 - g0).cross(g2 - g0)
                        else:
                            normal = _bezier_normal(f, prim, circuit_meta)
                        normal = normal.normalized()
                        if htype != 1:
                            geo_normal = normal

                    R, diel_pass = _material_reflectance(
                        rd, normal, reflectivity, ior, albedo3, T)
                    # This route never spreads a continuation over the GGX
                    # lobe (the glossy fan lives in the raster resolve), so
                    # the mirror ray keeps only the share of the lobe it can
                    # stand for and the rest falls back to local shading --
                    # see ``_mirror_share``. Keeping the two routes agreed
                    # matters: a batch the raster front-end rejects lands here.
                    R *= _mirror_share(rough)
                    if bounces_left <= 0:
                        # Out of bounces: no reflected ray. Transmission stays
                        # gated by ``diel_pass`` -- see ``_scatter_impl``.
                        R = ti.math.vec3(0.0, 0.0, 0.0)

                    # Volumetric absorption (Beer-Lambert; glTF
                    # KHR_materials_volume semantics): this hit LEAVES a
                    # transmissive medium when the material transmits and the
                    # ray runs along the shading normal (rd . normal > 0) --
                    # the same side test ``_material_reflectance`` applies. The
                    # segment spent inside is ``t_hit - t_prev``, the distance
                    # from the surface the ray last crossed to this one: for a
                    # refracted ray, spawned AT the entry surface with
                    # ``t_prev = 0``, that is exactly its interior chord, and
                    # for a ray that walked through a partly-covering entry
                    # face without a bounce it is still the chord rather than
                    # the whole distance back to the camera. Exact for a single
                    # convex solid; nested media each attenuate by their own
                    # sigma over their own segment, which approximates the path
                    # integral through both. Multiplied into the running
                    # throughput BEFORE this hit's shading is accumulated and
                    # before any continuation weight is derived, so the surface
                    # itself and both branches are dimmed by the medium the
                    # light crossed to reach them. Built-in pipelines only: a
                    # custom fragment pipeline's parameter block is its own
                    # layout, so those slots of one are not a sigma.
                    if (T > 1e-4) and (htype == 1):
                        pid_a = tri_mat_id[f % tri_mat_id.shape[0], prim]
                        if pid_a < _USER_PIPELINE_BASE:
                            tma = f % tri_mat.shape[0]
                            sa = _MAT_ATTENUATION_SIGMA
                            sigma_a = ti.math.vec3(
                                tri_mat[tma, prim, sa],
                                tri_mat[tma, prim, sa + 1],
                                tri_mat[tma, prim, sa + 2])
                            if rd.dot(normal) > 0.0:
                                seg = ti.max(t_hit - t_prev, 0.0)
                                weight *= ti.math.vec3(
                                    ti.exp(-sigma_a[0] * seg),
                                    ti.exp(-sigma_a[1] * seg),
                                    ti.exp(-sigma_a[2] * seg))

                    # A transmissive surface refracts if it is solid geometry
                    # (is_glass) and transmits unbent if it is a zero-thickness
                    # circuit (is_pane); mutually exclusive by htype. Mirrors
                    # ``_scatter_impl``, including the four-way energy split,
                    # the metal-blended Fresnel ``R`` (the metal share
                    # reflects rather than transmits, so a fully metallic
                    # surface stays a mirror at any transmission), and the
                    # colour transport (vec3 weights; the albedo tints the
                    # metal lobe inside ``R`` and the transmitted share;
                    # decisions reduce to the maximum component).
                    is_glass = False
                    is_pane = False
                    if ti.static(refraction != 0):
                        if (T > 1e-4) and (bounces_left > 0) \
                                and (ior > 1.0 + 1e-4):
                            if htype == 1:
                                is_glass = True
                            else:
                                is_pane = True

                    one3 = ti.math.vec3(1.0, 1.0, 1.0)
                    tint = ti.math.clamp(albedo3, 0.0, 1.0)
                    # Only the dielectric-interior share transmits -- see the
                    # four-way split derivation in ``_scatter_impl``.
                    trans_share = diel_pass * T
                    # The glow lane has no colour channel; it takes the
                    # max-component reduction (exact when the components are
                    # equal, i.e. everywhere the old scalar transport reached).
                    r_glow = ti.max(R[0], ti.max(R[1], R[2]))
                    w_glow = ti.max(weight[0], ti.max(weight[1], weight[2]))
                    share = (weight * alpha) * (one3 - R - trans_share)
                    acc += ti.math.vec4(
                        share[0], share[1], share[2],
                        w_glow * alpha
                        * (1.0 - r_glow - trans_share)) * color
                    # The direct lights' share of the REFLECTED lobe, which
                    # the continuation spawned below cannot deliver: it is a
                    # ray, and a delta light has no geometry for a ray to
                    # find. Added at exactly the weight that ray took, which
                    # puts the lobe at unit weight alongside ``share``. See
                    # ``shading_taichi.direct_specular_lobe``.
                    # ``htype == 1`` (triangles/PN) only, for the reason
                    # given at the matching site in ``sheet_resolve_taichi``:
                    # a circuit is never material-shaded and its ``prim``
                    # indexes ``circuit_meta`` rather than ``tri_mat``.
                    # Gated on ``frag_shading`` as well, matching the sheet
                    # route: without it there is no per-fragment lighting on
                    # this hit at all, ``vis`` is never filled, and a
                    # vertex-shaded surface would gain an unshadowed highlight
                    # its own shading path never computed.
                    #
                    # The 4th (glow) lane takes nothing. Glow is the authored
                    # bloom channel, and a Fresnel highlight is not authored
                    # emission -- leaving it at zero keeps every existing
                    # bloom pass reading exactly what it read before.
                    if ti.static(direct_spec != 0 and frag_shading != 0) \
                            and (htype == 1) and (reflectivity >= 0.0) \
                            and (T > 1e-4):
                        sa = (weight * alpha) * (R + trans_share) \
                            * direct_specular_lobe(
                                f, prim, ro + t_hit * rd, -rd, normal,
                                geo_normal, reflectivity, rough, ior, albedo3,
                                tri_mat, light_pos, light_col, num_lights,
                                shadows, vis)
                        acc += ti.math.vec4(sa[0], sa[1], sa[2], 0.0)
                    refl_energy = alpha * R
                    refl_max = ti.max(refl_energy[0],
                                      ti.max(refl_energy[1], refl_energy[2]))
                    trans_energy = alpha * trans_share
                    cover_pass = 1.0 - alpha
                    cover3 = ti.math.vec3(cover_pass, cover_pass, cover_pass)

                    # Semi-transparent reflective surface: reflection into a
                    # split slot, pass-through stays primary (see
                    # ``default_scatter`` for why this way round).
                    split_refl = False
                    if ti.static(refraction != 0):
                        if (refl_max > MIN_ALPHA) \
                                and (cover_pass > MIN_ALPHA) \
                                and (bounces_left > 0):
                            split_refl = True

                    if is_glass:
                        wt = weight * trans_energy * tint
                        wt_max = ti.max(wt[0], ti.max(wt[1], wt[2]))
                        if wt_max > MIN_WEIGHT:
                            c, have_slot = _reserve_continuation_slot(
                                rs_alloc, rs_ro.shape[0])
                            if have_slot:
                                # Which side of the interface the ray crosses.
                                # The GEOMETRIC face normal decides, not the
                                # shading normal _refract_ray picks its own
                                # side from: inside-ness is geometric, and an
                                # interpolated normal tips past the silhouette
                                # at grazing angles (see _relative_ior). Exact
                                # 0 counts as exiting.
                                entering = rd.dot(geo_normal) < 0.0
                                rel = _relative_ior(rs_sca, r, ior, entering,
                                                    ior_stack)
                                rdt = _refract_ray(rd, normal, rel)
                                hp = ro + t_hit * rd
                                rorig = _offset_transmitted_origin(
                                    hp, rdt, geo_normal, normal)
                                for k in ti.static(range(3)):
                                    rs_ro[c, k] = rorig[k]
                                    rs_rd[c, k] = rdt[k]
                                for k in ti.static(range(4)):
                                    rs_acc[c, k] = 0.0
                                rs_sca[c, 0] = wt[0]
                                rs_sca[c, 1] = 0.0
                                rs_sca[c, 2] = 1e30
                                rs_sca[c, 3] = -1e30
                                rs_sca[c, 4] = base_dist + t_hit
                                rs_sca[c, 5] = wt[1]
                                rs_sca[c, 6] = wt[2]
                                rs_int[c, 0] = bounces_left - 1
                                rs_int[c, 1] = processed
                                rs_int[c, 2] = _ACTIVE
                                rs_int[c, 3] = 0
                                rs_pix[c] = pix
                                if ti.static(compact):
                                    rs_int[c, 4] = accum_pix
                                # The transmitted ray enters/exits a medium:
                                # push the hit's interpolated ior / pop (see
                                # _write_ior_stack; compiles out with the gate
                                # off).
                                _write_ior_stack(rs_sca, r, c, ior, entering,
                                                 True, ior_stack)
                        # Primary carries the heavier of reflection /
                        # coverage-miss; the lighter one takes a pool slot, so
                        # all three continuations are traced. At full coverage
                        # the miss is empty and the primary always reflects.
                        if (refl_max > MIN_ALPHA) \
                                and (refl_max >= cover_pass):
                            hit_point = ro + t_hit * rd
                            rd, nref = _reflect_frame(rd, normal, geo_normal)
                            ro = hit_point + nref * (10.0 * MIN_HIT_DISTANCE)
                            weight *= refl_energy
                            base_dist += t_hit
                            t_prev = 0.0
                            layer_prev = 1e30
                            seam_t = -1e30
                            bounces_left -= 1
                            bounced = True
                            break
                        else:
                            # The coverage-miss keeps the primary (it is the
                            # depth-layer walk), but the reflection is not
                            # therefore droppable: the pool is shared and
                            # append-only, so it takes a slot of its own, the
                            # same way ``split_refl`` does for a semi-transparent
                            # reflector below. Dropping it cost a partially
                            # covering glass fragment its whole Fresnel lobe --
                            # a dielectric's ~4% never outweighs the miss (see
                            # the matching branch in ``raster_taichi``).
                            rwt = weight * refl_energy
                            rwt_max = ti.max(rwt[0], ti.max(rwt[1], rwt[2]))
                            if rwt_max > MIN_WEIGHT:
                                c, have_slot = _reserve_continuation_slot(
                                    rs_alloc, rs_ro.shape[0])
                                if have_slot:
                                    rdr, nref = _reflect_frame(rd, normal,
                                                               geo_normal)
                                    hp = ro + t_hit * rd
                                    for k in ti.static(range(3)):
                                        rs_ro[c, k] = (
                                            hp[k] + nref[k]
                                            * (10.0 * MIN_HIT_DISTANCE))
                                        rs_rd[c, k] = rdr[k]
                                    for k in ti.static(range(4)):
                                        rs_acc[c, k] = 0.0
                                    rs_sca[c, 0] = rwt[0]
                                    rs_sca[c, 1] = 0.0
                                    rs_sca[c, 2] = 1e30
                                    rs_sca[c, 3] = -1e30
                                    rs_sca[c, 4] = base_dist + t_hit
                                    rs_sca[c, 5] = rwt[1]
                                    rs_sca[c, 6] = rwt[2]
                                    rs_int[c, 0] = bounces_left - 1
                                    rs_int[c, 1] = processed
                                    rs_int[c, 2] = _ACTIVE
                                    rs_int[c, 3] = 0
                                    rs_pix[c] = pix
                                    if ti.static(compact):
                                        rs_int[c, 4] = accum_pix
                                    # A reflection stays in the medium it was
                                    # in: copy the parent stack verbatim.
                                    _write_ior_stack(rs_sca, r, c, ior, False,
                                                     False, ior_stack)
                            weight *= cover_pass
                            t_prev = t_hit
                            layer_prev = hit_layer
                    elif is_pane:
                        # Thin pane: unbent transmission merges into the
                        # pass-through along with the coverage-miss, so only the
                        # reflection needs a slot (see ``_scatter_impl``).
                        wt = weight * refl_energy
                        wt_max = ti.max(wt[0], ti.max(wt[1], wt[2]))
                        if wt_max > MIN_WEIGHT:
                            c, have_slot = _reserve_continuation_slot(
                                rs_alloc, rs_ro.shape[0])
                            if have_slot:
                                rdr, nref = _reflect_frame(rd, normal,
                                                           geo_normal)
                                hp = ro + t_hit * rd
                                for k in ti.static(range(3)):
                                    rs_ro[c, k] = (hp[k] + nref[k]
                                                   * (10.0 * MIN_HIT_DISTANCE))
                                    rs_rd[c, k] = rdr[k]
                                for k in ti.static(range(4)):
                                    rs_acc[c, k] = 0.0
                                rs_sca[c, 0] = wt[0]
                                rs_sca[c, 1] = 0.0
                                rs_sca[c, 2] = 1e30
                                rs_sca[c, 3] = -1e30
                                rs_sca[c, 4] = base_dist + t_hit
                                rs_sca[c, 5] = wt[1]
                                rs_sca[c, 6] = wt[2]
                                rs_int[c, 0] = bounces_left - 1
                                rs_int[c, 1] = processed
                                rs_int[c, 2] = _ACTIVE
                                rs_int[c, 3] = 0
                                rs_pix[c] = pix
                                if ti.static(compact):
                                    rs_int[c, 4] = accum_pix
                                # Pane reflection: same medium, stack copied.
                                _write_ior_stack(rs_sca, r, c, ior, False,
                                                 False, ior_stack)
                        weight *= cover3 + trans_energy * tint
                        t_prev = t_hit
                        layer_prev = hit_layer
                    elif split_refl:
                        wt = weight * refl_energy
                        wt_max = ti.max(wt[0], ti.max(wt[1], wt[2]))
                        if wt_max > MIN_WEIGHT:
                            c, have_slot = _reserve_continuation_slot(
                                rs_alloc, rs_ro.shape[0])
                            if have_slot:
                                rdr, nref = _reflect_frame(rd, normal,
                                                           geo_normal)
                                hp = ro + t_hit * rd
                                for k in ti.static(range(3)):
                                    rs_ro[c, k] = (hp[k] + nref[k]
                                                   * (10.0 * MIN_HIT_DISTANCE))
                                    rs_rd[c, k] = rdr[k]
                                for k in ti.static(range(4)):
                                    rs_acc[c, k] = 0.0
                                rs_sca[c, 0] = wt[0]
                                rs_sca[c, 1] = 0.0
                                rs_sca[c, 2] = 1e30
                                rs_sca[c, 3] = -1e30
                                rs_sca[c, 4] = base_dist + t_hit
                                rs_sca[c, 5] = wt[1]
                                rs_sca[c, 6] = wt[2]
                                rs_int[c, 0] = bounces_left - 1
                                rs_int[c, 1] = processed
                                rs_int[c, 2] = _ACTIVE
                                rs_int[c, 3] = 0
                                rs_pix[c] = pix
                                if ti.static(compact):
                                    rs_int[c, 4] = accum_pix
                                # split_refl reflection: same medium, stack
                                # copied.
                                _write_ior_stack(rs_sca, r, c, ior, False,
                                                 False, ior_stack)
                        weight *= cover3 + trans_energy * tint
                        t_prev = t_hit
                        layer_prev = hit_layer
                    # No split pool: reflect only while the reflection
                    # outweighs what shows through (see ``default_scatter``).
                    elif ((refl_max > MIN_ALPHA)
                          and (refl_max >= cover_pass)):
                        hit_point = ro + t_hit * rd
                        rd, nref = _reflect_frame(rd, normal, geo_normal)
                        ro = hit_point + nref * (10.0 * MIN_HIT_DISTANCE)
                        weight *= refl_energy
                        base_dist += t_hit
                        t_prev = 0.0
                        layer_prev = 1e30
                        seam_t = -1e30
                        bounces_left -= 1
                        bounced = True
                        break
                    else:
                        # Orphaned transmitted share (index-matched ior <= 1,
                        # pool absent, or bounces exhausted) continues unbent
                        # in the pass-through -- see ``_scatter_impl``.
                        weight *= cover3 + trans_energy * tint
                        t_prev = t_hit
                        layer_prev = hit_layer
                else:
                    # A custom scatter exists in the scene: every fragment's
                    # continuation is decided by its material's scatter func
                    # (built-ins / scatterless pipelines fall back to
                    # default_scatter), so reflection / refraction / pass-
                    # through are all user-overridable. Same front-to-back
                    # bookkeeping as the inline branch, expressed through the
                    # scatter's returned branches (mirrors ``wf_shade_event``).
                    sni = ti.math.vec3(0.0, 0.0, 0.0)
                    sfn = ti.math.vec3(0.0, 0.0, 0.0)
                    if htype == 1:
                        sni = _tri_normal_g(
                            mem_trim, f, prim, 1.0 - a - b, a, b, tri_norm,
                            tri_pos, tri_uvs, tri_tex_meta, textures,
                            num_colored_triangles)
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
                        sfn = (v1 - v0).cross(v2 - v0)
                    else:
                        sni = _bezier_normal(f, prim, circuit_meta)
                        sfn = sni
                    s_ior = 0.0
                    s_trans = 0.0
                    if htype == 1:
                        s_ior, s_trans = _tri_ior_transmission_g(
                            mem_trim, f, prim, 1.0 - a - b, a, b, tri_extra,
                            col_row, tri_uvs, tri_tex_meta, textures,
                            num_colored_triangles)
                    else:
                        if ti.static(has_bez != 0):
                            cmr = f % circuit_meta.shape[0]
                            s_ior = circuit_meta[cmr, prim, _M_IOR]
                            s_trans = circuit_meta[cmr, prim, _M_TRANSMISSION]
                    hit_point = ro + t_hit * rd
                    # Volumetric absorption, identical rule to the inline
                    # branch above (see it for the side test, the segment
                    # length and the built-in-pipeline-only slot layout): the
                    # attenuation wraps the scatter call -- applied to the
                    # running throughput before its contribution accumulates
                    # and before any continuation weight is derived -- because
                    # the scatter contract's injected signature is fixed and
                    # must not grow an argument.
                    if (s_trans > 1e-4) and (htype == 1):
                        pid_a = tri_mat_id[f % tri_mat_id.shape[0], prim]
                        if pid_a < _USER_PIPELINE_BASE:
                            tma = f % tri_mat.shape[0]
                            sa = _MAT_ATTENUATION_SIGMA
                            sigma_a = ti.math.vec3(
                                tri_mat[tma, prim, sa],
                                tri_mat[tma, prim, sa + 1],
                                tri_mat[tma, prim, sa + 2])
                            if rd.dot(sni.normalized()) > 0.0:
                                seg = ti.max(t_hit - t_prev, 0.0)
                                weight *= ti.math.vec3(
                                    ti.exp(-sigma_a[0] * seg),
                                    ti.exp(-sigma_a[1] * seg),
                                    ti.exp(-sigma_a[2] * seg))
                    zero3 = ti.math.vec3(0.0, 0.0, 0.0)
                    contrib = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                    pass_w = zero3
                    refl_orig = zero3
                    refl_dir = zero3
                    refl_w = zero3
                    trans_orig = zero3
                    trans_dir = zero3
                    trans_w = zero3
                    if htype == 1:
                        (contrib, pass_w, refl_orig, refl_dir, refl_w,
                         trans_orig, trans_dir, trans_w) = _run_frag_scatter(
                            frag_scatters, tri_mat_id, f, prim, rd, sni, sfn,
                            hit_point, color, albedo3, alpha, reflectivity,
                            s_ior, s_trans, tri_mat, bounces_left, refraction)
                    else:
                        # Circuits are never material-shaded, so no user
                        # pipeline (hence no user scatter) can own them; they
                        # always take the built-in continuation -- as a thin
                        # pane, not refracting solid geometry.
                        (contrib, pass_w, refl_orig, refl_dir, refl_w,
                         trans_orig, trans_dir, trans_w) = circuit_scatter(
                            rd, sni, sfn, hit_point, color, albedo3, alpha,
                            reflectivity, s_ior, s_trans, tri_mat, f, prim,
                            bounces_left, refraction)
                    w_glow = ti.max(weight[0],
                                    ti.max(weight[1], weight[2]))
                    acc += ti.math.vec4(weight[0], weight[1], weight[2],
                                        w_glow) * contrib
                    if ti.static(refraction != 0):
                        wt = weight * trans_w
                        wt_max = ti.max(wt[0], ti.max(wt[1], wt[2]))
                        trans_w_max = ti.max(trans_w[0],
                                             ti.max(trans_w[1], trans_w[2]))
                        if (trans_w_max > 0.0) and (wt_max > MIN_WEIGHT) \
                                and (bounces_left > 0):
                            c, have_slot = _reserve_continuation_slot(
                                rs_alloc, rs_ro.shape[0])
                            if have_slot:
                                for k in ti.static(range(3)):
                                    rs_ro[c, k] = trans_orig[k]
                                    rs_rd[c, k] = trans_dir[k]
                                for k in ti.static(range(4)):
                                    rs_acc[c, k] = 0.0
                                rs_sca[c, 0] = wt[0]
                                rs_sca[c, 1] = 0.0
                                rs_sca[c, 2] = 1e30
                                rs_sca[c, 3] = -1e30
                                rs_sca[c, 4] = base_dist + t_hit
                                rs_sca[c, 5] = wt[1]
                                rs_sca[c, 6] = wt[2]
                                rs_int[c, 0] = bounces_left - 1
                                rs_int[c, 1] = processed
                                rs_int[c, 2] = _ACTIVE
                                rs_int[c, 3] = 0
                                rs_pix[c] = pix
                                if ti.static(compact):
                                    rs_int[c, 4] = accum_pix
                                # A scene carrying ANY custom fragment scatter
                                # gets no nested IOR: this arm owns every
                                # fragment once one exists, and the fixed
                                # scatter-injection signature cannot carry a
                                # relative index or say what medium the
                                # returned trans_dir leaves the ray in. So the
                                # value passed as ior is unchanged (s_ior, the
                                # material's own) and the transmitted branch
                                # copies the parent stack verbatim -- it
                                # continues in the parent medium. Bookkeeping
                                # stays sound (a wrong-but-consistent medium
                                # beats arena noise); extending the contract is
                                # future work (DESIGN_mesh_identity_open.md §H).
                                _write_ior_stack(rs_sca, r, c, s_ior, False,
                                                 False, ior_stack)
                    refl_w_max = ti.max(refl_w[0],
                                        ti.max(refl_w[1], refl_w[2]))
                    if (refl_w_max > 0.0) and (bounces_left > 0):
                        ro = refl_orig
                        rd = refl_dir
                        weight *= refl_w
                        base_dist += t_hit
                        t_prev = 0.0
                        layer_prev = 1e30
                        seam_t = -1e30
                        bounces_left -= 1
                        bounced = True
                        break
                    else:
                        weight *= pass_w
                        t_prev = t_hit
                        layer_prev = hit_layer
                if ti.max(weight[0], ti.max(weight[1], weight[2])) \
                        < MIN_WEIGHT:
                    done = True
                    break

            if ti.static(opaque_closest):
                if (not done) and (not bounced):
                    done = True
            else:
                if (not done) and (not bounced) and (num_hits < KBUF):
                    done = True
            if ti.static(weight_floor_exit):
                # The same significance floor the in-loop test applies to
                # pass-through hits, reached by rays the bounce branches'
                # ``break`` skipped it for. Completion: do NOT touch
                # ALLOC_TRUNC_SURFACES -- what this drops is sub-floor
                # transport, not image the ceiling cuts short.
                if ti.max(weight[0], ti.max(weight[1], weight[2])) \
                        < MIN_WEIGHT:
                    done = True
            if processed >= MAX_SURFACES_PER_RAY:
                # Truncation, not completion: the blocks above have already set
                # ``done`` for every ray that finished on its own terms, so a
                # ray still active here is one the ceiling is cutting short --
                # it either bounced or had hits left to drain. Counted, because
                # what it drops is image (``truncation.py``). The store is
                # unchanged, so the frame is byte-identical.
                if not done:
                    ti.atomic_add(rs_alloc[ALLOC_TRUNC_SURFACES], 1)
                done = True

            for k in ti.static(range(3)):
                rs_ro[r, k] = ro[k]
                rs_rd[r, k] = rd[k]
            for k in ti.static(range(4)):
                rs_acc[r, k] = acc[k]
            # Columns 7+ (the nested-IOR stack) are deliberately NOT rewritten:
            # a ray that continues as its own reflection / pass-through stays
            # in the medium it was in, so its stack survives untouched. That is
            # only sound because every slot's stack columns are zeroed once per
            # tile on the host (tracer._run_wavefront_tiles), so an unwritten
            # stack reads as air rather than as the previous occupant's media.
            rs_sca[r, 0] = weight[0]
            rs_sca[r, 1] = t_prev
            rs_sca[r, 2] = layer_prev
            rs_sca[r, 3] = seam_t
            rs_sca[r, 4] = base_dist
            rs_sca[r, 5] = weight[1]
            rs_sca[r, 6] = weight[2]
            rs_int[r, 0] = bounces_left
            rs_int[r, 1] = processed
            rs_int[r, 2] = _DONE if done else _ACTIVE
            if ti.static(first_iter != 0):
                # Fused generation never materialised rs_pix; surviving rays'
                # later iterations (and only they) read it.
                rs_pix[r] = r
            if done:
                # Terminated: commit this branch's premultiplied colour and its
                # leftover throughput (what the background shows through) into
                # the shared per-pixel accumulator. With an environment map the
                # leftover throughput samples the map in the ray's final
                # direction instead (so mirrors and glass reflect the sky).
                if (env_w > 0) and (ti.max(weight[0], ti.max(
                        weight[1], weight[2])) > 0.0):
                    ec = _sample_env_map(f, rd, env_off, env_w, env_h,
                                         env_intensity, textures)
                    for k in ti.static(range(3)):
                        acc[k] += weight[k] * ec[k]
                    weight = ti.math.vec3(0.0, 0.0, 0.0)
                for k in ti.static(range(4)):
                    ti.atomic_add(pix_accum[accum_pix, k], acc[k])
                for k in ti.static(range(3)):
                    ti.atomic_add(pix_accum[accum_pix, 4 + k], weight[k])
        else:
            # Ray escaped to the background this segment: commit its colour +
            # leftover (background) throughput, then retire.
            f = time_start + (ray_offset + pix) // pixels_per_frame
            ff = ti.cast(f, ti.f32)
            ro = ti.math.vec3(rs_ro[r, 0], rs_ro[r, 1], rs_ro[r, 2])
            rd = ti.math.vec3(rs_rd[r, 0], rs_rd[r, 1], rs_rd[r, 2])
            inv_rd = ti.math.vec3(_safe_inverse(rd[0]),
                                  _safe_inverse(rd[1]),
                                  _safe_inverse(rd[2]))

            w_bg = ti.math.vec3(1.0, 1.0, 1.0)
            if ti.static(first_iter == 0):
                w_bg = ti.math.vec3(rs_sca[r, 0], rs_sca[r, 5], rs_sca[r, 6])
            if (env_w > 0) and (ti.max(w_bg[0], ti.max(
                    w_bg[1], w_bg[2])) > 0.0):
                ec = _sample_env_map(f, rd, env_off, env_w, env_h,
                                     env_intensity, textures)
                for k in ti.static(range(3)):
                    ti.atomic_add(
                        pix_accum[accum_pix, k], w_bg[k] * ec[k])
                w_bg = ti.math.vec3(0.0, 0.0, 0.0)
            if ti.static(first_iter == 0):
                # First iteration's accumulator is implicitly zero; adding it
                # would be a no-op, so the read is skipped entirely.
                for k in ti.static(range(4)):
                    ti.atomic_add(
                        pix_accum[accum_pix, k], rs_acc[r, k])
            for k in ti.static(range(3)):
                ti.atomic_add(pix_accum[accum_pix, 4 + k], w_bg[k])
            rs_int[r, 2] = _DONE
