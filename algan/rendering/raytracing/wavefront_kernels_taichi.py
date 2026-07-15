"""Wavefront (stage-split) variant of deterministic general ray-trace
kernel.

This module splits the tracing process into small per-stage kernels connected by
per-ray state in global memory, driven by a host-side iteration loop:

* :func:`wavefront_generate_rays`      -- initialise per-ray state with primary rays.
* :func:`wavefront_traverse` -- for each *active* ray, gather the KBUF nearest
  hits (reusing the unchanged ``_collect_hits_tri``) into global state.
* :func:`wavefront_shade`    -- applies a pipeline of built-in fragment shaders (and optionally user-provided custom fragment shaders)..
* :func:`wf_composite`         -- composite each ray's accumulator over the
  background.

Between iterations the host compacts the still-active rays with a PyTorch
``nonzero`` (see ``render_triangles_wavefront`` in ``primitives.py``), so each
launch processes only rays that still have work -- warps refill as rays drop
out, which is the divergence fix.
"""
import taichi as ti

from algan.rendering.raytracing.raytrace_kernels_taichi import (
    DEPTH_TIE_EPSILON,
    KBUF,
    MAX_SHADOW_LIGHTS,
    MAX_SURFACES_PER_RAY,
    MIN_ALPHA,
    MIN_HIT_DISTANCE,
    MIN_WEIGHT,
    NODE_ARG,
    PN_SEAM_DEPTH_EPSILON,
    _bezier_normal,
    _collect_hits,
    _comes_after,
    _generate_ray,
    _pn_normal,
    _safe_inverse,
    _sample_circuit_color,
    _shade_pn_hit,
    _shade_tri_hit,
    _shadow_occluded,
    _triangle_color,
    _flat_triangle_color,
    _triangle_extra,
    _triangle_normal,
    _nearest_surface_g,
    finalize_pixel_color,
)
from algan.rendering.raytracing.shading_taichi import (
    _MID_UNLIT,
    _USER_PIPELINE_BASE,
)
from algan.rendering.raytracing.settings import SOFT_SHADOW_SAMPLES

# Per-ray status codes (rs_int column 2).
_ACTIVE = 0
_DONE = 1


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
_LT_DIRECTIONAL = 1
_LT_AMBIENT = 2
_LT_HEMISPHERE = 3
_LT_ENV_SH = 6

# Golden-angle increment of the deterministic soft-shadow sample fan.
_GOLDEN_ANGLE = 2.3999632297286533

_PI = 3.141592653589793

# Deferred shadows (opt-in ``DEFER_WF_SHADOWS``) pack a per-(K-buffer hit,
# light) occlusion bit into a single int32, so at most 32 // KBUF lights fit.
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
    index (0/1 = not refractive); columns 0-5 are reflectivity/roughness."""
    te = f % extra.shape[0]
    return (w0 * extra[te, prim, 6] + w1 * extra[te, prim, 7]
            + w2 * extra[te, prim, 8])


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
    maps can share the buffer with the color maps)."""
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
    in which case that property is sampled per fragment instead."""
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
def _flat_corner_ior(f, prim, w0, w1, w2, extra: ti.template(),
                     tri_uvs: ti.template(), tri_tex_meta: ti.template(),
                     textures: ti.template(), num_colored_triangles: ti.i32):
    """Index of refraction of a triangle hit: per-vertex (``_corner_ior``)
    unless the material map's bitmask marks it texture-driven (bit 2 /
    channel 2)."""
    # See _flat_triangle_extra: a promoted constant-material triangle has no
    # per-vertex extra row, so its IOR is read from the material map (bit 4);
    # the guard is a no-op (always true) for every non-promoted batch.
    ior = 1.0
    if prim < extra.shape[1]:
        ior = _corner_ior(f, prim, w0, w1, w2, extra)
    if prim >= num_colored_triangles:
        idx = prim - num_colored_triangles
        if tri_tex_meta[idx, 3] >= 0:
            if (tri_tex_meta[idx, 9] & 4) != 0:
                u, v = _tri_uv(f, idx, w0, w1, w2, tri_uvs)
                m = _sample_tex_vec5(f, u, v, tri_tex_meta[idx, 3],
                                     tri_tex_meta[idx, 4],
                                     tri_tex_meta[idx, 5], textures)
                ior = m[2]
    return ior


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
    extra per-vertex tangent array is needed."""
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
def _flat_corner_ior_trim(f, prim, w0, w1, w2, tri_extra: ti.template(),
                          col_row: ti.template(), tri_uvs: ti.template(),
                          tex_meta: ti.template(), textures: ti.template()):
    ior = 1.0
    cr = col_row[prim]
    if cr >= 0:
        ior = _corner_ior(f, cr, w0, w1, w2, tri_extra)
    if tex_meta[prim, 3] >= 0:
        if (tex_meta[prim, 9] & 4) != 0:
            u, v = _tri_uv(f, prim, w0, w1, w2, tri_uvs)
            m = _sample_tex_vec5(f, u, v, tex_meta[prim, 3], tex_meta[prim, 4],
                                 tex_meta[prim, 5], textures)
            ior = m[2]
    return ior


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
def _tri_ior_g(mem_trim: ti.template(), f, prim, w0, w1, w2,
               tri_extra: ti.template(), col_row: ti.template(),
               tri_uvs: ti.template(), tex_meta: ti.template(),
               textures: ti.template(), num_colored: ti.template()):
    ior = 1.0
    if ti.static(mem_trim != 0):
        ior = _flat_corner_ior_trim(
            f, prim, w0, w1, w2, tri_extra, col_row, tri_uvs, tex_meta,
            textures)
    else:
        ior = _flat_corner_ior(
            f, prim, w0, w1, w2, tri_extra, tri_uvs, tex_meta, textures,
            num_colored)
    return ior


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


# ---------------------------------------------------------------------------
# PN (curved point-normal) patch texture sampling. Unlike flat triangles, PN
# patches have no dedicated uv/tex-meta kernel arrays (the general wavefront
# shade kernel is at Taichi's 64-arg ceiling); the per-corner UVs and the
# per-patch texture metadata ride in the *widened* pn_extra array built by
# _merge_scene: cols 15-20 per-corner UV (u0,v0,u1,v1,u2,v2), 21-23 color map
# (offset, w, h) into the shared ``textures`` buffer, 24-26 material map, 27-29
# normal map, 30 material bitmask. A map offset of -1 means "no map" (fall back
# to the per-vertex value). All three maps sample the same shared ``textures``
# buffer the flat path uses.
# ---------------------------------------------------------------------------


@ti.func
def _pn_uv(te, prim, w0, w1, w2, pn_extra: ti.template()):
    """Barycentric UV of a PN hit (per-corner UVs live in pn_extra cols 15-20)."""
    u = (w0 * pn_extra[te, prim, 15] + w1 * pn_extra[te, prim, 17]
         + w2 * pn_extra[te, prim, 19])
    v = (w0 * pn_extra[te, prim, 16] + w1 * pn_extra[te, prim, 18]
         + w2 * pn_extra[te, prim, 20])
    return u, v


@ti.func
def _pn_hit_color(f, prim, w0, w1, w2, pn_colors: ti.template(),
                  pn_extra: ti.template(), textures: ti.template()):
    """Color + alpha of a PN hit: the bilinearly-sampled color map (pn_extra
    cols 21-23) if present, else the per-vertex pn_colors."""
    te = f % pn_extra.shape[0]
    color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    alpha = 0.0
    coff = ti.cast(pn_extra[te, prim, 21], ti.i32)
    if coff < 0:
        color, alpha = _triangle_color(f, prim, w0, w1, w2, pn_colors)
    else:
        u, v = _pn_uv(te, prim, w0, w1, w2, pn_extra)
        m = _sample_tex_vec5(f, u, v, coff,
                             ti.cast(pn_extra[te, prim, 22], ti.i32),
                             ti.cast(pn_extra[te, prim, 23], ti.i32), textures)
        color = ti.math.vec4(m[0], m[1], m[2], m[3])
        alpha = m[4]
    return color, alpha


@ti.func
def _pn_hit_extra(f, prim, w0, w1, w2, pn_extra: ti.template(),
                  textures: ti.template()):
    """(reflectivity, roughness) of a PN hit: per-vertex (``_triangle_extra``
    on pn_extra cols 0-5) unless the material map (cols 24-26) overrides the
    channel marked in the bitmask (col 30)."""
    reflectivity, roughness = _triangle_extra(f, prim, w0, w1, w2, pn_extra)
    te = f % pn_extra.shape[0]
    moff = ti.cast(pn_extra[te, prim, 24], ti.i32)
    if moff >= 0:
        flags = ti.cast(pn_extra[te, prim, 30], ti.i32)
        u, v = _pn_uv(te, prim, w0, w1, w2, pn_extra)
        m = _sample_tex_vec5(f, u, v, moff,
                             ti.cast(pn_extra[te, prim, 25], ti.i32),
                             ti.cast(pn_extra[te, prim, 26], ti.i32), textures)
        if (flags & 1) != 0:
            reflectivity = m[0]
        if (flags & 2) != 0:
            roughness = m[1]
    return reflectivity, roughness


@ti.func
def _pn_hit_ior(f, prim, w0, w1, w2, pn_extra: ti.template(),
                textures: ti.template()):
    """Index of refraction of a PN hit: per-vertex (``_corner_ior``) unless the
    material map's bitmask marks it texture-driven (bit 2 / channel 2)."""
    ior = _corner_ior(f, prim, w0, w1, w2, pn_extra)
    te = f % pn_extra.shape[0]
    moff = ti.cast(pn_extra[te, prim, 24], ti.i32)
    if moff >= 0:
        if (ti.cast(pn_extra[te, prim, 30], ti.i32) & 4) != 0:
            u, v = _pn_uv(te, prim, w0, w1, w2, pn_extra)
            m = _sample_tex_vec5(f, u, v, moff,
                                 ti.cast(pn_extra[te, prim, 25], ti.i32),
                                 ti.cast(pn_extra[te, prim, 26], ti.i32),
                                 textures)
            ior = m[2]
    return ior


@ti.func
def _pn_hit_normal(f, prim, a, b, pn_norm: ti.template(),
                   pn_ctrl: ti.template(), pn_extra: ti.template(),
                   textures: ti.template()):
    """Shading normal of a PN hit, perturbed by a tangent-space normal map
    (pn_extra cols 27-29) when present. The tangent frame is derived per hit
    from the patch's surface derivatives (dP/da, dP/db) and the UV gradients --
    the curved analogue of the flat triangle's edge/UV tangent frame."""
    normal = _pn_normal(f, prim, a, b, pn_norm, pn_ctrl)
    te = f % pn_extra.shape[0]
    noff = ti.cast(pn_extra[te, prim, 27], ti.i32)
    if noff >= 0:
        w0 = 1.0 - a - b
        u, v = _pn_uv(te, prim, w0, a, b, pn_extra)
        m = _sample_tex_vec5(f, u, v, noff,
                             ti.cast(pn_extra[te, prim, 28], ti.i32),
                             ti.cast(pn_extra[te, prim, 29], ti.i32), textures)
        tn = ti.math.vec3(m[0], m[1], m[2])
        if tn.norm() > 1e-6 and normal.norm() > 1e-9:
            nb = normal.normalized()
            # Patch surface derivatives dP/da (su) and dP/db (sv) at the hit.
            tp = f % pn_ctrl.shape[0]
            su = ti.math.vec3(0.0, 0.0, 0.0)
            sv = ti.math.vec3(0.0, 0.0, 0.0)
            for ci in ti.static(range(3)):
                su[ci] = (pn_ctrl[tp, prim, 3 + ci]
                          + 2.0 * a * pn_ctrl[tp, prim, 9 + ci]
                          + b * pn_ctrl[tp, prim, 15 + ci])
                sv[ci] = (pn_ctrl[tp, prim, 6 + ci]
                          + 2.0 * b * pn_ctrl[tp, prim, 12 + ci]
                          + a * pn_ctrl[tp, prim, 15 + ci])
            # UV gradients w.r.t. barycentric (a, b): linear in the corner UVs.
            du1 = pn_extra[te, prim, 17] - pn_extra[te, prim, 15]
            dv1 = pn_extra[te, prim, 18] - pn_extra[te, prim, 16]
            du2 = pn_extra[te, prim, 19] - pn_extra[te, prim, 15]
            dv2 = pn_extra[te, prim, 20] - pn_extra[te, prim, 16]
            det = du1 * dv2 - du2 * dv1
            if ti.abs(det) > 1e-12:
                inv_det = 1.0 / det
                tang = (su * dv2 - sv * dv1) * inv_det
                tang = tang - nb * nb.dot(tang)  # Gram-Schmidt vs normal
                if tang.norm() > 1e-9:
                    tang = tang.normalized()
                    bit = (sv * du1 - su * du2) * inv_det
                    bit = bit - nb * nb.dot(bit) - tang * tang.dot(bit)
                    if bit.norm() > 1e-9:
                        bit = bit.normalized()
                        pert = tang * tn[0] + bit * tn[1] + nb * tn[2]
                        if pert.norm() > 1e-9:
                            normal = pert.normalized()
    return normal


@ti.func
def _refract_ray(rd, n_out, ior):
    """Direction of the transmitted ray for incident unit direction ``rd``
    crossing a surface with outward unit normal ``n_out`` and index of
    refraction ``ior`` (relative to air). Snell's law, with the air<->medium
    side chosen from the sign of ``rd . n_out`` (entering when the ray opposes
    the outward normal, exiting otherwise). On total internal reflection the
    ray is mirror-reflected instead, so it always continues sensibly."""
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


# ---------------------------------------------------------------------------
# Ray-continuation (scatter) contract + built-in default. Shared by the
# monolithic ``wavefront_shade`` and the sorted ``wf_shade_event`` (which
# imports these), so both express bouncing through the same funcs. See the
# scatter contract in ``shading_taichi`` for the argument/return meaning.
# ---------------------------------------------------------------------------


@ti.func
def default_scatter(rd, n_interp, face_n, hit_point, shaded, alpha,
                    reflectivity, ior, params: ti.template(), f, prim,
                    bounces_left, refraction: ti.template()):
    """Default ray-continuation behaviour of a shaded surface event: exactly
    the classic kernel's opacity / reflectivity / Fresnel-glass logic.

    Returns ``(contrib, pass_w, refl_orig, refl_dir, refl_w, trans_orig,
    trans_dir, trans_w)``: the premultiplied colour contribution (the caller
    adds ``weight * contrib``), the pass-through weight multiplier, and the
    origin/direction/weight of the reflected branch and of the transmitted
    (glass split) branch (a weight of 0 disables that branch).
    """
    alpha = ti.math.clamp(alpha, 0.0, 1.0)
    reflectivity = ti.math.clamp(reflectivity, 0.0, 1.0)
    if bounces_left <= 0:
        reflectivity = 0.0

    # Glass = a transparent refractive surface: Fresnel reflectance R (Schlick)
    # from the IOR + incidence angle drives the energy split (diffuse
    # alpha*(1-R) + reflected R + refracted (1-R)*(1-alpha)). Non-refractive
    # surfaces keep R = reflectivity. Compiles out when refraction is off.
    is_glass = False
    gnrm = ti.math.vec3(0.0, 0.0, 0.0)
    R = reflectivity
    if ti.static(refraction != 0):
        if (alpha < 1.0 - MIN_ALPHA) and (bounces_left > 0) \
                and (ior > 1.0 + 1e-4):
            is_glass = True
            gnrm = n_interp.normalized()
            cosi = ti.abs(rd.dot(gnrm))
            r0 = (1.0 - ior) / (1.0 + ior)
            r0 = r0 * r0
            fr = r0 + (1.0 - r0) * ti.pow(1.0 - cosi, 5.0)
            # A manual ``reflectivity`` raises the reflectance floor like a
            # mirror coating (0 = pure Fresnel glass).
            R = reflectivity + (1.0 - reflectivity) * fr

    contrib = (alpha * (1.0 - R)) * shaded
    pass_w = 0.0
    refl_w = 0.0
    trans_w = 0.0
    zero3 = ti.math.vec3(0.0, 0.0, 0.0)
    refl_dir = zero3
    refl_orig = zero3
    trans_dir = zero3
    trans_orig = zero3
    if is_glass:
        rdt = _refract_ray(rd, gnrm, ior)
        trans_dir = rdt
        trans_orig = hit_point + rdt * (10.0 * MIN_HIT_DISTANCE)
        trans_w = (1.0 - R) * (1.0 - alpha)
        # Reflect the parent ray, Fresnel-weighted (decoupled from opacity --
        # a clear glass still reflects per R).
        nref = gnrm
        if nref.dot(rd) > 0.0:
            nref = -nref
        refl_dir = (rd - 2.0 * rd.dot(nref) * nref).normalized()
        refl_orig = hit_point + nref * (10.0 * MIN_HIT_DISTANCE)
        refl_w = R
    elif (reflectivity > MIN_ALPHA) and (alpha > MIN_ALPHA):
        # Opaque / translucent mirror (no refractive index).
        n = n_interp.normalized()
        if n.dot(rd) > 0.0:
            n = -n
        refl_dir = (rd - 2.0 * rd.dot(n) * n).normalized()
        refl_orig = hit_point + n * (10.0 * MIN_HIT_DISTANCE)
        refl_w = alpha * reflectivity
    else:
        pass_w = 1.0 - alpha
    return (contrib, pass_w, refl_orig, refl_dir, refl_w,
            trans_orig, trans_dir, trans_w)


@ti.func
def _run_frag_scatter(frag_scatters: ti.template(), pid_arr: ti.template(),
                      f, prim, rd, n_interp, face_n, hit_point, shaded, alpha,
                      reflectivity, ior, params: ti.template(), bounces_left,
                      refraction: ti.template()):
    """Per-primitive ray-continuation dispatch for the monolithic shade kernel:
    pick the material's scatter func by pipeline id (``pid_arr[f, prim]``) and
    return its 8-tuple. Built-in materials and user pipelines without a custom
    scatter use :func:`default_scatter`; a user pid whose pipeline supplied a
    scatter uses it. The pid switch mirrors ``_run_frag_pipeline``; ``None``
    entries of ``frag_scatters`` (scatterless user pipelines) compile out."""
    pid = pid_arr[f % pid_arr.shape[0], prim]
    (contrib, pass_w, refl_orig, refl_dir, refl_w,
     trans_orig, trans_dir, trans_w) = default_scatter(
        rd, n_interp, face_n, hit_point, shaded, alpha, reflectivity, ior,
        params, f, prim, bounces_left, refraction)
    for pi in ti.static(range(len(frag_scatters))):
        # ``bool(func) is True`` / ``bool(None) is False`` -- avoids an ``is
        # not`` comparison node, which Taichi's AST transformer rejects even
        # inside ``ti.static``. A scatterless pipeline's None entry compiles
        # its branch (and the None "call") out.
        if ti.static(bool(frag_scatters[pi])):
            if pid == _USER_PIPELINE_BASE + pi:
                (contrib, pass_w, refl_orig, refl_dir, refl_w,
                 trans_orig, trans_dir, trans_w) = frag_scatters[pi](
                    rd, n_interp, face_n, hit_point, shaded, alpha,
                    reflectivity, ior, params, f, prim, bounces_left,
                    refraction)
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
    background (mirrors the tail of ``render_triangles_stbvh``). State is indexed
    tile-locally by ``r``; the global ray is ``ray_offset + r``."""
    pixels_per_frame = width * height
    num_rays = rs_acc.shape[0]
    for r in range(num_rays):
        g = ray_offset + r
        f_rel = g // pixels_per_frame
        p = g - f_rel * pixels_per_frame
        weight = rs_sca[r, 0]
        csum = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        for ci in ti.static(range(4)):
            bg = ti.cast(out[f_rel, p, ci], ti.f32)
            csum[ci] = rs_acc[r, ci] * 255.0 + weight * bg
        color_final = finalize_pixel_color(csum, 1.0, tonemapping, tonemap_exposure)
        for ci in ti.static(range(4)):
            if ti.static(tonemapping == 3):
                out[f_rel, p, ci] = color_final[ci]
            else:
                out[f_rel, p, ci] = ti.cast(color_final[ci], ti.u8)
        if transparent != 0:
            bg_a = ti.cast(out[f_rel, p, 4], ti.f32)
            val = (1.0 - weight) * 255.0 + weight * bg_a
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
    ``wf_finalize_aa`` averages after all ``aa^2`` passes."""
    pixels_per_frame = width * height
    num_rays = rs_acc.shape[0]
    for r in range(num_rays):
        g = ray_offset + r
        f_rel = g // pixels_per_frame
        p = g - f_rel * pixels_per_frame
        idx = f_rel * pixels_per_frame + p
        weight = rs_sca[r, 0]
        for ci in ti.static(range(4)):
            bg = ti.cast(out[f_rel, p, ci], ti.f32)
            aa_accum[idx, ci] += rs_acc[r, ci] * 255.0 + weight * bg
        if transparent != 0:
            bg_a = ti.cast(out[f_rel, p, 4], ti.f32)
            aa_accum[idx, 4] += (1.0 - weight) * 255.0 + weight * bg_a


@ti.kernel
def wf_composite_accum(
        time_start: int, width: int, height: int, transparent: int,
        ray_offset: int,
        pix_accum: ti.types.ndarray(),
        tonemapping: ti.template(), tonemap_exposure: ti.f32,
        out: ti.types.ndarray()):
    """Composite the general path's per-pixel accumulator over the pre-filled
    background. Mirrors ``wf_composite`` arithmetic exactly, but reads the shared
    ``pix_accum`` (premultiplied colour cols 0-3 + summed leftover/background
    weight col 4, deposited by every terminating ray) instead of one ray slot --
    so a pixel whose ray split into reflected + refracted branches sums both. For
    a non-split pixel ``pix_accum[r] == (acc, weight)`` of its lone ray, so the
    result is byte-identical to ``wf_composite``. Indexed by local pixel ``r``;
    the global cell is ``ray_offset + r``."""
    pixels_per_frame = width * height
    num_primary = pix_accum.shape[0]
    for r in range(num_primary):
        g = ray_offset + r
        f_rel = g // pixels_per_frame
        p = g - f_rel * pixels_per_frame
        weight = pix_accum[r, 4]
        csum = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        for ci in ti.static(range(4)):
            bg = ti.cast(out[f_rel, p, ci], ti.f32)
            csum[ci] = pix_accum[r, ci] * 255.0 + weight * bg
        color_final = finalize_pixel_color(csum, 1.0, tonemapping, tonemap_exposure)
        for ci in ti.static(range(4)):
            if ti.static(tonemapping == 3):
                out[f_rel, p, ci] = color_final[ci]
            else:
                out[f_rel, p, ci] = ti.cast(color_final[ci], ti.u8)
        if transparent != 0:
            bg_a = ti.cast(out[f_rel, p, 4], ti.f32)
            val = (1.0 - weight) * 255.0 + weight * bg_a
            out[f_rel, p, 4] = ti.cast(
                ti.math.clamp(val + 0.5, 0.0, 255.0), ti.u8)


@ti.kernel
def wf_composite_accum_aa(
        time_start: int, width: int, height: int, transparent: int,
        ray_offset: int,
        pix_accum: ti.types.ndarray(), out: ti.types.ndarray(),
        aa_accum: ti.types.ndarray()):
    """Like ``wf_composite_accum`` but accumulates into a float buffer for
    in-place AA averaging."""
    pixels_per_frame = width * height
    num_primary = pix_accum.shape[0]
    for r in range(num_primary):
        g = ray_offset + r
        f_rel = g // pixels_per_frame
        p = g - f_rel * pixels_per_frame
        idx = f_rel * pixels_per_frame + p
        weight = pix_accum[r, 4]
        for ci in ti.static(range(4)):
            bg = ti.cast(out[f_rel, p, ci], ti.f32)
            aa_accum[idx, ci] += pix_accum[r, ci] * 255.0 + weight * bg
        if transparent != 0:
            bg_a = ti.cast(out[f_rel, p, 4], ti.f32)
            aa_accum[idx, 4] += (1.0 - weight) * 255.0 + weight * bg_a


@ti.kernel
def wf_finalize_aa(
        width: int, height: int, transparent: int,
        inv_samples: float,
        tonemapping: ti.template(), tonemap_exposure: ti.f32,
        aa_accum: ti.types.ndarray(), out: ti.types.ndarray()):
    """Average the AA float accumulator and write the final uint8 output.
    Called once after all ``aa^2`` sub-pixel passes have been accumulated by
    ``wf_composite_aa`` or ``wf_composite_accum_aa``."""
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
# Same stage split as the triangle path, but the traverse stage reuses the
# general ``_collect_hits`` (all three BVHs + the Matrix Pencil PN solver) and the
# shade stage replicates ``render_scene_stbvh``'s per-type drain loop. State
# carries an extra scalar, base_dist (rs_sca column 4), used by the bezier
# screen-constant border width and accumulated across mirror bounces.
# ---------------------------------------------------------------------------


@ti.kernel
def wavefront_generate_rays(
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_screen_w: float, half_screen_h: float, max_bounces: int,
        ray_offset: int, num_primary: int, jitter_x: float, jitter_y: float,
        near_clip: ti.f32,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
        rs_int: ti.types.ndarray(),
        rs_pix: ti.types.ndarray(), pix_accum: ti.types.ndarray(),
        rs_used: ti.types.ndarray()):
    """Initialise the ray pool (general path: rs_sca has a 5th column, base_dist).

    Slots ``[0, num_primary)`` are the one-per-pixel primary rays. The rest are a
    free pool, partitioned into a per-pixel sub-block of ``splits_per_pixel``
    spare slots (pixel ``p`` owns ``[num_primary + p*spp, num_primary +
    (p+1)*spp)``), handed out by ``wf_shade_general`` when a glass surface splits
    a ray. The allocation counter ``rs_used`` is *per pixel* (one entry per
    primary), so split threads on different pixels bump different addresses --
    no single global atomic to serialise on. Each ray records its target pixel
    in ``rs_pix``; the primary for slot ``r`` owns local pixel ``r`` (global cell
    ``ray_offset + r``). Per-pixel premultiplied colour + background weight
    accumulate in ``pix_accum`` (5 cols) as rays terminate, so split branches
    sharing a pixel sum correctly."""
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
            for k in ti.static(range(4)):
                rs_acc[r, k] = 0.0
            rs_sca[r, 0] = 1.0     # weight
            rs_sca[r, 1] = 0.0     # t_prev
            rs_sca[r, 2] = 1e30    # layer_prev
            rs_sca[r, 3] = -1e30   # seam_t
            rs_sca[r, 4] = t_near  # base_dist
            rs_int[r, 0] = max_bounces
            rs_int[r, 1] = 0
            rs_int[r, 2] = _ACTIVE
            rs_int[r, 3] = 0
            rs_pix[r] = r
            rs_used[r] = 0
            for k in ti.static(range(5)):
                pix_accum[r, k] = 0.0
        else:
            # Free pool slot: inactive until a split hands it out.
            rs_int[r, 2] = _DONE


@ti.kernel
def wavefront_traverse(
        active: ti.types.ndarray(), num_active: int,
        t_nodes: NODE_ARG, t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int, tri_pos: ti.types.ndarray(),
        p_nodes: NODE_ARG, p_node_miss: ti.types.ndarray(),
        p_leaf_prim: ti.types.ndarray(), p_leaf_tspan: ti.types.ndarray(),
        p_first_leaf: int, pn_ctrl: ti.types.ndarray(),
        pn_obb: ti.types.ndarray(),
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
        op_nodes: NODE_ARG, op_node_miss: ti.types.ndarray(),
        op_leaf_prim: ti.types.ndarray(), op_leaf_tspan: ti.types.ndarray(),
        op_first_leaf: int,
        ob_nodes: NODE_ARG, ob_node_miss: ti.types.ndarray(),
        ob_leaf_prim: ti.types.ndarray(), ob_leaf_tspan: ti.types.ndarray(),
        ob_first_leaf: int,
        pixel_world_scale: ti.types.ndarray(),
        layer_offset_triangles: float, layer_offset_pn: float,
        has_tri: ti.template(), has_pn: ti.template(), has_bez: ti.template(),
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
    its pixel; the global cell is ``ray_offset + rs_pix[r]``."""
    pixels_per_frame = width * height
    for i in range(num_active):
        r = active[i]
        ro = ti.math.vec3(0.0, 0.0, 0.0)
        rd = ti.math.vec3(0.0, 0.0, 0.0)
        t_prev = 0.0
        layer_prev = 1e30
        base_dist = 0.0
        f = 0
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
            f = time_start + (ray_offset + rs_pix[r]) // pixels_per_frame
        inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                              _safe_inverse(rd[2]))
        ff = ti.cast(f, ti.f32)
        pixel_size_per_t = pixel_world_scale[f]

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
                has_tri, has_pn, has_bez,
                ro, rd, inv_rd, f, ff, t_prev, layer_prev, 1e30,
                pixel_size_per_t, base_dist, layer_offset_triangles,
                layer_offset_pn,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                t_first_leaf, tri_pos,
                p_nodes, p_node_miss, p_leaf_prim, p_leaf_tspan,
                p_first_leaf, pn_ctrl, pn_obb,
                b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                b_first_leaf, circuit_meta, edges_2d, edge_accel)
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
                        has_tri, has_pn, has_bez,
                        ro, rd, inv_rd, f, ff, t_prev, layer_prev, 1e30,
                        pixel_size_per_t, base_dist, layer_offset_triangles,
                        layer_offset_pn,
                        ot_nodes, ot_node_miss, ot_leaf_prim,
                        ot_leaf_tspan, ot_first_leaf, tri_pos,
                        op_nodes, op_node_miss, op_leaf_prim,
                        op_leaf_tspan, op_first_leaf, pn_ctrl, pn_obb,
                        ob_nodes, ob_node_miss, ob_leaf_prim,
                        ob_leaf_tspan, ob_first_leaf, circuit_meta,
                        edges_2d, edge_accel)
                if opq_found == 0:
                    initial_opq_t = 1e30
                    initial_opq_layer = -1e30
            num_hits = _collect_hits(
                ro, rd, inv_rd, f, ff, t_prev, layer_prev,
                pixel_size_per_t, base_dist, layer_offset_triangles,
                layer_offset_pn,
                kb_t, kb_layer, kb_prim, kb_flags, kb_a, kb_b,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
                tri_pos,
                p_nodes, p_node_miss, p_leaf_prim, p_leaf_tspan, p_first_leaf,
                pn_ctrl, pn_obb,
                b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan, b_first_leaf,
                circuit_meta, edges_2d, edge_accel, has_tri, has_pn, has_bez,
                initial_opq_t, initial_opq_layer)
        rs_int[r, 3] = num_hits
        # num_hits == 0 leaves the ray _ACTIVE (not _DONE) so wf_shade_general
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
def wavefront_shadow(
        active: ti.types.ndarray(), num_active: int,
        t_nodes: NODE_ARG, t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int,
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_colors: ti.types.ndarray(), tri_uvs: ti.types.ndarray(),
        tri_tex_meta: ti.types.ndarray(), textures: ti.types.ndarray(),
        num_colored_triangles: ti.i32,
        p_nodes: NODE_ARG, p_node_miss: ti.types.ndarray(),
        p_leaf_prim: ti.types.ndarray(), p_leaf_tspan: ti.types.ndarray(),
        p_first_leaf: int,
        pn_ctrl: ti.types.ndarray(), pn_norm: ti.types.ndarray(),
        pn_extra: ti.types.ndarray(),
        pn_colors: ti.types.ndarray(), pn_obb: ti.types.ndarray(),
        b_nodes: NODE_ARG, b_node_miss: ti.types.ndarray(),
        b_leaf_prim: ti.types.ndarray(), b_leaf_tspan: ti.types.ndarray(),
        b_first_leaf: int,
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(),
        layer_offset_triangles: float, layer_offset_pn: float,
        has_tri: ti.template(), has_pn: ti.template(), has_bez: ti.template(),
        light_pos: ti.types.ndarray(), num_lights: int,
        time_start: int, width: int, height: int, ray_offset: int,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_sca: ti.types.ndarray(), rs_int: ti.types.ndarray(),
        rs_kt: ti.types.ndarray(), rs_ka: ti.types.ndarray(),
        rs_kb: ti.types.ndarray(), rs_kp: ti.types.ndarray(),
        rs_kf: ti.types.ndarray(), rs_pix: ti.types.ndarray(),
        rs_vis: ti.types.ndarray()):
    """Deferred binary hard-shadow stage for the general wavefront: for each
    active ray, precompute per-(K-buffer hit, light) occlusion into a packed
    int32 (bit ``q * MAX_SHADOW_LIGHTS + li``). Run between traverse and shade so
    the shade kernel reads visibility bits instead of inlining the heavy
    ``_shadow_occluded`` -> ``_nearest_surface_g`` -> PN-solver call graph
    (register-pressure relief -> higher shade-kernel occupancy). The per-hit
    shadow geometry mirrors ``wf_shade_general``'s inline block exactly, so the
    bits drive byte-identical shading. Because ``_collect_hits`` stops gathering
    at the first opaque hit, the K-buffer holds (almost) exactly the hits shade
    consumes, so few bits are computed and never read."""
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
                    prim = rs_kp[r, q]
                    if prim >= 0:
                        htype = rs_kf[r, q] & 3
                        if (htype == 1) or (htype == 2):
                            a = rs_ka[r, q]
                            b = rs_kb[r, q]
                            t_hit = rs_kt[r, q]
                            snrm = ti.math.vec3(0.0, 0.0, 0.0)
                            fnrm = ti.math.vec3(0.0, 0.0, 0.0)
                            if htype == 1:
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
                            else:
                                snrm = _pn_hit_normal(f, prim, a, b, pn_norm,
                                                      pn_ctrl, pn_extra,
                                                      textures)
                                tp = f % pn_ctrl.shape[0]
                                su = ti.math.vec3(0.0, 0.0, 0.0)
                                sv = ti.math.vec3(0.0, 0.0, 0.0)
                                for ci in ti.static(range(3)):
                                    su[ci] = (pn_ctrl[tp, prim, 3 + ci]
                                              + 2.0 * a * pn_ctrl[tp, prim,
                                                                  9 + ci]
                                              + b * pn_ctrl[tp, prim, 15 + ci])
                                    sv[ci] = (pn_ctrl[tp, prim, 6 + ci]
                                              + 2.0 * b * pn_ctrl[tp, prim,
                                                                  12 + ci]
                                              + a * pn_ctrl[tp, prim, 15 + ci])
                                fnrm = su.cross(sv)
                            if snrm.norm() > 1e-9:
                                snrm = snrm.normalized()
                            if snrm.dot(rd) > 0.0:
                                snrm = -snrm
                            if fnrm.norm() > 1e-9:
                                fnrm = fnrm.normalized()
                            if fnrm.dot(snrm) < 0.0:
                                fnrm = -fnrm
                            spos = ro + t_hit * rd
                            sorigin = spos + fnrm * (10.0 * MIN_HIT_DISTANCE)
                            for li in range(num_lights):
                                if li < _DEFERRED_SHADOW_LIGHTS:
                                    lp = ti.math.vec3(light_pos[tl, li, 0],
                                                      light_pos[tl, li, 1],
                                                      light_pos[tl, li, 2])
                                    to_light = lp - spos
                                    ldist = to_light.norm()
                                    if ldist > 1e-5:
                                        wi = to_light / ldist
                                        if (fnrm.dot(wi) > 1e-3) and \
                                                (snrm.dot(wi) > 1e-4):
                                            occ = _shadow_occluded(
                                                sorigin, wi, f, ff,
                                                ldist - 20.0 * MIN_HIT_DISTANCE,
                                                pixel_size_per_t, base_dist,
                                                layer_offset_triangles,
                                                layer_offset_pn,
                                                has_tri, has_pn, has_bez,
                                                t_nodes, t_node_miss,
                                                t_leaf_prim, t_leaf_tspan,
                                                t_first_leaf, tri_pos,
                                                tri_colors, tri_uvs,
                                                tri_tex_meta, textures,
                                                num_colored_triangles,
                                                p_nodes, p_node_miss,
                                                p_leaf_prim, p_leaf_tspan,
                                                p_first_leaf, pn_ctrl, pn_obb,
                                                pn_colors,
                                                b_nodes, b_node_miss,
                                                b_leaf_prim, b_leaf_tspan,
                                                b_first_leaf, circuit_meta,
                                                circuit_colors,
                                                circuit_border_colors,
                                                edges_2d, edge_accel)
                                            if occ > 0.5:
                                                bits |= (
                                                    1 << (q
                                                          * _DEFERRED_SHADOW_LIGHTS
                                                          + li))
        rs_vis[r] = bits


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
        p_nodes: NODE_ARG, p_node_miss: ti.types.ndarray(),
        p_leaf_prim: ti.types.ndarray(), p_leaf_tspan: ti.types.ndarray(),
        p_first_leaf: int,
        pn_ctrl: ti.types.ndarray(), pn_norm: ti.types.ndarray(),
        pn_extra: ti.types.ndarray(), pn_colors: ti.types.ndarray(),
        pn_obb: ti.types.ndarray(),
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
        # Fragment shading + binary hard shadows (compile-time templates, both
        # 0 on the default vertex-shaded path so the whole block below compiles
        # out -- byte-identical to the megakernel's vertex path) and their data.
        # ``refraction`` (also compile-time) enables Snell-law bending of the
        # transmitted ray for surfaces with a refractive index (extra cols 6-8).
        frag_shading: ti.template(), frag_pipelines: ti.template(),
        frag_scatters: ti.template(),
        shadows: ti.template(),
        refraction: ti.template(),
        has_tri: ti.template(), has_pn: ti.template(), has_bez: ti.template(),
        deferred_shadows: ti.template(),
        skip_unlit_normal: ti.template(),
        mem_trim: ti.template(),
        opaque_closest: ti.template(),
        # Fused generation's first host iteration (see wavefront_traverse's
        # ``gen_first``): the initial per-ray state was never materialised, so
        # it is used as compile-time constants here (acc = 0, weight = 1,
        # t_prev = 0, layer_prev = 1e30, seam_t = -1e30, base_dist = 0,
        # processed = 0, pix = r) instead of read from global state;
        # max_bounces rides in layer_offsets[7] (this kernel is at the CUDA
        # 64-arg ceiling). Survivors write their state back below exactly as
        # before (plus rs_pix), so iterations >= 1 run the classic kernel.
        first_iter: ti.template(),
        tri_mat_id: ti.types.ndarray(), tri_mat: ti.types.ndarray(),
        pn_mat_id: ti.types.ndarray(), pn_mat: ti.types.ndarray(),
        light_pos: ti.types.ndarray(), light_col: ti.types.ndarray(),
        num_lights: int,
        time_start: int, width: int, height: int, ray_offset: int,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
        rs_int: ti.types.ndarray(),
        rs_kt: ti.types.ndarray(), rs_kl: ti.types.ndarray(),
        rs_ka: ti.types.ndarray(), rs_kb: ti.types.ndarray(),
        rs_kp: ti.types.ndarray(), rs_kf: ti.types.ndarray(),
        rs_pix: ti.types.ndarray(), pix_accum: ti.types.ndarray(),
        rs_used: ti.types.ndarray(), rs_vis: ti.types.ndarray()):
    """Drain gathered hits front-to-back exactly as ``render_scene_stbvh``'s
    inner loop, with per-geometry-type shading and mirror bounces.

    When ``frag_shading`` is enabled, triangle/PN hits are material-shaded per
    fragment from the raw albedo (bezier circuits keep their sampled colour),
    and when ``shadows`` is also enabled each such fragment fires one binary
    shadow ray per light through all three BVHs -- the same per-fragment
    lighting/shadow model the megakernel runs in ``_trace_scene_ray``.

    When ``refraction`` is enabled, a transparent refractive surface (glass)
    reflects AND refracts at once (Fresnel reflectance ``R`` from the IOR +
    incidence angle): the reflected branch continues in this ray slot while the
    refracted (transmitted) branch is spawned into the pixel's spare sub-block
    (``num_primary + pix*splits_per_pixel + rs_used[pix]++`` -- a *per-pixel*
    counter, so split threads on different pixels never contend on one global
    address) to be picked up next host iteration. Every ray commits its colour +
    leftover background weight into ``pix_accum`` (via its ``rs_pix`` pixel) when
    it terminates, so the reflected and refracted branches sum correctly."""
    pixels_per_frame = width * height
    # Derived from array shapes (avoids two extra kernel args -- CUDA caps at 64):
    # pix_accum has one row per pixel; the pool is num_primary * split_k slots.
    num_primary = pix_accum.shape[0]
    splits_per_pixel = rs_ro.shape[0] // num_primary - 1
    # Unpack the two layer offsets (packed into one ndarray to stay within the
    # 64-arg ceiling); the body below references these names unchanged.
    layer_offset_triangles = layer_offsets[0]
    layer_offset_pn = layer_offsets[1]
    # Optional extras ride behind the two layer offsets in the same packed
    # ndarray (again: 64-arg ceiling): [2..5] = environment map placement
    # (offset, width, height, intensity) in the shared texel buffer -- rays
    # that retire without consuming all their throughput pick up the
    # environment in their final direction (skybox + correct reflections) --
    # and [6] = the camera's far clip distance (0 = disabled).
    env_off = 0
    env_w = 0
    env_h = 0
    env_intensity = 0.0
    far_clip = 0.0
    if layer_offsets.shape[0] > 6:
        env_off = ti.cast(layer_offsets[2] + 0.5, ti.i32)
        env_w = ti.cast(layer_offsets[3] + 0.5, ti.i32)
        env_h = ti.cast(layer_offsets[4] + 0.5, ti.i32)
        env_intensity = layer_offsets[5]
        far_clip = layer_offsets[6]
    for i in range(num_active):
        r = active[i]
        pix = r
        if ti.static(first_iter == 0):
            pix = rs_pix[r]
        num_hits = rs_int[r, 3]
        if num_hits > 0:
            f = time_start + (ray_offset + pix) // pixels_per_frame
            ro = ti.math.vec3(rs_ro[r, 0], rs_ro[r, 1], rs_ro[r, 2])
            rd = ti.math.vec3(rs_rd[r, 0], rs_rd[r, 1], rs_rd[r, 2])
            acc = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
            weight = 1.0
            t_prev = 0.0
            layer_prev = 1e30
            seam_t = -1e30
            base_dist = 0.0
            bounces_left = 0
            processed = 0
            if ti.static(first_iter != 0):
                bounces_left = ti.cast(layer_offsets[7] + 0.5, ti.i32)
            else:
                acc = ti.math.vec4(rs_acc[r, 0], rs_acc[r, 1], rs_acc[r, 2],
                                   rs_acc[r, 3])
                weight = rs_sca[r, 0]
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
                kb_t[q] = rs_kt[r, q]
                kb_layer[q] = rs_kl[r, q]
                kb_prim[q] = rs_kp[r, q]
                kb_flags[q] = rs_kf[r, q]
                kb_a[q] = rs_ka[r, q]
                kb_b[q] = rs_kb[r, q]

            bounced = False
            done = False
            drained = 0
            ro_seg = ro
            rd_seg = rd
            inv_rd_seg = ti.math.vec3(_safe_inverse(rd[0]),
                                      _safe_inverse(rd[1]),
                                      _safe_inverse(rd[2]))
            weight_seg = weight
            t_seg_end = 0.0
            ff = ti.cast(f, ti.f32)
            while drained < num_hits:
                sel = 0
                sel_found = 0
                for q in ti.static(range(KBUF)):
                    if (q < num_hits) and (kb_prim[q] >= 0):
                        if sel_found == 0:
                            sel = q
                            sel_found = 1
                        elif _comes_after(kb_t[sel], kb_layer[sel],
                                          kb_t[q], kb_layer[q]):
                            sel = q
                t_hit = kb_t[sel]
                t_seg_end = t_hit
                hit_layer = kb_layer[sel]
                prim = kb_prim[sel]
                flags = kb_flags[sel]
                a = kb_a[sel]
                b = kb_b[sel]
                if (far_clip > 0.0) and (base_dist + t_hit > far_clip):
                    # Past the camera's far distance. Hits drain front-to-back,
                    # so everything left is farther still -- retire the ray to
                    # the background/environment.
                    done = True
                    break
                kb_prim[sel] = -1
                drained += 1
                processed += 1
                htype = flags & 3
                edge_hit = (flags >> 2) & 1
                border = (flags >> 3) & 1

                seam_eps = PN_SEAM_DEPTH_EPSILON if htype == 2 \
                    else DEPTH_TIE_EPSILON
                if (edge_hit == 1) and (t_hit - seam_t <= seam_eps):
                    t_prev = t_hit
                    layer_prev = hit_layer
                    continue
                seam_t = t_hit if edge_hit == 1 else -1e30

                color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                alpha = 0.0
                reflectivity = 0.0
                if htype == 1:
                    w0 = 1.0 - a - b
                    color, alpha = _tri_color_g(mem_trim, f, prim, w0, a, b,
                                                tri_colors, col_row, tri_uvs,
                                                tri_tex_meta, textures,
                                                num_colored_triangles)
                    reflectivity, _rough = _tri_extra_g(
                        mem_trim, f, prim, w0, a, b, tri_extra, col_row,
                        tri_uvs, tri_tex_meta, textures, num_colored_triangles)
                elif htype == 2:
                    w0 = 1.0 - a - b
                    color, alpha = _pn_hit_color(f, prim, w0, a, b,
                                                 pn_colors, pn_extra, textures)
                    reflectivity, _rough = _pn_hit_extra(f, prim, w0, a, b,
                                                         pn_extra, textures)
                else:
                    color, alpha = _sample_circuit_color(
                        prim, f, a, b, border,
                        circuit_meta, circuit_colors, circuit_border_colors)

                # Fragment shading: ``color`` arrived as the interpolated raw
                # albedo for triangle/PN hits; evaluate the lighting model per
                # fragment. Bezier circuits (htype 0) keep their sampled colour.
                # Compiled out entirely on the default (vertex-shaded) path via
                # ti.static -- identical to ``render_scene_stbvh``.
                if ti.static(frag_shading != 0):
                    # Per-light shadow visibility for this hit (all-lit unless a
                    # binary shadow ray finds an opaque blocker). Compiled out
                    # when shadows are off; only triangle/PN hits cast/receive
                    # shadows. The light loop is a *runtime* loop (not
                    # ti.static-unrolled) so the heavy ``_shadow_occluded`` ->
                    # ``_nearest_surface`` -> PN solver call graph is inlined
                    # once, not once per light.
                    vis = ti.Vector([1.0] * MAX_SHADOW_LIGHTS)
                    if ti.static((shadows != 0) and (deferred_shadows != 0)):
                        # Deferred shadows: read the per-(hit, light) occlusion
                        # bits precomputed by ``wf_shadow_general`` for this hit's
                        # K-buffer slot (``sel``). Byte-identical to the inline
                        # path below; just relocated to a lean kernel.
                        sbits = rs_vis[r]
                        for li in range(num_lights):
                            if li < _DEFERRED_SHADOW_LIGHTS:
                                if ((sbits
                                     >> (sel * _DEFERRED_SHADOW_LIGHTS + li))
                                        & 1) != 0:
                                    vis[li] = 0.0
                    if ti.static((shadows != 0) and (deferred_shadows == 0)):
                        ff = ti.cast(f, ti.f32)
                        pixel_size_per_t = pixel_world_scale[f]
                        if (htype == 1) or (htype == 2):
                            # Smooth shading normal and the *geometric* face
                            # normal of the hit facet/patch.
                            snrm = ti.math.vec3(0.0, 0.0, 0.0)
                            fnrm = ti.math.vec3(0.0, 0.0, 0.0)
                            if htype == 1:
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
                            else:
                                snrm = _pn_hit_normal(f, prim, a, b, pn_norm,
                                                      pn_ctrl, pn_extra,
                                                      textures)
                                tp = f % pn_ctrl.shape[0]
                                su = ti.math.vec3(0.0, 0.0, 0.0)
                                sv = ti.math.vec3(0.0, 0.0, 0.0)
                                for ci in ti.static(range(3)):
                                    su[ci] = (pn_ctrl[tp, prim, 3 + ci]
                                              + 2.0 * a * pn_ctrl[tp, prim,
                                                                  9 + ci]
                                              + b * pn_ctrl[tp, prim, 15 + ci])
                                    sv[ci] = (pn_ctrl[tp, prim, 6 + ci]
                                              + 2.0 * b * pn_ctrl[tp, prim,
                                                                  12 + ci]
                                              + a * pn_ctrl[tp, prim, 15 + ci])
                                fnrm = su.cross(sv)
                            if snrm.norm() > 1e-9:
                                snrm = snrm.normalized()
                            if snrm.dot(rd) > 0.0:
                                snrm = -snrm
                            # Orient the geometric normal outward (same
                            # hemisphere as the shading normal) so a shadow ray
                            # fired near the terminator doesn't graze the
                            # adjacent uphill facet and report a spurious
                            # self-shadow. PN patches are curved (fnrm ~ snrm).
                            if fnrm.norm() > 1e-9:
                                fnrm = fnrm.normalized()
                            if fnrm.dot(snrm) < 0.0:
                                fnrm = -fnrm
                            spos = ro + t_hit * rd
                            sorigin = spos + fnrm * (10.0 * MIN_HIT_DISTANCE)
                            tl = f % light_pos.shape[0]
                            for li in range(num_lights):
                                if li < MAX_SHADOW_LIGHTS:
                                    lp = ti.math.vec3(light_pos[tl, li, 0],
                                                      light_pos[tl, li, 1],
                                                      light_pos[tl, li, 2])
                                    # Extended light rows carry a type id and a
                                    # soft-shadow radius; the compact 3-column
                                    # packing (plain point lights) keeps the
                                    # original single-ray path bit-for-bit.
                                    ltype = 0
                                    radius = 0.0
                                    if light_col.shape[2] > 3:
                                        ltype = ti.cast(
                                            light_col[tl, li, 3] + 0.5, ti.i32)
                                        radius = light_col[tl, li, 11]
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
                                            aref = ti.math.vec3(1.0, 0.0, 0.0)
                                            if ti.abs(wi[0]) > 0.9:
                                                aref = ti.math.vec3(
                                                    0.0, 1.0, 0.0)
                                            b1 = wi.cross(aref).normalized()
                                            b2 = wi.cross(b1)
                                        occ_sum = 0.0
                                        n_valid = 0.0
                                        for s in range(ns):
                                            wis = wi
                                            ldn = ldist
                                            ok = 1
                                            if radius > 0.0:
                                                ang = _GOLDEN_ANGLE * s
                                                rr = radius * ti.sqrt(
                                                    (ti.cast(s, ti.f32) + 0.5)
                                                    / ti.cast(ns, ti.f32))
                                                off = (ti.cos(ang) * b1
                                                       + ti.sin(ang) * b2) * rr
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
                                            # anyway).
                                            if (ok == 1) \
                                                    and (fnrm.dot(wis) > 1e-3) \
                                                    and (snrm.dot(wis) > 1e-4):
                                                n_valid += 1.0
                                                occ_sum += _shadow_occluded(
                                                    sorigin, wis, f, ff,
                                                    ldn - 20.0
                                                    * MIN_HIT_DISTANCE,
                                                    pixel_size_per_t, base_dist,
                                                    layer_offset_triangles,
                                                    layer_offset_pn,
                                                    has_tri, has_pn, has_bez,
                                                    t_nodes, t_node_miss,
                                                    t_leaf_prim, t_leaf_tspan,
                                                    t_first_leaf, tri_pos,
                                                    tri_colors, tri_uvs,
                                                    tri_tex_meta, textures,
                                                    num_colored_triangles,
                                                    p_nodes, p_node_miss,
                                                    p_leaf_prim, p_leaf_tspan,
                                                    p_first_leaf, pn_ctrl,
                                                    pn_obb,
                                                    pn_colors,
                                                    b_nodes, b_node_miss,
                                                    b_leaf_prim, b_leaf_tspan,
                                                    b_first_leaf, circuit_meta,
                                                    circuit_colors,
                                                    circuit_border_colors,
                                                    edges_2d, edge_accel)
                                        if n_valid > 0.0:
                                            vis[li] = 1.0 - occ_sum / n_valid
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
                        color = _shade_tri_hit(frag_pipelines, f, prim, a, b, rd,
                                               t_hit, ro, tri_pos, sn,
                                               tri_mat_id, tri_mat,
                                               light_pos, light_col, num_lights,
                                               color, shadows, vis)
                    elif htype == 2:
                        sn = ti.math.vec3(0.0, 0.0, 0.0)
                        if ti.static(skip_unlit_normal != 0):
                            if pn_mat_id[f % pn_mat_id.shape[0], prim] \
                                    != _MID_UNLIT:
                                sn = _pn_hit_normal(f, prim, a, b, pn_norm,
                                                    pn_ctrl, pn_extra, textures)
                        else:
                            sn = _pn_hit_normal(f, prim, a, b, pn_norm, pn_ctrl,
                                                pn_extra, textures)
                        color = _shade_pn_hit(frag_pipelines, f, prim, a, b, rd,
                                              t_hit, ro, pn_ctrl, sn,
                                              pn_mat_id, pn_mat,
                                              light_pos, light_col, num_lights,
                                              color, shadows, vis)

                if ti.static(len(frag_scatters) == 0):
                    # No custom scatter present: the built-in opacity /
                    # reflectivity / Fresnel-glass continuation, inline and
                    # byte-identical to the pre-scatter kernel.
                    alpha = ti.math.clamp(alpha, 0.0, 1.0)
                    reflectivity = ti.math.clamp(reflectivity, 0.0, 1.0)
                    if bounces_left <= 0:
                        reflectivity = 0.0

                    # Glass = a transparent refractive surface; it reflects AND
                    # refracts at once (Fresnel). Detect it and compute the
                    # Fresnel reflectance R(theta) (Schlick) from the IOR +
                    # incidence angle; R then drives the whole energy split
                    # (diffuse alpha*(1-R) + reflected R + refracted
                    # (1-R)*(1-alpha) = 1). Non-refractive surfaces keep R =
                    # reflectivity, so this is byte-identical (and compiles out)
                    # when refraction is off.
                    is_glass = False
                    gnrm = ti.math.vec3(0.0, 0.0, 0.0)
                    ior = 1.0
                    R = reflectivity
                    if ti.static(refraction != 0):
                        if (alpha < 1.0 - MIN_ALPHA) and (bounces_left > 0) \
                                and ((htype == 1) or (htype == 2)):
                            if htype == 1:
                                ior = _tri_ior_g(
                                    mem_trim, f, prim, 1.0 - a - b, a, b,
                                    tri_extra, col_row, tri_uvs, tri_tex_meta,
                                    textures, num_colored_triangles)
                            else:
                                ior = _pn_hit_ior(f, prim, 1.0 - a - b, a, b,
                                                  pn_extra, textures)
                            if ior > 1.0 + 1e-4:
                                is_glass = True
                                if htype == 1:
                                    gnrm = _tri_normal_g(
                                        mem_trim, f, prim, 1.0 - a - b, a, b,
                                        tri_norm, tri_pos, tri_uvs,
                                        tri_tex_meta, textures,
                                        num_colored_triangles)
                                else:
                                    gnrm = _pn_hit_normal(f, prim, a, b,
                                                          pn_norm, pn_ctrl,
                                                          pn_extra, textures)
                                gnrm = gnrm.normalized()
                                # Fresnel of the dielectric (Schlick): R0 =
                                # ((1-ior)/(1+ior))^2, fr = R0+(1-R0)(1-cos)^5.
                                cosi = ti.abs(rd.dot(gnrm))
                                r0 = (1.0 - ior) / (1.0 + ior)
                                r0 = r0 * r0
                                fr = r0 + (1.0 - r0) * ti.pow(1.0 - cosi, 5.0)
                                # A manual ``reflectivity`` raises the
                                # reflectance floor like a mirror coating: 0 =
                                # pure Fresnel glass; ~1 = a near-perfect mirror
                                # that still refracts the tiny remainder.
                                R = reflectivity + (1.0 - reflectivity) * fr

                    acc += (weight * alpha * (1.0 - R)) * color

                    if is_glass:
                        # Split: spawn the refracted (transmitted) branch into
                        # this pixel's spare sub-block -- the allocation counter
                        # is per pixel (rs_used[pix]), so different pixels bump
                        # different addresses (no single global atomic). The
                        # reflected branch continues in this slot; both commit
                        # to the same pixel.
                        wt = weight * (1.0 - R) * (1.0 - alpha)
                        if wt > MIN_WEIGHT:
                            c_local = ti.atomic_add(rs_used[pix], 1)
                            if c_local < splits_per_pixel:
                                c = (num_primary + pix * splits_per_pixel
                                     + c_local)
                                rdt = _refract_ray(rd, gnrm, ior)
                                hp = ro + t_hit * rd
                                for k in ti.static(range(3)):
                                    rs_ro[c, k] = (hp[k] + rdt[k]
                                                   * (10.0 * MIN_HIT_DISTANCE))
                                    rs_rd[c, k] = rdt[k]
                                for k in ti.static(range(4)):
                                    rs_acc[c, k] = 0.0
                                rs_sca[c, 0] = wt
                                rs_sca[c, 1] = 0.0
                                rs_sca[c, 2] = 1e30
                                rs_sca[c, 3] = -1e30
                                rs_sca[c, 4] = base_dist + t_hit
                                rs_int[c, 0] = bounces_left - 1
                                rs_int[c, 1] = processed
                                rs_int[c, 2] = _ACTIVE
                                rs_int[c, 3] = 0
                                rs_pix[c] = pix
                        # Reflect the parent, Fresnel-weighted (decoupled from
                        # the opacity -- a clear glass still reflects per R).
                        nref = gnrm
                        if nref.dot(rd) > 0.0:
                            nref = -nref
                        hit_point = ro + t_hit * rd
                        rd = (rd - 2.0 * rd.dot(nref) * nref).normalized()
                        ro = hit_point + nref * (10.0 * MIN_HIT_DISTANCE)
                        weight *= R
                        base_dist += t_hit
                        t_prev = 0.0
                        layer_prev = 1e30
                        seam_t = -1e30
                        bounces_left -= 1
                        bounced = True
                        break
                    elif (reflectivity > MIN_ALPHA) and (alpha > MIN_ALPHA):
                        # Opaque / translucent mirror (no refractive index):
                        # unchanged reflection, gated by opacity as before.
                        normal = ti.math.vec3(0.0, 0.0, 0.0)
                        if htype == 1:
                            normal = _tri_normal_g(
                                mem_trim, f, prim, 1.0 - a - b, a, b, tri_norm,
                                tri_pos, tri_uvs, tri_tex_meta, textures,
                                num_colored_triangles)
                        elif htype == 2:
                            normal = _pn_hit_normal(f, prim, a, b, pn_norm,
                                                    pn_ctrl, pn_extra, textures)
                        else:
                            normal = _bezier_normal(f, prim, circuit_meta)
                        normal = normal.normalized()
                        if normal.dot(rd) > 0.0:
                            normal = -normal
                        hit_point = ro + t_hit * rd
                        rd = (rd - 2.0 * rd.dot(normal) * normal).normalized()
                        ro = hit_point + normal * (10.0 * MIN_HIT_DISTANCE)
                        weight *= alpha * reflectivity
                        base_dist += t_hit
                        t_prev = 0.0
                        layer_prev = 1e30
                        seam_t = -1e30
                        bounces_left -= 1
                        bounced = True
                        break
                    else:
                        weight *= 1.0 - alpha
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
                    elif htype == 2:
                        sni = _pn_hit_normal(f, prim, a, b, pn_norm, pn_ctrl,
                                             pn_extra, textures)
                        tp = f % pn_ctrl.shape[0]
                        su = ti.math.vec3(0.0, 0.0, 0.0)
                        sv = ti.math.vec3(0.0, 0.0, 0.0)
                        for ci in ti.static(range(3)):
                            su[ci] = (pn_ctrl[tp, prim, 3 + ci]
                                      + 2.0 * a * pn_ctrl[tp, prim, 9 + ci]
                                      + b * pn_ctrl[tp, prim, 15 + ci])
                            sv[ci] = (pn_ctrl[tp, prim, 6 + ci]
                                      + 2.0 * b * pn_ctrl[tp, prim, 12 + ci]
                                      + a * pn_ctrl[tp, prim, 15 + ci])
                        sfn = su.cross(sv)
                    else:
                        sni = _bezier_normal(f, prim, circuit_meta)
                        sfn = sni
                    s_ior = 1.0
                    if ti.static(refraction != 0):
                        if htype == 1:
                            s_ior = _tri_ior_g(
                                mem_trim, f, prim, 1.0 - a - b, a, b, tri_extra,
                                col_row, tri_uvs, tri_tex_meta, textures,
                                num_colored_triangles)
                        elif htype == 2:
                            s_ior = _pn_hit_ior(f, prim, 1.0 - a - b, a, b,
                                                pn_extra, textures)
                    hit_point = ro + t_hit * rd
                    contrib = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                    pass_w = 0.0
                    refl_orig = ti.math.vec3(0.0, 0.0, 0.0)
                    refl_dir = ti.math.vec3(0.0, 0.0, 0.0)
                    refl_w = 0.0
                    trans_orig = ti.math.vec3(0.0, 0.0, 0.0)
                    trans_dir = ti.math.vec3(0.0, 0.0, 0.0)
                    trans_w = 0.0
                    if htype == 1:
                        (contrib, pass_w, refl_orig, refl_dir, refl_w,
                         trans_orig, trans_dir, trans_w) = _run_frag_scatter(
                            frag_scatters, tri_mat_id, f, prim, rd, sni, sfn,
                            hit_point, color, alpha, reflectivity, s_ior,
                            tri_mat, bounces_left, refraction)
                    elif htype == 2:
                        (contrib, pass_w, refl_orig, refl_dir, refl_w,
                         trans_orig, trans_dir, trans_w) = _run_frag_scatter(
                            frag_scatters, pn_mat_id, f, prim, rd, sni, sfn,
                            hit_point, color, alpha, reflectivity, s_ior,
                            pn_mat, bounces_left, refraction)
                    else:
                        (contrib, pass_w, refl_orig, refl_dir, refl_w,
                         trans_orig, trans_dir, trans_w) = default_scatter(
                            rd, sni, sfn, hit_point, color, alpha,
                            reflectivity, s_ior, tri_mat, f, prim,
                            bounces_left, refraction)
                    acc += weight * contrib
                    if ti.static(refraction != 0):
                        wt = weight * trans_w
                        if (trans_w > 0.0) and (wt > MIN_WEIGHT) \
                                and (bounces_left > 0):
                            c_local = ti.atomic_add(rs_used[pix], 1)
                            if c_local < splits_per_pixel:
                                c = (num_primary + pix * splits_per_pixel
                                     + c_local)
                                for k in ti.static(range(3)):
                                    rs_ro[c, k] = trans_orig[k]
                                    rs_rd[c, k] = trans_dir[k]
                                for k in ti.static(range(4)):
                                    rs_acc[c, k] = 0.0
                                rs_sca[c, 0] = wt
                                rs_sca[c, 1] = 0.0
                                rs_sca[c, 2] = 1e30
                                rs_sca[c, 3] = -1e30
                                rs_sca[c, 4] = base_dist + t_hit
                                rs_int[c, 0] = bounces_left - 1
                                rs_int[c, 1] = processed
                                rs_int[c, 2] = _ACTIVE
                                rs_int[c, 3] = 0
                                rs_pix[c] = pix
                    if (refl_w > 0.0) and (bounces_left > 0):
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
                if weight < MIN_WEIGHT:
                    done = True
                    break

            if ti.static(opaque_closest):
                if (not done) and (not bounced):
                    done = True
            else:
                if (not done) and (not bounced) and (num_hits < KBUF):
                    done = True
            if processed >= MAX_SURFACES_PER_RAY:
                done = True

            for k in ti.static(range(3)):
                rs_ro[r, k] = ro[k]
                rs_rd[r, k] = rd[k]
            for k in ti.static(range(4)):
                rs_acc[r, k] = acc[k]
            rs_sca[r, 0] = weight
            rs_sca[r, 1] = t_prev
            rs_sca[r, 2] = layer_prev
            rs_sca[r, 3] = seam_t
            rs_sca[r, 4] = base_dist
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
                if (env_w > 0) and (weight > 0.0):
                    ec = _sample_env_map(f, rd, env_off, env_w, env_h,
                                         env_intensity, textures)
                    for k in ti.static(range(3)):
                        acc[k] += weight * ec[k]
                    weight = 0.0
                for k in ti.static(range(4)):
                    ti.atomic_add(pix_accum[pix, k], acc[k])
                ti.atomic_add(pix_accum[pix, 4], weight)
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

            w_bg = 1.0
            if ti.static(first_iter == 0):
                w_bg = rs_sca[r, 0]
            if (env_w > 0) and (w_bg > 0.0):
                ec = _sample_env_map(f, rd, env_off, env_w, env_h,
                                     env_intensity, textures)
                for k in ti.static(range(3)):
                    ti.atomic_add(pix_accum[pix, k], w_bg * ec[k])
                w_bg = 0.0
            if ti.static(first_iter == 0):
                # First iteration's accumulator is implicitly zero; adding it
                # would be a no-op, so the read is skipped entirely.
                for k in ti.static(range(4)):
                    ti.atomic_add(pix_accum[pix, k], rs_acc[r, k])
            ti.atomic_add(pix_accum[pix, 4], w_bg)
            rs_int[r, 2] = _DONE
