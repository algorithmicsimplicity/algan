"""Wavefront (stage-split) variant of the deterministic triangle ray-trace
kernel.

The megakernel ``render_triangles_stbvh`` runs the whole per-ray state machine
-- BVH traversal, hit gathering, depth-peel shading and mirror bounces -- in a
single thread, which on the GTX 1050 is occupancy-starved by register pressure
(~78 regs) and divergence-bound (rays in a warp finish their traversal/peel
loops at wildly different iteration counts; ~30% warp execution efficiency).

This module splits that state machine into small per-stage kernels connected by
per-ray state in global memory, driven by a host-side iteration loop:

* :func:`wf_gen_triangle`      -- initialise per-ray state with primary rays.
* :func:`wf_traverse_triangle` -- for each *active* ray, gather the KBUF nearest
  hits (reusing the unchanged ``_collect_hits_tri``) into global state.
* :func:`wf_shade_triangle`    -- replicate the megakernel's inner drain loop
  exactly (seam-merge, shade, mirror bounce, transmittance) on global state.
* :func:`wf_composite`         -- composite each ray's accumulator over the
  background.

Between iterations the host compacts the still-active rays with a PyTorch
``nonzero`` (see ``render_triangles_wavefront`` in ``primitives.py``), so each
launch processes only rays that still have work -- warps refill as rays drop
out, which is the divergence fix. Each stage kernel is small (few live
registers) so it runs at much higher occupancy than the megakernel.

The math is byte-for-byte the megakernel's: every intersection / shading
helper is imported unchanged from :mod:`ray_trace_taichi`; only the
*orchestration* (where state lives, and that rays advance in lockstep host
iterations rather than a per-thread ``while``) differs.
"""
import taichi as ti

from algan.rendering.raytracing.ray_trace_taichi import (
    DEPTH_TIE_EPSILON,
    KBUF,
    MAX_SHADOW_LIGHTS,
    MAX_SURFACES_PER_RAY,
    MIN_ALPHA,
    MIN_HIT_DISTANCE,
    MIN_WEIGHT,
    PN_SEAM_DEPTH_EPSILON,
    _bezier_normal,
    _collect_hits,
    _collect_hits_tri,
    _collect_hits_tri_knots,
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
    _flat_triangle_alpha,
    _triangle_extra,
    _triangle_normal,
    _accumulate_glow,
    _accumulate_glow_triangles,
    finalize_pixel_color,
)

# Per-ray status codes (rs_int column 2).
_ACTIVE = 0
_DONE = 1


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


@ti.kernel
def wf_gen_triangle(
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_screen_w: float, half_screen_h: float, max_bounces: int,
        ray_offset: int, jitter_x: float, jitter_y: float,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
        rs_int: ti.types.ndarray()):
    """Initialise per-ray state with the primary camera ray (mirrors the
    per-ray setup at the top of ``render_triangles_stbvh``). State is indexed
    tile-locally by ``r``; the global ray is ``ray_offset + r`` (screen tiling)."""
    pixels_per_frame = width * height
    num_rays = rs_ro.shape[0]
    for r in range(num_rays):
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
        for k in ti.static(range(3)):
            rs_ro[r, k] = ro[k]
            rs_rd[r, k] = rd[k]
        for k in ti.static(range(4)):
            rs_acc[r, k] = 0.0
        rs_sca[r, 0] = 1.0     # weight
        rs_sca[r, 1] = 0.0     # t_prev
        rs_sca[r, 2] = 1e30    # layer_prev
        rs_sca[r, 3] = -1e30   # seam_t
        rs_int[r, 0] = max_bounces  # bounces_left
        rs_int[r, 1] = 0            # processed
        rs_int[r, 2] = _ACTIVE      # status
        rs_int[r, 3] = 0            # num_hits


@ti.kernel
def wf_traverse_triangle(
        active: ti.types.ndarray(), num_active: int,
        t_nodes: ti.types.ndarray(), t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int, tri_pos: ti.types.ndarray(),
        layer_offset_triangles: float,
        time_start: int, width: int, height: int, ray_offset: int,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_sca: ti.types.ndarray(), rs_int: ti.types.ndarray(),
        rs_kt: ti.types.ndarray(), rs_kl: ti.types.ndarray(),
        rs_ka: ti.types.ndarray(), rs_kb: ti.types.ndarray(),
        rs_kp: ti.types.ndarray(), rs_kf: ti.types.ndarray()):
    """Gather the KBUF nearest hits for each active ray into global state.
    No shading state in its call graph -> few live registers. State is indexed
    tile-locally by the active ray; the global ray is ``ray_offset + r``."""
    pixels_per_frame = width * height
    for i in range(num_active):
        r = active[i]
        ro = ti.math.vec3(rs_ro[r, 0], rs_ro[r, 1], rs_ro[r, 2])
        rd = ti.math.vec3(rs_rd[r, 0], rs_rd[r, 1], rs_rd[r, 2])
        inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                              _safe_inverse(rd[2]))
        t_prev = rs_sca[r, 1]
        layer_prev = rs_sca[r, 2]
        f = time_start + (ray_offset + r) // pixels_per_frame
        ff = ti.cast(f, ti.f32)

        kb_t = ti.Vector([0.0] * KBUF)
        kb_layer = ti.Vector([0.0] * KBUF)
        kb_prim = ti.Vector([0] * KBUF)
        kb_flags = ti.Vector([0] * KBUF)
        kb_a = ti.Vector([0.0] * KBUF)
        kb_b = ti.Vector([0.0] * KBUF)
        num_hits = _collect_hits_tri(
            ro, rd, inv_rd, f, ff, t_prev, layer_prev,
            layer_offset_triangles,
            kb_t, kb_layer, kb_prim, kb_flags, kb_a, kb_b,
            t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
            t_first_leaf, tri_pos)
        rs_int[r, 3] = num_hits
        if num_hits > 0:
            for q in ti.static(range(KBUF)):
                rs_kt[r, q] = kb_t[q]
                rs_kl[r, q] = kb_layer[q]
                rs_kp[r, q] = kb_prim[q]
                rs_kf[r, q] = kb_flags[q]
                rs_ka[r, q] = kb_a[q]
                rs_kb[r, q] = kb_b[q]


@ti.kernel
def wf_traverse_triangle_knots(
        active: ti.types.ndarray(), num_active: int,
        t_nodes: ti.types.ndarray(), t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int,
        knot_val: ti.types.ndarray(), knot_base: ti.types.ndarray(),
        sched_id: ti.types.ndarray(), sched_seg: ti.types.ndarray(),
        sched_z: ti.types.ndarray(), sched_nknots: ti.types.ndarray(),
        layer_offset_triangles: float,
        time_start: int, width: int, height: int, ray_offset: int,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_sca: ti.types.ndarray(), rs_int: ti.types.ndarray(),
        rs_kt: ti.types.ndarray(), rs_kl: ti.types.ndarray(),
        rs_ka: ti.types.ndarray(), rs_kb: ti.types.ndarray(),
        rs_kp: ti.types.ndarray(), rs_kf: ti.types.ndarray()):
    """Knot-geometry twin of :func:`wf_traverse_triangle`: positions are
    reconstructed from the compressed knot representation
    (``_collect_hits_tri_knots``). Tests whether isolating the traverse stage --
    so the knot reconstruction's extra live registers no longer share the
    megakernel's register budget -- closes the knot megakernel's gap."""
    pixels_per_frame = width * height
    for i in range(num_active):
        r = active[i]
        ro = ti.math.vec3(rs_ro[r, 0], rs_ro[r, 1], rs_ro[r, 2])
        rd = ti.math.vec3(rs_rd[r, 0], rs_rd[r, 1], rs_rd[r, 2])
        inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                              _safe_inverse(rd[2]))
        t_prev = rs_sca[r, 1]
        layer_prev = rs_sca[r, 2]
        f = time_start + (ray_offset + r) // pixels_per_frame
        ff = ti.cast(f, ti.f32)

        kb_t = ti.Vector([0.0] * KBUF)
        kb_layer = ti.Vector([0.0] * KBUF)
        kb_prim = ti.Vector([0] * KBUF)
        kb_flags = ti.Vector([0] * KBUF)
        kb_a = ti.Vector([0.0] * KBUF)
        kb_b = ti.Vector([0.0] * KBUF)
        num_hits = _collect_hits_tri_knots(
            ro, rd, inv_rd, f, ff, t_prev, layer_prev,
            layer_offset_triangles,
            kb_t, kb_layer, kb_prim, kb_flags, kb_a, kb_b,
            t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
            t_first_leaf, knot_val, knot_base, sched_id, sched_seg,
            sched_z, sched_nknots)
        rs_int[r, 3] = num_hits
        if num_hits == 0:
            rs_int[r, 2] = _DONE
        else:
            for q in ti.static(range(KBUF)):
                rs_kt[r, q] = kb_t[q]
                rs_kl[r, q] = kb_layer[q]
                rs_kp[r, q] = kb_prim[q]
                rs_kf[r, q] = kb_flags[q]
                rs_ka[r, q] = kb_a[q]
                rs_kb[r, q] = kb_b[q]


@ti.kernel
def wf_shade_triangle(
        active: ti.types.ndarray(), num_active: int,
        t_nodes: ti.types.ndarray(), t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int,
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_extra: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        time_start: int, width: int, height: int, ray_offset: int,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
        rs_int: ti.types.ndarray(),
        rs_kt: ti.types.ndarray(), rs_kl: ti.types.ndarray(),
        rs_ka: ti.types.ndarray(), rs_kb: ti.types.ndarray(),
        rs_kp: ti.types.ndarray(), rs_kf: ti.types.ndarray()):
    """Drain the gathered hits front-to-back, blending and bouncing exactly as
    the megakernel's inner loop. Traverses STBVH for glow accumulation."""
    pixels_per_frame = width * height
    for i in range(num_active):
        r = active[i]
        num_hits = rs_int[r, 3]
        if num_hits > 0:
            f = time_start + (ray_offset + r) // pixels_per_frame
            ro = ti.math.vec3(rs_ro[r, 0], rs_ro[r, 1], rs_ro[r, 2])
            rd = ti.math.vec3(rs_rd[r, 0], rs_rd[r, 1], rs_rd[r, 2])
            acc = ti.math.vec4(rs_acc[r, 0], rs_acc[r, 1], rs_acc[r, 2],
                               rs_acc[r, 3])
            weight = rs_sca[r, 0]
            t_prev = rs_sca[r, 1]
            layer_prev = rs_sca[r, 2]
            seam_t = rs_sca[r, 3]
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
                kb_prim[sel] = -1
                drained += 1
                processed += 1
                edge_hit = (flags >> 2) & 1

                if (edge_hit == 1) and (t_hit - seam_t <= DEPTH_TIE_EPSILON):
                    t_prev = t_hit
                    layer_prev = hit_layer
                    continue
                seam_t = t_hit if edge_hit == 1 else -1e30

                w0 = 1.0 - a - b
                color, alpha = _flat_triangle_color(f, prim, w0, a, b,
                                                    tri_colors, tri_uvs, tri_tex_meta,
                                                    textures, num_colored_triangles)
                reflectivity, _rough = _triangle_extra(f, prim, w0, a, b,
                                                       tri_extra)
                alpha = ti.math.clamp(alpha, 0.0, 1.0)
                reflectivity = ti.math.clamp(reflectivity, 0.0, 1.0)
                if bounces_left <= 0:
                    reflectivity = 0.0

                acc += (weight * alpha * (1.0 - reflectivity)) * color

                if (reflectivity > MIN_ALPHA) and (alpha > MIN_ALPHA):
                    normal = _triangle_normal(f, prim, 1.0 - a - b, a, b,
                                              tri_norm, tri_pos)
                    normal = normal.normalized()
                    if normal.dot(rd) > 0.0:
                        normal = -normal
                    hit_point = ro + t_hit * rd
                    rd = (rd - 2.0 * rd.dot(normal) * normal).normalized()
                    ro = hit_point + normal * (10.0 * MIN_HIT_DISTANCE)
                    weight *= alpha * reflectivity
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
                if weight < MIN_WEIGHT:
                    done = True
                    break

            glow_rgb = _accumulate_glow_triangles(
                ro_seg, rd_seg, inv_rd_seg, t_seg_end, f, ff,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
                tri_pos, tri_colors, tri_extra, num_colored_triangles
            )
            acc[0] += weight_seg * glow_rgb[0]
            acc[1] += weight_seg * glow_rgb[1]
            acc[2] += weight_seg * glow_rgb[2]

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
            rs_int[r, 0] = bounces_left
            rs_int[r, 1] = processed
            rs_int[r, 2] = _DONE if done else _ACTIVE
        else:
            f = time_start + (ray_offset + r) // pixels_per_frame
            ff = ti.cast(f, ti.f32)
            ro = ti.math.vec3(rs_ro[r, 0], rs_ro[r, 1], rs_ro[r, 2])
            rd = ti.math.vec3(rs_rd[r, 0], rs_rd[r, 1], rs_rd[r, 2])
            inv_rd = ti.math.vec3(_safe_inverse(rd[0]),
                                  _safe_inverse(rd[1]),
                                  _safe_inverse(rd[2]))
            glow_rgb = _accumulate_glow_triangles(
                ro, rd, inv_rd, 1e30, f, ff,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
                tri_pos, tri_colors, tri_extra, num_colored_triangles
            )
            weight = rs_sca[r, 0]
            rs_acc[r, 0] += weight * glow_rgb[0]
            rs_acc[r, 1] += weight * glow_rgb[1]
            rs_acc[r, 2] += weight * glow_rgb[2]
            rs_int[r, 2] = _DONE


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
def wf_gen_general(
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_screen_w: float, half_screen_h: float, max_bounces: int,
        ray_offset: int, num_primary: int, jitter_x: float, jitter_y: float,
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
            for k in ti.static(range(3)):
                rs_ro[r, k] = ro[k]
                rs_rd[r, k] = rd[k]
            for k in ti.static(range(4)):
                rs_acc[r, k] = 0.0
            rs_sca[r, 0] = 1.0     # weight
            rs_sca[r, 1] = 0.0     # t_prev
            rs_sca[r, 2] = 1e30    # layer_prev
            rs_sca[r, 3] = -1e30   # seam_t
            rs_sca[r, 4] = 0.0     # base_dist
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
def wf_traverse_general(
        active: ti.types.ndarray(), num_active: int,
        t_nodes: ti.types.ndarray(), t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int, tri_pos: ti.types.ndarray(),
        p_nodes: ti.types.ndarray(), p_node_miss: ti.types.ndarray(),
        p_leaf_prim: ti.types.ndarray(), p_leaf_tspan: ti.types.ndarray(),
        p_first_leaf: int, pn_ctrl: ti.types.ndarray(),
        pn_obb: ti.types.ndarray(),
        b_nodes: ti.types.ndarray(), b_node_miss: ti.types.ndarray(),
        b_leaf_prim: ti.types.ndarray(), b_leaf_tspan: ti.types.ndarray(),
        b_first_leaf: int, circuit_meta: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_offsets: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(),
        layer_offset_triangles: float, layer_offset_pn: float,
        has_tri: ti.template(), has_pn: ti.template(), has_bez: ti.template(),
        time_start: int, width: int, height: int, ray_offset: int,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_sca: ti.types.ndarray(), rs_int: ti.types.ndarray(),
        rs_kt: ti.types.ndarray(), rs_kl: ti.types.ndarray(),
        rs_ka: ti.types.ndarray(), rs_kb: ti.types.ndarray(),
        rs_kp: ti.types.ndarray(), rs_kf: ti.types.ndarray(),
        rs_pix: ti.types.ndarray()):
    """Gather KBUF nearest hits across all three BVHs for each active ray
    (reuses the unchanged general ``_collect_hits``, Matrix Pencil solver
    included). The frame is taken from the ray's *pixel* (``rs_pix``), not its
    slot index -- a spawned (split) ray lives in a spare slot whose index is not
    its pixel; the global cell is ``ray_offset + rs_pix[r]``."""
    pixels_per_frame = width * height
    for i in range(num_active):
        r = active[i]
        ro = ti.math.vec3(rs_ro[r, 0], rs_ro[r, 1], rs_ro[r, 2])
        rd = ti.math.vec3(rs_rd[r, 0], rs_rd[r, 1], rs_rd[r, 2])
        inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                              _safe_inverse(rd[2]))
        t_prev = rs_sca[r, 1]
        layer_prev = rs_sca[r, 2]
        base_dist = rs_sca[r, 4]
        f = time_start + (ray_offset + rs_pix[r]) // pixels_per_frame
        ff = ti.cast(f, ti.f32)
        pixel_size_per_t = pixel_world_scale[f]

        kb_t = ti.Vector([0.0] * KBUF)
        kb_layer = ti.Vector([0.0] * KBUF)
        kb_prim = ti.Vector([0] * KBUF)
        kb_flags = ti.Vector([0] * KBUF)
        kb_a = ti.Vector([0.0] * KBUF)
        kb_b = ti.Vector([0.0] * KBUF)
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
            circuit_meta, edges_2d, edge_offsets, has_tri, has_pn, has_bez)
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
def wf_shadow_general(
        active: ti.types.ndarray(), num_active: int,
        t_nodes: ti.types.ndarray(), t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int,
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_colors: ti.types.ndarray(), tri_uvs: ti.types.ndarray(),
        tri_tex_meta: ti.types.ndarray(), textures: ti.types.ndarray(),
        num_colored_triangles: ti.i32,
        p_nodes: ti.types.ndarray(), p_node_miss: ti.types.ndarray(),
        p_leaf_prim: ti.types.ndarray(), p_leaf_tspan: ti.types.ndarray(),
        p_first_leaf: int,
        pn_ctrl: ti.types.ndarray(), pn_norm: ti.types.ndarray(),
        pn_extra: ti.types.ndarray(),
        pn_colors: ti.types.ndarray(), pn_obb: ti.types.ndarray(),
        b_nodes: ti.types.ndarray(), b_node_miss: ti.types.ndarray(),
        b_leaf_prim: ti.types.ndarray(), b_leaf_tspan: ti.types.ndarray(),
        b_first_leaf: int,
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_offsets: ti.types.ndarray(),
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
                                if li < MAX_SHADOW_LIGHTS:
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
                                                edges_2d, edge_offsets)
                                            if occ > 0.5:
                                                bits |= (
                                                    1 << (q * MAX_SHADOW_LIGHTS
                                                          + li))
        rs_vis[r] = bits


@ti.kernel
def wf_shade_general(
        active: ti.types.ndarray(), num_active: int,
        # Triangle STBVH (for shadow rays) + geometry/shading data.
        t_nodes: ti.types.ndarray(), t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int,
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_extra: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        # PN patch STBVH + geometry/shading data.
        p_nodes: ti.types.ndarray(), p_node_miss: ti.types.ndarray(),
        p_leaf_prim: ti.types.ndarray(), p_leaf_tspan: ti.types.ndarray(),
        p_first_leaf: int,
        pn_ctrl: ti.types.ndarray(), pn_norm: ti.types.ndarray(),
        pn_extra: ti.types.ndarray(), pn_colors: ti.types.ndarray(),
        pn_obb: ti.types.ndarray(),
        # Bezier STBVH + geometry/shading data.
        b_nodes: ti.types.ndarray(), b_node_miss: ti.types.ndarray(),
        b_leaf_prim: ti.types.ndarray(), b_leaf_tspan: ti.types.ndarray(),
        b_first_leaf: int,
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_offsets: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(),
        layer_offset_triangles: float, layer_offset_pn: float,
        # Fragment shading + binary hard shadows (compile-time templates, both
        # 0 on the default vertex-shaded path so the whole block below compiles
        # out -- byte-identical to the megakernel's vertex path) and their data.
        # ``refraction`` (also compile-time) enables Snell-law bending of the
        # transmitted ray for surfaces with a refractive index (extra cols 6-8).
        frag_shading: ti.template(), shadows: ti.template(),
        refraction: ti.template(),
        has_tri: ti.template(), has_pn: ti.template(), has_bez: ti.template(),
        deferred_shadows: ti.template(),
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
    for i in range(num_active):
        r = active[i]
        pix = rs_pix[r]
        num_hits = rs_int[r, 3]
        if num_hits > 0:
            f = time_start + (ray_offset + pix) // pixels_per_frame
            ro = ti.math.vec3(rs_ro[r, 0], rs_ro[r, 1], rs_ro[r, 2])
            rd = ti.math.vec3(rs_rd[r, 0], rs_rd[r, 1], rs_rd[r, 2])
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
                    color, alpha = _flat_triangle_color(f, prim, w0, a, b,
                                                        tri_colors, tri_uvs, tri_tex_meta,
                                                        textures, num_colored_triangles)
                    reflectivity, _rough = _flat_triangle_extra(
                        f, prim, w0, a, b, tri_extra, tri_uvs, tri_tex_meta,
                        textures, num_colored_triangles)
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
                            if li < MAX_SHADOW_LIGHTS:
                                if ((sbits >> (sel * MAX_SHADOW_LIGHTS + li))
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
                                    to_light = lp - spos
                                    ldist = to_light.norm()
                                    if ldist > 1e-5:
                                        wi = to_light / ldist
                                        # Skip lights below the geometric/shading
                                        # horizon (self-shadow acne / no direct
                                        # light to occlude anyway).
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
                                                edges_2d, edge_offsets)
                                            vis[li] = 1.0 - occ
                    if htype == 1:
                        color = _shade_tri_hit(f, prim, a, b, rd, t_hit, ro,
                                               tri_pos, tri_norm,
                                               tri_mat_id, tri_mat,
                                               light_pos, light_col, num_lights,
                                               color, shadows, vis)
                    elif htype == 2:
                        color = _shade_pn_hit(f, prim, a, b, rd, t_hit, ro,
                                              pn_ctrl, pn_norm,
                                              pn_mat_id, pn_mat,
                                              light_pos, light_col, num_lights,
                                              color, shadows, vis)

                alpha = ti.math.clamp(alpha, 0.0, 1.0)
                reflectivity = ti.math.clamp(reflectivity, 0.0, 1.0)
                if bounces_left <= 0:
                    reflectivity = 0.0

                # Glass = a transparent refractive surface; it reflects AND
                # refracts at once (Fresnel). Detect it and compute the Fresnel
                # reflectance R(theta) (Schlick) from the IOR + incidence angle;
                # R then drives the whole energy split (diffuse alpha*(1-R) +
                # reflected R + refracted (1-R)*(1-alpha) = 1). Non-refractive
                # surfaces keep R = reflectivity, so this is byte-identical (and
                # compiles out) when refraction is off.
                is_glass = False
                gnrm = ti.math.vec3(0.0, 0.0, 0.0)
                ior = 1.0
                R = reflectivity
                if ti.static(refraction != 0):
                    if (alpha < 1.0 - MIN_ALPHA) and (bounces_left > 0) \
                            and ((htype == 1) or (htype == 2)):
                        if htype == 1:
                            ior = _flat_corner_ior(
                                f, prim, 1.0 - a - b, a, b, tri_extra,
                                tri_uvs, tri_tex_meta, textures,
                                num_colored_triangles)
                        else:
                            ior = _pn_hit_ior(f, prim, 1.0 - a - b, a, b,
                                              pn_extra, textures)
                        if ior > 1.0 + 1e-4:
                            is_glass = True
                            if htype == 1:
                                gnrm = _flat_triangle_normal(
                                    f, prim, 1.0 - a - b, a, b, tri_norm,
                                    tri_pos, tri_uvs, tri_tex_meta, textures,
                                    num_colored_triangles)
                            else:
                                gnrm = _pn_hit_normal(f, prim, a, b, pn_norm,
                                                      pn_ctrl, pn_extra,
                                                      textures)
                            gnrm = gnrm.normalized()
                            # Fresnel reflectance of the dielectric (Schlick):
                            # R0 = ((1-ior)/(1+ior))^2, fr = R0 + (1-R0)(1-cos)^5.
                            cosi = ti.abs(rd.dot(gnrm))
                            r0 = (1.0 - ior) / (1.0 + ior)
                            r0 = r0 * r0
                            fr = r0 + (1.0 - r0) * ti.pow(1.0 - cosi, 5.0)
                            # A manual ``reflectivity`` raises the reflectance
                            # floor like a mirror coating: 0 = pure Fresnel
                            # glass; ~1 = a near-perfect mirror that still
                            # refracts the tiny transmitted remainder.
                            R = reflectivity + (1.0 - reflectivity) * fr

                acc += (weight * alpha * (1.0 - R)) * color

                if is_glass:
                    # Split: spawn the refracted (transmitted) branch into this
                    # pixel's spare sub-block -- the allocation counter is per
                    # pixel (rs_used[pix]), so different pixels bump different
                    # addresses (no single global atomic). The reflected branch
                    # continues in this slot; both commit to the same pixel.
                    wt = weight * (1.0 - R) * (1.0 - alpha)
                    if wt > MIN_WEIGHT:
                        c_local = ti.atomic_add(rs_used[pix], 1)
                        if c_local < splits_per_pixel:
                            c = num_primary + pix * splits_per_pixel + c_local
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
                    # Reflect the parent, Fresnel-weighted (decoupled from the
                    # opacity -- a clear glass still reflects per R).
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
                        normal = _flat_triangle_normal(
                            f, prim, 1.0 - a - b, a, b, tri_norm, tri_pos,
                            tri_uvs, tri_tex_meta, textures,
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
                if weight < MIN_WEIGHT:
                    done = True
                    break

            glow_rgb = _accumulate_glow(
                ro_seg, rd_seg, inv_rd_seg, t_seg_end, f, ff,
                has_tri, has_pn, has_bez,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
                tri_pos, tri_colors, tri_extra, num_colored_triangles,
                p_nodes, p_node_miss, p_leaf_prim, p_leaf_tspan, p_first_leaf,
                pn_ctrl, pn_colors, pn_extra,
                b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan, b_first_leaf,
                circuit_meta, circuit_colors, edges_2d, edge_offsets
            )
            acc[0] += weight_seg * glow_rgb[0]
            acc[1] += weight_seg * glow_rgb[1]
            acc[2] += weight_seg * glow_rgb[2]

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
            if done:
                # Terminated: commit this branch's premultiplied colour and its
                # leftover throughput (what the background shows through) into
                # the shared per-pixel accumulator.
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
            glow_rgb = _accumulate_glow(
                ro, rd, inv_rd, 1e30, f, ff,
                has_tri, has_pn, has_bez,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
                tri_pos, tri_colors, tri_extra, num_colored_triangles,
                p_nodes, p_node_miss, p_leaf_prim, p_leaf_tspan, p_first_leaf,
                pn_ctrl, pn_colors, pn_extra,
                b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan, b_first_leaf,
                circuit_meta, circuit_colors, edges_2d, edge_offsets
            )
            weight = rs_sca[r, 0]
            rs_acc[r, 0] += weight * glow_rgb[0]
            rs_acc[r, 1] += weight * glow_rgb[1]
            rs_acc[r, 2] += weight * glow_rgb[2]

            for k in ti.static(range(4)):
                ti.atomic_add(pix_accum[pix, k], rs_acc[r, k])
            ti.atomic_add(pix_accum[pix, 4], rs_sca[r, 0])
            rs_int[r, 2] = _DONE
