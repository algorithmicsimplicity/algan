"""UNSUPPORTED legacy variant: *textured-surface* wavefront shade kernel
(Surface / flat-triangle scenes only).

This variant is no longer maintained and no longer works; the monolithic
general wavefront (``wavefront_kernels_taichi``) is the only supported
deterministic tracer. The module is kept for reference;
``ALGAN_WF_TEXTURED=1`` still routes here, unsupported. Original design
rationale follows.

This is a proof-of-concept alternative to the per-vertex shade kernel in
``wavefront_kernels_taichi``: instead of reading colour / material / surface
properties out of per-vertex arrays, every triangle carries UV coordinates and
three integer indexes that look up three texture banks --

* **colour**   -- RGBA + glow (5 channels),
* **material** -- the shading parameter block prefixed with the pipeline id
  (13 channels, always a 1x1 constant texture per primitive),
* **surface**  -- metalness / roughness / IOR / transmission (4 channels)
  used to decide how the ray scatters (reflect / refract / pass through).

The banks are built by ``scene_builder._build_textured_scene``: a property
group that is constant across a surface becomes a shared 1x1 texture and one
that varies per vertex a per-triangle 2x2 texture whose bilinear lookup (at the
canonical corner UVs) reproduces the corner values and blends between them --
an approximation of barycentric interpolation.

The shade kernel starts lean (triangle-only, no shadows / scatter dispatch /
normal maps) and grows one feature at a time behind compile-time templates so
each feature's marginal occupancy / performance cost can be measured in
isolation (``settings.WF_TEXTURED_FEATURES``):

* ``feat_bez``       -- bezier-circuit hit shading (+ bezier BVH traversal),
* ``feat_scatter``   -- the monolith's generic per-material scatter dispatch
  (via ``default_scatter``) in place of the inline built-in bounce,
* ``feat_shadows``   -- one binary hard-shadow ray per light (triangle
  occluders), gating each fragment's direct lighting,
* ``feat_normalmap`` -- tangent-space normal-map perturbation of the shading
  normal.

Ray generation, BVH traversal and compositing reuse the existing
``wavefront_generate_rays`` / ``wavefront_traverse`` / ``wf_composite_accum``
kernels.
"""
import taichi as ti

from algan.rendering.raytracing.raytrace_kernels_taichi import (
    DEPTH_TIE_EPSILON,
    KBUF,
    MAX_SURFACES_PER_RAY,
    MIN_ALPHA,
    MIN_HIT_DISTANCE,
    MIN_WEIGHT,
    NODE_ARG,
    _bezier_normal,
    _comes_after,
    _nearest_triangle_hit,
    _safe_inverse,
    _sample_circuit_color,
    _triangle_normal,
)
from algan.rendering.raytracing.shading_taichi import (
    MAX_SHADOW_LIGHTS,
    _MID_DEFAULT,
    _MID_LAMBERT,
    _MID_PHONG,
    _MID_PHYSICAL,
    _MID_STANDARD,
    _stage_default,
    _stage_lambert,
    _stage_phong,
    _stage_physical,
    _stage_standard,
)
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _material_reflectance, _offset_transmitted_origin, _refract_ray,
    _reserve_continuation_slot, default_scatter)

# Per-ray status codes (rs_int column 2), matching wavefront_generate_rays.
_ACTIVE = 0
_DONE = 1


@ti.func
def _sample_bank(f, u, v, idx, meta: ti.template(), bank: ti.template(),
                 NC: ti.template()):
    """Bilinear sample of texture ``idx`` (all ``NC`` channels) from a flat
    texel bank. ``meta[idx]`` gives ``(offset, width, height)``; a 1x1 texture
    returns its single texel unchanged, a 2x2 blends its four texels. Texels are
    stored column-major (``offset + cx * height + cy``), matching the builder."""
    offset = meta[idx, 0]
    width = meta[idx, 1]
    height = meta[idx, 2]
    fw = ti.cast(width, ti.f32)
    fh = ti.cast(height, ti.f32)

    px = ti.math.clamp(u * (fw - 1.0), 0.0, ti.max(fw - 1.0, 0.0))
    py = ti.math.clamp(v * (fh - 1.0), 0.0, ti.max(fh - 1.0, 0.0))
    x0 = ti.floor(px)
    y0 = ti.floor(py)
    xr = px - x0
    yr = py - y0

    out = ti.Vector([0.0] * NC)
    sum_w = 0.0
    tc = f % bank.shape[0]
    npt = bank.shape[1]
    for corner in ti.static(range(4)):
        cx = ti.cast(x0 + (corner % 2), ti.i32)
        cy = ti.cast(y0 + (corner // 2), ti.i32)
        w = (xr if (corner % 2) == 1 else 1.0 - xr) * (
            yr if (corner // 2) == 1 else 1.0 - yr)
        cx = ti.math.clamp(cx, 0, ti.max(width - 1, 0))
        cy = ti.math.clamp(cy, 0, ti.max(height - 1, 0))
        aidx = ti.math.clamp(offset + cx * height + cy, 0, npt - 1)
        for ci in ti.static(range(NC)):
            out[ci] += w * bank[tc, aidx, ci]
        sum_w += w
    return out / ti.max(sum_w, 1e-6)


@ti.func
def _tri_geom(f, prim, tri_pos: ti.template()):
    """The three world-space corners of a flat triangle at frame ``f``."""
    tp = f % tri_pos.shape[0]
    v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                      tri_pos[tp, prim, 2])
    v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                      tri_pos[tp, prim, 5])
    v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                      tri_pos[tp, prim, 8])
    return v0, v1, v2


@ti.func
def _shade_normal_textured(feat_normalmap: ti.template(), f, prim, w0, w1, w2,
                           uu, vv, tri_norm: ti.template(),
                           tri_pos: ti.template(), tri_uv: ti.template(),
                           nmap_idx: ti.template(), nmap_meta: ti.template(),
                           nmap_bank: ti.template()):
    """Interpolated vertex normal, perturbed by a tangent-space normal map when
    ``feat_normalmap`` is compiled in and the triangle carries one
    (``nmap_idx[prim] >= 0``). The tangent frame is derived per hit from the
    triangle's corners and UVs (mirrors ``_flat_triangle_normal``), so a scene
    with no normal maps is byte-identical to the plain interpolated normal."""
    normal = _triangle_normal(f, prim, w0, w1, w2, tri_norm, tri_pos)
    if ti.static(feat_normalmap != 0):
        if nmap_idx[prim] >= 0:
            m = _sample_bank(f, uu, vv, nmap_idx[prim], nmap_meta, nmap_bank, 3)
            tn = ti.math.vec3(m[0], m[1], m[2])
            if tn.norm() > 1e-6 and normal.norm() > 1e-9:
                nb = normal.normalized()
                v0, v1, v2 = _tri_geom(f, prim, tri_pos)
                tu = f % tri_uv.shape[0]
                du1 = tri_uv[tu, prim, 2] - tri_uv[tu, prim, 0]
                dv1 = tri_uv[tu, prim, 3] - tri_uv[tu, prim, 1]
                du2 = tri_uv[tu, prim, 4] - tri_uv[tu, prim, 0]
                dv2 = tri_uv[tu, prim, 5] - tri_uv[tu, prim, 1]
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


@ti.func
def _shadow_occluded_tri(ro, rd, f, ff, max_t, layer_offset_triangles,
                         t_nodes: ti.template(), t_node_miss: ti.template(),
                         t_leaf_prim: ti.template(), t_leaf_tspan: ti.template(),
                         t_first_leaf, tri_pos: ti.template()):
    """Lean binary hard-shadow test against the triangle BVH: 1.0 if any
    triangle lies between the shaded point and the light (within ``max_t``),
    else 0.0. Treats every triangle as an opaque occluder (one traversal per
    shadow ray); enough to measure the shadow-ray cost on opaque surfaces."""
    inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                          _safe_inverse(rd[2]))
    best_t, best_prim, _w1, _w2, _layer = _nearest_triangle_hit(
        0, ro, rd, inv_rd, f, ff, 0.0, 1e30, max_t, layer_offset_triangles,
        t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf, tri_pos)
    occluded = 0.0
    if (best_prim >= 0) and (best_t < max_t):
        occluded = 1.0
    return occluded


@ti.func
def _run_material(pid, mat_bank: ti.template(), texoff, f, pos, view_dir,
                  n_interp, face_n, albedo, glow,
                  light_pos: ti.template(), light_col: ti.template(),
                  num_lights, shadows: ti.template(), vis):
    """Shade a fragment with its material's built-in pipeline. ``texoff`` is the
    material texture's (1x1) texel row in ``mat_bank``; the parameter block sits
    in channels 1.. (channel 0 is the pipeline id). Reuses the exact built-in
    stage funcs, so the result matches the per-vertex kernel given identical
    parameters. ``vis`` carries per-light shadow visibilities (used iff
    ``shadows``). Unlit / user pipelines pass the colour through."""
    out = ti.math.vec3(albedo[0], albedo[1], albedo[2])
    g = glow
    if pid == _MID_DEFAULT:
        r = _stage_default(pos, view_dir, n_interp, face_n, out, g,
                           mat_bank, f, texoff, 1,
                           light_pos, light_col, num_lights, shadows, vis)
        out = ti.math.vec3(r[0], r[1], r[2])
        g = r[3]
    elif pid == _MID_LAMBERT:
        r = _stage_lambert(pos, view_dir, n_interp, face_n, out, g,
                           mat_bank, f, texoff, 1,
                           light_pos, light_col, num_lights, shadows, vis)
        out = ti.math.vec3(r[0], r[1], r[2])
        g = r[3]
    elif pid == _MID_PHONG:
        r = _stage_phong(pos, view_dir, n_interp, face_n, out, g,
                         mat_bank, f, texoff, 1,
                         light_pos, light_col, num_lights, shadows, vis)
        out = ti.math.vec3(r[0], r[1], r[2])
        g = r[3]
    elif pid == _MID_STANDARD:
        r = _stage_standard(pos, view_dir, n_interp, face_n, out, g,
                            mat_bank, f, texoff, 1,
                            light_pos, light_col, num_lights, shadows, vis)
        out = ti.math.vec3(r[0], r[1], r[2])
        g = r[3]
    elif pid == _MID_PHYSICAL:
        r = _stage_physical(pos, view_dir, n_interp, face_n, out, g,
                            mat_bank, f, texoff, 1,
                            light_pos, light_col, num_lights, shadows, vis)
        out = ti.math.vec3(r[0], r[1], r[2])
        g = r[3]
    return ti.math.vec4(out[0], out[1], out[2], g)


@ti.kernel
def wf_shade_textured(
        active: ti.types.ndarray(), num_active: int,
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_uv: ti.types.ndarray(),
        color_idx: ti.types.ndarray(), mat_idx: ti.types.ndarray(),
        surf_idx: ti.types.ndarray(),
        color_bank: ti.types.ndarray(), color_meta: ti.types.ndarray(),
        mat_bank: ti.types.ndarray(), mat_meta: ti.types.ndarray(),
        surf_bank: ti.types.ndarray(), surf_meta: ti.types.ndarray(),
        # Normal-map feature bank (placeholder / idx -1 when unused).
        nmap_idx: ti.types.ndarray(), nmap_bank: ti.types.ndarray(),
        nmap_meta: ti.types.ndarray(),
        # Bezier feature: circuit colour data (placeholder when unused).
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        # Shadow feature: triangle BVH for the shadow rays.
        t_nodes: NODE_ARG, t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int,
        pixel_world_scale: ti.types.ndarray(),
        layer_offset_triangles: float,
        light_pos: ti.types.ndarray(), light_col: ti.types.ndarray(),
        num_lights: int, refraction: ti.template(),
        feat_bez: ti.template(), feat_scatter: ti.template(),
        feat_shadows: ti.template(), feat_normalmap: ti.template(),
        time_start: int, width: int, height: int, ray_offset: int,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
        rs_int: ti.types.ndarray(),
        rs_kt: ti.types.ndarray(), rs_kl: ti.types.ndarray(),
        rs_ka: ti.types.ndarray(), rs_kb: ti.types.ndarray(),
        rs_kp: ti.types.ndarray(), rs_kf: ti.types.ndarray(),
        rs_pix: ti.types.ndarray(), pix_accum: ti.types.ndarray(),
        rs_alloc: ti.types.ndarray()):
    """Drain each active ray's gathered hits front-to-back, shading every hit
    from its three texture lookups and continuing the ray per the surface
    texture (opacity pass-through / mirror reflection / Fresnel glass split).
    Features (beziers / scatter dispatch / shadows / normal maps) are compiled
    in per the ``feat_*`` templates."""
    pixels_per_frame = width * height
    for i in range(num_active):
        r = active[i]
        pix = rs_pix[r]
        num_hits = rs_int[r, 3]
        if num_hits > 0:
            f = time_start + (ray_offset + pix) // pixels_per_frame
            ff = ti.cast(f, ti.f32)
            ro = ti.math.vec3(rs_ro[r, 0], rs_ro[r, 1], rs_ro[r, 2])
            rd = ti.math.vec3(rs_rd[r, 0], rs_rd[r, 1], rs_rd[r, 2])
            acc = ti.math.vec4(rs_acc[r, 0], rs_acc[r, 1], rs_acc[r, 2],
                               rs_acc[r, 3])
            weight = ti.math.vec3(rs_sca[r, 0], rs_sca[r, 5], rs_sca[r, 6])
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

                if (edge_hit == 1) and (t_hit - seam_t <= DEPTH_TIE_EPSILON):
                    t_prev = t_hit
                    layer_prev = hit_layer
                    continue
                seam_t = t_hit if edge_hit == 1 else -1e30

                w0 = 1.0 - a - b
                color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                # Raw surface colour before lighting -- tints the metal
                # Fresnel lobe and the transmitted share (colour transport).
                albedo3 = ti.math.vec3(0.0, 0.0, 0.0)
                alpha = 0.0
                reflectivity = -1.0
                ior = 0.0
                T = 0.0
                is_tri = True
                if ti.static(feat_bez != 0):
                    is_tri = htype == 1

                if is_tri:
                    # Interpolated corner UV (canonical: (w1, w2)).
                    tu = f % tri_uv.shape[0]
                    uu = (w0 * tri_uv[tu, prim, 0] + a * tri_uv[tu, prim, 2]
                          + b * tri_uv[tu, prim, 4])
                    vv = (w0 * tri_uv[tu, prim, 1] + a * tri_uv[tu, prim, 3]
                          + b * tri_uv[tu, prim, 5])

                    cs = _sample_bank(f, uu, vv, color_idx[prim],
                                      color_meta, color_bank, 5)
                    color = ti.math.vec4(cs[0], cs[1], cs[2], cs[3])
                    albedo3 = ti.math.vec3(cs[0], cs[1], cs[2])
                    alpha = cs[4]

                    si = surf_idx[prim]
                    if si >= 0:
                        ss = _sample_bank(f, uu, vv, si, surf_meta, surf_bank, 4)
                        reflectivity = ss[0]
                        ior = ss[2]
                        T = ti.math.clamp(ss[3], 0.0, 1.0)

                    mi = mat_idx[prim]
                    if mi >= 0:
                        sn = _shade_normal_textured(
                            feat_normalmap, f, prim, w0, a, b, uu, vv,
                            tri_norm, tri_pos, tri_uv,
                            nmap_idx, nmap_meta, nmap_bank)
                        v0, v1, v2 = _tri_geom(f, prim, tri_pos)
                        face_n = (v1 - v0).cross(v2 - v0)
                        pos = ro + t_hit * rd

                        # Per-light shadow visibility (feature). Fire one binary
                        # shadow ray per light from the offset hit point.
                        vis = ti.Vector([1.0] * MAX_SHADOW_LIGHTS)
                        if ti.static(feat_shadows != 0):
                            snf = sn.normalized()
                            if snf.dot(rd) > 0.0:
                                snf = -snf
                            fn = face_n
                            if fn.norm() > 1e-9:
                                fn = fn.normalized()
                            if fn.dot(snf) < 0.0:
                                fn = -fn
                            sorigin = pos + fn * (10.0 * MIN_HIT_DISTANCE)
                            tl = f % light_pos.shape[0]
                            for li in range(num_lights):
                                if li < MAX_SHADOW_LIGHTS:
                                    lp = ti.math.vec3(light_pos[tl, li, 0],
                                                      light_pos[tl, li, 1],
                                                      light_pos[tl, li, 2])
                                    to_light = lp - pos
                                    ldist = to_light.norm()
                                    if ldist > 1e-5:
                                        wi = to_light / ldist
                                        if (fn.dot(wi) > 1e-3) and \
                                                (snf.dot(wi) > 1e-4):
                                            occ = _shadow_occluded_tri(
                                                sorigin, wi, f, ff,
                                                ldist - 20.0 * MIN_HIT_DISTANCE,
                                                layer_offset_triangles,
                                                t_nodes, t_node_miss,
                                                t_leaf_prim, t_leaf_tspan,
                                                t_first_leaf, tri_pos)
                                            vis[li] = 1.0 - occ

                        tmi = f % mat_bank.shape[0]
                        texoff = mat_meta[mi, 0]
                        pid = ti.cast(mat_bank[tmi, texoff, 0] + 0.5, ti.i32)
                        color = _run_material(
                            pid, mat_bank, texoff, f, pos, -rd, sn, face_n,
                            color, color[3], light_pos, light_col, num_lights,
                            feat_shadows, vis)
                elif ti.static(feat_bez != 0):
                    # Bezier circuit hit: sampled colour, no material shading.
                    color, alpha = _sample_circuit_color(
                        prim, f, a, b, border, circuit_meta, circuit_colors,
                        circuit_border_colors)
                    albedo3 = ti.math.vec3(color[0], color[1], color[2])

                # Continuation (scatter). Two compile-time variants: the inline
                # built-in bounce (default) or the monolith's generic
                # per-material scatter dispatch (feat_scatter), which routes the
                # same behaviour through ``default_scatter``.
                alpha = ti.math.clamp(alpha, 0.0, 1.0)

                if ti.static(feat_scatter == 0):
                    normal = ti.math.vec3(0.0, 0.0, 0.0)
                    if is_tri:
                        normal = _triangle_normal(
                            f, prim, w0, a, b, tri_norm, tri_pos)
                    elif ti.static(feat_bez != 0):
                        normal = _bezier_normal(f, prim, circuit_meta)
                    normal = normal.normalized()
                    R, diel_pass = _material_reflectance(
                        rd, normal, reflectivity, ior, albedo3)
                    if bounces_left <= 0:
                        # Out of bounces: no reflected ray. Transmission stays
                        # gated by ``diel_pass`` -- see ``_scatter_impl``.
                        R = ti.math.vec3(0.0, 0.0, 0.0)

                    # Transmission alone gates glass; this kernel is
                    # triangle-only, so there is no thin-pane case here. The
                    # metal-blended Fresnel ``R`` stands (the metal share
                    # reflects rather than transmits) -- see ``_scatter_impl``,
                    # including the colour transport (vec3 weights, albedo
                    # tint, max-component decisions).
                    is_glass = False
                    if ti.static(refraction != 0):
                        if (T > 1e-4) and (bounces_left > 0) \
                                and (ior > 1.0 + 1e-4) and is_tri:
                            is_glass = True

                    one3 = ti.math.vec3(1.0, 1.0, 1.0)
                    tint = ti.math.clamp(albedo3, 0.0, 1.0)
                    # Only the dielectric-interior share transmits -- see the
                    # four-way split derivation in ``_scatter_impl``.
                    trans_share = diel_pass * T
                    r_glow = ti.max(R[0], ti.max(R[1], R[2]))
                    w_glow = ti.max(weight[0], ti.max(weight[1], weight[2]))
                    share = (weight * alpha) * (one3 - R - trans_share)
                    acc += ti.math.vec4(
                        share[0], share[1], share[2],
                        w_glow * alpha
                        * (1.0 - r_glow - trans_share)) * color
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
                                and (alpha < 1.0 - MIN_ALPHA) \
                                and (bounces_left > 0):
                            split_refl = True

                    if is_glass:
                        wt = weight * trans_energy * tint
                        wt_max = ti.max(wt[0], ti.max(wt[1], wt[2]))
                        if wt_max > MIN_WEIGHT:
                            c, have_slot = _reserve_continuation_slot(
                                rs_alloc, rs_ro.shape[0])
                            if have_slot:
                                rdt = _refract_ray(rd, normal, ior)
                                hp = ro + t_hit * rd
                                face_normal = normal
                                if is_tri:
                                    tp = f % tri_pos.shape[0]
                                    v0 = ti.math.vec3(
                                        tri_pos[tp, prim, 0],
                                        tri_pos[tp, prim, 1],
                                        tri_pos[tp, prim, 2])
                                    v1 = ti.math.vec3(
                                        tri_pos[tp, prim, 3],
                                        tri_pos[tp, prim, 4],
                                        tri_pos[tp, prim, 5])
                                    v2 = ti.math.vec3(
                                        tri_pos[tp, prim, 6],
                                        tri_pos[tp, prim, 7],
                                        tri_pos[tp, prim, 8])
                                    face_normal = (v1 - v0).cross(v2 - v0)
                                rorig = _offset_transmitted_origin(
                                    hp, rdt, face_normal, normal)
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
                        # Primary carries the heavier of reflection /
                        # coverage-miss (see ``_scatter_impl``).
                        if ((refl_max > MIN_ALPHA)
                                and (refl_max >= cover_pass)):
                            nref = normal
                            if nref.dot(rd) > 0.0:
                                nref = -nref
                            hit_point = ro + t_hit * rd
                            rd = (rd - 2.0 * rd.dot(nref) * nref).normalized()
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
                            weight *= cover_pass
                            t_prev = t_hit
                            layer_prev = hit_layer
                    elif split_refl:
                        wt = weight * refl_energy
                        wt_max = ti.max(wt[0], ti.max(wt[1], wt[2]))
                        if wt_max > MIN_WEIGHT:
                            c, have_slot = _reserve_continuation_slot(
                                rs_alloc, rs_ro.shape[0])
                            if have_slot:
                                nref = normal
                                if nref.dot(rd) > 0.0:
                                    nref = -nref
                                rdr = (rd - 2.0 * rd.dot(nref)
                                       * nref).normalized()
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
                        weight *= cover3 + trans_energy * tint
                        t_prev = t_hit
                        layer_prev = hit_layer
                    # No split pool: reflect only while the reflection
                    # outweighs what shows through (see ``default_scatter``).
                    elif ((refl_max > MIN_ALPHA)
                          and (refl_max >= cover_pass)):
                        nref = normal
                        if nref.dot(rd) > 0.0:
                            nref = -nref
                        hit_point = ro + t_hit * rd
                        rd = (rd - 2.0 * rd.dot(nref) * nref).normalized()
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
                    # Generic scatter dispatch (mirrors the monolith's custom-
                    # scatter block): compute the shading + geometric normals and
                    # ior, then let ``default_scatter`` return the continuation
                    # branches. Byte-identical behaviour to the inline block.
                    sni = ti.math.vec3(0.0, 0.0, 0.0)
                    sfn = ti.math.vec3(0.0, 0.0, 0.0)
                    if is_tri:
                        sni = _triangle_normal(f, prim, w0, a, b, tri_norm,
                                               tri_pos)
                        v0, v1, v2 = _tri_geom(f, prim, tri_pos)
                        sfn = (v1 - v0).cross(v2 - v0)
                    elif ti.static(feat_bez != 0):
                        sni = _bezier_normal(f, prim, circuit_meta)
                        sfn = sni
                    hit_point = ro + t_hit * rd
                    (contrib, pass_w, refl_orig, refl_dir, refl_w,
                     trans_orig, trans_dir, trans_w) = default_scatter(
                        rd, sni, sfn, hit_point, color, albedo3, alpha,
                        reflectivity, ior, T, mat_bank, f, prim, bounces_left,
                        refraction)
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

            if (not done) and (not bounced) and (num_hits < KBUF):
                done = True
            if processed >= MAX_SURFACES_PER_RAY:
                done = True

            for k in ti.static(range(3)):
                rs_ro[r, k] = ro[k]
                rs_rd[r, k] = rd[k]
            for k in ti.static(range(4)):
                rs_acc[r, k] = acc[k]
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
            if done:
                for k in ti.static(range(4)):
                    ti.atomic_add(pix_accum[pix, k], acc[k])
                for k in ti.static(range(3)):
                    ti.atomic_add(pix_accum[pix, 4 + k], weight[k])
        else:
            # Escaped to background: commit colour + leftover throughput.
            for k in ti.static(range(4)):
                ti.atomic_add(pix_accum[pix, k], rs_acc[r, k])
            ti.atomic_add(pix_accum[pix, 4], rs_sca[r, 0])
            ti.atomic_add(pix_accum[pix, 5], rs_sca[r, 5])
            ti.atomic_add(pix_accum[pix, 6], rs_sca[r, 6])
            rs_int[r, 2] = _DONE
