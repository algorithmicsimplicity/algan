"""Specialized deterministic ray-trace kernel for batches with NO point-normal
(PN) patches -- i.e. flat triangles and/or bezier circuits only (the default
``pn_triangles=False`` case: meshes, polygons, text and 2D vector shapes).

The general ``render_scene_stbvh`` always compiles the PN-patch Matrix Pencil solver
(``_pn_intersect``) into its call graph, even for scenes
that contain no PN patches. That solver is the dominant register consumer, so a
no-PN scene rendered by the general kernel runs at ~128 regs / ~25% occupancy
carrying code it never executes. This module is the general kernel with the PN
traversal and PN shading removed -- byte-identical output on any no-PN batch
(the removed paths are inert when ``has_pn == 0``), but a much lighter kernel
(no Matrix Pencil solver) that runs at higher occupancy.

Mirrors the ``render_triangles_stbvh`` specialization (cf. the triangle-only
kernel) one rung up: triangle + bezier, but still no PN.
"""
import taichi as ti

from algan.rendering.raytracing.ray_trace_taichi import (
    BARYCENTRIC_EPSILON,
    BVH_ARITY,
    DEPTH_TIE_EPSILON,
    KBUF,
    LEAF_SIZE,
    MAX_SHADOW_LIGHTS,
    MAX_SURFACES_PER_RAY,
    MIN_ALPHA,
    MIN_HIT_DISTANCE,
    MIN_WEIGHT,
    TRIANGLE_EDGE_EPSILON,
    _M_BASIS_U,
    _M_BASIS_V,
    _M_BORDER_W,
    _M_CENTER,
    _M_FILLED,
    _M_NORMAL,
    _bezier_normal,
    _comes_after,
    _generate_ray,
    _node_intersected,
    _safe_inverse,
    _sample_circuit_color,
    _shade_tri_hit,
    _triangle_color,
    _flat_triangle_color,
    _flat_triangle_alpha,
    _triangle_extra,
    _triangle_normal,
    _accumulate_glow,
    finalize_pixel_color,
)


@ti.func
def _collect_hits_no_pn(ro, rd, inv_rd, f, ff, t_prev, layer_prev,
                        pixel_size_per_t, base_dist, layer_offset_triangles,
                        hit_t: ti.template(), hit_layer: ti.template(),
                        hit_prim: ti.template(), hit_flags: ti.template(),
                        hit_a: ti.template(), hit_b: ti.template(),
                        t_nodes: ti.template(), t_node_miss: ti.template(),
                        t_leaf_prim: ti.template(),
                        t_leaf_tspan: ti.template(),
                        t_first_leaf, tri_pos: ti.template(),
                        b_nodes: ti.template(), b_node_miss: ti.template(),
                        b_leaf_prim: ti.template(),
                        b_leaf_tspan: ti.template(),
                        b_first_leaf, circuit_meta: ti.template(),
                        edges_2d: ti.template(), edge_offsets: ti.template(),
                        has_tri: ti.i32, has_bez: ti.i32) -> ti.i32:
    """``_collect_hits`` with the PN-patch BVH/solver removed: gathers the
    up-to-KBUF nearest triangle and bezier hits. Identical triangle/bezier
    acceptance and opaque-pruning logic; no Matrix Pencil solver in the call graph. """
    count = 0
    worst_idx = 0
    worst_t = 1e30
    worst_layer = -1e30
    opq_t = 1e30
    opq_layer = -1e30

    # --- Triangle BVH ---
    tp = f % tri_pos.shape[0]
    node = -1
    if has_tri != 0:
        node = 0
    while node != -1:
        window_hi = worst_t + DEPTH_TIE_EPSILON if count == KBUF else 1e30
        window_hi = ti.min(window_hi, opq_t + DEPTH_TIE_EPSILON)
        if _node_intersected(node, ff, ro, inv_rd,
                             t_prev - DEPTH_TIE_EPSILON, window_hi, t_nodes):
            if node >= t_first_leaf:
                base = (node - t_first_leaf) * LEAF_SIZE
                for j in ti.static(range(LEAF_SIZE)):
                    prim = t_leaf_prim[base + j]
                    tspan = t_leaf_tspan[base + j]
                    if ((prim >= 0) and ((tspan & 0xFFFF) <= f)
                            and (f <= ((tspan >> 16) & 0x7FFF))):
                        v0 = ti.math.vec3(tri_pos[tp, prim, 0],
                                          tri_pos[tp, prim, 1],
                                          tri_pos[tp, prim, 2])
                        v1 = ti.math.vec3(tri_pos[tp, prim, 3],
                                          tri_pos[tp, prim, 4],
                                          tri_pos[tp, prim, 5])
                        v2 = ti.math.vec3(tri_pos[tp, prim, 6],
                                          tri_pos[tp, prim, 7],
                                          tri_pos[tp, prim, 8])
                        e1 = v1 - v0
                        e2 = v2 - v0
                        pv = rd.cross(e2)
                        det = e1.dot(pv)
                        if ti.abs(det) > 1e-12:
                            inv_det = 1.0 / det
                            tvec = ro - v0
                            w1 = tvec.dot(pv) * inv_det
                            qv = tvec.cross(e1)
                            w2 = rd.dot(qv) * inv_det
                            if ((w1 >= -BARYCENTRIC_EPSILON)
                                    and (w2 >= -BARYCENTRIC_EPSILON)
                                    and (w1 + w2 <= 1.0 + BARYCENTRIC_EPSILON)):
                                t = e2.dot(qv) * inv_det
                                layer = layer_offset_triangles + ti.cast(
                                    prim, ti.f32)
                                accept = ((t > MIN_HIT_DISTANCE)
                                          and _comes_after(t, layer, t_prev,
                                                           layer_prev)
                                          and not _comes_after(
                                              t, layer, opq_t, opq_layer))
                                if accept and (count == KBUF):
                                    accept = _comes_after(worst_t, worst_layer,
                                                          t, layer)
                                if accept:
                                    slot = worst_idx
                                    if count < KBUF:
                                        slot = count
                                        count += 1
                                    hit_t[slot] = t
                                    hit_layer[slot] = layer
                                    hit_prim[slot] = prim
                                    w0 = 1.0 - w1 - w2
                                    eh = 1 if (ti.min(w0, ti.min(w1, w2))
                                               < TRIANGLE_EDGE_EPSILON) else 0
                                    hit_flags[slot] = 1 | (eh << 2)
                                    hit_a[slot] = w1
                                    hit_b[slot] = w2
                                    if (tspan < 0) and _comes_after(
                                            opq_t, opq_layer, t, layer):
                                        opq_t = t
                                        opq_layer = layer
                                    if count == KBUF:
                                        worst_idx = 0
                                        worst_t = hit_t[0]
                                        worst_layer = hit_layer[0]
                                        for q in ti.static(range(1, KBUF)):
                                            if _comes_after(hit_t[q],
                                                            hit_layer[q],
                                                            worst_t,
                                                            worst_layer):
                                                worst_idx = q
                                                worst_t = hit_t[q]
                                                worst_layer = hit_layer[q]
                node = t_node_miss[node]
            else:
                node = BVH_ARITY * node + 1
        else:
            node = t_node_miss[node]

    # --- Bezier BVH (window tightened by the triangle hits) ---
    num_meta_frames = circuit_meta.shape[0]
    num_edge_frames = edges_2d.shape[0]
    node = -1
    if has_bez != 0:
        node = 0
    while node != -1:
        window_hi = worst_t + DEPTH_TIE_EPSILON if count == KBUF else 1e30
        window_hi = ti.min(window_hi, opq_t + DEPTH_TIE_EPSILON)
        if _node_intersected(node, ff, ro, inv_rd,
                             t_prev - DEPTH_TIE_EPSILON, window_hi, b_nodes):
            if node >= b_first_leaf:
                base = (node - b_first_leaf) * LEAF_SIZE
                for j in ti.static(range(LEAF_SIZE)):
                    circuit = b_leaf_prim[base + j]
                    tspan = b_leaf_tspan[base + j]
                    if ((circuit >= 0) and ((tspan & 0xFFFF) <= f)
                            and (f <= ((tspan >> 16) & 0x7FFF))):
                        tm = f % num_meta_frames
                        n = ti.math.vec3(circuit_meta[tm, circuit, _M_NORMAL],
                                         circuit_meta[tm, circuit, _M_NORMAL + 1],
                                         circuit_meta[tm, circuit, _M_NORMAL + 2])
                        denom = rd.dot(n)
                        layer = ti.cast(circuit, ti.f32)
                        if ti.abs(denom) > 1e-9:
                            center = ti.math.vec3(
                                circuit_meta[tm, circuit, _M_CENTER],
                                circuit_meta[tm, circuit, _M_CENTER + 1],
                                circuit_meta[tm, circuit, _M_CENTER + 2])
                            t = (center - ro).dot(n) / denom
                            accept = ((t > MIN_HIT_DISTANCE)
                                      and _comes_after(t, layer, t_prev,
                                                       layer_prev)
                                      and not _comes_after(
                                          t, layer, opq_t, opq_layer))
                            if accept and (count == KBUF):
                                accept = _comes_after(worst_t, worst_layer,
                                                      t, layer)
                            if accept:
                                hit = ro + t * rd - center
                                bu = ti.math.vec3(
                                    circuit_meta[tm, circuit, _M_BASIS_U],
                                    circuit_meta[tm, circuit, _M_BASIS_U + 1],
                                    circuit_meta[tm, circuit, _M_BASIS_U + 2])
                                bv = ti.math.vec3(
                                    circuit_meta[tm, circuit, _M_BASIS_V],
                                    circuit_meta[tm, circuit, _M_BASIS_V + 1],
                                    circuit_meta[tm, circuit, _M_BASIS_V + 2])
                                u = hit.dot(bu)
                                v = hit.dot(bv)

                                te = f % num_edge_frames
                                crossings = 0
                                min_dist_sq = 1e30
                                for e in range(edge_offsets[circuit],
                                               edge_offsets[circuit + 1]):
                                    x0 = edges_2d[te, e, 0]
                                    y0 = edges_2d[te, e, 1]
                                    x1 = edges_2d[te, e, 2]
                                    y1 = edges_2d[te, e, 3]
                                    if (y0 > v) != (y1 > v):
                                        x_cross = x0 + (v - y0) * (x1 - x0) / (y1 - y0)
                                        if x_cross > u:
                                            crossings += 1
                                    dx = x1 - x0
                                    dy = y1 - y0
                                    seg_t = ((u - x0) * dx + (v - y0) * dy) / ti.max(
                                        dx * dx + dy * dy, 1e-12)
                                    seg_t = ti.math.clamp(seg_t, 0.0, 1.0)
                                    cx = x0 + seg_t * dx - u
                                    cy = y0 + seg_t * dy - v
                                    min_dist_sq = ti.min(min_dist_sq,
                                                         cx * cx + cy * cy)

                                pixel_size = pixel_size_per_t * (base_dist + t)
                                border_w = (circuit_meta[tm, circuit, _M_BORDER_W]
                                            * pixel_size)
                                in_border = min_dist_sq < border_w * border_w
                                outline_w = 0.6 * pixel_size
                                inside = False
                                if circuit_meta[tm, circuit, _M_FILLED] > 0.5:
                                    inside = ((crossings % 2) == 1) or (
                                        min_dist_sq < outline_w * outline_w)
                                if inside or in_border:
                                    slot = worst_idx
                                    if count < KBUF:
                                        slot = count
                                        count += 1
                                    hit_t[slot] = t
                                    hit_layer[slot] = layer
                                    hit_prim[slot] = circuit
                                    hit_flags[slot] = (
                                        (1 if in_border else 0) << 3)
                                    hit_a[slot] = u
                                    hit_b[slot] = v
                                    if (tspan < 0) and _comes_after(
                                            opq_t, opq_layer, t, layer):
                                        opq_t = t
                                        opq_layer = layer
                                    if count == KBUF:
                                        worst_idx = 0
                                        worst_t = hit_t[0]
                                        worst_layer = hit_layer[0]
                                        for q in ti.static(range(1, KBUF)):
                                            if _comes_after(hit_t[q],
                                                            hit_layer[q],
                                                            worst_t,
                                                            worst_layer):
                                                worst_idx = q
                                                worst_t = hit_t[q]
                                                worst_layer = hit_layer[q]
                node = b_node_miss[node]
            else:
                node = BVH_ARITY * node + 1
        else:
            node = b_node_miss[node]
    return count


@ti.func
def _trace_no_pn_ray(ro, rd, inv_rd, f, ff, pixel_size_per_t,
                     layer_offset_triangles, max_bounces,
                     t_nodes: ti.template(), t_node_miss: ti.template(),
                     t_leaf_prim: ti.template(), t_leaf_tspan: ti.template(),
                     t_first_leaf, tri_pos: ti.template(),
                     tri_norm: ti.template(), tri_extra: ti.template(),
                     tri_colors: ti.template(), tri_uvs: ti.template(),
                     tri_tex_meta: ti.template(), textures: ti.template(),
                     num_colored_triangles: ti.i32,
                     b_nodes: ti.template(), b_node_miss: ti.template(),
                     b_leaf_prim: ti.template(), b_leaf_tspan: ti.template(),
                     b_first_leaf, circuit_meta: ti.template(),
                     circuit_colors: ti.template(),
                     circuit_border_colors: ti.template(),
                     edges_2d: ti.template(), edge_offsets: ti.template(),
                     has_tri: ti.i32, has_bez: ti.i32,
                     frag_shading: ti.template(),
                     tri_mat_id: ti.template(), tri_mat: ti.template(),
                     light_pos: ti.template(), light_col: ti.template(),
                     num_lights):
    """Trace one primary ray through a no-PN scene (flat triangles and/or
    bezier circuits), returning its premultiplied RGB + glow accumulator and
    remaining transmittance. The no-PN specialization of
    :func:`_trace_scene_ray`, factored out so the kernel can average several
    jittered sub-pixel rays in place."""
    acc = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    weight = 1.0
    t_prev = 0.0
    layer_prev = 1e30
    base_dist = 0.0
    bounces_left = max_bounces
    seam_t = -1e30

    kb_t = ti.Vector([0.0] * KBUF)
    kb_layer = ti.Vector([0.0] * KBUF)
    kb_prim = ti.Vector([0] * KBUF)
    kb_flags = ti.Vector([0] * KBUF)
    kb_a = ti.Vector([0.0] * KBUF)
    kb_b = ti.Vector([0.0] * KBUF)

    processed = 0
    done = False
    while (not done) and (processed < MAX_SURFACES_PER_RAY):
        ro_seg = ro
        rd_seg = rd
        inv_rd_seg = inv_rd
        weight_seg = weight
        num_hits = _collect_hits_no_pn(
            ro, rd, inv_rd, f, ff, t_prev, layer_prev,
            pixel_size_per_t, base_dist, layer_offset_triangles,
            kb_t, kb_layer, kb_prim, kb_flags, kb_a, kb_b,
            t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
            t_first_leaf, tri_pos,
            b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
            b_first_leaf, circuit_meta, edges_2d, edge_offsets,
            has_tri, has_bez)
        if num_hits == 0:
            glow_rgb = _accumulate_glow(
                ro_seg, rd_seg, inv_rd_seg, 1e30, f, ff,
                has_tri, 0, has_bez,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
                tri_pos, tri_colors, tri_extra, num_colored_triangles,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
                tri_pos, tri_colors, tri_extra,
                b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan, b_first_leaf,
                circuit_meta, circuit_colors, edges_2d, edge_offsets
            )
            acc[0] += weight * glow_rgb[0]
            acc[1] += weight * glow_rgb[1]
            acc[2] += weight * glow_rgb[2]
            break

        bounced = False
        drained = 0
        t_seg_end = 0.0
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
            kb_prim[sel] = -1  # consume
            drained += 1
            processed += 1
            htype = flags & 3
            edge_hit = (flags >> 2) & 1
            border = (flags >> 3) & 1

            # No PN patches here, so the seam window is always the flat
            # (triangle/coplanar) tie epsilon.
            if (edge_hit == 1) and (t_hit - seam_t <= DEPTH_TIE_EPSILON):
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
                reflectivity, _rough = _triangle_extra(f, prim, w0, a, b,
                                                       tri_extra)
            else:
                color, alpha = _sample_circuit_color(
                    prim, f, a, b, border,
                    circuit_meta, circuit_colors, circuit_border_colors)

            # Fragment shading: material-shade flat-triangle hits per fragment
            # from the raw albedo; bezier circuits keep their sampled colour.
            # Compiled out on the default (vertex-shaded) path via ti.static.
            # (The lean no-PN kernel never casts shadows -- a shadowed render is
            # forced onto the general kernel -- so shadows are disabled here.)
            if ti.static(frag_shading != 0):
                if htype == 1:
                    color = _shade_tri_hit(f, prim, a, b, rd, t_hit, ro,
                                           tri_pos, tri_norm,
                                           tri_mat_id, tri_mat,
                                           light_pos, light_col, num_lights,
                                           color, 0,
                                           ti.Vector([1.0] * MAX_SHADOW_LIGHTS))

            alpha = ti.math.clamp(alpha, 0.0, 1.0)
            reflectivity = ti.math.clamp(reflectivity, 0.0, 1.0)
            if bounces_left <= 0:
                reflectivity = 0.0

            acc += (weight * alpha * (1.0 - reflectivity)) * color

            if (reflectivity > MIN_ALPHA) and (alpha > MIN_ALPHA):
                normal = ti.math.vec3(0.0, 0.0, 0.0)
                if htype == 1:
                    normal = _triangle_normal(f, prim, 1.0 - a - b, a, b,
                                              tri_norm, tri_pos)
                else:
                    normal = _bezier_normal(f, prim, circuit_meta)
                normal = normal.normalized()
                if normal.dot(rd) > 0.0:
                    normal = -normal
                hit_point = ro + t_hit * rd
                rd = (rd - 2.0 * rd.dot(normal) * normal).normalized()
                ro = hit_point + normal * (10.0 * MIN_HIT_DISTANCE)
                inv_rd = ti.math.vec3(_safe_inverse(rd[0]),
                                      _safe_inverse(rd[1]),
                                      _safe_inverse(rd[2]))
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
            has_tri, 0, has_bez,
            t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
            tri_pos, tri_colors, tri_extra, num_colored_triangles,
            t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
            tri_pos, tri_colors, tri_extra,
            b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan, b_first_leaf,
            circuit_meta, circuit_colors, edges_2d, edge_offsets
        )
        acc[0] += weight_seg * glow_rgb[0]
        acc[1] += weight_seg * glow_rgb[1]
        acc[2] += weight_seg * glow_rgb[2]
        if (not done) and (not bounced) and (num_hits < KBUF):
            done = True
    return acc, weight


@ti.kernel
def render_no_pn_stbvh(
        # Triangle STBVH + packed geometry.
        t_nodes: ti.types.ndarray(), t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int,
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_extra: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        # Bezier STBVH + packed geometry.
        b_nodes: ti.types.ndarray(), b_node_miss: ti.types.ndarray(),
        b_leaf_prim: ti.types.ndarray(), b_leaf_tspan: ti.types.ndarray(),
        b_first_leaf: int,
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_offsets: ti.types.ndarray(),
        # Per-frame camera and pixel scale.
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(),
        # Render parameters.
        time_start: int, time_end: int, width: int, height: int,
        half_screen_w: float, half_screen_h: float,
        layer_offset_triangles: float,
        max_bounces: int, transparent: int,
        has_tri: ti.i32, has_bez: ti.i32,
        # Fragment shading (compile-time): 0 = baked vertex colours (default);
        # 1 = material-shade each triangle hit per fragment.
        frag_shading: ti.template(),
        tri_mat_id: ti.types.ndarray(), tri_mat: ti.types.ndarray(),
        light_pos: ti.types.ndarray(), light_col: ti.types.ndarray(),
        num_lights: int,
        # Anti-alias level: a^2 jittered sub-pixel rays averaged per pixel.
        aa_level: int,
        tonemapping: ti.template(),
        tonemap_exposure: ti.f32,
        # Output buffer, pre-filled with the background; blended in place.
        out: ti.types.ndarray()):
    """Deterministic renderer for batches with no PN patches (flat triangles
    and/or bezier circuits). Identical output to ``render_scene_stbvh`` on such
    a batch, without the PN Matrix Pencil solver in its call graph."""
    pixels_per_frame = width * height
    num_rays = (time_end - time_start) * pixels_per_frame
    inv_aa = 1.0 / ti.cast(aa_level, ti.f32)
    inv_samples = inv_aa * inv_aa

    for ray_id in range(num_rays):
        f_rel = ray_id // pixels_per_frame
        p = ray_id - f_rel * pixels_per_frame
        f = time_start + f_rel
        ff = ti.cast(f, ti.f32)
        py = p // width
        px = p - py * width
        pixel_size_per_t = pixel_world_scale[f]

        # Average a^2 rays on a regular sub-pixel grid (matches super-sampling
        # at ``aa_level`` and averaging down, with no super-sampled buffer).
        csum = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        asum = 0.0
        for si in range(aa_level):
            for sj in range(aa_level):
                jx = (ti.cast(si, ti.f32) + 0.5) * inv_aa
                jy = (ti.cast(sj, ti.f32) + 0.5) * inv_aa
                ro, rd = _generate_ray(f, px, py, jx, jy,
                                       half_screen_w, half_screen_h,
                                       cam_origin, screen_point,
                                       pixel_basis_x, pixel_basis_y)
                inv_rd = ti.math.vec3(_safe_inverse(rd[0]),
                                      _safe_inverse(rd[1]),
                                      _safe_inverse(rd[2]))
                acc, weight = _trace_no_pn_ray(
                    ro, rd, inv_rd, f, ff, pixel_size_per_t,
                    layer_offset_triangles, max_bounces,
                    t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                    t_first_leaf, tri_pos, tri_norm, tri_extra, tri_colors,
                    tri_uvs, tri_tex_meta, textures, num_colored_triangles,
                    b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                    b_first_leaf, circuit_meta, circuit_colors,
                    circuit_border_colors, edges_2d, edge_offsets,
                    has_tri, has_bez,
                    frag_shading, tri_mat_id, tri_mat,
                    light_pos, light_col, num_lights)
                for ci in ti.static(range(4)):
                    csum[ci] += (acc[ci] * 255.0
                                 + weight * ti.cast(out[f_rel, p, ci], ti.f32))
                if transparent != 0:
                    asum += ((1.0 - weight) * 255.0
                             + weight * ti.cast(out[f_rel, p, 4], ti.f32))

        color_final = finalize_pixel_color(csum, inv_samples, tonemapping, tonemap_exposure)
        for ci in ti.static(range(4)):
            out[f_rel, p, ci] = ti.cast(color_final[ci], ti.u8)
        if transparent != 0:
            out[f_rel, p, 4] = ti.cast(
                ti.math.clamp(asum * inv_samples + 0.5, 0.0, 255.0), ti.u8)
