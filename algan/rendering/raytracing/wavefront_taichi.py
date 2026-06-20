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
    MAX_SURFACES_PER_RAY,
    MIN_ALPHA,
    MIN_HIT_DISTANCE,
    MIN_WEIGHT,
    PN_SEAM_DEPTH_EPSILON,
    _bezier_normal,
    _collect_hits,
    _collect_hits_tri,
    _comes_after,
    _generate_ray,
    _pn_normal,
    _safe_inverse,
    _sample_circuit_color,
    _triangle_color,
    _triangle_extra,
    _triangle_normal,
)

# Per-ray status codes (rs_int column 2).
_ACTIVE = 0
_DONE = 1


@ti.kernel
def wf_gen_triangle(
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_screen_w: float, half_screen_h: float, max_bounces: int,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
        rs_int: ti.types.ndarray()):
    """Initialise per-ray state with the primary camera ray (mirrors the
    per-ray setup at the top of ``render_triangles_stbvh``)."""
    pixels_per_frame = width * height
    num_rays = rs_ro.shape[0]
    for r in range(num_rays):
        f_rel = r // pixels_per_frame
        p = r - f_rel * pixels_per_frame
        f = time_start + f_rel
        py = p // width
        px = p - py * width
        ro, rd = _generate_ray(f, px, py, 0.5, 0.5,
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
        time_start: int, width: int, height: int,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_sca: ti.types.ndarray(), rs_int: ti.types.ndarray(),
        rs_kt: ti.types.ndarray(), rs_kl: ti.types.ndarray(),
        rs_ka: ti.types.ndarray(), rs_kb: ti.types.ndarray(),
        rs_kp: ti.types.ndarray(), rs_kf: ti.types.ndarray()):
    """Gather the KBUF nearest hits for each active ray into global state.
    No shading state in its call graph -> few live registers."""
    pixels_per_frame = width * height
    for i in range(num_active):
        r = active[i]
        ro = ti.math.vec3(rs_ro[r, 0], rs_ro[r, 1], rs_ro[r, 2])
        rd = ti.math.vec3(rs_rd[r, 0], rs_rd[r, 1], rs_rd[r, 2])
        inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                              _safe_inverse(rd[2]))
        t_prev = rs_sca[r, 1]
        layer_prev = rs_sca[r, 2]
        f = time_start + r // pixels_per_frame
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
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_extra: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
        rs_int: ti.types.ndarray(),
        rs_kt: ti.types.ndarray(), rs_kl: ti.types.ndarray(),
        rs_ka: ti.types.ndarray(), rs_kb: ti.types.ndarray(),
        rs_kp: ti.types.ndarray(), rs_kf: ti.types.ndarray()):
    """Drain the gathered hits front-to-back, blending and bouncing exactly as
    the megakernel's inner loop. No traversal in its call graph."""
    pixels_per_frame = width * height
    for i in range(num_active):
        r = active[i]
        num_hits = rs_int[r, 3]
        if num_hits > 0:
            f = time_start + r // pixels_per_frame
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
                edge_hit = (flags >> 2) & 1

                if (edge_hit == 1) and (t_hit - seam_t <= DEPTH_TIE_EPSILON):
                    t_prev = t_hit
                    layer_prev = hit_layer
                    continue
                seam_t = t_hit if edge_hit == 1 else -1e30

                w0 = 1.0 - a - b
                color, alpha = _triangle_color(f, prim, w0, a, b, tri_colors)
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


@ti.kernel
def wf_composite(
        time_start: int, width: int, height: int, transparent: int,
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
        out: ti.types.ndarray()):
    """Composite each ray's premultiplied accumulator over the pre-filled
    background (mirrors the tail of ``render_triangles_stbvh``)."""
    pixels_per_frame = width * height
    num_rays = rs_acc.shape[0]
    for r in range(num_rays):
        f_rel = r // pixels_per_frame
        p = r - f_rel * pixels_per_frame
        weight = rs_sca[r, 0]
        for ci in ti.static(range(4)):
            bg = ti.cast(out[f_rel, p, ci], ti.f32)
            val = rs_acc[r, ci] * 255.0 + weight * bg
            out[f_rel, p, ci] = ti.cast(
                ti.math.clamp(val + 0.5, 0.0, 255.0), ti.u8)
        if transparent != 0:
            bg_a = ti.cast(out[f_rel, p, 4], ti.f32)
            val = (1.0 - weight) * 255.0 + weight * bg_a
            out[f_rel, p, 4] = ti.cast(
                ti.math.clamp(val + 0.5, 0.0, 255.0), ti.u8)


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
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
        rs_int: ti.types.ndarray()):
    """Initialise per-ray state with primary rays (general path: rs_sca has a
    5th column, base_dist, initialised to 0)."""
    pixels_per_frame = width * height
    num_rays = rs_ro.shape[0]
    for r in range(num_rays):
        f_rel = r // pixels_per_frame
        p = r - f_rel * pixels_per_frame
        f = time_start + f_rel
        py = p // width
        px = p - py * width
        ro, rd = _generate_ray(f, px, py, 0.5, 0.5,
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


@ti.kernel
def wf_traverse_general(
        active: ti.types.ndarray(), num_active: int,
        t_nodes: ti.types.ndarray(), t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int, tri_pos: ti.types.ndarray(),
        p_nodes: ti.types.ndarray(), p_node_miss: ti.types.ndarray(),
        p_leaf_prim: ti.types.ndarray(), p_leaf_tspan: ti.types.ndarray(),
        p_first_leaf: int, pn_ctrl: ti.types.ndarray(),
        b_nodes: ti.types.ndarray(), b_node_miss: ti.types.ndarray(),
        b_leaf_prim: ti.types.ndarray(), b_leaf_tspan: ti.types.ndarray(),
        b_first_leaf: int, circuit_meta: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_offsets: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(),
        layer_offset_triangles: float, layer_offset_pn: float,
        has_tri: ti.i32, has_pn: ti.i32, has_bez: ti.i32,
        time_start: int, width: int, height: int,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_sca: ti.types.ndarray(), rs_int: ti.types.ndarray(),
        rs_kt: ti.types.ndarray(), rs_kl: ti.types.ndarray(),
        rs_ka: ti.types.ndarray(), rs_kb: ti.types.ndarray(),
        rs_kp: ti.types.ndarray(), rs_kf: ti.types.ndarray()):
    """Gather KBUF nearest hits across all three BVHs for each active ray
    (reuses the unchanged general ``_collect_hits``, Matrix Pencil solver included)."""
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
        f = time_start + r // pixels_per_frame
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
            pn_ctrl,
            b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan, b_first_leaf,
            circuit_meta, edges_2d, edge_offsets, has_tri, has_pn, has_bez)
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
def wf_shade_general(
        active: ti.types.ndarray(), num_active: int,
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_extra: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        pn_ctrl: ti.types.ndarray(), pn_norm: ti.types.ndarray(),
        pn_extra: ti.types.ndarray(), pn_colors: ti.types.ndarray(),
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
        rs_int: ti.types.ndarray(),
        rs_kt: ti.types.ndarray(), rs_kl: ti.types.ndarray(),
        rs_ka: ti.types.ndarray(), rs_kb: ti.types.ndarray(),
        rs_kp: ti.types.ndarray(), rs_kf: ti.types.ndarray()):
    """Drain gathered hits front-to-back exactly as ``render_scene_stbvh``'s
    inner loop, with per-geometry-type shading and mirror bounces."""
    pixels_per_frame = width * height
    for i in range(num_active):
        r = active[i]
        num_hits = rs_int[r, 3]
        if num_hits > 0:
            f = time_start + r // pixels_per_frame
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
                    color, alpha = _triangle_color(f, prim, w0, a, b,
                                                   tri_colors)
                    reflectivity, _rough = _triangle_extra(f, prim, w0, a, b,
                                                           tri_extra)
                elif htype == 2:
                    w0 = 1.0 - a - b
                    color, alpha = _triangle_color(f, prim, w0, a, b,
                                                   pn_colors)
                    reflectivity, _rough = _triangle_extra(f, prim, w0, a, b,
                                                           pn_extra)
                else:
                    color, alpha = _sample_circuit_color(
                        prim, f, a, b, border,
                        circuit_meta, circuit_colors, circuit_border_colors)

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
                    elif htype == 2:
                        normal = _pn_normal(f, prim, a, b, pn_norm, pn_ctrl)
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
