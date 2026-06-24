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
        ray_offset: int,
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
    the megakernel's inner loop. No traversal in its call graph."""
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
        ray_offset: int,
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
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


@ti.kernel
def wf_composite_accum(
        time_start: int, width: int, height: int, transparent: int,
        ray_offset: int,
        pix_accum: ti.types.ndarray(), out: ti.types.ndarray()):
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
        for ci in ti.static(range(4)):
            bg = ti.cast(out[f_rel, p, ci], ti.f32)
            val = pix_accum[r, ci] * 255.0 + weight * bg
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
        ray_offset: int, num_primary: int,
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
        has_tri: ti.i32, has_pn: ti.i32, has_bez: ti.i32,
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
        rs_used: ti.types.ndarray()):
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
                    color, alpha = _flat_triangle_color(f, prim, w0, a, b,
                                                        tri_colors, tri_uvs, tri_tex_meta,
                                                        textures, num_colored_triangles)
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
                    if ti.static(shadows != 0):
                        ff = ti.cast(f, ti.f32)
                        pixel_size_per_t = pixel_world_scale[f]
                        if (htype == 1) or (htype == 2):
                            # Smooth shading normal and the *geometric* face
                            # normal of the hit facet/patch.
                            snrm = ti.math.vec3(0.0, 0.0, 0.0)
                            fnrm = ti.math.vec3(0.0, 0.0, 0.0)
                            if htype == 1:
                                snrm = _triangle_normal(f, prim, 1.0 - a - b,
                                                        a, b, tri_norm, tri_pos)
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
                                snrm = _pn_normal(f, prim, a, b, pn_norm,
                                                  pn_ctrl)
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
                            ior = _corner_ior(f, prim, 1.0 - a - b, a, b,
                                              tri_extra)
                        else:
                            ior = _corner_ior(f, prim, 1.0 - a - b, a, b,
                                              pn_extra)
                        if ior > 1.0 + 1e-4:
                            is_glass = True
                            if htype == 1:
                                gnrm = _triangle_normal(f, prim, 1.0 - a - b,
                                                        a, b, tri_norm, tri_pos)
                            else:
                                gnrm = _pn_normal(f, prim, a, b, pn_norm,
                                                  pn_ctrl)
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
            for k in ti.static(range(4)):
                ti.atomic_add(pix_accum[pix, k], rs_acc[r, k])
            ti.atomic_add(pix_accum[pix, 4], rs_sca[r, 0])
            rs_int[r, 2] = _DONE
