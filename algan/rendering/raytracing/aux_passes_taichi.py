"""The auxiliary-pass (AOV) trace: one pinhole ray per output pixel centre.

Compositing passes -- planar depth, a camera-space normal and the id of the
Mob behind a pixel -- follow Cycles' ``pass_alpha_threshold`` convention: the
pass records the FIRST surface along the pixel-centre ray whose colour alpha is
at least ``alpha_threshold`` (0.5 as the host calls it), and a pixel whose ray
finds none is a miss.

One self-contained kernel, independent of which renderer drew the frame: the
ray is always the pixel-CENTRE pinhole ray at the OUTPUT resolution, whatever
``samples_per_pixel``, the anti-aliasing route or a camera lens say, so a
deterministic and a path-traced render of the same scene produce the same
passes. The peel is ``_shadow_gather_occluded``'s (``raytrace_kernels_taichi``):
``_collect_hits`` gathers up to ``kbuf`` hits per traversal of the untrimmed
trees, the drain visits them in the renderers' ``_comes_after`` order with
the renderers' seam merge, and each surface's alpha is fetched exactly the way
both renderers composite it (``_tri_color_g`` with the renderers' mip
footprint, ``_circuit_alpha`` for circuits). Camera rays carry no source
identity and are not shadow rays, so both of those arms compile out of the
gather.

Two deliberate differences from the renderers' camera peel:

* path-tracer area-light panels (triangles at or past ``quad_base``) are not
  there: the deterministic renderer has no such geometry, and the passes must
  agree. A panel is flagged opaque, so the gather that found one stopped
  collecting at it; a skipped panel therefore forces the next gather rather
  than letting a short buffer end the ray.
* the host scales a circuit's anti-crack outline dilation (a fixed 0.6 of a
  RENDER pixel inside ``_collect_hits``) down to nothing, keeping every stroke
  its world width (``aux_passes.trace_aux_passes``). A centre sample is inside a
  filled circuit or it is not; the dilation exists for the renderers' coverage
  and would otherwise make the passes depend on the AA route.

Outputs are written with rows TOP-DOWN (the kernels' ``py`` counts from the
bottom), frame-major, at the output resolution:

* ``aux_depth [F, H*W]`` f32 -- planar depth along the camera's forward axis
  (``(base_dist + t_hit) * cos``), 0 on a miss (the host writes +inf there);
* ``aux_normal [F, H*W, 3]`` f32 -- the unit shading normal turned toward the
  viewer, in camera space (x screen right, y screen up, z toward the camera);
  zero on a miss;
* ``aux_mob [F, H*W]`` i32 -- the source Mob id through the host's optional
  tables, -1 when unknown, -2 on a miss.
"""
from algan.rendering.raytracing.arena_args_taichi import (
    ArenaView,
    arena_packed,
)
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    NODE_ARG,
    _bezier_normal,
    _circuit_alpha,
    _collect_hits,
    _comes_after,
    _generate_ray,
    _safe_inverse,
    depth_tie_epsilon,
    kbuf,
    max_surfaces_per_ray,
)
from algan.rendering.raytracing.shading_taichi import (
    _USER_PIPELINE_BASE,
    _prep_normal,
    _two_sided_normal,
)
from algan.rendering.raytracing.texture_mips_taichi import _triangle_uv_footprint
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _tri_color_g,
    _tri_normal_g,
)
from algan.taichi_compat import ti

#: ``aux_mob`` value of a pixel whose ray found no surface at or above the
#: alpha threshold.
AUX_MISS_MOB = -2
#: ``aux_mob`` value of a hit whose source Mob is not known (no table, or the
#: table has no entry for the surface).
AUX_UNKNOWN_MOB = -1


@ti.kernel
def aux_trace_arena(
        num_cells: ti.i32, time_start: ti.i32, width: ti.i32, height: ti.i32,
        half_screen_w: ti.f32, half_screen_h: ti.f32,
        near_clip: ti.f32, far_clip: ti.f32, alpha_threshold: ti.f32,
        layer_offset_triangles: ti.f32, num_colored_triangles: ti.i32,
        # Multiplier on pixel_world_scale for the circuit classification's
        # pixel size (the host shrinks the outline dilation with it and grows
        # the border column of its circuit_meta copy by the inverse), and the
        # one for the texture mip footprint (the render's AA factor, so the
        # footprint is one OUTPUT pixel on every route).
        pixel_size_mul: ti.f32, footprint_mul: ti.f32,
        # First path-tracer area-light panel triangle (INT32_MAX: none).
        quad_base: ti.i32,
        # Valid lengths of the two source-id tables (0: table absent).
        num_tri_src: ti.i32, num_circ_src: ti.i32,
        t_nodes: NODE_ARG, t_first_leaf: ti.i32,
        b_nodes: NODE_ARG, b_first_leaf: ti.i32,
        refit: ti.template(), has_tri: ti.template(), has_bez: ti.template(),
        aux_depth: ti.types.ndarray(),
        aux_normal: ti.types.ndarray(),
        aux_mob: ti.types.ndarray(),
        arena_f32: ti.types.ndarray(),
        arena_i32: ti.types.ndarray(),
        aoff: ti.types.ndarray(),
        ashp: ti.types.ndarray()):
    """First alpha-threshold hit of every output pixel's centre ray.

    ``num_cells = frames * width * height``; cell ``c`` is frame
    ``time_start + c // (width * height)`` (batch-relative, as every camera
    array and per-frame table is indexed) and pixel ``c % (width * height)``
    counted from the bottom row, the renderers' order. See the module
    docstring for the outputs.
    """
    # Arena-bound parameters (arena_args_taichi): each name is
    # rebound to a window into its dtype's buffer, at the offset
    # the host wrote into aoff. Order is _AUX_TRACE_ARENA's.
    t_leaf_prim = ti.static(ArenaView(arena_i32, aoff[0], (ashp[0],)))
    t_leaf_tspan = ti.static(ArenaView(arena_i32, aoff[1], (ashp[1],)))
    tri_pos = ti.static(ArenaView(arena_f32, aoff[2], (ashp[2], ashp[3], ashp[4])))
    tri_norm = ti.static(ArenaView(arena_f32, aoff[3], (ashp[5], ashp[6], ashp[7])))
    tri_colors = ti.static(ArenaView(
        arena_f32, aoff[4], (ashp[8], ashp[9], ashp[10], ashp[11])))
    tri_uvs = ti.static(ArenaView(arena_f32, aoff[5], (ashp[12], ashp[13], ashp[14])))
    tri_tex_meta = ti.static(ArenaView(arena_i32, aoff[6], (ashp[15], ashp[16])))
    textures = ti.static(ArenaView(arena_f32, aoff[7], (ashp[17], ashp[18], ashp[19])))
    tri_mat_id = ti.static(ArenaView(arena_i32, aoff[8], (ashp[20], ashp[21])))
    tri_mat = ti.static(ArenaView(arena_f32, aoff[9], (ashp[22], ashp[23], ashp[24])))
    tri_obj = ti.static(ArenaView(arena_i32, aoff[10], (ashp[25], ashp[26])))
    b_leaf_prim = ti.static(ArenaView(arena_i32, aoff[11], (ashp[27],)))
    b_leaf_tspan = ti.static(ArenaView(arena_i32, aoff[12], (ashp[28],)))
    circuit_meta = ti.static(ArenaView(
        arena_f32, aoff[13], (ashp[29], ashp[30], ashp[31])))
    circuit_colors = ti.static(ArenaView(
        arena_f32, aoff[14], (ashp[32], ashp[33], ashp[34], ashp[35])))
    circuit_border_colors = ti.static(ArenaView(
        arena_f32, aoff[15], (ashp[36], ashp[37], ashp[38], ashp[39])))
    edges_2d = ti.static(ArenaView(arena_f32, aoff[16], (ashp[40], ashp[41], ashp[42])))
    edge_accel = ti.static(ArenaView(arena_i32, aoff[17], (ashp[43],)))
    cam_origin = ti.static(ArenaView(arena_f32, aoff[18], (ashp[44], ashp[45])))
    screen_point = ti.static(ArenaView(arena_f32, aoff[19], (ashp[46], ashp[47])))
    pixel_basis_x = ti.static(ArenaView(arena_f32, aoff[20], (ashp[48], ashp[49])))
    pixel_basis_y = ti.static(ArenaView(arena_f32, aoff[21], (ashp[50], ashp[51])))
    pixel_world_scale = ti.static(ArenaView(arena_f32, aoff[22], (ashp[52],)))
    tri_src = ti.static(ArenaView(arena_i32, aoff[23], (ashp[53],)))
    circ_src = ti.static(ArenaView(arena_i32, aoff[24], (ashp[54],)))
    pixels_per_frame = width * height
    for cell in range(num_cells):
        f_rel = cell // pixels_per_frame
        p = cell - f_rel * pixels_per_frame
        f = time_start + f_rel
        py = p // width
        px = p - py * width
        ro, rd = _generate_ray(f, px, py, 0.5, 0.5, half_screen_w,
                               half_screen_h, cam_origin, screen_point,
                               pixel_basis_x, pixel_basis_y)
        # The camera basis, from the CAMERA origin (before any near-clip
        # advance -- the same forward _axis_cos and the near plane use).
        cam = ti.math.vec3(cam_origin[f, 0], cam_origin[f, 1], cam_origin[f, 2])
        fwd = (ti.math.vec3(screen_point[f, 0], screen_point[f, 1],
                            screen_point[f, 2]) - cam).normalized()
        cos_ax = rd.dot(fwd)
        pixel_size_per_t = pixel_world_scale[f] * pixel_size_mul * cos_ax
        footprint_scale = pixel_world_scale[f] * footprint_mul
        base_dist = 0.0
        if near_clip > 0.0:
            # The renderers' planar near plane: the origin advances to it and
            # the skipped distance seeds base_dist.
            t_near = near_clip / ti.max(cos_ax, 1e-6)
            ro = ro + rd * t_near
            base_dist = t_near
        inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                              _safe_inverse(rd[2]))
        ff = ti.cast(f, ti.f32)

        found = 0
        found_type = 0
        found_prim = 0
        depth = 0.0
        nrm = ti.math.vec3(0.0, 0.0, 0.0)
        t_prev = 0.0
        layer_prev = 1e30
        seam_t = -1e30
        step = 0
        alive = 1
        while (alive == 1) and (step < max_surfaces_per_ray):
            kb_t = ti.Vector([0.0] * kbuf)
            kb_layer = ti.Vector([0.0] * kbuf)
            kb_prim = ti.Vector([0] * kbuf)
            kb_flags = ti.Vector([0] * kbuf)
            kb_a = ti.Vector([0.0] * kbuf)
            kb_b = ti.Vector([0.0] * kbuf)
            num_hits = _collect_hits(
                refit, ro, rd, inv_rd, f, ff, t_prev, layer_prev,
                pixel_size_per_t, base_dist, layer_offset_triangles,
                kb_t, kb_layer, kb_prim, kb_flags, kb_a, kb_b,
                # The node_miss slots are never indexed by the gather, so the
                # leaf array stands in for them rather than binding two more.
                t_nodes, t_leaf_prim, t_leaf_prim, t_leaf_tspan, t_first_leaf,
                tri_pos,
                b_nodes, b_leaf_prim, b_leaf_prim, b_leaf_tspan, b_first_leaf,
                circuit_meta, edges_2d, edge_accel, has_tri, has_bez,
                1e30, -1e30,
                # No source identity on a camera ray: (-1, _, 0) compiles the
                # identity-aware acceptance floor out; tri_pos is never read.
                -1, -1, 0.0, 0.0, tri_pos, 0,
                # Not a shadow ray: a non-casting primitive stays visible.
                0)
            if num_hits == 0:
                alive = 0
            # A skipped area-light panel is opaque-flagged, so this gather
            # stopped collecting at it: a short buffer no longer proves the
            # ray is exhausted.
            skipped_opaque = 0
            drained = 0
            while (alive == 1) and (drained < num_hits) \
                    and (step < max_surfaces_per_ray):
                step += 1
                # Nearest unconsumed slot, scalar-tracked with ti.static
                # selects so the kb_* vectors are never dynamically indexed
                # (the _shadow_gather_occluded / wavefront_shade drain).
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
                if (far_clip > 0.0) and (base_dist + t_hit > far_clip):
                    # Past the far plane (a distance along the ray from the
                    # camera, as both renderers measure it); hits drain
                    # front to back, so nothing nearer is left.
                    alive = 0
                else:
                    htype = flags & 3
                    edge_hit = (flags >> 2) & 1
                    border = (flags >> 3) & 1
                    if (edge_hit == 1) and (t_hit - seam_t <= depth_tie_epsilon):
                        # The renderers' seam merge: the same crossing seen
                        # through the neighbouring triangle of a shared edge.
                        t_prev = t_hit
                        layer_prev = hit_layer
                    elif (htype == 1) and (prim >= quad_base):
                        skipped_opaque = 1
                        t_prev = t_hit
                        layer_prev = hit_layer
                    else:
                        seam_t = t_hit if edge_hit == 1 else -1e30
                        w0 = 1.0 - a - b
                        alpha = 0.0
                        du = 0.0
                        dv = 0.0
                        color4 = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                        if htype == 1:
                            du, dv = _triangle_uv_footprint(
                                0, f, prim, rd, (base_dist + t_hit) * footprint_scale,
                                tri_pos, tri_uvs, tri_tex_meta,
                                num_colored_triangles)
                            color4, alpha = _tri_color_g(
                                0, f, prim, w0, a, b, tri_colors, tri_colors,
                                tri_uvs, tri_tex_meta, textures,
                                num_colored_triangles, du, dv)
                        else:
                            alpha = _circuit_alpha(
                                prim, f, a, b, border, circuit_meta,
                                circuit_colors, circuit_border_colors)
                        alpha = ti.math.clamp(alpha, 0.0, 1.0)
                        if alpha >= alpha_threshold:
                            alive = 0
                            found = 1
                            found_type = htype
                            found_prim = prim
                            depth = (base_dist + t_hit) * cos_ax
                            view_dir = -rd
                            if htype == 1:
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
                                snrm = _tri_normal_g(
                                    0, f, prim, w0, a, b, tri_norm, tri_pos,
                                    tri_uvs, tri_tex_meta, textures,
                                    num_colored_triangles, du, dv)
                                if snrm.norm() <= 1e-12:
                                    snrm = fnrm
                                flat = 0.0
                                pid = tri_mat_id[f % tri_mat_id.shape[0], prim]
                                if pid < _USER_PIPELINE_BASE:
                                    flat = tri_mat[f % tri_mat.shape[0], prim, 10]
                                # Turned toward the viewer on the GEOMETRIC
                                # normal (_faces_viewer), as the renderers
                                # turn two-sided geometry, then blended
                                # toward the face normal exactly as they do.
                                nrm = _two_sided_normal(snrm, fnrm, flat, view_dir)
                                nrm = _prep_normal(nrm, fnrm, flat, view_dir)
                                away = nrm.dot(rd)
                                if away > 0.0:
                                    # A smooth normal can still lean past the
                                    # silhouette; bend it onto the view plane
                                    # so the pass never faces away.
                                    bent = nrm - rd * away
                                    if bent.norm() > 1e-6:
                                        nrm = bent.normalized()
                                    else:
                                        nrm = fnrm
                                        if nrm.dot(rd) > 0.0:
                                            nrm = -nrm
                            else:
                                nrm = _bezier_normal(f, prim, circuit_meta)
                                if nrm.norm() > 1e-9:
                                    nrm = nrm.normalized()
                                if nrm.dot(rd) > 0.0:
                                    nrm = -nrm
                        else:
                            t_prev = t_hit
                            layer_prev = hit_layer
            if (alive == 1) and (num_hits < kbuf) and (skipped_opaque == 0):
                alive = 0

        # Camera space: x screen right, y screen up, z toward the camera
        # (OpenGL view space). pixel_basis_x/y are screen_half_height times
        # the unit right/up vectors; orthonormalised against forward so a
        # scaled camera cannot shear the result.
        toward = -fwd
        bx = ti.math.vec3(pixel_basis_x[f, 0], pixel_basis_x[f, 1],
                          pixel_basis_x[f, 2])
        by = ti.math.vec3(pixel_basis_y[f, 0], pixel_basis_y[f, 1],
                          pixel_basis_y[f, 2])
        right = bx - fwd * bx.dot(fwd)
        if right.norm() > 1e-12:
            right = right.normalized()
        up = by - fwd * by.dot(fwd) - right * by.dot(right)
        if up.norm() > 1e-12:
            up = up.normalized()
        mob = -2
        if found == 1:
            mob = -1
            if found_type == 1:
                if (num_tri_src > 0) and (found_prim < tri_obj.shape[1]):
                    sid = ti.cast(tri_obj[f % tri_obj.shape[0], found_prim], ti.i32)
                    if (sid >= 0) and (sid < num_tri_src):
                        mob = tri_src[sid]
            else:
                if (found_prim >= 0) and (found_prim < num_circ_src):
                    mob = circ_src[found_prim]
            if mob < -1:
                mob = -1
        out_p = (height - 1 - py) * width + px
        aux_depth[f_rel, out_p] = depth
        aux_normal[f_rel, out_p, 0] = nrm.dot(right)
        aux_normal[f_rel, out_p, 1] = nrm.dot(up)
        aux_normal[f_rel, out_p, 2] = nrm.dot(toward)
        aux_mob[f_rel, out_p] = mob


#: What ``aux_trace`` binds through the arena, in offset-table order:
#: ``aoff[i]`` is the i-th entry's element offset into its dtype's
#: buffer and ``ashp`` holds their shapes end to end. The kernel's
#: binding prologue reads those slots by literal index, so the two
#: are one edit apart -- ``tests/unit_tests/test_arena_args.py``
#: fails if they stop agreeing.
_AUX_TRACE_ARENA = (
    ("t_leaf_prim", "i32", 1),
    ("t_leaf_tspan", "i32", 1),
    ("tri_pos", "f32", 3),
    ("tri_norm", "f32", 3),
    ("tri_colors", "f32", 4),
    ("tri_uvs", "f32", 3),
    ("tri_tex_meta", "i32", 2),
    ("textures", "f32", 3),
    ("tri_mat_id", "i32", 2),
    ("tri_mat", "f32", 3),
    ("tri_obj", "i32", 2),
    ("b_leaf_prim", "i32", 1),
    ("b_leaf_tspan", "i32", 1),
    ("circuit_meta", "f32", 3),
    ("circuit_colors", "f32", 4),
    ("circuit_border_colors", "f32", 4),
    ("edges_2d", "f32", 3),
    ("edge_accel", "i32", 1),
    ("cam_origin", "f32", 2),
    ("screen_point", "f32", 2),
    ("pixel_basis_x", "f32", 2),
    ("pixel_basis_y", "f32", 2),
    ("pixel_world_scale", "f32", 1),
    ("tri_src", "i32", 1),
    ("circ_src", "i32", 1),
)

#: The argument list the launch site passes (``aux_passes.trace_aux_passes``).
#: The kept (non-arena) names appear in the kernel's own declaration order.
_AUX_TRACE_PARAMS = (
    "num_cells", "time_start", "width", "height", "half_screen_w",
    "half_screen_h", "near_clip", "far_clip", "alpha_threshold",
    "layer_offset_triangles", "num_colored_triangles", "pixel_size_mul",
    "footprint_mul", "quad_base", "num_tri_src", "num_circ_src",
    "t_nodes", "t_leaf_prim", "t_leaf_tspan", "t_first_leaf", "tri_pos",
    "tri_norm", "tri_colors", "tri_uvs", "tri_tex_meta", "textures",
    "tri_mat_id", "tri_mat", "tri_obj",
    "b_nodes", "b_leaf_prim", "b_leaf_tspan", "b_first_leaf", "circuit_meta",
    "circuit_colors", "circuit_border_colors", "edges_2d", "edge_accel",
    "cam_origin", "screen_point", "pixel_basis_x", "pixel_basis_y",
    "pixel_world_scale", "tri_src", "circ_src",
    "refit", "has_tri", "has_bez", "aux_depth", "aux_normal", "aux_mob",
)

_aux_trace_launch = arena_packed(
    __name__, "aux_trace_arena", _AUX_TRACE_PARAMS, _AUX_TRACE_ARENA)


def aux_trace(*args):
    """Pack the arena-bound arguments, then launch ``aux_trace_arena``.

    Takes ``_AUX_TRACE_PARAMS`` positionally; see `arena_args_taichi`.
    """
    return _aux_trace_launch(*args)
