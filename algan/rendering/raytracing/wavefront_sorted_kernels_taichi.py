"""UNSUPPORTED legacy variant: Cycles-style sorted material dispatch for the
deterministic wavefront.

This pipeline is no longer maintained and no longer works; the monolithic
``wavefront_shade`` kernel -- which handles custom scatter and normal maps
itself, and is faster on the built-in materials -- is the only supported
deterministic shade path. The module is kept for reference;
``set_material_sorting(True)`` still routes here, unsupported. Original
design rationale follows.

The monolithic fragment-shading shade kernel (``wavefront_shade``) compiles
*every* material's lighting model, the shadow traversal call graph and the
bounce logic into one kernel and dispatches materials with a *runtime* switch
(``_run_frag_pipeline``): a warp whose rays hit different materials serialises
through every branch, and the union of all that code sets the kernel's register
footprint (the occupancy killer on these megakernels). This module splits the
per-hit work the way Blender's Cycles does:

* :func:`wf_peel` -- *surface evaluation*: drains each ray's K-buffer doing only
  material-independent work (seam/tie de-dup, bezier-circuit compositing, and
  albedo / normal / reflectivity / IOR sampling of the next triangle-or-PN hit),
  then **suspends** the ray at that hit as a pending *shading event*: a compact
  per-ray hit record (``rs_hit``) plus a sort key ``(geometry type << 8) |
  material pipeline id`` (``rs_key``). No lighting, no BVH traversal.
* the **host** buckets the pending events by key (cheap ``nonzero`` masks over
  the key array -- the "sort by material id"),
* :func:`wf_shadow_event` -- one launch over *all* pending events (occlusion is
  material-independent): fires the per-light shadow rays for the event and
  packs the visibility bits into ``rs_vis``. This is the only stage that pays
  for the heavy ``_shadow_occluded`` -> PN-solver call graph.
* :func:`wf_shade_event` -- one launch **per material bucket**, with the
  bucket's composed pipeline func and scatter func injected as ``ti.template()``
  arguments: Taichi compiles a dedicated kernel per material, so there is no
  runtime material switch, a warp never mixes materials, and each kernel
  carries only its own material's code (it is entirely geometry-free -- the
  event record already holds the interpolated surface attributes). The scatter
  func decides how the ray continues (pass through / mirror bounce / glass
  split); :func:`default_scatter` reproduces the classic
  opacity/reflectivity/Fresnel behaviour and users can override it per pipeline
  (``FragmentStage(..., scatter=...)``) to customise ray bouncing.

Ray lifecycle (``rs_int`` column 2): ``ST_TRAVERSE`` (0, fresh / just bounced /
K-buffer exhausted with possibly more surfaces behind) -> traverse ->
``ST_PEEL`` (2, K-buffer holds unconsumed hits) -> peel -> ``ST_SHADE`` (3,
event pending; ``rs_key`` valid) -> shade -> back to ``ST_PEEL`` (continued
through a transparent surface), ``ST_TRAVERSE`` (bounced) or ``ST_DONE`` (1,
colour + leftover throughput committed to ``pix_accum``). The vertex-shaded
(fragment-shading off) path never enters this module -- it has no per-material
work to sort and keeps the classic single shade kernel byte-identical.

Extra per-ray state (allocated only on this path): ``rs_hit`` [pool, 16]
(0-2 interpolated shading normal, 3-5 geometric face normal, 6-9 albedo
RGB+glow, 10 alpha, 11 reflectivity, 12 index of refraction, 13 t_hit,
14 hit layer, 15 transmission), ``rs_key`` / ``rs_eprim`` [pool] int32, and ``rs_int`` gains a
5th column (4: hits drained from the current K-buffer, so the drain position
survives the peel -> shade -> peel round trip).

Intentional divergence from the monolith: lighting stages here receive the
*normal-mapped* shading normal (``_flat_triangle_normal`` / ``_pn_hit_normal``)
-- the monolith's ``_shade_tri_hit`` predates normal maps and lights with the
unperturbed vertex normal, so normal maps affected only shadows and mirror
directions there. Scenes without normal maps are unaffected (the two normals
are identical then).
"""
import taichi as ti

from algan.rendering.raytracing.raytrace_kernels_taichi import (
    DEPTH_TIE_EPSILON,
    KBUF,
    MAX_SHADOW_LIGHTS,
    MAX_SURFACES_PER_RAY,
    MIN_HIT_DISTANCE,
    MIN_WEIGHT,
    NODE_ARG,
    PN_SEAM_DEPTH_EPSILON,
    _comes_after,
    _sample_circuit_color,
    _shadow_occluded,
)
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _ACTIVE,
    _DONE,
    _flat_corner_ior,
    _flat_corner_transmission,
    _flat_triangle_color,
    _flat_triangle_extra,
    _flat_triangle_normal,
    _pn_hit_color,
    _pn_hit_extra,
    _pn_hit_ior,
    _pn_hit_normal,
    _pn_hit_transmission,
    _reserve_continuation_slot,
    default_scatter,
)

# Ray status codes (rs_int column 2). ST_TRAVERSE/ST_DONE alias the classic
# path's codes so ``wavefront_generate_rays`` / ``wavefront_traverse`` and the
# glass-split spawn slots need no changes.
ST_TRAVERSE = _ACTIVE   # 0: needs a (re-)traverse to refill the K-buffer
ST_DONE = _DONE         # 1: retired; contribution committed to pix_accum
ST_PEEL = 2             # 2: K-buffer holds unconsumed hits
ST_SHADE = 3            # 3: suspended at a material event (rs_key/rs_hit valid)


@ti.kernel
def wf_peel(
        active: ti.types.ndarray(), num_active: int,
        # Flat-triangle shading data (sampling only -- no BVH).
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_extra: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        # PN patch shading data.
        pn_ctrl: ti.types.ndarray(), pn_norm: ti.types.ndarray(),
        pn_extra: ti.types.ndarray(), pn_colors: ti.types.ndarray(),
        # Bezier circuit shading data.
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        # Per-primitive material pipeline ids (the sort keys' low byte).
        tri_mat_id: ti.types.ndarray(), pn_mat_id: ti.types.ndarray(),
        refraction: ti.template(),
        has_tri: ti.template(), has_pn: ti.template(), has_bez: ti.template(),
        time_start: int, width: int, height: int, ray_offset: int,
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
        rs_int: ti.types.ndarray(),
        rs_kt: ti.types.ndarray(), rs_kl: ti.types.ndarray(),
        rs_ka: ti.types.ndarray(), rs_kb: ti.types.ndarray(),
        rs_kp: ti.types.ndarray(), rs_kf: ti.types.ndarray(),
        rs_pix: ti.types.ndarray(), rs_hit: ti.types.ndarray(),
        rs_key: ti.types.ndarray(), rs_eprim: ti.types.ndarray(),
        pix_accum: ti.types.ndarray()):
    """Surface-evaluation stage: resume draining each ray's K-buffer
    front-to-back (same selection, seam de-dup and transparency arithmetic as
    the monolithic kernel), compositing bezier-circuit hits inline and
    suspending the ray at its next triangle/PN *material event* with the
    interpolated surface attributes written to ``rs_hit`` and the sort key to
    ``rs_key``. Consumed K-buffer slots are marked in ``rs_kp`` (and counted in
    ``rs_int[r, 4]``) so the drain position survives suspension."""
    pixels_per_frame = width * height
    for i in range(num_active):
        r = active[i]
        pix = rs_pix[r]
        num_hits = rs_int[r, 3]
        if num_hits == 0:
            # Ray escaped to the background this segment: commit its colour +
            # leftover (background) throughput, then retire.
            for k in ti.static(range(4)):
                ti.atomic_add(pix_accum[pix, k], rs_acc[r, k])
            ti.atomic_add(pix_accum[pix, 4], rs_sca[r, 0])
            ti.atomic_add(pix_accum[pix, 5], rs_sca[r, 5])
            ti.atomic_add(pix_accum[pix, 6], rs_sca[r, 6])
            rs_int[r, 2] = ST_DONE
        else:
            f = time_start + (ray_offset + pix) // pixels_per_frame
            acc = ti.math.vec4(rs_acc[r, 0], rs_acc[r, 1], rs_acc[r, 2],
                               rs_acc[r, 3])
            weight = ti.math.vec3(rs_sca[r, 0], rs_sca[r, 5], rs_sca[r, 6])
            t_prev = rs_sca[r, 1]
            layer_prev = rs_sca[r, 2]
            seam_t = rs_sca[r, 3]
            processed = rs_int[r, 1]
            drained = rs_int[r, 4]

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

            done = False
            suspended = False
            while (drained < num_hits) and (not done) and (not suspended):
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
                rs_kp[r, sel] = -1  # consumption must survive suspension
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

                if htype == 0:
                    # Bezier circuit: composited inline (never material-shaded),
                    # so no event is emitted for it. There is no reflectance
                    # term here, which is sound only because a scene with any
                    # PBR circuit never reaches this pipeline -- see
                    # ``bez_has_reflective``, which gates it on metalness >= 0.
                    color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                    alpha = 0.0
                    if ti.static(has_bez != 0):
                        color, alpha = _sample_circuit_color(
                            prim, f, a, b, border,
                            circuit_meta, circuit_colors,
                            circuit_border_colors)
                    alpha = ti.math.clamp(alpha, 0.0, 1.0)
                    w_glow = ti.max(weight[0],
                                    ti.max(weight[1], weight[2]))
                    acc += ti.math.vec4(weight[0], weight[1], weight[2],
                                        w_glow) * alpha * color
                    weight *= 1.0 - alpha
                    t_prev = t_hit
                    layer_prev = hit_layer
                    if ti.max(weight[0], ti.max(weight[1], weight[2])) \
                            < MIN_WEIGHT:
                        done = True
                else:
                    # Triangle / PN patch: sample every material-independent
                    # surface attribute here and suspend as a shading event.
                    w0 = 1.0 - a - b
                    color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                    alpha = 0.0
                    refl = -1.0
                    ior = 0.0
                    transmission = 0.0
                    ni = ti.math.vec3(0.0, 0.0, 0.0)
                    fn = ti.math.vec3(0.0, 0.0, 0.0)
                    pid = 0
                    if htype == 1:
                        if ti.static(has_tri != 0):
                            color, alpha = _flat_triangle_color(
                                f, prim, w0, a, b, tri_colors, tri_uvs,
                                tri_tex_meta, textures, num_colored_triangles)
                            refl, _rough = _flat_triangle_extra(
                                f, prim, w0, a, b, tri_extra, tri_uvs,
                                tri_tex_meta, textures, num_colored_triangles)
                            ior = _flat_corner_ior(
                                f, prim, w0, a, b, tri_extra, tri_uvs,
                                tri_tex_meta, textures,
                                num_colored_triangles)
                            transmission = _flat_corner_transmission(
                                f, prim, w0, a, b, tri_extra, tri_uvs,
                                tri_tex_meta, textures,
                                num_colored_triangles)
                            ni = _flat_triangle_normal(
                                f, prim, w0, a, b, tri_norm, tri_pos,
                                tri_uvs, tri_tex_meta, textures,
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
                            fn = (v1 - v0).cross(v2 - v0)
                            pid = tri_mat_id[f % tri_mat_id.shape[0], prim]
                    else:
                        if ti.static(has_pn != 0):
                            color, alpha = _pn_hit_color(
                                f, prim, w0, a, b, pn_colors, pn_extra,
                                textures)
                            refl, _rough = _pn_hit_extra(
                                f, prim, w0, a, b, pn_extra, textures)
                            ior = _pn_hit_ior(f, prim, w0, a, b, pn_extra,
                                              textures)
                            transmission = _pn_hit_transmission(
                                f, prim, w0, a, b, pn_extra, textures)
                            ni = _pn_hit_normal(f, prim, a, b, pn_norm,
                                                pn_ctrl, pn_extra, textures)
                            tp = f % pn_ctrl.shape[0]
                            su = ti.math.vec3(0.0, 0.0, 0.0)
                            sv = ti.math.vec3(0.0, 0.0, 0.0)
                            for ci in ti.static(range(3)):
                                su[ci] = (pn_ctrl[tp, prim, 3 + ci]
                                          + 2.0 * a * pn_ctrl[tp, prim, 9 + ci]
                                          + b * pn_ctrl[tp, prim, 15 + ci])
                                sv[ci] = (pn_ctrl[tp, prim, 6 + ci]
                                          + 2.0 * b * pn_ctrl[tp, prim,
                                                              12 + ci]
                                          + a * pn_ctrl[tp, prim, 15 + ci])
                            fn = su.cross(sv)
                            pid = pn_mat_id[f % pn_mat_id.shape[0], prim]
                    for k in ti.static(range(3)):
                        rs_hit[r, k] = ni[k]
                        rs_hit[r, 3 + k] = fn[k]
                    for k in ti.static(range(4)):
                        rs_hit[r, 6 + k] = color[k]
                    rs_hit[r, 10] = alpha
                    rs_hit[r, 11] = refl
                    rs_hit[r, 12] = ior
                    rs_hit[r, 15] = transmission
                    rs_hit[r, 13] = t_hit
                    rs_hit[r, 14] = hit_layer
                    rs_key[r] = (htype << 8) | pid
                    rs_eprim[r] = prim
                    suspended = True

            status = ST_SHADE
            if not suspended:
                # K-buffer exhausted without a material event: same tail as
                # the monolith -- a short buffer means the scene truly had no
                # more surfaces (retire), a full one may hide more (re-trace).
                if (not done) and (num_hits < KBUF):
                    done = True
                if processed >= MAX_SURFACES_PER_RAY:
                    done = True
                status = ST_DONE if done else ST_TRAVERSE

            for k in ti.static(range(4)):
                rs_acc[r, k] = acc[k]
            rs_sca[r, 0] = weight[0]
            rs_sca[r, 1] = t_prev
            rs_sca[r, 2] = layer_prev
            rs_sca[r, 3] = seam_t
            rs_sca[r, 5] = weight[1]
            rs_sca[r, 6] = weight[2]
            rs_int[r, 1] = processed
            rs_int[r, 4] = drained if suspended else 0
            rs_int[r, 2] = status
            if status == ST_DONE:
                for k in ti.static(range(4)):
                    ti.atomic_add(pix_accum[pix, k], acc[k])
                for k in ti.static(range(3)):
                    ti.atomic_add(pix_accum[pix, 4 + k], weight[k])


@ti.kernel
def wf_shadow_event(
        active: ti.types.ndarray(), num_active: int,
        # Triangle STBVH + occlusion data.
        t_nodes: NODE_ARG, t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int,
        tri_pos: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        # PN patch STBVH + occlusion data.
        p_nodes: NODE_ARG, p_node_miss: ti.types.ndarray(),
        p_leaf_prim: ti.types.ndarray(), p_leaf_tspan: ti.types.ndarray(),
        p_first_leaf: int,
        pn_ctrl: ti.types.ndarray(), pn_obb: ti.types.ndarray(),
        pn_colors: ti.types.ndarray(),
        # Bezier STBVH + occlusion data.
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
        rs_sca: ti.types.ndarray(), rs_hit: ti.types.ndarray(),
        rs_pix: ti.types.ndarray(), rs_vis: ti.types.ndarray()):
    """Shadow stage of the sorted pipeline: for every ray suspended at a
    shading event, fire one binary shadow ray per light from the event's
    surface point and pack the per-light occlusion into ``rs_vis`` bit ``li``.
    Occlusion is material-independent, so this runs once over *all* pending
    events (no per-bucket launches) and is the only sorted-path kernel that
    inlines the heavy ``_shadow_occluded`` -> PN-solver call graph -- the
    per-material shade kernels just read the bits. The surface normals come
    from the event record (no re-interpolation), oriented exactly as the
    monolith's inline shadow block."""
    pixels_per_frame = width * height
    for i in range(num_active):
        r = active[i]
        pix = rs_pix[r]
        f = time_start + (ray_offset + pix) // pixels_per_frame
        ff = ti.cast(f, ti.f32)
        ro = ti.math.vec3(rs_ro[r, 0], rs_ro[r, 1], rs_ro[r, 2])
        rd = ti.math.vec3(rs_rd[r, 0], rs_rd[r, 1], rs_rd[r, 2])
        base_dist = rs_sca[r, 4]
        t_hit = rs_hit[r, 13]
        pixel_size_per_t = pixel_world_scale[f]
        snrm = ti.math.vec3(rs_hit[r, 0], rs_hit[r, 1], rs_hit[r, 2])
        fnrm = ti.math.vec3(rs_hit[r, 3], rs_hit[r, 4], rs_hit[r, 5])
        if snrm.norm() > 1e-9:
            snrm = snrm.normalized()
        if snrm.dot(rd) > 0.0:
            snrm = -snrm
        # Orient the geometric normal into the shading normal's hemisphere so
        # a terminator-adjacent shadow ray doesn't graze the uphill facet.
        if fnrm.norm() > 1e-9:
            fnrm = fnrm.normalized()
        if fnrm.dot(snrm) < 0.0:
            fnrm = -fnrm
        spos = ro + t_hit * rd
        sorigin = spos + fnrm * (10.0 * MIN_HIT_DISTANCE)
        tl = f % light_pos.shape[0]
        bits = 0
        for li in range(num_lights):
            if li < MAX_SHADOW_LIGHTS:
                lp = ti.math.vec3(light_pos[tl, li, 0],
                                  light_pos[tl, li, 1],
                                  light_pos[tl, li, 2])
                to_light = lp - spos
                ldist = to_light.norm()
                if ldist > 1e-5:
                    wi = to_light / ldist
                    # Lights below the geometric/shading horizon are skipped
                    # (no direct light to occlude / self-shadow acne).
                    if (fnrm.dot(wi) > 1e-3) and (snrm.dot(wi) > 1e-4):
                        occ = _shadow_occluded(
                            0, sorigin, wi, f, ff,
                            ldist - 20.0 * MIN_HIT_DISTANCE,
                            pixel_size_per_t, base_dist,
                            layer_offset_triangles, layer_offset_pn,
                            has_tri, has_pn, has_bez,
                            t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                            t_first_leaf, tri_pos, tri_colors, tri_uvs,
                            tri_tex_meta, textures, num_colored_triangles,
                            p_nodes, p_node_miss, p_leaf_prim, p_leaf_tspan,
                            p_first_leaf, pn_ctrl, pn_obb, pn_colors,
                            b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                            b_first_leaf, circuit_meta, circuit_colors,
                            circuit_border_colors, edges_2d, edge_accel)
                        if occ > 0.5:
                            bits |= 1 << li
        rs_vis[r] = bits


@ti.kernel
def wf_shade_event(
        active: ti.types.ndarray(), num_active: int,
        # Per-primitive material parameter blocks of this bucket's geometry
        # type (tri_mat or pn_mat), indexed [frame % T, prim, slot].
        params: ti.types.ndarray(),
        light_pos: ti.types.ndarray(), light_col: ti.types.ndarray(),
        num_lights: int,
        pipeline_fn: ti.template(), scatter_fn: ti.template(),
        shadows: ti.template(), refraction: ti.template(),
        time_start: int, width: int, height: int, ray_offset: int,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
        rs_int: ti.types.ndarray(),
        rs_hit: ti.types.ndarray(), rs_eprim: ti.types.ndarray(),
        rs_pix: ti.types.ndarray(), pix_accum: ti.types.ndarray(),
        rs_alloc: ti.types.ndarray(), rs_vis: ti.types.ndarray()):
    """Material stage of the sorted pipeline, launched once per material
    bucket: every ray in ``active`` is suspended at an event of the *same*
    material, whose composed pipeline func and scatter func are compile-time
    template arguments -- a dedicated Taichi kernel per material with no
    runtime material switch and no geometry access (the event record carries
    the surface attributes; ``params`` is the bucket's geometry-type material
    block). Shades the event, applies the scatter's continuation decision
    (pass-through / mirror bounce in place / glass split into the shared
    continuation pool) and commits terminated rays to ``pix_accum``. Pool
    overflow raises ``rs_alloc[1]`` so the host can retry the tile exactly."""
    pixels_per_frame = width * height
    for i in range(num_active):
        r = active[i]
        pix = rs_pix[r]
        f = time_start + (ray_offset + pix) // pixels_per_frame
        ro = ti.math.vec3(rs_ro[r, 0], rs_ro[r, 1], rs_ro[r, 2])
        rd = ti.math.vec3(rs_rd[r, 0], rs_rd[r, 1], rs_rd[r, 2])
        acc = ti.math.vec4(rs_acc[r, 0], rs_acc[r, 1], rs_acc[r, 2],
                           rs_acc[r, 3])
        weight = ti.math.vec3(rs_sca[r, 0], rs_sca[r, 5], rs_sca[r, 6])
        base_dist = rs_sca[r, 4]
        bounces_left = rs_int[r, 0]
        processed = rs_int[r, 1]
        prim = rs_eprim[r]
        ni = ti.math.vec3(rs_hit[r, 0], rs_hit[r, 1], rs_hit[r, 2])
        fn = ti.math.vec3(rs_hit[r, 3], rs_hit[r, 4], rs_hit[r, 5])
        albedo = ti.math.vec4(rs_hit[r, 6], rs_hit[r, 7], rs_hit[r, 8],
                              rs_hit[r, 9])
        alpha = rs_hit[r, 10]
        refl = rs_hit[r, 11]
        ior = rs_hit[r, 12]
        transmission = rs_hit[r, 15]
        t_hit = rs_hit[r, 13]
        hit_layer = rs_hit[r, 14]

        # Per-light shadow visibility, precomputed by wf_shadow_event.
        # Compiled out when shadows are off.
        vis = ti.Vector([1.0] * MAX_SHADOW_LIGHTS)
        if ti.static(shadows != 0):
            sbits = rs_vis[r]
            for li in range(num_lights):
                if li < MAX_SHADOW_LIGHTS:
                    if ((sbits >> li) & 1) != 0:
                        vis[li] = 0.0

        pos = ro + t_hit * rd
        rgb = ti.math.vec3(albedo[0], albedo[1], albedo[2])
        shaded = pipeline_fn(pos, -rd, ni, fn, rgb, albedo[3],
                             params, f, prim,
                             light_pos, light_col, num_lights, shadows, vis)

        (contrib, pass_w, refl_orig, refl_dir, refl_w,
         trans_orig, trans_dir, trans_w) = scatter_fn(
            rd, ni, fn, pos, shaded, rgb, alpha, refl, ior, transmission,
            params, f, prim, bounces_left, refraction)

        w_glow = ti.max(weight[0], ti.max(weight[1], weight[2]))
        acc += ti.math.vec4(weight[0], weight[1], weight[2],
                            w_glow) * contrib
        done = False
        if ti.static(refraction != 0):
            # Transmitted branch: append to the tile-wide shared pool.
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
                    rs_int[c, 2] = ST_TRAVERSE
                    rs_int[c, 3] = 0
                    rs_int[c, 4] = 0
                    rs_pix[c] = pix

        status = ST_PEEL
        refl_w_max = ti.max(refl_w[0], ti.max(refl_w[1], refl_w[2]))
        if (refl_w_max > 0.0) and (bounces_left > 0):
            # Bounce: continue in this slot along the scatter's reflected
            # branch; the stale K-buffer is abandoned (fresh traverse).
            for k in ti.static(range(3)):
                rs_ro[r, k] = refl_orig[k]
                rs_rd[r, k] = refl_dir[k]
            weight *= refl_w
            rs_sca[r, 1] = 0.0       # t_prev
            rs_sca[r, 2] = 1e30      # layer_prev
            rs_sca[r, 3] = -1e30     # seam_t
            rs_sca[r, 4] = base_dist + t_hit
            rs_int[r, 0] = bounces_left - 1
            rs_int[r, 4] = 0
            status = ST_TRAVERSE
        else:
            # Pass through to the next depth layer of the current K-buffer.
            weight *= pass_w
            rs_sca[r, 1] = t_hit     # t_prev
            rs_sca[r, 2] = hit_layer  # layer_prev
            if ti.max(weight[0], ti.max(weight[1], weight[2])) \
                    < MIN_WEIGHT:
                done = True
        if processed >= MAX_SURFACES_PER_RAY:
            done = True

        rs_sca[r, 0] = weight[0]
        rs_sca[r, 5] = weight[1]
        rs_sca[r, 6] = weight[2]
        for k in ti.static(range(4)):
            rs_acc[r, k] = acc[k]
        if done:
            # Terminated: commit this branch's premultiplied colour and its
            # leftover throughput into the shared per-pixel accumulator.
            for k in ti.static(range(4)):
                ti.atomic_add(pix_accum[pix, k], acc[k])
            for k in ti.static(range(3)):
                ti.atomic_add(pix_accum[pix, 4 + k], weight[k])
            rs_int[r, 2] = ST_DONE
        else:
            rs_int[r, 2] = status
