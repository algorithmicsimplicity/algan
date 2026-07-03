"""Minimal deferred-shading (G-buffer) prototype for the deterministic ray
tracer.

This is an experimental alternative to the in-kernel fragment shading of
``render_scene_stbvh`` (see :func:`algan.rendering.raytracing.shading_taichi._shade_fragment`).
Instead of evaluating the lighting model inside the trace megakernel, the kernel
here finds, for each primary ray, the *nearest* surface hit and writes its raw
surface attributes (position, view dir, interpolated + face normal, raw albedo,
primitive id / type) into a per-pixel **G-buffer**. A second pass then shades the
whole screen at once in PyTorch -- reusing the exact ``_shade_fragment`` lighting
math -- and composites the result into the output buffer.

It is a *prototype*: only the nearest opaque hit is shaded (no transparent-layer
compositing, no mirror bounces, no shadows), which is exact for the all-opaque
scenes it is meant to benchmark (e.g. ``benchmarks/neural_net_benchmark.py``).
The point is to measure whether the deferred (G-buffer + PyTorch) path is
competitive with the fused megakernel before building out full custom-shader
support. Gated behind ``ALGAN_GBUFFER=1`` in
:func:`algan.rendering.raytracing.primitives.render_batch_ray_traced`.

G-buffer layout (``gb_f32[n, 16]`` per pixel, ``gb_i32[n, 3]``)::

    gb_f32: 0:3 pos   3:6 view_dir(-rd)   6:9 n_interp   9:12 face_n   12:16 albedo(rgb+glow)
    gb_i32: 0 valid(1/0)   1 prim   2 hit_type (0 bezier, 1 triangle, 2 PN)
"""
import torch
import torch.nn.functional as F
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
    _comes_after,
    _flat_triangle_color,
    _generate_ray,
    _nearest_surface,
    _pn_normal,
    _safe_inverse,
    _sample_circuit_color,
    _triangle_color,
    _triangle_extra,
    _triangle_normal,
)

# Per-ray status codes (rs_int column 2), matching wavefront_taichi.
_ACTIVE = 0
_DONE = 1

# Per-recorded-hit G-buffer width (gb_f32[..., GB_HIT_W]); see wf_drain_record_gbuffer.
#   0     compositing coefficient  weight * alpha * (1 - reflectivity)
#   1:4   hit position
#   4:7   view direction (-rd)
#   7:10  interpolated shading normal
#   10:13 geometric face normal
#   13:17 raw albedo (rgb + glow)
GB_HIT_W = 17

# Mirrors shading_taichi.AMBIENT_STRENGTH / material_shaders.AMBIENT_STRENGTH.
AMBIENT_STRENGTH = 0.1


@ti.kernel
def gbuffer_nearest_general(
        # Triangle STBVH + packed geometry.
        t_nodes: ti.types.ndarray(), t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int,
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_colors: ti.types.ndarray(), tri_uvs: ti.types.ndarray(),
        tri_tex_meta: ti.types.ndarray(), textures: ti.types.ndarray(),
        num_colored_triangles: ti.i32,
        # PN patch STBVH + packed geometry.
        p_nodes: ti.types.ndarray(), p_node_miss: ti.types.ndarray(),
        p_leaf_prim: ti.types.ndarray(), p_leaf_tspan: ti.types.ndarray(),
        p_first_leaf: int,
        pn_ctrl: ti.types.ndarray(), pn_norm: ti.types.ndarray(),
        pn_colors: ti.types.ndarray(), pn_obb: ti.types.ndarray(),
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
        time_start: int, width: int, height: int,
        half_screen_w: float, half_screen_h: float,
        layer_offset_triangles: float, layer_offset_pn: float,
        # Output G-buffer.
        gb_f32: ti.types.ndarray(), gb_i32: ti.types.ndarray()):
    pixels_per_frame = width * height
    num_rays = gb_i32.shape[0]
    for ray_id in range(num_rays):
        f_rel = ray_id // pixels_per_frame
        p = ray_id - f_rel * pixels_per_frame
        f = time_start + f_rel
        ff = ti.cast(f, ti.f32)
        py = p // width
        px = p - py * width
        pixel_size_per_t = pixel_world_scale[f]
        ro, rd = _generate_ray(f, px, py, 0.5, 0.5,
                               half_screen_w, half_screen_h,
                               cam_origin, screen_point,
                               pixel_basis_x, pixel_basis_y)
        inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                              _safe_inverse(rd[2]))
        (found, t_hit, hit_layer, prim, htype, a, b, border,
         edge_hit) = _nearest_surface(
            ro, rd, inv_rd, f, ff, 0.0, 1e30,
            pixel_size_per_t, 0.0, layer_offset_triangles, layer_offset_pn,
            t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
            tri_pos,
            p_nodes, p_node_miss, p_leaf_prim, p_leaf_tspan, p_first_leaf,
            pn_ctrl, pn_obb,
            b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan, b_first_leaf,
            circuit_meta, edges_2d, edge_offsets)

        gb_i32[ray_id, 0] = found
        gb_i32[ray_id, 1] = prim
        gb_i32[ray_id, 2] = htype
        if found == 1:
            pos = ro + t_hit * rd
            view = -rd
            n = ti.math.vec3(0.0, 0.0, 0.0)
            face = ti.math.vec3(0.0, 0.0, 0.0)
            color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
            if htype == 1:
                w0 = 1.0 - a - b
                n = _triangle_normal(f, prim, w0, a, b, tri_norm, tri_pos)
                tp = f % tri_pos.shape[0]
                v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                                  tri_pos[tp, prim, 2])
                v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                                  tri_pos[tp, prim, 5])
                v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                                  tri_pos[tp, prim, 8])
                face = (v1 - v0).cross(v2 - v0)
                color, _alpha = _flat_triangle_color(
                    f, prim, w0, a, b, tri_colors, tri_uvs, tri_tex_meta,
                    textures, num_colored_triangles)
            elif htype == 2:
                n = _pn_normal(f, prim, a, b, pn_norm, pn_ctrl)
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
                face = su.cross(sv)
                w0 = 1.0 - a - b
                color, _alpha = _triangle_color(f, prim, w0, a, b, pn_colors)
            else:
                n = _bezier_normal(f, prim, circuit_meta)
                face = n
                color, _alpha = _sample_circuit_color(
                    prim, f, a, b, border, circuit_meta, circuit_colors,
                    circuit_border_colors)

            gb_f32[ray_id, 0] = pos[0]
            gb_f32[ray_id, 1] = pos[1]
            gb_f32[ray_id, 2] = pos[2]
            gb_f32[ray_id, 3] = view[0]
            gb_f32[ray_id, 4] = view[1]
            gb_f32[ray_id, 5] = view[2]
            gb_f32[ray_id, 6] = n[0]
            gb_f32[ray_id, 7] = n[1]
            gb_f32[ray_id, 8] = n[2]
            gb_f32[ray_id, 9] = face[0]
            gb_f32[ray_id, 10] = face[1]
            gb_f32[ray_id, 11] = face[2]
            gb_f32[ray_id, 12] = color[0]
            gb_f32[ray_id, 13] = color[1]
            gb_f32[ray_id, 14] = color[2]
            gb_f32[ray_id, 15] = color[3]


def _normalize(v):
    return F.normalize(v, p=2, dim=-1, eps=1e-12)


def shade_gbuffer_torch(gb_f32, mat, mat_id, f_idx, light_pos, light_col,
                        num_lights):
    """Vectorized PyTorch port of ``shading_taichi._shade_fragment``.

    Shades every pixel of the G-buffer at once. ``mat`` is the per-pixel
    canonical 12-slot material block, ``mat_id`` the per-pixel core-material id
    (0 default, 1 unlit, 2 lambert, 3 phong, 4 standard), ``f_idx`` the per-pixel
    frame index (for time-varying lights). ``light_pos``/``light_col`` are the
    ``[T, L, 3]`` packed lights. Returns ``[n, 4]`` RGB + glow in ``[0, 1]``.
    """
    pos = gb_f32[:, 0:3]
    view = _normalize(gb_f32[:, 3:6])
    n_interp = gb_f32[:, 6:9]
    face_n = gb_f32[:, 9:12]
    albedo = gb_f32[:, 12:15]
    glow = gb_f32[:, 15:16]

    emissive = mat[:, 0:3]
    emissive_intensity = mat[:, 3:4]
    specular = mat[:, 4:7]
    shininess = mat[:, 7:8]
    roughness = mat[:, 8:9]
    metalness = mat[:, 9:10]
    flat = mat[:, 10:11]
    env = mat[:, 11:12]

    # Shading normal, optionally blended toward the (sign-aligned) face normal.
    n = _normalize(n_interp)
    fn = _normalize(face_n)
    sign = torch.where((fn * n).sum(-1, keepdim=True) < 0.0, -1.0, 1.0)
    fn = fn * sign
    n_flat = _normalize(torch.lerp(n, fn, flat))
    n = torch.where(flat > 1e-4, n_flat, n)
    # Two-sided: light the face the viewer actually sees.
    n = torch.where((n * view).sum(-1, keepdim=True) < 0.0, -n, n)

    mid = mat_id.view(-1, 1)
    out = albedo.clone()
    one = torch.ones_like(albedo)
    for li in range(num_lights):
        tl = f_idx % light_pos.shape[0]
        lp = light_pos[tl, li]          # [n, 3]
        lc = light_col[tl, li]          # [n, 3]

        # default (0): diffuse lerp toward the light colour.
        inc = _normalize(pos - lp)
        d = (-(inc * n).sum(-1, keepdim=True)).clamp_min(0.0)
        diffuse_def = d.pow(5) * 0.5
        out_def = out * (1.0 - diffuse_def) + lc * diffuse_def

        ld = _normalize(lp - pos)
        n_dot_l = (n * ld).sum(-1, keepdim=True).clamp_min(0.0)
        ambient = out * (AMBIENT_STRENGTH * env)

        # lambert (2)
        out_lam = ambient + out * lc * n_dot_l + emissive * emissive_intensity

        # phong (3): Blinn-Phong diffuse + specular.
        half = _normalize(ld + view)
        n_dot_h = (n * half).sum(-1, keepdim=True).clamp_min(0.0)
        diffuse = out * lc * n_dot_l
        spec_term = n_dot_h.clamp_min(1e-4).pow(shininess.clamp_min(1e-3))
        gate = (n_dot_l > 0.0).to(out.dtype)
        out_phong = (ambient + (diffuse + specular * lc * spec_term * gate)
                     + emissive * emissive_intensity)

        # standard (4): Cook-Torrance GGX PBR.
        rgb = out
        n_dot_v = (n * view).sum(-1, keepdim=True).clamp_min(1e-4)
        v_dot_h = (view * half).sum(-1, keepdim=True).clamp_min(0.0)
        f0 = 0.04 * (1.0 - metalness) + rgb * metalness
        fresnel = f0 + (one - f0) * (1.0 - v_dot_h).clamp_min(0.0).pow(5)
        a_r = (roughness * roughness).clamp_min(1e-4)
        a2 = a_r * a_r
        denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0
        ndf = a2 / (3.14159265 * denom * denom).clamp_min(1e-7)
        r = roughness + 1.0
        k = (r * r) / 8.0
        gv = n_dot_v / (n_dot_v * (1.0 - k) + k).clamp_min(1e-6)
        gl = n_dot_l / (n_dot_l * (1.0 - k) + k).clamp_min(1e-6)
        geom = gv * gl
        spec = (ndf * geom) * fresnel / (4.0 * n_dot_v * n_dot_l).clamp_min(1e-4)
        k_d = (one - fresnel) * (1.0 - metalness)
        diffuse_std = k_d * rgb * lc * n_dot_l
        direct = diffuse_std + spec * lc * n_dot_l
        ambient_std = (rgb * (1.0 - metalness) + f0 * metalness) * (
            AMBIENT_STRENGTH * env)
        out_std = ambient_std + direct + emissive * emissive_intensity

        out = torch.where(mid == 0, out_def,
              torch.where(mid == 2, out_lam,
              torch.where(mid == 3, out_phong,
              torch.where(mid == 4, out_std, out))))  # mid==1: unchanged

    out = out.clamp(0.0, 1.0)
    return torch.cat((out, glow), -1)


# ---------------------------------------------------------------------------
# Deferred wavefront (transparency + reflections).
#
# Same stage-split ping-pong as wavefront_taichi (gen -> (traverse -> drain ->
# shade -> compact)* -> composite), but the shade stage is moved out of the
# kernel into PyTorch: wf_drain_record_gbuffer replays the megakernel's exact
# front-to-back drain / mirror-bounce / transmittance control flow, and instead
# of evaluating the lighting model in-kernel it records, per drained hit, the
# compositing coefficient (weight*alpha*(1-reflectivity), all material-sampled
# so computable here) plus the raw shading inputs into a per-ray G-buffer.
# shade_accumulate_wavefront then material-shades those hits in PyTorch and
# scatter-adds coef * shaded into rs_acc. The ray-geometry state machine (bounce
# rays, transmittance weight, termination) is unchanged and stays in Taichi.
# ---------------------------------------------------------------------------


@ti.kernel
def wf_traverse_gbuffer(
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
        time_start: int, width: int, height: int,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_sca: ti.types.ndarray(), rs_int: ti.types.ndarray(),
        rs_kt: ti.types.ndarray(), rs_kl: ti.types.ndarray(),
        rs_ka: ti.types.ndarray(), rs_kb: ti.types.ndarray(),
        rs_kp: ti.types.ndarray(), rs_kf: ti.types.ndarray()):
    """Gather the KBUF nearest hits for each active ray (general _collect_hits,
    OBB-culled PN solver included). Identical to wavefront_taichi.wf_traverse_general
    but threads ``pn_obb`` through (the legacy kernel predates the OBB cull)."""
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
            pn_ctrl, pn_obb,
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
def wf_drain_record_gbuffer(
        active: ti.types.ndarray(), num_active: int,
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_extra: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        pn_ctrl: ti.types.ndarray(), pn_norm: ti.types.ndarray(),
        pn_extra: ti.types.ndarray(), pn_colors: ti.types.ndarray(),
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_sca: ti.types.ndarray(), rs_int: ti.types.ndarray(),
        rs_kt: ti.types.ndarray(), rs_kl: ti.types.ndarray(),
        rs_ka: ti.types.ndarray(), rs_kb: ti.types.ndarray(),
        rs_kp: ti.types.ndarray(), rs_kf: ti.types.ndarray(),
        gb_f32: ti.types.ndarray(), gb_i32: ti.types.ndarray(),
        gb_count: ti.types.ndarray()):
    """Drain the gathered hits front-to-back exactly as render_scene_stbvh's
    inner loop (seam-merge, transmittance, mirror bounce), but *record* each
    drained hit's compositing coefficient and shading inputs into the G-buffer
    instead of shading in-kernel. ``rs_acc`` is left untouched -- PyTorch adds
    each iteration's coef*shaded contribution. The ray-geometry state machine
    (ro/rd/weight/bounce/status) is byte-identical to wf_shade_general."""
    pixels_per_frame = width * height
    for i in range(num_active):
        r = active[i]
        num_hits = rs_int[r, 3]
        slot = 0
        if num_hits > 0:
            f = time_start + r // pixels_per_frame
            ro = ti.math.vec3(rs_ro[r, 0], rs_ro[r, 1], rs_ro[r, 2])
            rd = ti.math.vec3(rs_rd[r, 0], rs_rd[r, 1], rs_rd[r, 2])
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
                n_interp = ti.math.vec3(0.0, 0.0, 0.0)
                face_n = ti.math.vec3(0.0, 0.0, 0.0)
                if htype == 1:
                    w0 = 1.0 - a - b
                    color, alpha = _flat_triangle_color(
                        f, prim, w0, a, b, tri_colors, tri_uvs, tri_tex_meta,
                        textures, num_colored_triangles)
                    reflectivity, _rough = _triangle_extra(f, prim, w0, a, b,
                                                           tri_extra)
                    n_interp = _triangle_normal(f, prim, w0, a, b, tri_norm,
                                                tri_pos)
                    tp = f % tri_pos.shape[0]
                    v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                                      tri_pos[tp, prim, 2])
                    v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                                      tri_pos[tp, prim, 5])
                    v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                                      tri_pos[tp, prim, 8])
                    face_n = (v1 - v0).cross(v2 - v0)
                elif htype == 2:
                    w0 = 1.0 - a - b
                    color, alpha = _triangle_color(f, prim, w0, a, b, pn_colors)
                    reflectivity, _rough = _triangle_extra(f, prim, w0, a, b,
                                                           pn_extra)
                    n_interp = _pn_normal(f, prim, a, b, pn_norm, pn_ctrl)
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
                    face_n = su.cross(sv)
                else:
                    color, alpha = _sample_circuit_color(
                        prim, f, a, b, border, circuit_meta, circuit_colors,
                        circuit_border_colors)
                    n_interp = _bezier_normal(f, prim, circuit_meta)
                    face_n = n_interp

                alpha = ti.math.clamp(alpha, 0.0, 1.0)
                reflectivity = ti.math.clamp(reflectivity, 0.0, 1.0)
                if bounces_left <= 0:
                    reflectivity = 0.0

                coef = weight * alpha * (1.0 - reflectivity)
                pos = ro + t_hit * rd
                view = -rd
                gb_f32[r, slot, 0] = coef
                gb_f32[r, slot, 1] = pos[0]
                gb_f32[r, slot, 2] = pos[1]
                gb_f32[r, slot, 3] = pos[2]
                gb_f32[r, slot, 4] = view[0]
                gb_f32[r, slot, 5] = view[1]
                gb_f32[r, slot, 6] = view[2]
                gb_f32[r, slot, 7] = n_interp[0]
                gb_f32[r, slot, 8] = n_interp[1]
                gb_f32[r, slot, 9] = n_interp[2]
                gb_f32[r, slot, 10] = face_n[0]
                gb_f32[r, slot, 11] = face_n[1]
                gb_f32[r, slot, 12] = face_n[2]
                gb_f32[r, slot, 13] = color[0]
                gb_f32[r, slot, 14] = color[1]
                gb_f32[r, slot, 15] = color[2]
                gb_f32[r, slot, 16] = color[3]
                gb_i32[r, slot, 0] = htype
                gb_i32[r, slot, 1] = prim
                slot += 1

                if (reflectivity > MIN_ALPHA) and (alpha > MIN_ALPHA):
                    normal = n_interp.normalized()
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
            rs_sca[r, 0] = weight
            rs_sca[r, 1] = t_prev
            rs_sca[r, 2] = layer_prev
            rs_sca[r, 3] = seam_t
            rs_sca[r, 4] = base_dist
            rs_int[r, 0] = bounces_left
            rs_int[r, 1] = processed
            rs_int[r, 2] = _DONE if done else _ACTIVE
        gb_count[r] = slot


def _gather_material(htype, prim, f_idx, merged, device):
    """Per-row (mat_id, 12-slot block) for recorded hits, gathered by hit type.
    Triangle/PN hits read their material arrays; bezier hits keep the unlit
    default (id 1), as the megakernel leaves them."""
    m = htype.shape[0]
    mat = torch.tensor(_MAT_DEFAULTS_GB, device=device,
                       dtype=torch.float32).view(1, 12).repeat(m, 1)
    mat_id = torch.ones(m, dtype=torch.long, device=device)
    for type_id, id_key, mat_key in ((1, "tri_mat_id", "tri_mat"),
                                     (2, "pn_mat_id", "pn_mat")):
        mask = htype == type_id
        if not bool(mask.any()):
            continue
        mat_arr = merged[mat_key]
        id_arr = merged[id_key]
        pm = prim[mask]
        fm = f_idx[mask]
        mat[mask] = mat_arr[fm % mat_arr.shape[0], pm].float()
        mat_id[mask] = id_arr[fm % id_arr.shape[0], pm].long()
    return mat, mat_id


# Canonical 12-slot material defaults (mirror primitives._MAT_DEFAULTS); kept
# here to avoid a circular import.
_MAT_DEFAULTS_GB = [0.0, 0.0, 0.0, 1.0, 0.0666, 0.0666, 0.0666, 30.0, 1.0, 0.0,
                    0.0, 1.0]


def shade_accumulate_wavefront(active, gb_f32, gb_i32, gb_count, rs_acc,
                               merged, light_pos, light_col, num_lights,
                               pixels_per_frame, time_start, block):
    """Shade this iteration's recorded hits in PyTorch and scatter-add
    ``coef * shaded`` into ``rs_acc`` per ray. Processes the active rays in
    blocks so temporaries stay bounded."""
    device = rs_acc.device
    na = active.shape[0]
    for off in range(0, na, block):
        rb = active[off:min(off + block, na)].long()   # [B] global ray ids
        cnt = gb_count[rb]                              # [B]
        maxc = int(cnt.max().item()) if rb.numel() else 0
        if maxc == 0:
            continue
        B = rb.shape[0]
        gbf = gb_f32[rb, :maxc]                          # [B, maxc, GB_HIT_W]
        gbi = gb_i32[rb, :maxc]                          # [B, maxc, 2]
        slot_idx = torch.arange(maxc, device=device).view(1, maxc)
        valid = slot_idx < cnt.view(B, 1)               # [B, maxc]
        flat = valid.reshape(-1)
        f32 = gbf.reshape(-1, GB_HIT_W)[flat]           # [M, GB_HIT_W]
        i32 = gbi.reshape(-1, 2)[flat]                  # [M, 2]
        if f32.shape[0] == 0:
            continue
        coef = f32[:, 0:1]
        htype = i32[:, 0]
        prim = i32[:, 1].long()

        local_ray = (torch.arange(B, device=device).view(B, 1)
                     .expand(B, maxc).reshape(-1)[flat])   # [M] in [0, B)
        ray_global = rb[local_ray]
        f_idx = ray_global // pixels_per_frame + int(time_start)

        mat, mat_id = _gather_material(htype, prim, f_idx, merged, device)
        shaded = shade_gbuffer_torch(f32[:, 1:GB_HIT_W], mat, mat_id, f_idx,
                                     light_pos, light_col, int(num_lights))
        contrib = coef * shaded                          # [M, 4]

        per_ray = torch.zeros(B, 4, device=device, dtype=rs_acc.dtype)
        per_ray.index_add_(0, local_ray, contrib)
        rs_acc[rb] += per_ray
