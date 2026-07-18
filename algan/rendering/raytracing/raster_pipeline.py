"""Host orchestration of the hybrid raster front-end (see raster_taichi.py).

``raster_iteration_zero`` replaces the deterministic wavefront's first
iteration for one screen tile: per-frame torch binning of triangle candidates
into fixed-size (prim, chunk) pairs, the opaque z-prepass, exact-counted
transparent fragment emission, a torch (cub radix) sort by
``(local_pixel << 32 | depth_bits)``, per-pixel run tables, and the
``raster_first_shade`` resolve. All transient buffers are ordinary torch
CUDA tensors (not arena-backed): an allocation failure raises
``torch.OutOfMemoryError``, which the render loop's existing halve-and-retry
already handles.
"""
from __future__ import annotations

import torch

from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.raster_taichi import (
    RASTER_CHUNK,
    Z_SENTINEL,
    raster_bez_count,
    raster_bez_write,
    raster_first_shade,
    raster_shadow,
    raster_tri_count,
    raster_tri_write,
    raster_tri_z,
)


def _project_verts(verts, ro, sp, pbx, pby, half_w, half_h):
    """Continuous pixel coordinates of world points for one frame.

    Inverts the kernel's ``_generate_ray`` mapping using only the tensors the
    tracer already has: the screen plane is spanned by the reciprocal basis
    vectors ``pbx, pby`` at ``sp`` (`pix = sp + u*pbx + v*pby`), so a world
    point projects along its camera ray onto that plane and decomposes onto
    (pbx, pby) via cross products. Returns ``(px, py, front)`` with ``front``
    False for points behind (or parallel to) the screen plane.
    """
    nvec = torch.linalg.cross(pbx, pby)
    d = verts - ro
    wpn = (d * nvec).sum(-1)
    s = float(torch.dot(sp - ro, nvec))
    safe = torch.where(wpn.abs() < 1e-12, torch.full_like(wpn, 1e-12), wpn)
    td = s / safe
    front = (wpn.abs() >= 1e-12) & (td > 0)
    hit = ro + td.unsqueeze(-1) * d
    rel = hit - sp
    dsq = float(nvec.dot(nvec))
    u = (torch.linalg.cross(rel, pby.expand_as(rel)) * nvec).sum(-1) / dsq
    v = (torch.linalg.cross(pbx.expand_as(rel), rel) * nvec).sum(-1) / dsq
    return u * half_h + half_w, v * half_h + half_h, front


def _class_pairs(mask, x0, x1, y0, y1, f, device):
    """Expand one class's clipped bboxes into [P, 8] int32 pair rows
    ``(prim, f, x0, y0, bw, bh, off, 0)`` of <= RASTER_CHUNK candidates.
    """
    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return None
    bx0 = x0[idx]
    by0 = y0[idx]
    bw = x1[idx] - bx0 + 1
    bh = y1[idx] - by0 + 1
    area = bw * bh
    nch = (area + (RASTER_CHUNK - 1)) // RASTER_CHUNK
    total = int(nch.sum())
    rep = torch.repeat_interleave(
        torch.arange(idx.numel(), device=device), nch)
    base = torch.cumsum(nch, 0) - nch
    off = (torch.arange(total, device=device) - base[rep]) * RASTER_CHUNK
    rows = torch.stack([
        idx[rep], torch.full_like(rep, f), bx0[rep], by0[rep],
        bw[rep], bh[rep], off, torch.zeros_like(rep),
    ], -1)
    return rows.to(torch.int32).contiguous()


def _screen_bbox(px, py, front, width, row_lo, row_hi):
    """Clipped integer screen bbox (x0, x1, y0, y1) + an on-screen ``reach``
    mask from projected corner pixel coords ``px, py`` [M, K] and per-corner
    ``front`` [M, K]. Primitives with any corner behind the camera plane fall
    back to the full clipped tile window (row range).
    """
    all_front = front.all(-1)
    xmin = px.amin(-1)
    xmax = px.amax(-1)
    ymin = py.amin(-1)
    ymax = py.amax(-1)
    # +-1 pixel of padding absorbs the intersection epsilons.
    fx0 = (xmin - 1.0).floor().clamp_(0, width - 1).long()
    fx1 = (xmax + 1.0).ceil().clamp_(0, width - 1).long()
    fy0 = (ymin - 1.0).floor().clamp_(row_lo, row_hi).long()
    fy1 = (ymax + 1.0).ceil().clamp_(row_lo, row_hi).long()
    x0 = torch.where(all_front, fx0, torch.zeros_like(fx0))
    x1 = torch.where(all_front, fx1, torch.full_like(fx1, width - 1))
    y0 = torch.where(all_front, fy0, torch.full_like(fy0, row_lo))
    y1 = torch.where(all_front, fy1, torch.full_like(fy1, row_hi))
    # Drop prims whose (unclamped) bbox misses the screen / row window
    # entirely -- clamping alone would leave a spurious one-row band.
    on_screen = ((xmax >= -1.0) & (xmin <= width + 1.0)
                 & (ymax >= row_lo - 1.0) & (ymin <= row_hi + 1.0))
    reach = torch.where(all_front, on_screen, torch.ones_like(on_screen))
    return x0, x1, y0, y1, reach


# The 8 corners of a unit AABB as a bool selector (hi where True, lo where
# False), used to project a primitive's world bounding box to screen space.
_AABB_SEL = None


def _aabb_corners(lo, hi):
    """[M, 8, 3] world corners of the per-primitive AABBs (lo, hi)."""
    global _AABB_SEL
    if _AABB_SEL is None or _AABB_SEL.device != lo.device:
        _AABB_SEL = torch.tensor(
            [[cx, cy, cz] for cx in (0, 1) for cy in (0, 1) for cz in (0, 1)],
            dtype=torch.bool, device=lo.device)
    return torch.where(_AABB_SEL.unsqueeze(0), hi.unsqueeze(1), lo.unsqueeze(1))


def _frame_pairs(merged, f, width, height, row_lo, row_hi,
                 cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
                 half_w, half_h, device):
    """(opaque_pairs, transparent_pairs) for frame ``f``'s triangles over rows
    [row_lo, row_hi] of the current tile (either may be None).
    """
    tri_pos = merged["tri_pos"]
    verts = tri_pos[f % tri_pos.shape[0]].view(-1, 3, 3)
    valid_all = merged["tri_frame_valid"]
    opq_all = merged["tri_frame_opaque"]
    valid = valid_all[f % valid_all.shape[0]].bool()
    if bool(merged.get("has_uncertain_texture_alpha")):
        # A texture with unproven alpha may make any "opaque" prim
        # translucent; route everything through the sorted transparent path
        # (correct, just no z culling) -- mirrors the opaque-prepass gate.
        opq = torch.zeros_like(valid)
    else:
        opq = valid & opq_all[f % opq_all.shape[0]].bool()

    px, py, front = _project_verts(
        verts, cam_origin[f], screen_point[f],
        pixel_basis_x[f], pixel_basis_y[f], half_w, half_h)
    x0, x1, y0, y1, reach = _screen_bbox(px, py, front, width, row_lo, row_hi)
    pairs_o = _class_pairs(valid & opq & reach, x0, x1, y0, y1, f, device)
    pairs_t = _class_pairs(valid & ~opq & reach, x0, x1, y0, y1, f, device)
    return pairs_o, pairs_t


def _frame_bez_pairs(merged, f, width, height, row_lo, row_hi,
                     cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
                     half_w, half_h, device):
    """Bezier candidate pairs for frame ``f``'s tile rows (or None). Circuits
    project their per-frame world AABB (8 corners) to a conservative screen
    bbox and route entirely through the sorted transparent path.
    """
    valid_all = merged["bez_frame_valid"]
    valid = valid_all[f % valid_all.shape[0]].bool()
    if not bool(valid.any()):
        return None
    lo_all = merged["bez_frame_lo"]
    hi_all = merged["bez_frame_hi"]
    lo = lo_all[f % lo_all.shape[0]]
    hi = hi_all[f % hi_all.shape[0]]
    corners = _aabb_corners(lo, hi)                       # [M, 8, 3]
    px, py, front = _project_verts(
        corners, cam_origin[f], screen_point[f],
        pixel_basis_x[f], pixel_basis_y[f], half_w, half_h)
    x0, x1, y0, y1, reach = _screen_bbox(px, py, front, width, row_lo, row_hi)
    return _class_pairs(valid & reach, x0, x1, y0, y1, f, device)


def raster_iteration_zero(
        merged, cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
        pixel_world_scale, layer_offsets, gen_meta, light_pos, light_col,
        num_lights, col_row_arr, frag_flag, frag_pipelines, skip_unlit_normal,
        refraction_flag, time_start, width, height, half_w, half_h,
        tile_start, tn_primary, state, rs_pix, pix_accum, rs_alloc,
        shadow_flag, t_bvh, pn_bvh, bez_bvh,
        layer_offset_triangles, layer_offset_pn):
    """Raster, sort and resolve-shade one wavefront tile's primary rays.

    Triangles use the opaque z-prepass + sorted-transparent path; bezier
    circuits route entirely through the sorted-transparent path (tagged by a
    negative packed id, emitted at lower indices so a coplanar circuit layers
    over a triangle). Postcondition matches the classic first traverse+shade:
    ``pix_accum`` holds every retired pixel's colour + leftover weight, bounced
    pixels' ray slots are ACTIVE with their continuation state, and split
    branches occupy shared-pool slots via ``rs_alloc`` (overflow flag intact).
    """
    (rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
     _rs_kt, _rs_kl, _rs_ka, _rs_kb, _rs_kp, _rs_kf) = state
    device = pix_accum.device
    ppf = width * height
    g0 = tile_start
    g1 = tile_start + tn_primary
    has_tri = 1 if int(merged.get("num_triangles", 0)) > 0 else 0
    has_pn = 1 if int(merged.get("num_pn", 0)) > 0 else 0
    has_bez = 1 if int(merged.get("num_circuits", 0)) > 0 else 0

    pairs_o, pairs_t, pairs_b = [], [], []
    for f_rel in range(g0 // ppf, (g1 - 1) // ppf + 1):
        f = time_start + f_rel
        lo_p = max(g0 - f_rel * ppf, 0)
        hi_p = min(g1 - f_rel * ppf, ppf)
        row_lo = lo_p // width
        row_hi = (hi_p - 1) // width
        po, pt = _frame_pairs(
            merged, f, width, height, row_lo, row_hi,
            cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
            half_w, half_h, device)
        if po is not None:
            pairs_o.append(po)
        if pt is not None:
            pairs_t.append(pt)
        if has_bez:
            pb = _frame_bez_pairs(
                merged, f, width, height, row_lo, row_hi,
                cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
                half_w, half_h, device)
            if pb is not None:
                pairs_b.append(pb)

    zbuf = torch.full((tn_primary,), Z_SENTINEL, dtype=torch.int64,
                      device=device)
    ss = 1 if rt_settings.RASTER_SS else 0
    tri_pos = merged["tri_pos"]
    cam_args = (cam_origin, screen_point, pixel_basis_x, pixel_basis_y)
    geo_args = (int(time_start), int(width), int(height),
                float(half_w), float(half_h), int(tile_start),
                int(tn_primary))
    # Bezier shading/geometry arrays (also the count/write inputs).
    bez_geom = (pixel_world_scale, merged["circuit_meta"],
                merged["circuit_colors"], merged["edges_2d"],
                merged["edge_accel"])
    if pairs_o:
        po = torch.cat(pairs_o, 0) if len(pairs_o) > 1 else pairs_o[0]
        raster_tri_z(po, int(po.shape[0]), tri_pos, *cam_args, *geo_args,
                     ss, zbuf)

    # Count surviving transparent fragments per pair for both geometry types.
    # Bezier is laid out first (lower indices) so a coplanar circuit sorts
    # ahead of a triangle at equal depth (circuits-over-triangles layering).
    pt = torch.cat(pairs_t, 0) if len(pairs_t) > 1 else (
        pairs_t[0] if pairs_t else None)
    pb = torch.cat(pairs_b, 0) if len(pairs_b) > 1 else (
        pairs_b[0] if pairs_b else None)
    n_bez = 0
    bez_offsets = None
    if pb is not None:
        bcounts = torch.zeros((pb.shape[0],), dtype=torch.int32,
                              device=device)
        raster_bez_count(pb, int(pb.shape[0]), *cam_args, *bez_geom,
                         *geo_args, zbuf, bcounts)
        bcounts64 = bcounts.long()
        bez_offsets = (torch.cumsum(bcounts64, 0) - bcounts64).to(torch.int32)
        n_bez = int(bcounts64.sum())
    n_tri = 0
    tri_offsets = None
    if pt is not None:
        tcounts = torch.zeros((pt.shape[0],), dtype=torch.int32,
                              device=device)
        raster_tri_count(pt, int(pt.shape[0]), tri_pos, *cam_args, *geo_args,
                         ss, zbuf, tcounts)
        tcounts64 = tcounts.long()
        tri_offsets = (n_bez + torch.cumsum(tcounts64, 0)
                       - tcounts64).to(torch.int32)
        n_tri = int(tcounts64.sum())
    num_frags = n_bez + n_tri

    run_start = torch.zeros((tn_primary,), dtype=torch.int32, device=device)
    run_len = torch.zeros((tn_primary,), dtype=torch.int32, device=device)
    if num_frags > 0:
        frag_key = torch.empty((num_frags,), dtype=torch.int64, device=device)
        frag_t = torch.empty((num_frags,), dtype=torch.float32, device=device)
        frag_prim = torch.empty((num_frags,), dtype=torch.int32,
                                device=device)
        frag_ab = torch.empty((num_frags, 2), dtype=torch.float32,
                              device=device)
        frag_flags = torch.empty((num_frags,), dtype=torch.int32,
                                 device=device)
        if n_bez > 0:
            raster_bez_write(pb, int(pb.shape[0]), bez_offsets, *cam_args,
                             *bez_geom, *geo_args, zbuf, frag_key, frag_t,
                             frag_prim, frag_ab, frag_flags)
        if n_tri > 0:
            raster_tri_write(pt, int(pt.shape[0]), tri_offsets, tri_pos,
                             *cam_args, *geo_args, ss, zbuf, frag_key, frag_t,
                             frag_prim, frag_ab, frag_flags)
        # Stable: exactly-coincident fragments keep emission order (bezier
        # first, then per-primitive) -- a deterministic coplanar layering.
        order = torch.argsort(frag_key, stable=True)
        frag_key = frag_key[order]
        frag_t = frag_t[order].contiguous()
        frag_prim = frag_prim[order].contiguous()
        frag_ab = frag_ab[order].contiguous()
        frag_flags = frag_flags[order].contiguous()
        pix = (frag_key >> 32)
        change = torch.ones((num_frags,), dtype=torch.bool, device=device)
        change[1:] = pix[1:] != pix[:-1]
        starts = change.nonzero(as_tuple=True)[0]
        lens = torch.diff(torch.cat(
            [starts, torch.tensor([num_frags], device=device)]))
        run_pix = pix[starts]
        run_start[run_pix] = starts.to(torch.int32)
        run_len[run_pix] = lens.to(torch.int32)
    else:
        # Non-empty placeholders keep the ndarray args valid.
        frag_t = torch.zeros((1,), dtype=torch.float32, device=device)
        frag_prim = torch.zeros((1,), dtype=torch.int32, device=device)
        frag_ab = torch.zeros((1, 2), dtype=torch.float32, device=device)
        frag_flags = torch.zeros((1,), dtype=torch.int32, device=device)

    # Deferred hard-shadow pre-pass: pack per-fragment occlusion bits into
    # rs_vis before the resolve reads them (see raster_shadow). Runs even when
    # there are no transparent fragments -- the opaque z-hits still receive
    # shadows. A 1-element placeholder keeps the resolve's ndarray arg valid
    # when shadows are off.
    import os as _os
    if shadow_flag:
        rs_vis = torch.zeros((tn_primary,), dtype=torch.int32, device=device)
        raster_shadow(
            int(tn_primary), run_start, run_len, frag_t, frag_prim, frag_ab,
            zbuf,
            t_bvh.blocks, t_bvh.node_miss, t_bvh.leaf_prim, t_bvh.leaf_tspan,
            int(t_bvh.first_leaf),
            merged["tri_pos"], merged["tri_norm"], merged["tri_colors"],
            merged["tri_uvs"], merged["tri_tex_meta"], merged["textures"],
            int(merged["num_colored_triangles"]),
            pn_bvh.blocks, pn_bvh.node_miss, pn_bvh.leaf_prim,
            pn_bvh.leaf_tspan, int(pn_bvh.first_leaf),
            merged["pn_ctrl"], merged["pn_obb"], merged["pn_colors"],
            bez_bvh.blocks, bez_bvh.node_miss, bez_bvh.leaf_prim,
            bez_bvh.leaf_tspan, int(bez_bvh.first_leaf),
            merged["circuit_meta"], merged["circuit_colors"],
            merged["circuit_border_colors"], merged["edges_2d"],
            merged["edge_accel"],
            light_pos, light_col, int(num_lights),
            pixel_world_scale,
            float(layer_offset_triangles), float(layer_offset_pn),
            has_tri, has_pn, has_bez, ss,
            int(time_start), int(width), int(height), int(tile_start),
            cam_origin, screen_point, pixel_basis_x, pixel_basis_y, gen_meta,
            rs_vis)
    else:
        rs_vis = torch.zeros((1,), dtype=torch.int32, device=device)

    raster_first_shade(
        int(tn_primary), run_start, run_len, frag_t, frag_prim, frag_ab,
        frag_flags,
        zbuf,
        merged["tri_pos"], merged["tri_norm"], merged["tri_extra"],
        merged["tri_colors"], merged["tri_uvs"], merged["tri_tex_meta"],
        merged["textures"], int(merged["num_colored_triangles"]),
        col_row_arr,
        merged["tri_mat_id"], merged["tri_mat"],
        merged["circuit_meta"], merged["circuit_colors"],
        merged["circuit_border_colors"],
        light_pos, light_col, int(num_lights),
        layer_offsets,
        int(frag_flag), frag_pipelines,
        int(refraction_flag), int(skip_unlit_normal), ss, has_bez,
        int(shadow_flag),
        int(time_start), int(width), int(height), int(tile_start),
        cam_origin, screen_point, pixel_basis_x, pixel_basis_y, gen_meta,
        rs_ro, rs_rd, rs_acc, rs_sca, rs_int, rs_pix, pix_accum, rs_alloc,
        rs_vis)
