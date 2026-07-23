"""Host orchestration for the deterministic hybrid raster front-end.

The frontend operates on the current wavefront *linear ray tile* (normally one
or more row bands), not on fixed square GPU tiles.  Each primitive is split into
``RASTER_CHUNK``-sized candidate chunks, exact hits are emitted, and the
surviving fragment records are ordered by the classic deterministic
``(depth-bin, descending layer)`` relation.  Future work should benchmark a
true square screen-tile/bin architecture for better projection reuse and cache
locality.

Large transient arrays are allocated from ``ManualMemory`` so failed raster
attempts can restore the arena pointer and retry a smaller primary tile.  Torch
sort/index scratch remains allocator-owned because PyTorch's radix sort cannot
write directly into an arena view.
"""
from __future__ import annotations

import torch

from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    DEPTH_TIE_EPSILON,
)
from algan.rendering.raytracing.raster_taichi import (
    RASTER_CHUNK,
    Z_SENTINEL,
    raster_bez_count,
    raster_bez_write,
    raster_bez_z,
    raster_first_shade,
    raster_shadow_event_build,
    raster_shadow_trace,
    raster_tri_count,
    raster_tri_write,
    raster_tri_z,
)


def precompute_triangle_projection(
        merged, cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
        half_w, half_h, memory):
    """Prepare one compact projection record per frame and flat triangle.

    Columns 0:3 are continuous screen x, 3:6 screen y, 6:9 reciprocal
    perspective divisors, and 9 is one when all vertices are safely in front of
    the camera plane.  Invalid/straddling triangles retain zeros and use the
    exact ray-cast fallback in the kernels.  The result is camera-specific and
    is built once per render batch, rather than once per primitive chunk and
    raster phase.
    """
    tri_pos = merged["tri_pos"]
    # Every materialized input may be independently deduplicated to T=1.  The
    # projection table must span the longest dynamic input, then index every
    # source modulo its own time dimension; using only the camera length would
    # freeze moving geometry whenever the camera itself is static.
    frames = max(
        int(tri_pos.shape[0]), int(cam_origin.shape[0]),
        int(screen_point.shape[0]), int(pixel_basis_x.shape[0]),
        int(pixel_basis_y.shape[0]),
        int(merged["tri_frame_valid"].shape[0]),
    )
    ntri = int(merged.get("num_triangles", 0))
    out = memory.get_tensor((max(1, frames), max(1, ntri), 10), torch.float32)
    out.zero_()
    if ntri == 0:
        return out

    frame_ids = torch.arange(frames, device=tri_pos.device)
    verts = tri_pos.index_select(
        0, frame_ids % tri_pos.shape[0]).view(frames, ntri, 3, 3)
    ro = cam_origin.index_select(0, frame_ids % cam_origin.shape[0])
    sp = screen_point.index_select(0, frame_ids % screen_point.shape[0])
    pbx = pixel_basis_x.index_select(0, frame_ids % pixel_basis_x.shape[0])
    pby = pixel_basis_y.index_select(0, frame_ids % pixel_basis_y.shape[0])
    nvec = torch.linalg.cross(pbx, pby)
    n2 = (nvec * nvec).sum(-1)
    big_d = ((sp - ro) * nvec).sum(-1)
    d = verts - ro[:, None, None, :]
    denom = (d * nvec[:, None, None, :]).sum(-1)
    sign = torch.where(big_d >= 0, torch.ones_like(big_d), -torch.ones_like(big_d))
    cam_ok = ((n2 > 1e-30) & (big_d.abs() > 1e-20))[:, None]
    vert_front = denom * sign[:, None, None] > 1e-9
    valid = cam_ok & vert_front.all(-1)
    # A triangle with NO vertex in front of the camera-origin plane cannot be
    # hit by any forward primary ray (every point of the triangle is a convex
    # combination of its vertices, so its plane projection is <= 0, while any
    # ray point at t > 0 projects > 0). Column 9 therefore carries a
    # three-state flag: 1 = all-front (screen-space rasterization valid),
    # 0 = straddling/degenerate (full-window ray-cast fallback), -1 = provably
    # behind (the host culls the primitive from candidate emission entirely,
    # instead of the old full-window fallback that made every behind-camera
    # primitive a full-screen candidate scan).
    behind = cam_ok & ~vert_front.any(-1)

    safe_denom = torch.where(
        denom.abs() > 1e-20, denom, torch.ones_like(denom))
    hit = ro[:, None, None, :] + (
        big_d[:, None, None, None] / safe_denom[..., None]) * d
    rel = hit - sp[:, None, None, :]
    safe_n2 = n2.clamp_min(1e-30)
    u = (torch.linalg.cross(rel, pby[:, None, None, :])
         * nvec[:, None, None, :]).sum(-1) / safe_n2[:, None, None]
    v = (torch.linalg.cross(pbx[:, None, None, :], rel)
         * nvec[:, None, None, :]).sum(-1) / safe_n2[:, None, None]
    sx = u * half_h + half_w
    sy = v * half_h + half_h
    inv_d = torch.where(valid[..., None], 1.0 / safe_denom,
                        torch.zeros_like(safe_denom))
    flag = valid.to(sx.dtype) - behind.to(sx.dtype)
    packed = torch.cat((sx, sy, inv_d, flag.unsqueeze(-1)), -1)
    out[:frames, :ntri].copy_(packed)
    return out


def _class_pairs(mask, x0, x1, y0, y1, f, device):
    """Expand clipped bboxes into ``(primitive, frame, bbox, chunk-offset)``."""
    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return None
    bx0 = x0[idx]
    by0 = y0[idx]
    bw = x1[idx] - bx0 + 1
    bh = y1[idx] - by0 + 1
    area = bw * bh
    nch = (area + (RASTER_CHUNK - 1)) // RASTER_CHUNK
    # Avoid an explicit GPU scalar read. repeat_interleave determines the
    # dynamic output length; subsequent shape metadata is host-visible.
    rep = torch.repeat_interleave(torch.arange(idx.numel(), device=device), nch)
    if rep.numel() == 0:
        return None
    base = torch.cumsum(nch, 0) - nch
    off = (torch.arange(rep.shape[0], device=device) - base[rep]) * RASTER_CHUNK
    rows = torch.stack([
        idx[rep], torch.full_like(rep, f), bx0[rep], by0[rep],
        bw[rep], bh[rep], off, torch.zeros_like(rep),
    ], -1)
    return rows.to(torch.int32).contiguous()


def _screen_bbox(px, py, front, width, row_lo, row_hi):
    """Conservative clipped bbox from projected corners/vertices.

    Camera-plane straddlers still fall back to the full current row band (the
    callers cull provably-behind primitives before candidate emission).  A
    future implementation could instead clip straddling triangles/polygons
    against the camera or near plane before projection.
    """
    all_front = front.all(-1)
    xmin = px.amin(-1)
    xmax = px.amax(-1)
    ymin = py.amin(-1)
    ymax = py.amax(-1)
    fx0 = (xmin - 1.0).floor().clamp_(0, width - 1).long()
    fx1 = (xmax + 1.0).ceil().clamp_(0, width - 1).long()
    fy0 = (ymin - 1.0).floor().clamp_(row_lo, row_hi).long()
    fy1 = (ymax + 1.0).ceil().clamp_(row_lo, row_hi).long()
    x0 = torch.where(all_front, fx0, torch.zeros_like(fx0))
    x1 = torch.where(all_front, fx1, torch.full_like(fx1, width - 1))
    y0 = torch.where(all_front, fy0, torch.full_like(fy0, row_lo))
    y1 = torch.where(all_front, fy1, torch.full_like(fy1, row_hi))
    on_screen = ((xmax >= -1.0) & (xmin <= width + 1.0)
                 & (ymax >= row_lo - 1.0) & (ymin <= row_hi + 1.0))
    reach = torch.where(all_front, on_screen, torch.ones_like(on_screen))
    return x0, x1, y0, y1, reach


_AABB_SEL = None


def _aabb_corners(lo, hi):
    """``[..., 3]`` box bounds to ``[..., 8, 3]`` corners (any batch shape)."""
    global _AABB_SEL
    if _AABB_SEL is None or _AABB_SEL.device != lo.device:
        _AABB_SEL = torch.tensor(
            [[cx, cy, cz] for cx in (0, 1) for cy in (0, 1) for cz in (0, 1)],
            dtype=torch.bool, device=lo.device)
    sel = _AABB_SEL.view((1,) * (lo.dim() - 1) + _AABB_SEL.shape)
    return torch.where(sel, hi.unsqueeze(-2), lo.unsqueeze(-2))


def _project_points(verts, ro, sp, pbx, pby, half_w, half_h):
    """Vectorized point projection without Python scalar conversions."""
    nvec = torch.linalg.cross(pbx, pby)
    d = verts - ro
    wpn = (d * nvec).sum(-1)
    big_d = ((sp - ro) * nvec).sum(-1)
    safe = torch.where(wpn.abs() < 1e-12, torch.ones_like(wpn), wpn)
    td = big_d / safe
    front = (wpn.abs() >= 1e-12) & (td > 0)
    hit = ro + td.unsqueeze(-1) * d
    rel = hit - sp
    dsq = (nvec * nvec).sum().clamp_min(1e-30)
    u = (torch.linalg.cross(rel, pby.expand_as(rel)) * nvec).sum(-1) / dsq
    v = (torch.linalg.cross(pbx.expand_as(rel), rel) * nvec).sum(-1) / dsq
    return u * half_h + half_w, v * half_h + half_h, front


def _frame_pairs(merged, tri_screen, f, width, row_lo, row_hi, device):
    valid_all = merged["tri_frame_valid"]
    opaque_all = merged["tri_frame_opaque"]
    valid = valid_all[f % valid_all.shape[0]].bool()
    uncertain = merged["tri_alpha_uncertain"]
    uncertain = uncertain[f % uncertain.shape[0]].bool()
    opaque = valid & opaque_all[f % opaque_all.shape[0]].bool() & ~uncertain
    screen = tri_screen[f % tri_screen.shape[0]]
    px = screen[:, 0:3]
    py = screen[:, 3:6]
    front = (screen[:, 9] > 0.5).unsqueeze(-1).expand(-1, 3)
    x0, x1, y0, y1, reach = _screen_bbox(px, py, front, width, row_lo, row_hi)
    # Flag -1 marks a triangle with every vertex behind the camera plane --
    # provably unhittable by a primary ray, so drop it instead of emitting
    # full-window candidate pairs (see precompute_triangle_projection).
    reach = reach & (screen[:, 9] > -0.5)
    return (
        _class_pairs(opaque & reach, x0, x1, y0, y1, f, device),
        _class_pairs(valid & ~opaque & reach, x0, x1, y0, y1, f, device),
    )


def _frame_bez_pairs(merged, f, width, row_lo, row_hi,
                     cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
                     half_w, half_h, device):
    valid_all = merged["bez_frame_valid"]
    opaque_all = merged["bez_frame_opaque"]
    valid = valid_all[f % valid_all.shape[0]].bool()
    opaque = valid & opaque_all[f % opaque_all.shape[0]].bool()
    lo_all = merged["bez_frame_lo"]
    hi_all = merged["bez_frame_hi"]
    lo = lo_all[f % lo_all.shape[0]]
    hi = hi_all[f % hi_all.shape[0]]
    corners = _aabb_corners(lo, hi)
    px, py, front = _project_points(
        corners,
        cam_origin[f % cam_origin.shape[0]],
        screen_point[f % screen_point.shape[0]],
        pixel_basis_x[f % pixel_basis_x.shape[0]],
        pixel_basis_y[f % pixel_basis_y.shape[0]],
        half_w, half_h)
    x0, x1, y0, y1, reach = _screen_bbox(px, py, front, width, row_lo, row_hi)
    # An AABB with no corner in front of the camera plane cannot contain a
    # primary-ray hit (the box is convex, forward ray points project > 0);
    # cull it instead of emitting full-window candidate pairs.
    reach = reach & front.any(-1)
    return (
        _class_pairs(opaque & reach, x0, x1, y0, y1, f, device),
        _class_pairs(valid & ~opaque & reach, x0, x1, y0, y1, f, device),
    )


def precompute_circuit_screen_bounds(
        merged, cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
        half_w, half_h, width, memory):
    """Batched once-per-window screen bounds for bezier circuits.

    The bezier analogue of :func:`precompute_triangle_projection`: the
    per-frame fallback (``_frame_bez_pairs``) re-projects every circuit's
    AABB corners per (tile, frame) -- ~130 small tensor dispatches per call
    that dominate host time on circuit-only scenes.  Only the row-band clamp
    of the candidate bbox is tile-dependent, so everything else is computed
    here in one pass batched over all frames and consumed per tile by
    ``_window_pairs``.  Byte-identical to the per-frame path by
    construction: identical elementwise arithmetic, batched over the frame
    dimension only.

    Every source is independently deduplicatable to T=1, so the tables span
    the longest source and each source is read modulo its own length --
    exactly the indexing the per-frame path applies (see the matching comment
    in ``precompute_triangle_projection``).

    Returns ``(pre_f, pre_x, pre_m)`` arena tensors:
      pre_f ``[F, C, 4]`` f32: unclamped bbox rows ``floor(ymin-1)`` /
          ``ceil(ymax+1)`` and the raw projected ``ymin`` / ``ymax`` extremes
          (for the per-tile row-band on-screen test);
      pre_x ``[F, C, 2]`` i64: the fully clamped bbox columns ``x0`` / ``x1``
          (tiles are row bands, so columns are never tile-clamped);
      pre_m ``[F, C, 5]`` bool: ``all_front``, the all-front reach base
          (``all_front & x_on``), the straddler reach base
          (``~all_front & front_any``), and the opaque / translucent class
          masks.
    """
    lo_all = merged["bez_frame_lo"]
    hi_all = merged["bez_frame_hi"]
    valid_all = merged["bez_frame_valid"]
    opaque_all = merged["bez_frame_opaque"]
    frames = max(
        int(lo_all.shape[0]), int(hi_all.shape[0]),
        int(valid_all.shape[0]), int(opaque_all.shape[0]),
        int(cam_origin.shape[0]), int(screen_point.shape[0]),
        int(pixel_basis_x.shape[0]), int(pixel_basis_y.shape[0]),
    )
    device = lo_all.device
    frame_ids = torch.arange(frames, device=device)
    lo = lo_all.index_select(0, frame_ids % lo_all.shape[0])
    hi = hi_all.index_select(0, frame_ids % hi_all.shape[0])
    corners = _aabb_corners(lo, hi)                          # [F, C, 8, 3]
    ro = cam_origin.index_select(0, frame_ids % cam_origin.shape[0])
    sp = screen_point.index_select(0, frame_ids % screen_point.shape[0])
    pbx = pixel_basis_x.index_select(0, frame_ids % pixel_basis_x.shape[0])
    pby = pixel_basis_y.index_select(0, frame_ids % pixel_basis_y.shape[0])

    # _project_points, batched over the leading frame dimension.
    nvec = torch.linalg.cross(pbx, pby)                      # [F, 3]
    d = corners - ro[:, None, None, :]
    wpn = (d * nvec[:, None, None, :]).sum(-1)               # [F, C, 8]
    big_d = ((sp - ro) * nvec).sum(-1)                       # [F]
    safe = torch.where(wpn.abs() < 1e-12, torch.ones_like(wpn), wpn)
    td = big_d[:, None, None] / safe
    front = (wpn.abs() >= 1e-12) & (td > 0)
    hit = ro[:, None, None, :] + td.unsqueeze(-1) * d
    rel = hit - sp[:, None, None, :]
    dsq = (nvec * nvec).sum(-1).clamp_min(1e-30)             # [F]
    u = (torch.linalg.cross(rel, pby[:, None, None, :].expand_as(rel))
         * nvec[:, None, None, :]).sum(-1) / dsq[:, None, None]
    v = (torch.linalg.cross(pbx[:, None, None, :].expand_as(rel), rel)
         * nvec[:, None, None, :]).sum(-1) / dsq[:, None, None]
    px = u * half_h + half_w
    py = v * half_h + half_h

    # _screen_bbox's tile-independent parts (x is never row-band clamped).
    all_front = front.all(-1)                                # [F, C]
    front_any = front.any(-1)
    xmin = px.amin(-1)
    xmax = px.amax(-1)
    ymin = py.amin(-1)
    ymax = py.amax(-1)
    fx0 = (xmin - 1.0).floor().clamp_(0, width - 1).long()
    fx1 = (xmax + 1.0).ceil().clamp_(0, width - 1).long()
    x0 = torch.where(all_front, fx0, torch.zeros_like(fx0))
    x1 = torch.where(all_front, fx1, torch.full_like(fx1, width - 1))
    x_on = (xmax >= -1.0) & (xmin <= width + 1.0)
    valid = valid_all.index_select(0, frame_ids % valid_all.shape[0]).bool()
    opaque = valid & opaque_all.index_select(
        0, frame_ids % opaque_all.shape[0]).bool()

    ncirc = int(lo.shape[1])
    pre_f = memory.get_tensor((frames, ncirc, 4), torch.float32)
    pre_f.copy_(torch.stack(
        ((ymin - 1.0).floor(), (ymax + 1.0).ceil(), ymin, ymax), -1))
    pre_x = memory.get_tensor((frames, ncirc, 2), torch.int64)
    pre_x.copy_(torch.stack((x0, x1), -1))
    # all_front implies front_any (eight corners), so the all-front reach
    # base omits the redundant ``& front_any``.
    pre_m = memory.get_tensor((frames, ncirc, 5), torch.bool)
    pre_m.copy_(torch.stack(
        (all_front, all_front & x_on, ~all_front & front_any,
         opaque, valid & ~opaque), -1))
    return pre_f, pre_x, pre_m


def precompute_triangle_screen_bounds(merged, tri_screen, width, memory):
    """Batched once-per-window candidate screen bounds for flat triangles.

    Companion of :func:`precompute_circuit_screen_bounds`, consuming the
    per-batch projection table (``tri_screen``) instead of re-projecting.
    ``_frame_pairs`` derived the same bboxes and class masks per
    (tile, frame); only the row-band clamp is tile-dependent, so the rest is
    batched here over all frames into the same three-table schema consumed by
    ``_window_pairs``.  ``tri_screen`` column 9 is the three-state front flag
    (1 all-front / 0 straddling / -1 behind); all three vertices share it, so
    ``all_front`` is the flag itself and the behind-cull folds into the
    straddler reach base.  Byte-identical to the per-frame path by
    construction.
    """
    valid_all = merged["tri_frame_valid"]
    opaque_all = merged["tri_frame_opaque"]
    unc_all = merged["tri_alpha_uncertain"]
    frames = max(
        int(tri_screen.shape[0]), int(valid_all.shape[0]),
        int(opaque_all.shape[0]), int(unc_all.shape[0]),
    )
    device = tri_screen.device
    frame_ids = torch.arange(frames, device=device)
    screen = tri_screen.index_select(0, frame_ids % tri_screen.shape[0])
    px = screen[..., 0:3]
    py = screen[..., 3:6]
    flag = screen[..., 9]
    all_front = flag > 0.5
    not_behind = flag > -0.5

    xmin = px.amin(-1)
    xmax = px.amax(-1)
    ymin = py.amin(-1)
    ymax = py.amax(-1)
    fx0 = (xmin - 1.0).floor().clamp_(0, width - 1).long()
    fx1 = (xmax + 1.0).ceil().clamp_(0, width - 1).long()
    x0 = torch.where(all_front, fx0, torch.zeros_like(fx0))
    x1 = torch.where(all_front, fx1, torch.full_like(fx1, width - 1))
    x_on = (xmax >= -1.0) & (xmin <= width + 1.0)
    valid = valid_all.index_select(0, frame_ids % valid_all.shape[0]).bool()
    unc = unc_all.index_select(0, frame_ids % unc_all.shape[0]).bool()
    opaque = valid & opaque_all.index_select(
        0, frame_ids % opaque_all.shape[0]).bool() & ~unc

    ntri = int(screen.shape[1])
    pre_f = memory.get_tensor((frames, ntri, 4), torch.float32)
    pre_f.copy_(torch.stack(
        ((ymin - 1.0).floor(), (ymax + 1.0).ceil(), ymin, ymax), -1))
    pre_x = memory.get_tensor((frames, ntri, 2), torch.int64)
    pre_x.copy_(torch.stack((x0, x1), -1))
    # all_front (flag == 1) implies not-behind, so the all-front reach base
    # omits the redundant ``& not_behind``.
    pre_m = memory.get_tensor((frames, ntri, 5), torch.bool)
    pre_m.copy_(torch.stack(
        (all_front, all_front & x_on, ~all_front & not_behind,
         opaque, valid & ~opaque), -1))
    return pre_f, pre_x, pre_m


def _class_pairs_flat(mask, x0, x1, y0, y1, f_abs, device):
    """Chunk expansion over a ``[frames, C]`` window in one pass.

    Row content and ordering are identical to per-frame ``_class_pairs``
    calls concatenated in ascending frame order: flattening row-major keeps
    the (frame, circuit) lexicographic order the per-frame loop produced.
    """
    ncirc = mask.shape[1]
    idx = mask.reshape(-1).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return None
    bx0 = x0.reshape(-1)[idx]
    by0 = y0.reshape(-1)[idx]
    bw = x1.reshape(-1)[idx] - bx0 + 1
    bh = y1.reshape(-1)[idx] - by0 + 1
    area = bw * bh
    nch = (area + (RASTER_CHUNK - 1)) // RASTER_CHUNK
    rep = torch.repeat_interleave(
        torch.arange(idx.numel(), device=device), nch)
    if rep.numel() == 0:
        return None
    base = torch.cumsum(nch, 0) - nch
    off = (torch.arange(rep.shape[0], device=device) - base[rep]) * RASTER_CHUNK
    rows = torch.stack([
        (idx % ncirc)[rep], f_abs.index_select(0, idx // ncirc)[rep],
        bx0[rep], by0[rep], bw[rep], bh[rep], off, torch.zeros_like(rep),
    ], -1)
    return rows.to(torch.int32).contiguous()


def _window_pairs(bounds, time_start, g0, g1, ppf, width, device):
    """Emit a tile's candidate pairs for all covered frames at once.

    Consumes the ``precompute_circuit_screen_bounds`` /
    ``precompute_triangle_screen_bounds`` tables (both use the same schema);
    only the per-frame row-band clamp of the bbox and the chunk expansion
    happen here.  Replicates ``_frame_bez_pairs`` / ``_frame_pairs`` +
    ``_screen_bbox`` byte-for-byte (same clamp/cast order on the same
    values, per-frame bounds supplied as broadcast tensors instead of
    Python scalars).
    """
    pre_f, pre_x, pre_m = bounds
    f0_rel = g0 // ppf
    f1_rel = (g1 - 1) // ppf
    f_rel = torch.arange(f0_rel, f1_rel + 1, device=device)
    f_abs = f_rel + time_start
    lo_p = (g0 - f_rel * ppf).clamp_(min=0)
    hi_p = (g1 - f_rel * ppf).clamp_(max=ppf)
    row_lo = (lo_p // width).view(-1, 1)                     # [Ft, 1] i64
    row_hi = ((hi_p - 1) // width).view(-1, 1)
    rows = f_abs % pre_f.shape[0]
    fy = pre_f.index_select(0, rows)                         # [Ft, C, 4]
    x01 = pre_x.index_select(0, rows)
    m = pre_m.index_select(0, rows)
    all_front = m[..., 0]
    rl_f = row_lo.to(torch.float32)
    rh_f = row_hi.to(torch.float32)
    fy0 = fy[..., 0].clamp(min=rl_f, max=rh_f).long()
    fy1 = fy[..., 1].clamp(min=rl_f, max=rh_f).long()
    y0 = torch.where(all_front, fy0, row_lo)
    y1 = torch.where(all_front, fy1, row_hi)
    y_on = (fy[..., 3] >= rl_f - 1.0) & (fy[..., 2] <= rh_f + 1.0)
    reach = (m[..., 1] & y_on) | m[..., 2]
    x0 = x01[..., 0]
    x1 = x01[..., 1]
    return (
        _class_pairs_flat(m[..., 3] & reach, x0, x1, y0, y1, f_abs, device),
        _class_pairs_flat(m[..., 4] & reach, x0, x1, y0, y1, f_abs, device),
    )


def _arena_tensor(memory, shape, dtype, fill=None):
    out = memory.get_tensor(shape, dtype)
    if fill is not None:
        out.fill_(fill)
    return out


def _exact_fragment_order(frag_key, frag_ref, layer_offset_triangles):
    """Return one gather order matching classic depth-bin/layer semantics."""
    is_bez = frag_ref < 0
    bez_code = (-frag_ref - 1).clamp_min(0)
    bez_layer = bez_code >> 1
    tri_layer = frag_ref + int(layer_offset_triangles)
    layer = torch.where(is_bez, bez_layer, tri_layer).to(torch.int64)
    layer_order = torch.argsort(layer, descending=True, stable=True)

    key_l = frag_key.index_select(0, layer_order)
    pixel = key_l >> 32
    t_bits = (key_l & 0xFFFFFFFF).to(torch.int32)
    # dtype-view reinterprets IEEE bits; it does not allocate a numeric cast.
    t = t_bits.view(torch.float32)
    depth_bin = torch.floor(t / DEPTH_TIE_EPSILON).to(torch.int64)
    depth_bin.clamp_(0, 0x7FFFFFFF)
    primary_key = (pixel << 32) | depth_bin
    depth_order = torch.argsort(primary_key, stable=True)
    return layer_order.index_select(0, depth_order)


def raster_iteration_zero(
        merged, tri_screen, tri_bounds, bez_bounds, memory,
        cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
        pixel_world_scale, layer_offsets, gen_meta, light_pos, light_col,
        num_lights, col_row_arr, frag_flag, frag_pipelines, skip_unlit_normal,
        refraction_flag, time_start, width, height, half_w, half_h,
        tile_start, tn_primary, state, rs_pix, pix_accum, rs_alloc,
        shadow_flag, t_bvh, pn_bvh, bez_bvh,
        layer_offset_triangles, layer_offset_pn, max_bounces):
    """Raster, order, resolve, and seed compact primary continuations."""
    (rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
     _rs_kt, _rs_kl, _rs_ka, _rs_kb, _rs_kp, _rs_kf) = state
    device = pix_accum.device
    ppf = width * height
    g0 = tile_start
    g1 = tile_start + tn_primary
    has_tri = 1 if int(merged.get("num_triangles", 0)) > 0 else 0
    has_pn = 1 if int(merged.get("num_pn", 0)) > 0 else 0
    has_bez = 1 if int(merged.get("num_circuits", 0)) > 0 else 0

    tri_opaque, tri_trans, bez_opaque, bez_trans = [], [], [], []
    use_tri_pre = bool(has_tri) and tri_bounds is not None
    use_bez_pre = bool(has_bez) and bez_bounds is not None
    if (has_tri and not use_tri_pre) or (has_bez and not use_bez_pre):
        for f_rel in range(g0 // ppf, (g1 - 1) // ppf + 1):
            f = time_start + f_rel
            lo_p = max(g0 - f_rel * ppf, 0)
            hi_p = min(g1 - f_rel * ppf, ppf)
            row_lo = lo_p // width
            row_hi = (hi_p - 1) // width
            if has_tri and not use_tri_pre:
                po, pt = _frame_pairs(
                    merged, tri_screen, f, width, row_lo, row_hi, device)
                if po is not None:
                    tri_opaque.append(po)
                if pt is not None:
                    tri_trans.append(pt)
            if has_bez and not use_bez_pre:
                po, pt = _frame_bez_pairs(
                    merged, f, width, row_lo, row_hi, cam_origin,
                    screen_point, pixel_basis_x, pixel_basis_y, half_w,
                    half_h, device)
                if po is not None:
                    bez_opaque.append(po)
                if pt is not None:
                    bez_trans.append(pt)
    if use_tri_pre:
        po, pt = _window_pairs(
            tri_bounds, time_start, g0, g1, ppf, width, device)
        if po is not None:
            tri_opaque.append(po)
        if pt is not None:
            tri_trans.append(pt)
    if use_bez_pre:
        po, pt = _window_pairs(
            bez_bounds, time_start, g0, g1, ppf, width, device)
        if po is not None:
            bez_opaque.append(po)
        if pt is not None:
            bez_trans.append(pt)

    def _cat(parts):
        return (torch.cat(parts, 0) if len(parts) > 1
                else (parts[0] if parts else None))

    po_t, pt, po_b, pb = map(_cat,
                             (tri_opaque, tri_trans, bez_opaque, bez_trans))
    zbuf = _arena_tensor(memory, (tn_primary,), torch.int64, Z_SENTINEL)
    ss = 1 if rt_settings.RASTER_SS else 0
    tri_pos = merged["tri_pos"]
    cam_args = (cam_origin, screen_point, pixel_basis_x, pixel_basis_y)
    geo_args = (int(time_start), int(width), int(height), float(half_w),
                float(half_h), int(tile_start), int(tn_primary))
    tri_color_args = (
        merged["tri_colors"], col_row_arr, merged["tri_uvs"],
        merged["tri_tex_meta"], merged["textures"],
        int(merged["num_colored_triangles"]),
    )
    bez_geom = (
        pixel_world_scale, merged["circuit_meta"], merged["circuit_colors"],
        merged["circuit_border_colors"], merged["edges_2d"],
        merged["edge_accel"],
    )
    if po_t is not None:
        raster_tri_z(po_t, int(po_t.shape[0]), tri_pos, tri_screen, *cam_args,
                     *geo_args, ss, float(layer_offset_triangles), zbuf)
    if po_b is not None:
        raster_bez_z(po_b, int(po_b.shape[0]), *cam_args,
                     pixel_world_scale, merged["circuit_meta"],
                     merged["circuit_colors"], merged["edges_2d"],
                     merged["edge_accel"], *geo_args, zbuf)

    bcounts = tcounts = None
    if pb is not None:
        bcounts = _arena_tensor(memory, (pb.shape[0],), torch.int32, 0)
        raster_bez_count(pb, int(pb.shape[0]), *cam_args, *bez_geom,
                         *geo_args, zbuf, bcounts)
    if pt is not None:
        tcounts = _arena_tensor(memory, (pt.shape[0],), torch.int32, 0)
        raster_tri_count(pt, int(pt.shape[0]), tri_pos, tri_screen,
                         *tri_color_args, *cam_args, *geo_args, ss,
                         float(layer_offset_triangles), zbuf, tcounts)

    count_parts = [x for x in (bcounts, tcounts) if x is not None]
    num_frags = 0
    bez_offsets = tri_offsets = None
    if count_parts:
        counts = (torch.cat(count_parts, 0) if len(count_parts) > 1
                  else count_parts[0])
        counts64 = counts.to(torch.int64)
        prefix = torch.cumsum(counts64, 0) - counts64
        num_frags = int(counts64.sum().item())
        cursor = 0
        if bcounts is not None:
            bez_offsets = prefix[cursor:cursor + bcounts.shape[0]].to(torch.int32)
            cursor += bcounts.shape[0]
        if tcounts is not None:
            tri_offsets = prefix[cursor:cursor + tcounts.shape[0]].to(torch.int32)

    run_offsets = _arena_tensor(
        memory, (tn_primary + 1,), torch.int32, 0)
    if num_frags:
        frag_key_u = _arena_tensor(memory, (num_frags,), torch.int64)
        frag_ref_u = _arena_tensor(memory, (num_frags,), torch.int32)
        frag_ab_u = _arena_tensor(memory, (num_frags, 2), torch.float32)
        if bcounts is not None:
            raster_bez_write(pb, int(pb.shape[0]), bez_offsets, *cam_args,
                             *bez_geom, *geo_args, zbuf, frag_key_u,
                             frag_ref_u, frag_ab_u)
        if tcounts is not None:
            raster_tri_write(pt, int(pt.shape[0]), tri_offsets, tri_pos,
                             tri_screen, *tri_color_args, *cam_args, *geo_args,
                             ss, float(layer_offset_triangles), zbuf, frag_key_u,
                             frag_ref_u, frag_ab_u)
        order = _exact_fragment_order(
            frag_key_u, frag_ref_u, layer_offset_triangles)
        frag_key = _arena_tensor(memory, (num_frags,), torch.int64)
        frag_ref = _arena_tensor(memory, (num_frags,), torch.int32)
        frag_ab = _arena_tensor(memory, (num_frags, 2), torch.float32)
        frag_key.copy_(frag_key_u.index_select(0, order))
        frag_ref.copy_(frag_ref_u.index_select(0, order))
        frag_ab.copy_(frag_ab_u.index_select(0, order))
        pix = frag_key >> 32
        counts_per_pixel = torch.bincount(pix, minlength=tn_primary)
        run_offsets[1:].copy_(torch.cumsum(
            counts_per_pixel.to(torch.int32), 0))
    else:
        frag_key = _arena_tensor(memory, (1,), torch.int64, 0)
        frag_ref = _arena_tensor(memory, (1,), torch.int32, 0)
        frag_ab = _arena_tensor(memory, (1, 2), torch.float32, 0)

    # Exact sparse shadow queue. The upper bound is one accepted triangle event
    # per raw fragment plus one terminal visibility winner per pixel; the build
    # kernel only reserves entries that survive seam/transport decisions.
    frag_shadow_id = _arena_tensor(
        memory, (max(1, num_frags),), torch.int32, -1)
    z_shadow_id = _arena_tensor(memory, (tn_primary,), torch.int32, -1)
    if shadow_flag:
        max_events = max(1, num_frags + tn_primary)
        event_pos = _arena_tensor(memory, (max_events, 3), torch.float32)
        event_snrm = _arena_tensor(memory, (max_events, 3), torch.float32)
        event_fnrm = _arena_tensor(memory, (max_events, 3), torch.float32)
        event_frame = _arena_tensor(memory, (max_events,), torch.int32)
        event_count = _arena_tensor(memory, (1,), torch.int32, 0)
        raster_shadow_event_build(
            int(tn_primary), run_offsets, frag_key, frag_ref, frag_ab, zbuf,
            tri_pos, tri_screen, merged["tri_norm"], merged["tri_extra"],
            merged["tri_colors"], merged["tri_uvs"],
            merged["tri_tex_meta"], merged["textures"],
            int(merged["num_colored_triangles"]), col_row_arr,
            merged["circuit_meta"], merged["circuit_colors"],
            merged["circuit_border_colors"], merged["edges_2d"],
            merged["edge_accel"], pixel_world_scale,
            float(layer_offset_triangles), int(refraction_flag), ss, has_bez,
            int(time_start), int(width), int(height), int(tile_start),
            *cam_args, gen_meta, int(max_bounces),
            frag_shadow_id, z_shadow_id, event_pos, event_snrm, event_fnrm,
            event_frame, event_count)
        num_events = int(event_count.item())
        shadow_vis = _arena_tensor(
            memory, (max(1, num_events), max(1, int(num_lights))),
            torch.float32, 1.0)
        if num_events:
            from algan.rendering.raytracing.refit_bvh import RefitBVH
            raster_shadow_trace(
                num_events, event_pos, event_snrm, event_fnrm, event_frame,
                t_bvh.blocks, t_bvh.node_miss, t_bvh.leaf_prim,
                t_bvh.leaf_tspan, int(t_bvh.first_leaf),
                merged["tri_pos"], merged["tri_colors"], merged["tri_uvs"],
                merged["tri_tex_meta"], merged["textures"],
                int(merged["num_colored_triangles"]),
                pn_bvh.blocks, pn_bvh.node_miss, pn_bvh.leaf_prim,
                pn_bvh.leaf_tspan, int(pn_bvh.first_leaf),
                merged["pn_ctrl"], merged["pn_obb"], merged["pn_colors"],
                bez_bvh.blocks, bez_bvh.node_miss, bez_bvh.leaf_prim,
                bez_bvh.leaf_tspan, int(bez_bvh.first_leaf),
                merged["circuit_meta"], merged["circuit_colors"],
                merged["circuit_border_colors"], merged["edges_2d"],
                merged["edge_accel"], light_pos, light_col, int(num_lights),
                pixel_world_scale, float(layer_offset_triangles),
                float(layer_offset_pn),
                1 if isinstance(t_bvh, RefitBVH) else 0,
                has_tri, has_pn, has_bez, shadow_vis)
    else:
        shadow_vis = _arena_tensor(memory, (1, 1), torch.float32, 1.0)

    raster_first_shade(
        int(tn_primary), run_offsets, frag_key, frag_ref, frag_ab, zbuf,
        merged["tri_pos"], tri_screen, merged["tri_norm"],
        merged["tri_extra"], merged["tri_colors"], merged["tri_uvs"],
        merged["tri_tex_meta"], merged["textures"],
        int(merged["num_colored_triangles"]), col_row_arr,
        merged["tri_mat_id"], merged["tri_mat"],
        merged["circuit_meta"], merged["circuit_colors"],
        merged["circuit_border_colors"], pixel_world_scale,
        merged["edges_2d"], merged["edge_accel"],
        light_pos, light_col, int(num_lights),
        layer_offsets, int(frag_flag), frag_pipelines, int(refraction_flag),
        int(skip_unlit_normal), ss, has_bez, int(shadow_flag),
        int(time_start), int(width), int(height), int(tile_start),
        *cam_args, gen_meta, rs_ro, rs_rd, rs_acc, rs_sca, rs_int, rs_pix,
        pix_accum, rs_alloc, frag_shadow_id, z_shadow_id, shadow_vis)
