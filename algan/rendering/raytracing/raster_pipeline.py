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

import os

import torch

from algan.settings import SETTINGS

rt_settings = SETTINGS.raytracing
from algan.rendering.raytracing.raster_taichi import (
    _AA_DUMP_COLS as AA_DUMP_COLS,
    _AA_MASK_ALL as AA_MASK_ALL,
)
from algan.rendering.raytracing.raster_taichi import (
    _BEZ_BORDER_BITS,
    AA_FULL_COVERAGE,
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
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    DEPTH_TIE_EPSILON,
)

# Per-fragment walk dump (DESIGN_analytic_aa_v2.md ss7.1). Debug-only:
# ``ALGAN_AA_DUMP="px,py,frame"`` (kernel pixel coordinates -- an output PNG's
# row y is kernel row ``height - 1 - y``) makes both walk kernels record one
# row per fragment they process at that pixel, printed after the launch and
# kept here for the golden-walk harness (``benchmarks/_aa_dump_check.py``) to
# recompute and diff. Read live per launch; the kernels compile the dump path
# out entirely when it is off, so the production kernels are untouched.
_AA_DUMP_ROWS = 512
LAST_AA_DUMP = {}


def _aa_dump_request():
    """The requested (px, py, frame) from ``ALGAN_AA_DUMP``, or ``None``."""
    spec = os.environ.get("ALGAN_AA_DUMP", "")
    if not spec:
        return None
    try:
        px, py, f = (int(v) for v in spec.split(","))
    except ValueError:
        return None
    return px, py, f


def _aa_dump_buffer(req, device):
    """A fresh dump buffer with the control row filled.

    A plain torch allocation, never an arena tensor: this is a diagnostic and
    must not perturb the runtime memory model's view of a chunk's footprint.
    """
    buf = torch.zeros((_AA_DUMP_ROWS, AA_DUMP_COLS), dtype=torch.float32,
                      device=device)
    buf[0, 0] = float(req[0])
    buf[0, 1] = float(req[1])
    buf[0, 2] = float(req[2])
    return buf


_AA_DUMP_PLACEHOLDERS = {}


def _aa_dump_arg(device):
    """The 1-row stand-in the kernels take when the dump is off."""
    t = _AA_DUMP_PLACEHOLDERS.get(device)
    if t is None:
        t = torch.zeros((1, AA_DUMP_COLS), dtype=torch.float32, device=device)
        _AA_DUMP_PLACEHOLDERS[device] = t
    return t


_AA_DUMP_NOTES = {0: "", 1: "eff-skip", 2: "bounce", 3: "occl", 4: "far-clip",
                  5: "invalid", 6: "seam-skip"}
_AA_DUMP_KINDS = {0: "tri", 1: "bez", 2: "z-tri", 3: "z-bez"}


def _aa_dump_emit(tag, buf):
    """Print the walk dump and stash it (numpy) for the harness."""
    rows = buf.cpu().numpy()
    n = min(int(rows[0, 3]), rows.shape[0] - 1)
    LAST_AA_DUMP[tag] = rows[1 : 1 + n].copy()
    if n <= 0:
        return
    px, py, f = int(rows[0, 0]), int(rows[0, 1]), int(rows[0, 2])
    print(f"[aa-dump:{tag}] pixel ({px},{py}) frame {f}: {n} rows")
    for r in rows[1 : 1 + n]:
        if r[0] < 0:
            svis = " ".join(f"{v:.4f}" for v in r[16:24])
            print(f"[aa-dump:{tag}]   end bounced={int(r[1])} done={int(r[2])}"
                  f" processed={int(r[3])} vis_all={r[4]:.5f}"
                  f" acc=({r[5]:.4f},{r[6]:.4f},{r[7]:.4f},{r[8]:.4f})"
                  f" w=({r[9]:.4f},{r[10]:.4f},{r[11]:.4f}) svis=[{svis}]")
            continue
        kind = _AA_DUMP_KINDS.get(int(r[1]), "?")
        note = _AA_DUMP_NOTES.get(int(r[2]), "?")
        svis = " ".join(f"{v:.4f}" for v in r[16:24])
        print(f"[aa-dump:{tag}]   q={int(r[0]):3d} {kind:5s} {note:9s}"
              f" ref={int(r[3])} sid={int(r[4])} face={int(r[5])}"
              f" msk={int(r[6]):02x} cov={r[7]:.5f} pop={int(r[8])}"
              f" corr={r[9]:.5f} eff={r[10]:.5f} a_mat={r[11]:.4f}"
              f" alpha={r[12]:.5f} ts={r[13]:.4f} rmax={r[14]:.4f}"
              f" t={r[15]:.5f} svis=[{svis}]")


def precompute_triangle_projection(
    merged,
    cam_origin,
    screen_point,
    pixel_basis_x,
    pixel_basis_y,
    half_w,
    half_h,
    memory,
):
    """Prepare one compact projection record per frame and flat triangle.

    Columns 0:3 are continuous screen x, 3:6 screen y, 6:9 reciprocal
    perspective divisors, and 9 is one when all vertices are safely in front of
    the camera plane.  Invalid/straddling triangles retain zeros and use the
    exact ray-cast fallback in the kernels.  The result is camera-specific and
    is built once per render batch, rather than once per primitive chunk and
    raster phase.

    Under triangle analytic anti-aliasing the table is widened to 13 columns,
    10:13 holding the reciprocal SCREEN lengths of the three edges (edge ``i``
    faces vertex ``i``).  ``_ss_pixel``'s edge functions are cross products, so
    dividing by the edge length turns each into the signed distance in pixels
    from the pixel centre to that edge -- the quantity analytic coverage is a
    box filter of.  The extra columns are allocated only when the feature is on
    (3 floats per frame per triangle is real memory on a dense scene), and the
    kernels gate on the width the host actually produced, never on the live
    toggle, so a table and a kernel can never disagree mid-render.
    """
    tri_pos = merged["tri_pos"]
    # Every materialized input may be independently deduplicated to T=1.  The
    # projection table must span the longest dynamic input, then index every
    # source modulo its own time dimension; using only the camera length would
    # freeze moving geometry whenever the camera itself is static.
    frames = max(
        int(tri_pos.shape[0]),
        int(cam_origin.shape[0]),
        int(screen_point.shape[0]),
        int(pixel_basis_x.shape[0]),
        int(pixel_basis_y.shape[0]),
        int(merged["tri_frame_valid"].shape[0]),
    )
    ntri = int(merged.get("num_triangles", 0))
    ncol = 13 if rt_settings.analytic_aa_tri_active() else 10
    out = memory.get_tensor((max(1, frames), max(1, ntri), ncol), torch.float32)
    out.zero_()
    # Reported rather than inferred: ``frames`` here is the longest dynamic
    # input, not the batch's frame count (any input may be deduplicated to
    # T=1), and the row count is clamped to at least one.
    memory.note_scope_params(tri_screen_cells=max(1, frames) * max(1, ntri) * ncol)
    if ntri == 0:
        return out

    frame_ids = torch.arange(frames, device=tri_pos.device)
    verts = tri_pos.index_select(0, frame_ids % tri_pos.shape[0]).view(
        frames, ntri, 3, 3
    )
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
    behind = ~cam_ok | (cam_ok & ~vert_front.any(-1))

    safe_denom = torch.where(denom.abs() > 1e-20, denom, torch.ones_like(denom))
    hit = (
        ro[:, None, None, :] + (big_d[:, None, None, None] / safe_denom[..., None]) * d
    )
    rel = hit - sp[:, None, None, :]
    safe_n2 = n2.clamp_min(1e-30)
    u = (torch.linalg.cross(rel, pby[:, None, None, :]) * nvec[:, None, None, :]).sum(
        -1
    ) / safe_n2[:, None, None]
    v = (torch.linalg.cross(pbx[:, None, None, :], rel) * nvec[:, None, None, :]).sum(
        -1
    ) / safe_n2[:, None, None]
    sx = u * half_h + half_w
    sy = v * half_h + half_h
    inv_d = torch.where(
        valid[..., None], 1.0 / safe_denom, torch.zeros_like(safe_denom)
    )
    flag = valid.to(sx.dtype) - behind.to(sx.dtype)
    parts = [sx, sy, inv_d, flag.unsqueeze(-1)]
    if ncol >= 13:
        # Edge i faces vertex i: edge 0 is V1->V2, edge 1 is V2->V0, edge 2 is
        # V0->V1 -- the same cyclic assignment _ss_pixel's e0/e1/e2 use.
        nxt = [1, 2, 0]
        prv = [2, 0, 1]
        ex = sx[..., prv] - sx[..., nxt]
        ey = sy[..., prv] - sy[..., nxt]
        elen = torch.sqrt(ex * ex + ey * ey)
        inv_len = torch.where(
            valid[..., None] & (elen > 1e-12),
            1.0 / elen.clamp_min(1e-12),
            torch.zeros_like(elen),
        )
        parts.append(inv_len)
    packed = torch.cat(parts, -1)
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
    rows = torch.stack(
        [
            idx[rep],
            torch.full_like(rep, f),
            bx0[rep],
            by0[rep],
            bw[rep],
            bh[rep],
            off,
            torch.zeros_like(rep),
        ],
        -1,
    )
    return rows.to(torch.int32).contiguous()


def _screen_bbox(px, py, front, width, row_lo, row_hi, clip=None):
    """Conservative clipped bbox from projected corners/vertices.

    ``clip`` is the ``_clipped_screen_extents`` result for the same primitives.
    Straddlers it could bound are treated exactly like all-front primitives --
    real bbox, real on-screen test -- and only the rest fall back to the full
    current row band.  (Callers cull provably-behind primitives separately,
    before candidate emission.)
    """
    all_front = front.all(-1)
    xmin = px.amin(-1)
    xmax = px.amax(-1)
    ymin = py.amin(-1)
    ymax = py.amax(-1)
    bounded = all_front
    if clip is not None:
        clip_x0, clip_x1, clip_y0, clip_y1, clip_ok = clip
        # All-front rows keep the projected-vertex extents *expression*, not
        # merely an equal value, so their bbox stays bit-for-bit what it was
        # before straddler clipping existed.
        bounded = all_front | clip_ok
        xmin = torch.where(all_front, xmin, clip_x0)
        xmax = torch.where(all_front, xmax, clip_x1)
        ymin = torch.where(all_front, ymin, clip_y0)
        ymax = torch.where(all_front, ymax, clip_y1)
    fx0 = (xmin - 1.0).floor().clamp_(0, width - 1).long()
    fx1 = (xmax + 1.0).ceil().clamp_(0, width - 1).long()
    fy0 = (ymin - 1.0).floor().clamp_(row_lo, row_hi).long()
    fy1 = (ymax + 1.0).ceil().clamp_(row_lo, row_hi).long()
    x0 = torch.where(bounded, fx0, torch.zeros_like(fx0))
    x1 = torch.where(bounded, fx1, torch.full_like(fx1, width - 1))
    y0 = torch.where(bounded, fy0, torch.full_like(fy0, row_lo))
    y1 = torch.where(bounded, fy1, torch.full_like(fy1, row_hi))
    on_screen = (
        (xmax >= -1.0)
        & (xmin <= width + 1.0)
        & (ymax >= row_lo - 1.0)
        & (ymin <= row_hi + 1.0)
    )
    reach = torch.where(bounded, on_screen, torch.ones_like(on_screen))
    return x0, x1, y0, y1, reach


# Hull edges as ``(from, to)`` vertex-index pairs, for the clip below.
_TRI_EDGES = ((0, 1), (1, 2), (2, 0))
# Matches ``_aabb_corners``' bit order: corner index is (x, y, z) hi-bits with
# x most significant, so the three axis neighbours of a corner differ by 4/2/1.
_BOX_EDGES = tuple((i, i ^ bit) for bit in (4, 2, 1) for i in range(8) if not (i & bit))
_EDGE_CACHE = {}


def _edge_index(edges, device):
    key = (edges, device)
    cached = _EDGE_CACHE.get(key)
    if cached is None:
        flat = torch.tensor(edges, dtype=torch.long, device=device)
        cached = (flat[:, 0], flat[:, 1])
        _EDGE_CACHE[key] = cached
    return cached


# Fraction of a primitive's own maximum depth that the straddler clip plane
# sits in front of the camera plane.  Small enough that the sliver it discards
# projects far outside any screen (see ``_clipped_screen_extents``), large
# enough that every clipped projection stays comfortably inside float32.
_CLIP_DEPTH_FRACTION = 1e-5


def _straddle_clip_wanted(straddles):
    """Whether to pay for the camera-plane clip at all.

    ``straddles`` is the per-primitive straddler mask.  The clip is a saving
    only when something actually straddles; on an ordinary scene where nothing
    does it is pure overhead (~8% of a triangle-heavy render), so it is skipped
    outright.  One host transfer per call, next to the one
    ``_class_any_flags`` already makes.
    """
    return rt_settings.RASTER_STRADDLE_CLIP and bool(straddles.any().item())


def _clipped_screen_extents(verts, edges, ro, sp, pbx, pby, half_w, half_h):
    """Screen extent of the part of a convex hull a primary ray can reach.

    ``verts`` is ``[F, N, K, 3]`` world points whose convex hull bounds each
    primitive and ``edges`` the ``(from, to)`` vertex-index pairs of that hull's
    edges; the camera arguments are ``[F, 3]``.

    A primitive straddling the camera plane has no bounded projection: its
    vertices behind the plane project to the wrong side of the screen, and ones
    on it project to infinity.  The front-end therefore used to hand every
    straddler the entire window as its candidate bbox.  An orbiting camera --
    one travelling around the scene without turning to follow it -- puts most of
    the scene in exactly that state, and at HD one full-window primitive-frame
    is ~65k candidate chunks, so a few hundred of them exhaust render memory
    before a single ray is cast.

    Perspective projection maps lines to lines, so the projection of the part of
    a convex hull in front of the camera is the convex hull of the projections
    of that hull clipped to the front half-space: its front vertices plus one
    point per edge crossing the plane.  Clipping a hair *in front* of the plane
    -- at ``_CLIP_DEPTH_FRACTION`` of the primitive's own maximum depth -- keeps
    every projected point finite.  The sliver between that plane and the camera
    plane is discarded, which is only unsafe for a primitive passing essentially
    through the camera origin: everything else in it projects far outside any
    screen.  ``bounded`` reports that case as False so callers keep the
    full-window fallback for it.

    Returns ``(xmin, xmax, ymin, ymax, bounded)``, all ``[F, N]``; the extents
    are in the same continuous pixel space as
    :func:`precompute_triangle_projection` and are meaningless where ``bounded``
    is False.
    """
    ro_b = ro[:, None, None, :]
    sp_b = sp[:, None, None, :]
    pbx_b = pbx[:, None, None, :]
    pby_b = pby[:, None, None, :]
    nvec = torch.linalg.cross(pbx, pby)
    nvec_b = nvec[:, None, None, :]
    inv_n2 = (1.0 / (nvec * nvec).sum(-1).clamp_min(1e-30))[:, None, None]
    big_d = ((sp - ro) * nvec).sum(-1)
    # Depth measured so that "in front of the camera plane" is positive,
    # matching precompute_triangle_projection's front test.
    sign = torch.where(big_d >= 0, 1.0, -1.0)
    front_d = big_d * sign  # [F]

    rel_verts = verts - ro_b
    depth = (rel_verts * nvec_b).sum(-1) * sign[:, None, None]
    clip_depth = (_CLIP_DEPTH_FRACTION * depth.amax(-1)).clamp_min(1e-30)
    keep = depth >= clip_depth.unsqueeze(-1)

    lo_i, hi_i = _edge_index(edges, verts.device)
    depth_a = depth.index_select(-1, lo_i)
    depth_b = depth.index_select(-1, hi_i)
    crosses = keep.index_select(-1, lo_i) ^ keep.index_select(-1, hi_i)
    span = depth_b - depth_a
    step = (clip_depth.unsqueeze(-1) - depth_a) / torch.where(
        span.abs() > 1e-30, span, torch.ones_like(span)
    )
    vert_a = verts.index_select(-2, lo_i)
    rel_edge = (
        vert_a + step.unsqueeze(-1) * (verts.index_select(-2, hi_i) - vert_a)
    ) - ro_b

    # Vertices project through their own depth; edge points sit on the clip
    # plane *by construction*, so they use that depth analytically rather than
    # a recomputed one that floating point could push to the far side.
    rel = torch.cat((rel_verts, rel_edge), -2)
    scale = torch.cat(
        (
            front_d[:, None, None] / depth.clamp_min(1e-30),
            (front_d[:, None] / clip_depth).unsqueeze(-1).expand_as(step),
        ),
        -1,
    )
    hit = ro_b + scale.unsqueeze(-1) * rel
    offset = hit - sp_b
    u = (torch.linalg.cross(offset, pby_b.expand_as(offset)) * nvec_b).sum(-1) * inv_n2
    v = (torch.linalg.cross(pbx_b.expand_as(offset), offset) * nvec_b).sum(-1) * inv_n2
    sx = u * half_h + half_w
    sy = v * half_h + half_h

    usable = torch.cat((keep, crosses), -1) & torch.isfinite(sx) & torch.isfinite(sy)
    big = torch.full_like(sx, torch.inf)
    xmin = torch.where(usable, sx, big).amin(-1)
    xmax = torch.where(usable, sx, -big).amax(-1)
    ymin = torch.where(usable, sy, big).amin(-1)
    ymax = torch.where(usable, sy, -big).amax(-1)

    # The discarded sliver is only reachable on screen from within a ball about
    # the camera origin of radius clip_depth * (1 + screen_radius / big_d):
    # anything further off the view axis than that projects past the screen
    # edge.  Test the primitive's world AABB, padded by that radius, against the
    # camera origin -- cheap, and a superset of the primitive itself.
    screen_radius = torch.sqrt(
        (pbx.norm(dim=-1) * (float(half_w) / float(half_h))) ** 2
        + pby.norm(dim=-1) ** 2
    )
    pad = clip_depth * (1.0 + screen_radius / big_d.abs().clamp_min(1e-30))[:, None]
    pad = pad.unsqueeze(-1)
    near_camera = (
        (ro_b.squeeze(-2) >= verts.amin(-2) - pad)
        & (ro_b.squeeze(-2) <= verts.amax(-2) + pad)
    ).all(-1)
    bounded = keep.any(-1) & ~near_camera
    return xmin, xmax, ymin, ymax, bounded


_AABB_SEL = None


def _aabb_corners(lo, hi):
    """``[..., 3]`` box bounds to ``[..., 8, 3]`` corners (any batch shape)."""
    global _AABB_SEL
    if _AABB_SEL is None or _AABB_SEL.device != lo.device:
        _AABB_SEL = torch.tensor(
            [[cx, cy, cz] for cx in (0, 1) for cy in (0, 1) for cz in (0, 1)],
            dtype=torch.bool,
            device=lo.device,
        )
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


def _frame_one(value, f):
    """One frame of a possibly time-deduplicated input, keeping the axis."""
    return value[f % value.shape[0]].unsqueeze(0)


def _frame_pairs(
    merged,
    tri_screen,
    f,
    width,
    row_lo,
    row_hi,
    cam_origin,
    screen_point,
    pixel_basis_x,
    pixel_basis_y,
    half_w,
    half_h,
    device,
):
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
    tri_pos = merged["tri_pos"]
    clip = None
    if _straddle_clip_wanted((screen[:, 9] > -0.5) & ~front[:, 0]):
        clip = [
            bound[0]
            for bound in _clipped_screen_extents(
                _frame_one(tri_pos, f).view(1, -1, 3, 3),
                _TRI_EDGES,
                _frame_one(cam_origin, f),
                _frame_one(screen_point, f),
                _frame_one(pixel_basis_x, f),
                _frame_one(pixel_basis_y, f),
                half_w,
                half_h,
            )
        ]
    x0, x1, y0, y1, reach = _screen_bbox(px, py, front, width, row_lo, row_hi, clip)
    # Flag -1 marks a triangle with every vertex behind the camera plane --
    # provably unhittable by a primary ray, so drop it instead of emitting
    # full-window candidate pairs (see precompute_triangle_projection).
    reach = reach & (screen[:, 9] > -0.5)
    return (
        _class_pairs(opaque & reach, x0, x1, y0, y1, f, device),
        _class_pairs(valid & ~opaque & reach, x0, x1, y0, y1, f, device),
    )


def _frame_bez_pairs(
    merged,
    f,
    width,
    row_lo,
    row_hi,
    cam_origin,
    screen_point,
    pixel_basis_x,
    pixel_basis_y,
    half_w,
    half_h,
    device,
):
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
        half_w,
        half_h,
    )
    clip = None
    if _straddle_clip_wanted(front.any(-1) & ~front.all(-1)):
        clip = [
            bound[0]
            for bound in _clipped_screen_extents(
                corners.unsqueeze(0),
                _BOX_EDGES,
                _frame_one(cam_origin, f),
                _frame_one(screen_point, f),
                _frame_one(pixel_basis_x, f),
                _frame_one(pixel_basis_y, f),
                half_w,
                half_h,
            )
        ]
    x0, x1, y0, y1, reach = _screen_bbox(px, py, front, width, row_lo, row_hi, clip)
    # An AABB with no corner in front of the camera plane cannot contain a
    # primary-ray hit (the box is convex, forward ray points project > 0);
    # cull it instead of emitting full-window candidate pairs.
    reach = reach & front.any(-1)
    return (
        _class_pairs(opaque & reach, x0, x1, y0, y1, f, device),
        _class_pairs(valid & ~opaque & reach, x0, x1, y0, y1, f, device),
    )


def precompute_circuit_screen_bounds(
    merged,
    cam_origin,
    screen_point,
    pixel_basis_x,
    pixel_basis_y,
    half_w,
    half_h,
    width,
    memory,
):
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
        int(lo_all.shape[0]),
        int(hi_all.shape[0]),
        int(valid_all.shape[0]),
        int(opaque_all.shape[0]),
        int(cam_origin.shape[0]),
        int(screen_point.shape[0]),
        int(pixel_basis_x.shape[0]),
        int(pixel_basis_y.shape[0]),
    )
    device = lo_all.device
    frame_ids = torch.arange(frames, device=device)
    lo = lo_all.index_select(0, frame_ids % lo_all.shape[0])
    hi = hi_all.index_select(0, frame_ids % hi_all.shape[0])
    corners = _aabb_corners(lo, hi)  # [F, C, 8, 3]
    ro = cam_origin.index_select(0, frame_ids % cam_origin.shape[0])
    sp = screen_point.index_select(0, frame_ids % screen_point.shape[0])
    pbx = pixel_basis_x.index_select(0, frame_ids % pixel_basis_x.shape[0])
    pby = pixel_basis_y.index_select(0, frame_ids % pixel_basis_y.shape[0])

    # _project_points, batched over the leading frame dimension.
    nvec = torch.linalg.cross(pbx, pby)  # [F, 3]
    d = corners - ro[:, None, None, :]
    wpn = (d * nvec[:, None, None, :]).sum(-1)  # [F, C, 8]
    big_d = ((sp - ro) * nvec).sum(-1)  # [F]
    safe = torch.where(wpn.abs() < 1e-12, torch.ones_like(wpn), wpn)
    td = big_d[:, None, None] / safe
    front = (wpn.abs() >= 1e-12) & (td > 0)
    hit = ro[:, None, None, :] + td.unsqueeze(-1) * d
    rel = hit - sp[:, None, None, :]
    dsq = (nvec * nvec).sum(-1).clamp_min(1e-30)  # [F]
    u = (
        torch.linalg.cross(rel, pby[:, None, None, :].expand_as(rel))
        * nvec[:, None, None, :]
    ).sum(-1) / dsq[:, None, None]
    v = (
        torch.linalg.cross(pbx[:, None, None, :].expand_as(rel), rel)
        * nvec[:, None, None, :]
    ).sum(-1) / dsq[:, None, None]
    px = u * half_h + half_w
    py = v * half_h + half_h

    # _screen_bbox's tile-independent parts (x is never row-band clamped).
    all_front = front.all(-1)  # [F, C]
    front_any = front.any(-1)
    xmin = px.amin(-1)
    xmax = px.amax(-1)
    ymin = py.amin(-1)
    ymax = py.amax(-1)
    bounded = all_front
    if _straddle_clip_wanted(front_any & ~all_front):
        clip_x0, clip_x1, clip_y0, clip_y1, clip_ok = _clipped_screen_extents(
            corners, _BOX_EDGES, ro, sp, pbx, pby, half_w, half_h
        )
        # The behind-cull outranks the clip: a box with no corner in front
        # stays dropped rather than becoming an emittable bounded candidate.
        bounded = all_front | (clip_ok & front_any)
        xmin = torch.where(all_front, xmin, clip_x0)
        xmax = torch.where(all_front, xmax, clip_x1)
        ymin = torch.where(all_front, ymin, clip_y0)
        ymax = torch.where(all_front, ymax, clip_y1)
    fx0 = (xmin - 1.0).floor().clamp_(0, width - 1).long()
    fx1 = (xmax + 1.0).ceil().clamp_(0, width - 1).long()
    x0 = torch.where(bounded, fx0, torch.zeros_like(fx0))
    x1 = torch.where(bounded, fx1, torch.full_like(fx1, width - 1))
    x_on = (xmax >= -1.0) & (xmin <= width + 1.0)
    valid = valid_all.index_select(0, frame_ids % valid_all.shape[0]).bool()
    opaque = valid & opaque_all.index_select(0, frame_ids % opaque_all.shape[0]).bool()

    ncirc = int(lo.shape[1])
    memory.note_scope_params(bez_bounds_cells=frames * ncirc)
    pre_f = memory.get_tensor((frames, ncirc, 4), torch.float32)
    pre_f.copy_(
        torch.stack(((ymin - 1.0).floor(), (ymax + 1.0).ceil(), ymin, ymax), -1)
    )
    pre_x = memory.get_tensor((frames, ncirc, 2), torch.int64)
    pre_x.copy_(torch.stack((x0, x1), -1))
    # all_front implies front_any (eight corners), so the bounded reach base
    # omits the redundant ``& front_any``: a clipped straddler kept a front
    # corner by construction.
    pre_m = memory.get_tensor((frames, ncirc, 5), torch.bool)
    pre_m.copy_(
        torch.stack(
            (bounded, bounded & x_on, ~bounded & front_any, opaque, valid & ~opaque), -1
        )
    )
    return pre_f, pre_x, pre_m, _class_any_flags(pre_m)


def precompute_triangle_screen_bounds(
    merged,
    tri_screen,
    cam_origin,
    screen_point,
    pixel_basis_x,
    pixel_basis_y,
    half_w,
    half_h,
    width,
    memory,
):
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

    Straddlers get the camera-plane clip of :func:`_clipped_screen_extents`
    rather than the whole window; the camera arguments are needed for that and
    for nothing else.
    """
    valid_all = merged["tri_frame_valid"]
    opaque_all = merged["tri_frame_opaque"]
    unc_all = merged["tri_alpha_uncertain"]
    tri_pos = merged["tri_pos"]
    frames = max(
        int(tri_screen.shape[0]),
        int(valid_all.shape[0]),
        int(opaque_all.shape[0]),
        int(unc_all.shape[0]),
    )
    device = tri_screen.device
    frame_ids = torch.arange(frames, device=device)
    screen = tri_screen.index_select(0, frame_ids % tri_screen.shape[0])
    px = screen[..., 0:3]
    py = screen[..., 3:6]
    flag = screen[..., 9]
    all_front = flag > 0.5
    not_behind = flag > -0.5

    ntri = int(screen.shape[1])
    xmin = px.amin(-1)
    xmax = px.amax(-1)
    ymin = py.amin(-1)
    ymax = py.amax(-1)
    bounded = all_front
    if _straddle_clip_wanted(not_behind & ~all_front):
        clip_x0, clip_x1, clip_y0, clip_y1, clip_ok = _clipped_screen_extents(
            tri_pos.index_select(0, frame_ids % tri_pos.shape[0]).view(
                frames, ntri, 3, 3
            ),
            _TRI_EDGES,
            cam_origin.index_select(0, frame_ids % cam_origin.shape[0]),
            screen_point.index_select(0, frame_ids % screen_point.shape[0]),
            pixel_basis_x.index_select(0, frame_ids % pixel_basis_x.shape[0]),
            pixel_basis_y.index_select(0, frame_ids % pixel_basis_y.shape[0]),
            half_w,
            half_h,
        )
        # The behind-cull outranks the clip: a provably-behind triangle stays
        # dropped rather than becoming an emittable bounded candidate.
        bounded = all_front | (clip_ok & not_behind)
        xmin = torch.where(all_front, xmin, clip_x0)
        xmax = torch.where(all_front, xmax, clip_x1)
        ymin = torch.where(all_front, ymin, clip_y0)
        ymax = torch.where(all_front, ymax, clip_y1)
    fx0 = (xmin - 1.0).floor().clamp_(0, width - 1).long()
    fx1 = (xmax + 1.0).ceil().clamp_(0, width - 1).long()
    x0 = torch.where(bounded, fx0, torch.zeros_like(fx0))
    x1 = torch.where(bounded, fx1, torch.full_like(fx1, width - 1))
    x_on = (xmax >= -1.0) & (xmin <= width + 1.0)
    valid = valid_all.index_select(0, frame_ids % valid_all.shape[0]).bool()
    unc = unc_all.index_select(0, frame_ids % unc_all.shape[0]).bool()
    opaque = (
        valid
        & opaque_all.index_select(0, frame_ids % opaque_all.shape[0]).bool()
        & ~unc
    )

    memory.note_scope_params(tri_bounds_cells=frames * ntri)
    pre_f = memory.get_tensor((frames, ntri, 4), torch.float32)
    pre_f.copy_(
        torch.stack(((ymin - 1.0).floor(), (ymax + 1.0).ceil(), ymin, ymax), -1)
    )
    pre_x = memory.get_tensor((frames, ntri, 2), torch.int64)
    pre_x.copy_(torch.stack((x0, x1), -1))
    # ``bounded`` already implies not-behind, so its reach base omits the
    # redundant ``& not_behind``.
    pre_m = memory.get_tensor((frames, ntri, 5), torch.bool)
    pre_m.copy_(
        torch.stack(
            (bounded, bounded & x_on, ~bounded & not_behind, opaque, valid & ~opaque),
            -1,
        )
    )
    return pre_f, pre_x, pre_m, _class_any_flags(pre_m)


def _class_any_flags(pre_m):
    """Host per-frame (opaque, translucent) candidate-existence flags.

    Conservative superset of every tile's per-class emission mask: the
    per-tile reach ``(m1 & y_on) | m2`` is contained in ``m1 | m2``, so a
    frame whose flag is False provably yields an empty mask for every tile
    and ``_window_pairs`` can skip its tensor work -- most importantly the
    synchronizing ``.nonzero()`` in ``_class_pairs_flat`` -- outright.  One
    host transfer per window (RASTER_PAIR_FLAGS kill-switch), instead of
    per-tile syncs.
    """
    if not rt_settings.RASTER_PAIR_FLAGS:
        return None
    reach_base = pre_m[..., 1] | pre_m[..., 2]
    return (
        torch.stack(
            ((pre_m[..., 3] & reach_base).any(1), (pre_m[..., 4] & reach_base).any(1)),
            -1,
        )
        .cpu()
        .tolist()
    )


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
    rep = torch.repeat_interleave(torch.arange(idx.numel(), device=device), nch)
    if rep.numel() == 0:
        return None
    base = torch.cumsum(nch, 0) - nch
    off = (torch.arange(rep.shape[0], device=device) - base[rep]) * RASTER_CHUNK
    rows = torch.stack(
        [
            (idx % ncirc)[rep],
            f_abs.index_select(0, idx // ncirc)[rep],
            bx0[rep],
            by0[rep],
            bw[rep],
            bh[rep],
            off,
            torch.zeros_like(rep),
        ],
        -1,
    )
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
    pre_f, pre_x, pre_m, cls_any = bounds
    f0_rel = g0 // ppf
    f1_rel = (g1 - 1) // ppf
    need_op = need_tr = True
    if cls_any is not None:
        # Host flags (``_class_any_flags``): skip a class -- or the whole
        # call -- when none of the tile's covered frames has any candidate
        # of it. Exact: the skipped ``_class_pairs_flat`` would have found
        # an all-false mask and returned None.
        nf = len(cls_any)
        need_op = need_tr = False
        for fr in range(f0_rel, f1_rel + 1):
            row = cls_any[(fr + time_start) % nf]
            need_op = need_op or row[0]
            need_tr = need_tr or row[1]
        if not (need_op or need_tr):
            return None, None
    f_rel = torch.arange(f0_rel, f1_rel + 1, device=device)
    f_abs = f_rel + time_start
    lo_p = (g0 - f_rel * ppf).clamp_(min=0)
    hi_p = (g1 - f_rel * ppf).clamp_(max=ppf)
    row_lo = (lo_p // width).view(-1, 1)  # [Ft, 1] i64
    row_hi = ((hi_p - 1) // width).view(-1, 1)
    rows = f_abs % pre_f.shape[0]
    fy = pre_f.index_select(0, rows)  # [Ft, C, 4]
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
        _class_pairs_flat(m[..., 3] & reach, x0, x1, y0, y1, f_abs, device)
        if need_op
        else None,
        _class_pairs_flat(m[..., 4] & reach, x0, x1, y0, y1, f_abs, device)
        if need_tr
        else None,
    )


def _arena_tensor(memory, shape, dtype, fill=None, *, persist=False):
    out = memory.get_tensor(shape, dtype, persist=persist)
    if fill is not None:
        out.fill_(fill)
    return out


def _check_circuit_ref_capacity(merged):
    """Guard the circuit id room left in a packed fragment ref.

    ``_pack_bez_ref`` shifts the circuit index up by ``_BEZ_BORDER_BITS`` to make
    room for the border/fill blend weight, into a signed 32-bit lane. Overflow
    would silently mis-address a circuit rather than fail, and the ceiling
    (~8.4M circuits in one batch) is far beyond what fits in memory -- but a
    silent wrap is worth one host-side comparison per batch.
    """
    meta = merged.get("circuit_meta", None)
    if meta is None:
        return
    num_circuits = int(meta.shape[1])
    limit = (1 << (31 - _BEZ_BORDER_BITS)) - 1
    if num_circuits > limit:
        raise OverflowError(
            f"{num_circuits} bezier circuits in one render batch exceeds the "
            f"{limit} a packed fragment ref can address; reduce the frame "
            f"batch size (SETTINGS.computing) or split the scene."
        )


def _exact_fragment_order(frag_key, frag_ref, layer_offset_triangles):
    """Return one gather order matching classic depth-bin/layer semantics."""
    is_bez = frag_ref < 0
    bez_code = (-frag_ref - 1).clamp_min(0)
    # Mirrors ``raster_taichi._decode_bez_ref``: the low bits carry the
    # fragment's border/fill blend weight, not part of the circuit id.
    bez_layer = bez_code >> _BEZ_BORDER_BITS
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


def prepare_sparse_raster_coverage(
    merged,
    tri_screen,
    tri_bounds,
    bez_bounds,
    memory,
    cam_origin,
    screen_point,
    pixel_basis_x,
    pixel_basis_y,
    pixel_world_scale,
    col_row_arr,
    time_start,
    time_end,
    width,
    height,
    half_w,
    half_h,
    layer_offset_triangles,
):
    """Emit one exact, ordered primary-hit stream for the whole frame window.

    Unlike :func:`raster_iteration_zero`, this never allocates a tile-pixel
    z-buffer, CSR table, coverage mask, or ray state.  Candidate bboxes launch
    exact intersection COUNT/WRITE passes; the resulting hit records are
    ordered in sparse hit space, then truncated after each pixel's first
    proven-opaque hit.  The persistent result is allocated from the arena's
    reverse pointer so forward coverage-sized wavefront state can coexist with
    it and be reset independently.

    Returns ``None`` when no exact pixel is covered, otherwise a dict containing
    compact ``frag_*``, ``covered_idx``, and ``run_offsets`` arrays.

    Under analytic circuit coverage the per-fragment ``frag_cov`` lane carries
    the fraction of the pixel square each circuit fragment covers, and the
    opaque truncation below only fires on a FULLY covering opaque hit -- a
    silhouette pixel of an opaque shape still has to blend with what is behind
    it.  The effective toggle is captured once here and handed to the resolve in
    the returned dict, so the emission and the shading passes can never compile
    for different modes.
    """
    device = merged["tri_pos"].device
    ppf = int(width) * int(height)
    g0 = 0
    g1 = (int(time_end) - int(time_start)) * ppf
    has_tri = int(merged.get("num_triangles", 0)) > 0
    has_bez = int(merged.get("num_circuits", 0)) > 0

    tri_opaque, tri_trans, bez_opaque, bez_trans = [], [], [], []
    use_tri_pre = has_tri and tri_bounds is not None
    use_bez_pre = has_bez and bez_bounds is not None
    if (has_tri and not use_tri_pre) or (has_bez and not use_bez_pre):
        for f_rel in range(g0 // ppf, (g1 - 1) // ppf + 1):
            f = int(time_start) + f_rel
            lo_p = max(g0 - f_rel * ppf, 0)
            hi_p = min(g1 - f_rel * ppf, ppf)
            row_lo = lo_p // width
            row_hi = (hi_p - 1) // width
            if has_tri and not use_tri_pre:
                po, pt = _frame_pairs(
                    merged,
                    tri_screen,
                    f,
                    width,
                    row_lo,
                    row_hi,
                    cam_origin,
                    screen_point,
                    pixel_basis_x,
                    pixel_basis_y,
                    half_w,
                    half_h,
                    device,
                )
                if po is not None:
                    tri_opaque.append(po)
                if pt is not None:
                    tri_trans.append(pt)
            if has_bez and not use_bez_pre:
                po, pt = _frame_bez_pairs(
                    merged,
                    f,
                    width,
                    row_lo,
                    row_hi,
                    cam_origin,
                    screen_point,
                    pixel_basis_x,
                    pixel_basis_y,
                    half_w,
                    half_h,
                    device,
                )
                if po is not None:
                    bez_opaque.append(po)
                if pt is not None:
                    bez_trans.append(pt)
    if use_tri_pre:
        po, pt = _window_pairs(
            tri_bounds, int(time_start), g0, g1, ppf, int(width), device
        )
        if po is not None:
            tri_opaque.append(po)
        if pt is not None:
            tri_trans.append(pt)
    if use_bez_pre:
        po, pt = _window_pairs(
            bez_bounds, int(time_start), g0, g1, ppf, int(width), device
        )
        if po is not None:
            bez_opaque.append(po)
        if pt is not None:
            bez_trans.append(pt)

    def _cat(parts):
        return torch.cat(parts, 0) if len(parts) > 1 else (parts[0] if parts else None)

    po_t, pt, po_b, pb = map(_cat, (tri_opaque, tri_trans, bez_opaque, bez_trans))
    specs = [
        ("bez", po_b, True),
        ("bez", pb, False),
        ("tri", po_t, True),
        ("tri", pt, False),
    ]
    specs = [s for s in specs if s[1] is not None]
    if not specs:
        return None

    ss = 1 if rt_settings.RASTER_SS else 0
    aa_bez = rt_settings.analytic_aa_bez_mode()
    aa_hw = float(rt_settings.ANALYTIC_AA_BEZ_MIN_HALF_WIDTH)
    # Triangle coverage needs the edge-length columns the projection table only
    # carries when the host built it wide. Gate on the width actually produced,
    # never on the live toggle: a flip between the per-batch precompute and here
    # would otherwise read columns that do not exist.
    aa_tri = (
        1 if (rt_settings.analytic_aa_tri_active() and tri_screen.shape[2] >= 13) else 0
    )
    # 3/4 select the RUN-CORRECTED representation under rule A (clamp) /
    # B (redistribute) for corr > 1 (DESIGN_analytic_aa_v2.md ss4.4; see
    # _tri_run_mode). Value 2 belonged to the deleted cells accounting.
    if aa_tri and rt_settings.ANALYTIC_AA_RUN:
        aa_tri = 4 if rt_settings.ANALYTIC_AA_RUN_RULE == "redistribute" else 3
    # The sample-less-triangle policy rides along in the value the GEOMETRY
    # kernels see, so each policy compiles (and caches) its own _ss_pixel. The
    # resolve and the shadow-event build keep the plain mode value: the policy
    # reaches them as a per-fragment mask bit, so they compile once per mode.
    # Sliver policy AND the representation ride in the geometry kernels'
    # template value (1 + sliver + 4 * repr), so every combination gets its
    # own compiled variant and its own offline-cache entry (see _sliver_mode /
    # _tri_repr). The sliver knob is INERT under run mode (v2 ss4.1): pin it
    # to drop so it cannot fork pointless cache entries.
    aa_tri_ss = 0
    if aa_tri:
        aa_tri_ss = 1 + (
            2 if aa_tri >= 3 else rt_settings.analytic_aa_sliver_mode()
        ) + 4 * min(aa_tri - 1, 2)
    aa_grp = 1 if ((aa_bez or aa_tri) and rt_settings.ANALYTIC_AA_SEAM) else 0
    tri_pos = merged["tri_pos"]
    cam_args = (cam_origin, screen_point, pixel_basis_x, pixel_basis_y)
    geo_args = (
        int(time_start),
        int(width),
        int(height),
        float(half_w),
        float(half_h),
        0,
        int(g1),
    )
    tri_color_args = (
        merged["tri_colors"],
        col_row_arr,
        merged["tri_uvs"],
        merged["tri_tex_meta"],
        merged["textures"],
        int(merged["num_colored_triangles"]),
    )
    bez_geom = (
        pixel_world_scale,
        merged["circuit_meta"],
        merged["circuit_colors"],
        merged["circuit_border_colors"],
        merged["edges_2d"],
        merged["edge_accel"],
    )
    _check_circuit_ref_capacity(merged)

    # All forward allocations in this scope are discovery scratch.  Only the
    # final compact arrays use persist=True (reverse arena) and survive.
    #
    # Value-dependent: every buffer here is sized from the fragment count the
    # COUNT kernel produces, so calibration measures the exact bytes-per-
    # fragment/per-covered-pixel here and learns the density separately.
    with memory.scope("sparse_discovery"), memory.temp():
        dummy_z = _arena_tensor(memory, (1,), torch.int64, Z_SENTINEL)
        # Candidate (primitive, tile) pair count: a value-dependent driver of
        # the per-pair count arrays, independent of the fragment count.
        memory.note_scope_params(
            num_pairs=sum(int(pairs.shape[0]) for _kind, pairs, _op in specs)
        )
        count_parts = []
        for kind, pairs, opaque in specs:
            counts = _arena_tensor(memory, (pairs.shape[0],), torch.int32, 0)
            # Per-pair acceptance bits (bit j = chunk pixel j survived): the
            # write pass replays these instead of recomputing the acceptance
            # chain (see raster_tri_count).
            accepts = _arena_tensor(memory, (pairs.shape[0],), torch.int32, 0)
            if kind == "bez":
                raster_bez_count(
                    pairs,
                    int(pairs.shape[0]),
                    *cam_args,
                    *bez_geom,
                    *geo_args,
                    0,
                    dummy_z,
                    aa_bez,
                    aa_hw,
                    0,
                    counts,
                    accepts,
                )
            else:
                raster_tri_count(
                    pairs,
                    int(pairs.shape[0]),
                    tri_pos,
                    tri_screen,
                    *tri_color_args,
                    *cam_args,
                    *geo_args,
                    ss,
                    float(layer_offset_triangles),
                    0,
                    dummy_z,
                    aa_tri_ss,
                    0,
                    counts,
                    accepts,
                )
            count_parts.append((kind, pairs, opaque, counts, accepts))

        counts_all = (
            torch.cat([s[3] for s in count_parts], 0)
            if len(count_parts) > 1
            else count_parts[0][3]
        )
        counts64 = counts_all.to(torch.int64)
        prefix = torch.cumsum(counts64, 0) - counts64
        num_frags = int(counts64.sum().item())
        if num_frags == 0:
            return None
        # Pre-truncation emitted-fragment count: this sizes the discovery
        # scratch (frag_*_u) and, for an opaque-heavy pixel, exceeds the final
        # kept count. The arena footprint recorded before the return reserves
        # for it so later chunks are sized to fit the discovery peak.
        discovery_frags = num_frags

        frag_key_u = _arena_tensor(memory, (num_frags,), torch.int64)
        frag_ref_u = _arena_tensor(memory, (num_frags,), torch.int32)
        frag_ab_u = _arena_tensor(memory, (num_frags, 2), torch.float32)
        # Coverage lane + sub-pixel sample mask, pre-filled full-coverage /
        # all-samples. Only triangles under analytic AA write the mask;
        # circuits never group (unique key) so an all-samples mask leaves their
        # coverage untouched, and geometry without coverage is unaffected.
        frag_cov_u = _arena_tensor(memory, (num_frags,), torch.float32, 1.0)
        frag_msk_u = _arena_tensor(memory, (num_frags,), torch.int32, AA_MASK_ALL)
        opaque_u = _arena_tensor(memory, (num_frags,), torch.bool, False)
        pair_cursor = 0
        frag_cursor = 0
        for kind, pairs, opaque, counts, accepts in count_parts:
            npairs = int(pairs.shape[0])
            offsets = prefix[pair_cursor : pair_cursor + npairs].to(torch.int32)
            n_spec = int(counts.to(torch.int64).sum().item())
            if kind == "bez":
                raster_bez_write(
                    pairs,
                    npairs,
                    offsets,
                    *cam_args,
                    *bez_geom,
                    *geo_args,
                    0,
                    dummy_z,
                    aa_bez,
                    aa_hw,
                    0,
                    frag_key_u,
                    frag_ref_u,
                    frag_ab_u,
                    frag_cov_u,
                    accepts,
                )
            else:
                raster_tri_write(
                    pairs,
                    npairs,
                    offsets,
                    tri_pos,
                    tri_screen,
                    *tri_color_args,
                    *cam_args,
                    *geo_args,
                    ss,
                    float(layer_offset_triangles),
                    0,
                    dummy_z,
                    aa_tri_ss,
                    0,
                    frag_key_u,
                    frag_ref_u,
                    frag_ab_u,
                    frag_cov_u,
                    frag_msk_u,
                    accepts,
                )
            if opaque and n_spec:
                opaque_u[frag_cursor : frag_cursor + n_spec].fill_(True)
            pair_cursor += npairs
            frag_cursor += n_spec

        order = _exact_fragment_order(frag_key_u, frag_ref_u, layer_offset_triangles)
        key_s = frag_key_u.index_select(0, order)
        ref_s = frag_ref_u.index_select(0, order)
        ab_s = frag_ab_u.index_select(0, order)
        cov_s = frag_cov_u.index_select(0, order)
        msk_s = frag_msk_u.index_select(0, order)
        opaque_s = opaque_u.index_select(0, order)
        if aa_bez or aa_tri:
            # A partially covering opaque hit does not hide what is behind it,
            # so it must not terminate its pixel's run: it stays an ordinary
            # alpha fragment (its alpha already carries the coverage). Under
            # the run representation the test is the SAMPLED claim, matching
            # the prepass and the resolve's magnitude (v2 ss4.1): a full-mask
            # fragment occludes every sample whatever its exact area says.
            if aa_tri >= 3:
                full_s = (msk_s & AA_MASK_ALL) == AA_MASK_ALL
                opaque_s = opaque_s & torch.where(
                    ref_s >= 0, full_s, cov_s >= AA_FULL_COVERAGE
                )
            else:
                opaque_s = opaque_s & (cov_s >= AA_FULL_COVERAGE)
        pix_s = key_s >> 32
        covered, counts = torch.unique_consecutive(pix_s, return_counts=True)

        # The dense z-buffer retained only the nearest opaque hit and discarded
        # every transparent/opaque record behind it.  Reproduce that relation
        # in sparse sorted space: each pixel keeps the prefix through its first
        # opaque event.  The sort uses the exact same depth-bin/layer keys.
        if bool(opaque_s.any().item()):
            num_cov = int(covered.numel())
            positions = torch.arange(num_frags, dtype=torch.int64, device=device)
            segments = torch.repeat_interleave(
                torch.arange(num_cov, dtype=torch.int64, device=device), counts
            )
            starts = torch.cumsum(counts, 0) - counts
            ends = starts + counts - 1
            first_opaque = torch.full(
                (num_cov,), num_frags, dtype=torch.int64, device=device
            )
            opaque_pos = opaque_s.nonzero(as_tuple=True)[0]
            first_opaque.scatter_reduce_(
                0,
                segments.index_select(0, opaque_pos),
                opaque_pos,
                reduce="amin",
                include_self=True,
            )
            keep_end = torch.minimum(first_opaque, ends)
            keep = positions <= keep_end.index_select(0, segments)
            if int(keep.sum().item()) != num_frags:
                keep_idx = keep.nonzero(as_tuple=True)[0]
                key_s = key_s.index_select(0, keep_idx)
                ref_s = ref_s.index_select(0, keep_idx)
                ab_s = ab_s.index_select(0, keep_idx)
                cov_s = cov_s.index_select(0, keep_idx)
                msk_s = msk_s.index_select(0, keep_idx)
                pix_s = key_s >> 32
                covered, counts = torch.unique_consecutive(pix_s, return_counts=True)
                num_frags = int(key_s.shape[0])

        num_covered = int(covered.numel())
        frag_key = _arena_tensor(memory, (num_frags,), torch.int64, persist=True)
        frag_ref = _arena_tensor(memory, (num_frags,), torch.int32, persist=True)
        frag_ab = _arena_tensor(memory, (num_frags, 2), torch.float32, persist=True)
        frag_cov = _arena_tensor(memory, (num_frags,), torch.float32, persist=True)
        frag_msk = _arena_tensor(memory, (num_frags,), torch.int32, persist=True)
        covered_idx = _arena_tensor(memory, (num_covered,), torch.int32, persist=True)
        run_offsets = _arena_tensor(
            memory, (num_covered + 1,), torch.int32, 0, persist=True
        )
        frag_key.copy_(key_s)
        frag_ref.copy_(ref_s)
        frag_ab.copy_(ab_s)
        frag_cov.copy_(cov_s)
        frag_msk.copy_(msk_s)
        covered_idx.copy_(covered.to(torch.int32))
        run_offsets[1:].copy_(torch.cumsum(counts.to(torch.int32), 0))

        # Recorded for calibration: the fragment/covered counts are this
        # scope's value-dependent drivers and are only known once the COUNT
        # kernel has run, so they are attached from inside the scope.
        memory.note_scope_params(
            frames=int(time_end) - int(time_start),
            discovery_frags=int(discovery_frags),
            num_fragments=int(num_frags),
            num_covered=int(num_covered),
            pixels=int(width) * int(height),
        )

    # Reserve for the next chunk's discovery peak: the pre-truncation scratch
    # (frag_*_u: 8+4+8+4+4+1 B/frag) plus the persistent compact result
    # (frag_key/ref/ab/cov/msk: 8+4+8+4+4 B/frag; covered_idx/run_offsets:
    # 4+4 B/covered) coexist in the arena at the copy. Amortized per output
    # frame so the render-chunk preflight sizes later chunks to fit it instead
    # of over-committing.
    discovery_bytes = discovery_frags * 29 + num_frags * 28 + num_covered * 8
    rt_settings.note_sparse_discovery_footprint(
        discovery_bytes, int(time_end) - int(time_start)
    )

    return {
        "frag_key": frag_key,
        "frag_ref": frag_ref,
        "frag_ab": frag_ab,
        "frag_cov": frag_cov,
        "frag_msk": frag_msk,
        "covered_idx": covered_idx,
        "run_offsets": run_offsets,
        "num_fragments": num_frags,
        "num_covered": num_covered,
        # Pinned here so the resolve compiles for the mode the fragments were
        # emitted in, whatever the live toggle says by then.
        "aa_bez": aa_bez,
        "aa_tri": aa_tri,
        "aa_grp": aa_grp,
    }


def shade_sparse_raster_coverage(
    coverage,
    covered_start,
    covered_end,
    merged,
    tri_screen,
    memory,
    cam_origin,
    screen_point,
    pixel_basis_x,
    pixel_basis_y,
    pixel_world_scale,
    layer_offsets,
    gen_meta,
    light_pos,
    light_col,
    num_lights,
    col_row_arr,
    frag_flag,
    frag_pipelines,
    skip_unlit_normal,
    refraction_flag,
    time_start,
    width,
    height,
    half_w,
    half_h,
    state,
    rs_pix,
    pix_accum,
    rs_alloc,
    shadow_flag,
    t_bvh,
    pn_bvh,
    bez_bvh,
    layer_offset_triangles,
    layer_offset_pn,
    max_bounces,
):
    """Resolve one compact covered-pixel slice and seed its continuations."""
    (
        rs_ro,
        rs_rd,
        rs_acc,
        rs_sca,
        rs_int,
        _rs_kt,
        _rs_kl,
        _rs_ka,
        _rs_kb,
        _rs_kp,
        _rs_kf,
    ) = state
    c0, c1 = int(covered_start), int(covered_end)
    num_covered = c1 - c0
    covered_idx = coverage["covered_idx"][c0:c1]
    # Slice boundaries come from a host mirror of run_offsets, copied once
    # per coverage (one queue drain) instead of two synchronizing ``.item()``
    # reads per slice -- a chunk resolves tens of slices, and each drain
    # stalls the prep/render overlap. The mirror stays valid across the
    # OOM-retry loop (coverage is immutable once discovered).
    ro_host = coverage.get("run_offsets_host")
    if ro_host is None:
        ro_host = coverage["run_offsets"].cpu()
        coverage["run_offsets_host"] = ro_host
    event_start = int(ro_host[c0])
    event_end = int(ro_host[c1])
    num_frags = event_end - event_start
    frag_key = coverage["frag_key"][event_start:event_end]
    frag_ref = coverage["frag_ref"][event_start:event_end]
    frag_ab = coverage["frag_ab"][event_start:event_end]
    frag_cov = coverage["frag_cov"][event_start:event_end]
    frag_msk = coverage["frag_msk"][event_start:event_end]
    # Pinned at emission (see prepare_sparse_raster_coverage) so the resolve can
    # never compile for a different mode than the fragments were written in.
    aa_bez = int(coverage.get("aa_bez", 0))
    aa_tri = int(coverage.get("aa_tri", 0))
    aa_grp = int(coverage.get("aa_grp", 0))
    # Continuation-ray supersampling is independent of the fragment lanes -- it
    # only changes how many secondary rays a reflective/refractive hit spawns --
    # so it is read live rather than pinned with them.
    sec_aa = rt_settings.effective_analytic_aa_secondary_samples()
    run_offsets = _arena_tensor(memory, (num_covered + 1,), torch.int32)
    torch.sub(coverage["run_offsets"][c0 : c1 + 1], event_start, out=run_offsets)

    has_tri = 1 if int(merged.get("num_triangles", 0)) > 0 else 0
    has_pn = 1 if int(merged.get("num_pn", 0)) > 0 else 0
    has_bez = 1 if int(merged.get("num_circuits", 0)) > 0 else 0
    ss = 1 if rt_settings.RASTER_SS else 0
    tri_pos = merged["tri_pos"]
    dump_req = _aa_dump_request()
    cam_args = (cam_origin, screen_point, pixel_basis_x, pixel_basis_y)
    # Vestigial by design: the dense path folds the nearest opaque hit into a
    # separate z-winner that raster_first_shade / raster_shadow_event_build
    # re-derive via _terminal_z_hit (has_z == 1). The sparse path instead emits
    # that opaque hit as an ordinary fragment (kept as each pixel's last record
    # by the opaque truncation), so there is no separate z-winner: this buffer
    # stays all-sentinel and both kernels see has_z == 0. It is passed only to
    # satisfy their signatures; do not remove without also dropping the zbuf
    # parameter from those kernels.
    zbuf = _arena_tensor(memory, (num_covered,), torch.int64, Z_SENTINEL)
    frag_shadow_id = _arena_tensor(memory, (max(1, num_frags),), torch.int32, -1)
    z_shadow_id = _arena_tensor(memory, (num_covered,), torch.int32, -1)

    if shadow_flag:
        max_events = max(1, num_frags)
        event_pos = _arena_tensor(memory, (max_events, 3), torch.float32)
        event_snrm = _arena_tensor(memory, (max_events, 3), torch.float32)
        event_fnrm = _arena_tensor(memory, (max_events, 3), torch.float32)
        event_frame = _arena_tensor(memory, (max_events,), torch.int32)
        event_msk = _arena_tensor(memory, (max_events,), torch.int32, 0xF)
        event_count = _arena_tensor(memory, (1,), torch.int32, 0)
        # World-space pixel footprint per event, for sub-pixel shadow sampling.
        # One row when it is off, so the argument always exists.
        event_dp = _arena_tensor(
            memory, (max_events if sec_aa > 1 else 1, 6), torch.float32
        )
        sdump_buf = (
            _aa_dump_buffer(dump_req, zbuf.device)
            if dump_req
            else _aa_dump_arg(zbuf.device)
        )
        raster_shadow_event_build(
            num_covered,
            run_offsets,
            frag_key,
            frag_ref,
            frag_ab,
            frag_cov,
            frag_msk,
            zbuf,
            tri_pos,
            tri_screen,
            merged["tri_norm"],
            merged["tri_extra"],
            merged["tri_colors"],
            merged["tri_uvs"],
            merged["tri_tex_meta"],
            merged["textures"],
            int(merged["num_colored_triangles"]),
            col_row_arr,
            merged["tri_obj"],
            merged["tri_mat_id"],
            merged["circuit_meta"],
            merged["circuit_colors"],
            merged["circuit_border_colors"],
            merged["edges_2d"],
            merged["edge_accel"],
            pixel_world_scale,
            float(layer_offset_triangles),
            int(refraction_flag),
            ss,
            has_bez,
            aa_bez,
            aa_tri,
            aa_grp,
            sec_aa,
            1,
            covered_idx,
            num_covered,
            1,
            int(time_start),
            int(width),
            int(height),
            0,
            *cam_args,
            gen_meta,
            int(max_bounces),
            frag_shadow_id,
            z_shadow_id,
            event_pos,
            event_snrm,
            event_fnrm,
            event_frame,
            event_dp,
            event_msk,
            event_count,
            1 if dump_req else 0,
            sdump_buf,
        )
        if dump_req:
            _aa_dump_emit("shadow", sdump_buf)
        num_events = int(event_count.item())
        shadow_vis = _arena_tensor(
            memory, (max(1, num_events), max(1, int(num_lights))), torch.float32, 1.0
        )
        if num_events:
            from algan.rendering.raytracing.refit_bvh import RefitBVH

            raster_shadow_trace(
                num_events,
                event_pos,
                event_snrm,
                event_fnrm,
                event_frame,
                event_msk,
                t_bvh.blocks,
                t_bvh.node_miss,
                t_bvh.leaf_prim,
                t_bvh.leaf_tspan,
                int(t_bvh.first_leaf),
                merged["tri_pos"],
                merged["tri_colors"],
                merged["tri_uvs"],
                merged["tri_tex_meta"],
                merged["textures"],
                int(merged["num_colored_triangles"]),
                pn_bvh.blocks,
                pn_bvh.node_miss,
                pn_bvh.leaf_prim,
                pn_bvh.leaf_tspan,
                int(pn_bvh.first_leaf),
                merged["pn_ctrl"],
                merged["pn_obb"],
                merged["pn_colors"],
                bez_bvh.blocks,
                bez_bvh.node_miss,
                bez_bvh.leaf_prim,
                bez_bvh.leaf_tspan,
                int(bez_bvh.first_leaf),
                merged["circuit_meta"],
                merged["circuit_colors"],
                merged["circuit_border_colors"],
                merged["edges_2d"],
                merged["edge_accel"],
                light_pos,
                light_col,
                int(num_lights),
                pixel_world_scale,
                float(layer_offset_triangles),
                float(layer_offset_pn),
                1 if isinstance(t_bvh, RefitBVH) else 0,
                has_tri,
                has_pn,
                has_bez,
                event_dp,
                sec_aa,
                shadow_vis,
                int(shadow_flag),
            )
    else:
        shadow_vis = _arena_tensor(memory, (1, 1), torch.float32, 1.0)

    rdump_buf = (
        _aa_dump_buffer(dump_req, zbuf.device)
        if dump_req
        else _aa_dump_arg(zbuf.device)
    )
    raster_first_shade(
        num_covered,
        run_offsets,
        frag_key,
        frag_ref,
        frag_ab,
        frag_cov,
        frag_msk,
        zbuf,
        merged["tri_pos"],
        tri_screen,
        merged["tri_norm"],
        merged["tri_extra"],
        merged["tri_colors"],
        merged["tri_uvs"],
        merged["tri_tex_meta"],
        merged["textures"],
        int(merged["num_colored_triangles"]),
        col_row_arr,
        merged["tri_obj"],
        merged["tri_mat_id"],
        merged["tri_mat"],
        merged["circuit_meta"],
        merged["circuit_colors"],
        merged["circuit_border_colors"],
        pixel_world_scale,
        merged["edges_2d"],
        merged["edge_accel"],
        light_pos,
        light_col,
        int(num_lights),
        layer_offsets,
        int(frag_flag),
        frag_pipelines,
        int(refraction_flag),
        int(skip_unlit_normal),
        ss,
        has_bez,
        aa_bez,
        aa_tri,
        aa_grp,
        sec_aa,
        float(rt_settings.ANALYTIC_AA_SECONDARY_MIN_ENERGY),
        int(rt_settings.glossy_reflection_mode()),
        # Boolean here on purpose: this kernel only ever tests shadows != 0
        # (visibility itself was traced by raster_shadow_trace), so folding
        # the any-hit mode (2/3) to 1 keeps one compiled variant instead of
        # one per mode of the pipeline's biggest kernel.
        1 if shadow_flag else 0,
        0,
        1,
        covered_idx,
        num_covered,
        1,
        int(time_start),
        int(width),
        int(height),
        0,
        *cam_args,
        gen_meta,
        rs_ro,
        rs_rd,
        rs_acc,
        rs_sca,
        rs_int,
        rs_pix,
        pix_accum,
        rs_alloc,
        frag_shadow_id,
        z_shadow_id,
        shadow_vis,
        1 if dump_req else 0,
        rdump_buf,
    )
    if dump_req:
        _aa_dump_emit("resolve", rdump_buf)
    return covered_idx


def raster_iteration_zero(
    merged,
    tri_screen,
    tri_bounds,
    bez_bounds,
    memory,
    cam_origin,
    screen_point,
    pixel_basis_x,
    pixel_basis_y,
    pixel_world_scale,
    layer_offsets,
    gen_meta,
    light_pos,
    light_col,
    num_lights,
    col_row_arr,
    frag_flag,
    frag_pipelines,
    skip_unlit_normal,
    refraction_flag,
    time_start,
    width,
    height,
    half_w,
    half_h,
    tile_start,
    tn_primary,
    state,
    rs_pix,
    pix_accum,
    rs_alloc,
    shadow_flag,
    t_bvh,
    pn_bvh,
    bez_bvh,
    layer_offset_triangles,
    layer_offset_pn,
    max_bounces,
    prefill=0,
    env_active=0,
):
    """Raster, order, resolve, and seed compact primary continuations.

    ``prefill`` (RASTER_EMPTY_SKIP, fixed per batch by the tracer): the host
    pre-filled every primary's ``pix_accum`` row with the retired-empty
    result and pre-marked the pool DONE, so a tile whose z-buffer stays
    all-sentinel and whose fragment stream is empty needs no resolve (nor
    shadow-event) launch at all -- unless an environment map is active
    (``env_active``), in which case every empty pixel still samples it.

    Returns ``True`` when it took that whole-tile empty early-out (so
    ``pix_accum`` is still the untouched retired-empty constant and the
    caller can composite it with the lean ``empty`` kernel variant), else
    ``False``.
    """
    (
        rs_ro,
        rs_rd,
        rs_acc,
        rs_sca,
        rs_int,
        _rs_kt,
        _rs_kl,
        _rs_ka,
        _rs_kb,
        _rs_kp,
        _rs_kf,
    ) = state
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
                    merged,
                    tri_screen,
                    f,
                    width,
                    row_lo,
                    row_hi,
                    cam_origin,
                    screen_point,
                    pixel_basis_x,
                    pixel_basis_y,
                    half_w,
                    half_h,
                    device,
                )
                if po is not None:
                    tri_opaque.append(po)
                if pt is not None:
                    tri_trans.append(pt)
            if has_bez and not use_bez_pre:
                po, pt = _frame_bez_pairs(
                    merged,
                    f,
                    width,
                    row_lo,
                    row_hi,
                    cam_origin,
                    screen_point,
                    pixel_basis_x,
                    pixel_basis_y,
                    half_w,
                    half_h,
                    device,
                )
                if po is not None:
                    bez_opaque.append(po)
                if pt is not None:
                    bez_trans.append(pt)
    if use_tri_pre:
        po, pt = _window_pairs(tri_bounds, time_start, g0, g1, ppf, width, device)
        if po is not None:
            tri_opaque.append(po)
        if pt is not None:
            tri_trans.append(pt)
    if use_bez_pre:
        po, pt = _window_pairs(bez_bounds, time_start, g0, g1, ppf, width, device)
        if po is not None:
            bez_opaque.append(po)
        if pt is not None:
            bez_trans.append(pt)

    def _cat(parts):
        return torch.cat(parts, 0) if len(parts) > 1 else (parts[0] if parts else None)

    po_t, pt, po_b, pb = map(_cat, (tri_opaque, tri_trans, bez_opaque, bez_trans))
    skip_empty = bool(prefill) and not env_active
    if skip_empty and po_t is None and pt is None and po_b is None and pb is None:
        # No candidate pairs anywhere in the tile: the z-buffer would stay
        # all-sentinel and the fragment stream empty, so the pre-filled
        # retired-empty state already IS the resolve's postcondition. Skip
        # every launch, including the shadow-event build (nothing to accept).
        return True, None, 0
    zbuf = _arena_tensor(memory, (tn_primary,), torch.int64, Z_SENTINEL)
    ss = 1 if rt_settings.RASTER_SS else 0
    # Read once per tile so every kernel below compiles for the same mode.
    aa_bez = rt_settings.analytic_aa_bez_mode()
    aa_hw = float(rt_settings.ANALYTIC_AA_BEZ_MIN_HALF_WIDTH)
    # Gate triangle coverage on the projection-table width the host actually
    # built, not the live toggle (see prepare_sparse_raster_coverage).
    aa_tri = (
        1 if (rt_settings.analytic_aa_tri_active() and tri_screen.shape[2] >= 13) else 0
    )
    # 3/4 = run-corrected A/B; as in the sparse path.
    if aa_tri and rt_settings.ANALYTIC_AA_RUN:
        aa_tri = 4 if rt_settings.ANALYTIC_AA_RUN_RULE == "redistribute" else 3
    aa_tri_ss = 0
    if aa_tri:
        aa_tri_ss = 1 + (
            2 if aa_tri >= 3 else rt_settings.analytic_aa_sliver_mode()
        ) + 4 * min(aa_tri - 1, 2)
    aa_grp = 1 if ((aa_bez or aa_tri) and rt_settings.ANALYTIC_AA_SEAM) else 0
    dump_req = _aa_dump_request()
    tri_pos = merged["tri_pos"]
    cam_args = (cam_origin, screen_point, pixel_basis_x, pixel_basis_y)
    geo_args = (
        int(time_start),
        int(width),
        int(height),
        float(half_w),
        float(half_h),
        int(tile_start),
        int(tn_primary),
    )
    tri_color_args = (
        merged["tri_colors"],
        col_row_arr,
        merged["tri_uvs"],
        merged["tri_tex_meta"],
        merged["textures"],
        int(merged["num_colored_triangles"]),
    )
    bez_geom = (
        pixel_world_scale,
        merged["circuit_meta"],
        merged["circuit_colors"],
        merged["circuit_border_colors"],
        merged["edges_2d"],
        merged["edge_accel"],
    )
    if po_t is not None:
        raster_tri_z(
            po_t,
            int(po_t.shape[0]),
            tri_pos,
            tri_screen,
            *cam_args,
            *geo_args,
            ss,
            float(layer_offset_triangles),
            aa_tri_ss,
            zbuf,
        )
    if po_b is not None:
        raster_bez_z(
            po_b,
            int(po_b.shape[0]),
            *cam_args,
            pixel_world_scale,
            merged["circuit_meta"],
            merged["circuit_colors"],
            merged["edges_2d"],
            merged["edge_accel"],
            *geo_args,
            aa_bez,
            aa_hw,
            zbuf,
        )

    # Circuit candidate specs, as (pairs, partial_only). Under analytic
    # coverage the proven-opaque candidates run the transparent pass too: their
    # fully covered pixels already claimed the z-buffer above, and the
    # partially covered silhouette pixels are exactly the ones that must blend
    # rather than be dropped.
    bez_specs = []
    if pb is not None:
        bez_specs.append((pb, 0))
    if aa_bez and po_b is not None:
        bez_specs.append((po_b, 1))

    # Same story for triangles: under analytic coverage the proven-opaque
    # candidates also run the transparent pass, restricted to their partially
    # covered silhouette pixels.
    tri_specs = []
    if pt is not None:
        tri_specs.append((pt, 0))
    if aa_tri and po_t is not None:
        tri_specs.append((po_t, 1))

    bcounts = []
    tcounts = []
    baccepts = []
    taccepts = []
    for bpairs, partial_only in bez_specs:
        counts_b = _arena_tensor(memory, (bpairs.shape[0],), torch.int32, 0)
        # Acceptance bits the write pass replays (see raster_tri_count);
        # valid because nothing touches zbuf between count and write.
        accept_b = _arena_tensor(memory, (bpairs.shape[0],), torch.int32, 0)
        raster_bez_count(
            bpairs,
            int(bpairs.shape[0]),
            *cam_args,
            *bez_geom,
            *geo_args,
            1,
            zbuf,
            aa_bez,
            aa_hw,
            partial_only,
            counts_b,
            accept_b,
        )
        bcounts.append(counts_b)
        baccepts.append(accept_b)
    for tpairs, partial_only in tri_specs:
        counts_t = _arena_tensor(memory, (tpairs.shape[0],), torch.int32, 0)
        accept_t = _arena_tensor(memory, (tpairs.shape[0],), torch.int32, 0)
        raster_tri_count(
            tpairs,
            int(tpairs.shape[0]),
            tri_pos,
            tri_screen,
            *tri_color_args,
            *cam_args,
            *geo_args,
            ss,
            float(layer_offset_triangles),
            1,
            zbuf,
            aa_tri_ss,
            partial_only,
            counts_t,
            accept_t,
        )
        tcounts.append(counts_t)
        taccepts.append(accept_t)

    count_parts = list(bcounts) + list(tcounts)
    num_frags = 0
    bez_offsets = []
    tri_offsets = []
    if count_parts:
        counts = torch.cat(count_parts, 0) if len(count_parts) > 1 else count_parts[0]
        counts64 = counts.to(torch.int64)
        prefix = torch.cumsum(counts64, 0) - counts64
        num_frags = int(counts64.sum().item())
        cursor = 0
        for counts_b in bcounts:
            bez_offsets.append(
                prefix[cursor : cursor + counts_b.shape[0]].to(torch.int32)
            )
            cursor += counts_b.shape[0]
        for counts_t in tcounts:
            tri_offsets.append(
                prefix[cursor : cursor + counts_t.shape[0]].to(torch.int32)
            )
            cursor += counts_t.shape[0]

    if skip_empty and num_frags == 0 and po_t is None and po_b is None:
        # Transparent candidates existed but every fragment was culled and no
        # opaque pair touched the z-buffer: same retired-empty postcondition.
        return True, None, 0

    run_offsets = _arena_tensor(memory, (tn_primary + 1,), torch.int32, 0)
    if num_frags:
        frag_key_u = _arena_tensor(memory, (num_frags,), torch.int64)
        frag_ref_u = _arena_tensor(memory, (num_frags,), torch.int32)
        frag_ab_u = _arena_tensor(memory, (num_frags, 2), torch.float32)
        # Coverage lane + sample mask, pre-filled full (see the sparse path).
        frag_cov_u = _arena_tensor(memory, (num_frags,), torch.float32, 1.0)
        frag_msk_u = _arena_tensor(memory, (num_frags,), torch.int32, AA_MASK_ALL)
        for (bpairs, partial_only), offsets, accept_b in zip(
            bez_specs, bez_offsets, baccepts
        ):
            raster_bez_write(
                bpairs,
                int(bpairs.shape[0]),
                offsets,
                *cam_args,
                *bez_geom,
                *geo_args,
                1,
                zbuf,
                aa_bez,
                aa_hw,
                partial_only,
                frag_key_u,
                frag_ref_u,
                frag_ab_u,
                frag_cov_u,
                accept_b,
            )
        for (tpairs, partial_only), offsets, accept_t in zip(
            tri_specs, tri_offsets, taccepts
        ):
            raster_tri_write(
                tpairs,
                int(tpairs.shape[0]),
                offsets,
                tri_pos,
                tri_screen,
                *tri_color_args,
                *cam_args,
                *geo_args,
                ss,
                float(layer_offset_triangles),
                1,
                zbuf,
                aa_tri_ss,
                partial_only,
                frag_key_u,
                frag_ref_u,
                frag_ab_u,
                frag_cov_u,
                frag_msk_u,
                accept_t,
            )
        order = _exact_fragment_order(frag_key_u, frag_ref_u, layer_offset_triangles)
        frag_key = _arena_tensor(memory, (num_frags,), torch.int64)
        frag_ref = _arena_tensor(memory, (num_frags,), torch.int32)
        frag_ab = _arena_tensor(memory, (num_frags, 2), torch.float32)
        frag_cov = _arena_tensor(memory, (num_frags,), torch.float32)
        frag_msk = _arena_tensor(memory, (num_frags,), torch.int32)
        frag_key.copy_(frag_key_u.index_select(0, order))
        frag_ref.copy_(frag_ref_u.index_select(0, order))
        frag_ab.copy_(frag_ab_u.index_select(0, order))
        frag_cov.copy_(frag_cov_u.index_select(0, order))
        frag_msk.copy_(frag_msk_u.index_select(0, order))
        pix = frag_key >> 32
        counts_per_pixel = torch.bincount(pix, minlength=tn_primary)
        run_offsets[1:].copy_(torch.cumsum(counts_per_pixel.to(torch.int32), 0))
    else:
        frag_key = _arena_tensor(memory, (1,), torch.int64, 0)
        frag_ref = _arena_tensor(memory, (1,), torch.int32, 0)
        frag_ab = _arena_tensor(memory, (1, 2), torch.float32, 0)
        frag_cov = _arena_tensor(memory, (1,), torch.float32, 1.0)
        frag_msk = _arena_tensor(memory, (1,), torch.int32, AA_MASK_ALL)

    # Covered-pixel-compacted resolve (RASTER_COVERED_SHADE). A pixel needs
    # shading iff it has a fragment (nrun > 0) or a z-prepass winner -- exactly
    # the ``total > 0`` condition raster_first_shade's prefill early-out tests.
    # Compacting that per-pixel mask into the ascending nonzero list lets the
    # resolve launch one thread per covered pixel instead of per tile pixel,
    # while the untouched empties keep their retired-empty pre-fill. Requires
    # prefill and no env map (empty pixels would still sample the sky).
    use_covered = bool(prefill) and not env_active and rt_settings.RASTER_COVERED_SHADE
    covered_idx = None
    num_covered = 0
    if use_covered:
        nrun = run_offsets[1:] - run_offsets[:-1]
        covered_mask = (nrun > 0) | (zbuf != Z_SENTINEL)
        covered_idx = covered_mask.nonzero(as_tuple=True)[0].to(torch.int32)
        num_covered = int(covered_idx.numel())
        if num_covered == 0:
            # Candidates existed but produced no fragment and no z-winner
            # (degenerate / behind-camera geometry): the whole tile is still
            # the untouched retired-empty constant, like the earlier skips.
            return True, None, 0
    if covered_idx is None:
        covered_idx = _arena_tensor(memory, (1,), torch.int32, 0)

    # Exact sparse shadow queue. The upper bound is one accepted triangle event
    # per raw fragment plus one terminal visibility winner per pixel; the build
    # kernel only reserves entries that survive seam/transport decisions.
    frag_shadow_id = _arena_tensor(memory, (max(1, num_frags),), torch.int32, -1)
    z_shadow_id = _arena_tensor(memory, (tn_primary,), torch.int32, -1)
    if shadow_flag:
        max_events = max(1, num_frags + tn_primary)
        event_pos = _arena_tensor(memory, (max_events, 3), torch.float32)
        event_snrm = _arena_tensor(memory, (max_events, 3), torch.float32)
        event_fnrm = _arena_tensor(memory, (max_events, 3), torch.float32)
        event_frame = _arena_tensor(memory, (max_events,), torch.int32)
        event_msk = _arena_tensor(memory, (max_events,), torch.int32, 0xF)
        event_count = _arena_tensor(memory, (1,), torch.int32, 0)
        sec_aa = rt_settings.effective_analytic_aa_secondary_samples()
        event_dp = _arena_tensor(
            memory, (max_events if sec_aa > 1 else 1, 6), torch.float32
        )
        sdump_buf = (
            _aa_dump_buffer(dump_req, zbuf.device)
            if dump_req
            else _aa_dump_arg(zbuf.device)
        )
        raster_shadow_event_build(
            int(tn_primary),
            run_offsets,
            frag_key,
            frag_ref,
            frag_ab,
            frag_cov,
            frag_msk,
            zbuf,
            tri_pos,
            tri_screen,
            merged["tri_norm"],
            merged["tri_extra"],
            merged["tri_colors"],
            merged["tri_uvs"],
            merged["tri_tex_meta"],
            merged["textures"],
            int(merged["num_colored_triangles"]),
            col_row_arr,
            merged["tri_obj"],
            merged["tri_mat_id"],
            merged["circuit_meta"],
            merged["circuit_colors"],
            merged["circuit_border_colors"],
            merged["edges_2d"],
            merged["edge_accel"],
            pixel_world_scale,
            float(layer_offset_triangles),
            int(refraction_flag),
            ss,
            has_bez,
            aa_bez,
            aa_tri,
            aa_grp,
            sec_aa,
            1 if use_covered else 0,
            covered_idx,
            int(num_covered),
            0,
            int(time_start),
            int(width),
            int(height),
            int(tile_start),
            *cam_args,
            gen_meta,
            int(max_bounces),
            frag_shadow_id,
            z_shadow_id,
            event_pos,
            event_snrm,
            event_fnrm,
            event_frame,
            event_dp,
            event_msk,
            event_count,
            1 if dump_req else 0,
            sdump_buf,
        )
        if dump_req:
            _aa_dump_emit("shadow", sdump_buf)
        num_events = int(event_count.item())
        shadow_vis = _arena_tensor(
            memory, (max(1, num_events), max(1, int(num_lights))), torch.float32, 1.0
        )
        if num_events:
            from algan.rendering.raytracing.refit_bvh import RefitBVH

            raster_shadow_trace(
                num_events,
                event_pos,
                event_snrm,
                event_fnrm,
                event_frame,
                event_msk,
                t_bvh.blocks,
                t_bvh.node_miss,
                t_bvh.leaf_prim,
                t_bvh.leaf_tspan,
                int(t_bvh.first_leaf),
                merged["tri_pos"],
                merged["tri_colors"],
                merged["tri_uvs"],
                merged["tri_tex_meta"],
                merged["textures"],
                int(merged["num_colored_triangles"]),
                pn_bvh.blocks,
                pn_bvh.node_miss,
                pn_bvh.leaf_prim,
                pn_bvh.leaf_tspan,
                int(pn_bvh.first_leaf),
                merged["pn_ctrl"],
                merged["pn_obb"],
                merged["pn_colors"],
                bez_bvh.blocks,
                bez_bvh.node_miss,
                bez_bvh.leaf_prim,
                bez_bvh.leaf_tspan,
                int(bez_bvh.first_leaf),
                merged["circuit_meta"],
                merged["circuit_colors"],
                merged["circuit_border_colors"],
                merged["edges_2d"],
                merged["edge_accel"],
                light_pos,
                light_col,
                int(num_lights),
                pixel_world_scale,
                float(layer_offset_triangles),
                float(layer_offset_pn),
                1 if isinstance(t_bvh, RefitBVH) else 0,
                has_tri,
                has_pn,
                has_bez,
                event_dp,
                sec_aa,
                shadow_vis,
                int(shadow_flag),
            )
    else:
        shadow_vis = _arena_tensor(memory, (1, 1), torch.float32, 1.0)

    rdump_buf = (
        _aa_dump_buffer(dump_req, zbuf.device)
        if dump_req
        else _aa_dump_arg(zbuf.device)
    )
    raster_first_shade(
        int(tn_primary),
        run_offsets,
        frag_key,
        frag_ref,
        frag_ab,
        frag_cov,
        frag_msk,
        zbuf,
        merged["tri_pos"],
        tri_screen,
        merged["tri_norm"],
        merged["tri_extra"],
        merged["tri_colors"],
        merged["tri_uvs"],
        merged["tri_tex_meta"],
        merged["textures"],
        int(merged["num_colored_triangles"]),
        col_row_arr,
        merged["tri_obj"],
        merged["tri_mat_id"],
        merged["tri_mat"],
        merged["circuit_meta"],
        merged["circuit_colors"],
        merged["circuit_border_colors"],
        pixel_world_scale,
        merged["edges_2d"],
        merged["edge_accel"],
        light_pos,
        light_col,
        int(num_lights),
        layer_offsets,
        int(frag_flag),
        frag_pipelines,
        int(refraction_flag),
        int(skip_unlit_normal),
        ss,
        has_bez,
        aa_bez,
        aa_tri,
        aa_grp,
        rt_settings.effective_analytic_aa_secondary_samples(),
        float(rt_settings.ANALYTIC_AA_SECONDARY_MIN_ENERGY),
        int(rt_settings.glossy_reflection_mode()),
        # Boolean on purpose -- see the sparse-path raster_first_shade call.
        1 if shadow_flag else 0,
        1 if prefill else 0,
        1 if use_covered else 0,
        covered_idx,
        int(num_covered),
        0,
        int(time_start),
        int(width),
        int(height),
        int(tile_start),
        *cam_args,
        gen_meta,
        rs_ro,
        rs_rd,
        rs_acc,
        rs_sca,
        rs_int,
        rs_pix,
        pix_accum,
        rs_alloc,
        frag_shadow_id,
        z_shadow_id,
        shadow_vis,
        1 if dump_req else 0,
        rdump_buf,
    )
    if dump_req:
        _aa_dump_emit("resolve", rdump_buf)
    # The resolve ran and wrote pix_accum, so the composite must read it.
    # Hand the covered list to the caller: under post-process tonemapping the
    # composite is a linear blend that is a no-op on empty pixels, so it can
    # run over exactly these covered pixels (byte-identical, mode 3 only).
    return False, (covered_idx if use_covered else None), num_covered
