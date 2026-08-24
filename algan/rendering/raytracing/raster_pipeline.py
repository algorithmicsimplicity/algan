"""Host orchestration for the deterministic hybrid raster front-end.

The frontend emits exact primary coverage for the whole prepared frame window
at once (not per GPU tile): each primitive is split into ``RASTER_CHUNK``-sized
candidate chunks, exact hits are emitted, the surviving fragment records are
ordered by the classic deterministic ``(depth-bin, descending layer)``
relation, and ``sheets.compact_sheets`` aggregates them into the per-pixel
sheet records the resolve kernel consumes (DESIGN_sheet_resolve.md).  Future
work should benchmark a true square screen-tile/bin architecture for better
projection reuse and cache locality.

Large transient arrays are allocated from ``ManualMemory`` so failed raster
attempts can restore the arena pointer and retry a smaller primary slice.
Torch sort/index scratch remains allocator-owned because PyTorch's radix sort
cannot write directly into an arena view.
"""

from __future__ import annotations

import math

import torch

from algan.environment import env_str
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    MIN_HIT_DISTANCE,
)
from algan.settings import SETTINGS

rt_settings = SETTINGS.raytracing
from algan.rendering.raytracing.raster_taichi import (
    _AA_BACKFACE_BIT as AA_BACKFACE_BIT,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_DUMP_COLS as AA_DUMP_COLS,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_MASK_ALL as AA_MASK_ALL,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_MAT_OPAQUE_BIT as AA_MAT_OPAQUE_BIT,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_ONE_MESH_BIT as AA_ONE_MESH_BIT,
)
from algan.rendering.raytracing.raster_taichi import (
    _BEZ_BORDER_BITS,
    AA_FULL_COVERAGE,
    RASTER_CHUNK,
    Z_SENTINEL,
    _aa_run_full,
    raster_bez_count,
    raster_bez_write,
    raster_shadow_trace,
    raster_tri_count,
    raster_tri_write,
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


def _aa_group(aa_bez, aa_tri):
    """The emission's ``aa_grp`` value for this batch.

    One definition, because the emission truncation below and the historical
    kernel readers had to agree about it, and they drifted once --
    ``ANALYTIC_AA_ONE_MESH`` set ``aa_grp = 3``, which ``_aa_run_full`` treats
    as the relaxed gate, while the truncation still tested
    ``ANALYTIC_AA_RUN_FULL`` alone and therefore withheld the mitigation. That
    combination truncates fragment lists whose area donors the relaxed
    semantics require, which is exactly the interior notch ss6.3.2 documents;
    measured on CUDA, it cost a flat quad -8% of ink wobble where wiring both
    gave -63%. Route every reader through this and ``_aa_run_full`` so the
    question can only be answered once.

    0 no grouping, 1 seam grouping, 2 + the relaxed emission-truncation gate
    (ss6.3.2), 3 + the one-mesh coverage ceiling (ss6.6, which implies 2 --
    the ceiling is only worth reading once the relaxed gate keeps its area
    donors). The ladder is a single integer, so a level inherits every level
    below it. The fragment walk's higher rungs (occlusion-write scaling, run
    caps, exact run lanes) are gone with the walk: the sheet resolve's
    per-sheet claim arithmetic subsumes them (DESIGN_sheet_resolve.md ss7),
    and its only emission-side dependency is the truncation gate here.
    """
    aa_grp = 1 if ((aa_bez or aa_tri) and rt_settings.ANALYTIC_AA_SEAM) else 0
    if aa_grp and rt_settings.ANALYTIC_AA_RUN_FULL:
        aa_grp = 2
    if aa_grp and rt_settings.ANALYTIC_AA_ONE_MESH:
        aa_grp = 3
    return aa_grp


def _aa_dump_request():
    """The requested (px, py, frame) from ``ALGAN_AA_DUMP``, or ``None``."""
    spec = env_str("ALGAN_AA_DUMP", "")
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
    buf = torch.zeros((_AA_DUMP_ROWS, AA_DUMP_COLS), dtype=torch.float32, device=device)
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


_AA_DUMP_NOTES = {
    0: "",
    1: "eff-skip",
    2: "bounce",
    3: "occl",
    4: "far-clip",
    5: "invalid",
    6: "seam-skip",
}
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
            print(
                f"[aa-dump:{tag}]   end bounced={int(r[1])} done={int(r[2])}"
                f" processed={int(r[3])} vis_all={r[4]:.5f}"
                f" acc=({r[5]:.4f},{r[6]:.4f},{r[7]:.4f},{r[8]:.4f})"
                f" w=({r[9]:.4f},{r[10]:.4f},{r[11]:.4f}) svis=[{svis}]"
            )
            continue
        kind = _AA_DUMP_KINDS.get(int(r[1]), "?")
        note = _AA_DUMP_NOTES.get(int(r[2]), "?")
        svis = " ".join(f"{v:.4f}" for v in r[16:24])
        print(
            f"[aa-dump:{tag}]   q={int(r[0]):3d} {kind:5s} {note:9s}"
            f" ref={int(r[3])} sid={int(r[4])} face={int(r[5])}"
            f" msk={int(r[6]):02x} cov={r[7]:.5f} pop={int(r[8])}"
            f" corr={r[9]:.5f} eff={r[10]:.5f} a_mat={r[11]:.4f}"
            f" alpha={r[12]:.5f} ts={r[13]:.4f} rmax={r[14]:.4f}"
            f" t={r[15]:.5f} svis=[{svis}]"
        )


def precompute_triangle_projection(
    merged,
    cam_origin,
    screen_point,
    pixel_basis_x,
    pixel_basis_y,
    half_w,
    half_h,
    memory,
    persist=False,
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
    out = memory.get_tensor(
        (max(1, frames), max(1, ntri), ncol), torch.float32, persist=persist
    )
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
    persist=False,
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
    pre_f = memory.get_tensor((frames, ncirc, 4), torch.float32, persist=persist)
    pre_f.copy_(
        torch.stack(((ymin - 1.0).floor(), (ymax + 1.0).ceil(), ymin, ymax), -1)
    )
    pre_x = memory.get_tensor((frames, ncirc, 2), torch.int64, persist=persist)
    pre_x.copy_(torch.stack((x0, x1), -1))
    # all_front implies front_any (eight corners), so the bounded reach base
    # omits the redundant ``& front_any``: a clipped straddler kept a front
    # corner by construction.
    pre_m = memory.get_tensor((frames, ncirc, 5), torch.bool, persist=persist)
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
    persist=False,
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
    pre_f = memory.get_tensor((frames, ntri, 4), torch.float32, persist=persist)
    pre_f.copy_(
        torch.stack(((ymin - 1.0).floor(), (ymax + 1.0).ceil(), ymin, ymax), -1)
    )
    pre_x = memory.get_tensor((frames, ntri, 2), torch.int64, persist=persist)
    pre_x.copy_(torch.stack((x0, x1), -1))
    # ``bounded`` already implies not-behind, so its reach base omits the
    # redundant ``& not_behind``.
    pre_m = memory.get_tensor((frames, ntri, 5), torch.bool, persist=persist)
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
    del is_bez, bez_code, bez_layer, tri_layer
    layer_order = torch.argsort(layer, descending=True, stable=True)
    del layer

    key_l = frag_key.index_select(0, layer_order)
    pixel = key_l >> 32
    t_bits = (key_l & 0xFFFFFFFF).to(torch.int32)
    del key_l
    # dtype-view reinterprets IEEE bits; it does not allocate a numeric cast.
    t = t_bits.view(torch.float32)
    depth_bin = torch.floor(t / DEPTH_TIE_EPSILON).to(torch.int64)
    del t, t_bits
    depth_bin.clamp_(0, 0x7FFFFFFF)
    primary_key = (pixel << 32) | depth_bin
    del pixel, depth_bin
    depth_order = torch.argsort(primary_key, stable=True)
    del primary_key
    # Named so the two permutations are freed before the caller's gathers
    # run: at 4K they are 56 MB each and this is the discovery peak.
    order = layer_order.index_select(0, depth_order)
    del layer_order, depth_order
    return order


def _gather_fragment_arrays(idx, key, ref, ab, cov, msk, opq):
    """The six-array fragment gather ``idx`` drives, as one pass.

    Returns the same six tensors ``index_select`` would. One kernel launch
    reads ``idx`` once instead of six times (``RASTER_FUSED_GATHER``,
    DESIGN_optimization_targets.md T5); a gather copies bits, so the two arms
    are bit-identical and the flag is there for the A/B rather than for a
    choice.

    Only for a gather whose SOURCES outlive it. Fusing forces all six outputs
    to exist before the first is written, so at a site that *replaces* its
    inputs -- the opaque-prefix truncation below, which rebinds each name and
    lets the old array die as the new one lands -- the fused form holds twelve
    arrays where the sequential one holds seven. Measured on a 4K frame that
    is +53 MB of peak for 4 ms of gather, so that site stays sequential.
    """
    m = int(idx.shape[0])
    if not rt_settings.RASTER_FUSED_GATHER or m == 0:
        return tuple(t.index_select(0, idx) for t in (key, ref, ab, cov, msk, opq))
    from algan.rendering.raytracing.sheet_compact_taichi import (
        gather_fragment_arrays,
    )

    out_key = torch.empty(m, dtype=key.dtype, device=key.device)
    out_ref = torch.empty(m, dtype=ref.dtype, device=ref.device)
    out_ab = torch.empty((m, 2), dtype=ab.dtype, device=ab.device)
    out_cov = torch.empty(m, dtype=cov.dtype, device=cov.device)
    out_msk = torch.empty(m, dtype=msk.dtype, device=msk.device)
    out_opq = torch.empty(m, dtype=opq.dtype, device=opq.device)
    # Taichi has no bool ndarray, so the flags ride as the bytes they are.
    gather_fragment_arrays(
        idx,
        m,
        key,
        ref,
        ab,
        cov,
        msk,
        opq.view(torch.uint8),
        out_key,
        out_ref,
        out_ab,
        out_cov,
        out_msk,
        out_opq.view(torch.uint8),
    )
    return out_key, out_ref, out_ab, out_cov, out_msk, out_opq


def _tri_obj_row(pix, ppf, time_start, rows):
    """The ``tri_obj`` row the KERNELS read for a fragment at compact pixel
    ``pix``, which is the row the host has to read to ask the same question.

    A fragment's key carries ``lp = (f - time_start) * ppf + p - tile_start``
    (``raster_taichi._pair_pixel``), so the pixel index is CHUNK-relative; the
    resolve turns it back into a frame with ``f = time_start + g // ppf`` and
    indexes ``tri_obj[f % rows]``, which is BATCH-relative. Every other frame
    derivation in this module adds ``time_start`` for exactly that reason
    (``_window_pairs``'s ``f_abs``, the per-frame pair loops); the ONE-MESH
    reduction did not, so on any chunk that did not start at frame 0 it asked a
    different frame's surface map than the kernel it feeds.

    Measured before the fix, and the reach is why this is an alignment rather
    than a bug report: over all six ``tests/full_renders`` scenes and every
    chunk starting past frame 0, **not one fragment's surface id moved** between
    the two rows (``benchmarks/_notch_scene_check.py``). A diced primitive's
    row -> SOURCE SURFACE map is frame-invariant whenever all its patches belong
    to one surface, which every PN primitive in those scenes is, so the rows
    differ only for a primitive carrying SEVERAL surfaces -- a packed-grid
    ``Surface``, or several meshes batched into one primitive. The two
    derivations must not be allowed to disagree while that is the only thing
    standing between them and a wrong answer.
    """
    return ((pix // ppf) + time_start) % rows


def _shadow_identity_epsilons(merged):
    """The shadow acceptance floors for this batch, in world units.

    Identity-aware rejection (``SHADOW_IDENTITY_REJECT``,
    DESIGN_mesh_identity_open.md ssI) replaces the absolute
    ``MIN_HIT_DISTANCE`` on the shadow path with a floor proportional to the
    batch's own scene scale -- the diagonal of the merged triangle bounding
    box over every frame of the batch. That is what decouples the shadow path
    from scene scale: 1e-4 is only ever the right number for a scene about
    ten units across.

    Returns ``(eps_self, eps_near)``: the floor a hit on the ray's own
    triangle keeps, and the (by default zero) share of it a hit on another
    triangle of the same mesh keeps. Degenerate batches -- no triangles, or a
    bounding box that is not finite -- fall back to ``MIN_HIT_DISTANCE`` so a
    pathological scene can never end up with a zero or NaN floor.
    """
    tri_pos = merged["tri_pos"]
    scale = 0.0
    if tri_pos.numel():
        # tri_pos is [frames, N, 9]: three vertices, three coordinates each.
        verts = tri_pos.reshape(-1, 3, 3)
        lo = verts.amin(dim=(0, 1))
        hi = verts.amax(dim=(0, 1))
        diag = (hi - lo).norm().item()
        if math.isfinite(diag):
            scale = diag
    eps_self = float(rt_settings.SHADOW_EPS_RELATIVE) * scale
    if not (eps_self > 0.0) or not math.isfinite(eps_self):
        eps_self = float(MIN_HIT_DISTANCE)
    # Clamped, not merely scaled: a negative fraction would make the same-mesh
    # floor negative, and `t > eps_near` would then accept hits at t <= 0 --
    # geometry BEHIND the ray origin occluding the light. NaN fails the
    # comparison and lands on 0.0 too.
    eps_near = eps_self * float(rt_settings.SHADOW_NEAR_FRACTION)
    if not (eps_near > 0.0):
        eps_near = 0.0
    return eps_self, eps_near


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
    env_in_composite=False,
):
    """Emit one exact, ordered primary-hit stream for the whole frame window.

    No tile-pixel z-buffer, coverage mask, or ray state is allocated.
    Candidate bboxes launch exact intersection COUNT/WRITE passes; the
    resulting hit records are ordered in sparse hit space, truncated after
    each pixel's first proven-opaque hit, then compacted into per-pixel
    sheets (DESIGN_sheet_resolve.md) for the resolve.  The persistent result
    is allocated from the arena's reverse pointer so forward coverage-sized
    wavefront state can coexist with it and be reset independently.

    Returns ``None`` when no exact pixel is covered, otherwise a dict
    containing compact ``frag_*``, ``covered_idx`` and ``run_offsets``
    arrays plus the ``sheet_*`` arrays the resolve consumes.

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
    # 3/4 select the RUN-CORRECTED representation for corr > 1 under rule A
    # (clamp) / B (redistribute) (DESIGN_analytic_aa_v2.md ss4.4). Both map to
    # the same emission representation (raster_taichi._tri_repr == 2); the
    # rule itself now lives in the sheet resolve's per-sheet redistribution.
    # Value 2 belonged to the deleted cells accounting.
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
        aa_tri_ss = (
            1
            + (2 if aa_tri >= 3 else rt_settings.analytic_aa_sliver_mode())
            + 4 * min(aa_tri - 1, 2)
        )
    aa_grp = _aa_group(aa_bez, aa_tri)
    # The sheet resolve (DESIGN_sheet_resolve.md) consumes EXACT areas, so
    # every geometry kind present must emit them: triangles need the run
    # representation (aa_tri >= 3), circuits their SDF coverage. The route
    # decision (analytic_raster_route_active) promises exactly this; there is
    # no other resolve to fall back to, so an emission-side disagreement is a
    # bug to surface, not a fallback to take — the same precedent as the
    # analytic-raster/use_raster check in tracer.py.
    if (has_tri and aa_tri < 3) or (has_bez and aa_bez <= 0):
        raise RuntimeError(
            "The sparse route is served by the sheet resolve, but the "
            "emission cannot produce sheets under the current analytic-AA "
            "settings (the route decision and the emission disagree)."
        )
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
        key_s, ref_s, ab_s, cov_s, msk_s, opaque_s = _gather_fragment_arrays(
            order, frag_key_u, frag_ref_u, frag_ab_u, frag_cov_u, frag_msk_u, opaque_u
        )
        del order
        # Kept before the coverage adjustment below overwrites it: this is
        # MATERIAL opacity, which the one-mesh rule needs, while the adjusted
        # opaque_s means "occludes every sample".
        mat_opaque_s = opaque_s
        if aa_bez or aa_tri:
            # A partially covering opaque hit does not hide what is behind it,
            # so it must not terminate its pixel's run: it stays an ordinary
            # alpha fragment (its alpha already carries the coverage). Under
            # the run representation the test is the SAMPLED claim, matching
            # the prepass and the resolve's magnitude (v2 ss4.1): a full-mask
            # fragment occludes every sample whatever its exact area says.
            if aa_tri >= 3:
                full_s = (msk_s & AA_MASK_ALL) == AA_MASK_ALL
                if _aa_run_full(aa_grp):
                    # ss6.3.2 needs the sheet claims to SEE their area donors,
                    # and this truncation is what hides them: a full-mask
                    # fragment cuts its pixel's prefix right there, so the
                    # empty-mask donors that complete its sheet's tiling never
                    # reach the compaction. The sheet then sums E over the one
                    # fragment it can see and darkens the pixel by (1 - E) --
                    # correct at a silhouette, a NOTCH in an interior tiling,
                    # and indistinguishable after the cut. Measured before this
                    # was added: 531 interior pixels of a flat quad and 920 of a
                    # Cylinder darkened by a mean 0.027, which is why
                    # _aa_line_check got worse while the coverage harness (which
                    # scores silhouette pixels only) said it got better.
                    #
                    # So under the relaxed gate a fragment must own every sample
                    # AND cover the pixel to terminate the prefix. This is the
                    # reason the shipped corr = 1 short-circuit is load-bearing
                    # rather than lazy: without it, a full mask is the renderer's
                    # only remaining evidence that the sheet tiles the pixel.
                    full_s = full_s & (cov_s >= AA_FULL_COVERAGE)
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
            del opaque_pos, starts
            keep_end = torch.minimum(first_opaque, ends)
            del first_opaque, ends
            keep = positions <= keep_end.index_select(0, segments)
            del positions, segments, keep_end
            truncated = int(keep.sum().item()) != num_frags
            keep_idx = keep.nonzero(as_tuple=True)[0] if truncated else None
            del keep
            if truncated:
                # One at a time, and NOT through _gather_fragment_arrays: each
                # rebinding frees the array it replaces, which is worth more
                # here than the fused gather's traffic saving (see its
                # docstring).
                key_s = key_s.index_select(0, keep_idx)
                ref_s = ref_s.index_select(0, keep_idx)
                ab_s = ab_s.index_select(0, keep_idx)
                cov_s = cov_s.index_select(0, keep_idx)
                msk_s = msk_s.index_select(0, keep_idx)
                mat_opaque_s = mat_opaque_s.index_select(0, keep_idx)
                del keep_idx
                pix_s = key_s >> 32
                covered, counts = torch.unique_consecutive(pix_s, return_counts=True)
                num_frags = int(key_s.shape[0])

        # -- ONE-MESH PIXELS (DESIGN_mesh_identity.md ss6.6) -----------------
        # A pixel every one of whose fragments is an OPAQUE triangle of a single
        # surface. There the mesh's coverage is its near sheet's exact area and
        # nothing else: both sheets project to the same silhouette, so the far
        # sheet must not add coverage on top. Marked here rather than in the
        # kernel because it is a segment reduction over the CSR the host already
        # has, and it rides in a spare frag_msk flag bit so no kernel argument
        # changes.
        one_mesh_cap = None
        # The one-mesh ceiling stays under the sheet resolve, as DATA on the
        # sheet record. DESIGN_sheet_resolve.md §7 first deleted it as
        # "subsumed by per-sheet claims", and the Phase-2 ink-wobble A/B
        # REFUTED that: with the cap gone the coarse Cylinder's far sheet
        # re-claims the corr residue and wobble regresses 2-4x (0.015 ->
        # 0.060; the exact-fit angles go 0.000 -> 0.032). Only the §6.7 run
        # lanes are truly subsumed (compaction has no budget to truncate).
        if rt_settings.ANALYTIC_AA_ONE_MESH and num_frags:
            tri_obj = merged["tri_obj"]
            ppf = int(width) * int(height)
            frame_of = _tri_obj_row(pix_s, ppf, int(time_start), tri_obj.shape[0])
            safe_ref = ref_s.clamp_min(0).to(torch.int64)
            sid = tri_obj[frame_of, safe_ref].to(torch.int64)
            # A bezier fragment has ref < 0 and no surface id; a pixel holding
            # one is never single-mesh. Same for a non-opaque fragment, whose
            # far sheet is legitimately visible THROUGH the near one.
            usable = (ref_s >= 0) & mat_opaque_s
            sid = torch.where(usable, sid, torch.full_like(sid, -1))
            del frame_of, safe_ref, usable
            seg = torch.repeat_interleave(
                torch.arange(covered.numel(), dtype=torch.int64, device=device),
                counts,
            )
            lo = torch.full(
                (covered.numel(),), 1 << 40, dtype=torch.int64, device=device
            )
            hi = torch.full((covered.numel(),), -1, dtype=torch.int64, device=device)
            lo.scatter_reduce_(0, seg, sid, reduce="amin", include_self=True)
            hi.scatter_reduce_(0, seg, sid, reduce="amax", include_self=True)
            one_mesh = (lo == hi) & (lo >= 0)
            del lo, hi, sid
            # The mesh's coverage CEILING: the larger of the two sheets' exact
            # areas. Well inside a silhouette a closed solid's sheets tile to
            # the same area, so this is that area and the far sheet gets no room
            # -- the suppression rule, recovered. At the BOUNDARY the near
            # sheet's projected area shrinks toward zero while the footprint
            # does not, and there the larger sheet is the right answer where
            # suppression under-covers (measured: a 0.045-radius rod diced to
            # (256, 2) is nearly all boundary, and suppression flips its signed
            # error to -0.0344 and notches 1676 of 3508 interior pixels).
            # Accumulated in FLOAT64 and rounded back, and that is load-bearing
            # rather than cautious. ``scatter_add_`` is a float atomic add, so its
            # summation order is not reproducible on CUDA -- measured, a 400k-into-
            # 5k reduction of this shape spreads 1.5e-05 across runs. That would be
            # invisible in a colour, but this feeds a THRESHOLD: the kernel clips
            # only when ``eff > frag_cap - mesh_ink``, so a ceiling that wobbles in
            # its low bits flips borderline fragments in and out of being clipped,
            # which is a finite coverage change, which bloom then amplifies. It was
            # measured: two consecutive renders of ``materials_and_lighting``
            # differed by up to 28 channel values over 9.6% of a frame, while the
            # same scene with the rule OFF is bit-identical run to run.
            #
            # In float64 the reassociation error lands ~9 orders below a float32
            # ulp, so rounding the ceiling to float32 absorbs it: the reduction is
            # bitwise reproducible in practice (verified over 6 runs, spread 0.0),
            # and any residual last-bit float64 difference cannot survive the cast.
            # Cost is nothing measurable -- it is one pass over the fragments,
            # against a render this sits at ~1% of.
            is_back = (msk_s & AA_BACKFACE_BIT) != 0
            acc = torch.float64
            cov_acc = cov_s.to(acc)
            front = torch.zeros(covered.numel(), dtype=acc, device=device)
            back = torch.zeros_like(front)
            zero = torch.zeros((), dtype=acc, device=device)
            front.scatter_add_(0, seg, torch.where(is_back, zero, cov_acc))
            back.scatter_add_(0, seg, torch.where(is_back, cov_acc, zero))
            cap_pix = torch.maximum(front, back).clamp_max_(1.0).to(cov_s.dtype)
            del front, back, cov_acc, is_back
            msk_s = msk_s | torch.where(
                one_mesh.index_select(0, seg),
                torch.full_like(msk_s, AA_ONE_MESH_BIT),
                torch.zeros_like(msk_s),
            )
            one_mesh_cap = (one_mesh, seg, cap_pix)

        # SHEET_SAMPLE_DEPTH: mark MATERIAL-opaque triangles so the compaction
        # can tell a depth-gate enforcer sheet (material-opaque, full sample
        # union, full exact coverage) from a translucent one. Classification is
        # uniform within a surface -- the bit comes from the material, and one
        # band never spans two meshes -- so per-fragment is per-band. Rides the
        # mask word as data; every reader masks with AA_MASK_ALL or tests named
        # flag bits, so it is inert where unread.
        if rt_settings.SHEET_SAMPLE_DEPTH and num_frags:
            msk_s = msk_s | torch.where(
                mat_opaque_s & (ref_s >= 0),
                torch.full_like(msk_s, AA_MAT_OPAQUE_BIT),
                torch.zeros_like(msk_s),
            )

        num_covered = int(covered.numel())
        # Per-fragment so the kernels index it exactly like frag_cov; 2.0 is the
        # "no ceiling" sentinel, which every non-one-mesh pixel keeps.
        cap_s = torch.full_like(cov_s, 2.0)
        if one_mesh_cap is not None:
            cap_s = torch.where(
                one_mesh_cap[0].index_select(0, one_mesh_cap[1]),
                one_mesh_cap[2].index_select(0, one_mesh_cap[1]),
                cap_s,
            )
            one_mesh_cap = None
            del one_mesh, seg, cap_pix
        frag_key = _arena_tensor(memory, (num_frags,), torch.int64, persist=True)
        frag_ref = _arena_tensor(memory, (num_frags,), torch.int32, persist=True)
        frag_ab = _arena_tensor(memory, (num_frags, 2), torch.float32, persist=True)
        frag_cov = _arena_tensor(memory, (num_frags,), torch.float32, persist=True)
        frag_msk = _arena_tensor(memory, (num_frags,), torch.int32, persist=True)
        frag_cap = _arena_tensor(memory, (num_frags,), torch.float32, persist=True)
        covered_idx = _arena_tensor(memory, (num_covered,), torch.int32, persist=True)
        run_offsets = _arena_tensor(
            memory, (num_covered + 1,), torch.int32, 0, persist=True
        )
        frag_key.copy_(key_s)
        frag_ref.copy_(ref_s)
        frag_ab.copy_(ab_s)
        frag_cov.copy_(cov_s)
        frag_msk.copy_(msk_s)
        frag_cap.copy_(cap_s)
        covered_idx.copy_(covered.to(torch.int32))
        run_offsets[1:].copy_(torch.cumsum(counts.to(torch.int32), 0))
        # Everything above now lives in the arena. The sheet compaction below
        # is this function's memory peak, so the host copies are released
        # before it starts rather than at the return.
        del key_s, ref_s, ab_s, cov_s, msk_s, cap_s, pix_s, mat_opaque_s
        del opaque_s, covered, counts

        # -- SHEET COMPACTION (DESIGN_sheet_resolve.md P1/P2) ---------------
        # Aggregation happens here, once, before any kernel: the resolve then
        # composites a few depth-sorted sheets per pixel instead of walking
        # the raw fragment list. Intermediates are allocator-owned (like the
        # torch sort scratch above); only the final sheet arrays persist.
        from algan.rendering.raytracing.sheets import compact_sheets

        stream = compact_sheets(
            {
                "frag_key": frag_key,
                "frag_ref": frag_ref,
                "frag_ab": frag_ab,
                "frag_cov": frag_cov,
                "frag_msk": frag_msk,
                "frag_cap": frag_cap,
                "covered_idx": covered_idx,
                "run_offsets": run_offsets,
                "num_fragments": num_frags,
                "num_covered": num_covered,
            },
            merged,
            cam_origin,
            pixel_world_scale,
            int(time_start),
            int(width),
            int(height),
            band_rule="prim",
            band_c=2.0,
            tri_screen=tri_screen,
            shade_split=bool(rt_settings.SHEET_SHADE_SPLIT),
            positioned_depth=bool(rt_settings.SHEET_POSITIONED_DEPTH),
            sample_depth=bool(rt_settings.SHEET_SAMPLE_DEPTH),
        )
        ns = int(stream["num_sheets"])
        sheet_key = _arena_tensor(memory, (ns,), torch.int64, persist=True)
        sheet_ref = _arena_tensor(memory, (ns,), torch.int32, persist=True)
        sheet_ab = _arena_tensor(memory, (ns, 2), torch.float32, persist=True)
        sheet_cov = _arena_tensor(memory, (ns,), torch.float32, persist=True)
        sheet_msk = _arena_tensor(memory, (ns,), torch.int32, persist=True)
        sheet_cap_t = _arena_tensor(memory, (ns,), torch.float32, persist=True)
        sheet_offsets = _arena_tensor(
            memory, (num_covered + 1,), torch.int32, persist=True
        )
        sheet_key.copy_(stream["sheet_key"])
        sheet_ref.copy_(stream["sheet_ref"])
        sheet_ab.copy_(stream["sheet_ab"])
        # The resolve consumes the COMPOSITING weights, not the record: they
        # are the sheet's own area and union everywhere except inside a band
        # the shading-class split subdivided, where they carry §4.4's
        # additive sibling arithmetic (``sheets._sibling_weights``).
        sheet_cov.copy_(stream["sheet_wgt"])
        sheet_msk.copy_(stream["sheet_wmsk"])
        sheet_cap_t.copy_(stream["sheet_cap"])
        sheet_offsets.copy_(stream["sheet_offsets"].to(torch.int32))
        stream = None
        sheet_data = {
            "sheet_key": sheet_key,
            "sheet_ref": sheet_ref,
            "sheet_ab": sheet_ab,
            "sheet_cov": sheet_cov,
            "sheet_msk": sheet_msk,
            "sheet_cap": sheet_cap_t,
            "sheet_offsets": sheet_offsets,
            "num_sheets": ns,
            # Pinned with the emission like aa_*: the resolve's env
            # handling must match the frame buffer this batch prefilled.
            "env_in_composite": bool(env_in_composite),
        }

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
    # 28 B/fragment of compact result, plus 32 B of persistent sheet record +
    # the sheet CSR. The torch-side sort and scatter intermediates are
    # allocator-owned, like the fragment sort's.
    per_frag = 28
    discovery_bytes = discovery_frags * 29 + num_frags * per_frag + num_covered * 8
    discovery_bytes += sheet_data["num_sheets"] * 32 + (num_covered + 1) * 4
    rt_settings.note_sparse_discovery_footprint(
        discovery_bytes, int(time_end) - int(time_start)
    )

    result = {
        "frag_key": frag_key,
        "frag_ref": frag_ref,
        "frag_ab": frag_ab,
        "frag_cov": frag_cov,
        "frag_msk": frag_msk,
        "frag_cap": frag_cap,
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
    result.update(sheet_data)
    result["sheets"] = True
    return result


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
    tri_pids,
    skip_unlit_normal,
    refraction_flag,
    ior_stack_flag,
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
    bez_bvh,
    layer_offset_triangles,
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

    if not coverage.get("sheets"):
        # The sheet compaction is the only resolve since the fragment walk's
        # deletion; a sparse coverage dict without sheet records means the
        # emission and this call disagree about the route. Fail loudly
        # rather than paint nothing.
        raise RuntimeError(
            "Sparse raster coverage reached the resolve without sheet "
            "records; the fragment walk that consumed raw fragment lists "
            "is deleted."
        )
    # THE SHEET RESOLVE (DESIGN_sheet_resolve.md). The emission compacted
    # this window's fragments into per-pixel sheets; composite and shade
    # those instead of walking the raw fragment list.
    # Shadows run as two launches of the SAME kernel body: mode 1 walks
    # the transport and writes one candidate event per accepted lit
    # triangle sheet (event identity = sheet index, no atomics), the
    # host compacts + traces them, and mode 2 shades reading the traced
    # visibility. Shadow-free batches take mode 0 in one launch.
    from algan.rendering.raytracing.sheet_resolve_taichi import (
        sheet_resolve_shade,
    )

    so_host = coverage.get("sheet_offsets_host")
    if so_host is None:
        so_host = coverage["sheet_offsets"].cpu()
        coverage["sheet_offsets_host"] = so_host
    s_start = int(so_host[c0])
    s_end = int(so_host[c1])
    sheet_offsets = _arena_tensor(memory, (num_covered + 1,), torch.int32)
    torch.sub(coverage["sheet_offsets"][c0 : c1 + 1], s_start, out=sheet_offsets)
    (rs_ro, rs_rd, rs_acc, rs_sca, rs_int, *_stubs) = state
    sec_aa = rt_settings.effective_analytic_aa_secondary_samples()
    dump_req = _aa_dump_request()
    sdump = (
        _aa_dump_buffer(dump_req, covered_idx.device)
        if dump_req
        else _aa_dump_arg(covered_idx.device)
    )
    num_slice_sheets = s_end - s_start
    pre_args = (
        num_covered,
        sheet_offsets,
        coverage["sheet_key"][s_start:s_end],
        coverage["sheet_ref"][s_start:s_end],
        coverage["sheet_ab"][s_start:s_end],
        coverage["sheet_cov"][s_start:s_end],
        coverage["sheet_msk"][s_start:s_end],
        coverage["sheet_cap"][s_start:s_end],
        merged["tri_pos"],
        merged["tri_norm"],
        merged["tri_extra"],
        merged["tri_colors"],
        merged["tri_uvs"],
        merged["tri_tex_meta"],
        merged["textures"],
        int(merged["num_colored_triangles"]),
        col_row_arr,
        merged["tri_mat_id"],
        merged["tri_mat"],
        merged["circuit_meta"],
        merged["circuit_colors"],
        merged["circuit_border_colors"],
        light_pos,
        light_col,
        int(num_lights),
        layer_offsets,
        int(frag_flag),
        frag_pipelines,
        int(tri_pids),
        int(refraction_flag),
        int(ior_stack_flag),
        int(skip_unlit_normal),
        1 if int(merged.get("num_circuits", 0)) > 0 else 0,
        sec_aa,
        float(rt_settings.ANALYTIC_AA_SECONDARY_MIN_ENERGY),
        int(rt_settings.glossy_reflection_mode()),
        1 if coverage.get("env_in_composite") else 0,
    )
    post_args = (
        covered_idx,
        int(time_start),
        int(width),
        int(height),
        cam_origin,
        screen_point,
        pixel_basis_x,
        pixel_basis_y,
        gen_meta,
        rs_ro,
        rs_rd,
        rs_acc,
        rs_sca,
        rs_int,
        rs_pix,
        pix_accum,
        rs_alloc,
        1 if dump_req else 0,
        sdump,
    )
    dummy_i = _arena_tensor(memory, (1,), torch.int32, 0)
    dummy_f3 = _arena_tensor(memory, (1, 3), torch.float32)
    dummy_f6 = _arena_tensor(memory, (1, 6), torch.float32)
    # RGB visibility payload: one triple per (event, light), channel-last.
    dummy_vis = _arena_tensor(memory, (1, 1, 3), torch.float32, 1.0)
    # Shadow terminator (RENDERER_WORK_QUEUE.md item 20), read live like the
    # other shadow gates. It is read before the mode split because all three
    # sheet_resolve_shade launches take it as a template and the shadow arm's
    # event_toff size depends on it (full-size only in mode 1, whose trace is
    # what reads rows back).
    term_mode = int(rt_settings.shadow_terminator_mode())
    term_on = term_mode == 1
    if shadow_flag:
        S = max(1, num_slice_sheets)
        sheet_accept = _arena_tensor(memory, (S,), torch.int32, 0)
        event_pos = _arena_tensor(memory, (S, 3), torch.float32)
        event_snrm = _arena_tensor(memory, (S, 3), torch.float32)
        event_fnrm = _arena_tensor(memory, (S, 3), torch.float32)
        event_frame = _arena_tensor(memory, (S,), torch.int32, 0)
        event_msk = _arena_tensor(memory, (S,), torch.int32, 0xF)
        event_dp = _arena_tensor(memory, (S if sec_aa > 1 else 1, 6), torch.float32)
        event_toff = _arena_tensor(memory, (S if term_on else 1, 3), torch.float32)
        sheet_event_id = _arena_tensor(memory, (S,), torch.int32, -1)
        sheet_resolve_shade(
            *pre_args,
            1,
            term_mode,
            sheet_accept,
            event_pos,
            event_snrm,
            event_fnrm,
            event_frame,
            event_msk,
            event_dp,
            event_toff,
            sheet_event_id,
            dummy_vis,
            *post_args,
        )
        acc_idx = sheet_accept[:num_slice_sheets].nonzero(as_tuple=True)[0]
        num_events = int(acc_idx.numel())
        # RGB visibility: one triple per (event, light), channel-last -- the
        # layout raster_shadow_trace writes and sheet_resolve_shade's mode 2
        # reads back.
        shadow_vis = _arena_tensor(
            memory,
            (max(1, num_events), max(1, int(num_lights)), 3),
            torch.float32,
            1.0,
        )
        if num_events:
            sheet_event_id[:num_slice_sheets].scatter_(
                0,
                acc_idx,
                torch.arange(num_events, dtype=torch.int32, device=acc_idx.device),
            )
            ev_pos = event_pos.index_select(0, acc_idx)
            ev_snrm = event_snrm.index_select(0, acc_idx)
            ev_fnrm = event_fnrm.index_select(0, acc_idx)
            ev_frame = event_frame.index_select(0, acc_idx)
            ev_msk = event_msk.index_select(0, acc_idx)
            # Identity-aware shadow rejection (SHADOW_IDENTITY_REJECT): hand
            # the trace each accepted event's SOURCE triangle, so it can tell
            # a hit on the surface the ray left from a hit on a different one
            # and spare the latter the acceptance floor entirely. The sheet
            # ref IS that triangle (-1 for a bezier-sourced event, which the
            # kernel reads as "no identity" and traces with the old epsilon).
            identity_on = bool(rt_settings.SHADOW_IDENTITY_REJECT)
            if identity_on:
                ev_src_prim = (
                    coverage["sheet_ref"][s_start:s_end]
                    .index_select(0, acc_idx)
                    .to(torch.int32)
                )
                eps_self, eps_near = _shadow_identity_epsilons(merged)
            else:
                ev_src_prim = dummy_i
                eps_self, eps_near = float(MIN_HIT_DISTANCE), 0.0
            ev_dp = event_dp.index_select(0, acc_idx) if sec_aa > 1 else event_dp
            ev_toff = event_toff.index_select(0, acc_idx) if term_on else event_toff
            from algan.rendering.raytracing.refit_bvh import RefitBVH

            raster_shadow_trace(
                num_events,
                ev_pos,
                ev_snrm,
                ev_fnrm,
                ev_frame,
                ev_msk,
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
                merged["tri_extra"],
                int(merged["num_colored_triangles"]),
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
                1 if isinstance(t_bvh, RefitBVH) else 0,
                1 if int(merged.get("num_triangles", 0)) > 0 else 0,
                1 if int(merged.get("num_circuits", 0)) > 0 else 0,
                ev_dp,
                ev_toff,
                sec_aa,
                shadow_vis,
                int(shadow_flag),
                # Identity-aware rejection: the hit-side surface map, the
                # per-event source triangle, the two scene-scaled floors, and
                # the compile-time gate. With the toggle off the kernel never
                # reads either array (1-element dummies keep the signature)
                # and every acceptance test compiles to exactly the
                # pre-identity predicate.
                merged["tri_obj"] if identity_on else dummy_i,
                ev_src_prim,
                eps_self,
                eps_near,
                1 if identity_on else 0,
                term_mode,
            )
        sheet_resolve_shade(
            *pre_args,
            2,
            term_mode,
            sheet_accept,
            event_pos,
            event_snrm,
            event_fnrm,
            event_frame,
            event_msk,
            event_dp,
            event_toff,
            sheet_event_id,
            shadow_vis,
            *post_args,
        )
    else:
        # Shadow-free resolve: mode 0 compiles no shadow logic at all, so the
        # terminator gate cannot change this launch's output -- but it is a
        # ti.template(), and forwarding the live setting here would compile a
        # second variant of this kernel per gate value for nothing. Pass a
        # literal 0; only the mode 1 / mode 2 launches above take a meaningful
        # gate.
        sheet_resolve_shade(
            *pre_args,
            0,
            0,
            dummy_i,
            dummy_f3,
            dummy_f3,
            dummy_f3,
            dummy_i,
            dummy_i,
            dummy_f6,
            dummy_f3,
            dummy_i,
            dummy_vis,
            *post_args,
        )
    if dump_req:
        _aa_dump_emit("sheet-resolve", sdump)
    return covered_idx
