"""Packed acceleration tables for planar Bezier-circuit edge queries.

The ray tracer represents every cubic circuit as a sampled 2D polyline.  A
naive point classification walks every sampled edge twice: once for the
horizontal even/odd crossing test and once for the nearest visible-border
segment.  This module builds two compact CSR indices over those unchanged
edges:

* scanline bins map a local-space y interval to the edges that may cross it;
* a uniform 2D grid maps cells to visible-border edges whose AABBs overlap.

The tables and their per-circuit floating-point bounds are packed into one
``int32`` tensor.  Float metadata is stored by bit reinterpretation, allowing
all kernels to replace the old ``edge_offsets`` argument without increasing
Taichi's runtime-argument count.

A third, optional table is a coarse **cell classification grid** over each
circuit's local bounds (see ``_build_cell_classification``): per cell it
stores whether every point of the cell provably shares one even/odd crossing
parity (plus that parity), and a conservative squared lower bound on the
distance from anywhere in the cell to the nearest border-visible edge.  The
point-metric kernel uses it to skip both edge loops for pixels deep inside or
far outside a filled region -- by far the most common case for text -- while
staying byte-identical: the classification is only consulted where it is
provably equivalent to the exact predicates.
"""

import os

import torch


BEZIER_SCAN_BINS = 16
BEZIER_SPATIAL_GRID = 8
BEZIER_SPATIAL_CELLS = BEZIER_SPATIAL_GRID * BEZIER_SPATIAL_GRID

# Cell-classification grid resolution (cells per axis, per circuit). The
# runtime dimension actually used is stored per batch in the header (it is
# halved when a batch has too many (frame, circuit) groups for the memory
# budget), so kernels read it from the header rather than this constant.
# ALGAN_BEZ_CLASS=0 disables the table (kernels then always run the exact
# edge loops -- the byte-identical A/B reference).
BEZIER_CLASS_GRID = max(4, int(os.environ.get("ALGAN_BEZ_CLASS_GRID", "32")))
BEZIER_CLASS_ENABLED = os.environ.get("ALGAN_BEZ_CLASS", "1") == "1"
# Memory budget for the classification section, in int32 words.
BEZIER_CLASS_MAX_WORDS = int(
    float(os.environ.get("ALGAN_BEZ_CLASS_MAX_MB", "64")) * 1024 * 1024 // 4)
# Truncation radius of the distance field, in cell half-diagonals. Cells
# farther than this from every visible edge store the (still conservative)
# truncated value. Build cost grows roughly quadratically with this radius
# (each edge is inserted into every cell within it), while the benefit only
# needs the cap to exceed the query radius -- a few screen pixels in plane
# units -- so a small multiple of the cell size is plenty.
_CLASS_DIST_CAP_HALF_DIAGS = 4.0
# Expanded (edge x cell) pair budget per processed frame chunk. The expansion
# arrays are the build's only large transients and they live in the GPU
# merge's *headroom* (outside the render arena), which is tight on small
# cards -- the arena preflight estimates the merge peak from its input bytes
# and knows nothing about this scratch, so it must stay small (~tens of MB).
# A single frame whose candidate pairs exceed the hard cap keeps the all-zero
# (never-consulted, always-safe) classification instead of risking a
# transient OOM during the merge.
_CLASS_PAIR_BUDGET = 1_000_000
_CLASS_PAIR_HARD_CAP = 4_000_000

# Fixed header layout for each (edge frame, circuit) record.
_BEZ_EDGE_START = 0
_BEZ_EDGE_END = 1
_BEZ_MIN_U = 2
_BEZ_MIN_V = 3
_BEZ_MAX_U = 4
_BEZ_MAX_V = 5
_BEZ_GRID_INV_U = 6
_BEZ_GRID_INV_V = 7
_BEZ_SCAN_INV_V = 8
_BEZ_CLASS_INV_U = 9
_BEZ_CLASS_INV_V = 10
_BEZ_CLASS_BASE = 11
_BEZ_CLASS_DIM = 12
BEZIER_SCAN_OFFSET_BASE = 13
BEZIER_SPATIAL_OFFSET_BASE = BEZIER_SCAN_OFFSET_BASE + BEZIER_SCAN_BINS + 1
BEZIER_ACCEL_HEADER_SIZE = BEZIER_SPATIAL_OFFSET_BASE + BEZIER_SPATIAL_CELLS + 1

# Public aliases used by the Taichi module.  Keeping the field constants here
# makes the Python builder and kernel decoder share one source of truth.
BEZIER_EDGE_START = _BEZ_EDGE_START
BEZIER_EDGE_END = _BEZ_EDGE_END
BEZIER_MIN_U = _BEZ_MIN_U
BEZIER_MIN_V = _BEZ_MIN_V
BEZIER_MAX_U = _BEZ_MAX_U
BEZIER_MAX_V = _BEZ_MAX_V
BEZIER_GRID_INV_U = _BEZ_GRID_INV_U
BEZIER_GRID_INV_V = _BEZ_GRID_INV_V
BEZIER_SCAN_INV_V = _BEZ_SCAN_INV_V
BEZIER_CLASS_INV_U = _BEZ_CLASS_INV_U
BEZIER_CLASS_INV_V = _BEZ_CLASS_INV_V
BEZIER_CLASS_BASE = _BEZ_CLASS_BASE
BEZIER_CLASS_DIM = _BEZ_CLASS_DIM


def _float_bits(values: torch.Tensor) -> torch.Tensor:
    """Return float32 values reinterpreted as int32 without a copy."""
    return values.to(torch.float32).contiguous().view(torch.int32)


def _grouped_offsets(keys: torch.Tensor, num_groups: int,
                     bins_per_group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort flat integer ``keys`` and return order + per-group CSR offsets.

    The returned offsets are relative to the beginning of the sorted reference
    list and have shape ``[num_groups, bins_per_group + 1]``.
    """
    num_keys = num_groups * bins_per_group
    if keys.numel() == 0:
        return (torch.empty((0,), dtype=torch.long, device=keys.device),
                torch.zeros((num_groups, bins_per_group + 1),
                            dtype=torch.long, device=keys.device))

    order = torch.argsort(keys)
    counts = torch.bincount(keys, minlength=num_keys).reshape(
        num_groups, bins_per_group)
    group_totals = counts.sum(dim=1)
    group_bases = group_totals.cumsum(dim=0) - group_totals
    local_prefix = torch.cat(
        (torch.zeros((num_groups, 1), dtype=torch.long, device=keys.device),
         counts.cumsum(dim=1)),
        dim=1,
    )
    return order, local_prefix + group_bases.unsqueeze(1)


def _expand_intervals(source: torch.Tensor,
                      counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand integer intervals for selected source rows.

    Returns ``(expanded_source_rows, local_offsets)`` where each source row is
    repeated ``counts[row]`` times and local offsets run from zero to count-1.
    """
    if source.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=source.device)
        return empty, empty

    selected_counts = counts[source]
    expanded_source = torch.repeat_interleave(source, selected_counts)
    total = expanded_source.numel()
    if total == 0:
        empty = torch.empty((0,), dtype=torch.long, device=source.device)
        return empty, empty

    interval_starts = selected_counts.cumsum(dim=0) - selected_counts
    local_offsets = (torch.arange(total, device=source.device)
                     - torch.repeat_interleave(interval_starts,
                                               selected_counts))
    return expanded_source, local_offsets


def _build_cell_classification(edges_2d, circuit_of_edge, finite,
                               min_u, min_v, max_u, max_v, grid_dim):
    """Classify a ``grid_dim`` x ``grid_dim`` grid over each circuit's bounds.

    Returns ``(flags, dist_bits)``, both ``[T * C, grid_dim * grid_dim]``
    int32:

    * ``flags`` bit 0: every point of the cell provably has the same even/odd
      crossing parity as the cell center (no edge passes within a conservative
      margin of the cell rectangle); bit 1: that parity.
    * ``dist_bits``: f32 bits of a conservative *lower* bound on the squared
      distance from anywhere in the cell to the nearest border-visible edge,
      truncated at ``_CLASS_DIST_CAP_HALF_DIAGS`` half-diagonals.

    All-zero rows (flag 0, distance bound 0) are the safe default: the kernel
    fast path never triggers on them and falls back to the exact loops.

    Conservativeness notes. The stored distance bound uses
    ``point_to_segment(center) - half_diag - margin``, a lower bound on the
    cell-rectangle-to-segment distance minus margin; ``margin`` (relative to
    the circuit extent and coordinate magnitude) also absorbs the f32 rounding
    of both this build and the kernel's cell lookup. Parity is uniform across
    an edge-free cell because the winding parity of a point only changes when
    the point crosses an edge; the kernel's floating-point crossing predicate
    matches the mathematical parity whenever the pixel is farther than float
    noise from every edge, which the same margin guarantees.
    """
    device = edges_2d.device
    dtype = edges_2d.dtype
    T, E = edges_2d.shape[:2]
    C = min_u.shape[1]
    G = int(grid_dim)
    cells = G * G
    num_groups = T * C

    flags_out = torch.zeros((num_groups, cells), dtype=torch.int32,
                            device=device)
    dist_out = torch.zeros((num_groups, cells), dtype=torch.int32,
                           device=device)
    if E == 0 or C == 0:
        return flags_out, dist_out

    extent_u = (max_u - min_u).clamp_min(1e-12)
    extent_v = (max_v - min_v).clamp_min(1e-12)
    cell_w = extent_u / G
    cell_h = extent_v / G
    half_diag = 0.5 * torch.sqrt(cell_w * cell_w + cell_h * cell_h)
    coord_mag = torch.maximum(
        torch.maximum(min_u.abs(), max_u.abs()),
        torch.maximum(min_v.abs(), max_v.abs()))
    margin = 1e-4 * torch.maximum(extent_u, extent_v) + 1e-6 * coord_mag
    dist_cap = _CLASS_DIST_CAP_HALF_DIAGS * half_diag

    x0 = edges_2d[..., 0]
    y0 = edges_2d[..., 1]
    x1 = edges_2d[..., 2]
    y1 = edges_2d[..., 3]
    visible = edges_2d[..., 4] > 0.5

    # Static circuits repeat identical edge rows across frames (edges live in
    # circuit-local plane coordinates, so camera motion never perturbs them).
    # Classify each *distinct* frame once and broadcast its rows afterwards --
    # text-heavy scenes are mostly static per batch, so this divides the build
    # cost by the batch's repetition factor. torch.unique compares exact
    # values, so a reused classification is bitwise what that frame would have
    # built for itself.
    _, frame_class = torch.unique(edges_2d.reshape(T, -1), dim=0,
                                  return_inverse=True)
    frame_class = frame_class.cpu()
    rep_of_class = {}
    rep_frames = []
    for t in range(T):
        cls = int(frame_class[t])
        if cls not in rep_of_class:
            rep_of_class[cls] = t
            rep_frames.append(t)

    # Chunk representative frames by their *estimated* expanded pair counts,
    # not a fixed frame stride: the distance-pass expansion arrays are the
    # build's only large transients and they live in the GPU merge's headroom,
    # which the arena preflight sizes without knowing about this scratch. The
    # estimate below reproduces the per-edge cell-range math; the exact
    # in-chunk count still enforces the hard cap.
    ge_reach = (dist_cap + half_diag + margin)[:, circuit_of_edge]
    ge_min_u = min_u[:, circuit_of_edge]
    ge_min_v = min_v[:, circuit_of_edge]
    ge_cell_w = cell_w[:, circuit_of_edge]
    ge_cell_h = cell_h[:, circuit_of_edge]
    fx0 = torch.floor((torch.minimum(x0, x1) - ge_reach - ge_min_u)
                      / ge_cell_w).clamp_(0, G - 1)
    fx1 = torch.floor((torch.maximum(x0, x1) + ge_reach - ge_min_u)
                      / ge_cell_w).clamp_(0, G - 1)
    fy0 = torch.floor((torch.minimum(y0, y1) - ge_reach - ge_min_v)
                      / ge_cell_h).clamp_(0, G - 1)
    fy1 = torch.floor((torch.maximum(y0, y1) + ge_reach - ge_min_v)
                      / ge_cell_h).clamp_(0, G - 1)
    est = (fx1 - fx0 + 1) * (fy1 - fy0 + 1)
    frame_pairs = torch.where(finite, est, torch.zeros_like(est)).sum(
        1, dtype=torch.float64).cpu()
    del ge_reach, ge_min_u, ge_min_v, ge_cell_w, ge_cell_h
    del fx0, fx1, fy0, fy1, est

    idx = 0
    while idx < len(rep_frames):
        chunk = [rep_frames[idx]]
        chunk_total = float(frame_pairs[rep_frames[idx]])
        idx += 1
        while (idx < len(rep_frames)
               and chunk_total + float(frame_pairs[rep_frames[idx]])
               <= _CLASS_PAIR_BUDGET):
            chunk_total += float(frame_pairs[rep_frames[idx]])
            chunk.append(rep_frames[idx])
            idx += 1
        S = len(chunk)
        sel = torch.tensor(chunk, device=device, dtype=torch.long)
        # Flat (chunk frame, edge) rows and their (chunk frame, circuit) group.
        g_local = (torch.arange(S, device=device).view(-1, 1) * C
                   + circuit_of_edge.view(1, -1)).reshape(-1)
        ex0 = x0.index_select(0, sel).reshape(-1)
        ey0 = y0.index_select(0, sel).reshape(-1)
        ex1 = x1.index_select(0, sel).reshape(-1)
        ey1 = y1.index_select(0, sel).reshape(-1)
        evis = visible.index_select(0, sel).reshape(-1)
        efin = finite.index_select(0, sel).reshape(-1)

        gm_min_u = min_u.index_select(0, sel).reshape(-1)
        gm_min_v = min_v.index_select(0, sel).reshape(-1)
        gm_cell_w = cell_w.index_select(0, sel).reshape(-1)
        gm_cell_h = cell_h.index_select(0, sel).reshape(-1)
        gm_half_diag = half_diag.index_select(0, sel).reshape(-1)
        gm_margin = margin.index_select(0, sel).reshape(-1)
        gm_cap = dist_cap.index_select(0, sel).reshape(-1)

        e_min_u = gm_min_u[g_local]
        e_min_v = gm_min_v[g_local]
        e_cell_w = gm_cell_w[g_local]
        e_cell_h = gm_cell_h[g_local]
        e_reach = (gm_cap + gm_half_diag + gm_margin)[g_local]

        lo_u = torch.minimum(ex0, ex1)
        hi_u = torch.maximum(ex0, ex1)
        lo_v = torch.minimum(ey0, ey1)
        hi_v = torch.maximum(ey0, ey1)

        # --- Truncated distance field (edge -> nearby cell centers) --------
        cx0 = torch.floor((lo_u - e_reach - e_min_u) / e_cell_w).long()
        cx1 = torch.floor((hi_u + e_reach - e_min_u) / e_cell_w).long()
        cy0 = torch.floor((lo_v - e_reach - e_min_v) / e_cell_h).long()
        cy1 = torch.floor((hi_v + e_reach - e_min_v) / e_cell_h).long()
        for vals in (cx0, cx1, cy0, cy1):
            vals.clamp_(0, G - 1)
        nx = cx1 - cx0 + 1
        ny = cy1 - cy0 + 1
        pair_counts = torch.where(efin, nx * ny, torch.zeros_like(nx))
        total_pairs = int(pair_counts.sum().item())
        if total_pairs > _CLASS_PAIR_HARD_CAP:
            # Over-budget chunk: leave the safe all-zero classification.
            continue
        sources = torch.nonzero(efin, as_tuple=False).flatten()
        expanded, local = _expand_intervals(sources, pair_counts)

        n_groups_chunk = S * C
        d_all = (gm_cap.view(-1, 1).expand(n_groups_chunk, cells)
                 .reshape(-1).clone())
        d_vis = d_all.clone()
        if expanded.numel() > 0:
            # Free each expansion-sized ([pairs]) temporary as soon as it is
            # consumed: this scratch is the only multi-MB transient of the
            # build and it competes with the merge for pool headroom.
            pnx = nx[expanded]
            pcx = cx0[expanded] + local % pnx
            pcy = cy0[expanded] + local // pnx
            pg = g_local[expanded]
            del local, pnx
            key = pg * cells + pcy * G + pcx
            ucen = (gm_min_u[pg]
                    + (pcx.to(dtype) + 0.5) * gm_cell_w[pg])
            vcen = (gm_min_v[pg]
                    + (pcy.to(dtype) + 0.5) * gm_cell_h[pg])
            del pcx, pcy
            sx0 = ex0[expanded]
            sy0 = ey0[expanded]
            dx = ex1[expanded] - sx0
            dy = ey1[expanded] - sy0
            seg_t = (((ucen - sx0) * dx + (vcen - sy0) * dy)
                     / (dx * dx + dy * dy).clamp_min(1e-12)).clamp_(0.0, 1.0)
            du = sx0 + seg_t * dx - ucen
            dv = sy0 + seg_t * dy - vcen
            del sx0, sy0, dx, dy, seg_t, ucen, vcen
            d_adj = (torch.sqrt(du * du + dv * dv)
                     - gm_half_diag[pg] - gm_margin[pg]).clamp_min(0.0)
            del du, dv
            vis_mask = evis[expanded]
            del expanded, pg
            d_all.scatter_reduce_(0, key, d_adj, reduce="amin",
                                  include_self=True)
            if bool(vis_mask.any()):
                d_vis.scatter_reduce_(0, key[vis_mask], d_adj[vis_mask],
                                      reduce="amin", include_self=True)
            del key, d_adj, vis_mask

        # --- Crossing parity at cell centers (difference-array prefix) -----
        nonh = efin & (ey0 != ey1)
        ry0 = (torch.floor((lo_v - e_min_v) / e_cell_h - 0.5).long()
               - 1).clamp_(0, G - 1)
        ry1 = (torch.ceil((hi_v - e_min_v) / e_cell_h - 0.5).long()
               + 1).clamp_(0, G - 1)
        row_counts = torch.where(nonh, (ry1 - ry0 + 1).clamp_min(0),
                                 torch.zeros_like(ry0))
        row_sources = torch.nonzero(nonh, as_tuple=False).flatten()
        row_expanded, row_local = _expand_intervals(row_sources, row_counts)
        diff = torch.zeros((n_groups_chunk * G * (G + 1),),
                           dtype=torch.int32, device=device)
        if row_expanded.numel() > 0:
            ry = ry0[row_expanded] + row_local
            rg = g_local[row_expanded]
            vc = gm_min_v[rg] + (ry.to(dtype) + 0.5) * gm_cell_h[rg]
            ry0_ = ey0[row_expanded]
            ry1_ = ey1[row_expanded]
            # The exact kernel predicate re-evaluated per candidate row makes
            # the generous row expansion above safe against float boundaries.
            crossing = (ry0_ > vc) != (ry1_ > vc)
            if bool(crossing.any()):
                ry = ry[crossing]
                rg = rg[crossing]
                vc = vc[crossing]
                rx0 = ex0[row_expanded][crossing]
                ry0_ = ry0_[crossing]
                rx1 = ex1[row_expanded][crossing]
                ry1_ = ry1_[crossing]
                x_cross = rx0 + (vc - ry0_) * (rx1 - rx0) / (ry1_ - ry0_)
                kx = torch.ceil((x_cross - gm_min_u[rg]) / gm_cell_w[rg]
                                - 0.5).long().clamp_(0, G)
                base = rg * (G * (G + 1)) + ry * (G + 1)
                ones = torch.ones_like(base, dtype=torch.int32)
                diff.scatter_add_(0, base, ones)
                diff.scatter_add_(0, base + kx, -ones)
        counts = diff.view(n_groups_chunk, G, G + 1).cumsum(
            -1, dtype=torch.int32)[..., :G]
        parity = (counts & 1).reshape(n_groups_chunk, cells)

        uniform = (d_all.view(n_groups_chunk, cells) > 0.0)
        flags = uniform.to(torch.int32) | (parity << 1)
        d_vis = d_vis.clamp_min(0.0)
        dist_sq = (d_vis * d_vis).to(torch.float32).view(
            n_groups_chunk, cells)

        dist_bits = dist_sq.view(torch.int32)
        for i, t in enumerate(chunk):
            flags_out[t * C:(t + 1) * C] = flags[i * C:(i + 1) * C]
            dist_out[t * C:(t + 1) * C] = dist_bits[i * C:(i + 1) * C]

    # Broadcast each representative frame's rows to its duplicate frames.
    for t in range(T):
        r = rep_of_class[int(frame_class[t])]
        if r != t:
            flags_out[t * C:(t + 1) * C] = flags_out[r * C:(r + 1) * C]
            dist_out[t * C:(t + 1) * C] = dist_out[r * C:(r + 1) * C]

    return flags_out, dist_out


def build_bezier_edge_acceleration(
        edges_2d: torch.Tensor,
        edge_offsets: torch.Tensor,
) -> torch.Tensor:
    """Build packed scanline and spatial-bin tables for sampled circuit edges.

    Parameters
    ----------
    edges_2d:
        ``[T, E, 5]`` float tensor containing ``x0, y0, x1, y1`` and the
        border-visible flag.
    edge_offsets:
        ``[C + 1]`` integer offsets delimiting each circuit's edge range.

    Returns
    -------
    torch.Tensor
        Flat contiguous ``int32`` buffer decoded by the Taichi kernels.
    """
    if edges_2d.ndim != 3 or edges_2d.shape[-1] < 5:
        raise ValueError(
            f"edges_2d must have shape [T, E, >=5], got {tuple(edges_2d.shape)}")
    if edge_offsets.ndim != 1 or edge_offsets.numel() < 2:
        raise ValueError(
            f"edge_offsets must have shape [C + 1], got {tuple(edge_offsets.shape)}")

    device = edges_2d.device
    offsets = edge_offsets.to(device=device, dtype=torch.long)
    num_frames, num_edges = edges_2d.shape[:2]
    num_circuits = offsets.numel() - 1
    if int(offsets[0].item()) != 0 or int(offsets[-1].item()) != num_edges:
        raise ValueError(
            "edge_offsets must start at zero and end at edges_2d.shape[1]")
    edge_counts = offsets[1:] - offsets[:-1]
    if bool((edge_counts < 0).any()):
        raise ValueError("edge_offsets must be monotonically non-decreasing")

    circuit_of_edge = torch.repeat_interleave(
        torch.arange(num_circuits, device=device), edge_counts)
    if circuit_of_edge.numel() != num_edges:
        raise ValueError("edge_offsets do not describe every sampled edge")

    x0 = edges_2d[..., 0]
    y0 = edges_2d[..., 1]
    x1 = edges_2d[..., 2]
    y1 = edges_2d[..., 3]
    edge_lo_u = torch.minimum(x0, x1)
    edge_lo_v = torch.minimum(y0, y1)
    edge_hi_u = torch.maximum(x0, x1)
    edge_hi_v = torch.maximum(y0, y1)

    # Degenerate segments are encoded as 1e9 sentinels by the primitive
    # builder.  Exclude them from bounds and both tables exactly as the old
    # point loop effectively did (their crossing is false and border flag 0).
    finite = (torch.isfinite(edge_lo_u) & torch.isfinite(edge_lo_v)
              & torch.isfinite(edge_hi_u) & torch.isfinite(edge_hi_v)
              & (edge_lo_u.abs() < 1e8) & (edge_lo_v.abs() < 1e8)
              & (edge_hi_u.abs() < 1e8) & (edge_hi_v.abs() < 1e8))

    scatter_index = circuit_of_edge.view(1, -1).expand(num_frames, -1)
    inf = torch.tensor(float("inf"), device=device, dtype=edges_2d.dtype)
    neg_inf = torch.tensor(float("-inf"), device=device,
                           dtype=edges_2d.dtype)

    def reduce_bounds(values: torch.Tensor, valid: torch.Tensor,
                      reduce: str) -> torch.Tensor:
        initial = inf if reduce == "amin" else neg_inf
        prepared = torch.where(valid, values, initial)
        out = torch.full((num_frames, num_circuits), initial,
                         dtype=edges_2d.dtype, device=device)
        out.scatter_reduce_(1, scatter_index, prepared, reduce=reduce,
                            include_self=True)
        return out

    min_u = reduce_bounds(edge_lo_u, finite, "amin")
    min_v = reduce_bounds(edge_lo_v, finite, "amin")
    max_u = reduce_bounds(edge_hi_u, finite, "amax")
    max_v = reduce_bounds(edge_hi_v, finite, "amax")
    empty_circuit = ~torch.isfinite(min_u) | ~torch.isfinite(min_v)
    min_u = torch.where(empty_circuit, torch.zeros_like(min_u), min_u)
    min_v = torch.where(empty_circuit, torch.zeros_like(min_v), min_v)
    max_u = torch.where(empty_circuit, min_u + 1.0, max_u)
    max_v = torch.where(empty_circuit, min_v + 1.0, max_v)

    extent_u = (max_u - min_u).clamp_min(1e-12)
    extent_v = (max_v - min_v).clamp_min(1e-12)
    grid_inv_u = BEZIER_SPATIAL_GRID / extent_u
    grid_inv_v = BEZIER_SPATIAL_GRID / extent_v
    scan_inv_v = BEZIER_SCAN_BINS / extent_v

    num_groups = num_frames * num_circuits
    frame_ids = torch.arange(num_frames, device=device).view(-1, 1).expand(
        -1, num_edges).reshape(-1)
    edge_ids = torch.arange(num_edges, device=device).view(1, -1).expand(
        num_frames, -1).reshape(-1)
    circuit_ids = circuit_of_edge.view(1, -1).expand(
        num_frames, -1).reshape(-1)
    group_ids = frame_ids * num_circuits + circuit_ids

    flat_finite = finite.reshape(-1)
    flat_lo_u = edge_lo_u.reshape(-1)
    flat_lo_v = edge_lo_v.reshape(-1)
    flat_hi_u = edge_hi_u.reshape(-1)
    flat_hi_v = edge_hi_v.reshape(-1)
    flat_y0 = y0.reshape(-1)
    flat_y1 = y1.reshape(-1)
    group_min_u = min_u.reshape(-1)[group_ids]
    group_min_v = min_v.reshape(-1)[group_ids]
    group_grid_inv_u = grid_inv_u.reshape(-1)[group_ids]
    group_grid_inv_v = grid_inv_v.reshape(-1)[group_ids]
    group_scan_inv_v = scan_inv_v.reshape(-1)[group_ids]

    # Scanline table.  Including the bin containing the upper endpoint may add
    # a harmless extra candidate; the exact half-open crossing predicate still
    # runs in the kernel, preserving the original classification at bin edges.
    scan_start = torch.floor(
        (flat_lo_v - group_min_v) * group_scan_inv_v).to(torch.long)
    scan_end = torch.floor(
        (flat_hi_v - group_min_v) * group_scan_inv_v).to(torch.long)
    scan_start.clamp_(0, BEZIER_SCAN_BINS - 1)
    scan_end.clamp_(0, BEZIER_SCAN_BINS - 1)
    scan_valid = flat_finite & (flat_y0 != flat_y1)
    scan_counts_per_edge = (scan_end - scan_start + 1).clamp_min(0)
    scan_sources = torch.nonzero(scan_valid, as_tuple=False).flatten()
    scan_expanded, scan_local = _expand_intervals(
        scan_sources, scan_counts_per_edge)
    if scan_expanded.numel() > 0:
        scan_bins = scan_start[scan_expanded] + scan_local
        scan_keys = group_ids[scan_expanded] * BEZIER_SCAN_BINS + scan_bins
        scan_order, scan_offsets = _grouped_offsets(
            scan_keys, num_groups, BEZIER_SCAN_BINS)
        scan_refs = edge_ids[scan_expanded][scan_order].to(torch.int32)
    else:
        scan_refs = torch.empty((0,), dtype=torch.int32, device=device)
        scan_offsets = torch.zeros(
            (num_groups, BEZIER_SCAN_BINS + 1), dtype=torch.long,
            device=device)

    # Spatial border table.  Each visible edge is inserted into every uniform
    # grid cell touched by its endpoint AABB.  A radius query checks all cells
    # touched by the query square, which is a conservative superset of every
    # segment that could be nearer than that radius.
    spatial_x0 = torch.floor(
        (flat_lo_u - group_min_u) * group_grid_inv_u).to(torch.long)
    spatial_y0 = torch.floor(
        (flat_lo_v - group_min_v) * group_grid_inv_v).to(torch.long)
    spatial_x1 = torch.floor(
        (flat_hi_u - group_min_u) * group_grid_inv_u).to(torch.long)
    spatial_y1 = torch.floor(
        (flat_hi_v - group_min_v) * group_grid_inv_v).to(torch.long)
    for values in (spatial_x0, spatial_y0, spatial_x1, spatial_y1):
        values.clamp_(0, BEZIER_SPATIAL_GRID - 1)
    nx = (spatial_x1 - spatial_x0 + 1).clamp_min(0)
    ny = (spatial_y1 - spatial_y0 + 1).clamp_min(0)
    spatial_counts_per_edge = nx * ny
    border_visible = edges_2d[..., 4].reshape(-1) > 0.5
    spatial_valid = flat_finite & border_visible
    spatial_sources = torch.nonzero(spatial_valid,
                                    as_tuple=False).flatten()
    spatial_expanded, spatial_local = _expand_intervals(
        spatial_sources, spatial_counts_per_edge)
    if spatial_expanded.numel() > 0:
        expanded_nx = nx[spatial_expanded]
        cell_x = spatial_x0[spatial_expanded] + spatial_local % expanded_nx
        cell_y = spatial_y0[spatial_expanded] + spatial_local // expanded_nx
        cells = cell_y * BEZIER_SPATIAL_GRID + cell_x
        spatial_keys = (group_ids[spatial_expanded] * BEZIER_SPATIAL_CELLS
                        + cells)
        spatial_order, spatial_offsets = _grouped_offsets(
            spatial_keys, num_groups, BEZIER_SPATIAL_CELLS)
        spatial_refs = edge_ids[spatial_expanded][spatial_order].to(
            torch.int32)
    else:
        spatial_refs = torch.empty((0,), dtype=torch.int32, device=device)
        spatial_offsets = torch.zeros(
            (num_groups, BEZIER_SPATIAL_CELLS + 1), dtype=torch.long,
            device=device)

    header_words = num_groups * BEZIER_ACCEL_HEADER_SIZE
    scan_base = header_words
    spatial_base = scan_base + scan_refs.numel()
    class_base = spatial_base + spatial_refs.numel()

    # Cell classification section: pick the largest grid dimension whose
    # [num_groups, dim, dim, 2] table fits the word budget; halve it for very
    # circuit- or frame-heavy batches, and drop the table entirely (header
    # base -1 -> kernels keep the exact loops) when even the smallest grid
    # cannot fit.
    class_dim = 0
    if BEZIER_CLASS_ENABLED:
        class_dim = BEZIER_CLASS_GRID
        while (class_dim > 4
               and num_groups * class_dim * class_dim * 2
               > BEZIER_CLASS_MAX_WORDS):
            class_dim //= 2
        if num_groups * class_dim * class_dim * 2 > BEZIER_CLASS_MAX_WORDS:
            class_dim = 0
    if class_dim:
        cls_flags, cls_dist = _build_cell_classification(
            edges_2d, circuit_of_edge, finite, min_u, min_v, max_u, max_v,
            class_dim)
        class_words = torch.stack((cls_flags, cls_dist), dim=-1).reshape(-1)
        class_bases = (class_base + torch.arange(
            num_groups, device=device, dtype=torch.long)
            * (class_dim * class_dim * 2)).to(torch.int32)
        class_inv_u = class_dim / extent_u
        class_inv_v = class_dim / extent_v
    else:
        class_words = torch.empty((0,), dtype=torch.int32, device=device)
        class_bases = torch.full((num_groups,), -1, dtype=torch.int32,
                                 device=device)
        class_inv_u = torch.zeros_like(extent_u)
        class_inv_v = torch.zeros_like(extent_v)

    total_words = class_base + class_words.numel()
    if total_words >= 2 ** 31:
        raise OverflowError(
            "Bezier acceleration buffer exceeds int32 addressable range")

    headers = torch.empty((num_groups, BEZIER_ACCEL_HEADER_SIZE),
                          dtype=torch.int32, device=device)
    headers[:, BEZIER_EDGE_START] = offsets[:-1].to(torch.int32).repeat(
        num_frames)
    headers[:, BEZIER_EDGE_END] = offsets[1:].to(torch.int32).repeat(
        num_frames)
    float_meta = torch.stack((
        min_u, min_v, max_u, max_v, grid_inv_u, grid_inv_v, scan_inv_v,
        class_inv_u, class_inv_v,
    ), dim=-1).reshape(num_groups, 9)
    headers[:, BEZIER_MIN_U:BEZIER_CLASS_INV_V + 1] = _float_bits(float_meta)
    headers[:, BEZIER_CLASS_BASE] = class_bases
    headers[:, BEZIER_CLASS_DIM] = class_dim
    headers[:, BEZIER_SCAN_OFFSET_BASE:BEZIER_SPATIAL_OFFSET_BASE] = (
        scan_offsets + scan_base).to(torch.int32)
    headers[:, BEZIER_SPATIAL_OFFSET_BASE:BEZIER_ACCEL_HEADER_SIZE] = (
        spatial_offsets + spatial_base).to(torch.int32)

    return torch.cat((headers.reshape(-1), scan_refs, spatial_refs,
                      class_words), dim=0).contiguous()
