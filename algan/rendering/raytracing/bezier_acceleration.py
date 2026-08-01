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
"""
from __future__ import annotations

import torch

BEZIER_SCAN_BINS = 16
BEZIER_SPATIAL_GRID = 8
BEZIER_SPATIAL_CELLS = BEZIER_SPATIAL_GRID * BEZIER_SPATIAL_GRID

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
BEZIER_SCAN_OFFSET_BASE = 9
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
    total_words = spatial_base + spatial_refs.numel()
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
    ), dim=-1).reshape(num_groups, 7)
    headers[:, BEZIER_MIN_U:BEZIER_SCAN_INV_V + 1] = _float_bits(float_meta)
    headers[:, BEZIER_SCAN_OFFSET_BASE:BEZIER_SPATIAL_OFFSET_BASE] = (
        scan_offsets + scan_base).to(torch.int32)
    headers[:, BEZIER_SPATIAL_OFFSET_BASE:BEZIER_ACCEL_HEADER_SIZE] = (
        spatial_offsets + spatial_base).to(torch.int32)

    return torch.cat((headers.reshape(-1), scan_refs, spatial_refs),
                     dim=0).contiguous()
