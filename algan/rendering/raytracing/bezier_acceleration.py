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

A batch holds a header for every (frame, circuit) group, but a circuit whose
sampled edges are the same in every frame of the batch -- text that only moves,
rotates or fades, most of a network's synapses while one of them pulses -- has
its bins built once and shared by all of its frames. On the backprop video's
network scenes 99.8% of circuits were static, and a 139-frame batch went from
17 s and an 11.6 GB peak to 0.01 s (``build_bezier_edge_acceleration``).
"""

from __future__ import annotations

import torch

from algan.environment import env_int
from algan.rendering.mps_compat import clamp_floor

# Resolution of the two indices. Both are pure acceleration: a query that a
# coarser index sends more edges to still exact-tests every one of them, so
# neither moves rendered output. Finer bins prune harder and cost header
# memory per (edge frame, circuit) record -- the header below is sized from
# them -- so they are exposed for tuning against glyph-heavy scenes. Read once
# here, and the kernels read the derived offsets, so host and device always
# agree on one layout.
bezier_scan_bins = max(1, env_int("ALGAN_BEZIER_SCAN_BINS", 16))
bezier_spatial_grid = max(1, env_int("ALGAN_BEZIER_SPATIAL_GRID", 8))
BEZIER_SPATIAL_CELLS = bezier_spatial_grid * bezier_spatial_grid

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
BEZIER_SPATIAL_OFFSET_BASE = BEZIER_SCAN_OFFSET_BASE + bezier_scan_bins + 1
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


def _grouped_offsets(
    keys: torch.Tensor, num_groups: int, bins_per_group: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort flat integer ``keys`` and return order + per-group CSR offsets.

    The returned offsets are relative to the beginning of the sorted reference
    list and have shape ``[num_groups, bins_per_group + 1]``.
    """
    num_keys = num_groups * bins_per_group
    if keys.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.long, device=keys.device),
            torch.zeros(
                (num_groups, bins_per_group + 1), dtype=torch.long, device=keys.device
            ),
        )

    # Stable, so the edges within a bin keep their ascending edge order. What
    # the kernels compute can depend on the order they visit a bin's edges,
    # and an unstable sort orders equal keys by the whole batch's layout: the
    # same frame rendered in a different batch moved a corner pixel.
    order = torch.argsort(keys, stable=True)
    counts = torch.bincount(keys, minlength=num_keys).reshape(
        num_groups, bins_per_group
    )
    group_totals = counts.sum(dim=1)
    group_bases = group_totals.cumsum(dim=0) - group_totals
    local_prefix = torch.cat(
        (
            torch.zeros((num_groups, 1), dtype=torch.long, device=keys.device),
            counts.cumsum(dim=1),
        ),
        dim=1,
    )
    return order, local_prefix + group_bases.unsqueeze(1)


def _expand_intervals(
    source: torch.Tensor, counts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
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
    local_offsets = torch.arange(total, device=source.device) - torch.repeat_interleave(
        interval_starts, selected_counts
    )
    return expanded_source, local_offsets


#: Upper bound on the (frame, edge) pairs one construction chunk holds. Every
#: pair carries about twenty int64/float temporaries plus its expanded bin
#: entries, so a chunk of this size peaks near a gigabyte; building a whole
#: batch at once (a 277-frame batch of the backprop video's network: 30 M pairs)
#: oversubscribed a 16 GB card and spent seconds paging. Chunks never change
#: the tables, only how many pairs are in flight.
_CHUNK_PAIRS = 1 << 22

#: The integer type each float width is compared through, bit for bit.
_BIT_VIEWS = {2: torch.int16, 4: torch.int32, 8: torch.int64}


def _static_circuits(
    edges_2d: torch.Tensor, circuit_of_edge: torch.Tensor, num_circuits: int
) -> torch.Tensor:
    """``[C]`` bool: whether every edge of a circuit is the same in every frame.

    *The same* is bit for bit, compared through an integer view: ``-0.0``
    against ``0.0`` is a change, a NaN repeated with the same bits is not. A
    static circuit's tables at a later frame are then exactly its frame-0
    tables, which is what lets :func:`build_bezier_edge_acceleration` build
    them once.
    """
    num_frames, num_edges, cols = edges_2d.shape
    bits = edges_2d.contiguous().view(_BIT_VIEWS[edges_2d.element_size()])
    same = torch.ones((num_edges,), dtype=torch.bool, device=edges_2d.device)
    first = bits[:1]
    # Compared a slice of frames at a time, so the boolean scratch stays near
    # 64 M elements however long the batch is.
    step = max(1, (64 << 20) // max(1, num_edges * cols))
    for f0 in range(1, num_frames, step):
        same &= (bits[f0 : f0 + step] == first).all(-1).all(0)
    changed = torch.zeros((num_circuits,), dtype=torch.long, device=edges_2d.device)
    changed.index_add_(0, circuit_of_edge, (~same).long())
    return changed == 0


def _build_group_tables(
    edges_2d: torch.Tensor,
    frame_ids: torch.Tensor,
    edge_ids: torch.Tensor,
    group_ids: torch.Tensor,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bounds and both CSR tables for ``num_groups`` (frame, circuit) groups.

    ``frame_ids``/``edge_ids`` list every edge of every group, each group's
    edges in ascending order, and ``group_ids`` says which group (0 to
    ``num_groups - 1``) each pair belongs to. Returns the ``[G, 7]`` float
    metadata (min u, min v, max u, max v and the three inverse bin sizes),
    the ``[G, bins + 1]`` scanline and ``[G, cells + 1]`` spatial offsets --
    relative to the start of the reference lists returned beside them -- and
    the two lists of edge ids.
    """
    device = edges_2d.device
    values = edges_2d[frame_ids, edge_ids]
    x0 = values[:, 0]
    y0 = values[:, 1]
    x1 = values[:, 2]
    y1 = values[:, 3]
    edge_lo_u = torch.minimum(x0, x1)
    edge_lo_v = torch.minimum(y0, y1)
    edge_hi_u = torch.maximum(x0, x1)
    edge_hi_v = torch.maximum(y0, y1)

    # Degenerate segments are encoded as 1e9 sentinels by the primitive
    # builder.  Exclude them from bounds and both tables exactly as the old
    # point loop effectively did (their crossing is false and border flag 0).
    finite = (
        torch.isfinite(edge_lo_u)
        & torch.isfinite(edge_lo_v)
        & torch.isfinite(edge_hi_u)
        & torch.isfinite(edge_hi_v)
        & (edge_lo_u.abs() < 1e8)
        & (edge_lo_v.abs() < 1e8)
        & (edge_hi_u.abs() < 1e8)
        & (edge_hi_v.abs() < 1e8)
    )

    inf = torch.tensor(float("inf"), device=device, dtype=edges_2d.dtype)
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=edges_2d.dtype)

    def reduce_bounds(values: torch.Tensor, reduce: str) -> torch.Tensor:
        initial = inf if reduce == "amin" else neg_inf
        prepared = torch.where(finite, values, initial)
        out = torch.full((num_groups,), initial, dtype=edges_2d.dtype, device=device)
        out.scatter_reduce_(0, group_ids, prepared, reduce=reduce, include_self=True)
        return out

    min_u = reduce_bounds(edge_lo_u, "amin")
    min_v = reduce_bounds(edge_lo_v, "amin")
    max_u = reduce_bounds(edge_hi_u, "amax")
    max_v = reduce_bounds(edge_hi_v, "amax")
    empty_circuit = ~torch.isfinite(min_u) | ~torch.isfinite(min_v)
    min_u = torch.where(empty_circuit, torch.zeros_like(min_u), min_u)
    min_v = torch.where(empty_circuit, torch.zeros_like(min_v), min_v)
    max_u = torch.where(empty_circuit, min_u + 1.0, max_u)
    max_v = torch.where(empty_circuit, min_v + 1.0, max_v)

    extent_u = clamp_floor(max_u - min_u, 1e-12)
    extent_v = clamp_floor(max_v - min_v, 1e-12)
    grid_inv_u = bezier_spatial_grid / extent_u
    grid_inv_v = bezier_spatial_grid / extent_v
    scan_inv_v = bezier_scan_bins / extent_v
    float_meta = torch.stack(
        (min_u, min_v, max_u, max_v, grid_inv_u, grid_inv_v, scan_inv_v), dim=-1
    )

    group_min_u = min_u[group_ids]
    group_min_v = min_v[group_ids]

    # Scanline table.  Including the bin containing the upper endpoint may add
    # a harmless extra candidate; the exact half-open crossing predicate still
    # runs in the kernel, preserving the original classification at bin edges.
    group_scan_inv_v = scan_inv_v[group_ids]
    scan_start = torch.floor((edge_lo_v - group_min_v) * group_scan_inv_v).to(
        torch.long
    )
    scan_end = torch.floor((edge_hi_v - group_min_v) * group_scan_inv_v).to(torch.long)
    del group_scan_inv_v
    scan_start.clamp_(0, bezier_scan_bins - 1)
    scan_end.clamp_(0, bezier_scan_bins - 1)
    scan_valid = finite & (y0 != y1)
    scan_counts_per_edge = (scan_end - scan_start + 1).clamp_min(0)
    del scan_end
    scan_sources = torch.nonzero(scan_valid, as_tuple=False).flatten()
    scan_expanded, scan_local = _expand_intervals(scan_sources, scan_counts_per_edge)
    del scan_valid, scan_counts_per_edge, scan_sources
    if scan_expanded.numel() > 0:
        scan_bins = scan_start[scan_expanded] + scan_local
        scan_keys = group_ids[scan_expanded] * bezier_scan_bins + scan_bins
        del scan_bins
        scan_order, scan_offsets = _grouped_offsets(
            scan_keys, num_groups, bezier_scan_bins
        )
        del scan_keys
        scan_refs = edge_ids[scan_expanded][scan_order].to(torch.int32)
        del scan_order
    else:
        scan_refs = torch.empty((0,), dtype=torch.int32, device=device)
        scan_offsets = torch.zeros(
            (num_groups, bezier_scan_bins + 1), dtype=torch.long, device=device
        )
    del scan_start, scan_expanded, scan_local

    # Spatial border table.  Each visible edge is inserted into every uniform
    # grid cell touched by its endpoint AABB.  A radius query checks all cells
    # touched by the query square, which is a conservative superset of every
    # segment that could be nearer than that radius.
    group_grid_inv_u = grid_inv_u[group_ids]
    group_grid_inv_v = grid_inv_v[group_ids]
    spatial_x0 = torch.floor((edge_lo_u - group_min_u) * group_grid_inv_u).to(
        torch.long
    )
    spatial_y0 = torch.floor((edge_lo_v - group_min_v) * group_grid_inv_v).to(
        torch.long
    )
    spatial_x1 = torch.floor((edge_hi_u - group_min_u) * group_grid_inv_u).to(
        torch.long
    )
    spatial_y1 = torch.floor((edge_hi_v - group_min_v) * group_grid_inv_v).to(
        torch.long
    )
    del group_grid_inv_u, group_grid_inv_v, group_min_u, group_min_v
    for coordinate in (spatial_x0, spatial_y0, spatial_x1, spatial_y1):
        coordinate.clamp_(0, bezier_spatial_grid - 1)
    nx = (spatial_x1 - spatial_x0 + 1).clamp_min(0)
    ny = (spatial_y1 - spatial_y0 + 1).clamp_min(0)
    del spatial_x1, spatial_y1
    spatial_counts_per_edge = nx * ny
    del ny
    border_visible = values[:, 4] > 0.5
    spatial_valid = finite & border_visible
    spatial_sources = torch.nonzero(spatial_valid, as_tuple=False).flatten()
    del border_visible, spatial_valid
    spatial_expanded, spatial_local = _expand_intervals(
        spatial_sources, spatial_counts_per_edge
    )
    del spatial_sources, spatial_counts_per_edge
    if spatial_expanded.numel() > 0:
        expanded_nx = nx[spatial_expanded]
        cell_x = spatial_x0[spatial_expanded] + spatial_local % expanded_nx
        cell_y = spatial_y0[spatial_expanded] + spatial_local // expanded_nx
        del expanded_nx
        cells = cell_y * bezier_spatial_grid + cell_x
        del cell_x, cell_y
        spatial_keys = group_ids[spatial_expanded] * BEZIER_SPATIAL_CELLS + cells
        del cells
        spatial_order, spatial_offsets = _grouped_offsets(
            spatial_keys, num_groups, BEZIER_SPATIAL_CELLS
        )
        del spatial_keys
        spatial_refs = edge_ids[spatial_expanded][spatial_order].to(torch.int32)
    else:
        spatial_refs = torch.empty((0,), dtype=torch.int32, device=device)
        spatial_offsets = torch.zeros(
            (num_groups, BEZIER_SPATIAL_CELLS + 1), dtype=torch.long, device=device
        )
    return float_meta, scan_offsets, scan_refs, spatial_offsets, spatial_refs


def _dedup_disabled() -> bool:
    """``ALGAN_OPT_DISABLE=acceldedup``: build every (frame, circuit) group."""
    from algan.animation_timeline.timeline import _opt_disabled

    return _opt_disabled("acceldedup")


def build_bezier_edge_acceleration(
    edges_2d: torch.Tensor,
    edge_offsets: torch.Tensor,
) -> torch.Tensor:
    """Build packed scanline and spatial-bin tables for sampled circuit edges.

    Every (frame, circuit) group gets a header -- its edge range, bounds and
    the offsets of its bins -- and the kernels read group ``f * C + c``'s
    header, follow its offsets to edge ids, and read those edges from
    ``edges_2d[f]`` themselves. A circuit whose edges are the same in every
    frame (:func:`_static_circuits`) therefore needs its bins only once: its
    header at every later frame is a copy of its frame-0 header, pointing at
    the edge ids, in the order, that its own bins would have held. Only frame
    0 and the circuits that change go through construction, in chunks of at
    most ``_CHUNK_PAIRS`` pairs. ``ALGAN_OPT_DISABLE=acceldedup`` builds every
    group, as one chunk, which is the buffer this function returned before.

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
            f"edges_2d must have shape [T, E, >=5], got {tuple(edges_2d.shape)}"
        )
    if edge_offsets.ndim != 1 or edge_offsets.numel() < 2:
        raise ValueError(
            f"edge_offsets must have shape [C + 1], got {tuple(edge_offsets.shape)}"
        )

    device = edges_2d.device
    offsets = edge_offsets.to(device=device, dtype=torch.long)
    num_frames, num_edges = edges_2d.shape[:2]
    num_circuits = offsets.numel() - 1
    if int(offsets[0].item()) != 0 or int(offsets[-1].item()) != num_edges:
        raise ValueError("edge_offsets must start at zero and end at edges_2d.shape[1]")
    edge_counts = offsets[1:] - offsets[:-1]
    if bool((edge_counts < 0).any()):
        raise ValueError("edge_offsets must be monotonically non-decreasing")

    circuit_of_edge = torch.repeat_interleave(
        torch.arange(num_circuits, device=device), edge_counts
    )
    if circuit_of_edge.numel() != num_edges:
        raise ValueError("edge_offsets do not describe every sampled edge")

    num_groups = num_frames * num_circuits
    if num_frames == 0:
        return torch.empty((0,), dtype=torch.int32, device=device)
    if _dedup_disabled():
        # Every group built, in the frame-major pair order of one flattened
        # [T, E] grid: the buffer this function returned before the dedup.
        frame_ids = (
            torch.arange(num_frames, device=device)
            .view(-1, 1)
            .expand(-1, num_edges)
            .reshape(-1)
        )
        edge_ids = (
            torch.arange(num_edges, device=device)
            .view(1, -1)
            .expand(num_frames, -1)
            .reshape(-1)
        )
        group_ids = frame_ids * num_circuits + circuit_of_edge[edge_ids]
        chunks = [(frame_ids, edge_ids, group_ids, num_groups)]
        static = torch.zeros((num_circuits,), dtype=torch.bool, device=device)
        dynamic_ids = torch.arange(num_circuits, device=device)
        compact_of_group = torch.arange(num_groups, device=device)
        del frame_ids, edge_ids, group_ids
    else:
        static = _static_circuits(edges_2d, circuit_of_edge, num_circuits)
        dynamic = ~static
        dynamic_ids = torch.nonzero(dynamic, as_tuple=False).flatten()
        num_dynamic = dynamic_ids.numel()
        dynamic_rank = torch.cumsum(dynamic.long(), dim=0) - 1
        dynamic_edges = torch.nonzero(
            dynamic[circuit_of_edge], as_tuple=False
        ).flatten()
        # Groups are numbered compactly: frame 0's C circuits, then each later
        # frame's dynamic circuits, frame-major. Each chunk covers a contiguous
        # run of that numbering and lists its groups' pairs frame-major,
        # ascending in edge -- the order the bins' stable sort keeps.
        chunks = [
            (
                torch.zeros((num_edges,), dtype=torch.long, device=device),
                torch.arange(num_edges, device=device),
                circuit_of_edge,
                num_circuits,
            )
        ]
        if num_dynamic:
            frames_per_chunk = max(1, _CHUNK_PAIRS // max(1, dynamic_edges.numel()))
            edge_rank = dynamic_rank[circuit_of_edge[dynamic_edges]]
            for f0 in range(1, num_frames, frames_per_chunk):
                count = min(num_frames, f0 + frames_per_chunk) - f0
                frame_ids = torch.arange(f0, f0 + count, device=device)
                chunks.append(
                    (
                        frame_ids.repeat_interleave(dynamic_edges.numel()),
                        dynamic_edges.repeat(count),
                        (
                            torch.arange(count, device=device) * num_dynamic
                        ).repeat_interleave(dynamic_edges.numel())
                        + edge_rank.repeat(count),
                        count * num_dynamic,
                    )
                )
        # A static circuit reads its frame-0 header in every frame.
        frame_of_group = torch.arange(num_frames, device=device).view(-1, 1)
        compact_of_group = torch.where(
            (frame_of_group == 0) | static.view(1, -1),
            torch.arange(num_circuits, device=device).view(1, -1),
            num_circuits
            + (frame_of_group - 1) * num_dynamic
            + dynamic_rank.view(1, -1),
        ).reshape(-1)
        del dynamic, dynamic_rank, dynamic_edges, frame_of_group

    num_compact = num_circuits + (num_frames - 1) * dynamic_ids.numel()
    compact = torch.empty(
        (num_compact, BEZIER_ACCEL_HEADER_SIZE), dtype=torch.int32, device=device
    )
    compact_circuit = torch.cat(
        (
            torch.arange(num_circuits, device=device),
            dynamic_ids.repeat(num_frames - 1),
        )
    )
    compact[:, BEZIER_EDGE_START] = offsets[:-1][compact_circuit].to(torch.int32)
    compact[:, BEZIER_EDGE_END] = offsets[1:][compact_circuit].to(torch.int32)
    del compact_circuit

    # Bin offsets are written relative to the chunk lists' running totals here;
    # the bases of the two reference regions are only known once every chunk
    # is built, and are added below.
    scan_refs, spatial_refs = [], []
    scan_total = spatial_total = 0
    row = 0
    while chunks:
        frame_ids, edge_ids, group_ids, count = chunks.pop(0)
        meta, scan_offsets, scan_list, spatial_offsets, spatial_list = (
            _build_group_tables(edges_2d, frame_ids, edge_ids, group_ids, count)
        )
        del frame_ids, edge_ids, group_ids
        rows = slice(row, row + count)
        compact[rows, BEZIER_MIN_U : BEZIER_SCAN_INV_V + 1] = _float_bits(meta)
        compact[rows, BEZIER_SCAN_OFFSET_BASE:BEZIER_SPATIAL_OFFSET_BASE] = (
            scan_offsets + scan_total
        ).to(torch.int32)
        compact[rows, BEZIER_SPATIAL_OFFSET_BASE:BEZIER_ACCEL_HEADER_SIZE] = (
            spatial_offsets + spatial_total
        ).to(torch.int32)
        del meta, scan_offsets, spatial_offsets
        scan_refs.append(scan_list)
        spatial_refs.append(spatial_list)
        scan_total += scan_list.numel()
        spatial_total += spatial_list.numel()
        row += count

    header_words = num_groups * BEZIER_ACCEL_HEADER_SIZE
    scan_base = header_words
    spatial_base = scan_base + scan_total
    total_words = spatial_base + spatial_total
    if total_words >= 2**31:
        raise OverflowError(
            "Bezier acceleration buffer exceeds int32 addressable range"
        )
    compact[:, BEZIER_SCAN_OFFSET_BASE:BEZIER_SPATIAL_OFFSET_BASE] += scan_base
    compact[:, BEZIER_SPATIAL_OFFSET_BASE:BEZIER_ACCEL_HEADER_SIZE] += spatial_base

    out = torch.empty((total_words,), dtype=torch.int32, device=device)
    torch.index_select(
        compact,
        0,
        compact_of_group,
        out=out[:header_words].view(num_groups, BEZIER_ACCEL_HEADER_SIZE),
    )
    del compact, compact_of_group
    cursor = scan_base
    for refs in (*scan_refs, *spatial_refs):
        out[cursor : cursor + refs.numel()] = refs
        cursor += refs.numel()
    return out
