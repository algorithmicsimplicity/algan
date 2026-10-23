"""The Bezier edge tables are built once for circuits that do not change.

``build_bezier_edge_acceleration`` gives every (frame, circuit) group a header,
and the kernels reach a group's edges only through it: they read group
``f * C + c``'s bounds and bin offsets, follow the offsets to edge ids, and
read those edges from ``edges_2d[f]``. A circuit whose edges are bitwise the
same in every frame shares its frame-0 bins, and the rest are built in bounded
chunks. These tests decode a batch's tables exactly as the kernels address
them and hold the result to the tables the builder made before it shared
anything (``_reference_build``, a frozen copy of it), and hold
``ALGAN_OPT_DISABLE=acceldedup`` to that buffer word for word.
"""

from __future__ import annotations

import pytest
import torch

from algan.rendering.mps_compat import clamp_floor
from algan.rendering.raytracing import bezier_acceleration as ba
from algan.rendering.raytracing.bezier_acceleration import (
    BEZIER_ACCEL_HEADER_SIZE,
    BEZIER_EDGE_END,
    BEZIER_EDGE_START,
    BEZIER_MIN_U,
    BEZIER_SCAN_INV_V,
    BEZIER_SCAN_OFFSET_BASE,
    BEZIER_SPATIAL_CELLS,
    BEZIER_SPATIAL_OFFSET_BASE,
    _expand_intervals,
    _float_bits,
    _grouped_offsets,
    bezier_scan_bins,
    bezier_spatial_grid,
    build_bezier_edge_acceleration,
)


def _reference_build(
    edges_2d: torch.Tensor,
    edge_offsets: torch.Tensor,
) -> torch.Tensor:
    """The builder as it was before static circuits shared bins (frozen copy).

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

    scatter_index = circuit_of_edge.view(1, -1).expand(num_frames, -1)
    inf = torch.tensor(float("inf"), device=device, dtype=edges_2d.dtype)
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=edges_2d.dtype)

    def reduce_bounds(
        values: torch.Tensor, valid: torch.Tensor, reduce: str
    ) -> torch.Tensor:
        initial = inf if reduce == "amin" else neg_inf
        prepared = torch.where(valid, values, initial)
        out = torch.full(
            (num_frames, num_circuits), initial, dtype=edges_2d.dtype, device=device
        )
        out.scatter_reduce_(
            1, scatter_index, prepared, reduce=reduce, include_self=True
        )
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

    extent_u = clamp_floor(max_u - min_u, 1e-12)
    extent_v = clamp_floor(max_v - min_v, 1e-12)
    grid_inv_u = bezier_spatial_grid / extent_u
    grid_inv_v = bezier_spatial_grid / extent_v
    scan_inv_v = bezier_scan_bins / extent_v

    num_groups = num_frames * num_circuits
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
    circuit_ids = circuit_of_edge.view(1, -1).expand(num_frames, -1).reshape(-1)
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
    scan_start = torch.floor((flat_lo_v - group_min_v) * group_scan_inv_v).to(
        torch.long
    )
    scan_end = torch.floor((flat_hi_v - group_min_v) * group_scan_inv_v).to(torch.long)
    scan_start.clamp_(0, bezier_scan_bins - 1)
    scan_end.clamp_(0, bezier_scan_bins - 1)
    scan_valid = flat_finite & (flat_y0 != flat_y1)
    scan_counts_per_edge = (scan_end - scan_start + 1).clamp_min(0)
    scan_sources = torch.nonzero(scan_valid, as_tuple=False).flatten()
    scan_expanded, scan_local = _expand_intervals(scan_sources, scan_counts_per_edge)
    if scan_expanded.numel() > 0:
        scan_bins = scan_start[scan_expanded] + scan_local
        scan_keys = group_ids[scan_expanded] * bezier_scan_bins + scan_bins
        scan_order, scan_offsets = _grouped_offsets(
            scan_keys, num_groups, bezier_scan_bins
        )
        scan_refs = edge_ids[scan_expanded][scan_order].to(torch.int32)
    else:
        scan_refs = torch.empty((0,), dtype=torch.int32, device=device)
        scan_offsets = torch.zeros(
            (num_groups, bezier_scan_bins + 1), dtype=torch.long, device=device
        )

    # Spatial border table.  Each visible edge is inserted into every uniform
    # grid cell touched by its endpoint AABB.  A radius query checks all cells
    # touched by the query square, which is a conservative superset of every
    # segment that could be nearer than that radius.
    spatial_x0 = torch.floor((flat_lo_u - group_min_u) * group_grid_inv_u).to(
        torch.long
    )
    spatial_y0 = torch.floor((flat_lo_v - group_min_v) * group_grid_inv_v).to(
        torch.long
    )
    spatial_x1 = torch.floor((flat_hi_u - group_min_u) * group_grid_inv_u).to(
        torch.long
    )
    spatial_y1 = torch.floor((flat_hi_v - group_min_v) * group_grid_inv_v).to(
        torch.long
    )
    for values in (spatial_x0, spatial_y0, spatial_x1, spatial_y1):
        values.clamp_(0, bezier_spatial_grid - 1)
    nx = (spatial_x1 - spatial_x0 + 1).clamp_min(0)
    ny = (spatial_y1 - spatial_y0 + 1).clamp_min(0)
    spatial_counts_per_edge = nx * ny
    border_visible = edges_2d[..., 4].reshape(-1) > 0.5
    spatial_valid = flat_finite & border_visible
    spatial_sources = torch.nonzero(spatial_valid, as_tuple=False).flatten()
    spatial_expanded, spatial_local = _expand_intervals(
        spatial_sources, spatial_counts_per_edge
    )
    if spatial_expanded.numel() > 0:
        expanded_nx = nx[spatial_expanded]
        cell_x = spatial_x0[spatial_expanded] + spatial_local % expanded_nx
        cell_y = spatial_y0[spatial_expanded] + spatial_local // expanded_nx
        cells = cell_y * bezier_spatial_grid + cell_x
        spatial_keys = group_ids[spatial_expanded] * BEZIER_SPATIAL_CELLS + cells
        spatial_order, spatial_offsets = _grouped_offsets(
            spatial_keys, num_groups, BEZIER_SPATIAL_CELLS
        )
        spatial_refs = edge_ids[spatial_expanded][spatial_order].to(torch.int32)
    else:
        spatial_refs = torch.empty((0,), dtype=torch.int32, device=device)
        spatial_offsets = torch.zeros(
            (num_groups, BEZIER_SPATIAL_CELLS + 1), dtype=torch.long, device=device
        )

    header_words = num_groups * BEZIER_ACCEL_HEADER_SIZE
    scan_base = header_words
    spatial_base = scan_base + scan_refs.numel()
    total_words = spatial_base + spatial_refs.numel()
    if total_words >= 2**31:
        raise OverflowError(
            "Bezier acceleration buffer exceeds int32 addressable range"
        )

    headers = torch.empty(
        (num_groups, BEZIER_ACCEL_HEADER_SIZE), dtype=torch.int32, device=device
    )
    headers[:, BEZIER_EDGE_START] = offsets[:-1].to(torch.int32).repeat(num_frames)
    headers[:, BEZIER_EDGE_END] = offsets[1:].to(torch.int32).repeat(num_frames)
    float_meta = torch.stack(
        (
            min_u,
            min_v,
            max_u,
            max_v,
            grid_inv_u,
            grid_inv_v,
            scan_inv_v,
        ),
        dim=-1,
    ).reshape(num_groups, 7)
    headers[:, BEZIER_MIN_U : BEZIER_SCAN_INV_V + 1] = _float_bits(float_meta)
    headers[:, BEZIER_SCAN_OFFSET_BASE:BEZIER_SPATIAL_OFFSET_BASE] = (
        scan_offsets + scan_base
    ).to(torch.int32)
    headers[:, BEZIER_SPATIAL_OFFSET_BASE:BEZIER_ACCEL_HEADER_SIZE] = (
        spatial_offsets + spatial_base
    ).to(torch.int32)

    return torch.cat((headers.reshape(-1), scan_refs, spatial_refs), dim=0).contiguous()


def _decode(buffer, num_groups):
    """Every group's header words and bin contents, in (group, bin) order.

    Read the way the kernels read them: header ``g`` at ``g * H``, a bin's
    edge ids at ``buffer[begin:end]`` for the begin/end words of its offsets.
    Two buffers that decode alike give every kernel query the same answer,
    wherever in the buffer their bins happen to sit.
    """
    headers = (
        buffer[: num_groups * BEZIER_ACCEL_HEADER_SIZE]
        .view(num_groups, BEZIER_ACCEL_HEADER_SIZE)
        .long()
    )

    def bins(base, count):
        begin = headers[:, base : base + count].reshape(-1)
        end = headers[:, base + 1 : base + count + 1].reshape(-1)
        rows, local = _expand_intervals(torch.arange(begin.numel()), end - begin)
        return end - begin, buffer[begin[rows] + local]

    return (
        headers[:, :BEZIER_SCAN_OFFSET_BASE],
        *bins(BEZIER_SCAN_OFFSET_BASE, bezier_scan_bins),
        *bins(BEZIER_SPATIAL_OFFSET_BASE, BEZIER_SPATIAL_CELLS),
    )


def _assert_decode_alike(buffer, reference, num_groups):
    for got, want in zip(_decode(buffer, num_groups), _decode(reference, num_groups)):
        assert torch.equal(got, want)


def _batch(seed, frames=None, circuits=None, moving=None):
    """A random batch: some circuits fixed in every frame, the rest moving.

    Includes empty circuits, the 1e9 sentinel rows of degenerate segments,
    hidden borders and horizontal edges, which the two tables leave out.
    """
    generator = torch.Generator().manual_seed(seed)
    frames = frames or int(torch.randint(2, 9, (), generator=generator))
    circuits = circuits or int(torch.randint(1, 30, (), generator=generator))
    counts = torch.randint(0, 25, (circuits,), generator=generator)
    counts[int(torch.randint(0, circuits, (), generator=generator))] = 0
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    edges = torch.randn((1, int(offsets[-1]), 6), generator=generator) * 3
    edges[..., 4] = (torch.rand(edges.shape[:2], generator=generator) > 0.3).float()
    edges = edges.repeat(frames, 1, 1)
    if edges.shape[1] > 4:
        edges[:, 0, :4] = 1e9
        edges[:, 1, 3] = edges[:, 1, 1]
    if moving is None:
        moving = float(torch.rand((), generator=generator))
    for circuit in torch.nonzero(torch.rand(circuits, generator=generator) < moving):
        start, end = int(offsets[circuit]), int(offsets[circuit + 1])
        edges[1:, start:end, :4] += 0.4 * torch.randn(
            (frames - 1, end - start, 4), generator=generator
        )
    return edges, offsets.to(torch.int32)


@pytest.fixture
def every_group_built(monkeypatch):
    monkeypatch.setattr(ba, "_dedup_disabled", lambda: True)


@pytest.mark.parametrize("seed", range(12))
def test_the_disabled_arm_is_the_previous_buffer(every_group_built, seed):
    edges, offsets = _batch(seed)
    assert torch.equal(
        build_bezier_edge_acceleration(edges, offsets), _reference_build(edges, offsets)
    )


@pytest.mark.parametrize("chunk_pairs", [1 << 22, 40, 1])
@pytest.mark.parametrize("moving", [0.0, 0.3, 1.0, None])
@pytest.mark.parametrize("seed", range(6))
def test_shared_bins_answer_every_query_as_before(
    monkeypatch, seed, moving, chunk_pairs
):
    monkeypatch.setattr(ba, "_CHUNK_PAIRS", chunk_pairs)
    edges, offsets = _batch(seed, moving=moving)
    num_groups = edges.shape[0] * (offsets.numel() - 1)
    _assert_decode_alike(
        build_bezier_edge_acceleration(edges, offsets),
        _reference_build(edges, offsets),
        num_groups,
    )


def test_a_fixed_circuit_reads_its_frame_zero_bins_in_every_frame():
    edges, offsets = _batch(3, frames=6, circuits=12, moving=0.0)
    circuits = offsets.numel() - 1
    buffer = build_bezier_edge_acceleration(edges, offsets)
    headers = buffer[: 6 * circuits * BEZIER_ACCEL_HEADER_SIZE].view(6, circuits, -1)
    assert torch.equal(headers, headers[:1].expand_as(headers))
    reference = _reference_build(edges, offsets)
    assert buffer.numel() < reference.numel()


def test_unchanged_means_the_same_bits():
    """-0.0 and 0.0 compare equal but are a change; a NaN kept bit for bit is not."""
    edges = torch.ones((3, 4, 6))
    offsets = torch.tensor([0, 2, 4])
    circuit_of_edge = torch.tensor([0, 0, 1, 1])
    assert ba._static_circuits(edges, circuit_of_edge, 2).tolist() == [True, True]
    signed = edges.clone()
    signed[:, 1, 0] = 0.0
    signed[2, 1, 0] = -0.0
    assert ba._static_circuits(signed, circuit_of_edge, 2).tolist() == [False, True]
    nan = edges.clone()
    nan[:, 3, 2] = float("nan")
    assert ba._static_circuits(nan, circuit_of_edge, 2).tolist() == [True, True]
    nan[1, 2, 0] = float("nan")
    assert ba._static_circuits(nan, circuit_of_edge, 2).tolist() == [True, False]
    for batch in (signed, nan):
        _assert_decode_alike(
            build_bezier_edge_acceleration(batch, offsets),
            _reference_build(batch, offsets),
            3 * 2,
        )


def test_one_frame_and_no_edges_match_the_previous_buffer():
    edges, offsets = _batch(5, frames=2)
    one = edges[:1].contiguous()
    assert torch.equal(
        build_bezier_edge_acceleration(one, offsets), _reference_build(one, offsets)
    )
    empty = torch.zeros((4, 0, 6))
    nothing = torch.zeros(3, dtype=torch.int32)
    assert torch.equal(
        build_bezier_edge_acceleration(empty, nothing), _reference_build(empty, nothing)
    )
