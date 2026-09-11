"""Nearest/dominant sheet references with phase-local reduction scratch."""

from __future__ import annotations

from typing import NamedTuple

import torch

from algan.rendering.mps_compat import (
    kernel_index,
    reduction_index_dtype,
    taichi_reduction_index_dtype,
)
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.sheet_workspace import CompactionWorkspace


class SheetStatistics(NamedTuple):
    nearest_fragment: torch.Tensor
    representative_fragment: torch.Tensor
    pixel: torch.Tensor
    min_position: torch.Tensor
    first_sorted: torch.Tensor | None
    fragment_count: torch.Tensor | None


def sheet_statistics(
    band,
    masks,
    positions,
    original_positions,
    pixels,
    coverage,
    num_bands,
    *,
    mask_all,
    positioned,
    diagnostics,
    workspace=None,
):
    """Reduce exact position/count/max statistics without retaining scratch.

    Results use ordinary caller-owned tensors; only those results leave the
    stage. Integer min/max and the f32 area maximum keep their existing
    reduction boundaries and tie-breaking. The count and unrestricted first
    position are returned only for diagnostics. No coverage sum is fused into
    this integer/reference work.
    """
    device = coverage.device
    n, nb = int(coverage.numel()), int(num_bands)
    idx_dtype = reduction_index_dtype()
    workspace = workspace or CompactionWorkspace(device=device)
    # These arrays (or their int64 conversions) survive the stage. Do not
    # allocate them from scratch then rely on .to being a copy on every device.
    min_position = torch.full((nb,), n, dtype=idx_dtype, device=device)
    representative = torch.full((nb,), n, dtype=idx_dtype, device=device)
    counts = None
    with workspace.stage():
        first = (
            torch.full((nb,), n, dtype=idx_dtype, device=device)
            if diagnostics
            else workspace.tensor((nb,), idx_dtype, n)
        )
        native = bool(rt_settings.sheet_band_stats_kernel and nb)
        cmax = workspace.tensor((nb,), torch.float32, 0)
        if native:
            from algan.rendering.raytracing.sheet_compact_taichi import (
                band_stats_reduce,
                band_stats_rep_orig,
            )

            first_p = workspace.tensor((nb if positioned else 1,), idx_dtype, n)
            min_p = workspace.tensor((nb if positioned else 1,), idx_dtype, n)
            counts = (
                torch.zeros(nb, dtype=idx_dtype, device=device) if diagnostics else None
            )
            count_arg = counts if diagnostics else workspace.tensor((1,), idx_dtype, 0)
            band_stats_reduce(
                kernel_index(band.contiguous()),
                masks.contiguous(),
                original_positions.contiguous(),
                coverage.contiguous(),
                n,
                int(mask_all),
                first,
                min_position,
                first_p,
                min_p,
                cmax,
                count_arg,
                bool(positioned),
                taichi_reduction_index_dtype(),
                bool(diagnostics),
            )
            band_stats_rep_orig(
                kernel_index(band.contiguous()),
                original_positions,
                coverage,
                cmax,
                n,
                representative,
                taichi_reduction_index_dtype(),
            )
        else:
            pos_src = original_positions.to(idx_dtype)
            positions_src = positions.to(idx_dtype)
            first.scatter_reduce_(
                0, band, positions_src, reduce="amin", include_self=True
            )
            min_position.scatter_reduce_(
                0, band, pos_src, reduce="amin", include_self=True
            )
            if positioned:
                first_p = workspace.tensor((nb,), idx_dtype, n)
                min_p = workspace.tensor((nb,), idx_dtype, n)
                owns_sample = (masks & mask_all) != 0
                big = workspace.tensor((), idx_dtype, n)
                # Each masked stream is consumed before the same bytes are
                # reused for the next amin, rather than retained per output.
                with workspace.stage():
                    masked = workspace.tensor((n,), idx_dtype)
                    torch.where(owns_sample, positions_src, big, out=masked)
                    first_p.scatter_reduce_(
                        0, band, masked, reduce="amin", include_self=True
                    )
                    torch.where(owns_sample, pos_src, big, out=masked)
                    min_p.scatter_reduce_(
                        0, band, masked, reduce="amin", include_self=True
                    )
            cmax.scatter_reduce_(0, band, coverage, reduce="amax", include_self=True)
            with workspace.stage():
                candidate = workspace.tensor((n,), idx_dtype)
                big = workspace.tensor((), idx_dtype, n)
                is_max = coverage >= cmax.index_select(0, band)
                torch.where(is_max, pos_src, big, out=candidate)
                representative.scatter_reduce_(
                    0, band, candidate, reduce="amin", include_self=True
                )
            if diagnostics:
                counts = torch.zeros(nb, dtype=torch.int64, device=device)
                counts.scatter_add_(0, band, torch.ones_like(band))
        first_long = first.to(torch.int64)
        nearest = original_positions.index_select(0, first_long)
        pixel = pixels.index_select(0, first_long)
        min_position = min_position.to(torch.int64)
        if positioned:
            first_p = first_p.to(torch.int64)
            has_position = first_p < n
            nearest = torch.where(
                has_position,
                original_positions.index_select(0, first_p.clamp_max(max(n - 1, 0))),
                nearest,
            )
            min_position = torch.where(
                has_position, min_p.to(torch.int64), min_position
            )
        result = SheetStatistics(
            nearest,
            representative.to(torch.int64),
            pixel,
            min_position,
            first_long if diagnostics else None,
            counts.to(torch.int64) if diagnostics else None,
        )
    return result
