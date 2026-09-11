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
from algan.rendering.raytracing.array_ops import gather_rows, require_tensor_outputs
from algan.rendering.raytracing.sheet_workspace import CompactionWorkspace


class SheetStatistics(NamedTuple):
    nearest_fragment: torch.Tensor
    representative_fragment: torch.Tensor
    pixel: torch.Tensor
    min_position: torch.Tensor
    first_sorted: torch.Tensor | None
    fragment_count: torch.Tensor | None

    @classmethod
    def allocate(cls, workspace, count, *, diagnostics):
        """Allocate in the caller's stage so results outlive reduction scratch."""
        return cls(
            *(workspace.tensor((count,), torch.int64) for _ in range(4)),
            workspace.tensor((count,), torch.int64) if diagnostics else None,
            workspace.tensor((count,), torch.int64) if diagnostics else None,
        )


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
    out=None,
):
    """Reduce exact position/count/max statistics without retaining scratch.

    Results use ordinary tensors unless ``out`` supplies disjoint caller-owned
    destinations allocated before this helper's scratch stage. Only results
    leave the stage. Integer min/max and the f32 area maximum keep their existing
    reduction boundaries and tie-breaking. The count and unrestricted first
    position are returned only for diagnostics. No coverage sum is fused into
    this integer/reference work.
    """
    device = coverage.device
    n, nb = int(coverage.numel()), int(num_bands)
    idx_dtype = reduction_index_dtype()
    workspace = workspace or CompactionWorkspace(device=device)
    if workspace.device != device:
        raise ValueError("statistics workspace must share the input device")
    if out is not None:
        require_tensor_outputs(
            out,
            (((nb,), torch.int64),) * 4 + (((nb,), torch.int64),) * 2
            if diagnostics
            else (((nb,), torch.int64),) * 4 + (None, None),
            device=device,
            inputs=(band, masks, positions, original_positions, pixels, coverage),
        )
    # Allocate ordinary outputs before the stage; explicit destinations are
    # already owned by the caller. Narrow reduction results are stage-local.
    if out is None:
        out = SheetStatistics(
            torch.empty(nb, dtype=torch.int64, device=device),
            torch.empty(nb, dtype=torch.int64, device=device),
            torch.empty(nb, dtype=torch.int64, device=device),
            torch.empty(nb, dtype=torch.int64, device=device),
            torch.empty(nb, dtype=torch.int64, device=device) if diagnostics else None,
            torch.empty(nb, dtype=torch.int64, device=device) if diagnostics else None,
        )
    with workspace.stage():
        min_position = (
            out.min_position
            if idx_dtype == torch.int64
            else workspace.tensor((nb,), idx_dtype)
        )
        representative = (
            out.representative_fragment
            if idx_dtype == torch.int64
            else workspace.tensor((nb,), idx_dtype)
        )
        first = (
            out.first_sorted
            if diagnostics and idx_dtype == torch.int64
            else workspace.tensor((nb,), idx_dtype)
        )
        min_position.fill_(n)
        representative.fill_(n)
        first.fill_(n)
        counts = None
        native = bool(rt_settings.sheet_band_stats_kernel and nb)
        cmax = workspace.tensor((nb,), torch.float32, 0)
        if native:
            from algan.rendering.raytracing.sheet_compact_taichi import (
                band_stats_reduce,
                band_stats_rep_orig,
            )

            first_p = workspace.tensor((nb if positioned else 1,), idx_dtype, n)
            min_p = workspace.tensor((nb if positioned else 1,), idx_dtype, n)
            if diagnostics:
                counts = (
                    out.fragment_count
                    if idx_dtype == torch.int64
                    else workspace.tensor((nb,), idx_dtype)
                )
                counts.zero_()
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
            pos_src = (
                original_positions
                if original_positions.dtype == idx_dtype
                else workspace.copy(original_positions, idx_dtype)
            )
            positions_src = (
                positions
                if positions.dtype == idx_dtype
                else workspace.copy(positions, idx_dtype)
            )
            first.scatter_reduce_(
                0, band, positions_src, reduce="amin", include_self=True
            )
            min_position.scatter_reduce_(
                0, band, pos_src, reduce="amin", include_self=True
            )
            if positioned:
                first_p = workspace.tensor((nb,), idx_dtype, n)
                min_p = workspace.tensor((nb,), idx_dtype, n)
                # Each masked stream is consumed before the same bytes are
                # reused for the next amin, rather than retained per output.
                with workspace.stage():
                    low = workspace.tensor(masks.shape, masks.dtype)
                    owns_sample = workspace.tensor(masks.shape, torch.bool)
                    torch.bitwise_and(masks, mask_all, out=low)
                    torch.ne(low, 0, out=owns_sample)
                    big = workspace.tensor((), idx_dtype, n)
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
                gathered = workspace.gather(cmax, band)
                is_max = workspace.tensor(coverage.shape, torch.bool)
                torch.ge(coverage, gathered, out=is_max)
                torch.where(is_max, pos_src, big, out=candidate)
                representative.scatter_reduce_(
                    0, band, candidate, reduce="amin", include_self=True
                )
            if diagnostics:
                counts = out.fragment_count
                counts.zero_()
                with workspace.stage():
                    ones = workspace.tensor(band.shape, band.dtype, 1)
                    counts.scatter_add_(0, band, ones)
        first_long = (
            first if first.dtype == torch.int64 else workspace.copy(first, torch.int64)
        )
        gather_rows(original_positions, first_long, out=out.nearest_fragment)
        gather_rows(pixels, first_long, out=out.pixel)
        out.min_position.copy_(min_position)
        if positioned:
            with workspace.stage():
                first_p = workspace.copy(first_p, torch.int64)
                has_position = workspace.tensor((nb,), torch.bool)
                torch.lt(first_p, n, out=has_position)
                first_p.clamp_max_(max(n - 1, 0))
                nearest = workspace.gather(original_positions, first_p)
                torch.where(
                    has_position,
                    nearest,
                    out.nearest_fragment,
                    out=out.nearest_fragment,
                )
                # min_p's narrower reference arm contains only bounded indices;
                # where promotes it to the same int64 result as the old cast.
                torch.where(has_position, min_p, out.min_position, out=out.min_position)
        out.representative_fragment.copy_(representative)
        if diagnostics:
            out.first_sorted.copy_(first_long)
            out.fragment_count.copy_(counts)
    return out
