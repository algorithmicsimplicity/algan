"""Resolver-ready sheet records with explicit arena ownership."""

from __future__ import annotations

from typing import NamedTuple

import torch

from algan.rendering.mps_compat import kernel_index


class SheetBuffers(NamedTuple):
    """Persistent resolver records, not the diagnostic raw-area representation."""

    sheet_key: torch.Tensor
    sheet_ref: torch.Tensor
    sheet_ab: torch.Tensor
    sheet_cov: torch.Tensor
    sheet_msk: torch.Tensor
    sheet_cap: torch.Tensor
    sheet_offsets: torch.Tensor

    @property
    def num_sheets(self):
        return self.sheet_key.numel()


def finish_sheet_buffers(
    memory,
    covered,
    final,
    nearest,
    representative,
    frag_key,
    frag_ref,
    frag_ab,
    frag_cap,
    weights,
    masks,
    sheet_pixels,
):
    """Gather straight into reverse-arena output and build its CSR.

    ``final`` maps walk order to the unordered sheet table; nearest and
    representative are indices into the original fragment stream. The first
    determines depth, the second determines shading. Weights/masks and pixels
    have already been put in walk order. No coverage arithmetic is performed
    here: the copy preserves f32 and packed integer bits exactly.

    Input/output spans must be disjoint. The caller keeps discovery scratch
    alive through these launches; only the returned reverse allocations need
    to survive it. Library sorting/reduction workspace remains external.
    """
    from algan.rendering.raytracing.sheet_output_taichi import (
        copy_sheet_records,
        write_sheet_offsets,
    )

    n, pixels = int(final.numel()), int(covered.numel())
    if n >= 2**31 or pixels >= 2**31:
        raise OverflowError("sheet records and covered ordinals must fit int32")
    out = SheetBuffers(
        memory.get_tensor((n,), torch.int64, persist=True),
        memory.get_tensor((n,), torch.int32, persist=True),
        memory.get_tensor((n, 2), torch.float32, persist=True),
        memory.get_tensor((n,), torch.float32, persist=True),
        memory.get_tensor((n,), torch.int32, persist=True),
        memory.get_tensor((n,), torch.float32, persist=True),
        memory.get_tensor((pixels + 1,), torch.int32, persist=True),
    )
    if n:
        copy_sheet_records(
            kernel_index(final),
            kernel_index(nearest),
            kernel_index(representative),
            frag_key,
            frag_ref,
            frag_ab,
            frag_cap,
            weights,
            masks,
            out.sheet_key,
            out.sheet_ref,
            out.sheet_ab,
            out.sheet_cov,
            out.sheet_msk,
            out.sheet_cap,
            n,
        )
    write_sheet_offsets(covered, sheet_pixels, out.sheet_offsets, pixels, n)
    return out
