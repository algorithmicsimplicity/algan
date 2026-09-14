"""Owned sorted fragment payloads used throughout sheet reduction."""

from __future__ import annotations

from typing import NamedTuple

import torch

from algan.rendering.raytracing.array_ops import gather_rows, require_tensor_outputs


class SortedFragments(NamedTuple):
    pixel: torch.Tensor
    depth: torch.Tensor
    coverage: torch.Tensor
    mask: torch.Tensor

    @classmethod
    def allocate(cls, workspace, count):
        return cls(
            workspace.tensor((count,), torch.int64),
            workspace.tensor((count,), torch.float32),
            workspace.tensor((count,), torch.float32),
            workspace.tensor((count,), torch.int32),
        )


def gather_sorted_fragments(pixel, depth, coverage, mask, order, *, out=None):
    """Copy the sorted payload without reinterpreting integer or depth bits.

    Validate every destination before writing any field. Coverage must remain
    a private copy: the closed-shell ceiling modifies it in place.
    """
    inputs = (pixel, depth, coverage, mask)
    n, device = order.numel(), pixel.device
    dtypes = (torch.int64, torch.float32, torch.float32, torch.int32)
    if (
        order.ndim != 1
        or order.dtype not in (torch.int32, torch.int64)
        or order.device != device
        or any(
            x.shape != pixel.shape or x.ndim != 1 or x.device != device or x.dtype != dt
            for x, dt in zip(inputs, dtypes)
        )
    ):
        raise ValueError(
            "sorted fragment fields must be matching vectors with canonical dtypes"
        )
    layouts = tuple(((n,), dt) for dt in dtypes)
    if out is None:
        out = SortedFragments(
            *(torch.empty((n,), dtype=dt, device=device) for dt in dtypes)
        )
    require_tensor_outputs(out, layouts, device=device, inputs=(*inputs, order))
    for source, dest in zip(inputs, out):
        gather_rows(source, order, out=dest)
    return out
