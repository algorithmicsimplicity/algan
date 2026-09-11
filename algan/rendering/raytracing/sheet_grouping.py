"""Destination-aware grouping without changing the existing grouping policies."""

from __future__ import annotations

from typing import NamedTuple

import torch

from algan.rendering.mps_compat import mps_friendly
from algan.rendering.raytracing.array_ops import (
    group_ids_from_starts,
    require_tensor_outputs,
)
from algan.rendering.raytracing.sheet_order import stable_lexsort
from algan.rendering.raytracing.sheet_workspace import CompactionWorkspace


class RankGroups(NamedTuple):
    """Per-fragment IDs and the parent/rank of each resulting group."""

    ids: torch.Tensor
    parent: torch.Tensor
    rank: torch.Tensor


class RankPoolGroups(NamedTuple):
    """Compositing-group count and optional per-rank-band destination.

    ``ids is None`` preserves the no-pooling path. A supplied destination is
    unused in that case and its contents are unspecified.
    """

    count: int
    ids: torch.Tensor | None


def validate_inverse(out, *keys):
    """Check destination metadata and input aliases before grouping writes."""
    if not keys:
        raise ValueError("grouping requires at least one key")
    shape, device = keys[0].shape, keys[0].device
    if len(shape) != 1 or any(
        k.shape != shape
        or k.device != device
        or k.dtype not in (torch.int32, torch.int64)
        for k in keys
    ):
        raise ValueError(
            "grouping keys must be equal-length integer vectors on one device"
        )
    if out is not None:
        require_tensor_outputs(
            (out,), ((shape, torch.int64),), device=device, inputs=keys
        )


def unique_ids(keys, *, consecutive=False, out=None):
    """Retain unique's policy and dynamic keys, optionally owning its inverse.

    PyTorch unique has no destination API. Its temporary inverse is copied and
    released here; this does not claim to remove the library's workspace.
    """
    validate_inverse(out, keys)
    unique = torch.unique_consecutive if consecutive else torch.unique
    values, inverse = unique(keys, return_inverse=True)
    if out is not None:
        out.copy_(inverse)
        inverse = out
    return values, inverse


def class_groups(band, classes, base, *, out=None, workspace=None):
    """The existing packed or MPS pair grouping, with an exact inverse output.

    The inverse is fixed-size and allocated before scratch. Group labels have
    dynamic size and ordinary ownership; callers may copy these smaller labels
    into their result stage after this function has released sorting scratch.
    """
    validate_inverse(out, band, classes)
    device, n = band.device, band.numel()
    if workspace is not None and workspace.device != device:
        raise ValueError("grouping workspace and inputs must share a device")
    workspace = workspace or CompactionWorkspace(device=device)
    if not mps_friendly():
        with workspace.stage():
            key = workspace.tensor((n,), torch.int64)
            # An int64 out buffer does not promote int32 arithmetic in torch.
            key.copy_(band)
            key.mul_(base).add_(classes)
            labels, inverse = unique_ids(key, out=out)
        # labels belongs to unique, not the scratch stage.
        labels.div_(base, rounding_mode="floor")
        return labels.numel(), inverse, labels
    if out is None:
        out = torch.empty((n,), dtype=torch.int64, device=device)
    if n == 0:
        return 0, out, torch.empty((0,), dtype=band.dtype, device=device)
    with workspace.stage():
        # Same exact narrowing and stable order as the established MPS arm.
        order = workspace.tensor((n,), torch.int64)
        stable_lexsort(
            workspace.copy(band, torch.int32),
            workspace.copy(classes, torch.int32),
            out=order,
            workspace=workspace,
        )
        bands = workspace.gather(band, order)
        cls = workspace.gather(classes, order)
        starts = workspace.tensor((n,), torch.bool, True)
        with workspace.stage():
            changed_class = workspace.tensor((max(0, n - 1),), torch.bool)
            torch.ne(bands[1:], bands[:-1], out=starts[1:])
            torch.ne(cls[1:], cls[:-1], out=changed_class)
            starts[1:].logical_or_(changed_class)
        group_sorted = workspace.tensor((n,), torch.int64)
        group_ids_from_starts(starts, out=group_sorted)
        out.scatter_(0, order, group_sorted)
        count = int(group_sorted[-1]) + 1
    # Every duplicate index writes the identical integer band, as in the
    # existing MPS implementation. No boolean compaction/readback is needed.
    labels = torch.empty((count,), dtype=band.dtype, device=device)
    labels.scatter_(0, out, band)
    return count, out, labels
