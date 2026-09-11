"""Stable sort output ownership shared by the sheet ordering stages."""

from __future__ import annotations

import torch

from algan.rendering.raytracing import device_sort
from algan.rendering.raytracing.array_ops import gather_rows, require_tensor_outputs
from algan.rendering.raytracing.sheet_workspace import CompactionWorkspace


def stable_lexsort(*keys, out=None, workspace=None):
    """Stable LSD key ordering, with optional caller-owned int64 permutation.

    Keys are one-dimensional, with the most significant first. Sort values,
    per-pass indices, gathered keys and composed permutations are short-lived
    workspace, not outputs. Sort-library/driver workspace remains external.
    Native device radix sorting retains its capability gate and int32 indices.
    """
    if not keys:
        if out is not None:
            raise ValueError("a sort destination requires at least one key")
        return None
    shape, device = keys[0].shape, keys[0].device
    if len(shape) != 1 or any(k.shape != shape or k.device != device for k in keys):
        raise ValueError("sort keys must be equal-length vectors on one device")
    if workspace is not None and workspace.device != device:
        raise ValueError("sort workspace and keys must share a device")
    if out is not None:
        require_tensor_outputs(
            (out,), ((shape, torch.int64),), device=device, inputs=keys
        )
    workspace = workspace or CompactionWorkspace(device=device)
    # The default result must outlive our scratch stage, including standalone
    # callers that provide a workspace but no destination.
    if out is None:
        out = torch.empty(shape, dtype=torch.int64, device=device)
    with workspace.stage():
        if all(device_sort.radix_sort_available(key) for key in keys):
            native = workspace.tensor(shape, torch.int32)
            device_sort.stable_lexsort(*keys, out=native, workspace=workspace)
            out.copy_(native)
            return out
        # Only one composed permutation and one pass's indices are needed in
        # addition to the caller's output; reuse them for every significant key.
        step = workspace.tensor(shape, torch.int64) if len(keys) > 1 else out
        other = workspace.tensor(shape, torch.int64) if len(keys) > 1 else out
        current = out
        for pass_index, key in enumerate(reversed(keys)):
            with workspace.stage():
                selected = key if pass_index == 0 else workspace.gather(key, current)
                values = workspace.tensor(shape, key.dtype)
                torch.sort(
                    selected,
                    stable=True,
                    out=(values, out if pass_index == 0 else step),
                )
                if pass_index:
                    gather_rows(current, step, out=other)
                    current, other = other, current
        if current is not out:
            out.copy_(current)
    return out
