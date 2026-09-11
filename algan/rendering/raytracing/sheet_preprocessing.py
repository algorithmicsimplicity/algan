"""Checked fragment metadata destinations, separate from sorted payloads."""

from __future__ import annotations

from typing import NamedTuple

import torch

from algan.rendering.raytracing.array_ops import (
    gather_frame_table,
    require_tensor_outputs,
)
from algan.rendering.raytracing.raster_taichi import _AA_BACKFACE_BIT as AA_BACKFACE_BIT
from algan.rendering.raytracing.raster_taichi import _AA_FULL_DUST as FULL_DUST
from algan.rendering.raytracing.raster_taichi import _AA_MASK_ALL as AA_MASK_ALL
from algan.rendering.raytracing.raster_taichi import (
    _AA_MAT_OPAQUE_BIT as AA_MAT_OPAQUE_BIT,
)
from algan.rendering.raytracing.raster_taichi import _AA_SLIVER_BIT as AA_SLIVER_BIT
from algan.rendering.raytracing.sheet_workspace import CompactionWorkspace


class FragmentMetadata(NamedTuple):
    pixel: torch.Tensor
    depth: torch.Tensor
    frame: torch.Tensor
    triangle: torch.Tensor
    reference: torch.Tensor
    group: torch.Tensor

    @classmethod
    def allocate(cls, workspace, count, *, triangle=None):
        return cls(
            workspace.tensor((count,), torch.int64),
            workspace.tensor((count,), torch.float32),
            workspace.tensor((count,), torch.int64),
            workspace.tensor((count,), torch.bool) if triangle is None else triangle,
            workspace.tensor((count,), torch.int64),
            workspace.tensor((count,), torch.int64),
        )


def fragment_metadata(
    keys,
    references,
    masks,
    positions,
    tri_obj,
    pixels_per_frame,
    time_start,
    *,
    out=None,
    workspace=None,
):
    """Decode immutable raw-stream facts without retaining lookup scratch.

    All destinations are validated together before mutation. Group IDs preserve
    the surface/facing key and each circuit's unique negative key. Depths copy
    their packed float32 bits, not a numerical conversion. Surface IDs widen
    before multiplication. Primitive bounds remain a producer-side contract.
    """
    n, device = keys.numel(), keys.device
    inputs = (keys, references, masks, positions)
    dtypes = (torch.int64, torch.int32, torch.int32, torch.int64)
    if any(
        x.shape != (n,) or x.dtype != dt or x.device != device
        for x, dt in zip(inputs, dtypes)
    ):
        raise ValueError("fragment metadata needs canonical matching input vectors")
    if (
        tri_obj.ndim != 2
        or tri_obj.shape[0] == 0
        or tri_obj.dtype not in (torch.int32, torch.int64)
        or tri_obj.device != device
    ):
        raise ValueError("surface IDs must be a 2D integer table on the input device")
    if (
        isinstance(pixels_per_frame, bool)
        or not isinstance(pixels_per_frame, int)
        or pixels_per_frame <= 0
    ):
        raise ValueError("pixels per frame must be a positive integer")
    workspace = workspace or CompactionWorkspace(device=device)
    if workspace.device != device:
        raise ValueError("metadata workspace must share the input device")
    dtypes = (
        torch.int64,
        torch.float32,
        torch.int64,
        torch.bool,
        torch.int64,
        torch.int64,
    )
    if out is None:
        out = FragmentMetadata(
            *(torch.empty((n,), dtype=dt, device=device) for dt in dtypes)
        )
    require_tensor_outputs(
        out,
        tuple(((n,), dt) for dt in dtypes),
        device=device,
        inputs=(*inputs, tri_obj),
    )
    pixel, depth, frame, triangle, reference, group = out
    with workspace.stage():
        integer = workspace.tensor((n,), torch.int64)
        torch.bitwise_right_shift(keys, 32, out=pixel)
        torch.bitwise_and(keys, 0xFFFFFFFF, out=integer)
        depth.view(torch.int32).copy_(integer)
        torch.div(pixel, pixels_per_frame, rounding_mode="floor", out=frame)
        torch.ge(references, 0, out=triangle)
        reference.copy_(references).clamp_min_(0)
        _surface_ids(tri_obj, frame, reference, time_start, group, workspace)
        group.mul_(2)
        torch.bitwise_and(masks, AA_BACKFACE_BIT, out=integer)
        integer.ne_(0)
        group.add_(integer)
        integer.copy_(positions).add_(2).neg_()
        torch.where(triangle, group, integer, out=group)
    return out


def _surface_ids(table, frames, references, time_start, out, workspace):
    """Copy a validated surface table into a wide metadata destination."""
    with workspace.stage():
        target = (
            out
            if table.dtype == torch.int64
            else workspace.tensor(out.shape, table.dtype)
        )
        gather_frame_table(
            table,
            frames,
            references,
            time_start=time_start,
            out=target,
            workspace=workspace,
        )
        if target is not out:
            out.copy_(target)


class SampleDepthMetadata(NamedTuple):
    """Sample mask, surface identity and the two depth-competition policies."""

    mask: torch.Tensor
    surface: torch.Tensor
    enforcer: torch.Tensor
    subject: torch.Tensor

    @classmethod
    def allocate(cls, workspace, count):
        return cls(
            workspace.tensor((count,), torch.int32),
            workspace.tensor((count,), torch.int64),
            workspace.tensor((count,), torch.bool),
            workspace.tensor((count,), torch.bool),
        )


def sample_depth_metadata(
    pixel,
    reference,
    mask,
    coverage,
    weight,
    only_band,
    tri_obj,
    pixels_per_frame,
    time_start,
    *,
    out=None,
    workspace=None,
):
    """Classify final sheets without changing signed-weight or dust policies.

    A sole, nonnegatively weighted opaque full sheet near unit area may enforce
    depth; positioned, non-sliver sheets of a sole band may be subjects.
    Circuits and subdivided bands participate on neither side. Coverage is
    tested at the same float32 subtraction/absolute-value boundary as before.
    Every destination is checked before any field is modified.
    """
    n, device = pixel.numel(), pixel.device
    inputs = (pixel, reference, mask, coverage, weight, only_band)
    dtypes = (
        torch.int64,
        torch.int32,
        torch.int32,
        torch.float32,
        torch.float32,
        torch.bool,
    )
    if any(
        x.shape != (n,) or x.dtype != dt or x.device != device
        for x, dt in zip(inputs, dtypes)
    ):
        raise ValueError("sample-depth metadata needs canonical matching input vectors")
    if (
        tri_obj.ndim != 2
        or tri_obj.shape[0] == 0
        or tri_obj.dtype not in (torch.int32, torch.int64)
        or tri_obj.device != device
    ):
        raise ValueError("surface IDs must be a 2D integer table on the input device")
    if (
        isinstance(pixels_per_frame, bool)
        or not isinstance(pixels_per_frame, int)
        or pixels_per_frame <= 0
    ):
        raise ValueError("pixels per frame must be a positive integer")
    workspace = workspace or CompactionWorkspace(device=device)
    if workspace.device != device:
        raise ValueError("metadata workspace must share the input device")
    dtypes = (torch.int32, torch.int64, torch.bool, torch.bool)
    if out is None:
        out = SampleDepthMetadata(
            *(torch.empty((n,), dtype=dt, device=device) for dt in dtypes)
        )
    require_tensor_outputs(
        out,
        tuple(((n,), dt) for dt in dtypes),
        device=device,
        inputs=(*inputs, tri_obj),
    )
    low, surface, enforcer, subject = out
    with workspace.stage():
        frame = workspace.tensor((n,), torch.int64)
        torch.div(pixel, pixels_per_frame, rounding_mode="floor", out=frame)
        safe_ref = workspace.copy(reference, torch.int64).clamp_min_(0)
        _surface_ids(tri_obj, frame, safe_ref, time_start, surface, workspace)
        torch.bitwise_and(mask, AA_MASK_ALL, out=low)
        eligible = workspace.tensor((n,), torch.bool)
        test = workspace.tensor((n,), torch.bool)
        flags = workspace.tensor((n,), torch.int32)
        torch.ge(reference, 0, out=eligible)
        eligible.logical_and_(only_band)
        torch.ge(weight, 0.0, out=test)
        eligible.logical_and_(test)
        torch.eq(low, AA_MASK_ALL, out=enforcer)
        enforcer.logical_and_(eligible)
        torch.bitwise_and(mask, AA_MAT_OPAQUE_BIT, out=flags)
        torch.ne(flags, 0, out=test)
        enforcer.logical_and_(test)
        delta = workspace.tensor((n,), torch.float32)
        torch.sub(coverage, 1.0, out=delta)
        delta.abs_()
        torch.le(delta, FULL_DUST, out=test)
        enforcer.logical_and_(test)
        torch.ne(low, 0, out=subject)
        subject.logical_and_(eligible)
        torch.bitwise_and(mask, AA_SLIVER_BIT, out=flags)
        torch.eq(flags, 0, out=test)
        subject.logical_and_(test)
    return out
