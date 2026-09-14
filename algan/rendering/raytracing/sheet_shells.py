"""Stage-owned closed-shell grouping and in-place coverage ceilings.

A declared closed surface spends at most ``max(front_area, back_area)`` per
pixel, in true depth order. This applies authored opacity once to an ordinary
solid while preserving genuine same-facing self-overlaps: the cap is not
clamped to one. Undeclared/transmissive surfaces and circuits keep separate
pass-through segments. This is the existing fragment-space allowance policy,
not a visibility-weighted allowance spent by the resolver.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from algan.rendering.mps_compat import (
    accumulate_dtype,
    index_copy_rows,
    kernel_index,
    taichi_accumulate_dtype,
)
from algan.rendering.raytracing.array_ops import (
    gather_frame_table,
    gather_rows,
    group_ids_from_starts,
    require_tensor_outputs,
)
from algan.rendering.raytracing.raster_taichi import _AA_BACKFACE_BIT as BACKFACE_BIT
from algan.rendering.raytracing.sheet_preprocessing import _surface_ids
from algan.rendering.raytracing.sheet_workspace import CompactionWorkspace


class ShellSegments(NamedTuple):
    """Sorted-stream segment keys and facing flags, not reduction scratch."""

    key: torch.Tensor
    back: torch.Tensor

    @classmethod
    def allocate(cls, workspace, count):
        return cls(
            workspace.tensor((count,), torch.int64),
            workspace.tensor((count,), torch.bool),
        )


def shell_segments(
    pixel,
    mask,
    frame,
    reference,
    triangle,
    order,
    positions,
    tri_obj,
    tri_closed,
    time_start,
    *,
    out=None,
    workspace=None,
):
    """Prepare exact closed-shell keys, or return None without a declaration.

    ``pixel`` and ``mask`` are sorted payloads; frame/reference/triangle are raw
    fragment metadata, addressed by ``order``. ``positions`` is the sorted
    position arange. Table rows wrap independently after adding ``time_start``.
    Bounds and permutation values remain the producer's responsibility.

    All destinations are checked before any write. Without ``out`` the returned
    record owns ordinary tensors; with it the result is the supplied record.
    No active declaration (including an empty stream) returns None and leaves
    caller destinations untouched. Metadata lookups never survive this call.
    """
    n, device = pixel.numel(), pixel.device
    inputs = (pixel, mask, frame, reference, triangle, order, positions)
    dtypes = (
        torch.int64,
        torch.int32,
        torch.int64,
        torch.int64,
        torch.bool,
        torch.int64,
        torch.int64,
    )
    if any(
        x.shape != (n,) or x.dtype != dt or x.device != device
        for x, dt in zip(inputs, dtypes)
    ):
        raise ValueError("shell metadata needs canonical matching input vectors")
    if (
        tri_obj.ndim != 2
        or tri_obj.shape[0] == 0
        or tri_obj.dtype not in (torch.int32, torch.int64)
        or tri_obj.device != device
        or tri_closed.ndim != 2
        or tri_closed.shape[0] == 0
        or tri_closed.shape[1] != tri_obj.shape[1]
        or tri_closed.device != device
        or tri_closed.is_complex()
    ):
        raise ValueError("shell metadata needs compatible real frame/primitive tables")
    workspace = workspace or CompactionWorkspace(device=device)
    if workspace.device != device:
        raise ValueError("shell workspace must share the input device")
    if out is not None:
        require_tensor_outputs(
            out,
            (((n,), torch.int64), ((n,), torch.bool)),
            device=device,
            inputs=(*inputs, tri_obj, tri_closed),
        )
    if n == 0:
        return None
    with workspace.stage():
        closed = workspace.tensor((n,), torch.bool)
        with workspace.stage():
            raw_closed = workspace.tensor((n,), torch.bool)
            values = workspace.tensor((n,), tri_closed.dtype)
            gather_frame_table(
                tri_closed,
                frame,
                reference,
                time_start=time_start,
                out=values,
                workspace=workspace,
            )
            torch.gt(values, 0.5, out=raw_closed)
            raw_closed.logical_and_(triangle)
            active = bool(raw_closed.any())
            if active:
                gather_rows(raw_closed, order, out=closed)
        if not active:
            return None
        if out is None:
            out = ShellSegments(
                torch.empty((n,), dtype=torch.int64, device=device),
                torch.empty((n,), dtype=torch.bool, device=device),
            )
        # Preserve the old maximum over *all* references, including a circuit's
        # safe reference. Changing the key ordering changes the global FP scan.
        with workspace.stage():
            surface = workspace.tensor((n,), torch.int64)
            _surface_ids(tri_obj, frame, reference, time_start, surface, workspace)
            gather_rows(surface, order, out=out.key)
        stride = int(out.key.amax()) + 2
        integer = workspace.tensor((n,), torch.int64)
        torch.mul(pixel, stride, out=integer)
        out.key.add_(integer)
        integer.copy_(positions).add_(1).neg_()
        torch.where(closed, out.key, integer, out=out.key)
        torch.bitwise_and(mask, BACKFACE_BIT, out=integer)
        torch.ne(integer, 0, out=out.back)
    return out


def apply_shell_ceiling(
    segments,
    depth,
    coverage,
    *,
    order_builder,
    use_kernel,
    workspace=None,
):
    """Modify private float32 coverage without retaining shell scratch.

    Segment keys/facings and exact depths are immutable. The in/out coverage
    must be contiguous and disjoint from them. Both arms keep the library's
    global exclusive scan, the standalone spent subtraction, the cap's
    accumulator -> float32 -> accumulator rounding, and the denominator floor
    that also changes the final multiplicand. No fusion/reassociation is added.
    """
    n, device = depth.numel(), depth.device
    inputs = (segments.key, segments.back, depth)
    if any(
        x.shape != (n,) or x.dtype != dt or x.device != device
        for x, dt in zip(inputs, (torch.int64, torch.bool, torch.float32))
    ):
        raise ValueError("shell ceiling needs matching key, facing and depth vectors")
    require_tensor_outputs(
        (coverage,), (((n,), torch.float32),), device=device, inputs=inputs
    )
    workspace = workspace or CompactionWorkspace(device=device)
    if workspace.device != device:
        raise ValueError("shell workspace must share the input device")
    if n == 0:
        return coverage
    with workspace.stage():
        # Sorting's optional local kernel needs contiguous inputs; retain exact
        # caller values in staged copies only for standalone strided callers.
        key = (
            segments.key
            if segments.key.is_contiguous()
            else workspace.copy(segments.key)
        )
        back = (
            segments.back
            if segments.back.is_contiguous()
            else workspace.copy(segments.back)
        )
        sort_depth = depth if depth.is_contiguous() else workspace.copy(depth)
        segments = ShellSegments(key, back)
        order = order_builder(key, sort_depth, workspace=workspace)
        acc = accumulate_dtype()
        # Always copy, including float32 compatibility mode. Native code uses
        # this spare buffer as a reassociation barrier while coverage is INOUT.
        scratch = workspace.copy(coverage, acc)
        ordered = workspace.gather(scratch, order)
        exclusive = workspace.tensor((n,), acc)
        torch.cumsum(ordered, 0, out=exclusive)
        exclusive.sub_(ordered)
        if use_kernel:
            from algan.rendering.raytracing.sheet_compact_taichi import (
                solid_shell_ceiling,
            )

            solid_shell_ceiling(
                segments.key.contiguous(),
                kernel_index(order.contiguous()),
                segments.back.contiguous().view(torch.uint8),
                exclusive,
                scratch,
                n,
                coverage,
                taichi_accumulate_dtype(),
            )
        else:
            _reference_ceiling(segments, order, ordered, exclusive, coverage, workspace)
    return coverage


def _reference_ceiling(segments, order, ordered, exclusive, coverage, workspace):
    """Reference arithmetic with explicit destinations and nested result stages."""
    n, acc = ordered.numel(), ordered.dtype
    with workspace.stage():
        group = workspace.tensor((n,), torch.int64)
        with workspace.stage():
            keys = workspace.gather(segments.key, order)
            starts = workspace.tensor((n,), torch.bool, True)
            torch.ne(keys[1:], keys[:-1], out=starts[1:])
            group_ids_from_starts(starts, out=group)
            count = int(group[-1]) + 1
        spent = workspace.tensor((n,), acc)
        with workspace.stage():
            # Dense group IDs mean the kth nonzero is exactly group's kth first
            # position. No boolean-indexed group IDs or first-index scatter.
            starts = workspace.tensor((n,), torch.bool, True)
            torch.ne(group[1:], group[:-1], out=starts[1:])
            first = workspace.tensor((count, 1), torch.int64)
            torch.nonzero(starts, out=first)
            base = workspace.gather(exclusive, first.view(-1))
            gather_rows(base, group, out=spent)
            torch.sub(exclusive, spent, out=spent)
        scale = workspace.tensor((n,), acc)
        with workspace.stage():
            front = workspace.tensor((count,), acc, 0)
            back = workspace.tensor((count,), acc, 0)
            with workspace.stage():
                back_facing = workspace.gather(segments.back, order)
                values = workspace.copy(ordered)
                values.masked_fill_(back_facing, 0)
                front.scatter_add_(0, group, values)
                values.copy_(ordered)
                back_facing.logical_not_()
                values.masked_fill_(back_facing, 0)
                back.scatter_add_(0, group, values)
            torch.maximum(front, back, out=front)
            rounded = workspace.copy(front, torch.float32)
            front.copy_(rounded)
            gather_rows(front, group, out=scale)
        scale.sub_(spent).clamp_min_(0.0)
        ordered.clamp_min_(1e-12)
        scale.div_(ordered).clamp_max_(1.0)
        ordered.mul_(scale)
        result = workspace.copy(ordered, torch.float32)
        index_copy_rows(coverage, order, result)
