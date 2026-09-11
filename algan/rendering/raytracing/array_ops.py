"""Small host-side array operations with caller-owned output storage."""

from __future__ import annotations

import torch


def require_disjoint_output(output: torch.Tensor, *inputs: torch.Tensor):
    """Reject overlapping byte ranges without reading device values.

    Strided inputs use their bounding interval, conservatively rejecting an
    output in a stride gap. Disjoint views of the same arena remain valid.
    This helper checks overlap only; callers validate output shape and dtype.
    """
    if not output.numel():
        return
    out_start = output.data_ptr()
    out_end = (
        out_start
        + (
            1
            + sum((n - 1) * stride for n, stride in zip(output.shape, output.stride()))
        )
        * output.element_size()
    )
    for source in inputs:
        if not source.numel() or source.device != output.device:
            continue
        start = source.data_ptr()
        end = (
            start
            + (
                1
                + sum(
                    (n - 1) * stride for n, stride in zip(source.shape, source.stride())
                )
            )
            * source.element_size()
        )
        if start < out_end and out_start < end:
            raise ValueError("input and output must not overlap")


def csr_offsets(counts: torch.Tensor, *, out: torch.Tensor | None = None):
    """Build integer CSR offsets including the terminal total.

    Counts must be a nonnegative int32/int64 vector; callers guarantee the
    nonnegativity and that their chosen output dtype can hold the total. This
    helper validates metadata only, without a device readback. The default
    output is int64. A caller can supply an arena-backed contiguous output of
    length ``counts.numel() + 1``. Input and output must not overlap.

    The returned ``offsets[:-1]`` are exclusive starts and ``offsets[-1]`` is
    the total, including for empty counts. PyTorch may still use internal scan
    workspace; caller-owned output is not a claim of zero external allocation.
    """
    integer_dtypes = (torch.int32, torch.int64)
    if counts.ndim != 1 or counts.dtype not in integer_dtypes:
        raise ValueError("CSR counts must be a one-dimensional int32/int64 tensor")
    n = counts.numel()
    if out is None:
        out = torch.empty(n + 1, dtype=torch.int64, device=counts.device)
    if (
        out.shape != (n + 1,)
        or out.dtype not in integer_dtypes
        or out.device != counts.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "CSR output must be a contiguous integer vector of length n + 1 on the counts device"
        )
    require_disjoint_output(out, counts)
    out[0] = 0
    torch.cumsum(counts, dim=0, dtype=out.dtype, out=out[1:])
    return out


def require_tensor_outputs(outputs, layouts, *, device, inputs=()):
    """Validate all destination metadata and disjointness before any writes.

    Each layout is ``(shape, dtype)`` or ``None`` for an omitted output. The
    latter requires a ``None`` destination too. All present outputs must be
    pairwise disjoint and disjoint from inputs. No device values are read.
    """
    if len(outputs) != len(layouts):
        raise ValueError("wrong number of output tensors")
    present = []
    for output, layout in zip(outputs, layouts):
        if layout is None:
            if output is not None:
                raise ValueError("disabled output must be None")
            continue
        shape, dtype = layout
        if (
            not isinstance(output, torch.Tensor)
            or output.shape != shape
            or output.dtype != dtype
            or output.device != device
            or not output.is_contiguous()
        ):
            raise ValueError("output shape, dtype, device or layout does not match")
        require_disjoint_output(output, *inputs, *present)
        present.append(output)


def gather_rows(source: torch.Tensor, indices: torch.Tensor, *, out=None):
    """Gather dimension-zero rows exactly, optionally into a caller destination.

    Indices must be a one-dimensional integer tensor of valid nonnegative row
    indices. Validate metadata and byte-range aliasing before any output write;
    bounds remain the caller's responsibility, without another device readback.
    MPS integer output uses a local copy kernel where available. Its fallback
    retains the measured exact advanced-indexing path (and its temporary).
    """
    from algan.rendering.mps_compat import gather_exact, mps_friendly

    if (
        source.ndim == 0
        or indices.ndim != 1
        or indices.dtype not in (torch.int32, torch.int64)
        or indices.device != source.device
    ):
        raise ValueError("row gather needs an array and integer indices on one device")
    if out is None:
        return gather_exact(source, indices)
    shape = (indices.numel(), *source.shape[1:])
    require_tensor_outputs(
        (out,), ((shape, source.dtype),), device=source.device, inputs=(source, indices)
    )
    if not out.numel():
        return out
    if mps_friendly() and source.dtype in (torch.int32, torch.int64):
        from algan.rendering.taichi_runtime import _live_arch, taichi_launch_is_local

        if (
            source.is_contiguous()
            and indices.is_contiguous()
            and out.numel() < 2**31
            and source.numel() < 2**31
            and _live_arch() is not None
            and taichi_launch_is_local(source.device)
        ):
            from algan.rendering.raytracing.array_copy_taichi import gather_rows_into

            gather_rows_into(
                source.view(-1),
                indices,
                out.view(-1),
                indices.numel(),
                source.numel() // source.shape[0],
            )
        else:
            out.copy_(gather_exact(source, indices))
    else:
        torch.index_select(source, 0, indices, out=out)
    return out


def group_ids_from_starts(starts: torch.Tensor, *, out=None):
    """Inclusive boundary scan minus one, with an optional integer destination.

    A true flag begins a group. Callers provide a true first flag for nonempty
    streams; like the original expression, a leading false flag produces -1.
    Validate metadata, capacity and aliasing without reading device values.
    No converted boundary vector or separate subtraction result is allocated.
    """
    if starts.ndim != 1 or starts.dtype != torch.bool:
        raise ValueError("group starts must be a one-dimensional boolean tensor")
    if out is None:
        out = torch.empty(starts.shape, dtype=torch.int64, device=starts.device)
    if out.dtype not in (torch.int32, torch.int64):
        raise ValueError("group IDs must use int32 or int64")
    require_tensor_outputs(
        (out,), ((starts.shape, out.dtype),), device=starts.device, inputs=(starts,)
    )
    # The inclusive scan must fit too, before subtracting one from each ID.
    if starts.numel() > torch.iinfo(out.dtype).max:
        raise ValueError("group-start scan exceeds the output integer capacity")
    torch.cumsum(starts, 0, dtype=out.dtype, out=out)
    out.sub_(1)
    return out
