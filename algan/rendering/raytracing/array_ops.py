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
