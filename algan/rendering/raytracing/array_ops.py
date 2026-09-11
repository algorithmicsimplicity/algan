"""Small host-side array operations with caller-owned output storage."""

from __future__ import annotations

import torch


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
    if n:
        input_end = (
            counts.data_ptr() + ((n - 1) * counts.stride(0) + 1) * counts.element_size()
        )
        output_end = out.data_ptr() + out.numel() * out.element_size()
        if counts.data_ptr() < output_end and out.data_ptr() < input_end:
            raise ValueError("CSR input and output must not overlap")
    out[0] = 0
    torch.cumsum(counts, dim=0, dtype=out.dtype, out=out[1:])
    return out
