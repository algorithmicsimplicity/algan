"""Local fused stream operations with bounded arena scratch and torch fallbacks.

Only integer operations and copies are fused. Coverage reductions, stable sort
semantics, sample ownership, shell ceilings and depth rules are unchanged.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch


def stream_kernel_available(tensor):
    """Do not initialize a compiler or stage arrays to a different device."""
    from algan.rendering.taichi_runtime import _live_arch, taichi_launch_is_local

    return (
        tensor.shape[0] < 2**31
        and _live_arch() is not None
        and taichi_launch_is_local(tensor.device)
    )


def _empty(shape, dtype, device, memory=None):
    if memory is None:
        return torch.empty(shape, dtype=dtype, device=device)
    return memory.get_tensor(shape, dtype)


def gather_group_stream(order, pixel, group, depth, cov, mask, *, memory=None):
    """Return sorted fields and group starts; scratch belongs to the caller.

    With an arena, the caller must enclose the compaction in ``memory.temp``.
    All returned fields are consumed before that scope ends.
    """
    from algan.rendering.raytracing.sheet_stream_taichi import (
        gather_group_stream as launch,
    )

    n = order.numel()
    fields = [
        _empty((n,), arr.dtype, arr.device, memory) for arr in (pixel, depth, cov, mask)
    ]
    starts = _empty((n,), torch.bool, pixel.device, memory)
    if n:
        launch(
            order, pixel, group, depth, cov, mask, *fields, starts.view(torch.uint8), n
        )
    return (*fields, starts)


def gather_sheet_records(
    final, nearest, rep, key, ref, ab, cap, cov, mask, nfrag, fused, band
):
    """Gather final records directly; no sorted packed-key or representative temporary."""
    from algan.rendering.raytracing.sheet_stream_taichi import (
        gather_sheet_records as launch,
    )

    n = final.numel()
    sources = {
        "sheet_key": key,
        "sheet_pix": key,
        "sheet_ref": ref,
        "sheet_ab": ab,
        "sheet_cap": cap,
        "sheet_cov": cov,
        "sheet_msk": mask,
        "sheet_nfrag": nfrag,
        "sheet_fused": fused,
    }
    out = {
        name: torch.empty((n, *src.shape[1:]), dtype=src.dtype, device=src.device)
        for name, src in sources.items()
    }
    # Metal zero-copy cannot import a zero-byte buffer. Reuse an existing
    # output when the template removes band writes, rather than allocating
    # a dummy. No kernel is launched for an empty result.
    band_out = (
        torch.empty(n, dtype=torch.int64, device=key.device)
        if band is not None
        else out["sheet_pix"]
    )
    if n:
        launch(
            final,
            nearest,
            rep,
            key,
            ref,
            ab,
            cap,
            cov,
            mask,
            nfrag,
            fused.view(torch.uint8),
            band if band is not None else final,
            out["sheet_key"],
            out["sheet_pix"],
            out["sheet_ref"],
            out["sheet_ab"],
            out["sheet_cap"],
            out["sheet_cov"],
            out["sheet_msk"],
            out["sheet_nfrag"],
            out["sheet_fused"].view(torch.uint8),
            band_out,
            n,
            band is not None,
        )
    return out, band_out if band is not None else None


def pixel_runs(key, *, memory=None):
    """Return covered pixels, counts and int32 CSR offsets for pixel-sorted keys.

    There is one scalar readback for the exact output shape. Scan scratch is
    arena-backed when supplied, and released before return, including failures.
    No per-pixel or run-length ceiling is introduced.
    """
    from algan.rendering.raytracing.sheet_stream_taichi import (
        pixel_run_flags,
        write_pixel_runs,
    )

    n = key.numel()
    if n >= 2**31:
        raise OverflowError("pixel-run offsets require fewer than 2**31 fragments")
    if not n:
        return (
            torch.empty(0, dtype=torch.int64, device=key.device),
            torch.empty(0, dtype=torch.int64, device=key.device),
            torch.zeros(1, dtype=torch.int32, device=key.device),
        )
    with memory.temp() if memory is not None else nullcontext():
        flags = _empty((n,), torch.int32, key.device, memory)
        prefix = _empty((n,), torch.int32, key.device, memory)
        pixel_run_flags(key, flags, n)
        torch.cumsum(flags, 0, dtype=torch.int32, out=prefix)
        nr = int(prefix[-1].item())
        covered = torch.empty(nr, dtype=torch.int64, device=key.device)
        offsets = torch.empty(nr + 1, dtype=torch.int32, device=key.device)
        write_pixel_runs(key, prefix, covered, offsets, n, nr)
        counts = (offsets[1:] - offsets[:-1]).to(torch.int64)
    return covered, counts, offsets


def truncate_pixel_runs(opaque, offsets):
    """Retain exact opaque prefixes and return (indices or None, counts, CSR).

    Every nonempty run keeps at least its first fragment, so the covered-pixel
    list is unchanged. Counts replace the second ``unique_consecutive`` and
    an index-writing kernel replaces ``keep.nonzero``.
    """
    from algan.rendering.raytracing.sheet_stream_taichi import (
        retained_pixel_counts,
        retained_pixel_indices,
    )

    nr = offsets.numel() - 1
    n = opaque.numel()
    counts = torch.empty(nr, dtype=torch.int32, device=opaque.device)
    new_offsets = torch.zeros(nr + 1, dtype=torch.int32, device=opaque.device)
    if not n:
        return None, counts.to(torch.int64), new_offsets
    retained_pixel_counts(opaque.view(torch.uint8), offsets, counts, nr)
    torch.cumsum(counts, 0, dtype=torch.int32, out=new_offsets[1:])
    retained = int(new_offsets[-1].item())
    indices = None
    if retained != n:
        indices = torch.empty(retained, dtype=torch.int32, device=opaque.device)
        retained_pixel_indices(offsets, new_offsets, indices, nr)
    return indices, counts.to(torch.int64), new_offsets


def fragment_run_order(primary_key, layer):
    """Sort the full primary key once, then only break ties by descending layer.

    This reproduces two stable global sorts, including original-index ties,
    without quantizing depth or imposing a maximum length on a tie run.
    """
    from algan.rendering.mps_compat import gather_packed_key
    from algan.rendering.raytracing.sheet_sort_taichi import key_run_order

    order = torch.argsort(primary_key, stable=True)
    n = primary_key.numel()
    if n > 1:
        run_key = gather_packed_key(primary_key, order)
        inverse_layer = torch.bitwise_not(layer)
        key_run_order(run_key, inverse_layer, inverse_layer, order, n, False, False)
    return order
