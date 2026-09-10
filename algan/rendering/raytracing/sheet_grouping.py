"""Bounded-memory integer grouping for sheet streams already ordered in runs.

The caller owns ordering preconditions and chooses a local compiler backend.
All scans count at most one boundary per input, so int32 is exact below 2**31
rows. Keys and public group IDs retain their original integer precision.
"""

from __future__ import annotations

import torch

_INT_DTYPES = (torch.int32, torch.int64)


def _check_vector(tensor, name, *, dtypes=_INT_DTYPES, device=None, length=None):
    """Reject unsupported metadata before importing a buffer into a kernel."""
    if tensor.ndim != 1 or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous one-dimensional tensor")
    if tensor.dtype not in dtypes:
        raise TypeError(f"{name} has unsupported dtype {tensor.dtype}")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}")
    if length is not None and tensor.numel() != length:
        raise ValueError("sheet group inputs must have matching lengths")


def unique_sorted_ids(keys):
    """Consecutive unique values and inverse IDs, without a global sort.

    ``keys`` is a contiguous one-dimensional ordered integer tensor. The two
    kernels plus a prefix scan never transfer the keys to the host; only the
    result count is read back. Each unique output has one writer.
    """
    from algan.rendering.raytracing.sheet_grouping_taichi import (
        unique_boundaries,
        unique_scatter,
    )

    _check_vector(keys, "keys")
    n = keys.numel()
    if n >= 2**31:
        raise ValueError("sheet grouping requires fewer than 2**31 rows")
    inverse = torch.empty(n, dtype=torch.int64, device=keys.device)
    if not n:
        return torch.empty_like(keys), inverse
    starts = torch.empty(n, dtype=torch.int32, device=keys.device)
    unique_boundaries(keys, starts, n)
    prefix = torch.cumsum(starts, 0, dtype=torch.int32)
    del starts
    unique = torch.empty(int(prefix[-1]), dtype=keys.dtype, device=keys.device)
    unique_scatter(keys, prefix, unique, inverse, n)
    return unique, inverse


def class_groups(bands, classes, starts):
    """Exact (band, class) groups within existing surface/facing runs.

    Each run owns a disjoint increasing interval of band IDs. IDs may decrease
    inside a run (conflict ranks), and classes may span the full 25-bit range.
    Sorting each run is consequently equivalent to global lexicographic
    sorting. No floating-point encoding, truncated scan or fixed-size run
    array is used. Outputs match the wide-key integer CPU reference.
    """
    from algan.rendering.raytracing.sheet_grouping_taichi import (
        class_order,
        class_scatter,
    )

    _check_vector(bands, "bands")
    n = bands.numel()
    _check_vector(classes, "classes", device=bands.device, length=n)
    _check_vector(
        starts,
        "starts",
        dtypes=(torch.bool, torch.uint8, torch.int32, torch.int64),
        device=bands.device,
        length=n,
    )
    if n >= 2**31:
        raise ValueError("sheet grouping requires fewer than 2**31 rows")
    inverse = torch.empty(n, dtype=torch.int64, device=bands.device)
    if not n:
        return 0, inverse, torch.empty_like(inverse)
    order = torch.empty(n, dtype=torch.int32, device=bands.device)
    boundaries = torch.empty_like(order)
    # The Metal import bridge supports u8, not torch.bool. Reinterpret the
    # same bytes rather than silently staging a whole flag array through CPU.
    start_bytes = starts.view(torch.uint8) if starts.dtype == torch.bool else starts
    class_order(start_bytes, bands, classes, order, boundaries, n)
    prefix = torch.cumsum(boundaries, 0, dtype=torch.int32)
    del boundaries
    count = int(prefix[-1])
    group_band = torch.empty(count, dtype=torch.int64, device=bands.device)
    class_scatter(bands, order, prefix, inverse, group_band, n)
    return count, inverse, group_band


def prepare_fragments(
    keys, refs, masks, objects, pixels_per_frame, time_start, backface_bit
):
    """Unpack integer sheet metadata, retaining the packed depth bits exactly.

    Inputs must be contiguous and local to the selected compiler. The caller
    owns primitive-reference bounds. Object rows repeat periodically, including
    nonzero batch starts. No coverage, position or shading arithmetic changes.
    """
    from algan.rendering.raytracing.sheet_grouping_taichi import fragment_metadata

    _check_vector(keys, "keys", dtypes=(torch.int64,))
    n = keys.numel()
    _check_vector(refs, "refs", device=keys.device, length=n)
    _check_vector(masks, "masks", device=keys.device, length=n)
    if objects.ndim != 2 or not objects.is_contiguous():
        raise ValueError("objects must be a contiguous two-dimensional tensor")
    if objects.dtype not in _INT_DTYPES:
        raise TypeError(f"objects has unsupported dtype {objects.dtype}")
    if objects.device != keys.device:
        raise ValueError(f"objects must be on {keys.device}")
    if n and not objects.numel():
        raise ValueError("nonempty fragments require an object table")
    if objects.numel() >= 2**31:
        raise ValueError("object table exceeds 32-bit launch bounds")
    if not 0 < pixels_per_frame < 2**63:
        raise ValueError("pixels_per_frame must be a positive signed 64-bit integer")
    if not -(2**31) <= backface_bit < 2**31:
        raise ValueError("backface_bit exceeds 32-bit launch bounds")
    if n >= 2**31 or not -(2**31) <= time_start < 2**31:
        raise ValueError("fragment metadata exceeds 32-bit launch bounds")
    device = keys.device
    pix = torch.empty(n, dtype=torch.int64, device=device)
    depth = torch.empty(n, dtype=torch.float32, device=device)
    frames = torch.empty_like(pix)
    triangles = torch.empty(n, dtype=torch.bool, device=device)
    safe_refs = torch.empty_like(pix)
    positions = torch.empty_like(pix)
    groups = torch.empty_like(pix)
    if not n:
        # Empty MPS tensors have a null MTLBuffer and cannot be imported.
        return pix, depth, frames, triangles, safe_refs, positions, groups
    fragment_metadata(
        keys,
        refs,
        masks,
        objects,
        pix,
        depth,
        frames,
        triangles.view(torch.uint8),
        safe_refs,
        positions,
        groups,
        n,
        pixels_per_frame,
        time_start,
        backface_bit,
    )
    return pix, depth, frames, triangles, safe_refs, positions, groups


def unique_sorted_ids_reference(keys):
    """Integer-exact Torch fallback when a local Metal launch is unavailable.

    MPSGraph unique can merge distinct wide integer keys. Compare boundaries
    as integers and gather with advanced indexing, never float-backed unique
    or index_select. This also covers small inputs below the kernel threshold.
    """
    n = keys.numel()
    if not n:
        return torch.empty_like(keys), torch.empty(
            0, dtype=torch.int64, device=keys.device
        )
    if n >= 2**31:
        raise ValueError("sheet grouping requires fewer than 2**31 rows")
    starts = torch.ones(n, dtype=torch.bool, device=keys.device)
    starts[1:] = keys[1:] != keys[:-1]
    inverse = torch.cumsum(starts, 0, dtype=torch.int32).to(torch.int64) - 1
    unique = keys[torch.nonzero(starts).flatten()]
    return unique, inverse
