from __future__ import annotations

import torch


def _flat_frames(x, last_dims):
    """Collapse camera tensors like [T, 1, 1, 3] to [T, *last_dims]."""
    return x.reshape(x.shape[0], *last_dims).float()


def _expand_frames(x, num_frames):
    if x.shape[0] == num_frames:
        return x
    return x.expand(num_frames, *x.shape[1:])


def _pixel_bases(screen_basis):
    """Per-frame world-space steps corresponding to one unit of normalized
    screen coordinate, matching the camera's projection exactly.

    The camera projects a world point by intersecting its view ray with the
    plane ``normal = basis_row_2`` through the screen center, then taking raw
    dot products with ``basis_row_0/1``. The screen basis is rotation x
    non-uniform scale, so under camera rotation its rows are *not* mutually
    orthogonal (``row0 . row2 != 0``) -- the projection is anisotropic and
    changes with orientation. The exact inverse image of screen coordinate
    (u, v) is ``screen_point + u * d0 + v * d1`` where ``d0, d1`` are the
    first two columns of the inverse basis matrix (the reciprocal basis:
    ``d_i . row_j = delta_ij``), which both lies on the projection plane and
    reproduces the dot products.
    """
    eye = torch.eye(3, device=screen_basis.device).unsqueeze(0) * 1e-12
    dual = torch.linalg.inv(screen_basis + eye)
    return dual[:, :, 0].contiguous(), dual[:, :, 1].contiguous()


def _unify_time(tensors, error_context):
    """Expand a set of tensors whose leading (time) dims are each 1 or T to a
    common T. Returns the expanded tensors and T.
    """
    T = max(t.shape[0] for t in tensors)
    for t in tensors:
        if t.shape[0] not in (1, T):
            raise ValueError(
                f"{error_context}: incompatible frame counts "
                f"{[tuple(t.shape) for t in tensors]}")
    return [_expand_frames(t, T) for t in tensors], T


def _cat_collections(tensors, dim, error_context):
    """Concatenate per-collection tensors along ``dim``, broadcasting their
    (possibly different) time dimensions to a common length first. A single
    collection is passed through without copying (the kernel indexes each
    array's time dimension independently, so no expansion is needed).
    """
    if len(tensors) == 1:
        return tensors[0]
    tensors, _ = _unify_time(tensors, error_context)
    return torch.cat(tensors, dim).contiguous()


def _cat_mat_blocks(blocks, error_context):
    """Concatenate per-collection parameter blocks ``[Tm, N, W]`` along the
    primitive axis, right-zero-padding narrower blocks to the widest ``W`` first.

    Built-in materials pack a 12-slot block while custom fragment pipelines pack
    a wider one; padding lets them share a single per-scene array. The padding
    slots are never read (each stage reads only its own slice), so a built-in
    (or built-in-only) scene is unaffected -- with no wide blocks present ``W``
    stays 12 and no padding happens.
    """
    if len(blocks) == 1:
        return blocks[0]
    max_w = max(b.shape[-1] for b in blocks)
    padded = []
    for b in blocks:
        if b.shape[-1] < max_w:
            pad = torch.zeros((*b.shape[:-1], max_w - b.shape[-1]),
                              dtype=b.dtype, device=b.device)
            b = torch.cat([b, pad], dim=-1)
        padded.append(b)
    return _cat_collections(padded, 1, error_context)