"""Host tables for the path tracer's density grids (``pt_volume_taichi``)."""

from __future__ import annotations

import torch

from algan.errors import AlganConfigurationError
from algan.rendering.raytracing.pt_volume_taichi import (
    PT_VOL_HEADER,
    PT_VOL_STRIDE,
    PT_VOL_XFORM,
)
from algan.rendering.raytracing.shading_taichi import (
    _MAT_DENSITY_VOLUME,
    _MAT_MEDIUM_CLOSED,
    _MID_PHYSICAL,
)

#: Largest index the kernel forms into ``vol_f`` (32-bit).
_VOL_F_LIMIT = (1 << 31) - 1


def _used_keys(merged):
    """The density keys the batch's closed physical interiors carry, sorted."""
    mat = merged.get("tri_mat")
    if mat is None or mat.shape[-1] <= _MAT_DENSITY_VOLUME:
        return []
    physical = merged["tri_mat_id"] == _MID_PHYSICAL
    closed = mat[..., _MAT_MEDIUM_CLOSED] > 0.5
    keys = mat[..., _MAT_DENSITY_VOLUME]
    physical, closed = torch.broadcast_tensors(physical, closed)
    keyed = keys.expand_as(physical)[physical & closed]
    keyed = keyed[keyed > 0.5]
    if keyed.numel() == 0:
        return []
    return sorted(int(k) for k in torch.unique(torch.round(keyed)).tolist())


def build_volume_tables(merged, density_volumes, num_frames, media_offset):
    """``(vol_i, vol_f)`` CPU tensors in ``pt_volume_taichi``'s layout.

    ``density_volumes`` is the render loop's snapshot, ``(entries, xforms)``
    (``density_grid._density_volume_snapshot``), or None. Only the grids the
    batch's triangles name are packed. A batch that names none -- or has no
    medium walk (``media_offset == 0``) -- gets the empty table ``[0, 1]``,
    which every lookup misses.

    Raises :class:`~algan.errors.AlganConfigurationError` when a triangle names
    a grid the snapshot does not hold: its owner -- the Mob the material was
    set on -- is not spawned for these frames, so there is no frame to lay the
    grid in. Rendering the interior homogeneous instead would be a silent
    substitution.
    """
    empty = (
        torch.tensor([0, 1], dtype=torch.int32),
        torch.zeros(1, dtype=torch.float32),
    )
    if int(media_offset) <= 0:
        return empty
    used = _used_keys(merged)
    if not used:
        return empty
    entries, xforms = density_volumes if density_volumes is not None else ((), None)
    column = {int(key): (q, grid) for q, (key, grid) in enumerate(entries)}
    missing = [key for key in used if key not in column]
    if missing:
        raise AlganConfigurationError(
            "A Mob's interior uses a DensityGrid whose owner -- the Mob "
            "set_material was called on -- is not spawned in these frames, so "
            "the grid has no frame to sit in. Spawn the Mob the material was "
            "set on, or set the material on the Mob that is spawned."
        )
    xforms = torch.as_tensor(xforms, dtype=torch.float32).detach().cpu()
    picked = xforms[:, [column[key][0] for key in used]]
    rows = int(picked.shape[0])
    if rows not in (1, int(num_frames)):
        raise RuntimeError(
            f"density grid transforms cover {rows} frames, the batch {int(num_frames)}"
        )
    words = [len(used), rows]
    floats = [picked.reshape(-1)]
    offset = rows * len(used) * PT_VOL_XFORM
    for key in used:
        density, majorant, n, m = column[key][1]._packed_grid()
        words += [key, n[0], n[1], n[2], offset, m[0], m[1], m[2]]
        floats += [density, majorant]
        offset += int(density.numel()) + int(majorant.numel())
    if offset > _VOL_F_LIMIT:
        raise AlganConfigurationError(
            f"The density grids in this batch hold {offset} values together, "
            "more than the path tracer can index; sample them more coarsely."
        )
    vol_i = torch.tensor(words, dtype=torch.int32)
    assert vol_i.numel() == PT_VOL_HEADER + PT_VOL_STRIDE * len(used)
    return vol_i, torch.cat(floats).to(torch.float32).contiguous()


def prepare_volumes(memory, merged, density_volumes, num_frames, media_offset):
    """Upload :func:`build_volume_tables` into the render arena."""
    from algan.rendering.raytracing.tracer import _arena_copy

    vol_i, vol_f = build_volume_tables(
        merged, density_volumes, num_frames, media_offset
    )
    with memory.scope("pt_volumes", values=int(vol_f.numel())):
        vol_i = _arena_copy(memory, vol_i)
        vol_f = _arena_copy(memory, vol_f)
    return vol_i, vol_f
