"""Batch-sized closed-shell identity for deterministic straight-ray segments."""

from __future__ import annotations

import torch

from algan.rendering.raytracing import settings as rt_settings


def _build_closed_shell_table(memory, merged):
    """Return an arena table of dense shell IDs and its number of identities.

    tri_closed already excludes physical transmission at geometry packing.
    Its time axis and tri_obj's collapse independently; broadcasting preserves
    both. IDs are dense across the entire batch, so each ray needs one bit per
    shell, not a fixed-capacity stack that loses excess overlapping shells.
    The all-opaque case needs no tracking: a coverage ray stops at its first
    hit. A one-element -1 table disables the lookup in every inactive case.
    """
    closed = merged.get("tri_closed") if rt_settings.solid_shell_alpha else None
    if closed is not None and merged.get("tri_has_translucent", True):
        obj = merged["tri_obj"].to(torch.int32)
        shell = torch.where((closed > 0.5) & (obj >= 0), obj, -1)
        ids, inverse = torch.unique(shell, sorted=True, return_inverse=True)
        missing = int(ids.numel() > 0 and int(ids[0].item()) < 0)
        count = int(ids.numel()) - missing
        if count:
            dense = (inverse - missing).to(torch.int32)
            table = memory.get_tensor(tuple(dense.shape), torch.int32)
            table.copy_(dense)
            return table, count
    table = memory.get_tensor((1, 1), torch.int32)
    table.fill_(-1)
    return table, 0
