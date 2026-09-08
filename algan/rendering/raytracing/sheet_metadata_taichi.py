"""Integer-only diagnostics over the compaction's existing fragment groups."""

from algan.taichi_compat import ti


@ti.kernel
def group_counts(
    new_group: ti.types.ndarray(),
    band_id: ti.types.ndarray(),
    order: ti.types.ndarray(),
    is_tri: ti.types.ndarray(),
    partial: ti.types.ndarray(),
    n: ti.i32,
):
    # A sheet belongs to exactly one original (pixel, surface, facing) group.
    # Consequently that group split iff its fragments have differing final
    # sheet IDs, including both conflict-rank and shading-class subdivisions.
    # Groups are disjoint; their total scan length is bounded by n, without a
    # per-group length limit. Circuit-only groups are excluded from diagnostics.
    ti.loop_config(block_dim=128)
    for start in range(n):
        if new_group[start] != 0:
            if is_tri[order[start]] != 0:
                split = 0
                j = start + 1
                while j < n:
                    if new_group[j] != 0:
                        break
                    if band_id[j] != band_id[start]:
                        split = 1
                        break
                    j += 1
                ti.atomic_add(partial[start // 256, 0], 1)
                ti.atomic_add(partial[start // 256, 1], split)
