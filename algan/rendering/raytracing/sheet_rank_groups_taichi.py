"""Dense conflict-rank IDs from ordered parents and their prefix counts."""

from algan.taichi_compat import ti


@ti.kernel
def rank_groups(
    parent: ti.types.ndarray(), rank: ti.types.ndarray(), ends: ti.types.ndarray(),
    groups: ti.types.ndarray(), cid_band: ti.types.ndarray(),
    rank_of_cid: ti.types.ndarray(), n: ti.i32, parents: ti.i32,
):
    for i in range(n):
        p = ti.cast(parent[i], ti.i32)
        start = 0
        if p > 0:
            start = ends[p - 1]
        groups[i] = start + rank[i]
        if i < parents:
            begin = 0
            if i > 0:
                begin = ends[i - 1]
            for j in range(begin, ends[i]):
                cid_band[j] = i
                rank_of_cid[j] = j - begin
