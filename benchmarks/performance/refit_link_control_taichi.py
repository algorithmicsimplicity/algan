"""Original sibling-vector link fetch for the UHD experiment's control arm."""

from algan.rendering.raytracing.stbvh import bvh_arity, bvh_block_f16
from algan.taichi_compat import ti


@ti.func
def legacy_refit_link(row, c, blocks: ti.template()):
    ts_a = blocks[row, 6]
    ts_b = blocks[row, 7]
    w = 0
    for cc in ti.static(range(bvh_arity)):
        if cc == c:
            if ti.static(bvh_block_f16):
                w = ti.cast(ti.bit_cast(ts_a[cc], ti.u16), ti.i32) | (
                    ti.cast(ti.bit_cast(ts_b[cc], ti.u16), ti.i32) << 16)
            else:
                w = ti.bit_cast(ts_a[cc], ti.i32)
    return w
