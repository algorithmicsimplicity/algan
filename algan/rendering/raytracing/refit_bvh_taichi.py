"""Refit one tree level directly into its traversal blocks.

Topology and bounds reductions stay unchanged. One thread owns a sibling
block; child-node unions come from the already completed deeper level.
"""

from algan.taichi_compat import ti


@ti.func
def _half_directed_bits(value, up: ti.template()):
    x = ti.min(ti.max(value, -65504.0), 65504.0)
    h = ti.cast(x, ti.f16)
    decoded = ti.cast(h, ti.f32)
    bits = ti.cast(ti.bit_cast(h, ti.u16), ti.i32)
    monotone = 0xffff - bits
    if bits < 0x8000:
        monotone = bits + 0x8000
    if ti.static(up):
        if decoded < x:
            monotone += 1
    else:
        if decoded > x:
            monotone -= 1
    result = 0xffff - monotone
    if monotone >= 0x8000:
        result = monotone - 0x8000
    # Same outward subnormal flush as stbvh._half_bits_directed.
    magnitude = result & 0x7fff
    if magnitude > 0 and magnitude < 0x400:
        if ti.static(up):
            result = 0
            if monotone >= 0x8000:
                result = 0x400
        else:
            result = 0
            if monotone < 0x8000:
                result = 0x8400
    return ti.cast(result, ti.i16)


@ti.kernel
def refit_pack_level(
    frame_lo: ti.types.ndarray(dtype=ti.f32, ndim=3),
    frame_hi: ti.types.ndarray(dtype=ti.f32, ndim=3),
    child_kind: ti.types.ndarray(dtype=ti.u8, ndim=2),
    child_ref: ti.types.ndarray(dtype=ti.i64, ndim=2),
    opaque: ti.types.ndarray(dtype=ti.u8, ndim=2),
    nocast: ti.types.ndarray(dtype=ti.i32, ndim=1),
    primitive_ids: ti.types.ndarray(dtype=ti.i64, ndim=1),
    node_lo: ti.types.ndarray(dtype=ti.f32, ndim=3),
    node_hi: ti.types.ndarray(dtype=ti.f32, ndim=3),
    blocks: ti.types.ndarray(ndim=3),
    level_start: ti.i32,
    level_end: ti.i32,
    half: ti.template(),
    arity: ti.template(),
    remap: ti.template(),
    later_ties: ti.template(),
):
    for frame, local_node in ti.ndrange(frame_lo.shape[0], level_end - level_start):
        node = level_start + local_node
        row = frame * child_kind.shape[0] + node
        union_lo = ti.Vector([float("inf"), float("inf"), float("inf")])
        union_hi = ti.Vector([-float("inf"), -float("inf"), -float("inf")])
        for slot in ti.static(range(arity)):
            kind = child_kind[node, slot]
            ref = ti.cast(child_ref[node, slot], ti.i32)
            lo = ti.Vector([1e17, 1e17, 1e17])
            hi = ti.Vector([-1e17, -1e17, -1e17])
            word = -1
            if kind == 1:
                for axis in ti.static(range(3)):
                    lo[axis] = frame_lo[frame, ref, axis]
                    hi[axis] = frame_hi[frame, ref, axis]
                prim = ref
                if ti.static(remap):
                    prim = ti.cast(primitive_ids[ref], ti.i32)
                word = prim | -2147483648
                word |= ti.cast(opaque[frame % opaque.shape[0], ref], ti.i32) << 30
                word |= nocast[ref] << 29
            elif kind == 2:
                for axis in ti.static(range(3)):
                    lo[axis] = node_lo[frame, ref, axis]
                    hi[axis] = node_hi[frame, ref, axis]
                word = ref
            if not (hi >= lo).all():
                word = -1
            # Torch keeps the later equal value on CUDA, the first on CPU.
            # Explicit comparisons preserve the corresponding +/-0 bits.
            for axis in ti.static(range(3)):
                if ti.static(later_ties):
                    if lo[axis] <= union_lo[axis]:
                        union_lo[axis] = lo[axis]
                    if hi[axis] >= union_hi[axis]:
                        union_hi[axis] = hi[axis]
                else:
                    if lo[axis] < union_lo[axis]:
                        union_lo[axis] = lo[axis]
                    if hi[axis] > union_hi[axis]:
                        union_hi[axis] = hi[axis]
            for axis in ti.static(range(3)):
                if ti.static(half):
                    blocks[row, axis, slot] = _half_directed_bits(lo[axis], False)
                    blocks[row, axis + 3, slot] = _half_directed_bits(hi[axis], True)
                else:
                    blocks[row, axis, slot] = lo[axis]
                    blocks[row, axis + 3, slot] = hi[axis]
            if ti.static(half):
                blocks[row, 6, slot] = ti.cast(word & 0xffff, ti.i16)
                blocks[row, 7, slot] = ti.cast((word >> 16) & 0xffff, ti.i16)
            else:
                blocks[row, 6, slot] = ti.bit_cast(word, ti.f32)
                blocks[row, 7, slot] = 0.0
        for axis in ti.static(range(3)):
            node_lo[frame, node, axis] = union_lo[axis]
            node_hi[frame, node, axis] = union_hi[axis]
