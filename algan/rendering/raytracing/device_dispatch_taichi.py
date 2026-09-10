"""Portable integer scans and fused typed queue packing, with no host counts."""

from algan.taichi_compat import ti

_BLOCK = 256
_VEC6 = ti.types.vector(6, ti.f32)


@ti.kernel
def reset_dispatch_header(header: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    for i in range(4):
        header[i] = 0


@ti.kernel
def count_selected(n: ti.i32,
                   accepted: ti.types.ndarray(dtype=ti.i32, ndim=1),
                   rank: ti.types.ndarray(dtype=ti.i32, ndim=1),
                   totals: ti.types.ndarray(dtype=ti.i32, ndim=1),
                   header: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    # n-1 avoids overflow near INT32_MAX. The empty block initializes total[0].
    for block in range(ti.max(1, (n - 1) // _BLOCK + 1)):
        running = 0
        for j in range(_BLOCK):
            i = block * _BLOCK + j
            if i < n:
                value = accepted[i]
                if value != 0 and value != 1:
                    ti.atomic_or(header[3], 1)
                rank[i] = running
                if value != 0:
                    running += 1
        totals[block] = running


@ti.kernel
def scan_dispatch_blocks(n: ti.i32,
                         values: ti.types.ndarray(dtype=ti.i32, ndim=1),
                         offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
                         totals: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    for block in range(ti.max(1, (n - 1) // _BLOCK + 1)):
        running = 0
        for j in range(_BLOCK):
            i = block * _BLOCK + j
            if i < n:
                offsets[i] = running
                running += values[i]
        totals[block] = running


@ti.kernel
def add_dispatch_carries(n: ti.i32,
                         offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
                         parent: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    for i in range(n):
        offsets[i] += parent[i // _BLOCK]


@ti.kernel
def pack_primary_shadow_events(
        n: ti.i32, capacity: ti.i32,
        header: ti.types.ndarray(dtype=ti.i32, ndim=1),
        rank: ti.types.ndarray(dtype=ti.i32, ndim=1),
        offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
        total: ti.types.ndarray(dtype=ti.i32, ndim=1),
        accepted: ti.types.ndarray(dtype=ti.i32, ndim=1),
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        pos: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
        snrm: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
        fnrm: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
        frame: ti.types.ndarray(dtype=ti.i32, ndim=1),
        mask: ti.types.ndarray(dtype=ti.i32, ndim=1),
        dp: ti.types.ndarray(dtype=_VEC6, ndim=1),
        toff: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
        out_pos: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
        out_snrm: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
        out_fnrm: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
        out_frame: ti.types.ndarray(dtype=ti.i32, ndim=1),
        out_mask: ti.types.ndarray(dtype=ti.i32, ndim=1),
        out_source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        out_dp: ti.types.ndarray(dtype=_VEC6, ndim=1),
        out_toff: ti.types.ndarray(dtype=ti.math.vec3, ndim=1),
        reverse: ti.types.ndarray(dtype=ti.i32, ndim=1),
        footprint: ti.template(), terminator: ti.template()):
    reserved = 0
    if n > 0:
        reserved = total[0]
    header[1] = reserved
    header[2] = ti.cast(reserved > capacity, ti.i32)
    header[0] = 0
    if header[2] == 0 and header[3] == 0:
        header[0] = reserved
    for i in range(n):
        if header[2] == 0 and header[3] == 0:
            reverse[i] = -1
            if accepted[i] != 0:
                event = offsets[i // _BLOCK] + rank[i]
                if event >= 0 and event < capacity:
                    out_pos[event] = pos[i]
                    out_snrm[event] = snrm[i]
                    out_fnrm[event] = fnrm[i]
                    out_frame[event] = frame[i]
                    out_mask[event] = mask[i]
                    out_source[event] = source[i]
                    if ti.static(footprint):
                        out_dp[event] = dp[i]
                    if ti.static(terminator):
                        out_toff[event] = toff[i]
                    reverse[i] = event
                else:
                    ti.atomic_or(header[3], 2)
