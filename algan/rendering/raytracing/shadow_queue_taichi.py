"""Exact copies between dense shadow events and arena-owned trace payloads."""

from algan.taichi_compat import ti


@ti.kernel
def gather_shadow_payload(
        indices: ti.types.ndarray(), n: ti.i32,
        pos: ti.types.ndarray(), snrm: ti.types.ndarray(), fnrm: ti.types.ndarray(),
        frame: ti.types.ndarray(), mask: ti.types.ndarray(),
        footprint: ti.types.ndarray(), terminator: ti.types.ndarray(),
        with_footprint: ti.template(), with_terminator: ti.template(),
        out_pos: ti.types.ndarray(), out_snrm: ti.types.ndarray(), out_fnrm: ti.types.ndarray(),
        out_frame: ti.types.ndarray(), out_mask: ti.types.ndarray(),
        out_footprint: ti.types.ndarray(), out_terminator: ti.types.ndarray()):
    for i in range(n):
        source = indices[i]
        for c in ti.static(range(3)):
            out_pos[i, c] = pos[source, c]
            out_snrm[i, c] = snrm[source, c]
            out_fnrm[i, c] = fnrm[source, c]
        out_frame[i] = frame[source]
        out_mask[i] = mask[source]
        if ti.static(with_footprint):
            for c in ti.static(range(6)):
                out_footprint[i, c] = footprint[source, c]
        if ti.static(with_terminator):
            for c in ti.static(range(3)):
                out_terminator[i, c] = terminator[source, c]


@ti.kernel
def scatter_shadow_visibility(
        indices: ti.types.ndarray(), source: ti.types.ndarray(),
        destination: ti.types.ndarray(), n: ti.i32, num_lights: ti.i32):
    # Accepted event indices are unique, even after sorting. Unused light
    # slots and unaccepted rows keep the destination's existing all-lit fill.
    for event, light in ti.ndrange(n, num_lights):
        row = indices[event]
        for c in ti.static(range(3)):
            destination[row, 3 * light + c] = source[event, light, c]
