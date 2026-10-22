"""The reflection/refraction gather on RT cores.

``wavefront_traverse_events`` gathers, for every active ray, the ``kbuf``
nearest hits strictly after the ray's cursor in ``_comes_after`` order, cut
after the first opaque one (``_collect_hits``), into ``hit_f`` / ``hit_i``
and the count into ``rs_int[:, 3]``. For a batch with triangles and no Bezier
circuits, :func:`gather` fills the same buffers from an OptiX launch
(``__raygen__gather``), and the shade kernel after it is unchanged.

The hit set is the same; what can differ is the hardware's float arithmetic
for each hit (its watertight triangle test, its ``t`` and barycentrics).
"""

from __future__ import annotations

import torch

from algan.rendering.raytracing.optix.runtime import new_params

_GAS_KEY = "_optix_gather_gas"

#: Launches taken and rays gathered on RT cores, for tests and harnesses.
STATS = {"launches": 0, "rays": 0}


def gather_gas(runtime, merged, tri_pos):
    """This batch's gather GASes (every visible triangle), built on first use.

    Shared with the shadow GASes when no triangle declines to cast a shadow:
    the two then contain the same primitives in the same classes.
    """
    from algan.rendering.raytracing.optix.gas import build_batch_gas
    from algan.rendering.raytracing.optix.shadows import shadow_gas

    cached = merged.get(_GAS_KEY)
    if cached is not None:
        return cached or None
    device = tri_pos.device
    valid = merged.get("tri_frame_valid")
    opaque = merged.get("tri_frame_opaque")
    casts = merged.get("tri_frame_casts")
    gas = None
    if valid is not None and opaque is not None:
        all_cast = casts is None or bool(casts.all())
        shared = shadow_gas(runtime, merged, tri_pos) if all_cast else None
        if shared is not None and not getattr(shared, "all_translucent", False):
            gas = shared
        else:
            gas = build_batch_gas(runtime, tri_pos, valid.to(device), opaque.to(device))
    merged[_GAS_KEY] = gas if gas is not None else False
    return gas


def gather(
    runtime,
    gas,
    *,
    active,
    num_active,
    rs_ro,
    rs_rd,
    rs_sca,
    rs_int,
    rs_pix,
    hit_f,
    hit_i,
    time_start,
    ray_offset,
    pixels_per_frame,
    layer_offset,
):
    """Launch the OptiX gather over ``active[:num_active]``."""
    from algan.rendering.raytracing.raytrace_kernels_taichi import (
        INV_DEPTH_TIE_EPSILON,
        depth_tie_epsilon,
        min_hit_distance,
    )

    for name, tensor, dtype in (
        ("active", active, torch.int32),
        ("rs_ro", rs_ro, torch.float32),
        ("rs_rd", rs_rd, torch.float32),
        ("rs_sca", rs_sca, torch.float32),
        ("rs_int", rs_int, torch.int32),
        ("rs_pix", rs_pix, torch.int32),
        ("hit_f", hit_f, torch.float32),
        ("hit_i", hit_i, torch.int32),
    ):
        if tensor.dtype != dtype:
            raise TypeError(f"OptiX gather: {name} is {tensor.dtype}, not {dtype}")
    for tensor in (rs_ro, rs_rd, rs_sca, rs_int):
        if tensor.stride(-1) != 1:
            raise ValueError("OptiX gather: ray-state rows must be contiguous")
    if hit_f.stride(-1) != 1 or hit_i.stride(-1) != 1 or active.stride(0) != 1:
        raise ValueError("OptiX gather: hit buffers must be contiguous along rays")
    params = new_params()
    params.handles = gas.handles.data_ptr()
    params.num_handles = gas.num_handles
    params.min_hit_distance = float(min_hit_distance)
    params.inv_bin = float(INV_DEPTH_TIE_EPSILON)
    params.bin_width = float(depth_tie_epsilon)
    params.layer_offset = float(layer_offset)
    params.active = active.data_ptr()
    params.rs_ro = rs_ro.data_ptr()
    params.rs_rd = rs_rd.data_ptr()
    params.rs_sca = rs_sca.data_ptr()
    params.rs_int = rs_int.data_ptr()
    params.rs_pix = rs_pix.data_ptr()
    params.hit_f = hit_f.data_ptr()
    params.hit_i = hit_i.data_ptr()
    params.num_active = int(num_active)
    params.time_start = int(time_start)
    params.ray_offset = int(ray_offset)
    params.pixels_per_frame = int(pixels_per_frame)
    params.ro_stride = int(rs_ro.stride(0))
    params.rd_stride = int(rs_rd.stride(0))
    params.sca_stride = int(rs_sca.stride(0))
    params.int_stride = int(rs_int.stride(0))
    params.hit_f_q = int(hit_f.stride(0))
    params.hit_f_c = int(hit_f.stride(1))
    params.hit_i_q = int(hit_i.stride(0))
    params.hit_i_c = int(hit_i.stride(1))
    runtime.launch("gather", params, int(num_active))
    STATS["launches"] += 1
    STATS["rays"] += int(num_active)
