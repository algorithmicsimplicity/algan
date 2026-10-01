"""Auxiliary compositing passes (depth, normal, source-Mob id) for a render.

``Scene.get_frames(..., aux_passes=True, aux_sink=callback)`` hands
``callback`` one dict per yielded frame batch, immediately before the batch is
yielded, holding CPU tensors whose leading dimension is that batch's frame
count:

``"depth"`` -- ``float32 [F, H, W]``
    Planar camera depth in world units: the distance from the camera location
    along its forward axis to the recorded surface. ``+inf`` where the pixel
    records nothing.
``"normal"`` -- ``float32 [F, H, W, 3]``
    The recorded surface's unit shading normal, turned toward the viewer, in
    camera space: x screen right, y screen up, z toward the camera. Zero where
    the pixel records nothing.
``"mob_id"`` -- ``int32 [F, H, W]``
    ``Mob.id`` of the recorded surface's source, read through the optional
    host tables ``merged["tri_obj_source_ids"]`` (global surface id -> Mob id)
    and ``merged["circuit_source_ids"]`` (merged circuit index -> Mob id);
    ``-1`` when the table or its entry is missing, ``-2`` where the pixel
    records nothing.

Rows are top-down, at the OUTPUT resolution, in the order of the yielded
frames. Treat the tensors as read-only: a background-only batch's are
broadcast views (:func:`empty_aux_passes`).

What a pixel records is Cycles' ``pass_alpha_threshold`` rule: the first
surface along ONE pinhole ray through the pixel centre, between the camera's
near plane and far distance, whose colour alpha is at least
:data:`AUX_ALPHA_THRESHOLD`. That ray does not depend on ``samples_per_pixel``,
on the anti-aliasing route or on a camera lens, so every renderer produces the
same passes for the same scene. The trace itself is
``aux_passes_taichi.aux_trace``; this module drives it and packages its
output.

Cost model: nothing here runs unless passes were requested. When they are, the
tracer calls :func:`trace_aux_passes` inside each render chunk after the
chunk's own scratch has been released, so the pass's arena scratch (about 20
bytes per output pixel per frame, plus two small id tables and a copy of the
circuit metadata) overlaps the render's rather than adding to it, and the
chunk memory model measures it like any other allocation.
"""

from __future__ import annotations

import numpy as np
import torch

#: Coverage a surface needs to be recorded (Cycles' pass_alpha_threshold).
AUX_ALPHA_THRESHOLD = 0.5
#: ``mob_id`` of a pixel that records no surface.
MISS_MOB_ID = -2
#: ``mob_id`` of a recorded surface whose source Mob is unknown.
UNKNOWN_MOB_ID = -1
#: The keys of every aux dict, in a fixed order.
AUX_PASS_KEYS = ("depth", "normal", "mob_id")

#: Scale on the circuit classification's pixel size. ``_collect_hits`` dilates
#: a filled circuit by a hard-coded 0.6 of a RENDER pixel (anti-crack for the
#: renderers' coverage), which would make a centre sample depend on the AA
#: route and claim pixels whose centre lies outside the shape. Shrinking the
#: pixel size by this factor, while the border column of a circuit_meta copy
#: grows by its inverse, keeps every stroke its world width and takes the
#: dilation to 0.0006 px. A power of two, so both rescalings are exact.
_CIRCUIT_PIXEL_SCALE = 1.0 / 1024.0

_INT32_MAX = 2**31 - 1


def empty_aux_passes(num_frames, height, width):
    """Aux passes for ``num_frames`` frames on which nothing was recorded.

    What a background-only frame batch (no renderable primitives in its window)
    delivers: depth ``+inf``, normal zero, ``mob_id`` :data:`MISS_MOB_ID`.

    The tensors are one frame each broadcast along the frame axis (stride 0),
    so treat them as read-only: a background-only window can be hundreds of
    frames long, and at 20 bytes per pixel materialized passes would cost
    gigabytes of host memory to say "nothing here" (``.contiguous()`` makes a
    writable copy).
    """
    num_frames, height, width = int(num_frames), int(height), int(width)
    shape = (num_frames, height, width)
    depth = torch.full((1, height, width), float("inf"))
    normal = torch.zeros((1, height, width, 3))
    mob_id = torch.full((1, height, width), MISS_MOB_ID, dtype=torch.int32)
    return {
        "depth": depth.expand(shape),
        "normal": normal.expand(*shape, 3),
        "mob_id": mob_id.expand(shape),
    }


def concat_aux_passes(parts, height, width):
    """Join per-chunk aux dicts along the frame axis, in order."""
    parts = [part for part in parts if part is not None]
    if not parts:
        return empty_aux_passes(0, height, width)
    if len(parts) == 1:
        return parts[0]
    return {key: torch.cat([part[key] for part in parts], 0) for key in AUX_PASS_KEYS}


def aux_frame_count(aux):
    """Frame count of an aux dict (its leading dimension)."""
    return int(aux["depth"].shape[0])


def _host_id_table(values):
    """An optional id table as a flat host int32 array (``None`` if absent)."""
    if values is None:
        return None
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    table = np.asarray(values)
    if table.size == 0:
        return None
    return np.ascontiguousarray(table.reshape(-1).astype(np.int32, copy=False))


def _arena_id_table(memory, values):
    """Upload an optional id table into the arena for the kernel's gather.

    Returns ``(tensor, valid_length)``; an absent table is a one-element
    placeholder with length 0, which the kernel reads as "unknown".
    """
    table = _host_id_table(values)
    n = 0 if table is None else int(table.shape[0])
    out = memory.get_tensor((max(n, 1),), torch.int32)
    if n:
        out.copy_(torch.from_numpy(table))
    else:
        out.fill_(UNKNOWN_MOB_ID)
    return out, n


def _circuit_meta_for_aux(memory, merged):
    """Arena copy of the circuit metadata for the aux trace.

    The border column is pre-divided by :data:`_CIRCUIT_PIXEL_SCALE` (see
    there), so strokes keep their world width at the trace's pixel size.
    """
    from algan.rendering.raytracing.raytrace_kernels_taichi import _M_BORDER_W

    meta = merged["circuit_meta"]
    if int(merged.get("num_circuits", 0)) <= 0 or meta.numel() == 0:
        return meta
    out = memory.get_tensor(tuple(meta.shape), torch.float32)
    out.copy_(meta)
    out[..., _M_BORDER_W] *= 1.0 / _CIRCUIT_PIXEL_SCALE
    return out


def trace_aux_passes(
    memory,
    merged,
    tri_bvh,
    bez_bvh,
    cam_origin,
    screen_point,
    pixel_basis_x,
    pixel_basis_y,
    pixel_world_scale,
    time_start,
    time_end,
    screen_width,
    screen_height,
    near_clip,
    far_clip,
    layer_offset_triangles,
    has_tri,
    has_bez,
    anti_alias_level,
):
    """Trace the aux passes of batch frames ``[time_start, time_end)``.

    Called by ``tracer.render_batch_raytraced``'s chunk renderer with the
    batch's live render inputs: the arena-resident merged scene and its
    UNTRIMMED trees (real ones -- a deferred build must have run), the arena
    camera arrays, ``pixel_world_scale`` as the render computed it (per render
    pixel, hence ``anti_alias_level``, the render's supersampling factor, which
    turns it back into one OUTPUT pixel for the texture footprint), and the
    batch-relative frame range the chunk rendered. ``screen_width`` and
    ``screen_height`` are the OUTPUT resolution.

    Scratch comes from ``memory``; the caller rewinds it afterwards. Returns
    the aux dict (see the module docstring) as fresh CPU tensors that never
    alias the arena.
    """
    from algan.rendering.raytracing.aux_passes_taichi import aux_trace
    from algan.rendering.raytracing.refit_bvh import RefitBVH

    num_frames = int(time_end) - int(time_start)
    width = int(screen_width)
    height = int(screen_height)
    if num_frames <= 0 or width <= 0 or height <= 0:
        return empty_aux_passes(max(num_frames, 0), height, width)
    pixels = width * height
    refit = 1 if isinstance(tri_bvh, RefitBVH) else 0

    with memory.scope("aux_passes", aux_cells=num_frames * pixels):
        aux_depth = memory.get_tensor((num_frames, pixels), torch.float32)
        aux_normal = memory.get_tensor((num_frames, pixels, 3), torch.float32)
        aux_mob = memory.get_tensor((num_frames, pixels), torch.int32)
        circuit_meta = _circuit_meta_for_aux(memory, merged)
        tri_src, num_tri_src = _arena_id_table(memory, merged.get("tri_obj_source_ids"))
        circ_src, num_circ_src = _arena_id_table(
            memory, merged.get("circuit_source_ids")
        )
    quad_base = merged.get("pt_quad_base")
    quad_base = _INT32_MAX if quad_base is None else int(quad_base)
    # One thread per (frame, pixel); keep each launch's flat index in int32.
    frames_per_launch = max(1, _INT32_MAX // pixels)
    for f0 in range(0, num_frames, frames_per_launch):
        f1 = min(num_frames, f0 + frames_per_launch)
        aux_trace(
            int((f1 - f0) * pixels),
            int(time_start) + f0,
            width,
            height,
            float(width // 2),
            float(height // 2),
            float(near_clip),
            float(far_clip),
            float(AUX_ALPHA_THRESHOLD),
            float(layer_offset_triangles),
            int(merged["num_colored_triangles"]),
            float(_CIRCUIT_PIXEL_SCALE),
            float(max(1, int(anti_alias_level))),
            quad_base,
            num_tri_src,
            num_circ_src,
            tri_bvh.blocks,
            tri_bvh.leaf_prim,
            tri_bvh.leaf_tspan,
            int(tri_bvh.first_leaf),
            merged["tri_pos"],
            merged["tri_norm"],
            merged["tri_colors"],
            merged["tri_uvs"],
            merged["tri_tex_meta"],
            merged["textures"],
            merged["tri_mat_id"],
            merged["tri_mat"],
            merged["tri_obj"],
            bez_bvh.blocks,
            bez_bvh.leaf_prim,
            bez_bvh.leaf_tspan,
            int(bez_bvh.first_leaf),
            circuit_meta,
            merged["circuit_colors"],
            merged["circuit_border_colors"],
            merged["edges_2d"],
            merged["edge_accel"],
            cam_origin,
            screen_point,
            pixel_basis_x,
            pixel_basis_y,
            pixel_world_scale,
            tri_src,
            circ_src,
            refit,
            int(has_tri),
            int(has_bez),
            aux_depth[f0:f1],
            aux_normal[f0:f1],
            aux_mob[f0:f1],
        )
    # ``copy=True``: on a CPU render ``.to("cpu")`` would return the arena
    # view itself, which the next chunk overwrites.
    depth = aux_depth.to("cpu", copy=True).view(num_frames, height, width)
    normal = aux_normal.to("cpu", copy=True).view(num_frames, height, width, 3)
    mob_id = aux_mob.to("cpu", copy=True).view(num_frames, height, width)
    depth.masked_fill_(mob_id == MISS_MOB_ID, float("inf"))
    return {"depth": depth, "normal": normal, "mob_id": mob_id}
