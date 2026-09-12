"""Bounded light-major shadow queues; the serial fan remains the fallback."""

from __future__ import annotations

import weakref

import torch

from algan.rendering.raytracing import raster_taichi as kernels
from algan.rendering.raytracing import settings as rt_settings
from algan.settings._startup import _SOFT_SHADOW_SAMPLES

# Metadata only, with weak ownership of the immutable packed-light table.
_LAYOUT_CACHE = {}
_SCRATCH_LIMIT = 64 * 1024 * 1024
_SORT_MIN_RAYS = 8192


def _fan_layout(light_col, num_lights, sec_aa, secondary):
    """Read small light metadata once per packed table, not once per bounce.

    Bound each light separately across the table's frames. This accepts the
    actual packed fan sizes (including custom budgets) rather than assuming
    the live settings still match a previously packed table.
    """
    # Arena slices share a version counter: filling unrelated ray scratch
    # invalidates light_col._version too. The renderer attaches the CPU table
    # it packed, which is immutable for this render job and avoids readbacks.
    source = getattr(light_col, "_algan_shadow_fan_metadata", light_col)
    key = (id(light_col), id(source), source._version, num_lights, sec_aa, secondary)
    cached = _LAYOUT_CACHE.get(key)
    if cached is not None and cached[0]() is light_col:
        return cached[1:]
    col = source.detach().cpu()
    counts = []
    for li in range(num_lights):
        bound = 1
        for row in col[:, li].tolist():
            radius = row[11] if len(row) > 11 else 0.0
            fan = int(row[16 + secondary] + 0.5) if len(row) > 17 else 0
            ns = (fan if fan > 0 else _SOFT_SHADOW_SAMPLES) if radius > 0 else 1
            if sec_aa > 1 and fan == 0:
                ns = max(ns, 4)
            bound = max(bound, ns)
        counts.append(bound)
    starts = [0]
    layout = []
    for count in counts:
        layout.extend((sk, starts[-1]) for sk in range(count))
        starts.append(starts[-1] + count)
    offsets = torch.tensor(starts, dtype=torch.int32, device=light_col.device)
    layout_tensor = torch.tensor(layout, dtype=torch.int32, device=light_col.device)
    if len(_LAYOUT_CACHE) >= 8:
        _LAYOUT_CACHE.clear()
    _LAYOUT_CACHE[key] = (weakref.ref(light_col), offsets, layout_tensor, starts[-1])
    return offsets, layout_tensor, starts[-1]


def _event_capacity(memory, events, fan_slots, sort_rays):
    # All queue payloads, sort indices and keys are arena-backed. Leave the
    # allocator's existing external headroom for the backend sort workspace.
    bytes_per_ray = 28 + 4 + 12 + (16 if sort_rays else 0)
    budget = min(_SCRATCH_LIMIT, max(0, memory.get_num_bytes_remaining() - 256))
    return min(events, budget // max(1, fan_slots * bytes_per_ray))


def make_shadow_tracer(memory, sort_sources=None):
    """Bind a job arena without changing the established kernel call contract.

    ``sort_sources`` is scheduling metadata, NOT shadow identity. In particular,
    bounce events retain their historical -1 acceptance identity while being
    ordered by the real source triangle and actual light-ray direction.
    """
    if not rt_settings.shadow_ray_parallel:
        return kernels.raster_shadow_trace

    def trace(*args):
        p = dict(zip(kernels._RASTER_SHADOW_TRACE_PARAMS, args, strict=True))
        events, lights = int(p["num_events"]), int(p["num_lights"])
        if events == 0 or lights == 0:
            return kernels.raster_shadow_trace(*args)
        sort_rays = bool(
            rt_settings.shadow_secondary_sort
            and p["secondary"]
            and sort_sources is not None
        )
        offsets, layout, slots = _fan_layout(
            p["light_col"], lights, int(p["sec_aa"]), int(p["secondary"])
        )
        sort_rays = sort_rays and events * slots >= _SORT_MIN_RAYS
        capacity = _event_capacity(memory, events, slots, sort_rays)
        if capacity < 1:
            return kernels.raster_shadow_trace(*args)
        arena_tail = [p[name] for name, _, _ in kernels._SHADOW_QUEUE_TRACE_ARENA]
        with memory.temp():
            ray_data = memory.get_tensor((capacity * slots, 7), torch.float32)
            valid = memory.get_tensor((capacity * slots,), torch.int32)
            occ = memory.get_tensor((capacity * slots, 3), torch.float32)
            keys = memory.get_tensor(
                (capacity * slots if sort_rays else 1,), torch.int64
            )
            order = memory.get_tensor(
                (capacity * slots if sort_rays else 1,), torch.int64
            )
            source = sort_sources if sort_sources is not None else valid
            for first in range(0, events, capacity):
                count = min(capacity, events - first)
                rays = count * slots
                kernels.shadow_queue_prepare(
                    count,
                    first,
                    lights,
                    p["event_pos"],
                    p["event_snrm"],
                    p["event_fnrm"],
                    p["event_frame"],
                    p["event_msk"],
                    p["event_dp"],
                    p["event_toff"],
                    p["light_pos"],
                    p["light_col"],
                    p["sec_aa"],
                    p["shadow_term"],
                    p["adaptive_taps"],
                    p["secondary"],
                    offsets,
                    source,
                    int(p["tri_pos"].shape[1]),
                    ray_data,
                    valid,
                    keys,
                    sort_rays,
                )
                if sort_rays:
                    torch.sort(
                        keys[:rays], stable=True, out=(keys[:rays], order[:rays])
                    )
                trace_args = (
                    count,
                    first,
                    rays,
                    p["event_frame"],
                    p["event_src_prim"],
                    p["t_nodes"],
                    p["t_first_leaf"],
                    p["num_colored_triangles"],
                    p["b_nodes"],
                    p["b_first_leaf"],
                    p["layer_offset_triangles"],
                    p["refit"],
                    p["has_tri"],
                    p["has_bez"],
                    p["shadow_anyhit"],
                    p["shadow_identity"],
                    p["eps_self"],
                    p["eps_near"],
                    layout,
                    ray_data,
                    valid,
                    order,
                    occ,
                    sort_rays,
                )
                kernels.shadow_queue_trace(*trace_args, 0, *arena_tail)
                if p["sec_aa"] > 1 and p["adaptive_taps"]:
                    kernels.shadow_queue_trace(*trace_args, 1, *arena_tail)
                kernels.shadow_queue_reduce(
                    count, first, lights, offsets, valid, occ, p["shadow_vis"]
                )

    return trace
