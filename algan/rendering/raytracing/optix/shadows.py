"""Shadow rays on RT cores: an arm of the light-major shadow queue.

``shadow_queue.make_shadow_tracer`` hands a batch here when
``SETTINGS.raytracing.experimental.optix`` is on. The queue's own kernels
still emit the rays (``shadow_queue_prepare``, the fan body every shadow path
shares) and reduce them in reference order (``shadow_queue_reduce``); only the
traversal in between changes: an OptiX launch per adaptive-tap phase, then
``optix_shadow_resolve`` for the rays that crossed translucent casters.

Triangles only: a batch's Bezier circuits are marched in software inside
the resolve and their transmittance multiplied on. A launch keeps the software
queue (or the serial fan) when the batch has no triangles or a GAS cannot be
built.
"""

from __future__ import annotations

import torch

from algan.logging.logger import get_logger
from algan.rendering.raytracing import raster_taichi as kernels
from algan.rendering.raytracing.optix._abi import OptixError
from algan.rendering.raytracing.optix.runtime import SHADOW_HITS, new_params

#: Diagnostics for tests and the parity harness: rays traced, rays the
#: hardware answered alone, rays resolved from a list, rays the software march
#: re-ran. Counting costs a readback per chunk, so it is off unless
#: ``COUNTING`` is set.
COUNTING = False
STATS = {"launches": 0, "rays": 0, "answered": 0, "listed": 0, "marched": 0}

#: Per-ray check against the software walk: with ``VERIFY`` set, every
#: phase-0 queue chunk is traced a second time by ``shadow_queue_trace`` and
#: the two occlusions compared ray by ray into ``VERIFY_STATS``. Slow; for
#: tests and the parity harness.
VERIFY = False
VERIFY_STATS = {
    "rays": 0,
    "identical": 0,
    "within_1e-6": 0,
    "lit_vs_blocked": 0,
    "max_abs": 0.0,
}

logger = get_logger(__name__)

_LAUNCH_FAILURE_LOGGED = [False]


def _launch_failed(exc):
    """Log a launch failure once; the caller falls back to software."""
    if not _LAUNCH_FAILURE_LOGGED[0]:
        _LAUNCH_FAILURE_LOGGED[0] = True
        logger.warning(
            "OptiX: a shadow launch failed, so those shadow rays were traced in "
            "software: %s",
            exc,
        )


#: Per-ray scratch beyond the queue's own: the status word and the list.
_EXTRA_BYTES_PER_RAY = 4 + 16 * SHADOW_HITS

_GAS_KEY = "_optix_shadow_gas"

#: Arena bytes one queue chunk may take (the software queue's is 64 MiB). Each
#: chunk costs six launches whatever its size, and at 112 bytes a ray the
#: software queue's chunks are small enough for the host to dominate.
SCRATCH_LIMIT = 512 * 1024 * 1024

#: Trace the adaptive hard-light taps in shadow_queue_trace's two phases
#: (diagonal pair, then the off-diagonal pair where the first disagreed)
#: instead of all four in one launch. Same answers; the single launch traces
#: more rays and launches half as often, which is what the RT cores want.
TWO_PHASE = False


def batch_runtime(device):
    """The OptiX runtime for ``device``, or ``None`` (logged once)."""
    from algan.rendering.raytracing.optix import runtime_for

    return runtime_for(device)


def shadow_gas(runtime, merged, tri_pos):
    """This batch's shadow GASes (casters only), built on first use."""
    from algan.rendering.raytracing import raytrace_kernels_taichi as rk
    from algan.rendering.raytracing.optix.gas import build_batch_gas

    cached = merged.get(_GAS_KEY)
    if cached is not None:
        return cached or None
    device = tri_pos.device
    valid = merged.get("tri_frame_valid")
    opaque = merged.get("tri_frame_opaque")
    casts = merged.get("tri_frame_casts")
    gas = None
    if valid is not None and opaque is not None:
        if casts is None:
            casts = torch.ones_like(valid)
        # An opaque hit ends a ray in hardware as full occlusion, which is the
        # march's answer only while an opaque surface's leftover light (alpha
        # 1 - 1e-6, pass-through 1e-6) falls under min_alpha and the march's
        # surface cap cannot stop it short (shadow_dispatch's own bounds).
        exact_opaque = rk.min_alpha >= 1e-5 and rk.max_surfaces_per_ray >= 256
        gas = build_batch_gas(
            runtime,
            tri_pos,
            valid.to(device),
            opaque.to(device),
            casts=casts.to(device),
            all_translucent=not exact_opaque,
        )
    merged[_GAS_KEY] = gas if gas is not None else False
    return gas


def make_optix_shadow_tracer(memory, merged, software):
    """A ``raster_shadow_trace``-compatible tracer on RT cores, or ``None``.

    ``software`` is the tracer this batch would use otherwise; every launch
    the OptiX arm cannot take goes there.
    """
    if merged is None or memory is None:
        return None
    from algan.rendering.raytracing.optix_shadow_taichi import optix_shadow_resolve
    from algan.rendering.raytracing.raytrace_kernels_taichi import (
        INV_DEPTH_TIE_EPSILON,
        depth_tie_epsilon,
        min_hit_distance,
    )
    from algan.rendering.raytracing.shadow_queue import _event_capacity, _fan_layout

    def trace(*args):
        p = dict(zip(kernels._RASTER_SHADOW_TRACE_PARAMS, args, strict=True))
        events, lights = int(p["num_events"]), int(p["num_lights"])
        tri_pos = p["tri_pos"]
        if (
            events == 0
            or lights == 0
            or not int(p["has_tri"])
            or tri_pos.device.type != "cuda"
        ):
            return software(*args)
        runtime = batch_runtime(tri_pos.device)
        if runtime is None:
            return software(*args)
        gas = shadow_gas(runtime, merged, tri_pos)
        if gas is None:
            return software(*args)
        offsets, layout, slots = _fan_layout(
            p["light_col"], lights, int(p["sec_aa"]), int(p["secondary"])
        )
        capacity = _event_capacity(
            memory,
            events,
            slots,
            False,
            extra_bytes_per_ray=_EXTRA_BYTES_PER_RAY,
            limit=SCRATCH_LIMIT,
        )
        if capacity < 1:
            return software(*args)
        identity = int(p["shadow_identity"]) != 0
        tri_obj = p["tri_obj"]
        arena_tail = [p[name] for name, _, _ in kernels._SHADOW_QUEUE_TRACE_ARENA]
        adaptive = int(p["sec_aa"]) > 1 and int(p["adaptive_taps"])
        # One launch over every tap (phase 2) unless asked for the reference
        # two-phase schedule; see the raygen.
        phases = ((0, 1) if adaptive else (0,)) if TWO_PHASE else (2,)
        try:
            with memory.temp():
                ray_data = memory.get_tensor((capacity * slots, 7), torch.float32)
                valid = memory.get_tensor((capacity * slots,), torch.int32)
                occ = memory.get_tensor((capacity * slots, 3), torch.float32)
                status = memory.get_tensor((capacity * slots,), torch.int32)
                hits = memory.get_tensor(
                    (capacity * slots, 4 * SHADOW_HITS), torch.float32
                )
                keys = memory.get_tensor((1,), torch.int64)
                if VERIFY:
                    occ_sw = memory.get_tensor((capacity * slots, 3), torch.float32)
                params = new_params()
                params.handles = gas.handles.data_ptr()
                params.num_handles = gas.num_handles
                params.min_hit_distance = float(min_hit_distance)
                params.inv_bin = float(INV_DEPTH_TIE_EPSILON)
                params.bin_width = float(depth_tie_epsilon)
                params.layer_offset = float(p["layer_offset_triangles"])
                params.queue_data = ray_data.data_ptr()
                params.queue_valid = valid.data_ptr()
                params.layout = layout.data_ptr()
                params.occ_out = occ.data_ptr()
                params.event_frame = p["event_frame"].data_ptr()
                params.status = status.data_ptr()
                params.hits = hits.data_ptr()
                params.identity = 1 if identity else 0
                if identity:
                    params.event_src_prim = p["event_src_prim"].data_ptr()
                    params.tri_obj = tri_obj.data_ptr()
                    params.tri_obj_frames = int(tri_obj.shape[0])
                    params.tri_obj_count = int(tri_obj.shape[1])
                    params.eps_self = float(p["eps_self"])
                    params.eps_near = float(p["eps_near"])
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
                        valid,
                        int(tri_pos.shape[1]),
                        ray_data,
                        valid,
                        keys,
                        False,
                    )
                    params.num_rays = rays
                    params.num_events = count
                    params.event_first = first
                    for phase in phases:
                        params.phase = phase
                        runtime.launch("shadow", params, rays)
                        optix_shadow_resolve(
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
                            ray_data,
                            status,
                            hits,
                            occ,
                            *arena_tail,
                            p["projected"],
                        )
                        if COUNTING:
                            _count(status[:rays])
                        if VERIFY and phase in (0, 2):
                            _verify(
                                p,
                                count,
                                first,
                                rays,
                                layout,
                                ray_data,
                                valid,
                                keys,
                                occ,
                                occ_sw,
                                status,
                                arena_tail,
                            )
                    kernels.shadow_queue_reduce(
                        count, first, lights, offsets, valid, occ, p["shadow_vis"]
                    )

        except OptixError as exc:
            # A launch the driver refused (or past its 2^30-thread limit):
            # the software tracer redoes the whole call, rewriting every
            # (event, light) cell of shadow_vis, chunks already reduced
            # included.
            _launch_failed(exc)
            return software(*args)

    return trace


def _verify(
    p,
    count,
    first,
    rays,
    layout,
    ray_data,
    valid,
    order,
    occ,
    occ_sw,
    status,
    arena_tail,
):
    """``shadow_queue_trace`` over the same phase-0 rays, compared per ray."""
    kernels.shadow_queue_trace(
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
        occ_sw,
        False,
        0,
        *arena_tail,
        p["projected"],
    )
    traced = status[:rays] != 0
    # Phase 0 of the software queue skips the hard fans' off-diagonal taps.
    sk = layout[:, 0].repeat_interleave(count)[:rays]
    traced &= ~((valid[:rays] == 2) & (sk >= 2))
    a = occ[:rays][traced]
    b = occ_sw[:rays][traced]
    diff = (a - b).abs().amax(-1)
    VERIFY_STATS["rays"] += int(traced.sum())
    VERIFY_STATS["identical"] += int((diff == 0).sum())
    VERIFY_STATS["within_1e-6"] += int((diff <= 1e-6).sum())
    blocked_a = (a == 1).all(-1)
    blocked_b = (b == 1).all(-1)
    lit_a = (a == 0).all(-1)
    lit_b = (b == 0).all(-1)
    VERIFY_STATS["lit_vs_blocked"] += int(
        ((blocked_a & lit_b) | (lit_a & blocked_b)).sum()
    )
    if diff.numel():
        VERIFY_STATS["max_abs"] = max(VERIFY_STATS["max_abs"], float(diff.max()))


def _count(status):
    STATS["launches"] += 1
    traced = status != 0
    STATS["rays"] += int(traced.sum())
    STATS["answered"] += int((status == 1).sum())
    STATS["listed"] += int((status >= 2).sum())
    STATS["marched"] += int((status == -1).sum())
