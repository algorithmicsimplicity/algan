"""Stable device-counted primary shadow queues with caller-owned arena scratch."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from algan.rendering.arena_regions import TypedArenaView
from algan.rendering.device_dispatch import DeviceCount, DispatchPlan, DispatchStage
from algan.rendering.raytracing import device_dispatch_taichi as kernels
from algan.taichi_compat import ti

_BLOCK = 256


def _reset(extent, bindings):
    kernels.reset_dispatch_header(extent.header.tensor)


def _count(extent, b):
    kernels.count_selected(
        b.n, b.inputs["accepted"], b.rank, b.totals, extent.header.tensor
    )


def _scan(extent, b):
    values = b.totals
    for n, offsets, totals in b.scan:
        kernels.scan_dispatch_blocks(n, values, offsets, totals)
        values = totals
    for i in range(len(b.scan) - 2, -1, -1):
        n, offsets, _ = b.scan[i]
        kernels.add_dispatch_carries(n, offsets, b.scan[i + 1][1])


def _pack(extent, b):
    src, out = b.inputs, b.outputs
    kernels.pack_primary_shadow_events(
        b.n,
        extent.capacity,
        extent.header.tensor,
        b.rank,
        b.scan[0][1],
        b.scan[-1][2],
        src["accepted"],
        src["source"],
        src["pos"],
        src["snrm"],
        src["fnrm"],
        src["frame"],
        src["mask"],
        src["dp"],
        src["toff"],
        out["pos"],
        out["snrm"],
        out["fnrm"],
        out["frame"],
        out["mask"],
        out["source"],
        out["dp"],
        out["toff"],
        b.reverse,
        int(b.footprint),
        int(b.terminator),
    )


_PRIMARY_SHADOW_PLAN = DispatchPlan(
    (
        DispatchStage("reset", _reset),
        DispatchStage("count", _count),
        DispatchStage("scan", _scan),
        DispatchStage("pack", _pack),
    )
)


def _synchronize(device):
    # Commit/error boundary, not a per-stage fence. Includes both frameworks.
    ti.sync()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass
class PrimaryShadowDispatch:
    n: int
    inputs: dict
    outputs: dict
    reverse: torch.Tensor
    footprint: bool
    terminator: bool
    rank: torch.Tensor
    totals: torch.Tensor
    scan: tuple
    extent: DeviceCount
    views: tuple

    def run(self, trace, shadow_vis):
        output = TypedArenaView.bind(
            shadow_vis, name="shadow_vis", access="write", dtype=torch.float32, ndim=3
        )
        if shadow_vis.shape[0] < max(1, self.extent.capacity):
            raise ValueError("Shadow visibility buffer is smaller than queue capacity")
        return _PRIMARY_SHADOW_PLAN.run(
            self.extent,
            self.views + (output,),
            self,
            consume=lambda _extent, _bindings, _metadata: trace(),
            synchronize=lambda: _synchronize(shadow_vis.device),
        )


def prepare_primary_shadow_dispatch(
    memory,
    *,
    n,
    accepted,
    source,
    pos,
    snrm,
    fnrm,
    frame,
    mask,
    dp,
    toff,
    reverse,
    footprint,
    terminator,
    capacity=None,
):
    """Allocate and bind without executing. Normal capacity n cannot overflow.

    Smaller explicit capacities are supported for retry/contract tests, never
    by dropping an accepted suffix. All scan sizes derive from checked n.
    """
    if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= (1 << 31) - 1:
        raise ValueError("Source count must fit signed int32")
    capacity = n if capacity is None else capacity
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, int)
        or not 0 <= capacity <= (1 << 31) - 1
    ):
        raise ValueError("Queue capacity must fit signed int32")
    allocate = memory.get_tensor
    inputs = {
        "accepted": accepted,
        "source": source,
        "pos": pos,
        "snrm": snrm,
        "fnrm": fnrm,
        "frame": frame,
        "mask": mask,
        "dp": dp,
        "toff": toff,
    }
    views = []
    for name, tensor in inputs.items():
        width = (
            6 if name == "dp" else 3 if name in ("pos", "snrm", "fnrm", "toff") else 1
        )
        dtype = torch.float32 if width > 1 else torch.int32
        view = TypedArenaView.bind(
            tensor,
            name=name,
            access="read",
            dtype=dtype,
            ndim=2 if width > 1 else 1,
            vector_width=width,
        )
        required = (
            n if (name != "dp" or footprint) and (name != "toff" or terminator) else 1
        )
        if tensor.shape[0] < required:
            raise ValueError(f"{name}: source buffer is too small")
        views.append(view)
    devices = {v.tensor.device for v in views}
    if len(devices) != 1 or memory.data.device not in devices:
        raise ValueError("All dispatch bindings must be on the arena's device")
    outputs = {}
    for name in ("pos", "snrm", "fnrm", "frame", "mask", "source", "dp", "toff"):
        width = (
            6 if name == "dp" else 3 if name in ("pos", "snrm", "fnrm", "toff") else 1
        )
        size = max(1, capacity)
        if (name == "dp" and not footprint) or (name == "toff" and not terminator):
            size = 1
        tensor = allocate(
            (size, width) if width > 1 else (size,),
            torch.float32 if width > 1 else torch.int32,
        )
        outputs[name] = tensor
        views.append(TypedArenaView.bind(tensor, name="out_" + name, access="write"))
    if reverse.dtype != torch.int32 or reverse.ndim != 1 or reverse.shape[0] < n:
        raise ValueError("Reverse mapping must be int32 with one slot per source")
    views.append(TypedArenaView.bind(reverse, name="reverse", access="write"))
    header = TypedArenaView.bind(
        allocate((4,), torch.int32), name="header", access="readwrite"
    )
    extent = DeviceCount(header, capacity)
    views.append(header)
    rank = allocate((max(1, n),), torch.int32)
    nb = max(1, (n + _BLOCK - 1) // _BLOCK)
    totals = allocate((nb,), torch.int32)
    views.extend(
        (
            TypedArenaView.bind(rank, name="rank", access="readwrite"),
            TypedArenaView.bind(totals, name="totals", access="readwrite"),
        )
    )
    scan = []
    level = 0
    while True:
        parents = max(1, (nb + _BLOCK - 1) // _BLOCK)
        offsets = allocate((nb,), torch.int32)
        sums = allocate((parents,), torch.int32)
        scan.append((nb, offsets, sums))
        views.extend(
            (
                TypedArenaView.bind(
                    offsets, name=f"offsets_{level}", access="readwrite"
                ),
                TypedArenaView.bind(sums, name=f"sums_{level}", access="readwrite"),
            )
        )
        if parents == 1:
            break
        nb, level = parents, level + 1
    return PrimaryShadowDispatch(
        n,
        inputs,
        outputs,
        reverse,
        bool(footprint),
        bool(terminator),
        rank,
        totals,
        tuple(scan),
        extent,
        tuple(views),
    )
