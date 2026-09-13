"""Compare CUDA shadow scheduling on identical live primary/bounce queues.

This uses real scene allocations and preserves each queue until both schedules
have consumed it. Device-profiler times exclude preparation and synchronization.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import torch  # noqa: E402

from algan import SETTINGS  # noqa: E402
from algan.rendering.raytracing import raster_taichi as kernels  # noqa: E402
from algan.rendering.raytracing import shadow_queue  # noqa: E402
from algan.settings._startup import render_device  # noqa: E402
from algan.taichi_compat import ti  # noqa: E402
from algan.utils.profiling_utils import (  # noqa: E402
    _collect_taichi_kernel_gpu,
    enable_taichi_kernel_profiler,
)
from benchmarks._shadow_queue_check import (  # noqa: E402
    render_graphics,
    render_soft,
    sync,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("graphics", "soft"), default="graphics")
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--height", type=int, default=396)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--tag", default="shadow_schedule")
    args = parser.parse_args()
    if min(args.width, args.height, args.pairs) < 1 or args.budget < 0:
        parser.error("positive dimensions/pairs and nonnegative budget required")
    if render_device().type != "cuda":
        raise RuntimeError("This benchmark requires CUDA")
    if not enable_taichi_kernel_profiler():
        raise RuntimeError("Device profiler unavailable")
    out = Path("algan_outputs") / args.tag
    out.mkdir(parents=True, exist_ok=True)
    data = {
        "gpu": torch.cuda.get_device_name(render_device()),
        "scene": args.scene,
        "resolution": [args.width, args.height],
        "budget": args.budget,
        "queues": [],
    }
    print("SHADOW_SCHEDULE " + json.dumps(data), flush=True)
    slots = kernels._RASTER_SHADOW_TRACE_PARAMS
    order_slot = slots.index("light_major")
    vis_slot = slots.index("shadow_vis")
    secondary_slot = slots.index("secondary")
    lights_slot = slots.index("num_lights")
    original = shadow_queue.make_shadow_tracer
    snapshot = SETTINGS.snapshot()

    def checked_make(memory, sort_sources=None):
        def check(*call):
            call = list(call)
            pointers = memory.get_pointers()
            call[order_slot] = False
            kernels.raster_shadow_trace(*call)
            expected = call[vis_slot].clone()
            call[order_slot] = True
            kernels.raster_shadow_trace(*call)
            sync()
            actual = call[vis_slot]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            readings = {"reference": [], "coherent": []}
            for name in ["reference", "coherent", "coherent", "reference"] * args.pairs:
                call[order_slot] = name == "coherent"
                ti.profiler.clear_kernel_profiler_info()
                kernels.raster_shadow_trace(*call)
                sync()
                rows = [
                    row
                    for row in _collect_taichi_kernel_gpu()
                    if row["name"] == "raster_shadow_trace_arena"
                ]
                if not rows:
                    raise AssertionError("No shadow device-time records")
                readings[name].append(sum(row["total_ms"] for row in rows))
            if memory.get_pointers() != pointers:
                raise AssertionError("Shadow scheduling leaked arena storage")
            row = {
                "secondary": bool(call[secondary_slot]),
                "events": call[0],
                "lights": call[lights_slot],
                "visibility_scalars": actual.numel(),
                "differing_values": 0,
                "device_ms": readings,
                "median_device_ms": {
                    name: statistics.median(values) for name, values in readings.items()
                },
            }
            data["queues"].append(row)
            print("SHADOW_SCHEDULE " + json.dumps(row), flush=True)

        return check

    try:
        SETTINGS.raytracing.experimental.set(
            shadow_ray_parallel=False,
            shadow_ray_budget=args.budget,
            shadow_bounce_rays=1 if args.budget else 0,
        )
        shadow_queue.make_shadow_tracer = checked_make
        render = render_graphics if args.scene == "graphics" else render_soft
        render(str(out / "parity"), resolution=(args.width, args.height))
        if {row["secondary"] for row in data["queues"]} != {False, True}:
            raise AssertionError("Both primary and bounce queues must be exercised")
        (out / "summary.json").write_text(json.dumps(data, indent=2))
        print("SHADOW_SCHEDULE " + json.dumps(data), flush=True)
    finally:
        shadow_queue.make_shadow_tracer = original
        SETTINGS.restore(snapshot)


if __name__ == "__main__":
    main()
