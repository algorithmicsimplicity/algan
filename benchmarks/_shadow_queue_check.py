"""Matched-input shadow parity and alternating warm full-render A/B.

Run on the actual render device, with the published Quadrants compiler:
    python benchmarks/_shadow_queue_check.py --width 160 --height 90 --pairs 2
Every queue is compared before shade consumes it, avoiding reflective split
pixels' unrelated atomic-accumulation noise. Results are emitted immediately.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from algan import SETTINGS
from algan.rendering.raytracing import raster_taichi as kernels
from algan.rendering.raytracing import shadow_queue
from algan.settings._startup import render_device
from algan.taichi_compat import ti
from benchmarks._shadow_budget_check import render as render_soft


def sync():
    ti.sync()
    device = torch.device(render_device()).type
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def render_graphics(out_prefix, resolution):
    from algan import PREVIEW, Scene
    from algan.scene_manager import SceneManager
    from benchmarks.performance.graphics_scene import scene as record

    SceneManager.reset()
    settings = PREVIEW.set(resolution=resolution)
    scene = Scene()
    scene.set_video_settings(settings)
    record(1.0)
    return scene.save_frame(out_prefix + ".png", settings, at=0.5, overwrite=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute visibility tolerance (default: exact). Pixel tolerance stays 2.",
    )
    parser.add_argument("--scene", choices=("soft", "graphics"), default="soft")
    parser.add_argument(
        "--queue-pairs",
        type=int,
        default=0,
        help="Also time matched shadow queues, independently of full renders",
    )
    parser.add_argument(
        "--sort-ab",
        action="store_true",
        help="Include a ray-parallel arm without secondary sorting",
    )
    args = parser.parse_args()
    if (
        min(args.width, args.height, args.pairs) < 1
        or args.queue_pairs < 0
        or not np.isfinite(args.atol)
        or args.atol < 0
    ):
        parser.error(
            "resolution and pairs must be positive; queue-pairs cannot be negative"
        )
    render = render_graphics if args.scene == "graphics" else render_soft
    device = str(render_device())
    print(
        "SHADOW_QUEUE "
        + json.dumps(
            {
                "event": "metadata",
                "device": device,
                "compiler": str(ti.__version__),
                "resolution": [args.width, args.height],
                "scene": args.scene,
            }
        ),
        flush=True,
    )
    if args.require_mps and device != "mps":
        raise RuntimeError(f"Requested GPU validation but Algan resolved {device}")
    out = Path("algan_outputs/shadow_queue") / args.scene
    out.mkdir(parents=True, exist_ok=True)
    real_make = shadow_queue.make_shadow_tracer
    stats = {
        "calls": 0,
        "primary": 0,
        "secondary": 0,
        "max_abs": 0.0,
        "scalar_values": 0,
        "differing_values": 0,
        "visibility_atol": args.atol,
    }
    vis_index = kernels._RASTER_SHADOW_TRACE_PARAMS.index("shadow_vis")
    secondary_index = kernels._RASTER_SHADOW_TRACE_PARAMS.index("secondary")
    queue_times = {"serial": [], "parallel": []}
    old_threshold = shadow_queue._SORT_MIN_RAYS
    shadow_queue._SORT_MIN_RAYS = 1  # ensure the sorted arm is actually exercised

    def checked_make(memory, sort_sources=None):
        SETTINGS.raytracing.experimental.shadow_ray_parallel = True
        candidate = real_make(memory, sort_sources)

        def check(*call):
            kernels.raster_shadow_trace(*call)
            expected = call[vis_index].clone()
            before = memory.get_pointers()
            candidate(*call)
            sync()
            actual = call[vis_index]
            if memory.get_pointers() != before:
                raise AssertionError("queue leaked arena allocation")
            delta = (actual - expected).abs()
            diff = float(delta.max().item())
            stats["differing_values"] += int((actual != expected).sum().item())
            stats["max_abs"] = max(stats["max_abs"], diff)
            stats["calls"] += 1
            stats["secondary" if call[secondary_index] else "primary"] += 1
            stats["scalar_values"] += actual.numel()
            if not torch.allclose(actual, expected, rtol=0, atol=args.atol):
                raise AssertionError(
                    f"Shadow visibility mismatch: max abs {diff}, call {stats['calls']}"
                )
            if args.queue_pairs:
                readings = {"serial": [], "parallel": []}
                for mode in [
                    "serial",
                    "parallel",
                    "parallel",
                    "serial",
                ] * args.queue_pairs:
                    sync()
                    start = time.perf_counter()
                    (candidate if mode == "parallel" else kernels.raster_shadow_trace)(
                        *call
                    )
                    sync()
                    readings[mode].append(time.perf_counter() - start)
                for name, values in readings.items():
                    queue_times[name].append(float(np.median(values)))
                print(
                    "SHADOW_QUEUE "
                    + json.dumps(
                        {
                            "event": "queue",
                            "secondary": bool(call[secondary_index]),
                            "events": call[0],
                            "median_seconds": {
                                k: float(np.median(v)) for k, v in readings.items()
                            },
                        }
                    ),
                    flush=True,
                )

        return check

    SETTINGS.raytracing.experimental.shadow_ray_parallel = True
    SETTINGS.raytracing.experimental.shadow_secondary_sort = True
    shadow_queue.make_shadow_tracer = checked_make
    try:
        render(str(out / "parity"), resolution=(args.width, args.height))
    finally:
        shadow_queue.make_shadow_tracer = real_make
        shadow_queue._SORT_MIN_RAYS = old_threshold
    if not stats["primary"] or not stats["secondary"]:
        raise AssertionError(f"Missing primary/secondary coverage: {stats}")
    print("SHADOW_QUEUE " + json.dumps({"event": "parity", **stats}), flush=True)

    # Warm every arm, then alternate mirrored orders. The unsorted arm isolates
    # the scheduling sort cost from the one-worker-per-ray kernel change.
    modes = ["serial", "parallel"]
    if args.sort_ab:
        modes.insert(1, "parallel_unsorted")
    times = {name: [] for name in modes}
    order = modes + (modes + list(reversed(modes))) * args.pairs
    for i, name in enumerate(order):
        SETTINGS.raytracing.experimental.shadow_ray_parallel = name != "serial"
        SETTINGS.raytracing.experimental.shadow_secondary_sort = (
            name != "parallel_unsorted"
        )
        sync()
        start = time.perf_counter()
        render(str(out / name), resolution=(args.width, args.height))
        sync()
        elapsed = time.perf_counter() - start
        if i >= len(modes):
            times[name].append(elapsed)
        print(
            "SHADOW_QUEUE "
            + json.dumps(
                {
                    "event": "render",
                    "arm": name,
                    "warm": i >= len(modes),
                    "seconds": elapsed,
                }
            ),
            flush=True,
        )
    medians = {name: float(np.median(values)) for name, values in times.items()}
    from PIL import Image

    frames = {
        name: np.asarray(
            Image.open(out / (name + ".png")).convert("RGB"), dtype=np.int16
        )
        for name in modes
    }
    frame_diffs = {
        name: int(np.abs(frame - frames["serial"]).max())
        for name, frame in frames.items()
        if name != "serial"
    }
    if any(value > 2 for value in frame_diffs.values()):
        raise AssertionError(f"Rendered pixel regression: {frame_diffs}")
    summary = {
        "event": "summary",
        "scene": args.scene,
        "device": device,
        "resolution": [args.width, args.height],
        "parity": stats,
        "seconds": times,
        "median_seconds": medians,
        "matched_queue_seconds": queue_times,
        "max_frame_channel_difference": frame_diffs,
        "parallel_over_serial": medians["parallel"] / medians["serial"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SHADOW_QUEUE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
