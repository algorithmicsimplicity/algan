"""Warm mirrored CUDA workload comparisons with lossless frame parity.

Run the complete normalized storyboard in every arm. Only measured renders
enter the summary; each arm first pays for its own compilation and caches.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("ALGAN_USE_DAEMON", "0")
os.environ.setdefault("ALGAN_VIDEO_ENCODER", "software")

import torch  # noqa: E402

import algan  # noqa: E402
from algan import SETTINGS, Scene  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402
from algan.settings._startup import render_device  # noqa: E402
from benchmarks._sheet_stream_check import compare_videos, sync  # noqa: E402

ARMS = {
    "reference": (False, False, False, False),
    "coherent": (True, False, False, False),
    "fused": (False, True, False, False),
    "combined": (True, True, False, True),
    "parallel": (False, False, True, False),
    "pinned": (False, False, False, True),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("explainer", "graphics"), required=True)
    parser.add_argument(
        "--quality", choices=("PREVIEW", "MD", "UHD", "SMOKE_TEST"), default="PREVIEW"
    )
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--arms", default="reference,coherent")
    parser.add_argument("--tag", default="workload_ab")
    args = parser.parse_args()
    names = args.arms.split(",")
    if (
        args.frames < 1
        or args.pairs < 1
        or args.warmups < 1
        or "reference" not in names
        or any(n not in ARMS for n in names)
    ):
        parser.error(
            "positive frames/pairs and known arms including reference required"
        )
    device = render_device()
    if device.type != "cuda":
        raise RuntimeError(f"CUDA required, got {device}")
    preset = getattr(algan, args.quality)
    record = importlib.import_module(f"benchmarks.performance.{args.scene}_scene").scene
    out = (
        Path("algan_outputs") / args.tag / f"{args.scene}_{args.quality}_{args.frames}"
    )
    out.mkdir(parents=True, exist_ok=True)
    data = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "scene": args.scene,
        "frames": args.frames,
        "resolution": preset.resolution,
        "arms": names,
        "compiler": [
            (d.metadata["Name"], d.version)
            for d in importlib.metadata.distributions()
            if "quadrant" in d.metadata["Name"].lower()
        ],
        "measurements": [],
    }
    print("WORKLOAD_AB " + json.dumps(data), flush=True)
    times = {name: [] for name in names}
    snapshot = SETTINGS.snapshot()
    try:
        sequence = names * args.warmups + (names + list(reversed(names))) * args.pairs
        for index, name in enumerate(sequence):
            major, fused, parallel, pinned = ARMS[name]
            SETTINGS.raytracing.experimental.set(
                shadow_light_major=major,
                sheet_fused_stream=fused,
                sheet_device_runs=False,
                sheet_fragment_run_sort=False,
                shadow_ray_parallel=parallel,
                pinned_frame_readback=pinned,
            )
            SceneManager.reset()
            scene = Scene()
            scene.set_video_settings(preset)
            sync()
            start = time.perf_counter()
            record(args.frames / preset.frames_per_second)
            authored = time.perf_counter()
            result = scene.save_video(
                str(out / f"{name}.mp4"),
                preset,
                overwrite=True,
                animate_fade_out=False,
                ffmpeg_params=["-crf", "0", "-preset", "ultrafast"],
            )
            sync()
            elapsed = time.perf_counter() - start
            warm = index >= len(names) * args.warmups
            if warm:
                times[name].append(elapsed)
            item = {
                "arm": name,
                "warm": warm,
                "seconds": elapsed,
                "author_seconds": authored - start,
                "render_seconds": elapsed - (authored - start),
                "authored_duration": float(scene._recorded_end_time_for_render()),
                "output": str(result.output_path),
            }
            data["measurements"].append(item)
            (out / "summary.json").write_text(json.dumps(data, indent=2))
            print("WORKLOAD_AB " + json.dumps(item), flush=True)
        data["medians"] = {
            name: statistics.median(values) for name, values in times.items()
        }
        data["means"] = {
            name: statistics.mean(values) for name, values in times.items()
        }
        data["parity"] = {
            name: compare_videos(out / "reference.mp4", out / f"{name}.mp4")
            for name in names
            if name != "reference"
        }
        (out / "summary.json").write_text(json.dumps(data, indent=2))
        print("WORKLOAD_AB " + json.dumps(data), flush=True)
        if any(p["max_channel_difference"] > 2 for p in data["parity"].values()):
            raise AssertionError(f"Renderer pixel regression: {data['parity']}")
    finally:
        SETTINGS.restore(snapshot)


if __name__ == "__main__":
    main()
