"""Matched warm video A/B for fused compaction, run construction and fragment sorting.

Every arm records the same complete workload clip. Cold runs are excluded;
mirrored arm ordering limits thermal/order bias. Videos are encoded losslessly
so decoded-frame parity measures renderer output, not lossy codec propagation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("ALGAN_USE_DAEMON", "0")
os.environ.setdefault("ALGAN_VIDEO_ENCODER", "software")

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import algan  # noqa: E402
from algan import SETTINGS, Scene  # noqa: E402
from algan.rendering.raytracing import sheet_stream  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402
from algan.settings._startup import render_device  # noqa: E402
from algan.taichi_compat import ti  # noqa: E402

SWITCHES = ("sheet_fused_stream", "sheet_device_runs", "sheet_fragment_run_sort")
ARMS = {
    "reference": (False, False, False),
    "fused": (True, False, False),
    "runs": (False, True, False),
    "sort": (False, False, True),
    "all": (True, True, True),
}


def sync():
    # Before the first render, Quadrants need not have a program yet.
    from algan.rendering.taichi_runtime import _live_arch

    if _live_arch() is not None:
        ti.sync()
    if render_device().type == "mps":
        torch.mps.synchronize()
    elif render_device().type == "cuda":
        torch.cuda.synchronize()


def compare_videos(reference, candidate):
    a = imageio.get_reader(str(reference), format="ffmpeg")
    b = imageio.get_reader(str(candidate), format="ffmpeg")
    maximum = changed = frames = 0
    from itertools import zip_longest

    try:
        for left, right in zip_longest(a, b):
            if left is None or right is None or left.shape != right.shape:
                raise AssertionError("video frame counts/shapes differ")
            delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
            maximum = max(maximum, int(delta.max()))
            changed += int(np.count_nonzero(delta))
            frames += 1
    finally:
        a.close()
        b.close()
    return {
        "frames": frames,
        "max_channel_difference": maximum,
        "differing_channels": changed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", choices=("explainer", "graphics"), default="explainer"
    )
    parser.add_argument(
        "--quality", choices=("PREVIEW", "MD", "UHD", "SMOKE_TEST"), default="PREVIEW"
    )
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--arms", default="reference,all")
    parser.add_argument("--require-mps", action="store_true")
    args = parser.parse_args()
    names = args.arms.split(",")
    if (
        args.frames < 1
        or args.pairs < 1
        or "reference" not in names
        or any(name not in ARMS for name in names)
    ):
        parser.error(
            "positive frames/pairs and known arms including reference are required"
        )
    device = str(render_device())
    if args.require_mps and device != "mps":
        raise RuntimeError(f"MPS required, got {device}")
    preset = getattr(algan, args.quality)
    record = importlib.import_module(f"benchmarks.performance.{args.scene}_scene").scene
    out = (
        Path("algan_outputs/sheet_stream")
        / f"{args.scene}_{args.quality}_{args.frames}"
    )
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "event": "metadata",
        "device": device,
        "compiler": str(ti.__version__),
        "scene": args.scene,
        "quality": args.quality,
        "frames": args.frames,
        "resolution": preset.resolution,
        "arms": names,
        "encoding": "libx264 lossless",
    }
    print("SHEET_STREAM " + json.dumps(metadata), flush=True)
    counts = {}
    originals = {}
    for operation in (
        "gather_group_stream",
        "gather_sheet_records",
        "pixel_runs",
        "truncate_pixel_runs",
        "fragment_run_order",
    ):
        fn = getattr(sheet_stream, operation)
        originals[operation] = fn

        def counted(*call, _fn=fn, _name=operation, **kwargs):
            counts[_name] = counts.get(_name, 0) + 1
            return _fn(*call, **kwargs)

        setattr(sheet_stream, operation, counted)
    times = {name: [] for name in names}
    measurements = []
    try:
        sequence = names + (names + list(reversed(names))) * args.pairs
        for i, name in enumerate(sequence):
            for switch, enabled in zip(SWITCHES, ARMS[name]):
                setattr(SETTINGS.raytracing.experimental, switch, enabled)
            SETTINGS.raytracing.experimental.shadow_ray_parallel = False
            counts.clear()
            sync()
            start = time.perf_counter()
            SceneManager.reset()
            scene = Scene()
            scene.set_video_settings(preset)
            record(args.frames / preset.frames_per_second)
            result = scene.save_video(
                str(out / f"{name}.mp4"),
                preset,
                overwrite=True,
                animate_fade_out=False,
                ffmpeg_params=["-crf", "0", "-preset", "ultrafast"],
            )
            sync()
            elapsed = time.perf_counter() - start
            warm = i >= len(names)
            if warm:
                times[name].append(elapsed)
            reading = {
                "event": "render",
                "arm": name,
                "warm": warm,
                "seconds": elapsed,
                "engaged": dict(counts),
                "output": str(result.output_path),
            }
            measurements.append(reading)
            print("SHEET_STREAM " + json.dumps(reading), flush=True)
            # Save partial results even if a later variant or runner fails.
            (out / "readings.json").write_text(json.dumps(measurements, indent=2))
    finally:
        for name, fn in originals.items():
            setattr(sheet_stream, name, fn)
    parity = {
        name: compare_videos(out / "reference.mp4", out / f"{name}.mp4")
        for name in names
        if name != "reference"
    }
    medians = {name: statistics.median(values) for name, values in times.items()}
    summary = {
        **metadata,
        "event": "summary",
        "seconds": times,
        "median_seconds": medians,
        "parity": parity,
        "readings": measurements,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SHEET_STREAM " + json.dumps(summary), flush=True)
    if any(
        check["frames"] != args.frames or check["max_channel_difference"] > 2
        for check in parity.values()
    ):
        raise AssertionError(f"frame parity failure: {parity}")
    for name in names:
        if name == "reference":
            continue
        active = [r for r in measurements if r["arm"] == name][-1]["engaged"]
        needed = [
            ("gather_group_stream", ARMS[name][0]),
            ("pixel_runs", ARMS[name][1]),
            ("fragment_run_order", ARMS[name][2]),
        ]
        if any(enabled and not active.get(op) for op, enabled in needed):
            raise AssertionError(f"optimization did not engage for {name}: {active}")


if __name__ == "__main__":
    main()
