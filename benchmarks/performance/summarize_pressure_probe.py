"""Summarize pressure telemetry and compare every completed lossless render."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _compare_videos(reference, candidate):
    # Decode without importing the renderer or creating a CUDA context.
    from itertools import zip_longest

    import imageio.v2 as imageio
    import numpy as np

    left_reader = imageio.get_reader(str(reference), format="ffmpeg")
    right_reader = imageio.get_reader(str(candidate), format="ffmpeg")
    maximum = changed = frames = 0
    try:
        for left, right in zip_longest(left_reader, right_reader):
            if left is None or right is None or left.shape != right.shape:
                raise AssertionError("video frame counts/shapes differ")
            delta = np.abs(
                np.asarray(left, dtype=np.int16) - np.asarray(right, dtype=np.int16)
            )
            maximum = max(maximum, int(delta.max()))
            changed += int(np.count_nonzero(delta))
            frames += 1
    finally:
        left_reader.close()
        right_reader.close()
    return {
        "frames": frames,
        "max_channel_difference": maximum,
        "differing_channels": changed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument(
        "--label", default="current", help="Arm label for older single-policy traces"
    )
    args = parser.parse_args()
    raw = json.loads((args.directory / "summary.json").read_text())
    data = {key: value for key, value in raw.items() if key != "events"}
    for row in data["measurements"]:
        row.setdefault("arm", args.label)
        samples = [
            e
            for e in raw["events"]
            if e["run"] == row["run"] and e["phase"] == "render"
        ]
        after = next(e for e in reversed(samples) if e["event"] == "render_after")
        row["memory"] = {
            "min_host_available": min(e["host_available"] for e in samples),
            "peak_rss": max(e["rss"] for e in samples),
            "peak_private": max(
                (e["private"] for e in samples if e["private"] is not None),
                default=None,
            ),
            "peak_cuda_allocated": max(e["cuda_allocated"] for e in samples),
            "after_rss": after["rss"],
            "after_private": after["private"],
            "after_cuda_reserved": after["cuda_reserved"],
            "after_host_available": after["host_available"],
        }
    # Keep the exact cleanup decisions; collapse the 1 Hz samples into extrema.
    data["events"] = [e for e in raw["events"] if e["event"] != "sample"]
    data["all_runs_by_arm"] = {
        arm: {
            "renders": len(rows),
            "resets": sum(r["resets"] for r in rows),
            "median_render_seconds": statistics.median(
                r["render_seconds"] for r in rows
            ),
        }
        for arm in sorted({r.get("arm", "current") for r in data["measurements"]})
        if (rows := [r for r in data["measurements"] if r.get("arm", "current") == arm])
    }
    if args.compare:
        data["parity"] = {
            str(row["run"]): _compare_videos(
                args.directory / "run_0.mp4",
                args.directory / f"run_{row['run']}.mp4",
            )
            for row in data["measurements"][1:]
        }
        if any(p["max_channel_difference"] > 2 for p in data["parity"].values()):
            raise AssertionError(data["parity"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(
        json.dumps(
            {"arms": data["all_runs_by_arm"], "parity": data.get("parity")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
