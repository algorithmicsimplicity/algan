"""Compare decoded benchmark frames without retaining an entire UHD video."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    sentinel = object()
    streams = (iio.get_reader(str(p), format="FFMPEG") for p in (args.control, args.candidate))
    for index, (a, b) in enumerate(itertools.zip_longest(*streams, fillvalue=sentinel)):
        if a is sentinel or b is sentinel:
            raise ValueError("videos have different frame counts")
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            raise ValueError("videos have different frame dimensions")
        delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
        rows.append({
            "frame": index,
            "max_channel_delta": int(delta.max()),
            "mean_channel_delta": float(delta.mean()),
            "changed_pixels": int(np.any(delta != 0, axis=2).sum()),
            "pixels_over_2": int(np.any(delta > 2, axis=2).sum()),
        })
        if index == 0 or rows[-1]["pixels_over_2"]:
            Image.fromarray(a).save(args.out / f"frame_{index}_control.png")
            Image.fromarray(b).save(args.out / f"frame_{index}_candidate.png")
            Image.fromarray(np.minimum(delta * 16, 255).astype(np.uint8)).save(
                args.out / f"frame_{index}_diff_x16.png"
            )
    if not rows:
        raise ValueError("videos contain no frames")
    result = {"control": str(args.control), "candidate": str(args.candidate), "frames": rows}
    (args.out / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
