"""A/B of the denoiser's precision: half against float32, same process.

``denoise_precision`` is a runtime setting (the network is re-cast when it
changes; no ``ti.static`` gate is involved), so both arms run in ONE process
on ONE scene and the comparison is on the raw frames ``Scene.get_frames``
returns -- byte counts, not a lossy video.

Reports, per arm, the denoiser's wall time per frame (device-synchronised, so
on CUDA it is the filter and nothing else) and, between the arms, the largest
and mean absolute pixel difference in 8-bit counts. What the numbers mean:

* the time ratio is the reason the switch exists -- on a T4 the fp32 filter
  was the largest device-side item of a path-traced frame
  (``benchmarks/performance/reports/t4_2026_09/pt_baseline_1.md``);
* the pixel difference is the cost: half-float rounding through sixteen
  convolutions. A max of a few counts on an 8-bit frame is the expected
  order; a large or structured difference is a bug in the cast, not
  rounding.

Usage::

    uv run python benchmarks/_denoise_precision_check.py
    uv run python benchmarks/_denoise_precision_check.py --resolution 1280x720 --spp 8

On a CPU box the fp16 arm is forced (``"fp16"`` rather than ``"auto"``, which
would resolve to fp32 there) and is SLOW -- that arm exists to prove the
plumbing, not to be timed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ["ALGAN_USE_DAEMON"] = "0"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

FRAMES = 2


def _parse_resolution(text):
    w, h = (int(p) for p in text.lower().split("x"))
    return w, h


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=_parse_resolution, default=(640, 360))
    parser.add_argument("--spp", type=int, default=8)
    args = parser.parse_args(argv)

    import numpy as np
    import torch

    import algan as A
    import benchmarks._pt_shadow_anyhit_check as scene_mod
    from algan import SETTINGS
    from algan.rendering.denoise import denoise as denoise_mod
    from algan.scene_manager import SceneManager
    from algan.settings._startup import render_device

    device = render_device()
    timings = {"fp32": [], "fp16": []}
    current = {"arm": None}
    original_call = denoise_mod.Denoiser.__call__

    def timed_call(self, color, albedo, normal):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = original_call(self, color, albedo, normal)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings[current["arm"]].append(
            (time.perf_counter() - t0) / max(1, int(color.shape[0]))
        )
        return out

    denoise_mod.Denoiser.__call__ = timed_call

    SceneManager.reset()
    SETTINGS.raytracing.set(samples_per_pixel=args.spp, shadows=True, denoise=True)
    q = A.SMOKE_TEST.set(resolution=args.resolution)
    SceneManager.instance().current_scene.set_video_settings(q)
    scene_mod.build_scene()
    scene = SceneManager.instance().current_scene

    frames = {}
    for arm, choice in (("fp32", "fp32"), ("fp16", "fp16")):
        current["arm"] = arm
        SETTINGS.raytracing.experimental.set(denoise_precision=choice)
        dtype, channels_last = denoise_mod.resolve_precision(device)
        print(f"{arm}: device={device} dtype={dtype} channels_last={channels_last}")
        # Two passes: the first pays every warm-up (kernel compile, weights,
        # cuDNN autotune); the second is the measurement and the comparison.
        for _ in range(2):
            timings[arm].clear()
            frames[arm] = torch.cat(
                [f.cpu() for f in scene.get_frames(0, FRAMES)]
            ).numpy()
    SETTINGS.raytracing.experimental.set(denoise_precision="auto")

    a = frames["fp32"].astype(np.int32)
    b = frames["fp16"].astype(np.int32)
    diff = np.abs(a - b)
    per_frame_ms = {
        arm: 1000.0 * float(np.mean(t)) if t else float("nan")
        for arm, t in timings.items()
    }
    ratio = per_frame_ms["fp32"] / per_frame_ms["fp16"] if per_frame_ms["fp16"] else 0
    summary = {
        "device": str(device),
        "resolution": list(args.resolution),
        "spp": args.spp,
        "frames": FRAMES,
        "denoise_ms_per_frame": per_frame_ms,
        "speedup_fp32_over_fp16": round(ratio, 2),
        "pixel_diff_max": int(diff.max()),
        "pixel_diff_mean": round(float(diff.mean()), 4),
        "pixels_over_2": int((diff > 2).sum()),
        "pixels_total": int(diff.size),
    }
    print(
        f"denoise per frame: fp32 {per_frame_ms['fp32']:.1f} ms, "
        f"fp16 {per_frame_ms['fp16']:.1f} ms  ({ratio:.2f}x)"
    )
    print(
        f"fp16 vs fp32 pixels: max |diff| {summary['pixel_diff_max']} counts, "
        f"mean {summary['pixel_diff_mean']}, "
        f"{summary['pixels_over_2']} of {summary['pixels_total']} channel "
        "samples differ by more than 2"
    )
    print("RESULTS " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
