"""Traverse-isolated perf benchmark for STBVH node-layout changes.

Renders the same animated flat-triangle scene as ``_node_pack_parity.py``
(PREVIEW video, ray-traced shadows) with ``wavefront_traverse`` and
``wavefront_shade`` monkeypatched behind ``torch.cuda.synchronize()`` timing
fences, and reports per-kernel totals (median over reps, after a warm-up
render that absorbs compile + cold GPU clocks). Run once before and once
after a layout change; the cross-process wall-time noise on Pascal is why the
kernels are sync-fenced individually rather than read from end-to-end time.

    .venv/Scripts/python.exe benchmarks/_node_pack_kp.py [reps]
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _node_pack_parity import build_and_animate  # noqa: E402

import algan.rendering.raytracing.tracer as tracer_mod  # noqa: E402
from algan import PREVIEW, SceneManager  # noqa: E402
from algan.rendering.raytracing import (  # noqa: E402
    set_fragment_shading,
    set_ray_traced_shadows,
)
from algan.utils.algan_utils import render_to_file  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3

_times = {}
_counts = {}


def _timed(name, orig):
    def wrapper(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = orig(*a, **k)
        torch.cuda.synchronize()
        _times[name] += time.perf_counter() - t0
        _counts[name] += 1
        return r

    return wrapper


_HOOKED = ("wavefront_traverse", "wavefront_shade")
for _name in _HOOKED:
    setattr(tracer_mod, _name, _timed(_name, getattr(tracer_mod, _name)))


def render_once():
    SceneManager.reset()
    set_fragment_shading(True)
    set_ray_traced_shadows(True)
    build_and_animate()
    for name in _HOOKED:
        _times[name] = 0.0
        _counts[name] = 0
    t0 = time.perf_counter()
    render_to_file(
        file_name="node_pack_kp", output_dir=OUT_DIR, render_settings=PREVIEW
    )
    total = time.perf_counter() - t0
    return {name: _times[name] for name in _HOOKED}, dict(_counts), total


def main():
    render_once()  # warm-up (compiles + GPU clocks)
    per_kernel = {name: [] for name in _HOOKED}
    totals = []
    counts = None
    for _ in range(REPS):
        times, counts, total = render_once()
        for name in _HOOKED:
            per_kernel[name].append(times[name])
        totals.append(total)
    for name in _HOOKED:
        med = statistics.median(per_kernel[name])
        print(
            f"{name:20s} median {med:7.2f} s over {counts[name]:4d} "
            f"launches   (runs: "
            f"{', '.join(f'{t:.2f}' for t in per_kernel[name])})",
            flush=True,
        )
    print(
        f"{'end-to-end':20s} median {statistics.median(totals):7.2f} s   "
        f"(runs: {', '.join(f'{t:.2f}' for t in totals)})"
    )


if __name__ == "__main__":
    main()
