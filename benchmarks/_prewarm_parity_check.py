"""Parity check for worker-side render-batch prewarm (ALGAN_PREFETCH_MERGE):
projection (vertex shade + pack) and merge/STBVH build moved onto the
batch-prep worker must be byte-identical to the classic render-thread path.

Renders the same short animated video twice -- prewarm on / off -- with the
animate-memory budget shrunk so the render loop splits into several batches
(exercising the worker-thread prewarm + the render thread's projection skip),
then decodes both mp4s and compares every frame pixel-exactly.

    .venv/Scripts/python.exe benchmarks/_prewarm_parity_check.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import algan.render_loop as rl  # noqa: E402
from algan import (  # noqa: E402
    BLUE,
    GREEN,
    IN,
    LD,
    LEFT,
    ORANGE,
    PURPLE,
    RED,
    RIGHT,
    TEAL,
    UP,
    WHITE,
    YELLOW,
    Off,
    SceneManager,
    Sphere,
    Square,
    Sync,
    render_to_file,
)
from algan.settings.defaults import COMPUTING_DEFAULTS  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)

_COLORS = [BLUE, RED, GREEN, YELLOW, WHITE, ORANGE, PURPLE, TEAL]

# Count batches + worker prewarms so the check can prove it exercised the
# multi-batch worker path rather than a single-batch fallback.
_counts = {"batches": 0, "prewarmed": 0}
_orig_prewarm = rl.RenderLoopMixin._prewarm_render_batch


def _counting_prewarm(self, primitives, render_state):
    _counts["batches"] += 1
    r = _orig_prewarm(self, primitives, render_state)
    if primitives and getattr(primitives[0], "_rt_projected", False):
        _counts["prewarmed"] += 1
    return r


rl.RenderLoopMixin._prewarm_render_batch = _counting_prewarm


def build():
    rng = np.random.default_rng(20260713)
    mobs = []
    with Off():
        for i in range(8):
            x = float(rng.uniform(-3.0, 3.0))
            y = float(rng.uniform(-1.8, 1.8))
            z = float(rng.uniform(-1.5, 1.5))
            col = _COLORS[i % len(_COLORS)].set_opacity(0.6)
            if i % 3 == 0:
                m = Sphere(grid_height=10, grid_width=10).scale(0.7)
            else:
                m = Square(color=col).scale(1.0)
            m.move(RIGHT * x + UP * y + IN * z)
            m.spawn()
            mobs.append(m)
    with Sync(runtime=1.0):
        for m in mobs:
            m.move(LEFT * 1.2)


def render(tag, prewarm):
    os.environ["ALGAN_PREFETCH_MERGE"] = "1" if prewarm else "0"
    SceneManager.reset()
    build()
    _counts["batches"] = _counts["prewarmed"] = 0
    render_to_file(
        file_name=f"prewarm_{tag}",
        output_dir=OUT_DIR,
        output_path="",
        render_settings=LD,
        file_extension="mp4",
    )
    print(
        f"  {tag}: {_counts['batches']} batches, {_counts['prewarmed']} prewarmed",
        flush=True,
    )
    return os.path.join(OUT_DIR, f"prewarm_{tag}.mp4")


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f.astype(np.int32))
    cap.release()
    return frames


def main():
    # Shrink the animate budget so the video splits into several batches --
    # the prewarm then runs on the prefetch worker for every batch after the
    # first, which is the path under test.
    COMPUTING_DEFAULTS.portion_of_memory_used_for_animating *= 1e-4

    p_off = render("off", False)
    p_on = render("on", True)
    a, b = read_frames(p_off), read_frames(p_on)
    if len(a) != len(b):
        print(f"FAIL: frame count differs ({len(a)} vs {len(b)})")
        return
    worst = 0
    bad = 0
    for _i, (fa, fb) in enumerate(zip(a, b)):
        d = int(np.abs(fa - fb).max()) if fa.shape == fb.shape else 999
        worst = max(worst, d)
        bad += d > 0
    print(f"{len(a)} frames compared, {bad} differ, max |d| = {worst}")
    print(f"prewarm parity: {'PASS' if worst == 0 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
