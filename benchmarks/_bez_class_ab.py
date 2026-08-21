"""In-process alternating wall-time A/B: bezier cell classification + BVH
deferral (+ the always-on count->write survivor mask) on a text-heavy moving
scene at MD-like settings.

A = classification OFF, deferral OFF; B = both ON (defaults). Alternating
A/B/A/B in one process to defuse thermal-throttling drift.

Run: .venv/Scripts/python.exe benchmarks/_bez_class_ab.py
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")

from algan import (
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    OUT,
    RIGHT,
    UP,
    Off,
    RenderSettings,
    Sphere,
    Square,
    Sync,
    Text,
    render_to_file,
)
from algan.rendering.raytracing import bezier_acceleration as bez_accel
from algan.rendering.raytracing import settings as rt_settings

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "_tc_out"))
SETTINGS = RenderSettings((1280, 720), 15)


def build_scene():
    with Off():
        lines = []
        for i, s in enumerate(
            (
                "Gradient descent updates the weights",
                "backpropagation computes the gradient",
                "loss = sum((y - t)^2) / N",
            )
        ):
            t = Text(s).scale(0.55).move(UP * (1.8 - 1.2 * i))
            t.spawn()
            lines.append(t)
        sq = Square().scale(0.9).move(LEFT * 4 + DOWN * 2)
        sq.set_color(BLUE)
        sq.spawn()
        sp = Sphere().scale(0.5).move(RIGHT * 4 + DOWN * 2)
        sp.set_color(GREEN)
        sp.spawn()
    with Sync():
        for i, t in enumerate(lines):
            t.move(RIGHT * (0.4 + 0.2 * i))
        sq.rotate(120, OUT)
        sp.move(LEFT * 1.5)


def render_once(name, enabled):
    bez_accel.BEZIER_CLASS_ENABLED = enabled
    rt_settings.set_bvh_defer(enabled)
    build_scene()
    t0 = time.perf_counter()
    render_to_file(file_name=name, output_dir=OUT_DIR, render_settings=SETTINGS)
    return time.perf_counter() - t0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Warmup (kernel prep / compile) outside the measurement.
    render_once("ab_warm", True)
    times = {"off": [], "on": []}
    for rep in range(3):
        times["off"].append(render_once(f"ab_off_{rep}", False))
        times["on"].append(render_once(f"ab_on_{rep}", True))
    off = min(times["off"])
    on = min(times["on"])
    print(f"class+defer OFF : {[f'{t:.2f}' for t in times['off']]} best {off:.2f}s")
    print(f"class+defer ON  : {[f'{t:.2f}' for t in times['on']]} best {on:.2f}s")
    print(f"speedup {off / on:.3f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
