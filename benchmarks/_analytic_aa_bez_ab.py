"""Analytic AA: speed AND quality against supersampling.

Scenes: ``text`` and ``shapes`` are bezier circuits (phase 1), ``meshes`` is
flat triangles (phase 2).

The point of analytic coverage is to retire the ``supersampling`` tax, so
the comparison that matters is not "toggle on vs off at the same resolution"
but:

    aa=2, supersampled      the current default
    aa=1, analytic          the proposed replacement
    aa=1, no AA             the floor (what you get by just dropping aa)
    aa=4, supersampled      the quality reference both are scored against

Arms alternate in one process (cross-process wall-clock swings ~2x with
thermal state), and each arm's frames are scored by mean L1 against the aa=4
reference.  Video encode is common to every arm and is included in the wall
time, so the reported speedup UNDERSTATES the render-side win.

Run: .venv/Scripts/python.exe benchmarks/_analytic_aa_bez_ab.py [scenes...]
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Benchmarks must never be measured inside a warm daemon: it keeps adaptive
# renderer state (the memory model's batch-size fit) across runs, so one
# benchmark would be timed against whatever ran before it.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

from algan import (  # noqa: E402
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    OUT,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Circle,
    Off,
    RenderSettings,
    SceneManager,
    Sphere,
    Square,
    Sync,
    Text,
    Torus,
    Triangle,
    render_to_file,
)
from algan.rendering.raytracing import (  # noqa: E402
    set_fragment_shading,
)
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.rendering.shaders.materials import (  # noqa: E402
    MeshStandardMaterial,
)
from algan.rendering.taichi_runtime import _sync_devices  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_rt2_out")
os.makedirs(OUT_DIR, exist_ok=True)

# Resolution decides what this measures. At MD a short circuit scene sits on
# the "tiny-scene render floor" -- CPU prep and video encode dominate, so even
# a free 4x cut in render work only moves the wall ~1.2x. Raise it (env
# override) to see the render-side win. AA_AB_RES=2560x1440 is a good probe.
RES = tuple(int(v) for v in os.environ.get("AA_AB_RES", "1280x720").split("x"))
FPS = int(os.environ.get("AA_AB_FPS", "15"))
REPS = int(os.environ.get("AA_AB_REPS", "3"))

# (tag, supersampling, analytic)
ARMS = [
    ("aa2_super", 2, False),
    ("aa1_analytic", 1, True),
    ("aa1_none", 1, False),
]
REF = ("aa4_ref", 4, False)


def build_scene(name):
    if name == "text":
        with Off():
            Text("Analytic anti-aliasing").scale(0.5).move(UP * 1.2).spawn()
            body = Text("coverage instead of supersampling").scale(0.3)
            body.move(DOWN * 0.2)
            body.spawn()
            Text("gjqy 0123456789").scale(0.28).move(DOWN * 1.2).spawn()
        body.move(RIGHT * 0.4)
        return
    if name == "meshes":
        # Flat-triangle meshes: dense shared-edge interiors and long moving
        # silhouettes, which is what phase 2 exists for. Triangle coverage rides
        # on the master toggle, so no extra setting here.
        with Off():
            a = Sphere().scale(1.3).move(LEFT * 1.5).set_color(BLUE)
            a.spawn()
            b = Sphere().scale(0.9).move(RIGHT * 1.3 + UP * 0.5)
            b.set_color(GREEN)
            b.spawn()
            t = Torus().scale(0.8).move(DOWN * 0.9).set_color(RED)
            t.spawn()
        with Sync():
            a.move(RIGHT * 0.7)
            t.rotate(50, RIGHT)
        return
    if name == "mirror":
        # What continuation-ray supersampling costs, on the content that pays
        # for it: a mirror ball whose every pixel spawns secondary rays, plus a
        # partially reflective floor-ish backdrop. The aa=2 arm supersamples
        # those secondary rays too (four primaries, four reflections), so this is
        # a like-for-like reflected-image comparison.
        with Off():
            ball = Sphere().scale(1.4)
            ball.set_material(MeshStandardMaterial(metalness=1.0, roughness=0.0))
            ball.spawn()
            a = Sphere().scale(0.5).move(LEFT * 2.5 + UP * 0.9).set_color(RED)
            a.spawn()
            b = Sphere().scale(0.45).move(RIGHT * 2.3 + DOWN * 0.7)
            b.set_color(YELLOW)
            b.spawn()
            t = Text("mirror").scale(0.4).move(UP * 1.7)
            t.spawn()
        with Sync():
            a.move(RIGHT * 0.6)
            ball.rotate(30, UP)
        return
    if name == "shapes":
        with Off():
            sq = Square(color=RED).scale(1.1).move(LEFT * 1.6)
            sq.spawn()
            ci = Circle(color=GREEN).scale(0.9).move(RIGHT * 1.4)
            ci.opacity = 0.6
            ci.spawn()
            tr = Triangle(color=YELLOW).scale(0.9).move(DOWN * 1.0)
            tr.spawn()
            ring = Circle(color=BLUE, border_width=3).scale(0.7)
            ring.move(UP * 1.2)
            ring.spawn()
        with Sync():
            sq.rotate(40, OUT)
            ci.move(LEFT * 0.9)
            tr.rotate(-30, OUT)
        return
    raise SystemExit(f"unknown scene {name}")


def render(name, tag, aa, analytic):
    SceneManager.reset()
    rt_settings.set_analytic_aa(analytic)
    # The mirror scene needs per-fragment shading for its metal to reflect.
    set_fragment_shading(True)
    build_scene(name)
    fname = f"aaAB_{name}_{tag}"
    _sync_devices()
    t0 = time.perf_counter()
    render_to_file(
        file_name=fname,
        output_dir=OUT_DIR,
        output_path="",
        render_settings=RenderSettings(RES, FPS, supersampling=aa),
        file_extension="mp4",
    )
    _sync_devices()
    dt = time.perf_counter() - t0
    rt_settings.set_analytic_aa(False)
    return os.path.join(OUT_DIR, fname + ".mp4"), dt


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f.astype(np.float64))
    cap.release()
    return frames


def main():
    scenes = sys.argv[1:] or ["text", "shapes", "meshes"]
    for name in scenes:
        print(f"=== {name} ({RES[0]}x{RES[1]}) ===", flush=True)
        ref_path, _ = render(name, *REF)
        ref = read_frames(ref_path)

        times = {tag: [] for tag, _, _ in ARMS}
        paths = {}
        # Warm-up pass (kernel instantiation, caches) then timed repetitions,
        # alternating so thermal drift hits every arm equally.
        for rep in range(REPS + 1):
            for tag, aa, analytic in ARMS:
                path, dt = render(name, tag, aa, analytic)
                paths[tag] = path
                if rep:
                    times[tag].append(dt)

        base = float(np.median(times[ARMS[0][0]]))
        for tag, aa, _ in ARMS:
            med = float(np.median(times[tag]))
            frames = read_frames(paths[tag])
            if len(frames) != len(ref):
                err = float("nan")
            else:
                err = float(
                    np.mean([np.abs(a - r).mean() for a, r in zip(frames, ref)])
                )
            print(
                f"  {tag:14s} aa={aa}  {med:6.2f}s "
                f"({base / max(med, 1e-9):4.2f}x vs aa2)   "
                f"L1 vs aa4 reference {err:6.3f}",
                flush=True,
            )
        print(flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
