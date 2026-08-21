"""Parity + timing harness for the animation/prep optimizations, using the
bezier_rendering benchmark scene.

Modes:
    golden [settings]   render the scene and save as the golden reference
    check  [settings]   render the scene and pixel-compare against the golden

settings: preview (default) or hd

    .venv/Scripts/python.exe benchmarks/_prep_opt_check.py golden preview
    .venv/Scripts/python.exe benchmarks/_prep_opt_check.py check preview
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import manim as mn  # noqa: E402

from algan import *  # noqa: E402
from algan.mobs.neural_nets.neural_net import NeuralNetMLP  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402
from algan.utils.algan_utils import render_to_file  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)


def Boxed(mob, color=BLUE, buffer=0.1, *args, **kwargs):
    return Group(
        mob,
        SurroundingRectangle(
            mob,
            *args,
            color=color.lerp(BLACK, 0.8).lerp(PURE_BLUE, 0.1).set_opacity(0.95),
            border_color=torch.lerp(color, BLACK, 0.2),
            buffer=buffer,
            border_width=1,
            **kwargs,
        ),
    )


def GlowTex(c, *args, **kwargs):
    m = (
        ManimMob(mn.MathTex(*args, **kwargs))
        .set(
            color=c + GLOW * 0.01,
            border_color=torch.lerp(c, WHITE, 0.9),
            border_width=0.8,
        )
        .scale(0.75)
    )
    return m


text_string = ("a" * 50 + "\n") * 50


def text_scene():
    with Sync(run_time=0.25):
        nn = NeuralNetMLP([3, 3, 3]).spawn()
        mob = Boxed(GlowTex(GREEN, text_string)).spawn()
    with Sync(run_time=0.25):
        mob.move(LEFT)
        nn.move(LEFT)


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    setting_name = sys.argv[2] if len(sys.argv) > 2 else "preview"
    settings = {"preview": PREVIEW, "hd": HD}[setting_name]
    golden_path = os.path.join(OUT_DIR, f"golden_{setting_name}.mp4")

    scene = SceneManager.reset()
    scene.set_render_settings(settings)
    t0 = time.perf_counter()
    text_scene()
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    name = f"parity_{setting_name}_{mode}"
    render_to_file(
        file_name=name,
        output_dir=OUT_DIR,
        output_path="",
        render_settings=settings,
        file_extension="mp4",
    )
    t_render = time.perf_counter() - t0
    out_path = os.path.join(OUT_DIR, f"{name}.mp4")
    print(f"[timing] scene build: {t_build:.2f}s   render_to_file: {t_render:.2f}s")

    if mode == "golden":
        if os.path.exists(golden_path):
            os.remove(golden_path)
        os.replace(out_path, golden_path)
        print(f"golden saved to {golden_path}")
        return

    a = read_frames(golden_path)
    b = read_frames(out_path)
    if len(a) != len(b):
        print(f"FRAME COUNT MISMATCH: golden {len(a)} vs new {len(b)}")
        sys.exit(1)
    max_diff = 0
    n_over2 = 0
    n_diff = 0
    for _i, (fa, fb) in enumerate(zip(a, b)):
        d = np.abs(fa.astype(np.int16) - fb.astype(np.int16))
        max_diff = max(max_diff, int(d.max()))
        n_diff += int((d > 0).sum())
        n_over2 += int((d > 2).sum())
    total = sum(f.size for f in a)
    print(
        f"frames: {len(a)}  max abs diff: {max_diff}  "
        f"nonzero: {n_diff}/{total}  >2: {n_over2}"
    )
    if max_diff == 0:
        print("BYTE-IDENTICAL (decoded frames)")
    elif n_over2 == 0:
        print("within tolerance (<=2), NOT byte-identical")
    else:
        print("PARITY FAILURE")
        sys.exit(1)


if __name__ == "__main__":
    main()
