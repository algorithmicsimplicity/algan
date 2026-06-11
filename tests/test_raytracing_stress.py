"""Stress test for the ray traced pipeline: many overlapping primitives with
anti-aliasing over a few hundred frames. Memory use is a fixed per-frame
output buffer, so this mainly exercises batching, the merged scene cache and
post-processing under load.

Run directly: python tests/test_raytracing_stress.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import torch

from algan import (
    DOWN,
    LEFT,
    OUT,
    RIGHT,
    UP,
    Circle,
    Sphere,
    Square,
    Sync,
)
from algan.settings.render_settings import RenderSettings
from algan.utils.algan_utils import render_to_file

OUT_DIR = os.path.join(os.path.dirname(__file__), "raytracing_outputs")
SETTINGS = RenderSettings((640, 400), 10, anti_alias_level=2, fxaa=False)


def build_scene():
    mobs = []
    torch.manual_seed(3)
    for i in range(4):
        for j in range(3):
            loc = LEFT * 2.4 + RIGHT * 1.6 * i + DOWN * 1.4 + UP * 1.4 * j
            if (i + j) % 3 == 0:
                m = Sphere(radius=0.5)
            elif (i + j) % 3 == 1:
                m = Square()
            else:
                m = Circle()
            m.spawn()
            m.move(loc)
            mobs.append(m)
    with Sync():
        for k, m in enumerate(mobs):
            m.move(RIGHT * 0.7 if k % 2 == 0 else LEFT * 0.7)
    with Sync():
        for m in mobs:
            m.rotate(180, OUT)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from algan.rendering.raytracing import enable_ray_tracing

    enable_ray_tracing()
    build_scene()
    render_to_file(file_name="stress_raytraced", output_dir=OUT_DIR,
                   output_path="", render_settings=SETTINGS,
                   file_extension="mp4")
    path = os.path.join(OUT_DIR, "stress_raytraced.mp4")
    cap = cv2.VideoCapture(path)
    n, nonblack = 0, 0
    mid_frame = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.mean() > 1:
            nonblack += 1
        if n == 25:
            mid_frame = frame
        n += 1
    cap.release()
    print(f"frames={n}, non-black={nonblack}")
    if mid_frame is not None:
        cv2.imwrite(os.path.join(OUT_DIR, "stress_frame.png"), mid_frame)
    assert n > 0, "stress render produced no frames"
    assert nonblack > n // 2, "stress render produced mostly empty video"
    print("stress test passed")


if __name__ == "__main__":
    main()
