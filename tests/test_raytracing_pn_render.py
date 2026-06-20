"""End-to-end smoke test for ray traced PN (curved point-normal) triangles.

Run directly: python tests/test_raytracing_pn_render.py

Renders a coarsely tessellated sphere next to a flat square through the full
scene pipeline twice -- once with flat ray traced triangles, once with
``enable_ray_tracing(pn_triangles=True)`` -- at low resolution. Checks that
both videos render, that the flat-geometry half of the frame is unaffected
by the PN mode (flat triangles and bezier circuits must be byte-stable),
and that the coarse sphere actually changes (curved silhouette/interior),
without the two renders diverging grossly.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from algan import IN, LEFT, RIGHT, Sphere, Square
from algan.settings.render_settings import RenderSettings
from algan.utils.algan_utils import render_to_file

OUT_DIR = os.path.join(os.path.dirname(__file__), "raytracing_outputs_pn")
SETTINGS = RenderSettings((320, 200), 10, anti_alias_level=1, fxaa=False)


def build_scene():
    sphere = Sphere(radius=1.0, grid_height=8).spawn()
    sphere.move(LEFT * 1.5)
    square = Square().spawn()
    square.move(RIGHT * 1.8 + IN * 0.05)
    sphere.move(RIGHT * 0.5)  # a short animation, so several frames render


def render(name, pn_triangles):
    from algan.rendering.raytracing import enable_ray_tracing

    enable_ray_tracing(pn_triangles=pn_triangles)
    build_scene()
    render_to_file(file_name=name, output_dir=OUT_DIR, output_path="",
                   render_settings=SETTINGS, file_extension="mp4")
    return os.path.join(OUT_DIR, f"{name}.mp4")


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame.astype(np.float64))
    cap.release()
    return frames


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    flat_path = render("pn_smoke_flat", pn_triangles=False)
    pn_path = render("pn_smoke_curved", pn_triangles=True)
    flat = read_frames(flat_path)
    pn = read_frames(pn_path)
    assert len(flat) >= 5 and len(pn) >= 5, (
        f"renders too short: flat={len(flat)} pn={len(pn)} frames")
    assert len(flat) == len(pn), "frame count differs between modes"

    width = flat[0].shape[1]
    sphere_diff = 0.0
    square_diff = 0.0
    for a, b in zip(flat, pn):
        d = np.abs(a - b)
        sphere_diff = max(sphere_diff, d[:, : width // 2].mean())
        square_diff = max(square_diff, d[:, width // 2:].mean())
    # The flat square (and anything bezier) must not change; mp4 encoding
    # noise stays well below this once the pixels are identical.
    assert square_diff < 1.0, (
        f"flat geometry changed under PN mode (mean diff {square_diff:.2f})")
    # The coarse sphere must change (curved silhouette), but not vanish.
    assert sphere_diff > 0.5, (
        f"PN mode left the coarse sphere identical (mean diff "
        f"{sphere_diff:.3f}) -- are patches actually curved?")
    assert sphere_diff < 40.0, (
        f"PN sphere diverged grossly from the flat render (mean diff "
        f"{sphere_diff:.1f})")
    print(f"ok: PN end-to-end render (sphere half mean diff up to "
          f"{sphere_diff:.2f}, flat half {square_diff:.3f}; "
          f"{len(pn)} frames at {flat[0].shape[1]}x{flat[0].shape[0]})")


if __name__ == "__main__":
    main()
