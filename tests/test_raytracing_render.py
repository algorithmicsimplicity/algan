"""End-to-end comparison of the rasterized and ray traced render pipelines.

Run directly: python tests/test_raytracing_render.py

Renders the same animated scene through both pipelines, reports per-frame
PSNR between the two videos, and writes a side-by-side comparison image.
The two renderers are expected to differ slightly (screen-space vs exact
depth/interpolation, polyline sampling), so this checks for gross agreement
rather than pixel equality. Coplanar overlaps are deliberately avoided by
small z offsets: depth ties are inherently ambiguous and the two pipelines
break them differently.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from algan import (
    DOWN,
    IN,
    LEFT,
    ORIGIN,
    OUT,
    PURE_BLUE,
    PURE_GREEN,
    PURE_RED,
    RIGHT,
    UP,
    ImageMob,
    Sphere,
    Square,
    Sync,
    Text,
    TriangleTriangulated,
)
from algan.settings.render_settings import RenderSettings
from algan.utils.algan_utils import render_to_file

OUT_DIR = os.path.join(os.path.dirname(__file__), "raytracing_outputs_v2")
SETTINGS = RenderSettings((640, 400), 10, anti_alias_level=1, fxaa=False)


def build_and_animate_scene():
    tri = TriangleTriangulated(
        torch.stack(
            (
                UP * 0.7,
                F.normalize(RIGHT + DOWN, p=2, dim=-1) * 0.7,
                F.normalize(LEFT + DOWN, p=2, dim=-1) * 0.7,
            )
        ),
        color=torch.stack([PURE_RED, PURE_BLUE, PURE_GREEN]),
    ).spawn()
    sphere = Sphere(radius=0.5).spawn()
    sphere.move(LEFT * 2)
    square = Square().spawn()
    square.move(RIGHT * 2 + IN * 0.05)  # offset: avoid an ambiguous z tie
    text = Text("STBVH").spawn()
    text.move(UP * 1.4 + OUT * 0.05)
    gradient = (torch.stack(torch.meshgrid(
        torch.linspace(0, 1, 32), torch.linspace(0, 1, 32),
        indexing="ij"), -1))
    image = torch.cat((gradient, torch.ones(32, 32, 2)), -1)
    image_mob = ImageMob(image).spawn()
    image_mob.scale(0.6)
    image_mob.move(DOWN * 1.4 + RIGHT * 1.5 + IN * 0.1)
    with Sync():
        tri.move(RIGHT * 1.2)
        square.move(DOWN * 0.8)
    tri.rotate(360, OUT)
    # Animated camera: per-frame ray origins/bases must be honored.
    tri.scene.camera.rotate(30, UP, about_point=ORIGIN)


def render(name, ray_traced):
    if ray_traced:
        from algan.rendering.raytracing import enable_ray_tracing

        enable_ray_tracing()
    else:
        try:
            from algan.rendering.raytracing import disable_ray_tracing

            disable_ray_tracing()
        except ImportError:
            pass
    build_and_animate_scene()
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


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 10 * np.log10(255.0 ** 2 / mse)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raster_path = render("raster", ray_traced=False)
    rt_path = render("raytraced", ray_traced=True)

    raster = read_frames(raster_path)
    rt = read_frames(rt_path)
    n = min(len(raster), len(rt))
    assert n > 0, "no frames decoded"
    print(f"frames: raster={len(raster)} raytraced={len(rt)}")
    scores = [psnr(raster[i], rt[i]) for i in range(n)]
    print("per-frame PSNR (dB):", " ".join(f"{s:.1f}" for s in scores))
    mean_psnr = float(np.mean(scores))
    print(f"mean PSNR: {mean_psnr:.2f} dB  min: {min(scores):.2f} dB")

    mid = n // 2
    side = np.concatenate((raster[mid], rt[mid]), axis=1).astype(np.uint8)
    cmp_path = os.path.join(OUT_DIR, "side_by_side.png")
    cv2.imwrite(cmp_path, side)
    diff = np.abs(raster[mid] - rt[mid]).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT_DIR, "diff.png"), diff)
    print(f"wrote {cmp_path}")

    assert mean_psnr > 25, f"renders disagree badly (mean PSNR {mean_psnr:.2f} dB)"
    print("end-to-end raytracing comparison passed")


if __name__ == "__main__":
    main()
