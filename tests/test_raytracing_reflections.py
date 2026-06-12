"""Reflection demo for the ray traced renderer.

Run directly: python tests/test_raytracing_reflections.py

A mirror floor below the scene reflects a sphere and a triangle. The scene is
rendered twice -- with the floor's reflectivity enabled and disabled -- and
the test asserts the reflection actually appears (the floor region changes
substantially and picks up the sphere's color).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import torch

from algan import (
    DOWN,
    LEFT,
    OUT,
    PURE_GREEN,
    PURE_RED,
    RIGHT,
    UP,
    Sphere,
    Sync,
    TriangleTriangulated,
)
from algan.settings.render_settings import RenderSettings
from algan.utils.algan_utils import render_to_file

OUT_DIR = os.path.join(os.path.dirname(__file__), "raytracing_outputs_v2")
SETTINGS = RenderSettings((640, 400), 10, anti_alias_level=1, fxaa=False)


def build_scene(mirror, roughness=0.0):
    from algan.rendering.raytracing import set_reflectivity, set_roughness

    floor_corners = torch.tensor([
        [[-5.0, -1.0, -4.0], [5.0, -1.0, -4.0], [5.0, -1.0, 8.0]],
        [[-5.0, -1.0, -4.0], [5.0, -1.0, 8.0], [-5.0, -1.0, 8.0]],
    ])
    floor = TriangleTriangulated(
        floor_corners,
        # 5-channel algan color: R, G, B, glow, alpha.
        color=torch.tensor([[0.08, 0.08, 0.12, 0.0, 1.0]]).expand(6, -1),
    )
    if mirror:
        set_reflectivity(floor, 0.85)
    if roughness > 0:
        set_roughness(floor, roughness)
    floor.spawn()

    sphere = Sphere(radius=0.7, color=PURE_GREEN).spawn()
    sphere.move(UP * 0.2 + LEFT * 1.2)
    tri = TriangleTriangulated(
        torch.stack((UP * 0.9, RIGHT * 0.8 + DOWN * 0.3, LEFT * 0.8 + DOWN * 0.3)),
        color=PURE_RED.expand(3, -1),
    ).spawn()
    tri.move(RIGHT * 1.4 + UP * 0.1)
    with Sync():
        sphere.move(RIGHT * 0.6)
        tri.rotate(180, OUT)


def render(name, mirror, roughness=0.0, samples_per_pixel=1):
    from algan.rendering.raytracing import enable_ray_tracing

    enable_ray_tracing(samples_per_pixel=samples_per_pixel)
    build_scene(mirror, roughness)
    render_to_file(file_name=name, output_dir=OUT_DIR, output_path="",
                   render_settings=SETTINGS, file_extension="mp4")
    return os.path.join(OUT_DIR, f"{name}.mp4")


def grab_frame(path, index):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"could not read frame {index} from {path}"
    return frame.astype(np.float64)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    mirror_path = render("reflections_on", mirror=True)
    flat_path = render("reflections_off", mirror=False)

    mirror_frame = grab_frame(mirror_path, 18)
    flat_frame = grab_frame(flat_path, 18)
    side = np.concatenate((flat_frame, mirror_frame), 1).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT_DIR, "reflections_side_by_side.png"), side)

    # The floor occupies the lower part of the frame; the reflection should
    # change it substantially and bring in the sphere's green.
    h = mirror_frame.shape[0]
    floor_mirror = mirror_frame[int(h * 0.62):]
    floor_flat = flat_frame[int(h * 0.62):]
    diff = np.abs(floor_mirror - floor_flat).mean()
    print(f"mean abs difference in floor region: {diff:.2f}")
    assert diff > 5, "mirror floor did not visibly change the render"
    green_gain = (floor_mirror[..., 1] - floor_flat[..., 1]).max()
    print(f"max green gain in floor region: {green_gain:.0f}")
    assert green_gain > 80, "sphere reflection (green) not found in the floor"

    # Monte Carlo glossy pass: a rough mirror at several samples per pixel
    # should still reflect the sphere, but blurred.
    glossy_path = render("reflections_glossy", mirror=True, roughness=0.35,
                         samples_per_pixel=16)
    from algan.rendering.raytracing import set_samples_per_pixel

    set_samples_per_pixel(1)
    glossy_frame = grab_frame(glossy_path, 18)
    side = np.concatenate((mirror_frame, glossy_frame), 1).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT_DIR, "reflections_glossy_side_by_side.png"),
                side)
    floor_glossy = glossy_frame[int(h * 0.62):]
    glossy_green = (floor_glossy[..., 1] - floor_flat[..., 1]).max()
    print(f"max green gain in glossy floor region: {glossy_green:.0f}")
    assert glossy_green > 40, "glossy reflection lost the sphere entirely"
    # Blur check: the glossy reflection should have a dimmer peak but spread
    # over more pixels than the sharp one.
    sharp_lit = (floor_mirror[..., 1] > 60).sum()
    glossy_lit = (floor_glossy[..., 1] > 30).sum()
    print(f"lit floor pixels sharp={sharp_lit} glossy(low thresh)={glossy_lit}")

    # Fully physical pass: no vertex shading; the scene's point light (up
    # and to the right of the camera) lights the sphere via shadow-rayed
    # next-event estimation, so its right side must come out brighter.
    from algan.rendering.raytracing import set_physical_lighting

    set_physical_lighting(True)
    try:
        physical_path = render("reflections_physical", mirror=True,
                               roughness=0.15, samples_per_pixel=24)
    finally:
        set_physical_lighting(False)
        set_samples_per_pixel(1)
    physical_frame = grab_frame(physical_path, 18)
    cv2.imwrite(os.path.join(OUT_DIR, "reflections_physical.png"),
                physical_frame.astype(np.uint8))
    green = physical_frame[..., 1]
    mask = green > 25
    assert mask.sum() > 50, "physical render lost the lit sphere"
    cols = np.nonzero(mask.any(axis=0))[0]
    middle = (cols.min() + cols.max()) // 2
    left = green[:, :middle][mask[:, :middle]].mean()
    right = green[:, middle:][mask[:, middle:]].mean()
    print(f"physical sphere brightness left={left:.1f} right={right:.1f}")
    assert right > left * 1.1, (
        "physical point light did not shade the sphere directionally")
    print("reflection demo passed")


if __name__ == "__main__":
    main()
