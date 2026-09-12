"""Render the Algan logo: the banner (README hero, docs social card) and the square icon.

Usage:  python branding/logo_scene.py BANNER_HD [outdir]
        python branding/logo_scene.py ICON_HD

Quality keys: PREVIEW, BANNER, BANNER_HD, BANNER_4K, ICON, ICON_HD, ICON_4K (NONE prints layout only).
The first run in a fresh process pays the Taichi kernel compile; later runs take seconds.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

from algan import *  # noqa: F403

# The vendored Manim is registered by importing algan first, so this import stays below it.
# isort: off
import manim as mn  # noqa: E402

# isort: on

BACK_LON = (
    270.0  # env-map longitude that lies behind the camera (camera looks toward 90)
)
QUALITY = sys.argv[1] if len(sys.argv) > 1 else "PREVIEW"
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "algan_outputs/branding"
DIAG = ""
ICON = QUALITY.startswith("ICON")

# Franklin Gothic Heavy: the thickest installed face; the video uses the same one.
FONT = "Franklin Gothic"
WEIGHT = "HEAVY"
BANNER_CAM_X = -0.3
WAVE_BLUE = "#4da3ff"
CUBE_BLUE = Color("#2b52e6")
CONE_GOLD = Color("#f2c85a")
FLOOR_TOP = -1.5


def studio_env(back_lon, H=512, W=1024):
    lat = torch.linspace(90, -90, H).view(H, 1).expand(H, W)
    lon = torch.linspace(0, 360, W).view(1, W).expand(H, W)
    d = (lon - back_lon + 180) % 360 - 180
    env = torch.zeros(H, W, 3)
    # near-black sky with a faint grey zenith so the floor is not pure black
    env += ((lat.clamp(min=0) / 90) ** 2.2 * 0.10).unsqueeze(-1)
    # behind the camera: bright band high, falling to dark low -> chrome gradient on flat letters
    band = (
        torch.sigmoid((lat + 6) / 2.5)
        * torch.sigmoid((5.5 - lat) / 1.2)
        * torch.exp(-((d / 80) ** 2))
        * 1.05
    )
    env += band.unsqueeze(-1) * torch.tensor([0.92, 0.96, 1.0])
    # softbox up and behind-left of the camera: the sharp highlight on the solids
    box = ((d + 25).abs() < 22).float() * ((lat > 30) & (lat < 55)).float() * 3.0
    env += box.unsqueeze(-1)
    # below the horizon: dark
    env *= (0.15 + 0.85 * torch.sigmoid((lat + 6) / 3)).unsqueeze(-1)
    return env


Scene.set_environment_map(studio_env(BACK_LON), intensity=1.0, ambient=True)

with Off():
    floor = Prism(width=80, height=0.3, depth=60, color=Color("#07080b"))
    floor.set_material(
        MeshStandardMaterial(color=Color("#07080b"), metalness=0.38, roughness=0.0)
    )
    floor.move(UP * (FLOOR_TOP - 0.15)).spawn()

    word = Text(
        "\u039b" if ICON else "\u039blgan", font=FONT, weight=WEIGHT, font_size=200
    )
    word.set_material(
        MeshStandardMaterial(color=Color("#eef1f5"), metalness=1.0, roughness=0.08)
    )
    word.scale_to_height(3.2 if ICON else 3.1)
    word.spawn()
    wmin = word.get_bounding_box_min()[0, 0]
    wmax = word.get_bounding_box_max()[0, 0]
    # Banner: centre wordmark + solids (the cone's right edge is 1.05 + 3.15 + 0.9 past the
    # word) on the camera, whatever the face's width.
    left = -3.7 if ICON else BANNER_CAM_X - 0.5 * ((wmax[0] - wmin[0]).item() + 5.1)
    word.move(UP * (FLOOR_TOP + 0.05 - wmin[1]) + RIGHT * (left - wmin[0]))
    lam = word.character_mobs[0]
    lmin, lmax = lam.get_bounding_box_min()[0, 0], lam.get_bounding_box_max()[0, 0]
    lam_h = (lmax[1] - lmin[1]).item()
    lam_w = (lmax[0] - lmin[0]).item()
    bar_y = lmin[1].item() + 0.38 * lam_h
    bar_half = 0.5 * lam_w * 0.66
    cx = 0.5 * (lmin[0] + lmax[0]).item()

    ts = np.linspace(-bar_half, bar_half, 64)
    pts = np.stack(
        [ts, 0.13 * np.sin(np.pi * 1.5 * ts / bar_half), np.zeros_like(ts)], -1
    )
    vm = mn.VMobject(stroke_width=24, stroke_color=WAVE_BLUE).set_points_smoothly(pts)
    wave = ManimMob(vm)
    wave.color = Color(WAVE_BLUE, glow=0.5)
    wave.spawn()
    wave.move_to(torch.tensor([cx + 0.07 * lam_w, bar_y, 0.05]))

    wmax = word.get_bounding_box_max()[0, 0]
    if ICON:
        P = {"sphere": (-0.15, 1.3), "cube": (1.05, -1.7), "cone": (2.6, 0.1)}
        R = {"sphere": 0.85, "cube": 1.55, "cone": (0.8, 1.8)}
    else:
        x0 = wmax[0].item() + 1.05
        P = {"sphere": (x0, 1.6), "cube": (x0 + 1.9, -0.5), "cone": (x0 + 3.15, 1.1)}
        R = {"sphere": 1.0, "cube": 1.7, "cone": (0.9, 2.0)}
    sphere = Sphere(radius=R["sphere"], color=WHITE).set_material(
        MeshStandardMaterial(color=WHITE, metalness=1.0, roughness=0.04)
    )
    sphere.move(
        RIGHT * P["sphere"][0] + UP * (FLOOR_TOP + R["sphere"]) + OUT * P["sphere"][1]
    ).spawn()

    cube = Cube(size=R["cube"], fill_opacity=1.0).set_material(
        MeshStandardMaterial(color=CUBE_BLUE, metalness=0.1, roughness=0.4)
    )
    cube.rotate(38, UP).move(
        RIGHT * P["cube"][0] + UP * (FLOOR_TOP + R["cube"] / 2) + OUT * P["cube"][1]
    ).spawn()

    cone = Cone(radius=R["cone"][0], height=R["cone"][1], closed=True).set_material(
        MeshStandardMaterial(color=CONE_GOLD, metalness=0.15, roughness=0.35)
    )
    cone.move(
        RIGHT * P["cone"][0] + UP * (FLOOR_TOP + R["cone"][1] / 2) + OUT * P["cone"][1]
    ).spawn()

    RectAreaLight(
        location=UP * 12 + OUT * 4 + LEFT * 4,
        target=ORIGIN,
        width=6,
        height=6,
        samples=9,
        color=WHITE,
        intensity=30,
    ).spawn()
    PointLight(
        location=RIGHT * 10 + UP * 4 + OUT * 10, color=Color("#cfe0ff"), intensity=0.5
    ).spawn()

    cam = Scene.get_camera()
    if ICON:
        cam.move(UP * 1.6 + OUT * 2.0 + LEFT * 0.3).look_at(LEFT * 0.3 + DOWN * 0.85)
    else:
        cam.move(UP * 1.6 + LEFT * 0.3 + IN * 5.0).look_at(LEFT * 0.3 + DOWN * 0.35)

    if QUALITY == "NONE":
        for name, m in [
            ("word", word),
            ("lam", lam),
            ("wave", wave),
            ("sphere", sphere),
            ("cube", cube),
            ("cone", cone),
        ]:
            print(
                name,
                m.get_bounding_box_min()[0, 0].tolist(),
                m.get_bounding_box_max()[0, 0].tolist(),
                "spawned",
                m.is_spawned(),
            )
        print(
            "wave color",
            wave.color if hasattr(wave, "color") else None,
            type(wave).__name__,
        )
        print("wave children", len(wave.get_descendants()))
        sys.exit(0)

quality = {
    "PREVIEW": PREVIEW,
    "LD": LD,
    "HD": HD,
    "UHD": UHD,
    "BANNER": VideoSettings(resolution=(704, 264), supersampling=2),
    "BANNER_HD": VideoSettings(resolution=(1920, 720), supersampling=2),
    "BANNER_4K": VideoSettings(resolution=(3840, 1440), supersampling=2),
    "ICON": VideoSettings(resolution=(400, 400), supersampling=2),
    "ICON_HD": VideoSettings(resolution=(1024, 1024), supersampling=2),
    "ICON_4K": VideoSettings(resolution=(2048, 2048), supersampling=2),
}[QUALITY]

os.makedirs(OUTDIR, exist_ok=True)
name = ("algan-icon" if ICON else "algan-banner") + f"_{QUALITY}.png"
Scene.save_frame(os.path.join(OUTDIR, name), quality)
print("wrote", os.path.join(OUTDIR, name))
