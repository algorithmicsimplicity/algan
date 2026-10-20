"""Render the Algan logo: the banner (README hero, docs social card) and the square icon.

Usage:  python branding/logo_scene.py BANNER_HD [outdir]
        python branding/logo_scene.py ICON_HD

Quality keys: PREVIEW, BANNER, BANNER_HD, BANNER_4K, ICON, ICON_HD, ICON_4K (NONE prints layout only).
The banner is the full lockup, the wordmark and the three solids; the icon is the symbol alone,
the lambda and its wave, which is what survives at avatar sizes. The first run in a fresh
process pays the Taichi kernel compile; later runs take seconds.
"""

from __future__ import annotations

import os
import sys

import brand
import torch

from algan import *  # noqa: F403

QUALITY = sys.argv[1] if len(sys.argv) > 1 else "PREVIEW"
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "algan_outputs/branding"
ICON = QUALITY.startswith("ICON")

quality = {
    "NONE": PREVIEW,
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
ASPECT = quality.resolution[0] / quality.resolution[1]
WORD_HEIGHT = 3.1
# Degrees the camera looks down: enough that the floor reads as a surface under the logo.
ELEVATION = 8.0

Scene.set_environment_map(brand.studio_env(), intensity=1.0, ambient=True)

with Off():
    brand.make_floor().spawn()
    word = brand.wordmark("Λ" if ICON else "Λlgan", WORD_HEIGHT)
    brand.float_word(word, 0.0)
    word.spawn()
    lam = word.character_mobs[0]
    bar = brand.Crossbar(lam).build().spawn()

    cam = Scene.get_camera()
    if ICON:
        lo, hi = word.get_bounding_box_min()[0, 0], word.get_bounding_box_max()[0, 0]
        look = torch.tensor([0.5 * (lo[0] + hi[0]).item(), brand.FLOOR_TOP + 1.25, 0.0])
        # Square: the lambda's height decides, with room for its reflection below.
        cam.move_to(
            brand.frame_camera(cam, look, 2.2 * (hi[1] - lo[1]).item(), 1.0, 10.0)
        )
    else:
        right = word.get_bounding_box_max()[0, 0][0].item()
        sphere, cube, cone = brand.make_solids(WORD_HEIGHT)
        for mob in brand.place_solids((sphere, cube, cone), right, WORD_HEIGHT):
            mob.spawn()
        left, right = brand.logo_extent(word, WORD_HEIGHT)
        look = torch.tensor([0.5 * (left + right), brand.FLOOR_TOP + 1.45, 0.0])
        cam.move_to(
            brand.frame_camera(cam, look, right - left, ASPECT, ELEVATION, margin=1.2)
        )
    cam.look_at(look)
    brand.make_lights()

    if QUALITY == "NONE":
        for name, m in [("word", word), ("lam", lam), ("bar", bar)] + (
            [] if ICON else [("sphere", sphere), ("cube", cube), ("cone", cone)]
        ):
            print(
                name,
                [round(v, 3) for v in m.get_bounding_box_min()[0, 0].tolist()],
                [round(v, 3) for v in m.get_bounding_box_max()[0, 0].tolist()],
            )
        print("camera", cam.get_center().flatten().tolist(), "look", look.tolist())
        sys.exit(0)

os.makedirs(OUTDIR, exist_ok=True)
name = ("algan-icon" if ICON else "algan-banner") + f"_{QUALITY}.png"
Scene.save_frame(os.path.join(OUTDIR, name), quality)
print("wrote", os.path.join(OUTDIR, name))
