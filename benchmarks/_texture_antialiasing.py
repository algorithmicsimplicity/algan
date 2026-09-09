"""Small primary/mirror UV-minification A/B through both production renderers.

Run from the repository root with its Python environment. Images and JSON are
written to algan_outputs/texture_antialiasing. Each route warms both arms, then
alternates five measured A/B renders. Cold compilation is not filter cost. No baseline files
or external assets are needed.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

if __name__ == "__main__":
    os.environ["ALGAN_AUTO_DAEMON"] = "0"
    os.environ["ALGAN_USE_DAEMON"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algan import (  # noqa: E402
    BLACK,
    SETTINGS,
    SMOKE_TEST,
    UP,
    WHITE,
    MeshBasicMaterial,
    MeshStandardMaterial,
    Off,
    Scene,
    SceneManager,
    Surface,
)


def checker(size=256):
    tex = torch.zeros((size, size, 5))
    tex[..., :3] = ((torch.arange(size)[:, None] + torch.arange(size)[None, :]) % 2)[
        ..., None
    ]
    tex[..., 4] = 1
    return tex


def build_scene(scene, reflection, texture=None, shift=0.0):
    scene.set_background(BLACK)
    Scene.clear_lights()
    tex = checker() if texture is None else texture
    if reflection:
        mirror = Surface(
            lambda uv: torch.stack(
                ((uv[..., 0] - 0.5) * 3, (uv[..., 1] - 0.5) * 3, uv[..., 0] * 0), -1
            ),
            grid_width=2,
            grid_height=2,
        )
        mirror.set_material(MeshStandardMaterial(color=WHITE, metalness=1, roughness=0))
        mirror.rotate(45, UP).spawn(animate=False)
        # Both side walls lie outside the camera view. Centre pixels can only
        # acquire the pattern by a reflected ray (also assert with black walls).
        for side in (-1, 1):
            wall = Surface(
                lambda uv, side=side: torch.stack(
                    (
                        uv[..., 0] * 0 + side * 5,
                        (uv[..., 1] - 0.5) * 8,
                        (uv[..., 0] - 0.5) * 8,
                    ),
                    -1,
                ),
                color_texture=tex,
                grid_width=2,
                grid_height=2,
            )
            wall.set_material(MeshBasicMaterial())
            wall.spawn(animate=False)
    else:
        wall = Surface(
            lambda uv: torch.stack(
                (
                    (uv[..., 0] - 0.5) * 5 + shift,
                    (uv[..., 1] - 0.5) * 5,
                    uv[..., 0] * 0,
                ),
                -1,
            ),
            color_texture=tex,
            grid_width=2,
            grid_height=2,
        )
        wall.set_material(MeshBasicMaterial())
        wall.spawn(animate=False)


def render_frame(
    path, spp, enabled, reflection=False, texture=None, shift=0.0, analytic=True
):
    snap = SETTINGS.snapshot()
    SceneManager.reset()
    try:
        SETTINGS.raytracing.set(
            samples_per_pixel=spp,
            denoise=False,
            texture_antialiasing=enabled,
            shadows=False,
            analytic_aa=analytic,
            max_bounces=3,
        )
        video = SMOKE_TEST.set(resolution=(64, 64), supersampling=1)
        with Scene(video_settings=video) as scene:
            with Off():
                build_scene(scene, reflection, texture, shift)
            start = time.perf_counter()
            result = scene.save_frame(path, video_settings=video, overwrite=True)
            elapsed = time.perf_counter() - start
            assert result.rendered
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[..., :3]
        return image, elapsed
    finally:
        SceneManager.reset()
        SETTINGS.restore(snap)


def main():
    out = Path("algan_outputs/texture_antialiasing").resolve()
    out.mkdir(parents=True, exist_ok=True)
    metrics = []
    for spp in (1, 4):
        for reflection in (False, True):
            timings = {False: [], True: []}
            images = {}
            for run in range(6):
                # Alternate order as well as arms to reduce thermal/order bias.
                for enabled in (False, True) if run % 2 == 0 else (True, False):
                    name = f"spp{spp}_{'mirror' if reflection else 'primary'}_{'mip' if enabled else 'bilinear'}"
                    image, elapsed = render_frame(
                        out / f"{name}.png", spp, enabled, reflection
                    )
                    images[enabled] = image
                    if run:
                        timings[enabled].append(elapsed)
            for enabled in (False, True):
                name = f"spp{spp}_{'mirror' if reflection else 'primary'}_{'mip' if enabled else 'bilinear'}"
                patch = images[enabled][26:38, 26:38].astype(np.float64)
                metrics.append(
                    {
                        "case": name,
                        "warm_seconds": statistics.median(timings[enabled]),
                        "warm_samples": timings[enabled],
                        "mean": float(patch.mean()),
                        "std": float(patch.std()),
                    }
                )
                print(json.dumps(metrics[-1]), flush=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
