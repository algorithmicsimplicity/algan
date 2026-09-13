"""Render one soft-shadowed frame and save it, for the shadow-ray-budget A/B.

Two questions, one script:

* **Is the kill switch byte-identical?** ``ALGAN_SHADOW_RAY_BUDGET=0`` on the
  new code must reproduce the pre-budget renderer exactly. Render here with
  the budget off, render the same scene from a checkout that predates the
  budget, and ``--compare`` the two ``.npy`` frames: every channel must
  agree.
* **What does the budget do to the picture?** Render with the default budget
  and compare against the legacy frame: the difference should live in the
  penumbrae only, as dither rather than a shifted edge.

The scene is a ``RectAreaLight`` (16 rows at ``samples=4``) and a soft
``DirectionalLight`` over a ground slab with two blockers, plus a glossy
sphere so a secondary (bounce) hit's shadow is in the frame too::

    ALGAN_SHADOW_RAY_BUDGET=0 python benchmarks/_shadow_budget_check.py out/legacy
    python benchmarks/_shadow_budget_check.py out/budget16
    python benchmarks/_shadow_budget_check.py --compare out/legacy out/budget16

Each arm is its own process: the budget is read when the lights are packed,
so one process could flip it, but the legacy reference has to come from a
tree without the budget anyway.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import numpy as np  # noqa: E402


def render(out_prefix: str, resolution=(480, 270)):
    from algan import (
        BLUE,
        DOWN,
        GRAY_C,
        IN,
        LD,
        LEFT,
        ORIGIN,
        RIGHT,
        SETTINGS,
        UP,
        WHITE,
        Cube,
        DirectionalLight,
        MeshLambertMaterial,
        MeshStandardMaterial,
        Off,
        Prism,
        RectAreaLight,
        Scene,
        Sphere,
    )
    from algan.scene_manager import SceneManager

    settings = LD.set(resolution=resolution, frames_per_second=1, supersampling=1)
    SETTINGS.raytracing.set(shadows=True)
    SceneManager.reset()
    scene = Scene()
    scene.set_video_settings(settings)
    with Off():
        Scene.clear_lights()
        RectAreaLight(
            location=UP * 4.0 + IN * 1.0,
            target=ORIGIN,
            width=3.0,
            height=2.0,
            samples=4,
            color=WHITE,
            intensity=2.0,
        ).spawn(animate=False)
        DirectionalLight(
            location=RIGHT * 4 + UP * 5 + IN * 2,
            target=ORIGIN,
            color=WHITE,
            intensity=0.6,
            shadow_angle=3.0,
        ).spawn(animate=False)
        ground = Prism(width=10, height=0.2, depth=10, color=GRAY_C).set_material(
            MeshLambertMaterial()
        )
        ground.move(DOWN * 1.2).spawn(animate=False)
        Cube(size=0.9).set_material(MeshLambertMaterial(color=BLUE)).move(
            LEFT * 1.4 + DOWN * 0.55
        ).spawn(animate=False)
        Prism(width=0.3, height=1.6, depth=1.2).set_material(
            MeshLambertMaterial(color=WHITE)
        ).move(RIGHT * 0.6 + DOWN * 0.3).spawn(animate=False)
        Sphere(radius=0.7, color=GRAY_C).set_material(
            MeshStandardMaterial(metalness=0.9, roughness=0.12)
        ).move(RIGHT * 2.4 + DOWN * 0.4 + IN * 0.5).spawn(animate=False)
    result = Scene.save_frame(out_prefix + ".png", settings, overwrite=True)
    from PIL import Image

    frame = np.asarray(Image.open(out_prefix + ".png").convert("RGB"))
    np.save(out_prefix + ".npy", frame)
    print(f"wrote {out_prefix}.png / .npy {frame.shape}")
    return result


def compare(a: str, b: str):
    fa = np.load(a + ".npy").astype(np.int32)
    fb = np.load(b + ".npy").astype(np.int32)
    if fa.shape != fb.shape:
        print(f"SHAPE MISMATCH {fa.shape} vs {fb.shape}")
        return 2
    d = np.abs(fa - fb)
    n_diff = int((d > 0).any(-1).sum())
    print(
        f"pixels differing: {n_diff} of {d.shape[0] * d.shape[1]} "
        f"({100.0 * n_diff / (d.shape[0] * d.shape[1]):.2f}%); "
        f"max |d| = {int(d.max())}, mean |d| over differing = "
        f"{(d.sum() / max(1, 3 * n_diff)):.2f}"
    )
    if n_diff:
        hist = np.bincount(d.max(-1).ravel(), minlength=8)
        print("  |d| histogram (max over channels):", {i: int(c) for i, c in enumerate(hist) if c and i})
    return 0 if n_diff == 0 else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="output prefix, or two prefixes with --compare")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    args = parser.parse_args(argv)
    if args.compare:
        return compare(args.paths[0], args.paths[1])
    os.makedirs(os.path.dirname(os.path.abspath(args.paths[0])), exist_ok=True)
    render(args.paths[0], (args.width, args.height))
    return 0


if __name__ == "__main__":
    sys.exit(main())
