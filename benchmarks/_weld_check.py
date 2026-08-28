"""Qualification runs for surface seam/pole welding (``ALGAN_WELD_SURFACE_SEAMS``).

``DESIGN_mesh_identity.md`` ss3.1 welds a closed surface's wraparound column
against column 0 and collapses each degenerate pole fan to a single vertex. It
measured that an untextured Sphere/Cylinder/Torus/Cone scene is **byte-identical**
across the gate (the welded vertices were coincident to 1.7e-07 and the dropped
triangles had zero area), and left the gate off for one stated reason:

    "The remaining risk is a texture-mapped or normal-mapped closed surface."

That risk is specific and it is a shape hazard, not a numerical one. The POLE weld
changes the triangle list, so every per-vertex attribute -- uv included -- has to
go through the same indices. The U-SEAM wrap deliberately does **not** weld:
wrapping it would give the last cell column ``u = 0`` where the texture needs
``u = 1``, running the map backwards across that column. The duplicate uv column
exists precisely to carry that discontinuity, so a weld that "tidied" it would
mirror the last column of every texture.

This script renders that case. Three shapes:

* ``plain``    -- the untextured control ss3.1 already measured. Must stay
  byte-identical; if it moves, something other than uv is wrong.
* ``checker``  -- a Sphere carrying a high-frequency CHECKERBOARD colour texture.
  A checkerboard is the instrument on purpose: a smooth photo hides a one-column
  uv error, while a checker turns it into a visible seam or a mirrored column.
* ``normals``  -- a Sphere carrying a normal texture, so the lit shading depends
  on a per-texel attribute rather than on colour alone.

``weld_surface_seams`` is a live runtime setting (it is read per primitive build
and the triangle-index cache keys on it), so both arms run in ONE process and
thermal drift is irrelevant. Triangle counts are read out of the scene merge as
engagement proof -- "byte-identical" means nothing if the weld never ran.

Usage:
    .venv/Scripts/python.exe benchmarks/_weld_check.py [quality]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from algan import *  # noqa: F403,E402
from algan.rendering.raytracing import raster_pipeline as rp  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402

OUT_DIR = os.path.join("algan_outputs", "weld_check")
PINNED_BYTES = 1_400_000_000


def checker_texture(width=64, height=32, squares=8):
    """A ``[W, H, 5]`` opaque checkerboard.

    High frequency and axis-aligned in uv, so a one-column uv error shows up as
    a mismatched column rather than as a slightly softer gradient.
    """
    texture = torch.zeros(width, height, 5)
    us = (torch.arange(width) * squares // width) % 2
    vs = (torch.arange(height) * squares // height) % 2
    check = (us[:, None] ^ vs[None, :]).float()
    texture[..., 0] = check
    texture[..., 1] = 1.0 - check
    texture[..., 2] = check * 0.5
    texture[..., 4] = 1.0
    return texture


def normal_texture(width=64, height=32, bumps=6):
    """A ``[W, H, 3]`` tangent-space normal map of smooth diagonal bumps."""
    us = torch.linspace(0, 2 * np.pi * bumps, width)
    vs = torch.linspace(0, 2 * np.pi * bumps, height)
    nx = torch.sin(us)[:, None].expand(width, height) * 0.6
    ny = torch.sin(vs)[None, :].expand(width, height) * 0.6
    nz = torch.sqrt((1.0 - nx**2 - ny**2).clamp_min(1e-4))
    return torch.stack([nx, ny, nz], dim=-1)


def scene_plain():
    """ss3.1's own control: every closed family, no textures."""
    Scene.set_background(DARKER_GRAY)
    with Off():
        AmbientLight(color=WHITE, intensity=0.3).spawn(animate=False)
        PointLight(location=LEFT * 4 + UP * 3 + OUT * 4).spawn(animate=False)
        Sphere(radius=0.8, resolution=(48, 24), color=YELLOW).move(LEFT * 2.4).spawn(
            animate=False
        )
        Cylinder(radius=0.5, height=1.6, color=RED).move(LEFT * 0.8).spawn(
            animate=False
        )
        Cone(radius=0.55, height=1.5, color=GREEN).move(RIGHT * 0.8).spawn(
            animate=False
        )
        Torus(major_radius=0.7, minor_radius=0.22, color=BLUE).move(RIGHT * 2.4).spawn(
            animate=False
        )


def scene_checker():
    Scene.set_background(DARKER_GRAY)
    with Off():
        AmbientLight(color=WHITE, intensity=0.6).spawn(animate=False)
        PointLight(location=LEFT * 4 + UP * 3 + OUT * 4).spawn(animate=False)
        # Big and centred, so the pole fans and the u-seam are both on screen at
        # a readable scale.
        Sphere(radius=1.7, resolution=(64, 32), color_texture=checker_texture()).spawn(
            animate=False
        )


def scene_normals():
    Scene.set_background(DARKER_GRAY)
    with Off():
        AmbientLight(color=WHITE, intensity=0.3).spawn(animate=False)
        PointLight(location=LEFT * 4 + UP * 3 + OUT * 4).spawn(animate=False)
        Sphere(
            radius=1.7,
            resolution=(64, 32),
            color=WHITE,
            normal_texture=normal_texture(),
        ).spawn(animate=False)


SHAPES = {"plain": scene_plain, "checker": scene_checker, "normals": scene_normals}


def render_once(name, build, weld, quality):
    """Render one arm and report (image, triangle count reaching the resolve)."""
    path = os.path.join(OUT_DIR, f"weld_{name}_{'on' if weld else 'off'}.png")
    SETTINGS.raytracing.experimental.set(weld_surface_seams=bool(weld))
    SceneManager.reset()
    SETTINGS.computing.set(available_memory_override=PINNED_BYTES)

    tri_counts = []
    original = rp.prepare_sparse_raster_coverage

    def spy(*args, **kwargs):
        merged = kwargs.get("merged", args[0] if args else None)
        if merged is not None and "tri_obj" in merged:
            tri_counts.append(int(merged["tri_obj"].shape[-1]))
        return original(*args, **kwargs)

    rp.prepare_sparse_raster_coverage = spy
    try:
        scene = SceneManager.instance().current_scene
        scene.set_video_settings(quality)
        build()
        Scene.save_frame(path, quality)
    finally:
        rp.prepare_sparse_raster_coverage = original

    import cv2

    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"could not read back {path}")
    return img.astype(np.int16), (max(tri_counts) if tri_counts else -1)


def main():
    quality = globals()[sys.argv[1]] if len(sys.argv) > 1 else MD
    shapes = sys.argv[2].split(",") if len(sys.argv) > 2 else list(SHAPES)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"quality={quality.resolution}")
    print(
        f"{'shape':9s} {'tris off':>9s} {'tris on':>8s} {'max|d|':>7s} "
        f"{'px>2':>7s} {'of':>8s}"
    )
    print("-" * 54)
    for name in shapes:
        build = SHAPES[name]
        off_img, off_tris = render_once(name, build, False, quality)
        on_img, on_tris = render_once(name, build, True, quality)
        delta = np.abs(off_img - on_img)
        moved = int((delta.max(axis=-1) > 2).sum())
        total = int(delta.shape[0] * delta.shape[1])
        print(
            f"{name:9s} {off_tris:9d} {on_tris:8d} {int(delta.max()):7d} "
            f"{moved:7d} {total:8d}",
            flush=True,
        )
    print()
    print(
        "tris on < tris off proves the weld engaged (each pole drops W-1 "
        "degenerate triangles). max|d| must be 0 on `plain`; on `checker` and "
        "`normals` any movement is a uv-indexing bug, since the u-seam is not "
        "welded and the pole rows carry no distinguishable texture detail."
    )


if __name__ == "__main__":
    main()
