"""Fragment (per-fragment / Phong) shading for the deterministic ray tracer.

Run directly: python tests/test_raytracing_fragment_shading.py

The deterministic ray tracer normally shades per vertex (Gouraud): the material
shader is evaluated at the triangle corners and the kernel interpolates the
baked colours. With ``enable_ray_tracing(fragment_shading=True)`` the core lit
materials are instead shaded per fragment in-kernel from the raw albedo, a
per-primitive material block and the scene's point lights.

This test renders a strongly specular sphere both ways and checks that:
* fragment shading produces a *different* image (the specular highlight moves /
  sharpens -- Gouraud smears it across the coarse tessellation), and
* a mixed scene (lit mesh + Text + a non-core material) renders without error
  under fragment shading (bezier and non-core-material pass-through paths).

The renders go through mp4, so the difference threshold is set well above codec
noise rather than checking exact pixels.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import torch

from algan import LEFT, OUT, RIGHT, UP, Sphere, Text
from algan.rendering.shaders.materials import (
    MeshStandardMaterial,
    MeshToonMaterial,
)
from algan.settings.render_settings import RenderSettings
from algan.utils.algan_utils import render_to_file

OUT_DIR = os.path.join(os.path.dirname(__file__), "raytracing_outputs_v2")
SETTINGS = RenderSettings((480, 320), 6, anti_alias_level=1, fxaa=False)


def _build_specular_sphere():
    # Low roughness -> a tight specular highlight, where Gouraud (vertex) and
    # Phong (fragment) shading differ the most on a tessellated sphere.
    (Sphere(radius=0.9)
     .set_material(MeshStandardMaterial(roughness=0.12, metalness=0.0))
     .spawn())


def _build_mixed_scene():
    (Sphere(radius=0.7)
     .set_material(MeshStandardMaterial(roughness=0.2))
     .spawn()).move(LEFT * 1.5)
    # A non-core material (toon) must fall back to vertex shading (pass-through).
    (Sphere(radius=0.7)
     .set_material(MeshToonMaterial())
     .spawn()).move(RIGHT * 1.5)
    Text("FRAG").spawn().move(UP * 1.4 + OUT * 0.05)  # bezier pass-through


def _render(name, build, fragment_shading):
    from algan.rendering.raytracing import (
        enable_ray_tracing,
        set_fragment_shading,
    )

    enable_ray_tracing()
    set_fragment_shading(fragment_shading)
    try:
        build()
        render_to_file(file_name=name, output_dir=OUT_DIR, output_path="",
                       render_settings=SETTINGS, file_extension="mp4")
    finally:
        set_fragment_shading(False)
    return os.path.join(OUT_DIR, f"{name}.mp4")


def _read_frames(path):
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

    vertex_path = _render("frag_vertex", _build_specular_sphere, False)
    fragment_path = _render("frag_fragment", _build_specular_sphere, True)

    vert = np.asarray(_read_frames(vertex_path))
    frag = np.asarray(_read_frames(fragment_path))
    assert len(vert) > 0 and len(frag) > 0, "no frames decoded"
    n = min(len(vert), len(frag))
    vert, frag = vert[:n], frag[:n]

    # The fragment-shaded sphere is lit somewhere (early frames are the spawn
    # fade-in, so check globally rather than a single frame).
    assert frag.max() > 20, "fragment-shaded render is blank"

    diff = np.abs(vert.astype(np.float64) - frag.astype(np.float64))
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    print(f"vertex vs fragment shading: max|diff|={max_diff:.1f} "
          f"mean|diff|={mean_diff:.3f} over {n} frames")
    # A real per-fragment specular highlight differs from the Gouraud version by
    # far more than mp4 codec noise (a handful of levels).
    assert max_diff > 30, (
        f"fragment shading barely changed the image (max|diff|={max_diff:.1f}); "
        "expected a visible specular-highlight difference")

    # Side by side at the brightest (most fully spawned) fragment frame.
    bright = int(frag.reshape(n, -1).max(1).argmax())
    side = np.concatenate((vert[bright], frag[bright]), axis=1).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT_DIR, "fragment_side_by_side.png"), side)

    # Mixed scene (core + non-core material + Text) must render without error.
    mixed_path = _render("frag_mixed", _build_mixed_scene, True)
    mixed = np.asarray(_read_frames(mixed_path))
    assert len(mixed) > 0, "mixed-scene render produced no frames"
    assert mixed.max() > 20, "mixed-scene render is blank"

    print("fragment shading test passed")


if __name__ == "__main__":
    main()
