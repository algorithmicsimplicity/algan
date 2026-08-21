"""Smoke test for general-wavefront refraction.

A transparent sphere sits in front of three opaque colored backdrop spheres.
We render it twice through the general wavefront: once with a glass refractive
index (1.5) and once with none (1.0 = straight transparency). Refraction must
bend the transmitted rays, so the backdrop seen *through* the sphere is
distorted -- the two images should differ substantially inside the sphere's
silhouette and be identical outside it (rays that miss the sphere are
unaffected).

    .venv/Scripts/python.exe benchmarks/_wf_refraction_check.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algan import (  # noqa: E402
    BLUE,
    GREEN,
    IN,
    RED,
    RIGHT,
    WHITE,
    MeshLambertMaterial,
    SceneManager,
    Sphere,
    Sync,
)
from algan.rendering.raytracing import (  # noqa: E402
    enable_ray_tracing,
    set_refractive_index,
)
from algan.rendering.raytracing.primitives import set_wavefront  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)


def render(ior, tag):
    SceneManager.reset()
    enable_ray_tracing(1, pn_triangles=True)
    set_wavefront(True)  # keep both renders on the wavefront
    with Sync():
        # Opaque colored backdrop behind the glass.
        for x, c in ((-1.4, RED), (0.0, GREEN), (1.4, BLUE)):
            (
                Sphere()
                .scale(0.55)
                .move(RIGHT * x + IN * 3.0)
                .set_material(MeshLambertMaterial(color=c))
                .spawn()
            )
        # Transparent sphere in front; refractive only when ior > 1.
        glass = (
            Sphere()
            .scale(1.3)
            .set_material(MeshLambertMaterial(color=WHITE, opacity=0.12))
        )
        if ior > 1.0:
            set_refractive_index(glass, ior)
        glass.spawn()
    scene = SceneManager.instance()
    out = os.path.join(OUT_DIR, f"wf_refraction_{tag}.png")
    frames = scene.save_frame(out)
    arr = frames[-1].permute(1, 2, 0).float().cpu().numpy() * 255.0
    return arr, out


def main():
    plain, p_path = render(1.0, "off")
    glass, g_path = render(1.5, "on")
    d = np.abs(plain.astype(np.float64) - glass.astype(np.float64)).mean(axis=2)
    changed = d > 8.0
    H, W = d.shape
    print("resolution:", d.shape)
    print("saved:", p_path, "|", g_path)
    print(
        f"max|d|={d.max():.1f}  mean|d|={d.mean():.3f}  "
        f"changed(>8)={int(changed.sum())} ({100.0 * changed.mean():.2f}%)"
    )
    if changed.any():
        ys, xs = np.nonzero(changed)
        cx, cy = xs.mean(), ys.mean()
        print(
            f"changed-region center=({cx:.0f},{cy:.0f}) of ({W},{H}); "
            f"bbox x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]"
        )
        # Refraction should change a meaningful but localized region centered
        # near the glass sphere (image center), not the whole frame.
        central = (abs(cx - W / 2) < W * 0.2) and (abs(cy - H / 2) < H * 0.2)
        localized = changed.mean() < 0.5
        substantial = changed.sum() > 500 and d.max() > 30
        print("REFRACTION_OK:", bool(central and localized and substantial))
    else:
        print("REFRACTION_OK: False (no change -> refraction had no effect)")


if __name__ == "__main__":
    import torch

    with torch.inference_mode():
        main()
