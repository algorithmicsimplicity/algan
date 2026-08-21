"""Validate simultaneous reflection + refraction (glass) in the general wavefront.

A transparent sphere sits in front of three colored backdrop spheres, with a
bright sphere placed up/front so it reflects off the glass. We render three ways:

  reflect : reflectivity only (no refractive index)  -> reflection, no bending
  refract : refractive index only (reflectivity 0)   -> bending, no reflection
  both    : reflectivity AND refractive index         -> a ray that splits into
            a reflected + refracted pair, so BOTH effects appear at once

If the split works, ``both`` must differ from ``refract`` (it adds the
reflection) AND from ``reflect`` (it adds the refraction). Saves all three PNGs.

    .venv/Scripts/python.exe benchmarks/_wf_reflect_refract_check.py
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
    OUT,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    MeshLambertMaterial,
    SceneManager,
    Sphere,
    Sync,
)
from algan.rendering.raytracing import (  # noqa: E402
    enable_ray_tracing,
    set_reflectivity,
    set_refractive_index,
)
from algan.rendering.raytracing.primitives import set_wavefront  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)


def render(reflect, ior, tag):
    SceneManager.reset()
    enable_ray_tracing(1, pn_triangles=True)
    set_wavefront(True)
    with Sync():
        for x, c in ((-1.4, RED), (0.0, GREEN), (1.4, BLUE)):
            (
                Sphere()
                .scale(0.55)
                .move(RIGHT * x + IN * 3.0)
                .set_material(MeshLambertMaterial(color=c))
                .spawn()
            )
        # Bright object up/front to reflect off the glass.
        (
            Sphere()
            .scale(0.8)
            .move(UP * 2.6 + OUT * 1.5)
            .set_material(MeshLambertMaterial(color=YELLOW))
            .spawn()
        )
        glass = (
            Sphere()
            .scale(1.3)
            .set_material(MeshLambertMaterial(color=WHITE, opacity=0.12))
        )
        if reflect > 0.0:
            set_reflectivity(glass, reflect)
        if ior > 1.0:
            set_refractive_index(glass, ior)
        glass.spawn()
    scene = SceneManager.instance()
    out = os.path.join(OUT_DIR, f"wf_rr_{tag}.png")
    frames = scene.save_frame(out)
    arr = frames[-1].permute(1, 2, 0).float().cpu().numpy() * 255.0
    return arr, out


def diff(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64)).mean(axis=2)
    return d.max(), float((d > 8.0).sum())


def main():
    refl, p_r = render(0.4, 1.0, "reflect")
    refr, p_f = render(0.0, 1.5, "refract")
    both, p_b = render(0.4, 1.5, "both")
    print("saved:", p_r, "|", p_f, "|", p_b)
    m_rf, n_rf = diff(both, refr)  # both vs refract-only -> reflection added
    m_rl, n_rl = diff(both, refl)  # both vs reflect-only -> refraction added
    print(f"both vs refract-only: max|d|={m_rf:.1f} changed={n_rf}  (reflection added)")
    print(f"both vs reflect-only: max|d|={m_rl:.1f} changed={n_rl}  (refraction added)")
    adds_reflection = (m_rf > 25) and (n_rf > 300)
    adds_refraction = (m_rl > 25) and (n_rl > 300)
    print("REFLECT_AND_REFRACT_OK:", bool(adds_reflection and adds_refraction))


if __name__ == "__main__":
    import torch

    with torch.inference_mode():
        main()
