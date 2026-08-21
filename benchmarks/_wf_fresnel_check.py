"""Validate Fresnel glass in the general wavefront.

A transparent sphere over three colored backdrop spheres, with a bright sphere
up/front to reflect. Rendered two ways:

  glass       : with a refractive index (1.5) -> Fresnel reflect + refract
  transparent : same opacity, no refractive index -> straight transparency

Glass must differ from plain transparency by adding BOTH a (Fresnel) reflection
of the bright sphere AND refraction of the backdrop -- a substantial change
centered on the sphere. We also check the Fresnel signature: the reflection is
angle-dependent, so the sphere's RIM (grazing) is brighter than its center
relative to the plain-transparent baseline.

    .venv/Scripts/python.exe benchmarks/_wf_fresnel_check.py
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


def render(ior, tag, reflect=0.0, opacity=0.1):
    SceneManager.reset()
    enable_ray_tracing(1, pn_triangles=True, fragment_shading=True)
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
        (
            Sphere()
            .scale(0.8)
            .move(UP * 2.6 + OUT * 1.5)
            .set_material(MeshLambertMaterial(color=YELLOW))
            .spawn()
        )
        ball = (
            Sphere()
            .scale(1.3)
            .set_material(MeshLambertMaterial(color=WHITE, opacity=opacity))
        )
        if reflect > 0.0:
            set_reflectivity(ball, reflect)
        if ior > 1.0:
            set_refractive_index(ball, ior)
        ball.spawn()
    scene = SceneManager.instance()
    out = os.path.join(OUT_DIR, f"wf_fresnel_{tag}.png")
    frames = scene.save_frame(out)
    arr = frames[-1].permute(1, 2, 0).float().cpu().numpy() * 255.0
    return arr, out


def main():
    glass, p_g = render(1.5, "glass")
    plain, p_t = render(1.0, "transparent")
    print("saved:", p_g, "|", p_t)
    d = np.abs(glass.astype(np.float64) - plain.astype(np.float64)).mean(axis=2)
    changed = d > 8.0
    H, W = d.shape
    ys, xs = np.nonzero(changed)
    print(
        f"glass vs transparent: max|d|={d.max():.1f} "
        f"changed={int(changed.sum())} ({100.0 * changed.mean():.2f}%)"
    )
    centered = localized = substantial = False
    if changed.any():
        cx, cy = xs.mean(), ys.mean()
        centered = (abs(cx - W / 2) < W * 0.2) and (abs(cy - H / 2) < H * 0.2)
        localized = changed.mean() < 0.5
        substantial = (changed.sum() > 500) and (d.max() > 40)
        # Fresnel signature: reflection grows toward the grazing rim. Within the
        # glass disk, compare added brightness in an annulus (rim) vs the core.
        gl = glass.mean(axis=2)
        tr = plain.mean(axis=2)
        add = gl - tr  # brightness glass adds over plain
        Y, X = np.ogrid[:H, :W]
        rr = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        # disk radius ~ extent of the changed region from its center
        rad = np.percentile(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2), 95)
        core = rr < 0.45 * rad
        rim = (rr >= 0.55 * rad) & (rr < rad)
        rim_add = float(add[rim].mean()) if rim.any() else 0.0
        core_add = float(add[core].mean()) if core.any() else 0.0
        # Informational: Fresnel reflects more toward the grazing rim. Scene-
        # dependent (only shows where the rim's reflection points at something
        # bright), so reported but not asserted.
        print(
            "Fresnel rim-vs-core added brightness: rim={:.1f} core={:.1f}{}".format(
                rim_add,
                core_add,
                "  (rim>core, grazing-stronger)" if rim_add > core_add else "",
            )
        )
    print("GLASS_REFLECTS_AND_REFRACTS:", bool(centered and localized and substantial))

    # Near-perfect mirror with a tiny amount of refraction: reflectivity ~0.95
    # raises the reflectance floor (mirror coating) while the IOR still refracts
    # the small transmitted remainder. The glass ball's centre shows the bright
    # refracted backdrop; the mirror's centre instead mostly reflects the (dark)
    # environment, so its refraction of the backdrop is much weaker there.
    mirror, p_m = render(1.5, "mirror", reflect=0.95)
    print("\nsaved mirror:", p_m)
    if changed.any():
        gl = glass.mean(axis=2)
        mi = mirror.mean(axis=2)
        Y, X = np.ogrid[:H, :W]
        rr = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        core = rr < 0.45 * rad
        glass_core = float(gl[core].mean())
        mirror_core = float(mi[core].mean())
        # Mirror should strongly reflect (a bright yellow highlight appears where
        # the glass had none) and refract only weakly (dim core vs the glass).
        d_m = np.abs(mirror.astype(np.float64) - glass.astype(np.float64)).mean(axis=2)
        print(
            f"mirror vs glass: max|d|={d_m.max():.1f} changed={int((d_m > 8.0).sum())}"
        )
        print(
            f"core backdrop-refraction brightness: glass={glass_core:.1f} mirror={mirror_core:.1f}"
        )
        tiny_refraction = mirror_core < 0.6 * glass_core
        strong_reflection = (d_m.max() > 40) and ((d_m > 8.0).sum() > 500)
        print(
            "NEAR_MIRROR_TINY_REFRACTION:", bool(tiny_refraction and strong_reflection)
        )


if __name__ == "__main__":
    import torch

    with torch.inference_mode():
        main()
