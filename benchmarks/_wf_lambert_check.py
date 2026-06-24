import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algan import (  # noqa: E402
    Sync, Sphere, SceneManager, RIGHT, UP, IN, OUT,
    RED, GREEN, BLUE, WHITE, YELLOW, MeshLambertMaterial,
)
from algan.rendering.raytracing import (  # noqa: E402
    enable_ray_tracing, set_reflectivity, set_refractive_index)
from algan.rendering.raytracing.primitives import set_wavefront  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)


def render(tag):
    SceneManager.reset()
    enable_ray_tracing(1, pn_triangles=True, fragment_shading=True)
    set_wavefront(True)
    with Sync():
        (Sphere().scale(0.8).move(UP * 2.6 + OUT * 1.5)
         .set_material(MeshLambertMaterial(color=YELLOW)).spawn())
    scene = SceneManager.instance()
    out = os.path.join(OUT_DIR, f"wf_{tag}.png")
    frames = scene.save_frame(out)
    arr = (frames[-1].permute(1, 2, 0).float().cpu().numpy() * 255.0)
    return arr, out


def main():
    render("lambert")


if __name__ == "__main__":
    import torch
    with torch.inference_mode():
        main()
