"""Reproduce the user's HD/AA=2 wavefront OOM and confirm it is fixed.

Renders one full-resolution frame of the neural_net scene (PN patches + a bezier
text label + fragment shading) through the general wavefront at HD with AA=2 (so
the frame is super-sampled to 3840x2160 = ~8.3M rays, which the new ray-offset
tiling splits into several wavefront_tile_rays tiles). Each tile's per-ray state
is pool-allocated and released, so the ~hundreds-of-MB no longer piles up.

If this completes without OutOfRenderMemory / CUDA OOM, the fix holds.

    .venv/Scripts/python.exe benchmarks/_wf_oom_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algan import *  # noqa: F401,F403
from algan.mobs.neural_nets.neural_net import NeuralNetMLP  # noqa: E402
from algan.rendering.raytracing import enable_ray_tracing  # noqa: E402
from algan.rendering.raytracing.primitives import set_wavefront  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)

enable_ray_tracing(1, pn_triangles=True, fragment_shading=True, shadows=False)
set_wavefront(True)
HD.super_sampling_anti_aliasing = 2

scene = SceneManager.instance()
scene.set_render_settings(HD)

with Off():
    nn = (
        NeuralNetMLP([10, 10, 10])
        .set_material(MeshStandardMaterial(color=GREEN))
        .spawn()
    )
    Text("Neural Network").move_next_to(nn, UP).spawn()
nn.move(DOWN)

out = os.path.join(OUT_DIR, "wf_oom_hd_aa2.png")
scene.save_frame(out)
print(
    "RENDERED OK (no OOM):",
    out,
    "exists:",
    os.path.exists(out),
    "bytes:",
    os.path.getsize(out) if os.path.exists(out) else 0,
)
