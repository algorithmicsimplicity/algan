"""Probe what the rim-speckle fragments actually are.

Replaces the model's material with a debug fragment stage that encodes, per
fragment:

  R = 1 when the GEOMETRIC normal faces away from the viewer (a back-facing
      facet won primary visibility)
  G = max(shading_normal . view_dir, 0)   (how front-on the shading normal is)
  B = max(shading_normal . light_dir, 0)  for light 0, AFTER the two-sided flip
      that ``_prep_normal`` would apply

so a saturated red pixel on the silhouette means "this pixel is shaded from a
back-facing triangle".
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO / "tests" / "full_renders")

import numpy as np  # noqa: E402

from algan import *  # noqa: E402
from algan import HD, SETTINGS, Scene  # noqa: E402
from algan.rendering.raytracing.shading_taichi import (  # noqa: E402
    _faces_viewer,
    _light_eval,
)
from algan.rendering.shaders.fragment_shaders import FragmentStage  # noqa: E402
from algan.taichi_compat import ti  # noqa: E402

OUTDIR = REPO / "benchmarks" / "_gltf_probe_out"
OUTDIR.mkdir(parents=True, exist_ok=True)
SETTINGS.paths.set(output_root=str(OUTDIR), output_directory=".")
SIZE = float(os.environ.get("SIZE", "7.0"))


@ti.func
def _probe(
    pos,
    view_dir,
    n_interp,
    face_n,
    in_rgb,
    in_glow,
    params: ti.template(),
    f,
    prim,
    off,
    light_pos: ti.template(),
    light_col: ti.template(),
    num_lights,
    shadows: ti.template(),
    vis,
):
    n = n_interp.normalized()
    back = 0.0
    fl = face_n.norm()
    if fl > 1e-12:
        fn = face_n * (1.0 / fl)
        if fn.dot(view_dir) < 0.0:
            back = 1.0
    if not _faces_viewer(n, face_n, view_dir):
        n = -n
    ndl = 0.0
    for li in range(num_lights):
        ld, _lc, _sw, _fr = _light_eval(light_pos, light_col, f, li, pos, n)
        d = ti.max(n.dot(ld), 0.0)
        if d > ndl:
            ndl = d
    return ti.math.vec4(back, ti.max(n.dot(view_dir), 0.0), ndl, in_glow)


PROBE = FragmentStage(_probe, [])

Scene.set_background(BLACK)

with Off():
    AmbientLight(color=WHITE, intensity=0.55).spawn(animate=False)
    DirectionalLight(
        location=RIGHT * 4 + UP * 5 + OUT * 4,
        target=ORIGIN,
        color=WHITE,
        intensity=1.0,
    ).spawn(animate=False)
    model = Model3D("assets/textured_icosphere.glb", fit_to_size=SIZE).move(UP * 0.2)
    stack = [model]
    while stack:
        mob = stack.pop()
        stack.extend(getattr(mob, "children", ()) or ())
        if getattr(mob, "texture_map", None) is not None:
            mob.texture_map = None
        if getattr(mob, "normal_texture_map", None) is not None:
            mob.normal_texture_map = None
    model.set_fragment_shader(PROBE)

model.spawn(animate=False)
Scene.save_frame("probe", HD)

import cv2  # noqa: E402

img = cv2.imread(str(OUTDIR / "probe.png"))  # BGR
b, g, r = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
ys, xs = np.where(r > 128)
print(f"back-facing fragments: {len(ys)}")
if len(ys):
    print("  bbox", xs.min(), xs.max(), ys.min(), ys.max())
    for y, x in list(zip(ys, xs))[:20]:
        print("   ", x, y, "back=1", "ndotv", g[y, x], "ndotl", b[y, x])
