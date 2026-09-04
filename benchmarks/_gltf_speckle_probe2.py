"""What does ``_faces_viewer`` decide at the over-bright rim fragments?

The rim overshoot only exists on the lit path (MeshBasicMaterial rims are clean
monotone AA blends), and the only place the lit stages can invert a fragment's
lighting is the two-sided flip in ``_prep_normal``.  This probe replaces the
model's material with a debug stage that encodes that decision per fragment:

  R = 1  the flip fired (``_faces_viewer`` said the surface faces away)
  G = 1  the flip was decided by the SHADING-normal fallback
         (``|face_n . n| <= 0.1``, the branch the docstring warns about)
  B = 0.5 + 0.5 * (n . view_dir) before the flip
         (< 0.5 means the shading normal already points away from the viewer)

The base-colour texture is flattened so the readout is not modulated by the
checker, but the NORMAL map is kept -- it is part of what ``_faces_viewer``
sees.
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO / "tests" / "full_renders")

import numpy as np  # noqa: E402

from algan import *  # noqa: E402
from algan import HD, SETTINGS, Scene  # noqa: E402
from algan.rendering.shaders.fragment_shaders import FragmentStage  # noqa: E402
from algan.taichi_compat import ti  # noqa: E402

OUTDIR = REPO / "benchmarks" / "_gltf_probe2_out"
OUTDIR.mkdir(parents=True, exist_ok=True)
SETTINGS.paths.set(output_root=str(OUTDIR), output_directory=".")
SIZE = float(os.environ.get("SIZE", "2.6"))
KEEP_NMAP = os.environ.get("NMAP", "1") == "1"


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
    # Mirror _faces_viewer exactly, but report which branch decided it.
    n = n_interp.normalized()
    ndv = n.dot(view_dir)
    side = ndv
    fallback = 1.0
    fl = face_n.norm()
    if fl > 1e-12:
        fn = face_n * (1.0 / fl)
        d = fn.dot(n)
        if ti.abs(d) > 0.1:
            fallback = 0.0
            side = fn.dot(view_dir)
            if d < 0.0:
                side = -side
    flipped = 0.0
    if side < 0.0:
        flipped = 1.0
    return ti.math.vec4(flipped, fallback, 0.5 + 0.5 * ndv, in_glow)


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
        tmap = getattr(mob, "texture_map", None)
        if tmap is not None:
            new = tmap.clone()
            new[..., 0], new[..., 1], new[..., 2] = 0.85, 0.55, 0.25
            mob.texture_map = new
        if not KEEP_NMAP and getattr(mob, "normal_texture_map", None) is not None:
            mob.normal_texture_map = None
    model.set_fragment_shader(PROBE)

model.spawn(animate=False)
Scene.save_frame("probe2", HD)

import cv2  # noqa: E402

img = cv2.imread(str(OUTDIR / "probe2.png"))  # BGR
b, g, r = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
cov = img.astype(int).sum(2) > 12
print("covered fragments:", int(cov.sum()))
print("flip fired (R>128):", int((r > 128).sum()))
print("shading-normal fallback (G>128):", int((g > 128).sum()))
print("n.view_dir < 0 (B<120):", int((b[cov] < 120).sum()))

ys, xs = np.where((r > 128) & cov)
if len(ys):
    print("flip bbox x", xs.min(), xs.max(), "y", ys.min(), ys.max())
    for y, x in list(zip(ys, xs))[:25]:
        print("   ", x, y, "fallback", g[y, x], "ndotv_enc", b[y, x])
