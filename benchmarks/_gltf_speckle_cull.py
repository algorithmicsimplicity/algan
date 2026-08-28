"""Decisive test for the rim speckle: is the offending fragment on a
BACK-FACING facet of the closed sphere?

Renders the real ``ThreeDModelMob`` (same PBR material, normal map, textures)
twice at HD.  ``CULL=1`` monkeypatches the glTF loader so every triangle whose
geometric normal points away from the camera is dropped before the mesh mob is
built -- the front shell alone.  Everything else is identical.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO / "tests" / "full_renders")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from algan import *  # noqa: E402
from algan import HD, SETTINGS, Scene  # noqa: E402
from algan.mobs.three_d_models import gltf_loader  # noqa: E402

OUTDIR = REPO / "benchmarks" / "_gltf_cull_out"
OUTDIR.mkdir(parents=True, exist_ok=True)
SETTINGS.paths.set(output_root=str(OUTDIR), output_directory=".")

CULL = os.environ.get("CULL", "0") == "1"
SIZE = float(os.environ.get("SIZE", "7.0"))
FLAT = os.environ.get("FLAT_NORMALS", "0") == "1"

if CULL:
    _orig = gltf_loader._convert_mesh

    def _culled(geom, material_index, name):
        data = _orig(geom, material_index, name)
        if data is None:
            return data
        v = data.vertices.numpy()
        f = data.faces.numpy()
        p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        fn = np.cross(p1 - p0, p2 - p0)
        fn /= np.linalg.norm(fn, axis=-1, keepdims=True) + 1e-20
        # The model is centred on the origin and the camera looks down -Z from
        # +Z, so "faces the camera" is a positive z component of the normal at
        # the (unit-sphere) centroid; use the true view vector for accuracy.
        cam = np.array([0.0, 0.0, 12.0 / max(SIZE / 2.0, 1e-6)], dtype=np.float32)
        centroid = (p0 + p1 + p2) / 3.0
        view = cam[None] - centroid
        view /= np.linalg.norm(view, axis=-1, keepdims=True)
        keep = (fn * view).sum(-1) > 0.0
        print(f"culling: keeping {int(keep.sum())} of {len(f)} triangles")
        data.faces = torch.as_tensor(f[keep])
        return data

    gltf_loader._convert_mesh = _culled

Scene.set_background(DARKER_GRAY)

with Off():
    AmbientLight(color=WHITE, intensity=0.55).spawn(animate=False)
    DirectionalLight(
        location=RIGHT * 4 + UP * 5 + OUT * 4,
        target=ORIGIN,
        color=WHITE,
        intensity=1.0,
    ).spawn(animate=False)
    model = ThreeDModelMob(
        "assets/textured_icosphere.glb",
        normalize=True,
        normalize_size=SIZE,
        smooth_normals=not FLAT,
    ).move(UP * 0.2)
    # Flat albedo so any variation is shading, not texture.
    stack = [model]
    while stack:
        mob = stack.pop()
        stack.extend(getattr(mob, "children", ()) or ())
        tmap = getattr(mob, "texture_map", None)
        if tmap is None:
            continue
        new = tmap.clone()
        new[..., 0] = 0.85
        new[..., 1] = 0.55
        new[..., 2] = 0.25
        mob.texture_map = new

model.spawn(animate=False)

name = f"cull{int(CULL)}_flat{int(FLAT)}"
Scene.save_frame(name, HD)

import cv2  # noqa: E402

img = cv2.imread(str(OUTDIR / f"{name}.png"))
med = cv2.medianBlur(img, 3)
d = (img.astype(np.int32) - med.astype(np.int32)).max(2)
print(f"{name}: {int((d > 25).sum())} speckle pixels, max excess {d.max()}")
g = img[..., 2].astype(int)
bg = int(np.median(g[:40, :40]))
for y in range(500, 800, 20):
    xs = np.where(np.abs(g[y] - bg) > 4)[0]
    if len(xs) < 5:
        continue
    l = xs[0]
    print(y, "L", l, g[y, l - 1 : l + 6])
