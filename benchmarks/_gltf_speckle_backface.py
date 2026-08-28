"""Is the rim speckle the BACK of the sphere leaking through the silhouette?

Rebuilds the ``textured_icosphere.glb`` mesh as a plain ``TriangleMesh`` and
renders it twice: once whole, once with every triangle facing away from the
camera deleted. If the speckle is a back-facing facet winning at a silhouette
edge, arm B is clean.

Usage: ``_gltf_speckle_backface.py`` (writes to benchmarks/_gltf_bf_out).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO / "tests" / "full_renders")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

from algan import *  # noqa: E402
from algan import PREVIEW, SETTINGS, Scene  # noqa: E402
from algan.mobs.three_d_models.mesh import (  # noqa: E402
    TriangleMesh,
)

OUTDIR = REPO / "benchmarks" / "_gltf_bf_out"
OUTDIR.mkdir(parents=True, exist_ok=True)
SETTINGS.paths.set(output_root=str(OUTDIR), output_directory=".")

CULL = os.environ.get("BF_CULL", "0") == "1"

scene_tm = trimesh.load("assets/textured_icosphere.glb", process=False, force="scene")
geom = list(scene_tm.geometry.values())[0]
verts = np.asarray(geom.vertices, dtype=np.float32)
faces = np.asarray(geom.faces, dtype=np.int64)
vnorm = np.asarray(geom.vertex_normals, dtype=np.float32)
uvs = np.asarray(geom.visual.uv, dtype=np.float32)
mat = geom.visual.material
base_img = np.asarray(mat.baseColorTexture, dtype=np.float32) / 255.0

# The model is normalized to 2.6 across and moved UP*0.2, matching the scene.
scale = 2.6 / float(np.abs(verts).max() * 2)
verts = verts * scale
verts[:, 1] += 0.2

if CULL:
    # Default camera sits on +Z looking at the origin; drop triangles whose
    # geometric normal points away from it.
    cam = np.array([0.0, 0.0, 12.0], dtype=np.float32)
    p0, p1, p2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    fn /= np.linalg.norm(fn, axis=-1, keepdims=True) + 1e-20
    centroid = (p0 + p1 + p2) / 3.0
    view = cam[None] - centroid
    view /= np.linalg.norm(view, axis=-1, keepdims=True)
    keep = (fn * view).sum(-1) > 0.0
    print(f"keeping {int(keep.sum())} of {len(faces)} triangles")
    faces = faces[keep]

with Off():
    AmbientLight(color=WHITE, intensity=0.55).spawn(animate=False)
    DirectionalLight(
        location=RIGHT * 4 + UP * 5 + OUT * 4,
        target=ORIGIN,
        color=WHITE,
        intensity=1.0,
    ).spawn(animate=False)
    mesh = TriangleMesh(
        vertices=torch.as_tensor(verts),
        faces=torch.as_tensor(faces),
        normals=torch.as_tensor(vnorm),
        uvs=torch.as_tensor(uvs),
        texture=torch.as_tensor(base_img[..., :3]),
    )

Scene.set_background(DARKER_GRAY)
mesh.spawn(animate=False)

name = "bf_culled" if CULL else "bf_whole"
Scene.save_frame(name, HD if os.environ.get("BF_HD") == "1" else PREVIEW)

import cv2  # noqa: E402

img = cv2.imread(str(OUTDIR / f"{name}.png"))
med = cv2.medianBlur(img, 3)
d = (img.astype(np.int32) - med.astype(np.int32)).max(2)
ys, xs = np.where(d > 25)
print(f"{name}: {len(ys)} speckle pixels, max excess {d.max()}")
for y, x in list(zip(ys, xs))[:25]:
    print("   ", x, y, img[y, x], med[y, x])
