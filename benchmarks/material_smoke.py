"""Smoke render: exercises the material system through the real pipeline.

Renders a couple of frames of spheres with Phong / Standard / Normal materials
(Phong carries both vector shader params -- emissive and specular -- which is the
plumbing we most want to validate). Saves a single PNG.
"""

from __future__ import annotations

import os

from algan import *  # noqa: F401,F403

with Sync():
    a = (
        Sphere()
        .move(LEFT * 3)
        .set_material(
            MeshPhongMaterial(
                color=BLUE, specular=0xFFFFFF, shininess=60, emissive=0x330000
            )
        )
        .spawn()
    )
    b = (
        Sphere()
        .set_material(MeshStandardMaterial(color=RED, metalness=1.0, roughness=0.25))
        .spawn()
    )
    c = Sphere().move(RIGHT * 3).set_material(MeshNormalMaterial()).spawn()

a.move(UP * 0.5)

scene = SceneManager.instance()
out = os.path.join(os.path.dirname(__file__), "material_smoke.png")
scene.save_frame(out)
print(
    "SAVED",
    out,
    "exists:",
    os.path.exists(out),
    "bytes:",
    os.path.getsize(out) if os.path.exists(out) else 0,
)
