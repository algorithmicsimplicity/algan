"""Smoke render for Tier 1: materials under the physical path tracer.

Enables physical lighting, then renders metal vs. dielectric MeshStandardMaterial
spheres plus a Lambert sphere. With Tier 1, set_material auto-routes each
material's (metalness, roughness) into the per-hit path-traced shading, so the
metal should read as a dark tinted metal with sharp reflections and the
dielectric/Lambert as diffuse. Saves a single PNG.
"""

from __future__ import annotations

import os

from algan import *  # noqa: F401,F403
from algan.rendering.raytracing import enable_ray_tracing

# Physical path tracer; low sample count to keep the smoke test quick-ish
# (noisy but enough to read metal vs. diffuse).
enable_ray_tracing(physical_lighting=True, samples_per_pixel=8)

with Sync():
    Sphere().move(LEFT * 3).set_material(
        MeshStandardMaterial(color=RED, metalness=1.0, roughness=0.2)
    ).spawn()
    Sphere().set_material(
        MeshStandardMaterial(color=RED, metalness=0.0, roughness=0.6)
    ).spawn()
    Sphere().move(RIGHT * 3).set_material(MeshLambertMaterial(color=BLUE)).spawn()

scene = SceneManager.instance()
out = os.path.join(os.path.dirname(__file__), "material_smoke_physical.png")
scene.save_frame(out)
print(
    "SAVED",
    out,
    "exists:",
    os.path.exists(out),
    "bytes:",
    os.path.getsize(out) if os.path.exists(out) else 0,
)
