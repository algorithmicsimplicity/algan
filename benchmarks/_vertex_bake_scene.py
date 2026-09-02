"""A scene that exercises the *per-vertex* material shaders, for A/B use.

``material_shaders``' torch shaders only run when a primitive is **not**
shaded per fragment (``RayTracedTrianglePrimitive._shaded_per_fragment``):
every core material is ported to the shading kernel, so the default render
never calls them. ``set_fragment_shading(False)`` is the supported way back to
the vertex bake, and this fixture takes it -- otherwise an A/B of those
functions measures nothing at all.

Records only, like ``tests/fast/scene.py``: drive it with
``benchmarks/_torch_compile_ab.py --scene benchmarks/_vertex_bake_scene.py``.

Polyhedra only (no Sphere/Cylinder/Torus): the PN family costs ~20 s of extra
Taichi specialisation per arm and this measures torch, not kernels. One
``PointLight`` and one grid per material, because the vertex bake sees only
point lights and shades once per light per primitive collection.
"""

from __future__ import annotations

from algan import *
from algan.rendering.raytracing import set_fragment_shading

# The whole point of the fixture: send core materials down the torch
# per-vertex bake instead of the in-kernel fragment shader.
set_fragment_shading(False)

Scene.set_background(DARKER_GRAY)

COLUMNS = 8
ROWS = 3


def _grid(build, origin):
    mobs = []
    for row in range(ROWS):
        for col in range(COLUMNS):
            mob = build()
            mob.move(
                origin + RIGHT * (col - (COLUMNS - 1) / 2) * 0.75 + DOWN * row * 0.75
            )
            mobs.append(mob)
    return Group(*mobs)


with Off():
    PointLight(location=LEFT * 3 + UP * 2 + OUT * 4, color=WHITE, intensity=1.0).spawn(
        animate=False
    )

    lambert = _grid(
        lambda: Cube(size=0.5).set_material(MeshLambertMaterial(color=ORANGE)),
        UP * 2.0,
    )
    standard = _grid(
        lambda: Icosahedron(edge_length=0.32).set_material(
            MeshStandardMaterial(color=RED, roughness=0.35, metalness=0.4)
        ),
        DOWN * 0.4,
    )
    physical = _grid(
        lambda: Octahedron(edge_length=0.34).set_material(
            MeshPhysicalMaterial(
                color=TEAL,
                roughness=0.4,
                metalness=0.2,
                clearcoat=0.6,
                sheen=0.4,
                sheen_color=WHITE,
            )
        ),
        DOWN * 2.8,
    )

with Sync(runtime=0.2):
    lambert.spawn()
    standard.spawn()
    physical.spawn()

with Sync(runtime=1.0):
    lambert.rotate(90, UP)
    standard.rotate(90, RIGHT)
    physical.rotate(90, OUT)
