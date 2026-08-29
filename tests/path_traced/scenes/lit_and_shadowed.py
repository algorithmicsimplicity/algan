"""Lit transport under the path tracer: NEE, shadows, emission, GGX, bleed.

A white diffuse floor seen obliquely under one point light (next-event
estimation with a shadow-casting pillar), an emissive slab lighting its
neighbourhood through the sampled emitter table, a rough metal prism (VNDF
specular with energy compensation), and a red wall close enough to bleed
onto the floor through the diffuse bounce. The metal prism turns during the
second so specular paths move. More samples than the other scenes: lit
content is where the estimator actually has variance.
"""

from algan import *

SETTINGS.raytracing.set(samples_per_pixel=48, shadows=True)

Scene.set_background(BLACK)
Scene.clear_light_sources()

with Off():
    PointLight(
        location=UP * 3.0 + OUT * 4.0 + LEFT * 1.5,
        color=WHITE,
        intensity=2.0,
    ).spawn(animate=False)

    floor = Prism(dimensions=(9.0, 0.2, 5.0))
    floor.set_material(MeshLambertMaterial(color=WHITE))
    floor.move(DOWN * 1.4)
    floor.spawn(animate=False)

    pillar = Prism(dimensions=(0.7, 2.4, 0.7))
    pillar.set_material(MeshLambertMaterial(color=WHITE))
    pillar.move(LEFT * 1.8 + DOWN * 0.1)
    pillar.spawn(animate=False)

    wall = Prism(dimensions=(0.2, 2.6, 2.4))
    wall.set_material(MeshLambertMaterial(color=RED))
    wall.move(RIGHT * 3.4 + DOWN * 0.1)
    wall.spawn(animate=False)

    glow = Prism(dimensions=(1.1, 1.1, 0.08))
    glow.set_material(
        MeshLambertMaterial(color=BLACK, emissive=WHITE, emissive_intensity=1.5)
    )
    glow.move(UP * 1.6 + RIGHT * 1.2 - OUT * 1.2)
    glow.spawn(animate=False)

    metal = Prism(dimensions=(1.0, 1.0, 1.0))
    metal.set_material(MeshStandardMaterial(color=WHITE, metalness=1.0, roughness=0.35))
    metal.move(RIGHT * 0.9 + DOWN * 0.8)
    metal.spawn(animate=False)

metal.rotate(60, UP)
