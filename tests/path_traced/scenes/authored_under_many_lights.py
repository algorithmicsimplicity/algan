"""Authored-appearance materials under more lights than the shadow cap.

The one branch of ``pt_shade`` no other scene in this suite reaches: a toon
floor and pillar and a Manim-shaded box under 24 point lights on a ring.
Twenty-four is past ``max_shadow_lights`` (16), so at the shipped
``pt_authored_light_sampling = "auto"`` these surfaces SAMPLE their light rows
rather than summing them (roadmap section 6a-bis) -- every light casts a
shadow, and the per-crossing cost is one shadow ray instead of sixteen.

Deliberately: dim lights and an off-white albedo (24 lights at full strength
is a white frame, which pins the comparison at 255 and measures nothing, and a
toon material's bands only show below saturation), an ambient and a hemisphere
row so the deterministic ambient fill runs beside the sampled rows, one lit
``MeshLambertMaterial`` sphere so the physically-integrated branch is in the
same frame as the authored one, and a standing pillar rather than a flat panel
so its shadow lands on the floor strip the near-edge-on default camera
actually shows (the framing every other scene in this suite uses).
"""

import math

from algan import *
from algan.rendering.shaders.materials import ManimMaterial

SETTINGS.raytracing.set(samples_per_pixel=32, shadows=True)

Scene.set_background(BLACK)
Scene.clear_lights()

with Off():
    NUM_LIGHTS = 24
    for i in range(NUM_LIGHTS):
        angle = 2.0 * math.pi * i / NUM_LIGHTS
        PointLight(
            location=(
                RIGHT * (4.0 * math.cos(angle))
                + UP * (2.6 + 1.8 * math.sin(angle))
                + OUT * (3.5 + 2.0 * math.sin(angle * 2.0))
            ),
            color=WHITE,
            intensity=0.9 / NUM_LIGHTS,
        ).spawn(animate=False)
    AmbientLight(color=WHITE, intensity=0.08).spawn(animate=False)
    HemisphereLight(color=WHITE, ground_color=BLUE, intensity=0.12).spawn(animate=False)

    floor = Prism(width=9.0, height=0.2, depth=5.0)
    floor.set_material(MeshToonMaterial(color=WHITE * 0.75))
    floor.set_opacity(1.0)
    floor.move(DOWN * 1.4)
    floor.spawn(animate=False)

    pillar = Prism(width=0.6, height=2.2, depth=0.6)
    pillar.set_material(MeshToonMaterial(color=WHITE * 0.75))
    pillar.set_opacity(1.0)
    pillar.move(LEFT * 0.4 + DOWN * 0.2)
    pillar.spawn(animate=False)

    box = Prism(width=1.2, height=1.2, depth=1.2)
    box.set_material(ManimMaterial(color=RED))
    box.set_opacity(1.0)
    box.move(RIGHT * 1.8 + DOWN * 0.6)
    box.spawn(animate=False)

    ball = Sphere(radius=0.8)
    ball.set_material(MeshLambertMaterial(color=BLUE))
    ball.set_opacity(1.0)
    ball.move(LEFT * 2.4 + DOWN * 0.6)
    ball.spawn(animate=False)

box.rotate(60, UP)
