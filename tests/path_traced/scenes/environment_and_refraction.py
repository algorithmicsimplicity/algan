"""Environment lighting, escape, mirror reflection and refraction under PT.

A deterministic (analytically built, no RNG) equirect environment map with a
bright sun patch on the camera side lights the scene through the env NEE
table; escaping rays fold the map, so the background IS the map. A
near-mirror sphere reflects it, and a glass prism refracts it through the
nested-IOR stack. The sphere drifts sideways during the second.
"""

import torch

from algan import *

SETTINGS.raytracing.set(samples_per_pixel=8)

# Analytic sky: a vertical gradient with a bright sun square on the -z side
# (the side the default camera sits on; equirect u = 0.25 faces -z).
_H, _W = 32, 64
_v = torch.linspace(0.0, 1.0, _H).view(-1, 1, 1)
_env = 0.08 + 0.35 * (1.0 - _v) * torch.tensor([0.4, 0.6, 1.0]).view(1, 1, 3)
_env = _env.expand(_H, _W, 3).clone()
_env[10:18, 12:20] = 5.0
Scene.set_environment_map(_env)
Scene.clear_light_sources()

with Off():
    floor = Prism(dimensions=(8.0, 2.5, 0.2))
    floor.set_material(MeshLambertMaterial(color=WHITE))
    floor.move(DOWN * 1.6)
    floor.spawn(animate=False)

    mirror = Sphere(radius=0.9)
    mirror.set_material(
        MeshStandardMaterial(color=WHITE, metalness=1.0, roughness=0.06)
    )
    mirror.move(LEFT * 1.7 + UP * 0.2)
    mirror.spawn(animate=False)

    glass = Prism(dimensions=(1.2, 1.2, 1.2))
    glass.set_material(
        MeshPhysicalMaterial(color=WHITE, transmission=1.0, ior=1.5, roughness=0.0)
    )
    glass.move(RIGHT * 1.7 + UP * 0.2)
    glass.spawn(animate=False)

mirror.move(RIGHT * 0.8)
