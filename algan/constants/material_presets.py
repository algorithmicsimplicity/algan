"""Ready-to-use PBR material presets.

The presets are reusable material configuration objects intended for
``Mob.set_material``::

    from algan import *

    table = Prism().set_material(WOOD).spawn()
    window = Prism(color=BLUE_A).set_material(GLASS).spawn()

Presets with ``color=None`` preserve the Mob's existing color.  Presets named
for a naturally colored substance (for example ``WOOD`` and ``COPPER``)
provide a representative base color.  These presets describe surface response
and, where appropriate, a flat base color; they do not add texture maps.
"""

from __future__ import annotations

from algan.rendering.shaders.materials import (
    MeshPhysicalMaterial,
    MeshStandardMaterial,
)

__all__ = [
    "WOOD",
    "GLASS",
    "PLASTIC",
    "RUBBER",
    "CERAMIC",
    "STONE",
    "MIRROR",
    "BRUSHED_METAL",
    "CHROME",
    "COPPER",
]


# Natural, diffuse surfaces.
WOOD = MeshStandardMaterial(color=0x8B5A2B, roughness=0.75, metalness=0.0)
RUBBER = MeshStandardMaterial(roughness=0.9, metalness=0.0)
STONE = MeshStandardMaterial(color=0x8A8175, roughness=0.95, metalness=0.0)

# Dielectrics.  Their unset color lets an authored Mob color tint the preset.
GLASS = MeshPhysicalMaterial(
    roughness=0.05,
    metalness=0.0,
    transmission=1.0,
    ior=1.5,
)
PLASTIC = MeshPhysicalMaterial(
    roughness=0.35,
    metalness=0.0,
    clearcoat=0.25,
    clearcoat_roughness=0.25,
)
CERAMIC = MeshPhysicalMaterial(
    roughness=0.25,
    metalness=0.0,
    clearcoat=0.1,
    clearcoat_roughness=0.15,
)

# Conductors.
MIRROR = MeshStandardMaterial(color=0xFFFFFF, roughness=0.0, metalness=1.0)
BRUSHED_METAL = MeshStandardMaterial(roughness=0.35, metalness=1.0)
CHROME = MeshStandardMaterial(color=0xD9D9D9, roughness=0.08, metalness=1.0)
COPPER = MeshStandardMaterial(color=0xB87333, roughness=0.22, metalness=1.0)
