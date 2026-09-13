"""A general 3-D graphics workload: what a Three.js scene asks of a renderer.

Where ``explainer_scene.py`` is flat, unlit and full of small circuits, this
is the other kind of scene Algan renders -- lit, shadowed, reflective solids
under a moving camera:

* **an environment map** as backdrop and image-based light, so the metals
  have something to reflect;
* **a lighting rig** -- a directional key, a point fill, a spot and a
  rectangular area light (soft shadows), with ray-traced shadows on;
* **materials** -- glass (refraction), chrome and brushed metal (glossy
  reflection), Lambert, PBR with roughness, and a textured parametric
  ``Surface``, on a polished ground slab that reflects the lot;
* **geometry families** -- PN-diced ``Surface`` solids (spheres, torus,
  cylinder, cone), flat ``Polyhedron`` meshes (a skyline of prisms), and an
  imported textured glTF ``Model3D``;
* **motion** -- the camera turns about the scene through the whole clip at
  constant speed, so every frame re-projects everything, while the model
  spins, the spheres bob and the torus rolls.

There is no text. Shading, shadow rays and reflection/refraction
continuations are the cost here, with the per-frame PN dice and the
re-projection under the moving camera on the preparation side.

The storyboard is written as fractions of the clip length so the same scene
measures at any duration::

    python benchmarks/performance/graphics_scene.py --quality PREVIEW --seconds 6
    python benchmarks/performance/graphics_scene.py --quality UHD --seconds 0.5

Two profiled runs; read RUN 2 (`agent_guidance/gpu_harnesses.md`).
"""

from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _profile_cli  # noqa: E402

from algan import *  # noqa: E402

ENV_MAP = os.path.join(HERE, "world_map.png")
MODEL = os.path.join(
    os.path.dirname(os.path.dirname(HERE)),
    "tests",
    "full_renders",
    "assets",
    "textured_icosphere.glb",
)


def ripple(uv):
    """A gently rippled sheet: curvature everywhere, so the dice has work."""
    u = uv[..., :1] * 2 - 1
    v = uv[..., 1:] * 2 - 1
    r2 = u * u + v * v
    return u * 1.2 * RIGHT + v * 1.2 * OUT + torch.cos(r2 * 4.0) * 0.18 * UP


def scene(seconds: float):
    """Record the graphics clip; ``seconds`` is its whole authored length."""
    SETTINGS.raytracing.set(shadows=True)
    Scene.set_background(DARKER_GRAY)
    Scene.set_environment_map(ENV_MAP, intensity=0.8, ambient=True)

    def part(fraction):
        return seconds * fraction

    with Off():
        # The rig. The default point light is kept as the fill; the area
        # light's 3x3 cells plus the three others stay inside the 16-slot
        # shadow cap, so nothing is silently unshadowed.
        DirectionalLight(
            location=RIGHT * 6 + UP * 8 + OUT * 5,
            target=ORIGIN,
            color=WHITE,
            intensity=0.9,
        ).spawn(animate=False)
        SpotLight(
            location=LEFT * 4 + UP * 6 + IN * 2,
            target=LEFT * 1.5 + DOWN * 1.0,
            color=ORANGE,
            intensity=1.4,
            cone_angle=35.0,
            penumbra=0.3,
        ).spawn(animate=False)
        RectAreaLight(
            location=UP * 5.5 + IN * 3,
            target=ORIGIN,
            width=3.0,
            height=2.0,
            samples=3,
            color=BLUE_A,
            intensity=2.0,
        ).spawn(animate=False)

        # A polished slab for everything to stand on and reflect in.
        ground = Prism(width=14, height=0.2, depth=14, color=GRAY_D).set_material(
            MeshStandardMaterial(metalness=0.55, roughness=0.18)
        )
        ground.move(DOWN * 1.5)

        # Hero row: glass, chrome, brushed metal, and the imported model.
        glass = Sphere(radius=0.75, color=WHITE).set_material(
            MeshPhysicalMaterial(transmission=1.0, ior=1.5, roughness=0.0)
        )
        glass.move(LEFT * 2.4 + DOWN * 0.65)
        chrome = Sphere(radius=0.7, color=GRAY_A).set_material(
            MeshStandardMaterial(metalness=1.0, roughness=0.05)
        )
        chrome.move(LEFT * 0.6 + DOWN * 0.7)
        torus = Torus(ring_radius=0.75, tube_radius=0.26, color=GOLD).set_material(
            MeshStandardMaterial(metalness=1.0, roughness=0.35)
        )
        torus.move(RIGHT * 1.4 + DOWN * 0.5)
        model = Model3D(MODEL, fit_to_size=1.7)
        model.move(RIGHT * 3.3 + DOWN * 0.55)

        # Back row: PN solids under ordinary materials, and a textured sheet.
        pillar = Cylinder(radius=0.45, height=1.6, closed=True).set_material(
            MeshLambertMaterial(color=GREEN)
        )
        pillar.move(LEFT * 3.4 + DOWN * 0.6 + IN * 2.4)
        cone = Cone(radius=0.6, height=1.3, closed=True).set_material(
            MeshStandardMaterial(color=RED, roughness=0.6)
        )
        cone.move(LEFT * 1.4 + DOWN * 0.75 + IN * 2.6)
        sheet = Surface(
            ripple,
            color_texture=get_checkerboard((MAROON, YELLOW), 6),
            grid_width=17,
            grid_height=17,
        ).set_material(MeshStandardMaterial(roughness=0.5))
        sheet.move(RIGHT * 1.2 + DOWN * 1.1 + IN * 2.8)
        ball = Sphere(radius=0.5).set_material(
            MeshStandardMaterial(color=BLUE, roughness=0.4, metalness=0.2)
        )
        ball.move(RIGHT * 3.6 + DOWN * 0.9 + IN * 2.4)

        # A skyline of flat meshes at the back: many small triangle groups.
        skyline = Group(
            [
                Prism(width=0.5, height=0.5 + 0.35 * ((k * 7) % 5), depth=0.5)
                .set_material(
                    MeshStandardMaterial(
                        color=mix(BLUE_E, TEAL, (k % 4) / 3.0), roughness=0.7
                    )
                )
                .move(RIGHT * (0.7 * k - 3.85) + IN * 4.6)
                for k in range(12)
            ]
        )
        for tower in skyline:
            # Stand each on the slab regardless of its height.
            tower.move(DOWN * (1.4 - float(tower.get_height()) * 0.5))

        # A few floating flat solids in front, for silhouettes over the glass.
        gems = Group(
            [
                Icosahedron(edge_length=0.35).set_material(
                    MeshStandardMaterial(color=PURPLE_A, roughness=0.3, metalness=0.5)
                ),
                Octahedron(edge_length=0.4).set_material(
                    MeshLambertMaterial(color=YELLOW)
                ),
                Cube(size=0.4, opacity=0.5).set_material(
                    MeshLambertMaterial(color=TEAL_A)
                ),
            ]
        )
        for k, gem in enumerate(gems):
            gem.move(LEFT * 1.0 + RIGHT * 1.9 * k + UP * 1.3 + OUT * 1.2)

        camera = Scene.get_camera()
        camera.move(UP * 1.6 + OUT * 2.5).look_at(DOWN * 0.4)

    with Sync(runtime=part(0.06)):
        ground.spawn()
        skyline.spawn()
        pillar.spawn()
        cone.spawn()
        sheet.spawn()
        ball.spawn()
    with Lag(0.2, runtime=part(0.10)):
        glass.spawn()
        chrome.spawn()
        torus.spawn()
        model.spawn()
        gems.spawn()

    # The rest of the clip: a constant-speed camera turn while the actors
    # move. ``Sync`` blocks inside a linear ``Seq`` keep every sub-motion on
    # its own easing.
    remaining = part(0.84)
    with Sync(runtime=remaining, easing=easings.identity):
        camera.rotate(75, UP, about=ORIGIN)
        model.rotate(360, UP)
        torus.rotate(180, RIGHT)
        skyline.rotate(35, UP, about=IN * 4.6)
    # Recorded in parallel with the block above: it opens at the same
    # authoring time because Sync advanced the clock only once.
    with Seq(runtime=remaining * 0.5, easing=easings.ease_in_out_sine):
        glass.move(UP * 0.9)
        glass.move(DOWN * 0.9)
    with Seq(runtime=remaining * 0.5, easing=easings.ease_in_out_sine):
        chrome.move(UP * 0.6 + LEFT * 0.3)
        chrome.move(DOWN * 0.6 + RIGHT * 0.3)


def mix(a, b, t):
    """Linear colour blend; ``Color`` is a tensor, so this stays a Color."""
    return a * (1.0 - t) + b * t


if __name__ == "__main__":
    _profile_cli.run(scene, "graphics", default_seconds=6.0)
