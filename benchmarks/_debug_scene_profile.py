"""Profile the debug/debug.py materials_and_lighting scene end-to-end.

Runs profile_scene (2 passes: cold+warm) on the full 174-frame scene at
PREVIEW settings and writes the standard profiling report.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algan import *  # noqa: F403
from algan.utils.profiling_utils import profile_scene


def build_scene():
    Scene.set_background(DARKER_GRAY)
    SETTINGS.raytracing.set(shadows=True)

    # ------------------------------------------------------------------
    # Act 1 -- the material zoo, two labelled rows of identical spheres.
    # ------------------------------------------------------------------
    with Off():
        ambient = AmbientLight(color=WHITE, intensity=0.35).spawn(animate=False)
        key_light = DirectionalLight(
            location=RIGHT * 4 + UP * 5 + OUT * 4,
            target=ORIGIN,
            color=WHITE,
            intensity=1.0,
            shadow_angle=0.4,
        ).spawn(animate=False)

        title = Text(
            "MATERIALS AND LIGHTING",
            font_size=42,
            weight="BOLD",
            color=WHITE,
        ).move(UP * 2.9)

        lit = Group(
            Sphere(radius=0.5).set_material(MeshBasicMaterial(color=BLUE)),
            Sphere(radius=0.5).set_material(MeshLambertMaterial(color=GREEN)),
            Sphere(radius=0.5).set_material(
                MeshPhongMaterial(color=ORANGE, specular=WHITE, shininess=80)
            ),
            Sphere(radius=0.5).set_material(
                MeshStandardMaterial(color=RED, roughness=0.2, metalness=0.75)
            ),
            Sphere(radius=0.5).set_material(MeshToonMaterial(color=TEAL, bands=4)),
            Sphere(radius=0.5, color=WHITE).set_material(MeshNormalMaterial()),
        ).arrange_in_line(RIGHT, buffer=0.62)
        lit.move(UP * 1.3 - lit.get_center())

        lit_labels = Group(
            Text("Basic", font_size=21, color=GRAY_A),
            Text("Lambert", font_size=21, color=GRAY_A),
            Text("Phong", font_size=21, color=GRAY_A),
            Text("Standard", font_size=21, color=GRAY_A),
            Text("Toon", font_size=21, color=GRAY_A),
            Text("Normal", font_size=21, color=GRAY_A),
        )
        for mob, label in zip(lit, lit_labels):
            label.move_to(mob.get_center() + DOWN * 0.85)

        exotic = Group(
            Sphere(radius=0.5).set_material(MeshMatcapMaterial(color=GOLD)),
            Sphere(radius=0.5).set_material(MeshDepthMaterial(near=4.0, far=11.0)),
            Sphere(radius=0.5).set_material(
                MeshPhysicalMaterial(
                    color=BLUE_A,
                    roughness=0.1,
                    clearcoat=0.85,
                    transmission=0.5,
                    ior=1.45,
                )
            ),
            Sphere(radius=0.5, color=GREEN_A).set_material(GLASS),
            Sphere(radius=0.5).set_material(MIRROR),
            Sphere(radius=0.5).set_material(COPPER),
        ).arrange_in_line(RIGHT, buffer=0.62)
        exotic.move(DOWN * 0.5 - exotic.get_center())

        exotic_labels = Group(
            Text("Matcap", font_size=21, color=GRAY_A),
            Text("Depth", font_size=21, color=GRAY_A),
            Text("Physical", font_size=21, color=GRAY_A),
            Text("GLASS", font_size=21, color=GRAY_A),
            Text("MIRROR", font_size=21, color=GRAY_A),
            Text("COPPER", font_size=21, color=GRAY_A),
        )
        for mob, label in zip(exotic, exotic_labels):
            label.move_to(mob.get_center() + DOWN * 0.85)

    with Seq():
        title.spawn()
        with Lag(0.14, run_time=1.3):
            for mob in lit:
                mob.spawn()
        with Sync(run_time=0.4):
            lit_labels.spawn()
        with Lag(0.14, run_time=1.3):
            for mob in exotic:
                mob.spawn()
        with Sync(run_time=0.4):
            exotic_labels.spawn()

    # ------------------------------------------------------------------
    # Act 2 -- material parameters are ordinary animatable attributes.
    # ------------------------------------------------------------------
    with Seq():
        with Sync(run_time=2.0):
            lit[2].shininess = 12
            lit[3].roughness = 0.85
            lit[3].metalness = 0.15
            exotic[2].clearcoat = 0.1
            exotic[2].transmission = 0.05
            key_light.move(LEFT * 9)
            for mob in lit:
                mob.rotate(150, UP)
            for mob in exotic:
                mob.rotate(-150, UP)
        with Sync(run_time=1.4):
            key_light.move(RIGHT * 9)
            lit[3].roughness = 0.2
            lit[3].metalness = 0.75
        Scene.wait(0.2)

    # ------------------------------------------------------------------
    # Act 3 -- neutral probes in front of a wall.
    # ------------------------------------------------------------------
    with Sync(run_time=0.7):
        lit.despawn()
        lit_labels.despawn()
        exotic.despawn()
        exotic_labels.despawn()
        title.despawn()

    with Off():
        wall = (
            Prism(dimensions=(17.0, 5.2, 0.3))
            .set_material(MeshLambertMaterial(color=GRAY_D))
            .move(IN * 2.4 + DOWN * 0.9)
        )
        probes = Group(
            *[
                Sphere(radius=0.6).set_material(
                    MeshStandardMaterial(color=WHITE, roughness=0.6)
                )
                for _ in range(4)
            ]
        ).arrange_in_line(RIGHT, buffer=1.5)
        probes.move(DOWN * 0.9 - probes.get_center())
        light_label = Text(
            "point / spot / rect-area / hemisphere lights  +  shadows",
            font_size=23,
            color=TEAL_A,
        ).move(DOWN * 3.15)

    with Seq():
        with Off():
            ambient.despawn()
            key_light.despawn()
            point_light = PointLight(
                location=LEFT * 3.6 + UP * 0.6 + OUT * 2.2,
                color=YELLOW,
                intensity=2.2,
                decay=1.0,
            ).spawn(animate=False)
            spot_light = SpotLight(
                location=LEFT * 1.2 + UP * 2.4 + OUT * 2.2,
                target=LEFT * 1.2 + DOWN * 0.9,
                color=BLUE_A,
                intensity=5.0,
                cone_angle=22.0,
                penumbra=0.35,
            ).spawn(animate=False)
            rect_light = RectAreaLight(
                location=RIGHT * 1.2 + UP * 1.8 + OUT * 2.2,
                target=RIGHT * 1.2 + DOWN * 0.9,
                color=GREEN_A,
                intensity=3.0,
                width=1.8,
                height=1.0,
            ).spawn(animate=False)
            hemisphere = HemisphereLight(
                color=MAROON_A,
                ground_color=BLUE_E,
                intensity=0.5,
            ).spawn(animate=False)
        with Sync(run_time=0.7):
            wall.spawn()
            probes.spawn()
            light_label.spawn()
        with Sync(run_time=1.8):
            point_light.move(RIGHT * 2.4)
            spot_light.move(RIGHT * 2.4)
            rect_light.move(RIGHT * 2.4)
        with Sync(run_time=1.4):
            point_light.move(LEFT * 2.4)
            spot_light.move(LEFT * 2.4)
            rect_light.move(LEFT * 2.4)
        Scene.wait(0.2)

    # ------------------------------------------------------------------
    # Act 4 -- emissive glow + bloom, opacity.
    # ------------------------------------------------------------------
    with Sync(run_time=0.7):
        probes.despawn()
        wall.despawn()
        hemisphere.despawn()
        rect_light.despawn()
        spot_light.despawn()
        point_light.despawn()
        light_label.despawn()

    with Off():
        AmbientLight(color=WHITE, intensity=0.45).spawn(animate=False)
        DirectionalLight(
            location=RIGHT * 4 + UP * 5 + OUT * 4,
            target=ORIGIN,
            color=WHITE,
            intensity=1.0,
        ).spawn(animate=False)
        emitters = Group(
            Sphere(radius=0.6).set_material(MeshBasicMaterial(color=YELLOW)),
            Sphere(radius=0.6).set_material(MeshBasicMaterial(color=TEAL)),
            Sphere(radius=0.6).set_material(MeshStandardMaterial(color=RED)),
            Sphere(radius=0.6).set_material(MeshStandardMaterial(color=BLUE)),
        ).arrange_in_line(RIGHT, buffer=1.5)
        emitters.move(-emitters.get_center())
        glow_label = Text(
            "glow + bloom + tonemapping                    opacity",
            font_size=23,
            color=TEAL_A,
        ).move(DOWN * 3.15)

    with Seq():
        with Sync(run_time=0.6):
            emitters.spawn()
            glow_label.spawn()
        with Sync(run_time=1.8):
            emitters[0].glow = 1.0
            emitters[1].glow = 2.5
            emitters[2].opacity = 0.2
            emitters[3].opacity = 0.55
        with Sync(run_time=1.2):
            emitters[0].glow = 0.0
            emitters[1].glow = 0.0
            emitters[2].opacity = 1.0
            emitters[3].opacity = 1.0
        Scene.wait(0.3)


if __name__ == "__main__":
    profile_scene(build_scene, PREVIEW, tag="_debug_scene")
