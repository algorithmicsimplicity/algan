"""In-process settings-arm decomposition of the debug scene's heavy window.

Authors Acts 1-2 of debug/debug.py (frames 0..~80, the expensive material-zoo
window) and renders it once per arm with different runtime raytracing settings,
timing each render. All arms share one process (warm kernels, warm GPU), so
relative differences are meaningful. Arms whose settings force new kernel
variants pay a one-time compile on first use; run the script twice and use the
second run's numbers if in doubt.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import algan.rendering.raytracing.settings as rt_settings
from algan import *  # noqa: F403
from algan.scene_manager import SceneManager

OUT_DIR = os.path.join("algan_outputs", "profiling")


def build_acts_1_2():
    Scene.set_background(DARKER_GRAY)
    SETTINGS.raytracing.set(shadows=True)

    with Off():
        AmbientLight(color=WHITE, intensity=0.35).spawn(animate=False)
        key_light = DirectionalLight(
            location=RIGHT * 4 + UP * 5 + OUT * 4,
            target=ORIGIN,
            color=WHITE,
            intensity=1.0,
            shadow_angle=0.4,
        ).spawn(animate=False)

        title = Text(
            "MATERIALS AND LIGHTING", font_size=42, weight="BOLD", color=WHITE
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


def run_arm(name, setup):
    scene = SceneManager.reset()
    scene.set_video_settings(PREVIEW)
    build_acts_1_2()
    setup()
    t0 = time.perf_counter()
    Scene.save_video(os.path.join(OUT_DIR, f"arms_{name}.mp4"), PREVIEW, overwrite=True)
    dt = time.perf_counter() - t0
    print(f"ARM {name:>28s}: {dt:8.2f}s", flush=True)
    return dt


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    arms = [
        ("baseline_warmup", lambda: None),
        ("baseline", lambda: None),
        ("shadows_off", lambda: SETTINGS.raytracing.set(shadows=False)),
        ("secondary_1", lambda: rt_settings.set_analytic_aa(True, secondary=1)),
        (
            "secondary_1_shadows_off",
            lambda: (
                SETTINGS.raytracing.set(shadows=False),
                rt_settings.set_analytic_aa(True, secondary=1),
            ),
        ),
        ("bounces_2", lambda: SETTINGS.raytracing.set(max_bounces=2)),
        ("baseline_again", lambda: None),
    ]
    for name, setup in arms:
        # Reset the arm-scoped knobs before each arm.
        SETTINGS.raytracing.set(shadows=True, max_bounces=8)
        rt_settings.set_analytic_aa(True, secondary=4)
        run_arm(name, setup)


if __name__ == "__main__":
    main()
