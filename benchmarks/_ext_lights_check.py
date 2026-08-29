"""Visual validation of the extended lighting features: directional / ambient /
hemisphere / spot / rect-area lights, falloff, soft (penumbra) shadows,
environment maps (skybox + IBL + reflections) and camera fov/near/far.

Renders one PNG per feature into benchmarks/_tc_out/ via scene.save_frame.
Run: .venv/Scripts/python.exe benchmarks/_ext_lights_check.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algan import (  # noqa: E402
    BLUE,
    DOWN,
    GRAY,
    GREEN,
    IN,
    LEFT,
    ORIGIN,
    OUT,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    AmbientLight,
    DirectionalLight,
    HemisphereLight,
    MeshLambertMaterial,
    MeshStandardMaterial,
    Off,
    PointLight,
    RectAreaLight,
    Scene,
    SceneManager,
    Sphere,
    SpotLight,
)
from algan.mobs.shapes_2d import QuadTriangulated  # noqa: E402
from algan.rendering.raytracing.primitives import set_reflectivity  # noqa: E402
from algan.rendering.raytracing.settings import set_shadows  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)


def ground(y=-1.5, half=8.0):
    corners = torch.tensor(
        ((-half, y, -half), (half, y, -half), (half, y, half), (-half, y, half))
    ).float()
    return QuadTriangulated(corners, color=GRAY)


def tilt_camera(deg=-25, dolly=0.0):
    cam = Scene.get_camera()
    with Off():
        cam.rotate(deg, RIGHT, about=ORIGIN)
        if dolly:
            cam.move(cam.get_forward_direction() * dolly)


def save(tag):
    scene = SceneManager.instance()
    path = os.path.join(OUT_DIR, f"extl_{tag}.png")
    scene.save_frame(path)
    print("saved", path)


def fresh():
    SceneManager.reset()
    set_shadows(False)


def scene_directional_soft():
    """Directional 'sun' + dim ambient; soft-edged shadows on the ground."""
    fresh()
    set_shadows(True)
    scene = SceneManager.instance()
    scene.light_sources = [
        DirectionalLight(
            location=UP * 10 + RIGHT * 6 + OUT * 3,
            target=ORIGIN,
            color=WHITE,
            intensity=1.0,
            shadow_angle=6.0,
        ).spawn(animate=False),
        AmbientLight(color=WHITE, intensity=0.6).spawn(animate=False),
    ]
    with Off():
        ground().spawn(animate=False)
        (
            Sphere()
            .scale(1.0)
            .move(DOWN * 0.5)
            .set_material(MeshLambertMaterial(color=RED))
            .spawn(animate=False)
        )
        (
            Sphere()
            .scale(0.7)
            .move(LEFT * 2.5 + DOWN * 0.8)
            .set_material(MeshLambertMaterial(color=BLUE))
            .spawn(animate=False)
        )
    tilt_camera()
    save("directional_soft")


def scene_spot():
    """Spot light pool with a soft penumbra edge and inverse-square decay."""
    fresh()
    set_shadows(True)
    scene = SceneManager.instance()
    scene.light_sources = [
        SpotLight(
            location=UP * 6 + OUT * 2,
            target=DOWN * 1.5,
            color=WHITE,
            intensity=40.0,
            angle=22.0,
            penumbra=0.6,
            decay=2.0,
        ).spawn(animate=False),
        AmbientLight(color=WHITE, intensity=0.35).spawn(animate=False),
    ]
    with Off():
        ground().spawn(animate=False)
        (
            Sphere()
            .scale(0.8)
            .move(DOWN * 0.7)
            .set_material(MeshLambertMaterial(color=GREEN))
            .spawn(animate=False)
        )
    tilt_camera()
    save("spot")


def scene_hemisphere():
    """Sky-blue from above, warm orange from below, no point lights."""
    fresh()
    scene = SceneManager.instance()
    scene.light_sources = [
        HemisphereLight(color=BLUE, ground_color=(1.0, 0.45, 0.1), intensity=1.0).spawn(
            animate=False
        ),
    ]
    with Off():
        (
            Sphere()
            .scale(1.4)
            .set_material(MeshLambertMaterial(color=WHITE))
            .spawn(animate=False)
        )
    save("hemisphere")


def scene_area():
    """Rect area light overhead: smooth lighting + a soft contact shadow."""
    fresh()
    set_shadows(True)
    scene = SceneManager.instance()
    scene.light_sources = [
        RectAreaLight(
            location=UP * 4 + RIGHT * 2,
            target=ORIGIN,
            width=4.0,
            height=4.0,
            samples=16,
            color=WHITE,
            intensity=1.1,
        ).spawn(animate=False),
        AmbientLight(color=WHITE, intensity=0.35).spawn(animate=False),
    ]
    with Off():
        ground().spawn(animate=False)
        (
            Sphere()
            .scale(0.9)
            .move(DOWN * 0.6)
            .set_material(MeshLambertMaterial(color=YELLOW))
            .spawn(animate=False)
        )
    tilt_camera()
    save("area")


def scene_env():
    """Environment map: skybox + diffuse IBL + a mirror sphere reflecting it."""
    fresh()
    scene = SceneManager.instance()
    scene.light_sources = [
        PointLight(location=UP * 3 + RIGHT * 5 + OUT * 3, color=WHITE).spawn(
            animate=False
        ),
    ]
    Scene.set_environment_map(
        os.path.join(os.path.dirname(__file__), "..", "world_map.jpg"),
        intensity=1.0,
        ambient=True,
    )
    with Off():
        mirror = Sphere().scale(1.2).move(LEFT * 1.5)
        set_reflectivity(mirror, 0.85)
        mirror.spawn(animate=False)
        (
            Sphere()
            .scale(0.9)
            .move(RIGHT * 1.8)
            .set_material(
                MeshStandardMaterial(color=WHITE, roughness=0.6, metalness=0.0)
            )
            .spawn(animate=False)
        )
    save("env")
    Scene.set_environment_map(None)


def scene_fov():
    """fov 15 (telephoto) vs 90 (wide) on the same sphere row."""
    for fov in (15, 90):
        fresh()
        with Off():
            for i, c in enumerate((RED, GREEN, BLUE)):
                (
                    Sphere()
                    .scale(0.6)
                    .move(RIGHT * (i - 1) * 1.8)
                    .set_material(MeshLambertMaterial(color=c))
                    .spawn(animate=False)
                )
            Scene.get_camera().set_fov(fov)
        save(f"fov{fov}")


def scene_near_far():
    """Near clips the closest sphere; far clips the farthest."""
    fresh()
    cam = Scene.get_camera()
    cam.set_near(4.0)  # camera sits at OUT*7; sphere at OUT*4 is 3 away
    cam.set_far(9.0)  # sphere at IN*3 is 10 away
    with Off():
        for pos, c in ((OUT * 4, RED), (ORIGIN, GREEN), (IN * 3 + UP, BLUE)):
            (
                Sphere()
                .scale(0.6)
                .move(pos + RIGHT * pos[..., 2])
                .set_material(MeshLambertMaterial(color=c))
                .spawn(animate=False)
            )
    save("near_far")
    cam.set_near(0.0)
    cam.set_far(0.0)


def main():
    scene_directional_soft()
    scene_spot()
    scene_hemisphere()
    scene_area()
    scene_env()
    scene_fov()
    scene_near_far()
    print("ALL SCENES RENDERED")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
