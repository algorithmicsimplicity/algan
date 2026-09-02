from __future__ import annotations

from algan.animation_timeline.animation_contexts import Off
from algan.constants.color import WHITE
from algan.constants.spatial import ORIGIN, OUTWARD, RIGHT, UP
from algan.manim_defaults import MANIM_FOCAL_DISTANCE, manim_fov
from algan.rendering.camera import Camera
from algan.rendering.lights import PointLight


def default_scene_initializer(scene):
    scene.camera = Camera(scene=scene)
    scene_camera = scene.get_camera()
    with Off(animation_manager=scene_camera.animation_manager):
        scene_camera.move_to(ORIGIN + OUTWARD * MANIM_FOCAL_DISTANCE)
        scene_camera.look_at(ORIGIN)
        scene_camera.set_fov(manim_fov())
        scene_camera.spawn(animate=False)
    scene.light_sources = []
    PointLight(
        scene=scene,
        location=scene.camera.location + UP * 1 + RIGHT * 5 + OUTWARD * 1,
        color=WHITE,
    ).spawn(animate=False)
