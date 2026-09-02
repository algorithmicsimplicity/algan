from algan.constants.spatial import ORIGIN, UP, RIGHT, OUTWARD
from algan.constants.color import WHITE
from algan.manim_defaults import MANIM_FOCAL_DISTANCE, manim_fov
from algan.rendering.camera import Camera
from algan.rendering.lights import PointLight

def default_scene_initializer(scene):
    scene.camera = Camera(scene=scene)
    scene_camera = scene.get_camera()
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
