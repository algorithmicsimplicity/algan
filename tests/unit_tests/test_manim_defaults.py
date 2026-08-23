"""Tests for ``Scene.use_manim_defaults()`` and the Manim coordinate conversion."""

from __future__ import annotations

import math

import pytest
import torch

from algan import SETTINGS, Scene
from algan.manim_defaults import (
    MANIM_FOCAL_DISTANCE,
    MANIM_FRAME_HEIGHT,
    MANIM_LIGHT_SOURCE,
    from_manim_coordinates,
    manim_fov,
    to_manim_coordinates,
)


@pytest.fixture
def scene():
    """A fresh Scene, with the process-global style settings restored after."""
    style = SETTINGS.style
    saved_style = (style.default_material, style.background_color.clone())
    saved_tonemapping = SETTINGS.raytracing.tonemapping
    created = Scene()
    try:
        yield created
    finally:
        SETTINGS.style.set(
            default_material=saved_style[0], background_color=saved_style[1]
        )
        SETTINGS.raytracing.set(tonemapping=saved_tonemapping)


def test_manim_fov_matches_manims_projection():
    # Manim's ThreeDCamera scales a point by focal_distance / (focal_distance - z),
    # which is a pinhole eye focal_distance from the frame plane. The frame is
    # frame_height tall there, so that is the angle the camera spans.
    assert manim_fov() == pytest.approx(22.619864948, abs=1e-6)
    assert manim_fov() == pytest.approx(
        math.degrees(2 * math.atan((MANIM_FRAME_HEIGHT / 2) / MANIM_FOCAL_DISTANCE))
    )


def test_manim_fov_frames_eight_units_at_the_origin_plane():
    # The whole point of the angle: at the distance the camera sits from the
    # origin, exactly Manim's 8 units of height must be in view.
    half_height = MANIM_FOCAL_DISTANCE * math.tan(math.radians(manim_fov()) / 2)
    assert half_height == pytest.approx(MANIM_FRAME_HEIGHT / 2)


def test_coordinate_conversion_mirrors_z_only():
    converted = from_manim_coordinates((1.0, 2.0, 3.0))
    assert converted.flatten().tolist() == [1.0, 2.0, -3.0]


def test_coordinate_conversion_is_its_own_inverse():
    points = torch.tensor([[1.0, -2.0, 3.0], [0.0, 4.0, -5.0]])
    round_tripped = to_manim_coordinates(from_manim_coordinates(points))
    assert torch.equal(round_tripped, points)


def test_coordinate_conversion_preserves_shape():
    points = torch.randn(2, 5, 4, 3)
    assert from_manim_coordinates(points).shape == points.shape


def test_use_manim_defaults_positions_the_camera(scene):
    scene.use_manim_defaults()
    camera = scene.get_camera()
    # Manim's eye sits focal_distance from the frame plane; mirrored into Algan's
    # -z-faces-the-viewer convention that is (0, 0, -20).
    assert camera.location.flatten().tolist() == pytest.approx(
        [0.0, 0.0, -MANIM_FOCAL_DISTANCE]
    )
    assert camera.get_fov() == pytest.approx(manim_fov(), abs=1e-4)
    assert not bool(camera.orthographic)


def test_use_manim_defaults_installs_manims_light(scene):
    scene.use_manim_defaults()
    lights = scene.get_light_sources()
    assert len(lights) == 1
    assert lights[0].location.flatten().tolist() == pytest.approx(
        from_manim_coordinates(MANIM_LIGHT_SOURCE).flatten().tolist()
    )


def test_use_manim_defaults_turns_off_tonemapping(scene):
    # Manim writes its colours out untouched. Algan's tonemap darkens every fill
    # by about 10/255, which reads as a colour error rather than a roll-off.
    SETTINGS.raytracing.set(tonemapping=True)
    scene.use_manim_defaults()
    assert SETTINGS.raytracing.tonemapping is False


def test_use_manim_defaults_installs_the_manim_material(scene):
    from algan.rendering.shaders.material_shaders import manim_shader
    from algan.rendering.shaders.materials import ManimMaterial

    scene.use_manim_defaults()
    # The default material is Manim's 3-D shading, not merely an unlit
    # passthrough: Manim shades 3-D geometry via get_shaded_rgb, and this is
    # what makes a material-less imported 3-D mob do the same.
    assert isinstance(SETTINGS.style.default_material, ManimMaterial)
    assert SETTINGS.style.default_material.shader is manim_shader


def test_use_manim_defaults_sets_the_coordinate_convention(scene):
    assert scene.manim_coordinates is False
    scene.use_manim_defaults()
    assert scene.manim_coordinates is True


def test_use_manim_defaults_flags_are_independent(scene):
    camera_before = scene.get_camera().location.clone()
    scene.use_manim_defaults(camera=False, shading=False, coordinates=False)
    assert torch.equal(scene.get_camera().location, camera_before)
    assert scene.manim_coordinates is False
    # Lights belong to the shading group, so they are untouched too.
    assert len(scene.get_light_sources()) == 1


def test_use_manim_defaults_leaves_video_settings_alone_by_default(scene):
    before = scene.video_settings.resolution
    scene.use_manim_defaults()
    assert scene.video_settings.resolution == before


def test_use_manim_defaults_can_set_manims_video_settings(scene):
    scene.use_manim_defaults(video_settings=True)
    assert scene.video_settings.resolution == (1920, 1080)
    assert scene.video_settings.frames_per_second == 60


def test_use_manim_defaults_returns_the_scene(scene):
    assert scene.use_manim_defaults() is scene


def test_manim_mob_mirrors_z_only_under_the_convention(scene):
    manim = pytest.importorskip("manim")

    depth = 1.5
    source = manim.Square(side_length=2).shift(manim.OUT * depth)

    scene.manim_coordinates = False
    plain = manim.Square(side_length=2).shift(manim.OUT * depth)
    unmirrored = ManimMobFor(plain, scene)
    assert unmirrored[..., 2].flatten()[0].item() == pytest.approx(depth)

    scene.manim_coordinates = True
    mirrored = ManimMobFor(source, scene)
    # Manim's OUT is +z and Algan's is -z, so what Manim puts nearer its camera
    # has to land nearer Algan's, which is -z.
    assert mirrored[..., 2].flatten()[0].item() == pytest.approx(-depth)
    # x and y are shared between the two conventions and must not move.
    assert torch.allclose(mirrored[..., :2], unmirrored[..., :2])


def ManimMobFor(manim_mob, scene):
    """Import ``manim_mob`` into ``scene`` and return its control point locations."""
    from algan.mobs.manim_mob import ManimMob

    return ManimMob(manim_mob, scene=scene, add_to_scene=False).control_points.location


def test_reset_scene_drops_the_manim_convention(scene):
    scene.use_manim_defaults()
    assert scene.manim_coordinates is True
    scene.reset_scene()
    # reset_scene() re-runs the initializer, restoring Algan's own camera and
    # lighting, so the coordinate convention that went with the Manim viewpoint
    # has to go with them.
    assert scene.manim_coordinates is False
    assert scene.get_camera().location.flatten().tolist() != pytest.approx(
        [0.0, 0.0, -MANIM_FOCAL_DISTANCE]
    )


def test_clear_keeps_the_manim_convention(scene):
    # clear() only despawns Mobs; it leaves the camera and lights in place, so
    # the convention that matches them must survive it.
    scene.use_manim_defaults()
    scene.clear()
    assert scene.manim_coordinates is True
