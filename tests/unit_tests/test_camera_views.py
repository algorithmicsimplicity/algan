"""Live captures, timeline synchronization and render-state preservation."""

from __future__ import annotations

import contextlib

import pytest
import torch

import algan
from algan import CameraView, Circle, Off, Scene, Square, Sync
from algan.rendering.camera_views import _render_pass, _texture_from_capture

pytestmark = pytest.mark.usefixtures("fresh_scene")


def _frames(scene, indices):
    with contextlib.closing(
        scene.get_frames(
            0,
            len(indices),
            frame_indices=indices,
            post_processes=(),
        )
    ) as stream:
        return torch.cat(list(stream))


def _scene():
    return Scene(
        algan.PREVIEW.set(resolution=(96, 64), frames_per_second=8),
        background=algan.BLACK,
    )


@pytest.mark.fast
def test_camera_view_is_independent_and_scene_owned():
    scene = _scene()
    with Off():
        scene.camera.rotate(30, algan.UP, about=algan.ORIGIN)
        target = Circle(radius=0.3, scene=scene)
        view = CameraView(scene=scene, resolution=(60, 40))
        torch.testing.assert_close(
            view.camera.screen.location, scene.camera.screen.location
        )
        torch.testing.assert_close(view.camera.screen.basis, scene.camera.screen.basis)
        view.focus_on(target)
    main_location = scene.camera.location.clone()
    camera_location = view.camera.location.clone()
    with Off():
        view.move(algan.RIGHT)
    torch.testing.assert_close(scene.camera.location, main_location)
    torch.testing.assert_close(view.camera.location, camera_location)
    assert view.camera.scene is scene
    assert view.camera not in view.get_descendants()
    assert view.camera is not scene.camera
    assert "CameraView" in algan.__all__


@pytest.mark.parametrize(
    "resolution", [(0, 40), (40, -1), (1.5, 30), (True, 30), (40,), None]
)
def test_invalid_capture_resolution(resolution):
    with pytest.raises(algan.AlganConfigurationError, match="resolution"):
        CameraView(resolution=resolution)


def test_camera_and_exclusions_cannot_cross_scenes():
    one = _scene()
    two = _scene()
    with pytest.raises(algan.AlganConfigurationError, match="same Scene"):
        CameraView(one.camera, scene=two)
    with pytest.raises(algan.AlganConfigurationError, match="same Scene"):
        CameraView(scene=two, exclude=[Square(scene=one)])


def test_render_pass_does_not_replace_authored_scene():
    scene = _scene()
    view = CameraView(scene=scene)
    camera, settings, actors = scene.camera, scene.video_settings, scene.actors
    copy = _render_pass(scene, view.camera, [], (30, 20))
    assert copy is not scene
    assert copy.timeline_manager is scene.timeline_manager
    assert scene.camera is camera
    assert scene.video_settings is settings
    assert scene.actors is actors
    assert copy.video_settings.resolution == (30, 20)


def test_capture_retains_hdr_and_orientation():
    from algan.utils.color_space import srgb_to_linear

    frames = torch.tensor(
        [[[[4.0, 0.2, 0.1, 0.5, 128.0]], [[0.1, 0.5, 0.8, 0.0, 255.0]]]]
    )
    texture = _texture_from_capture(frames)
    decoded = (
        srgb_to_linear(texture[..., :3])
        if algan.SETTINGS.raytracing.linear_color_space
        else texture[..., :3]
    )
    straight = frames[..., :3] / (frames[..., 4:] / 255)
    torch.testing.assert_close(decoded, straight.transpose(-3, -2).flip(-2))
    assert texture[0, 0, 1, 3] == pytest.approx(0.5 * 255 / 128)
    assert texture[0, 0, 1, 4] == pytest.approx(128 / 255)


def test_live_render_tracks_animation_and_restores_state():
    scene = _scene()
    with Off():
        target = Square(size=1, color=algan.RED, scene=scene).spawn()
        view = CameraView(scene=scene, resolution=(48, 32), height=3)
        view.focus_on(target, buffer_portion=0.5)
        view.move_to(algan.RIGHT * 3 + algan.OUT)
        view.spawn()
    with Sync():
        target.color = algan.BLUE
        target.move(algan.LEFT)
        view.camera.move(algan.LEFT)
    camera, settings = scene.camera, scene.video_settings
    position = target.location.clone()
    frames = _frames(scene, (0, 4, 8))
    assert frames.shape == (3, 64, 96, 3)
    assert frames[0, :, 60:, 0].max() > 180
    assert frames[2, :, 60:, 2].max() > 180
    assert not torch.equal(frames[0], frames[2])
    assert scene.camera is camera
    assert scene.video_settings is settings
    torch.testing.assert_close(target.location, position)
    again = _frames(scene, (4,))
    assert (again[0].int() - frames[1].int()).abs().max() <= 2


def test_two_views_capture_geometry_outside_the_main_camera():
    scene = _scene()
    with Off():
        left = Square(size=1, color=algan.RED).move_to(algan.LEFT * 9).spawn()
        right = Square(size=1, color=algan.BLUE).move_to(algan.RIGHT * 9).spawn()
        a = CameraView(resolution=(48, 32), height=2).focus_on(left, 0.3)
        b = CameraView(resolution=(48, 32), height=2).focus_on(right, 0.3)
        a.move_to(algan.LEFT * 2.5 + algan.OUT).spawn()
        b.move_to(algan.RIGHT * 2.5 + algan.OUT).spawn()
    frame = _frames(scene, (0,))[0]
    assert frame[32, 27, 0] > 180
    assert frame[32, 27, 0] > frame[32, 27, 2]
    assert frame[32, 69, 2] > 180
    assert frame[32, 69, 2] > frame[32, 69, 0]
    # Hiding one subject in its own view does not affect the other camera.
    a.capture_exclude = (left,)
    frame = _frames(scene, (0,))[0]
    assert frame[32, 27].max() == 0
    assert frame[32, 69, 2] > 180
    b.camera.set_far(0.1)
    frame = _frames(scene, (0,))[0]
    assert frame.max() == 0


@pytest.mark.parametrize("legacy_tonemap", [False, True])
def test_empty_capture_composes_background_without_double_exposure(legacy_tonemap):
    scene = _scene()
    scene.set_background(algan.Color((0.3, 0.2, 0.4)))
    algan.SETTINGS.raytracing.tonemap_exposure = 1.7
    algan.SETTINGS.raytracing.tonemapping = True
    algan.SETTINGS.raytracing.experimental.post_process_tonemap = not legacy_tonemap
    if legacy_tonemap:
        # The byte-buffer route does not support linear color-space rendering.
        algan.SETTINGS.raytracing.linear_color_space = False
        algan.SETTINGS.raytracing.tonemapping = False
        algan.SETTINGS.raytracing.tonemap_exposure = 1
    with Off():
        CameraView(resolution=(24, 16), height=3).move(algan.OUT).spawn()
    frame = _frames(scene, (0,))[0]
    assert (frame[32, 48].int() - frame[5, 5].int()).abs().max() <= 2
    assert (
        algan.SETTINGS.raytracing.experimental.post_process_tonemap
        is not legacy_tonemap
    )


def test_transparent_capture_and_display_opacity():
    scene = _scene()
    scene.set_background(algan.TRANSPARENT)
    with Off():
        target = Square(size=1, color=algan.RED).spawn()
        target.opacity = 0.5
        view = CameraView(resolution=(32, 32), height=2).focus_on(target, 0.3)
        view.move_to(algan.RIGHT * 3 + algan.OUT).spawn()
        view.opacity = 0.5
    frame = _frames(scene, (0,))[0]
    source, inset = frame[32, 48], frame[32, 73]
    assert source[3] == pytest.approx(128, abs=2)
    assert inset[3] == pytest.approx(64, abs=2)
    # Frame export is premultiplied; the display's half opacity halves RGB too.
    assert (source[:3].float() * 0.5 - inset[:3].float()).abs().max() <= 2


def test_compatibility_camera_display_is_live():
    import algan.manim as mn
    from algan.rendering.camera import Camera

    scene = _scene()
    with Off():
        target = Square(size=1, color=algan.BLUE).move_to(algan.RIGHT * 9).spawn()
        camera = Camera(scene=scene)
        camera.center_on(target, buffer_portion=0.5)
        display = mn.ImageMobjectFromCamera(camera, resolution=(48, 32))
        display.add_display_frame()
        display.spawn()
    frames = _frames(scene, (0,))
    assert frames[0, 32, 48, 2] > 180
    with Off():
        display.set_opacity(0.0)
    assert display.opacity.max() == 0


def test_batch_splitting_and_early_close_restore_authoring(monkeypatch):
    from algan.rendering import camera_views

    scene = _scene()
    with Off():
        target = Square(size=1, color=algan.RED).spawn()
        view = CameraView(resolution=(24, 16), height=2).focus_on(target, 0.3)
        view.move_to(algan.RIGHT * 3 + algan.OUT).spawn()
    target.color = algan.BLUE
    expected = _frames(scene, (0, 3, 8))
    monkeypatch.setattr(camera_views, "_window_size", lambda views: 2)
    algan.SETTINGS.computing.max_animation_batch_size = 1
    actual = _frames(scene, (0, 3, 8))
    assert (actual.int() - expected.int()).abs().max() <= 2
    position = view.location.clone()
    with contextlib.closing(scene.get_frames(0, 9, post_processes=())) as stream:
        next(stream)
    assert scene.memory is None
    assert not hasattr(scene, "_geometry_view")
    torch.testing.assert_close(view.location, position)


def test_capture_geometry_uses_its_resolution_and_eye():
    from algan.mobs.nonplanar_circuit import camera_eye
    from algan.rendering.camera_views import _geometry_view

    scene = _scene()
    shape = Square(stroke_width=4, filled=False)
    view = CameraView(resolution=(48, 32))
    with Off():
        view.camera.move(algan.RIGHT)
    original = shape.get_render_primitives().stroke_width.clone()
    capture = _render_pass(scene, view.camera, scene.actors, view.capture_resolution)
    with _geometry_view(scene, capture):
        width = shape.get_render_primitives().stroke_width
        torch.testing.assert_close(width, original / 2)
        torch.testing.assert_close(camera_eye(shape), view.camera.location)
        assert scene.camera is not view.camera
    torch.testing.assert_close(shape.get_render_primitives().stroke_width, original)
    assert not hasattr(scene, "_geometry_view")
