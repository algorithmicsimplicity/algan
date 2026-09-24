from __future__ import annotations

import pytest
import torch

from algan import ORIGIN, PREVIEW, Camera, Off, Scene, Seq, easings


def _project_to_screen(camera, points):
    # Intersect viewing rays with the actual screen plane, then apply the
    # basis used by the renderer. Independent of visible_size_at().
    basis = camera._get_render_screen_basis()
    eye = camera.location
    screen = camera.screen.location
    ray = points - eye
    forward = basis[..., 2, :]
    ratio = ((screen - eye) * forward).sum(-1, keepdim=True) / (ray * forward).sum(
        -1, keepdim=True
    )
    offset = eye + ratio * ray - screen
    return torch.stack(
        ((offset * basis[..., 0, :]).sum(-1), (offset * basis[..., 1, :]).sum(-1)),
        -1,
    )


@pytest.mark.parametrize("resolution", [(640, 360), (360, 640)])
@pytest.mark.parametrize("posed", [False, True])
def test_near_orthographic_preserves_origin_plane_frame(resolution, posed):
    scene = Scene(PREVIEW.set(resolution=resolution))
    camera = scene.camera
    with Off():
        if posed:
            camera.move_to((4, 2, 9))
            camera.look_at(torch.tensor([1.0, -1.0, 0.0]))
            camera.set_fov(37)
    points = torch.cat((ORIGIN.view(1, 1, 3), camera.right, camera.up), -2)
    size_before = camera.visible_size_at(ORIGIN).clone()
    projection_before = _project_to_screen(camera, points)
    basis_before = camera.basis.clone()
    with Off():
        assert camera.set_near_orthographic(1000) is camera
    torch.testing.assert_close(camera.visible_size_at(ORIGIN), size_before)
    torch.testing.assert_close(
        _project_to_screen(camera, points), projection_before, atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(camera.basis, basis_before)
    assert (camera.screen.location - camera.location).norm().item() == pytest.approx(
        1000
    )
    assert camera.orthographic


def test_default_near_orthographic_keeps_eight_unit_frame_and_is_repeatable():
    scene = Scene(PREVIEW.set(resolution=(640, 360)))
    camera = scene.camera
    expected = torch.tensor([128 / 9, 8.0])
    torch.testing.assert_close(camera.visible_size_at(ORIGIN).flatten(), expected)
    for distance in (1e5, 1e5, 1000):
        with Off():
            camera.set_near_orthographic(distance)
        torch.testing.assert_close(camera.visible_size_at(ORIGIN).flatten(), expected)


def test_near_orthographic_preserves_framing_through_animation_and_seeking():
    scene = Scene()
    camera = scene.camera
    before = camera.visible_size_at(ORIGIN).clone()
    with Seq(runtime=2, easing=easings.identity):
        camera.set_near_orthographic(1000)
    times = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    scene.timeline_manager.set_state_to_times(times)
    torch.testing.assert_close(
        camera.visible_size_at(ORIGIN), before.expand(len(times), -1, -1)
    )
    positions = camera.location.clone()
    screens = camera.screen.location.clone()
    for index in (3, 0, 4, 1, 2):
        scene.timeline_manager.set_state_to_times(times[index : index + 1])
        torch.testing.assert_close(camera.location[0], positions[index])
        torch.testing.assert_close(camera.screen.location[0], screens[index])
    scene.timeline_manager.clear_buffers()


def test_near_orthographic_constructor_preserves_configured_origin_frame():
    scene = Scene()
    kwargs = {"scene": scene, "location": (0, 0, 8), "screen_half_height": 3, "fov": 40}
    perspective = Camera(**kwargs)
    orthographic = Camera(orthographic=True, **kwargs)
    torch.testing.assert_close(
        orthographic.visible_size_at(ORIGIN), perspective.visible_size_at(ORIGIN)
    )
