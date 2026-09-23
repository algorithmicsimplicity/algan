from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from algan import ORIGIN, OUT, PREVIEW, UP, CameraView, Off, Scene, Seq, easings
from algan.errors import AlganConfigurationError
from algan.utils.tensor_utils import unsquish

pytestmark = pytest.mark.usefixtures("fresh_scene")


@pytest.mark.fast
def test_flight_tracks_target_through_waypoint_and_is_seek_independent():
    scene = Scene()
    camera = scene.camera
    with Off():
        camera.move_to((0, 0, 7))
        camera.look_at(ORIGIN)
    start_screen_distance = (camera.screen.location - camera.location).norm().item()
    with Seq(runtime=2, easing=easings.identity):
        camera.fly_to((4, 2, 7), look_at=ORIGIN, via=(2, 3, 8))
    authored = camera.location.clone()
    times = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    scene.timeline_manager.set_state_to_times(times)
    positions, bases = camera.location.clone(), camera.basis.clone()
    torch.testing.assert_close(positions[2].flatten(), torch.tensor([2.0, 3.0, 8.0]))
    torch.testing.assert_close(positions[-1], authored[0])
    torch.testing.assert_close(camera.forward, F.normalize(-positions, dim=-1))
    torch.testing.assert_close(
        camera.right[..., 1], torch.zeros_like(camera.right[..., 1])
    )
    screen_delta = camera.screen.location - positions
    torch.testing.assert_close(screen_delta, camera.forward * start_screen_distance)
    matrix = unsquish(bases, -1, 3)
    torch.testing.assert_close(
        matrix @ matrix.transpose(-1, -2), torch.eye(3).expand_as(matrix)
    )
    for index in (3, 1, 4, 0, 2):
        scene.timeline_manager.set_state_to_times(times[index : index + 1])
        torch.testing.assert_close(camera.location[0], positions[index])
        torch.testing.assert_close(camera.basis[0], bases[index])
    scene.timeline_manager.clear_buffers()
    torch.testing.assert_close(camera.location, authored)


def test_flight_moves_aim_and_position_and_can_preserve_direction():
    scene = Scene()
    camera = scene.camera
    with Off():
        camera.move_to((0, 0, 7))
    target = torch.tensor([2.0, 1.0, 0.0])
    start = camera.location.clone()
    initial_target = start + camera.forward * (target - start).norm(
        dim=-1, keepdim=True
    )
    with Seq(easing=easings.identity):
        camera.fly_to((4, 0, 6), look_at=target)
    scene.timeline_manager.set_state_to_times(torch.tensor([0.5]))
    aim = (initial_target + target) * 0.5 - camera.location
    torch.testing.assert_close(camera.forward, F.normalize(aim, dim=-1))
    scene.timeline_manager.clear_buffers()
    forward = camera.forward.clone()
    with Off():
        camera.fly_to((1, 2, 3))
    torch.testing.assert_close(camera.forward, forward)


def test_tilted_start_levels_out_without_a_first_frame_jump():
    scene = Scene()
    camera = scene.camera
    with Off():
        camera.move_to((0, 0, 7))
        camera.look_at(ORIGIN)
        camera.rotate(30, OUT)
    start_basis = camera.basis.clone()
    with Seq(runtime=2, easing=easings.identity):
        camera.fly_to((1, 0, 7))
    scene.timeline_manager.set_state_to_times(torch.tensor([0.0, 1.0, 2.0]))
    torch.testing.assert_close(camera.basis[0], start_basis[0], atol=1e-5, rtol=0)
    # Halfway, half of the 30 degree roll remains; at the end it is level.
    right = camera.right.reshape(3, 3)
    assert right[1, 1].item() == pytest.approx(math.sin(math.radians(15)), abs=1e-5)
    assert right[2, 1].item() == pytest.approx(0.0, abs=1e-6)
    scene.timeline_manager.clear_buffers()


def test_look_at_waypoint_curves_the_aim():
    scene = Scene()
    camera = scene.camera
    with Off():
        camera.move_to((0, 0, 7))
        camera.look_at(ORIGIN)
    with Seq(easing=easings.identity):
        camera.fly_to((0, 0, 7), look_at=(2, 0, 0), look_at_via=(0, 2, 0))
    scene.timeline_manager.set_state_to_times(torch.tensor([0.5]))
    torch.testing.assert_close(
        camera.forward.flatten(), F.normalize(torch.tensor([0.0, 2.0, -7.0]), dim=0)
    )
    scene.timeline_manager.clear_buffers()
    with pytest.raises(AlganConfigurationError, match="look_at_via"):
        camera.fly_to((1, 0, 7), look_at_via=(0, 1, 0))


def test_camera_view_measures_its_own_capture_aspect():
    scene = Scene(PREVIEW.set(resolution=(90, 160)))
    view = CameraView(resolution=(400, 200), scene=scene)
    width, height = view.camera.visible_size_at(ORIGIN).flatten().tolist()
    assert width == pytest.approx(2 * height)
    main_width, main_height = scene.camera.visible_size_at(ORIGIN).flatten().tolist()
    assert main_width == pytest.approx(main_height * 90 / 160)


def test_vertical_flight_keeps_a_finite_orthonormal_frame():
    scene = Scene()
    with Off():
        scene.camera.fly_to((0, 7, 0), look_at=ORIGIN)
    matrix = unsquish(scene.camera.basis, -1, 3)
    torch.testing.assert_close(
        matrix @ matrix.transpose(-1, -2), torch.eye(3).expand_as(matrix)
    )
    torch.testing.assert_close(scene.camera.forward.flatten(), -UP)


def test_visible_size_uses_forward_depth_and_portrait_aspect():
    scene = Scene(PREVIEW.set(resolution=(90, 160)))
    camera = scene.camera
    with Off():
        camera.move_to((0, 0, 10))
        camera.set_fov(90)
    torch.testing.assert_close(
        camera.visible_size_at((3, 0, 0)).flatten(), torch.tensor([11.25, 20.0])
    )
    with Off():
        camera.fly_to((10, 0, 0), look_at=ORIGIN)
    torch.testing.assert_close(
        camera.visible_size_at(ORIGIN).flatten(), torch.tensor([11.25, 20.0])
    )
    with pytest.raises(AlganConfigurationError, match="front"):
        camera.visible_size_at((11, 0, 0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position": (1, 2)},
        {"position": (1, float("nan"), 3)},
        {"position": (0, 0, 0), "look_at": (0, 0, 0)},
        {"position": (0, 0, 7), "via": (1, float("inf"), 2)},
    ],
)
def test_invalid_flight_points_fail_before_recording(kwargs):
    with pytest.raises(AlganConfigurationError):
        Scene().camera.fly_to(**kwargs)
