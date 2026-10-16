"""The camera's two projections: perspective, and the true orthographic one.

``Camera.set_orthographic`` makes every camera ray parallel to the forward
axis, starting on the plane through the camera, so the visible frame is one
rectangle of fixed world size at every depth. These hold the camera's side of
that without rendering: the frame it keeps, the dolly zoom an animated switch
is, the way back to perspective, the old ``set_near_orthographic`` spelling,
and the screen-relative layout built on the frame. The renders are in
``test_orthographic_render.py``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from algan import (
    ORIGIN,
    OUT,
    PREVIEW,
    RIGHT,
    UP,
    Camera,
    Off,
    Scene,
    Seq,
    Square,
    easings,
)
from algan.errors import AlganConfigurationError
from algan.rendering.projection import effective_depth


def _screen_coords(camera, points):
    """Normalized screen coordinates of world points.

    Built from the camera's render inputs -- the screen basis and the
    ``(persp, spread)`` terms -- by the projection ``algan.rendering.projection``
    documents, independently of ``visible_size_at`` and of the layout code.
    """
    basis = camera._get_render_screen_basis()
    eye = camera.location
    screen = camera.screen.location
    persp, spread = camera._projection_terms()
    forward = F.normalize(basis[..., 2, :], dim=-1)
    ray = points - eye
    depth = (ray * forward).sum(-1, keepdim=True)
    distance = ((screen - eye) * forward).sum(-1, keepdim=True)
    offset = (
        (ray - depth * forward)
        * distance
        / effective_depth(depth, distance, persp, spread)
    )
    return torch.stack(
        ((offset * basis[..., 0, :]).sum(-1), (offset * basis[..., 1, :]).sum(-1)),
        -1,
    )


@pytest.mark.parametrize("resolution", [(640, 360), (360, 640)])
@pytest.mark.parametrize("posed", [False, True])
def test_orthographic_keeps_the_origin_plane_frame(resolution, posed):
    scene = Scene(PREVIEW.set(resolution=resolution))
    camera = scene.camera
    with Off():
        if posed:
            camera.move_to((4, 2, 9))
            camera.look_at(torch.tensor([1.0, -1.0, 0.0]))
            camera.set_fov(37)
    points = torch.cat((ORIGIN.view(1, 1, 3), camera.right, camera.up), -2)
    size_before = camera.visible_size_at(ORIGIN).clone()
    coords_before = _screen_coords(camera, points)
    location = camera.location.clone()
    screen = camera.screen.location.clone()
    basis = camera.basis.clone()
    fov = camera.get_fov()
    with Off():
        assert camera.set_orthographic() is camera
    assert camera.orthographic
    torch.testing.assert_close(camera.visible_size_at(ORIGIN), size_before)
    torch.testing.assert_close(
        _screen_coords(camera, points), coords_before, atol=1e-5, rtol=1e-5
    )
    # Only the projection changed: the camera did not move or turn.
    torch.testing.assert_close(camera.location, location)
    torch.testing.assert_close(camera.screen.location, screen)
    torch.testing.assert_close(camera.basis, basis)
    assert camera.get_fov() == pytest.approx(fov)


def test_the_orthographic_frame_is_manims_at_every_depth():
    scene = Scene(PREVIEW.set(resolution=(640, 360)))
    camera = scene.camera
    expected = torch.tensor([128 / 9, 8.0])
    torch.testing.assert_close(camera.visible_size_at(ORIGIN).flatten(), expected)
    with Off():
        camera.set_orthographic()
    # The default camera sits 20 units out: 19 is a unit in front of it.
    for z in (-40.0, -5.0, 0.0, 5.0, 19.0):
        torch.testing.assert_close(camera.visible_size_at(OUT * z).flatten(), expected)
    # No perspective: a point keeps its place on screen however deep it is.
    coords = _screen_coords(
        camera, torch.tensor([[[1.0, 0.5, 15.0], [1.0, 0.5, -40.0]]])
    )
    torch.testing.assert_close(coords[..., 0, :], coords[..., 1, :])


def test_set_orthographic_takes_a_frame_height():
    scene = Scene(PREVIEW.set(resolution=(640, 360)))
    camera = scene.camera
    with Off():
        camera.set_orthographic(frame_height=3)
    torch.testing.assert_close(
        camera.visible_size_at(OUT * 7).flatten(), torch.tensor([16 / 3, 3.0])
    )
    for bad in (0, -1, float("inf"), float("nan"), "tall"):
        with pytest.raises(AlganConfigurationError):
            camera.set_orthographic(frame_height=bad)


def test_the_switch_is_a_dolly_zoom_through_animation_and_seeking():
    scene = Scene()
    camera = scene.camera
    origin_frame = camera.visible_size_at(ORIGIN).clone()
    near_frame = camera.visible_size_at(OUT * 10).clone()
    with Seq(runtime=2, easing=easings.identity):
        camera.set_orthographic()
    # Read as its end state as soon as it is recorded.
    assert camera.orthographic
    times = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
    scene.timeline_manager.set_state_to_times(times)
    # The frame on the origin plane holds still...
    torch.testing.assert_close(
        camera.visible_size_at(ORIGIN), origin_frame.expand(len(times), -1, -1)
    )
    # ...while a plane 10 units nearer grows from its perspective frame to it.
    near = camera.visible_size_at(OUT * 10)[..., 1].flatten()
    assert float(near[0]) == pytest.approx(float(near_frame[..., 1]))
    assert float(near[-1]) == pytest.approx(float(origin_frame[..., 1]))
    assert bool((near[1:] > near[:-1]).all())
    # The camera itself never moves.
    positions = camera.location.clone()
    screens = camera.screen.location.clone()
    torch.testing.assert_close(positions, positions[:1].expand_as(positions))
    torch.testing.assert_close(screens, screens[:1].expand_as(screens))
    sizes = camera.visible_size_at(OUT * 10).clone()
    for index in (3, 0, 4, 1, 2):
        scene.timeline_manager.set_state_to_times(times[index : index + 1])
        torch.testing.assert_close(camera.visible_size_at(OUT * 10)[0], sizes[index])
    scene.timeline_manager.clear_buffers()


def test_set_perspective_restores_the_pinhole():
    scene = Scene()
    camera = scene.camera
    depths = (-5.0, 0.0, 5.0)
    before = [camera.visible_size_at(OUT * z).clone() for z in depths]
    fov = camera.get_fov()
    assert camera._is_pinhole()
    with Off():
        camera.set_orthographic()
    assert not camera._is_pinhole()
    with Seq(runtime=1):
        assert camera.set_perspective() is camera
    assert not camera.orthographic
    # Exactly the pinhole again, so renders take the perspective path bit for bit.
    assert camera._is_pinhole()
    for z, size in zip(depths, before):
        torch.testing.assert_close(camera.visible_size_at(OUT * z), size)
    assert camera.get_fov() == pytest.approx(fov)


def test_set_fov_and_set_distance_to_screen_switch_back_to_perspective():
    scene = Scene()
    camera = scene.camera
    with Off():
        camera.set_orthographic()
        camera.set_fov(30)
    assert camera._is_pinhole()
    assert camera.get_fov() == pytest.approx(30)
    with Off():
        camera.set_orthographic()
        camera.set_distance_to_screen(12)
    assert camera._is_pinhole()
    distance = (camera.screen.location - camera.location).norm()
    assert float(distance) == pytest.approx(12)


def test_set_near_orthographic_is_set_orthographic():
    scene = Scene(PREVIEW.set(resolution=(640, 360)))
    camera = scene.camera
    before = camera.visible_size_at(ORIGIN).clone()
    location = camera.location.clone()
    with Off():
        assert camera.set_near_orthographic() is camera
    assert camera.orthographic
    torch.testing.assert_close(camera.visible_size_at(OUT * 7), before)
    # It no longer backs the eye away.
    torch.testing.assert_close(camera.location, location)
    with pytest.raises(AlganConfigurationError, match="set_orthographic"):
        camera.set_near_orthographic(1000)


def test_the_orthographic_constructor_keeps_the_configured_origin_frame():
    scene = Scene()
    kwargs = {"scene": scene, "location": (0, 0, 8), "screen_half_height": 3, "fov": 40}
    perspective = Camera(**kwargs)
    orthographic = Camera(orthographic=True, **kwargs)
    assert orthographic.orthographic
    assert not perspective.orthographic
    frame = perspective.visible_size_at(ORIGIN)
    torch.testing.assert_close(orthographic.visible_size_at(ORIGIN), frame)
    torch.testing.assert_close(orthographic.visible_size_at(OUT * 4), frame)


def test_center_on_resizes_an_orthographic_frame():
    scene = Scene(PREVIEW.set(resolution=(640, 360)))
    camera = scene.camera
    square = Square(size=2).move_to(RIGHT * 3 + UP - OUT * 6)
    with Off():
        camera.set_orthographic()
        camera.center_on(square, buffer_portion=0.5)
    eye = camera.location.reshape(3)
    # Centred across the view, without dollying in (which would change nothing).
    torch.testing.assert_close(eye, torch.tensor([3.0, 1.0, 20.0]))
    # The square and half of it again for the margin, top to bottom.
    assert float(camera.visible_size_at(ORIGIN)[..., 1]) == pytest.approx(3.0)


def test_screen_layout_under_an_orthographic_camera_ignores_depth():
    scene = Scene(PREVIEW.set(resolution=(640, 360)))
    with Off():
        scene.camera.set_orthographic()
        squares = [Square(size=1).move_to(OUT * z) for z in (12.0, 0.0, -30.0)]
        for square in squares:
            square.move_to_screen_edge(RIGHT, buffer=0.5)
    # The frame is 128/9 wide at every depth, so each lands 0.5 inside its edge.
    for square in squares:
        right = float(square.get_boundary_point(RIGHT).reshape(3)[0])
        assert right == pytest.approx(64 / 9 - 0.5, abs=1e-4)
    with Off():
        for square in squares:
            square.move_center_to_screen_position((0.25, 0.75))
    for square in squares:
        torch.testing.assert_close(
            square.get_center().reshape(3)[:2],
            torch.tensor([-32 / 9, 2.0]),
            atol=1e-4,
            rtol=1e-4,
        )
    # fit_to_screen sizes by the frame alone: a Square filling the frame is as
    # tall as it at any depth.
    with Off():
        for square in squares:
            square.fit_to_screen()
    for square in squares:
        assert float(square.get_height()) == pytest.approx(8.0, abs=1e-4)
