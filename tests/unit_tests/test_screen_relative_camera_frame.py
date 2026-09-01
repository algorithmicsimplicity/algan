"""The screen-relative movement helpers name their directions in the camera's frame.

``move_to_screen_edge(RIGHT)`` means the right of the *screen*, so it has to
follow ``camera.right`` rather than world ``+x``. The helpers used to hand the
caller's world vector straight to
:meth:`~algan.rendering.camera.Camera.project_point_onto_screen_border`: only
the frustum planes it intersects were camera-aware, the ray cast at them was
not. Under a 60-degree yaw that made ``move_to_screen_edge(LEFT)`` displace a
Mob along world *+x* and come to rest off the *right* of the frame, and made
every screen-plane move change the Mob's distance to the camera, so it changed
size on the way.

The third axis is the one place the frame is not just a relabelling: ``z`` runs
out of the screen towards the viewer, i.e. along ``-camera.forward``, so
``RIGHT + OUT`` casts along the diagonal of the camera's right and the
direction back towards the eye.

The default camera's right/up/-forward are exactly ``RIGHT``, ``UP`` and
``OUT``, so none of this may move a Mob that an unrotated camera is looking at
-- ``test_the_default_camera_is_left_untouched`` is the guard on that. That
exactness is also what makes the frame's world-frame fast path fire: built from
``forward`` instead, the default camera gives ``diag(1, 1, -1)``, the ``None``
never comes back, and every caller takes the rotated path to compute a result it
already had.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from algan import (
    IN,
    ORIGIN,
    OUT,
    RIGHT,
    UP,
    Off,
    SceneManager,
    Square,
)
from algan.utils.tensor_utils import unsquish

# Every pose that separates the camera's frame from the world's: a yaw takes
# ``right`` off world +x, a roll takes ``up`` off world +y, a pitch tilts both
# out of the world's axis planes, and the composite leaves no axis shared.
POSES = {
    "yaw": [(60, UP)],
    "roll": [(30, OUT)],
    "pitch": [(40, RIGHT)],
    "yaw_pitch_roll": [(50, UP), (35, RIGHT), (20, OUT)],
}

EDGES = {"right": RIGHT, "left": -RIGHT, "up": UP, "down": -UP}


@pytest.fixture(autouse=True)
def fresh_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _posed_scene(pose_name):
    scene = SceneManager.instance().current_scene
    with Off():
        for angle, axis in POSES[pose_name]:
            scene.camera.rotate(angle, axis, about=ORIGIN)
    return scene


def _camera_frame(scene):
    """``(right, up, -forward)``, the frame the helpers read directions in."""
    camera = scene.camera
    return [
        v.reshape(-1)
        for v in (
            camera.get_right_direction(),
            camera.get_up_direction(),
            -camera.get_forward_direction(),
        )
    ]


def _rendered_screen_coords(scene, points):
    """Normalized screen coords of world points, via the *renderer's* projection.

    Built from ``_get_render_screen_basis`` -- what the ray generator inverts --
    so these tests check where the Mob actually lands on screen rather than
    re-deriving it from the code under test.
    """
    camera = scene.camera
    right, up, forward = unsquish(camera._get_render_screen_basis(), -1, 3).reshape(
        3, 3
    )
    screen = camera.screen.location.reshape(-1)
    eye = camera.location.reshape(-1)
    aspect = scene.video_settings.resolution[0] / scene.video_settings.resolution[1]
    ray = points.reshape(-1, 3) - eye
    on_plane = eye + ray * (
        torch.dot(screen - eye, forward) / (ray @ forward)
    ).unsqueeze(-1)
    offset = on_plane - screen
    return torch.stack(((offset @ right) / aspect + 1, (offset @ up) + 1), -1) * 0.5


def _depth(scene, point):
    """Distance from the camera along its forward axis -- what perspective scales by."""
    camera = scene.camera
    forward = camera.get_forward_direction().reshape(-1)
    return float((point.reshape(-1) - camera.location.reshape(-1)) @ forward)


@pytest.mark.parametrize("pose_name", sorted(POSES))
@pytest.mark.parametrize("edge_name", sorted(EDGES))
def test_move_to_screen_edge_lands_against_the_named_screen_edge(pose_name, edge_name):
    scene = _posed_scene(pose_name)
    square = Square()

    square.move_to_screen_edge(EDGES[edge_name])

    screen = _rendered_screen_coords(scene, square.get_center()).reshape(-1)
    # Which screen axis the edge is on, and which end of it: LEFT/DOWN are the
    # low end, RIGHT/UP the high end.
    axis = 0 if edge_name in ("left", "right") else 1
    if edge_name in ("right", "up"):
        assert float(screen[axis]) > 0.7
    else:
        assert float(screen[axis]) < 0.3
    # The other screen axis is untouched: the Mob slid along the edge's axis
    # only, it did not drift diagonally.
    assert float(screen[1 - axis]) == pytest.approx(0.5, abs=1e-3)


@pytest.mark.parametrize("pose_name", sorted(POSES))
def test_opposite_edges_are_symmetric_about_the_middle_of_the_screen(pose_name):
    scene = _posed_scene(pose_name)

    left = Square()
    left.move_to_screen_edge(-RIGHT)
    right = Square()
    right.move_to_screen_edge(RIGHT)

    left_x = float(_rendered_screen_coords(scene, left.get_center()).reshape(-1)[0])
    right_x = float(_rendered_screen_coords(scene, right.get_center()).reshape(-1)[0])
    assert left_x + right_x == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("pose_name", sorted(POSES))
@pytest.mark.parametrize("edge_name", sorted(EDGES))
def test_a_screen_plane_edge_move_keeps_the_mobs_distance_to_the_camera(
    pose_name, edge_name
):
    # The whole point of the camera frame: an edge direction has no z component,
    # so the Mob travels in the plane parallel to the screen and its apparent
    # size does not change. Casting along a world axis used to cost it depth --
    # 2.9 units of a 7-unit distance for RIGHT under a 60-degree yaw.
    scene = _posed_scene(pose_name)
    square = Square()
    before = _depth(scene, square.get_center())

    square.move_to_screen_edge(EDGES[edge_name])

    assert _depth(scene, square.get_center()) == pytest.approx(before, abs=1e-4)


@pytest.mark.parametrize("pose_name", sorted(POSES))
@pytest.mark.parametrize("edge_name", sorted(EDGES))
def test_the_displacement_runs_along_the_cameras_own_axis(pose_name, edge_name):
    scene = _posed_scene(pose_name)
    right, up, _ = _camera_frame(scene)
    axis = right if edge_name in ("left", "right") else up
    # Signed: the displacement runs *towards* the named edge, so LEFT and DOWN
    # travel along minus the camera's axis. Comparing magnitudes would pass a
    # move that went the wrong way, which is exactly the bug this file is about.
    expected = -axis if edge_name in ("left", "down") else axis
    square = Square()
    before = square.get_center().reshape(-1).clone()

    square.move_to_screen_edge(EDGES[edge_name])

    displacement = square.get_center().reshape(-1) - before
    torch.testing.assert_close(
        F.normalize(displacement, p=2, dim=-1),
        expected,
        atol=1e-4,
        rtol=0,
    )


@pytest.mark.parametrize("pose_name", sorted(POSES))
def test_move_to_screen_corner_lands_against_both_named_edges(pose_name):
    scene = _posed_scene(pose_name)
    square = Square()

    square.move_to_screen_corner((UP, RIGHT))

    screen = _rendered_screen_coords(scene, square.get_center()).reshape(-1)
    assert float(screen[0]) > 0.7
    assert float(screen[1]) > 0.7


@pytest.mark.parametrize("pose_name", sorted(POSES))
def test_move_to_screen_position_places_the_mob_in_the_named_quadrant(pose_name):
    # Built out of the four screen corners, so it inherits the frame from them.
    # Under a yaw this used to land at (1.96, 0.06) -- off the frame entirely --
    # for a request near the top left.
    scene = _posed_scene(pose_name)
    square = Square()

    square.move_to_screen_position(0.1, 0.9)

    screen = _rendered_screen_coords(scene, square.get_center()).reshape(-1)
    assert 0.0 < float(screen[0]) < 0.5
    assert 0.5 < float(screen[1]) < 1.0


@pytest.mark.parametrize("pose_name", sorted(POSES))
def test_an_out_component_casts_along_minus_the_cameras_forward(pose_name):
    # z runs out of the screen towards the viewer, so RIGHT + OUT travels along
    # the diagonal of camera.right and -camera.forward until it leaves the
    # frustum -- across the screen and towards the eye at once.
    scene = _posed_scene(pose_name)
    right, _, out = _camera_frame(scene)
    square = Square()
    before = square.get_center().reshape(-1).clone()

    square.move_to_screen_edge(RIGHT + OUT)

    displacement = square.get_center().reshape(-1) - before
    torch.testing.assert_close(
        F.normalize(displacement, p=2, dim=-1),
        F.normalize(right + out, p=2, dim=-1),
        atol=1e-4,
        rtol=0,
    )
    # ... and it really did come closer to the camera, rather than just sliding
    # across an edge move's screen-parallel plane.
    assert _depth(scene, square.get_center()) < _depth(scene, before)


@pytest.mark.parametrize("pose_name", sorted(POSES))
def test_move_off_screen_leaves_by_the_named_screen_edge(pose_name):
    scene = _posed_scene(pose_name)
    square = Square()

    square.move_off_screen(RIGHT, despawn=False)

    screen = _rendered_screen_coords(scene, square.get_center()).reshape(-1)
    assert float(screen[0]) > 1.0
    assert float(screen[1]) == pytest.approx(0.5, abs=1e-3)


@pytest.mark.parametrize("pose_name", sorted(POSES))
def test_the_shared_frame_is_right_up_and_minus_forward(pose_name):
    # Both mixins read directions through _screen_axes, so its rows are the
    # contract. The third is *minus* forward: built from forward it gave
    # diag(1, 1, -1) for the default camera, which is never equal to the
    # identity, so the world-frame fast path below could not fire at all.
    scene = _posed_scene(pose_name)
    square = Square()

    axes = square._screen_axes()

    assert axes is not None
    torch.testing.assert_close(
        axes.reshape(3, 3), torch.stack(_camera_frame(scene)), atol=1e-6, rtol=0
    )


@pytest.mark.parametrize(
    "direction", [RIGHT, -RIGHT, UP, -UP, RIGHT + UP, OUT, RIGHT + OUT, IN]
)
def test_the_default_camera_is_left_untouched(direction):
    # The default camera's frame *is* the world frame, so the mapping must be
    # the exact identity there -- not identity up to a matmul's rounding, which
    # would shift every rendered baseline by a pixel for no reason.
    scene = SceneManager.instance().current_scene
    square = Square()
    mapped = square._screen_relative_direction(direction)

    assert square._screen_axes() is None
    assert mapped is direction
    assert scene.camera.get_right_direction().reshape(-1).tolist() == [1.0, 0.0, 0.0]
