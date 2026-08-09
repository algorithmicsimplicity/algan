import pytest
import torch

from algan import LEFT, ORIGIN, RIGHT, UP, Cube, Group, Mob, Off, SceneManager, Square
from algan.errors import AlganConfigurationError
from algan.utils.tensor_utils import unsquish


@pytest.fixture(autouse=True)
def fresh_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _screen_rectangle_at_z_zero(scene, bottom_left, top_right):
    camera = scene.camera
    half_height = camera.screen_scale_factor * (
        -camera.location[..., 2:3] / camera.screen_distance
    )
    half_width = half_height * (
        scene.video_settings.resolution[0] / scene.video_settings.resolution[1]
    )
    lower = torch.cat(
        (
            half_width * (2 * bottom_left[0] - 1),
            half_height * (2 * bottom_left[1] - 1),
            torch.zeros_like(half_height),
        ),
        -1,
    )
    upper = torch.cat(
        (
            half_width * (2 * top_right[0] - 1),
            half_height * (2 * top_right[1] - 1),
            torch.zeros_like(half_height),
        ),
        -1,
    )
    return lower, upper


def test_group_fits_exactly_into_normalized_screen_rectangle():
    scene = SceneManager.instance().current_scene
    left = Square(add_to_scene=False).move(2 * LEFT)
    right = Square(add_to_scene=False).move(2 * RIGHT)
    group = Group(left, right, add_to_scene=False)
    bottom_left = (0.1, 0.2)
    top_right = (0.6, 0.7)

    assert (
        group.fit_to_screen_rectangle(
            bottom_left, top_right, preserve_aspect_ratio=False
        )
        is group
    )

    bbox = group.get_bounding_box()
    expected_lower, expected_upper = _screen_rectangle_at_z_zero(
        scene, bottom_left, top_right
    )
    torch.testing.assert_close(bbox.amin(-2, keepdim=True), expected_lower)
    torch.testing.assert_close(bbox.amax(-2, keepdim=True), expected_upper)


def test_none_screen_rectangle_corners_default_to_whole_screen():
    scene = SceneManager.instance().current_scene
    square = Square(add_to_scene=False)
    expected_lower, expected_upper = _screen_rectangle_at_z_zero(scene, (0, 0), (1, 1))

    square.fit_to_screen_rectangle(preserve_aspect_ratio=False)

    bbox = square.get_bounding_box()
    torch.testing.assert_close(bbox.amin(-2, keepdim=True), expected_lower)
    torch.testing.assert_close(bbox.amax(-2, keepdim=True), expected_upper)


def test_exact_screen_rectangle_fit_animates_scale_and_position_together():
    scene = SceneManager.instance().current_scene
    square = Square().spawn(animate=False)
    source_size = square.get_axis_aligned_size()
    expected_lower, expected_upper = _screen_rectangle_at_z_zero(scene, (0, 0), (1, 1))
    target_size = expected_upper - expected_lower

    square.fit_to_screen_rectangle(preserve_aspect_ratio=False)
    scene.timeline_manager.set_state_to_times(torch.tensor([0.5]))

    torch.testing.assert_close(
        square.get_axis_aligned_size(), (source_size + target_size) * 0.5
    )
    torch.testing.assert_close(
        square.get_center(), (expected_lower + expected_upper) * 0.5
    )


def test_screen_rectangle_fit_can_preserve_aspect_ratio():
    scene = SceneManager.instance().current_scene
    group = Group(
        Square(add_to_scene=False).move(2 * LEFT),
        Square(add_to_scene=False).move(2 * RIGHT),
        add_to_scene=False,
    )
    source_ratio = group.get_width() / group.get_height()
    bottom_left = (0.25, 0.2)
    top_right = (0.75, 0.8)
    target_lower, target_upper = _screen_rectangle_at_z_zero(
        scene, bottom_left, top_right
    )

    group.fit_to_screen_rectangle(bottom_left, top_right, preserve_aspect_ratio=True)

    torch.testing.assert_close(group.get_width() / group.get_height(), source_ratio)
    torch.testing.assert_close(
        group.get_width(), (target_upper - target_lower)[..., 0:1]
    )
    torch.testing.assert_close(group.get_center(), (target_lower + target_upper) * 0.5)


def _rendered_screen_coords(scene, points):
    """Normalized screen coords of world points, via the *renderer's* projection.

    Deliberately built from ``get_render_screen_basis`` -- what the ray generator
    inverts -- rather than from the layout code's own screen frame, so these
    tests check the fit against what actually ends up on screen.
    """
    camera = scene.camera
    right, up, forward = unsquish(camera.get_render_screen_basis(), -1, 3).reshape(3, 3)
    screen = camera.screen.location.reshape(-1)
    eye = camera.location.reshape(-1)
    aspect = (
        scene.video_settings.resolution[0] / scene.video_settings.resolution[1]
    )
    ray = points.reshape(-1, 3) - eye
    on_plane = eye + ray * (torch.dot(screen - eye, forward) / (ray @ forward)).unsqueeze(
        -1
    )
    offset = on_plane - screen
    return torch.stack(((offset @ right) / aspect + 1, (offset @ up) + 1), -1) * 0.5


@pytest.mark.parametrize("camera_rotation", [0, 25])
def test_screen_rectangle_fit_keeps_a_mob_with_depth_inside_the_frame(camera_rotation):
    # Perspective magnifies whatever ends up nearest the camera, so fitting the
    # depth-less middle slice of a Cube used to leave half of it off-screen.
    scene = SceneManager.instance().current_scene
    if camera_rotation:
        with Off():
            scene.camera.rotate(camera_rotation, UP, about_point=ORIGIN)
    cube = Cube(add_to_scene=False)

    cube.fit_to_screen_rectangle()

    screen = _rendered_screen_coords(scene, cube.get_boundary_points_recursive())
    assert float(screen.amin(0).min()) >= -1e-3
    assert float(screen.amax(0).max()) <= 1 + 1e-3
    # ... and its fitted camera-aligned enclosure fills the frame rather than
    # shrinking away from the overflow.
    bbox_screen = _rendered_screen_coords(
        scene, cube._get_bounding_box_aligned_to(cube._screen_axes())
    )
    assert float(bbox_screen.amax(0).max()) > 0.99


def test_screen_rectangle_fit_measures_the_rectangle_in_the_cameras_frame():
    # The rectangle is the camera's, so it must follow the camera's right and up
    # axes; measuring it along the world x and y axes collapsed the Mob as the
    # camera turned away from them.
    scene = SceneManager.instance().current_scene
    with Off():
        scene.camera.rotate(30, UP, about_point=ORIGIN)
    square = Square(add_to_scene=False)

    square.fit_to_screen_rectangle((0.1, 0.2), (0.6, 0.7), preserve_aspect_ratio=False)

    screen = _rendered_screen_coords(
        scene, square._get_bounding_box_aligned_to(square._screen_axes())
    )
    torch.testing.assert_close(
        screen.amin(0), torch.tensor([0.1, 0.2]), atol=1e-3, rtol=0
    )
    torch.testing.assert_close(
        screen.amax(0), torch.tensor([0.6, 0.7]), atol=1e-3, rtol=0
    )


def test_screen_rectangle_fit_uses_a_camera_aligned_bounding_box():
    scene = SceneManager.instance().current_scene
    with Off():
        scene.camera.rotate(30, UP, about_point=ORIGIN)
    square = Square(add_to_scene=False)
    with Off():
        square.rotate(30, UP, about_point=ORIGIN)

    square.fit_to_screen_rectangle((0.1, 0.2), (0.6, 0.7), preserve_aspect_ratio=False)

    screen = _rendered_screen_coords(scene, square.get_boundary_points_recursive())
    torch.testing.assert_close(
        screen.amin(0), torch.tensor([0.1, 0.2]), atol=1e-3, rtol=0
    )
    torch.testing.assert_close(
        screen.amax(0), torch.tensor([0.6, 0.7]), atol=1e-3, rtol=0
    )


def test_layout_size_scale_and_center_helpers_are_chainable():
    square = Square(add_to_scene=False)

    torch.testing.assert_close(
        square.get_axis_aligned_size(), torch.tensor([[[2.0, 2.0, 0.0]]])
    )
    assert square.move_center_to((1, 2, 3)) is square
    torch.testing.assert_close(square.get_center(), torch.tensor([[[1.0, 2.0, 3.0]]]))
    assert square.scale_to_width(4) is square
    torch.testing.assert_close(square.get_width(), torch.tensor([[[4.0]]]))
    assert square.scale_to_height(1) is square
    torch.testing.assert_close(square.get_height(), torch.tensor([[[1.0]]]))
    torch.testing.assert_close(square.get_depth(), torch.tensor([[[0.0]]]))


def test_move_center_to_screen_position_accepts_screen_edges():
    scene = SceneManager.instance().current_scene
    square = Square(add_to_scene=False)
    expected, _ = _screen_rectangle_at_z_zero(scene, (1, 0), (1, 1))

    assert square.move_center_to_screen_position((1, 0)) is square
    torch.testing.assert_close(square.get_center(), expected)


@pytest.mark.parametrize(
    ("bottom_left", "top_right"),
    [
        ((-0.1, 0), (1, 1)),
        ((0, 0), (1.1, 1)),
        ((0.5, 0), (0.5, 1)),
        ((0.75, 0), (0.25, 1)),
        ((0, 0), (1, float("nan"))),
    ],
)
def test_screen_rectangle_fit_rejects_invalid_rectangles(bottom_left, top_right):
    with pytest.raises(AlganConfigurationError, match="screen rectangle"):
        Square(add_to_scene=False).fit_to_screen_rectangle(bottom_left, top_right)


def test_screen_rectangle_fit_rejects_zero_sized_mobs():
    with pytest.raises(AlganConfigurationError, match="zero width or height"):
        Mob(add_to_scene=False).fit_to_screen_rectangle()
