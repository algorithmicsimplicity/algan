import math

import pytest
import torch

from algan import Mob, Scene, SceneManager

# In the fast suite: moving a Mob is the most-used recorded operation there is,
# and these check the path it traces rather than only where it ends up.
pytestmark = pytest.mark.fast


def _empty_scene(scene):
    scene.camera = None
    scene.light_sources = []


@pytest.fixture(autouse=True)
def fresh_scene_stack():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _midpoint_of_recorded_move(angle_degrees):
    scene = Scene(scene_initializer=_empty_scene)
    mob = Mob(location=[0, 0, 0]).spawn(animate=False)
    mob.move_to([2, 0, 0], angle_degrees, arc_normal=[0, 0, 1])
    scene.timeline_manager.set_state_to_times(torch.tensor([0.5]))
    location = mob.location.clone()
    scene.terminate()
    return location


@pytest.mark.parametrize(
    ("angle_degrees", "expected_y"),
    [
        (90, math.sqrt(2) - 1),
        (-90, 1 - math.sqrt(2)),
        (180, 1),
        (270, 1 + math.sqrt(2)),
        (450, -(1 + math.sqrt(2))),
    ],
)
def test_arc_move_preserves_signed_sweep_at_animation_midpoint(
    angle_degrees, expected_y
):
    midpoint = _midpoint_of_recorded_move(angle_degrees)
    expected = torch.tensor([[[1.0, expected_y, 0.0]]])
    torch.testing.assert_close(midpoint, expected, atol=2e-5, rtol=2e-5)


def test_zero_angle_uses_the_straight_line_limit():
    midpoint = _midpoint_of_recorded_move(0)
    torch.testing.assert_close(
        midpoint, torch.tensor([[[1.0, 0.0, 0.0]]]), atol=5e-6, rtol=0
    )


@pytest.mark.parametrize("angle_degrees", [0.001, 0.00001])
def test_shallow_arc_remains_finite_and_reaches_target(angle_degrees):
    scene = Scene(scene_initializer=_empty_scene)
    mob = Mob(location=[0, 0, 0]).spawn(animate=False)
    target = torch.tensor([[[2.0, 0.0, 0.0]]])

    mob.move_to(target, angle_degrees, arc_normal=[0, 0, 1])
    torch.testing.assert_close(mob.location, target, atol=1e-6, rtol=0)

    scene.timeline_manager.set_state_to_times(torch.tensor([0.5]))
    assert torch.isfinite(mob.location).all()
    torch.testing.assert_close(
        mob.location[..., 0], torch.tensor([[1.0]]), atol=2e-5, rtol=0
    )
    scene.terminate()


def test_non_unit_normal_supports_an_arbitrary_arc_plane():
    scene = Scene(scene_initializer=_empty_scene)
    mob = Mob(location=[0, 0, 0]).spawn(animate=False)
    mob.move_to(
        [0, 2, 0],
        180,
        arc_normal=[5, 0, 0],
    )
    scene.timeline_manager.set_state_to_times(torch.tensor([0.5]))

    torch.testing.assert_close(
        mob.location,
        torch.tensor([[[0.0, 1.0, 1.0]]]),
        atol=2e-5,
        rtol=2e-5,
    )
    scene.terminate()


def test_batched_zero_and_circular_sweeps_are_supported_together():
    scene = Scene(scene_initializer=_empty_scene)
    mob = Mob(
        location=[[0, 0, 0], [0, 0, 0]],
        add_to_scene=False,
    ).spawn(animate=False)
    mob.move_to(
        [[2, 0, 0], [2, 0, 0]],
        torch.tensor([0.0, 90.0]).view(1, 2, 1),
        arc_normal=[0, 0, 1],
    )
    scene.timeline_manager.set_state_to_times(torch.tensor([0.5]))

    expected = torch.tensor([[[1.0, 0.0, 0.0], [1.0, math.sqrt(2) - 1, 0.0]]])
    torch.testing.assert_close(mob.location, expected, atol=2e-5, rtol=2e-5)
    scene.terminate()


def test_non_recursive_arc_move_leaves_child_location_unchanged():
    scene = Scene(scene_initializer=_empty_scene)
    parent = Mob(location=[0, 0, 0], add_to_scene=False)
    child = Mob(location=[0, 1, 0], add_to_scene=False)
    parent.add_children(child)
    parent.spawn(animate=False)

    parent.move_to(
        [2, 0, 0],
        90,
        arc_normal=[0, 0, 1],
        recursive=False,
    )
    scene.timeline_manager.set_state_to_times(torch.tensor([0.5]))

    torch.testing.assert_close(
        parent.location,
        torch.tensor([[[1.0, math.sqrt(2) - 1, 0.0]]]),
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        child.location, torch.tensor([[[0.0, 1.0, 0.0]]]), atol=1e-6, rtol=0
    )
    scene.terminate()


@pytest.mark.parametrize(
    ("target", "angle", "normal", "message"),
    [
        ([1, 0, 1], 90, [0, 0, 1], "perpendicular"),
        ([1, 0, 0], 360, [0, 0, 1], "multiple-of-360"),
        ([1, 0, 0], 90, [0, 0, 0], "non-zero vector"),
    ],
)
def test_invalid_arc_geometry_is_rejected(target, angle, normal, message):
    scene = Scene(scene_initializer=_empty_scene)
    mob = Mob(location=[0, 0, 0], add_to_scene=False)
    with pytest.raises(ValueError, match=message):
        mob.move_to(target, angle, arc_normal=normal)
    scene.terminate()
