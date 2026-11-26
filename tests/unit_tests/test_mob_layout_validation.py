"""Layout and placement arguments that used to fail badly or do nothing.

Render-free: every Mob is laid out inside ``Off()``.
"""

import pytest
import torch

from algan import ORIGIN, RIGHT, UP, Circle, Group, Off, Scene, SceneManager, Square
from algan.errors import AlganConfigurationError


@pytest.fixture(autouse=True)
def fresh_scene_stack():
    SceneManager.reset()
    yield
    SceneManager.reset()


@pytest.mark.parametrize(
    "call",
    [
        lambda mob: mob.move_center_to((1, 2)),
        lambda mob: mob.move_to((1, 2), 90),
    ],
    ids=["move_center_to", "move_to-arc"],
)
def test_a_two_component_point_is_told_it_needs_three(call):
    # Both failed with a bare "size of tensor a (2) must match" from deep in
    # the arithmetic, while move_to's straight path already said why.
    with Scene(), Off():
        mob = Square()
        with pytest.raises(AlganConfigurationError, match="3-D vector"):
            call(mob)


@pytest.mark.parametrize(
    "call",
    [
        lambda mob, other: mob.move_next_to(other, ORIGIN),
        lambda mob, other: mob.move_next_to(other, RIGHT, align_edge=ORIGIN),
        lambda mob, other: mob.align_with(other, ORIGIN),
    ],
    ids=["move_next_to", "move_next_to-align_edge", "align_with"],
)
def test_a_zero_side_direction_is_refused(call):
    # A zero direction normalised to zero: move_next_to put the Mob on the
    # other's centre, and align_with did nothing, both silently.
    with Scene(), Off():
        mob, other = Square(), Circle().move(RIGHT * 3)
        before = mob.location.clone()
        with pytest.raises(AlganConfigurationError, match="non-zero direction"):
            call(mob, other)
        assert torch.equal(mob.location, before)


def test_an_only_child_excluded_from_the_boundary_is_not_measured():
    # The one-child shortcut skipped the flag, so a Group whose only member
    # is excluded still measured it.
    with Scene(), Off():
        label = Square().move(RIGHT * 5)
        label.exclude_from_boundary = True
        g = Group(label)
        g.location = ORIGIN
        edge = g.get_edge_point(RIGHT)
        torch.testing.assert_close(edge, g.location)


def test_a_group_of_only_excluded_members_can_still_be_placed():
    # torch.cat([]) raised from every edge query once every child was
    # excluded, taking move_next_to, align_with and arrange_in_line with it.
    with Scene(), Off():
        a, b = Square(), Square().move(RIGHT * 2)
        a.exclude_from_boundary = b.exclude_from_boundary = True
        g = Group(a, b)
        g.move_next_to(Circle(), UP)
        g.get_boundary_points_recursive()
