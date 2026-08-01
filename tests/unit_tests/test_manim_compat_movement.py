"""Relative placement of the Manim-compatibility Mobs.

These Mobs delegate transforms to a backing Manim Mobject, and two mismatches at
that boundary used to make every relative-placement helper raise:

* Algan points carry a leading batch dimension (``(1, 1, 3)``) that Manim's
  in-place point arithmetic cannot broadcast against its own ``(N, 3)`` points.
* A compatibility Mob's :attr:`~.Mob.location` is the center of the backing
  Mobject's *own* points, which is not the composite's center when it also has
  submobjects -- an :class:`~.Arrow` carries its tip as a submobject, so its
  location sits behind its visible center.

The second is why displacements have to reach Manim as a ``shift`` rather than as
an absolute ``move_to`` target: only then does the Mob travel exactly as far as
the caller asked.
"""

import math

import pytest
import torch

from algan import (
    DOWN,
    LEFT,
    RIGHT,
    SETTINGS,
    UP,
    Arrow,
    Axes,
    SceneManager,
    Square,
    Star,
)


@pytest.fixture(autouse=True)
def fresh_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


# Axes' backing Mobject holds no points of its own, Star's holds all of them,
# and Arrow's holds some with the rest in a submobject -- one of each way the
# location anchor can relate to the visible center.
COMPAT_MOBS = {
    "Axes": lambda: Axes(x_range=[-1, 1, 1], y_range=[-1, 1, 1]),
    "Star": Star,
    "Arrow": lambda: Arrow(LEFT, RIGHT),
}
EDGES = {"left": LEFT, "right": RIGHT, "up": UP, "down": DOWN}


def _center(mob):
    return mob.get_center().reshape(-1, 3)[0]


def _backing_center(mob):
    return torch.as_tensor(
        mob.get_manim_mobject().get_center(), dtype=torch.get_default_dtype()
    )


def _buffer():
    return torch.tensor(SETTINGS.style.buffer, dtype=torch.get_default_dtype())


def _gap_to_screen_border(scene, mob, edge):
    boundary = mob.get_boundary_in_direction(edge)
    border = scene.camera.project_point_onto_screen_border(boundary, edge)
    return (border - boundary).norm(p=2, dim=-1).reshape(-1)[0]


@pytest.mark.parametrize("name", sorted(COMPAT_MOBS))
def test_move_shifts_the_center_by_exactly_the_displacement(name):
    mob = COMPAT_MOBS[name]()
    displacement = UP * 2 + RIGHT
    start = _center(mob)
    backing_start = _backing_center(mob)

    assert mob.move(displacement) is mob

    torch.testing.assert_close(_center(mob) - start, displacement, atol=2e-5, rtol=0)
    # The backing Mobject has to travel with the Algan geometry, or delegated
    # queries (Axes.c2p, Arrow.get_start, ...) go on reporting the old position.
    torch.testing.assert_close(
        _backing_center(mob) - backing_start, displacement, atol=2e-5, rtol=0
    )


@pytest.mark.parametrize("name", sorted(COMPAT_MOBS))
@pytest.mark.parametrize("edge_name", sorted(EDGES))
def test_move_to_edge_insets_the_boundary_by_the_buffer(name, edge_name):
    scene = SceneManager.instance().current_scene
    edge = EDGES[edge_name]
    mob = COMPAT_MOBS[name]()
    start = _center(mob)

    assert mob.move_to_edge(edge) is mob

    torch.testing.assert_close(
        _gap_to_screen_border(scene, mob, edge), _buffer(), atol=2e-5, rtol=0
    )
    # Moving to an edge may not drag the Mob along the other axes.
    off_axis = edge.abs() < 0.5
    torch.testing.assert_close(
        _center(mob)[off_axis], start[off_axis], atol=2e-5, rtol=0
    )


@pytest.mark.parametrize("name", sorted(COMPAT_MOBS))
def test_move_to_corner_insets_both_boundaries_by_the_buffer(name):
    scene = SceneManager.instance().current_scene
    mob = COMPAT_MOBS[name]()

    assert mob.move_to_corner(UP, RIGHT) is mob

    for edge in (UP, RIGHT):
        torch.testing.assert_close(
            _gap_to_screen_border(scene, mob, edge), _buffer(), atol=2e-5, rtol=0
        )


@pytest.mark.parametrize("name", sorted(COMPAT_MOBS))
@pytest.mark.parametrize("direction_name", sorted(EDGES))
def test_move_next_to_leaves_exactly_the_buffer_gap(name, direction_name):
    direction = EDGES[direction_name]
    target = Square().move_to(RIGHT)
    mob = COMPAT_MOBS[name]()

    assert mob.move_next_to(target, direction) is mob

    torch.testing.assert_close(
        mob.get_boundary_in_direction(-direction),
        target.get_boundary_in_direction(direction) + direction * _buffer(),
        atol=2e-5,
        rtol=0,
    )


@pytest.mark.parametrize("name", sorted(COMPAT_MOBS))
def test_relative_moves_are_recorded_as_animations(name):
    scene = SceneManager.instance().current_scene
    mob = COMPAT_MOBS[name]().spawn(animate=False)
    start = _center(mob)
    displacement = UP * 2

    mob.move(displacement)

    def center_at(time):
        scene.timeline_manager.set_state_to_times(
            torch.tensor([time], dtype=torch.get_default_dtype())
        )
        return _center(mob).clone()

    torch.testing.assert_close(center_at(0.0), start, atol=2e-5, rtol=0)
    torch.testing.assert_close(center_at(1.0), start + displacement, atol=2e-5, rtol=0)
    # Halfway through it must be in transit, not already parked at the target.
    travelled = float((center_at(0.5) - start)[1])
    assert 0 < travelled < float(displacement[1])


@pytest.mark.parametrize("name", sorted(COMPAT_MOBS))
def test_batched_algan_points_are_accepted_where_manim_wants_one_point(name):
    """Algan's ``(1, 1, 3)`` attribute tensors must survive the Manim boundary."""
    mob = COMPAT_MOBS[name]()
    anchor = Square().move_to(UP + RIGHT)
    assert anchor.location.shape == (1, 1, 3)
    anchor_point = anchor.location.reshape(-1, 3)[0]

    assert mob.move_to(anchor.location) is mob
    torch.testing.assert_close(_center(mob), anchor_point, atol=2e-5, rtol=0)

    # Same conversion reached through a Manim-only keyword: a half turn about
    # the point the Mob is centered on is a point reflection, which leaves any
    # shape's bounding-box center exactly where it was.
    assert mob.rotate(math.pi, about_point=anchor.location) is mob
    torch.testing.assert_close(_center(mob), anchor_point, atol=2e-5, rtol=0)

    # And through the generic delegation to an un-overridden Manim method.
    before = _center(mob)
    assert mob.shift(anchor.location) is mob
    torch.testing.assert_close(_center(mob) - before, anchor_point, atol=2e-5, rtol=0)
