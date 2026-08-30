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

import manim as mn
import pytest
import torch

from algan import (
    DOWN,
    LEFT,
    RIGHT,
    SETTINGS,
    UP,
    ManimMob,
    Off,
    SceneManager,
    Square,
    Sync,
    easings,
)
from algan.manim import Arrow, Axes, Star


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
    boundary = mob.get_boundary_point(edge)
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

    assert mob.move_to_screen_edge(edge) is mob

    torch.testing.assert_close(
        _gap_to_screen_border(scene, mob, edge), _buffer(), atol=2e-5, rtol=0
    )
    # Moving to an edge may not drag the Mob along the other axes.
    off_axis = edge.abs() < 0.5
    torch.testing.assert_close(
        _center(mob)[off_axis], start[off_axis], atol=2e-5, rtol=0
    )


@pytest.mark.parametrize("name", sorted(COMPAT_MOBS))
def test_move_to_screen_corner_insets_both_boundaries_by_the_buffer(name):
    scene = SceneManager.instance().current_scene
    mob = COMPAT_MOBS[name]()

    assert mob.move_to_screen_corner((UP, RIGHT)) is mob

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
        mob.get_boundary_point(-direction),
        target.get_boundary_point(direction) + direction * _buffer(),
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

    # Same conversion reached through ``about_point``: a half turn about the
    # point the Mob is centered on is a point reflection, which leaves any
    # shape's bounding-box center exactly where it was.  The angle is in
    # degrees -- ``rotate`` on a compatibility Mob is Algan's, not Manim's.
    assert mob.rotate(180, about=anchor.location) is mob
    torch.testing.assert_close(_center(mob), anchor_point, atol=2e-5, rtol=0)

    # And through the generic delegation to an un-overridden Manim method.
    before = _center(mob)
    assert mob.shift(anchor.location) is mob
    torch.testing.assert_close(_center(mob) - before, anchor_point, atol=2e-5, rtol=0)


# ---------------------------------------------------------------------------
# A parent-driven transform used to desynchronize the backing Mobject, leaving
# the next delegated call to teleport the Mob back. Fixed; these hold it fixed.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(COMPAT_MOBS))
def test_a_direct_move_keeps_the_backing_mobject_in_step(name):
    """The supported path: the compat overrides shift the Manim object too."""
    mob = COMPAT_MOBS[name]()
    mob.move(UP * 1.35)
    torch.testing.assert_close(_center(mob), _backing_center(mob), atol=2e-5, rtol=0)


@pytest.mark.parametrize("name", sorted(COMPAT_MOBS))
def test_a_parent_group_move_keeps_the_backing_mobject_in_step(name):
    from algan import Group

    mob = COMPAT_MOBS[name]()
    Group(mob).move(UP * 1.35)
    torch.testing.assert_close(_center(mob), _backing_center(mob), atol=2e-5, rtol=0)


def test_rotating_after_a_parent_move_does_not_teleport_the_mob():
    from algan import Group

    star = Star()
    Group(star).move(UP * 1.35)
    before = _center(star).clone()
    star.rotate(36)
    # An in-place rotation may shift a Mob whose anchor is off its visual
    # centre, but never by most of the displacement the parent just applied.
    assert float((_center(star) - before).norm()) < 0.2


# ---------------------------------------------------------------------------
# ``rotate`` is a name Algan and Manim both use and disagree about: degrees
# against radians, and opposite ``OUT`` vectors. A compatibility Mob used to
# take Manim's reading, so the documented ``graph.rotate(20, UP)`` turned the
# plot by 20 *radians*, and turned it by morphing between the two poses rather
# than rotating. These pin Algan's reading.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(COMPAT_MOBS))
def test_rotate_measures_degrees_exactly_as_a_plain_manim_mob_does(name):
    """The compatibility subclass may not redefine an inherited Algan verb."""
    compat = COMPAT_MOBS[name]()
    plain = ManimMob(compat.get_manim_mobject().copy())

    with Off():
        compat.rotate(20, UP)
        plain.rotate(20, UP)

    # Compared about each Mob's own centre, because the two disagree about the
    # pivot and only about the pivot: the plain wrapper turns about ``location``
    # and the compatibility Mob about the composite centre Manim would use.
    torch.testing.assert_close(
        compat.control_points.location - compat.get_center(),
        plain.control_points.location - plain.get_center(),
        atol=2e-5,
        rtol=0,
    )


def test_the_default_rotation_agrees_with_manims_despite_the_opposite_axis():
    """Algan's ``OUT`` is Manim's ``-OUT``, and the two wind opposite ways
    around an axis, so the conventions cancel: an unqualified turn of n degrees
    is the same rotation as Manim's turn of n degrees' worth of radians.
    """
    star, expected = Star(), mn.Star()

    with Off():
        star.rotate(37)
    expected.rotate(math.radians(37))

    torch.testing.assert_close(
        star.control_points.location.reshape(-1, 3),
        torch.as_tensor(expected.points, dtype=torch.get_default_dtype()),
        atol=2e-5,
        rtol=0,
    )


def test_rotate_turns_the_mob_rather_than_morphing_between_two_poses():
    """A morph cannot express a full turn: its two endpoints are one shape."""
    scene = SceneManager.instance().current_scene
    star = Star().spawn(animate=False)
    start = star.control_points.location.clone()

    # Linear timing, so a quarter of the run time really is a quarter turn.
    with Sync(duration=1, easing=easings.identity):
        star.rotate(360, UP)

    def points_at(time):
        scene.timeline_manager.set_state_to_times(
            torch.tensor([time], dtype=torch.get_default_dtype())
        )
        return star.control_points.location.clone()

    torch.testing.assert_close(points_at(1.0), start, atol=2e-5, rtol=0)
    # A quarter of the way through, a turn about UP has the star edge-on: its
    # width has collapsed into depth, which no blend of two identical poses
    # could produce.
    quarter = points_at(0.25)
    assert float(quarter[..., 2].abs().max()) > 0.5
    assert float(quarter[..., 0].abs().max()) < 0.1


def test_rotate_pivots_about_the_composite_center_like_manim():
    """An Arrow's location is its shaft's centre; it must still turn in place."""
    arrow = Arrow(LEFT, RIGHT)
    expected = arrow.get_manim_mobject().copy()
    expected.rotate(math.radians(90))

    with Off():
        arrow.rotate(90)

    torch.testing.assert_close(
        _center(arrow),
        torch.as_tensor(expected.get_center(), dtype=torch.get_default_dtype()),
        atol=2e-5,
        rtol=0,
    )
