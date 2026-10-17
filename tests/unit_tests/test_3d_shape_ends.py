"""Where a cone, a cylinder and a 3-D line say their ends are, and where they put them.

``Cone.get_start``/``get_end``/``get_direction`` and ``Line3D.get_start``/
``get_end`` report a solid's ends; ``Cylinder.move_between_points``,
``set_start_point``, ``set_end_point``, ``set_direction`` and ``add_bases``
place a tube's ends; ``Line3D.parallel_to``/``perpendicular_to`` build a line
from another. Every expectation is checked against the vertices the surface
actually carries, not against the formulas that placed them, so a getter and
the geometry cannot agree on the same wrong answer.

A feature test, so it sits outside the fast suite: it only breaks when these
shapes themselves change.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from algan import (
    DOWN,
    LEFT,
    OUT,
    RIGHT,
    UP,
    Arrow3D,
    Cone,
    Cylinder,
    Dot3D,
    Line3D,
    Off,
    Scene,
    Sync,
    easings,
)

ATOL = 1e-4


@pytest.fixture
def scene():
    with Scene() as active:
        yield active


def _point(value) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.get_default_dtype()).reshape(3)


def _flat(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(-1, 3)[-1]


def _assert_point(actual, expected):
    torch.testing.assert_close(_flat(actual), _point(expected), atol=ATOL, rtol=0)


def _end_rings(mob, axis):
    """The surface's vertices at each end of ``axis``: ``(at start, at end)``.

    Picked by where they lie rather than by grid index, so the grid's own
    layout -- which uv component runs along the axis -- does not matter.
    """
    axis = F.normalize(_point(axis), dim=0)
    points = mob.grid.location.reshape(-1, 3)
    along = points @ axis
    return (
        points[along < along.min() + ATOL],
        points[along > along.max() - ATOL],
    )


def _assert_ring(ring, centre, axis, radius):
    """``ring`` is a circle of ``radius`` round ``centre``, square to ``axis``."""
    offsets = ring - _point(centre)
    axis = F.normalize(_point(axis), dim=0)
    assert len(ring) >= 3
    torch.testing.assert_close(
        offsets.norm(dim=-1), torch.full((len(ring),), float(radius)), atol=ATOL, rtol=0
    )
    torch.testing.assert_close(
        offsets @ axis, torch.zeros(len(ring)), atol=ATOL, rtol=0
    )


# --- Cone -------------------------------------------------------------------


def test_a_cone_reports_its_base_and_tip(scene):
    direction = F.normalize(torch.tensor([1.0, 2.0, 3.0]), dim=0)
    cone = Cone(radius=0.5, height=2, direction=direction * 3)

    _assert_point(cone.get_start(), -direction)
    _assert_point(cone.get_end(), direction)
    # A direction, not the vector it was given as.
    _assert_point(cone.get_direction(), direction)

    base, tip = _end_rings(cone, direction)
    _assert_ring(base, -direction, direction, 0.5)
    _assert_point(tip.mean(0), direction)


def test_a_cones_ends_follow_it_as_it_moves_turns_and_scales(scene):
    """REGRESSION. The ends were stored when the cone was built, so a moved,
    turned or scaled cone -- an Arrow3D's tip among them -- still reported the
    base and tip it was built with, and the direction it first pointed.
    """
    cone = Cone(radius=0.5, height=2)
    cone.scale(2)
    cone.rotate(90, OUT)
    cone.move(RIGHT * 3)

    # Twice as tall, lying along -x, centred at x = 3.
    _assert_point(cone.get_start(), RIGHT * 5)
    _assert_point(cone.get_end(), RIGHT)
    _assert_point(cone.get_direction(), LEFT)

    base, tip = _end_rings(cone, LEFT)
    _assert_ring(base, RIGHT * 5, LEFT, 1.0)
    _assert_point(tip.mean(0), RIGHT)


def test_a_cones_ends_follow_a_recorded_move(scene):
    """REGRESSION. See ``test_a_cones_ends_follow_it_as_it_moves_turns_and_scales``:
    read back mid-animation, the stored ends did not move at all.
    """
    cone = Cone(height=2)
    with Off():
        cone.spawn()
    with Sync(runtime=1, easing=easings.identity):
        cone.move(RIGHT * 2)

    scene.timeline_manager.set_state_to_times(torch.tensor([0.0, 0.5, 1.0]))
    ends = cone.get_end().reshape(3, 3)
    torch.testing.assert_close(
        ends,
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]),
        atol=ATOL,
        rtol=0,
    )


def test_an_arrows_tip_ends_where_the_arrow_does(scene):
    """REGRESSION. ``arrow.cone`` is a Cone, so it inherited the stale ends:
    once the arrow moved, its own tip disagreed with ``arrow.get_end()``.
    """
    arrow = Arrow3D(start=LEFT, end=RIGHT, tip_length=0.4)
    with Off():
        arrow.move(UP * 2)
        arrow.rotate(90, OUT)

    _assert_point(arrow.get_end(), UP * 3)
    _assert_point(arrow.cone.get_end(), arrow.get_end())
    _assert_point(arrow.cone.get_start(), UP * 2.6)
    _assert_point(arrow.cone.get_direction(), arrow.get_unit_vector())


# --- Cylinder ---------------------------------------------------------------


@pytest.mark.parametrize("height", [1.0, 2.0, 0.5])
def test_move_between_points_spans_exactly_those_points(scene, height):
    """REGRESSION (for any height but 1). The tube's axis was set to the full
    span while its side is built ``height`` times along that axis, so a
    cylinder not one unit long came out ``height`` times too long: two units
    tall and moved between points two apart, it spanned four.
    """
    cylinder = Cylinder(radius=0.2, height=height)
    cylinder.move_between_points(LEFT + DOWN, RIGHT + UP)

    axis = RIGHT + UP
    start, end = _end_rings(cylinder, axis)
    _assert_ring(start, LEFT + DOWN, axis, 0.2)
    _assert_ring(end, RIGHT + UP, axis, 0.2)


@pytest.mark.parametrize("height", [2.0, 0.5])
def test_moving_one_end_leaves_the_other_where_it_was(scene, height):
    """REGRESSION. ``set_start_point`` and ``set_end_point`` took the ends to
    be half the axis either way of the centre, which is only so for a
    cylinder one unit long: on any other, the end meant to stay put moved.
    """
    half = height / 2

    lengthened = Cylinder(radius=0.2, height=height)
    lengthened.set_end_point(UP * 3)
    start, end = _end_rings(lengthened, UP)
    _assert_ring(start, DOWN * half, UP, 0.2)
    _assert_ring(end, UP * 3, UP, 0.2)

    lowered = Cylinder(radius=0.2, height=height)
    lowered.set_start_point(DOWN * 3)
    start, end = _end_rings(lowered, UP)
    _assert_ring(start, DOWN * 3, UP, 0.2)
    _assert_ring(end, UP * half, UP, 0.2)


def test_set_direction_turns_the_tube_in_place_without_resizing_it(scene):
    """REGRESSION. The new ends were laid out from ``height`` alone, so a
    cylinder whose height was not 1 came out ``height`` times too long, and
    one scaled since it was built went back to its built length. They were
    laid out about the centre of its bounding box, too, which the vertices
    round a ring put a little off the axis, so the tube crept sideways.
    """
    tall = Cylinder(radius=0.2, height=2)
    tall.set_direction(RIGHT)
    start, end = _end_rings(tall, RIGHT)
    _assert_ring(start, LEFT, RIGHT, 0.2)
    _assert_ring(end, RIGHT, RIGHT, 0.2)

    scaled = Cylinder(radius=0.2, height=1)
    scaled.scale(2)
    scaled.set_direction(RIGHT)
    start, end = _end_rings(scaled, RIGHT)
    _assert_ring(start, LEFT, RIGHT, 0.4)
    _assert_ring(end, RIGHT, RIGHT, 0.4)


def test_capping_a_moved_and_turned_tube_caps_it_where_it_is(scene):
    """REGRESSION. ``add_bases`` placed the end discs from the direction and
    height the tube was built with, about the origin, so a tube moved or
    turned before it was capped had its caps floating where it had been built.
    """
    cylinder = Cylinder(radius=0.2, height=1.5)
    cylinder.rotate(90, OUT)
    cylinder.move(RIGHT * 2)
    cylinder.add_bases()

    # Lying along -x now, so its start is the +x end.
    _assert_point(cylinder.bottom_cap.location, RIGHT * 2.75)
    _assert_point(cylinder.top_cap.location, RIGHT * 1.25)
    start, end = _end_rings(cylinder, LEFT)
    _assert_ring(start, RIGHT * 2.75, LEFT, 0.2)
    _assert_ring(end, RIGHT * 1.25, LEFT, 0.2)


def test_a_capped_tube_keeps_its_caps_on_its_ends_as_it_moves(scene):
    cylinder = Cylinder(radius=0.2, height=1, closed=True)
    cylinder.move_between_points(DOWN, UP * 2)

    _assert_point(cylinder.bottom_cap.location, DOWN)
    _assert_point(cylinder.top_cap.location, UP * 2)


# --- Line3D -----------------------------------------------------------------


def test_a_line_reports_the_points_it_spans(scene):
    line = Line3D(start=LEFT + DOWN, end=RIGHT + UP * 2, radius=0.05)

    _assert_point(line.get_start(), LEFT + DOWN)
    _assert_point(line.get_end(), RIGHT + UP * 2)
    start, end = _end_rings(line, RIGHT * 2 + UP * 3)
    _assert_ring(start, LEFT + DOWN, RIGHT * 2 + UP * 3, 0.05)
    _assert_ring(end, RIGHT + UP * 2, RIGHT * 2 + UP * 3, 0.05)

    # What comes back is the caller's to change.
    line.get_start().add_(10)
    _assert_point(line.get_start(), LEFT + DOWN)


def test_a_lines_ends_follow_it_as_it_moves_and_turns(scene):
    """REGRESSION. The ends were the points the line was last stretched
    between, so a moved or turned line still reported where it had been.
    """
    line = Line3D(start=LEFT, end=RIGHT, radius=0.05)
    line.move(UP)
    _assert_point(line.get_start(), LEFT + UP)
    _assert_point(line.get_end(), RIGHT + UP)

    line.rotate(90, OUT)
    _assert_point(line.get_start(), UP * 0)
    _assert_point(line.get_end(), UP * 2)


def test_a_line_can_be_stretched_between_mobs(scene):
    # A Mob stands for its centre: the middle of its bounding box, which on a
    # tessellated sphere is near its location but not exactly on it.
    dots = Dot3D(point=LEFT * 2), Dot3D(point=UP * 3)
    line = Line3D(radius=0.05).set_start_and_end_attrs(*dots)

    _assert_point(line.get_start(), _flat(dots[0].get_center()))
    _assert_point(line.get_end(), _flat(dots[1].get_center()))


def test_parallel_to_runs_the_same_way_centred_on_the_point(scene):
    base = Line3D(start=LEFT, end=RIGHT + UP * 2, radius=0.05)
    point = OUT * 2
    parallel = Line3D.parallel_to(base, point, length=3)

    direction = F.normalize(_point(RIGHT * 2 + UP * 2), dim=0)
    _assert_point(parallel.get_start(), _point(point) - direction * 1.5)
    _assert_point(parallel.get_end(), _point(point) + direction * 1.5)

    dot = Dot3D(point=DOWN)
    through_dot = Line3D.parallel_to(base, dot, length=3)
    centre = _flat(dot.get_center())
    _assert_point(through_dot.get_start(), centre - direction * 1.5)
    _assert_point(through_dot.get_end(), centre + direction * 1.5)


@pytest.mark.parametrize(
    "point",
    [UP * 2, OUT * 2, RIGHT * 0.5 + UP + OUT, LEFT * 3 + DOWN - OUT * 2],
    ids=["above", "in-front", "diagonal", "beyond-the-start"],
)
def test_perpendicular_to_meets_the_line_square_on(scene, point):
    """REGRESSION. The direction was any perpendicular to the line, whatever
    the point, so the result generally passed the line by without meeting it:
    from a point in front of an x-axis line it ran up the screen, not back to
    the axis.
    """
    base = Line3D(start=LEFT, end=RIGHT, radius=0.05)
    perpendicular = Line3D.perpendicular_to(base, point, length=2)

    start, end = _flat(perpendicular.get_start()), _flat(perpendicular.get_end())
    _assert_point((start + end) / 2, point)
    assert float((end - start).norm()) == pytest.approx(2, abs=ATOL)
    along = F.normalize(end - start, dim=0)
    # Square to the line, and on a line through the foot of the point on it.
    assert float(along @ _point(RIGHT)) == pytest.approx(0, abs=ATOL)
    foot = _point(point) * _point(RIGHT)
    torch.testing.assert_close(
        torch.linalg.cross(along, _point(point) - foot),
        torch.zeros(3),
        atol=ATOL,
        rtol=0,
    )


def test_perpendicular_to_from_a_point_on_the_line_is_still_square_to_it(scene):
    base = Line3D(start=LEFT, end=RIGHT, radius=0.05)
    perpendicular = Line3D.perpendicular_to(base, RIGHT * 0.25, length=1)

    start, end = _flat(perpendicular.get_start()), _flat(perpendicular.get_end())
    _assert_point((start + end) / 2, RIGHT * 0.25)
    assert float((end - start).norm()) == pytest.approx(1, abs=ATOL)
    assert float(F.normalize(end - start, dim=0) @ _point(RIGHT)) == pytest.approx(
        0, abs=ATOL
    )
