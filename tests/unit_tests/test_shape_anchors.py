"""Where a shape's ``location`` sits, and therefore what it turns about.

``location`` is not just where a Mob is: it is the point
:meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.rotate` and
:meth:`~algan.animatable_base.mob_transform.MobTransformMixin.scale` work about,
so a shape anchored anywhere but its own centroid orbits that point instead of
turning in place. A ``Triangle``'s bounding box is centred a quarter of a unit
above its centroid, which was enough to make a spinning triangle look like it
was also drifting upward.

These tests pin the anchor for the shapes where the two differ, and pin that
they still coincide for the symmetric shapes -- where a drifting anchor would
move geometry that has no reason to move.
"""

from __future__ import annotations

import math

import pytest
import torch

from algan import (
    Annulus,
    Circle,
    Cube,
    Line,
    Polygon,
    Polyhedron,
    Prism,
    Rectangle,
    RegularPolygon,
    Square,
    Triangle,
)
from algan.constants.spatial import LEFT, OUT, RIGHT

# In the fast suite: the anchor is read by every transform and by the frame the
# texture grid is laid out on, so a change to it moves output far from here.
pytestmark = pytest.mark.fast


def _location(mob):
    return mob.location.reshape(-1, 3)[0]


def _bounding_box_centre(mob):
    box = mob.get_bounding_box().reshape(-1, 3)
    return (box.amin(-2) + box.amax(-2)) * 0.5


def test_a_triangle_is_anchored_at_its_centroid_not_its_box():
    """The case the whole change is about.

    A default ``Triangle`` has vertices at radius 1 about the origin, so its
    centroid is the origin while its box -- y from -0.5 to 1 -- is centred a
    quarter of a unit above it.
    """
    triangle = Triangle()
    assert _location(triangle) == pytest.approx([0.0, 0.0, 0.0], abs=1e-5)
    assert float(_bounding_box_centre(triangle)[1]) == pytest.approx(0.25, abs=1e-3)


def test_a_spinning_triangle_stays_where_it_is():
    """Rotating in place must not move what the viewer sees.

    Half a turn about an anchor a quarter of a unit off centre used to leave the
    triangle half a unit higher than it started, which is what an updater
    spinning one looked like: a shape drifting upward while it was only being
    moved sideways.
    """
    from algan import Off, Scene

    def centroid(mob):
        # Measured from the live control points, so this is where the shape
        # actually balances now -- not where it was anchored when it was built.
        # (Its bounding box is no use here: half a turn genuinely puts a
        # triangle point-down, which moves the box even about a perfect pivot.)
        from algan.mobs.bezier_circuit import _circuit_centroid

        points = mob.control_points.location.reshape(-1, 3)
        return _circuit_centroid(points, (points.amin(-2) + points.amax(-2)) * 0.5)

    with Scene():
        triangle = Triangle().spawn()
        before = centroid(triangle).clone()
        with Off():
            triangle.rotate(180, OUT)
        after = centroid(triangle)

    assert before == pytest.approx([0.0, 0.0, 0.0], abs=1e-4)
    assert after == pytest.approx(before, abs=1e-4)


@pytest.mark.parametrize(
    "factory",
    [Square, Circle, Rectangle, Annulus, lambda: RegularPolygon(n=6), Cube, Prism],
    ids=["square", "circle", "rectangle", "annulus", "hexagon", "cube", "prism"],
)
def test_a_symmetric_shape_is_anchored_at_its_own_centre(factory):
    """A point-symmetric shape's centroid is its centre, exactly.

    The centroid is measured rather than derived, so this is also what says the
    measurement does not introduce a wobble of its own: an anchor an ulp off
    centre would move the frame every shape's texture grid is laid out on.
    """
    assert _location(factory()) == pytest.approx([0.0, 0.0, 0.0], abs=1e-5)


def test_a_polygon_is_anchored_at_its_area_centroid():
    """An L, whose centroid is nowhere near the middle of its box.

    Its two rectangles -- 2 x 0.5 along the bottom and 0.5 x 1.5 up the left --
    put the centroid at 19/28 on both axes, while the box's centre is at 1.
    """
    corners = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    expected = 19 / 28
    assert _location(Polygon(corners)) == pytest.approx(
        [expected, expected, 0.0], abs=1e-4
    )


def test_a_regular_polygon_with_an_odd_number_of_sides_is_centred():
    """Odd sides are what makes a box centre and a centroid disagree."""
    for sides in (3, 5, 7, 9):
        assert _location(RegularPolygon(n=sides)) == pytest.approx(
            [0.0, 0.0, 0.0], abs=1e-5
        ), f"a {sides}-sided polygon is off centre"


def test_a_line_is_anchored_at_its_midpoint():
    """A straight path encloses no area, so it falls back to its own centroid.

    Sampling a path has to reach its far end to get this right: dropping the
    last point of an open path puts the anchor a sixteenth of the way short.
    """
    line = Line(LEFT, RIGHT * 3)
    assert _location(line) == pytest.approx([1.0, 0.0, 0.0], abs=1e-4)

    diagonal = Line(torch.zeros(3), torch.tensor([4.0, 2.0, 0.0]))
    assert _location(diagonal) == pytest.approx([2.0, 1.0, 0.0], abs=1e-4)


def test_an_arc_is_anchored_inside_the_region_it_sweeps():
    """A curved path is measured, not approximated by its control points.

    A half disc of radius 1 balances at 4 / (3 pi) from the middle of its
    straight edge, which no combination of its control points lands on.
    """
    from algan.mobs.bezier_circuit import _circuit_centroid

    steps = 64
    angles = torch.linspace(0.0, math.pi, steps + 1)
    rim = torch.stack((angles.cos(), angles.sin(), torch.zeros_like(angles)), dim=-1)
    # Straight cubic segments through the rim points and back along the
    # diameter: the shape is a half disc to within the sampling of its rim.
    corners = torch.cat((rim, torch.zeros(1, 3)), dim=0)
    control_points = torch.cat(
        [
            torch.stack([start * (1 - a) + end * a for a in torch.linspace(0, 1, 4)])
            for start, end in zip(corners, corners.roll(-1, 0))
        ]
    )
    box_centre = (control_points.amin(-2) + control_points.amax(-2)) * 0.5
    centroid = _circuit_centroid(control_points, box_centre)
    assert float(centroid[0]) == pytest.approx(0.0, abs=1e-3)
    assert float(centroid[1]) == pytest.approx(4 / (3 * math.pi), abs=1e-3)


def test_a_polyhedron_is_anchored_on_itself_and_not_on_the_world_origin():
    """Vertices are world coordinates, so the anchor has to be derived.

    Left to the Mob default it stayed at the origin, and a solid built away from
    the origin then turned about the origin rather than about itself.
    """
    tetrahedron = Polyhedron(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
    )
    assert _location(tetrahedron) == pytest.approx([0.5, 0.5, 0.5], abs=1e-5)


def test_an_explicit_location_still_wins():
    """The derived anchor is a default, not an override of the caller."""
    tetrahedron = Polyhedron(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
        location=RIGHT * 2,
    )
    assert _location(tetrahedron) == pytest.approx([2.0, 0.0, 0.0], abs=1e-5)
