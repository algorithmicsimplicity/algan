"""Accessors that read a shape's geometry where it is now, and the setters beside them."""

import inspect

import pytest
import torch

from algan import (
    LEFT,
    OUT,
    RED,
    RIGHT,
    UP,
    Circle,
    Cone,
    Dot,
    Line,
    Line3D,
    Off,
    Scene,
)
from algan.errors import AlganConfigurationError, AlganWarning
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.mobs.image_mob import ImageMob


def _width(mob):
    points = mob.control_points.location.reshape(-1, 3)
    return float(points[:, 0].max() - points[:, 0].min())


def test_circle_radius_reads_and_sets_the_drawn_radius():
    with Scene():
        circle = Circle(radius=1.5).spawn()
        assert float(circle.radius) == pytest.approx(1.5, abs=1e-4)
        assert _width(circle) == pytest.approx(3.0, abs=1e-3)
        with Off():
            circle.radius = 2
        assert float(circle.radius) == pytest.approx(2, abs=1e-4)
        assert _width(circle) == pytest.approx(4.0, abs=1e-3)


def test_circle_radius_assignment_is_animated():
    with Scene() as scene:
        circle = Circle(radius=1).spawn()
        start = scene.animation_manager.context.current_time
        circle.radius = 2
        assert scene.animation_manager.context.current_time - start == pytest.approx(1)


def test_arc_endpoints_are_the_arc_ends():
    with Scene():
        arc = Line(LEFT, RIGHT, path_arc=90).spawn()
        assert arc.get_start().reshape(-1, 3).shape[0] == 1
        torch.testing.assert_close(arc.get_start().reshape(-1), LEFT, atol=1e-5, rtol=0)
        torch.testing.assert_close(arc.get_end().reshape(-1), RIGHT, atol=1e-5, rtol=0)
        assert float(arc.get_length().reshape(-1)[0]) == pytest.approx(2, abs=1e-5)


def test_cone_ends_follow_the_cone():
    with Scene():
        cone = Cone(height=2, direction=RIGHT).spawn()
        torch.testing.assert_close(
            cone.get_start().reshape(-1), LEFT, atol=1e-5, rtol=0
        )
        with Off():
            cone.move(UP * 3)
            cone.rotate(90, OUT)
        torch.testing.assert_close(
            cone.get_start().reshape(-1), UP * 2, atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            cone.get_end().reshape(-1), UP * 4, atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            cone.get_direction().reshape(-1), UP, atol=1e-5, rtol=0
        )


def test_line3d_ends_follow_the_line():
    with Scene():
        line = Line3D(LEFT, RIGHT * 2).spawn()
        torch.testing.assert_close(
            line.get_start().reshape(-1), LEFT, atol=1e-5, rtol=0
        )
        with Off():
            line.move(UP)
        torch.testing.assert_close(
            line.get_start().reshape(-1), LEFT + UP, atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            line.get_end().reshape(-1), RIGHT * 2 + UP, atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            (line.get_end() - line.get_start()).reshape(-1),
            RIGHT * 3,
            atol=1e-5,
            rtol=0,
        )


@pytest.mark.parametrize("value", [(0.1, 0.2, 0.3), [0.1, 0.2, 0.3]])
def test_color_channels_accept_sequences(value):
    color = RED.set_rgb(value)
    torch.testing.assert_close(
        color.rgb.reshape(-1), torch.tensor(value), atol=1e-6, rtol=0
    )


def test_scene_takes_no_memory_argument():
    assert "memory" not in inspect.signature(Scene.__init__).parameters
    with pytest.raises(TypeError):
        Scene(memory=object())


@pytest.mark.parametrize(
    ("cls", "name"),
    [(BezierCircuitCubic, "get_animatable_attrs"), (ImageMob, "setattr_absolute")],
)
def test_methods_that_always_raised_are_gone(cls, name):
    assert not hasattr(cls, name)


@pytest.mark.parametrize("make", [lambda: Circle(radius=0), lambda: Dot(radius=0)])
def test_a_zero_radius_circle_warns_that_it_cannot_grow(make):
    # Built with all-zero control points and basis, it could never be resized:
    # ``circle.radius = 1`` left it at 0 without a word.
    with Scene(), pytest.warns(AlganWarning, match="radius 0") as caught:
        make()
    assert caught[0].filename == __file__


def test_a_negative_radius_circle_warns_that_it_faces_inward():
    with Scene(), pytest.warns(AlganWarning, match="negative radius"):
        circle = Circle(radius=-1)
    assert float(circle.radius) == pytest.approx(1.0)  # built as it always was


def test_a_positive_radius_circle_does_not_warn():
    import warnings

    with Scene(), warnings.catch_warnings():
        warnings.simplefilter("error", AlganWarning)
        Circle(radius=0.5)
        Dot()


@pytest.mark.parametrize("path_arc", [360, -360, 720, 1080])
def test_a_full_turn_line_arc_is_refused(path_arc):
    # The arc's radius divides by sin(path_arc / 2): a whole number of turns
    # built a "line" with coordinates around 1e16.
    with Scene(), pytest.raises(AlganConfigurationError, match="path_arc"):
        Line(LEFT, RIGHT, path_arc=path_arc)


@pytest.mark.parametrize("path_arc", [60, 359, -270, 400])
def test_other_line_arcs_still_build_finite_points(path_arc):
    with Scene():
        line = Line(LEFT, RIGHT, path_arc=path_arc)
    points = line.control_points.location
    assert bool(points.isfinite().all())
    assert float(points.abs().max()) < 1e3


def test_non_finite_control_points_are_refused_by_name():
    # They failed deep inside the frame derivation with
    # "IndexError: index 4 is out of bounds for axis 1 with size 4".
    points = torch.zeros(4, 3)
    points[1, 0] = float("nan")
    with Scene(), pytest.raises(AlganConfigurationError, match="control_points"):
        BezierCircuitCubic(points)
