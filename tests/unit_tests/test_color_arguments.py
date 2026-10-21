"""Every color argument takes every spelling a Mob's ``color`` takes.

``mob.color = ...`` parses a CSS name, a hex string, a hex int, an RGB
sequence and a three-channel tensor through
:func:`~algan.constants.color.to_color`. Several other places a color is handed
over read it as a tensor instead, so only a five-channel ``Color`` -- the named
constants -- was sure to work there: a hex int became a one-channel tensor of
16711680 and drew a blown-out glow, and the other spellings failed with errors
about tensor sizes or a missing ``shape``.

A feature test, outside the fast suite: each case breaks only when its own
entry point changes.
"""

from __future__ import annotations

import pytest
import torch

import algan
import algan.manim as mn
from algan import BLUE, GREEN, Color, DrawBorderThenFill, Indicate, Square, Sync
from algan.constants import easings
from algan.constants.color import to_color
from algan.mobs.shapes_2d import TriangleVertices
from algan.scene_manager import SceneManager

RED_ROW = Color((1.0, 0.0, 0.0)).reshape(5)

SPELLINGS = pytest.mark.parametrize(
    "red",
    [
        0xFF0000,
        (1.0, 0.0, 0.0),
        [1.0, 0.0, 0.0],
        "#FF0000",
        "red",
        torch.tensor([1.0, 0.0, 0.0]),
    ],
    ids=["hex int", "tuple", "list", "hex string", "css name", "rgb tensor"],
)


@pytest.fixture(autouse=True)
def _fresh_scenes():
    SceneManager.reset()
    yield
    SceneManager.reset()


def materialize(*times: float):
    SceneManager.instance().current_scene.timeline_manager.set_state_to_times(
        torch.tensor(times, dtype=torch.get_default_dtype())
    )


def _first_row(color):
    return color.reshape(-1, color.shape[-1])[0]


def _square():
    return Square(color=BLUE, add_to_scene=False).spawn(False)


@SPELLINGS
def test_pulse_color_takes_every_spelling(red):
    """REGRESSION. A hex int pulsed every channel, glow and alpha included, to
    16711680; an RGB sequence or tensor failed with "The size of tensor a (5)
    must match the size of tensor b (3)", and a string with "'str' object has
    no attribute 'shape'".
    """
    square = _square()
    with Sync(easing=easings.identity):
        square.pulse_color(red)
    materialize(0.5)
    torch.testing.assert_close(_first_row(square.color), RED_ROW)


@SPELLINGS
def test_pulse_color_settles_on_every_spelling(red):
    square = _square()
    with Sync(easing=easings.identity):
        square.pulse_color(GREEN, new_color=red)
    materialize(1.0)
    torch.testing.assert_close(_first_row(square.color), RED_ROW)


@SPELLINGS
def test_indicate_flashes_every_spelling(red):
    """REGRESSION. ``Indicate(mob, color=...)`` hands its color to
    :meth:`~algan.animatable_base.mob.Mob.pulse_color`, and failed the same way.
    """
    square = _square()
    Indicate(square, color=red, runtime=1.0)
    materialize(0.5)
    torch.testing.assert_close(_first_row(square.color), RED_ROW)


@SPELLINGS
def test_wave_color_takes_every_spelling(red):
    """The wave pulses each part through ``pulse_color``, so it draws exactly
    what the ``Color`` spelling draws, at every point of the sweep.
    """
    times = (0.5, 1.0, 1.5, 2.0)

    def swept(color):
        SceneManager.reset()
        square = _square()
        square.wave_color(color, refine_resolution=False)
        materialize(*times)
        return square.color.clone()

    reference = swept(Color((1.0, 0.0, 0.0)))
    assert not torch.allclose(reference[0], reference[1]), "the wave should move"
    torch.testing.assert_close(swept(red), reference)


@SPELLINGS
def test_draw_border_then_fill_takes_every_stroke_spelling(red):
    """REGRESSION. The outline color was handed to the opacity helper as
    given, and anything but a tensor failed with "'int' object has no
    attribute 'clone'" (or 'tuple', or 'str').
    """
    square = _square()
    DrawBorderThenFill([square], stroke_color=red)
    materialize(0.3)
    torch.testing.assert_close(_first_row(square.stroke_color), RED_ROW)


@pytest.mark.parametrize("red", [0xFF0000, (1.0, 0.0, 0.0)], ids=["hex int", "tuple"])
@pytest.mark.parametrize("shape", [Square, algan.Circle, algan.Line])
def test_a_shape_takes_a_hex_int_stroke_color(shape, red):
    """REGRESSION. ``Square(stroke_color=0xFF0000)`` stored 16711680 in every
    channel, and drew an outline glowing sixteen million times over.
    """
    mob = shape(stroke_color=red, add_to_scene=False)
    torch.testing.assert_close(_first_row(mob.stroke_color), RED_ROW)


@pytest.mark.parametrize("name", ["red", "teal", "RED"])
@pytest.mark.parametrize(
    ("build", "read"),
    [
        (lambda name: Square(color=name), lambda mob: mob.color),
        (lambda name: Square(color=name, fill_opacity=0.5), lambda mob: mob.color),
        (lambda name: Square(stroke_color=name), lambda mob: mob.stroke_color),
        (lambda name: algan.Line(color=name), lambda mob: mob.stroke_color),
        (lambda name: algan.Circle(stroke_color=name), lambda mob: mob.stroke_color),
    ],
    ids=["fill", "faded fill", "stroke", "line", "circle stroke"],
)
def test_a_color_name_means_one_color_in_every_shape_keyword(build, read, name):
    """REGRESSION. A shape's style keywords looked a name up among the palette
    constants before parsing it, so "red" was Manim's #FC6255 there and CSS red
    everywhere else: ``Square(color="red")`` changed hue when given a
    ``fill_opacity``, and ``Line(color="red")`` was not a red Square's red.
    """
    color = _first_row(read(build(name)))
    torch.testing.assert_close(color[:3], to_color(name).reshape(5)[:3])


def test_an_empty_circuit_takes_a_string_stroke_color():
    """REGRESSION. An ``empty`` circuit hides its stroke by setting its alpha,
    which a string has no method for: "'str' object has no attribute
    'as_subclass'".
    """
    circuit = algan.BezierCircuitCubic(
        Square(add_to_scene=False).control_points.location,
        empty=True,
        stroke_color="#FF0000",
        add_to_scene=False,
    )
    stroke = _first_row(circuit.stroke_color)
    torch.testing.assert_close(stroke[:4], RED_ROW[:4])
    assert float(stroke[4]) == 0.0


POINTS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
RED_RGBA = torch.tensor([1.0, 0.0, 0.0, 1.0])


def _line_cloud(red):
    # Manim's add_line returns None, and so does this one.
    cloud = mn.Mobject1D()
    cloud.add_line(*POINTS, color=red)
    return cloud


@SPELLINGS
@pytest.mark.parametrize(
    "paint",
    [
        lambda red: mn.PMobject(points=POINTS, color=red),
        lambda red: mn.PMobject().add_points(POINTS, color=red),
        lambda red: mn.PMobject(points=POINTS).set_color(red),
        lambda red: mn.PMobject(points=POINTS).set_color_by_gradient(red, BLUE),
        _line_cloud,
    ],
    ids=["constructor", "add_points", "set_color", "gradient", "add_line"],
)
def test_a_point_cloud_takes_every_spelling(paint, red):
    """REGRESSION. A hex int became a two-wide RGBA, and a string failed with
    "'str' object has no attribute 'to'".
    """
    cloud = paint(red)
    torch.testing.assert_close(cloud.rgbas[0], RED_RGBA)


@SPELLINGS
@pytest.mark.parametrize(
    "solid",
    [algan.Tetrahedron, algan.Octahedron, algan.Icosahedron, algan.Dodecahedron],
)
def test_a_polyhedron_takes_every_spelling(solid, red):
    """REGRESSION. The faces' vertex Mob reshaped ``color`` before anything
    parsed it, so only a tensor got through: anything else failed with
    "'str' object has no attribute 'reshape'" (or 'int', or 'tuple').
    """
    mob = solid(color=red, add_to_scene=False)
    # The faces; the vertex-and-edge graph keeps its own graph_config colors.
    faces = [
        part for part in mob.get_descendants() if isinstance(part, TriangleVertices)
    ]
    assert faces
    for face in faces:
        torch.testing.assert_close(_first_row(face.color), RED_ROW)


@pytest.mark.parametrize("shape", [algan.Star, algan.Arc, algan.NumberLine])
def test_a_manim_shape_takes_a_bare_rgb_tensor(shape):
    """REGRESSION. The Manim layer widens a bare RGB tensor to five channels,
    but converts only a ``Color`` back into a Manim color, so the five numbers
    reached ``ManimColor`` as they were and it refused them.
    """
    mob = shape(color=torch.tensor([1.0, 0.0, 0.0]), add_to_scene=False)
    reference = shape(color=Color((1.0, 0.0, 0.0)), add_to_scene=False)
    for part, expected in zip(mob.get_descendants(), reference.get_descendants()):
        torch.testing.assert_close(part.color, expected.color)


@pytest.mark.parametrize(
    ("color", "expected"),
    [
        (lambda: Color(algan.YELLOW, glow=2), [1.0, 1.0, 0.0, 2.0, 1.0]),
        (lambda: Color(algan.YELLOW, opacity=0.3), [1.0, 1.0, 0.0, 0.0, 0.3]),
        (lambda: Color((1, 0, 0, 0.5), opacity=0.2), [1.0, 0.0, 0.0, 0.0, 0.2]),
        (lambda: Color((1, 0, 0, 2, 0.5)), [1.0, 0.0, 0.0, 2.0, 0.5]),
        (lambda: Color("#FF000080"), [1.0, 0.0, 0.0, 0.0, 128 / 255]),
        (lambda: Color((1, 0, 0)), [1.0, 0.0, 0.0, 0.0, 1.0]),
    ],
)
def test_glow_and_opacity_keywords_override_a_five_channel_color(color, expected):
    """REGRESSION. Both keywords were dropped when ``rgb`` already carried
    them, so ``Color(YELLOW, glow=2)`` built a color with no glow at all.
    """
    torch.testing.assert_close(
        color().as_subclass(torch.Tensor).reshape(-1), torch.tensor(expected)
    )
