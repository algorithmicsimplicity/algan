"""Manim idioms a ported script writes on a compatibility Mob, answered usefully.

Each case is a line verbatim Manim code is full of -- ``x.animate``,
``b.set_color(a.get_color())``, a corner radius taken from a width -- that the
compatibility layer used to answer with an opaque error or a silently wrong
picture. A feature test, outside the fast suite: each case breaks only when the
compatibility layer changes.
"""

from __future__ import annotations

import pytest
import torch

import algan
import algan.manim as mn
from algan import RED, Arc, Axes, Square
from algan.scene_manager import SceneManager


@pytest.fixture(autouse=True)
def _fresh_scenes():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _row(color):
    return color.reshape(-1, color.shape[-1])[0]


@pytest.mark.parametrize(
    "build",
    [lambda: Arc(angle=90), lambda: Axes(), lambda: mn.Square()],
    ids=["root Arc", "root Axes", "mn.Square"],
)
def test_animate_names_the_algan_spelling(build):
    """REGRESSION. ``.animate`` was delegated to Manim's ``_AnimationBuilder``,
    which ``functools.wraps`` could not wrap: ``TypeError: __annotations__ must
    be set to a dict object``. A native Square answers with the migration hint.
    """
    mob = build()
    with pytest.raises(AttributeError, match="Algan records animations"):
        mob.animate
    assert not hasattr(mob, "animate")


def test_matching_an_unfilled_mobs_color_keeps_the_target_visible():
    """REGRESSION. ``get_color`` on an unfilled compatibility Mob returned its
    fill row, whose alpha is the fill opacity of 0, so ``set_color`` with it
    made the target's stroke invisible.
    """
    arc = Arc(angle=90, color=RED)
    color = arc.get_color()
    torch.testing.assert_close(_row(color)[:3], _row(arc.stroke_color)[:3])
    assert float(_row(color)[4]) == 1.0
    square = mn.Square()
    square.set_color(color)
    stroke = _row(square.stroke_color)
    torch.testing.assert_close(stroke[:3], _row(RED)[:3])
    assert float(stroke[4]) == 1.0


def test_get_color_of_a_filled_mob_is_its_fill():
    square = mn.Square(color="#58C4DD", fill_color="#FC6255", fill_opacity=0.5)
    color = _row(square.get_color())
    torch.testing.assert_close(color[:3], _row(square.color)[:3])
    assert float(color[4]) == 1.0


def test_a_scalar_from_algan_reaches_manim_as_a_number():
    """REGRESSION. A ``(1, 1, 1)`` tensor such as ``get_width()`` reached Manim
    as a one-element array, and ``RoundedRectangle`` (which tests
    ``isinstance(corner_radius, float)``) failed with ``zip() argument 2 is
    longer than argument 1``.
    """
    width = Square().get_width()
    mob = algan.RoundedRectangle(corner_radius=width * 0.1)
    assert mob.manim_mobject.corner_radius == pytest.approx(0.2)
