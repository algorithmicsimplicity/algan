"""Manim idioms a ported script writes on a compatibility Mob, answered usefully.

Each case is a line verbatim Manim code is full of -- ``x.animate``,
``b.set_color(a.get_color())``, a corner radius taken from a width -- that the
compatibility layer used to answer with an opaque error or a silently wrong
picture. A feature test, outside the fast suite: each case breaks only when the
compatibility layer changes.
"""

from __future__ import annotations

import pytest

import algan.manim as mn
from algan import Arc, Axes
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
