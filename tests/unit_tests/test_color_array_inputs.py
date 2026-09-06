"""Building a :class:`~algan.constants.color.Color` out of NumPy arrays.

Every colour that comes in through the Manim compatibility layer arrives this
way. ``ManimColor.to_rgba()`` returns a NumPy array
(``algan/external_libraries/manim/utils/color/core.py``), and a ``VMobject``'s
``fill_opacity`` is whatever ``set_fill`` stored -- an array with one entry per
submobject. ``algan/mobs/manim_mob.py``'s ``convert_manim_color`` passes both
straight into ``Color(rgb, glow=0, opacity=a)``.

``Color.__new__`` used to branch on ``str`` / ``tuple`` / ``list`` /
``torch.Tensor`` and had no NumPy case at all, so an ``ndarray`` fell through
every branch and was splatted into the five-channel tuple unconverted. That
survived only because NumPy still coerced a size-1 array to a scalar. It has
been deprecated since NumPy 1.25 and **raises** from NumPy 2.4 with
``TypeError: only 0-dimensional arrays can be converted to Python scalars``,
which took every Manim import down at scene-construction time --
``tests/full_renders/scenes/manim_compat_and_plots.py`` died in ``Axes(...)``.

Two separate defects, and both are covered below: the array-valued ``opacity``
that raised, and a 4- or 5-wide ``ndarray`` ``rgb``, which never raised but
produced a **six**- or seven-channel colour by splatting the alpha in beside
the ``glow``/``opacity`` arguments.

The deprecation warnings are promoted to errors here rather than checked on a
render, so the NumPy version installed decides nothing: on 2.3 the old code
warned and on 2.4 it raised, and this file fails on both.

Feature tests for the colour constructor: unmarked, so outside the fast suite.
"""

import numpy as np
import pytest
import torch

from algan.constants.color import Color

#: A silent NumPy deprecation here is this bug, one release early -- so every
#: test in the file fails on the warning rather than waiting for the raise.
pytestmark = pytest.mark.filterwarnings("error::DeprecationWarning")


def _channels(color):
    return [round(float(x), 6) for x in color.reshape(-1)]


def test_an_ndarray_rgb_with_an_array_opacity_is_accepted():
    """The exact shape ``convert_manim_color`` produces for an invisible fill."""
    color = Color(np.array([1.0, 1.0, 1.0]), glow=0, opacity=np.array([0.0]))
    assert _channels(color) == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_a_four_wide_ndarray_supplies_its_own_alpha():
    """Not a crash before this fix -- a silently six-channel colour."""
    color = Color(np.array([0.1, 0.2, 0.3, 0.4]))
    assert _channels(color) == [0.1, 0.2, 0.3, 0.0, 0.4]


def test_a_five_wide_ndarray_supplies_glow_and_alpha():
    color = Color(np.array([0.1, 0.2, 0.3, 2.0, 0.5]))
    assert _channels(color) == [0.1, 0.2, 0.3, 2.0, 0.5]


@pytest.mark.parametrize(
    "opacity",
    [np.float64(0.25), np.array(0.25), np.array([0.25]), torch.tensor([0.25])],
    ids=["numpy scalar", "0-d array", "size-1 array", "size-1 tensor"],
)
def test_every_array_like_opacity_reduces_to_the_same_scalar(opacity):
    assert _channels(Color((0.1, 0.2, 0.3), opacity=opacity))[4] == 0.25


def test_an_array_opacity_still_overrides_a_hex_strings_own_alpha():
    """The string branch compares ``opacity == 1``, which on an array is an array.

    That comparison is why the coercion happens before the branches rather than
    inside the array one.
    """
    assert _channels(Color("#58C4DD", opacity=np.array([0.5])))[4] == 0.5


def test_an_empty_opacity_array_is_refused_rather_than_guessed():
    from algan.errors import InvalidColorError

    with pytest.raises(InvalidColorError):
        Color((0.1, 0.2, 0.3), opacity=np.array([]))


@pytest.mark.parametrize(
    "rgb",
    [(0.1, 0.2, 0.3), [0.1, 0.2, 0.3], torch.tensor([0.1, 0.2, 0.3]), "#1A334D"],
    ids=["tuple", "list", "tensor", "hex"],
)
def test_the_forms_that_already_worked_are_unchanged(rgb):
    """The controls: nothing above may have moved the non-NumPy paths."""
    channels = _channels(Color(rgb))
    assert len(channels) == 5
    assert channels[3] == 0.0
    assert channels[4] == 1.0
