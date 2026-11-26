r"""A Text Pango cannot draw says so, rather than coming back empty.

Pango right-aligns a right-to-left line inside the layout width Manim gives it,
which puts the line past the edge of the surface it is drawn on: ``Text("שלום")``
was an invisible Mob, and in ``Text("abc\nשלום")`` the second line vanished,
both without a word. Laying such lines out is not supported yet; these pin
that the loss is reported, at the script's line.
"""

from __future__ import annotations

import itertools
import warnings

import pytest

from algan.errors import UnsupportedFeatureWarning
from algan.mobs import text as text_module

_unique = itertools.count()


def unique(text: str) -> str:
    # A repeated Text is built from outlines Algan remembers, without Pango.
    return f"{text} {next(_unique)}"


@pytest.fixture
def pango():
    pytest.importorskip("manimpango")
    if not text_module._pango_available():
        pytest.skip("manimpango is installed but did not import")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("שלום", ["שלום"]),
        ("مرحبا", ["مرحبا"]),
        ("abc\nשלום", ["שלום"]),
        ("123 שלום abc", ["123 שלום abc"]),  # digits are not strongly directional
        ("abc שלום", []),  # a left-to-right line, with a Hebrew word in it
        ("\u200eשלום", []),  # the left-to-right mark sets the line's direction
        ("hello\n\n  ", []),
    ],
)
def test_right_to_left_lines_are_those_whose_first_strong_character_is(text, expected):
    assert text_module._right_to_left_lines(text) == expected


def test_a_right_to_left_text_warns_at_the_callers_line(pango):
    source = unique("abc\nשלום")
    with pytest.warns(UnsupportedFeatureWarning) as caught:
        text_module.Text(source, add_to_scene=False)
    [warning] = [w for w in caught if issubclass(w.category, UnsupportedFeatureWarning)]
    assert warning.filename == __file__
    message = str(warning.message)
    assert "right-to-left" in message
    assert repr(source.split("\n")[1]) in message
    assert "\\u200e" in message

    # Built again from the remembered outlines, it says so again.
    with pytest.warns(UnsupportedFeatureWarning, match="right-to-left"):
        text_module.Text(source, add_to_scene=False)


def test_a_left_to_right_mark_draws_the_line_without_a_warning(pango):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnsupportedFeatureWarning)
        text = text_module.Text(unique("\u200eשלום"), add_to_scene=False)
    assert len(text) > 0


def test_a_non_blank_text_that_draws_nothing_warns():
    raised = []
    with pytest.warns(UnsupportedFeatureWarning, match="no glyphs for 'x'"):
        text_module._warn_of_dropped_pango_lines("x", 0, raised)
    [(category, message)] = raised
    assert category is UnsupportedFeatureWarning

    raised.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnsupportedFeatureWarning)
        text_module._warn_of_dropped_pango_lines("   ", 0, raised)
        text_module._warn_of_dropped_pango_lines("x", 1, raised)
    assert raised == []
