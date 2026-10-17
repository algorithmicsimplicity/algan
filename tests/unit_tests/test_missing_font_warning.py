"""A font that is not installed is reported at the line that asked for it.

Pango draws text in a substitute font when no installed font has the name it
is given, and says nothing. The vendored Manim logged the whole installed
list, hundreds of names on a Mac, and took a name installed under another
case or spacing for a missing one.
"""

from __future__ import annotations

import itertools
import linecache
import re
import warnings

import pytest

from algan.errors import UnsupportedFeatureWarning
from algan.mobs import text as text_module

#: Registered for the whole suite by tests/conftest.py, on every platform.
TEST_FONT = "Algan Test Sans"

_unique = itertools.count()


def unique(text: str) -> str:
    # A repeated Text is built from outlines Algan remembers, without Manim;
    # a new one goes through Manim's font check.
    return f"{text} {next(_unique)}"


@pytest.fixture
def pango():
    pytest.importorskip("manimpango")
    if not text_module._pango_available():
        pytest.skip("manimpango is installed but did not import")


def test_a_missing_font_warns_at_the_callers_line_naming_the_closest(pango):
    with pytest.warns(UnsupportedFeatureWarning) as caught:
        text_module.Text(unique("typo"), font="Algan Test Sanz", add_to_scene=False)
    [warning] = [w for w in caught if issubclass(w.category, UnsupportedFeatureWarning)]
    assert warning.filename == __file__
    message = str(warning.message)
    assert "No installed font is named 'Algan Test Sanz'" in message
    # Closest first; the other test faces may follow it.
    assert re.search(f"The closest installed fonts? (is|are) {TEST_FONT!r}", message)
    assert "warn_missing_font=False" in message


def test_a_text_built_from_remembered_outlines_warns_at_its_own_line(pango):
    """Algan typesets a source once per process and remembers the outlines.

    A later Text of the same source is built from them without Manim, so
    without Manim's font check, and said nothing. A warm daemon runs every
    script in one process, so only its first run said the font was missing.
    """
    source = unique("remembered")
    with pytest.warns(UnsupportedFeatureWarning) as typeset:
        text_module.Text(source, font="Algan Test Sanz", add_to_scene=False)
    with pytest.warns(UnsupportedFeatureWarning) as remembered:
        text_module.Text(source, add_to_scene=False, font="Algan Test Sanz")
    [first] = typeset
    [second] = remembered
    assert second.filename == __file__
    assert linecache.getline(__file__, second.lineno).strip() == (
        'text_module.Text(source, add_to_scene=False, font="Algan Test Sanz")'
    )
    assert str(second.message) == str(first.message)


def test_warn_missing_font_false_says_nothing(pango):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnsupportedFeatureWarning)
        text_module.Text(
            unique("quiet"),
            font="Algan Test Sanz",
            warn_missing_font=False,
            add_to_scene=False,
        )


@pytest.mark.parametrize("spelling", ["AlganTestSans", "algan  test sans"])
def test_a_font_installed_under_another_case_or_spacing_is_sent_as_installed(
    pango, spelling
):
    """Manim tries three capitalizations; fontconfig also ignores blanks."""
    import manim

    with warnings.catch_warnings():
        warnings.simplefilter("error", UnsupportedFeatureWarning)
        text = manim.Text(unique("spelling"), font=spelling)
    assert text.font == TEST_FONT


@pytest.fixture
def pango_has(monkeypatch):
    """Make ``fonts`` every family Pango has, whatever this machine has."""

    def set_fonts(*fonts):
        monkeypatch.setattr(text_module, "_pango_font_families", lambda: list(fonts))
        return list(fonts)

    return set_fonts


def test_the_check_returns_the_installed_spelling_and_knows_generic_names(
    pango_has,
):
    installed = pango_has("DejaVu Sans", "DejaVu Sans Mono")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnsupportedFeatureWarning)
        assert text_module._installed_font("dejavusansmono", installed) == (
            "DejaVu Sans Mono"
        )
        # Pango maps these to a font of that kind, listed or not.
        for generic in ("mono", "Monospace", "serif", "system-ui"):
            assert text_module._installed_font(generic, installed) == generic


def test_a_font_registered_after_manim_listed_the_fonts_is_installed(pango_has):
    """Manim's ``Text.font_list()`` is cached for the process."""
    pango_has("DejaVu Sans", "Registered Later")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnsupportedFeatureWarning)
        assert text_module._installed_font("registered later", ["DejaVu Sans"]) == (
            "Registered Later"
        )


@pytest.mark.parametrize(
    ("font", "suggestion"),
    [
        (
            "DejaVu San",
            "The closest installed fonts are 'DejaVu Sans', 'DejaVu Sans Mono'.",
        ),
        ("Inte", "The closest installed font is 'Inter'."),
        ("Zapfino", None),
    ],
)
def test_the_warning_names_the_closest_installed_fonts(pango_has, font, suggestion):
    installed = pango_has("DejaVu Sans", "DejaVu Sans Mono", "Inter")
    with pytest.warns(UnsupportedFeatureWarning) as caught:
        assert text_module._installed_font(font, installed) == font
    message = str(caught[0].message)
    assert f"No installed font is named {font!r}" in message
    if suggestion is None:
        assert "closest" not in message
    else:
        assert suggestion in message
