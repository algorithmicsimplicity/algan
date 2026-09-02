"""What Algan says when an external tool it needs is not installed.

The average person who ``pip install algan`` has no TeX distribution, and used
to meet a ``rich``-formatted line from the vendored Manim followed by a raw
``FileNotFoundError: 'latex'`` -- which names neither the program to install
nor the fact that ``Text`` needs none of it.
"""

from __future__ import annotations

import shutil

import pytest

import algan
from algan.errors import AlganConfigurationError
from algan.mobs import text as text_module
from algan.mobs.text import Tex


@pytest.fixture
def no_latex(monkeypatch):
    """A machine with no TeX distribution on PATH."""
    real_which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name, *a, **k: None
        if name in text_module._LATEX_BINARIES
        else real_which(name, *a, **k),
    )


def test_tex_without_latex_raises_a_configuration_error(no_latex):
    with pytest.raises(AlganConfigurationError) as raised:
        Tex("x^2")
    message = str(raised.value)
    assert "latex" in message
    assert "dvisvgm" in message
    assert "apt install" in message, "name the install command per platform"
    assert "brew" in message
    assert "MiKTeX" in message
    assert "Text(" in message, "say that Text needs no LaTeX"


def test_only_the_missing_binary_is_named(monkeypatch):
    real_which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name, *a, **k: None if name == "dvisvgm" else real_which(name, *a, **k),
    )
    with pytest.raises(AlganConfigurationError) as raised:
        Tex("x^2")
    assert "dvisvgm" in str(raised.value)
    assert "latex and dvisvgm" not in str(raised.value)


def test_nothing_is_written_before_the_error(no_latex, monkeypatch):
    """The guard comes before the scratch directories are created."""
    monkeypatch.setattr(
        text_module,
        "make_manim_dir",
        lambda: pytest.fail("nothing may be written for a run that cannot happen"),
    )
    with pytest.raises(AlganConfigurationError):
        Tex(r"\frac{1}{2}")


@pytest.mark.parametrize(
    ("build", "label"),
    [
        (lambda: algan.MathTex("x^2"), "MathTex"),
        (lambda: algan.Title("Chapter"), "Title"),
        (lambda: algan.Matrix([[1, 2], [3, 4]]), "Matrix"),
    ],
)
def test_manim_wrapped_latex_classes_get_the_same_guard(no_latex, build, label):
    """``MathTex`` and friends never enter ``text.py``: the wrapper guards them."""
    with pytest.raises(AlganConfigurationError) as raised:
        build()
    assert "latex" in str(raised.value), label


def test_a_manim_wrapper_that_needs_no_latex_is_unaffected(no_latex):
    assert algan.Square().location is not None
    assert algan.Text("prose") is not None


def test_a_non_latex_tex_is_unaffected(no_latex):
    """``latex=False`` renders through Pango and needs no TeX at all."""
    mob = Tex("prose", latex=False)
    assert mob is not None
