r"""``Text`` must reach its LaTeX fallback when manimpango is absent.

On Linux the ``pango`` extra is opt-in, and both the installation guide and
``algan check`` promise that ``Text`` still typesets -- through LaTeX text mode
-- without it. It did not: ``slant`` and ``weight`` have string defaults, so
every construction normalized them against Pango's name list and died on the
import before any fallback could run.

Once it ran, it degraded silently: monospace fonts came out serif, text was
70% of Pango's size, ``~`` became an accent, and a multi-line ``Text`` failed
outright, its newlines spliced into ``\text{}`` inside ``align*``.
"""

from __future__ import annotations

import builtins
import shutil
import warnings

import pytest
import torch

from algan.constants.color import GREY, WHITE, Color
from algan.constants.spatial import RIGHT, UP
from algan.errors import UnsupportedFeatureWarning
from algan.mobs import text as text_module

needs_latex = pytest.mark.skipif(
    shutil.which("latex") is None or shutil.which("dvisvgm") is None,
    reason="needs a TeX distribution with dvisvgm",
)


@pytest.fixture
def latex_text(monkeypatch):
    """Make ``Text`` take its LaTeX fallback, whether or not Pango is here."""
    monkeypatch.setattr(text_module, "_pango_available", lambda: False)
    monkeypatch.setattr(text_module, "_LATEX_TEXT_FALLBACK_WARNED", True)
    return text_module.Text


def _height(mob):
    return float(mob.get_length_in_direction(UP).reshape(-1)[0])


def _glyph_count(text):
    return sum(not char.isspace() for char in text)


@pytest.fixture
def no_manimpango(monkeypatch):
    """Make ``import manimpango`` fail, whether or not it is installed."""
    monkeypatch.setattr(text_module, "_PANGO_NAMES", {})
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "manimpango" or name.startswith("manimpango."):
            raise ImportError("No module named 'manimpango'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    yield
    text_module._PANGO_NAMES.clear()


def test_pango_names_are_empty_without_manimpango(no_manimpango):
    assert text_module._pango_names("slant") == ()
    assert text_module._pango_names("weight") == ()


def test_defaults_do_not_raise_without_manimpango(no_manimpango):
    assert text_module._pango_style("slant", "NORMAL") == "NORMAL"
    assert text_module._pango_style("weight", "NORMAL") == "NORMAL"


def test_case_is_still_normalized_without_manimpango(no_manimpango):
    assert text_module._pango_style("weight", "bold") == "BOLD"


def test_unvalidatable_name_is_accepted_without_manimpango(no_manimpango):
    """Nothing can honour it in LaTeX text mode, and nothing knows it is wrong."""
    assert text_module._pango_style("weight", "BOLDER") == "BOLDER"


def test_bad_name_still_rejected_when_manimpango_is_present():
    pytest.importorskip("manimpango")
    from algan.errors import AlganConfigurationError

    with pytest.raises(AlganConfigurationError):
        text_module._pango_style("weight", "BOLDER")


# -- the fallback itself ---------------------------------------------------


def test_monospace_and_sans_font_names_pick_a_computer_modern_family():
    for font in ("DejaVu Sans Mono", "Courier New", "Consolas", "Menlo", "monospace"):
        assert text_module._latex_family(font) == "tt", font
    for font in ("DejaVu Sans", "Arial", "Helvetica", "sans-serif"):
        assert text_module._latex_family(font) == "sf", font
    for font in ("", "DejaVu Serif", "Times New Roman"):
        assert text_module._latex_family(font) == "rm", font


def test_escapes_keep_spaces_and_break_character_changing_ligatures():
    escape = text_module._escape_plain_text
    assert escape("a  b") == r"a\ \ b"
    assert escape("x--y ``q''") == r"x-{}-y\ `{}`q'{}'"
    assert escape("fi", ligatures=False) == "f{}i"
    assert escape("fi") == "fi"
    assert escape("~<|") == r"$\sim$\textless{}\textbar{}"
    assert escape(r"\{", family="tt") == r"\char92 \char123 "


def test_layout_is_one_row_per_line_and_one_segment_per_styled_run():
    text = "ab x\n\n  cd"
    colors = [None] * 8 + [0, 0]
    layout = text_module._latex_text_layout(text, font="Mono", char_colors=colors)

    first, second = layout.tex_strings
    assert first.startswith(r"&\text{\ttfamily ab\ x}")
    # The glyph-less blank line rides on the segment before it.
    assert first.count(r"\\[") == 1
    assert second.startswith(r"\\[")
    assert second.endswith(r"&\text{\ttfamily \ \ cd}")
    assert layout.segment_colors == (None, 0)
    # Single-line text keeps the plain source TextTriangulated always used.
    assert text_module._latex_text_layout("mesh").tex_strings == (r"\text{mesh}",)


@needs_latex
def test_fallback_warns_once_naming_the_extra(monkeypatch):
    monkeypatch.setattr(text_module, "_pango_available", lambda: False)
    monkeypatch.setattr(text_module, "_LATEX_TEXT_FALLBACK_WARNED", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        text_module.Text("first", font="DejaVu Sans Mono", add_to_scene=False)
        text_module.Text("second", add_to_scene=False)

    fallback = [w for w in caught if issubclass(w.category, UnsupportedFeatureWarning)]
    assert len(fallback) == 1
    message = str(fallback[0].message)
    assert 'pip install "algan[pango]"' in message
    assert "`font`" in message


@needs_latex
def test_multiline_code_with_color_map_colors_the_right_glyphs(latex_text):
    # The backpropagation video's code card, which used to fail to compile.
    lines = [
        "for l in (4, 3, 2, 1):",
        "    G[l] = outer(a[l-1], delta)",
        "    # comment",
    ]
    source = "\n".join(lines)
    comment = source.index("# comment")
    code = latex_text(
        source,
        font="DejaVu Sans Mono",
        line_spacing=1.0,
        color_map={f"[{comment}:{comment + 9}]": Color(GREY)},
        add_to_scene=False,
    )

    assert len(code) == _glyph_count(source)
    grey = _glyph_count(source[:comment])
    for i in range(len(code)):
        expected = GREY if i >= grey else WHITE
        actual = code.character_mobs[i].color.reshape(-1)[:3]
        assert torch.allclose(actual, expected.reshape(-1)[:3], atol=1e-3), i


@needs_latex
@pytest.mark.parametrize("font", ["", "DejaVu Sans", "DejaVu Sans Mono"])
def test_special_characters_are_set_one_glyph_each(latex_text, font):
    # A character TeX swallowed, fused or turned into an accent would change
    # the count; "--" fused would be one en dash.
    source = r"~1,000 {x} 100% #a_b & $c ^ d < e > f | g \ h -- i ``j'' !` ?`"
    text = latex_text(source, font=font, add_to_scene=False)
    assert len(text) == _glyph_count(source)


@needs_latex
@pytest.mark.parametrize("source", ["", "   ", "\nabc", "abc\n", "a\n\n  \nb"])
def test_blank_lines_and_empty_text_typeset(latex_text, source):
    assert len(latex_text(source, add_to_scene=False)) == _glyph_count(source)


@needs_latex
def test_text_triangulated_takes_multiple_lines():
    mob = text_module.TextTriangulated("two\nlines", add_to_scene=False)
    assert len(mob.character_mobs) == len("twolines")


@needs_latex
@pytest.mark.parametrize("font", ["", "DejaVu Sans", "DejaVu Sans Mono"])
def test_fallback_is_the_size_pango_text_is(monkeypatch, font):
    pytest.importorskip("manimpango")
    if not text_module._pango_available():
        pytest.skip("manimpango is installed but did not import")

    def measure():
        capital = text_module.Text("H", font=font, add_to_scene=False)
        two_lines = text_module.Text("H\nH", font=font, add_to_scene=False)
        wide = text_module.Text("HHHHHHHHHH", font=font, add_to_scene=False)
        return (
            _height(capital),
            _height(two_lines) - _height(capital),
            float(wide.get_length_in_direction(RIGHT).reshape(-1)[0]),
        )

    pango = measure()
    monkeypatch.setattr(text_module, "_pango_available", lambda: False)
    monkeypatch.setattr(text_module, "_LATEX_TEXT_FALLBACK_WARNED", True)
    latex = measure()

    capital, pitch, width = (b / a for a, b in zip(pango, latex))
    assert capital == pytest.approx(1, abs=0.02)
    assert pitch == pytest.approx(1, abs=0.02)
    # Computer Modern's letters are not DejaVu's shapes, but no longer 70%.
    assert width == pytest.approx(1, abs=0.15)
