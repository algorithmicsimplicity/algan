"""Native Tex must consume custom templates without leaking them into Mob kwargs."""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import pytest

from algan import SETTINGS, Tex
from algan.mobs import text as text_module


@pytest.fixture(autouse=True)
def _isolated_tex_cache(tmp_path, fresh_scene):
    SETTINGS.paths.set(cache_directory=str(tmp_path))


def _template_kwargs(kind, preamble):
    if kind == "preamble":
        return {"preamble": preamble}
    template = text_module.mn.TexTemplate()
    template.add_to_preamble(preamble)
    return {"tex_template": template}


@pytest.mark.parametrize("kind", ["preamble", "tex_template"])
def test_custom_command_typesets_like_expanded_formula(kind):
    custom = Tex(r"\mycmd", **_template_kwargs(kind, r"\newcommand{\mycmd}{xyz}"))
    expanded = Tex("xyz")
    assert len(custom.character_mobs) == len(expanded.character_mobs) == 3


@pytest.mark.parametrize("kind", ["preamble", "tex_template"])
def test_custom_template_does_not_reuse_default_formula_glyphs(kind):
    # Prime any in-process glyph memo, then typeset the same source differently.
    assert len(Tex(r"\alpha").character_mobs) == 1
    custom = Tex(r"\alpha", **_template_kwargs(kind, r"\renewcommand{\alpha}{xyz}"))
    assert len(custom.character_mobs) == len(Tex("xyz").character_mobs) == 3
    assert len(Tex(r"\alpha").character_mobs) == 1


def test_mutated_template_does_not_reuse_previous_glyphs():
    template = text_module.mn.TexTemplate()
    template.add_to_preamble(r"\newcommand{\mycmd}{x}")
    assert len(Tex(r"\mycmd", tex_template=template).character_mobs) == 1
    template.add_to_preamble(r"\renewcommand{\mycmd}{xyz}")
    assert len(Tex(r"\mycmd", tex_template=template).character_mobs) == 3


@pytest.mark.parametrize("kind", ["preamble", "tex_template"])
@pytest.mark.parametrize("latex", [True, False])
def test_template_routing_keeps_geometry_kwargs_clean(monkeypatch, kind, latex):
    """Check routing without requiring a working external LaTeX compiler."""
    mn = import_module("manim")
    glyph = mn.VMobject().set_points_as_corners(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]]
    )
    calls = []
    preamble = r"\newcommand{\mycmd}{xyz}"
    options = _template_kwargs(kind, preamble)

    def typeset(*strings, **kwargs):
        calls.append(kwargs)
        if latex:
            part = SimpleNamespace(tex_string=strings[0], submobjects=[glyph])
            return SimpleNamespace(submobjects=[part])
        return SimpleNamespace(submobjects=[glyph])

    original_mob_init = text_module.Mob.__init__
    original_from_batches = text_module.BezierCircuitCubic.from_batches
    geometry_calls = []

    def checked_mob_init(self, *args, **kwargs):
        assert "preamble" not in kwargs
        assert "tex_template" not in kwargs
        geometry_calls.append("mob")
        original_mob_init(self, *args, **kwargs)

    def checked_from_batches(cls, *args, **kwargs):
        assert "preamble" not in kwargs
        assert "tex_template" not in kwargs
        geometry_calls.append("bezier")
        return original_from_batches(*args, **kwargs)

    monkeypatch.setattr(text_module, "_require_latex_toolchain", lambda: None)
    monkeypatch.setattr(mn, "MathTex" if latex else "Text", typeset, raising=False)
    monkeypatch.setattr(text_module.Mob, "__init__", checked_mob_init)
    monkeypatch.setattr(
        text_module.BezierCircuitCubic,
        "from_batches",
        classmethod(checked_from_batches),
    )

    Tex("x", latex=latex, **options)

    assert len(calls) == 1
    if latex:
        template = calls[0]["tex_template"]
        assert preamble in template.preamble
        if kind == "tex_template":
            assert template is options["tex_template"]
    else:
        assert "tex_template" not in calls[0]
    assert "mob" in geometry_calls
    assert "bezier" in geometry_calls
