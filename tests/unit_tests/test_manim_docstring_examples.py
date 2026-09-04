"""Guard the two ways vendored Manim used to leak into the documentation build.

The docs job builds with ``-W``, so anything either of these lets through is a
red CI run rather than a cosmetic blemish:

* every compatibility wrapper in :mod:`algan.mobs.manim_compat` inherits its
  backing class's docstring, and ~100 of Manim's carry ``.. manim::`` example
  blocks. Algan does not register that directive, so each one is an "Unknown
  directive type" error -- and even if it were registered, the body is Manim
  scene code that will not run under Algan;
* :func:`~algan.utils.docbuild.module_parsing.parse_module_attributes` walks
  every ``.py`` under the package to build ``autodoc_type_aliases``. The
  vendored subtree under ``algan/external_libraries`` is not documented here, so
  an alias found in it can only point at a page that is never generated.
"""

from __future__ import annotations

import algan.mobs.manim_compat as manim_compat
from algan.mobs.manim_compat import _MANIM_DIRECTIVE_RE, _strip_manim_examples
from algan.utils.docbuild.module_parsing import (
    EXCLUDED_DIRECTORY_NAMES,
    parse_module_attributes,
)


def _directive_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if _MANIM_DIRECTIVE_RE.match(line)]


def test_no_wrapper_docstring_carries_a_manim_directive():
    offenders = {
        name
        for name in manim_compat._WRAPPED_MANIM_CLASS_NAMES
        if _directive_lines(getattr(manim_compat, name).__doc__ or "")
    }
    assert not offenders


def test_at_least_one_wrapper_needed_stripping():
    """Fail loudly if the fixture above went vacuous (renamed upstream field)."""
    stripped = [
        name
        for name in manim_compat._WRAPPED_MANIM_CLASS_NAMES
        if name not in manim_compat._WRAPPER_DOCSTRINGS
        and _directive_lines(getattr(manim_compat._manim, name).__doc__ or "")
    ]
    assert len(stripped) > 50


def test_strip_drops_the_whole_examples_section():
    doc = """Summary line.

    Parameters
    ----------
    radius
        The radius.

    Examples
    --------
    The first example shows a circle.

    .. manim:: CircleExample
        :save_last_frame:

        class CircleExample(Scene):
            def construct(self):
                self.add(Circle())

    See Also
    --------
    :class:`Arc`
    """

    result = _strip_manim_examples(doc)

    assert "Examples" not in result
    assert "CircleExample" not in result
    assert "The first example shows a circle." not in result
    # Sections on either side survive.
    assert "Parameters" in result
    assert "See Also" in result
    assert ":class:`Arc`" in result


def test_strip_handles_a_space_before_the_colons():
    """Manim's ``Torus`` writes ``.. manim :: ExampleTorus``."""
    doc = """Summary.

    .. manim :: ExampleTorus
        :save_last_frame:

        class ExampleTorus(ThreeDScene):
            def construct(self):
                self.add(Torus())

    Trailing prose.
    """

    result = _strip_manim_examples(doc)

    assert "ExampleTorus" not in result
    assert "Summary." in result
    assert "Trailing prose." in result


def test_strip_leaves_a_clean_docstring_alone():
    doc = "Summary.\n\nParameters\n----------\nradius\n    The radius.\n"

    assert _strip_manim_examples(doc) is doc
    assert _strip_manim_examples(None) is None


def test_alias_parsing_skips_the_vendored_subtree():
    alias_docs, data, typevars = parse_module_attributes()

    for excluded in EXCLUDED_DIRECTORY_NAMES:
        for found in (alias_docs, data, typevars):
            assert not [name for name in found if name.startswith(f"{excluded}.")]
