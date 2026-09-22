====================
Text and Mathematics
====================

Explanatory animations are mostly labels and formulae, so Algan gives text and
LaTeX first-class treatment. Both are cubic Bezier circuits underneath -- real
outlines, not bitmaps -- so they stay crisp at any zoom and morph into other
shapes like anything else.

* :class:`~algan.mobs.text.Text` -- a string rendered with a font.
* :class:`~algan.mobs.text.Tex` -- LaTeX.
* :class:`~algan.mobs.typst.Typst` and :class:`~algan.mobs.typst.MathTypst` --
  optional Typst markup and mathematics, with selectable labeled parts.
* :class:`~algan.mobs.numeric_display.DecimalNumber` -- a number you can animate.

Plain Text
==========

.. algan:: TextBasic

    from algan import *

    title = Text("Euler's identity", font_size=64).move(UP * 1.5).spawn()
    formula = Tex(r"e^{i\pi} + 1 = 0", font_size=80).spawn()

    with Seq(runtime=3):
        formula.color = YELLOW
        title.move(UP * 0.5)

    Scene.save_video()

:class:`~algan.mobs.text.Text` accepts the styling arguments you would expect:

.. algan:: TextStyles

    from algan import *

    lines = Group([
        Text("plain", font_size=44),
        Text("bold", font_size=44, weight="BOLD"),
        Text("italic", font_size=44, slant="ITALIC"),
        Text("colored words", font_size=44, color_map={"colored": YELLOW}),
    ])
    lines.arrange_in_line(DOWN, buffer=0.35).move_to(ORIGIN).spawn()
    lines.wait()

    Scene.save_video()

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Argument
     - Meaning
   * - ``font_size``
     - Point size. ``48`` by default; ``100`` fills most of the frame.
   * - ``color``
     - Color of the whole string. Defaults to
       ``SETTINGS.style.text_color`` (white).
   * - ``font``
     - Font family name, e.g. ``"Times New Roman"``.
   * - ``weight``
     - ``"NORMAL"``, ``"BOLD"``, ...
   * - ``slant``
     - ``"NORMAL"`` or ``"ITALIC"``.
   * - ``color_map``
     - Text-to-color: ``{"word": YELLOW}`` colors just that substring.
       ``font_map``, ``slant_map``, ``weight_map`` do the same for font, slant
       and weight.
   * - ``line_spacing``
     - Gap between lines of a multi-line string.
   * - ``gradient``
     - A color gradient across the string.

Because ``font_size`` and :meth:`~algan.animatable_base.mob.Mob.scale` both change apparent size, pick
one and stay with it. ``font_size`` is usually clearer for a fixed label;
``scale`` is what you animate.

Typst
=====

Install the optional compiler with ``pip install "algan[typst]"``. The Python
wheel includes Typst itself; these classes need neither a separate executable
nor LaTeX or Pango. Existing ``Text`` and ``Tex`` workflows keep their current
backends.

:class:`~algan.mobs.typst.Typst` accepts Typst markup, including embedded
``$ math $``. :class:`~algan.mobs.typst.MathTypst` supplies the math delimiters
for you. Both produce Algan vector geometry, so ordinary movement, materials,
color changes and shape matching apply.

Label markup with ``#box[words] <label>``. In math, use
``{{ expression : label }}`` for a named group and ``{{ expression }}`` for
an unnamed group. Unnamed groups are numbered from zero; named groups do not
consume an index. Repeated labels select every occurrence.

.. algan:: TypstLabeledParts

    from algan import *

    title = Typst("A #box[Typst] <backend> equation", font_size=42)
    title.move(UP * 1.5).spawn()
    equation = MathTypst("{{ a^2 + b^2 : lhs }} = {{ c^2 }}", font_size=64).spawn()
    with Sync():
        title.select("backend").color = BLUE
        equation.select("lhs").color = YELLOW
        equation.select(0).move(RIGHT * 0.5)
    Scene.save_video()

``select()`` returns a view over the displayed paths. Edit it directly; it
does not need spawning. Missing labels raise ``KeyError`` and missing unnamed
indices raise ``IndexError``. Selections also work with ``batch=True``, which
packs the geometry for rendering, though packed parts share one lifespan.

Use ``typst_preamble`` for Typst definitions and rules, such as
``'#set text(font: "DejaVu Sans")'``, and ``font_paths`` for additional font
directories. ``font_size`` defaults to 48 points; ``height`` can instead set
the total height in world units. Typst-authored colors are preserved and
``stroke_width=None`` preserves strokes such as fraction bars. Syntax errors
are reported by the compiler with source locations.

Compiled source and SVG files are cached in
``SETTINGS.paths.cache_directory / "manim" / "Typst"``. The source, preamble,
compiler version and additional font directory paths identify the compiled
result. If you change fonts or files imported by the preamble in place, clear
the cache to recompile. Importing Algan does not require the optional compiler.

For glyph alignment, pass ``track_baselines=True`` and use
``equation.get_baseline_frame(equation.select(0)[0])``. It returns the current
world-space origin, right reference point and up reference point, following
affine transformations of that glyph. ``baseline_frames`` returns these for
all tracked paths. The same selection API is available on ``algan.manim.Typst``
and ``algan.manim.MathTypst``.

LaTeX
=====

:class:`~algan.mobs.text.Tex` compiles LaTeX in **math mode**, so you never have to wrap
anything in ``$``:

.. algan:: TextMathTex

    from algan import *

    formula = Tex(r"\frac{d}{dx}\left(x^2\right) = 2x", font_size=60).spawn()
    with Seq(runtime=2):
        formula.color = YELLOW
        formula.scale(1.3)

    Scene.save_video()

For a run of ordinary prose inside a formula, wrap it in ``\text{...}`` as you
would anywhere else in LaTeX.

.. important::

    LaTeX requires a working TeX installation on your machine (any of TeX Live,
    MiKTeX or MacTeX). Algan caches the compiled glyph geometry, so only the
    first render of a given string pays the LaTeX cost.

Always use raw strings (``r"..."``) for LaTeX, so Python does not eat the
backslashes.

Animating parts of a formula
============================

Pass several strings to :class:`~algan.mobs.text.Tex` and each becomes a separate **segment**,
retrieved with :meth:`~algan.mobs.text.Tex.get_segment` and animated
independently:

.. algan:: TextTexParts

    from algan import *

    formula = Tex("e^{i\\pi}", "+ 1", "= 0", font_size=90).spawn()
    with Lag(0.5):
        for i in range(len(formula.tex_strings)):
            formula.get_segment(i).color = YELLOW

    Scene.save_video()

This is the standard way to draw attention to one term of an equation: split the
formula where you want the seams, then animate that segment.

.. note::

    Segments are not ``children``. A multi-part :class:`~algan.mobs.text.Tex` keeps every glyph
    in one packed batch, so ``formula.children`` has a single entry -- looping
    over it colors the whole formula at once and any surrounding
    :class:`~.Lag` has nothing to stagger. Reach for
    :meth:`~algan.mobs.text.Tex.get_segment` whenever you want the pieces you
    passed in, and index the Mob directly (``formula[3]``) for individual glyphs.

Per-glyph animation
===================

Every :class:`~algan.mobs.text.Text` and :class:`~algan.mobs.text.Tex` exposes its individual glyphs as
``character_mobs``, a list of Mobs you can animate one at a time:

.. algan:: TextGlyphs

    from algan import *

    word = Text("ALGAN", font_size=90).spawn()
    with Lag(0.3):
        for glyph in word.character_mobs:
            glyph.color = YELLOW

    Scene.save_video()

Combined with :class:`~.Lag`, this gives you cascading effects across a string
for free. Note that ``character_mobs`` contains only visible glyphs -- spaces are
not included.

Matching terms and shapes
=========================

Use :func:`~algan.animations.transform_matching_parts.TransformMatchingTex` when
rearranging an equation. It matches the LaTeX source of each term before using
Algan's normal morph, so an ``x`` travels to the new ``x`` instead of turning into
whichever glyph is closest. Give each term a separate constructor argument, or
isolate terms with double braces, such as ``Tex(r"{{x}} + {{y}}")``. An ungrouped
``Tex("x+y")`` has just one term; the matcher does not parse symbolic algebra.

.. algan:: MatchingEquationTerms

    from algan import *

    equation = Tex("x", "+", "y", "=", "z", font_size=64).spawn()
    equation.get_segment(0).color = YELLOW
    target = Tex("z", "-", "y", "=", "x", font_size=64)
    target.get_segment(4).color = YELLOW
    equation = TransformMatchingTex(equation, target, runtime=2)
    equation = TransformMatchingTex(
        equation,
        Tex("z", "-", "y", "=", "a", font_size=64),
        key_map={"x": "a"},
    )
    Scene.save_video()

This works with native :class:`~algan.mobs.text.Tex`,
:class:`~algan.mobs.manim_compat.MathTex`, compatible
``algan.manim`` LaTeX objects, and groups of equations. Keys are exact source
strings, including whitespace. ``key_map`` overrides natural matches. Repeated
occurrences of a key are collected and morphed together, with surplus geometry
growing or shrinking through the ordinary morph machinery.

For anagrams and geometric arrangements, use
:func:`~algan.animations.transform_matching_parts.TransformMatchingShapes`.
It matches individual packed glyphs and shapes by their ordered geometry after
centering, scaling to unit height and rounding to three decimal places. Color,
position and uniform scale do not affect matching; rotation, point ordering and
topology do. Horizontal shapes use their largest extent in place of height.
To override a shape match, pass Mob exemplars: ``key_map={square: circle}``.

Both helpers fade unmatched parts independently in place by default. Choose
``transform_mismatches=True`` to morph them into one another, or
``fade_transform_mismatches=True`` to cross-dissolve while moving and resizing.
These two options are mutually exclusive.

Spawn the source before calling either helper, and keep the returned Mob for
subsequent animation. It is a spawned copy of the target, retains the target's
hierarchy and text indexing, and replaces the source in its parents. The target
argument is unchanged. Timing follows ``Seq``, ``Sync`` and ``Lag`` as usual;
``runtime`` supplies an explicit duration in seconds, and ``Off`` makes the
replacement immediate. A packed slice cannot be the source because it shares
its owner's lifespan; pass the complete text or shape instead.

The hand-writing effect
=======================

:meth:`~algan.mobs.text.Tex.write` traces each glyph's outline and then fills it, one glyph after
another, for the classic "written by hand" look:

.. algan:: TextWrite

    from algan import *

    Text("Hand written", font_size=64).spawn(False).write(runtime=3)

    Scene.save_video()

Note the ``spawn(False)`` before ``write()``. Without ``False`` the text would
first play its ordinary fade-in and *then* be written. ``write()`` deliberately
does not change the text's spawned state; Algan keeps lifespan management
separate from animations.

``write()`` takes ``runtime`` for the whole sequence and ``lag_ratio`` for how
much each glyph overlaps the next (``0`` writes them all at once). It is shorthand
for :func:`~.DrawBorderThenFill` applied to the glyphs -- that function works
on any iterable of Mobs, so you can use it on shapes too. See
:doc:`../galleries/built_in_animations` for that.

Animated Numbers
================

:class:`~algan.mobs.numeric_display.DecimalNumber` renders a number and animates through the values
in between when you change it:

.. algan:: TextDecimalNumber

    from algan import *

    counter = DecimalNumber(0.0, decimal_places=2).scale(2).spawn()
    with Seq(runtime=3):
        counter.value = 100.0

    Scene.save_video()

``decimal_places`` fixes the digits after the point and
``integer_places`` sets an initial minimum width before it. If the value later
needs more integer digits, the display grows automatically; the extra slots remain
available so its width stays stable afterwards.

See Also
========

* :doc:`positioning_and_layout` -- placing labels next to what they label.
* :doc:`../galleries/built_in_animations` -- drawing attention to a term you just
  introduced.
* :doc:`audio_and_speech` -- syncing text with narration.
* :doc:`images_and_textures` -- painting a gradient or an image across glyphs.
* :doc:`importing_from_manim` -- ``MathTex``, ``Title`` and the rest of Manim's
  text mobjects.
* :doc:`../new_user_tutorials/combining_animations` -- the ``Lag`` context that
  makes the per-glyph effects above cascade.
