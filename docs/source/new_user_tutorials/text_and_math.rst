====================
Text and Mathematics
====================

Explanatory animations are mostly labels and formulae, so Algan gives text and
LaTeX first-class treatment. Both are cubic Bezier circuits underneath -- real
outlines, not bitmaps -- so they stay crisp at any zoom and morph into other
shapes like anything else.

* :class:`~.Text` -- a string rendered with a font.
* :class:`~.Tex` -- LaTeX.
* :class:`~.NumericDisplay` -- a number you can animate.

Plain Text
==========

.. algan:: TextBasic

    from algan import *

    title = Text("Euler's identity", font_size=64).move(UP * 1.5).spawn()
    formula = Tex(r"e^{i\pi} + 1 = 0", font_size=80).spawn()

    with Seq(run_time=3):
        formula.color = YELLOW
        title.move(UP * 0.5)

    Scene.save_video()

:class:`~.Text` accepts the styling arguments you would expect:

.. algan:: TextStyles

    from algan import *

    lines = Group([
        Text("plain", font_size=44),
        Text("bold", font_size=44, weight="BOLD"),
        Text("italic", font_size=44, slant="ITALIC"),
        Text("coloured words", font_size=44, t2c={"coloured": YELLOW}),
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
     - Colour of the whole string. Defaults to
       ``SETTINGS.style.text_color`` (white).
   * - ``font``
     - Font family name, e.g. ``"Times New Roman"``.
   * - ``weight``
     - ``"NORMAL"``, ``"BOLD"``, ...
   * - ``slant``
     - ``"NORMAL"`` or ``"ITALIC"``.
   * - ``t2c``
     - Text-to-colour: ``{"word": YELLOW}`` colours just that substring.
       ``t2f``, ``t2s``, ``t2w`` do the same for font, slant and weight.
   * - ``line_spacing``
     - Gap between lines of a multi-line string.
   * - ``gradient``
     - A colour gradient across the string.

Because ``font_size`` and :meth:`~algan.animatable_base.mob.Mob.scale` both change apparent size, pick
one and stay with it. ``font_size`` is usually clearer for a fixed label;
``scale`` is what you animate.

LaTeX
=====

:class:`~.Tex` compiles LaTeX in **math mode**, so you never have to wrap
anything in ``$``:

.. algan:: TextMathTex

    from algan import *

    formula = Tex(r"\frac{d}{dx}\left(x^2\right) = 2x", font_size=60).spawn()
    with Seq(run_time=2):
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

Pass several strings to :class:`~.Tex` and each becomes a separate **segment**,
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

    Segments are not ``children``. A multi-part :class:`~.Tex` keeps every glyph
    in one packed batch, so ``formula.children`` has a single entry -- looping
    over it colours the whole formula at once and any surrounding
    :class:`~.Lag` has nothing to stagger. Reach for
    :meth:`~algan.mobs.text.Tex.get_segment` whenever you want the pieces you
    passed in, and index the Mob directly (``formula[3]``) for individual glyphs.

Per-glyph animation
===================

Every :class:`~.Text` and :class:`~.Tex` exposes its individual glyphs as
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

The hand-writing effect
=======================

:meth:`~algan.mobs.text.Tex.write` traces each glyph's outline and then fills it, one glyph after
another, for the classic "written by hand" look:

.. algan:: TextWrite

    from algan import *

    Text("Hand written", font_size=64).spawn(False).write(run_time=3)

    Scene.save_video()

Note the ``spawn(False)`` before ``write()``. Without ``False`` the text would
first play its ordinary fade-in and *then* be written. ``write()`` deliberately
does not change the text's spawned state; Algan keeps lifespan management
separate from animations.

``write()`` takes ``run_time`` for the whole sequence and ``lag_ratio`` for how
much each glyph overlaps the next (``0`` writes them all at once). It is shorthand
for :func:`~.draw_border_then_fill` applied to the glyphs -- that function works
on any iterable of Mobs, so you can use it on shapes too:

.. code-block:: python

    draw_border_then_fill([circle, square], run_time=2)

Animated Numbers
================

:class:`~.NumericDisplay` renders a number and animates through the values
in between when you change it:

.. algan:: TextNumericDisplay

    from algan import *

    counter = NumericDisplay(0.0, num_decimal_places=2).scale(2).spawn()
    with Seq(run_time=3):
        counter.value = 100.0

    Scene.save_video()

``num_decimal_places`` fixes the digits after the point and
``num_integer_places`` sets an initial minimum width before it. If the value later
needs more integer digits, the display grows automatically; the extra slots remain
available so its width stays stable afterwards.

Where to next
-------------

* :doc:`positioning_and_layout` -- placing labels next to what they label.
* :doc:`built_in_animations` -- drawing attention to a term you just introduced.
* :doc:`../advanced_user_tutorials/audio_and_speech` -- syncing text with
  narration.
