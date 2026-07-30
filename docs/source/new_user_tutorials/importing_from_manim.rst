====================
Importing From Manim
====================

Algan ships a good collection of Mobs, but nowhere near as extensive a one as
`Manim's <https://docs.manim.community/en/stable/>`_. So Algan lets you import
Manim Mobjects directly and animate them as Algan Mobs, which gives you Manim's
whole geometry library -- axes, plots, number planes, braces, tables, graphs,
matrices, code blocks -- for free.

Manim is a dependency of Algan, so ``import manim`` works out of the box; there
is nothing extra to install.

.. note::

    You can import Mobjects, but not Manim *animations*. Manim's ``Transform``,
    ``Create``, ``FadeIn`` and friends have no meaning on Algan's timeline. Use
    Algan's own animation system instead -- most of the common ones have direct
    equivalents (see :doc:`built_in_animations`).

The ManimMob
============

Wrap any Manim Mobject in a :class:`~.ManimMob` and you have an Algan Mob:

.. algan:: ImportingManimMob

    from algan import *
    import manim as mn

    # Let's grab a complex plane from the Manim library.
    mob = ManimMob(mn.ComplexPlane().add_coordinates()).spawn()

    # Now we have a Mob we can animate with Algan.
    with Seq(run_time=5):
        mob.scale(0.5)
        mob.rotate(90, OUT)

    Scene.save_video()

.. important::

    Do not use both ``from algan import *`` and ``from manim import *`` -- the
    two libraries share many names and the definitions would clash. Import one of
    them under a short alias, as with ``import manim as mn`` above.

Plotting on axes
================

Graphing is the usual reason to reach for Manim's geometry, and it needs no
wrapping at all: :class:`~.Axes` is available under its own name, and what
:meth:`Axes.plot` hands back is an ordinary Algan Mob that you spawn and animate
like any other.

.. algan:: ImportingManimPlot

    from algan import *
    import numpy as np

    axes = Axes(x_range=(-3, 3, 1), y_range=(-1.5, 1.5, 0.5), x_length=9, y_length=4.5)

    # plot() returns a Mob of its own -- spawn it like any other.
    graph = axes.plot(lambda x: np.sin(x), color=YELLOW)

    with Off():
        axes.spawn()
        graph.spawn()

    with Seq(run_time=2):
        graph.rotate(20, UP)
        graph.rotate(-20, UP)

    Scene.save_video()

The same holds for the other builder methods -- ``plot_parametric_curve``,
``get_axis_labels``, ``get_graph_label``, ``Brace.get_text`` -- so you can keep
each returned piece as its own Mob and animate them independently.

Importing a finished Manim diagram
==================================

When you would rather assemble the whole thing in Manim, build it there,
collect it into a single ``VGroup``, and wrap that group once:

.. code-block:: python

    from algan import *
    import manim as mn
    import numpy as np

    diagram = mn.VGroup()
    axes = mn.Axes(x_range=(-3, 3, 1), y_range=(-1.5, 1.5, 0.5), x_length=9, y_length=4.5)
    diagram.add(axes, axes.plot(lambda x: np.sin(x), color=mn.YELLOW))

    plot = ManimMob(diagram).spawn()

Wrapping the finished ``VGroup`` gives you one Algan Mob whose parts are its
children, so you can animate the whole diagram together or reach into
``plot.children`` for an individual piece.

Colours and coordinates
=======================

The two libraries use different conventions, and the boundary is the
``ManimMob`` constructor:

* **Colours.** Inside Manim code use Manim's colours (``mn.YELLOW``); once the
  Mob is an Algan Mob, use Algan's (``YELLOW``).
* **Coordinates.** Manim's ``UP``/``RIGHT`` and Algan's agree in direction, but
  Manim's z axis points the other way. Constructing in Manim and animating in
  Algan keeps each side self-consistent, which is another reason to do all the
  construction on one side of the line.
* **Sizes.** Manim's default frame is 8 units tall; Algan's visible area at the
  origin plane is about 7. Imported diagrams usually want a modest
  :meth:`~.Mob.scale` or :meth:`~.Mob.fit_to_screen_rectangle`.

Native compatibility classes
============================

For convenience, Algan also exposes many of Manim's composite Mobjects under
their own names, so they can be constructed without touching ``manim``:
:class:`~.Axes`, :class:`~.NumberPlane`, :class:`~.NumberLine`,
:class:`~.BarChart`, :class:`~.Table`, :class:`~.Brace`, :class:`~.Graph`,
:class:`~.Arc`, :class:`~.Annulus`, :class:`~.Ellipse`, :class:`~.Star`,
:class:`~.Arrow` and more.

.. code-block:: python

    from algan import *

    plane = NumberPlane().scale(0.7).spawn()
    plane.rotate(30, UP)

They keep their Manim constructor arguments and methods, and are Algan Mobs in
every other respect: Algan's positioning, sizing and animation methods all work
on them, and their sizes can equally be set through their own Manim keyword
arguments (``x_length``, ``y_length``).

If you are coming to Algan from Manim, :doc:`../manim_user_quickstart/index` maps
the concepts across.
