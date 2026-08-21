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
:ref:`Axes.plot <reference-manim-axes-plot>` hands back is an ordinary Algan Mob that you spawn and animate
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

Three-dimensional Manim geometry
================================

Manim's 3-D Mobjects import too, and they arrive as real 3-D geometry rather
than as something flattened to face you. A :class:`~.Surface` -- and everything
built on it, ``Sphere``, ``Torus``, ``Cone`` -- is a grid of curved quad tiles
in Manim, and each tile becomes one of the same curved patches Algan's own
3-D shapes are made of: it is diced to triangles afresh in every frame, as
finely as that frame needs and no more, so the silhouette stays smooth however
far you push the camera in.

.. algan:: ImportingManim3D

    from algan import *
    import manim as mn
    import numpy as np

    with Off():
        ball = ManimMob(mn.Sphere(resolution=(16, 8)))
        coil = ManimMob(mn.ParametricFunction(
            lambda t: np.array([1.8 * np.cos(t), 1.8 * np.sin(t), t / 3 - 3.1]),
            t_range=[0, 6 * np.pi], stroke_width=8,
        ))
        # Manim builds both around its own z axis, which points away from the
        # camera in Algan; stand them upright before animating.
        Group(ball, coil).rotate(-75, RIGHT).spawn()

    with Sync(run_time=4):
        ball.rotate(360, UP)
        coil.rotate(360, UP)

    Scene.save_video()

A 3-D *path* -- a ``ParametricFunction`` tracing a helix, a knot, a field line --
keeps its true position in space while its stroke keeps a constant width on
screen, which is what Manim draws and what a solid tube would not.

Because all of this is ordinary geometry, the rest of Algan applies to it: an
imported sphere casts and receives ray-traced shadows, shows up in reflections
and refractions, and takes an Algan
:doc:`material <../advanced_user_tutorials/shaders_and_materials>` like any
native shape.

Two things to know. Manim tiles a surface at a fixed ``resolution``, and that
tiling is the shape Algan reproduces -- raising it gives a rounder object, at
the usual cost. And Manim's z axis points opposite to Algan's, so a ``Sphere``
imported as-is has its poles pointing at the camera; rotate it about
:data:`~algan.constants.spatial.RIGHT` if you want them upright, as above.

Importing an SVG
================

An ``.svg`` file drawn in Inkscape, Illustrator or Figma comes in through the
same door. :class:`~.SVGMobject` parses the file into cubic Bezier outlines,
which is exactly what Algan's 2-D shapes are made of, so the result is a
first-class Mob -- it scales without pixelating, takes Algan colours, and morphs
into other shapes:

.. algan-doc-check: skip -- needs logo.svg, which does not ship with the docs

.. code-block:: python

    from algan import *

    logo = SVGMobject("logo.svg").scale(2).spawn()
    logo.color = BLUE
    logo.rotate(360, UP)

    Scene.save_video()

Each path in the file becomes its own Mob inside the result, reachable through
:ref:`children <reference-mob-children>` (the top-level Mob holds a
:class:`~.Group`, whose children are the paths), so you can animate an individual
piece without disturbing the rest. Unlike a :class:`~.Group`, an
:class:`~.SVGMobject` is not itself indexable -- ``logo[0]`` raises.

The path is resolved by Manim, which looks for it relative to the working
directory rather than to your script -- so unlike :class:`~.ImageMob`, launching
Python from a different directory can lose an SVG that sits beside your ``.py``
file. Pass an absolute path if that matters.

Only path geometry is imported. Embedded raster images, filters, gradients and
text-as-text do not survive the conversion -- convert text to outlines in your
editor before exporting.

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
  :meth:`~algan.animatable_base.mob.Mob.scale` or :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.fit_to_screen_rectangle`.

.. _manim-defaults:

Matching Manim's framing exactly
================================

Rather than rescaling each import, you can point the whole Scene at Manim's own
defaults with :meth:`Scene.use_manim_defaults() <algan.scene.Scene.use_manim_defaults>`.
Call it once, before you build anything::

    from algan import *
    import manim as mn

    Scene.use_manim_defaults()

    ManimMob(mn.Square()).spawn()
    ManimMob(mn.Sphere()).spawn()

    Scene.save_video()

It sets the four things that decide where imported geometry lands, and what
colour it comes out:

* **The frame.** Manim's frame is 8 world units tall and its width follows from
  the aspect ratio, which is the same convention Algan's vertical ``fov`` uses.
* **The camera.** Manim's ``ThreeDCamera`` is a pinhole camera 20 units from the
  frame plane, so the vertical field of view is
  ``2 * atan(4 / 20)`` = 22.62 degrees. Manim's plain 2-D camera is a flat
  orthographic projection instead, but the two agree exactly at ``z = 0``, so this
  one camera reproduces 2-D scenes exactly and 3-D scenes with Manim's own
  perspective.
* **The lighting.** Manim's single light, in Manim's position, and flat unlit
  colour as the default shading -- which is what Manim's renderer actually does
  to a vector Mobject -- with tonemapping off, so a flat fill comes out
  byte-identical to Manim's.
* **The z axis.** Manim's ``OUT`` is ``+z`` and Algan's is ``-z``, so
  ``use_manim_defaults()`` also mirrors imported geometry in z. Without that a
  converted 3-D scene renders back-to-front; flat ``z = 0`` geometry is
  unaffected either way.

Each part can be declined -- ``use_manim_defaults(shading=False)`` keeps Algan's
lighting while taking Manim's framing. Two extras are off by default:
``video_settings=True`` also switches the output to Manim's 1920x1080 at 60 fps,
and ``shape_defaults=True`` makes Algan's *own* shapes (``Square``, ``Circle``,
...) adopt Manim's colours and stroke styling.

How close it gets
-----------------

Close enough that the two are hard to tell apart, but not byte-identical, and
the residue is worth knowing:

* **Flat fills are exact.** A filled region renders to the same bytes as Manim's.
* **Unfilled strokes land within about a third of a pixel** at 854x480.
* **A filled shape's outline is offset by half its stroke width.** Manim centres
  a stroke on the path; Algan draws a filled circuit's border inside it. At
  Manim's default ``stroke_width`` of 4 that is about 1 pixel at 854x480.
* **3-D solids are shaded differently.** Manim shades a face by a two-point
  gradient across it (``shading_factor`` 0.2); Algan ray-traces. Silhouettes,
  positions and base colours agree, the shading within a face does not.

``benchmarks/_manim_defaults_parity_check.py`` renders the same scene through
both engines and reports these numbers, if you want to measure them yourself.

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

Where to next
=============

* :doc:`../manim_user_quickstart/index` -- the full concept-by-concept mapping,
  if you are porting a Manim project.
* :doc:`../advanced_user_tutorials/three_d_models` -- importing ``.glb`` /
  ``.fbx`` models rather than Manim geometry.
* :doc:`../advanced_user_tutorials/index` -- materials, lighting, cameras,
  audio and performance.
* :doc:`../reference` -- the full API reference.
