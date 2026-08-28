====================
Importing From Manim
====================

Algan supports all of the same Mobjects that Manim does
`Manim's <https://docs.manim.community/en/stable/>`_, implemented as native Algan mobs.
If you prefer, you can also create Mobjects using Manim
and then import them to Algan using :class:`~.ManimMob`.

Manim is a dependency of Algan, so ``import manim`` works out of the box; there
is nothing extra to install.

.. note::

    You can import Manim *objects* (geometry), but not Manim *animations*.
    Manim's ``Transform``, ``Create``, and ``FadeIn`` don't exist on Algan's
    timeline. Instead, animate the imported Mobs using Algan's own animation
    methods and contexts.

The ManimMob
============

Wrap any Manim Mobject in :class:`~.ManimMob` to turn it into an Algan Mob:

.. algan:: ImportingManimMob

    from algan import *
    import manim as mn

    # Let's grab a complex plane from the Manim library.
    plane = ManimMob(mn.ComplexPlane().add_coordinates()).spawn()

    # Now we have a Mob we can animate with Algan.
    with Seq(run_time=5):
        plane.scale(0.5)
        plane.rotate(90, OUT)

    Scene.save_video()

.. important::

    Do not use both ``from algan import *`` and ``from manim import *`` in the
    same file, as many class names clash. Import Manim under an alias (like
    ``import manim as mn``).

Plotting Functions on Axes
==========================

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

Helper methods like ``plot_parametric_curve``, ``get_axis_labels``, and
``get_graph_label`` work the same way: they return Mobs that you can spawn and
animate independently.

Importing a finished Manim diagram
==================================

When you would rather assemble the whole thing in Manim, build it there,
collect it into a single ``VGroup``, and wrap that group once:

.. algan:: ImportingManimVGroup

    from algan import *
    import manim as mn
    import numpy as np

    diagram = mn.VGroup()
    axes = mn.Axes(x_range=(-3, 3, 1), y_range=(-1.5, 1.5, 0.5), x_length=9, y_length=4.5)
    diagram.add(axes, axes.plot(lambda x: np.sin(x), color=mn.YELLOW))

    plot = ManimMob(diagram).spawn()
    plot.scale(0.7)

    Scene.save_video()

Wrapping the finished ``VGroup`` gives you one Algan Mob whose parts are its
children, so you can animate the whole diagram together or reach into
``plot.children`` for an individual piece.

3-D Manim Geometry
==================

Manim's 3-D Mobjects import too, and they arrive as real 3-D geometry rather
than as something flattened to face you. A :class:`~.Surface` (and everything
built on it e.g. ``Sphere``, ``Torus``, ``Cone``) is a grid of curved quad tiles
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

Importing an SVG
================

An ``.svg`` file drawn in Inkscape, Illustrator or Figma comes in through the
same door. :class:`~.SVGMobject` parses the file into cubic Bezier outlines,
which is exactly what Algan's 2-D shapes are made of, so the result is a
first-class Mob: it scales without pixelating, takes Algan colors, and morphs
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
:class:`~.SVGMobject` is not itself indexable.

The path is resolved by Manim, which looks for it relative to the working
directory rather than to your script. So unlike :class:`~.ImageMob`, launching
Python from a different directory can lose an SVG that sits beside your ``.py``
file. Pass an absolute path if that matters.

Only path geometry is imported. Embedded raster images, filters, gradients and
text-as-text do not survive the conversion. Convert text to outlines in your
editor before exporting.

Colors and Coordinates
======================

The two libraries use different conventions, and the boundary is the
``ManimMob`` constructor:

* **Colors.** Inside Manim code use Manim's colors (``mn.YELLOW``); once the
  Mob is an Algan Mob, use Algan's (``YELLOW``).
* **Coordinates.** Manim's ``UP``/``RIGHT`` and Algan's agree in direction, but
  Manim's z axis points the other way. Constructing in Manim and animating in
  Algan keeps each side self-consistent, which is another reason to do all the
  construction on one side of the line.
* **Sizes.** Manim's default frame is 8 units tall; Algan's visible area at the
  origin plane is about 7. Imported diagrams usually want a modest
  :meth:`~algan.animatable_base.mob.Mob.scale` or :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.fit_to_screen`.

Matching Manim's framing exactly
================================

Rather than rescaling each import, you can point the whole Scene at Manim's own
defaults with :meth:`Scene.use_manim_defaults() <algan.scene.Scene.use_manim_defaults>`.
Call it once, before you build anything, and imported geometry lands on the
pixels Manim would have put it on: same 8-unit frame height, same perspective,
same light position, same black background, and the z mirror that keeps a
converted 3-D scene from rendering back-to-front.

.. code-block:: python

    from algan import *
    import manim as mn

    Scene.use_manim_defaults()

    square = ManimMob(mn.Square(color=mn.BLUE)).spawn()
    square.move(RIGHT)

    Scene.save_video()


:ref:`Matching Manim's defaults <migrating-manim-defaults>` in the
:doc:`../manim_migration_guide` is the full treatment: what each of the four
parts takes from Manim, how to decline one, and the two extras that are off by
default.

.. _manim-defaults:

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
  positions and base colors agree, the shading within a face does not.

Native Compatibility Classes
============================

For convenience, Algan also exposes many of Manim's composite Mobjects under
their own names, so they can be constructed without touching ``manim``:
:class:`~.Axes`, :class:`~.NumberPlane`, :class:`~.NumberLine`,
:class:`~.BarChart`, :class:`~.Table`, :class:`~.Brace`, :class:`~.Graph`,
:class:`~.Arc`, :class:`~.Annulus`, :class:`~.Ellipse`, :class:`~.Star`,
:class:`~.Arrow` and more.

They keep their Manim constructor arguments and methods, and are Algan Mobs in
every other respect: Algan's positioning, sizing and animation methods all work
on them, and their sizes can equally be set through their own Manim keyword
arguments (``x_length``, ``y_length``).

.. algan:: ImportingCompatClasses

    from algan import *

    plane = NumberPlane().scale(0.7).spawn()
    plane.rotate(30, UP)

    Scene.save_video()

See Also
========

* :doc:`../manim_migration_guide` -- the full concept-by-concept mapping,
  if you are porting a Manim project.
* :doc:`../galleries/mob_gallery` -- which of these classes are available under
  their own names, alongside Algan's native shapes.
* :doc:`three_d_models` -- importing ``.glb`` / ``.fbx`` models rather than
  Manim geometry.
* :doc:`shaders_and_materials` -- giving imported geometry an Algan material.
* :doc:`../reference` -- the full API reference.
