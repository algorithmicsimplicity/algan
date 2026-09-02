===============
The Mob Gallery
===============

This is a tour of the Mobs Algan ships with. Every one of them is a
:class:`.Mob`, so everything in :doc:`../new_user_tutorials/basic_animations` applies to all of them:
spawn them, move them, color them, morph them into each other.

The complete list, with every constructor argument, is in the
:doc:`mobs reference <../reference_index/mobs>`.

2-D Shapes
==========

Algan's 2-D shapes are cubic Bezier circuits (:class:`~.BezierCircuitCubic`),
which means they stay perfectly smooth however far you zoom in. A
:class:`~algan.mobs.shapes_2d.Circle` is a real circle, not a many-sided polygon.

.. algan:: GalleryShapes2D

    from algan import *

    shapes = Group([
        Circle(), Square(), Triangle(), RegularPolygon(6),
        Rectangle(width=1.4, height=0.8), Line(LEFT * 0.5, RIGHT * 0.5),
        Dot(), Point(),
    ])
    shapes.arrange_in_grid(2, row_buffer=0.9).scale(0.7).spawn()
    shapes.wait()

    Scene.save_video()

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Notes
   * - :class:`~algan.mobs.shapes_2d.Circle`
     - ``radius``.
   * - :class:`~algan.mobs.shapes_2d.Dot`
     - A small filled circle, for marking points.
   * - :class:`~algan.mobs.shapes_2d.Square`
     - ``size``.
   * - :class:`~algan.mobs.shapes_2d.Rectangle`
     - ``width``, ``height``.
   * - :class:`~algan.mobs.shapes_2d.RegularPolygon`
     - ``n`` sides. :class:`~algan.mobs.shapes_2d.Triangle` is ``RegularPolygon(3)``.
   * - :class:`~algan.mobs.shapes_2d.Polygon`
     - An arbitrary closed outline through the points you give it.
   * - :class:`~.Quad`
     - A four-cornered shape from four explicit corners.
   * - :class:`~algan.mobs.shapes_2d.Line`
     - From a start point to an end point.
   * - :class:`~algan.mobs.shapes_2d.Point`
     - A single point; mostly useful as an invisible anchor.
   * - :class:`~algan.mobs.shapes_2d.SurroundingRectangle`
     - A box drawn around another Mob, sized to fit it.

All of them take ``color``, plus ``stroke_width`` and ``stroke_color`` for
their outline:

.. algan:: GalleryBorders

    from algan import *

    with Off():
        Group([Circle(color=BLUE, stroke_color=WHITE, stroke_width=w).scale(0.8)
               for w in (0, 4, 16)]).arrange_in_line(RIGHT, buffer=0.4).spawn()

    Scene.wait(1)

    Scene.save_video()

On a filled shape the border is drawn *inside* the outline, so raising
``stroke_width`` eats into the fill instead of growing the silhouette.
This makes bordered text stay legible and neighbouring glyphs never fuse. An unfilled
shape (``filled=False``, and :class:`~algan.mobs.shapes_2d.Line`) has no interior to eat into, so
its stroke stays centred on the path.

More 2-D shapes from the compatibility layer
--------------------------------------------

A second family of outline shapes comes from Algan's Manim compatibility layer
(see :doc:`../advanced_user_tutorials/importing_from_manim`). They spawn, move and animate like any other
Mob, but they are constructed with Manim's arguments:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Notes
   * - :class:`~.Arc`
     - Sized with ``radius``, ``start_angle``, and ``angle``.
   * - :class:`~.Annulus`
     - A ring defined by ``inner_radius`` and ``outer_radius``.
   * - :class:`~.Ellipse`
     - Sized with ``width`` and ``height``.
   * - :class:`~.Star`
     - Defined by ``n`` points, ``outer_radius``, and ``inner_radius``.
   * - :class:`~.Arrow`
     - An arrow from a start point to an end point, with a customizable head.

These come from the Manim compatibility layer, but they take the same
``stroke_width`` and ``stroke_color`` in the same units as the native shapes
above -- ``algan.manim`` is where Manim's own (double) stroke unit lives:

.. algan:: GalleryCompatShapes

    from algan import *

    with Off():
        Group([
            Arc(radius=1, start_angle=0, angle=3.14),
            Annulus(inner_radius=0.5, outer_radius=1.0),
            Ellipse(width=2, height=1.2),
            Star(n=6, color=BLUE, stroke_color=WHITE, stroke_width=2),
            Arrow(start=LEFT, end=RIGHT),
        ]).arrange_in_line(RIGHT, buffer=0.4).scale(0.8).spawn()

    Scene.wait(1)

    Scene.save_video()

3-D Shapes
==========

3-D shapes are triangle meshes, and they come in two families that differ in
how Algan turns them into triangles.

* **Curved shapes:** Built on :class:`~algan.mobs.surfaces.surface.Surface`. These are tessellated
  dynamically so they stay smooth even when the camera moves close.
* **Faceted polyhedra:** Flat-sided solids built from explicit polygon faces.

.. algan:: GalleryShapes3D

    from algan import *

    shapes = Group([
        Sphere(radius=0.45), Cube(size=0.75), Cylinder(radius=0.35, height=0.8),
        Cone(base_radius=0.45, height=0.8), Torus(ring_radius=0.55, tube_radius=0.2),
        Tetrahedron(edge_length=1.0), Octahedron(edge_length=0.6),
        Icosahedron(edge_length=0.5),
    ])
    shapes.arrange_in_grid(2, row_buffer=0.6).spawn()
    shapes.wait()

    Scene.save_video()

Curved 3-D Shapes:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Notes
   * - :class:`~algan.mobs.shapes_3d.Sphere`
     - Sized with ``radius``.
   * - :class:`~algan.mobs.shapes_3d.Cylinder`
     - Sized with ``radius``, ``height``, and ``direction``.
   * - :class:`~algan.mobs.shapes_3d.Cone`
     - Sized with ``base_radius``, ``height``, and ``direction``.
   * - :class:`~algan.mobs.shapes_3d.Torus`
     - Sized with ``ring_radius`` and ``tube_radius``.
   * - :class:`~algan.mobs.shapes_3d.Dot3D`, :class:`~algan.mobs.shapes_3d.Line3D`
     - A small 3-D sphere and cylinder for marking points and 3-D segments.
   * - :class:`~algan.mobs.shapes_3d.Arrow3D`
     - A 3-D arrow combining a cylinder shaft and cone tip.

Faceted 3-D Shapes:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Notes
   * - :class:`~algan.mobs.shapes_3d.Cube`
     - Sized with ``size``.
   * - :class:`~algan.mobs.shapes_3d.Prism`
     - A 3-D box sized with ``width`` / ``height`` / ``depth``. Great for walls, pedestals, or floors.
   * - :class:`~algan.mobs.shapes_3d.Tetrahedron`, :class:`~algan.mobs.shapes_3d.Octahedron`, :class:`~algan.mobs.shapes_3d.Icosahedron`, :class:`~algan.mobs.shapes_3d.Dodecahedron`
     - Platonic solids sized with ``edge_length``.
   * - :class:`~algan.mobs.shapes_3d.Polyhedron`, :class:`~algan.mobs.shapes_3d.ConvexHull3D`
     - Custom 3-D solids constructed from your own vertices and faces, or point clouds.

Unlike 2-D shapes, 3-D shapes respond to light. See :doc:`../new_user_tutorials/three_d_basics` to
get started and
:doc:`../advanced_user_tutorials/lighting_and_shadows` for the full lighting
model.

Parametric Surfaces
===================

:class:`~algan.mobs.surfaces.surface.Surface` lets you build any custom curved 3-D surface by defining a
function that maps 2-D coordinates ``(u, v)`` in ``[0, 1]`` to 3-D points in
space:

.. algan:: GallerySurface
    :save_last_frame:

    from algan import *
    import torch

    def ripple(uv):
        x = uv[..., :1] * 4 - 2
        y = uv[..., 1:] * 4 - 2
        return torch.cat((x, y, torch.cos(torch.sqrt(x ** 2 + y ** 2) * 3) * 0.4), -1)

    Surface(ripple, checkered_color=BLUE).rotate(60, RIGHT).spawn()

    Scene.save_video()

The function receives a batched tensor of ``(u, v)`` pairs and must return the
matching points, so write it with torch operations rather than a Python loop.
:doc:`../new_user_tutorials/three_d_basics` works through this in more detail, and
:class:`~algan.mobs.surfaces.surface.Surface` also accepts texture maps -- see
:doc:`../advanced_user_tutorials/images_and_textures`.

Text and Mathematics
====================

:class:`~algan.mobs.text.Text` renders a string with a font; :class:`~algan.mobs.text.Tex` and
:class:`~.MathTex` render LaTeX. Both are cubic Bezier circuits, so they scale
and morph like any other 2-D shape.

.. algan:: GalleryText
    :save_last_frame:

    from algan import *

    with Off():
        Text("Text renders a font", font_size=48).move(UP * 0.9).spawn()
        Tex(r"e^{i\pi} + 1 = 0", font_size=56).spawn()
        MathTex(r"\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}",
                font_size=56, color=YELLOW).move(DOWN * 1.0).spawn()

    Scene.wait(1)

    Scene.save_video()

:doc:`../advanced_user_tutorials/text_and_math` covers these properly, including per-glyph animation and
the hand-writing effect.

Images and Imported Models
==========================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Notes
   * - :class:`~.ImageMob`
     - An image file or RGBA array as a flat, textured surface.
   * - :class:`~.Model3D`
     - A ``.glb`` / ``.gltf`` / ``.fbx`` model, with its materials and rigid
       node animation.
   * - :class:`~.ManimMob`
     - Any Manim Mobject, converted to Algan geometry.
   * - :class:`~.DotCloud`
     - A large set of points, drawn efficiently.

.. algan:: GalleryImageMob
    :save_last_frame:

    from algan import *

    ImageMob('world_map.png').scale(2).rotate(25, UP).spawn()

    Scene.save_video()

Image paths are resolved against the working directory and then against the
directory holding your script, so an image sitting beside your ``.py`` file
loads no matter where you launch Python from. See
:doc:`../advanced_user_tutorials/images_and_textures` and
:doc:`../advanced_user_tutorials/three_d_models`.

Diagrams and Plots
==================

Coordinate axes, function graphs, bar charts, and data tables come from Algan's
Manim compatibility layer. They animate as standard Algan Mobs:

.. algan:: GalleryAxes

    from algan import *
    import numpy as np

    axes = Axes(x_range=(-3, 3, 1), y_range=(-1.5, 1.5, 0.5), x_length=9, y_length=4.5)
    graph = axes.plot(lambda x: np.sin(x), color=YELLOW)
    with Off():
        axes.spawn()
        graph.spawn()

    with Seq(runtime=2):
        graph.rotate(20, UP)
        graph.rotate(-20, UP)

    Scene.save_video()

:class:`~.Axes`, :class:`~.NumberPlane`, :class:`~.NumberLine`,
:class:`~.BarChart`, :class:`~.Table`, :class:`~.Brace` and :class:`~.Graph` are
all available. They keep their Manim constructor arguments and methods
(``axes.plot``, ``axes.c2p``, ``axes.add_coordinates()``), and what those methods
return is itself an Algan Mob you can spawn and animate independently.

:doc:`../advanced_user_tutorials/importing_from_manim` covers this, and the route for anything the
compatibility layer does not expose.

Grouping
========

:class:`~algan.mobs.group.Group` collects Mobs so you can move, scale and color them as one,
and provides :meth:`~.Group.arrange_in_line` and
:meth:`~.Group.arrange_in_grid` for layout. See :doc:`../new_user_tutorials/child_mobs`.

Animated Numbers
================

:class:`~algan.mobs.numeric_display.DecimalNumber` shows a number you can animate, counting smoothly
between values:

.. algan:: GalleryDecimalNumber

    from algan import *

    counter = DecimalNumber(0.0, decimal_places=1,
                             integer_places=3).scale(2).spawn()
    counter.value = 99.9

    Scene.save_video()

:doc:`../advanced_user_tutorials/text_and_math` shows it counting over a longer run and covers the
formatting options.

Where To Next
=============

* :doc:`built_in_animations` -- the ready-made animations to apply to these
  shapes.
* :doc:`../advanced_user_tutorials/positioning_and_layout` -- getting these shapes where you want them.
* :doc:`../advanced_user_tutorials/text_and_math` -- labels, formulae and animated numbers in full.
* :doc:`../new_user_tutorials/three_d_basics` -- lighting, cameras and surfaces for the 3-D shapes.
* :doc:`../advanced_user_tutorials/importing_from_manim` -- axes, plots and the rest of the compatibility
  layer.
* :doc:`../advanced_user_tutorials/shaders_and_materials` -- how the 3-D shapes
  respond to light.
* :doc:`../advanced_user_tutorials/extending_algan` -- building a shape of your
  own, and packing thousands of them into one Mob.
