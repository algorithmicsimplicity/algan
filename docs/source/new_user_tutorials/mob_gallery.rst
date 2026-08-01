===============
The Mob Gallery
===============

This is a tour of the Mobs Algan ships with. Every one of them is a
:class:`.Mob`, so everything in :doc:`basic_animations` applies to all of them:
spawn them, move them, colour them, morph them into each other.

The complete list, with every constructor argument, is in the
:doc:`mobs reference <../reference_index/mobs>`.

2-D Shapes
==========

Algan's 2-D shapes are cubic Bezier circuits (:class:`~.BezierCircuitCubic`),
which means they stay perfectly smooth however far you zoom in -- a
:class:`~.Circle` is a real circle, not a many-sided polygon.

.. algan:: GalleryShapes2D

    from algan import *

    shapes = Group([
        Circle(), Square(), Triangle(), RegularPolygon(6),
        Rectangle(width=1.4, height=0.8), Line(LEFT * 0.5, RIGHT * 0.5),
        Dot(), Point(),
    ])
    shapes.arrange_in_grid(2, buffer=0.9).scale(0.7).spawn()
    shapes.wait()

    Scene.save_video()

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Notes
   * - :class:`~.Circle`
     - ``radius``.
   * - :class:`~.Dot`
     - A small filled circle, for marking points.
   * - :class:`~.Square`
     - ``side_length``.
   * - :class:`~.Rectangle`
     - ``width``, ``height``.
   * - :class:`~.RegularPolygon`
     - ``n`` sides. :class:`~.Triangle` is ``RegularPolygon(3)``.
   * - :class:`~.Polygon`
     - An arbitrary closed outline through the points you give it.
   * - :class:`~.Quad`
     - A four-cornered shape from four explicit corners.
   * - :class:`~.Line`
     - From a start point to an end point.
   * - :class:`~.Point`
     - A single point; mostly useful as an invisible anchor.
   * - :class:`~.SurroundingRectangle`
     - A box drawn around another Mob, sized to fit it.
   * - :class:`~.Arc`, :class:`~.Annulus`, :class:`~.Ellipse`, :class:`~.Star`, :class:`~.Arrow`
     - Extra outline shapes.

Every 2-D shape takes ``color``, plus ``border_width`` and ``border_color``
for its outline:

.. code-block:: python

    Circle(color=BLUE, border_color=WHITE, border_width=4)

On a filled shape the border is drawn *inside* the outline, so raising
``border_width`` eats into the fill instead of growing the silhouette --
bordered text stays legible and neighbouring glyphs never fuse. An unfilled
shape (``filled=False``, and :class:`~.Line`) has no interior to eat into, so
its stroke stays centred on the path.

3-D Shapes
==========

3-D shapes are triangle meshes. Algan tessellates them to the resolution the
current render needs, so they stay smooth as you move the camera in.

.. algan:: GalleryShapes3D

    from algan import *

    shapes = Group([
        Sphere(radius=0.45), Cube(side_length=0.75), Cylinder(radius=0.35, height=0.8),
        Cone(base_radius=0.45, height=0.8), Torus(major_radius=0.55, minor_radius=0.2),
        Tetrahedron(edge_length=1.0), Octahedron(edge_length=0.6),
        Icosahedron(edge_length=0.5),
    ])
    shapes.arrange_in_grid(2, buffer=0.6).spawn()
    shapes.wait()

    Scene.save_video()

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Notes
   * - :class:`~.Sphere`
     - ``radius``.
   * - :class:`~.Cube`
     - ``side_length``. A :class:`~.Prism` with equal sides.
   * - :class:`~.Prism`
     - ``dimensions`` -- a box. Handy as a floor or a wall.
   * - :class:`~.Cylinder`
     - ``radius``, ``height``, ``direction``.
   * - :class:`~.Cone`
     - ``base_radius``, ``height``, ``direction``.
   * - :class:`~.Torus`
     - ``major_radius``, ``minor_radius``.
   * - :class:`~.Tetrahedron`, :class:`~.Octahedron`, :class:`~.Icosahedron`, :class:`~.Dodecahedron`
     - Platonic solids, sized by ``edge_length``.
   * - :class:`~.Polyhedron`, :class:`~.ConvexHull3D`
     - Build a solid from your own vertices and faces, or from a point cloud.
   * - :class:`~.Dot3D`, :class:`~.Line3D`, :class:`~.Arrow3D`
     - 3-D equivalents of the 2-D markers, for annotating 3-D scenes.

.. note::

    Watch the defaults: :class:`~.Torus` defaults to ``major_radius=3``, which
    is wider than the visible frame. Pass explicit sizes when you are laying
    several shapes out together.

Unlike 2-D shapes, 3-D shapes respond to light. See :doc:`three_d_basics` to
get started and
:doc:`../advanced_user_tutorials/lighting_and_shadows` for the full lighting
model.

Arbitrary Surfaces
==================

:class:`~.Surface` builds a curved surface from a function mapping 2-D
parameters ``(u, v)`` -- both in ``[0, 1]`` -- to points in space. Every 3-D
shape above is a :class:`~.Surface` underneath.

.. algan:: GallerySurface

    from algan import *
    import torch

    def saddle(uv):
        x = uv[..., :1] * 4 - 2
        y = uv[..., 1:] * 4 - 2
        return torch.cat((x, y, (x ** 2 - y ** 2) * 0.4), -1)

    surface = Surface(saddle, checkered_color=BLUE).spawn()
    with Seq(run_time=3):
        surface.rotate(60, RIGHT)
        surface.rotate(360, OUT)

    Scene.save_video()

The function receives a batched tensor of ``(u, v)`` pairs and must return the
matching points, so write it with torch operations rather than a Python loop.
:class:`~.Surface` also accepts texture maps -- see
:doc:`../advanced_user_tutorials/images_and_textures`.

Text and Mathematics
====================

:class:`~.Text` renders a string with a font; :class:`~.Tex` and
:class:`~.MathTex` render LaTeX. Both are cubic Bezier circuits, so they scale
and morph like any other 2-D shape.

.. algan:: GalleryText

    from algan import *

    title = Text("Euler's identity", font_size=64).move(UP * 1.5).spawn()
    formula = Tex(r"e^{i\pi} + 1 = 0", font_size=80).spawn()

    with Seq(run_time=3):
        formula.color = YELLOW
        title.move(UP * 0.5)

    Scene.save_video()

:doc:`text_and_math` covers these properly, including per-glyph animation and
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
   * - :class:`~.ThreeDModelMob`
     - A ``.glb`` / ``.gltf`` / ``.fbx`` model, with its materials and rigid
       node animation.
   * - :class:`~.ManimMob`
     - Any Manim Mobject, converted to Algan geometry.
   * - :class:`~.DotCloud`
     - A large set of points, drawn efficiently.

.. algan:: GalleryImageMob

    from algan import *

    photo = ImageMob('world_map.png').scale(2).spawn()
    with Seq(run_time=2):
        photo.rotate(30, UP)
        photo.rotate(-30, UP)

    Scene.save_video()

Image paths are resolved against the working directory and then against the
directory holding your script, so an image sitting beside your ``.py`` file
loads no matter where you launch Python from. See
:doc:`../advanced_user_tutorials/images_and_textures` and
:doc:`../advanced_user_tutorials/three_d_models`.

Diagrams and Plots
==================

Axes, plots, number lines and other diagram furniture come from Algan's Manim
compatibility layer, built on Algan's bundled Manim geometry. They are available
under their own names and animate as ordinary Algan Mobs:

.. algan:: GalleryAxes

    from algan import *
    import numpy as np

    axes = Axes(x_range=(-3, 3, 1), y_range=(-1.5, 1.5, 0.5), x_length=9, y_length=4.5)
    graph = axes.plot(lambda x: np.sin(x), color=YELLOW)
    with Off():
        axes.spawn()
        graph.spawn()

    with Seq(run_time=2):
        graph.rotate(20, UP)
        graph.rotate(-20, UP)

    Scene.save_video()

:class:`~.Axes`, :class:`~.NumberPlane`, :class:`~.NumberLine`,
:class:`~.BarChart`, :class:`~.Table`, :class:`~.Brace` and :class:`~.Graph` are
all available. They keep their Manim constructor arguments and methods
(``axes.plot``, ``axes.c2p``, ``axes.add_coordinates()``), and what those methods
return is itself an Algan Mob you can spawn and animate independently.

:doc:`importing_from_manim` covers this, and the route for anything the
compatibility layer does not expose.

Grouping
========

:class:`~.Group` collects Mobs so you can move, scale and colour them as one,
and provides :meth:`~.Group.arrange_in_line` and
:meth:`~.Group.arrange_in_grid` for layout. See :doc:`child_mobs`.

Numbers
=======

:class:`~.NumericDisplay` shows a number you can animate, counting smoothly
between values:

.. algan:: GalleryNumericDisplay

    from algan import *

    counter = NumericDisplay(0.0, num_decimal_places=2).scale(2).spawn()
    with Seq(run_time=3):
        counter.value = 100.0

    Scene.save_video()
