=====================
Manim Migration Guide
=====================

If you already know Manim, you do not need the tutorial series to get started.
This page maps what you know onto Algan: the ``self.play`` model becomes lazy,
Scene-owned recording; angles become degrees; mobjects become Mobs. It ends with
a step-by-step sequence for porting a larger project.

Two things are worth knowing before you read the rest:

* **Algan records animations rather than playing them.** Nothing renders until
  :meth:`~algan.scene.Scene.save_video` or
  :meth:`~algan.scene.Scene.save_frame` materializes the timeline.
* **Manim's geometry is still available.** Algan bundles Manim and exposes
  :class:`~algan.mobs.manim_mob.ManimMob` plus compatibility classes for
  ``Axes``, ``NumberPlane``, ``Brace`` and the rest, so a diagram you already
  have keeps working. :doc:`advanced_user_tutorials/importing_from_manim` covers
  that route in full.

.. seealso::

   :doc:`new_user_tutorials/index` -- the tutorial series, if you would rather
   learn Algan on its own terms than by translation.

A minimal translation
=====================

Manim:

.. algan-doc-check: skip -- the Manim side of a migration comparison

.. code-block:: python

    from manim import *

    class Example(Scene):
        def construct(self):
            square = Square(color=BLUE)
            self.play(Create(square))
            self.play(square.animate.shift(RIGHT))
            self.wait(1)

Algan:

.. algan:: MigratingHelloSquare

    from algan import *

    square = Square(color=BLUE).spawn()
    square.move(RIGHT)
    Scene.wait(1)

    Scene.save_video("example.mp4")

The Algan calls record events on ``scene.timeline_manager``. Rendering happens
only when ``save_video`` or ``save_frame`` materializes that timeline.

Adding and removing objects
===========================

Common translations are:

==============================  =============================================
Manim                           Algan
==============================  =============================================
``self.add(mob)``               ``mob.spawn(animate=False)``
``self.play(Create(mob))``      ``mob.spawn()``
``self.play(FadeOut(mob))``     ``mob.despawn()``
``self.remove(mob)``            ``mob.despawn(animate=False)``
``self.wait(t)``                ``Scene.wait(t)`` or ``mob.wait(t)``
==============================  =============================================

Before a Mob is spawned, modifications configure its initial state without
creating timeline animation. After spawning, animatable modifications record
animations by default.

.. seealso::

   :doc:`new_user_tutorials/getting_started` -- spawning and despawning from
   first principles.

Replacing ``self.play``
=======================

Algan methods such as :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move`,
:meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to`,
:meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.rotate`,
:meth:`~algan.animatable_base.mob.Mob.scale`,
:meth:`~algan.animatable_base.mob_morph.MobMorphMixin.become`, and animatable
attribute assignments record their own animations:

.. algan:: MigratingRecordedMethods

    from algan import *

    square = Square().spawn()
    square.move(RIGHT)
    square.rotate(90, OUT)
    square.color = RED

    Scene.save_video()

These operations are sequential unless placed in another animation context.
Use :class:`~algan.animation_timeline.animation_contexts.Sync` for one Manim
``self.play`` call containing several simultaneous animations:

.. algan:: MigratingSync

    from algan import *

    square = Square().spawn()
    with Sync():
        square.move(RIGHT)
        square.rotate(90, OUT)

    Scene.save_video()

There are four contexts, and between them they cover what ``self.play`` did:

=========================  ====================================================
Context                    What it does to the changes inside it
=========================  ====================================================
``Seq()``                  One after another. This is the default outside any
                           context.
``Sync()``                 All at once -- one Manim ``self.play`` with several
                           animations in it.
``Lag(ratio)``             Overlapping: each change starts when the previous one
                           is ``ratio`` of the way through. ``Lag(0)`` is
                           ``Sync``, ``Lag(1)`` is ``Seq``.
``Off()``                  Instantly, in a single frame, recording no animation.
=========================  ====================================================

``runtime`` sets a context's total runtime; ``runtime_per_part`` sets the default
runtime of each child animation inside it. Contexts nest, and a nested context
counts as one animation to its parent, which is what makes a complex multi-Mob
sequence readable:

.. algan:: MigratingNesting

    from algan import *

    square = Square(color=BLUE).move(RIGHT * 2).spawn()
    circle = Circle(color=YELLOW).move(LEFT * 2).spawn()

    with Sync():                 # the square and the circle move together
        with Seq():              # ... but the square's two moves are ordered
            square.move(LEFT)
            square.move(DOWN)
        circle.move(RIGHT)

    Scene.save_video()

.. seealso::

   :doc:`new_user_tutorials/combining_animations` -- all of this properly,
   including rate functions and the timing recipes.

Angles are in degrees
=====================

Manim measures angles in radians. Algan measures them in **degrees**, and this is
the easiest difference to miss, because a radian value is a perfectly legal
argument -- it simply produces a very small rotation instead of raising:

==============================  =============================================
Manim                           Algan
==============================  =============================================
``mob.rotate(PI / 2)``          ``mob.rotate(90)``
``mob.rotate(TAU)``             ``mob.rotate(360)``
``mob.rotate(-3 * TAU / 8)``    ``mob.rotate(-135)``
==============================  =============================================

If you would rather not convert in your head, multiply by ``DEGREES`` or
``RADIANS``. Both are exported by ``from algan import *``, and both yield the
degrees that Algan's API expects:

.. algan:: MigratingDegrees

    from algan import *

    square = Square().spawn()
    square.rotate(180 * DEGREES)  # 180 degrees
    square.rotate(PI * RADIANS)   # pi radians -- the same half turn

    Scene.save_video()

.. important::

    Algan's ``DEGREES`` is the reciprocal of Manim's. Manim's native unit is
    radians, so its ``DEGREES`` converts degrees *to* radians; Algan's native unit
    is already degrees, so ``DEGREES`` is 1. Copying ``rotate(90 * DEGREES)`` from
    a Manim script gives 90 degrees in Algan and 1.57 in Manim -- both correct, for
    different reasons.

A few Manim-parity surfaces keep Manim's radians on purpose. These take
**radians**, not degrees:

* ``RegularPolygon(start_angle=...)`` and ``Line(path_arc=...)``.
* ``Wiggle(rotation_angle=...)``.

The ``u_range`` / ``v_range`` parametric domains of
:class:`~algan.mobs.shapes_3d.Sphere`, ``Cone``, ``Cylinder`` and ``Torus`` keep
Manim's *names* but take Algan's **degrees**, so Manim's ``(0, PI)`` is
``(0, 180)`` here. All of them restrict the geometry, so a partial range builds a
partial shape: an open shell with uncapped cut edges, re-tessellated from
scratch rather than carved out of the whole shape's grid. Which parameter is
which follows Manim: ``Sphere``'s ``u_range`` is the azimuth and its
``v_range`` runs pole to pole, while ``Cylinder``'s and ``Cone``'s ``v_range``
is the azimuth about the axis (the extent along the axis comes from
``height``, as it does in Manim). The zero of each sweep is Algan's own seam
rather than Manim's, so a half shape comes out rotated relative to Manim's --
the docstrings say where each one starts.

Every one of those is a *constructor argument* handed straight to Manim. No
Algan **method** is among them, so there is no Mob anywhere whose ``rotate``
wants radians -- the Manim-compatibility mobs (``Axes``, ``NumberPlane``,
``Arc``, ``Brace``, ``VGroup`` and everything else deriving from
:class:`~algan.mobs.manim_compat.ManimCompatMob`) included.

Everything else -- including ``rotate``, ``orbit``, ``move(arc_angle=...)``,
camera field of view and Euler angles, and light cone angles -- is in degrees.

.. _migrating-manim-defaults:

Matching Manim's defaults
=========================

Manim's frame is 14.22 by 8 world units; Algan's default camera shows 12.44 by
7. So a ported script's geometry is the right shape but the wrong size, and
anything that positioned itself against Manim's frame lands somewhere Algan did
not choose.

:meth:`Scene.use_manim_defaults() <algan.scene.Scene.use_manim_defaults>` fixes
that for the whole Scene, rather than making you rescale each object. Call it
once, before you build anything:

.. algan:: MigratingManimDefaults

    from algan import *
    import manim as mn

    Scene.use_manim_defaults()

    square = ManimMob(mn.Square(color=mn.BLUE)).spawn()
    square.move(RIGHT)

    Scene.save_video()

It sets four things, and each can be declined with a keyword:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Keyword
     - What it takes from Manim
   * - ``camera``
     - Manim's 8-unit frame height and its ``ThreeDCamera``'s viewpoint -- a
       pinhole eye 20 units from the frame plane, so a vertical field of view of
       22.62 degrees. Manim's plain 2-D camera is orthographic instead, but the
       two agree exactly at ``z = 0``, so this one camera reproduces 2-D scenes
       exactly and 3-D scenes with Manim's own perspective.
   * - ``shading``
     - Manim's single light in Manim's position, ``ManimMaterial`` as the
       default material for 3-D Mobs, and tonemapping off -- so a flat fill
       comes out byte-identical to Manim's.
   * - ``background``
     - Black, Manim's default.

All three default to ``True``. Two extras default to ``False``:
``video_settings=True`` also switches the output to Manim's 1920x1080 at 60 fps,
and ``shape_defaults=True`` makes Algan's *own* shapes (``Square``, ``Circle``,
...) adopt Manim's colors and stroke styling rather than Algan's.

.. code-block:: python

    Scene.use_manim_defaults(shading=False)              # Manim framing, Algan lighting
    Scene.use_manim_defaults(shape_defaults=True)        # Manim-looking Algan shapes

The result is close enough that the two engines are hard to tell apart, but not
byte-identical.
:ref:`How close it gets <manim-defaults>` in
:doc:`advanced_user_tutorials/importing_from_manim` measures the residue --
flat fills are exact, unfilled strokes land within about a third of a pixel, a
filled shape's outline is offset by half its stroke width, and 3-D solids are
shaded by a ray tracer rather than by Manim's two-point gradient.

Without ``use_manim_defaults``, imported diagrams usually want a modest
:meth:`~algan.animatable_base.mob.Mob.scale` or
:meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.fit_to_screen`
instead.

Transforms
==========

Use :meth:`~algan.animatable_base.mob_morph.MobMorphMixin.become` for geometry
morphing:

.. algan:: MigratingBecome

    from algan import *

    shape = Circle().spawn()
    shape = shape.become(Square(add_to_scene=False))

    Scene.save_video()

Passing ``add_to_scene=False`` avoids registering the temporary target as a
separate actor, saving some compute and memory -- and stopping Algan from
warning that you built a Mob and never spawned it.

Text, TeX, and imported Manim mobjects
======================================

Algan provides native :class:`~algan.mobs.text.Text` and
:class:`~algan.mobs.text.Tex` mobs. ``MathTex`` and ``Title`` are *not* native:
they are compatibility wrappers around Manim's classes, so they take Manim's
arguments (``tex_to_color_map``, ``include_underline``) but are single Mobs with
no per-glyph views -- ``formula[0]`` raises, and ``write()`` and ``get_segment()``
live on :class:`~algan.mobs.text.Tex`. Where a script does not need Manim's own
arguments, ``Tex`` is the better landing place, and it produces the same
outlines: ``MathTex("x^2")`` matches ``Tex("x^2", font_size=48)``.

.. note::

    ``Title`` is the mobject most people meet when Manim's frame size bites (see
    `Matching Manim's defaults`_ above). Manim's ``to_edge(UP)`` leaves a 0.5 gap
    below its own frame top, putting the title's top at ``y = 3.5`` -- exactly
    Algan's top border. Nothing is cut off, but it sits flush against the frame
    edge with no margin. Call
    :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to_screen_edge` to
    inset it by Algan's usual buffer, or place it by hand with
    ``.move(DOWN * 1)``.

For a compatible Manim vector mobject, wrap it in
:class:`~algan.mobs.manim_mob.ManimMob`:

.. algan:: MigratingManimMob

    from algan import *
    import manim as mn

    diagram = ManimMob(mn.ComplexPlane().add_coordinates()).scale(0.5).spawn()
    diagram.rotate(30, OUT)

    Scene.save_video("diagram.mp4")

``ManimMob`` converts cubic-Bezier vector geometry and supported image
submobjects. It does not provide an arbitrary bridge to Manim's renderer, so
unsupported mobject geometry can still raise ``NotImplementedError``.

.. important::

    Do not use both ``from algan import *`` and ``from manim import *`` -- the
    two libraries share many names and the definitions would clash. Import one
    of them under a short alias, as with ``import manim as mn`` above.

.. seealso::

   * :doc:`advanced_user_tutorials/text_and_math` -- Algan's own text and LaTeX,
     per-glyph animation and the hand-writing effect.
   * :doc:`advanced_user_tutorials/importing_from_manim` -- the compatibility
     layer in detail, including axes, plots, SVG import and 3-D Manim geometry.

Groups and hierarchies
======================

Algan's :class:`~algan.mobs.group.Group` can be used to group any type of Algan
Mob:

.. algan:: MigratingGroup

    from algan import *

    dots = Group([Circle(radius=0.1) for _ in range(8)])
    dots.arrange_in_line(RIGHT).spawn()
    dots.rotate(180, OUT)

    Scene.save_video()

Parent transformations propagate to descendants. Slices return Group views
without adding new actors to the Scene. Do not combine children from multiple
Scenes.

.. seealso::

   :doc:`new_user_tutorials/child_mobs` -- parent/child propagation, Groups and
   the layout helpers.

Camera and lights
=================

A new Algan Scene is initialized with a camera and a point light. Access them
through the Scene:

.. algan:: MigratingCameraLights

    from algan import *

    ball = Sphere(radius=1, color=BLUE).spawn()

    camera = Scene.get_camera()
    Scene.clear_lights()
    PointLight(location=UP * 4 + LEFT * 4 + OUT * 4).spawn()

    with Seq(runtime=3, easing=easings.identity):
        camera.rotate(180, UP, about=ORIGIN)

    Scene.save_video()

Unlike Manim's renderer configuration, camera, lights, environment maps, and
audio state belong to each Scene.

.. seealso::

   * :doc:`advanced_user_tutorials/cameras` -- field of view, clipping and
     camera animation.
   * :doc:`advanced_user_tutorials/lighting_and_shadows` -- every light type,
     shadows and environment maps.

Rendering output
================

Use the unified Scene APIs:

.. code-block:: python

    scene.save_video("preview.mp4", PREVIEW)
    scene.save_frame("thumbnail.png", THUMBNAIL)

A bare filename is placed in Algan's configured output directory. A relative or
absolute path containing a parent directory is honored directly.

.. seealso::

   :doc:`advanced_user_tutorials/saving_videos_and_images` -- quality presets,
   output paths and rendering stills.

Settings and quality
====================

Runtime defaults live under :ref:`algan.SETTINGS <reference-settings>`:

.. code-block:: python

    SETTINGS.video.set(HD)
    SETTINGS.raytracing.set(samples_per_pixel=4)

Built-in presets are immutable, so this creates a modified copy:

.. code-block:: python

    fast_hd = HD.set(frames_per_second=24, supersampling=1)

The render device is a setting: ``SETTINGS.computing.set(render_device="cuda")``,
at the top of the script. The animation device is not -- set
``ALGAN_ANIMATION_DEVICE`` before importing Algan.

.. seealso::

   :doc:`advanced_user_tutorials/settings` -- the settings system in full, and
   :doc:`advanced_user_tutorials/performance_and_quality` for which knob to
   reach for.

Migrating larger projects
=========================

A reliable migration sequence is:

#. Put one logical animation in ``with Scene() as scene:`` and render through
   ``scene.save_video``.
#. Translate object creation and initial positioning before adding ``spawn``.
#. Replace each ``self.play`` group with direct animated calls inside ``Sync``.
#. Replace chained ``animate`` operations with Mob methods or animatable
   attribute assignments.
#. Construct transform targets with ``add_to_scene=False``.
#. Wrap unsupported Manim vector objects with ``ManimMob`` selectively.
#. Move quality, path, and ray-tracing defaults to ``SETTINGS`` or per-render
   ``video_settings`` arguments.

Validate migrated scenes visually. Algan uses a different renderer and material
model, so even equivalent geometry and timing are not expected to be pixel
identical to Manim output -- ``use_manim_defaults()`` is how you get as close as
Algan gets.

Once a project has more than a few scenes, Algan's
:class:`~algan.project.Project` replaces Manim's one-class-per-scene convention:
see :doc:`advanced_user_tutorials/multi_scene_projects`.

Manim names that still work
===========================

Algan otherwise gives each thing exactly one name, but the compatibility layer
is a deliberate exception: a handful of Manim spellings are exported so a ported
script keeps reading the way its author wrote it. ``Mobject`` is
:class:`~algan.animatable_base.mob.Mob`, ``GenericGraph`` is
:class:`~algan.mobs.manim_compat.Graph`, and Manim's OpenGL-renderer class names
(``OpenGLVMobject``, ``OpenGLGroup``, ...) resolve to their renderer-independent
Algan equivalents. They are the same objects, not wrappers -- ``Mob is Mobject``
is ``True`` -- so there is no conversion step and no behavioural difference.

Prefer the Algan name in new code; there is nothing to fix in old code.

Manim names that do not
=======================

Those spellings are the exception, not the rule. Where Manim's word for
something is not Algan's, the root namespace carries Algan's only, and the
Manim spelling raises an error naming the one to use. Everything in this table
still works exactly as written under ``import algan.manim as mn``, which is
that namespace's whole purpose.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Manim's spelling, at the root
     - Algan's
   * - ``mobject=`` on ``Brace``, ``Indicate``, ``Circumscribe``, ``ApplyMatrix``,
       ``Wiggle`` and the rest
     - ``mob=``
   * - ``element_to_mobject=`` on ``Table``, ``Matrix`` and their variants
     - ``element_to_mob=``
   * - ``SVGMobject``, ``MobjectMatrix``, ``MobjectTable``, ``DashedVMobject``,
       ``CurvesAsSubmobjects``
     - ``SVGMob``, ``MobMatrix``, ``MobTable``, ``DashedMob``,
       ``CurvesAsChildren``
   * - ``Cone(base_radius=..., show_base=..., u_min=...)``,
       ``Cylinder(show_ends=...)``, ``checkerboard_colors=``
     - ``radius``, ``closed``, ``u_range`` -- one vocabulary across
       ``Sphere``, ``Cone``, ``Cylinder``, ``Torus`` and ``Surface``, with
       ``direction`` defaulting to ``UP`` on both ``Cone`` and ``Cylinder``.
       The checkerboard is a texture map rather than a second vertex colour:
       ``color_texture=get_checkerboard((BLUE, BLUE_E))``
   * - ``RegularPolygon(num_vertices=...)``
     - ``RegularPolygon(n=...)``
   * - ``Dot(point=...)``
     - ``Dot(location=...)``, the name of the attribute it sets

An angle is the one mistake that cannot be caught this way, because a radian
measure is a legal degree measure. ``Arc(angle=PI / 2)`` builds a 1.57 degree
sliver rather than a quarter arc, so a non-integer angle smaller than a full
turn warns and says what the degree spelling would be::

    `angle=1.5708` looks like radians; Algan takes degrees (did you mean `angle=90`?)

Whole numbers never warn -- ``Arc(angle=5)`` is a real five degree arc.

Manim animations, on the other hand, do not come across: ``Transform``,
``Create``, ``FadeIn`` and friends have no meaning on Algan's timeline. Most of
the common ones have a direct equivalent in
:doc:`galleries/built_in_animations`.

Where To Next
=============

* :doc:`galleries/mob_gallery` -- every built-in Mob, with pictures.
* :doc:`galleries/built_in_animations` -- the equivalents of Manim's
  ``Indicate``, ``Circumscribe``, ``ApplyMatrix`` and the rest.
* :doc:`advanced_user_tutorials/importing_from_manim` -- the compatibility layer
  in detail, and the route for anything it does not expose.
* :doc:`new_user_tutorials/index` -- the tutorial series, if you would rather
  learn Algan on its own terms than by translation.
* :doc:`reference` -- the full API reference.
