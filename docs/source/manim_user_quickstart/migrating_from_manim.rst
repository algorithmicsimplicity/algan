====================
Migrating from Manim
====================

Algan keeps many familiar Manim concepts—mobjects, spatial constants, grouped
objects, text, TeX, and scene timelines—but its authoring model is different.
The most important migration is not a class-name substitution: it is moving
from immediate ``Scene.play`` calls to lazy, Scene-owned animation recording.

A minimal translation
=====================

Manim:

.. code-block:: python

    from manim import *

    class Example(Scene):
        def construct(self):
            square = Square(color=BLUE)
            self.play(Create(square))
            self.play(square.animate.shift(RIGHT))
            self.wait(1)

Algan:

.. code-block:: python

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

Replacing ``self.play``
=======================

Algan methods such as ``move``, ``move_to``, ``rotate``, ``scale``, ``become``,
and animatable attribute assignments record their own animations:

.. code-block:: python

    square = Square().spawn()
    square.move(RIGHT)
    square.rotate(90, OUT)
    square.color = RED

These operations are sequential unless placed in another animation context.
Use :class:`~algan.animation_timeline.animation_contexts.Sync` for one Manim
``self.play`` call containing several simultaneous animations:

.. code-block:: python

    with Sync():
        square.move(RIGHT)
        square.rotate(90, OUT)

Use ``Seq``, ``Lag``, and ``Off`` for sequential, overlapping, and instantaneous
changes. ``run_time`` controls a context's total duration; ``run_time_unit``
controls the default duration of each child animation.

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

.. code-block:: python

    square.rotate(180 * DEGREES)  # 180 degrees
    square.rotate(PI * RADIANS)   # pi radians -- the same half turn

.. important::

    Algan's ``DEGREES`` is the reciprocal of Manim's. Manim's native unit is
    radians, so its ``DEGREES`` converts degrees *to* radians; Algan's native unit
    is already degrees, so ``DEGREES`` is 1. Copying ``rotate(90 * DEGREES)`` from
    a Manim script gives 90 degrees in Algan and 1.57 in Manim -- both correct, for
    different reasons.

A few Manim-parity surfaces keep Manim's radians on purpose. These take
**radians**, not degrees:

* The Manim-compatibility mobs -- ``Axes``, ``NumberPlane``, ``Arc``, ``Brace``,
  ``VGroup`` and everything else deriving from
  :class:`~algan.mobs.manim_compat.ManimCompatMob` -- override ``rotate`` to
  forward straight to Manim, so ``angle`` there is in radians.
  :class:`~algan.mobs.manim_mob.ManimMob` itself is an ordinary Algan Mob and its
  ``rotate`` takes degrees.
* ``RegularPolygon(start_angle=...)`` and ``Line(path_arc=...)``.
* ``Wiggle(rotation_angle=...)``.
* The ``u_range`` / ``v_range`` parametric domains of :class:`~algan.mobs.shapes_3d.Sphere`,
  ``Cone``, ``Cylinder`` and ``Torus`` -- these are parameter intervals rather
  than rotations.

Everything else -- including ``rotate``, ``orbit``, ``move(path_arc_angle=...)``,
camera field of view and Euler angles, and light cone angles -- is in degrees.

Transforms
==========

Use :meth:`~algan.animatable_base.mob_morph.MobMorphMixin.become` for geometry morphing:

.. code-block:: python

    shape = Circle().spawn()
    shape = shape.become(Square(add_to_scene=False))

Passing ``add_to_scene=False`` avoids
registering the temporary target as a separate actor, saving some compute and memory.

Text, TeX, and imported Manim mobjects
======================================

Algan provides native :class:`~algan.mobs.text.Text` and
:class:`~algan.mobs.text.Tex` mobs. For a compatible Manim vector mobject, wrap
it in :class:`~algan.mobs.manim_mob.ManimMob`:

.. code-block:: python

    import manim as mn
    from algan import *

    source = mn.SVGMobject("diagram.svg")
    diagram = ManimMob(source).spawn()
    diagram.rotate(30, OUT)
    Scene.save_video("diagram.mp4")

``ManimMob`` converts cubic-Bezier vector geometry and supported image
submobjects. It does not provide an arbitrary bridge to Manim's renderer, so
unsupported mobject geometry can still raise ``NotImplementedError``.

Groups and hierarchies
======================

Algan's :class:`~algan.mobs.group.Group` can be used to group any type
of Algan Mob.

.. code-block:: python

    dots = Group(*(Circle(radius=0.1) for _ in range(8)))
    dots.arrange_in_line(RIGHT).spawn()

Parent transformations propagate to descendants. Slices return Group views
without adding new actors to the Scene. Do not combine children from multiple
Scenes.

Camera and lights
=================

A new Algan Scene is initialized with a camera and a point light. Access them
through the Scene:

.. code-block:: python

    camera = Scene.get_camera()
    lights = Scene.get_light_sources()

    camera.move(OUT)
    Scene.clear_light_sources()
    PointLight(location=UP + LEFT + OUT).spawn()

Unlike Manim's renderer configuration, camera, lights, environment maps, and
audio state belong to each Scene.

Rendering output
================

Use the unified Scene APIs:

.. code-block:: python

    scene.save_video("preview.mp4", PREVIEW)
    scene.save_frame("thumbnail.png", THUMBNAIL)

A bare filename is placed in Algan's configured output directory. A relative or
absolute path containing a parent directory is honored directly.

Settings and quality
====================

Runtime defaults live under :data:`algan.SETTINGS`:

.. code-block:: python

    SETTINGS.video.set(HD)
    SETTINGS.raytracing.set(samples_per_pixel=4)

Built-in presets are immutable, so this creates a modified copy:

.. code-block:: python

    fast_hd = HD.set(frames_per_second=24, anti_alias_level=1)

Device selection is initialization-only. Set ``ALGAN_RENDER_DEVICE`` and
``ALGAN_ANIMATION_DEVICE`` environment variables before importing Algan rather than assigning a
runtime device field.

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
identical to Manim output.
