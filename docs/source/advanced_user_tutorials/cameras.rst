=======
Cameras
=======

Every Scene has one :class:`~algan.rendering.camera.Camera`, and it is a
:class:`~algan.animatable_base.mob.Mob` -- so you move,
rotate and animate it with the same methods you use on everything else, inside the
same animation contexts.

.. code-block:: python

    from algan import *

    camera = Scene.get_camera()

By default it sits at ``ORIGIN + OUT * 7`` looking at the ``ORIGIN``, with a
perspective projection whose vertical field of view is about 53°. With the default
resolution that makes the visible area at the origin plane roughly 12.4 × 7 world
units.

Moving the Camera
=================

The camera responds to every :class:`~algan.animatable_base.mob.Mob` movement and
orientation method.
:doc:`../new_user_tutorials/positioning_and_layout` lists them all; these are the
ones that matter for camera work:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Call
     - Effect
   * - ``camera.move(OUT * 2)``
     - Dolly back, keeping the same aim.
   * - ``camera.rotate(deg, UP, about_point=ORIGIN)``
     - Turntable: swing around the scene, staying pointed at it.
   * - ``camera.look_at(point)``
     - Turn to face a point without moving.
   * - ``camera.orbit(deg, UP, about_point=p)``
     - Travel around ``p`` *without* turning -- a tracking shot past the subject.
   * - ``camera.move_to_make_mob_center_of_view(mob)``
     - Reframe so a given Mob is centred.

The turntable is the workhorse of 3-D explanation, and the thing to get right is
using :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.rotate`
rather than :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.orbit`:

.. algan:: CameraTurntable

    from algan import *

    with Off():
        Group([Cube(side_length=0.8, color=BLUE).move(RIGHT * 1.6 * i)
               for i in (-1, 0, 1)]).spawn()

    with Seq(run_time=4, rate_func=rate_funcs.identity):
        Scene.get_camera().rotate(360, UP, about_point=ORIGIN)

    Scene.save_video()

``rotate`` carries the camera's orientation with it, so it stays aimed at the
origin all the way round. ``orbit`` would move it along the same circle while
leaving it pointing in its original direction, and the scene would swing out of
frame.

.. important::

    Give camera moves ``rate_func=rate_funcs.identity``. The default
    ``rate_funcs.smooth`` eases in and out, which reads as the *world* speeding up
    and slowing down rather than as a camera move.

    Moving the camera is not expensive in itself. The renderer's acceleration
    structure bounds where the *geometry* goes, so sweeping the camera around a
    scene that is standing still does not inflate it. What costs is how much of
    the frame the geometry fills: surfaces are diced more finely as they grow on
    screen, and a batch of frames is tessellated for its most demanding frame.
    A move that ends in a close-up is the expensive kind -- see
    :doc:`performance_and_quality`.

Following a subject
===================

For a camera that keeps tracking something whose path you do not know in advance,
use an updater (see :doc:`../new_user_tutorials/updaters`):

.. algan:: CameraTracking

    from algan import *

    with Off():
        ball = Sphere(radius=0.6, color=YELLOW).spawn()
        Group([Cube(side_length=0.5, color=BLUE).move(RIGHT * x + DOWN * 1.5)
               for x in (-3, 0, 3)]).spawn()

    camera = Scene.get_camera()
    camera.add_updater(lambda self, t: self.look_at(ball.location))

    with Seq(run_time=3):
        ball.move(RIGHT * 3 + UP * 1.5)
        ball.move(LEFT * 6)

    Scene.save_video()

.. _camera-fov:

Field of View
=============

``fov`` is the vertical field of view in degrees. A small fov is a telephoto lens
-- flattened perspective, as though the subject were far away; a large fov is
wide-angle, with exaggerated depth:

.. algan:: CameraFov

    from algan import *

    with Off():
        Group([Cube(side_length=0.8, color=BLUE).move(IN * 1.6 * i + RIGHT * 0.9 * i)
               for i in range(4)]).spawn()

    camera = Scene.get_camera()
    with Seq(run_time=3):
        camera.set_fov(20)
        camera.set_fov(90)

    Scene.save_video()

.. code-block:: python

    camera.set_fov(30)          # telephoto
    camera.fov = 30             # equivalent property form
    print(camera.get_fov())     # read it back

:meth:`~algan.rendering.camera.Camera.set_fov` works by moving the camera's screen,
so on a spawned
camera it animates like any other camera change -- which is what makes the
dolly-zoom above possible.

Algan also exposes the underlying perspective controls directly:
:meth:`~algan.rendering.camera.Camera.set_distance_to_screen` moves the focus point relative to the
screen plane, and the constructor's ``screen_distance`` / ``screen_scale`` set
them up front. ``fov`` is derived from these, so use one or the other, not both.

.. _camera-aspect-fov:

Aspect ratio widens the horizontal field of view
------------------------------------------------

``fov`` is *vertical* (as in Three.js). The horizontal field of view is derived
from it and the output aspect ratio, so **changing the resolution's shape changes
how wide the camera sees** while the vertical stays fixed. The default 53 degree
vertical fov gives roughly:

=====================  ==============================
Resolution             Horizontal field of view
=====================  ==============================
1920 x 1080 (16:9)     ~82 degrees
2560 x 1080 (21:9)     ~96 degrees
1438 x 426 (3.4:1)     ~119 degrees
=====================  ==============================

At 119 degrees the frame edge is 59 degrees off axis, and a sphere there is
projected as an ellipse stretched by ``1 / cos(59 deg)``, about 1.9x. That is
correct perspective -- a real 119 degree lens does the same thing -- but it is
rarely what a wide banner or panorama still is after.

For the near-orthographic look of a long lens, narrow the fov and pull the camera
back by the same factor, so that ``distance * tan(fov / 2)`` (the visible
half-height at the subject) is unchanged:

.. code-block:: python

    import math

    camera = Scene.get_camera()            # starts at OUT * 7, 3.5 units of
    camera.set_fov(math.degrees(2 * math.atan(3.5 / 70)))   # half-height at z=0
    camera.move_to(OUT * 70)               # 10x the distance, same framing

Use :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to` or
``move`` to reposition a camera. ``move_center_to`` centres a *bounding box*, and
a Camera's box spans both it and its internal screen plane, so the camera would
land half the screen distance too far back.

If you want no perspective at all, see :ref:`Orthographic Projection
<camera-orthographic>` below.

.. _camera-orthographic:

Orthographic Projection
=======================

An orthographic camera has no perspective at all: parallel lines stay parallel and
distance does not change apparent size. That is what you want for technical
diagrams, cross-sections, and anything where the viewer must compare sizes.

.. algan:: CameraOrthographic

    from algan import *

    with Off():
        Scene.get_camera().set_to_orthographic()
        cubes = Group([Cube(side_length=0.8, color=BLUE).move(IN * 1.6 * i + RIGHT * 0.9 * i)
                       for i in range(4)]).spawn()

    with Seq(run_time=3):
        cubes.rotate(360, UP, about_point=ORIGIN)

    Scene.save_video()

All four cubes are the same apparent size however far away they are. Compare it
with :ref:`the perspective version <camera-fov>` above.

:meth:`~algan.rendering.camera.Camera.set_near_orthographic` gives a near-orthographic projection --
a very long lens rather than a true parallel projection -- which keeps a little
depth cue while staying nearly distortion-free.

Clipping Planes
===============

``near`` and ``far`` are clip distances measured from the camera. Geometry closer
than ``near`` or further than ``far`` is not drawn; past ``far`` the background or
environment map shows through. ``0`` disables each, which is the default.

.. algan:: CameraClipping

    from algan import *

    with Off():
        Scene.get_camera().set_far(11)
        Group([Sphere(radius=0.4, color=BLUE).move(IN * 1.8 * i + RIGHT * 1.1 * i)
               for i in range(5)]).spawn()

    Scene.wait(1)

    Scene.save_video()

Five spheres are spawned; the two beyond 11 units are clipped away.

.. code-block:: python

    camera.set_near(0.5)        # hide anything within 0.5 units of the camera
    camera.set_far(50)          # hide anything beyond 50 units
    camera.set_far(0)           # no far clipping (the default)

The most common use for ``near`` is stopping foreground geometry from filling the
frame when the camera pushes into a scene.

.. important::

    Like the projection mode, the clip planes are camera *configuration* rather
    than animated attributes: they are read when a frame batch is prepared, not
    recorded on the timeline. Set them once, before spawning, and render separate
    videos if you need to show two different settings.

Screen Coordinates
==================

The camera is also what converts between world space and what the viewer sees, so
these Mob methods all resolve against it:

* :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to_screen_position` and
  :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.move_center_to_screen_position` -- place a Mob at fractional
  screen coordinates.
* :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to_edge` and
  :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to_corner` -- rest against a
  screen border.
* :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.fit_to_screen_rectangle` -- scale and move to fill a screen
  rectangle.

Each of them resolves the camera *once*, when the call is recorded, so a later
camera move will not keep the Mob pinned there. For something that must stay in a
fixed screen position through a camera move -- a caption, a legend -- attach it to
the camera as a child, or drive it with an updater:

.. code-block:: python

    caption = Text("figure 1", font_size=32)
    Scene.get_camera().add_children([caption])
    with Off():
        caption.move_to_screen_position(0.15, 0.1)
        caption.spawn()

Because children follow their parent's basis (see
:doc:`../new_user_tutorials/child_mobs`), the caption now travels with the camera.

See Also
========

* :doc:`lighting_and_shadows` -- lights, and the rig that goes with a camera move.
* :doc:`performance_and_quality` -- what actually makes a render expensive, and what
  to do about it.
* :doc:`../new_user_tutorials/three_d_basics` -- the gentler introduction.
