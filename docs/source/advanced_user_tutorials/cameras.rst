=======
Cameras
=======

Every Scene in Algan has one :class:`~algan.rendering.camera.Camera`. And
because the camera is itself a :class:`~algan.animatable_base.mob.Mob`, you move,
rotate, and animate it using the exact same methods you use for everything else.

.. code-block:: python

    from algan import *

    camera = Scene.get_camera()

By default, the camera sits at ``ORIGIN + OUT * 7`` aimed towards ``ORIGIN``,
using a perspective projection with a vertical field of view of ~53°. At the
origin plane (``z = 0``), the default framing spans approximately 12.4 × 7 world
units.

Moving and Animating the Camera
===============================

The camera supports all standard :class:`~algan.animatable_base.mob.Mob` movement and
orientation method. These are the
ones that matter for camera work:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Effect
   * - ``camera.move(OUT * 2)``
     - Dolly back, keeping the same aim.
   * - ``camera.rotate(deg, UP, about_point=ORIGIN)``
     - Turntable: swing around the scene, staying pointed at it.
   * - ``camera.look_at(point)``
     - Turn to face a point without moving.
   * - ``camera.orbit(deg, UP, about_point=p)``
     - Swings along a circle around ``p`` *without* changing its pointing direction.
   * - ``camera.move_to_make_mob_center_of_view(mob)``
     - Automatically reframes so the target Mob is centered.

The turntable shot is the classic way to show off a 3-D scene. Notice that we use
:meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.rotate` with
``about_point``:

.. algan:: CameraTurntable

    from algan import *

    with Off():
        Group([Cube(side_length=0.8, color=BLUE).move(RIGHT * 1.6 * i)
               for i in (-1, 0, 1)]).spawn()

    with Seq(run_time=4, rate_func=rate_funcs.identity):
        Scene.get_camera().rotate(360, UP, about_point=ORIGIN)

    Scene.save_video()

``rotate`` turns the camera's orientation along with its circular path, so it
stays pointed straight at the center throughout the turn.

.. tip::

    For continuous camera rotations, pass ``rate_func=rate_funcs.identity`` so
    the speed stays constant rather than easing in and out.

Tracking a Moving Target
========================

To make the camera continuously follow an object as it moves, attach a simple
updater (see :doc:`../new_user_tutorials/updaters`):

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

Field of View (FOV)
===================

``fov`` sets the vertical field of view in degrees. Small FOVs act like a
telephoto lens (flattening depth and perspective), while large FOVs give a
wide-angle view with exaggerated perspective:

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

Because :meth:`~algan.rendering.camera.Camera.set_fov` works by adjusting the
distance to the internal screen plane, it animates smoothly on the timeline like
any other property, making dramatic "dolly zoom" effects simple.

Algan also exposes the underlying perspective controls directly:
:meth:`~algan.rendering.camera.Camera.set_distance_to_screen` moves the focus point relative to the
screen plane, and the constructor's ``screen_distance`` / ``screen_scale`` set
them up front. ``fov`` is derived from these, so use one or the other, not both.

.. _camera-orthographic:

Near-Orthographic Projection
============================

If you are building technical diagrams, engineering cross-sections, or 2-D plots
where you need exact parallel lines without perspective distortion, use
:meth:`~algan.rendering.camera.Camera.set_near_orthographic`:

.. algan:: CameraOrthographic

    from algan import *

    with Off():
        Scene.get_camera().set_near_orthographic()
        cubes = Group([Cube(side_length=0.8, color=BLUE).move(IN * 1.6 * i + RIGHT * 0.9 * i)
                       for i in range(4)]).spawn()

    with Seq(run_time=3):
        cubes.rotate(360, UP, about_point=ORIGIN)

    Scene.save_video()

This pushes the camera far away while narrowing the lens, removing perspective
foreshortening so distant and near objects appear identical in scale.

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

Setting ``camera.set_near(0.5)`` is the standard way to stop foreground objects
from blocking the view when flying a camera deep into a scene.

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
* :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.fit_to_screen` -- scale and move to fill a screen
  rectangle.

Each of them resolves the camera *once*, when the call is recorded, so a later
camera move will not keep the Mob pinned there. For something that must stay in a
fixed screen position through a camera move (e.g. a caption, a legend) attach it to
the camera as a child, or drive it with an updater:

.. algan:: CameraChildCaption

    from algan import *

    with Off():
        Group([Cube(side_length=0.8, color=BLUE).move(RIGHT * 1.6 * i)
               for i in (-1, 0, 1)]).spawn()

        caption = Text("figure 1", font_size=32)
        Scene.get_camera().add_children([caption])
        caption.move_to_screen_position(0.15, 0.1)
        caption.spawn()

    with Seq(run_time=3, rate_func=rate_funcs.identity):
        Scene.get_camera().rotate(90, UP, about_point=ORIGIN)

    Scene.save_video()

Because child Mobs automatically inherit their parent's movement and rotation,
the caption stays perfectly pinned to the screen throughout the turn.

See Also
========

* :doc:`../new_user_tutorials/three_d_basics` -- the gentler introduction.
* :doc:`positioning_and_layout` -- the movement and orientation methods this page
  applies to the camera, in full.
* :doc:`../new_user_tutorials/updaters` -- the updater used above to track a
  moving subject.
* :doc:`lighting_and_shadows` -- lights, and the rig that goes with a camera move.
* :doc:`renderer_limitations` -- what the camera model does not do, including
  true orthographic projection and depth of field.
* :doc:`performance_and_quality` -- what actually makes a render expensive, and what
  to do about it.
