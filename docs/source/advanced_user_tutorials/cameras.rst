=======
Cameras
=======

Every Scene in Algan has one :class:`~algan.rendering.camera.Camera`. And
because the camera is itself a :class:`~algan.animatable_base.mob.Mob`, you move,
rotate, and animate it using the exact same methods you use for everything else.

.. code-block:: python

    from algan import *

    camera = Scene.get_camera()

By default, the Scene's camera sits at ``ORIGIN + OUT * 20`` aimed towards
``ORIGIN``, using a perspective projection with a vertical field of view of
approximately 22.62°. At the origin plane (``z = 0``), the framing is 8 world
units tall; its width follows the output aspect ratio (approximately 14.22
world units at 16:9, or 4.5 at 9:16).

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
   * - ``camera.rotate(deg, UP, about=ORIGIN)``
     - Turntable: swing around the scene, staying pointed at it.
   * - ``camera.look_at(point)``
     - Turn to face a point without moving.
   * - ``camera.orbit(deg, UP, about=p)``
     - Swings along a circle around ``p`` *without* changing its pointing direction.
   * - ``camera.center_on(mob)``
     - Automatically reframes so the target Mob is centered.
   * - ``camera.fly_to(position, look_at=target, via=waypoint, look_at_via=aim_waypoint)``
     - Move position and aim together, with a level horizon and optional curves.
   * - ``camera.visible_size_at(point)``
     - Query visible width and height in world units at the point's forward depth.

For a shot that changes both position and target, ``fly_to`` recomputes the
view direction at every frame. ``via`` is a world-space waypoint passed halfway
through the eased motion; omit it for a straight path. ``look_at_via`` does the
same for the target, so the aim can sweep along an arc. World ``UP`` keeps the
horizon level, with the starting right direction used at a vertical view; a
camera that starts tilted levels out gradually over the move.

.. code-block:: python

    with Seq(runtime=3):
        camera.fly_to((4, 2, 7), look_at=ORIGIN, via=(2, 3, 8))

    width, height = camera.visible_size_at(ORIGIN).flatten()

``visible_size_at`` uses the Scene's current resolution, so the same query
works for a narrow 9:16 frame. On the camera of a ``CameraView`` it uses that
view's capture resolution instead. It measures the plane parallel to the screen
through the point; an off-axis point uses forward depth rather than ray length.

The turntable shot is the classic way to show off a 3-D scene. Notice that we use
:meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.rotate` with
``about``:

.. algan:: CameraTurntable

    from algan import *

    with Off():
        Group([Cube(size=0.8, color=BLUE).move(RIGHT * 1.6 * i)
               for i in (-1, 0, 1)]).spawn()

    with Seq(runtime=4, easing=easings.identity):
        Scene.get_camera().rotate(360, UP, about=ORIGIN)

    Scene.save_video()

``rotate`` turns the camera's orientation along with its circular path, so it
stays pointed straight at the center throughout the turn.

.. tip::

    For continuous camera rotations, pass ``easing=easings.identity`` so
    the speed stays constant rather than easing in and out.

Tracking a Moving Target
========================

To make the camera continuously follow an object as it moves, attach a simple
updater (see :doc:`../new_user_tutorials/updaters`):

.. algan:: CameraTracking

    from algan import *

    with Off():
        ball = Sphere(radius=0.6, color=YELLOW).spawn()
        Group([Cube(size=0.5, color=BLUE).move(RIGHT * x + DOWN * 1.5)
               for x in (-3, 0, 3)]).spawn()

    camera = Scene.get_camera()
    camera.add_updater(lambda self, t: self.look_at(ball.location))

    with Seq(runtime=3):
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
        Group([Cube(size=0.8, color=BLUE).move(IN * 1.6 * i + RIGHT * 0.9 * i)
               for i in range(4)]).spawn()

    camera = Scene.get_camera()
    with Seq(runtime=3):
        camera.set_fov(20)
        camera.set_fov(90)

    Scene.save_video()

Because :meth:`~algan.rendering.camera.Camera.set_fov` works by adjusting the
distance to the internal screen plane, it animates smoothly on the timeline like
any other property, making dramatic "dolly zoom" effects simple.

Algan also exposes the underlying perspective controls directly:
:meth:`~algan.rendering.camera.Camera.set_distance_to_screen` moves the focus point relative to the
screen plane, and the constructor's ``screen_distance`` / ``screen_half_height`` set
them up front. ``fov`` is derived from these, so use one or the other, not both.

.. _camera-orthographic:

Near-Orthographic Projection
============================

If you are building technical diagrams, engineering cross-sections, or 2-D plots
where you want almost parallel lines with minimal perspective distortion, use
:meth:`~algan.rendering.camera.Camera.set_near_orthographic`:

.. algan:: CameraOrthographic

    from algan import *

    with Off():
        Scene.get_camera().set_near_orthographic()
        cubes = Group([Cube(size=0.8, color=BLUE).move(IN * 1.6 * i + RIGHT * 0.9 * i)
                       for i in range(4)]).spawn()

    with Seq(runtime=3):
        cubes.rotate(360, UP, about=ORIGIN)

    Scene.save_video()

This moves both the eye and its internal screen while narrowing the lens. The
visible frame on the plane parallel to the screen through ``ORIGIN`` stays the
same throughout the animation: the default 16:9 frame remains approximately
14.2 by 8 world units, so text and shapes on that plane keep their apparent size.
Other depths retain a small amount of perspective because this is a distant
perspective camera. ``distance`` sets the eye-to-screen distance in world units
and defaults to ``1e5``. If the origin plane is at or behind the eye, the current
screen plane is used as the framing reference instead.

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

.. _camera-depth-of-field:

Depth of Field
==============

The camera is a pinhole by default: everything is in focus. Opening its
``aperture`` turns it into a thin lens. The plane ``focus_distance`` in front
of the camera stays sharp, and anything nearer or farther blurs into a disk
that grows with its distance from that plane. Depth of field is rendered by the
path tracer, so it needs ``samples_per_pixel > 1``:

.. code-block:: python

    from algan import *

    SETTINGS.raytracing.set(samples_per_pixel=64)

    subject = Sphere(radius=0.8, color=BLUE).spawn()
    foreground = Cube(size=0.6, color=RED).move_to(LEFT * 1.5 + OUT * 8).spawn()
    background = Cube(size=2, color=GREEN).move_to(RIGHT * 3 + IN * 12).spawn()

    camera = Scene.get_camera()
    with Off():
        camera.aperture = 0.6        # lens diameter, world units
        camera.focus_at(subject)     # focus_distance = the sphere's depth

    Scene.wait(1)
    with Seq(runtime=2):
        camera.focus_at(foreground)  # rack focus to the red cube
    Scene.wait(1)

    Scene.save_video()

Both values are in world units:

* ``aperture`` is the lens **diameter**. A point at distance ``d`` along the
  camera's forward axis blurs into a disk of diameter
  ``aperture * |d - focus_distance| / d``, measured on the plane in focus: an
  object at infinity blurs by exactly the aperture's size seen at that plane.
  ``0`` (the default) is a pinhole.
* ``focus_distance`` is measured along the forward axis, so the region in focus
  is a plane parallel to the screen, not a sphere around the camera. It
  defaults to ``20``, the default camera's distance to ``ORIGIN``: opening the
  aperture of a camera you have not moved keeps the ``ORIGIN`` plane sharp.

Unlike the clip planes, both are **animated attributes**. Writing either after
the camera is spawned -- and the Scene's camera always is -- records a tween
over the current context's runtime, which is how a rack focus is made; wrap
setup in ``with Off():``, as above. :meth:`~algan.rendering.camera.Camera.focus_at`
takes a Mob or a point and animates ``focus_distance`` onto its depth,
re-measuring it on every frame of the pull so a moving subject is sharp when
the pull lands. Record that move first: animations replay in the order they
were written, so in ``with Sync(): subject.move(OUT * 3); camera.focus_at(subject)``
the pull follows the subject, while a move written after ``focus_at`` is not
seen by it. After the pull the distance stays put; a subject that keeps moving
drifts out of focus until you call ``focus_at`` again.

A few things to know:

* **The deterministic renderer has no lens.** A render with an open aperture
  and ``samples_per_pixel == 1`` raises
  :class:`~algan.errors.UnsupportedFeatureError` naming depth of field. To
  preview such a shot quickly as a pinhole, set
  ``SETTINGS.raytracing.set(unsupported_feature_policy="warn")`` for the
  preview and back to ``"error"`` for the final render.
* **Blur costs samples.** Every lens position is a random choice, so adaptive
  sampling never stops a depth-of-field pixel early: each one takes the full
  ``samples_per_pixel`` (and the denoiser smooths what remains). Wide
  apertures on bright, small, far-out-of-focus shapes need the most samples.
* **Flat 2-D content blurs too.** Text and shapes are ordinary geometry at a
  depth, so a caption far from the focus plane goes soft. Keep what must stay
  legible near ``focus_distance``, or composite it separately.
* **The background colour, image or callable stays sharp**: it is a
  screen-space backdrop, not an object in the scene. An environment map is
  scenery at infinity and blurs like any distant object.
* The **near-orthographic** mode backs the camera far away (``distance``, the
  eye-to-screen distance, is ``1e5`` by default) and moves ``focus_distance``
  out with it, so the plane in focus stays where it was. From that far, any
  world-sized aperture blurs by a tiny angle: depth of field is effectively
  invisible there. (Call ``focus_at`` again if you move the focus afterwards.)
* The auxiliary passes of :ref:`saving-render-passes` are always pinhole, so
  a depth pass stays sharp for compositing even when the colour pass is
  defocused.

Screen Coordinates
==================

The camera is also what converts between world space and what the viewer sees, so
these Mob methods all resolve against it:

* :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to_screen_position` and
  :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.move_center_to_screen_position` -- place a Mob at fractional
  screen coordinates.
* :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to_screen_edge` and
  :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to_screen_corner` -- rest against a
  screen border.
* :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.fit_to_screen` -- scale and move to fill a screen
  rectangle.

Their directions are the camera's, not the world's: ``move_to_screen_edge(RIGHT)``
follows ``camera.right``, so it means the right of the frame however the camera is
turned, and the Mob slides in the plane parallel to the screen without changing its
distance to the camera. The third axis points out of the screen towards the viewer
(``OUT`` is the camera's ``-forward``), so ``move_to_screen_edge(RIGHT + OUT)`` casts
along the diagonal of the two until that ray leaves the frustum.

Each of them resolves the camera *once*, when the call is recorded, so a later
camera move will not keep the Mob pinned there. For something that must stay in a
fixed screen position through a camera move (e.g. a caption, a legend) attach it to
the camera as a child, or drive it with an updater:

.. algan:: CameraChildCaption

    from algan import *

    with Off():
        Group([Cube(size=0.8, color=BLUE).move(RIGHT * 1.6 * i)
               for i in (-1, 0, 1)]).spawn()

        caption = Text("figure 1", font_size=32)
        Scene.get_camera().add_children([caption])
        caption.move_to_screen_position(0.15, 0.1)
        caption.spawn()

    with Seq(runtime=3, easing=easings.identity):
        Scene.get_camera().rotate(90, UP, about=ORIGIN)

    Scene.save_video()

Because child Mobs automatically inherit their parent's movement and rotation,
the caption stays perfectly pinned to the screen throughout the turn.

Live Insets and Multiple Views
==============================

:class:`~algan.mobs.camera_view.CameraView` displays another camera's live view
inside the same Scene. Its camera and its display are independent: animate
``view.camera`` to pan or zoom the source region, and animate ``view`` to position,
scale, rotate or fade the inset. ``view.focus_on(subject)`` fits a subject using
the capture's aspect ratio; it measures the subject when called, so move the
camera with the subject to keep tracking it.

.. algan:: LiveCameraInset

    from algan import *

    with Off():
        detail = Group([
            Circle(radius=0.15, color=YELLOW).move(LEFT * 0.3),
            Square(size=0.3, color=BLUE).move(RIGHT * 0.3),
        ]).spawn()
        inset = CameraView(resolution=(480, 320), height=2.5)
        inset.focus_on(detail, buffer_portion=0.4)
        inset.move_to(RIGHT * 3 + UP * 1.5 + OUT * 0.2)
        border = SurroundingRectangle(inset, buffer=0, filled=False, stroke_color=WHITE)
        inset.add_children(border)
        inset.spawn()

    with Sync(runtime=2):
        detail.move(LEFT * 2)
        inset.camera.move(LEFT * 2)
        inset.move(DOWN)
    Scene.save_video()

Construct another ``CameraView`` for another simultaneous angle. Each omitted
``camera`` creates an independent camera matching the main camera's current pose
and lens. Pass ``camera=existing_camera`` to share a source camera between displays.
For a side view, for example, rotate ``view.camera`` around the subject with
``view.camera.rotate(90, UP, about=subject.location)``. ``set_fov``, near/far
clipping, lights, environment maps and the normal renderer remain available.

The display is an ordinary unlit textured surface in world space, so it supports
perspective, occlusion and opacity. Place it in front of nearby objects with
``OUT`` when using it as an inset. Attach it to the main camera as described above
to keep its screen position during main-camera motion. The capture resolution
defaults to ``(640, 360)`` pixels; increasing the display's world size does not
increase its capture resolution.

All camera-view displays and their children (including borders and captions)
are excluded from every secondary capture to prevent recursive images. Pass
``exclude=[label, source_box]`` to omit additional Mobs and their descendants from
one view. Exclusion does not hide them from the main camera. Inherited scene
backgrounds, including render-time background overrides, appear in the captures.
Exposure, tone mapping and post-processing apply once to the composed result.

Views are live in videos, ``save_frame`` and the interactive viewer, including
seeking backwards. Each additional view requires another render of the scene;
captures are processed in bounded windows and use the same timeline timestamps
as the main view. As with repeated renders, updaters should compute their state
from timeline time instead of relying on how many times they are called.

``mn.ImageMobjectFromCamera(algan_camera)`` uses this live pipeline too. Cameras
from other libraries retain its ``pixel_array`` snapshot behavior. Algan's native
``CameraView`` is the authoring interface; Manim's Scene subclasses are not needed.

See Also
========

* :doc:`../new_user_tutorials/three_d_basics` -- the gentler introduction.
* :doc:`positioning_and_layout` -- the movement and orientation methods this page
  applies to the camera, in full.
* :doc:`../new_user_tutorials/updaters` -- the updater used above to track a
  moving subject.
* :doc:`lighting_and_shadows` -- lights, and the rig that goes with a camera move.
* :doc:`renderer_limitations` -- what the camera model does not do, including
  true orthographic projection and motion blur, and which renderer draws depth
  of field.
* :doc:`performance_and_quality` -- what actually makes a render expensive, and what
  to do about it.
