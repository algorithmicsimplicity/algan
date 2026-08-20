=====================
Your First 3-D Scene
=====================

Everything so far has been flat, but Algan's scene is genuinely
three-dimensional and its renderer is a GPU ray tracer -- 3-D objects are lit,
they occlude each other correctly, and they can reflect and refract.

Nothing new is required to use it. 3-D Mobs spawn, move and rotate exactly like
2-D ones; the differences are that they respond to light and that you now have a
camera worth moving.

Depth
=====

The third axis is ``OUT`` (towards the viewer) and ``IN`` (away from it). The
default camera sits at ``OUT * 7`` looking at the ``ORIGIN``, so moving something
``IN`` pushes it away and makes it look smaller:

.. algan:: ThreeDDepth

    from algan import *

    with Off():
        cubes = Group([Cube(side_length=0.8, color=BLUE).move(IN * 1.6 * i + RIGHT * 0.9 * i)
                       for i in range(4)]).spawn()

    with Seq(run_time=3):
        cubes.rotate(360, UP, about_point=ORIGIN)

    Scene.save_video()

The four cubes are identical; the further ones look smaller because the default
camera is a perspective camera. For technical diagrams where equal things must
*look* equal, switch it to a near-orthographic camera, which flattens
foreshortening to almost nothing:

.. code-block:: python

    with Off():
        Scene.get_camera().set_near_orthographic()

.. important::

    The projection mode and the clip planes (:meth:`~.Camera.set_near`,
    :meth:`~.Camera.set_far`) are camera *configuration*, not animated
    attributes. Set them once, before spawning your Mobs; changing them part way
    through a script does not give you a reliable mid-video switch. To show a
    before and after, render two videos.

Lighting
========

Every scene starts with one white :class:`~.PointLight` above and to the right of
the camera. That is what gives 3-D shapes their shading, and it is why a
:class:`~.Sphere` reads as a ball rather than a flat disc.

Lights are Mobs, so you animate them like anything else:

.. code-block:: python

    light = Scene.get_light_sources()[0]
    with Seq(run_time=4):
        light.orbit(360, OUT, about_point=ORIGIN)

.. note::

    Flat 2-D shapes and text are drawn in their own colour and are **not** lit.
    If a :class:`~.Square` and a :class:`~.Cube` look different, that is why.

Moving the Camera
=================

:meth:`~algan.scene.Scene.get_camera` returns the active :class:`~algan.rendering.camera.Camera`, which is also a
Mob. The move you will use most is a turntable -- rotating the camera about the
origin so the scene appears to spin:

.. algan:: ThreeDCamera

    from algan import *

    ball = Sphere(radius=1, color=BLUE).spawn()

    camera = Scene.get_camera()
    with Sync(run_time=3, rate_func=rate_funcs.identity):
        camera.rotate(180, UP, about_point=ORIGIN)
        ball.move(UP * 0.8)

    Scene.save_video()

Note the two details that make this look right:

* :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.rotate` with ``about_point=ORIGIN``, not :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.orbit`.
  ``rotate`` carries the camera's orientation around with it, so it keeps facing
  the scene; ``orbit`` would move it while leaving it pointing the same way, and
  the scene would slide out of frame.
* ``rate_func=rate_funcs.identity``, so the camera turns at a constant speed
  instead of easing in and out. Camera moves almost always want this.

To aim the camera somewhere specific, :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.look_at` turns it to face a
point, and :meth:`~.Camera.move_to_make_mob_center_of_view` frames a given Mob.

Casting Shadows
===============

Ray-traced shadows are off by default because they cost render time. Turn them on
with one setting:

.. algan:: ThreeDShadows

    from algan import *

    SETTINGS.raytracing.set(shadows=True)

    with Off():
        Scene.clear_light_sources()
        DirectionalLight(location=UP * 8 + RIGHT * 4 + OUT * 4, target=ORIGIN,
                         color=WHITE, intensity=3).spawn()
        AmbientLight(color=WHITE, intensity=0.3).spawn()

        Sphere(radius=0.8, color=BLUE).move(UP * 0.7).spawn()
        Cube(side_length=4, color=GREY).move(DOWN * 2.6).spawn()

    Scene.wait(2)

    Scene.save_video()

Three things are worth copying from that example:

* The scene setup is wrapped in ``with Off():``. Lights are Mobs, so spawning
  them costs a second of timeline each -- without ``Off()`` the video would open
  with two seconds of darkness while the lights fade in.
* :meth:`~algan.scene.Scene.clear_light_sources` drops the default light first, so the
  lighting is entirely yours. A :class:`~.DirectionalLight` (parallel rays, like
  the sun) plus a dim :class:`~.AmbientLight` fill is a good default rig -- the
  ambient light stops the unlit side going pure black.
* The shadow is hard-edged, because the light has no size. Giving the sun an
  angular size softens the edge -- see
  :doc:`../advanced_user_tutorials/lighting_and_shadows` for that and the rest
  of the lighting model.

Curved Surfaces
===============

:class:`~.Surface` turns a function of two parameters into a 3-D surface. Both
parameters run over ``[0, 1]``, and the function must handle batched tensors:

.. algan:: ThreeDSurface

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

``checkered_color`` tints alternating cells, which makes the shape of a surface
much easier to read than a single flat colour. Algan tessellates the surface to
whatever resolution the current render needs, so it stays smooth as the camera
moves in.

Materials
=========

By default 3-D shapes get a simple diffuse shading. For metal, plastic, glass and
so on, apply a Three.js-style material *before* spawning:

.. algan:: ThreeDMaterials

    from algan import *

    metal = Sphere(radius=0.8, color=RED).move(LEFT * 1.8).set_material(
        MeshStandardMaterial(metalness=1.0, roughness=0.2))
    plastic = Sphere(radius=0.8, color=BLUE).move(RIGHT * 1.8).set_material(
        MeshStandardMaterial(metalness=0.0, roughness=0.6))
    with Sync():
        metal.spawn()
        plastic.spawn()

    with Seq(run_time=3):
        metal.roughness = 0.9

    Scene.save_video()

A material's properties land on the Mob as animatable attributes, so
``metal.roughness = 0.9`` animates like any other change.

.. important::

    :meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_material`,
    :meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_shader` and
    :meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_fragment_shader`
    must all be called **before**
    :meth:`~algan.animatable_base.animatable.Animatable.spawn`.

Keeping 3-D Renders Fast
========================

3-D costs more than 2-D, and a few habits keep the edit-preview loop quick:

* Leave the default ``LD`` quality on while you work and pass ``HD`` only for
  the final render: ``Scene.save_video("final", HD)``.
* Leave ``shadows`` off until you need them.
* Keep ``SETTINGS.raytracing.samples_per_pixel`` at ``1`` (the default). That
  selects the fast deterministic renderer; higher values switch to Monte Carlo
  path tracing, which is much slower.
* Watch how close the camera gets, not how far it travels. A :class:`~.Surface`
  is diced more finely the more of the frame it fills, so a close-up is what
  makes a render slow. If one runs out of memory, raising ``render_tolerance``
  on the surfaces nearest the camera usually fixes it -- at ``HD`` and above,
  raise ``render_tolerance_pixels`` alongside it, since that is the bound in
  force once the frame is large.

See :doc:`../advanced_user_tutorials/performance_and_quality` for the details.

Where to next
-------------

* :doc:`../advanced_user_tutorials/cameras` -- field of view, clipping,
  orthographic projection, camera animation.
* :doc:`../advanced_user_tutorials/lighting_and_shadows` -- every light type,
  soft shadows, environment maps.
* :doc:`../advanced_user_tutorials/shaders_and_materials` -- the full material
  catalogue.
* :doc:`../advanced_user_tutorials/reflections_and_glass` -- mirrors and
  refraction.
* :doc:`../advanced_user_tutorials/three_d_models` -- importing ``.glb`` and
  ``.fbx`` models.
