====================
Your First 3-D Scene
====================

Everything so far has been flat 2-D shapes, but Algan's scene is genuinely
three-dimensional and its renderer is a full-featured ray tracer:
3-D objects are lit, they occlude each other correctly, and they can reflect and refract,
if their material allows so.

Nothing new is required to use 3-D. 3-D Mobs spawn, move and rotate exactly like
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
camera is a perspective camera.

Moving the Camera
=================

:meth:`~algan.scene.Scene.get_camera` returns the active :class:`~algan.rendering.camera.Camera`,
which is also a Mob. That means everything you've learned so far also applies to the camera.
So, we can animate a camera movement as follows:

.. algan:: ThreeDCamera

    from algan import *

    ball = Sphere(radius=1, color=BLUE).spawn()

    camera = Scene.get_camera()
    with Sync(run_time=3, rate_func=rate_funcs.identity):
        camera.rotate(180, UP, about_point=ORIGIN)
        ball.move(UP * 0.8)

    Scene.save_video()

To aim the camera somewhere specific, :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.look_at`
turns it to face a point, and :meth:`~.Camera.move_to_make_mob_center_of_view` frames a given Mob.

.. seealso::

    :doc:`../advanced_user_tutorials/cameras` -- field of view, clipping planes,
    near-orthographic projection, and how to follow a moving subject.

Lighting
========

By default, every scene starts with one white :class:`~.PointLight` above and to the right of
the camera. That is what gives 3-D shapes their shading, and it is why a :class:`~.Sphere`
reads as a ball rather than a flat disc.
Lights are Mobs, so you animate them like anything else.
:meth:`~algan.scene.Scene.get_light_sources` returns a list of all mobs in the scene
which act as light sources.

.. algan:: ThreeDLightOrbit

    from algan import *

    ball = Sphere(radius=1.2, color=BLUE).spawn()

    light = Scene.get_light_sources()[0]
    with Seq(run_time=4, rate_func=rate_funcs.identity):
        light.orbit(360, OUT, about_point=ORIGIN)

    Scene.save_video()

.. note::

    Flat 2-D shapes and text are drawn in their own color and are **not** lit.
    This is why a :class:`~.Square` and a :class:`~.Cube` look different.

Casting Shadows
===============

Ray-traced shadows are off by default because they make rendering noticeably slower.
You can turn them on with one setting:

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

Two things worth noticing:

* :meth:`~algan.scene.Scene.clear_light_sources` drops the default light first, so the
  lighting is entirely yours.
* The shadow is hard-edged, because the light has no size. Giving the sun an
  angular size softens the edge.

.. seealso::

    :doc:`../advanced_user_tutorials/lighting_and_shadows` -- every light type,
    soft shadows, environment maps, and how to build a three-point rig.

Curved Surfaces
===============

:class:`~.Surface` allows you to define a manifold surface of any shape,
by providing a function which maps two intrinsic coordinates ``u`` and ``v``
(each in ``[0, 1]``) to a coordinate in 3-D space. The function you provide must
handle batched tensors.

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

.. seealso::

    :doc:`../advanced_user_tutorials/images_and_textures` -- painting an image
    or a per-texel material property across a surface, and wrapping a map onto a
    globe.

Materials
=========

By default 3-D shapes get a simple diffuse shading. For metal, plastic, glass and
so on, apply a material *before* spawning:

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

Once a mob has used
:meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_material`,
that material's properties become animatable attributes of the mob, so
``metal.roughness = 0.9`` animates like any other change.

.. important::

    :meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_material`,
    :meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_shader` and
    :meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_fragment_shader`
    must all be called **before**
    :meth:`~algan.animatable_base.animatable.Animatable.spawn`.

.. seealso::

    * :doc:`../advanced_user_tutorials/shaders_and_materials` -- the full
      material catalogue and custom shaders.
    * :doc:`../advanced_user_tutorials/reflections_and_glass` -- what
      ``metalness``, ``roughness`` and ``transmission`` do to rays.

More 3-D
========

3-D rendering is too broad a topic to cover in one tutorial, so if you want to
learn more, these advanced tutorials pick up where this one stops:

* :doc:`../advanced_user_tutorials/cameras` -- field of view, clipping,
  orthographic projection, camera animation.
* :doc:`../advanced_user_tutorials/lighting_and_shadows` -- every light type,
  soft shadows, environment maps.
* :doc:`../advanced_user_tutorials/shaders_and_materials` -- the full material
  catalogue, custom shaders.
* :doc:`../advanced_user_tutorials/reflections_and_glass` -- mirrors and
  refraction.
* :doc:`../advanced_user_tutorials/three_d_models` -- importing ``.glb`` and
  ``.fbx`` model assets.
* :doc:`../advanced_user_tutorials/renderer_limitations` -- what the renderer
  does not do, and where its approximations show.

