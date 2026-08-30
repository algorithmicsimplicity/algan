====================
Lighting and Shadows
====================

Algan's ray-traced renderer supports a full set of light sources modelled on
`Three.js <https://threejs.org/>`_'s lights, plus ray-traced shadows and
environment maps (image-based lighting). This tutorial covers all of them.

.. note::

   Lights, shadows and environment maps affect **3-D objects** (those built on
   :class:`~.Surface`, such as :class:`~.Sphere`, :class:`~.Cylinder` and
   :class:`~.Cone`, and imported 3-D models). Flat 2-D shapes and text are drawn
   with their own color and are not lit.

The Default Light
=================

Every scene starts with a single white :class:`~.PointLight` positioned above and
to the right of the camera. That is what gives 3-D shapes their shading.

Because lights are :class:`~.Mob` s, you spawn, move, animate, and despawn them
just like any other shape. **Spawning a light registers it with the Scene
automatically**:

.. algan:: LightingRegistering

    from algan import *

    ball = Sphere(radius=1.2, color=WHITE).spawn()

    # The scene's existing lights (index 0 is the default point light)
    lights = Scene.get_light_sources()

    with Off():
        # Spawning a new light automatically adds it to the scene
        PointLight(location=LEFT * 5 + OUT * 3, color=BLUE, intensity=2).spawn()

        # Remove the default light
        Scene.remove_light(lights[0])

    ball.rotate(180, UP)

    Scene.save_video()

To start with a blank slate call :meth:`~algan.scene.Scene.clear_lights`,
which removes all existing lights.

.. important::

   Wrap scene setup in ``with Off():``. Lights are Mobs, so each ``spawn()``
   records a one-second fade by default, and building a three-light rig outside
   ``Off()`` would make your video open with three seconds of darkness.

Because lights are standard Mobs, you can animate them directly on the timeline:

.. algan:: LightingAnimatedLight

    from algan import *

    ball = Sphere(radius=1.2, color=WHITE).spawn()

    light = Scene.get_light_sources()[0]
    with Seq(run_time=4, rate_func=rate_funcs.identity):
        light.orbit(360, OUT, about=ORIGIN)
        light.color = BLUE

    Scene.save_video()

Light Types
===========

Every light takes an ``intensity`` multiplier, and every light's ``location``,
``color`` and ``intensity`` are animatable. All of them are importable from
``algan`` directly.

Point Light
-----------

:class:`~.PointLight` emits light in all directions from a single point. By
default, it has no distance falloff (keeping scene lighting clean and even), but
you can opt into physically-correct attenuation with ``decay`` and a finite
``distance`` range:

.. algan:: LightingPointLight

    from algan import *

    with Off():
        Scene.clear_lights()
        # Inverse-square falloff (decay=2), fading out by 20 units
        PointLight(location=UP * 4, color=WHITE, intensity=30, decay=2,
                   distance=20).spawn()

        Group([Sphere(radius=0.55, color=WHITE).move(RIGHT * x)
               for x in (-1.8, 0, 1.8)]).spawn()

    Scene.wait(2)

    Scene.save_video()

Note that turning on ``decay`` usually means raising ``intensity`` a long way,
because the light now falls off with distance.

Directional Light
-----------------

:class:`~.DirectionalLight` models a distant source like the sun: all its rays are
parallel, pointing from the light toward its ``target`` (the origin by default).
Distance does not matter, only direction.

.. algan:: LightingDirectionalLight

    from algan import *

    with Off():
        Scene.clear_lights()
        DirectionalLight(location=UP * 10 + RIGHT * 6, target=ORIGIN,
                         color=WHITE, intensity=2).spawn()

        Group([Sphere(radius=0.55, color=WHITE).move(RIGHT * x)
               for x in (-1.8, 0, 1.8)]).spawn()

    Scene.wait(2)

    Scene.save_video()

Ambient Light
-------------

:class:`~.AmbientLight` adds a flat, direction-less term to every surface. Use it
to lift shadows and fill in the dark side of objects so they never go fully black.
Almost every rig wants a little of it.

.. algan:: LightingAmbientLight

    from algan import *

    with Off():
        Scene.clear_lights()
        DirectionalLight(location=UP * 8 + RIGHT * 6 + OUT * 4, target=ORIGIN,
                         color=WHITE, intensity=2).spawn()
        AmbientLight(color=WHITE, intensity=0.4).spawn()

        ball = Sphere(radius=1.2, color=BLUE).spawn()

    ball.rotate(180, UP)

    Scene.save_video()

The ambient term is what keeps the side of the sphere facing away from the key
light off pure black.

Animating Intensity
-------------------

A light's ``intensity`` is an animatable attribute like its ``color``: writing
it after spawn records the change on the timeline, so a light can brighten or
dim over a shot.

.. algan:: LightingAnimatedIntensity

    from algan import *

    with Off():
        Scene.clear_lights()
        PointLight(location=UP * 4 + OUT * 2, color=WHITE).spawn()

        Group([Sphere(radius=0.55, color=WHITE).move(RIGHT * x)
               for x in (-1.8, 0, 1.8)]).spawn()

    light = Scene.get_light_sources()[0]
    with Seq(run_time=3):
        light.intensity = 5

    Scene.save_video()

.. note::

    A light's *shape* parameters (``decay``, ``distance``, cone angles and
    emitter sizes), are still plain per-light constants rather than animatable
    attributes: they are read when a frame batch is prepared rather than
    recorded on the timeline. To show two of those settings, render two videos.

Hemisphere Light
----------------

:class:`~.HemisphereLight` is a soft outdoor fill: surfaces facing ``up`` receive
the light's ``color`` (the "sky"), surfaces facing down receive its
``ground_color``, and side-facing surfaces blend between the two.

.. algan:: LightingHemisphereLight

    from algan import *

    with Off():
        Scene.clear_lights()
        HemisphereLight(color=BLUE, ground_color=(0.4, 0.3, 0.1),
                        intensity=0.8).spawn()

        Group([Sphere(radius=0.55, color=WHITE).move(RIGHT * x)
               for x in (-1.8, 0, 1.8)]).spawn()

    Scene.wait(2)

    Scene.save_video()

Spot Light
----------

:class:`~.SpotLight` is a cone of light aimed at a ``target``. ``cone_angle`` sets
the cone's half-angle in degrees and ``penumbra`` (0-1) softens its edge. Like a point
light, it supports ``decay`` and ``distance``.

.. algan:: LightingSpotHemisphere

    from algan import *

    with Off():
        Scene.clear_lights()
        SpotLight(location=UP * 5 + OUT * 2, target=ORIGIN, color=WHITE,
                  intensity=40, cone_angle=28, penumbra=0.6, decay=2).spawn()
        HemisphereLight(color=BLUE, ground_color=(0.3, 0.2, 0.1), intensity=0.5).spawn()

        Group([Sphere(radius=0.55, color=WHITE).move(RIGHT * x)
               for x in (-1.8, 0, 1.8)]).spawn()
        Cube(side_length=5, color=GREY).move(DOWN * 3.1).spawn()

    Scene.wait(2)

    Scene.save_video()

Rect-Area Light
---------------

:class:`~.RectAreaLight` is a glowing rectangle (a softbox). It produces smooth,
soft lighting and, with shadows enabled, soft-edged shadows, because Algan samples
it at a grid of ``samples`` emitter points. More samples give a smoother result at
a proportional cost.

.. algan:: LightingRectAreaLight

    from algan import *

    SETTINGS.raytracing.set(shadows=True)

    with Off():
        Scene.clear_lights()
        RectAreaLight(location=UP * 5, target=ORIGIN, width=4, height=4,
                      samples=16, color=WHITE, intensity=1.2).spawn()
        AmbientLight(color=WHITE, intensity=0.2).spawn()

        Sphere(radius=0.8, color=BLUE).move(UP * 0.7).spawn()
        Cube(side_length=4, color=GREY).move(DOWN * 2.6).spawn()

    Scene.wait(2)

    Scene.save_video()

Ray-Traced Shadows
==================

Shadows are **off by default**, because they cost render time. Turn them on with
one setting before rendering::

    SETTINGS.raytracing.set(shadows=True)

With shadows on, each lit surface point fires shadow rays toward every light; a
light blocked by another object does not contribute to that point.

Soft Shadows
------------

By default shadows are hard-edged. To get a soft penumbra, give the light a
non-zero emitter size:

- **Point** and **spot** lights take a ``shadow_radius`` -- the world-space radius
  of the emitting disk.
- **Directional** lights take a ``shadow_angle`` -- the angular size of the source
  in degrees, like the sun's ~0.5°.
- **Rect-area** lights are soft automatically; their penumbra smoothness is set by
  ``samples``.

.. algan:: LightingSoftShadow

    from algan import *

    SETTINGS.raytracing.set(shadows=True)

    with Off():
        Scene.clear_lights()
        DirectionalLight(location=UP * 8 + RIGHT * 4 + OUT * 4, target=ORIGIN,
                         color=WHITE, intensity=3, shadow_angle=3).spawn()
        AmbientLight(color=WHITE, intensity=0.3).spawn()

        Sphere(radius=0.8, color=BLUE).move(UP * 0.7).spawn()
        Cube(side_length=4, color=GREY).move(DOWN * 2.6).spawn()

    Scene.wait(2)

    Scene.save_video()

Algan traces a fixed fan of shadow rays across the emitter to build the penumbra.
The number of rays comes from the environment variable
``ALGAN_SOFT_SHADOW_SAMPLES`` (default 8), which is baked into the kernels and so
must be set **before** ``import algan``. Raise it for smoother penumbras at a
proportional cost.

.. note::

   Deterministic shadow rays *do* respect transparency: the light is multiplied
   through each occluder's opacity, so stacked translucent surfaces compound and a
   fully opaque one blocks. Ambient and emissive terms are unaffected.

   What neither renderer does is refractive *shadow* transport: a shadow ray
   travels a straight line through glass under both, so there are no caustics
   (see :doc:`renderer_limitations`). The path tracer honours every light type
   on this page -- area lights and emissive surfaces as true sampled emitters,
   the rest exactly as the deterministic stages define them -- at the cost of
   a large increase in render time; see :ref:`renderer-capabilities` for what
   it gives up.

.. admonition:: How many lights can cast shadows?
   :class: seealso

   Shadow-casting lights are collected into a fixed-size per-pixel list whose
   length is a compile-time constant (default 16, enough for a key/fill/rim rig
   plus a 4×4-sample area light). Lights beyond that are still *lit*, just not
   shadowed, and each sample of a :class:`~.RectAreaLight` counts toward the limit,
   so an under-capped area light simply gets a shallower shadow. If you need denser
   area-light penumbras or a larger rig, set ``ALGAN_MAX_SHADOW_LIGHTS`` before the
   first render (more GPU registers, slightly lower shadow-kernel occupancy).

Environment Maps
================

An environment map wraps the scene in a 360° image. It acts as a **skybox**
(visible in the background and in reflections and refractions) and, optionally, as
**image-based lighting** (the whole scene lit by the colors of the map).

Pass an equirectangular image (a longitude × latitude panorama, sky at the top) to
:meth:`Scene.set_environment_map <.Scene.set_environment_map>`, also available as
the top-level ``set_environment_map``:

.. algan:: LightingEnvironmentMap

    from algan import *

    set_environment_map("world_map.png", intensity=1.0, ambient=True)

    # A mirror sphere reflects the environment; other objects are lit by it.
    mirror = Sphere().move(LEFT * 1.5).set_material(
        MeshStandardMaterial(metalness=1.0, roughness=0.05))
    mirror.spawn()

    Sphere().move(RIGHT * 1.5).spawn()

    Scene.save_video()

Any equirectangular image works. A real studio panorama gives a much better result.

- ``intensity`` scales the map's brightness.
- ``ambient=True`` (the default) also lights surfaces from the map. Set it to
  ``False`` to use the map only as a backdrop and in reflections, without it
  contributing diffuse light.
- Pass ``None`` to remove a previously-set environment map.

You can also pass a ``[height, width, 3]`` tensor or NumPy array instead of a file
path. Image paths resolve against the working directory and then your script's
directory.

An environment map is the single biggest improvement you can make to a metal or
glass object: a mirror with nothing to reflect renders black. See

:doc:`reflections_and_glass`.

Building a Rig
==============

Real lighting is rarely one light. The standard three-point setup is a bright key
with the shadow, a dimmer fill opposite it to keep the shadow side readable, and a
rim from behind to separate the subject from the background, plus a little
ambient:

.. algan:: LightingThreePointRig

    from algan import *

    SETTINGS.raytracing.set(shadows=True)

    with Off():
        Scene.clear_lights()             # drop the default light

        # Key light: bright, from above and to one side, with a soft shadow.
        SpotLight(location=UP * 6 + RIGHT * 4 + OUT * 4, target=ORIGIN,
                  color=WHITE, intensity=60, cone_angle=30, penumbra=0.5,
                  decay=2, shadow_radius=0.3).spawn()

        # Fill: dimmer, from the opposite side, no shadow.
        PointLight(location=LEFT * 6 + OUT * 2, color=WHITE, intensity=4).spawn()

        # Rim: from behind, slightly cool.
        DirectionalLight(location=IN * 8 + UP * 4, target=ORIGIN,
                         color=Color((0.6, 0.7, 1.0))).spawn()

        # Ambient: stops the shadow side going pure black.
        AmbientLight(color=WHITE, intensity=0.25).spawn()

        Sphere().spawn()
        Cube(side_length=6, color=GREY).move(DOWN * 4).spawn()   # Ground floor

    Scene.wait(2)

    Scene.save_video()

See Also
========

- :doc:`../new_user_tutorials/three_d_basics` -- the gentler introduction to
  lights and 3-D shapes.
- :doc:`renderer_limitations` -- which objects and materials are lit and
  shadowed at all, and where the shadow approximations show.
- :doc:`cameras` -- moving the camera through a lit scene.
- :doc:`shaders_and_materials` -- how materials respond to these lights.
- :doc:`reflections_and_glass` -- mirrors, metals and refraction.
- :doc:`images_and_textures` -- normal maps, which change how a surface
  responds to these lights per texel.
- :doc:`performance_and_quality` -- what shadows and extra lights cost.
