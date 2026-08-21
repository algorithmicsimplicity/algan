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
   with their own colour and are not lit.

The Default Light
=================

Every scene starts with a single white :class:`~.PointLight` positioned above and
to the right of the camera. That is what gives 3-D shapes their shading.

Lights are :class:`~.Mob` s, so a light is spawned, animated and despawned like
anything else -- and **spawning one registers it with the Scene automatically**:

.. code-block:: python

    from algan import *

    # The scene's existing lights (a live list; index 0 is the default one).
    lights = Scene.get_light_sources()

    # Add another light: spawning is all it takes.
    PointLight(location=LEFT * 5 + OUT * 3, color=BLUE).spawn()

    # Take control of the lighting completely.
    Scene.clear_light_sources()

    # Or remove one particular light.
    Scene.remove_light_source(lights[0])

.. important::

   Wrap scene setup in ``with Off():``. Lights are Mobs, so each ``spawn()``
   records a one-second fade by default -- build a three-light rig outside
   ``Off()`` and your video opens with three seconds of darkness.

Because they are Mobs, lights animate:

.. code-block:: python

    light = Scene.get_light_sources()[0]
    with Seq(run_time=4, rate_func=rate_funcs.identity):
        light.orbit(360, OUT, about_point=ORIGIN)
        light.color = BLUE

Light Types
===========

Every light takes an ``intensity`` multiplier, and every light's ``location`` and
``color`` are animatable. All of them are importable from ``algan`` directly.

Point Light
-----------

:class:`~.PointLight` emits in all directions from a single point. By default it
has no distance falloff -- an Algan convention that keeps scenes evenly lit -- but
you can opt into physically-correct attenuation with ``decay`` and a finite
``distance`` range:

.. code-block:: python

    # Physically-attenuated point light (inverse-square falloff, fades out by 20 units).
    PointLight(location=UP * 4, color=WHITE, intensity=30, decay=2, distance=20).spawn()

Note that turning on ``decay`` usually means raising ``intensity`` a long way,
because the light now falls off with distance.

Directional Light
-----------------

:class:`~.DirectionalLight` models a distant source like the sun: all its rays are
parallel, pointing from the light toward its ``target`` (the origin by default).
Distance does not matter, only direction.

.. code-block:: python

    DirectionalLight(location=UP * 10 + RIGHT * 6, target=ORIGIN, color=WHITE).spawn()

Ambient Light
-------------

:class:`~.AmbientLight` adds a flat, direction-less term to every surface. Use it
to lift shadows and fill in the dark side of objects so they never go fully black.
Almost every rig wants a little of it.

.. code-block:: python

    AmbientLight(color=WHITE, intensity=0.4).spawn()

Hemisphere Light
----------------

:class:`~.HemisphereLight` is a soft outdoor fill: surfaces facing ``up`` receive
the light's ``color`` (the "sky"), surfaces facing down receive its
``ground_color``, and side-facing surfaces blend between the two.

.. code-block:: python

    HemisphereLight(color=BLUE, ground_color=(0.4, 0.3, 0.1), intensity=0.8).spawn()

Spot Light
----------

:class:`~.SpotLight` is a cone of light aimed at a ``target``. ``angle`` sets the
cone's half-angle in degrees and ``penumbra`` (0-1) softens its edge. Like a point
light, it supports ``decay`` and ``distance``.

.. algan:: LightingSpotHemisphere

    from algan import *

    with Off():
        Scene.clear_light_sources()
        SpotLight(location=UP * 5 + OUT * 2, target=ORIGIN, color=WHITE,
                  intensity=40, angle=28, penumbra=0.6, decay=2).spawn()
        HemisphereLight(color=BLUE, ground_color=(0.3, 0.2, 0.1), intensity=0.5).spawn()

        Group([Sphere(radius=0.55, color=WHITE).move(RIGHT * x)
               for x in (-1.8, 0, 1.8)]).spawn()
        Cube(side_length=5, color=GREY).move(DOWN * 3.1).spawn()

    Scene.wait(2)

    Scene.save_video()

Rect-Area Light
---------------

:class:`~.RectAreaLight` is a glowing rectangle -- a softbox. It produces smooth,
soft lighting and, with shadows enabled, soft-edged shadows, because Algan samples
it at a grid of ``samples`` emitter points. More samples give a smoother result at
a proportional cost.

.. code-block:: python

    RectAreaLight(location=UP * 5, target=ORIGIN, width=4, height=4,
                  samples=16, color=WHITE, intensity=1.2).spawn()

Ray-Traced Shadows
==================

Shadows are **off by default**, because they cost render time. Turn them on with
one setting before rendering:

.. code-block:: python

    from algan import *

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
        Scene.clear_light_sources()
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

   What the deterministic renderer cannot do is refractive transport -- caustics,
   and light bent as it passes through glass. That is the reason to reach for the
   Monte Carlo path tracer, at a large cost in time. Note that raising
   ``samples_per_pixel`` above 1 also gives up most of this page: see
   :ref:`renderer-capabilities`.

.. admonition:: How many lights can cast shadows?
   :class: seealso

   Shadow-casting lights are collected into a fixed-size per-pixel list whose
   length is a compile-time constant (default 16 -- enough for a key/fill/rim rig
   plus a 4×4-sample area light). Lights beyond that are still *lit*, just not
   shadowed, and each sample of a :class:`~.RectAreaLight` counts toward the limit,
   so an under-capped area light simply gets a shallower shadow. If you need denser
   area-light penumbras or a larger rig, set ``ALGAN_MAX_SHADOW_LIGHTS`` before the
   first render (more GPU registers, slightly lower shadow-kernel occupancy).

Environment Maps
================

An environment map wraps the scene in a 360° image. It acts as a **skybox**
(visible in the background and in reflections and refractions) and, optionally, as
**image-based lighting** -- the whole scene lit by the colours of the map.

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

Any equirectangular image works -- the world map here is just one that ships with
these docs. A real studio panorama gives a much better result.

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

Real lighting is rarely one light. The standard three-point setup -- a bright key
with the shadow, a dimmer fill opposite it to keep the shadow side readable, and a
rim from behind to separate the subject from the background -- plus a little
ambient:

.. code-block:: python

    from algan import *

    SETTINGS.raytracing.set(shadows=True)

    with Off():
        Scene.clear_light_sources()             # drop the default light

        # Key light: bright, from above and to one side, with a soft shadow.
        SpotLight(location=UP * 6 + RIGHT * 4 + OUT * 4, target=ORIGIN,
                  color=WHITE, intensity=60, angle=30, penumbra=0.5,
                  decay=2, shadow_radius=0.3).spawn()
        # Fill: dimmer, from the opposite side, no shadow.
        PointLight(location=LEFT * 6 + OUT * 2, color=WHITE, intensity=4).spawn()
        # Rim: from behind, slightly cool.
        DirectionalLight(location=IN * 8 + UP * 4, target=ORIGIN,
                         color=Color((0.6, 0.7, 1.0))).spawn()
        # Ambient: stops the shadow side going pure black.
        AmbientLight(color=WHITE, intensity=0.25).spawn()

        Sphere().spawn()
        Cube(side_length=6, color=GREY).move(DOWN * 4).spawn()   # floor

    Scene.wait(2)
    Scene.save_video()

See Also
========

- :doc:`renderer_limitations` -- which objects and materials are lit and
  shadowed at all, and where the shadow approximations show.
- :doc:`cameras` -- moving the camera through a lit scene.
- :doc:`shaders_and_materials` -- how materials respond to these lights.
- :doc:`reflections_and_glass` -- mirrors, metals and refraction.
- :doc:`performance_and_quality` -- what shadows and extra lights cost.
