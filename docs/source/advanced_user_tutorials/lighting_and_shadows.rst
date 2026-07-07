====================
Lighting and Shadows
====================

Algan's ray-traced renderer supports a full set of light sources modelled on
`Three.js <https://threejs.org/>`_'s lights, plus ray-traced shadows,
environment maps (image-based lighting) and physical camera controls. This
tutorial covers all of them.

.. note::

   Lights, shadows and environment maps affect **3-D objects** (those built on
   :class:`~.Surface`, such as :class:`~.Sphere`, :class:`~.Cylinder` and
   :class:`~.Cone`, and imported 3-D models). Flat 2-D shapes and text are drawn
   with their own colour and are not lit.

The Default Light
=================

Every scene starts with a single white :class:`~.PointLight` positioned above
and to the right of the camera (see :doc:`lights_camera_action`). You can access
it, animate it, add more lights, or replace the whole set.

.. code-block:: python

    from algan import *

    # The scene's existing lights (a list; index 0 is the default one).
    lights = Scene.get_light_sources()

    # Add another light.
    Scene.add_light_source(PointLight(location=LEFT * 5 + OUT * 3, color=BLUE).spawn())

Light Types
===========

All light types are :class:`~.Mob` s, so their ``location`` and ``color`` can be
animated like any other mob, and every light takes an ``intensity`` multiplier.
Import them from ``algan`` directly.

Point Light
-----------

:class:`~.PointLight` emits in all directions from a single point. By default it
has no distance falloff (an Algan convention that keeps scenes evenly lit), but
you can opt into physically-correct attenuation with ``decay`` and a finite
``distance`` range:

.. code-block:: python

    # Physically-attenuated point light (inverse-square falloff, fades out by 20 units).
    PointLight(location=UP * 4, color=WHITE, intensity=30, decay=2, distance=20).spawn()

Directional Light
-----------------

:class:`~.DirectionalLight` models a distant source like the sun: all its rays
are parallel, pointing from the light toward its ``target`` (the origin by
default). Distance does not matter, only direction.

.. code-block:: python

    DirectionalLight(location=UP * 10 + RIGHT * 6, target=ORIGIN, color=WHITE).spawn()

Ambient Light
-------------

:class:`~.AmbientLight` adds a flat, direction-less term to every surface. Use it
to lift shadows and fill in the dark side of objects so they never go fully
black.

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
cone's half-angle (degrees) and ``penumbra`` (0-1) softens the edge of the cone.
Like a point light, it supports ``decay`` and ``distance``.

.. code-block:: python

    SpotLight(location=UP * 6, target=ORIGIN, color=WHITE, intensity=40,
              angle=25, penumbra=0.5, decay=2).spawn()

Rect-Area Light
---------------

:class:`~.RectAreaLight` is a glowing rectangle (a softbox). It produces smooth,
soft lighting and — with shadows enabled — soft-edged shadows, because Algan
samples it at a grid of ``samples`` emitter points. More samples give a smoother
result at a proportional cost.

.. code-block:: python

    RectAreaLight(location=UP * 5, target=ORIGIN, width=4, height=4,
                  samples=16, color=WHITE, intensity=1.2).spawn()

Ray-Traced Shadows
==================

Shadows are **off by default**. Turn them on with
:func:`~algan.rendering.raytracing.set_ray_traced_shadows` before rendering:

.. code-block:: python

    from algan import *
    from algan.rendering.raytracing import set_ray_traced_shadows

    set_ray_traced_shadows(True)

    # ... build your scene ...
    render_to_file()

With shadows on, each lit surface point fires shadow rays toward every light; a
light that is blocked by another object does not contribute to that point.

Soft Shadows
------------

By default shadows are hard-edged. To get a soft penumbra, give the light a
non-zero emitter size:

- **Point** and **spot** lights take a ``shadow_radius`` (the world-space radius
  of the emitting disk).
- **Directional** lights take a ``shadow_angle`` (the angular size of the source
  in degrees, like the sun's ~0.5°).
- **Rect-area** lights are soft automatically — their penumbra smoothness is set
  by ``samples``.

.. code-block:: python

    from algan import *
    from algan.rendering.raytracing import set_ray_traced_shadows

    set_ray_traced_shadows(True)

    # A sun with a soft-edged shadow.
    DirectionalLight(location=UP * 10 + RIGHT * 4, target=ORIGIN,
                     color=WHITE, shadow_angle=4).spawn()
    AmbientLight(color=WHITE, intensity=0.4).spawn()

    Sphere().move(UP).spawn()          # caster
    # ... a ground plane to catch the shadow ...

    render_to_file()

Algan traces a fixed fan of shadow rays across the emitter to build the
penumbra; the number of rays is controlled by the environment variable
``ALGAN_SOFT_SHADOW_SAMPLES`` (default 8). Raise it for smoother penumbras at a
proportional cost.

.. note::

   Deterministic shadows are hard/soft binary visibility and ignore
   transparency. For fully physical soft shadows through translucent media, use
   the Monte-Carlo path tracer instead
   (:func:`~algan.rendering.raytracing.set_samples_per_pixel` with a value
   greater than 1).

.. admonition:: How many lights can cast shadows?
   :class: seealso

   Shadow-casting lights are collected into a fixed-size per-pixel list whose
   length is a compile-time constant (default 8). Lights beyond that are still
   *lit*, just not shadowed. This is also how many samples of a
   :class:`~.RectAreaLight` cast shadows: the light is *lit* from all its
   samples, but only the first 8 are shadow-tested — which is plenty for a clean
   penumbra. If you have a rig of more than 8 distinct shadow-casters, raise
   ``ALGAN_MAX_SHADOW_LIGHTS`` before the first render (more GPU registers,
   slightly lower shadow-kernel occupancy) and check the result — pushing it
   high can over-brighten the core of a large area light's shadow.

Environment Maps
================

An environment map wraps the scene in a 360° image. It acts as a **skybox**
(visible in the background and in reflections and refractions) and, optionally,
as **image-based lighting** — the whole scene is lit by the colours of the map.

Pass an equirectangular image (a longitude × latitude panorama, sky at the top)
to :meth:`Scene.set_environment_map <.Scene.set_environment_map>` (also available
as the top-level ``set_environment_map``):

.. code-block:: python

    from algan import *
    from algan.rendering.raytracing import set_reflectivity

    set_environment_map("studio_panorama.jpg", intensity=1.0, ambient=True)

    # A mirror sphere reflects the environment; other objects are lit by it.
    mirror = Sphere().move(LEFT * 1.5)
    set_reflectivity(mirror, 0.9)
    mirror.spawn()

    Sphere().move(RIGHT * 1.5).spawn()

    render_to_file()

- ``intensity`` scales the map's brightness.
- ``ambient=True`` (the default) also lights surfaces from the map (image-based
  lighting). Set ``ambient=False`` to use the map only as a backdrop and in
  reflections, without it contributing diffuse light.
- Pass ``None`` to remove a previously-set environment map.

You can also pass a ``[height, width, 3]`` tensor or NumPy array instead of a
file path.

Physical Camera Controls
========================

The :class:`~.Camera` supports the field-of-view and clipping controls familiar
from Three.js, in addition to Algan's ``screen_distance`` / ``screen_scale``
perspective controls (see :doc:`lights_camera_action`).

Field of View
-------------

``fov`` is the camera's vertical field of view in degrees. A small fov is a
telephoto lens (flattened perspective, distant subject); a large fov is a
wide-angle lens. You can set it at construction or animate it:

.. code-block:: python

    camera = Scene.get_camera()
    camera.set_fov(30)          # telephoto
    # camera.fov = 30           # equivalent (property form)
    print(camera.get_fov())     # read it back

Because ``set_fov`` moves the camera's screen, it animates like any other camera
change when the camera has been spawned.

Near and Far Clipping
--------------------

``near`` and ``far`` are clip distances measured from the camera. Geometry closer
than ``near`` or farther than ``far`` is not drawn (past ``far``, the background
or environment map shows through). ``0`` disables each (the default).

.. code-block:: python

    camera = Scene.get_camera()
    camera.set_near(0.5)        # hide anything within 0.5 units of the camera
    camera.set_far(50)          # hide anything beyond 50 units

Putting It Together
==================

A small three-point-light studio with soft shadows and a fill from an ambient
light:

.. code-block:: python

    from algan import *
    from algan.rendering.raytracing import set_ray_traced_shadows

    set_ray_traced_shadows(True)

    Scene.get_light_sources().clear()   # drop the default light
    # Key light (bright, soft shadow), fill (dimmer), and a rim from behind.
    Scene.add_light_source(
        SpotLight(location=UP * 6 + RIGHT * 4 + OUT * 4, target=ORIGIN,
                  color=WHITE, intensity=60, angle=30, penumbra=0.5,
                  decay=2, shadow_radius=0.3).spawn())
    Scene.add_light_source(
        PointLight(location=LEFT * 6 + OUT * 2, color=WHITE, intensity=4).spawn())
    Scene.add_light_source(
        DirectionalLight(location=IN * 8 + UP * 4, target=ORIGIN,
                         color=(0.6, 0.7, 1.0)).spawn())
    Scene.add_light_source(AmbientLight(color=WHITE, intensity=0.25).spawn())

    Sphere().spawn()
    # ... a ground plane to catch the shadows ...

    render_to_file()

See Also
========

- :doc:`lights_camera_action` — animating the default light and camera.
- :doc:`shaders_and_materials` — how materials respond to these lights.
