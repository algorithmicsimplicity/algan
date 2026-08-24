=====================
Reflections and Glass
=====================

Algan's renderer is a ray tracer, so reflection and refraction are not tricks --
rays really do bounce off metal and bend through glass. This tutorial covers how to
ask for them and how to make them look like something.

Everything here is driven by
:meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_material`.
There are no separate reflectivity or refractive-index setters: a
:class:`~algan.rendering.shaders.materials.MeshStandardMaterial`'s
``metalness`` and ``roughness`` control reflection, and a
:class:`~algan.rendering.shaders.materials.MeshPhysicalMaterial`'s ``transmission``
and ``ior`` control refraction.
See :doc:`shaders_and_materials` for the material catalogue.

.. important::

    :meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_material` must be called **before** the Mob is spawned, and
    reflection and refraction apply to 3-D objects only.

Reflections
===========

``metalness`` is how metallic a surface is: ``0`` is a dielectric (plastic, stone,
painted wood), ``1`` is bare metal. ``roughness`` is how polished it is: ``0`` is a
perfect mirror, ``1`` is fully diffuse. Keep it low where you want the scene to
appear in the surface -- see `Glossy Reflections`_ for what a high one does.

.. algan:: ReflectionsMetalFloor

    from algan import *

    with Off():
        Prism(dimensions=(9, 0.2, 9), color=GREY).move(DOWN * 1.4).set_material(
            MeshStandardMaterial(metalness=0.8, roughness=0.1)).spawn()
        balls = Group([Sphere(radius=0.5, color=c).move(RIGHT * x + DOWN * 0.6)
                       for c, x in ((RED, -1.8), (YELLOW, 0), (BLUE, 1.8))]).spawn()
        Scene.get_camera().move(UP * 1.2).look_at(ORIGIN)

    with Seq(run_time=3):
        balls.move(UP * 1.4)
        balls.move(DOWN * 1.4)

    Scene.save_video()

A flat polished floor is the easiest way to show off reflections, and a useful
trick generally: it grounds objects that would otherwise float in a void.

.. important::

    **A mirror needs something to reflect.** ``metalness=1`` means the surface has
    no diffuse color of its own at all, so a fully metallic object in an otherwise
    empty scene renders *black* -- it is faithfully reflecting a black background.

    Three ways to fix that, in order of effectiveness:

    1. Give the scene an environment map, so there is always something to reflect
       (see :doc:`lighting_and_shadows`).
    2. Put other objects around it.
    3. Back the metalness off to ``0.7``-``0.9`` so some base color shows through.

    Note also that a *convex* mirror shrinks whatever it reflects enormously. Flat
    or gently curved reflectors read much better than a mirrored sphere.

The reflection you see is of the neighbouring geometry as *it* is shaded, so a
reflected object can legitimately look different from the object itself -- the
reflection sees a different side of it, lit differently.

Glass and Refraction
====================

Refraction needs a
:class:`~algan.rendering.shaders.materials.MeshPhysicalMaterial` with a non-zero ``transmission``
and an ``ior`` (index of refraction) above 1:

.. algan:: ReflectionsGlass

    from algan import *

    with Off():
        Group([Square(color=c).scale(0.5).move(RIGHT * x + UP * y)
               for c, x, y in ((RED, -1.2, 0.8), (GREEN, 1.2, 0.8),
                               (YELLOW, -1.2, -0.8), (BLUE, 1.2, -0.8))]
              ).move(IN * 3).spawn()
        glass = Sphere(radius=1.1, color=WHITE).set_material(
            MeshPhysicalMaterial(transmission=1.0, ior=1.5, roughness=0.0)).spawn()

    with Seq(run_time=3):
        glass.move(RIGHT * 1.5)
        glass.move(LEFT * 3)

    Scene.save_video()

The sphere acts as a lens, inverting and displacing what is behind it. That
displacement is the whole visual point of refraction -- glass with nothing
interesting behind it looks like nothing at all, so **always put a patterned
backdrop behind a glass object**.

Useful indices of refraction:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Material
     - ``ior``
     - Notes
   * - Air / vacuum
     - 1.0
     - No bending. Values ``<= 1`` disable the dielectric lobe.
   * - Water
     - 1.33
     -
   * - Common glass
     - 1.5
     - The :class:`~algan.rendering.shaders.materials.MeshPhysicalMaterial` default.
   * - Sapphire
     - 1.77
     -
   * - Diamond
     - 2.42
     - Strong bending, obvious total internal reflection.

Transmission and opacity are independent
----------------------------------------

This trips people up, so it is worth stating plainly:

* :ref:`opacity <reference-mob-opacity>` is *coverage* -- how much of the pixel the object occupies
  at all. Lower it and the object fades out; light passes straight through the part
  that is not there, undeviated.
* ``transmission`` is *transparency* -- how much light passes through the part of
  the object that **is** there. That light is refracted, tinted, and subject to
  total internal reflection.

For glass you want ``transmission=1.0`` and ``opacity=1.0``. Fading a glass object
in and out is a change to ``opacity``; making it more or less glassy is a change to
``transmission``.

A transmissive material's color tints the light passing through it, so
``MeshPhysicalMaterial(color=GREEN, transmission=1.0)`` gives green glass.

Controlling Bounce Depth
========================

Every reflection or refraction spawns a new ray, and ``max_bounces`` caps how far
that goes:

.. code-block:: python

    SETTINGS.raytracing.set(max_bounces=8)    # the default

A solid glass sphere needs at least four bounces to look right (in the front
surface, out the back, and the internal reflections). Two mirrors facing each other
will happily consume any budget you give them. Rays that run out of bounces stop
contributing, which shows up as unexpectedly dark patches inside glass -- if that is
what you are seeing, raise ``max_bounces``.

Lowering it is a good way to speed up a draft render of a scene full of glass.

Glossy Reflections
==================

``roughness`` decides how much of a surface's specular energy is spent on a
*traced* reflection at all, and that is what makes brushed metal look different
from chrome:

.. algan:: ReflectionsGlossy

    from algan import *

    with Off():
        Group([Square(color=c).scale(0.6).move(RIGHT * x + UP * y)
               for c, x, y in ((RED, -2, 1), (GREEN, 2, 1),
                               (YELLOW, -2, -1), (BLUE, 2, -1))]
              ).move(IN * 4).spawn()

        chrome = Sphere(radius=0.9, color=GREY).move(LEFT * 1.6).set_material(
            MeshStandardMaterial(metalness=1.0, roughness=0.0)).spawn()
        brushed = Sphere(radius=0.9, color=GREY).move(RIGHT * 1.6).set_material(
            MeshStandardMaterial(metalness=1.0, roughness=0.35)).spawn()

    Scene.wait(2)

    Scene.save_video()

The left sphere reflects the squares behind the camera's subject; the right one
does not.

A mirror (``roughness=0``) reflects the scene exactly. As roughness rises the
reflected image fades out and the surface's own shading -- its broad specular
highlight and the ambient/environment term -- takes over: half the reflection is
still traced at ``roughness=0.15``, almost none of it by ``0.35``. So a rough
metal reads as a rough metal, but it does not show you a picture of the room.

.. note::

    The reason it fades rather than blurring is a sample budget. The default
    single-sample renderer spends at most four traced rays per reflective pixel,
    and four rays cannot integrate a wide lobe: they either dither (and the
    dither crawls as the object moves) or land as four ghost copies of the
    reflected image. Drawing the reflection sharp instead is worse still --
    a reflection is usually a heavily minified image, so a sharp one aliases
    into hard bright patches across the surface.

    ``SETTINGS.raytracing.set(glossy_reflection=True)`` opts into a real blur,
    and it does not spend more rays to get one -- it spends **one**. The
    reflection is traced in the mirror direction into a buffer of its own, and
    that buffer is blurred by how wide the lobe is on screen before it is
    composited back with the lobe's exact energy (the split-sum approximation:
    ``DESIGN_glossy_prefilter.md`` in the renderer, Karis 2013 for the theory).
    Blurring a picture is what a wide lobe does; sampling it four times is not.
    Nothing dithers, nothing ghosts, and nothing crawls when the object moves,
    because a blur radius is a smooth function of position where a four-tap fan
    is not.

    Two things to know before turning it on. It reflects **what is on screen**,
    so a reflector shows the background where it should show something behind
    the camera or outside the frame -- give the scene an environment map (below)
    and it has something to reflect either way. And a rough metal will read
    *darker* than it does with this off: with it off the surface keeps its
    ambient fill in place of the reflection it is not drawing, and with it on
    that energy goes into the reflection, which is only as bright as the room
    around it.

    The four-tap fan is still there for comparison --
    ``set(glossy_reflection=True, prefilter=False)`` -- but it is not the
    recommended path.

    For fully physical rough reflections, raise
    ``SETTINGS.raytracing.samples_per_pixel`` above 1 to switch to the Monte
    Carlo path tracer, which jitters every bounce and has the samples to
    resolve it.

    That switch is not free on this page's subject matter: the path tracer supports
    neither refraction nor environment maps, so a glass scene or an
    environment-lit metal scene will raise rather than render. See
    :ref:`renderer-capabilities` before reaching for it.

Environment Maps
================

For any serious metal or glass work, an environment map is the highest-value change
you can make. It gives every reflective surface something to reflect and every
refractive one something to bend, and it lights the whole scene:

.. algan:: ReflectionsEnvironmentMap

    from algan import *

    set_environment_map("world_map.png", intensity=1.0, ambient=True)

    Sphere().set_material(
        MeshStandardMaterial(metalness=1.0, roughness=0.05)).spawn()

    Scene.save_video()

Any equirectangular image works; a real studio panorama gives a far better result
than the world map used here.

See :doc:`lighting_and_shadows` for the details.

Performance
===========

Reflection and refraction are the most expensive features in the renderer, because
each bounce is another full ray traversal:

* A **refractive** object splits each ray into a reflected *and* a refracted ray, so
  glass costs more than metal.
* Refraction is implemented only by the deterministic wavefront renderer, which is
  where every ``samples_per_pixel == 1`` render already goes. Raising
  ``samples_per_pixel`` selects the Monte Carlo path tracer, which cannot honor
  refraction or environment maps -- see :ref:`renderer-capabilities`.
* ``max_bounces`` multiplies the cost of everything reflective in the scene.

While you are iterating on a shot, lower ``max_bounces`` and leave the resolution at
the default; raise both for the final render. See
:doc:`performance_and_quality`.

See Also
========

- :doc:`shaders_and_materials` -- the full material catalogue, and custom scatter
  functions for ray-continuation behaviour Algan does not provide.
- :doc:`lighting_and_shadows` -- environment maps and lighting rigs.
- :doc:`images_and_textures` -- varying roughness, reflectivity and IOR per texel.
- :doc:`renderer_limitations` -- exactly where reflection and refraction stop,
  and what happens at the end of the bounce budget.
- :doc:`performance_and_quality` -- the settings on this page, and what each
  costs.
