===================
Images and Textures
===================

Algan can colour a surface from an image or an array instead of a single flat
colour, and can drive a surface's *material* properties -- roughness, reflectivity,
refractive index, surface normals -- from images too.

There are three separate mechanisms, and it is worth knowing which is which:

1. :class:`~.ImageMob` -- an image as a flat, textured Mob. What you want for
   showing a picture.
2. :class:`~.Surface` texture arguments -- per-texel colour and material properties
   on any 3-D surface, sampled in the ray tracing kernel.
3. :meth:`~algan.scene.Scene.set_background_color` -- an image behind the whole scene (see
   :doc:`backgrounds_and_post_processing`).

.. note::

    The Three.js material classes accept Three.js's image slots (``map``,
    ``normalMap``, ``roughnessMap``, ...) for API parity but **do not sample them**.
    Texturing goes through :class:`~.Surface`, as described below.

Showing an Image
================

:class:`~.ImageMob` takes an image file path or an RGBA array and gives you a flat
textured surface:

.. algan:: TexturesImageMob

    from algan import *

    photo = ImageMob('world_map.png').scale(2).spawn()
    with Seq(run_time=2):
        photo.rotate(30, UP)
        photo.rotate(-30, UP)

    Scene.save_video()

Image paths are resolved against the working directory and then against the
directory holding your script, so an image sitting beside your ``.py`` file loads
regardless of where you launch Python from. The same resolution applies to
:meth:`~algan.scene.Scene.set_background_color`,
:meth:`~algan.scene.Scene.set_environment_map` and
:class:`~algan.mobs.three_d_models.model_mob.ThreeDModelMob`.

Instead of a path you can pass a ``[H, W, 4]`` or ``[H, W, 5]`` tensor, which is how
you texture something with data you computed rather than loaded.

.. important::

    The per-material texture arguments on :class:`~algan.mobs.surfaces.surface.Surface`
    -- ``color_texture``, ``roughness_texture``, ``normal_texture`` and the rest --
    take **tensors only**. Handing one a file path raises ``TypeError``. Load the
    image yourself first, with :func:`~algan.utils.file_utils.get_image`, or use
    :meth:`~algan.mobs.surfaces.surface.Surface.set_color_by_image`, which takes a
    path and orients the image onto the surface's ``(u, v)`` axes for you.

Reshaping a textured surface
============================

:class:`~.ImageMob` is itself a :class:`~.Surface`, so you can change its *shape*
while it keeps its texture. That is how you wrap a map onto a globe:

.. algan:: TexturesReshaping

    from algan import *

    # Start as a flat plane coloured by our image file.
    mob = ImageMob('world_map.png').scale(2).spawn()
    mob.wait()

    with Seq(run_time_unit=10, rate_func=rate_funcs.identity):
        for shape in (Sphere(radius=2), Cylinder(radius=1, height=2)):
            # Change the surface shape; the texture comes along.
            mob.set_shape_to(shape)
            mob.rotate(360, UP)
            mob.rotate(360, RIGHT)

    Scene.save_video()

:meth:`~.Surface.set_shape_to` re-maps the surface's intrinsic (UV) coordinates onto
a new shape, and the texture follows them. Any :class:`~.Surface` works as a target.

.. note::

    A low-resolution surface is automatically resized to a higher-resolution grid
    when the target shape needs one, so morphing a flat plane into a sphere does not
    come out faceted.

Texturing Any Surface
=====================

:class:`~.Surface` and everything built on it (:class:`~.Sphere`,
:class:`~.Cylinder`, :class:`~.Cone`, :class:`~.Torus`, your own surface functions)
take texture arguments at construction:

.. algan:: TexturesColorTexture

    from algan import *
    import torch

    # A 16x16 checkerboard as an RGB + glow + opacity texture.
    checker = torch.zeros(16, 16, 5)
    grid = (torch.arange(16).view(-1, 1) + torch.arange(16).view(1, -1)) % 2
    checker[..., 0] = grid          # red channel
    checker[..., 2] = 1 - grid      # blue channel
    checker[..., 4] = 1.0           # opacity

    globe = Sphere(radius=1.5, color_texture=checker).spawn()
    with Seq(run_time=3):
        globe.rotate(360, UP)

    Scene.save_video()

The available texture arguments:

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Argument
     - Shape
     - What it drives
   * - ``color_texture``
     - ``[W, H, 5]``
     - Base colour: red, green, blue, glow, opacity.
   * - ``roughness_texture``
     - ``[W, H, 1]``
     - How blurred reflections are, per texel.
   * - ``reflectivity_texture``
     - ``[W, H, 1]``
     - Metalness, per texel.
   * - ``refractive_index_texture``
     - ``[W, H, 1]``
     - Index of refraction, per texel.
   * - ``normal_texture``
     - ``[W, H, 3]``
     - Tangent-space normal map; perturbs the shading normal.
   * - ``glow_texture``
     - ``[W, H, 1]``
     - Glow strength, per texel.

Colour and the three material property maps are sampled **bilinearly per fragment,
inside the ray tracing kernel**, for both flat and curved (PN) triangles. A property
without a map keeps the ordinary per-vertex value, and maps of different resolutions
are resampled to a common one.

Animating a texture
-------------------

A texture map is an ordinary animatable attribute, so you animate it the way you
animate a colour or a location: **assign a new one**. Algan interpolates the old
texture to the new one per texel over the current context's duration.

.. code-block:: python

    surface = Sphere(color_texture=day).spawn()
    with Seq(run_time=3):
        surface.color_texture = night     # cross-fades, texel by texel

The replacement must have the same shape as the texture it replaces. Pass a single
image when you construct the Mob -- one map, not a sequence of them; there is no
time axis on a texture argument.

Normal maps
-----------

A ``normal_texture`` is a tangent-space normal map with components in ``[-1, 1]``:
x along increasing ``u``, y along increasing ``v``, z along the smooth surface
normal, so ``(0, 0, 1)`` means "unperturbed".

.. important::

    Under the default vertex-shaded pipeline, lighting is baked at the vertices, so a
    normal map only affects things evaluated per fragment: mirror reflections,
    refraction, ray-traced shadows, and fragment shading. If a normal map appears to
    do nothing to the diffuse shading, that is why -- see
    :doc:`shaders_and_materials`.

Glow maps
---------

``glow_texture`` is the exception to the per-fragment rule: glow is consumed by the
glow accumulator per *vertex*, so the map is baked down to the surface grid
resolution. Raise ``grid_width`` / ``grid_height`` if you need more detail from it.

Choosing a resolution
=====================

A surface's texture detail is limited by two independent things: the resolution of
the image you supply, and (for glow) the surface's own grid resolution.
:class:`~.Surface` sizes its grid automatically between ``min_grid_resolution`` and
``max_grid_resolution``, and dices curved triangles at render time to
``render_tolerance`` -- a fraction of screen height, so a surface that fills the
frame gets more triangles than one in the distance.

Textures also cost render memory. If a heavily textured scene runs out of it, reduce
the texture resolution before reducing anything else; see
:doc:`performance_and_quality`.

See Also
========

- :doc:`three_d_models` -- imported models bring their own textures and materials.
- :doc:`shaders_and_materials` -- what each material property does.
- :doc:`reflections_and_glass` -- the reflection and refraction those maps drive.
- :doc:`backgrounds_and_post_processing` -- an image behind the whole scene.
