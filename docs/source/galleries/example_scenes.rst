==============
Example Scenes
==============

Complete scripts that show what the renderer can do when a scene is built
for realism. Each runs as it is from a checkout of the repository; the videos
below were rendered from them.

Sunlit Swimming Pool
====================

.. raw:: html

   <video autoplay muted loop playsinline controls
          style="width: 100%; max-width: 960px; display: block; margin: 0 auto 1.5em auto; border-radius: 8px;"
          poster="../_static/examples/water-pool.jpg">
     <source src="../_static/examples/water-pool.mp4" type="video/mp4">
   </video>

Wind-ruffled water over a tiled pool on a sunny day, path traced at 80 samples
per pixel. The video loops seamlessly every six seconds. The scene puts several
features together:

* **The water** is a flat :class:`~algan.mobs.surfaces.surface.Surface`
  deformed by a :func:`~.Homotopy` that sums nine Gerstner waves. In a
  Gerstner wave the surface points move in small circles, so the crests come
  out sharp and the troughs broad, as on real water. Each wave's speed comes
  from deep-water dispersion, rounded so that it completes a whole number of
  cycles per loop.
* **Refraction and reflection** come from a
  :class:`~algan.rendering.shaders.materials.MeshPhysicalMaterial` with
  ``transmission=1`` and ``ior=1.33``, the refractive index of water. A faint
  cyan tint in its ``color`` colours the light that passes through. See
  :doc:`../advanced_user_tutorials/reflections_and_glass`.
* **The sky** is a procedural HDR environment map: a float tensor passed to
  :meth:`~algan.scene.Scene.set_environment_map`, with a sun disc hundreds of
  times brighter than white, which puts glints on the wave crests. A
  :class:`~algan.rendering.lights.DirectionalLight` along the same direction
  casts the shadows.
* **The caustics**, the bright net of light on the pool floor, are computed by
  the script itself, because Algan's renderers do not trace caustics (see
  :doc:`../advanced_user_tutorials/renderer_limitations`). Every frame it
  refracts a dense grid of sun rays through the same wave surface with Snell's
  law, gathers where they land on the floor, and writes the result into the
  floor's :attr:`~algan.mobs.surfaces.surface.Surface.color_texture`. A
  texture is an animatable attribute (see
  :doc:`../advanced_user_tutorials/images_and_textures`), so assigning one per
  frame inside a :class:`~.Seq` animates it in step with the waves.
* **The materials** are built from procedural textures, such as
  :func:`~algan.mobs.surfaces.procedural_textures.get_grid_lines` for the tiles
  and :func:`~algan.mobs.surfaces.procedural_textures.get_noise` for the stone
  and the clouds. A small random shade per tile breaks up the repetition.

Render it with:

.. code-block:: bash

    python examples/water_pool.py           # the video, 1080p at 80 samples per pixel
    python examples/water_pool.py still     # one 1080p frame at 256 samples per pixel
    python examples/water_pool.py draft     # a quick low-resolution preview

The full video takes about 45 minutes on a Tesla T4 GPU and much longer on a
CPU, so start with ``draft``.

.. literalinclude:: ../../../examples/water_pool.py
   :language: python
