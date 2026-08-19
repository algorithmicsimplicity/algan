=======================
Performance and Quality
=======================

Algan trades render time for image quality along several independent axes. This
tutorial covers which knob does what, in roughly the order you should reach for
them, and what to do when a render runs out of memory.

The short version: **work at low quality and render the final version once.**

.. code-block:: python

    # While iterating: use the defaults, which are already a preview quality.
    Scene.save_video("draft")

    # For the final render, override the settings for that render only.
    Scene.save_video("final", HD)

Passing a preset to :meth:`~algan.scene.Scene.save_video` applies it to that render alone and
restores the Scene's own settings afterwards, so you never have to remember to put
it back.

The Fast Feedback Loop
======================

Before tuning any renderer setting, get rid of the fixed costs. A fresh
``python scene.py`` pays several seconds of library import plus Taichi kernel
preparation before the first pixel appears. The render daemon pays that once and
then re-runs your script on demand:

.. code-block:: bash

    python -m algan.daemon

A re-render then costs only the render itself -- around a second for a simple scene, against roughly twenty

.. important::

    Never edit Algan's own ``*_taichi.py`` kernel sources while a render process or
    daemon is running -- the JIT reads them at first launch and can compile
    half-edited code. Restart the daemon after changing any Algan source.

Video Settings
==============

``SETTINGS.video`` holds resolution, frame rate and anti-aliasing. The presets:

.. list-table::
   :header-rows: 1
   :widths: 18 22 14 46

   * - Preset
     - Resolution
     - FPS
     - Use for
   * - ``SMOKE_TEST``
     - 32 × 32
     - 2
     - Automated tests; checks a scene runs at all.
   * - ``PREVIEW``
     - 704 × 396
     - 10
     - The fastest usable look at a scene.
   * - ``LD``
     - 864 × 486
     - 15
     - The default. Iterating on timing and composition.
   * - ``MD``
     - 1280 × 720
     - 30
     - Checking detail; acceptable to publish.
   * - ``HD``
     - 1920 × 1080
     - 30
     - Normal final output.
   * - ``PRODUCTION``
     - 2560 × 1440
     - 60
     - High-motion final output.
   * - ``UHD``
     - 3840 × 2160
     - 60
     - 4K final output.
   * - ``THUMBNAIL``
     - 1280 × 720
     - 1
     - A single still.

Cost scales with pixels *and* frames, so ``HD`` at 30 fps is roughly ten times
``LD`` at 15 fps. Presets are immutable; ``HD.set(frames_per_second=60)`` returns a
modified copy rather than changing ``HD``.

``anti_alias_level`` (default ``2``) multiplies the rendered resolution in each axis,
so it costs its square: level 2 renders 4× the pixels. Dropping it to ``1`` and
enabling ``fxaa`` is the single cheapest way to speed up a draft. See
:doc:`backgrounds_and_post_processing`.

Renderer Settings
=================

``SETTINGS.raytracing`` controls what the renderer produces.

samples_per_pixel
-----------------

This is the biggest single decision, because it selects the renderer:

.. code-block:: python

    SETTINGS.raytracing.set(samples_per_pixel=1)     # default: deterministic
    SETTINGS.raytracing.set(samples_per_pixel=64)    # Monte Carlo path tracer

* ``1`` uses the **deterministic renderer**: a hybrid pipeline that rasterizes
  primary visibility and traces rays for reflection, refraction and shadows. Fast,
  noise-free, and what every example in this documentation uses.
* Above ``1`` switches to the **Monte Carlo path tracer**, which gives true global
  illumination and physically-correct soft shadows and rough reflections, at
  dramatically higher cost -- and needs a high sample count to be free of noise.

Only reach for path tracing when you specifically need full light transport. It is
also a separate GPU kernel with its own cold compile of several minutes on first use.

.. _renderer-capabilities:

What each renderer supports
---------------------------

Raising ``samples_per_pixel`` is not a pure quality dial: it changes renderer, and
several features are implemented only in the deterministic one. Algan checks this
before it allocates anything and refuses rather than silently dropping them.

.. list-table::
   :header-rows: 1
   :widths: 38 31 31

   * - Feature
     - Deterministic (``spp == 1``)
     - Monte Carlo (``spp > 1``)
   * - Environment maps
     - Yes
     - **Not supported**
   * - Refractive materials (glass)
     - Yes
     - **Not supported**
   * - Custom fragment-shader pipelines
     - Yes
     - **Not supported**
   * - Extended lights
     - Yes
     - **Not supported**
   * - Global illumination, caustics
     - No
     - Yes

"Extended lights" means any light carrying parameters beyond a position and a
colour -- a cone angle, a ground colour, an emitter radius, a distance falloff.
:class:`~.PointLight` is the only one that is not: :class:`~.SpotLight`,
:class:`~.DirectionalLight`, :class:`~.AmbientLight` and
:class:`~.RectAreaLight` are all extended, so a scene using any of them cannot
be path traced. Note that a Scene starts with a point light, so clearing the
lights and adding your own is often what first trips this.

If a scene requests an unsupported feature, Algan raises
:class:`~algan.errors.UnsupportedFeatureError` naming the features it cannot
honor. Either set ``samples_per_pixel`` back to ``1``, remove the feature, or opt
into the older behaviour explicitly:

.. code-block:: python

    SETTINGS.raytracing.set(unsupported_feature_policy="warn")    # or "ignore"

The default is ``"error"``. ``RenderResult.render_plan`` records which backend ran
and which features were requested, if you want to check programmatically.

shadows
-------

Off by default. Turning shadows on makes every lit surface point fire rays at every
light, so the cost scales with the number of shadow-casting lights. Soft shadows
multiply that by the soft-shadow sample count (``ALGAN_SOFT_SHADOW_SAMPLES``,
default 8). Leave shadows off while blocking out a scene.

max_bounces
-----------

Default ``8``. Caps how far a reflected or refracted ray can keep going. Glass is the
expensive case, because each hit spawns both a reflected and a refracted ray. Lower
it for drafts of reflective scenes; raise it if glass interiors look unexpectedly
dark. See :doc:`reflections_and_glass`.

What Makes a Scene Expensive
============================

In rough order of impact:

1. **How much of the frame the geometry fills.** ``render_tolerance`` on a
   :class:`~.Surface` is expressed as a fraction of screen height, so a surface
   filling the frame is diced far more finely than the same surface in the distance.
   The dice is decided per curved triangle, per frame and per direction: the part of
   a surface that is close to the camera, or turned edge-on, is refined without the
   rest of the mesh following it; a frame where the surface is small or off screen
   costs what that frame needs rather than what the closest frame of the batch needs;
   and a direction the surface is straight along costs nothing, so a long thin
   cylinder pays for its circumference and not for its length. Raising
   ``render_tolerance`` slightly is a large speedup on close-up surfaces.
2. **Resolution × anti-aliasing × frame count.** Straightforwardly multiplicative.
3. **Refraction.** Splits every ray in two, and routes the batch to the general
   wavefront tracer.
4. **Shadows**, multiplied by the number of lights.
5. **Triangle count.** Imported models and high-resolution
   :class:`~.Surface` grids.
6. **Glow and bloom**, which add a full-frame post-processing pass.
7. **Distinct shaders and materials.** Mobs with different shaders are batched
   separately, so reusing one shader function across many Mobs batches better than
   defining an equivalent one per Mob.

Running Out of Memory
=====================

Render memory is a fixed budget -- ``SETTINGS.computing.rendering_memory_fraction``
(default ``0.4``) of the device's memory. When a batch does not fit, Algan shrinks
the frame window and retries. If even a single frame will not fit, it raises:

.. code-block:: text

    OutOfRenderMemory: The prepared scene plus one rendered frame does not fit in
    the allocated render memory. Please lower the resolution, anti-alias level, or
    scene complexity.

What to try, in order:

1. **Raise ``render_tolerance``** on the surfaces that fill the frame, for the reason
   above. This is the first thing to reach for whenever the scene contains a
   :class:`~.Surface` close to the camera.
2. **Lower ``anti_alias_level``** to 1. Cuts the pixel count fourfold.
3. **Drop to a smaller preset** for the draft.
4. **Reduce geometry**: fewer Mobs on screen at once, coarser
   ``grid_width``/``grid_height`` on surfaces.
5. **Reduce texture resolution.** Textures live in render memory too.
6. **Raise ``rendering_memory_fraction``** if the device genuinely has room spare:
   ``SETTINGS.computing.set(rendering_memory_fraction=0.6)``.

Shortening the camera move is *not* on that list. Framing a subject closer costs
more, but travelling further does not: geometry that leaves the frame stops being
tessellated and stops being rasterized.

Devices
=======

The animation and render devices are chosen at import time and cannot be changed
afterwards, so they are environment variables set **before** ``import algan``:

.. code-block:: python

    import os
    os.environ["ALGAN_RENDER_DEVICE"] = "cuda"     # or "auto", "mps", "cpu"
    os.environ["ALGAN_ANIMATION_DEVICE"] = "cpu"

    from algan import *

``ALGAN_RENDER_DEVICE`` defaults to ``auto``, which picks CUDA, then MPS, then CPU.
The animation device defaults to ``cpu``, which is usually right: materializing the
timeline is Python-bound rather than compute-bound, and the geometry is moved to the
render device afterwards anyway.

:meth:`~algan.settings.abstract_settings.Settings.set` raises a specific error,
rather than a generic unknown
setting error, if you try to set the render device at runtime.

Caching
=======

Algan caches aggressively. Everything lives under ``~/.algan/cache``:

* **Compiled Taichi kernels.** Cold compilation takes minutes; after that it is
  instant. Version-keyed and never invalidated by scene content.
* **LaTeX and font glyph geometry.** Only the first render of a given string pays the
  LaTeX cost.
* **Surface tessellations** and **audio**.

Clear it with the helper rather than by hand:

.. code-block:: python

    from algan import clear_cache

    clear_cache()                     # content caches; keeps compiled kernels
    clear_cache(taichi_kernels=True)  # everything, including compiled kernels

The kernel cache is spared by default because it is expensive to rebuild and is never
invalidated by anything you write in a scene.

Measuring
=========

Wall-clock timing of GPU work is noisy -- thermal throttling alone can swing
throughput about twofold between processes -- so do not compare two runs of a script
and conclude much. ``algan/utils/profiling_utils.py`` hooks every Taichi kernel and
pipeline stage and reports device times, which is what to use when you actually need
numbers.

For a rough sense of where a render's time goes, the render log already prints
per-batch fetch and render times as it goes.

See Also
========

- :doc:`settings` -- how the settings system works, including temporary overrides.
- :doc:`reflections_and_glass` -- the cost of bounces.
- :doc:`lighting_and_shadows` -- the cost of lights and shadows.
- :doc:`backgrounds_and_post_processing` -- anti-aliasing and bloom.
