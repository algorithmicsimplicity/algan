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

``supersampling`` -- ``ssaa`` for short -- defaults to ``2`` and
multiplies the rendered resolution in each axis, so it costs its square: level 2
renders 4× the pixels. Dropping it to ``1`` and enabling ``fxaa`` is the single
cheapest way to speed up a draft. See :doc:`backgrounds_and_post_processing`.

Renderer Settings
=================

``SETTINGS.raytracing`` controls what the renderer produces.

.. _renderer-settings:

samples_per_pixel
-----------------

This is the biggest single decision, because it selects the renderer:

.. code-block:: python

    SETTINGS.raytracing.set(samples_per_pixel=1)     # default: deterministic
    SETTINGS.raytracing.set(samples_per_pixel=64)    # path tracer

* ``1`` uses the **deterministic renderer**: a hybrid pipeline that rasterizes
  primary visibility and traces rays for reflection, refraction and shadows. Fast,
  noise-free, and what every example in this documentation uses.
* Above ``1`` switches to the **path tracer**, which gives true global
  illumination, physically-correct soft shadows and rough reflections, area and
  emissive lighting, and image-based environment lighting, at higher cost per
  frame. Its raw output is noisy at low sample counts, so by default it
  is **denoised** (``SETTINGS.raytracing.denoise``, on) with a neural filter
  guided by the render's own albedo and normal information; turn it off to see
  or gate the raw estimator.

**The path tracer is the fallback for scenes the deterministic renderer cannot
do.** Three kinds of scene fail there and render here:

* **Many lights.** The deterministic renderer's cost grows with every light
  and it shadows at most 16 light slots (:ref:`limits-truncation`; a 4x4 area
  light spends 16 on its own). The path tracer samples lights instead of
  summing them, so its cost per shading point does not depend on how many
  there are, and every light casts a shadow. That holds for
  authored-appearance materials as well -- toon, normal, matcap, depth,
  Manim's material and custom fragment pipelines, whose lighting is *defined*
  as a sum over the light rows: past the shadow cap the path tracer samples
  those rows too, so they get a shadow from every light rather than from the
  first 16 (``SETTINGS.raytracing.experimental.pt_authored_light_sampling``,
  ``"auto"`` by default; ``"off"`` restores the exact sum and its cap). The
  price is that their lighting becomes an estimate that converges with
  ``samples_per_pixel``, which is why the default keeps the exact sum on a rig
  small enough to afford it.
* **Reflective and transparent geometry that exhausts render memory.** The
  deterministic renderer splits a ray at every reflective or refractive
  surface, so enough such surfaces make one frame not fit; the path tracer
  never splits, so its memory per path is fixed whatever the scene does.
* **Global illumination**: colour bleed, soft shadows from real area emitters,
  rough reflections that see the scene.

When a render fails for one of these reasons the warning or error names the
switch. The setting to reach for is a modest sample count and a *short* bounce
budget -- direct lighting needs one bounce, and the denoiser handles the
residual noise:

.. code-block:: python

    SETTINGS.raytracing.set(samples_per_pixel=16, max_bounces=2)

Raise ``max_bounces`` when the scene needs indirect light, and
``samples_per_pixel`` when the denoised result still shows structure in the
noise. Reach for path tracing only for these reasons: for flat 2-D artwork and
text the deterministic renderer is not merely cheaper but *better*, since it
resolves those edges with exact analytic coverage where the path tracer must
estimate them by sampling. The path tracer renders at output resolution and
anti-aliases by jittering its samples inside each pixel, so
``supersampling`` does not apply to it.

``samples_per_pixel`` is a **ceiling, not a count**. A pixel stops early only
when it was never going to change: the renderer knows which pixels it resolved
deterministically -- flat 2-D artwork, text, unlit transparency and the
background all composite with no randomness at all -- and gives those
``SETTINGS.raytracing.experimental.pt_min_samples`` samples (4) plus whatever
their own error estimate asks for, up to
``SETTINGS.raytracing.experimental.pt_error_target`` (0.02). **Any pixel whose
light was estimated by sampling runs to the full count**, so no lit surface,
shadow or reflection is ever cut short. That is where the saving comes from on
a typical frame, and why it does not cost accuracy where it would show.
``RenderResult.render_plan.path_samples_mean`` reports the samples per pixel a
render actually took (0 means the path tracer did not run). What adaptive
sampling does change is the *anti-aliasing* of 2-D edges, which is estimated
by jittering samples inside the pixel: raise ``pt_min_samples`` if a
text-heavy frame's edges look coarser than you want, or set
``pt_error_target = 0`` to restore uniform sampling and give every pixel
exactly ``samples_per_pixel``.

The denoiser only touches pixels whose light was *estimated*. Under adaptive
sampling (the default) the path tracer knows which pixels took a random
decision; every other pixel -- unlit 2-D content, the background -- is exact
and is passed through untouched, so text and vector graphics come out exactly
as rendered, and a frame with nothing to denoise costs nothing to denoise.
``pt_error_target = 0`` turns adaptive sampling off and with it this
pass-through: the filter then runs over the whole frame.

Noise is stable from frame to frame by default: static regions of a
path-traced animation get the same estimate every frame, so residual noise
reads as a fixed grain rather than a shimmer. Set
``SETTINGS.raytracing.experimental.pt_animated_seed = True`` to re-roll the
noise every frame instead.

``SETTINGS.raytracing.experimental.pt_seed`` changes the noise pattern without
changing what the render converges to -- useful for checking that a feature
you are looking at is real and not a shape in the noise. The path tracer does
not promise that two renders of the same scene produce identical frames; it
promises they converge to the same image.

.. _renderer-capabilities:

What each renderer supports
---------------------------

Raising ``samples_per_pixel`` is not a pure quality dial: it changes renderer.
The path tracer refuses nothing -- it is the fallback, so every feature the
deterministic renderer accepts renders there too -- but several are *reached*
differently, which the table below spells out. Algan still checks compatibility
before it allocates anything, and would refuse rather than silently drop a
feature it could not honour.

.. list-table::
   :header-rows: 1
   :widths: 38 31 31

   * - Feature
     - Deterministic (``spp == 1``)
     - Path tracer (``spp > 1``)
   * - Materials, textures, all light types
     - Yes
     - Yes
   * - Refractive materials (glass)
     - Yes
     - Yes
   * - Custom fragment-shader pipelines
     - Yes
     - Yes (shaded as authored; diffuse for indirect bounces; their light rows
       are sampled past the shadow cap)
   * - Custom scatter overrides
     - Yes
     - Yes (as a delta lobe; no NEE coverage)
   * - Environment maps
     - Yes (order-1 SH diffuse)
     - Yes (importance-sampled, full map)
   * - Analytic anti-aliasing
     - Yes
     - No -- jittered sub-pixel samples instead
   * - Global illumination, emissive surfaces as lights
     - No
     - Yes
   * - Denoising (``denoise``, default on)
     - Not applicable (noise-free)
     - Yes

A **custom scatter override** is a fragment pipeline that redefines how a ray
continues (``FragmentStage(..., scatter=...)``). Both renderers honour it. The
deterministic one *splits* into the reflected and transmitted branches the
scatter returns; the path tracer picks one of the three branches at random,
weighted by the branch weights, and continues along it as a **delta lobe** --
your direction, your density, weight 1, no MIS. The one consequence is that a
scatter surface is outside next-event estimation, so light reaches it only
through the sampled continuation and it converges more slowly than a
physically-integrated material in the same place.

No feature reaches this today, but the mechanism stands: were a scene to request
one a renderer could not honour, Algan raises
:class:`~algan.errors.UnsupportedFeatureError` naming the features rather than
dropping them. Either set ``samples_per_pixel`` back to ``1``, remove the
feature, or opt into the older behaviour explicitly:

.. code-block:: python

    SETTINGS.raytracing.set(unsupported_feature_policy="warn")    # or "ignore"

The default is ``"error"``. ``RenderResult.render_plan`` records which backend ran
and which features were requested, if you want to check programmatically.

This table covers only the split between the two renderers.
:doc:`renderer_limitations` is the complete list of what the renderer does not
do -- which objects are lit and shadowed, which texture maps are sampled, where
reflection and refraction stop, and the hard limits.

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

1. **How much of the frame the geometry fills.** ``render_tolerance_pixels`` on a
   :class:`~algan.mobs.surfaces.surface.Surface` (default ``0.5``) is how far a
   drawn triangle may sit from the true surface, in output pixels, so a surface
   filling the frame is diced far more finely than the same surface in the distance.
   The dice is decided per curved triangle, per frame and per direction: the part of
   a surface that is close to the camera, or turned edge-on, is refined without the
   rest of the mesh following it; a frame where the surface is small or off screen
   costs what that frame needs rather than what the closest frame of the batch needs;
   and a direction the surface is straight along costs nothing, so a long thin
   cylinder pays for its circumference and not for its length. Raising it slightly
   is a large speedup on close-up surfaces, and passing ``None`` removes the bound
   altogether.

   The budget is stated at a reference frame height of 1000 px and scaled down in
   proportion on anything shorter, because a low-resolution frame needs finer
   dicing than its pixel count alone suggests -- each of its pixels covers more of
   the object, and the antialiasing computes coverage from the microtriangles
   crossing a pixel. So the default is worth 0.5 px from 1080p up and 0.2 px at
   ``PREVIEW``, and halving it halves both.
2. **Resolution × anti-aliasing × frame count.** Straightforwardly multiplicative.
3. **Refraction.** Splits every ray in two, and routes the batch to the general
   wavefront tracer.
4. **Shadows**, multiplied by the number of lights.

   That multiplication is the deterministic renderer's, and it is why a scene
   with dozens of lights belongs on the path tracer instead. There, a lit
   surface point does not sum every light: it *chooses* one per shadow ray,
   and it chooses by descending a tree built over the emitters -- every point,
   spot and area-light cell and every emissive triangle -- that weighs how far
   away an emitter is and which way it faces as well as how bright it is. So
   the cost per shading point grows with the *logarithm* of the light count
   rather than with the count, and the shadow rays that do get fired are aimed
   at lights that can actually illuminate the point instead of at whichever
   one happens to be brightest. On a floor under 32 falling-off point lights
   that is roughly nine times less noise at the same
   ``samples_per_pixel``. Directional lights and an environment map have no
   position to sort by and are chosen by brightness alone, as before.
5. **Triangle count.** Imported models and high-resolution
   :class:`~algan.mobs.surfaces.surface.Surface` grids.
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

1. **Raise ``render_tolerance_pixels``** on the surfaces that fill the frame, for
   the reason above. This is the first thing to reach for whenever the scene
   contains a :class:`~algan.mobs.surfaces.surface.Surface` close to the camera,
   and it works at every resolution.
2. **Lower ``supersampling``** (``ssaa``) to 1. Cuts the pixel
   count fourfold.
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

The **render** device is a setting, so it belongs at the top of the script:

.. code-block:: python

    from algan import *

    SETTINGS.computing.set(render_device="cuda")   # or "auto", "mps", "cpu"

It defaults to ``auto``, which picks CUDA, then MPS, then CPU, and starts at
whatever ``ALGAN_RENDER_DEVICE`` said if you prefer the environment. It can be
changed between renders, at the cost of a fresh kernel-preparation pass whenever
the change crosses the CPU/GPU line -- see :doc:`settings` for that and for the
two changes that are refused rather than mishandled.

The pipeline's own PyTorch arithmetic -- everything between the ray-tracing
kernels -- runs through ``torch.compile`` by default wherever that is supported,
which fuses each chain of small operations into one kernel.
``SETTINGS.computing.set(torch_compile=False)`` turns it off; the first render
of a process pays the compile and every later one benefits. See :doc:`settings`.

The **animation** device is chosen at import time and cannot be changed
afterwards, so it is an environment variable set **before** ``import algan``:

.. code-block:: python

    import os
    os.environ["ALGAN_ANIMATION_DEVICE"] = "cpu"

    from algan import *

It defaults to ``cpu``, which is usually right: materializing the timeline is
Python-bound rather than compute-bound, and the geometry is moved to the render
device afterwards anyway.

:meth:`~algan.settings.abstract_settings.Settings.set` raises a specific error,
rather than a generic unknown setting error, if you try to set the animation
device at runtime.

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
    clear_cached_kernels()  # everything, including compiled kernels

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

- :doc:`renderer_limitations` -- the complete list of what the renderer cannot do.
- :doc:`the_render_daemon` -- the warm process that removes the fixed start-up
  cost, and how to control it.
- :doc:`settings` -- how the settings system works, including temporary overrides.
- :doc:`saving_videos_and_images` -- the quality presets, per render.
- :doc:`reflections_and_glass` -- the cost of bounces.
- :doc:`lighting_and_shadows` -- the cost of lights and shadows.
- :doc:`backgrounds_and_post_processing` -- anti-aliasing and bloom.
