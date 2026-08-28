========
Settings
========

Algan exposes one process-global settings root named
:ref:`algan.SETTINGS <reference-settings>`.
Its sections are grouped by lifecycle and responsibility:

``SETTINGS.video``
    Default resolution, frame rate, anti-aliasing, and related output settings.

``SETTINGS.style``
    Default colors, frame/background behavior, shaders, and scene-end style.

``SETTINGS.paths``
    Output and content-cache paths.

``SETTINGS.computing``
    Runtime-adjustable memory budgets and authoring controls.

``SETTINGS.raytracing``
    What the renderer produces: sampling, bounces, shadows, lighting and
    tonemapping. Internal performance switches live on
    ``SETTINGS.raytracing.experimental``.

Mutating live settings
======================

Section objects keep stable identity. Modify them in place with ``set`` rather
than assigning a replacement object:

.. code-block:: python

    SETTINGS.video.set(frames_per_second=60)
    SETTINGS.paths.set(output_directory="renders")
    SETTINGS.raytracing.set(samples_per_pixel=4)

This is valid:

.. code-block:: python

    SETTINGS.video.set(HD)

It copies every field from the ``HD`` preset into the live video settings
section. Keyword arguments may override copied fields:

.. code-block:: python

    SETTINGS.video.set(HD, frames_per_second=60)

This is intentionally invalid because replacing a section would break code that
retains a reference to it:

.. code-block:: python

    SETTINGS.video = HD  # raises AlganConfigurationError

Supported settings versus experimental switches
===============================================

``SETTINGS.raytracing`` exposes the settings that change what a render *looks
like*:

.. code-block:: python

    SETTINGS.raytracing.set(samples_per_pixel=4)   # path-traced quality
    SETTINGS.raytracing.set(max_bounces=6)
    SETTINGS.raytracing.set(shadows=True)
    SETTINGS.raytracing.set(tonemap_exposure=1.2)

The renderer also carries a large number of performance and capability
switches — kernel fusion, rasterization gates, memory ratios, and so on. Those
are real, but their names and behaviour follow the current kernels and can
change between releases, so they live behind ``experimental``:

.. code-block:: python

    SETTINGS.raytracing.experimental.set(hybrid_raster=False)

Setting one of them on ``SETTINGS.raytracing`` directly raises an error telling
you where it lives. ``dir(SETTINGS.raytracing)`` lists only the supported
settings, and ``dir(SETTINGS.raytracing.experimental)`` lists the rest.

Immutable presets
=================

Built-in video constants such as ``PREVIEW`` and ``HD`` are immutable presets.
Calling ``set`` on a preset returns a modified preset copy:

.. code-block:: python

    portrait_hd = HD.set(resolution=(1080, 1920))

    assert portrait_hd.resolution == (1080, 1920)
    assert HD.resolution == (1920, 1080)

The same distinction applies to captured ray-tracing presets. Mutable sections
change in place; presets return new values.

Temporary overrides
===================

Use a section override for one section:

.. code-block:: python

    with SETTINGS.video.override(frames_per_second=12):
        Scene.save_video("draft.mp4")

Use the root override for multiple sections:

.. code-block:: python

    with SETTINGS.override(
        video={"resolution": (640, 360), "frames_per_second": 12},
        raytracing={"samples_per_pixel": 1},
    ):
        Scene.save_video("fast_preview.mp4")

Both forms restore the previous settings even when the body raises an exception.
For longer-lived save/restore flows, use ``snapshot = SETTINGS.snapshot()`` and
``SETTINGS.restore(snapshot)``.

Per-render settings
===================

A Scene captures its current video settings when it is constructed. You can
also provide settings explicitly:

.. code-block:: python

    with Scene(video_settings=PREVIEW) as scene:
        Square().spawn()
        scene.save_video("preview.mp4", HD)

The override applies to that render only; afterwards the Scene returns to its
own settings. ``save_frame`` takes the same argument and likewise restores
every derived render value after the still has been written.

This per-render form is usually what you want, because it has no ordering
constraints. ``SETTINGS.video`` is read when a Scene is *constructed*, and
Algan creates its default Scene as soon as you build your first Mob, so a
``SETTINGS.video.set(...)`` placed after that point will not affect the Scene
that is already running. Either set it at the top of your script, or pass the
settings to ``save_video``.

Style defaults
==============

``SETTINGS.style`` holds the defaults a Scene picks up when you do not say
otherwise. Unlike the other sections it is mostly about *authoring*, so its
fields are the ones you set once at the top of a script:

``background_color``
    The color behind everything, as an Algan
    :class:`~algan.constants.color.Color`. Defaults to ``BLACK``. This is the
    process-wide default; ``Scene.set_background_color(...)`` changes one Scene
    and the ``background_color=`` argument to
    :meth:`~algan.scene.Scene.save_video` changes one render, each overriding
    the one before it. It also accepts an image path or ``TRANSPARENT`` -- see
    :doc:`backgrounds_and_post_processing` and
    :doc:`transparent_backgrounds`.

``frame``
    Color of the letterbox area outside the rendered frame, when the output
    aspect ratio does not fill the canvas. Defaults to ``BLACK``.

``text_color``
    Default color for :class:`~algan.mobs.text.Text` and
    :class:`~algan.mobs.text.Tex`. Defaults to ``WHITE``.

``buffer``
    Default gap, in world units, left by the layout methods
    (``move_next_to``, ``arrange_in_line``, ``arrange_in_grid``). Defaults to
    ``0.6``. Must be finite and non-negative.

``fade_out_on_scene_end``
    Whether a render fades everything out at the end. Defaults to ``False``.
    ``save_video(animate_fade_out=...)`` overrides it per render.

``default_material``
    The material a 3-D Mob is shaded with when it sets none of its own.
    Defaults to :class:`~algan.rendering.shaders.materials.DiffuseMaterial`
    (Lambert diffuse), installed when Algan is imported;
    :meth:`~algan.scene.Scene.use_manim_defaults` replaces it with
    ``ManimMaterial``, so imported 3-D geometry shades the way Manim shades
    it. Must be a ``Material`` instance -- a value without a ``.shader``
    attribute is rejected -- and flat 2-D content never consults it. See
    :doc:`shaders_and_materials`.

``shape_style_profile``
    Whose per-shape styling defaults the built-in shapes adopt: ``"algan"``
    (the default) keeps Algan's own -- a red filled ``Square`` with a wide
    white border, say -- while ``"manim"`` asks them to adopt Manim
    Community's constructor defaults instead (an unfilled white ``Square``
    outline of stroke width 4). Enabling it reads each shape's defaults out
    of the installed ``manim`` package once, so the values follow whatever
    version you have; a shape Manim does not define simply keeps Algan's
    default. An explicit keyword always wins over the profile:
    ``Square(color=BLUE)`` is blue either way.

.. code-block:: python

    SETTINGS.style.set(shape_style_profile="manim")  # Manim-looking shapes
    SETTINGS.style.set(shape_style_profile="algan")  # back to Algan's

.. code-block:: python

    SETTINGS.style.set(background_color=Color([0.05, 0.05, 0.15]), buffer=0.3)

Choosing the render device
==========================

``SETTINGS.computing.render_device`` is where a render's primitives are built
and the ray tracer runs. It accepts a device string, a ``torch.device``, or
``'auto'`` (CUDA, then MPS, then CPU), and it starts at whatever
``ALGAN_RENDER_DEVICE`` said -- also ``auto`` by default.

.. code-block:: python

    from algan import *

    SETTINGS.computing.set(render_device="cpu")   # or "cuda", "cuda:1", "auto"

    Square(color=RED).spawn()
    Scene.save_video("on_the_cpu")

Set it **at the top of the script**, before creating any Mob. It can be changed
between renders, but the change is not free: Taichi's compute backend is chosen
from it, so crossing the CPU/GPU line discards every kernel compiled so far and
the next render pays a fresh kernel-preparation pass.

Two changes are refused rather than silently mishandled:

* **While a render is running.** Batch preparation launches kernels on a worker
  thread, so a change mid-render could pull the backend out from under it.
* **Once a textured Mob exists.** A texture is wide enough that its per-frame
  window is allocated on the render device when the Mob is created, and nothing
  re-asks afterwards. Choose the device before creating one.

Initialization-only configuration
=================================

Some configuration affects Torch/Taichi process initialization and therefore
must be set through environment variables **before importing Algan**. It has no
runtime Python settings object.

Supported initialization variables include:

``ALGAN_ANIMATION_DEVICE``
    Torch device used for animation materialization. The default is ``cpu``.
    Unlike the render device this one cannot change: every Mob's authoring
    state is allocated on it from the first one onward.

``ALGAN_HOME`` and ``ALGAN_CACHE_DIR``
    Base and content-cache locations.

``TI_OFFLINE_CACHE_FILE_PATH``
    Taichi offline kernel-cache location.

``ALGAN_SOFT_SHADOW_SAMPLES`` and ``ALGAN_HDR_BUFFER_F16``
    Values baked into renderer runtime layout or kernels.

Example:

.. code-block:: python

    import os

    os.environ["ALGAN_ANIMATION_DEVICE"] = "cpu"

    from algan import *

Changing these variables after ``import algan`` does not reinitialize the
process. ``ALGAN_RENDER_DEVICE`` is not among them: it only supplies the
starting value of ``SETTINGS.computing.render_device``, which is settable.

See Also
========

* :doc:`performance_and_quality` -- which of these settings to reach for, and what
  each one costs.
* :doc:`saving_videos_and_images` -- the per-render form, which is usually what
  you want.
* :doc:`backgrounds_and_post_processing` -- the anti-aliasing and tonemapping
  settings in context.
* :doc:`lighting_and_shadows` -- ``SETTINGS.raytracing.shadows``.
* :doc:`the_render_daemon` -- why a warm daemon refuses a script that wants
  different initialization-only or import-time values.
* :doc:`multi_scene_projects` -- ``video_settings`` on a whole project, instead
  of a global mutation at the top of the file.
