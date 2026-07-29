========
Settings
========

Algan exposes one process-global settings root named :data:`algan.SETTINGS`.
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

Initialization-only configuration
=================================

Some configuration affects Torch/Taichi process initialization and therefore
must be set through environment variables **before importing Algan**. It has no
runtime Python settings object.

Supported initialization variables include:

``ALGAN_ANIMATION_DEVICE``
    Torch device used for animation materialization. The default is ``cpu``.

``ALGAN_RENDER_DEVICE``
    Render device, or ``auto`` to select CUDA, MPS, then CPU.

``ALGAN_HOME`` and ``ALGAN_CACHE_DIR``
    Base and content-cache locations.

``TI_OFFLINE_CACHE_FILE_PATH``
    Taichi offline kernel-cache location.

``ALGAN_SOFT_SHADOW_SAMPLES`` and ``ALGAN_HDR_BUFFER_F16``
    Values baked into renderer runtime layout or kernels.

Example:

.. code-block:: python

    import os

    os.environ["ALGAN_RENDER_DEVICE"] = "cuda"
    os.environ["ALGAN_ANIMATION_DEVICE"] = "cpu"

    from algan import *

Changing these variables after ``import algan`` does not reinitialize the
process.
