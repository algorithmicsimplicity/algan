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
    Live renderer and ray-tracing feature configuration.

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
        scene.save_video("preview.mp4", video_settings=HD)

The ``save_video`` override applies to that render. After rendering, the Scene
is reset to its previous settings. ``save_frame`` likewise restores every
derived render value after the still has been written.

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
