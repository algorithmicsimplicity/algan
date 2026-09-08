========================
Saving Videos and Images
========================

Algan provides two main methods for rendering your scenes: :meth:`~algan.scene.Scene.save_video`
for videos and :meth:`~algan.scene.Scene.save_frame` for one or multiple frames (images).
Both of them use the same protocol for deciding where the output is produced,
and both can be passed :class:`~.VideoSettings` as a parameter.
``Scene.save_frame`` additionally takes an ``at`` parameter which is used
to specify the set of time stamps to render frames at. ``at`` defaults to None
which means the current point in time when ``Scene.save_frame`` is called.

.. note::

    Neither method alters your scene, so you can call them anywhere in your
    script and as many times as you like -- including from inside a ``with``
    block that has not finished yet, which renders everything recorded so far.
    ``save_video`` takes ``reset=True`` if you *do* want the old destructive
    behaviour: despawn everything and rebuild the timeline.

Changing the video settings
===========================

By default, videos are rendered at low quality: 864x486 at 15 frames per
second. That keeps your edit-and-preview loop fast. When you want a
better-looking render, pass a quality preset as the second parameter:

.. algan:: SavingHelloWorldHD

    from algan import *

    text = Text('Hello World!', font_size=100)
    text.spawn()

    Scene.save_video("my_video", HD)

Algan provides the following built-in presets:

.. list-table::
   :header-rows: 1
   :widths: 20 25 15

   * - Preset
     - Resolution
     - Frame rate
   * - ``SMOKE_TEST``
     - 32 x 32
     - 2
   * - ``PREVIEW``
     - 704 x 396
     - 10
   * - ``LD`` (the default)
     - 864 x 486
     - 15
   * - ``MD``
     - 1280 x 720
     - 30
   * - ``HD``
     - 1920 x 1080
     - 30
   * - ``PRODUCTION``
     - 2560 x 1440
     - 60
   * - ``UHD``
     - 3840 x 2160
     - 60
   * - ``THUMBNAIL``
     - 1280 x 720
     - 1

The preset applies to that render only. If you want to change the default for
every render in your script, set it once on the global
:doc:`SETTINGS <../reference_index/settings>` object instead:

.. code-block:: python

    from algan import *

    SETTINGS.video.set(HD)

Presets are immutable, so you can safely build variations from them without
disturbing the original:

.. code-block:: python

    HD_60 = HD.set(frames_per_second=60)

See :meth:`~algan.scene.Scene.save_video` for the full list of parameters,
:class:`~.VideoSettings` for building custom settings from scratch, and
:doc:`settings` for everything else you can configure.
:doc:`performance_and_quality` says which of these knobs is worth reaching for
when a render is too slow.

Where the output file goes
==========================

By default, the file is saved into an ``algan_outputs`` directory next to your
script file, as an ``.mp4`` for videos (or ``.mov`` if your background is
transparent) and a ``.png`` for frames. This can be changed by adding paths or
extensions to the name parameter.

.. code-block:: python

    Scene.save_video("my_video")                    # algan_outputs/my_video.mp4
    Scene.save_video("my_video.mov")                # algan_outputs/my_video.mov
    Scene.save_video("renders/final.mp4")           # renders/final.mp4
    Scene.save_video("/tmp/absolute.mp4")           # exactly where you said
    Scene.save_frame("shot")                        # shot.png at t=current_time
    Scene.save_frame("shot.jpg", at=-0.5)           # shot.jpg at t=current_time - 0.5s
    Scene.save_frame("shot.png", at=[0.5, 1, 1.5])  # shot_0.5.png, shot_1.png, shot_1.5.png

A bare name goes to the output directory; anything with a directory in it is
used exactly as written. Passing a sequence to ``at`` appends each time stamp to
the stem and returns a list of results, one per frame; a single ``at`` (or none)
writes the one file you named and returns one result.

:meth:`~algan.scene.Scene.save_video` and :meth:`~algan.scene.Scene.save_frame` return a
small result object describing what it did, which is handy in scripts:

.. code-block:: python

    result = Scene.save_video("my_video")
    print(result.output_path, result.walltime_seconds)

Choosing the video encoder
==========================

By default videos are encoded with FFmpeg's ``libx264`` at high quality
(``-crf 17 -preset slower``), which is CPU work. On machines with an NVIDIA
GPU whose driver exposes the NVENC encoder engine, Algan instead encodes with
``h264_nvenc``, leaving the CPU to the renderer; on everything else it keeps
the software encoder. The choice is automatic and silent: once per process,
Algan looks for an FFmpeg binary that can actually drive NVENC -- first the
binary named by the ``FFMPEG_BINARY`` environment variable (if set), then the
one moviepy is configured with, then ``ffmpeg`` on the ``PATH`` -- checks each
for the encoder and a working test encode, and encodes with the first that
passes, running that binary for the encode. This matters because moviepy is
often configured with a stripped-down static FFmpeg build that has no NVENC
encoders even on machines where the system's ffmpeg can use them. When no
candidate qualifies, videos fall back to software encoding on moviepy's own
binary.

You can pin the choice yourself by setting the ``ALGAN_VIDEO_ENCODER``
environment variable to ``software`` (always ``libx264``, today's exact
behaviour), ``nvenc`` (always ``h264_nvenc``), or ``auto`` (the default).
The automatic choice only applies when you have not passed an explicit
``codec``; if your ``ffmpeg_params`` carry x264 rate-control flags
(``-preset`` / ``-crf``) they are honoured by staying on ``libx264``.

.. code-block:: bash

    # Keep encoding on the CPU even when an NVENC encoder is available.
    ALGAN_VIDEO_ENCODER=software python my_scene.py

Exporting additive glow and coverage together
==============================================

Enable :meth:`~.Scene.set_premultiplied_over` to put opaque geometry,
translucent geometry and additive glow in one transparent clip:

.. code-block:: python

    from algan import *

    Scene.set_background(TRANSPARENT)
    Scene.set_premultiplied_over()
    Circle(color=YELLOW, glow=1.0).spawn()
    Scene.save_video("glow.mov")

The equivalent constructor keyword is ``Scene(premultiplied_over=True)``.
The setting defaults to False, belongs to one Scene, survives ``reset()``, and
is ignored for opaque backgrounds. It also applies to ``save_frame`` and
``get_frames``. Ordinary image/video viewers generally do not display this
export correctly; use it as a compositing intermediate.

Bloom adds to RGB and leaves coverage alpha untouched. A halo can therefore
contain RGB even where alpha is zero. After decoding RGB from sRGB to linear
light, combine the clip with a linear-light backdrop using:

.. math::

    C_{out} = C_{clip} + (1 - \alpha_{clip}) C_{backdrop}.

For transparent backdrops, the output alpha is
``alpha_clip + (1 - alpha_clip) * alpha_backdrop``. Opaque geometry replaces
the backdrop; a zero-alpha halo adds light without obscuring it.

This mode requires ``SETTINGS.raytracing.linear_color_space=True``,
``tonemapping=False``, and
``SETTINGS.raytracing.experimental.post_process_tonemap=True`` (all defaults).
Incompatible settings raise an error before rendering. Exposure scales the
clip's light only; apply a shared exposure or a nonlinear tone curve after
compositing if it should affect the backdrop too.

The standard bloom pass honors the mode whether it is the default, explicitly
passed as ``post_processes=(bloom_filter,)``, or tuned with
``functools.partial``. Custom passes must preserve coverage alpha and linear
premultiplied RGB themselves. Render with ``TRANSPARENT`` for foreground-only
bloom; any authored partially transparent background is part of the exported
layer and can contribute to its bloom.

Precision and parity
--------------------

``.mov`` defaults to ProRes 4444 in this mode, and to lossless PNG otherwise.
``.mkv`` and ``.avi`` retain PNG; ``.webm`` retains VP9 with alpha. An explicit
``codec=`` wins. The built-in ProRes profile and pixel format remain in place
when adding custom ``ffmpeg_params``; later caller options can override them.

RGB and alpha are still quantized to **8 bits before encoding**. ProRes is an
editor-compatible container codec here, not an increase in source precision.
RGB above linear 1 is clipped, and lossy codecs can introduce additional error.
PNG output preserves the samples, including RGB at zero alpha, but these
samples require the same custom interpretation as the video.

The export contains bloom from its own layer. It does not reproduce every
opaque Algan render over arbitrary footage: the normal opaque bloom pass also
blurs backdrop light visible through translucent geometry and antialiased
edges. Nonlinear tonemapping and content-dependent antialiasing likewise do
not commute with compositing. The guarantee is the linear over equation for
the exported layer, not universal parity with backdrop-dependent rendering.
Colored transmission/refraction of an external plate also cannot be represented
by a single coverage alpha.

Resolve / Fusion processing contract
------------------------------------

The stored RGB is ``sRGB(linear premultiplied RGB)``. It is **not**
``alpha * sRGB(straight RGB)``. Merely choosing an alpha-mode label is not
sufficient to guarantee the correct color processing. The required order is:

1. Load the RGB and alpha without discarding RGB at zero alpha or multiplying
   RGB by alpha.
2. Decode the clip's RGB directly from sRGB to linear Rec.709, leaving alpha
   untouched. Do not surround this decoding step with alpha divide/multiply
   operations (including a color node's Pre-Divide/Post-Multiply option).
3. Convert the backdrop to the same linear working space using its own input
   color interpretation.
4. Apply premultiplied over. In Fusion this is a Normal/Over Merge with its
   Additive/Subtractive control at Additive. Do not use an Add blend mode:
   that would lose the geometry's occlusion of the backdrop.
5. Apply the desired display transform to the combined result.

Resolve's import and color-management behavior varies with the project setup.
This processing contract is tested numerically and through FFmpeg, but the
Resolve UI chain has not been validated interactively. Before production use,
test a clip containing an opaque patch, a partial-alpha edge and a nonzero-RGB,
zero-alpha patch over a mid-grey and a bright colored plate. The opaque patch
must replace the plate, and the zero-alpha patch must add its decoded light.
Check the partial-alpha edge numerically too: preserving the halo alone does
not prove the input color transform is correct.

Working with projects
=====================

A single ``.py`` file with one ``Scene.save_video()`` at the end is great for
quickly making a short video. But when working on longer videos with
multiple scenes, you want to be able to chunk up the rendering correspondingly --
re-render scene 7 without touching the other eleven, keep the output names
stable, and stitch the result together at the end.

:class:`~algan.project.Project` is the layer that does that. It owns a list of
zero-parameter scene functions, their identifiers, their output directories and
the concatenation into the complete video, and it can render any subset of them
from the command line. See :doc:`multi_scene_projects`.

See Also
========

* :meth:`~algan.scene.Scene.save_video` and
  :meth:`~algan.scene.Scene.save_frame` -- the full parameter lists.
* :doc:`settings` -- how ``video_settings`` and the output paths resolve, and
  how to override settings temporarily.
* :doc:`performance_and_quality` -- what each quality setting costs.
* :doc:`multi_scene_projects` -- rendering a video made of many scenes.
* :doc:`transparent_backgrounds` -- rendering with an alpha channel, and why
  that changes the container.
* :doc:`backgrounds_and_post_processing` -- ``background`` and
  ``post_processes``.
