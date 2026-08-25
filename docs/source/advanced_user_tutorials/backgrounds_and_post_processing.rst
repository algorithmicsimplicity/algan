===============================
Backgrounds and Post-Processing
===============================

Apart from the ray-traced render itself, there are two
important steps which shape how your output will look:
* **Backgrounds:** What fills empty pixels where no geometry is present.
* **Post-processing:** Image filters and effects (like bloom, anti-aliasing, and
  tonemapping) that run over rendered frames before video encoding.

Backgrounds
===========

A Scene's background can be a solid color, an image, or a procedural function.
Set it for one
render by passing ``background_color`` to
:meth:`~algan.scene.Scene.save_video`, or for the Scene as a whole with
:meth:`~algan.scene.Scene.set_background_color`:

.. code-block:: python

    Scene.save_video("my_video", background_color=BLUE)   # For this render only
    Scene.set_background_color(BLUE)                      # Across the whole Scene

.. note::

    The background is Scene state, not timeline state: it applies to the entire
    video rather than from a point onwards. It cannot be animated, but a
    procedural background can vary with time -- see below.

The default background is black.

A solid color
-------------

Any Algan color works, and arithmetic on colors is a convenient way to get a
dark tint:

.. code-block:: python

    Scene.save_video("my_video", background_color=Color([0.05, 0.05, 0.15]))

.. important::
    Be careful with ``BLUE * 0.15``. Colors have five components including
    opacity, so scaling one scales its alpha too, and an alpha below 1 makes
    the whole output *transparent*, which changes the container to ``.mov``.
    Build a dark color with an explicit :class:`~algan.constants.color.Color`, or use
    ``(BLUE * 0.15).set_opacity(1.0)`` after scaling.

An image
--------

Pass a path and the image is scaled to the frame:

.. algan:: BackgroundsImage

    from algan import *

    Circle(color=YELLOW, glow=0.4).scale(0.8).spawn()
    Scene.wait(1)

    Scene.save_video(background_color='world_map.png')

Paths resolve against the working directory and then your script's directory.

For a background that is *part of the 3-D world* (i.e. visible in reflections and
refractions, and correct as the camera turns) you want an environment map
instead, not a background image. See :doc:`lighting_and_shadows`.

A procedural background
-----------------------

Pass a callable ``(x, y, time) -> color`` for a background computed per pixel per
frame. ``x`` and ``y`` are normalized screen coordinates and ``time`` is in
seconds; all three arrive as broadcastable torch tensors:

.. algan:: BackgroundsProcedural

    from algan import *
    import torch

    def sunset(x, y, t):
        # x, y and t broadcast together, creating a [frames, H, W, 1] tensor base
        base = torch.zeros_like(t + y + x)
        return torch.cat([base + 0.35 * y + 0.05,    # Red
                          base + 0.10 * y + 0.02,    # Green
                          base + 0.30 * (1 - y) + 0.06, # Blue
                          base,                      # Glow
                          base + 1.0], -1)           # Opacity

    Circle(color=YELLOW).scale(0.8).spawn()
    Scene.wait(1)

    Scene.save_video(background_color=sunset)

Because ``time`` is passed in, a procedural background can animate (e.g. a drifting
gradient, a pulse, a scrolling pattern) even though the background itself is not
on the animation timeline.

For maximum speed a background callable can be a Taichi ``@ti.func`` instead, which
receives scalar normalized coordinates and a time and returns a color vector; it is
evaluated for the whole render batch by one Taichi kernel writing directly into the
output buffer.

.. note::

    A procedural background is treated as **opaque**, because its alpha cannot be
    known without evaluating the render. For transparent output use a color with
    alpha below 1 -- see :doc:`transparent_backgrounds`.

Post-Processing
===============

Post-processing passes run on each finished frame. The default is bloom (a soft
glow around bright areas), and you choose the set per render:

.. code-block:: python

    Scene.save_video("my_video")                        # Default: bloom enabled
    Scene.save_video("my_video", post_processes=())     # Disable all post-processing

Bloom and Glow
--------------

Bloom is what makes a Mob's :ref:`glow <reference-mob-glow>` attribute light up
surrounding pixels:

.. algan:: BackgroundsGlow

    from algan import *

    dot = Dot(color=YELLOW).scale(2).spawn()
    with Seq(run_time=2):
        dot.glow = 1.0
        dot.glow = 0.0

    Scene.save_video()

.. tip::

    A little glow goes a long way. Values between ``0.3`` and ``0.5`` give a nice
    luminous shine; ``1.0`` on large objects will wash out the frame.

To customise the bloom filter, pass the pass itself with different arguments:

.. code-block:: python

    from algan.rendering.post_processing.bloom import bloom_filter
    from functools import partial

    Scene.save_video("my_video", post_processes=(partial(bloom_filter, strength=8),))

``strength`` scales the effect, ``kernel_size`` and ``scale_factor`` trade quality
for speed, and ``glow_spread`` sets how far the glow reaches.

Bloom blurs the glow at two scales and sums them: a tight *rim* (``rim_frac``) that
hugs the source outline, and a wide, faint *tail* (``glow_spread``, defaulting to
10% of the frame height) weighted by ``tail_weight``. The tail is what makes a
single glowing Mob look luminous rather than outlined. In a
scene with many emitters the tails sum into a haze that fills the gaps between
them, and raising ``glow`` to compensate only deepens it. When you want glow that
is bright *and* local, narrow the tail and pay for it in strength:

.. code-block:: python

    tight = partial(bloom_filter, glow_spread=0.015, tail_weight=0.15, strength=60)
    Scene.save_video("out", post_processes=(tight,))

Anti-aliasing
-------------

Algan supports three anti-aliasing techniques:

* **Supersampling (SSAA):** ``SETTINGS.video.anti_alias_level`` (default ``2``)
  renders at 2x resolution and downsamples.
* **Analytic AA:** ``SETTINGS.raytracing.analytic_aa`` (on by default) resolves
  vector edges analytically inside the rasterizer with almost zero overhead.
**FXAA** -- ``SETTINGS.video.fxaa`` (off by default) is a cheap post-pass that
  smooths remaining edges. Useful when you have had to drop the supersampling level
  for speed.

.. code-block:: python

    SETTINGS.video.set(anti_alias_level=1, fxaa=True)   # Fast draft mode
    SETTINGS.video.set(anti_alias_level=2, fxaa=False)  # High quality default

Tonemapping
-----------

By default, tonemapping is off so authored RGB values remain exact (e.g.
``WHITE`` is exactly ``255``).

If your scene features high-dynamic-range lighting (bright specular highlights,
intense glow, environment maps), enable filmic tonemapping to smoothly roll off
highlights:

.. code-block:: python

    SETTINGS.raytracing.set(tonemapping=True)
    SETTINGS.raytracing.set(tonemap_method="agx")     # "neutral" or "agx"
    SETTINGS.raytracing.set(tonemap_exposure=1.2)

Custom Post-Processing Passes
-----------------------------

A custom post-processing pass is any callable that takes a batch of frames and
returns modified frames:

.. algan:: BackgroundsCustomPass

    from algan import *
    from algan.rendering.post_processing.bloom import bloom_filter

    def desaturate(frames, memory=None):
        # Convert RGB channels to grayscale
        grey = frames[..., :3].mean(-1, keepdim=True)
        frames[..., :3] = grey
        return frames

    square = Square(color=RED).scale(1.5).spawn()
    square.rotate(180, OUT)

    Scene.save_video(post_processes=(bloom_filter, desaturate))

Frames arrive as a torch tensor on the render device.

The ``memory`` keyword is **required**, not optional: Algan always calls a pass as
``process(frames, memory=arena)``, so a pass declared as ``def desaturate(frames)``
raises ``TypeError`` part way through the render. Give it a default of ``None`` and
ignore it, as above, or accept ``**kwargs``. Use it when you want to allocate from
Algan's render arena rather than the torch allocator -- see
:doc:`performance_and_quality`.

See Also
========

- :doc:`transparent_backgrounds` -- rendering with an alpha channel.
- :doc:`lighting_and_shadows` -- environment maps, the 3-D equivalent of a
  background image.
- :doc:`images_and_textures` -- painting an image onto geometry rather than
  behind it.
- :doc:`performance_and_quality` -- what anti-aliasing and bloom cost.
- :doc:`renderer_limitations` -- what analytic anti-aliasing does and does not
  resolve.
- :doc:`settings` -- the settings system these knobs live in.
- :doc:`saving_videos_and_images` -- passing ``background_color`` and
  ``post_processes`` per render.
