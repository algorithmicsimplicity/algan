===============================
Backgrounds and Post-Processing
===============================

Two things happen either side of the renderer: the **background** is what fills
every pixel no geometry covers, and **post-processing** is what runs over the
finished frame before it is encoded.

Backgrounds
===========

A Scene's background can be a colour, an image, or a function. Set it for one
render by passing ``background_color`` to
:meth:`~algan.scene.Scene.save_video`, or for the Scene as a whole with
:meth:`~algan.scene.Scene.set_background_color`:

.. code-block:: python

    Scene.save_video("out", background_color=BLUE)     # this render only
    Scene.set_background_color(BLUE)                   # the whole Scene

.. note::

    The background is Scene state, not timeline state: it applies to the entire
    video rather than from a point onwards. It cannot be animated, but a
    procedural background can vary with time -- see below.

The default background is black.

A solid colour
--------------

Any Algan colour works, and arithmetic on colours is a convenient way to get a
dark tint:

.. code-block:: python

    Scene.save_video("out", background_color=Color([0.05, 0.05, 0.15]))

.. important::

    Be careful with ``BLUE * 0.15``. Colours have five components including
    opacity, so scaling one scales its alpha too -- and an alpha below 1 makes
    the whole output *transparent*, which changes the container to ``.mov``.
    Build a dark colour with an explicit :class:`~algan.constants.color.Color`, or use
    ``BLUE.set_opacity(1.0)`` after scaling.

An image
--------

Pass a path and the image is scaled to the frame:

.. algan:: BackgroundsImage

    from algan import *

    Circle(color=YELLOW, glow=0.4).scale(0.8).spawn()
    Scene.wait(1)

    Scene.save_video(background_color='world_map.png')

Paths resolve against the working directory and then your script's directory.

For a background that is *part of the 3-D world* -- visible in reflections and
refractions, and correct as the camera turns -- you want an environment map
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
        # x, y and t all broadcast together, giving one colour per pixel per frame.
        base = torch.zeros_like(t + y + x)
        return torch.cat([base + 0.35 * y + 0.05,
                          base + 0.10 * y + 0.02,
                          base + 0.30 * (1 - y) + 0.06,
                          base,
                          base + 1.0], -1)

    Circle(color=YELLOW).scale(0.8).spawn()
    Scene.wait(1)

    Scene.save_video(background_color=sunset)

The shape rules are the thing to get right:

* ``x`` has shape ``[1, W, 1]``, ``y`` has ``[H, 1, 1]`` and ``time`` has
  ``[frames, 1, 1, 1]``. Combining all three -- as ``torch.zeros_like(t + y + x)``
  does above -- gives a ``[frames, H, W, 1]`` base you can build channels on.
* Return the channels concatenated on the last axis. Five channels (red, green,
  blue, glow, opacity) is the safe choice; supply fewer and the missing ones are
  filled from the last channel you gave, which can make the background
  accidentally transparent.
* Alternatively return a single flat colour vector, which is treated as
  resolution-free and used everywhere.

Because ``time`` is passed in, a procedural background can animate -- a drifting
gradient, a pulse, a scrolling pattern -- even though the background itself is not
on the timeline.

For maximum speed a background callable can be a Taichi ``@ti.func`` instead, which
receives scalar normalized coordinates and a time and returns a colour vector; it is
evaluated for the whole render batch by one Taichi kernel writing directly into the
output buffer.

.. note::

    A procedural background is treated as **opaque**, because its alpha cannot be
    known without evaluating the render. For transparent output use a colour with
    alpha below 1 -- see :doc:`transparent_backgrounds`.

Post-Processing
===============

Post-processing passes run on each finished frame. The default is bloom (a soft
glow around bright areas), and you choose the set per render:

.. code-block:: python

    Scene.save_video("out")                     # default: bloom
    Scene.save_video("out", post_processes=())  # nothing
    Scene.save_frame("still", post_processes=())  # stills take the same argument

Bloom and glow
--------------

Bloom is what makes a Mob's
:ref:`glow <reference-mob-glow>` attribute visible. ``glow`` marks a
Mob as emitting light into nearby pixels; the bloom pass is what actually spreads it:

.. algan:: BackgroundsGlow

    from algan import *

    dot = Dot(color=YELLOW).scale(2).spawn()
    with Seq(run_time=2):
        dot.glow = 1.0
        dot.glow = 0.0

    Scene.save_video()

``glow`` can be set on the Mob or baked into a colour's fourth component, and it is
an ordinary animatable attribute, so it animates like anything else.

.. important::

    Glow is powerful and easy to overdo. Values around ``0.3``-``0.5`` read as "this
    is bright"; ``1.0`` on a large Mob will wash out most of the frame. If a scene
    looks like a blurry smear, the glow values are too high.

    Bloom also amplifies small numerical differences, so a scene with strong glow can
    show larger frame-to-frame variation between runs than an equivalent scene
    without it.

To customise it, pass the pass itself with different arguments:

.. code-block:: python

    from algan.rendering.post_processing.bloom import bloom_filter
    from functools import partial

    Scene.save_video("out", post_processes=(partial(bloom_filter, strength=8),))

``strength`` scales the effect, ``kernel_size`` and ``scale_factor`` trade quality
for speed, and ``glow_spread`` sets how far the glow reaches.

Bloom blurs the glow at two scales and sums them: a tight *rim* (``rim_frac``) that
hugs the source outline, and a wide, faint *tail* (``glow_spread``, defaulting to
10% of the frame height) weighted by ``tail_weight``. The tail is what makes a
single glowing Mob look luminous rather than outlined -- but it accumulates. In a
scene with many emitters the tails sum into a haze that fills the gaps between
them, and raising ``glow`` to compensate only deepens it. When you want glow that
is bright *and* local, narrow the tail and pay for it in strength:

.. code-block:: python

    tight = partial(bloom_filter, glow_spread=0.015, tail_weight=0.15, strength=60)
    Scene.save_video("out", post_processes=(tight,))

Anti-aliasing
-------------

Algan anti-aliases in two independent ways:

* **Supersampling** -- ``SETTINGS.video.anti_alias_level`` (default ``2``) renders
  at that multiple in each axis and downsamples. This is the main mechanism and the
  one that costs render time: level 2 is 4× the pixels.
* **Analytic AA** -- ``SETTINGS.raytracing.analytic_aa`` (on by default) computes
  edge coverage analytically inside the rasterizer, which gives clean edges much
  more cheaply than supersampling.
* **FXAA** -- ``SETTINGS.video.fxaa`` (off by default) is a cheap post-pass that
  smooths remaining edges. Useful when you have had to drop the supersampling level
  for speed.

.. code-block:: python

    SETTINGS.video.set(anti_alias_level=1, fxaa=True)   # fast draft
    SETTINGS.video.set(anti_alias_level=2, fxaa=False)  # the default

Tonemapping
-----------

The renderer composites in linear HDR, and bloom and downsampling happen in
linear light -- which is why a bright glow keeps its colour instead of clipping
to white. Encoding to the output frame is the last step.

**Tonemapping is off by default**, so an authored colour lands on the pixel it
names: ``WHITE`` renders as 255, ``RED`` renders as ``(255, 0, 0)``, and a
background you chose comes back byte for byte. Values above 1.0 are clipped.

.. code-block:: python

    SETTINGS.raytracing.set(tonemapping=True)        # off by default
    SETTINGS.raytracing.set(tonemap_method="agx")    # "neutral" (default) or "agx"
    SETTINGS.raytracing.set(tonemap_exposure=1.2)    # brighten the whole render

Turning tonemapping on applies a filmic curve, which buys highlight roll-off:
values above 1.0 stay distinguishable from one another instead of all clipping
to 255, which suits a scene carrying real HDR -- strong glow, bright speculars,
an environment map.

It costs a shift on *every* value, not just the ones above 1.0. The curve is
not the identity anywhere except at 0: mid-tones lose about 10/255, an authored
255 renders as 222, and saturated colours desaturate slightly. That is inherent
rather than a tuning problem -- a curve that is the identity on ``[0, 1]`` has
nowhere left to put anything above 1.0, so display white and highlight headroom
compete for the same top byte and you can have one or the other. Which is why
the default is off: most Algan frames are flat fills and text, where the
authored colour matters more than roll-off in the highlights.

``tonemap_exposure`` is the right control for "the whole scene is too dark" --
reach for it before you start raising every light's intensity. It applies
whether or not tonemapping is on.

Writing your own pass
---------------------

A post-process is a callable taking the frame batch and Algan's render arena, and
returning the frames modified. Pass any number of them and they run in order:

.. code-block:: python

    from algan.rendering.post_processing.bloom import bloom_filter

    def desaturate(frames, memory=None):
        grey = frames[..., :3].mean(-1, keepdim=True)
        frames[..., :3] = grey
        return frames

    Scene.save_video("out", post_processes=(bloom_filter, desaturate))

Frames arrive as a torch tensor on the render device.

The ``memory`` keyword is **required**, not optional: Algan always calls a pass as
``process(frames, memory=arena)``, so a pass declared as ``def desaturate(frames)``
raises ``TypeError`` part way through the render. Give it a default of ``None`` and
ignore it, as above, or accept ``**kwargs``. Use it when you want to allocate from
Algan's render arena rather than the torch allocator; see
:doc:`performance_and_quality`.

See Also
========

- :doc:`transparent_backgrounds` -- rendering with an alpha channel.
- :doc:`lighting_and_shadows` -- environment maps, the 3-D equivalent of a
  background image.
- :doc:`performance_and_quality` -- what anti-aliasing and bloom cost.
- :doc:`settings` -- the settings system these knobs live in.
