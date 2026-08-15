=======================
Transparent Backgrounds
=======================

Algan can write video with an alpha channel, so a render can be composited over
something else -- a slide, a screen recording, a video editor timeline. It needs two
things: a background whose opacity is below one, and an output container that
supports transparency.

Fully transparent output
========================

.. code-block:: python

    from algan import *

    with Scene() as scene:
        Square().spawn()
        scene.save_video("transparent.mov", background_color=TRANSPARENT)

Partially transparent output
============================

Any colour with an opacity below one works, so you can render over a tint rather
than over nothing:

.. code-block:: python

    from algan import *

    with Scene() as scene:
        Square().spawn()
        scene.save_video(
            "red_overlay.mov",
            background_color=RED.set_opacity(0.5),
        )

Containers
==========

Use a ``.mov`` path. If the path has no extension at all, ``save_video`` chooses
``.mov`` automatically when the background is transparent. An explicit ``.mp4``
path is **rejected**, because MP4 does not support Algan's alpha-channel output --
you get an error rather than a silently opaque video.

.. code-block:: python

    scene.save_video("out.mov", background_color=TRANSPARENT)   # explicit
    scene.save_video("out", background_color=TRANSPARENT)       # -> out.mov
    scene.save_video("out.mp4", background_color=TRANSPARENT)   # error

.. warning::

    Do not pass a bare ``.webm`` path. Algan's default codec for transparent
    output is ``png``, which ffmpeg cannot put in a WebM container: the render
    reports success and writes a file that will not play. WebM needs its codec
    stated explicitly.

.. code-block:: python

    scene.save_video(
        "out.webm",
        background_color=TRANSPARENT,
        codec="libvpx-vp9",
        ffmpeg_params=["-pix_fmt", "yuva420p"],
    )

Codec and encoder options can be overridden through ``save_video``'s ``codec``,
``audio_codec`` and ``ffmpeg_params`` arguments if your compositing tool wants
something specific.

How transparency is decided
===========================

The Scene determines transparency from its final background tensor -- that is, from
the alpha of whatever colour the background resolves to.

.. important::

    Procedural background callables are always treated as **opaque**, because their
    alpha cannot be known without evaluating the render. If you want a transparent
    output, use a colour with alpha below one; do not try to return a low alpha from
    a background function.

    Watch out for colour arithmetic too: an Algan
    :class:`~algan.constants.color.Color` includes its
    opacity, so ``BLUE * 0.5`` halves the alpha as well as the brightness and will
    silently give you transparent output. Use
    :ref:`Color.set_opacity <reference-color-set-opacity>` to say what you mean.

See Also
========

* :doc:`backgrounds_and_post_processing` -- opaque colours, images and procedural
  backgrounds.
* :doc:`lighting_and_shadows` -- environment maps, for a background that is part of
  the 3-D world.
