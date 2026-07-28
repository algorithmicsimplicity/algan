=======================
Transparent Backgrounds
=======================

Algan can write video with an alpha channel when both the background and output
container support transparency. Use a ``.mov`` or ``.webm`` path and pass a
background whose opacity is below one.

Fully transparent output:

.. code-block:: python

    from algan import *

    with Scene() as scene:
        Square().spawn()
        scene.save_video("transparent.mov", background_color=TRANSPARENT)

Partially transparent output:

.. code-block:: python

    from algan import *

    with Scene() as scene:
        Square().spawn()
        scene.save_video(
            "red_overlay.mov",
            background_color=RED.set_opacity(0.5),
        )

If the path has no extension, ``save_video`` chooses ``.mov`` automatically for
transparent output. An explicit ``.mp4`` path is rejected because MP4 does not
support Algan's alpha-channel output.

The Scene determines transparency from its final background tensor. Procedural
background callables are treated as opaque because their alpha cannot be known
without evaluating the render.
