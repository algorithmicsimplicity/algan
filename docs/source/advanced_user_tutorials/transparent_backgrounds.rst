=======================
Transparent Backgrounds
=======================

Algan can produce output video files with an alpha channel (opacity/transparency),
all you need to do is 1) make sure the output file format supports transparency (e.g. .mov, .webm)
and 2) make sure the background color has an alpha value (opacity) less than 1.

Example with fully transparent background:

.. code-block:: python

    from algan import *

    mob = Square().spawn()

    render_to_file(file_extension='mov', background_color=TRANSPARENT)

Example with 50% transparent background:

.. code-block:: python

    from algan import *

    mob = Square().spawn()

    render_to_file(file_extension='mov', background_color=RED.set_opacity(0.5))
