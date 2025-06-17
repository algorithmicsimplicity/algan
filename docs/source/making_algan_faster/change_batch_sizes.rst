====================
Changing Batch Sizes
====================

To better make use of GPU capability, Algan performs animating
and rendering in batches of time steps. This means that instead of rendering one frame
at a time, Algan will actually render multiple frames in the same GPU kernel.
This results in faster rendering but uses more memory.
To improve rendering times you can try changing the default batch sizes for actors (animating)
and frames (rendering).
E.g.

.. code-block:: python

    # Render in batches of 100 frames at a time.
    algan.defaults.batch_defaults.DEFAULT_BATCH_SIZE_FRAMES = 100

    # Animate in batches of 200 time steps at a time. Batch size actors should
    # always be larger than or equal to batch size frames, and preferably
    # batch size actors should be an integer multiple of batch size frames.
    algan.defaults.batch_defaults.DEFAULT_BATCH_SIZE_ACTORS = 200

    render_to_file()

Note that if Algan runs out of memory during rendering, it will restart the rendering process
with halved batch size. This restarting process takes some time, so you may actually
get faster rendering times reducing batch sizes in some cases, as it leads to fewer restarts.
