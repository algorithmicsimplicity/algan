===============
Getting Started
===============

.. note::

  Before proceeding, install Algan and make sure it is running properly by
  following the detailed steps in :doc:`installation`.

.. important::

  If you installed Algan using the recommended Python management tool ``uv``,
  either activate its virtual environment (following the instructions ``uv venv``
  prints) or run your scripts through ``uv run``, as below. Otherwise Python will
  not find Algan.

Your First Algan Program
************************

The quickest way to start creating animations with Algan is to write a Python script and execute it.

Let's make a very simple animation. Create a new file named ``my_first_algan.py``,
and copy this code into it:

.. algan:: GettingStartedHelloWorld

    from algan import *

    text = Text('Hello World!', font_size=100)
    text.spawn()

    Scene.save_video("my_video")

Now run the script from your terminal using ``uv run python my_first_algan.py``.
If the execution is successful, you should find a new directory named `algan_outputs`
in the same directory as your Python script, and inside of that directory there should
be a video file `my_video.mp4` . Open this video file,
and you will see your first Algan animation playing: "Hello World!" appearing on screen.

***********
Explanation
***********

Let's break down this minimal program line-by-line to see what's going on:

.. code-block:: python

   from algan import *

This line imports all of Algan's functionality, making it available to use in your script.
All of your Algan scripts will start with this.

The next line

.. code-block:: python

    text = Text('Hello World!', font_size=100)


creates a Text object. In Algan, any object that can be displayed and animated
on screen is called a **Mob** (short for Moveable Object). Here, we create a :class:`.Text` object,
which is a type of :class:`.Mob` that displays text. We initialize it with the content "Hello World!"
and set its font size to 100 to take up most of the screen. This mob is then assigned the name *text* so we can
refer to it later in the script.

.. code-block:: python

    text.spawn()

This line *spawns* the mob we previously created. This step is crucial as mobs will not appear on screen,
and will not be animatable, until they have been spawned. By default, a mob will play a simple fade-in animation
when it is spawned. Without calling :meth:`~.Animatable.spawn`, your :class:`.Mob` will not appear in the final video.

.. code-block:: python

    Scene.save_video("my_video")

This final line instructs Algan to process all of the previously created animations and mobs you've defined
in your script and render them into a video file.

Where the file goes
===================

By default, the video is saved into an `algan_outputs` directory next to your
script, as a .mp4 file. You control that by what you pass to
:meth:`~algan.scene.Scene.save_video`:

.. code-block:: python

    Scene.save_video("my_video")              # algan_outputs/my_video.mp4
    Scene.save_video("my_video.mov")          # algan_outputs/my_video.mov
    Scene.save_video("renders/final.mp4")     # renders/final.mp4
    Scene.save_video("/tmp/absolute.mp4")     # exactly where you said

A bare name goes to the output directory; anything with a directory in it is
used exactly as written. If you leave the extension off, Algan picks ``.mp4``
for you (or ``.mov`` if your background is transparent).

:meth:`~algan.scene.Scene.save_video` returns a small result object describing
what it did, which is handy in scripts:

.. code-block:: python

    result = Scene.save_video("my_video")
    print(result.output_path, result.duration_seconds)

Rendering Faster While You Work
*******************************

You have probably noticed that the run above took a while, and that almost none
of that was your animation. Every fresh ``python my_first_algan.py`` re-imports
the library and re-prepares Algan's GPU kernels before it can draw a single
pixel. For a one-second Hello World, that fixed cost is most of the wait.

You only have to pay it once. The render daemon keeps a warm process alive and
re-runs your script on demand:

.. code-block:: bash

    uv run python -m algan.daemon my_first_algan.py --watch

Leave that running in its own terminal. With ``--watch`` it re-renders every
time you save the file; you can also press Enter in the daemon's terminal to
force a re-render, or ``q`` to quit. The first render still pays the startup
cost, but every one after it costs only the render itself -- around a second
for a scene this size, against roughly a minute from cold.

.. note::

    The daemon is entirely optional, and everything in these tutorials works
    with a plain ``uv run python my_first_algan.py``. It is worth starting as
    soon as you are iterating on a scene rather than running it once. See
    :doc:`../advanced_user_tutorials/performance_and_quality` for the rest of
    its options.

Video Settings
**************

By default, videos are rendered at low quality: 864x486 at 15 frames per
second. That keeps your edit-and-preview loop fast. When you want a
better-looking render, pass a quality preset as the second argument:

.. algan:: GettingStartedHelloWorldHD

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

There are two more presets for specialised uses: ``THUMBNAIL`` (a single
1280 x 720 frame) and ``SMOKE_TEST`` (32 x 32, for checking a script runs at all).

See :meth:`~algan.scene.Scene.save_video` for the full list of parameters,
:class:`~.VideoSettings` for building custom settings from scratch,
:doc:`../advanced_user_tutorials/settings` for everything else you can
configure, and :doc:`../advanced_user_tutorials/performance_and_quality` for
which of those settings is worth changing.

Saving Images
*************

To save a still image of the current scene, use
:meth:`~algan.scene.Scene.save_frame`. It resolves paths exactly like
:meth:`~algan.scene.Scene.save_video` does, but defaults to a ``.png``
extension:

.. code-block:: python

    from algan import *

    text = Text('Hello World!', font_size=100)
    text.spawn()

    Scene.save_frame("my_screen_shot", HD)

By default it captures the scene as it stands at the current point in your
script. Pass ``at`` to capture a specific moment, or several at once:

.. code-block:: python

    Scene.save_frame("shot.png", at=2.5)             # shot.png at t=2.5s
    Scene.save_frame("sheet.png", at=[0, 1, 2])      # sheet_0.png, sheet_1.png, ...

.. note::

    :meth:`~algan.scene.Scene.save_frame` never changes your scene, so you can
    drop it in anywhere while you are building an animation to see what things
    look like at that point.

Rendering more than once
************************

Saving does not consume your scene. You can keep animating afterwards and
render again:

.. code-block:: python

    from algan import *

    square = Square().spawn()
    Scene.save_video("part_one")

    square.move(RIGHT)
    Scene.save_video("part_one_and_two")

Note that Algan records animations onto a single timeline, so the second video
contains everything the first one did *plus* the new movement. To render
genuinely independent clips, give each one its own :class:`~algan.scene.Scene`:

.. code-block:: python

    from algan import *

    with Scene() as intro:
        Text("Chapter 1", scene=intro).spawn()
        intro.save_video("intro")

    with Scene() as outro:
        Text("The End", scene=outro).spawn()
        outro.save_video("outro")

Where to next
*************

The remaining new-user tutorials build on each other in order:

* :doc:`basic_animations` -- the three ways to change a Mob, and how Algan turns
  them into animation.
* :doc:`mob_gallery` -- everything you can put on screen.
* :doc:`positioning_and_layout` -- getting Mobs exactly where you want them.
* :doc:`text_and_math` -- labels, LaTeX and animated numbers.
* :doc:`controlling_animations` -- controlling *when* things happen and how long
  they take.
* :doc:`built_in_animations` -- ready-made animations for drawing attention and
  transforming diagrams.
* :doc:`updaters` -- rules that hold continuously instead of for a fixed time.
* :doc:`child_mobs` -- building complex objects out of simple ones.
* :doc:`three_d_basics` -- your first 3-D scene.
* :doc:`importing_from_manim` -- borrowing Manim's geometry library.
