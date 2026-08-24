===============
Getting Started
===============

.. note::

  Before proceeding, install Algan and make sure it is running properly by
  following the detailed steps in :doc:`../installation`.

.. important::

  If you installed Algan using the recommended Python management tool ``uv``,
  either activate its virtual environment (following the instructions ``uv venv``
  prints) or run your scripts through ``uv run``, as below. Otherwise Python will
  not find Algan.

Your First Algan Program
========================

The simplest way to use Algan is to write a Python script and run it.

Let's make a very simple animation. Create a new file named ``my_first_algan.py``,
and copy this code into it:

.. algan:: GettingStartedHelloWorld

    from algan import *

    text = Text('Hello World!', font_size=100)
    text.spawn()
    text.wait(1)
    text.despawn()

    Scene.save_video("my_video")

Now run the script from your terminal using ``uv run python my_first_algan.py``.
If the execution is successful, you should find a new directory named
``algan_outputs`` in the same directory as your Python script, and inside of that
directory there should be a video file ``my_video.mp4``. Open this video file,
and you will see your first Algan animation playing: "Hello World!" appearing on
screen.

Explanation
===========

Let's break down this minimal program line-by-line to see what's going on:

.. code-block:: python

   from algan import *

This line imports all of Algan's functionality, making it available to use in your script.
All of your Algan scripts will start with this line.

The next line

.. code-block:: python

    text = Text('Hello World!', font_size=100)


creates a Text object. In Algan, any object that can be animated and appears on screen is
called a **Mob** (short for Moveable Object). Here, we create a :class:`.Text` object,
which is a type of :class:`.Mob` that displays text. We initialize it with the content "Hello World!"
and a font size of 100. This mob is then assigned the name *text* so we can
refer to it later in the script.

.. code-block:: python

    text.spawn()

This line *spawns* the mob we previously created. This step is crucial as mobs will not appear on screen,
and will not be animatable, until they have been spawned. By default, a mob will play a simple fade-in animation
when it is spawned. Without calling :meth:`~.Animatable.spawn`, your :class:`.Mob` will not appear in the final video.

.. code-block:: python

    text.wait(1)

This line uses :meth:`~.Animatable.wait` with a value of 1 to do... nothing! The mob waits unchanged for one second.

.. code-block:: python

    text.despawn()

And this *despawns* the mob, removing it from the scene with a simple fade-out animation. Mobs do not need
to be despawned, and if they are not despawned they will stick around until the end of the video.

.. code-block:: python

    Scene.save_video("my_video")

This final line instructs Algan to process all of the previously created mobs and animations you've defined
in your script and render them into a video file with the given name.

.. seealso::

    :doc:`../advanced_user_tutorials/text_and_math` -- everything
    :class:`~algan.mobs.text.Text` can do, plus LaTeX with
    :class:`~algan.mobs.text.Tex`, per-glyph animation, the hand-writing effect
    and animated numbers.

Where To Next
=============

* :doc:`basic_animations` -- continue with the new-user tutorials to learn about
  animating mobs.
* :doc:`../galleries/mob_gallery` -- see all of Algan's available mobs.
* :doc:`../advanced_user_tutorials/saving_videos_and_images` -- customize the
  output file and the video quality.
