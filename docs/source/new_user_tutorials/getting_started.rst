==========
Getting Started
==========

.. note::

  Before proceeding, install Algan and make sure it is running properly by
  following the detailed steps in :doc:`installation`.

.. important::

  If you installed Algan using the recommended Python management tool ``uv``,
  it's crucial to either activate the corresponding virtual environment (by following the
  instructions displayed when running ``uv venv``).

Running Your First Algan Program
********************************

The quickest way to start creating animations with Algan is to write a Python script and execute it.

Let's make a very simple animation. Create a new file named ``my_first_algan.py``, copy the code provided below into it, and then run it from your terminal using ``python my_first_algan.py``.

.. algan:: GettingStartedHelloWorld

    from algan import *

    my_first_mob = Text('Hello World!', font_size=100)
    my_first_mob.spawn()

    render_to_file()

If the execution is successful, you should find a new video file named `algan_render_output.mp4` in the same directory as your Python script. Open this video file, and you will see your first Algan animation playing: "Hello World!" appearing on screen.

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

    my_first_mob = Text('Hello World!', font_size=100)


creates a visual object. In Algan, any object that can be displayed and animated
on screen is called a **Mob** (short for Moveable Object). Here, we create a :class:`.Text` object,
which is a type of :class:`.Mob` specifically designed
to display text. We initialize it with the content "Hello World!" and set its font size
to 100 to take up most of the screen. This mob is then assigned the name *my_first_mob* so we can
refer to it later in the script.

.. code-block:: python

    my_first_mob.spawn()

This line *spawns* the mob we previously created. This step is crucial as mobs will not appear on screen,
and will not be animatable, until they have been spawned. By default, a mob will play a simple fade-in animation when it is spawned.
Without calling :meth:`~.Animatable.spawn`, your :class:`.Mob` will not appear in the final video.

.. code-block:: python

    render_to_file()

This final line instructs Algan to process all of the previously created animations and mobs you've defined
in your script and render them into a video file.
By default, the video will be saved in the same directory where you ran your Python script
under the name 'algan_render_output.mp4'.
You can change this behavior by specifying the `output_file` and `output_directory` parameters of
the :func:`.render_to_file` function.

