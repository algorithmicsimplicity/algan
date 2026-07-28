===============
Getting Started
===============

.. note::

  Before proceeding, install Algan and make sure it is running properly by
  following the detailed steps in :doc:`installation`.

.. important::

  If you installed Algan using the recommended Python management tool ``uv``,
  it's crucial to either activate the corresponding virtual environment (by following the
  instructions displayed when running ``uv venv``).

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
By default, the video will be saved in a `algan_outputs` directory under the same directory
where you ran your Python script from, and will be saved as a .mp4 file. You can change
this by providing a full path, or a file extension when you call :func:`~.save_video`.

.. code-block:: python

    # Save at path/to/my/video.mov
    Scene.save_video('path/to/my/video.mov')

Video Settings
**************

By default, videos are rendered at low quality (TODO insert actual fps and resolution of LQ here),
To change this, you can use :func:`~.set_video_settings` .

.. algan:: GettingStartedHelloWorldHD

    from algan import *
    Scene.set_video_settings(HD)

    text = Text('Hello World!', font_size=100)
    text.spawn()

    Scene.save_video("my_video")

.. important::

    You should always set the video settings immediately after importing Algan and before creating any Mobs,
    because some Mob behaviours depend on the video settings.

See :func:`~.save_video` for a description of the available parameters, and see :class:`~.VideoSettings`
for making custom video settings. Algan provides the following built in video
settings: PREVIEW, LD, MD, HD, PRODUCTION, UHD.

Saving Images
*************

If you want to save a screen-shot of the current Scene state to an image, you
can use :func:`~.save_frame` . :func:`~.save_frame` accepts the same parameters as
:func:`~.save_video`, and defaults to a .png file extension.

.. algan:: GettingStartedHelloWorldFrame

    from algan import *
    Scene.set_video_settings(HD)

    text = Text('Hello World!', font_size=100)
    text.spawn()

    Scene.save_frame("my_screen_shot")