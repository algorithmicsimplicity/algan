==========
Getting Started
==========

.. note::

  Before proceeding, install Algan and make sure it is running properly by
  following the steps in :doc:`installation`.


.. important::

  If you installed Algan in the recommended way, using the
  Python management tool ``uv``, then you either need to make sure the corresponding
  virtual environment is activated (follow the instructions printed on running ``uv venv``),
  or you need to remember to prefix the ``manim`` command in the console with ``uv run``;
  that is, ``uv run manim ...``.

Running a Minimal Algan Program
******************************

The simplest way to use Algan is to make a new Python file, and run it.
Create a new file named my_first_algan.py, copy the below code into it, and then run it with ``python my_first_algan.py``.

.. algan:: GettingStartedHelloWorld

    from algan import *

    Text('Hello World!', font_size=100).spawn()

    render_to_file()

If successful, you should now see a new file named algan_render_output.mp4 in the same directory as the Python file you
ran. Open that video, and you should see the above animation play.

There are 3 lines in this program. The first imports the algan library, making all of its tools and functions
available to use.

The second creates a :class:`.Text` object which will be displayed. In Algan, objects which
will be displayed on screen are called Mobs (Moveable Objects), and must inherit from the :class:`.Mob` base class.
Algan provides :class:`.Mob` s for displaying various simple shapes,
text, and LaTeX. You can also import Mobs from Manim, or combine mobs together to build up more complex objects.

.. important::

    Note that the .spawn() method is called on the Mob, this is necessary to make it appear on screen!

The third and final line tells Algan to render all of the previously specified animations into a video file. By
default the video will be output in the same directory as the running file, but you can change this behaviour
by specifying the output_file and output_directory parameters of :func:`~.utils.algan_utils.render_to_file`.

