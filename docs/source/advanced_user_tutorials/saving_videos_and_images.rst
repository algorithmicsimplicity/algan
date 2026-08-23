========================
Saving Videos and Images
========================

Algan provides two main methods for rendering your scenes: :meth:`~algan.scene.Scene.save_video`
for videos and :meth:`~algan.scene.Scene.save_frame` for one or multiple frames (images).
Both of them use the same protocol for deciding where the output is produced,
and both can be passed :class:`~.VideoSettings` as a parameter.
``Scene.save_frame`` additionally takes an ``at`` parameter which is used
to specify the set of time stamps to render frames at. ``at`` defaults to None
which means the current point in time when ``Scene.save_frame`` is called.

.. note::

    :meth:`~algan.scene.Scene.save_video` and :meth:`~algan.scene.Scene.save_frame`
    never alter your scene in any way, so you can use them anywhere
    in your script, and use them multiple times per script.

Changing the video settings
===========================

By default, videos are rendered at low quality: 864x486 at 15 frames per
second. That keeps your edit-and-preview loop fast. When you want a
better-looking render, pass a quality preset as the second parameter:

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

See :meth:`~algan.scene.Scene.save_video` for the full list of parameters,
:class:`~.VideoSettings` for building custom settings from scratch,
:doc:`../advanced_user_tutorials/settings` for everything else you can
configure.

Where the output file goes
==========================

By default, the file is saved into an `algan_outputs` directory next to your
script file, as a .mp4 file for videos (or ``.mov`` if your background is transparent)
and a .png for frames. This can be changed by adding paths or extensions to the name parameter.

.. code-block:: python

    Scene.save_video("my_video")                    # algan_outputs/my_video.mp4
    Scene.save_video("my_video.mov")                # algan_outputs/my_video.mov
    Scene.save_video("renders/final.mp4")           # renders/final.mp4
    Scene.save_video("/tmp/absolute.mp4")           # exactly where you said
    Scene.save_frame("shot")                        # shot.png at t=current_time
    Scene.save_frame("shot.jpg", at=-0.5)           # shot.jpg at t=current_time - 0.5s
    Scene.save_frame("shot.png", at=[0.5, 1, 1.5])  # sheet_0.png at t=0.5, sheet_1.png at t=1, ...

A bare name goes to the output directory; anything with a directory in it is
used exactly as written.

:meth:`~algan.scene.Scene.save_video` and :meth:`~algan.scene.Scene.save_frame` return a
small result object describing what it did, which is handy in scripts:

.. code-block:: python

    result = Scene.save_video("my_video")
    print(result.output_path, result.duration_seconds)

Working with projects
=====================

A single ``.py`` file with one ``Scene.save_video()`` at the end is great for
quickly making a short video. But when working on longer videos with
multiple scenes, you want to be able to chunk up the rendering correspondingly.

The :class:`~algan.project.Project` class is how you do that in Algan.
You give it a list of zero-parameter functions, each of which authors one Scene,
and it owns the identifiers, the output directories and the concatenation into
the complete video.

.. code-block:: python

    from algan import *

    def intro():
        Text("Gradient Descent", font_size=90).spawn()
        Scene.wait(2)

    def the_loss_surface():
        surface = Sphere(color=BLUE).spawn()
        surface.rotate(360, UP)

    def outro():
        Text("Thanks for watching", font_size=70).spawn()
        Scene.wait(2)

    project = Project([intro, the_loss_surface, outro], file_path="gradient_descent.mp4")

Note that the scene functions do **not** call ``Scene.save_video()``. The project
renders them; calling it yourself would render a second, unmanaged video.

Scene IDs are zero-based positions in the list, so ``intro`` is scene ``0``. Output
stems combine the two: ``0_intro``, ``1_the_loss_surface``, ``2_outro``. Because the
identifier comes from the position rather than from render order, rendering any
subset produces exactly the names rendering the whole project would.

Rendering
=========

.. code-block:: python

    project.render_video()                 # every scene
    project.render_video(1)                # just scene 1, by ID
    project.render_video("the_loss_surface")   # ... or by name
    project.render_video([0, "outro"])     # a mix

    project.concatenate_videos()           # stitch into file_path

``render_video`` skips any :meth:`~algan.scene.Scene.save_frame` calls in your scene
functions, and :meth:`~algan.project.Project.render_screenshots` does the opposite:
it runs the save-frame calls and renders no video.
This means you can sprinkle ``Scene.save_frame`` calls throughout your scene functions,
then use ``render_screenshots`` to get a quick, cheap, image-sequence view of your video.

.. code-block:: python

    project.render_screenshots("the_loss_surface", frames="perturbed", stop_early=True)

``frames`` selects save-frame calls by index (``3``), glob (``"s05_*"``) or plain
substring, matching with or without the generated ``s<scene>_f<index>_`` prefix.
``stop_early=True`` stops executing each scene as soon as every pattern has matched.

Driving it from the command line
================================

:meth:`~algan.project.Project.run_cli` turns the same script into a small tool, which
is usually how you want to work once a project has more than a few scenes:

.. code-block:: python

    if __name__ == "__main__":
        project.run_cli()

.. code-block:: bash

    python video.py --render-video                    # everything
    python video.py --render-video 1 outro            # two scenes
    python video.py --render-screenshots --frames "perturbed" --stop-early
    python video.py --concatenate-videos
    python video.py --render-video --video-settings HD

``run_cli`` returns ``True`` when it dispatched a project action and ``False`` when the
arguments contained none, so a script can fall back to its own behaviour. It ignores
arguments it does not recognize, which keeps it usable under launchers that add their
own, but it does handle ``-h``/``--help``, so parse your own help first if you want it.

Settings, output and narration
==============================

``Project`` takes a ``video_settings`` used to author and render every scene, which
keeps a project internally consistent without a global mutation at the top of the
file. Individual calls can still override it:

.. code-block:: python

    project = Project(scenes, video_settings=PREVIEW)     # while working
    project.render_video(video_settings=HD)               # for the final pass

Videos, screenshots and transcripts go to ``video_directory``,
``screenshot_directory`` and ``transcript_directory``. A bare directory name lands
under Algan's usual output directory; a path with an explicit parent is used as
given. A ``speech_source`` passed to the constructor is installed on every Scene's
audio manager, so narration is configured once rather than per scene, see
:doc:`audio_and_speech`.

See Also
========

* :class:`~algan.project.Project` -- the full API.
* :doc:`settings` -- how ``video_settings`` and the output paths resolve.
* :doc:`audio_and_speech` -- the speech generators a project can install.
