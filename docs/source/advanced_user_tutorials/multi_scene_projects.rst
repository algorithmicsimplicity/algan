====================
Multi-Scene Projects
====================

A single ``.py`` file with one ``Scene.save_video()`` at the end is the right shape
for one shot. A finished video is usually a dozen of them, and once you are there you
want to re-render scene 7 without touching the other eleven, keep the output names
stable, and stitch the result together at the end.

:class:`~algan.project.Project` is that layer. You give it a list of zero-argument
functions, each of which authors one Scene, and it owns the identifiers, the output
directories and the concatenation.

Defining a project
==================

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
functions, and :meth:`~algan.project.Project.render_screenshots` does the opposite --
it runs the save-frame calls and renders no video. That split is what makes stills
cheap to iterate on:

.. code-block:: python

    project.render_screenshots("the_loss_surface", frames="perturbed", stop_early=True)

``frames`` selects save-frame calls by index (``3``), glob (``"s05_*"``) or plain
substring, matching with or without the generated ``s<scene>_f<index>_`` prefix.
``stop_early=True`` abandons each scene as soon as every pattern has matched, so
iterating on an early frame does not author the rest of the scene you are not looking
at.

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
own -- but it does handle ``-h``/``--help``, so parse your own help first if you want it.

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
audio manager, so narration is configured once rather than per scene -- see
:doc:`audio_and_speech`.

See Also
========

* :class:`~algan.project.Project` -- the full API.
* :func:`~algan.project.algan_scene` -- marks a zero-argument function as a scene
  entry point for ``render_all_funcs``.
* :doc:`saving_videos_and_images` -- the single-Scene form these calls wrap.
* :doc:`settings` -- how ``video_settings`` and the output paths resolve.
* :doc:`audio_and_speech` -- the speech generators a project can install.
* :doc:`performance_and_quality` -- why you draft at ``PREVIEW`` and render the
  final pass once.
