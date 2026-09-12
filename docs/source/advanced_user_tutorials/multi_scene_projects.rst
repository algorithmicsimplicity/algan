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

Validate authoring before rendering
===================================

Use :meth:`~algan.project.Project.validate` to author every selected scene and
resolve its Speech timing without producing images or video:

.. code-block:: python

    report = project.validate()
    for scene in report.scenes:
        print(scene.name, scene.duration_seconds, scene.checkpoints)
    print(report.duration_seconds)

Validation skips scene save-frame and save-video calls. It still obtains audio
from the configured speech source and writes project transcripts, so it can
populate a speech cache and can fail on unavailable narration. The returned
``transcript`` is unwrapped source text, suitable for comparison with the original
script. Pretty-printed transcript files can wrap hyphenated words across lines.

This is an authoring check: it does not run frame-dependent updaters, compile
shaders, or prove visual correctness. Inspect selected checkpoints and motion
clips afterwards. Use ``python video.py --validate`` for the CLI equivalent.

Profile scenes and estimate an export
=====================================

:meth:`~algan.project.Project.profile` uses the existing
``algan.utils.profiling_utils.profile_scene`` helper with the project's scene
selection, narration source and output directories:

.. code-block:: python

    project.profile("intro", video_settings=HD)
    estimate = project.estimate_render_time(["intro", "the_loss_surface"],
                                            video_settings=HD)
    print(estimate.estimated_seconds, estimate.range_seconds)

Both calls render complete reference scenes twice by default. Select short
representative scenes to limit cost. Profiling reports include authoring time,
timeline evaluation, geometry preparation, kernel materialization
(frontend/cache/JIT), rendering, post-processing and the encoder drain. Project
profiling defaults to wall timers without enabling the GPU kernel profiler;
pass ``kernel_profiler=True`` for GPU-only timings and runtime reinitialization.
Reports and profile clips go under ``video_directory/profiling`` by default.

The estimate authors the entire project to measure duration, then profiles only
the selected references. It scales their weighted warm render rate and reports
observed first-pass overhead separately. Its range uses the fastest and slowest
sampled scene rates, not statistical confidence bounds. Unseen shaders, denser
scenes, different devices, settings or encoders can invalidate the estimate.
Authoring and concatenation are excluded. Pass the intended ``save_video_kwargs``
when encoder settings differ from the defaults. Project profiling restores its
temporary timing hooks when it finishes or fails, so later daemon jobs do not
inherit stage-timer overhead. Calling ``profile_scene`` directly retains its
existing persistent-instrumentation behavior for benchmark scripts.

.. code-block:: bash

    python video.py --profile intro --profile-runs 2 --video-settings HD
    python video.py --estimate-render-time intro the_loss_surface --video-settings HD

Diagnose cache permissions
==========================

``algan check`` reports resolved cache paths and tests writes with a temporary
file, including text, speech, kernel and daemon directories. It also probes the
output directory for the current working directory; a script's own destination
can differ. Failed path checks produce a nonzero exit status.

If Pango previously reported only ``error while writing to output stream``, the
text-cache preflight now identifies the inaccessible directory. Set
``SETTINGS.paths.cache_directory`` to a writable location, or set
``ALGAN_CACHE_DIR`` before starting Python. Kernel cache and daemon home use
``TI_OFFLINE_CACHE_FILE_PATH`` and ``ALGAN_HOME`` respectively. Run diagnostics
under the same permissions and sandbox as the render.

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
