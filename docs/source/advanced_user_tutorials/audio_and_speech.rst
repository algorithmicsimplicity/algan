================
Audio and Speech
================

Algan lets you synchronize animations directly with sound files and voice-over
narration. Each :class:`~algan.scene.Scene` maintains its own
:class:`~algan.sound.audio_effect.AudioManager`, audio tracks, and speech
source.

Audio Contexts
==============

Use :class:`~algan.animation_timeline.animation_contexts.Audio` to align an
animation block's runtime to a sound file:

.. algan-doc-check: skip -- requires music.wav asset

.. code-block:: python

    from algan import *

    circle = Circle().spawn()

    with Audio("music.wav"):
        circle.rotate(360, OUT)
        circle.scale(2)

    Scene.save_video("music_scene.mp4")

This flips the usual animation workflow on its head: instead of manually
calculating how many seconds each visual action should take, the audio clip
determines the runtime, and all animations inside the block automatically scale
to match.

Both ``Audio`` and ``Speech`` contexts take ``wait_at_end``, a number of extra seconds to hold after the
clip finishes, so you can add longer pauses to parts of the narration.
``Speech`` defaults it to 1 second; ``Audio`` defaults it to 0.

``Audio`` accepts string file paths or MoviePy ``AudioFileClip`` objects (allowing you
to trim or preprocess audio beforehand):

.. code-block:: python

    from moviepy import AudioFileClip

    clip = AudioFileClip("music.mp3").subclipped(10, 20)
    with Audio(clip):
        mob.move(RIGHT)

Speech Contexts
===============

:class:`~algan.animation_timeline.animation_contexts.Speech` is an
:class:`~algan.animation_timeline.animation_contexts.Audio` context whose clip
is generated from a script segment, so the block runs for exactly as long as
the line takes to say:

.. algan-doc-check: skip -- needs a system text-to-speech engine (eSpeak)

.. code-block:: python

    from algan import *

    title = Text("Gradient descent").spawn()

    with Speech("Gradient descent follows the slope downhill."):
        title.move(UP)
        title.color = BLUE

    Scene.save_video("gradient_descent.mp4")

.. important::

    The default speech generator synthesizes through ``pyttsx3``, which drives a
    **system** text-to-speech engine rather than shipping one. macOS and Windows
    have one built in (NSSpeechSynthesizer and SAPI5), so ``Speech`` works out of
    the box there. On Linux (including CI containers and most Docker images)
    ``pyttsx3`` falls back to **eSpeak**, which is not installed by default and
    is not a Python dependency Algan can pull in for you. Without it, a
    ``Speech`` context raises at synthesis time. You
    can install it with:

    .. code-block:: bash

        sudo apt install espeak-ng   # Debian / Ubuntu
        sudo dnf install espeak-ng   # Fedora

    If you would rather not depend on a system engine at all, supply your own
    generator (see `A custom speech generator`_) or use recorded narration.

By default, the Scene's AudioManager uses Algan's pyttsx3 speech generator. Each
``Speech`` context appends its script to ``scene.audio_manager.video_transcript``.
When a video contains audio, ``save_video`` writes that transcript beside the
video as ``<video_stem>_script.txt``.

Using recorded narration
========================

For recorded narration, configure the specific Scene's AudioManager rather than
a process-global singleton:

.. algan-doc-check: skip -- needs narration.wav/.txt, which do not ship with the docs

.. code-block:: python

    from algan import *
    from algan.utils.audio_utils import get_speech_generator_from_file

    generator = get_speech_generator_from_file(
        audio_file="narration.wav",
        transcript_file="narration.txt",
    )
    Scene.current().audio_manager.set_speech_source(generator)

    diagram = Circle().spawn()
    with Speech("First we draw a circle."):
        diagram.scale(1.5)

    Scene.save_video("narrated_diagram.mp4")

``get_speech_generator_from_file`` aligns the transcript to the audio and
returns a callable. Each Speech segment asks that callable for the matching
subclip. The optional audio dependencies used for alignment are available via
Algan's ``audio`` extra.

A custom speech generator
=========================

A speech generator is any callable accepting a script string and returning a
MoviePy audio clip:

.. code-block:: python

    from moviepy import AudioFileClip

    def speech_generator(script):
        # Select or synthesize a clip for this exact script segment.
        return AudioFileClip("prepared_segment.wav")

    Scene.current().audio_manager.set_speech_source(speech_generator)

The generator is Scene-local. Two Scenes can use different voices or recorded
sources in the same process without interfering with one another.

Composing narration and sound effects
=====================================

Audio contexts nest like other animation contexts. For example, a sound effect
can run in parallel with a visual change inside a narration segment:

.. code-block:: python

    with Speech("The object now transforms into a triangle."):
        with Sync():
            with Audio("whoosh.wav"):
                pass
            mob.become(Triangle(add_to_scene=False))

Following the transcript in the viewer
======================================

Call ``Scene.view()`` after your ``Speech`` blocks to inspect their narration
alongside the animation. The right-hand panel has **Fragments** and **Transcript**
tabs. The transcript includes the text from every entered ``Speech`` block up to
that call, in authoring order, with the original punctuation and spacing.

During playback or seeking, the transcript scrolls to the current position and
highlights the word being spoken. Clicking a timed word seeks to the first video
frame at or after its narration starts. Arrow keys, Home and End switch tabs when
a tab has keyboard focus; transcript words also work with Enter and Space.
Silence between words or after a sentence has no highlighted word. Overlapping
narration can highlight more than one word.

Recorded narration from ``get_speech_generator_from_file`` uses its existing
word alignment, including the padding at the beginning of each selected clip.
The default text-to-speech engine and custom generators that return only an audio
clip have no word alignment: their word timings are estimated across the clip's
duration and labeled as estimates in the transcript panel. Opening the viewer
does not run a new alignment model or synthesize speech again.

The transcript is a snapshot. With ``block=False``, speech added afterwards is
visible only in a new viewer. Non-speech ``Audio`` blocks are not transcript
entries. Speech suppressed by a non-animated context remains readable but has no
clickable word timings, because there is no corresponding sound on the timeline.

See Also
========

* :doc:`../new_user_tutorials/combining_animations` -- the animation contexts
  ``Audio`` and ``Speech`` extend.
* :doc:`text_and_math` -- putting on screen what the narration is saying.
* :doc:`multi_scene_projects` -- installing one ``speech_source`` across every
  scene of a video, and where transcripts land.
* :doc:`saving_videos_and_images` -- the ``audio_codec`` and ``ffmpeg_params``
  arguments to ``save_video``.
