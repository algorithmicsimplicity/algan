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

Listening in the interactive viewer
-----------------------------------

``Scene.view()`` plays the scene's existing ``Speech`` and ``Audio`` clips when
**Play** is pressed. Clips retain their recorded start times, overlap as they do
in the video, and include silent gaps. No extra option is required. **Stop**
pauses the sound; timeline clicks and transcript-word clicks seek silently, and
pressing **Play** resumes from the new position. Playback pauses both sound and
picture while waiting for an unrendered frame.

The mix is snapshotted when the viewer opens and prepared in memory on first
playback, without writing an audio file or synthesizing speech again. Later
authoring requires a new viewer. Scenes without audio keep their silent playback.
The same controls work in ``Project.view()``: changing tabs stops the previous
scene's sound, and unselected scenes are not loaded to prepare audio.

Following the transcript in the viewer
======================================

Open :meth:`Scene.view() <algan.scene.Scene.view>` and select **Transcript** in
its right-hand panel. **Fragments** returns to the existing pixel, fragment and
attribute inspector; switching tabs preserves both panels.

The transcript includes every ``Speech`` script recorded before that viewer
was opened, in authoring order, with the original spelling and whitespace.
Later additions require opening a new viewer. Playback, timeline clicks and
manual time/frame jumps highlight the words at the playhead and scroll the
transcript to keep that passage visible. Clicking a word pauses playback and
seeks to the first available video frame at or after its narration begins.
During pauses, no word is highlighted; overlapping narration can highlight
more than one word. The transcript scrolls independently of the video and
hierarchy.

Recorded narration produced by ``get_speech_generator_from_file`` retains its
word alignment, including the padding around each trimmed clip. Timings follow
the audio effect's actual Scene offset, not the animation's stretched runtime
or ``wait_at_end`` pause. A custom generator can supply word timestamps as
shown below. Sources without usable word alignment, including the default
pyttsx3 generator, use **estimated** word timings distributed over the clip's
duration; the viewer explicitly labels these estimates. Text with no playable
audio remains visible without seek links. Opening the viewer does not synthesize
speech or run a new alignment job.

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

Supplying word timings from a custom generator
----------------------------------------------

Optionally attach ``algan_word_timestamps`` to the returned clip. It is a
sequence of ``(word, start, end)`` triples, with finite times in **seconds
relative to the returned clip's start**, not the original recording or the
Scene. Keep the entries in spoken order and inside the clip's duration. Each
interval is start-inclusive and end-exclusive. The displayed transcript keeps
the script's spelling and punctuation rather than the alignment's normalized
spelling.

.. algan-doc-check: skip -- needs a prepared_segment.wav asset

.. code-block:: python

    from moviepy import AudioFileClip

    def speech_generator(script):
        # This prepared clip narrates "Hello, world!" and is at least 1.1 s long.
        clip = AudioFileClip("prepared_segment.wav")
        clip.algan_word_timestamps = (
            ("Hello", 0.10, 0.45),
            ("world", 0.60, 1.10),
        )
        return clip

Word sequences that do not match the script, or malformed/out-of-range timing
metadata, fall back to the explicitly labeled estimates rather than shifting
subsequent words to incorrect timestamps. A generator that trims or changes the
speed of a clip must adjust its timestamps to match that returned clip.

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

See Also
========

* :doc:`../new_user_tutorials/combining_animations` -- the animation contexts
  ``Audio`` and ``Speech`` extend.
* :doc:`text_and_math` -- putting on screen what the narration is saying.
* :doc:`multi_scene_projects` -- installing one ``speech_source`` across every
  scene of a video, and where transcripts land.
* :doc:`saving_videos_and_images` -- the ``audio_codec`` and ``ffmpeg_params``
  arguments to ``save_video``.
