================
Audio and Speech
================

Audio belongs to a Scene. Each :class:`~algan.scene.Scene` has its own
:class:`~algan.sound.audio_effect.AudioManager`, transcript, speech source, and
list of timed audio effects. This keeps separately-authored Scenes from sharing
narration state accidentally.

Audio contexts
==============

Use :class:`~algan.animation_timeline.animation_contexts.Audio` to align an
animation context with a sound file. Its duration is taken from the audio clip:

.. algan-doc-check: skip -- needs music.wav, which does not ship with the docs

.. code-block:: python

    from algan import *

    with Scene() as scene:
        circle = Circle().spawn()

        with Audio("music.wav"):
            circle.rotate(360, OUT)
            circle.scale(2)

        scene.save_video("music_scene.mp4")

This inverts the usual relationship between animation and timing. Ordinarily you
say how long an animation takes; inside an ``Audio`` or ``Speech`` context the
*sound* sets the duration and the animations inside are fitted to it, exactly as a
``run_time`` would (see
:doc:`../new_user_tutorials/combining_animations`). That is what makes narrated
video practical to author: write the narration, and the visuals follow it.

Both contexts take ``wait_at_end``, a number of extra seconds to hold after the
clip finishes -- useful for letting a point land before the next line starts.
``Speech`` defaults it to 1 second; ``Audio`` defaults it to 0.

``Audio`` accepts either a path string or a MoviePy ``AudioFileClip``. You can
therefore trim or otherwise prepare a clip before passing it to Algan:

.. code-block:: python

    from moviepy import AudioFileClip

    clip = AudioFileClip("music.mp3").subclipped(10, 20)
    with Audio(clip):
        mob.move(RIGHT)

The context adds an :class:`~algan.sound.audio_effect.AudioEffect` to its owning
Scene at the context's start time. During ``save_video``, Algan composes all
Scene effects into a temporary audio track and passes it to the video writer.

Speech contexts
===============

:class:`~algan.animation_timeline.animation_contexts.Speech` is an Audio context
whose clip is generated from a script segment:

.. algan-doc-check: skip -- needs a system text-to-speech engine (eSpeak)

.. code-block:: python

    from algan import *

    with Scene() as scene:
        title = Text("Gradient descent").spawn()

        with Speech("Gradient descent follows the slope downhill."):
            title.move(UP)
            title.color = BLUE

        scene.save_video("gradient_descent.mp4")

.. important::

    The default speech generator synthesizes through ``pyttsx3``, which drives a
    **system** text-to-speech engine rather than shipping one. macOS and Windows
    have one built in (NSSpeechSynthesizer and SAPI5), so ``Speech`` works out of
    the box there. On Linux -- including CI containers and most Docker images --
    ``pyttsx3`` falls back to **eSpeak**, which is not installed by default and
    is not a Python dependency Algan can pull in for you. Without it, a
    ``Speech`` context raises at synthesis time:

    .. code-block:: shell

        sudo apt install espeak-ng          # Debian/Ubuntu
        sudo dnf install espeak-ng          # Fedora

    If you would rather not depend on a system engine at all, supply your own
    generator (see `A custom speech generator`_) or use recorded narration --
    neither goes near ``pyttsx3``.

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

    with Scene() as scene:
        generator = get_speech_generator_from_file(
            audio_file="narration.wav",
            transcript_file="narration.txt",
        )
        scene.audio_manager.set_speech_source(generator)

        diagram = Circle().spawn()
        with Speech("First we draw a circle."):
            diagram.scale(1.5)

        scene.save_video("narrated_diagram.mp4")

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

    scene.audio_manager.set_speech_source(speech_generator)

The generator is Scene-local. Two Scenes can use different voices or recorded
sources in the same process without interfering with one another.

Composing narration and sound effects
=====================================

Audio contexts nest like other animation contexts. For example, a sound effect
can run in parallel with a visual change inside a narration segment:

.. code-block:: python

    with Speech("The object now changes shape."):
        with Sync():
            with Audio("transform.wav"):
                pass
            mob.become(Triangle(scene=mob.scene, add_to_scene=False))

All nested contexts must resolve to the same Scene's AnimationManager. Mixing
mobs or explicit animation managers from different Scenes in one context is
rejected.

Practical notes
===============

* Use audio formats supported by MoviePy/FFmpeg.
* Keep the source clip open until rendering finishes; Algan closes its composed
  output clip after writing.
* Audio is rendered only when the Scene contains effects.
* Audio effects stay on the Scene after ``save_video`` returns, along with the
  rest of the timeline. Pass ``reset=True`` to discard them together with the
  Scene's other authored state.
* ``save_frame`` does not render audio and never modifies the Scene.

See Also
========

* :doc:`../new_user_tutorials/combining_animations` -- the animation contexts
  ``Audio`` and ``Speech`` extend.
* :doc:`../new_user_tutorials/text_and_math` -- putting on screen what the
  narration is saying.
