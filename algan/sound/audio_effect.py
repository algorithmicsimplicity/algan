"""Per-Scene audio: timed sound effects and the narration source.

:class:`AudioManager` is owned by each :class:`~algan.scene.Scene` -- there is no
process-global audio state, so two Scenes in one process can use different voices
or recorded narration without interfering. It holds the Scene's speech generator,
its accumulated transcript, and the list of effects to mix.

:class:`AudioEffect` is one sound placed at one time on the Scene's timeline.
The :class:`~algan.animation_timeline.animation_contexts.Audio` and
:class:`~algan.animation_timeline.animation_contexts.Speech` contexts add them,
taking their runtime from the clip so the animations inside are fitted to the
sound. ``save_video`` composes every effect into one track and hands it to the
video writer.

See :doc:`/advanced_user_tutorials/audio_and_speech`.
"""

from __future__ import annotations

from algan.utils.audio_utils import get_pyttsx_speech_generator


class AudioManager:
    """Per-scene audio transcript and speech-source state."""

    def __init__(self, scene=None, speech_generator=None):
        self.scene = scene
        self.speech_generator = speech_generator
        self.video_transcript = ""

    def set_speech_source(self, speech_generator):
        self.speech_generator = speech_generator
        return self

    def append_script(self, script):
        self.video_transcript += script.strip(" ") + "\n\n"
        return self

    def get_speech(self, script):
        if self.speech_generator is None:
            return get_pyttsx_speech_generator(script)
        return self.speech_generator(script)


class AudioEffect:
    """One sound placed at one moment on a Scene's timeline.

    A Scene collects these as it is authored and mixes them into a single
    track when :meth:`~algan.scene.Scene.save_video` runs -- each clip is
    offset to its own start time, so overlapping effects simply play over each
    other.

    You do not usually build one: the
    :class:`~algan.animation_timeline.animation_contexts.Audio` and
    :class:`~algan.animation_timeline.animation_contexts.Speech` contexts add
    one for you, at the time the block opens, and take the block's runtime from
    the clip's own duration so the animations inside are fitted to the sound.

    Parameters
    ----------
    audio_clip
        A moviepy ``AudioClip`` -- what ``AudioFileClip(path)`` returns, or the
        clip a speech generator produced.
    start_time
        A zero-argument callable returning the scene time, in seconds, at which
        the clip should begin. It is a callable rather than a number because a
        context's start is only final once the block it sits in has been
        rescaled, which happens after the effect is registered.

    Attributes
    ----------
    audio_clip
        The clip, as supplied.
    start_time_func
        The callable supplied as ``start_time``.

    See Also
    --------
    :class:`~algan.animation_timeline.animation_contexts.Audio` : Play a sound
        file over a block of animation.
    :class:`~algan.animation_timeline.animation_contexts.Speech` : Narrate a
        block, generating the audio from text.

    Examples
    --------
    Reach for the context rather than the class:

    .. code-block:: python

        from algan import *

        square = Square().spawn()
        with Audio("chime.wav"):
            square.rotate(90)

        Scene.save_video("chimed")
    """

    def __init__(self, audio_clip, start_time):
        self.audio_clip = audio_clip
        self.start_time_func = start_time
