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
    def __init__(self, audio_clip, start_time):
        self.audio_clip = audio_clip
        self.start_time_func = start_time
