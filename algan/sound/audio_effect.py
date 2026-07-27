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
