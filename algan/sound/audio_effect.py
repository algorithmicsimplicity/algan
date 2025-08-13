from algan.utils.audio_utils import get_pyttsx_speech_generator


class AudioManager:
    _speech_generator = None
    _video_transcript = ""

    def __init__(self):
        raise RuntimeError("AudioManager cannot be instantiated.")

    @classmethod
    def set_speech_source(cls, speech_generator):
        cls._speech_generator = speech_generator

    @classmethod
    def append_script(cls, script):
        cls._video_transcript += script.strip(' ') + '\n\n'

    @classmethod
    def get_speech(cls, script):
        if cls._speech_generator is None:
            return get_pyttsx_speech_generator(script)
        return cls._speech_generator(script)


class AudioEffect:
    def __init__(self, audio_clip, start_time):
        self.audio_clip = audio_clip
        self.start_time_func = start_time
