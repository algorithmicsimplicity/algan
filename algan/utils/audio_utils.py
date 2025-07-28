import bisect
import os

from moviepy import AudioFileClip
import pyttsx3

from algan.settings.defaults import DIRECTORY_DEFAULTS


def get_speech_generator_from_file(audio_file, transcript=None):
    import nltk

    nltk.download("averaged_perceptron_tagger_eng")
    from forcealign import ForceAlign

    align = ForceAlign(audio_file=audio_file, transcript=transcript)
    words = align.inference()

    def strip_nonchars(x):
        for _ in "~`\"'][(),.?/><!@#$%^&*+-=-_\\|:;":
            x = x.replace(_, "")
        return x.upper()

    cum_num_chars = [0]
    for word in words:
        word.word = strip_nonchars(word.word)
        cum_num_chars.append(cum_num_chars[-1] + len(word.word) + 1)
    cum_num_chars = cum_num_chars[1:]

    def get_word_from_char_inds(start_ind, length):
        index_start = bisect.bisect_right(cum_num_chars, start_ind)
        index_end = bisect.bisect_right(cum_num_chars, start_ind + length)
        return words[index_start], words[index_end]

    full_transcript = " ".join([_.word for _ in words])

    def generate_speech(transcript):
        # Provide path to audio file and corresponding transcript
        transcript = strip_nonchars(transcript)
        start_word, end_word = get_word_from_char_inds(
            full_transcript.index(transcript), len(transcript)
        )

        ac = AudioFileClip(audio_file)
        return ac.subclipped(
            max(start_word.time_start - 0.05, 0),
            min(end_word.time_end + 0.05, ac.duration),
        )

    return generate_speech


def get_pyttsx_speech_generator(script):
    engine = pyttsx3.init()  # object creation
    file = os.path.join(DIRECTORY_DEFAULTS.cache_directory, "temp_ttsx_output.mp3")
    engine.save_to_file(script, file)
    engine.runAndWait()
    engine.stop()
    return AudioFileClip(file)
