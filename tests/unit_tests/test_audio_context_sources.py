"""What ``Audio`` takes as its source, and how it says when it cannot."""

from __future__ import annotations

import gc
import sys

import numpy as np
import pytest
from PIL import Image
from scipy.io import wavfile

from algan.animation_timeline.animation_contexts import Audio
from algan.mobs.shapes_2d import Square
from algan.utils import audio_utils

SAMPLE_RATE = 8000


@pytest.fixture
def half_second_wav(tmp_path):
    path = tmp_path / "half_a_second.wav"
    wavfile.write(path, SAMPLE_RATE, np.zeros(SAMPLE_RATE // 2, dtype=np.int16))
    return path


def test_a_path_object_is_read_like_a_string(fresh_scene, half_second_wav):
    """Only a ``str`` was opened; anything else was taken for a clip.

    ``Audio(Path(...))`` failed with "'PosixPath' object has no attribute
    'duration'".
    """
    context = Audio(half_second_wav)
    with context:
        Square().spawn()

    assert context.runtime == pytest.approx(0.5, abs=0.01)


def test_a_missing_file_names_where_it_was_looked_for(
    fresh_scene, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    unraisable = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)

    with pytest.raises(FileNotFoundError) as raised:
        Audio("whoosh.wav")
    message = str(raised.value)
    context = type(raised.value.__context__)
    # Let go of the traceback, whose frames would keep a half-built reader
    # alive, and collect it now.
    del raised
    gc.collect()

    assert "'whoosh.wav'" in message
    assert repr(str(tmp_path / "whoosh.wav")) in message
    # moviepy found out only after FFmpeg had failed: its error came chained
    # to an internal TypeError, and the half-built reader's __del__ printed an
    # AttributeError once the traceback had gone.
    assert context is type(None)
    assert [type(hook.exc_value).__name__ for hook in unraisable] == []


def test_an_absolute_path_is_not_resolved_again(fresh_scene, tmp_path):
    missing = tmp_path / "whoosh.wav"
    with pytest.raises(FileNotFoundError) as raised:
        Audio(missing)
    assert str(raised.value) == f"Audio found no file at {str(missing)!r}."


def _opening_error(monkeypatch, open_audio):
    """The message ``open_audio`` raises, and what the collector printed after."""
    unraisable = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    with pytest.raises(OSError) as raised:
        open_audio()
    message = str(raised.value)
    # Let go of the traceback, whose frames keep the half-built reader alive,
    # and collect it now.
    del raised
    gc.collect()
    return message, [type(hook.exc_value).__name__ for hook in unraisable]


@pytest.mark.parametrize(
    "contents", [b"not audio at all\n", b"", b"RIFF"], ids=["text", "empty", "header"]
)
def test_a_file_ffmpeg_cannot_open_is_named_with_ffmpegs_reason(
    fresh_scene, tmp_path, monkeypatch, contents
):
    """moviepy raised FFmpeg's whole report, chained to an internal TypeError,
    and the half-built reader's __del__ then printed an AttributeError.
    """
    damaged = tmp_path / "whoosh.wav"
    damaged.write_bytes(contents)

    message, unraisable = _opening_error(monkeypatch, lambda: Audio(damaged))

    assert message.startswith(f"Could not read {str(damaged)!r} as audio. FFmpeg says ")
    assert "Invalid data found when processing input" in message
    assert message.endswith(
        "Check that it plays in another player, and if it does not, export it again."
    )
    assert unraisable == []


def _silent_video(path):
    # Written as Algan writes its videos, through moviepy's FFmpeg.
    from moviepy import ColorClip

    clip = ColorClip(size=(16, 16), color=(0, 0, 255), duration=0.4)
    clip.write_videofile(str(path), fps=5, audio=False, logger=None)
    clip.close()


def _picture(path):
    Image.new("RGB", (8, 8), (255, 0, 0)).save(path)


@pytest.mark.parametrize(
    ("name", "make"),
    [("title.mp4", _silent_video), ("chime.png", _picture)],
    ids=["silent video", "picture"],
)
def test_a_file_without_a_sound_track_says_so(
    fresh_scene, tmp_path, monkeypatch, name, make
):
    """A video without sound raised ``KeyError: 'audio_bitrate'``, and a
    picture moviepy's failure to read a duration from FFmpeg's report.
    """
    path = tmp_path / name
    make(path)

    message, unraisable = _opening_error(monkeypatch, lambda: Audio(path))

    assert message == (
        f"Could not read {str(path)!r} as audio. FFmpeg finds a video stream in "
        f"it, but no audio stream."
    )
    assert unraisable == []


def test_a_directory_is_named_as_one(fresh_scene, tmp_path):
    """It was "Audio found no file at", of a path that is there."""
    with pytest.raises(IsADirectoryError) as raised:
        Audio(tmp_path)
    assert str(raised.value) == (
        f"Audio takes an audio file, and {str(tmp_path)!r} is a directory."
    )


def test_a_recorded_narration_ffmpeg_cannot_open_is_named(tmp_path, monkeypatch):
    recording = tmp_path / "voice.wav"
    recording.write_bytes(b"not audio at all\n")
    transcript = tmp_path / "script.txt"
    transcript.write_text("Hello world", encoding="utf-8")

    message, unraisable = _opening_error(
        monkeypatch,
        lambda: audio_utils.get_speech_generator_from_file(recording, transcript),
    )

    assert message.startswith(f"Could not read {str(recording)!r} as audio.")
    assert unraisable == []


def test_something_that_is_neither_a_path_nor_a_clip_is_named(fresh_scene):
    with pytest.raises(TypeError, match="path of an audio file .* got int"):
        Audio(3)


def test_a_url_is_left_for_ffmpeg_to_fetch(fresh_scene, monkeypatch):
    import moviepy

    opened = []

    class FakeClip:
        duration = 2.0

        def __init__(self, source):
            opened.append(source)

    monkeypatch.setattr(moviepy, "AudioFileClip", FakeClip)
    url = "https://example.com/whoosh.wav"

    context = Audio(url)
    with context:
        Square().spawn()

    assert opened == [url]
    assert context.runtime == pytest.approx(2.0)
