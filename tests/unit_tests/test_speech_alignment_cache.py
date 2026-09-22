"""Recorded speech caches follow the input contents and reject incomplete data."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import moviepy
import pytest

from algan.errors import AudioTranscriptMismatchError
from algan.utils import audio_utils


@pytest.fixture
def recording(tmp_path, monkeypatch):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"first recording")
    transcript = tmp_path / "script.txt"
    transcript.write_text("Hello world", encoding="utf-8")
    cache = tmp_path / "cache"
    calls, clips = [], []

    def open_audio(path):
        clip = SimpleNamespace(duration=10, closed=False)
        clip.close = lambda: setattr(clip, "closed", True)
        clip.subclipped = lambda start, end: SimpleNamespace(duration=end - start)
        clips.append(clip)
        return clip

    def align(audio_file, transcript_file, **kwargs):
        calls.append((audio_file, transcript_file, kwargs))
        text = transcript.read_text(encoding="utf-8").replace("-", " ")
        return [
            [audio_utils.strip_nonchars(word), i + 0.1, i + 0.9]
            for i, word in enumerate(text.split())
        ]

    monkeypatch.setattr(moviepy, "AudioFileClip", open_audio)
    monkeypatch.setattr(audio_utils, "align_large_audio_torchaudio_robust", align)
    monkeypatch.setattr(
        audio_utils,
        "SETTINGS",
        SimpleNamespace(paths=SimpleNamespace(cache_directory=cache)),
    )
    return SimpleNamespace(
        audio=audio,
        transcript=transcript,
        cache=cache,
        calls=calls,
        clips=clips,
        create=lambda: audio_utils.get_speech_generator_from_file(audio, transcript),
    )


def test_unchanged_inputs_reuse_alignment_and_keep_relative_clip_times(recording):
    recording.create()
    generator = recording.create()
    assert len(recording.calls) == 1
    assert len(list(recording.cache.glob("audio/*.json"))) == 1
    clip = generator("Hello world")
    assert clip.algan_word_timestamps[0] == ("HELLO", 0.05, 0.85)
    assert clip.algan_word_timestamps[1][1:] == pytest.approx((1.05, 1.85))


def test_transcript_edit_invalidates_alignment(recording):
    recording.create()
    recording.transcript.write_text("Hello reader", encoding="utf-8")
    generator = recording.create()
    assert len(recording.calls) == 2
    assert generator("Hello reader").algan_word_timestamps[-1][0] == "READER"


def test_same_size_audio_replacement_with_same_duration_and_mtime_invalidates(
    recording,
):
    recording.create()
    old = recording.audio.stat()
    recording.audio.write_bytes(b"other recording")
    os.utime(recording.audio, ns=(old.st_atime_ns, old.st_mtime_ns))
    recording.create()
    assert len(recording.calls) == 2


def test_identical_relative_names_in_other_directories_do_not_share_cache(
    recording, tmp_path, monkeypatch
):
    monkeypatch.chdir(recording.audio.parent)
    audio_utils.get_speech_generator_from_file("voice.wav", "script.txt")
    other = tmp_path / "other"
    other.mkdir()
    (other / "voice.wav").write_bytes(recording.audio.read_bytes())
    (other / "script.txt").write_bytes(recording.transcript.read_bytes())
    monkeypatch.chdir(other)
    audio_utils.get_speech_generator_from_file("voice.wav", "script.txt")
    assert len(recording.calls) == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("_ALIGNMENT_CACHE_VERSION", 123),
        ("MODEL_ID", "another-model"),
        ("CHUNK_DURATION_S", 30),
    ],
)
def test_alignment_configuration_and_version_are_in_the_cache_key(
    recording, monkeypatch, field, value
):
    recording.create()
    monkeypatch.setattr(audio_utils, field, value)
    recording.create()
    assert len(recording.calls) == 2


@pytest.mark.parametrize(
    "corruption",
    [
        "truncated",
        "version",
        "missing",
        "extra",
        "wrong-word",
        "nan",
        "infinity",
        "negative",
        "reversed",
        "overlap",
        "past-end",
        "non-numeric",
        "shape",
        "empty",
        "boolean",
        "huge",
    ],
)
def test_unusable_cache_entries_are_realigned(recording, corruption):
    recording.create()
    path = next(recording.cache.glob("audio/*.json"))
    entry = json.loads(path.read_text(encoding="utf-8"))
    rows = entry["words"]
    if corruption == "truncated":
        path.write_text('{"words":', encoding="utf-8")
    else:
        if corruption == "version":
            entry["version"] = -1
        elif corruption == "missing":
            rows.pop()
        elif corruption == "extra":
            rows.append(rows[-1])
        elif corruption == "wrong-word":
            rows[0][0] = "OTHER"
        elif corruption == "nan":
            rows[0][1] = float("nan")
        elif corruption == "infinity":
            rows[0][2] = float("inf")
        elif corruption == "negative":
            rows[0][1] = -1
        elif corruption == "reversed":
            rows[0][2] = 0
        elif corruption == "overlap":
            rows[1][1] = 0.5
        elif corruption == "past-end":
            rows[-1][2] = 20
        elif corruption == "non-numeric":
            rows[0][1] = "soon"
        elif corruption == "shape":
            rows[0].pop()
        elif corruption == "empty":
            entry["words"] = []
        elif corruption == "boolean":
            rows[0][1] = False
        elif corruption == "huge":
            rows[0][1] = 10**1000
        path.write_text(json.dumps(entry), encoding="utf-8")
    recording.create()
    assert len(recording.calls) == 2
    recording.create()
    assert len(recording.calls) == 2


def test_incomplete_fresh_alignment_is_not_cached_and_closes_reader(
    recording, monkeypatch
):
    monkeypatch.setattr(
        audio_utils,
        "align_large_audio_torchaudio_robust",
        lambda *args, **kwargs: [["HELLO", 0.1, 0.9]],
    )
    with pytest.raises(AudioTranscriptMismatchError, match="complete, ordered"):
        recording.create()
    assert not list(recording.cache.glob("audio/*.json"))
    assert recording.clips[-1].closed


def test_inputs_changed_during_alignment_are_not_published(recording, monkeypatch):
    align = audio_utils.align_large_audio_torchaudio_robust

    def changing(*args, **kwargs):
        result = align(*args, **kwargs)
        recording.audio.write_bytes(b"edited during alignment")
        return result

    monkeypatch.setattr(audio_utils, "align_large_audio_torchaudio_robust", changing)
    with pytest.raises(AudioTranscriptMismatchError, match="changed during"):
        recording.create()
    assert not list(recording.cache.glob("audio/*.json"))
    assert recording.clips[-1].closed


def test_failed_cache_publication_leaves_no_partial_entry(recording, monkeypatch):
    def fail(*args):
        raise OSError("replacement failed")

    monkeypatch.setattr(audio_utils.os, "replace", fail)
    with pytest.raises(OSError, match="replacement failed"):
        recording.create()
    assert not list(recording.cache.glob("audio/*"))
    assert recording.clips[-1].closed
