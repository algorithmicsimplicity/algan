"""Synthesised narration reaches its cache whole or not at all.

``get_pyttsx_speech_generator`` caches each line's speech under a hash of its
text and synthesises only when that file is missing. It used to synthesise
straight into the cache path, so a Ctrl-C, a daemon cancel or a crash part way
through left a truncated file that every later run took for the finished one:
narration cut short, with the scene's timing fitted to it, or a decode error
that pointed nowhere near the cache. The engine here is a stand-in that writes
part of a file and is interrupted, or writes all of it.
"""

from __future__ import annotations

import os

import pytest

from algan import SETTINGS
from algan.utils import audio_utils


class _Engine:
    def __init__(self, interrupt):
        self.interrupt = interrupt
        self.target = None

    def save_to_file(self, script, path):
        self.target = path

    def runAndWait(self):  # noqa: N802  (pyttsx3's spelling)
        with open(self.target, "wb") as out:
            out.write(b"half a sentence")
            if self.interrupt:
                raise KeyboardInterrupt
            out.write(b" and the rest of it")

    def stop(self):
        pass


@pytest.fixture
def cache(tmp_path, monkeypatch):
    SETTINGS.paths.cache_directory = str(tmp_path)
    monkeypatch.setattr(audio_utils, "_cached_audio_file_clip", lambda file: file)
    return tmp_path / "audio"


def _synthesise(monkeypatch, interrupt):
    engine = _Engine(interrupt)
    monkeypatch.setattr(audio_utils.pyttsx3, "init", lambda: engine)
    return audio_utils.get_pyttsx_speech_generator("Hello there.")


def test_an_interrupted_synthesis_leaves_nothing_in_the_cache(cache, monkeypatch):
    with pytest.raises(KeyboardInterrupt):
        _synthesise(monkeypatch, interrupt=True)
    assert list(cache.iterdir()) == []

    # So the next run synthesises again, and caches the whole line.
    path = _synthesise(monkeypatch, interrupt=False)
    with open(path, "rb") as cached:
        assert cached.read() == b"half a sentence and the rest of it"
    assert [p.name for p in cache.iterdir()] == [os.path.basename(path)]
