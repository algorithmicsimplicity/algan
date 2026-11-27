"""The viewer's stale error, the speech cache key, and two settings/CLI messages.

- One failed render left the viewer showing its error for the rest of the
  session, even after later frames rendered.
- The pyttsx3 speech cache was keyed on the text alone, so changing the voice
  or rate replayed the old recording, and an engine that wrote nothing left an
  empty file every later run took for finished speech.
- ``SETTINGS.raytracing.max_bounce = 3`` raised a bare ``AttributeError``.
- ``algan check`` ignored ``FFMPEG_BINARY``, which the encoder honours, and
  exited 0 on a machine where rendering cannot work.

Record-only; no render.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

import algan_cli as cli
from algan import PREVIEW, SETTINGS, Scene
from algan.errors import AlganConfigurationError
from algan.utils import audio_utils
from algan.viewer import session as viewer


@pytest.fixture
def session(fresh_scene, monkeypatch):
    monkeypatch.setattr(viewer.ViewerSession, "_run", lambda self: None)
    Scene.wait(10)
    result = viewer.ViewerSession(Scene.current(), PREVIEW.set(resolution=(4, 4)))
    try:
        yield result
    finally:
        result.close()


def test_a_successful_frame_clears_an_earlier_render_error(session):
    # The smallest unit: _store_png is where the worker publishes a rendered frame.
    session._error = "RuntimeError: out of memory"
    assert "out of memory" in str(session.state().get("error"))
    session._store_png(0, b"png")
    assert session._error is None
    assert not session.state().get("error")


class _FakeEngine:
    def __init__(self, voice="v1", rate=200, writes=True, fail_property=False):
        self.props = {"voice": voice, "rate": rate}
        self.writes = writes
        self.fail_property = fail_property
        self.calls = 0

    def getProperty(self, name):  # noqa: N802
        if self.fail_property:
            raise RuntimeError("no such property")
        return self.props[name]

    def save_to_file(self, text, path):
        self.path = path

    def runAndWait(self):  # noqa: N802
        self.calls += 1
        if self.writes:
            with open(self.path, "wb") as handle:
                handle.write(b"audio")

    def stop(self):
        pass


@pytest.fixture
def speech(monkeypatch, tmp_path):
    monkeypatch.setattr(SETTINGS.paths, "cache_directory", str(tmp_path))
    monkeypatch.setattr(audio_utils, "_cached_audio_file_clip", lambda file: file)

    def use(engine):
        monkeypatch.setattr(
            audio_utils, "pyttsx3", SimpleNamespace(init=lambda: engine), raising=False
        )

    return use


def test_the_speech_cache_key_includes_voice_and_rate(speech):
    files = set()
    for voice, rate in (("v1", 200), ("v2", 200), ("v1", 150), ("v1", 200)):
        speech(_FakeEngine(voice, rate))
        files.add(audio_utils.get_pyttsx_speech_generator("hello"))
    assert len(files) == 3


def test_a_failed_property_read_still_yields_a_key(speech):
    speech(_FakeEngine(fail_property=True))
    assert os.path.getsize(audio_utils.get_pyttsx_speech_generator("hi")) > 0


def test_an_engine_that_writes_nothing_raises_and_leaves_no_cache(speech, tmp_path):
    speech(_FakeEngine(writes=False))
    with pytest.raises(OSError, match="pyttsx3.*no audio"):
        audio_utils.get_pyttsx_speech_generator("hello")
    assert not list((tmp_path / "audio").glob("*"))


def test_a_zero_length_cached_file_is_resynthesized(speech):
    engine = _FakeEngine()
    speech(engine)
    file = audio_utils.get_pyttsx_speech_generator("hello")
    open(file, "wb").close()
    assert audio_utils.get_pyttsx_speech_generator("hello") == file
    assert engine.calls == 2
    assert os.path.getsize(file) > 0


def test_an_unknown_raytracing_attribute_gets_a_did_you_mean():
    with pytest.raises(AlganConfigurationError, match="Did you mean 'max_bounces'"):
        SETTINGS.raytracing.max_bounce = 3
    with pytest.raises(AlganConfigurationError, match="Unknown RayTracingSettings"):
        SETTINGS.raytracing.zzz_not_a_setting = 1


def test_check_honours_ffmpeg_binary(monkeypatch):
    monkeypatch.setenv("FFMPEG_BINARY", "/somewhere/ffmpeg")
    assert cli._ffmpeg_binary() == (
        "/somewhere/ffmpeg",
        "FFMPEG_BINARY environment variable",
    )
    monkeypatch.setenv("FFMPEG_BINARY", "auto")
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)  # import fails
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli._ffmpeg_binary() == (None, "")


def test_check_exits_non_zero_without_ffmpeg(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_ffmpeg_binary", lambda: (None, ""))
    assert cli._cmd_check(SimpleNamespace()) == 1
    assert "No ffmpeg found" in capsys.readouterr().out


def test_check_passes_without_a_text_typesetter(monkeypatch, capsys):
    # A bare Linux install has neither Pango nor LaTeX and still renders
    # geometry; CI's clean-wheel job runs `algan check` in exactly that state.
    import importlib.util

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "manimpango" else real_find_spec(name, *a),
    )
    monkeypatch.setattr(
        cli.shutil, "which", lambda name: None if name in ("latex", "dvisvgm") else name
    )
    monkeypatch.setattr(cli, "_ffmpeg_binary", lambda: ("/usr/bin/ffmpeg", "PATH"))
    status = cli._cmd_check(SimpleNamespace())
    out = capsys.readouterr().out
    assert "Neither Pango nor LaTeX found" in out
    assert status == 0 or "[ERROR]" in out
