"""Speech/Scene/HTTP integration; never synthesize speech or render frames."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from algan import Audio, Off, Scene, Seq, Speech
from algan.viewer.session import ViewerSession


def source(script):
    return SimpleNamespace(duration=2.0)


@pytest.mark.fast
def test_speech_cues_share_the_resolved_audio_clock(fresh_scene):
    scene = Scene.current()
    scene.audio_manager.set_speech_source(source)
    scene.wait(1)
    with Seq(runtime=12):
        scene.wait(1)
        with Speech("one two", wait_at_end=1):
            scene.wait(1)
    cue = scene.audio_manager._speech_cues[0]
    effect = scene.effects[-1]
    assert cue.start_time_func is effect.start_time_func
    block = scene.audio_manager._transcript_snapshot()["blocks"][0]
    assert block["words"][0]["start"] == pytest.approx(effect.start_time_func())
    assert block["words"][-1]["end"] == pytest.approx(effect.start_time_func() + 2)
    assert scene.audio_manager.video_transcript == "one two\n\n"


@pytest.mark.fast
def test_speech_is_scene_local_and_reset_clears_cues(fresh_scene):
    first = Scene.current()
    first.audio_manager.set_speech_source(source)
    with Speech("first"):
        pass
    second = Scene()
    second.audio_manager.set_speech_source(source)
    with Speech("second"):
        pass
    assert first.audio_manager._transcript_snapshot()["text"] == "first"
    assert second.audio_manager._transcript_snapshot()["text"] == "second"
    second.reset()
    assert second.audio_manager._transcript_snapshot() == {"text": "", "blocks": []}


def test_audio_is_not_transcript_and_suppressed_speech_is_untimed(fresh_scene):
    scene = Scene.current()
    scene.audio_manager.set_speech_source(source)
    with Audio(source("")):
        pass
    with Off(), Speech("not sounded"):
        pass
    blocks = scene.audio_manager._transcript_snapshot()["blocks"]
    assert len(blocks) == 1
    assert blocks[0]["text"] == "not sounded"
    assert blocks[0]["timing"] == "unavailable"


def test_viewer_transcript_is_authenticated_and_frozen_at_launch(
    fresh_scene, monkeypatch
):
    scene = Scene.current()
    scene.audio_manager.set_speech_source(source)
    with Speech("before viewer"):
        pass
    monkeypatch.setattr(ViewerSession, "_run", lambda self: None)
    with scene.view(block=False, open_browser=False) as handle:
        # The route is just a snapshot; the render lock need never be acquired.
        with (
            handle.session._scene_lock,
            urllib.request.urlopen(
                handle.url_for("/api/transcript"), timeout=3
            ) as response,
        ):
            payload = json.load(response)
        with Speech("after viewer"):
            pass
        assert payload["text"] == "before viewer"
        assert handle.session.transcript() == payload
        payload["blocks"].clear()
        assert len(handle.session.transcript()["blocks"]) == 1
        bare = handle.url.split("?")[0].rstrip("/") + "/api/transcript"
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(bare, timeout=3)
        assert exc.value.code == 403
        with urllib.request.urlopen(handle.url_for("/"), timeout=3) as response:
            page = response.read().decode()
        assert 'aria-controls="transcript-panel"' in page
        assert 'id="fragments"' in page


def test_recorded_speech_preserves_alignment_relative_to_padded_clip(
    tmp_path, monkeypatch
):
    import moviepy

    from algan.utils import audio_utils

    rows = [("HELLO", 1, 1.2), ("WELL", 2, 2.3), ("KNOWN", 2.4, 2.7), ("WORLD", 3, 3.3)]
    recording = SimpleNamespace(duration=10)
    recording.subclipped = lambda start, end: SimpleNamespace(duration=end - start)
    monkeypatch.setattr(moviepy, "AudioFileClip", lambda path: recording)
    monkeypatch.setattr(
        audio_utils,
        "SETTINGS",
        SimpleNamespace(paths=SimpleNamespace(cache_directory=str(tmp_path))),
    )
    monkeypatch.setattr(
        audio_utils, "align_large_audio_torchaudio_robust", lambda *args: rows
    )
    generator = audio_utils.get_speech_generator_from_file("recording.wav", "words.txt")
    sound = generator("Hello,\nwell-known world!")
    assert [row[0] for row in sound._algan_word_timings] == [row[0] for row in rows]
    assert sound._algan_word_timings[0][1] == pytest.approx(0.05)
    assert sound._algan_word_timings[-1][2] == pytest.approx(2.35)
    # A second source uses the existing CSV, not another alignment pass.
    monkeypatch.setattr(
        audio_utils,
        "align_large_audio_torchaudio_robust",
        lambda *args: pytest.fail("alignment cache was ignored"),
    )
    cached = audio_utils.get_speech_generator_from_file("recording.wav", "words.txt")
    assert (
        cached("Hello well-known world")._algan_word_timings
        == sound._algan_word_timings
    )


def test_recorded_source_fallback_keeps_original_text(tmp_path, monkeypatch):
    import moviepy

    from algan.utils import audio_utils

    monkeypatch.setattr(
        moviepy, "AudioFileClip", lambda path: SimpleNamespace(duration=10)
    )
    monkeypatch.setattr(
        audio_utils,
        "SETTINGS",
        SimpleNamespace(paths=SimpleNamespace(cache_directory=str(tmp_path))),
    )
    monkeypatch.setattr(
        audio_utils,
        "align_large_audio_torchaudio_robust",
        lambda *args: [("OTHER", 0, 1)],
    )
    monkeypatch.setattr(audio_utils, "get_pyttsx_speech_generator", lambda text: text)
    generator = audio_utils.get_speech_generator_from_file("recording.wav", "words.txt")
    assert generator("Missing-words\nHERE!") == "Missing-words\nHERE!"
    assert generator("...") == "..."  # empty normalized pattern must not IndexError
