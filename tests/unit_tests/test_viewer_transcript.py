"""Transcript HTTP snapshots, without starting the frame-rendering worker."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from algan import PREVIEW, Scene, Speech
from algan.viewer.session import ViewerSession


def get(handle, path):
    with urllib.request.urlopen(handle.url_for(path), timeout=3) as response:
        return json.load(response)


def test_transcript_is_frozen_at_view_and_does_not_wait_for_the_renderer(
    fresh_scene, monkeypatch
):
    monkeypatch.setattr(ViewerSession, "_run", lambda self: None)
    scene = Scene.current()
    scene.audio_manager.set_speech_source(
        lambda text: SimpleNamespace(
            duration=2,
            algan_word_timestamps=[("HELLO", 0.1, 0.5), ("WORLD", 1.0, 1.5)],
        )
    )
    Scene.wait(2)
    with Speech("Hello, world!", wait_at_end=1):
        Scene.wait(1)
    with Scene.view(PREVIEW, block=False, open_browser=False) as handle:
        # The HTTP handler runs on another thread; taking the scene lock here
        # would time out this request if transcript retrieval tried to render.
        with handle.session._scene_lock:
            before = get(handle, "/api/transcript")
        assert before["text"] == "Hello, world!\n\n"
        row = before["blocks"][0]
        assert row["words"][0]["start"] == pytest.approx(2.1)
        assert row["words"][-1]["end"] == pytest.approx(3.5)
        assert row["timing"] == "aligned"

        with Speech("authored after view"):
            Scene.wait(1)
        assert len(scene.audio_manager._speech_blocks) == 2
        assert get(handle, "/api/transcript") == before
        handle.session.set_resolution("SMOKE_TEST")
        assert get(handle, "/api/transcript") == before
        scene.reset()
        assert get(handle, "/api/transcript") == before
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(handle._server.origin + "/api/transcript", timeout=3)
        assert caught.value.code == 403


def test_empty_scene_serves_an_empty_transcript_and_the_tab_assets(
    fresh_scene, monkeypatch
):
    monkeypatch.setattr(ViewerSession, "_run", lambda self: None)
    with Scene.view(PREVIEW, block=False, open_browser=False) as handle:
        assert get(handle, "/api/transcript") == {"text": "", "blocks": []}
        with urllib.request.urlopen(handle.url, timeout=3) as response:
            html = response.read().decode()
        assert 'id="fragments-tab"' in html
        assert 'id="transcript-tab"' in html
        assert "/static/transcript.js" in html
        with urllib.request.urlopen(
            handle.url_for("/static/transcript.js"), timeout=3
        ) as response:
            assert b"class TranscriptView" in response.read()
