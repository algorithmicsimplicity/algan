"""Viewer audio mixing and authenticated HTTP, without rendering or TTS."""

from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from moviepy import AudioClip

from algan.viewer.audio import SceneAudio
from algan.viewer.server import ViewerServer


def clip(value, duration=1):
    value = np.atleast_1d(value)
    return AudioClip(
        lambda t: np.broadcast_to(value, np.shape(t) + value.shape), duration=duration
    )


def effect(audio, start=0):
    return SimpleNamespace(audio_clip=audio, start_time_func=lambda: start)


def pcm(data):
    with wave.open(io.BytesIO(data), "rb") as reader:
        rate = reader.getframerate()
        samples = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
        return rate, samples.reshape(-1, reader.getnchannels()) / 32768


def test_mix_preserves_offsets_overlap_channels_silence_and_duration():
    track = SceneAudio(
        [effect(clip(0.125), 0.25), effect(clip([0.25, -0.25]), 0.75)], 2, 8000
    )
    rate, samples = pcm(track.wav())
    assert rate == 8000
    assert samples.shape == (16000, 2)
    np.testing.assert_array_equal(samples[:2000], 0)
    np.testing.assert_array_equal(samples[2000:6000], 0.125)
    np.testing.assert_array_equal(samples[6000:10000], [[0.375, -0.125]] * 4000)
    np.testing.assert_array_equal(samples[10000:14000], [[0.25, -0.25]] * 4000)
    np.testing.assert_array_equal(samples[14000:], 0)


def test_opening_snapshot_freezes_effects_starts_and_clip_duration():
    start = [0.5]
    audio = clip(0.25)
    effects = [SimpleNamespace(audio_clip=audio, start_time_func=lambda: start[0])]
    track = SceneAudio(effects, 2, 8000)
    effects.append(effect(clip(0.5)))
    start[0] = 0
    audio.duration = 0
    _, samples = pcm(track.wav())
    np.testing.assert_array_equal(samples[:4000], 0)
    np.testing.assert_array_equal(samples[4000:12000], 0.25)
    np.testing.assert_array_equal(samples[12000:], 0)


def test_negative_starts_and_ends_beyond_the_view_are_clipped():
    track = SceneAudio([effect(clip(0.25), -0.5), effect(clip(0.5), 0.75)], 1, 8000)
    _, samples = pcm(track.wav())
    assert samples.shape == (8000, 1)
    np.testing.assert_array_equal(samples[:4000], 0.25)
    np.testing.assert_array_equal(samples[4000:6000], 0)
    np.testing.assert_array_equal(samples[6000:], 0.5)


def test_only_overlapping_effects_advertise_audio():
    for effects in ([], [effect(clip(1), 2)], [effect(clip(1), -2)]):
        track = SceneAudio(effects, 1, 8000)
        assert not track.available
        assert track.wav() is None


def test_mono_vectors_do_not_broadcast_into_a_quadratic_array():
    mono = AudioClip(lambda t: np.ones_like(t) * 0.125, duration=3)
    track = SceneAudio([effect(mono), effect(clip([0.25, -0.25], 3))], 3, 8000)
    _, samples = pcm(track.wav())
    assert samples.shape == (24000, 2)  # crosses the 16384-sample chunk boundary
    np.testing.assert_array_equal(samples, [[0.375, -0.125]] * 24000)


def test_pcm_clips_overload_instead_of_wrapping_and_replaces_nonfinite_samples():
    track = SceneAudio([effect(clip([2, -2]))], 0.01, 8000)
    _, samples = pcm(track.wav())
    np.testing.assert_array_equal(samples, [[32767 / 32768, -1]] * 80)
    track = SceneAudio([effect(clip([np.nan, np.inf]))], 0.01, 8000)
    _, samples = pcm(track.wav())
    np.testing.assert_array_equal(samples, [[0, 32767 / 32768]] * 80)


def test_sampling_is_lazy_cached_and_does_not_close_source_readers():
    calls = []
    audio = clip(0.25)
    get_frame = audio.get_frame
    audio.get_frame = lambda times: (calls.append(times), get_frame(times))[1]
    audio.close = lambda: pytest.fail("The Scene still owns this clip")
    track = SceneAudio([effect(audio)], 1, 8000)
    assert not calls
    first = track.wav()
    assert len(calls) == 1
    assert track.wav() is first
    assert len(calls) == 1
    track.close()
    with pytest.raises(RuntimeError, match="closed"):
        track.wav()
    assert audio.get_frame(0) == 0.25


def test_concurrent_requests_share_one_mix_and_close_cancels_at_a_chunk_boundary():
    from concurrent.futures import ThreadPoolExecutor

    entered = threading.Event()
    release = threading.Event()
    audio = clip(0.25, duration=3)
    get_frame = audio.get_frame
    calls = []

    def read(times):
        calls.append(times)
        entered.set()
        assert release.wait(3)
        return get_frame(times)

    audio.get_frame = read
    track = SceneAudio([effect(audio)], 3, 8000)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(track.wav)
        assert entered.wait(3)
        second = executor.submit(track.wav)
        release.set()
        assert first.result(3) is second.result(3)
    assert len(calls) == 2

    entered.clear()
    release.clear()
    track = SceneAudio([effect(audio)], 3, 8000)
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = executor.submit(track.wav)
        assert entered.wait(3)
        closed = executor.submit(track.close)
        assert track._closed.wait(3)
        release.set()
        with pytest.raises(RuntimeError, match="closed"):
            pending.result(3)
        closed.result(3)


def test_http_audio_is_authenticated_scoped_and_served_as_wav():
    track = SceneAudio([effect(clip(0.25))], 1, 8000)
    versions = []
    session = SimpleNamespace(audio=track.wav)

    def resolve(version):
        versions.append(version)
        if version != "7":
            raise ValueError("The selected scene changed")
        return session

    catalogue = SimpleNamespace(session_for_request=resolve)
    server = ViewerServer(catalogue)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(server.url_for("/audio.wav?s=7"), timeout=3) as r:
            assert r.headers["Content-Type"] == "audio/wav"
            assert r.headers["Cache-Control"] == "no-store"
            assert r.read() == track.wav()
        assert versions == ["7"]
        for url, code in (
            (server.origin + "/audio.wav?s=7", 403),
            (server.url_for("/audio.wav?s=6"), 400),
        ):
            with pytest.raises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(url, timeout=3)
            assert caught.value.code == code
        assert versions == ["7", "6"], "unauthenticated requests never access audio"
        track.available = False
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(server.url_for("/audio.wav?s=7"), timeout=3)
        assert caught.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        track.close()


def test_scene_view_uses_speech_and_audio_without_resynthesis_or_scene_lock(
    fresh_scene, monkeypatch
):
    from algan import PREVIEW, Audio, Scene, Speech
    from algan.viewer.session import ViewerSession

    monkeypatch.setattr(ViewerSession, "_run", lambda self: None)
    scene = Scene.current()
    speech_calls = []

    def speech(text):
        speech_calls.append(text)
        return clip(0.25, duration=0.5)

    scene.audio_manager.set_speech_source(speech)
    Scene.wait(0.25)
    with Speech("hello", wait_at_end=0):
        Scene.wait(0.5)
    with Audio(clip(0.5, duration=0.5)):
        Scene.wait(0.5)
    with Scene.view(
        PREVIEW.set(audio_sample_rate=8000), block=False, open_browser=False
    ) as handle:
        with handle.session._scene_lock:
            with urllib.request.urlopen(handle.url_for("/api/state"), timeout=3) as r:
                assert json.load(r)["has_audio"] is True
            with urllib.request.urlopen(handle.url_for("/audio.wav"), timeout=3) as r:
                before = r.read()
        assert speech_calls == ["hello"]
        rate, samples = pcm(before)
        assert rate == 8000
        np.testing.assert_array_equal(samples[:2000], 0)
        np.testing.assert_array_equal(samples[2000:6000], 0.25)
        np.testing.assert_array_equal(samples[6000:10000], 0.5)
        with Speech("authored later", wait_at_end=0):
            Scene.wait(0.5)
        handle.session.set_resolution("SMOKE_TEST")
        scene.reset()
        assert handle.session.audio() == before
        assert speech_calls == ["hello", "authored later"]


def test_scene_view_without_audio_stays_silent(fresh_scene, monkeypatch):
    from algan import PREVIEW, Scene
    from algan.viewer.session import ViewerSession

    monkeypatch.setattr(ViewerSession, "_run", lambda self: None)
    with Scene.view(PREVIEW, block=False, open_browser=False) as handle:
        assert handle.session.state()["has_audio"] is False
        assert handle.session.audio() is None
