"""Viewer caching is bounded by encoded bytes, without a GPU render."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from algan import PREVIEW, Scene
from algan.viewer import session as viewer


@pytest.fixture
def session(fresh_scene, monkeypatch):
    monkeypatch.setattr(viewer, "_frame_cache_budget", lambda: 10)
    monkeypatch.setattr(viewer.ViewerSession, "_run", lambda self: None)
    Scene.wait(10)
    result = viewer.ViewerSession(Scene.current(), PREVIEW.set(resolution=(4, 4)))
    try:
        yield result
    finally:
        result.close()


def _assert_accounted(session):
    assert session._cache_bytes == sum(map(len, session._cache.values()))
    assert session._cache_bytes <= session._cache_limit_bytes
    assert len(session._cache) <= viewer.MAX_CACHED_FRAMES


def _wait_for_requests(session, indices):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        with session._lock:
            if set(session._frame_waiters) == set(indices):
                return
        time.sleep(0.005)
    pytest.fail("frame requests did not register")


@pytest.mark.parametrize(
    ("available", "expected"), [(1024, 32), (1, 1), (1 << 40, 256 << 20)]
)
def test_budget_scales_with_available_memory_and_has_a_ceiling(
    monkeypatch, available, expected
):
    monkeypatch.setattr(
        viewer.psutil, "virtual_memory", lambda: SimpleNamespace(available=available)
    )
    assert viewer._frame_cache_budget() == expected


def test_speculation_stops_at_bytes_before_the_frame_count_cap(session):
    for index in range(3):
        assert session._store_png(index, b"abc")
    assert not session._store_png(3, b"defg")
    assert list(session._cache) == [0, 1, 2]
    _assert_accounted(session)
    assert session.state()["cached_bytes"] == 9
    assert session.state()["cache_limit_bytes"] == 10


def test_replacement_is_counted_once_and_survives_later_eviction(session, monkeypatch):
    session._cache_limit_bytes = 100
    monkeypatch.setattr(viewer, "MAX_CACHED_FRAMES", 3)
    session._wanted = 5
    session._store_png(0, b"old")
    session._store_png(1, b"one")
    session._store_png(0, b"newer")
    session._store_png(2, b"two")
    session._wanted = 3
    session._store_png(3, b"three")
    assert list(session._cache) == [0, 2, 3]
    assert session.frame(0) == b"newer"
    assert session._cache_bytes == 13
    _assert_accounted(session)


def test_cache_hits_refresh_eviction_order(session, monkeypatch):
    session._cache_limit_bytes = 100
    monkeypatch.setattr(viewer, "MAX_CACHED_FRAMES", 3)
    for index in range(3):
        session._store_png(index, b"abc")
    assert session.frame(0) == b"abc"
    session._wanted = 3
    session._store_png(3, b"new")
    assert list(session._cache) == [2, 0, 3]
    _assert_accounted(session)


def test_requested_frame_survives_speculation_and_larger_replacements(session):
    session._wanted = 2
    session._store_png(2, b"want")
    session._store_png(0, b"old!")
    assert not session._store_png(1, b"more")
    assert not session._store_png(0, b"too big")
    assert session.frame(2) == b"want"
    assert session.frame(0) == b"old!"
    _assert_accounted(session)


def test_a_seek_evicts_enough_old_bytes_to_retain_the_new_requested_frame(session):
    session._store_png(0, b"zero")
    session._store_png(1, b"one!")
    session._wanted = 99
    assert session._store_png(99, b"new frame")
    assert list(session._cache) == [99]
    assert session._cache_bytes == 9
    _assert_accounted(session)


def test_smaller_replacement_releases_its_old_byte_charge(session):
    session._store_png(0, b"12345678")
    session._store_png(0, b"12")
    assert session._store_png(1, b"34567890")
    _assert_accounted(session)
    assert session._cache_bytes == 10


def test_prefetch_chunk_length_uses_free_bytes_and_seek_still_works_when_full(
    session, monkeypatch
):
    calls = []

    def render(start, end, **kwargs):
        calls.append((start, end))
        for index in range(start, end):
            session._store_png(index, b"abcd")

    monkeypatch.setattr(session, "_render_range", render)
    session._store_png(0, b"abcd")
    session._requested_generation = session._generation
    assert session._render_next()
    assert calls == [(1, 2)]
    assert not session._render_next()
    session.prefetch(50)
    assert session._render_next()
    assert calls[-1] == (50, 51)
    assert session.frame(50) == b"abcd"
    _assert_accounted(session)


def test_oversized_frame_is_delivered_without_caching_or_repeated_prefetch(session):
    payload = b"larger than the cache"
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(session.frame, 0, timeout=3)
        _wait_for_requests(session, [0])
        assert not session._store_png(0, payload)
        assert pending.result(3) is payload
    assert not session._cache
    assert not session._frame_deliveries
    assert not session._frame_waiters
    assert not session._render_next()
    _assert_accounted(session)


def test_concurrent_requested_frames_are_delivered_even_when_only_one_fits(
    session, monkeypatch
):
    session._cache_limit_bytes = 4

    def render(start, end, **kwargs):
        session._store_png(start, bytes([start]) * 4)

    monkeypatch.setattr(session, "_render_range", render)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(session.frame, 0, timeout=3)
        _wait_for_requests(session, [0])
        second = executor.submit(session.frame, 1, timeout=3)
        _wait_for_requests(session, [0, 1])
        # Publish both before either waiter can consume the result. A cache
        # eviction must not swallow the first HTTP response.
        with session._lock:
            assert session._render_next()
            assert session._render_next()
            _assert_accounted(session)
        assert first.result(3) == bytes([0]) * 4
        assert second.result(3) == bytes([1]) * 4
    assert not session._frame_deliveries
    assert not session._frame_waiters


def test_oversized_playhead_does_not_abandon_other_pending_requests(
    session, monkeypatch
):
    session._cache_limit_bytes = 100
    session._store_png(99, b"seed")
    oversized = b"x" * 101

    def render(start, end, **kwargs):
        for index in range(start, end):
            payload = oversized if index == 51 else bytes([index]) * 4
            if not session._store_png(index, payload):
                break

    monkeypatch.setattr(session, "_render_range", render)
    with ThreadPoolExecutor(max_workers=3) as executor:
        first = executor.submit(session.frame, 50, timeout=3)
        _wait_for_requests(session, [50])
        second = executor.submit(session.frame, 0, timeout=3)
        _wait_for_requests(session, [50, 0])
        playhead = executor.submit(session.frame, 51, timeout=3)
        _wait_for_requests(session, [50, 0, 51])
        with session._lock:
            # The first chunk delivers both frame 50 and the uncacheable
            # playhead. Frame 0 still needs service before prefetch stops.
            assert session._render_next()
            assert session._render_next()
            assert not session._render_next()
        assert first.result(3) == bytes([50]) * 4
        assert second.result(3) == bytes([0]) * 4
        assert playhead.result(3) is oversized
    assert not session._frame_deliveries
    assert not session._frame_waiters
    _assert_accounted(session)


def test_exceeding_the_estimate_stops_encoding_and_closes_the_frame_generator(
    session, monkeypatch
):
    closed = threading.Event()

    def frames(start, end):
        try:
            yield torch.zeros((6, 2, 2, 3), dtype=torch.uint8)
        finally:
            closed.set()

    monkeypatch.setattr(session.scene, "get_frames", frames)
    monkeypatch.setattr(
        session, "_store", lambda index, frame: session._store_png(index, b"abcd")
    )
    settings = session.scene.video_settings
    with session._scene():
        session._render_range(0, 6, store=True)
    assert closed.is_set()
    assert list(session._cache) == [0, 1]
    assert session.scene.video_settings == settings
    _assert_accounted(session)


def test_resolution_change_and_close_release_all_cached_bytes(session):
    session._store_png(0, b"abcd")
    session.set_resolution("SMOKE_TEST")
    assert not session._cache
    assert session._frame_bytes_estimate == 0
    _assert_accounted(session)
    session._store_png(0, b"abcd")
    session.close()
    assert not session._cache
    assert not session._store_png(1, b"late")
    _assert_accounted(session)


def test_timed_out_waiters_do_not_retain_deliveries(session):
    with pytest.raises(TimeoutError):
        session.frame(0, timeout=0.01)
    assert not session._frame_waiters
    session._store_png(0, b"oversized PNG payload")
    assert not session._frame_deliveries
