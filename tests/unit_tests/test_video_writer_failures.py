"""Encoder failures and cancellation must release the video pipeline."""

from __future__ import annotations

import contextlib
import threading
import time
from queue import Empty
from types import SimpleNamespace

import pytest

from algan import render_loop


class _Frame:
    def __init__(self, index):
        self.index = index

    def numpy(self):
        return self.index


class _Writer:
    def __init__(self, *, fail_at=None, close_error=None):
        self.fail_at = fail_at
        self.close_error = close_error
        self.error = BrokenPipeError("encoder pipe failed")
        self.frames = []
        self.closed = False
        self.closed_while_writing = False
        self.thread = None

    def write_frame(self, frame):
        self.thread = threading.current_thread()
        assert not self.closed, "frame written after encoder close"
        if frame == self.fail_at:
            raise self.error
        self.frames.append(frame)

    def close(self):
        self.closed_while_writing = self.thread is not None and self.thread.is_alive()
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class _Scene(render_loop.RenderLoopMixin):
    def __init__(self, count, *, render_error=None, fail_after=0):
        self.count = count
        self.render_error = render_error
        self.fail_after = fail_after
        self.scene_times = [[0, 0]]
        self.frames_per_second = 1
        self.frames_closed = False
        self.animation_manager = SimpleNamespace(
            context=SimpleNamespace(prev_context=None)
        )
        self.timeline_manager = SimpleNamespace(
            preserving_authoring_state=lambda **kwargs: contextlib.nullcontext()
        )

    def _recorded_end_time_for_render(self):
        return self.count

    def _initialize_frames(self):
        pass

    def _scene_has_renderable_actors(self, *_):
        return True

    def get_frames(self, *_args, **_kwargs):
        try:
            for index in range(self.count):
                if self.render_error is not None and index == self.fail_after:
                    raise self.render_error
                yield [_Frame(index)]
        finally:
            self.frames_closed = True


@pytest.fixture
def workers(monkeypatch):
    instances = []
    original = render_loop._VideoWriter

    class TrackedWriter(original):
        def __init__(self, file_writer):
            super().__init__(file_writer)
            instances.append(self)

    monkeypatch.setattr(render_loop, "_VideoWriter", TrackedWriter)
    monkeypatch.setattr(
        render_loop,
        "_render_progress",
        lambda count: contextlib.nullcontext(lambda: None),
    )
    yield instances
    for worker in instances:
        worker.abort()


def _render(scene, writer, tmp_path):
    """Bound the test itself if a regression reintroduces a blocked producer."""
    source = tmp_path / "temporary.mp4"
    target = tmp_path / "output.mp4"
    source.write_bytes(b"new video")
    target.write_bytes(b"existing video")
    errors = []

    def run():
        try:
            scene._render_to_video_impl(
                writer,
                str(source),
                str(target),
                despawn_camera_and_lights=False,
                preserve_authoring_state=True,
            )
        except BaseException as exc:
            errors.append(exc)

    producer = threading.Thread(target=run, daemon=True)
    producer.start()
    producer.join(timeout=5)
    stuck = producer.is_alive()
    if stuck:
        # The fake stream is finite. Free its queue so a broken implementation
        # can still exit; there is no real encoder or render process to kill.
        deadline = time.monotonic() + 2
        while producer.is_alive() and time.monotonic() < deadline:
            with contextlib.suppress(AttributeError, Empty):
                scene.frame_queue.get_nowait()
            producer.join(timeout=0.01)
    assert not stuck, "video producer did not report the encoder failure promptly"
    return errors, source, target


@pytest.mark.parametrize(("count", "fail_at"), [(1, 0), (40, 0), (40, 12), (4, 3)])
def test_encoder_failure_reaches_producer_without_replacing_output(
    workers, tmp_path, count, fail_at
):
    scene = _Scene(count)
    writer = _Writer(fail_at=fail_at)
    errors, source, target = _render(scene, writer, tmp_path)

    assert errors == [writer.error]
    assert target.read_bytes() == b"existing video"
    assert source.exists()
    assert writer.closed
    assert not writer.closed_while_writing
    assert scene.frames_closed
    assert scene.scene_times == [[0, 0]]
    assert not workers[0]._thread.is_alive()
    assert workers[0].queue.empty()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("fail_after", [0, 4])
def test_render_failure_or_cancellation_stops_worker_before_close(
    workers, tmp_path, error_type, fail_after
):
    error = error_type("render stopped")
    scene = _Scene(12, render_error=error, fail_after=fail_after)
    writer = _Writer(close_error=OSError("cleanup also failed"))
    errors, _, target = _render(scene, writer, tmp_path)

    assert errors == [error]
    assert writer.closed
    assert not writer.closed_while_writing
    assert scene.frames_closed
    assert scene.scene_times == [[0, 0]]
    assert target.read_bytes() == b"existing video"
    assert not workers[0]._thread.is_alive()
    assert workers[0].queue.empty()


def test_success_flushes_every_frame_before_closing_and_publishing(workers, tmp_path):
    scene = _Scene(40)
    writer = _Writer()
    errors, source, target = _render(scene, writer, tmp_path)

    assert errors == []
    assert writer.frames == list(range(40))
    assert writer.closed
    assert not writer.closed_while_writing
    assert scene.frames_closed
    assert scene.scene_times == [[0, 0]]
    assert not source.exists()
    assert target.read_bytes() == b"new video"
    assert not workers[0]._thread.is_alive()


def test_close_failure_is_reported_without_publishing_output(workers, tmp_path):
    error = OSError("encoder finalization failed")
    writer = _Writer(close_error=error)
    errors, _, target = _render(_Scene(3), writer, tmp_path)

    assert errors == [error]
    assert target.read_bytes() == b"existing video"
    assert not workers[0]._thread.is_alive()


def test_preservation_setup_failure_still_stops_started_worker(workers, tmp_path):
    error = RuntimeError("cannot preserve timeline")
    scene = _Scene(3)

    def fail(**_kwargs):
        raise error

    scene.timeline_manager.preserving_authoring_state = fail
    writer = _Writer()
    errors, _, _ = _render(scene, writer, tmp_path)

    assert errors == [error]
    assert writer.closed
    assert scene.scene_times == [[0, 0]]
    assert not workers[0]._thread.is_alive()


def test_encoder_failure_wakes_a_producer_waiting_on_a_full_queue(workers):
    entered = threading.Event()
    release = threading.Event()
    error = BrokenPipeError("encoder failed with a backlog")

    def write_frame(_frame):
        entered.set()
        assert release.wait(3), "test did not release the encoder"
        raise error

    worker = render_loop._VideoWriter(SimpleNamespace(write_frame=write_frame))
    worker.start()
    worker.put(_Frame(0))
    errors = []

    def produce():
        try:
            worker.put(_Frame(9))
        except BaseException as exc:
            errors.append(exc)

    producer = threading.Thread(target=produce, daemon=True)
    try:
        assert entered.wait(2)
        for index in range(8):
            worker.put(_Frame(index + 1))
        assert worker.queue.full()
        producer.start()
        producer.join(timeout=0.15)
        assert producer.is_alive(), "producer should be waiting for queue space"
        release.set()
        producer.join(timeout=2)
        assert not producer.is_alive()
        assert errors == [error]
    finally:
        release.set()
        worker.abort()
        if producer.ident is not None:
            producer.join(timeout=2)
