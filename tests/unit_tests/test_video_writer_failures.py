"""Encoder failures and cancellation must release the video pipeline."""

from __future__ import annotations

import contextlib
import threading
import time
import warnings
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


def test_a_held_frame_is_one_queue_entry_written_its_repeat_count(workers):
    written = []
    worker = render_loop._VideoWriter(
        SimpleNamespace(write_frame=lambda array: written.append(array))
    )
    worker.start()
    worker.put(_Frame(1))
    # Far more repeats than the queue holds entries: must not block.
    worker.put(_Frame(2), 500)
    worker.put(_Frame(3))
    worker.finish()
    assert [int(a) for a in written] == [1] + [2] * 500 + [3]


def test_an_encoder_that_cannot_start_raises_its_own_error(tmp_path):
    """A TypeError starting the writer was retried with moviepy 1's spelling,
    ``withmask``, which the moviepy Algan requires refuses: the error shown
    was about that keyword, not what had gone wrong.
    """
    from algan.utils.algan_utils import get_file_writer

    with pytest.raises(TypeError, match="not int"):
        get_file_writer(
            str(tmp_path / "clip.mp4"), (16, 16), "libx264", 2, False,
            ["-crf", 18], None, None,
        )  # fmt: skip


def test_a_destination_that_cannot_be_replaced_keeps_the_finished_video(
    workers, tmp_path, monkeypatch
):
    """Windows refuses to replace a video another program has open. Publishing
    used to delete the old file first and then rename, and the caller's
    cleanup deleted the temporary, so the finished render was lost.
    """

    def locked(source, target):
        raise PermissionError(13, "Access is denied", target)

    monkeypatch.setattr(render_loop.os, "replace", locked)
    errors, source, target = _render(_Scene(3), _Writer(), tmp_path)

    (error,) = errors
    assert isinstance(error, PermissionError)
    assert str(source) in str(error)
    assert "open" in str(error)
    assert error.algan_kept_video == str(source)
    assert source.read_bytes() == b"new video"
    assert target.read_bytes() == b"existing video"


def test_save_video_leaves_a_video_it_could_not_publish_in_place(tmp_path, monkeypatch):
    from algan import SMOKE_TEST, Square
    from algan.scene_manager import SceneManager

    scene = SceneManager.instance().current_scene
    Square().spawn()
    kept = []

    def render(file_writer, file_path, file_path_out, **_kwargs):
        file_writer.close()
        with open(file_path, "wb") as handle:
            handle.write(b"finished")
        kept.append(file_path)
        error = PermissionError("destination is open elsewhere")
        error.algan_kept_video = file_path
        raise error

    monkeypatch.setattr(scene, "_render_to_video", render)
    with pytest.raises(PermissionError, match="open elsewhere"):
        scene.save_video(tmp_path / "clip.mp4", SMOKE_TEST)
    (path,) = kept
    with open(path, "rb") as handle:
        assert handle.read() == b"finished"


def test_an_odd_sized_h264_video_says_it_will_not_play_in_a_browser(
    tmp_path, monkeypatch
):
    """With an odd side moviepy passes libx264 no pixel format, and FFmpeg
    picks 4:4:4, which browsers and QuickTime refuse; nothing said so.
    """
    from algan.errors import UnsupportedFeatureWarning
    from algan.utils import algan_utils

    monkeypatch.setattr(algan_utils, "_WARNED_ODD_H264", set())
    params = ["-crf", "17"]
    with pytest.warns(UnsupportedFeatureWarning, match="34x34"):
        algan_utils._warn_if_h264_cannot_be_420((33, 33), "libx264", params)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        algan_utils._warn_if_h264_cannot_be_420((33, 33), "libx264", params)
        algan_utils._warn_if_h264_cannot_be_420((32, 32), "libx264", params)
        algan_utils._warn_if_h264_cannot_be_420((33, 32), "prores_ks", params)
        algan_utils._warn_if_h264_cannot_be_420(
            (35, 32), "libx264", ["-pix_fmt", "yuv444p"]
        )
    with pytest.warns(UnsupportedFeatureWarning, match="height"):
        writer = algan_utils.get_file_writer(
            str(tmp_path / "odd.mp4"), (48, 31), "libx264", 2, False,
            params, None, None,
        )  # fmt: skip
    writer.close()


def test_save_video_temporaries_are_its_own_and_hidden(tmp_path, monkeypatch):
    """``clip.mp4`` rendered through ``clip_temp.mp4`` and ``clip_temp.wav``
    and then deleted both, whoever had made them, and two processes rendering
    one destination wrote the same temporary.
    """
    import os

    from algan import SMOKE_TEST, Square
    from algan.scene_manager import SceneManager

    scene = SceneManager.instance().current_scene
    Square().spawn()
    for name in ("clip_temp.mp4", "clip_temp.wav"):
        (tmp_path / name).write_bytes(b"the user's own")
    used = []

    def render(file_writer, file_path, file_path_out, **_kwargs):
        file_writer.close()
        used.append(file_path)
        os.replace(file_path, file_path_out)

    monkeypatch.setattr(scene, "_render_to_video", render)
    scene.save_video(tmp_path / "clip.mp4", SMOKE_TEST)
    (path,) = used
    assert os.path.basename(path) == f".clip.{os.getpid()}.tmp.mp4"
    for name in ("clip_temp.mp4", "clip_temp.wav"):
        assert (tmp_path / name).read_bytes() == b"the user's own"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "clip.mp4",
        "clip_temp.mp4",
        "clip_temp.wav",
    ]
