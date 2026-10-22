"""Frame-local video batches render each frame alike in any batch window.

A video batch normally makes some choices across all its frames -- circuit
chord counts and outline bounds are the batch's widest, replay runs over the
whole window -- so where the window boundaries fall can move an animated
frame's pixels. ``frame_local_video_batches`` makes those choices per frame,
as still batches do, and the output then no longer depends on the windows.
"""

from __future__ import annotations

import pytest
import torch

import algan.render_loop as render_loop
import algan.rendering.raytracing.settings as rt_settings
from algan import RED, RIGHT, UP, Circle, Off, SceneManager, Square, Sync
from algan.settings.video_settings import VideoSettings

SMALL = VideoSettings((64, 36), 10, supersampling=1)


@pytest.fixture(autouse=True)
def restore_switches():
    previous = (rt_settings.frame_local_video_batches, rt_settings.first_batch_frames)
    SceneManager.reset()
    yield
    rt_settings.frame_local_video_batches, rt_settings.first_batch_frames = previous
    SceneManager.reset()


def _author():
    scene = SceneManager.reset()
    scene.set_video_settings(SMALL)
    circle = Circle(color=RED).scale(0.1)
    square = Square().scale(0.3).move(RIGHT * 2)
    with Off():
        circle.spawn()
        square.spawn()
    # A circle growing from a few pixels to most of the frame takes more
    # chords every frame, so a batch-wide chord count depends on the window.
    with Sync(runtime=1.2):
        circle.scale(12)
        square.rotate(90, UP)
    return scene


def _render(tmp_path, monkeypatch, name, *, frame_local, first_batch_frames):
    rt_settings.frame_local_video_batches = frame_local
    rt_settings.first_batch_frames = first_batch_frames
    scene = _author()
    frames, windows, flags = [], [], []
    original_put = render_loop._VideoWriter.put
    original_batch = render_loop.RenderLoopMixin._render_primitive_batch

    def put(writer, frame, copies=1):
        if frame is not None:
            frames.extend(frame.clone() for _ in range(copies))
        return original_put(writer, frame, copies)

    def batch(self, primitives, start_ind, end_ind, *args, **kwargs):
        windows.append((int(start_ind), int(end_ind)))
        flags.append((self._frame_local_batches, self._still_batches))
        return original_batch(self, primitives, start_ind, end_ind, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setenv("ALGAN_PREFETCH_BATCHES", "1")
        patch.setattr(render_loop._VideoWriter, "put", put)
        patch.setattr(render_loop.RenderLoopMixin, "_render_primitive_batch", batch)
        scene.save_video(tmp_path / f"{name}.mp4", SMALL, post_processes=())
    return frames, windows, flags


def test_frame_local_video_frames_do_not_depend_on_the_batch_windows(
    tmp_path, monkeypatch
):
    whole, whole_windows, flags = _render(
        tmp_path, monkeypatch, "whole", frame_local=True, first_batch_frames=0
    )
    split, split_windows, _ = _render(
        tmp_path, monkeypatch, "split", frame_local=True, first_batch_frames=3
    )

    assert len(whole_windows) == 1
    assert split_windows[0] == (0, 3)
    assert len(split_windows) == 2
    # Frame-local preparation, but not a still job's per-frame signature.
    assert set(flags) == {(True, False)}
    assert len(split) == len(whole) > 3
    for index, (expected, actual) in enumerate(zip(whole, split)):
        assert torch.equal(expected, actual), f"frame {index} differs"


def test_video_batches_are_batch_wide_by_default(tmp_path, monkeypatch):
    _frames, windows, flags = _render(
        tmp_path, monkeypatch, "default", frame_local=False, first_batch_frames=0
    )

    assert len(windows) == 1
    assert set(flags) == {(False, False)}


@pytest.mark.parametrize("frame_local", [False, True])
def test_a_still_job_is_frame_local_and_signed_either_way(
    tmp_path, monkeypatch, frame_local
):
    rt_settings.frame_local_video_batches = frame_local
    scene = _author()
    flags = []
    original_batch = render_loop.RenderLoopMixin._render_primitive_batch

    def batch(self, primitives, start_ind, end_ind, *args, **kwargs):
        flags.append((self._frame_local_batches, self._still_batches))
        return original_batch(self, primitives, start_ind, end_ind, *args, **kwargs)

    monkeypatch.setattr(render_loop.RenderLoopMixin, "_render_primitive_batch", batch)
    scene.save_frame(tmp_path / "still.png", SMALL, at=[0.3, 0.9])

    assert flags
    assert set(flags) == {(True, True)}


@pytest.mark.parametrize(
    ("frame_local", "still", "lone", "signature", "per_frame"),
    [
        (False, False, False, False, False),  # a video batch, as before
        (True, False, False, False, True),  # a frame-local video batch
        (True, True, False, True, True),  # a still batch
        (True, True, True, False, True),  # a still alone in its batch
    ],
)
def test_the_merge_is_told_how_to_decide_per_frame(
    monkeypatch, frame_local, still, lone, signature, per_frame
):
    import types

    from algan.rendering.raytracing import scene_builder, tracer

    calls = []

    def merge(primitives, **kwargs):
        calls.append(kwargs)
        return {"textures": torch.zeros(1)}

    monkeypatch.setattr(scene_builder, "_merge_scene", merge)
    monkeypatch.setattr(tracer, "_environment_radiance", lambda _scene: None)
    scene = render_loop.RenderLoopMixin.__new__(render_loop.RenderLoopMixin)
    scene._frame_local_batches = frame_local
    scene._still_batches = still
    primitive = types.SimpleNamespace(_rt_prepared_host_scene=None)
    state = {"lights": [], "light_objects": [], "light_active": []}
    if lone:
        state["lone_still"] = True

    scene._prepare_merged_host_scene([primitive], render_state=state)

    assert calls[0]["frame_signature"] is signature
    assert calls[0]["per_frame_decode"] is per_frame
