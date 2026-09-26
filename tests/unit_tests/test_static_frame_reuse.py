"""Static frame reuse writes repeated frames again instead of rendering them.

The written video must be identical, frame for frame, to rendering every
frame; only frames the recording proves unchanged may be skipped.
"""

from __future__ import annotations

import pytest
import torch

import algan.render_loop as render_loop
import algan.rendering.raytracing.settings as rt_settings
from algan import RED, RIGHT, UP, Circle, Off, Scene, SceneManager, Seq, Square
from algan.settings.video_settings import VideoSettings

SMALL = VideoSettings((40, 24), 10, supersampling=1)


@pytest.fixture(autouse=True)
def restore_switch():
    previous = rt_settings.reuse_static_frames
    SceneManager.reset()
    yield
    rt_settings.reuse_static_frames = previous
    SceneManager.reset()


def _author(updater=False):
    scene = SceneManager.reset()
    scene.set_video_settings(SMALL)
    square = Square(color=RED).scale(0.4)
    with Off():
        square.spawn()
    with Seq(runtime=0.5):
        square.move(RIGHT)
    Scene.wait(1.0)
    with Off():
        # An instant change inside a hold must reach the frame after it.
        Circle().scale(0.3).move(UP).spawn()
    Scene.wait(0.6)
    if updater:
        follower = Circle().scale(0.1)
        with Off():
            follower.spawn()
        follower.add_updater(lambda mob, t: mob.set_non_recursive(location=RIGHT * t))
    with Seq(runtime=0.4):
        square.move(UP * 0.5)
    Scene.wait(0.5)
    return scene


def _render(tmp_path, monkeypatch, name, reuse, updater=False):
    rt_settings.reuse_static_frames = reuse
    scene = _author(updater)
    frames, windows = [], []
    original_put = render_loop._VideoWriter.put
    original_get_frames = render_loop.RenderLoopMixin.get_frames

    def put(writer, frame):
        if frame is not None:
            frames.append(frame.clone())
        return original_put(writer, frame)

    def get_frames(self, *args, **kwargs):
        windows.append(kwargs.get("frame_indices"))
        return original_get_frames(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(render_loop._VideoWriter, "put", put)
        patch.setattr(render_loop.RenderLoopMixin, "get_frames", get_frames)
        scene.save_video(tmp_path / f"{name}.mp4", SMALL, post_processes=())
    return frames, windows


@pytest.mark.parametrize("updater", [False, True])
def test_reused_frames_match_rendering_every_frame(tmp_path, monkeypatch, updater):
    _render(tmp_path, monkeypatch, "warm", False, updater)
    reference, dense = _render(tmp_path, monkeypatch, "off", False, updater)
    reused, sparse = _render(tmp_path, monkeypatch, "on", True, updater)
    assert dense == [None]
    assert len(sparse) == 1
    assert sparse[0] is not None
    assert len(sparse[0]) < len(reference)
    assert len(reused) == len(reference)
    for index, (expected, actual) in enumerate(zip(reference, reused)):
        assert torch.equal(expected, actual), f"frame {index} differs"


def test_frames_are_rendered_where_the_recording_changes(tmp_path, monkeypatch):
    rt_settings.reuse_static_frames = True
    scene = _author(updater=True)
    start = scene.scene_times[-1][0]
    end = round(scene._recorded_end_time_for_render() * scene.frames_per_second)
    frame_indices, repeats = scene._static_frame_runs(start, end, None, ())
    rendered = set(frame_indices)
    assert sum(repeats) == end - start
    fps = scene.frames_per_second
    # The whole first move, the frame of the instant spawn, and every frame
    # while the updater runs (from 2.1 s to the end) are rendered.
    assert set(range(0, int(0.5 * fps) + 1)) <= rendered
    assert int(round(1.5 * fps)) in rendered
    assert set(range(int(round(2.1 * fps)), end)) <= rendered
    # The holds in between are not.
    assert not set(range(int(0.5 * fps) + 2, int(round(1.5 * fps)))) & rendered


def test_reuse_is_off_for_the_path_tracer_and_callable_backgrounds(monkeypatch):
    rt_settings.reuse_static_frames = True
    scene = _author()
    end = round(scene._recorded_end_time_for_render() * scene.frames_per_second)
    assert scene._static_frame_runs(0, end, None, ()) is not None
    assert scene._static_frame_runs(0, end, lambda x, y, t: x, ()) is None
    assert (
        scene._static_frame_runs(0, end, None, (lambda frames, memory: frames,)) is None
    )
    monkeypatch.setattr(rt_settings, "samples_per_pixel", 4)
    assert scene._static_frame_runs(0, end, None, ()) is None
