from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from algan.errors import AlganConfigurationError
from algan.render_loop import RenderLoopMixin, _primitive_source_device
from algan.rendering.primitives.bezier_circuit_primitive import (
    BezierCircuitPrimitive,
)
from algan.rendering.raytracing.primitives import RayTracedTrianglePrimitive
from algan.settings import SETTINGS
from algan.settings._startup import render_device


def test_grouped_triangle_stays_on_its_source_device():
    source = RayTracedTrianglePrimitive(
        corners=torch.zeros((1, 3, 3)),
        colors=torch.ones((1, 3, 5)),
        normals=torch.zeros((1, 3, 3)),
    )

    grouped = RayTracedTrianglePrimitive(triangle_collection=[source])

    assert grouped.corners.device.type == "cpu"
    assert grouped.colors.device.type == "cpu"
    assert grouped.reflectivity.device.type == "cpu"


def test_grouped_bezier_stays_on_its_source_device():
    source = SimpleNamespace(
        corners=torch.zeros((1, 1, 4, 3)),
        num_segments_per_circuit=torch.ones((1,), dtype=torch.long),
        num_texture_points=0,
        filled=True,
        colors=torch.zeros((1, 1, 1, 5)),
        next_segment_inds=torch.zeros((1, 1, 1, 1), dtype=torch.long),
        normals=torch.zeros((1, 1, 3)),
        stroke_width=torch.zeros((1, 1, 1)),
        stroke_color=torch.zeros((1, 1, 5)),
        glow_radius=torch.zeros((1, 1, 1)),
        mob_center=torch.zeros((1, 1, 3)),
        grid_width=torch.ones((1, 1, 1)),
        grid_height=torch.ones((1, 1, 1)),
        basis1=torch.zeros((1, 1, 3)),
        basis2=torch.zeros((1, 1, 3)),
    )

    grouped = BezierCircuitPrimitive(triangle_collection=[source])

    assert grouped.corners.device.type == "cpu"
    assert grouped.colors.device.type == "cpu"
    assert grouped.next_segment_inds.device.type == "cpu"


def test_render_state_snapshot_uses_camera_source_device():
    camera = SimpleNamespace(
        location=torch.zeros((2, 1, 3)),
        screen=SimpleNamespace(location=torch.zeros((2, 1, 3))),
        _get_render_screen_basis=lambda: torch.eye(3).expand(2, -1, -1),
    )
    # frames_per_second converts the frame window to times for the light
    # lifespan filter (lights outside the window are left out of the snapshot).
    scene = SimpleNamespace(camera=camera, light_sources=[], frames_per_second=10)

    state = RenderLoopMixin._materialize_render_state(scene, 0, 2)

    assert state["ray_origin"].device.type == "cpu"
    assert state["screen_point"].device.type == "cpu"
    assert state["screen_basis"].device.type == "cpu"


def test_primitive_source_device_uses_primitive_tensor():
    primitive = SimpleNamespace(corners=torch.zeros((1, 3, 3)))

    assert _primitive_source_device(primitive).type == "cpu"


def test_render_device_setting_is_live_and_batch_prep_follows_it():
    """The device is a setting, and nothing here may hold a copy of it.

    Batch prep decides per tensor whether to move it to the render device, so a
    module that bound the device at import would keep preparing for the old one
    after a change and hand the tracer tensors on the wrong device.
    """
    original = SETTINGS.computing.render_device
    try:
        SETTINGS.computing.set(render_device=torch.device("meta"))
        assert render_device() == torch.device("meta")
    finally:
        SETTINGS.computing.set(render_device=original)
    assert render_device() == original


def test_animation_device_setting_is_initialization_only():
    with pytest.raises(AlganConfigurationError, match="ALGAN_ANIMATION_DEVICE"):
        SETTINGS.computing.set(animation_device=torch.device("meta"))


def test_get_frames_releases_arena_and_restores_background_on_error():
    allocated = SimpleNamespace(data=object())

    class FailingScene(RenderLoopMixin):
        def _get_frames_impl(self, *args, **kwargs):
            self.background_frame = "temporary"
            self.memory = allocated
            raise RuntimeError("boom")
            yield  # pragma: no cover - makes this a generator

    scene = FailingScene.__new__(FailingScene)
    scene.background_frame = "original"
    scene.memory = None

    with pytest.raises(RuntimeError, match="boom"):
        next(scene.get_frames(0, 1))

    assert scene.background_frame == "original"
    assert scene.memory is None
    assert allocated.data is None


@pytest.mark.fast
@pytest.mark.parametrize("ending", ["complete", "error", "close"])
def test_render_reclaims_arena_before_deferred_pressure_reset(monkeypatch, ending):
    """Render, GC and runtime teardown must agree on the arena's lifetime."""
    import algan.render_loop as render_loop
    from algan.rendering import taichi_runtime as runtime
    from algan.utils import memory_utils

    allocated = SimpleNamespace(data=object())
    original_memory = SimpleNamespace(data=object())
    frozen = False
    calls = []

    @contextmanager
    def freeze_scene():
        nonlocal frozen
        frozen = True
        try:
            yield
        finally:
            frozen = False

    class RenderScene(RenderLoopMixin):
        def _get_frames_impl(self, *_args, **_kwargs):
            self.memory = allocated
            self.background_frame = "temporary"
            runtime._PRESSURE_RESET_PENDING = True
            try:
                yield "frame"
                if ending == "error":
                    raise RuntimeError("boom")
            finally:
                calls.append("worker joined")

    def reclaim(force_gc):
        assert force_gc is False
        assert not runtime.render_is_active()
        assert not frozen
        assert allocated.data is None
        assert scene.memory is original_memory
        assert scene.background_frame == "original"
        calls.append("reclaim")

    monkeypatch.setattr(render_loop, "ensure_taichi_for_render", lambda: None)
    monkeypatch.setattr(render_loop, "scene_excluded_from_gc", freeze_scene)
    monkeypatch.setattr(memory_utils, "release_torch_memory", reclaim)
    monkeypatch.setattr(runtime, "_RENDER_JOBS_ACTIVE", 0)
    monkeypatch.setattr(runtime, "_PRESSURE_RESET_PENDING", False)
    scene = RenderScene.__new__(RenderScene)
    scene.memory = original_memory
    scene.background_frame = "original"
    frames = scene.get_frames(0, 2, post_processes=())
    assert next(frames) == "frame"
    assert runtime.render_is_active()
    assert allocated.data is not None
    assert calls == []
    if ending == "close":
        frames.close()
    elif ending == "error":
        with pytest.raises(RuntimeError, match="boom"):
            next(frames)
    else:
        assert list(frames) == []
    assert calls == ["worker joined", "reclaim"]
    assert not runtime._PRESSURE_RESET_PENDING


@pytest.mark.fast
def test_video_defers_pressure_cleanup_until_encoder_and_frame_locals_are_gone(
    monkeypatch,
):
    import weakref

    import algan.render_loop as render_loop
    from algan.rendering import taichi_runtime as runtime
    from algan.utils import memory_utils

    calls = []
    last_frame = None

    class Frame:
        pass

    class VideoScene(RenderLoopMixin):
        def _render_to_video_impl(self, *_args, **_kwargs):
            nonlocal last_frame
            frame = Frame()
            last_frame = weakref.ref(frame)
            with runtime.render_job_holding_the_arch():
                runtime._PRESSURE_RESET_PENDING = True
            assert calls == []
            calls.append("encoder closed")

    def reclaim(force_gc):
        assert force_gc is False
        assert last_frame() is None
        assert not runtime.render_is_active()
        calls.append("reclaim")

    monkeypatch.setattr(render_loop, "ensure_taichi_for_render", lambda: None)
    monkeypatch.setattr(memory_utils, "release_torch_memory", reclaim)
    monkeypatch.setattr(runtime, "_RENDER_JOBS_ACTIVE", 0)
    monkeypatch.setattr(runtime, "_PRESSURE_RESET_PENDING", False)
    VideoScene.__new__(VideoScene)._render_to_video(None, None, None)
    assert calls == ["encoder closed", "reclaim"]
