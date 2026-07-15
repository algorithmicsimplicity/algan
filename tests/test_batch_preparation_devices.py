from types import SimpleNamespace

import pytest
import torch

from algan.render_loop import RenderLoopMixin, _primitive_source_device
from algan.rendering.primitives.bezier_circuit_primitive import (
    BezierCircuitPrimitive,
)
from algan.rendering.raytracing.primitives import RayTracedTrianglePrimitive
from algan.settings.defaults import COMPUTING_DEFAULTS


def test_grouped_triangle_stays_on_its_source_device(monkeypatch):
    monkeypatch.setattr(COMPUTING_DEFAULTS, "render_device", torch.device("meta"))
    source = RayTracedTrianglePrimitive(
        corners=torch.zeros((1, 3, 3)),
        colors=torch.ones((1, 3, 5)),
        normals=torch.zeros((1, 3, 3)),
    )

    grouped = RayTracedTrianglePrimitive(triangle_collection=[source])

    assert grouped.corners.device.type == "cpu"
    assert grouped.colors.device.type == "cpu"
    assert grouped.reflectivity.device.type == "cpu"


def test_grouped_bezier_stays_on_its_source_device(monkeypatch):
    monkeypatch.setattr(COMPUTING_DEFAULTS, "render_device", torch.device("meta"))
    source = SimpleNamespace(
        corners=torch.zeros((1, 1, 4, 3)),
        num_segments_per_circuit=torch.ones((1,), dtype=torch.long),
        num_texture_points=0,
        filled=True,
        colors=torch.zeros((1, 1, 1, 5)),
        next_segment_inds=torch.zeros((1, 1, 1, 1), dtype=torch.long),
        normals=torch.zeros((1, 1, 3)),
        border_width=torch.zeros((1, 1, 1)),
        border_color=torch.zeros((1, 1, 5)),
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


def test_render_state_snapshot_uses_camera_source_device(monkeypatch):
    monkeypatch.setattr(COMPUTING_DEFAULTS, "render_device", torch.device("meta"))
    camera = SimpleNamespace(
        location=torch.zeros((2, 1, 3)),
        screen=SimpleNamespace(location=torch.zeros((2, 1, 3))),
        get_render_screen_basis=lambda: torch.eye(3).expand(2, -1, -1),
    )
    scene = SimpleNamespace(camera=camera, light_sources=[])

    state = RenderLoopMixin._materialize_render_state(scene, 0, 2)

    assert state["ray_origin"].device.type == "cpu"
    assert state["screen_point"].device.type == "cpu"
    assert state["screen_basis"].device.type == "cpu"


def test_primitive_source_device_ignores_render_default(monkeypatch):
    monkeypatch.setattr(COMPUTING_DEFAULTS, "render_device", torch.device("meta"))
    primitive = SimpleNamespace(corners=torch.zeros((1, 3, 3)))

    assert _primitive_source_device(primitive).type == "cpu"


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
