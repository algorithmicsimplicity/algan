"""Circuit reuse must preserve contours, ordering and live rendering metadata."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from algan import PREVIEW, RIGHT, Annulus, Off, Scene, Square, Sync, Text
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.bezier_geometry_cache import _BezierGeometryCache
from algan.rendering.raytracing.primitives import RayTracedBezierCircuitPrimitive


def _primitive(scene, frames=3):
    with Off():
        mobs = [
            Square(add_to_scene=False),
            Annulus(add_to_scene=False),
            Square(add_to_scene=False),
        ]
    primitive = RayTracedBezierCircuitPrimitive(
        triangle_collection=[mob.get_render_primitives() for mob in mobs]
    )
    primitive.scene = scene
    for attr in primitive.frame_dependent_source_attrs:
        value = getattr(primitive, attr)
        setattr(primitive, attr, value.expand(frames, *value.shape[1:]).clone())
    return primitive


def _build(primitive, chords=None):
    corners = primitive.corners.float().contiguous()
    if chords is None:
        chords = torch.full((corners.shape[1],), 4, dtype=torch.long)
    primitive._build_circuit_geometry(corners, chords)
    return {
        name: getattr(primitive, name).clone()
        for name in (
            "_rt_edges",
            "_rt_edge_offsets",
            "_rt_circuit_of_segment",
            "_rt_circuit_meta",
            "_rt_circuit_colors",
            "_rt_circuit_border_colors",
        )
    }


def _same(actual, expected):
    for name in expected:
        a, b = torch.broadcast_tensors(actual[name], expected[name])
        assert torch.equal(a, b), name


def test_static_edges_build_once_across_frames_and_new_primitives(monkeypatch):
    with Scene() as scene:
        first = _primitive(scene)
        monkeypatch.setattr(rt_settings, "bezier_geometry_cache", False)
        expected = _build(first)
        monkeypatch.setattr(rt_settings, "bezier_geometry_cache", True)
        actual = _build(first)
        _same(actual, expected)
        assert actual["_rt_edges"].shape[0] == 1
        second = _primitive(scene, frames=5)
        _same(
            _build(second),
            {
                name: value[:1] if value.ndim >= 3 else value
                for name, value in expected.items()
            },
        )
        assert scene._bezier_geometry_cache.hits == 1
        assert scene._bezier_geometry_cache.misses == 1


@pytest.mark.fast
def test_text_reuses_geometry_across_materialized_fade_windows(monkeypatch):
    # The performance contract spans Text packing, timeline replay, collection
    # merging and renderer ownership. A change to any of those can silently
    # turn a stationary text fade back into per-frame contour construction.
    with Scene(video_settings=PREVIEW.set(frames_per_second=4)) as scene:
        with Off():
            text = Text("B8Oi", font="Algan Test Sans").spawn()
            moving = Square(size=0.2).spawn()
        with Sync(runtime=1):
            text.opacity = 0.3
            moving.move(RIGHT)
        scene.scene_times.append([0, 4])
        scene._initialize_frames()
        monkeypatch.setattr(scene, "_render_device_prep_budget", lambda: 100_000_000)
        actors = [
            scene.camera,
            scene.camera.screen,
            *scene.light_sources,
            *scene.actors,
        ]
        outputs = []
        try:
            for start in (0, 2):
                with scene._batch_prep_context():
                    primitives, end, _ = scene._get_batch_of_primitives(
                        start, start + 2, actors, 100_000_000
                    )
                assert end == start + 2
                circuits = [
                    p
                    for p in primitives
                    if isinstance(p, RayTracedBezierCircuitPrimitive)
                ]
                assert circuits
                colors = []
                for primitive in circuits:
                    actual = _build(primitive)
                    colors.append(actual["_rt_circuit_colors"].reshape(2, -1, 5))
                    monkeypatch.setattr(rt_settings, "bezier_geometry_cache", False)
                    _same(actual, _build(primitive))
                    monkeypatch.setattr(rt_settings, "bezier_geometry_cache", True)
                outputs.append(torch.cat(colors, 1))
            assert scene._bezier_geometry_cache.hits >= 1
            assert not torch.equal(outputs[0], outputs[1])
        finally:
            scene.timeline_manager.clear_buffers()
            if hasattr(scene, "_bezier_geometry_cache"):
                scene._bezier_geometry_cache.clear()


def test_moving_circuit_does_not_disable_reuse_of_static_neighbours(monkeypatch):
    with Scene() as scene:
        primitive = _primitive(scene)
        segments = primitive.num_segments_per_object.flatten().long()
        start, end = int(segments[0]), int(segments[:2].sum())
        # The moving circuit is between static circuits, so partitioning must
        # restore order rather than simply concatenate the two groups.
        primitive.corners[1:, start:end, :, 0] += 0.25
        primitive.mob_center[1:, 1, 0] += 0.25
        monkeypatch.setattr(rt_settings, "bezier_geometry_cache", False)
        expected = _build(primitive)
        monkeypatch.setattr(rt_settings, "bezier_geometry_cache", True)
        _same(_build(primitive), expected)
        primitive.corners[2, start:end, :, 1] += 0.5
        primitive.mob_center[2, 1, 1] += 0.5
        # A genuine contour change as well as a rigid translation: cached
        # local edges must not accidentally conceal an animated control point.
        primitive.corners[2, start, 1, 0] += 0.1875
        actual = _build(primitive)
        assert not torch.equal(actual["_rt_edges"], expected["_rt_edges"])
        assert scene._bezier_geometry_cache.hits == 1
        monkeypatch.setattr(rt_settings, "bezier_geometry_cache", False)
        _same(actual, _build(primitive))


@pytest.mark.parametrize("change", ["corners", "plane", "topology", "samples", "wedge"])
def test_geometry_changes_cannot_reuse_stale_edges(monkeypatch, change):
    with Scene() as scene:
        primitive = _primitive(scene)
        _build(primitive)
        chords = None
        if change == "corners":
            primitive.corners[:, 0, 1, 0] += 0.125
        elif change == "plane":
            primitive.normals[..., 0] += 0.25
        elif change == "topology":
            primitive.next_segment_inds[:, 0] = 2
        elif change == "samples":
            chords = torch.full((primitive.corners.shape[1],), 8, dtype=torch.long)
        else:
            monkeypatch.setattr(rt_settings, "analytic_aa_bez_wedge", False)
        actual = _build(primitive, chords)
        assert scene._bezier_geometry_cache.hits == 0
        assert scene._bezier_geometry_cache.misses == 2
        monkeypatch.setattr(rt_settings, "bezier_geometry_cache", False)
        _same(actual, _build(primitive, chords))


def test_cached_geometry_keeps_colors_opacity_and_materials_live(monkeypatch):
    with Scene() as scene:
        primitive = _primitive(scene)
        before = _build(primitive)
        primitive.colors[1:, ..., :3] *= 0.25
        primitive.colors[2, ..., 4] = 0
        primitive.stroke_color[..., :3] = 0.1
        primitive.stroke_width[:] = 5
        primitive.basis1 *= 2
        primitive.transmission[:] = 0.4
        primitive._rt_projection_aa = 2
        actual = _build(primitive)
        assert scene._bezier_geometry_cache.hits == 1
        assert not torch.equal(actual["_rt_circuit_meta"], before["_rt_circuit_meta"])
        assert not torch.equal(
            actual["_rt_circuit_colors"], before["_rt_circuit_colors"]
        )
        monkeypatch.setattr(rt_settings, "bezier_geometry_cache", False)
        _same(actual, _build(primitive))


def test_distinct_scenes_do_not_share_geometry():
    with Scene() as first:
        _build(_primitive(first))
    with Scene() as second:
        _build(_primitive(second))
    assert first._bezier_geometry_cache is not second._bezier_geometry_cache
    assert second._bezier_geometry_cache.hits == 0


def test_camera_capture_does_not_inherit_main_pass_cache():
    from algan.rendering.camera_views import _render_pass

    with Scene() as scene:
        _build(_primitive(scene))
        capture = _render_pass(scene, scene.camera, scene.actors, (120, 80))
        assert not hasattr(capture, "_bezier_geometry_cache")
        _build(_primitive(capture))
        assert capture._bezier_geometry_cache is not scene._bezier_geometry_cache


def test_cache_is_bounded_and_evicts_least_recently_used():
    cache = _BezierGeometryCache(max_bytes=100)
    tensors = (torch.zeros(1, 1, 6), torch.zeros(2, dtype=torch.int32))
    keys = [(False, (), bytes([i])) for i in range(4)]
    for key in keys[:3]:
        cache.put(key, tensors)
    assert cache.bytes == 99
    assert cache.get(keys[0]) is tensors
    cache.put(keys[3], tensors)
    assert cache.get(keys[1]) is None
    assert cache.get(keys[0]) is tensors
    cache.put((False, (), b"large"), (torch.zeros(100),))
    assert cache.bytes <= 100
    cache.clear()
    assert cache.bytes == 0
    assert not cache._entries


def test_large_geometry_still_collapses_frames_without_retaining_it():
    with Scene() as scene:
        scene._bezier_geometry_cache = _BezierGeometryCache(max_bytes=1)
        actual = _build(_primitive(scene))
    assert actual["_rt_edges"].shape[0] == 1
    assert scene._bezier_geometry_cache.bytes == 0


def test_autograd_geometry_is_not_cached():
    with Scene() as scene:
        primitive = _primitive(scene)
        primitive.corners.requires_grad_(True)
        _build(primitive)
    assert not hasattr(scene, "_bezier_geometry_cache")


@pytest.mark.parametrize("screen_height", [200, 960])
@pytest.mark.parametrize("z_index", [0, 1000])
def test_projection_rechecks_camera_sampling_and_bounds(
    monkeypatch, screen_height, z_index
):
    monkeypatch.setattr(rt_settings, "pn_criterion_kernel", False)

    def camera(distance):
        return SimpleNamespace(
            ray_origin=torch.tensor([[0.0, 0.0, distance]]),
            screen_point=torch.tensor([[0.0, 0.0, distance - 2]]),
            screen_basis=torch.eye(3).unsqueeze(0),
            screen_height=screen_height,
            output_screen_height=screen_height,
            analytic_raster=True,
        )

    with Scene() as scene:
        for distance in (20.0, 5.0, 20.0):
            primitive = _primitive(scene, frames=1)
            primitive.z_index.fill_(z_index)
            primitive._has_z_index = bool(z_index)
            primitive.project_to_screen(camera(distance), [])
            actual = {
                name: value.clone()
                for name, value in vars(primitive).items()
                if name.startswith("_rt_") and isinstance(value, torch.Tensor)
            }
            monkeypatch.setattr(rt_settings, "bezier_geometry_cache", False)
            reference = _primitive(scene, frames=1)
            reference.z_index.fill_(z_index)
            reference._has_z_index = bool(z_index)
            reference.project_to_screen(camera(distance), [])
            _same(actual, {name: getattr(reference, name) for name in actual})
            monkeypatch.setattr(rt_settings, "bezier_geometry_cache", True)
        assert scene._bezier_geometry_cache.hits >= 1
