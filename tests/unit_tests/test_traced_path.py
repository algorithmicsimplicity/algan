"""History-based motion trails must not depend on the order frames are requested."""

from __future__ import annotations

import contextlib

import pytest
import torch

import algan
from algan import BLUE, ORIGIN, RIGHT, UP, Dot, Off, Scene, Seq, TracedPath, easings
from algan.errors import AlganConfigurationError


def _sample(scene, trail, times):
    timeline = scene.timeline_manager
    with timeline.preserving_authoring_state():
        try:
            timeline.set_state_to_times(torch.tensor(times, dtype=torch.float32))
            return trail.get_boundary_points().clone()
        finally:
            timeline.clear_buffers()


def _unique_vertices(points):
    return torch.unique_consecutive(points, dim=0)


@pytest.mark.fast
def test_trace_is_independent_of_frame_order_and_batching(fresh_scene):
    scene = Scene.current()
    dot = Dot().spawn(animate=False)
    trail = TracedPath(dot.get_center, sample_interval=0.25).spawn(animate=False)
    with Seq(runtime=2, easing=easings.linear):
        dot.move(2 * RIGHT)
    authored = dot.location.clone()
    order = [1.5, 0.5, 1.0, 1.5]
    batched = _sample(scene, trail, order)
    for index, time in enumerate(order):
        separate = _sample(scene, trail, [time])[0]
        torch.testing.assert_close(
            _unique_vertices(batched[index]), _unique_vertices(separate)
        )
    torch.testing.assert_close(batched[1, -1], RIGHT * 0.5, atol=2e-5, rtol=0)
    torch.testing.assert_close(dot.location, authored)
    assert trail._trace_points is None
    assert "TracedPath" in algan.__all__
    assert "_PointSource" not in algan.__all__


def test_finite_trail_has_exact_endpoints_and_disappears_when_still(fresh_scene):
    scene = Scene.current()
    dot = Dot().spawn(animate=False)
    trail = TracedPath(dot, dissipating_time=0.7, sample_interval=0.25).spawn(
        animate=False
    )
    with Seq(runtime=2, easing=easings.linear):
        dot.move(2 * RIGHT)
    Scene.wait(2)
    points = _sample(scene, trail, [1.23])[0]
    torch.testing.assert_close(points[0], RIGHT * 0.53, atol=2e-5, rtol=0)
    torch.testing.assert_close(points[-1], RIGHT * 1.23, atol=2e-5, rtol=0)
    timeline = scene.timeline_manager
    try:
        timeline.set_state_to_times(torch.tensor([3.0]))
        assert trail.get_render_primitives() is None
    finally:
        timeline.clear_buffers()


def test_complete_trace_persists_during_a_hold(fresh_scene):
    scene = Scene.current()
    dot = Dot().spawn(animate=False)
    trail = TracedPath(dot, sample_interval=0.25).spawn(animate=False)
    dot.move(RIGHT)
    Scene.wait(2)
    first = _unique_vertices(_sample(scene, trail, [1])[0])
    later = _unique_vertices(_sample(scene, trail, [3])[0])
    torch.testing.assert_close(first, later)


@pytest.mark.parametrize("source_kind", ["mob", "bound_method", "closure"])
def test_trace_follows_source_across_history_detachment(fresh_scene, source_kind):
    from algan import Square

    scene = Scene.current()
    dot = Dot().spawn(animate=False)
    source = {
        "mob": dot,
        "bound_method": dot.get_center,
        "closure": lambda: dot.get_center(),
    }[source_kind]
    trail = TracedPath(source, sample_interval=0.25).spawn(animate=False)
    # Put history boundaries just beyond float32 sample times as well as
    # exercising multiple morph/detach incarnations in one requested batch.
    with Seq(runtime=4.00000004, easing=easings.linear):
        dot.move(RIGHT)
        dot.become(Square(location=RIGHT * 3))
        dot.move(UP)
        dot.detach_history()
        dot.move(UP)
    authored = dot.get_center().clone()
    identity = dot.id
    batched = _sample(scene, trail, [3.5, 0.5, 1.5])
    torch.testing.assert_close(batched[1, :3, 0], torch.tensor([0, 0.25, 0.5]))
    torch.testing.assert_close(batched[2, -1], RIGHT * 2, atol=2e-5, rtol=0)
    torch.testing.assert_close(batched[0, -1], RIGHT * 3 + UP * 1.5, atol=2e-5, rtol=0)
    for index, time in enumerate([3.5, 0.5, 1.5]):
        torch.testing.assert_close(
            _unique_vertices(batched[index]),
            _unique_vertices(_sample(scene, trail, [time])[0]),
        )
    assert dot.id == identity
    torch.testing.assert_close(dot.get_center(), authored)


def test_trace_starts_at_its_rescaled_spawn_and_excludes_earlier_motion(fresh_scene):
    scene = Scene.current()
    dot = Dot().spawn(animate=False)
    with Seq(runtime=6, easing=easings.linear):
        dot.move(RIGHT)
        trail = TracedPath(dot, sample_interval=0.5).spawn(animate=False)
        dot.move(UP)
        Scene.wait(1)
    assert trail.lifespan.start() == 2
    points = _sample(scene, trail, [3])[0]
    torch.testing.assert_close(points[0], RIGHT, atol=2e-5, rtol=0)
    torch.testing.assert_close(points[-1], RIGHT + UP * 0.5, atol=2e-5, rtol=0)


def test_updater_motion_is_sampled_in_three_dimensions(fresh_scene):
    scene = Scene.current()
    dot = Dot().spawn(animate=False)

    def spiral(mob, elapsed):
        t = elapsed.reshape(-1, 1, 1)
        mob.location = torch.cat((torch.cos(t), torch.sin(t), t * 0.2), -1)

    dot.add_updater(spiral)
    trail = TracedPath(lambda: dot.location, sample_interval=0.25).spawn(animate=False)
    Scene.wait(3)
    times = torch.arange(9) * 0.25
    expected = torch.stack((times.cos(), times.sin(), times * 0.2), -1)
    torch.testing.assert_close(_sample(scene, trail, [2])[0], expected)
    timeline = scene.timeline_manager
    try:
        timeline.set_state_to_times(torch.tensor([0, 1, 2.0]))
        primitive = trail.get_render_primitives()
        assert primitive.corners.shape == (3, 8, 4, 3)
        assert float(primitive.corners[2, ..., 2].max()) == pytest.approx(0.4)
        assert bool(torch.isfinite(primitive.corners).all())
        assert bool(torch.isfinite(primitive.basis1).all())
        assert bool(torch.isfinite(primitive.basis2).all())
    finally:
        timeline.clear_buffers()


def test_removing_source_updater_captures_only_the_removal_boundary(fresh_scene):
    scene = Scene.current()
    dot = Dot().spawn(animate=False)

    def move(mob, elapsed):
        mob.location = elapsed.reshape(-1, 1, 1) * RIGHT

    updater = dot.add_updater(move)
    trail = TracedPath(dot, sample_interval=0.25).spawn(animate=False)
    Scene.wait(1)
    dot.remove_updater(updater)
    Scene.wait(1)
    assert dot.location.shape == (1, 1, 3)
    torch.testing.assert_close(dot.location.reshape(3), RIGHT)
    points = _sample(scene, trail, [1.5])[0]
    torch.testing.assert_close(points[-1], RIGHT)
    torch.testing.assert_close(points[2], RIGHT * 0.5)


def test_a_preview_does_not_freeze_later_context_rescaling(fresh_scene):
    scene = Scene.current()
    dot = Dot().spawn(animate=False)
    trail = TracedPath(dot, sample_interval=0.25).spawn(animate=False)
    with Seq(runtime=4, easing=easings.linear):
        dot.move(RIGHT)
        _sample(scene, trail, [0.5])
        dot.move(UP)
    points = _sample(scene, trail, [1])[0]
    torch.testing.assert_close(points[-1], RIGHT * 0.5, atol=2e-5, rtol=0)
    assert scene._recorded_end_time_for_render() == 4


def test_clone_follows_same_source_and_trail_transforms_leave_source_alone(fresh_scene):
    scene = Scene.current()
    dot = Dot().spawn(animate=False)
    trail = TracedPath(dot, sample_interval=0.25).spawn(animate=False)
    clone = trail.clone(spawn=False).spawn(animate=False)
    assert clone._point_source is trail._point_source
    assert len(scene.timeline_manager._traced_paths) == 2
    with Off():
        clone.move(UP)
        clone.color = BLUE
    dot.move(RIGHT)
    original = _sample(scene, trail, [0.5])
    shifted = _sample(scene, clone, [0.5])
    torch.testing.assert_close(shifted, original + UP)
    torch.testing.assert_close(dot.location.reshape(3), RIGHT)


def test_scene_inference_and_active_actor_filtering(fresh_scene):
    scene = Scene.current()
    dot = Dot().spawn(animate=False)
    trail = TracedPath(dot, sample_interval=0.5).spawn(animate=False)
    dot.move(RIGHT)
    with Scene() as other:
        inferred = TracedPath(dot.get_center)
        assert inferred.scene is scene
        assert not other.timeline_manager._traced_paths
        with pytest.raises(AlganConfigurationError, match="same|its Scene"):
            TracedPath(dot, scene=other)
        sample = _sample(scene, trail, [0.5])
        assert sample.shape == (1, 2, 3)
        assert scene._recorded_end_time_for_render() == 1
    try:
        scene.timeline_manager.set_state_to_times(
            torch.tensor([0.5]), active_mobs=[dot]
        )
        assert trail._trace_points is None
        # The source is needed for history even when it is not a rendered actor.
        scene.timeline_manager.set_state_to_times(
            torch.tensor([0.5]), active_mobs=[trail]
        )
        assert trail._trace_points is not None
    finally:
        scene.timeline_manager.clear_buffers()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_interval": 0},
        {"sample_interval": -1},
        {"sample_interval": float("nan")},
        {"sample_interval": True},
        {"dissipating_time": 0},
        {"dissipating_time": float("inf")},
    ],
)
def test_invalid_sample_settings_are_rejected(fresh_scene, kwargs):
    with pytest.raises(AlganConfigurationError):
        TracedPath(lambda: ORIGIN, **kwargs)
    assert not Scene.current().timeline_manager._traced_paths


@pytest.mark.parametrize(
    "source",
    [
        None,
        1,
        lambda: [1, 2],
        lambda: [1, 2, float("nan")],
        lambda: [[1, 2, 3], [4, 5, 6]],
    ],
)
def test_invalid_point_sources_are_rejected(fresh_scene, source):
    with pytest.raises(AlganConfigurationError):
        TracedPath(source)


def test_failed_sampling_cleans_buffers_and_allows_retry(fresh_scene):
    scene = Scene.current()
    dot = Dot().spawn(animate=False)
    fail = [False]

    def point():
        if fail[0]:
            raise ValueError("unavailable")
        return dot.location

    trail = TracedPath(point, sample_interval=0.25).spawn(animate=False)
    dot.move(RIGHT)
    dot.detach_history()
    dot.move(UP)
    authored_dict = dot.__dict__
    authored = dot.location.clone()
    fail[0] = True
    with pytest.raises(AlganConfigurationError, match="3-D point"):
        _sample(scene, trail, [0.5])
    assert not scene.timeline_manager._sampling_traced_paths
    assert dot.__dict__ is authored_dict
    torch.testing.assert_close(dot.location, authored)
    fail[0] = False
    points = _sample(scene, trail, [0.5])
    assert points.shape == (1, 3, 3)
    torch.testing.assert_close(points[0, -1], RIGHT * 0.5)


def test_constant_and_unspawned_traces_have_no_visible_geometry(fresh_scene):
    scene = Scene.current()
    trail = TracedPath(lambda: [1, 2, 3])
    Scene.wait(1)
    try:
        scene.timeline_manager.set_state_to_times(torch.tensor([0.5]))
        assert trail.get_render_primitives() is None
        scene.timeline_manager.clear_buffers()
        trail.spawn(animate=False)
        Scene.wait(1)
        scene.timeline_manager.set_state_to_times(torch.tensor([1.5]))
        assert trail.get_render_primitives() is None
    finally:
        scene.timeline_manager.clear_buffers()


def test_sampling_storage_is_bounded_by_dissipation_and_clear_buffers(fresh_scene):
    scene = Scene.current()
    dot = Dot().spawn(animate=False)
    trail = TracedPath(dot, dissipating_time=0.7, sample_interval=0.25).spawn(
        animate=False
    )
    Scene.wait(1000)
    times = trail._sample_times(torch.tensor([2.0, 10.0, 999.0]))
    assert times.shape == (3, 5)
    assert trail._get_memory_used_per_timestep() < 10000
    timeline = scene.timeline_manager
    before = {
        name: data.current_state.shape
        for name, data in timeline.attr_to_timeline.items()
    }
    _sample(scene, trail, [998.0])
    assert before == {
        name: data.current_state.shape
        for name, data in timeline.attr_to_timeline.items()
    }
    assert trail._trace_points is None


def test_rendered_trace_survives_backward_scrubbing_and_batch_changes(
    tmp_path, fresh_scene
):
    from algan import BLACK, LEFT, OUT, PREVIEW

    with Scene(
        PREVIEW.set(resolution=(160, 120), frames_per_second=4), background=BLACK
    ) as scene:
        dot = Dot(LEFT * 1.5, opacity=0).spawn(animate=False)
        trail = TracedPath(
            dot, stroke_color=BLUE, stroke_width=10, sample_interval=1 / 16
        ).spawn(animate=False)
        with Seq(runtime=2, easing=easings.linear):
            dot.orbit(240, OUT, about=ORIGIN)
        Scene.wait(1)

        def frames(indices):
            with contextlib.closing(
                scene.get_frames(
                    0, len(indices), frame_indices=indices, post_processes=()
                )
            ) as output:
                return torch.cat([frame.cpu().clone() for frame in output])

        batch = frames((0, 2, 6))
        assert not bool(batch[0].any())
        assert int(batch[2].max()) > 50
        assert not torch.equal(batch[1], batch[2])
        for index, frame in ((6, batch[2]), (2, batch[1])):
            assert int((frames((index,))[0].int() - frame.int()).abs().max()) <= 2
        assert trail._trace_points is None
        # Useful when investigating a pixel failure, without maintaining a
        # platform-specific render baseline for this focused feature check.
        from PIL import Image

        Image.fromarray(batch[2].numpy()).save(tmp_path / "traced_path.png")
