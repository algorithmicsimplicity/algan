"""Geometry and compatibility regressions from the October bug audit."""

import numpy as np
import pytest
import torch

from algan import (
    DOWN,
    LEFT,
    RED,
    RIGHT,
    UP,
    Axes,
    Graph,
    Lag,
    Line,
    Off,
    Scene,
    Seq,
    Square,
    Sync,
    easings,
)


@pytest.mark.parametrize(
    ("context", "duration", "starts"),
    [
        (lambda: Seq(equalize_runtimes=True), 4, [0, 2]),
        (lambda: Seq(equalize_runtimes=True, runtime=6), 6, [0, 3]),
        (lambda: Lag(0.5, equalize_runtimes=True), 3, [0, 1]),
        (lambda: Sync(equalize_runtimes=True), 2, [0, 0]),
    ],
)
@pytest.mark.parametrize("child_durations", [(2, 1), (1, 2)])
def test_equalized_contexts_schedule_every_child_inside_parent(
    context, duration, starts, child_durations
):
    with Scene() as scene:
        a, b = Square().spawn(False), Square().spawn(False)
        with context() as parent:
            with Sync(runtime=child_durations[0], easing=easings.identity) as first:
                a.move(RIGHT)
            with Sync(runtime=child_durations[1], easing=easings.identity) as second:
                b.move(UP)
        assert parent.timespan.end == pytest.approx(duration)
        assert [first.timespan.start, second.timespan.start] == pytest.approx(starts)
        assert second.timespan.end == pytest.approx(duration)
        scene.timeline_manager.set_state_to_times(
            torch.tensor([duration], dtype=torch.float32)
        )
        torch.testing.assert_close(a.location.reshape(-1), RIGHT)
        torch.testing.assert_close(b.location.reshape(-1), UP)


@pytest.mark.parametrize("sheared", [False, True])
def test_basis_round_trip_preserves_control_points_and_descendants(sheared):
    with Scene(), Off():
        shape = Square()
        child = Square().move(RIGHT * 3)
        shape.add_children(child)
        basis = (
            torch.tensor([[1.0, 0.5, 0], [0, 1, 0.25], [0, 0, 1]])
            if sheared
            else torch.eye(3)
        )
        shape.basis = basis.reshape(9)
        original = [
            mob.location.clone() for mob in shape.get_descendants(include_self=True)
        ]
        shape.basis = shape.basis
        shape.basis = torch.eye(3).reshape(9)
        shape.basis = basis.reshape(9)
        for mob, expected in zip(shape.get_descendants(include_self=True), original):
            torch.testing.assert_close(mob.location, expected, atol=5e-6, rtol=1e-5)


def test_line_endpoint_change_preserves_appearance():
    with Scene(), Off():
        line = Line(
            LEFT,
            RIGHT,
            stroke_color=RED,
            stroke_width=20,
            opacity=0.4,
            cap_style="square",
            joint_type="bevel",
            miter_limit=3,
            z_index=2,
        )
        color, opacity = line.stroke_color.clone(), line.opacity.clone()
        line.z_index = 7
        before_order = line.z_index
        line.put_start_and_end_on(DOWN, UP)
        torch.testing.assert_close(line.get_start().reshape(-1), DOWN)
        torch.testing.assert_close(line.get_end().reshape(-1), UP)
        torch.testing.assert_close(line.stroke_color, color)
        torch.testing.assert_close(line.opacity, opacity)
        assert line.stroke_width.reshape(-1)[0] == 20
        assert (line.cap_style, line.joint_type, line.miter_limit, line.z_index) == (
            "square",
            "bevel",
            3,
            before_order,
        )


@pytest.mark.parametrize(
    ("method", "args", "count"),
    [
        ("add_vertices", (3,), 4),
        ("add_edges", ((2, 3),), 5),
        ("remove_vertices", (1,), 1),
        ("remove_edges", ((1, 2),), 2),
    ],
)
def test_graph_topology_changes_sync_native_geometry(method, args, count):
    with Scene(), Off():
        graph = Graph([1, 2], [(1, 2)]).spawn(False)
        result = getattr(graph, method)(*args)
        assert result is not graph
        native = [
            mob
            for mob in graph.get_descendants()
            if hasattr(mob, "control_points") and not getattr(mob, "empty", False)
        ]
        assert len(native) == count
        assert all(mob.is_spawned() for mob in native)


def test_axes_queries_follow_movement_after_structural_synchronization():
    with Scene(), Off():
        axes = Axes()
        axes.add_coordinates([], [])
        before = np.asarray(axes.c2p(0, 0))
        axes.move(RIGHT)
        np.testing.assert_allclose(
            np.asarray(axes.c2p(0, 0)) - before, [1, 0, 0], atol=1e-6
        )
        axes.add_coordinates([], [])
        axes.move(UP)
        np.testing.assert_allclose(
            np.asarray(axes.c2p(0, 0)) - before, [1, 1, 0], atol=1e-6
        )
