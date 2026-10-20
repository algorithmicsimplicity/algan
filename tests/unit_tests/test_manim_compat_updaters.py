"""Manim-backed Mobs moved and measured inside an updater.

A render replays an updater over a batch of frames at once, so the numbers it
works with hold one value per frame, while a compatibility Mob's backing Manim
object holds one pose. Moving such a Mob there -- ``move_to``, Manim's
``next_to``, ``shift`` and ``align_to``, or ``move`` on a composite such as an
Arrow -- failed the render, and Manim's queries (``get_top``, ``get_end``)
answered for the authored pose on every frame. ``c2p``, ``n2p`` and friends,
handed an updater's ``t``, came back in Manim's own array shapes -- ``c2p`` as
a ``(3, frames)`` column that ``move_to`` cannot take even when the updater is
first applied -- and ``point_from_proportion`` raised.

A ``ValueTracker`` read there held the batch's first frame on every frame, so a
dot steered by one stood still for a whole batch and then jumped.
"""

import math

import numpy as np
import pytest
import torch

import algan.manim as mn
from algan import (
    DOWN,
    LEFT,
    RIGHT,
    SMOKE_TEST,
    UP,
    Arrow,
    Axes,
    Brace,
    ComplexPlane,
    DecimalNumber,
    Dot,
    Group,
    MathTex,
    NumberLine,
    Scene,
    Seq,
    Square,
    Sync,
    Vector,
    easings,
)

FRAMES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
MANIM_BUFF = 0.25

BUILDERS = [
    pytest.param(lambda: Arrow(LEFT, RIGHT), id="Arrow"),
    pytest.param(lambda: Vector(DOWN), id="Vector"),
    pytest.param(lambda: MathTex("x"), id="MathTex"),
    pytest.param(lambda: Brace(Square(add_to_scene=False)), id="Brace"),
]


def _close(actual, expected):
    torch.testing.assert_close(
        actual.reshape(expected.shape).float(), expected.float(), atol=1e-4, rtol=0
    )


def _box(mob):
    corners = mob.get_bounding_box()
    return corners.amin(-2).reshape(-1, 3), corners.amax(-2).reshape(-1, 3)


@pytest.mark.parametrize("build", BUILDERS)
def test_move_to_and_move_place_the_mob_on_every_frame(build):
    with Scene() as scene:
        mob = build().spawn(animate=False)
        other = build().spawn(animate=False)
        start = other.get_center().reshape(3)
        mob.add_updater(lambda m, t: m.move_to(t * RIGHT + UP))
        other.add_updater(lambda m, t: m.move(t * UP))
        Scene.wait(2)

        scene.timeline_manager.set_state_to_times(FRAMES)
        _close(mob.get_center(), FRAMES[:, None] * RIGHT + UP)
        _close(other.get_center(), start + FRAMES[:, None] * UP)


@pytest.mark.parametrize("build", BUILDERS)
def test_manim_placement_follows_a_moving_mob(build):
    with Scene() as scene:
        dot = Dot(LEFT * 2)
        above, beside, aligned = build(), build(), build()
        Group(dot, above, beside, aligned).spawn(animate=False)
        above.add_updater(lambda m, t: m.next_to(dot, UP))
        beside.add_updater(lambda m, t: m.next_to(dot, RIGHT).shift(t * UP))
        aligned.add_updater(lambda m, t: m.align_to(dot, LEFT))
        with Seq(runtime=2):
            dot.move(RIGHT * 4)

        scene.timeline_manager.set_state_to_times(FRAMES)
        dot_low, dot_high = _box(dot)
        dot_center = (dot_low + dot_high) / 2
        low, high = _box(above)
        _close(low[:, 1], dot_high[:, 1] + MANIM_BUFF)
        _close((low[:, 0] + high[:, 0]) / 2, dot_center[:, 0])
        low, high = _box(beside)
        _close(low[:, 0], dot_high[:, 0] + MANIM_BUFF)
        _close((low[:, 1] + high[:, 1]) / 2, dot_center[:, 1] + FRAMES)
        low, _ = _box(aligned)
        _close(low[:, 0], dot_low[:, 0])


def test_manim_queries_answer_for_each_frame():
    with Scene() as scene:
        arrow = Arrow(LEFT, RIGHT)
        on_top, on_end = Dot(), Dot()
        Group(arrow, on_top, on_end).spawn(animate=False)
        authored_center = arrow.get_center().reshape(3)
        authored_end = arrow.get_end().reshape(3)
        on_top.add_updater(lambda m, t: m.move_to(arrow.get_top()))
        on_end.add_updater(lambda m, t: m.move_to(arrow.get_end()))
        with Seq(runtime=2):
            arrow.move(RIGHT * 4 + UP)

        scene.timeline_manager.set_state_to_times(FRAMES)
        shift = arrow.get_center().reshape(-1, 3) - authored_center
        low, high = _box(arrow)
        top = torch.stack([(low[:, 0] + high[:, 0]) / 2, high[:, 1], low[:, 2]], -1)
        _close(on_top.get_center()[..., :2], top[:, None, :2])
        _close(on_end.get_center(), authored_end + shift)


def test_axes_coordinates_follow_an_updaters_time():
    with Scene() as scene:
        axes = Axes(x_range=[0, 4], y_range=[0, 4])
        graph = axes.plot(lambda x: x**2 / 4, x_range=[0, 4])
        line = NumberLine(x_range=[-4, 4], length=8)
        dots = [Dot() for _ in range(4)]
        Group(axes, graph, line, *dots).spawn(animate=False)
        coordinates = {}

        def read_back(mob, t):
            coordinates["x"], coordinates["y"] = axes.p2c(mob.get_center())

        dots[0].add_updater(lambda m, t: m.move_to(axes.c2p(t, t * 2)))
        dots[1].add_updater(lambda m, t: m.move_to(axes.i2gp(t, graph)))
        dots[2].add_updater(lambda m, t: m.move_to(line.n2p(t * 2 - 3)))
        dots[3].add_updater(lambda m, t: m.move_to(graph.point_from_proportion(t / 2)))
        dots[0].add_updater(read_back)
        Scene.wait(2)

        scene.timeline_manager.set_state_to_times(FRAMES)
        for dot, expected in zip(
            dots,
            (
                [axes.c2p(float(t), float(t) * 2) for t in FRAMES],
                [axes.i2gp(float(t), graph) for t in FRAMES],
                [line.n2p(float(t) * 2 - 3) for t in FRAMES],
                [graph.point_from_proportion(float(t) / 2) for t in FRAMES],
            ),
        ):
            _close(
                dot.get_center(), torch.stack([torch.as_tensor(p) for p in expected])
            )
        _close(coordinates["x"], FRAMES)
        _close(coordinates["y"], FRAMES * 2)


def test_c2p_handed_an_updaters_t_answers_with_a_point():
    with Scene():
        axes = Axes(x_range=[0, 4], y_range=[0, 4])
        t = torch.ones(1, 1, 1)

        point = axes.c2p(t, t * 2)
        assert point.shape == (3,)
        np.testing.assert_allclose(point.numpy(), axes.c2p(1, 2))
        x, y = axes.p2c(point)
        assert (float(x), float(y)) == pytest.approx((1, 2))


def test_a_dot_and_its_labels_follow_a_graph_through_a_render(tmp_path):
    with Scene() as scene:
        axes = Axes(x_range=[0, 6], y_range=[-1.5, 1.5])
        line = NumberLine(x_range=[-5, 5], length=8).move(DOWN * 3)
        dot = Dot()
        label = MathTex("x")
        pointer = Vector(DOWN)
        Group(axes, line, dot, label, pointer).spawn(animate=False)
        dot.add_updater(lambda m, t: m.move_to(axes.c2p(t * 2, torch.sin(t * 2))))
        label.add_updater(lambda m, t: m.next_to(dot, UP))
        pointer.add_updater(lambda m, t: m.next_to(line.n2p(t * 2 - 3), UP))
        Scene.wait(2)

        result = scene.save_video(tmp_path / "follow.mp4", SMOKE_TEST)

    assert result.status == "rendered"


def test_a_tracker_read_as_a_number_holds_each_frames_value():
    """Manim code uses ``get_value()`` as a Python number, frame by frame."""
    with Scene() as scene:
        tracker = mn.ValueTracker(0).spawn(animate=False)
        display = DecimalNumber(0.0, decimal_places=1)
        sine, lifted = Dot(), Dot()
        Group(display, sine, lifted).spawn(animate=False)
        labels = []

        def show(mob, t):
            value = tracker.get_value()
            labels.append(f"{value:.1f}")
            mob.set_value(value)

        display.add_updater(show)
        sine.add_updater(lambda m, t: m.move_to(RIGHT * math.sin(tracker.get_value())))
        lifted.add_updater(lambda m, t: m.move_to(UP * float(tracker)))
        with Seq(runtime=2, easing=easings.linear):
            tracker.set_value(4)
        labels.clear()

        scene.timeline_manager.set_state_to_times(FRAMES)
        values = FRAMES * 2
        assert labels == [f"{value:.1f}" for value in values.tolist()]
        _close(display.value, values)
        _close(sine.get_center()[..., 0], torch.sin(values))
        _close(lifted.get_center()[..., 1], values)


def test_a_complex_tracker_holds_each_frames_value():
    with Scene() as scene:
        z = mn.ComplexValueTracker(0).spawn(animate=False)
        plane = ComplexPlane()
        on_plane, by_parts = Dot(), Dot()
        Group(plane, on_plane, by_parts).spawn(animate=False)
        on_plane.add_updater(lambda m, t: m.move_to(plane.n2p(z.get_value())))
        by_parts.add_updater(
            lambda m, t: m.move_to(RIGHT * z.get_value().real + UP * z.get_value().imag)
        )
        with Seq(runtime=2, easing=easings.linear):
            z.set_value(2 + 4j)

        scene.timeline_manager.set_state_to_times(FRAMES)
        expected = torch.stack([FRAMES, FRAMES * 2, torch.zeros(5)], -1)
        _close(by_parts.get_center(), expected)
        _close(
            on_plane.get_center(),
            torch.stack(
                [
                    torch.as_tensor(plane.n2p(complex(x, y)))
                    for x, y in expected[:, :2].tolist()
                ]
            ),
        )


def test_a_dot_steered_by_a_tracker_renders(tmp_path):
    with Scene() as scene:
        axes = Axes(x_range=[0, 10], y_range=[0, 100, 10])
        x = mn.ValueTracker(0)

        def height(value):
            return 2 * (value - 5) ** 2

        graph = axes.plot(height)
        dot = Dot(axes.c2p(0, height(0)))
        Group(axes, graph, dot).spawn(animate=False)
        x.spawn(animate=False)
        dot.add_updater(
            lambda m, t: m.move_to(axes.c2p(x.get_value(), height(x.get_value())))
        )
        with Sync(runtime=2):
            x.set_value(5)

        result = scene.save_video(tmp_path / "tracker.mp4", SMOKE_TEST)

    assert result.status == "rendered"
