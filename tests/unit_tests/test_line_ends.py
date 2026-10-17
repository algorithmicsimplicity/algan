"""A Line's ends: where they are, and moving them every frame.

``get_start``/``get_end`` read the first point of every cubic, which is the
start of only the first: an arc reported eight starts and eight segment
lengths, and an arc handed to Manim (``Brace(arc)``) arrived as a chord of its
first segment. ``put_start_and_end_on`` built a Line to become, which cannot be
built for the batch of frames an updater runs on, so the Manim idiom of a
line kept between two moving dots failed the render.
"""

import pytest
import torch

from algan import DOWN, LEFT, ORIGIN, RIGHT, UP, Brace, Dot, Group, Line, Scene, Sync
from algan.mobs.manim_compat import to_manim
from algan.utils.mob_utils import batch_mobs


def test_an_arc_reports_the_ends_and_chord_of_the_whole_arc():
    with Scene():
        arc = Line(LEFT * 2, RIGHT * 2, path_arc=90)

        torch.testing.assert_close(arc.get_start().reshape(-1), LEFT * 2)
        torch.testing.assert_close(
            arc.get_end().reshape(-1), RIGHT * 2, atol=1e-6, rtol=0
        )
        assert arc.get_length().shape == (1, 1)
        assert float(arc.get_length()) == pytest.approx(4)


def test_a_pack_of_lines_reports_one_start_and_length_per_line():
    with Scene():
        lines = batch_mobs(
            [Line(LEFT, RIGHT), Line(DOWN * 2, UP * 2, path_arc=90), Line(ORIGIN, UP)]
        )

        torch.testing.assert_close(
            lines.get_start()[0], torch.stack([LEFT, DOWN * 2, ORIGIN])
        )
        torch.testing.assert_close(
            lines.get_length()[0], torch.tensor([2.0, 4.0, 1.0])
        )
        assert float(lines[1].get_length()) == pytest.approx(4)


def test_an_arc_reaches_manim_whole():
    with Scene():
        arc = Line(LEFT * 2, RIGHT * 2, path_arc=90)

        converted = to_manim(arc)
        assert len(converted.points) == arc.control_points.location.shape[-2]
        assert converted.width == pytest.approx(4)
        assert Brace(arc).manim_mobject.width == pytest.approx(4, abs=0.05)


def _line_between(first, second):
    line = Line(first.get_center(), second.get_center())
    Group(first, second, line).spawn(animate=False)
    line.add_updater(
        lambda mob, t: mob.put_start_and_end_on(first.get_center(), second.get_center())
    )
    return line


@pytest.mark.parametrize(
    ("first", "second"),
    [(LEFT, RIGHT), (RIGHT, LEFT), (ORIGIN, ORIGIN)],
    ids=["apart", "authored_the_other_way", "at_one_point"],
)
def test_an_updater_keeps_a_line_between_two_moving_dots(first, second):
    with Scene() as scene:
        dot, other = Dot(first), Dot(second)
        line = _line_between(dot, other)
        dot.move(UP * 1.5)
        other.move(DOWN * 1.5 + LEFT * 3)
        with Sync():
            dot.move(RIGHT * 4)
            other.move(RIGHT * 6)

        times = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        scene.timeline_manager.set_state_to_times(times)
        torch.testing.assert_close(
            line.get_start(), dot.get_center(), atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            line.get_end(), other.get_center(), atol=1e-5, rtol=0
        )
        assert torch.all(line.stroke_width == 5)
