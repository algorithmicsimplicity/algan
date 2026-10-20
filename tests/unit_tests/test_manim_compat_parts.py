"""The parts of a Manim-backed Mob are the ones on screen.

Manim hands out pieces of a Mobject all the time -- ``formula[0]``,
``axes.x_axis``, ``axes.get_x_axis()``, ``formula.get_part_by_tex("x")``,
``table.get_rows()`` -- and a script edits them to edit the whole. Indexing a
compatibility Mob raised ``TypeError: Mob object is not iterable``, and every
other route handed back a converted copy that nothing drew: ``x_axis.color =
RED`` and ``get_part_by_tex("x").set_color(RED)`` ran without complaint and
changed nothing in the video.
"""

import cv2
import pytest
import torch

import algan.manim as mn
from algan import (
    BLUE,
    DOWN,
    RED,
    RIGHT,
    SMOKE_TEST,
    UP,
    Arrow,
    Axes,
    Group,
    MathTex,
    Off,
    Scene,
    Table,
)


def _red_pixels(path):
    image = cv2.imread(str(path)).astype(int)
    b, g, r = image[..., 0], image[..., 1], image[..., 2]
    return int(((r > 150) & (g < 110) & (b < 110)).sum())


def _drawn(owner, part):
    """Whether ``part`` is one of the Mobs that render ``owner``."""
    return part in owner.get_descendants() and part in owner.scene.actors


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda eq, ax: eq[0].set_color(RED), id="formula[0]"),
        pytest.param(lambda eq, ax: setattr(eq[1:], "color", RED), id="slice"),
        pytest.param(
            lambda eq, ax: eq.get_part_by_tex("x").set_color(RED), id="get_part_by_tex"
        ),
        pytest.param(lambda eq, ax: setattr(ax.x_axis, "color", RED), id="x_axis"),
        pytest.param(lambda eq, ax: ax.get_x_axis().set_color(RED), id="get_x_axis"),
    ],
)
def test_editing_a_part_edits_the_video(tmp_path, edit):
    def render(name, edited):
        with Scene() as scene:
            eq = MathTex("x", "+", "y").scale(2).move(UP * 2).spawn(animate=False)
            ax = Axes(x_length=6, y_length=3).move(DOWN).spawn(animate=False)
            if edited:
                with Off():
                    edit(eq, ax)
            frame = scene.save_frame(
                tmp_path / f"{name}.png", SMOKE_TEST.set(resolution=(320, 180))
            )
        return _red_pixels(frame.output_path)

    assert render("plain", False) == 0
    assert render("edited", True) > 20


def test_a_formula_indexes_like_manims():
    with Scene():
        eq = MathTex("x^2", "+", "y").spawn(animate=False)
        parts = list(eq)

        assert len(eq) == 3
        assert [eq[0], eq[1], eq[2]] == parts
        assert all(_drawn(eq, part) for part in parts)
        assert eq[-1] is eq[2]
        # A substring is indexed by glyph, as in Manim.
        assert len(eq[0]) == 2
        assert _drawn(eq, eq[0][1])

        x, plus, y = eq
        assert (x, plus, y) == tuple(parts)

        view = eq[1:]
        assert isinstance(view, Group)
        assert list(view) == parts[1:]
        assert view not in eq.scene.actors
        # A view: it recurses over the parts without becoming their parent.
        assert all(view not in part.parents for part in view)


def test_manim_hands_back_the_parts_on_screen():
    with Scene() as scene:
        ax = Axes(x_range=[-3, 3], y_range=[-2, 2])
        table = Table([["1", "2"], ["3", "4"]])
        graph = mn.Graph([1, 2, 3], [(1, 2), (2, 3)])
        Group(ax, table, graph).spawn(animate=False)
        actors = len(scene.actors)

        assert ax.x_axis is ax.get_x_axis() is ax[0]
        assert _drawn(ax, ax.x_axis)
        torch.testing.assert_close(
            torch.as_tensor(ax.x_axis.n2p(2), dtype=torch.float32),
            torch.as_tensor(ax.c2p(2, 0), dtype=torch.float32),
        )
        row = table.get_rows()[1]
        assert all(_drawn(table, entry) for entry in row)
        assert graph[2] is graph.vertices[2]
        assert _drawn(graph, graph[2])
        assert _drawn(graph, graph.edges[(1, 2)])
        # Reading a part is a query: it registers nothing.
        assert len(scene.actors) == actors

        curve = ax.plot(lambda x: x / 2)
        assert curve in scene.actors
        assert curve not in ax.get_descendants()


def test_a_part_held_through_an_edit_of_the_whole_stays_on_screen():
    with Scene():
        arrow = Arrow(2 * RIGHT, 2 * UP).spawn(animate=False)
        eq = MathTex("x", "+", "y").spawn(animate=False)
        x, tip = eq[0], arrow.tip

        eq.set_color_by_tex("y", BLUE)
        arrow.put_start_and_end_on(DOWN, UP)
        eq.scale(2)

        assert eq[0] is x
        assert arrow.tip is tip
        assert _drawn(eq, x)
        assert not x.is_despawned()
        # Handed back to Manim, the part is the one Manim's own methods know.
        assert eq.index_of_part(eq[1]) == 1


def test_an_edit_that_adds_parts_rebuilds_them_on_screen():
    with Scene():
        ax = Axes(x_range=[0, 4], y_range=[0, 4]).spawn(animate=False)
        before = ax.x_axis

        ax.add_coordinates()

        assert ax.x_axis is not before
        assert before.is_despawned()
        assert _drawn(ax, ax.x_axis)
        assert not ax.x_axis.is_despawned()


def test_a_part_replaced_by_become_survives_later_manim_edits():
    """``become`` splices its result in; the next Manim edit has to keep it."""
    with Scene():
        eq = MathTex("x", "+", "y").spawn(animate=False)
        z = MathTex("z", add_to_scene=False).move_to(eq[0])
        with Off():
            replacement = eq[0].become(z)

        assert eq[0] is replacement
        eq.set_color_by_tex("y", BLUE)
        eq.scale(2)

        assert eq[0] is replacement
        assert not replacement.is_despawned()
        assert len(replacement.get_family()) == len(z.get_family())


def test_a_packed_conversion_says_its_parts_cannot_be_picked_out():
    with Scene():
        eq = MathTex("x", "+", "y", batch=True)
        with pytest.raises(TypeError, match="batch=True"):
            eq[0]
