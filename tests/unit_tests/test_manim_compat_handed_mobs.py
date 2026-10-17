"""A Manim-backed Mob handed to a Manim group is the one the group draws.

Manim builds a group out of the very Mobjects it is handed, so a script holds
on to them: ``a = MathTex("a"); VGroup(a, b).arrange(RIGHT)`` and then
``a.set_color(RED)``, or ``Write(a)``. The group's conversion built copies of
its members instead, which left the script's ``a`` an unspawned double: edits
to it changed nothing in the video, arranging the group did not move it, and
the render warned that it was never spawned.
"""

import pytest
import torch

import algan.manim as mn
from algan import LEFT, RED, RIGHT, UP, Scene


def _center(mob):
    return mob.get_center().reshape(-1, 3)[-1]


def _drawn(owner, part):
    """Whether ``part`` is one of the Mobs that render ``owner``."""
    return part in owner.get_descendants() and part in owner.scene.actors


def test_a_vgroup_draws_the_members_it_was_handed():
    with Scene() as scene:
        a, b = mn.MathTex("a"), mn.MathTex("b")
        group = mn.VGroup(a, b).arrange(RIGHT).spawn()

        assert group[0] is a
        assert group[1] is b
        assert _drawn(group, a)
        assert _drawn(group, b)
        # Arranged with the group: the members sit side by side.
        assert _center(a)[0] < _center(b)[0] - 0.1
        assert not scene._never_spawned_root_mobs()


def test_editing_a_member_edits_the_group_on_screen():
    with Scene():
        a, b = mn.MathTex("a"), mn.MathTex("b")
        group = mn.VGroup(a, b).arrange(RIGHT).spawn()
        before = _center(a).clone()
        a.set_color(RED)
        a.shift(UP)

        torch.testing.assert_close(
            _center(group[0]), before + torch.as_tensor(UP).reshape(3)
        )
        assert torch.allclose(
            group[0].get_descendants()[-1].color.reshape(-1, 5)[-1, :3],
            torch.as_tensor(RED).reshape(-1)[:3],
            atol=1e-4,
        )


def test_a_member_manim_moves_while_building_is_drawn_where_it_moved():
    with Scene():
        label = mn.MathTex("x")
        dot = mn.LabeledDot(label).shift(LEFT * 3).spawn()

        assert _drawn(dot, label)
        torch.testing.assert_close(_center(label), _center(dot), atol=1e-3, rtol=0)


def test_a_label_handed_back_is_the_label_handed_in():
    with Scene():
        axes = mn.Axes(x_range=[0, 3], y_range=[0, 9]).spawn()
        graph = axes.plot(lambda x: x**2).spawn()
        label = mn.MathTex("x^2")
        placed = axes.get_graph_label(graph, label=label)

        assert placed is label
        # Moved beside the graph's end, away from where it was built.
        assert _center(label)[0] > 1


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: mn.VGroup(mn.MathTex("a"), mn.MathTex("b")), id="VGroup"),
        pytest.param(
            lambda: mn.Table(
                [["1", "2"]], col_labels=[mn.MathTex("p"), mn.MathTex("q")]
            ),
            id="Table labels",
        ),
        pytest.param(
            lambda: mn.MobjectMatrix([[mn.Circle(), mn.Square()]]), id="MobjectMatrix"
        ),
    ],
)
def test_members_built_inline_are_not_reported_unspawned(build):
    with Scene() as scene:
        build().spawn()
        assert not scene._never_spawned_root_mobs()


def test_a_mob_already_on_screen_is_not_taken_in():
    with Scene():
        circle = mn.Circle().spawn()
        frame = mn.SurroundingRectangle(circle).spawn()
        group = mn.VGroup(circle).spawn()

        assert not circle.parents
        assert circle not in frame.get_descendants()
        assert circle not in group.get_descendants()


def test_a_reference_group_leaves_its_members_alone():
    with Scene():
        a = mn.MathTex("a")
        mn.VGroup(a, add_to_scene=False)

        assert not a.parents
