"""A packed Manim-backed Mob hands Manim the geometry it draws.

``batch=True`` packs a conversion's parts into one Mob, and the parts it was
packed from stay behind as scaffolding, in the pose they were converted in.
Every Manim edit synchronized the backing Mobject from that scaffolding, so
after a move ``formula.scale(2)`` scaled the formula where it had been built,
and the formula jumped back there. A Manim method that edits the Mob also
unpacked it for good, by grafting in an unpacked conversion.
"""

import manim as mn
import pytest
import torch

import algan.manim as amn
from algan import BLUE, RIGHT, UP, Group, ManimMob, MathTex, Off, Scene


def _center(mob):
    return mob.get_center().reshape(-1, 3)[0]


@pytest.mark.parametrize("edit", ["scale", "set_color_by_tex"])
def test_a_packed_formula_is_edited_where_it_is(edit):
    edits = {
        "scale": lambda eq: eq.scale(2),
        "set_color_by_tex": lambda eq: eq.set_color_by_tex("y", BLUE),
    }
    centers = {}
    with Scene():
        for batch in (False, True):
            eq = MathTex("x", "+", "y", batch=batch).spawn(animate=False)
            with Off():
                Group(eq).move(UP)
                eq.move(RIGHT * 2)
                edits[edit](eq)
            centers[batch] = _center(eq)
    torch.testing.assert_close(centers[True], centers[False], atol=1e-4, rtol=0)


def test_a_manim_edit_keeps_a_packed_formula_packed():
    with Scene() as scene:
        eq = MathTex("x", "+", "y", batch=True).spawn(animate=False)
        (pack,) = eq.get_non_component_children()
        actors = len(scene.actors)

        eq.set_color_by_tex("y", BLUE)

        assert eq.get_non_component_children() == [pack]
        assert len(scene.actors) == actors


def test_a_moved_packed_manim_mob_reaches_manim_where_it_is():
    with Scene():
        pair = ManimMob(
            mn.VGroup(mn.Square(), mn.Circle().shift(mn.RIGHT * 3)), batch=True
        )
        pair.move(UP * 2)

        box = amn.SurroundingRectangle(pair)
        torch.testing.assert_close(
            _center(box), torch.tensor([1.5, 2.0, 0.0]), atol=1e-4, rtol=0
        )
