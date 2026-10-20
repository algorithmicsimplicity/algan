"""A Manim method that edits a spawned Manim-backed Mob leaves no copy behind.

A delegated edit (``next_to``, ``align_to``, ``add_tip``, ...) converts the
edited Manim object again and grafts the new children in. The children they
replaced were left spawned, and so was the conversion's own root, so a copy of
the Mob stayed on screen, frozen where the edit put it, while the Mob itself
moved on.
"""

import pytest
import torch

from algan import LEFT, RIGHT, UP, Arrow, Brace, Dot, Group, MathTex, Scene, Square


def _visible_at(scene, time):
    """The ids of every actor with any opacity at ``time``."""
    scene.timeline_manager.set_state_to_times(torch.tensor([float(time)]))
    return {id(actor) for actor in scene.actors if float(actor.opacity.max()) > 0}


def _family(mob):
    return {id(part) for part in (mob, *mob.get_descendants())}


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: Arrow(LEFT, RIGHT), id="Arrow"),
        pytest.param(lambda: MathTex("x"), id="MathTex"),
        pytest.param(lambda: Brace(Square(add_to_scene=False)), id="Brace"),
    ],
)
def test_an_edit_on_a_spawned_mob_leaves_only_the_mob(build):
    with Scene() as scene:
        dot = Dot(LEFT * 3)
        mob = build()
        group = Group(dot, mob).spawn(animate=False)
        mob.next_to(dot, UP)
        mob.move(RIGHT * 6)

        visible = _visible_at(scene, 1.0)
        assert visible <= _family(group)
        # The grafted children are the ones on screen now.
        assert any(id(part) in visible for part in mob.get_descendants())
