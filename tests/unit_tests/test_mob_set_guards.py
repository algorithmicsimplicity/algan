"""``Mob.set`` / ``set_non_recursive`` and the state they leave behind.

Render-free: each test records on a fresh Scene and reads the timeline back.
"""

import pytest
import torch

from algan import RED, UP, Circle, Group, Off, Scene, SceneManager, Square


@pytest.fixture(autouse=True)
def fresh_scene_stack():
    SceneManager.reset()
    yield
    SceneManager.reset()


@pytest.mark.parametrize(
    "bad_kwargs",
    [{"colour": RED}, {"opacity": 2.0}],
    ids=["unknown-name", "out-of-range"],
)
def test_a_failed_set_non_recursive_still_lets_transforms_carry_children(bad_kwargs):
    # set_non_recursive raised between switching recursion off and back on,
    # so after a caught error (a notebook, the viewer) every later move,
    # rotate or scale of that Mob silently left its children behind.
    with Scene(), Off():
        child = Circle().move(UP * 2)
        g = Group(Square(), child)
        with pytest.raises(Exception):  # noqa: B017 -- either error will do
            g.set_non_recursive(**bad_kwargs)
        assert g._prevent_recursive_sets is False
        before = child.location.clone()
        g.move(UP)
        assert torch.allclose(child.location - before, torch.tensor([0.0, 1.0, 0.0]))
