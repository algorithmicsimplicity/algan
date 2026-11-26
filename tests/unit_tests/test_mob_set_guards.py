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


def test_set_refuses_an_attribute_only_another_mob_has():
    # The check admitted any name in the Scene-wide attribute registry, so
    # once one Sphere had a material, ``square.set(roughness=0.9)`` recorded
    # a second of nothing. The same held for a second Sphere, because the
    # material's attributes are attached to the Sphere class.
    from algan import Sphere
    from algan.rendering.shaders.materials import MeshStandardMaterial

    with Scene(), Off():
        shiny = Sphere().set_material(MeshStandardMaterial(roughness=0.2))
        for other in (Square(), Sphere()):
            with pytest.raises(AttributeError, match="roughness"):
                other.set(roughness=0.9)
            with pytest.raises(AttributeError, match="roughness"):
                other.set_non_recursive(roughness=0.9)

        # A Group passes the write on to the member that has it ...
        g = Group(shiny, Square())
        g.set(roughness=0.5)
        assert float(shiny.roughness.flatten()[0]) == pytest.approx(0.5)
        # ... but on its own, without recursion, nothing would take it.
        with pytest.raises(AttributeError, match="roughness"):
            g.set_non_recursive(roughness=0.7)
        assert shiny.set(roughness=0.3) is shiny


def test_detaching_or_morphing_an_unspawned_mob_leaves_it_unspawned():
    # detach_history re-spawned unconditionally, so configuring a shape
    # before spawning it -- ``s.become(Circle(add_to_scene=False))`` -- put
    # it on screen at that moment.
    with Scene():
        s = Square()
        s.become(Circle(add_to_scene=False))
        assert not s.is_spawned()
        t = Square()
        assert t.detach_history() is t
        assert not t.is_spawned()

        # A spawned Mob is still on screen afterwards, as before.
        u = Square().spawn(animate=False)
        u.become(Circle(add_to_scene=False))
        assert u.is_spawned()
