"""The plain geometry declarations on a Mob, and assigning them after spawn.

``two_sided``, ``closed_shell``, ``casts_shadows``, ``receives_shadows`` and
``casts_caustics`` are read when render primitives are built, so whatever
value they hold then applies to every frame. Render-free.
"""

import warnings

import pytest

from algan import Group, Scene, SceneManager, Sphere, Square
from algan.errors import AlganWarning

_FLAGS = (
    "two_sided",
    "closed_shell",
    "casts_shadows",
    "receives_shadows",
    "casts_caustics",
)


@pytest.fixture(autouse=True)
def fresh_scene_stack():
    SceneManager.reset()
    yield
    SceneManager.reset()


@pytest.mark.parametrize("flag", _FLAGS)
def test_assigning_a_declaration_after_spawn_warns_that_it_is_whole_video(flag):
    # ``sphere.casts_shadows = False`` three seconds in reads as "no shadow
    # from here on", but the renderer reads the flag when it builds the
    # primitive, so the shadow went from frame 0 -- without a word, where
    # set_material in the same position raises.
    with Scene():
        sphere = Sphere().spawn(animate=False)
        value = not getattr(sphere, flag)
        with pytest.warns(AlganWarning, match="every frame"):
            setattr(sphere, flag, value)
        assert getattr(sphere, flag) is value


def test_an_unspawned_group_with_a_spawned_member_warns_too():
    with Scene():
        member = Sphere().spawn(animate=False)
        g = Group(member, Sphere())
        assert not g.is_spawned()
        with pytest.warns(AlganWarning, match="every frame"):
            g.casts_shadows = False


def test_declarations_before_spawn_and_class_defaults_are_untouched():
    with Scene():
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sphere = Sphere()
            sphere.casts_shadows = False
            sphere.two_sided = True
        assert sphere.casts_shadows is False
        assert sphere.two_sided is True
        # The built-in solids declare themselves one-sided on the class.
        assert Sphere.two_sided is False
        assert Sphere().two_sided is False
        assert Square().two_sided is True
        assert Square.casts_shadows is True


def test_become_adopts_the_target_declarations_without_warning():
    with Scene():
        source = Square().spawn(animate=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            source.become(Sphere(add_to_scene=False))
        assert not [w for w in caught if "every frame" in str(w.message)]
