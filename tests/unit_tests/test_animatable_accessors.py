"""A Mob's ``set_<attr>`` / ``get_<attr>`` accessors are built on first use.

They used to be installed as a closure pair per attribute on every Mob, and on
every view of a packed Mob, which was most of the objects a view allocated.
Built lazily, they must still answer exactly as the eager ones did.
"""

from __future__ import annotations

import inspect

import pytest
import torch

import algan.manim as mn
from algan import RED, RIGHT, DecimalNumber, SceneManager, Square
from algan.animatable_base.animatable import Animatable
from algan.mobs.bezier_circuit import BezierCircuitCubic


@pytest.fixture(autouse=True)
def fresh_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _accessor_names(mob):
    return {name for name in dir(mob) if name.startswith(("set_", "get_"))}


def test_an_accessor_is_built_on_first_use_and_kept():
    square = Square()
    assert "get_glow" not in square.__dict__
    assert {"set_glow", "get_glow", "set_color", "get_color"} <= set(dir(square))
    assert torch.equal(square.get_glow(), square.glow)
    assert square.get_glow is square.get_glow
    square.set_color(RED)
    reference = Square()
    reference.color = RED
    assert torch.equal(square.color, reference.color)
    with pytest.raises(AttributeError, match="no attribute 'get_nothing'"):
        square.get_nothing  # noqa: B018


def test_an_accessor_still_shadows_the_class_method_of_its_name():
    # Mob defines its own ``set_location``; the generated accessor has always
    # shadowed it on the instance, and it is the one that answers a Manim
    # timing keyword with the context to write instead.
    square = Square().spawn()
    assert "set_location" in square.__dict__
    with pytest.raises(TypeError, match=r"with Seq\(runtime=2\)"):
        square.set_location(RIGHT, runtime=2)


def test_an_attribute_registered_after_construction_keeps_its_class_setter():
    number = DecimalNumber(1.0)
    assert number.set_value.__func__ is type(number).set_value
    number.set_value(3.0)
    assert float(number.value) == 3.0


def test_a_manim_compatible_mob_answers_its_algan_accessors_itself():
    square = mn.Square()
    assert square.set_color.__qualname__.endswith("setattr_general")
    square.set_color(RED)
    reference = mn.Square()
    reference.color = RED
    assert torch.equal(square.color, reference.color)


def test_views_and_copies_hold_their_sources_accessors_unbuilt():
    square = torch.tensor(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]
    )
    points = torch.cat(
        [
            torch.stack([a * (1 - t) + b * t for t in torch.linspace(0, 1, 4)])
            for a, b in zip(square, square.roll(-1, 0))
        ]
    )
    pack = BezierCircuitCubic.from_batches(
        [points + i for i in range(3)], add_to_scene=False
    )
    view = pack[1]
    copy = pack.clone(add_to_scene=False, spawn=False)
    for clone in (view, copy):
        assert _accessor_names(clone) == _accessor_names(pack)
        for part in clone.get_descendants():
            built = {k for k in part.__dict__ if k.startswith(("set_", "get_"))}
            assert built <= {"set_location"}
    assert torch.equal(view.get_location(), view.location)


def _subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


def test_every_getattr_override_builds_accessors_first():
    # An accessor not yet built is found through ``__getattr__``. An override
    # that answers unknown names itself -- the Manim-compatible Mobs hand them
    # to their Manim object -- would answer ``set_color`` wrongly unless it
    # asks ``_generated_accessor`` first.
    overrides = {
        cls
        for cls in _subclasses(Animatable)
        if "__getattr__" in vars(cls) and cls.__module__.startswith("algan.")
    }
    assert overrides
    for cls in overrides:
        source = inspect.getsource(vars(cls)["__getattr__"])
        assert "_generated_accessor" in source, cls.__qualname__
