"""Mob attribute guards and the wording of a missing-attribute error.

- ``mob.position = ...``, ``mob.alpha = 0.5`` and the like succeeded, parking a
  value on the instance that nothing reads. They raise and name the spelling
  that works.
- A typo of a real method or attribute (``Square().rotat``) says "Did you mean
  'rotate'?". The suggestion is worded only when the error is shown, because
  library ``hasattr`` probes reach ``__getattr__`` thousands of times a scene.
- More Manim method names get a pointer to Algan's spelling, and ``remove``'s
  pointed at a method that does not exist (``remove_children``).
- ``mob.color = None`` wrote five zero channels and hid the Mob; a lone float
  as a color and a negative glow were accepted. All three raise.

Record-only; no render.
"""

from __future__ import annotations

import pickle

import pytest

from algan import Scene
from algan.errors import AlganConfigurationError, InvalidColorError
from algan.mobs.shapes_2d import Square
from algan.mobs.shapes_3d import Cone, Cube, Prism
from algan.scene_manager import SceneManager


@pytest.fixture(autouse=True)
def fresh_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


# --------------------------------------------------------------- hints


def test_remove_hint_names_remove_child():
    with pytest.raises(AttributeError, match=r"remove_child\(") as info:
        Square().remove
    assert "remove_children" not in str(info.value)


def test_get_top_has_a_hint():
    with pytest.raises(AttributeError, match=r"get_edge_point\(UP\)"):
        Square().get_top


@pytest.mark.parametrize(
    "name",
    [
        "get_bottom",
        "get_left",
        "get_right",
        "set_x",
        "set_y",
        "set_z",
        "center",
        "align_to",
        "scale_to_fit_width",
        "scale_to_fit_height",
    ],
)
def test_manim_names_hint_at_algan_spellings(name):
    with pytest.raises(AttributeError, match="in Algan use"):
        getattr(Square(), name)


def test_typo_suggests_the_real_method():
    with pytest.raises(AttributeError, match=r"Did you mean 'rotate'\?"):
        Square().rotat


def test_unrelated_name_gets_no_suggestion():
    with pytest.raises(AttributeError) as info:
        Square().zzzzzz
    assert "Did you mean" not in str(info.value)


def test_a_private_name_is_a_plain_attribute_error():
    with pytest.raises(AttributeError) as info:
        Square()._rotat
    assert type(info.value) is AttributeError
    assert "Did you mean" not in str(info.value)


def test_hasattr_does_not_word_the_suggestion(monkeypatch):
    import difflib

    calls = []
    monkeypatch.setattr(
        difflib, "get_close_matches", lambda *a, **k: calls.append(a) or []
    )
    assert not hasattr(Square(), "rotat")
    assert calls == []


def test_the_error_pickles_as_the_message_it_shows():
    with pytest.raises(AttributeError) as info:
        Square().rotat
    restored = pickle.loads(pickle.dumps(info.value))
    assert type(restored) is AttributeError
    assert str(restored) == str(info.value)


# --------------------------------------------------------------- guards


@pytest.mark.parametrize(
    ("name", "spelling"),
    [
        ("position", "location"),
        ("pos", "location"),
        ("rotation", "rotate"),
        ("alpha", "opacity"),
        ("visible", "spawn"),
        ("colour", "color"),
    ],
)
def test_guarded_name_assignment_names_the_working_spelling(name, spelling):
    square = Square()
    with pytest.raises(AttributeError, match=spelling):
        setattr(square, name, 1)
    # The failed write left nothing behind, and a read is an ordinary miss.
    assert name not in vars(square)
    assert not hasattr(square, name)
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(square, name)


def test_guards_do_not_leak_into_suggestions():
    with pytest.raises(AttributeError) as info:
        Square().positio
    assert "'position'" not in str(info.value)


def test_subclasses_keep_their_own_width_height_and_size():
    # width/height/size are deliberately unguarded: these subclasses own them.
    cuboid = Prism(width=2, height=3, depth=4)
    assert cuboid.width == 2
    assert cuboid.height == 3
    cuboid.width = 5
    assert cuboid.width == 5
    assert Cone().height is not None
    assert Cube(size=2).size == 2


# --------------------------------------------------------------- colors


def test_assigning_none_to_color_raises_instead_of_hiding_the_mob():
    with Scene():
        square = Square().spawn(animate=False)
        with pytest.raises(InvalidColorError, match="opacity = 0"):
            square.color = None
        assert float(square.color.reshape(-1)[-1]) == 1.0


def test_a_bare_float_is_refused_on_construction_and_assignment():
    with pytest.raises(InvalidColorError, match="single number"):
        Square(color=1.0)
    square = Square()
    with pytest.raises(InvalidColorError, match="single number"):
        square.color = 1.0


def test_negative_glow_is_rejected_on_assignment():
    with Scene():
        square = Square().spawn(animate=False)
        with pytest.raises(AlganConfigurationError, match="glow must be zero or more"):
            square.glow = -1
        square.glow = 0.5
