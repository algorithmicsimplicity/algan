"""Pure-Python tests for :mod:`algan.settings.abstract_settings`."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import ClassVar

import pytest

from algan.errors import AlganConfigurationError
from algan.settings.abstract_settings import (
    Settings,
    _alias_setter,
    _is_special_var,
    settings_aliases,
)


@settings_aliases(fps="frames_per_second")
@dataclass
class Video(Settings):
    frames_per_second: int = 30
    name: str = "clip"
    tags: list = field(default_factory=list)
    LIMIT: ClassVar[int] = 1000
    _private: int = 0

    def __post_init__(self):
        if self.frames_per_second <= 0:
            raise ValueError("frames_per_second must be positive")


@dataclass
class Plain(Settings):
    a: int = 1
    b: list = field(default_factory=lambda: [1, 2])


class NonDataclass(Settings):
    x: int
    y: int

    def __init__(self, x=1, y=2):
        self.x = x
        self.y = y

    def _validate(self):
        if self.x < 0:
            raise AlganConfigurationError("x must be >= 0")


# --- _is_special_var -------------------------------------------------------


@pytest.mark.parametrize(
    "annotation",
    [
        "ClassVar[int]",
        "InitVar[int]",
        "KW_ONLY",
        ClassVar[int],
        ClassVar,
        dataclasses.InitVar[int],
        dataclasses.KW_ONLY,
    ],
)
def test_is_special_var_true(annotation):
    assert _is_special_var(annotation)


@pytest.mark.parametrize("annotation", ["int", "list[str]", int, list[int], None])
def test_is_special_var_false(annotation):
    assert not _is_special_var(annotation)


# --- declared fields and generated setters ---------------------------------


def test_declared_field_names_skip_classvar_and_private():
    assert Video._declared_field_names() == {"frames_per_second", "name", "tags"}


def test_generated_setters_exist_and_are_named():
    assert Video.set_frames_per_second.__name__ == "set_frames_per_second"
    assert not hasattr(Video, "set_LIMIT")
    assert not hasattr(Video, "set__private")
    v = Video()
    assert v.set_name("other") is v
    assert v.name == "other"


def test_existing_setter_is_not_overwritten():
    @dataclass
    class Custom(Settings):
        a: int = 1

        def set_a(self, value):
            return "custom"

    assert Custom().set_a(5) == "custom"


def test_inherited_fields_get_setters():
    @dataclass
    class Child(Plain):
        c: int = 3

    child = Child()
    child.set_a(10)
    child.set_c(11)
    assert (child.a, child.c) == (10, 11)
    assert Child._declared_field_names() == {"a", "b", "c"}


# --- set() -------------------------------------------------------------------


def test_set_mutates_in_place_and_returns_self():
    v = Video()
    assert v.set(frames_per_second=60, name="x") is v
    assert v.frames_per_second == 60
    assert v.name == "x"


def test_set_preserves_identity_of_untouched_fields():
    v = Video(tags=["a"])
    tags = v.tags
    v.set(frames_per_second=24)
    assert v.tags is tags


def test_set_deepcopies_changed_values():
    v = Video()
    new_tags = ["x"]
    v.set(tags=new_tags)
    assert v.tags == ["x"]
    assert v.tags is not new_tags


def test_set_unknown_key_suggests_close_match():
    with pytest.raises(
        AlganConfigurationError, match="Did you mean 'frames_per_second'"
    ):
        Video().set(frames_per_secnd=1)


def test_set_unknown_key_lists_aliases():
    with pytest.raises(AlganConfigurationError, match="Aliases: fps"):
        Video().set(zzz=1)


def test_set_validation_error_is_wrapped_and_leaves_object_unchanged():
    v = Video(frames_per_second=25)
    with pytest.raises(AlganConfigurationError, match="must be positive"):
        v.set(frames_per_second=0)
    assert v.frames_per_second == 25


def test_set_from_source():
    src = Video(frames_per_second=50, name="src")
    v = Video()
    v.set(src, name="mine")
    assert (v.frames_per_second, v.name) == (50, "mine")


def test_set_from_source_of_wrong_type():
    with pytest.raises(AlganConfigurationError, match="expected another Video"):
        Video().set(Plain())


def test_set_source_of_subclass_rejected():
    @dataclass
    class Child(Plain):
        pass

    with pytest.raises(AlganConfigurationError):
        Plain().set(Child())


def test_set_none_value_is_stored():
    @dataclass
    class Opt(Settings):
        a: object = 1

    o = Opt()
    o.set(a=None)
    assert o.a is None


# --- assignment ---------------------------------------------------------------


def test_assignment_routes_through_set_validation():
    v = Video()
    with pytest.raises(AlganConfigurationError):
        v.frames_per_second = 0
    assert v.frames_per_second == 30
    v.frames_per_second = 12
    assert v.frames_per_second == 12


def test_assignment_of_unknown_public_attribute_rejected():
    v = Video()
    with pytest.raises(
        AlganConfigurationError, match="Unknown Video setting 'frame_rate'"
    ):
        v.frame_rate = 60
    assert "frame_rate" not in v.__dict__


def test_assignment_of_private_attribute_allowed():
    v = Video()
    v._scratch = 5
    assert v._scratch == 5


def test_construction_validates():
    with pytest.raises(ValueError):
        Video(frames_per_second=-1)


# --- presets ----------------------------------------------------------------


def test_preset_set_returns_copy():
    preset = Video(frames_per_second=30).as_preset()
    assert preset.is_preset
    new = preset.set(frames_per_second=60)
    assert new is not preset
    assert new.is_preset
    assert preset.frames_per_second == 30
    assert new.frames_per_second == 60


def test_preset_assignment_rejected_including_alias():
    preset = Video().as_preset()
    with pytest.raises(AlganConfigurationError, match="immutable"):
        preset.frames_per_second = 5
    with pytest.raises(AlganConfigurationError, match="immutable"):
        preset.fps = 5
    assert preset.frames_per_second == 30


def test_preset_override_rejected():
    preset = Video().as_preset()
    with (
        pytest.raises(AlganConfigurationError, match="immutable preset"),
        preset.override(frames_per_second=1),
    ):
        pass


def test_preset_set_validation_error():
    with pytest.raises(AlganConfigurationError):
        Video().as_preset().set(frames_per_second=0)


# --- clone / as_mutable / to_dict -----------------------------------------------


def test_clone_is_independent():
    v = Video(tags=["a"])
    c = v.clone()
    c.tags.append("b")
    assert v.tags == ["a"]
    assert not c.is_preset


def test_clone_preset_flag():
    v = Video()
    assert v.clone(preset=True).is_preset
    p = v.as_preset()
    assert p.clone().is_preset
    m = p.as_mutable()
    assert not m.is_preset
    m.frames_per_second = 5
    assert p.frames_per_second == 30


def test_to_dict_uses_declared_names_and_deep_copies():
    v = Video(tags=["a"])
    d = v.to_dict()
    assert d == {"frames_per_second": 30, "name": "clip", "tags": ["a"]}
    assert "fps" not in d
    assert d["tags"] is not v.tags


def test_to_dict_round_trips():
    v = Video(frames_per_second=48, name="n")
    w = Video()
    w.set(**v.to_dict())
    assert w == v


# --- override ------------------------------------------------------------------


def test_override_restores_values():
    v = Video()
    with v.override(frames_per_second=60, name="tmp") as yielded:
        assert yielded is v
        assert v.frames_per_second == 60
        assert v.name == "tmp"
    assert v.frames_per_second == 30
    assert v.name == "clip"


def test_override_restores_on_exception():
    v = Video()

    def check_then_raise():
        assert v.frames_per_second == 60
        raise RuntimeError

    with pytest.raises(RuntimeError), v.override(fps=60):
        check_then_raise()
    assert v.frames_per_second == 30


def test_override_invalid_value_leaves_object_unchanged():
    v = Video()
    with pytest.raises(AlganConfigurationError), v.override(frames_per_second=0):
        pytest.fail("body must not run")
    assert v.frames_per_second == 30


def test_nested_override():
    v = Video()
    with v.override(frames_per_second=1):
        with v.override(frames_per_second=2):
            assert v.frames_per_second == 2
        assert v.frames_per_second == 1
    assert v.frames_per_second == 30


@pytest.mark.xfail(
    strict=True,
    reason="BUG: override() restores from deep-copied to_dict(), so every "
    "untouched field is replaced by an equal copy on exit, breaking identity",
)
def test_override_preserves_identity_of_untouched_fields():
    v = Video(tags=["a"])
    tags = v.tags
    with v.override(frames_per_second=60):
        pass
    assert v.tags is tags


# --- aliases ------------------------------------------------------------------


def test_alias_in_constructor_set_and_attribute():
    v = Video(fps=24)
    assert v.frames_per_second == 24
    assert v.fps == 24
    v.set(fps=25)
    assert v.frames_per_second == 25
    v.fps = 26
    assert v.frames_per_second == 26
    assert "fps" not in v.__dict__
    assert v.set_fps(27) is v
    assert v.frames_per_second == 27


def test_alias_listed_in_dir_not_in_fields():
    v = Video()
    assert "fps" in dir(v)
    assert "fps" not in {f.name for f in dataclasses.fields(v)}


def test_alias_and_declared_name_together_rejected():
    with pytest.raises(AlganConfigurationError, match="given twice"):
        Video().set(fps=1, frames_per_second=2)
    with pytest.raises(AlganConfigurationError, match="given twice"):
        Video(fps=1, frames_per_second=2)


def test_alias_assignment_validates():
    v = Video()
    with pytest.raises(AlganConfigurationError):
        v.fps = 0
    assert v.frames_per_second == 30


def test_missing_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        Video().nope


def test_alias_to_unknown_field_rejected():
    with pytest.raises(AlganConfigurationError, match="no setting 'missing'"):

        @settings_aliases(m="missing")
        @dataclass
        class Bad(Settings):
            a: int = 1


def test_alias_shadowing_field_rejected():
    with pytest.raises(AlganConfigurationError, match="already the name"):

        @settings_aliases(a="b")
        @dataclass
        class Bad(Settings):
            a: int = 1
            b: int = 2


def test_aliases_do_not_leak_to_other_classes():
    assert Plain._ALIASES == {}
    assert Settings._ALIASES == {}
    with pytest.raises(AlganConfigurationError):
        Plain().set(fps=1)


def test_stacked_alias_decorators_merge():
    @settings_aliases(x="a")
    @settings_aliases(y="a")
    @dataclass
    class Two(Settings):
        a: int = 0

    assert Two._ALIASES == {"x": "a", "y": "a"}
    assert Two(x=3).y == 3


def test_subclass_inherits_aliases_for_set():
    @dataclass
    class Child(Video):
        extra: int = 0

    c = Child()
    c.set(fps=10)
    assert c.frames_per_second == 10
    assert c.fps == 10


@pytest.mark.xfail(
    strict=True,
    reason="BUG: settings_aliases wraps only the decorated class's __init__; a "
    "@dataclass subclass regenerates __init__ and rejects the inherited alias",
)
def test_subclass_inherits_alias_in_constructor():
    @dataclass
    class Child(Video):
        extra: int = 0

    assert Child(fps=10).frames_per_second == 10


def test_alias_setter_function():
    s = _alias_setter("fps", "frames_per_second")
    assert s.__name__ == "set_fps"
    assert s.__qualname__ == "set_fps"
    assert "frames_per_second" in s.__doc__
    v = Video()
    assert s(v, 7) is v
    assert v.frames_per_second == 7


# --- non-dataclass subclass ----------------------------------------------------


def test_non_dataclass_set_and_validate():
    n = NonDataclass()
    xid = n.y
    n.set(x=5)
    assert (n.x, n.y) == (5, xid)
    with pytest.raises(AlganConfigurationError, match="x must be"):
        n.set(x=-1)
    assert n.x == 5


def test_non_dataclass_assignment_validates():
    n = NonDataclass()
    with pytest.raises(AlganConfigurationError):
        n.x = -3
    n.y = 9
    assert n.y == 9


def test_non_dataclass_preset():
    p = NonDataclass().as_preset()
    q = p.set(x=4)
    assert q.is_preset
    assert q.x == 4
    assert p.x == 1
    with pytest.raises(AlganConfigurationError):
        p.x = 3


def test_non_dataclass_unknown_key():
    with pytest.raises(AlganConfigurationError, match="Unknown NonDataclass"):
        NonDataclass().set(z=1)
