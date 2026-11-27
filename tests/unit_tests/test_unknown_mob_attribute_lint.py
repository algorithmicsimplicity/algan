"""A render warns about names set on a Mob that look like misspelt Algan attributes.

``sq.opactiy = 0.5`` lands in the instance ``__dict__`` and changes nothing; the
render job warns once per (class, name) when the name is a near-miss of a real
attribute or a known wrong spelling. Record-only: no render runs. The wiring into
the render job is asserted from source because ``get_frames`` needs a renderer.
"""

from __future__ import annotations

import inspect
import warnings

import pytest

from algan.animatable_base.mob import (
    Mob,
    find_unknown_mob_attributes,
    warn_unknown_mob_attributes,
)
from algan.constants.color import RED
from algan.errors import UnknownMobAttributeWarning
from algan.mobs.shapes_2d import Circle, Square
from algan.mobs.shapes_3d import Prism, Sphere
from algan.render_loop import RenderLoopMixin
from algan.scene_manager import SceneManager


@pytest.fixture(autouse=True)
def fresh_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _scene():
    return SceneManager.instance().current_scene


def _lint():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_unknown_mob_attributes(_scene())
    return [w for w in caught if issubclass(w.category, UnknownMobAttributeWarning)]


def test_typo_warns_with_suggestion():
    sq = Square().spawn()
    sq.opactiy = 0.5
    (w,) = _lint()
    text = str(w.message)
    assert "`opactiy` was set on a Square" in text
    assert "changed nothing in the video" in text
    assert "Did you mean `opacity`?" in text


def test_warning_is_not_in_the_public_namespace():
    import algan

    assert "UnknownMobAttributeWarning" not in algan.__all__
    assert UnknownMobAttributeWarning.__name__ == "UnknownMobAttributeWarning"
    assert issubclass(UnknownMobAttributeWarning, UserWarning)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("width", "scale_to_width"),
        ("height", "scale_to_height"),
        ("size", "scale"),
        ("fill_color", "`color`"),
        ("fill_colour", "`color`"),
        ("loc", "`location`"),
    ],
)
def test_known_wrong_spellings_name_the_working_one(name, expected):
    sq = Square().spawn()
    setattr(sq, name, 2)
    (w,) = _lint()
    assert expected in str(w.message)


def test_class_defined_name_is_never_flagged():
    prism = Prism().spawn()
    prism.width = 3  # a real attribute on Prism
    assert find_unknown_mob_attributes([prism]) == []
    assert _lint() == []


def test_user_data_that_resembles_nothing_is_not_flagged():
    sq = Square().spawn()
    sq.my_label = "a"
    sq.frame_counter = 3
    assert _lint() == []


def test_names_set_during_construction_are_not_flagged():
    class Tagged(Square):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.width_hint = 2  # set by the class's own constructor
            self.loc = 1  # even a wrong spelling, because it is library code

    class Sub(Tagged):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.size = 4

    Sub().spawn()
    assert _lint() == []


def test_only_names_added_after_construction_count():
    class Tagged(Square):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.height = 2

    t = Tagged().spawn()
    t.width = 3
    (w,) = _lint()
    assert "`width` was set on a Tagged" in str(w.message)


def test_one_warning_per_class_and_name():
    a, b = Square().spawn(), Square().spawn()
    a.opactiy = 0.2
    b.opactiy = 0.4
    c = Circle().spawn()
    c.opactiy = 0.4
    assert len(_lint()) == 2
    assert _lint() == []  # already reported for this scene


def test_correct_scripts_are_silent():
    sq = Square().spawn()
    sq.opacity = 0.5
    sq.color = RED
    sq.scale_to_width(2)
    sq.move([1, 0, 0])
    clone = sq.clone()
    sq.become(Circle())
    sphere = Sphere().spawn()
    sphere.roughness = 0.3
    assert _lint() == []
    # Audit mode lists at most what the author set after construction. Whether
    # ``roughness`` shows up depends on whether an earlier test already made
    # the class declare it (generated accessors are cached on the class), so
    # only its being the sole possible entry is asserted.
    audit = find_unknown_mob_attributes([sq, clone, sphere], gate=False)
    assert all(entry == ("Sphere", "roughness", "") for entry in audit)


def test_mob_without_a_construction_record_is_skipped():
    sq = Square()
    sq.__dict__.pop("_ctor_names")
    sq.opactiy = 1
    assert find_unknown_mob_attributes([sq]) == []


def test_lint_never_raises():
    class Hostile:
        @property
        def __dict__(self):
            raise RuntimeError("boom")

    assert find_unknown_mob_attributes([Hostile(), None]) == []
    warn_unknown_mob_attributes(object())


def test_escalated_warning_propagates():
    sq = Square().spawn()
    sq.opactiy = 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnknownMobAttributeWarning)
        with pytest.raises(UnknownMobAttributeWarning):
            warn_unknown_mob_attributes(_scene())


def test_init_wrapper_is_transparent():
    assert inspect.signature(Square.__init__) == inspect.signature(
        Square.__init__.__wrapped__
    )
    assert getattr(Mob.__init__, "_records_ctor_names", False)


def test_render_mixin_method_warns_on_a_stub_scene():
    class Stub(RenderLoopMixin):
        pass

    scene = _scene()
    sq = Square().spawn()
    sq.opactiy = 0.5
    stub = Stub.__new__(Stub)
    stub.__dict__.update(
        actors=list(scene.actors), camera=scene.camera, light_sources=[]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stub._warn_unknown_mob_attributes()
    assert any(issubclass(w.category, UnknownMobAttributeWarning) for w in caught)


def test_every_render_entry_point_reaches_the_lint():
    """``get_frames`` starts every render job and calls the lint; save_video,
    save_frame and the viewer all render through it.
    """
    assert "_warn_unknown_mob_attributes()" in inspect.getsource(
        RenderLoopMixin.get_frames
    )
    from algan import _still_frames, render_loop
    from algan.viewer import session

    for module in (_still_frames, session, render_loop):
        assert ".get_frames(" in inspect.getsource(module)


def test_library_set_names_after_construction_are_not_flagged():
    from algan.rendering.shaders.materials import MeshStandardMaterial

    sq = Square()
    sq.set_material(MeshStandardMaterial())  # stores `material` post-construction
    sq.spawn()
    assert _lint() == []


def test_accessors_built_on_first_use_are_not_flagged():
    # ``set_opacity`` is built on first use and kept in the instance dict
    # (Animatable._generated_accessor); it once read as a near-miss of
    # ``stroke_opacity``.
    sq = Square().spawn()
    sq.set_opacity(0.5)
    sq.get_color()
    assert "set_opacity" in vars(sq)
    assert find_unknown_mob_attributes([sq], gate=False) == []
    assert _lint() == []
