"""Contracts of the presentation contexts: ``SlideShow``, ``NoExtra``, ``OnInit``
and ``ComposedEasing``.

These four are the contexts that do something *besides* timing: they hook
into spawn, despawn or construction, or they bend the rate function of what
is recorded inside them. All of it is recorded, not rendered, so every check
here materializes the timeline at chosen times and reads the attributes back --
no render, no Taichi.
"""

from __future__ import annotations

import pytest
import torch

from algan import (
    RIGHT,
    Circle,
    ComposedEasing,
    MeshStandardMaterial,
    NoExtra,
    Off,
    OnInit,
    Scene,
    Seq,
    SlideShow,
    Square,
)
from algan.constants import easings


@pytest.fixture
def scene():
    with Scene() as active:
        yield active


def _cursor(scene):
    return float(scene.animation_manager.context.timespan.current_time)


def _opacities(scene, mobs, times):
    """Each Mob's peak opacity at each time, as ``[[mob0_t0, mob1_t0], ...]``."""
    scene.timeline_manager.set_state_to_times(torch.tensor(times, dtype=torch.float))
    return [
        [round(float(mob.opacity[i].max()), 3) for mob in mobs]
        for i in range(len(times))
    ]


def _x_positions(scene, mob, times):
    scene.timeline_manager.set_state_to_times(torch.tensor(times, dtype=torch.float))
    return [float(mob.location[i].reshape(-1, 3)[0, 0]) for i in range(len(times))]


# --------------------------------------------------------------------------
# SlideShow
# --------------------------------------------------------------------------
def test_slideshow_holds_each_entrance_for_a_second_before_the_next(scene):
    with SlideShow():
        first = Square().spawn()
        after_first = _cursor(scene)
        second = Circle().spawn()
        after_second = _cursor(scene)

    # One second to fade in, one second to read it, for each Mob in turn.
    assert after_first == pytest.approx(2.0)
    assert after_second == pytest.approx(4.0)
    assert _opacities(scene, [first, second], [0.5, 1.5, 2.5, 3.5]) == [
        [0.5, 0.0],
        [1.0, 0.0],
        [1.0, 0.5],
        [1.0, 1.0],
    ]


def test_slideshow_clears_everything_it_spawned_together(scene):
    with SlideShow():
        first = Square().spawn()
        second = Circle().spawn()

    # Both fade out over the same second, straight after the last hold.
    assert _opacities(scene, [first, second], [4.5, 5.5]) == [
        [0.5, 0.5],
        [0.0, 0.0],
    ]
    assert first.is_despawned()
    assert second.is_despawned()


def test_slideshow_pauses_after_clearing_the_slide(scene):
    """REGRESSION. The pause ran after the SlideShow's context had closed, so
    it recorded nothing, and whatever came next started the instant the slide
    was gone.
    """
    with SlideShow():
        Square().spawn()
    # Fade in, hold, fade out, and then the documented pause.
    assert _cursor(scene) == pytest.approx(4.0)
    following = Circle().spawn()
    assert _opacities(scene, [following], [3.5, 4.5]) == [[0.0], [0.5]]


def test_slideshow_leaves_mobs_from_outside_the_block_alone(scene):
    with Off():
        backdrop = Square().spawn()
    with SlideShow():
        Circle().spawn()

    assert not backdrop.is_despawned()
    assert _opacities(scene, [backdrop], [5.0]) == [[1.0]]


def test_a_despawn_inside_a_slideshow_is_held_like_an_entrance(scene):
    with SlideShow():
        shown = Square().spawn()
        shown.despawn()
        after_exit = _cursor(scene)

    # In for one second, read for one, out for one, then hold the empty slide.
    assert after_exit == pytest.approx(4.0)
    assert _opacities(scene, [shown], [2.5, 3.5]) == [[0.5], [0.0]]


def test_a_mob_built_in_a_slideshow_can_still_be_shown_after_it(scene):
    with SlideShow():
        Square().spawn()
        kept_for_later = Circle()
    end_of_show = _cursor(scene)
    kept_for_later.spawn()

    assert not kept_for_later.is_despawned()
    assert _opacities(scene, [kept_for_later], [end_of_show + 1.5]) == [[1.0]]


def test_a_slideshow_that_raises_records_no_clear_up(scene):
    """REGRESSION. The clear-up also ran on the way out of a block that raised,
    so a failed slide still recorded a second of fade-out the script never
    reached.
    """
    with Off():
        Square().spawn()
    end_before = scene.animation_manager.context.timespan.original_end
    shown = []

    def author_a_failing_slide():
        with SlideShow():
            shown.append(Circle().spawn())
            raise RuntimeError("authoring failed")

    with pytest.raises(RuntimeError, match="authoring failed"):
        author_a_failing_slide()

    # The failure propagates unchanged, and nothing is recorded on its way
    # out: the failed block adds no time and despawns nothing.
    assert scene.animation_manager.context.timespan.original_end == end_before
    assert not shown[0].is_despawned()


# --------------------------------------------------------------------------
# NoExtra
# --------------------------------------------------------------------------
def test_no_extra_drops_the_slideshow_hold(scene):
    with SlideShow():
        Square().spawn()
        start = _cursor(scene)
        with NoExtra():
            helper = Circle().spawn()
        after_helper = _cursor(scene)

    # The helper's entrance still follows on in sequence -- NoExtra keeps the
    # enclosing timing -- but no reading pause is added after it.
    assert after_helper - start == pytest.approx(1.0)
    assert _opacities(scene, [helper], [start + 0.5, start + 1.0]) == [[0.5], [1.0]]


def test_no_extra_drops_the_hold_after_a_despawn_too(scene):
    with SlideShow():
        shown = Square().spawn()
        start = _cursor(scene)
        with NoExtra():
            shown.despawn()
        assert _cursor(scene) - start == pytest.approx(1.0)


def test_a_mob_spawned_under_no_extra_is_still_cleared_with_the_slide(scene):
    with SlideShow():
        Square().spawn()
        with NoExtra():
            helper = Circle().spawn()

    assert helper.is_despawned()
    assert _opacities(scene, [helper], [3.5, 4.5]) == [[0.5], [0.0]]


# --------------------------------------------------------------------------
# OnInit
# --------------------------------------------------------------------------
def test_on_init_runs_on_every_mob_built_inside_the_block(scene):
    seen = []
    before = Square()
    with OnInit(seen.append):
        square = Square()
        circle = Circle()
    after = Square()

    assert square in seen
    assert circle in seen
    assert before not in seen
    assert after not in seen
    # The hook also reaches the parts a Mob builds for itself, so it has to
    # be written for any Mob, not only the top-level ones.
    built = [*square.get_descendants(), *circle.get_descendants()]
    assert all(any(mob is part for part in built) for mob in seen)


def test_on_init_applies_shared_setup_to_what_it_builds(scene):
    material = MeshStandardMaterial(metalness=1.0, roughness=0.25)
    with OnInit(lambda mob: mob.set_material(material)):
        shapes = [Square(), Circle()]
    plain = Square()

    for shape in shapes:
        assert shape.material is material
        shape.spawn()
    assert getattr(plain, "material", None) is not material


def test_nested_on_init_hooks_run_innermost_first(scene):
    calls = []
    with OnInit(lambda mob: calls.append(("outer", mob))):
        with OnInit(lambda mob: calls.append(("inner", mob))):
            square = Square()
        circle = Circle()

    square_calls = [label for label, mob in calls if mob is square]
    circle_calls = [label for label, mob in calls if mob is circle]
    assert square_calls == ["inner", "outer"]
    assert circle_calls == ["outer"]


# --------------------------------------------------------------------------
# ComposedEasing
# --------------------------------------------------------------------------
PROGRESS = [0.2, 0.4, 0.6, 0.8]


def _move_right_once(scene, context_factory):
    with Off():
        square = Square().spawn()
    start = _cursor(scene)
    with context_factory():
        square.move(RIGHT)
    return square, start, _cursor(scene)


def test_composed_easing_bends_progress_before_the_base_easing(scene):
    square, start, end = _move_right_once(
        scene, lambda: ComposedEasing(easings.ease_in_quad)
    )

    assert end - start == pytest.approx(1.0)
    expected = [float(easings.smooth(torch.tensor(t**2))) for t in PROGRESS]
    actual = _x_positions(scene, square, [start + t for t in PROGRESS])
    assert actual == pytest.approx(expected, abs=1e-4)


def test_composed_easing_keeps_a_base_easing_set_around_it(scene):
    with Seq(easing=easings.linear):
        square, start, _ = _move_right_once(
            scene, lambda: ComposedEasing(easings.ease_in_quad)
        )

    actual = _x_positions(scene, square, [start + t for t in PROGRESS])
    assert actual == pytest.approx([t**2 for t in PROGRESS], abs=1e-4)


def test_composed_easing_keeps_a_base_easing_set_inside_it(scene):
    def factory():
        return ComposedEasing(easings.ease_in_quad)

    with Off():
        square = Square().spawn()
    start = _cursor(scene)
    with factory(), Seq(easing=easings.linear):
        square.move(RIGHT)

    actual = _x_positions(scene, square, [start + t for t in PROGRESS])
    assert actual == pytest.approx([t**2 for t in PROGRESS], abs=1e-4)


def test_composing_the_identity_changes_nothing(scene):
    square, start, _ = _move_right_once(scene, lambda: ComposedEasing(easings.identity))
    expected = [float(easings.smooth(torch.tensor(t))) for t in PROGRESS]
    actual = _x_positions(scene, square, [start + t for t in PROGRESS])
    assert actual == pytest.approx(expected, abs=1e-4)


def test_nested_composed_easings_compose_rather_than_replace(scene):
    """REGRESSION. The inner ``ComposedEasing`` replaced the outer one, so two
    nested ``ease_in_quad`` gave ``t**2`` rather than ``t**4``.
    """
    with Off():
        square = Square().spawn()
    start = _cursor(scene)
    with (
        Seq(easing=easings.linear),
        ComposedEasing(easings.ease_in_quad),
        ComposedEasing(easings.ease_in_quad),
    ):
        square.move(RIGHT)

    # Squaring progress twice is the fourth power.
    actual = _x_positions(scene, square, [start + t for t in PROGRESS])
    assert actual == pytest.approx([t**4 for t in PROGRESS], abs=1e-4)


# --------------------------------------------------------------------------
# combine_easing: one rate function across the whole block
# --------------------------------------------------------------------------
def _two_moves(scene, **seq_kwargs):
    with Off():
        square = Square().spawn()
    start = _cursor(scene)
    with Seq(**seq_kwargs):
        square.move(RIGHT)
        square.move(RIGHT)
    return square, start


def test_each_animation_is_eased_on_its_own_by_default(scene):
    square, start = _two_moves(scene)
    times = [start + 0.5, start + 1.5]
    assert _x_positions(scene, square, times) == pytest.approx([0.5, 1.5], abs=1e-4)


def test_combine_easing_eases_a_sequence_as_one_motion(scene):
    """REGRESSION. ``combine_easing=True`` raised ``AttributeError: 'function'
    object has no attribute 'set_full_time'`` for every plain easing,
    ``smooth`` included.
    """
    square, start = _two_moves(scene, combine_easing=True)

    # Two one-unit moves read as a single smooth two-unit move over both
    # seconds: x(t) = 2 * smooth(t / 2), with no stop in the middle.
    times = [0.25, 0.5, 1.0, 1.5, 1.75]
    expected = [2 * float(easings.smooth(torch.tensor(t / 2))) for t in times]
    actual = _x_positions(scene, square, [start + t for t in times])
    assert actual == pytest.approx(expected, abs=1e-4)


def test_combine_easing_follows_the_block_through_its_rescaling(scene):
    """REGRESSION. Raised the same ``AttributeError``."""
    square, start = _two_moves(scene, runtime=4, combine_easing=True)

    times = [1.0, 2.0, 3.0, 4.0]
    expected = [2 * float(easings.smooth(torch.tensor(t / 4))) for t in times]
    actual = _x_positions(scene, square, [start + t for t in times])
    assert actual == pytest.approx(expected, abs=1e-4)


def test_combine_easing_gives_each_animation_its_slice_of_the_curve(scene):
    """REGRESSION. Raised the same ``AttributeError``."""
    square, start = _two_moves(scene, easing=easings.ease_in_quad, combine_easing=True)

    # The second move spans the second half of the block, so it takes the
    # second half of the curve, rescaled to land exactly on its own target.
    def second_half(u):
        return 1 + (u**2 - 0.25) / (1 - 0.25)

    times = [1.0, 1.5, 2.0]
    expected = [second_half(t / 2) for t in times]
    actual = _x_positions(scene, square, [start + t for t in times])
    assert actual == pytest.approx(expected, abs=1e-4)


def test_the_composed_easing_documentation_example_runs(scene):
    """``ComposedEasing``'s docstring example, so the name it uses exists."""
    with Off():
        square = Square().spawn()
    start = _cursor(scene)
    with ComposedEasing(easings.ease_out_bounce):
        square.move(RIGHT)
    assert _cursor(scene) - start == pytest.approx(1.0)
    assert _x_positions(scene, square, [start + 1.0]) == pytest.approx([1.0])
