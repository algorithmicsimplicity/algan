"""When a Mob's changes happen: ``retroactive``, ``is_animating``, functions of
your own, and taking every updater off at once.

``retroactive()`` records a block back at an earlier time, ``is_animating()``
says whether a change would be recorded at all, ``animate_function`` and
``animate_function_of_time`` drive a function of your own by progress or by
seconds, and ``remove_all_updaters`` ends a Mob's updaters. All of it is read
back the way a render sees it, by materializing frames.

A feature test, so it sits outside the fast suite: it only breaks when these
helpers themselves change.
"""

from __future__ import annotations

import pytest
import torch

from algan import RIGHT, UP, Group, Off, Scene, Seq, Square, Sync, easings


@pytest.fixture
def scene():
    with Scene() as active:
        yield active


def _now(scene) -> float:
    return float(scene.animation_manager.context.timespan.current_time)


def _coordinate_at(scene, mob, times, axis=0):
    scene.timeline_manager.set_state_to_times(torch.tensor(times))
    return [round(float(value), 4) for value in mob.location[..., axis].reshape(-1)]


def _spawned_square():
    square = Square()
    with Off():
        square.spawn()
    return square


# --- retroactive ------------------------------------------------------------


def test_a_retroactive_block_is_recorded_back_at_the_start(scene):
    square = _spawned_square()
    square.wait(3)

    with square.retroactive(), Sync(easing=easings.identity):
        square.move(RIGHT)
    # Authoring carries on from where it was, not from the end of the block.
    assert _now(scene) == 3

    assert _coordinate_at(scene, square, [0.0, 0.5, 1.0, 2.0, 3.0]) == [
        0.0,
        0.5,
        1.0,
        1.0,
        1.0,
    ]


def test_each_retroactive_block_starts_where_the_last_one_was_entered(scene):
    square = _spawned_square()
    square.wait(3)
    with square.retroactive():
        square.move(RIGHT)
    square.wait(2)

    with square.retroactive(), Sync(easing=easings.identity):
        square.move(UP)
    assert _now(scene) == 5

    assert _coordinate_at(scene, square, [3.0, 3.5, 4.0, 5.0], axis=1) == [
        0.0,
        0.5,
        1.0,
        1.0,
    ]


def test_a_retroactive_block_restores_the_time_when_it_raises(scene):
    square = _spawned_square()
    square.wait(2)
    rewound_to = []

    def fail_inside_the_block():
        with square.retroactive():
            rewound_to.append(_now(scene))
            raise RuntimeError("the block's own mistake")

    with pytest.raises(RuntimeError, match="the block's own"):
        fail_inside_the_block()
    assert rewound_to == [0]
    assert _now(scene) == 2


# --- is_animating -----------------------------------------------------------


def test_a_mob_animates_once_spawned_and_outside_off(scene):
    square = Square()
    assert not square.is_animating()

    with Off():
        square.spawn()
        assert not square.is_animating()
    assert square.is_animating()

    with Off():
        assert not square.is_animating()
    with Seq():
        assert square.is_animating()


def test_a_parent_animates_through_a_spawned_child(scene):
    child = Square()
    parent = Group(child)
    assert not parent.is_animating()

    with Off():
        child.spawn()
    assert parent.is_animating()


# --- functions of your own --------------------------------------------------


def test_animate_function_sweeps_its_progress_and_holds_the_end(scene):
    square = _spawned_square()
    seen = []

    def slide(mob, t):
        seen.append(t)
        mob.set(location=RIGHT * t)

    with Sync(easing=easings.identity):
        assert square.animate_function(slide, t=4) is square
    # Run once as it is recorded, at its end value.
    assert [float(t) for t in seen] == [4.0]
    square.wait(1)

    assert _coordinate_at(scene, square, [0.0, 0.5, 1.0, 1.5]) == [0.0, 2.0, 4.0, 4.0]


def test_animate_function_progress_follows_the_contexts_easing(scene):
    square = _spawned_square()
    with Sync(easing=easings.smooth):
        square.animate_function(lambda mob, t: mob.set(location=RIGHT * t))

    eased = _coordinate_at(scene, square, [0.25, 0.5, 0.75])
    expected = easings.smooth(torch.tensor([0.25, 0.5, 0.75]))
    assert eased == pytest.approx(expected.tolist(), abs=1e-3)


def test_a_function_of_time_is_handed_seconds_unscaled_by_the_easing(scene):
    square = _spawned_square()
    with Seq(runtime=2, easing=easings.smooth):
        assert (
            square.animate_function_of_time(lambda mob, t: mob.set(location=RIGHT * t))
            is square
        )

    # Elapsed seconds, run straight through the two seconds the context gave
    # it: an easing shapes progress, and this is a clock.
    assert _coordinate_at(scene, square, [0.0, 0.5, 1.0, 1.5, 1.9]) == pytest.approx(
        [0.0, 0.5, 1.0, 1.5, 1.9], abs=1e-3
    )


def _slide_along(mob, t, speed, axis):
    mob.set(location=axis * speed * t)


def test_a_function_of_time_passes_its_extra_arguments_on(scene):
    """REGRESSION. An argument after the placeholder was recorded under the
    name ``args``, so the function was called with ``args=3`` as a keyword
    and raised ``TypeError: ... unexpected keyword argument 'args'``.
    """
    square = _spawned_square()
    square.animate_function_of_time(_slide_along, 0, 3, axis=UP)
    assert _coordinate_at(scene, square, [0.5, 0.9], axis=1) == pytest.approx(
        [1.5, 2.7], abs=1e-3
    )


def test_animate_function_passes_its_extra_arguments_on(scene):
    """REGRESSION. As for a function of time: extra positional arguments
    raised as an unexpected keyword ``args``.
    """
    square = _spawned_square()
    with Sync(easing=easings.identity):
        square.animate_function(_slide_along, 1, 3, UP)
    assert _coordinate_at(scene, square, [0.5, 1.0], axis=1) == pytest.approx(
        [1.5, 3.0], abs=1e-3
    )


# --- remove_all_updaters ----------------------------------------------------


def test_remove_all_updaters_stops_this_mobs_updaters_and_no_others(scene):
    mover, other = _spawned_square(), _spawned_square()
    mover.add_updater(lambda mob, t: mob.set(location=RIGHT * t))
    mover.add_updater(lambda mob, t: mob.set(opacity=1 / (1 + t)))
    other.add_updater(lambda mob, t: mob.set(location=UP * t))
    mover.wait(1)

    mover.remove_all_updaters()
    mover.wait(1)

    # Kept where its updaters left it, rather than snapping back.
    assert _coordinate_at(scene, mover, [0.5, 1.0, 1.5, 2.0]) == [0.5, 1.0, 1.0, 1.0]
    scene.timeline_manager.set_state_to_times(torch.tensor([0.5, 2.0]))
    opacity = mover.opacity.reshape(2, -1)[:, 0]
    assert opacity.tolist() == pytest.approx([1 / 1.5, 1 / 2], abs=1e-3)
    assert _coordinate_at(scene, other, [0.5, 2.0], axis=1) == [0.5, 2.0]


def test_remove_all_updaters_twice_or_on_a_mob_without_any_is_harmless(scene):
    square, bare = _spawned_square(), _spawned_square()
    square.add_updater(lambda mob, t: mob.set(location=RIGHT * t))
    square.wait(1)

    square.remove_all_updaters()
    square.remove_all_updaters()
    bare.remove_all_updaters()
    square.wait(1)

    assert _coordinate_at(scene, square, [1.0, 2.0]) == [1.0, 1.0]
    assert _coordinate_at(scene, bare, [2.0]) == [0.0]
