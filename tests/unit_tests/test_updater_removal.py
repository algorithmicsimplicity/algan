"""Tests that removing an updater leaves the rest of the recording as it was.

Run directly: .venv/Scripts/python.exe -m pytest tests/unit_tests/test_updater_removal.py -q

``remove_updater`` materializes the timeline at the removal time while the
script is still authoring, so that the Mob keeps the state the updater left it
in. That materialization must not leave anything behind that changes how the
other recorded animations replay. These are logic-level checks: scenes are
recorded and the timeline is materialized at chosen times, as the render loop
does it, with no rendering.
"""

import math
import warnings
from types import SimpleNamespace

import pytest
import torch

from algan import Mob
from algan.animation_timeline.animation_contexts import Audio, Off, Seq, Speech, Sync
from algan.constants import easings
from algan.constants.spatial import OUT, RIGHT, UP
from algan.errors import AlganWarning
from algan.scene_manager import SceneManager


@pytest.fixture(autouse=True)
def fresh_scene():
    SceneManager.reset()


def _materialize(times, mob, attr="location"):
    """The values of ``mob``'s ``attr`` at ``times`` ([T, rows, channels])."""
    timeline = mob.scene.timeline_manager
    with Off(
        record_attr_modifications=False, record_funcs=False, priority_level=math.inf
    ):
        timeline.set_state_to_times(torch.tensor(times, dtype=torch.float32))
        out = mob.get_animated_attribute(attr).clone()
    timeline.clear_buffers()
    return out


def _x(times, mob):
    return _materialize(times, mob)[:, 0, 0].tolist()


def _stretched_block(block, with_updater):
    """Two one-second moves in ``block``, which stretches them to two seconds
    each, with an updater on an unrelated Mob added and removed between them.
    """
    mob = Mob().spawn(animate=False)
    other = Mob().spawn(animate=False)
    with block():
        mob.move(RIGHT)
        if with_updater:
            other.remove_updater(other.add_updater(lambda _mob, _t: None))
        mob.move(RIGHT)
    SceneManager.instance().current_scene.wait(1)
    return mob


def _speech(duration):
    """A ``Speech`` line whose generated clip lasts ``duration`` seconds."""
    scene = SceneManager.instance().current_scene
    scene.audio_manager.set_speech_source(
        lambda _text: SimpleNamespace(duration=duration)
    )
    return Speech("A line of narration.", wait_at_end=0, easing=easings.identity)


@pytest.mark.parametrize(
    "block",
    [
        pytest.param(lambda: Seq(runtime=4, easing=easings.identity), id="Seq"),
        # Both moves run over the whole block, which lands on the same x as
        # the two in sequence: x = t / 2.
        pytest.param(lambda: Sync(runtime=4, easing=easings.identity), id="Sync"),
        # Stands in for a four-second clip: Audio only reads its duration
        # while authoring, and Speech times its block the same way.
        pytest.param(
            lambda: Audio(SimpleNamespace(duration=4.0), easing=easings.identity),
            id="Audio",
        ),
        pytest.param(lambda: _speech(4.0), id="Speech"),
    ],
)
def test_removing_an_updater_in_a_block_rescaled_later_keeps_earlier_animations(
    block,
):
    """``remove_updater`` inside a block whose length is set when it exits (a
    ``runtime``, an ``Audio`` or ``Speech`` clip) used to freeze the replay
    windows of the block's earlier animations at their unscaled times. The
    first move was then applied twice over its whole span: the Mob raced
    ahead to x = 1.5 at t = 1 and snapped back at t = 2.
    """
    times = [0.0, 1.0, 1.5, 2.0, 3.0, 4.0]
    no_updater = _x(times, _stretched_block(block, with_updater=False))
    SceneManager.reset()
    actual = _x(times, _stretched_block(block, with_updater=True))
    assert actual == pytest.approx([0.0, 0.5, 0.75, 1.0, 1.5, 2.0], abs=1e-5)
    assert actual == pytest.approx(no_updater, abs=1e-6)


def test_removing_an_updater_in_a_compressed_block_keeps_its_instant_writes():
    """The compressed case: in ``Seq(runtime=1)`` the instant opacity write
    between the first two moves never showed after an updater was removed
    between them.
    """
    mob = Mob().spawn(animate=False)
    other = Mob().spawn(animate=False)
    with Seq(runtime=1, easing=easings.identity):
        mob.move(RIGHT)
        with Off():
            mob.opacity = 0.3
        other.remove_updater(other.add_updater(lambda _mob, _t: None))
        mob.move(RIGHT)
        with Off():
            mob.opacity = 0.6
        mob.move(RIGHT)
    SceneManager.instance().current_scene.wait(1)
    opacity = _materialize([0.0, 0.25, 0.4, 0.5, 0.75, 1.5], mob, "opacity")
    assert opacity[:, 0, 0].tolist() == pytest.approx(
        [1.0, 1.0, 0.3, 0.3, 0.6, 0.6], abs=1e-5
    )


def _follow_then_animate(remove_first, animate):
    """A dot follows a leader through an updater, then in one ``Sync`` the
    updater is removed and ``animate(dot)`` runs.
    """
    scene = SceneManager.instance().current_scene
    leader = Mob().spawn(animate=False)
    dot = Mob().spawn(animate=False)
    updater = dot.add_updater(lambda mob, _t: mob.move_to(leader.location + UP))
    leader.move(RIGHT)
    with Sync(easing=easings.identity):
        if remove_first:
            dot.remove_updater(updater)
            animate(dot)
        else:
            animate(dot)
            dot.remove_updater(updater)
    scene.wait(1)
    return dot


def _move_then_move_on(dot):
    with Seq():
        dot.move(UP)
        dot.move(RIGHT)


@pytest.mark.parametrize(
    "animate",
    [
        pytest.param(lambda dot: dot.move(UP), id="move"),
        # The second move is recorded after the first and starts after the
        # removal: it is rebuilt from the first one's new end.
        pytest.param(_move_then_move_on, id="Seq"),
        pytest.param(lambda dot: (dot.move(UP), dot.rotate(90, OUT)), id="rotate"),
        # A destination, not a displacement: it is reached from where the
        # updater left the dot, not missed by how far the updater had moved it.
        pytest.param(lambda dot: dot.move_to(RIGHT * 3 + UP * 3), id="move_to"),
        pytest.param(
            lambda dot: setattr(dot, "location", RIGHT * 3 + UP * 3), id="assignment"
        ),
        pytest.param(lambda dot: dot.set(location=RIGHT * 3 + UP * 3), id="set"),
    ],
)
def test_an_animation_written_before_remove_updater_in_a_sync_still_runs(animate):
    """In one ``Sync`` the removal and the animation start together, so their
    order must not decide whether the animation exists. Written after the
    move, the removal's write of the updater's state used to be recorded
    after it and hold its value over it: the dot never moved, and a warning
    said to swap the lines. Now the animation starts from where the updater
    left the dot either way, and nothing warns.
    """
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", AlganWarning)
        dot = _follow_then_animate(remove_first=False, animate=animate)
    after = _materialize(times, dot)
    after_basis = _materialize(times, dot, "basis")
    after_location = dot.location.clone()

    SceneManager.reset()
    dot = _follow_then_animate(remove_first=True, animate=animate)
    assert after == pytest.approx(_materialize(times, dot), abs=1e-5)
    assert after_basis == pytest.approx(_materialize(times, dot, "basis"), abs=1e-5)
    assert after_location == pytest.approx(dot.location, abs=1e-5)


def test_the_move_written_before_remove_updater_completes():
    """The finding's scenario, as numbers: the dot follows the leader to
    x = 1, then moves up from y = 1 to y = 2 over the ``Sync``.
    """
    dot = _follow_then_animate(remove_first=False, animate=lambda d: d.move(UP))
    xy = _materialize([1.0, 1.5, 2.0, 2.5], dot)[:, 0, :2]
    assert xy[:, 0].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0], abs=1e-5)
    assert xy[:, 1].tolist() == pytest.approx([1.0, 1.5, 2.0, 2.0], abs=1e-5)


@pytest.mark.parametrize(
    "write",
    [
        pytest.param(lambda d: d.move_to(RIGHT * 3 + UP * 3), id="move_to"),
        pytest.param(lambda d: setattr(d, "location", RIGHT * 3 + UP * 3), id="assign"),
    ],
)
def test_a_destination_written_before_remove_updater_is_reached(write):
    """``move_to`` records a change from the state the line saw. Under a live
    updater that state is stale: here the dot is authored at the origin while
    the updater holds it at (1, 1). Re-based onto the updater's state with the
    recorded change, the dot ended at (4, 4) instead of (3, 3).
    """
    dot = _follow_then_animate(remove_first=False, animate=write)
    xy = _materialize([1.0, 1.5, 2.0, 2.5], dot)[:, 0, :2]
    assert xy[:, 0].tolist() == pytest.approx([1.0, 2.0, 3.0, 3.0], abs=1e-5)
    assert xy[:, 1].tolist() == pytest.approx([1.0, 2.0, 3.0, 3.0], abs=1e-5)
    assert dot.location.view(-1)[:2].tolist() == pytest.approx([3.0, 3.0], abs=1e-5)


def test_removing_an_updater_in_the_middle_of_a_running_animation_warns():
    """A move that started before the removal and is still running cannot be
    put under the updater's state (it was running under the updater until
    then), so the removal still holds that state over it, and says so.
    """
    scene = SceneManager.instance().current_scene
    leader = Mob().spawn(animate=False)
    dot = Mob().spawn(animate=False)
    updater = dot.add_updater(lambda mob, _t: mob.move_to(leader.location + UP))
    leader.move(RIGHT)
    with Sync(easing=easings.identity):
        dot.move(UP)
        with Seq():
            scene.wait(0.5)
            with pytest.warns(AlganWarning, match="cancels its"):
                dot.remove_updater(updater)
