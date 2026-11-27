"""Tests that a block or animated call that raises leaves nothing recorded.

Run directly: .venv/Scripts/python.exe -m pytest tests/unit_tests/test_failed_block_rollback.py -q

In a notebook or a REPL an exception is caught and the script goes on, so a
``with`` block (or an animated function) whose body raised must not leave the
part it recorded before the error on the timeline. These are logic-level
checks: scenes are recorded and the timeline is materialized at chosen times,
as the render loop does it, with no rendering.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from algan import Mob, animated_function
from algan.animation_timeline.animation_contexts import Audio, Off, Seq, Sync
from algan.constants import easings
from algan.constants.spatial import RIGHT, UP
from algan.errors import AlganWarning
from algan.scene_manager import SceneManager


@pytest.fixture(autouse=True)
def fresh_scene():
    SceneManager.reset()


def _x(times, mob):
    timeline = mob.scene.timeline_manager
    with Off(
        record_attr_modifications=False, record_funcs=False, priority_level=math.inf
    ):
        timeline.set_state_to_times(torch.tensor(times, dtype=torch.float32))
        out = mob.get_animated_attribute("location")[:, 0, 0].tolist()
    timeline.clear_buffers()
    return out


def _in_block(block, *steps):
    """Run ``steps`` inside ``block``, then raise ``KeyError`` from it."""
    with block:
        for step in steps:
            step()
        raise KeyError("in the block")


def _clean_run():
    """The script with the failed part left out: one move right."""
    mob = Mob().spawn(animate=False)
    mob.move(RIGHT)
    return _x([0.0, 0.5, 1.0, 2.0], mob), mob.location.clone()


def test_a_with_block_that_raised_records_nothing():
    """The failed ``Sync``'s move used to stay recorded over [0, 1] and add to
    the next move: x = 0, 1, 2 at t = 0, .5, 1 instead of 0, .5, 1.
    """
    expected, location = _clean_run()
    SceneManager.reset()
    mob = Mob().spawn(animate=False)
    with pytest.raises(KeyError):
        _in_block(Sync(), lambda: mob.move(RIGHT), lambda: mob.move(UP))
    mob.move(RIGHT)
    assert _x([0.0, 0.5, 1.0, 2.0], mob) == pytest.approx(expected, abs=1e-6)
    assert mob.location == pytest.approx(location, abs=1e-6)


@animated_function
def _rise_then_fail(mob, height=1.0):
    mob.location = mob.location + UP * height
    raise ValueError("in the animated function")


def test_an_animated_call_that_raised_records_nothing():
    """The call used to stay on the timeline, half-applied, and raise again at
    every later render of a frame it covers.
    """
    expected, location = _clean_run()
    SceneManager.reset()
    mob = Mob().spawn(animate=False)
    with pytest.raises(ValueError):
        _rise_then_fail(mob)
    mob.move(RIGHT)
    assert _x([0.0, 0.5, 1.0, 2.0], mob) == pytest.approx(expected, abs=1e-6)
    assert mob.location == pytest.approx(location, abs=1e-6)
    timeline = mob.scene.timeline_manager
    assert len(timeline.function_timeline.function_applications) == 1


def test_an_error_caught_inside_a_block_keeps_the_rest_of_the_block():
    """Only the failed call goes: the block around it records as usual."""
    mob = Mob().spawn(animate=False)
    with Seq(easing=easings.identity):
        mob.move(RIGHT)
        with pytest.raises(ValueError):
            _rise_then_fail(mob)
        mob.move(RIGHT)
    assert _x([0.0, 1.0, 1.5, 2.0], mob) == pytest.approx(
        [0.0, 1.0, 1.5, 2.0], abs=1e-6
    )
    assert mob.location[..., 1].item() == pytest.approx(0.0, abs=1e-6)


def test_a_failed_block_takes_back_its_updaters_and_their_removals():
    """An updater added in the block is gone, and one removed in it runs on."""
    mob = Mob().spawn(animate=False)
    kept = mob.add_updater(lambda m, t: None)
    with pytest.raises(KeyError):
        _in_block(
            Sync(),
            lambda: mob.add_updater(lambda m, t: m.set(location=RIGHT * t)),
            lambda: mob.remove_updater(kept),
        )
    updaters = mob.scene.timeline_manager.function_timeline.updaters
    assert len(updaters) == 1
    assert updaters[kept].time.end_event is None


def test_a_failed_audio_block_takes_back_its_sound():
    scene = SceneManager.instance().current_scene
    mob = Mob().spawn(animate=False)
    with pytest.raises(KeyError):
        _in_block(Audio(SimpleNamespace(duration=2.0)), lambda: mob.move(RIGHT))
    assert scene.effects == []


def test_a_failed_block_that_spawned_keeps_its_recording_and_says_so():
    """A spawn stamps a lifespan in place, which the edit log cannot take back,
    so the block's recording stays and a warning says the timeline has it.
    """
    mob = Mob().spawn(animate=False)
    with (
        pytest.warns(AlganWarning, match="stays on the timeline"),
        pytest.raises(KeyError),
    ):
        _in_block(Sync(), lambda: mob.move(RIGHT), lambda: Mob().spawn())
    assert mob.location[..., 0].item() == pytest.approx(1.0, abs=1e-6)


def _fails(_mob, _t):
    raise RuntimeError("in the updater")


def test_an_updater_whose_first_application_raised_is_not_kept():
    """It used to stay registered and raise again at every render."""
    mob = Mob().spawn(animate=False)
    with pytest.raises(RuntimeError):
        mob.add_updater(_fails)
    assert mob.scene.timeline_manager.function_timeline.updaters == []
