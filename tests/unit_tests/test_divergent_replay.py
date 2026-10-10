"""A recorded call that reaches a different Mob when re-run says so.

Algan renders an animated function's frames by running it again. A function
that reads a mutable object the script changes after the call reaches, when
re-run, whatever the object holds by then: in the backpropagation video a
number line's marker kept its *current* label, so the frames of the call that
faded in "0.200" faded in "0.850" -- a label made 23 s later -- and "0.200"
stayed blank. Nothing said why.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
import torch

from algan import RED, Off, Scene, Sphere, Square, Sync, animated_function
from algan.errors import DivergentReplayWarning
from algan.scene_manager import SceneManager


@pytest.fixture(autouse=True)
def fresh_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


@animated_function(animated_args={"t": 0.0})
def _fade_in_holders_target(mob, holder, t=1.0):
    holder.target.opacity = t


def _materialize(scene, times):
    scene.timeline_manager.set_state_to_times(torch.tensor(times))
    scene.timeline_manager.clear_buffers()


def test_a_call_that_reaches_a_later_mob_warns_naming_it():
    with Scene() as scene:
        first = Square(name="first").spawn(animate=False)
        with Off():
            first.opacity = 0
        holder = SimpleNamespace(target=first)
        _fade_in_holders_target(first, holder)
        holder.target = Square(name="later").spawn(animate=False)
        scene.wait(1)

        with pytest.warns(DivergentReplayWarning, match="'later'"):
            _materialize(scene, [0.5])


def test_it_warns_once_per_function():
    with Scene() as scene:
        mob = Square().spawn(animate=False)
        holder = SimpleNamespace(target=mob)
        _fade_in_holders_target(mob, holder)
        _fade_in_holders_target(mob, holder)
        holder.target = Square().spawn(animate=False)
        scene.wait(1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _materialize(scene, [0.5, 1.5])
            _materialize(scene, [0.25])
        assert len([w for w in caught if w.category is DivergentReplayWarning]) == 1


def test_a_call_that_reaches_the_same_mobs_is_quiet():
    with Scene() as scene:
        mob = Square().spawn(animate=False)
        holder = SimpleNamespace(target=mob)
        _fade_in_holders_target(mob, holder)
        Square().spawn(animate=False)
        scene.wait(1)

        with warnings.catch_warnings():
            warnings.simplefilter("error", DivergentReplayWarning)
            _materialize(scene, [0.5])


def test_replays_algan_hands_to_a_clone_are_quiet():
    """``become`` and a re-sampling wave move recorded history onto Mobs made
    after the calls were recorded; their replays write there by design.
    """
    with Scene() as scene:
        square = Square().spawn()
        square.move([1.0, 0.0, 0.0])
        square.become(Square().scale(2))
        sphere = Sphere().spawn()
        with Sync():
            sphere.wave_color(RED)
        sphere.move([0.0, 1.0, 0.0])
        scene.wait(1)

        with warnings.catch_warnings():
            warnings.simplefilter("error", DivergentReplayWarning)
            _materialize(scene, [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 4.5])
