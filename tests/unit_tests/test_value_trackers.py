"""``ValueTracker`` and ``ComplexValueTracker``, Manim's animatable numbers.

They draw nothing; what matters is the number they hold over time. That is read
two ways: while authoring, as a plain Python number, and inside an updater,
where a render computes a whole batch of frames at once and the read has to
give every one of them its own value.

A feature test, so it sits outside the fast suite: it only breaks when the
trackers themselves change.
"""

from __future__ import annotations

import pytest
import torch

import algan
from algan import RIGHT, UP, Dot3D, Off, Scene
from algan.manim import ComplexValueTracker, ValueTracker

#: Frames across a one-second animation starting at time zero.
TIMES = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])


@pytest.fixture
def scene():
    with Scene() as active:
        yield active


def _spawned(mob):
    with Off():
        return mob.spawn()


def test_a_tracker_reads_back_the_number_it_was_given(scene):
    tracker = ValueTracker(2.5)
    assert tracker.get_value() == 2.5
    assert isinstance(tracker.get_value(), float)
    assert float(tracker) == 2.5
    assert tracker.set_value(-1) is tracker
    assert tracker.get_value() == -1.0


def test_increments_and_in_place_operators_keep_the_tracker(scene):
    tracker = ValueTracker(1)
    tracker.increment_value(2)
    assert tracker.get_value() == 3.0

    same = tracker
    tracker += 1.5
    assert tracker is same
    assert tracker.get_value() == 4.5
    tracker -= 0.5
    assert tracker is same
    assert tracker.get_value() == 4.0


def test_a_spawned_tracker_animates_toward_a_new_value(scene):
    tracker = _spawned(ValueTracker(0))
    tracker.set_value(4)

    assert tracker.get_value() == 4.0
    scene.timeline_manager.set_state_to_times(TIMES)
    values = tracker.value.reshape(-1)
    assert values[0] == 0
    assert values[-1] == pytest.approx(4)
    assert torch.all(values[1:] >= values[:-1])
    assert values[2] == pytest.approx(2, abs=1e-3)


def test_an_updater_reads_the_tracker_on_every_frame_of_a_batch(scene):
    """REGRESSION. ``get_value`` returned the batch's first frame as a float,
    so the updater the "rebuild inside an updater" error recommends -- a
    display following a tracker -- showed that one number for the whole batch.
    """
    tracker = _spawned(ValueTracker(0))
    display = _spawned(algan.DecimalNumber(0.0))
    display.add_updater(lambda mob, t: mob.set_value(tracker.get_value()))
    dot = _spawned(Dot3D(radius=0.1))
    dot.add_updater(lambda mob, t: mob.move_to(RIGHT * tracker.get_value()))
    tracker.set_value(5)

    scene.timeline_manager.set_state_to_times(TIMES)
    expected = tracker.value.reshape(-1)
    assert len(set(expected.tolist())) == len(TIMES)
    shown = display.value.reshape(len(TIMES), -1)[:, 0]
    assert torch.allclose(shown, expected)
    assert torch.allclose(dot.location.reshape(len(TIMES), 3)[:, 0], expected)


def test_a_complex_tracker_reads_back_and_increments(scene):
    tracker = ComplexValueTracker(1 + 2j)
    assert tracker.get_value() == 1 + 2j
    assert isinstance(tracker.get_value(), complex)
    tracker.increment_value(1j)
    assert complex(tracker) == 1 + 3j
    assert tracker.set_value(-2) is tracker
    assert tracker.get_value() == -2 + 0j


def test_an_updater_reads_a_complex_tracker_on_every_frame(scene):
    """REGRESSION. As for ``ValueTracker``: ``get_value`` returned the batch's
    first frame, so the dot sat at that one value for the whole batch.
    """
    tracker = _spawned(ComplexValueTracker(0))
    dot = _spawned(Dot3D(radius=0.1))

    def follow(mob, t):
        value = tracker.get_value()
        mob.move_to(RIGHT * value.real + UP * value.imag)

    dot.add_updater(follow)
    tracker.set_value(4 + 2j)

    scene.timeline_manager.set_state_to_times(TIMES)
    expected = tracker.complex_value.reshape(len(TIMES), 2)
    location = dot.location.reshape(len(TIMES), 3)
    assert torch.allclose(location[:, :2], expected)
    assert location[-1, :2].tolist() == pytest.approx([4, 2])
