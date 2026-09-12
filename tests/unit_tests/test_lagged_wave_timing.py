"""``animate_lagged_by_location`` reserves the span its wave occupies.

Run directly: .venv/Scripts/python.exe -m pytest tests/unit_tests/test_lagged_wave_timing.py -q

A lagged wave starts every part at the block's cursor and staggers it by the
part's position along the direction, so the whole wave spans one runtime plus
one lag. It then moves the enclosing context's cursor past that span -- which
means it also has to extend that context's *end* past it, or the block ends
before its own cursor and whatever follows in a ``Seq`` starts on top of it.

The extension used to be written to ``original_end_time``. ``TimelineSpan`` has
no such field, so Python simply attached one to the span object: no error, no
reader, and none of the timing this file checks.
"""

import pytest
import torch

from algan import RIGHT, Off, Scene, Seq, Square
from algan.scene_manager import SceneManager
from algan.utils.animation_utils import animate_lagged_by_location


@pytest.fixture(autouse=True)
def fresh_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _wave_parts(count=3):
    """Spawned mobs spread along RIGHT, so the wave has something to lag by."""
    with Off():
        mobs = [Square().spawn() for _ in range(count)]
        for index, mob in enumerate(mobs):
            mob.move(RIGHT * index)
    return mobs


def test_a_wave_that_records_nothing_still_reserves_its_runtime():
    """The span is the wave's, not its contents'.

    Each part is recorded inside a context of the wave's full runtime, so a
    wave that records anything at all extends the block through that child.
    One that records nothing had nothing to extend it, and left the block
    ending at the instant the wave began.
    """
    with Scene() as scene:
        parts = _wave_parts()
        with Seq() as block:
            start = block.timespan.current_time
            animate_lagged_by_location(parts, lambda mob: None, RIGHT, lag_duration=1)

            runtime = block.runtime_per_part
            assert block.timespan.original_end == pytest.approx(start + runtime + 1)
        assert scene.animation_manager.context.timespan.original_end > start


def test_the_block_never_ends_before_its_own_cursor():
    """Where the next animation starts is where this one may not still be.

    In a ``Seq`` the cursor lands on the end of the wave, so an end left behind
    it is the block claiming to be over while its cursor sits in the future --
    and the video's duration is read off exactly that end.
    """
    with Scene():
        parts = _wave_parts()
        with Seq() as block:
            animate_lagged_by_location(parts, lambda mob: None, RIGHT, lag_duration=2)

            assert block.timespan.original_end >= block.timespan.current_time


def test_a_recording_wave_spans_one_runtime_plus_one_lag():
    """The ordinary case, unchanged: the parts are staggered inside the span."""
    with Scene():
        parts = _wave_parts()
        with Seq() as block:
            start = block.timespan.current_time
            animate_lagged_by_location(
                parts, lambda mob: mob.move(RIGHT), RIGHT, lag_duration=1
            )

            runtime = block.runtime_per_part
            assert block.timespan.original_end == pytest.approx(start + runtime + 1)
            assert block.timespan.current_time == pytest.approx(start + runtime + 1)


def test_a_wave_does_not_shorten_a_block_that_already_runs_longer():
    """Extending is a maximum, like every other write to a context's end."""
    with Scene():
        parts = _wave_parts()
        with Seq() as block:
            block.wait(10)
            end_before = block.timespan.original_end
            start = block.timespan.current_time
            animate_lagged_by_location(parts, lambda mob: None, RIGHT, lag_duration=1)

            assert block.timespan.original_end == pytest.approx(
                max(end_before, start + block.runtime_per_part + 1)
            )
            assert block.timespan.original_end >= end_before


def test_the_span_has_no_stray_end_time_attribute():
    """The name that was written for years, and the field it was meant to be."""
    with Scene():
        parts = _wave_parts()
        with Seq() as block:
            animate_lagged_by_location(parts, lambda mob: None, RIGHT)

            assert not hasattr(block.timespan, "original_end_time")
            assert isinstance(block.timespan.original_end, (int, float, torch.Tensor))
