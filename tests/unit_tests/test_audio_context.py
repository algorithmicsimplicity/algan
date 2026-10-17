"""An Audio block lasts as long as its sound, whatever it animates.

The block's runtime is taken from its clip, and stretching what is inside to
it is how an animation is fitted to a sound. A block that animated nothing had
nothing to stretch and took no time, so the sound played over what came next:
two lines of narration with nothing animated under the first started together.
"""

import numpy as np
import pytest
from moviepy import AudioClip

from algan import RED, Audio, Off, Scene, Speech, Square


def _clip(duration):
    return AudioClip(lambda t: np.zeros(np.shape(t) + (1,)), duration=duration)


def _start(context):
    return context._audio_effect.start_time_func()


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(lambda square: None, id="empty"),
        pytest.param(lambda square: square.set_non_recursive(color=RED), id="edit"),
    ],
)
def test_a_block_that_animates_nothing_still_lasts_its_sound(body):
    with Scene() as scene:
        square = Square().spawn(animate=False)
        start = scene.animation_manager.context.timespan.original_end
        first = Audio(_clip(1.5))
        with first, Off():
            body(square)
        second = Audio(_clip(0.5))
        with second:
            square.rotate(90)

        assert _start(second) - _start(first) == pytest.approx(1.5)
        end = scene.animation_manager.context.timespan.original_end
        assert end - start == pytest.approx(2.0)


def test_consecutive_lines_of_narration_do_not_overlap():
    with Scene() as scene:
        scene.audio_manager.set_speech_source(lambda text: _clip(len(text) / 10))
        Square().spawn(animate=False)
        lines = [Speech(text, wait_at_end=0) for text in ("Hello.", "Goodbye now.")]
        for line in lines:
            with line:
                pass

        assert _start(lines[1]) - _start(lines[0]) == pytest.approx(0.6)


def test_a_sound_inside_off_still_takes_no_time():
    with Scene() as scene:
        Square().spawn(animate=False)
        start = scene.animation_manager.context.timespan.original_end
        with Off(), Audio(_clip(1.5)):
            pass

        assert scene.animation_manager.context.timespan.original_end == start
