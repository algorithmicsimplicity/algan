"""Native renders for DecimalNumber's visibility and precision transitions."""

from __future__ import annotations

import contextlib

import pytest
import torch
from PIL import Image

from algan import BLACK, PREVIEW, RIGHT, DecimalNumber, Scene, Square, Sync, easings


def _frames(scene, indices):
    with contextlib.closing(
        scene.get_frames(0, len(indices), frame_indices=indices, post_processes=())
    ) as frames:
        return torch.cat([frame.cpu().clone() for frame in frames])


@pytest.mark.parametrize("precision", [None, 3])
def test_replayed_number_is_blank_until_spawn_and_after_despawn(tmp_path, precision):
    settings = PREVIEW.set(resolution=(240, 160), frames_per_second=4)
    with Scene(settings, background=BLACK) as scene:
        number = DecimalNumber(0, integer_places=3, significant_figures=precision)
        number.scale(2)
        clock = Square(opacity=0).spawn(animate=False)
        clock.add_updater(lambda _mob, t: number.set_value(-100 * t))
        scene.wait(1)
        number.spawn(animate=False)
        scene.wait(1)
        number.despawn(animate=False)
        scene.wait(1)
        frames = _frames(scene, (2, 6, 10))
        assert not bool(frames[0].any())
        assert not bool(frames[2].any())
        assert int(frames[1].max()) > 50
        # A backwards seek must not retain the visible frame's glyph choices.
        assert not bool(_frames(scene, (2,))[0].any())

    with Scene(settings, background=BLACK) as control:
        DecimalNumber(-150, integer_places=3, significant_figures=precision).scale(
            2
        ).spawn(animate=False)
        control.wait(2)
        expected = _frames(control, (6,))[0]

    assert int((frames[1].int() - expected.int()).abs().max()) <= 2
    Image.fromarray(frames[1].numpy()).save(tmp_path / f"visible_{precision}.png")


@pytest.mark.parametrize(
    ("initial", "target", "midpoint"),
    [(0, 2000, 1000), (-0.0002, 0, -0.0001)],
)
def test_significant_animation_matches_static_control(
    tmp_path, initial, target, midpoint
):
    settings = PREVIEW.set(resolution=(320, 180), frames_per_second=4)

    def number(value):
        return (
            DecimalNumber(value, significant_figures=3)
            .scale(2)
            .rotate(12)
            .move(RIGHT * 0.3)
            .spawn(animate=False)
        )

    with Scene(settings, background=BLACK) as scene:
        display = number(initial)
        with Sync(runtime=1, easing=easings.identity):
            display.value = target
        scene.wait(1)
        actual = _frames(scene, (2,))[0]

    with Scene(settings, background=BLACK) as control:
        number(midpoint)
        control.wait(1)
        expected = _frames(control, (2,))[0]

    assert int(actual.max()) > 50
    assert int((actual.int() - expected.int()).abs().max()) <= 2
    Image.fromarray(actual.numpy()).save(tmp_path / f"significant_{target}.png")
