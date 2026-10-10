"""Which glyphs a text's entrance shows, and when.

A Tex/Text enters with a one-second wave that fades its glyphs in. It used to
fade every glyph to the text's own opacity, so a glyph hidden before the text
spawned faded in with the rest; and a glyph hidden with ``Off()`` while the
wave ran faded in too, and only vanished once the wave was over.
"""

from __future__ import annotations

import pytest
import torch

from algan import Off, Scene, Seq, Sync, Tex
from algan.scene_manager import SceneManager

TIMES = [0.05, 0.3, 0.6, 0.9, 1.2, 1.6, 2.5]
HIDDEN = (2, 3)


@pytest.fixture(autouse=True)
def fresh_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _glyph_opacity(author):
    """Each glyph's opacity at ``TIMES``, as ``[len(TIMES), glyphs]``."""
    with Scene() as scene:
        text = Tex(r"\text{abcdef}")
        author(scene, text)
        scene.wait(2)
        scene.timeline_manager.set_state_to_times(torch.tensor(TIMES))
        opacity = text._character_batch.opacity.detach().clone()
        scene.timeline_manager.clear_buffers()
    return opacity.reshape(len(TIMES), -1)


def _hide(text):
    for index in HIDDEN:
        text.character_mobs[index].opacity = 0


def _shown(opacity):
    return [i for i in range(opacity.shape[1]) if i not in HIDDEN]


def test_a_glyph_hidden_before_the_spawn_stays_hidden():
    def author(scene, text):
        _hide(text)
        text.spawn()

    opacity = _glyph_opacity(author)

    assert torch.all(opacity[:, list(HIDDEN)] == 0)
    assert torch.allclose(opacity[-1, _shown(opacity)], torch.tensor(1.0))


@pytest.mark.parametrize("at", [0.0, 0.3])
def test_a_glyph_hidden_during_the_entrance_hides_from_then(at):
    def author(scene, text):
        with Sync():
            text.spawn()
            with Seq():
                if at:
                    scene.wait(at)
                with Off():
                    _hide(text)

    opacity = _glyph_opacity(author)

    after = [k for k, t in enumerate(TIMES) if t >= at]
    assert torch.all(opacity[after][:, list(HIDDEN)] == 0)
    assert torch.allclose(opacity[-1, _shown(opacity)], torch.tensor(1.0))


def test_the_text_opacity_is_still_what_the_glyphs_fade_to():
    def author(scene, text):
        text.opacity = 0.5
        text.spawn()

    opacity = _glyph_opacity(author)

    assert torch.allclose(opacity[-1], torch.tensor(0.5))
    assert torch.all(opacity <= 0.5 + 1e-6)


def test_spawning_the_whole_text_after_one_glyph_leaves_that_glyph_lit():
    """The remaining glyphs used to be revealed by an animation of their own
    and then faded in again by the text's wave, which zeroed every glyph --
    including the one already on screen.
    """

    def author(scene, text):
        text.character_mobs[0].spawn()
        text.spawn()

    opacity = _glyph_opacity(author)
    late = [k for k, t in enumerate(TIMES) if t >= 1.0]

    assert torch.all(opacity[late, 0] > 0.99)
    # The others enter with one wave from zero, after the first glyph's.
    assert torch.all(opacity[TIMES.index(0.9), 1:] == 0)
    assert torch.all(opacity[TIMES.index(1.2), 1:] < 0.2)
    assert torch.allclose(opacity[-1], torch.tensor(1.0))
