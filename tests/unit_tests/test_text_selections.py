"""Selections of a text's glyphs, and of a pack's members, as Mobs of their own.

``text[i:j]`` and ``text.get_segment(k)`` are Groups over a view of the packed
glyph batch. Their pivot used to be a row nothing but the Group moved, so after
the text itself moved, a ``move_to`` / ``rotate`` / ``scale`` of the selection
worked about where the glyphs had been. Building that view under a live camera
updater also warned about a hierarchy change that never happened.

A view of several members of any pack (``pack[2:5]``) scaled and turned each
member about its own pivot, so a scaled column grew in place without spreading.

A text's own (container) opacity row read 0 after its entrance, so
``text.opacity = text.opacity * k`` hid it for good.
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from algan import Off, Scene, Square, Tex, batch_mobs
from algan.animation_timeline.animation_contexts import Sync
from algan.constants.spatial import RIGHT, UP
from algan.errors import HierarchyChangedDuringUpdaterWarning
from algan.scene_manager import SceneManager


@pytest.fixture(autouse=True)
def fresh_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _now(scene):
    return scene.animation_manager.context.timespan.current_time


def _materialize(scene, times, read):
    tm = scene.timeline_manager
    with Off(
        record_attr_modifications=False, record_funcs=False, priority_level=math.inf
    ):
        tm.set_state_to_times(torch.tensor(times, dtype=torch.float32))
        out = read()
    tm.clear_buffers()
    return out


def _glyph_locations_after(select_before_moving, animate_text_move):
    with Scene() as scene:
        text = Tex(r"\text{abcdef}").spawn()
        if select_before_moving:
            selection = text[0:3]

        def move_text():
            text.move(RIGHT * 2)
            text.rotate(90)
            text.scale(1.5)

        if animate_text_move:
            move_text()
        else:
            with Off():
                move_text()
        if not select_before_moving:
            selection = text[0:3]
        selection.scale(2)
        selection.rotate(45)
        selection.move_to(UP)
        authored = text._character_batch.location.clone()
        end = _now(scene)
        rendered = _materialize(
            scene,
            [end - 0.5, end + 0.5],
            lambda: text._character_batch.location.clone(),
        )
    return authored, rendered


@pytest.mark.parametrize("animate_text_move", [False, True])
def test_a_glyph_selection_follows_the_text_it_was_taken_from(animate_text_move):
    late_authored, late_rendered = _glyph_locations_after(False, animate_text_move)
    early_authored, early_rendered = _glyph_locations_after(True, animate_text_move)

    assert torch.allclose(early_authored, late_authored, atol=1e-5)
    assert torch.allclose(early_rendered, late_rendered, atol=1e-5)
    assert torch.allclose(early_rendered[1:], early_authored, atol=1e-5)


def _glyph_after_segment_move_to(select_before_moving):
    with Scene():
        text = Tex("a", "b", "c").spawn()
        if select_before_moving:
            segment = text.get_segment(1)
        with Off():
            text.move(RIGHT * 3)
        if not select_before_moving:
            segment = text.get_segment(1)
        segment.move_to(UP)
        assert torch.allclose(segment.location.reshape(3), UP, atol=1e-5)
        return text._character_batch.location.reshape(-1, 3)[1].clone()


def test_a_segment_moved_to_a_point_arrives_there_after_the_text_moved():
    early = _glyph_after_segment_move_to(True)
    late = _glyph_after_segment_move_to(False)

    assert torch.allclose(early, late, atol=1e-5)
    assert torch.allclose(early, UP, atol=0.05)


def test_selecting_glyphs_of_a_text_riding_an_updated_camera_does_not_warn():
    with Scene() as scene:
        text = Tex("a", "b", "c").spawn()
        scene.camera.add_children(text)
        scene.camera.add_updater(lambda camera, t: camera.move(RIGHT * 0.01))
        scene.wait(0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("error", HierarchyChangedDuringUpdaterWarning)
            text.get_segment(1).color = torch.tensor([1.0, 0.0, 0.0])
            text[0].move(UP * 0.1)


def test_a_text_reads_its_own_opacity_after_its_entrance():
    with Scene() as scene:
        text = Tex("abc").spawn()
        assert torch.allclose(text.opacity, torch.tensor(1.0))
        assert _materialize(
            scene, [1.5], lambda: float(text.get_animated_attribute("opacity"))
        ) == pytest.approx(1.0)


def test_scaling_a_text_opacity_from_an_updater_keeps_it_visible():
    with Scene() as scene:
        text = Tex("abc").spawn()
        text.add_updater(lambda label, t: label.set(opacity=label.opacity * 0.99))
        scene.wait(1)
        opacity = _materialize(
            scene, [1.5], lambda: text._character_batch.opacity.clone()
        )
    assert torch.all(opacity > 0.5)


def _column(scene):
    squares = [Square().move(UP * i) for i in range(4)]
    return batch_mobs(squares).spawn()


def test_scaling_a_view_of_a_pack_scales_it_as_a_block():
    with Scene() as scene:
        pack = _column(scene)
        pack[0:3].scale(2)
        heights = pack.location.reshape(-1, 3)[:, 1]
        assert torch.allclose(heights, torch.tensor([-1.0, 1.0, 3.0, 3.0]), atol=1e-5)
        end = _now(scene)
        rendered = _materialize(scene, [end - 0.5], lambda: pack.location.clone())
        assert torch.allclose(
            rendered.reshape(-1, 3)[:, 1],
            torch.tensor([-0.5, 1.0, 2.5, 3.0]),
            atol=1e-4,
        )


def test_rotating_a_view_of_a_pack_turns_it_about_its_middle():
    with Scene() as scene:
        pack = _column(scene)
        pack[0:3].rotate(90)
        locations = pack.location.reshape(-1, 3)
        expected = torch.tensor(
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 3.0, 0.0]]
        )
        assert torch.allclose(locations, expected, atol=1e-5)


def test_a_view_of_one_member_still_scales_in_place():
    with Scene():
        pack = _column(None)
        with Sync():
            pack[2].scale(0.5)
        assert torch.allclose(
            pack.location.reshape(-1, 3)[:, 1], torch.tensor([0.0, 1.0, 2.0, 3.0])
        )
