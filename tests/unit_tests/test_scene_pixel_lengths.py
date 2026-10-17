"""``Scene.length_to_pixels`` and ``pixels_to_length`` measure what is drawn,
and every public Scene method answers on the class as on an instance.

A feature test, outside the fast suite: it breaks only when the Scene's own
methods change.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from PIL import Image

import algan
from algan import DOWN, IN, ORIGIN, OUT, SMOKE_TEST, UP, WHITE, Scene, Square
from algan.scene import active_scene_method
from algan.scene_manager import SceneManager

CLIP = SMOKE_TEST.set(resolution=(320, 180))


@pytest.fixture(autouse=True)
def _fresh_scenes():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _drawn_height(path):
    rows = np.where(np.asarray(Image.open(path).convert("L")).max(1) > 128)[0]
    return rows.max() - rows.min() + 1


def test_length_to_pixels_is_the_height_a_shape_is_drawn(tmp_path):
    """REGRESSION. The conversion took the frame to be 2 world units tall, the
    camera's normalized screen, where the default camera sees 8: a 2-unit square
    drawn 45 pixels tall at a height of 180 was reported as 180 pixels.
    """
    square = Square(color=WHITE).spawn(False)
    side = float(
        (square.get_edge_point(UP) - square.get_edge_point(DOWN))[..., 1].max()
    )
    Scene.set_video_settings(CLIP)
    Scene.save_frame(tmp_path / "square.png")
    assert Scene.length_to_pixels(side) == pytest.approx(
        _drawn_height(tmp_path / "square.png"), abs=1.5
    )
    assert Scene.length_to_pixels(side) == pytest.approx(side * 180 / 8)


def test_a_length_further_away_covers_fewer_pixels():
    Scene.set_video_settings(CLIP)
    near = Scene.length_to_pixels(1.0, at=OUT * 5)
    middle = Scene.length_to_pixels(1.0)
    far = Scene.length_to_pixels(1.0, at=IN * 5)
    assert near > middle > far
    height = Scene.get_camera().visible_size_at(IN * 5).reshape(-1)[1]
    assert far == pytest.approx(180 / float(height))


@pytest.mark.parametrize("at", [ORIGIN, (0.0, 0.0, -3.0), [1.0, 2.0, 2.0]])
def test_pixels_to_length_inverts_length_to_pixels(at):
    Scene.set_video_settings(CLIP)
    assert Scene.pixels_to_length(Scene.length_to_pixels(2.5, at=at), at=at) == (
        pytest.approx(2.5)
    )


def test_the_conversion_follows_the_resolution():
    Scene.set_video_settings(CLIP)
    low = Scene.length_to_pixels(1.0)
    Scene.set_video_settings(SMOKE_TEST.set(resolution=(640, 360)))
    assert Scene.length_to_pixels(1.0) == pytest.approx(2 * low)


def _public_scene_methods():
    for name, value in vars(Scene).items():
        if name.startswith("_") or isinstance(value, (staticmethod, classmethod)):
            continue
        if inspect.isfunction(value) or isinstance(value, active_scene_method):
            yield name, value


def test_every_public_scene_method_answers_on_the_class():
    """REGRESSION. ``Scene.despawn_mobs()``, ``Scene.reset()``,
    ``Scene.add_actor(mob)``, ``Scene.add_effect(effect)``,
    ``Scene.length_to_pixels(2)``, ``Scene.pixels_to_length(10)`` and
    ``Scene.background_is_transparent()`` raised "missing 1 required positional
    argument", where every other method resolves the active Scene.
    """
    plain = [
        name
        for name, value in _public_scene_methods()
        if not isinstance(value, active_scene_method)
    ]
    assert plain == []


def test_class_calls_reach_the_active_scene():
    with algan.Scene() as scene:
        square = Square(add_to_scene=False)
        assert Scene.add_actor(square) is scene
        assert square in scene.actors
        assert Scene.background_is_transparent() is scene.background_is_transparent()
        assert Scene.length_to_pixels(2.0) == scene.length_to_pixels(2.0)
        Square().spawn()
        assert Scene.despawn_mobs(animate=False) is scene
        assert Scene.reset() is scene
        assert square not in scene.actors
