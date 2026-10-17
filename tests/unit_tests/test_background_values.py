"""What ``set_background`` and the ``background=`` arguments take.

Stored as given, a color spelled the way Algan takes one everywhere else -- an
``(r, g, b)`` tuple, a list, a hex int -- failed once the render began, with
``'tuple' indices must be integers or slices, not tuple``. A callable that
cannot take ``(x, y, time)``, or that returned no color, failed as the first
frame's background was filled, with Python's or torch's error about the call.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import torch
from PIL import Image

from algan import SMOKE_TEST, Scene, Square
from algan.constants.color import Color, to_color
from algan.errors import AlganConfigurationError


def _pixels(path):
    with Image.open(path) as image:
        return np.asarray(image)


@pytest.mark.parametrize("color", [(0.1, 0.2, 0.3), [0.1, 0.2, 0.3], 0x1A334D])
def test_a_color_spelled_as_algan_takes_one_elsewhere_is_a_background(color):
    with Scene(video_settings=SMOKE_TEST) as scene:
        scene.set_background(color)
        torch.testing.assert_close(scene.background, to_color(color))
        assert not scene.background_is_transparent()


def test_a_tuple_background_renders_as_its_color(tmp_path):
    with Scene(video_settings=SMOKE_TEST) as scene:
        Square(scene=scene).spawn()
        spelled = scene.save_frame(tmp_path / "tuple.png", background=(0.1, 0.2, 0.3))
        color = scene.save_frame(
            tmp_path / "color.png", background=Color((0.1, 0.2, 0.3))
        )
    np.testing.assert_array_equal(
        _pixels(spelled.output_path), _pixels(color.output_path)
    )


@pytest.mark.parametrize("value", [(0.1, 0.2), 0.5])
def test_a_value_that_is_no_color_is_refused_when_it_is_set(value):
    with (
        Scene(video_settings=SMOKE_TEST) as scene,
        pytest.raises(AlganConfigurationError, match=r"is not a color\. Give"),
    ):
        scene.set_background(value)


def _two(x, y):
    return x.new_tensor((0.1, 0.2, 0.3))


def _four(x, y, t, extra):
    return x.new_tensor((0.1, 0.2, 0.3))


@pytest.mark.parametrize(
    ("callback", "problem"),
    [(_two, "too many positional arguments"), (_four, "missing a required")],
)
@pytest.mark.parametrize("given_to", ["set_background", "save_frame", "save_video"])
def test_a_callable_that_cannot_take_x_y_time_is_refused_before_the_render(
    tmp_path, monkeypatch, callback, problem, given_to
):
    with Scene(video_settings=SMOKE_TEST) as scene:
        Square(scene=scene).spawn()

        def fail(*_args, **_kwargs):
            raise AssertionError("the render started with a background it cannot call")

        monkeypatch.setattr(scene, "_render_to_video", fail)
        monkeypatch.setattr("algan._still_frames._StillBatch.render", fail)
        if given_to == "set_background":
            give = partial(scene.set_background, callback)
        else:
            give = partial(
                getattr(scene, given_to), tmp_path / "out", background=callback
            )
        with pytest.raises(
            AlganConfigurationError,
            match=rf"cannot be called as background\(x, y, time\): {problem}",
        ):
            give()


def test_every_callable_that_takes_x_y_time_is_accepted():
    class Gradient:
        def __call__(self, x, y, time):
            return x.new_tensor((0.1, 0.2, 0.3))

    def tinted(x, y, time, tint=0.5):
        return x.new_tensor((tint, tint, tint))

    with Scene(video_settings=SMOKE_TEST) as scene:
        for callback in (
            lambda x, y, t: x,
            lambda x, *rest: x,
            Gradient(),
            partial(tinted, tint=0.2),
        ):
            assert scene.set_background(callback).background is callback


def test_a_callable_that_returns_no_color_says_so(tmp_path):
    with Scene(video_settings=SMOKE_TEST) as scene:
        Square(scene=scene).spawn()
        with pytest.raises(
            AlganConfigurationError, match=r"background callable returned None"
        ):
            scene.save_frame(tmp_path / "none.png", background=lambda x, y, t: None)


@pytest.mark.parametrize("color", ["red", "#FF0000"])
def test_a_callable_may_return_a_color_name_or_hex_string(tmp_path, color):
    """A name or hex string was data torch could not take."""
    with Scene(video_settings=SMOKE_TEST) as scene:
        Square(scene=scene).spawn()
        returned = scene.save_frame(
            tmp_path / "returned.png", background=lambda x, y, t: color
        )
        constant = scene.save_frame(
            tmp_path / "constant.png",
            background=lambda x, y, t: x.new_tensor((1.0, 0.0, 0.0)),
        )
    np.testing.assert_array_equal(
        _pixels(returned.output_path), _pixels(constant.output_path)
    )


@pytest.mark.parametrize("level", [1, True, 0.5])
def test_a_number_a_callable_returns_is_still_a_gray_level(tmp_path, level):
    """Read as a hex int, a callback's ``1`` drew a dark blue, not white."""
    with Scene(video_settings=SMOKE_TEST) as scene:
        Square(scene=scene).spawn()
        returned = scene.save_frame(
            tmp_path / "returned.png", background=lambda x, y, t: level
        )
        constant = scene.save_frame(
            tmp_path / "constant.png",
            background=lambda x, y, t: x.new_tensor((1.0, 1.0, 1.0)) * float(level),
        )
    np.testing.assert_array_equal(
        _pixels(returned.output_path), _pixels(constant.output_path)
    )


def test_a_path_object_is_read_like_the_string_of_its_path(tmp_path):
    image = tmp_path / "backdrop.png"
    Image.new("RGB", (8, 8), (200, 30, 60)).save(image)
    with Scene(video_settings=SMOKE_TEST) as scene:
        Square(scene=scene).spawn()
        from_path = scene.save_frame(tmp_path / "path.png", background=image)
        from_string = scene.save_frame(tmp_path / "string.png", background=str(image))
        # Loaded as the image it names, not stored as the path object.
        assert torch.is_tensor(scene.set_background(image).background)
    np.testing.assert_array_equal(
        _pixels(from_path.output_path), _pixels(from_string.output_path)
    )


def test_a_still_with_a_misspelt_background_is_refused_when_it_is_requested(
    tmp_path, monkeypatch
):
    """A still's background reached set_background only at render, so a
    deferred project batch finished authoring before the typo was found.
    """
    with Scene(video_settings=SMOKE_TEST) as scene:
        Square(scene=scene).spawn()

        def fail(*args, **kwargs):
            raise AssertionError("the render started despite the misspelling")

        monkeypatch.setattr(scene, "set_background", fail)
        with pytest.raises(
            AlganConfigurationError, match=r"background 'bleu' is neither a color"
        ):
            scene.save_frame(tmp_path / "typo.png", background="bleu")


def test_a_pil_image_is_refused_with_how_to_use_it():
    with (
        Scene(video_settings=SMOKE_TEST) as scene,
        pytest.raises(AlganConfigurationError, match=r"PIL image.*file's path"),
    ):
        scene.set_background(Image.new("RGB", (8, 8)))

