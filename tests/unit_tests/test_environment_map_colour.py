"""Environment maps: what a map's values mean, and where its file is found.

``Scene.set_environment_map`` takes an image file, an integer tensor or a float
tensor. The first two are 8-bit sRGB like every other authored color, and the
linear working space decodes them at the render boundary, so the map's
backdrop shows the image's own bytes. A float tensor is radiance, used as
given. These tests render small stills of a ball lit and mirrored by the map
and compare the ball and the backdrop around it. (A still with nothing in it at
all skips the tracer and draws the plain background, map or no map.)

A feature test, so it sits outside the fast suite: it only breaks when
environment maps themselves change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from algan import SMOKE_TEST, Scene, Sphere
from algan.rendering.raytracing import settings as rt_settings
from algan.scene_manager import SceneManager
from algan.utils.color_space import srgb_to_linear

STILL = SMOKE_TEST.set(resolution=(64, 36))
GREY = 128
# 128 / 255 taken for linear light and encoded at the write: what a byte map
# of grey 128 showed before it was decoded.
DOUBLE_ENCODED_GREY = 188


def _grey_bytes():
    return np.full((16, 32, 3), GREY, dtype=np.uint8)


@pytest.fixture(autouse=True)
def _fresh_scenes():
    SceneManager.reset()
    yield
    SceneManager.reset()


@pytest.fixture
def working_space():
    """Restore the working space a test switches."""
    previous = rt_settings.linear_color_space
    yield
    rt_settings.set_linear_color_space(previous)


def _still(tmp_path, name, source, **kwargs):
    """A still of ``source`` as the environment, with a ball it lights and mirrors."""
    with Scene() as scene:
        scene.set_environment_map(source, **kwargs)
        Sphere(radius=0.5).spawn()
        result = scene.save_frame(tmp_path / f"{name}.png", STILL)
    return np.asarray(Image.open(result.output_path)).astype(np.int16)


def _backdrop(frame):
    """A corner pixel: the map, with nothing in front of it."""
    return frame[2, 2].tolist()


def test_an_8_bit_map_shows_its_own_bytes_as_the_backdrop(tmp_path):
    """REGRESSION. An image file or an integer tensor was divided by 255 and
    taken for linear light, so the write encoded it a second time: a grey-128
    map showed as a 188 backdrop, where a background of that grey shows 128.
    """
    image = tmp_path / "grey.png"
    Image.fromarray(_grey_bytes()).save(image)

    from_file = _still(tmp_path, "file", str(image))
    from_tensor = _still(tmp_path, "tensor", torch.from_numpy(_grey_bytes()))

    assert _backdrop(from_file) == [GREY] * 3
    assert _backdrop(from_tensor) == [GREY] * 3
    assert np.array_equal(from_file, from_tensor)


def test_an_8_bit_map_lights_and_mirrors_as_its_decoded_float_twin(tmp_path):
    """REGRESSION. The same mistake lit and mirrored the ball with the encoded
    values, as brightly as it drew the backdrop. Decoded at the render
    boundary, a byte map is exactly the float map of its linear values.
    """
    as_bytes = _still(tmp_path, "bytes", torch.from_numpy(_grey_bytes()))
    linear = srgb_to_linear(torch.from_numpy(_grey_bytes()).float() / 255)
    as_floats = _still(tmp_path, "floats", linear)

    ball = as_bytes[18, 32]
    assert ball.tolist() != _backdrop(as_bytes), "the ball should be in frame"
    assert np.abs(as_bytes - as_floats).max() <= 1


def test_a_float_map_is_radiance_and_is_not_decoded(tmp_path):
    """Values are kept as authored, so a float map's backdrop is its values
    encoded: 128 / 255 of linear light writes 188, and values above 1 are not
    clipped away before ``intensity`` scales them.
    """
    as_floats = _still(tmp_path, "floats", torch.full((16, 32, 3), GREY / 255))
    brighter_than_white = _still(
        tmp_path, "hdr", torch.full((16, 32, 3), 2 * GREY / 255), intensity=0.5
    )
    assert _backdrop(as_floats) == [DOUBLE_ENCODED_GREY] * 3
    assert np.array_equal(brighter_than_white, as_floats)


@pytest.mark.usefixtures("working_space")
def test_the_working_space_at_render_time_decides(tmp_path):
    """The map is decoded when rendered, not when set, as a Mob's colors are:
    the display-referred space neither decodes nor encodes, so a byte map's
    backdrop is its bytes there too, and switching back changes nothing.
    """
    grey = torch.from_numpy(_grey_bytes())
    linear = _still(tmp_path, "linear", grey)

    rt_settings.set_linear_color_space(False)
    display = _still(tmp_path, "display", grey)
    rt_settings.set_linear_color_space(True)
    again = _still(tmp_path, "again", grey)

    assert _backdrop(display) == [GREY] * 3
    assert np.array_equal(linear, again)


def test_a_relative_path_is_found_beside_the_running_script(tmp_path, monkeypatch):
    """REGRESSION. Every other asset path is looked for beside the script when
    the working directory does not have it, which is what lets a script load
    the image next to it from wherever it is launched. The environment map
    opened the path as given, so ``set_environment_map("sky.png")`` failed
    with ``FileNotFoundError`` unless Python was started in the script's
    directory.
    """
    from algan.settings import path_settings

    project = tmp_path / "project"
    project.mkdir()
    script = project / "scene.py"
    script.write_text("")
    sky = np.zeros((16, 32, 3), dtype=np.uint8)
    sky[:8] = (40, 90, 200)
    Image.fromarray(sky).save(project / "sky.png")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(path_settings, "_main_script_path", lambda: str(script))

    with Scene() as scene:
        scene.set_environment_map("sky.png")
        expected = torch.from_numpy(sky).float() / 255
        torch.testing.assert_close(scene.environment_map, expected)


def test_a_pathlib_path_is_a_path(tmp_path):
    """REGRESSION. Only a ``str`` was read as a file: a :class:`~pathlib.Path`
    went to ``torch.tensor`` and failed with "Could not infer dtype of
    PosixPath".
    """
    image = tmp_path / "grey.png"
    Image.fromarray(_grey_bytes()).save(image)

    with Scene() as scene:
        scene.set_environment_map(Path(image))
        torch.testing.assert_close(
            scene.environment_map, torch.full((16, 32, 3), GREY / 255)
        )


def test_an_unreadable_path_names_the_file(tmp_path):
    with Scene() as scene, pytest.raises(FileNotFoundError, match="missing.png"):
        scene.set_environment_map(str(tmp_path / "missing.png"))
