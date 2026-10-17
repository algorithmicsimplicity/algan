"""An image background renders at whatever quality the render asks for.

``set_background("photo.png")`` fits the image to the Scene's frame as it
stands, so ``Scene.save_video("intro", SMOKE_TEST)`` -- the documented one-off
quality override -- or a ``save_frame`` with its own settings raised
``background resolution does not match render output``. A ``pathlib.Path``
was stored as the background unread and failed at render with ``'PosixPath'
object is not subscriptable``, and a file beside the script was refused as
"neither a color nor the path of an image file that exists".
"""

import cv2
import numpy as np
import pytest
from PIL import Image

from algan import SMOKE_TEST, Scene, Square

SMALL = SMOKE_TEST.set(resolution=(64, 36))


@pytest.fixture
def photo(tmp_path):
    # A gradient both ways, so a flipped or transposed fit shows.
    y, x = np.mgrid[0:90, 0:160]
    pixels = np.stack([x * 255 // 159, y * 255 // 89, np.zeros_like(x)], -1)
    path = tmp_path / "photo.png"
    Image.fromarray(pixels.astype(np.uint8)).save(path)
    return path


def _frame(tmp_path, name, background, scene_settings):
    with Scene(video_settings=scene_settings) as scene:
        scene.set_background(background)
        Square(size=0.2).spawn(animate=False)
        result = scene.save_frame(tmp_path / f"{name}.png", SMALL)
    return cv2.imread(str(result.output_path)).astype(int)


def test_an_image_background_set_at_one_quality_renders_at_another(tmp_path, photo):
    reference = _frame(tmp_path, "reference", str(photo), SMALL)
    refit = _frame(tmp_path, "refit", str(photo), SMOKE_TEST.set(resolution=(320, 180)))
    assert np.abs(refit - reference).max() <= 2


def test_an_image_background_can_be_a_path(tmp_path, photo):
    reference = _frame(tmp_path, "reference", str(photo), SMALL)
    assert np.abs(_frame(tmp_path, "path", photo, SMALL) - reference).max() <= 2


def test_a_video_with_its_own_quality_keeps_the_image_background(tmp_path, photo):
    with Scene() as scene:
        scene.set_background(str(photo))
        Square(size=0.2).spawn(animate=False)
        result = scene.save_video(tmp_path / "clip.mp4", SMALL)
    assert result.status == "rendered"


def test_an_image_background_beside_the_script_is_found(tmp_path, photo, monkeypatch):
    from algan.settings import path_settings

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(path_settings, "_main_script_path", lambda: tmp_path / "v.py")
    with Scene(video_settings=SMALL) as scene:
        scene.set_background("photo.png")
        assert scene.background.dim() == 4
