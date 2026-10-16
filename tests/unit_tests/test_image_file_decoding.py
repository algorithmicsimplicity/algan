"""Every loader that reads an image file decodes its mode the same way (#150).

``ImageMob``, ``set_texture`` and material maps read files through ``get_image``;
3-D model textures, embedded glTF images and environment maps have loaders of
their own. Each of them used to interpret a file's raw Pillow samples directly,
so a palette came through as indices, CMYK as RGBA and 16-bit greyscale as
almost entirely white.
"""

import numpy as np
import pytest
import torch
from PIL import Image, PngImagePlugin

from algan import Scene
from algan.mobs.three_d_models.gltf_loader import _image_to_float_hwc
from algan.mobs.three_d_models.model_mob import _load_image_hwc
from algan.settings import path_settings
from algan.settings.video_settings import SMOKE_TEST
from algan.utils.file_utils import _decode_pil_image, get_image

# Spans the 16-bit range, including values an 8-bit conversion would clip to
# white (everything above 255) or round together (1 and 257).
SIXTEEN_BIT_SAMPLES = [[0, 1, 257], [32768, 65534, 65535]]


def _sixteen_bit_png(path, **save_kwargs):
    Image.fromarray(np.array(SIXTEEN_BIT_SAMPLES, dtype=np.uint16)).save(
        path, **save_kwargs
    )
    return path


def _expected_sixteen_bit():
    return torch.tensor(SIXTEEN_BIT_SAMPLES, dtype=torch.float32) / 65535


def _image_in_mode(tmp_path, mode):
    """A 3x2 image file in Pillow ``mode`` with distinct pixel values."""
    if mode == "I;16":
        return _sixteen_bit_png(tmp_path / "image.png")
    if mode == "P":
        image = Image.frombytes("P", (3, 2), bytes([0, 1, 2, 2, 1, 0]))
        image.putpalette([0, 255, 255, 255, 0, 0, 10, 20, 30] + [0] * 759)
        image.info["transparency"] = 0
    else:
        channels = len(Image.new(mode, (1, 1)).getbands())
        samples = np.random.default_rng(0).integers(0, 256, (2, 3, channels))
        image = Image.frombytes(mode, (3, 2), samples.astype(np.uint8).tobytes())
    # CMYK has no PNG encoding; TIFF keeps it lossless.
    path = tmp_path / ("image.tif" if mode == "CMYK" else "image.png")
    image.save(path)
    with Image.open(path) as reopened:
        assert reopened.mode == mode
    return path


def test_sixteen_bit_greyscale_keeps_its_precision(tmp_path):
    path = _sixteen_bit_png(tmp_path / "depth.png")
    with Image.open(path) as image:
        assert image.mode == "I;16"

    result = get_image(path).cpu()

    assert result.shape == (2, 3, 5)
    for channel in range(3):
        torch.testing.assert_close(result[..., channel], _expected_sixteen_bit())
    assert torch.equal(result[..., 4], torch.ones(2, 3))


def test_sixteen_bit_greyscale_transparency_becomes_alpha(tmp_path):
    path = _sixteen_bit_png(tmp_path / "keyed.png", transparency=257)

    result = get_image(path).cpu()

    expected_alpha = (torch.tensor(SIXTEEN_BIT_SAMPLES) != 257).float()
    assert torch.equal(result[..., 4], expected_alpha)
    torch.testing.assert_close(result[..., 0], _expected_sixteen_bit())


def test_sixteen_bit_png_opened_as_mode_i_by_older_pillow(tmp_path, monkeypatch):
    # Pillow before 10.3 opened a 16-bit greyscale PNG as "I" (32-bit signed)
    # rather than "I;16"; the project's minimum is Pillow 9.
    path = _sixteen_bit_png(tmp_path / "depth.png")
    monkeypatch.setitem(PngImagePlugin._MODES, (16, 0), ("I", "I;16B"))
    with Image.open(path) as image:
        assert image.mode == "I"
        decoded = _decode_pil_image(image)

    assert decoded.shape == (2, 3, 3)
    torch.testing.assert_close(
        torch.from_numpy(decoded[..., 1]), _expected_sixteen_bit()
    )


def test_float_images_are_taken_as_already_normalized():
    values = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)

    decoded = _decode_pil_image(Image.fromarray(values))

    assert decoded.dtype == np.float32
    assert np.array_equal(decoded, np.repeat(values[..., None], 3, -1))


@pytest.mark.parametrize("mode", ["L", "LA", "P", "CMYK", "RGB", "RGBA", "I;16"])
def test_model_textures_decode_like_image_mobs(tmp_path, mode):
    path = _image_in_mode(tmp_path, mode)
    # get_image's own decoding is pinned per mode by the audit regressions.
    expected = get_image(path).cpu()
    expected_alpha = expected[..., 4]

    texture_file = _load_image_hwc(path)
    with Image.open(path) as image:
        embedded = _image_to_float_hwc(image)

    for decoded in (texture_file, embedded):
        assert decoded.dtype == torch.float32
        assert decoded.shape[:2] == (2, 3)
        assert decoded.shape[-1] in (3, 4)
        torch.testing.assert_close(decoded[..., :3], expected[..., :3])
        alpha = decoded[..., 3] if decoded.shape[-1] == 4 else torch.ones(2, 3)
        torch.testing.assert_close(alpha, expected_alpha)


@pytest.mark.parametrize("mode", ["L", "LA", "P", "CMYK", "RGB", "RGBA", "I;16"])
def test_environment_map_files_decode_like_image_mobs(tmp_path, mode):
    path = _image_in_mode(tmp_path, mode)
    expected = get_image(path).cpu()[..., :3]

    scene = Scene(video_settings=SMOKE_TEST)
    scene.set_environment_map(str(path))
    torch.testing.assert_close(scene.environment_map.cpu(), expected)

    # A pathlib.Path names a file too, as it does for ImageMob.
    scene.set_environment_map(path)
    torch.testing.assert_close(scene.environment_map.cpu(), expected)


def test_background_and_environment_images_beside_the_script_are_found(
    tmp_path, monkeypatch
):
    project = tmp_path / "project"
    project.mkdir()
    script = project / "lesson.py"
    script.write_text("")
    Image.new("RGB", (4, 4), (200, 30, 30)).save(project / "sky.png")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(path_settings, "_main_script_path", lambda: str(script))

    scene = Scene(video_settings=SMOKE_TEST)
    scene.set_background("sky.png")
    scene.set_environment_map("sky.png")

    red = torch.tensor([200, 30, 30]) / 255
    torch.testing.assert_close(scene.environment_map[0, 0].cpu(), red)
    torch.testing.assert_close(
        scene.background_frame[0, 0, 0, :3].cpu(), red, atol=1.5 / 255, rtol=0
    )
