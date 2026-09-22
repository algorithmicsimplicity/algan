"""Imported color units come from the source dtype, never its brightest pixel."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from algan.mobs.three_d_models.gltf_loader import (
    _convert_material,
    _convert_mesh,
    _image_to_float_hwc,
    _normalize_color,
)


@pytest.mark.parametrize("maximum", [0, 1, 2, 255])
@pytest.mark.parametrize("pil", [False, True])
def test_byte_images_are_normalized_even_when_every_pixel_is_dark(maximum, pil):
    pixels = np.array([[[0, maximum, maximum, maximum]]], dtype=np.uint8)
    pixels.setflags(write=False)
    image = Image.fromarray(pixels) if pil else pixels

    actual = _image_to_float_hwc(image)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, torch.tensor(pixels.astype(np.float32) / 255))
    assert int(pixels.max()) == maximum


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.dtype(">u2")])
def test_unsigned_color_and_grayscale_samples_use_the_full_declared_range(dtype):
    maximum = np.iinfo(dtype).max
    pixels = np.array([[0, 1, maximum]], dtype=dtype)
    image = _image_to_float_hwc(pixels)
    assert image.shape == (1, 3, 1)
    assert image.flatten().tolist() == pytest.approx([0, 1 / maximum, 1])
    assert _normalize_color(pixels) == pytest.approx((0, 1 / maximum, 1, 1))


@pytest.mark.parametrize("alpha", [None, 0, 1, 2, 255])
def test_byte_rgb_defaults_to_opaque_and_rgba_preserves_small_alpha(alpha):
    color = np.array([255, 0, 1] + ([] if alpha is None else [alpha]), dtype=np.uint8)
    assert _normalize_color(color) == pytest.approx(
        (1, 0, 1 / 255, 1 if alpha is None else alpha / 255)
    )
    assert _normalize_color([255, 0, 0]) == (1, 0, 0, 1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_normalized_floats_keep_their_values(dtype):
    values = np.array([0, 1 / 255, 1, 0.25], dtype=dtype)
    assert _normalize_color(values) == pytest.approx(values)
    torch.testing.assert_close(
        _image_to_float_hwc(values.reshape(1, 1, 4)),
        torch.tensor(values.astype(np.float32)).reshape(1, 1, 4),
    )


def test_boolean_images_are_normalized_masks():
    assert _image_to_float_hwc(np.array([[False, True]])).flatten().tolist() == [0, 1]


@pytest.mark.parametrize(
    "values",
    [
        np.array([256], dtype=np.int32),
        np.array([-1], dtype=np.int16),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([1.5]),
    ],
)
def test_invalid_or_ambiguous_sample_ranges_are_refused(values):
    with pytest.raises(ValueError, match="must be in"):
        _normalize_color(values)
    with pytest.raises(ValueError, match="must be in"):
        _image_to_float_hwc(values.reshape(1, 1, 1))


def test_material_textures_and_vertex_colors_share_the_same_units():
    image = np.ones((2, 2, 3), dtype=np.uint8)
    material = SimpleNamespace(
        baseColorFactor=np.array([255, 0, 0], dtype=np.uint8),
        baseColorTexture=image,
    )
    converted = _convert_material(SimpleNamespace(material=material))
    assert converted.base_color == (1, 0, 0, 1)
    assert converted.opacity == 1
    torch.testing.assert_close(converted.diffuse_image, torch.full((2, 2, 3), 1 / 255))
    mesh = _convert_mesh(
        SimpleNamespace(
            vertices=np.zeros((3, 3)),
            faces=np.array([[0, 1, 2]]),
            visual=SimpleNamespace(vertex_colors=np.ones((3, 4), dtype=np.uint8)),
        ),
        0,
        "dark mesh",
    )
    torch.testing.assert_close(mesh.vertex_colors, torch.full((3, 4), 1 / 255))
