"""Contracts of the Manim-compatible raster helpers and pure plot helpers.

``image_compat`` converts between Algan's 5-channel (RGB, glow, alpha) float
textures and Manim's uint8 RGBA pixel arrays.  A rounding or channel-order slip
here silently shifts every pixel of every ImageMobject, so these tests pin the
conversions and the array-level manim-compat methods.  Constructing a tiny
ImageMobject is cheap (no kernel compilation), so those methods are covered too.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch
from PIL import Image
from PIL.Image import Resampling

from algan.errors import AlganConfigurationError
from algan.mobs.image_compat import (
    ImageMobject,
    _color_to_rgb8,
    _load_rgba5,
    _rgba5_to_uint8,
)
from algan.mobs.plots import convert_points_to_path, get_corners


def _rand_rgba(h=4, w=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 4), dtype=np.uint8)


# --------------------------------------------------------------------------
# _load_rgba5
# --------------------------------------------------------------------------


def test_load_rgba5_uint8_rgba_scaled_to_unit_range():
    a = _rand_rgba()
    r = _load_rgba5(a)
    assert r.shape == (4, 6, 5)
    assert r.dtype == torch.get_default_dtype()
    np.testing.assert_allclose(r[..., :3].numpy(), a[..., :3] / 255, atol=1e-6)
    np.testing.assert_allclose(r[..., 4].numpy(), a[..., 3] / 255, atol=1e-6)
    assert torch.all(r[..., 3] == 0)  # glow channel defaults to zero


def test_load_rgba5_2d_array_is_greyscale():
    a = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    r = _load_rgba5(a)
    assert r.shape == (2, 2, 5)
    for c in range(3):
        np.testing.assert_allclose(r[..., c].numpy(), a / 255, atol=1e-6)
    assert torch.all(r[..., 4] == 1)  # opaque by default


def test_load_rgba5_single_channel_array_is_greyscale():
    a = np.array([[[10], [200]]], dtype=np.uint8)
    r = _load_rgba5(a)
    assert r.shape == (1, 2, 5)
    np.testing.assert_allclose(r[0, 1, :3].numpy(), [200 / 255] * 3, atol=1e-6)


def test_load_rgba5_rgb_array_gets_opaque_alpha():
    a = _rand_rgba()[..., :3]
    r = _load_rgba5(a)
    assert r.shape == (4, 6, 5)
    assert torch.all(r[..., 4] == 1)


def test_load_rgba5_float_array_is_clamped():
    a = np.array([[[-0.5, 0.5, 2.0, 1.0]]], dtype=np.float32)
    r = _load_rgba5(a)
    assert r.min() >= 0
    assert r.max() <= 1
    np.testing.assert_allclose(r[0, 0, :3].numpy(), [0.0, 0.5, 1.0], atol=1e-6)


def test_load_rgba5_tensor_input_passthrough():
    t = torch.rand(3, 2, 4)
    r = _load_rgba5(t)
    assert r.shape == (3, 2, 5)
    torch.testing.assert_close(r[..., :3], t[..., :3])
    torch.testing.assert_close(r[..., 4], t[..., 3])


@pytest.mark.parametrize("as_path", [True, False])
def test_load_rgba5_from_file(tmp_path, as_path):
    a = _rand_rgba(3, 5)
    p = tmp_path / "img.png"
    Image.fromarray(a).save(p)
    r = _load_rgba5(p if as_path else str(p))
    assert r.shape == (3, 5, 5)
    np.testing.assert_array_equal(_rgba5_to_uint8(r), a)


# --------------------------------------------------------------------------
# _rgba5_to_uint8
# --------------------------------------------------------------------------


def test_rgba5_to_uint8_drops_glow_and_keeps_alpha():
    v = torch.tensor([[[1.0, 0.5, 0.0, 0.9, 0.25]]])
    out = _rgba5_to_uint8(v)
    assert out.dtype == np.uint8
    assert out.shape == (1, 1, 4)
    np.testing.assert_array_equal(out[0, 0], [255, 128, 0, 64])


def test_rgba5_to_uint8_clamps_out_of_range():
    v = torch.tensor([[[-1.0, 2.0, 0.5, 0.0, 3.0]]])
    np.testing.assert_array_equal(_rgba5_to_uint8(v)[0, 0], [0, 255, 128, 255])


def test_uint8_roundtrip_is_lossless():
    a = _rand_rgba(8, 8, seed=3)
    np.testing.assert_array_equal(_rgba5_to_uint8(_load_rgba5(a)), a)


def test_rgba5_to_uint8_accepts_grad_tensor():
    v = torch.rand(2, 2, 5, requires_grad=True)
    out = _rgba5_to_uint8(v)
    assert out.shape == (2, 2, 4)


# --------------------------------------------------------------------------
# _color_to_rgb8
# --------------------------------------------------------------------------


def test_color_to_rgb8_hex_string():
    np.testing.assert_array_equal(_color_to_rgb8("#FF0000"), [255, 0, 0])
    np.testing.assert_array_equal(_color_to_rgb8("#00FF00"), [0, 255, 0])


def test_color_to_rgb8_sequence_ignores_alpha_and_clips():
    np.testing.assert_array_equal(_color_to_rgb8([0.5, 0.5, 0.5, 0.1]), [128] * 3)
    np.testing.assert_array_equal(_color_to_rgb8([-1, 2, 0.0]), [0, 255, 0])


def test_color_to_rgb8_object_with_to_rgb():
    class C:
        def to_rgb(self):
            return np.array([0.0, 1.0, 0.2])

    np.testing.assert_array_equal(_color_to_rgb8(C()), [0, 255, 51])


def test_color_to_rgb8_algan_color():
    from algan.constants.color import Color

    out = _color_to_rgb8(Color("#0000FF"))
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, [0, 0, 255])


# --------------------------------------------------------------------------
# ImageMobject array-level manim-compat methods
# --------------------------------------------------------------------------


def test_image_mobject_pixel_array_matches_input():
    a = _rand_rgba()
    m = ImageMobject(a)
    np.testing.assert_array_equal(m.get_pixel_array(), a)
    np.testing.assert_array_equal(m.orig_alpha_pixel_array, a[..., 3])
    assert m.path is None


def test_image_mobject_invert_flips_rgb_only():
    a = _rand_rgba()
    m = ImageMobject(a, invert=True)
    np.testing.assert_array_equal(m.pixel_array[..., :3], 255 - a[..., :3])
    np.testing.assert_array_equal(m.pixel_array[..., 3], a[..., 3])


def test_image_mobject_set_opacity_scales_original_alpha():
    a = _rand_rgba()
    m = ImageMobject(a)
    m.set_opacity(0.5)
    expected = np.rint(a[..., 3].astype(np.float32) * 0.5).astype(np.uint8)
    np.testing.assert_array_equal(m.pixel_array[..., 3], expected)
    # Repeated calls scale from the original, not compound.
    m.set_opacity(0.5)
    np.testing.assert_array_equal(m.pixel_array[..., 3], expected)
    m.set_opacity(1.0)
    np.testing.assert_array_equal(m.pixel_array[..., 3], a[..., 3])


def test_image_mobject_fade_is_one_minus_darkness():
    a = _rand_rgba()
    m = ImageMobject(a)
    m.fade(0.25)
    expected = np.rint(a[..., 3].astype(np.float32) * 0.75).astype(np.uint8)
    np.testing.assert_array_equal(m.pixel_array[..., 3], expected)


def test_image_mobject_fill_opacity_kwarg():
    a = _rand_rgba()
    m = ImageMobject(a, fill_opacity=0.0)
    assert np.all(m.pixel_array[..., 3] == 0)


def test_image_mobject_set_color_sets_rgb_and_optional_alpha():
    a = _rand_rgba()
    m = ImageMobject(a)
    m.set_color("#FF0000")
    assert np.all(m.pixel_array[..., :3] == [255, 0, 0])
    np.testing.assert_array_equal(m.pixel_array[..., 3], a[..., 3])
    m.set_color([0, 0, 1], alpha=0.5)
    assert np.all(m.pixel_array[..., :3] == [0, 0, 255])
    assert np.all(m.pixel_array[..., 3] == 128)


def test_image_mobject_sync_texture_layout():
    a = _rand_rgba(4, 6)
    m = ImageMobject(a)
    m.set_opacity(1.0)  # forces _sync_texture
    tex = m.color_texture
    assert tex.shape[-3:] == (6, 4, 5)
    # texture is pixel array transposed and flipped along the row axis
    expected = torch.from_numpy(a).float() / 255
    expected = expected.transpose(0, 1).flip(1)
    torch.testing.assert_close(tex[..., :3].cpu().float(), expected[..., :3])
    torch.testing.assert_close(tex[..., 4].cpu().float(), expected[..., 3])
    assert torch.all(tex[..., 3] == 0)


def test_image_mobject_interpolate_color():
    a = np.zeros((2, 3, 4), np.uint8)
    b = np.full((2, 3, 4), 200, np.uint8)
    m1, m2, m = ImageMobject(a), ImageMobject(b), ImageMobject(a)
    m.interpolate_color(m1, m2, 0.25)
    assert np.all(m.pixel_array == 50)
    assert m.pixel_array.dtype == np.uint8


def test_image_mobject_interpolate_color_shape_mismatch():
    m1 = ImageMobject(_rand_rgba(2, 2))
    m2 = ImageMobject(_rand_rgba(3, 3))
    with pytest.raises(AssertionError):
        m1.interpolate_color(m1, m2, 0.5)


def test_image_mobject_get_style_reports_mean_alpha():
    a = _rand_rgba()
    a[..., 3] = 51
    m = ImageMobject(a)
    assert m.get_style()["fill_opacity"] == pytest.approx(0.2)


def test_image_mobject_resampling_algorithm():
    m = ImageMobject(_rand_rgba(), resampling_algorithm=Resampling.NEAREST)
    assert m.resampling_algorithm == int(Resampling.NEAREST)
    m.set_resampling_algorithm(int(Resampling.LANCZOS))
    assert m.resampling_algorithm == int(Resampling.LANCZOS)
    with pytest.raises(AlganConfigurationError):
        m.set_resampling_algorithm("bicubic")


def test_image_mobject_from_path_records_path(tmp_path):
    a = _rand_rgba(3, 3)
    p = tmp_path / "x.png"
    Image.fromarray(a).save(p)
    m = ImageMobject(str(p))
    assert m.path == pathlib.Path(p)
    np.testing.assert_array_equal(m.pixel_array, a)


# --------------------------------------------------------------------------
# plots.convert_points_to_path / get_corners
# --------------------------------------------------------------------------


def test_convert_points_to_path_structure():
    from svgelements import Close, Line, Move

    pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
    path = convert_points_to_path(pts)
    segs = list(path)
    assert isinstance(segs[0], Move)
    assert sum(isinstance(s, Line) and not isinstance(s, Close) for s in segs) == 3
    assert isinstance(segs[-1], Close)
    assert len(segs) == 5
    assert tuple(segs[-1].end) == (0, 0)


def test_get_corners_axis_aligned():
    start = torch.tensor([0.0, 0.0, 0.0])
    direction = torch.tensor([0.0, 1.0, 0.0])
    facing = torch.tensor([0.0, 0.0, 1.0])
    c = get_corners(start, direction, 0.5, 2.0, facing)
    assert c.shape == (4, 3)
    expected = torch.tensor([[0.5, 0, 0], [0.5, 2, 0], [-0.5, 2, 0], [-0.5, 0, 0]])
    torch.testing.assert_close(c, expected)


def test_get_corners_batched():
    start = torch.zeros(3, 3)
    start[:, 0] = torch.arange(3.0)
    direction = torch.tensor([[0.0, 1.0, 0.0]]).expand(3, 3)
    facing = torch.tensor([[0.0, 0.0, 1.0]]).expand(3, 3)
    c = get_corners(start, direction, 0.1, 1.0, facing)
    assert c.shape == (3, 4, 3)
    # rectangle width is 2 * width, height is height
    torch.testing.assert_close((c[:, 0] - c[:, 3]).norm(dim=-1), torch.full((3,), 0.2))
    torch.testing.assert_close((c[:, 1] - c[:, 0]).norm(dim=-1), torch.ones(3))
