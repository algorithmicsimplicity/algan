"""``algan.manim``'s raster images keep Manim's pixel-array contract.

Manim's ``ImageMobject`` is a numpy RGBA array with a quad around it, and
everything it does to its look is arithmetic on that array: ``set_color``
paints every pixel's RGB, ``set_opacity`` and ``fade`` scale the alpha the
image was loaded with, ``interpolate_color`` lerps two arrays. Algan's
compatibility class keeps that array as its public state and re-derives the
texture the renderer samples from it, so these tests read both -- the array a
Manim script inspects, and the ``color_texture`` that is actually drawn.

Vendored Manim is the reference wherever it can be built; its abstract base
cannot (its constructor asks the subclass for pixels). Manim truncates where
Algan rounds (``int(255 * alpha)``, a float array assigned into ``uint8``), so
the alphas and weights here are chosen to land on whole levels in both.

Feature tests: unmarked, so outside the fast suite.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL.Image import Resampling

import algan.manim as mn
from algan import BLUE, Scene
from algan.external_libraries import manim


@pytest.fixture
def scene():
    with Scene() as active:
        yield active


def _pixels(channels=4):
    """A 2 x 3 picture whose pixels all differ.

    Every alpha is a multiple of four, so halving or quartering it is a whole
    level whether it is rounded (Algan) or truncated (Manim).
    """
    pixels = np.empty((2, 3, 4), dtype=np.uint8)
    pixels[..., 0] = [[10, 20, 30], [40, 50, 60]]
    pixels[..., 1] = [[100, 110, 120], [130, 140, 150]]
    pixels[..., 2] = [[200, 210, 220], [230, 240, 250]]
    pixels[..., 3] = [[252, 128, 0], [64, 200, 100]]
    return pixels[..., :channels].copy()


def _pair(pixels, **kwargs):
    """The same picture as Algan's compatibility image and as Manim's own."""
    ours = mn.ImageMobject(pixels.copy(), add_to_scene=False, **kwargs)
    theirs = manim.ImageMobject(pixels.copy(), **kwargs)
    return ours, theirs


def _drawn_pixels(image):
    """The image's color texture, read back as Manim's ``[H, W, RGBA]`` array.

    The texture is ``[W, H, 5]`` (RGB, glow, alpha) with ``v`` running up the
    picture while an array's rows run down it, so it is turned back first.
    """
    texture = image.color_texture.as_subclass(torch.Tensor).flip(-2).transpose(-3, -2)
    rgba = torch.cat((texture[..., :3], texture[..., 4:5]), -1)
    return (rgba.cpu().numpy() * 255).round().astype(np.uint8)


def _corners(image):
    """Top-left, top-right, bottom-left, bottom-right: Manim's ``points`` order."""
    grid = image.grid.location.reshape(image.grid_width, image.grid_height, 3)
    return torch.stack((grid[0, -1], grid[-1, -1], grid[0, 0], grid[-1, 0]))


@pytest.mark.parametrize("shape", [(2, 4), (2, 4, 1)], ids=["2-D", "one-channel"])
def test_a_grey_array_loads_as_manims_grey_rgba(scene, shape):
    """REGRESSION. A 2-D or one-channel array -- Manim's own ``ImageFromArray``
    example -- was refused, its columns or its lone channel taken for the color
    channels, where Manim reads it as grey RGB under an opaque alpha.
    """
    grey = np.uint8([[0, 100, 30, 200], [255, 0, 5, 33]]).reshape(shape)

    image = mn.ImageMobject(grey, add_to_scene=False)

    reference = manim.ImageMobject(grey).get_pixel_array()
    assert image.get_pixel_array().dtype == np.uint8
    np.testing.assert_array_equal(image.get_pixel_array(), reference)
    np.testing.assert_array_equal(_drawn_pixels(image), reference)


@pytest.mark.parametrize("channels", [3, 4], ids=["RGB", "RGBA"])
@pytest.mark.parametrize("invert", [False, True], ids=["plain", "inverted"])
def test_an_array_image_keeps_manims_pixel_array(scene, channels, invert):
    image, reference = _pair(_pixels(channels), invert=invert)

    np.testing.assert_array_equal(image.pixel_array, reference.pixel_array)
    assert image.get_pixel_array() is image.pixel_array
    np.testing.assert_array_equal(
        image.orig_alpha_pixel_array, reference.orig_alpha_pixel_array
    )
    # The texture is what the renderer samples, so it must be the same pixels.
    np.testing.assert_array_equal(_drawn_pixels(image), image.pixel_array)


@pytest.mark.parametrize("scale_to_resolution", [None, 540, 1080])
def test_an_image_is_sized_by_manims_scale_to_resolution_rule(
    scene, scale_to_resolution
):
    """Manim places an image pixel for pixel at ``scale_to_resolution`` rows
    per frame height (8 units), so a 90 x 60 picture is 2/3 of a unit tall at
    1080 and twice that at 540; ``None`` here means Manim's default.
    """
    kwargs = (
        {}
        if scale_to_resolution is None
        else {"scale_to_resolution": scale_to_resolution}
    )
    image, reference = _pair(np.zeros((90, 60, 4), dtype=np.uint8), **kwargs)

    assert float(image.get_height()) == pytest.approx(reference.height, rel=1e-6)
    assert float(image.get_width()) == pytest.approx(reference.width, rel=1e-6)


def test_reset_points_leaves_a_fresh_image_on_manims_corners(scene):
    """Manim's ``reset_points`` re-centres the quad at its natural size, which
    is exactly where a freshly loaded image already is.
    """
    image, reference = _pair(_pixels())

    assert image.reset_points() is image
    reference.reset_points()

    expected = torch.as_tensor(reference.points, dtype=torch.get_default_dtype())
    torch.testing.assert_close(_corners(image), expected, atol=1e-6, rtol=0)


def test_set_color_paints_every_pixel_like_manim(scene):
    image, reference = _pair(_pixels())

    assert image.set_color("#336699") is image
    reference.set_color("#336699")
    np.testing.assert_array_equal(image.pixel_array, reference.pixel_array)
    # Without ``alpha`` the transparency the picture had is kept.
    np.testing.assert_array_equal(image.pixel_array[..., 3], _pixels()[..., 3])

    image.set_color(BLUE, alpha=0.2)
    reference.set_color(manim.BLUE, alpha=0.2)
    np.testing.assert_array_equal(image.pixel_array, reference.pixel_array)
    assert (image.pixel_array[..., 3] == 51).all()
    np.testing.assert_array_equal(_drawn_pixels(image), image.pixel_array)


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        pytest.param(BLUE, manim.BLUE, id="Algan-constant"),
        pytest.param(
            manim.ManimColor("#336699"), manim.ManimColor("#336699"), id="ManimColor"
        ),
        pytest.param("#FF8000", "#FF8000", id="hex-string"),
        pytest.param((0.2, 0.4, 0.6), (0.2, 0.4, 0.6), id="float-triple"),
        pytest.param(
            np.array([0.2, 0.4, 0.6, 1.0]),
            np.array([0.2, 0.4, 0.6, 1.0]),
            id="RGBA-array",
        ),
    ],
)
def test_set_color_takes_the_color_spellings_manim_takes(scene, ours, theirs):
    image, reference = _pair(_pixels())

    image.set_color(ours)
    reference.set_color(theirs)

    np.testing.assert_array_equal(image.pixel_array, reference.pixel_array)


def test_set_color_repaints_child_images_too(scene):
    """Manim's ``set_color`` walks its submobjects; an image's children are
    repainted with it.
    """
    parent = mn.ImageMobject(_pixels(), add_to_scene=False)
    child = mn.ImageMobject(_pixels(3), add_to_scene=False)
    parent.add_children(child)

    parent.set_color("#336699")

    assert (child.pixel_array[..., :3] == (0x33, 0x66, 0x99)).all()
    np.testing.assert_array_equal(_drawn_pixels(child), child.pixel_array)


def test_set_opacity_and_fade_scale_the_alpha_the_image_was_loaded_with(scene):
    loaded = _pixels()[..., 3]
    image, reference = _pair(_pixels())

    assert image.set_opacity(0.5) is image
    reference.set_opacity(0.5)
    np.testing.assert_array_equal(image.pixel_array, reference.pixel_array)
    np.testing.assert_array_equal(image.pixel_array[..., 3], loaded // 2)

    # Relative to the loaded alpha, not to the last call's.
    image.set_opacity(0.25)
    reference.set_opacity(0.25)
    np.testing.assert_array_equal(image.pixel_array, reference.pixel_array)
    np.testing.assert_array_equal(image.pixel_array[..., 3], loaded // 4)

    assert image.fade(0.5) is image
    reference.fade(0.5)
    np.testing.assert_array_equal(image.pixel_array, reference.pixel_array)
    np.testing.assert_array_equal(image.pixel_array[..., 3], loaded // 2)

    image.fade(0)
    reference.fade(0)
    np.testing.assert_array_equal(image.pixel_array, reference.pixel_array)
    np.testing.assert_array_equal(image.pixel_array, _pixels())
    np.testing.assert_array_equal(_drawn_pixels(image), image.pixel_array)


def test_interpolate_color_lerps_two_pixel_arrays_and_returns_the_image(scene):
    """REGRESSION. It returned ``None`` where Manim's returns the image, so a
    call chained on it failed.
    """
    start = np.full((2, 2, 4), (0, 100, 200, 255), dtype=np.uint8)
    end = np.full((2, 2, 4), (200, 40, 0, 55), dtype=np.uint8)
    ours = [mn.ImageMobject(p, add_to_scene=False) for p in (start, end, start)]
    theirs = [manim.ImageMobject(p.copy()) for p in (start, end, start)]

    for alpha, expected in ((0, start), (0.25, (50, 85, 150, 205)), (1, end)):
        assert ours[2].interpolate_color(ours[0], ours[1], alpha) is ours[2]
        theirs[2].interpolate_color(theirs[0], theirs[1], alpha)
        np.testing.assert_array_equal(ours[2].pixel_array, theirs[2].pixel_array)
        np.testing.assert_array_equal(
            ours[2].pixel_array, np.broadcast_to(expected, start.shape)
        )
        np.testing.assert_array_equal(_drawn_pixels(ours[2]), ours[2].pixel_array)


def test_interpolate_color_refuses_pictures_of_different_shapes(scene):
    small = mn.ImageMobject(np.zeros((2, 2, 4), np.uint8), add_to_scene=False)
    tall = mn.ImageMobject(np.zeros((3, 2, 4), np.uint8), add_to_scene=False)

    # Manim asserts the shapes match; so does Algan.
    with pytest.raises(AssertionError, match="incompatible"):
        small.interpolate_color(small, tall, 0.5)


def test_get_style_reports_the_opacity_the_pixels_carry(scene):
    """Manim hands back its ``fill_opacity`` attribute, which is 1 for any
    picture it loads. Algan reads the same key off the alpha channel, so the
    two agree on a loaded image and Algan's also follows ``set_opacity``.
    """
    opaque = np.full((2, 2, 4), 255, dtype=np.uint8)
    image, reference = _pair(opaque)
    assert image.get_style()["fill_opacity"] == pytest.approx(
        reference.get_style()["fill_opacity"]
    )

    image.set_opacity(0.4)
    assert image.get_style()["fill_opacity"] == pytest.approx(102 / 255)

    half_transparent = opaque.copy()
    half_transparent[0, :, 3] = 0
    style = mn.ImageMobject(half_transparent, add_to_scene=False).get_style()
    assert style["fill_opacity"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "algorithm",
    sorted(set(manim.RESAMPLING_ALGORITHMS.values())),
    ids=lambda algorithm: algorithm.name,
)
def test_set_resampling_algorithm_takes_manims_constants_and_chains(scene, algorithm):
    image = mn.ImageMobject(_pixels(), add_to_scene=False)
    assert image.resampling_algorithm == Resampling.BICUBIC  # Manim's default

    assert image.set_resampling_algorithm(algorithm) is image
    assert image.resampling_algorithm == algorithm

    built = mn.ImageMobject(
        _pixels(), resampling_algorithm=algorithm, add_to_scene=False
    )
    assert built.resampling_algorithm == algorithm


def test_set_resampling_algorithm_refuses_an_algorithm_name(scene):
    """Manim takes the Pillow constants -- ``RESAMPLING_ALGORITHMS`` maps the
    names onto them -- and raises ``ValueError`` for the name itself.
    """
    image, reference = _pair(_pixels())

    with pytest.raises(ValueError):
        reference.set_resampling_algorithm("nearest")
    with pytest.raises(ValueError):
        image.set_resampling_algorithm("nearest")
    assert image.resampling_algorithm == Resampling.BICUBIC


def test_the_abstract_base_leaves_the_pixels_to_its_subclasses(scene):
    abstract = mn.AbstractImageMobject(scale_to_resolution=None, add_to_scene=False)

    with pytest.raises(NotImplementedError):
        abstract.get_pixel_array()
    assert abstract.set_resampling_algorithm(Resampling.NEAREST) is abstract
    assert abstract.resampling_algorithm == Resampling.NEAREST


class _ForeignCamera:
    """Anything with a ``pixel_array`` -- which is what Manim's cameras are."""

    def __init__(self, pixel_array):
        self.pixel_array = pixel_array


def test_a_foreign_camera_display_is_three_units_tall_like_manims(scene):
    """Manim sizes a camera's display at a fixed height of 3, its width
    following the picture's aspect.
    """
    pixels = _pixels()

    display = mn.ImageMobjectFromCamera(_ForeignCamera(pixels), add_to_scene=False)

    np.testing.assert_array_equal(display.pixel_array, pixels)
    assert float(display.get_height()) == pytest.approx(3)
    assert float(display.get_width()) == pytest.approx(3 * 3 / 2)


def test_a_foreign_camera_display_reads_the_cameras_current_pixels(scene):
    camera = _ForeignCamera(_pixels())
    display = mn.ImageMobjectFromCamera(camera, add_to_scene=False)

    camera.pixel_array = np.full((2, 3, 4), 7, dtype=np.uint8)
    assert display.get_pixel_array() is camera.pixel_array
    assert display.pixel_array is camera.pixel_array

    # A torch camera image is handed back as the numpy array Manim's API uses.
    camera.pixel_array = torch.full((2, 3, 4), 9, dtype=torch.uint8)
    pixels = display.get_pixel_array()
    assert isinstance(pixels, np.ndarray)
    np.testing.assert_array_equal(pixels, np.full((2, 3, 4), 9, dtype=np.uint8))


def test_a_foreign_camera_display_fades_its_pixels(scene):
    display = mn.ImageMobjectFromCamera(_ForeignCamera(_pixels()), add_to_scene=False)

    assert display.set_opacity(0.5) is display

    np.testing.assert_array_equal(display.pixel_array[..., 3], _pixels()[..., 3] // 2)
    np.testing.assert_array_equal(_drawn_pixels(display), display.pixel_array)


def test_a_live_camera_display_fades_the_display_not_a_snapshot(scene):
    """An Algan camera renders live, so there are no pixels to scale: the
    display's own opacity carries the fade.
    """
    display = mn.ImageMobjectFromCamera(
        scene.camera, resolution=(48, 32), add_to_scene=False
    )
    snapshot = display.pixel_array.copy()

    assert display.set_opacity(0.25) is display

    assert display.opacity.reshape(-1).tolist() == pytest.approx([0.25])
    np.testing.assert_array_equal(display.pixel_array, snapshot)
    # Three units tall like any camera display, as wide as the capture.
    assert float(display.get_height()) == pytest.approx(3)
    assert float(display.get_width()) == pytest.approx(3 * 48 / 32)
