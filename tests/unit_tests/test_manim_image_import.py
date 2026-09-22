"""Manim images keep their world-space corners across the geometry bridge."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from algan import Off, Scene
from algan.external_libraries import manim
from algan.mobs.image_mob import ImageMob
from algan.mobs.manim_compat import _sync_image_geometry_to_manim
from algan.mobs.manim_mob import ManimMob


@pytest.fixture
def scene():
    with Scene() as active:
        yield active


def _source_image(size, pose):
    pixels = np.full((*size, 4), 255, dtype=np.uint8)
    source = manim.ImageMobject(pixels)
    source.height = 2
    if pose == "rotated":
        source.rotate(np.pi / 3)
    elif pose == "tilted":
        source.rotate(np.pi / 4, axis=np.array([1.0, 2.0, 3.0]))
    elif pose == "stretched":
        source.stretch(1.75, 0).stretch(0.6, 1)
        source.rotate(np.pi / 6)
    elif pose == "sheared":
        source.apply_matrix(np.array([[1, 0.4, 0], [0.2, 1, 0], [0.3, -0.2, 1]]))
    elif pose == "reflected":
        source.stretch(-1, 0)
    if pose != "identity":
        source.shift(np.array([1.5, -0.75, 2.0]))
    return source


def _corners(image):
    grid = image.grid.location.reshape(image.grid_width, image.grid_height, 3)
    return torch.stack((grid[0, -1], grid[-1, -1], grid[0, 0], grid[-1, 0]))


@pytest.mark.parametrize("size", [(4, 4), (6, 2), (2, 6)])
@pytest.mark.parametrize(
    "pose",
    [
        "identity",
        "translated",
        "rotated",
        "tilted",
        "stretched",
        "sheared",
        "reflected",
    ],
)
@pytest.mark.parametrize("textured", [False, True])
def test_manim_image_import_preserves_all_four_corners(scene, size, pose, textured):
    source = _source_image(size, pose)
    original = source.points.copy()

    image = ImageMob(source, textured=textured, add_to_scene=False)

    actual = _corners(image)
    expected = torch.as_tensor(original, dtype=actual.dtype, device=actual.device)
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.allclose(image.location.reshape(3), expected.mean(0), atol=1e-6)
    np.testing.assert_array_equal(source.points, original)


def test_manim_mob_import_preserves_image_pose_and_texture_corners(scene):
    source = _source_image((2, 3), "tilted")
    source.pixel_array[0, 0] = [255, 0, 0, 255]
    source.pixel_array[-1, -1] = [0, 0, 255, 255]

    converted = ManimMob(manim.VMobject().add(source), add_to_scene=False)

    image = next(
        child for child in converted.get_descendants() if isinstance(child, ImageMob)
    )
    actual = _corners(image)
    expected = torch.as_tensor(source.points, dtype=actual.dtype, device=actual.device)
    assert torch.allclose(actual, expected, atol=1e-6)
    texture = image.color_texture
    assert torch.equal(texture[0, -1, :3], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(texture[-1, 0, :3], torch.tensor([0.0, 0.0, 1.0]))


@pytest.mark.parametrize("textured", [False, True])
@pytest.mark.parametrize("imported", [False, True])
def test_manim_image_round_trip_uses_live_geometry_corners(scene, textured, imported):
    source = _source_image((2, 6), "stretched")
    image = ImageMob(
        source if imported else torch.ones((2, 6, 4)),
        textured=textured,
        add_to_scene=False,
    )
    with Off():
        image.rotate(35, torch.tensor([1.0, 2.0, 0.0]))
        image.scale(torch.tensor([1.25, 0.75, 1.0]))
        image.move(torch.tensor([-0.5, 0.25, 1.0]))
    expected = _corners(image).detach().cpu().numpy().copy()

    _sync_image_geometry_to_manim(image, source)
    round_tripped = ImageMob(source, textured=textured, add_to_scene=False)

    np.testing.assert_allclose(source.points, expected, atol=1e-6)
    np.testing.assert_allclose(
        _corners(round_tripped).detach().cpu().numpy(), expected, atol=1e-6
    )
