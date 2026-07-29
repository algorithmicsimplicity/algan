from __future__ import annotations

import pytest
import torch

from algan.constants.color import Color
from algan.mobs.image_mob import ImageMob
from algan.mobs.shapes_3d import Cylinder
from algan.mobs.surfaces.surface import Surface
from algan.scene_manager import SceneManager


@pytest.fixture(autouse=True)
def reset_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


def test_set_shape_to_increases_resolution_to_match_finer_surface():
    source = Surface(
        grid_width=2,
        grid_height=3,
        add_to_scene=False,
    ).spawn(animate=False)
    target = Cylinder(
        grid_width=7,
        grid_height=5,
        add_to_scene=False,
    )

    source.set_shape_to(target)

    assert (source.grid_width, source.grid_height) == (7, 5)
    assert source.grid.location.shape[-2:] == (35, 3)
    expected = target.coord_function(source.get_base_grid().clone()).reshape(1, -1, 3)
    assert torch.allclose(source.grid.location, expected, atol=1e-6)

    source.scene.timeline_manager.set_state_to_times(torch.tensor([0.0, 0.5, 1.0]))
    assert source.grid.location.shape == (3, 35, 3)
    assert torch.allclose(source.grid.location[-1:], expected, atol=1e-6)


def test_set_shape_to_does_not_reduce_either_resolution_axis():
    source = Surface(
        grid_width=8,
        grid_height=3,
        add_to_scene=False,
    )
    target = Cylinder(
        grid_width=5,
        grid_height=7,
        add_to_scene=False,
    )

    source.set_shape_to(target)

    assert (source.grid_width, source.grid_height) == (8, 7)


def test_image_mob_textured_false_uses_one_vertex_color_per_pixel():
    pixels = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.25],
                [0.0, 1.0, 0.0, 0.50],
                [0.0, 0.0, 1.0, 0.75],
            ],
            [
                [1.0, 1.0, 0.0, 1.00],
                [0.0, 1.0, 1.0, 0.80],
                [1.0, 0.0, 1.0, 0.60],
            ],
        ]
    )

    image = ImageMob(pixels, textured=False, add_to_scene=False)

    expected = (
        Color.add_defaults(pixels)
        .transpose(-3, -2)
        .flip(-2)
        .reshape(1, -1, 5)
    )
    assert (image.grid_width, image.grid_height) == (3, 2)
    assert image.color_texture is None
    assert torch.equal(image.grid.color, expected)
    assert image.get_render_primitives().texture_map is None


def test_image_mob_remains_texture_backed_by_default():
    pixels = torch.zeros((3, 4, 4))
    pixels[..., 3] = 1

    image = ImageMob(pixels, add_to_scene=False)

    assert (image.grid_width, image.grid_height) == (2, 2)
    assert image.color_texture is not None
    assert image.get_render_primitives().texture_map is not None
