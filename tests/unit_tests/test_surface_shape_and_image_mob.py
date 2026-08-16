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


def test_set_shape_to_keeps_textured_resolution_history_independent():
    pixels = torch.ones((3, 4, 4))
    source = ImageMob(pixels).spawn(animate=False)
    source.wait()
    target = Cylinder(
        grid_width=7,
        grid_height=6,
        add_to_scene=False,
    )

    source.set_shape_to(target)

    historical = [
        actor
        for actor in source.scene.actors
        if isinstance(actor, ImageMob) and actor is not source
    ]
    assert len(historical) == 1
    historical = historical[0]
    assert historical.id != source.id
    assert historical.grid.id != source.grid.id
    assert historical.grid.location.shape[-2:] == (4, 3)
    assert source.grid.location.shape[-2:] == (42, 3)

    # The renderer wraps each textured primitive in a singleton collection.
    # Both the frozen 2x2 surface and the live 7x6 surface must therefore keep
    # a matching number of triangle corners and UV coordinates.
    for surface in (historical, source):
        primitive = surface.get_render_primitives()
        assert (
            primitive.corners.shape[-2]
            == primitive.uvs.shape[-3] * primitive.uvs.shape[-2]
        )
        type(primitive)(triangle_collection=[primitive])


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

    expected = Color.add_defaults(pixels).transpose(-3, -2).flip(-2).reshape(1, -1, 5)
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


def test_textured_primitive_visible_when_only_its_texture_is_opaque():
    """A cut-out image must not be culled by its own transparent corners.

    A textured quad is two triangles whose corners are the image's corners, and
    every cut-out PNG is transparent there. Deciding visibility from the corner
    colours alone dropped whichever triangle had no opaque corner, chopping the
    picture along the quad's diagonal; the texture's alpha decides instead.
    """
    pixels = torch.zeros((4, 4, 4))
    pixels[1:3, 1:3, :] = 1.0  # opaque only in the middle

    image = ImageMob(pixels, add_to_scene=False)
    primitive = image.get_render_primitives()
    primitive._stash_texture_maps()
    corners = primitive.corners.float()
    primitive._pack_frame_visibility(
        corners.amin(-2), corners.amax(-2), primitive.colors.float(), "test"
    )

    assert float(primitive.colors[..., -1].amax()) == 0.0
    assert bool((primitive._rt_frame_hi >= primitive._rt_frame_lo).all())


@pytest.mark.parametrize("channels", [3, 4, 5])
def test_set_color_by_function_accepts_rgb_rgba_and_glow_colours(channels):
    """The documented three- and four-channel returns must not raise.

    Colours are stored five-channel (RGB + glow + alpha). A caller writing the
    natural ``torch.cat((r, g, b, a), -1)`` used to hit a bare broadcast error
    from the timeline buffer -- "size of tensor a (4) must match ... (5)" --
    even though the docstring advertised RGBA.
    """
    surface = Surface(grid_width=3, grid_height=3, add_to_scene=False)

    surface.set_color_by_function(
        lambda uv: torch.full((*uv.shape[:-1], channels), 0.5)
    )

    assert surface.grid.color.shape[-1] == 5
    stored = surface.grid.color
    assert torch.allclose(stored[..., :3], torch.full_like(stored[..., :3], 0.5))
    if channels == 3:
        # alpha defaults to opaque, glow to none.
        assert torch.allclose(stored[..., 3], torch.zeros_like(stored[..., 3]))
        assert torch.allclose(stored[..., 4], torch.ones_like(stored[..., 4]))
