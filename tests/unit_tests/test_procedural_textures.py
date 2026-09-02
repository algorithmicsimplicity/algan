"""The procedural texture generators, and the pattern they replaced.

``Surface(checkered_color=...)`` used to paint alternating *vertices*, which is
wrong twice over: the grid is laid out row-major, so on an even-width grid every
row starts on the same parity and the "checkerboard" came out as stripes; and
the pattern's resolution was the geometry's, so a flat surface -- which needs no
tessellation at all -- got no pattern. The replacement is a texture map, and
these tests pin both halves of that: the generated image really is a
checkerboard, and it is independent of the mesh under it.
"""

from __future__ import annotations

import pytest
import torch

from algan.animation_timeline.animation_contexts import Off
from algan.constants.color import BLACK, BLUE, GREEN, RED, WHITE
from algan.errors import AlganConfigurationError
from algan.mobs.shapes_3d import Sphere
from algan.mobs.surfaces.procedural_textures import (
    get_bricks,
    get_checkerboard,
    get_gradient,
    get_grid_lines,
    get_noise,
    get_polka_dots,
    get_radial_gradient,
    get_stripes,
)
from algan.mobs.surfaces.surface import Surface
from algan.scene import Scene
from algan.scene_manager import SceneManager

GENERATORS = [
    get_bricks,
    get_checkerboard,
    get_gradient,
    get_grid_lines,
    get_noise,
    get_polka_dots,
    get_radial_gradient,
    get_stripes,
]


def _plane(uv):
    return torch.cat(((uv - 0.5) * 2, torch.zeros_like(uv[..., :1])), -1)


@pytest.mark.parametrize("generator", GENERATORS, ids=lambda f: f.__name__)
def test_every_generator_returns_a_surface_ready_image(generator):
    texture = generator()
    assert texture.dim() == 3
    assert texture.shape[-1] == 5, "five channels: R, G, B, glow, alpha"
    assert texture.dtype == torch.float32
    assert torch.isfinite(texture).all()
    assert float(texture[..., :3].min()) >= 0
    assert float(texture[..., 4].max()) <= 1


def test_checkerboard_alternates_along_both_axes():
    """REGRESSION. The vertex-based checker alternated along one flattened
    index, so on an even-width grid it produced vertical stripes rather than a
    board. A texel's color here depends on both of its axes.
    """
    texture = get_checkerboard((RED, BLACK), resolution=4, texture_resolution=8)
    red = texture[..., 0] > 0.5
    # 8 texels over 4 squares: two texels per square, alternating in both axes.
    for u in range(8):
        for v in range(8):
            assert bool(red[u, v]) == ((u // 2 + v // 2) % 2 == 0), (u, v)


def test_checkerboard_detail_does_not_come_from_the_mesh():
    """The other half of the old bug: a flat surface got no pattern at all,
    because the checker lived on the (minimal) vertex grid.
    """
    SceneManager.reset()
    with Scene() as scene, Off():
        plane = Surface(
            _plane,
            grid_width=2,
            grid_height=2,
            color_texture=get_checkerboard((RED, WHITE), 6),
            add_to_scene=False,
            scene=scene,
        )
    # Four vertices, six squares per axis: the pattern cannot have come from
    # the mesh, and the renderer shades from the map rather than from the
    # corners the grid samples for it.
    assert plane.grid_width == plane.grid_height == 2
    assert plane._has_color_texture
    assert len(set(map(tuple, plane.color_texture.reshape(-1, 5).tolist()))) == 2


def test_three_colors_give_diagonals_not_a_board():
    texture = get_checkerboard((RED, GREEN, BLUE), resolution=3, texture_resolution=3)
    # (u + v) % 3, so the diagonal u + v == 3 is one color throughout.
    assert torch.equal(texture[0, 0], texture[1, 2])
    assert torch.equal(texture[0, 0], texture[2, 1])
    assert not torch.equal(texture[0, 0], texture[0, 1])


def test_a_single_color_fills_the_whole_map():
    texture = get_checkerboard(RED, resolution=4)
    assert torch.equal(texture, texture[:1, :1].expand_as(texture))


def test_pattern_counts_and_texture_size_are_separate_knobs():
    assert tuple(get_checkerboard(resolution=(4, 2)).shape) == (128, 64, 5)
    assert tuple(get_checkerboard(resolution=4, texture_resolution=(16, 32)).shape) == (
        16,
        32,
        5,
    )


def test_gradient_runs_from_the_first_color_to_the_last():
    texture = get_gradient((BLACK, WHITE))
    assert float(texture[0, 0, 0]) < 0.05
    assert float(texture[-1, 0, 0]) > 0.95
    # Along u by default, so v changes nothing.
    assert torch.equal(texture[:, 0], texture[:, -1])
    rotated = get_gradient((BLACK, WHITE), angle=90)
    assert torch.equal(rotated[0], rotated[-1])


def test_radial_gradient_is_brightest_at_its_center():
    texture = get_radial_gradient((WHITE, BLACK))
    middle = texture.shape[0] // 2, texture.shape[1] // 2
    assert float(texture[middle][0]) > float(texture[0, 0][0])


def test_noise_is_reproducible_from_a_seed_and_fresh_without_one():
    assert torch.equal(get_noise(seed=11), get_noise(seed=11))
    assert not torch.equal(get_noise(seed=11), get_noise(seed=12))
    assert not torch.equal(get_noise(), get_noise())


def test_stripes_band_across_one_axis():
    texture = get_stripes((RED, BLACK), resolution=4, texture_resolution=8)
    # Angle 0: stripes run along v, so every column is uniform.
    assert torch.equal(texture[:, 0], texture[:, -1])
    assert not torch.equal(texture[0, 0], texture[2, 0])


def test_dots_and_grid_lines_sit_where_they_are_asked_to():
    dots = get_polka_dots(WHITE, BLACK, resolution=4, radius=0.3)
    cell = dots.shape[0] // 4
    assert float(dots[cell // 2, cell // 2, 0]) > 0.9, "a dot is centred in its cell"
    assert float(dots[cell - 1, cell - 1, 0]) < 0.1, "and its corners are background"

    lines = get_grid_lines(WHITE, BLACK, resolution=4, line_width=0.2)
    assert float(lines[0, 0, 0]) > 0.9, "lines sit on the cell boundaries"
    assert float(lines[cell // 2, cell // 2, 0]) < 0.1


def test_bricks_offset_alternate_courses():
    texture = get_bricks(RED, resolution=(4, 4), mortar_color=BLACK)
    height = texture.shape[1] // 4
    mortar = texture[..., 0] < 0.5
    # The vertical joints of one course fall inside the bricks of the next.
    first = mortar[:, height // 2]
    second = mortar[:, height + height // 2]
    assert not torch.equal(first, second)


def test_bad_arguments_are_rejected_with_an_algan_error():
    with pytest.raises(AlganConfigurationError, match="at least one color"):
        get_checkerboard([])
    with pytest.raises(AlganConfigurationError, match="at least 1"):
        get_checkerboard(RED, resolution=0)
    with pytest.raises(AlganConfigurationError, match="u, v"):
        get_checkerboard(RED, resolution=(2, 2, 2))
    with pytest.raises(AlganConfigurationError, match="at least 2 texels"):
        get_checkerboard(RED, texture_resolution=1)
    with pytest.raises(AlganConfigurationError, match="radius"):
        get_radial_gradient(radius=0)


def test_a_generated_map_is_accepted_as_a_color_texture():
    SceneManager.reset()
    with Scene() as scene, Off():
        sphere = Sphere(
            color_texture=get_checkerboard((RED, WHITE), 6, texture_resolution=32),
            add_to_scene=False,
            scene=scene,
        )
    assert (sphere.texture_width, sphere.texture_height) == (32, 32)
    assert sphere.color_texture.shape == (32, 32, 5)
