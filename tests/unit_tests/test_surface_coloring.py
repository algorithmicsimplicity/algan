"""``Surface``'s per-vertex painters: a checkerboard, a height map, a shape.

``set_checkerboard_colors`` and ``set_color_by_axis`` paint the grid's own
vertices, so a surface's grid is the thing to read back: its rows run u-major,
row ``u * grid_height + v`` being the vertex at ``(u, v)``.
``set_location_by_function`` is the primitive that shapes a surface and that
``set_shape_to`` morphs through.

A feature test, so it sits outside the fast suite: it only breaks when these
methods themselves change.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from algan import BLUE, GREEN, RED, RIGHT, YELLOW, Off, Scene, Sphere, Surface
from algan.constants.color import to_color
from algan.errors import InvalidColorError

#: Frames across a one-second animation starting at time zero.
TIMES = torch.tensor([0.0, 0.5, 1.0])

#: Every single-color spelling a mob's ``color`` takes, as (red, blue) pairs.
SPELLINGS = [
    pytest.param((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), id="rgb-tuple"),
    pytest.param("#FF0000", "#0000FF", id="hex-string"),
    pytest.param("red", "blue", id="css-name"),
    pytest.param(0xFF0000, 0x0000FF, id="hex-int"),
]


@pytest.fixture
def scene():
    with Scene() as active:
        yield active


def _rgb(color) -> torch.Tensor:
    return to_color(color).reshape(-1)[:3]


def _vertex_rgb(surface) -> torch.Tensor:
    """The grid's colors as ``[u, v, rgb]``."""
    return surface.grid.color.reshape(surface.grid_width, surface.grid_height, -1)[
        ..., :3
    ]


def _ramp(uv):
    """A plane in x/y whose height runs from -1 to 1 along u."""
    return torch.cat((uv[..., :1], uv[..., 1:], uv[..., :1] * 2 - 1), -1)


def _ramp_surface(**kwargs):
    # Five columns along u put the heights at -1, -0.5, 0, 0.5 and 1.
    return Surface(_ramp, grid_width=5, grid_height=3, **kwargs)


def _flat(uv):
    return torch.cat((uv, torch.zeros_like(uv[..., :1])), -1)


def _lifted(uv):
    return torch.cat((uv, torch.ones_like(uv[..., :1])), -1)


def test_two_colors_alternate_along_both_grid_axes(scene):
    with Off():
        sphere = Sphere(grid_width=6, grid_height=5).set_checkerboard_colors(
            BLUE, YELLOW
        )
    rgb = _vertex_rgb(sphere)

    assert torch.allclose(rgb[0, 0], _rgb(BLUE))
    assert torch.allclose(rgb[0, 1], _rgb(YELLOW))
    assert torch.allclose(rgb[1, 0], _rgb(YELLOW))
    # A checkerboard: every step along u or along v changes the color, and
    # every diagonal step keeps it.
    assert (rgb[1:] != rgb[:-1]).any(-1).all()
    assert (rgb[:, 1:] != rgb[:, :-1]).any(-1).all()
    assert torch.equal(rgb[1:, 1:], rgb[:-1, :-1])


def test_three_colors_run_in_diagonal_stripes(scene):
    with Off():
        sphere = Sphere(grid_width=6, grid_height=5).set_checkerboard_colors(
            RED, GREEN, BLUE
        )
    rgb = _vertex_rgb(sphere)

    palette = [_rgb(RED), _rgb(GREEN), _rgb(BLUE)]
    for u in range(6):
        for v in range(5):
            assert torch.allclose(rgb[u, v], palette[(u + v) % 3])


@pytest.mark.parametrize(("red", "blue"), SPELLINGS)
def test_a_checkerboard_takes_every_color_spelling(scene, red, blue):
    """REGRESSION. Each color was reshaped raw to the grid's five channels,
    so only a five-channel ``Color`` got through: an RGB tuple, a hex string
    or a CSS name -- all of which ``Square(color=...)`` takes, and which the
    docstring promises as "anything ``Color()`` accepts" -- raised a shape or
    dtype error.
    """
    with Off():
        sphere = Sphere(grid_width=4, grid_height=4).set_checkerboard_colors(red, blue)
    rgb = _vertex_rgb(sphere)

    assert torch.allclose(rgb[0, 0], _rgb(red))
    assert torch.allclose(rgb[0, 1], _rgb(blue))
    assert sphere.grid.color.shape[-1] == 5


def test_a_checkerboard_can_set_the_opacity_alongside(scene):
    with Off():
        sphere = Sphere(grid_width=4, grid_height=4).set_checkerboard_colors(
            BLUE, YELLOW, opacity=0.25
        )
    assert torch.allclose(
        sphere.grid.opacity, torch.full_like(sphere.grid.opacity, 0.25)
    )


def test_a_checkerboard_without_colors_leaves_the_surface_alone(scene):
    sphere = Sphere(grid_width=4, grid_height=4)
    before = sphere.grid.color.clone()

    assert sphere.set_checkerboard_colors() is sphere
    assert torch.equal(sphere.grid.color, before)


def test_a_checkerboard_rejects_a_name_that_is_no_color(scene):
    sphere = Sphere(grid_width=4, grid_height=4)
    with pytest.raises(InvalidColorError, match="not-a-color"):
        sphere.set_checkerboard_colors(BLUE, "not-a-color")


def test_a_recorded_checkerboard_cross_fades_in(scene):
    with Off():
        sphere = Sphere(grid_width=4, grid_height=4).spawn()
    before = sphere.grid.color.clone()
    sphere.set_checkerboard_colors(BLUE, YELLOW)
    after = sphere.grid.color.clone()

    scene.timeline_manager.set_state_to_times(TIMES)
    colors = sphere.grid.color
    assert torch.allclose(colors[0], before[0])
    assert torch.allclose(colors[1], (before[0] + after[0]) / 2, atol=1e-4)
    assert torch.allclose(colors[-1], after[0])


def test_a_plain_scale_spreads_over_the_surfaces_own_height(scene):
    with Off():
        surface = _ramp_surface().set_color_by_axis(colorscale=[RED, BLUE])
    rgb = _vertex_rgb(surface)

    red, blue = _rgb(RED), _rgb(BLUE)
    for u, height in enumerate([-1.0, -0.5, 0.0, 0.5, 1.0]):
        expected = red + (blue - red) * (height + 1) / 2
        assert torch.allclose(rgb[u], expected.expand(3, -1), atol=1e-6)


def test_color_value_pairs_pin_each_color_to_a_height(scene):
    with Off():
        surface = _ramp_surface().set_color_by_axis(
            colorscale=[(RED, -0.5), (YELLOW, 0.0), (BLUE, 0.5)]
        )
    rgb = _vertex_rgb(surface)[:, 0]

    # Below the first pivot and above the last the scale is clamped.
    assert torch.allclose(rgb[0], _rgb(RED))
    assert torch.allclose(rgb[1], _rgb(RED))
    assert torch.allclose(rgb[2], _rgb(YELLOW))
    assert torch.allclose(rgb[3], _rgb(BLUE))
    assert torch.allclose(rgb[4], _rgb(BLUE))


def test_color_value_pairs_may_come_in_any_order(scene):
    """REGRESSION. The pivots went to ``searchsorted`` as written, which
    needs them ascending: ``[(BLUE, 1), (RED, -1)]`` painted every vertex red
    instead of the same gradient as ``[(RED, -1), (BLUE, 1)]``.
    """
    with Off():
        ascending = _ramp_surface().set_color_by_axis(
            colorscale=[(RED, -1.0), (BLUE, 1.0)]
        )
        descending = _ramp_surface().set_color_by_axis(
            colorscale=[(BLUE, 1.0), (RED, -1.0)]
        )
    assert torch.allclose(descending.grid.color, ascending.grid.color)
    assert not torch.allclose(_vertex_rgb(descending)[0], _vertex_rgb(descending)[-1])


@pytest.mark.parametrize(("red", "blue"), SPELLINGS)
def test_a_height_map_takes_every_color_spelling(scene, red, blue):
    """REGRESSION. The same raw reshape as the checkerboard, in both the
    plain-sequence and the ``(color, value)`` forms of the scale.
    """
    with Off():
        plain = _ramp_surface().set_color_by_axis(colorscale=[red, blue])
        pinned = _ramp_surface().set_color_by_axis(
            colorscale=[(red, -1.0), (blue, 1.0)]
        )
    for surface in (plain, pinned):
        rgb = _vertex_rgb(surface)
        assert torch.allclose(rgb[0], _rgb(red).expand(3, -1))
        assert torch.allclose(rgb[-1], _rgb(blue).expand(3, -1))


def test_axes_supply_the_range_the_scale_spreads_over(scene):
    # A plot's axes carry their ranges as (min, max, step).
    axes = SimpleNamespace(z_range=(-2, 2, 1))
    with Off():
        surface = _ramp_surface().set_color_by_axis(axes, [RED, BLUE])
    rgb = _vertex_rgb(surface)[:, 0]

    red, blue = _rgb(RED), _rgb(BLUE)
    # Heights -1..1 fill only the middle half of a -2..2 scale.
    assert torch.allclose(rgb[0], red + (blue - red) * 0.25, atol=1e-6)
    assert torch.allclose(rgb[2], red + (blue - red) * 0.5, atol=1e-6)
    assert torch.allclose(rgb[-1], red + (blue - red) * 0.75, atol=1e-6)


def test_axis_picks_the_coordinate_that_is_read(scene):
    with Off():
        surface = _ramp_surface().set_color_by_axis(colorscale=[RED, BLUE], axis=1)
    rgb = _vertex_rgb(surface)

    # y runs along v, so the colors change across v and hold along u.
    assert torch.allclose(rgb[:, 0], _rgb(RED).expand(5, -1))
    assert torch.allclose(rgb[:, -1], _rgb(BLUE).expand(5, -1))
    assert torch.allclose(rgb[1:], rgb[:-1])


def test_colors_is_another_name_for_colorscale(scene):
    with Off():
        by_scale = _ramp_surface().set_color_by_axis(colorscale=[RED, BLUE])
        by_colors = _ramp_surface().set_color_by_axis(colors=[RED, BLUE])
    assert torch.allclose(by_colors.grid.color, by_scale.grid.color)


def test_an_unknown_keyword_is_rejected_by_name(scene):
    surface = _ramp_surface()
    with pytest.raises(ValueError, match="cmap"):
        surface.set_color_by_axis(colorscale=[RED, BLUE], cmap="viridis")


@pytest.mark.parametrize("colorscale", [None, []], ids=["none", "empty"])
def test_no_scale_leaves_the_surface_alone(scene, colorscale):
    surface = _ramp_surface()
    before = surface.grid.color.clone()

    assert surface.set_color_by_axis(colorscale=colorscale) is surface
    assert torch.equal(surface.grid.color, before)


def test_a_shape_function_gives_offsets_from_the_surfaces_location(scene):
    with Off():
        surface = Surface(_flat, grid_width=3, grid_height=3).move_to(RIGHT * 2)
        assert surface.set_location_by_function(_lifted) is surface

    grid = surface.grid.location.reshape(3, 3, 3)
    uv = surface.get_base_grid()
    assert torch.allclose(grid[..., 0], uv[..., 0] + 2)
    assert torch.allclose(grid[..., 1], uv[..., 1])
    assert torch.allclose(grid[..., 2], torch.ones(3, 3))
    assert surface.coord_function_active is _lifted


def test_a_recorded_shape_change_morphs_through_the_frames_between(scene):
    with Off():
        surface = Surface(_flat, grid_width=3, grid_height=3).spawn()
    surface.set_location_by_function(_lifted)

    scene.timeline_manager.set_state_to_times(TIMES)
    heights = surface.grid.location[..., 2]
    assert torch.allclose(heights[0], torch.zeros(9))
    assert torch.allclose(heights[1], torch.full((9,), 0.5), atol=1e-4)
    assert torch.allclose(heights[-1], torch.ones(9))
