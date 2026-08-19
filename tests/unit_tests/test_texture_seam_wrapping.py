"""A closed surface's texture has to wrap at its seam, not clamp.

The renderer addresses a ``[W, H]`` map as ``u * (W - 1)`` and clamps (see
``_sample_texture`` in ``raytrace_kernels_taichi.py`` and ``_sample_tex_vec5``
in ``wavefront_kernels_taichi.py``), so texel 0 sits at ``u == 0`` and texel
``W-1`` at ``u == 1``. On a :class:`~.Sphere` those are the same meridian, so
the map arrived stretched by ``W / (W - 1)`` and cut by a hard seam wherever
column 0 disagreed with column ``W-1``.

:func:`wrap_pad_texture` repeats column 0 at column ``W``, which puts texel
``i`` at ``u == i / W`` and lets the clamping sampler interpolate across the
seam. These tests pin the padding, the closure predicate that selects it, and
the sampling identity that makes it the right amount of padding -- with a
reference implementation of the kernel's own addressing, because that
convention is what the padding is calibrated against.

Feature tests for the texture path: unmarked, so outside the fast suite.
"""

import torch

from algan.mobs.shapes_3d import Cylinder, Sphere, Torus
from algan.mobs.surfaces.surface import (
    Surface,
    surface_closed_axes,
    wrap_pad_texture,
)
from algan.scene_manager import SceneManager


def _checker(width, height):
    """A ``[W, H, 5]`` checkerboard, opaque, red/blue by parity."""
    texture = torch.zeros(width, height, 5)
    parity = (torch.arange(width).view(-1, 1) + torch.arange(height).view(1, -1)) % 2
    texture[..., 0] = parity
    texture[..., 2] = 1 - parity
    texture[..., 4] = 1.0
    return texture


def _sample(texture, u, v):
    """The renderer's own bilinear texel addressing, in plain torch.

    Mirrors ``_sample_texture``: ``u`` addresses the first axis as
    ``u * (W - 1)``, ``v`` the second as ``v * (H - 1)``, both clamped, then a
    bilinear blend of the four surrounding texels.
    """
    width, height = texture.shape[-3], texture.shape[-2]
    px = min(max(u * (width - 1), 0.0), width - 1.0)
    py = min(max(v * (height - 1), 0.0), height - 1.0)
    x0, y0 = int(px), int(py)
    x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
    xr, yr = px - x0, py - y0
    return (
        texture[x0, y0] * (1 - xr) * (1 - yr)
        + texture[x1, y0] * xr * (1 - yr)
        + texture[x0, y1] * (1 - xr) * yr
        + texture[x1, y1] * xr * yr
    )


def test_a_sphere_grid_closes_on_u_and_a_plane_closes_on_neither():
    SceneManager.reset()
    sphere = Sphere(radius=1.5)
    assert surface_closed_axes(
        sphere._reshape_grid_for_render(sphere.grid.location)
    ) == (True, False), "a sphere's u wraps a full turn; its v runs pole to pole"

    plane = Surface(grid_width=6, grid_height=6)
    assert surface_closed_axes(plane._reshape_grid_for_render(plane.grid.location)) == (
        False,
        False,
    ), "a flat plane closes on neither axis"


def test_a_torus_closes_on_both_axes_and_a_cylinder_only_on_u():
    SceneManager.reset()
    torus = Torus()
    assert surface_closed_axes(torus._reshape_grid_for_render(torus.grid.location)) == (
        True,
        True,
    )

    cylinder = Cylinder(radius=1, height=2)
    assert surface_closed_axes(
        cylinder._reshape_grid_for_render(cylinder.grid.location)
    ) == (True, False), "a cylinder's two rim rows sit at different heights"


def test_a_single_sample_axis_is_not_mistaken_for_a_wraparound():
    """One column is its own first and last, which the coincidence test cannot
    distinguish from a surface that closes.
    """
    # W = 1: four distinct samples along v, one column along u.
    line = torch.arange(4.0).view(1, 4, 1).expand(1, 4, 3)
    assert surface_closed_axes(line) == (False, False)


def test_padding_repeats_the_first_row_or_column_of_a_closed_axis():
    texture = _checker(8, 6).unsqueeze(0)

    padded_u = wrap_pad_texture(texture, (True, False))
    assert padded_u.shape == (1, 9, 6, 5)
    assert torch.equal(padded_u[:, :8], texture), "the original texels must survive"
    assert torch.equal(padded_u[:, 8], texture[:, 0]), "column W must repeat column 0"

    padded_both = wrap_pad_texture(texture, (True, True))
    assert padded_both.shape == (1, 9, 7, 5)
    assert torch.equal(padded_both[:, :, 6], padded_both[:, :, 0])


def test_an_open_surface_keeps_the_very_tensor_it_was_given():
    """Not merely equal: unchanged, so nothing textured that does not close
    pays a copy or moves a pixel.
    """
    texture = _checker(8, 6).unsqueeze(0)
    assert wrap_pad_texture(texture, (False, False)) is texture
    assert wrap_pad_texture(None, (True, True)) is None


def test_padding_puts_every_texel_an_equal_step_apart_around_the_seam():
    """The point of one extra column: ``u = 0`` and ``u = 1`` land on one texel.

    Texel ``i`` of a padded W-column map sits at ``u == i / W``, so the wrap
    cell spans the same ``1 / W`` as every interior cell -- which is what makes
    the pattern continuous where the surface closes.
    """
    width = 8
    # v = 0.5 of a 5-row map addresses row 2 exactly, so nothing below blends
    # two rows and hides a column difference behind their average.
    padded = wrap_pad_texture(_checker(width, 5), (True, False))

    assert torch.equal(_sample(padded, 0.0, 0.5), _sample(padded, 1.0, 0.5)), (
        "u = 0 and u = 1 are the same meridian and must sample the same colour"
    )
    for i in range(width):
        assert torch.equal(_sample(padded, i / width, 0.5), padded[i, 2]), (
            f"texel {i} should land exactly on u = {i}/{width}"
        )
    # Halfway across the wrap cell blends the last column into the first, the
    # same blend an interior cell gets.
    seam = _sample(padded, (width - 0.5) / width, 0.5)
    interior = _sample(padded, (width - 1.5) / width, 0.5)
    assert torch.allclose(seam, (padded[width - 1, 2] + padded[0, 2]) / 2)
    assert torch.allclose(interior, (padded[width - 2, 2] + padded[width - 1, 2]) / 2)


def test_an_unpadded_map_would_seam_a_sphere():
    """The bug this fixes, stated as a measurement on the old addressing."""
    width = 8
    unpadded = _checker(width, 5)
    left = _sample(unpadded, 0.0, 0.5)
    right = _sample(unpadded, 1.0, 0.5)
    assert not torch.equal(left, right), (
        "column 0 and column W-1 differ, and both sat on the seam meridian"
    )


def test_a_spheres_render_primitive_carries_a_wrapped_colour_map():
    SceneManager.reset()
    sphere = Sphere(radius=1.5, color_texture=_checker(8, 8)).spawn()
    texture_map = sphere.get_render_primitives().texture_map

    assert texture_map.shape[-3] == 9, "the u axis must gain its wrap column"
    assert texture_map.shape[-2] == 8, "v runs pole to pole and must not wrap"
    assert torch.equal(texture_map[..., 0, :, :], texture_map[..., -1, :, :])


def test_a_planes_render_primitive_carries_the_map_unchanged():
    SceneManager.reset()
    plane = Surface(color_texture=_checker(8, 8), grid_width=6, grid_height=6).spawn()
    texture_map = plane.get_render_primitives().texture_map

    assert texture_map.shape[-3:] == (8, 8, 5)


def test_material_and_normal_maps_wrap_with_the_colour_map():
    """They are sampled against the same uvs, so they seam the same way."""
    SceneManager.reset()
    sphere = Sphere(
        radius=1.5,
        roughness_texture=torch.rand(8, 6, 1),
        normal_texture=torch.rand(8, 6, 3),
    ).spawn()
    primitive = sphere.get_render_primitives()

    assert primitive.material_texture_map.shape[-3:-1] == (9, 6)
    assert torch.equal(
        primitive.material_texture_map[..., 0, :, :],
        primitive.material_texture_map[..., -1, :, :],
    )
    assert primitive.normal_texture_map.shape[-3:-1] == (9, 6)
    assert torch.equal(
        primitive.normal_texture_map[..., 0, :, :],
        primitive.normal_texture_map[..., -1, :, :],
    )


def test_a_wrapped_surface_prices_its_extra_copy_into_the_batch_sizer():
    """A colour texture dominates a textured surface's render memory, and the
    wrap pad is one more live copy of it while the premultiply clones off it.
    """
    SceneManager.reset()
    sphere = Sphere(radius=1.5, color_texture=_checker(32, 32)).spawn()

    assert sphere._color_texture_bytes_per_timestep() == 32 * 32 * 5 * 4 * 2, (
        "before a primitive build nothing knows the surface closes"
    )
    sphere.get_render_primitives()
    assert sphere._color_texture_bytes_per_timestep() == 32 * 32 * 5 * 4 * 3
