"""The CPU surface build's fast paths must reproduce the reference ops exactly.

``compute_grid_vertex_normals`` crosses the grid's sides elementwise where
that is bit-identical to ``torch.linalg.cross`` on this build, and
``grid_to_triangle_vertices`` builds an unwelded grid's triangle soup from
six shifted views instead of an index gather. Neither may move a bit.
"""

from __future__ import annotations

import math

import pytest
import torch

from algan.mobs.surfaces import surface as surface_module
from algan.mobs.surfaces.surface import (
    compute_grid_vertex_normals,
    get_grid_to_triangle_indices,
    grid_to_triangle_vertices,
)


def _sphere_grid(frames, width, height, seed=0):
    generator = torch.Generator().manual_seed(seed)
    u = torch.linspace(0, 1, width).view(width, 1)
    v = torch.linspace(0, 1, height).view(1, height)
    theta, phi = u * 2 * math.pi, v * math.pi
    grid = torch.stack(
        (
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(phi).expand(width, height),
        ),
        -1,
    )
    rotation = torch.linalg.qr(torch.randn(frames, 3, 3, generator=generator))[0]
    offset = torch.randn(frames, 1, 1, 3, generator=generator)
    return (
        grid.view(1, width, height, 3) @ rotation.transpose(-1, -2).unsqueeze(1)
    ) * (3.7) + offset


GRIDS = [
    _sphere_grid(4, 33, 17),
    _sphere_grid(2, 61, 61, seed=1),
    torch.randn(2, 3, 5, 7, 3, generator=torch.Generator().manual_seed(2)),
    torch.randn(1, 2, 2, 3, generator=torch.Generator().manual_seed(3)),
]


def _bits(tensor):
    return tensor.contiguous().view(torch.int32)


def _bits_up_to_nan(tensor):
    """``_bits`` with every NaN replaced by one canonical NaN.

    IEEE 754 leaves the sign and payload of an operation's NaN result
    unspecified, and they do differ by instruction sequence: on arm64
    ``torch.linalg.cross`` returns a NaN with the sign bit set where the
    elementwise form returns a positive one. Which NaN is not a value the
    renderer can observe; where the NaNs are, and every other bit, still are.
    """
    return _bits(torch.where(tensor.isnan(), math.nan, tensor))


@pytest.mark.parametrize("grid", GRIDS)
def test_elementwise_cross_is_bit_identical(grid):
    sides = [
        surface_module._wrapped_difference(grid, axis, shift)
        for axis, shift in ((-3, 1), (-2, 1), (-3, -1), (-2, -1))
    ]
    for a, b in zip(sides, sides[1:] + sides[:1]):
        assert torch.equal(
            _bits(surface_module._grid_cross(a, b)),
            _bits(torch.linalg.cross(a, b, dim=-1)),
        )


def test_elementwise_cross_keeps_special_values():
    a = torch.tensor(
        [[math.nan, 1.0, -0.0], [math.inf, 0.0, 2.0], [-0.0, -0.0, 0.0]]
    ).repeat(4, 1)
    b = torch.tensor([[1.0, math.inf, 3.0], [0.0, -0.0, 1.0], [2.0, -0.0, 5.0]]).repeat(
        4, 1
    )
    assert torch.equal(
        _bits_up_to_nan(surface_module._elementwise_cross(a, b)),
        _bits_up_to_nan(torch.linalg.cross(a, b, dim=-1)),
    )


@pytest.mark.parametrize("grid", GRIDS)
def test_vertex_normals_match_the_reference_cross(grid, monkeypatch):
    fast = compute_grid_vertex_normals(grid)
    monkeypatch.setattr(surface_module, "_elementwise_cross_matches", lambda _: False)
    reference = compute_grid_vertex_normals(grid)
    assert torch.equal(_bits(fast), _bits(reference))


@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("flip", [False, True])
def test_unwelded_soup_matches_the_index_gather(grid, flip):
    if flip:
        # _reshape_grid_for_render hands over a flipped (copied) grid.
        grid = grid.flip(-2)
    width, height = grid.shape[-3], grid.shape[-2]
    indices = get_grid_to_triangle_indices(width, height, grid.device)
    reference = grid.reshape(*grid.shape[:-3], width * height, grid.shape[-1])[
        ..., indices, :
    ]
    assert torch.equal(grid_to_triangle_vertices(grid), reference)
    colors = torch.rand(*grid.shape[:-1], 5)
    reference = colors.reshape(*colors.shape[:-3], width * height, 5)[..., indices, :]
    assert torch.equal(grid_to_triangle_vertices(colors), reference)
