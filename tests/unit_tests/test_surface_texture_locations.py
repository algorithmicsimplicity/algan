"""Where a texel sits in the world, for textures written in terms of space.

A texture is addressed in ``(u, v)``, so a map whose content depends on world
position -- "colour everything above the equator" -- has no natural way to be
written. :meth:`~.Surface.get_texture_locations` supplies the missing half: the
world position of every texel, laid out like the map itself.

The position it must report is the one the renderer will draw the texel at, and
that is *not* ``coord_function(u, v)``. The kernel interpolates the triangle
corners' UVs barycentrically and the geometry beneath them is the logical PN
patch, so a texel lands on that patch at the barycentric coordinate its UV
resolves to. On a stock ``Sphere`` the two answers differ tangentially by ~12%
of a grid cell -- invisible on a small map, a scalloped boundary on a large one.
Reading the mesh instead also means the answer survives a grid the coordinate
function no longer describes, which is most of why it is worth having.

These tests pin that: the reported positions sit on the drawn surface, agree
with a reference implementation built from the renderer's own triangle list,
honour the wrap-padding convention on closed axes, and track a grid written by
hand.

Feature tests for the texture path: unmarked, so outside the fast suite.
"""

import pytest
import torch

from algan.constants.spatial import RIGHT
from algan.mobs.shapes_3d import Sphere, Torus
from algan.mobs.surfaces import surface as surface_module
from algan.mobs.surfaces.surface import (
    Surface,
    compute_grid_vertex_normals,
    grid_to_triangle_vertices,
)
from algan.rendering.logical_pn import (
    evaluate_logical_pn,
    logical_pn_control_points,
)
from algan.scene_manager import SceneManager


def _mesh(surface):
    """The surface's current vertex grid, ``[W, H, 3]``."""
    return surface._reshape_grid_for_render(surface.grid.location)[0]


def _renderer_position(surface, u, v):
    """Where the renderer puts UV ``(u, v)``, from its own triangle list.

    The slow, obvious implementation of what ``get_texture_locations`` computes
    in closed form: gather the triangles exactly as ``_build_render_primitive``
    does, find the one whose UV triangle contains the coordinate, and evaluate
    its PN patch at the barycentric weights that lands on.
    """
    grid = _mesh(surface)
    normals = compute_grid_vertex_normals(grid)
    triangle_uvs = grid_to_triangle_vertices(surface.get_base_grid()).reshape(-1, 3, 2)
    controls = logical_pn_control_points(
        grid_to_triangle_vertices(grid).reshape(-1, 3, 3),
        grid_to_triangle_vertices(normals).reshape(-1, 3, 3),
    )

    a, b, c = triangle_uvs.unbind(-2)
    edge_1, edge_2 = b - a, c - a
    point = torch.tensor([u, v]) - a
    denominator = edge_1[:, 0] * edge_2[:, 1] - edge_2[:, 0] * edge_1[:, 1]
    weight_1 = (point[:, 0] * edge_2[:, 1] - edge_2[:, 0] * point[:, 1]) / denominator
    weight_2 = (edge_1[:, 0] * point[:, 1] - point[:, 0] * edge_1[:, 1]) / denominator
    inside = (
        (weight_1 >= -1e-6) & (weight_2 >= -1e-6) & (weight_1 + weight_2 <= 1e-6 + 1)
    )
    index = int(inside.nonzero()[0])
    barycentric = torch.stack((weight_1[index], weight_2[index])).view(1, 2)
    return evaluate_logical_pn(controls[index : index + 1].unsqueeze(0), barycentric)[
        0, 0, 0
    ]


def test_every_texel_lands_on_the_surface_it_is_drawn_on():
    """A sphere's texels must all be a radius from its centre, to within the
    tolerance the mesh itself was built to.
    """
    SceneManager.reset()
    sphere = Sphere(radius=1.5)

    locations = sphere.get_texture_locations((64, 64))

    assert locations.shape == (64, 64, 3)
    error = (locations.norm(dim=-1) - 1.5).abs().max()
    assert error <= sphere.geometry_tolerance, (
        f"texels stray {error:.5f} off a surface built to {sphere.geometry_tolerance}"
    )


def test_texel_positions_match_the_renderers_own_triangles():
    """The closed-form cell lookup must agree with a brute-force search over
    the very triangles the renderer is handed.
    """
    SceneManager.reset()
    sphere = Sphere(radius=1.0)
    width, height = 32, 24
    locations = sphere.get_texture_locations((width, height))

    # u is wrap-padded (the sphere closes on it), v is not.
    for i, j in ((0, 0), (7, 5), (13, 17), (31, 23), (20, 0)):
        expected = _renderer_position(sphere, i / width, j / (height - 1))
        assert torch.allclose(locations[i, j], expected, atol=1e-5), (
            f"texel {(i, j)} disagrees with the triangle the renderer samples it on"
        )


def test_a_flat_surface_reports_its_exact_analytic_positions():
    """With no curvature to interpolate there is one right answer, and the
    coordinate function gives it.
    """
    SceneManager.reset()
    plane = Surface(grid_width=5, grid_height=5)

    locations = plane.get_texture_locations((16, 9))

    base = torch.stack(
        (
            torch.linspace(0, 1, 16).view(-1, 1).expand(-1, 9),
            torch.linspace(0, 1, 9).view(1, -1).expand(16, -1),
        ),
        -1,
    )
    assert torch.allclose(locations, plane.coord_function(base), atol=1e-6)


def test_a_closed_axis_leaves_room_for_the_wrap():
    """Texel ``W-1`` of a closed axis sits one step short of the seam, because
    the map is wrap-padded before it is sampled -- clamping instead would put
    it *on* the seam and squash the map by a texel.
    """
    SceneManager.reset()
    sphere = Sphere(radius=1.0)
    width = 8

    locations = sphere.get_texture_locations((width, 5))

    # One ring of texels, away from the poles where every column coincides.
    # The steps around it are near-uniform rather than exactly so: 8 texels do
    # not divide the grid's 19 columns, so each lands at a different point
    # within its cell. Clamping would make the closing step 0 -- nothing like
    # a percent away.
    ring = locations[:, 2]
    steps = (ring.roll(-1, 0) - ring).norm(dim=-1)
    assert steps.max() / steps.min() < 1.02, (
        "the step from the last column back to the first must match every other "
        f"step around the ring, got {steps.tolist()}"
    )

    plane = Surface(grid_width=5, grid_height=5)
    open_axis = plane.get_texture_locations((width, 5))
    assert torch.allclose(
        open_axis[0, :, 0], plane.coord_function(torch.zeros(1, 2))[..., 0], atol=1e-6
    ), "an open axis puts its first texel on the domain edge"
    assert torch.allclose(
        open_axis[-1, :, 0],
        plane.coord_function(torch.tensor([[1.0, 0.0]]))[..., 0],
        atol=1e-6,
    ), "and its last texel on the other edge"


def test_positions_come_from_the_grid_not_the_coordinate_function():
    """The case the helper exists for: a grid written by hand leaves
    ``coord_function`` describing a shape that is no longer there.
    """
    SceneManager.reset()
    sphere = Sphere(radius=1.0)
    grid = _mesh(sphere)
    sphere.grid.location = (grid * torch.tensor([1.0, 3.0, 1.0])).reshape(1, -1, 3)

    locations = sphere.get_texture_locations((32, 32))

    assert locations[..., 1].abs().max() > 2.9, (
        "the stretched grid is the shape that gets drawn, so it is the shape "
        "the texels sit on"
    )
    assert sphere.coord_function(sphere.get_base_grid())[..., 1].abs().max() < 1.1, (
        "meanwhile the coordinate function still describes the original sphere"
    )


def test_positions_follow_the_surface_when_it_moves():
    SceneManager.reset()
    sphere = Sphere(radius=1.0).spawn()
    before = sphere.get_texture_locations((16, 16))

    sphere.move(RIGHT * 3)

    after = sphere.get_texture_locations((16, 16))
    assert torch.allclose(after - before, torch.tensor([3.0, 0.0, 0.0]), atol=1e-5)


def test_ignoring_normals_reports_the_flat_triangles_that_are_drawn():
    """``ignore_normals`` leaves the renderer with no vertex normals, so it
    draws flat triangles; the reported positions have to be flat too.
    """
    SceneManager.reset()
    sphere = Sphere(radius=1.0, ignore_normals=True)
    grid = _mesh(sphere)
    width, height = sphere.grid_width, sphere.grid_height

    locations = sphere.get_texture_locations((17, 13))

    # Plain barycentric interpolation over the same two triangles per cell.
    u = torch.arange(17.0) / 17  # u closes, so it is wrap-padded
    v = torch.arange(13.0) / 12
    fu = (u * (width - 1)).view(-1, 1)
    fv = (v * (height - 1)).view(1, -1)
    i = fu.floor().clamp(0, width - 2)
    j = fv.floor().clamp(0, height - 2)
    s, t = fu - i, fv - j
    i = i.long().expand(-1, 13)
    j = j.long().expand(17, -1)
    lower = ((s + t) <= 1.0).unsqueeze(-1)
    corner_0 = torch.where(lower, grid[i, j], grid[i + 1, j])
    corner_1 = grid[i, j + 1]
    corner_2 = torch.where(lower, grid[i + 1, j], grid[i + 1, j + 1])
    weight_1 = torch.where(lower[..., 0], t, 1 - s).unsqueeze(-1)
    weight_2 = torch.where(lower[..., 0], s, s + t - 1).unsqueeze(-1)
    flat = (
        (1 - weight_1 - weight_2) * corner_0 + weight_1 * corner_1 + weight_2 * corner_2
    )

    assert torch.allclose(locations, flat, atol=1e-5)


def test_resolution_defaults_to_the_texture_then_to_the_grid():
    SceneManager.reset()
    plain = Surface(grid_width=6, grid_height=4)
    assert plain.get_texture_locations().shape == (6, 4, 3), (
        "with no texture to match, the grid resolution is the sensible default"
    )

    textured = Surface(
        grid_width=6, grid_height=4, color_texture=torch.zeros(12, 20, 5)
    )
    assert textured.get_texture_locations().shape == (12, 20, 3), (
        "a surface that has a texture should default to that texture's shape"
    )
    assert textured.get_texture_locations(9).shape == (9, 9, 3)


def test_chunking_is_invisible():
    """Large maps are resolved a block of texels at a time; the blocking must
    not be able to change an answer.
    """
    SceneManager.reset()
    torus = Torus()
    whole = torus.get_texture_locations((40, 24))

    original = surface_module._TEXTURE_LOCATION_CHUNK_TEXELS
    try:
        surface_module._TEXTURE_LOCATION_CHUNK_TEXELS = 25
        chunked = torus.get_texture_locations((40, 24))
    finally:
        surface_module._TEXTURE_LOCATION_CHUNK_TEXELS = original

    assert torch.equal(whole, chunked)


def test_a_packed_surface_says_why_it_cannot_answer():
    SceneManager.reset()
    pack = Sphere.from_batches(torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))

    with pytest.raises(ValueError, match="packed surface"):
        pack.get_texture_locations((8, 8))


def test_a_degenerate_grid_and_an_empty_resolution_are_rejected():
    SceneManager.reset()
    sliver = Surface(grid_width=1, grid_height=6)
    with pytest.raises(ValueError, match="at least 2 grid points"):
        sliver.get_texture_locations((8, 8))

    plane = Surface(grid_width=4, grid_height=4)
    with pytest.raises(ValueError, match="positive on both axes"):
        plane.get_texture_locations((8, 0))
