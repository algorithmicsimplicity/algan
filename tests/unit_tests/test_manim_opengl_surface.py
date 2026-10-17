"""``algan.manim``'s OpenGL surface classes, against Manim's own.

Manim's ``OpenGLSurface`` is ``uv_func`` sampled over ``u_range x v_range`` --
``resolution`` counts *samples* per axis, laid out u-major the way its
``init_points`` builds them -- plus a triangle index list over those samples.
Algan builds a native :class:`~algan.mobs.surfaces.surface.Surface` from the
same samples and keeps the query API. The vendored Manim subset replaces these
classes with inert stand-ins (they exist only for ``manim --renderer=opengl``),
so the expectations are written out here from Manim Community 0.21's
``opengl_surface.py``: ``init_points``, ``compute_triangle_indices``,
``get_unit_normals`` and ``OpenGLTexturedSurface.init_points``.

Feature tests: unmarked, so outside the fast suite.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import algan.manim as mn
from algan import OUT, RIGHT, SETTINGS, Cylinder, Off, Scene, Sphere, Square
from algan.mobs.surfaces.surface import Surface


@pytest.fixture
def scene():
    with Scene() as active:
        yield active


def _saddle(u, v):
    """A uv_func as Manim writes one: two floats in, one point out."""
    return np.array([u, v, u * v])


def _unit_saddle(uv):
    """The same saddle over ``[-1, 1]^2`` as a native Surface's coord_function."""
    x = uv[..., :1] * 2 - 1
    y = uv[..., 1:] * 2 - 1
    return torch.cat((x, y, x * y), -1)


def _manim_samples(uv_func, u_range, v_range, resolution):
    """``OpenGLSurface.init_points``' un-nudged block, as ``[nu, nv, 3]``."""
    nu, nv = resolution
    return np.array(
        [
            [np.asarray(uv_func(u, v), dtype=float) for v in np.linspace(*v_range, nv)]
            for u in np.linspace(*u_range, nu)
        ]
    )


def _manim_triangle_indices(nu, nv):
    """``OpenGLSurface.compute_triangle_indices`` from Manim Community 0.21."""
    index_grid = np.arange(nu * nv).reshape((nu, nv))
    indices = np.zeros(6 * (nu - 1) * (nv - 1), dtype=int)
    indices[0::6] = index_grid[:-1, :-1].flatten()  # Top left
    indices[1::6] = index_grid[+1:, :-1].flatten()  # Bottom left
    indices[2::6] = index_grid[:-1, +1:].flatten()  # Top right
    indices[3::6] = index_grid[:-1, +1:].flatten()  # Top right
    indices[4::6] = index_grid[+1:, :-1].flatten()  # Bottom left
    indices[5::6] = index_grid[+1:, +1:].flatten()  # Bottom right
    return indices


def _samples(surface):
    """The surface's vertex grid as ``[nu, nv, 3]``."""
    grid = surface.grid.location.reshape(surface.grid_width, surface.grid_height, 3)
    return grid.detach().cpu().double().numpy()


def _saddle_normals(resolution):
    """The saddle's analytic unit normals, ``cross(d/du, d/dv) = (-v, -u, 1)``."""
    u, v = torch.meshgrid(
        torch.linspace(0, 1, resolution[0]),
        torch.linspace(0, 1, resolution[1]),
        indexing="ij",
    )
    normals = torch.stack((-v, -u, torch.ones_like(u)), -1).reshape(-1, 3)
    return normals / normals.norm(dim=-1, keepdim=True)


def _picture():
    """A 2 x 3 RGBA picture whose pixels all differ."""
    pixels = np.full((2, 3, 4), 255, dtype=np.uint8)
    pixels[..., 0] = [[10, 20, 30], [40, 50, 60]]
    pixels[..., 1] = [[70, 80, 90], [100, 110, 120]]
    return pixels


@pytest.mark.parametrize("resolution", [(2, 2), (5, 4), (4, 6)])
def test_samples_uv_func_over_its_ranges_like_manims_init_points(scene, resolution):
    surface = mn.OpenGLSurface(
        _saddle,
        u_range=(-1, 2),
        v_range=(0.5, 3),
        resolution=resolution,
        add_to_scene=False,
    )

    assert (surface.grid_width, surface.grid_height) == resolution
    assert (surface.u_range, surface.v_range) == ((-1, 2), (0.5, 3))
    np.testing.assert_allclose(
        _samples(surface),
        _manim_samples(_saddle, (-1, 2), (0.5, 3), resolution),
        atol=1e-6,
    )


def _components_first(u, v):
    u, v = torch.as_tensor(u), torch.as_tensor(v)
    return torch.stack((u, v, u * v))


def _components_last(u, v):
    u, v = torch.as_tensor(u), torch.as_tensor(v)
    return torch.stack((u, v, u * v), dim=-1)


@pytest.mark.parametrize("resolution", [(3, 3), (5, 3)])
@pytest.mark.parametrize(
    "uv_func",
    [
        pytest.param(lambda u, v: (u, v, u * v), id="tuple"),
        pytest.param(_saddle, id="numpy"),
        pytest.param(_components_first, id="torch-components-first"),
        pytest.param(_components_last, id="torch-components-last"),
    ],
)
def test_a_vectorised_result_is_read_along_its_component_axis(
    scene, uv_func, resolution
):
    """REGRESSION. A components-first result -- Manim's own ``(x, y, z)``
    evaluated on the whole grid, or ``torch.stack((x, y, z))`` -- has the shape
    of a components-last one once the v axis has three samples, and was read as
    one, scrambling the surface.
    """
    surface = mn.OpenGLSurface(
        uv_func,
        u_range=(-1, 1),
        v_range=(0, 2),
        resolution=resolution,
        add_to_scene=False,
    )

    np.testing.assert_allclose(
        _samples(surface),
        _manim_samples(_saddle, (-1, 1), (0, 2), resolution),
        atol=1e-6,
    )


def test_a_uv_func_that_only_takes_floats_is_sampled_point_by_point(scene):
    """Manim only ever calls ``uv_func`` with two floats, so a function written
    with ``math`` must work even though it cannot take a whole grid.
    """

    def tube(u, v):
        return (math.cos(u), math.sin(u), v)

    surface = mn.OpenGLSurface(
        tube,
        u_range=(0, math.pi),
        v_range=(-1, 1),
        resolution=(5, 3),
        add_to_scene=False,
    )

    np.testing.assert_allclose(
        _samples(surface),
        _manim_samples(tube, (0, math.pi), (-1, 1), (5, 3)),
        atol=1e-6,
    )


def test_without_a_uv_func_the_surface_is_the_unit_uv_plane(scene):
    plane = mn.OpenGLSurface(resolution=(3, 2), add_to_scene=False)

    assert (plane.u_range, plane.v_range) == ((0, 1), (0, 1))
    assert plane.uv_func(0.25, 0.5) == (0.25, 0.5, 0.0)
    np.testing.assert_allclose(
        _samples(plane),
        _manim_samples(lambda u, v: (u, v, 0.0), (0, 1), (0, 1), (3, 2)),
        atol=1e-6,
    )

    saddle = mn.OpenGLSurface(_saddle, resolution=(2, 2), add_to_scene=False)
    np.testing.assert_array_equal(saddle.uv_func(0.5, 3.0), _saddle(0.5, 3.0))


def test_resolution_counts_samples_and_defaults_to_manims(scene):
    def samples(**kwargs):
        surface = mn.OpenGLSurface(add_to_scene=False, **kwargs)
        return surface.grid_width, surface.grid_height

    assert samples() == (101, 101)
    assert samples(resolution=4) == (4, 4)
    assert samples(resolution=(5, 3)) == (5, 3)


@pytest.mark.parametrize("resolution", [(3, 2), (4, 5)])
def test_triangle_indices_follow_manims_layout(scene, resolution):
    """REGRESSION. ``Surface.__init__`` overwrote the sample-count
    ``resolution`` with its own argument, ``None``, so ``get_triangle_indices``
    raised ``TypeError``.
    """
    surface = mn.OpenGLSurface(_saddle, resolution=resolution, add_to_scene=False)

    assert surface.resolution == resolution
    indices = surface.get_triangle_indices()
    assert indices.tolist() == _manim_triangle_indices(*resolution).tolist()
    if resolution == (3, 2):
        # Two cells, two triangles each, written out.
        assert indices.tolist() == [0, 2, 1, 1, 2, 3, 2, 4, 3, 3, 4, 5]

    # Manim's sort reorders triangles for its painter; Algan's ray tracer needs
    # no order, but either way every triangle survives.
    triangles = sorted(map(tuple, indices.reshape(-1, 3).tolist()))
    assert surface.sort_faces_back_to_front() is surface
    assert surface.sort_faces_back_to_front(OUT) is surface
    after = surface.get_triangle_indices().reshape(-1, 3).tolist()
    assert sorted(map(tuple, after)) == triangles


def test_unit_normals_follow_the_surface_when_it_moves(scene):
    """REGRESSION. The nudged points were evaluated where the surface was built
    while its points were read where it is now, so after any move or turn the
    normals crossed two unrelated vectors.

    Manim keeps the nudged points beside the points, so they travel together.
    The tolerance covers float32 finite differences over a 1e-5 nudge.
    """
    surface = mn.OpenGLSurface(_saddle, resolution=(4, 5), add_to_scene=False)
    expected = _saddle_normals((4, 5))
    torch.testing.assert_close(surface.get_unit_normals(), expected, atol=1e-2, rtol=0)

    with Off():
        surface.move(RIGHT * 2)
        surface.rotate(90, OUT)

    # A quarter turn anticlockwise about OUT, applied to row vectors.
    quarter_turn = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    torch.testing.assert_close(
        surface.get_unit_normals(), expected @ quarter_turn, atol=1e-2, rtol=0
    )
    points, du_points, dv_points = surface.get_surface_points_and_nudged_points()
    torch.testing.assert_close(points, surface.grid.location.reshape(-1, 3))
    assert (du_points - points).norm(dim=-1).max() < 1e-4
    assert (dv_points - points).norm(dim=-1).max() < 1e-4


def test_surface_group_holds_its_surfaces_and_moves_them_together(scene):
    first = mn.OpenGLSurface(_saddle, resolution=(2, 2), add_to_scene=False)
    second = mn.OpenGLSurface(resolution=(3, 3), add_to_scene=False)

    group = mn.OpenGLSurfaceGroup(first, second, add_to_scene=False)

    members = list(group)
    assert [id(member) for member in members] == [id(first), id(second)]
    # Manim's group has no samples of its own.
    assert group.resolution == (0, 0)
    sized = mn.OpenGLSurfaceGroup(resolution=(4, 3), add_to_scene=False)
    assert sized.resolution == (4, 3)

    before = [member.grid.location.clone() for member in members]
    with Off():
        group.move(RIGHT)
    for member, location in zip(members, before):
        torch.testing.assert_close(member.grid.location, location + RIGHT)


def test_textured_surface_takes_its_uv_surfaces_samples(scene):
    """REGRESSION. It read its uv_surface's ``resolution``, which
    ``Surface.__init__`` had reset to ``None``, and so sampled a 101 x 101 grid
    of its own instead of the uv_surface's points.
    """
    plane = mn.OpenGLSurface(
        u_range=(-2, 2), v_range=(-1, 1), resolution=(3, 2), add_to_scene=False
    )

    textured = mn.OpenGLTexturedSurface(plane, _picture(), add_to_scene=False)

    assert textured.resolution == (3, 2)
    assert (textured.grid_width, textured.grid_height) == (3, 2)
    torch.testing.assert_close(textured.grid.location, plane.grid.location)


def test_textured_surface_lays_the_picture_upright_across_u_and_v(scene):
    """Manim reads the picture with ``v`` reversed (``init_points``' "Reverse
    y-direction"), so its left column lands at ``u_min`` and its top row at
    ``v_max``: pixel ``(row, column)`` of an ``H x W`` picture is drawn at
    ``u = (column + 1/2) / W``, ``v = 1 - (row + 1/2) / H``.
    """
    picture = _picture()
    height, width = picture.shape[:2]
    plane = mn.OpenGLSurface(
        u_range=(-2, 2), v_range=(-1, 1), resolution=(3, 2), add_to_scene=False
    )
    textured = mn.OpenGLTexturedSurface(plane, picture, add_to_scene=False)

    texels = textured.color_texture.as_subclass(torch.Tensor)[..., :3].reshape(-1, 3)
    with SETTINGS.raytracing.override(texture_antialiasing=True):
        where = textured.get_texture_locations().reshape(-1, 3)
    for row in range(height):
        for column in range(width):
            color = torch.as_tensor(picture[row, column, :3] / 255, dtype=texels.dtype)
            texel = (texels - color).abs().sum(-1).argmin()
            u = (column + 0.5) / width
            v = 1 - (row + 0.5) / height
            expected = torch.tensor([-2 + 4 * u, -1 + 2 * v, 0.0])
            torch.testing.assert_close(where[texel], expected, atol=1e-5, rtol=0)


def test_textured_native_surface_keeps_the_shape_it_was_built_with(scene):
    """REGRESSION. A Surface built from a ``coord_function`` was sampled through
    its class's own ``coord_function`` -- the base Surface's flat plane -- and
    so came out flat.
    """
    native = Surface(
        coord_function=_unit_saddle, grid_width=4, grid_height=3, add_to_scene=False
    )

    textured = mn.OpenGLTexturedSurface(native, _picture(), add_to_scene=False)

    assert (textured.grid_width, textured.grid_height) == (4, 3)
    torch.testing.assert_close(textured.grid.location, native.grid.location)


@pytest.mark.parametrize("solid", [Sphere, Cylinder], ids=lambda solid: solid.__name__)
def test_textured_built_in_solid_keeps_its_samples(scene, solid):
    # Cylinder is here because its coord_function broadcasts against its own
    # basis rows, so it must be handed its uv as a grid, not as a flat list.
    shape = solid(grid_width=7, grid_height=5, add_to_scene=False)

    textured = mn.OpenGLTexturedSurface(shape, _picture(), add_to_scene=False)

    assert (textured.grid_width, textured.grid_height) == (7, 5)
    torch.testing.assert_close(
        textured.grid.location, shape.grid.location, atol=1e-5, rtol=0
    )


@pytest.mark.parametrize("native", [False, True], ids=["OpenGLSurface", "Surface"])
def test_surface_mesh_draws_its_lattice_on_the_surface(scene, native):
    """REGRESSION. For a Surface built from a ``coord_function`` the lines were
    sampled through its class's own ``coord_function`` -- a flat plane -- so
    the mesh missed the surface it was drawn for.
    """
    if native:
        surface = Surface(
            coord_function=_unit_saddle, grid_width=5, grid_height=5, add_to_scene=False
        )
    else:
        surface = mn.OpenGLSurface(
            _saddle,
            u_range=(-1, 1),
            v_range=(-1, 1),
            resolution=(5, 5),
            add_to_scene=False,
        )

    mesh = mn.OpenGLSurfaceMesh(surface, resolution=(3, 4), add_to_scene=False)

    ends = torch.cat(
        [torch.cat((line.get_start(), line.get_end())).reshape(-1, 3) for line in mesh]
    )
    torch.testing.assert_close(ends[:, 2], ends[:, 0] * ends[:, 1], atol=1e-6, rtol=0)
    lattice = {
        (round(x, 5), round(y, 5))
        for x in np.linspace(-1, 1, 3).tolist()
        for y in np.linspace(-1, 1, 4).tolist()
    }
    assert {(round(x, 5), round(y, 5)) for x, y in ends[:, :2].tolist()} == lattice


def test_textured_surface_and_mesh_need_a_surface(scene):
    """Manim refuses anything but an OpenGLSurface; Algan also takes a native
    Surface, and refuses the rest with ``TypeError``.
    """
    square = Square(add_to_scene=False)

    with pytest.raises(TypeError, match="uv_surface"):
        mn.OpenGLTexturedSurface(square, _picture(), add_to_scene=False)
    with pytest.raises(TypeError, match="uv_surface"):
        mn.OpenGLSurfaceMesh(square, add_to_scene=False)
