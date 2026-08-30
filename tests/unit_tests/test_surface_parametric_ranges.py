"""``u_range`` / ``v_range`` on the Manim-parity curved shapes.

Two things are under test, and the second matters as much as the first. The
ranges must actually restrict the geometry -- they were accepted and stored but
never read by ``Sphere`` and ``Cylinder``, so a partial range silently built a
whole shape. And the *default* ranges must reproduce the sampling those shapes
had before the ranges were honoured, vertex for vertex: every committed render
baseline is a photograph of that grid, and a rounding difference in the
longitude lerp would move pixels in scenes that never mention a range.
"""

from __future__ import annotations

import math

import pytest
import torch

from algan.constants.math import PI
from algan.mobs.shapes_3d import Cylinder, Sphere
from algan.scene_manager import SceneManager


@pytest.fixture(autouse=True)
def reset_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _legacy_sphere_coords(uv: torch.Tensor, radius: float) -> torch.Tensor:
    """``Sphere.coord_function`` exactly as it read before ranges were honoured."""
    x = uv[..., 0]
    y = uv[..., 1]
    longitude = -math.pi * (1 - x) + x * math.pi
    latitude = -math.pi * 0.5 * (1 - y) + y * math.pi * 0.5
    coords = torch.stack(
        (
            torch.cos(latitude) * torch.cos(longitude),
            torch.sin(latitude),
            torch.cos(latitude) * torch.sin(longitude),
        ),
        dim=-1,
    )
    return coords * radius


def _legacy_cylinder_coords(mob: Cylinder, uv: torch.Tensor) -> torch.Tensor:
    """``Cylinder.coord_function`` exactly as it read before ranges were honoured."""
    uv = uv.clone()
    uv[..., 1:] /= uv[..., 1:].amax()
    u = -uv[..., :1]
    v = uv[..., 1:]
    return (
        (u * math.pi * 2).sin() * mob.radius * mob.get_right_basis()
        + (v - 0.5) * mob.height * mob.get_up_basis()
        + (u * math.pi * 2).cos() * mob.radius * mob.get_forward_basis()
    )


@pytest.mark.parametrize("radius", [1, 0.8, 2])
def test_default_sphere_grid_is_bit_identical_to_the_legacy_sampling(radius):
    sphere = Sphere(radius=radius, add_to_scene=False)
    base_grid = sphere.get_base_grid().clone()

    produced = sphere.coord_function(base_grid.clone())
    expected = _legacy_sphere_coords(base_grid, radius)

    assert torch.equal(produced, expected)


def test_explicit_default_sphere_ranges_are_bit_identical_to_omitting_them():
    implicit = Sphere(add_to_scene=False)
    explicit = Sphere(u_range=(0, 2 * PI), v_range=(0, PI), add_to_scene=False)

    assert (implicit.grid_width, implicit.grid_height) == (
        explicit.grid_width,
        explicit.grid_height,
    )
    assert torch.equal(implicit.grid.location, explicit.grid.location)


@pytest.mark.parametrize(("radius", "height"), [(1, 1), (0.4, 1.6)])
def test_default_cylinder_grid_is_bit_identical_to_the_legacy_sampling(radius, height):
    cylinder = Cylinder(radius=radius, height=height, add_to_scene=False)
    base_grid = cylinder.get_base_grid().clone()

    produced = cylinder.coord_function(base_grid.clone())
    expected = _legacy_cylinder_coords(cylinder, base_grid)

    assert torch.equal(produced, expected)


def test_explicit_default_cylinder_range_is_bit_identical_to_omitting_it():
    implicit = Cylinder(add_to_scene=False)
    explicit = Cylinder(v_range=(0, 2 * PI), add_to_scene=False)

    assert (implicit.grid_width, implicit.grid_height) == (
        explicit.grid_width,
        explicit.grid_height,
    )
    assert torch.equal(implicit.grid.location, explicit.grid.location)


def _extents(mob):
    points = mob.grid.location.detach().reshape(-1, 3)
    return points.amin(dim=0), points.amax(dim=0)


def test_sphere_u_range_restricts_the_azimuth_to_the_out_half():
    # ``u`` starts at LEFT and turns through OUT, so half of it is the half
    # facing the camera: OUT is -z, and no vertex may cross into +z.
    low, high = _extents(Sphere(u_range=(0, PI), add_to_scene=False).spawn())

    assert high[2] <= 1e-6
    assert low[2] < -0.99
    # The azimuth is cut, not the poles: full height and full width survive.
    assert low[1] == pytest.approx(-1, abs=1e-5)
    assert high[1] == pytest.approx(1, abs=1e-5)
    assert low[0] < -0.99
    assert high[0] > 0.99


def test_sphere_u_range_second_half_is_the_in_half():
    low, high = _extents(Sphere(u_range=(PI, 2 * PI), add_to_scene=False).spawn())

    assert low[2] >= -1e-6
    assert high[2] > 0.99


def test_sphere_v_range_runs_pole_to_pole():
    bottom_low, bottom_high = _extents(
        Sphere(v_range=(0, PI / 2), add_to_scene=False).spawn()
    )
    top_low, top_high = _extents(
        Sphere(v_range=(PI / 2, PI), add_to_scene=False).spawn()
    )

    # v = 0 is the DOWN pole and v = PI the UP one.
    assert bottom_low[1] == pytest.approx(-1, abs=1e-5)
    assert bottom_high[1] <= 1e-6
    assert top_low[1] >= -1e-6
    assert top_high[1] == pytest.approx(1, abs=1e-5)


def test_partial_sphere_vertices_all_lie_on_the_sphere():
    sphere = Sphere(
        radius=0.7,
        u_range=(0, PI / 2),
        v_range=(PI / 4, 3 * PI / 4),
        add_to_scene=False,
    ).spawn()

    radii = sphere.grid.location.detach().reshape(-1, 3).norm(dim=-1)

    assert torch.allclose(radii, torch.full_like(radii, 0.7), atol=1e-5)


def test_cylinder_v_range_restricts_the_azimuth_to_the_left_half():
    low, high = _extents(Cylinder(v_range=(0, PI), add_to_scene=False).spawn())

    assert high[0] <= 1e-6
    assert low[0] < -0.99
    # The cut runs the length of the tube, so the axial extent is untouched.
    assert low[1] == pytest.approx(-0.5, abs=1e-5)
    assert high[1] == pytest.approx(0.5, abs=1e-5)


def test_cylinder_v_range_second_half_is_the_right_half():
    low, high = _extents(Cylinder(v_range=(PI, 2 * PI), add_to_scene=False).spawn())

    assert low[0] >= -1e-6
    assert high[0] > 0.99


def test_partial_cylinder_vertices_all_lie_on_the_cylinder():
    cylinder = Cylinder(radius=0.6, v_range=(0, PI / 2), add_to_scene=False).spawn()

    points = cylinder.grid.location.detach().reshape(-1, 3)
    radii = points[:, [0, 2]].norm(dim=-1)

    assert torch.allclose(radii, torch.full_like(radii, 0.6), atol=1e-5)


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (Sphere, {"u_range": (0, PI)}),
        (Sphere, {"v_range": (0, PI / 2)}),
        (Cylinder, {"v_range": (0, PI)}),
    ],
)
def test_a_partial_range_no_longer_reproduces_the_whole_shape(factory, kwargs):
    # The defect this file exists for: the ranges used to be stored and never
    # read, so a partial shape came out with the extents of the whole one.
    whole_low, whole_high = _extents(factory(add_to_scene=False).spawn())
    partial_low, partial_high = _extents(factory(add_to_scene=False, **kwargs).spawn())

    assert not torch.allclose(whole_low, partial_low, atol=1e-4) or not torch.allclose(
        whole_high, partial_high, atol=1e-4
    )
