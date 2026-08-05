import math
import warnings

import pytest
import torch

from algan.mobs.shapes_3d import Cylinder, Sphere
from algan.mobs.surfaces.surface import Surface


def _stretched_plane(uv):
    """A flat plane whose parameterization bunches up towards one corner.

    Every PN patch of a plane *is* the plane, so the geometric error is zero at
    any resolution; only a same-parameter comparison sees the stretching.
    """
    u = uv[..., :1] ** 3
    v = uv[..., 1:] ** 3
    return torch.cat((u * 2 - 1, v * 2 - 1, torch.zeros_like(u)), -1)


class _Superellipsoid(Surface):
    """A Go-stone-like solid of revolution: poles and rim are both singular.

    ``exponent`` below 1 gives the meridian an infinite parameter derivative at
    the equator and collapses the polar caps, so the parameterization stretches
    without bound near ``v = 0``, ``0.5`` and ``1``.
    """

    def __init__(self, thickness=0.36, exponent=2 / 2.6, **kwargs):
        self.thickness = thickness
        self.exponent = exponent
        super().__init__(**kwargs)

    def coord_function(self, uv):
        theta = uv[..., :1] * (2 * math.pi)
        alpha = (uv[..., 1:] - 0.5) * math.pi
        radius = torch.cos(alpha).clamp_min(0) ** self.exponent
        z = torch.sin(alpha)
        z = z.sign() * z.abs() ** self.exponent * self.thickness
        return torch.cat((radius * torch.cos(theta), radius * torch.sin(theta), z), -1)


class _CachedSurface(Surface):
    search_calls = 0

    def __init__(self, curvature=1.0, **kwargs):
        self.curvature = curvature
        super().__init__(**kwargs)

    def coord_function(self, uv):
        u = uv[..., :1] * 2 - 1
        v = uv[..., 1:] * 2 - 1
        return torch.cat((u, v, (u.square() + v.square()) * self.curvature), -1)

    def _find_geometry_resolution(self, _surface_function):
        type(self).search_calls += 1
        return 5 + int(self.curvature), 5


class _OtherCachedSurface(_CachedSurface):
    pass


def test_surface_autotune_default():
    # Test that default instantiation of Surface performs auto-tuning
    surf = Surface(
        coord_function=lambda uv: torch.cat(
            ((uv - 0.5) * 2, torch.zeros_like(uv[..., :1])), -1
        )
    )
    # The default tolerance is 0.01. Since the surface is completely flat, a very low resolution
    # (like 4x4) is enough because flat triangles match the shape function exactly.
    assert surf.grid_height >= 4
    assert surf.grid_width >= 4
    print(f"Flat Surface auto-tuned to: {surf.grid_width}x{surf.grid_height}")


def test_auto_resolution_reuses_each_subclass_cache():
    _CachedSurface.clear_geometry_resolution_cache()
    _CachedSurface.search_calls = 0

    first = _CachedSurface(curvature=1.0)
    second = _CachedSurface(curvature=1.0)

    assert _CachedSurface.search_calls == 1
    assert (first.grid_width, first.grid_height) == (6, 5)
    assert (second.grid_width, second.grid_height) == (6, 5)
    assert len(_CachedSurface._geometry_resolution_cache) == 1


def test_builtin_surface_subclass_skips_repeated_resolution_search(monkeypatch):
    Sphere.clear_geometry_resolution_cache()
    search_calls = 0
    original_search = Sphere._find_geometry_resolution

    def counting_search(self, surface_function):
        nonlocal search_calls
        search_calls += 1
        return original_search(self, surface_function)

    monkeypatch.setattr(Sphere, "_find_geometry_resolution", counting_search)

    first = Sphere(radius=1.5, geometry_tolerance=0.05, max_grid_resolution=80)
    second = Sphere(
        center=torch.tensor((3.0, -2.0, 1.0)),
        radius=1.5,
        geometry_tolerance=0.05,
        max_grid_resolution=80,
    )

    assert search_calls == 1
    assert (first.grid_width, first.grid_height) == (
        second.grid_width,
        second.grid_height,
    )
    Sphere.clear_geometry_resolution_cache()


def test_auto_resolution_cache_distinguishes_geometry_and_fitting_policy():
    _CachedSurface.clear_geometry_resolution_cache()
    _CachedSurface.search_calls = 0

    _CachedSurface(curvature=1.0, geometry_tolerance=0.01)
    _CachedSurface(curvature=2.0, geometry_tolerance=0.01)
    _CachedSurface(curvature=1.0, geometry_tolerance=0.005)

    assert _CachedSurface.search_calls == 3
    assert len(_CachedSurface._geometry_resolution_cache) == 3


def test_auto_resolution_caches_are_isolated_between_subclasses():
    _CachedSurface.clear_geometry_resolution_cache()
    _OtherCachedSurface.clear_geometry_resolution_cache()
    _CachedSurface.search_calls = 0
    _OtherCachedSurface.search_calls = 0

    _CachedSurface(curvature=1.0)
    _OtherCachedSurface(curvature=1.0)

    assert _CachedSurface.search_calls == 1
    assert _OtherCachedSurface.search_calls == 1
    assert (
        _CachedSurface._geometry_resolution_cache
        is not _OtherCachedSurface._geometry_resolution_cache
    )


def test_sphere_autotune():
    # Geometry tolerance is an absolute world-space construction constraint.
    sphere_coarse = Sphere(geometry_tolerance=0.05)
    sphere_fine = Sphere(geometry_tolerance=0.005)

    # Check that finer tolerance results in higher resolution grid
    assert sphere_fine.grid_width > sphere_coarse.grid_width
    assert sphere_fine.grid_height > sphere_coarse.grid_height
    print(
        f"Sphere auto-tuned (tolerance=0.05) to: {sphere_coarse.grid_width}x{sphere_coarse.grid_height}"
    )
    print(
        f"Sphere auto-tuned (tolerance=0.005) to: {sphere_fine.grid_width}x{sphere_fine.grid_height}"
    )


def test_cylinder_autotune():
    # Test construction-time logical PN topology selection.
    cyl_coarse = Cylinder(geometry_tolerance=0.05)
    cyl_fine = Cylinder(geometry_tolerance=0.01)
    assert cyl_fine.grid_width >= cyl_coarse.grid_width
    print(
        f"Cylinder auto-tuned (tolerance=0.05) to: {cyl_coarse.grid_width}x{cyl_coarse.grid_height}"
    )
    print(
        f"Cylinder auto-tuned (tolerance=0.01) to: {cyl_fine.grid_width}x{cyl_fine.grid_height}"
    )


def test_cylinder_autotune_rectangular():
    # Test independent rectangular topology search.
    cyl = Cylinder(geometry_tolerance=0.01, grid_aspect_ratio=None)
    # The flat direction (v/height) should require minimal resolution (4)
    assert cyl.grid_height == 4
    # The curved direction (u/width) should require more resolution (> 4)
    assert cyl.grid_width > 4
    print(f"Cylinder rectangular auto-tuned to: {cyl.grid_width}x{cyl.grid_height}")


def test_manual_resolution_override():
    # If the user specifies grid_height/grid_width, auto-tuning should be bypassed
    surf = Surface(
        coord_function=lambda uv: torch.cat(
            ((uv - 0.5) * 2, torch.zeros_like(uv[..., :1])), -1
        ),
        grid_height=12,
        grid_width=15,
    )
    assert surf.grid_height == 12
    assert surf.grid_width == 15

    sphere = Sphere(grid_height=30)
    assert sphere.grid_height == 30
    assert sphere.grid_width == 30


def test_error_constraint():
    sphere = Sphere(geometry_tolerance=0.01)
    error = sphere._compute_pn_geometry_error(
        sphere.coord_function_active,
        sphere.grid_width,
        sphere.grid_height,
    )
    print(f"Sphere actual error: {error.item():.6f}, threshold: 0.010000")
    assert error <= sphere.geometry_tolerance


def test_reparameterization_alone_is_not_geometric_error():
    # A plane is reproduced exactly by PN triangles at any resolution, so a
    # stretched parameterization must not cost a single extra vertex.
    surf = Surface(coord_function=_stretched_plane, geometry_tolerance=0.001)
    assert surf.grid_width == surf._min_grid_resolution
    assert surf.grid_height == surf._min_grid_resolution


def test_singular_parameterization_does_not_exhaust_the_grid_budget():
    # The metric used to compare points at matching parameters, which counts
    # tangential slip as error. Near a collapsed pole that term never falls
    # below tolerance, so shapes like this one ran to max_grid_resolution.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        stone = _Superellipsoid(geometry_tolerance=0.001)

    assert stone.grid_width * stone.grid_height < 2000
    assert max(stone.grid_width, stone.grid_height) < stone._max_grid_resolution


@pytest.mark.parametrize("tolerance", [0.01, 0.001])
def test_chosen_grid_is_within_tolerance_and_not_oversized(tolerance):
    stone = _Superellipsoid(geometry_tolerance=tolerance)
    width, height = stone.grid_width, stone.grid_height

    def error(w, h):
        return stone._compute_pn_geometry_error(
            stone.coord_function_active, w, h
        ).item()

    assert error(width, height) <= tolerance
    # And it is not paying for resolution it does not need. Halving is a coarse
    # probe deliberately: the metric samples a finite set of points per
    # triangle, so a one-step-smaller grid can measure marginally either way.
    assert error(max(4, width // 2), max(4, height // 2)) > tolerance


def test_unreachable_tolerance_still_warns():
    with pytest.warns(RuntimeWarning, match="max_grid_resolution"):
        surf = _Superellipsoid(geometry_tolerance=1e-7, max_grid_resolution=8)
    assert surf.grid_width == 8
    assert surf.grid_height == 8
