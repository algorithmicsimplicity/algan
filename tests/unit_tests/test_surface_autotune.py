import torch
from algan.mobs.surfaces.surface import Surface
from algan.mobs.shapes_3d import Sphere, Cylinder

def test_surface_autotune_default():
    # Test that default instantiation of Surface performs auto-tuning
    surf = Surface(
        coord_function=lambda uv: torch.cat(((uv - 0.5) * 2, torch.zeros_like(uv[..., :1])), -1)
    )
    # The default tolerance is 0.01. Since the surface is completely flat, a very low resolution
    # (like 4x4) is enough because flat triangles match the shape function exactly.
    assert surf.grid_height >= 4
    assert surf.grid_width >= 4
    print(f"Flat Surface auto-tuned to: {surf.grid_width}x{surf.grid_height}")

def test_sphere_autotune():
    # Geometry tolerance is an absolute world-space construction constraint.
    sphere_coarse = Sphere(geometry_tolerance=0.05)
    sphere_fine = Sphere(geometry_tolerance=0.005)
    
    # Check that finer tolerance results in higher resolution grid
    assert sphere_fine.grid_width > sphere_coarse.grid_width
    assert sphere_fine.grid_height > sphere_coarse.grid_height
    print(f"Sphere auto-tuned (tolerance=0.05) to: {sphere_coarse.grid_width}x{sphere_coarse.grid_height}")
    print(f"Sphere auto-tuned (tolerance=0.005) to: {sphere_fine.grid_width}x{sphere_fine.grid_height}")

def test_cylinder_autotune():
    # Test construction-time logical PN topology selection.
    cyl_coarse = Cylinder(geometry_tolerance=0.05)
    cyl_fine = Cylinder(geometry_tolerance=0.01)
    assert cyl_fine.grid_width >= cyl_coarse.grid_width
    print(f"Cylinder auto-tuned (tolerance=0.05) to: {cyl_coarse.grid_width}x{cyl_coarse.grid_height}")
    print(f"Cylinder auto-tuned (tolerance=0.01) to: {cyl_fine.grid_width}x{cyl_fine.grid_height}")

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
        coord_function=lambda uv: torch.cat(((uv - 0.5) * 2, torch.zeros_like(uv[..., :1])), -1),
        grid_height=12,
        grid_width=15
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
