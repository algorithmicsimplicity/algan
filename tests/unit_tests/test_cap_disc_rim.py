"""A cap disc sizes its own rim against ``geometry_tolerance``.

A ``_CapDisc`` used to inherit the body's ring count and nothing downstream
could refine it -- a flat patch's PN boundary is its own straight chords, so
every render-time criterion returns zero and a coarse rim stayed coarse. The
rim is now grown in whole multiples of the body's ring count until the chord
polygon tracks the true rim curve within the body's ``geometry_tolerance``,
capped at ``max_grid_resolution``.

These tests read only built geometry and the shapes' documented parameters:
the deviation is measured off the cap's own vertices against the analytic
circle they are meant to approximate, not by re-running the implementation's
rim sampling.
"""

import torch

from algan.mobs.shapes_3d import Cone, Cylinder, Line3D

TOLERANCE = 0.0005

#: Radius of the capped shape each builder below makes.
CYLINDER_RADIUS = 0.45
CONE_RADIUS = 0.55


def _built_grid(mob):
    """The vertices a surface actually carries, as ``[W, H, 3]`` world points."""
    grid = mob._reshape_grid_for_render(mob.grid.location)
    return grid.reshape(-1, mob.grid_width, mob.grid_height, 3)[0]


def _make_capped_cylinder(**kwargs):
    body = Cylinder(radius=CYLINDER_RADIUS, height=1.0, show_ends=True, **kwargs)
    grid = _built_grid(body)
    # Rings are the constant-v rows (the azimuth runs on u); v=0/v=1 are the
    # two ends the caps close.
    return {
        body.bottom_cap: grid[:, 0],
        body.top_cap: grid[:, -1],
    }


def _make_capped_cone(**kwargs):
    body = Cone(base_radius=CONE_RADIUS, height=1.1, show_base=True, **kwargs)
    grid = _built_grid(body)
    # The azimuth runs on v; u=0 is the base ring (u=1 collapses to the tip).
    return {body.base_circle: grid[0]}


_BUILDERS = [_make_capped_cylinder, _make_capped_cone]


def _rim_points(cap):
    """The cap's built rim vertices in world space, seam duplicate dropped."""
    grid = _built_grid(cap)
    center = cap.location.reshape(-1, 3)[0]
    radii = (grid - center).norm(dim=-1)
    # One boundary row is the rim, the other the welded centre.
    rim = grid[:, -1] if radii[:, -1].max() >= radii[:, 0].max() else grid[:, 0]
    return rim[:-1]


def _rim_deviation(cap, radius):
    """Worst chord sagitta of the cap's built rim polygon, in world units."""
    points = _rim_points(cap)
    center = cap.location.reshape(-1, 3)[0]
    radii = (points - center).norm(dim=-1)
    # Precondition for the measurement to mean anything: every built vertex
    # sits on the circle it is supposed to sample.
    assert torch.allclose(radii, torch.full_like(radii, radius), atol=1e-5)
    midpoints = (points + points.roll(-1, dims=0)) * 0.5
    deviations = radius - (midpoints - center).norm(dim=-1)
    return float(deviations.max())


def test_rim_polygon_meets_geometry_tolerance():
    """The external invariant: the built rim polygon is within tolerance of
    the true rim curve, measured from the vertices the mob actually carries.
    """
    for cap in _make_capped_cylinder():
        assert _rim_deviation(cap, CYLINDER_RADIUS) <= TOLERANCE
    for cap in list(_make_capped_cone()):
        assert _rim_deviation(cap, CONE_RADIUS) <= TOLERANCE


def test_expected_rim_counts_at_the_default_tolerance():
    """70 chords for the cylinder's r=0.45 cap, 84 for the cone's r=0.55 --
    the smallest whole multiples of the body's ring counts inside tolerance.
    """
    cylinder_caps = _make_capped_cylinder()
    assert all(cap.grid_width == 71 for cap in cylinder_caps)
    (cone_cap,) = _make_capped_cone()
    assert cone_cap.grid_width == 85


def test_body_ring_vertices_survive_on_the_rim():
    """The weld: every body ring vertex is still exactly a rim vertex.

    The count grows only in whole multiples of the body's ring count, so no
    original sample can be lost; this checks the positions too.
    """
    for builder in _BUILDERS:
        rings = builder()
        for cap, ring in rings.items():
            assert (cap.grid_width - 1) % (ring.shape[0] - 1) == 0
            rim = _rim_points(cap)
            distances = (ring.unsqueeze(1) - rim.unsqueeze(0)).norm(dim=-1)
            assert float(distances.amin(dim=-1).amax()) < 1e-5


def test_body_tolerance_reaches_the_disc():
    """A coarser body gets a coarser cap, a finer one a finer cap."""
    for builder in _BUILDERS:
        coarse = list(builder(geometry_tolerance=0.005))
        fine = list(builder(geometry_tolerance=1e-4))
        nominal = list(builder())
        coarse_widths = [cap.grid_width for cap in coarse]
        nominal_widths = [cap.grid_width for cap in nominal]
        fine_widths = [cap.grid_width for cap in fine]
        assert max(coarse_widths) < min(nominal_widths)
        assert max(nominal_widths) < min(fine_widths)
        # And each looser rim really meets its own, looser, bound.
        radius = CYLINDER_RADIUS if builder is _make_capped_cylinder else CONE_RADIUS
        for cap in coarse:
            assert _rim_deviation(cap, radius) <= 0.005


def test_the_rim_cannot_exceed_max_grid_resolution():
    """A rim that cannot meet tolerance inside the grid cap degrades instead
    of raising -- and still keeps the whole-multiple weld.
    """
    for builder in _BUILDERS:
        rings = builder(max_grid_resolution=40)
        for cap, ring in rings.items():
            assert cap.grid_width <= 40
            assert (cap.grid_width - 1) % (ring.shape[0] - 1) == 0
            assert cap._max_grid_resolution == 40


def test_line3d_caps_keep_their_inherited_count():
    """A thin tube's rim already meets tolerance at its inherited count, so
    the fix must leave it alone rather than churn geometry it need not touch.
    """
    line = Line3D(thickness=0.02)
    assert line.grid_width == 24
    assert line.bottom_cap.grid_width == line.top_cap.grid_width == 24
