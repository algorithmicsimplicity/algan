"""Welding a closed surface grid's u-seam and its collapsed poles.

``get_grid_to_triangle_indices`` builds two triangles per grid cell and never
bridges column ``W-1`` back to column 0, so a closed surface of revolution's
wraparound is a genuine two-copy seam -- the two columns differ by up to 1.7e-07
in f32 and are not bitwise equal, while every interior shared edge *is* a
bit-identical duplicate of the same gather. A watertight intersection test
(``DESIGN_mesh_identity.md`` §3.2) fixes numerical ambiguity; at that seam it
would open a crack rather than close one, because the gap is real geometry.

``ALGAN_WELD_SURFACE_SEAMS`` closes it structurally instead: the wrap cell
indexes column 0, and a collapsed pole row becomes one vertex, which drops the
``W-1`` degenerate triangles each pole contributes.

These are index-tensor assertions -- no render, no Taichi -- but they are left
out of the fast suite: nothing else in the codebase can break them, so they are
feature tests for the weld itself.
"""

from __future__ import annotations

import torch

from algan.mobs.surfaces.surface import (
    get_grid_to_triangle_indices,
    surface_weld_flags,
)

W, H = 8, 5
CPU = torch.device("cpu")


def _tris(weld):
    return get_grid_to_triangle_indices(W, H, CPU, weld).view(-1, 3)


def _degenerate(tris):
    return int(
        (
            (tris[:, 0] == tris[:, 1])
            | (tris[:, 1] == tris[:, 2])
            | (tris[:, 0] == tris[:, 2])
        )
        .sum()
        .item()
    )


def test_the_unwelded_topology_is_exactly_what_it_always_was():
    """The default must stay byte-identical: it is what every baseline holds."""
    tris = _tris((False, False, False))
    assert tris.shape == ((W - 1) * (H - 1) * 2, 3)
    # Cell (0, 0) is the two triangles the original construction emitted.
    assert tris[0].tolist() == [0, 1, H]
    assert tris[1].tolist() == [H, 1, H + 1]


def test_welding_the_seam_never_indexes_the_duplicate_column():
    """The wrap cell must reference column 0, leaving column W-1 unreferenced.

    That is what makes the seam a shared edge instead of two copies a rounding
    error apart.
    """
    tris = _tris((True, False, False))
    # Same triangle count: welding the seam re-points vertices, it drops none.
    assert tris.shape == ((W - 1) * (H - 1) * 2, 3)
    columns = tris // H
    assert not bool((columns == W - 1).any()), "column W-1 is still gathered"
    assert bool((columns == 0).any())


def test_welding_a_pole_drops_exactly_its_degenerate_triangles():
    for weld, dropped in (
        ((False, True, False), W - 1),
        ((False, False, True), W - 1),
        ((False, True, True), 2 * (W - 1)),
    ):
        tris = _tris(weld)
        assert tris.shape[0] == (W - 1) * (H - 1) * 2 - dropped, weld
        assert _degenerate(tris) == 0, weld


def test_a_fully_welded_grid_has_no_degenerate_triangles_and_one_pole_vertex():
    tris = _tris((True, True, True))
    assert _degenerate(tris) == 0
    rows = tris % H
    # Every reference to a pole row resolves to column 0's vertex.
    for pole_row in (0, H - 1):
        at_pole = tris[rows == pole_row]
        assert set((at_pole // H).tolist()) == {0}, pole_row


def test_welding_is_a_pure_reindex_of_the_same_grid_points():
    """No welded index may point outside the grid."""
    for weld in ((True, False, False), (False, True, True), (True, True, True)):
        tris = _tris(weld)
        assert int(tris.min()) >= 0
        assert int(tris.max()) < W * H, weld


def test_weld_flags_read_the_geometry_and_respect_the_gate():
    from algan import Off, Scene, Sphere
    from algan.rendering.raytracing import settings as rt_settings

    with Scene(), Off():
        sphere = Sphere(radius=1.0, resolution=(16, 9)).spawn()
        grid = sphere._reshape_grid_for_render(sphere.grid.location)
        # A plane: no wrap, no poles.
        flat = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, 6), torch.linspace(0, 1, 4), indexing="ij"
            ),
            dim=-1,
        )
        flat = torch.cat([flat, torch.zeros_like(flat[..., :1])], dim=-1)
        assert surface_weld_flags(grid) == (False, False, False), (
            "the weld must be inert until ALGAN_WELD_SURFACE_SEAMS is on"
        )
        original = rt_settings.WELD_SURFACE_SEAMS
        try:
            rt_settings.set_weld_surface_seams(True)
            assert surface_weld_flags(grid) == (True, True, True), (
                "a Sphere wraps in u and collapses at both poles"
            )
            assert surface_weld_flags(flat) == (False, False, False), (
                "a flat open patch has nothing to weld"
            )
        finally:
            rt_settings.set_weld_surface_seams(original)
