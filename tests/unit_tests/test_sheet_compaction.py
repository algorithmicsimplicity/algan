"""Sheet compaction (DESIGN_sheet_resolve.md P1/P2) on synthetic streams.

Pure tensor assertions against hand-built fragment streams -- no render, no
kernel launch. These pin the semantics the Phase-2 resolve consumes: keying,
banding, the exact-area sum, the mask union, the fusion detector, the dominant
fragment, ordering, and the CSR.
"""

from __future__ import annotations

import torch

from algan.rendering.raytracing.raster_taichi import (
    _AA_BACKFACE_BIT as BACKFACE,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_MASK_ALL as MASK_ALL,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_SLIVER_BIT as SLIVER,
)
from algan.rendering.raytracing.sheets import compact_sheets


def _key(pix, t):
    tb = int(torch.tensor([t], dtype=torch.float32).view(torch.int32)[0].item())
    return (int(pix) << 32) | (tb & 0xFFFFFFFF)


def _coverage(frags, num_tris=8):
    """Build a coverage dict + merged scene from ``(pix, t, ref, cov, msk)``
    rows (already in (pixel, depth) order, like the emission's).

    Triangles are placed flat-on at depth ``t`` with a small extent so the
    ``prim`` band rule sees a tiny per-fragment scale; the camera sits at the
    origin looking down +z.
    """
    n = len(frags)
    pix = torch.tensor([f[0] for f in frags], dtype=torch.int64)
    t = torch.tensor([f[1] for f in frags], dtype=torch.float32)
    ref = torch.tensor([f[2] for f in frags], dtype=torch.int32)
    cov = torch.tensor([f[3] for f in frags], dtype=torch.float32)
    msk = torch.tensor([f[4] for f in frags], dtype=torch.int32)
    key = torch.tensor([_key(f[0], f[1]) for f in frags], dtype=torch.int64)
    covered, counts = torch.unique_consecutive(pix, return_counts=True)
    run_offsets = torch.zeros(covered.numel() + 1, dtype=torch.int32)
    run_offsets[1:] = torch.cumsum(counts.to(torch.int32), 0)
    coverage = {
        "frag_key": key,
        "frag_ref": ref,
        "frag_ab": torch.zeros(n, 2),
        "frag_cov": cov,
        "frag_msk": msk,
        "frag_cap": torch.full((n,), 2.0),
        "covered_idx": covered.to(torch.int32),
        "run_offsets": run_offsets,
        "num_fragments": n,
        "num_covered": int(covered.numel()),
    }
    # tri_obj: triangles 0..3 belong to surface 0, 4..7 to surface 1.
    tri_obj = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1][:num_tris]])
    # Flat-on unit-scale triangles at per-ref depth 1 + ref * 0.001 (the band
    # rule reads vertex distances, and only gaps matter here).
    tp = torch.zeros(1, num_tris, 9)
    for r in range(num_tris):
        z = 1.0 + 0.001 * r
        tp[0, r] = torch.tensor([0.0, 0, z, 0.05, 0, z, 0, 0.05, z])
    merged = {"tri_obj": tri_obj, "tri_pos": tp}
    cam = torch.zeros(1, 3)
    pws = torch.full((1,), 1e-3)
    return coverage, merged, cam, pws


def _compact(frags, band_rule="prim", band_c=4.0):
    coverage, merged, cam, pws = _coverage(frags)
    return compact_sheets(
        coverage,
        merged,
        cam,
        pws,
        time_start=0,
        width=4,
        height=4,
        band_rule=band_rule,
        band_c=band_c,
    )


def test_one_sheet_tiling_sums_exact_area_and_unions_masks():
    # Two triangles of one surface tiling a pixel 0.4/0.6 with disjoint masks.
    out = _compact(
        [
            (3, 1.0, 0, 0.4, 0b00001111),
            (3, 1.0001, 1, 0.6, 0b11110000),
        ]
    )
    assert out["num_sheets"] == 1
    assert torch.allclose(out["sheet_cov"], torch.tensor([1.0]))
    assert int(out["sheet_msk"][0]) & MASK_ALL == MASK_ALL
    assert not bool(out["sheet_fused"][0])
    # Dominant fragment is the larger-area one.
    assert int(out["sheet_ref"][0]) == 1
    assert int(out["sheet_nfrag"][0]) == 2


def test_facing_separates_a_closed_surface_into_two_sheets():
    out = _compact(
        [
            (0, 1.0, 0, 0.5, 0b0111),
            (0, 2.0, 1, 0.5, BACKFACE | 0b0111),
        ]
    )
    assert out["num_sheets"] == 2
    # Near sheet first (classic order), and the facing flag survives.
    assert int(out["sheet_msk"][0]) & BACKFACE == 0
    assert int(out["sheet_msk"][1]) & BACKFACE != 0


def test_band_rule_splits_a_depth_gap_and_facing_rule_fuses_it():
    # Two same-surface same-facing sheets 1.0 apart in depth with DISJOINT
    # masks (the fill rule alone cannot separate them): the relative depth
    # rule must split them; the no-banding fallback fuses them.
    frags = [
        (0, 1.0, 0, 0.5, 0b00001111),
        (0, 2.0, 1, 0.5, 0b11110000),
    ]
    split = _compact(frags, band_rule="prim")
    assert split["num_sheets"] == 2
    assert split["num_split_groups"] == 1

    fused = _compact(frags, band_rule="facing")
    assert fused["num_sheets"] == 1
    assert fused["num_split_groups"] == 0


def test_mask_conflict_splits_overlapping_layers_at_any_depth():
    # The fill rule is the sheet-membership oracle: two same-id same-facing
    # fragments whose masks OVERLAP hold two sheets whatever their depths
    # (a mid-morph self-overlap has no depth gap to split on). Each layer
    # must become its own sheet so a translucent surface a ray crosses twice
    # attenuates twice.
    frags = [
        (0, 1.0000, 0, 0.5, 0b00111100),
        (0, 1.0001, 1, 0.5, 0b00111100),
    ]
    for rule in ("prim", "facing"):
        out = _compact(frags, band_rule=rule)
        assert out["num_sheets"] == 2, rule
        assert not bool(out["sheet_fused"].any()), rule
        assert torch.allclose(out["sheet_cov"], torch.tensor([0.5, 0.5])), rule
    # And the oracle attenuates twice across the two layers.
    from algan.rendering.raytracing.sheets import resolve_pixel_reference

    claims, T = resolve_pixel_reference(
        [0.5, 0.5],
        [0b00111100, 0b00111100],
        [False, False],
        alphas=[0.5, 0.5],
    )
    assert claims[1] < claims[0]  # the second layer sees attenuated samples


def test_adjacent_same_sheet_fragments_do_not_split():
    # Fragments of one sheet a hair apart in depth: gaps far under the
    # primitive scale, so the prim rule must NOT split them.
    out = _compact(
        [
            (0, 1.0000, 0, 0.3, 0b0011),
            (0, 1.0004, 1, 0.3, 0b1100),
        ],
        band_rule="prim",
    )
    assert out["num_sheets"] == 1
    assert out["num_split_groups"] == 0


def test_bezier_fragments_stand_alone():
    # Two circuit fragments and a triangle in one pixel: circuits never group.
    out = _compact(
        [
            (0, 1.0, -1, 0.5, MASK_ALL),
            (0, 1.5, -300, 0.4, MASK_ALL),
            (0, 2.0, 2, 0.9, 0b1111),
        ]
    )
    assert out["num_sheets"] == 3
    assert int(out["sheet_ref"][0]) == -1
    assert int(out["sheet_ref"][1]) == -300
    assert int(out["sheet_ref"][2]) == 2


def test_donor_only_sheet_is_flagged_areal():
    # A sheet of empty-mask sliver donors: union empty => the sliver (areal)
    # flag must be set so a walk treats its area as positionless.
    out = _compact(
        [
            (0, 1.0, 0, 0.15, 0),
            (0, 1.0002, 1, 0.10, 0),
        ]
    )
    assert out["num_sheets"] == 1
    assert torch.allclose(out["sheet_cov"], torch.tensor([0.25]))
    assert int(out["sheet_msk"][0]) & SLIVER != 0


def test_csr_aligns_with_covered_idx_and_order_is_classic():
    # Pixel 1: surfaces 0 (two frags) then 1; pixel 5: one bezier. The sheet
    # stream must be pixel-major with each pixel's sheets in the order their
    # nearest fragments appeared in the emission.
    out = _compact(
        [
            (1, 1.0, 0, 0.4, 0b0011),
            (1, 1.0002, 4, 0.5, 0b1100),
            (1, 1.0005, 1, 0.4, 0b0100),
            (5, 0.7, -1, 0.8, MASK_ALL),
        ]
    )
    # Surface 0's two fragments interleave with surface 1's at gaps far
    # inside the primitive scale: compaction unifies them into one sheet
    # regardless (no consecutive-run requirement).
    assert out["num_sheets"] == 3
    offsets = out["sheet_offsets"].tolist()
    assert offsets == [0, 2, 3]
    assert out["sheet_pix"].tolist() == [1, 1, 5]
    # Pixel 1's first sheet is surface 0 (its nearest fragment leads).
    assert int(out["sheet_ref"][0]) in (0, 1)
    assert int(out["sheet_ref"][1]) == 4
    assert abs(float(out["sheet_cov"][0]) - 0.8) < 1e-6
    assert int(out["sheet_nfrag"][0]) == 2


def test_band_scale_is_per_pixel_slope_not_raw_extent():
    # The measured fusion defect: a big triangle (large camera-distance
    # extent) 1.0 BEHIND a same-id fragment must still band-split when its
    # projected size says its per-pixel depth slope is small. Masks are
    # DISJOINT so the fill-rule conflict split cannot mask the depth
    # question. With no tri_screen the conservative raw extent fuses them;
    # with the projection table the slope splits them.
    frags = [
        (0, 1.0, 0, 0.38, 0b00001001),
        (0, 2.0, 1, 1.0, 0b11110110),
    ]
    coverage, merged, cam, pws = _coverage(frags)
    # Make triangle 1 a huge wall: vertices spanning depths 1.0 .. 5.0.
    merged["tri_pos"][0, 1] = torch.tensor([0.0, 0, 1.9, 6.8, 0, 5.0, 0, 6.0, 2.1])
    fused = compact_sheets(
        coverage, merged, cam, pws, 0, 4, 4, band_rule="prim", band_c=2.0
    )
    assert fused["num_sheets"] == 1  # raw extent (4.0) swamps the 1.0 gap
    # Projection table: both triangles project large (500 px), flag valid.
    ts = torch.zeros(1, 8, 10)
    ts[..., 0:3] = torch.tensor([0.0, 500.0, 0.0])
    ts[..., 3:6] = torch.tensor([0.0, 0.0, 500.0])
    ts[..., 9] = 1.0
    split = compact_sheets(
        coverage,
        merged,
        cam,
        pws,
        0,
        4,
        4,
        band_rule="prim",
        band_c=2.0,
        tri_screen=ts,
    )
    assert split["num_sheets"] == 2
    assert not bool(split["sheet_fused"].any())


def test_oracle_energy_and_exactness():
    from algan.rendering.raytracing.sheets import resolve_pixel_reference

    # A full-union interior tiling composites at exactly 1 and blacks out T.
    claims, T = resolve_pixel_reference([1.0], [MASK_ALL], [False])
    assert claims == [1.0]
    assert all(t == 0.0 for t in T)

    # A full-union silhouette sheet paints its exact area, and the leftover
    # is exactly the complement (energy conserved).
    claims, T = resolve_pixel_reference([0.6], [MASK_ALL], [False])
    assert abs(claims[0] - 0.6) < 1e-12
    assert abs(sum(T) / len(T) - 0.4) < 1e-12

    # The dust band keeps a genuine tiling at exactly 1.
    claims, _ = resolve_pixel_reference([0.9995], [MASK_ALL], [False])
    assert claims == [1.0]

    # Partial union with representative samples: corr = 1, sampled compositing.
    claims, T = resolve_pixel_reference([0.25], [0b0011], [False])
    assert abs(claims[0] - 0.25) < 1e-12
    assert abs(sum(T) / len(T) - 0.75) < 1e-12

    # A sub-sample rod (area 0.4, one sample): claim exact, and the clamped
    # occlusion residue redistributes onto unowned samples so the background
    # still receives exactly 1 - area.
    claims, T = resolve_pixel_reference([0.4], [0b0001], [False])
    assert abs(claims[0] - 0.4) < 1e-12
    assert abs(sum(T) / len(T) - 0.6) < 1e-12

    # Areal sheets (donors, circuits) claim alpha * area uniformly.
    claims, T = resolve_pixel_reference([0.3], [SLIVER], [False])
    assert abs(claims[0] - 0.3) < 1e-12
    assert abs(sum(T) / len(T) - 0.7) < 1e-12

    # Translucency: two stacked full sheets at alpha 0.5 telescope exactly.
    claims, T = resolve_pixel_reference(
        [1.0, 1.0], [MASK_ALL, MASK_ALL], [False, False], alphas=[0.5, 0.5]
    )
    assert abs(claims[0] - 0.5) < 1e-12
    assert abs(claims[1] - 0.25) < 1e-12
    assert abs(sum(T) / len(T) - 0.25) < 1e-12

    # A transmitting sheet passes its share through per sample.
    claims, T = resolve_pixel_reference([1.0], [MASK_ALL], [False], trans=[0.5])
    assert abs(claims[0] - 1.0) < 1e-12
    assert abs(sum(T) / len(T) - 0.5) < 1e-12

    # Front sheet then far sheet of a closed solid at a silhouette: the far
    # sheet's residual re-claim is the documented bounded inter-sheet error
    # (ss6.1) -- pin its size so a change is a decision, not an accident.
    claims, T = resolve_pixel_reference(
        [0.25, 0.25], [0b0011, BACKFACE | 0b0011], [False, False]
    )
    assert abs(claims[0] - 0.25) < 1e-12
    # corr = 1 on both (area == Q), so the far sheet sees zeroed samples.
    assert claims[1] == 0.0


def test_oracle_one_mesh_ceiling_stops_the_far_sheet_reclaim():
    from algan.rendering.raytracing.raster_taichi import (
        _AA_ONE_MESH_BIT as ONE_MESH,
    )
    from algan.rendering.raytracing.sheets import resolve_pixel_reference

    # The measured Phase-2 regression case in miniature: a closed mesh's two
    # full-mask sheets at a silhouette, exact area 0.25 but owning 3 of 8
    # samples (Q = 0.375 > area, corr < 1). Without the ceiling the far sheet
    # re-claims part of the corr residue; with it the mesh's total claim is
    # exactly the near sheet's area.
    covs = [0.25, 0.25]
    msks = [ONE_MESH | 0b0111, ONE_MESH | BACKFACE | 0b0111]
    free, _T = resolve_pixel_reference(covs, msks, [False, False])
    capped, _T2 = resolve_pixel_reference(
        covs, msks, [False, False], caps=[0.25, 0.25]
    )
    assert free[1] > 0.0  # the re-claim exists uncapped
    assert abs(capped[0] - 0.25) < 1e-12
    assert capped[1] == 0.0  # the ceiling leaves the far sheet no room
    # A ceiling of 2.0 is the "no ceiling" sentinel and must change nothing.
    sentinel, _T3 = resolve_pixel_reference(
        covs, msks, [False, False], caps=[2.0, 2.0]
    )
    assert sentinel == free


def test_sheet_resolve_setting_reaches_the_live_module():
    from algan import SETTINGS
    from algan.rendering.raytracing import settings as rt

    old = rt.SHEET_RESOLVE
    try:
        SETTINGS.raytracing.experimental.set(sheet_resolve=True)
        assert rt.SHEET_RESOLVE is True
        SETTINGS.raytracing.experimental.set(sheet_resolve=False)
        assert rt.SHEET_RESOLVE is False
    finally:
        rt.SHEET_RESOLVE = old


def test_interleaved_fragments_keep_exact_key_depth():
    # sheet_key must be the NEAREST fragment's frag_key verbatim.
    out = _compact(
        [
            (2, 1.5, 0, 0.5, 0b0011),
            (2, 2.5, 1, 0.5, 0b1100),
        ],
        band_rule="facing",
    )
    t0 = (out["sheet_key"][0] & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
    assert float(t0) == 1.5
