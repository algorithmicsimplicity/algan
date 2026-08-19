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
    # Two same-surface same-facing sheets 1.0 apart in depth (a fold): the
    # relative rule must split them; the no-banding fallback fuses them, and
    # the overlapping masks then trip the fusion detector.
    frags = [
        (0, 1.0, 0, 0.5, 0b00111100),
        (0, 2.0, 1, 0.5, 0b00111100),
    ]
    split = _compact(frags, band_rule="prim")
    assert split["num_sheets"] == 2
    assert not bool(split["sheet_fused"].any())
    assert split["num_split_groups"] == 1

    fused = _compact(frags, band_rule="facing")
    assert fused["num_sheets"] == 1
    assert bool(fused["sheet_fused"][0])
    assert fused["num_split_groups"] == 0


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
