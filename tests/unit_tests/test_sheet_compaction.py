"""Sheet compaction (DESIGN_sheet_resolve.md P1/P2) on synthetic streams.

Pure tensor assertions against hand-built fragment streams -- no render. These
pin the semantics the Phase-2 resolve consumes: keying, banding, the exact-area
sum, the mask union, the fusion detector, the dominant fragment, ordering, and
the CSR.

The two ``sheet_rank_kernel`` tests are the exception to "no kernel launch":
they run the conflict-rank scan through both arms and compare, which is the
only way to pin a kernel against the torch expression it replaced. They pay one
small Taichi compile between them, which is why nothing here is marked ``fast``.
"""

from __future__ import annotations

import pytest
import torch

from algan.rendering.raytracing.raster_taichi import (
    _AA_BACKFACE_BIT as BACKFACE,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_LOSE_SHIFT as LOSE_SHIFT,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_MASK_ALL as MASK_ALL,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_MAT_OPAQUE_BIT as MAT_OPAQUE,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_SLIVER_BIT as SLIVER,
)
from algan.rendering.raytracing.sheets import compact_sheets


def _key(pix, t):
    tb = int(torch.tensor([t], dtype=torch.float32).view(torch.int32)[0].item())
    return (int(pix) << 32) | (tb & 0xFFFFFFFF)


def _coverage(frags, num_tris=8, tri_norm=None):
    """Build a coverage dict + merged scene from ``(pix, t, ref, cov, msk)``
    rows (already in (pixel, depth) order, like the emission's).

    Triangles are placed flat-on at depth ``t`` with a small extent so the
    ``prim`` band rule sees a tiny per-fragment scale; the camera sits at the
    origin looking down +z. ``tri_norm`` optionally supplies the per-triangle
    vertex normals ``[1, num_tris, 9]`` the shade-split class reads.
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
    if tri_norm is not None:
        merged["tri_norm"] = tri_norm
    cam = torch.zeros(1, 3)
    pws = torch.full((1,), 1e-3)
    return coverage, merged, cam, pws


def _compact(
    frags,
    band_rule="prim",
    band_c=4.0,
    shade_split=False,
    tri_norm=None,
    positioned_depth=True,
    sample_depth=False,
    tri_extra=None,
):
    coverage, merged, cam, pws = _coverage(frags, tri_norm=tri_norm)
    if tri_extra is not None:
        merged["tri_extra"] = tri_extra
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
        shade_split=shade_split,
        positioned_depth=positioned_depth,
        sample_depth=sample_depth,
    )


def _t(out, i):
    """The distance packed into sheet ``i``'s key."""
    bits = (out["sheet_key"][i] & 0xFFFFFFFF).to(torch.int32)
    return float(bits.view(torch.float32))


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


def test_a_seam_rank_split_composites_as_one_band():
    # ...and the same key ALSO fires on a seam, where one layer's adjacent
    # triangles fail to partition the samples by a sliver -- a T-junction
    # between two adaptively diced patches, say. Walked as independent
    # occluders those sub-bands under-claim ((1-a)(1-b) > 1-(a+b)) and admit
    # whatever is behind an opaque surface: the white Line3D inside
    # solids_and_camera's red Arrow3D, as a bright speck on the cone shoulder.
    # A band that owns EVERY sample at about unit area is one layer whatever
    # the fill rule did, so its sub-bands composite as §4.4 siblings instead.
    frags = [
        (0, 1.0000, 0, 0.80, 0b11011111),
        (0, 1.0001, 1, 0.21, 0b00110000),  # conflicts on lane 4 -> rank 1
    ]
    for rule in ("prim", "facing"):
        out = _compact(frags, band_rule=rule)
        # The SPLIT is untouched: two sheets, each recording its own area and
        # its own union. Only the compositing weights pool.
        assert out["num_sheets"] == 2, rule
        assert torch.allclose(out["sheet_cov"], torch.tensor([0.80, 0.21])), rule
        assert [int(m) & MASK_ALL for m in out["sheet_msk"]] == [0b11011111, 0b00110000]
        wgt = out["sheet_wgt"]
        # Negative = "this band continues at the next sheet": the first sibling
        # claims against the undimmed visibility and defers the band's single
        # occlusion write to the second.
        assert wgt[0] < 0, rule
        assert wgt[1] > 0, rule
        # Both carry the BAND's union, and their magnitudes sum to the band's
        # own coverage factor -- 1.0, the clamped 1.01 of a full union.
        assert [int(m) & MASK_ALL for m in out["sheet_wmsk"]] == [MASK_ALL, MASK_ALL]
        assert float(wgt.abs().sum()) == pytest.approx(1.0, abs=1e-5)

    # And the oracle then leaves nothing of the pixel for what is behind.
    from algan.rendering.raytracing.sheets import resolve_pixel_reference

    out = _compact(frags)
    _claims, T = resolve_pixel_reference(
        [float(w) for w in out["sheet_wgt"]],
        [int(m) for m in out["sheet_wmsk"]],
        [False, False],
    )
    assert max(T) == pytest.approx(0.0, abs=1e-5)


def test_a_partial_union_rank_split_is_never_pooled():
    # The pooling test needs a FULL union, and that is the half of it that
    # carries the argument. On a partial one -- a silhouette -- exact area and
    # sample count disagree by up to a whole sample cell for reasons that have
    # nothing to do with layering, so no threshold between them could tell one
    # layer from two. The two-layer fixture above is exactly that shape.
    frags = [
        (0, 1.0000, 0, 0.5, 0b00111100),
        (0, 1.0001, 1, 0.5, 0b00111100),
    ]
    out = _compact(frags)
    assert out["num_sheets"] == 2
    assert torch.equal(out["sheet_wgt"], out["sheet_cov"])
    assert torch.equal(out["sheet_wmsk"], out["sheet_msk"])


def test_an_interleaved_band_composites_sheet_by_sheet():
    # §4.4's arithmetic is a BAND's: every sibling takes the band's union and
    # its share of the band's coverage factor, and that is only paid back when
    # the deferral chain reaches the band's last sheet and writes the summed
    # occlusion there. Where another surface sits between them in the walk the
    # chain cannot close, so the band composites sheet by sheet on its own
    # areas and its own masks -- what it did before any split existed.
    frags = [
        (0, 1.0000, 0, 0.80, 0b11011111),
        (0, 1.0001, 4, 0.50, 0b11111111),  # surface 1, between the two
        (0, 1.0002, 1, 0.21, 0b00110000),
    ]
    out = _compact(frags, band_rule="facing")
    assert out["num_sheets"] == 3
    assert torch.equal(out["sheet_wgt"], out["sheet_cov"])
    assert torch.equal(out["sheet_wmsk"], out["sheet_msk"])


def test_sheet_rank_pool_setting_reaches_the_live_module():
    from algan import SETTINGS
    from algan.rendering.raytracing import sheets as sh

    frags = [
        (0, 1.0000, 0, 0.80, 0b11011111),
        (0, 1.0001, 1, 0.21, 0b00110000),
    ]
    old = sh.sheet_rank_pool
    try:
        SETTINGS.raytracing.experimental.set(sheet_rank_pool=False)
        assert sh.sheet_rank_pool is False
        out = _compact(frags)
        assert torch.equal(out["sheet_wgt"], out["sheet_cov"])
        assert torch.equal(out["sheet_wmsk"], out["sheet_msk"])
        SETTINGS.raytracing.experimental.set(sheet_rank_pool=True)
        assert sh.sheet_rank_pool is True
    finally:
        sh.sheet_rank_pool = old


def test_conflict_rank_kernel_agrees_with_torch_arm_through_compact_sheets():
    # The kernel arm of the conflict-rank scan must produce the SAME
    # compaction as the torch arm, on every array, not just the same sheet
    # count. The stream is conflict-heavy: five same-surface same-facing
    # fragments in ONE band (the facing rule never splits), whose masks
    # overlap enough to walk the unclamped ranks 0,1,2,3 past nothing -- plus
    # a donor (empty mask, rank 0) riding with the first owner.
    frags = [
        (3, 1.0000, 0, 0.30, 0b11110000),
        (3, 1.0001, 1, 0.25, 0b11110000),
        (3, 1.0002, 2, 0.20, 0b11110000),
        (3, 1.0003, 3, 0.15, 0b00111111),
        (3, 1.0004, 4, 0.10, 0),
    ]
    from algan import SETTINGS
    from algan.rendering.raytracing import settings as rt

    old = rt.sheet_rank_kernel
    try:
        SETTINGS.raytracing.experimental.set(sheet_rank_kernel=False)
        torch_arm = _compact(frags, band_rule="facing")
        SETTINGS.raytracing.experimental.set(sheet_rank_kernel=True)
        kernel_arm = _compact(frags, band_rule="facing")
    finally:
        rt.sheet_rank_kernel = old
    # The stream really is conflict-heavy: the ranks subdivide the band.
    assert torch_arm["num_sheets"] >= 3
    assert set(torch_arm) == set(kernel_arm)
    for key in sorted(torch_arm):
        a, b = torch_arm[key], kernel_arm[key]
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), key
        else:
            assert a == b, key


def test_conflict_rank_kernel_matches_the_torch_arm_including_a_clear_first_flag():
    # Direct two-arm check of sheets._conflict_rank on synthetic streams,
    # including THE LEADING-RUN CASE: when band_start[0] is clear the torch
    # arm's cummax makes the leading run one band starting at row 0, and the
    # kernel must walk that band rather than leave its rows unwritten
    # (compact_sheets itself never produces this input -- its first flag is
    # always set -- but the helper must agree on ANY input).
    from algan import SETTINGS
    from algan.rendering.raytracing import settings as rt
    from algan.rendering.raytracing.sheets import _conflict_rank

    def both_arms(band_start, order, msk):
        positions = torch.arange(msk.numel(), dtype=torch.int64)
        old = rt.sheet_rank_kernel
        try:
            SETTINGS.raytracing.experimental.set(sheet_rank_kernel=False)
            want = _conflict_rank(band_start, order, msk, positions)
            SETTINGS.raytracing.experimental.set(sheet_rank_kernel=True)
            got = _conflict_rank(band_start, order, msk, positions)
        finally:
            rt.sheet_rank_kernel = old
        return want, got

    gen = torch.Generator().manual_seed(19)
    n = 512
    msk = torch.randint(0, 1 << 8, (n,), generator=gen, dtype=torch.int32)
    order = torch.randperm(n, generator=gen)

    scattered = torch.rand(n, generator=gen) < 0.05
    scattered[0] = True
    want, got = both_arms(scattered, order, msk)
    assert want.dtype == torch.int32
    assert torch.equal(want, got)

    leading = scattered.clone()
    leading[:8] = False
    leading[0] = False
    want, got = both_arms(leading, order, msk)
    assert torch.equal(want, got)
    assert int(want[0]) == 0  # row 0 has no earlier fragments in its band


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
    capped, _T2 = resolve_pixel_reference(covs, msks, [False, False], caps=[0.25, 0.25])
    assert free[1] > 0.0  # the re-claim exists uncapped
    assert abs(capped[0] - 0.25) < 1e-12
    assert capped[1] == 0.0  # the ceiling leaves the far sheet no room
    # A ceiling of 2.0 is the "no ceiling" sentinel and must change nothing.
    sentinel, _T3 = resolve_pixel_reference(covs, msks, [False, False], caps=[2.0, 2.0])
    assert sentinel == free


def _norms(rows):
    """[1, 8, 9] vertex-normal table from per-triangle ``(n0, n1, n2)``."""
    tn = torch.zeros(1, 8, 9)
    for r, (n0, n1, n2) in rows.items():
        tn[0, r] = torch.tensor([*n0, *n1, *n2], dtype=torch.float32)
    return tn


def test_shade_split_separates_flat_faces_at_a_crease():
    # Two flat-shaded triangles of ONE surface tiling a pixel with a hard
    # normal discontinuity (a cube edge). Fused, the resolve shades the whole
    # pixel at the dominant face; split, each face is a sibling sheet shading
    # with its own normal -- the interior-edge AA case.
    frags = [
        (3, 1.0, 0, 0.4, 0b00001111),
        (3, 1.0001, 1, 0.6, 0b11110000),
    ]
    z = (0.0, 0.0, 1.0)
    x = (1.0, 0.0, 0.0)
    tn = _norms({0: (z, z, z), 1: (x, x, x)})
    fused = _compact(frags, tri_norm=tn)
    assert fused["num_sheets"] == 1
    split = _compact(frags, shade_split=True, tri_norm=tn)
    assert split["num_sheets"] == 2
    assert not bool(split["sheet_fused"].any())
    # Siblings keep their own exact areas and shading references.
    assert sorted(split["sheet_cov"].tolist()) == [
        pytest.approx(0.4),
        pytest.approx(0.6),
    ]
    assert sorted(split["sheet_ref"].tolist()) == [0, 1]


def _resolve_band(out, alpha=1.0):
    """Resolve one pixel's sheets through the oracle on the WEIGHTS the kernel
    consumes, returning ``(total claim, per-sheet claims)``.
    """
    from algan.rendering.raytracing.sheets import resolve_pixel_reference

    covs = out["sheet_wgt"].tolist()
    msks = [int(v) for v in out["sheet_wmsk"].tolist()]
    claims, _ = resolve_pixel_reference(
        covs, msks, [False] * len(covs), alphas=[alpha] * len(covs)
    )
    return sum(claims), claims


def test_shade_split_siblings_composite_additively():
    # §4.4: siblings claim by exact area against the SAME incoming
    # visibility, and the band occludes once by the summed claim -- so a
    # split band commits exactly the coverage the unsplit one did. Walked as
    # independent occluders they would not: the first sibling's write dims
    # the samples the second reads and the band under-claims, which is what
    # let the geometry behind an interior crease show through as a seam.
    z = (0.0, 0.0, 1.0)
    x = (1.0, 0.0, 0.0)
    tn = _norms({0: (z, z, z), 1: (x, x, x)})
    cases = (
        # partitioned samples, areas matching them
        [(3, 1.0, 0, 0.5, 0b00001111), (3, 1.0001, 1, 0.5, 0b11110000)],
        # partitioned samples, areas NOT matching them (corr > 1 on the
        # nearer sibling -- rule B's residue used to land on its co-sibling)
        [(3, 1.0, 0, 0.77, 0b11010111), (3, 1.0001, 1, 0.23, 0b00101000)],
        # a DONOR sibling: real area, no samples of its own
        [(3, 1.0, 0, 0.93, MASK_ALL), (3, 1.0001, 1, 0.07, 0)],
        # donor first
        [(3, 1.0, 0, 0.02, 0), (3, 1.0001, 1, 0.98, MASK_ALL)],
    )
    for frags in cases:
        fused = _compact(frags, tri_norm=tn)
        split = _compact(frags, shade_split=True, tri_norm=tn)
        assert fused["num_sheets"] == 1, frags
        assert split["num_sheets"] == 2, frags
        # At every material alpha, including translucent: the band's write is
        # the unsplit band's, so what it commits cannot depend on the split.
        for alpha in (1.0, 0.6, 0.25):
            whole, _ = _resolve_band(fused, alpha)
            total, claims = _resolve_band(split, alpha)
            assert total == pytest.approx(whole, abs=1e-6), (frags, alpha)
            # And each sibling claims its own exact share of that coverage.
            area = sum(f[3] for f in frags)
            for claim, cov in zip(claims, split["sheet_cov"].tolist()):
                assert claim == pytest.approx(whole * cov / area, abs=1e-6)


def test_shade_split_leaves_an_areal_band_whole():
    # An areal band -- every fragment position-less -- has no samples for
    # siblings to blend across and nothing to anti-alias, so it stays whole.
    z = (0.0, 0.0, 1.0)
    x = (1.0, 0.0, 0.0)
    tn = _norms({0: (z, z, z), 1: (x, x, x)})
    donors = [(3, 1.0, 0, 0.3, 0), (3, 1.0001, 1, 0.2, 0)]
    assert _compact(donors, shade_split=True, tri_norm=tn)["num_sheets"] == 1
    # A sub-sample rod (corr > 1) DOES split: the band's write is made whole
    # at its last sheet, so rule B's residue still lands on unowned samples.
    rod = [(3, 1.0, 0, 0.3, 0b00000001), (3, 1.0001, 1, 0.2, 0b00000010)]
    fused = _compact(rod, tri_norm=tn)
    split = _compact(rod, shade_split=True, tri_norm=tn)
    assert split["num_sheets"] == 2
    whole, _ = _resolve_band(fused)
    total, _ = _resolve_band(split)
    assert total == pytest.approx(whole, abs=1e-6)


def test_weights_are_the_record_where_a_band_holds_one_sheet():
    # Everything the split does not touch hands the resolve its own area and
    # mask -- the byte-identity the flag-off path rests on.
    frags = [
        (3, 1.0, 0, 0.4, 0b00001111),
        (3, 1.0001, 1, 0.6, 0b11110000),
        (5, 1.0, 4, 0.25, 0b00000011),
    ]
    for flag in (False, True):
        out = _compact(frags, shade_split=flag)
        assert torch.equal(out["sheet_wgt"], out["sheet_cov"]), flag
        assert torch.equal(out["sheet_wmsk"], out["sheet_msk"]), flag


def test_shade_split_uses_the_geometric_normal_for_zero_vertex_normals():
    # The Polyhedron family authors NO vertex normals (all zero); the shade
    # kernel substitutes the geometric cross-product normal, so the class
    # must too. The helper's flat-on triangles share the geometric normal
    # +z, so a crease has to come from reshaping one of them.
    frags = [
        (3, 1.0, 0, 0.4, 0b00001111),
        (3, 1.0001, 1, 0.6, 0b11110000),
    ]
    zeros = (0.0, 0.0, 0.0)
    tn = _norms({0: (zeros, zeros, zeros), 1: (zeros, zeros, zeros)})
    coverage, merged, cam, pws = _coverage(frags, tri_norm=tn)
    # Tilt triangle 1 so its geometric normal leaves +z.
    merged["tri_pos"][0, 1] = torch.tensor([0.0, 0, 1.0, 0.05, 0, 1.05, 0, 0.05, 1.0])
    kw = {"time_start": 0, "width": 4, "height": 4, "band_rule": "facing"}
    fused = compact_sheets(coverage, merged, cam, pws, **kw)
    assert fused["num_sheets"] == 1
    split = compact_sheets(coverage, merged, cam, pws, shade_split=True, **kw)
    assert split["num_sheets"] == 2
    # And two COPLANAR zero-normal triangles (one planar face) stay one sheet.
    tn0 = _norms({0: (zeros, zeros, zeros), 1: (zeros, zeros, zeros)})
    same = _compact(frags, shade_split=True, tri_norm=tn0)
    assert same["num_sheets"] == 1


def test_shade_split_keeps_smooth_triangles_in_one_sheet():
    # Vertex normals VARY across each triangle (diced curved geometry):
    # class 0 everywhere, so the toggle must not change the compaction.
    frags = [
        (3, 1.0, 0, 0.4, 0b00001111),
        (3, 1.0001, 1, 0.6, 0b11110000),
    ]
    z = (0.0, 0.0, 1.0)
    tilt_a = (0.0995, 0.0, 0.995)
    tilt_b = (0.0, 0.0995, 0.995)
    tn = _norms({0: (z, tilt_a, tilt_b), 1: (tilt_a, z, tilt_b)})
    for flag in (False, True):
        out = _compact(frags, shade_split=flag, tri_norm=tn)
        assert out["num_sheets"] == 1, flag
        assert int(out["sheet_ref"][0]) == 1, flag


def test_shade_split_without_normals_table_is_inert():
    # A merged scene with no tri_norm cannot classify: the split must not
    # invent classes (everything stays class 0 / grouped as before).
    frags = [
        (3, 1.0, 0, 0.4, 0b00001111),
        (3, 1.0001, 1, 0.6, 0b11110000),
    ]
    out = _compact(frags, shade_split=True)
    assert out["num_sheets"] == 1


def test_sheet_shade_split_setting_reaches_the_live_module():
    from algan import SETTINGS
    from algan.rendering.raytracing import settings as rt

    old = rt.sheet_shade_split
    try:
        SETTINGS.raytracing.experimental.set(sheet_shade_split=True)
        assert rt.sheet_shade_split is True
        SETTINGS.raytracing.experimental.set(sheet_shade_split=False)
        assert rt.sheet_shade_split is False
    finally:
        rt.sheet_shade_split = old


def test_sheet_resolve_setting_reaches_the_live_module():
    from algan import SETTINGS
    from algan.rendering.raytracing import settings as rt

    old = rt.sheet_resolve
    try:
        SETTINGS.raytracing.experimental.set(sheet_resolve=True)
        assert rt.sheet_resolve is True
        SETTINGS.raytracing.experimental.set(sheet_resolve=False)
        assert rt.sheet_resolve is False
    finally:
        rt.sheet_resolve = old


def test_sheet_rank_kernel_setting_reaches_the_live_module():
    from algan import SETTINGS
    from algan.rendering.raytracing import settings as rt

    old = rt.sheet_rank_kernel
    try:
        SETTINGS.raytracing.experimental.set(sheet_rank_kernel=True)
        assert rt.sheet_rank_kernel is True
        SETTINGS.raytracing.experimental.set(sheet_rank_kernel=False)
        assert rt.sheet_rank_kernel is False
    finally:
        rt.sheet_rank_kernel = old


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


# ---------------------------------------------------------------------------
# Positioned-fragment depth (rt_settings.sheet_positioned_depth)
# ---------------------------------------------------------------------------
#
# Two surfaces crossing inside one pixel, in the shape the axis triad of
# ``tests/full_renders/scenes/solids_and_camera.py`` makes: an Arrow3D shaft
# buried in a Dot3D and punching out through it. Both sheets end up claiming
# the whole pixel, so which one paints it is decided entirely by which sheet
# sorts first -- and the shaft's only claim to being in front is a sample-less
# AREA DONOR at the leading corner of the pixel. The sphere is nearer at every
# sample the resolve actually compares them at.
_CROSSING = [
    # surface 0 (refs 0-3): a donor with no samples, then the real thing,
    # which is BEHIND surface 1 at both of the sample groups it owns.
    (0, 1.0000, 0, 0.02, 0),
    # surface 1 (refs 4-7): nearer at every sample.
    (0, 1.0050, 4, 0.50, 0b00001111),
    (0, 1.0060, 5, 0.50, 0b11110000),
    (0, 1.0120, 1, 0.49, 0b00001111),
    (0, 1.0130, 2, 0.49, 0b11110000),
]


def test_a_sampleless_donor_does_not_decide_which_surface_paints_the_pixel():
    out = _compact(_CROSSING, band_rule="facing")
    assert out["num_sheets"] == 2
    # Surface 1 sorts first: its nearest POSITIONED fragment (1.005) leads
    # surface 0's (1.012). The 1.0 donor no longer speaks for surface 0.
    assert [int(r) for r in out["sheet_ref"]] == [4, 1]
    assert _t(out, 0) == pytest.approx(1.005)
    assert _t(out, 1) == pytest.approx(1.012)


def test_the_legacy_arm_lets_the_donor_decide():
    # The defect, pinned so the toggle is known to reach the behaviour and not
    # merely to exist: off, surface 0 leads on a fragment that owns no sample.
    out = _compact(_CROSSING, band_rule="facing", positioned_depth=False)
    assert [int(r) for r in out["sheet_ref"]] == [1, 4]
    assert _t(out, 0) == pytest.approx(1.0)


def test_a_donor_only_sheet_keeps_its_donor_depth():
    # Nothing better exists for a position-less sheet, so the fallback must
    # keep the nearest donor rather than leaving the sheet unordered.
    out = _compact(
        [
            (0, 1.0000, 0, 0.20, 0),
            (0, 1.0010, 1, 0.10, 0),
            (0, 1.0050, 4, 1.00, MASK_ALL),
        ],
        band_rule="facing",
    )
    assert out["num_sheets"] == 2
    assert _t(out, 0) == pytest.approx(1.0)
    assert int(out["sheet_msk"][0]) & SLIVER != 0


def test_positioned_depth_is_inert_when_every_fragment_owns_a_sample():
    # The common case must be byte-identical, which is what bounds the change:
    # only a sheet whose nearest fragment is a donor can move.
    frags = [
        (0, 1.0000, 0, 0.40, 0b00001111),
        (0, 1.0010, 1, 0.60, 0b11110000),
        (0, 1.0050, 4, 1.00, MASK_ALL),
    ]
    on = _compact(frags, band_rule="facing")
    off = _compact(frags, band_rule="facing", positioned_depth=False)
    for field in ("sheet_key", "sheet_ref", "sheet_cov", "sheet_msk", "sheet_wgt"):
        assert torch.equal(on[field], off[field]), field


def test_sheet_positioned_depth_setting_reaches_the_live_module():
    from algan import SETTINGS
    from algan.rendering.raytracing import settings as rt

    old = rt.sheet_positioned_depth
    try:
        SETTINGS.raytracing.experimental.set(sheet_positioned_depth=False)
        assert rt.sheet_positioned_depth is False
        SETTINGS.raytracing.experimental.set(sheet_positioned_depth=True)
        assert rt.sheet_positioned_depth is True
    finally:
        rt.sheet_positioned_depth = old


# ---------------------------------------------------------------------------
# Per-sample depth gating (rt_settings.sheet_sample_depth)
# ---------------------------------------------------------------------------
#
# Two surfaces crossing inside one pixel, both claiming the whole sample
# union at full exact coverage (the DESIGN_sheet_resolve.md ss6.1.1 case the
# walk order cannot resolve: one scalar depth per sheet decides the pixel).
# Per-sample nearest-owner depths interleave: surface 1 is strictly nearer at
# samples {0,1,2}, surface 0 at {3..7}. The MAT_OPAQUE bit is what the
# emission pipeline folds into fragment masks for material-opaque triangles.
_CROSSING_FULL = [
    # surface 0 (refs 0-3), nearest fragment leads the stream
    (0, 1.0000, 0, 0.375, 0b11100000),
    # surface 1 (refs 4-7)
    (0, 1.0010, 4, 0.375, 0b00000111),
    (0, 1.0020, 1, 0.625, 0b00011111),
    (0, 1.0030, 5, 0.625, 0b11111000),
]


def _opaque(frags):
    return [(p, t, r, c, m | MAT_OPAQUE) for p, t, r, c, m in frags]


def _lose(msk_word):
    """A mask word's per-sample lose bits."""
    return (int(msk_word) >> LOSE_SHIFT) & MASK_ALL


def test_interpenetrating_full_union_sheets_cede_strictly_nearer_samples():
    out = _compact(_opaque(_CROSSING_FULL), band_rule="facing", sample_depth=True)
    assert out["num_sheets"] == 2
    # The walk order itself is untouched: surface 0's nearest fragment still
    # leads, so sheet 0 is surface 0 and sheet 1 is surface 1.
    assert _t(out, 0) == pytest.approx(1.000)
    assert _t(out, 1) == pytest.approx(1.001)
    # Each sheet carries the exact per-sample cession mask: surface 0 loses
    # exactly {0,1,2} where surface 1 is strictly nearer; surface 1 loses
    # exactly {3..7}.
    assert _lose(out["sheet_msk"][0]) == 0b00000111
    assert _lose(out["sheet_msk"][1]) == 0b11111000
    # The weights the kernel consumes carry the same bits.
    assert _lose(out["sheet_wmsk"][0]) == 0b00000111
    assert _lose(out["sheet_wmsk"][1]) == 0b11111000
    # Resolved through the oracle, the claims become winner-per-sample:
    # 5/8 + 3/8 instead of one surface taking all eight samples.
    total, claims = _resolve_band(out)
    assert total == pytest.approx(1.0)
    assert claims[0] == pytest.approx(0.625)
    assert claims[1] == pytest.approx(0.375)


def test_a_translucent_sheet_sets_no_floor():
    # Without the MAT_OPAQUE bit surface 0 is no enforcer, so surface 1 finds
    # nothing to cede {3..7} to -- while surface 0 is still gated by surface
    # 1's own full-union opaque sheet.
    frags = [
        (0, 1.0000, 0, 0.375, 0b11100000),
        (0, 1.0010, 4, 0.375, 0b00000111 | MAT_OPAQUE),
        (0, 1.0020, 1, 0.625, 0b00011111),
        (0, 1.0030, 5, 0.625, 0b11111000 | MAT_OPAQUE),
    ]
    out = _compact(frags, band_rule="facing", sample_depth=True)
    assert _lose(out["sheet_msk"][0]) == 0b00000111
    assert _lose(out["sheet_msk"][1]) == 0


def test_a_partial_coverage_sheet_is_no_enforcer():
    # Surface 0 covers only 0.95 of the pixel: outside the full-coverage dust
    # band it must set no floor for surface 1, whatever its sample union says.
    frags = [
        (0, 1.0000, 0, 0.35, 0b11100000),
        (0, 1.0010, 4, 0.375, 0b00000111),
        (0, 1.0020, 1, 0.60, 0b00011111),
        (0, 1.0030, 5, 0.625, 0b11111000),
    ]
    out = _compact(_opaque(frags), band_rule="facing", sample_depth=True)
    assert _t(out, 0) == pytest.approx(1.000)
    # Sheet 0 is surface 0 (its nearest fragment leads). Had its 0.95 area
    # made it an enforcer, surface 1 would have lost {3..7} to it; being
    # merely a subject it still cedes its own {0,1,2} to surface 1.
    assert _lose(out["sheet_msk"][0]) == 0b00000111
    assert _lose(out["sheet_msk"][1]) == 0


def test_same_surface_sheets_never_gate_each_other():
    # Front and back shells of ONE solid: both would be enforcers and both
    # subjects, but a floor only binds across DIFFERENT surfaces.
    out = _compact(
        _opaque(
            [
                (0, 1.0, 0, 1.0, MASK_ALL),
                (0, 2.0, 1, 1.0, BACKFACE | MASK_ALL),
            ]
        ),
        band_rule="facing",
        sample_depth=True,
    )
    assert out["num_sheets"] == 2
    assert _lose(out["sheet_msk"][0]) == 0
    assert _lose(out["sheet_msk"][1]) == 0


def test_an_areal_sheet_gets_no_lose_bits():
    # A donor-only sheet is position-less; the gate never applies to it even
    # though the other surface is strictly nearer at every sample.
    out = _compact(
        _opaque(
            [
                (0, 1.0, 4, 1.0, MASK_ALL),
                (0, 2.0, 0, 0.30, 0),
                (0, 2.0002, 1, 0.10, 0),
            ]
        ),
        band_rule="facing",
        sample_depth=True,
    )
    assert out["num_sheets"] == 2
    assert int(out["sheet_msk"][1]) & SLIVER != 0
    assert _lose(out["sheet_msk"][1]) == 0


def test_ties_within_the_epsilon_set_no_bits():
    # Strictly-nearer with margin: a 5e-5 depth gap keeps today's walk order
    # on BOTH sheets.
    out = _compact(
        _opaque(
            [
                (0, 1.0000, 4, 1.0, MASK_ALL),
                (0, 1.00005, 0, 1.0, MASK_ALL),
            ]
        ),
        band_rule="facing",
        sample_depth=True,
    )
    assert out["num_sheets"] == 2
    assert _lose(out["sheet_msk"][0]) == 0
    assert _lose(out["sheet_msk"][1]) == 0


def test_setting_off_leaves_every_output_at_today_s_semantics():
    on = _compact(_opaque(_CROSSING_FULL), band_rule="facing", sample_depth=True)
    off = _compact(_opaque(_CROSSING_FULL), band_rule="facing", sample_depth=False)
    for field in (
        "sheet_key",
        "sheet_ref",
        "sheet_cov",
        "sheet_wgt",
        "sheet_nfrag",
        "sheet_fused",
    ):
        assert torch.equal(on[field], off[field]), field
    for i in range(int(on["num_sheets"])):
        assert _lose(off["sheet_msk"][i]) == 0
        # The ONLY difference in either mask word is its lose bits.
        assert int(off["sheet_msk"][i]) == int(on["sheet_msk"][i]) ^ (
            _lose(on["sheet_msk"][i]) << LOSE_SHIFT
        )


def test_multi_sheet_bands_neither_cede_nor_floor():
    # The deferral exemption: a band split into shade-class siblings claims
    # against band-pooled arithmetic whose single occlusion write ignores
    # slots, so its sheets are exempt on BOTH sides -- they neither receive
    # lose bits from the nearer other-surface enforcer nor act as one for it.
    z = (0.0, 0.0, 1.0)
    x = (1.0, 0.0, 0.0)
    tn = _norms({0: (z, z, z), 1: (x, x, x), 4: (z, z, z)})
    frags = [
        # surface 1: near full-union opaque enforcer, strictly nearer than
        # every sample the crease siblings own
        (0, 0.5, 4, 1.0, MASK_ALL),
        # surface 0 crease pair: one band, TWO sibling sheets
        (0, 1.001, 0, 0.5, 0b00001111),
        (0, 1.002, 1, 0.5, 0b11110000),
    ]
    split = _compact(
        _opaque(frags),
        band_rule="facing",
        shade_split=True,
        tri_norm=tn,
        sample_depth=True,
    )
    assert split["num_sheets"] == 3
    for i in range(3):
        assert _lose(split["sheet_msk"][i]) == 0, i


def test_a_sheet_cedes_everything_it_loses_or_nothing():
    # A lane's depth is its FRAGMENT's centroid depth, not the lane's own, so a
    # sheet losing only a thin share is reading the margin least entitled to
    # decide -- and ceding there measurably regressed pixels the sheet already
    # won. Surface 0 spans the pixel in two fragments (lanes 0-5 near, lanes
    # 6-7 far) and surface 1 sits between them at every lane, so each is an
    # enforcer for the other and the two cede in opposite proportions: surface
    # 0 loses 2 of 8, exactly the 0.25 floor and so NOT more than it, and cedes
    # nothing; surface 1 loses 6 of 8 and cedes all six.
    frags = [
        (0, 1.000, 0, 0.75, 0b00111111),
        (0, 1.005, 4, 1.00, 0b11111111),
        (0, 1.010, 1, 0.25, 0b11000000),
    ]
    out = _compact(_opaque(frags), band_rule="facing", sample_depth=True)
    assert out["num_sheets"] == 2
    assert _t(out, 0) == pytest.approx(1.000)
    assert _lose(out["sheet_msk"][0]) == 0
    assert _lose(out["sheet_msk"][1]) == 0b00111111


def test_sheet_sample_depth_setting_reaches_the_live_module():
    from algan import SETTINGS
    from algan.rendering.raytracing import settings as rt

    old = rt.sheet_sample_depth
    try:
        SETTINGS.raytracing.experimental.set(sheet_sample_depth=False)
        assert rt.sheet_sample_depth is False
        SETTINGS.raytracing.experimental.set(sheet_sample_depth=True)
        assert rt.sheet_sample_depth is True
    finally:
        rt.sheet_sample_depth = old
