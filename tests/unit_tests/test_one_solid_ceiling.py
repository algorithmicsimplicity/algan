"""The one-mesh coverage ceiling on solids the walk refracts.

The resolve caps a single surface's total claim on a pixel at the larger of
its two sheets' exact areas (``sheets.resolve_pixel_reference``,
DESIGN_sheet_resolve.md §7), so a closed solid's far sheet cannot re-claim the
near sheet's ``corr < 1`` residue: what a partially covering sheet's samples
say beyond its exact area belongs to whatever lies BESIDE the solid. That cap
used to bind on opaque meshes only. Clear glass is not opaque, so its far
sheet claimed the residue and spent it on a ray through the solid's back face,
which at a grazing silhouette totally reflects into the sphere: the faint dark
rim left round a clear sphere on white after the R1 fix.

A pixel of one surface of any material now carries ``_AA_ONE_SOLID_BIT`` where
the batch's primaries can refract, and the ceiling binds there once the walk
has refracted at a full-alpha sheet of the solid, whose continuation carries
everything behind it. These pin that rule on hand-built inputs, at the oracle
and at the emission; the render tests (the rim, the kernel against this oracle
on the refracting variant, and solids that must render exactly as before) are
in ``test_sheet_resolve_shading.py``.

Pure tensor assertions and the emission's two small reduction kernels; no
render. Not ``fast``: nothing the timeline does can break them.
"""

from __future__ import annotations

import pytest
import torch

from algan.rendering.raytracing import raster_pipeline
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.raster_taichi import (
    _AA_BACKFACE_BIT as BACKFACE,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_NUM_SAMPLES as N,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_ONE_MESH_BIT as ONE_MESH,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_ONE_SOLID_BIT as ONE_SOLID,
)
from algan.rendering.raytracing.sheets import resolve_pixel_reference

# ------------------------------------------------------------------- oracle

#: A silhouette pixel of the clear sphere in ``test_sheet_resolve_shading``
#: (192x108, pixel (88, 71)), as the compaction hands it to the resolve: the
#: near sheet and the far one on the same three of eight samples, the far one
#: a little larger, so the ceiling is its area.
NEAR, FAR = 0.35797, 0.35910
SAMPLES = 0b10001001
CAP = max(NEAR, FAR)


def _silhouette(flags=ONE_SOLID, alphas=(1.0, 1.0), refracts=(True, True)):
    msks = [flags | SAMPLES, flags | BACKFACE | SAMPLES]
    return resolve_pixel_reference(
        [NEAR, FAR],
        msks,
        [False, False],
        list(alphas),
        caps=[CAP, CAP],
        refracts=None if refracts is None else list(refracts),
    )


def test_a_refracted_near_sheet_leaves_the_far_sheet_only_the_ceiling():
    """Behind clear glass the far sheet gets ``cap - near`` and no more.

    Uncapped, the near sheet's ``corr = NEAR / (3/8) < 1`` leaves each of its
    samples ``1 - corr`` of their transmittance, and the far sheet, on the same
    samples, claims ``FAR * (1 - corr)`` of it: 0.0163 of the pixel, which the
    walk spent on a totally reflected ray into the sphere. Capped, it may claim
    only what its larger footprint adds, and the rest is the background's.
    """
    claims, T = _silhouette()
    assert claims[0] == pytest.approx(NEAR, abs=1e-12)
    assert claims[1] == pytest.approx(CAP - NEAR, abs=1e-9)

    free, T_free = _silhouette(refracts=None)
    corr = NEAR / (bin(SAMPLES).count("1") / N)
    assert free[1] == pytest.approx(FAR * (1.0 - corr), abs=1e-9)
    assert free[1] > 10 * claims[1]
    # What the far sheet no longer claims is left for the background.
    assert sum(T) / N - sum(T_free) / N == pytest.approx(free[1] - claims[1], abs=1e-9)


@pytest.mark.parametrize(
    "case",
    [
        # A transmitter the walk does not refract: translucent, or glass with
        # refraction off, no bounces left, or at an index of 1.
        {"refracts": None},
        {"refracts": (False, False)},
        # Glass below full alpha: its far side shows through what it leaves.
        {"alphas": (0.6, 0.6)},
        # A translucent near sheet in front of a refracted far one.
        {"alphas": (0.5, 1.0), "refracts": (False, True)},
    ],
    ids=["no refraction", "unrefracted", "glass at alpha 0.6", "translucent in front"],
)
def test_a_solid_nothing_seals_composites_as_without_the_flag(case):
    """The flag alone binds nothing: before a full-alpha refracted sheet, the
    solid's far side is legitimately visible THROUGH its near one, so the
    pixel composites exactly as it did before the one-solid flag existed.
    """
    flagged = _silhouette(**case)
    plain = _silhouette(flags=0, **case)
    assert flagged == plain


def test_without_the_flag_a_refracted_sheet_caps_nothing():
    """A pixel the host did not flag (two surfaces, a stroke, or a batch whose
    primaries cannot refract) keeps every sheet's claim; refraction alone
    only makes the sheet occlude as opaque.
    """
    claims, _T = _silhouette(flags=0)
    free, _T_free = _silhouette(flags=0, refracts=(False, False))
    assert claims == free


def test_a_refracted_sheet_occludes_as_opaque():
    """A refracted sheet passes its light on through rays of its own, so its
    transmitted share does not reach the sheets behind it: the kernel's
    ``ts_w = 0`` on the glass path.
    """
    covs, msks = [1.0, 1.0], [0xFF, 0xFF]
    seen, _T = resolve_pixel_reference(covs, msks, [False] * 2, trans=[0.7, 0.0])
    glass, _T = resolve_pixel_reference(
        covs, msks, [False] * 2, trans=[0.7, 0.0], refracts=[True, False]
    )
    assert seen[1] == pytest.approx(0.7, abs=1e-12)
    assert glass == [1.0, 0.0]


def test_one_mesh_pixels_keep_their_ceiling_under_both_flags():
    """An opaque mesh's pixels carry both flags where primaries refract; the
    one-mesh bit binds as before, refracted or not.
    """
    both = _silhouette(flags=ONE_MESH | ONE_SOLID, refracts=None)
    one_mesh = _silhouette(flags=ONE_MESH, refracts=None)
    assert both == one_mesh
    assert both[0][1] == pytest.approx(CAP - NEAR, abs=1e-9)


# ----------------------------------------------------------------- emission

#: Triangles 0-3 are surface 0 and 4-7 surface 1.
TRI_OBJ = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.int32)
PPF = 16

#: ``(pixel, ref, area, backface, opaque)`` rows in stream order. A ref below 0
#: is a circuit, which has no surface id.
STREAM = [
    # 0: an opaque solid's two sheets -- one mesh, and one solid.
    (0, 0, 0.30, False, True),
    (0, 1, 0.32, True, True),
    # 1: a glass solid's two sheets -- one solid only.
    (1, 2, 0.40, False, False),
    (1, 3, 0.38, True, False),
    # 2: two surfaces -- neither.
    (2, 2, 0.50, False, False),
    (2, 4, 0.50, False, False),
    # 3: a circuit over the solid -- neither.
    (3, -5, 0.20, False, False),
    (3, 2, 0.60, False, False),
    # 4: one surface, opaque and transmissive faces -- one solid only.
    (4, 0, 0.25, False, True),
    (4, 3, 0.25, True, False),
]


def _stream():
    def key(pix, j):
        depth = int(torch.tensor([1.0 + 0.01 * j]).view(torch.int32)[0])
        return (pix << 32) | depth

    key_s = torch.tensor([key(r[0], j) for j, r in enumerate(STREAM)])
    ref_s = torch.tensor([r[1] for r in STREAM], dtype=torch.int32)
    cov_s = torch.tensor([r[2] for r in STREAM], dtype=torch.float32)
    msk_s = torch.tensor(
        [0b111 | (BACKFACE if r[3] else 0) for r in STREAM], dtype=torch.int32
    )
    opaque_s = torch.tensor([r[4] for r in STREAM])
    _pix, counts = torch.unique_consecutive(
        torch.tensor([r[0] for r in STREAM]), return_counts=True
    )
    return key_s, ref_s, cov_s, msk_s, opaque_s, counts


def _caps(kernel, solid, monkeypatch):
    monkeypatch.setattr(rt_settings, "sheet_one_mesh_kernel", kernel)
    key_s, ref_s, cov_s, msk_s, opaque_s, counts = _stream()
    msk, cap = raster_pipeline._one_mesh_pixel_caps(
        key_s, ref_s, cov_s, msk_s, opaque_s, counts, TRI_OBJ, PPF, 0, solid=solid
    )
    rows = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    return msk, cap, rows


def _per_pixel(values, rows):
    """One value per pixel, after checking every fragment of it agrees."""
    out = []
    for p in range(int(rows.max()) + 1):
        vals = set(values[rows == p].tolist())
        assert len(vals) == 1, f"pixel {p}'s fragments disagree: {vals}"
        out.append(vals.pop())
    return out


@pytest.mark.parametrize("kernel", [True, False], ids=["kernel arm", "torch arm"])
def test_the_emission_flags_one_solid_pixels_only_where_primaries_refract(
    kernel, monkeypatch
):
    msk, cap, rows = _caps(kernel, True, monkeypatch)
    one_mesh = _per_pixel((msk & ONE_MESH) != 0, rows)
    one_solid = _per_pixel((msk & ONE_SOLID) != 0, rows)
    caps = _per_pixel(cap, rows)
    assert one_mesh == [True, False, False, False, False]
    assert one_solid == [True, True, False, False, True]
    # The ceiling is max(front, back) wherever either flag is set, and the 2.0
    # "no ceiling" sentinel elsewhere.
    expected = [0.32, 0.40, 2.0, 2.0, 0.25]
    assert caps == pytest.approx(expected, abs=1e-7)

    # A batch whose primaries cannot refract flags no solid, and every other
    # value is what the emission wrote before the flag existed.
    msk_off, cap_off, _rows = _caps(kernel, False, monkeypatch)
    assert not bool(((msk_off & ONE_SOLID) != 0).any())
    assert torch.equal(msk_off, msk & ~ONE_SOLID)
    assert _per_pixel(cap_off, rows) == pytest.approx(
        [0.32, 2.0, 2.0, 2.0, 2.0], abs=1e-7
    )


@pytest.mark.parametrize("solid", [True, False], ids=["solid", "no solid"])
def test_both_emission_arms_agree_bit_for_bit(solid, monkeypatch):
    k_msk, k_cap, _rows = _caps(True, solid, monkeypatch)
    t_msk, t_cap, _rows = _caps(False, solid, monkeypatch)
    assert torch.equal(k_msk, t_msk)
    assert torch.equal(k_cap.view(torch.int32), t_cap.view(torch.int32))
