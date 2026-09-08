"""Unbroken sibling runs preserve coverage; interleaved runs stay independent."""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.sheets import (
    AA_MASK_ALL,
    AA_ONE_MESH_BIT,
    AA_SLIVER_BIT,
    _sibling_weights,
)
from algan.rendering.taichi_runtime import init_taichi
from algan.settings._startup import render_device


@pytest.mark.parametrize("friendly", [False, True])
@pytest.mark.parametrize("mask_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("pattern", ["empty", "single", "mixed", "long", "random"])
def test_sibling_kernel_matches_original(pattern, mask_dtype, friendly, monkeypatch):
    device = render_device()
    if device.type == "mps" and not friendly:
        pytest.skip("Metal has no wide float accumulator")
    monkeypatch.setattr(SETTINGS.computing, "mps_friendly", friendly)
    init_taichi()
    generator = torch.Generator().manual_seed(9835)
    patterns = {
        "empty": [],
        "single": [0],
        # Bands 0/3 are uninterrupted; band 1 is interleaved; band 2 is single.
        "mixed": [0, 0, 1, 2, 1, 3, 3, 3, 4],
        "long": [0] * 4097 + [1],
        "random": torch.randint(0, 67, (257,), generator=generator).tolist(),
    }
    ids = patterns[pattern]
    n, bands = len(ids), max(ids, default=-1) + 1
    band = torch.tensor(ids, dtype=torch.int64)
    cov = torch.rand(n, generator=generator)
    mask = torch.randint(
        0, AA_MASK_ALL + 1, (n,), generator=generator, dtype=mask_dtype
    )
    mask |= AA_SLIVER_BIT | AA_ONE_MESH_BIT
    area = torch.zeros(bands).scatter_add_(0, band, cov)
    correction = torch.rand(bands, generator=generator)
    union = torch.tensor(
        [AA_MASK_ALL, 0, 1, 0x55, 0x81] * ((bands + 4) // 5), dtype=mask_dtype
    )[:bands]
    # Zero area with nonzero correction catches the tiny divide-floor hazard.
    if n > 3:
        cov[band == 0] = 0
        area[0] = 0
    args = [value.to(device) for value in (band, cov, mask, area, union, correction)]
    before = [value.clone() for value in args]
    retained = []
    for enabled in (False, True, False, True):
        monkeypatch.setattr(rt_settings, "sheet_sibling_weights_kernel", enabled)
        result = _sibling_weights(*args)
        retained.append(result)
    reference = retained[0]
    for weights, masks in retained[1:]:
        torch.testing.assert_close(weights, reference[0], rtol=2e-6, atol=1e-7)
        assert torch.equal(torch.signbit(weights), torch.signbit(reference[0]))
        assert torch.equal(masks, reference[1])
        assert bool(torch.isfinite(weights).all())
    for value, snapshot in zip(args, before, strict=True):
        assert torch.equal(value, snapshot)


def test_sibling_kernel_uses_band_union_only_for_uninterrupted_runs(monkeypatch):
    init_taichi()
    device = render_device()
    monkeypatch.setattr(rt_settings, "sheet_sibling_weights_kernel", True)
    # The independent oracle: the first pair shares its band's full coverage;
    # interleaved band 1 keeps its own mask/area, including the sliver flag.
    args = [
        torch.tensor([0, 0, 1, 2, 1], dtype=torch.int64, device=device),
        torch.tensor([0.25, 0.75, 0.2, 0.3, 0.4], device=device),
        torch.tensor(
            [1, AA_SLIVER_BIT, AA_SLIVER_BIT, 4, 8], dtype=torch.int32, device=device
        ),
        torch.tensor([1.0, 0.6, 0.3], device=device),
        torch.tensor([AA_MASK_ALL, AA_MASK_ALL, 4], dtype=torch.int32, device=device),
        torch.tensor([0.8, 1.0, 0.3], device=device),
    ]
    weights, masks = _sibling_weights(*args)
    torch.testing.assert_close(weights.cpu(), torch.tensor([-0.2, 0.6, 0.2, 0.3, 0.4]))
    assert masks.cpu().tolist() == [AA_MASK_ALL, AA_MASK_ALL, AA_SLIVER_BIT, 4, 8]


def test_sibling_kernel_accepts_offset_views_and_preserves_canaries(monkeypatch):
    from algan.rendering.mps_compat import taichi_accumulate_dtype
    from algan.rendering.raytracing.sheet_sibling_taichi import (
        sibling_band_counts,
        sibling_coverage_weights,
    )

    init_taichi()
    device = render_device()
    band_base = torch.tensor([-99, 0, 0, 1, -99], dtype=torch.int64, device=device)
    cov_base = torch.tensor([-99.0, 0.25, 0.75, 0.5, -99.0], device=device)
    mask_base = torch.tensor([-99, 1, 2, 4, -99], dtype=torch.int32, device=device)
    counts_base = torch.full((5, 2), -99, dtype=torch.int32, device=device)
    counts = counts_base[1:3]
    counts.zero_()
    weight_base = torch.full((5,), -99.0, device=device)
    result_mask_base = torch.full((5,), -99, dtype=torch.int32, device=device)
    area = torch.tensor([1.0, 0.5], device=device)
    union = torch.tensor([AA_MASK_ALL, 4], dtype=torch.int32, device=device)
    corr = torch.tensor([1.0, 0.5], device=device)
    band, cov, mask = band_base[1:4], cov_base[1:4], mask_base[1:4]
    sibling_band_counts(band, 3, counts)
    sibling_coverage_weights(
        band,
        cov,
        mask,
        area,
        union,
        corr,
        counts,
        3,
        weight_base[1:4],
        result_mask_base[1:4],
        taichi_accumulate_dtype(),
    )
    torch.testing.assert_close(
        weight_base.cpu(), torch.tensor([-99.0, -0.25, 0.75, 0.5, -99.0])
    )
    assert result_mask_base.cpu().tolist() == [-99, AA_MASK_ALL, AA_MASK_ALL, 4, -99]
    assert counts_base.cpu().tolist() == [
        [-99, -99],
        [2, 1],
        [1, 1],
        [-99, -99],
        [-99, -99],
    ]


@pytest.mark.parametrize("union", [0, 0x55, AA_MASK_ALL])
def test_sibling_tiny_area_floor_and_partial_union(union, monkeypatch):
    init_taichi()
    monkeypatch.setattr(SETTINGS.computing, "mps_friendly", True)
    device = render_device()
    args = [
        torch.tensor([0, 0], dtype=torch.int64, device=device),
        torch.tensor([1e-14, 2e-14], device=device),
        torch.tensor([AA_SLIVER_BIT, AA_SLIVER_BIT], dtype=torch.int32, device=device),
        torch.tensor([3e-14], device=device),
        torch.tensor([union], dtype=torch.int32, device=device),
        torch.tensor([1.0], device=device),
    ]
    monkeypatch.setattr(rt_settings, "sheet_sibling_weights_kernel", True)
    weights, masks = _sibling_weights(*args)
    factor = max(union.bit_count(), 1) / 8
    torch.testing.assert_close(weights.cpu(), torch.tensor([-0.01, 0.02]) * factor)
    assert masks.cpu().tolist() == [union, union]
