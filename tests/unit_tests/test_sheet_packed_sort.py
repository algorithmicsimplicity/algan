"""Packed keys preserve the full stable order, or retain the reference sort."""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing.sheets import (
    _key_depth_order,
    _lexsort,
    _packed_depth_order,
    _pixel_group_order,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "case",
    [
        "normal",
        "ties",
        "extreme_origin",
        "overflow",
        "negative",
        "negative_zero",
        "nan",
        "inf",
        "tiny",
        "empty",
        "strided",
        "positive_zero",
        "float64",
        "int32_keys",
    ],
)
def test_packed_order_matches_reference(device, case):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if case == "empty":
        n = 0
    elif case == "tiny":
        n = 8191
    else:
        n = 262145 if device == "cuda" else 8193
    ids = torch.arange(n, dtype=torch.int64, device=device)
    pix, group = ids % 11, ids * 7 % 13 - 6
    # Include adjacent float32 values: no depth quantization is permissible.
    depth = (ids % 17 + 0x3F800000).to(torch.int32).view(torch.float32)
    if case == "ties":
        pix.zero_()
        group.zero_()
        depth.fill_(1)
    elif case == "extreme_origin":
        pix += torch.iinfo(torch.int64).min
        group += torch.iinfo(torch.int64).max - 6
    elif case == "overflow":
        group[0], group[1] = torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max
    elif case == "negative":
        depth[0] = -1
    elif case == "negative_zero":
        depth[0] = -0.0
    elif case in ("nan", "inf"):
        depth[0] = float(case)
    elif case == "strided":
        depth = depth.repeat_interleave(2)[::2]
    elif case == "positive_zero":
        depth[0] = 0.0
    elif case == "float64":
        depth = depth.double()
    elif case == "int32_keys":
        pix, group = pix.int(), group.int()
    eligible = device == "cuda" and case in (
        "normal",
        "ties",
        "extreme_origin",
        "positive_zero",
    )
    with SETTINGS.raytracing.experimental.override(
        sheet_pixel_sort=False, sheet_packed_sort=True
    ):
        for keys in ((pix, group), (group,)):
            expected = _lexsort(*keys, depth)
            packed = _packed_depth_order(keys, depth)
            assert (packed is not None) == eligible
            if eligible:
                assert torch.equal(packed, expected)
            actual = (
                _pixel_group_order(pix, group, depth, None)
                if len(keys) == 2
                else _key_depth_order(group, depth)
            )
            assert torch.equal(actual, expected)


@pytest.mark.parametrize("pixel_bits", [30, 31])
def test_combined_capacity_is_checked_before_tensor_arithmetic(pixel_bits):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    ids = torch.arange(32768, dtype=torch.int64, device="cuda")
    pix = (ids % 2) * ((1 << pixel_bits) - 1)
    group = (ids // 2 % 2) * ((1 << 32) - 1)
    depth = torch.ones(32768, dtype=torch.float32, device="cuda")
    packed = _packed_depth_order((pix, group), depth)
    assert (packed is not None) == (pixel_bits == 30)
    if packed is not None:
        assert torch.equal(packed, _lexsort(pix, group, depth))


@pytest.mark.parametrize("columns", [1, 2])
@pytest.mark.parametrize("offset", [-1, 0])
def test_packing_starts_at_the_measured_queue_threshold(columns, offset):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    n = (32768 if columns == 2 else 262144) + offset
    ids = torch.arange(n, dtype=torch.int64, device="cuda")
    keys = (ids % 5,) if columns == 1 else (ids % 5, ids % 3)
    depth = torch.ones(n, dtype=torch.float32, device="cuda")
    result = _packed_depth_order(keys, depth)
    assert (result is not None) == (offset == 0)
    if result is not None:
        assert torch.equal(result, _lexsort(*keys, depth))


@pytest.mark.parametrize("bound", [31, 2**31, 2**40])
def test_bounded_fallback_preserves_stable_order_and_inputs(bound):
    from algan.settings._startup import render_device

    device = render_device()
    pix = torch.tensor([3, 1, 1, 3, 1, 3], dtype=torch.int64, device=device)
    group = torch.tensor([0, 7, 7, 0, -2, 0], dtype=torch.int64, device=device)
    depth = torch.tensor([1.0, 2.0, 2.0, 0.5, 4.0, 1.0], device=device)
    original = [v.clone() for v in (pix, group, depth)]
    with SETTINGS.raytracing.experimental.override(sheet_pixel_sort=False):
        actual = _pixel_group_order(pix, group, depth, None, key_bounds=(bound, bound))
    assert actual.cpu().tolist() == [4, 1, 2, 3, 0, 5]
    for before, after in zip(original, (pix, group, depth)):
        assert torch.equal(before, after)


def test_bounds_do_not_disable_the_existing_packed_sort(monkeypatch):
    from algan.rendering.raytracing import sheets

    pix = torch.tensor([0, 0, 1], dtype=torch.int64)
    group = torch.tensor([1, 0, 0], dtype=torch.int64)
    depth = torch.ones(3)
    expected = torch.tensor([1, 0, 2])

    def packed(keys, values):
        assert keys[0] is pix
        assert keys[1] is group
        assert values is depth
        return expected

    monkeypatch.setattr(sheets, "_packed_depth_order", packed)
    with SETTINGS.raytracing.experimental.override(
        sheet_pixel_sort=False, sheet_packed_sort=True
    ):
        actual = _pixel_group_order(pix, group, depth, None, key_bounds=(2, 2))
    assert actual is expected


def test_sorted_grouping_preserves_missing_and_repeated_ids_on_render_device():
    from algan.rendering.raytracing.sheets import _unique_sorted_ids
    from algan.settings._startup import render_device

    ids = torch.tensor([0, 0, 3, 3, 3, 8], dtype=torch.int64, device=render_device())
    unique, inverse = _unique_sorted_ids(ids)
    assert unique.cpu().tolist() == [0, 3, 8]
    assert inverse.cpu().tolist() == [0, 0, 1, 1, 1, 2]
