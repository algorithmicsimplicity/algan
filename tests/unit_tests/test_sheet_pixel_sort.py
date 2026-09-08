"""Per-pixel sheet ordering must match the stable global lexicographic sort."""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing.sheets import (
    _key_depth_order,
    _lexsort,
    _pixel_group_order,
    _sheet_walk_order,
    _unique_sorted_ids,
)
from algan.rendering.taichi_runtime import init_taichi, taichi_launch_is_local


@pytest.fixture(autouse=True)
def initialized_sort_kernel(monkeypatch):
    from algan.rendering.raytracing import sheet_sort_taichi

    init_taichi()
    calls = []
    original = sheet_sort_taichi.pixel_group_order

    def counted(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(sheet_sort_taichi, "pixel_group_order", counted)
    return calls


@pytest.mark.parametrize("lengths", [[], [0], [1], [2, 0, 7, 16, 17], [257, 3, 4097]])
def test_pixel_order_matches_global_sort(lengths, initialized_sort_kernel):
    device = SETTINGS.computing.render_device
    gen = torch.Generator().manual_seed(174)
    n = sum(lengths)
    pix = torch.repeat_interleave(
        torch.arange(len(lengths)), torch.tensor(lengths, dtype=torch.int64)
    )
    group = torch.randint(-17, 19, (n,), generator=gen, dtype=torch.int64)
    depth = torch.randint(-3, 5, (n,), generator=gen).to(torch.float32)
    offsets = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    # Exercise signed 64-bit group keys without narrowing or packed-key overflow.
    group[::5] += 1 << 40
    group[1::7] -= 1 << 40
    pix, group, depth, offsets = (x.to(device) for x in (pix, group, depth, offsets))
    expected = _lexsort(pix, group, depth)
    with SETTINGS.raytracing.experimental.override(sheet_pixel_sort=True):
        actual = _pixel_group_order(pix, group, depth, offsets)
    assert torch.equal(actual, expected)
    assert bool(initialized_sort_kernel) == bool(
        n and taichi_launch_is_local(pix.device)
    )
    with SETTINGS.raytracing.experimental.override(sheet_pixel_sort=False):
        assert torch.equal(_pixel_group_order(pix, group, depth, offsets), expected)


@pytest.mark.parametrize("n", [12, 129])
def test_pixel_order_preserves_depth_ties_and_nonfinite_order(n):
    device = SETTINGS.computing.render_device
    pix = torch.zeros(n, dtype=torch.int64, device=device)
    group = torch.zeros_like(pix)
    values = [float("nan"), float("inf"), -0.0, 0.0, -float("inf"), 1.0]
    depth = torch.tensor((values * n)[:n], dtype=torch.float32, device=device)
    offsets = torch.tensor([0, n], dtype=torch.int32, device=device)
    with SETTINGS.raytracing.experimental.override(sheet_pixel_sort=True):
        actual = _pixel_group_order(pix, group, depth, offsets)
    assert torch.equal(actual, _lexsort(pix, group, depth))


def test_missing_pixel_offsets_keeps_reference_sort():
    pix = torch.tensor([1, 0, 1])
    group = torch.tensor([7, 7, 3])
    depth = torch.tensor([2.0, 3.0, 1.0])
    with SETTINGS.raytracing.experimental.override(sheet_pixel_sort=True):
        assert torch.equal(
            _pixel_group_order(pix, group, depth, None), _lexsort(pix, group, depth)
        )


def test_uninitialized_runtime_keeps_reference_sort(
    monkeypatch, initialized_sort_kernel
):
    from algan.rendering import taichi_runtime

    monkeypatch.setattr(taichi_runtime, "_live_arch", lambda: None)
    device = SETTINGS.computing.render_device
    pix = torch.tensor([0, 0], device=device)
    group = torch.tensor([2, 1], device=device)
    depth = torch.tensor([1.0, 2.0], device=device)
    offsets = torch.tensor([0, 2], dtype=torch.int32, device=device)
    with SETTINGS.raytracing.experimental.override(sheet_pixel_sort=True):
        assert torch.equal(
            _pixel_group_order(pix, group, depth, offsets), _lexsort(pix, group, depth)
        )
    assert not initialized_sort_kernel


@pytest.mark.parametrize("ids", [[], [0], [-(2**40), -(2**40), 0, 3, 3, 2**40]])
def test_ordered_band_group_ids_match_reference(ids):
    keys = torch.tensor(ids, dtype=torch.int64, device=SETTINGS.computing.render_device)
    with SETTINGS.raytracing.experimental.override(sheet_pixel_sort=False):
        expected = _unique_sorted_ids(keys)
    with SETTINGS.raytracing.experimental.override(sheet_pixel_sort=True):
        actual = _unique_sorted_ids(keys)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize("n", [0, 1, 17, 4097])
def test_shell_key_depth_order_matches_global_lexsort(n):
    gen = torch.Generator().manual_seed(762)
    device = SETTINGS.computing.render_device
    key = torch.randint(-2, 3, (n,), generator=gen, dtype=torch.int64).to(device)
    depth = torch.randint(-4, 5, (n,), generator=gen).float().to(device)
    depth[::13] = float("nan")
    depth[1::19] = float("inf")
    with SETTINGS.raytracing.experimental.override(sheet_pixel_sort=True):
        actual = _key_depth_order(key, depth)
    assert torch.equal(actual, _lexsort(key, depth))


@pytest.mark.parametrize("lengths", [[0], [1], [7, 0, 129, 4097]])
def test_final_sheet_walk_order_matches_global_position_sort(lengths):
    gen = torch.Generator().manual_seed(478)
    pix = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
    position = (pix << 32) + torch.randint(0, 19, pix.shape, generator=gen)
    pix, position = (x.to(SETTINGS.computing.render_device) for x in (pix, position))
    with SETTINGS.raytracing.experimental.override(sheet_pixel_sort=True):
        actual = _sheet_walk_order(pix, position)
    assert torch.equal(actual, torch.argsort(position, stable=True))
