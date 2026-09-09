"""Per-pixel sheet ordering must match the stable global lexicographic sort.

**The reference is computed on the host**, and that is load-bearing rather than
tidy. ``_lexsort``'s torch arm carries each key through an ``index_select`` per
pass, and on MPS an integer gather rounds through a float32 above 2**24
(``mps_compat._MPS_EXACT_INT_BITS``) -- so on an Apple GPU the device's own
``_lexsort`` is not a reliable answer for the wide keys these cases
deliberately use, and comparing a kernel against it was comparing two device
paths and trusting the wrong one. The host arm has no such ceiling. Where a
case's keys stay inside what every backend orders exactly, the device's torch
arm is checked against the same host reference as well.
"""

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


def _host_lexsort(*keys, device):
    """``_lexsort`` computed on the CPU and moved to ``device``.

    Exact at every key width, on every backend, and independent of whatever the
    device's sort does with signed zeros -- MPS's orders -0.0 before +0.0 where
    the CPU (and the kernel under test) call them equal.
    """
    return _lexsort(*(key.cpu() for key in keys)).to(device)


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
@pytest.mark.parametrize("wide_keys", [False, True])
def test_pixel_order_matches_global_sort(lengths, wide_keys, initialized_sort_kernel):
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
    if wide_keys:
        # Signed 64-bit group keys, without narrowing or packed-key overflow.
        # Past 2**24 the device's own torch arm may not be exact (see the
        # module docstring), so only the kernel is judged in this arm.
        group[::5] += 1 << 40
        group[1::7] -= 1 << 40
    pix, group, depth, offsets = (x.to(device) for x in (pix, group, depth, offsets))
    expected = _host_lexsort(pix, group, depth, device=device)
    with SETTINGS.raytracing.experimental.override(sheet_pixel_sort=True):
        actual = _pixel_group_order(pix, group, depth, offsets)
    assert torch.equal(actual, expected)
    assert bool(initialized_sort_kernel) == bool(
        n and taichi_launch_is_local(pix.device)
    )
    with SETTINGS.raytracing.experimental.override(sheet_pixel_sort=False):
        fallback = _pixel_group_order(pix, group, depth, offsets)
    if not wide_keys:
        assert torch.equal(fallback, expected)


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
    # Host reference: -0.0 and +0.0 tie there, as they do in the kernel's own
    # comparator. MPS's torch sort orders every -0.0 before every +0.0, which
    # is the backend disagreeing with the CPU rather than the kernel with
    # either -- and a depth, being a distance, is never a negative zero.
    assert torch.equal(actual, _host_lexsort(pix, group, depth, device=device))


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
    assert torch.equal(actual, _host_lexsort(key, depth, device=device))


@pytest.mark.parametrize("lengths", [[0], [1], [7, 0, 129, 4097]])
def test_final_sheet_walk_order_matches_global_position_sort(lengths):
    gen = torch.Generator().manual_seed(478)
    pix = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
    position = (pix << 32) + torch.randint(0, 19, pix.shape, generator=gen)
    pix, position = (x.to(SETTINGS.computing.render_device) for x in (pix, position))
    with SETTINGS.raytracing.experimental.override(sheet_pixel_sort=True):
        actual = _sheet_walk_order(pix, position)
    assert torch.equal(actual, torch.argsort(position, stable=True))
