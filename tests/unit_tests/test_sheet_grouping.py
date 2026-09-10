"""Exact integer oracles for local grouping, including long and mixed runs."""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing.sheet_grouping import class_groups, unique_sorted_ids
from algan.rendering.taichi_runtime import init_taichi


@pytest.fixture(autouse=True)
def _compiler():
    init_taichi()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("count", [0, 1, 2, 257, 8193])
def test_unique_matches_integer_reference(dtype, count):
    gen = torch.Generator().manual_seed(17)
    keys = torch.randint(-(1 << 28), 1 << 28, (count,), generator=gen).sort().values
    keys = keys.repeat_interleave(3).to(dtype)
    if dtype == torch.int64:
        keys *= 1 << 30
    expected = torch.unique_consecutive(keys, return_inverse=True)
    actual = unique_sorted_ids(keys.to(SETTINGS.computing.render_device))
    assert all(torch.equal(a.cpu(), e) for a, e in zip(actual, expected))


@pytest.mark.parametrize("lengths", [[], [1], [2, 3, 1], [16, 17, 31, 128], [4097]])
@pytest.mark.parametrize("uniform", [False, True])
def test_class_groups_match_wide_cpu_keys(lengths, uniform):
    gen = torch.Generator().manual_seed(29)
    band_parts, class_parts, start_parts = [], [], []
    offset = 0
    for n in lengths:
        local = torch.randint(0, max(1, n // 3), (n,), generator=gen)
        band_parts.append(local + offset)
        classes = torch.randint((1 << 24) - 3, (1 << 24) + 4, (n,), generator=gen)
        if uniform:
            classes.fill_((1 << 25) - 1)
        class_parts.append(classes)
        starts = torch.zeros(n, dtype=torch.bool)
        starts[0] = True
        start_parts.append(starts)
        offset += int(local.max()) + 1
    bands = torch.cat(band_parts) if lengths else torch.empty(0, dtype=torch.int64)
    classes = torch.cat(class_parts) if lengths else torch.empty_like(bands)
    starts = torch.cat(start_parts) if lengths else torch.empty(0, dtype=torch.bool)
    keys, inverse = torch.unique(bands * (1 << 25) + classes, return_inverse=True)
    dev = SETTINGS.computing.render_device
    count, actual, groups = class_groups(bands.to(dev), classes.to(dev), starts.to(dev))
    assert count == len(keys)
    assert torch.equal(actual.cpu(), inverse)
    assert torch.equal(groups.cpu(), keys // (1 << 25))


def test_class_groups_reject_mismatched_lengths():
    device = SETTINGS.computing.render_device
    with pytest.raises(ValueError, match="matching lengths"):
        class_groups(
            torch.zeros(2, dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.ones(2, dtype=torch.bool, device=device),
        )
