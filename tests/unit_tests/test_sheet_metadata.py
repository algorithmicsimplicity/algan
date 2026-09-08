"""Direct sheet diagnostics and CSR must preserve the original tensor results."""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing import sheets
from algan.rendering.taichi_runtime import init_taichi, taichi_launch_is_local


@pytest.mark.parametrize("lengths", [[], [1], [3, 1, 19, 0, 257, 4097]])
@pytest.mark.parametrize("split", [False, True])
def test_group_counts_match_reference(lengths, split, monkeypatch):
    from algan.rendering.raytracing import sheet_metadata_taichi

    init_taichi()
    device = SETTINGS.computing.render_device
    n = sum(lengths)
    starts = torch.zeros(n, dtype=torch.bool)
    bands = torch.empty(n, dtype=torch.int64)
    triangles = torch.empty(n, dtype=torch.bool)
    first = []
    at = 0
    for group, length in enumerate(lengths):
        if length:
            starts[at] = True
            triangles[at : at + length] = group % 3 != 1
            # Interleaved sheet IDs inside a group; first appearances need
            # not follow sheet-ID order after conflict-rank/class splitting.
            count = min(length, 7) if split else 1
            local = torch.arange(length).remainder(count).flip(0)
            bands[at : at + length] = len(first) + local
            first.extend(at + int((local == i).nonzero()[0]) for i in range(count))
        at += length
    order = torch.randperm(n, generator=torch.Generator().manual_seed(991))
    original_triangles = torch.empty_like(triangles)
    original_triangles[order] = triangles
    inputs = [
        starts,
        bands,
        order,
        original_triangles,
        torch.tensor(first, dtype=torch.int64),
    ]
    inputs = [x.to(device) for x in inputs]
    calls = []
    original = sheet_metadata_taichi.group_counts

    def counted(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(sheet_metadata_taichi, "group_counts", counted)
    with SETTINGS.raytracing.experimental.override(sheet_metadata_kernel=False):
        expected = sheets._sheet_group_counts(*inputs, len(first))
    with SETTINGS.raytracing.experimental.override(sheet_metadata_kernel=True):
        actual = sheets._sheet_group_counts(*inputs, len(first))
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert all(x.ndim == 0 and x.dtype == torch.int64 for x in actual)
    assert bool(calls) == bool(n and taichi_launch_is_local(inputs[0].device))


@pytest.mark.parametrize("lengths", [[], [1], [4097], [1, 7, 2, 257, 1]])
def test_sheet_offsets_match_reference(lengths):
    device = SETTINGS.computing.render_device
    covered = torch.arange(len(lengths), dtype=torch.int64) * 179 + (1 << 35)
    pix = torch.repeat_interleave(covered, torch.tensor(lengths, dtype=torch.int64))
    covered, pix = covered.to(device), pix.to(device)
    with SETTINGS.raytracing.experimental.override(sheet_metadata_kernel=False):
        expected = sheets._sheet_offsets(covered, pix)
    with SETTINGS.raytracing.experimental.override(sheet_metadata_kernel=True):
        actual = sheets._sheet_offsets(covered, pix)
    assert torch.equal(actual, expected)
    assert actual.tolist() == [0, *torch.tensor(lengths).cumsum(0).tolist()]


def test_uninitialized_group_counts_fall_back(monkeypatch):
    from algan.rendering import taichi_runtime

    monkeypatch.setattr(taichi_runtime, "_live_arch", lambda: None)
    device = SETTINGS.computing.render_device
    inputs = [
        torch.tensor([True]),
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([True]),
        torch.tensor([0]),
    ]
    inputs = [x.to(device) for x in inputs]
    with SETTINGS.raytracing.experimental.override(sheet_metadata_kernel=True):
        actual = sheets._sheet_group_counts(*inputs, 1)
    assert [int(x) for x in actual] == [1, 0]
