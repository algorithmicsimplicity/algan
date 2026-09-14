"""Rank-pooling maps retain their outputs, not reduction or membership scratch."""

from __future__ import annotations

import pytest
import torch

from algan.rendering.raytracing import sheets
from algan.rendering.raytracing.sheet_grouping import RankPoolGroups
from tests.unit_tests.test_sheet_fragment_workspace import (
    _bits_equal,
    _poison,
    _workspace,
)
from tests.unit_tests.test_sheet_preprocessing_workspace import _strided


def _inputs(device, pattern="mixed", dtype=torch.int64):
    parents = [0, 0, 2, 2, 7, 7]
    ranks = [0, 1, 0, 1, 0, 1]
    bands = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    areas = [0.2, 0.3, 0.1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1, 0.1]
    masks = [3, 12, 48, 192] * 2 + [1, 2, 4, 8]
    if pattern == "empty":
        parents = ranks = bands = areas = masks = []
    elif pattern == "single":
        parents, ranks, bands, areas, masks = [7], [0], [0], [0.5], [255]
    elif pattern == "unsplit":
        parents, ranks = list(range(6)), [0] * 6
    elif pattern == "partial":
        masks = [1] * len(masks)
    elif pattern == "too_much_area":
        areas = [0.7] * len(areas)
    return (
        torch.tensor(parents, dtype=dtype, device=device),
        torch.tensor(ranks, dtype=dtype, device=device),
        torch.tensor(bands, dtype=dtype, device=device),
        torch.tensor(areas, dtype=torch.float32, device=device),
        torch.tensor(masks, dtype=torch.int32, device=device),
        len(parents),
    )


def _oracle(parents, ranks, bands, areas, masks, nb):
    labels, pool = torch.unique(parents, return_inverse=True)
    if labels.numel() == nb:
        return nb, None
    pooled = []
    for index in range(labels.numel()):
        selected = pool[bands] == index
        # Same specified rounding boundary, without the production reduction.
        area = areas[selected].double().sum().float()
        union = 0
        for mask in masks[selected].tolist():
            union |= mask & sheets.AA_MASK_ALL
        pooled.append(
            union == sheets.AA_MASK_ALL and area <= sheets.sheet_rank_pool_layers
        )
    fuse = torch.tensor(pooled, device=parents.device)
    pairs = list(zip(pool.tolist(), torch.where(fuse[pool], 0, ranks).tolist()))
    labels = sorted(set(pairs))
    lookup = {pair: i for i, pair in enumerate(labels)}
    result = torch.tensor(
        [lookup[pair] for pair in pairs], dtype=torch.int64, device=parents.device
    )
    return len(labels), None if len(labels) == nb else result


@pytest.mark.parametrize(
    "pattern", ["empty", "single", "unsplit", "mixed", "partial", "too_much_area"]
)
@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("reuse", [False, True])
def test_pool_destinations_preserve_grouping_and_none(
    monkeypatch, pattern, owned, native, reuse
):
    memory, ws = _workspace()
    monkeypatch.setattr(sheets.rt_settings, "sheet_mask_kernel", native)
    monkeypatch.setattr(sheets, "sheet_group_reuse", reuse)
    monkeypatch.setattr(sheets, "sheet_pixel_sort", reuse)
    args = _inputs(ws.device, pattern)
    expected_count, expected = _oracle(*args)
    snapshots = [x.clone() for x in args[:-1]]
    with ws.stage():
        out = ws.tensor((args[-1],), torch.int64) if owned else None
        floor = memory.current_pointer
        actual = sheets._rank_pool_groups(*args, out=out, workspace=ws)
        assert isinstance(actual, RankPoolGroups)
        count, groups = actual
        assert count == expected_count
        assert memory.current_pointer == floor
        _poison(memory)
        if expected is None:
            assert groups is None
        else:
            assert not owned or groups is out
            assert torch.equal(groups, expected)
        assert all(_bits_equal(a, b) for a, b in zip(args[:-1], snapshots))
    assert memory.current_pointer == ws._live_bytes == ws._depth == 0


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_pool_maps_accept_integer_width_and_stride(dtype, strided):
    memory, ws = _workspace()
    *values, nb = _inputs(ws.device, dtype=dtype)
    if strided:
        values = list(map(_strided, values))
    expected_count, expected = _oracle(*values, nb)
    with ws.stage():
        out = ws.tensor((nb,), torch.int64)
        count, actual = sheets._rank_pool_groups(*values, nb, out=out, workspace=ws)
        _poison(memory)
        assert count == expected_count
        assert actual is out
        assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "problem",
    [
        "shape",
        "dtype",
        "stride",
        "parent_alias",
        "rank_alias",
        "coverage_alias",
        "band_alias",
        "device",
        "count",
        "source",
    ],
)
def test_pool_destinations_fail_before_mutation(problem):
    memory, ws = _workspace()
    *values, nb = _inputs(ws.device)
    out = torch.full((nb,), 91, dtype=torch.int64, device=ws.device)
    if problem == "shape":
        out = out[:-1]
    elif problem == "dtype":
        out = out.float()
    elif problem == "stride":
        out = _strided(out)
    elif problem == "parent_alias":
        out = values[0]
    elif problem == "rank_alias":
        out = values[1]
    elif problem == "coverage_alias":
        out = values[3].view(torch.int64)
    elif problem == "band_alias":
        out = values[2][:nb]
    elif problem == "device":
        out = torch.empty(nb, dtype=torch.int64, device="meta")
    elif problem == "count":
        nb -= 1
    else:
        values[4] = values[4].float()
    tensors = [x for x in (*values, out) if x.device.type != "meta"]
    snapshots = [x.clone() for x in tensors]
    with pytest.raises(ValueError):
        sheets._rank_pool_groups(*values, nb, out=out, workspace=ws)
    assert all(_bits_equal(a, b) for a, b in zip(tensors, snapshots))
    assert memory.current_pointer == 0


def test_pool_reclaims_reduction_scratch_before_second_unique(monkeypatch):
    memory, ws = _workspace()
    args = _inputs(ws.device)
    calls = []
    original = sheets._unique_sorted_ids
    with ws.stage():
        out = ws.tensor((args[-1],), torch.int64)
        floor = memory.current_pointer

        def spy(*a, **kw):
            calls.append(memory.current_pointer - floor)
            if len(calls) == 2:
                # Only the parent inverse and key survive into this grouping.
                assert calls[-1] == 16 * args[-1]
                _poison(memory)
            return original(*a, **kw)

        monkeypatch.setattr(sheets, "_unique_sorted_ids", spy)
        count, actual = sheets._rank_pool_groups(*args, out=out, workspace=ws)
        assert len(calls) == 2
        assert memory.current_pointer == floor
        expected_count, expected = _oracle(*args)
        assert count == expected_count
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("failure", ["reduce", "second_unique"])
def test_pool_failure_releases_all_scratch(monkeypatch, failure):
    memory, ws = _workspace()
    args = _inputs(ws.device)
    name = "_band_reduce" if failure == "reduce" else "_unique_sorted_ids"
    original = getattr(sheets, name)
    calls = []

    def fail(*a, **kw):
        result = original(*a, **kw)
        calls.append(True)
        if failure == "reduce" or len(calls) == 2:
            raise LookupError("injected pool failure")
        return result

    monkeypatch.setattr(sheets, name, fail)
    with ws.stage():
        out = ws.tensor((args[-1],), torch.int64)
        sentinel = ws.tensor((5,), torch.uint8, 113)
        before = memory.get_pointers()
        with pytest.raises(LookupError, match="injected pool failure"):
            sheets._rank_pool_groups(*args, out=out, workspace=ws)
        assert memory.get_pointers() == before
        _poison(memory)
        assert torch.all(sentinel == 113)
    assert ws._depth == ws._live_bytes == 0
