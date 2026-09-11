"""Grouping keeps exact IDs and restores scratch, with checked destinations."""

from __future__ import annotations

from functools import partial

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing import sheet_grouping as grouping
from algan.rendering.raytracing import sheets
from algan.rendering.raytracing.array_ops import group_ids_from_starts
from algan.rendering.raytracing.sheet_workspace import CompactionWorkspace
from algan.rendering.taichi_runtime import init_taichi
from algan.utils.memory_utils import ManualMemory


def _workspace():
    init_taichi()
    memory = ManualMemory(0, device=SETTINGS.computing.render_device, num_bytes=1 << 20)
    memory._poison = 217
    return memory, CompactionWorkspace(memory)


def _poison(memory):
    with memory.temp():
        memory.get_tensor((memory.get_num_bytes_remaining(),), torch.uint8).fill_(191)


@pytest.mark.parametrize("narrow", [False, True])
@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("case", ["empty", "singleton", "duplicates", "strided"])
def test_class_groups_match_packed_oracle_and_survive_scratch(
    monkeypatch, narrow, owned, case
):
    memory, ws = _workspace()
    band = torch.tensor([2, 0, 2, 0, 1, 1, 0, 2], device=ws.device)
    cls = torch.tensor([7, 0, 2, 0, 3, 3, 7, 2], device=ws.device)
    if case == "empty":
        band, cls = band[:0], cls[:0]
    elif case == "singleton":
        band, cls = band[:1], cls[:1]
    elif case == "strided":
        band, cls = band.repeat_interleave(2)[::2], cls.repeat_interleave(2)[::2]
    keys, inverse = torch.unique(band * (1 << 25) + cls, return_inverse=True)
    labels = keys // (1 << 25)
    monkeypatch.setattr(grouping, "mps_friendly", lambda: narrow)
    out = memory.get_tensor(band.shape, torch.int64, persist=True) if owned else None
    pointers = memory.get_pointers()
    for _ in range(2):
        count, actual, parents = grouping.class_groups(
            band, cls, 1 << 25, out=out, workspace=ws
        )
        assert count == keys.numel()
        if owned:
            assert actual is out
        assert memory.get_pointers() == pointers
        assert ws._depth == ws._live_bytes == 0
        _poison(memory)
        assert torch.equal(actual, inverse)
        assert torch.equal(parents, labels)


@pytest.mark.parametrize("narrow", [False, True])
def test_class_groups_accept_int32_keys(monkeypatch, narrow):
    memory, ws = _workspace()
    monkeypatch.setattr(grouping, "mps_friendly", lambda: narrow)
    bands = torch.tensor([4, 2, 4, 2], dtype=torch.int32, device=ws.device)
    classes = torch.tensor([7, 7, 7, 8], dtype=torch.int32, device=ws.device)
    count, inverse, labels = grouping.class_groups(
        bands, classes, 1 << 25, workspace=ws
    )
    _poison(memory)
    assert count == 3
    assert inverse.dtype == torch.int64
    assert inverse.tolist() == [2, 0, 2, 1]
    assert labels.tolist() == [2, 2, 4]


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("size", [0, 1, 4])
def test_rank_group_output_and_parent_descriptors(monkeypatch, native, owned, size):
    memory, ws = _workspace()
    monkeypatch.setattr(sheets, "sheet_rank_groups", native)
    parent = torch.arange(size, device=ws.device).repeat_interleave(5)
    rank = torch.tensor([0, 1, 0, 2, 1], dtype=torch.int32, device=ws.device).repeat(
        size
    )
    keys, inverse = torch.unique(parent * 16 + rank, return_inverse=True)
    out = memory.get_tensor(parent.shape, torch.int64, persist=True) if owned else None
    pointers = memory.get_pointers()
    result = sheets._sheet_rank_groups(parent, rank, out=out, workspace=ws)
    assert isinstance(result, grouping.RankGroups)
    assert memory.get_pointers() == pointers
    assert ws._depth == ws._live_bytes == 0
    _poison(memory)
    if owned:
        assert result.ids is out
    assert torch.equal(result.ids, inverse)
    assert torch.equal(result.parent, keys // 16)
    assert torch.equal(result.rank, keys % 16)


@pytest.mark.parametrize("consecutive", [False, True])
@pytest.mark.parametrize("size", [0, 1, 19])
def test_unique_keeps_policy_and_destination(consecutive, size):
    memory, ws = _workspace()
    keys = (torch.arange(size, device=ws.device) % 5) * (1 << 49)
    expected = (torch.unique_consecutive if consecutive else torch.unique)(
        keys, return_inverse=True
    )
    out = memory.get_tensor(keys.shape, torch.int64, persist=True)
    pointers = memory.get_pointers()
    actual = grouping.unique_ids(keys, consecutive=consecutive, out=out)
    assert actual[1] is out
    assert memory.get_pointers() == pointers
    _poison(memory)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize("reuse", [False, True])
def test_uniform_class_reuse_honors_destination(monkeypatch, reuse):
    memory, ws = _workspace()
    monkeypatch.setattr(sheets, "sheet_group_reuse", reuse)
    band = torch.tensor([0, 0, 1, 2], device=ws.device)
    cls = torch.tensor([3, 3, 7, 7], device=ws.device)
    starts = torch.tensor([True, False, True, True], device=ws.device)
    out = memory.get_tensor(band.shape, torch.int64, persist=True)
    count, inverse, labels = sheets._sheet_class_groups(
        band, cls, starts, 3, out=out, workspace=ws
    )
    assert inverse is out
    _poison(memory)
    assert count == 3
    assert torch.equal(inverse, band)
    assert labels.tolist() == [0, 1, 2]


@pytest.mark.parametrize("method", ["class", "rank", "unique"])
@pytest.mark.parametrize(
    "problem", ["shape", "dtype", "stride", "alias", "keys", "device"]
)
def test_grouping_rejects_invalid_output_before_mutation(method, problem):
    memory, ws = _workspace()
    band = torch.tensor([0, 0, 1, 1], device=ws.device)
    other = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=ws.device)
    storage = torch.full((8,), 123, dtype=torch.int64, device=ws.device)
    out = storage[:4]
    if problem == "shape":
        out = storage
    elif problem == "dtype":
        out = storage[:2].view(torch.int32)
    elif problem == "stride":
        out = storage[::2]
    elif problem == "alias":
        band = out
    elif problem == "keys":
        band = band.reshape(2, 2)
    else:
        out = torch.empty(4, dtype=torch.int64, device="meta")
    pointers = memory.get_pointers()
    expected = storage.clone()
    if method == "class":
        call = partial(
            grouping.class_groups, band, other, 1 << 25, out=out, workspace=ws
        )
    elif method == "rank":
        call = partial(sheets._sheet_rank_groups, band, other, out=out, workspace=ws)
    else:
        call = partial(grouping.unique_ids, band, out=out)
    with pytest.raises(ValueError):
        call()
    assert memory.get_pointers() == pointers
    assert torch.equal(storage, expected)


@pytest.mark.parametrize("narrow", [False, True])
def test_grouping_failure_unwinds_workspace(monkeypatch, narrow):
    memory, ws = _workspace()
    monkeypatch.setattr(grouping, "mps_friendly", lambda: narrow)
    band = torch.tensor([0, 0, 1, 1], device=ws.device)
    cls = torch.tensor([2, 1, 1, 2], device=ws.device)
    out = memory.get_tensor((4,), torch.int64, persist=True).fill_(193)
    pointers = memory.get_pointers()

    def fail(*args, **kwargs):
        assert memory.current_pointer > pointers[0]
        raise LookupError("injected grouping failure")

    monkeypatch.setattr(grouping, "stable_lexsort" if narrow else "unique_ids", fail)
    with pytest.raises(LookupError, match="injected grouping failure"):
        grouping.class_groups(band, cls, 1 << 25, out=out, workspace=ws)
    assert memory.get_pointers() == pointers
    assert ws._depth == ws._live_bytes == 0
    _poison(memory)
    assert torch.equal(out, torch.full_like(out, 193))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "flags", [[], [True], [False], [True, False, True, True, False]]
)
def test_group_start_scan_preserves_exact_expression(dtype, flags):
    starts = torch.tensor(flags, dtype=torch.bool)
    expected = starts.to(dtype).cumsum(0) - 1
    out = torch.empty(starts.shape, dtype=dtype)
    assert group_ids_from_starts(starts, out=out) is out
    assert torch.equal(out, expected)


@pytest.mark.parametrize("problem", ["shape", "dtype", "stride", "alias", "input"])
def test_group_start_scan_rejects_bad_metadata(problem):
    storage = torch.full((8,), 57, dtype=torch.int64)
    starts = torch.tensor([True, False, True, False])
    out = storage[:4]
    if problem == "shape":
        out = storage
    elif problem == "dtype":
        out = storage[:4].view(torch.float64)
    elif problem == "stride":
        out = storage[::2]
    elif problem == "alias":
        starts = storage.view(torch.bool)[:4]
    else:
        starts = starts.to(torch.int32)
    expected = storage.clone()
    with pytest.raises(ValueError):
        group_ids_from_starts(starts, out=out)
    assert torch.equal(storage, expected)


@pytest.mark.parametrize("parent_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rank_dtype", [torch.int32, torch.int64])
def test_native_rank_grouping_normalizes_supported_integer_widths(
    monkeypatch, parent_dtype, rank_dtype
):
    memory, ws = _workspace()
    monkeypatch.setattr(sheets, "sheet_rank_groups", True)
    parent = torch.tensor([0, 0, 1, 1, 1], dtype=parent_dtype, device=ws.device)
    ranks = torch.tensor([0, 1, 0, 1, 2], dtype=rank_dtype, device=ws.device)
    out = memory.get_tensor(parent.shape, torch.int64, persist=True)
    got = sheets._sheet_rank_groups(parent, ranks, out=out, workspace=ws)
    _poison(memory)
    assert got.ids.tolist() == [0, 1, 2, 3, 4]
    assert got.parent.tolist() == [0, 0, 1, 1, 1]
    assert got.rank.tolist() == [0, 1, 0, 1, 2]


def test_pair_grouping_never_builds_a_wide_mps_key(monkeypatch):
    memory, ws = _workspace()
    monkeypatch.setattr(grouping, "mps_friendly", lambda: True)
    bands = torch.tensor([2**26 + 3, 2**26 + 1, 2**26 + 3, 2**26 + 1], device=ws.device)
    classes = torch.tensor([2**25 - 1, 7, 2**25 - 2, 7], device=ws.device)
    keys, expected = torch.unique(bands * (1 << 25) + classes, return_inverse=True)
    out = memory.get_tensor(bands.shape, torch.int64, persist=True)
    count, inverse, labels = grouping.class_groups(
        bands, classes, 1 << 25, out=out, workspace=ws
    )
    _poison(memory)
    assert count == len(keys)
    assert torch.equal(inverse, expected)
    assert torch.equal(labels, keys // (1 << 25))


def test_int32_packed_class_arithmetic_is_widened_before_multiplication(monkeypatch):
    _, ws = _workspace()
    monkeypatch.setattr(grouping, "mps_friendly", lambda: False)
    bands = torch.tensor([512, 513, 512], dtype=torch.int32, device=ws.device)
    classes = torch.tensor([1, 0, 0], dtype=torch.int32, device=ws.device)
    count, inverse, labels = grouping.class_groups(
        bands, classes, 1 << 25, workspace=ws
    )
    assert count == 3
    assert inverse.tolist() == [1, 2, 0]
    assert labels.tolist() == [512, 512, 513]


def test_int32_reference_rank_keys_are_widened_before_multiplication(monkeypatch):
    _, ws = _workspace()
    monkeypatch.setattr(sheets, "sheet_rank_groups", False)
    parents = torch.tensor(
        [2**29, 2**29, 2**29 + 1], dtype=torch.int32, device=ws.device
    )
    ranks = torch.tensor([1, 0, 0], dtype=torch.int32, device=ws.device)
    result = sheets._sheet_rank_groups(parents, ranks, workspace=ws)
    assert result.ids.tolist() == [1, 0, 2]
    assert result.parent.tolist() == [2**29, 2**29, 2**29 + 1]
    assert result.rank.tolist() == [0, 1, 0]
