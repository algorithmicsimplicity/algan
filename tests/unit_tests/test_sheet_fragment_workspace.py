"""Sorted fragment payloads are private, exact and bounded by compaction."""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing import sheets
from algan.rendering.raytracing.sheet_fragments import (
    SortedFragments,
    gather_sorted_fragments,
)
from algan.rendering.raytracing.sheet_workspace import CompactionWorkspace
from algan.rendering.taichi_runtime import init_taichi
from algan.utils.memory_utils import ManualMemory
from tests.unit_tests.test_sheet_compaction import _coverage


def _workspace():
    init_taichi()
    memory = ManualMemory(0, device=SETTINGS.computing.render_device, num_bytes=1 << 20)
    memory._poison = 211
    return memory, CompactionWorkspace(memory)


def _poison(memory):
    with memory.temp():
        memory.get_tensor((memory.get_num_bytes_remaining(),), torch.uint8).fill_(173)


def _bits_equal(left, right):
    return torch.equal(
        left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
    )


def _fields(device):
    pixel = torch.tensor([2**40 + 1, 2**40 + 3, 7, 11], device=device)
    depth = (
        torch.tensor(
            [0x80000000, 0x7FC00023, 0x7F800000, 0x3F800001],
            dtype=torch.int64,
            device=device,
        )
        .to(torch.int32)
        .view(torch.float32)
    )
    area = torch.tensor([0.125, 0.33333334, 1.0, 0.75], device=device)
    mask = torch.tensor(
        [-2147221505, 0x101, 3, 0xFFFFFF], dtype=torch.int32, device=device
    )
    return SortedFragments(pixel, depth, area, mask)


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("size", [0, 1, 4])
def test_sorted_fragment_destinations_are_exact_and_private(owned, size):
    memory, ws = _workspace()
    fields = _fields(ws.device)
    order = torch.tensor([3, 1, 0, 2], device=ws.device)[:size]
    expected = [value[order].clone() for value in fields]
    with ws.stage():
        out = SortedFragments.allocate(ws, size) if owned else None
        floor = memory.current_pointer
        actual = gather_sorted_fragments(*fields, order, out=out)
        if owned:
            assert actual is out
        assert memory.current_pointer == floor
        _poison(memory)
        assert all(_bits_equal(a, e) for a, e in zip(actual, expected))
        source_cov = fields.coverage.clone()
        actual.coverage.zero_()
        assert _bits_equal(fields.coverage, source_cov)
    assert ws._depth == ws._live_bytes == 0
    assert memory.current_pointer == 0


@pytest.mark.parametrize(
    "problem",
    [
        "shape",
        "dtype",
        "stride",
        "input_alias",
        "output_alias",
        "source",
        "order",
        "device",
    ],
)
def test_sorted_fragment_validation_is_atomic(problem):
    memory, ws = _workspace()
    fields = list(_fields(ws.device))
    order = torch.tensor([3, 1, 0, 2], device=ws.device)
    storage = torch.full((16,), 91, dtype=torch.int64, device=ws.device)
    out = [
        storage[:4],
        storage[4:6].view(torch.float32),
        storage[6:8].view(torch.float32),
        storage[8:10].view(torch.int32),
    ]
    if problem == "shape":
        out[3] = out[3][:2]
    elif problem == "dtype":
        out[3] = out[3].view(torch.float32)
    elif problem == "stride":
        out[3] = storage[8:12].view(torch.int32)[::2]
    elif problem == "input_alias":
        fields[0] = out[0]
    elif problem == "output_alias":
        out[3] = out[2].view(torch.int32)
    elif problem == "source":
        fields[3] = fields[3].to(torch.int64)
    elif problem == "order":
        order = order.view(2, 2)
    else:
        out[3] = torch.empty(4, dtype=torch.int32, device="meta")
    snapshot = storage.clone()
    with pytest.raises(ValueError):
        gather_sorted_fragments(*fields, order, out=SortedFragments(*out))
    assert _bits_equal(storage, snapshot)


@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("failure_site", [None, "sorted", "rank", "class"])
def test_compaction_owns_whole_forward_lifetime(monkeypatch, diagnostics, failure_site):
    memory, ws = _workspace()
    normals = torch.tensor([[[0.0, 0.0, 1.0] * 3, [0.0, 1.0, 0.0] * 3] * 4])
    coverage, merged, cam, pws = _coverage(
        [(0, 1.0, 0, 0.3, 15), (0, 1.001, 1, 0.7, 240), (3, 1.0, 4, 0.5, 255)],
        tri_norm=normals,
    )
    coverage = {
        k: v.to(ws.device) if torch.is_tensor(v) else v for k, v in coverage.items()
    }
    merged = {
        k: v.to(ws.device) if torch.is_tensor(v) else v for k, v in merged.items()
    }
    args = (coverage, merged, cam.to(ws.device), pws.to(ws.device), 0, 4, 4)
    expected = sheets.compact_sheets(*args, shade_split=True, sample_depth=True)
    sentinel = memory.get_tensor((7,), torch.int32, persist=True).fill_(197)
    memory.get_tensor((5,), torch.uint8).fill_(113)
    before = memory.get_pointers()
    if failure_site is not None:
        name = {
            "sorted": "gather_sorted_fragments",
            "rank": "_sheet_rank_groups",
            "class": "_sheet_class_groups",
        }[failure_site]
        original = getattr(sheets, name)

        def fail_after(*a, **kw):
            original(*a, **kw)
            raise LookupError("injected stream ownership failure")

        monkeypatch.setattr(sheets, name, fail_after)
        with pytest.raises(LookupError, match="injected stream ownership failure"):
            sheets.compact_sheets(
                *args,
                shade_split=True,
                sample_depth=True,
                diagnostics=diagnostics,
                resolver_memory=None if diagnostics else memory,
                workspace=ws,
            )
        assert memory.get_pointers() == before
    else:
        actual = sheets.compact_sheets(
            *args,
            shade_split=True,
            sample_depth=True,
            diagnostics=diagnostics,
            resolver_memory=None if diagnostics else memory,
            workspace=ws,
        )
        assert memory.current_pointer == before[0]
        _poison(memory)
        renamed = {"sheet_cov": "sheet_wgt", "sheet_msk": "sheet_wmsk"}
        fields = actual if diagnostics else actual._asdict()
        for name, value in fields.items():
            if torch.is_tensor(value):
                wanted = expected[name if diagnostics else renamed.get(name, name)]
                assert torch.equal(value, wanted), name
    assert ws._depth == ws._live_bytes == 0
    _poison(memory)
    assert torch.all(sentinel == 197)
