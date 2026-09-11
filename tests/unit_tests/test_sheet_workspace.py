"""Compaction scratch is stage-owned; results survive reuse of those bytes."""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing import settings as rt
from algan.rendering.raytracing import sheets
from algan.rendering.raytracing.sheet_statistics import sheet_statistics
from algan.rendering.raytracing.sheet_workspace import CompactionWorkspace
from algan.rendering.taichi_runtime import init_taichi
from algan.taichi_compat import ti
from algan.utils.memory_utils import InsufficientMemoryException, ManualMemory


def _arena():
    init_taichi()
    memory = ManualMemory(0, device=SETTINGS.computing.render_device, num_bytes=1 << 17)
    memory._poison = 255
    return memory, CompactionWorkspace(memory)


def _overwrite_free(memory):
    with memory.temp():
        memory.get_tensor((memory.get_num_bytes_remaining(),), torch.uint8).fill_(213)


def _same_bits(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(
        actual.detach().cpu().contiguous().view(torch.uint8),
        expected.detach().cpu().contiguous().view(torch.uint8),
    )


def test_workspace_scopes_peak_alignment_and_reverse_ownership():
    memory, workspace = _arena()
    memory.get_tensor((1,), torch.uint8)
    original = memory.get_pointers()
    with pytest.raises(RuntimeError, match="inside a stage"):
        workspace.tensor((2,), torch.int64)
    with workspace.stage():
        outer = workspace.tensor((3,), torch.int64, 19)
        outer_end = memory.current_pointer
        with workspace.stage():
            workspace.tensor((7,), torch.int32, 41)
            peak = memory.current_pointer - original[0]
        assert memory.current_pointer == outer_end
        assert outer.tolist() == [19] * 3
        with workspace.stage():
            workspace.tensor((1,), torch.int32, 17)
        assert workspace.peak_bytes == peak  # maximum overlap, not sum of stages
        persistent = memory.get_tensor((4,), torch.int32, persist=True)
        persistent.fill_(211)
    assert memory.current_pointer == original[0]
    assert workspace._depth == workspace._live_bytes == 0
    _overwrite_free(memory)
    assert persistent.tolist() == [211] * 4
    assert memory.current_reverse_pointer < original[1]


@pytest.mark.parametrize("failure", [ValueError, InsufficientMemoryException])
def test_workspace_failure_rewinds_and_can_be_reused(failure):
    memory, workspace = _arena()
    before = memory.get_pointers()

    def fail():
        with workspace.stage():
            workspace.tensor((37,), torch.float64)
            with workspace.stage():
                workspace.tensor((15,), torch.int32)
                raise failure("injected")

    with pytest.raises(failure, match="injected"):
        fail()
    assert memory.get_pointers() == before
    assert workspace._depth == workspace._live_bytes == 0
    with workspace.stage():
        source = torch.tensor([0.0, -0.0, 0.75], device=workspace.device)
        copy = workspace.copy(source)
        assert copy.data_ptr() != source.data_ptr()
        copy.zero_()
        assert source[-1] == 0.75
    assert memory.get_pointers() == before


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("float32", [False, True])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_band_reductions_preserve_rounding_and_skip_unused_stores(
    monkeypatch, native, float32, diagnostics
):
    memory, workspace = _arena()
    monkeypatch.setattr(rt, "sheet_mask_kernel", native)
    if float32:
        monkeypatch.setattr(sheets, "accumulate_dtype", lambda: torch.float32)
        monkeypatch.setattr(sheets, "taichi_accumulate_dtype", lambda: ti.f32)
    device = workspace.device
    band = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64, device=device)
    mask = torch.tensor(
        [1, 1, 2, 0, 4 | sheets.AA_SLIVER_BIT], dtype=torch.int32, device=device
    )
    coverage = torch.tensor([0.125, 0.25, 0.0625, 0.375, 0.5], device=device)
    before = memory.get_pointers()
    area, union, fused, sliver = sheets._band_reduce(
        band,
        mask,
        coverage,
        3,
        want_sliver=diagnostics,
        want_fused=diagnostics,
        workspace=workspace,
    )
    assert memory.get_pointers() == before
    _overwrite_free(memory)
    _same_bits(area, torch.tensor([0.4375, 0.875, 0.0]))
    assert union.tolist() == [3, 4, 0]
    if diagnostics:
        assert fused.tolist() == [True, False, False]
        assert sliver.tolist() == [0, 1, 0]
    else:
        assert fused is sliver is None
    assert coverage.tolist() == [0.125, 0.25, 0.0625, 0.375, 0.5]


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("positioned", [False, True])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_sheet_statistics_owned_outputs_match_independent_tie_oracle(
    monkeypatch, native, positioned, diagnostics
):
    memory, workspace = _arena()
    monkeypatch.setattr(rt, "sheet_band_stats_kernel", native)
    device = workspace.device
    band = torch.tensor([0, 0, 0, 1, 1, 2], device=device)
    mask = torch.tensor([0, 1, 2, 0, 0, 4], dtype=torch.int32, device=device)
    original = torch.tensor([3, 0, 4, 2, 1, 5], device=device)
    positions = torch.arange(6, device=device)
    pixel = torch.tensor([7, 7, 7, 9, 9, 11], device=device)
    cov = torch.tensor([0.25, 0.5, 0.5, 0.125, 0.125, 0.75], device=device)
    before = memory.get_pointers()
    actual = sheet_statistics(
        band,
        mask,
        positions,
        original,
        pixel,
        cov,
        3,
        mask_all=sheets.AA_MASK_ALL,
        positioned=positioned,
        diagnostics=diagnostics,
        workspace=workspace,
    )
    assert memory.get_pointers() == before
    _overwrite_free(memory)
    assert actual.nearest_fragment.tolist() == ([0, 2, 5] if positioned else [3, 2, 5])
    assert actual.representative_fragment.tolist() == [0, 1, 5]
    assert actual.pixel.tolist() == [7, 9, 11]
    assert actual.min_position.tolist() == [0, 1, 5]
    if diagnostics:
        assert actual.first_sorted.tolist() == [0, 3, 5]
        assert actual.fragment_count.tolist() == [3, 2, 1]
    else:
        assert actual.first_sorted is actual.fragment_count is None


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("leading_flag", [False, True])
def test_rank_output_ownership_and_csr_groups(monkeypatch, native, leading_flag):
    memory, workspace = _arena()
    monkeypatch.setattr(rt, "sheet_rank_kernel", native)
    monkeypatch.setattr(sheets, "sheet_rank_groups", native)
    device = workspace.device
    order = torch.tensor([1, 0, 2, 4, 3, 5], device=device)
    mask = torch.tensor([1, 3, 2, 4, 4, 1], dtype=torch.int32, device=device)
    starts = torch.tensor(
        [leading_flag, False, False, True, False, False], device=device
    )
    positions = torch.arange(6, device=device)
    rank = memory.get_tensor((6,), torch.int32)
    before = memory.get_pointers()
    result = sheets._conflict_rank(
        starts, order, mask, positions, out=rank, workspace=workspace
    )
    assert result is rank
    expected = [0, 1, 1, 0, 1, 0]
    assert rank.tolist() == expected
    parent = torch.tensor([0, 0, 0, 1, 1, 1], device=device)
    groups, cid, ranks = sheets._sheet_rank_groups(parent, rank, workspace=workspace)
    assert memory.get_pointers() == before
    _overwrite_free(memory)
    assert rank.tolist() == expected
    assert groups.tolist() == [0, 1, 1, 2, 3, 2]
    assert cid.tolist() == [0, 0, 1, 1]
    assert ranks.tolist() == [0, 1, 0, 1]


@pytest.mark.parametrize(
    ("native", "reduce", "reuse"),
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    ],
)
def test_lane_table_destination_survives_scratch_reuse(
    monkeypatch, native, reduce, reuse
):
    memory, workspace = _arena()
    monkeypatch.setattr(rt, "sheet_sample_depth_kernel", native)
    monkeypatch.setattr(rt, "sheet_depth_reduce_kernel", reduce)
    monkeypatch.setattr(rt, "sheet_depth_buffer_reuse", reuse)
    device = workspace.device
    band = torch.tensor([0, 0, 1, 1], device=device)
    mask = torch.tensor([3, 4, 0, 8], dtype=torch.int32, device=device)
    depth = torch.tensor([-0.0, 0.5, 0.75, 1.0], device=device)
    out = memory.get_tensor((3, sheets.AA_NUM_SAMPLES), torch.float32)
    before = memory.get_pointers()
    assert (
        sheets._lane_first_owners(band, mask, depth, 3, 4, out=out, workspace=workspace)
        is out
    )
    expected = torch.full(out.shape, float("inf"))
    expected[0, :2] = -0.0
    expected[0, 2] = 0.5
    expected[1, 3] = 1.0
    assert memory.get_pointers() == before
    _overwrite_free(memory)
    _same_bits(out, expected)


def test_caller_output_layout_validation_and_empty_lane_table():
    memory, workspace = _arena()
    device = workspace.device
    empty = torch.empty(0, dtype=torch.int64, device=device)
    empty_i32 = empty.to(torch.int32)
    with pytest.raises(ValueError, match="int32"):
        sheets._conflict_rank(
            empty.bool(), empty, empty_i32, empty, out=empty, workspace=workspace
        )
    with pytest.raises(ValueError, match="float32"):
        sheets._lane_first_owners(
            empty, empty_i32, empty.float(), 0, 0, out=empty, workspace=workspace
        )
    out = memory.get_tensor((2, sheets.AA_NUM_SAMPLES), torch.float32)
    sheets._lane_first_owners(
        empty, empty_i32, empty.float(), 2, 0, out=out, workspace=workspace
    )
    assert torch.isposinf(out).all()


def test_outputs_reject_aliases_before_mutating_inputs():
    memory, workspace = _arena()
    device = workspace.device
    masks = torch.ones(8, dtype=torch.int32, device=device)
    positions = torch.arange(8, device=device)
    starts = torch.ones(8, dtype=torch.bool, device=device)
    with pytest.raises(ValueError, match="overlap"):
        sheets._conflict_rank(
            starts, positions, masks, positions, out=masks, workspace=workspace
        )
    assert masks.tolist() == [1] * 8
    depth = torch.arange(8, dtype=torch.float32, device=device)
    band = torch.zeros(8, dtype=torch.int64, device=device)
    with pytest.raises(ValueError, match="overlap"):
        sheets._lane_first_owners(
            band, masks, depth, 1, 8, out=depth.reshape(1, 8), workspace=workspace
        )
    assert depth.tolist() == list(range(8))
