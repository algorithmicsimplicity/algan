"""Reduction destinations outlive scratch and reject aliasing before writes."""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing import settings as rt
from algan.rendering.raytracing import sheet_statistics as stats_module
from algan.rendering.raytracing import sheets
from algan.rendering.raytracing.sheet_reduction_buffers import (
    BandReduction,
    SheetWeights,
)
from algan.rendering.raytracing.sheet_statistics import (
    SheetStatistics,
    sheet_statistics,
)
from algan.rendering.raytracing.sheet_workspace import CompactionWorkspace
from algan.rendering.taichi_runtime import init_taichi
from algan.taichi_compat import ti
from algan.utils.memory_utils import ManualMemory


def _workspace():
    init_taichi()
    memory = ManualMemory(0, device=SETTINGS.computing.render_device, num_bytes=1 << 18)
    memory._poison = 255
    return memory, CompactionWorkspace(memory)


def _poison_free(memory):
    with memory.temp():
        memory.get_tensor((memory.get_num_bytes_remaining(),), torch.uint8).fill_(211)


def _same(actual, expected):
    for got, want in zip(actual, expected):
        if want is None:
            assert got is None
        else:
            assert got.dtype == want.dtype
            assert torch.equal(
                got.contiguous().view(torch.uint8), want.contiguous().view(torch.uint8)
            )


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("narrow", [False, True])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_band_destinations_survive_scratch_reuse(
    monkeypatch, native, narrow, diagnostics
):
    memory, ws = _workspace()
    dev = ws.device
    monkeypatch.setattr(rt, "sheet_mask_kernel", native)
    if narrow:
        monkeypatch.setattr(sheets, "accumulate_dtype", lambda: torch.float32)
        monkeypatch.setattr(sheets, "taichi_accumulate_dtype", lambda: ti.f32)
    band = torch.tensor([0, 0, 1, 1, 2], device=dev)
    mask = torch.tensor([1, 1, 2, 4, 0], dtype=torch.int32, device=dev)
    cov = torch.tensor([0.125, 0.25, 0.375, 0.1875, 0.5], device=dev)
    expected = sheets._band_reduce(
        band, mask, cov, 3, want_fused=diagnostics, want_sliver=diagnostics
    )
    before = memory.get_pointers()
    with ws.stage():
        out = BandReduction.allocate(
            ws, 3, want_fused=diagnostics, want_sliver=diagnostics
        )
        floor = memory.current_pointer
        actual = sheets._band_reduce(
            band,
            mask,
            cov,
            3,
            want_fused=diagnostics,
            want_sliver=diagnostics,
            workspace=ws,
            out=out,
        )
        assert all(got is dst for got, dst in zip(actual, out))
        assert memory.current_pointer == floor
        _poison_free(memory)
        _same(actual, expected)
    assert memory.get_pointers() == before


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("narrow", [False, True])
@pytest.mark.parametrize("positioned", [False, True])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_statistics_destinations_survive_both_index_widths(
    monkeypatch, native, narrow, positioned, diagnostics
):
    memory, ws = _workspace()
    dev = ws.device
    monkeypatch.setattr(rt, "sheet_band_stats_kernel", native)
    if narrow:
        monkeypatch.setattr(stats_module, "reduction_index_dtype", lambda: torch.int32)
        monkeypatch.setattr(
            stats_module, "taichi_reduction_index_dtype", lambda: ti.i32
        )
    band = torch.tensor([0, 0, 0, 1, 1, 2], device=dev)
    mask = torch.tensor([0, 1, 2, 0, 0, 4], dtype=torch.int32, device=dev)
    original = torch.tensor([3, 0, 4, 2, 1, 5], device=dev)
    positions = torch.arange(6, device=dev)
    pixels = torch.tensor([7, 7, 7, 9, 9, 11], device=dev)
    cov = torch.tensor([0.25, 0.5, 0.5, 0.125, 0.125, 0.75], device=dev)
    args = (band, mask, positions, original, pixels, cov, 3)
    flags = {
        "mask_all": sheets.AA_MASK_ALL,
        "positioned": positioned,
        "diagnostics": diagnostics,
    }
    expected = sheet_statistics(*args, **flags)
    before = memory.get_pointers()
    with ws.stage():
        out = SheetStatistics.allocate(ws, 3, diagnostics=diagnostics)
        floor = memory.current_pointer
        result = sheet_statistics(*args, **flags, workspace=ws, out=out)
        assert result is out
        assert memory.current_pointer == floor
        _poison_free(memory)
        _same(result, expected)
    assert memory.get_pointers() == before


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("bands", [[], [0], [0, 1], [0, 0, 1], [0, 1, 0]])
def test_sibling_destinations_include_empty_singleton_and_interleaved(
    monkeypatch, native, bands
):
    memory, ws = _workspace()
    dev, n = ws.device, len(bands)
    monkeypatch.setattr(rt, "sheet_sibling_weights_kernel", native)
    band = torch.tensor(bands, dtype=torch.int64, device=dev)
    cov = torch.full((n,), 0.25, device=dev)
    masks = torch.full((n,), 3, dtype=torch.int32, device=dev)
    area = torch.tensor([0.5, 0.25], device=dev)
    union = torch.tensor([3, 3], dtype=torch.int32, device=dev)
    corr = torch.tensor([1.0, 0.5], device=dev)
    args = (band, cov, masks, area, union, corr)
    expected = sheets._sibling_weights(*args)
    before = memory.get_pointers()
    with ws.stage():
        out = SheetWeights.allocate(ws, n)
        floor = memory.current_pointer
        actual = sheets._sibling_weights(*args, workspace=ws, out=out)
        assert all(a is b for a, b in zip(actual, out))
        assert memory.current_pointer == floor
        _poison_free(memory)
        _same(actual, expected)
    assert memory.get_pointers() == before


@pytest.mark.parametrize(
    "bad", ["input_alias", "output_alias", "dtype", "shape", "stride", "disabled"]
)
def test_invalid_band_destination_is_rejected_before_any_mutation(bad):
    memory, ws = _workspace()
    dev = ws.device
    band = torch.arange(3, device=dev)
    mask = torch.tensor([1, 2, 4], dtype=torch.int32, device=dev)
    cov = torch.tensor([0.125, 0.25, 0.5], device=dev)
    with ws.stage():
        out = BandReduction.allocate(ws, 3, want_fused=False, want_sliver=False)
        out.area.fill_(19)
        out.union.fill_(23)
        if bad == "input_alias":
            invalid = out._replace(area=cov)
        elif bad == "output_alias":
            invalid = out._replace(union=out.area.view(torch.int32))
        elif bad == "dtype":
            invalid = out._replace(area=out.area.view(torch.int32))
        elif bad == "shape":
            invalid = out._replace(area=out.area[:2])
        elif bad == "stride":
            invalid = out._replace(area=torch.empty(6, device=dev)[::2])
        else:
            invalid = out._replace(fused=torch.ones(3, dtype=torch.bool, device=dev))
        before = memory.get_pointers()
        with pytest.raises(ValueError):
            sheets._band_reduce(
                band,
                mask,
                cov,
                3,
                want_sliver=False,
                want_fused=False,
                workspace=ws,
                out=invalid,
            )
        assert memory.get_pointers() == before
        assert out.area.tolist() == [19] * 3
        assert out.union.tolist() == [23] * 3
        assert cov.tolist() == [0.125, 0.25, 0.5]


@pytest.mark.parametrize("which", ["statistics", "siblings"])
def test_other_destinations_reject_output_aliases_before_writes(which):
    memory, ws = _workspace()
    dev = ws.device
    band = torch.arange(3, device=dev)
    mask = torch.ones(3, dtype=torch.int32, device=dev)
    cov = torch.ones(3, device=dev)
    with ws.stage():
        if which == "statistics":
            out = SheetStatistics.allocate(ws, 3, diagnostics=False)
            for value in out[:4]:
                value.fill_(17)
            bad = out._replace(pixel=out.min_position)
            with pytest.raises(ValueError, match="overlap"):
                sheet_statistics(
                    band,
                    mask,
                    band,
                    band,
                    band,
                    cov,
                    3,
                    mask_all=255,
                    positioned=True,
                    diagnostics=False,
                    workspace=ws,
                    out=bad,
                )
            assert all(t.tolist() == [17] * 3 for t in out[:4])
        else:
            out = SheetWeights.allocate(ws, 3)
            out.coverage.fill_(17)
            bad = out._replace(mask=out.coverage.view(torch.int32))
            with pytest.raises(ValueError, match="overlap"):
                sheets._sibling_weights(
                    band, cov, mask, cov, mask, cov, workspace=ws, out=bad
                )
            assert out.coverage.tolist() == [17] * 3


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("narrow", [False, True])
def test_composite_destinations_preserve_dust_sliver_and_partial_union_rules(
    monkeypatch, native, narrow
):
    from algan.rendering.raytracing.sheet_reduction_buffers import BandComposite

    memory, ws = _workspace()
    dev = ws.device
    monkeypatch.setattr(rt, "sheet_mask_kernel", native)
    if narrow:
        monkeypatch.setattr(sheets, "accumulate_dtype", lambda: torch.float32)
        monkeypatch.setattr(sheets, "taichi_accumulate_dtype", lambda: ti.f32)
    band = torch.tensor([0, 0, 1, 2, 3, 4], device=dev)
    mask = torch.tensor(
        [15, 240, 255, 1, 0, sheets.AA_SLIVER_BIT | 3], dtype=torch.int32, device=dev
    )
    cov = torch.tensor([0.25, 0.75, 0.5, 0.0625, 0.25, 0.125], device=dev)
    expected = sheets._band_composite(band, 5, cov, mask)
    assert expected.correction.tolist() == [1.0, 0.5, 0.5, 2.0, 0.5]
    assert expected.split.tolist() == [True, True, True, False, False]
    before = memory.get_pointers()
    with ws.stage():
        out = BandComposite.allocate(ws, 5)
        floor = memory.current_pointer
        result = sheets._band_composite(band, 5, cov, mask, workspace=ws, out=out)
        assert result is out
        assert memory.current_pointer == floor
        _poison_free(memory)
        _same(result, expected)
    assert memory.get_pointers() == before
