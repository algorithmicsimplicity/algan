"""Sorted fragment runs are scanned once and invalidated after membership changes."""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing import raster_pipeline as raster
from algan.rendering.raytracing import settings as rt
from algan.rendering.raytracing.pixel_runs import PixelRunCSR
from algan.rendering.taichi_runtime import init_taichi
from algan.utils.memory_utils import ManualMemory


def _arena():
    init_taichi()
    return ManualMemory(0, device=SETTINGS.computing.render_device, num_bytes=1 << 16)


@pytest.mark.parametrize("pixels", [[], [7], [0, 0, 7, 7, 7, 90]])
def test_pixel_runs_share_terminal_offsets_and_arena_ownership(pixels):
    memory = _arena()
    values = torch.tensor(pixels, dtype=torch.int64, device=memory.data.device)
    reference = PixelRunCSR.from_sorted_pixels(values)
    before = memory.get_pointers()
    with memory.temp():
        result = PixelRunCSR.from_sorted_pixels(values, memory=memory)
        assert result.offsets.dtype == torch.int32
        assert result.offsets.tolist() == reference.offsets.tolist()
        assert result.covered.tolist() == reference.covered.tolist()
        assert result.counts.tolist() == reference.counts.tolist()
        assert result.offsets[-1] == len(pixels)
        assert result.num_fragments == len(pixels)
        assert result.starts.untyped_storage().data_ptr() == memory.data.data_ptr()
        assert memory.current_reverse_pointer == before[1]
        assert (
            result.require_counts(result.counts, len(pixels)).tolist()
            == result.starts.tolist()
        )
    assert memory.get_pointers() == before


def test_run_record_rejects_stale_counts_even_when_fragment_length_is_unchanged():
    pixels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    old = PixelRunCSR.from_sorted_pixels(pixels)
    replacement = PixelRunCSR.from_sorted_pixels(torch.tensor([0, 1, 1, 1]))
    with pytest.raises(ValueError, match="different fragment stream"):
        old.require_counts(replacement.counts, 4)
    with pytest.raises(ValueError, match="different fragment stream"):
        old.require_counts(old.counts, 3)
    assert old.offsets.tolist() == [0, 2, 4]
    assert replacement.offsets.tolist() == [0, 1, 4]


@pytest.mark.parametrize(
    "bad", [torch.zeros(3), torch.zeros((2, 2), dtype=torch.int64)]
)
def test_pixel_runs_validate_before_arena_allocation(bad):
    memory = _arena()
    before = memory.get_pointers()
    with pytest.raises(ValueError, match="integer vector"):
        PixelRunCSR.from_sorted_pixels(bad.to(memory.data.device), memory=memory)
    assert memory.get_pointers() == before


@pytest.mark.parametrize("native", [False, True])
def test_opaque_prefix_reuses_csr_and_rebuilds_after_truncation(monkeypatch, native):
    memory = _arena()
    device = memory.data.device
    monkeypatch.setattr(rt, "raster_opaque_trunc_kernel", native)
    pixels = torch.tensor([1, 1, 1, 4, 4, 9], device=device)
    opaque = torch.tensor([False, True, False, True, False, False], device=device)
    with memory.temp():
        runs = PixelRunCSR.from_sorted_pixels(pixels, memory=memory)
        reference = raster._opaque_prefix_keep(opaque, runs.counts, 6)

        def no_scan(*args, **kwargs):
            raise AssertionError("consumer rebuilt an unchanged CSR")

        monkeypatch.setattr(raster, "csr_offsets", no_scan)
        keep = raster._opaque_prefix_keep(opaque, runs.counts, 6, runs=runs)
        assert torch.equal(keep, reference)
        assert keep.tolist() == [True, True, False, True, False, True]
        replacement = PixelRunCSR.from_sorted_pixels(pixels[keep], memory=memory)
        assert replacement.offsets.tolist() == [0, 2, 3, 4]
        assert runs.offsets.tolist() == [0, 3, 5, 6]  # no mutation of stale record
        with pytest.raises(ValueError, match="different fragment stream"):
            raster._opaque_prefix_keep(opaque[keep], replacement.counts, 4, runs=runs)


@pytest.mark.parametrize("native", [False, True])
def test_one_mesh_cap_shares_csr_independently_of_kernel_gate(monkeypatch, native):
    memory = _arena()
    device = memory.data.device
    monkeypatch.setattr(rt, "sheet_one_mesh_kernel", native)
    pixel = torch.tensor([1, 1, 5, 5], device=device)
    key = (pixel << 32) | torch.tensor([1.0, 2.0, 1.0, 2.0], device=device).view(
        torch.int32
    ).long()
    ref = torch.tensor([0, 1, 0, 2], dtype=torch.int32, device=device)
    cov = torch.tensor([0.125, 0.5, 0.25, 0.75], device=device)
    masks = torch.tensor([1, 2, 1, 2], dtype=torch.int32, device=device)
    opaque = torch.ones(4, dtype=torch.bool, device=device)
    tri_obj = torch.tensor([[3, 3, 4]], dtype=torch.int32, device=device)
    with memory.temp():
        runs = PixelRunCSR.from_sorted_pixels(pixel, memory=memory)
        expected = raster._one_mesh_pixel_caps(
            key, ref, cov, masks.clone(), opaque, runs.counts, tri_obj, 16, 0
        )

        def no_scan(*args, **kwargs):
            raise AssertionError("one-mesh consumer rebuilt an unchanged CSR")

        monkeypatch.setattr(raster, "csr_offsets", no_scan)
        actual = raster._one_mesh_pixel_caps(
            key, ref, cov, masks.clone(), opaque, runs.counts, tri_obj, 16, 0, runs=runs
        )
        for got, want in zip(actual, expected):
            assert torch.equal(got, want)


def test_narrow_run_capacity_is_checked_without_allocating_or_scanning():
    memory = _arena()
    before = memory.get_pointers()
    # A broadcast view exercises metadata overflow without a multi-GiB array.
    pixels = torch.zeros(1, dtype=torch.int64, device=memory.data.device).expand(2**31)
    with pytest.raises(OverflowError, match="fit int32"):
        PixelRunCSR.from_sorted_pixels(pixels, memory=memory)
    assert memory.get_pointers() == before
