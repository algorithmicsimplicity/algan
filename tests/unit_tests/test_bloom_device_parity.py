"""Bloom must write every arena pixel, including clamped interpolation borders."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from algan.rendering import mps_zero_copy
from algan.rendering.post_processing import bloom, bloom_kernels_taichi as kernels
from algan.rendering.taichi_runtime import init_taichi
from algan.settings._startup import render_device
from algan.taichi_compat import ti
from algan.utils.memory_utils import ManualMemory


@pytest.mark.parametrize("arch,available,installed,expected", [
    (ti.cpu, True, True, False),
    (ti.metal, False, False, False),
    (ti.metal, True, False, False),
    (ti.metal, True, True, True),
])
def test_mps_bloom_requires_live_metal_imports(monkeypatch, arch, available, installed, expected):
    monkeypatch.setattr(ti.lang.impl, "current_cfg", lambda: SimpleNamespace(arch=arch))
    monkeypatch.setattr(mps_zero_copy, "zero_copy_available", lambda: available)
    monkeypatch.setattr(mps_zero_copy, "installed", lambda: installed)
    assert kernels.can_use_bloom_taichi("mps") is expected


def _arena(device):
    memory = ManualMemory(0, device=device, num_bytes=64 * 2**20)
    guard = memory.get_tensor((137,), torch.uint8)
    guard.fill_(91)
    return memory, guard


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("batch,height,width,scale,channels", [
    (1, 127, 193, 3, 3),
    (2, 64, 80, 4, 4),
    (1, 270, 480, 8, 3),
])
def test_bloom_resize_matches_cpu_with_shared_arena(
    monkeypatch, fallback, batch, height, width, scale, channels,
):
    init_taichi()
    device = render_device()
    if fallback:
        monkeypatch.setattr(kernels, "can_use_bloom_taichi", lambda _device: False)
    else:
        assert kernels.can_use_bloom_taichi(device)
    generator = torch.Generator().manual_seed(76)
    source = torch.rand((batch, height, width, channels), generator=generator)
    reference_down = F.interpolate(
        source.permute(0, 3, 1, 2), scale_factor=1 / scale,
        mode="bilinear", align_corners=False, antialias=True,
    )
    reference_up = F.interpolate(
        reference_down, size=(height, width), mode="bilinear", align_corners=False,
    )
    memory, guard = _arena(device)
    tensor = memory.get_tensor(source.shape)
    tensor.copy_(source)
    down = memory.get_tensor(reference_down.shape)
    down.fill_(-91)
    before = mps_zero_copy.STATS["converted_launches"]
    bloom._downsample_bloom(tensor, down, memory, scale)
    torch.testing.assert_close(down.cpu(), reference_down, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(tensor.cpu(), source, atol=0, rtol=0)
    # Feed an independent CPU result so a downsample error cannot cancel or
    # contaminate an upsample error. The sentinel detects skipped writes.
    up_input = memory.get_tensor(reference_down.shape)
    up_input.copy_(reference_down)
    up = memory.get_tensor(reference_up.shape)
    up.fill_(-91)
    bloom._upsample_bloom(up_input, up, memory)
    torch.testing.assert_close(up.cpu(), reference_up, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(up_input.cpu(), reference_down, atol=0, rtol=0)
    assert (guard.cpu() == 91).all()
    if device.type == "mps" and not fallback:
        assert mps_zero_copy.STATS["converted_launches"] - before == 2


@pytest.mark.parametrize("batch,channels", [(1, 4), (2, 5)])
def test_full_bloom_matches_cpu_reference(monkeypatch, batch, channels):
    init_taichi()
    device = render_device()
    original_gate = kernels.can_use_bloom_taichi
    assert original_gate(device)
    monkeypatch.setattr(bloom, "_should_bypass_bloom", lambda: False)
    generator = torch.Generator().manual_seed(1234 + channels)
    frames = torch.randint(1, 220, (batch, 127, 193, channels), dtype=torch.uint8, generator=generator)
    frames[..., 3] = 100
    cpu_memory, _ = _arena("cpu")
    monkeypatch.setattr(kernels, "can_use_bloom_taichi", lambda _device: False)
    reference = bloom.bloom_filter(frames.clone(), memory=cpu_memory, scale_factor=64)
    monkeypatch.setattr(kernels, "can_use_bloom_taichi", original_gate)
    memory, guard = _arena(device)
    tensor = memory.get_tensor(frames.shape, torch.uint8)
    tensor.copy_(frames)
    before = mps_zero_copy.STATS["converted_launches"]
    actual = bloom.bloom_filter(tensor, memory=memory, scale_factor=64).cpu()
    torch.testing.assert_close(actual, reference, atol=3e-6, rtol=3e-6)
    quantized = lambda value: value.mul(255).clamp(0, 255).to(torch.uint8).to(torch.int16)
    assert (quantized(actual) - quantized(reference)).abs().max().item() <= 1
    assert torch.equal(tensor.cpu(), frames)
    assert (guard.cpu() == 91).all()
    if device.type == "mps":
        assert mps_zero_copy.STATS["converted_launches"] - before == 2
