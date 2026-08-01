import math

import pytest
import torch
import torch.nn.functional as F

from algan.rendering.post_processing import bloom as bloom_module
from algan.rendering.post_processing.bloom import bloom_filter
from algan.utils.memory_utils import ManualMemory


def _legacy_fft_conv1d(input_tensor, kernel, dim):
    fft_length = input_tensor.shape[dim] + kernel.shape[dim] - 1
    spectrum = torch.fft.rfft(input_tensor, n=fft_length, dim=dim)
    kernel_spectrum = torch.fft.rfft(kernel, n=fft_length, dim=dim)
    result = torch.fft.irfft(spectrum * kernel_spectrum, n=fft_length, dim=dim)
    left = (kernel.shape[dim] - 1) // 2
    return torch.ops.aten.slice(result, dim, left, left + input_tensor.shape[dim], 1)


def _legacy_bloom_reference(x, *, scale_factor):
    scale_factor = max(int(scale_factor * x.shape[-3] / 2160), 1)
    x = x.float().div(255)
    output = x.clone()
    channel_indices = [0, 1, 2, 4] if x.shape[-1] == 5 else slice(0, 3)
    color = x[..., channel_indices]
    glow = x[..., 3:4]
    glow.pow_(3)
    color.mul_(glow).mul_(30)
    color = F.interpolate(
        color.permute(0, 3, 1, 2),
        scale_factor=1 / scale_factor,
        mode="bilinear",
        antialias=True,
    )
    height = color.shape[-2]
    sigma_rim = max(0.004 * height, 1.0)
    sigma_tail = max(0.10 * height, sigma_rim * 1.5)
    accumulated = torch.zeros_like(color)
    for sigma, weight in ((sigma_rim, 1.0), (sigma_tail, 0.6)):
        radius = max(1, int(math.ceil(3.0 * sigma)))
        samples = torch.linspace(-radius, radius, 2 * radius + 1)
        gaussian = torch.exp(-0.5 * (samples / sigma) ** 2)
        gaussian.div_(gaussian.sum())
        channels = color.shape[1]
        horizontal = gaussian.view(1, 1, -1).expand(channels, 1, -1)
        vertical = gaussian.view(1, -1, 1).expand(channels, -1, 1)
        blurred = _legacy_fft_conv1d(color, horizontal, -1)
        blurred = _legacy_fft_conv1d(blurred, vertical, -2)
        accumulated.add_(blurred, alpha=weight)
    accumulated = F.interpolate(
        accumulated, size=x.shape[-3:-1], mode="bilinear", antialias=True
    ).permute(0, 2, 3, 1)
    output[..., channel_indices] += accumulated
    return output


def _managed_cpu_memory():
    # 0.5% of the configured 2 GiB CPU allowance: large enough for these tiny
    # frames while still exercising the real managed bump allocator.
    return ManualMemory(0.005, device=torch.device("cpu"), managed=True)


def test_default_bloom_invokes_fft(monkeypatch):
    fft_invoked = False
    original_rfft = torch.fft.rfft

    def mock_fft(*_args, **_kwargs):
        nonlocal fft_invoked
        fft_invoked = True
        return original_rfft(*_args, **_kwargs)

    monkeypatch.setattr(bloom_module.torch.fft, "rfft", mock_fft)
    frames = torch.randint(1, 220, (1, 8, 12, 4), dtype=torch.uint8)
    frames[..., 3] = 100
    memory = _managed_cpu_memory()
    bloom_filter(frames, memory=memory)
    assert fft_invoked, "default bloom must invoke FFT"


def test_default_bloom_short_circuits_no_glow(monkeypatch):
    def unexpected_blur(*_args, **_kwargs):
        raise AssertionError("no-glow bloom must skip the blur pipeline")

    monkeypatch.setattr(bloom_module, "_downsample_bloom", unexpected_blur)
    frames = torch.randint(1, 220, (1, 8, 12, 4), dtype=torch.uint8)
    frames[..., 3].zero_()
    memory = _managed_cpu_memory()
    output = bloom_filter(frames, memory=memory)
    assert output is frames


@pytest.mark.parametrize("channels", [4, 5])
def test_bloom_matches_legacy_render_tolerance_nondivisible(channels):
    torch.manual_seed(1234 + channels)
    frames = torch.randint(1, 220, (1, 9, 13, channels), dtype=torch.uint8)
    frames[..., 3] = 100
    scale_factor = 480  # int(480 * 9 / 2160) == 2
    reference = _legacy_bloom_reference(frames.clone(), scale_factor=scale_factor)
    memory = _managed_cpu_memory()
    actual = bloom_filter(frames.clone(), memory=memory, scale_factor=scale_factor)
    if channels == 5:
        reference = reference[..., [0, 1, 2, 4]]
        actual = actual[..., [0, 1, 2, 4]]
    else:
        reference = reference[..., :3]
        actual = actual[..., :3]
    reference_u8 = reference.mul(255).clamp_max(255).to(torch.uint8)
    actual_u8 = actual.mul(255).clamp_max(255).to(torch.uint8)
    difference = (actual_u8.to(torch.int16) - reference_u8.to(torch.int16)).abs()
    assert difference.max().item() <= 2
