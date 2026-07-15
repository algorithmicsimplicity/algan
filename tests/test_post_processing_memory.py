import math
from functools import partial

import pytest
import torch
import torch.nn.functional as F

from algan.rendering.post_processing import bloom as bloom_module
from algan.rendering.post_processing.bloom import bloom_filter
from algan.rendering.post_processing.post_process import (
    get_post_process_memory_required,
    post_process_frames,
)
from algan.rendering.raytracing import settings as rt_settings
from algan.utils.memory_utils import ManualMemory


def _legacy_fft_conv1d(input_tensor, kernel, dim):
    fft_length = input_tensor.shape[dim] + kernel.shape[dim] - 1
    spectrum = torch.fft.rfft(input_tensor, n=fft_length, dim=dim)
    kernel_spectrum = torch.fft.rfft(kernel, n=fft_length, dim=dim)
    result = torch.fft.irfft(
        spectrum * kernel_spectrum, n=fft_length, dim=dim
    )
    left = (kernel.shape[dim] - 1) // 2
    return torch.ops.aten.slice(
        result, dim, left, left + input_tensor.shape[dim], 1
    )


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
        color.permute(0, 3, 1, 2), scale_factor=1 / scale_factor,
        mode="bilinear", antialias=True,
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


def _run_and_compare(shape, dtype, aa, post_processes, fxaa, monkeypatch,
                     *, tonemap_enabled=False, tonemapping=True,
                     tonemap_method="neutral"):
    monkeypatch.setattr(rt_settings, "POST_PROCESS_TONEMAP", tonemap_enabled)
    monkeypatch.setattr(rt_settings, "TONEMAPPING", tonemapping)
    monkeypatch.setattr(rt_settings, "TONEMAP_METHOD", tonemap_method)
    monkeypatch.setattr(rt_settings, "TONEMAP_EXPOSURE", 1.25)

    if dtype == torch.uint8:
        frames = torch.randint(1, 220, shape, dtype=dtype)
        frames[..., 3] = 100
    else:
        frames = torch.rand(shape, dtype=dtype) * 2.0
        frames[..., 3] = 0.7

    memory = _managed_cpu_memory()
    predicted = get_post_process_memory_required(
        shape, dtype, aa, post_processes, fxaa,
        tonemap_enabled=tonemap_enabled,
        tonemapping=tonemapping,
        tonemap_method=tonemap_method,
        device=torch.device("cpu"),
    )
    output = post_process_frames(
        memory, frames, aa, post_processes=post_processes,
        apply_fxaa=fxaa,
    )
    assert output.dtype == torch.uint8
    assert memory.max_pointer == predicted


@pytest.mark.parametrize("channels", [4, 5])
def test_exact_post_process_memory_for_aa(channels, monkeypatch):
    _run_and_compare(
        (2, 16, 24, channels), torch.float32, 2, (), False, monkeypatch
    )


@pytest.mark.parametrize("channels", [4, 5])
def test_exact_post_process_memory_for_fxaa(channels, monkeypatch):
    _run_and_compare(
        (2, 8, 12, channels), torch.uint8, 1, (), True, monkeypatch
    )


@pytest.mark.parametrize("channels", [4, 5])
def test_exact_post_process_memory_for_bloom(channels, monkeypatch):
    _run_and_compare(
        (1, 8, 12, channels), torch.uint8, 1,
        (bloom_filter,), False, monkeypatch,
    )


@pytest.mark.parametrize("method", ["neutral", "agx"])
@pytest.mark.parametrize("channels", [4, 5])
def test_exact_post_process_memory_for_tonemap(method, channels, monkeypatch):
    _run_and_compare(
        (2, 8, 12, channels), torch.float32, 1, (), False, monkeypatch,
        tonemap_enabled=True, tonemapping=True, tonemap_method=method,
    )


def test_estimator_accounts_for_unaligned_entry_pointer(monkeypatch):
    monkeypatch.setattr(rt_settings, "POST_PROCESS_TONEMAP", True)
    monkeypatch.setattr(rt_settings, "TONEMAPPING", True)
    monkeypatch.setattr(rt_settings, "TONEMAP_METHOD", "neutral")
    shape = (1, 3, 3, 5)  # 45 uint8 bytes: leaves the forward pointer mod 4 == 1.
    memory = _managed_cpu_memory()
    frames = memory.get_tensor(shape, torch.uint8)
    frames.fill_(100)
    entry_pointer = memory.current_pointer
    assert entry_pointer % 4 == 1
    predicted = get_post_process_memory_required(
        shape, torch.uint8, 1, (), False,
        tonemap_enabled=True, tonemapping=True, tonemap_method="neutral",
        initial_pointer=entry_pointer, device=torch.device("cpu"),
    )
    post_process_frames(memory, frames, 1)
    assert memory.max_pointer - entry_pointer == predicted


def test_default_bloom_invokes_fft(monkeypatch):
    fft_invoked = False
    def mock_fft(*_args, **_kwargs):
        nonlocal fft_invoked
        fft_invoked = True
        return torch.fft.rfft(*_args, **_kwargs)

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


def test_exact_bloom_memory_with_zero_iterations(monkeypatch):
    _run_and_compare(
        (1, 8, 12, 4), torch.uint8, 1,
        (partial(bloom_filter, num_iterations=0),), False, monkeypatch,
    )


def test_exact_bloom_memory_with_downsample_fallback(monkeypatch):
    # At H=16 this custom factor resolves to a 2x downsample and exercises the
    # CPU separable-resize scratch in addition to convolution scratch.
    _run_and_compare(
        (1, 16, 24, 5), torch.uint8, 1,
        (partial(bloom_filter, scale_factor=270),), False, monkeypatch,
    )


def test_custom_post_process_exact_planner_protocol(monkeypatch):
    monkeypatch.setattr(rt_settings, "POST_PROCESS_TONEMAP", False)

    def custom_stage(frame, gain=1, *, memory):
        output = memory.clone(frame)
        output.add_(gain)
        return output

    def plan_custom_stage(*, sizer, frame_shape, frame_dtype, device,
                          args, kwargs):
        assert device.type == "cpu"
        assert args == ()
        assert kwargs == {"gain": 2}
        sizer.alloc(frame_shape, frame_dtype)
        return frame_shape, frame_dtype

    custom_stage.algan_memory_planner = plan_custom_stage
    process = partial(custom_stage, gain=2)
    shape = (1, 4, 6, 4)
    frames = torch.full(shape, 10, dtype=torch.uint8)
    memory = _managed_cpu_memory()
    predicted = get_post_process_memory_required(
        shape, torch.uint8, 1, (process,), False,
        device=torch.device("cpu"),
    )
    output = post_process_frames(
        memory, frames, 1, post_processes=(process,),
    )
    assert memory.max_pointer == predicted
    assert torch.equal(output, torch.full((1, 4, 6, 3), 12,
                                          dtype=torch.uint8))


@pytest.mark.parametrize("channels", [4, 5])
def test_bloom_matches_legacy_render_tolerance_nondivisible(channels):
    torch.manual_seed(1234 + channels)
    frames = torch.randint(1, 220, (1, 9, 13, channels), dtype=torch.uint8)
    frames[..., 3] = 100
    scale_factor = 480  # int(480 * 9 / 2160) == 2
    reference = _legacy_bloom_reference(
        frames.clone(), scale_factor=scale_factor
    )
    memory = _managed_cpu_memory()
    actual = bloom_filter(
        frames.clone(), memory=memory, scale_factor=scale_factor
    )
    if channels == 5:
        reference = reference[..., [0, 1, 2, 4]]
        actual = actual[..., [0, 1, 2, 4]]
    else:
        reference = reference[..., :3]
        actual = actual[..., :3]
    reference_u8 = reference.mul(255).clamp_max(255).to(torch.uint8)
    actual_u8 = actual.mul(255).clamp_max(255).to(torch.uint8)
    difference = (
        actual_u8.to(torch.int16) - reference_u8.to(torch.int16)
    ).abs()
    assert difference.max().item() <= 2
