"""Compact ray keys preserve frame/octant priority and coarse Morton order."""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing.tracer import _ray_sort_key_format
from algan.rendering.raytracing.wavefront_kernels_taichi import wavefront_ray_sort_keys
from algan.rendering.taichi_runtime import (
    init_taichi,
    taichi_arch_is_cuda,
    taichi_launch_is_local,
)


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("frames", [1, 16, 17, 1024])
def test_compact_format_bounds_full_sparse_frame_window(device, frames):
    with SETTINGS.raytracing.experimental.override(wf_ray_sort_compact=True):
        actual = _ray_sort_key_format(frames, torch.device(device))
    expected = (
        (torch.int32, 2) if device == "cuda" and frames <= 16 else (torch.int64, 0)
    )
    assert actual == expected


def test_compact_format_can_be_disabled():
    with SETTINGS.raytracing.experimental.override(wf_ray_sort_compact=False):
        assert _ray_sort_key_format(1, torch.device("cuda")) == (torch.int64, 0)


@pytest.mark.parametrize("frame", [0, 15])
def test_compact_keys_are_exact_high_bits_of_original_keys(frame):
    init_taichi()
    device = torch.device("cuda" if taichi_arch_is_cuda() else "cpu")
    if not taichi_launch_is_local(device):
        pytest.skip("requires a local CPU or CUDA kernel")
    # Includes clamped coordinates, quantization boundaries, every direction
    # octant and a permuted queue whose slots differ from their pixel indices.
    origins = torch.tensor(
        [
            [-1, 0, 1024],
            [0, 1, 2],
            [3, 4, 5],
            [511, 512, 513],
            [1021, 1022, 1023],
            [8, 32, 128],
            [101, 201, 301],
            [63, 127, 255],
        ],
        dtype=torch.float32,
        device=device,
    )
    directions = torch.tensor(
        [[1 if c & (1 << axis) else -1 for axis in range(3)] for c in range(8)],
        dtype=torch.float32,
        device=device,
    )
    active = torch.tensor([7, 2, 5, 0, 4, 6, 3, 1], dtype=torch.int32, device=device)
    pixels = torch.arange(8, dtype=torch.int32, device=device) + frame * 16
    original = torch.empty(8, dtype=torch.int64, device=device)
    compact = torch.empty(8, dtype=torch.int32, device=device)
    inputs = (active, 8, origins, directions, pixels, 16, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    wavefront_ray_sort_keys(*inputs, original, 0)
    wavefront_ray_sort_keys(*inputs, compact, 2)
    assert torch.equal(compact.to(torch.int64), original >> 6)
    assert bool((compact >= 0).all())
    assert torch.equal(compact.to(torch.int64) >> 27, torch.full_like(original, frame))
    assert torch.equal((compact.to(torch.int64) >> 24) & 7, active.to(torch.int64))
