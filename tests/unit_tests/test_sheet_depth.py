"""The local depth reduction must exclude every crossing of the same surface."""

from __future__ import annotations

import pytest
import torch

from algan.rendering.raytracing.raster_taichi import _AA_NUM_SAMPLES
from algan.rendering.raytracing.sheet_depth_taichi import (
    sheet_depth_lose,
    sheet_lane_depths,
)
from algan.rendering.taichi_runtime import init_taichi


@pytest.mark.parametrize("cede", [0.0, 0.25, 0.5, 1.0])
def test_competing_depth_matches_unbounded_reference(cede):
    init_taichi()
    gen = torch.Generator().manual_seed(42)
    pixel = torch.repeat_interleave(
        torch.tensor([0, 3, 9, 17]), torch.tensor([1, 7, 64, 3])
    )
    n = pixel.numel()
    sid = torch.randint(0, 5, (n,), generator=gen)
    depth = torch.randint(0, 5, (n, _AA_NUM_SAMPLES), generator=gen).float()
    depth[::3, 0] = float("inf")
    mask = torch.randint(
        0, 1 << _AA_NUM_SAMPLES, (n,), generator=gen, dtype=torch.int32
    )
    subject = torch.randint(0, 2, (n,), generator=gen, dtype=torch.uint8)
    enforcer = torch.randint(0, 2, (n,), generator=gen, dtype=torch.uint8)
    enforcer[pixel == 17] = 0  # no eligible competitor
    epsilon, shift = 1e-5, 16
    expected = torch.zeros(n, dtype=torch.int32)
    for i in range(n):
        competitors = (pixel == pixel[i]) & (sid != sid[i]) & enforcer.bool()
        other = (
            depth[competitors].amin(0)
            if competitors.any()
            else torch.full((_AA_NUM_SAMPLES,), float("inf"))
        )
        owns = ((mask[i] >> torch.arange(_AA_NUM_SAMPLES)) & 1).bool()
        lost = owns & (other < depth[i] - epsilon) & subject[i].bool()
        if lost.sum() > cede * owns.sum():
            expected[i] = (
                sum(1 << lane for lane in range(_AA_NUM_SAMPLES) if lost[lane]) << shift
            )
    got = torch.full_like(expected, -1)
    sheet_depth_lose(
        pixel, sid, depth, mask, subject, enforcer, n, epsilon, cede, shift, got
    )
    assert torch.equal(got, expected)


def test_empty_depth_stream():
    init_taichi()
    index = torch.empty(0, dtype=torch.int64)
    mask = torch.empty(0, dtype=torch.int32)
    flag = torch.empty(0, dtype=torch.uint8)
    sheet_depth_lose(
        index,
        index,
        torch.empty(0, _AA_NUM_SAMPLES),
        mask,
        flag,
        flag,
        0,
        1e-5,
        0.25,
        16,
        mask,
    )


def test_lane_depth_gather_preserves_missing_owner_and_float_bits():
    init_taichi()
    depth = torch.tensor([0.0, -0.0, 1.25, 999.0])
    first = torch.arange(2 * _AA_NUM_SAMPLES, dtype=torch.int32) % 5
    out = torch.empty(2, _AA_NUM_SAMPLES)
    sheet_lane_depths(first, depth, depth.numel(), out)
    expected = torch.where(
        first < depth.numel(), depth[first.clamp_max(3)], float("inf")
    )
    assert torch.equal(out.flatten().view(torch.int32), expected.view(torch.int32))
