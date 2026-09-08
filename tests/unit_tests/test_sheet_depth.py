"""The local depth reduction must exclude every crossing of the same surface."""

from __future__ import annotations

import pytest
import torch

from algan.rendering.raytracing.raster_taichi import _AA_NUM_SAMPLES
from algan.rendering.raytracing.sheet_depth_taichi import (
    sheet_depth_lose,
    sheet_lane_depths,
    sheet_lane_depths_inplace,
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


@pytest.mark.parametrize("sheets", [0, 1, 257])
@pytest.mark.parametrize("empty_depth", [False, True])
def test_depth_buffer_reuse_preserves_bits_and_storage_boundaries(sheets, empty_depth):
    from algan.settings._startup import render_device

    init_taichi()
    device = render_device()
    # Include signed zero, non-finite values and a subnormal. This gather must
    # preserve bits, not merely compare equal as floating-point numbers.
    depth = torch.tensor([0.0, -0.0, 1.25, float("inf"), -float("inf"), 1e-40])
    if empty_depth:
        depth = depth[:0]
    n = depth.numel()
    owners = torch.arange(sheets * _AA_NUM_SAMPLES, dtype=torch.int32) % (n + 1)
    expected = torch.full((owners.numel(),), float("inf"))
    if n:
        valid = owners < n
        expected[valid] = depth[owners[valid].long()]
    # Nonzero offsets exercise the Metal import path. Canaries catch writes
    # through the base storage instead of the requested view.
    backing = torch.full(
        (owners.numel() + 18,), -12345, dtype=torch.int32, device=device
    )
    first = backing[9:-9]
    first.copy_(owners)
    depth_backing = torch.cat((torch.tensor([777.0]), depth)).to(device)
    sheet_lane_depths_inplace(first, depth_backing[1:], n)
    actual = first.view(torch.float32).view(sheets, _AA_NUM_SAMPLES)
    assert actual.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
    assert actual.storage_offset() == 9
    assert torch.equal(
        actual.cpu().flatten().view(torch.int32), expected.view(torch.int32)
    )
    assert torch.equal(backing[:9].cpu(), torch.full((9,), -12345, dtype=torch.int32))
    assert torch.equal(backing[-9:].cpu(), torch.full((9,), -12345, dtype=torch.int32))


def test_depth_buffer_reuse_matches_owner_reference_across_live_toggles():
    from algan import SETTINGS
    from algan.rendering.raytracing.sheets import _lane_first_owners
    from algan.settings._startup import render_device

    init_taichi()
    gen = torch.Generator().manual_seed(72)
    # Grow and shrink successive tables; keep each returned tensor alive to
    # ensure a subsequent call cannot overwrite a previous result.
    retained = []
    for nb, n in [(1, 7), (257, 2049), (3, 65)]:
        band = torch.randint(nb, (n,), generator=gen)
        mask = torch.randint(256, (n,), generator=gen, dtype=torch.int32)
        depth = torch.rand(n, generator=gen)
        expected = torch.full((nb, _AA_NUM_SAMPLES), float("inf"))
        for i in reversed(range(n)):
            for lane in range(_AA_NUM_SAMPLES):
                if (int(mask[i]) >> lane) & 1:
                    expected[band[i], lane] = depth[i]
        tensors = [value.to(render_device()) for value in (band, mask, depth)]
        for reuse in [False, True, False, True]:
            with SETTINGS.raytracing.experimental.override(
                sheet_sample_depth_kernel=True,
                sheet_depth_reduce_kernel=True,
                sheet_depth_buffer_reuse=reuse,
            ):
                out = _lane_first_owners(*tensors, nb, n)
            retained.append((out, expected))
    for out, expected in retained:
        assert torch.equal(out.cpu().view(torch.int32), expected.view(torch.int32))
