"""Resolver output writes exact records directly into persistent arena storage."""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing.sheet_buffers import finish_sheet_buffers
from algan.rendering.taichi_runtime import init_taichi
from algan.utils.memory_utils import ManualMemory


@pytest.mark.parametrize("empty", [False, True])
def test_sheet_finalizer_preserves_bits_and_survives_forward_reuse(empty):
    init_taichi()
    device = SETTINGS.computing.render_device
    memory = ManualMemory(0, device=device, num_bytes=8192)
    memory._poison = 255
    final = torch.tensor([2, 0, 1], dtype=torch.int64)
    nearest = torch.tensor([0, 1, 2], dtype=torch.int64)
    representative = torch.tensor([2, 0, 1], dtype=torch.int64)
    keys = torch.tensor([(2**25 + 1) << 32 | 0x3F800001, 0x40000003, 0x40400007])
    refs = torch.tensor([2**24 + 1, -2147221505, 7], dtype=torch.int32)
    ab = torch.tensor([[0.2, -0.0], [0.3, 0.7], [0.4, 0.6]], dtype=torch.float32)
    caps = torch.tensor([0.1, 0.5, 2.0], dtype=torch.float32)
    weights = torch.tensor([-0.25, -0.0, 0.75], dtype=torch.float32)
    masks = torch.tensor([0x01000001, -1, 0x7FFFFFFF], dtype=torch.int32)
    sheet_pixels = torch.tensor([1, 1, 9], dtype=torch.int64)
    covered = torch.tensor([1, 9], dtype=torch.int32)
    if empty:
        final, nearest, representative = final[:0], nearest[:0], representative[:0]
        keys, refs, ab, caps = keys[:0], refs[:0], ab[:0], caps[:0]
        weights, masks, sheet_pixels, covered = (
            weights[:0],
            masks[:0],
            sheet_pixels[:0],
            covered[:0],
        )
    expected = (
        keys[nearest[final]],
        refs[representative[final]],
        ab[representative[final]],
        weights,
        masks,
        caps[representative[final]],
        torch.tensor([0] if empty else [0, 2, 3], dtype=torch.int32),
    )
    with memory.temp():
        # Place every source in forward scratch and free it after the copy.
        sources = [
            memory.clone(t.to(device))
            for t in (
                covered,
                final,
                nearest,
                representative,
                keys,
                refs,
                ab,
                caps,
                weights,
                masks,
                sheet_pixels,
            )
        ]
        result = finish_sheet_buffers(memory, *sources)
    assert memory.current_pointer == 0
    memory.get_tensor((memory.get_num_bytes_remaining(),), torch.uint8).fill_(238)
    for actual, reference in zip(result, expected):
        assert actual.untyped_storage()._cdata == memory.data.untyped_storage()._cdata
        # Bitwise, including signed zero and full-width packed integer fields.
        assert torch.equal(
            actual.cpu().view(torch.uint8), reference.contiguous().view(torch.uint8)
        )
