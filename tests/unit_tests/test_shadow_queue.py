"""Arena shadow copies preserve exact indices, masks and visibility layout."""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing.shadow_queue import (
    _gather_shadow_payload,
    _scatter_shadow_visibility,
)
from algan.rendering.taichi_runtime import init_taichi
from algan.utils.memory_utils import ManualMemory


@pytest.mark.parametrize("with_footprint", [False, True])
@pytest.mark.parametrize("with_terminator", [False, True])
@pytest.mark.parametrize("order", [[], [3], [3, 0, 2]])
def test_shadow_gather_is_exact_and_scoped(with_footprint, with_terminator, order):
    init_taichi()
    device = SETTINGS.computing.render_device
    memory = ManualMemory(0, device=device, num_bytes=8192)
    memory._poison = 255
    position = torch.arange(12, dtype=torch.float32).view(4, 3)
    smooth = position * 0.1
    face = position * -0.5
    # These values catch integer->float->integer copying on MPS.
    frame = torch.tensor([2**25 + 1, 2**25 + 3, 1, 9], dtype=torch.int32)
    mask = torch.tensor([-2147221505, 0x7FFFFFFF, -1, 0x01000001], dtype=torch.int32)
    footprint = torch.arange(24, dtype=torch.float32).view(4, 6)
    terminator = position * 0.02
    if not with_footprint:
        footprint = torch.full((1, 6), float("nan"))
    if not with_terminator:
        terminator = torch.full((1, 3), float("nan"))
    inputs_cpu = (position, smooth, face, frame, mask, footprint, terminator)
    inputs = [value.to(device) for value in inputs_cpu]
    indices = torch.tensor(order, dtype=torch.int64, device=device)
    before = memory.get_pointers()
    with memory.temp():
        result = _gather_shadow_payload(
            memory,
            indices,
            *inputs,
            with_footprint=with_footprint,
            with_terminator=with_terminator,
        )
        for i, (actual, source) in enumerate(zip(result, inputs_cpu)):
            enabled = (
                i < 5 or (i == 5 and with_footprint) or (i == 6 and with_terminator)
            )
            if enabled:
                expected = source.index_select(0, indices.cpu())
                assert torch.equal(actual.cpu(), expected)
                assert (
                    actual.untyped_storage()._cdata
                    == memory.data.untyped_storage()._cdata
                )
            else:
                assert actual is inputs[i]
    assert memory.get_pointers() == before


@pytest.mark.parametrize("lights", [1, 3, 8])
@pytest.mark.parametrize("order", [[], [3], [3, 0, 2]])
def test_visibility_scatter_keeps_padding_and_unaccepted_rows(lights, order):
    init_taichi()
    device = SETTINGS.computing.render_device
    memory = ManualMemory(0, device=device, num_bytes=8192)
    destination = memory.get_tensor((5, 24), torch.float32)
    destination.fill_(1.0)
    indices = torch.tensor(order, dtype=torch.int64, device=device)
    source = (
        torch.arange(len(order) * lights * 3, dtype=torch.float32).view(
            len(order), lights, 3
        )
        / 100
    )
    expected = torch.ones(5, 24)
    for i, row in enumerate(order):
        expected[row, : lights * 3] = source[i].reshape(-1)
    assert (
        _scatter_shadow_visibility(destination, indices, source.to(device))
        is destination
    )
    assert torch.equal(destination.cpu(), expected)
