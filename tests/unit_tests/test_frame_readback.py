"""Frame consumers own ready, top-down CPU storage after the arena is reused."""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.post_processing.post_process import _frames_to_host


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("pinned", [False, True])
def test_readback_is_ready_and_survives_reuse_on_a_nondefault_stream(count, pinned):
    if not torch.cuda.is_available():
        pytest.skip("CUDA readback")
    source = (
        torch.arange(count * 7 * 11 * 4, dtype=torch.int32)
        .to(torch.uint8)
        .reshape(count, 7, 11, 4)
    )
    expected = source.flip(-3)
    snapshot = SETTINGS.snapshot()
    try:
        SETTINGS.raytracing.experimental.pinned_frame_readback = pinned
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            frame = source.cuda()
            first = _frames_to_host(frame)
            # No caller-side sync: returned bytes must already be consumable.
            torch.testing.assert_close(first, expected, rtol=0, atol=0)
            frame.zero_()
            second = _frames_to_host(frame)
        assert first.device.type == "cpu"
        assert first.is_pinned() == pinned
        torch.testing.assert_close(first, expected, rtol=0, atol=0)
        assert not second.any()
        assert first.data_ptr() != second.data_ptr()
    finally:
        SETTINGS.restore(snapshot)


def test_cpu_readback_keeps_independent_storage():
    frame = torch.ones((2, 7, 11, 3), dtype=torch.uint8)
    result = _frames_to_host(frame)
    frame.zero_()
    assert result.all()


def test_pinning_failure_keeps_the_pageable_fallback(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA readback")
    frame = torch.ones((1, 7, 11, 3), dtype=torch.uint8, device="cuda")
    snapshot = SETTINGS.snapshot()
    try:
        SETTINGS.raytracing.experimental.pinned_frame_readback = True

        def unavailable(*args, **kwargs):
            raise RuntimeError("host page locking is unavailable")

        monkeypatch.setattr(torch, "empty_like", unavailable)
        result = _frames_to_host(frame)
        assert not result.is_pinned()
        assert result.all()
    finally:
        SETTINGS.restore(snapshot)
