"""How projection sources and host-built tables are uploaded.

``scene_builder._to_device`` skips the stream synchronisation only for
pageable host memory, which the driver stages before the copy call returns; a
pinned tensor's copy is truly asynchronous, so it must take the blocking copy,
and ``async_source_upload=False`` is the all-blocking A/B arm.
"""

from __future__ import annotations

import pytest
import torch

from algan.rendering.raytracing import scene_builder


class _RecordingTensor(torch.Tensor):
    """Records the ``non_blocking`` of every ``.to``; pinned on demand."""

    pinned = False
    calls: list = []

    def is_pinned(self, *args, **kwargs):
        return type(self).pinned

    def to(self, *args, **kwargs):
        type(self).calls.append(kwargs.get("non_blocking", False))
        return super().to(*args, **kwargs)


@pytest.fixture
def recording():
    _RecordingTensor.calls = []
    _RecordingTensor.pinned = False
    yield _RecordingTensor
    _RecordingTensor.pinned = False


@pytest.mark.parametrize(
    ("pinned", "enabled", "expected"),
    [(False, True, True), (True, True, False), (False, False, False)],
    ids=["pageable", "pinned", "switched-off"],
)
def test_source_uploads_skip_the_sync_only_for_pageable_memory(
    recording, monkeypatch, pinned, enabled, expected
):
    monkeypatch.setattr(scene_builder, "async_source_upload", enabled)
    recording.pinned = pinned
    value = torch.arange(6.0).as_subclass(recording)
    moved = scene_builder._to_device(value, torch.device("cpu"))
    assert recording.calls == [expected]
    assert torch.equal(moved.as_subclass(torch.Tensor), torch.arange(6.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs pinned memory")
def test_a_really_pinned_source_takes_the_blocking_copy(monkeypatch):
    calls = []
    original = torch.Tensor.to

    def spy(self, *args, **kwargs):
        calls.append(kwargs.get("non_blocking", False))
        return original(self, *args, **kwargs)

    value = torch.arange(6.0).pin_memory()
    monkeypatch.setattr(torch.Tensor, "to", spy)
    moved = scene_builder._to_device(value, torch.device("cuda"))
    assert calls == [False]
    assert torch.equal(moved.cpu(), torch.arange(6.0))
