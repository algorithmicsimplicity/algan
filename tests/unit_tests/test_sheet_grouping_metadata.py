"""Integration and integer-precision regressions for local sheet grouping."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from algan import SETTINGS
from algan.rendering import taichi_runtime
from algan.rendering.raytracing import sheets
from algan.rendering.raytracing.sheet_grouping import (
    class_groups,
    prepare_fragments,
    unique_sorted_ids,
)


@pytest.fixture(autouse=True)
def _compiler():
    taichi_runtime.init_taichi()


@pytest.mark.parametrize("local", [False, True])
def test_integrated_grouping_matches_cpu_oracle(local, monkeypatch):
    monkeypatch.setattr(sheets, "_local_mps_grouping", lambda tensor: local)
    device = SETTINGS.computing.render_device
    band = torch.tensor([1, 0, 1, 0, 2, 2], dtype=torch.int64)
    cls = torch.tensor([2**24, 2**24 + 1, 2**24 + 1, 2**24, 7, 7])
    starts = torch.tensor([True, False, False, False, True, False])
    keys, inv = torch.unique(band * (1 << 25) + cls, return_inverse=True)
    actual = sheets._sheet_class_groups(
        band.to(device), cls.to(device), starts.to(device), 3
    )
    assert actual[0] == len(keys)
    assert torch.equal(actual[1].cpu(), inv)
    assert torch.equal(actual[2].cpu(), keys // (1 << 25))
    ordered = torch.tensor([2**35, 2**35, 2**35 + 1, 2**35 + 7])
    result = sheets._unique_sorted_ids(ordered.to(device))
    reference = torch.unique_consecutive(ordered, return_inverse=True)
    assert all(torch.equal(a.cpu(), b) for a, b in zip(result, reference))


def test_integer_grouping_accepts_color_tensor_views():
    from algan.constants.color import Color

    device = SETTINGS.computing.render_device
    bands = torch.tensor([0, 1, 0, 1, 2], dtype=torch.int64, device=device).as_subclass(
        Color
    )
    classes = torch.tensor(
        [17, 19, 17, 19, 2], dtype=torch.int64, device=device
    ).as_subclass(Color)
    starts = torch.tensor([True, False, False, False, True], device=device)
    count, inverse, grouped = class_groups(bands, classes, starts)
    assert count == 3
    assert torch.equal(inverse.cpu(), torch.tensor([0, 1, 0, 1, 2]))
    assert torch.equal(grouped.cpu(), torch.arange(3))
    keys = torch.tensor([2**40, 2**40, 2**40 + 1], device=device).as_subclass(Color)
    unique, inverse = unique_sorted_ids(keys)
    assert torch.equal(unique.cpu(), torch.tensor([2**40, 2**40 + 1]))
    assert torch.equal(inverse.cpu(), torch.tensor([0, 0, 1]))


@pytest.mark.parametrize(
    ("device", "count", "dtype", "ndim", "contiguous", "expected"),
    [
        ("mps", 8192, torch.int64, 1, True, True),
        ("cpu", 8192, torch.int64, 1, True, False),
        ("cuda", 8192, torch.int64, 1, True, False),
        ("mps", 0, torch.int64, 1, True, False),
        ("mps", 8191, torch.int64, 1, True, False),
        ("mps", 2**31, torch.int64, 1, True, False),
        ("mps", 8192, torch.float32, 1, True, False),
        ("mps", 8192, torch.int64, 2, True, False),
        ("mps", 8192, torch.int64, 1, False, False),
    ],
)
def test_gate_guards_local_bounded_integer_inputs(
    monkeypatch, device, count, dtype, ndim, contiguous, expected
):
    # Metadata-only stand-in: overflow tests never allocate a huge tensor.
    tensor = SimpleNamespace(
        device=torch.device(device),
        numel=lambda: count,
        dtype=dtype,
        ndim=ndim,
        is_contiguous=lambda: contiguous,
    )
    monkeypatch.setattr(taichi_runtime, "_live_arch", lambda: object())
    monkeypatch.setattr(taichi_runtime, "taichi_launch_is_local", lambda device: True)
    with SETTINGS.raytracing.experimental.override(sheet_mps_grouping=True):
        assert sheets._local_mps_grouping(tensor) is expected
        monkeypatch.setattr(
            taichi_runtime, "taichi_launch_is_local", lambda device: False
        )
        assert not sheets._local_mps_grouping(tensor)
        monkeypatch.setattr(taichi_runtime, "_live_arch", lambda: None)
        assert not sheets._local_mps_grouping(tensor)
    with SETTINGS.raytracing.experimental.override(sheet_mps_grouping=False):
        assert not sheets._local_mps_grouping(tensor)


@pytest.mark.parametrize("time_start", [0, 3, 7])
@pytest.mark.parametrize("count", [0, 1, 8193])
def test_fragment_metadata_matches_torch(time_start, count, monkeypatch):
    device = SETTINGS.computing.render_device
    gen = torch.Generator().manual_seed(37)
    ppf = 3840 * 2160
    pixels = torch.randint(0, ppf * 5, (count,), generator=gen)
    depths = torch.rand(count, generator=gen) * 100
    refs = torch.randint(-17, 11, (count,), generator=gen, dtype=torch.int32)
    masks = torch.randint(0, 1 << 16, (count,), generator=gen, dtype=torch.int32)
    objects = torch.arange(33, dtype=torch.int64).reshape(3, 11) + (1 << 34)
    keys = (pixels << 32) | depths.view(torch.int32).to(torch.int64)
    frame = pixels // ppf
    safe = refs.clamp_min(0).to(torch.int64)
    pos = torch.arange(count, dtype=torch.int64)
    surface = objects[(frame + time_start) % 3, safe]
    bit = sheets.AA_BACKFACE_BIT
    groups = torch.where(refs >= 0, surface * 2 + ((masks & bit) != 0), -(pos + 2))
    expected = pixels, depths, frame, refs >= 0, safe, pos, groups
    inputs = keys.to(device), refs.to(device), masks.to(device), objects.to(device)
    actual = prepare_fragments(*inputs, ppf, time_start, bit)
    assert all(
        a.dtype == b.dtype and torch.equal(a.cpu(), b) for a, b in zip(actual, expected)
    )
    # The real integration is exercised on both the CPU test arm and Metal.
    monkeypatch.setattr(sheets, "_local_mps_grouping", lambda tensor: True)
    actual = sheets._sheet_fragment_metadata(*inputs, ppf, time_start)
    assert all(
        a.dtype == b.dtype and torch.equal(a.cpu(), b) for a, b in zip(actual, expected)
    )


def test_int32_prefix_scan_preserves_low_counter_bits():
    counts = torch.tensor([1 << 24, *([1] * 33)], dtype=torch.int32)
    expected = torch.cumsum(counts, 0, dtype=torch.int32)
    actual = torch.cumsum(
        counts.to(SETTINGS.computing.render_device), 0, dtype=torch.int32
    )
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.skipif(
    SETTINGS.computing.render_device.type != "mps", reason="Metal import bridge"
)
def test_class_group_inputs_never_stage_through_the_host():
    from algan.rendering.mps_zero_copy import STATS

    bands = torch.tensor([1, 0, 1, 0, 2], dtype=torch.int64, device="mps")
    classes = torch.tensor(
        [2**24, 2**24 + 1, 2**24, 2**24 + 1, 7], dtype=torch.int64, device="mps"
    )
    starts = torch.tensor([True, False, False, False, True], device="mps")
    before = dict(STATS)
    count, inverse, groups = class_groups(bands, classes, starts)
    assert count == 3
    assert torch.equal(inverse.cpu(), torch.tensor([1, 0, 1, 0, 2]))
    assert torch.equal(groups.cpu(), torch.arange(3))
    assert STATS["converted_launches"] >= before["converted_launches"] + 2
    assert STATS["staged_arguments"] == before["staged_arguments"]
    assert STATS["host_arguments"] == before["host_arguments"]


def test_fragment_metadata_preserves_depth_bits_and_storage_offsets():
    device = SETTINGS.computing.render_device
    # Include signed zero, infinities and a NaN payload: unpacking is bitwise,
    # not arithmetic, even though the renderer normally supplies finite depth.
    depth_bits = torch.tensor(
        [0, -(1 << 31), 0x3F800001, 0x7F800000, -0x800000, 0x7FC01234],
        dtype=torch.int32,
    )
    pixels = torch.tensor([0, 1, 101, 202, 303, 404], dtype=torch.int64)
    packed = (pixels << 32) | (depth_bits.to(torch.int64) & 0xFFFFFFFF)

    def offset_copy(value):
        padded = torch.empty(value.numel() + 7, dtype=value.dtype, device=device)
        view = padded[7:]
        view.copy_(value)
        assert view.is_contiguous() and view.storage_offset() == 7
        return view

    keys = offset_copy(packed)
    refs = offset_copy(torch.tensor([0, -1, 1, -3, 0, 1], dtype=torch.int32))
    masks = offset_copy(torch.zeros(6, dtype=torch.int32))
    objects = torch.tensor([[2, 3], [4, 5]], dtype=torch.int64, device=device)
    actual = prepare_fragments(keys, refs, masks, objects, 101, 1, sheets.AA_BACKFACE_BIT)
    assert torch.equal(actual[0].cpu(), pixels)
    assert torch.equal(actual[1].cpu().view(torch.int32), depth_bits)
    assert torch.equal(actual[2].cpu(), pixels // 101)
    assert torch.equal(actual[4].cpu(), torch.tensor([0, 0, 1, 0, 0, 1]))


@pytest.mark.parametrize("bad", ["dtype", "shape", "stride", "device"])
def test_class_grouping_rejects_unsupported_inputs_before_launch(bad):
    device = SETTINGS.computing.render_device
    bands = torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=device)
    classes = torch.zeros_like(bands)
    starts = torch.tensor([True, False, True, False], device=device)
    exception = ValueError
    if bad == "dtype":
        classes = classes.float()
        exception = TypeError
    elif bad == "shape":
        classes = classes.reshape(2, 2)
    elif bad == "stride":
        classes = torch.zeros(8, dtype=torch.int64, device=device)[::2]
    else:
        classes = torch.empty(4, dtype=torch.int64, device="meta")
    with pytest.raises(exception):
        class_groups(bands, classes, starts)


@pytest.mark.parametrize("field", ["keys", "refs", "masks", "objects"])
def test_fragment_metadata_rejects_floating_identifiers(field):
    device = SETTINGS.computing.render_device
    inputs = {
        "keys": torch.zeros(1, dtype=torch.int64, device=device),
        "refs": torch.zeros(1, dtype=torch.int32, device=device),
        "masks": torch.zeros(1, dtype=torch.int32, device=device),
        "objects": torch.zeros((1, 1), dtype=torch.int64, device=device),
    }
    inputs[field] = inputs[field].float()
    with pytest.raises(TypeError, match=field):
        prepare_fragments(**inputs, pixels_per_frame=1, time_start=0, backface_bit=1)


def test_fragment_metadata_rejects_empty_object_table_for_nonempty_stream():
    device = SETTINGS.computing.render_device
    keys = torch.zeros(1, dtype=torch.int64, device=device)
    objects = torch.empty((0, 1), dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="object table"):
        prepare_fragments(keys, keys, keys, objects, 1, 0, 1)
