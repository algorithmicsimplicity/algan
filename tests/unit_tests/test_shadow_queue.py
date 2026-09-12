"""Queue bounds, scheduling-only settings, and the fixed-order fan reduction."""

from __future__ import annotations

import pytest
import torch

from algan.errors import AlganConfigurationError
from algan.rendering.raytracing import raster_taichi as kernels
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.shadow_queue import (
    _event_capacity,
    _fan_layout,
    make_shadow_tracer,
)
from algan.rendering.taichi_runtime import init_taichi
from algan.settings import SETTINGS
from algan.settings._startup import render_device
from algan.utils.memory_utils import ManualMemory


def test_queue_switch_is_live_and_experimental(monkeypatch):
    SETTINGS.raytracing.experimental.shadow_ray_parallel = False
    assert make_shadow_tracer(None) is kernels.raster_shadow_trace
    SETTINGS.raytracing.experimental.shadow_ray_parallel = True
    assert rt_settings.shadow_ray_parallel
    assert callable(make_shadow_tracer(None))
    with pytest.raises(AlganConfigurationError):
        SETTINGS.raytracing.shadow_ray_parallel = False


def test_light_layout_bounds_actual_packed_fans_and_invalidates_mutation():
    col = torch.zeros((2, 3, 18))
    col[:, 1:, 11] = 0.2
    col[:, 1, 16] = 3
    col[:, 2, 16] = 19  # greater than the usual budget
    col[:, 1:, 17] = 1
    offsets, layout, slots = _fan_layout(col, 3, 2, 0)
    assert offsets.tolist() == [0, 4, 7, 26]
    assert slots == 26
    assert layout[4:7].tolist() == [[0, 4], [1, 4], [2, 4]]
    assert _fan_layout(col, 3, 2, 1)[0].tolist() == [0, 4, 5, 6]
    col[1, 2, 16] = 23
    assert _fan_layout(col, 3, 2, 0)[2] == 30


def test_queue_capacity_accounts_for_sort_storage_and_tiny_arenas():
    arena = ManualMemory(0, device="cpu", num_bytes=65536)
    ordinary = _event_capacity(arena, 10000, 9, False)
    sorted_capacity = _event_capacity(arena, 10000, 9, True)
    assert 0 < sorted_capacity < ordinary < 10000
    arena.current_pointer = len(arena) - 8
    assert _event_capacity(arena, 10000, 9, True) == 0


def test_reduce_preserves_masked_rgb_fans_and_adaptive_early_exit():
    init_taichi()
    device = render_device()
    # Four events, one hard light with four taps. All untraced result slots
    # are poisoned, so reading an invalid or adaptively-skipped tap fails.
    valid = torch.tensor(
        [[2, 2, 0, 0], [2, 2, 2, 0], [2, 2, 2, 0], [2, 2, 2, 0]],
        dtype=torch.int32,
        device=device,
    ).flatten()
    occ = torch.full((16, 3), float("nan"), device=device)
    occ[0] = torch.tensor([0.2, 0.4, 0.6], device=device)
    occ[4] = occ[0]  # event 0 exits after the equal diagonal pair
    occ[1] = 0.0
    occ[5] = 1.0
    occ[9] = 0.25
    occ[13] = 0.75  # event 1 visits all four taps
    occ[6] = 0.25
    occ[10] = 0.5
    occ[14] = 0.75  # event 2 skips its uncovered first tap
    offsets = torch.tensor([0, 4], dtype=torch.int32, device=device)
    output = torch.full((6, 1, 3), -7.0, device=device)
    kernels.shadow_queue_reduce(4, 1, 1, offsets, valid, occ, output)
    expected = torch.tensor(
        [[0.8, 0.6, 0.4], [0.5] * 3, [0.5] * 3, [1.0] * 3], device=device
    )
    torch.testing.assert_close(output[1:5, 0], expected, rtol=0, atol=1e-7)
    assert (output[[0, 5]] == -7).all()


@pytest.mark.parametrize("fail", [False, True])
def test_dispatch_chunks_and_restores_arena_even_on_failure(monkeypatch, fail):
    from algan.rendering.raytracing import shadow_queue

    monkeypatch.setattr(rt_settings, "shadow_ray_parallel", True)
    monkeypatch.setattr(shadow_queue, "_SCRATCH_LIMIT", 1200)
    arena = ManualMemory(0, device="cpu", num_bytes=8192)
    offsets = torch.tensor([0, 4], dtype=torch.int32)
    layout = torch.tensor([[i, 0] for i in range(4)], dtype=torch.int32)
    monkeypatch.setattr(shadow_queue, "_fan_layout", lambda *a: (offsets, layout, 4))
    calls = []
    monkeypatch.setattr(
        kernels, "shadow_queue_prepare", lambda *a: calls.append((a[0], a[1]))
    )
    monkeypatch.setattr(kernels, "shadow_queue_trace", lambda *a: None)

    def reduce(*args):
        if fail:
            raise RuntimeError("injected reduction failure")

    monkeypatch.setattr(kernels, "shadow_queue_reduce", reduce)
    params = dict.fromkeys(kernels._RASTER_SHADOW_TRACE_PARAMS)
    params.update(
        num_events=17,
        num_lights=1,
        sec_aa=1,
        secondary=0,
        adaptive_taps=0,
        tri_pos=torch.empty((1, 0, 9)),
    )
    before = arena.get_pointers()

    def call():
        return make_shadow_tracer(arena)(*params.values())

    if fail:
        with pytest.raises(RuntimeError, match="injected reduction failure"):
            call()
    else:
        call()
        assert calls == [(6, 0), (6, 6), (5, 12)]
    assert arena.get_pointers() == before


def test_insufficient_scratch_uses_serial_fallback(monkeypatch):
    from algan.rendering.raytracing import shadow_queue

    monkeypatch.setattr(rt_settings, "shadow_ray_parallel", True)
    monkeypatch.setattr(shadow_queue, "_fan_layout", lambda *a: (None, None, 4))
    arena = ManualMemory(0, device="cpu", num_bytes=64)
    params = dict.fromkeys(kernels._RASTER_SHADOW_TRACE_PARAMS)
    params.update(num_events=17, num_lights=1, sec_aa=1, secondary=0)
    serial = []
    monkeypatch.setattr(kernels, "raster_shadow_trace", lambda *a: serial.append(a))
    before = arena.get_pointers()
    make_shadow_tracer(arena)(*params.values())
    assert len(serial) == 1
    assert serial[0] == tuple(params.values())
    assert arena.get_pointers() == before


def test_prepare_light_major_planes_masks_and_actual_direction_keys():
    init_taichi()
    device = render_device()
    events, lights = 3, 2
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]], device=device
    )
    normal = torch.tensor([[0.0, 0.0, 1.0]] * events, device=device)
    frame = torch.zeros(events, dtype=torch.int32, device=device)
    mask = torch.tensor([15, 1, 0], dtype=torch.int32, device=device)
    dp = torch.zeros((events, 6), device=device)
    toff = torch.zeros((events, 3), device=device)
    lp = torch.tensor([[[0.0, 0.0, 3.0], [0.0, 0.0, 4.0]]], device=device)
    lc = torch.zeros((1, lights, 18), device=device)
    lc[:, :, :3] = 1
    lc[:, 1, 11] = 0.2
    lc[:, 1, 16:] = 3
    offsets, _, slots = _fan_layout(lc, lights, 2, 1)
    sources = torch.tensor([2, 1, -1], dtype=torch.int32, device=device)
    data = torch.full((events * slots, 7), float("nan"), device=device)
    valid = torch.full((events * slots,), -1, dtype=torch.int32, device=device)
    keys = torch.empty(events * slots, dtype=torch.int64, device=device)
    kernels.shadow_queue_prepare(
        events,
        0,
        lights,
        pos,
        normal,
        normal,
        frame,
        mask,
        dp,
        toff,
        lp,
        lc,
        2,
        0,
        1,
        1,
        offsets,
        sources,
        3,
        data,
        valid,
        keys,
        True,
    )
    # Four hard taps (adaptive = 2), then three budgeted soft taps (1).
    assert valid.reshape(slots, events).tolist() == [
        [2, 2, 0],
        [2, 0, 0],
        [2, 0, 0],
        [2, 0, 0],
        [1, 1, 0],
        [1, 1, 0],
        [1, 1, 0],
    ]
    ids = torch.arange(events * slots, device=device)
    accepted = valid != 0
    directions = data[accepted, 3:6]
    octants = (
        (directions >= 0).to(torch.int64) * torch.tensor([1, 2, 4], device=device)
    ).sum(-1)
    light_ids = (ids[accepted] >= events * 4).to(torch.int64)
    expected = (light_ids * 8 + octants) * 4 + sources[ids[accepted] % events] + 1
    assert torch.equal(keys[accepted], expected)
    assert (keys[~accepted] == torch.iinfo(torch.int64).max).all()


def test_layout_uses_host_metadata_despite_unrelated_arena_writes():
    arena = ManualMemory(0, device="cpu", num_bytes=4096)
    host = torch.zeros((1, 1, 18))
    host[..., 11] = 0.2
    host[..., 16] = 3
    table = arena.clone(host)
    table._algan_shadow_fan_metadata = host
    first = _fan_layout(table, 1, 2, 0)
    version = table._version
    arena.get_tensor((10,)).fill_(123)
    assert table._version != version  # proves the arena version-counter hazard
    second = _fan_layout(table, 1, 2, 0)
    assert second[0] is first[0]
    assert second[1] is first[1]
    assert second[2] == 3
    host[..., 16] = 5
    assert _fan_layout(table, 1, 2, 0)[2] == 5
