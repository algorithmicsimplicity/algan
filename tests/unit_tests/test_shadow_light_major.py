"""Changing CUDA lane scheduling must preserve every shadow visibility cell."""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing import raster_taichi as kernels
from algan.rendering.raytracing import settings as rt_module
from algan.rendering.raytracing import shadow_queue
from algan.settings._startup import render_device


@pytest.mark.parametrize(
    ("events", "lights", "device", "expected"),
    [
        (16384, 2, "cuda", True),
        (16383, 2, "cuda", False),
        (16384, 1, "cuda", False),
        (16384, 2, "cpu", False),
        (16384, 2, "mps", False),
    ],
)
def test_light_major_dispatch_keeps_small_and_non_cuda_queues(
    monkeypatch, events, lights, device, expected
):
    monkeypatch.setattr(shadow_queue.rt_settings, "shadow_light_major", True)
    assert (
        shadow_queue._use_light_major(events, lights, torch.device(device)) is expected
    )
    monkeypatch.setattr(shadow_queue.rt_settings, "shadow_light_major", False)
    assert not shadow_queue._use_light_major(events, lights, torch.device(device))


@pytest.mark.parametrize(
    ("scene_name", "budget", "device_glue"),
    [
        ("soft", 0, False),
        ("soft", 16, False),
        ("graphics", 16, False),
        ("graphics", 16, True),
    ],
)
def test_light_major_keeps_primary_and_secondary_visibility_exact(
    monkeypatch, tmp_path, scene_name, budget, device_glue
):
    if render_device().type != "cuda":
        pytest.skip("CUDA scheduling optimization")
    from benchmarks._shadow_queue_check import render_graphics, render_soft

    render = render_soft if scene_name == "soft" else render_graphics
    counts = [0, 0]
    params = kernels._RASTER_SHADOW_TRACE_PARAMS
    vis_slot = params.index("shadow_vis")
    order_slot = params.index("light_major")
    secondary_slot = params.index("secondary")
    snapshot = SETTINGS.snapshot()

    def make_checked(memory, sort_sources=None, merged=None):
        def trace(*call):
            args = list(call)
            args[order_slot] = False
            kernels.raster_shadow_trace(*args)
            expected = args[vis_slot].clone()
            before = memory.get_pointers()
            args[order_slot] = True
            kernels.raster_shadow_trace(*args)
            torch.testing.assert_close(args[vis_slot], expected, rtol=0, atol=0)
            assert memory.get_pointers() == before
            counts[int(args[secondary_slot])] += 1

        return trace

    # With the device glue (wf_device_shadow_glue) the bounce drain's events
    # trace through deferred_shadow_trace: the same fan, launched over every
    # event row with the count on the device, scheduled the same two ways. Its
    # table is sized for every row and written below the count only; the rest
    # is whatever the arena held (an earlier bounce's -1 map words read as NaN),
    # so only the written slots are compared.
    deferred = kernels.deferred_shadow_trace
    deferred_params = kernels._DEFERRED_SHADOW_TRACE_PARAMS
    deferred_vis = deferred_params.index("shadow_vis")
    deferred_order = deferred_params.index("light_major")
    deferred_count = deferred_params.index("event_count")

    def checked_deferred(*call):
        args = list(call)
        written = int(args[deferred_count][0])
        args[deferred_order] = False
        deferred(*args)
        expected = args[deferred_vis][:written].clone()
        args[deferred_order] = True
        deferred(*args)
        torch.testing.assert_close(
            args[deferred_vis][:written], expected, rtol=0, atol=0
        )
        counts[1] += 1

    monkeypatch.setattr(rt_module, "wf_device_shadow_glue", device_glue)
    monkeypatch.setattr(shadow_queue, "make_shadow_tracer", make_checked)
    monkeypatch.setattr(kernels, "deferred_shadow_trace", checked_deferred)
    try:
        SETTINGS.raytracing.experimental.set(
            shadow_ray_budget=budget, shadow_bounce_rays=0 if budget == 0 else 1
        )
        render(str(tmp_path / "parity"), resolution=(128, 72))
        assert all(counts), f"Missing primary or secondary coverage: {counts}"
    finally:
        SETTINGS.restore(snapshot)
