"""Disk-cache reuse must not be confused with first-in-process warm-up."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from algan.utils import profiling_utils as profiler
from algan.utils import taichi_source_key as source_key


def _stats(**updates):
    return {
        "hits": 0,
        "misses": 0,
        "poisoned": 0,
        "verified": 0,
        "keyed": 0,
        "key_seconds": 0.0,
        **updates,
    }


def _result(cache):
    return {
        "total": 1.0,
        "times": {},
        "counts": {},
        "exclusive_times": {},
        "launch_times": {},
        "device_sync_times": {},
        "launches": [("example", 1, 10, 0.1)],
        "source_key_cache": cache,
        "peak_alloc_mb": 0,
        "peak_reserved_mb": 0,
        "scene_stats": [],
        "cprofile_path": "disabled",
        "kernel_gpu": {},
    }


def test_first_pass_with_hits_is_not_reported_as_jit_compilation():
    cache = {
        "skipped_reason": None,
        "authoring": _stats(),
        "render": _stats(hits=34, keyed=34),
    }
    text = profiler.format_report([_result(cache), _result(cache)])
    assert "RUN 1 (first pass in this process)" in text
    assert "RUN 2 (repeat pass in this process)" in text
    assert "render: 34 hits, 0 misses" in text
    assert "includes Taichi JIT compile" not in text
    assert "incl. JIT compile" not in text
    assert "steady state" not in text
    assert "backend cache may still hit" in text


@pytest.mark.parametrize("reason", [None, "disabled by ALGAN_TAICHI_SOURCE_KEY=0"])
def test_cache_reporting_distinguishes_unavailable_from_zero_lookups(reason):
    cache = {"skipped_reason": reason, "authoring": _stats(), "render": _stats()}
    text = profiler._format_source_key_cache(cache)
    if reason is None:
        assert "render: 0 hits, 0 misses" in text
        assert "unavailable" not in text
    else:
        assert f"unavailable: {reason}" in text
        assert "0 hits" not in text
    assert "not recorded" in profiler._format_source_key_cache(None)


def test_each_pass_reports_authoring_and_render_deltas_without_clearing_counters(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(source_key, "STATS", _stats(hits=10, keyed=10))
    monkeypatch.setattr(source_key, "skipped_reason", lambda: None)
    scene = SimpleNamespace(
        set_video_settings=lambda _: None,
        _recorded_end_time_for_render=lambda: 1.0,
    )
    monkeypatch.setattr(profiler.SceneManager, "reset", lambda: scene)
    monkeypatch.setattr(profiler, "TIMERS", profiler.StageTimers())
    monkeypatch.setattr(profiler, "KERNEL_PROFILER", False)
    monkeypatch.setattr(profiler.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(profiler, "_sync_devices", lambda: None)
    monkeypatch.setattr(profiler, "_collect_taichi_kernel_gpu", lambda: {})

    def author():
        source_key.STATS["misses"] += 2
        source_key.STATS["keyed"] += 2
        source_key.STATS["key_seconds"] += 0.125

    def save(path, **options):
        source_key.STATS["hits"] += 3
        source_key.STATS["keyed"] += 3
        source_key.STATS["key_seconds"] += 0.25
        return SimpleNamespace(status="rendered", output_path=path)

    monkeypatch.setattr(profiler.Scene, "save_video", save)
    for _ in range(2):
        result = profiler.run_once(
            author, None, telemetry=False, output_directory=tmp_path
        )
        cache = result["source_key_cache"]
        assert cache["authoring"] == _stats(misses=2, keyed=2, key_seconds=0.125)
        assert cache["render"] == _stats(hits=3, keyed=3, key_seconds=0.25)
    assert source_key.STATS["hits"] == 16
    assert source_key.STATS["misses"] == 4
    assert source_key.STATS["keyed"] == 20


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_report_names_selected_device_even_if_cuda_is_available(monkeypatch, device):
    monkeypatch.setattr(profiler, "render_device", lambda: profiler.torch.device(device))
    monkeypatch.setattr(profiler.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        profiler.torch.cuda,
        "get_device_name",
        lambda *_: pytest.fail("not rendering on CUDA"),
    )
    assert f"device: {device}\n" in profiler.format_report([])
