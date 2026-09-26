"""Host-only memory reclaims back off while they free nothing.

``release_torch_memory`` reclaims (a garbage collection, the torch cache, the
native heaps) whenever the host is short of RAM. On a desktop that shortage is
mostly other processes', and a render reclaims at every chunk boundary, so an
ineffective reclaim doubles how many following host-only reclaims are skipped;
one that frees memory resets it. GPU pressure and forced reclaims never skip.
"""

from __future__ import annotations

import types

import pytest
import torch


@pytest.fixture
def host_only(monkeypatch):
    from algan.utils import memory_utils as mu

    monkeypatch.setattr(mu, "_HOST_RECLAIM_MAX_BACKOFF", 32)
    events = []
    available = {"now": 2 << 30, "gain": 0}

    def status():
        return types.SimpleNamespace(
            physical_total=16 << 30, physical_available=available["now"]
        )

    def collect():
        events.append("gc")
        available["now"] += available["gain"]

    monkeypatch.setattr(mu, "_host_memory_pressure", lambda: True)
    monkeypatch.setattr(mu, "_gpu_memory_pressure", lambda: False)
    monkeypatch.setattr(mu, "_windows_memory_status", status)
    monkeypatch.setattr(mu, "_cached_cuda_program_has_headroom", lambda: True)
    monkeypatch.setattr(mu, "_reclaimable_cuda_bytes", lambda: 0)
    monkeypatch.setattr(mu.gc, "collect", collect)
    monkeypatch.setattr(mu, "_malloc_trim", lambda: None)
    monkeypatch.setattr(
        mu, "_reset_quadrants_runtime_for_memory_pressure", lambda: False
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.mps, "is_available", lambda: False)
    return mu, events, available


def test_ineffective_host_reclaims_back_off_exponentially(host_only):
    mu, events, _available = host_only
    ran = []
    for i in range(100):
        before = len(events)
        mu.release_torch_memory(force_gc=False)
        if len(events) > before:
            ran.append(i)
    # Skips of 1, 2, 4, 8, 16, 32, 32, ... after each reclaim that freed nothing.
    assert ran == [0, 2, 5, 10, 19, 36, 69]


def test_an_effective_host_reclaim_resets_the_backoff(host_only):
    mu, events, available = host_only
    available["gain"] = 256 << 20
    for _ in range(10):
        mu.release_torch_memory(force_gc=False)
    assert len(events) == 10


@pytest.mark.parametrize("reason", ["forced", "gpu"])
def test_forced_and_gpu_reclaims_never_back_off(host_only, monkeypatch, reason):
    mu, events, _available = host_only
    mu.release_torch_memory(force_gc=False)  # arms the backoff
    if reason == "gpu":
        monkeypatch.setattr(mu, "_gpu_memory_pressure", lambda: True)
    for _ in range(5):
        mu.release_torch_memory(force_gc=reason == "forced")
    assert len(events) == 6


def test_opt_disable_restores_every_host_reclaim(host_only, monkeypatch):
    import algan.animation_timeline.timeline as tl

    mu, events, _available = host_only
    monkeypatch.setattr(tl, "_OPT_DISABLED", frozenset({"hostbackoff"}))
    for _ in range(6):
        mu.release_torch_memory(force_gc=False)
    assert len(events) == 6


def test_no_windows_telemetry_keeps_every_host_reclaim(host_only, monkeypatch):
    mu, events, _available = host_only
    monkeypatch.setattr(mu, "_windows_memory_status", lambda: None)
    for _ in range(6):
        mu.release_torch_memory(force_gc=False)
    assert len(events) == 6
