"""A render's start: the collection before its freeze, and its arena.

Both used to pay for work that changes nothing in the common case: a full
garbage collection that walks the whole live scene, and handing the last
render's arena back to the driver only to allocate the same size again. Each
must still do that work whenever it could change the arena's size.
"""

from __future__ import annotations

import gc
import weakref

import pytest
import torch

import algan.animation_timeline.timeline as timeline_module
from algan.utils import memory_utils
from algan.utils.memory_utils import ManualMemory, scene_excluded_from_gc

MIB = 1 << 20


@pytest.fixture
def optimizations(monkeypatch):
    """Every ALGAN_OPT_DISABLE switch off, whatever the environment says."""
    monkeypatch.setattr(timeline_module, "_OPT_DISABLED", frozenset())
    return monkeypatch


@pytest.mark.parametrize(
    ("held", "host_pressured", "disabled", "full"),
    [
        (8 * MIB, False, (), False),
        (0, False, (), False),
        (memory_utils._FROZEN_GARBAGE_DEVICE_BYTES + 1, False, (), True),
        (0, True, (), True),
        (0, False, ("rendergc",), True),
    ],
)
def test_the_collection_is_full_only_when_garbage_could_matter(
    optimizations, held, host_pressured, disabled, full
):
    optimizations.setattr(memory_utils, "_render_device_bytes_held", lambda: held)
    optimizations.setattr(memory_utils, "_host_memory_pressure", lambda: host_pressured)
    optimizations.setattr(timeline_module, "_OPT_DISABLED", frozenset(disabled))

    assert memory_utils._pre_render_collection_is_full() is full


def test_a_cpu_render_holds_no_device_bytes_garbage_could_pin(monkeypatch):
    monkeypatch.setattr(memory_utils, "render_device", lambda: torch.device("cpu"))

    assert memory_utils._render_device_bytes_held() == 0


class _Node:
    pass


@pytest.mark.parametrize("full", [False, True])
def test_the_scene_is_frozen_after_the_collection_either_way(optimizations, full):
    optimizations.setattr(memory_utils, "_pre_render_collection_is_full", lambda: full)
    generations = []
    collect = gc.collect

    def recording_collect(generation=2):
        generations.append(generation)
        return collect(generation)

    # An old cycle, the kind a full collection exists for: freezing and
    # unfreezing parks it in the oldest generation without collecting it.
    cycle = _Node()
    cycle.self = cycle
    alive = weakref.ref(cycle)
    del cycle
    gc.freeze()
    gc.unfreeze()

    optimizations.setattr(gc, "collect", recording_collect)
    with scene_excluded_from_gc():
        frozen = gc.get_freeze_count()
        collected_old_cycle = alive() is None
    optimizations.undo()

    assert generations == [2 if full else 1]
    assert frozen > 0
    assert gc.get_freeze_count() == 0
    if full:
        assert collected_old_cycle
    gc.collect()


def _segment(total, allocated=0, device=0, kind="large"):
    return {
        "device": device,
        "total_size": total,
        "allocated_size": allocated,
        "active_size": allocated,
        "segment_type": kind,
    }


def test_a_cached_arena_is_sized_as_after_emptying_the_cache():
    # 100 MiB free, plus 60 MiB the cache would return (a 40 MiB arena and a
    # 20 MiB scratch segment); the half-used segment and another device's
    # segment stay. Half of 160 MiB is 80 MiB: the 40 MiB segment is not it.
    segments = [
        _segment(40 * MIB),
        _segment(20 * MIB),
        _segment(30 * MIB, allocated=MIB),
        _segment(80 * MIB, device=1),
    ]
    assert memory_utils._cached_arena_bytes(100 * MIB, segments, 0, 0.5) is None
    # A quarter of 160 MiB is exactly the cached 40 MiB segment.
    assert memory_utils._cached_arena_bytes(100 * MIB, segments, 0, 0.25) == 40 * MIB
    # The allocator rounds a large request up to 2 MiB: 39.5 MiB takes the
    # 40 MiB segment too, and is sized from the same free figure.
    portion = (39 * MIB + MIB // 2) / (160 * MIB)
    assert memory_utils._cached_arena_bytes(100 * MIB, segments, 0, portion) == int(
        160 * MIB * portion
    )


def test_a_small_or_unmatched_arena_takes_the_ordinary_path():
    assert memory_utils._cached_arena_bytes(8 * MIB, [_segment(2 * MIB)], 0, 1) is None
    assert (
        memory_utils._cached_arena_bytes(0, [_segment(40 * MIB, kind="small")], 0, 1)
        is None
    )


class _FakeCuda:
    """Just enough of torch.cuda and torch.empty for the arena's start."""

    def __init__(self, monkeypatch, free, segments):
        self.events = []
        events = self.events

        class DeviceContext:
            def __init__(self, device):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        monkeypatch.setattr(torch.cuda, "device", DeviceContext)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(
            torch.cuda, "mem_get_info", lambda *_a: events.append("info") or (free, 0)
        )
        monkeypatch.setattr(
            torch.cuda,
            "memory_snapshot",
            lambda: events.append("snapshot") or segments,
        )
        monkeypatch.setattr(
            torch.cuda, "empty_cache", lambda: events.append("empty_cache")
        )
        real_empty = torch.empty

        def empty(shape, device=None, dtype=None):
            if device is not None and torch.device(device).type == "cuda":
                events.append(("allocate", shape[0]))
                return real_empty(shape, dtype=dtype)
            return real_empty(shape, device=device, dtype=dtype)

        monkeypatch.setattr(torch, "empty", empty)
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)


def test_the_last_arena_is_taken_from_the_cache_and_the_rest_returned(optimizations):
    cuda = _FakeCuda(optimizations, 100 * MIB, [_segment(40 * MIB), _segment(20 * MIB)])
    optimizations.setattr(
        memory_utils.SETTINGS.computing, "available_memory_override", None
    )

    memory = ManualMemory(0.25, device=torch.device("cuda"))

    assert len(memory) == 40 * MIB
    # Allocated before the cache is emptied, so the cached segment is the one
    # it takes; emptied after, so every other free segment goes back as the
    # ordinary sizing's empty_cache would have sent it.
    assert cuda.events == ["info", "snapshot", ("allocate", 40 * MIB), "empty_cache"]


@pytest.mark.parametrize("why", ["switch", "allocator_config", "override"])
def test_the_cache_is_not_consulted_when_it_cannot_be_trusted(optimizations, why):
    cuda = _FakeCuda(optimizations, 100 * MIB, [_segment(40 * MIB)])
    optimizations.setattr(
        memory_utils.SETTINGS.computing,
        "available_memory_override",
        64 * MIB if why == "override" else None,
    )
    if why == "switch":
        optimizations.setattr(
            timeline_module, "_OPT_DISABLED", frozenset({"arenareuse"})
        )
    if why == "allocator_config":
        optimizations.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    memory = ManualMemory(0.25, device=torch.device("cuda"))

    assert "snapshot" not in cuda.events
    expected = 16 * MIB if why == "override" else 25 * MIB
    assert len(memory) == expected
