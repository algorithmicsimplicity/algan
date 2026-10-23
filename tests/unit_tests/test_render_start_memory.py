"""A render's start: the collection before its freeze, and its arena.

Both used to pay for work that changes nothing in the common case: a full
garbage collection that walks the whole live scene, and handing the last
render's arena back to the driver only to allocate the same size again. Each
must still do that work whenever it could change the arena's size or leave a
discarded Scene's garbage frozen through the render.
"""

from __future__ import annotations

import gc
import types
import weakref

import pytest
import torch

import algan.animation_timeline.timeline as timeline_module
from algan.utils import memory_utils
from algan.utils.memory_utils import ManualMemory, scene_excluded_from_gc

MIB = 1 << 20


@pytest.fixture
def optimizations(monkeypatch):
    """Every ALGAN_OPT_DISABLE switch off, whatever the environment says, and
    no Scene discarded since the last full collection.
    """
    monkeypatch.setattr(timeline_module, "_OPT_DISABLED", frozenset())
    monkeypatch.setattr(
        memory_utils, "_scene_discards_collected", memory_utils._scene_discards
    )
    return monkeypatch


@pytest.mark.parametrize(
    ("held", "host_pressured", "disabled", "discarded", "full"),
    [
        (8 * MIB, False, (), False, False),
        (0, False, (), False, False),
        (memory_utils._FROZEN_GARBAGE_DEVICE_BYTES + 1, False, (), False, True),
        (0, True, (), False, True),
        (0, False, ("rendergc",), False, True),
        (0, False, (), True, True),
    ],
)
def test_the_collection_is_full_only_when_garbage_could_matter(
    optimizations, held, host_pressured, disabled, discarded, full
):
    optimizations.setattr(memory_utils, "_render_device_bytes_held", lambda: held)
    optimizations.setattr(memory_utils, "_host_memory_pressure", lambda: host_pressured)
    optimizations.setattr(timeline_module, "_OPT_DISABLED", frozenset(disabled))
    if discarded:
        optimizations.setattr(
            memory_utils, "_scene_discards", memory_utils._scene_discards + 1
        )

    assert memory_utils._pre_render_collection_is_full() is full


def test_a_cpu_render_holds_no_device_bytes_garbage_could_pin(monkeypatch):
    monkeypatch.setattr(memory_utils, "render_device", lambda: torch.device("cpu"))

    assert memory_utils._render_device_bytes_held() == 0


def test_garbage_holding_a_block_holds_its_whole_segment(monkeypatch):
    # A 2 MiB tensor kept by old garbage in the free tail of the last arena's
    # 6 GiB segment: empty_cache cannot return that segment, so the next arena
    # would be gigabytes smaller. Allocated bytes alone would say 2 MiB.
    monkeypatch.setattr(memory_utils, "render_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda _device: {
            "allocated_bytes.all.current": 2 * MIB,
            "inactive_split_bytes.all.current": (6 << 30) - 2 * MIB,
        },
    )

    held = memory_utils._render_device_bytes_held()

    assert held == 6 << 30
    assert held > memory_utils._FROZEN_GARBAGE_DEVICE_BYTES


@pytest.mark.parametrize("how", ["manager_reset", "scene_reset", "terminate"])
def test_a_discarded_scene_gets_one_full_collection(optimizations, how):
    from algan import Scene
    from algan.scene_manager import SceneManager

    optimizations.setattr(memory_utils, "_render_device_bytes_held", lambda: 0)
    optimizations.setattr(memory_utils, "_host_memory_pressure", lambda: False)
    scene = SceneManager.reset()
    with scene_excluded_from_gc():
        pass
    # Rendering the same Scene again keeps the young-generation collection.
    assert not memory_utils._pre_render_collection_is_full()

    if how == "manager_reset":
        SceneManager.reset()
    elif how == "scene_reset":
        scene.reset()
    else:
        with Scene():
            pass
    assert memory_utils._pre_render_collection_is_full()
    with scene_excluded_from_gc():
        pass
    assert not memory_utils._pre_render_collection_is_full()
    SceneManager.reset()


def test_resetting_an_empty_manager_discards_nothing(optimizations):
    from algan.scene_manager import SceneManager

    optimizations.setattr(SceneManager.instance(), "_scene_stack", [])
    before = memory_utils._scene_discards
    SceneManager.reset()
    assert memory_utils._scene_discards == before


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


def _segment(
    total, allocated=0, device=0, kind="large", pool=(0, 0), stream=0, expandable=False
):
    return {
        "device": device,
        "total_size": total,
        "allocated_size": allocated,
        "active_size": allocated,
        "segment_type": kind,
        "segment_pool_id": pool,
        "stream": stream,
        "is_expandable": expandable,
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


@pytest.mark.parametrize("odd", [{"pool": (1, 0)}, {"expandable": True}])
def test_a_private_pool_or_expandable_free_segment_falls_back(odd):
    # empty_cache releases those on terms of their own, so the free figure
    # cannot be read off the snapshot at all.
    segments = [_segment(40 * MIB), _segment(20 * MIB, **odd)]
    assert memory_utils._cached_arena_bytes(100 * MIB, segments, 0, 0.25) is None


def test_only_a_segment_on_the_current_stream_is_taken():
    segments = [_segment(40 * MIB, stream=7), _segment(20 * MIB)]
    assert memory_utils._cached_arena_bytes(100 * MIB, segments, 0, 0.25, 0) is None
    assert memory_utils._cached_arena_bytes(100 * MIB, segments, 0, 0.25, 7) == (
        40 * MIB
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
        self.history = False
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
        monkeypatch.setattr(
            torch.cuda,
            "current_stream",
            lambda *_a: types.SimpleNamespace(cuda_stream=0),
        )
        monkeypatch.setattr(
            torch._C, "_cuda_isHistoryEnabled", lambda: self.history, raising=False
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


@pytest.mark.parametrize("why", ["switch", "allocator_config", "override", "history"])
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
    if why == "history":
        cuda.history = True

    memory = ManualMemory(0.25, device=torch.device("cuda"))

    assert "snapshot" not in cuda.events
    expected = 16 * MIB if why == "override" else 25 * MIB
    assert len(memory) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_real_cuda_arena_is_reused_at_the_ordinary_size(optimizations):
    """The arena taken from the cache is the last one's block, at the size the
    ordinary path measures. Other processes can move the free figure between
    the steps, so a few attempts are allowed.
    """
    from algan.utils.memory_utils import get_num_available_bytes

    optimizations.setattr(
        memory_utils.SETTINGS.computing, "available_memory_override", None
    )
    optimizations.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    optimizations.delenv("PYTORCH_ALLOC_CONF", raising=False)
    device = torch.device("cuda")
    portion = 0.05
    seen = []
    for _attempt in range(3):
        first = ManualMemory(portion, device=device)
        pointer = first.data.data_ptr()
        first.data = None
        del first
        second = ManualMemory(portion, device=device)
        size = len(second)
        reused = second.data.data_ptr() == pointer
        second.data = None
        del second
        ordinary = int(get_num_available_bytes(device) * portion)
        seen.append((size, ordinary, reused))
        if reused and size == ordinary:
            return
    pytest.fail(f"arena (size, ordinary size, reused) per attempt: {seen}")


def test_a_cached_arena_on_a_large_card_is_the_clamped_size():
    """The cache path clamps as ``ManualMemory`` does.

    Otherwise a 48 GB card's arena, taken from the cache, would skip the
    8 GiB addressing clamp and crash the first arena-packed launch.
    """
    free = 40 << 30
    clamped = memory_utils._MAX_ARENA_BYTES
    rounded = -(-clamped // (2 * MIB)) * (2 * MIB)
    segments = [_segment(rounded)]
    assert memory_utils._cached_arena_bytes(free, segments, 0, 0.4) == clamped
