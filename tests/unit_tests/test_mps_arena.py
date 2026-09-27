"""Managed arena routing, external byte accounting, and native storage lifetime."""

from __future__ import annotations

import gc
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import psutil
import pytest
import torch

from algan.rendering import mps_arena
from algan.settings._startup import render_device
from algan.utils import memory_utils as mu


def test_external_accounting_does_not_load_or_initialize_metal(monkeypatch):
    monkeypatch.setattr(mps_arena, "_NATIVE", None)

    def forbidden(*args):
        pytest.fail("a telemetry query must not initialize a native runtime")

    monkeypatch.setattr(mps_arena.importlib, "import_module", forbidden)
    assert mps_arena.allocated_bytes() == 0


@pytest.mark.parametrize("size", [0, -1, mps_arena.MAX_ARENA_BYTES + 1])
def test_invalid_sizes_are_refused_before_loading_native_code(monkeypatch, size):
    monkeypatch.setattr(mps_arena, "_native", lambda: pytest.fail("loaded native code"))
    with pytest.raises(ValueError, match="size"):
        mps_arena.allocate(size)


@pytest.mark.parametrize("size", [True, False, 1.5, "1024", None])
def test_non_integer_sizes_are_not_truncated(monkeypatch, size):
    monkeypatch.setattr(mps_arena, "_native", lambda: pytest.fail("loaded native code"))
    with pytest.raises(TypeError):
        mps_arena.allocate(size)


def test_missing_native_binary_has_an_actionable_error_not_a_heap_fallback(monkeypatch):
    monkeypatch.setattr(mps_arena, "_NATIVE", None)
    monkeypatch.setattr(mps_arena.sys, "platform", "darwin")

    def missing(*args):
        raise ImportError("not installed")

    monkeypatch.setattr(mps_arena.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="Command Line Tools"):
        mps_arena.allocate(1024)
    assert mps_arena._NATIVE is None


@pytest.mark.parametrize(
    ("managed", "size", "native"),
    [(True, 128, True), (False, 1, False), (True, 0, False)],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:1", "mps"])
def test_only_nonempty_managed_mps_arenas_bypass_torch(
    monkeypatch, managed, size, native, device
):
    calls = []
    ordinary_empty = torch.empty

    def pooled(shape, *, device, dtype):
        calls.append(("torch", shape, str(device), dtype))
        return ordinary_empty(shape, device="cpu", dtype=dtype)

    def standalone(count):
        calls.append(("native", count))
        return ordinary_empty(count, device="cpu", dtype=torch.uint8)

    monkeypatch.setattr(torch, "empty", pooled)
    monkeypatch.setattr(mps_arena, "allocate", standalone)
    arena = mu.ManualMemory(0, device=device, managed=managed, num_bytes=size)
    assert len(arena) == size
    assert len(calls) == 1
    assert calls[0][0] == ("native" if native and device == "mps" else "torch")


def test_mps_clamp_is_applied_before_native_allocation(monkeypatch):
    calls = []

    def allocate(count):
        calls.append(count)
        return torch.empty(0, dtype=torch.uint8)

    monkeypatch.setattr(mps_arena, "allocate", allocate)
    mu.ManualMemory(0, device="mps", num_bytes=1 << 35)
    assert calls == [mu._MPS_MAX_ARENA_BYTES]
    assert mu._MPS_MAX_ARENA_BYTES == mps_arena.MAX_ARENA_BYTES


def test_mps_device_index_is_not_silently_ignored(monkeypatch):
    monkeypatch.setattr(mps_arena, "allocate", lambda count: pytest.fail("allocated"))
    with pytest.raises(ValueError, match="mps:0"):
        mu.ManualMemory(0, device="mps:1", num_bytes=1024)


def test_live_native_bytes_are_counted_after_reclamation_not_added_to_driver(
    monkeypatch,
):
    order = []
    monkeypatch.setenv("ALGAN_MPS_HOST_SHARE", "0")
    monkeypatch.setenv("ALGAN_MPS_MEMORY_CAP", "0")
    monkeypatch.setattr(mu.SETTINGS.computing, "available_memory_override", None)
    monkeypatch.setattr(
        "algan.rendering.mps_zero_copy.clear_import_cache",
        lambda: order.append("imports"),
    )
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: order.append("drain"))
    monkeypatch.setattr(
        torch.mps, "current_allocated_memory", lambda: order.append("torch") or 100
    )
    monkeypatch.setattr(
        mps_arena, "allocated_bytes", lambda: order.append("external") or 200
    )
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 1000)
    monkeypatch.setattr(
        torch.mps,
        "driver_allocated_memory",
        lambda: pytest.fail("double counted native memory"),
    )
    assert (
        mu.get_num_available_bytes("mps") == 1000 - int(1000 * mu._MPS_HEADROOM) - 300
    )
    assert order == ["imports", "drain", "torch", "external"]


def test_native_allocation_errors_are_not_hidden_or_retried(monkeypatch):
    def no_memory(size):
        raise RuntimeError(
            "MPS backend out of memory allocating a standalone render arena"
        )

    monkeypatch.setattr(
        mps_arena, "_native", lambda: SimpleNamespace(allocate=no_memory)
    )
    with pytest.raises(RuntimeError, match="out of memory") as error:
        mu.ManualMemory(0, device="mps", num_bytes=1024)
    assert mu.is_cuda_oom(error.value)


def test_old_torch_metal_dlpack_failure_explains_the_mps_requirement(monkeypatch):
    monkeypatch.setattr(
        mps_arena, "_native", lambda: SimpleNamespace(allocate=lambda size: object())
    )

    def unsupported(capsule):
        raise RuntimeError("Unsupported device_type: 8")

    monkeypatch.setattr(torch, "from_dlpack", unsupported)
    with pytest.raises(RuntimeError, match="PyTorch 2.13"):
        mps_arena.allocate(1024)


@pytest.fixture
def native_mps():
    # CPU CI deliberately retains an older Torch. Actual MPS regression CI
    # uses the supported Metal/DLPack build, and must NOT skip a missing owner.
    if sys.platform != "darwin" or render_device().type != "mps":
        pytest.skip("requires the MPS render arm")
    assert torch.backends.mps.is_available()
    from algan.rendering import mps_zero_copy

    mps_zero_copy.clear_import_cache()
    torch.mps.synchronize()
    gc.collect()
    native = mps_arena._native()
    assert native.MAX_ARENA_BYTES == mu._MPS_MAX_ARENA_BYTES
    before = mps_arena.allocated_bytes()
    yield native
    mps_zero_copy.clear_import_cache()
    torch.mps.synchronize()
    gc.collect()
    assert mps_arena.allocated_bytes() == before


def test_native_capsule_abandoned_or_consumed_has_exactly_one_owner(native_mps):
    before = mps_arena.allocated_bytes()
    capsule = native_mps.allocate(1024)
    assert mps_arena.allocated_bytes() == before + 1024
    del capsule
    assert mps_arena.allocated_bytes() == before
    capsule = native_mps.allocate(1024)
    tensor = torch.from_dlpack(capsule)
    with pytest.raises(RuntimeError):
        torch.from_dlpack(capsule)
    del capsule
    tensor[:4].fill_(9)
    assert tensor[:4].cpu().tolist() == [9] * 4
    assert mps_arena.allocated_bytes() == before + 1024
    del tensor


@pytest.mark.parametrize("size", [0, -1, 1 << 31, True, 1.5])
def test_native_entry_point_validates_before_allocation(native_mps, size):
    with pytest.raises((ValueError, TypeError)):
        native_mps.allocate(size)


def test_typed_offset_views_keep_storage_after_the_arena_dies(native_mps):
    before = mps_arena.allocated_bytes()
    arena = mu.ManualMemory(0, device="mps", num_bytes=4096)
    arena.get_tensor((3,), torch.uint8)
    view = arena.get_tensor((4, 8), torch.float32)
    assert view.storage_offset() > 0
    storage = view.untyped_storage().data_ptr()
    view.copy_(torch.arange(32, device="mps").reshape(4, 8))
    del arena
    gc.collect()
    assert mps_arena.allocated_bytes() == before + 4096
    torch.mps.empty_cache()
    assert view.untyped_storage().data_ptr() == storage
    expected = torch.arange(32).reshape(4, 8).flip((0, 1)).tolist()
    assert view.flip((0, 1)).cpu().tolist() == expected
    assert view.amax().item() == 31
    del view


def test_quadrants_import_keeps_external_storage_after_cache_eviction(native_mps):
    from algan.rendering.mps_zero_copy import clear_import_cache, import_tensor
    from algan.rendering.taichi_runtime import ensure_taichi_for_render

    ensure_taichi_for_render()
    before = mps_arena.allocated_bytes()
    tensor = mps_arena.allocate(1024).view(torch.float32)
    array = import_tensor(tensor)
    assert array is not None
    del tensor
    clear_import_cache()
    gc.collect()
    assert mps_arena.allocated_bytes() == before + 1024
    del array


def test_native_storage_can_be_released_on_a_worker_thread(native_mps):
    before = mps_arena.allocated_bytes()
    retained = [mps_arena.allocate(4096)]
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(retained.clear).result()
    assert mps_arena.allocated_bytes() == before


def test_encoded_torch_work_survives_last_user_reference(native_mps):
    for _ in range(8):
        tensor = mps_arena.allocate(1 << 20)
        tensor.fill_(17)
        result = tensor + 1
        del tensor  # Do not introduce a host fence before freeing the owner.
        assert result[::65536].cpu().tolist() == [18] * 16
        del result


def test_large_arena_allocation_churn_exceeds_the_original_failure_threshold(
    native_mps,
):
    """The heap-only reproducer failed around cycle 322 with a 1.2 GB arena."""
    size = 1_202_590_840
    if psutil.virtual_memory().total < 4 << 30:
        pytest.skip("the bounded large-arena regression requires at least 4 GiB RAM")
    before = mps_arena.allocated_bytes()
    for cycle in range(384):
        # One arena at a time, only its end sentinels touched. Never scale this
        # with free RAM or disable the system-pressure safety stop.
        assert psutil.virtual_memory().available >= 1536 << 20, "memory safety stop"
        arena = mu.ManualMemory(0, device="mps", num_bytes=size)
        arena.data[:8].fill_(7)
        arena.data[-8:].fill_(13)
        assert arena.data[:8].cpu().tolist() == [7] * 8, cycle
        assert arena.data[-8:].cpu().tolist() == [13] * 8, cycle
        del arena
        torch.mps.empty_cache()
        assert mps_arena.allocated_bytes() == before, cycle
