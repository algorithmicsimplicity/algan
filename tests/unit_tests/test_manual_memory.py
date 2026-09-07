import types

import pytest
import torch

from algan.utils.memory_utils import (
    InsufficientMemoryException,
    ManualMemory,
    get_num_available_bytes,
)


def _arena(num_bytes=128):
    return ManualMemory(
        0,
        device=torch.device("cpu"),
        managed=True,
        num_bytes=num_bytes,
    )


def test_manual_memory_accounts_for_dtype_alignment_and_bytes():
    memory = _arena()

    byte_values = memory.get_tensor((3,), torch.uint8)
    half_values = memory.get_tensor((2,), torch.float16)
    float_values = memory.get_tensor((2,), torch.float32)

    arena_storage = memory.data.untyped_storage()._cdata
    assert byte_values.untyped_storage()._cdata == arena_storage
    assert half_values.untyped_storage()._cdata == arena_storage
    assert float_values.untyped_storage()._cdata == arena_storage
    # uint8 [0:3], one alignment byte, float16 [4:8], float32 [8:16].
    assert memory.current_pointer == 16
    assert memory.max_pointer == 16


def test_temp_scope_restores_pointer_when_operation_raises():
    memory = _arena()
    memory.get_tensor((4,), torch.float32)
    before = memory.get_pointers()

    def allocate_then_fail():
        memory.get_tensor((8,), torch.float32)
        raise RuntimeError("failed")

    with memory.temp(), pytest.raises(RuntimeError, match="failed"):
        allocate_then_fail()

    assert memory.get_pointers() == before


def test_failed_allocation_does_not_advance_arena():
    memory = _arena(num_bytes=16)
    memory.get_tensor((3,), torch.float32)
    before = memory.get_pointers()

    with pytest.raises(InsufficientMemoryException):
        memory.get_tensor((2,), torch.float32)

    assert memory.get_pointers() == before


def test_reverse_allocation_charges_alignment_padding():
    memory = _arena(num_bytes=15)
    values = memory.get_tensor((2,), torch.float32, persist=True)

    assert values.untyped_storage()._cdata == memory.data.untyped_storage()._cdata
    # Reverse allocations align the end pointer down from 15 to 12, then use
    # eight payload bytes.
    assert memory.current_reverse_pointer == 4
    assert memory.max_pointer == 11


def test_cuda_available_bytes_clears_the_requested_device(monkeypatch):
    events = []

    class DeviceContext:
        def __init__(self, device):
            self.device = torch.device(device)

        def __enter__(self):
            events.append(("enter", self.device))

        def __exit__(self, *_args):
            events.append(("exit", self.device))

    monkeypatch.setattr(torch.cuda, "device", DeviceContext)
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: events.append(("empty", None))
    )

    def mem_get_info(device):
        events.append(("info", torch.device(device)))
        return 123, 456

    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)

    assert get_num_available_bytes(torch.device("cuda:2")) == 123
    assert events == [
        ("enter", torch.device("cuda:2")),
        ("empty", None),
        ("info", torch.device("cuda:2")),
        ("exit", torch.device("cuda:2")),
    ]


def test_memory_pressure_answers_per_device(monkeypatch):
    """A device without CUDA is not automatically "under pressure".

    ``release_torch_memory`` is called from twenty sites, nineteen with
    ``force_gc=False`` so a steady-state call is cheap. Answering ``True`` for
    every non-CUDA device made every one of those pay a full ``gc.collect()``
    on a Metal render -- and, before the gate below, an import-cache drop and a
    device drain with it. Metal reports the same two numbers CUDA does, so the
    same ratio decides; only a device with no telemetry keeps the conservative
    default.
    """
    from algan.utils import memory_utils as mu

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 8 << 30)

    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 1 << 30)
    assert mu._gpu_memory_pressure() is False, "an eighth of the pool is not pressure"

    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 7 << 30)
    assert mu._gpu_memory_pressure() is True

    # No telemetry at all keeps the conservative answer.
    monkeypatch.setattr(torch.mps, "is_available", lambda: False)
    assert mu._gpu_memory_pressure() is True


def test_an_unpressured_mps_reclaim_keeps_the_import_cache(monkeypatch):
    """The zero-copy import cache is not dropped by a steady-state call.

    Dropping it makes the next launch of every kernel re-import every arena
    array it takes; the widest take twenty. It is still dropped under pressure,
    which is what the leak it exists for actually is, and on any forced call.
    """
    from algan.rendering import mps_zero_copy
    from algan.utils import memory_utils as mu

    cleared = []
    monkeypatch.setattr(mps_zero_copy, "clear_import_cache", lambda: cleared.append(1))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None)
    monkeypatch.setattr(mu, "_gpu_memory_pressure", lambda *a, **k: False)

    mu.release_torch_memory(force_gc=False)
    assert cleared == [], "a steady-state reclaim dropped the import cache"

    mu.release_torch_memory(force_gc=True)
    assert cleared == [1], "a forced reclaim must still drop it"

    monkeypatch.setattr(mu, "_gpu_memory_pressure", lambda *a, **k: True)
    mu.release_torch_memory(force_gc=False)
    assert cleared == [1, 1], "a pressured reclaim must still drop it"


def test_the_mps_free_figure_drains_before_it_measures(monkeypatch):
    """MPS reclaims before measuring, exactly as the CUDA branch does.

    The import cache is dropped first because an entry there is a storage
    ``empty_cache`` cannot reclaim.

    What is measured afterwards is ``current_allocated_memory``, the live
    bytes, NOT ``driver_allocated_memory``: the driver figure behaves as a
    high-water mark on Metal and does not come back down after the drain, so
    sizing from it charges each render for the previous one's peak. On the Mac
    runner that cost the second UHD render a third of its arena (1898 -> 1226 MB,
    reproducibly) and 4x its launch count, and holding it to the first figure
    took the warm pass from 968-1523 s to 568.5 s.
    """
    from algan.rendering import mps_zero_copy
    from algan.utils import memory_utils as mu

    order = []
    monkeypatch.setattr(
        mps_zero_copy, "clear_import_cache", lambda: order.append("clear")
    )
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: order.append("drain"))
    monkeypatch.setattr(
        torch.mps,
        "current_allocated_memory",
        lambda: order.append("measure") or (1 << 30),
    )
    # The high-water figure is deliberately NOT what the arena is sized from.
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 4 << 30)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 5 << 30)

    # Pinned, or this test reads the machine it runs on: the figure is also
    # bounded by host free memory, so a busy CI box would change the answer.
    _plenty_of_host_memory(monkeypatch)

    free = mu.get_num_available_bytes(torch.device("mps"))

    assert order == ["clear", "drain", "measure"], order
    # A reserve is held back from the recommended max: live bytes alone left no
    # margin at all, and a warm UHD pass was killed outright at warm chunk 14
    # with the driver figure pinned at the ceiling.
    budget = (5 << 30) - int((5 << 30) * mu._MPS_HEADROOM)
    assert free == budget - (1 << 30), "sized from the high-water mark, not live bytes"
    assert free < (5 << 30) - (1 << 30), "no headroom was reserved"


def _plenty_of_host_memory(monkeypatch, available=64 << 30):
    """Make the host-memory bound non-binding, so a test measures what it means."""
    from algan.utils import memory_utils as mu

    monkeypatch.setattr(
        mu.psutil, "virtual_memory", lambda: types.SimpleNamespace(available=available)
    )


def test_the_mps_figure_is_bounded_by_host_memory(monkeypatch):
    """Unified memory: the arena cannot exceed what the MACHINE has free.

    ``recommendedMaxWorkingSetSize`` describes what the GPU should hold, and on
    Apple silicon the CPU side, the video encoder and the OS draw on the very
    same pool. Sizing from the GPU figure alone hands out memory that is not
    there: on the 7 GB Mac runner the render peaked at 4.87 G of driver
    allocation beside 1.45 G of process RSS, and the jobs that then died did so
    at 3.14 G and 3.56 G -- under the recommendation, where a GPU working-set
    story cannot reach.
    """
    from algan.utils import memory_utils as mu

    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 0)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 64 << 30)
    monkeypatch.setattr(
        "algan.rendering.mps_zero_copy.clear_import_cache", lambda: None
    )

    # A machine with room to spare: the GPU figure decides.
    _plenty_of_host_memory(monkeypatch)
    roomy = mu.get_num_available_bytes(torch.device("mps"))

    # The same GPU figure on a machine that is nearly full: the host decides.
    _plenty_of_host_memory(monkeypatch, available=mu._HOST_RESERVE_BYTES + (1 << 30))
    cramped = mu.get_num_available_bytes(torch.device("mps"))

    assert cramped == 1 << 30, "the host bound did not apply"
    assert cramped < roomy

    # And a machine with less free than the reserve hands out nothing rather
    # than going negative.
    _plenty_of_host_memory(monkeypatch, available=0)
    assert mu.get_num_available_bytes(torch.device("mps")) == 0
