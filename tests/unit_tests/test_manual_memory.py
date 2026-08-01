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
