"""The arena-offset calling convention, on tensors built by hand.

``test_arena_binding_live.py`` is the other half: this file pins the mechanics,
that one pins the claim about the real renderer.
"""

import pytest
import torch

from algan.rendering.arena_binding import (
    METAL_BUFFER_SLOTS,
    ArenaBindingError,
    arena_slot,
    arena_storage_ptr,
    offset_table,
    plan_bindings,
    unpack,
)
from algan.utils.memory_utils import ManualMemory


@pytest.fixture
def arena():
    return ManualMemory(0.0, device="cpu", num_bytes=1 << 16)


def test_allocation_is_addressable_as_an_offset(arena):
    x = arena.get_tensor((8, 3), torch.float32)
    slot = arena_slot(x, arena)
    assert slot is not None
    assert slot.shape == (8, 3)
    assert slot.numel == 24
    assert slot.nbytes == 96
    assert slot.byte_offset % slot.itemsize == 0
    assert slot.elem_offset * slot.itemsize == slot.byte_offset


def test_every_allocation_shares_one_storage(arena):
    """The premise §1.2 rests on: many arrays, one buffer, many offsets."""
    tensors = [
        arena.get_tensor((4,), torch.float32),
        arena.get_tensor((7, 2), torch.float32),
        arena.get_tensor((5,), torch.int32),
        arena.get_tensor((3,), torch.int64),
    ]
    ptrs = {t.untyped_storage().data_ptr() for t in tensors}
    assert ptrs == {arena_storage_ptr(arena)}
    offsets = [arena_slot(t, arena).byte_offset for t in tensors]
    assert offsets == sorted(offsets), "bump allocator hands out rising offsets"
    assert len(set(offsets)) == len(offsets)


def test_round_trip_through_offset_only(arena):
    """Rebuild from (arena, offset, dtype, shape) and get the values back.

    This is what a shader does with a base pointer and an offset, so a
    mismatch here is a mismatch there.
    """
    x = arena.get_tensor((6, 2), torch.float32)
    x.copy_(torch.arange(12, dtype=torch.float32).reshape(6, 2))
    slot = arena_slot(x, arena)
    assert torch.equal(unpack(arena, slot), x)


def test_round_trip_preserves_dtype_families(arena):
    for dtype, value in (
        (torch.float32, 1.5),
        (torch.int32, -7),
        (torch.int64, 1 << 40),
        (torch.uint8, 250),
    ):
        x = arena.get_tensor((4,), dtype)
        x.fill_(value)
        assert torch.equal(unpack(arena, arena_slot(x, arena)), x)


def test_a_tensor_outside_the_arena_is_not_an_error(arena):
    """A texture or a persistent table keeps its own binding; that is normal."""
    outside = torch.zeros(4, dtype=torch.float32)
    assert arena_slot(outside, arena) is None


def test_non_contiguous_view_is_refused(arena):
    """A shader has no stride vector, so this would read the wrong elements."""
    x = arena.get_tensor((8, 4), torch.float32)
    with pytest.raises(ArenaBindingError, match="non-contiguous"):
        arena_slot(x[:, ::2], arena)


def test_a_misaligned_arena_view_cannot_be_built_in_the_first_place(arena):
    """Why the offset convention is safe: torch will not make the bad view.

    ``(device const float*)(arena + off)`` is undefined in MSL for an ``off``
    that is not a multiple of 4, so a misaligned slot would be a wrong-pixels
    bug rather than a crash. It cannot arise: torch refuses the reinterpretation
    that would produce one, independently of the arena's own alignment logic.
    ``arena_slot`` keeps a check for it anyway, for a backend that one day
    builds tensors without going through ``view``.
    """
    raw = arena.get_tensor((32,), torch.uint8)
    with pytest.raises(RuntimeError, match="divisible by 4"):
        raw[1:9].view(torch.float32)


def test_plan_counts_bindings_not_arguments(arena):
    """49 arena arrays are 2 bindings; what is outside keeps its slot."""
    args = [arena.get_tensor((4,), torch.float32) for _ in range(49)]
    args.append(torch.zeros(4))  # a texture, say
    args.append(17)  # a scalar: setBytes, not a buffer
    plan = plan_bindings(args, arena)
    assert len(plan.slots) == 49
    assert plan.passthrough == (49,)
    assert plan.non_tensor == (50,)
    assert plan.bindings == 3
    assert plan.fits


def test_plan_with_nothing_packed_needs_no_arena_binding(arena):
    plan = plan_bindings([torch.zeros(2), torch.zeros(2)], arena)
    assert plan.slots == ()
    assert plan.bindings == 2


def test_plan_reports_when_passthrough_alone_overflows(arena):
    """The failure the convention cannot fix: too many non-arena arrays."""
    args = [torch.zeros(2) for _ in range(METAL_BUFFER_SLOTS + 1)]
    plan = plan_bindings(args, arena)
    assert not plan.fits
    assert plan.bindings == METAL_BUFFER_SLOTS + 1


def test_offset_table_is_in_argument_order(arena):
    a = arena.get_tensor((4,), torch.float32)
    b = arena.get_tensor((9,), torch.float32)
    plan = plan_bindings([a, 3, b], arena)
    table = offset_table(plan)
    assert table.tolist() == [
        arena_slot(a, arena).byte_offset,
        arena_slot(b, arena).byte_offset,
    ]
    assert table.dtype == torch.int32


def test_offset_table_can_emit_element_offsets(arena):
    x = arena.get_tensor((4,), torch.float32)
    plan = plan_bindings([x], arena)
    assert offset_table(plan, elements=True).tolist() == [
        arena_slot(x, arena).byte_offset // 4
    ]


def test_persistent_allocations_are_the_same_storage(arena):
    """``persist=True`` allocates from the far end, still inside the arena."""
    near = arena.get_tensor((4,), torch.float32)
    far = arena.get_tensor((4,), torch.float32, persist=True)
    assert arena_slot(near, arena) is not None
    far_slot = arena_slot(far, arena)
    assert far_slot is not None
    assert far_slot.byte_offset > arena_slot(near, arena).byte_offset
