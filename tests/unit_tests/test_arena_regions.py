"""Byte-region alias, layout and lifetime contracts."""

from __future__ import annotations

import pytest
import torch

from algan.rendering.arena_region_args import pack_regions
from algan.rendering.arena_regions import (
    ArenaRegionError,
    RegionLayout,
    TypedArenaView,
    lease_regions,
    validate_regions,
)
from algan.settings import SETTINGS
from algan.utils.memory_utils import ManualMemory


@pytest.fixture
def arena():
    saved = SETTINGS.snapshot()
    SETTINGS.raytracing.experimental.device_dispatch = True
    memory = ManualMemory(0, device=torch.device("cpu"), num_bytes=65536)
    try:
        yield memory
    finally:
        SETTINGS.restore(saved)


def test_runtime_experimental_setting():
    saved = SETTINGS.snapshot()
    try:
        SETTINGS.raytracing.experimental.device_dispatch = True
        assert SETTINGS.raytracing.device_dispatch is True
        with pytest.raises((AttributeError, ValueError), match="experimental"):
            SETTINGS.raytracing.device_dispatch = False
        SETTINGS.raytracing.experimental.device_dispatch = False
        assert SETTINGS.raytracing.device_dispatch is False
    finally:
        SETTINGS.restore(saved)


def test_cross_dtype_writable_alias(arena):
    a = arena.get_tensor((12,), torch.float32)
    read = TypedArenaView.bind(a, name="float_read")
    write = TypedArenaView.bind(
        a.view(torch.int32)[3:6], name="integer_write", access="write"
    )
    with pytest.raises(ArenaRegionError, match="Writable alias"):
        validate_regions((read, write))
    alias = TypedArenaView.bind(a.view(torch.int32)[3:6], name="alias")
    metadata = validate_regions((read, alias))
    assert all(m.readonly and m.immutable_layout for m in metadata)
    assert "alias" not in metadata[0].disjoint


def test_alignment_empty_offsets_and_layout(arena):
    arena.get_tensor((7,), torch.uint8)
    a = arena.get_tensor((8, 3), torch.float32)
    layout = RegionLayout.from_tensor(a[4:4])
    assert layout.byte_offset == a.storage_offset() * 4 + 48
    assert layout.byte_length == 0
    assert layout.alignment >= 4
    with pytest.raises(ArenaRegionError, match="contiguous"):
        TypedArenaView.bind(a.T, name="strided")
    with pytest.raises(ArenaRegionError, match="vector"):
        TypedArenaView.bind(a, name="wrong_vector", vector_width=4)


@pytest.mark.parametrize("persistent", [False, True])
def test_rewind_identity_survives_address_reuse(arena, persistent):
    pointers = arena.get_pointers()
    a = arena.get_tensor((8,), torch.int32, persist=persistent)
    old = TypedArenaView.bind(a[1:], name="old")
    arena.set_pointers(pointers)
    replacement = arena.get_tensor((8,), torch.int32, persist=persistent)
    assert replacement.data_ptr() == a.data_ptr()
    with pytest.raises(ArenaRegionError, match="reclaimed"):
        old.validate()
    TypedArenaView.bind(replacement, name="new").validate()


def test_live_lease_rejects_rewind_atomically(arena):
    start = arena.get_pointers()
    a = arena.get_tensor((8,), torch.int32)
    b = arena.get_tensor((8,), torch.float32, persist=True)
    views = (TypedArenaView.bind(a, name="a"), TypedArenaView.bind(b, name="b"))
    current = arena.get_pointers()
    with lease_regions(views):
        with pytest.raises(ArenaRegionError, match="unfinished"):
            arena.set_pointers(start)
        assert arena.get_pointers() == current
        for view in views:
            view.validate()
    arena.set_pointers(start)


def test_persistent_survives_temporary_rewind(arena):
    p = arena.get_tensor((3,), torch.float32, persist=True)
    view = TypedArenaView.bind(p, name="persistent")
    with arena.temp():
        a = arena.get_tensor((5,), torch.int32)
        transient = TypedArenaView.bind(a, name="temp")
    view.validate()
    with pytest.raises(ArenaRegionError):
        transient.validate()
    arena.reset()
    with pytest.raises(ArenaRegionError):
        view.validate()


def test_metadata_mutation_rejected(arena):
    a = arena.get_tensor((8,), torch.int32)
    view = TypedArenaView.bind(a, name="shape")
    a.unsqueeze_(0)
    with pytest.raises(ArenaRegionError, match="metadata changed"):
        view.validate()


def test_packing_narrows_bases_and_preserves_wide_integer_shapes(arena):
    arena.get_tensor((17,), torch.uint8)
    a = arena.get_tensor((2, 3), torch.float32)
    b = arena.get_tensor((3,), torch.float32)
    buf, offsets, shapes = pack_regions((("a", "f32", 2), ("b", "f32", 1)), (a, b))
    assert buf.data_ptr() == a.data_ptr()
    assert buf.numel() == 9
    assert offsets.tolist() == [0, 6]
    assert shapes.tolist() == [2, 3, 3]
    wide = torch.empty((0, (1 << 24) + 3), dtype=torch.int32)
    _, off, shp = pack_regions((("wide", "i32", 2),), (wide,))
    assert off.tolist() == [0]
    assert shp.tolist() == [0, (1 << 24) + 3]


@pytest.fixture
def large_offset_arena(arena):
    # A sparse file reserves address space, not an 8 GiB resident tensor.
    # Only small endpoint slices are ever written. Do not replace this with
    # torch.empty/fill_ or an anonymous mapping charged to process commit.
    import mmap
    import sys
    import tempfile

    if sys.platform != "linux":
        pytest.skip("Large-offset regression uses Linux sparse file mappings")
    size = (1 << 33) + 4096
    with tempfile.TemporaryFile() as backing:
        backing.truncate(size)
        mapping = mmap.mmap(backing.fileno(), size)
    arena._poison = -1  # Never fault in the entire sparse backing allocation.
    arena.data = torch.frombuffer(mapping, dtype=torch.uint8)
    arena.length = arena.data.numel()
    arena.current_reverse_pointer = arena.length
    # Torch owns a reference to the mapping through its storage; let it close
    # only after the arena and every test view have released that storage.
    return arena


@pytest.mark.parametrize(
    ("dtype", "tag"), [(torch.uint8, "u8"), (torch.float32, "f32")]
)
def test_persistent_regions_beyond_int32_storage_offsets(
    large_offset_arena, dtype, tag
):
    memory = large_offset_arena
    pointers = memory.get_pointers()
    values = memory.get_tensor((6,), dtype, persist=True)
    values.fill_(3)
    assert values.storage_offset() > (1 << 31) - 1
    view = TypedArenaView.bind(values, name="high")
    view.validate()
    buf, offsets, shapes = pack_regions((("high", tag, 1),), (values,))
    assert buf.data_ptr() == values.data_ptr()
    assert buf.numel() == 6
    assert offsets.tolist() == [0]
    assert shapes.tolist() == [6]
    assert torch.equal(buf, values)
    memory.set_pointers(pointers)
    with pytest.raises(ArenaRegionError, match="reclaimed"):
        view.validate()


def test_large_host_allocation_is_not_an_int32_kernel_argument(large_offset_arena):
    memory = large_offset_arena
    # A backing byte allocation is legal for lifetime tracking even if the
    # whole allocation is too large to bind as one int32-indexed ndarray.
    backing = memory.get_tensor((memory.length,), torch.uint8)
    layout = RegionLayout.from_tensor(backing)
    assert layout.byte_length == memory.length
    with pytest.raises(ArenaRegionError, match="Region shape"):
        TypedArenaView.bind(backing, name="too_large")
    TypedArenaView.bind(backing[-8:], name="small_tail").validate()
    matrix = backing[: 1 << 32].reshape(1 << 16, 1 << 16)
    with pytest.raises(ArenaRegionError, match="Rebased region end"):
        TypedArenaView.bind(matrix, name="too_many_elements")


@pytest.mark.parametrize("last_offset", [(1 << 31) - 4, (1 << 31) + 4])
def test_pack_rejects_rebased_end_and_offset_overflow(large_offset_arena, last_offset):
    memory = large_offset_arena
    backing = memory.get_tensor((memory.length,), torch.uint8)
    first, last = backing[:1], backing[last_offset : last_offset + 4]
    # Both buffers fit individually. The first case's last offset also fits,
    # but its exclusive end does not: checking only offsets is insufficient.
    TypedArenaView.bind(first, name="first").validate()
    TypedArenaView.bind(last, name="last").validate()
    with pytest.raises(ArenaRegionError, match="Rebased region end"):
        pack_regions((("first", "u8", 1), ("last", "u8", 1)), (first, last))


def test_empty_host_layout_still_checks_kernel_shapes_and_strides():
    wide_shape = torch.empty((0, 1 << 31), dtype=torch.uint8)
    assert RegionLayout.from_tensor(wide_shape).byte_length == 0
    with pytest.raises(ArenaRegionError, match="Region shape"):
        TypedArenaView.bind(wide_shape, name="wide_shape")
    wide_stride = torch.empty((0, 1 << 16, 1 << 16), dtype=torch.uint8)
    assert RegionLayout.from_tensor(wide_stride).byte_length == 0
    with pytest.raises(ArenaRegionError, match="Region stride"):
        TypedArenaView.bind(wide_stride, name="wide_stride")
