"""Byte-region alias, layout and lifetime contracts."""
from __future__ import annotations

import pytest
import torch

from algan.rendering.arena_region_args import pack_regions
from algan.rendering.arena_regions import (
    ArenaRegionError, RegionLayout, TypedArenaView, lease_regions, validate_regions,
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
    write = TypedArenaView.bind(a.view(torch.int32)[3:6], name="integer_write", access="write")
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
    assert layout.byte_length == 0 and layout.alignment >= 4
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
    assert buf.data_ptr() == a.data_ptr() and buf.numel() == 9
    assert offsets.tolist() == [0, 6] and shapes.tolist() == [2, 3, 3]
    wide = torch.empty((0, (1 << 24) + 3), dtype=torch.int32)
    _, off, shp = pack_regions((("wide", "i32", 2),), (wide,))
    assert off.tolist() == [0] and shp.tolist() == [0, (1 << 24) + 3]
