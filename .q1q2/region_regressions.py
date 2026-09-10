

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


@pytest.mark.parametrize("dtype,tag", [(torch.uint8, "u8"), (torch.float32, "f32")])
def test_persistent_regions_beyond_int32_storage_offsets(large_offset_arena, dtype, tag):
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
    assert offsets.tolist() == [0] and shapes.tolist() == [6]
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
    matrix = backing[:1 << 32].reshape(1 << 16, 1 << 16)
    with pytest.raises(ArenaRegionError, match="Rebased region end"):
        TypedArenaView.bind(matrix, name="too_many_elements")


@pytest.mark.parametrize("last_offset", [(1 << 31) - 4, (1 << 31) + 4])
def test_pack_rejects_rebased_end_and_offset_overflow(large_offset_arena, last_offset):
    memory = large_offset_arena
    backing = memory.get_tensor((memory.length,), torch.uint8)
    first, last = backing[:1], backing[last_offset:last_offset + 4]
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
