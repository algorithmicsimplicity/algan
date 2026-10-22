"""The render's per-chunk and per-tile host work does not synchronize the
device where it needs no value back (``arena_args_taichi.defer_syncs``).

Every change behind the switch is a scheduling change: the values a kernel
receives are the ones the blocking path produced. These tests hold both
halves -- the same values with the switch on and off, and (on CUDA) no
synchronization on the switched paths, checked with torch's sync debug mode
set to raise.
"""

import contextlib

import pytest
import torch

from algan.rendering.raytracing import arena_args_taichi as aat
from algan.rendering.raytracing import raster_pipeline, tracer

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@contextlib.contextmanager
def _defer(on):
    saved = aat.defer_syncs
    aat.defer_syncs = on
    try:
        yield
    finally:
        aat.defer_syncs = saved


def test_the_switch_follows_opt_disable_after_its_first_read(monkeypatch):
    """A/B scripts swap the timeline's ``_OPT_DISABLED`` between renders, and a
    warm daemon adopts each client's ``ALGAN_OPT_DISABLE``; neither may be
    pinned by whichever value the first render read.
    """
    import algan.animation_timeline.timeline as tl

    with _defer(None):
        monkeypatch.setattr(tl, "_OPT_DISABLED", frozenset())
        assert aat.defer_syncs_enabled()
        monkeypatch.setattr(tl, "_OPT_DISABLED", frozenset({"tilesync"}))
        assert not aat.defer_syncs_enabled()
        monkeypatch.setattr(tl, "_OPT_DISABLED", frozenset())
        assert aat.defer_syncs_enabled()
    # An explicit assignment still overrides the environment.
    with _defer(False):
        monkeypatch.setattr(tl, "_OPT_DISABLED", frozenset())
        assert not aat.defer_syncs_enabled()


def test_large_uploads_are_not_staged_through_pinned_memory():
    assert tracer._PINNED_STAGING_MAX_BYTES == 1 << 20


@contextlib.contextmanager
def _no_sync_allowed():
    """Raise on any synchronizing CUDA call inside the block."""
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


class _Memory:
    """The one ``ManualMemory`` method ``_arena_copy`` uses."""

    def __init__(self, device):
        self.device = device

    def get_tensor(self, shape, dtype):
        return torch.empty(tuple(shape), dtype=dtype, device=self.device)


# --- sheet slice bounds -------------------------------------------------------


def _coverage(num_covered, device="cpu", seed=0):
    gen = torch.Generator().manual_seed(seed)
    counts = torch.randint(0, 5, (num_covered,), generator=gen)
    offsets = torch.zeros(num_covered + 1, dtype=torch.int32)
    offsets[1:] = torch.cumsum(counts, 0)
    return {"sheet_offsets": offsets.to(device)}, offsets.tolist()


@pytest.mark.parametrize("on", [True, False])
def test_sheet_slice_bounds_are_the_csr_offsets_at_every_tile(on):
    """Every slice reads exactly its two CSR entries.

    The slices are uniform tiles, a shrunken retry, tiles of the new size and
    the clamped last one.
    """
    n = 1000
    coverage, ref = _coverage(n)
    slices = [(0, 128), (128, 256), (256, 300), (256, 290), (290, 324)]
    c = 324
    while c < n:
        slices.append((c, min(n, c + 34)))
        c = min(n, c + 34)
    with _defer(on):
        for c0, c1 in slices:
            assert raster_pipeline._sheet_slice_bounds(coverage, c0, c1) == (
                ref[c0],
                ref[c1],
            )


def test_sheet_slice_bounds_read_a_tiling_once():
    """The first slice's read covers the whole regular tiling of the chunk."""
    coverage, ref = _coverage(500, seed=3)
    with _defer(True):
        raster_pipeline._sheet_slice_bounds(coverage, 0, 100)
        known = dict(coverage["sheet_offsets_at"])
        for c0 in range(100, 500, 100):
            raster_pipeline._sheet_slice_bounds(coverage, c0, min(500, c0 + 100))
    assert coverage["sheet_offsets_at"] == known
    assert known == {c: ref[c] for c in (0, 100, 200, 300, 400, 500)}


@cuda
def test_sheet_slice_bounds_read_one_small_transfer_on_cuda():
    coverage, ref = _coverage(4096, device="cuda", seed=5)
    with _defer(True):
        assert raster_pipeline._sheet_slice_bounds(coverage, 0, 1024) == (
            ref[0],
            ref[1024],
        )
        # Cached: the next tiles of the same size cost nothing at all.
        with _no_sync_allowed():
            for c0 in range(1024, 4096, 1024):
                assert raster_pipeline._sheet_slice_bounds(coverage, c0, c0 + 1024) == (
                    ref[c0],
                    ref[c0 + 1024],
                )
    assert "sheet_offsets_host" not in coverage


# --- glossy frame bounds ------------------------------------------------------


@pytest.mark.parametrize("frames", [1, 3])
def test_gloss_frame_bounds_match_the_searchsorted_reference(frames):
    ppf = 64
    gen = torch.Generator().manual_seed(frames)
    covered = torch.unique(torch.randint(0, frames * ppf, (90,), generator=gen))
    with _defer(False):
        reference = tracer._gloss_frame_bounds(covered, ppf, frames)
    with _defer(True):
        assert tracer._gloss_frame_bounds(covered, ppf, frames) == reference
    assert reference[0] == 0
    assert reference[-1] == covered.numel()


@cuda
def test_one_frame_gloss_bounds_need_no_readback():
    covered = torch.arange(10, 4000, 7, device="cuda")
    with _defer(True), _no_sync_allowed():
        bounds = tracer._gloss_frame_bounds(covered, 4096, 1)
    assert bounds == [0, covered.numel()]


# --- uploads ------------------------------------------------------------------


@cuda
@pytest.mark.parametrize(
    ("source", "dtype"),
    [
        (
            torch.linspace(-3.0, 7.0, 12, dtype=torch.float64).reshape(3, 4),
            torch.float32,
        ),
        (torch.arange(20, dtype=torch.int64).reshape(4, 5).t(), torch.int32),
        (torch.rand(5, 3), None),
    ],
)
def test_staged_arena_copy_lands_the_blocking_copy_bytes(source, dtype):
    memory = _Memory("cuda")
    with _defer(False):
        reference = tracer._arena_copy(memory, source, dtype).cpu()
    source = source.clone()
    with _defer(True):
        with _no_sync_allowed():
            staged = tracer._arena_copy(memory, source, dtype)
        # The caller may reuse its tensor at once: the staged bytes are a copy.
        source.zero_()
        torch.cuda.synchronize()
    assert staged.dtype == reference.dtype
    assert staged.shape == reference.shape
    assert torch.equal(staged.cpu(), reference)


@cuda
def test_pack_uploads_a_missed_table_without_synchronizing():
    arena = torch.empty(1 << 16, dtype=torch.uint8, device="cuda")
    a = arena[0:24].view(torch.float32).view(2, 3)
    b = arena[64 : 64 + 32].view(torch.int32).view(8)
    spec = (("a", "f32", 2), ("b", "i32", 1))
    aat.clear_pack_cache()
    with _defer(False):
        _f, _i, aoff_ref, ashp_ref = aat.pack(spec, (a, b))
    aat.clear_pack_cache()
    try:
        with _defer(True), _no_sync_allowed():
            _f, _i, aoff, ashp = aat.pack(spec, (a, b))
        assert aoff.is_cuda
        assert aoff.tolist() == aoff_ref.tolist() == [0, 16]
        assert ashp.tolist() == ashp_ref.tolist() == [2, 3, 8]
    finally:
        aat.clear_pack_cache()


# --- compaction counts carrying the tile's allocator words --------------------


def test_compaction_count_carries_the_extra_words():
    """``select(also_read=...)`` returns the same slots and reads the words."""
    from algan.rendering.taichi_runtime import init_taichi
    from algan.settings._startup import render_device
    from algan.utils.memory_utils import ManualMemory

    init_taichi()
    device = render_device()
    memory = ManualMemory(0, device=device, num_bytes=4096)
    memory.data.zero_()
    rs_int = memory.get_tensor((6, 5), torch.int32)
    rs_int.zero_()
    rs_int[:, 2].copy_(torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.int32))
    words = memory.get_tensor((3,), torch.int32)
    words.copy_(torch.tensor([5, 0, 2], dtype=torch.int32))

    plain = tracer._ArenaRayCompactor(memory, 6)
    reference = sorted(plain.select(rs_int, 0, scan_pool=True).tolist())
    assert plain.also_values is None

    carried = tracer._ArenaRayCompactor(memory, 6)
    got = carried.select(rs_int, 0, scan_pool=True, also_read=words)
    assert sorted(got.tolist()) == reference == [0, 2, 5]
    assert carried.also_values == [5, 0, 2]
