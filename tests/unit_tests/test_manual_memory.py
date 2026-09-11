import types

import psutil
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


def test_memory_pressure_answers_for_the_render_device(monkeypatch):
    """GPU pressure follows the configured render device, not installed GPUs."""
    from algan.utils import memory_utils as mu

    monkeypatch.setattr(mu, "render_device", lambda: torch.device("mps"))
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 8 << 30)

    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 1 << 30)
    assert mu._gpu_memory_pressure() is False, "an eighth of the pool is not pressure"

    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 7 << 30)
    assert mu._gpu_memory_pressure() is True

    # A CPU render has no GPU pressure even if CUDA is installed. Host pressure
    # is checked separately by release_torch_memory.
    monkeypatch.setattr(mu, "render_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda *_args: (_ for _ in ()).throw(AssertionError("queried unused CUDA")),
    )
    assert mu._gpu_memory_pressure() is False


def test_an_unpressured_cpu_reclaim_skips_gc(monkeypatch):
    """Steady-state CPU cleanup does not collect without host pressure."""
    from algan.utils import memory_utils as mu

    events = []
    monkeypatch.setattr(mu, "render_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(mu, "_host_memory_pressure", lambda: False)
    monkeypatch.setattr(mu.gc, "collect", lambda: events.append("gc"))

    mu.release_torch_memory(force_gc=False)
    assert events == []


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
    _machine_with_ram(monkeypatch)

    free = mu.get_num_available_bytes(torch.device("mps"))

    assert order == ["clear", "drain", "measure"], order
    # A reserve is held back from the recommended max: live bytes alone left no
    # margin at all, and a warm UHD pass was killed outright at warm chunk 14
    # with the driver figure pinned at the ceiling.
    budget = (5 << 30) - int((5 << 30) * mu._MPS_HEADROOM)
    assert free == budget - (1 << 30), "sized from the high-water mark, not live bytes"
    assert free < (5 << 30) - (1 << 30), "no headroom was reserved"


def _machine_with_ram(monkeypatch, total=1024 << 30):
    """Pin the machine's total RAM, which the MPS free figure is capped against.

    Default is large enough that the cap never binds, so a test that is about
    something else measures that something else rather than the box it runs on.
    """
    from algan.utils import memory_utils as mu

    monkeypatch.setattr(
        mu.psutil, "virtual_memory", lambda: types.SimpleNamespace(total=total)
    )


def test_the_mps_figure_is_capped_by_a_share_of_total_ram(monkeypatch):
    """Unified memory: the arena is capped against the machine's TOTAL RAM.

    The GPU pool is carved out of the same RAM as everything else, so sizing
    from Metal's advice alone can leave the machine with nothing -- which on the
    7 GB runner showed up as a kill, three wedges (two below the GPU
    recommendation) and one 455 s chunk that thrashed and recovered.

    Total RAM is used rather than ``psutil``'s ``available`` because two
    attempts at the latter died with "Insufficient memory to ray trace a single
    frame": it excludes the GPU pool and moves constantly, ranging 3.87 to
    6.19 G when summed with ``driver_allocated`` across a single render. Total
    RAM does not move.
    """
    from algan.utils import memory_utils as mu

    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 0)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 64 << 30)
    monkeypatch.setattr(
        "algan.rendering.mps_zero_copy.clear_import_cache", lambda: None
    )
    monkeypatch.setenv("ALGAN_MPS_HOST_SHARE", "0.4")

    # A small machine: the share of total RAM decides, and the GPU's generous
    # recommendation does not get to.
    _machine_with_ram(monkeypatch, total=8 << 30)
    assert mu.get_num_available_bytes(torch.device("mps")) == int((8 << 30) * 0.4)

    # A large one: the GPU-side figure binds first and the cap never applies.
    _machine_with_ram(monkeypatch, total=1024 << 30)
    budget = (64 << 30) - int((64 << 30) * mu._MPS_HEADROOM)
    assert mu.get_num_available_bytes(torch.device("mps")) == budget

    # And 0 disables the cap outright.
    monkeypatch.setenv("ALGAN_MPS_HOST_SHARE", "0")
    _machine_with_ram(monkeypatch, total=8 << 30)
    assert mu.get_num_available_bytes(torch.device("mps")) == budget


def test_a_metal_arena_stays_inside_what_one_mpsndarray_can_address():
    """A Metal arena is clamped to ``INT_MAX`` bytes, whatever the machine has.

    Not a memory budget. The arena is one ``uint8`` tensor and every allocation
    is a view of it, so torch's MPS backend describes the whole buffer as a flat
    ``MPSNDArray`` of ``storage_bytes / element_size`` -- which for the ``uint8``
    and ``bool`` views a render takes is the arena's byte count. Past ``INT_MAX``
    Metal refuses that descriptor with ``abort()``, so the process dies with
    SIGABRT inside whichever ordinary op touched such a view first (``flip`` in
    ``_frames_to_host``, ``fill_`` on the opaque mask, ``amax`` in the bloom
    filter) -- a crash with nothing about memory in it.

    Measured on the hosted runner, since no Mac here can be asked: writing
    ``t[4096:8192]`` of a ``2**31 - 4096`` byte buffer returns 0, and the same
    four kilobytes into a ``2**31 + 4096`` byte buffer returns -6 with that
    assertion. `DESIGN_mps_support.md` §4.7 has the run.

    A 16 GB Mac reaches it: ``_MPS_HOST_SHARE`` of 16 GB is 6.4 G free and
    ``rendering_memory_fraction`` of that is 2.56 G, over the ceiling by 19%.
    """
    from algan.utils import memory_utils as mu

    ceiling = mu._MPS_MAX_ARENA_BYTES
    assert ceiling <= (1 << 31) - 4096, "must stay inside the measured-safe range"
    over = int((16 << 30) * mu._MPS_HOST_SHARE * 0.4)
    assert over > ceiling, "the 16 GB machine this is about must exceed it"
    assert mu._addressable_arena_bytes(torch.device("mps"), over) == ceiling

    # Under it nothing moves, and the clamp is Metal's alone: a CUDA or CPU
    # arena of the same size is one buffer indexed with 64-bit offsets.
    assert mu._addressable_arena_bytes(torch.device("mps"), 1 << 20) == 1 << 20
    assert mu._addressable_arena_bytes(torch.device("cuda"), over) == over
    assert mu._addressable_arena_bytes(torch.device("cpu"), over) == over


def test_a_cpu_render_is_sized_against_the_machine_not_a_flat_2gb(monkeypatch):
    """The CPU branch scales with the box, as the CUDA and MPS branches do.

    It used to return a flat 2 GB, and the render arena is
    ``rendering_memory_fraction`` (0.4) of it, so **every** CPU machine got a
    0.75 GB arena -- too small for one 4K frame. A UHD CPU render therefore died
    with "Insufficient memory to ray trace a single frame" on a 7 GB runner and
    would have on a 512 GB workstation too. That is the same defect the MPS
    branch carried as a 1 GiB clamp, and it was found the same way: by rendering
    the scene and reading the error.
    """
    from algan.settings import computing_settings as cs

    monkeypatch.setattr(
        cs.psutil if hasattr(cs, "psutil") else psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(total=64 << 30),
        raising=False,
    )
    assert cs._default_cpu_memory() == int((64 << 30) * cs._CPU_MEMORY_SHARE)

    # A frame-sized arena on a small machine, and never below the old floor.
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: types.SimpleNamespace(total=1 << 30)
    )
    assert cs._default_cpu_memory() == 2 * cs.GIGABYTES, "dropped below the old floor"

    # And a platform that cannot report its memory keeps the old constant.
    def _no_telemetry():
        raise OSError("no meminfo here")

    monkeypatch.setattr(psutil, "virtual_memory", _no_telemetry)
    assert cs._default_cpu_memory() == 2 * cs.GIGABYTES


def test_malloc_trim_is_linux_only(monkeypatch):
    from algan.utils import memory_utils as mu

    calls = []
    monkeypatch.setattr(mu.sys, "platform", "darwin")
    monkeypatch.setattr(mu.ctypes, "CDLL", lambda *_args, **_kwargs: calls.append(1))

    assert mu._malloc_trim() is False
    assert calls == [], "malloc_trim must never be looked up off Linux"


def test_host_memory_pressure_honors_a_finite_cgroup_before_host_ram(monkeypatch):
    from algan.utils import memory_utils as mu

    # Host RAM looks plentiful, but the process group is over the deliberately
    # earlier hard-cgroup threshold. This is the shape of the 4 GiB container
    # OOM that motivated the native cleanup path.
    monkeypatch.setattr(mu, "_linux_cgroup_memory_usage", lambda: (3 << 30, 4 << 30))
    monkeypatch.setattr(
        mu.psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(total=64 << 30, available=60 << 30),
    )
    assert mu._host_memory_pressure() is True

    monkeypatch.setattr(mu, "_linux_cgroup_memory_usage", lambda: (1 << 30, 4 << 30))
    assert mu._host_memory_pressure() is False


def test_pressure_cleanup_trims_then_resets_only_if_pressure_remains(monkeypatch):
    from algan.utils import memory_utils as mu

    events = []
    readings = iter((True, True))
    monkeypatch.setattr(mu, "_host_memory_pressure", lambda: next(readings))
    monkeypatch.setattr(mu, "_gpu_memory_pressure", lambda: False)
    monkeypatch.setattr(mu.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(mu, "_malloc_trim", lambda: events.append("trim") or True)
    monkeypatch.setattr(
        mu,
        "_reset_quadrants_runtime_for_memory_pressure",
        lambda: events.append("reset") or True,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.mps, "is_available", lambda: False)

    mu.release_torch_memory(force_gc=False)
    assert events == ["gc", "trim", "reset", "trim"]


def test_pressure_cleanup_keeps_quadrants_if_trim_relaxes_pressure(monkeypatch):
    from algan.utils import memory_utils as mu

    events = []
    readings = iter((True, False))
    monkeypatch.setattr(mu, "_host_memory_pressure", lambda: next(readings))
    monkeypatch.setattr(mu, "_gpu_memory_pressure", lambda: False)
    monkeypatch.setattr(mu.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(mu, "_malloc_trim", lambda: events.append("trim") or True)
    monkeypatch.setattr(
        mu,
        "_reset_quadrants_runtime_for_memory_pressure",
        lambda: events.append("reset") or True,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.mps, "is_available", lambda: False)

    mu.release_torch_memory(force_gc=False)
    assert events == ["gc", "trim"]


def test_force_gc_does_not_force_native_pressure_cleanup(monkeypatch):
    from algan.utils import memory_utils as mu

    events = []
    monkeypatch.setattr(mu, "_host_memory_pressure", lambda: False)
    monkeypatch.setattr(mu, "_gpu_memory_pressure", lambda: False)
    monkeypatch.setattr(mu.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(mu, "_malloc_trim", lambda: events.append("trim") or True)
    monkeypatch.setattr(
        mu,
        "_reset_quadrants_runtime_for_memory_pressure",
        lambda: events.append("reset") or True,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.mps, "is_available", lambda: False)

    mu.release_torch_memory(force_gc=True)
    assert events == ["gc"]


@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (1,), (2, 3)])
def test_scalar_empty_and_multidimensional_allocations(shape, persist):
    memory = _arena(num_bytes=129)
    memory.get_tensor((3,), torch.uint8)
    value = memory.get_tensor(shape, torch.float32, persist=persist)
    assert value.shape == shape
    assert value.dtype == torch.float32
    assert value.untyped_storage()._cdata == memory.data.untyped_storage()._cdata
    assert memory.current_pointer <= memory.current_reverse_pointer
    assert (
        memory.max_pointer
        == memory.current_pointer + len(memory) - memory.current_reverse_pointer
    )


@pytest.mark.parametrize("persist", [False, True])
def test_scalar_clone_and_cast_copy_directly_into_arena(persist):
    memory = _arena()
    value = torch.tensor(3.25)
    clone = memory.clone(value, persist=persist)
    cast = memory.cast(value, torch.float64, persist=persist)
    assert clone.shape == cast.shape == ()
    assert clone.item() == cast.item() == value.item()
    assert clone.untyped_storage()._cdata == memory.data.untyped_storage()._cdata
    assert cast.untyped_storage()._cdata == memory.data.untyped_storage()._cdata
    value.fill_(9)
    assert clone.item() == cast.item() == 3.25


@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.parametrize(
    ("shape", "error"),
    [((-1,), ValueError), ((0, -1), ValueError), ((1.5,), TypeError)],
)
def test_invalid_shape_does_not_change_allocation_state(shape, error, persist):
    memory = _arena()
    memory.get_tensor((3,), torch.uint8)
    before = memory.get_pointers(), memory.max_pointer
    with pytest.raises(error):
        memory.get_tensor(shape, persist=persist)
    assert (memory.get_pointers(), memory.max_pointer) == before


@pytest.mark.parametrize("persist", [False, True])
def test_view_construction_failure_does_not_change_allocation_state(
    monkeypatch, persist
):
    memory = _arena()
    memory.get_tensor((3,), torch.uint8)
    before = memory.get_pointers(), memory.max_pointer

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected view failure")

    monkeypatch.setattr(torch.Tensor, "view", fail)
    with pytest.raises(RuntimeError, match="injected view failure"):
        memory.get_tensor((3,), persist=persist)
    assert (memory.get_pointers(), memory.max_pointer) == before


@pytest.mark.parametrize("persist", [False, True])
def test_dtype_alignment_exhaustion_keeps_state(persist):
    memory = _arena(num_bytes=7)
    memory.get_tensor((3,), torch.uint8)
    before = memory.get_pointers(), memory.max_pointer
    with pytest.raises(InsufficientMemoryException):
        memory.get_tensor((1,), torch.float32, persist=persist)
    assert (memory.get_pointers(), memory.max_pointer) == before
