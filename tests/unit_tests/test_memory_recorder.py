"""Contracts for the ManualMemory allocation recorder.

The recorder is what lets the render memory tables be *measured* rather than
hand-maintained: because ``get_tensor`` is the arena's only allocation entry
point, recording there captures every current and future buffer with no
per-site annotation. These tests pin the properties the calibration generator
depends on -- exact event streams, correct peaks, and survival of the
render-failure unwind path.
"""

import pytest
import torch

from algan.utils.memory_utils import (
    ManualMemory,
    begin_cuda_peak,
    cuda_peak_scope,
    end_cuda_peak,
    peak_allocated,
    reset_peak_floor,
)


def _arena(num_bytes=1024):
    return ManualMemory(
        0, device=torch.device("cpu"), managed=True, num_bytes=num_bytes
    )


def _allocs(record):
    return [event for event in record.events if event[0] == "alloc"]


def test_recording_is_off_by_default():
    memory = _arena()
    memory.get_tensor((4,), torch.float32)
    assert memory._recorder is None
    # scope() must be free (and reusable) when not recording.
    with memory.scope("unused") as record:
        assert record is None


def test_recorder_captures_numel_dtype_and_persist_per_allocation():
    memory = _arena()
    with memory.recording() as recorder:
        with memory.scope("frame_buffers"):
            memory.get_tensor((2, 3), torch.float32)
            memory.get_tensor((5,), torch.uint8, persist=True)

    scope = recorder.scopes("frame_buffers")[0]
    kinds = _allocs(scope)
    assert [(event[2], event[3], event[4], event[5]) for event in kinds] == [
        ("torch.float32", False, 6, 4),
        ("torch.uint8", True, 5, 1),
    ]
    assert scope.alloc_count == 2
    # Alignment-free byte total: 6*4 forward + 5*1 reverse.
    assert scope.total_bytes() == 29


def test_scope_peaks_are_measured_relative_to_entry_and_split_by_direction():
    memory = _arena()
    memory.get_tensor((8,), torch.float32)  # 32 bytes before the scope opens.
    with memory.recording() as recorder:
        with memory.scope("wavefront"):
            memory.get_tensor((10,), torch.float32)
            memory.get_tensor((3,), torch.float32, persist=True)

    scope = recorder.scopes("wavefront")[0]
    assert scope.entry_forward == 32
    assert scope.peak_forward == 40
    assert scope.peak_reverse == 12


def test_temp_nesting_is_recorded_and_peak_survives_the_release():
    memory = _arena()
    with memory.recording() as recorder:
        with memory.scope("postprocess"):
            memory.get_tensor((4,), torch.float32)
            with memory.temp():
                memory.get_tensor((16,), torch.float32)
            memory.get_tensor((4,), torch.float32)

    scope = recorder.scopes("postprocess")[0]
    assert [event[0] for event in scope.events] == [
        "alloc", "temp_push", "alloc", "temp_pop", "alloc"
    ]
    # The transient 64 bytes are the peak even though they were released.
    assert scope.peak_forward == 80
    assert scope.exit_forward == 32


def test_nested_scopes_form_a_tree_and_parents_absorb_child_peaks():
    memory = _arena()
    with memory.recording() as recorder:
        with memory.scope("outer"):
            memory.get_tensor((2,), torch.float32)
            with memory.scope("inner"):
                memory.get_tensor((6,), torch.float32)

    outer = recorder.scopes("outer")[0]
    inner = recorder.scopes("inner")[0]
    assert [child.name for child in outer.children] == ["inner"]
    assert inner.peak_forward == 24
    assert outer.peak_forward == 32
    # Allocations are attributed to the innermost scope's event stream, but
    # counted by every enclosing scope.
    assert len(_allocs(outer)) == 1
    assert len(_allocs(inner)) == 1
    assert outer.alloc_count == 2


def test_scope_closes_and_records_when_the_body_raises():
    memory = _arena()
    with memory.recording() as recorder:
        with pytest.raises(RuntimeError, match="boom"):
            with memory.scope("wavefront"):
                memory.get_tensor((4,), torch.float32)
                raise RuntimeError("boom")
        # The recorder stack unwound, so a following scope is a sibling.
        with memory.scope("postprocess"):
            memory.get_tensor((1,), torch.float32)

    assert [child.name for child in recorder.root.children] == [
        "wavefront", "postprocess"
    ]


def test_reset_during_an_open_scope_does_not_corrupt_later_recordings():
    # ManualMemory.reset() runs on the render-failure path while an exception
    # is still unwinding through open scopes; the deferred pop must not then
    # attach itself to whatever scope is open next.
    memory = _arena()
    with memory.recording() as recorder:
        with pytest.raises(RuntimeError, match="oom"):
            with memory.scope("wavefront"):
                memory.get_tensor((4,), torch.float32)
                try:
                    raise RuntimeError("oom")
                finally:
                    memory.reset()
        with memory.scope("retry"):
            memory.get_tensor((2,), torch.float32)

    assert [child.name for child in recorder.root.children] == ["retry"]
    assert recorder.scopes("retry")[0].peak_forward == 8
    # The truncated scope is not silently averaged in, but is kept for
    # diagnostics.
    assert [record.name for record in recorder.abandoned] == ["wavefront"]


def test_recording_restores_the_previous_recorder_on_exit():
    memory = _arena()
    with memory.recording():
        pass
    assert memory._recorder is None
    memory.get_tensor((4,), torch.float32)  # must not raise


def test_caller_qualname_names_the_allocating_function_not_memory_utils():
    memory = _arena()

    def allocate_the_buffer():
        memory.get_tensor((4,), torch.float32)

    with memory.recording() as recorder:
        with memory.scope("frame_buffers"):
            allocate_the_buffer()

    qualname = _allocs(recorder.scopes("frame_buffers")[0])[0][1]
    assert qualname.startswith(__name__)
    assert "allocate_the_buffer" in qualname


def test_unmanaged_arena_is_never_recorded():
    # The prefetch worker's projection scratch is unmanaged and runs off the
    # render thread; recording it would interleave into another scope's stream.
    memory = ManualMemory(0, device=torch.device("cpu"), managed=False)
    with memory.recording() as recorder:
        with memory.scope("frame_buffers"):
            memory.get_tensor((4,), torch.float32)

    assert _allocs(recorder.scopes("frame_buffers")[0]) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_peak_scope_preserves_the_displaced_process_peak():
    reset_peak_floor()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    big = torch.empty(4 << 20, dtype=torch.uint8, device="cuda")
    outer = torch.cuda.max_memory_allocated()
    del big
    torch.cuda.empty_cache()

    with cuda_peak_scope("cuda") as inner_peak:
        small = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
        del small
    measured = inner_peak()

    assert measured >= (1 << 20)
    # The live counter was reset by the scope, but the outer peak survives.
    assert torch.cuda.max_memory_allocated() < outer
    assert peak_allocated() >= outer
    reset_peak_floor()


def test_cuda_peak_helpers_are_inert_off_cuda():
    token = begin_cuda_peak(torch.device("cpu"))
    assert token is None
    assert end_cuda_peak(token) == 0
    with cuda_peak_scope(torch.device("cpu")) as peak:
        assert peak() == 0
