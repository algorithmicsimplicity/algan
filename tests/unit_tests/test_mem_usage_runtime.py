"""The replayed trace must equal what ManualMemory actually does.

Every calibrated memory figure is produced by replaying a recorded allocation
stream. If the replay and the arena disagree by even one byte the whole scheme
is unsound, so these tests drive both and compare exactly.
"""

import pytest
import torch

from algan.rendering.mem_usage_runtime import (
    ALLOC,
    TEMP_POP,
    TEMP_PUSH,
    is_monotone_in_frames,
    peak_bytes,
    replay,
    trace_from_events,
)
from algan.utils.memory_utils import ManualMemory


def _arena(num_bytes=1 << 20):
    return ManualMemory(
        0, device=torch.device("cpu"), managed=True, num_bytes=num_bytes
    )


def _record(build, num_frames, initial_pointer=0):
    """Run ``build`` against a real arena; return (events, peak, arena)."""
    memory = _arena()
    if initial_pointer:
        memory.get_tensor((initial_pointer,), torch.uint8)
    with memory.recording() as recorder:
        entry = memory.current_pointer
        with memory.scope("under_test"):
            build(memory, num_frames)
        scope = recorder.scopes("under_test")[0]
    return scope.events, scope.peak_forward, entry, memory


# --------------------------------------------------------------------------
# Replay vs. the real arena
# --------------------------------------------------------------------------

def _mixed_dtype_workload(memory, frames):
    memory.get_tensor((frames, 3), torch.float32)
    memory.get_tensor((5,), torch.uint8)
    memory.get_tensor((frames * 2 + 1,), torch.float64)
    memory.get_tensor((7,), torch.uint8)
    memory.get_tensor((frames,), torch.int32)


def _temp_scoped_workload(memory, frames):
    memory.get_tensor((frames, 4), torch.float32)
    with memory.temp():
        memory.get_tensor((frames * 16,), torch.float32)
        memory.get_tensor((3,), torch.uint8)
    memory.get_tensor((frames,), torch.uint8)


def _nested_temp_workload(memory, frames):
    memory.get_tensor((frames,), torch.uint8)
    with memory.temp():
        memory.get_tensor((frames * 2,), torch.float32)
        with memory.temp():
            memory.get_tensor((frames * 8,), torch.float64)
        memory.get_tensor((frames * 3,), torch.float32)


@pytest.mark.parametrize("build", [
    _mixed_dtype_workload, _temp_scoped_workload, _nested_temp_workload,
])
@pytest.mark.parametrize("initial_pointer", [0, 1, 3, 7])
def test_replayed_peak_equals_the_arena_peak_exactly(build, initial_pointer):
    # Build a trace from two frame counts, then check it reproduces the arena
    # at frame counts it was never fitted on.
    low_events, _, _, _ = _record(build, 1, initial_pointer)
    high_events, _, _, _ = _record(build, 2, initial_pointer)
    trace = trace_from_events(
        low_events, ((1, low_events), (2, high_events)))

    for frames in (1, 2, 3, 5, 8, 13):
        _, observed, entry, _ = _record(build, frames, initial_pointer)
        assert peak_bytes(trace, frames, entry) == observed, (
            f"frames={frames} initial_pointer={initial_pointer}")


@pytest.mark.parametrize("initial_pointer", [0, 1, 3, 7])
def test_alignment_padding_is_reproduced_not_bounded(initial_pointer):
    # A uint8 run leaves the pointer unaligned; the following float64 must be
    # padded by exactly the same amount in both the arena and the replay.
    def build(memory, frames):
        memory.get_tensor((frames,), torch.uint8)
        memory.get_tensor((1,), torch.float64)

    events, observed, entry, _ = _record(build, 3, initial_pointer)
    trace = trace_from_events(events, ((3, events), (4, _record(
        build, 4, initial_pointer)[0])))
    assert peak_bytes(trace, 3, entry) == observed


def test_replay_charges_released_temp_bytes_to_the_peak():
    trace = (
        (ALLOC, 0, 4, 4, False),
        (TEMP_PUSH, False),
        (ALLOC, 0, 64, 4, False),
        (TEMP_POP,),
        (ALLOC, 0, 4, 4, False),
    )
    result = replay(trace, 2)
    # Peak includes the transient block; the end pointer does not.
    assert result.peak == (8 + 128) * 4
    assert result.end == 16 * 4


# --------------------------------------------------------------------------
# Trace construction
# --------------------------------------------------------------------------

def test_trace_solves_intercept_and_slope_exactly():
    low = [("alloc", "m.f", "torch.float32", False, 7, 4)]
    high = [("alloc", "m.f", "torch.float32", False, 13, 4)]
    trace = trace_from_events(low, ((1, low), (3, high)))
    # numel = 4 + 3 * frames
    assert trace == ((ALLOC, 4, 3, 4, False),)


def test_trace_construction_rejects_data_dependent_control_flow():
    # Bloom short-circuits when there is nothing to glow; a calibration corpus
    # that hits both paths must fail loudly rather than fit the shorter one.
    low = [("alloc", "m.f", "torch.float32", False, 4, 4)]
    high = [
        ("alloc", "m.f", "torch.float32", False, 8, 4),
        ("alloc", "m.g", "torch.float32", False, 2, 4),
    ]
    with pytest.raises(ValueError, match="data-dependent control flow"):
        trace_from_events(low, ((1, low), (2, high)))


def test_trace_construction_rejects_a_nonlinear_allocation():
    low = [("alloc", "m.f", "torch.float32", False, 4, 4)]
    high = [("alloc", "m.f", "torch.float32", False, 9, 4)]
    with pytest.raises(ValueError, match="not affine"):
        trace_from_events(low, ((1, low), (3, high)))


def test_trace_construction_rejects_a_changed_dtype():
    low = [("alloc", "m.f", "torch.float32", False, 4, 4)]
    high = [("alloc", "m.f", "torch.float16", False, 8, 2)]
    with pytest.raises(ValueError, match="changed dtype/persist"):
        trace_from_events(low, ((1, low), (2, high)))


# --------------------------------------------------------------------------
# Monotonicity -- required by _max_duration_that_fits
# --------------------------------------------------------------------------

def test_monotonicity_check_accepts_a_growing_trace():
    assert is_monotone_in_frames(((ALLOC, 3, 5, 4, False),))


def test_monotonicity_check_rejects_a_shrinking_trace():
    # A negative slope would make the chunk-size binary search non-monotone and
    # trip render_loop's "fit was not monotone" guard.
    assert not is_monotone_in_frames(((ALLOC, 400, -3, 4, False),))


# --------------------------------------------------------------------------
# Reverse (persistent) allocations
# --------------------------------------------------------------------------

def test_reverse_allocations_are_exact_when_the_arena_length_is_known():
    memory = _arena(4096)
    with memory.recording() as recorder:
        with memory.scope("under_test"):
            memory.get_tensor((3,), torch.uint8, persist=True)
            memory.get_tensor((5,), torch.float64, persist=True)
        scope = recorder.scopes("under_test")[0]

    trace = tuple(
        (ALLOC, event[4], 0, event[5], event[3]) for event in scope.events
    )
    result = replay(trace, 1, arena_length=4096)
    assert result.reverse_peak == scope.peak_reverse


def test_reverse_allocations_are_over_estimated_without_the_arena_length():
    trace = ((ALLOC, 5, 0, 8, True),)
    exact = replay(trace, 1, arena_length=4096).reverse_peak
    bounded = replay(trace, 1).reverse_peak
    assert bounded >= exact
    assert bounded - exact <= 7


def test_replay_rejects_an_unknown_event_tag():
    with pytest.raises(ValueError, match="unknown trace event"):
        replay((("Z", 1),), 1)
