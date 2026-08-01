"""Replay engine for the measured render-memory tables in ``mem_usage``.

The tables record, per (scope, route key), the *ordered allocation stream* a
render performs -- not a fitted byte total. Each event's element count is exact
and affine in the frame count, so replaying the stream through the same
alignment arithmetic :class:`~algan.utils.memory_utils.ManualMemory` uses
reproduces its ``max_pointer`` byte-for-byte at any entry pointer.

Why a trace rather than a formula:

* ``ManualMemory`` aligns each allocation to its dtype's item size, so a
  scope's byte total is piecewise-affine with a sawtooth of up to seven bytes
  per allocation -- no polynomial reproduces it, and a bound would forfeit the
  exactness the post-processing tests assert.
* Bloom's working shapes are not polynomial in resolution at all
  (``radius = ceil(3 * sigma)``, FFT lengths rounded to fast sizes), so
  resolution is pinned in the route key instead of fitted. That leaves the
  frame count as the only free continuous axis, and every allocation is exactly
  ``a + b * frames`` elements along it.

This module is hand-written and separately tested; ``mem_usage`` is generated
and holds only data, so its diffs contain no logic to review.
"""

from __future__ import annotations

from functools import partial

# Event tags. Kept as one-character strings so the generated tables stay
# compact and readable.
ALLOC = "A"
TEMP_PUSH = "("
TEMP_POP = ")"


def callable_identity(process):
    """Stable cross-process identity for a post-process callable.

    Returns ``None`` when no stable identity exists -- a closure, a lambda, or
    arguments that do not ``repr`` reproducibly. Callers must then fall back to
    an in-process cache: writing such an entry to disk would key one run's
    measurement to a different run's callable.
    """
    target = process
    args = ()
    kwargs = {}
    if isinstance(process, partial):
        target = process.func
        args = tuple(process.args)
        kwargs = dict(process.keywords or {})
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if not module or not qualname:
        return None
    if "<locals>" in qualname or "<lambda>" in qualname:
        return None
    try:
        detail = repr(args) + repr(sorted(kwargs.items()))
    except Exception:
        return None
    if "<" in detail and " object at 0x" in detail:
        # A default repr embeds an address; not reproducible.
        return None
    return f"{module}.{qualname}{detail}"


def post_process_chain_id(post_processes):
    """Identity of a whole post-process chain, or ``None`` when unstable."""
    parts = []
    for process in post_processes or ():
        identity = callable_identity(process)
        if identity is None:
            return None
        parts.append(identity)
    return "|".join(parts)


def align_up(pointer, item_size):
    """Forward-arena alignment: mirrors ``ManualMemory.get_tensor``."""
    return pointer + (-pointer) % item_size


def align_down(pointer, item_size):
    """Reverse-arena (``persist=True``) alignment, which rounds downwards."""
    return pointer - (pointer % item_size)


def event_numel(event, num_frames):
    """Element count of an ``ALLOC`` event at ``num_frames``."""
    _, a, b, _item_size, _persist = event
    return int(a) + int(b) * int(num_frames)


class ReplayResult:
    """Outcome of replaying one trace.

    ``peak`` and ``end`` are relative to the entry pointer, so they can be
    added to whatever pointer the arena happens to be at.
    """

    __slots__ = ("peak", "end", "reverse_peak", "reverse_end", "alloc_count")

    def __init__(self, peak, end, reverse_peak, reverse_end, alloc_count):
        self.peak = peak
        self.end = end
        self.reverse_peak = reverse_peak
        self.reverse_end = reverse_end
        self.alloc_count = alloc_count

    def __repr__(self):
        return (f"ReplayResult(peak={self.peak}, end={self.end}, "
                f"reverse_peak={self.reverse_peak}, "
                f"alloc_count={self.alloc_count})")


def replay(trace, num_frames, initial_pointer=0, *, arena_length=None,
           initial_reverse_used=0):
    """Replay ``trace`` and return its exact arena footprint.

    Parameters
    ----------
    trace
        Ordered events as stored in ``mem_usage.TRACES``.
    num_frames
        Frames in the chunk being sized.
    initial_pointer
        Forward pointer the scope starts from. Alignment padding depends on it,
        which is why it is an input rather than assumed zero.
    arena_length
        Total arena bytes. Only needed when the trace contains ``persist``
        allocations: the reverse pointer aligns *downwards* from the end of the
        arena, so its padding depends on the arena's absolute size. When it is
        unknown, reverse allocations are charged their worst-case alignment,
        which over-estimates by at most seven bytes each.
    initial_reverse_used
        Bytes already committed to the reverse (persistent) end.

    Notes
    -----
    Nested ``temp`` scopes restore the forward pointer on pop, and the reverse
    pointer too when the recorded push had ``clear_persist`` set -- matching
    :class:`~algan.utils.memory_utils.TempMemoryContext`. Released bytes still
    count towards ``peak``, which is what the batcher must reserve.
    """
    num_frames = int(num_frames)
    forward = int(initial_pointer)
    reverse_used = int(initial_reverse_used)
    exact_reverse = arena_length is not None

    peak_forward = forward
    peak_reverse = reverse_used
    alloc_count = 0
    stack = []

    for event in trace:
        tag = event[0]
        if tag == ALLOC:
            item_size = int(event[3])
            num_bytes = event_numel(event, num_frames) * item_size
            if event[4]:
                if exact_reverse:
                    pointer = align_down(
                        int(arena_length) - reverse_used, item_size)
                    reverse_used = int(arena_length) - (pointer - num_bytes)
                else:
                    # Worst case: the reverse pointer needs the full
                    # ``item_size - 1`` bytes of downward padding.
                    reverse_used += num_bytes + item_size - 1
                peak_reverse = max(peak_reverse, reverse_used)
            else:
                forward = align_up(forward, item_size) + num_bytes
                peak_forward = max(peak_forward, forward)
            alloc_count += 1
        elif tag == TEMP_PUSH:
            stack.append((forward, reverse_used, bool(event[1])))
        elif tag == TEMP_POP:
            if not stack:
                # An unbalanced pop can only come from a truncated recording;
                # ignoring it keeps a replay from drifting arbitrarily.
                continue
            saved_forward, saved_reverse, clear_persist = stack.pop()
            forward = saved_forward
            if clear_persist:
                reverse_used = saved_reverse
        else:
            raise ValueError(f"unknown trace event tag {tag!r}")

    return ReplayResult(
        peak=peak_forward - int(initial_pointer),
        end=forward - int(initial_pointer),
        reverse_peak=peak_reverse - int(initial_reverse_used),
        reverse_end=reverse_used - int(initial_reverse_used),
        alloc_count=alloc_count,
    )


def peak_bytes(trace, num_frames, initial_pointer=0, **kwargs):
    """Forward-arena bytes a trace needs above ``initial_pointer``."""
    return replay(trace, num_frames, initial_pointer, **kwargs).peak


def is_monotone_in_frames(trace, initial_pointer=0, max_frames=64):
    """Whether the replayed peak never decreases as the frame count grows.

    ``render_loop._max_duration_that_fits`` binary-searches the chunk size and
    raises if the fit predicate is not monotone, so this is a hard requirement
    on every shipped trace rather than an observation about them. The generator
    asserts it before writing a table.
    """
    previous = None
    for frames in range(1, int(max_frames) + 1):
        current = peak_bytes(trace, frames, initial_pointer)
        if previous is not None and current < previous:
            return False
        previous = current
    return True


def trace_from_events(events, frames_by_event):
    """Build a replayable trace from two recordings at different frame counts.

    ``events`` is the recorded stream from
    :class:`~algan.utils.memory_utils.AllocationRecorder` (whose allocation
    events carry an absolute ``numel``); ``frames_by_event`` supplies the
    matching element counts from a second recording taken at a different frame
    count. Each allocation's ``numel = a + b * frames`` is solved exactly from
    the pair.

    Raises
    ------
    ValueError
        If the two recordings disagree in length or event kind -- which means
        the scope has data-dependent control flow and cannot be represented as
        one trace.
    """
    (low_frames, low_events), (high_frames, high_events) = frames_by_event
    if len(low_events) != len(high_events):
        raise ValueError(
            f"allocation count changed between {low_frames} and {high_frames} "
            f"frames ({len(low_events)} vs {len(high_events)}): the scope has "
            f"data-dependent control flow and cannot be traced")
    if int(high_frames) == int(low_frames):
        raise ValueError("need two distinct frame counts to solve the slope")

    span = int(high_frames) - int(low_frames)
    trace = []
    for index, (low, high) in enumerate(zip(low_events, high_events)):
        if low[0] != high[0]:
            raise ValueError(
                f"event {index} changed kind between recordings "
                f"({low[0]!r} vs {high[0]!r})")
        if low[0] != "alloc":
            trace.append(
                (TEMP_PUSH, bool(low[1])) if low[0] == "temp_push"
                else (TEMP_POP,))
            continue
        if low[2] != high[2] or low[3] != high[3]:
            raise ValueError(
                f"event {index} changed dtype/persist between recordings "
                f"({low[2]}, persist={low[3]} vs {high[2]}, "
                f"persist={high[3]})")
        delta = int(high[4]) - int(low[4])
        if delta % span:
            raise ValueError(
                f"event {index} is not affine in the frame count: numel went "
                f"{low[4]} -> {high[4]} over {span} frames")
        slope = delta // span
        intercept = int(low[4]) - slope * int(low_frames)
        trace.append((ALLOC, intercept, slope, int(low[5]), bool(low[3])))
    return tuple(trace)
