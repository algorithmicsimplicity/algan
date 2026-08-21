"""Parity + A/B for the event interval index (``ALGAN_OPT_DISABLE=eventindex``).

``FunctionTimeline.get_functions_for_times`` / ``get_updaters_for_times`` used
to test every event the scene ever recorded against every queried time. That is
O(scene) per batch with O(scene) batches -- O(n^2) over a render. They now prune
to a candidate range first (:func:`_event_interval_index`).

The pruning must be *exactly* transparent: same events, same order (replay
re-executes in recorded order). This asserts that against the unpruned scan on
randomized interval layouts chosen to hit the cases the two bounds care about --
including one long-lived event, which is the layout that defeats the lower
bound and must therefore still be correct, just slower.

    .venv/Scripts/python.exe benchmarks/_event_index_parity.py
"""

from __future__ import annotations

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _memory_cap import cap_process_memory  # noqa: E402

# Event counts come from parameters here, not from an authored scene.
cap_process_memory(float(os.environ.get("ALGAN_BENCH_MEM_GB", "4")))

import torch  # noqa: E402

from algan.animation_timeline.timeline import (  # noqa: E402
    _event_interval_index,
    _events_overlapping,
)


class Span:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class Event:
    """Enough of a FunctionApplication for the two lookups."""

    def __init__(self, i, start, end):
        self.i = i
        self.time = Span(start, end)
        self.replay_end = None


def reference(events, times):
    """The scan the index replaces."""
    return [
        e
        for e in events
        if bool(((e.time.start <= times) & (times < e.time.end)).any())
    ]


def indexed(events, starts, ends, index, times):
    candidates = _events_overlapping(index, times)
    if candidates.numel() == 0:
        return []
    t = times.view(1, -1)
    active = (
        (starts[candidates].view(-1, 1) <= t) & (t < ends[candidates].view(-1, 1))
    ).any(1)
    return [events[i] for i in candidates[active].sort().values.tolist()]


def make_events(n, span_seconds, layout, rng):
    events = []
    for i in range(n):
        start = rng.uniform(0, span_seconds)
        if layout == "short":
            dur = rng.uniform(0.05, 1.5)
        elif layout == "mixed":
            dur = rng.uniform(0.05, 1.5) if rng.random() < 0.95 else rng.uniform(5, 60)
        elif layout == "one_long":
            dur = span_seconds if i == 0 else rng.uniform(0.05, 1.5)
        elif layout == "degenerate":
            dur = 0.0  # empty interval: never active, must never be returned
        else:
            raise ValueError(layout)
        events.append(Event(i, start, start + dur))
    return events


def main():
    rng = random.Random(0)
    span = 260.0  # ~ the reference scene's length in seconds
    fps = 15.0
    print(
        f"{'layout':>12}{'events':>8}{'windows':>9}{'scan':>10}{'indexed':>10}{'speedup':>9}"
    )
    for layout in ("short", "mixed", "one_long", "degenerate"):
        for n in (2_000, 26_000):
            events = make_events(n, span, layout, rng)
            starts = torch.tensor([e.time.start for e in events], dtype=torch.float32)
            ends = torch.tensor([e.time.end for e in events], dtype=torch.float32)
            index = _event_interval_index(starts, ends)

            windows = []
            for start_frame in range(0, int(span * fps) - 50, 97):
                windows.append(
                    torch.arange(start_frame, start_frame + 50).to(torch.float32) / fps
                )
            # Also the edge cases: before everything, after everything, and a
            # single instant.
            windows.append(torch.tensor([-5.0, -4.0]))
            windows.append(torch.tensor([span + 10, span + 20]))
            windows.append(torch.tensor([span * 0.5]))
            # Unsorted times: only min/max feed the prune, so this must hold.
            shuffled = windows[0].tolist()
            rng.shuffle(shuffled)
            windows.append(torch.tensor(shuffled))

            for times in windows:
                want = reference(events, times)
                got = indexed(events, starts, ends, index, times)
                assert [e.i for e in want] == [e.i for e in got], (
                    f"{layout} n={n}: index returned "
                    f"{[e.i for e in got][:8]} want {[e.i for e in want][:8]}"
                )

            t0 = time.perf_counter()
            for times in windows:
                reference(events, times)
            t_scan = time.perf_counter() - t0
            t0 = time.perf_counter()
            for times in windows:
                indexed(events, starts, ends, index, times)
            t_idx = time.perf_counter() - t0
            print(
                f"{layout:>12}{n:>8}{len(windows):>9}"
                f"{t_scan * 1e3:>9.1f}ms{t_idx * 1e3:>9.1f}ms{t_scan / max(t_idx, 1e-9):>8.1f}x"
            )
    print("\nall layouts: index result identical to the full scan, order included")


if __name__ == "__main__":
    main()
