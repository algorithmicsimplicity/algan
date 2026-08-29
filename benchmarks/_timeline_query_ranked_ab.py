"""A/B: rank-deduplicated Taichi kernels vs the torch attribute-state query.

``benchmarks/_timeline_query_parity.py`` times the *original* kernels against
the torch query, and those kernels run one binary search per ``(frame, row)``
pair -- while the torch path searches once per distinct *rank* and skips rows
that are constant across the window. That comparison is between two algorithms,
not two backends.

``_timeline_query_ranked_taichi`` gives the kernels the same two savings. This
script checks both new kernels are byte-identical to the torch query and times
all four implementations in the two regimes that matter:

* **whole timeline** -- ``times`` spanning the entire scene, which is what
  ``_timeline_query_parity.bench()`` uses. Every frame lands on its own rank
  (rank dedup cannot fire) and roughly half the rows change (row dedup fires
  weakly).
* **render window** -- ``times`` spanning a couple of seconds mid-scene, which
  is what a real batch fetch asks for. A handful of distinct ranks cover the
  whole window and ~0.1% of rows change, so both savings fire hard.

Run on a CPU-arch Taichi (this is CPU batch-prep work, so a CUDA-arch run would
stage every argument through VRAM -- see ``_timeline_query_parity``'s docstring)::

    ALGAN_RENDER_DEVICE=cpu ALGAN_USE_DAEMON=0 uv run python benchmarks/_timeline_query_ranked_ab.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _memory_cap import cap_process_memory  # noqa: E402

# Shapes come from parameters here, not from a real scene, so cap before torch
# is imported (see _memory_cap and _timeline_query_parity).
cap_process_memory(float(os.environ.get("ALGAN_RANKED_AB_MEM_GB", "10")))

import torch  # noqa: E402

from algan.rendering.taichi_runtime import _sync_devices, init_taichi  # noqa: E402

init_taichi()

from _timeline_query_ranked_taichi import (  # noqa: E402
    query_ranked_compact,
    query_ranked_placed,
    query_ranked_scatter,
    query_ranked_walk,
)

from algan.animation_timeline import timeline as tl  # noqa: E402


def make_edits(n_rows, n_edits, channels, *, block=8, seed=0, duplicate_times=True):
    """Edit log shaped like ``prepare_for_queries`` output.

    Kept identical to ``_timeline_query_parity.make_edits`` so the two scripts'
    numbers are comparable: execution order, non-decreasing timestamps, small
    contiguous row blocks (real scenes measure U/N between 1.7 and 4.4), some
    tied timestamps, and the terminating all-rows ``inf`` edit.
    """
    g = torch.Generator().manual_seed(seed)
    edits = []
    t = 0.0
    for i in range(n_edits):
        begin = int(torch.randint(0, n_rows, (1,), generator=g))
        size = int(torch.randint(1, 2 * block, (1,), generator=g))
        end = min(n_rows, begin + size)
        if not (duplicate_times and i % 3 == 1):
            t += float(torch.rand(1, generator=g))
        edits.append(
            {
                "indexes": torch.arange(begin, end),
                "values": torch.randn(end - begin, channels, generator=g),
                "timestamp": t,
            }
        )
    edits.append(
        {
            "indexes": torch.arange(n_rows),
            "values": torch.randn(n_rows, channels, generator=g),
            "timestamp": float("inf"),
        }
    )
    return edits, t


def _query_plan(times, index):
    """Distinct query ranks (ascending) and the frames that map to each.

    ``rank`` is the number of distinct edit timestamps <= t, which is the only
    thing a frame's answer depends on -- the same quantity ``_query_row_states``
    dedups on. Frames are grouped by rank into a CSR rather than assumed sorted,
    so unordered or repeated query times stay correct.
    """
    query_ranks = torch.searchsorted(
        index.unique_timestamps, times.contiguous(), right=True
    )
    unique_ranks, inverse = torch.unique(query_ranks, return_inverse=True)
    S = int(unique_ranks.shape[0])
    order = torch.argsort(inverse, stable=True)
    counts = torch.bincount(inverse, minlength=S)
    frame_head = torch.zeros(S + 1, dtype=torch.int64)
    torch.cumsum(counts, 0, out=frame_head[1:])
    # Both are bounded by T, so int32 avoids Taichi's i64 -> i32 range_for cast
    # warning on the inner frame loop.
    return (
        query_ranks,
        unique_ranks,
        inverse,
        frame_head.to(torch.int32),
        order.to(torch.int32),
    )


def taichi_compact(times, N, index, active_rows):
    """Faithful mirror of the torch path.

    Search per (distinct rank, row), expand over frames, then place into the
    global row layout.
    """
    T = int(times.shape[0])
    D = int(index.sorted_values.shape[1])
    dtype = index.sorted_values.dtype
    rows = torch.arange(N, dtype=torch.int64) if active_rows is None else active_rows
    R = int(rows.shape[0])
    if T == 0 or R == 0 or index.keys.shape[0] == 0:
        return torch.zeros((T, N, D), dtype=dtype)

    query_ranks, unique_ranks, inverse, _, _ = _query_plan(times, index)
    if int(unique_ranks.shape[0]) < T:
        ranks = unique_ranks
    else:
        ranks, inverse = query_ranks, None
    S = int(ranks.shape[0])

    compact = torch.empty((S, R, D), dtype=dtype)
    query_ranked_compact(
        ranks,
        rows,
        index.head,
        index.keys,
        index.sorted_values,
        compact,
        int(index.unique_timestamps.shape[0]),
    )
    if inverse is not None:
        compact = compact.index_select(0, inverse)
    if active_rows is None:
        return compact
    out = tl._sparsely_written_zeros((T, N, D), dtype, times.device)
    out.index_copy_(1, rows, compact)
    return out


def taichi_placed(times, N, index, active_rows):
    """Monotone per-row walk writing straight into the placed [T, N, D]."""
    T = int(times.shape[0])
    D = int(index.sorted_values.shape[1])
    dtype = index.sorted_values.dtype
    rows = torch.arange(N, dtype=torch.int64) if active_rows is None else active_rows
    if T == 0 or int(rows.shape[0]) == 0 or index.keys.shape[0] == 0:
        return torch.zeros((T, N, D), dtype=dtype)

    _, unique_ranks, _, frame_head, frame_ids = _query_plan(times, index)
    out = tl._sparsely_written_zeros((T, N, D), dtype, times.device)
    query_ranked_placed(
        unique_ranks,
        frame_head,
        frame_ids,
        rows,
        index.head,
        index.keys,
        index.sorted_values,
        out,
        int(index.unique_timestamps.shape[0]),
    )
    return out


def taichi_split(times, N, index, active_rows):
    """Split form of the walk.

    Phase 1 fills a small [R, S] landing-index buffer; phase 2 does one fused
    expand-and-place pass with the row index innermost.
    """
    T = int(times.shape[0])
    D = int(index.sorted_values.shape[1])
    dtype = index.sorted_values.dtype
    rows = torch.arange(N, dtype=torch.int64) if active_rows is None else active_rows
    R = int(rows.shape[0])
    if T == 0 or R == 0 or index.keys.shape[0] == 0:
        return torch.zeros((T, N, D), dtype=dtype)

    _, unique_ranks, inverse, _, _ = _query_plan(times, index)
    low = torch.empty((R, int(unique_ranks.shape[0])), dtype=torch.int32)
    query_ranked_walk(
        unique_ranks,
        rows,
        index.head,
        index.keys,
        low,
        int(index.unique_timestamps.shape[0]),
    )
    out = tl._sparsely_written_zeros((T, N, D), dtype, times.device)
    query_ranked_scatter(inverse.to(torch.int32), rows, low, index.sorted_values, out)
    return out


def torch_query(times, N, edits, index, active_rows):
    return tl.generate_array_states(
        times, N, edits, active_rows=active_rows, prepared=index
    )


def original_kernel(times, N, index, active_rows):
    return tl._generate_array_states_taichi(
        times,
        N,
        index,
        active_rows,
        int(times.shape[0]),
        int(index.sorted_values.shape[1]),
        index.sorted_values.dtype,
        times.device,
    )


def window_stats(times, index, N, active_rows):
    """How hard each dedup fires for this (times, index) pair."""
    query_ranks = torch.searchsorted(
        index.unique_timestamps, times.contiguous(), right=True
    )
    S = int(torch.unique(query_ranks).shape[0])
    rows = torch.arange(N, dtype=torch.int64) if active_rows is None else active_rows
    bases = rows * index.unique_timestamps.shape[0]
    lo = torch.searchsorted(index.keys, bases + query_ranks.amin())
    hi = torch.searchsorted(index.keys, bases + query_ranks.amax())
    changing = int((lo != hi).sum())
    return S, changing, int(rows.shape[0])


IMPLS = (
    (
        "torch",
        lambda times, N, edits, index, rows: torch_query(times, N, edits, index, rows),
    ),
    (
        "ti-compact",
        lambda times, N, edits, index, rows: taichi_compact(times, N, index, rows),
    ),
    (
        "ti-placed",
        lambda times, N, edits, index, rows: taichi_placed(times, N, index, rows),
    ),
    (
        "ti-split",
        lambda times, N, edits, index, rows: taichi_split(times, N, index, rows),
    ),
    (
        "ti-original",
        lambda times, N, edits, index, rows: original_kernel(times, N, index, rows),
    ),
)


def check_parity():
    """Byte-identity on small, edge and selection cases."""
    failures = 0
    cases = []
    for seed in range(4):
        n_rows, n_edits, channels = 37, 23, 4
        edits, t_max = make_edits(n_rows, n_edits, channels, seed=seed)
        for label, times in (
            ("full span", torch.linspace(-0.5, t_max + 0.5, 11)),
            ("narrow window", torch.linspace(t_max * 0.4, t_max * 0.4 + 0.3, 9)),
            ("repeated times", torch.tensor([1.0, 1.0, 2.0, 1.0])),
            ("unordered times", torch.tensor([3.0, 0.5, 2.0, 0.5, 9.0])),
            ("single time", torch.tensor([1.5])),
        ):
            for rows_label, rows in (
                ("all rows", None),
                ("empty selection", torch.zeros(0, dtype=torch.int64)),
                ("single row", torch.tensor([5])),
                ("sparse selection", torch.arange(0, n_rows, 7)),
            ):
                cases.append(
                    (f"seed={seed} {label} / {rows_label}", n_rows, edits, times, rows)
                )

    # Rows nothing ever edits, and non-finite recorded values (a multiply by a
    # mask would turn those into NaN; both paths must write a hard zero).
    edge_edits = [
        {
            "indexes": torch.tensor([1, 2]),
            "values": torch.tensor([[float("inf"), 1.0], [2.0, 3.0]]),
            "timestamp": 1.0,
        },
        {
            "indexes": torch.arange(5),
            "values": torch.randn(5, 2),
            "timestamp": float("inf"),
        },
    ]
    for times in (
        torch.tensor([0.0]),
        torch.tensor([1.0, 1.0, 2.0]),
        torch.tensor([-3.0, 0.5, 1.5]),
    ):
        for rows in (None, torch.tensor([0, 1, 4]), torch.tensor([3])):
            cases.append(
                (f"edge times={times.tolist()} rows={rows}", 5, edge_edits, times, rows)
            )

    for label, n_rows, edits, times, rows in cases:
        index = tl._prepare_array_state_queries(times, n_rows, edits)
        ref = torch_query(times, n_rows, edits, index, rows)
        for name, fn in IMPLS[1:]:
            got = fn(times, n_rows, edits, index, rows)
            if not torch.equal(ref, got):
                failures += 1
                n_diff = int((ref != got).sum())
                print(f"  FAIL {name}: {label} ({n_diff} elements differ)")
        if ref.isnan().any():
            failures += 1
            print(f"  FAIL NaN leaked: {label}")
    print(
        f"parity ({len(cases)} cases x {len(IMPLS) - 1} impls): {'PASS' if failures == 0 else f'{failures} FAILURES'}"
    )
    return failures


#: Every implementation's full [T, N, D] result is held at once for the
#: comparison, plus a transient.
_BUDGET = int(os.environ.get("ALGAN_RANKED_AB_BUDGET", 1_400_000_000))

#: Timed runs per implementation; the minimum is reported.
_REPEATS = int(os.environ.get("ALGAN_RANKED_AB_REPEATS", "7"))

_CONFIGS = (
    # (N, n_edits, D, T, active fraction)
    (20_000, 2_000, 4, 50, 1.0),
    (100_000, 8_000, 5, 52, 1.0),
    (100_000, 8_000, 5, 52, 0.33),
    # The real colour-attribute shape of one rl2/s05 batch.
    (260_564, 27_000, 5, 52, 0.33),
)


def bench(regime, window):
    print(f"\n{regime}:")
    for n_rows, n_edits, channels, n_frames, frac in _CONFIGS:
        projected = 4 * 4 * n_frames * n_rows * channels
        if projected > _BUDGET:
            print(
                f"  N={n_rows} D={channels} T={n_frames}: SKIPPED ({projected / 1e9:.2f} GB > budget)"
            )
            continue
        edits, t_max = make_edits(n_rows, n_edits, channels, seed=1)
        if window is None:
            times = torch.linspace(0.0, t_max, n_frames)
        else:
            start = t_max * 0.5
            times = torch.linspace(start, start + window, n_frames)
        active_rows = (
            None
            if frac >= 1.0
            else torch.arange(0, n_rows, int(1 / frac), dtype=torch.int64)
        )
        index = tl._prepare_array_state_queries(times, n_rows, edits)
        S, changing, R = window_stats(times, index, n_rows, active_rows)

        # Time one implementation at a time, holding only the reference beside
        # it: a loop that keeps every result live makes each later impl fault
        # in its buffer against progressively more resident memory, which on
        # these [T, N, D] shapes is worth more than the thing being measured.
        # Best-of-N for the same reason -- the first run pays first-touch on a
        # fresh allocation, later ones reuse the freed block. A few cores and a
        # few hundred MB per result make single-shot numbers swing by 3x, so
        # take the minimum, which is the least noise-contaminated estimate.
        ref = torch_query(times, n_rows, edits, index, active_rows)
        timings = {}
        mismatched = []
        for name, fn in IMPLS:
            warm = fn(times, n_rows, edits, index, active_rows)
            if not torch.equal(ref, warm):
                mismatched.append(name)
            del warm
            best = float("inf")
            for _ in range(_REPEATS):
                _sync_devices()
                s = time.perf_counter()
                got = fn(times, n_rows, edits, index, active_rows)
                _sync_devices()
                best = min(best, time.perf_counter() - s)
                del got
            timings[name] = best
        print(
            f"  N={n_rows:>7} D={channels} T={n_frames} R={R:>7} "
            f"| distinct ranks {S:>2}/{n_frames}, changing rows {changing / max(1, R):5.1%}"
        )
        base = timings["torch"]
        cells = "  ".join(
            f"{name} {timings[name] * 1e3:7.1f}ms ({base / timings[name]:.2f}x)"
            for name, _ in IMPLS
        )
        print(f"      {cells}")
        if mismatched:
            raise SystemExit(f"outputs differ from torch: {mismatched}")
        del ref, index, edits


if __name__ == "__main__":
    failures = check_parity()
    bench("whole-timeline times (what _timeline_query_parity.bench uses)", None)
    bench("render-window times (2s mid-scene, what a real batch fetches)", 2.0)
    raise SystemExit(1 if failures else 0)
