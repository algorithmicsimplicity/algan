"""Parity + A/B for the attribute-timeline state query.

``AttributeTimeline.rematerialize_state_at_times`` used to answer its per-row
"which edit is live at time t" search with two Taichi kernels
(``_query_state_from_edits`` / ``_query_selected_state_from_edits``). Taichi's
arch is the *render* device, so launching them with the CPU animation tensors
made Taichi stage every argument -- the CSR edit table *and* the whole
``[T, N, D]`` result, both ways -- through VRAM, on the batch-prep worker
thread that is otherwise deliberately kept off the GPU. On a long scene that
is hundreds of MB of driver allocation per batch racing the in-flight render;
it crashed rl2/animations/main.py with ``CUDA_ERROR_OUT_OF_MEMORY`` inside
``cuMemAllocAsync``.

The query is now a flat ``torch.searchsorted`` over a composite
``row * n_ranks + rank(timestamp)`` key (``timeline.EditQueryIndex``). This
script asserts the replacement is byte-identical to the kernels it replaces
(and to a brute-force reference), and times both.

    .venv/Scripts/python.exe benchmarks/_timeline_query_parity.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _memory_cap import cap_process_memory  # noqa: E402

# Sizes here come from parameters, not from a real scene, so a mis-sized
# generator can eat the machine (it has: an earlier version of make_edits
# produced blocks of n_rows//4 rows, i.e. an edit log ~250x larger than any
# real attribute timeline, and exhausted system RAM). Cap first, import torch
# second. Note WDDM charges CUDA allocations against this process commit too,
# so the Taichi arm's device staging comes out of the same budget.
cap_process_memory(float(os.environ.get("ALGAN_BENCH_MEM_GB", "4")))

import torch  # noqa: E402

from algan.rendering.taichi_runtime import init_taichi, sync_devices  # noqa: E402

init_taichi()

from algan.animation_timeline import timeline as tl  # noqa: E402


def make_edits(n_rows, n_edits, channels, *, block=8, seed=0, duplicate_times=True):
    """Synthesize an edit log shaped like ``prepare_for_queries`` output.

    Edits are emitted in execution order with non-decreasing timestamps (which
    is what ``_resolve_replay_windows`` guarantees along every row), each
    touching one contiguous block of at most ``2 * block`` rows, and the log is
    terminated by the all-rows ``inf`` edit that carries the post-animation
    current state.

    ``block`` is what keeps the total CSR size realistic. Measured on
    rl2/s05: location N=507,710 U=879,578, color N=260,564 U=540,506, opacity
    N=134,949 U=599,035 -- i.e. U/N between 1.7 and 4.4, so each edit touches
    a *small* block of rows, never a fraction of the whole buffer.
    """
    g = torch.Generator().manual_seed(seed)
    edits = []
    t = 0.0
    for i in range(n_edits):
        begin = int(torch.randint(0, n_rows, (1,), generator=g))
        size = int(torch.randint(1, 2 * block, (1,), generator=g))
        end = min(n_rows, begin + size)
        # Same-end edits are legal and must resolve to the earliest-executed
        # one, so keep some timestamps tied.
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


def reference(times, n_rows, edits):
    """Brute-force: the live edit for a row at time t is the first edit
    touching that row whose timestamp is > t (edits in execution order).
    """
    channels = edits[0]["values"].shape[1]
    out = torch.zeros(times.shape[0], n_rows, channels)
    per_row = [[] for _ in range(n_rows)]
    for edit in edits:
        for k, row in enumerate(edit["indexes"].tolist()):
            per_row[row].append((edit["timestamp"], edit["values"][k]))
    for ti, t in enumerate(times.tolist()):
        for row in range(n_rows):
            for stamp, value in per_row[row]:
                if stamp > t:
                    out[ti, row] = value
                    break
    return out


def both(times, n_rows, edits, active_rows):
    prepared = tl._prepare_array_state_queries(times, n_rows, edits)
    torch_out = tl.generate_array_states(
        times, n_rows, edits, active_rows=active_rows, prepared=prepared
    )
    taichi_out = tl._generate_array_states_taichi(
        times,
        n_rows,
        prepared,
        None if active_rows is None else active_rows.to(torch.int64),
        times.shape[0],
        prepared.sorted_values.shape[1],
        prepared.sorted_values.dtype,
        times.device,
    )
    return torch_out, taichi_out


def check_small():
    failures = 0
    for seed in range(6):
        n_rows, n_edits, channels = 37, 23, 4
        edits, t_max = make_edits(n_rows, n_edits, channels, seed=seed)
        times = torch.linspace(-0.5, t_max + 0.5, 11)
        for label, active_rows in (
            ("all rows", None),
            ("empty selection", torch.zeros(0, dtype=torch.int64)),
            ("single row", torch.tensor([5])),
            ("sparse selection", torch.arange(0, n_rows, 7)),
        ):
            torch_out, taichi_out = both(times, n_rows, edits, active_rows)
            ok = torch.equal(torch_out, taichi_out)
            ref = reference(times, n_rows, edits)
            if active_rows is not None:
                mask = torch.zeros(n_rows, dtype=torch.bool)
                mask[active_rows] = True
                ref = ref * mask.view(1, -1, 1)
            ref_ok = torch.equal(torch_out, ref)
            if not (ok and ref_ok):
                failures += 1
                print(
                    f"  FAIL seed={seed} {label}: taichi_match={ok} ref_match={ref_ok}"
                )
    print(f"small-case parity: {'PASS' if failures == 0 else f'{failures} FAILURES'}")
    return failures


def check_edge_cases():
    failures = 0
    # Rows nothing ever edits, a single timestamp, T == 1, and non-finite
    # recorded values (a multiply-by-mask would turn those into NaN).
    edits = [
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
        for active_rows in (None, torch.tensor([0, 1, 4]), torch.tensor([3])):
            torch_out, taichi_out = both(times, 5, edits, active_rows)
            if not torch.equal(torch_out, taichi_out):
                failures += 1
                print(f"  FAIL edge times={times.tolist()} rows={active_rows}")
            if torch_out.isnan().any():
                failures += 1
                print(f"  FAIL NaN leaked times={times.tolist()} rows={active_rows}")
    print(f"edge-case parity: {'PASS' if failures == 0 else f'{failures} FAILURES'}")
    return failures


#: Skip any configuration whose live tensors are projected past this (bytes).
#: Both implementations' full [T, N, D] results plus a transient are held at
#: once for the comparison. Under the process cap this also has to leave room
#: for the Taichi arm's *device* staging, which WDDM charges against the
#: process's system-memory commit -- so keep the A/B shapes at or below the
#: real-scene ones and measure the full set separately in bench_real_shapes().
_BENCH_BUDGET = int(os.environ.get("ALGAN_BENCH_BUDGET", 900_000_000))

#: Attribute shapes measured on one real rl2/s05 batch, for the torch-only
#: timing: (label, N, U-per-row, D, R).
_REAL_SHAPES = (
    ("location", 507_710, 1.73, 3, 136_181),
    ("color", 260_564, 2.07, 5, 85_159),
    ("opacity", 134_949, 4.44, 1, 29_521),
    ("basis", 18_877, 1.13, 9, 3_226),
)


def bench():
    print("\nA/B (CPU animation tensors, Taichi arch = cuda):")
    for n_rows, n_edits, channels, n_frames, frac in (
        (20_000, 2_000, 4, 50, 1.0),
        (20_000, 2_000, 9, 50, 0.25),
        (100_000, 8_000, 5, 52, 1.0),
        (100_000, 8_000, 5, 52, 0.33),
        # The real color-attribute shape of one rl2/s05 batch.
        (260_564, 27_000, 5, 52, 0.33),
    ):
        # Both results plus one transient are live at the comparison.
        projected = 3 * 4 * n_frames * n_rows * channels
        if projected > _BENCH_BUDGET:
            print(
                f"  N={n_rows} D={channels} T={n_frames}: SKIPPED "
                f"({projected / 1e9:.2f} GB projected > budget)"
            )
            continue
        edits, t_max = make_edits(n_rows, n_edits, channels, seed=1)
        times = torch.linspace(0.0, t_max, n_frames)
        active_rows = (
            None
            if frac >= 1.0
            else torch.arange(0, n_rows, int(1 / frac), dtype=torch.int64)
        )
        prepared = tl._prepare_array_state_queries(times, n_rows, edits)
        args = (
            times,
            n_rows,
            prepared,
            active_rows,
            n_frames,
            channels,
            torch.float32,
            times.device,
        )

        warm = tl._generate_array_states_taichi(*args)  # warm/compile
        del warm  # nothing else may hold a [T, N, D] buffer during the timing
        sync_devices()
        s = time.perf_counter()
        a = tl._generate_array_states_taichi(*args)
        sync_devices()
        taichi_s = time.perf_counter() - s

        warm = tl.generate_array_states(
            times, n_rows, edits, active_rows=active_rows, prepared=prepared
        )
        del warm
        s = time.perf_counter()
        b = tl.generate_array_states(
            times, n_rows, edits, active_rows=active_rows, prepared=prepared
        )
        torch_s = time.perf_counter() - s

        rows = n_rows if active_rows is None else active_rows.numel()
        match = torch.equal(a, b)
        print(
            f"  N={n_rows:>7} D={channels} T={n_frames} R={rows:>7} "
            f"out={a.numel() * 4 / 1e6:6.1f}MB | taichi {taichi_s * 1e3:8.1f}ms "
            f"torch {torch_s * 1e3:8.1f}ms  {taichi_s / torch_s:5.2f}x  "
            f"identical={match}"
        )
        del a, b, prepared, edits
        if not match:
            raise SystemExit("A/B outputs differ")


def bench_real_shapes():
    """Torch-only timing at the attribute shapes of one real rl2/s05 batch.

    The Taichi arm is not run here: on this 4 GB card it cannot stage a
    ``[52, 507710, 3]`` result at all -- that is the crash being fixed -- and
    the host cap tightens it further, since WDDM charges the staging against
    the process commit.
    """
    print("\nreal rl2/s05 batch shapes (T=52), torch query only:")
    n_frames = 52
    total = 0.0
    for label, n_rows, u_ratio, channels, active in _REAL_SHAPES:
        n_edits = max(1, int((n_rows * u_ratio - n_rows) / 8))
        edits, t_max = make_edits(n_rows, n_edits, channels, seed=2)
        times = torch.linspace(0.0, t_max, n_frames)
        rows = torch.linspace(0, n_rows - 1, active).to(torch.int64).unique()
        prepared = tl._prepare_array_state_queries(times, n_rows, edits)

        warm = tl._query_row_states(times, prepared, rows)
        del warm
        s = time.perf_counter()
        compact = tl._query_row_states(times, prepared, rows)
        query_s = time.perf_counter() - s

        s = time.perf_counter()
        out = torch.zeros(
            (n_frames, n_rows, channels), dtype=torch.float32
        ).index_copy_(1, rows, compact)
        place_s = time.perf_counter() - s
        total += query_s + place_s
        print(
            f"  {label:<9} N={n_rows:>7} D={channels} R={rows.numel():>7} "
            f"U={prepared.keys.shape[0]:>8} | query {query_s * 1e3:7.1f}ms "
            f"+ zero/scatter {place_s * 1e3:6.1f}ms  "
            f"(taichi would have staged {out.numel() * 4 / 1e6:.0f}MB each way)"
        )
        del out, compact, prepared, edits
    print(f"  total per batch: {total * 1e3:.0f}ms")
    print("  measured old path: ~14.7s per batch fetch (see module docstring)")


if __name__ == "__main__":
    failures = check_small() + check_edge_cases()
    bench()
    bench_real_shapes()
    raise SystemExit(1 if failures else 0)
