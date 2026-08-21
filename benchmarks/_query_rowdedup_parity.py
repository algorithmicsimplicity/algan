"""Parity for the row dedup in ``_query_row_states`` (``ALGAN_OPT_DISABLE=rowdedup``).

``_query_row_states`` answers "what is row r's value at rank q" with one binary
search per (row, distinct-rank) pair. The dedup searches each row once at the
window's lowest and highest rank instead: the landing index is monotone in the
rank, so equal endpoints pin every intermediate rank to the same edit and the
row's value is constant across the window -- gathered once and broadcast. Only
rows with an edit boundary inside the window pay the per-rank search.

Three arms are compared, exactly:

  * the dedup path (default);
  * the dense path (``rowdedup`` disabled) -- the shipped behaviour before;
  * a brute-force per-(time, row) evaluation straight from the edit list, so a
    bug shared by both torch paths cannot pass silently.

Layout coverage includes the shapes that pick each branch: mostly-constant
windows (the dedup fast path -- asserted engaged, not assumed), boundary-heavy
windows (the break-even bail), an all-rows single edit, rows with no edits at
all, unsorted all-distinct times (the ``inverse is None`` path, where the
endpoint ranks come from ``amin``/``amax`` rather than the ends of a sorted
array), one- and two-rank windows, and scattered ``rows=`` subsets.

    .venv/Scripts/python.exe benchmarks/_query_rowdedup_parity.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

os.environ.setdefault("ALGAN_RENDER_DEVICE", "cpu")

import torch  # noqa: E402

import algan.animation_timeline.timeline as tl  # noqa: E402


def brute_force(times, N, edits, rows):
    """Per-(time, row) evaluation from the edit list, independent of the index.

    The state of row r at time t is the *pre-modification* value stored by the
    earliest-in-edit-order edit of r whose timestamp is strictly greater than
    t, or zero when no such edit exists. (Edit timestamps are the replay-
    extended *end* times; ``generate_array_states``'s docstring guarantees they
    are non-decreasing along every row, and the stable sort keeps edit order
    within equal timestamps.)
    """
    D = edits[0]["values"].shape[1]
    out = torch.zeros((times.shape[0], rows.shape[0], D))
    for j, r in enumerate(rows.tolist()):
        row_edits = [
            (e["timestamp"], e["values"][(e["indexes"] == r).nonzero().view(-1)[0]])
            for e in edits
            if bool((e["indexes"] == r).any())
        ]
        for i, t in enumerate(times.tolist()):
            for ts, value in row_edits:
                if ts > t:
                    out[i, j] = value
                    break
    return out


def run_query(times, N, edits, rows, disabled):
    tl._OPT_DISABLED = frozenset({"rowdedup"} if disabled else ())
    prepared = tl._prepare_array_state_queries(times, N, edits)
    return tl._query_row_states(times, prepared, rows)


def classify(times, N, edits, rows):
    """Replicate the endpoint test to report which branch a case exercises."""
    prepared = tl._prepare_array_state_queries(times, N, edits)
    n_ranks = prepared.unique_timestamps.shape[0]
    bases = (torch.arange(N) if rows is None else rows) * n_ranks
    ranks = torch.unique(
        torch.searchsorted(prepared.unique_timestamps, times.contiguous(), right=True)
    )
    S, R = int(ranks.shape[0]), int(bases.shape[0])
    if S <= 1:
        return "skip (S==1)", 0, R, S
    low_lo = torch.searchsorted(prepared.keys, bases + ranks.amin())
    low_hi = torch.searchsorted(prepared.keys, bases + ranks.amax())
    chg = int((low_lo != low_hi).sum())
    if chg * S > R * (S - 2):
        return "bail (mostly changing)", chg, R, S
    return "dedup", chg, R, S


def block_edits(N, n_mobs, times_per_mob, D, seed):
    """Edits authored in blocks, the way a scene records them: each mob owns a
    contiguous row range and gets a few edits at its own staggered times.
    """
    g = torch.Generator().manual_seed(seed)
    edits = []
    per = N // n_mobs
    t = 0.0
    for m in range(n_mobs):
        rows = torch.arange(m * per, (m + 1) * per, dtype=torch.int64)
        for _ in range(times_per_mob):
            t += float(torch.rand((), generator=g)) * 2.0 + 0.05
            edits.append(
                {
                    "indexes": rows,
                    "values": torch.randn((rows.shape[0], D), generator=g),
                    "timestamp": t,
                }
            )
    return edits, t


def cases():
    N, D = 4000, 3
    edits, t_end = block_edits(N, n_mobs=40, times_per_mob=3, D=D, seed=0)

    # A window over a narrow slice of the schedule: most mobs' boundaries lie
    # outside it, a few inside -- the render-batch shape the dedup is for.
    yield "blocks/mixed", torch.linspace(t_end * 0.30, t_end * 0.36, 48), N, edits, None
    # The same window through a scattered row subset (the compact-buffer path).
    subset = torch.unique(
        torch.randint(0, N, (320,), generator=torch.Generator().manual_seed(1))
    )
    yield (
        "blocks/rows-subset",
        torch.linspace(t_end * 0.30, t_end * 0.36, 48),
        N,
        edits,
        subset,
    )
    # A window past every edit: every row reads zero through the empty mask.
    yield (
        "blocks/all-past-end",
        torch.linspace(t_end + 1.0, t_end + 2.0, 16),
        N,
        edits,
        None,
    )
    # Unsorted, all-distinct times: S == T, inverse is None, endpoints must
    # come from amin/amax rather than the ends of a sorted rank array.
    g = torch.Generator().manual_seed(2)
    shuffled = t_end * 0.30 + torch.rand(37, generator=g) * t_end * 0.06
    yield "blocks/unsorted", shuffled, N, edits, None
    # One- and two-rank windows: the S guard and the S==2 wash case.
    yield "blocks/one-rank", torch.full((9,), t_end * 0.5), N, edits, None
    first_ts = min(e["timestamp"] for e in edits)
    yield (
        "blocks/two-ranks",
        torch.tensor([first_ts - 0.01, first_ts + 0.01]),
        N,
        edits,
        None,
    )

    # Every row changes inside the window: boundary-heavy, must take the bail.
    g = torch.Generator().manual_seed(3)
    hot = [
        {
            "indexes": torch.tensor([r]),
            "values": torch.randn((1, D), generator=g),
            "timestamp": 1.0 + float(torch.rand((), generator=g)),
        }
        for r in range(600)
    ]
    yield "hot/every-row-changes", torch.linspace(0.9, 2.1, 24), 600, hot, None

    # One scene-length edit over all rows: everything constant, zero changing.
    one = [
        {
            "indexes": torch.arange(1000, dtype=torch.int64),
            "values": torch.randn(
                (1000, D), generator=torch.Generator().manual_seed(4)
            ),
            "timestamp": 100.0,
        }
    ]
    yield "one-long/all-const", torch.linspace(0.0, 5.0, 32), 1000, one, None

    # Rows with no edits at all interleaved with edited rows, D=1.
    sparse = [
        {
            "indexes": torch.arange(0, 500, 7, dtype=torch.int64),
            "values": torch.randn(
                (len(range(0, 500, 7)), 1), generator=torch.Generator().manual_seed(5)
            ),
            "timestamp": 2.0,
        },
        {
            "indexes": torch.arange(0, 500, 13, dtype=torch.int64),
            "values": torch.randn(
                (len(range(0, 500, 13)), 1), generator=torch.Generator().manual_seed(6)
            ),
            "timestamp": 3.5,
        },
    ]
    yield "sparse/unedited-rows", torch.linspace(1.0, 4.0, 20), 500, sparse, None


def main():
    branches_seen = set()
    for name, times, N, edits, rows in cases():
        branch, chg, R, S = classify(times, N, edits, rows)
        branches_seen.add(branch.split(" ")[0])
        dense = run_query(times, N, edits, rows, disabled=True)
        dedup = run_query(times, N, edits, rows, disabled=False)
        assert torch.equal(dense, dedup), (
            f"{name}: dedup differs from dense path "
            f"({int((dense != dedup).any(-1).sum())} (time, row) cells)"
        )
        check_rows = rows if rows is not None else torch.arange(min(N, 400))
        reference = brute_force(times, N, edits, check_rows)
        got = (
            dedup[:, : check_rows.shape[0]]
            if rows is not None
            else dedup[:, check_rows]
        )
        assert torch.equal(got, reference), (
            f"{name}: torch paths differ from brute force"
        )
        print(f"  {name:<26} {branch:<24} S={S:<3} R={R:<5} changing={chg}")

    # Vacuity guard: the sweep must have exercised the fast path, the bail and
    # the S==1 skip, or this compared the dense path against itself.
    missing = {"dedup", "bail", "skip"} - branches_seen
    assert not missing, f"branches never exercised: {missing}"
    print("\nrow-dedup parity holds (dense == dedup == brute force on every case)")


if __name__ == "__main__":
    main()
