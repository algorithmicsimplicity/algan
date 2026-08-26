"""Rank-deduplicated Taichi kernels for the attribute-timeline state query.

The kernels in ``algan/animation_timeline/utils_taichi.py`` answer the query the
literal way: one binary search per ``(frame, row)`` pair over the whole edit
table. The torch implementation that replaced them
(:func:`algan.animation_timeline.timeline._query_row_states`) does strictly less
work, in two independent ways:

1. **Rank dedup.** A frame's answer depends on its time only through
   ``rank = number of distinct edit timestamps <= t``. Frames sharing a rank
   share an answer, so the search runs once per *distinct* rank ``S <= T`` and
   the result is expanded back over the frames.
2. **Row-constancy dedup.** A row whose search lands on the same key at the
   window's lowest and highest rank is constant across the window, so it is
   gathered once and broadcast instead of searched per rank.

Comparing the old kernels against that is comparing two different algorithms.
These kernels do the same work the torch path does, so the remaining difference
is Taichi-vs-torch rather than search-count:

``query_ranked_compact``
    The faithful mirror: one search per ``(distinct rank, row)``, writing the
    compact ``[S, R, D]`` result the caller expands with ``index_select``.

``query_ranked_placed``
    The formulation a kernel makes available and torch does not. Ranks arrive
    sorted, so a row's landing position is monotone in the rank: seek once with
    a binary search and *walk* the row's remaining ranks, which costs
    ``O(row_length + S)`` instead of ``O(S log U)`` and subsumes the
    row-constancy special case (a constant row simply never advances). It also
    writes straight into the ``[T, N, D]`` result at the row's global position,
    fusing the expand and the scatter that the torch path pays as two extra
    full-size passes.

Both restrict the binary search to the row's own CSR segment
``[head[j], head[j + 1])``. That is exactly equivalent to torch's search over
the whole globally-sorted ``keys`` -- the composite key ``j * n_ranks + rank``
can only land inside row ``j``'s segment or on its end -- but torch cannot
express it, because one fused ``searchsorted`` has to see the whole array.

Output is byte-identical to the torch path by construction: equal landing
positions select equal rows of ``sorted_values``, and rows with no live edit are
written as zeros rather than multiplied by a mask, so a recorded ``inf``/``NaN``
cannot leak.
"""

import taichi as ti


@ti.func
def _lower_bound(keys: ti.template(), lo, hi, target):
    """First index in ``[lo, hi)`` whose key is >= ``target``.

    Matches ``torch.searchsorted(..., right=False)`` on the row's segment.
    ``keys`` is ``ti.template()`` because a ``@ti.func`` takes an ndarray only
    as a template argument -- a plain annotation is a scalar-by-value binding.
    """
    low = lo
    high = hi
    while low < high:
        mid = (low + high) // 2
        if keys[mid] < target:
            low = mid + 1
        else:
            high = mid
    return low


@ti.kernel
def query_ranked_compact(
        ranks: ti.types.ndarray(),  # [S] (int64) distinct query ranks, ascending
        rows: ti.types.ndarray(),  # [R] (int64) global row ids
        head: ti.types.ndarray(),  # [N + 1] (int64) CSR row boundaries
        keys: ti.types.ndarray(),  # [U] (int64) row * n_ranks + rank(timestamp)
        sorted_values: ti.types.ndarray(),  # [U, D]
        out: ti.types.ndarray(),  # [S, R, D]
        n_ranks: ti.i64,
):
    """One search per (distinct rank, row), into the compact result."""
    D = sorted_values.shape[1]
    for s, r in ti.ndrange(ranks.shape[0], rows.shape[0]):
        j = rows[r]
        start = head[j]
        end = head[j + 1]
        low = _lower_bound(keys, start, end, j * n_ranks + ranks[s])
        if low < end:
            for d in range(D):
                out[s, r, d] = sorted_values[low, d]
        else:
            for d in range(D):
                out[s, r, d] = 0.0


@ti.kernel
def query_ranked_placed(
        ranks: ti.types.ndarray(),  # [S] (int64) distinct query ranks, ascending
        frame_head: ti.types.ndarray(),  # [S + 1] (int64) frames-per-rank CSR
        frame_ids: ti.types.ndarray(),  # [T] (int64) frame ids grouped by rank
        rows: ti.types.ndarray(),  # [R] (int64) global row ids
        head: ti.types.ndarray(),  # [N + 1] (int64) CSR row boundaries
        keys: ti.types.ndarray(),  # [U] (int64)
        sorted_values: ti.types.ndarray(),  # [U, D]
        out: ti.types.ndarray(),  # [T, N, D], zeroed by the caller
        n_ranks: ti.i64,
):
    """Monotone walk per row, written straight into the placed result.

    Parallel over rows: each row seeks once and then advances a cursor through
    its own edits as the rank increases, so the inner loop never re-searches.
    Frames are grouped by rank (``frame_head`` / ``frame_ids``) rather than
    assumed sorted, so unordered or repeated query times stay correct.
    """
    D = sorted_values.shape[1]
    S = ranks.shape[0]
    for r in range(rows.shape[0]):
        j = rows[r]
        start = head[j]
        end = head[j + 1]
        base = j * n_ranks
        # Seek to the first rank, then never look backwards: ranks ascend and
        # so does the landing position within the row.
        low = _lower_bound(keys, start, end, base + ranks[0])
        for s in range(S):
            target = base + ranks[s]
            while low < end:
                if keys[low] >= target:
                    break
                low += 1
            for m in range(frame_head[s], frame_head[s + 1]):
                t = frame_ids[m]
                if low < end:
                    for d in range(D):
                        out[t, j, d] = sorted_values[low, d]
                else:
                    for d in range(D):
                        out[t, j, d] = 0.0


@ti.kernel
def query_ranked_walk(
        ranks: ti.types.ndarray(),  # [S] (int64) distinct query ranks, ascending
        rows: ti.types.ndarray(),  # [R] (int64) global row ids
        head: ti.types.ndarray(),  # [N + 1] (int64)
        keys: ti.types.ndarray(),  # [U] (int64)
        low: ti.types.ndarray(),  # [R, S] (int32) out: landing index, -1 if none
        n_ranks: ti.i64,
):
    """Phase 1 of the split form: the monotone walk, landing indices only.

    Writes ``[R, S]`` -- a few MB against the result's hundreds -- so the walk
    keeps its ``O(row_length + S)`` search cost without inheriting
    :func:`query_ranked_placed`'s access pattern. ``int32`` is safe: ``low``
    indexes the edit table, whose length is bounded by total recorded edit-rows.
    """
    S = ranks.shape[0]
    for r in range(rows.shape[0]):
        j = rows[r]
        start = head[j]
        end = head[j + 1]
        base = j * n_ranks
        pos = _lower_bound(keys, start, end, base + ranks[0])
        for s in range(S):
            target = base + ranks[s]
            while pos < end:
                if keys[pos] >= target:
                    break
                pos += 1
            # Explicit narrowing: pos is i64 (it indexes the i64 key array),
            # low is i32 (see the docstring on why that is safe).
            if pos < end:
                low[r, s] = ti.cast(pos, ti.i32)
            else:
                low[r, s] = -1


@ti.kernel
def query_ranked_scatter(
        frame_rank: ti.types.ndarray(),  # [T] (int32) rank slot for each frame
        rows: ti.types.ndarray(),  # [R] (int64) global row ids
        low: ti.types.ndarray(),  # [R, S] (int32) from query_ranked_walk
        sorted_values: ti.types.ndarray(),  # [U, D]
        out: ti.types.ndarray(),  # [T, N, D], zeroed by the caller
):
    """Phase 2 of the split form: expand over frames and place, in one pass.

    Parallel over ``(frame, row)`` with the row innermost, so consecutive
    threads write consecutive ``out[t, j, :]`` -- the sequential pattern
    ``index_select`` has. Fusing the expand and the placement means the values
    are gathered once and stored once, where the torch path materializes the
    compact result, expands it into a second full-size buffer, and copies that
    into the third.
    """
    D = sorted_values.shape[1]
    for t, r in ti.ndrange(frame_rank.shape[0], rows.shape[0]):
        j = rows[r]
        source = low[r, frame_rank[t]]
        if source >= 0:
            for d in range(D):
                out[t, j, d] = sorted_values[source, d]
        else:
            for d in range(D):
                out[t, j, d] = 0.0
