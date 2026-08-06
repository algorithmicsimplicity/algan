"""The attribute-timeline state query: per-row "which edit is live at t".

Answered by one flat ``torch.searchsorted`` over the composite
``row * n_ranks + rank(timestamp)`` key built by
``_prepare_array_state_queries`` (see ``EditQueryIndex``). Byte-identity
against the Taichi kernels this replaced is checked by
``benchmarks/_timeline_query_parity.py``; these tests pin the semantics
without needing a GPU.
"""

import torch

from algan.animation_timeline.timeline import (
    _prepare_array_state_queries,
    _query_row_states,
    generate_array_states,
)


def _make_edits(n_rows, n_edits, channels, *, seed=0):
    """An edit log shaped like ``prepare_for_queries`` output: execution order,
    non-decreasing timestamps (with ties), terminated by the all-rows ``inf``
    edit carrying the current state.
    """
    g = torch.Generator().manual_seed(seed)
    edits = []
    timestamp = 0.0
    for i in range(n_edits):
        begin = int(torch.randint(0, n_rows, (1,), generator=g))
        end = min(n_rows, begin + int(torch.randint(1, 9, (1,), generator=g)))
        if i % 3:  # some edits share an end time
            timestamp += float(torch.rand(1, generator=g))
        edits.append(
            {
                "indexes": torch.arange(begin, end),
                "values": torch.randn(end - begin, channels, generator=g),
                "timestamp": timestamp,
            }
        )
    edits.append(
        {
            "indexes": torch.arange(n_rows),
            "values": torch.randn(n_rows, channels, generator=g),
            "timestamp": float("inf"),
        }
    )
    return edits, timestamp


def _reference(times, n_rows, edits):
    """The live edit for a row at time t is the first edit touching that row
    (in execution order) whose timestamp is > t; rows with none read as zero.
    """
    channels = edits[0]["values"].shape[1]
    out = torch.zeros(times.shape[0], n_rows, channels)
    for time_index, t in enumerate(times.tolist()):
        for row in range(n_rows):
            for edit in edits:
                hits = (edit["indexes"] == row).nonzero()
                if hits.numel() and edit["timestamp"] > t:
                    out[time_index, row] = edit["values"][int(hits[0])]
                    break
    return out


def test_query_matches_brute_force_reference():
    n_rows = 24
    edits, last = _make_edits(n_rows, 15, 3, seed=3)
    times = torch.linspace(-0.5, last + 0.5, 9)

    actual = generate_array_states(times, n_rows, edits)

    assert torch.equal(actual, _reference(times, n_rows, edits))


def test_selected_rows_match_the_full_query_and_leave_the_rest_zero():
    n_rows = 24
    edits, last = _make_edits(n_rows, 15, 4, seed=4)
    times = torch.linspace(0.0, last, 7)
    rows = torch.tensor([1, 2, 3, 17, 23])

    full = generate_array_states(times, n_rows, edits)
    selected = generate_array_states(times, n_rows, edits, active_rows=rows)

    assert torch.equal(selected[:, rows], full[:, rows])
    unselected = torch.ones(n_rows, dtype=torch.bool)
    unselected[rows] = False
    assert not selected[:, unselected].any()


def test_rows_with_no_live_edit_read_as_zero_without_nan():
    # A row is queried before anything has edited it, and the recorded values
    # are non-finite: masking (not multiplying) keeps the untouched rows at a
    # clean zero instead of inf * 0 == NaN.
    edits = [
        {
            "indexes": torch.tensor([1]),
            "values": torch.tensor([[float("inf"), float("-inf")]]),
            "timestamp": 1.0,
        },
        {
            "indexes": torch.arange(3),
            "values": torch.zeros(3, 2),
            "timestamp": float("inf"),
        },
    ]
    times = torch.tensor([0.0, 2.0])

    actual = generate_array_states(times, 3, edits)

    assert not actual.isnan().any()
    assert torch.equal(actual[0, 1], torch.tensor([float("inf"), float("-inf")]))
    assert torch.equal(actual[1, 1], torch.zeros(2))


def test_query_is_chunked_without_changing_the_result(monkeypatch):
    # Long frame windows are split so the temporary key buffer stays bounded;
    # the split must not be observable.
    n_rows = 12
    edits, last = _make_edits(n_rows, 9, 2, seed=5)
    times = torch.linspace(0.0, last, 16)
    prepared = _prepare_array_state_queries(times, n_rows, edits)

    unchunked = _query_row_states(times, prepared)
    monkeypatch.setattr(
        "algan.animation_timeline.timeline._QUERY_CHUNK_BYTES", 1, raising=True
    )
    chunked = _query_row_states(times, prepared)

    assert torch.equal(chunked, unchunked)


def test_empty_selection_and_single_frame_windows():
    n_rows = 10
    edits, _ = _make_edits(n_rows, 6, 3, seed=6)

    empty = generate_array_states(
        torch.tensor([0.5]), n_rows, edits, active_rows=torch.zeros(0, dtype=torch.long)
    )

    assert empty.shape == (1, n_rows, 3)
    assert not empty.any()
