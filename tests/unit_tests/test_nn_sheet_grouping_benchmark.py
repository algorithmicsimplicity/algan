"""Warm benchmark accounting must not turn noisy timings into a speedup claim."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.setenv("ALGAN_USE_DAEMON", "0")
    return importlib.import_module("benchmarks.performance.nn_sheet_grouping_ab")


@pytest.mark.parametrize("sequence", ["AB", "BA", "ABBA", "ABBABAAB"])
def test_valid_sequences_pair_control_and_candidate(benchmark, sequence):
    benchmark._validate_sequence(sequence)


@pytest.mark.parametrize("sequence", ["", "A", "ABX", "AA", "AABB", "ABCX"])
def test_invalid_sequences_are_rejected(benchmark, sequence):
    with pytest.raises(ValueError, match="AB or BA pairs"):
        benchmark._validate_sequence(sequence)


def _row(arm, seconds, phase="measured"):
    return {"grouping": {"arm": arm}, "seconds": seconds, "phase": phase}


def test_summary_excludes_warmup_and_parity(benchmark):
    rows = [
        _row("A", 999, "warmup"),
        _row("B", 1000, "warmup"),
        _row("A", 10),
        _row("B", 8),
        _row("B", 12),
        _row("A", 20),
        _row("A", 2000, "parity_unmeasured"),
        _row("B", 3000, "parity_unmeasured"),
    ]
    result = benchmark._timing_summary(rows)
    assert result["A"]["seconds"] == [10, 20]
    assert result["B"]["seconds"] == [8, 12]
    assert result["A"]["mean"] == result["A"]["median"] == 15
    assert result["B"]["mean"] == result["B"]["median"] == 10
    assert result["mean_reduction_percent"] == pytest.approx(100 / 3)
    assert [pair["ratio_b_over_a"] for pair in result["adjacent_pairs"]] == [0.8, 0.6]


def test_summary_keeps_disagreeing_mean_and_median(benchmark):
    result = benchmark._timing_summary(
        [_row(arm, value) for arm, value in zip("ABBAAB", [10, 20, 20, 100, 100, 200])]
    )
    assert result["median_reduction_percent"] == 80
    assert result["mean_reduction_percent"] < 0
    assert result["A"]["minimum"] == 10
    assert result["B"]["maximum"] == 200


def test_empty_summary_has_no_speedup_claim(benchmark):
    assert benchmark._timing_summary([]) == {"A": {"seconds": []}, "B": {"seconds": []}}


def test_default_order_balances_positions_and_pairs(benchmark):
    sequence = "ABBABAAB"
    benchmark._validate_sequence(sequence)
    assert sum(i for i, arm in enumerate(sequence) if arm == "A") == sum(
        i for i, arm in enumerate(sequence) if arm == "B"
    )
