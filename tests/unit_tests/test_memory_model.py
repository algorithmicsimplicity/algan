"""Contracts of the runtime chunk-memory model.

The model replaces hand-written byte formulas with a line fitted to what the
arena actually reached. These pin the properties batch sizing depends on --
especially the ones whose failure modes are silent: a planner that never
escapes probing, or a line drawn so optimistically that the first full-size
chunk over-commits.
"""

import pytest

from algan.rendering.memory_model import (
    DEFAULT_SAFETY,
    HISTORY,
    PROBE_GROWTH,
    PROBE_SAFETY,
    ChunkMemoryModel,
    PeakRatioModel,
    chunk_signature,
)

SIG = ("sig",)
GB = 1 << 30


def test_uncalibrated_model_probes_a_single_frame():
    model = ChunkMemoryModel()
    assert model.predict(SIG, 4) is None
    assert model.plan(SIG, 100, GB) == 1


def test_one_observation_is_enough_to_escape_probing():
    # Requiring two samples deadlocks: planning would pin every chunk to one
    # frame, so a second frame count would never be observed.
    model = ChunkMemoryModel()
    model.observe(SIG, 1, 10_000_000)
    assert model.plan(SIG, 100, GB) > 1


def test_growth_is_bounded_while_evidence_is_thin():
    model = ChunkMemoryModel()
    model.observe(SIG, 1, 1_000)
    # Memory is effectively unlimited, so only the growth ceiling constrains it.
    assert model.plan(SIG, 10_000, 1 << 50) == PROBE_GROWTH
    model.observe(SIG, PROBE_GROWTH, 1_000 * PROBE_GROWTH)
    assert model.plan(SIG, 10_000, 1 << 50) == PROBE_GROWTH * PROBE_GROWTH


def test_single_sample_planning_uses_the_wider_margin():
    model = ChunkMemoryModel()
    model.observe(SIG, 2, 2_000_000)
    lone = model.predict(SIG, 2)
    model.observe(SIG, 4, 4_000_000)
    paired = model.predict(SIG, 2)
    # Same line, but the one-sample case reserves more.
    assert lone > paired
    assert PROBE_SAFETY > DEFAULT_SAFETY


def test_line_is_fitted_from_the_largest_chunks():
    # The first chunk of a job runs cheaper than steady state. Fitting through
    # it tilts the slope down, which is the direction that over-commits.
    model = ChunkMemoryModel()
    model.observe(SIG, 1, 1_000_000)      # unrepresentatively cheap
    model.observe(SIG, 10, 20_000_000)
    model.observe(SIG, 20, 40_000_000)
    intercept, slope = model._line(SIG)
    assert slope == pytest.approx(2_000_000)


def test_recent_observations_take_the_worst_peak_at_each_size():
    model = ChunkMemoryModel()
    model.observe(SIG, 4, 1_000_000)
    model.observe(SIG, 4, 3_000_000)
    model.observe(SIG, 4, 2_000_000)
    assert model._samples(SIG)[4] == 3_000_000


def test_a_dense_chunk_ages_out_instead_of_handicapping_the_render():
    # One unusually heavy chunk must not shrink every later batch for the rest
    # of the job. It raises the line while it is in the window and is forgotten
    # afterwards.
    model = ChunkMemoryModel()
    for _ in range(HISTORY):
        model.observe(SIG, 10, 10_000_000)
    steady = model.predict(SIG, 10)

    model.observe(SIG, 10, 90_000_000)
    assert model.predict(SIG, 10) > steady

    # Later chunks of ordinary weight push the spike out of the window.
    for _ in range(HISTORY):
        model.observe(SIG, 10, 10_000_000)
    assert model.predict(SIG, 10) == steady


def test_the_window_is_bounded():
    model = ChunkMemoryModel()
    for index in range(HISTORY * 4):
        model.observe(SIG, 1 + index, 1_000)
    assert len(model._by_signature[SIG]) == HISTORY


def test_prediction_is_monotone_in_the_frame_count():
    # render_loop binary-searches durations elsewhere and the retry path halves
    # them; a non-monotone predictor makes both incoherent.
    model = ChunkMemoryModel()
    model.observe(SIG, 2, 2_000_000)
    model.observe(SIG, 9, 9_000_000)
    previous = -1
    for frames in range(1, 200):
        current = model.predict(SIG, frames)
        assert current >= previous
        previous = current


def test_plan_never_exceeds_what_predict_says_fits():
    model = ChunkMemoryModel()
    model.observe(SIG, 4, 40_000_000)
    model.observe(SIG, 16, 160_000_000)
    budget = 500_000_000
    planned = model.plan(SIG, 10_000, budget)
    assert model.predict(SIG, planned) <= budget
    assert model.predict(SIG, planned + 1) > budget


def test_plan_returns_one_when_nothing_fits():
    model = ChunkMemoryModel()
    model.observe(SIG, 1, 10 * GB)
    assert model.plan(SIG, 100, 1 << 20) == 1


def test_a_flat_measurement_falls_back_to_per_frame_cost():
    # If the peak does not grow with frames the slope is meaningless; charging
    # the whole peak per frame is the safe reading.
    model = ChunkMemoryModel()
    model.observe(SIG, 4, 8_000_000)
    model.observe(SIG, 8, 8_000_000)
    _intercept, slope = model._line(SIG)
    assert slope > 0


def test_signatures_separate_batches_on_different_lines():
    small = chunk_signature(
        width=64, height=64, channels=4, dtype="torch.float32",
        samples_per_pixel=1, num_triangles=10, num_circuits=0)
    wide = chunk_signature(
        width=1920, height=64, channels=4, dtype="torch.float32",
        samples_per_pixel=1, num_triangles=10, num_circuits=0)
    byte_buffer = chunk_signature(
        width=64, height=64, channels=4, dtype="torch.uint8",
        samples_per_pixel=1, num_triangles=10, num_circuits=0)
    assert small != wide
    assert small != byte_buffer
    # Ordinary geometry drift keeps the fit; an order of magnitude starts a new
    # one, so a scene that grows steadily does not discard usable evidence.
    nudged = chunk_signature(
        width=64, height=64, channels=4, dtype="torch.float32",
        samples_per_pixel=1, num_triangles=11, num_circuits=0)
    exploded = chunk_signature(
        width=64, height=64, channels=4, dtype="torch.float32",
        samples_per_pixel=1, num_triangles=10_000, num_circuits=0)
    assert small == nudged
    assert small != exploded


def test_peak_ratio_uses_the_seed_until_something_is_measured():
    ratio = PeakRatioModel(seed=6.0)
    assert not ratio.is_calibrated()
    assert ratio.factor() == 6.0


def test_peak_ratio_supersedes_the_seed_once_measured():
    # The seeded guesses were far off: the merge's real ratio measures well
    # under 1x against inputs the guess multiplied by six.
    ratio = PeakRatioModel(seed=6.0, safety=1.25)
    ratio.observe(1_000_000, 470_000)
    assert ratio.is_calibrated()
    assert ratio.factor() < 6.0


def test_peak_ratio_never_drops_below_one():
    # A build cannot peak below the inputs it has already materialised, and a
    # sub-1x multiplier would under-reserve on a batch whose inputs are not
    # already resident.
    ratio = PeakRatioModel(seed=6.0, safety=1.25)
    ratio.observe(1_000_000, 10_000)
    assert ratio.factor() >= 1.0


def test_peak_ratio_forgets_a_heavy_build():
    ratio = PeakRatioModel(seed=2.0, safety=1.0)
    for _ in range(HISTORY):
        ratio.observe(1_000, 1_000)
    steady = ratio.factor()
    ratio.observe(1_000, 9_000)
    assert ratio.factor() > steady
    for _ in range(HISTORY):
        ratio.observe(1_000, 1_000)
    assert ratio.factor() == steady


def test_peak_ratio_ignores_degenerate_samples():
    ratio = PeakRatioModel(seed=3.0)
    ratio.observe(0, 5_000_000)
    assert not ratio.is_calibrated()
    assert ratio.factor() == 3.0


def test_model_is_isolated_per_signature():
    model = ChunkMemoryModel()
    model.observe(("a",), 1, 1_000_000)
    assert model.plan(("b",), 100, GB) == 1
