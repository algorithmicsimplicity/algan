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
    AffineFrameCost,
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
    model.observe(SIG, 1, 1_000_000)  # unrepresentatively cheap
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


def test_capacity_bound_chunks_never_price_one_frame_at_the_whole_arena():
    # Chunks are planned to fill the arena, so batches of different density --
    # which share a signature, since geometry counts are bucketed -- both peak
    # just under capacity. A line through two nearly equal peaks at different
    # frame counts is almost flat, with an intercept that swallows the arena.
    # Read literally it says a *single* frame needs more memory than the whole
    # arena, which pins every chunk to one frame and makes render_loop's
    # preflight reject batches until it gives up with OutOfRenderMemory.
    arena = 1_204_000_000
    model = ChunkMemoryModel()
    model.observe(SIG, 1, 70_000_000)
    model.observe(SIG, 8, 560_000_000)
    model.observe(SIG, 15, 1_190_000_000)
    model.observe(SIG, 20, 1_203_000_000)

    intercept, _slope = model._line(SIG)
    # The intercept is the frame-independent cost; a chunk that was measured in
    # full bounds it, so it can never exceed the smallest measured peak.
    assert intercept <= min(model._samples(SIG).values())
    assert model.predict(SIG, 1) < arena
    assert model.plan(SIG, 100, arena) > 1
    # Still conservative: no sample is read as cheaper than it measured.
    assert model.predict(SIG, 15) >= 1_190_000_000


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
        width=64,
        height=64,
        channels=4,
        dtype="torch.float32",
        samples_per_pixel=1,
        num_triangles=10,
        num_circuits=0,
    )
    wide = chunk_signature(
        width=1920,
        height=64,
        channels=4,
        dtype="torch.float32",
        samples_per_pixel=1,
        num_triangles=10,
        num_circuits=0,
    )
    byte_buffer = chunk_signature(
        width=64,
        height=64,
        channels=4,
        dtype="torch.uint8",
        samples_per_pixel=1,
        num_triangles=10,
        num_circuits=0,
    )
    assert small != wide
    assert small != byte_buffer
    # Ordinary geometry drift keeps the fit; an order of magnitude starts a new
    # one, so a scene that grows steadily does not discard usable evidence.
    nudged = chunk_signature(
        width=64,
        height=64,
        channels=4,
        dtype="torch.float32",
        samples_per_pixel=1,
        num_triangles=11,
        num_circuits=0,
    )
    exploded = chunk_signature(
        width=64,
        height=64,
        channels=4,
        dtype="torch.float32",
        samples_per_pixel=1,
        num_triangles=10_000,
        num_circuits=0,
    )
    assert small == nudged
    assert small != exploded


def test_affine_frame_cost_reads_one_sample_as_a_pure_per_frame_cost():
    cost = AffineFrameCost()
    assert cost.max_frames_for() is None
    assert cost.actor_share() is None

    cost.observe(10, 100, budget_bytes=250)
    assert cost.fixed_bytes() == 0.0
    assert cost.max_frames_for() == 25
    # One sample cannot separate the two parts at all.
    assert cost.actor_share() is None


def test_affine_frame_cost_separates_the_actor_set_from_the_frames():
    # 900 units of actor set + 1 per frame. Read as if it all scaled, 10 frames
    # costing 910 of a 1000 budget looks like room for 10 more; in fact there
    # is room for 90.
    cost = AffineFrameCost()
    cost.observe(10, 910, budget_bytes=1_000)
    cost.observe(20, 920, budget_bytes=1_000)

    assert cost.fixed_bytes() == pytest.approx(900.0)
    assert cost.max_frames_for() == 100
    # 900 fixed of the 920 spent at the widest window measured.
    assert cost.actor_share() == pytest.approx(900 / 920)


def test_affine_frame_cost_reports_zero_when_no_window_is_short_enough():
    # The fixed part alone overruns the budget: shortening the window cannot
    # help, and the caller has to build a batch with fewer actors instead.
    cost = AffineFrameCost()
    cost.observe(10, 1_100, budget_bytes=1_000)
    cost.observe(20, 1_200, budget_bytes=1_000)

    assert cost.max_frames_for() == 0
    assert cost.actor_share() == pytest.approx(1000 / 1200)


def test_affine_frame_cost_intercept_cannot_exceed_a_measured_cost():
    # Same guard as the other fits: a batch measured in full below the claimed
    # fixed cost disproves it.
    cost = AffineFrameCost()
    cost.observe(100, 1_000, budget_bytes=10_000)
    cost.observe(101, 1_000_000, budget_bytes=10_000)
    assert cost.fixed_bytes() <= 1_000


def test_affine_frame_cost_keeps_the_worst_cost_seen_at_a_frame_count():
    cost = AffineFrameCost()
    cost.observe(10, 100, budget_bytes=1_000)
    cost.observe(10, 300, budget_bytes=1_000)
    assert cost.max_frames_for() == 33


def test_peak_ratio_uses_the_seed_until_something_is_measured():
    ratio = PeakRatioModel(seed=6.0)
    assert not ratio.is_calibrated()
    assert ratio.predict(1_000_000) == 6_000_000


def test_peak_ratio_supersedes_the_seed_once_measured():
    # The seeded guesses were far off: the merge's real peak measures well
    # under the inputs the guess multiplied by six.
    ratio = PeakRatioModel(seed=6.0, safety=1.25)
    ratio.observe(1_000_000, 470_000)
    assert ratio.is_calibrated()
    assert ratio.predict(1_000_000) < 6_000_000


def test_peak_ratio_never_drops_below_the_inputs():
    # A build cannot peak below the inputs it has already materialised, and a
    # sub-1x bound would under-reserve on a batch whose inputs are not already
    # resident.
    ratio = PeakRatioModel(seed=6.0, safety=1.25)
    ratio.observe(1_000_000, 10_000)
    assert ratio.predict(1_000_000) >= 1_000_000


def test_peak_ratio_forgets_a_heavy_build():
    ratio = PeakRatioModel(seed=2.0, safety=1.0)
    for _ in range(HISTORY):
        ratio.observe(1_000, 1_000)
    steady = ratio.predict(10_000)
    ratio.observe(1_000, 9_000)
    assert ratio.predict(10_000) > steady
    for _ in range(HISTORY):
        ratio.observe(1_000, 1_000)
    assert ratio.predict(10_000) == steady


def test_peak_ratio_ignores_degenerate_samples():
    ratio = PeakRatioModel(seed=3.0)
    ratio.observe(0, 5_000_000)
    assert not ratio.is_calibrated()
    assert ratio.predict(1_000_000) == 3_000_000


def test_peak_ratio_does_not_charge_a_small_builds_fixed_cost_per_byte():
    # A job's first merge is typically its smallest, and it pays the whole
    # fixed cost (kernel workspace, allocator growth) on tiny inputs. Read as a
    # pure ratio that measured over 20x, and it then throttled every batch for
    # the rest of the window to a twentieth of the headroom it really had --
    # which is exactly what a real render was doing.
    ratio = PeakRatioModel(seed=6.0, safety=1.25)
    ratio.observe(10_000_000, 217_000_000)  # 21.7x, nearly all fixed
    ratio.observe(56_000_000, 100_000_000)  # 1.8x once the fixed part is paid

    big = 1_000_000_000
    assert ratio.predict(big) < 4 * big
    # The fixed part is still carried: it is charged once, not dropped.
    assert ratio.predict(big) > 1.8 * big


def test_peak_ratio_separates_the_fixed_part_from_the_rate():
    ratio = PeakRatioModel(seed=6.0, safety=1.0)
    # peak = 100 MB + 2 * inputs.
    ratio.observe(50_000_000, 200_000_000)
    ratio.observe(100_000_000, 300_000_000)
    assert ratio.predict(200_000_000) == pytest.approx(500_000_000, rel=1e-6)


def test_peak_ratio_reads_a_budget_back_as_an_input_size():
    # Callers size frame windows, and input bytes scale with frames while the
    # fixed part does not. Dividing a budget by the whole prediction would pin
    # a window at whatever size it already had, however much room the fixed
    # part left for more frames.
    ratio = PeakRatioModel(seed=6.0, safety=1.0)
    # peak = 900 MB + 1 * inputs.
    ratio.observe(100_000_000, 1_000_000_000)
    ratio.observe(200_000_000, 1_100_000_000)

    assert ratio.max_inputs_for(1_500_000_000) == pytest.approx(600_000_000, rel=1e-6)
    # A budget below the fitted fixed part still admits what the rate the
    # builds actually demonstrated allows -- reporting zero there is how a
    # render ends up rejecting every window however short.
    assert ratio.max_inputs_for(500_000_000) == pytest.approx(50_000_000, rel=1e-6)


def test_peak_ratio_never_reads_worse_than_the_plain_ratio():
    # The affine fit and the pure ratio fail in opposite directions, and both
    # have pinned a render to single frames: a ratio charges a small build's
    # fixed cost to every byte, while an intercept fitted from a run of large
    # builds approaches the headroom and rejects every window however short.
    # The fit may only improve on the ratio, never inflate it.
    ratio = PeakRatioModel(seed=6.0, safety=1.0)
    ratio.observe(1_000_000_000, 1_700_000_000)
    ratio.observe(1_010_000_000, 1_701_000_000)

    # A near-flat pair fits an intercept close to the whole measured peak.
    assert ratio.fixed_for_test() > 1_000_000_000
    # ...but a tiny build is still predicted from the rate it demonstrated.
    assert ratio.predict(7_000_000) < 100_000_000
    # ...and the budget still admits a usable input size.
    assert ratio.max_inputs_for(1_800_000_000) > 100_000_000


def test_peak_ratio_intercept_cannot_exceed_a_measured_peak():
    # Two heavy builds whose peaks barely differ fit a nearly flat line whose
    # intercept swallows the whole pool. Left alone it rejects every window
    # however small -- a render collapsed to single frames on it -- yet a build
    # that peaked at 4 MB has already disproved a gigabyte of fixed cost.
    ratio = PeakRatioModel(seed=6.0, safety=1.0)
    ratio.observe(120_000_000, 2_000_000_000)
    ratio.observe(123_000_000, 2_004_000_000)
    ratio.observe(300_000, 4_000_000)

    # A tiny build is predicted near the tiny peak already seen, not near the
    # two-gigabyte intercept the heavy pair fits.
    assert ratio.predict(300_000) < 5_000_000
    assert ratio.predict(0) <= 4_000_000


def test_model_is_isolated_per_signature():
    model = ChunkMemoryModel()
    model.observe(("a",), 1, 1_000_000)
    assert model.plan(("b",), 100, GB) == 1
