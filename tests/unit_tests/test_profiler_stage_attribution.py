"""The stage profiler's exclusive column must exclude the kernels a stage ran.

A stage's ``excl`` is read as "work this stage did itself", and every report
that guided an optimization round has been read that way. Nested *stages* were
always subtracted; Taichi **kernel** launches were not, because the kernel hooks
write straight into ``TIMERS.times`` instead of opening a stage. On a 4K render
that made ``wavefront_loop`` report 13.2 s of "unattributed host work" when 12.5 s
of it was two kernels listed by name in the same table.

These tests pin the bookkeeping contract rather than any timing: they drive
``StageTimers`` directly with known durations, so they are fast and
device-independent.
"""

import time

import pytest

from algan.utils.profiling_utils import StageTimers


def _spin(seconds):
    """Busy-wait, so the measured runtime is not at the mercy of sleep jitter."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


UNIT = 0.02


def test_kernel_time_is_subtracted_from_the_enclosing_stage():
    timers = StageTimers()
    with timers.stage("outer"):
        _spin(UNIT)
        timers.charge_kernel_to_parent(UNIT)

    assert timers.counts["outer"] == 1
    # The stage ran for at least UNIT and charged UNIT of it to a kernel, so
    # its own time is what is left over -- not the whole span.
    assert timers.times["outer"] >= UNIT
    assert timers.exclusive_times["outer"] == pytest.approx(
        timers.times["outer"] - UNIT, abs=1e-9
    )
    assert timers.exclusive_times["outer"] < timers.times["outer"]


def test_kernels_and_nested_stages_are_both_subtracted_once():
    timers = StageTimers()
    with timers.stage("outer"):
        timers.charge_kernel_to_parent(UNIT)  # a kernel outer launched itself
        with timers.stage("inner"):
            _spin(UNIT)
            timers.charge_kernel_to_parent(UNIT)  # a kernel inner launched

    inner = timers.times["inner"]
    outer = timers.times["outer"]
    # inner's own time excludes only its own kernel.
    assert timers.exclusive_times["inner"] == pytest.approx(inner - UNIT, abs=1e-9)
    # outer's own time excludes its own kernel and the whole of inner -- inner's
    # kernel must not be subtracted from outer a second time.
    assert timers.exclusive_times["outer"] == pytest.approx(
        outer - UNIT - inner, abs=1e-9
    )


def test_sibling_stages_do_not_inherit_each_others_kernels():
    timers = StageTimers()
    with timers.stage("outer"):
        with timers.stage("first"):
            timers.charge_kernel_to_parent(UNIT)
        with timers.stage("second"):
            pass

    assert timers.exclusive_times["first"] == pytest.approx(
        timers.times["first"] - UNIT, abs=1e-9
    )
    # "second" launched nothing, so its exclusive time is its whole span.
    assert timers.exclusive_times["second"] == pytest.approx(
        timers.times["second"], abs=1e-9
    )


def test_a_kernel_outside_any_stage_corrupts_nothing():
    timers = StageTimers()
    timers.charge_kernel_to_parent(UNIT)
    with timers.stage("outer"):
        _spin(UNIT)

    # The stray charge landed at depth 0, which no stage reads.
    assert timers.exclusive_times["outer"] == pytest.approx(
        timers.times["outer"], abs=1e-9
    )
