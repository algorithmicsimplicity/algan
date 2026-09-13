"""Worker-thread stages must not be charged against the render thread's clock.

With scene prefetch on, batch b+1's preparation runs on a worker thread while
batch b renders on the main thread. ``StageTimers`` sums every stage into one
table regardless of thread, and the report used to subtract that sum from the
main thread's wall clock -- so the overlap was double-counted and the
"(unaccounted ...)" line read negative (-60% of a prep-heavy PREVIEW render,
-107% of a cold one). These tests pin the per-thread ledger and the budget
line that reads it.
"""

import threading
import time

import pytest

from algan.utils.profiling_utils import StageTimers, render_thread_budget


def _spin(seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


UNIT = 0.02


def _result(timers, total):
    return {
        "total": total,
        "times": dict(timers.times),
        "exclusive_times": dict(timers.exclusive_times),
        "main_times": dict(timers.main_times),
        "main_exclusive_times": dict(timers.main_exclusive_times),
        "worker_times": dict(timers.worker_times),
        "worker_exclusive_times": dict(timers.worker_exclusive_times),
    }


def test_worker_thread_stages_land_in_the_worker_ledger():
    timers = StageTimers()

    def prep():
        with timers.stage("prep"):
            _spin(UNIT)

    worker = threading.Thread(target=prep, name="algan-batch-prep-0")
    t0 = time.perf_counter()
    worker.start()
    with timers.stage("render"):
        _spin(UNIT)
    worker.join()
    total = time.perf_counter() - t0

    # The all-thread table still has both -- that is what the stage table
    # is read from.
    assert timers.times["prep"] >= UNIT
    assert timers.times["render"] >= UNIT
    # But each is on exactly one side of the per-thread ledger.
    assert "prep" not in timers.main_times
    assert "render" not in timers.worker_times
    assert timers.worker_exclusive_times["prep"] == pytest.approx(
        timers.exclusive_times["prep"]
    )
    assert timers.main_exclusive_times["render"] == pytest.approx(
        timers.exclusive_times["render"]
    )

    accounted, unaccounted, worker_secs = render_thread_budget(_result(timers, total))
    # Only the render thread is in the wall clock's budget, so the budget
    # cannot go negative however much the worker did in parallel.
    assert accounted == pytest.approx(timers.exclusive_times["render"])
    assert unaccounted >= 0
    assert worker_secs == pytest.approx(timers.exclusive_times["prep"])


def test_summing_both_threads_would_have_gone_negative():
    """The failure this guards against, stated as arithmetic: a worker that
    is busy for the whole of the main thread's span makes the all-thread sum
    exceed the wall clock.
    """
    timers = StageTimers()
    done = threading.Event()

    def prep():
        with timers.stage("prep"):
            done.wait()

    worker = threading.Thread(target=prep, name="algan-batch-prep-0")
    t0 = time.perf_counter()
    worker.start()
    with timers.stage("render"):
        _spin(UNIT)
    done.set()
    worker.join()
    total = time.perf_counter() - t0

    both = sum(timers.exclusive_times.values())
    assert both > total, "the worker's span overlapped the render's"
    _accounted, unaccounted, _worker = render_thread_budget(_result(timers, total))
    assert unaccounted >= 0


def test_kernel_hooks_use_the_same_ledger():
    """A kernel launched from a worker thread is worker time, and one launched
    on the main thread is main time -- the hook goes through ``_record``.
    """
    timers = StageTimers()
    timers._record("kernel: k", UNIT, UNIT)

    def from_worker():
        timers._record("kernel: k", UNIT, UNIT)

    worker = threading.Thread(target=from_worker)
    worker.start()
    worker.join()

    assert timers.times["kernel: k"] == pytest.approx(2 * UNIT)
    assert timers.main_times["kernel: k"] == pytest.approx(UNIT)
    assert timers.worker_times["kernel: k"] == pytest.approx(UNIT)


def test_a_result_without_thread_tables_reads_as_main_thread():
    """Reports of older result dicts degrade to what they printed before."""
    res = {"total": 1.0, "times": {"a": 0.4}, "exclusive_times": {"a": 0.4}}
    accounted, unaccounted, worker = render_thread_budget(res)
    assert accounted == pytest.approx(0.4)
    assert unaccounted == pytest.approx(0.6)
    assert worker == 0
