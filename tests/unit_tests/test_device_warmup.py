"""The start-up device warm-up must stay out of a render's way.

``algan/rendering/device_warmup.py`` loads the torch kernels the renderer uses
on a background thread while a scene script authors. Held here:

* every warm-up op is a valid call on this torch build (checked on the CPU,
  so it runs everywhere; on CUDA when there is one) -- an op that raised would
  be skipped silently and its loading cost would come back;
* it starts only for a CUDA render device, never with
  ``ALGAN_STARTUP_WARMUP=0``, never in a precompile worker and never from the
  import of a test runner;
* the first render job stops and joins it, so the render never shares the
  device with it.
"""

from __future__ import annotations

import pytest
import torch

from algan.rendering import device_warmup


@pytest.fixture(autouse=True)
def _clean_warmup():
    device_warmup._reset_for_tests()
    yield
    device_warmup._reset_for_tests()


def test_every_warmup_op_is_a_valid_call_on_the_cpu():
    failures = []
    with torch.inference_mode():
        for name, op in device_warmup._ops(torch.device("cpu")):
            try:
                op()
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_cuda_warmup_runs_every_op_and_is_joined(monkeypatch):
    import algan.settings._startup as startup

    monkeypatch.setattr(startup, "render_device", lambda: torch.device("cuda"))
    assert device_warmup.start("test")
    assert device_warmup.finish(timeout=60)
    run = device_warmup.LAST_RUN
    assert run["failed"] == 0
    assert run["ops"] + run["stopped"] >= 1
    # Idempotent: a finished thread is not restarted in the same process.
    assert not device_warmup.start("again")


def test_finish_stops_the_steps_not_yet_taken(monkeypatch):
    """The render never waits for more than the step in flight."""
    import threading
    import time

    started = threading.Event()

    def slow():
        started.set()
        time.sleep(0.05)

    monkeypatch.setattr(
        device_warmup, "_ops", lambda device: [(f"s{k}", slow) for k in range(200)]
    )
    device_warmup._start_thread(torch.device("cpu"), "test")
    assert started.wait(10)
    began = time.perf_counter()
    assert device_warmup.finish(timeout=10)
    assert time.perf_counter() - began < 1.0
    run = device_warmup.LAST_RUN
    assert run["stopped"]
    assert run["ops"] < 200
    assert run["failed"] == 0


def test_a_failing_step_is_skipped_not_raised(monkeypatch):
    def boom():
        raise RuntimeError("no such kernel on this build")

    monkeypatch.setattr(
        device_warmup, "_ops", lambda device: [("boom", boom), ("ok", lambda: None)]
    )
    device_warmup._start_thread(torch.device("cpu"), "test")
    device_warmup._THREAD.join(10)
    assert device_warmup.LAST_RUN["failed"] == 1
    assert device_warmup.LAST_RUN["ops"] == 1


def test_nothing_starts_for_a_cpu_or_mps_render_device(monkeypatch):
    import algan.settings._startup as startup

    for device in ("cpu", "mps"):
        monkeypatch.setattr(startup, "render_device", lambda d=device: torch.device(d))
        assert not device_warmup.start("test")
        assert not device_warmup.active()


def test_the_switch_and_the_worker_flag_turn_it_off(monkeypatch):
    monkeypatch.setenv("ALGAN_STARTUP_WARMUP", "0")
    assert not device_warmup.enabled()
    monkeypatch.delenv("ALGAN_STARTUP_WARMUP")
    assert device_warmup.enabled()
    monkeypatch.setenv("ALGAN_PRECOMPILE_WORKER", "1")
    assert not device_warmup.enabled()


def test_a_test_runner_never_starts_it_at_import(monkeypatch):
    started = []
    monkeypatch.setattr(device_warmup, "start", lambda reason="": started.append(1))
    assert device_warmup.start_at_import() is False
    assert started == []


def test_the_first_render_job_joins_it(monkeypatch):
    from algan.rendering import taichi_runtime

    calls = []
    monkeypatch.setattr(device_warmup, "finish", lambda *a, **k: calls.append(1))
    with (
        taichi_runtime.render_job_holding_the_arch(),
        taichi_runtime.render_job_holding_the_arch(),
    ):
        pass
    # Once per outermost job, not per nested one.
    assert calls == [1]
