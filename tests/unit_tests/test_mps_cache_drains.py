"""During a render, only the render job's thread drains torch's MPS caches.

``torch.mps.empty_cache()`` also clears torch's MPSGraph cache, deleting every
cached graph, while an MPS op on another thread holds its graph by a raw
pointer between looking it up and encoding it. The batch-prep worker drained
that way under host pressure while the render thread ran ``index_select`` on
the GPU, and on the 7 GB Mac runner 3 of 40 UHD renders died of it
(EXC_BAD_ACCESS inside ``index_select_out_mps``). A drain asked for on another
thread now waits for the render job's thread: its next reclaim, or the end of
the job (``memory_utils.drain_mps_caches``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import types

import pytest
import torch

from algan.rendering import mps_zero_copy, taichi_runtime
from algan.settings._startup import render_device
from algan.utils import memory_utils as mu

#: The batch-prep worker's thread name, as ``render_loop`` names it.
WORKER = "algan-batch-prep_0"


@pytest.fixture
def drains(monkeypatch):
    """Where each MPS drain and import-cache clear ran, on any machine.

    torch.mps is stubbed so the reclaim takes this for an MPS machine, and the
    threads named in the returned set see host pressure. Nothing else in the
    reclaim acts: no collection, no heap trim, no Quadrants reset, and the
    render job holds no allocator limit.
    """
    events = []
    pressured = set()

    def note(kind):
        return lambda: events.append((kind, threading.current_thread().name))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "empty_cache", note("drain"))
    monkeypatch.setattr(mps_zero_copy, "clear_import_cache", note("clear"))
    monkeypatch.setattr(
        mu,
        "_host_memory_pressure",
        lambda *a, **k: threading.current_thread().name in pressured,
    )
    monkeypatch.setattr(mu, "_gpu_memory_pressure", lambda *a, **k: False)
    monkeypatch.setattr(mu.gc, "collect", lambda *a: 0)
    monkeypatch.setattr(mu, "_malloc_trim", lambda: False)
    monkeypatch.setattr(
        mu, "_reset_quadrants_runtime_for_memory_pressure", lambda: False
    )
    monkeypatch.setattr(mu, "render_device", lambda: torch.device("cpu"))
    # The free figure's inputs, pinned so it reads nothing of this machine.
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 0)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 4 << 30)
    monkeypatch.setattr(
        mu.psutil, "virtual_memory", lambda: types.SimpleNamespace(total=64 << 30)
    )
    monkeypatch.setattr(mu, "_MPS_DRAIN_DEFERRED", False, raising=False)
    return events, pressured


def _on(name, fn, *args):
    """Call ``fn(*args)`` on a new thread called ``name``; raise what it raised."""
    raised = []

    def run():
        try:
            fn(*args)
        except BaseException as exc:  # noqa: BLE001
            raised.append(exc)

    thread = threading.Thread(target=run, name=name)
    thread.start()
    thread.join()
    if raised:
        raise raised[0]


def _drains(events):
    return [thread for kind, thread in events if kind == "drain"]


def test_a_drain_asked_for_off_the_render_thread_waits_for_the_render_thread(drains):
    """The worker's pressured and forced reclaims, and a free-figure measure.

    None of them drains on the worker during the job. The render thread's
    next reclaim makes one drain for all three, though it sees no pressure
    itself, and one asked for after that is made when the job ends. With no
    job running, the worker drains at once, as before.
    """
    events, pressured = drains
    render = threading.current_thread().name
    pressured.add(WORKER)
    with taichi_runtime.render_job_holding_the_arch():
        # After each primitive it projects, and on its retry path.
        _on(WORKER, mu.release_torch_memory, False)
        _on(WORKER, mu.release_torch_memory, True)
        _on(WORKER, mu.get_num_available_bytes, torch.device("mps"))
        assert events == [], (
            "a thread other than the render job's drained torch's MPS caches "
            "during the job"
        )
        mu.release_torch_memory(force_gc=False)
        assert events == [("clear", render), ("drain", render)]
        mu.release_torch_memory(force_gc=False)
        assert len(events) == 2, "a deferred drain was made more than once"
        _on(WORKER, mu.release_torch_memory, False)
        assert len(events) == 2
    assert _drains(events) == [render, render]
    assert all(thread == render for _, thread in events)
    _on(WORKER, mu.release_torch_memory, False)
    assert _drains(events)[2:] == [WORKER]


#: A render job's thread running ``index_select`` over strided MPS views, the
#: op the UHD renders died in, with a matmul keeping the GPU busy, while a
#: second thread reclaims under host pressure as the batch-prep worker does
#: after each primitive it projects. Both ops take a cached graph, so either
#: can die; before the fix, each of the three deaths that showed where was in
#: the matmul (run 42424700295). Collection is stubbed out so that thread
#: reclaims as fast as it can; it yields the GIL between reclaims, or the
#: render thread would barely run. The render thread reclaims every 25
#: rounds, as the render loop does several times a chunk. Prints how many
#: drains each thread made, and how many rounds the render thread ran.
_INDEX_SELECT_BESIDE_A_RECLAIMING_THREAD = textwrap.dedent(
    """
    import gc
    import sys
    import threading
    import time

    import torch

    from algan.rendering.taichi_runtime import render_job_holding_the_arch
    from algan.utils import memory_utils as mu

    render = threading.get_ident()
    drains = {"render": 0, "other": 0}
    empty_cache = torch.mps.empty_cache

    def counted_empty_cache():
        drains["render" if threading.get_ident() == render else "other"] += 1
        empty_cache()

    torch.mps.empty_cache = counted_empty_cache
    mu._host_memory_pressure = lambda *a, **k: True
    mu._reset_quadrants_runtime_for_memory_pressure = lambda: False
    gc.collect = lambda *a: 0
    stop = threading.Event()

    def reclaim():
        while not stop.is_set():
            mu.release_torch_memory(force_gc=False)
            time.sleep(0)

    device = torch.device("mps")
    rows = torch.randn(8192, 48, device=device)
    busy = torch.randn(768, 768, device=device)
    rounds = 0
    with render_job_holding_the_arch():
        other = threading.Thread(target=reclaim, daemon=True)
        other.start()
        deadline = time.perf_counter() + float(sys.argv[1])
        while time.perf_counter() < deadline:
            busy = (busy @ busy).tanh_()
            # A new length each round, so each round can build a new graph.
            index = torch.randint(0, 8192, (512 + 7 * (rounds % 211),), device=device)
            rows[:, 3:35].index_select(0, index)
            rows.t().index_select(1, index)
            rounds += 1
            if rounds % 25 == 0:
                mu.release_torch_memory(force_gc=False)
                torch.mps.synchronize()
        stop.set()
        other.join()
    torch.mps.synchronize()
    print("DRAINS", drains["render"], drains["other"], rounds, flush=True)
    """
)

#: How long the render thread runs beside the reclaiming one.
_RACE_SECONDS = 10


def test_an_mps_op_survives_a_reclaim_on_another_thread():
    """On the Apple GPU: the render thread does not die of another's reclaim.

    ``torch.mps.empty_cache()`` called in a loop on a second thread killed a
    process running MPSGraph ops on its main thread within 1.5 s, 4 times of
    4, each while building an op's feeds dictionary from a deleted graph:
    twice EXC_BAD_ACCESS, as in the UHD renders, and twice an unrecognized
    selector sent to whatever then occupied its memory (run 42422403489).
    Before the fix the child here died 4 times of 4, each process within
    6.4 s of starting, where one that lives its 10 s takes 14 to 18 s; after
    it, 6 of 6 lived (run 42424700295).
    """
    if sys.platform != "darwin" or render_device().type != "mps":
        pytest.skip("requires the MPS render arm")
    env = dict(os.environ, ALGAN_USE_DAEMON="0", ALGAN_AUTO_DAEMON="0")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _INDEX_SELECT_BESIDE_A_RECLAIMING_THREAD,
            str(_RACE_SECONDS),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert proc.returncode == 0, (
        f"the render thread's process died (returncode {proc.returncode}) while "
        f"another thread reclaimed: {proc.stderr[-3000:]}"
    )
    report = next(
        (line for line in proc.stdout.split("\n") if line.startswith("DRAINS")), ""
    )
    assert report, proc.stdout[-2000:] + proc.stderr[-2000:]
    _, on_render, on_other, rounds = report.split()
    assert on_other == "0", (
        f"the reclaiming thread drained torch's MPS caches {on_other} times "
        "during the render job"
    )
    assert int(on_render) > 0, "the render thread never made the drains asked of it"
    assert int(rounds) > 0
