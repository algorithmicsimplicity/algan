"""A render on MPS holds torch's allocator to Metal's working set.

Metal refuses no allocation past ``recommendedMaxWorkingSetSize``, nor past
the machine's RAM; it fails the work that needs more memory resident than the
machine can give it: torch raises nothing, the work's outputs hold whatever
their buffers held, and MPS may print "Insufficient Memory
(kIOGPUCommandBufferCallbackErrorOutOfMemory)" to stderr or may print nothing.
torch's own limit is 1.7 times the working set, more than a 7 GB Mac has, so
nothing stopped a render getting there.
``memory_utils.hold_mps_allocator_to_the_working_set`` lowers the limit for the
length of a render job, which turns the same shortage into torch's "MPS backend
out of memory", the error the render's retries split a window on.

torch checks that limit against each request rather than the heap it opens,
and opens 1 GiB heaps until the device passes its low watermark, so Algan
starts the allocator with a low one (``algan.utils.mps_low_watermark``) and
holds it 2% under the working set.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
import torch

from algan.rendering import taichi_runtime
from algan.settings._startup import render_device
from algan.utils import memory_utils as mu

#: torch's ``set_per_process_memory_fraction`` calls a render should make: 2%
#: under the working set when it starts, and torch's own default when it ends.
HELD, TORCH_DEFAULT = 0.98, 1.7


@pytest.fixture
def limits(monkeypatch):
    """The limits a render job gives torch's MPS allocator, on any machine."""
    calls = []
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "set_per_process_memory_fraction", calls.append)
    monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raising=False)
    monkeypatch.setattr(mu, "_MPS_ALLOCATOR_HELD", False, raising=False)
    return calls


def _render_on(monkeypatch, limits, device):
    """The limits set before, during and after a nested render job."""
    monkeypatch.setattr(mu, "render_device", lambda: torch.device(device))
    before = len(limits)
    with (
        taichi_runtime.render_job_holding_the_arch(),
        taichi_runtime.render_job_holding_the_arch(),
    ):
        inside = limits[before:]
    return inside, limits[before:]


def test_a_render_on_mps_holds_torch_to_the_working_set_until_it_ends(
    limits, monkeypatch
):
    """Held once for the outermost job, and torch's own limit back after it.

    Nothing else changes: a CPU or CUDA render never calls into torch.mps,
    and a limit the user set with ``PYTORCH_MPS_HIGH_WATERMARK_RATIO``
    stands, 0.0 (no limit) included.
    """
    assert _render_on(monkeypatch, limits, "mps") == ([HELD], [HELD, TORCH_DEFAULT])
    for device in ("cpu", "cuda"):
        assert _render_on(monkeypatch, limits, device) == ([], [])
    for ratio in ("1.5", "0.0"):
        monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", ratio)
        assert _render_on(monkeypatch, limits, "mps") == ([], [])


def test_a_render_that_fails_still_gives_the_limit_back(limits, monkeypatch):
    """An exception out of the render, or out of the job's own cleanup."""
    import algan.rendering.mps_zero_copy as zero_copy

    monkeypatch.setattr(mu, "render_device", lambda: torch.device("mps"))
    with (
        pytest.raises(RuntimeError, match="the render"),
        taichi_runtime.render_job_holding_the_arch(),
    ):
        raise RuntimeError("the render")
    assert limits == [HELD, TORCH_DEFAULT]

    def failing_cleanup():
        raise RuntimeError("the cleanup")

    monkeypatch.setattr(zero_copy, "clear_import_cache", failing_cleanup)
    with (
        pytest.raises(RuntimeError, match="the cleanup"),
        taichi_runtime.render_job_holding_the_arch(),
    ):
        pass
    assert limits == [HELD, TORCH_DEFAULT] * 2


LOW = "PYTORCH_MPS_LOW_WATERMARK_RATIO"


@pytest.fixture
def low_watermark(monkeypatch):
    """``mps_low_watermark``, with torch's allocator not started until a test says.

    Imported here rather than at the top so that the Apple GPU tests below
    still run against a tree without the module, where the last one fails.
    """
    from algan.utils import mps_low_watermark

    started = []
    monkeypatch.setattr(
        torch._C,
        "_accelerator_isAllocatorInitialized",
        lambda: bool(started),
        raising=False,
    )
    monkeypatch.setattr(mps_low_watermark, "_IN_EFFECT", False)
    return mps_low_watermark, started


def test_algan_gives_torchs_mps_allocator_a_low_watermark_before_it_starts(
    low_watermark,
):
    """On a Mac with nothing chosen, before torch's allocator has read it."""
    module, _ = low_watermark
    environ = {}
    assert module.set_before_torch_starts(environ, "darwin")
    assert environ == {LOW: "0.75"}
    assert module.in_effect()
    for platform in ("linux", "win32"):
        environ = {}
        assert not module.set_before_torch_starts(environ, platform)
        assert environ == {}
        assert not module.in_effect()


def test_a_watermark_already_chosen_stands(low_watermark):
    """The user's watermarks, or a parent Algan process's, are left as they are.

    A low watermark counts only if it is low enough, and 0 turns torch's
    pressure handling off. A high watermark of the user's own means the render
    does not hold the allocator, so there is nothing for a low one to serve.
    """
    module, _ = low_watermark
    chosen = (
        ("0.5", True),
        ("0.75", True),
        ("1.4", False),
        ("0", False),
        ("no", False),
    )
    for raw, counts in chosen:
        environ = {LOW: raw}
        assert module.set_before_torch_starts(environ, "darwin") is counts
        assert environ == {LOW: raw}
        assert module.in_effect() is counts
    environ = {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "1.5"}
    assert not module.set_before_torch_starts(environ, "darwin")
    assert environ == {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "1.5"}


def test_an_allocator_that_already_started_is_left_alone(low_watermark):
    """MPS used before ``import algan``: nothing is set, and the hold knows.

    The allocator has read the environment and will not read it again, so a
    value set now would change nothing in this process; the hold says at
    debug level that its limit can be overshot.
    """
    module, started = low_watermark
    started.append(True)
    environ = {}
    assert not module.set_before_torch_starts(environ, "darwin")
    assert environ == {}
    assert not module.in_effect()


#: Run in a process of its own, so that what it allocates is gone with it.
#: The buffers are never written or read, so no command buffer can fail on
#: them, and it stops at 1.05 working sets. What fails is work that needs more
#: memory resident than the machine has: run 42351612808 built the Bezier
#: header of 5500 triangles over 100 frames on this runner, the device reached
#: 7.26 G against a working set of 4.67 G, Metal printed
#: kIOGPUCommandBufferCallbackErrorOutOfMemory twice, nothing raised, and
#: 102,317,937 of its 121,000,000 words were wrong. In run 42355691483 a
#: ``torch.add`` over three 2 GiB tensors left its output all zeros and
#: printed nothing at all.
_ALLOCATE_PAST_THE_WORKING_SET = textwrap.dedent(
    """
    import torch

    from algan.rendering.taichi_runtime import render_job_holding_the_arch
    from algan.utils.memory_utils import is_cuda_oom

    working_set = torch.mps.recommended_max_memory()
    chunk = 256 << 20
    held = []
    with render_job_holding_the_arch():
        try:
            while len(held) * chunk < 1.05 * working_set:
                held.append(torch.empty(chunk, dtype=torch.uint8, device="mps"))
        except RuntimeError as exc:
            share = len(held) * chunk / working_set
            print("REFUSED", is_cuda_oom(exc), f"{share:.3f}", flush=True)
        else:
            print("ALLOWED", f"{len(held) * chunk / working_set:.3f}", flush=True)
    held.append(torch.empty(chunk, dtype=torch.uint8, device="mps"))
    print("AFTER", f"{len(held) * chunk / working_set:.3f}", flush=True)
    """
)


def _child_environment():
    """This process's environment for a child that imports Algan afresh.

    No daemon, and neither of torch's watermarks, so the child's own import
    sets the low one as it would for a user's script.
    """
    env = dict(os.environ, ALGAN_USE_DAEMON="0", ALGAN_AUTO_DAEMON="0")
    env.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
    env.pop(LOW, None)
    return env


def test_an_mps_render_cannot_allocate_past_the_working_set():
    """On the Apple GPU: refused inside a render, with an error its retries take.

    Refused before the buffers reach the working set, and torch's own limit,
    1.7 times it, is back once the render ends: the same process then takes
    one more.
    """
    if sys.platform != "darwin" or render_device().type != "mps":
        pytest.skip("requires the MPS render arm")
    proc = subprocess.run(
        [sys.executable, "-c", _ALLOCATE_PAST_THE_WORKING_SET],
        capture_output=True,
        text=True,
        timeout=600,
        env=_child_environment(),
    )
    lines = proc.stdout.split("\n")
    report = next(
        (line for line in lines if line.startswith(("REFUSED", "ALLOWED"))), ""
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert not report.startswith("ALLOWED"), (
        f"torch let a render job allocate {report.split()[1]} times Metal's "
        "working set and raised nothing; past it Metal fails work that needs "
        "more memory resident than the machine has, and nothing raises"
    )
    _, oom, share = report.split()
    assert oom == "True", "the refusal is not an error the render retries"
    assert float(share) <= 1.0
    assert any(line.startswith("AFTER") for line in lines), proc.stderr[-2000:]


#: Written buffers of 256 MiB under a render job until torch refuses one, then
#: one in-place add over all of them in one command buffer, checked once the
#: job has given torch its own limit back. Each request is one torch puts in a
#: 1 GiB heap while the device is under its low watermark. Under torch's
#: default watermark, run 42374877006 had 17 of them in 5.00 G of heaps (1.07
#: working sets) under a limit at the working set, and the add left every
#: buffer as it was, with nothing raised or printed.
_ONE_ADD_OVER_ALL_A_RENDER_HOLDS = textwrap.dedent(
    """
    import torch

    from algan.rendering.taichi_runtime import render_job_holding_the_arch
    from algan.utils.memory_utils import is_cuda_oom

    working_set = torch.mps.recommended_max_memory()
    chunk = 256 << 20
    held = []
    with render_job_holding_the_arch():
        try:
            while len(held) * chunk < 1.1 * working_set:
                held.append(
                    torch.full((chunk // 4,), len(held) + 1, dtype=torch.int32, device="mps")
                )
                torch.mps.synchronize()
        except RuntimeError as exc:
            if not is_cuda_oom(exc):
                raise
        device = torch.mps.driver_allocated_memory()
        for value in held:
            value.add_(1)
        torch.mps.synchronize()
    wrong = [i for i, value in enumerate(held) if bool((value != i + 2).any())]
    print("HELD", len(held), f"{device / working_set:.4f}", len(wrong), flush=True)
    """
)


def test_one_command_buffer_over_all_an_mps_render_holds_comes_back_right():
    """On the Apple GPU: the heaps under a render's limit stay off the working set.

    The device ends under 0.99 working sets, short of the 0.994 at which one
    add over exact-size heaps was last seen right, and the add over every
    buffer comes back right. It writes nearly the whole working set, so it
    runs only where that is at most 12 GiB.
    """
    if sys.platform != "darwin" or render_device().type != "mps":
        pytest.skip("requires the MPS render arm")
    working_set = torch.mps.recommended_max_memory()
    if working_set > 12 << 30:
        pytest.skip(f"writes nearly all of a {working_set / 2**30:.1f} GiB working set")
    proc = subprocess.run(
        [sys.executable, "-c", _ONE_ADD_OVER_ALL_A_RENDER_HOLDS],
        capture_output=True,
        text=True,
        timeout=600,
        env=_child_environment(),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = next(
        (line for line in proc.stdout.split("\n") if line.startswith("HELD")), ""
    )
    assert report, proc.stdout[-2000:] + proc.stderr[-2000:]
    _, count, share, wrong = report.split()
    assert float(share) < 0.99, (
        f"{count} buffers of 256 MiB took the device to {share} working sets "
        "under a render's limit: torch opened a heap past it, and one command "
        "buffer over heaps that near the working set fails without a word"
    )
    assert wrong == "0", f"one add over {count} buffers left {wrong} of them wrong"
