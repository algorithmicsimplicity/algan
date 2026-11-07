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

#: torch's ``set_per_process_memory_fraction`` calls a render should make: the
#: working set when it starts, and torch's own default when it ends.
HELD, TORCH_DEFAULT = 1.0, 1.7


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


def test_an_mps_render_cannot_allocate_past_the_working_set():
    """On the Apple GPU: refused inside a render, with an error its retries take.

    Refused before the buffers reach the working set, and torch's own limit,
    1.7 times it, is back once the render ends: the same process then takes
    one more.
    """
    if sys.platform != "darwin" or render_device().type != "mps":
        pytest.skip("requires the MPS render arm")
    env = dict(os.environ, ALGAN_USE_DAEMON="0", ALGAN_AUTO_DAEMON="0")
    env.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)
    proc = subprocess.run(
        [sys.executable, "-c", _ALLOCATE_PAST_THE_WORKING_SET],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
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
