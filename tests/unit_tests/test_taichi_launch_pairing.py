"""Which (tensor device, Taichi arch) pairings are free, and which arch serves
which render device.

Two questions the engine has to get right and got wrong in the same way: both
treated "same device family" as "same device", so an Apple GPU passed for a
CUDA one and a Metal program passed for a CUDA program. Neither could be caught
on the machines the suite runs on -- the answers only differ on hardware CI does
not have -- so every case here drives the two accessors that read the outside
world (:func:`_live_arch` and :func:`render_device`) through monkeypatch, which
is what makes the MPS and multi-GPU rows testable on a CPU-only box.

Measured evidence for the MPS rows is in ``DESIGN_mps_support.md``: the same
kernel over the same 32 MB runs in 1.09 ms on the CPU arch and 58.01 ms on Metal
with MPS tensors, because Taichi copies each argument to the host and back
around the launch. ``taichi_launch_is_local`` used to call that free.
"""

from __future__ import annotations

import pytest
import torch

from algan.rendering import taichi_runtime
from algan.taichi_compat import ti

# Deliberately not marked ``fast``. These assertions fail when
# ``taichi_runtime`` itself changes and at no other time -- nothing elsewhere in
# the package can move them, because they monkeypatch the two accessors that
# read the outside world -- which is what ``tests/README.md`` calls a feature
# test. Being cheap (0.5 s) is not a reason to put it in the development loop.


@pytest.fixture
def arch(monkeypatch):
    """Pin the live Taichi arch and the render device, together.

    Returns a setter so a case reads as the pairing it is about. ``live=None``
    is "Taichi is not up yet", which is the path that answers from the render
    device instead.
    """

    def set_pairing(live, device):
        monkeypatch.setattr(taichi_runtime, "_live_arch", lambda: live)
        monkeypatch.setattr(
            taichi_runtime, "render_device", lambda: torch.device(device)
        )

    return set_pairing


# ---------------------------------------------------------------------------
# taichi_launch_is_local
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("live", "render", "tensor", "expected"),
    [
        # The two pairings Taichi can bind without copying: its core implements
        # Device::import_memory for CpuDevice and CudaDevice, and nothing else.
        (ti.cpu, "cpu", "cpu", True),
        (ti.cuda, "cuda", "cuda", True),
        # A host tensor on a GPU arch stages -- true of CUDA, and the reason
        # this is not simply `device.type == "cuda"`.
        (ti.cuda, "cuda", "cpu", False),
        (ti.metal, "mps", "cpu", False),
        # A device tensor on the CPU arch stages the other way.
        (ti.cpu, "cpu", "cuda", False),
        (ti.cpu, "cpu", "mps", False),
        # THE REGRESSION. Both halves name the same Apple GPU and Taichi still
        # copies through the host, because its Metal backend cannot import a
        # pointer torch allocated. A device-equality test answered True here.
        (ti.metal, "mps", "mps", False),
        (ti.vulkan, "mps", "mps", False),
        # ... and it is the arch that decides, not the render device: an MPS
        # tensor is no more bindable when the program happens to be on CUDA.
        (ti.cuda, "cuda", "mps", False),
    ],
)
def test_only_cpu_on_cpu_and_cuda_on_cuda_avoid_staging(
    arch, live, render, tensor, expected
):
    arch(live, render)
    assert taichi_runtime.taichi_launch_is_local(torch.device(tensor)) is expected


def test_launch_locality_answers_before_taichi_is_up(arch):
    """With no live program the answer comes from the render device.

    Asking must never force initialization -- call sites read this while
    deciding whether to build kernel arguments at all -- so the uninitialized
    path has to give the same answers the live one will.
    """
    arch(None, "cpu")
    assert taichi_runtime.taichi_launch_is_local(torch.device("cpu")) is True
    assert taichi_runtime.taichi_launch_is_local(torch.device("cuda")) is False

    arch(None, "cuda")
    assert taichi_runtime.taichi_launch_is_local(torch.device("cuda")) is True
    assert taichi_runtime.taichi_launch_is_local(torch.device("cpu")) is False

    arch(None, "mps")
    assert taichi_runtime.taichi_launch_is_local(torch.device("mps")) is False


# ---------------------------------------------------------------------------
# _arch_matches_render_device
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("live", "render", "expected"),
    [
        (ti.cpu, "cpu", True),
        (ti.cuda, "cuda", True),
        # Either SPIR-V backend really does serve an Apple GPU: ti.gpu picks
        # metal, TI_ARCH=vulkan picks vulkan, both run on the same device.
        (ti.metal, "mps", True),
        (ti.vulkan, "mps", True),
        # THE REGRESSION. `live != ti.cpu` made every GPU backend
        # interchangeable, so a device moving between two of them kept whichever
        # program was up and launched every kernel on the wrong one.
        (ti.cuda, "mps", False),
        (ti.metal, "cuda", False),
        (ti.vulkan, "cuda", False),
        # Crossing the CPU/GPU line was always caught, and still is.
        (ti.cpu, "cuda", False),
        (ti.cuda, "cpu", False),
        (ti.metal, "cpu", False),
    ],
)
def test_arch_must_serve_the_render_device(arch, live, render, expected):
    arch(live, render)
    assert taichi_runtime._arch_matches_render_device() is expected


def test_no_live_program_never_matches(arch):
    """``ensure_taichi_for_render`` distinguishes "no program" from "wrong
    program", and only this function can tell it which it is looking at.
    """
    arch(None, "cpu")
    assert taichi_runtime._arch_matches_render_device() is False


def test_unrecognised_device_keeps_the_coarse_rule(arch):
    """A device type the mapping has never seen falls back to "any GPU arch".

    Deliberately not a re-initialization: `ti.init` drops every compiled kernel,
    and paying that on every render of a device that may well be served
    correctly is worse than the imprecision it would buy.
    """
    arch(ti.metal, "xpu")
    assert taichi_runtime._arch_matches_render_device() is True
    arch(ti.cpu, "xpu")
    assert taichi_runtime._arch_matches_render_device() is False


def test_cpu_and_cuda_arch_predicates_agree_with_the_live_program(arch):
    """The two predicates the pairing rule is built from, read directly."""
    arch(ti.cpu, "cpu")
    assert taichi_runtime.taichi_arch_is_cpu() is True
    assert taichi_runtime.taichi_arch_is_cuda() is False

    arch(ti.cuda, "cuda")
    assert taichi_runtime.taichi_arch_is_cpu() is False
    assert taichi_runtime.taichi_arch_is_cuda() is True

    arch(ti.metal, "mps")
    assert taichi_runtime.taichi_arch_is_cpu() is False
    assert taichi_runtime.taichi_arch_is_cuda() is False
