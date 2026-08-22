"""Guards on what the tonemap does to a colour.

These call the post stage directly rather than rendering a scene: the tonemap
is the last thing to touch a frame, so handing ``_finalize_on_device`` a
synthetic linear-HDR buffer exercises exactly the code under test and nothing
else. ``benchmarks/_tonemap_transfer_probe.py`` is the same technique with a
wider ramp and a printed table.

Both shipping implementations are covered wherever it is cheap to do so -- the
standalone Taichi kernel (``post_tonemap_kernel=True``, the default) and the
torch pipeline it replaced -- because the defects these guard against were
present in *both*, which is precisely why an agreement-between-implementations
check did not catch them. See ``TONEMAP_FINDINGS.md``.

Deliberately not marked ``fast``: nothing outside this module can break them.
"""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.post_processing.post_process import _finalize_on_device
from algan.rendering.raytracing import settings as rt_settings
from algan.settings._startup import _RENDER_DEVICE
from algan.utils.memory_utils import ManualMemory

# Both implementations, as (id, use_taichi_kernel).
IMPLEMENTATIONS = [("kernel", True), ("torch", False)]


def encode(values, *, tonemapping, exposure=1.0, method="neutral", kernel=True):
    """Push a list of neutral linear-HDR values through the post stage.

    Returns one ``(r, g, b)`` tuple of uint8 per input value -- what that value
    would land on in the encoded frame.
    """
    frame = torch.zeros(
        (1, 1, len(values), 4), dtype=torch.float32, device=_RENDER_DEVICE
    )
    for i, v in enumerate(values):
        frame[0, 0, i, :3] = v

    memory = ManualMemory(0.0, device=_RENDER_DEVICE, managed=False, num_bytes=1 << 22)
    was_kernel = rt_settings.POST_TONEMAP_KERNEL
    rt_settings.set_post_tonemap_kernel(kernel)
    try:
        out = _finalize_on_device(
            frame,
            4,
            memory,
            tonemap_enabled=True,
            tonemapping=tonemapping,
            tonemap_method=method,
            exposure=exposure,
        )
        return [
            tuple(int(x) for x in out[0, 0, i, :3].tolist()) for i in range(len(values))
        ]
    finally:
        rt_settings.set_post_tonemap_kernel(was_kernel)


def test_tonemapping_is_off_by_default():
    # An authored colour should land on the pixel it names. The curve cannot do
    # that -- it is the identity nowhere except at 0 -- so the default is off.
    assert SETTINGS.raytracing.tonemapping is False


@pytest.mark.parametrize(("impl", "kernel"), IMPLEMENTATIONS)
def test_off_is_exact_passthrough_of_authored_bytes(impl, kernel):
    authored = [0, 1, 17, 64, 128, 191, 254, 255]
    got = encode([k / 255.0 for k in authored], tonemapping=False, kernel=kernel)
    assert [c[0] for c in got] == authored


@pytest.mark.parametrize(("impl", "kernel"), IMPLEMENTATIONS)
def test_exposure_applies_with_tonemapping_off(impl, kernel):
    # Regression: exposure used to be dropped on the floor whenever the curve
    # was off, so every exposure produced identical bytes. It is the documented
    # brightness control and -- with tonemapping off by default -- the only one.
    values = [0.25, 0.5, 0.75]
    at_half = encode(values, tonemapping=False, exposure=0.5, kernel=kernel)
    at_one = encode(values, tonemapping=False, exposure=1.0, kernel=kernel)
    at_two = encode(values, tonemapping=False, exposure=2.0, kernel=kernel)

    assert [c[0] for c in at_one] == [64, 128, 191]
    assert [c[0] for c in at_half] == [32, 64, 96]
    assert at_half != at_one
    assert at_two != at_one


@pytest.mark.parametrize(("impl", "kernel"), IMPLEMENTATIONS)
def test_exposure_of_one_is_exact(impl, kernel):
    # The default must move no pixel, or flipping the default would have
    # silently re-baselined every scene.
    values = [k / 255.0 for k in (0, 33, 128, 200, 255)]
    assert encode(values, tonemapping=False, exposure=1.0, kernel=kernel) == encode(
        values, tonemapping=False, kernel=kernel
    )


@pytest.mark.parametrize(("impl", "kernel"), IMPLEMENTATIONS)
def test_agx_maps_neutral_to_neutral(impl, kernel):
    """A tonemap may darken a grey; it may not tint one.

    This is the guard for the transposed Rec.2020 -> Rec.709 matrix, which gave
    every neutral a fixed +52% red / -56% green and rendered authored grey
    ``(128, 128, 128)`` as magenta ``(255, 77, 180)``. The invariant is that
    the two spaces share a white point, so the conversion maps white to white.
    """
    for value, out in zip(
        (0.25, 0.5, 0.75, 1.0),
        encode([0.25, 0.5, 0.75, 1.0], tonemapping=True, method="agx", kernel=kernel),
    ):
        assert max(out) - min(out) <= 1, f"agx tinted neutral {value}: {out}"


def test_agx_implementations_agree():
    """Useful, but on its own it proves nothing about correctness.

    This test passed throughout the period when both implementations carried
    the transposed matrix: two wrong implementations agree perfectly. It is
    here to catch one of them drifting from the other, and
    ``test_agx_maps_neutral_to_neutral`` above is what checks either is right.
    """
    values = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0]
    taichi = encode(values, tonemapping=True, method="agx", kernel=True)
    torch_ = encode(values, tonemapping=True, method="agx", kernel=False)
    assert taichi == torch_


def test_neutral_curve_transfer_is_unchanged():
    """Pin the shipped neutral curve, so a change to it has to be deliberate.

    These are Khronos PBR Neutral's own numbers. They are recorded here not
    because they are desirable -- ``1.0 -> 222`` is why the default is off --
    but so that editing the curve shows up as an edit to this list.
    """
    values = [0.0, 0.08, 0.25, 0.5, 0.76, 1.0, 2.0, 4.0]
    expected = [0, 10, 54, 117, 184, 222, 245, 251]
    for kernel in (True, False):
        got = [c[0] for c in encode(values, tonemapping=True, kernel=kernel)]
        assert got == expected, f"kernel={kernel}"


def test_curve_moves_values_that_were_already_in_range():
    """The finding that prompted the default flip, kept as a live measurement.

    Everything here is inside the display range before the curve runs, so an
    ideal HDR-only tonemap would leave it alone. None of it survives untouched
    except black -- which is why "only tonemap HDR values" is not something the
    curve can be tuned into doing.
    """
    values = [0.08, 0.25, 0.5, 0.75, 1.0]
    on = [c[0] for c in encode(values, tonemapping=True)]
    off = [c[0] for c in encode(values, tonemapping=False)]
    assert all(a < b for a, b in zip(on, off)), (on, off)
    # White is the worst of them, and by a wide margin over the ~10/255 that
    # was previously on record.
    assert off[-1] - on[-1] == 33
