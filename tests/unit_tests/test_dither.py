"""The dither on the final rounding to bytes.

A slow, dark gradient spans only a few byte values, so rounding each pixel to
the nearest one draws it as flat bands. ``SETTINGS.raytracing.dither`` (on by
default) adds a fixed offset of under half a byte to each pixel first. These
push synthetic frames through the post stage, as ``test_tonemapping.py`` does,
in both shipping implementations: the standalone Taichi kernel and its torch
twin.

A feature test, so it sits outside the fast suite: only a change to the post
stage breaks it.
"""

from __future__ import annotations

import pytest
import torch

from algan.rendering.post_processing.post_process import _finalize_on_device
from algan.settings import SETTINGS
from algan.settings._startup import render_device
from algan.utils.memory_utils import ManualMemory

IMPLEMENTATIONS = [("kernel", True), ("torch", False)]


@pytest.fixture(autouse=True)
def _post_stage():
    """The shipped post stage with no curve. Restored by settings isolation."""
    SETTINGS.raytracing.set(tonemapping=False, tonemap_exposure=1.0)
    SETTINGS.raytracing.experimental.set(post_process_tonemap=True)


def _bytes(frame, *, kernel, dither=True):
    """Finalize ``[H, W, 3]`` working-space values to ``[H, W, 3]`` bytes."""
    SETTINGS.raytracing.set(dither=dither)
    SETTINGS.raytracing.experimental.set(post_tonemap_kernel=kernel)
    height, width, _ = frame.shape
    hdr = torch.zeros((1, height, width, 4), device=render_device())
    hdr[0, ..., :3] = frame.to(render_device())
    memory = ManualMemory(0.0, device=render_device(), managed=False, num_bytes=1 << 22)
    out = _finalize_on_device(
        hdr,
        4,
        memory,
        tonemap_enabled=True,
        tonemapping=False,
        tonemap_method="neutral",
        exposure=1.0,
    )
    return out[0].cpu().long()


def _ramp(low, high, height=64, width=256):
    """A horizontal ramp from ``low`` to ``high`` bytes, as display values."""
    row = torch.linspace(low, high, width, dtype=torch.float64) / 255.0
    return row.view(1, width, 1).expand(height, width, 3).float()


def test_dither_is_on_by_default():
    assert SETTINGS.raytracing.dither is True


@pytest.mark.parametrize(("impl", "kernel"), IMPLEMENTATIONS)
def test_a_shallow_gradient_keeps_its_slope_instead_of_banding(impl, kernel):
    """Two bytes of rise over 256 pixels: rounding draws three flat bands.

    Dithered, each column's average over the rows follows the ramp, so the
    eye (and the encoder's low frequencies) see the slope rather than steps.
    """
    SETTINGS.raytracing.set(linear_color_space=False)
    frame = _ramp(20.0, 22.0)
    want = torch.linspace(20.0, 22.0, frame.shape[1], dtype=torch.float64)

    plain = _bytes(frame, kernel=kernel, dither=False)[..., 0].double().mean(0)
    dithered = _bytes(frame, kernel=kernel)[..., 0].double().mean(0)

    assert len(plain.unique()) == 3
    assert (plain - want).abs().max() > 0.45
    assert (dithered - want).abs().max() < 0.15


@pytest.mark.parametrize("linear", [True, False], ids=["linear", "display"])
@pytest.mark.parametrize(("impl", "kernel"), IMPLEMENTATIONS)
def test_every_authored_byte_survives_the_dither(impl, kernel, linear):
    """A flat authored colour lands on the byte it names, at every pixel.

    Under the linear space the value reaches the post stage decoded and comes
    back through the OETF a hair off its byte; the dither stops short of half
    a byte so that hair cannot tip it.
    """
    SETTINGS.raytracing.set(linear_color_space=linear)
    authored = torch.arange(256, dtype=torch.float64) / 255.0
    if linear:
        authored = torch.where(
            authored <= 0.04045,
            authored / 12.92,
            ((authored + 0.055) / 1.055) ** 2.4,
        )
    frame = authored.view(1, 256, 1).expand(64, 256, 3).float()
    got = _bytes(frame, kernel=kernel)
    assert torch.equal(got, torch.arange(256).view(1, 256, 1).expand(64, 256, 3))


@pytest.mark.parametrize(("impl", "kernel"), IMPLEMENTATIONS)
def test_the_dither_moves_no_value_by_more_than_one_byte(impl, kernel):
    SETTINGS.raytracing.set(linear_color_space=False)
    generator = torch.Generator().manual_seed(0)
    frame = torch.rand((48, 80, 3), generator=generator)
    plain = _bytes(frame, kernel=kernel, dither=False)
    dithered = _bytes(frame, kernel=kernel)
    assert (dithered - plain).abs().max() <= 1
    assert not torch.equal(dithered, plain)


def test_both_implementations_dither_alike():
    SETTINGS.raytracing.set(linear_color_space=True)
    frame = _ramp(3.0, 9.0, height=40, width=200)
    assert torch.equal(_bytes(frame, kernel=True), _bytes(frame, kernel=False))
