"""Transparent output: the premultiplied background contract, and the codec
each container needs to carry alpha.

Both regressions here were silent rather than loud. The background one shipped
a picture that was merely *wrong* -- a half-opaque background came out too
bright, and only at partial opacity, so a fully transparent or fully opaque
scene looked perfect. The WebM one never produced a file at all: the codec was
chosen without consulting the container, so FFmpeg was handed a PNG track to
put in a WebM and the render hung after paying for every frame.
"""

from __future__ import annotations

import pytest
import torch

from algan.errors import AlganConfigurationError
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.scene_builder import _prefill_background
from algan.utils.algan_utils import (
    _TRANSPARENT_CODECS,
    _check_transparent_container_is_supported,
)


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _prefill_solid(color, *, channels, linear, monkeypatch):
    """Prefill one pixel with a solid background and hand back the row."""
    monkeypatch.setattr(rt_settings, "linear_color_space", linear)
    dtype = torch.float32 if linear else torch.uint8
    out = torch.zeros((1, 1, channels), dtype=dtype)
    _prefill_background(out, torch.tensor(color), frame_offset=0, device=out.device)
    return out[0, 0].tolist()


@pytest.mark.fast
@pytest.mark.parametrize("opacity", [0.25, 0.5, 0.75])
def test_transparent_background_color_is_premultiplied_by_its_own_alpha(
    opacity, monkeypatch
):
    """Color must arrive premultiplied, because the tonemap divides alpha out.

    The composite adds ``weight * bg`` to geometry that already carries its own
    coverage, and the tonemap then unpremultiplies before encoding. A raw color
    here survived that round trip as ``encode(color / a) * a`` -- brighter the
    more transparent the background was, which is the opposite of correct.
    """
    red, glow = 0.2, 0.0
    row = _prefill_solid(
        [red, 0.0, 0.0, glow, opacity],
        channels=5,
        linear=True,
        monkeypatch=monkeypatch,
    )

    expected = _srgb_to_linear(red) * opacity * 255
    assert row[0] == pytest.approx(expected, abs=1e-3)
    # Alpha itself is carried, not folded away, and the other channels are
    # untouched: only color is premultiplied.
    assert row[4] == pytest.approx(opacity * 255, abs=1e-3)
    assert row[1] == row[2] == 0.0


@pytest.mark.fast
def test_premultiply_is_linear_in_alpha(monkeypatch):
    """Halving the background's alpha halves the color it stores.

    Stated without reference to the transfer function: this is what
    "premultiplied" *means*, and it is exactly what the bug broke -- the stored
    value used to grow sublinearly, landing 36/255 where 25.5 was correct.
    """
    full = _prefill_solid(
        [0.2, 0.0, 0.0, 0.0, 1.0], channels=5, linear=True, monkeypatch=monkeypatch
    )
    half = _prefill_solid(
        [0.2, 0.0, 0.0, 0.0, 0.5], channels=5, linear=True, monkeypatch=monkeypatch
    )
    assert half[0] == pytest.approx(full[0] * 0.5, abs=1e-3)


@pytest.mark.fast
def test_encoded_buffer_premultiplies_the_encoded_value(monkeypatch):
    """The 8-bit buffer holds encoded color, so alpha multiplies that.

    Premultiplying across the transfer function instead of after it is the
    subtle way to get this wrong, and it would not show up in the linear path.
    """
    row = _prefill_solid(
        [0.4, 0.0, 0.0, 0.0, 0.5], channels=5, linear=False, monkeypatch=monkeypatch
    )
    assert row[0] == pytest.approx(round(0.4 * 0.5 * 255), abs=1)


@pytest.mark.fast
def test_opaque_render_is_untouched_by_the_premultiply(monkeypatch):
    """A 4-channel render has no alpha channel to premultiply against.

    The guard that matters for every existing baseline: opaque output must not
    move by so much as a bit.
    """
    row = _prefill_solid(
        [0.2, 0.5, 0.7, 0.0, 1.0], channels=4, linear=True, monkeypatch=monkeypatch
    )
    for got, authored in zip(row[:3], (0.2, 0.5, 0.7)):
        assert got == pytest.approx(_srgb_to_linear(authored) * 255, abs=1e-3)


@pytest.mark.fast
def test_fully_transparent_background_contributes_no_color(monkeypatch):
    """Alpha 0 means the background's color cannot leak into the composite.

    ``TRANSPARENT`` is black, so this was invisible with the built-in constant;
    a transparent *red* background used to bleed red into anti-aliased edges.
    """
    row = _prefill_solid(
        [1.0, 0.0, 0.0, 0.0, 0.0], channels=5, linear=True, monkeypatch=monkeypatch
    )
    assert row[:3] == [0.0, 0.0, 0.0]


@pytest.mark.fast
def test_image_background_is_premultiplied_per_pixel(monkeypatch):
    """The image path owes the composite the same contract as a solid color."""
    monkeypatch.setattr(rt_settings, "linear_color_space", False)
    # A leading padding row, then two pixels: opaque white, then half-opaque.
    rows = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [255, 255, 255, 0, 255],
            [255, 255, 255, 0, 128],
        ],
        dtype=torch.uint8,
    )
    out = torch.zeros((1, 2, 5), dtype=torch.uint8)
    _prefill_background(out, rows, frame_offset=0, device=out.device)

    assert out[0, 0, 0].item() == 255
    assert out[0, 1, 0].item() == pytest.approx(128, abs=1)
    assert out[0, 1, 4].item() == 128


def test_every_alpha_container_names_a_codec_that_can_carry_alpha():
    """The map is the fix: a codec is never chosen blind to its container."""
    assert _TRANSPARENT_CODECS[".mov"][0] == "png"
    codec, params = _TRANSPARENT_CODECS[".webm"]
    assert codec == "libvpx-vp9"
    # FFmpeg emits VP9's alpha layer only when the *output* pixel format asks
    # for it; without this the alpha is dropped silently, not loudly.
    assert "yuva420p" in params
    assert ".mp4" not in _TRANSPARENT_CODECS


@pytest.mark.parametrize("suffix", [".mov", ".webm", ".mkv", ".avi"])
def test_containers_that_can_hold_alpha_are_accepted(suffix):
    _check_transparent_container_is_supported(f"scene{suffix}")


@pytest.mark.parametrize("suffix", [".mp4", ".gif"])
def test_containers_that_cannot_hold_alpha_are_refused_by_name(suffix):
    """Refused up front, and naming the format: the alternative is a hang."""
    with pytest.raises(AlganConfigurationError, match=suffix.lstrip(".").upper()):
        _check_transparent_container_is_supported(f"scene{suffix}")
