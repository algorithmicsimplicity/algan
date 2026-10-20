"""Each tonemapping choice, measured against the curve it is documented to be.

``SETTINGS.raytracing.tonemapping`` turns a curve on, ``tonemap_method`` picks
Khronos PBR Neutral (``"neutral"``) or AgX (``"agx"``), and
``tonemap_exposure`` scales the colour before either -- or before the plain
per-channel clip when the curve is off. These tests push known linear colours
through the post stage and compare the bytes with float64 transcriptions of
the published curves: the Khronos PBR Neutral reference, and AgX as three.js
ships it (Filament's constants, including the "Linearize" step a linear
pipeline needs before its sRGB OETF). The inputs include chromatic and
over-range colours, because a grey cannot see a transposed gamut matrix.

``test_tonemapping.py`` pins the neutral curve on greys and the agreement of
the two implementations; this file pins each curve itself, in both colour
working spaces and both shipping implementations (the standalone Taichi
kernel, and the torch pipeline it replaced), on the optional float16 frame
buffer as well as float32, and from ``SETTINGS`` through
``post_process_frames``.

A feature test, so it sits outside the fast suite: only a change to the
tonemap breaks it.
"""

from __future__ import annotations

import math

import pytest
import torch

from algan.rendering.post_processing.post_process import (
    _finalize_on_device,
    post_process_frames,
)
from algan.settings import SETTINGS
from algan.settings._startup import render_device
from algan.utils.memory_utils import ManualMemory

COLORS = [
    [0.0, 0.0, 0.0],
    [0.01, 0.01, 0.01],
    [0.18, 0.18, 0.18],  # mid grey
    [0.5, 0.5, 0.5],
    [1.0, 1.0, 1.0],
    [4.0, 4.0, 4.0],
    [1.0, 0.2, 0.05],  # saturated orange
    [0.1, 0.6, 0.3],
    [0.02, 0.9, 0.9],
    [0.05, 0.1, 2.5],  # over-range blue
    [3.0, 0.0, 0.0],  # over-range primary
]
MID_GREY = COLORS.index([0.18, 0.18, 0.18])


# -- the published curves, in float64 ------------------------------------------


def _oetf(v):
    """The sRGB OETF, from the specification."""
    v = min(max(v, 0.0), 1.0)
    return v * 12.92 if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055


def _clip(rgb):
    return [min(max(c, 0.0), 1.0) for c in rgb]


def _khronos_neutral(rgb):
    """Khronos PBR Neutral, transcribed from the Khronos reference GLSL."""
    start_compression = 0.8 - 0.04
    desaturation = 0.15
    x = min(rgb)
    offset = x - 6.25 * x * x if x < 0.08 else 0.04
    color = [c - offset for c in rgb]
    peak = max(color)
    if peak < start_compression:
        return color
    d = 1.0 - start_compression
    new_peak = 1.0 - d * d / (peak + d - start_compression)
    color = [c * new_peak / peak for c in color]
    g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0)
    return [c * (1.0 - g) + new_peak * g for c in color]


# three.js (``tonemapping_pars_fragment``) writes each GLSL ``mat3`` column by
# column; these are the rows. Every one maps white to white, which is what
# identifies the orientation -- see ``test_the_agx_reference_keeps_white``.
_SRGB_TO_REC2020 = (
    (0.6274, 0.3293, 0.0433),
    (0.0691, 0.9195, 0.0113),
    (0.0164, 0.0880, 0.8956),
)
_AGX_INSET = (
    (0.856627153315983, 0.0951212405381588, 0.0482516061458583),
    (0.137318972929847, 0.761241990602591, 0.101439036467562),
    (0.11189821299995, 0.0767994186031903, 0.811302368396859),
)
_AGX_OUTSET = (
    (1.1271005818144368, -0.11060664309660323, -0.016493938717834573),
    (-0.1413297634984383, 1.157823702216272, -0.016493938717834257),
    (-0.14132976349843826, -0.11060664309660294, 1.2519364065950405),
)
_REC2020_TO_SRGB = (
    (1.6605, -0.5876, -0.0728),
    (-0.1246, 1.1329, -0.0083),
    (-0.0182, -0.1006, 1.1187),
)
_AGX_MIN_EV = -12.47393
_AGX_MAX_EV = 4.026069


def _mat(m, v):
    return [sum(m[row][col] * v[col] for col in range(3)) for row in range(3)]


def _agx_contrast(x):
    x2 = x * x
    x4 = x2 * x2
    return (
        15.5 * x4 * x2
        - 40.14 * x4 * x
        + 31.96 * x4
        - 6.868 * x2 * x
        + 0.4298 * x2
        + 0.1191 * x
        - 0.00232
    )


def _agx(rgb, *, linearize):
    """AgX as three.js's ``AgXToneMapping``; ``linearize`` is its own step.

    three.js always linearizes, because its output always goes on to the
    sRGB OETF. Without that encode -- Algan's display-referred space writes the
    curve's output as bytes -- the step is left out.
    """
    c = _mat(_AGX_INSET, _mat(_SRGB_TO_REC2020, rgb))
    span = _AGX_MAX_EV - _AGX_MIN_EV
    c = [(math.log2(max(x, 1e-10)) - _AGX_MIN_EV) / span for x in c]
    c = [_agx_contrast(x) for x in _clip(c)]
    c = _mat(_AGX_OUTSET, c)
    if linearize:
        c = [max(x, 0.0) ** 2.2 for x in c]
    return _clip(_mat(_REC2020_TO_SRGB, c))


def _expected(rgb, *, curve, exposure, linear):
    """The bytes a linear colour should land on under one tonemap choice."""
    exposed = [c * exposure for c in rgb]
    if curve == "off":
        out = _clip(exposed)
    elif curve == "neutral":
        out = _clip(_khronos_neutral(exposed))
    else:
        out = _agx(exposed, linearize=linear)
    if linear:
        out = [_oetf(c) for c in out]
    return [math.floor(255 * c + 0.5) for c in out]


# -- driving the post stage ------------------------------------------------------


def _memory():
    return ManualMemory(0.0, device=render_device(), managed=False, num_bytes=1 << 22)


def _finalize(colors, *, curve, exposure=1.0, dtype=torch.float32):
    """Encode linear colours the way the post stage's last step does."""
    frame = torch.zeros((1, 1, len(colors), 4), dtype=torch.float32)
    frame[0, 0, :, :3] = torch.as_tensor(colors, dtype=torch.float32)
    frame = frame.to(device=render_device(), dtype=dtype)
    out = _finalize_on_device(
        frame,
        4,
        _memory(),
        tonemap_enabled=True,
        tonemapping=curve != "off",
        tonemap_method="neutral" if curve == "off" else curve,
        exposure=exposure,
    )
    assert out.dtype == torch.uint8
    assert tuple(out.shape) == (1, 1, len(colors), 3)
    return out[0, 0].cpu()


def _assert_curve(got, colors, **expectation):
    want = torch.tensor([_expected(c, **expectation) for c in colors])
    worst = (got.long() - want).abs().max()
    assert worst <= 1, f"{expectation}: got {got.tolist()}, want {want.tolist()}"


@pytest.fixture(autouse=True)
def _tonemap_defaults():
    """The shipped post stage: HDR buffer and the standalone kernel.

    Undithered, because these pin each curve to the byte and the dither moves
    a value that is not already a byte by one; ``test_dither.py`` covers it.
    Restored afterwards by the suite's settings isolation.
    """
    SETTINGS.raytracing.set(linear_color_space=True, dither=False)
    SETTINGS.raytracing.experimental.set(
        post_process_tonemap=True, post_tonemap_kernel=True
    )


@pytest.fixture
def implementation(request):
    """Select the finalize implementation for one test."""
    SETTINGS.raytracing.experimental.set(post_tonemap_kernel=request.param)
    return request.param


@pytest.fixture
def working_space(request):
    """Select the colour working space for one test."""
    SETTINGS.raytracing.set(linear_color_space=request.param)
    return request.param


IMPLEMENTED_BY = pytest.mark.parametrize(
    "implementation", [True, False], indirect=True, ids=["kernel", "torch"]
)
IN_EITHER_SPACE = pytest.mark.parametrize(
    "working_space", [True, False], indirect=True, ids=["linear", "display"]
)


@IN_EITHER_SPACE
@IMPLEMENTED_BY
@pytest.mark.parametrize("exposure", [1.0, 0.6])
def test_neutral_is_the_khronos_curve(working_space, implementation, exposure):
    """``"neutral"`` is Khronos PBR Neutral of the exposed colour.

    Chromatic and over-range inputs exercise the desaturation the reference
    applies above its compression start, which a grey never reaches with a
    hue to lose.
    """
    got = _finalize(COLORS, curve="neutral", exposure=exposure)
    _assert_curve(got, COLORS, curve="neutral", exposure=exposure, linear=working_space)


@IMPLEMENTED_BY
@pytest.mark.parametrize("exposure", [1.0, 0.6])
def test_agx_is_three_js_agx_under_the_linear_working_space(implementation, exposure):
    """REGRESSION. AgX's output was encoded twice in the linear working space.

    The AgX sigmoid produces display-encoded values (roughly a 2.2 power), and
    three.js and Filament decode them -- the "Linearize" step,
    ``pow(max(0, c), 2.2)`` -- before the gamut conversion, because the sRGB
    OETF follows. Algan applies that OETF too, last, but left the Linearize
    step out, in both implementations: every value washed out, and linear mid
    grey 0.18 landed on byte 187 instead of three.js's 128.
    """
    got = _finalize(COLORS, curve="agx", exposure=exposure)
    _assert_curve(got, COLORS, curve="agx", exposure=exposure, linear=True)
    if exposure == 1.0:
        assert got[MID_GREY].tolist() == [128, 128, 128]


def test_the_agx_reference_keeps_white():
    """The reference's own guard: every matrix row sums to one.

    Both spaces share the D65 white point, so each conversion maps white to
    white. A transposed matrix fails this, which is how the renderer once
    rendered grey as magenta in both of its implementations at once.
    """
    for matrix in (_SRGB_TO_REC2020, _AGX_INSET, _AGX_OUTSET, _REC2020_TO_SRGB):
        for row in matrix:
            assert sum(row) == pytest.approx(1.0, abs=2e-4)


@IMPLEMENTED_BY
def test_agx_in_the_display_referred_space_writes_the_curve_output(implementation):
    """With no OETF after it, the curve's display-encoded output is the byte.

    This arm never had the Linearize step and must not gain it.
    """
    SETTINGS.raytracing.set(linear_color_space=False)
    got = _finalize(COLORS, curve="agx")
    _assert_curve(got, COLORS, curve="agx", exposure=1.0, linear=False)
    assert got[MID_GREY].tolist() == [127, 127, 127]


@IN_EITHER_SPACE
@IMPLEMENTED_BY
@pytest.mark.parametrize("exposure", [1.0, 0.5, 1.5])
def test_off_clips_each_channel_after_exposure(working_space, implementation, exposure):
    """No curve: scale by the exposure, clip each channel to [0, 1], encode.

    An over-range channel clips on its own, so an over-range primary stays
    that primary and an over-range blue keeps its red and green.
    """
    got = _finalize(COLORS, curve="off", exposure=exposure)
    _assert_curve(got, COLORS, curve="off", exposure=exposure, linear=working_space)


def _f16_ramp():
    """Chromatic linear colours from 0 to 1.6, every one exact in float16."""
    ramp = torch.linspace(0.0, 1.6, 1601)
    return torch.stack([ramp, ramp.flip(0), ramp.roll(537)], dim=-1).half()


@pytest.mark.parametrize("curve", ["off", "neutral", "agx"])
def test_a_float16_frame_buffer_encodes_like_float32(curve):
    """REGRESSION. The torch tonemap computed in half precision on an f16 buffer.

    ``hdr_buffer_f16`` stores the frame buffer as float16 to halve it; the
    tonemap is meant to read it and compute in float32, as the Taichi kernel
    does. The torch pipeline (``post_tonemap_kernel=False``) widened to float32
    only for a premultiplied export, so every other f16 frame went through the
    exposure, the curve and the OETF in half precision and moved bytes -- by up
    to one with no curve, two under neutral and four under AgX -- against both
    the same values in float32 and the kernel.
    """
    colors = _f16_ramp()
    SETTINGS.raytracing.experimental.set(post_tonemap_kernel=False)
    half = _finalize(colors, curve=curve, exposure=0.9, dtype=torch.float16)
    single = _finalize(colors.float(), curve=curve, exposure=0.9)
    assert torch.equal(half, single)
    SETTINGS.raytracing.experimental.set(post_tonemap_kernel=True)
    kernel = _finalize(colors, curve=curve, exposure=0.9, dtype=torch.float16)
    assert (half.long() - kernel.long()).abs().max() <= 1


@pytest.mark.parametrize(
    ("tonemapping", "method", "exposure"),
    [(False, "neutral", 1.5), (True, "neutral", 0.8), (True, "agx", 0.8)],
    ids=["off", "neutral", "agx"],
)
def test_the_post_stage_takes_its_curve_from_settings(tonemapping, method, exposure):
    """``post_process_frames`` reads the tonemap choice from ``SETTINGS`` live.

    The composite hands over linear light in byte range; the expected bytes are
    the chosen curve of the exposed colour, encoded. The AgX case relies on the
    Linearize step that ``test_agx_is_three_js_agx_under_the_linear_working_space``
    pins.
    """
    SETTINGS.raytracing.set(
        tonemapping=tonemapping, tonemap_method=method, tonemap_exposure=exposure
    )
    frames = torch.zeros((1, 1, len(COLORS), 4))
    frames[0, 0, :, :3] = torch.tensor(COLORS) * 255
    out = post_process_frames(
        _memory(), frames.to(render_device()), 1, post_processes=()
    )
    assert out.dtype == torch.uint8
    assert tuple(out.shape) == (1, 1, len(COLORS), 3)
    curve = method if tonemapping else "off"
    _assert_curve(out[0, 0], COLORS, curve=curve, exposure=exposure, linear=True)
