"""The sRGB transfer functions, checked against the standard and each other.

The order of these checks is deliberate. Every anchor is asserted against the
sRGB specification's own arithmetic first, and only then are the torch and
Taichi implementations compared with each other. Agreement between two
implementations is not evidence of correctness -- the AgX output matrix shipped
transposed in *both* of its implementations and they agreed perfectly, which is
how a neutral grey came to render as saturated magenta (``TONEMAP_FINDINGS.md``).
An external invariant is the only kind that would have caught it.
"""

# Deliberately no ``from __future__ import annotations``: this module defines a
# ``@ti.kernel``, and the future import turns its runtime-evaluated
# ``ti.types.ndarray()`` annotations into strings, which Taichi rejects at
# compile time. It is the same hazard CLAUDE.md records for ``*_taichi.py``
# files (ruff's ``I002`` is disabled there for exactly this reason) -- it
# applies to any module with a kernel in it, tests included.

import numpy as np
import pytest
import torch

from algan.utils.color_space import (
    LINEAR_SRGB_CUTOFF,
    SRGB_LINEAR_CUTOFF,
    linear_to_srgb,
    srgb_to_linear,
)

# Values spanning both segments of the piecewise curve, its knee, and outside
# the nominal range in both directions.
PROBE = [0.0, 0.001, 0.01, 0.04045, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 8.0]


def _reference_to_linear(c):
    """The sRGB EOTF, written out from the specification."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _reference_to_srgb(c):
    """The sRGB OETF, written out from the specification."""
    return c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def test_matches_the_srgb_specification():
    """Both directions reproduce the standard's own arithmetic."""
    got = srgb_to_linear(torch.tensor(PROBE, dtype=torch.float64))
    want = torch.tensor([_reference_to_linear(c) for c in PROBE], dtype=torch.float64)
    assert torch.allclose(got, want, atol=1e-12)

    got = linear_to_srgb(torch.tensor(PROBE, dtype=torch.float64))
    want = torch.tensor([_reference_to_srgb(c) for c in PROBE], dtype=torch.float64)
    assert torch.allclose(got, want, atol=1e-12)


def test_published_anchor_values():
    """The two numbers anyone can look up, to five decimals."""
    assert srgb_to_linear(torch.tensor([0.5])).item() == pytest.approx(
        0.21404, abs=1e-5
    )
    assert linear_to_srgb(torch.tensor([0.5])).item() == pytest.approx(
        0.73536, abs=1e-5
    )


def test_endpoints_are_exact():
    """Black stays black and white stays white, in both directions.

    White in particular is load-bearing: the acceptance gate for the linear
    working space is that an authored 255 still renders 255, and that fails at
    the first decimal if either endpoint drifts.
    """
    for fn in (srgb_to_linear, linear_to_srgb):
        assert fn(torch.tensor([0.0])).item() == 0.0
        assert fn(torch.tensor([1.0])).item() == pytest.approx(1.0, abs=1e-7)


def test_round_trip_is_the_identity():
    """Decode-then-encode returns the input.

    This is the property the whole design rests on: unlit flat content passes
    through both conversions with no arithmetic in between, so it must come out
    exactly as it went in.
    """
    x = torch.linspace(0.0, 1.0, 4097, dtype=torch.float64)
    assert torch.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-9)
    assert torch.allclose(srgb_to_linear(linear_to_srgb(x)), x, atol=1e-9)


def test_round_trip_survives_byte_quantisation():
    """Every one of the 256 encodable bytes survives a decode/encode round trip.

    Stronger than the continuous check above and closer to what the renderer
    actually does to a flat fill.
    """
    bytes_in = torch.arange(256, dtype=torch.float64) / 255.0
    out = linear_to_srgb(srgb_to_linear(bytes_in))
    assert torch.equal((out * 255).round(), (bytes_in * 255).round())


def test_monotonic():
    """Both directions are strictly increasing, so neither inverts an ordering."""
    x = torch.linspace(0.0, 4.0, 2001, dtype=torch.float64)
    for fn in (srgb_to_linear, linear_to_srgb):
        assert torch.all(torch.diff(fn(x)) > 0)


def test_negative_and_over_range_do_not_produce_nan():
    """``pow`` of a negative base is NaN, and one such pixel poisons a frame.

    A linear value can arrive slightly negative from interpolation or a filter
    kernel, so both directions clamp before the power.
    """
    x = torch.tensor([-1.0, -0.05, -1e-9, 0.0, 1e-9, 4.0, 1e4])
    for fn in (srgb_to_linear, linear_to_srgb):
        out = fn(x)
        assert not torch.isnan(out).any()
        assert torch.all(out[:3] == 0.0)


def test_computes_in_at_least_float32_but_preserves_dtype():
    """float16 buffers must not compute the power in half precision.

    The HDR buffer is float16 under ``ALGAN_HDR_BUFFER_F16=1`` while the Taichi
    kernels are always f32, so computing in the buffer's dtype would make the
    two paths disagree.
    """
    x = torch.linspace(0.01, 1.0, 64)
    half = linear_to_srgb(x.half())
    assert half.dtype == torch.float16
    # Within half precision's own resolution of the f32 answer.
    assert torch.allclose(half.float(), linear_to_srgb(x), atol=1e-3)


def test_cutoffs_are_the_standard_ones():
    assert SRGB_LINEAR_CUTOFF == 0.04045
    assert LINEAR_SRGB_CUTOFF == 0.0031308


def test_taichi_twins_agree_with_torch():
    """The kernel implementations match the torch ones.

    Runs last, and deliberately so: it is a consistency check between two
    implementations, which is only meaningful once the checks above have
    established that the torch one is right.
    """
    ti = pytest.importorskip("taichi")
    from algan.rendering.raytracing.color_space_taichi import (
        linear_to_srgb_f,
        srgb_to_linear_f,
    )

    ti.init(arch=ti.cpu)

    probe = np.array(PROBE, dtype=np.float32)
    decoded = np.zeros_like(probe)
    encoded = np.zeros_like(probe)

    @ti.kernel
    def run(x: ti.types.ndarray(), d: ti.types.ndarray(), e: ti.types.ndarray()):
        for i in range(x.shape[0]):
            d[i] = srgb_to_linear_f(x[i])
            e[i] = linear_to_srgb_f(x[i])

    run(probe, decoded, encoded)

    torch_decoded = srgb_to_linear(torch.from_numpy(probe)).numpy()
    torch_encoded = linear_to_srgb(torch.from_numpy(probe)).numpy()

    assert np.allclose(decoded, torch_decoded, atol=1e-6)
    assert np.allclose(encoded, torch_encoded, atol=1e-6)
