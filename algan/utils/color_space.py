"""The sRGB transfer functions, and the working-space split they implement.

Algan authors color **display-referred**: ``RED = (255, 0, 0)`` names a pixel,
not a radiance, and a flat fill of it must come back out of the encoder as the
bytes the user typed. Light, on the other hand, only adds up correctly in
**linear** light -- sRGB encoding is concave, so ``encode(a) + encode(b)`` is far
more than ``encode(a + b)``, and summing two encoded halves overshoots instead
of landing on the encoded sum.

Those two facts are reconciled the way every other renderer reconciles them, and
the way three.js does specifically: decode authored color into a linear working
space at the render boundary, do every arithmetic operation there, and apply the
OETF once at the final byte write. Unlit flat content passes through decode and
then encode with no arithmetic in between, which is the identity, so it is
untouched; only pixels that were actually *computed* move.

These are the exact piecewise sRGB functions, not a gamma-2.2 approximation. The
difference is not cosmetic near black: the linear segment below the knee is what
keeps dark values from being crushed, and a 2.2 power would put an authored 26
on a visibly different byte.

Both directions clamp negatives to zero before the power. A linear value can go
slightly negative through interpolation or a filter kernel, and ``pow`` of a
negative base is NaN -- one such pixel poisons a whole frame's statistics.

The Taichi twins live in
``algan.rendering.raytracing.color_space_taichi`` and must be kept in step;
``tests/unit_tests/test_color_space.py`` checks the two against each other *and*
against the standard's own constants, because two implementations agreeing
proves nothing if both are wrong (see ``TONEMAP_FINDINGS.md`` on the AgX matrix
that was transposed in both of its implementations).
"""

from __future__ import annotations

import torch

#: Below this encoded value the sRGB curve is a straight line, not a power.
SRGB_LINEAR_CUTOFF = 0.04045
#: The same knee, expressed in linear light.
LINEAR_SRGB_CUTOFF = 0.0031308


def _as_float(t):
    """Compute in at least float32.

    The HDR frame buffer is float16 under ``ALGAN_HDR_BUFFER_F16=1``, and
    ``pow(x, 1/2.4)`` in half precision loses enough mantissa to shift a byte.
    The Taichi kernels are always f32, so computing in the buffer's dtype would
    also make the torch path and the kernel path disagree -- the same class of
    divergence an earlier audit found in the tonemap.
    """
    if t.dtype in (torch.float32, torch.float64):
        return t, None
    return t.float(), t.dtype


def srgb_to_linear(c):
    """Decode display-referred sRGB values to linear light.

    Parameters
    ----------
    c
        Tensor of sRGB-encoded values, nominally in ``[0, 1]``.

    Returns
    -------
    torch.Tensor
        The same shape in linear light, in the input's dtype.
    """
    c, restore = _as_float(c)
    c = c.clamp_min(0.0)
    out = torch.where(
        c <= SRGB_LINEAR_CUTOFF,
        c / 12.92,
        ((c + 0.055) / 1.055) ** 2.4,
    )
    return out if restore is None else out.to(restore)


def linear_to_srgb(c):
    """Encode linear light to display-referred sRGB values.

    Parameters
    ----------
    c
        Tensor of linear-light values. Values above 1.0 are encoded, not
        clamped -- clamping is the caller's decision, and a tonemap may run
        first.

    Returns
    -------
    torch.Tensor
        The same shape, sRGB-encoded, in the input's dtype.
    """
    c, restore = _as_float(c)
    c = c.clamp_min(0.0)
    out = torch.where(
        c <= LINEAR_SRGB_CUTOFF,
        c * 12.92,
        1.055 * c ** (1.0 / 2.4) - 0.055,
    )
    return out if restore is None else out.to(restore)
