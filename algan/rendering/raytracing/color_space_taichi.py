"""The sRGB transfer functions as Taichi ``@ti.func``s.

Kernel twins of ``algan.utils.color_space``; read that module's docstring for
why the working space is split at all. This one lives on its own rather than
inside a kernel module because both ends of the pipeline need it -- the tracer
kernels decode the background at the composite, and the post-process kernel
encodes at the byte write -- and a shared leaf module keeps that from becoming
an import cycle.

Kept deliberately free of any other import so it stays a leaf.
"""

import taichi as ti

# The sRGB knee, in encoded and in linear terms respectively.
SRGB_LINEAR_CUTOFF = 0.04045
LINEAR_SRGB_CUTOFF = 0.0031308


@ti.func
def srgb_to_linear_f(c: ti.f32) -> ti.f32:
    """Decode one display-referred channel to linear light.

    Clamped at zero before the power: ``pow`` of a negative base is NaN, and a
    linear value can arrive slightly negative from interpolation.
    """
    x = ti.max(c, 0.0)
    out = x / 12.92
    if x > SRGB_LINEAR_CUTOFF:
        out = ti.pow((x + 0.055) / 1.055, 2.4)
    return out


@ti.func
def linear_to_srgb_f(c: ti.f32) -> ti.f32:
    """Encode one linear-light channel to display-referred sRGB.

    Values above 1.0 are encoded rather than clamped -- clamping is the
    caller's decision, and a tonemap may have run first.
    """
    x = ti.max(c, 0.0)
    out = x * 12.92
    if x > LINEAR_SRGB_CUTOFF:
        out = 1.055 * ti.pow(x, 1.0 / 2.4) - 0.055
    return out


@ti.func
def srgb_to_linear_v3(c):
    """Decode an RGB triple to linear light."""
    return ti.math.vec3(
        srgb_to_linear_f(c[0]), srgb_to_linear_f(c[1]), srgb_to_linear_f(c[2])
    )


@ti.func
def linear_to_srgb_v3(c):
    """Encode an RGB triple to display-referred sRGB."""
    return ti.math.vec3(
        linear_to_srgb_f(c[0]), linear_to_srgb_f(c[1]), linear_to_srgb_f(c[2])
    )
