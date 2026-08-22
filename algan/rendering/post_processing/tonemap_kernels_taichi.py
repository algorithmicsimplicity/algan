"""Standalone post-process tonemap kernel.

Under post-process tonemapping the frame reaches ``_finalize_on_device`` as a
linear-HDR float buffer that bloom has already run on (the physically-correct
bloom-before-tonemap order). The tonemap itself was a torch pipeline
(``_neutral_tonemap``), ~20 elementwise ops per pixel over every frame -- the
dominant cost of the move to post-process tonemapping (~2.3s on an empty MD
render). This kernel does the same tonemap + quantize in one Taichi pass,
reusing the exact ``pbr_neutral_tonemap`` / ``agx_tonemap`` ti.funcs the
in-composite path used, so it computes in f32 (fast even where the buffer is
f16) and keeps the tonemap a post-processing step -- just implemented on the
GPU kernel side.

NOTE: filename ends in ``_taichi`` so ruff never injects
``from __future__ import annotations`` (which breaks Taichi kernel
compilation).
"""

import taichi as ti

from algan.rendering.raytracing.color_space_taichi import linear_to_srgb_v3
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    agx_tonemap,
    pbr_neutral_tonemap,
)


@ti.kernel
def tonemap_to_u8(frame: ti.types.ndarray(), out: ti.types.ndarray(),
                  method: ti.template(), exposure: ti.f32,
                  transparent: ti.template(), linear_color: ti.template()):
    """Tonemap a linear-HDR frame (channels 0-2 in [0, 1+HDR]) to uint8.

    ``frame`` is ``[N, H, W, C]`` float (C = 4 opaque [R,G,B,glow] or 5
    transparent [R,G,B,glow,alpha]); ``out`` is ``[N, H, W, 3]`` (opaque) or
    ``[N, H, W, 4]`` (transparent RGBA) uint8. The glow channel (3) is
    dropped. ``method``: 0 = clamp only, 1 = neutral, 2 = AgX. Matches
    ``finalize_pixel_color`` / the torch ``_neutral_tonemap`` arithmetic
    (tonemap the exposure-scaled 0-1 colour, then ``*255 + 0.5`` round).
    Alpha is already byte-range (never normalised) so it is only clamped.
    """
    for f, y, x in ti.ndrange(frame.shape[0], frame.shape[1], frame.shape[2]):
        c = ti.math.vec3(frame[f, y, x, 0], frame[f, y, x, 1],
                         frame[f, y, x, 2])
        if ti.static(method == 1):
            c = pbr_neutral_tonemap(c * exposure)
        elif ti.static(method == 2):
            c = agx_tonemap(c * exposure)
        else:
            # Exposure applies here too, not just under a curve: it is the
            # documented "the whole scene is too dark" control, and with
            # tonemapping off (the default) it is the only one. Exact at the
            # default exposure of 1.0, so this moves no pixel by itself.
            c = ti.math.clamp(c * exposure, 0.0, 1.0)
        if ti.static(linear_color):
            # The OETF, last, after exposure and after any curve -- the order
            # three.js uses (`tonemapping_fragment` then `colorspace_fragment`).
            if ti.static(transparent):
                # RGB arrives premultiplied by coverage while alpha is carried
                # separately in channel 4, and the transfer function is not
                # linear, so encoding the premultiplied value is wrong: a
                # half-covered white pixel would store encode(0.5) = 188 where
                # the premultiplied-correct answer is 0.5 * encode(1.0) = 127.
                # Unpremultiply, encode, re-premultiply.
                a = ti.math.clamp(frame[f, y, x, 4] * (1.0 / 255.0), 0.0, 1.0)
                if a > 1e-6:
                    c = linear_to_srgb_v3(c * (1.0 / a)) * a
                else:
                    # Fully transparent: nothing is visible through it, and
                    # dividing by ~0 would blow the colour up before the
                    # multiply brought it back.
                    c = ti.math.vec3(0.0, 0.0, 0.0)
            else:
                c = linear_to_srgb_v3(c)
        for ci in ti.static(range(3)):
            out[f, y, x, ci] = ti.cast(
                ti.math.clamp(c[ci] * 255.0 + 0.5, 0.0, 255.0), ti.u8)
        if ti.static(transparent):
            out[f, y, x, 3] = ti.cast(
                ti.math.clamp(frame[f, y, x, 4], 0.0, 255.0), ti.u8)
