"""FXAA moves only the pixels it finds an edge at.

The filter resamples the frame through a grid: each pixel's own position plus
a sub-pixel offset along the edge it sits on, and no offset where there is no
edge. So a frame with no edge in it has to come back as it went in. It did
not: the base positions were spaced corner to corner (``linspace(-1, 1, W)``)
and then read with the pixel-centre convention, which put pixel ``i`` at
``i + i / (W - 1) - 0.5`` -- half a pixel out at either border, right only in
the middle -- so every FXAA render was softened and slightly stretched toward
its edges, whatever the edge detection decided.

"No offset where there is no edge" was a product with the edge mask, and off
an edge the offset is not always a number: in a flat BLACK neighbourhood the
blend is 0/0. ``nan * False`` is NaN, so those pixels were sampled at a NaN
coordinate -- which PyTorch's x86 sampler reads as coordinate 0, the frame's
top-left pixel, and its aarch64 one as NaN.

Tensor assertions on the filter itself: no Taichi, no render.
"""

from __future__ import annotations

import torch

from algan.rendering.post_processing.anti_aliasing.fxaa import fxaa
from algan.utils.memory_utils import ManualMemory


def _filter(image):
    memory = ManualMemory(0, device="cpu", num_bytes=64 << 20)
    return fxaa(image, memory=memory, luma_space="perceptual").clone()


def test_a_frame_with_no_edge_comes_back_unchanged():
    """A ramp whose neighbouring pixels differ by 0.01 of luma, far under the
    0.0625 minimum edge threshold, in both directions at once so that a wrong
    column and a wrong row both show. Wide and short on purpose: the error was
    largest where a dimension is small.
    """
    height, width = 8, 16
    ramp = torch.linspace(0.3, 0.45, width).view(1, 1, 1, width) + torch.linspace(
        0.0, 0.07, height
    ).view(1, 1, height, 1)
    image = ramp.expand(1, 3, height, width).contiguous()

    out = _filter(image)

    # Sampling at a pixel's own centre is exact up to the rounding of the
    # centre's coordinate; the defect moved these values by 0.004.
    assert (out - image).abs().max().item() < 1e-6


def test_a_faint_one_pixel_feature_stays_in_its_own_pixel():
    """One pixel 0.04 above a black frame -- under the edge threshold, so it
    is not FXAA's to blur -- near a corner, where the grid error was largest.
    It used to come out at under half its value, the rest smeared into the
    pixels beside it.
    """
    image = torch.zeros(1, 3, 8, 16)
    image[..., 1, 3] = 0.04

    out = _filter(image)

    assert out[0, 0, 1, 3].item() > 0.04 - 1e-6
    assert (out - image).abs().max().item() < 1e-6


def test_a_flat_black_area_is_not_painted_with_the_top_left_pixel():
    """A bright block in the top-left corner of a black frame. The black
    pixels clear of the block have no contrast and no luma, the 0/0 case, and
    used to come back as the block's colour (as NaN on aarch64). The block's
    own border is an edge and FXAA's to move, so it is left out.
    """
    image = torch.zeros(1, 3, 8, 16)
    image[..., :3, :3] = 0.9

    out = _filter(image)

    assert not torch.isnan(out).any()
    assert out[0, :, 5:, :].abs().max().item() == 0.0
    assert out[0, :, :, 5:].abs().max().item() == 0.0
