"""What the post stage hands the video writer: the supersample filter, the byte
layout and the row order.

``post_process_frames`` is the last thing a frame passes through before it is
encoded. It box-filters a supersampled buffer down to the output resolution
(``_supersample_box_filter``), finalises it to bytes (``_finalize_on_device``)
and hands it to the host top-down (``_frames_to_host``). These tests pin what a
user sees of that: a still rendered with ``supersampling`` 2 or 3 is the
high-resolution render of the same picture averaged block by block in linear
light; HDR headroom above the byte range survives the average and is clipped
only after it; an opaque frame comes back ``[frames, height, width, 3]`` uint8
and a transparent one ``[frames, height, width, 4]`` with its coverage
averaged as bytes; rows come back top-down; and the frames the host receives
are its own, not views of the render arena the next batch reuses.

Every expectation is computed from the specification -- the block mean and the
sRGB OETF as the standard writes it -- not from Algan's own helpers, so the
code under test cannot agree with itself on a wrong answer. The stills are
compared as PNGs because the video's encoder is lossy (``yuv420p`` at CRF 17);
both run the same post stage.

A feature test, so it sits outside the fast suite: only a change to the post
stage breaks it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from algan import Scene
from algan.rendering.post_processing.post_process import (
    _frames_to_host,
    _supersample_box_filter,
    post_process_frames,
)
from algan.settings import SETTINGS
from algan.settings._startup import render_device
from algan.settings.video_settings import SMOKE_TEST
from algan.utils.memory_utils import ManualMemory


def _encode(linear):
    """The byte a linear-light value lands on: clamp, the sRGB OETF, round."""
    v = linear.double().clamp(0.0, 1.0)
    encoded = torch.where(v <= 0.0031308, v * 12.92, 1.055 * v.pow(1 / 2.4) - 0.055)
    return torch.floor(encoded * 255 + 0.5)


def _quantize(display):
    """The byte a display-referred value lands on: clamp and round."""
    return torch.floor(display.double().clamp(0.0, 1.0) * 255 + 0.5)


def _block_mean(frames, level):
    """Average each ``level x level`` block of ``[N, H, W, C]``, in float64."""
    n, height, width, channels = frames.shape
    blocks = frames.double().reshape(
        n, height // level, level, width // level, level, channels
    )
    return blocks.mean(dim=(2, 4))


def _memory(managed=False):
    return ManualMemory(0.0, device=render_device(), managed=managed, num_bytes=1 << 22)


@pytest.fixture(autouse=True)
def _post_stage_defaults():
    """The shipped post stage: linear working space, HDR buffer, no curve.

    Undithered, because these compute each byte from the specification and
    the dither moves a value that is not already a byte by one;
    ``test_dither.py`` covers it. Restored afterwards by the suite's settings
    isolation.
    """
    SETTINGS.raytracing.set(
        linear_color_space=True, tonemapping=False, tonemap_exposure=1.0, dither=False
    )
    SETTINGS.raytracing.experimental.set(post_process_tonemap=True)


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


def _post(frames, level, **options):
    """Run frames through the post stage with no post-process passes."""
    return post_process_frames(_memory(), frames, level, post_processes=(), **options)


@pytest.mark.parametrize("level", [2, 3])
@pytest.mark.parametrize("arm", ["dispatched", "eager"])
def test_box_filter_is_the_mean_of_each_block(level, arm):
    """The supersample filter averages each block, at every level and offset.

    ``dispatched`` is the wrapper the pipeline calls -- compiled wherever
    ``torch.compile`` runs -- and ``eager`` is the function it falls back to.
    Content with no symmetry catches a filter that reads the wrong offsets.
    """
    generator = torch.Generator().manual_seed(level)
    frames = torch.rand((2, 3 * level, 2 * level, 4), generator=generator) * 600
    out = torch.empty((2, 3, 2, 4), dtype=torch.float32)
    box = (
        _supersample_box_filter
        if arm == "dispatched"
        else _supersample_box_filter.eager
    )
    box(out, frames, level)
    torch.testing.assert_close(
        out.double(), _block_mean(frames, level), rtol=1e-6, atol=1e-4
    )


@pytest.mark.parametrize("level", [2, 3])
def test_box_filter_averages_a_byte_buffer_without_wrapping(level):
    """The in-composite route hands the filter uint8 frames; it must widen them.

    Summing bytes in their own dtype would wrap at 256, so a block of 255s would
    average to garbage.
    """
    generator = torch.Generator().manual_seed(10 + level)
    frames = torch.randint(
        0, 256, (1, 2 * level, 2 * level, 4), generator=generator, dtype=torch.uint8
    )
    frames[0, :level, :level] = 255
    out = torch.empty((1, 2, 2, 4), dtype=torch.float32)
    _supersample_box_filter.eager(out, frames, level)
    torch.testing.assert_close(
        out.double(), _block_mean(frames, level), rtol=1e-6, atol=1e-4
    )
    assert torch.all(out[0, 0, 0] == 255)


@pytest.mark.parametrize(
    "working_space", [True, False], indirect=True, ids=["linear", "display"]
)
@pytest.mark.parametrize(
    "implementation", [True, False], indirect=True, ids=["kernel", "torch"]
)
def test_supersampling_averages_light_then_clips(working_space, implementation):
    """Half the samples lit is half the light, and over-range light is kept.

    The composite carries linear light in byte range, with headroom above 255.
    Under the linear working space half a pixel's samples at full white is
    linear 0.5, which encodes to 188 -- not the 128 an average of encoded bytes
    would give. One sample at twice white is the same light as two at white,
    so it lands on the same byte: the average runs before anything clips.
    """
    blocks = [
        (0, 255, 255, 0),  # half the samples lit
        (510, 0, 0, 0),  # one sample at twice white: the same light
        (1020, 1020, 1020, 1020),  # over range everywhere: clipped after
        (64, 64, 64, 64),  # flat: the filter is the identity
    ]
    frames = torch.zeros((1, 2, 2 * len(blocks), 4), device=render_device())
    for index, (a, b, c, d) in enumerate(blocks):
        block = torch.tensor([[a, b], [c, d]], dtype=torch.float32)
        frames[0, :, 2 * index : 2 * index + 2, :3] = block[..., None]

    out = _post(frames, 2)

    assert out.dtype == torch.uint8
    assert tuple(out.shape) == (1, 1, len(blocks), 3)
    means = torch.tensor(
        [sum(block) / 4 / 255 for block in blocks], dtype=torch.float64
    )
    want = _encode(means) if working_space else _quantize(means)
    for channel in range(3):
        assert out[0, 0, :, channel].tolist() == want.tolist()
    if working_space:
        assert out[0, 0, 0, 0] == 188
        assert out[0, 0, 1, 0] == 188


@pytest.mark.parametrize("level", [2, 3])
@pytest.mark.parametrize(
    "implementation", [True, False], indirect=True, ids=["kernel", "torch"]
)
def test_supersampled_frames_are_the_encoded_block_means(level, implementation):
    """Any content, at levels 2 and 3, over several frames and HDR values.

    The expectation is the float64 block mean, normalised, clipped and encoded
    with the OETF, then flipped to the top-down order the host receives; the
    glow channel is not part of the output. Within one byte, for the float32
    arithmetic the pipeline does it in.
    """
    generator = torch.Generator().manual_seed(level)
    frames = (torch.rand((3, 2 * level, 4 * level, 4), generator=generator) * 400).to(
        render_device()
    )
    expected = _encode(_block_mean(frames.cpu(), level)[..., :3] / 255).flip(1)

    out = _post(frames, level)

    assert out.dtype == torch.uint8
    assert tuple(out.shape) == (3, 2, 4, 3)
    assert (out.double() - expected).abs().max() <= 1


@pytest.mark.parametrize(
    "implementation", [True, False], indirect=True, ids=["kernel", "torch"]
)
def test_transparent_supersampled_frames_average_coverage_as_bytes(implementation):
    """The alpha lane is byte coverage: averaged as it is, never normalised.

    The colour lanes arrive premultiplied by that coverage, and the OETF is not
    linear, so the encoder unpremultiplies, encodes and premultiplies again. A
    linear 0.2 at coverages 200, 100, 0, 100 averages to coverage 100 and
    encodes to ``OETF(0.2) * 100/255`` -- encoding the premultiplied mean
    directly would give a byte some 30 higher.
    """
    coverage = torch.tensor([[200.0, 100.0], [0.0, 100.0]])
    frames = torch.zeros((1, 2, 2, 5), device=render_device())
    frames[0, :, :, :3] = (0.2 * coverage)[..., None]
    frames[0, :, :, 3] = 37.0  # glow: dropped from the output
    frames[0, :, :, 4] = coverage

    out = _post(frames, 2)

    assert out.dtype == torch.uint8
    assert tuple(out.shape) == (1, 1, 1, 4)
    alpha = 100 / 255
    color = torch.floor(
        255
        * alpha
        * (1.055 * torch.tensor(0.2, dtype=torch.float64) ** (1 / 2.4) - 0.055)
        + 0.5
    )
    assert out[0, 0, 0].tolist() == [int(color)] * 3 + [100]
    naive = _encode(torch.tensor(0.2 * 100 / 255))
    assert int(naive) - int(color) >= 25


def test_host_frames_are_top_down_and_survive_the_arena_reuse():
    """Rows come back top-down, and the host owns what it receives.

    The composite writes rows bottom-up. The video writer may still be encoding
    a batch after the next render has reused the arena, so the frames handed
    over must not be views of it.
    """
    memory = _memory(managed=True)
    frames = memory.get_tensor((2, 3, 2, 4), torch.float32)
    frames.zero_()
    for row, channel in enumerate((0, 1, 2)):  # bottom red, middle green, top blue
        frames[:, row, :, channel] = 255.0
    frames[1] *= 0.5  # the second frame is dimmer, so frame order is visible

    out = post_process_frames(memory, frames, 1, post_processes=())

    assert out.device.type == "cpu"
    assert out.is_contiguous()
    top, middle, bottom = out[0, 0, 0], out[0, 1, 0], out[0, 2, 0]
    assert top.tolist() == [0, 0, 255]
    assert middle.tolist() == [0, 255, 0]
    assert bottom.tolist() == [255, 0, 0]
    assert int(out[1, 0, 0, 2]) == int(_encode(torch.tensor(0.5)))
    before = out.clone()
    memory.data.fill_(0xAB)
    assert torch.equal(out, before)


def test_frames_to_host_reverses_rows_only():
    """The flip is of rows: frames and columns keep their order."""
    frames = torch.arange(2 * 3 * 4 * 3, dtype=torch.int32).reshape(2, 3, 4, 3)
    frames = frames.to(torch.uint8).to(render_device())
    out = _frames_to_host(frames)
    assert out.device.type == "cpu"
    assert out.dtype == torch.uint8
    for n in range(2):
        for row in range(3):
            assert torch.equal(out[n, row], frames[n, 2 - row].cpu())
    frames.zero_()
    assert out.any()


# -- real renders ---------------------------------------------------------------


def _pattern_bytes(width, height):
    """The test background's bytes, ``[height, width, 3]`` top-down.

    No symmetry in either axis or between channels, so a misaligned block, a
    flipped axis or a swapped channel all show.
    """
    ix = torch.arange(width).view(1, -1)
    iy = torch.arange(height).view(-1, 1)
    red = (ix * 37 + iy * 101) % 256
    green = (ix * 11 + iy * 59 + 7) % 256
    blue = (ix * iy * 13 + 3) % 256
    # ``y`` is 0 at the bottom of the frame, so the last row is iy = 0.
    return torch.stack([red, green, blue], dim=-1).flip(0).double()


def _pattern(x, y, time):
    """The same pattern as a procedural background, on the sample grid it is given."""
    width, height = x.shape[1], y.shape[0]
    ix = (x * width).round()
    iy = (y * height).round()
    red = ((ix * 37 + iy * 101) % 256) / 255
    green = ((ix * 11 + iy * 59 + 7) % 256) / 255
    blue = ((ix * iy * 13 + 3) % 256) / 255
    channels = [c.expand(height, width, 1) for c in (red, green, blue)]
    return torch.cat(channels, dim=-1) * torch.ones_like(time)


def _still(path, settings):
    with Scene(video_settings=settings) as scene:
        result = scene.save_frame(path, background=_pattern, post_processes=())
    assert result.status == "rendered"
    with Image.open(result.output_path) as image:
        assert image.mode == "RGB"
        return torch.from_numpy(np.asarray(image, dtype=np.float64).copy())


@pytest.mark.parametrize("level", [2, 3])
def test_a_supersampled_still_is_its_high_resolution_still_box_filtered(
    tmp_path, level
):
    """``supersampling=k`` at ``(w, h)`` is the ``(k*w, k*h)`` render, averaged.

    The background is evaluated on the supersampled grid, so the high
    resolution still holds exactly the samples the supersampled one averages.
    Its bytes are the pattern's own -- authored colour round-trips -- and every
    pixel of the supersampled still is the mean of its block taken in linear
    light, then encoded. An average of the encoded bytes misses by tens.
    """
    width, height = 4, 3
    low = _still(
        tmp_path / "low.png",
        SMOKE_TEST.set(resolution=(width, height), supersampling=level),
    )
    high = _still(
        tmp_path / "high.png",
        SMOKE_TEST.set(resolution=(width * level, height * level), supersampling=1),
    )

    assert torch.equal(high, _pattern_bytes(width * level, height * level))
    c = high / 255
    linear = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    blocks = linear.reshape(height, level, width, level, 3).mean(dim=(1, 3))
    assert (low - _encode(blocks)).abs().max() <= 1

    display_average = torch.floor(
        high.reshape(height, level, width, level, 3).mean(dim=(1, 3)) + 0.5
    )
    assert (low - display_average).abs().max() >= 10
