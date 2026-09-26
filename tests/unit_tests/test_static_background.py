"""A still image background prepares once per chunk, not once per frame.

``_prepare_background_for_chunk`` hands a single-frame image background over as
a :class:`~algan.rendering.raytracing.scene_builder._StaticImageBackground` --
one frame's rows standing for the whole chunk -- instead of a copy per frame.
Averaging it down and prefilling the frame buffer from it must produce exactly
the bytes the per-frame copies produce.
"""

from __future__ import annotations

import pytest
import torch

import algan.animation_timeline.timeline as timeline_module
from algan.render_loop import _prepare_background_for_chunk
from algan.rendering.raytracing import scene_builder
from algan.rendering.raytracing.scene_builder import (
    _downsample_background,
    _prefill_background,
    _StaticImageBackground,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _prepare(image, frames, aa, screen, disabled, device="cpu"):
    previous = timeline_module._OPT_DISABLED
    timeline_module._OPT_DISABLED = frozenset({"staticbg"} if disabled else ())
    try:
        return _prepare_background_for_chunk(
            image,
            screen_width=screen[0],
            screen_height=screen[1],
            anti_alias_level=aa,
            current_ind=3,
            new_ind=3 + frames,
            frames_per_second=1,
            device=torch.device(device),
        )
    finally:
        timeline_module._OPT_DISABLED = previous


@pytest.mark.parametrize("channels", [3, 4, 5])
@pytest.mark.parametrize("out_channels", [4, 5])
@pytest.mark.parametrize("aa", [1, 2])
@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_static_background_matches_the_per_frame_copies(
    channels, out_channels, aa, linear, device, monkeypatch
):
    monkeypatch.setattr(scene_builder.rt_settings, "linear_color_space", linear)
    frames, screen = 5, (7, 4)
    generator = torch.Generator().manual_seed(channels * 10 + aa)
    image = torch.rand(1, screen[1] * aa, screen[0] * aa, channels, generator=generator)
    # Exact endpoints, where rounding and clamping differ most.
    image[0, 0, 0] = 0.0
    image[0, -1, -1] = 1.0

    static = _prepare(image, frames, aa, screen, disabled=False, device=device)
    copies = _prepare(image, frames, aa, screen, disabled=True, device=device)
    assert isinstance(static, _StaticImageBackground)
    assert torch.is_tensor(copies)

    if aa > 1:
        static = _downsample_background(static, aa, frames, screen[1], screen[0])
        copies = _downsample_background(copies, aa, frames, screen[1], screen[0])
    dtype = torch.float32 if linear else torch.uint8
    pixels = screen[0] * screen[1]
    # Two sub-chunks of the chunk, as the tracer prefills its frame windows.
    for offset, count in ((0, 2), (2, 3)):
        expected = torch.full(
            (count, pixels, out_channels), 77, dtype=dtype, device=device
        )
        actual = expected.clone()
        _prefill_background(expected, copies, offset, device, background_frames=frames)
        _prefill_background(actual, static, offset, device, background_frames=frames)
        assert torch.equal(actual, expected)


def test_static_background_still_rejects_the_wrong_resolution():
    aa, screen, frames = 2, (3, 2), 4
    image = torch.rand(1, screen[1] * aa, screen[0] * aa, 4)
    static = _prepare(image, frames, aa, screen, disabled=False)
    out = torch.empty((frames, screen[0] * screen[1], 4), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="background resolution"):
        _prefill_background(out, static, 0, out.device, background_frames=frames)
