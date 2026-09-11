import pytest
import torch

from algan.render_loop import _prepare_background_for_chunk
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.scene_builder import _prefill_background
from algan.scene import Scene
from algan.settings.video_settings import SMOKE_TEST
from algan.taichi_compat import ti


@ti.func
def _coordinate_background(x, y, time):
    return ti.Vector([x, y, time, 1.0])


@ti.func
def _rgb_background(x, y, time):
    return ti.Vector([0.25, 0.5, 0.75])


def _reference_to_linear(c):
    """The sRGB EOTF from the specification, not from Algan's transcription."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


@pytest.fixture
def linear_space():
    """Render in the linear working space, on the float HDR buffer it needs."""
    previous = rt_settings.linear_color_space
    rt_settings.set_linear_color_space(True)
    try:
        yield
    finally:
        rt_settings.set_linear_color_space(previous)


def _deferred_background(callback, *, width=3, height=2, aa=1, first_frame=2):
    return _prepare_background_for_chunk(
        callback,
        screen_width=width,
        screen_height=height,
        anti_alias_level=aa,
        current_ind=first_frame,
        new_ind=first_frame + 3,
        frames_per_second=10,
        device=torch.device("cpu"),
    )


def test_taichi_background_fills_the_whole_frame_batch():
    deferred = _deferred_background(_coordinate_background)
    assert deferred.is_taichi_func

    result = torch.empty((3, 6, 4), dtype=torch.uint8)
    _prefill_background(result, deferred, frame_offset=1, device=result.device)

    x = torch.arange(3, dtype=torch.float32).view(1, 1, 3, 1) / 3
    y = torch.arange(2, dtype=torch.float32).view(1, 2, 1, 1) / 2
    time = torch.arange(3, 6, dtype=torch.float32).view(3, 1, 1, 1) / 10
    expected = torch.cat(
        (
            x.expand(3, 2, 3, 1),
            y.expand(3, 2, 3, 1),
            time.expand(3, 2, 3, 1),
            torch.ones((3, 2, 3, 1)),
        ),
        dim=-1,
    )
    expected = torch.floor(expected * 255 + 0.5).to(torch.uint8).view(3, 6, 4)

    assert torch.equal(result, expected)


def test_scene_constructor_defers_taichi_background_until_render():
    scene = Scene(
        background=_coordinate_background,
        video_settings=SMOKE_TEST,
    )

    assert scene.background_frame is _coordinate_background
    assert scene.background_is_transparent() is False


def test_taichi_background_averages_aa_samples_without_an_intermediate():
    deferred = _deferred_background(
        _coordinate_background, width=2, height=1, aa=2, first_frame=0
    )
    result = torch.empty((1, 2, 4), dtype=torch.uint8)

    _prefill_background(result, deferred, frame_offset=0, device=result.device)

    # The procedural background is evaluated at the four supersampled
    # coordinates for each output pixel. Each sample is quantized before the
    # average, matching the existing Torch-callable background path.
    expected = torch.tensor([[[32, 64, 0, 255], [160, 64, 0, 255]]], dtype=torch.uint8)
    assert torch.equal(result, expected)


def test_torch_background_time_is_absolute_across_render_chunks():
    observed = []

    def background(x, y, time):
        observed.append(time.clone())
        return (x + 0 * y + time).expand(-1, -1, -1, 4)

    deferred = _deferred_background(background, width=1, height=1, first_frame=5)
    result = torch.empty((2, 1, 4), dtype=torch.uint8)
    _prefill_background(result, deferred, frame_offset=2, device=result.device)

    assert [time.item() for time in observed] == pytest.approx([0.7, 0.8])


def _rgb_callback(kind):
    """The same flat ``(0.25, 0.5, 0.75)`` background, written both ways."""
    if kind == "taichi":
        return _rgb_background

    def callback(x, y, time):
        colour = torch.tensor([0.25, 0.5, 0.75])
        return colour * torch.ones_like(x + y + time)

    return callback


@pytest.mark.parametrize("callback_kind", ["torch", "taichi"])
def test_a_callable_background_is_decoded_into_the_linear_buffer(
    callback_kind, linear_space
):
    """A procedural background is authored display-referred, like every colour.

    It composites against geometry that has already been decoded (``rs_acc *
    255 + weight * bg``), so it owes the composite linear light. Skipping the
    decode left a callable background the one ingest in the renderer still
    handing encoded values to linear arithmetic -- 0.5 arriving as 128 where a
    solid colour or an image of the same grey arrives as 54.6.
    """
    deferred = _deferred_background(_rgb_callback(callback_kind), width=2, height=1)
    # Four channels and float: a callable background is always opaque (see
    # ``Scene.background_is_transparent``), and the linear space renders on the
    # float HDR buffer.
    result = torch.zeros((1, 2, 4), dtype=torch.float32)
    _prefill_background(result, deferred, frame_offset=0, device=result.device)

    expected = [255 * _reference_to_linear(c) for c in (0.25, 0.5, 0.75)]
    assert result[..., :3].flatten().tolist() == pytest.approx(expected * 2, abs=1e-3)
    # Glow is not a colour: not decoded, and not read off one either (see the
    # channel-fill test below).
    assert result[..., 3].flatten().tolist() == [0.0, 0.0]


def test_torch_callable_background_keeps_a_dark_gradient_off_the_byte_grid(
    linear_space,
):
    """The float HDR buffer holds what the callback drew, at full precision.

    Quantizing to bytes *in linear light* is what bands a dark ramp: the
    bottom of this gradient spans less than a byte, so every one of these
    pixels used to land on the same handful of integers. Strict monotonicity
    is the property a gradient has and a staircase does not.
    """

    def background(x, y, time):
        return (x * 0.01 + 0 * y + 0 * time).expand(-1, -1, -1, 3)

    deferred = _deferred_background(background, width=8, height=1)
    result = torch.zeros((1, 8, 4), dtype=torch.float32)
    _prefill_background(result, deferred, frame_offset=0, device=result.device)

    ramp = result[0, :, 0]
    assert torch.all(ramp[1:] > ramp[:-1])


@pytest.mark.parametrize("callback_kind", ["torch", "taichi"])
def test_an_rgb_callable_background_does_not_glow(callback_kind):
    """Three channels are three channels: glow is not the blue one.

    The frame buffer is ``[R, G, B, glow]``, and the glow channel is what bloom
    blurs. Filled from the source's last channel, an ordinary RGB background
    bloomed over its whole area at the strength of its own blue -- a blue-grey
    ground lit up.
    """
    deferred = _deferred_background(_rgb_callback(callback_kind), width=2, height=1)
    result = torch.zeros((1, 2, 4), dtype=torch.uint8)
    _prefill_background(result, deferred, frame_offset=0, device=result.device)

    assert result[..., :3].flatten().tolist() == [64, 128, 191] * 2
    assert result[..., 3].flatten().tolist() == [0, 0]
