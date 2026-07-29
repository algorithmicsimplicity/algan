import pytest
import taichi as ti
import torch

from algan.render_loop import _prepare_background_for_chunk
from algan.rendering.raytracing.scene_builder import _prefill_background
from algan.scene import Scene
from algan.settings.video_settings import SMOKE_TEST


@ti.func
def _coordinate_background(x, y, time):
    return ti.Vector([x, y, time, 1.0])


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
        background_frame=_coordinate_background,
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
