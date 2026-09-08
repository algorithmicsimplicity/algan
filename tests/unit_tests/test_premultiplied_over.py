"""The exported layer's linear-light contract, independently of its consumer.

Feature tests: intentionally outside --fast. No Resolve installation is assumed;
the codec test proves sample transport, not the editor's color-management chain.
"""

from __future__ import annotations

import subprocess
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from algan import SMOKE_TEST, TRANSPARENT, Circle, Color, Scene
from algan.errors import AlganConfigurationError
from algan.rendering.post_processing.bloom import bloom_filter
from algan.rendering.post_processing.post_process import (
    _finalize_on_device,
    post_process_frames,
)
from algan.rendering.raytracing import settings as rt_settings
from algan.settings._startup import render_device
from algan.utils.algan_utils import _transparent_encoder, get_file_writer
from algan.utils.memory_utils import ManualMemory


def _memory():
    return ManualMemory(0.0, device=render_device(), managed=False, num_bytes=1 << 23)


def _encode(rgb):
    return torch.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def _decode(rgb):
    return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _frame():
    frame = torch.zeros((1, 16, 16, 5), device=render_device())
    frame[:, 6:10, 6:10, :3] = torch.tensor([0.04, 0.01, 0.02], device=render_device())
    frame[:, 6:10, 6:10, 3] = 0.35
    frame[:, 6:10, 6:10, 4] = 128
    return frame


def test_setting_is_scene_local_and_survives_reset():
    with Scene(premultiplied_over=True) as first:
        assert first.premultiplied_over is True
        with Scene() as second:
            assert second.premultiplied_over is False
            assert Scene.set_premultiplied_over() is second
            assert second.set_premultiplied_over(False) is second
        assert first.premultiplied_over is True
        first.reset()
        assert first.premultiplied_over is True


def test_the_flag_reads_off_a_scene_that_never_ran_init():
    """The render loop reads ``self.premultiplied_over`` with no fallback.

    Nothing guarantees a Scene reaching the render loop was built by
    ``__init__``: the memory-preflight fixtures construct one with
    ``Scene.__new__`` and set only the handful of attributes they exercise. A
    setting that lived solely on the instance therefore turned every render
    path into an ``AttributeError`` for them, which is how this landed on CI
    rather than in the suite. The class-level default is the guarantee, and
    off is the value that leaves those renders as they were.
    """
    assert Scene.premultiplied_over is False
    assert Scene.__new__(Scene).premultiplied_over is False


@pytest.mark.parametrize("value", ["false", 1, None])
def test_setting_requires_a_boolean(value):
    with pytest.raises(AlganConfigurationError, match="premultiplied_over"):
        Scene(premultiplied_over=value)
    with (
        Scene() as scene,
        pytest.raises(AlganConfigurationError, match="premultiplied_over"),
    ):
        scene.set_premultiplied_over(value)


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("linear_color_space", False),
        ("tonemapping", True),
        ("post_process_tonemap", False),
    ],
)
def test_incompatible_settings_fail_before_render_or_audio(
    setting, value, monkeypatch, tmp_path
):
    monkeypatch.setattr(rt_settings, setting, value)
    with Scene(background=TRANSPARENT, premultiplied_over=True) as scene:
        scene.set_background(TRANSPARENT)

        def unexpected(*args, **kwargs):
            raise AssertionError("validation must precede audio and rendering")

        monkeypatch.setattr(scene, "save_audio", unexpected)
        monkeypatch.setattr(scene, "_render_primitive_batch", unexpected)
        with pytest.raises(AlganConfigurationError, match=setting):
            scene.save_video(tmp_path / "invalid.mov")
        with pytest.raises(AlganConfigurationError, match=setting):
            list(scene.get_frames(0, 1))


def test_bloom_keeps_alpha_and_does_not_mutate_its_input():
    frame = _frame()
    before = frame.clone()
    result = bloom_filter(frame, memory=_memory(), premultiplied_over=True)
    assert torch.equal(frame, before)
    assert torch.equal(result[..., 4], before[..., 4])
    assert torch.equal(result[..., 3], before[..., 3])
    halo = before[..., 4] == 0
    assert result[..., :3][halo].max() > 0


@pytest.mark.parametrize("kernel", [False, True])
def test_encoding_preserves_additive_light_and_coverage(kernel, monkeypatch):
    monkeypatch.setattr(rt_settings, "linear_color_space", True)
    monkeypatch.setattr(rt_settings, "post_tonemap_kernel", kernel)
    frame = torch.tensor(
        [
            [
                [
                    [0.1, 0.2, 0.3, 0, 0],
                    [0.1, 0.2, 0.3, 0, 32],
                    [0.1, 0.2, 0.3, 0, 255],
                    [2.0, 0.0, 0.0, 0, 0],
                ]
            ]
        ],
        device=render_device(),
    )
    result = _finalize_on_device(
        frame,
        5,
        _memory(),
        tonemap_enabled=True,
        tonemapping=False,
        tonemap_method="neutral",
        exposure=1,
        premultiplied_over=True,
    )
    expected = (_encode(frame[..., :3].clamp(0, 1)) * 255).round().to(torch.uint8)
    assert torch.equal(result[..., :3], expected)
    assert torch.equal(result[..., 3], frame[..., 4].to(torch.uint8))


@pytest.mark.parametrize(
    "process",
    [
        bloom_filter,
        partial(bloom_filter, strength=12),
        partial(bloom_filter, premultiplied_over=False),
    ],
)
def test_explicit_and_tuned_bloom_honor_scene_mode(process):
    frame = _frame()
    # The public post stage receives byte-range RGB/glow, even in float buffers.
    frame[..., :4] *= 255
    result = post_process_frames(
        _memory(), frame, 1, (process,), premultiplied_over=True
    )
    expected_alpha = frame[..., 4].flip(-2).to(torch.uint8).cpu()
    assert torch.equal(result[..., 3], expected_alpha)
    assert result[..., :3][result[..., 3] == 0].max() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("exposure", [1.0, 0.7])
def test_kernel_and_torch_agree_for_a_color_and_alpha_ramp(
    monkeypatch, dtype, exposure
):
    frame = torch.zeros((1, 4, 257, 5), device=render_device(), dtype=dtype)
    frame[..., :3] = torch.linspace(0, 1.2, 257, device=render_device())[
        None, None, :, None
    ]
    frame[..., 4] = torch.tensor([0, 1, 127, 255], device=render_device())[:, None]
    results = []
    for kernel in (False, True):
        monkeypatch.setattr(rt_settings, "post_tonemap_kernel", kernel)
        results.append(
            _finalize_on_device(
                frame,
                5,
                _memory(),
                tonemap_enabled=True,
                tonemapping=False,
                tonemap_method="neutral",
                exposure=exposure,
                premultiplied_over=True,
            ).clone()
        )
    assert torch.equal(*results)


@pytest.mark.parametrize(
    "plate", [(0, 0, 0), (0.2, 0.2, 0.2), (0.1, 0.6, 0.25), (1, 1, 1)]
)
def test_export_composites_as_foreground_light_plus_transmitted_plate(plate):
    frame = _frame()
    bloomed = bloom_filter(frame, memory=_memory(), premultiplied_over=True)
    encoded = _finalize_on_device(
        bloomed,
        5,
        _memory(),
        tonemap_enabled=True,
        tonemapping=False,
        tonemap_method="neutral",
        exposure=1,
        premultiplied_over=True,
    )
    backdrop = torch.tensor(plate, device=render_device())
    expected = bloomed[..., :3] + (1 - frame[..., 4:5] / 255) * backdrop
    actual = (
        _decode(encoded[..., :3].float() / 255)
        + (1 - encoded[..., 3:].float() / 255) * backdrop
    )
    # Compare displayed output, including saturation; no disagreement mask.
    error = (_encode(expected.clamp(0, 1)) - _encode(actual.clamp(0, 1))).abs()
    assert error.max() * 255 <= 2


@pytest.mark.parametrize("empty", [False, True])
def test_real_render_threads_the_mode_through_both_render_routes(empty):
    settings = SMOKE_TEST.set(resolution=(32, 32), supersampling=1, fxaa=False)
    with Scene(
        video_settings=settings, background=TRANSPARENT, premultiplied_over=True
    ) as scene:
        if not empty:
            Circle(glow=0.4, opacity=0.5, scene=scene).spawn(animate=False)
        result = torch.cat(list(scene.get_frames(0, 1)))
        assert result.shape == (1, 32, 32, 4)
        if empty:
            assert torch.count_nonzero(result) == 0
        else:
            halo = result[..., 3] == 0
            assert halo.any()
            assert result[..., :3][halo].max() > 0
            assert result[..., 3].max() < 255


def test_codec_defaults():
    assert _transparent_encoder(".mov", False) == ("png", [])
    codec, params = _transparent_encoder(".mov", True)
    assert codec == "prores_ks"
    assert params[params.index("-profile:v") + 1] == "4444"
    assert params[params.index("-pix_fmt") + 1] == "yuva444p10le"
    for suffix in (".webm", ".mkv", ".avi"):
        assert _transparent_encoder(suffix, True) == _transparent_encoder(suffix, False)


@pytest.mark.parametrize(
    ("enabled", "codec", "params", "expected"),
    [
        (False, None, None, "png"),
        (True, None, None, "prores_ks"),
        (True, None, ["-vendor", "apl0"], "prores_ks"),
        (True, "png", ["-compression_level", "9"], "png"),
    ],
)
def test_save_video_selects_codec_and_preserves_caller_options(
    enabled, codec, params, expected, monkeypatch, tmp_path
):
    from algan.utils import algan_utils

    seen = []

    def writer(*args):
        seen.append(args)
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(algan_utils, "get_file_writer", writer)
    monkeypatch.setattr(algan_utils, "check_codec_is_available", lambda codec: None)
    monkeypatch.setattr(algan_utils, "resolve_encode_binary", lambda codec: None)
    with Scene(video_settings=SMOKE_TEST, premultiplied_over=enabled) as scene:
        scene.set_background(TRANSPARENT)
        monkeypatch.setattr(scene, "save_audio", lambda *args, **kwargs: None)
        monkeypatch.setattr(scene, "_render_to_video", lambda *args, **kwargs: None)
        original = list(params) if params is not None else None
        scene.save_video(tmp_path / "clip.mov", codec=codec, ffmpeg_params=params)
        assert seen[0][2] == expected
        actual_params = seen[0][5]
        if expected == "prores_ks":
            assert actual_params[:4] == _transparent_encoder(".mov", True)[1]
        if params is not None:
            assert actual_params[-len(params) :] == params
            assert params == original


def test_opaque_render_is_inert_and_no_glow_composites_against_renderer():
    settings = SMOKE_TEST.set(resolution=(32, 32), supersampling=1, fxaa=False)
    with Scene(
        video_settings=settings, background=TRANSPARENT, premultiplied_over=True
    ) as scene:
        Circle(color=Color((0.2, 0.1, 0.3)), opacity=0.5, scene=scene).spawn(
            animate=False
        )
        layer = torch.cat(list(scene.get_frames(0, 1, post_processes=())))
        for plate in ((0.0, 0.0, 0.0), (0.2, 0.2, 0.2), (0.1, 0.6, 0.25)):
            scene.set_background(Color(plate))
            opaque = torch.cat(list(scene.get_frames(0, 1, post_processes=())))
            source = _decode(layer[..., :3].float() / 255)
            background = _decode(torch.tensor(plate))
            composed = source + (1 - layer[..., 3:].float() / 255) * background
            error = (_encode(composed.clamp(0, 1)) * 255 - opaque.float()).abs()
            assert error.max() <= 2
        scene.set_premultiplied_over(False)
        control = torch.cat(list(scene.get_frames(0, 1, post_processes=())))
        assert torch.equal(control, opaque)


@pytest.mark.parametrize("codec", ["png", "prores_ks"])
def test_mov_round_trip_preserves_rgb_outside_alpha(codec, tmp_path):
    from moviepy.config import FFMPEG_BINARY

    path = tmp_path / "patches.mov"
    frame = np.empty((32, 32, 4), dtype=np.uint8)
    frame[..., :3] = [89, 124, 149]
    frame[..., 3] = 0
    frame[:, 8:16, 3] = 128
    frame[:, 16:24, 3] = 255
    params = _transparent_encoder(".mov", True)[1] if codec == "prores_ks" else []
    writer = get_file_writer(str(path), (32, 32), codec, 1, True, params, None, "mp3")
    try:
        writer.write_frame(frame)
    finally:
        writer.close()
    decoded = subprocess.run(
        [
            FFMPEG_BINARY,
            "-v",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    actual = np.frombuffer(decoded, dtype=np.uint8).reshape(frame.shape)
    tolerance = 0 if codec == "png" else 3
    assert np.max(np.abs(actual.astype(np.int16) - frame.astype(np.int16))) <= tolerance
    assert actual[:, :8, :3].min() > 0
    assert actual[:, :8, 3].max() == 0
