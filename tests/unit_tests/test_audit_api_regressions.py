"""Public API regressions from the October 2026 bug audit."""

import os
import shutil
import subprocess

import manim as mn
import numpy as np
import pytest
import torch
from PIL import Image

from algan import (
    OUT,
    SETTINGS,
    WHITE,
    Camera,
    Color,
    ImageMob,
    Mob,
    Off,
    Rectangle,
    Scene,
    SpotLight,
)
from algan.mobs import text as text_module
from algan.utils.algan_utils import concatenate_videos
from algan.utils.file_utils import get_image
from algan.utils.video_encoding import _software_ffmpeg_binary


@pytest.mark.parametrize("mode", ["L", "LA", "P", "CMYK", "RGB", "RGBA"])
def test_file_images_are_normalized_before_color_channels_are_added(tmp_path, mode):
    pixels = Image.new(mode, (2, 3))
    if mode == "P":
        pixels.putpalette([0, 255, 255] + [0] * 765)
        pixels.info["transparency"] = 0
    elif mode == "CMYK":
        pixels.paste((255, 0, 0, 0), (0, 0, 2, 3))
    elif mode in ("LA", "RGBA"):
        pixels.putalpha(64)
    path = tmp_path / ("image.tif" if mode == "CMYK" else "image.png")
    pixels.save(path)
    expected = torch.tensor(np.array(pixels.convert("RGBA")), dtype=torch.float32) / 255
    result = get_image(path)
    assert result.shape == (3, 2, 5)
    torch.testing.assert_close(result[..., :3].cpu(), expected[..., :3])
    torch.testing.assert_close(result[..., 4].cpu(), expected[..., 3])
    assert torch.count_nonzero(result[..., 3]) == 0
    ImageMob(str(path))


@pytest.mark.parametrize("shape", [(2,), (2, 3)])
@pytest.mark.parametrize("channels", [3, 4, 5])
def test_color_constructor_preserves_batch_axes(shape, channels):
    data = torch.linspace(0, 1, int(np.prod(shape)) * channels).reshape(
        *shape, channels
    )
    before = data.clone()
    result = Color(data, glow=0.7, opacity=0.8)
    expected = torch.stack(
        [Color(row, glow=0.7, opacity=0.8) for row in data.reshape(-1, channels)]
    )
    assert result.shape == (*shape, 5)
    torch.testing.assert_close(result.reshape(-1, 5), expected)
    torch.testing.assert_close(data, before)


def test_high_rank_color_accepts_scalar_opacity_without_mutating_input():
    original = WHITE.set_opacity(torch.full((2, 1, 1, 1, 1), 0.5))
    changed = original.set_opacity(0.25)
    assert changed.shape == original.shape == (2, 1, 1, 1, 5)
    torch.testing.assert_close(changed[..., :4], original[..., :4])
    assert (changed.opacity == 0.25).all()
    assert (original.opacity == 0.5).all()
    assert WHITE.opacity.item() == 1


def test_camera_constructed_with_rotated_basis_has_matching_screen_and_framing():
    with Scene(), Off():
        rotated = Camera(location=(0, 0, 10)).rotate(90, OUT)
        supplied = Camera(location=rotated.location, basis=rotated.basis)
        torch.testing.assert_close(
            supplied.get_corner_pixels(), rotated.get_corner_pixels()
        )
        subject = Rectangle(width=8, height=2)
        rotated.center_on(subject)
        supplied.center_on(subject)
        torch.testing.assert_close(supplied.location, rotated.location)
        assert supplied.location[..., 2].item() > 0


def test_light_target_mob_matches_its_location():
    with Scene(), Off():
        target = Mob(location=(1, 2, 3))
        light = SpotLight(location=(0, 0, 3))
        light.set_target(target)
        actual = light._directions(light.location).clone()
        light.set_target(target.location)
        torch.testing.assert_close(light._directions(light.location), actual)
        torch.testing.assert_close(
            actual.reshape(-1), torch.tensor([1, 2, 0], dtype=torch.float32) / 5**0.5
        )


@pytest.mark.parametrize("plain", [r"C:\Users\Ada", "trailing\\", "#$%&_{}~^\\"])
@pytest.mark.parametrize("triangulated", [False, True])
def test_plain_text_wrappers_compile_literal_tex_characters(
    monkeypatch, plain, triangulated
):
    if not shutil.which("latex") or not shutil.which("dvisvgm"):
        pytest.skip("full text construction requires latex and dvisvgm")
    if not triangulated:
        monkeypatch.delattr(mn, "Text")
    cls = text_module.TextTriangulated if triangulated else text_module.Text
    with Scene(), Off():
        result = cls(plain)
        assert result.text == plain
        assert result.character_mobs


def _clip(path, color):
    # The binary save_video encodes with: imageio-ffmpeg's bundled build where
    # no FFmpeg is on the PATH, as on the macOS CI arms.
    subprocess.run(
        [
            _software_ffmpeg_binary(),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color={color}:s=16x16:r=2:d=1",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_concat_rejects_missing_manifest_before_overwriting_output(tmp_path):
    existing = tmp_path / "output.mp4"
    existing.write_bytes(b"previous complete output")
    clip = tmp_path / "0_intro.mp4"
    clip.write_bytes(b"placeholder; validation must happen before ffmpeg")
    with pytest.raises(FileNotFoundError, match="1_missing.mp4"):
        concatenate_videos(tmp_path, input_files=[clip, "1_missing.mp4"])
    assert existing.read_bytes() == b"previous complete output"


def _frames(path):
    frames = subprocess.run(
        [
            _software_ffmpeg_binary(),
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(frames, dtype=np.uint8).reshape(-1, 16, 16, 3)


def test_concat_real_ffmpeg_handles_quotes_and_preserves_explicit_order(tmp_path):
    folder = tmp_path / "O'Brien space é"
    folder.mkdir()
    first, second = folder / "2_red's clip.mp4", folder / "1_blue's clip.mp4"
    _clip(first, "red")
    _clip(second, "blue")
    output = concatenate_videos(folder, input_files=[first, second])
    assert output is not None
    frames = _frames(output)
    assert len(frames) == 4
    assert frames[0, :, :, 0].mean() > 240
    assert frames[-1, :, :, 2].mean() > 240


@pytest.mark.skipif(os.name == "nt", reason="a backslash separates paths on Windows")
def test_concat_keeps_backslashes_that_are_part_of_a_posix_name(tmp_path):
    folder = tmp_path / "back\\slash"
    folder.mkdir()
    first, second = folder / "red\\clip.mp4", folder / "blue\\clip.mp4"
    _clip(first, "red")
    _clip(second, "blue")
    output = concatenate_videos(folder, input_files=[first, second])
    assert output is not None
    frames = _frames(output)
    assert len(frames) == 4
    assert frames[0, :, :, 0].mean() > 240
    assert frames[-1, :, :, 2].mean() > 240


@pytest.mark.skipif(os.name == "nt", reason="a shell script stands in for FFmpeg")
def test_concat_runs_the_ffmpeg_save_video_encodes_with(monkeypatch, tmp_path):
    folder = tmp_path / "clips"
    folder.mkdir()
    _clip(folder / "0_red.mp4", "red")
    _clip(folder / "1_blue.mp4", "blue")
    log = tmp_path / "pinned.log"
    pinned = tmp_path / "pinned-ffmpeg"
    pinned.write_text(
        f'#!/bin/sh\necho ran >> "{log}"\nexec "{_software_ffmpeg_binary()}" "$@"\n'
    )
    pinned.chmod(0o755)
    SETTINGS.paths.set(ffmpeg_binary=str(pinned))
    # With no FFmpeg on the PATH, a bare "ffmpeg" cannot start at all.
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    output = concatenate_videos(folder)

    assert output is not None
    assert log.read_text() == "ran\n"
    assert len(_frames(output)) == 4
