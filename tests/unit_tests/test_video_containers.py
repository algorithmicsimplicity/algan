"""Every container ``save_video`` accepts must come out as a file that plays.

``select_video_encoder`` picks H.264 and the audio default was MP3 whatever
the container, so an opaque ``.webm`` or ``.gif`` -- both on the accepted
list -- asked FFmpeg for a stream its muxer refuses, as did a transparent
``.webm`` with sound. A short render fits in the pipe before FFmpeg refuses
anything, and the exit status was never read, so the temporary file was
published and reported as rendered: 262 bytes of WebM, or an empty GIF.
"""

from __future__ import annotations

import math
import subprocess
import wave

import numpy as np
import pytest

from algan import RIGHT, SETTINGS, SMOKE_TEST, Audio, Scene, Seq, Square
from algan.constants.color import BLUE, TRANSPARENT
from algan.errors import UnsupportedFeatureWarning
from algan.utils.algan_utils import _SUPPORTED_VIDEO_CONTAINERS
from algan.utils.video_encoding import _software_ffmpeg_binary

VIDEO = SMOKE_TEST.set(resolution=(48, 32), frames_per_second=10)


def _beep(path, seconds=0.3):
    rate = 44100
    tone = np.sin(np.arange(int(rate * seconds)) * (2 * math.pi * 440 / rate))
    samples = (tone * 0.3 * 32767).astype("<i2").repeat(2)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(samples.tobytes())
    return str(path)


def _render(path, *, background=None, sound=None, **kwargs):
    with Scene(video_settings=VIDEO) as scene:
        square = Square(color=BLUE).spawn(animate=False)
        with Audio(sound) if sound else Seq(runtime=0.3):
            square.move(RIGHT)
        return scene.save_video(str(path), background=background, **kwargs)


def _streams(path):
    """``(video codec, audio codec or None, decoded frame count)`` of ``path``."""
    decoded = subprocess.run(
        [_software_ffmpeg_binary(), "-hide_banner", "-i", str(path)]
        + ["-map", "0:v:0", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    assert decoded.returncode == 0, decoded.stderr
    info = decoded.stderr
    video = info.split("Video: ", 1)[1].split()[0].rstrip(",")
    audio = (
        info.split("Audio: ", 1)[1].split()[0].rstrip(",")
        if "Audio: " in info
        else None
    )
    frames = int(info.rsplit("frame=", 1)[1].split()[0])
    return video, audio, frames


@pytest.fixture(autouse=True)
def _output_in(tmp_path):
    SETTINGS.paths.set(output_root=str(tmp_path), output_directory=".")


@pytest.mark.parametrize("suffix", _SUPPORTED_VIDEO_CONTAINERS)
def test_every_container_plays(tmp_path, suffix):
    result = _render(tmp_path / f"clip{suffix}")
    assert result.status == "rendered"
    video, _, frames = _streams(result.output_path)
    assert frames == 3
    if suffix == ".webm":
        assert video == "vp9"
    elif suffix == ".gif":
        assert video == "gif"


def test_webm_sound_is_opus_with_or_without_alpha(tmp_path):
    sound = _beep(tmp_path / "beep.wav")
    for background in (None, TRANSPARENT):
        result = _render(tmp_path / "clip.webm", background=background, sound=sound)
        assert _streams(result.output_path)[:2] == ("vp9", "opus")


def test_a_gif_leaves_the_sound_out_and_says_so(tmp_path):
    sound = _beep(tmp_path / "beep.wav")
    with pytest.warns(UnsupportedFeatureWarning, match="GIF has no audio track"):
        result = _render(tmp_path / "clip.gif", sound=sound)
    assert _streams(result.output_path)[:2] == ("gif", None)


def test_a_file_ffmpeg_refuses_is_an_error_not_a_render(tmp_path):
    """An explicit codec the container cannot hold is the caller's to make."""
    with pytest.raises(RuntimeError, match="FFmpeg failed writing"):
        _render(tmp_path / "clip.webm", codec="libx264")
    assert not (tmp_path / "clip.webm").exists()
