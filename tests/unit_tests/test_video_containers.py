"""Every container ``save_video`` accepts must come out as a playable file.

``.webm`` and ``.gif`` were on the list of supported containers, but an opaque
render was still encoded with ``libx264``, which neither muxer takes: every
frame was rendered, and then FFmpeg refused the stream and the writer died with
``BrokenPipeError``. WebM's audio had the same fault one track over -- MP3,
where WebM carries only Vorbis or Opus -- so even a *transparent* WebM, whose
video codec was already right, failed as soon as the Scene had sound.

The bug is entirely in what the container is handed, so these tests run the
real writer and the real FFmpeg on synthetic frames instead of rendering: a
fake ``_render_to_video`` streams solid frames through the writer that
``save_video`` built and publishes the file the way the real one does.
"""

from __future__ import annotations

import math
import os
import struct
import subprocess
import wave

import numpy as np
import pytest

import algan
from algan import Audio, Square, VideoSettings
from algan.errors import AlganWarning
from algan.scene_manager import SceneManager

SIZE = 16
FRAMES_PER_SECOND = 2
RED = (220, 30, 30)
BLUE = (30, 30, 220)


def _ffmpeg():
    import moviepy.config

    return moviepy.config.FFMPEG_BINARY


def _streams(path):
    """The ``Stream`` lines FFmpeg reports for ``path``."""
    report = subprocess.run(
        [_ffmpeg(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
    ).stderr
    return [line.strip() for line in report.splitlines() if "Stream #" in line]


def _decoded_frames(path):
    raw = subprocess.run(
        [
            _ffmpeg(),
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
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, SIZE, SIZE, 3)


def _write_tone(path, seconds=1.0, rate=22050):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(
            b"".join(
                struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
                for i in range(int(seconds * rate))
            )
        )


@pytest.fixture
def scene(monkeypatch):
    """The active Scene, with one spawned Square and solid red-then-blue frames."""
    scene = SceneManager.instance().current_scene
    square = Square().spawn(animate=False)

    def render_solid_frames(file_writer, file_path, file_path_out, **_kwargs):
        for color in (RED, RED, BLUE, BLUE):
            file_writer.write_frame(np.full((SIZE, SIZE, 3), color, dtype=np.uint8))
        file_writer.close()
        os.replace(file_path, file_path_out)

    monkeypatch.setattr(scene, "_render_to_video", render_solid_frames)
    scene.square = square
    return scene


def _save(path, **kwargs):
    return algan.Scene.save_video(
        path,
        video_settings=VideoSettings((SIZE, SIZE), FRAMES_PER_SECOND, supersampling=1),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("suffix", "codec"),
    [(".webm", "Video: vp9"), (".gif", "Video: gif")],
)
def test_an_opaque_render_is_encoded_with_a_codec_its_container_takes(
    scene, tmp_path, suffix, codec
):
    result = _save(tmp_path / f"clip{suffix}")

    assert result.status == "rendered"
    assert any(codec in line for line in _streams(result.output_path))
    frames = _decoded_frames(result.output_path).astype(float)
    assert len(frames) == 4
    # Lossy (VP9) or quantized (GIF), but each frame is still its own color.
    assert np.abs(frames[0] - RED).mean() < 12
    assert np.abs(frames[-1] - BLUE).mean() < 12


@pytest.mark.parametrize("transparent", [False, True], ids=["opaque", "transparent"])
def test_a_webm_carries_the_scenes_sound_as_opus(scene, tmp_path, transparent):
    tone = tmp_path / "tone.wav"
    _write_tone(tone)
    with Audio(str(tone)):
        scene.square.move(algan.RIGHT)
    extra = {"background": algan.TRANSPARENT} if transparent else {}

    result = _save(tmp_path / "voiced.webm", **extra)

    streams = _streams(result.output_path)
    assert any("Video: vp9" in line for line in streams)
    assert any("Audio: opus" in line for line in streams)


def test_a_gif_leaves_the_sound_out_and_says_so(scene, tmp_path):
    tone = tmp_path / "tone.wav"
    _write_tone(tone)
    with Audio(str(tone)):
        scene.square.move(algan.RIGHT)

    with pytest.warns(AlganWarning, match="no audio track"):
        result = _save(tmp_path / "voiced.gif")

    streams = _streams(result.output_path)
    assert any("Video: gif" in line for line in streams)
    assert not any("Audio" in line for line in streams)


def test_caller_parameters_still_win_over_the_containers_defaults(
    scene, tmp_path, monkeypatch
):
    """The container's parameters go first, so FFmpeg takes the caller's last."""
    from algan.utils import algan_utils

    seen = {}
    real_writer = algan_utils.get_file_writer

    def recording_writer(*args):
        seen["codec"], seen["params"] = args[2], args[5]
        return real_writer(*args)

    monkeypatch.setattr(algan_utils, "get_file_writer", recording_writer)

    _save(tmp_path / "tuned.webm", ffmpeg_params=["-crf", "40"])

    assert seen["codec"] == "libvpx-vp9"
    assert seen["params"][-2:] == ["-crf", "40"]
