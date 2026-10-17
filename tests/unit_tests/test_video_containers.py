"""Every container ``save_video`` offers writes a video that plays.

The codec used to be chosen without asking the container, and H.264 with MP3
audio went into all of them. MP4, MOV, MKV and AVI take that; WebM and GIF
refuse it, and FFmpeg said so on a pipe nobody read. A clip big enough to fill
the pipe failed with "Broken pipe" once its frames were paid for; a smaller one
left an empty file that ``save_video`` reported rendered.

These encode real clips through moviepy's FFmpeg and decode them again. A
feature test, outside the fast suite: it breaks only when encoding changes.
"""

from __future__ import annotations

import math

import imageio_ffmpeg
import moviepy
import numpy as np
import pytest

from algan import (
    BLUE,
    RIGHT,
    SMOKE_TEST,
    TRANSPARENT,
    YELLOW,
    Circle,
    Scene,
    Seq,
    Square,
)
from algan.animation_timeline.animation_contexts import Audio
from algan.errors import AlganWarning
from algan.scene_manager import SceneManager

CLIP = SMOKE_TEST.set(resolution=(64, 36), frames_per_second=10)
FRAMES = 5  # Half a second at 10 frames per second.


@pytest.fixture(autouse=True)
def _fresh_scenes():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _tone(seconds, rate=44100):
    samples = int(seconds * rate)
    wave = np.sin(np.linspace(0, 2 * math.pi * 440 * seconds, samples)) * 0.5
    return moviepy.AudioArrayClip(wave.reshape(-1, 1).repeat(2, 1), fps=rate)


def _save(path, *, audio=False, **kwargs):
    """Half a second of a square and a glowing disc passing each other."""
    with Scene() as scene:
        square = Square(color=BLUE).spawn(False)
        disc = Circle(color=YELLOW, glow=0.3).spawn(False)
        with Audio(_tone(FRAMES / 10)) if audio else Seq(runtime=FRAMES / 10):
            square.move(RIGHT)
            disc.move(-RIGHT)
        return scene.save_video(path, CLIP, **kwargs)


def _decode(path):
    """The video and audio codecs FFmpeg reads, and the frames as RGB bytes."""
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    meta = next(reader)
    width, height = meta["size"]
    frames = [np.frombuffer(f, np.uint8).reshape(height, width, 3) for f in reader]
    audio = meta.get("audio_codec")
    return (
        meta["codec"].rstrip(","),
        audio.rstrip(",") if audio else None,
        np.stack(frames).astype(np.int16),
    )


@pytest.mark.parametrize(
    ("suffix", "codec"),
    [
        (".mp4", "h264"),
        (".mov", "h264"),
        (".mkv", "h264"),
        (".avi", "h264"),
        (".webm", "vp9"),
        (".gif", "gif"),
    ],
)
def test_every_container_writes_a_video_that_plays(tmp_path, suffix, codec):
    """REGRESSION for WebM and GIF, which took H.264 and wrote nothing."""
    result = _save(tmp_path / f"clip{suffix}")
    assert result.rendered
    written, _, frames = _decode(result.output_path)
    assert written == codec
    assert len(frames) == FRAMES


@pytest.mark.parametrize(
    ("suffix", "background", "audio_codec"),
    [
        (".mp4", None, "mp3"),
        (".webm", None, "opus"),
        (".webm", TRANSPARENT, "opus"),
        (".mov", TRANSPARENT, "mp3"),
    ],
    ids=["mp4", "webm", "transparent webm", "transparent mov"],
)
def test_the_audio_is_encoded_as_the_container_takes_it(
    tmp_path, suffix, background, audio_codec
):
    """REGRESSION. WebM carries Opus or Vorbis, and was handed MP3: with a
    soundtrack, even the transparent WebM that picked its own video codec
    failed.
    """
    result = _save(tmp_path / f"clip{suffix}", audio=True, background=background)
    _, written, frames = _decode(result.output_path)
    assert written == audio_codec
    assert len(frames) == FRAMES


def test_a_gif_leaves_out_the_audio_and_says_so(tmp_path):
    """GIF has no audio track to put a soundtrack in."""
    with pytest.warns(AlganWarning, match="no audio track"):
        result = _save(tmp_path / "clip.gif", audio=True)
    codec, audio, frames = _decode(result.output_path)
    assert (codec, audio, len(frames)) == ("gif", None, FRAMES)


def test_a_gif_keeps_the_colors_of_every_frame(tmp_path):
    """Each frame gets a palette drawn from that frame. FFmpeg's fixed default
    palette dithers flat fills into noise: against the lossless render, almost
    one pixel in five came out more than 8 levels off, and the worst 98.
    """
    _, _, lossless = _decode(_save(tmp_path / "lossless.avi", codec="png").output_path)
    _, _, gif = _decode(_save(tmp_path / "clip.gif").output_path)
    assert np.abs(gif - lossless).max() <= 2


def test_a_named_codec_is_used_as_named(tmp_path):
    """Only the default changes with the container: VP8 asked for is VP8."""
    result = _save(tmp_path / "clip.webm", codec="libvpx")
    assert _decode(result.output_path)[0] == "vp8"


def test_a_video_ffmpeg_refuses_is_an_error_and_replaces_nothing(tmp_path):
    """REGRESSION. moviepy's writer closes FFmpeg without asking how it exited,
    so only a frame that failed to write could report a refusal. Every frame of
    a clip this small fits in the pipe: H.264 named for a WebM left a file of
    a few hundred bytes in place of the earlier render, reported rendered.
    """
    target = tmp_path / "clip.webm"
    target.write_bytes(b"an earlier render")
    with pytest.raises(OSError, match="supported for WebM"):
        _save(target, codec="libx264")
    assert target.read_bytes() == b"an earlier render"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["clip.webm"]
