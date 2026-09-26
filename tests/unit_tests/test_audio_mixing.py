"""The fast narration mix must reproduce moviepy's, sample for sample.

``Scene.save_audio`` mixes a Scene's audio effects with
:func:`~algan.utils.audio_utils.write_composite_audio` instead of
``CompositeAudioClip.write_audiofile``, and speech clips open their decoder
without probing the file again. Both are pure speedups: the written file must
be byte-identical to what moviepy writes for the same composite.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

moviepy = pytest.importorskip("moviepy")

from algan.utils import audio_utils  # noqa: E402


def _tone(path, seconds, frequency, channels=2):
    samples = int(44100 * seconds)
    wave = np.sin(np.linspace(0, 2 * math.pi * frequency * seconds, samples)) * 0.3
    clip = moviepy.AudioArrayClip(
        wave.reshape(-1, 1).repeat(channels, 1),
        fps=44100,
    )
    clip.write_audiofile(str(path), fps=44100, logger=None)
    return str(path)


@pytest.fixture
def tones(tmp_path):
    return [
        _tone(tmp_path / f"tone{i}.wav", seconds, frequency)
        for i, (seconds, frequency) in enumerate(
            ((1.3, 220.0), (0.7, 330.0), (2.1, 441.0), (0.4, 97.0))
        )
    ]


def _composite(paths, starts, duration, lazy):
    clips = []
    for path, start in zip(paths, starts):
        clip = (
            audio_utils._cached_audio_file_clip(path)
            if lazy
            else moviepy.AudioFileClip(path)
        )
        clips.append(clip.with_start(start))
    composite = moviepy.CompositeAudioClip(clips)
    composite.duration = duration
    return composite


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "starts",
    [
        # Back to back with gaps, as narration is recorded.
        (0.25, 1.7, 2.53, 4.9),
        # Overlapping, so several members mix into one chunk.
        (0.0, 0.6, 0.61, 1.0),
    ],
)
@pytest.mark.parametrize("lazy", [False, True])
def test_fast_mix_is_byte_identical_to_moviepy(tmp_path, tones, starts, lazy):
    duration = 5.6
    reference = tmp_path / "moviepy.wav"
    composite = _composite(tones, starts, duration, lazy=False)
    composite.write_audiofile(
        str(reference), fps=44100, codec="pcm_s32le", nbytes=4, logger=None
    )
    composite.close()

    fast = tmp_path / "fast.wav"
    composite = _composite(tones, starts, duration, lazy=lazy)
    assert audio_utils.write_composite_audio(
        composite, str(fast), fps=44100, nbytes=4, codec="pcm_s32le"
    )
    composite.close()
    assert _digest(fast) == _digest(reference)


def test_fast_mix_declines_a_member_it_cannot_vouch_for(tmp_path, tones):
    clip = moviepy.AudioFileClip(tones[0]).with_start(0.1)

    class Custom(type(clip)):
        def is_playing(self, t):
            return super().is_playing(t)

    clip.__class__ = Custom
    composite = moviepy.CompositeAudioClip([clip])
    composite.duration = 2.0
    assert not audio_utils.write_composite_audio(
        composite, str(tmp_path / "x.wav"), fps=44100, nbytes=4, codec="pcm_s32le"
    )
    assert not (tmp_path / "x.wav").exists()
    clip.close()


def test_probed_reader_matches_the_probing_constructor(tones):
    from moviepy.audio.io.readers import FFMPEG_AudioReader

    for path in tones:
        probing = FFMPEG_AudioReader(
            path, decode_file=False, fps=44100, nbytes=2, buffersize=200000
        )
        probed = audio_utils._probed_audio_reader(
            FFMPEG_AudioReader, path, 44100, probing.duration
        )
        expected = {
            key: value
            for key, value in vars(probing).items()
            if key not in ("infos", "bitrate", "proc")
        }
        actual = {
            key: value
            for key, value in vars(probed).items()
            if key not in ("infos", "bitrate", "proc")
        }
        # A moviepy upgrade that adds a constructor attribute shows up here.
        assert expected.keys() == actual.keys()
        for key, value in expected.items():
            if isinstance(value, np.ndarray):
                assert np.array_equal(value, actual[key]), key
            else:
                assert value == actual[key], key
        # In mix-sized chunks: moviepy's reader mishandles a request wider
        # than half its buffer.
        times = np.arange(0, probing.duration, 1 / 44100)
        for start in range(0, len(times), 2000):
            chunk = times[start : start + 2000]
            assert np.array_equal(probing.get_frame(chunk), probed.get_frame(chunk))
        probing.close()
        probed.close()
