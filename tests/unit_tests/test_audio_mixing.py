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


def _tone_snr(samples, rate, frequency):
    t = np.arange(len(samples)) / rate
    basis = np.stack(
        [np.sin(2 * np.pi * frequency * t), np.cos(2 * np.pi * frequency * t)], 1
    )
    fit = basis @ np.linalg.lstsq(basis, samples, rcond=None)[0]
    return 10 * np.log10(fit.var() / (samples - fit).var())


@pytest.mark.parametrize("lazy", [False, True])
def test_another_output_rate_is_resampled_not_sampled_nearest(tmp_path, lazy):
    # Files decode at 44.1 kHz. Sampled on a 48 kHz grid by np.round, a 6 kHz
    # tone came out at 12 dB SNR; resampled by FFmpeg it stays clean.
    source = str(tmp_path / "tone.wav")
    wave = 0.3 * np.sin(2 * np.pi * 6000 * np.arange(88200) / 44100)
    moviepy.AudioArrayClip(wave[:, None].repeat(2, 1), fps=44100).write_audiofile(
        source, fps=44100, logger=None
    )
    out = tmp_path / "out.wav"
    composite = _composite([source], [0.0], 2.0, lazy=lazy)
    assert audio_utils.write_composite_audio(
        composite, str(out), fps=48000, nbytes=4, codec="pcm_s32le"
    )
    composite.close()
    written = moviepy.AudioFileClip(str(out), fps=48000)
    assert written.reader.infos["audio_fps"] == 48000
    samples = written.get_frame(np.arange(24000, 72000) / 48000)[:, 0]
    written.close()
    assert _tone_snr(samples, 48000, 6000.0) > 50


def test_mix_closes_each_decoder_once_past_it_and_files_open_none_early(
    tmp_path, tones
):
    # Every decoder is an ffmpeg process and two pipes. They were opened by
    # Audio("file") at authoring and kept for good, so ~120 sounds exhausted
    # macOS's 256 descriptors; now each opens in the mix and closes after it.
    clips = [audio_utils._cached_audio_file_clip(path) for path in tones * 3]
    assert all(clip._reader is None for clip in clips), "a miss is lazy too"
    opened, peak = [], [0]
    for clip in clips:
        open_reader = clip.open_reader

        def counted(clip=clip, open_reader=open_reader):
            opened.append(clip)
            live = sum(other._reader is not None for other in clips)
            peak[0] = max(peak[0], live + 1)
            return open_reader()

        clip.open_reader = counted
    starts = [1.5 * i for i in range(len(clips))]
    composite = moviepy.CompositeAudioClip(
        [clip.with_start(start) for clip, start in zip(clips, starts)]
    )
    composite.duration = starts[-1] + 3
    first, second = tmp_path / "first.wav", tmp_path / "second.wav"
    for path in (first, second):
        assert audio_utils.write_composite_audio(
            composite, str(path), fps=44100, nbytes=4, codec="pcm_s32le"
        )
        assert all(clip._reader is None for clip in clips)
    assert len(opened) == 2 * len(clips)
    assert peak[0] <= 5  # the three-deep prefetch plus the overlapping members
    assert _digest(first) == _digest(second)


def test_audio_from_a_file_opens_no_decoder_while_authoring(fresh_scene, tones):
    from algan import Audio, Scene

    with Audio(tones[0]) as block:
        Scene.wait(0.1)
    assert block.audio_clip._reader is None


def _seconds_written(path):
    import wave as wave_module

    with wave_module.open(str(path), "rb") as reader:
        return reader.getnframes() / reader.getframerate()


def test_audio_track_is_as_long_as_the_video(fresh_scene, tmp_path):
    from algan import SMOKE_TEST, Audio, Scene, Seq, Sync

    scene = Scene.current()
    scene.set_video_settings(SMOKE_TEST.set(frames_per_second=2))
    sound = moviepy.AudioClip(lambda t: np.full(np.shape(t) + (2,), 0.25), duration=1.2)
    with Audio(sound):  # ends at 1.2 s; at 2 fps the video stops at 1.0 s
        Scene.wait(0.1)
    assert scene.save_audio(tmp_path / "rounded.wav") is not None
    assert _seconds_written(tmp_path / "rounded.wav") == pytest.approx(1.0, abs=1e-4)
    # From inside a block, the innermost context's end (3.2 s) is not the
    # end of what is recorded: the enclosing Sync already runs to 4.2 s.
    with Sync():
        with Seq():
            Scene.wait(3)
        with Seq():
            Scene.wait(2)
            scene.save_audio(tmp_path / "midblock.wav")
    assert _seconds_written(tmp_path / "midblock.wav") == pytest.approx(4, abs=1e-4)


@pytest.mark.parametrize("stereo_bed", [False, True])
def test_mono_in_memory_clips_mix(tmp_path, tones, stereo_bed):
    # A 1-D mono AudioClip times the column mask made a len(t)-square array,
    # and a (n, 1) AudioArrayClip returns two columns while reporting one:
    # both raised ValueError in the mix.
    function = moviepy.AudioClip(lambda t: 0.125 + 0 * t, duration=1.0)
    array = moviepy.AudioArrayClip(np.full((44100, 1), 0.25), fps=44100)
    clips = [function.with_start(0.0), array.with_start(0.5)]
    if stereo_bed:
        clips.append(moviepy.AudioFileClip(tones[0]).with_start(2.0))
    composite = moviepy.CompositeAudioClip(clips)
    composite.duration = 2.0
    out = tmp_path / "mono.wav"
    assert audio_utils.write_composite_audio(
        composite, str(out), fps=44100, nbytes=2, codec="pcm_s16le"
    )
    composite.close()
    import wave as wave_module

    with wave_module.open(str(out), "rb") as reader:
        channels = reader.getnchannels()
        samples = np.frombuffer(reader.readframes(reader.getnframes()), "<i2")
    samples = samples.reshape(-1, channels) / 32768
    assert channels == (2 if stereo_bed else 1)
    assert samples.shape[0] == 88200
    np.testing.assert_allclose(samples[100:22000], 0.125, atol=1e-4)
    np.testing.assert_allclose(samples[22100:44000], 0.375, atol=1e-4)
    np.testing.assert_allclose(samples[44200:66000], 0.25, atol=1e-4)
