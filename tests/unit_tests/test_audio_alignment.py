"""Recorded-speech chunk retries, without downloading a model.

The model, the processor and torchaudio's alignment are stand-ins. The audio is
not: each test reads a real WAV file through soundfile, so the reading code
runs as it does for a user, whatever torchaudio is installed. The stand-in
torchaudio deliberately has no ``info`` or ``load``; torchaudio 2.9 removed the
first, and the second needs ``torchcodec``, which nothing installs.
"""

from __future__ import annotations

import sys
import wave
from types import ModuleType, SimpleNamespace

import pytest
import torch

from algan.errors import AlganConfigurationError, AudioTranscriptMismatchError
from algan.utils import audio_utils
from algan.utils.audio_utils import align_large_audio_torchaudio_robust


def _write_wav(path, num_frames, sample_rate, channels=1):
    """A silent 16-bit PCM WAV file, written with the standard library."""
    with wave.open(str(path), "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(bytes(2 * channels * num_frames))
    return path


@pytest.fixture
def alignment(monkeypatch, tmp_path):
    state = SimpleNamespace(calls=0, loads=[], modes=[], duration=300)
    transcript = tmp_path / "transcript.txt"
    transcript.write_text(" ".join(["A"] * 125))
    state.transcript = transcript
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class Tokenizer:
        encoder = {"A": 0}
        decoder = {0: "A"}
        pad_token_id = 0

        def __call__(self, text):
            return SimpleNamespace(input_ids=[self.encoder[c] for c in text])

    class Processor:
        tokenizer = Tokenizer()
        feature_extractor = SimpleNamespace(sampling_rate=100)

        @classmethod
        def from_pretrained(cls, _model_id):
            return cls()

        def __call__(self, waveform, **_kwargs):
            return SimpleNamespace(input_values=waveform)

    class Model:
        @classmethod
        def from_pretrained(cls, _model_id):
            return cls()

        def to(self, _device):
            return self

        def __call__(self, _values):
            return SimpleNamespace(logits=torch.zeros((1, 100, 2)))

    # Spy on the real reader: record each window, then read it for real.
    read = audio_utils._read_audio_window

    def read_window(path, frame_offset, num_frames):
        state.loads.append((frame_offset, num_frames))
        assert len(state.loads) <= 3, "alignment repeated an audio window"
        waveform, sample_rate = read(path, frame_offset, num_frames)
        assert waveform.dtype == torch.float32
        assert waveform.shape == (1, num_frames)
        return waveform, sample_rate

    monkeypatch.setattr(audio_utils, "_read_audio_window", read_window)
    # The recording at the test's duration, at the processor's 100 Hz.
    state.audio = lambda: _write_wav(
        tmp_path / "voice.wav", int(state.duration * 100), 100
    )

    def forced_align(_emissions, tokens, **_kwargs):
        state.calls += 1
        assert state.calls <= 12, "alignment exceeded its retry budget"
        return tokens, tokens

    def merge_tokens(tokens, _scores):
        mode = state.modes[min(state.calls - 1, len(state.modes) - 1)]
        spans = [
            SimpleNamespace(start=index * 0.5, end=(index + 1) * 0.5)
            for index in range(len(tokens))
        ]
        if mode == "boundary":
            spans[-1] = SimpleNamespace(start=99, end=100)
        elif mode == "zero":
            for span in spans:
                span.start = span.end = 0
        elif mode == "nan":
            for span in spans:
                span.start = span.end = float("nan")
        return spans

    functional = ModuleType("torchaudio.functional")
    functional.forced_align = forced_align
    functional.merge_tokens = merge_tokens
    torchaudio = ModuleType("torchaudio")
    torchaudio.functional = functional
    transformers = ModuleType("transformers")
    transformers.Wav2Vec2ForCTC = Model
    transformers.Wav2Vec2Processor = Processor
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)
    monkeypatch.setitem(sys.modules, "torchaudio.functional", functional)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return state


@pytest.mark.parametrize("after_success", [False, True])
def test_retry_exhaustion_fails_without_reusing_a_previous_chunk(
    alignment, after_success
):
    alignment.modes = ["success", "boundary"] if after_success else ["boundary"]
    with pytest.raises(
        AudioTranscriptMismatchError, match="after 10 attempts"
    ) as caught:
        align_large_audio_torchaudio_robust(alignment.audio(), alignment.transcript)

    assert alignment.calls == 10 + int(after_success)
    assert len(alignment.loads) == 1 + int(after_success)
    assert "voice.wav" in str(caught.value)
    assert f"transcript word {121 if after_success else 1}" in str(caught.value)
    if after_success:
        assert alignment.loads[1][0] > alignment.loads[0][0]


@pytest.mark.parametrize("mode", ["zero", "nan"])
def test_nonadvancing_alignment_is_rejected_before_another_load(alignment, mode):
    alignment.modes = [mode]
    with pytest.raises(AudioTranscriptMismatchError, match="no forward progress"):
        align_large_audio_torchaudio_robust(alignment.audio(), alignment.transcript)
    assert alignment.calls == 1
    assert len(alignment.loads) == 1


def test_successful_retry_then_next_chunk_keeps_order_and_advances(alignment):
    alignment.modes = ["boundary", "success"]
    segments = align_large_audio_torchaudio_robust(
        alignment.audio(), alignment.transcript
    )

    assert len(segments) == 125
    assert alignment.calls == 3
    assert len(alignment.loads) == 2
    assert alignment.loads[1][0] > alignment.loads[0][0]
    assert all(word == "A" and end > start for word, start, end in segments)
    assert all(left[2] <= right[1] for left, right in zip(segments, segments[1:]))


def test_final_chunk_does_not_drop_its_last_word(alignment):
    alignment.duration = 5
    alignment.transcript.write_text("A A A")
    alignment.modes = ["success"]
    segments = align_large_audio_torchaudio_robust(
        alignment.audio(), alignment.transcript
    )
    assert len(segments) == 3
    assert alignment.calls == 1
    assert segments[-1][-1] == pytest.approx(0.075)


def test_empty_transcript_returns_without_aligning(alignment):
    alignment.transcript.write_text("")
    assert (
        align_large_audio_torchaudio_robust(alignment.audio(), alignment.transcript)
        == []
    )
    assert alignment.loads == []
    assert alignment.calls == 0


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_chunk_duration_is_rejected_before_loading_dependencies(duration):
    with pytest.raises(AlganConfigurationError, match="finite and positive"):
        align_large_audio_torchaudio_robust(
            "unused.wav", "unused.txt", chunk_duration_s=duration
        )


def test_a_window_of_a_real_recording_reads_without_torchaudio(tmp_path):
    """The length, rate and samples come from the file, through soundfile.

    torchaudio is no part of it: on torchaudio 2.9 and later ``info`` is gone
    and ``load`` needs ``torchcodec``, so reading through either crashed every
    fresh ``algan[audio]`` install.
    """
    path = tmp_path / "stereo.wav"
    left = list(range(0, 2000, 2))
    right = [-v for v in left]
    with wave.open(str(path), "wb") as target:
        target.setnchannels(2)
        target.setsampwidth(2)
        target.setframerate(8000)
        target.writeframes(
            b"".join(
                a.to_bytes(2, "little", signed=True)
                + b.to_bytes(2, "little", signed=True)
                for a, b in zip(left, right)
            )
        )

    assert audio_utils._audio_frames_and_rate(path) == (1000, 8000)
    waveform, sample_rate = audio_utils._read_audio_window(path, 100, 5)
    assert sample_rate == 8000
    assert waveform.dtype == torch.float32
    expected = torch.tensor([left[100:105], right[100:105]]) / 32768
    assert torch.equal(waveform, expected)


def test_a_file_that_is_not_audio_names_the_formats_alignment_reads(tmp_path):
    path = tmp_path / "voice.m4a"
    path.write_bytes(b"not audio at all")
    with pytest.raises(AlganConfigurationError, match="WAV, FLAC, OGG and MP3"):
        audio_utils._audio_frames_and_rate(path)
