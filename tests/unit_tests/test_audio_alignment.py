"""Recorded-speech chunk retries, without downloading a model or reading audio."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from algan.errors import AlganConfigurationError, AudioTranscriptMismatchError
from algan.utils.audio_utils import align_large_audio_torchaudio_robust


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

    def load(_path, *, frame_offset, num_frames):
        state.loads.append((frame_offset, num_frames))
        assert len(state.loads) <= 3, "alignment repeated an audio window"
        return torch.zeros((1, 32)), 100

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
    torchaudio.info = lambda _path: SimpleNamespace(
        num_frames=state.duration * 100, sample_rate=100
    )
    torchaudio.load = load
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
        align_large_audio_torchaudio_robust("voice.wav", alignment.transcript)

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
        align_large_audio_torchaudio_robust("voice.wav", alignment.transcript)
    assert alignment.calls == 1
    assert len(alignment.loads) == 1


def test_successful_retry_then_next_chunk_keeps_order_and_advances(alignment):
    alignment.modes = ["boundary", "success"]
    segments = align_large_audio_torchaudio_robust("voice.wav", alignment.transcript)

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
    segments = align_large_audio_torchaudio_robust("voice.wav", alignment.transcript)
    assert len(segments) == 3
    assert alignment.calls == 1
    assert segments[-1][-1] == pytest.approx(0.075)


def test_empty_transcript_returns_without_aligning(alignment):
    alignment.transcript.write_text("")
    assert align_large_audio_torchaudio_robust("voice.wav", alignment.transcript) == []
    assert alignment.loads == []
    assert alignment.calls == 0


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_chunk_duration_is_rejected_before_loading_dependencies(duration):
    with pytest.raises(AlganConfigurationError, match="finite and positive"):
        align_large_audio_torchaudio_robust(
            "unused.wav", "unused.txt", chunk_duration_s=duration
        )
