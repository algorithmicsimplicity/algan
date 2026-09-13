"""Narration metadata and its attachment to Algan's lazy audio timeline."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from algan.sound.transcript import _SpeechBlock, snapshot_transcript


def block(text="Hello, world!", *, duration=2, timestamps=None, start=3):
    clip = SimpleNamespace(duration=duration, algan_word_timestamps=timestamps)
    result = _SpeechBlock.from_clip(text, clip)
    if start is not None:
        result.start_time_func = lambda: start
    return result


def test_alignment_preserves_original_text_and_word_offsets():
    text = "🙂  Well-timed,\n\tnaïve — words!"
    entry = block(
        text,
        timestamps=[
            ("WELL", 0.1, 0.3),
            ("TIMED", 0.4, 0.6),
            ("NAÏVE", 0.8, 1.0),
            ("WORDS", 1.3, 1.8),
        ],
    )
    result = snapshot_transcript([entry], 10)
    row = result["blocks"][0]
    assert row["text"] == text
    assert row["timing"] == "aligned"
    assert result["text"] == text + "\n\n"
    assert [w["text"] for w in row["words"]] == [
        "🙂",
        "Well-timed,",
        "naïve",
        "—",
        "words!",
    ]
    assert row["words"][1]["start"] == pytest.approx(3.1)
    assert row["words"][1]["end"] == pytest.approx(3.6)
    assert row["words"][0]["start"] is None
    assert row["words"][3]["start"] is None
    for word in row["words"]:
        assert text[word["offset"] : word["end_offset"]] == word["text"]


def test_custom_alignment_can_keep_a_hyphenated_word_together():
    entry = block("Well-timed!", timestamps=[("well-timed", 0.1, 1.5)])
    assert entry.timing == "aligned"
    assert entry.words[0].end == 1.5


@pytest.mark.parametrize(
    "timestamps",
    [
        [("WRONG", 0, 1)],
        [],
        "not timestamps",
        [("HELLO", float("nan"), 1), ("WORLD", 1, 2)],
        [("HELLO", -1, 1), ("WORLD", 1, 2)],
        [("HELLO", 1, 0.5), ("WORLD", 1, 2)],
        [("HELLO", 0, 1), ("WORLD", 1, float("inf"))],
        [("HELLO", 0, 1), ("WORLD", 1, 3)],
        [(None, 0, 1), ("WORLD", 1, 2)],
        [("HELLO", 1, 1.5), ("WORLD", 0.5, 1)],
    ],
)
def test_unusable_alignment_is_labeled_estimated_not_silently_misassigned(timestamps):
    entry = block(timestamps=timestamps)
    assert entry.timing == "estimated"
    row = snapshot_transcript([entry], 10)["blocks"][0]
    assert row["words"][0]["start"] == 3
    assert row["words"][-1]["end"] == 5
    json.dumps(row, allow_nan=False)


def test_estimates_use_audio_duration_not_the_trailing_animation_pause():
    entry = block("a bbb", duration=2)
    row = snapshot_transcript([entry], 20)["blocks"][0]
    assert row["timing"] == "estimated"
    assert [w["start"] for w in row["words"]] == [3, 3.5]
    assert [w["end"] for w in row["words"]] == [3.5, 5]


def test_snapshot_freezes_the_resolved_offset_without_stretching_word_times():
    origin = [3.0]
    entry = block(timestamps=[("HELLO", 0.1, 0.4), ("WORLD", 1.1, 1.5)])
    entry.start_time_func = lambda: origin[0]
    origin[0] = 7.0
    blocks = [entry]
    snapshot = snapshot_transcript(blocks, 20)
    origin[0] = 12.0
    blocks.append(block("added later"))
    assert len(snapshot["blocks"]) == 1
    assert snapshot["blocks"][0]["words"][0]["start"] == 7.1
    assert snapshot["blocks"][0]["words"][1]["end"] == 8.5


@pytest.mark.parametrize(("start", "duration"), [(None, 2), (3, 0), (15, 2), (-5, 2)])
def test_inaudible_text_is_kept_but_is_not_a_seek_target(start, duration):
    entry = block(start=start, duration=duration)
    row = snapshot_transcript([entry], 10)["blocks"][0]
    assert row["text"] == "Hello, world!"
    assert row["timing"] == "unavailable"
    assert all(w["start"] is None and w["end"] is None for w in row["words"])


def test_words_crossing_the_view_boundary_are_clipped_not_discarded():
    entry = block(
        "one two three",
        duration=3,
        start=-0.5,
        timestamps=[("ONE", 0, 1), ("TWO", 1, 2), ("THREE", 2, 3)],
    )
    row = snapshot_transcript([entry], 1.25)["blocks"][0]
    assert [(w["start"], w["end"]) for w in row["words"]] == [
        (0, 0.5),
        (0.5, 1.25),
        (None, None),
    ]
    assert row["text"] == "one two three"


def test_empty_transcript_and_punctuation_only_blocks_are_safe():
    assert snapshot_transcript([], 0) == {"text": "", "blocks": []}
    for text in ("", " \n\t ", "— …"):
        row = snapshot_transcript([block(text)], 10)["blocks"][0]
        assert all(w["start"] is None for w in row["words"])
        json.dumps(row, allow_nan=False)


@pytest.mark.fast
def test_speech_records_the_same_lazy_origin_as_its_audio_effect(fresh_scene):
    from algan import Scene, Seq, Speech

    scene = Scene.current()
    scene.audio_manager.set_speech_source(
        lambda text: SimpleNamespace(
            duration=2, algan_word_timestamps=[("HELLO", 0.1, 0.4), ("WORLD", 1.1, 1.5)]
        )
    )
    with Seq(runtime=12):
        Scene.wait(2)
        with Speech("Hello, world!", wait_at_end=2):
            Scene.wait(1)
    entry = scene.audio_manager._speech_blocks[0]
    effect = scene.effects[-1]
    assert entry.start_time_func is effect.start_time_func
    assert effect.start_time_func() == pytest.approx(4)
    row = snapshot_transcript([entry], 12)["blocks"][0]
    assert row["words"][0]["start"] == pytest.approx(4.1)
    assert row["words"][-1]["end"] == pytest.approx(5.5)
    assert row["end"] == pytest.approx(6)  # no parent stretching or wait_at_end


@pytest.mark.fast
def test_speech_is_scene_local_and_reset_discards_its_metadata(fresh_scene):
    from algan import Scene, Speech

    scene = Scene.current()
    scene.audio_manager.set_speech_source(lambda text: SimpleNamespace(duration=1))
    with Speech("first"):
        Scene.wait(1)
    with Scene() as other:
        other.audio_manager.set_speech_source(lambda text: SimpleNamespace(duration=1))
        with Speech("second"):
            Scene.wait(1)
        assert [b.text for b in other.audio_manager._speech_blocks] == ["second"]
        assert [b.text for b in scene.audio_manager._speech_blocks] == ["first"]
        other.reset()
        assert other.audio_manager._speech_blocks == []
    assert scene.audio_manager.video_transcript == "first\n\n"


def test_suppressed_speech_keeps_text_without_inventing_narration(fresh_scene):
    from algan import Off, Scene, Speech

    scene = Scene.current()
    scene.audio_manager.set_speech_source(lambda text: SimpleNamespace(duration=1))
    with Off(), Speech("silent words"):
        pass
    row = snapshot_transcript(scene.audio_manager._speech_blocks, 10)["blocks"][0]
    assert row["text"] == "silent words"
    assert row["timing"] == "unavailable"


def test_recorded_generator_retains_padded_subclip_relative_word_times(
    monkeypatch, tmp_path
):
    import moviepy

    from algan.utils import audio_utils

    cuts = []

    def subclipped(start, end):
        cuts.append((start, end))
        return SimpleNamespace(duration=end - start)

    monkeypatch.setattr(
        moviepy,
        "AudioFileClip",
        lambda path: SimpleNamespace(duration=20, subclipped=subclipped),
    )
    monkeypatch.setattr(
        audio_utils,
        "SETTINGS",
        SimpleNamespace(paths=SimpleNamespace(cache_directory=tmp_path)),
    )
    monkeypatch.setattr(
        audio_utils,
        "align_large_audio_torchaudio_robust",
        lambda *args: [
            ("HELLO", 10, 10.4),
            ("WORLD", 10.8, 11.1),
            ("NEXT", 12, 12.2),
        ],
    )
    generator = audio_utils.get_speech_generator_from_file("voice.wav", "script.txt")
    clip = generator("Hello,\nworld!")
    assert cuts[0] == pytest.approx((9.95, 11.6))
    assert [row[0] for row in clip.algan_word_timestamps] == ["HELLO", "WORLD"]
    assert clip.algan_word_timestamps[0][1:] == pytest.approx((0.05, 0.45))
    assert clip.algan_word_timestamps[1][1:] == pytest.approx((0.85, 1.15))
    assert _SpeechBlock.from_clip("Hello,\nworld!", clip).timing == "aligned"
