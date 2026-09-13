"""Dependency-free tests of the narration payload, without starting a renderer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# This small stdlib-only module also runs in a viewer-only environment without
# installing Torch/Quadrants. Integration with the real Scene is tested separately.
_spec = importlib.util.spec_from_file_location(
    "_algan_transcript_test_data",
    Path(__file__).resolve().parents[2] / "algan/sound/transcript.py",
)
data = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = data
_spec.loader.exec_module(data)


def clip(duration=3, rows=None):
    result = SimpleNamespace(duration=duration)
    if rows is not None:
        result._algan_word_timings = rows
    return result


def snapshot(text, sound, start=5):
    cue = data._speech_cue(text, sound, None if start is None else lambda: start)
    return data._snapshot_transcript([cue])["blocks"][0]


def test_original_text_and_whitespace_survive_alignment():
    text = "  Hello,\nback-propagation!  —  café 😊\n"
    rows = [
        ("HELLO", 0.05, 0.3),
        ("BACK", 0.5, 0.7),
        ("PROPAGATION", 0.75, 1),
        ("CAFÉ", 1.2, 1.5),
    ]
    block = snapshot(text, clip(rows=rows))
    assert block["timing"] == "aligned"
    assert (
        "".join(w["before"] + w["text"] for w in block["words"]) + block["after"]
        == text
    )
    assert block["words"][0]["start"] == pytest.approx(5.05)
    assert block["words"][1]["start"] == pytest.approx(5.5)
    assert block["words"][1]["end"] == pytest.approx(6)
    assert block["words"][2]["start"] is None  # punctuation is preserved, not narrated
    assert block["words"][4]["start"] is None  # emoji offsets never cross into JS


def test_estimates_cover_only_the_clip_not_a_context_hold():
    block = snapshot("one three", clip(duration=4), start=10)
    assert block["timing"] == "estimated"
    assert block["words"][0]["start"] == 10
    assert block["words"][0]["end"] == 11.5
    assert block["words"][1]["end"] == 14


@pytest.mark.parametrize(
    "rows",
    [
        [("WRONG", 0, 1)],
        [("ONE", float("nan"), 1)],
        [("ONE", 1, 0.5)],
        [("ONE", 0, 99)],
        [("ONE", "bad", 1)],
        [("ONE", 0)],
    ],
)
def test_invalid_metadata_falls_back_to_explicit_estimates(rows):
    block = snapshot("one", clip(rows=rows))
    assert block["timing"] == "estimated"
    assert block["words"][0]["end"] == 8


@pytest.mark.parametrize("text", ["", " \n ", "... — 😊"])
def test_empty_and_punctuation_only_speech_is_preserved(text):
    block = snapshot(text, clip())
    assert block["text"] == text
    assert block["timing"] == "unavailable"
    assert all(word["start"] is None for word in block["words"])


def test_suppressed_audio_has_text_but_no_fake_seek_time():
    block = snapshot("still in the script", clip(), start=None)
    assert block["timing"] == "unavailable"
    assert all(word["start"] is None for word in block["words"])


def test_resolve_lazy_start_once_at_snapshot_without_scaling_word_offsets():
    start = [1]
    cue = data._speech_cue("one", clip(rows=[("ONE", 0.25, 0.75)]), lambda: start[0])
    start[0] = 12
    first = data._snapshot_transcript([cue])
    start[0] = 30
    assert first["blocks"][0]["words"][0]["start"] == 12.25
    assert first["blocks"][0]["words"][0]["end"] == 12.75
    assert data._snapshot_transcript([cue])["blocks"][0]["words"][0]["start"] == 30.25


def test_transcript_is_authored_order_even_when_speech_overlaps():
    cues = [
        data._speech_cue("later", clip(), lambda: 5),
        data._speech_cue("earlier", clip(), lambda: 0),
    ]
    result = data._snapshot_transcript(cues)
    assert result["text"] == "later\n\nearlier"
    assert [b["words"][0]["start"] for b in result["blocks"]] == [5, 0]


@pytest.mark.parametrize("duration", [0, -1, float("inf"), float("nan")])
def test_invalid_duration_never_reaches_json_as_a_nonfinite_time(duration):
    block = snapshot("one", clip(duration=duration))
    assert block["timing"] == "unavailable"
    assert block["words"][0]["start"] is None
