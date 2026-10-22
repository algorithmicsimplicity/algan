"""Fast pure-Python tests for subtitle cue helpers (no scenes, no rendering)."""

from __future__ import annotations

import math
import re
from types import SimpleNamespace

import pytest

from algan.errors import AlganConfigurationError
from algan.sound import subtitles as st
from algan.sound.transcript import (
    _aligned_words,
    _parts,
    _SpeechBlock,
    snapshot_transcript,
)

# ---------------------------------------------------------------- _seconds


@pytest.mark.parametrize(
    ("value", "expected"), [(0, 0.0), (1, 1.0), ("2.5", 2.5), (-3, -3.0)]
)
def test_seconds_accepts_finite_numbers(value, expected):
    assert st._seconds(value, "x") == expected


@pytest.mark.parametrize(
    "value", [None, "abc", True, False, math.nan, math.inf, -math.inf, 10**400]
)
def test_seconds_rejects_non_finite_and_bool(value):
    with pytest.raises(AlganConfigurationError, match="x must be"):
        st._seconds(value, "x")


@pytest.mark.parametrize("value", [0, -1, 0.0])
def test_seconds_positive_rejects_non_positive(value):
    with pytest.raises(AlganConfigurationError, match="positive"):
        st._seconds(value, "x", positive=True)


# ---------------------------------------------------------------- options


def test_options_defaults():
    o = st._SubtitleOptions()
    assert (o.include_speech, o.max_chars_per_line, o.max_lines, o.max_duration) == (
        True,
        42,
        2,
        6.0,
    )


def test_options_coerce_max_duration_to_float():
    o = st._SubtitleOptions(max_duration=3)
    assert isinstance(o.max_duration, float)
    assert o.max_duration == 3.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"include_speech": 1},
        {"include_speech": None},
        {"max_chars_per_line": 0},
        {"max_chars_per_line": True},
        {"max_chars_per_line": 4.0},
        {"max_lines": -1},
        {"max_lines": "2"},
        {"max_duration": 0},
        {"max_duration": math.nan},
        {"max_duration": True},
    ],
)
def test_options_reject_invalid_values(kwargs):
    with pytest.raises(AlganConfigurationError):
        st._SubtitleOptions(**kwargs)


# ---------------------------------------------------------------- _caption_text


def test_caption_text_strips_blank_lines_and_normalizes_endings():
    assert st._caption_text("  a  \r\n\r\n  b\rc\n\n") == "a\nb\nc"


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "a\0b", None, 5, b"abc"])
def test_caption_text_rejects_empty_nul_and_non_str(text):
    with pytest.raises(AlganConfigurationError):
        st._caption_text(text)


def test_caption_text_keeps_unicode():
    assert st._caption_text("héllo → 世界") == "héllo → 世界"


# ---------------------------------------------------------------- _wrapped


def test_wrapped_respects_width_and_authored_breaks():
    out = st._wrapped("one two three four\nfive", 9)
    assert out == "one two\nthree\nfour\nfive"
    assert all(len(line) <= 9 for line in out.splitlines())


def test_wrapped_does_not_break_long_words_or_hyphens():
    assert st._wrapped("supercalifragilistic well-timed", 5) == (
        "supercalifragilistic\nwell-timed"
    )


def test_wrapped_drops_blank_lines():
    assert st._wrapped("a\n\nb", 10) == "a\nb"


# ---------------------------------------------------------------- _timestamp


@pytest.mark.parametrize(
    ("ms", "sep", "expected"),
    [
        (0, ",", "00:00:00,000"),
        (1, ".", "00:00:00.001"),
        (59_999, ",", "00:00:59,999"),
        (60_000, ",", "00:01:00,000"),
        (3_600_000, ".", "01:00:00.000"),
        (3_723_004, ",", "01:02:03,004"),
        (100 * 3_600_000, ",", "100:00:00,000"),
    ],
)
def test_timestamp_formatting(ms, sep, expected):
    assert st._timestamp(ms, sep) == expected


# ---------------------------------------------------------------- _serialize_cues


def test_serialize_empty_srt_and_vtt():
    assert st._serialize_cues([], "srt") == ""
    assert st._serialize_cues([], "vtt") == "WEBVTT\n\n"


def test_serialize_srt_basic_and_sorted():
    out = st._serialize_cues([(2.0, 3.0, "second"), (0.5, 1.25, "first")], "srt")
    assert out == (
        "1\n00:00:00,500 --> 00:00:01,250\nfirst\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nsecond\n\n"
    )


def test_serialize_vtt_header_and_dot_separator():
    out = st._serialize_cues([(0.0, 1.0, "hi")], "vtt")
    assert out == "WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\nhi\n\n"


def test_serialize_vtt_escapes_markup_and_arrow():
    out = st._serialize_cues([(0, 1, "a < b && c > d --> e")], "vtt")
    payload = out.split("\n")[4]
    assert payload == "a &lt; b &amp;&amp; c &gt; d --&gt; e"
    # Only one timing arrow per cue.
    assert out.count("-->") == 1


def test_serialize_srt_writes_text_literally():
    out = st._serialize_cues([(0, 1, "x<0 & y>1")], "srt")
    assert "x<0 & y>1" in out


def test_serialize_rounds_half_up_and_enforces_min_duration():
    out = st._serialize_cues([(0.0004, 0.0004, "a"), (1.0005, 1.0001, "b")], "srt")
    assert "00:00:00,000 --> 00:00:00,001" in out
    assert "00:00:01,001 --> 00:00:01,002" in out


def test_serialize_clamps_negative_start():
    out = st._serialize_cues([(-1.0, -0.5, "a")], "srt")
    assert "00:00:00,000 --> 00:00:00,001" in out


def test_serialize_multiline_payload_has_no_blank_lines():
    out = st._serialize_cues([(0, 1, "a\n\n\nb")], "srt")
    assert out == "1\n00:00:00,000 --> 00:00:01,000\na\nb\n\n"


def test_serialize_stable_for_equal_starts():
    out = st._serialize_cues([(0, 1, "A"), (0, 2, "B")], "srt")
    assert out.index("A") < out.index("B")


def test_serialize_rejects_empty_text():
    with pytest.raises(AlganConfigurationError):
        st._serialize_cues([(0, 1, "   ")], "srt")


def test_serialize_rollover_to_next_second():
    out = st._serialize_cues([(0.9996, 59.9996, "a")], "srt")
    assert "00:00:01,000 --> 00:01:00,000" in out


# ---------------------------------------------------------------- _parts


def test_parts_normalizes_like_alignment():
    assert _parts("Well-timed, it's _ok_!") == ["WELL", "TIMED", "ITS", "OK"]
    assert _parts("—, ...") == []


# ---------------------------------------------------------------- speech cues


def _row(text, timings, start=0.0, end=None):
    words = []
    for m, t in zip(re.finditer(r"\S+", text), timings):
        s, e = t if t is not None else (None, None)
        words.append(
            {
                "text": m.group(),
                "offset": m.start(),
                "end_offset": m.end(),
                "start": s,
                "end": e,
            }
        )
    if end is None:
        end = max((w["end"] for w in words if w["end"] is not None), default=0.0)
    return {"text": text, "start": start, "end": end, "words": words}


def test_speech_cues_unavailable_row_yields_nothing():
    row = _row("hello", [(0, 1)])
    row["start"] = None
    assert st._speech_cues(row, st._SubtitleOptions()) == []


def test_speech_cues_single_group():
    row = _row("hello there", [(0.0, 0.5), (0.5, 1.0)])
    assert st._speech_cues(row, st._SubtitleOptions()) == [(0.0, 1.0, "hello there")]


def test_speech_cues_split_on_max_duration():
    row = _row("a b c", [(0, 2), (2, 4), (4, 6)])
    cues = st._speech_cues(row, st._SubtitleOptions(max_duration=3))
    assert cues == [(0, 2, "a"), (2, 4, "b"), (4, 6, "c")]


def test_speech_cues_split_on_lines():
    row = _row("aaa bbb ccc", [(0, 0.1), (0.1, 0.2), (0.2, 0.3)])
    cues = st._speech_cues(row, st._SubtitleOptions(max_chars_per_line=3, max_lines=1))
    assert [c[2] for c in cues] == ["aaa", "bbb", "ccc"]


def test_speech_cues_wrap_within_cue():
    row = _row("aaa bbb ccc", [(0, 0.1), (0.1, 0.2), (0.2, 0.3)])
    cues = st._speech_cues(row, st._SubtitleOptions(max_chars_per_line=7, max_lines=2))
    assert cues == [(0, 0.3, "aaa bbb\nccc")]


def test_speech_cues_paragraph_break_splits():
    row = _row("one two\n\nthree", [(0, 1), (1, 2), (2, 3)])
    cues = st._speech_cues(row, st._SubtitleOptions())
    assert cues == [(0, 2, "one two"), (2, 3, "three")]


def test_speech_cues_single_newline_is_kept():
    row = _row("one\ntwo", [(0, 1), (1, 2)])
    assert st._speech_cues(row, st._SubtitleOptions()) == [(0, 2, "one\ntwo")]


def test_speech_cues_attach_untimed_punctuation():
    row = _row("hello , world !", [(0, 1), None, (1, 2), None])
    cues = st._speech_cues(row, st._SubtitleOptions(max_duration=1))
    assert cues == [(0, 1, "hello ,"), (1, 2, "world !")]


def test_speech_cues_leading_punctuation_joins_first_word():
    row = _row("“ quoted", [None, (0, 1)])
    assert st._speech_cues(row, st._SubtitleOptions()) == [(0, 1, "“ quoted")]


def test_speech_cues_skip_untimed_spoken_words():
    row = _row("a b c", [(0, 1), None, (2, 3)])
    cues = st._speech_cues(row, st._SubtitleOptions())
    assert cues == [(0, 3, "a c")]


def test_speech_cues_only_punctuation_uses_row_bounds():
    row = _row("... !", [None, None], start=1.0, end=2.0)
    assert st._speech_cues(row, st._SubtitleOptions()) == [(1.0, 2.0, "... !")]


def test_speech_cues_overlong_single_word_stays_one_cue():
    row = _row("long", [(0, 10)])
    assert st._speech_cues(row, st._SubtitleOptions(max_duration=1)) == [
        (0, 10, "long")
    ]


# ---------------------------------------------------------------- transcript helpers


def _matches(text):
    return list(re.finditer(r"\S+", text))


def test_aligned_words_maps_hyphenated_words():
    words = _aligned_words(
        _matches("well-timed, ok"),
        [("WELL", 0, 0.5), ("TIMED", 0.5, 1.0), ("OK", 1.0, 1.5)],
        2.0,
    )
    assert [(w.start, w.end) for w in words] == [(0, 1.0), (1.0, 1.5)]


@pytest.mark.parametrize(
    "timestamps",
    [
        [("WRONG", 0, 1)],
        [("HI", 1, 0.5)],
        [("HI", 0, 5)],
        [("HI", 0, math.nan)],
        [("HI", 0, 1), ("EXTRA", 1, 1.5)],
        [(3, 0, 1)],
        [("HI", "x", 1)],
    ],
)
def test_aligned_words_rejects_mismatch(timestamps):
    assert _aligned_words(_matches("hi"), timestamps, 2.0) is None


def test_speech_block_estimates_by_letter_weight():
    clip = SimpleNamespace(duration=3.0)
    block = _SpeechBlock.from_clip("ab , abcd", clip)
    assert block.timing == "estimated"
    assert [(w.start, w.end) for w in block.words] == [
        (0.0, 1.0),
        (None, None),
        (1.0, 3.0),
    ]


def test_speech_block_invalid_duration_becomes_zero():
    block = _SpeechBlock.from_clip("hi", SimpleNamespace(duration=math.nan))
    assert block.duration == 0.0


def test_snapshot_transcript_clips_to_duration():
    block = _SpeechBlock.from_clip("aa bb", SimpleNamespace(duration=2.0))
    block.start_time_func = lambda: 0.5
    snap = snapshot_transcript([block], 2.0)
    row = snap["blocks"][0]
    assert (row["start"], row["end"]) == (0.5, 2.0)
    assert [(w["start"], w["end"]) for w in row["words"]] == [(0.5, 1.5), (1.5, 2.0)]
    assert snap["text"] == "aa bb\n\n"


def test_snapshot_transcript_unscheduled_block_is_unavailable():
    block = _SpeechBlock.from_clip("aa", SimpleNamespace(duration=1.0))
    row = snapshot_transcript([block], 5.0)["blocks"][0]
    assert row["timing"] == "unavailable"
    assert row["start"] is None


# ---------------------------------------------------------------- _scene_cues


def _scene(captions=(), blocks=()):
    return SimpleNamespace(
        audio_manager=SimpleNamespace(
            _speech_blocks=list(blocks), _subcaptions=list(captions)
        )
    )


def _cap(text, start, duration, offset=0.0):
    return st._Subcaption(text, duration, offset, lambda: start)


def test_scene_cues_clip_sort_and_drop_outside():
    scene = _scene(
        [
            _cap("late", 5.0, 10.0),
            _cap("early", -1.0, 2.0),
            _cap("gone", 20.0, 1.0),
            _cap("before", -5.0, 1.0),
            _cap("offset", 1.0, 1.0, offset=0.5),
        ]
    )
    cues = st._scene_cues(scene, st._SubtitleOptions(), duration=8.0)
    assert cues == [(0.0, 1.0, "early"), (1.5, 2.5, "offset"), (5.0, 8.0, "late")]


def test_scene_cues_include_speech_toggle():
    block = _SpeechBlock.from_clip("hi", SimpleNamespace(duration=1.0))
    block.start_time_func = lambda: 0.0
    scene = _scene([], [block])
    assert st._scene_cues(scene, st._SubtitleOptions(), duration=2.0) == [
        (0.0, 1.0, "hi")
    ]
    assert (
        st._scene_cues(scene, st._SubtitleOptions(include_speech=False), duration=2.0)
        == []
    )


@pytest.mark.parametrize("duration", [-1.0, math.nan, "x"])
def test_scene_cues_reject_bad_duration(duration):
    with pytest.raises(AlganConfigurationError):
        st._scene_cues(_scene(), st._SubtitleOptions(), duration=duration)


def test_scene_cues_reject_non_finite_caption_origin():
    scene = _scene([_cap("x", math.nan, 1.0)])
    with pytest.raises(AlganConfigurationError, match="Caption start"):
        st._scene_cues(scene, st._SubtitleOptions(), duration=2.0)


def test_scene_cues_zero_duration_scene_is_empty():
    scene = _scene([_cap("x", 0.0, 1.0)])
    assert st._scene_cues(scene, st._SubtitleOptions(), duration=0.0) == []


# ---------------------------------------------------------------- _write_subtitles


def test_write_subtitles_overwrite_and_exclusive(tmp_path):
    path = tmp_path / "out.srt"
    st._write_subtitles(path, [(0, 1, "a")], "srt", overwrite=True)
    st._write_subtitles(path, [(0, 1, "b")], "srt", overwrite=False)
    assert "a" in path.read_text(encoding="utf-8")
    st._write_subtitles(path, [(0, 1, "c")], "srt", overwrite=True)
    text = path.read_bytes().decode("utf-8")
    assert "c" in text
    assert "\r" not in text
