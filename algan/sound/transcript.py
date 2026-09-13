"""Scene-local narration cues, using the very same clock as AudioEffect.

A clip's private ``_algan_word_timings`` contains (normalized word, start, end)
rows in *clip-relative seconds*. Recorded narration already computes these when
selecting a subclip. Sources without alignment get explicitly estimated timings;
opening a viewer must not synthesize audio or download an alignment model.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass


def _normalized_words(text):
    return [
        normalized
        for token in text.replace("-", " ").split()
        if (normalized := re.sub(r"[\W_]+", "", token).upper())
    ]


@dataclass(frozen=True)
class _TranscriptWord:
    text: str
    before: str
    start: float | None
    end: float | None


@dataclass(frozen=True)
class _SpeechCue:
    text: str
    words: tuple[_TranscriptWord, ...]
    after: str
    timing: str
    start_time_func: Callable[[], float] | None


def _speech_cue(text, clip, start_time_func):
    """Capture text/alignment now, but leave the effect's start time lazy."""
    tokens = list(re.finditer(r"\S+", text))
    groups = [_normalized_words(token.group()) for token in tokens]
    expected = [word for group in groups for word in group]
    duration = float(clip.duration)
    usable = bool(expected) and math.isfinite(duration) and duration > 0
    rows = getattr(clip, "_algan_word_timings", None)
    aligned = []
    if usable and rows is not None:
        try:
            previous = 0.0
            for word, start, end in rows:
                start, end = float(start), float(end)
                if not (
                    math.isfinite(start)
                    and math.isfinite(end)
                    and previous <= start < end <= duration
                ):
                    raise ValueError("invalid clip-relative word interval")
                aligned.append((str(word), start, end))
                previous = start
        except (TypeError, ValueError, OverflowError):
            aligned = []
    if [row[0] for row in aligned] != expected:
        aligned = []

    # Punctuation alone is not a narrated word. Preserve it, including its
    # spacing, without inventing a timing or dropping it from the transcript.
    timing = "unavailable"
    intervals = [(None, None)] * len(tokens)
    if usable and start_time_func is not None:
        if aligned:
            timing = "aligned"
            cursor = 0
            for i, group in enumerate(groups):
                if group:
                    intervals[i] = (
                        aligned[cursor][1],
                        aligned[cursor + len(group) - 1][2],
                    )
                    cursor += len(group)
        else:
            timing = "estimated"
            weights = [sum(map(len, group)) for group in groups]
            total, cursor = sum(weights), 0
            for i, weight in enumerate(weights):
                if weight:
                    intervals[i] = (
                        duration * cursor / total,
                        duration * (cursor + weight) / total,
                    )
                    cursor += weight
    words = []
    previous = 0
    for token, (start, end) in zip(tokens, intervals):
        words.append(
            _TranscriptWord(token.group(), text[previous : token.start()], start, end)
        )
        previous = token.end()
    return _SpeechCue(text, tuple(words), text[previous:], timing, start_time_func)


def _snapshot_transcript(cues):
    """Detach JSON-ready data at viewer launch; later authoring cannot alter it.

    Parent animation contexts can move a sound's start, but AudioEffect does not
    stretch its waveform. Only the start is resolved through the timeline;
    word offsets and durations must *not* be scaled with animation runtimes.
    """
    cues = tuple(cues)
    blocks = []
    for cue in cues:
        start = (
            float(cue.start_time_func()) if cue.start_time_func is not None else None
        )
        if start is not None and not math.isfinite(start):
            start = None
        blocks.append(
            {
                "text": cue.text,
                "timing": cue.timing if start is not None else "unavailable",
                "words": [
                    {
                        "text": word.text,
                        "before": word.before,
                        "start": start + word.start
                        if start is not None and word.start is not None
                        else None,
                        "end": start + word.end
                        if start is not None and word.end is not None
                        else None,
                    }
                    for word in cue.words
                ],
                "after": cue.after,
            }
        )
    return {"text": "\n\n".join(cue.text for cue in cues), "blocks": blocks}
