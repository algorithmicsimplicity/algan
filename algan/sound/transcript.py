"""Scene-owned narration metadata; no synthesis, alignment or rendering here.

Word times on a clip are seconds from the *trimmed clip's* start. Parent
animation contexts can move an audio effect, but do not time-stretch its audio,
so only the block's origin is resolved lazily; word offsets stay in audio time.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass


def _parts(text: str) -> list[str]:
    """The same word normalization used by recorded-narration alignment."""
    return [
        normalized
        for part in text.replace("-", " ").split()
        if (normalized := re.sub(r"[\W_]+", "", part).upper())
    ]


@dataclass(frozen=True)
class _Word:
    offset: int
    end_offset: int
    start: float | None
    end: float | None


def _aligned_words(matches, timestamps, duration):
    """Map normalized alignment words back to the original display spelling.

    For example, one displayed ``well-timed,`` consumes the alignment's WELL
    and TIMED. Punctuation and whitespace are never reconstructed from the
    aligner's uppercase text. Reject mismatches instead of silently shifting
    every subsequent word to the wrong timestamp.
    """
    try:
        rows = []
        previous = 0.0
        for text, start, end in timestamps:
            if not isinstance(text, str):
                return None
            start, end = float(start), float(end)
            if not (
                math.isfinite(start)
                and math.isfinite(end)
                and previous <= start <= end <= duration + 1e-6
            ):
                return None
            rows.append((_parts(text), start, min(end, duration)))
            previous = start
        words = []
        cursor = 0
        for match in matches:
            expected = _parts(match.group())
            if not expected:
                words.append(_Word(match.start(), match.end(), None, None))
                continue
            first = cursor
            actual = []
            while cursor < len(rows) and len(actual) < len(expected):
                actual.extend(rows[cursor][0])
                cursor += 1
            if actual != expected:
                return None
            words.append(
                _Word(match.start(), match.end(), rows[first][1], rows[cursor - 1][2])
            )
        return tuple(words) if cursor == len(rows) else None
    except (TypeError, ValueError, OverflowError):
        return None


@dataclass
class _SpeechBlock:
    text: str
    duration: float
    words: tuple[_Word, ...]
    timing: str
    start_time_func: Callable[[], float] | None = None

    @classmethod
    def from_clip(cls, text: str, clip) -> _SpeechBlock:
        text = text.strip(" ")  # Match AudioManager.video_transcript.
        duration = float(clip.duration)
        if not math.isfinite(duration) or duration < 0:
            duration = 0.0
        matches = list(re.finditer(r"\S+", text))
        timestamps = getattr(clip, "algan_word_timestamps", None)
        words = (
            _aligned_words(matches, timestamps, duration)
            if timestamps is not None
            else None
        )
        if words is not None:
            return cls(text, duration, words, "aligned")

        # Legacy/custom clips and pyttsx3 do not necessarily expose timings.
        # This is explicitly an estimate, never described as forced alignment.
        weights = [len("".join(_parts(match.group()))) for match in matches]
        total = sum(weights)
        cursor = 0
        words = []
        for match, weight in zip(matches, weights):
            start = duration * cursor / total if total and weight else None
            cursor += weight
            end = duration * cursor / total if total and weight else None
            words.append(_Word(match.start(), match.end(), start, end))
        return cls(text, duration, tuple(words), "estimated")


def snapshot_transcript(blocks: Iterable[_SpeechBlock], duration: float) -> dict:
    """Resolve narration once, before the viewer's worker starts.

    Keep all authored text, including suppressed/zero-length blocks and words
    outside the viewable duration, but give inaudible words no seek target.
    Returning only JSON primitives also severs all links to mutable contexts.
    """
    rows = []
    for block in blocks:
        origin = (
            float(block.start_time_func())
            if block.start_time_func is not None
            else None
        )
        audible = (
            origin is not None
            and math.isfinite(origin)
            and origin < duration
            and origin + block.duration > 0
            and block.duration > 0
        )
        words = []
        for word in block.words:
            start = end = None
            if audible and word.start is not None:
                a, b = origin + word.start, origin + word.end
                if a < duration and b > 0 and b > a:
                    start, end = max(0.0, a), min(duration, b)
            words.append(
                {
                    "text": block.text[word.offset : word.end_offset],
                    "offset": word.offset,
                    "end_offset": word.end_offset,
                    "start": start,
                    "end": end,
                }
            )
        rows.append(
            {
                "text": block.text,
                "start": max(0.0, origin) if audible else None,
                "end": min(duration, origin + block.duration) if audible else None,
                "timing": block.timing if audible else "unavailable",
                "words": words,
            }
        )
    return {"text": "".join(row["text"] + "\n\n" for row in rows), "blocks": rows}
