"""Subtitle cues derived from authored captions and the existing speech metadata.

No audio synthesis, alignment, materialization or rendering happens here. Cue
origins are resolved at export, after enclosing animation contexts have closed.
"""

from __future__ import annotations

import html
import math
import os
import re
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from algan.errors import AlganConfigurationError
from algan.settings import SETTINGS
from algan.sound.transcript import _parts, snapshot_transcript


def _seconds(value, name, *, positive=False):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AlganConfigurationError(f"{name} must be finite seconds") from exc
    if (
        isinstance(value, bool)
        or not math.isfinite(result)
        or (positive and result <= 0)
    ):
        raise AlganConfigurationError(
            f"{name} must be {'positive ' if positive else ''}finite seconds"
        )
    return result


@dataclass(frozen=True)
class _Subcaption:
    text: str
    duration: float
    offset: float
    origin: Callable[[], float]


@dataclass(frozen=True)
class _SubtitleOptions:
    include_speech: bool = True
    max_chars_per_line: int = 42
    max_lines: int = 2
    max_duration: float = 6.0

    def __post_init__(self):
        if not isinstance(self.include_speech, bool):
            raise AlganConfigurationError("include_speech must be a boolean")
        for name in ("max_chars_per_line", "max_lines"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise AlganConfigurationError(f"{name} must be a positive integer")
        object.__setattr__(
            self,
            "max_duration",
            _seconds(self.max_duration, "max_duration", positive=True),
        )


def _caption_text(text):
    if not isinstance(text, str) or not text.strip() or "\0" in text:
        raise AlganConfigurationError(
            "Caption content must be nonempty text without NUL"
        )
    # Blank payload lines end an SRT/VTT cue. Keep authored line breaks, while
    # removing empty lines and normalizing platform-specific line endings.
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _wrapped(text, width):
    return "\n".join(
        line
        for authored_line in text.splitlines()
        for line in textwrap.wrap(
            authored_line, width=width, break_long_words=False, break_on_hyphens=False
        )
    )


def _speech_cues(row, options):
    if row["start"] is None:
        return []
    cues = []
    paragraph = []
    for word in row["words"]:
        if paragraph and re.search(
            r"\n\s*\n", row["text"][paragraph[-1]["end_offset"] : word["offset"]]
        ):
            cues.extend(_speech_paragraph_cues(row, paragraph, options))
            paragraph = []
        paragraph.append(word)
    if paragraph:
        cues.extend(_speech_paragraph_cues(row, paragraph, options))
    return cues


def _speech_paragraph_cues(row, paragraph, options):
    # Keep untimed punctuation with a spoken word, so it cannot become a cue
    # spanning the entire clip or bring a clipped-out word back into the text.
    words = []
    for word in paragraph:
        if _parts(word["text"]):
            item = dict(word)
            if not words:
                item["offset"] = paragraph[0]["offset"]
            words.append(item)
        elif words:
            words[-1]["end_offset"] = word["end_offset"]
    if not words:
        words = [dict(word, start=row["start"], end=row["end"]) for word in paragraph]
    cues = []
    group = []

    def content(items):
        parts = []
        previous = None
        for item in items:
            if previous is not None:
                gap = row["text"][previous["end_offset"] : item["offset"]]
                parts.append(gap if gap.isspace() else " ")
            parts.append(row["text"][item["offset"] : item["end_offset"]])
            previous = item
        return "".join(parts)

    def bounds(items):
        return (
            min(word["start"] for word in items),
            max(word["end"] for word in items),
        )

    def flush():
        if group:
            start, end = bounds(group)
            cues.append(
                (start, end, _wrapped(content(group), options.max_chars_per_line))
            )

    for word in words:
        if word["start"] is None:
            continue
        candidate = [*group, word]
        start, end = bounds(candidate)
        if group and (
            len(_wrapped(content(candidate), options.max_chars_per_line).splitlines())
            > options.max_lines
            or end - start > options.max_duration
        ):
            flush()
            group = []
        group.append(word)
    flush()
    return cues


def _scene_cues(scene, options, *, duration=None):
    if duration is None:
        duration = float(scene._recorded_end_time_for_render())
    duration = _seconds(duration, "Scene duration")
    if duration < 0:
        raise AlganConfigurationError("Scene duration must be non-negative")
    cues = []
    if options.include_speech:
        transcript = snapshot_transcript(scene.audio_manager._speech_blocks, duration)
        for row in transcript["blocks"]:
            cues.extend(_speech_cues(row, options))
    for caption in scene.audio_manager._subcaptions:
        start = _seconds(caption.origin(), "Caption start") + caption.offset
        cues.append((start, start + caption.duration, caption.text))
    # Negative offsets and captions extending past the video are legal, but
    # subtitles only describe the part that lies in this Scene's video.
    return sorted(
        (
            (max(0.0, start), min(duration, end), text)
            for start, end, text in cues
            if text and start < duration and end > 0 and end > start
        ),
        key=lambda cue: cue[0],
    )


def _subtitle_destination(file_path, subtitle_format, *, default=None):
    from algan.utils.algan_utils import _resolve_output_destination

    default_name = file_path is None
    if default_name:
        file_path = default if default is not None else SETTINGS.paths.output_filename
    raw = os.fspath(file_path)
    path = Path(raw).expanduser()
    directory = path.is_dir() or raw.endswith(
        tuple(separator for separator in (os.sep, os.altsep) if separator)
    )
    suffix = "" if default_name or directory else path.suffix.lower().lstrip(".")
    if subtitle_format is None:
        subtitle_format = suffix or "srt"
    if not isinstance(subtitle_format, str) or subtitle_format.lower() not in (
        "srt",
        "vtt",
    ):
        raise AlganConfigurationError("Subtitle format must be 'srt' or 'vtt'")
    subtitle_format = subtitle_format.lower()
    if suffix and suffix != subtitle_format:
        raise AlganConfigurationError(
            "Subtitle extension must match its srt/vtt format"
        )
    if default_name:
        file_path = path.with_suffix("." + subtitle_format)
    destination = _resolve_output_destination(file_path, "." + subtitle_format)
    if directory:
        destination = destination.with_suffix("." + subtitle_format)
    return destination, subtitle_format


def _timestamp(milliseconds, separator):
    seconds, ms = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{ms:03d}"


def _serialize_cues(cues, subtitle_format):
    blocks = ["WEBVTT\n"] if subtitle_format == "vtt" else []
    separator = "." if subtitle_format == "vtt" else ","
    for index, (start, end, text) in enumerate(sorted(cues, key=lambda cue: cue[0]), 1):
        start_ms = max(0, math.floor(start * 1000 + 0.5))
        end_ms = max(start_ms + 1, math.floor(end * 1000 + 0.5))
        # Subtitle text is literal, including math comparisons and '-->'.
        payload = html.escape(_caption_text(text), quote=False)
        blocks.append(
            f"{index}\n{_timestamp(start_ms, separator)} --> "
            f"{_timestamp(end_ms, separator)}\n{payload}\n"
        )
    return "\n".join(blocks) + ("\n" if blocks else "")


def _write_subtitles(destination, cues, subtitle_format, overwrite):
    content = _serialize_cues(cues, subtitle_format)
    try:
        with destination.open(
            "w" if overwrite else "x", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(content)
    except FileExistsError:
        if overwrite:
            raise
    return destination
