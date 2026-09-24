#!/usr/bin/env python3
"""Check a local video's metadata and optionally decode it with FFmpeg.

No network access or file modification. This does NOT assess visual correctness,
spoken words, constant frame pacing, or meaningful alpha/compositing behavior.
Exit 0 means requested mechanical checks passed; exit 1 means verification failed.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def positive_float(value: str) -> float:
    try:
        number = float(Fraction(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError("must be a number or fraction, e.g. 30000/1001") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def numeric(value: Any) -> float | None:
    try:
        number = float(Fraction(str(value)))
    except (ValueError, TypeError, ZeroDivisionError):
        return None
    return number if math.isfinite(number) else None


def executable(value: str) -> str:
    resolved = shutil.which(value)
    if resolved:
        return resolved
    path = Path(value).expanduser()
    if path.is_file():
        return str(path.resolve())
    raise RuntimeError(f"Executable not found: {value}")


def run(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout,
                                check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{Path(command[0]).name} exceeded the {timeout:g}s timeout; verification is incomplete.") from exc
    except OSError as exc:
        raise RuntimeError(f"Cannot execute {command[0]}: {exc}") from exc
    if result.returncode:
        details = result.stderr.strip()[-3000:]
        raise RuntimeError(f"{Path(command[0]).name} failed (exit {result.returncode}): {details}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--width", type=positive_int)
    parser.add_argument("--height", type=positive_int)
    parser.add_argument("--fps", type=positive_float,
                        help="Expected reported average frame rate; fractions supported.")
    parser.add_argument("--duration", type=positive_float,
                        help="Expected video duration in seconds, not frame count.")
    parser.add_argument("--duration-tolerance", type=positive_float, default=0.1)
    parser.add_argument("--fps-tolerance", type=positive_float, default=0.001)
    audio = parser.add_mutually_exclusive_group()
    audio.add_argument("--require-audio", action="store_true")
    audio.add_argument("--forbid-audio", action="store_true")
    parser.add_argument("--decode", action="store_true",
                        help="Decode the first video stream and all audio streams to a null sink.")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--timeout", type=positive_float, default=120.0,
                        help="Per-subprocess timeout in seconds; increase for long decodes.")
    args = parser.parse_args(argv)
    report: dict[str, Any] = {"verification_passed": False, "errors": []}
    try:
        path = args.video.expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"Video does not exist: {path}")
        if path.stat().st_size == 0:
            raise RuntimeError(f"Video is empty: {path}")
        probe = run([executable(args.ffprobe), "-v", "error", "-show_streams",
                     "-show_format", "-of", "json", str(path)], args.timeout)
        try:
            metadata = json.loads(probe.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("FFprobe returned invalid JSON.") from exc
        streams = metadata.get("streams", [])
        videos = [s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic", 0)]
        audios = [s for s in streams if s.get("codec_type") == "audio"]
        if not videos:
            raise RuntimeError("No playable video stream found.")
        video = videos[0]
        fps = numeric(video.get("avg_frame_rate"))
        if not fps or fps <= 0:
            fps = numeric(video.get("r_frame_rate"))
        duration = numeric(video.get("duration"))
        duration_source = "video_stream"
        if duration is None:
            duration = numeric(metadata.get("format", {}).get("duration"))
            duration_source = "container_fallback"
        errors: list[str] = []
        for key in ("width", "height"):
            expected = getattr(args, key)
            actual = video.get(key)
            if expected is not None and actual != expected:
                errors.append(f"Expected {key} {expected}; found {actual}.")
        if args.fps is not None and (fps is None or abs(fps - args.fps) > args.fps_tolerance):
            errors.append(f"Expected fps {args.fps:g}; found {fps}.")
        if args.duration is not None and (duration is None or abs(duration - args.duration) > args.duration_tolerance):
            errors.append(f"Expected duration {args.duration:g}s; found {duration}s ({duration_source}).")
        if args.require_audio and not audios:
            errors.append("Expected an audio stream; none found.")
        if args.forbid_audio and audios:
            errors.append("Expected no audio streams; at least one was found.")
        report = {
            "path": str(path), "size_bytes": path.stat().st_size,
            "video_stream_index": video.get("index"), "width": video.get("width"),
            "height": video.get("height"), "reported_fps": fps,
            "duration_seconds": duration, "duration_source": duration_source,
            "codec": video.get("codec_name"), "pixel_format": video.get("pix_fmt"),
            "audio_streams": [{"codec": s.get("codec_name"),
                               "sample_rate": s.get("sample_rate"),
                               "channels": s.get("channels"),
                               "duration_seconds": numeric(s.get("duration"))}
                              for s in audios],
            "full_decode_passed": None, "errors": errors,
            "scope": "Metadata/optional decode only; no visual, semantic audio, pacing, or alpha-content assessment.",
        }
        if args.decode:
            # Explicit stream index avoids accidentally decoding an attached cover image.
            run([executable(args.ffmpeg), "-v", "error", "-xerror", "-nostdin",
                 "-i", str(path), "-map", f"0:{video['index']}", "-map", "0:a?",
                 "-f", "null", "-"], args.timeout)
            report["full_decode_passed"] = True
        report["verification_passed"] = not errors
    except RuntimeError as exc:
        report.setdefault("errors", []).append(str(exc))
        report["verification_passed"] = False
    print(json.dumps(report, indent=2))
    return 0 if report["verification_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
