r"""Turn a ``list_notebook_session_output`` response into a readable log.

That MCP tool returns one JSON object whose ``log`` key is **a JSON string**,
not a string: it parses to a list of ``{"stream_name", "time", "data"}``
records, one per output chunk, and the transcript is the ``data`` fields
joined. The whole response also reliably blows the tool-result token cap and
gets spilled to a file, so the response is read from that file rather than from
the tool output.

Usage::

    uv run python scripts/kaggle/read_output.py <spilled-tool-result.json> \\
        --out /tmp/run.log

It writes the transcript and prints a digest of the lines that decide whether a
run is worth reading: the render device first (a run on the wrong machine is
worthless and looks fine), then the step boundaries, verdicts and the RESULTS
line.

It also accepts a raw transcript (``--raw``) so the same digest can be taken of
a log that was already decoded.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# In reading order: which machine, then what happened on it. The device
# patterns come first because everything below them is meaningless if the run
# landed on a P100 and fell back to the CPU.
DIGEST_PATTERNS: list[tuple[str, str]] = [
    (
        "device",
        r"ALGAN_DEVICE|Rendering device set to|render device:|^GPU |Tesla|NVIDIA",
    ),
    ("steps", r"^=== STEP |^--- step "),
    ("timing", r"^RUN [12] |warm \(steady state\)|median|^\s*(SKIP|OK|FAILED)\s"),
    ("verdicts", r"PASS:|FAIL:|VACUOUS|Traceback|Error:|error:"),
    ("results", r"^RESULTS |^ALL STEPS OK|^FAILED steps:|sha256"),
]


def transcript_from_response(payload: dict) -> str:
    """Join the ``data`` fields of the response's ``log`` records."""
    log = payload.get("log")
    if log is None:
        raise SystemExit("the response has no 'log' key; is this the right file?")
    if isinstance(log, str):
        log = json.loads(log)
    if isinstance(log, dict):  # a single record, seen on very short runs
        log = [log]
    return "".join(entry.get("data", "") for entry in log)


def digest(text: str, context: int = 0) -> list[str]:
    lines = text.splitlines()
    out: list[str] = []
    for label, pattern in DIGEST_PATTERNS:
        regex = re.compile(pattern)
        hits = [line for line in lines if regex.search(line)]
        out.append(f"--- {label} ({len(hits)} lines)")
        out.extend(hit[:400] for hit in hits[: 60 if context == 0 else context])
        if len(hits) > 60:
            out.append(f"    ... {len(hits) - 60} more")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "path", help="the spilled tool-result JSON, or a raw transcript"
    )
    parser.add_argument(
        "--raw", action="store_true", help="path is already a transcript"
    )
    parser.add_argument("--out", default=None, help="write the transcript here")
    parser.add_argument(
        "--no-digest", action="store_true", help="only write the transcript"
    )
    args = parser.parse_args(argv)

    source = Path(args.path).read_text(encoding="utf-8", errors="replace")
    text = source if args.raw else transcript_from_response(json.loads(source))

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"# {len(text)} chars -> {args.out}", file=sys.stderr)

    if not args.no_digest:
        print("\n".join(digest(text)))
    elif not args.out:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
