#!/usr/bin/env python3
"""Every archive in ``tests/baselines.json`` is fetchable *and* is the right file.

The heavy render baselines live as release assets under a ``baselines-<date>``
tag with a committed pointer (``tests/README.md``, "Where the heavy baselines
live"). A suite that cannot resolve its baselines **skips**, so a pointer that
has drifted from what is hosted does not fail anything -- it quietly stops
comparing six dense scenes and the path tracer while reading as a clean run.

This gate exists because a HEAD request is not enough, and that is not
hypothetical. Under ``baselines-2026-09-09.1`` two assets were uploaded with a
hyphen missing from the filename, and the two failures looked nothing alike:

* ``full_renders-cpu_eager.tar.gz`` was simply absent -- a HEAD check catches
  this one.
* ``path_traced-cpu.tar.gz`` *existed* and returned 200, but held the
  **superseded** archive from the previous tag. ``7cbd375`` had rebaselined
  that entry, and the newer file was the one wearing the mangled name. A HEAD
  check passes, ``tests/baseline_store.py`` then rejects it on the sha256, and
  the suite skips.

So the pointer's sha256 is the thing worth checking, and checking it means
downloading. That is ~20 MB and a few seconds in the cheapest job of the
release, against a failure mode whose whole character is that nothing goes red.

When an archive is missing, the report lists what the tag *does* carry, because
the mangled-name case is obvious the moment the two lists sit next to each
other and invisible otherwise.

Usage::

    python scripts/gate/verify_baseline_pointer.py
    python scripts/gate/verify_baseline_pointer.py --pointer tests/baselines.json

A null ``tag`` is the supported bootstrap state (``baseline_store`` treats it as
an unbaselined offline machine) and is reported as skipped, not failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT))

from tests.baseline_store import BaselinePointerError, load_pointer  # noqa: E402

#: Generous: these are ~10 MB files and a release must not fail on a slow hop.
_TIMEOUT_SECONDS = 180


class VerificationError(Exception):
    """One archive is missing, unreadable, or is not the file the pointer names."""


def asset_names(repository: str, tag: str) -> list[str] | None:
    """Asset filenames GitHub reports under ``tag``, or ``None`` if unreadable.

    Only ever used to make a failure legible, so every error is swallowed: an
    unauthenticated rate-limit must not turn a real mismatch into a crash, nor
    a clean run into a red one.
    """
    url = f"https://api.github.com/repos/{repository}/releases/tags/{tag}"
    request = urllib.request.Request(
        url, headers={"Accept": "application/vnd.github+json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return [a["name"] for a in json.load(response).get("assets", [])]
    except Exception:
        return None


def verify_archive(url: str, expected_sha256: str, expected_size: int | None) -> int:
    """Download ``url`` and return its size, or raise :class:`VerificationError`.

    Streamed rather than read whole: the archives are ~10 MB each today, but
    nothing about the pointer format caps that, and a gate is a bad place to
    discover a memory ceiling.
    """
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                size += len(chunk)
    except urllib.error.HTTPError as exc:
        raise VerificationError(f"HTTP {exc.code} {exc.reason}") from exc
    except Exception as exc:
        raise VerificationError(f"{type(exc).__name__}: {exc}") from exc

    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise VerificationError(
            f"sha256 mismatch -- the asset exists but is a different file "
            f"(expected {expected_sha256[:12]}…, got {actual[:12]}…, {size} bytes)"
        )
    # Checked after the digest so the message names the useful discrepancy
    # first; a size mismatch with a matching digest is not reachable in
    # practice, but a pointer whose size field has drifted is worth saying.
    if expected_size is not None and size != expected_size:
        raise VerificationError(
            f"size mismatch: pointer says {expected_size} bytes, got {size}"
        )
    return size


def verify_pointer(pointer_path: Path, repository: str) -> int:
    """Verify every archive. Returns a process exit code."""
    try:
        pointer = load_pointer(pointer_path)
    except BaselinePointerError as exc:
        print(f"::error::{exc}")
        return 1

    tag = pointer.get("tag")
    if tag is None:
        print(f"{pointer_path}: tag is null -- nothing is published, nothing to check")
        return 0
    base_url = pointer["base_url"]
    archives = pointer["archives"]
    print(f"{pointer_path}: {len(archives)} archives at {tag}\n")

    failures: dict[str, str] = {}
    total = 0
    for key, entry in sorted(archives.items()):
        url = f"{base_url}/{tag}/{entry['file']}"
        try:
            total += verify_archive(url, entry["sha256"], entry.get("size"))
        except VerificationError as exc:
            print(f"  FAIL {key:26} {entry['file']:34} {exc}")
            failures[key] = entry["file"]
        else:
            print(f"  ok   {key:26} {entry['file']:34} {entry['sha256'][:12]}…")

    if not failures:
        print(f"\nall {len(archives)} archives verified ({total / 1e6:.1f} MB)")
        return 0

    print(f"\n{len(failures)} of {len(archives)} archives did not verify.")
    hosted = asset_names(repository, str(tag))
    if hosted is not None:
        print(f"\nWhat {tag} actually carries:")
        for name in sorted(hosted):
            print(f"  {name}")
        missing = {f for f in failures.values() if f not in hosted}
        if missing:
            print(
                "\nNamed by the pointer but absent from the release: "
                + ", ".join(sorted(missing))
                + "\nIf a hosted name looks like one of these with a character "
                "dropped, the upload mangled it -- re-upload under the exact "
                "name rather than editing the pointer to match."
            )
    print(
        f"\n::error::baseline archives did not verify: {sorted(failures)}. "
        f"Publish them under tag {tag!r} (scripts/package_baselines.py) before "
        f"releasing -- an unresolvable baseline makes the render suites skip, "
        f"which reads as a clean run."
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pointer",
        type=Path,
        default=REPO_ROOT / "tests" / "baselines.json",
        help="the pointer file to verify (default: tests/baselines.json)",
    )
    parser.add_argument(
        "--repository",
        default="algorithmicsimplicity/algan",
        help="owner/repo whose releases host the archives, for the failure report",
    )
    args = parser.parse_args(argv)
    return verify_pointer(args.pointer, args.repository)


if __name__ == "__main__":
    raise SystemExit(main())
