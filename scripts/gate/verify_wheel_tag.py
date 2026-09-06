#!/usr/bin/env python3
r"""Fail the build when a Linux wheel's platform tag promises more than it keeps.

``manylinux_X_Y`` is a claim about the oldest glibc the wheel runs on, and
Quadrants' own build stamps ``manylinux_2_27`` unconditionally
(``qd_build/entry.py::build_wheel``) -- it is a constant in their source, not a
measurement. Nothing downstream checks it either: pip reads the tag, decides
the wheel is installable on any glibc at or above it, and the mistake surfaces
as ``version GLIBC_2.34 not found`` at ``import quadrants`` on a user's
machine, long after the release.

This is the check that makes the tag a fact. ``auditwheel show`` reads the
versioned symbols the extension module actually references and answers with the
lowest policy the wheel satisfies; this compares that answer to the tag the
matrix says the wheel will be stamped with, and exits non-zero if the wheel
needs a newer glibc than the tag admits::

    python scripts/gate/verify_wheel_tag.py dist/quadrants-*.whl \\
        --expect manylinux_2_28_x86_64

A *stricter* wheel than promised is the failure. The reverse -- a wheel that
would run on something older than the tag says -- is safe and passes: it only
means the container could have been older, which costs reach rather than
correctness.

``--from-file`` reads saved ``auditwheel show`` output instead of running it,
which is how ``tests/unit_tests/test_quadrants_wheels.py`` exercises the
parsing without an auditwheel install or a real wheel.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

#: `auditwheel show` states its verdict in this sentence. The tag is quoted in
#: current auditwheel and was not always, hence the optional quotes. Matched
#: against whitespace-collapsed text because auditwheel hard-wraps its prose:
#: the sentence really does arrive split as "is consistent with the\nfollowing
#: platform tag", and a regex written for one line silently finds nothing.
#: A platform tag has no `.` in it, so word characters alone end the match at
#: the closing quote or the sentence's period without needing to name either.
VERDICT_RE = re.compile(
    r"consistent with the following platform tag:\s*[\"']?([A-Za-z0-9_]+)"
)

#: The pre-PEP-600 names, in the glibc versions PEP 600 gives them. auditwheel
#: still prints these for old policies, and a comparison that did not know them
#: would read `manylinux2014` as "unparseable" and fail a wheel that is fine.
LEGACY_ALIASES = {
    "manylinux1": (2, 5),
    "manylinux2010": (2, 12),
    "manylinux2014": (2, 17),
}

MANYLINUX_RE = re.compile(r"^manylinux_(\d+)_(\d+)_(.+)$")
LEGACY_RE = re.compile(r"^(manylinux1|manylinux2010|manylinux2014)_(.+)$")


def parse_tag(tag: str) -> tuple[tuple[int, int], str] | None:
    """``manylinux_2_28_x86_64`` -> ``((2, 28), "x86_64")``; None if not one."""
    match = MANYLINUX_RE.match(tag)
    if match:
        return (int(match.group(1)), int(match.group(2))), match.group(3)
    match = LEGACY_RE.match(tag)
    if match:
        return LEGACY_ALIASES[match.group(1)], match.group(2)
    return None


def auditwheel_verdict(text: str) -> str:
    """The platform tag `auditwheel show` says the wheel is consistent with."""
    match = VERDICT_RE.search(" ".join(text.split()))
    if not match:
        raise SystemExit(
            "could not find auditwheel's verdict in its output. It reports the "
            "tag as 'is consistent with the following platform tag: \"...\"'.\n"
            f"What it printed:\n{text}"
        )
    return match.group(1)


def check(measured: str, expected: str) -> list[str]:
    """Why `expected` is not a tag this wheel can keep -- empty when it is."""
    problems = []
    want = parse_tag(expected)
    if want is None:
        return [f"--expect {expected!r} is not a manylinux tag"]

    got = parse_tag(measured)
    if got is None:
        # `linux_x86_64` is auditwheel's answer for a wheel that satisfies no
        # manylinux policy at all -- normally an external shared library that
        # no policy allows, which stamping cannot fix.
        return [
            f"auditwheel puts this wheel at {measured!r}, which is not a "
            f"manylinux policy at all, so {expected!r} cannot be stamped on it. "
            "Run `auditwheel show` on it: something it links is outside the "
            "policy's allowed set."
        ]

    (measured_glibc, measured_arch) = got
    (expected_glibc, expected_arch) = want
    if measured_arch != expected_arch:
        problems.append(
            f"architecture mismatch: the wheel is {measured_arch}, the tag says "
            f"{expected_arch}. A leg is building on the wrong runner."
        )
    if measured_glibc > expected_glibc:
        problems.append(
            f"the wheel needs glibc {measured_glibc[0]}.{measured_glibc[1]} "
            f"({measured}) but would be stamped {expected}, which claims "
            f"{expected_glibc[0]}.{expected_glibc[1]}. pip would install it on "
            "systems it cannot run on. Build in an older container, or stamp "
            "the tag the wheel earns."
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--expect",
        required=True,
        help="the platform tag the wheel is about to be stamped with",
    )
    parser.add_argument(
        "--from-file",
        type=Path,
        help="read `auditwheel show` output from here instead of running it",
    )
    args = parser.parse_args(argv)

    if args.from_file is not None:
        text = args.from_file.read_text(encoding="utf-8")
    else:
        completed = subprocess.run(
            [sys.executable, "-m", "auditwheel", "show", str(args.wheel)],
            capture_output=True,
            text=True,
        )
        text = completed.stdout + completed.stderr
        if completed.returncode != 0:
            raise SystemExit(f"auditwheel show failed:\n{text}")

    measured = auditwheel_verdict(text)
    problems = check(measured, args.expect)
    if problems:
        raise SystemExit(
            f"{args.wheel.name}: refusing to stamp a tag the wheel cannot "
            "keep:\n  - " + "\n  - ".join(problems)
        )
    print(
        f"{args.wheel.name}: auditwheel says {measured}, stamping "
        f"{args.expect} -- the tag is a promise this wheel keeps"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
