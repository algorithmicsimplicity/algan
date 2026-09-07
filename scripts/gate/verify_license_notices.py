#!/usr/bin/env python3
"""Every vendored license notice reaches the built distributions.

`algan/external_libraries/` is copied from three MIT-licensed projects, and MIT
requires the copyright and permission notice in every copy. A PyPI artifact is
a copy, and it cannot be edited after upload -- only yanked -- so this is a gate
rather than a convention.

The specific way it went wrong once is worth stating, because the shape recurs:
Manim is **double-licensed** and ships two notices, `LICENSE` (3blue1brown LLC)
and `LICENSE.community` (the Manim Community Developers). `pyproject.toml` said
``algan/external_libraries/*/LICENSE``, which matches one of them, and the built
wheel carried only the 3blue1brown half -- not the half covering the 0.21.0 code
actually in the tree. Nothing failed: the wheel installed, imported and rendered.

So the expected set is **derived from the repository** rather than written out
here. Add a vendored library and the gate covers it without being edited; narrow
the packaging glob and the gate fails. The one thing hard-coded is that Manim's
pair must both be there, because that pair is the trap this exists for.

Usage::

    python scripts/gate/verify_license_notices.py dist/*.whl dist/*.tar.gz

With no arguments it checks everything under ``dist/``.
"""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Notices that must exist in the tree *and* reach every distribution. Adding a
#: vendored library extends this automatically; the assertion below is what
#: keeps a shrinking glob from also shrinking the gate.
REQUIRED_PAIR = (
    "algan/external_libraries/manim/LICENSE",
    "algan/external_libraries/manim/LICENSE.community",
)


def expected_notices() -> list[str]:
    """Notice paths, relative to the repository root, that must be shipped."""
    found = ["LICENSE"]
    found += sorted(
        str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        for path in sorted(REPO_ROOT.glob("algan/external_libraries/*/LICENSE*"))
    )
    missing_pair = [name for name in REQUIRED_PAIR if name not in found]
    if missing_pair:
        raise SystemExit(
            f"{missing_pair} is missing from the repository. Manim is "
            "double-licensed and both notices must be vendored; "
            "scripts/vendor_manim.py writes them."
        )
    return found


def _check_wheel(path: Path, expected: list[str]) -> list[str]:
    """Notices absent from the wheel, either as files or as declarations."""
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        try:
            metadata_name = next(
                name for name in names if name.endswith(".dist-info/METADATA")
            )
        except StopIteration:
            return ["<no .dist-info/METADATA>"]
        licenses = metadata_name[: -len("METADATA")] + "licenses/"
        declared = {
            line.split(": ", 1)[1]
            for line in archive.read(metadata_name).decode("utf-8").splitlines()
            if line.startswith("License-File: ")
        }

    problems = [
        f"{name} (not in {licenses})"
        for name in expected
        if licenses + name not in names
    ]
    problems += [
        f"{name} (no License-File header)" for name in expected if name not in declared
    ]
    return problems


def _check_sdist(path: Path, expected: list[str]) -> list[str]:
    """Notices absent from the sdist, which ships them in the source tree."""
    with tarfile.open(path) as archive:
        names = set(archive.getnames())
    roots = {name.split("/", 1)[0] for name in names if "/" in name}
    if len(roots) != 1:
        return [f"<sdist has {len(roots)} top-level directories, expected 1>"]
    root = roots.pop()
    return [name for name in expected if f"{root}/{name}" not in names]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "distributions",
        nargs="*",
        type=Path,
        help="wheels and sdists to check (default: everything under dist/)",
    )
    args = parser.parse_args(argv)

    paths = args.distributions or sorted(
        [*REPO_ROOT.glob("dist/*.whl"), *REPO_ROOT.glob("dist/*.tar.gz")]
    )
    if not paths:
        print("::error::no distributions to check -- build them first")
        return 1

    expected = expected_notices()
    print(f"expecting {len(expected)} notices: {expected}")

    failed = False
    for path in paths:
        if path.suffix == ".whl":
            problems = _check_wheel(path, expected)
        elif path.name.endswith(".tar.gz"):
            problems = _check_sdist(path, expected)
        else:
            print(f"::error::{path.name}: not a wheel or sdist")
            failed = True
            continue

        if problems:
            failed = True
            print(f"::error::{path.name} is missing license notices:")
            for problem in problems:
                print(f"    {problem}")
        else:
            print(f"  ok  {path.name}: all {len(expected)} notices present")

    if failed:
        print(
            "::error::MIT requires the notice in every copy, and a PyPI upload "
            "cannot be edited afterwards. Check `license-files` in pyproject.toml "
            "-- the glob must end in LICENSE*, not LICENSE."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
