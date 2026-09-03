r"""Package the heavy render baselines for a GitHub release, and pin them.

The full-render and path-traced baselines do not gate CI and are per machine
(see ``tests/baseline_store.py``), so they are hosted as release assets rather
than carried in every clone. This script is the producer side of that: it
turns each committed ``expected_outputs_<key>/`` into one tarball and rewrites
``tests/baselines.json`` with the tag and the sha256 the test harness will
verify against.

    uv run python scripts/package_baselines.py --tag baselines-2026-09-03

Then upload what it wrote and commit the pointer::

    gh release create baselines-2026-09-03 dist/baselines/*.tar.gz \\
        --title "Render baselines" --notes "..."
    git add tests/baselines.json && git commit

The tarballs are byte-reproducible -- members sorted, mtimes, ownership and
modes normalized, gzip header mtime zeroed -- so re-running this on another
machine over the same mp4s produces the same sha256. That is what makes the
pinned digest a fact about the baselines rather than about the machine that
happened to package them, and it lets anyone verify an uploaded asset.

``--verify`` re-packages without writing anything and reports whether the
pointer file still describes the working tree, which is the check that catches
a rebaseline that was committed but never uploaded.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = REPO_ROOT / "tests"
POINTER_PATH = TESTS_ROOT / "baselines.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dist" / "baselines"

#: The suites whose baselines are hosted. ``tests/fast`` is deliberately not
#: here: it is 368 KB, it is the only render baseline CI compares against, and
#: keeping it in git means an ordinary clone and an ordinary CI run never touch
#: the network for baselines at all.
HOSTED_SUITES = ("full_renders", "path_traced")


def display(path: Path) -> str:
    """``path`` relative to the repository when it is inside it, else as-is."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def discover(suite: str):
    """Yield ``(device_key, directory)`` for one suite's committed baselines."""
    suite_dir = TESTS_ROOT / suite
    for directory in sorted(suite_dir.glob("expected_outputs_*")):
        if not directory.is_dir():
            continue
        if not any(entry.is_file() for entry in directory.iterdir()):
            continue
        yield directory.name[len("expected_outputs_") :], directory


def write_archive(source: Path, destination: Path) -> str:
    """Write a reproducible ``.tar.gz`` of ``source``; return its sha256."""
    files = sorted(
        (path for path in source.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(source).as_posix(),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 and a fixed filename keep the gzip header itself constant; without
    # this the digest changes on every run.
    with (
        open(destination, "wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in files:
            info = archive.gettarinfo(
                str(path), arcname=path.relative_to(source).as_posix()
            )
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with open(path, "rb") as handle:
                archive.addfile(info, handle)

    digest = hashlib.sha256()
    with open(destination, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(output_dir: Path) -> dict:
    """Package every hosted baseline directory; return the ``archives`` map."""
    archives = {}
    for suite in HOSTED_SUITES:
        for key, directory in discover(suite):
            name = f"{suite}-{key}.tar.gz"
            path = output_dir / name
            digest = write_archive(directory, path)
            archives[f"{suite}/{key}"] = {
                "file": name,
                "sha256": digest,
                "size": path.stat().st_size,
            }
            print(f"{display(path)}  {digest}  {path.stat().st_size} bytes")
    return archives


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tag",
        help="the release tag the assets will be uploaded under; written to "
        "tests/baselines.json. Omit with --verify.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"where to write the tarballs (default: {display(DEFAULT_OUTPUT_DIR)})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="package into a temporary directory and report whether "
        "tests/baselines.json already describes the working tree; write nothing.",
    )
    args = parser.parse_args(argv)

    if args.verify:
        import tempfile

        with tempfile.TemporaryDirectory() as scratch:
            archives = build(Path(scratch))
        pointer = json.loads(POINTER_PATH.read_text(encoding="utf-8"))
        recorded = pointer.get("archives", {})
        stale = [
            name
            for name in sorted(set(archives) | set(recorded))
            if archives.get(name, {}).get("sha256")
            != recorded.get(name, {}).get("sha256")
        ]
        if stale:
            print(
                "\ntests/baselines.json does not describe the working tree:\n  "
                + "\n  ".join(stale)
                + "\n\nRe-run without --verify (and upload the result) after a "
                "rebaseline.",
                file=sys.stderr,
            )
            return 1
        print("\ntests/baselines.json matches the working tree.")
        return 0

    if not args.tag:
        parser.error("--tag is required unless --verify is given")

    archives = build(args.output_dir)
    pointer = json.loads(POINTER_PATH.read_text(encoding="utf-8"))
    pointer["tag"] = args.tag
    pointer["archives"] = archives
    POINTER_PATH.write_text(
        json.dumps(pointer, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(
        f"\nWrote {display(POINTER_PATH)} for tag {args.tag}.\n"
        f"Upload with:\n"
        f"  gh release create {args.tag} {display(args.output_dir)}/*.tar.gz \\\n"
        f"      --title 'Render baselines {args.tag}' --notes 'Baselines for ...'\n"
        f"Assets must be uploaded before the pointer is pushed: a pushed tag "
        f"with no assets makes every fetch warn and every comparison skip."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
