r"""Package the heavy render baselines for a GitHub release, and pin them.

The full-render and path-traced baselines do not gate CI and are per machine
(see ``tests/baseline_store.py``), so they are hosted as release assets rather
than carried in every clone. This script is the producer side of that: it
turns each local ``expected_outputs_<key>/`` into one tarball and rewrites
``tests/baselines.json`` with the tag and sha256 the test harness verifies.

    <venv-python> scripts/package_baselines.py --tag baselines-2026-09-03

Then upload everything it wrote and commit the pointer::

    gh release create baselines-2026-09-03 dist/baselines/*.tar.gz \
        --title "Render baselines" --notes "..."
    git add tests/baselines.json && git commit

A machine normally has only the device baselines it just rendered. When a
previous release is already pinned, unchanged archives are therefore downloaded
from that release, verified, and copied into the new output directory. The new
tag can always contain the complete set without requiring one machine to render
every device/platform baseline.

The tarballs are byte-reproducible -- members sorted, mtimes, ownership and
modes normalized, gzip header mtime zeroed -- so re-running this on another
machine over the same mp4s produces the same sha256. That is what makes the
pinned digest a fact about the baselines rather than about the machine that
happened to package them, and it lets anyone verify an uploaded asset.

``--verify`` re-packages any local hosted baselines without writing and reports
whether they match the published pointer. A clean checkout has no heavy
baselines locally, so verification instead requires a published tag.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = REPO_ROOT / "tests"
POINTER_PATH = TESTS_ROOT / "baselines.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dist" / "baselines"

#: Suites whose heavy baselines are hosted. ``tests/fast`` deliberately stays
#: in git: it is small and is the only render baseline normal CI compares.
HOSTED_SUITES = ("full_renders", "path_traced")


def display(path: Path) -> str:
    """``path`` relative to the repository when it is inside it, else as-is."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def discover(suite: str):
    """Yield ``(device_key, directory)`` for one suite's local baselines."""
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
    """Package every local hosted baseline directory; return its archives map."""
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def carry_forward(pointer: dict, output_dir: Path, archives: dict) -> dict:
    """Copy unchanged archives from the currently published release.

    The pointer has one release tag for the complete archive set. After the
    heavy MP4s leave git, a rebaseline machine generally owns only the device
    tree it changed, so every other archive is copied from the previous release
    and verified against the old pointer before it is allowed into the new tag.
    """
    recorded = pointer.get("archives", {})
    missing = [name for name in recorded if name not in archives]
    if not missing:
        return archives

    tag = pointer.get("tag")
    base = str(pointer.get("base_url") or "").rstrip("/")
    if not tag or not base:
        raise RuntimeError(
            "cannot carry forward missing baselines without a published tag/base_url"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in missing:
        entry = recorded[name]
        destination = output_dir / entry["file"]
        url = f"{base}/{tag}/{entry['file']}"
        print(f"Carrying forward {name} from {url}")
        with (
            urllib.request.urlopen(url, timeout=60) as response,
            open(destination, "wb") as handle,
        ):
            while block := response.read(1 << 20):
                handle.write(block)
        digest = _sha256(destination)
        size = destination.stat().st_size
        if digest != entry.get("sha256") or size != entry.get("size"):
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"carried-forward archive {name} failed its pinned digest/size"
            )
        archives[name] = dict(entry)
    return archives


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tag",
        help="new release tag; written to tests/baselines.json. Omit with --verify.",
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
        help="package local hosted baselines temporarily and verify they match "
        "the published pointer; write nothing.",
    )
    args = parser.parse_args(argv)

    if args.verify:
        import tempfile

        with tempfile.TemporaryDirectory() as scratch:
            archives = build(Path(scratch))
        pointer = json.loads(POINTER_PATH.read_text(encoding="utf-8"))
        recorded = pointer.get("archives", {})
        tag = pointer.get("tag")
        if not tag:
            print(
                "tests/baselines.json has no published release tag.",
                file=sys.stderr,
            )
            return 1
        stale = [
            name
            for name in sorted(archives)
            if archives[name].get("sha256") != recorded.get(name, {}).get("sha256")
        ]
        if stale:
            print(
                "\ntests/baselines.json does not describe the local hosted baselines:\n  "
                + "\n  ".join(stale)
                + "\n\nRe-run without --verify (and upload the result) after a "
                "rebaseline.",
                file=sys.stderr,
            )
            return 1
        if archives:
            print("\ntests/baselines.json matches every local hosted baseline.")
        else:
            print(
                f"No local hosted baselines are checked out; pointer is published "
                f"at tag {tag} with {len(recorded)} archive(s)."
            )
        return 0

    if not args.tag:
        parser.error("--tag is required unless --verify is given")

    pointer = json.loads(POINTER_PATH.read_text(encoding="utf-8"))
    local_archives = build(args.output_dir)
    if not local_archives:
        parser.error(
            "no local hosted baseline directories found; render/rebaseline at "
            "least one hosted suite before publishing a new baseline tag"
        )
    archives = carry_forward(pointer, args.output_dir, local_archives)
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
        f"with missing assets makes every fetch warn and every comparison skip."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
