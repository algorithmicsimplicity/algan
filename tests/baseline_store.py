"""Resolving a render suite's baseline directory, from the tree or a release.

The full-render and path-traced baselines are the repository's weight problem:
they are ~4 MB mp4s re-committed whole on every rebaseline, they account for
most of the blobs in history, and -- unlike ``tests/fast`` -- **CI never
compares against them**. They gate locally, on whichever machine rendered
them (see the header of ``tests/full_renders/test_full_renders.py``). So every
clone pays for an artifact almost no clone uses.

This module is what lets those baselines live outside git, as a tarball
attached to a GitHub release, without changing what the suites do when they
are present. Resolution order, for one (suite, device key):

1. ``ALGAN_BASELINE_DIR``: a directory holding ``<suite>/<key>/`` trees, for
   a machine that keeps its own baselines or has no network. Its answer is
   final -- a missing directory under it skips the comparison rather than
   falling through to a download.
2. The in-repo ``expected_outputs_<key>/``, when it exists and has files.
   This is why introducing the fetcher changes nothing on its own: while the
   baselines are still committed, they are still what runs.
3. The cache: ``~/.algan/cache/baselines/<tag>/`` (see :func:`_cache_root`),
   accepted only when the marker written at extraction time matches the
   sha256 pinned in ``tests/baselines.json`` (a partial or tampered extract
   is re-fetched, never trusted).
4. A one-time download of that release asset, verified against the same
   sha256 and extracted atomically (temp directory + rename), so a killed
   download never poisons the cache.

Every failure returns ``None`` after one warning, and the caller skips its
comparison -- which is the behaviour those suites already have on a machine
with no baselines for its device. A rendering regression must never be
reported as "the download failed", and a failed download must never be
reported as a passing test: the suites print the skip reason, and
``tests/README.md`` repeats the standing warning that a render suite which
skipped compared nothing.

``tests/baselines.json`` carries a null ``tag`` until the assets are actually
uploaded. That is not a broken state: step 3 and 4 are simply skipped, so an
unpublished pointer behaves exactly like an offline machine, silently, and
the committed baselines answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
import warnings
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
POINTER_PATH = TESTS_ROOT / "baselines.json"

_DOWNLOAD_TIMEOUT_SECONDS = 60
#: Per-process memo, keyed by ``(suite, key)``: an entry maps to the resolved
#: directory, or to ``None`` once this process has warned about it. Without it
#: six scenes in one suite pay six timeouts on an offline machine.
_resolved: dict = {}


class BaselinePointerError(RuntimeError):
    """``tests/baselines.json`` is missing, unparseable or malformed.

    Raised rather than warned: an unreadable pointer file is a repository
    error that every suite would hit, not a property of the machine running
    them, and degrading to "no baselines" would turn it into a wall of skips
    that look like an unbaselined device.
    """


def load_pointer(path: Path = POINTER_PATH) -> dict:
    """The parsed pointer file, validated enough to trust its shape."""
    try:
        pointer = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BaselinePointerError(f"{path} does not exist") from exc
    except (OSError, ValueError) as exc:
        raise BaselinePointerError(f"{path} could not be read: {exc}") from exc

    if not isinstance(pointer, dict):
        raise BaselinePointerError(f"{path} must hold a JSON object")
    archives = pointer.get("archives")
    if not isinstance(archives, dict):
        raise BaselinePointerError(f"{path} has no 'archives' object")
    for name, entry in archives.items():
        if not isinstance(entry, dict):
            raise BaselinePointerError(f"{path}: archive {name!r} is not an object")
        for field in ("file", "sha256"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise BaselinePointerError(f"{path}: archive {name!r} has no {field!r}")
        if len(entry["sha256"]) != 64:
            raise BaselinePointerError(
                f"{path}: archive {name!r} has a malformed sha256"
            )
    return pointer


def archive_key(suite: str, key: str) -> str:
    """The ``archives`` key for one suite and device, e.g. ``full_renders/cuda``."""
    return f"{suite}/{key}"


def _cache_root() -> Path:
    """The machine-global baseline cache.

    Deliberately ``_startup._CACHE_DIRECTORY`` (which honours ``ALGAN_HOME``
    and ``ALGAN_CACHE_DIR``) rather than ``SETTINGS.paths.cache_directory``:
    both render suites re-point the live setting at a per-suite scratch
    directory for the duration of a render, and a 10 MB download that lands
    there is re-fetched on every run. Measured, not assumed -- the first
    version of this cached into ``tests/path_traced/algan_cache/``.
    """
    from algan.settings._startup import _CACHE_DIRECTORY

    return Path(_CACHE_DIRECTORY) / "baselines"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _has_files(directory: Path) -> bool:
    try:
        return any(entry.is_file() for entry in directory.iterdir())
    except OSError:
        return False


def _safe_members(archive: tarfile.TarFile, destination: Path) -> list:
    """The members of ``archive`` that may be written under ``destination``.

    Python 3.12's ``filter="data"`` would do this, but ``requires-python`` is
    3.9, where ``TarFile.extractall`` still happily writes through ``..`` and
    follows a symlink out of the destination. The archive is sha256-pinned, so
    this is defence in depth rather than the only check -- but the pin is only
    as good as the pointer file, and an extractor that cannot escape its
    destination is worth more than an argument about who can edit JSON.
    """
    destination = destination.resolve()
    members = []
    for member in archive.getmembers():
        if not member.isfile():
            # Directories are implied by the file paths; links and devices
            # have no business in a directory of mp4s.
            continue
        target = (destination / member.name).resolve()
        if target != destination and destination not in target.parents:
            raise BaselinePointerError(
                f"baseline archive member {member.name!r} escapes its destination"
            )
        members.append(member)
    # A list, not a generator: every member is checked before the first one is
    # written, so a bad archive extracts nothing at all rather than everything
    # up to the offending member.
    return members


def extract_archive(archive_path: Path, destination: Path) -> None:
    """Extract ``archive_path`` into a fresh ``destination`` directory."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(destination, members=_safe_members(archive, destination))


def _install(archive_path: Path, target: Path, digest: str) -> None:
    """Put the archive's contents at ``target``, atomically, with its marker.

    The marker is written last and inside the staged directory, so a killed
    extraction leaves a directory that fails its marker check and is refetched
    rather than one that looks complete.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=str(target.parent), prefix=".staging-"))
    try:
        extract_archive(archive_path, staging)
        (staging / ".sha256").write_text(digest, encoding="utf-8")
        if target.exists():
            shutil.rmtree(target)
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _cached(target: Path, digest: str) -> bool:
    marker = target / ".sha256"
    try:
        return marker.read_text(encoding="utf-8").strip() == digest
    except OSError:
        return False


def _download(url: str, target: Path, digest: str) -> bool:
    """Fetch ``url`` into the cache at ``target``. False (warned) on failure."""
    with tempfile.TemporaryDirectory() as scratch:
        archive_path = Path(scratch) / "baselines.tar.gz"
        try:
            with (
                urllib.request.urlopen(
                    url, timeout=_DOWNLOAD_TIMEOUT_SECONDS
                ) as response,
                open(archive_path, "wb") as handle,
            ):
                shutil.copyfileobj(response, handle)
        except Exception as exc:  # URLError, timeout, HTTP error, disk full...
            warnings.warn(
                f"Could not download the render baselines from {url} ({exc}); "
                f"the comparison will be skipped. Set ALGAN_BASELINE_DIR to a "
                f"local copy, or re-baseline on this machine.",
                stacklevel=2,
            )
            return False
        actual = _sha256(archive_path)
        if actual != digest:
            warnings.warn(
                f"The baseline archive at {url} has sha256 {actual}, but "
                f"tests/baselines.json pins {digest}; refusing it and skipping "
                f"the comparison.",
                stacklevel=2,
            )
            return False
        _install(archive_path, target, digest)
    return True


def resolve_baseline_dir(
    suite: str,
    key: str,
    local_dir: Path,
    *,
    pointer_path: Path = POINTER_PATH,
    use_cache: bool = True,
) -> Path | None:
    """The directory to compare ``suite``'s renders against, or ``None``.

    ``None`` means "no baselines are available for this device", which every
    caller turns into a skip. Any reason for it has already been warned about,
    once per process.
    """
    memo_key = (suite, key, str(local_dir))
    if use_cache and memo_key in _resolved:
        return _resolved[memo_key]

    resolved = _resolve_uncached(suite, key, local_dir, pointer_path)
    if use_cache:
        _resolved[memo_key] = resolved
    return resolved


def _resolve_uncached(
    suite: str, key: str, local_dir: Path, pointer_path: Path
) -> Path | None:
    override = (os.getenv("ALGAN_BASELINE_DIR") or "").strip()
    if override:
        # Final by design: a machine that says where its baselines live must
        # not silently end up comparing against a downloaded set instead.
        candidate = Path(override) / suite / key
        if _has_files(candidate):
            return candidate
        warnings.warn(
            f"ALGAN_BASELINE_DIR is set, but {candidate} holds no baselines; "
            f"skipping the {suite} comparison for {key}.",
            stacklevel=3,
        )
        return None

    if _has_files(local_dir):
        return local_dir

    pointer = load_pointer(pointer_path)
    tag = pointer.get("tag")
    entry = pointer["archives"].get(archive_key(suite, key))
    if not tag or entry is None:
        # Nothing published for this device (or nothing published at all).
        # Silent: this is the same state as an unbaselined device, which the
        # suites already report as a skip with their own message.
        return None

    target = _cache_root() / str(tag) / suite / key
    digest = entry["sha256"]
    if _cached(target, digest):
        return target

    if os.getenv("ALGAN_NO_BASELINE_DOWNLOAD") == "1":
        warnings.warn(
            f"ALGAN_NO_BASELINE_DOWNLOAD is set and {suite}/{key} is not in "
            f"the cache; skipping the comparison.",
            stacklevel=3,
        )
        return None

    base = str(pointer.get("base_url") or "").rstrip("/")
    url = f"{base}/{tag}/{entry['file']}"
    if not _download(url, target, digest):
        return None
    return target


def _reset_for_tests() -> None:
    """Forget the memoized answers (the tests exercise several outcomes)."""
    _resolved.clear()
