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
2. A local ``expected_outputs_<key>/``, when it exists and has files. Normal
   checkouts do not carry these anymore; this is a developer's freshly
   rendered/rebaselined copy and deliberately wins over the release asset.
3. The cache: ``~/.algan/cache/baselines/<tag>/`` (see :func:`_cache_root`),
   accepted only when the marker written at extraction time matches the
   sha256 pinned in ``tests/baselines.json`` (a partial or tampered extract
   is re-fetched, never trusted).
4. A one-time download of that release asset, verified against the same
   sha256 and extracted atomically (temp directory + rename), so a killed
   download never poisons the cache.

Every failure returns ``None`` from :func:`resolve_baseline_dir` after one
warning, and records why. A rendering regression must never be reported as
"the download failed" -- but nor may an absent baseline be reported as a
passing test, and a *skip* is close enough to green to have hidden exactly
that: ``tests/full_renders`` skipped all six scenes for the whole life of the
``cpu_eager`` key, comparing nothing while reading as a clean run.

So the render suites call :func:`require_baseline_dir`, which raises
:class:`BaselinesUnavailableError` carrying that reason, and an unresolvable
baseline is a test failure. Rendering a first baseline for a new device is
still possible: the ``ALGAN_UPDATE_*_BASELINES`` paths write the tree and
never reach the comparison.

macOS is the one exception, because it is the one platform where nothing is
published and no machine can fix that by fetching:
``ALGAN_ALLOW_UNBASELINED_MACOS=1`` lets the four ``macos_*`` keys skip.
:func:`macos_opt_out_permits` is where that is decided, and how narrowly.

The committed ``tests/baselines.json`` carries a published release tag. A null
tag remains a supported bootstrap/test state: steps 3 and 4 are skipped and
the resolver behaves like an unbaselined offline machine, silently.
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
from typing import NamedTuple

TESTS_ROOT = Path(__file__).resolve().parent
POINTER_PATH = TESTS_ROOT / "baselines.json"

_DOWNLOAD_TIMEOUT_SECONDS = 60
#: Per-process memo, keyed by ``(suite, key, local_dir)``: an entry maps to
#: ``(directory, unavailable)`` -- exactly one of which is ``None``. Without it
#: six scenes in one suite pay six timeouts on an offline machine, and warn six
#: times about it.
_resolved: dict = {}

#: Set to ``1`` to let the macOS device keys below *skip* their comparison
#: instead of failing it. See :func:`macos_opt_out_permits`.
MACOS_OPT_OUT_ENV = "ALGAN_ALLOW_UNBASELINED_MACOS"

#: The keys that opt-out covers: a Mac renders on the CPU or on MPS, and
#: nothing is committed or published for either. The ``_mpsfriendly`` variants
#: of both are covered too -- that mode is documented as not bit-identical, so
#: it is keyed apart and equally unbaselined.
MACOS_OPT_OUT_KEYS = frozenset({"macos_cpu", "macos_mps"})


class BaselineUnavailable(NamedTuple):
    """Why nothing resolved, and whether the device is merely unbaselined.

    The distinction is the whole value of the type. "This device has no
    baselines" is a state a machine can legitimately be in and opt out of;
    "the download failed" and "the digest did not match" are not, and must
    keep failing however the opt-out is set, or the knob would re-hide the
    class of problem that motivated failing in the first place.
    """

    reason: str
    unbaselined: bool = False


def macos_opt_out_permits(key: str) -> bool:
    """True when ``key`` is a macOS key excused by the opt-out variable.

    Set ``ALGAN_ALLOW_UNBASELINED_MACOS=1`` on a Mac to get the pre-failure
    behaviour back for that machine: the scene still renders -- which is most
    of what these suites exercise, kernel compilation, tessellation, LaTeX,
    fonts and the encoder -- and only the pixel comparison is skipped.

    It is deliberately not a general "allow missing baselines" switch. Every
    other device either has published baselines or is one somebody should
    publish, and a blanket knob would let the CPU and CUDA suites go quiet
    again. It also only ever applies to :attr:`BaselineUnavailable.unbaselined`
    -- a Mac with a failed download or a bad digest still fails.
    """
    if os.getenv(MACOS_OPT_OUT_ENV) != "1":
        return False
    return key.partition("_mpsfriendly")[0] in MACOS_OPT_OUT_KEYS


class BaselinesUnavailableError(RuntimeError):
    """No baselines could be resolved for one (suite, device key).

    Raised by :func:`require_baseline_dir` so a render suite *fails* rather
    than skipping. Its message is the reason the resolver recorded -- an
    unpublished device key, a download that did not come back, a digest that
    did not match -- because "no baselines" on its own sends the reader
    looking for a rendering bug that is not there.

    ``unbaselined`` says the cause was simply that this device has no
    baselines, which is the only cause :func:`macos_opt_out_permits` may
    excuse.
    """

    def __init__(self, unavailable: BaselineUnavailable) -> None:
        super().__init__(unavailable.reason)
        self.unbaselined = unavailable.unbaselined


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
    3.10, where ``TarFile.extractall`` still happily writes through ``..`` and
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


def _download(url: str, target: Path, digest: str) -> str | None:
    """Fetch ``url`` into the cache at ``target``. The reason, on failure."""
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
            reason = (
                f"Could not download the render baselines from {url} ({exc}). "
                f"Set ALGAN_BASELINE_DIR to a local copy, or re-baseline on "
                f"this machine."
            )
            warnings.warn(reason, stacklevel=2)
            return reason
        actual = _sha256(archive_path)
        if actual != digest:
            reason = (
                f"The baseline archive at {url} has sha256 {actual}, but "
                f"tests/baselines.json pins {digest}; refusing it rather than "
                f"comparing against bytes nobody published."
            )
            warnings.warn(reason, stacklevel=2)
            return reason
        _install(archive_path, target, digest)
    return None


def resolve_baseline_dir(
    suite: str,
    key: str,
    local_dir: Path,
    *,
    pointer_path: Path = POINTER_PATH,
    use_cache: bool = True,
) -> Path | None:
    """The directory to compare ``suite``'s renders against, or ``None``.

    ``None`` means "no baselines are available for this device". Callers that
    must not proceed without them should use :func:`require_baseline_dir`,
    which raises the reason instead of discarding it; this function stays for
    callers that want to inspect availability without failing.
    """
    return _resolve(suite, key, local_dir, pointer_path, use_cache)[0]


def require_baseline_dir(
    suite: str,
    key: str,
    local_dir: Path,
    *,
    pointer_path: Path = POINTER_PATH,
    use_cache: bool = True,
) -> Path:
    """The directory to compare against; raise if there is not one.

    This is what the render suites call. A comparison that cannot run is a
    failed test, not a skipped one: a skipped render suite compared nothing,
    and reads as green while it does it.
    """
    resolved, unavailable = _resolve(suite, key, local_dir, pointer_path, use_cache)
    if resolved is None:
        raise BaselinesUnavailableError(unavailable)
    return resolved


def _resolve(
    suite: str, key: str, local_dir: Path, pointer_path: Path, use_cache: bool
) -> tuple[Path | None, BaselineUnavailable | None]:
    memo_key = (suite, key, str(local_dir))
    if use_cache and memo_key in _resolved:
        return _resolved[memo_key]

    resolved = _resolve_uncached(suite, key, local_dir, pointer_path)
    if use_cache:
        _resolved[memo_key] = resolved
    return resolved


def _resolve_uncached(
    suite: str, key: str, local_dir: Path, pointer_path: Path
) -> tuple[Path | None, BaselineUnavailable | None]:
    """``(directory, None)``, or ``(None, why not)``.

    The reason is what the caller shows when it fails, so each one names the
    step that did not produce a directory and what would fix it.
    """
    override = (os.getenv("ALGAN_BASELINE_DIR") or "").strip()
    if override:
        # Final by design: a machine that says where its baselines live must
        # not silently end up comparing against a downloaded set instead.
        candidate = Path(override) / suite / key
        if _has_files(candidate):
            return candidate, None
        reason = (
            f"ALGAN_BASELINE_DIR is set, but {candidate} holds no baselines "
            f"for {suite}/{key}. Its answer is final -- unset it to fall back "
            f"to the published archive."
        )
        warnings.warn(reason, stacklevel=3)
        # Unbaselined: the machine said where its baselines live and has none
        # for this device, which is the same state as an unpublished key.
        return None, BaselineUnavailable(reason, unbaselined=True)

    if _has_files(local_dir):
        return local_dir, None

    pointer = load_pointer(pointer_path)
    tag = pointer.get("tag")
    entry = pointer["archives"].get(archive_key(suite, key))
    if not tag or entry is None:
        # Nothing published for this device (or nothing published at all).
        # Not warned: the caller raises this reason, and a warning beside the
        # failure that quotes it says the same thing twice.
        return None, BaselineUnavailable(
            f"No baselines are published for {suite}/{key}"
            + (f" under tag {tag}" if tag else " (the pointer names no tag)")
            + f", and {local_dir} does not exist. Either this device has "
            f"never been baselined -- render one with the suite's "
            f"ALGAN_UPDATE_* variable, review it, and publish it with "
            f"scripts/package_baselines.py -- or the pointer names a device "
            f"key nothing produces."
            + (
                f" On a Mac, {MACOS_OPT_OUT_ENV}=1 skips this comparison "
                f"instead of failing it."
                # Not when it is already set: the caller is about to quote
                # this as its skip reason, where advice to set it reads as a
                # suggestion that it did not work.
                if key.partition("_mpsfriendly")[0] in MACOS_OPT_OUT_KEYS
                and not macos_opt_out_permits(key)
                else ""
            ),
            unbaselined=True,
        )

    target = _cache_root() / str(tag) / suite / key
    digest = entry["sha256"]
    if _cached(target, digest):
        return target, None

    if os.getenv("ALGAN_NO_BASELINE_DOWNLOAD") == "1":
        reason = (
            f"ALGAN_NO_BASELINE_DOWNLOAD is set and {suite}/{key} is not in "
            f"the cache at {target}. Unset it, or point ALGAN_BASELINE_DIR at "
            f"a local copy."
        )
        warnings.warn(reason, stacklevel=3)
        # Not "unbaselined": baselines exist for this device, the machine just
        # refused to fetch them. The macOS opt-out must not cover that.
        return None, BaselineUnavailable(reason)

    base = str(pointer.get("base_url") or "").rstrip("/")
    url = f"{base}/{tag}/{entry['file']}"
    failure = _download(url, target, digest)
    if failure is not None:
        return None, BaselineUnavailable(failure)
    return target, None


def _reset_for_tests() -> None:
    """Forget the memoized answers (the tests exercise several outcomes)."""
    _resolved.clear()
