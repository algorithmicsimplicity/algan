"""Resolving the OIDN RT weights, degrading to denoise-off when they can't be.

Resolution order:

1. ``denoise_weights`` (``ALGAN_DENOISE_WEIGHTS``): an explicit ``.tza`` path
   for users who ship their own copy; no hash check, their file is their
   choice.
2. The cache: ``SETTINGS.paths.cache_directory / "oidn" / FILE_NAME``,
   accepted only when its sha256 matches the pinned official hash (a partial
   download is re-fetched, never trusted).
3. A one-time download of the official file from the oidn-weights repository
   (the Git-LFS media URL; the plain raw URL serves only the LFS pointer).
   Written atomically (temp file + rename) so a killed download never
   poisons the cache.

Every failure returns ``None`` after one WARNING -- denoising turns itself
off, the render still finishes. A bad override or a hash mismatch is
remembered for the process: neither fixes itself. A failed download is not:
a network blip on a warm daemon's first render must not leave every later
render noisy, so a later render job tries again once
``_DOWNLOAD_RETRY_SECONDS`` have passed. An offline machine therefore pays
one connection timeout per path-traced render at most every ten minutes.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
import urllib.request
from pathlib import Path

from algan.logging.logger import get_logger
from algan.rendering.raytracing import settings as rt_settings

logger = get_logger("raytracing")

FILE_NAME = "rt_hdr_alb_nrm.tza"
#: The Git-LFS media endpoint -- ``raw.githubusercontent.com`` returns the
#: 132-byte LFS pointer for this repository, not the archive. Pinned to an
#: oidn-weights commit rather than ``master``: an upstream re-upload would
#: otherwise fail the hash check below on every fresh install.
WEIGHTS_COMMIT = "28883d1769d5930e13cf7f1676dd852bd81ed9e7"
WEIGHTS_URL = (
    "https://media.githubusercontent.com/media/RenderKit/oidn-weights/"
    f"{WEIGHTS_COMMIT}/{FILE_NAME}"
)
#: sha256 of the official archive (also the oid in its LFS pointer).
WEIGHTS_SHA256 = "e586ef2ff48d7fbb7611986405220ed8fc5c13ca79bfc40be4dc742fbf959e1a"
_DOWNLOAD_TIMEOUT_SECONDS = 30
#: How long a failed download is remembered before a render tries again.
_DOWNLOAD_RETRY_SECONDS = 600.0

#: Process-level memo: None = not tried, "" = tried and failed for good
#: (stay off), otherwise the resolved path.
_resolved: str | None = None
#: ``time.monotonic()`` of the last failed download, or None.
_download_failed_at: float | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_path() -> Path:
    from algan.settings import SETTINGS

    return Path(SETTINGS.paths.cache_directory) / "oidn" / FILE_NAME


def _download(target: Path) -> bool | None:
    """Fetch the weights into ``target``: True on success, False when the
    file itself is wrong (stay off), None when the fetch failed (try later).
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(
            WEIGHTS_URL, timeout=_DOWNLOAD_TIMEOUT_SECONDS
        ) as response:
            data = response.read()
    except Exception as exc:  # URLError, socket timeout, HTTP errors, ...
        logger.warning(
            f"Could not download the denoiser weights ({exc}); rendering "
            f"without denoising. Retry with network access, or point "
            f"ALGAN_DENOISE_WEIGHTS at a local {FILE_NAME}."
        )
        return None
    if hashlib.sha256(data).hexdigest() != WEIGHTS_SHA256:
        logger.warning(
            "The downloaded denoiser weights do not match the pinned sha256; "
            "rendering without denoising."
        )
        return False
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_name, target)
    except OSError:
        with open(target, "wb") as f:  # rename failed; plain write
            f.write(data)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return True


def weights_path() -> str | None:
    """The ``.tza`` file to load, or ``None`` (already warned) if none can
    be had. The answer is remembered for the process, except a failed
    download, which is retried after ``_DOWNLOAD_RETRY_SECONDS``.
    """
    global _resolved, _download_failed_at
    if _resolved is not None:
        return _resolved or None
    if (
        _download_failed_at is not None
        and time.monotonic() - _download_failed_at < _DOWNLOAD_RETRY_SECONDS
    ):
        return None

    override = str(rt_settings.denoise_weights or "").strip()
    if override:
        if Path(override).is_file():
            _resolved = override
            return _resolved
        logger.warning(
            f"denoise_weights points at {override!r}, which does not exist; "
            f"rendering without denoising."
        )
        _resolved = ""
        return None

    cache = _cache_path()
    try:
        if cache.is_file() and _sha256(cache) == WEIGHTS_SHA256:
            _resolved = str(cache)
            return _resolved
    except OSError:
        pass
    fetched = _download(cache)
    if fetched:
        _resolved = str(cache)
        _download_failed_at = None
        return _resolved
    if fetched is None:
        _download_failed_at = time.monotonic()
    else:
        _resolved = ""
    return None


def _reset_for_tests() -> None:
    """Forget the memoized answer (tests exercise the failure paths)."""
    global _resolved, _download_failed_at
    _resolved = None
    _download_failed_at = None
