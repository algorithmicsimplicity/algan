"""Diagnostics for directories used by native writers with opaque errors."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from algan.errors import AlganConfigurationError


def _ensure_writable_directory(path, *, purpose: str, remedy: str) -> Path:
    """Create a directory and actually test writing, including ACL/sandbox rules."""
    directory = Path(path).expanduser().resolve()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # tempfile retries PermissionError up to TMP_MAX times on Windows
        # when os.access incorrectly claims this directory is writable. Use
        # one exclusive creation instead, so a denied ACL fails promptly.
        probe = directory / f".algan-write-check-{uuid4().hex}"
        created = False
        try:
            with probe.open("xb") as stream:
                created = True
                stream.write(b"algan")
                stream.flush()
        finally:
            if created:
                probe.unlink()
    except OSError as exc:
        raise AlganConfigurationError(
            f"Cannot write {purpose} directory '{directory}': {exc}. "
            f"{remedy} Check directory permissions, sandbox access, and free disk space."
        ) from exc
    return directory
