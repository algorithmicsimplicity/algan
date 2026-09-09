"""The release gate that checks the baseline pointer against what is hosted.

The gate this covers exists because the render suites fail *quietly*: an
archive they cannot resolve makes them skip, and a skip reads as a clean run.
So the gate's own failure paths are worth pinning -- a gate that stops
detecting is the same silence one level up.

Two shapes, and the second is the reason the check is not a HEAD request.
Under ``baselines-2026-09-09.1`` one asset was absent, and another *existed*
under the expected name holding the superseded archive from the previous tag:
200 OK, wrong bytes, silent skip. Both are pinned below.

Staged over ``file://`` like ``test_baseline_store.py``: urllib treats it as
any other URL, so the fetch and the digest run for real with no network and no
mocking of the code under test. The one thing that cannot be exercised offline
is the GitHub asset listing in the failure report, which is best-effort by
construction -- it returns ``None`` rather than raising when it cannot read.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# By path, not by name: this must not depend on how the run was invoked having
# put scripts/ on sys.path. Matches test_baseline_store.py's reasoning.
_spec = importlib.util.spec_from_file_location(
    "algan_gate_verify_baseline_pointer",
    REPO_ROOT / "scripts" / "gate" / "verify_baseline_pointer.py",
)
verify_baseline_pointer = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = verify_baseline_pointer
_spec.loader.exec_module(verify_baseline_pointer)


def _archive(payload: bytes) -> bytes:
    """A deterministic one-file tarball, so digests are stable across runs."""
    raw = io.BytesIO()
    with (
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        info = tarfile.TarInfo("scene.mp4")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return raw.getvalue()


def _publish(tmp_path: Path, files: dict[str, bytes], *, tag="baselines-test") -> Path:
    """Write ``files`` as release assets under ``base_url/tag/``."""
    directory = tmp_path / "release" / tag
    directory.mkdir(parents=True, exist_ok=True)
    for name, blob in files.items():
        (directory / name).write_bytes(blob)
    return directory


def _pointer(tmp_path: Path, archives: dict, *, tag="baselines-test") -> Path:
    path = tmp_path / "baselines.json"
    path.write_text(
        json.dumps(
            {
                "base_url": (tmp_path / "release").as_uri(),
                "tag": tag,
                "archives": archives,
            }
        ),
        encoding="utf-8",
    )
    return path


def _entry(blob: bytes, name: str) -> dict:
    return {
        "file": name,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size": len(blob),
    }


def test_a_matching_pointer_passes(tmp_path, capsys):
    blob = _archive(b"a render")
    _publish(tmp_path, {"full_renders-cuda.tar.gz": blob})
    pointer = _pointer(
        tmp_path, {"full_renders/cuda": _entry(blob, "full_renders-cuda.tar.gz")}
    )

    assert verify_baseline_pointer.main(["--pointer", str(pointer)]) == 0
    assert "all 1 archives verified" in capsys.readouterr().out


def test_an_absent_archive_fails(tmp_path, capsys):
    """The shape a HEAD request already caught."""
    blob = _archive(b"a render")
    _publish(tmp_path, {})
    pointer = _pointer(
        tmp_path, {"full_renders/cuda": _entry(blob, "full_renders-cuda.tar.gz")}
    )

    assert verify_baseline_pointer.main(["--pointer", str(pointer)]) == 1
    assert "full_renders/cuda" in capsys.readouterr().out


def test_an_asset_holding_the_wrong_file_fails(tmp_path, capsys):
    """The shape that HEAD missed, and the whole reason this gate downloads.

    The asset is present and served, so a HEAD check passes; it is simply a
    different archive than the pointer names. Left undetected this reaches
    ``baseline_store``, which rejects it on the sha256 and skips the suite.
    """
    published = _archive(b"the superseded render")
    expected = _archive(b"the rebaselined render")
    assert published != expected
    _publish(tmp_path, {"path_traced-cpu.tar.gz": published})
    pointer = _pointer(
        tmp_path, {"path_traced/cpu": _entry(expected, "path_traced-cpu.tar.gz")}
    )

    assert verify_baseline_pointer.main(["--pointer", str(pointer)]) == 1
    out = capsys.readouterr().out
    assert "sha256 mismatch" in out
    assert "different file" in out


def test_a_size_mismatch_fails(tmp_path, capsys):
    """A pointer whose size field drifted from its digest is still wrong."""
    blob = _archive(b"a render")
    _publish(tmp_path, {"full_renders-cuda.tar.gz": blob})
    entry = _entry(blob, "full_renders-cuda.tar.gz") | {"size": len(blob) + 1}
    pointer = _pointer(tmp_path, {"full_renders/cuda": entry})

    assert verify_baseline_pointer.main(["--pointer", str(pointer)]) == 1


def test_a_pointer_with_no_size_field_is_still_verified(tmp_path, capsys):
    """``size`` is advisory; ``sha256`` is the contract and is not optional."""
    blob = _archive(b"a render")
    _publish(tmp_path, {"full_renders-cuda.tar.gz": blob})
    entry = _entry(blob, "full_renders-cuda.tar.gz")
    del entry["size"]
    pointer = _pointer(tmp_path, {"full_renders/cuda": entry})

    assert verify_baseline_pointer.main(["--pointer", str(pointer)]) == 0


def test_a_null_tag_is_the_bootstrap_state_and_passes(tmp_path, capsys):
    """``baseline_store`` treats a null tag as an unbaselined offline machine.

    Nothing is published, so there is nothing to check and nothing is wrong.
    Failing here would make the documented bootstrap state unreleasable.
    """
    pointer = _pointer(tmp_path, {}, tag=None)

    assert verify_baseline_pointer.main(["--pointer", str(pointer)]) == 0
    assert "nothing to check" in capsys.readouterr().out


def test_a_malformed_pointer_fails_rather_than_raising(tmp_path, capsys):
    """A bad pointer is a gate failure with a message, not a traceback."""
    path = tmp_path / "baselines.json"
    path.write_text('{"archives": "not an object"}', encoding="utf-8")

    assert verify_baseline_pointer.main(["--pointer", str(path)]) == 1
    assert "::error::" in capsys.readouterr().out


def test_a_missing_pointer_fails_rather_than_raising(tmp_path, capsys):
    assert (
        verify_baseline_pointer.main(["--pointer", str(tmp_path / "absent.json")]) == 1
    )
    assert "::error::" in capsys.readouterr().out


def test_the_asset_listing_never_raises(tmp_path):
    """Best-effort by construction: it only makes a failure legible.

    An unauthenticated rate-limit or an offline runner must not turn a real
    mismatch into a crash, nor a clean run into a red one.
    """
    assert (
        verify_baseline_pointer.asset_names("nonexistent/repository", "no-such-tag")
        is None
    )


# The committed pointer's own shape -- that it parses, and that every key names
# a suite the repository has -- is already pinned by
# test_baseline_store.py::test_the_committed_pointer_file_is_well_formed. Not
# repeated here: two tests asserting one fact means one of them stops being
# maintained. Nothing here is marked `fast`; per CLAUDE.md that mark is for
# tests a change *elsewhere* is liable to break, and these fail only when the
# gate itself changes.
