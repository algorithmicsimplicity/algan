"""The baseline resolver: precedence, verification, and how it fails.

The whole point of this module is that a machine without the hosted baselines
*skips*, and a machine with a wrong copy *never compares against it*. Both of
those are silent-failure shapes -- a suite that skips everything looks green,
and a suite that compares against the wrong bytes fails for the wrong reason --
so each path is pinned here rather than left to the render suites, which only
run on a machine with baselines for its device.

The download path is exercised for real over ``file://``: urllib treats it
like any other URL, so the fetch, the sha256 check and the extraction all run
as they would against a release asset, with no network and no mocking of the
code under test.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
import tarfile
import warnings
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]

# Imported by path rather than by name: these tests must not depend on pytest
# having inserted tests/ into sys.path, which is a property of how the run was
# invoked.
_spec = importlib.util.spec_from_file_location(
    "algan_test_baseline_store", TESTS_ROOT / "baseline_store.py"
)
baseline_store = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = baseline_store
_spec.loader.exec_module(baseline_store)


@pytest.fixture(autouse=True)
def _forget_memo():
    baseline_store._reset_for_tests()
    yield
    baseline_store._reset_for_tests()


@pytest.fixture(autouse=True)
def _no_ambient_override(monkeypatch):
    """A developer's own ALGAN_BASELINE_DIR must not steer these tests."""
    monkeypatch.delenv("ALGAN_BASELINE_DIR", raising=False)
    monkeypatch.delenv("ALGAN_NO_BASELINE_DOWNLOAD", raising=False)


def _make_archive(directory: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(destination, "wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(directory).as_posix())
    return hashlib.sha256(destination.read_bytes()).hexdigest()


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


def _published(tmp_path: Path, *, contents=b"not really an mp4", tag="baselines-test"):
    """A staged 'release': a real tarball at base_url/tag/file, plus a pointer."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "scene.mp4").write_bytes(contents)
    asset = tmp_path / "release" / tag / "full_renders-cuda.tar.gz"
    digest = _make_archive(source, asset)
    pointer = _pointer(
        tmp_path,
        {"full_renders/cuda": {"file": asset.name, "sha256": digest}},
        tag=tag,
    )
    return pointer, asset, digest


def test_the_committed_pointer_file_is_well_formed():
    """The file that ships in the repository has to parse and validate."""
    pointer = baseline_store.load_pointer()
    assert isinstance(pointer.get("base_url"), str)
    # Every archive names a suite that exists and a device key, and the null
    # tag is the published/unpublished switch.
    for name in pointer["archives"]:
        suite, _, key = name.partition("/")
        assert key, f"{name} does not name a device"
        assert (TESTS_ROOT / suite).is_dir(), f"{name} names no suite directory"
    assert "fast" not in {n.partition("/")[0] for n in pointer["archives"]}, (
        "tests/fast stays in git: it is the only render baseline CI compares "
        "against, and hosting it would put a download in every CI run"
    )


def test_the_committed_pointer_describes_the_committed_baselines():
    """A rebaseline that is committed but not re-packaged fails here.

    Without this the pointer silently goes stale, and the day the mp4s leave
    the tree every suite starts comparing against last release's pixels.
    """
    sys.path.insert(0, str(TESTS_ROOT.parent / "scripts"))
    try:
        import package_baselines
    finally:
        sys.path.pop(0)

    assert package_baselines.main(["--verify"]) == 0


def test_the_local_directory_wins(tmp_path):
    """While the mp4s are committed, the fetcher must change nothing."""
    pointer, _, _ = _published(tmp_path)
    local = tmp_path / "expected_outputs_cuda"
    local.mkdir()
    (local / "scene.mp4").write_bytes(b"the committed one")

    resolved = baseline_store.resolve_baseline_dir(
        "full_renders", "cuda", local, pointer_path=pointer
    )
    assert resolved == local


def test_an_empty_local_directory_does_not_count(tmp_path, monkeypatch):
    """A leftover empty directory must not shadow the hosted baselines.

    Removing the mp4s with ``git rm`` leaves nothing behind, but an interrupted
    rebaseline or a stale checkout can, and 'compared nothing, reported green'
    is exactly the failure this suite exists to prevent.
    """
    pointer, _, _ = _published(tmp_path)
    monkeypatch.setattr(baseline_store, "_cache_root", lambda: tmp_path / "cache")
    local = tmp_path / "expected_outputs_cuda"
    local.mkdir()

    resolved = baseline_store.resolve_baseline_dir(
        "full_renders", "cuda", local, pointer_path=pointer
    )
    assert resolved is not None
    assert (resolved / "scene.mp4").read_bytes() == b"not really an mp4"


def test_a_published_archive_is_downloaded_verified_and_cached(tmp_path, monkeypatch):
    pointer, asset, digest = _published(tmp_path)
    cache = tmp_path / "cache"
    monkeypatch.setattr(baseline_store, "_cache_root", lambda: cache)
    local = tmp_path / "absent"

    resolved = baseline_store.resolve_baseline_dir(
        "full_renders", "cuda", local, pointer_path=pointer
    )
    assert resolved == cache / "baselines-test" / "full_renders" / "cuda"
    assert (resolved / "scene.mp4").read_bytes() == b"not really an mp4"
    assert (resolved / ".sha256").read_text() == digest

    # Second call is served from the cache: deleting the asset must not matter.
    baseline_store._reset_for_tests()
    asset.unlink()
    assert (
        baseline_store.resolve_baseline_dir(
            "full_renders", "cuda", local, pointer_path=pointer
        )
        == resolved
    )


def test_a_cache_whose_marker_does_not_match_is_refetched(tmp_path, monkeypatch):
    """The pin is what makes the cache trustworthy, not the directory's name."""
    pointer, _, digest = _published(tmp_path)
    cache = tmp_path / "cache"
    monkeypatch.setattr(baseline_store, "_cache_root", lambda: cache)
    stale = cache / "baselines-test" / "full_renders" / "cuda"
    stale.mkdir(parents=True)
    (stale / "scene.mp4").write_bytes(b"last release's pixels")
    (stale / ".sha256").write_text("0" * 64)

    resolved = baseline_store.resolve_baseline_dir(
        "full_renders", "cuda", tmp_path / "absent", pointer_path=pointer
    )
    assert resolved is not None
    assert (resolved / "scene.mp4").read_bytes() == b"not really an mp4"
    assert (resolved / ".sha256").read_text() == digest


def test_a_digest_mismatch_is_refused_rather_than_compared_against(
    tmp_path, monkeypatch
):
    pointer, asset, _ = _published(tmp_path)
    monkeypatch.setattr(baseline_store, "_cache_root", lambda: tmp_path / "cache")
    asset.write_bytes(b"something else entirely")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = baseline_store.resolve_baseline_dir(
            "full_renders", "cuda", tmp_path / "absent", pointer_path=pointer
        )
    assert resolved is None
    assert any("sha256" in str(w.message) for w in caught)


def test_a_missing_asset_warns_once_and_skips(tmp_path, monkeypatch):
    pointer, asset, _ = _published(tmp_path)
    monkeypatch.setattr(baseline_store, "_cache_root", lambda: tmp_path / "cache")
    asset.unlink()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            resolved = baseline_store.resolve_baseline_dir(
                "full_renders", "cuda", tmp_path / "absent", pointer_path=pointer
            )
    assert resolved is None
    assert len([w for w in caught if "Could not download" in str(w.message)]) == 1


def test_an_unpublished_tag_is_silent(tmp_path):
    """The state this lands in: a pointer with no release behind it yet."""
    pointer = _pointer(tmp_path, {}, tag=None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = baseline_store.resolve_baseline_dir(
            "full_renders", "cuda", tmp_path / "absent", pointer_path=pointer
        )
    assert resolved is None
    assert not caught


def test_no_baseline_download_opts_out(tmp_path, monkeypatch):
    pointer, _, _ = _published(tmp_path)
    monkeypatch.setattr(baseline_store, "_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setenv("ALGAN_NO_BASELINE_DOWNLOAD", "1")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = baseline_store.resolve_baseline_dir(
            "full_renders", "cuda", tmp_path / "absent", pointer_path=pointer
        )
    assert resolved is None
    assert any("ALGAN_NO_BASELINE_DOWNLOAD" in str(w.message) for w in caught)


def test_baseline_dir_override_is_final(tmp_path, monkeypatch):
    """An explicit local root must not fall through to a download."""
    pointer, _, _ = _published(tmp_path)
    monkeypatch.setattr(baseline_store, "_cache_root", lambda: tmp_path / "cache")
    root = tmp_path / "mine"
    (root / "full_renders" / "cuda").mkdir(parents=True)
    (root / "full_renders" / "cuda" / "scene.mp4").write_bytes(b"mine")
    monkeypatch.setenv("ALGAN_BASELINE_DIR", str(root))

    assert baseline_store.resolve_baseline_dir(
        "full_renders", "cuda", tmp_path / "absent", pointer_path=pointer
    ) == (root / "full_renders" / "cuda")

    baseline_store._reset_for_tests()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert (
            baseline_store.resolve_baseline_dir(
                "path_traced", "cuda", tmp_path / "absent", pointer_path=pointer
            )
            is None
        )
    assert any("ALGAN_BASELINE_DIR" in str(w.message) for w in caught)


def test_an_archive_cannot_write_outside_its_destination(tmp_path):
    """Defence in depth: extractall on 3.9 would happily follow ``..``."""
    escaping = tmp_path / "escaping.tar.gz"
    with tarfile.open(escaping, "w:gz") as archive:
        payload = tmp_path / "payload"
        payload.write_bytes(b"x")
        archive.add(payload, arcname="../escaped.mp4")

    with pytest.raises(baseline_store.BaselinePointerError, match="escapes"):
        baseline_store.extract_archive(escaping, tmp_path / "destination")
    assert not (tmp_path / "escaped.mp4").exists()


@pytest.mark.parametrize(
    "pointer_text",
    ["not json at all", "[]", "{}", '{"archives": {"a/b": {"file": "f.tar.gz"}}}'],
)
def test_a_malformed_pointer_raises_rather_than_degrading(tmp_path, pointer_text):
    path = tmp_path / "baselines.json"
    path.write_text(pointer_text, encoding="utf-8")
    with pytest.raises(baseline_store.BaselinePointerError):
        baseline_store.load_pointer(path)


def test_a_missing_pointer_file_raises(tmp_path):
    with pytest.raises(baseline_store.BaselinePointerError):
        baseline_store.load_pointer(tmp_path / "nope.json")
