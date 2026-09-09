"""The baseline resolver: precedence, verification, and how it fails.

The whole point of this module is that a machine without the hosted baselines
*fails*, and a machine with a wrong copy *never compares against it*. Both are
silent-failure shapes -- a suite that resolves nothing compared nothing, and a
suite that compares against the wrong bytes fails for the wrong reason -- so
each path is pinned here rather than left to the render suites, which only run
on a machine with baselines for its device.

The failure used to be a skip, which is how ``tests/full_renders`` came to
skip all six scenes for the whole life of its ``cpu_eager`` key while reading
as green. :func:`~baseline_store.require_baseline_dir` raises instead, and the
reason travels with it.

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
    """A developer's own ALGAN_BASELINE_DIR must not steer these tests.

    The macOS opt-out is cleared for the same reason, and it is not
    hypothetical: CI sets ``ALGAN_ALLOW_UNBASELINED_MACOS=1`` for the whole
    macOS job, which changes what the resolver's unbaselined message says.
    Every test that cares about the opt-out sets it itself.
    """
    monkeypatch.delenv("ALGAN_BASELINE_DIR", raising=False)
    monkeypatch.delenv("ALGAN_NO_BASELINE_DOWNLOAD", raising=False)
    monkeypatch.delenv(baseline_store.MACOS_OPT_OUT_ENV, raising=False)


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
    # Every archive names a suite that exists and a device key. The committed
    # pointer is published; null tags remain covered below as a bootstrap state.
    assert isinstance(pointer.get("tag"), str)
    assert pointer["tag"]
    for name in pointer["archives"]:
        suite, _, key = name.partition("/")
        assert key, f"{name} does not name a device"
        assert (TESTS_ROOT / suite).is_dir(), f"{name} names no suite directory"
    assert "fast" not in {n.partition("/")[0] for n in pointer["archives"]}, (
        "tests/fast stays in git: it is the only render baseline CI compares "
        "against, and hosting it would put a download in every CI run"
    )


def test_the_committed_pointer_is_published_and_local_rebaselines_match():
    """A local rebaseline that was not re-packaged fails here.

    A clean checkout has no heavy baselines, so --verify instead checks that
    the committed pointer names a published release.
    """
    sys.path.insert(0, str(TESTS_ROOT.parent / "scripts"))
    try:
        import package_baselines
    finally:
        sys.path.pop(0)

    assert package_baselines.main(["--verify"]) == 0


def test_the_local_directory_wins(tmp_path):
    """A freshly rendered local baseline must win over the release asset."""
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


def test_a_resolvable_baseline_is_required_without_complaint(tmp_path):
    local = tmp_path / "expected_outputs_cuda"
    local.mkdir()
    (local / "scene.mp4").write_bytes(b"not really an mp4")
    pointer = _pointer(tmp_path, {})

    assert (
        baseline_store.require_baseline_dir(
            "full_renders", "cuda", local, pointer_path=pointer
        )
        == local
    )


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        (None, "the pointer names no tag"),
        ("baselines-test", "baselines-test"),
    ],
)
def test_an_unresolvable_baseline_raises_rather_than_skipping(tmp_path, tag, expected):
    """The whole point of the change: a comparison that cannot run is a failure.

    ``tests/full_renders`` skipped all six of its scenes for the whole life of
    the ``cpu_eager`` key -- a suite comparing nothing, reading as a clean run
    -- because the resolver's ``None`` became a skip. The reason has to reach
    the failure, too: "no baselines" alone sends the reader hunting a
    rendering bug that is not there.
    """
    pointer = _pointer(tmp_path, {}, tag=tag)

    with pytest.raises(baseline_store.BaselinesUnavailableError) as excinfo:
        baseline_store.require_baseline_dir(
            "full_renders", "cuda", tmp_path / "absent", pointer_path=pointer
        )
    message = str(excinfo.value)
    assert "full_renders/cuda" in message
    assert expected in message


def test_a_download_failure_reaches_the_raised_reason(tmp_path, monkeypatch):
    """A failed fetch must not be mistaken for a rendering regression."""
    pointer, asset, _ = _published(tmp_path)
    monkeypatch.setattr(baseline_store, "_cache_root", lambda: tmp_path / "cache")
    asset.unlink()

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with pytest.raises(baseline_store.BaselinesUnavailableError) as excinfo:
            baseline_store.require_baseline_dir(
                "full_renders", "cuda", tmp_path / "absent", pointer_path=pointer
            )
    assert "Could not download" in str(excinfo.value)
    # Baselines exist for this device; the fetch failed. Not the state the
    # macOS opt-out is allowed to excuse.
    assert not excinfo.value.unbaselined


@pytest.mark.parametrize(
    "key",
    ["macos_cpu", "macos_mps", "macos_cpu_mpsfriendly", "macos_mps_mpsfriendly"],
)
def test_the_macos_opt_out_covers_both_mac_devices_and_their_modes(key, monkeypatch):
    monkeypatch.setenv(baseline_store.MACOS_OPT_OUT_ENV, "1")
    assert baseline_store.macos_opt_out_permits(key)


@pytest.mark.parametrize("key", ["cpu", "cpu_eager", "cuda", "mps"])
def test_the_macos_opt_out_covers_nothing_else(key, monkeypatch):
    """It is an opt-out for one platform, not for missing baselines at large.

    A blanket switch would let the CPU and CUDA suites go quiet again, which
    is the failure that made these comparisons fail rather than skip.
    """
    monkeypatch.setenv(baseline_store.MACOS_OPT_OUT_ENV, "1")
    assert not baseline_store.macos_opt_out_permits(key)


@pytest.mark.parametrize("value", [None, "", "0", "true", "yes"])
def test_the_macos_opt_out_is_off_unless_it_is_exactly_one(value, monkeypatch):
    if value is None:
        monkeypatch.delenv(baseline_store.MACOS_OPT_OUT_ENV, raising=False)
    else:
        monkeypatch.setenv(baseline_store.MACOS_OPT_OUT_ENV, value)
    assert not baseline_store.macos_opt_out_permits("macos_cpu")


def test_an_unbaselined_macos_key_names_the_opt_out(tmp_path):
    """The failure has to say the knob exists, or nobody finds it."""
    pointer = _pointer(tmp_path, {})

    with pytest.raises(baseline_store.BaselinesUnavailableError) as excinfo:
        baseline_store.require_baseline_dir(
            "full_renders", "macos_mps", tmp_path / "absent", pointer_path=pointer
        )
    assert excinfo.value.unbaselined
    assert baseline_store.MACOS_OPT_OUT_ENV in str(excinfo.value)


def test_the_opt_out_advice_is_dropped_once_the_opt_out_is_set(tmp_path, monkeypatch):
    """The caller quotes this reason as its *skip* reason when the knob is on.

    Advice to set a variable that is already set reads as a report that it did
    not work, so the sentence is left out. Pinned because the macOS CI job runs
    the whole unit suite with the knob on, and the test above would silently
    invert there without this one holding the other side.
    """
    monkeypatch.setenv(baseline_store.MACOS_OPT_OUT_ENV, "1")
    pointer = _pointer(tmp_path, {})

    with pytest.raises(baseline_store.BaselinesUnavailableError) as excinfo:
        baseline_store.require_baseline_dir(
            "full_renders", "macos_mps", tmp_path / "absent", pointer_path=pointer
        )
    assert excinfo.value.unbaselined
    assert baseline_store.MACOS_OPT_OUT_ENV not in str(excinfo.value)


def test_a_missing_asset_warns_only_once(tmp_path, monkeypatch):
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
    """Defence in depth: extractall on 3.10 would happily follow ``..``."""
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
