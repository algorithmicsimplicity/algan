"""Offline tests for the downstream Quadrants distribution rewrite."""

from __future__ import annotations

import csv
import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_helper():
    path = REPO_ROOT / "scripts" / "rebrand_quadrants_wheel.py"
    spec = importlib.util.spec_from_file_location("rebrand_quadrants_wheel", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def helper():
    return _load_helper()


def _fake_wheel(path: Path, version: str) -> None:
    dist_info = f"quadrants-{version}.dist-info"
    metadata = (
        "Metadata-Version: 2.4\n"
        "Name: quadrants\n"
        f"Version: {version}\n"
        "Summary: test wheel\n"
    ).encode()
    wheel = (
        b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: false\n"
        b"Tag: cp311-cp311-manylinux_2_27_x86_64\n"
    )
    members = {
        "quadrants/__init__.py": f"__version__ = {version!r}\n".encode(),
        f"{dist_info}/METADATA": metadata,
        f"{dist_info}/WHEEL": wheel,
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        rows = [(name, "", str(len(data))) for name, data in members.items()]
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(rows)
        writer.writerow((f"{dist_info}/RECORD", "", ""))
        archive.writestr(f"{dist_info}/RECORD", stream.getvalue())


def test_rebrand_changes_distribution_but_not_import_package(helper, tmp_path):
    upstream = tmp_path / "quadrants-1.3.0.post1-cp311-cp311-manylinux_2_27_x86_64.whl"
    _fake_wheel(upstream, helper.DOWNSTREAM_VERSION)
    downstream = helper.rebrand_wheel(upstream)
    assert downstream.name.startswith("algan_quadrants-1.3.0.post1-")

    with zipfile.ZipFile(downstream) as wheel:
        names = set(wheel.namelist())
        assert "quadrants/__init__.py" in names
        assert not any(
            name.startswith("quadrants-1.3.0.post1.dist-info/") for name in names
        )
        prefix = "algan_quadrants-1.3.0.post1.dist-info/"
        metadata = wheel.read(prefix + "METADATA").decode()
        assert "Name: algan-quadrants\n" in metadata
        assert "Version: 1.3.0.post1\n" in metadata
        record = wheel.read(prefix + "RECORD").decode()
        assert prefix + "METADATA" in record
        assert prefix + "RECORD,," in record


def test_rebrand_refuses_a_version_not_used_at_native_build_time(helper, tmp_path):
    upstream = tmp_path / "quadrants-1.3.0-cp311-cp311-manylinux_2_27_x86_64.whl"
    _fake_wheel(upstream, "1.3.0")
    with pytest.raises(ValueError, match="expected version"):
        helper.rebrand_wheel(upstream)


def test_validate_refuses_a_wheel_without_quadrants_import(helper, tmp_path):
    path = (
        tmp_path
        / "algan_quadrants-1.3.0.post1-cp311-cp311-manylinux_2_27_x86_64.whl"
    )
    prefix = "algan_quadrants-1.3.0.post1.dist-info/"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(
            prefix + "METADATA",
            "Name: algan-quadrants\nVersion: 1.3.0.post1\n",
        )
        wheel.writestr(prefix + "RECORD", "")
    with pytest.raises(ValueError, match="retain the import package"):
        helper.validate_downstream_wheel(path)
