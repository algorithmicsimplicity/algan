#!/usr/bin/env python3
"""Rebrand a patched Quadrants wheel for Algan's PyPI distribution.

The native build still happens as upstream ``quadrants``. For release builds
``SETUPTOOLS_SCM_PRETEND_VERSION`` pins the build-time version to
``1.3.0.post2`` so CMake and Python agree on the downstream version. This
script then changes only the *distribution* name from ``quadrants`` to
``algan-quadrants``. The import package stays ``quadrants``.

Changing the distribution name after compilation is safe because Quadrants'
native build consumes the version, not the PyPI distribution name. The wheel
is rewritten rather than merely renamed: ``METADATA``, the ``.dist-info``
directory, and ``RECORD`` are all updated.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import zipfile
from pathlib import Path

UPSTREAM_DISTRIBUTION = "quadrants"
DOWNSTREAM_DISTRIBUTION = "algan-quadrants"
DOWNSTREAM_WHEEL_DISTRIBUTION = "algan_quadrants"
DOWNSTREAM_VERSION = "1.3.0.post2"


def _record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def _renamed_info(info: zipfile.ZipInfo, name: str) -> zipfile.ZipInfo:
    clone = zipfile.ZipInfo(name, date_time=info.date_time)
    for attribute in (
        "compress_type",
        "comment",
        "extra",
        "create_system",
        "create_version",
        "extract_version",
        "reserved",
        "flag_bits",
        "volume",
        "internal_attr",
        "external_attr",
    ):
        setattr(clone, attribute, getattr(info, attribute))
    return clone


def _rewrite_metadata(data: bytes) -> bytes:
    text = data.decode("utf-8")
    lines = text.splitlines(keepends=True)
    name_hits = 0
    version_hits = 0
    rewritten: list[str] = []
    for line in lines:
        bare = line.rstrip("\r\n")
        ending = line[len(bare) :]
        if bare == f"Name: {UPSTREAM_DISTRIBUTION}":
            name_hits += 1
            line = f"Name: {DOWNSTREAM_DISTRIBUTION}{ending}"
        elif bare.startswith("Version: "):
            version_hits += 1
            actual = bare.removeprefix("Version: ")
            if actual != DOWNSTREAM_VERSION:
                raise ValueError(
                    f"wheel version is {actual!r}; expected {DOWNSTREAM_VERSION!r}. "
                    "The build must set SETUPTOOLS_SCM_PRETEND_VERSION before "
                    "compilation so the native and Python versions agree."
                )
        rewritten.append(line)
    if name_hits != 1:
        raise ValueError(
            f"expected exactly one 'Name: {UPSTREAM_DISTRIBUTION}' in METADATA, "
            f"found {name_hits}"
        )
    if version_hits != 1:
        raise ValueError(
            f"expected exactly one Version field in METADATA, found {version_hits}"
        )
    return "".join(rewritten).encode("utf-8")


def _wheel_parts(path: Path) -> tuple[str, str, list[str]]:
    if path.suffix != ".whl":
        raise ValueError(f"not a wheel: {path}")
    parts = path.stem.split("-")
    if len(parts) < 5:
        raise ValueError(f"malformed wheel filename: {path.name}")
    return parts[0], parts[1], parts[2:]


def rebrand_wheel(path: Path, *, delete_original: bool = False) -> Path:
    path = Path(path)
    distribution, version, remaining = _wheel_parts(path)
    if distribution != UPSTREAM_DISTRIBUTION:
        raise ValueError(
            f"{path.name}: expected upstream distribution {UPSTREAM_DISTRIBUTION!r}, "
            f"got {distribution!r}"
        )
    if version != DOWNSTREAM_VERSION:
        raise ValueError(
            f"{path.name}: expected version {DOWNSTREAM_VERSION!r}, got {version!r}"
        )

    old_prefix = f"{UPSTREAM_DISTRIBUTION}-{version}.dist-info/"
    new_prefix = f"{DOWNSTREAM_WHEEL_DISTRIBUTION}-{version}.dist-info/"
    output = path.with_name(
        "-".join([DOWNSTREAM_WHEEL_DISTRIBUTION, version, *remaining]) + ".whl"
    )
    temporary = output.with_suffix(output.suffix + ".tmp")

    rows: list[tuple[str, str, str]] = []
    record_info: zipfile.ZipInfo | None = None
    metadata_seen = False
    try:
        with (
            zipfile.ZipFile(path, "r") as source,
            zipfile.ZipFile(temporary, "w") as target,
        ):
            target.comment = source.comment
            for info in source.infolist():
                name = info.filename
                if name.endswith(("RECORD.jws", "RECORD.p7s")):
                    raise ValueError(
                        f"{path.name} is signed; rebranding would invalidate its "
                        "wheel signature"
                    )
                renamed = (
                    new_prefix + name[len(old_prefix) :]
                    if name.startswith(old_prefix)
                    else name
                )
                if name == old_prefix + "RECORD":
                    record_info = info
                    continue
                data = source.read(info)
                if name == old_prefix + "METADATA":
                    data = _rewrite_metadata(data)
                    metadata_seen = True
                target.writestr(_renamed_info(info, renamed), data)
                if not renamed.endswith("/"):
                    rows.append((renamed, _record_hash(data), str(len(data))))

            if record_info is None:
                raise ValueError(f"{path.name}: no {old_prefix}RECORD member")
            if not metadata_seen:
                raise ValueError(f"{path.name}: no {old_prefix}METADATA member")

            record_name = new_prefix + "RECORD"
            stream = io.StringIO(newline="")
            writer = csv.writer(stream, lineterminator="\n")
            for row in sorted(rows):
                writer.writerow(row)
            writer.writerow((record_name, "", ""))
            record = stream.getvalue().encode("utf-8")
            target.writestr(_renamed_info(record_info, record_name), record)
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    validate_downstream_wheel(output)
    if delete_original:
        path.unlink()
    return output


def validate_downstream_wheel(path: Path) -> None:
    path = Path(path)
    distribution, version, _ = _wheel_parts(path)
    if distribution != DOWNSTREAM_WHEEL_DISTRIBUTION:
        raise ValueError(
            f"{path.name}: expected wheel distribution "
            f"{DOWNSTREAM_WHEEL_DISTRIBUTION!r}"
        )
    if version != DOWNSTREAM_VERSION:
        raise ValueError(
            f"{path.name}: expected version {DOWNSTREAM_VERSION!r}, got {version!r}"
        )

    prefix = f"{DOWNSTREAM_WHEEL_DISTRIBUTION}-{version}.dist-info/"
    with zipfile.ZipFile(path, "r") as wheel:
        names = set(wheel.namelist())
        metadata_name = prefix + "METADATA"
        record_name = prefix + "RECORD"
        if metadata_name not in names or record_name not in names:
            raise ValueError(
                f"{path.name}: downstream .dist-info metadata is incomplete"
            )
        if "quadrants/__init__.py" not in names:
            raise ValueError(
                f"{path.name}: rebranding must retain the import package 'quadrants'"
            )
        metadata = wheel.read(metadata_name).decode("utf-8").replace("\r\n", "\n")
        if f"Name: {DOWNSTREAM_DISTRIBUTION}\n" not in metadata:
            raise ValueError(f"{path.name}: METADATA has the wrong distribution name")
        if f"Version: {DOWNSTREAM_VERSION}\n" not in metadata:
            raise ValueError(f"{path.name}: METADATA has the wrong version")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument("--delete-original", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    for wheel in args.wheels:
        if args.validate_only:
            validate_downstream_wheel(wheel)
            print(f"validated {wheel}")
        else:
            output = rebrand_wheel(wheel, delete_original=args.delete_original)
            print(f"{wheel.name} -> {output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
