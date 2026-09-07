#!/usr/bin/env python3
"""Rebrand a patched Quadrants wheel for Algan's PyPI distribution.

The native build still happens as upstream ``quadrants``. For release builds
``SETUPTOOLS_SCM_PRETEND_VERSION`` pins the build-time version to
``1.3.0.post3`` so CMake and Python agree on the downstream version. This
script then changes only the *distribution* name from ``quadrants`` to
``algan-quadrants``. The import package stays ``quadrants``.

Changing the distribution name after compilation is safe because Quadrants'
native build consumes the version, not the PyPI distribution name. The wheel
is rewritten rather than merely renamed: ``METADATA``, the ``.dist-info``
directory, and ``RECORD`` are all updated.

Quadrants is Apache-2.0, and this is a *modified* redistribution of it, so the
rebrand also carries the license obligations that a modified build has and a
plain rename does not:

* The upstream ``LICENSE`` (the Apache 2.0 text) must reach the recipient.
  Upstream already ships it in ``.dist-info/licenses/`` and the rebrand copies
  it across, but that is now checked rather than assumed -- a wheel without it
  fails here instead of on PyPI.
* Section 4(b) wants a prominent statement that the files were changed. Two go
  in: ``MODIFICATIONS.txt`` inside the ``.dist-info``, listing the patches
  actually applied, and a header block prepended to the METADATA description,
  which is what renders at the top of the PyPI page.

Upstream ships no ``NOTICE`` file, so section 4(c) has nothing to propagate.
The statement of changes is generated from ``quadrants_patches/`` rather than
written out here, because that glob is also what the build applies -- see the
"Apply the patches" step in ``.github/workflows/quadrants_build.yaml``. Add a
patch and the notice grows with it.
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
DOWNSTREAM_VERSION = "1.3.0.post3"

UPSTREAM_VERSION = "1.3.0"
UPSTREAM_HOMEPAGE = "https://github.com/Genesis-Embodied-AI/quadrants"
ALGAN_REPOSITORY = "https://github.com/algorithmicsimplicity/algan"
PATCH_DIRECTORY = Path(__file__).resolve().parent.parent / "quadrants_patches"
MODIFICATIONS_MEMBER = "MODIFICATIONS.txt"
LICENSE_MEMBER = "licenses/LICENSE"


def _applied_patches() -> list[str]:
    """The patch files the build applies, in the order it applies them.

    Same glob and same sort as the workflow's "Apply the patches" step, so the
    notice cannot drift from what actually went into the binaries.
    """
    if not PATCH_DIRECTORY.is_dir():
        raise ValueError(
            f"{PATCH_DIRECTORY} is missing -- run this from an Algan checkout; "
            "the statement of changes is generated from it"
        )
    patches = sorted(path.name for path in PATCH_DIRECTORY.glob("[0-9]*.patch"))
    if not patches:
        raise ValueError(
            f"no patches in {PATCH_DIRECTORY} -- a wheel with nothing applied "
            f"is stock Quadrants and must not be published as "
            f"{DOWNSTREAM_DISTRIBUTION}"
        )
    return patches


def _modifications_notice() -> str:
    patches = "\n".join(f"    {name}" for name in _applied_patches())
    title = (
        f"{DOWNSTREAM_DISTRIBUTION} {DOWNSTREAM_VERSION} "
        "-- a MODIFIED build of Quadrants"
    )
    return f"""\
{title}
{"=" * len(title)}

This distribution is not the upstream Quadrants release. It is upstream
Quadrants v{UPSTREAM_VERSION} with the patches below applied to its sources
before compilation, republished under a different distribution name so that
it can be depended on by name. The import package is deliberately unchanged,
so `import quadrants` in this environment gets these modified binaries.

    Upstream project:  {UPSTREAM_HOMEPAGE}
    Upstream license:  Apache License 2.0 -- the full text is the LICENSE file
                       beside this one, in the same .dist-info directory.
    Modified by:       Algorithmic Simplicity, for the Algan animation engine
                       ({ALGAN_REPOSITORY})

Statement of changes (Apache License 2.0, section 4(b))
-------------------------------------------------------

The files changed are those touched by the following patches, applied in this
order:

{patches}

Each patch is published in full, with notes on what it changes and why, at

    {ALGAN_REPOSITORY}/tree/master/quadrants_patches

which is the authoritative description of how this build differs from
upstream.

The modifications are copyright (c) 2025-2026 Algorithmic Simplicity and are
licensed under the Apache License 2.0, on the same terms as the original.
"""


def _metadata_preamble() -> str:
    return f"""\
> **This is a modified build of Quadrants, not the upstream release.**
> `{DOWNSTREAM_DISTRIBUTION}` is upstream
> [Quadrants]({UPSTREAM_HOMEPAGE}) v{UPSTREAM_VERSION} with
> [Algan's patches]({ALGAN_REPOSITORY}/tree/master/quadrants_patches)
> applied before compilation; `MODIFICATIONS.txt` in this distribution's
> `.dist-info` lists them. The import package is unchanged --
> `import quadrants`. Licensed under the Apache License 2.0, like the
> original. For upstream Quadrants, install `quadrants` instead.

---

"""


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

    # The description body starts after the first blank line. Prepending the
    # "this is modified" block there puts it at the top of the PyPI page, which
    # is the most prominent place this distribution has.
    separator = next(
        (index for index, line in enumerate(rewritten) if not line.rstrip("\r\n")),
        None,
    )
    if separator is None:
        raise ValueError(
            "METADATA has no description body to carry the modification notice"
        )
    ending = rewritten[separator][len(rewritten[separator].rstrip("\r\n")) :] or "\n"
    preamble = _metadata_preamble()
    if ending != "\n":
        preamble = preamble.replace("\n", ending)
    rewritten.insert(separator + 1, preamble)

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

    # Generated before the wheel is opened: a missing quadrants_patches/ should
    # fail before anything is written, not halfway through.
    modifications = _modifications_notice().encode("utf-8")

    rows: list[tuple[str, str, str]] = []
    record_info: zipfile.ZipInfo | None = None
    metadata_seen = False
    license_seen = False
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
                elif name == old_prefix + LICENSE_MEMBER:
                    license_seen = True
                target.writestr(_renamed_info(info, renamed), data)
                if not renamed.endswith("/"):
                    rows.append((renamed, _record_hash(data), str(len(data))))

            if record_info is None:
                raise ValueError(f"{path.name}: no {old_prefix}RECORD member")
            if not metadata_seen:
                raise ValueError(f"{path.name}: no {old_prefix}METADATA member")
            if not license_seen:
                # Apache 2.0 section 4(a): recipients get a copy of the license.
                # Upstream ships it; if a build ever stops doing so, that has to
                # surface here rather than on PyPI.
                raise ValueError(
                    f"{path.name}: no {old_prefix}{LICENSE_MEMBER} -- the "
                    "upstream Apache 2.0 text must travel with the wheel"
                )

            # Apache 2.0 section 4(b): a prominent statement that files changed.
            modifications_name = new_prefix + MODIFICATIONS_MEMBER
            target.writestr(
                _renamed_info(record_info, modifications_name), modifications
            )
            rows.append(
                (
                    modifications_name,
                    _record_hash(modifications),
                    str(len(modifications)),
                )
            )

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

        # Apache 2.0 obligations for a modified redistribution. These are the
        # last gate before Trusted Publishing, and a PyPI upload is permanent.
        if prefix + LICENSE_MEMBER not in names:
            raise ValueError(
                f"{path.name}: missing {LICENSE_MEMBER} -- the upstream Apache "
                "2.0 text must ship with a redistribution of Quadrants"
            )
        if prefix + MODIFICATIONS_MEMBER not in names:
            raise ValueError(
                f"{path.name}: missing {MODIFICATIONS_MEMBER} -- a modified "
                "build must state that it changed the files"
            )
        notice = wheel.read(prefix + MODIFICATIONS_MEMBER).decode("utf-8")
        missing = [name for name in _applied_patches() if name not in notice]
        if missing:
            raise ValueError(
                f"{path.name}: {MODIFICATIONS_MEMBER} does not list {missing} -- "
                "the statement of changes is stale against quadrants_patches/"
            )
        if "modified build of Quadrants" not in metadata:
            raise ValueError(
                f"{path.name}: METADATA description does not say this is a "
                "modified build"
            )


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
