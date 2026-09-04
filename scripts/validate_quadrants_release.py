#!/usr/bin/env python3
"""Validate the complete ``algan-quadrants`` wheel matrix before PyPI upload."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from rebrand_quadrants_wheel import (
    DOWNSTREAM_VERSION,
    DOWNSTREAM_WHEEL_DISTRIBUTION,
    validate_downstream_wheel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_resolver():
    path = REPO_ROOT / ".github" / "workflows" / "scripts" / "resolve_wheel_matrix.py"
    spec = importlib.util.spec_from_file_location("resolve_wheel_matrix", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_release(directory: Path) -> list[Path]:
    resolver = _load_resolver()
    wheels = sorted(Path(directory).glob("*.whl"))
    expected_count = len(resolver.PLATFORMS) * len(resolver.PYTHONS)
    if len(wheels) != expected_count:
        raise ValueError(
            f"expected {expected_count} wheels for the complete release matrix, "
            f"found {len(wheels)}: {[wheel.name for wheel in wheels]}"
        )

    for wheel in wheels:
        validate_downstream_wheel(wheel)

    for platform, platform_spec in resolver.PLATFORMS.items():
        platform_tag = platform_spec["wheel_tag"]
        for python_version in resolver.PYTHONS:
            cp = "cp" + python_version.replace(".", "")
            matches = [
                wheel
                for wheel in wheels
                if f"-{cp}-{cp}-" in wheel.name and platform_tag in wheel.name
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one {platform}/py{python_version} wheel "
                    f"({cp}, {platform_tag}), found {[wheel.name for wheel in matches]}"
                )

    expected_prefix = f"{DOWNSTREAM_WHEEL_DISTRIBUTION}-{DOWNSTREAM_VERSION}-"
    if any(not wheel.name.startswith(expected_prefix) for wheel in wheels):
        raise ValueError(f"every release wheel must start with {expected_prefix!r}")
    return wheels


def main(argv: list[str] | None = None) -> int:
    directory = Path(argv[0]) if argv else Path("dist")
    wheels = validate_release(directory)
    print(f"validated {len(wheels)} wheels for algan-quadrants {DOWNSTREAM_VERSION}")
    for wheel in wheels:
        print(f"  {wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
