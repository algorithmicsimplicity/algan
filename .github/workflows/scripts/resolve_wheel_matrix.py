"""Resolve one Quadrants wheel-build request into GitHub Actions step outputs.

`quadrants_build.yaml` takes two free-text inputs -- which platforms to build
and which Pythons to build them for -- and every job downstream needs one
answer. This turns that pair into a per-platform `true`/`false` and a JSON
array of Python versions, as `name=value` lines on stdout for the caller to
append to `$GITHUB_OUTPUT`.

Kept out of the YAML for the reason `resolve_gpu_request.py` is: the failure
mode of a shell-and-`jq` version is an *empty or wrong matrix*, which produces
a green run that built nothing (or built the wrong Python) after the runner
minutes have already been spent. Here it is testable
(`tests/unit_tests/test_quadrants_wheels.py`).

The platform table is also the single source of truth for which runner image
each platform builds on, and `scripts/build_quadrants_wheels.py` imports it
from here rather than keeping a second copy that could drift.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# platform -> the runner it builds on, and the wheel platform tag `build.py`
# stamps for it (`qd_build/entry.py::build_wheel`). The tag is what
# `scripts/build_quadrants_wheels.py --install` matches a downloaded wheel
# against, so it belongs next to the runner rather than in the driver.
#
# Both images are pinned rather than `-latest`, and that is a finding rather
# than a preference: `taichi_build.yaml` documents a `macos-latest` roll that
# moved Xcode forward and broke the build outright. These two are the images
# Quadrants' own CI builds v1.3.0 on (`macosx.yml`, `win.yml`), so they are the
# configuration upstream actually tests.
PLATFORMS: dict[str, dict[str, str]] = {
    "linux": {
        "runner": "ubuntu-22.04",
        "wheel_tag": "manylinux_2_27_x86_64",
        "label": "linux-x86_64",
    },
    "macos": {
        "runner": "macos-26",
        "wheel_tag": "macosx_13_0_arm64",
        "label": "macos-arm64",
    },
    "windows": {
        "runner": "windows-2025",
        "wheel_tag": "win_amd64",
        "label": "windows-x86_64",
    },
}

# What both Algan and Quadrants declare in `requires-python` (>=3.10,<3.14).
# A wheel is per-interpreter -- it carries a compiled extension module -- so
# this is a real matrix dimension, not a formality.
PYTHONS: tuple[str, ...] = ("3.10", "3.11", "3.12", "3.13")

DEFAULTS = {
    # Every platform, because the whole point of the fork is that each one
    # carries patches the others cannot exercise: Metal (0001, 0002) is
    # macOS-only, and 0003 exists for a pre-Volta CUDA box that runs Windows.
    "platforms": "linux,macos,windows",
    # One Python by default. The build is ~15-20 minutes per wheel and cp311 is
    # what every other wheel in this repo is built for (`taichi_build.yaml`,
    # the `run_on_mac.yaml` arms). Widen it deliberately -- "3.10,3.11,3.12,3.13"
    # is a release matrix, twelve builds, not an iteration.
    "python_versions": "3.11",
}


def _split(raw: str) -> list[str]:
    """Split a comma/whitespace/newline separated list, dropping empties."""
    return [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]


def resolve(env: dict[str, str]) -> dict[str, str]:
    """Turn the dispatch inputs into the outputs the jobs gate on."""

    def pick(input_name: str, default: str) -> str:
        value = env.get(input_name, "")
        return value if value.strip() else default

    platforms = _split(pick("IN_PLATFORMS", DEFAULTS["platforms"]))
    if not platforms:
        raise SystemExit("no platforms selected")
    # "all" is spelled out rather than special-cased anywhere else, so that the
    # error below stays the only place that has to know the vocabulary.
    if platforms == ["all"]:
        platforms = list(PLATFORMS)
    unknown = [p for p in platforms if p not in PLATFORMS]
    if unknown:
        raise SystemExit(f"unknown platform(s) {unknown}; known: {sorted(PLATFORMS)}")

    pythons = _split(pick("IN_PYTHONS", DEFAULTS["python_versions"]))
    if not pythons:
        raise SystemExit("no Python versions selected")
    bad = [v for v in pythons if v not in PYTHONS]
    if bad:
        raise SystemExit(
            f"unsupported Python version(s) {bad}; Quadrants and Algan both "
            f"declare >=3.10,<3.14, so: {list(PYTHONS)}"
        )

    # Deduplicate while keeping the order asked for, so a repeated entry cannot
    # silently produce two jobs racing to upload the same artifact name.
    ordered_platforms = list(dict.fromkeys(platforms))
    ordered_pythons = list(dict.fromkeys(pythons))

    outputs = {name: str(name in ordered_platforms).lower() for name in PLATFORMS}
    # The runner image travels as an output so that `runs-on:` reads the table
    # above rather than a second copy pasted into the YAML, which is exactly
    # the kind of pair that drifts when one of the two is bumped.
    for name, spec in PLATFORMS.items():
        outputs[f"runner_{name}"] = spec["runner"]
    outputs["pythons"] = json.dumps(ordered_pythons)
    outputs["summary"] = (
        f"{','.join(ordered_platforms)} x py{','.join(ordered_pythons)} "
        f"({len(ordered_platforms) * len(ordered_pythons)} wheel(s))"
    )
    return outputs


def format_outputs(values: dict[str, str]) -> str:
    """Render as `$GITHUB_OUTPUT` lines. No value here is ever multi-line."""
    return "".join(f"{key}={value}\n" for key, value in values.items())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    sys.stdout.write(format_outputs(resolve(dict(os.environ))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
