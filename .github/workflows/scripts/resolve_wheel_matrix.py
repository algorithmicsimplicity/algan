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
# Every image is pinned rather than `-latest`, and that is a finding rather
# than a preference: `taichi_build.yaml` documents a `macos-latest` roll that
# moved Xcode forward and broke the build outright. `macos-26` and
# `windows-2025` are the images Quadrants' own CI builds v1.3.0 on
# (`macosx.yml`, `win.yml`), so they are the configuration upstream tests.
#
# **The two Linux images are deprecating together.** `ubuntu-22.04` and
# `ubuntu-22.04-arm` began deprecation 2026-09-17 (brownouts) and are
# unsupported after 2027-04-17, so both legs need a new image before then.
# Bumping the two strings to `-24.04` is the cheap version and silently raises
# the glibc floor of both wheels (see the aarch64 entry); the move that makes
# the stamped manylinux tag honest rather than merely newer is to build both in
# a manylinux container (`quay.io/pypa/manylinux_2_28_*`), which is what
# upstream's own CI does.
#
# A key is also a **job id and an artifact name component**, so it is
# `[a-z0-9_]+`: `quadrants_build.yaml` names one job per key,
# `scripts/build_quadrants_wheels.py`'s `ARTIFACT_RE` parses it back out of
# `quadrants-wheel-<key>-py<version>`, and a hyphen in an output name would be
# read as subtraction by GitHub's expression syntax
# (`needs.plan.outputs.runner_linux-arm64`).
PLATFORMS: dict[str, dict[str, str]] = {
    "linux": {
        "runner": "ubuntu-22.04",
        "wheel_tag": "manylinux_2_27_x86_64",
        "label": "linux-x86_64",
    },
    # aarch64 is the one platform here that needs nothing new from Quadrants:
    # `qd_build/llvm.py` already matches `("Linux", "aarch64")` and downloads
    # `taichi-llvm-22.1.0-linux-aarch64.zip`, `qd_build/entry.py::build_wheel`
    # already stamps `manylinux_2_27_aarch64` for it, and upstream ships
    # `quadrants-1.3.0-*-manylinux_2_27_aarch64...whl` on PyPI for all four
    # Pythons -- so this is a configuration upstream builds too, not new ground.
    #
    # `ubuntu-22.04-arm` rather than `-24.04-arm`, and the reason is the wheel
    # tag rather than symmetry. `build_wheel` *stamps* `manylinux_2_27_aarch64`
    # (`python -m wheel tags --platform-tag`) and nothing audits it, so the tag
    # is a claim about the oldest glibc the wheel runs on and the runner image
    # is what makes it true or false. 22.04 is glibc 2.35, the same overclaim
    # the x86-64 leg above already ships; 24.04 is 2.39, which would put the
    # aarch64 wheel out of reach of Ubuntu 22.04 and Debian 12 -- JetPack 6 is
    # 22.04-based, i.e. exactly the CUDA-on-aarch64 user 0003 and 0005-0007
    # exist for -- while still claiming 2.27.
    "linux_arm64": {
        "runner": "ubuntu-22.04-arm",
        "wheel_tag": "manylinux_2_27_aarch64",
        "label": "linux-aarch64",
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
    "platforms": "linux,linux_arm64,macos,windows",
    # One Python by default. The build is ~15-20 minutes per wheel and cp311 is
    # what every other wheel in this repo is built for (`taichi_build.yaml`,
    # the `run_on_mac.yaml` arms). Widen it deliberately -- "3.10,3.11,3.12,3.13"
    # is a release matrix, sixteen builds, not an iteration.
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
    # The selection as a list, which is what `--check-publish` reads back. The
    # per-platform `true`/`false` outputs above are what a job's `if:` can gate
    # on; a release gate wants to ask "is every platform here?", and asking that
    # of a hand-written list of names in the YAML is how a new platform gets
    # left out of the completeness check that exists to catch exactly that.
    outputs["selected"] = json.dumps(ordered_platforms)
    outputs["summary"] = (
        f"{','.join(ordered_platforms)} x py{','.join(ordered_pythons)} "
        f"({len(ordered_platforms) * len(ordered_pythons)} wheel(s))"
    )
    return outputs


def publish_failures(env: dict[str, str]) -> list[str]:
    """Why this dispatch may not publish to PyPI -- empty when it may.

    A PyPI release is the complete patched matrix or it is nothing: a version
    published with one platform missing cannot be amended later, because PyPI
    files are immutable and the publish step uploads the whole directory. So
    this reads the *table* rather than a list of platform names copied into the
    workflow, and a platform added above is in the gate the moment it is added.
    """
    failures = []
    if env.get("APPLY_PATCHES") != "true":
        failures.append("publish requires apply_patches=true")

    selected = set(json.loads(env.get("SELECTED") or "[]"))
    for name in PLATFORMS:
        if name not in selected:
            failures.append(f"publish requires {name}")

    if set(json.loads(env.get("PYTHONS") or "[]")) != set(PYTHONS):
        failures.append(f"publish requires Python {list(PYTHONS)}")
    return failures


def format_outputs(values: dict[str, str]) -> str:
    """Render as `$GITHUB_OUTPUT` lines. No value here is ever multi-line."""
    return "".join(f"{key}={value}\n" for key, value in values.items())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-publish",
        action="store_true",
        help=(
            "instead of resolving, fail unless APPLY_PATCHES/SELECTED/PYTHONS "
            "describe the complete patched matrix a PyPI release requires"
        ),
    )
    args = parser.parse_args(argv)

    if args.check_publish:
        failures = publish_failures(dict(os.environ))
        if failures:
            raise SystemExit(
                "Refusing a partial/stock PyPI release:\n  - " + "\n  - ".join(failures)
            )
        print(
            f"complete patched matrix: {len(PLATFORMS)} platforms x "
            f"{len(PYTHONS)} Pythons = {len(PLATFORMS) * len(PYTHONS)} wheels"
        )
        return 0

    sys.stdout.write(format_outputs(resolve(dict(os.environ))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
