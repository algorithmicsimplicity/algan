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

# platform -> the runner it builds on, the container it builds *in* where there
# is one, and the wheel platform tag the finished wheel is stamped with. The
# tag is what `scripts/build_quadrants_wheels.py --install` matches a downloaded
# wheel against and what `scripts/validate_quadrants_release.py` files a release
# by, so it belongs next to the image that makes it true.
#
# Every image is pinned rather than `-latest`, and that is a finding rather
# than a preference: `taichi_build.yaml` documents a `macos-latest` roll that
# moved Xcode forward and broke the build outright. `macos-26` and
# `windows-2025` are the images Quadrants' own CI builds v1.3.0 on
# (`macosx.yml`, `win.yml`), so they are the configuration upstream tests.
#
# ---------------------------------------------------------------------------
# WHY THE LINUX LEGS BUILD IN A CONTAINER, AND WHY THE TWO CONTAINERS DIFFER
#
# A Linux wheel's `manylinux_X_Y` tag is a promise about the oldest glibc it
# runs on, and `qd_build/entry.py::build_wheel` stamps `manylinux_2_27`
# unconditionally -- nothing measures anything. On a stock GitHub runner that
# promise is broken by construction: measured on the published
# `algan_quadrants-1.3.0.post1-cp311-cp311-manylinux_2_27_x86_64.whl`, which
# this leg built on `ubuntu-22.04`, the maximum versioned symbol is
# **GLIBC_2.34** (`pthread_create`, `dlopen` -- the 2.34 libpthread/libdl
# merge). Seven versions of overclaim, which pip cannot see: it installs the
# wheel on any glibc >= 2.27 and the failure lands at `import quadrants` as
# `version GLIBC_2.34 not found`. Moving to a newer runner only makes it worse
# (24.04 is glibc 2.39, 26.04 is 2.43), which is why "just bump the image"
# was not the answer to `ubuntu-22.04`'s retirement on 2027-04-17.
#
# In a manylinux container the build image's glibc *is* the floor, the tag
# becomes a measured fact (`scripts/gate/verify_wheel_tag.py` fails the build
# if auditwheel disagrees with the tag below), and the host runner stops
# mattering -- so the runner can track whatever GitHub currently supports
# without touching the wheels. Measured, first container build (run
# 34032073850): the x86-64 wheel came out at **GLIBC_2.27**, down from the
# 2.34 the `ubuntu-22.04` leg was shipping, which is RHEL 8, Ubuntu 20.04 and
# Debian 11 going from "pip installs it and the import fails" to working.
#
# The tag stamped is the **container's policy**, not that measurement, and the
# gap is deliberate: auditwheel said 2.27 and the wheel is stamped 2.28. A
# release matrix needs a tag that is the same on every build --
# `validate_quadrants_release.py` files sixteen wheels by it and
# `build_quadrants_wheels.py --install` matches on it -- whereas a measured
# tag moves whenever a dependency reaches for a newer symbol. The cost is
# glibc 2.27 exactly (Ubuntu 18.04, EOL); the gate still fails the build if
# the measurement ever exceeds the stamp, which is the direction that hurts.
#
# The two containers differ because **the prebuilt LLVM 22.1.0 toolchain that
# `download_llvm.py` fetches is not the same binary on the two architectures**,
# measured from the archives themselves:
#
#   x86_64  `bin/llvm-config` -> max GLIBC_2.14, GLIBCXX_3.4.21
#   aarch64 `bin/llvm-config` -> max GLIBC_2.34 (`__libc_start_main`),
#                                GLIBCXX_3.4.29
#
# So AlmaLinux 8 (glibc 2.28) can run the x86-64 clang and *cannot* run the
# aarch64 one: the aarch64 leg needs an image at glibc >= 2.34, and
# `manylinux_2_34` (AlmaLinux 9) is the lowest published one that clears it.
# This is not a guess about upstream's choice, it is upstream's choice --
# their PyPI wheels measure 2.27 on x86_64 and 2.34 on aarch64, and carry
# `manylinux_2_28` / `manylinux_2_34` as their second tag.
#
# `manylinux_2_34` is marked ALPHA by pypa, and the caveat their README
# attaches to it is **x86_64-only** (RHEL 9 derivatives target x86-64-v2, which
# auditwheel cannot detect, pypa/manylinux#1725) -- this uses it for aarch64
# alone, where that does not arise. Upstream ships production wheels built in
# it, and the alternative for aarch64 is strictly worse: no container at all
# means the runner's glibc (2.35 on 22.04, 2.39 on 24.04), which loses RHEL 9
# and Amazon Linux 2023 -- both exactly 2.34, and much of the aarch64 install
# base. Revisit if pypa promotes it, drops it, or ships a 2_34-class image that
# is not alpha; do not reach for `manylinux_2_34_x86_64` for the other leg
# without reading that issue first.
# ---------------------------------------------------------------------------
#
# A key is also a **job id and an artifact name component**, so it is
# `[a-z0-9_]+`: `quadrants_build.yaml` names one job per key,
# `scripts/build_quadrants_wheels.py`'s `ARTIFACT_RE` parses it back out of
# `quadrants-wheel-<key>-py<version>`, and a hyphen in an output name would be
# read as subtraction by GitHub's expression syntax
# (`needs.plan.outputs.runner_linux-arm64`).
PLATFORMS: dict[str, dict[str, str]] = {
    "linux": {
        "runner": "ubuntu-24.04",
        "container": "quay.io/pypa/manylinux_2_28_x86_64",
        "wheel_tag": "manylinux_2_28_x86_64",
        "label": "linux-x86_64",
    },
    # aarch64 needs nothing new from Quadrants itself: `qd_build/llvm.py`
    # already matches `("Linux", "aarch64")` and downloads
    # `taichi-llvm-22.1.0-linux-aarch64.zip`, `build_wheel` already has an
    # aarch64 arm, and upstream ships aarch64 wheels for all four Pythons -- so
    # this is a configuration upstream builds too, not new ground. What it does
    # need is its own container, for the reason above.
    "linux_arm64": {
        "runner": "ubuntu-24.04-arm",
        "container": "quay.io/pypa/manylinux_2_34_aarch64",
        "wheel_tag": "manylinux_2_34_aarch64",
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
    # The runner image, the container and the wheel tag all travel as outputs so
    # that the YAML reads the table above rather than a second copy pasted into
    # it, which is exactly the kind of pair that drifts when one of the two is
    # bumped. It is also what keeps the two Linux jobs textually identical
    # despite building in different containers for different tags.
    for name, spec in PLATFORMS.items():
        outputs[f"runner_{name}"] = spec["runner"]
        outputs[f"wheel_tag_{name}"] = spec["wheel_tag"]
        if "container" in spec:
            outputs[f"container_{name}"] = spec["container"]
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
