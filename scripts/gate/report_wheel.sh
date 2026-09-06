#!/usr/bin/env bash
# =============================================================================
# Say what a wheel build actually produced, on the job summary and on stdout.
#
#   bash scripts/gate/report_wheel.sh <dist-dir> <platform> <python-version>
#
# Shared by all four legs of `.github/workflows/quadrants_build.yaml` so that
# the two Linux wheels, the macOS one and the Windows one are described in the
# same words and the same order -- otherwise reading four job summaries side by
# side is four different exercises.
#
# It reports rather than gates: the `if: always()` steps that call it run after
# a failed build too, where "no wheel" is the answer and a non-zero exit here
# would only replace the real error with this one. The single exception is a
# wheel whose version came out as `0.1`, which is loud because it is silent
# everywhere else -- see below.
#
# `0.1` means `setuptools_scm` could not see the `v1.3.0` tag and fell back.
# The wheel still installs and still works, so nothing downstream complains;
# what breaks is provenance, because two builds of different sources then carry
# the same version string. A patched build comes out as
# `1.3.1.dev0+gab9a58ab5.d<date>` -- setuptools_scm sees the tag *and* a dirty
# tree, since `git apply` leaves the patches uncommitted -- and that is the
# version to expect. It is also, usefully, one pip will never confuse with
# PyPI's 1.3.0.
set -uo pipefail

DIST_DIR="${1:?usage: report_wheel.sh <dist-dir> <platform> <python-version>}"
PLATFORM="${2:?missing platform}"
PYVER="${3:?missing python version}"

emit() {
  echo "$1"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    echo "$1" >>"$GITHUB_STEP_SUMMARY"
  fi
}

# `stat` and the sha256 tool are spelled differently on each of the three
# runners this has to work on; BSD stat on macOS, GNU stat elsewhere, and
# `shasum -a 256` where `sha256sum` is absent.
file_bytes() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || echo 0
}

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    echo "(no sha256 tool)"
  fi
}

wheel="$(ls -1 "$DIST_DIR"/quadrants-*.whl 2>/dev/null | head -1)"
if [ -z "$wheel" ]; then
  emit "**$PLATFORM / py$PYVER**: no wheel in \`$DIST_DIR\`"
  exit 0
fi

name="$(basename "$wheel")"
bytes="$(file_bytes "$wheel")"
sha="$(file_sha256 "$wheel")"
# quadrants-<version>-<pytag>-<abi>-<plattag>.whl
version="$(echo "$name" | cut -d- -f2)"

emit "**$PLATFORM / py$PYVER** \`$name\`"
emit ""
emit "| | |"
emit "| --- | --- |"
emit "| version | \`$version\` |"
emit "| bytes | $bytes ($((bytes / 1048576)) MiB) |"
emit "| sha256 | \`$sha\` |"
emit ""

case "$version" in
  0.1*)
    emit ":warning: **version is \`$version\`** -- setuptools_scm did not see the \`v1.3.0\` tag."
    emit "Two builds of different sources would carry the same version string. Check the clone's tags."
    ;;
esac
