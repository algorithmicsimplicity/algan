#!/usr/bin/env bash
# taichi-dev/taichi#8745 under both kernel compilers, on whichever arch the arm
# pins. This is the Metal half of step 4 of the fact-finding gate in
# `taichi_patches/PLAN.md` §6: #8744 and #8794 were settled on Linux x64 (both
# reproduce on both compilers), and #8745 is the one that only exists on Metal,
# so it needs the Mac runner.
#
# Launched through `.github/workflows/run_on_mac.yaml`:
#
#   command: bash scripts/gate/metal_repro_8745.sh
#   arms:    mac-mps,linux-cpu        # linux-cpu is the control
#
# The arm sets `ALGAN_RENDER_DEVICE`, not the Taichi arch, and this repro never
# imports algan -- so the mapping below is what turns an arm into an arch. The
# control arm matters: the CPU answer is already known to be correct on both
# compilers, so a Linux run that disagrees means the harness is broken rather
# than the backend.
#
# Not `set -e`: both compilers must report even when the first one dies, which
# is the whole point of running them side by side.
set -uo pipefail

case "${ALGAN_RENDER_DEVICE:-cpu}" in
  mps) ARCH=metal ;;
  cuda) ARCH=cuda ;;
  *) ARCH=cpu ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
SCRIPT=benchmarks/_upstream_repro_8745.py

echo "=== #8745 on arch=$ARCH (device=${ALGAN_RENDER_DEVICE:-unset}) ==="
uname -a
sw_vers 2>/dev/null || true

run_arm() {
  local backend="$1"
  shift
  echo ""
  echo "--- backend=$backend ---"
  local started
  started=$(date +%s)
  REPRO_BACKEND="$backend" REPRO_ARCH="$ARCH" "$@" "$SCRIPT"
  local status=$?
  echo "--- backend=$backend exited $status in $(( $(date +%s) - started ))s ---"
  return $status
}

# Taichi comes from the project environment, which on the MPS arm is the
# patched fork wheel the harness installs -- the build Algan actually ships to
# a Mac, and therefore the one whose behaviour matters.
run_arm taichi uv run python
taichi_status=$?

# Quadrants is pulled in as an overlay rather than added to the project: PyPI
# ships `macosx_13_0_arm64` cp310-cp313 wheels, the harness sets up 3.11, and
# the repro's per-variant child processes inherit the overlay through
# `sys.executable`.
run_arm quadrants uv run --with quadrants==1.3.0 python
quadrants_status=$?

echo ""
echo "GATE-RESULT: arch=$ARCH taichi_exit=$taichi_status quadrants_exit=$quadrants_status"
echo "GATE-RESULT: read the REPRO-8745 verdict lines above, one per backend"
