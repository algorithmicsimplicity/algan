#!/usr/bin/env bash
# =============================================================================
# One scene, one box, one compiler, rendered on Metal and on the CPU, compared.
#
# This is the control `mps_probe_quadrants.sh` needed and did not have. That
# probe measured MPS against CPU on a **Quadrants** wheel built from
# `quadrants_patches/` and found them 255 apart across 95% of pixels, with the
# Metal arm about a third as bright (mean 16.4 against 47.6 -- dimmer, not
# blank). On its own that number cannot say whether the Quadrants port is
# wrong, because nobody had ever taken the same reading on **Taichi**: there
# are no committed macOS baselines, and `tests/full_renders` skips every
# comparison on a Mac, so how Algan's Metal path compares to its own CPU path
# is simply unmeasured on either compiler.
#
# So run it on both and read the difference of the differences:
#
#   Taichi shows a similar gap   -> Metal-vs-CPU is how Algan already behaves
#                                   here, and the probe was measuring devices,
#                                   not patches. The Quadrants port is not
#                                   implicated, and the real finding is that
#                                   Algan's Apple path is uncompared.
#   Taichi arms agree closely    -> the Quadrants port has a defect, and
#                                   `quadrants_patches/PORTING-NOTES.md` §5
#                                   ranks where to look first.
#
# Deliberately no build: this installs nothing. Under Taichi the harness's own
# `taichi_wheel_run_id` puts the patched wheel in place (and pins the backend
# with it); under Quadrants pass a wheel through `GATE_QD_WHEEL` or let it use
# whatever is installed.
#
#   # Taichi control -- the harness installs the patched wheel and pins taichi
#   command: bash scripts/gate/mps_vs_cpu_ab.sh
#   arms:    mac-mps
#   latex:   true
#
# The arm choice matters and is the opposite of the probe's: here MPS must be
# selectable *before* the command runs, because nothing in this script installs
# a wheel -- so the `mac-mps` arm, whose `algan check` proves the patched build
# is present, is exactly right.
# =============================================================================

set -o pipefail
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKDIR="${GATE_WORKDIR:-${RUNNER_TEMP:-/tmp}/gate-mps-vs-cpu}"
LOGDIR="${GATE_LOGDIR:-${GITHUB_WORKSPACE:-$PWD}/gate-logs}"
SCENE="${GATE_SCENE:-shapes_and_timeline}"
PYTHON="$REPO_ROOT/.venv/bin/python"
AB="$REPO_ROOT/scripts/gate/backend_pixel_ab.py"

STATUS="INCOMPLETE"
COMPILER=""
ARMS=""
PIXEL_MAX=""
PIXEL_VERDICT="not-run"
T0=$SECONDS
mkdir -p "$LOGDIR" "$WORKDIR"

stamp() { printf '[UTC +%02d:%02d]' $(( (SECONDS - T0) / 60 )) $(( (SECONDS - T0) % 60 )); }
say() { echo "$(stamp) $*"; }
rule() { echo; echo "=========================================================="; say "$*"; echo "=========================================================="; }

report() {
  rule "MPS-vs-CPU REPORT -- scene $SCENE"
  echo "compiler : ${COMPILER:-unknown}"
  echo "arms     : ${ARMS:-none}"
  echo "pixels   : $PIXEL_VERDICT${PIXEL_MAX:+, max channel delta $PIXEL_MAX}"
  echo
  echo "GATE-RESULT: gate=mps_vs_cpu_ab status=$STATUS compiler=${COMPILER:-unknown} \
scene=$SCENE arms=${ARMS// /,} pixels=$PIXEL_VERDICT max_delta=${PIXEL_MAX:--} \
total=$(( SECONDS - T0 ))s"
}
trap report EXIT

rule "0. what this process compiles with"
COMPILER="$("$PYTHON" -c "
from algan.taichi_compat import describe_backend
print(describe_backend().replace(' ', '-'))" 2>/dev/null | tail -1)"
say "compiler=$COMPILER"
"$PYTHON" -c "
from algan.rendering.mps_zero_copy import unavailable_reason, zero_copy_available
ok = zero_copy_available()
print('zero copy:', 'AVAILABLE' if ok else 'UNAVAILABLE -- ' + (unavailable_reason() or ''))
" 2>&1 | tail -1

rule "1. render $SCENE on each device"
for device in mps cpu; do
  say "rendering on $device"
  arm_log="$LOGDIR/mpsab-$device.log"
  if ALGAN_RENDER_DEVICE="$device" "$PYTHON" "$AB" --render \
      --out "$WORKDIR/$device" --scenes "$SCENE" >"$arm_log" 2>&1; then
    ARMS="$ARMS$device=ok "
    tail -n 4 "$arm_log" | sed 's/^/    /'
  else
    ARMS="$ARMS$device=FAILED "
    say "$device arm failed; last 60 lines:"
    tail -n 60 "$arm_log" | sed 's/^/    /'
    # A render killed by a signal prints nothing; macOS writes the report it
    # could not. The faulting thread and signal are what attribute a trap.
    say "macOS crash reports written in the last 10 minutes:"
    find "$HOME/Library/Logs/DiagnosticReports" -name '*.ips' -mmin -10 2>/dev/null \
      | while IFS= read -r crash; do
          say "  --- $crash"
          head -c 4000 "$crash" | sed 's/^/      /'
          cp "$crash" "$LOGDIR/" 2>/dev/null || true
        done
  fi
  cp "$WORKDIR/$device/$SCENE.mp4" "$LOGDIR/mpsab-$device-$SCENE.mp4" 2>/dev/null || true
done

case "$ARMS" in
  *FAILED*) STATUS="FAIL"; say "an arm did not render; nothing to compare"; exit 1 ;;
esac

rule "2. compare"
compare_log="$LOGDIR/mpsab-compare.log"
"$PYTHON" "$AB" --compare "$WORKDIR/mps" "$WORKDIR/cpu" 2>&1 | tee "$compare_log" | sed 's/^/    /'
PIXEL_MAX="$(grep -oE 'max_channel_delta=[0-9]+' "$compare_log" | grep -oE '[0-9]+' | head -1)"
PIXEL_VERDICT="$(grep -oE 'GATE-RESULT: [A-Z-]+' "$compare_log" | awk '{print $2}' | head -1)"
PIXEL_VERDICT="${PIXEL_VERDICT:-unknown}"
STATUS="PASS"
say "done"
