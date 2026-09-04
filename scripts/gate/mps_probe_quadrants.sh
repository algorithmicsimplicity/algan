#!/usr/bin/env bash
# =============================================================================
# Does Algan's Apple-GPU path work on a Quadrants wheel built from
# `quadrants_patches/`?
#
# `scripts/gate/quadrants_macos_build.sh` proved those patches *compile*. This
# is the other half, and the one that matters: it builds the same wheel,
# installs it, and then asks the Apple GPU to render with it.
#
#   `.github/workflows/run_on_mac.yaml`, arm mac-CPU (not mac-mps):
#       command: bash scripts/gate/mps_probe_quadrants.sh
#       env:     ALGAN_TAICHI_BACKEND=quadrants
#       taichi_wheel_run_id: none
#       timeout_minutes: 120
#
# **The arm is `mac-cpu` and this script selects MPS itself**, which looks
# backwards and is not. The arm only pins `ALGAN_RENDER_DEVICE`; the GPU is
# there either way. On `mac-mps` the harness's own "Report the environment
# Algan resolved" step runs `algan check` *before* the command -- at which
# point the patched wheel this script exists to build has not been built, so
# Algan correctly refuses to select MPS without one and the job dies before the
# script starts. Measured, on the first attempt: `AlganConfigurationError:
# ALGAN_RENDER_DEVICE requests MPS, but rendering on MPS needs the patched
# build`. So the device is switched on below, after the install, where the
# refusal would be a real finding instead of an ordering artefact.
#
# `taichi_wheel_run_id: none` stops the harness installing the patched *Taichi*
# wheel (and with it the `ALGAN_TAICHI_BACKEND=taichi` pin it writes to
# GITHUB_ENV when it does), and the `env:` entry is exported inside the run
# step, so it wins over anything the harness set. Without them this measures
# Taichi and says "quadrants" nowhere.
#
# WHY NOT `mps_probe.yaml`. That workflow is the Metal instrument, and it is
# wired to Taichi end to end: it downloads a `taichi_build.yaml` artifact and
# pins the backend. Generalising it would mean two more inputs and a second
# artifact path through a workflow whose whole value is that its readings are
# comparable across runs. Building the wheel inside one general-harness run
# needs neither, and cross-run artifact plumbing is what it avoids.
#
# WHAT IT CAN AND CANNOT SAY. There are no committed macOS pixel baselines
# (`tests/full_renders/test_full_renders.py` renders all six scenes on a Mac and
# skips every comparison), and `agent_guidance/gpu_harnesses.md` is explicit
# that neither GPU harness baselines pixels. So the pixel question is asked the
# only way it can be here: the SAME scene, SAME compiler, rendered on Metal and
# on this machine's own CPU, compared to each other. That is a real correctness
# reading -- the zero-copy path either writes what the CPU path writes or it
# does not -- and it is not a baseline.
# =============================================================================

set -o pipefail
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKDIR="${GATE_WORKDIR:-${RUNNER_TEMP:-/tmp}/gate-mps-quadrants}"
LOGDIR="${GATE_LOGDIR:-${GITHUB_WORKSPACE:-$PWD}/gate-logs}"
BUILD_WORKDIR="$WORKDIR/build"
AB_DIR="$WORKDIR/ab"
SCENE="${GATE_SCENE:-shapes_and_timeline}"
PYTHON="$REPO_ROOT/.venv/bin/python"

STATUS="INCOMPLETE"
FAILED_PHASE="startup"
WHEEL_NAME=""
DEVICE=""
COMPILER=""
ZERO_COPY=""
ZERO_COPY_WHY=""
SMOKE_COLD=""
SMOKE_WARM=""
PIXEL_MAX=""
PIXEL_VERDICT="not-run"
T0=$SECONDS

mkdir -p "$LOGDIR" "$WORKDIR"

stamp() { printf '[UTC +%02d:%02d]' $(( (SECONDS - T0) / 60 )) $(( (SECONDS - T0) % 60 )); }
say() { echo "$(stamp) $*"; }
rule() { echo; echo "=============================================================="; say "$*"; echo "=============================================================="; }
die() { FAILED_PHASE="$1"; shift; STATUS="FAIL"; say "FATAL ($FAILED_PHASE): $*"; exit 1; }

report() {
  rule "MPS PROBE REPORT -- Quadrants + quadrants_patches/ on a real Apple GPU"
  echo "wheel     : ${WHEEL_NAME:-(none built)}"
  echo "compiler  : ${COMPILER:-unknown}"
  echo "device    : ${DEVICE:-unknown}"
  echo "zero copy : ${ZERO_COPY:-unknown}${ZERO_COPY_WHY:+ ($ZERO_COPY_WHY)}"
  echo "smoke     : cold=${SMOKE_COLD:--}s warm=${SMOKE_WARM:--}s"
  echo "pixels    : mps vs cpu, scene $SCENE -- $PIXEL_VERDICT${PIXEL_MAX:+, max channel delta $PIXEL_MAX}"
  echo
  echo "GATE-RESULT: gate=mps_probe_quadrants status=$STATUS phase=$FAILED_PHASE \
wheel=${WHEEL_NAME:-none} compiler=${COMPILER:-unknown} device=${DEVICE:-unknown} \
zero_copy=${ZERO_COPY:-unknown} smoke_cold=${SMOKE_COLD:--}s smoke_warm=${SMOKE_WARM:--}s \
pixels=$PIXEL_VERDICT max_delta=${PIXEL_MAX:--} total=$(( SECONDS - T0 ))s"
}
trap report EXIT

# -----------------------------------------------------------------------------
rule "1. build the patched wheel"
FAILED_PHASE="build"
# Delegating rather than duplicating: that script carries every fix this runner
# needed (the stale xcrun cache, --force-bottle, the parallelism cap) and prints
# its own GATE-RESULT, which lands in this log above ours.
GATE_QD_PATCHES=1 GATE_WORKDIR="$BUILD_WORKDIR" GATE_LOGDIR="$LOGDIR" \
  bash "$REPO_ROOT/scripts/gate/quadrants_macos_build.sh" \
  || die build "the patched Quadrants build failed -- read its GATE-RESULT above"

wheel="$(ls -1 "$BUILD_WORKDIR"/quadrants-src/dist/*.whl 2>/dev/null | head -1)"
[ -n "$wheel" ] || die build "the build reported success but left no wheel"
WHEEL_NAME="$(basename "$wheel")"
say "built $WHEEL_NAME"

rule "2. install it over the PyPI Quadrants"
FAILED_PHASE="install"
# `uv pip`, not `uv run`/`uv sync`: a sync would resolve `quadrants` back to the
# PyPI release and silently undo this. Every probe below therefore calls
# `.venv/bin/python` directly, never `uv run`.
uv pip install --python "$PYTHON" --reinstall "$wheel" >/dev/null 2>&1 \
  || die install "could not install the built wheel into .venv"
"$PYTHON" -c "
import quadrants as qd
print('installed quadrants', qd.__version__)
from quadrants.lang._ndarray import ExternalMetalNdarray  # noqa: F401
print('patched: ExternalMetalNdarray present')
" || die install "the installed wheel is not the patched one (ExternalMetalNdarray missing)"

rule "3. what Algan resolved"
FAILED_PHASE="resolve"
# Now, and not before: with the patched wheel installed, asking for MPS is a
# question about the patch rather than about the order the harness runs its
# steps in. If Algan still refuses here, that IS the finding -- it means
# `zero_copy_available()` cannot see the entry points patch 0001 adds.
export ALGAN_RENDER_DEVICE=mps
"$PYTHON" -m algan.cli check 2>&1 | sed 's/^/    /'
COMPILER="$("$PYTHON" -c "
from algan.taichi_compat import describe_backend
print(describe_backend().replace(' ', '-'))" 2>/dev/null | tail -1)"
DEVICE="$("$PYTHON" -c "
import algan
from algan.settings import _startup
print(_startup.render_device().type)" 2>/dev/null | tail -1)"
say "compiler=$COMPILER device=$DEVICE"
case "$COMPILER" in
  quadrants*) ;;
  *) die resolve "this run is compiling with '$COMPILER', not Quadrants -- \
pass ALGAN_TAICHI_BACKEND=quadrants in the request's env and taichi_wheel_run_id: none" ;;
esac
[ "$DEVICE" = "mps" ] || die resolve "Algan resolved device '$DEVICE', not mps -- \
an MPS arm that fell back to the CPU measures nothing (gpu_harnesses.md)"

rule "4. zero-copy availability"
FAILED_PHASE="zero_copy"
# The whole point of patch 0001. `unavailable_reason()` is what says *why* when
# it is off, which on a stock wheel is "the entry point is missing" -- if this
# says that here, the patch did not reach the wheel Algan is using.
zc="$("$PYTHON" -c "
from algan.rendering.mps_zero_copy import unavailable_reason, zero_copy_available
available = zero_copy_available()
print('AVAILABLE' if available else 'UNAVAILABLE')
print((unavailable_reason() or '') if not available else '')
" 2>&1 | tail -2)"
ZERO_COPY="$(echo "$zc" | head -1)"
ZERO_COPY_WHY="$(echo "$zc" | tail -1)"
say "zero copy: $ZERO_COPY ${ZERO_COPY_WHY:+-- $ZERO_COPY_WHY}"

rule "5. a real render on Metal"
FAILED_PHASE="smoke"
smoke_log="$LOGDIR/mps-smoke.log"
"$PYTHON" "$REPO_ROOT/scripts/gpu_smoke.py" --runs 2 2>&1 | tee "$smoke_log" | sed 's/^/    /' \
  || die smoke "gpu_smoke.py failed on the Apple GPU"
SMOKE_COLD="$(grep -oE 'cold[^0-9]*([0-9.]+)' "$smoke_log" | grep -oE '[0-9.]+' | head -1)"
SMOKE_WARM="$(grep -oE 'warm[^0-9]*([0-9.]+)' "$smoke_log" | grep -oE '[0-9.]+' | head -1)"

rule "6. Metal against this machine's own CPU, same compiler"
FAILED_PHASE="pixels"
# Not a baseline comparison -- there are no committed macOS baselines. Two
# devices, one box, one compiler, one scene: the zero-copy path either writes
# what the CPU path writes or it does not. MPS-friendly mode substitutes
# float32 accumulators where Metal cannot run the wide ones, so a small delta
# is expected and the number is the reading, not a pass/fail.
ab="$REPO_ROOT/scripts/gate/backend_pixel_ab.py"
# Both arms run even when the first fails, and each keeps its whole output.
# The first version of this piped through `tail -3` and died on the first
# failure, which cost a whole 18-minute run to learn only that "the mps arm
# failed" -- the three surviving lines were the tail of an Algan advisory, not
# the exception. A render that fails here is a finding, so it has to arrive
# legible.
arm_status=""
for device in mps cpu; do
  say "rendering $SCENE on $device"
  arm_log="$LOGDIR/ab-$device.log"
  if ALGAN_RENDER_DEVICE="$device" "$PYTHON" "$ab" --render --out "$AB_DIR/$device" \
      --scenes "$SCENE" >"$arm_log" 2>&1; then
    arm_status="$arm_status$device=ok "
    tail -n 5 "$arm_log" | sed 's/^/    /'
  else
    arm_status="$arm_status$device=FAILED "
    say "the $device arm failed; last 60 lines of $arm_log:"
    tail -n 60 "$arm_log" | sed 's/^/    /'
  fi
done
say "pixel arms: $arm_status"

case "$arm_status" in
  *FAILED*)
    PIXEL_VERDICT="ARM-FAILED"
    die pixels "a pixel-A/B arm did not render ($arm_status) -- the dump above is why"
    ;;
esac

# Both clips travel home with the logs. `$AB_DIR` is under `$RUNNER_TEMP`,
# which the harness's `artifacts:` glob cannot reach, so a disagreement between
# the arms was diagnosable only by re-running with more instrumentation. At
# PREVIEW these are a couple of MB each.
for device in mps cpu; do
  cp "$AB_DIR/$device/$SCENE.mp4" "$LOGDIR/ab-$device-$SCENE.mp4" 2>/dev/null \
    || say "note: no $device clip to copy back"
done

compare_log="$LOGDIR/mps-vs-cpu.log"
"$PYTHON" "$ab" --compare "$AB_DIR/mps" "$AB_DIR/cpu" 2>&1 | tee "$compare_log" | sed 's/^/    /'
PIXEL_MAX="$(grep -oE 'max_channel_delta=[0-9]+' "$compare_log" | grep -oE '[0-9]+' | head -1)"
PIXEL_VERDICT="$(grep -oE 'GATE-RESULT: [A-Z-]+' "$compare_log" | awk '{print $2}' | head -1)"
PIXEL_VERDICT="${PIXEL_VERDICT:-unknown}"

STATUS="PASS"
FAILED_PHASE="none"
say "done"
