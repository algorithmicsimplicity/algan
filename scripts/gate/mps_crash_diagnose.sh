#!/usr/bin/env bash
# =============================================================================
# Why a dense scene dies on Metal, and whether bounding the batch stops it.
#
# `DESIGN_mps_support.md` §1.4: `materials_and_lighting` renders about two
# thirds of its frames on Metal and then dies with `Trace/BPT trap: 5`,
# printing nothing -- no traceback, no Metal diagnostic, no `.ips` report --
# on both kernel compilers. A render killed by a signal explains nothing by
# itself, so this run makes the box explain it instead:
#
#   1. what the device says it has, before anything renders;
#   2. the scene as it stands, under `lldb`, which turns a silent signal into a
#      native backtrace -- and, if the process was *killed* rather than
#      faulted, says that instead, since SIGKILL cannot be caught and lldb
#      reporting a kill is itself the answer;
#   3. the system's own account afterwards: crash reports, the unified log's
#      jetsam/memorystatus records, swap and page state;
#   4. the same scene with every frame window capped at 8 frames.
#
# Arm 4 is the discriminator, and it is the one arm that can be wrong in a
# useful direction. `benchmarks/_mps_batch_budget_repro.py` measured this scene
# on Linux CPU at ~5 GB of host memory, reached in the third of three windows
# (58, 47, 74 frames) -- and the third window is exactly where §1.4 saw Metal
# stop, on a runner with 7 GB shared between host and GPU. If capping the
# window to 8 frames renders the scene through, the trap is what a window
# costs. If it dies at 8 frames too, it is not, and the backtrace from arm 2 is
# the thing to read.
#
# What arm 4 is NOT is a test of `_render_device_pool_bytes`. That fix stops any
# device being handed an unsatisfiable budget, and the same script measured that
# it changes neither the windows nor the peak on this scene: the guards it feeds
# are gated on a CUDA device, so Metal barely reads them.
#
#   command: bash scripts/gate/mps_crash_diagnose.sh
#   arms:    mac-mps
#   latex:   true
#   timeout_minutes: 90
#
# `latex: true` because the scene draws `Text`, and the full-render scenes
# resolve their faces through the suite's fontconfig.
# =============================================================================

set -o pipefail
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGDIR="${GATE_LOGDIR:-${GITHUB_WORKSPACE:-$PWD}/gate-logs}"
SCENE="${GATE_SCENE:-materials_and_lighting}"
PYTHON="$REPO_ROOT/.venv/bin/python"
REPRO="$REPO_ROOT/benchmarks/_mps_batch_budget_repro.py"

STATUS="INCOMPLETE"
WHOLE_RC=""
CAPPED_RC=""
KILLER="unknown"
#: Frames per window in the capped arm. Small enough to be unambiguous against
#: the 74-frame window the render dies in, large enough to still batch.
CAP="${GATE_MAX_WINDOW:-8}"
T0=$SECONDS
mkdir -p "$LOGDIR"

stamp() { printf '[+%02d:%02d]' $(( (SECONDS - T0) / 60 )) $(( (SECONDS - T0) % 60 )); }
say() { echo "$(stamp) $*"; }
rule() { echo; echo "=========================================================="; say "$*"; echo "=========================================================="; }

report() {
  rule "MPS CRASH DIAGNOSIS -- scene $SCENE"
  echo "whole arm  (windows as sized)     : exit ${WHOLE_RC:-not-run}"
  echo "capped arm (windows <= $CAP frames) : exit ${CAPPED_RC:-not-run}"
  echo "what ended the whole arm          : $KILLER"
  echo
  echo "GATE-RESULT: gate=mps_crash_diagnose status=$STATUS scene=$SCENE \
whole=${WHOLE_RC:--} capped=${CAPPED_RC:--} cap=$CAP killer=$KILLER \
total=$(( SECONDS - T0 ))s"
}
trap report EXIT

# --------------------------------------------------------------------------
rule "1. what this box has"
sysctl -n hw.model hw.memsize hw.ncpu 2>/dev/null | sed 's/^/    /'
sysctl vm.swapusage 2>/dev/null | sed 's/^/    /'
vm_stat 2>/dev/null | head -8 | sed 's/^/    /'
"$PYTHON" - <<'PY' 2>&1 | sed 's/^/    /'
import torch

print("torch", torch.__version__, "mps", torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    # recommended_max_memory is Metal's recommendedMaxWorkingSetSize: the
    # figure the driver says a process should stay under. It is what a fixed
    # out-of-arena budget for MPS would be computed from.
    print(f"recommended_max_memory {torch.mps.recommended_max_memory() / 1e9:.2f} GB")
    print(f"driver_allocated       {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")
try:
    from algan.settings import _startup
    from algan.taichi_compat import describe_backend
    print("algan device", _startup.render_device().type, "backend", describe_backend())
    from algan.rendering.mps_zero_copy import unavailable_reason, zero_copy_available
    print("zero copy", "AVAILABLE" if zero_copy_available() else
          f"UNAVAILABLE -- {unavailable_reason() or ''}")
except Exception as exc:  # a probe must not take the run down
    print("algan probe failed:", exc)
PY

# The reports macOS wrote before this run are not ours; remember them so the
# ones we collect afterwards can be told apart from the runner's history.
BEFORE="$(mktemp)"
find "$HOME/Library/Logs/DiagnosticReports" /Library/Logs/DiagnosticReports \
  -name '*.ips' 2>/dev/null | sort > "$BEFORE"
say "$(wc -l < "$BEFORE") crash reports already on this box"

# --------------------------------------------------------------------------
rule "2. render the whole scene on Metal, under a debugger"
# `-k` runs its command when the target crashes, which is the whole point: a
# signalled render prints nothing, and `bt all` is the thing that names who
# raised it. A process the kernel *kills* never reaches those commands, and
# lldb prints the terminating signal instead -- which distinguishes a fault
# from a jetsam kill, the two candidates §1.4 could not separate.
WHOLE_LOG="$LOGDIR/crash-whole.log"
if command -v lldb >/dev/null 2>&1; then
  say "lldb $(lldb --version 2>&1 | head -1)"
  lldb --batch \
    -o "run" \
    -k "thread backtrace all" \
    -k "register read" \
    -k "quit 1" \
    -- "$PYTHON" "$REPRO" --scene "$SCENE" --ceiling-gb 0 \
    >"$WHOLE_LOG" 2>&1
  WHOLE_RC=$?
else
  say "no lldb on this box; running without a debugger"
  "$PYTHON" "$REPRO" --scene "$SCENE" --ceiling-gb 0 >"$WHOLE_LOG" 2>&1
  WHOLE_RC=$?
fi
say "whole arm exited $WHOLE_RC"
tail -n 120 "$WHOLE_LOG" | sed 's/^/    /'

# --------------------------------------------------------------------------
rule "3. what the system says ended it"
say "crash reports written during this run:"
AFTER="$(mktemp)"
find "$HOME/Library/Logs/DiagnosticReports" /Library/Logs/DiagnosticReports \
  -name '*.ips' 2>/dev/null | sort > "$AFTER"
NEW_REPORTS="$(comm -13 "$BEFORE" "$AFTER")"
if [ -z "$NEW_REPORTS" ]; then
  say "  none -- macOS wrote no report, so nothing faulted in a way it records"
else
  echo "$NEW_REPORTS" | while IFS= read -r crash; do
    [ -z "$crash" ] && continue
    say "  --- $crash"
    head -c 6000 "$crash" | sed 's/^/      /'
    cp "$crash" "$LOGDIR/" 2>/dev/null || true
  done
fi

# The unified log is where Metal driver errors and the kernel's memory-status
# decisions go; neither reaches the process's stderr, which is why §1.4 had
# nothing to read. `log show` needs no privilege for these.
say "jetsam / memorystatus records:"
log show --last 30m --style compact \
  --predicate 'eventMessage CONTAINS[c] "jetsam" OR eventMessage CONTAINS[c] "memorystatus" OR eventMessage CONTAINS[c] "lowswap"' \
  2>/dev/null | tail -n 40 | sed 's/^/    /' || say "    (log show unavailable)"

say "Metal / GPU records:"
log show --last 30m --style compact \
  --predicate 'senderImagePath CONTAINS "Metal" OR senderImagePath CONTAINS "AGX" OR eventMessage CONTAINS[c] "IOGPU"' \
  2>/dev/null | tail -n 60 | sed 's/^/    /' || say "    (log show unavailable)"

say "swap and page state after the arm:"
sysctl vm.swapusage 2>/dev/null | sed 's/^/    /'
vm_stat 2>/dev/null | head -8 | sed 's/^/    /'

if grep -qiE 'killed by signal 9|SIGKILL' "$WHOLE_LOG"; then
  KILLER="SIGKILL-kernel-kill"
elif grep -qiE 'stop reason = signal SIGTRAP|EXC_BREAKPOINT' "$WHOLE_LOG"; then
  KILLER="SIGTRAP-trap"
elif grep -qiE 'stop reason = signal SIGABRT|EXC_CRASH' "$WHOLE_LOG"; then
  KILLER="SIGABRT-abort"
elif grep -qiE 'out of memory|OutOfRenderMemory' "$WHOLE_LOG"; then
  KILLER="python-out-of-memory"
elif [ "$WHOLE_RC" = "0" ]; then
  KILLER="none-it-completed"
fi
say "verdict on the whole arm: $KILLER"

# --------------------------------------------------------------------------
rule "4. the discriminator: the same scene at $CAP frames per window"
# Nothing else changes -- same device, same pool, same kernels, same scene.
# Only how many frames share a merge, a materialization and an arena.
CAPPED_LOG="$LOGDIR/crash-capped.log"
"$PYTHON" "$REPRO" --scene "$SCENE" --max-window "$CAP" --ceiling-gb 0 \
  >"$CAPPED_LOG" 2>&1
CAPPED_RC=$?
say "capped arm exited $CAPPED_RC"
tail -n 60 "$CAPPED_LOG" | sed 's/^/    /'

if [ "$WHOLE_RC" = "0" ]; then
  STATUS="NO-REPRO"
elif [ "$CAPPED_RC" = "0" ]; then
  STATUS="WINDOW-SIZE-IS-THE-CAUSE"
else
  STATUS="NOT-THE-WINDOW--READ-THE-BACKTRACE"
fi
say "done"
