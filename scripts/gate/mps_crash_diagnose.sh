#!/usr/bin/env bash
# =============================================================================
# The §1.4 leak, reinstated and then not: two renders of the same scene on
# Metal, differing only in when the zero-copy import cache is released.
#
# `DESIGN_mps_support.md` §1.4 was "a dense scene dies two thirds of the way
# through, silently, on both compilers". It is a leak on the device:
# `mps_zero_copy`'s import cache pins a torch storage per buffer it has handed
# a kernel, and it used to be released once per render job -- so every batch's
# uploaded arrays stayed live until the last frame. The first run of this
# script measured it climbing from 0.64 GB to 6.74 GB over fifteen batches on
# a machine with 7 GB and no swap.
#
#   1. what the device says it has, before anything renders;
#   2. the scene with the defect reinstated (`--leak-cache`) -- expect the
#      trap, and expect `pinned=` to track `mps_alloc=` on the way there;
#   3. the system's own account: crash reports, jetsam records, page state;
#   4. the scene as the engine now stands -- expect it through to the end with
#      `mps_alloc` rising and falling per batch instead of only rising.
#
#   command: bash scripts/gate/mps_crash_diagnose.sh
#   arms:    mac-mps
#   latex:   true
#   timeout_minutes: 90
#
# `latex: true` because the scene draws `Text`, and the full-render scenes
# resolve their faces through the suite's fontconfig.
#
# NOT under a debugger. The first version ran arm 2 under `lldb --batch -o run`
# and got this, in thirteen seconds:
#
#     Process 3806 stopped
#     * thread #2, stop reason = exec
#         frame #0: 0x00000001000189c0 dyld`_dyld_start
#
# `run` stops at the exec event and `--batch` then quits, so the render never
# happened and the arm reported exit 0 -- a voided arm that reads exactly like
# a passing one. `-o "settings set target.process.stop-on-exec false"` before
# `run` is the fix if a native backtrace is ever wanted again; it is not wanted
# here, because the per-batch counters name the leak without one.
# =============================================================================

set -o pipefail
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGDIR="${GATE_LOGDIR:-${GITHUB_WORKSPACE:-$PWD}/gate-logs}"
SCENE="${GATE_SCENE:-materials_and_lighting}"
PYTHON="$REPO_ROOT/.venv/bin/python"
REPRO="$REPO_ROOT/benchmarks/_mps_batch_budget_repro.py"

STATUS="INCOMPLETE"
LEAK_RC=""
FIXED_RC=""
KILLER="unknown"
LEAK_PEAK=""
FIXED_PEAK=""
T0=$SECONDS
mkdir -p "$LOGDIR"

stamp() { printf '[+%02d:%02d]' $(( (SECONDS - T0) / 60 )) $(( (SECONDS - T0) % 60 )); }
say() { echo "$(stamp) $*"; }
rule() { echo; echo "=========================================================="; say "$*"; echo "=========================================================="; }

# Highest `mps_alloc=` any batch line in a log reported, which is the one
# number that says whether the render kept giving memory back.
peak_alloc() {
  grep -oE 'mps_alloc=[0-9.]+' "$1" 2>/dev/null \
    | grep -oE '[0-9.]+' | sort -g | tail -1
}

report() {
  rule "MPS LEAK DIAGNOSIS -- scene $SCENE"
  echo "leak arm  (cache pinned for the job) : exit ${LEAK_RC:-not-run}, peak mps_alloc ${LEAK_PEAK:-?} GB"
  echo "fixed arm (cache released with the memory) : exit ${FIXED_RC:-not-run}, peak mps_alloc ${FIXED_PEAK:-?} GB"
  echo "what ended the leak arm              : $KILLER"
  echo
  echo "GATE-RESULT: gate=mps_crash_diagnose status=$STATUS scene=$SCENE \
leak=${LEAK_RC:--} fixed=${FIXED_RC:--} leak_peak=${LEAK_PEAK:--} \
fixed_peak=${FIXED_PEAK:--} killer=$KILLER total=$(( SECONDS - T0 ))s"
}
trap report EXIT

# --------------------------------------------------------------------------
rule "1. what this box has"
sysctl -n hw.model hw.memsize hw.ncpu 2>/dev/null | sed 's/^/    /'
sysctl vm.swapusage 2>/dev/null | sed 's/^/    /'
"$PYTHON" - <<'PY' 2>&1 | sed 's/^/    /'
import torch

print("torch", torch.__version__, "mps", torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    print(f"recommended_max_memory {torch.mps.recommended_max_memory() / 1e9:.2f} GB")
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
rule "2. the defect reinstated: the import cache never released mid-render"
LEAK_LOG="$LOGDIR/leak-arm.log"
"$PYTHON" "$REPRO" --leak-cache --scene "$SCENE" --ceiling-gb 0 \
  >"$LEAK_LOG" 2>&1
LEAK_RC=$?
LEAK_PEAK="$(peak_alloc "$LEAK_LOG")"
say "leak arm exited $LEAK_RC, peak mps_alloc ${LEAK_PEAK:-?} GB"
grep -E 'batch frames|PEAK-RSS|ARM-DONE' "$LEAK_LOG" | sed 's/^/    /'

# --------------------------------------------------------------------------
rule "3. what the system says ended it"
AFTER="$(mktemp)"
find "$HOME/Library/Logs/DiagnosticReports" /Library/Logs/DiagnosticReports \
  -name '*.ips' 2>/dev/null | sort > "$AFTER"
NEW_REPORTS="$(comm -13 "$BEFORE" "$AFTER")"
if [ -z "$NEW_REPORTS" ]; then
  say "  no crash report -- nothing faulted in a way macOS records"
else
  echo "$NEW_REPORTS" | while IFS= read -r crash; do
    [ -z "$crash" ] && continue
    say "  --- $crash"
    head -c 6000 "$crash" | sed 's/^/      /'
    cp "$crash" "$LOGDIR/" 2>/dev/null || true
  done
fi

# The unified log is where the kernel's memory-status decisions go; they never
# reach the process's stderr, which is why §1.4 had nothing to read. Narrowed
# to this python, because runningboardd narrates every other process's jetsam
# bookkeeping continuously and drowns the one line that would matter.
say "jetsam records naming a python process:"
log show --last 20m --style compact \
  --predicate 'eventMessage CONTAINS[c] "jetsam" AND eventMessage CONTAINS[c] "python"' \
  2>/dev/null | tail -n 20 | sed 's/^/    /' || say "    (log show unavailable)"

say "swap and page state after the arm:"
sysctl vm.swapusage 2>/dev/null | sed 's/^/    /'
vm_stat 2>/dev/null | head -6 | sed 's/^/    /'

if grep -qiE 'killed by signal 9|SIGKILL' "$LEAK_LOG"; then
  KILLER="SIGKILL-kernel-kill"
elif [ "$LEAK_RC" = "133" ]; then
  KILLER="SIGTRAP-trap"
elif [ "$LEAK_RC" = "134" ]; then
  KILLER="SIGABRT-abort"
elif grep -qiE 'out of memory|OutOfRenderMemory' "$LEAK_LOG"; then
  KILLER="python-out-of-memory"
elif [ "$LEAK_RC" = "0" ]; then
  KILLER="none-it-completed"
fi
say "verdict on the leak arm: $KILLER"

# --------------------------------------------------------------------------
rule "4. the engine as it stands"
FIXED_LOG="$LOGDIR/fixed-arm.log"
"$PYTHON" "$REPRO" --scene "$SCENE" --ceiling-gb 0 >"$FIXED_LOG" 2>&1
FIXED_RC=$?
FIXED_PEAK="$(peak_alloc "$FIXED_LOG")"
say "fixed arm exited $FIXED_RC, peak mps_alloc ${FIXED_PEAK:-?} GB"
grep -E 'batch frames|PEAK-RSS|ARM-DONE' "$FIXED_LOG" | sed 's/^/    /'

if [ "$FIXED_RC" = "0" ] && [ "$LEAK_RC" != "0" ]; then
  STATUS="LEAK-CONFIRMED-AND-FIXED"
elif [ "$LEAK_RC" = "0" ] && [ "$FIXED_RC" = "0" ]; then
  STATUS="NO-REPRO--BOTH-ARMS-RENDERED"
elif [ "$FIXED_RC" != "0" ]; then
  STATUS="STILL-DYING--READ-THE-BATCH-LINES"
fi
say "done"
