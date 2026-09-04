#!/usr/bin/env bash
# =============================================================================
# Fact-finding gate, step 1 (`taichi_patches/PLAN.md` §6): build **Taichi
# v1.7.4 + everything in `taichi_patches/`** on GitHub's Apple-silicon runner.
#
# This is Track A's arm of the base decision, and its sibling is
# `scripts/gate/quadrants_macos_build.sh` (Track B, stock Quadrants v1.3.0).
# Both run on the *same* runner image through the same harness, print the same
# timing table and the same `GATE-RESULT:` line, so the two can be read side by
# side rather than being two measurements of two different machines.
#
# ---------------------------------------------------------------------------
# THE QUESTION
#
# `.github/workflows/taichi_build.yaml` pins `macos-15`, and that pin is a
# finding rather than a preference: `macos-latest` is macOS 26 carrying Apple
# clang 21, which refuses Taichi 1.7.4 outright --
#
#     taichi/common/core.h:170:27: error: identifier '_f' preceded by whitespace
#     in a literal operator declaration is deprecated
#     [-Werror,-Wdeprecated-literal-operator]
#
# -- because Taichi builds with `-Werror` and that diagnostic did not exist when
# the code was written.  `taichi_patches/0003-literal-operator-whitespace.patch`
# fixes 20 such declarations across `taichi/common/core.h` and
# `taichi/common/types.h`.  The gate's step 1 is not "does 0003 apply" (it does,
# strictly, onto a pristine v1.7.4) but **"is `operator""_f` the only thing
# clang 21 objects to, or merely the first?"**
#
# So this script is built to answer that in ONE run rather than in a series of
# round trips.  Whatever happens, it prints the first 40 compiler errors and a
# frequency table of every distinct `-W...` flag the build log mentions.  A
# failure therefore enumerates the remaining work instead of just reporting
# itself; a success proves 0003 was sufficient and hands over a wheel built on
# a current toolchain.
#
# The `mac-cpu` arm of `run_on_mac.yaml` is `macos-latest` -- see
# `.github/workflows/scripts/resolve_gpu_request.py:30-34` -- which is exactly
# the image `taichi_build.yaml` cannot use.  That is the point: this script
# runs *on the untested toolchain deliberately*, which is why it does not
# duplicate `taichi_build.yaml` (that workflow still exists, still pins
# macos-15, and is still what produces the wheel Algan ships).
#
# ---------------------------------------------------------------------------
# HOW TO LAUNCH IT
#
#   `.github/workflows/run_on_mac.yaml`, arm `mac-cpu`:
#       command: bash scripts/gate/taichi_macos_build.sh
#   The `.github/gpu-run/mac.json` body is printed at the end of a run and
#   repeated in the comment at the bottom of this file.
#
# The harness has already run `uv sync`; nothing here uses `.venv`.  Unlike its
# Quadrants sibling this script *does* need one thing from the repository --
# `taichi_patches/[0-9]*.patch`, which is the whole subject.  It reads them and
# nothing else.
#
# ---------------------------------------------------------------------------
# WHAT IS CARRIED OVER FROM `taichi_build.yaml`, AND WHAT IS NOT
#
# Carried over, because each one cost an attempt to discover (six of the seven
# attempts that workflow took failed on the toolchain, not on Taichi):
#
#   * **`brew install --force-bottle llvm@15`.**  `ti_build` hard-wires
#     `-DCLANG_EXECUTABLE=/opt/homebrew/opt/llvm@15/bin/clang` whether or not
#     that path exists.  Attempt 1 died on it in 100 seconds.  `--force-bottle`
#     because a missing bottle would otherwise become an hours-long source build
#     of LLVM; this way it fails in seconds and says so.  **Watch this step:**
#     homebrew-core's `llvm@15` has no `arm64_sequoia` and no `arm64_tahoe`
#     bottle -- its newest arm64 bottle is `arm64_sonoma` (macOS 14, 272 MB).
#     It poured on macos-15 for `taichi_build.yaml`, so Homebrew's older-bottle
#     fallback demonstrably works there; macOS 26 is one generation further and
#     is the single most likely place for this script to stop.  If it does, that
#     is itself a gate finding: Track A's toolchain has an expiry date.
#   * **`CC=/usr/bin/clang CXX=/usr/bin/clang++`, and the split matters.**
#     Homebrew's clang-15 cannot *link* against a current macOS SDK (attempt 2:
#     `ld: library 'System' not found` while compiling cmake's one-line test
#     program).  `ti_build` reads CC/CXX and then leaves `CMAKE_C_COMPILER`
#     alone, so this takes the C++ build off the 2022 toolchain while leaving
#     `-DCLANG_EXECUTABLE` on clang-15 -- which compiles the runtime to bitcode
#     that LLVM 15 then has to be able to load.  Newer clang would emit bitcode
#     LLVM 15 cannot read, so that half must stay at 15.
#   * **The stale `xcrun_db` repair.**  The image ships an `xcrun` cache clang
#     cannot mmap ("couldn't map cache file ... errno=Value too large to be
#     stored in data type"); on attempt 6 it took `setup.py clean` down with
#     SIGBUS before a file was compiled.
#   * **`TAICHI_CMAKE_ARGS` with CUDA / OpenGL / Vulkan / tests off**, and
#     `python3 build.py --python=native` (`build.py` downloads its own prebuilt
#     LLVM 15, so this does not build LLVM; `--python=native` builds against the
#     interpreter already on PATH rather than fetching one).
#
# Not carried over, deliberately:
#
#   * `sudo xcode-select -s /Applications/Xcode_16.4.app`.  There is no
#     Xcode 16.4 on `macos-latest`.  The developer directory is discovered and
#     re-selected in place, which is what actually rebuilds the cache.
#   * The `macos-15` pin.  Running on the newer image *is the experiment*.
#   * The incremental-rebuild timing and the symbol inspection.
#     `taichi_build.yaml` still takes both and this is not a substitute for it.
#   * `CMAKE_BUILD_PARALLEL_LEVEL` is pinned here (default 3, `GATE_JOBS`)
#     where that workflow leaves it alone: this runner has 3 CPUs and **7 GB**,
#     and CMake's default of cores+2 concurrent clang++ processes each carrying
#     LLVM headers is the plausible OOM.
#
# ---------------------------------------------------------------------------
# READING THE OUTPUT
#
# Every line is stamped `[UTC +mm:ss]`; the build goes to
# `gate-logs/build-cold.log` with a heartbeat once a minute rather than being
# streamed, because the Actions API serves a fixed window at the *end* of a job
# log and a full cold-build log would push the answer out of it.  The
# `GATE-RESULT:` line is the last thing printed.  Set
# `"artifacts": ["gate-logs/**"]` in the request to get the logs back whole.
# =============================================================================

set -o pipefail
set -u

# --- knobs ------------------------------------------------------------------
TI_REPO="${GATE_TI_REPO:-https://github.com/taichi-dev/taichi.git}"
TI_REF="${GATE_TI_REF:-v1.7.4}"
APPLY_PATCHES="${GATE_TI_PATCHES:-1}"     # 0 = stock v1.7.4, the control arm
JOBS="${GATE_JOBS:-3}"
HEARTBEAT="${GATE_HEARTBEAT:-60}"
REPO_ROOT="${GITHUB_WORKSPACE:-$PWD}"
PATCH_DIR="${GATE_PATCH_DIR:-$REPO_ROOT/taichi_patches}"
WORKDIR="${GATE_WORKDIR:-${RUNNER_TEMP:-/tmp}/gate-taichi}"
LOGDIR="${GATE_LOGDIR:-$REPO_ROOT/gate-logs}"
SRC="$WORKDIR/taichi-src"

TAICHI_CMAKE_ARGS="${TAICHI_CMAKE_ARGS:--DTI_WITH_CUDA:BOOL=OFF -DTI_WITH_OPENGL:BOOL=OFF -DTI_WITH_VULKAN:BOOL=OFF -DTI_BUILD_TESTS:BOOL=OFF}"
export TAICHI_CMAKE_ARGS

# --- bookkeeping ------------------------------------------------------------
STATUS="INCOMPLETE"
FAILED_PHASE="startup"
SEC_CLONE=""
SEC_PATCH=""
SEC_BREW=""
SEC_BUILD=""
SEC_SMOKE=""
WHEEL_NAME=""
WHEEL_BYTES=""
PATCHES_APPLIED=""
PATCHED="0"
SMOKE_IMPORT="not-run"
SMOKE_PATCH="not-run"
DISK_START=""

mkdir -p "$LOGDIR" "$WORKDIR"
BUILD_LOG="$LOGDIR/build-cold.log"
BREW_LOG="$LOGDIR/brew-llvm15.log"
PATCH_LOG="$LOGDIR/patches.log"
SMOKE_LOG="$LOGDIR/smoke.log"

fmt_elapsed() { printf '%02d:%02d' $(( $1 / 60 )) $(( $1 % 60 )); }
say() { printf '[%s +%s] %s\n' "$(date -u +%H:%M:%S)" "$(fmt_elapsed "$SECONDS")" "$*"; }
rule() { printf '\n===== %s =====\n' "$*"; }
free_disk() { df -h / 2>/dev/null | awk 'NR==2 {print $4}'; }

# Print stdin, or the given placeholder when stdin was empty. `pipefail` is on
# and `grep | head` reports 141 as often as it reports a real 1, so testing a
# pipeline's status is not a way to ask "did that match anything"; this is.
show_or() { local out; out="$(cat)"; if [ -n "$out" ]; then printf '%s\n' "$out"; else printf '%s\n' "$1"; fi; }

# Every warning flag clang named in a log, one per line, deduplicated of the
# bare `-Werror`. Clang spells a fatal warning `[-Werror,-Wdeprecated-literal-operator]`
# and a non-fatal one `[-Wunused-variable]`, so the bracket group has to be
# taken whole and then split -- matching `\[-W...\]` directly finds only the
# second form, which is precisely the half this gate does not care about.
# Scoped to bracketed groups so an echoed `-Wall -Werror -Wno-...` command line
# does not pollute the count.
werror_flags() {
  grep -oE '\[-W[^]]*\]' "$1" 2>/dev/null | tr -d '[]' | tr ',' '\n' \
    | sed 's/^ *//; s/ *$//' | grep -v '^-Werror$' | grep -v '^$'
}

run_logged() {
  local log="$1"; shift
  local label="$1"; shift
  local t0=$SECONDS last=0 pid lines tailline
  say "$label: started, output -> $log"
  "$@" >"$log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$HEARTBEAT"
    kill -0 "$pid" 2>/dev/null || break
    lines=$(wc -l <"$log" 2>/dev/null | tr -d ' ')
    lines=${lines:-0}
    tailline=$(tail -n 1 "$log" 2>/dev/null | tr -d '\r' | cut -c1-120)
    say "$label: ${lines} lines (+$(( lines - last ))), free $(free_disk) | $tailline"
    last=$lines
  done
  wait "$pid"
  local rc=$?
  say "$label: exited $rc after $(fmt_elapsed $(( SECONDS - t0 )))s"
  return $rc
}

# The reason this script exists in this shape. Taichi builds `-Werror`, so on a
# newer clang the failure is a *list* of diagnostics, not one bug -- and the
# expensive mistake would be to fix them one runner round trip at a time.
# Printed on success too: a clean run that still names ten new warning flags is
# a different fact from one that names none.
diagnose() {
  local log="$1"
  [ -f "$log" ] || { echo "(no log at $log)"; return 0; }
  echo "--- first 40 compiler/linker errors"
  grep -nE '(error|fatal error|Undefined symbols|ld: )' "$log" 2>/dev/null | head -40 \
    | show_or "(no line matched an error pattern)"
  echo
  echo "--- every distinct -W diagnostic in the log (count, flag)"
  echo "    each one is a candidate for a 0004 patch if it appears above as an error"
  werror_flags "$log" | sort | uniq -c | sort -rn | head -60 \
    | show_or "    (none -- clang named no warning flags at all)"
  echo
  echo "--- source files clang complained about (count, file)"
  grep -oE '[^ ]+\.(h|hpp|cpp|mm|cu|inc):[0-9]+:[0-9]+: (error|warning)' "$log" 2>/dev/null \
    | sed -E 's/:[0-9]+:[0-9]+: (error|warning)$//' | sort | uniq -c | sort -rn | head -25 \
    | show_or "    (none)"
  echo
  echo "--- CMake / configure failures"
  grep -nE 'CMake Error|Could NOT find|is not a full path|FATAL_ERROR' "$log" 2>/dev/null | head -20 \
    | show_or "(none matched)"
  echo
  echo "--- last 40 lines"
  tail -n 40 "$log" 2>/dev/null | show_or "(empty)"
}

report() {
  rule "GATE REPORT -- Taichi $TI_REF$( [ "$PATCHED" = 1 ] && echo ' + taichi_patches/' || echo ' (stock)' ) on macOS arm64"
  echo "runner : $(sw_vers -productName 2>/dev/null) $(sw_vers -productVersion 2>/dev/null) $(uname -m), \
$(sysctl -n hw.ncpu 2>/dev/null) cpus, $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024 )) GiB"
  echo "clang  : $(/usr/bin/clang --version 2>/dev/null | head -1)"
  echo "cmake  : $TAICHI_CMAKE_ARGS"
  echo "patches: ${PATCHES_APPLIED:-(none applied)}"
  echo "disk   : free at start $DISK_START, free now $(free_disk)"
  echo
  printf '| %-24s | %8s | %7s |\n' "phase" "seconds" "minutes"
  printf '| %-24s | %8s | %7s |\n' "------------------------" "--------" "-------"
  for kv in "clone:$SEC_CLONE" "apply patches:$SEC_PATCH" "brew llvm@15:$SEC_BREW" \
            "cold build:$SEC_BUILD" "wheel smoke test:$SEC_SMOKE"; do
    local k="${kv%%:*}" v="${kv#*:}"
    if [ -n "$v" ]; then
      printf '| %-24s | %8s | %7s |\n' "$k" "$v" "$(awk -v s="$v" 'BEGIN{printf "%.1f", s/60}')"
    else
      printf '| %-24s | %8s | %7s |\n' "$k" "-" "-"
    fi
  done
  echo
  echo "wheel  : ${WHEEL_NAME:-(none produced)}"
  [ -n "$WHEEL_BYTES" ] && echo "size   : $WHEEL_BYTES bytes ($(awk -v b="$WHEEL_BYTES" 'BEGIN{printf "%.1f", b/1048576}') MiB)"
  echo "smoke  : import taichi=$SMOKE_IMPORT  patched entry points=$SMOKE_PATCH"
  echo
  if [ -d "$SRC/dist" ]; then echo "--- ls -la dist/"; ls -la "$SRC/dist"; fi

  # Unconditional: on a failure this is the enumeration the run exists to
  # produce, and on a success it is the evidence that nothing else is lurking.
  rule "COMPILER DIAGNOSTICS (phase: $FAILED_PHASE)"
  case "$FAILED_PHASE" in
    brew) tail -n 40 "$BREW_LOG" 2>/dev/null ;;
    patch) cat "$PATCH_LOG" 2>/dev/null ;;
    smoke) cat "$SMOKE_LOG" 2>/dev/null ;;
    *) diagnose "$BUILD_LOG" ;;
  esac

  if [ "$STATUS" != "PASS" ]; then
    echo
    echo "--- toolchain at failure"
    xcode-select -p 2>&1 || true
    xcodebuild -version 2>&1 | head -2 || true
    /usr/bin/clang --version 2>&1 | head -1 || true
    "$(brew --prefix 2>/dev/null)/opt/llvm@15/bin/clang" --version 2>&1 | head -1 || true
    xcodebuild -showsdks 2>/dev/null | grep -i macos || true
    df -h / 2>&1 | head -3 || true
  fi

  rule "REQUEST BODY for .github/gpu-run/mac.json"
  cat <<'JSON'
{
  "_comment": "Gate step 1 (taichi_patches/PLAN.md §6): does patch 0003 let Taichi 1.7.4 + taichi_patches/ build on macos-latest (macOS 26 / Apple clang 21)? See scripts/gate/taichi_macos_build.sh.",
  "command": "bash scripts/gate/taichi_macos_build.sh",
  "arms": ["mac-cpu"],
  "env": {},
  "latex": false,
  "timeout_minutes": 90,
  "artifacts": ["gate-logs/**"]
}
JSON
  # The flag list belongs on the result line itself: it is the answer to gate
  # step 1, and the result line is the one thing guaranteed to survive a
  # truncated job log.
  local flags
  flags="$(werror_flags "$BUILD_LOG" | sort -u | tr '\n' ',' | sed 's/,$//')"
  [ -n "$flags" ] || flags="none"
  echo
  echo "GATE-RESULT: gate=taichi_macos_build ref=$TI_REF patched=$PATCHED status=$STATUS \
phase=$FAILED_PHASE clone=${SEC_CLONE:--}s patch=${SEC_PATCH:--}s brew=${SEC_BREW:--}s \
build=${SEC_BUILD:--}s total=${SECONDS}s wheel=${WHEEL_NAME:-none} bytes=${WHEEL_BYTES:-0} \
jobs=$JOBS smoke_import=$SMOKE_IMPORT smoke_patch=$SMOKE_PATCH werror_flags=$flags"
}
trap report EXIT

die() { FAILED_PHASE="$1"; STATUS="FAIL"; say "FAILED in phase '$1': ${2:-}"; exit 1; }

# =============================================================================
rule "0. runner facts"
FAILED_PHASE="runner-facts"
DISK_START="$(free_disk)"
say "starting; workdir=$WORKDIR logdir=$LOGDIR patches=$PATCH_DIR"
sw_vers || true
uname -a || true
sysctl -n machdep.cpu.brand_string 2>/dev/null || true
echo "cores: $(sysctl -n hw.ncpu 2>/dev/null)  memory: $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024 )) GiB"
xcodebuild -version 2>/dev/null || true
/usr/bin/clang --version 2>/dev/null | head -1 || true
cmake --version 2>/dev/null | head -1 || true
echo "Xcodes present: $(ls /Applications 2>/dev/null | grep -i '^Xcode' | tr '\n' ' ')"
xcodebuild -showsdks 2>/dev/null | grep -i macos || true
echo "python3: $(command -v python3) -> $(python3 -V 2>&1)"
df -h / || true

# -----------------------------------------------------------------------------
rule "1. repair the xcrun cache"
FAILED_PHASE="xcrun"
rm -f "${TMPDIR:-/tmp}/xcrun_db" || true
_devdir="$(xcode-select -p 2>/dev/null || true)"
if [ -n "$_devdir" ]; then
  case "$_devdir" in
    */Contents/Developer) sudo xcode-select -s "${_devdir%/Contents/Developer}" 2>/dev/null || true ;;
    *) sudo xcode-select -s "$_devdir" 2>/dev/null || true ;;
  esac
fi
say "developer dir: $(xcode-select -p 2>&1)"
say "xcrun clang:   $(xcrun --find clang 2>&1)"
xcrun clang --version 2>&1 | head -1 || true

# -----------------------------------------------------------------------------
rule "2. clone Taichi @ $TI_REF"
FAILED_PHASE="clone"
rm -rf "$SRC"
_t=$SECONDS
git clone --depth 1 --branch "$TI_REF" --recurse-submodules --shallow-submodules \
  "$TI_REPO" "$SRC" || die clone "git clone failed"
SEC_CLONE=$(( SECONDS - _t ))
cd "$SRC" || die clone "cannot cd into $SRC"
say "HEAD: $(git log -1 --format='%H %ci %s')"
say "checkout size: $(du -sh "$SRC" 2>/dev/null | cut -f1)   free: $(free_disk)"

# -----------------------------------------------------------------------------
rule "3. apply taichi_patches/"
FAILED_PHASE="patch"
_t=$SECONDS
if [ "$APPLY_PATCHES" = "1" ]; then
  # `git apply` is deliberately strict here for the same reason taichi_build.yaml
  # keeps it strict -- no fuzz, no 3-way merge -- so a patch that has drifted
  # from the tag fails loudly rather than half applying and producing a wheel
  # whose behaviour nobody can account for.
  # A newline list rather than an array: macOS's /bin/bash is 3.2, where
  # `"${arr[@]}"` on an empty array is an unbound-variable error under `set -u`.
  # A here-string feeds the loop in *this* shell, so `die` can still exit.
  patch_list="$(ls -1 "$PATCH_DIR"/[0-9]*.patch 2>/dev/null || true)"
  if [ -z "$patch_list" ]; then
    die patch "no patches found in $PATCH_DIR -- refusing to build a wheel that would look patched and not be. \
Set GATE_TI_PATCHES=0 to build stock v1.7.4 on purpose."
  fi
  say "patches to apply, in order:"
  printf '  %s\n' $patch_list
  : >"$PATCH_LOG"
  while IFS= read -r patch; do
    [ -n "$patch" ] || continue
    say "applying $(basename "$patch")"
    { echo "=== $(basename "$patch")"; git apply --verbose "$patch"; } >>"$PATCH_LOG" 2>&1 \
      || { cat "$PATCH_LOG"; die patch "git apply failed on $(basename "$patch") -- \
strict apply, no fuzz and no 3-way merge, so this means the patch has drifted from $TI_REF"; }
    PATCHES_APPLIED="${PATCHES_APPLIED}$(basename "$patch") "
  done <<EOF
$patch_list
EOF
  PATCHED=1
  cat "$PATCH_LOG"
  echo "--- resulting diffstat"
  git -c core.pager=cat diff --stat
  # 0003 is the whole subject of gate step 1, so check it landed rather than
  # trusting the exit status of a loop.
  if grep -qn 'operator"" _f' taichi/common/core.h 2>/dev/null; then
    say "WARNING: taichi/common/core.h still contains \`operator\"\" _f\` after patching"
  else
    say "core.h: no space-separated literal operators remain (0003 landed)"
  fi
  say "types.h space-separated literal operators remaining: \
$(grep -c 'operator"" ' taichi/common/types.h 2>/dev/null | head -1)"
else
  say "GATE_TI_PATCHES=0: building STOCK v1.7.4. Expect the -Wdeprecated-literal-operator \
failure on clang >= 21; this arm exists to prove the failure is real on this image."
  PATCHED=0
  PATCHES_APPLIED="(none, stock build)"
fi
SEC_PATCH=$(( SECONDS - _t ))

# -----------------------------------------------------------------------------
rule "4. brew install llvm@15 (the compiler ti_build hard-wires)"
FAILED_PHASE="brew"
_t=$SECONDS
export HOMEBREW_NO_INSTALL_CLEANUP=1 HOMEBREW_NO_ENV_HINTS=1
run_logged "$BREW_LOG" "brew llvm@15" brew install --force-bottle llvm@15 || {
  SEC_BREW=$(( SECONDS - _t ))
  die brew "brew install --force-bottle llvm@15 failed. homebrew-core's llvm@15 has no \
arm64_sequoia/arm64_tahoe bottle (newest arm64 bottle is arm64_sonoma, macOS 14), so this is the \
step most likely to stop on macos-latest. If it says 'no bottle available', Track A's toolchain \
has aged out of this runner image and that is a gate finding in its own right."
}
SEC_BREW=$(( SECONDS - _t ))
BREW_PREFIX="$(brew --prefix)"
BREW_CLANG="$BREW_PREFIX/opt/llvm@15/bin/clang"
[ -x "$BREW_CLANG" ] || die brew "$BREW_CLANG is missing after a successful brew install"
say "CLANG_EXECUTABLE (ti_build hard-wires this path): $BREW_CLANG"
"$BREW_CLANG" --version | head -2
say "free: $(free_disk)"

# -----------------------------------------------------------------------------
rule "5. cold build: python3 build.py --python=native"
FAILED_PHASE="build"
# The CC/CXX split, and it is the point -- see the header comment.
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
say "CC=$CC CXX=$CXX"
say "TAICHI_CMAKE_ARGS=$TAICHI_CMAKE_ARGS"
say "CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"
_t=$SECONDS
run_logged "$BUILD_LOG" "cold build" python3 build.py --python=native || {
  SEC_BUILD=$(( SECONDS - _t ))
  die build "python3 build.py --python=native failed after $(fmt_elapsed "$SEC_BUILD") -- \
the diagnostics section below is the deliverable"
}
SEC_BUILD=$(( SECONDS - _t ))
say "build log: $(wc -l <"$BUILD_LOG" | tr -d ' ') lines, $(du -h "$BUILD_LOG" | cut -f1)"

rule "6. the wheel"
FAILED_PHASE="wheel"
ls -la "$SRC/dist" || die wheel "no dist/ directory"
WHEEL_PATH="$(ls -1 "$SRC"/dist/*.whl 2>/dev/null | head -1)"
[ -n "$WHEEL_PATH" ] || die wheel "build.py exited 0 but produced no wheel"
WHEEL_NAME="$(basename "$WHEEL_PATH")"
WHEEL_BYTES="$(stat -f%z "$WHEEL_PATH" 2>/dev/null || stat -c%s "$WHEEL_PATH")"
say "wheel: $WHEEL_NAME  ($WHEEL_BYTES bytes)"
say "taichi 1.7.4 on PyPI is 50.4 MB for macOS; the no-Vulkan build measured 37 MB, for comparison"

rule "7. wheel smoke test"
FAILED_PHASE="smoke"
_t=$SECONDS
# A throwaway venv, so the build interpreter is left as the build left it. This
# does not touch the GPU: `taichi_build.yaml` deliberately runs nothing on it
# (`benchmarks/_mps_capability_probe.py` is what asks the GPU questions), and
# the same restraint applies here.
VENV="$WORKDIR/venv"
rm -rf "$VENV"
{
  python3 -m venv "$VENV" \
    && "$VENV/bin/python" -m pip install -q -U pip \
    && "$VENV/bin/python" -m pip install -q "$WHEEL_PATH"
} >"$SMOKE_LOG" 2>&1 || die smoke "could not install the wheel into a fresh venv"
if "$VENV/bin/python" -c 'import taichi as ti; print("taichi", ti.__version__)' >>"$SMOKE_LOG" 2>&1; then
  SMOKE_IMPORT="ok"
else
  SMOKE_IMPORT="FAILED"
fi
# Whether the wheel that came out is the patched one -- the difference between
# "the build succeeded" and "the build succeeded and carries the thing it was
# built for". A patch that applied and then got compiled out by an #ifdef would
# look identical without this.
if [ "$PATCHED" = "1" ]; then
  if "$VENV/bin/python" -c 'from taichi.lang._ndarray import ExternalMetalNdarray; print("ExternalMetalNdarray present")' \
       >>"$SMOKE_LOG" 2>&1; then SMOKE_PATCH="ok"; else SMOKE_PATCH="FAILED"; fi
  so="$(find "$SRC" -name 'taichi_python*.so' 2>/dev/null | head -1)"
  if [ -n "$so" ]; then
    echo "--- symbols in $so"
    echo "create_ndarray_from_metal_buffer: $(nm "$so" 2>/dev/null | grep -c 'create_ndarray_from_metal_buffer')"
    echo "import_external_mtl_buffer:       $(nm "$so" 2>/dev/null | grep -c 'import_external_mtl_buffer')"
  fi
else
  SMOKE_PATCH="n/a (stock build)"
fi
SEC_SMOKE=$(( SECONDS - _t ))
tail -n 25 "$SMOKE_LOG" || true
[ "$SMOKE_IMPORT" = "ok" ] || die smoke "import taichi failed on the freshly built wheel"

STATUS="PASS"
FAILED_PHASE="none"
say "done"
exit 0

# =============================================================================
# The request body, repeated here so it can be copied without running anything:
#
# {
#   "_comment": "Gate step 1 (taichi_patches/PLAN.md §6): does patch 0003 let Taichi 1.7.4 + taichi_patches/ build on macos-latest (macOS 26 / Apple clang 21)? See scripts/gate/taichi_macos_build.sh.",
#   "command": "bash scripts/gate/taichi_macos_build.sh",
#   "arms": ["mac-cpu"],
#   "env": {},
#   "latex": false,
#   "timeout_minutes": 90,
#   "artifacts": ["gate-logs/**"]
# }
#
# `mac-cpu`, not `mac-mps`: nothing here runs Algan or touches the GPU, and the
# arm only decides the runner image and `ALGAN_RENDER_DEVICE` (unused).
# `mac-cpu` also skips the harness's patched-Taichi-wheel download, which is
# pure cost -- and faintly absurd -- for a job that builds that wheel.
#
# Expected: ~1 min clone, seconds to patch, ~1-3 min for brew, then a cold
# build.  `taichi_build.yaml` measures ~15 min end to end on `macos-15` with the
# same CMake args; this runs at `-j3` rather than CMake's default, so 15-30
# minutes for the build is the range.  A failure will usually arrive in the
# first two minutes of compiling, since `common/core.h` and `common/types.h`
# are included nearly everywhere -- so an early exit is the *informative*
# outcome, not a wasted run.
#
# TWO USEFUL VARIANTS, both one dispatch away:
#   env {"GATE_TI_PATCHES": "0"}  -- stock v1.7.4 on this image.  Run this if
#       the patched build fails, to tell "0003 was insufficient" apart from
#       "something else about macos-latest is broken"; and run it once anyway
#       to have the unpatched diagnostic list to diff against.
#   env {"GATE_JOBS": "2"}        -- if the build dies without a compiler error
#       (a killed process, a truncated log), suspect memory before suspecting
#       Taichi.
# =============================================================================
