#!/usr/bin/env bash
# =============================================================================
# Fact-finding gate, step 2 (`taichi_patches/PLAN.md` §6): build **stock
# Quadrants v1.3.0** on GitHub's Apple-silicon runner and report what it cost.
#
# The question this answers is not "does Algan work on Quadrants" (that is gate
# step 3, `scripts/gate/backend_pixel_ab.py`). It is narrower and it is the one
# that gates Track B's *release channel*: **is the toolchain story as clean as
# `quadrants-src/.github/workflows/scripts_new/macosx/{1_prerequisites,2_build}.sh`
# imply** -- two files totalling 34 lines, which say `brew install llvm@22`,
# `git submodule update --init --recursive`, `./build.py wheel`. Algan's own
# Taichi 1.7.4 build took seven attempts to get a wheel out of the same class of
# runner, six of which failed on the toolchain rather than on Taichi
# (`.github/workflows/taichi_build.yaml`, and `DESIGN_mps_zero_copy.md` §4).
#
# Its sibling is `scripts/gate/taichi_macos_build.sh`, which builds Track A's
# arm -- Taichi v1.7.4 plus `taichi_patches/` -- on the *same* runner image so
# the two numbers can be read side by side.
#
# ---------------------------------------------------------------------------
# HOW TO LAUNCH IT
#
#   `.github/workflows/run_on_mac.yaml`, arm `mac-cpu`:
#       command: bash scripts/gate/quadrants_macos_build.sh
#   The exact `.github/gpu-run/mac.json` body is printed at the end of a run,
#   and is also reproduced in the comment at the bottom of this file.
#
# The harness has already run `uv sync` and created `.venv` by the time this
# starts; nothing here uses either. It shells out to the `python3` on PATH
# (`actions/setup-python` 3.11), builds in `$RUNNER_TEMP`, and touches no repo
# state at all -- this script is self-contained on purpose so that it can be
# run from any branch, and so that a failure is a fact about Quadrants and the
# runner rather than about Algan's tree.
#
# ---------------------------------------------------------------------------
# WHAT IT DOES DIFFERENTLY FROM QUADRANTS' OWN CI, AND WHY
#
# Their `macosx.yml` builds on `macos-26` with Python 3.10-3.13 and
# `fetch-depth: 0`; this runs on whatever `macos-latest` currently is (macOS 26
# as of 2026-09, per `resolve_gpu_request.py`'s `mac-cpu` arm) with one Python.
# Four deliberate deviations, each behind a knob so a second run can undo it:
#
#   1. `QD_WITH_VULKAN=OFF` (their `2_build.sh` sets `ON`).  `GATE_QD_VULKAN=1`
#      restores it.  Vulkan on macOS costs the LunarG SDK: `build.py` calls
#      `setup_vulkan()` **unconditionally** on Darwin
#      (`.github/workflows/scripts/qd_build/entry.py:65-75`), downloads
#      `vulkansdk-macos-1.4.321.0.zip`, extracts the `InstallVulkan.app` bundle
#      and drives its Qt installer headlessly (`qd_build/vulkan.py:41-78`) --
#      several GB on a runner with ~14 GB free.  It is also what a Track B
#      Algan fork would turn off anyway (`PLAN.md` §7.3 item 5).  With Vulkan
#      off, `setup_vulkan()` still runs, so this script pre-seeds its cache
#      directories to make `download_dep`'s "already there" early-return fire
#      (`qd_build/dep.py:97-98`) rather than patching their script.
#   2. `QD_BUILD_TESTS=OFF` (their `2_build.sh` sets `ON`).  `GATE_QD_TESTS=1`
#      restores it.  45 test TUs plus googletest, for a binary this gate never
#      runs; Algan's fork would not ship it.
#   3. `--depth 1` clone and `--depth 1` submodules.  Their CI takes full
#      history because `setuptools_scm` derives the wheel version from
#      `git describe`; a `--branch v1.3.0 --depth 1` clone still has the tag, so
#      the version comes out identical.  The script checks that and says so.
#   4. `CMAKE_BUILD_PARALLEL_LEVEL` is pinned (default 3, `GATE_JOBS`).  This is
#      the RAM guard.  CMake's default is cores+2 = 5 concurrent clang++
#      processes, each carrying LLVM 22 + Eigen headers, on a **7 GB** box.
#      That, not the link step, is the plausible OOM here: LLVM itself is a
#      prebuilt static archive (see below), so nothing in this build ever links
#      LLVM from objects.
#
# ---------------------------------------------------------------------------
# WHERE THE COMPILERS COME FROM (the thing the gate is actually asking about)
#
# Quadrants uses **two** LLVMs on macOS and they are not interchangeable:
#
#   * `LLVM_DIR` <- a ~989 MB prebuilt archive of LLVM 22.1.0 hosted in the org
#     repo `Genesis-Embodied-AI/quadrants-sdk-builds`, release tag
#     `llvm-22.1.0-202603120808` (`qd_build/llvm.py:22-42`).  `build.py`
#     downloads it; CMake consumes it via `find_package(LLVM CONFIG)`
#     (`cmake/QuadrantsCore.cmake:118-130`).  This is what the compiler links
#     against.  Nothing builds LLVM.
#   * `CLANG_EXECUTABLE` <- `$(brew --prefix)/opt/llvm@22/bin/clang`, hard-wired
#     by `qd_build/compiler.py:44-50`, which is why `1_prerequisites.sh` says
#     `brew install llvm@22`.  It compiles `runtime.cpp` to the `.bc` the JIT
#     loads, so its major version must match the LLVM above.  Exactly the role
#     Homebrew's `llvm@15` plays for Taichi 1.7.4 -- and exactly the path whose
#     absence killed the first run of `taichi_build.yaml` in 100 seconds.
#
# `llvm@22` is a real homebrew-core formula (LLVM 22.1.8, `keg_only`) with
# `arm64_tahoe` / `arm64_sequoia` / `arm64_sonoma` bottles: 331 MB down,
# 1.57 GB installed.  `--force-bottle` is carried over from `taichi_build.yaml`
# for the same reason it is there: if the bottle is ever missing for the
# runner's macOS, brew would build LLVM from source and eat hours, and this way
# it fails in seconds and says so.
#
# On macOS `setup_clang` is called with `as_compiler=False`
# (`qd_build/entry.py:59`), so `CMAKE_C_COMPILER`/`CMAKE_CXX_COMPILER` are left
# alone and the C++ build uses Xcode's clang.  That is the same CC/CXX split
# `taichi_build.yaml` had to construct by hand after Homebrew clang-15 failed to
# link against a current SDK ("ld: library 'System' not found"); Quadrants gets
# it structurally, and passes `-isysroot` to the brew clang for the bitcode
# compile (`CMakeLists.txt:220-226`).  So this script does **not** set CC/CXX.
#
# The stale-`xcrun_db` repair is carried over from `taichi_build.yaml`
# unchanged in spirit: that cache made clang die with "couldn't map cache file
# into memory (errno=Value too large...)" and took `setup.py clean` down with
# SIGBUS before a file was compiled.  It is cheap and it is not specific to
# Taichi, so it runs here too -- but the Xcode path is discovered rather than
# hard-coded, because `macos-latest` has no `Xcode_16.4.app`.
#
# ---------------------------------------------------------------------------
# READING THE OUTPUT
#
# Every line is stamped `[UTC +mm:ss]`, and the build itself is *not* streamed:
# it goes to `gate-logs/build-cold.log` while a heartbeat prints one progress
# line a minute.  Two reasons.  The Actions API serves a fixed window at the end
# of a job log whatever tail length is asked for (`taichi_build.yaml`'s "Report"
# step exists because of this), so anything that must be readable from outside
# has to be near the end; and a 40 MB cold-build log would push the answer out
# of that window.  The final `GATE-RESULT:` line is deliberately the last thing
# printed, with the timings, the wheel and the failing phase all on it.
#
# Set `"artifacts": ["gate-logs/**"]` in the request to get the full logs back.
# =============================================================================

set -o pipefail
set -u

# --- knobs ------------------------------------------------------------------
QD_REPO="${GATE_QD_REPO:-https://github.com/Genesis-Embodied-AI/quadrants.git}"
QD_REF="${GATE_QD_REF:-v1.3.0}"
WITH_VULKAN="${GATE_QD_VULKAN:-0}"
WITH_TESTS="${GATE_QD_TESTS:-0}"
JOBS="${GATE_JOBS:-3}"
HEARTBEAT="${GATE_HEARTBEAT:-60}"
WORKDIR="${GATE_WORKDIR:-${RUNNER_TEMP:-/tmp}/gate-quadrants}"
LOGDIR="${GATE_LOGDIR:-${GITHUB_WORKSPACE:-$PWD}/gate-logs}"
SRC="$WORKDIR/quadrants-src"

# --- bookkeeping ------------------------------------------------------------
STATUS="INCOMPLETE"
FAILED_PHASE="startup"
SEC_CLONE=""
SEC_SUBMODULES=""
SEC_PIP=""
SEC_BREW=""
SEC_BUILD=""
SEC_SMOKE=""
WHEEL_NAME=""
WHEEL_BYTES=""
QD_VERSION=""
SMOKE_CPU="not-run"
SMOKE_METAL="not-run"
DISK_START=""
DISK_END=""

mkdir -p "$LOGDIR" "$WORKDIR"
BUILD_LOG="$LOGDIR/build-cold.log"
SUBMOD_LOG="$LOGDIR/submodules.log"
PIP_LOG="$LOGDIR/pip.log"
BREW_LOG="$LOGDIR/brew-llvm22.log"
SMOKE_LOG="$LOGDIR/smoke.log"

fmt_elapsed() { printf '%02d:%02d' $(( $1 / 60 )) $(( $1 % 60 )); }
say() { printf '[%s +%s] %s\n' "$(date -u +%H:%M:%S)" "$(fmt_elapsed "$SECONDS")" "$*"; }
rule() { printf '\n===== %s =====\n' "$*"; }
free_disk() { df -h / 2>/dev/null | awk 'NR==2 {print $4}'; }

# Print stdin, or the given placeholder when stdin was empty. `pipefail` is on
# and `grep | head` reports 141 as often as it reports a real 1, so testing a
# pipeline's status is not a way to ask "did that match anything"; this is.
show_or() { local out; out="$(cat)"; if [ -n "$out" ]; then printf '%s\n' "$out"; else printf '%s\n' "$1"; fi; }

# Every warning flag clang named in a log, one per line, minus the bare
# `-Werror`. Clang spells a fatal warning `[-Werror,-Wsome-flag]` and a
# non-fatal one `[-Wsome-flag]`, so the bracket group has to be taken whole and
# then split; matching `\[-W...\]` directly finds only the second form.
# Quadrants also builds `-Werror` under clang
# (`cmake/QuadrantsCXXFlags.cmake:79-82`), so this is as relevant here.
werror_flags() {
  grep -oE '\[-W[^]]*\]' "$1" 2>/dev/null | tr -d '[]' | tr ',' '\n' \
    | sed 's/^ *//; s/ *$//' | grep -v '^-Werror$' | grep -v '^$'
}

# Run a long command with its output in a file and a heartbeat on stdout, so a
# truncated job log still shows where it got to.  Returns the command's status.
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

# One pass over a build log that turns it into the few facts worth reading.
# `-Werror` is on for clang in Quadrants too (`cmake/QuadrantsCXXFlags.cmake:79-82`),
# so enumerating the distinct diagnostics is what tells a "one bad warning"
# failure apart from a real port.
diagnose() {
  local log="$1"
  [ -f "$log" ] || { echo "(no log at $log)"; return 0; }
  echo "--- first 40 compiler/linker errors"
  grep -nE '(error|fatal error|Undefined symbols|ld: )' "$log" 2>/dev/null | head -40 | show_or "(no line matched an error pattern)"
  echo "--- distinct -W diagnostics (count, flag)"
  werror_flags "$log" | sort | uniq -c | sort -rn | head -40 | show_or "(clang named no warning flags)"
  echo "--- CMake / configure failures"
  grep -nE 'CMake Error|Could NOT find|FATAL_ERROR|is not a full path' "$log" 2>/dev/null | head -20 | show_or "(none)"
  echo "--- last 40 lines"
  tail -n 40 "$log" 2>/dev/null | show_or "(empty)"
}

report() {
  rule "GATE REPORT -- stock Quadrants $QD_REF on macOS arm64"
  echo "runner : $(sw_vers -productName 2>/dev/null) $(sw_vers -productVersion 2>/dev/null) $(uname -m), \
$(sysctl -n hw.ncpu 2>/dev/null) cpus, $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024 )) GiB"
  echo "config : QD_WITH_VULKAN=$( [ "$WITH_VULKAN" = 1 ] && echo ON || echo OFF ) \
QD_WITH_METAL=ON QD_BUILD_TESTS=$( [ "$WITH_TESTS" = 1 ] && echo ON || echo OFF ) jobs=$JOBS"
  echo "disk   : free at start $DISK_START, free at end $(free_disk) (was $DISK_END at build end)"
  echo
  printf '| %-24s | %8s | %7s |\n' "phase" "seconds" "minutes"
  printf '| %-24s | %8s | %7s |\n' "------------------------" "--------" "-------"
  for kv in "clone:$SEC_CLONE" "submodules:$SEC_SUBMODULES" "pip deps:$SEC_PIP" \
            "brew llvm@22:$SEC_BREW" "cold build:$SEC_BUILD" "wheel smoke test:$SEC_SMOKE"; do
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
  echo "version: ${QD_VERSION:-(unknown)}"
  echo "smoke  : qd.init(cpu)=$SMOKE_CPU  qd.init(metal)=$SMOKE_METAL"
  echo
  if [ -d "$SRC/dist" ]; then echo "--- ls -la dist/"; ls -la "$SRC/dist"; fi
  if [ "$STATUS" != "PASS" ]; then
    rule "DIAGNOSTICS (phase: $FAILED_PHASE)"
    case "$FAILED_PHASE" in
      build) diagnose "$BUILD_LOG" ;;
      pip) tail -n 40 "$PIP_LOG" 2>/dev/null ;;
      brew) tail -n 40 "$BREW_LOG" 2>/dev/null ;;
      submodules) tail -n 40 "$SUBMOD_LOG" 2>/dev/null ;;
      smoke) cat "$SMOKE_LOG" 2>/dev/null ;;
      *) : ;;
    esac
    echo
    echo "--- toolchain at failure"
    xcode-select -p 2>&1 || true
    /usr/bin/clang --version 2>&1 | head -1 || true
    "$(brew --prefix 2>/dev/null)/opt/llvm@22/bin/clang" --version 2>&1 | head -1 || true
    cmake --version 2>&1 | head -1 || true
    df -h / 2>&1 | head -3 || true
  else
    rule "BUILD LOG: distinct -W diagnostics that were NOT fatal"
    werror_flags "$BUILD_LOG" | sort | uniq -c | sort -rn | head -20 \
      | show_or "(none -- clang named no warning flags at all)"
  fi

  rule "REQUEST BODY for .github/gpu-run/mac.json"
  cat <<'JSON'
{
  "_comment": "Gate step 2 (taichi_patches/PLAN.md §6): build stock Quadrants v1.3.0 on the Apple-silicon runner. See scripts/gate/quadrants_macos_build.sh.",
  "command": "bash scripts/gate/quadrants_macos_build.sh",
  "arms": ["mac-cpu"],
  "env": {},
  "latex": false,
  "timeout_minutes": 120,
  "artifacts": ["gate-logs/**"]
}
JSON
  # Same field as the Taichi sibling prints, so the two result lines can be
  # diffed rather than read.
  local flags
  flags="$(werror_flags "$BUILD_LOG" | sort -u | tr '\n' ',' | sed 's/,$//')"
  [ -n "$flags" ] || flags="none"
  echo
  echo "GATE-RESULT: gate=quadrants_macos_build ref=$QD_REF status=$STATUS phase=$FAILED_PHASE \
clone=${SEC_CLONE:--}s submodules=${SEC_SUBMODULES:--}s pip=${SEC_PIP:--}s brew=${SEC_BREW:--}s \
build=${SEC_BUILD:--}s total=${SECONDS}s wheel=${WHEEL_NAME:-none} bytes=${WHEEL_BYTES:-0} \
vulkan=$( [ "$WITH_VULKAN" = 1 ] && echo on || echo off ) tests=$( [ "$WITH_TESTS" = 1 ] && echo on || echo off ) \
jobs=$JOBS smoke_cpu=$SMOKE_CPU smoke_metal=$SMOKE_METAL werror_flags=$flags"
}
trap report EXIT

die() { FAILED_PHASE="$1"; STATUS="FAIL"; say "FAILED in phase '$1': ${2:-}"; exit 1; }

# =============================================================================
rule "0. runner facts"
FAILED_PHASE="runner-facts"
DISK_START="$(free_disk)"
say "starting; workdir=$WORKDIR logdir=$LOGDIR"
sw_vers || true
uname -a || true
sysctl -n machdep.cpu.brand_string 2>/dev/null || true
echo "cores: $(sysctl -n hw.ncpu 2>/dev/null)  memory: $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024 )) GiB"
xcodebuild -version 2>/dev/null || true
/usr/bin/clang --version 2>/dev/null | head -1 || true
cmake --version 2>/dev/null | head -1 || echo "cmake: not on PATH yet (pip installs one)"
echo "Xcodes present: $(ls /Applications 2>/dev/null | grep -i '^Xcode' | tr '\n' ' ')"
echo "python3: $(command -v python3) -> $(python3 -V 2>&1)"
echo "brew:    $(command -v brew) -> $(brew --prefix 2>/dev/null)"
df -h / || true

# -----------------------------------------------------------------------------
rule "1. repair the xcrun cache"
# `taichi_build.yaml`'s sixth-attempt fix. The image ships a stale `xcrun_db`
# that clang cannot mmap; removing it and re-selecting the developer directory
# rebuilds it. The developer dir is *discovered* here -- macos-latest has no
# Xcode_16.4.app, so the hard-coded path in taichi_build.yaml does not apply.
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
rule "2. clone Quadrants @ $QD_REF"
FAILED_PHASE="clone"
rm -rf "$SRC"
_t=$SECONDS
git clone --depth 1 --branch "$QD_REF" "$QD_REPO" "$SRC" || die clone "git clone failed"
SEC_CLONE=$(( SECONDS - _t ))
cd "$SRC" || die clone "cannot cd into $SRC"
say "HEAD: $(git log -1 --format='%H %ci %s')"
# setuptools_scm derives the wheel version from `git describe`; Quadrants' own
# macosx.yml uses fetch-depth: 0 for exactly this. A --depth 1 clone *of a tag*
# still describes, but say so out loud rather than shipping a wheel called 0.1.
if git describe --tags >/dev/null 2>&1; then
  say "git describe --tags: $(git describe --tags)  (setuptools_scm will be happy)"
else
  say "WARNING: git describe failed on a shallow clone; fetching the tag explicitly"
  git fetch --depth 1 origin "refs/tags/$QD_REF:refs/tags/$QD_REF" || true
  say "git describe --tags: $(git describe --tags 2>&1)"
fi

rule "3. submodules"
FAILED_PHASE="submodules"
_t=$SECONDS
# 16 gitlinks at v1.3.0; SPIRV-Tools, SPIRV-Cross (the Genesis fork), eigen and
# Vulkan-Headers are most of the bytes. `--depth 1` where their CI takes full
# history -- ~157 MB of working tree instead of considerably more.
run_logged "$SUBMOD_LOG" "submodules" \
  git submodule update --init --recursive --depth 1 --jobs "$JOBS" \
  || die submodules "git submodule update failed"
SEC_SUBMODULES=$(( SECONDS - _t ))
say "checkout size: $(du -sh "$SRC" 2>/dev/null | cut -f1)   free: $(free_disk)"

# -----------------------------------------------------------------------------
rule "4. prerequisites (their 1_prerequisites.sh, minus the vestigial bits)"
FAILED_PHASE="pip"
_t=$SECONDS
# Their script runs, in order: pip install -U pip; pip install --group dev;
# pip install numpy; brew install llvm@22; submodules; brew install pybind11.
# `brew install pybind11` is skipped: Quadrants migrated to nanobind in #759
# (`685feb23b`) and nothing under cmake/ references pybind11 any more.
{
  set -x
  python3 -m pip install -U pip
  set +x
} >"$PIP_LOG" 2>&1 || die pip "pip self-upgrade failed"
say "pip: $(python3 -m pip --version)"
# `--group` needs pip >= 25.1. If this pip is older or the flag is refused,
# fall back to the explicit list rather than dying: the only thing the group is
# needed for is `--no-build-isolation` in their build_wheel (entry.py:47).
if python3 -m pip install --group dev >>"$PIP_LOG" 2>&1; then
  say "pip install --group dev: OK"
else
  say "pip install --group dev FAILED (pip too old, or group unsupported); using the explicit list"
  python3 -m pip install \
    "scikit-build-core>=0.10" "nanobind>=2.0.0,<2.14.0" "numpy>=2.0.0" "setuptools_scm>=6.0" \
    "setuptools>=77.0.0" "cmake<4.0.0" ninja wheel build psutil requests tqdm mslex \
    >>"$PIP_LOG" 2>&1 || die pip "explicit build-dependency install failed"
fi
python3 -m pip install numpy >>"$PIP_LOG" 2>&1 || die pip "numpy install failed"
SEC_PIP=$(( SECONDS - _t ))
say "cmake: $(cmake --version 2>&1 | head -1)   ninja: $(ninja --version 2>&1 | head -1)"
say "nanobind: $(python3 -c 'import nanobind;print(nanobind.__version__)' 2>&1 | tail -1)"

rule "5. brew install llvm@22 (this is CLANG_EXECUTABLE, not the linked LLVM)"
FAILED_PHASE="brew"
_t=$SECONDS
export HOMEBREW_NO_INSTALL_CLEANUP=1 HOMEBREW_NO_ENV_HINTS=1
# --force-bottle, for the reason taichi_build.yaml gives: without it a missing
# bottle silently turns into an hours-long source build of LLVM.
run_logged "$BREW_LOG" "brew llvm@22" brew install --force-bottle llvm@22 \
  || die brew "brew install --force-bottle llvm@22 failed -- see $BREW_LOG. \
If this says 'no bottle available', homebrew-core has no llvm@22 bottle for this macOS and \
the whole build story changes."
SEC_BREW=$(( SECONDS - _t ))
BREW_PREFIX="$(brew --prefix)"
BREW_CLANG="$BREW_PREFIX/opt/llvm@22/bin/clang"
[ -x "$BREW_CLANG" ] || die brew "$BREW_CLANG is missing after a successful brew install"
say "CLANG_EXECUTABLE will be: $BREW_CLANG"
"$BREW_CLANG" --version | head -2
say "keg size: $(du -sh "$BREW_PREFIX/opt/llvm@22/" 2>/dev/null | cut -f1)   free: $(free_disk)"

# -----------------------------------------------------------------------------
rule "6. build configuration"
FAILED_PHASE="configure"
CMAKE_ARGS="-DQD_WITH_METAL:BOOL=ON"
if [ "$WITH_VULKAN" = "1" ]; then
  CMAKE_ARGS="$CMAKE_ARGS -DQD_WITH_VULKAN:BOOL=ON"
  say "Vulkan ON: build.py will download and run the LunarG macOS SDK installer (GBs)"
else
  CMAKE_ARGS="$CMAKE_ARGS -DQD_WITH_VULKAN:BOOL=OFF"
  # `setup_basic_build_env` calls setup_vulkan() on Darwin no matter what the
  # CMake flag says (entry.py:65-75). Pre-seed the two cache directories it
  # writes so `download_dep`'s "outdir exists and is non-empty" early return
  # (dep.py:97-98) fires and its `if not (prefix/"macOS").exists()` guard
  # (vulkan.py:52) skips the installer. Nothing consumes VULKAN_SDK when
  # QD_WITH_VULKAN is OFF -- MoltenVK is only looked for inside
  # `if (QD_WITH_VULKAN)` (quadrants/rhi/CMakeLists.txt:28-68).
  _vkcache="$HOME/.cache/qd-build-cache"
  mkdir -p "$_vkcache/vulkan-macos-1.4.321.0-installer" "$_vkcache/vulkan-macos-1.4.321.0/macOS/lib"
  : >"$_vkcache/vulkan-macos-1.4.321.0-installer/.gate-stub"
  : >"$_vkcache/vulkan-macos-1.4.321.0/macOS/.gate-stub"
  say "Vulkan OFF: LunarG SDK download stubbed out at $_vkcache/vulkan-macos-1.4.321.0*"
fi
if [ "$WITH_TESTS" = "1" ]; then
  CMAKE_ARGS="$CMAKE_ARGS -DQD_BUILD_TESTS:BOOL=ON"
else
  CMAKE_ARGS="$CMAKE_ARGS -DQD_BUILD_TESTS:BOOL=OFF"
fi
export CMAKE_ARGS
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"
# Deliberately NOT setting CC/CXX. Quadrants calls setup_clang(as_compiler=False)
# on macOS, so the C++ build already uses Xcode's clang while CLANG_EXECUTABLE
# stays on brew's llvm@22 -- the split taichi_build.yaml had to build by hand.
say "CMAKE_ARGS=$CMAKE_ARGS"
say "CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL  MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET"

rule "7. cold build: ./build.py wheel"
FAILED_PHASE="build"
_t=$SECONDS
run_logged "$BUILD_LOG" "cold build" python3 build.py wheel || {
  SEC_BUILD=$(( SECONDS - _t ))
  DISK_END="$(free_disk)"
  die build "python3 build.py wheel failed after $(fmt_elapsed "$SEC_BUILD")"
}
SEC_BUILD=$(( SECONDS - _t ))
DISK_END="$(free_disk)"
say "build log: $(wc -l <"$BUILD_LOG" | tr -d ' ') lines, $(du -h "$BUILD_LOG" | cut -f1)"

rule "8. the wheel"
FAILED_PHASE="wheel"
ls -la "$SRC/dist" || die wheel "no dist/ directory"
WHEEL_PATH="$(ls -1 "$SRC"/dist/quadrants-*.whl 2>/dev/null | head -1)"
[ -n "$WHEEL_PATH" ] || die wheel "build.py exited 0 but produced no wheel"
WHEEL_NAME="$(basename "$WHEEL_PATH")"
WHEEL_BYTES="$(stat -f%z "$WHEEL_PATH" 2>/dev/null || stat -c%s "$WHEEL_PATH")"
QD_VERSION="${WHEEL_NAME#quadrants-}"
QD_VERSION="${QD_VERSION%%-*}"
say "wheel: $WHEEL_NAME  ($WHEEL_BYTES bytes)"
# The tag is stamped by `qd_build/entry.py:36-54`; if the version came out as
# 0.1 the shallow clone lost the tag and setuptools_scm guessed.
case "$QD_VERSION" in
  0.1*) say "WARNING: wheel version is $QD_VERSION -- setuptools_scm did not see the tag" ;;
  *)    say "wheel version: $QD_VERSION (expected ${QD_REF#v})" ;;
esac
say "PyPI quadrants 1.3.0 macosx_13_0_arm64 is 26.7 MB, for comparison"
echo "--- 15 largest members"
python3 - "$WHEEL_PATH" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
for i in sorted(z.infolist(), key=lambda i: -i.file_size)[:15]:
    print(f"  {i.file_size/1048576:8.2f} MiB  {i.filename}")
PY

rule "9. wheel smoke test (their 3_install.sh)"
FAILED_PHASE="smoke"
_t=$SECONDS
# In a throwaway venv, so the build interpreter is left as the build left it.
VENV="$WORKDIR/venv"
rm -rf "$VENV"
{
  python3 -m venv "$VENV" \
    && "$VENV/bin/python" -m pip install -q -U pip \
    && "$VENV/bin/python" -m pip install -q "$WHEEL_PATH"
} >"$SMOKE_LOG" 2>&1 || die smoke "could not install the wheel into a fresh venv"
if "$VENV/bin/python" -c 'import quadrants as qd; qd.init(arch=qd.cpu); print("cpu ok", qd.__version__)' \
     >>"$SMOKE_LOG" 2>&1; then SMOKE_CPU="ok"; else SMOKE_CPU="FAILED"; fi
# The Mac runner's GPU is real hardware even though the instance is virtualized
# (agent_guidance/gpu_harnesses.md), so metal is a fair question here. It is
# reported, not enforced: this gate is about the build.
if "$VENV/bin/python" -c 'import quadrants as qd; qd.init(arch=qd.metal); print("metal ok")' \
     >>"$SMOKE_LOG" 2>&1; then SMOKE_METAL="ok"; else SMOKE_METAL="FAILED"; fi
SEC_SMOKE=$(( SECONDS - _t ))
tail -n 25 "$SMOKE_LOG" || true
[ "$SMOKE_CPU" = "ok" ] || die smoke "qd.init(arch=qd.cpu) failed on the freshly built wheel"

STATUS="PASS"
FAILED_PHASE="none"
say "done"
exit 0

# =============================================================================
# The request body, repeated here so it can be copied without running anything:
#
# {
#   "_comment": "Gate step 2 (taichi_patches/PLAN.md §6): build stock Quadrants v1.3.0 on the Apple-silicon runner. See scripts/gate/quadrants_macos_build.sh.",
#   "command": "bash scripts/gate/quadrants_macos_build.sh",
#   "arms": ["mac-cpu"],
#   "env": {},
#   "latex": false,
#   "timeout_minutes": 120,
#   "artifacts": ["gate-logs/**"]
# }
#
# `mac-cpu`, not `mac-mps`: nothing here runs Algan, so the arm only decides the
# runner image and `ALGAN_RENDER_DEVICE` (unused).  `mac-cpu` also skips the
# harness's patched-Taichi-wheel download, which is pure cost for this job.
#
# 120 minutes is not the expected time, it is the ceiling.  Expected: ~5 min of
# clone + pip + brew, then a cold build of ~700 translation units (261 under
# `quadrants/`, 426 in SPIRV-Tools, 12 in SPIRV-Cross) at `-j3` on three
# virtualized M1 cores.  Taichi 1.7.4's comparable macOS build measures ~15 min
# end to end on this class of runner; Quadrants has more TUs, LLVM 22 headers
# rather than LLVM 15, and one fewer core than that reading assumed, so 25-45
# minutes for the build alone is the honest range.  Add ~10-15 min if
# `GATE_QD_VULKAN=1`.  If it has not produced a wheel by 90 minutes, something
# is wrong rather than slow.
# =============================================================================
