#!/usr/bin/env bash
# =============================================================================
# Build Quadrants + `quadrants_patches/` on Linux with **CUDA on**.
#
# This exists for one reason: `quadrants_patches/0003-pre-volta-cuda.patch`
# cannot be compiled anywhere else this project can reach. Quadrants forces
# `QD_WITH_CUDA=OFF` on Apple (`cmake/QuadrantsCore.cmake:31-35`), so the macOS
# gate build -- the one that covers the Metal patches -- never compiles a line
# of `codegen_cuda.cpp` or the LLVM runtime module the pre-Volta patch changes.
# A GitHub Linux runner has no GPU, but it does not need one: `QD_WITH_CUDA`
# defaults to ON with `QD_WITH_CUDA_TOOLKIT=OFF`, so the CUDA backend is built
# from LLVM's NVPTX target and talks to the driver API loaded at runtime.
#
# It applies the whole directory in numeric order (`[0-9]*.patch`), so since
# `0004-llvm-invariant-load-kernel-args.patch` joined it this build compiles
# that patch's `codegen_llvm.cpp` changes too. What proves 0004 *worked* is
# `.github/workflows/quadrants_build.yaml`, which builds the same set and then
# runs `quadrants_patches/verify_invariant_load.py` over the optimized IR.
#
# So this is a **compile check, not a behaviour check**. It answers "do the
# patched files still build", which is the half that can be automated. The half
# that cannot is whether sm_61 now loads the runtime module and runs a kernel;
# that needs the maintainer's GTX 1050, and `PLAN.md` §7.3 Prerequisite 0 is
# what describes it.
#
#   `.github/workflows/run_on_mac.yaml`, arm `linux-cpu`:
#       command: bash scripts/gate/quadrants_linux_build.sh
#
# Follows Quadrants' own `scripts_new/linux/{1_prerequisites,2_build}.sh`, with
# three deviations, each behind a knob: Vulkan and AMDGPU off (their CI builds
# both; Algan wants neither, and each costs TUs and dependencies), tests off,
# and `quadrants_patches/` applied. Unlike the macOS script this one applies the
# patches by DEFAULT -- there, the question was whether stock builds at all;
# here, stock is already known to build in their CI and the patches are the
# whole subject. `GATE_QD_PATCHES=0` gives the stock control arm.
#
# clang comes from the LLVM archive itself (`python download_llvm.py`, then
# `$LLVM_DIR/bin` on PATH), not from a distro package -- so unlike the macOS
# build there is no second Homebrew LLVM to install and no version to keep in
# step by hand.
#
# Output is stamped and the build is logged to a file with a heartbeat rather
# than streamed, for the reason the macOS script explains: the Actions API
# serves a window at the END of a job log, and a large build log pushes the
# answer out of it. The last line is always `GATE-RESULT:`.
# =============================================================================

set -o pipefail
set -u

QD_REPO="${GATE_QD_REPO:-https://github.com/Genesis-Embodied-AI/quadrants.git}"
QD_REF="${GATE_QD_REF:-v1.3.0}"
APPLY_PATCHES="${GATE_QD_PATCHES:-1}"
WITH_VULKAN="${GATE_QD_VULKAN:-0}"
WITH_AMDGPU="${GATE_QD_AMDGPU:-0}"
WITH_TESTS="${GATE_QD_TESTS:-0}"
JOBS="${GATE_JOBS:-$(nproc 2>/dev/null || echo 2)}"
HEARTBEAT="${GATE_HEARTBEAT:-60}"
WORKDIR="${GATE_WORKDIR:-${RUNNER_TEMP:-/tmp}/gate-quadrants-linux}"
LOGDIR="${GATE_LOGDIR:-${GITHUB_WORKSPACE:-$PWD}/gate-logs}"
PATCH_DIR="${GATE_QD_PATCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/quadrants_patches}"
SRC="$WORKDIR/quadrants-src"
BUILD_LOG="$LOGDIR/linux-build-cold.log"
PATCH_LOG="$LOGDIR/linux-patches.log"

STATUS="INCOMPLETE"
FAILED_PHASE="startup"
PATCHED=0
PATCHES_APPLIED=""
SEC_CLONE=""
SEC_SUBMODULES=""
SEC_PATCH=""
SEC_PREREQ=""
SEC_BUILD=""
WHEEL_NAME=""
WHEEL_BYTES=""
SMOKE_CPU="not-run"
CUDA_TUS="not-checked"
T0=$SECONDS

mkdir -p "$LOGDIR"

stamp() { printf '[UTC +%02d:%02d]' $(( (SECONDS - T0) / 60 )) $(( (SECONDS - T0) % 60 )); }
say() { echo "$(stamp) $*"; }
rule() { echo; echo "=============================================================="; say "$*"; echo "=============================================================="; }
free_disk() { df -h . 2>/dev/null | awk 'NR==2{print $4}'; }
show_or() { local out; out="$(cat)"; if [ -n "$out" ]; then echo "$out"; else echo "$1"; fi; }

die() {
  FAILED_PHASE="$1"; shift
  STATUS="FAIL"
  say "FATAL ($FAILED_PHASE): $*"
  exit 1
}

# Run a long command into a log while printing one line a minute, so a build
# that is progressing looks different from one that has hung.
run_logged() {
  local log="$1" label="$2"; shift 2
  say "$label: logging to $log"
  "$@" >"$log" 2>&1 &
  local pid=$! t0=$SECONDS last=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 5
    if [ $(( SECONDS - last )) -ge "$HEARTBEAT" ]; then
      last=$SECONDS
      say "  $label: $(wc -l <"$log" 2>/dev/null || echo 0) lines, $(tail -n 1 "$log" 2>/dev/null | cut -c1-110)"
    fi
  done
  wait "$pid"
  local rc=$?
  say "$label: exited $rc after $(( SECONDS - t0 ))s"
  return $rc
}

report() {
  rule "GATE REPORT -- Quadrants $QD_REF$( [ "$PATCHED" = 1 ] && echo ' + quadrants_patches/' || echo ' (stock)' ) on Linux, CUDA ON"
  echo "runner : $(uname -srm), $(nproc 2>/dev/null) cpus, $(free -g 2>/dev/null | awk 'NR==2{print $2}') GiB, free disk $(free_disk)"
  echo "config : QD_WITH_CUDA=ON QD_WITH_VULKAN=$( [ "$WITH_VULKAN" = 1 ] && echo ON || echo OFF ) \
QD_WITH_AMDGPU=$( [ "$WITH_AMDGPU" = 1 ] && echo ON || echo OFF ) QD_BUILD_TESTS=$( [ "$WITH_TESTS" = 1 ] && echo ON || echo OFF ) jobs=$JOBS"
  echo "patches: ${PATCHES_APPLIED:-(none applied)}"
  echo "wheel  : ${WHEEL_NAME:-(none produced)}"
  echo "smoke  : qd.init(cpu)=$SMOKE_CPU"
  echo "cuda   : $CUDA_TUS"
  echo
  printf '| %-22s | %8s |\n' "phase" "seconds"
  for kv in "clone:$SEC_CLONE" "submodules:$SEC_SUBMODULES" "apply patches:$SEC_PATCH" \
            "prerequisites:$SEC_PREREQ" "cold build:$SEC_BUILD"; do
    printf '| %-22s | %8s |\n' "${kv%%:*}" "${kv#*:}"
  done
  if [ "$STATUS" != "PASS" ]; then
    rule "DIAGNOSTICS (phase: $FAILED_PHASE)"
    echo "--- first 40 errors"
    grep -nE '(error|fatal error|Undefined symbols|ld: )' "$BUILD_LOG" 2>/dev/null | head -40 | show_or "(no line matched an error pattern)"
    echo "--- distinct -W diagnostics"
    grep -oE '\[-W[^]]+\]' "$BUILD_LOG" 2>/dev/null | tr -d '[]' | tr ',' '\n' | grep -v '^-Werror$' \
      | sort | uniq -c | sort -rn | head -20 | show_or "(clang named no warning flags)"
    echo "--- last 40 lines"
    tail -n 40 "$BUILD_LOG" 2>/dev/null | show_or "(empty)"
  fi
  echo
  echo "GATE-RESULT: gate=quadrants_linux_build ref=$QD_REF patched=$PATCHED status=$STATUS \
phase=$FAILED_PHASE clone=${SEC_CLONE:--}s submodules=${SEC_SUBMODULES:--}s patch=${SEC_PATCH:--}s \
prereq=${SEC_PREREQ:--}s build=${SEC_BUILD:--}s total=$(( SECONDS - T0 ))s wheel=${WHEEL_NAME:-none} \
bytes=${WHEEL_BYTES:-0} smoke_cpu=$SMOKE_CPU cuda=$CUDA_TUS"
}
trap report EXIT

# -----------------------------------------------------------------------------
rule "0. runner facts"
uname -a
python3 -V
cmake --version 2>/dev/null | head -1 || echo "(no cmake yet)"
df -h . | tail -1

rule "1. clone Quadrants @ $QD_REF"
FAILED_PHASE="clone"
_t=$SECONDS
rm -rf "$SRC"
mkdir -p "$WORKDIR"
git clone --depth 1 --branch "$QD_REF" "$QD_REPO" "$SRC" || die clone "git clone failed"
SEC_CLONE=$(( SECONDS - _t ))
cd "$SRC" || die clone "cannot cd into $SRC"
say "HEAD: $(git log -1 --format='%H %ci %s')"

rule "2. submodules"
FAILED_PHASE="submodules"
_t=$SECONDS
run_logged "$LOGDIR/linux-submodules.log" "submodules" \
  git submodule update --init --recursive --depth 1 --jobs "$JOBS" \
  || die submodules "git submodule update failed"
SEC_SUBMODULES=$(( SECONDS - _t ))

rule "3. quadrants_patches/"
FAILED_PHASE="patch"
_t=$SECONDS
if [ "$APPLY_PATCHES" = "1" ]; then
  # Strict: no fuzz, no 3-way. A patch that has drifted from the tag must fail
  # here rather than half-applying into a wheel nobody can account for.
  patch_list="$(ls -1 "$PATCH_DIR"/[0-9]*.patch 2>/dev/null || true)"
  [ -n "$patch_list" ] || die patch "no patches in $PATCH_DIR (set GATE_QD_PATCHES=0 for the stock control)"
  : >"$PATCH_LOG"
  for patch in $patch_list; do
    say "applying $(basename "$patch")"
    { echo "=== $(basename "$patch")"; git apply --verbose "$patch"; } >>"$PATCH_LOG" 2>&1 \
      || { cat "$PATCH_LOG"; die patch "git apply failed on $(basename "$patch") -- strict apply, so it has drifted from $QD_REF"; }
    PATCHES_APPLIED="${PATCHES_APPLIED}$(basename "$patch") "
  done
  PATCHED=1
  cat "$PATCH_LOG"
  git -c core.pager=cat diff --stat
else
  say "GATE_QD_PATCHES=0: stock $QD_REF (the control arm)"
fi
SEC_PATCH=$(( SECONDS - _t ))

rule "4. prerequisites (their linux/1_prerequisites.sh)"
FAILED_PHASE="prereq"
_t=$SECONDS
sudo apt-get update -qq || true
sudo apt-get install -y -qq cmake ninja-build || die prereq "apt install failed"
python3 -m pip install -q -U pip || die prereq "pip self-upgrade failed"
# `--group dev` needs pip >= 25.1; fall back to the explicit list if this pip
# is older, rather than failing on a syntax the runner's pip does not know.
python3 -m pip install -q --group dev || python3 -m pip install -q \
  setuptools wheel scikit-build-core nanobind numpy || die prereq "dev deps failed"
LLVM_DIR="$(python3 download_llvm.py | tail -n 1)" || die prereq "download_llvm.py failed"
export PATH="$LLVM_DIR/bin:$PATH"
chmod +x "$LLVM_DIR"/bin/* 2>/dev/null || true
say "LLVM_DIR=$LLVM_DIR"
clang --version | head -1 || die prereq "clang from the LLVM archive is not runnable"
SEC_PREREQ=$(( SECONDS - _t ))
say "free disk after prerequisites: $(free_disk)"

rule "5. cold build (CUDA ON)"
FAILED_PHASE="build"
_t=$SECONDS
export CMAKE_ARGS="-DQD_WITH_CUDA:BOOL=ON \
-DQD_WITH_VULKAN:BOOL=$( [ "$WITH_VULKAN" = 1 ] && echo ON || echo OFF ) \
-DQD_WITH_AMDGPU:BOOL=$( [ "$WITH_AMDGPU" = 1 ] && echo ON || echo OFF ) \
-DQD_BUILD_TESTS:BOOL=$( [ "$WITH_TESTS" = 1 ] && echo ON || echo OFF )"
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
say "CMAKE_ARGS=$CMAKE_ARGS"
run_logged "$BUILD_LOG" "build.py wheel" python3 ./build.py wheel || die build "./build.py wheel failed"
SEC_BUILD=$(( SECONDS - _t ))

rule "6. the wheel, and what CUDA it compiled"
FAILED_PHASE="wheel"
wheel="$(ls -1 dist/*.whl 2>/dev/null | head -1)"
[ -n "$wheel" ] || die wheel "build reported success but produced no wheel in dist/"
WHEEL_NAME="$(basename "$wheel")"
WHEEL_BYTES="$(stat -c%s "$wheel" 2>/dev/null || echo 0)"
ls -la dist/
# The point of this leg: prove the CUDA backend -- the code 0003 patches -- was
# actually compiled. A build with CUDA silently off would otherwise pass and
# mean nothing, and the first version of this check was itself broken: it
# grepped the build log, and `grep -c ... || echo 0` yields "0\n0" on no match
# (grep -c prints its own 0 *and* exits 1), so the guard below could never fire.
#
# The wheel is the honest place to ask. `_lib/runtime/runtime_cuda.bc` is the
# runtime module compiled for the NVPTX target: it exists if and only if the
# build had CUDA on, it is the module whose `.sys` atomic defect (a) is about,
# and unlike a log line its name does not depend on the generator's output
# format. `qd._lib.core.with_cuda()` is NOT the check to use -- it also probes
# for libcuda.so, so it is False on every GPU-less runner regardless of how the
# binary was built.
CUDA_BC="$(unzip -l "$wheel" 2>/dev/null | grep -c 'runtime_cuda\.bc' | head -1)"
CUDA_BC="${CUDA_BC:-0}"
if [ "$CUDA_BC" -ge 1 ] 2>/dev/null; then
  CUDA_TUS="runtime_cuda.bc present -- CUDA backend compiled"
  say "$CUDA_TUS"
else
  CUDA_TUS="runtime_cuda.bc ABSENT -- CUDA was off, 0003 is NOT verified by this run"
  say "$CUDA_TUS"
  unzip -l "$wheel" 2>/dev/null | grep -E '_lib/runtime/' | head -10
  die wheel "the wheel carries no CUDA runtime module, so this build did not \
compile the code 0003 changes. Check that CMAKE_ARGS reached build.py."
fi

rule "7. smoke test"
FAILED_PHASE="smoke"
venv="$WORKDIR/smoke-venv"
python3 -m venv "$venv" >/dev/null 2>&1 || die smoke "cannot create a venv"
"$venv/bin/pip" install -q "$wheel" || die smoke "pip install of the built wheel failed"
# CPU only: this runner has no GPU, so `qd.init(arch=qd.cuda)` is expected to
# fail here and its failure would say nothing about the patch.
if "$venv/bin/python" -c "import quadrants as qd; qd.init(arch=qd.cpu); print('ok', qd.__version__)"; then
  SMOKE_CPU="ok"
else
  SMOKE_CPU="FAILED"
  die smoke "the built wheel does not import and init on cpu"
fi

STATUS="PASS"
FAILED_PHASE="none"
say "done"
