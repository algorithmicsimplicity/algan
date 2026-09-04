# The Quadrants patches Algan carries

Three patches against **Quadrants v1.3.0** (`ab9a58ab5`), applied in order onto
a pristine checkout of that tag:

    git clone --filter=blob:none https://github.com/Genesis-Embodied-AI/quadrants.git
    cd quadrants && git checkout v1.3.0
    git apply --verbose ../quadrants_patches/000*.patch

They are the Quadrants-side successors to `taichi_patches/`, which remains the
source of truth for the Taichi 1.7.4 fork Algan's Apple GPU path uses today.
Which compiler a process runs is `ALGAN_TAICHI_BACKEND` (`algan/taichi_compat.py`),
and why Quadrants is the base is `taichi_patches/PLAN.md` §6.1 — measured, not
argued: byte-identical pixels, a clean macOS build where Taichi 1.7.4 no longer
builds at all, and one upstream Metal miscompile that Quadrants does not have.

`PORTING-NOTES.md` is the per-hunk record: what ported unchanged, what had to be
rewritten and into what, what was **dropped because Quadrants already fixes it**,
and the ranked list of things to watch on the first real build. Read it before
touching a patch.

| | what it is | Taichi counterpart |
| --- | --- | --- |
| `0001-metal-zero-copy-ndarray.patch` | Imports a torch MPS tensor's `MTLBuffer` as a zero-copy ndarray with a non-zero byte offset, because Algan's arrays are slices of one arena. Without it Metal copies every argument through the host, and the second copy-back reverts what the kernel wrote through the first — an Apple GPU renders a black frame. | `taichi_patches/0001`, ported |
| `0002-metal-codegen-and-diagnostics.patch` | The `ContinueStmt`/`gen_label_` fix (a `continue` under a compile-time gate emits invalid SPIR-V), and diagnostics that name the kernel a Metal pipeline build failed on and print the MSL Metal rejected. | `taichi_patches/0002`, minus its first hunk |
| `0003-pre-volta-cuda.patch` | Lets `qd.init(qd.gpu)` work on pre-Volta CUDA at all: device-scope atomics in place of `.sys`, and a compute-capability gate on the warp-aggregated reduction. Not a port — new, and the reason `PLAN.md` §7.3 calls it Prerequisite 0. | none |

**Two things Taichi needed and Quadrants does not.** Its `common/core.h` already
spells `operator""_f`, so `taichi_patches/0003` has no counterpart here; and it
already carries the MSL narrowing-cast fix (`9542c0004`, #543), which is the
hunk `0002` loses. Both were predicted in `taichi_patches/README.md` and are the
small, concrete form of "the fork's patch set shrinks on the newer base".

## What has been verified, and what has not

Verified: all three apply cleanly and **in sequence** with strict `git apply`
(no fuzz, no 3-way) onto pristine v1.3.0 — 15 files, +480/−8 — and the C++ they
touch passes `clang-format`. 0002 is authored against a tree that already has
0001, exactly as the Taichi pair is, because they share `metal_device.mm`.

**Not verified: none of this has been compiled or run.** Nothing here was
written on a Mac or against a GPU. Two build legs exist to close half of that,
and they have to be two because no single machine can compile all three —
Quadrants forces `QD_WITH_CUDA=OFF` on Apple, so the macOS build never sees a
line of what 0003 changes:

    # Metal (0001, 0002) -- .github/workflows/run_on_mac.yaml, arm mac-cpu
    command: bash scripts/gate/quadrants_macos_build.sh
    env:     GATE_QD_PATCHES=1

    # CUDA (0003) -- same workflow, arm linux-cpu
    command: bash scripts/gate/quadrants_linux_build.sh

Both are compile checks. Neither can tell you that an Apple GPU renders
correctly or that an sm_61 card loads the runtime module: the first needs the
Mac probe (`.github/workflows/mps_probe.yaml`) against a wheel built from these,
and the second needs the maintainer's GTX 1050. Until then, treat every patch
here as unproven and read `PORTING-NOTES.md` §5 for where it is most likely to
be wrong.

## Upstreaming

0003 is written to be upstreamed rather than carried: minimal, no Algan-specific
naming, and a no-op on hardware that works today. Device scope is more correct
*and* faster than system scope, and the capability gate copies a pattern
Quadrants already applies elsewhere. A permanent fork delta for it is a rebase
tax for nothing. 0002's `ContinueStmt` hunk is the next best candidate — it is a
plain bug affecting any Metal or Vulkan user with a `continue` under a static
gate. 0001 is Algan-shaped and will likely stay a fork patch.
