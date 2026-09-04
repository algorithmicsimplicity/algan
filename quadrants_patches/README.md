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

**They apply, and they compile.** All three apply cleanly and **in sequence**
with strict `git apply` (no fuzz, no 3-way) onto pristine v1.3.0 — 15 files,
+480/−8 — and 0002 is authored against a tree that already has 0001, exactly as
the Taichi pair is, because they share `metal_device.mm`.

Compilation takes two legs, and it has to be two because no single machine can
build all three: Quadrants forces `QD_WITH_CUDA=OFF` on Apple, so the macOS
build never sees a line of what 0003 changes. Both passed on 2026-09-04, from
`.github/workflows/run_on_mac.yaml` at `d295007`:

| | | result |
| --- | --- | --- |
| **Metal** (0001, 0002) | `bash scripts/gate/quadrants_macos_build.sh` with `GATE_QD_PATCHES=1`, arm `mac-cpu` | PASS, 781 s build, `quadrants-1.3.1.dev0+gab9a58ab5-cp311-cp311-macosx_13_0_arm64.whl` (22.3 MiB), **`qd.init(metal)=ok`**, and clang named **no warning flags at all** |
| **CUDA** (0003) | `bash scripts/gate/quadrants_linux_build.sh`, arm `linux-cpu` | PASS, 1041 s build, `...-manylinux_2_27_x86_64.whl` (26.4 MiB), `qd.init(cpu)=ok`, `runtime_cuda.bc present -- CUDA backend compiled` |

That last field is load-bearing and was got wrong once. The first Linux run
"passed" while proving nothing: the check grepped the build log, and
`grep -c … || echo 0` yields `"0\n0"` on no match, so the guard could never
fire. It now asks the wheel instead — `_lib/runtime/runtime_cuda.bc` exists if
and only if the build had CUDA on — and dies when it is absent. (Not
`qd._lib.core.with_cuda()`: that also probes for `libcuda.so`, so it is False on
every GPU-less runner however the binary was built.)

**Still unverified: that any of it works.** A compile check cannot tell you that
an Apple GPU renders correctly or that an sm_61 card loads the runtime module.
The first needs `.github/workflows/mps_probe.yaml` against a wheel built from
these; the second needs the maintainer's GTX 1050, and `PORTING-NOTES.md` §7
lists exactly what to look for there (`atom.gpu.cas.b64` and no remaining
`atom.sys`). Read `PORTING-NOTES.md` §5 for where the Metal port is most likely
to be wrong — the ranked list starts with the nanobind integer default and the
`LaunchContextBufferCache` interaction.

## Upstreaming

0003 is written to be upstreamed rather than carried: minimal, no Algan-specific
naming, and a no-op on hardware that works today. Device scope is more correct
*and* faster than system scope, and the capability gate copies a pattern
Quadrants already applies elsewhere. A permanent fork delta for it is a rebase
tax for nothing. 0002's `ContinueStmt` hunk is the next best candidate — it is a
plain bug affecting any Metal or Vulkan user with a `continue` under a static
gate. 0001 is Algan-shaped and will likely stay a fork patch.
