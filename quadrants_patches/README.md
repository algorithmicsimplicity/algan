# The Quadrants patches Algan carries

Seven patches against **Quadrants v1.3.0** (`ab9a58ab5`, 2026-08-11 — the latest
public release; `v1.3.0b1`/`b2` are earlier betas despite sorting above it),
applied **in numeric order** onto a pristine checkout of that tag:

    git clone --filter=blob:none https://github.com/Genesis-Embodied-AI/quadrants.git
    cd quadrants && git checkout v1.3.0
    git apply --verbose ../quadrants_patches/[0-9]*.patch

They are the Quadrants-side successors to `taichi_patches/`, which remains the
source of truth for the Taichi 1.7.4 fork Algan's Apple GPU path uses today.
Which compiler a process runs is `ALGAN_TAICHI_BACKEND` (`algan/taichi_compat.py`),
and why Quadrants is the base is `taichi_patches/PLAN.md` §6.1 — measured, not
argued: byte-identical pixels, a clean macOS build where Taichi 1.7.4 no longer
builds at all, and one upstream Metal miscompile that Quadrants does not have.

`../taichi_patches/MIGRATION.md` is the account of the migration these came out
of — what was measured, what the measurements corrected, and what is still not
verified. `PORTING-NOTES.md` is the per-hunk record: what ported unchanged, what had to be
rewritten and into what, what was **dropped because Quadrants already fixes it**,
and the ranked list of things to watch on the first real build. Read it before
touching a patch.

| | what it is | Taichi counterpart |
| --- | --- | --- |
| `0001-metal-zero-copy-ndarray.patch` | Imports a torch MPS tensor's `MTLBuffer` as a zero-copy ndarray with a non-zero byte offset, because Algan's arrays are slices of one arena. Without it Metal copies every argument through the host, and the second copy-back reverts what the kernel wrote through the first — an Apple GPU renders a black frame. | `taichi_patches/0001`, ported |
| `0002-metal-codegen-and-diagnostics.patch` | The `ContinueStmt`/`gen_label_` fix (a `continue` under a compile-time gate emits invalid SPIR-V), and diagnostics that name the kernel a Metal pipeline build failed on and print the MSL Metal rejected. | `taichi_patches/0002`, minus its first hunk |
| `0003-pre-volta-cuda.patch` | Lets `qd.init(qd.gpu)` work on pre-Volta CUDA at all: device-scope atomics in place of `.sys`, and a compute-capability gate on the warp-aggregated reduction. Not a port — new, and the reason `../taichi_patches/PLAN.md` §7.3 calls it Prerequisite 0. | none |
| `0004-llvm-invariant-load-kernel-args.patch` | `!invariant.load` (plus `!dereferenceable`) on the loads that read a kernel's argument buffer, so LICM can hoist an ndarray's base pointer and shape dims out of the loop instead of re-reading them at every use site. Gated by a new `invariant_arg_loads` compile-config field. Not a port — new, and `../taichi_patches/PLAN.md` row 13. | none |
| `0005-cuda-max-reg.patch` | `qd.loop_config(max_reg=N)` carried through the frontend IR, `lower_ast`, `offload` and the CUDA codegen to a **per-kernel** PTX `.maxnreg`, and `qd.init(gpu_max_reg=N)` made real by finally passing it to `add_module` (`CU_JIT_MAX_REGISTERS` at module load). Not a port — new, and PLAN row 14. | none |
| `0006-cuda-readonly-ndarray-ldg.patch` | A `readonly_ndarray_ldg` compile-config flag under which loads from ndarray arguments the offloaded task never writes go through `ld.global.nc`, the read-only-cache path read-only SNodes already take. Not a port — new, and PLAN row 15. | none |
| `0007-cuda-fast-expf.patch` | Under `fast_math`, f32 `qd.exp` becomes `__nv_fast_expf` instead of `__nv_expf` — the branch shape `log`, `sin` and `cos` already have. Not a port — new, and PLAN row 18. | none |

0005-0007 are CUDA codegen and were written in one sitting, after 0004 and
**against a tree that did not have 0003 applied** (their `codegen_cuda.cpp` and
`llvm_context.cpp` pre-image blob ids are pristine v1.3.0's). That costs
nothing at apply time — every shared hunk is hundreds of lines from every
other — but see "Applying them".

**Two things Taichi needed and Quadrants does not.** Its `common/core.h` already
spells `operator""_f`, so `taichi_patches/0003` has no counterpart here; and it
already carries the MSL narrowing-cast fix (`9542c0004`, #543), which is the
hunk `0002` loses. Both were predicted in `taichi_patches/README.md` and are the
small, concrete form of "the fork's patch set shrinks on the newer base".

## Applying them

**Numeric order, all seven, strict `git apply`** — no fuzz, no 3-way. That is
what `scripts/gate/quadrants_linux_build.sh`,
`scripts/gate/quadrants_macos_build.sh` and
`.github/workflows/quadrants_build.yaml` all do: each globs
`quadrants_patches/[0-9]*.patch` and applies the sorted list, so nothing in the
tree applies a subset and nothing needs a hand-maintained list of names.

Measured against a pristine `v1.3.0` (`ab9a58ab5`) checkout: **0001-0004** apply
cleanly in that order, **19 files, +560/−13**. Adding 0005-0007's own diffstats
by the same count gives **32 files, +774/−21** for the seven — arithmetic, not a
re-measured diffstat. All seven **do apply, strictly and in order, and compile
with CUDA on**: `quadrants_build.yaml`'s Linux leg did exactly that on
2026-09-04 (run
[`33926192036`](https://github.com/algorithmicsimplicity/algan/actions/runs/33926192036)),
and the 0004 checks still pass on the resulting wheel.

Numeric order is the convention rather than a constraint, and it is worth being
precise about which, because several patches share a file:

| shares | with | authored against |
| --- | --- | --- |
| `rhi/metal/metal_device.h`, `.mm` | 0001 and 0002 | 0002 was written on a tree that already had 0001 |
| `python/export_lang.cpp` | 0001, 0004, 0005, 0006 | four different regions; 0006 was written on a tree with 0004's line in it, 0005 was not |
| `codegen/cuda/codegen_cuda.cpp` | 0003, 0005, 0006, 0007 | 0005 → `:692`, 0006 → `:616`, 0007 → `:356`, 0003 → `optimized_reduction` at `:362-388`; 0005-0007 were chained onto each other but not onto 0003 |
| `runtime/llvm/llvm_context.cpp` | 0003, 0005 | 0003 rewrites a syncscope in `module_from_file`; 0005 edits `mark_function_as_cuda_kernel` at `:1055` |
| `analysis/offline_cache_util.cpp`, `program/compile_config.h`, `codegen/llvm/codegen_llvm.cpp` | 0004 and 0006 (the first two), 0004 and 0005 (the third) | adjacent lines in the same lists; 0006 is written on 0004 |

In every one of them the hunks are far enough apart that `git apply` absorbs the
line offsets, so the order does not decide whether they apply: 0002 applies onto
pristine v1.3.0 on its own, and 0001-then-0002 and 0002-then-0001 produce a
**byte-identical tree** (checked). What the ordering does decide is *cosmetic
noise*: because 0005-0007 were authored without 0003, a full-set apply will
report a few "succeeded at N (offset M lines)" for `codegen_cuda.cpp` and
`llvm_context.cpp`. That is expected and is not drift. It does, however, mean
`git apply --index` or `-3` would refuse them on the blob-id mismatch — another
reason the scripts use plain strict `git apply`.

So numeric order is what to use — it is what every script does and the order the
patches were authored in — but a patch that fails to apply has drifted from the
tag rather than been applied out of order. Fix the patch; do not loosen the
apply. `PORTING-NOTES.md` says what each hunk is anchored on.

## Getting a patched wheel

Nothing here builds on the machine you are reading this on: the Metal patches
need an Apple GPU box and the CUDA ones a toolchain most checkouts do not have.
So the build happens on GitHub's runners and the wheels come back over the API.
One command does both:

    uv run python scripts/build_quadrants_wheels.py

That dispatches `.github/workflows/quadrants_build.yaml`, waits, and downloads
every wheel it produced into `quadrants_wheels/`. Narrower, when only one
platform is in question, or wider, for a release:

    uv run python scripts/build_quadrants_wheels.py --platforms macos
    uv run python scripts/build_quadrants_wheels.py --python 3.10,3.11,3.12,3.13
    uv run python scripts/build_quadrants_wheels.py --run-id <id> --install

**All three platforms Algan supports**, because each is the only place part of
this patch set can be compiled at all — Metal (0001, 0002) only exists on macOS,
where Quadrants forces `QD_WITH_CUDA=OFF`; 0003 is for a pre-Volta card in a
Windows box; 0004 is LLVM codegen that both CUDA legs see. Of the three CUDA
patches, 0005 is mostly frontend-IR and LLVM plumbing every leg compiles (only
its `codegen_cuda.cpp` line and `mark_function_as_cuda_kernel` are CUDA-shaped),
while **0006 and 0007 live entirely in `codegen_cuda.cpp`** — a file the macOS
leg never opens, so only the Linux and Windows legs can fail on them. One wheel per
platform per Python, ~15-20 minutes each, `fail-fast: false` so one platform's
failure still lands the others' wheels.

The wheels themselves are **gitignored** — 20-30 MiB each, and this repository
keeps binaries that size out of git (`tests/README.md`, "Where the heavy
baselines live"). `quadrants_wheels/manifest.json` beside them **is** committed:
run id, commit, and a sha256 per wheel and per patch, so which wheel a
measurement was taken on stays recoverable from git even though the bytes are
not. Attach the wheels to a release to share them; the manifest's digests are
what make an uploaded asset verifiable.

A patched wheel is also self-identifying, without anyone having to arrange it:
`git apply` leaves the patches uncommitted, so `setuptools_scm` sees a dirty
tree and stamps `1.3.1.dev0+gab9a58ab5.d<date>` rather than the `1.3.0` PyPI
ships — the `.d` is the dirty marker, and it is the tell that a wheel came from
here rather than from the index.

**First run of all three legs: PASS** — run
[`33850787142`](https://github.com/algorithmicsimplicity/algan/actions/runs/33850787142),
2026-09-04, cp311, patches applied:

| leg | runner | wheel |
| --- | --- | --- |
| linux | `ubuntu-22.04` | 26.34 MiB, and the 0004 IR arms still land (`invariant-load-arms-py3.11`) |
| macos | `macos-26` | 22.21 MiB, via `scripts/gate/quadrants_macos_build.sh` unchanged |
| windows | `windows-2025` | 26.51 MiB — `quadrants-1.3.1.dev0+gab9a58ab5.d20260904-cp311-cp311-win_amd64.whl`, sha256 `d6db5de8…`. **The first Windows wheel this fork has ever had**, and the only one 0003's sm_61 box can be tried on |

The download half of `build_quadrants_wheels.py` is the one thing not exercised
end to end from a sandbox: an artifact download redirects off `api.github.com`
to blob storage, which some egress policies refuse (the script says so rather
than raising). Everything up to the transfer — dispatch, run lookup by tag,
artifact listing and filtering — ran against this run, and the unzip/hash/
manifest half is covered in `tests/unit_tests/test_quadrants_wheels.py`.

**Running against a wheel on the Mac runner.** `.github/workflows/run_on_mac.yaml`
and `mps_probe.yaml` take a `quadrants_wheel` input: a `quadrants_build.yaml`
run id (its `quadrants-wheel-macos-py3.11` artifact is fetched, every Mac arm
being on 3.11) or a release-asset URL (installed directly). Either installs the
wheel after `uv sync` and pins `ALGAN_TAICHI_BACKEND=quadrants` for the run;
when both a Taichi and a Quadrants wheel are named, the Quadrants one wins.
With it, the pixel A/B and the probe (`scripts/gate/backend_pixel_ab.py
--scenes fast`, `scripts/gate/mps_probe_quadrants.sh`) measure a *given* wheel
rather than one the script builds first.

## What has been verified, and what has not

**0001-0004 apply, and they compile.** "Applying them" above is the apply half.
The gate runs below are the compile half, and they predate 0004: they measured
0001-0003 at 15 files, +492/−8. 0004 adds 5 files and 135 lines and was built
and checked separately, on Linux — see its own section. **0005-0007 are outside
every claim in this section**: they have not been applied, built or run, and
their own section (below, before "Upstreaming") is the record of that.

Compilation takes two legs, and it has to be two because no single machine can
build all of it: Quadrants forces `QD_WITH_CUDA=OFF` on Apple, so the macOS
build never sees a line of what 0003 changes, and no runner here has both an
Apple GPU and an NVIDIA one. Both passed on 2026-09-04, from
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

### The Metal port rendered wrong, why, and the fix

**Measured 2026-09-04 on the Mac runner's real Apple GPU, before the fix below.**
A wheel built from these patches installs, Algan resolves
`device=mps`, `zero_copy_available()` is **True**, and renders complete
(`gpu_smoke.py`: 16.4 s cold, 0.92 s warm). Then the picture is wrong.

The same scene rendered on Metal and on the same machine's CPU, same compiler,
via `scripts/gate/mps_vs_cpu_ab.sh`:

| compiler | pixels over tolerance | max delta | mean brightness, MPS vs CPU |
| --- | --- | --- | --- |
| Taichi 1.7.4 + `taichi_patches/` | 11,527 of 83,913,984 (**0.014 %**) | 221 | 47.62 vs 47.62 |
| Quadrants 1.3.0 + `quadrants_patches/` | 79,913,926 of 83,913,984 (**95 %**) | 255 | **16.41** vs 47.62 |

Taichi's two devices agree: a hundredth of a percent of pixels differ, at
localised edges, which is what MPS-friendly mode's float32 accumulators
predict. Quadrants' do not — almost every pixel differs and the Metal frame is
about **a third as bright**, which is why the control mattered: one number from
one compiler could not have told these apart, and the first reading was nearly
written up as a black frame it is not.

**The cause: the offset was recorded where this backend never reads it.** It is
not element shape, and it is not the kernels. Quadrants' Metal device advertises
a capability Taichi 1.7.4's does not —

    // rhi/metal/metal_device.mm, collect_metal_device_caps
    if (feature_64_bit_integer_math) {          // == family_apple3, so every Apple GPU
      caps.set(DeviceCapability::spirv_has_int64, 1);
      caps.set(DeviceCapability::spirv_has_physical_storage_buffer, 1);   // Quadrants only
    }

— and under that capability an ndarray is **not addressed through its `ExtArr`
descriptor at all**. `TaskCodegen::visit(ExternalPtrStmt)` loads a raw 64-bit GPU
address out of the args buffer's `DATA_PTR` slot and `at_buffer` dereferences it
(`OpConvertUToPtr` + `OpPtrAccessChain`); no `ExtArr` binding is even emitted,
because `buffer_binding_map_` is only populated by the descriptor branch this
path skips. The address is published host-side by `HostDeviceContextBlitter::
host_to_device` as `get_memory_physical_pointer(...)`, which on Metal is
`[mtl_buffer gpuAddress]` — **the base of the whole `MTLBuffer`**.

So `Ndarray::buffer_offset` never reached the GPU. 0001 carried it to the
descriptor bind site, faithfully porting the Taichi hunk, and on Apple silicon
that bind site is dead code. Every imported torch tensor with a non-zero
`storage_offset()` — which, per `taichi_patches/README.md` §0001, is every ray-state
slice `sheet_resolve_shade` takes and every array the raster and compaction
kernels touch — was read and written **at the base of its arena**, so the
aliasing views collapsed onto each other. Taichi is correct on the same hardware
for exactly one reason: its Metal backend never sets the capability, so its
ndarrays go through the descriptor its patch does offset.

0001 now adds the offset to the published address as well (`runtime/gfx/
runtime.cpp`, in `host_to_device`). The two arms are mutually exclusive — the
same `spirv_has_physical_storage_buffer` decides which the codegen emits and
which the runtime binds — so nothing is offset twice, and the descriptor arm
stays as the correct path for a device without the capability.

**Re-measured on the same hardware, and it lands on the reference.** Same scene,
same script, the only change being the fix above:

| | pixels over tolerance | max delta | worst frame | per-channel BGR, MPS vs CPU |
| --- | --- | --- | --- | --- |
| Quadrants, before | 79,914,286 of 83,913,984 | 255 | 4 | (16.7, 15.8, 16.7) vs (46.1, 48.1, 48.6) |
| **Quadrants, after** | **11,526** | **221** | **174** | (46.1, 48.1, 48.6) vs (46.1, 48.1, 48.6) |
| Taichi 1.7.4, reference | 11,527 | 221 | 174 | equal, mean 47.62 |

One pixel apart from Taichi in 83.9 million, at the same maximum on the same
frame: the residual is the float32-accumulator drift MPS-friendly mode produces
on both compilers, not anything the port does differently. **The Apple path
works on Quadrants.**

What is still not covered, and should not be read as covered: **one scene**, one
Apple GPU (a virtualized M1 whose per-launch numbers are not trustworthy even
though its compute is). Zero-copy engagement is no longer among the unknowns —
a later run reported `converted=29 launches (155 args), passthrough=0, 0 staged,
0 host`, so every argument took the path.

The attempt to widen it to a denser scene first found a **pre-existing Algan
bug rather than a port one**: `materials_and_lighting` died on Metal at frame
119 of 179 with `Trace/BPT trap: 5`, printing nothing — and the patched
*Taichi* wheel failed identically, same frame, same signal. That was a leak in
Algan's MPS import cache, filed and fixed as
`../algan/rendering/DESIGN_mps_support.md` §1.4. With it fixed the dense scene
renders on this wheel and agrees with the CPU (the numbers are under "The
second defect" below), so the bloom/glow/surface-heavy case is now measured.

### The second defect: the cache that drops the offsets

**Found by reading, not by rendering — and the scene above could not have seen
it.** `shapes_and_timeline` uses no bloom, glow, surface normals or glossy
prefilter, which are exactly the kernels the bug reaches.

Quadrants has a Python-side `LaunchContextBufferCache` with no Taichi
counterpart, keyed on argument *identity* (`lang/kernel.py:590`,
`(id(t_kernel), *[id(arg) for arg in args])`). A hit runs
`launch_ctx.copy(cached)` and skips the entire argument-processing block —
`set_args_ndarray` at `kernel.py:702` with it, and so
`set_arg_ndarray_buffer_offset` too. And `LaunchContextBuilder::copy`
(`program/launch_context_builder.cpp:60`) replays five members; the two
**0001 itself added** — `array_byte_offsets` and `array_byte_sizes` — were not
among them, because 0001 added ndarray-derived launch state without extending
the function that replays ndarray-derived launch state.

So it is not a stale offset from an earlier launch. It is *no* offset: the first
launch of a kernel binds each imported slice correctly, and **every launch after
it binds that slice at the base of the arena**, silently.

The cache is reachable only where Algan is: an `Ndarray` argument reports
`cacheable=True` (`lang/_func_base.py:713-720`) where a torch tensor reports
`False`, so this is the MPS-∩-Quadrants intersection again. Algan's import cache
returns the *same* `ExternalMetalNdarray` object per slice
(`mps_zero_copy.py:241`), so identity is stable and the hit rate is ~100 % after
the first launch of each argument set. Of the 55 kernels, 25 take a float and
are immune (a float is non-cacheable), 4 are unconditionally cacheable —
`apply_glow_and_opacity`, `gloss_pyramid_level`, `bloom_conv1d_f32`,
`bloom_upsample_bilinear_f32`, `grid_normals_sides_crosses` — and the rest are
cacheable whenever their integer arguments fall in CPython's small-int table,
which includes `compact_ray_slots` on the tail iterations of a wavefront.

The fix is two assignments and two asserts in `copy()`, and it belongs in 0001
rather than a patch of its own: the maps exist only because 0001 created them.
Replaying is exact rather than approximate — the key is argument identity,
`Ndarray::buffer_offset` is set once at import and never mutated, so a hit
necessarily wants the offsets the cached context already holds.

**The offset fix holds across a re-run — verified on hardware, 2026-09-04.**
Once `materials_and_lighting` could finish on Metal at all (the leak in
`../algan/rendering/DESIGN_mps_support.md` §1.4, fixed on master), it was
rendered on the Mac runner's Apple GPU with a wheel built from these patches
(`run_on_mac.yaml` run
[`33926483875`](https://github.com/algorithmicsimplicity/algan/actions/runs/33926483875),
`scripts/gate/mps_vs_cpu_ab.sh`, the wheel from `quadrants_build.yaml` run
`33850787142`): all 179 frames, 420 s on MPS against 209 s on that box's CPU,
and the two arms agree — mean brightness 55.33 vs 55.34, per-channel means
`(53.1, 57.5, 55.4)` on both, 76,866 of 49,902,336 channel samples (0.15 %)
over the tolerance of 2, max delta 131 at frame 42. This is the scene that
reaches every unconditionally cacheable kernel above (bloom, glow, the gloss
pyramid, the surface normals), so a cache hit binding at the base of the arena
would have shown as the dimmed picture defect 1 produced, not as an edge
residual with identical means. And the residual is Algan's, not the port's:
the Taichi 1.7.4 control on the same scene, same harness, same box (run
`33927559059`) reads 76,983 over tolerance, max 131, worst frame 42, the same
means — a difference of differences of 117 pixels in 49.9 million
(`../taichi_patches/MIGRATION.md` §10). The fast scene had already agreed the
same way on the same wheel (run
`33847294165`: means 39.23 on both arms, 1,059 of 12,545,280 over tolerance,
max 24).

**Still unverified: that the CUDA half works.** A compile check cannot tell you
that an sm_61 card loads the runtime module; that needs the maintainer's GTX
1050, and `PORTING-NOTES.md` §7 lists exactly what to look for there
(`atom.gpu.cas.b64` and no remaining `atom.sys`). Read `PORTING-NOTES.md` §5 for
where the Metal port is most likely to be wrong — the ranked list starts with
the nanobind integer default and the `LaunchContextBufferCache` interaction.

## 0004 — `!invariant.load` on kernel argument loads

The one patch here that is about speed rather than working at all, and the
only one with no Metal or CUDA in it: it is LLVM codegen, so it applies to
every backend.

**What it fixes.** A Taichi/Quadrants kernel takes one parameter, a
`RuntimeContext` holding a *pointer* to an argument buffer in global memory.
Every read of an ndarray's base pointer or of one of its shape dims is a load
from that buffer, emitted at **every use site inside the loop**, and the loads
carry no metadata: LLVM cannot prove the kernel's own stores do not write the
argument buffer, so LICM will not hoist them. `../taichi_patches/PLAN.md` §2.2 traces this to
plain `CreateLoad`s in `codegen_llvm.cpp` with no `!invariant.load` anywhere,
and the deleted `DESIGN_taichi_argument_loads.md` (recover with `git show
aa7d198^:DESIGN_taichi_argument_loads.md`) measured the cost on the shipped
renderer: **~3,100 of `sheet_resolve_shade`'s 37,100 static instructions** are
argument re-loads — 1737 `ld.u64` + 1383 `ld.u32` that re-derive values which
were constant before the kernel launched. It is also the whole of the arena
convention's penalty (+18% with every array bound, +1.7–3.0% as shipped),
because the offset table adds a third level to an already two-level dependent-load
chain.

The claim the optimizer is missing is simply true: the argument buffer is
written once by the host before the launch and is not writable from the kernel.
`!invariant.load` states exactly that, on every backend, and with it LICM hoists
the loads into the loop preheader.

**What it changes**, six edits across five files:

| file | change |
| --- | --- |
| `quadrants/codegen/llvm/codegen_llvm.{h,cpp}` | `mark_invariant_arg_load(load, callable)` — attaches `!invariant.load`, and wraps the four `CreateLoad`s that read the argument buffer: the ndarray data/grad pointer and each shape dim in `visit(ExternalPtrStmt *)`, the element load in `get_struct_arg` (which is what `ArgLoadStmt`, `ExternalTensorShapeAlongAxisStmt` and `ExternalTensorBasePtrStmt` all route through), and the buffer's own base pointer in `get_args_ptr`. The last one additionally gets `!dereferenceable(args_size)`, which is exact rather than a bound — `LaunchContextBuilder` allocates the buffer at precisely `Callable::args_size`. |
| `quadrants/program/compile_config.h` | `invariant_arg_loads`, default on. |
| `quadrants/analysis/offline_cache_util.cpp` | the new field, in the cache key. |
| `quadrants/python/export_lang.cpp` | one `def_rw`. |

**The gate is a compile-config field on purpose, and it has to be in the cache
key.** `get_offline_cache_key_of_compile_config` serializes an *explicit* list
of fields rather than the whole struct — `cache_loop_invariant_global_vars` is
already missing from it — so a flag added without that line would let the two
A/B arms share a compiled artifact and report the first arm's numbers as the
second's. That is the same class of silent-stale-arm bug `CLAUDE.md` warns about
for `ti.static` gates, and it is why this is not an env var read in codegen.
Being a config field also means Quadrants' generic plumbing gives
`qd.init(invariant_arg_loads=False)` **and** `QD_INVARIANT_ARG_LOADS=0` with no
Python change (`misc.py:435` iterates `dir(cfg)`).

**Not for a `@qd.real_func` callee, and that exclusion is correctness rather
than caution.** A kernel's argument buffer has no store to it anywhere in the
module. A callee's is an `alloca` in its *caller*, filled by
`set_args_ptr` / `set_struct_to_buffer` immediately before the call
(`visit(FuncCallStmt)`), so once the callee is inlined the store and these loads
sit in one function — and `!invariant.load` would license LLVM to move a load
above the store that initializes it, reading uninitialized stack. The guard is
`dynamic_cast<const Function *>(callable) == nullptr`. Quadrants' own PR #866
(`91c590563`, AMDGPU address-space tagging) reached the identical exclusion for
the identical reason on the identical two functions — but it landed 2026-08-28,
**after** v1.3.0, so it is not in this base and there is no helper here to reuse.
Expect a textual conflict at `get_struct_arg` and `get_args_ptr` on a rebase past
it; the two changes compose, they do not fight.

**What it deliberately does not do.** No `addrspace(1)` tagging. On NVPTX that
is what would upgrade the hoisted loads from `ld.global` to `ld.global.nc`
(`codegen_cuda.cpp:583-596` already pairs the two for read-only SNodes), but it
is backend-specific, it is `../taichi_patches/PLAN.md` row 15's job, and the plan's own order is
to land `!invariant.load` alone and confirm the hoist before stacking anything
on it.

**Built, and the hoist is confirmed. Not yet timed.**
`.github/workflows/quadrants_build.yaml`'s Linux leg builds it on the free
runner (clone `v1.3.0`, apply, `./build.py wheel`, ~20 minutes) and then runs
`verify_invariant_load.py` (beside this README) in one process per arm. On **LLVM 22 / clang,
x64 CPU backend**, over an eight-ndarray sum kernel:

| | `off` | `on` |
| --- | --- | --- |
| `!invariant.load` sites | 0 | 11 |
| `!dereferenceable` sites | 0 | 2 |
| **argument base-pointer re-loads inside the loop** | **18** | **0** |
| scalar / vector float loads in the loop | 16 / 0 | 0 / vectorized |
| loop body, lines | 112 | 30 |

So the metadata lands, the gate gates in both directions, and LICM does act on
it: every argument base-pointer re-load leaves the loop body, which is what
`DESIGN_taichi_argument_loads.md` §3a predicted and the whole point of the
patch. The scalar float loads disappearing is not a parse failure — with the
base pointers hoisted, LLVM vectorizes the body, which is a second win the
measurement was not looking for.

What is **not** established is the thing that matters commercially: **no timing,
and nothing on CUDA.** The runner is shared, 4-core and has no GPU, so a number
from it would be noise. `../taichi_patches/PLAN.md` §5 is still the order for that, on a real box,
against the wheel this job uploads: dump PTX (`print_kernel_llvm_ir_optimized`,
consumed by the new-PM O3 pipeline at `jit_cuda.cpp:291`/`:325`), confirm the
hoist there too, and only then time both arms. If only the arena penalty shrinks
and the shipped renderer does not speed up, the patch is still worth having and
§4's projection was wrong.

## 0005 — a per-kernel `.maxnreg`, and `gpu_max_reg` made real

**What it fixes.** `../taichi_patches/PLAN.md` row 14, and §2.2 is the finding
behind it: `gpu_max_reg` **is inert**. `JITSession::add_module(M, int max_reg = 0)`
is only ever called as `add_module(std::move(module))` (PLAN's locator is
Taichi's `llvm_runtime_executor.cpp:181`; 0005's hunk anchors at Quadrants'
`:173`), so the config field is read by the Python
setter and the offline-cache key and by nothing else — `ALGAN_GPU_MAX_REG` has
only ever changed the cache key, which is why `algan/rendering/taichi_runtime.py:27-30`
records that Algan has no register-cap setting at all. Quadrants reached the
same conclusion and **deleted** the option in `3e45a7a7c` (#890, 2026-08-26) —
which is *after* v1.3.0, so the field is still there on this base.

Two halves, and they are independent:

* **module-wide**: one line in `create_jit_module` passes `config_.gpu_max_reg`
  on CUDA, so `qd.init(gpu_max_reg=N)` reaches `CU_JIT_MAX_REGISTERS` at
  `cuModuleLoadDataEx`. CPU asserts a zero and AMDGPU ignores it, so every other
  arch keeps the old one-argument call.
* **per loop**: `qd.loop_config(max_reg=N)` becomes that kernel's PTX
  `.maxnreg` directive and nothing else's.

**What it changes**, 16 files, +81/−5:

| file | change |
| --- | --- |
| `quadrants/ir/frontend_ir.{h,cpp}` | `ForLoopConfig::max_reg` and `FrontendForStmt::max_reg`, the copy-ctor and `init_config` lines, the `ASTBuilder` reset, and `ASTBuilder::max_reg(int)` with a `QD_ASSERT(v >= 0)`. |
| `quadrants/transforms/lower_ast.cpp` | carries it onto the lowered `StructForStmt` / `RangeForStmt` (×2) / `MeshForStmt` — all four construction sites. |
| `quadrants/transforms/offload.cpp` | onto `OffloadedStmt`, three sites, each beside the existing `block_dim` assignment. |
| `quadrants/ir/statements.{h,cpp}` | the field on `RangeForStmt`, `StructForStmt`, `MeshForStmt` and `OffloadedStmt`, in each one's field list and in each one's `clone()`. |
| `quadrants/codegen/cuda/codegen_cuda.cpp` | `current_task->max_reg = stmt->max_reg;`, on the line after the one `current_task->block_dim` assignment. |
| `quadrants/codegen/llvm/llvm_compiled_data.h`, `codegen_llvm.cpp` | `OffloadedTask::max_reg`, passed to `mark_function_as_cuda_kernel` from `run_compilation`. Deliberately **not** in the task's `QD_IO_DEF`: a cached artifact carries the metadata in its module already. |
| `quadrants/runtime/llvm/llvm_context.{h,cpp}` | `mark_function_as_cuda_kernel(func, block_dim, max_reg)` emits the `maxnreg` nvvm annotation **and** the `"nvvm.maxnreg"` function attribute. Both, on purpose: since LLVM 21 the NVPTX backend reads the attribute, and only a module that arrives through the text or bitcode reader gets the legacy annotation auto-upgraded into one — a freshly built in-memory module does not. |
| `quadrants/runtime/llvm/llvm_runtime_executor.cpp` | the module-wide line above. |
| `quadrants/analysis/gen_offline_cache_key.cpp` | `emit(stmt->max_reg)` beside `emit(stmt->block_dim)` in the `FrontendForStmt` serializer. |
| `python/quadrants/lang/misc.py`, `quadrants/python/export_lang.cpp` | `loop_config(max_reg=)` → `_max_reg` → `ASTBuilder::max_reg`. |
| `docs/source/user_guide/gpu_execution_model.md` | one sentence under "Block". |

**Per kernel, not per module — for the directive.** `insert_nvvm_annotation`
and `addFnAttr` both take the `llvm::Function *`, and `run_compilation` walks
`offloaded_tasks` marking each entry separately, so `.maxnreg` lands inside one
`.entry` and caps that kernel alone. The *other* half is the module-level one:
`CU_JIT_MAX_REGISTERS` is an option on `cuModuleLoadDataEx`, so
`qd.init(gpu_max_reg=N)` caps **every kernel in the module** — every offloaded
task of every kernel that shares it, and the device-side runtime module too,
since `create_jit_module` is the path that loads that as well. That is the
knob's documented meaning, but it is worth saying out loud before anyone sets it
to tune one megakernel.

**Both set at once: the patch says the per-loop cap wins, and that is the one
claim here nothing checks.** It is stated in four places (the `.md`, the
`loop_config` docstring, `ForLoopConfig::max_reg`'s comment and the
`create_jit_module` comment) on the strength of ptxas treating a per-entry
`.maxnreg` as overriding `--maxrregcount`/`CU_JIT_MAX_REGISTERS`. That is the
documented PTX behaviour; it has **not** been verified here, and
`verify_cuda_patches.py` cannot see it — both of its arms pass `gpu_max_reg=0`
precisely so the module-wide cap cannot confound the `.maxnreg` count. Settling
it needs a third arm with both set and `cuobjdump --dump-resource-usage` on the
driver's cubin.

**The cache key is covered, on both halves, and neither needed a new line in
`get_offline_cache_key_of_compile_config`.** The per-loop cap is AST state, so it
goes in the *AST* key — that is the `gen_offline_cache_key.cpp` hunk, and
without it two loops differing only in `max_reg` would share an artifact, the
same failure mode 0004's section describes for a config field. The module-wide
`gpu_max_reg` was already in the compile-config key (PLAN §2.2: the field was
"read only by the pybind setter and the cache key"), which is exactly why it
could stay inert for so long without anyone noticing. **Not verified against
Quadrants' own `offline_cache_util.cpp`** — that claim is 1.7.4's, carried over.

**How it is gated, and that Algan does not use it.** Nothing in `algan/` calls
`loop_config(max_reg=)` (grep: no hits outside these patches), and
`taichi_init_kwargs()` (`algan/rendering/taichi_runtime.py:614-665`) passes no
`gpu_max_reg`. So installing a wheel with 0005 in it changes no Algan render:
0005 is a lever, not a change. PLAN §8's first bullet is the order for using it
— `ptxas -v` on `sheet_resolve_shade`, `wavefront_shade`, `pt_shade` and
`wavefront_traverse_events` first, *then* a cap, because §9 still lists the
register-pressure effect on the 161-register megakernel as an open question and
a cap chosen without the spill numbers is as likely to cost as to pay.

**Review findings.** Two are real work, the rest are "check on the first build":

* **The two-argument `add_module` is assumed, not shown.** No hunk in 0005
  touches `jit_session.h` or `jit_cuda.cpp`, so the whole module-wide half rests
  on Quadrants v1.3.0 still declaring `JITSession::add_module(std::unique_ptr<llvm::Module>, int max_reg = 0)`
  and on `JITSessionCUDA::add_module` still turning it into `CU_JIT_MAX_REGISTERS`.
  PLAN row 14 ("1 line at `llvm_runtime_executor.cpp:181`") says it does. If the
  Linux build fails on arity here, that is why, and the fix is one more hunk in
  `jit_session.h` / `jit_cuda.cpp` rather than a rewrite.
* **`Arch::cuda` gates the module-wide half but nothing gates the per-loop one.**
  `loop_config(max_reg=N)` on a CPU or Metal build records the value, serializes
  it into the AST key and is then ignored by codegen — a silent no-op plus a
  cache miss. That is the same shape `block_dim` has, so it is consistent rather
  than wrong; worth knowing before blaming a cap for a CPU rebuild.
* **`QD_ASSERT(v >= 0)` is the only validation.** PTX accepts `.maxnreg` in
  16-255; `max_reg=8` or `max_reg=1000` reaches ptxas and fails there, at JIT
  time, inside a render. A range check in `ASTBuilder::max_reg` would be two
  lines and is worth having before anyone drives this from a settings field.
* **`current_task->max_reg` is set at one site**, co-located with the file's only
  `current_task->block_dim` assignment, so whatever task kinds get a `block_dim`
  get a `max_reg`. Inferred from the hunk's context — the pristine
  `codegen_cuda.cpp` is not checked out here.
* **A rebase past #890 loses `CompileConfig::gpu_max_reg` entirely**, and with it
  the module-wide half and one line of `create_jit_module`. The per-loop half
  does not depend on it.

## 0006 — read-only ndarray arguments through `ld.global.nc`

**What it fixes.** PLAN row 15. `codegen_cuda.cpp` already has the read-only
cache path — `create_intrinsic_load`, an `__ldg` reached when
`mem_access_opt` marks a **SNode** `read_only` — and no ndarray ever reaches it:
`detect_read_only` flags SNodes only, and its `detect_external_ptr_access_in_task`
sibling feeds launch-time bookkeeping rather than codegen. Algan has no SNodes.
Every array it renders from is an ndarray argument, so the whole existing
mechanism is dead code for it.

**What it changes**, 4 files, +118/−1, and all of the substance is in one place:

| file | change |
| --- | --- |
| `quadrants/codegen/cuda/codegen_cuda.cpp` | `root_external_ptr`, `is_ldg_element_type`, `readonly_ndarray_args(OffloadedStmt *)` and `is_read_only_ndarray_load(GlobalLoadStmt *)` (~110 lines, memoized per offload), and the one-line change that makes `visit(GlobalLoadStmt *)`'s non-SNode arm pass that predicate instead of a literal `false`. |
| `quadrants/program/compile_config.h` | `readonly_ndarray_ldg`, **default off** (it was on in the first draft; see the review findings). |
| `quadrants/analysis/offline_cache_util.cpp` | the field, in the compile-config cache key, on the line after 0004's. |
| `quadrants/python/export_lang.cpp` | one `def_rw`, which is all Quadrants' generic plumbing needs to give both `qd.init(readonly_ndarray_ldg=False)` and `QD_READONLY_NDARRAY_LDG=0`. |

**How the analysis decides "written", and it is a whitelist.** For one offloaded
task: an argument is a *candidate* if some `ExternalPtrStmt` roots at its
`ArgLoadStmt`, and it is *written* unless every use of every pointer derived from
it is a `GlobalLoadStmt` (or the `MatrixPtrStmt` edge on the way to one). So
stores, atomics, and a pointer handed to any statement the author did not think
of all count as writes without being enumerated, and a store *through* a
`MatrixPtrStmt` — which `detect_external_ptr_access_in_task` does not see — is
caught. `ti.static`-gated stores need no special handling: a static branch is
resolved before the IR exists, so a store under a false gate is not in the task
and the variant that has it gets its own cache entry. A `@qd.real_func` callee
is not analysable at all — its IR is separate and numbers its arguments in its
own space — so **one `FuncCallStmt` anywhere in the task returns the empty set**
and the whole task falls back to plain loads. Algan uses no real functions
(`algan/utils/taichi_early_return.py` works on inlined `ti.func`s only), so that
bail costs it nothing.

**Aliasing between two arguments is not considered at all**, which the config
comment admits ("Sound on the same assumption the SNode path makes, that
distinct arguments do not alias"). For Algan that assumption is **false as
stated**, and the section below is why it is nevertheless not currently a wrong
picture.

**What `verify_cuda_patches.py` proves, and what it cannot see.** Its probe
kernel takes two ndarrays it only reads and one it reads *and* writes, and the
`on` arm must show at least one `ld.global.nc` **and** at least one plain
`ld.global` left over — the written array staying uncached is the control, and
the script fails the arm if it disappears. The `off` arm must show no `.nc` at
all, which is the gate. That is a real proof of the mechanism and of the flag.
It is not a proof of the *analysis*: one offload, three disjoint host arrays,
f32 only, no `MatrixPtrStmt`, no `ti.func`, no atomics, no arena. Nothing in it
resembles a 20-argument megakernel.

**Review findings.** In order of what would bite:

* **Unsound if a write ever reaches an argument through a pointer chain
  `root_external_ptr` does not walk** — severity *unsound*, and the one hole
  worth closing before this is enabled. `root_external_ptr` follows
  `MatrixPtrStmt → ExternalPtrStmt` and gives up on anything else, but
  `ExternalTensorBasePtrStmt` also yields an ndarray's base pointer (this
  README's 0004 section names it as one of the three statements routing through
  `get_struct_arg`). A store rooted there is invisible to `written` while a read
  through an ordinary `ExternalPtrStmt` still makes the same argument a
  *candidate* — read-only classification of a written array, which is exactly
  the stale-read the patch is trying to avoid. The cheap fix is a bail rather
  than a walk, in `readonly_ndarray_args`, beside the `FuncCallStmt` one:

      if (stmt->is<FuncCallStmt>() || stmt->is<ExternalTensorBasePtrStmt>()) {
        return {};
      }

  **Applied to the patch** (`165914d`), and compiled by the run under
  "what is verified" below. It costs nothing on any kernel that does not use
  one and removes the whole question. (Whether Algan reaches that lowering at all is unestablished — the
  `NODE_ARG` BVH arrays are vector-element ndarrays, `raytrace_kernels_taichi.py:401-404`,
  and they are read-only in every traversal kernel, so the *dangerous* half is
  hypothetical here. The guard is still worth two lines.)
* **Two Algan arguments alias byte for byte, by construction** — severity
  *unsound-in-principle, inert in practice*, and it deserves to be written down
  rather than discovered. `ManualMemory` claims one `torch.uint8` block and
  **every render-time tensor is a view into it** (`algan/utils/memory_utils.py:640-641,732`).
  The arena calling convention then hands a converted kernel `_whole_storage(...)`
  — a view of that *entire* allocation, per dtype
  (`algan/rendering/raytracing/arena_args_taichi.py:161-172,249`). So in
  `sheet_resolve_shade`, `arena_f32` and `arena_i32` are the **same bytes** at
  two types, and each of them contains every other ndarray argument (`rs_ro`,
  `rs_rd`, `pix_accum`, …) inside its span. Per *argument*, aliasing is total.
  It is not currently exploitable for two independent reasons, and both are
  accidents rather than design: the bump allocator makes distinct logical arrays
  byte-disjoint, so an address read through one argument is never an address
  written through another; and the arena arguments are themselves written
  (below), so they are never classified read-only in the first place. The
  standard `ld.global.nc` requirement is per address, not per object, so
  disjointness is what actually saves it — but nothing in Algan enforces
  disjointness on purpose, and nothing in the analysis would notice if it
  stopped holding.
* **On the seven widest kernels 0006 is close to a no-op, for the same reason**
  — severity *pixel-neutral, benefit-negating*. The arena convention collapses
  ~40 logical arrays into two ndarray arguments, and the analysis is per
  argument, so **one written table poisons every table sharing its dtype
  arena**. `sheet_resolve_shade` writes `sheet_memo` through `arena_f32`
  (`sheet_resolve_taichi.py:227,578-585`) and `sheet_accept` through `arena_i32`
  (`:228,622`), so both arena arguments are written and none of the 41
  arena-bound tables — `tri_pos`, `tri_norm`, `textures`, `light_*`, the whole
  cold read-only half the arena exists to carry — gets `.nc`. What is left for
  0006 is the arrays that stayed ordinary parameters, which are mostly the
  per-slot ray state the kernel *writes*. Measuring this patch on the shipped
  renderer will most likely measure nothing; the honest test is a kernel that
  takes its read-only tables as ordinary arguments, or a follow-up that makes
  the hint per *`ArenaView`* rather than per argument.
* **Element types the probe never exercises** — severity *would-not-compile /
  fatal at JIT*. `is_ldg_element_type` admits every storage primitive — f16,
  i8/u8 and i16/u16 among the eleven —
  and `create_intrinsic_load` picks between `nvvm_ldg_global_f` and
  `nvvm_ldg_global_i` by `isFloatingPointTy()`. An `llvm.nvvm.ldg.global.f` on a
  `half` needs an ISel pattern to exist for it in LLVM 22; if it does not, the
  failure is `Cannot select` at compile time, in a render, on exactly Algan's
  f16 BVH nodes. `verify_cuda_patches.py` loads only f32, so it cannot see this.
  The conservative first landing is to narrow the predicate to f32/f64/i32/i64
  and widen it once a real megakernel has compiled.
* **Missing includes** — severity *would-not-compile*, **refuted by the build**:
  run `33926192036` compiled the file as patched, so `codegen_cuda.cpp` does
  pull `std::set` and `irpass::analysis::gather_statements` in transitively,
  and `current_callable` / `current_offload` are reachable from the subclass.
  The concern as it was written, for the record — the patch adds only
  `#include "quadrants/program/function.h"`, and had the build failed it would
  have needed:

      +#include <set>
       #include "quadrants/util/io.h"
       #include "quadrants/ir/ir.h"
      +#include "quadrants/ir/analysis.h"
       #include "quadrants/ir/statements.h"

  Same class of thing: `current_callable` and `current_offload` are read from the
  `TaskCodeGenLLVM` base and must be at least `protected` there (0004 uses
  `current_callable` from inside the base itself, which does not prove it).
* **The flag defaulted to `true`** — severity *policy*, **fixed** (`165914d`).
  Algan sets nothing, so installing the wheel would have turned this on for
  every CUDA render with no Algan change and no opt-in, on an analysis that has
  never seen a real kernel. The patch now says `bool readonly_ndarray_ldg{false}`;
  a measurement opts in per process with `qd.init(readonly_ndarray_ldg=True)`
  (what `verify_cuda_patches.py`'s `on` arm does), and Algan's
  `taichi_init_kwargs()` will set it only once a CUDA full-render run has.

## 0007 — fast `expf` under `fast_math`

**What it fixes.** PLAN row 18, and it is the smallest patch here: 15 lines.
`codegen_cuda.cpp` gives `sin`, `cos` and `log` a `fast_math` branch that picks
libdevice's approximate routine, and `exp` alone was left on the generic
`UNARY_STD(exp)` macro, so `qd.exp` called `__nv_expf` however the process was
initialized. The patch replaces the macro invocation with the same explicit
branch the three others have: `__nv_fast_expf` under `fast_math`, `__nv_expf`
without it, `__nv_exp` for f64 (never affected) and `exp` for i32, all unchanged
from what the macro expanded to.

**How it is gated: it is not, separately — and Algan is already on the fast
side.** There is no new flag. `fast_math` decides it, and
`taichi_init_kwargs()` pins `"fast_math": True`
(`algan/rendering/taichi_runtime.py:628`). So unlike 0005 and 0006, **0007
changes every CUDA render the moment the wheel is installed**, with nothing to
set and nothing to opt into.

**Which Algan kernels this reaches.** Every f32 `ti.exp` in the package, which is
15 call sites in four files and two distinct uses:

| use | sites | argument range |
| --- | --- | --- |
| Beer-Lambert attenuation, `ti.exp(-sigma * seg)` | `wavefront_kernels_taichi.py:2956-2958,3312-3314`, `raytrace_kernels_taichi.py:3211-3213,3433-3435`, `path_tracer_taichi.py:1374` | unbounded above; the result underflows toward 0 as it grows |
| a shading fit — the `exp2` spelled as `ti.exp` at `wavefront_kernels_taichi.py:1045-1047`, and `dg = ti.exp(a * cos_theta + b)` at `shading_taichi.py:377` | 2 | small, `|x| ≲ 6.5` for the first |

Not the bloom gaussian weights (built in torch on the host,
`algan/rendering/post_processing/bloom.py`) and not tonemapping (no `ti.exp` in
`color_space_taichi.py`). So the exposure is attenuation and two shading terms.

**Is it different in kind from the `sin`/`cos`/`log` Algan already runs? No.**
`__nv_fast_expf` is libdevice's `__expf`: one `ex2.approx` on a scaled input,
with a documented error of 2 ulp plus ~1.16 per unit of `|x|` — argument-
dependent, exactly like `__nv_fast_sinf`'s. The one thing worth noticing is that
Beer-Lambert is the first place Algan feeds an *unbounded* argument to one of
these: at `|x| = 20` the bound is ~25 ulp. That is ~3e-6 relative, on a value of
~2e-9, multiplied into a colour that is then quantized to 1/255 — three orders
of magnitude below one channel value. **Direct** pixel movement from this cannot
reach the pixel-compare tolerance of 2.

**What can, and what to do about it.** The renderer takes discontinuous
decisions on these values — `ti.min(cx * cx, ti.exp(-6.4324058 * nv))` at
`wavefront_kernels_taichi.py:1047` is a comparison, and acceptance thresholds
elsewhere are — so a last-bit change can flip a branch and move one edge pixel by
a lot. That is the same exposure `fast_math` already carries for sin/cos/log, and
the answer is the same as `CLAUDE.md` prescribes for any legitimate rendering
change: clear the offline cache, run `pytest -q tests/full_renders` **on CUDA**,
look at the diff videos, and re-baseline `expected_outputs_cuda/` with
`ALGAN_UPDATE_FULL_RENDER_BASELINES=1` only if the differences are the expected
last-bit kind. CPU baselines cannot move — this is `codegen_cuda.cpp` — so CI,
which runs `tests/unit_tests tests/fast` on CPU, will not see it either way.

**Review findings.**

* **No cache-key line, and that is right only if `fast_math` is already in the
  key** — severity *would-silently-share-an-artifact* if it is not. 0007 adds no
  field, so the arms of any A/B are distinguished only by `fast_math` itself. In
  Taichi 1.7.4 that field is serialized in
  `get_offline_cache_key_of_compile_config` and Quadrants' PTX cache keys on it
  too (PLAN row 2: "IR text + SM + fast_math"), so it is almost certainly there —
  but **no hunk in this directory shows it**, and 0004's section is the record of
  what happens when a flag is missing from that list. Check it on the first
  build; the fix, if needed, is one `serializer(config.fast_math);`.
* **The macro-to-branch swap is balanced but load-bearing** — severity
  *would-not-compile* if it is wrong, and it is the only structural risk in the
  patch. `UNARY_STD(exp)` expands to a complete `else if (...) { ... }`, so
  removing it and opening the new branch with `} else if` requires the preceding
  `cos` branch's closing `}` to become the *new* branch's opener and the removed
  macro's line to leave the trailing `}` behind as the exp branch's close. The
  hunk does exactly that and the braces balance as written. Nothing else in the
  chain moves.
* **The comment's own arithmetic is off by the factor it then uses** — severity
  *cosmetic*. It says "2 ulp plus one per unit of `|x|`" and then computes ~25
  for `|x| = 20`, which is the documented 1.16 coefficient, not 1.0. Worth
  correcting to "~1.16 per unit of `|x|`" so the number and the sentence agree.
* **`verify_cuda_patches.py` reads the *unoptimized* IR for this one, and has
  to.** O3 inlines both libdevice routines and both end in an `ex2.approx`, so
  neither the optimized IR nor the PTX can tell them apart by name. The check is
  therefore "which function did codegen ask for", which is precisely what the
  patch changes — but it means the script proves **nothing about accuracy**, and
  nothing about pixels. Only `tests/full_renders` on CUDA can.

## 0005-0007 — what is verified, and what is not

**Nothing.** These three were written in one session against the v1.3.0 sources
as read, never applied to a checkout, never compiled, never run, and never
rendered with. Everything above that is not a quotation from a hunk in this
directory or a `file:line` in this repository is inference, and the review
findings in each section are the places that inference is thinnest.

Concretely, the order in which the unknowns should fall:

1. **They apply.** `git apply --check` of all seven onto pristine `v1.3.0`,
   expecting offsets on `codegen_cuda.cpp` and `llvm_context.cpp` (see
   "Applying them") and errors nowhere.
2. **They compile**, Linux with `QD_WITH_CUDA=ON` — the missing includes and the
   `add_module` arity in the review findings are what that leg decides.
3. **`verify_cuda_patches.py on` / `off` / `--compare`** on a box with a CUDA
   device: `.maxnreg` lands and gates (0005), `ld.global.nc` appears with a plain
   `ld.global` kept beside it (0006), `__nv_fast_expf` replaces `__nv_expf` in
   the unoptimized IR and only under `fast_math` (0007). On a GPU-less runner it
   writes `{"skipped": true}` and exits 0 — **a skip is not a pass**, and the
   compare prints `SKIP` rather than claiming one.
4. **`pytest -q tests/full_renders` on CUDA**, which is the only thing that can
   see 0007's pixels, and the only thing that would catch an unsound `.nc` from
   0006 as a picture rather than as an argument.

**Steps 1 and 2 are done** (2026-09-04, `quadrants_build.yaml` run
[`33926192036`](https://github.com/algorithmicsimplicity/algan/actions/runs/33926192036),
`ubuntu-22.04`, `QD_WITH_CUDA=ON`): all seven patches applied strictly in order
onto pristine `v1.3.0`, the wheel built
(`quadrants-1.3.1.dev0+gab9a58ab5.d20260904-cp311-cp311-manylinux_2_27_x86_64.whl`,
26 MiB), installed, and the 0004 checks still pass on it (`!invariant.load`
11 sites, loop base-pointer loads 18 → 0, the kwarg and env var both
honoured). `qd.init(gpu_max_reg=48, readonly_ndarray_ldg=True)` is accepted
and reads back. The "missing includes" and "`add_module` arity" findings above
are therefore refuted; the aliasing, element-type and precedence findings are
not touched by a compile and stand. That run predates the two 0006 fixes
(`165914d`: default off, base-pointer bail); run
[`33927637278`](https://github.com/algorithmicsimplicity/algan/actions/runs/33927637278)
rebuilt with them and passed every step, the gate step included.

**Steps 3 and 4 are not done.** No CUDA device has run
`verify_cuda_patches.py` (the CI arms wrote `{"skipped": true}` as designed),
and no CUDA render has been taken with any of the three on. Until then 0005 and
0006 are inert in Algan — nothing sets `max_reg`, `gpu_max_reg` or
`readonly_ndarray_ldg` — and 0007 is the one that *is* live on a CUDA render,
because Algan already runs `fast_math=True`: expect a last-bit change in every
kernel that calls f32 `exp`, and re-baseline `expected_outputs_cuda/`
deliberately, after looking, when the first CUDA render on this wheel says so.

## Upstreaming

0003 is written to be upstreamed rather than carried: minimal, no Algan-specific
naming, and a no-op on hardware that works today. Device scope is more correct
*and* faster than system scope, and the capability gate copies a pattern
Quadrants already applies elsewhere. A permanent fork delta for it is a rebase
tax for nothing.

0004 is the other one that belongs upstream rather than here: it names a
property of the argument buffer that is simply true, on every backend, and its
only Algan-specific thing is which kernels benefit. Its one caveat is the rebase
conflict its own section names — Quadrants PR #866 (`91c590563`) edits the same
two functions and postdates v1.3.0, so it will conflict textually; the two
changes compose, they do not fight.

0002's `ContinueStmt` hunk is the next best candidate — it is a plain bug
affecting any Metal or Vulkan user with a `continue` under a static gate. 0001
is Algan-shaped and will likely stay a fork patch.
