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
+492/−8 — and 0002 is authored against a tree that already has 0001, exactly as
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

What is still not covered, and should not be read as covered: one scene, one
Apple GPU (a virtualized M1 whose per-launch numbers are not trustworthy even
though its compute is), and `zero_copy_available()` is not the same claim as
"every argument took the zero-copy path" — the probe now prints
`mps_zero_copy.report()` so the next run says how many launches converted and
what, if anything, is still crossing the bus.

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

**Still unverified: that the CUDA half works.** A compile check cannot tell you
that an sm_61 card loads the runtime module.
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
