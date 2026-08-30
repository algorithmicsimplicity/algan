# Algan on Apple GPUs (MPS / Metal): measured verdict

Status: **NO-GO on the port as such.** Measured on GitHub's `macos-latest` runner
(Apple Silicon, macOS 26.5.2 arm64, torch 2.7.1, taichi 1.7.4) by
`benchmarks/_mps_capability_probe.py`, run from `.github/workflows/mps_probe.yaml`.

The macOS CI job is pinned to `ALGAN_RENDER_DEVICE=cpu` because MPS renders fail.
The workflow comment attributes that to `float64` and says supporting MPS "means
taking float64 out of the raster pipeline and the kernels". **That is true and it
is the least of it.** Three independent Metal limits block the port, f64 is the
smallest, and the one that decides the question is the kernel argument limit,
which nobody had looked at.

Read §1 for the verdict, §2 for the numbers behind it, §3 for what to do instead.

**Status, added later.** All three blockers are cleared and *measured cleared on
the hardware*: §1.1 by packing kernel arguments into arena offsets, §1.2 by
MPS-friendly mode, and §1.3/§1.3b by the forked Taichi in `taichi_patches/`,
which lets a kernel bind torch's own `MTLBuffer` instead of staging it through
the host. Every kernel in the renderer compiles on Metal, the zero-copy path is
engaged (40 converted launches, 244 arguments, nothing left on the staging
path), and **`benchmarks/_mps_render_smoke.py` draws a real frame** — the
fragment stream matching the CPU's to the unit: 59790 fragments, pixels
`[137724..282179]` over 30929 distinct, `frag_cov` summing to 40133.251 against
40133.251.

Getting from "black" to that took two defects nobody had looked for, both of
them **torch on MPS answering wrongly rather than failing**, and both in the
same place: §2.3b. MPS gathers an integer through 24 bits of mantissa, and
Algan's two wide int64s are composite keys — the packed fragment key at 2**50
and the shading-class key at 2**40 — so both lost the low bits that carry their
meaning. The verdict below is no longer NO-GO on any of the three counts.

**What is not yet clear** — §1.2c below. The macOS suite is at **7 failed, 2415
passed, 167 skipped**; the Linux control arm, running the same suite with
MPS-friendly mode forced on over a CPU render device, is **fully green**, which
is what says the mode itself is sound and the remainder is Metal.

**Scope, added later.** Everything below measures **Taichi on the Metal
backend**. Two of the three blockers (§1.1, §1.3) turn out to be properties of
Taichi's interop and codegen rather than of Metal, and they do not survive
dropping Taichi from the path in favour of hand-written Metal shaders dispatched
through `torch.mps.compile_shader` — see `DESIGN_metal_native_port.md`, which
sizes that alternative and does not contradict a number here.

**Timing caveat, added later.** `macos-latest` is a virtualized-GPU instance.
It is sound for the capability results — which are the ones the verdict rests on,
and they stand — but every millisecond figure in §1.3 and §2 should be read as
directional only, not as a number to plan against. The staging comparison is the
robust one (a host round-trip is a host round-trip); the compute-bound
"52x better than CPU" is the fragile one. Sizing any of this needs physical
Apple hardware.

---

## 1. The three blockers, in order of how much they cost to clear

### 1.1 Metal binds at most ~24 ndarray arguments — this is the blocker

A kernel with 24 ndarray arguments runs. **31 aborts the process:**

```
-[MTLComputePipelineDescriptorInternal setComputeFunction:withType:]:866:
    failed assertion `computeFunction must not be nil.'
```

The MSL function comes back nil — the shader did not compile — and Taichi does
not check before building the pipeline, so it is a `SIGABRT`, not an exception.
Consistent with Metal's 31-buffer-per-stage binding limit once Taichi's own
context and root buffers are counted.

Algan's renderer is built out of megakernels, and the important ones are all over
the line:

| kernel | ndarray args | fits in 24? |
| --- | --- | --- |
| `sheet_resolve_shade` | 49 | no |
| `wavefront_shade` | 38 | no |
| `raster_shadow_trace` | 30 | no |
| `wavefront_traverse_events` | 30 | no |
| `raster_tri_write` | 20 | yes |
| `raster_bez_write` | 18 | yes |
| `raster_tri_count` | 15 | yes |
| `raster_bez_count` | 14 | yes |

Every shading and ray-tracing kernel is blocked. Only the raster count/write
kernels fit. Clearing this means either splitting each megakernel into pieces of
at most 24 buffers — which cuts across exactly the fusion those kernels exist for
— or packing many arrays into single buffers and indexing inside the kernel,
which rewrites every signature in the renderer. Neither is a port; both are a
redesign of the kernel layer.

For contrast, Taichi's *own* ceiling is 64 arguments and it reports it politely
on both backends: `The number of elements in kernel arguments is too big! Do not
exceed 64 on metal backend.` The Metal limit is a third of that and announces
itself with an abort.

### 1.2 Metal has no f64, and no i64 atomics

f64, asked of Taichi directly on a Taichi-owned buffer, so torch is not in the
way:

```
RuntimeError: [spirv_ir_builder.cpp:get_primitive_type@299] Type f64 not supported.
```

This is the SPIR-V codegen, which serves **both** the Metal and Vulkan backends.
Metal Shading Language has no `double` at all, so there is no capability to
enable. (Torch refuses first with its own `Cannot convert a MPS Tensor to
float64`, which is why the earlier reading was ambiguous about whose limit it
was.)

i64 is the more interesting result, because it splits:

| | on Metal |
| --- | --- |
| i64 arithmetic, shifts, packing (`(depth << 32) \| layer`) | **works** |
| `ti.atomic_add` / `ti.atomic_min` on i64 | **aborts** |

```
Assertion failed: (p != nullptr), function bind_pipeline,
    file metal_device.mm, line 382.
```

Same failure on a torch tensor and on a Taichi-owned ndarray, so it is the
backend rather than the interop. The raster pipeline's 64-bit fragment keys are
therefore fine; `sheet_compact_taichi.py:547-554`'s i64 atomics are not.

**This is what rules out the obvious deterministic fix.** A Q32 fixed-point
accumulator is exactly order-independent and needs no f64, and it is measured
bit-identical run to run on MPS — but only through torch. As a Taichi kernel on
Metal it aborts with the above. f32 `ti.atomic_add` does work, so a
non-deterministic mode has a floor; a deterministic one does not, in a kernel.

> **Addressed — MPS-friendly mode.** The floor is what shipped.
> `SETTINGS.computing.mps_friendly` (`'auto'`, on exactly when the render
> device is MPS) narrows every renderer path this section names, in one place:
> `algan/rendering/mps_compat.py`. f64 accumulators become f32 —
> `accumulate_dtype()` on the torch side, `taichi_accumulate_dtype()` for the
> kernels, passed as a `ti.template()` dtype argument so Taichi compiles a
> variant per width rather than resolving a `ti.static` gate once. The int64
> atomics and the int64 amin/amax `scatter_reduce_`s of §2.3 become int32
> (`reduction_index_dtype()`); every value they reduce is a position, a count
> or a surface id, all bounded by the fragment count, so **that** narrowing
> costs nothing at all. `cummax`/`cummin` become a log-step scan of
> `maximum`/`minimum`, which is exact because both ops are idempotent and
> neither reassociates.
>
> The float narrowing does cost what this section says it costs, and the mode
> says so rather than pretending otherwise: **MPS-friendly mode is not
> deterministic**. Measured on the fast suite's own scene, CPU, mode off
> against mode on: 99.94% of channels identical, 0.019% differing by more than
> 2, worst 34 — concentrated on silhouettes, which is the signature of a
> ceiling that wobbles in its low bits flipping borderline fragments in and out
> of being clipped, exactly as §2.4 predicts.
>
> The mode is settable on any device, and that is deliberate: it is what lets a
> machine with no Apple GPU run the substitutions. `tests/unit_tests/
> test_mps_friendly.py` does, including both compiled kernel variants against
> each other, and it walks the AST of `algan/rendering/` so a new f64
> accumulator fails a test rather than a Mac.
>
> **Confirmed on the machine.** `benchmarks/_mps_metal_codegen_probe.py`, run
> from the `render` job, puts `i64_atomic_min` at
>
> ```
> RHI Error: (spirv-cross compiler) MSL currently does not support 64-bit atomics.
> ```
>
> — this section's abort, now with a diagnostic instead of an assertion failure
> — while the mode's `i32_atomic_min` and `f32_accumulate` replacements both
> pass. Everything §1.2 said is Metal's limit is Metal's limit, and the mode
> clears all of it: with it on, **every kernel in the renderer compiles and the
> whole pipeline runs to completion on an Apple GPU**, including
> `sheet_resolve_shade_arena`, the 49-argument megakernel §1.1 called blocked.

### 1.2b Taichi's MSL generator writes a cast C++ parses as a declaration

Not Metal's limit and not Algan's code: a **Taichi codegen bug**, found by the
first render that got far enough to hit it, and cleared by narrowing an
argument. It is here because anyone re-treading this path meets it.

A narrowing cast of a 64-bit ndarray load, bound to a name and read more than
once, comes out of Taichi's SPIR-V-to-MSL step as a nested functional cast:

```
program_source:67:42: error: indirection requires pointer operand ('int' invalid)
        int tmp16_i32 = (int(long(_76))) * 8;
                                         ^ ~
program_source:67:13: error: cannot initialize a variable of type 'int' with an
        rvalue of type 'int (long)'
```

C++'s most vexing parse: `int(long(_76))` is the function type `int(long)` with
a parameter named `_76`, so the `* 8` after it parses as a dereference. Metal
hands Taichi a nil function; Taichi builds a pipeline from it without checking;
the process aborts with §1.1's `computeFunction must not be nil`, which is why
the two look identical from outside and why the probe was needed to tell them
apart.

The probe's ladder is what makes the shape of it exact, and it is narrower than
it first looks:

| spelling | on Metal |
| --- | --- |
| `ti.cast(i64[i], ti.i32)`, used once (`i64_cast_mul`) | **compiles** |
| the same **bound to a name and read twice** (`named_cast_mul_temp`) | **aborts** |
| `sheet_lane_first_owner` with an i64 `band` (`lane_owner_real_i64`) | **aborts**, same source line |
| the same kernel with `band` already i32 (`lane_owner_real_i32`) | **compiles** |

So the fix is the argument dtype, not the kernel: `mps_compat.kernel_index`
narrows an index array on its way into a kernel when the mode is on, and the
kernel's own `ti.cast(..., ti.i32)` becomes a cast to the type the value
already has, which Taichi emits nothing for. The same kernel source serves both
widths and no kernel changed. It is applied to every array the kernels narrow
per element — the CSR counts/starts pairs, the gather and depth-order
permutations, the band ids, the sorted order.

### 1.2c `sheet_resolve_shade_arena` will not compile for some scenes — OPEN

The one blocker still standing, and the only thing between the Apple GPU and a
green suite. Seven tests fail with

```
Assertion failed: (p != nullptr), function bind_pipeline, file metal_device.mm, line 409.
```

which is a SIGABRT with no kernel name and no Python traceback — Metal's answer
to a pipeline built from a shader that did not compile. The compile log names
it, and the trace has to be taken with `pytest -s` and `PYTHONUNBUFFERED=1`,
because pytest captures stdout per test and replays it only when a test *fails*
— a SIGABRT is not a failure, so the buffer dies with the process and two
earlier attempts got the assertion and nothing else:

```
[Taichi compile] started ...sheet_resolve_taichi.sheet_resolve_shade_arena[specialization=0] at 14:27:36.312
Assertion failed: (p != nullptr), function bind_pipeline, ...
```

A `started` with **no matching `completed`**. So it is §1.1's 49-argument
megakernel — and the interesting part is that the *same kernel compiles fine* in
`benchmarks/_mps_render_smoke.py`, which logs `completed ... total=5.765s` for
it. Whatever fails is scene-dependent, and every failing test is about **lights
and shadows**: the glossy prefilter (3), the area-light soft shadow, the
deterministic shadow opacity, and the shadow-cap truncation.

Two hypotheses, neither tested:

* the shader grows with the shadow-light count (`MAX_SHADOW_LIGHTS` unrolling)
  until Metal's compiler refuses it — a size limit rather than a feature one;
* a code path taken only with shadows enabled hits the §1.2b MSL codegen bug in
  a form patch 0002 does not cover.

The next measurement is the MSL error text, which patch 0002's
`log_msl_source_context` exists to print and did **not** print here — so the
failure is arriving as a nil pipeline at *bind* rather than as a Metal compile
error at library creation, and the patch's guard is on the wrong side of it.
Moving that check is a wheel rebuild, which is why it is written down rather
than done.

The seventh failure is unrelated and also open:
`test_closed_shell_attenuates_once_at_authored_opacity` has the path-traced and
deterministic routes agreeing everywhere except one column of the interior,
where they differ by 86. It **passes on the CPU in both modes**, so it is
Metal-specific rather than an MPS-friendly substitution.

### 1.3b Two dtype views of one buffer cannot both be written — the arena

**Cleared** by the forked Taichi (`taichi_patches/0001`), which removes the
mechanism rather than working around it: a kernel binds torch's own `MTLBuffer`,
so there is no copy-back to revert anything. What follows is the measurement
that identified it, kept because it is the argument for the fork.

It is §1.3's staging meeting §1.1's fix, and it was not visible before, because
until both of the blockers above were cleared nothing got far enough to render
at all.

`arena_args_taichi.pack` binds a converted kernel's cold arrays as offsets into
the arena, which means the kernel takes `arena_f32` and `arena_i32` — **the same
allocation, reinterpreted**. Direct binding does not care: CPU and CUDA hand the
kernel one pointer twice. A backend that *stages* copies each argument to the
host, runs, and copies each back, and the second copy-back then reverts
everything the kernel wrote through the first. Measured, by the probe's
`device_two_typed_views`:

```
device_two_typed_views   FAIL -- AssertionError: tensor([0., 0., 0., ..., 0.])
```

The kernel wrote 1.5 through the f32 view and 9 through the i32 view of a
disjoint half of the same buffer; the f32 half came back **zero**. The plain
cases either side of it pass — `device_tensor_roundtrip` and
`device_view_roundtrip` — so staging is correct for an ordinary argument and
for a slice, and wrong exactly for the aliasing pair the arena convention
creates.

A render on the machine bears that out and adds a second symptom the same
staging could explain, in a stage that runs *before* any shading:

| | CPU | Metal |
| --- | --- | --- |
| fragments | 59790 | 69738 |
| covered pixels | 30929 | 10876 |
| sheets | 40956 | **128** |
| `frag_cov` min | 0.001000 | 0.000000 |

The frame is uniformly black. 128 sheets out of 69738 fragments is not a
rounding difference; something between the raster emission and the compaction
is reading data that is not there.

**Resolved, and it was two things, not one.** The fork fixed the aliasing and
the coverage column with it — `covered pixels` and `frag_cov min` came back
equal to the CPU's on the next run. The sheet count did not, and it was never
this section's defect: it was **§2.3b**, torch's own gather losing the low bits
of a composite key, which had been corrupting the fragment key underneath all
of this. Both are fixed; the render draws. The row this table should be read
for now is `sheets`, which is the one that outlived the staging fix and pointed
somewhere else entirely.

Three ways out were on the table, in increasing order of what they buy, and the
third is what shipped:

1. **Pack the arena arguments into a per-dtype staging buffer** at launch,
   rewriting the offset table to match. The aliasing goes away because the
   buffers are separate allocations, and the launch stops staging the *whole*
   arena twice — which is also most of §1.3's cost on the converted kernels.
   Contained in `pack`, and gated on the mode.
2. **Per-dtype arenas in `ManualMemory`**, so the views are disjoint by
   construction. Deeper, and it changes the allocator every backend uses.
3. **Taichi-owned ndarrays** (§3.3 step 1) — **done**, as `taichi_patches/0001`
   plus `algan/rendering/mps_zero_copy.py`. It removes staging altogether and is
   the only one of the three that also fixes §1.3's bandwidth verdict.

### 1.3 Taichi stages every torch tensor through host memory

`kernel_impl.py` copies any ndarray argument that is neither a host tensor nor a
CUDA tensor on a CUDA arch to the host before the launch and back after. The only
generic `Device::import_memory` implementations in Taichi's core are `CpuDevice`
and `CudaDevice` — there is none on the gfx device that serves Metal and Vulkan,
which is what makes `taichi_launch_is_local`'s rule (§3.2) right as written.

> **Amended.** The conclusion drawn here — that no externally allocated buffer
> can be adopted by either backend — is too strong for Metal. The Metal RHI
> carries its own non-virtual `MetalDevice::import_mtl_buffer`, and it is
> compiled into the wheel Algan already installs, though only reachable from
> C++. What blocks adoption is the Python frontend, not the backend.
> `DESIGN_mps_zero_copy.md` has the evidence and what a patch would cost. It
> changes nothing about this section's measurements or about §1.1 being the
> blocker.

Measured, 32 MB per launch:

| | bandwidth-bound | compute-bound |
| --- | --- | --- |
| CPU arch, host tensors | **1.09 ms** | 166.05 ms |
| Metal, torch MPS tensors | 58.01 ms (**53x worse**) | **7.45 ms** (22x better) |
| Metal, torch host tensors | 19.82 ms | **3.18 ms** (52x better) |
| Metal, **Taichi-owned ndarray** | **1.25 ms** | — |

Unlike the other two, this one has a fix (§3.3): a Taichi-owned `ti.ndarray` is
bound by device allocation with no copy-back callback, and is **46x faster than
the torch-MPS path on the same backend**.

---

## 2. Everything else the probe established

### 2.1 Which backend

`ti.gpu` resolves to **metal**, not vulkan — `ti.gpu` is
`[cuda, metal, vulkan, ...]` and metal comes first. The `test.yaml` comment
saying it "resolves through Vulkan" is wrong; the f64 error it saw comes from the
SPIR-V codegen that Metal also uses.

### 2.2 Vulkan is not an escape, on either side

Forcing `TI_ARCH=vulkan` (verified to have taken effect — the arm reports
`vulkan`) stages torch tensors exactly like Metal: 54.93 ms with MPS tensors
against Metal's 58.01, and 26.62 ms with host tensors against Metal's 19.82.
Same order, no escape, as §1.3's missing `import_memory` predicts.

And torch has no Vulkan compute device to pair with it: `is_vulkan_available()`
is False and `torch.zeros(4, device="vulkan")` raises `NotImplementedError`.
Torch's Vulkan backend is an Android mobile-inference path.

### 2.3 Torch op coverage on MPS

4 of the 25 ops the raster path calls fail, with `PYTORCH_ENABLE_MPS_FALLBACK`
cleared so a gap is loud:

| op | failure |
| --- | --- |
| `torch.zeros(dtype=float64)` | `TypeError` |
| `scatter_reduce_(reduce="amin")` on int64 | `RuntimeError` |
| `scatter_reduce_(reduce="amax")` on int64 | `RuntimeError` |
| `cummax` f32 | `NotImplementedError` |

The amin/amax gap is not incidental: `_one_mesh_pixel_caps`
(`raster_pipeline.py:1330-1332`) uses exactly those two for the per-pixel surface
id spread. Everything else **ran without raising**, including int64
`scatter_add_`, 64-bit shifts, `view(torch.uint8)`, `argsort`/`sort` on int64,
`unique_consecutive`, `bincount` — which is a weaker statement than it looks, and
§2.3b is what the difference cost.

### 2.3b Some int64 ops do not raise, they answer wrongly

The table above asks whether an op is *implemented*. It is not the same question
as whether it is *right*, and the ops that answer wrongly are worse than the ones
that fail: a `RuntimeError` names itself, and a silently truncated integer key
arrives as a picture nobody can explain.

`benchmarks/_mps_torch_op_probe.py` asks the second question — every op the
compaction uses, run on MPS and on the CPU over the same input in one process, at
the pipeline's real dtypes and magnitudes. Two ops disagree: `index_select` (and
`torch.gather`) over integer **values**, and `//` on int64.

**The gather round-trips integers through a float32.** That is not an
impression, it is what the returned values are. The probe builds the values on
the host, proves the move to the device is bit-exact, and only then gathers, so
what it catches is the gather alone:

| width | int32 | int64 |
| --- | --- | --- |
| 2**16 … 2**24 | exact | exact |
| 2**25 | **wrong** | **wrong** |
| 2**30 / 2**40 / 2**62 | **wrong** | **wrong** |

The boundary is 2**24 and it is *the same for both widths*, which already says
this is not about int64. And every value that comes back is exactly
`float32(correct_value)`, round-to-nearest:

| correct | MPS returned | `float32(correct)` |
| --- | --- | --- |
| 18271053 | 18271052 | 18271052 |
| 756440460 | 756440448 | 756440448 |
| 976314890686 | 976314892288 | 976314892288 |
| 3314435950399956755 | 3314436020488896512 | 3314436020488896512 |

**It is a torch dispatch defect, not a Metal limit.** In the same run, over the
same values at 2**40, `index_select` and `torch.gather` are wrong while
**advanced indexing `v[i]` is exact**, and so is a `repeat_interleave` slice
(`torch.take` is not implemented on MPS at all). The hardware moves those bits
correctly through one aten path and not through another. Nothing here appears
in any documentation we found; it was located by measurement, and it is worth an
upstream report.

Storing, moving, comparing, shifting, masking and multiply-add are exact at
every width. The index dtype is fine too: `index_select` over **float32 values**
with an int64 *index* is exact.

Both of Algan's wide int64s are composite keys, so both were hit:

* **The fragment key**, `pixel << 32 | bit_cast(depth)` (`raster_taichi.py:2039`),
  about 2**50 at 1080p. A 25-bit gather destroys bits 25..0, which masks the low
  word with `0xFC000000` — and every float32 in `[4, 8)` then decodes as
  **exactly 2.0**. That is not a story told after the fact: the arithmetic gives
  2.0 for 5.317023, 6.0 and 7.723102 alike, and the Apple GPU reported `depth
  min=2.000000 max=2.000000 distinct=1` where the CPU has `5.317023 .. 7.723102,
  distinct=37899`. Fixed by `mps_compat.gather_packed_key`, which gathers the two
  32-bit words and repacks.
* **The shading-class key**, `band * _SHADE_CLASS_BASE + cls` with a base of
  `1 << 25` (`sheets.py` §4.4), about 2**40 for a frame with 40956 bands. Here it
  is `unique` that merges rows differing only below the ceiling. Isolated by
  re-running the same compaction with the split off — 40956 sheets without it,
  128 with it — and fixed by `mps_compat.band_class_groups`, which groups the
  pairs by a two-pass stable sort instead of forming the product.

The general rule this leaves: **on MPS, do not put a value above 2**24 through a
gather, a division or a `unique`, at any integer width.** Composite keys are
where a renderer builds such values, and they are exactly the values whose low
bits carry the meaning.

`mps_compat.gather_packed_key` gathers with `v[i]`, which is one operation and
exact. Two earlier forms are worth knowing about because they bound the problem:
splitting the key into two 32-bit halves does **not** work (a half still reaches
2\*\*32, and the render that shipped it came back with the low word rounded from
`40e68475` to `40e68480` and its distinct depths down from 37899 to 22292), while
four 16-bit lanes does work at four gathers instead of one.

What `v[i]` costs is a **dependency on a dispatch**: nothing in torch's API
promises it keeps routing to a kernel that is exact here, and if it stops, every
Algan render on an Apple GPU goes quietly wrong. So the dependency is guarded
rather than assumed —
`test_mps_friendly.py::test_advanced_indexing_is_exact_above_the_mps_ceiling`
gathers past the ceiling on MPS whenever the machine has one, and fails loudly
if the answer changes, naming the lane split as the fallback. It selects its
device from `torch.backends.mps.is_available()` rather than from Algan's render
device, so it guards on any Apple machine including one without the patched
Taichi; on everything else it degenerates to a correctness check of the helper,
which is why a green Linux run does not clear it.

### 2.4 How non-deterministic MPS actually is

400k fragments reduced into 5k segments, six runs:

| | f32 `scatter_add_` | fixed-point i64 |
| --- | --- | --- |
| CPU | bit-identical | bit-identical |
| MPS | **not identical, spread 1.14e-05 = 192 f32 ulps at 1.0** | bit-identical |

(A second run measured 256 ulps — it varies, which is the point.) For scale,
`raster_pipeline.py:1345-1364` records that CUDA's unordered float atomics moved
two consecutive renders by up to 28 channel values over 9.6% of a frame, and that
is why the f64 accumulator exists. 192 ulps is a larger disturbance into the same
threshold.

---

## 3. What to do

### 3.1 Do now, regardless: stop offering a device that cannot render

Independent of everything above, and worth doing whether or not MPS is ever
supported. `auto` currently selects a device that fails 88 tests.

* `_startup._auto_render_device()` should not return MPS.
* An explicit `mps` should raise `AlganConfigurationError` from `coerce_device`
  naming the reason, instead of failing deep inside the renderer.
* `cli.py:45-46` prints "Apple Silicon MPS acceleration available", which is not
  true.
* Then the `test.yaml` macOS pin is redundant — the runner resolves to CPU on its
  own — and that 15-line comment becomes a pointer here.

### 3.2 Two engine bugs this turned up — **fixed**

Both are corrected in `taichi_runtime.py`, with the truth tables pinned by
`tests/unit_tests/test_taichi_launch_pairing.py`. Neither fix moves CPU or CUDA
behaviour: on those the new answers are the old ones for every reachable
pairing, and they differ only where the old rule was wrong.

* **`taichi_launch_is_local` was wrong for MPS.** It compared device *types*, so
  it answered `True` for an MPS tensor on a Metal arch — confirmed live by the
  probe — while Taichi copied through the host, and the gate it feeds turns the
  PN criterion kernels on precisely when a launch is free. The rule is not about
  device equality: Taichi implements `Device::import_memory` for `CpuDevice` and
  `CudaDevice` and for nothing else, so a torch allocation binds without copying
  only as a host tensor on a CPU arch or a CUDA tensor on a CUDA arch. Every
  other pairing stages, including one whose two halves name the same physical
  GPU. A `taichi_arch_is_cuda` companion to `taichi_arch_is_cpu` carries the
  second half of that.
* **`_arch_matches_render_device` could not switch between two GPU backends.**
  It compared `live != ti.cpu`, making every GPU backend interchangeable, so a
  render device moving between two of them kept whichever program was already up
  and launched every kernel on the wrong device with no re-init and no error —
  its docstring claimed to rule that case out while the code was what allowed
  it. It now compares against the backends that actually serve the device, from
  a written-out mapping rather than through `adaptive_arch_select`: resolving
  `ti.gpu` means probing Vulkan and OpenGL, which is the fallback chain
  `_taichi_arch` exists to avoid and which some headless configurations crash
  inside rather than declining.

The second is also what silently defeated this probe's own first attempt to
force Vulkan, which is how it was found.

### 3.3 If Apple GPU support is ever a priority, this is its shape

Not a port. In dependency order:

1. **Taichi-owned ndarrays instead of torch tensors** at the kernel boundary.
   Worth doing on its own merits — 46x on Metal, and on CUDA it removes the
   fast-launch path's staging concerns. The cost is that Taichi 1.7.4 offers
   `ScalarNdarray` only `from_numpy`/`to_numpy`, so data crossing from torch
   still goes through the host: measured 6.48 ms in / 15.63 ms out for 16 MB on
   Metal. That relocates the copy from per-launch to per-crossing, which pays for
   a sub-pipeline that launches many times between handoffs — the wavefront
   tracer, not the raster/sheet stages. (A patched Taichi would remove the
   per-crossing copy too, by importing torch's own `MTLBuffer` — see
   `DESIGN_mps_zero_copy.md`. It needs a forked wheel and it is worth doing
   *after* step 2, not before: packing is what shrinks the patch.)
2. **Kernel argument packing** so no kernel binds more than ~24 buffers. This is
   the large one and there is no way around it (§1.1). **Done**: every kernel
   now takes its buffers as views of the render arena, and none binds more than
   24. Written against the table in §1.1 rather than against a machine.
3. **f64 and i64 atomics out of the kernels**, replaced by f32 with a documented
   non-deterministic mode, since the deterministic fixed-point form cannot run in
   a Metal kernel (§1.2). **Done**: MPS-friendly mode, per §1.2's amendment.

Steps 2 and 3 are therefore code, and both are now **verified on an Apple GPU**:
with MPS-friendly mode on, every kernel in the renderer compiles on Metal and a
render runs end to end, 53-66 s for one 864x486 frame. The `render` job in
`.github/workflows/mps_probe.yaml` is what asks —
`ALGAN_RENDER_DEVICE=mps` on `macos-latest`, the codegen probe, one smoke frame
and then `tests/unit_tests tests/fast`, beside a Linux control arm that forces
the mode on over a CPU device so that a two-arm failure separates "the mode is
broken" from "what is left is Metal".

**Step 1 is now the blocker rather than the optimization**, which is the one
thing this document had the wrong way round. §1.3 was written up as a
performance finding with a fix worth doing "on its own merits"; §1.3b is the
same staging producing a *wrong answer*, and the frame is black until it is
addressed. Its option 1 -- packing each kernel's arena arguments into a
per-dtype staging buffer -- is the small end of step 1 and is where to start.

`test.yaml`'s macOS pin to `ALGAN_RENDER_DEVICE=cpu` stays until the render job
is green, which it is not: it gets through kernel compilation and dies on the
smoke frame's blackness check.

The payoff even then is compute-bound scenes only: 52x on the path tracer,
53x *worse* on the bandwidth-bound raster stages unless step 1 lands first.

**The right trigger to revisit** is upstream: Taichi exposing torch-MPS interop
to the Python frontend (the backend primitive already exists — see
`DESIGN_mps_zero_copy.md` — so this is a software limit rather than a hardware
one), or Metal's argument limit ceasing to bind through argument buffers.
Re-running the probe answers both in about three minutes.

### 3.4 What Mac users should be told meanwhile

The CPU path is the Mac path. It is the same arch CI exercises, so effort spent
making it faster pays twice.
