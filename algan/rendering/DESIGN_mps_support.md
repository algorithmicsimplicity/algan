# Algan on Apple GPUs (MPS / Metal): measured verdict

Source status (2026-09-09): **the MPS port is implemented and the test manifest
contains one strict expected failure**, for `ti.real_func` early return. The
glossy-prefilter entry was removed because it never reproduced on master (§4.5).
This is a source check, not a claim about the latest live CI conclusion. `.github/workflows/test.yaml` carries an ordinary `macos-latest`
`render=mps` arm on every gated PR and push: it pins `ALGAN_RENDER_DEVICE=mps`,
fails before pytest unless Algan's own startup resolver returns `mps`, and runs
`tests/unit_tests` plus `tests/fast` against the locked published
`algan-quadrants` — no private wheel anywhere in the path. Historical counts and their branch/commit identities are recorded in §4.
`tests/mps_known_failures.py` is the current exception list. Its strict xfail
runs rather than skipping; an unexpected pass fails the gate so the entry
cannot silently outlive the problem.

**§4 is the scoreboard**: what the arm reports, which failures have been
diagnosed, and which are still open. Read it before quoting a support claim
from this document, and distinguish a recorded run from the current branch. Earlier sections
also contain measurements, but their intermediate verdicts are superseded.

`mps_probe.yaml` remains the diagnostic workflow. It is no longer the only
place Apple-GPU regressions are exercised, which was the point of building the
arm.

The rest of this document is the measurement history that got the port here.
Its early NO-GO language is a superseded verdict, kept because the reasoning
is what the next capability question will be answered with.

Historical starting status (superseded): **NO-GO on the port as such.**
Measured on GitHub's `macos-latest` runner
(Apple Silicon, macOS 26.5.2 arm64, torch 2.7.1, taichi 1.7.4) by
`benchmarks/_mps_capability_probe.py`, run from `../../.github/workflows/mps_probe.yaml`.

Historically, the macOS CI job was pinned to `ALGAN_RENDER_DEVICE=cpu` because
MPS renders failed.
The workflow comment attributes that to `float64` and says supporting MPS "means
taking float64 out of the raster pipeline and the kernels". **That is true and it
is the least of it.** Three independent Metal limits block the port, f64 is the
smallest, and the one that decides the question is the kernel argument limit,
which nobody had looked at.

Read §1 for the original blocker analysis, §2 for the measurements behind it,
and the later status notes for how those blockers were cleared.

**Status, added later.** All three blockers are cleared and *measured cleared on
the hardware*: §1.1 by packing kernel arguments into arena offsets, §1.2 by
MPS-friendly mode, and §1.3/§1.3b by the forked Taichi in `../../taichi_patches`,
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

**Historical intermediate status.** At that point §1.2c remained and the macOS
suite was at **1 failed, 2425 passed, 167 skipped**; the Linux control arm,
running the same suite with
MPS-friendly mode forced on over a CPU render device, is **fully green**, which
is what says the mode itself is sound and the remainder is Metal.

The six that were §1.2c are fixed, and they were one defect: a `continue` under
a `ti.static` gate, which emits a block Taichi never reopens and so produces
invalid SPIR-V. It is Algan's kernel rather than Metal's limit, it has been in
`sheet_resolve_shade`'s shadow variant the whole time, and LLVM's indifference
to it is why CPU and CUDA never said so. §1.2c has the mechanism and the
Linux-only reproduction that found it.

The seventh (`test_closed_shell_attenuates_once_at_authored_opacity`) is
**cleared, and the macOS suite is green: 2447 passed, 167 skipped, 0 failed.**
It was the renderer's divide-by-zero guard, `band_area.clamp_min(1e-12)` in
`sheets._sibling_weights`, handing back a floor the division did not survive: a
band of zero area produced a NaN, and a NaN never trips the resolve's `eff <=
min_alpha` branch — comparisons against one are all false — so a sheet that
should have dropped out composited and one interior column came back attenuated
twice. Respelling the floor as `where(x < eps, eps, x)`, which is `clamp_min`
bit for bit and does not go through the call MPS mis-handles, fixes it: the
closed shell's sheets now read `[+1.00000, -0.00000, +0.00000]`, the CPU's
values sign included.

**§2.3c is the measurement, and it is only half-explained.** The sweep pins a
real defect — MPS rounds `clamp_min`'s scalar bound through float16, so every
floor below 5.96e-8 becomes 5.96e-8 — but that defect alone does not produce
the NaN the render showed, and §2.3c says so rather than papering over it.

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

For contrast, Taichi's *own* ceiling is a Python-side counter (`max_arg_num =
64` in 1.7's `kernel_impl.py`; Quadrants' `MAX_ARG_NUM` is 512), not a codegen
limit, and it reports it politely on both backends: `The number of elements in
kernel arguments is too big! Do not exceed 64 on metal backend.` The Metal limit
is a third of that and announces itself with an abort.

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
> `mps_compat.py`. f64 accumulators become f32 —
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
> each other, and it walks the AST of `` so a new f64
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

### 1.2c `sheet_resolve_shade_arena` will not compile for some scenes — FOUND

> **Resolved, and neither hypothesis below was right.** It is not a size limit
> and not §1.2b's cast: `sheet_resolve_shade_arena`'s **shadow variant is
> invalid SPIR-V**, and the invalid instruction is one Algan's own kernel asks
> for.
>
> ```
> error: line 301768: Load must appear in a block
>   %tmp112365_u1 = OpLoad %bool %tmp670_unknown
> ```
>
> `if ti.static(mode == 1): continue` (`sheet_resolve_taichi.py`, the end of
> the sheet walk) is the source. `ti.static` is resolved by Taichi's AST
> transformer, which emits the branch's statements inline and leaves **no
> IfStmt at all** — so what reaches the SPIR-V codegen is a bare `ContinueStmt`
> in the loop body followed by the whole rest of that body.
> `visit(ContinueStmt)` emits an `OpBranch` and sets `gen_label_`, relying on
> the *next IfStmt boundary* to open a new `OpLabel`; with no boundary left to
> reach, every statement after it is emitted into a block that has already been
> terminated.
>
> That is why it is scene-dependent: only `mode == 1`, the shadow event build,
> takes that gate, so only a render with shadows on compiles the broken
> variant. All six failures are shadow renders (the audit's `algan_render.py`
> defaults `shadows=True`, which is what put the three glossy-prefilter arms in
> the list).
>
> And it is why nothing said so. LLVM does not mind, so CPU and CUDA have been
> rendering it correctly for as long as it existed; spirv-opt declines to
> optimize (`SPIRV optimization failed`) and Taichi carries on with the
> unoptimized module; Metal's `spirv_cross::CompilerMSL` **constructor** throws
> while parsing it, which lands in `MetalDevice::create_pipeline`'s catch — the
> one path that discarded the exception text and returned `success` with a null
> pipeline (fixed in `taichi_patches/0002`).
>
> **Both halves are fixed.** The kernel gates the rest of the walk with
> `if ti.static(mode != 1):` instead of skipping it with a `continue`, which is
> the shape it wanted anyway; and `0002` opens a new block after every
> `continue` rather than only after one that ends an if-branch, so a stray
> `ti.static`-gated `continue` anywhere else cannot do this again.
>
> **Measured on Linux**, which is the other half of the finding: Taichi's
> Metal and Vulkan backends share the SPIR-V codegen, so a software Vulkan
> (`mesa-vulkan-drivers`' llvmpipe) reproduces it on a machine with no Apple
> GPU — `TI_ARCH=vulkan` on the same test gives the same warning, at the same
> launch site, and `vkCreateComputePipelines failed` where Metal gives a nil
> pipeline. Extracting the module from Taichi's offline cache and running
> `spirv-val` on it is what produced the error above; after the fix `spirv-val`
> reports nothing. **That was the loop to use for the next codegen question**,
> and it saved a 30-minute round trip through the Apple runner.
>
> **It does not work on the published Quadrants wheel**, which is worth knowing
> before planning a round against it. Measured 2026-09-08 on the Linux
> `algan-quadrants==1.3.0.post2` wheel with `mesa-vulkan-drivers` installed and
> `vulkaninfo` reporting an llvmpipe device: `qd.init(qd.vulkan)` answers
> `Arch=[Arch.vulkan] is not supported, falling back to CPU`, and
> `quadrants._lib.core.with_vulkan()` is **False** — the distribution is built
> without the Vulkan backend, so it carries no SPIR-V codegen to reproduce
> against. Taichi's wheel had one, which is what made the loop above possible.
> Restoring it means a Vulkan-enabled Quadrants build; until there is one, a
> codegen question costs an Apple runner. The check is one line and needs no
> probe: `python -c "from quadrants._lib import core; print(core.with_vulkan())"`.

What follows is how the section read while it was open. Kept, because the two
readings it got wrong are the ones anyone re-treading this path will get wrong
in the same order.

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

A `started` with **no matching `completed`** — which was itself a misreading,
and the first thing the next trace corrected: the compile *does* complete
(`frontend=2.371s, backend=5.510s`), and the abort lands after it, at the
pipeline creation a launch does. So it is §1.1's 49-argument
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

The seventh failure is unrelated, and it was **§2.3c** — a third torch op that
answers wrongly on MPS rather than failing.
`test_closed_shell_attenuates_once_at_authored_opacity` had the path-traced and
deterministic routes agreeing everywhere except one column of the interior,
where they differed by 86. It **passes on the CPU in both modes**, so it was
backend-specific rather than an MPS-friendly substitution. The localization
below is kept because it is what narrowed a whole frame to one expression; the
answer is at the end of it.

> **Localized**, by `benchmarks/_mps_closed_shell_probe.py`, which renders both
> routes and prints them side by side over a window wider than the assertion's:
>
> ```
> path traced (8 spp)   : interior mean  153.00 min 153 max 153   -- the oracle
> det, ceiling kernel   : interior mean  159.57 min 153 max 239
> columns > 2 across the window's rows: [21, 37, 44]   (CPU: [21, 44])
> ```
>
> Three facts fall out. **The path tracer is exactly right** — 153 is
> `0.6 * 255`, the authored opacity, uniform. **It is the deterministic route
> that moved**, which the assertion's wording hides. And the disagreement is at
> **column 37 only**: 21 and 44 are the cube's two silhouette edges and are
> there on the CPU too, while 37 is inside the window and its neighbours 38..43
> are clean — so this is not an edge the window clips, it is an isolated
> interior column.
>
> 239 is not a wobble, it is a specific composite: `1 - 0.4**3 = 0.936`, and
> `0.936 * 255 = 238.7`. That is **three** shell crossings each compositing at
> full coverage, where the closed-shell ceiling should have collapsed them to
> one attenuation of 0.6. So the question is why the ceiling
> (`sheet_compact_taichi.solid_shell_ceiling`) does not group those fragments
> into one segment on this backend — its segment key is
> `pix * K + shell_sid`, the arm that spends the allowance differences a global
> exclusive prefix, and in MPS-friendly mode that prefix is an **f32** cumsum
> rather than f64 (the kernel's own docstring says the exactness argument
> lapses there).
>
> **It is not the mode's non-determinism**, which was the first thing to rule
> out and the cheapest: §1.2's amendment predicts this symptom in as many words
> ("a ceiling that wobbles in its low bits flipping borderline fragments in and
> out of being clipped"), so the probe renders the deterministic arm twice in
> one process and compares. On the Apple GPU the two frames are **bit-identical
> over the whole frame**. There is a fixed wrong answer here, not a wobble.
>
> Two more discriminators, one of which does not exist:
>
> * **The ceiling's other arm cannot run on MPS.** `solid_shell_ceiling`'s
>   torch fallback calls `index_copy_`, and torch has not implemented
>   `aten::index_copy.out` for the MPS device, so the A/B that would separate
>   "the kernel" from "what the kernel was handed" raises there. (The probe now
>   runs it last and tolerates the failure; before that it took every other
>   reading down with it.)
> * **The ceiling switched off entirely** does run, and it is the reading that
>   places the defect. With `solid_shell_alpha=False` the Apple GPU's interior
>   is `mean 214.00 min 214 max 214`, uniform, **with no outlier at column 37**
>   — exactly `0.6 * (2 - 0.6) * 255`, which is two crossings compositing at
>   every pixel in the window. So the compaction hands the ceiling the same two
>   sheets there as anywhere else, and **the third layer is created by the
>   ceiling itself**.
>
> That last point is sharper than it first reads, and it rules out the obvious
> shape of the bug. `solid_shell_ceiling` can only ever *reduce* a coverage —
> it writes `denom * scale` with `scale <= 1` — so no clamp it makes could turn
> the ceiling-off composite of 214 into a *larger* 239. The extra layer cannot
> be "the second crossing was under-clamped". It has to be a third sheet, and
> the ceiling is upstream of sheet formation: the kernel clamps `cov_o` **in
> place**, and the compaction's band areas, shading-class aggregates and
> dominant-fragment choice all read that clamped copy afterwards (which is what
> the block comment above it says it is for). A wrong coverage written at one
> pixel therefore changes how its fragments group into sheets.
>
> **That measurement cleared the kernel.** The probe recomputes
> `solid_shell_ceiling`'s output from its own inputs in float64 on the host, and
> over 1305 fragments **no element differs by more than 1.2e-07** on either
> device — so the ceiling computes the right answer on MPS. What it also
> reports is the fact that matters: of those fragments, **604 are clamped to
> zero**. That is the ceiling's job, and it is the input the code below it was
> not written for.
>
> Dumping the sheets at the offending pixel says the rest. Column 37 carries
> **three** sheets on both devices — the cube's near vertical edge — and where
> the CPU has coverages `[+1.00000, -0.00000, +0.00000]`, the Apple GPU has
> `[+1.00000, nan, nan]`. A NaN coverage never trips the resolve's `eff <=
> min_alpha` branch, because every comparison against a NaN is false, so the two
> sheets the CPU drops both composite: `1 - 0.4**3`, the 239 above.
>
> Wrapping the three functions between the ceiling and the sheet record named
> the one that makes it — inputs all finite, output not — and recomputing that
> function's intermediates named the expression:
>
> ```
> band_area gathered: 0 non-finite
> clamp_min(1e-12):   0 non-finite
> share = cov / clamped: 474 non-finite
>   row 10: band 8 cov 0 area 0 clamped 0 corr 0 share nan p nan pop 3
> ```
>
> `clamped 0` is the reading the fix was built on: `sheets._sibling_weights`
> guards its divide with `band_area.clamp_min(1e-12)`, and here that returned
> the exact zero, so a band whose siblings all contributed zero coverage —
> which is precisely what the ceiling's 604 clamps create — divided `0 / 0`.
> Respelling the floor clears it, and the suite is green.
>
> **The synthetic sweep does not reproduce that reading**, and §2.3c keeps the
> discrepancy visible rather than resolving it by assertion: on the same
> machine `clamp_min(0.0, 1e-12)` returns 5.96e-8, not 0, and `0 / 5.96e-8` is
> 0. So the fp16 rounding §2.3c measures is a real defect but not, on its own,
> this NaN's cause. What is established is where the NaN was made and that the
> respelling removes it.
>
> The obvious candidate for the difference has been tested and is **not** it:
> the render's tensor is the output of an `index_select` computed on the device
> rather than a `full()` moved onto it, and §2.3b's whole finding was one gather
> answering differently from another, so the probe clamps a gathered tensor too.
> It still returns 5.96e-8, and its `0 / clamped` is finite. What that round
> *did* turn up is that the defect is **dispatch-dependent**: clamping a tensor
> a `scatter_add_` produced returns the honest `1e-12`, where `index_select` and
> advanced indexing both return 5.96e-8. So no single measurement of this op
> generalises, which is a reason to route every call site through one helper
> rather than to reason about which of them is safe.

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
   plus `mps_zero_copy.py`. It removes staging altogether and is
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

### 1.4 A dense scene dies at frame 119 of 179, silently, on both compilers — FIXED

**Resolved 2026-09-04, and it was a leak, not a limit.** `mps_zero_copy`'s
import cache pins a torch storage per buffer it has handed a kernel (it has to:
the imported ndarray keeps no reference, so nothing else stops the caching
allocator recycling a buffer under a live kernel), and it was released once, at
the end of the render job. The arena is one storage the job reuses, but every
other kernel argument — the uploaded scene arrays, the BVH nodes, the wavefront
queues — is a fresh allocation per batch, and each stayed pinned until the last
frame. Torch's *live* MPS bytes rose monotonically, 0.64 GB at batch 0:8 to
6.74 GB at batch 120:128, on a runner with 7 GB and no swap; host RSS never
passed 1.2 GB, which is why every host-side reading missed it and why capping
the window changed nothing. Neither compiler is involved, which is why both
died at the same frame.

The fix (`release_torch_memory` now clears the cache immediately before
`torch.mps.empty_cache()`, the two belonging together since a storage the cache
holds is precisely one `empty_cache` cannot reclaim) is measured by
`scripts/gate/mps_crash_diagnose.sh`, run 33851991845 on the Mac runner
(Taichi 1.7.4, patched wheel): the leak arm, with the defect reinstated, exits 1
at a 7.48 GB live-MPS peak, killed by Python running out of memory; the fixed
arm renders all 179 frames in 425 s with the peak flat at 0.64 GB across every
batch. `cache_stats()` reports what the cache holds, and the benchmark prints
it as `pinned=` beside `mps_alloc=` so the claim reads off a run. The cost is a
re-import per buffer per drain, and the Quadrants launch-context cache's hit
rate, since that cache keys on argument identity and a re-imported wrapper is
a new object — a fair trade against a render that cannot finish.

A second hole closed on the way (`b51966f`): the two budgets bounding a frame
window's cost outside the arena answered `float("inf")` for any device that
was not CUDA or CPU, which made every guard they feed unsatisfiable on Metal.
Both now read one `_render_device_pool_bytes`. It is not the 1.4 fix — the
windows and the peak are unchanged by it — but an unknown device now gets a
ceiling rather than a free pass.

The account as it stood before the fix follows, because the reasoning in it is
what led to the measurement.

**Measured 2026-09-04** on the Mac runner (`scripts/gate/mps_vs_cpu_ab.sh`,
`GATE_SCENE=materials_and_lighting`). Rendering that scene on Metal dies with
`Trace/BPT trap: 5` after `Rendering 119/179 frames (66%)` and prints **nothing**
— no Python traceback, no `QD_ASSERT`, no Metal diagnostic, and no `.ips` crash
report, which points at the process being killed rather than faulting. The CPU
arm of the same scene on the same machine completes in 198 s.

**It is not a compiler or a fork problem.** The patched Taichi 1.7.4 wheel and a
patched Quadrants 1.3.0 wheel fail *identically*: same scene, same hardware, same
signal, same frame. Stopping at the same frame on both says deterministic rather
than a race.

The first place to look is memory. Both arms log `Prepared batch does not fit the
render arena; binary-searching the largest fitting runtime` **twice** before
dying, and the runner is a virtualized M1 with 7 GB shared between host and GPU.
That makes a Metal allocation failure or a jetsam kill the leading hypothesis,
and neither would print anything through Algan.

Why it was never seen before: nothing renders a dense scene on Metal.
`tests/full_renders` renders all six scenes on a Mac and **skips every
comparison** (`tests/full_renders/test_full_renders.py`), the MPS probe renders
`Square`s, and `gpu_smoke.py` renders one moving square. This was found only
because the Quadrants migration needed a Metal-versus-CPU pixel comparison and
picked the heaviest scene to make it.

What it cost until the fix: **a Mac user could not render
`materials_and_lighting`, or anything of that weight, on either compiler.**
`scripts/gate/mps_crash_diagnose.sh` reinstates the defect on demand
(`--leak-cache`) and is how the fix is re-checked; the arm log lands in
`gate-logs/`.

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
id spread. Nor is the failure clean, as torch 2.13.0 showed (§4.11, T): the op
flips every sign bit of its output before it raises, and after a second failure
the process can compile no new kernel. So the renderer reduces through
`mps_compat.scatter_reduce_`, which computes an int64 extremum from two int32
ones. Everything else **ran without raising**, including int64
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
bits carry the meaning. So are bitmasks with a flag above bit 24: a fragment's
mask word became one on 2026-10-30, when the near cut (bit 29) and the one-solid
ceiling (bit 30) gave it flags there, and every `index_select` of it went wrong
on this arm (§4.9).

On torch 2.13.0, the version the MPS arm pins, the same probe found
`torch.gather` exact (run 42226130722); `index_select` still rounds, at int32
and int64 alike, and so does `repeat_interleave(values, counts)`, which torch
builds on it, while the counts-only form is exact (run 42305385094). That
version's `//`, `%`, floor `div` and `remainder` came back exact past 2**24 at
both widths, and `torch.unique` still merged distinct values (run
42329686918): asked for an inverse, or given a `dim`, it works on
`float32(int32(x))`, while its values-only and counts-only forms are exact
(§4.11, U). The rule stands either way: `mps_compat.gather_exact` is the
gather for such a value, `mps_compat.repeat_interleave_exact` the repeat
(§4.10), and `mps_compat.unique_exact` the `unique` (§4.11).

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

### 2.3c `clamp_min` rounds its floor through float16

The third op in this family, and the one that took the longest to find, because
unlike the two above it does not lose bits of a value — it silently substitutes
a different floor.

`benchmarks/_mps_torch_op_probe.py`'s `probe_epsilon_clamp` sweeps the floor's
magnitude, and the answer is a clean cliff:

| floor asked for | MPS `clamp_min(0.0)` returns |
| --- | --- |
| 1e-3 | 1e-3 |
| 6.1e-5 (f16 smallest normal) | 6.1e-5 |
| 1e-6 | 1e-6 |
| **5.96e-8 (f16 smallest subnormal)** | **5.96e-8** |
| 1e-8 | **5.96e-8** |
| 1e-12 | **5.96e-8** |
| 1.18e-38 (f32 smallest normal) | **5.96e-8** |
| 1.4e-45 (f32 smallest subnormal) | **5.96e-8** |

Every floor at or above float16's smallest subnormal is honoured exactly; every
floor below it comes back as that subnormal. **The scalar bound is rounded
through float16 on its way into the kernel** — that is the defect, and stating
it that way is worth more than "the clamp is wrong", because it says exactly
which call sites are affected and by how much.

Only the three `clamp` spellings are hit. `torch.maximum` against a 0-dim or a
full tensor, and `torch.where(x < eps, eps, x)`, are exact at every magnitude,
so each is a drop-in replacement.

**The exposure is every tiny floor in the renderer, and none of them is a NaN.**
Twenty-odd divisions guard their denominator this way — `sheets.py` at 1e-12,
`raster_pipeline.py` and `refit_bvh.py` at 1e-30, `primitives.py` at 1e-20 — and
on MPS all of them clamp to 5.96e-8 instead. That is a wrong denominator, up to
twenty-two orders of magnitude off the intended floor, and it announces itself
nowhere: it can only matter where the denominator is genuinely at or below the
floor, and it is a bounded perturbation rather than a poisoned value when it
does.

**What is NOT yet explained is the NaN that started this.** The instrumented
render printed `clamped 0` and `share nan` (§1.2c), which requires the clamp to
have returned an exact zero — and the synthetic sweep does not reproduce that:
`0 / clamp_min(0, 1e-12)` is `0 / 5.96e-8` = 0, and the probe's own check of
that composition **passes** on MPS. So two readings from the same machine
disagree, and the reconciling fact is not in hand. What is measured is that the
suite goes green with the respelling below, and that the closed shell's sheets
come back `[+1.00000, -0.00000, +0.00000]` — the CPU's values, sign included.
The fix is confirmed; this mechanism is not, and the difference is worth
keeping visible.

Fixed at the call site rather than in `mps_compat`, because it is not a
substitution — it is the same floor, spelled so that it does not go through the
call that mis-rounds it:

```python
share = cov.to(acc) / torch.where(area_g < 1e-12, 1e-12, area_g)
```

**`where(x < eps, eps, x)` is `clamp_min(eps)` bit for bit**, and that is
checkable rather than asserted: over `0`, `-0`, `±inf`, `NaN`, the subnormals
and values either side of the floor, the two agree in every bit of every
result. The comparison is `<` and not `>` for the one input where the order
matters: a comparison against a NaN is false either way, so only this order
leaves the NaN in the `x` arm, which is where `clamp_min` leaves it. (`where(x >
eps, x, eps)` returns `eps` there instead — the same value everywhere else, and
a silent NaN swallower.)

So there is nothing to gate on the mode and nothing to argue about byte
identity: CPU and CUDA compute what they always did, and MPS now computes it
too. `test_mps_friendly.py::test_a_band_of_zero_area_hands_its_siblings_a_finite_weight`
guards it on whatever device the machine has, like the gather guard in §2.3b.

**All twenty-three render-side sites are converted**, not just the one that was
caught. `mps_compat.clamp_floor(tensor, floor)` is the single spelling, gated on
the mode like everything else in that module — off it, it is literally
`tensor.clamp_min(floor)`, one op, so CPU and CUDA do not pay a compare and a
select for a defect they do not have, and no rendered frame can move.

Converting all of them rather than the one is a judgement about the failure
mode, and the judgement is that this defect's silence earns it. The floors run
1e-12 (`sheets`, `stbvh`, `bezier_acceleration`, `tracer`, `primitives`),
1e-20 (`primitives`) and 1e-30 (`raster_pipeline`, `refit_bvh`, `logical_pn`),
and the error is not confined to a denominator that was already degenerate:
with a 1e-30 floor an ordinary `1e-9` comes back as `5.96e-8`, sixty times
wrong, because the cliff is in the *input* as well as the bound.
`raster_pipeline`'s `n2` is a squared normal length, so a thin triangle lands
squarely in that gap and shades wrongly with nothing to show for it.

A literal bound below the cliff is now banned outright in ``:
`test_mps_friendly.py::test_the_renderer_floors_a_divide_through_clamp_floor`
walks the AST and names any that appear, the way the float64 guard beside it
does. It found one the manual pass had missed — a `torch.clamp(length,
min=1e-12)` in `primitives.py`, spelled differently from the other twenty-two —
which is the argument for having written it. `*_taichi.py` is exempt and must
be: those bodies reach MSL through Taichi and never touch torch's MPS dispatch.

### 2.3f The ceiling again, on a bitmask, in the raster write pass

§2.3b established the 2**24 gather ceiling and fixed the fragment key. This is
the same defect at a **different width and a worse shape**, found by the
in-situ attribution pass rather than by a synthetic probe, which is the point
of having both.

`raster_pipeline.prepare_sparse_raster_coverage` compacts the write pass to the
pairs whose count pass accepted a pixel, and carries the acceptance forward:

```python
live = accepts.nonzero(as_tuple=True)[0]
accepts_w = accepts.index_select(0, live)      # 32 bits, one per chunk pixel
```

`accepts` is a **bitmask**, not a number. Rounding a value that carries
magnitude is a rounding error; rounding a mask is nonsense — `0x8003FFFF` came
back `0x80040000`, the sign bit intact and eighteen bits of acceptance gone.
The write pass replays that mask instead of recomputing the acceptance chain,
so every cleared bit is a pixel the count pass accepted and the write pass then
skipped. Its fragment slot keeps whatever the arena held.

**That is where §4.2's zero keys come from.** A never-written slot reads as
`pixel 0, depth 0, coverage 0`, which sorts to the front of the fragment
stream — the Apple GPU's `pix[0..272664]`, `depth[0.0000..]`, `lo0=00000000`
and `frag_cov min 0.000000` against a CPU that starts at 147239, 18.3139 and
0.001002. It is also the stray lit pixel in the bottom row of
`test_pixel_rows_are_not_flipped`, at image row 35 = kernel row 0 = pixel 0.

The fix is §2.3b's, generalised: `mps_compat.gather_exact` is the ceiling-safe
gather for **any** integer whose low bits carry meaning, `gather_packed_key`
is now that function applied to the key, and the three gathers at this site
take it. `offsets` is included because it is a fragment-stream position and
passes 2**24 on a large frame rather than on this one, and `pairs` because it
carries a per-chunk offset column that grows with a primitive's bbox — neither
is measured wrong, and this says so rather than implying they were.

What the fix rests on is that advanced indexing `v[i]` is exact at int32 with
the sign bit set, which §2.3b measured only at int64. `probe_gather_isolated`
now sweeps real mask patterns at int32 — all-ones, sign-bit-only, either side
of the ceiling — and
`test_mps_friendly.py::test_advanced_indexing_is_exact_for_a_thirty_two_bit_mask`
is the standing guard, beside the int64 one it copies.

### 2.3d `index_reduce_` is not implemented at all

The loud one, and the only defect in this family that announces itself:

```
NotImplementedError: The operator 'aten::index_reduce.out' is not currently
implemented for the MPS device.
```

`refit_bvh._binary_split` reduces per-range centroid extents and per-bin box
unions with `Tensor.index_reduce_(0, seg, src, 'amin'/'amax')`, and that is the
BVH *build* rather than the refit — so it takes down every scene whose topology
is rebuilt, which on the gate's first run was ten path-tracer tests. The
suggested `PYTORCH_ENABLE_MPS_FALLBACK=1` is not the fix: it would move a
whole-BVH reduction to the host on every build, silently, for a gap that has a
device-side spelling.

**`scatter_reduce_` is that spelling**, and it is implemented on MPS at this
dtype and shape. It is the same reduction with the index broadcast to the
source's shape instead of addressing whole rows, so the substitution is a
spelling change and not a numerical one: `amin` and `amax` are exact under any
reduction order — unlike the float sums §6.6.4 widened — so nothing the backend
does to the order can move a value. `probe_index_reduce` measures both against
the CPU's `index_reduce_` over the same input; `mps_compat.index_reduce_` is
the one call site, gated on the mode like everything else here, and
`test_mps_friendly` pins the two arms against each other at both call-site
shapes.

### 2.3e The source-key index was poisoned on every renderer kernel

Not a wrong answer and not a missing op — an optimization silently off, on the
device that needs it most.

`taichi_source_key` skips the compiler frontend when it can describe a launch's
arguments by their type features, and it refuses (`Poison`) rather than guess
at a type it has no rule for. On an Apple GPU **every** converted argument
arrives as `ExternalMetalNdarray`, the patched build's imported-`MTLBuffer`
ndarray (`mps_zero_copy`), which had no rule — so every renderer kernel paid a
full frontend pass on a machine whose frontend passes are the slowest of the
three platforms, and `test_taichi_source_key` failed with **0 keyed**. The
symptom in a log is a wall of

```
RuntimeWarning: taichi_source_key: <kernel> cannot be source-keyed and pays the
full frontend: ndarray argument of type
quadrants.lang._ndarray.ExternalMetalNdarray has no key rule
```

It is an `Ndarray` subclass and reads the same four features as the three
allocated forms (`element_type`, rank, `grad`, `_qd_layout`). What it
deliberately does **not** contribute to the key is the buffer it adopted: a
compiled kernel depends on an argument's element type and rank, never on which
allocation it was bound to, and putting the pointer in the key would turn every
launch into a miss rather than a hit.

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

### 3.1 Do now, regardless: stop offering a device that cannot render — **DONE**

Independent of everything above, and worth doing whether or not MPS is ever
supported. `auto` selected a device that failed 88 tests.

All four are in, and what shipped is finer than what this section asked for:
the answer is not "never MPS" but "MPS exactly when it can render", which is a
property of the *installed compiler* rather than of the platform.

* `_startup._mps_is_usable()` gates both paths on
  `mps_zero_copy.zero_copy_available()` — the patched build's imported-MTLBuffer
  ndarray. `_auto_render_device()` returns MPS only when that answers True, and
  falls back to the CPU otherwise; a stock build is what §1.3b measured drawing
  a black frame, so this is the difference between a render and a wrong picture.
* An explicit `mps` raises `AlganConfigurationError` from `coerce_device`,
  quoting `mps_zero_copy.unavailable_reason()`, which names the compiler this
  process actually bound rather than saying "Taichi" at a Quadrants user.
* `algan_cli._cmd_check` prints the device Algan **resolved**
  (`SETTINGS.computing.render_device`) beside the accelerators that exist, and
  says so when MPS-friendly mode is on. The two are different questions and the
  old line answered only the second.
* The `test.yaml` macOS pin is gone; the matrix has an explicit `render=mps`
  arm instead. §3.3 has where that arm stands.

### 3.2 Two engine bugs this turned up — **fixed**

Both are corrected in `taichi_runtime.py`, with the truth tables pinned by
`../../tests/unit_tests/test_taichi_launch_pairing.py`. Neither fix moves CPU or CUDA
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
`../../.github/workflows/mps_probe.yaml` is what asks —
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

`test.yaml`'s macOS pin to `ALGAN_RENDER_DEVICE=cpu` is gone, replaced by an
explicit `render=mps` arm beside the `render=cpu` one. §4 is where that arm
stands and what it still fails.

The payoff even then is compute-bound scenes only: 52x on the path tracer,
53x *worse* on the bandwidth-bound raster stages unless step 1 lands first.

**The right trigger to revisit** is upstream: Taichi exposing torch-MPS interop
to the Python frontend (the backend primitive already exists — see
`DESIGN_mps_zero_copy.md` — so this is a software limit rather than a hardware
one), or Metal's argument limit ceasing to bind through argument buffers.
Re-running the probe answers both in about three minutes.

### 3.4 What Mac users are told

**The Apple GPU is the Mac path**, and it needs nothing installed for it:
`pip install algan` resolves the published patched `algan-quadrants`, whose
macOS arm64 wheels carry `quadrants_patches/0001`, so `auto` finds an Apple GPU
that can actually render and takes it. `algan check` prints the device that
came out, which is the question a user has (`docs/source/installation.rst`
walks through it).

The CPU path is still there and still supported — it is what `auto` falls back
to when the compiler cannot bind an `MTLBuffer` — so effort spent making it
faster is not wasted. It is no longer the *recommended* Mac path.

---

## 4. The gate's scoreboard

Everything above is a capability question answered on a probe. This section is
the other kind: what the **required** `render=mps` arm of
`.github/workflows/test.yaml` reports when it runs the ordinary portable scope
(`tests/unit_tests tests/fast`) on `macos-latest` against the locked published
`algan-quadrants`. It is written from runs, not from arguments, and it is the
section to update when one changes.

### 4.1 Where it stands

| run | result |
| --- | --- |
| [34102515789](https://github.com/algorithmicsimplicity/algan/actions/runs/34102515789) (2026-09-07, the arm's first) | **32 failed, 3311 passed, 168 skipped, 3 errors** in 2530 s |
| [34203241437](https://github.com/algorithmicsimplicity/algan/actions/runs/34203241437) (2026-09-08, the same scope on the Mac harness) | **9 failing**, 3544 passed, 217 skipped, 9 xfailed in 3149 s |
| [34210549355](https://github.com/algorithmicsimplicity/algan/actions/runs/34210549355) (2026-09-08, after §2.3f, six of the listed files) | six of the seven entries in those files XPASSed |
| [34213125405](https://github.com/algorithmicsimplicity/algan/actions/runs/34213125405) (2026-09-08, the full scope after §2.3f, **on a branch**) | **2 failing**: 1 failed, 3563 passed, 217 skipped, 2 xfailed in 3178 s — and the 1 is the last XPASS |
| [34302218686](https://github.com/algorithmicsimplicity/algan/actions/runs/34302218686) (2026-09-09, master at 2455d08) | **6 failed**, 3655 passed, 226 skipped, 1 xfailed in 2931 s — one stale strict XPASS (§4.5) and five regressions from that day's own Metal work, none of them a device defect (§4.6) |
| [34306711211](https://github.com/algorithmicsimplicity/algan/actions/runs/34306711211) (2026-09-09, the five repaired files and `tests/fast`, on the fix branch) | 92 passed, 4 skipped in 82 s |
| [40306266933](https://github.com/algorithmicsimplicity/algan/actions/runs/40306266933) (2026-10-26, the weekly full run) | **12 failed**, 8472 passed, 360 skipped, 3 xfailed in 4445 s — one a path-tracer bias every arm shared (#247), eleven this arm's own from four causes (§4.8) |
| [41159788488](https://github.com/algorithmicsimplicity/algan/actions/runs/41159788488) (2026-10-31, the four files of those eleven, on the fix branch) | 74 passed, 2 skipped, 4 xfailed in 347 s — the xfails are cause I's entries |
| [41164965493](https://github.com/algorithmicsimplicity/algan/actions/runs/41164965493) (2026-10-31, the caustic files with the walk in a kernel of its own, on the fix branch) | 41 passed, 2 skipped in 313 s — all ten caustic renders pass, cause I's four among them; the skips are the CPU-only comparison of the split with the single kernel |
| [41446470532](https://github.com/algorithmicsimplicity/algan/actions/runs/41446470532) (2026-11-02, the weekly full run, master at 757e0f0) | **8 failed**, 8646 passed, 378 skipped, 3 xfailed in 5425 s — two causes, J and K, both in code added on 2026-10-30 that had not run on this arm (§4.9) |
| [42226130722](https://github.com/algorithmicsimplicity/algan/actions/runs/42226130722) (2026-11-06, `test_sheet_resolve_shading.py` with cause J's gathers exact) | 6 passed — the 344 moved pixels are back |
| [42232166078](https://github.com/algorithmicsimplicity/algan/actions/runs/42232166078) (2026-11-06, `test_sheet_route_former_vetoes.py` with cause J's gathers exact, and the walk test) | 9 passed — the near-plane rim is 31 levels off its reference, as on the CPU, against 58 before |
| [42236806653](https://github.com/algorithmicsimplicity/algan/actions/runs/42236806653) (2026-11-06, `test_stroke_sheets.py` and `test_sheet_chain_kernels.py` with cause K fixed and the split's gathers exact) | 186 passed, 20 skipped — the six `closed_solid` cases among them; a 320x180 frame's split and unsplit compactions agree row for row |
| [42244121369](https://github.com/algorithmicsimplicity/algan/actions/runs/42244121369) (2026-11-06, the full scope with J and K fixed, on the fix branch, #249) | **0 failed**, 8717 passed, 378 skipped, 3 xfailed in 3579 s — the weekly run's 8646 and 8 plus the 63 tests the fixes added; the other three arms green too |
| [42297623070](https://github.com/algorithmicsimplicity/algan/actions/runs/42297623070) (2026-11-06, the stroke split's kernel arm on the GPU, cause O) | an HD frame raises at the split's tally scan, and a layout sweep finds int32 scans along dim 0 of 2-, 3-, 4- and 8-wide tensors missing from torch's metallib; with the tally's rows scanned apart, both split files 198 passed, 20 skipped |
| [42299224973](https://github.com/algorithmicsimplicity/algan/actions/runs/42299224973) (2026-11-06, the Bezier header built on the GPU, cause N) | 13 failed, then 166 passed, 3 skipped with the header's gather exact — a page of glyphs moved 7,962,290 of its 16,835,000 header words, then 0 |
| [42300780142](https://github.com/algorithmicsimplicity/algan/actions/runs/42300780142) (2026-11-06, the shell ceiling's torch arm, cause M) | the segment-prefix test failed, then passed; the render test errored summing an MPS tensor in float64, which the next run's test does on the CPU |
| [42300938376](https://github.com/algorithmicsimplicity/algan/actions/runs/42300938376) (2026-11-06, fragment refs past 2**24, cause L) | 19 failed, then 19 passed — four squares drawn past the 65,536th circuit are 1,197 pixels off by up to 255, then 0; the four related files 256 passed, 12 skipped |
| [42303103954](https://github.com/algorithmicsimplicity/algan/actions/runs/42303103954) (2026-11-06, cause O's scan again) | the 1,023/1,024-row threshold and its C++ stack, and the row scans exact to 5,000,000; then the unfixed split cases hung for 26 minutes in one process (cause P) and the run was cancelled |
| [42303270183](https://github.com/algorithmicsimplicity/algan/actions/runs/42303270183) (2026-11-06, cause M again) | 4 failed, then 4 passed; the four related files 207 passed, 20 skipped |
| [42305385094](https://github.com/algorithmicsimplicity/algan/actions/runs/42305385094) (2026-11-06, cause N's intervals and polylines) | 6 failed, then 211 passed, 3 skipped — 789,747 interval offsets and 1,020,086 vertex numbers moved, then 0 |
| [42308885414](https://github.com/algorithmicsimplicity/algan/actions/runs/42308885414) (2026-11-06, cause O on an M2 Pro) | the split kernel test's 12 MPS cases failed, then passed twice; both split files 198 passed, 20 skipped with the kernel arm on and off |
| [42329686918](https://github.com/algorithmicsimplicity/algan/actions/runs/42329686918) (2026-11-07, pixel ids and stream positions past 2**24, cause Q) | 5 failed, then 5 passed; the six related files 294 passed, 20 skipped; no default render on the runner plans a chunk past 2**24 pixels |
| [42351444222](https://github.com/algorithmicsimplicity/algan/actions/runs/42351444222), [42355723203](https://github.com/algorithmicsimplicity/algan/actions/runs/42355723203), [42371919983](https://github.com/algorithmicsimplicity/algan/actions/runs/42371919983) (2026-11-07, `unique` past 2**24, cause U, and the GPU's extra sheets) | 8,192 values from 2**24 came back as 4,097; the device tests failed, then 46 passed; the related files 222 passed, 10 skipped; the 8 extra sheets trace to an ulp of the camera ray's direction, with fast math on or off |
| [42351612808](https://github.com/algorithmicsimplicity/algan/actions/runs/42351612808), [42353661819](https://github.com/algorithmicsimplicity/algan/actions/runs/42353661819), [42355691483](https://github.com/algorithmicsimplicity/algan/actions/runs/42355691483), [42370277126](https://github.com/algorithmicsimplicity/algan/actions/runs/42370277126), [42372132733](https://github.com/algorithmicsimplicity/algan/actions/runs/42372132733), [42374877006](https://github.com/algorithmicsimplicity/algan/actions/runs/42374877006), [42378512143](https://github.com/algorithmicsimplicity/algan/actions/runs/42378512143), [42381446368](https://github.com/algorithmicsimplicity/algan/actions/runs/42381446368), [42385375749](https://github.com/algorithmicsimplicity/algan/actions/runs/42385375749), [42388655058](https://github.com/algorithmicsimplicity/algan/actions/runs/42388655058) (2026-11-07, the silent out-of-memory, cause X) | §4.10's header build reproduced at 7.26 G, two command buffers failed and 102,317,937 of 121,000,000 words wrong, with nothing raised; held to the working set it raised after 11.7 s at 4.42 G; the hold's tests failed 3, then 105 passed, 2 skipped; the heavy renders peaked the same with the hold on and off, and every frame matched; the unsynced BVH builds and an int32 sum that came back 0 were the same failure, one command buffer whose heaps came within 20 to 30 MiB of the working set; held at 1.0, torch's 1 GiB heaps let 17 buffers into 1.07 working sets and an add over them came back wrong, and with the low watermark at 0.75 and the hold at 0.98 the same buffers stopped at 0.964 and the add was right, with the reference renders' frames unchanged |
| [42352045841](https://github.com/algorithmicsimplicity/algan/actions/runs/42352045841), [42354063716](https://github.com/algorithmicsimplicity/algan/actions/runs/42354063716), [42356233755](https://github.com/algorithmicsimplicity/algan/actions/runs/42356233755), [42370479770](https://github.com/algorithmicsimplicity/algan/actions/runs/42370479770), [42372352869](https://github.com/algorithmicsimplicity/algan/actions/runs/42372352869) (2026-11-07, the BVH builders and the int64 extrema, causes R and T) | `index_select` of 1,048,576 range starts past 2**24 wrong for all; every int64 `amin`/`amax` scatter raised, one case to a process, flipping its output's sign bits, and a second failure left a process that aborted on its next new kernel; the new tests failed on the device, then 15 passed; a root level of 17,825,793 primitives built on the GPU is the CPU's; the fast render's frames matched before and after |
| [42355084729](https://github.com/algorithmicsimplicity/algan/actions/runs/42355084729), [42357151096](https://github.com/algorithmicsimplicity/algan/actions/runs/42357151096) (2026-11-07, the stroke split's denormal area and the sheet areas' order, causes V and W) | the area tests failed 20 of 20, then passed 20 of 20; replays moved 6 to 12 of an HD frame's 168,675 sheets and 283 to 309 of a solids frame's 1,843,088, then none; the split's six MPS weight cases failed, then passed; the five related files 346 passed, 23 skipped |
| [42422403489](https://github.com/algorithmicsimplicity/algan/actions/runs/42422403489), [42424700295](https://github.com/algorithmicsimplicity/algan/actions/runs/42424700295) (2026-11-08, the UHD segfault, cause Y) | a second thread calling `torch.mps.empty_cache()` killed a process running MPSGraph ops 4 times of 4, each within 1.5 s, and the allocator's drain alone, `driver_allocated_memory()` and `set_per_process_memory_fraction` killed none; UHD renders with the batch-prep worker's drain provoked died at `index_select_out_mps +1608` as before, in 5 of 8 processes; with drains kept to the render job's thread none of 7 such processes died, 0 of 21 renders crashed, the race test's child lived 6 times of 6 against 4 deaths of 4, the touched files 148 passed, 1 skipped, and every frame matched |

The first row is the baseline every entry below is measured against, and the
three green arms of that run (Linux 3.10, Linux 3.13, macOS CPU) are what says
the failures are the Apple GPU rather than the change that added the arm. The
second row is the same scope after the fixes in this document: the twelve
"failures" it reports are eleven **strict XPASSes** — this list's own entries,
now passing, which is exactly what strict is for — plus one unrelated
fast-suite curation guard. Nine tests actually fail, and they are
`tests/mps_known_failures.py`'s remaining entries.

**Counting causes rather than tests** is what makes the remainder tractable:
32 failures were six causes, and 23 of the 32 were two of them.

| # | cause | tests then | now | state |
| --- | --- | --- | --- | --- |
| A | `index_reduce_` is unimplemented on MPS | 10 | 0 | **fixed** — §2.3d |
| B | the glossy tile loop walks off its frame table | 13 | 0 | **fixed**: 11 by §4.4, the last 2 by §2.3f |
| C | an unlit/emissive slab renders black | 2 | 0 | **fixed** — §2.3f |
| D | the glossy prefilter loses its reflection | 2 | 0 | one fixed by §2.3f; the other **never reproduced on master** — §4.5 |
| E | the path-traced and deterministic composites disagree by 107 | 1 | 0 | **fixed** — §2.3f |
| F | a `ti.real_func` early return will not compile | 1 | 1 | open — §4.3 |
| G | the source-key index is poisoned on every kernel | 1 | 0 | **fixed** — §2.3e |
| H | one fragment lands in the wrong pixel | 1 | 0 | **fixed** — §2.3f |
| I | the caustic `pt_shade` is too large for Apple's shader compiler | 8 | 0 | **fixed**: the walk runs in a kernel of its own on Metal — §4.8 |
| J | a mask word with a flag above bit 24 goes through `index_select` | 2 | 0 | **fixed**: every such gather is `gather_exact` — §4.9 |
| K | the shell ceiling divides a fragment it does not clip, and Metal's division is not correctly rounded | 6 | 0 | **fixed** — §4.9 |
| L | a fragment's ref past 2**24, from a batch's 65,536th circuit on, goes through `index_select` | latent | 0 | **fixed** — §4.10 |
| M | the closed-shell ceiling's torch arm gathers its keys and rows through `index_select` | latent | 0 | **fixed** — §4.10 |
| N | the Bezier header, its bin intervals and the polylines' run starts go through `index_select` | latent | 0 | **fixed** — §4.10 |
| O | the stroke split's kernel arm scans an int32 layout torch 2.13's metallib lacks | latent | 0 | **fixed** — §4.10 |
| P | the zero-copy cache's weakref callback waits on a lock its own thread holds | a hang | 0 | **fixed** — §4.10 |
| Q | a pixel id or stream position past 2**24, from a 4K chunk's third frame, goes through `index_select` | latent | 0 | **fixed** — §4.10 |
| R | the BVH builders' range positions and ids, and the sliver strips, go through `index_select` and `repeat_interleave` | latent | 0 | **fixed** — §4.11 |
| S | seventeen more ids, in the scene build and the depth-order scan, go through the same two | latent | 0 | **fixed** — §4.11 |
| T | an int64 `scatter_reduce_` `amin` or `amax` flips its output's sign bits as it fails, and a second failure stops the process compiling kernels | latent | 0 | **fixed**: computed from two int32 reductions — §4.11 |
| U | `unique` asked for an inverse merges integers past 2**24 | latent | 0 | **fixed** — §4.11 |
| V | the stroke split's torch arm flushes a denormal area | latent | 0 | **fixed** — §4.11 |
| W | the sheet areas' float32 atomics sum in no fixed order | replay noise | 0 | **fixed**: summed in stream order — §4.11 |
| X | work past Metal's working set fails its command buffer, and nothing raises | latent | 0 | **fixed**: a render holds torch under the working set, and a low watermark keeps its heaps there — §4.11 |
| Y | a drain of torch's MPS caches on the batch-prep thread deletes the cached MPSGraph an op on the render thread is using | a segfault | 0 | **fixed**: drains stay on the render job's thread — §4.12 |
| — | `test_arena_binding_live` cannot size an arena | 3 errors | 0 | **fixed** — §4.4 |

**L to Q failed no weekly run.** L, M, N and Q were found by reading for J's
pattern after §4.9, and O and P by putting the stroke split's kernel arm on
the Apple GPU. Each was made to fail there before it was fixed (§4.10).

**Nor did R to X.** They are what §4.10 saw and did not chase, apart from S,
which reading for R's pattern found. Each but S was made to fail on the Apple
GPU before it was fixed; S's tests emulate the rounding, over ops the GPU
answered exactly (§4.11).

**Nor did Y**, the segfault §4.11 saw in 3 of 40 UHD probe renders and did
not chase. It was made to happen at will on the Apple GPU, and in a
real render, before it was fixed (§4.12).

**Two fixes took thirty of the thirty-two**, and neither was aimed at most of
what it cleared: §4.4's `empty_cache` (the arena was measuring torch's own
cache as occupied) took eleven, and §2.3f's gather (the raster acceptance mask
was losing its low bits) took nineteen — every remaining render failure except
one. B, C, E and H were one corrupted fragment stream wearing four faces, which
is the argument for counting causes rather than tests. It is also the argument
for the xfail list: both fixes were found by measuring the runs it produced,
not by reading code, and neither was predictable from the failure it was chased
from.

**What was left was one test**: the `ti.real_func` compile failure (§4.3), a
compiler defect a layer below Algan that blocks a test rather than a render.
Cause I added four for a while, a feature this backend could not compile
rather than a render it got wrong, until the walk moved to a kernel of its own
(§4.8).
The other entry — the glossy prefilter render whose reflection was absent — is
gone from the list without a fix to point at, because it never reproduced on
master at all (§4.5 has the run table). That is the one way this list can
mislead: an entry measured on a branch is a claim about that branch until a
master run repeats it, and this one never did.

**Observed on the arm and NOT a failure**, recorded so the next reader does not
chase it: the first render of a process warns that `torch.compile` refused
`raster_pipeline._triangle_projection_fused` with `InductorError: KeyError:
torch.float64`, and Algan runs that function eagerly for the rest of the
process. Nothing in it is float64 — MPS-friendly mode narrows every accumulator
— so the likely candidate is a Python float constant Dynamo traces as a float64
the Inductor MPS backend has no entry for. It costs one fused pass and one
warning per process, it is self-limiting, and one refused graph is not enough
evidence to declare `torch.compile` unsupported on the device.

### 4.2 B: the glossy tile loop — **fixed**, and the guard it still wants

Thirteen tests, one line: `tracer.py`'s

```
frame_end = gl_bounds[gl_frame + 1]
IndexError: list index out of range
```

`_gloss_frame_bounds` returns one ordinal per frame boundary and the tile loop
walks them in order, so running past the end means a boundary came back
**smaller than the number of covered pixels** — the loop still had pixels to
place after it had finished the last frame.

**It is not `searchsorted`**, which was the first hypothesis and the cheap one
to rule out: `probe_frame_bounds` in `benchmarks/_mps_torch_op_probe.py`
searches the int32 ordinals the call site actually uses (the int64 sequence
`probe_lookup` searches is a different call site, and the width was the only
difference between them) and MPS agrees with the CPU at 1, 8 and 179 frames.
So the bounds are a correct search of a wrong `covered_idx`, and
`benchmarks/_mps_gloss_bounds_probe.py` is what says which — it renders the
smallest scene that reaches the glossy route and prints the device's ordinals,
their range, whether they ascend, and both bounds.

Worth reading beside it: `test_viewer_fragments.py::test_pixel_rows_are_not_flipped`
fails with the cube's lit rows at 5..15 **plus a single stray pixel in row 35**,
the bottom row of a 36-row frame — one fragment composited at a pixel nothing
should have written. A corrupted covered ordinal explains both that and this,
and if it does then C and D are candidates for the same cause.

**Resolved by §2.3f, and here is the confirmation rather than the inference.**
On the same runner with the acceptance-mask gather fixed, the 32x32 probe scene
reads

```
covered_idx: dtype=torch.int32 n=400 min=198 max=825 ascending=True
device     : [0, 400]      host       : [0, 400]
```

— exactly this project's CPU box, pixel 0 gone and 400 covered pixels where
there were 320. The compaction's input matches too: `n=47610
pix[147239..272664]x23352 depth[18.3139..20.6677]`, `frag_cov min 0.001002`,
against the CPU's identical counts and ranges. Both remaining B entries, both
C entries, H and one of D XPASSed in the same run.

**The eleven that fixed themselves first are still the reading to be careful
with.**
Emptying the MPS cache before sizing the arena (§4.4) gave the render its full
budget, the tile sizing that follows changed, and eleven of these thirteen
stopped tripping the loop. Nothing about the loop was fixed: `gl_frame` can
still walk off the end at whatever window the arithmetic picks, so that intermediate result alone did not close the suspected boundary
defect. The later §2.3f measurement above resolved the corrupted stream; the
old entries are no longer in the current failure manifest. A future boundary
regression should be reproduced independently, not inferred from this old count.

**The fragment stream, measured on the same runner, is where the corruption
is.** `_mps_render_smoke` prints the compaction's input; the LD smoke scene
gives, on the Apple GPU against the same code on this project's CPU box:

| | CPU | MPS |
| --- | --- | --- |
| fragments | 47610 | 47194 |
| `pix` range x distinct | `[147239..272664]` x 23352 | **`[0..272664]`** x 22811 |
| `depth` range | `[18.3139..20.6677]` | **`[0.0000..20.6742]`** |
| fragment 0's low word (`lo0`) | `41a09196` | **`00000000`** |
| `frag_cov` min | 0.001002 | **0.000000** |

A key of exactly zero is a fragment slot **nothing wrote** — pixel 0, depth 0,
coverage 0 — and it sorts to the front, which is why `pix` starts at 0 and why
the 32x32 probe scene finds pixel 0 covered. So the defect is upstream of the
compaction: fragments are missing from the stream and their slots are being
read as real.

**And the op that makes those zero keys is found — §2.3f.** The same run's
attribution pass caught one gather disagreeing with the CPU; the next round,
with the pass printing its caller, named it:

```
FAIL  Tensor.index_select: 2/144 calls differ --
  raster_pipeline.py:1903 in prepare_sparse_raster_coverage:
  63/726 differ, first at 4: cpu -2147221505 vs mps -2147221504 (int32)
```

Every other op it checks — `unique_consecutive`, `argsort`, `cumsum`,
`searchsorted`, `unique`, `nonzero`, `scatter_add_`, `amin`/`amax` — agrees.
§2.3f has the mechanism and the fix; in short, that line gathers the raster
count pass's **32-bit acceptance mask**, MPS rounds it through 24 bits of
mantissa, and the write pass then skips pixels the count pass accepted —
leaving their fragment slots at zero, which is the whole table above.

### 4.3 F: `ti.real_func` with an early return, on a SPIR-V backend

```
RuntimeError: [spirv_ir_builder.cpp:query_value@1062] Value "tmp6" does not yet exist.
```

`test_taichi_early_return.py::test_a_real_function_is_not_rewritten` declares a
real function with an early `return` and launches it. The *claim* is about
Algan's rewrite pass keeping its hands off (`_outcome(early) is None`); the
launch is only how the test proves the function still works. It is the launch
that fails, in Quadrants' SPIR-V builder, and it has the shape of §1.2c's
defect — a block left un-reopened — one layer down in the compiler rather than
in an Algan kernel.

No renderer kernel uses `ti.real_func`, so this blocks a test rather than a
render. It is also the failure the Vulkan loop of §1.2c would have localized
for free, and that loop no longer exists on the published wheel (same section).

### 4.4 The three errors: an arena the render device would not size

`test_arena_binding_live` raised `OutOfRenderMemory` for a single **LD** frame
of a square and a cube on a 7 GB machine, which is not a plausible budget.
`get_num_available_bytes`'s MPS branch subtracted `driver_allocated_memory()`
from `recommended_max_memory()` **without emptying torch's cache first**, where
the CUDA branch beside it has always called `empty_cache()` before
`mem_get_info`. Cached-but-unused blocks are exactly the bytes the next arena
comes out of, so a process that had already rendered measured its own cache as
occupied — and a pytest session is one such process, hundreds of renders deep
by the time this module's fixture runs. The branch now empties the cache like
its CUDA twin, and the 1 GB cap beside it says what it is for.

### 4.5 D: the prefiltered reflection — **it never reproduced on master**

`test_prefiltered_reflection_is_substantially_wider` renders the audit tree's
calibration scene twice and asserts that turning the glossy route on widens the
reflected glow: measured 61.6 px rms against the throttled arm's 7.7, a 8.0x
ratio, asserted at 3.0x.

**On the branch this entry was written from, the assertion read
`nan > 3.0 * 74.9`,** and the `nan` is the finding.
`_reflection_spread` subtracts the window's own 10th percentile and divides by
the remainder's sum; a `nan` there means that sum is **zero**. So this is not a
reflection that came out too narrow, or too dim, or displaced — in the window
where the emitter's mirror image lands there is no signal above the wall's own
level at all. Categorical, not a tolerance.

**It looked like the device, and that was measured on three arms rather than
argued:**

| arm | result |
| --- | --- |
| macOS CPU (run 34102515789, the same runner and torch build) | passes |
| Linux, `auto` -> CPU, both Python legs | passes |
| Linux CPU with `ALGAN_MPS_FRIENDLY=1` — **the control arm** | **passes** (99.7 s off the mode, 84.6 s on it) |
| macOS MPS, branch `claude/awesome-fermi-lrapoq` @ 9583b4a | fails |

The third row is the one worth having, and it is §1.2c's discriminator applied
here: forcing the mode on over a CPU render device exercises every substitution
this port makes — the float32 accumulators, the int32 reductions, the log-step
scan that replaces `cummax`, and now `gather_exact` — with no Apple GPU in the
picture. It is green. So the remainder was Metal or torch's MPS backend, not
MPS-friendly mode, and not the renderer's own arithmetic.

**Except that the last row is the only run that ever saw it, and it is not
master.** The entry was measured on the branch, and by the time that branch
merged, master's own Apple-GPU arm was already rendering the reflection. Every
run since:

| run | ref | this test |
| --- | --- | --- |
| [34213125405](https://github.com/algorithmicsimplicity/algan/actions/runs/34213125405) (09-08 10:00) | branch `claude/awesome-fermi-lrapoq` @ 9583b4a | **fails** — the run the entry was written from |
| [34216675254](https://github.com/algorithmicsimplicity/algan/actions/runs/34216675254) (09-08 10:40) | master @ 0eacd92 (PR #115) | **passes** as an ordinary test — the xfail list was not on master yet and the arm was green |
| [34228608308](https://github.com/algorithmicsimplicity/algan/actions/runs/34228608308) (09-08 12:55) | master @ 0d73cab | XPASS(strict) |
| [34288448182](https://github.com/algorithmicsimplicity/algan/actions/runs/34288448182) (09-08 23:02) | master @ b392d60 | XPASS(strict) |
| [34302218686](https://github.com/algorithmicsimplicity/algan/actions/runs/34302218686) (09-09 02:13) | master @ 2455d08 | XPASS(strict) |
| [34306714389](https://github.com/algorithmicsimplicity/algan/actions/runs/34306714389) (09-09 03:21) | the whole file, on its own | XPASS(strict), and again with `ALGAN_MPS_SHARED_QUEUE=0` |
| [34307091566](https://github.com/algorithmicsimplicity/algan/actions/runs/34307091566) (09-09 03:27) | `ALGAN_MPS_SHARED_QUEUE=0` | XPASS(strict) |

The second row is the load-bearing one: the list did not exist at 0eacd92, so
that arm's green is the test passing on its own terms rather than an xfail
absorbing it. Six runs since, every one of them passing, against a single
failure on a branch. So the entry was **stale when it landed** — the branch it
came from (9583b4a) did not yet carry master's PR #115, and it merged (26ea787)
into a master whose arm was already green.

**It is not the shared Metal command queue.** The last two rows are the A/B:
`ALGAN_MPS_SHARED_QUEUE=0` restores the separate queues and the two per-launch
host fences (§3.3, `mps_zero_copy`), and the reflection is there either way. So
whatever put it back is not the ordering change PR #120 made, which is worth
saying because the timing invites that reading.

**Not established:** which change put the reflection back, and therefore
whether anything can take it away again. The bracket is one merge wide — PR
#115's Mac GPU integration is what master had and the branch did not — and
nothing in it was aimed at this test.

One thing worth keeping from the diagnosis, because it is what makes both
tables mean anything: the audit scene is committed
(`benchmarks/renderer_audit/scenes/calib_glossy.json`), so every green cell
above is a real render rather than the skip `_needs_audit_tree` would produce
on a tree without it. A skip misread as a pass would make all of this
meaningless. The test's sibling
`test_a_creases_siblings_share_the_pixels_prefiltered_claim` had the same shape
of failure, passes since §2.3f, and passes in every run above.

**Also not established:** whether this route is correct on CUDA. Every arm in
the first table resolves to a CPU render device, so they say "not the CPU path"
and nothing more. Nothing in any run read so far exercises the glossy prefilter
on a CUDA device.

If it ever comes back, where to look, in order: `_gloss_finish_frame`'s
pyramid build
(`gloss_pyramid_level` bottom-up, then `gloss_composite`'s trilinear fetch) —
those are the three kernels the route adds over the plain one, they run per
frame after the tile composite, and a pyramid that comes back empty on Metal
would produce exactly a flat window. `gl_main`'s blur-radius column is
initialised **negative** on purpose (`_gloss_clear`), and it is what marks a
pixel as having a prefiltered branch at all, so a scatter that fails to write
it leaves every pixel unmarked and the composite with nothing to fetch.

### 4.6 The 2026-09-09 six: five regressions and one stale entry

Run 34302218686 is the first red MPS arm since the list was written with
nothing in it about the device. Five of its six failures were made that day by
the Metal work in PR #120; the sixth is §4.5's entry outliving its defect,
which is what `strict` is for.

**The render.** `tests/fast` died in `mps_compat.band_class_groups` with
`scatter(): Expected self.dtype to be equal to src.dtype`. The function had
just traded a boolean gather for a scatter and allocated the destination with
`bands.new_empty(num_groups)` — and the fragment stream's band ids are a
`Color`-subclassed tensor, because torch propagates a Tensor subclass through
every op that touched one. `Color.new_empty` is not `torch.Tensor.new_empty`:
it returns an opaque black `[R, G, B, glow, opacity]` row, five float32 values,
whatever size is asked for. So the destination was neither the right dtype nor
the right length. The fix names both (`torch.empty(n, dtype=..., device=...)`),
and the lesson generalises past this call: **inside the renderer a tensor may
be a `Color`**, so a `new_*` method on one is not the plain constructor it
looks like. It is MPS-only because `band_class_groups` is, but there is nothing
Metal about it — `ALGAN_MPS_FRIENDLY=1` reproduces it on a CPU box in one
render.

**The four tests** all have one shape: each fakes *most* of what a decision
reads and lets the rest come from the machine, which is invisible until the
machine disagrees. On this arm the live arch is `metal`, the render device is
`mps`, and the compiler program really does dispatch on torch's command queue —
three facts they were quietly assuming away.

* `test_quadrants_pressure_reset_clears_runtime_state_when_safe` and
  `..._defers_while_a_render_is_active` fake `BACKEND`,
  `_already_initialized` and `render_is_active`, but not `_live_arch` — and
  Metal is the one arch that declines the reset outright, so both got the
  decline's answer instead of the one they assert.
* `test_shared_torch_queue_reads_the_live_programs_config` asserted that the
  live program does *not* dispatch on torch's queue before faking one that
  does. True on any other box; false on the arm the predicate exists for.
* `test_torch_compile.py::test_auto_follows_platform_support` reads only half
  of what `'auto'` resolves against: platform support, not the render device,
  which declines on Metal by design (§7.5 of the mac_2026_09 findings).

Each is now faked outright, which is what the tests around them already did.


### 4.7 `NDArray dimension length > INT_MAX`: the pool is two machines

Master @ dc63e21 killed the arm in under seven minutes, twice, with five
renders dying in

    MPSNDArray.mm:831: failed assertion
    `[MPSNDArray initWithDevice:descriptor:isTextureBacked:]
     Error: NDArray dimension length > INT_MAX'

which is `abort()`, not an exception — so xdist replaced the worker four times
and gave up (`-n 1` makes `--max-worker-restart` 4). Runs 34487179359 attempts
1 and 2 landed on different runners and produced **the same 19 failures off the
same five crashes, in the same order**: `19 failed, 1298 passed, 196 skipped`
both times. It is not a flake.

**But four other MPS runs of the same renderer code are clean**, and that is
the finding. Run 34483019642 @ 98367d6 — which differs from dc63e21 only in
`test_path_tracer.py` and `code_quality.yaml`, neither reachable from the tests
that die — ran the arm to completion: `1 failed, 4050 passed`. Runs 34493582338,
34494233578 and 34497639909 on `claude/vibrant-faraday-y8b713` reached 33-100%
clean, well past the test that dies first. So the variable is not the commit; it
is the machine.

**Which machine, though, is not settled, and the obvious proxy does not work.**
Setup-step duration looked like it separated them and does not: the two runners
that abort install BasicTeX in 21-26 s and the project in 10-12 s, but runner
1000003511 does the same work in 26 s and 13 s and stays clean. So there is no
classifier here yet — only the fact that repeated runs on one runner agree with
themselves and disagree across runners.

**What the aborts were doing.** `PYTHONFAULTHANDLER` is now set for the job;
the five stacks name three call sites, none of them about memory:

| site | the op |
| --- | --- |
| `post_process._frames_to_host` ×2 | `frame_out.flip(-3)`, `uint8` |
| `raster_pipeline.prepare_sparse_raster_coverage` ×2 | `opaque_u[a:b].fill_(True)`, `bool` |
| `bloom.bloom_filter` ×1 | `torch.amax(x[..., 3:4])`, `uint8` |

Three unrelated ordinary ops, and not one has an operand within three orders of
magnitude of `2**31` — these are 64x64 and LD frames. What they share is that
every operand is a **view of the render arena**, and all three are one byte
wide. The arena is a single `uint8` tensor handing out offset slices re-viewed
to the caller's dtype, and torch's MPS backend materialises such a view by
describing the whole underlying buffer as a flat `MPSNDArray` of
`storage_bytes / element_size` — which for a one-byte view *is* the arena's
byte count. That is the only quantity in the process that can reach `INT_MAX`,
and it is the only thing all three sites have in common.

**Confirmed by experiment, on a runner that does NOT crash.** Waiting to land
on one that does is not necessary, because the claim is about a buffer length,
not about a render — so ask it directly. Two subprocesses, each allocating one
`uint8` buffer and writing the same four kilobytes of it:

| buffer | `t[4096:8192].fill_(1)` |
| --- | --- |
| `2**31 - 4096` = 2147479552 B | returncode **0**, "survived" |
| `2**31 + 4096` = 2147487744 B | returncode **-6**, `NDArray dimension length > INT_MAX` |

Eight kilobytes of buffer either side of `INT_MAX` is the entire difference,
and the view written is 4 KiB in both. So the axis Metal is handed is the
**buffer's** length and not the view's, one-byte views of a big arena are
exactly the case that cannot work, and every one of the five stacks above is
that case. Run 34538422428, on a 7 GiB runner whose own arena is nowhere near the
ceiling — which is what makes the mechanism answerable without waiting to draw
a runner that aborts.

**And the arena is a function of the machine.** It is
`rendering_memory_fraction` (0.4) of `get_num_available_bytes`, which
`_MPS_HOST_SHARE` caps at 0.4 of total RAM — 0.16 of RAM, crossing `INT_MAX`
above ~13.4 GB. The one runner measured (1000003523) reports 7.0 GiB total,
4.67 GiB `recommended_max_memory` and a **1.12 GiB** arena, comfortably under;
a 16 GB machine would get 2.56 GB, over by 19%.

**What remains unmeasured is the arena on a runner that aborts.** No run that
crashed has printed its figure, so the last link — that these particular aborts
were the arena crossing the ceiling rather than some other buffer — is inferred
from the three call sites sharing an arena view, not measured. The gate step now
prints those four numbers on every run, so the next abort records it. Until one
does, the clamp below is a proven guard against a proven hazard, and not yet
demonstrated to be the fix for these runs.

`_addressable_arena_bytes` caps a Metal arena a megabyte under `INT_MAX` and
leaves every other device alone. It is an addressing limit, not a budget: past
it the arena is not describable in one dimension on this backend, and the
failure is a SIGABRT in an unrelated op rather than anything a caller can act
on. The megabyte of margin is there because the length Metal sees is the buffer
torch's allocator actually took, which it may round up from the request. Below
the ceiling the clamp is the identity, so it changes nothing on a 7 GiB runner
or on any Mac below ~13.4 GB.


### 4.8 The 2026-10-26 eleven: a kernel too large to compile, and three tests

The weekly full run 40306266933 failed this arm on twelve tests. One, the
dense conservative cube at index 1.35, failed on every arm: a bias in the path
tracer's MIS where a path refracts out of a scattering interior, fixed by #247
and nothing to do with the device. The other eleven were this arm's own, and
they were four causes.

**I: the caustic `pt_shade` will not compile (8 tests).** Every render in
`test_pt_caustics_render.py` that compiled the manifold walk in died making its
pipeline:

    Failed to create pipeline for kernel task 'pt_shade_arena_c170_26_1_t01'
    (RhiResult=-1). The SPIR-V shader was rejected by the backend driver

The diagnostic underneath is Apple's compiler service dropping its XPC
connection, not an error about any construct in the shader, so the question is
the shader's size. The glass-slab furnace,
rendered once with `pt_caustics` off and once with it on, with `QD_DUMP_MSL`
and `QD_SHADER_DUMP_DIR` capturing the translated shader (runs 41156317562 and
41158150876):

| `pt_shade_arena` variant | MSL bytes | lines | on Metal |
| --- | --- | --- | --- |
| without caustics | 9,023,921 | 114,538 | compiles |
| with caustics | 17,427,878 | 202,658 | XPC connection dropped |

The walk adds 8.4 MB, 93% on top of the variant that compiles. It is reached by more than `pt_caustics`:
`pt_subsurface_delta_lights`, on by default, builds the same variant for any
shadowed path-traced scene with a refracting scattering interior, because the
chains are how a point, spot or directional light reaches inside one. So on a
Mac every such scene failed, not only the opt-in feature.

The first answer was to decline: keep the kernel without the walk on Metal,
which renders as with both switches off, and list the four tests that assert
what only the chains deliver -- a point light focused through a glass plate, a
delta light reaching a subsurface block -- as strict known failures (run
41159788488). The fix splits the walk out of `pt_shade` instead. Where the
megakernel's caustic IR goes (after Quadrants' simplify passes, on CPU, which
tracks the MSL ratio):

| piece of the caustic block | IR lines |
| --- | --- |
| the Newton solve, its projections included | 22,063 |
| the legs' shadow and medium queries | 20,317 |
| the casters' material evaluation | 11,169 |
| the admissibility test | 6,564 |
| the light estimate after the walk | 3,348 |
| the caster probe, the bookkeeping, the geometry term | 31,363 |

Run 41163373007 compiled four trimmed variants of the kernel on the runner to
find the limit. With only the probe left in (154,639 lines, about 10.6 MB of
MSL) the kernel compiled in 48 s; with the walk's materials, legs,
admissibility and light estimate back it failed at 13.5 MB of MSL, and with
the legs alone back at 15.3 MB. So on Metal (`pt_caustics.caustic_chains_deferred`)
`pt_shade` is compiled with `caustics == 2`: it probes each light sample's
straight segment for casters -- which decides what the straight connection
keeps, so it has to stay -- and leaves a request where a chain is to be
walked, and a new kernel, `pt_caustic_chains`, walks them after it over the
same active slots and adds their light (`path_tracer_taichi._MQ_HEAD` has the
request's layout). The receiver's share of the firefly clamp rides along, so
the sum is clamped as the single kernel clamps it. A path defers one
receiver's chains per launch; a second receiver behind a partly transparent
first one lights as with caustics off, and is no receiver for the coverage
test either, so the two halves of the chain's MIS stay paired. CPU and CUDA
keep the single kernel; `pt_caustics_deferred` selects the split there, and
`test_deferred_chains_render_what_the_single_kernel_renders` holds the two to
the same image (on five caustic scenes they render the same PNG bytes). Run
41164965493 compiled the split on the runner -- `pt_shade` in 35-41 s,
`pt_caustic_chains` in 17.5 s -- and passed all ten caustic renders, the four
known failures among them, so cause I's entries are gone.

**The trace counters (1 test).** `ALGAN_TRACE_COUNTERS` failed inside the first
traversal kernel's compile with `qd.thread_idx() is only available in cuda or
cpu or amdgpu context`, and had it got past that, its `i64` atomics are what
§1.2 says Metal does not have. The diagnostic was always CPU and CUDA only (its
module says so), but nothing enforced it. `on_program_start` now refuses to
start counting on any other arch with a message naming the way to count
(`ALGAN_RENDER_DEVICE=cpu`), and the test skips off those two devices.

**The device shadow glue (1 test).** The test asserted that both lane
schedules build their shadow events on the device, and read
`{'device': 0, 'events': 0, 'host': 6}`: `tracer._device_shadow_glue` keeps the
host glue on Metal on purpose, because the int views and int32 prefix sum it
leans on are unverified there. The test now skips where the tracer declines.

**The precompile index hit (1 test).** The child script bound host tensors,
and asserted the main process found the kernel a precompile worker had
compiled: `(0, 1) == (1, 0)`, one miss where a hit was expected. Workers bind
their placeholders on the render device, and on MPS a launch imports a device
tensor as an external Metal ndarray but passes a host one as it is, which is a
different source key. The child now binds on the render device, as a render
does; on CPU and CUDA that changes nothing.

Run 41158150876 put the last three on the Mac arm: they pass (or skip, the
counters and the glue) with the rest of their files. Run 41159788488 then ran
all four files with the decline in: 74 passed, 2 skipped, and the four entries
under I strict-xfailed, with the decline's warning printed once.

### 4.9 The 2026-11-02 eight: mask words past the gather ceiling, and a division Metal rounds

The weekly full run 41446470532 failed this arm on eight tests, two causes
between them. Both were in code merged on 2026-10-30, between two weekly runs,
and pull requests run only the CI group on Linux and the macOS CPU arm, so
neither had run on an Apple GPU before. (The macOS CPU arm failed six of its
own in the same run, which have nothing to do with Metal: the CPU backend's
aarch64 code generation summed the glossy prefilter's four taps in a different
order in the fused finish than in the launches it replaces.
`glossy_prefilter_taichi._sealed` has the measurement and the fix.)

**J: a mask word with a flag above bit 24 (2 tests).** A fragment's mask word
keeps its eight sample bits in bits 0-7, and two flags added that day sit far
above them: `_AA_NEAR_CUT_BIT` (bit 29, on every fragment the near plane cuts)
and `_AA_ONE_SOLID_BIT` (bit 30, on a refracting batch's one-solid fragments).
With either set, the word is past the ceiling of §2.3b, and an `index_select`
returns `float32(word)`: the probe in run 42226130722 read `0x400000ff` back as
`0x40000100`, all eight samples gone and a carry into bit 8. The word went
through `index_select` in `raster_pipeline._gather_fragment_arrays`' torch arm
(the one MPS takes), the opaque-prefix truncation, `sheets._conflict_rank`'s
torch arm, and three places in `compact_sheets` (`msk_o`, `rep_msk`,
`sheet_msk_final`). Two renders showed it:

* `test_the_near_plane_opens_a_translucent_flat_solid`: 142 cut fragments came
  back with their low sample bits rounded away (`0x3B` read `0x40`, `0xFF`
  read `0x100`), the GPU built 2839 sheets where the CPU builds 2833, and a rim
  pixel landed 58 levels off its reference against a bound of 48 (run
  42224167488). With the fix it is 31, the CPU's own figure.
* `test_solids_the_walk_does_not_refract_render_as_before`: every sheet of a
  pixel the one-solid ceiling flagged came out with a rounded sample union,
  often an empty one, which the compaction forces to a sliver (`0x000000c4`
  became `0x40020100`), and 344 pixels of the four spheres moved, 304 of them
  on the glass sphere at opacity 0.6.

The stroke-pixel split's torch arm, which MPS also takes, gathered its keys,
mask words and row positions the same way, and no test saw it, because the
split's tests ran on CPU tensors only. On a 320x180 frame of solids and strokes
its sheet keys came back with their depth words rounded (`0x633f419b7891` read
`0x633f41800000`) and it made 5051 sheets where the unsplit compaction made
5075 (run 42228352763).

Every one of those gathers is now `mps_compat.gather_exact`, which is
`index_select` off MPS-friendly mode, so CPU and CUDA output does not change.
`test_sheet_route_former_vetoes.py::test_a_near_cut_mask_word_survives_the_mps_integer_gather`
reproduces the defect on any machine, by making `index_select` and
`torch.gather` round integers through a float32 in MPS-friendly mode, and
fails on the CPU without the fix. `test_stroke_sheets.py`'s split tests also
run on MPS tensors when MPS is the render device; with the old gathers all
sixty of those cases fail.

**K: the shell ceiling divides a fragment it does not clip (6 tests).** The six
failures of `test_stroke_pixel_split_holds_with_a_closed_solid` were its
`ceiling_kernel=True` cases on CPU tensors. On this arm Quadrants is
initialised for Metal, so `sheet_compact_taichi.solid_shell_ceiling` ran on the
Apple GPU with its CPU tensors staged to it. It wrote every fragment back as
`denom * min(left / denom, 1)`, and Metal's f32 division is not correctly
rounded, with `fast_math` on or off: `c / c` came back below 1 for 15,249 of
2**20 areas in [0.05, 0.95] (run 42228352763). A fragment of an undeclared
surface reaches the ceiling as a one-fragment segment whose allowance is its
own area, so the unsplit compaction wrote some of the stroke pixels'
fragments an ulp low, and the split, which never sends a stroke pixel to the
ceiling, did not. Repeated launches were bit-identical; it was the division,
not an atomic or an order. The kernel now writes back a fragment the allowance
covers unchanged and divides only one it clips. Wherever division is correctly
rounded that is the old expression bit for bit (1440 old/new pairs on the CPU,
both accumulator widths), and
`test_sheet_chain_kernels.py::test_solid_shell_ceiling_returns_what_it_does_not_clip_bit_for_bit`
holds it on every device. A fragment the ceiling does clip still divides, so on
Metal it can sit an ulp from the CPU's; it is the same split or unsplit, which
is what the compaction promises.

Runs 42226130722, 42232166078 and 42236806653 put the four affected files on
this arm with the fixes in: 6, 9 and 186 passed (20 skipped, as on master), and
the 320x180 frame's split and unsplit compactions agree row for row.

### 4.10 What J's pattern still reached: refs, the ceiling's keys, the Bezier header, pixel ids, a scan, and a deadlock

§4.9 made every gather of a mask word exact. The renderer was then read for
the rest of that pattern, an integer that can pass 2**24 going through
`index_select` or through an op torch builds on it, and every site found was
put on the Apple GPU before and after its fix (§4.1, 2026-11-06 and 07). Four
more causes were gathers (L, M, N, Q). Running the stroke split's kernel arm
on the GPU found a scan torch 2.13 cannot run there (O), and a hang in one of
those runs was a deadlock in the zero-copy cache (P). None of them failed a
weekly run, each for its own reason: L needs a batch of 65,536 circuits,
which no test drew; M and O sit behind settings whose defaults avoid them
(`sheet_shell_ceiling_kernel` on, `sheet_split_kernel_mps` off); N's code runs
on the CPU on a Mac's default route; P needs an imported tensor freed by the
cyclic garbage collector, which a failing MPS test arranges; and Q needs a
chunk of more than 2**24 pixels, which the runner never plans. Each fix has
a test that fails without it. Where the defect is a gather the test emulates
the rounding on the CPU (`tests/unit_tests/mps_rounding.py`), and when MPS is
the render device it runs on MPS tensors too.

**L: a fragment's ref past 2**24.** A fragment's ref is a triangle's index,
or for a Bezier circuit `-((circuit << 8) + border weight + 1)`
(`raster_taichi._pack_bez_ref`). So a circuit's ref passes 2**24 from a
batch's 65,536th circuit on, and a triangle's once one frame holds 2**24
triangle columns. Rounded, circuit 65,536's fill, ref -16,777,217, reads
-16,777,216, which decodes as circuit 65,535's full border, so the fill takes
the stroke's colour. A rounded triangle ref names its neighbour, and the
sample-depth rule reads the neighbour's surface off it. Six gathers carried a
ref on the route MPS takes: `raster_pipeline._gather_fragment_arrays`' torch
arm, the opaque-prefix truncation, the shadow events' source triangles
(`ev_src_prim`), and in `compact_sheets` the sample-depth `rep_ref` and both
`sheet_ref` outputs. Four squares drawn as circuits 65,536 to 65,539, behind
65,536 circuits off screen, came out 1,197 pixels of a 96x54 frame off the
same squares drawn as circuits 0 to 3, by up to 255, with 1,250 of 2,063
sheet refs moved. With the six gathers exact, none moved (run 42300938376).
`test_mps_fragment_refs.py` holds it: the render, unemulated on the GPU, and
the compaction's and the fragment gather's refs on synthetic streams. Only
the shadow site has no test, since it takes 2**24 triangles in one frame.

**M: the closed-shell ceiling's torch arm.** With `sheet_shell_ceiling_kernel`
off, `compact_sheets` orders and segments the ceiling's stream in torch, and
three steps of that gathered an integer: the key `pix_o * K + shell_sid`
(`k2`), `_lexsort`'s torch fallback over it, and `_segment_exclusive_prefix`'s
row positions. The key passes 2**24 from row 1,456 of a 4K frame with two
surfaces, and from row 971 of a 1080p frame with eight. Rounded, two solids'
segments in one pixel merge and share one allowance, and over the rounded
`_lexsort` even the kernel arm goes wrong. A 1080p frame of eight spheres
rendered 7,909 pixels up to 85 levels apart between the torch arm and the
kernel arm, and the float32 segment prefix over 2**24 + 2**16 rows found the
wrong row for 27,305 of them (runs 42300780142 and 42303270183). Those
gathers are `gather_exact` now. The arms then agree to float32 noise, at most
1.19e-07 on 17,489 of 663,659 clipped fragments. That noise has three
sources: K's division in the kernel, the torch arm's float32 `scatter_add_` of
the cap, whose order MPS does not fix, and the two arms' different summing of
the spend, which the CPU shows as well. The tests are
`test_sheet_chain_kernels.py::test_the_shell_ceiling_keeps_its_segments_past_the_mps_gather_ceiling`
and `::test_the_float32_segment_prefix_finds_its_rows_past_the_mps_gather_ceiling`.

**N: the Bezier header, its intervals and the polylines.**
`bezier_acceleration.build_bezier_edge_acceleration` wrote each (frame,
circuit) group's 91-word header with
`torch.index_select(compact, 0, compact_of_group, out=...)` from an int32
table. The header's bounds and inverse bin sizes are float32 bit patterns,
nearly all past 2**24, and its bin offsets pass 2**24 once the buffer does,
from 184,366 groups at the default bins. `_expand_intervals` and four
run-start expansions (`batch_arange`, `_packed_uniform_cubic_parameters`, and
the geometry cache's `_build_cached_circuit_edges` and
`_build_frame_local_circuit_edges`) used `repeat_interleave(values, counts)`.
Torch builds that as `values.index_select(0, repeat_interleave(counts))` on
every backend, so it rounds the same way; the counts-only form is exact.
Built on the Apple GPU, a frame of text, a circle, an arc and a stroke moved
100 of its 1,365 header words, by up to 64 ulps. A page of 1,850 glyphs over
100 frames moved 7,962,290 of 16,835,000, after which the Metal kernels read
72,725 of its 185,000 centroids as outside their own glyph (run 42299224973).
The interval expansion moved 789,747 of 18,359,815 offsets, and the polylines
1,020,086 of 18,816,186 vertex numbers (run 42305385094). With the header
gathered by `gather_exact(..., out=)` and the expansions by the new
`mps_compat.repeat_interleave_exact`, all are 0. On the default route this
was latent: `merge_on_gpu_active` and `project_on_gpu_active` are CUDA-only,
so a Mac builds the header on the CPU and uploads it, and the default route's
frame hashed the same before and after. In MPS-friendly mode that CPU build
pays for the exact gather with a header copy, 38 ms instead of 25 ms at
185,000 groups. The tests are `test_bezier_header_mps_gather.py` and
`test_bezier_polyline_mps_repeat.py`.

**O: a scan torch 2.13 does not ship for Metal.** With
`sheet_split_kernel_mps` on, `_split_stroke_pixels`' kernel arm scanned its
`[num_covered, 2]` int32 tally in place along dim 0, and on MPS that raised
`Failed to create function state object for: cumsum_vec_block_reduce_2_int`
from 1,024 covered pixels on. Torch 2.13.0 picks its MPS scan kernel by
layout. For a 2-, 3-, 4- or 8-wide tensor of 1,024 rows or more it asks for
`cumsum_vec_block_reduce_<w>_<type>`, and the metallib builds that for
bfloat, float, half, int_long (int32 in, int64 out) and long, never for an
int32 result. `torch.ones((1024, 2), dtype=torch.int32, device="mps").cumsum_(0)`
raises, and 1,023 rows run (runs 42297623070 and 42303103954). The split's
MPS test had used 200 pixels. The tally is `[2, num_covered]` now, each row
scanned on its own: a 1-D int32 scan is exact on MPS, and the boolean-mask
gather of the offsets measured exact against the CPU to 5,000,000 rows. The
12 MPS cases of `test_stroke_sheets.py::test_stroke_pixel_split_kernels_match_the_torch_arm`,
now on 6,000 pixels, all failed before the fix and pass after it (run
42308885414), and both split files pass with the setting on and off. No other
scan in the renderer is in that family. The kernel arm compacted an HD frame
in 19.3 ms there, against 22.4 ms for the torch arm and 19.4 ms for not
splitting at all, so the setting's default is unchanged.

**P: the zero-copy cache waited on its own lock.** `mps_zero_copy._owner_gone`,
the weakref callback that evicts an import whose owner has died, took
`_LOCK`, a plain `threading.Lock`. The cyclic garbage collector can run that
callback on any allocation, including the `weakref.ref` that
`_track_owner_locked` makes while the same thread holds the lock, and the
thread then waits on itself forever. A failed test's traceback keeps its MPS
tensors in reference cycles, so the next import after one could hang a
session. Run 42303103954's unfixed split cases did, for 26 minutes, with
`_owner_gone` under "Garbage-collecting" on top of the stack. The callback
now queues the death and evicts only if it can take the lock without
waiting, and whoever holds the lock drains the queue before and after its
own work.
`test_mps_zero_copy.py::test_an_owner_freed_while_the_cache_lock_is_held_is_evicted_without_deadlock`
frees an owner under the lock in a subprocess, which hung before the fix and
evicts it after.

**Q: pixel ids and stream positions past 2**24.** A fragment's pixel id is
chunk-relative, frame * width * height + pixel, so it passes 2**24 from a 4K
chunk's third frame and a 1080p chunk's ninth, and the compaction gathered
pixel ids through `index_select` on the default route: into the sorted
stream (`pix_o`), onto the sheets (`sheet_pix`) and into walk order. The
reference depth gate's sample keys, pixel * 8 + lane, pass 2**24 inside one
4K frame. On the Apple GPU a real 96x54 frame's fragments, moved into a 4K
chunk's third frame, lost their pixel boundaries in all three arms of the
compaction: of 2,394 sheets, 513 pixels' sheet offsets, 1,207 sheets' pixels
and 39 sheets' coverage (by up to 0.43) moved (run 42329686918). Seven sites
also spread an arange over runs with `repeat_interleave(values, counts)`,
which rounds the same way (34,044 of 17,498,962 values in that run), and the
opaque-prefix truncation's torch arm moved 65,536 of its 16,908,289 keep
flags. Every integer gather of a pixel id, a stream position, or a band,
group or sheet id in `compact_sheets` and its helpers, and in
`raster_pipeline`'s `_exact_fragment_order` fallback, `_opaque_prefix_keep`
and `_sheet_slice_bounds`, is `gather_exact` now. The seven repeats take
`repeat_interleave(counts)`, which is the arange gathered by those indices,
and exact. Default renders did not get there on the runner: its 1.2 GB arena
plans one 4K frame per chunk and at most seven of 1080p, so the largest
pixel id was 14,462,004 and the longest stream 8,084,938 fragments, and every
frame matched a render in one-frame chunks. A Mac with 16 GB or more gets the
largest arena MPS allows, 2**31 - 2**20 bytes
(`memory_utils._MPS_MAX_ARENA_BYTES`), and there the memory model plans 12
frames of a light 1080p scene, pixel ids to 24.9 million, while 4K stays at
two frames, 188,416 ids short. So Q is live for a light 1080p scene on such
a Mac when it goes through `compact_sheets`: the stroke-only explainer, the
measured scene that plans that many frames, persists its sheets by a path
with no rounding op. The tests are
`test_sheet_chain_kernels.py::test_the_compaction_keeps_neighbouring_pixels_apart_past_the_mps_gather_ceiling`
and `::test_the_reference_depth_gate_keeps_its_samples_apart_in_a_4k_frame`,
and `test_sheet_compaction.py::test_the_opaque_prefix_keeps_each_pixels_own_past_2_24_covered_pixels`.
The stream-position gathers are fixed but untested: they pass 2**24 only in a
stream of that many fragments.

**Seen and not chased** then, recorded so the next reader started from the
measurement. §4.11 chased all seven, and says what each one was:

* An out-of-memory error on MPS is silent. A chunk of 5,500 triangles over 100
  frames, built on the Apple GPU, printed `Insufficient Memory
  (kIOGPUCommandBufferCallbackErrorOutOfMemory)` to stderr, torch raised
  nothing, and 102.9M of its 121M header words came back wrong (run
  42305385094; the runner's recommended working set is 4.67 GB). **Fixed**:
  a render holds torch to the working set (X).
* The GPU makes 2,063 circuit sheets of L's frame where the CPU makes 2,055,
  before the fix and after it, and the frames match after it. The stroke-pixel
  split is not the cause. **Not a defect**: an ulp of Metal's ray direction.
* On MPS the stroke split's torch arm flushes a denormal area (1.4e-45) to 0
  where the kernel arm keeps it. No frame measured had one. **Fixed** (V).
* The general path sums areas with float32 atomics whose order MPS does not
  fix, so a sheet of three or more fragments can move by an ulp from one
  replay to the next: 27 of an HD frame's 168,675 sheets did. **Fixed**: the
  areas are summed in stream order (W).
* The BVH's gathers pass 2**24 once a BVH has 2**24 primitives:
  `refit_bvh.py`'s `sub_start`, `sub_count` and `sub_block` (lines 864-899)
  and `stbvh.py`'s slot ids (line 418), and the id expansions in
  `refit_bvh.py` (line 253) and `sliver_split.py` (line 137). None was
  measured. **Fixed** (R), except the slot ids, which go through
  `torch.gather` and were exact.
* `torch.unique` merges distinct values past 2**24 on torch 2.13.0: 2,010,925
  of 2,091,233 came back (run 42329686918). In the renderer it is reachable
  on MPS only in `_sheet_rank_groups`' torch arm, whose key reaches 2**25 in a
  4K frame, and MPS takes the kernel arm by default. **Fixed** (U), along with
  a second site, the solid shells' table.
* An int64 `scatter_reduce_` amin or amax raises on MPS, as §2.3 found (the
  renderer reduces in int32 there, §1.2), and it then poisoned the process:
  every new MPS kernel after it failed the same way, and the process died in
  an MPSLibrary `gatherND_base` assertion (run 42329686918). A probe should
  measure it last, or in a process of its own. Measured that way (T), the op
  also flips every sign bit of its output, a process survives one failure,
  and a second leaves it unable to compile a new kernel. **Fixed**: an int64
  extremum is computed from two int32 ones.

### 4.11 What §4.10 left: the BVH and scene ids, an int64 reduction, `unique`, two float sums, and a silent out-of-memory

§4.10 ended with seven things seen and not chased. All seven were put on the
Apple GPU on 2026-11-07 (§4.1). Six were defects and are fixed (R, T, U, V, W,
X), and reading for R's pattern found the same gather at seventeen more sites
(S). The seventh, the GPU's extra circuit sheets, is an ulp of Metal's
arithmetic and stays. None of them failed a weekly run: R and S need more than
2**24 primitives, ids or surfaces in one batch; T is off the default route;
U's sites are a torch arm MPS does not take and a table whose surface ids pass
2**24 only in a scene of that many; V and W change a bit or an ulp that no
frame measured has shown; and X needs a build larger than the runner's working
set. As in §4.10, each fix
has a test that fails without it, a gather's test emulates the rounding on the
CPU (`tests/unit_tests/mps_rounding.py`), and a test of an op MPS gets wrong
also runs that op on the Apple GPU: on any Apple machine where the op runs
under the macOS CPU arm's torch 2.7.1, and otherwise when MPS is the render
device.

**R: the BVH builders' positions past 2**24.** The torch BVH builder moves
each level's ranges with integer gathers: a range's start and count are
positions in the primitive order, and its id (`sub_block`) numbers it within
the level. `refit_bvh._build_level` gathered them with `index_select`, and
`_binary_split` spread the range ids over their primitives with
`repeat_interleave(values, counts)`. On the Mac, `index_select` of 1,048,576
range starts past 2**24 returned every one wrong (17,812,899 as 17,812,900),
and the `_binary_split` expansion over 16,781,312 ranges 3,075 of its
25,174,750 ids (run 42352045841). A level of 16,777,343 primitives then read
past the end of the order and raised `index 16777343 is out of bounds`.
Positions go wrong in a tree of more than 2**24 ever-visible primitives built
by the torch builder, which on MPS is any device build of more than 8,192
columns (`ALGAN_REFIT_HOST_TOPOLOGY`), and range ids in a level of more than
2**24 ranges; a CPU build takes the NumPy builder and never runs this code.
`sliver_split._strip_rows` had the pattern three times over: the sliver that
owns each strip, each sliver's first strip, which passes 2**24 at 524,288
slivers of the default 32 strips, and the triangle a strip cuts. `_build_level`
and `_strip_rows` gather with `gather_exact` now, and the two expansions take
the counts alone (`repeat_interleave(counts, output_size=...)`), which is the
arange gathered and exact. A root level of 17,825,793 primitives built on the
GPU, with a sync after each op, is then the CPU's level exactly (runs
42370479770 and 42372352869). `stbvh.py`'s slot ids were already exact: they
go through `torch.gather`, which this torch answers exactly (§2.3b). The tests
are `test_bvh_mps_gathers.py`'s four, emulated and on the device.

**S: seventeen more ids in the scene build.** Ten sites spread an arange over
runs with `repeat_interleave(values, counts)`: one in `bezier_acceleration.py`,
four in `bezier_geometry_cache.py` and five in `primitives.py`. They take
`repeat_interleave(counts)` now, the same tensor. `primitives.py`'s
`vert_circuit` spreads circuit ids that are not an arange and takes
`repeat_interleave_exact`. Four surface-id gathers in
`sheets._sample_depth_lose_reference`, one in `scene_builder.py` and the
owner gather of the depth-order diagnostic
(`float32_rounding.depth_order_record`) take `gather_exact`. In a Mac render
only the four `sheets` gathers and the diagnostic run on the GPU: the
reference depth gate, taken with
`ALGAN_SHEET_DEPTH_REDUCE_KERNEL=0` or for a batch too big for its kernel,
goes wrong past 2**24 surfaces, and the diagnostic past 2**24 Mobs. The rest
run on the projection and merge device, which is the CPU on a Mac, since GPU
projection and merge are CUDA-only, so MPS reaches them only through
`ALGAN_ANIMATION_DEVICE=mps` or a direct call, and each needs more than 2**24
circuits, members or triangles in one batch. `test_ids_mps_gathers.py` builds
each case alone and again behind 2**24 empty circuits or members, and each of
the sixteen sites, reverted alone, fails one of its eleven tests.
`test_depth_order_warning.py::test_owners_past_2_24_stay_apart_through_the_mps_gather`
finds 116 pairs instead of 152 without the fix. No Mac run was needed: the
fixes use only `v[i]` and the counts-only repeat, which run 42352045841
measured exact through 2**25.

**T: an int64 extremum corrupts its output, then the process.** §2.3 found
that an int64 `scatter_reduce_` `amin` or `amax` raises on MPS. Run one case to
a process (run 42352045841), every such call, `scatter_reduce_` or
`scatter_reduce`, with `include_self` on or off, raises `Failed to created
pipeline state object`. `prod` fails earlier, at function lookup, and
`index_reduce_` refuses int64 cleanly; sum, mean, `scatter_add_` and the int32
extrema work. Torch 2.13.0 runs the op as a ulong atomic kernel between two
passes that flip every element's sign bit (`ScatterGather.mm`), and the
compile fails between them, so the output comes back with every sign bit
flipped: 1,024 of 1,024 (run 42354063716). A process survives one failure.
After a second, of either reduction, it cannot compile a kernel it has not
compiled before, and the next such kernel aborts it: exit −6 in that run, and
the `gatherND_base` assertion of run 42329686918. The renderer asks for none of
these on its default MPS route, since its reductions use
`reduction_index_dtype()`, int32 in MPS-friendly mode, and its two int64 `amin`
sites (`_padding_edges` and `_split_promotable`) reduce CPU tensors. With
`ALGAN_MPS_FRIENDLY=0` on MPS, though, `reduction_index_dtype()` is int64 and
those reductions reach the op. So all 25 call sites go through
`mps_compat.scatter_reduce_` and `scatter_reduce` now, which compute an int64
`amin` or `amax` exactly from two int32 reductions, high words and then the
winners' low words, whenever the tensor is on MPS or the mode is on. A shape
the route cannot do raises before any op runs, and every other dtype and
reduction is the op itself. `test_mps_int64_reductions.py` checks the route
against the op on the CPU and on the GPU, and its
`::test_algan_reduces_through_mps_compat` fails on a direct call anywhere in
`algan/`. `benchmarks/_mps_torch_op_probe.py` runs each int64 case in an
interpreter of its own, last, with canary kernels after it.

**U: `unique` with an inverse merges integers past 2**24.** On torch 2.13.0's
MPS, `torch.unique` asked for an inverse, or given a `dim`, works on
`float32(int32(x))`. `unique(arange(2**24, 2**24 + 8), return_inverse=True)`
returns 16777216, 16777218, 16777220, 16777222 and 16777224, the last not in
its input, and 8,192 consecutive values from 2**24 come back as 4,097 (runs
42351444222 and 42371919983). From 2**32 an int64 keeps only its low word.
The values-only and counts-only forms are exact, and so are sort and argsort,
to 2**62 over 33.5 million rows. A record of every `unique` and
`unique_consecutive` that ran on MPS in a test session (run 42351444222) found
one with an inverse over values that can pass 2**24, the solid shells' table
in `shell_alpha.py`, and reading found one more, `_sheet_rank_groups`' torch
arm. In MPS-friendly mode the table takes the new `mps_compat.unique_exact`,
which sorts stably, compares neighbours, numbers the groups with an int32 scan
and scatters the numbers back, and the rank groups take `band_class_groups`
(§2.3b). Off the mode both run the op they ran before. On the Mac without the
fix, the device tests found "4 shells" where there are 5 and 4,199 rows of the
rank groups moved (run 42355723203). The tests are `test_mps_friendly.py`'s
four `unique_exact` ones, and two each in `test_sheet_rank_groups.py` and
`test_hybrid_transport.py`, emulated (`mps_rounding.merge_wide_uniques`) and on
the device.

**V: the stroke split's torch arm flushed a denormal area.** On MPS every float
op measured flushes a denormal to zero: add, multiply, `clamp_min`, `maximum`,
`scatter_add_` and `index_add_`. Copies keep it, and a NaN's payload: clone,
`index_select`, `x[i]`, a boolean mask and `index_copy_`. In Taichi on Metal a
copy is exact, and max, atomic add and add flush (run 42355084729). The
split's torch arm applied the stroke area rule, `(cov + 0.0).clamp_min_(0.0)`,
in float arithmetic, so it flushed a 1.4e-45 area that the kernel arm, which
applies the rule to the bits, kept. Linux's CPU did the same, because Taichi's
init turns flush-to-zero on for the main thread. Only the affected sheet's
`sheet_wgt` differed, and replacing all 4,117 zero sheet weights of an HD
explainer frame with 1.4e-45 moved no pixel. `sheets._stroke_area_bits`
applies the kernel's rule to the int32 view now: a set sign bit gives +0.0, a
NaN gets its quiet bit, and anything else keeps its bits.
`test_stroke_sheets.py::test_stroke_pixel_split_kernels_match_the_torch_arm`
feeds ±1.4e-45 and ±1.18e-38 areas on every device, and its six MPS cases that
compare the arms' weights failed before the fix and pass after it (runs
42355084729 and 42357151096).

**W: the sheet areas' float32 atomics had no fixed order.** In MPS-friendly
mode the general path sums each sheet's area with float32 atomics, since Metal
has neither float64 nor 64-bit integer atomics to sum them wider or in fixed
point, and Metal fixes no order for them. Replaying one captured compaction six
times moved 6 to 12 of an HD explainer frame's 168,675 sheets per replay, and
283 to 309 of a solids frame's 1,843,088, each a sum of 3 to 18 fragments
(runs 42355084729 and 42357151096). No final pixel moved in those renders. A
new kernel, `sheet_compact_taichi.sheet_band_reduce_runs`, runs one thread per
(pixel, surface, facing) group from the compaction's group starts and sums
each band's area left to right in stream order. `_band_reduce` takes it when
its accumulator is float32 and it is given the runs, which `compact_sheets`
gives all four of its sums: the sheet areas, the rank pool's threshold and the
two sibling-weight composites. `sheets.sheet_area_fixed_order`, on by
default, turns the kernel off for an A/B against the atomics. The replays are then bit-identical, and so are the
stroke split's arms on an HD frame. The cost did not measure: the compaction's
median was 24.17 ms against 24.38 ms on the explainer, and 213.35 against
210.20 ms on the solids. Off the mode the accumulator is float64, which was
measured stable bit for bit, and the kernel never runs.
`test_sheet_area_order.py`'s two tests, the areas against numpy's stream-order
float32 sums and replays through the rank pool and the sibling weights,
failed in 6 of 6 Mac runs before.

**X: an out-of-memory failure on MPS was silent.** Metal refuses no
allocation past `recommendedMaxWorkingSetSize`, nor past the machine's RAM. It
fails the work that needs more memory resident than it can give, and nothing
reads that failure: torch's `MPSStream::commitAndWait` reads only the kernels'
own error buffer, never the command buffer's status, and Quadrants 1.3.3's
`MetalStream::command_sync` does not either. So nothing raises, the outputs
hold whatever their buffers held, and MPS prints
`kIOGPUCommandBufferCallbackErrorOutOfMemory` or nothing at all. Nothing kept
Algan out of that region: torch's own limit is 1.7 working sets
(`default_high_watermark_ratio`), 7.93 G on the 7 GB runner against a working
set of 4.67 G. The arena planner budgets against the working set, but torch's
prep and merge work sits outside it, and a Bezier header for circuits whose
edges change every frame can be large: §4.10's case peaks 3.27 GB above
baseline on the CPU. Built on the GPU, it took the device to 7.26 G, two
command buffers failed, and 102,317,937 of its 121,000,000 words came back
wrong (run 42351612808). One `torch.add` of two 2 GiB tensors into a third, at
6.0 G or 1.29 working sets, less than the runner's RAM, left its output all
zeros and printed nothing; at 4.5 G it was right (run 42355691483).
`taichi_runtime.render_job_holding_the_arch` now holds torch to the working
set with `torch.mps.set_per_process_memory_fraction`
(`memory_utils.hold_mps_allocator_to_the_working_set`), from the start of the
outermost render job to its end, and puts torch's default back after it; the
hold was first 1.0 working sets and is 0.98 now, for the reason below. An
allocation that would cross the limit is then refused: torch frees its cache
and retries, then raises "MPS backend out of memory", which `is_cuda_oom`
matches, so the render's retries split the window as they do on CUDA. Held
at 1.0, the same header build raised after 11.7 s at 4.42 G. The environment
variable would not do: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0` alone breaks
torch's init ("invalid low watermark ratio 1.4"), and a ratio a user sets
there stands. The heavy reference renders peak at 2.4 to 3.3 G, and rendered
with the hold on and off in turn they reached the same peaks, the warm renders
were no slower, and every frame was identical (run 42370277126).
`test_mps_working_set_hold.py` checks the hold and its release after a render
that fails, and on the MPS arm that a render job cannot allocate past the
working set and can again after it.

Two more silent failures seen on 2026-11-07 were X as well. R's device BVH
builds, run without a sync after each op, went wrong where the same builds
synced after each op were the CPU's: a build of 1M primitives raised `index 4
is out of bounds` at 6.06 G (run 42356233755), and the level of 17,825,793
`index 17825793 is out of bounds` at 6.28 G. Committed after every op without
waiting, both were right even at 7.01 G, since no one command buffer then
needs that much resident, and under the hold the 1M build is right and the
level raises "MPS backend out of memory" (run 42374877006). And an int32 sum
that returned 0 at 0.9 working sets (run 42353661819) was one command buffer
reading 17 buffers of 256 MiB whose 1 GiB heaps came to 1.07 working sets;
each sum alone was right (run 42372132733). What fails is one command buffer
whose heaps come within 20 to 30 MiB of the working set: an add over
exact-size heaps was right at 0.994 working sets and wrong from 0.996 (run
42381446368). No default render reaches the BVH case, since the torch builder
runs on MPS only for a deferred tree of more than 8,192 columns, in a scene
that builds BVHs, and it needs about 1M primitives on this runner.

**A limit on requests is not a limit on heaps, so the hold needed a low
watermark.** Torch checks each request against its limit, not the heap the
request opens, and while the device is under the allocator's low watermark,
1.4 working sets by default, a request of 10 MiB to 512 MiB that finds no
room opens a new 1 GiB heap. Held at the working set, the device could
therefore end up to 1 GiB past it: 17 buffers of 256 MiB were let into 5.00 G
of heaps, 1.07 working sets, and one add over all 17 came back wrong in every
buffer with nothing printed (run 42374877006). Past the low watermark torch
counts the device as under pressure and gives each such request a heap of its
own size. Torch reads `PYTORCH_MPS_LOW_WATERMARK_RATIO` only when its
allocator starts, at the first MPS allocation or `torch.mps` memory call, so
`import algan` sets it to 0.75 on macOS before then
(`algan/utils/mps_low_watermark.py`), and leaves alone a watermark the user
set or an allocator that has already started. The last 1 GiB heap then opens
below 0.75 working sets and ends below 0.964 of them on this runner, and after
it only a 32 MiB heap, opened for a 1 to 10 MiB request, can pass the limit,
which is why the hold sits at 0.98. The same 256 MiB buffers then stopped at
18 in 4.50 G, 0.964 working sets, torch refused the 19th, and the add over all
18 was right. In three worst cases the device ended at 0.980, 0.986 and 0.985
working sets, and every add was right, and the 1M device BVH build matched
the CPU's tree at 4.07 G, against 5.06 G under the hold at 1.0 (run
42385375749). Holding 1,056 MiB under the working set instead also closed the
worst case measured (run 42381446368), but it gave up a gigabyte on every Mac
and refused that build. The reference renders never reached 0.75 working
sets, so they ran as before: the same peaks, times within the run-to-run
spread, and every frame compared identical. So did a UHD render with the
watermark at 0.5, past which it spent nearly all its time (runs 42385375749
and 42388655058).
`test_mps_working_set_hold.py::test_one_command_buffer_over_all_an_mps_render_holds_comes_back_right`
fills a render's hold with 256 MiB buffers and adds over all of them: it
failed at 1.0716 working sets before and passes, and three stub tests check
the import's rules on any platform.

**The GPU's extra circuit sheets are not a defect.** L's frame makes 2,063
sheets on the GPU and 2,055 on the CPU because the raster keeps 8 more
fragments there. In column 30 the front circuit's coverage is 0.9999976 on
Metal, just under `AA_FULL_COVERAGE` (1 − 1e-6), where the CPU's is 1.0, so
the opaque-prefix truncation keeps the fragment behind it. The CPU's kernel on
the GPU's inputs gives the CPU's raster and Metal's on the CPU's inputs the
GPU's, so the inputs agree and the arithmetic differs. The first intermediate
whose bits differ is the camera ray's direction, −0.12817332 against
−0.12817331, about an ulp of Metal's normalize and division, and turning fast
math off on Metal changes nothing (run 42355723203). Four squares show it as
well, so it needs no 65,536 circuits. The extra fragment weighs at most
2.4e-6, so L's frames are identical and other frames differ by at most one
level at one pixel. It costs memory and time only, and it stays.

**A coverage bug every device had.** Tracing those sheets found one that is
not the GPU's. The oriented wedge that draws a Bezier circuit's corner
(`raster_taichi._bez_pixel_hit`) handed its convex-or-reflex choice to the
crossing parity whenever one of its two side tests abstained, and at a right
angle with the pixel centre on a wall's line that parity is a rounding. A
square whose edge ran through pixel centres drew the pixels beside its corners
fully covered where they are half covered. It is fixed in a PR of its own,
#251, because it changes CPU and CUDA output too.

**Seen and not chased**, recorded so the next reader starts from the
measurement:

* `gather_exact`, `repeat_interleave_exact` and `unique_exact` switch on
  MPS-friendly mode, so with `ALGAN_MPS_FRIENDLY=0` on an MPS device every
  site of §4.9 to §4.11 rounds again. T's route also checks the device and
  stays exact.
* A process that started MPS before importing Algan keeps torch's default
  low watermark, and with it the 1 GiB overshoot: 17 buffers in 5.00 G, all
  wrong (run 42385375749). The hold still applies there, and says at debug
  level that it can be overshot. Falling back to 1,056 MiB under the working
  set in such a process would close it, at the cost of refusing the 1M device
  build there; it is not done.
* X's hold is torch's alone. Torch counts the whole device, the native arena,
  Quadrants' buffers and MPSGraph's included, but refuses only its own
  allocations, and pressure from elsewhere below the limit stays silent:
  only torch or Quadrants reading a command buffer's status would catch it.
  A fraction a user set at runtime is replaced during a render, and the release writes
  torch's default of 1.7, since torch has no getter. Only render jobs are
  held.
* A UHD render at the default host share (0.4 of RAM) raises
  `OutOfRenderMemory` on the 7 GB runner, with the hold and without it (run
  42351612808). It is loud, so it is a limit rather than a defect.
* Three UHD renders of 40 on the runner segfaulted, under the old hold and
  the new alike: all on the main thread, in `index_select_out_mps` calling
  `+[NSDictionary dictionaryWithObjects:forKeys:count:]`, with
  `EXC_BAD_ACCESS`, while the prefetch worker ran CPU torch work for the next
  batch (runs 42385375749 and 42388655058). The rate rose with how often a
  probe thread called `torch.mps.driver_allocated_memory()`: none in 16
  renders sampled every 0.5 s, one in 9 at 0.1 s and two in 15 at 20 ms. So a
  thread asking the driver for its memory during MPS work may provoke it,
  which is not proven. It was Y, the batch-prep worker draining torch's
  caches, and that query alone never crashed (§4.12).
* On MPS a denormal stroke area is kept by both split arms now, but the
  unsplit general path's float32 sum flushes it, so split and unsplit can
  give that sheet different weights. The resolve cannot tell them apart.
* Two float32 sums keep an unordered `scatter_add_`: `_band_reduce`'s torch
  arm, taken only with `sheet_mask_kernel` off, and the shell ceiling's cap
  (M's noise).
* `unique_consecutive` returns `int32(x)` as its values for an int64 of 2**31
  or more on MPS. No call site comes near that.
* Under torch 2.7.1, the locked version the macOS CPU arm keeps, `unique_exact`
  is not exact on MPS either: its values for int32 keys from 2**24 came back
  wrong, and the BVH level step has no int64 `index_add_` there (run
  42382117024). Algan renders nothing on MPS with that torch, so no MPS tensor
  reaches either, and their device tests ask the GPU only when MPS is the
  render device.
* Arange expansions remain on the animation device, in
  `mobs/bezier_circuit.py`, `mobs/nonplanar_circuit.py`,
  `animation_timeline/timeline.py` and `animatable_base/mob_morph.py`. That
  device is the CPU on a Mac.
* Two more coverage errors in the wedge, on every device, since fixed
  (`DESIGN_analytic_aa.md` §22.5). A segment is listed
  in every grid cell it crosses, so in `_bezier_point_metrics_in` the nearest
  segment's repeat can take the second slot from the corner's other wall and
  the wedge sees one wall twice: that is all 48 of the fragments a 160x90 grid
  of squares has more than 0.02 off their area, and barring the repeat alone
  made 12 others worse. And an apex exactly on a pixel centre leaves both side
  tests abstaining.

### 4.12 Y: a cache drain on the batch-prep thread, and the UHD segfault

§4.11 saw three UHD renders of 40 segfault on the 7 GB runner and did not
chase them. They were put on the Apple GPU on 2026-11-08 (§4.1), and they are
one defect, Y: the batch-prep worker drained torch's MPS caches while the
render thread was running an MPS op, and the drain deleted the graph the op
was using. It failed no weekly run, since it needs memory pressure while a
render runs on MPS, which a UHD render gives the 7 GB runner.

**Where they died.** Each crash was on the main thread, in
`index_select_out_mps` +1608 calling
`+[NSDictionary dictionaryWithObjects:forKeys:count:]`, with `EXC_BAD_ACCESS`
and "possible pointer authentication failure", while the batch-prep worker
(`algan-batch-prep_0`) worked on the next batch (runs 42385375749 and
42388655058).

**A drain deletes every cached graph.** In torch 2.13.0,
`torch.mps.empty_cache()` is `MPSHooks::emptyCache`, which drains the
allocator and then calls `MPSGraphCache::clear()`, and the clear deletes every
cached graph (`aten/src/ATen/mps/MPSHooks.mm`,
`aten/src/ATen/native/mps/OperationUtils.h`). An op gets its graph from
`LookUpOrCreateCachedGraph` as a raw pointer, and the cache's serial queue
orders the lookup against the clear but not the graph's use after it.
`index_select_out_mps` builds placeholders from the graph's tensors, then a
feeds dictionary keyed by them, then runs the graph
(`aten/src/ATen/native/mps/operations/Indexing.mm`). A clear on another thread
in that window leaves the keys pointing at freed objects, and building the
dictionary is where the renders died. `torch.accelerator.empty_cache()`
drains the allocator alone and leaves the graphs.

**The worker drained.** `release_torch_memory` drains torch's MPS caches
whenever memory is under pressure, and the batch-prep worker calls it after
each primitive it projects (`primitives._release_unpacked_geometry`, from
`_pack_projected_flat_geometry`) and in `project_to_screen`, from
`render_loop._prewarm_render_batch`. In a warm UHD render with nothing
forced, about 0.8 G of the runner's 7 G was free, and the worker drained four
times (run 42422403489).

**It was made to happen** (run 42422403489):

* A second thread calling `torch.mps.empty_cache()` in a loop killed a process
  running MPSGraph ops on its main thread 4 times of 4, each within 1.5 s and
  about ten ops: twice with `SIGSEGV`, and twice with `SIGABRT` on an
  unrecognized selector sent to whatever then held the graph's memory, each
  while `structured_mm_out_mps` built its feeds. The same loop killed none of 3
  with `torch.accelerator.empty_cache()`, none of 3 with
  `driver_allocated_memory()` and none of 2 with
  `set_per_process_memory_fraction`, each process running its 15 s, about a
  hundred ops.
* In UHD renders, host pressure forced on, so that the worker drained in every
  render, killed 1 of 4 processes, and a side thread also reclaiming every
  0.25 s, as the worker does, killed 4 of 4, with 4 crashes in 5 renders. Each
  died at `index_select_out_mps +1608` as before. The same two schedules with
  the allocator-only drain killed none of 3 processes, in 6 renders.
* `torch.mps.synchronize()` from a second thread aborted 2 of 2. That is
  another torch defect, and Algan calls it only on the render job's thread.

**The fix keeps drains on the render job's thread.**
`memory_utils.drain_mps_caches()` makes every MPS drain now, for
`release_torch_memory` and for `get_num_available_bytes` on MPS. While a
render job runs, only the thread that entered it drains
(`taichi_runtime.on_the_render_job_thread`). A request from any other thread
sets `_MPS_DRAIN_DEFERRED`, and the job's thread makes the drain at its next
`release_torch_memory`, pressured or not, or at the job's end, after the
worker's executor has shut down: one drain for however many requests came in.
A free figure measured on another thread is measured without the drain, and
only the job's thread sizes an MPS arena during a job. With no job active,
any thread drains at once, as before. `MPS_DRAIN_STATS` counts the drains, the
requests deferred and the drains that served them. One thread is enough
because it is the only one using cached graphs: GPU projection and merge are
CUDA-only (`project_on_gpu_active`, `merge_on_gpu_active`), so on a Mac the
worker's torch work runs on the CPU. In one UHD render all 1,376 Quadrants
launches were the main thread's, and the worker ran one MPS op, an
`aten.zeros`, which torch 2.13 fills with a Metal kernel and no graph (run
42424700295).

**Checked** (run 42424700295):

* Before the fix the two new tests failed, and the race test's child died 4
  times of 4, each process within 6.4 s of starting; each of the three that
  showed where died in the matmul. After it the child lived 6 times of 6,
  every drain on the job's thread, and the touched files 148 passed and 1
  skipped, which needs CUDA, in 85.8 s.
* No UHD render of 21 crashed after the fix: 8 with the reclaiming side
  thread, in 4 processes, 6 with host pressure forced, in 3, and 7 plain,
  without drains, with the allocator-only drain, and counting ops. Provoked
  processes went from 5 of 8 crashed to 0 of 7 (one-sided Fisher p = 0.019),
  and the child from 4 deaths of 4 to 0 of 6 (p = 0.005).
* Every full drain during a job was on the main thread. With the side thread,
  111 to 234 requests a render were served by 24 to 26 drains the render
  thread made anyway, and with pressure forced, 4 requests by 1.
* 18 frames of 18 decoded identically, with a largest difference of 0, before
  the fix and after, and between the plain arm and the arms without drains and
  with the side thread. On Linux the touched files 145 passed and 4 skipped,
  and `tests/fast/test_fast_render.py` matches its baseline. CUDA runs none
  of the new code, which is gated on `torch.mps.is_available()` or an MPS
  device; that is read from the code, not measured.

The tests are `tests/unit_tests/test_mps_cache_drains.py`'s two. A stubbed one
runs anywhere, in about 0.2 s, and checks that the worker's pressured and
forced reclaims and a free-figure measure drain nothing during a job, that the
render thread's next reclaim makes one drain for them, and that the job's end
makes a later one. On the Apple GPU, the other runs a render job's thread
doing `index_select` and a matmul beside a thread reclaiming in a loop, in
about 15 s.

**The cost is the drains the render thread already made.** A full drain costs
the render thread 30 to 110 ms, 0.6 to 2.7 s of a 36 to 93 s UHD render, and
the allocator-only drain 2 to 4 ms. The fix adds no drain: a deferred request
joins one the render thread makes, and those were fewer than the worker's
own. Warm render times were too noisy to show a cost (one arm spread up to
30 s), and rebuilding each graph after a full drain was not timed.

**The sampler only moved the drains.** §4.11 found the crash rate rose with
how often a probe thread called `torch.mps.driver_allocated_memory()`. That
call alone never crashed, in the race or as the worker's own queries through
the 21 renders after the fix, so the probe changed when a drain landed and
caused nothing itself. GIL traffic shifting how the threads interleave is the
likeliest reading, and it is not measured.

**Seen and not chased**, recorded so the next reader starts from the
measurement:

* The render thread's own drains stay full: it drains on every `force_gc=True`
  reclaim, 16 to 21 times in a UHD render. The allocator-only drain inside a
  render, with the graph clear kept for the job's ends, would cost 2 to 4 ms a
  drain rather than 30 to 110, but the graph cache would grow through the
  render, and each torch would need checking for what its
  `torch.accelerator.empty_cache` drains.
* `ALGAN_ANIMATION_DEVICE=mps` puts the worker's torch work on the GPU, and
  then the render thread's own drains can delete a graph the worker is using.
* Two render jobs entered on two threads at once would leave the second's
  drains to the first. Algan enters one at a time: the viewer serializes its
  renders under `_scene_lock`.
* The worker's prep retry (`force_gc=True`) leaves its MPS drain to the render
  thread now. Its collection and heap trim still run at once.
* Which line makes the worker's one MPS `zeros` was not traced.
