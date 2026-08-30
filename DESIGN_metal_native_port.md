# Native Metal shaders for Algan's kernels: feasibility

Status: **Feasible, measured, and started.** Stage 1 (the capability probe) and
stage 2 (the arena-offset marshalling layer) are landed; §5 has the rest. The
mechanism is materially better than `DESIGN_mps_support.md` concluded — but what
remains is a rewrite of the kernel layer, not a port — sized below at ~9,000
lines of Taichi across 52 kernels and 169 `@ti.func` helpers, whose cost is
dominated by the 92 compile-time specialization gates rather than by the kernels
themselves.

This doc answers a question `DESIGN_mps_support.md` never asked. That one
measured **Taichi on the Metal backend** and reached a defensible NO-GO. This one
asks what happens if **Taichi is not in the path at all**: hand-written Metal
Shading Language, dispatched onto `torch` MPS tensors directly. Two of that
doc's three blockers are properties of Taichi's interop and codegen, not of
Metal, and they do not survive the change of mechanism.

**`DESIGN_mps_support.md`'s verdict stands as written, for the path it
measured.** Nothing here contradicts a number in it. Read §1 for what changes,
§2 for what does not, §3 for the real cost, §4 for what the probe found.

**§1 is now measured, not argued.** Probe Q8 (`--section msl`) on
`macos-latest`, against this branch:

| question | answer |
| --- | --- |
| `compile_shader` compiles and dispatches hand-written MSL | **yes** |
| shader writes *through* the torch allocation (`data_ptr` unchanged) | **yes** — §1.1 confirmed |
| a sliced view binds at its own `storage_offset` | **yes** (offset 256 honoured, storage 0 untouched) |
| arena + offset table replaces 49 bindings with 2 | **yes**, `max_abs_error 0.0` |
| widest MSL kernel that binds | **30 buffers** (31 is a clean compile error, not an abort) |
| grid and threadgroup settable from Python | **yes** — `threads=`, `threads=`+`group_size=` |
| f32 vs f64 host reference, sRGB encode | **0** channel delta, both `pow` flavours |
| f32 atomic add | **yes**, exact (4096/4096) — a non-deterministic mode has its floor |
| MSL 64-bit atomics | **no** — `no matching function for atomic_fetch_add_explicit` |

Rows 1–6 and 8 are from run 33163279074; the grid and f32-atomic rows from
[33164064471](https://github.com/algorithmicsimplicity/algan/actions/runs/33164064471),
which re-ran them after §4.1 found two faults in the harness. §3's sizing is from
the repo's own AST. **Nothing in §4 is open.**

---

## 1. What the change of mechanism dissolves

### 1.1 The staging copy — gone, and with no C++ extension

`DESIGN_mps_support.md` §1.3 is the blocker that left every bandwidth-bound
stage far slower on Metal than on CPU (reported as 53x; see §3.3 on how much
weight that multiplier carries): Taichi's `kernel_impl.py` stages any
ndarray argument through host memory, because Taichi implements
`Device::import_memory` only for `CpuDevice` and `CudaDevice`.

That is a statement about Taichi. Torch has its own answer, and it is already in
the version this project pins:

```
torch 2.7.1 → torch.mps.compile_shader(source) → library; lib.kernel_name(tensor, scalar, ...)
```

It is a documented public API (`torch/mps/__init__.py`), and underneath it is
`at::native::mps::MetalShaderLibrary`
(`torch/include/ATen/native/mps/MetalShaderLibrary.h`), whose kernel handle
exposes exactly what a dispatch layer needs:

| capability | signature |
| --- | --- |
| bind a tensor's `MTLBuffer` at a buffer index | `setArg(unsigned idx, const at::TensorBase&)` |
| bind a scalar or a POD struct by value | `setArg(unsigned idx, const void* ptr, uint64_t size)` |
| 1-D / N-D dispatch, optional threadgroup size | `dispatch(length, groupSize)`, `dispatch(ArrayRef<uint64_t>, ...)` |
| compile a variant of one source | `getLibrary(std::initializer_list<std::string> params)` |
| pipeline + library caching | `libMap`, `cplMap` |

`setArg(idx, TensorBase&)` binds the tensor's own storage. On MPS that storage
*is* an `MTLBuffer` from torch's allocator, so there is no copy, no import, and
nothing to adopt — which is precisely the operation Taichi's gfx device cannot
perform. This removes §1.3 entirely rather than mitigating it, and it does so
without the ObjC++ extension such a port would otherwise need.

**Measured.** The `zero_copy` case takes a tensor's `data_ptr`, launches a
shader that writes to it, and checks both that the pointer is unchanged and that
the values arrived: `data_ptr_stable: true, written_in_place: true`. The
allocation Algan already holds is the one the shader wrote to. For contrast, the
same run's Q1 reports `engine believes launch is staging-free: False` for the
Taichi path on the same machine — the two mechanisms, one GPU, opposite answers.

The last row matters as much as the first: `getLibrary(params)` is a
source-parameterized compile with a library cache keyed on the parameters. That
is the mechanism `ti.template()` specialization has to land on (§3.2), and it
already exists.

### 1.2 The ~24-buffer argument limit — mostly dissolves, because of the arena

`DESIGN_mps_support.md` §1.1 calls this "the blocker", and on the numbers it is:
6 kernels bind more than 24 ndarrays, and they are all the ones that matter.
Measured afresh from the AST (the two rows this survey originally carried for
`wavefront_traverse` and `wavefront_shadow` are gone: both kernels were
unreachable and have since been deleted):

| kernel | ndarrays | template gates | scalars | body lines |
| --- | --- | --- | --- | --- |
| `sheet_resolve_shade` | 49 | 15 | 7 | 1065 |
| `wavefront_shade` | 38 | 19 | 11 | 1340 |
| `raster_shadow_trace` | 30 | 7 | 10 | 300 |
| `wavefront_traverse_events` | 30 | 6 | 14 | 183 |

The doc's read is that clearing this means splitting the megakernels — cutting
across the fusion they exist for — or "packing many arrays into single buffers
and indexing inside the kernel, which rewrites every signature in the renderer."

**The second option is much cheaper here than that sentence assumes, because
Algan already packed them.** `ManualMemory` (`utils/memory_utils.py:608`) is a
bump allocator over **one** `torch.empty((num_bytes,), dtype=torch.uint8)`
allocation, and every render-time tensor it hands out is a slice view of it —
`self.data[pointer:new_pointer]`, re-viewed to the requested dtype and shape.
Every arena-allocated argument to those eight kernels is therefore already the
same `MTLBuffer` at a different offset.

So the packing that Metal wants is not a rewrite of the data layout. It is:

* bind `arena.data` **once**, as `device uchar*`, at buffer index 0;
* pass one `setBytes` struct of `uint` offsets — the `pointer` each view starts
  at, which the arena already knows;
* inside the shader, reinterpret at offset (`(device const float*)(arena + off.tri_pos)`).

A 49-buffer kernel becomes 2 bindings. Non-arena arguments (textures, persistent
scene tables) stay as ordinary bindings and there is ample room left for them.
This is a real cost — every kernel signature and every launch site changes — but
it is mechanical, it is local to the dispatch layer, and it does not touch the
fusion. It is not the redesign the earlier doc priced.

One caveat worth stating rather than discovering: pointer arithmetic on a single
buffer gives up the aliasing guarantees separate bindings imply. Alignment is
already handled — the arena aligns each allocation to its element size — but a
shader that writes through two offset pointers into one buffer is the compiler's
worst aliasing case, and `device` qualifiers alone will not fix it.

**Measured on both sides, and the packing works.** Taking the arena side
first: every wide kernel's bindings, planned from the arguments it was actually
handed during a render. What §1.2 asserted from reading the allocator now comes
from launches —

| kernel | ndarray args | arena-backed | passthrough | bindings | seen in |
| --- | --- | --- | --- | --- | --- |
| `sheet_resolve_shade` | 49 | 48 | 1 | **3** | guard |
| `wavefront_shade` | 40 | 40 | 0 | **2** | fast scene |
| `wavefront_traverse_events` | 34 | 34 | 0 | **2** | fast scene |
| `raster_shadow_trace` | 32 | 24 | 8 | **10** | `materials_and_lighting` |
| `raster_tri_write` | 20 | 18 | 2 | **4** | guard |
| `raster_tri_count` | 15 | 14 | 1 | **3** | guard |
| `raster_bez_write` | 18 | 16 | 2 | **4** | guard |
| `raster_bez_count` | 14 | 13 | 1 | **3** | guard |

"guard" is `test_arena_binding_live.py`, whose small two-shape scene is what CI
re-checks on every run. The other three rows come from running the same
instrumentation over `tests/fast/scene.py` and
`tests/full_renders/scenes/materials_and_lighting.py`, which reach kernels that
scene does not; the guard skips a kernel it never launches rather than
pretending to cover it. The two path tracers remain unobserved — no scene run
so far reaches them. (`wavefront_traverse` and `wavefront_shadow` were also
unobserved, and that turned out to be the finding rather than a gap in the
scenes: neither had a reachable caller, and both have been deleted.)

The number that decides this is not how many arguments a kernel takes but how
many are **not** arena-backed, because those keep their own binding however well
the arena works. `sheet_resolve_shade` has one. The margin is not narrow.

`raster_shadow_trace` is the interesting row — a quarter of its arguments are
outside the arena — and it still lands at 10. Nothing observed comes close to
31.

Two properties the same test pins, because on Metal either would be wrong
pixels rather than a crash: every packed argument is **contiguous** (a shader
reconstitutes an array from a base pointer and a length; it has no stride
vector) and every offset is a whole number of elements. The second turns out to
be guaranteed twice over — torch refuses to build a misaligned reinterpreting
view at all, independently of the arena's own alignment.

Now the Metal side. The MSL ladder binds 30 buffers and stops:

```
30 buffers  ok
31 buffers  error  'buffer' attribute parameter is out of bounds: must be between 0 and 30
```

So the ceiling is 31 slots, indices 0–30 — six more than Taichi managed on the
same machine (Q4: 24), because Taichi spends slots on its own context and root
buffers. It is nowhere near 49, which settles that the arena convention is
*necessary* and not merely tidy.

Two things about that failure are worth keeping. It is a **compile error with a
line and a column**, where Taichi's equivalent was a `SIGABRT` inside
`setComputeFunction` — the hand-written path fails legibly. And the `arena` case
proves the way around it end to end: one `uchar` buffer, an offset table, three
float arrays packed into it, `max_abs_error: 0.0`.

The `view_offset` case answered better than the plan required: a slice taken at
`storage_offset 256` was written **at its own offset**, with storage zero
untouched. Torch's shim honours `storage_offset`, so arena views can be passed
as ordinary tensor arguments where a kernel stays under 31 bindings, and the
explicit offset table is needed only to get *under* that count — not to address
the memory correctly.

**Done, on the Taichi side, ahead of the port.** Every kernel in the table above
now takes its cold arrays through the arena
(`algan/rendering/raytracing/arena_args_taichi.py`). The widest kernel in the
package asks for 20 ndarray arguments; nothing is over 24, and
`tests/unit_tests/test_arena_args.py` fails if a new one appears.

Three things about how it landed are worth carrying into the port:

* **The hot arrays stay ordinary parameters.** Binding *everything* costs 18% of
  `sheet_resolve_shade`'s device time; binding everything except the seven
  slot-indexed ray-state arrays costs 1.7–3.0%, and keeping thirteen more on top
  of those bought nothing further. The cost is per *access*, so the split that
  matters is per-thread state (parameters) against scene tables (arena).
* **The aliasing caveat above did not materialise.** Giving each ray-state array
  its own arena is +19% against one shared arena's +18%; splitting stores from
  loads buys about a point. What the cost actually is —Taichi re-loading base
  pointers and shapes from a global-memory argument buffer at every use site —
  is in `DESIGN_taichi_argument_loads.md`, along with the Taichi fork that would
  remove it. An MSL kernel taking a `device uchar*` and a `setBytes` offset
  struct pays none of it: the offsets arrive in registers.
* **No launch site changed.** `arena_packed` wraps each converted kernel and
  splits the original positional argument list, so the ~12 call sites in
  `tracer.py`, `raster_pipeline.py` and `path_tracer.py` are untouched. The
  dispatch layer can do the same.

---

## 2. What does not change

### 2.1 f64 is genuinely absent from MSL — but it is contained

Metal Shading Language has no `double`. No mechanism recovers this; it is not a
codegen limit that hand-written MSL routes around.

The good news is the blast radius, which the earlier doc did not quantify:
**`ti.f64` appears in exactly one kernel module**,
`rendering/raytracing/sheet_compact_taichi.py` (12 occurrences). Everything else
in the kernel layer is already f32/i32/i64. The f64 accumulator exists for a
stated reason — `raster_pipeline.py:1345-1364` records CUDA's unordered float
atomics moving two consecutive renders by up to 28 channel values over 9.6% of a
frame — so removing it is a correctness decision, not a typing exercise, and it
is the same decision on Metal as the earlier doc reached.

### 2.2 i64 atomics: re-ask, do not assume

`DESIGN_mps_support.md` §1.2 measured `ti.atomic_add`/`atomic_min` on i64
*aborting* on Metal, and correctly concluded that the deterministic
fixed-point accumulator cannot run in a Taichi Metal kernel.

That measurement is of Taichi's SPIR-V path, so it was worth re-asking of
hand-written MSL. **Asked, and the answer is the same: no.**

```
atomic_u64_add   error   program_source:9:5: error: no matching function for call to
                         'atomic_fetch_add_explicit'
atomic_u64_min   error   program_source:9:5: error: no matching function for call to
                         'atomic_fetch_min_explicit'
```

`atomic_ulong` names a type the MSL standard library will not give an
`atomic_fetch_*` overload for on this toolchain. Note what kind of failure this
is: a **compile** error out of the MSL front end, not a device rejection at
pipeline build. That points at the language/stdlib level rather than at the
runner's virtualized GPU, which makes it unlikely — though not proven — that
real Apple silicon answers differently. If that distinction ever matters, the
case is one line to re-run on a physical Mac.

So the conclusion is the earlier doc's, now established for both mechanisms: a
deterministic mode needs the accumulation restructured — segmented reduction
over sorted keys, order-independent without needing a wide atomic — rather than
an atomic.

**f32** atomics do give a non-deterministic mode its floor: 4096 concurrent
`atomic_fetch_add_explicit` calls on a `device atomic_float*` summed to exactly
4096. The first run appeared to say otherwise and was measuring the harness
rather than Metal (§4.1).

### 2.3 The torch-side op gaps are unaffected

§2.3 of the earlier doc — `scatter_reduce_(amin/amax)` on int64, `cummax` f32,
f64 `zeros` — are gaps in torch's MPS backend, reached from Python, and a Metal
shader layer does not touch them. `_one_mesh_pixel_caps`
(`raster_pipeline.py:1330-1332`) still needs an answer. Two of the three are
plausibly *easier* once a shader layer exists, because a custom kernel can do
the per-pixel id spread directly instead of asking torch for it.

---

## 3. The real cost, which is not the kernels

### 3.1 Size

Comments and docstrings stripped, per module:

| module | raw | code | kernels | `@ti.func` |
| --- | --- | --- | --- | --- |
| `raytrace_kernels_taichi.py` | 4020 | 2695 | 3 | 58 |
| `wavefront_kernels_taichi.py` | 3720 | 2206 | 13 | 32 |
| `raster_taichi.py` | 3059 | 1575 | 5 | 34 |
| `shading_taichi.py` | 1752 | 691 | 0 | 33 |
| `sheet_resolve_taichi.py` | 1217 | 895 | 2 | 0 |
| `sheet_compact_taichi.py` | 647 | 195 | 13 | 0 |
| `logical_pn_taichi.py` | 498 | 297 | 3 | 6 |
| `glossy_prefilter_taichi.py` | 326 | 159 | 3 | 2 |
| others (7 modules) | 681 | 311 | 10 | 4 |
| **total** | **15,920** | **~9,024** | **52** | **169** |

MSL is more verbose than Taichi's Python surface — explicit types, no tuple
returns, no Python-level metaprogramming, manual struct plumbing where Taichi
infers. **Estimate 12,000–18,000 lines of MSL**, plus a Python dispatch and
specialization layer to replace what the `@ti.kernel` decorator does today.

Note that `shading_taichi.py` has **zero** kernels and 33 helpers: it is a
library inlined into other kernels. The 169 helpers, not the 52 kernels, are the
bulk of the semantic content, and they are where numerical drift will hide.

### 3.2 The specialization layer is the load-bearing new component

The kernel modules reference `ti.template` **590 times** and `ti.static` **428
times** (AST node counts, so comments and docstrings are excluded) — of which 92
are template gates on kernel signatures, 15 on `sheet_resolve_shade` and 19 on
`wavefront_shade`. Taichi gives this away: each
distinct combination of template arguments compiles to its own specialized
kernel, and `ti.static` gates fold out at compile time.

This is not incidental. CLAUDE.md is explicit that a `ti.static` gate resolves
when the kernel compiles, that flipping the setting behind it mid-process does
nothing, and that A/B arms must run one process each. `sheet_resolve_shade`'s
`mode` gate is load-bearing by design — "one kernel body for all three is what
makes a resolve/shadow desync structurally impossible."

In MSL that becomes some combination of:

* **function constants** (`[[function_constant(n)]]`) — specialized at pipeline
  build, which is the closest analogue and keeps one source;
* **preprocessor macros** through `getLibrary(params)` — which is how torch's own
  bundled kernels specialize on dtype;
* **generated source** from Python.

All three work. None is free, and the variant space is combinatorial: 19 gates is
not 19 variants. Taichi materializes a variant lazily on first launch with the
arguments actually used, and the equivalent bookkeeping — enumerate reachable
combinations, key a pipeline cache, keep cold-compile cost bounded — has to be
built. **This, not the shader arithmetic, is where the project's risk sits.**

### 3.3 What is bought — and why nobody here can yet say how much

From `DESIGN_mps_support.md` §1.3's own table, the compute-bound kernel on Metal
ran at 7.45 ms against the CPU arch's 166.05 ms — **22x**, through the staging
path, and 52x with the staging removed. Removing staging is exactly what §1.1
here does.

**Treat every one of those numbers as directional only.** They were taken on
GitHub's `macos-latest` runner, which is a virtualized Apple-silicon instance:
sound for asking whether an operation *works*, and — as the sub-section below
later established, after this paragraph was written — sound for sustained
arithmetic rate too, but not for anything with a per-launch cost in it, which a
staged Q5 launch is almost entirely made of. That caveat is not in
`DESIGN_mps_support.md`, and it cuts both ways —

* the *staging* result is the robust half. A host round-trip is a host
  round-trip, and a 46x gap between "copied through the host" and "bound in
  place" is far too large to be an artifact of virtualization. The ordering
  survives; the multiplier is not a number to plan against.
* the *compute-bound* result is the fragile half. "52x faster than the CPU arch"
  is exactly the claim a paravirtualized GPU distorts, in either direction, and
  it is the claim that decides whether the port pays.

So the shape of the argument holds — removing staging is a large win, and the
compute-bound kernels are where an Apple GPU could earn its place — while the
magnitude is unestablished. **Sizing the payoff needs real Apple hardware, and
no CI runner substitutes for it.** This is a reason to keep the port's early
stages cheap (§5) rather than a reason not to start.

Two further runs took the same Q5 measurement and are the evidence for the caveat
rather than exceptions to it. Three readings now exist of each arm:

| | bandwidth-bound (slower) | compute-bound (faster) |
| --- | --- | --- |
| earlier run (`DESIGN_mps_support.md`) | 53x | 22x |
| 33163279074 | 39x | 20x |
| 33164064471 | 39x | 30x |

The signs never move. The multipliers move by up to half, on the same runner
class, measuring the same thing. That is not a quantity anyone should plan
against — but it is ample to say *which* stages an Apple GPU would help.

#### The runner's GPU is real, and its dispatch is not

Run [33178820377](https://github.com/algorithmicsimplicity/algan/actions/runs/33178820377)
(`benchmarks/_mps_vs_cpu_torch_speed.py`, `macos-latest`, Linux control green)
took the measurement this section had been reasoning around: identical torch
tensor work on `cpu` and on `mps`, no Algan and no Taichi anywhere in the path,
`PYTORCH_ENABLE_MPS_FALLBACK` cleared, every result checksummed against the CPU
arm's. The machine is `VirtualMac2,1` / `Apple M1 (Virtual)`, 3 CPUs, 7 GB,
`kern.hv_vmm_present 1`, and MPS offers 5.0 GB.

| | cpu | mps | mps/cpu |
| --- | --- | --- | --- |
| matmul f32 512 | 0.278 ms (966 GFLOP/s) | 1.249 ms (215) | **0.22x** |
| matmul f32 1024 | 4.663 ms (461) | 2.619 ms (820) | 1.78x |
| matmul f32 2048 | 40.700 ms (422) | 11.171 ms (**1538**) | 3.64x |
| matmul f32 4096 | 283.298 ms (485) | 105.041 ms (1308) | 2.70x |
| conv2d f32 4x64x128 | 45.444 ms (106) | 3.932 ms (1229) | 11.56x |
| elementwise f32 16M | 9.210 ms (29.1 GB/s) | 5.714 ms (47.0) | 1.61x |
| reduction f32 16M | 1.942 ms (34.6 GB/s) | 2.251 ms (29.8) | 0.86x |
| one synchronized dispatch | 2.0 us | **432 us** | 0.005x |
| 64 MB host→device / device→host | — | 6.0 / 3.2 GB/s | — |

Two things follow, and they pull in opposite directions.

* **There is a real GPU here.** 1538 GFLOP/s of f32 is four times what these
  three cores sustain and not a number a software Metal implementation
  reaches. The compute-bound half of §3.3's argument — the half that decides
  whether the port pays — is therefore corroborated by a measurement with no
  Taichi in it, and it lands in the same place Q5's 20-30x did.
* **Per-launch cost on this runner is fiction.** 432 us for one dispatch against
  the CPU's 2.0 us, and 6.0 / 3.2 GB/s across a bus that is *the same DRAM*, are
  taxes on command submission and buffer mapping that no Apple laptop pays.
  That is what makes the 512 matmul come back at 0.22x: at 0.278 ms of work, the
  row is measuring submission and nothing else. It is also the reason Q5's
  staging multipliers swing by half between runs — a staged launch is
  per-launch cost with the arithmetic as a rounding error.

So the caveat sharpens rather than lifts. **Compute throughput on this runner is
worth reading; anything with a per-launch term in it is not.** The crossover here
sits somewhere between a 512 and a 1024 matmul — around a third of a millisecond
of arithmetic — which is a bound on this runner and not on hardware.

#### Why 3.6x and not 30x: the denominator

A follow-up run,
[33179777864](https://github.com/algorithmicsimplicity/algan/actions/runs/33179777864),
asked whether that modest ratio meant a GPU on a small virtualized slice or a cpu
arm that is not the weak baseline Q5's 20-30x was taken against. It is the
second, and both halves are now measured rather than inferred.

**The cpu arm runs Apple's own BLAS.** `torch.__config__.show()` on the runner
reports `BLAS_INFO=accelerate | LAPACK_INFO=accelerate`, so a torch matmul there
is an Accelerate SGEMM — a vendor kernel that reaches the AMX matrix
coprocessor, not the NEON loop three cores would otherwise give. Taichi's `cpu`
arch compiles neither: it emits an LLVM kernel over the arithmetic as written.
That is the whole difference between the two denominators, and it is why 22x and
3.6x are both correct measurements of the same GPU.

**Neither device has much headroom left.** The ceiling sweep runs the same matmul
at four sizes per device, past where launch cost and cache effects matter:

| device | dtype | 2048 | 4096 | 6144 | 8192 |
| --- | --- | --- | --- | --- | --- |
| cpu | f32 | 350.4 | 443.5 | 424.2 | 452.2 |
| mps | f32 | 1564.6 | 1237.3 | 1327.0 | 1293.4 |
| mps | f16 | 2086.4 | 2267.0 | 2262.8 | 2284.8 |

Both curves are flat: ~450 GFLOP/s sustained on three cores, ~1.3 TFLOP/s f32 and
~2.3 TFLOP/s f16 on the GPU. An M1's integrated GPU is worth roughly 2.6 TFLOP/s
f32 at peak, so this VM is being handed something close to a whole one at
ordinary SGEMM efficiency rather than a throttled slice. **The GPU here is not
small. The CPU it is being compared against is unusually good**, and a ratio
against a scalar baseline — which is what the renderer's kernels are — is the
one Q5 reports.

One row in the comparison table above should be read with this in mind and
otherwise discarded: `matmul_f16_1024` at 376x is not a GPU result. torch has no
native f16 GEMM on the CPU side, so that arm runs at 1.9 GFLOP/s against its own
525 in f32. The f16 row worth keeping is the ceiling sweep's, and what it says is
that the GPU's f16 path is a further 1.75x over its own f32 — real, and nothing
like 376.

---

## 4. What the probe found

Runs [33163279074](https://github.com/algorithmicsimplicity/algan/actions/runs/33163279074)
and [33164064471](https://github.com/algorithmicsimplicity/algan/actions/runs/33164064471),
`macos-latest`, Q8 (`--section msl`), Linux control green on both. The five
unknowns this section previously listed came back:

| unknown | answer |
| --- | --- |
| 1. Python binds an MPS tensor with no copy, at the index implied by argument order | **yes** (§1.1) |
| 2. Grid and threadgroup reachable from Python, or C++-only | **reachable** — `threads=`, `+group_size=` |
| 3. The real buffer ceiling | **31 slots, indices 0–30** (§1.2) |
| 4. Hand-written MSL 64-bit atomics | **no**, a compile error (§2.2) |
| 5. f32 shader vs an f64 host reference | **0 channel delta**, both `pow` flavours |

Unknown 5 is worth dwelling on, because it was the one most likely to sink the
port quietly. Max float error `2.04e-07`, and after rounding to u8 the shader and
the f64 host reference **agree on every channel** — with fast-math on, which is
MSL's default and the thing that was expected to bite. `precise::pow` was not
needed for sRGB. That does not license every kernel (a path tracer accumulates
differently from a tone curve), but it removes the fear that MSL arithmetic is
categorically off the CPU path.

The failure modes are also better than Taichi's on the same machine. Every
negative here is a **compile error naming a line and a column**; the Taichi arms
answered the same questions with `SIGABRT` inside `bind_pipeline` and
`setComputeFunction`. For a port measured in thousands of shader lines, the
difference between "error at 9:5" and "the process died" is most of the
debugging cost.

### 4.1 Two things the first run got wrong, both the harness's fault

Neither was a Metal result. Both are fixed, and run 33164064471 is where the
table's grid and f32-atomic rows come from. They are recorded rather than
quietly corrected, because each is a trap the port itself can fall into.

* **The f32 atomic case under-dispatched.** It reported `total: 1.0` against an
  expected `4096.0`, which reads as a broken Metal atomic and is nothing of the
  kind. The `grid` case established in the same run that the shim dispatches
  over **argument 0's element count** — and that kernel had the one-element
  accumulator in slot 0, so it ran on exactly one thread and added exactly once.
  The arrays are now ordered wide-first, and any case returning a `matches`
  verdict now reports `wrong_result` rather than `ok`, which is what let a wrong
  answer through in the first place. Re-run: **4096.0 of an expected 4096.0**,
  so f32 atomics are exact here and a non-deterministic mode has its floor.
* **The shim surface was printed truncated**, at 180 characters, which left
  unknown 2 unanswered for a whole run: the dump confirms a `_mps_MetalKernel`
  with something named `max_threads_per_…` and is clipped before the rest. A `dispatch_control` case
  now tries the plausible call forms and records which the shim accepts, and the
  section prints results untruncated. Re-run: **`threads=` and
  `threads=`+`group_size=` are both accepted**; `grid_size=` and `threadgroup=`
  are not.

That second answer is the one the port needed. Had the grid been *only* argument
0's element count, the arena convention would have put the whole arena — millions
of bytes — in slot 0 and launched a thread per byte, retiring almost all of them
on a guard; the way out would have been a thin ObjC++ extension reaching
`MetalKernelFunction::dispatch` directly. It is not needed. Thread count and
threadgroup size are both reachable from Python, so **the port needs no C++ at
all** — which was the last structural question hanging over it.

### 4.2 What no CI runner can answer

Anything with a **per-launch** term in it. `macos-latest` is a virtualized
Apple-silicon instance with a real GPU behind it (§3.3), so it establishes that
a shader compiles, binds, dispatches and returns the right bits — the whole of
§4's table, and what has to be true before performance is even a question — and
its *sustained arithmetic* rate can be read as well. What it charges for getting
work to that GPU cannot: 432 us per synchronized dispatch against its own CPU's
2.0 us, and 6.0 / 3.2 GB/s across unified memory.

That is precisely the axis this port's shape turns on. Launch overhead per
dispatch, and whether the many-small-kernel stages (`sheet_compact_taichi.py`
has 13 kernels averaging ~15 lines of code) want fusing on the way across, are
real questions and they need a physical Mac — a runner that inflates the
per-launch term by two orders of magnitude will say "fuse everything" whatever
the truth is. Q8 stays untimed for that reason rather than reporting a number
that reads authoritative and is not, and the probe's Q5 timings, its module
docstring, its printed verdict and the workflow header all carry the sharpened
caveat.

These runs illustrate why. Q5's compute-bound arm came back **22x**, **20x** and
**30x** faster than the CPU arch across three readings, and the bandwidth arm
**53x**, **39x** and **39x** slower. The signs never move; the multipliers move
by half. Neither is a number to plan against.

## 5. Recommended shape, if it goes ahead

Not one change. In dependency order, each stage independently useful:

1. **Probe `compile_shader`** (§4) — **done**, runs 33163279074 and 33164064471.
   All five unknowns answered, four of them the way the port needs; the fifth
   (64-bit atomics) is a real constraint on §2.2 and nothing else. Stage 2 is
   unblocked, and no ObjC++ extension is required anywhere.
2. **Arena-offset calling convention** — **done for the marshalling half**
   (`algan/rendering/arena_binding.py`), which is the half that survives.
   Scoped down from "convert the kernels" once §1.2 was measured, and the
   reason is worth stating: **the Taichi kernel bodies do not carry forward.**
   MSL replaces them, it does not translate them, so rewriting 9,000 lines of
   Taichi into offset indexing would buy the port nothing while costing the
   shipped CUDA and CPU renderers real indirection for a limit neither has.
   What both backends genuinely share is the marshalling — which arguments are
   arena-backed, at what offset, and whether what is left still fits in 31
   slots — and that is now a library with a live regression guard, imported by
   nothing on a render path, so it costs a shipped frame nothing. The kernel
   bodies belong with the shaders, in stages 3 and 6.
3. **A vertical slice**: `tonemap_to_u8` + the three `bloom_*` kernels. 2–3
   ndarrays each, ~130 lines of code across the two modules (207 raw), no BVH,
   no specialization beyond `tonemap_to_u8`'s 3 gates, and a pixel-comparable
   output. This proves dispatch, precision and the
   baseline story end to end at ~2% of the total size.
4. **The specialization layer** (§3.2), designed against the slice's real gates.
5. **The raster count/write kernels** — the four that already fit in 24 buffers.
6. **The megakernels**, largest last, on the convention stage 2 established.

Stages 1 and 2 are done. Stages 1–3 are perhaps a week in total. They answer
whether the remaining 90% is *possible* — dispatch, precision, specialization, baselines — and CI can carry
all of it. They do **not** answer whether it is worth it: that is the magnitude
question of §3.3, and the first honest reading of it comes from running stage 3's
slice on a physical Apple GPU. That is a cheap thing to ask of one machine once,
and it is the natural gate before stage 6.

Going straight at the megakernels is the way to spend months and find out at the
end.

**The honest summary: this is a multi-month rewrite of the renderer's kernel
layer with a hard validation constraint** — the repo's standard is byte-identical
output checked pixel-wise, CUDA and CPU baselines are separate committed sets,
and a Metal set would be a third, verifiable only on Apple hardware. What makes
it *worth considering* despite that is §1: the two blockers that made the Taichi
route hopeless are artifacts of Taichi, and the one piece of packing work Metal
genuinely demands, Algan's arena has already done.
