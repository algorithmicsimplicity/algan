# Native Metal shaders for Algan's kernels: feasibility

Status: **Feasible, and the mechanism is materially better than
`DESIGN_mps_support.md` concluded — but it is a rewrite of the kernel layer, not
a port.** Sized below at ~9,000 lines of Taichi across 52 kernels and 169
`@ti.func` helpers, whose cost is dominated by the 92 compile-time specialization
gates rather than by the kernels themselves.

This doc answers a question `DESIGN_mps_support.md` never asked. That one
measured **Taichi on the Metal backend** and reached a defensible NO-GO. This one
asks what happens if **Taichi is not in the path at all**: hand-written Metal
Shading Language, dispatched onto `torch` MPS tensors directly. Two of that
doc's three blockers are properties of Taichi's interop and codegen, not of
Metal, and they do not survive the change of mechanism.

**`DESIGN_mps_support.md`'s verdict stands as written, for the path it
measured.** Nothing here contradicts a number in it. Read §1 for what changes,
§2 for what does not, §3 for the real cost, §4 for what is still unmeasured.

**None of this has run on an Apple GPU.** The probe that would settle it is
written and waiting for one macOS CI run (§4). Until it reports, this doc is an
argument from the torch headers and the repo's own AST — not a measurement — and
it should be read as one.

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

The last row matters as much as the first: `getLibrary(params)` is a
source-parameterized compile with a library cache keyed on the parameters. That
is the mechanism `ti.template()` specialization has to land on (§3.2), and it
already exists.

### 1.2 The ~24-buffer argument limit — mostly dissolves, because of the arena

`DESIGN_mps_support.md` §1.1 calls this "the blocker", and on the numbers it is:
8 kernels bind more than 24 ndarrays, and they are all the ones that matter.
Measured afresh from the AST:

| kernel | ndarrays | template gates | scalars | body lines |
| --- | --- | --- | --- | --- |
| `sheet_resolve_shade` | 49 | 15 | 7 | 1065 |
| `wavefront_shade` | 38 | 19 | 11 | 1340 |
| `wavefront_traverse` | 34 | 6 | 14 | 184 |
| `raster_shadow_trace` | 30 | 7 | 10 | 300 |
| `wavefront_traverse_events` | 30 | 6 | 14 | 183 |
| `wavefront_shadow` | 29 | 4 | 12 | 173 |
| `path_trace_physical_stbvh` | 27 | 1 | 18 | 292 |
| `path_trace_scene_stbvh` | 25 | 1 | 16 | 224 |

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

That measurement is of Taichi's SPIR-V path. MSL's own 64-bit atomic support is
version- and GPU-family-dependent and has moved in recent Metal releases, so
whether a *hand-written* `atomic_ulong` min/add compiles and runs on the target
family is a separate question with a possibly different answer. **It is not
established either way in this repo**, and it cannot be from a Linux box. §4's
`atomic_u64_add` / `atomic_u64_min` cases are what settle it.

If it turns out MSL cannot do it either, the conclusion is the earlier doc's:
f32 atomics give a non-deterministic mode a floor, and a deterministic mode
needs the accumulation restructured (segmented reduction over sorted keys, which
is order-independent without needing a wide atomic) rather than an atomic.

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
GitHub's `macos-latest` runner, which is a virtualized-GPU macOS instance: sound
for asking whether an operation *works*, not for asking how fast it is. That
caveat is not in `DESIGN_mps_support.md`, and it cuts both ways —

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

---

## 4. What is unmeasured, and how to settle it

Everything in §1 is established from the torch 2.7.1 headers and Python source
in this checkout, and everything in §3 from the AST of this repo's kernel
modules. **Nothing here was run on an Apple GPU** — this checkout is Linux
x86_64 with a CPU-only torch (`mps.is_built()` is False), so the following are
reasoned, not measured:

1. That `lib.kernel(...)` from Python binds an MPS tensor with no copy, at the
   buffer index implied by argument order. (The C++ API does; the Python shim's
   exact surface is unverified.)
2. Whether the Python shim exposes **grid and threadgroup size**, or only the
   inferred 1-D length in the docstring's example. `MetalKernelFunction::dispatch`
   takes both at C++ level, so a thin ObjC++ extension recovers it if the shim
   does not — but which of those two it is changes the plumbing.
3. The **actual** buffer-binding ceiling under this API, against the 31/stage
   assumed here.
4. Whether hand-written MSL **64-bit atomics** compile and run on the runner's
   GPU family (§2.2) — the one fact that decides whether a deterministic
   accumulator is possible at all in a shader.
5. Whether a **numerically identical** result comes back — that an f32 shader
   reproducing one of the small kernels matches the CPU arch's output within the
   suites' 2-value tolerance. Precision, not speed.

All five are **capability** questions, and all five are now asked by
`benchmarks/_mps_capability_probe.py`'s **Q8 section** (`--section msl`), which
runs on the existing `.github/workflows/mps_probe.yaml` macOS arm. Nine cases
plus a buffer ladder, one subprocess each, no Taichi in the path:

| case | settles |
| --- | --- |
| `available` | `compile_shader` compiles and dispatches; dumps the Python shim's surface, which is unknown 2 |
| `zero_copy` | the shader wrote *through* the tensor whose `data_ptr` was taken beforehand — unknown 1 |
| `view_offset` | whether a sliced view binds at its own offset or at storage 0 — decides how §1.2 passes offsets |
| `arena` | the arena convention end to end: one `uchar` buffer + an offset table, in place of 49 bindings |
| `grid` | how the dispatch grid is inferred, measured with an atomic rather than assumed |
| `args_*` | the real binding ceiling, stepped finely around 31 — unknown 3 |
| `atomic_f32`, `atomic_u64_*` | unknown 4, and with it whether a deterministic accumulator can exist in a shader |
| `precision` | unknown 5: f32 sRGB encode against an f64 host reference, reported in u8 channel values against the suites' tolerance of 2, in both of MSL's `pow` flavours (fast-math is on by default, which is exactly what moves a rounded byte) |

Every case guards on an element count passed in a tensor rather than as a bare
scalar, so an unexpected answer to the grid or scalar-marshalling question shows
up as a recorded result instead of an out-of-bounds write.

**What that runner cannot answer is anything with a clock on it.** It is a
virtualized-GPU instance (§3.3), so it establishes that a shader compiles, binds,
dispatches and returns the right bits — which is the whole of 1–5 above, and is
what has to be true before performance is even a question. Launch overhead per
dispatch, and whether the many-small-kernel stages (`sheet_compact_taichi.py` has
14 kernels averaging ~35 lines) want fusing on the way across, are real questions
and they need a physical Mac. Q8 is untimed for that reason, rather than
reporting a number that reads authoritative and is not — and the probe's
existing Q5 timings, its module docstring, its printed verdict and the workflow
header now all carry the same caveat.

## 5. Recommended shape, if it goes ahead

Not one change. In dependency order, each stage independently useful:

1. **Probe `compile_shader`** (§4) — **written; needs one macOS run.** Answers
   the five capability unknowns, not the performance one, which CI cannot reach.
   Dispatch it from `.github/workflows/mps_probe.yaml`, or push this branch as a
   PR (the probe path is in the workflow's `paths` trigger). Read §4's table
   against what comes back before starting stage 2.
2. **Arena-offset calling convention**, on CUDA first. Bind one buffer plus an
   offsets struct. Testable and byte-identical on the current backend, where the
   full pixel-comparison suites exist to prove it — this is the single largest
   piece of the port and it does not need a Mac to land.
3. **A vertical slice**: `tonemap_to_u8` + the three `bloom_*` kernels. 2–3
   ndarrays each, ~130 lines of code across the two modules (207 raw), no BVH,
   no specialization beyond `tonemap_to_u8`'s 3 gates, and a pixel-comparable
   output. This proves dispatch, precision and the
   baseline story end to end at ~2% of the total size.
4. **The specialization layer** (§3.2), designed against the slice's real gates.
5. **The raster count/write kernels** — the four that already fit in 24 buffers.
6. **The megakernels**, largest last, on the convention stage 2 established.

Stages 1–3 are perhaps a week. They answer whether the remaining 90% is
*possible* — dispatch, precision, specialization, baselines — and CI can carry
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
