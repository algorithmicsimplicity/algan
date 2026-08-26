# Algan — Taichi Arch Coexistence: Design Document

Status: **DESIGN ONLY, Phase 0 partly run — currently NO-GO.** Nothing in §5 is
implemented and on today's evidence none of it should be. §8's experiments were
run on 2026-08-26 as far as a GPU-less box allows; §12 records what they
returned and what is still open. The short version:

* **§8.2 came back negative**, and more decisively than the design expected —
  it is readable from source after all. The three criterion kernels cannot
  receive host tensors, so there is no staging tax there to recover. That is
  §10's second kill criterion.
* **§8.1, the blocking experiment, is still unanswered** — it needs a CUDA
  machine. It is now one command:
  `ALGAN_RENDER_DEVICE=cuda uv run python benchmarks/_taichi_arch_coexistence_probe.py`.
* Everything in §4 that does not need CUDA **re-verified** (14/14 checks),
  except its launch-overhead claim, which **did not reproduce**.
* §8.3 came back cheap: a cold build is 4 seconds, of which Taichi compilation
  is 0.19 s.

Read §12 and then §10 before doing anything else here.

§8 is a set of go/no-go experiments that must run before any of the rest is
worth building, and **one of them can invalidate the whole design while another
can triple its value**. On today's kernel inventory this subsystem buys about 5%
of a CUDA render, which is not obviously worth its maintenance surface.

Goal: let Algan run Taichi kernels on the **CPU** inside a process whose Taichi
arch is **CUDA**, so that CPU batch-prep work can be written as kernels without
their arguments being staged through VRAM.

Everything in §2–§4 was verified on taichi 1.7.4 against this repository. Every
verification was **x64-against-x64** on a CPU-only box; the cuda-against-x64
pairing this design actually needs is argued from the mechanism, not measured
(§8.1 is the experiment that closes that gap).

---

## 1. The problem

Taichi's arch is chosen once, by `ti.init`, for the whole process
([`algan/rendering/taichi_runtime.py`](algan/rendering/taichi_runtime.py) picks
it from `_RENDER_DEVICE`). Every Algan kernel takes its arguments as torch
tensors, and Taichi **stages any argument that does not already live on its
arch's device** — `kernel_impl.py` passes `tensor.data_ptr()` straight through,
and the LLVM CUDA backend copies host pointers to device memory and back around
the launch.

Algan's animation and batch-prep tensors are host tensors:
`_ANIMATION_DEVICE` defaults to `cpu` while `_RENDER_DEVICE` defaults to `auto`.
So on the machine that matters, prep data is on the CPU and the arch is CUDA,
and a prep kernel copies **both its inputs and its whole result** across PCIe —
on the batch-prep worker thread that is deliberately kept off the GPU.

This is not hypothetical. It is why the timeline's own kernels were replaced by
torch: `timeline._generate_array_states_taichi` records that launching them with
CPU animation tensors staged the entire `[T, N, D]` result each way, "hundreds
of MB of driver allocation per batch racing the in-flight render", and crashed a
real scene with `CUDA_ERROR_OUT_OF_MEMORY` inside `cuMemAllocAsync`.

---

## 2. What it costs, measured

P13 in [`DESIGN_optimization_targets.md`](DESIGN_optimization_targets.md) landed
one prep kernel — the sides-and-crosses block of `compute_grid_vertex_normals` —
and it is **CPU-arch only** for exactly this reason
(`taichi_runtime.cpu_prep_kernel_enabled`). Its measured value:

| | |
| --- | --- |
| the block alone | **8.4–11.3x** |
| whole `compute_grid_vertex_normals` | **2–5x** (smallest shape is noisy; has read as low as 1.5x) |
| share of `get_render_primitives_batched` | 44.9% is the function, 76.8% of that is the block |
| share of a whole render | the stage is 15.8% of a 358 s render |

Multiplying through: the block is ~5.4% of a render, and the kernel removes most
of it — call it **~5% of total render time**. That is real, and today it is
available only when the render device is the CPU. This design is what would
extend it to CUDA renders.

**Five percent for a whole subsystem is a weak case on its own.** §8.2 is the
experiment that decides whether the real number is much larger.

---

## 3. What does not work

Three mechanisms suggest themselves. All three were tried; all three fail, and
the failures are worth recording because two of them fail *silently enough to
waste a day*.

### 3.1 Two `Program` objects — blocked in C++

`Program` is a hard singleton. Building a second `PyTaichi` runtime by hand and
calling `create_program()` fails immediately:

```
RuntimeError: [program.cpp:Program@141] Only one instance at a time
```

So "two Taichi worlds in one process, swap the module-global runtime" is not a
hack that can be made to work at the Python layer. It is refused underneath it.

### 3.2 Flipping `ti.init` per call — destroys the process's kernels

`ti.init` calls `impl.reset()`, which calls `Kernel.reset()` on **every**
registered kernel — and that method is two lines: rebind the runtime, and
`self.compiled_kernels = {}`. Every kernel Algan has compiled is dropped, so a
flip costs a re-materialize plus an offline-cache reload of all fifteen
`*_taichi.py` modules. `ti.init` itself measured ~170 ms on CPU with a trivial
kernel set, before any of that.

Worse, it is not merely slow. Reading a `ti.field` created before the flip
**segfaults the process** (exit 139) with no Python exception:

```
before re-init: 3.0
re-init done, now reading stale field...
EXIT=139
```

Algan happens to be safe from that specific trap — `algan/` holds no `ti.field`
or `ti.Ndarray` anywhere, only torch tensors passed as `ti.types.ndarray()` —
but the recompile cost alone rules flipping out for anything per-batch.

### 3.3 Building an AOT module for the other arch in-process — refused

`ti.aot.Module(arch)` checks the requested arch against the live one and, if
they differ, **warns and silently uses the current arch instead**:

```python
elif arch != curr_arch:
    warnings.warn(f"AOT compilation to a different arch than the current one "
                  f"is not yet supported, switching to {curr_arch}")
```

So the AOT module for the CPU cannot be built inside the CUDA process. It has to
be built where the arch is already x64, which is what makes §4 a two-process
design with a cache rather than a library call.

---

## 4. What does work: the C API

`libtaichi_c_api.so` is a **separate shared object** from
`taichi_python*.so`, with its own globals. The singleton guard string
`Only one instance at a time` appears once in `taichi_python*.so` and **zero**
times in `libtaichi_c_api.so`, and `ti_create_runtime` takes the arch as an
argument (`TI_ARCH_X64 = 4`, `TI_ARCH_CUDA = 3`).

Verified end to end, in one process:

1. `ti.init(arch=…)` — an ordinary Python Taichi `Program`, live throughout.
2. `ti_create_runtime(TI_ARCH_X64, 0)` — succeeds beside it. So does a *second*
   one; there is no guard.
3. An AOT module built in a separate x64 process loads with
   `ti_load_aot_module`, and `ti_get_aot_module_kernel` resolves a kernel by
   name.
4. `ti_import_cpu_memory(runtime, tensor.data_ptr(), nbytes)` wraps a torch CPU
   tensor **without copying it** — asserted by `data_ptr()` being unchanged
   after the launch.
5. `ti_launch_kernel` + `ti_wait` produce correct results, **interleaved** with
   Python-side kernel launches in the same process, in both orders.

```
runtime / module / kernel: 0x2f1ec0e0 0x2f72cd70 0x2fd67460
after C-API CPU kernel (expect 31.0): [31.0, 31.0, 31.0, 31.0]
tensor was not reallocated: True
after python-side kernel (expect 62.0): [62.0, 62.0, 62.0, 62.0]
after second C-API launch (expect 32.0): [32.0, 32.0, 32.0, 32.0]
interleaved again (expect 64.0): [64.0, 64.0, 64.0, 64.0]
```

Two properties fall out that are better than expected:

* **Zero copies.** Algan never uses `ti.field`; every kernel argument is already
  a torch tensor, which is exactly the case `ti_import_cpu_memory` handles. The
  marshalling story that is usually the hard part of AOT is a pointer and a
  size.
* ~~**Lower launch overhead than the Python path.** 77–89 µs per C-API launch
  (including a fresh `ti_import_cpu_memory` and a blocking `ti_wait`) against
  **173 µs** for an ordinary `@ti.kernel` call, which spends most of its time in
  Python argument validation.~~ **REFUTED — see §12.5.** Re-measured with
  interleaved arms and medians, the two paths are at parity: `ti_launch_kernel`
  alone is 72 µs of the C-API path's 95 µs, so the cost is the C API's own
  dispatch and there is nothing on the Python side left to remove. Do not count
  launch overhead as a benefit of this design.

One environment requirement: `ti_create_runtime` fails with a `runtime_lib_dir`
error unless `TI_LIB_DIR` points at `taichi/_lib/runtime`. The shim must set it.

---

## 5. Design

### 5.1 What is eligible

A kernel may go through this path only if it is:

* **Template-free.** AOT bakes in the argument signature. `ti.template()`
  specialization as the render kernels use it (the ~55 switches on
  `SETTINGS.raytracing.experimental`, `raster_first_shade`'s 16 template params)
  is not available; each distinct value would need its own AOT entry. Today's
  prep kernels already fit — they take plain `ti.types.ndarray()` and scalars.
* **Fixed in dtype and rank.** `ndim` and `dtype` must be declared, not
  inferred. Already true of `grid_normals_sides_crosses`; **not** true of the
  three criterion kernels §8.2 asks about, whose ndarray arguments are all bare
  `ti.types.ndarray()` (§12.5).
* **Worth it.** Per P13, a kernel wins where there are intermediates to fuse and
  loses where there are not. A kernel that does not clear the bar on a CPU-arch
  A/B must not be promoted here; the AOT path makes it faster to run, not
  faster to be worth running.

The render kernels are explicitly **out of scope**. They are template-heavy,
they want the GPU, and nothing here helps them.

### 5.2 Module layout

Keep kernels where they are — the `*_taichi.py` naming convention that the lint
config keys off stays. Add one marker so the build step can find them without
importing the whole package:

```python
# algan/mobs/surfaces/surface_kernels_taichi.py
AOT_KERNELS = ("grid_normals_sides_crosses",)
```

The build step imports each module that declares `AOT_KERNELS` and emits one AOT
module per source module. Nothing else about the modules changes, and they stay
launchable the ordinary way on a CPU arch — which keeps the existing
`cpu_prep_kernel_enabled` dispatch as the A/B arm and the correctness reference.

### 5.3 The build step

A subprocess, because §3.3 forbids doing it in-process:

```
python -m algan.rendering.aot_build --arch x64 --out <cache>/<fingerprint>/
```

with `ALGAN_RENDER_DEVICE=cpu` set so `init_taichi()` selects x64. It imports the
marked modules, calls `ti.aot.Module(ti.x64)`, `add_kernel` for each name, and
`save()`. Cold cost is an ordinary Taichi compile of a handful of small kernels
— seconds, not the minutes the megakernels take.

Run it lazily on first need, not at import: a process that renders on the CPU
arch, or never touches a prep kernel, must not pay for it.

### 5.4 Cache and invalidation

Key the cache directory on a fingerprint of (taichi version, arch, the exact
`taichi_init_kwargs()` dict, and the source of every module contributing
kernels). [`algan/daemon.py`](algan/daemon.py) already takes a content fingerprint of
every Algan source file at startup and re-checks it at every run, so the
mechanism exists; this needs a narrower version of it.

**Stale-cache-on-edit is the failure mode to design against.** It is the same
hazard CLAUDE.md records for the offline kernel cache not invalidating on
`@ti.func` edits, and it will be worse here because the artifact is a directory
built by another process. Include the `@ti.func` bodies each kernel transitively
calls in the fingerprint, or accept that editing a helper silently runs stale
code.

### 5.5 The runtime shim

One module, `algan/rendering/taichi_c_api.py`, holding:

* `ctypes` declarations for `TiNdArray`, `TiArgument` and the union — about 60
  lines of struct definitions.
* A `CpuRuntime` object owning the `TiRuntime`, the loaded module and a
  name → `TiKernel` map, created lazily and once.
* A `launch(name, *tensors, **scalars)` front door that builds the argument
  array, imports each tensor's pointer, launches, and waits.

Two contracts the shim must enforce, because the C API provides neither:

* **Errors are return codes, not exceptions.** `ti_get_last_error` after every
  call, raised as a Python exception. A missed check is a silent wrong answer.
* **Struct layouts are version-locked.** Pin taichi in `pyproject.toml` and add
  a unit test asserting `ctypes.sizeof(TiArgument)`, `sizeof(TiNdArray)` and the
  `TiArgumentType`/`TiDataType` enum values against known constants, so a
  taichi upgrade fails loudly instead of corrupting memory. This is the single
  most dangerous part of the design and it is cheap to guard.

### 5.6 Dispatch

`cpu_prep_kernel_enabled(name)` already exists and already answers "should this
call site use a kernel". Extend it to answer *which* mechanism:

| arch | mechanism |
| --- | --- |
| CPU | the ordinary Python `@ti.kernel` launch (today's path) |
| CUDA | the C-API CPU runtime |
| CUDA, shim unavailable or build failed | torch |

The third row is not optional. Every call site already has a working torch
fallback and must keep it; a failed AOT build is a performance regression, never
a broken render.

### 5.7 Threading and lifetime

Batch prep runs on a `ThreadPoolExecutor(max_workers=1)` worker while the main
thread renders. P13 established that Python-side Taichi launches from that
worker are safe alongside main-thread launches (twelve minutes of full renders,
twice, clean). **That evidence does not transfer to the C API**: it is a
different runtime object in a different shared object. Give the `CpuRuntime` to
the prep worker and serialize on it; do not launch into one `TiRuntime` from two
threads until something measures that it is safe.

`ti_destroy_runtime` at interpreter shutdown, guarded — the daemon keeps
processes alive across renders, so the runtime must survive a render, not be
rebuilt per batch.

---

## 6. Risks

| risk | severity | mitigation |
| --- | --- | --- |
| ctypes struct layout drifts on a taichi upgrade | **memory corruption** | pinned version + `sizeof` assertions (§5.5) |
| stale AOT cache after a `@ti.func` edit | silently wrong output | fingerprint transitive `@ti.func` sources (§5.4) |
| missed `ti_get_last_error` | silently wrong output | every call wrapped, no exceptions (§5.5) |
| concurrent launch into one `TiRuntime` | crash or corruption | single-threaded ownership (§5.7) |
| build subprocess fails in a packaged install | performance regression only | torch fallback is mandatory (§5.6) |
| Windows / macOS paths (`TI_LIB_DIR`, `.dll`/`.dylib`) | build breaks off Linux | Phase 1 exit criterion is all three platforms |

---

## 7. What this does not do

* It does not help the render kernels (§5.1).
* It does not make Taichi faster than torch. P13's loop-shape finding stands: a
  kernel wins where there are intermediates to fuse. This design removes the
  *staging tax* that currently makes CPU-arch-only the rule; it does not change
  which kernels deserve to exist.
* It does not remove `ALGAN_RENDER_DEVICE=cpu` as the simplest way to get these
  kernels. That path stays, and stays the correctness reference.

---

## 8. Phase 0 — go/no-go

None of §5 is worth writing until these run. §8.1 can invalidate the design;
§8.2 can multiply its value by several times.

### 8.1 Does a CUDA Python `Program` coexist with a C-API x64 runtime? (blocking)

> **STILL UNANSWERED — needs a CUDA machine.** Written up as
> `benchmarks/_taichi_arch_coexistence_probe.py`; run
> `ALGAN_RENDER_DEVICE=cuda uv run python benchmarks/_taichi_arch_coexistence_probe.py`
> and read its last three lines. Everything CUDA-independent re-verified (§12.1).

Everything verified in §4 was x64 against x64. The mechanism says arch is a
per-runtime argument and the two shared objects hold separate state, but the
CUDA program initializes a CUDA context and a device memory pool that the x64
runtime knows nothing about, and that pairing is untested.

Run the §4 sequence on a CUDA box with `ti.init(arch=ti.cuda)`: create the C-API
x64 runtime, load a CPU AOT module, launch it on a torch CPU tensor, and
interleave with a CUDA-side `@ti.kernel` launch. **Assert `data_ptr()` is
unchanged and that `nvidia-smi` shows no allocation growth across the CPU
launches** — a staging copy would be invisible in the result but is the entire
point.

If this fails, the design is dead and the fallback is §9.1.

### 8.2 Are the existing PN and bezier criterion kernels already staging? (value)

> **ANSWERED: no.** They cannot be — a second gate independent of
> `pn_criterion_kernel_active()` refuses any non-CUDA tensor. §12.2 has the
> trace; the paragraph below is left as written because its reasoning about
> *why* it looked unanswerable is what turned out to be wrong.

`pn_edge_chord_error`, `pn_patch_flatness_error` and `bezier_chord_hull_error`
run during batch prep and take whatever device their inputs are on
(`device = edge_controls.device`). `_ANIMATION_DEVICE` defaults to `cpu`. If
those inputs are host tensors on a CUDA render, **those three kernels are
already paying the full staging cost on every batch**, and this design would
recover it — which would make the payoff much larger than §2's ~5%.

The answer is **not** readable from the source, which is why it is an
experiment: `render_loop` has a render-device prep budget
(`_render_device_prep_budget`) and a `project_on_gpu` path
(`rt_settings.project_on_gpu_active()`), so some prep tensors can end up on the
render device and some cannot, depending on a VRAM budget. A CPU-only box cannot
distinguish the two, because there both devices are the same.

Cheap to check on a CUDA box: assert `edge_controls.device` at those three call
sites during a real render, **recording which arm of the projection/budget
switch was active**, and if they are host tensors, time the stage with the
kernels forced through their torch fallbacks for comparison.

This is the experiment most likely to change the decision, and it costs an
afternoon.

### 8.3 What does the build step actually cost, cold and warm?

> **ANSWERED: 4.0 s cold, of which Taichi compilation is 0.19 s.** The policy
> stands, but the cost is the subprocess, not the compile — so batch every
> eligible kernel into one build (§12.3).

Time the subprocess build of the currently-eligible kernels from an empty
cache, and the `ti_load_aot_module` from a warm one. If a cold build is minutes
rather than seconds, the lazy-on-first-need policy in §5.3 needs rethinking.

---

## 9. Alternatives considered

### 9.1 A CPU worker process, with torch shared memory

Run the prep kernels in a second process whose `ALGAN_RENDER_DEVICE=cpu`, and
share buffers with `Tensor.share_memory_()` so nothing is copied. Fully
supported, no ctypes, no AOT.

Rejected as the primary design because the buffers are the problem: batch prep
is a *thread* sharing the live Scene and timeline object graph, and it allocates
its `[T, N, D]` results on demand. Making the large ones shared-memory-allocated
up front is more invasive than the shim, and Windows `spawn` re-imports algan in
the child. It remains the fallback if §8.1 fails, scoped to one or two large
leaf computations rather than the whole prep stage.

### 9.2 Do not use Taichi on the CPU — use numba

`@njit(parallel=True, cache=True)` over `tensor.numpy()` (zero-copy for CPU
tensors) gets the same class of win with no arch coexistence at all. The cost is
a second kernel dialect and a new dependency, against a repository that already
has fifteen `*_taichi.py` modules and a documented set of Taichi hazards.

Genuinely competitive, and cheaper than this design. It is the right answer if
§8.2 comes back negative and the inventory stays at one kernel.

### 9.3 Move the prep work to the GPU instead

Not an option: the batch-prep thread exists to overlap with the render and to
keep VRAM free for it. Putting prep on the render device is the thing the
current architecture is arranged to avoid.

---

## 10. When not to build this

Kill criteria, stated up front so they are not negotiated away later:

* **§8.1 fails** — a CUDA program and a C-API x64 runtime cannot coexist. Fall
  back to §9.1.
* **§8.2 comes back negative** and the eligible-kernel inventory is still one
  kernel worth ~5% of a render. Then this is a subsystem — a build subprocess, a
  fingerprinted cache, ~200 lines of ctypes with a memory-corruption failure
  mode — in exchange for 5%, and §9.2 gets the same 5% for a fraction of the
  work. Prefer numba, or prefer doing nothing.
* **The struct-layout guard cannot be made to hold** across the taichi versions
  the project wants to support. A ctypes shim that can silently misread a union
  is worse than a slow render.

The case for building it is: §8.2 is positive, *or* the eligible inventory grows
past two or three kernels each carrying a real win. Option value on future
kernels is a legitimate argument, but it is not sufficient on its own, and it
should not be used to skip §8.

---

## 11. Reproducing §2–§4

The probes behind this document, in the order they answer the questions above:

* `benchmarks/_cpu_prep_kernels_ab.py` — §2, the payoff, on a CPU arch.
* `benchmarks/_grid_normals_kernel_ab.py` — §2, the block in isolation.
* `benchmarks/_taichi_loop_shapes_taichi.py` — the loop-shape caveat in §7.
* §3 and §4 were one-off probes rather than committed scripts; the exact
  sequences are transcribed above (the singleton error, the `Kernel.reset` body,
  the SIGSEGV, the AOT arch assertion, and the C-API round trip) so they can be
  re-run from the text. §4 is now
  `benchmarks/_taichi_arch_coexistence_probe.py`, which is also §8.1 (see §12).

---

## 12. Phase 0 results

Run 2026-08-26. **Machine: a 4-vCPU / 16 GB cloud container with no GPU**,
taichi 1.7.4, torch 2.7.1+cu126 running on CPU. That is the single most
important caveat on everything below: §8.1 and half of §8.2 are CUDA questions
and this box cannot answer them. Where a result is CUDA-independent it is
stated as measured; where it is not, it is stated as unanswered, not inferred.

The scripts:

| script | answers |
| --- | --- |
| `benchmarks/_taichi_c_api_shim.py` | the ctypes shim §5.5 describes |
| `benchmarks/_taichi_c_api_layout_check.py` | §5.5's version lock / §10's third kill criterion |
| `benchmarks/_taichi_aot_build.py` | §5.3's build step |
| `benchmarks/_taichi_arch_coexistence_probe.py` | §4 and §8.1 |
| `benchmarks/_taichi_aot_build_cost.py` | §8.3 |

### 12.1 §8.1 — UNANSWERED, and it is still blocking

No CUDA device here, so the cuda-against-x64 pairing remains untested and the
design's central mechanism remains argued rather than measured. Everything in
§4's sequence that does not require CUDA re-verified x64-against-x64 — 14 of 14
blocking checks:

* `ti_create_runtime(TI_ARCH_X64, 0)` succeeds beside a live Python `Program`,
  and so does a second one: there is no singleton guard in
  `libtaichi_c_api.so`, as §4 says.
* An AOT module built in a separate x64 process loads with
  `ti_load_aot_module`, and `ti_get_aot_module_kernel` resolves by name.
* `ti_import_cpu_memory` does not copy: `data_ptr()` is unchanged across the
  launch, for both the input and the output tensor.
* Results are correct against an independently-written torch oracle.
* C-API and Python-side launches interleave in both orders, repeatedly.
* A missing kernel name and a non-CPU tensor both raise rather than corrupting
  anything; `ti_destroy_runtime` is safe, and safe twice.

**To answer §8.1, run this on a CUDA box and read the last three lines:**

```
ALGAN_RENDER_DEVICE=cuda uv run python benchmarks/_taichi_arch_coexistence_probe.py
```

It detects the live arch itself and switches to §8.1's assertions. Two things
it does that §8.1 does not ask for, and that the experiment needs:

* **A positive control.** §8.1 asks for "no allocation growth across the CPU
  launches", which is also what a measurement that cannot see allocations
  reports. So the probe first stages a launch deliberately — the live CUDA
  program against host tensors, exactly the tax §1 describes — records the VRAM
  it moves, and *requires* that to exceed the C-API launch's movement by a
  margin. A run whose control shows nothing has proved nothing, and the probe
  says so rather than passing.
* **Driver-level accounting.** It reads `torch.cuda.mem_get_info`, not
  `torch.cuda.memory_allocated`: Taichi stages through its own allocator, which
  torch's accounting cannot see.

### 12.2 §8.2 — NEGATIVE, and readable from source after all

§8.2 says "the answer is **not** readable from the source". It is. There are
two independent gates, and the second is a plain device check:

1. `pn_criterion_kernel_active()` is `PN_CRITERION_KERNEL and
   project_on_gpu_active()`, and `project_on_gpu_active()` requires
   `_RENDER_DEVICE.type == "cuda"`.
2. Independently, `_pn_criterion_inputs` and `_bezier_criterion_inputs`
   (`algan/rendering/raytracing/primitives.py`) return `None` — the torch
   fallback — unless **every** tensor argument has `device.type == "cuda"` and
   dtype float32.

So `pn_edge_chord_error`, `pn_patch_flatness_error` and
`bezier_chord_hull_error` **cannot receive a host tensor**, on any render, under
any budget or retry arm. They are not staging today, and the design would
recover nothing from them. The full trace — every argument of all three kernels
to its allocation site, plus the OOM retry, the window shrink, the prefetch
worker split, `_slice_fetched_batch` and the unmanaged-memory early return — is
in `OX_STAGING_AUDIT.md`.

**This fires §10's second kill criterion.** §8.2 was the experiment most likely
to change the decision, and it changed it toward no.

Three things fell out of that audit that matter more than the answer:

* **The inventory in §5.1 is incomplete.** Outside the render kernels there are
  not one but three groups: the `cpu_prep_kernel_enabled` kernels (which
  `taichi_arch_is_cpu()` prevents from ever launching on a CUDA arch, so they
  cannot stage), the **timeline query kernels**
  (`_query_state_from_edits` / `_query_selected_state_from_edits`, behind
  `ALGAN_OPT_DISABLE=torchquery`), and the post-processing kernels (bloom and
  tonemap, gated on the tensor's device matching the arch).
* **The timeline query kernels are the design's real candidate.** They are the
  case §1 cites as the motivating example, they *would* stage every argument
  and the whole `[T, N, D]` result, and they were replaced by torch for exactly
  that reason. Whether they are worth recovering is a speed question that a
  CPU-arch A/B can answer without a GPU, because on a CPU arch there is no
  staging — see `OX_TIMELINE_QUERY_AB.md`.
* **`slack` is the one unchecked tensor argument** of `_pn_criterion_inputs`:
  it is on the render device by derivation from `source_corners`, not because
  anything checks it. No current path puts it elsewhere. Adding it to the
  checked tuple is a one-line hardening and is not part of this design.

### 12.3 §8.3 — cheap, and §5.3's lazy policy stands

`uv run python benchmarks/_taichi_aot_build_cost.py`:

| arm | subprocess wall | Taichi compile | `import algan` etc. |
| --- | --- | --- | --- |
| cold (empty offline cache) | 3.96 s | 0.19 s | 3.77 s |
| warm (populated cache) | 3.94 s | 0.12 s | 3.82 s |
| warm again | 3.60 s | 0.11 s | 3.48 s |

Artifact: 41.5 KiB for one kernel. Loading it back:
`ti_load_aot_module` + resolve is **21 ms**, and `ti_create_runtime(x64)` is
**147 ms** once per process.

Seconds, not minutes — so §5.3's lazy-on-first-need policy survives, but not
for the reason §8.3 anticipated. **The cost is not compilation, it is the
subprocess.** Taichi compiles this kernel in 0.19 s cold; the other 3.8 s is a
Python interpreter starting and importing algan, and that is a fixed toll paid
per build regardless of how many kernels the build contains. The marginal cost
of the *second* eligible kernel is ~0.1 s. If this is ever built, batch every
eligible kernel into one build; never build them one at a time.

### 12.4 §10's third kill criterion — the layout guard HOLDS

`uv run python benchmarks/_taichi_c_api_layout_check.py` passes: 11 structs and
13 enum values, every `sizeof`, `alignof` and `offsetof` agreeing.

It is built the strong way and that is the point. Rather than asserting
hand-copied constants — which agree with a wrong transcription just as happily
as with a right one — it compiles a C program against the **installed**
`taichi_core.h` and compares the compiler's own answers against what ctypes
computes for the shim's classes. `TiArgument` is 160 bytes with its union at
offset 8; a taichi upgrade that moves any of it fails this check instead of
corrupting memory at a launch. So §10's third criterion is not what kills this
design.

### 12.5 Two claims in §4 and §5.1 that did not survive

**§4's launch-overhead advantage does not reproduce.** §4 reports 77–89 µs per
C-API launch against 173 µs for an ordinary `@ti.kernel` call and lists "lower
launch overhead than the Python path" as one of two properties better than
expected. Measured here with interleaved arms and medians over nine rounds of
500 launches each:

| path | median | range |
| --- | --- | --- |
| C-API, whole shim `launch()` | 95 µs | 79–131 |
| of which building two `TiArgument`s | 10 µs | |
| `ti_launch_kernel` + `ti_wait` alone | 72 µs | 68–83 |
| Python `@ti.kernel` on x64 | 87–90 µs | 64–108 |

That is **parity, not a 2x advantage** — the ranges overlap heavily, and across
runs the ratio wandered from 0.75x to 1.27x. The breakdown says why it cannot
be much better: `ti_launch_kernel` alone is 72 of the 95 µs, so the cost is the
C API's own dispatch, not the shim's Python, and there is nothing on the Python
side left to remove. A single timed block per arm is what produces §4's kind of
number on a shared vCPU; the probe now interleaves and takes medians for
exactly this reason, and reports the claim as REFUTED without failing the run.

**§5.1's eligibility claim is wrong about the criterion kernels.** It says
kernels must be "fixed in dtype and rank — `ndim` and `dtype` must be declared,
not inferred. Again already true of the prep kernels." True of
`grid_normals_sides_crosses`, which declares
`ti.types.ndarray(dtype=ti.f32, ndim=4)`. **Not true** of
`pn_edge_chord_error`, `pn_patch_flatness_error` or `bezier_chord_hull_error`,
every one of whose ndarray arguments is a bare `ti.types.ndarray()` with
neither declared. Making them AOT-eligible would mean annotating nine to
fourteen arguments per kernel first. Moot while §8.2 is negative, but it is a
cost the design does not currently carry.

### 12.6 One thing that came back positive: AOT keeps the win

Not in §8, and it should have been — a build that is cheap and a kernel that
lost its speedup would be worthless. `grid_normals_sides_crosses` through the
AOT/C-API path against the same kernel launched the ordinary way, both against
torch, on the shapes the batched surface build passes:

| shape | torch | Python `@ti.kernel` | C-API AOT |
| --- | --- | --- | --- |
| `(4, 64, 64, 3)` | 2.35 ms | 0.43 ms (5.40x) | 0.45 ms (5.17x) |
| `(4, 256, 256, 3)` | 26.69 ms | 4.27 ms (6.25x) | 4.13 ms (6.46x) |
| `(4, 512, 1024, 3)` | 266.81 ms | 27.99 ms (9.53x) | 21.43 ms (12.45x) |

The two kernel columns track each other. The absolute multipliers move run to
run on this box — the same three shapes read 3.2x/13.1x/10.6x on an earlier
pass — so read the *comparison*, not the numbers: the AOT path does not cost
part of what it exists to deliver. If §8.1 passes, the payoff is §2's number
rather than some fraction of it.

### 12.7 Where the decision stands

Against §10's kill criteria:

| criterion | status |
| --- | --- |
| §8.1 fails | **unknown** — needs a CUDA box; one command, ~1 minute |
| §8.2 negative **and** inventory still one kernel | **first half fired.** The second half is open: the timeline query kernels are a real second candidate |
| layout guard cannot hold | **does not fire** — it holds, mechanically |

So the design is not dead, but its stated reason for existing is weaker than
when it was written: the payoff §8.2 hoped to find is not there. What is left
is §2's ~5%, against a subsystem with a memory-corruption failure mode, plus
whatever the timeline query kernels turn out to be worth. §10 already says what
to do with 5% and one kernel: prefer §9.2's numba, or prefer doing nothing.

**Recommended order for whoever picks this up:**

1. Read `OX_TIMELINE_QUERY_AB.md`. If the timeline query kernel loses to torch
   on a CPU arch, where nothing stages, then the design's own motivating
   example is not worth recovering and the inventory really is one kernel —
   take §10 at its word and stop.
2. Only if that comes back positive, run §8.1 on a CUDA box.
3. Do not start §5 before both.

And one thing worth doing whatever happens to this design, because it needs
none of it: the criterion kernels **run correctly on an x64 arch** (verified —
0.26 s cold compile, 0.36 ms warm launch), and they are excluded from CPU
renders by a gate whose stated reason is staging, which cannot occur when the
arch is the CPU. Whether that leaves a real win on the table on CPU renders is
a measurement nobody has taken.
