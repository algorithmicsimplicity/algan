# Taichi re-loads every kernel argument from global memory, at every use

**Status: measured, not implemented.** This is a fork proposal for Taichi 1.7.4,
written up because the measurement that produced it answers a question we asked
about the arena calling convention (`DESIGN_metal_native_port.md` §1.2) and then
turned out to be about something much larger: the shipped renderer pays this on
every kernel, on every backend, today.

Everything below was measured on a GTX 1050 (Pascal, sm_61, 5 SM, 4 GB),
Taichi 1.7.4 / LLVM 15, against a captured `sheet_resolve_shade` launch
(704x396 internal, 155k covered pixels) replayed by
`benchmarks/_arena_view_real_kernel_ab.py`.

## 1. What Taichi emits

A Taichi CUDA kernel takes exactly one parameter, a `RuntimeContext` passed
**by value** in constant memory (`ptxas` reports `352 bytes cmem[0]`). That
struct does not contain the arguments. It contains a *pointer* to an argument
buffer in global memory, and the argument buffer holds, per ndarray,
`{ {i32 shape...}, T* data }`.

Every read of a base pointer or a shape is therefore a load from global memory,
and in the optimized LLVM IR it carries no metadata at all:

```llvm
define void @k_split_..._range_for(%struct.RuntimeContext.6* nocapture readonly
                                   byval(%struct.RuntimeContext.6) align 8 %context)
...
%29 = load float*, float* addrspace(1)* %28, align 8
```

No `!invariant.load`, no `!noalias`, no `!alias.scope`. LLVM cannot prove the
kernel's own stores do not write the argument buffer, so LICM cannot hoist these
out of the loop, and it does not: a four-array probe kernel re-loads all four
base pointers on **every** iteration of its grid-stride loop, and so does the
real thing.

`sheet_resolve_shade`'s `range_for` body carries **1737 `ld.u64` + 1383 `ld.u32`
of pure argument traffic** — about 3100 loads out of 37100 instructions that
exist only to re-derive values that were constant before the kernel launched.

Taichi's own `cache_loop_invariant_global_vars` pass is on by default and does
not help here; Algan additionally runs with `advanced_optimization=False` (the
`pbr_neutral_tonemap` miscompile, see `CLAUDE.md`), which may gate further
CHI-level passes.

## 2. What it costs, and how we know

The arena convention replaces N ndarray parameters with one arena buffer plus an
offset table, so each access needs the table's base pointer *and* its entry
before it can compute an address — a third level in what was already a two-level
dependent-load chain. That is the whole measured penalty.

| arm | ndarray args | registers | instructions | loads | `fma.f32` | device |
| --- | --- | --- | --- | --- | --- | --- |
| shipped | 49 | 161 | 37,100 | 4,922 | 5,473 | 2.79 ms |
| arena, 3 buffers by dtype | 5 | 160 | 40,393 (+8.9%) | 7,152 (**+45%**) | 5,474 | 3.29 ms (**+18%**) |
| arena, layout baked as literals | 5 | 202 | 29,463 (-21%) | 3,238 (-34%) | 4,950 | 3.18 ms (+14%) |
| arena, 7 hottest kept as parameters | 14 | — | — | — | — | 2.82 ms (+1.7..3.0%) |

Instruction counts are static, from the compiled `range_for` entry; registers
and spills from `ptxas -arch=sm_61 -v`. Float work is the control: 5473 vs 5474
`fma.f32` — the two kernels do identical arithmetic. The delta is +1224
`ld.u64`, +1006 `ld.u32`, +816 `add.s32`, which is the table indirection and
nothing else.

Three candidate explanations were tested and eliminated:

* **Coalescing.** The benchmark packed arenas back-to-back, leaving all seven
  slot-indexed arrays at bases that are not multiples of 128 B while the shipped
  arm's separate torch allocations are 512 B aligned. Padding every base to
  128 B (`--align 128`) changes nothing: +17.9% either way.
* **Aliasing / one shared buffer.** Giving each of the seven ray-state arrays
  its own arena (`rsown`) is +19.0% — no better than all of them sharing one
  (+18%). Worth restating, because the earlier `role` and `fine` groupings both
  kept all seven in a single read-write arena, so neither could ever have shown
  an effect here.
* **Register pressure.** 161 registers shipped vs 160 arena, no spills in
  either; both land on 3 blocks/SM = 12 warps/SM (18.8% occupancy).

The third row is a **confounded** measurement and should not be read as "the
tables only cost 4 points". Baking the layout in as literals does 21% less work
but takes 202 registers, which drops the kernel to 2 blocks/SM = 8 warps
(12.5% occupancy) and loses more than it gains.

8.9% more instructions costs 18% of wall time because the added instructions are
dependent loads and this kernel has 12 warps/SM to hide them with.

## 3. What would change in Taichi

Two candidates, either of which removes the cause rather than working around it.
Both live in the LLVM codegen, and neither changes the language or the ABI a
host sees.

### 3a. Mark the argument-buffer loads `!invariant.load`

Where `ArgLoadStmt` and the ndarray base/shape reads are lowered
(`taichi/codegen/llvm/codegen_llvm.cpp`, `create_call`/`ArgLoadStmt` visitors),
attach `!invariant.load` to the emitted `LoadInst`. The argument buffer is
written once by the host before launch and is not writable from the kernel, so
the claim is sound for every backend, and it is exactly the claim LLVM needs:
with it, LICM hoists all ~3100 of these loads into the loop preheader.

Smallest possible change; the one to try first.

### 3b. Pass the argument buffer by value in the `RuntimeContext`

Taichi already passes the `RuntimeContext` `byval` — i.e. in CUDA's parameter
space, which is constant-cached and broadcast to a warp. It just does not put
the arguments there. CUDA allows 4 KB of kernel parameters; 49 ndarrays at
`{ptr, shape[3]}` is about 1.2 KB, so the widest kernel we have fits with room
to spare, and a kernel that did not fit could fall back to the current
indirection.

Then every argument read is `ld.param` — uniform, cached, and hoistable —
instead of a global load. Strictly better than 3a where it applies, and it also
removes the `cvta.to.global` chain. Bigger change: the launcher's argument
marshalling has to write into the parameter block rather than a device buffer,
and the non-CUDA backends need their own equivalent.

### What NOT to do

**Do not mark ndarray pointers `noalias`.** Algan's arrays are all
`ManualMemory` slices of one allocation. Distinct slices do not overlap, but
nothing in the type system says so, and the annotation would be a
wrong-pixels-at-some-future-date bug. The aliasing measurement above says it
would buy roughly nothing anyway.

**Do not hoist offsets into kernel-scope locals** as a workaround on our side.
Tried: Taichi's offload pass turns a local written in the serial prologue into a
*global temporary*, and the parallel loop reloads it from global memory on every
iteration — 14 loads against 6 for the naive inline form, plus an extra serial
task. Confirmed in the emitted PTX.

## 4. What it would buy

For the arena calling convention, the thing that motivated the measurement: the
+18% (all arrays arena-bound) and the +1.7..3.0% (`keep-raystate`, what we
shipped) both collapse. What is left after hoisting is one integer add per
access, which is under a percent.

For the shipped renderer, which is the larger prize and does not depend on the
arena work at all: `sheet_resolve_shade` loses ~3100 loads per loop iteration.
That is 8% of its static instruction count in an occupancy-starved kernel where
loads are the expensive kind, and every other megakernel — `pt_shade` (42
ndarrays), `wavefront_shade` (38), `wavefront_traverse_events` (30) — is built the same
way. This is not a Metal-port concern; it is a CUDA and CPU one.

Neither figure has been measured against a patched Taichi. The mechanism is
established (§1 and §2); the size of the win from fixing it is a projection.

## 5. If someone builds this

Order of work:

1. Build Taichi 1.7.4 from source unpatched, reproduce the numbers in §2 with
   `benchmarks/_arena_view_real_kernel_ab.py --both --policies dtype,keep:raystate`.
   Establishing that a from-source build reproduces the shipped wheel's timings
   is the whole gate — do not skip it.
2. Apply 3a alone. Re-dump PTX (`ti.init(print_kernel_asm=True)` writes
   `taichi_kernel_nvptx_*.ptx` to the CWD) and confirm the argument loads have
   moved out of the loop body before timing anything.
3. Time both arms again. The prediction is that the shipped arm gets faster and
   the arena penalty shrinks toward zero; if only the second happens, 3a is
   still worth having but the §4 claim about the renderer is wrong.
4. Only then consider 3b.

Watch for: `advanced_optimization` is off in Algan and on in Taichi's default,
and a from-source build must be tested the way Algan runs it
(`taichi_init_kwargs()`), not with a bare `ti.init` — see `CLAUDE.md`.
Register counts come from `ptxas -arch=sm_61 -v` (CUDA 11.0 is installed on this
box at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin`); the
occupancy arithmetic that turns them into warps/SM is in §2's third row.
