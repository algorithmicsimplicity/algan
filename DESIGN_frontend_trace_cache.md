# Algan — Frontend Trace Cache: Design Document

Status: DESIGN ONLY. Nothing here is implemented. Phase 0 (§9) is a set of
go/no-go experiments that must pass before any of the rest is worth building;
two of them can invalidate the whole approach.

Goal: remove the per-process Python cost of Taichi kernel *materialization* —
the AST walk that builds the frontend IR — on every run after the first, so a
cold Algan process reaches its first launch in ~5s of kernel prep instead of
~13s. This is the third patch in the same family as
[`algan/utils/taichi_warmstart.py`](algan/utils/taichi_warmstart.py) (memoize inside the AST transform)
and [`algan/utils/taichi_fast_launch.py`](algan/utils/taichi_fast_launch.py) (skip re-validation per
launch); it is the one that skips the transform altogether.

It does **not** speed up the daemon path ([`algan/daemon.py`](algan/daemon.py)),
which already avoids this cost entirely by keeping the process warm. Read §11
before building this: the daemon is a better answer for interactive work, and
this design only pays for itself on cold processes — CI, one-shot scripts,
`pytest`, and the first render of a session.

---

## 1. The cost being attacked

Measured on this machine (CUDA, taichi 1.7.4, warm offline cache, an otherwise
idle box), `Square().spawn(); Scene.save_frame(...)`, with
`ALGAN_LOG_TAICHI_COMPILES=1`:

```
save_frame                        27.4s
  6 kernel specializations:  frontend 12.9s   backend 7.0s
  raster_first_shade alone:  frontend 11.3s   backend 6.5s
```

* **frontend** = `Kernel.materialize` — `_get_tree_and_ctx` (source retrieval +
  `ast.parse`) plus `transform_tree`, the Python walk that emits frontend IR
  through pybind. This is what the design removes.
* **backend** = `Program.compile_kernel` — on a warm cache, the `.tic` load plus
  LLVM lowering. Untouched by this design.

One kernel is 88% of that frontend: `raster_first_shade` (16 `ti.template()`
params, 46 ndarray args). `wavefront_shade` (15/47) is the same size and carries
the same cost on scenes that route to it. Cost scales with the *inlined* body
size, because non-real `ti.func`s are re-transformed at every call site — the
six specializations here are one instantiation each, so this is not
variant-count cost, it is one walk of a very large body.

Under `cProfile`, `raster_first_shade`'s materialize is 18.0s profiled / 11.3s
unprofiled, of which **4.0s is `tottime` inside taichi pybind calls**. That 4.0s
is the irreducible floor for any Python-side replay: it is C++ work (IR node
construction, `type_check`) that a replay still has to request, one pybind call
at a time. So the realistic target for the biggest kernel is ~11.3s → ~5s, and
for a whole cold `save_frame`, ~27.4s → ~19-20s.

**If that ratio is not worth the machinery to you, stop here.** The rest of this
document assumes it is, for cold processes.

---

## 2. Why the obvious design (hand Taichi a precomputed key) is impossible

The original idea was: compute a cheap source-level key, look up the Taichi
offline-cache key it produced last time, and jump straight to
`Program.compile_kernel` without building IR. Every step of that is blocked by
the shipped pybind surface, verified against
`.venv/Lib/site-packages/taichi` (1.7.4):

| Requirement | Reality |
|---|---|
| Set the cache key on a kernel | `_ti_core.Kernel` exposes only `ast_builder`, `finalize_params`, `finalize_rets`, `insert_*_param`, `insert_ret`, `make_launch_context`, `no_activate`, `pop_argpack_stack`. No key setter. |
| Compile from a key alone | `Program.compile_kernel(config, caps, kernel_cxx)` takes the C++ kernel and derives the key internally from the frontend IR. |
| Load a compiled artifact directly | `_ti_core.CompiledKernelData` is opaque — its only Python attribute is `_pybind11_conduit_v1_`. No load/save. |
| Load an AOT module in-process | `taichi/aot/` is export-only. There is no Python loader; loading is C-API/C++ only (and see `DESIGN`-level notes in the AOT assessment: 2^16 template variants, runtime-composed `ti.func` templates, bare `ndarray()` annotations). |

Confirmed by experiment that the key *is* a hash of the frontend IR, not of the
kernel's own source text:

* A `@ti.func` inlined into a kernel, differing only by a Python global baked in
  at transform time (`x + 1` vs `x + 7`), produces **two distinct `.tic` files**
  and correct results from both — the key covers inlined bodies and captured
  constants. (Probe: one `@ti.func` reading a module-global `DELTA`, one kernel
  calling it, run twice against an isolated `offline_cache_file_path` on
  `arch=ti.cpu`; count the `.tic` files and check the returned value.)
* Shifting the source file (path/line numbers change, semantics identical) also
  produces a new `.tic`: **debug info participates in the key**.
* Defining one extra `@ti.kernel` *earlier in the file* — which bumps
  `Kernel.kernel_counter` and hence the C++ kernel name
  `f"{func.__name__}_c{counter}_{instance}"` — also produces a new `.tic`: **the
  generated kernel name participates in the key**. (Independently useful: this
  means Algan's kernel cache is already sensitive to kernel *definition order*.)

**Conclusion.** We cannot tell Taichi the key. We can only make Taichi compute
the same key, which means reproducing the frontend IR exactly. The only lever
Python has on that IR is the sequence of pybind calls the transform makes. So
the design is: **record that call sequence once, replay it later.**

The compensating advantage: we never need to know or store the Taichi key at
all. If the replay is faithful, the key follows for free.

---

## 3. Architecture

```
Kernel.materialize(key, args, arg_features)
  │
  ├─ source-level key K = fingerprint(§4)          ~1ms
  │
  ├─ trace store hit for K?  ──── no ──►  original path (transform_tree)
  │                                        │
  │                                        └─ RECORDER attached: capture the
  │                                           pybind call stream + captured
  │                                           environment, write trace[K]
  │
  └─ yes ─► validate captured environment (§4.5)
             │  mismatch ──► original path (and re-record)
             │
             └─ prog.create_kernel(replay_cb, kernel_name, autodiff_mode)
                  replay_cb re-issues the recorded pybind calls in order
                  ⇒ byte-identical frontend IR
                  ⇒ Taichi computes the same offline-cache key
                  ⇒ compile_kernel hits the existing .tic
```

Nothing downstream changes. `compiled_kernels[key]` is filled the same way,
`launch_kernel` is unaffected, and `taichi_fast_launch` sits downstream and
keeps working unmodified.

**Fail-safety is asymmetric and this is the central risk.** A *missing* trace
costs nothing (fall back). A *wrong* trace silently builds a different kernel:
it will not crash, it will compile and run the wrong code. Every mechanism in
§4 and §8 exists to make "wrong trace" unreachable, and the verification story
has to be taken as seriously as the perf story.

---

## 4. The source-level key

`K` is a SHA-256 over the canonical serialization of the following. All of it
must be computable **without** running the transform.

### 4.1 Toolchain and configuration
* `taichi.__version__` and `_ti_core.get_commit_hash()`.
* Arch name (`cuda` / `x64`), and the exact `taichi_init_kwargs()` dict from
  [`rendering/taichi_runtime.py`](algan/rendering/taichi_runtime.py) — it is the
  single point where Algan configures the compiler, so hashing it covers
  `default_fp`, `debug`, `fast_math`, `advanced_optimization`, etc.
* `sys.version_info[:3]` (the transform is Python-version-sensitive through
  `ast`).

### 4.2 Kernel identity
* `func.__module__`, `func.__qualname__`.
* **The generated C++ kernel name** `f"{func.__name__}_c{kernel_counter}_{instance}"`,
  because §2 shows it lands in the Taichi key. This makes `K` depend on kernel
  definition order — correct, and it makes trace hits and `.tic` hits invalidate
  together instead of drifting apart.
* `autodiff_mode`.

### 4.3 Source fingerprint
The transform inlines every non-real `ti.func` it reaches, so the fingerprint
must cover all of them. Reachability is only known *after* a transform, so use a
deliberately coarse, sound over-approximation:

> SHA-256 over the sorted list of `(path, size, sha256(bytes))` for every
> source file that defines any `@ti.kernel` or `@ti.func` currently importable
> in the process.

Collect the file set by scanning `sys.modules` at first use for objects
carrying taichi's kernel/func wrapper markers (`_primal` / `_is_taichi_function`
/ `_is_wrapped_kernel_or_func`) and taking `__code__.co_filename`; add the
absolute paths themselves to the hash, since debug info embeds them (§2). In
this repo that is the nine `*_taichi.py` files plus
`shaders/fragment_shaders.py` and `shaders/fragment_stage_library.py`, plus
whatever a user script defines.

Consequence, and it is the right trade: **editing any kernel source invalidates
every trace.** That is the same blast radius as a `.tic` invalidation, and edits
are rare compared to runs.

### 4.4 Instantiation features
The transform bakes template values and ndarray features into the IR (an
ndarray's `ndim` decides subscript arity; a `ti.static(bool)` template deletes
whole branches). `K` must therefore include exactly what Taichi's own
`TaichiCallableTemplateMapper.extract` keys by, which
[`algan/utils/taichi_fast_launch.py`](algan/utils/taichi_fast_launch.py) already reimplements — reuse
that key builder verbatim:

* per template arg: `int`/`bool`/`float`/`str`/`None` by value; **functions by
  `(module, qualname, sha256(source))` rather than identity** — identity is not
  stable across processes, and Algan injects runtime-composed `ti.func`s
  (`build_frag_pipelines()`) as flat tuples; tuples element-wise.
* per ndarray arg: `(element dtype, ndim, needs_grad, element shape)`.
* scalar args excluded (they are runtime values, not compile-time).

### 4.5 Captured environment
The transform also reads arbitrary Python state that is *not* an argument:
module globals, closure cells, `ti.static(...)` of a Python expression. §2's
`DELTA` experiment shows these are baked into the IR and the key.

The recorder captures this by wrapping the transformer's name resolution
(`ASTTransformerContext.get_var_by_name` and the `global_vars` lookup in
`kernel_impl._get_global_vars`) and logging every `(name, value)` it resolved to
a non-taichi Python object. The trace stores `[(name, sha256(repr-or-value))]`.

On a warm run, after `K` hits, re-resolve those *same names* in the live
globals/closure and compare hashes. Mismatch → fall back and re-record. Values
that cannot be hashed cheaply and deterministically (objects without a stable
`repr`, anything whose `repr` contains an address) **poison the trace**: mark it
non-cacheable at record time rather than guess.

This is the part most likely to be gotten subtly wrong, which is why §8's
verification is mandatory rather than optional.

---

## 5. The trace format

### 5.1 What is on the wire
Measured for `raster_first_shade` — by wrapping every callable in
`taichi._lib.core.taichi_python` plus every `ASTBuilder`/`Expr`/`ExprGroup`/
`Kernel` method with a counter and rendering one frame; identical totals in
every run observed — **317,097 pybind calls across 74 distinct entry points**,
plus DebugInfo/ExprGroup constructions (~117k DebugInfo objects appear as
arguments). Positional arity is 1-2 for 88% of calls (max 6). Argument types
observed, in frequency order: `Expr` (274k), `DebugInfo` (117k),
`CompileConfig` (84k), `Kernel` (43k), `ASTBuilder` (41k), `ExprGroup` (23k),
`DataType` (19k), `int`, `float`, `list`, `bool`, `numpy.bool`, `str`,
`BoundaryMode`.

Top entry points (full list in the probe output; the recorder must cover the
whole reachable surface, not just these):

```
83,650 Expr.type_check(Expr, CompileConfig)     10,876 make_const_expr_int(DataType, int)
79,145 Expr.set_dbg_info(Expr, DebugInfo)        8,878 ASTBuilder.expr_subscript(...)
42,893 Kernel.ast_builder(Kernel)                4,888 expr_mul(Expr, Expr)
23,068 ASTBuilder.expr_var(AB, Expr, DebugInfo)  3,879 ASTBuilder.expr_assign(...)
13,922 ExprGroup.push_back(ExprGroup, Expr)      3,657 make_const_expr_fp(DataType, float)
12,185 Expr.is_tensor(Expr)                        924 ASTBuilder.begin_frontend_if(...)
```

### 5.2 Encoding
A trace is a flat op stream. Every pybind object produced during the transform
gets a sequential **handle id**; because replay reissues the identical calls in
identical order, ids line up by construction — no name mangling, no identity
map to persist.

```
op := (opcode: u16, argc: u8, args: argc × u32)
arg := tagged u32:  top 3 bits = tag, low 29 bits = payload
  tag 0  HANDLE     payload = handle id
  tag 1  INT_SMALL  payload = zig-zag int fitting 29 bits
  tag 2  CONST      payload = index into the constant pool (i64/f64/str/bytes)
  tag 3  DTYPE      payload = index into the dtype table
  tag 4  DBGINFO    payload = index into the src-info string table
  tag 5  EXTERN     payload = index into the extern table (§5.3)
  tag 6  LIST       payload = index into the list pool (each entry = u32 args)
  tag 7  ENUM       payload = (enum-type id << 16) | value      e.g. BoundaryMode
```

Files (all little-endian, one directory per trace, or one zip):

```
header      magic, format version, taichi version, K, op count, handle count
opcodes     u16[n_ops]                     (opcode table is content, not code:
argidx      u32[n_ops+1]  prefix offsets    an unknown opcode name = miss)
argdata     u32[total_args]
consts      length-prefixed blobs
dtypes      canonical dtype names ("f32", "i32", vector/matrix specs)
srcinfo     deduplicated src-info strings
externs     extern descriptors (§5.3)
meta.json   K, captured environment, post-transform state (§6), stats
```

Size estimate: 317k ops × (2 + 1 + ~2.2×4) bytes ≈ 4MB raw, ~1MB with
`zlib.compress(level=1)`. Decompression must not eat the win — measure it in
Phase 0; if it does, store raw and rely on the OS page cache.

Store at `~/.algan/cache/frontend/<K[:2]>/<K>.trace` (the unified cache dir —
see `settings/_startup.py`; add a `frontend` bucket to
`clear_cache(...)`). LRU-evict by mtime against a size cap, same as the Taichi
cache's 1GB.

### 5.3 Externs
Not every argument is a handle produced inside the trace. The extern table
resolves the few that come from outside, by descriptor rather than by value:

| Extern | Descriptor | Resolved at replay by |
|---|---|---|
| the C++ `Kernel` | — | `kernel_cxx` passed to the create-kernel callback |
| its `ASTBuilder` | — | `kernel_cxx.ast_builder()`, hoisted once |
| `CompileConfig` | — | `impl.current_cfg()`, hoisted once |
| `DataType` | canonical name | `_ti_core` dtype lookup, built once per trace |
| enum values | `(type, int)` | reconstructed by value |

The 42,893 `Kernel.ast_builder` and ~84k `config` calls collapse to one each on
replay — a free ~1.0s of the 11.3s, and one of the reasons the replay floor is
below the measured 4.0s of C time.

### 5.4 DebugInfo interning
`_ti_core.DebugInfo(src_info_string)` is constructed per-expression (~117k
times) from a small set of distinct strings. Replay should build one DebugInfo
per distinct string and reuse it. **This is only legal if `set_dbg_info` copies
rather than aliases** — Phase 0 tests it with the oracle in §8.1. If interning
changes the key, drop it and construct per-op.

---

## 6. Recorder

Hook `Kernel.materialize` the same way `taichi_warmstart.apply()` hooks its
targets: monkeypatch from outside taichi, version-gated to 1.7.x, silent no-op
otherwise.

Recording is active **only inside `taichi_ast_generator`**, i.e. between
`create_kernel` entering the callback and leaving it. Instrumentation points,
all verified patchable in this build (297 attributes across the module and the
`ASTBuilder`/`Expr`/`ExprGroup`/`Kernel` pybind classes accept `setattr`):

1. every non-type callable in `taichi._lib.core.taichi_python`;
2. every method of `ASTBuilder`, `Expr`, `ExprGroup`, `Kernel`;
3. the constructors `_ti_core.DebugInfo` and `_ti_core.ExprGroup` — replaced
   with factory functions, since taichi's Python layer only ever *calls* them
   (`grep` confirms: 14 DebugInfo sites in `ops.py`, 14 in `ast_transformer.py`,
   …; one `ExprGroup()` site in `expr.py`; no direct `_ti_core.Expr(...)`
   construction in `taichi/lang`). Check no `isinstance(x, _ti_core.DebugInfo)`
   call site exists before replacing the attribute.

A wrapper records `(opcode, encoded args)`, calls through, and if the return
value is a pybind object assigns it the next handle id. Unknown callable, or an
argument type not in §5.2's tag set → **abort recording** for that kernel (leave
no trace file) and let the original path finish normally. Never raise.

Also record, at the end of the transform, the state the transform mutated that
later stages read:

* `ctx.func.has_print` (set by `build_Call`; governs the post-launch sync in
  `launch_kernel`).
* `ctx.returned` — only used to raise on a missing `return`; validated at record
  time, not replayed.
* nothing else: `self.arguments`, `template_slot_locations` and the mapper are
  built at decoration time, before materialize.

Recording overhead is ~5-10% of a transform (measured: 10.4s counted vs ~11.3s
uncounted, on a quiet box) and it only ever runs on a cold key, so it does not
need to be fast.

---

## 7. Replayer

On a hit, `materialize` skips `_get_tree_and_ctx` and `transform_tree`
completely (that also drops the `builtins.compile`/`getsourcelines` cost —
~1.0s of the 11.3s) and runs:

```python
def replay_cb(kernel_cxx):
    self.kernel_cpp = kernel_cxx
    rt = self.runtime
    rt.inside_kernel, rt.current_kernel, rt.compiling_callable = True, self, kernel_cxx
    try:
        h = [None] * trace.handle_count  # handle table, preallocated
        ext = resolve_externs(kernel_cxx)  # kernel, ast_builder, config, dtypes
        dbg = [DebugInfo(s) for s in trace.srcinfo]  # §5.4, if interning is legal
        ops, argidx, argdata, fns = (
            trace.ops,
            trace.argidx,
            trace.argdata,
            trace.fn_table,
        )
        hi = 0
        for i in range(len(ops)):
            a = decode_args(argdata, argidx[i], argidx[i + 1], h, ext, dbg, trace)
            r = fns[ops[i]](*a)
            if r is not None:
                h[hi] = r
                hi += 1
    finally:
        rt.inside_kernel = False
        rt.current_kernel = None
        rt.compiling_callable = None


taichi_kernel = impl.get_runtime().prog.create_kernel(
    replay_cb, kernel_name, self.autodiff_mode
)
self.compiled_kernels[key] = taichi_kernel
self.has_print = trace.meta["has_print"]
```

`fns` is resolved once per process from the opcode table to the real pybind
callables (**the unpatched originals** — the replayer must not run through the
recorder's wrappers).

Performance requirements for this loop, which is where the design lives or
dies: no per-op Python object allocation beyond the argument tuple, no dict
lookups in the hot path (opcode → callable is a list index), args decoded from
preallocated arrays. Budget: ≤1.5s of Python for 317k ops on top of the ~3.5s
of unavoidable C time.

Handle-id note: only calls that return a pybind object consume a handle id, and
which opcodes do is a static property of the opcode, so the recorder and
replayer agree without storing a per-op flag.

---

## 8. Correctness

### 8.1 The oracle: `.tic` filenames are the key
Taichi's offline cache is `~/.algan/cache/taichi/T<64 hex>.tic` plus a
`ticache.tcb` index, and the experiments in §2 show the hex *is* the key: any IR
difference produces a new file. That gives a cheap, exact equality test for
"did the replay build the same IR?":

```
snapshot the cache dir → run with replay → assert no new .tic appeared
```

Every Phase 0 question (interning, dbg-info handling, whether the recorder
itself perturbs the IR) is answerable with this oracle, on the CPU arch, in
seconds.

### 8.2 Verify mode
`ALGAN_TAICHI_TRACE_VERIFY=1`: on every trace hit, run the *real* transform as
well, with the recorder attached, and compare the freshly recorded op stream to
the stored one op-for-op; raise on the first divergence with the op index,
opcode names and decoded args. This mirrors
`ALGAN_TAICHI_FAST_LAUNCH_VERIFY=1` and is what the check script runs.

Note the ordering constraint: the verification transform must run into a
*throwaway* C++ kernel, or the process ends up with two kernels where one is
expected.

### 8.3 Tests
* `benchmarks/_frontend_trace_check.py`, following the `_*_check.py` convention:
  cold record → warm replay → assert (a) no new `.tic`, (b) verify mode clean,
  (c) frontend time drops by the expected factor, (d) rendered output is
  byte-identical to the non-traced run.
* One unit test that a source edit invalidates (`K` changes), one that a
  template-value change picks a different trace, one that a captured-global
  change falls back.
* The fast suite renders once with the cache cold and once warm; both must match
  the existing baseline. Mark the cold one `slow` if it costs more than a few
  seconds.

### 8.4 Kill switch and blast radius
`ALGAN_TAICHI_FRONTEND_TRACE=0` disables record and replay. The patch is applied
at [`algan/__init__.py:77`](../__init__.py:77), beside `_apply_taichi_warmstart()`
and `_apply_taichi_fast_launch()`, and both env vars join the registry in
[`environment.py:94`](../environment.py:94) next to `ALGAN_TAICHI_WARMSTART` /
`ALGAN_TAICHI_WARMSTART_VERIFY`. It is a silent no-op on any taichi other than
1.7.x, on any unknown opcode, on any hash mismatch, and on any exception in the
store. Deleting `~/.algan/cache/frontend/` is always a valid recovery.

---

## 9. Phase 0 — go/no-go experiments

Do these before writing anything else. Two of them can kill the design.

1. **Transform determinism (kills it).** Record the same kernel twice in one
   process and in two processes; op streams must be identical. Watch
   specifically for `ASTBuilder.make_id_expr(str)` and any string carrying an
   `id()` or a temp counter, and for dict/set iteration order leaking into the
   IR. *Evidence so far: the call count is exactly 317,097 in every run
   observed, which is necessary but not sufficient.*
2. **Replay dispatch cost (kills it).** Before building the store, synthesize
   the op stream in memory and time a bare replay loop over it. If the loop
   cannot beat ~6s for `raster_first_shade` (vs 11.3s), the win is not worth the
   machinery.
3. **Recorder transparency.** With the recorder attached but replay disabled, a
   render must produce no new `.tic` (§8.1). *Note: instrumented runs during the
   investigation did appear to recompile; that was almost certainly a concurrent
   benchmark process sharing the cache dir, but it is unproven and this
   experiment is how it gets settled.*
4. **DebugInfo interning legality** (§5.4), via the same oracle.
5. **Whole-surface coverage.** Record every kernel in the full render suite, not
   just `raster_first_shade`, and confirm the opcode set stays inside §5.2's tag
   system — in particular that no kernel drags in `SNode`, texture, argpack or
   real-func (`create_function`/`insert_func_call`) objects. Algan's kernels are
   ndarray+scalar only today (`grep`: no `ti.field`, no `real_func` under
   `algan/rendering`), and the recorder must refuse to record if that changes.

---

## 10. Implementation phases

| Phase | Deliverable |
|---|---|
| 0 | §9 experiments; a throwaway probe, no product code |
| 1 | `algan/utils/taichi_frontend_trace.py`: key builder (§4.1-4.4), recorder (§6), in-memory trace, verify-mode comparison. No disk, no replay. Proves determinism on the real suite. |
| 2 | Replayer (§7) + the `.tic` oracle test. Still in-memory (record then replay in one process) — this is where the perf number becomes real. |
| 3 | Serialization + store + eviction + `clear_cache` integration. |
| 4 | Captured-environment capture and validation (§4.5), which is the last correctness gap; `ALGAN_TAICHI_FRONTEND_TRACE` default OFF until the check script is green on the full render suite. |
| 5 | Default ON, with the fast-suite baselines re-verified. |

New files: `algan/utils/taichi_frontend_trace.py`,
`benchmarks/_frontend_trace_check.py`. Touched: `algan/__init__.py` (apply the
patch at line 77, and a `frontend_traces=` bucket in `clear_cache`, line 179),
`algan/environment.py` (register the two env vars). Nothing structural in
`algan/rendering/taichi_runtime.py` — its existing compile logger
(`ALGAN_LOG_TAICHI_COMPILES`) is the measurement tool throughout.

---

## 11. Alternatives, and when to prefer them

* **The daemon** ([`algan/daemon.py`](algan/daemon.py)) already gets warm re-render
  to ~1s by not paying any of this per run. For interactive authoring it wins
  outright, and this design should never be sold as competing with it.
* **More `taichi_warmstart`-style micro-optimization.** ~1.0s of the 11.3s is
  `Kernel.ast_builder` (42,893 calls) and `config` (83,650) getters that could
  be hoisted inside taichi's Python layer, and `Expr.set_dbg_info` is another
  79,145 calls whose cost buys only error messages. Cheap, low-risk, no new
  concepts, maybe 15-20% — versus this design's ~2.3x. If Phase 0 experiment 2
  fails, this is the fallback plan.
* **AOT + C-API.** Assessed and rejected: it removes the frontend but not the
  backend, needs every one of 2^16 template variants enumerated ahead of time,
  cannot express runtime-composed fragment pipelines, requires explicit
  dtype/ndim on all 46 bare `ndarray()` annotations, and has no Python loader,
  so all launch plumbing would move into ctypes.
* **Fewer/smaller kernel variants.** The frontend cost is proportional to
  inlined body size; splitting `raster_first_shade` would cut it directly. That
  is a renderer-architecture decision with its own perf consequences
  (`lean-triangle-only-kernel` measured ~37% *faster* execution from register
  pressure alone), and is out of scope here — but it is the only option that
  helps the cold-compile path too.

---

## 12. Known hazards

| Hazard | Mitigation |
|---|---|
| Wrong trace runs wrong code silently | Coarse source fingerprint (§4.3), captured-environment validation (§4.5), verify mode (§8.2), `.tic` oracle in CI (§8.1) |
| Kernel definition order changes the Taichi key | `K` includes the generated kernel name (§4.2), so traces and `.tic`s invalidate together |
| Absolute paths in debug info | Paths are in the source fingerprint; traces are machine-local and never shipped |
| Trace store grows without bound | LRU cap in the unified cache dir; `clear_cache` bucket |
| Editing a kernel while a render runs | Same existing hazard as the Taichi JIT (`taichi-jit-source-edit-hazard`); the fingerprint reads files at materialize time, so a half-saved file yields a miss, not a corrupt trace |
| taichi upgrade | Version gate (1.7.x) plus `__version__`/commit in `K`; a new taichi is a silent no-op until re-validated |
| Concurrent processes sharing the store | Write to a temp file and `os.replace` into place; readers never see a partial trace |
