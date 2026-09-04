# Taichi fork roadmap: what to take from Quadrants, what to build, and on which base

Status: **written 2026-09-03 as a plan; §6's gate was run and Track B begun on 2026-09-04.**
Quadrants is now the default compiler and carries its own patch set. **`MIGRATION.md` is the record
of what was executed and what it measured** — read it first if you want the current state; read this
for the design, the survey it rests on, and the items still untaken. §6.1 holds the gate's results,
and the "done" markers through §7.3 say which steps are closed and what closed them.

Written from a read-only survey of Taichi v1.7.4, the Quadrants fork at `b9e953111` (2026-09-02),
and this repository. It is self-contained: a fresh session can execute it from §0 without any other
context. Every claim carries the file, commit or measurement it rests on; "verified" means read in
source, "measured" means run on the stated machine, "projected" means an estimate. Nine of those
claims were falsified by running them — each is corrected in place, and `MIGRATION.md` §4 lists them
together.

## 0. How to use this document

**Sources to have on disk** (all read-only; nothing here modifies them):

```
# Pristine upstream, the base of Algan's current fork
git clone --depth 1 --branch v1.7.4 https://github.com/taichi-dev/taichi.git taichi-src
# The Quadrants fork, with history (needed for `git show <sha>` and `git log -S`)
git clone --filter=blob:none https://github.com/Genesis-Embodied-AI/quadrants.git quadrants-src
git -C quadrants-src checkout b9e953111        # the revision this plan was written against
```

Paths below are relative to those two trees, or to this repository. Quadrants' C++ lives under
`quadrants/`, its Python under `python/quadrants/`; its identifiers are renamed
(`taichi::`→`quadrants::`, `TI_`→`QD_`, `ti.`→`qd.`, `ticache`→`qdcache`), and its Python binding
layer is nanobind, not pybind11. Its pre-rename history is in `gstaichi-releases.md` (versions
`gstaichi` 1.0.1–4.7.0b1, then `quadrants` 0.3.0–1.3.0; the counter was reset at the rename).

**Algan's own prior work this plan builds on**: `taichi_patches/README.md` and the two patches;
`algan/rendering/taichi_runtime.py`; `algan/utils/taichi_warmstart.py`;
`algan/utils/taichi_fast_launch.py`; `algan/rendering/mps_zero_copy.py`; `algan/rendering/mps_compat.py`;
`algan/rendering/raytracing/arena_args_taichi.py`; `algan/rendering/DESIGN_mps_support.md`;
`algan/rendering/DESIGN_mps_zero_copy.md`. Three deleted design notes hold measurements cited here
and can be recovered with `git show aa7d198^:DESIGN_frontend_trace_cache.md`,
`git show aa7d198^:DESIGN_taichi_arch_coexistence.md`, `git show aa7d198^:DESIGN_taichi_argument_loads.md`.

**Decision vocabulary used in §4–§5**: *copy* = take Quadrants' commit(s) and port them (renames,
pybind11 for nanobind); *scratch* = implement in Algan's fork because Quadrants has not done it or
did it in a form that does not fit; *keep* = Algan's existing patch/workaround stays; *skip* = not
worth doing on either base.

**Verification conventions** (from `CLAUDE.md`): `uv run -m pytest -q --fast` after every Algan-side
change; the full suite before pushing; `pytest -q tests/full_renders` for anything that changes kernel
IR or codegen; re-baseline only after inspecting the diff. Any change to the compiler invalidates the
offline cache; clear it (`clear_cached_kernels()`) before A/B numbers, and run one process per arm for
anything a `ti.static` gate controls.

## 1. Context

Algan renders with ~18,000 lines of custom Taichi kernels (`algan/**/*_taichi.py`): 55 `@ti.kernel`s
(26 templated with `ti.template()` flags and tuples of injected `@ti.func` material stages; 25 with
bare `ti.types.ndarray()` arguments), 15–50 ndarray arguments each, hundreds of launches per render,
every argument a torch tensor, no `ti.field`, no autodiff, no GGUI, no AoT. It runs on CUDA, CPU and
Apple Metal. The Metal path needs a forked Taichi: `taichi_patches/0001` imports a torch MPS tensor's
`MTLBuffer` as a zero-copy ndarray with a non-zero byte offset (Algan's arrays are slices of one
arena), `0002` fixes an MSL narrowing-cast codegen bug and turns Metal pipeline-build aborts into
errors that name the kernel. The fork is built for macOS arm64 only by `.github/workflows/taichi_build.yaml`;
Linux and Windows users run the stock PyPI wheel.

The maintainer asked what else is worth patching in Taichi, across six questions: (1) the ~20 s
warm-cache kernel load; (2) one `ti.init` per thread; (3) AoT without a Python loader or final
bytecode; (4) codegen a tag could fix; (5) shader/scatter operations a `ti.func` cannot express;
(6) anything else. A first survey against Taichi 1.7.4 produced a proposal list. It then emerged that
**Quadrants** (Genesis-Embodied-AI's fork of Taichi, forked June 2025, `import quadrants as qd`, LLVM 22,
actively maintained: last commit 2026-09-02, releases every ~2 weeks, `quadrants` 1.3.0 on PyPI) has
already built several of those proposals. Upstream Taichi is dormant (last commit 2025-07-30; 1.7.4
is the latest release; issues #8791 "is Taichi still under development?" and #8795 "Python 3.14?"
unanswered; scikit-build 1.0 broke rebuilding it, #8799). So Algan's fork is its release channel
either way, and the real question is **which base to fork and what to carry**.

## 2. Facts the plan rests on

### 2.1 Where the warm-cache 20 s goes (Taichi 1.7.4)

- The offline-cache key is SHA-256 over compile-config fields + device caps + params/rets +
  **the frontend IR** + autodiff mode (`taichi/analysis/offline_cache_util.cpp:170-200`). Not the
  kernel name, not line numbers, not `kernel_counter` (probe-verified on the installed wheel; the
  deleted trace-cache note claimed otherwise and was wrong). It *does* cover inlined `@ti.func`
  bodies and captured globals. **Six places in this repo say the cache does not invalidate on
  `@ti.func` edits and are wrong**: `agent_guidance/taichi.md:2`,
  `algan/rendering/raytracing/settings.py:2455`, `tests/unit_tests/test_direct_specular_lobe.py:195`,
  `benchmarks/_watertight_check.py:42`, `benchmarks/_pn_deletion_profile.py:17`,
  `algan/rendering/raytracing/DESIGN_mesh_identity.md:1637`.
- Because the key needs the IR, the Python AST transform (`Kernel.materialize` → `transform_tree`,
  `python/taichi/lang/kernel_impl.py:634-683`) runs before any lookup; `Kernel::init` invokes the
  AST callback eagerly (`taichi/program/kernel.cpp:128`). `Kernel::set_kernel_key_for_cache` exists
  in C++ (`taichi/program/kernel.h:49`) with no pybind; `make_kernel_key` short-circuits when a key
  is preset (`taichi/compilation_manager/kernel_compilation_manager.cpp:180-196`).
- The `.tic` holds textual LLVM IR (`taichi/codegen/llvm/compiled_kernel_data.cpp:56,76`). CPU caches
  the O3'd module (`codegen_cpu.cpp:243`); **CUDA caches the unoptimized module** and runs the
  legacy O3 pipeline + NVPTX codegen + `cuModuleLoadDataEx` on every warm load
  (`taichi/runtime/cuda/jit_cuda.cpp:8-45,76-240`); `verifyModule` runs twice.
- Measured on the maintainer's CUDA box (warm cache, one `Square` `save_frame`): frontend 12.9 s,
  backend 7.0 s; `raster_first_shade` alone 11.3 s / 6.5 s (deleted `DESIGN_frontend_trace_cache.md` §1;
  taken before `algan/settings/_startup.py:27` raised `CUDA_CACHE_MAXSIZE` to 4 GiB, so part of the
  7.0 s may have been driver-side ptxas re-JIT). Measured on a 4-vCPU CPU-only box (2026-09-03,
  `tests/fast/scene.py`, one frame, warm): 23 kernels, frontend 3.98 s, `.tic` load 0.26 s,
  first-launch ORC codegen 0.96 s; the transform is ~94 % Python (pybind ≈0.5 s of 7.95 s profiled);
  5,496 inlined `ti.func` calls collapse to 130 distinct (func, signature) pairs and 86 % of
  func-transform time is repeats; `kernel_impl._inside_class` costs 1.1 s of a 3.0 s `import algan`.
  The unoptimized `sheet_resolve_shade_arena` module (CUDA's size class) is 31 MB of text, 1.4 MB as
  bitcode; LLVM-tool proxies on it: parse 0.59 s, O3 1.05 s, NVPTX llc 0.68 s.

### 2.2 Codegen and runtime (Taichi 1.7.4)

- Argument loads: `get_args_ptr`/`get_struct_arg` emit plain `CreateLoad`s
  (`taichi/codegen/llvm/codegen_llvm.cpp:2944,2961`); `ExternalPtrStmt` re-loads `data_ptr` and each
  shape dim at every use (`:1893-1928`); no `!invariant.load` anywhere. The CUDA JIT runs O3 with
  LICM/GVN (`jit_cuda.cpp:181-206`); what is missing is the aliasing fact. The deleted
  `DESIGN_taichi_argument_loads.md` measured ~3,100 of 37,100 static instructions in
  `sheet_resolve_shade` as argument re-loads and +18 % device time for the arena offset-table
  convention (GTX 1050).
- **`gpu_max_reg` is inert**: `JITSession::add_module(M, int max_reg = 0)` is only ever called as
  `add_module(std::move(module))` (`taichi/runtime/llvm/llvm_runtime_executor.cpp:181`); the field is
  read only by the pybind setter and the cache key. `ALGAN_GPU_MAX_REG` has only ever changed the
  cache key. Quadrants reached the same conclusion and deleted the option (`3e45a7a7c`, #890).
- `advanced_optimization` gates only CHI passes (`taichi/transforms/simplify.cpp:528-577`,
  `extract_constant.cpp:56`). **`cache_loop_invariant_global_vars` is NOT among them**: it runs under
  its own flag, default on (`compile_to_offloads.cpp:186`, `compile_config.h:27`), so it is live in
  Algan's renders today even with `advanced_optimization=False`.
- The CUDA launcher does a `malloc_async` + `memcpy` + `mem_free_async` of the arg buffer on every
  launch (`taichi/runtime/cuda/kernel_launcher.cpp:149-167`); `RuntimeContext` is byval with only a
  pointer to that buffer. `Program` is a hard singleton (`taichi/program/program.cpp:141`).
- 1.7.4 clamps the CUDA target to `sm_86` with `+ptx63` (`taichi/rhi/cuda/cuda_context.cpp:73-83`);
  LLVM 15's NVPTX knows nothing newer. Hopper/Blackwell run via driver JIT of that PTX.
- Nested tuples **do** work as `ti.template()` arguments (probe-verified; dict/list values fail via
  weakref). `shading_taichi.py:1387`, `agent_guidance/taichi.md:5` and `agent_guidance/rendering.md:41`
  are stale.

### 2.3 What Quadrants is (verified in `quadrants-src`)

- Fork of Taichi at 2025-06-03; 594 commits since; `import quadrants as qd`; **`import taichi` does
  not work** (`python/taichi/` holds one font file, no `__init__.py`). LLVM **22.1.0** from
  org-hosted prebuilt archives (`.github/workflows/scripts/qd_build/llvm.py:19-24`,
  `Genesis-Embodied-AI/quadrants-sdk-builds`; 1.7.4's LLVM 15 archives sit on three personal GitHub
  accounts). Build backend **scikit-build-core** (`pyproject.toml:88-97`, so #8799 does not apply);
  bindings **nanobind ≥2.0,<2.14** (`cmake/PythonNanobind.cmake`); `-Werror` still on but
  `common/core.h:170` already spells `operator""_f`, so it builds on macOS 26 / clang 21 (Quadrants'
  own CI: `macos-26`, `ubuntu-22.04`, manylinux 2_28 x86_64 + 2_34 aarch64, `windows-2025`).
- Backends kept: CPU x64/arm64, CUDA, AMDGPU, Metal (still SPIR-V-based via SPIRV-Cross), Vulkan;
  a Python interpreter backend for debugging. Removed: OpenGL, DX11/12, GGUI, C API, **AoT**, CLI,
  textures, ArgPack, phone-home `_version_check.py`. CMake options: `QD_WITH_{LLVM,METAL,CUDA}` on,
  `QD_WITH_{VULKAN,AMDGPU,CUDA_TOOLKIT}` off (`cmake/QuadrantsCore.cmake:1-7`).
- Wheels (`quadrants` 1.3.0, PyPI): cp310–cp313, `macosx_13_0_arm64` 26.7 MB, manylinux x86_64
  41.0 MB, manylinux aarch64 38.3 MB, win_amd64 31.0 MB (Taichi 1.7.4: 50.4 / 56.3 / — / 83.3 MB).
  `requires-python >=3.10,<3.14`; **no 3.14 either** (`ast.Str` at `ast_transformer.py:515,1626,1678`
  and `function_def_transformer.py:781`; CI deletes cp314 from the image). macOS minimum 13.0.
- CUDA target: clamp raised to `sm_121` (`quadrants/rhi/cuda/cuda_context.cpp:70`), no `+ptx63`,
  new pass manager at O3 (`jit_cuda.cpp:372-379`), `CallingConv::PTX_Kernel` required because O3
  strips `nvvm.annotations` (`llvm_context.cpp:1031`).
- **Every `ti.` name Algan's kernels use exists unchanged in Quadrants** (`ti.template`, `ti.static`,
  `ti.types.ndarray(dtype, ndim)` with an identical signature, `ti.math.*`, `ti.func/kernel`,
  `ti.Vector/Matrix`, atomics, `ti.ndrange`, dtypes, `ti.sync`, `ti.random`, profiler). No kernel
  body needs to change under `import quadrants as ti`. The breaking differences are all in Algan's
  runtime glue (§7.3, 13 items).

## 3. Answers to the six questions, after Quadrants

1. **Warm-cache load.** Two halves. Frontend: Quadrants' "fastcache" (opt-in `@qd.kernel(fastcache=True)`)
   keys a Python-side index on kernel source text + transitively visited `@qd.func` sources
   (re-hashed on load) + a pruned argument-type hash + config + device caps, stores the C++ cache key,
   and on a hit walks only the `FunctionDef` (declaring parameters) and never the body
   (`python/quadrants/lang/_fast_caching/`, `kernel.py:411-509`,
   `ast_transformers/function_def_transformer.py:566-569`; claimed 7.2 s → 0.3 s process-start load
   on a Genesis benchmark). **It is the right design and the wrong instrument for Algan**: it does
   not hash captured globals — it forbids them with a purity check that Algan's kernels violate on
   every `ALGAN_`-derived module constant and every `ti.static(rt_settings...)` read — and any
   `ti.template()` slot holding a tuple of funcs or a dtype fails closed, disabling it for exactly the
   26 templated megakernels. Decision: **build Algan's own source-keyed index (scratch, design
   borrowed)**, §4 item 1. Backend: Quadrants' `PtxCache` (**copy**) skips O3 + NVPTX on a warm hit;
   its three-week-old per-task artifact cache replaces the IR with PTX entirely (**not now**).
2. **Two `ti.init` environments in one thread.** Quadrants kept and hardened the singleton
   (`per_task_artifact_cache.h:110` relies on it). Cheap to fork (~15 C++ lines + ~150 Python) but
   the value case did not survive: one eligible CPU-prep kernel (~5 %, projected), and item 1 shrinks
   the +24 s device-switch cost anyway. **Skip.**
3. **AoT from Python / final bytecode.** Quadrants **deleted** AoT and the C API. The frontend skip is
   item 1; the "final bytecode" is the PTX cache (item 2) or, later, the per-task artifact tier.
   **Skip** the loader; keep `benchmarks/_taichi_c_api_*.py` as records only.
4. **Codegen.** Quadrants never added `!invariant.load` on argument loads, never wired `max_reg`,
   never consumed the read/write ndarray analysis for `__ldg`, never changed `exp` under fast_math.
   All four are **scratch**. What it did do and Algan should **copy**: persistent per-launch arg/result
   buffers, a per-pass IR dump for bisecting the `advanced_optimization` miscompile, two
   loop-invariant-caching correctness fixes, three compile-time wins.
5. **Shader expressiveness.** Unchanged in Quadrants: early `return` inside runtime control flow in an
   inlined func is still rejected (`ast_transformer.py:533-536`). Everything else missing is Algan's
   own stage contract. **Scratch** (Algan-side contract v2; a Python-only early-return transform).
6. **Anything else.** Metal: native f32 atomics, a shared torch command queue, a pending-launch valve
   (**copy**). Toolchain: Quadrants is the model for org-hosted LLVM, scikit-build-core, slim wheels,
   no phone-home; Python 3.14 is open on both. Correctness: several fixes worth copying (§5).
   Strategic: **rebasing the fork onto Quadrants is the better long-term bet, gated on a two-day
   fact-finding pass** (§6).

## 4. The proposal list, item by item

Columns: what Quadrants has; the decision; where the code is (Quadrants commit → Algan target).

| # | Item (question) | Quadrants status | Decision | Source / size |
|---|---|---|---|---|
| 1 | Source-keyed cache index that skips the AST transform on a hit (Q1 frontend) | **partial/different**: fastcache, opt-in, purity-gated, fails closed on func tuples | **scratch, design borrowed** | C++: pybind `Kernel.set_kernel_key_for_cache`/`get_cached_kernel_key` (~6 lines, `taichi/python/export_lang.cpp`), `DeviceCapabilityConfig.hashed_key()` (~10 lines; Quadrants `export_lang.cpp:330-333`, #850 `39f20102a`); Python: FunctionDef-only walk on a hit (3 lines, monkeypatchable; Quadrants `function_def_transformer.py:566-569`), new `algan/utils/taichi_source_key.py` (~400-600 lines) |
| 2 | PTX beside/instead of IR in the cache (Q1 backend) | **done twice**: `PtxCache` (whole-module PTX keyed on IR text + SM + fast_math; IR still parsed+verified) and a per-task PTX artifact that replaces the IR | **copy `PtxCache`**; per-task tier **not now** | `2e5ed0771` (#130) + `89987df5e` (#361, SM in key) + `5f926630c` (#580, honour `offline_cache=False`): `quadrants/runtime/cuda/ptx_cache.{h,cpp}` (~330 lines) + hook at `jit_cuda.cpp:313-318`. Per-task: `f6df02c2e` #893, `81ed162ac` #880, `bdb9b49fe` #875, `42cc5f74a` #864 — entangled, 3 days old, no eviction |
| 3 | Bitcode instead of textual IR | absent | **skip** (subsumed by 2) | — |
| 4 | Skip `verifyModule` on load / second verify in the CUDA JIT | absent (still twice) | **scratch, low** | 4 lines gated on `debug` |
| 5 | O3 before caching on CUDA | absent (PTX cache instead) | **skip** | — |
| 6 | CPU ORC object cache | absent | **scratch, low** (CPU backend ≈1.2 s) | ~120 lines `jit_cpu.cpp` |
| 7 | Memoise a `ti.func`'s IR per (func, signature) | absent (Python `Func.__call__` still re-parses per call site, `func.py:53-92`) | **scratch, after item 1** (a hit never enters a func body) | ~750 lines |
| 8 | `_inside_class` import cost (Q1 micro-fix) | `get_pos_info` memoised (#858 `895dd5ea1` — same fix Algan's warmstart has); `_inside_class` untouched | **scratch**: 5-line monkeypatch in `taichi_warmstart.py` (−1.1 s per process, measured); the other two micro-fixes are row 37 | `python/taichi/lang/kernel_impl.py:1067-1076` |
| 9 | Cache hit/miss stats API; mid-run cache flush; stale lock; int64 cap | stats: rich `CompileResult` + observation dataclasses (`_kernel_types.py:16-62`); flush: `Program.dump_cache_data_to_disk` pybound (`export_lang.cpp:488`); lock and cap unchanged | **copy** the flush pybind (1 line; a daemon killed by a signal currently loses every kernel it compiled, since `dump` runs only at `Program::finalize`) and the stats idea (~30 lines: three counters in `KernelCompilationManager` + a `Program` getter); **scratch** the stale-lock rule (Algan-side, ~15 lines in `init_taichi`: the metadata lock is a bare `O_EXCL` file with no staleness rule, so a crash while holding it leaves every later run missing every kernel and discarding what it compiles, with only a `TI_WARN`); cap **skip** (row 48) | Algan then deletes the fd-2 stderr hijack in `taichi_runtime.py:141-204`. Note: the cleaner evicts 25 % of entries **by count** (LRU on `last_used_at`, `taichi/util/offline_cache.h:225-262`), not "to 75 % of the cap" as `taichi_runtime.py:611-612` says; fix that comment |
| 10 | Launch-path cost (`taichi_fast_launch` residue) | extensive Python work, but `LaunchContextBufferCache` marks raw torch tensors non-cacheable (`_func_base.py:858-874`) so it would never engage for Algan | **keep** `taichi_fast_launch.py`; **copy** batched `set_args_*` pybinds (design: `export_lang.cpp:686-699`, `kernel.py:822-834`), `5a20dbc66` (#267, don't copy kernel params, 12 lines), and the persistent per-handle arg/result buffers from `4e35bd556` (#619; `kernel_launcher.cpp:365-377,463-480,533-535,579-592`, ~60 lines) | Consider `e8a44de15` (#654, drop per-launch `stream_synchronize`) after measuring |
| 11 | Two resident Programs (Q2) | absent; singleton hardened | **skip** | run `benchmarks/_taichi_arch_coexistence_probe.py` on a CUDA box only if a second prep kernel ever clears an A/B |
| 12 | AoT Python loader (Q3) | deleted | **skip** | — |
| 13 | `!invariant.load` on argument loads (Q4) | absent (only read-only SNodes, `codegen_cuda.cpp:583-616`) | **done (scratch), built and hoist-confirmed; not timed** — `quadrants_patches/0004-llvm-invariant-load-kernel-args.patch`, against **Quadrants v1.3.0**. Built by `.github/workflows/quadrants_build.yaml` on the Linux runner (~20 min); `verify_invariant_load.py` measures LLVM 22 / x64 CPU at 11 `!invariant.load` + 2 `!dereferenceable` sites on, 0 off, and argument base-pointer re-loads in the loop body **18 → 0** (body 112 → 30 lines, and the survivors vectorize). No CUDA and no timing yet — that needs a real box and §5's order | `mark_invariant_arg_load` helper + the 4 argument-buffer `CreateLoad`s (`visit(ExternalPtrStmt *)` ×2, `get_struct_arg`, `get_args_ptr`) + `!dereferenceable(args_size)`; gated by a new `invariant_arg_loads` compile-config field **which had to be added to `get_offline_cache_key_of_compile_config` by hand** — that function serializes an explicit field list, so a flag left out of it lets two A/B arms share an artifact (`cache_loop_invariant_global_vars` is already missing from it). Excludes `@qd.real_func` callees: their arg buffer is a caller `alloca` written just before the call, so inlining would let LLVM hoist a load above the store that fills it. 135 lines / 5 files, applies clean. `addrspace(1)` deliberately **not** included — that is row 15, and #866 (`91c590563`) is post-v1.3.0 so its `maybe_tag_amdgpu_global_ptr` is not in this base (expect a conflict on the same two functions when rebasing past it). Verify per §5 before believing anything about speed |
| 14 | Make `gpu_max_reg` real; per-kernel `maxnreg` | knob **deleted** (#890) | **scratch**: 1 line at `llvm_runtime_executor.cpp:181`, then `loop_config(max_reg=)` → `OffloadedStmt` → `maxnreg` nvvm annotation beside `maxntidx` in `llvm_context.cpp:837-847` (~75 lines) | measure ptxas -v first (§8) |
| 15 | Read-only ndarray hint → `ld.global.nc` / invariant | absent; but the per-arg read/write analysis is now consumed by LLVM codegen for another purpose (`codegen_llvm.cpp:2141`) — proof both halves work | **scratch** (~120 lines) | `detect_read_only.cpp`, `codegen_cuda.cpp:606-635` |
| 16 | Args byval in parameter space | absent | **skip** (item 10's persistent buffer covers the launch half; item 13 the load half) | — |
| 17 | `advanced_optimization` miscompile | not fixed; CSE made per-offload (`5f8138df2`, delicate — do not copy) | **copy** the per-pass IR dump `QD_DUMP_SIMPLIFY` (`quadrants/transforms/simplify.cpp:515-600`, ~40 lines) to bisect; keep the pin; **copy** `78ed263a9` (#376) and `3d9af7189` (#812), two silent stale-read fixes in `cache_loop_invariant_global_vars`, which runs in Algan today (§2.2) | then re-test `ALGAN_ADV_OPT=1` |
| 18 | Fast `expf` under fast_math | absent (`codegen_cuda.cpp:350` still `UNARY_STD(exp)`) | **scratch**, 8 lines | copy the `log` branch shape |
| 19 | Compile-time wins | done | **copy** `d5ab903d4` (#577, `whole_kernel_cse` 2.5× on large kernels, 1 file), `b65c5f6df` (#579, skip IR verifier unless debug), `f3ccac394` (#729, cfg pass on multi-task kernels); **no patch**: try `cfg_optimization=False` (documented ~6× faster compile for 1–5 % runtime) | one process per arm |
| 20 | Early `return` in an inlined func (Q5) | unchanged | **scratch**, Python-only (~250 lines, shippable as a monkeypatch) | `ast_transformer.py:821-823` in 1.7.4 |
| 21 | Stage contract v2, scene bundle, `algan.shading` helpers, user buffers, seed (Q5) | n/a (Algan-side) | **scratch** (Algan) | `shading_taichi.py:933-952,1384-1413,1539`; fix the nested-tuple docs first |
| 22 | Zero-copy MPS ndarray import (patch 0001) | **absent**: DLPack is export-only (`dlpack_funcs.h:13-23`); torch MPS tensors still round-trip through the host (`_func_base.py:846-857`); `import_mtl_buffer` exists but has no caller | **keep** | Quadrants' #846 (`97add824e`) shows the staging path 0001 removes is also *wrong* for partially-written arrays |
| 23 | MSL narrowing-cast fix (patch 0002 part 1) | **done independently** (`9542c0004`, #543, `spirv_ir_builder.cpp:945-961`) | drop on a rebase; upstream it to taichi-dev citing Quadrants | — |
| 24 | `ContinueStmt`/`gen_label_` fix (patch 0002 part 1b) | **not fixed** (`spirv_codegen.cpp:2032` byte-identical to 1.7.4) | **keep**; best upstream candidate Algan has | — |
| 25 | Metal diagnostics (patch 0002 part 2) | partial: nil library/function/pipeline guarded, `QD_DUMP_MSL`; no source window, no `.spv` dump | **keep**; adopt two ideas: the `QD_WARN`-not-`QD_ERROR` rationale (`metal_device.mm:190-198`) and the XPC-compiler-service diagnosis text | — |
| 26 | Metal f32 atomics | **done** (`b44f0e279`, #788: uncomment the caps, `set_msl_version(3,0,0)`, `MTLLanguageVersion3_0`; one file) | **copy as patch 0003** (~39 lines; skip its nil guards, 0002 has better ones) | needs macOS 13 |
| 27 | Metal 64-bit / f64 atomics | not possible (documented `atomics.md:30`; MSL has no double) | **keep** MPS-friendly narrowing | — |
| 28 | Metal 31-buffer limit | solved via physical storage buffer pointers (`eff73c380`, #392) — needs a **forked SPIRV-Cross submodule** and macOS 13 | **not now**; keep arena packing; revisit on a rebase | — |
| 29 | Shared torch MPS command queue | **done** (`718bb695e` #618 + `64d9ea240` #694): `qd.init(external_metal_command_queue=..., ..._is_torch_queue=True)`, `python/quadrants/interop/_torch_mps.py` | **copy as patch 0004** (~120 C++ + 75 Python) — deletes the per-launch `ti.sync()`+`torch.mps.synchronize()` pair in `mps_zero_copy.py` | — |
| 30 | Pending-launch drain valve | done (`kMaxPendingLaunches = 32`, `runtime/gfx/runtime.cpp:1218-1238`, from `ae2d1c0b5` #551) | **copy** (~10 lines); measure with 29 | hypothesis for upstream #8794 |
| 31 | `random_seed` on SPIR-V | done (`989e4ae4a`, #786, 11 lines) | **copy** | Algan has 3 `ti.random` sites |
| 32 | cp314 wheels (Q6) | **absent** (same `ast.Str` sites; CI excludes cp314) | **scratch** on either base: 3 `ast.Str` sites, `pybind11>=3` (1.7.4) , `scikit-build<1` pin (1.7.4 only) | — |
| 33 | Slim wheel and the `operator""_f` build fix (Q6) | Quadrants is the model: C API/OpenGL/DX gone (`cmake/QuadrantsCore.cmake:1-7`), `core.h:170` fixed | **copy** the `core.h` fix (~30 one-line edits; unblocks clang ≥ 21 / `macos-latest`); **scratch** the build config on 1.7.4: `TI_WITH_C_API=OFF` alone removes `libtaichi_c_api.so`, 53 MB of the 162 MB installed Linux wheel (measured); OpenGL/Vulkan/DX11/DX12 off (the maintainer's no-Vulkan macOS build measured 37 MB vs upstream's 50.4 MB); a 3-line guard so a Linux/Windows build stops compiling `external/SPIRV-Tools` (`cmake/TaichiCore.cmake:286-291`); bypass `build.py`'s unconditional `setup_vulkan()`/`setup_os_pkgs()`. Turning Vulkan off also removes the headless probe crash structurally: `with_vulkan` becomes a constant-false lambda (`taichi/python/export_misc.cpp:162`), so `taichi_runtime._taichi_arch`'s guard becomes belt-and-braces. Toolchain mirrors and notices are rows 52–53 | `taichi_build.yaml` |
| 34 | LLVM 20/22 for native sm_89/90/120 | done (`73a9bd7b4` #394; clamp 86→121) | **not portable to LLVM 15**; comes free with a rebase (row 50 for the 1.7.4-side decision) | — |
| 35 | CUDA graphs | done, mature, one graph per `@qd.kernel` (`graph_manager.cpp`) | **skip** (granularity does not fit Algan's per-stage kernels) | — |
| 36 | `BufferView`, `qd.Tensor`, tiles, subgroups, streams, AMDGPU | done | **skip** for Algan (BufferView is the arena convention as a type, over Quadrants-owned memory only) | — |
| 37 | Lazy `DebugInfo`/pos-info when `debug` is off; hoisted `config()`/`ast_builder()` getters (Q1 micro-fixes) | pos-info memoised (#858, same as Algan's warmstart); `DebugInfo(ctx.get_pos_info(node))` still constructed unconditionally at 12 sites (`ast_transformer.py:111,637,1146,...`); getters partly hoisted (`kernel.py:842-844,1010-1012`, a weakref to the `Program` because a pybind attribute lookup is ~200 ns vs 5 ns) | **scratch, low** (~30 + ~20 lines, fork-Python or monkeypatch): gate the 12 sites on `impl.current_cfg().debug` and hoist the two getters into the transformer context; mostly moot on hits once item 1 lands, so do it after item 1 and re-measure | pos-info was 13 % of the transform even with the memo; getters ~0.1 s (measured on the CPU box) |
| 38 | Frontend/CHI IR serializer; Python record-and-replay of the pybind stream; convert Algan's funcs to real functions and wire `irpass::inlining` (Q1 alternatives) | none of them (Quadrants deleted the unused inlining module, #316) | **skip**: all three are superseded by item 1 (the artifact already carries everything a launch needs); real functions still have no SPIR-V codegen and forbid Python-value arguments | — |
| 39 | Keep compiled kernels across `ti.init`; per-arch offline-cache subdirectory (Q2 variants) | absent (`qd.init` still resets and drops every kernel, `python/quadrants/lang/impl.py:632-643`) | **skip** (kernel ownership surgery, more invasive than row 11 for less; the per-arch directory only matters with two resident Programs) | — |
| 40 | Algan-owned AoT variant manifest; shipping pre-built kernels in the wheel; promoting the ctypes C-API shim (Q3) | AoT and the C API are deleted in Quadrants | **skip** (the manifest is an alternative to item 1 with worse key soundness; prebuilt kernels would need it plus per-SM PTX and a variant corpus; the shim has no i64 scalars and leaks imported memory). One note to act on: the 1.7.4 offline cache is not portable across machines because `cpu_max_num_threads` is in its key (`offline_cache_util.cpp:57-64`) and Algan does not pin it — pin it in `taichi_init_kwargs` (after an A/B, it also sets CPU thread counts) if a shared cache is ever wanted | `benchmarks/_taichi_aot_build*.py`, `_taichi_c_api_shim.py` stay as records |
| 41 | `ti.pow(x, 5.0)` Fresnel term → four multiplies at 11 sites (Q4, Algan-side) | n/a | **scratch (Algan), low**: a real-exponent `pow` lowers to `__nv_powf`; pixels move at the last bit, so re-baseline | `shading_taichi.py:921,1142,1209,1236`; `path_tracer_taichi.py:542,965,989,1009,1798`; `wavefront_kernels_taichi.py:1163,1182` |
| 42 | Rematerialise pure argument chains across offloads; CPU `ndrange` nested-loop lowering (Q4) | `ndrange(axes=)` (#710) reorders nesting but keeps the flattening; no rematerialisation change | **skip**: Algan already keeps offsets as `ti.static(ArenaView)` exprs and has no `ti.ndrange` in any big kernel (9 uses, all in small kernels) | — |
| 43 | The "64 runtime-argument ceiling" belief, and raising it (Q4/Q6) | still a Python-side counter (`python/quadrants/lang/_func_base.py:641`, `MAX_ARG_NUM`) | **docs only (scratch)**: it is `max_arg_num = 64` in `kernel_impl.py:693,961-964`, not a C++ or codegen limit (the args struct is sized from the parameter list). Do **not** raise it: every extra ndarray argument costs a `set_arg` and per-use argument loads, and Metal's 31 buffers remain; the arena work already brought the widest kernel to 20 | `DESIGN_renderer_structural_candidates.md`, `agent_guidance/rendering.md` |
| 44 | Path-traced emission slot the NEE table can sample; a BSDF-closure contract for custom scatter under the path tracer (Q5) | n/a | **scratch (Algan), later**: the emission slot is small and safe after row 21 (`path_tracer.py:246-256` reads built-in slots only); the BSDF closure (eval/sample/pdf with MIS) is a renderer design, not plumbing, and no test exercises custom scatter under the path tracer | — |
| 45 | Separately compiled shader stages via real functions + `cuLink`; CUDA texture objects on LLVM backends (Q5) | real functions are still emitted into the calling task's module (`codegen_llvm.cpp:2793-2822` in 1.7.4); no `cuLink*` bindings; textures removed | **skip** (CUDA-only, ~1,500 and ~1,000 lines, a non-inlined call in the hottest loop, conflicts with the arena/zero-copy design) | — |
| 46 | Pass the concrete arch with `enable_fallback=False` instead of `ti.gpu` (Q6, Algan-side) | Quadrants keeps `adaptive_arch_select` with the same silent CPU fallback (`python/quadrants/lang/misc.py:838-848`) | **scratch (Algan), ~10 lines** in `taichi_runtime._taichi_arch` / `taichi_init_kwargs`: never reaches the Vulkan/OpenGL probes, and turns the silent "falling back to CPU" (which leaves the live arch `cpu` while the render device says `cuda`, so `ensure_taichi_for_render` re-inits on every render, +24 s each) into an exception. Track A's Vulkan-off build removes the probe structurally too | `taichi_runtime.py:381-393,416-420` |
| 47 | `ti.sync()` off the main thread; `print` in a kernel forcing the slow launch path; the `[Taichi] Starting on arch=` stdout print (Q6) | sync: unchanged; print: still forces a sync (`python/quadrants/lang/kernel.py:880-882`); the banner is still an unconditional `print` (`misc.py:502`) | sync **skip** (Algan's worker deliberately never syncs; the off-thread failure is a raised `CUDA_ERROR_INVALID_CONTEXT`, not a crash); print **skip** (debug-only); banner **scratch, 2 lines** (route `misc.py:450` through the logger or gate it on the header env var), or wrap `ti.init` in a stdout redirect on the Algan side | `taichi/python/taichi/lang/misc.py:450`; `algan/__init__.py:83-89` |
| 48 | Widen `offline_cache_max_size_of_files` to 64-bit (Q6) | still `int` (`quadrants/program/compile_config.h:121`) | **skip** unless the cache ever exceeds 2 GB (44–66 megakernel artifacts at 30–45 MB); bitcode or the PTX tier makes it moot. The 8-line change is `compile_config.h:96`, `offline_cache.h:110`, `kernel_compilation_manager.{h,cpp}:147-161` if wanted | eviction semantics in row 9 |
| 49 | Metal SPIRV-Cross argument buffers; i64/f64 atomic emulation; a compile-time error for a 64-bit atomic on a device without the capability (Q6) | argument buffers not used (Quadrants took the physical-storage-buffer route, row 28); emulation not attempted (MSL has no 64-bit CAS and no `double`); a 64-bit atomic still fails at shader build | argument buffers **skip** (row 28 supersedes; ~200 Metal-only lines needing a Mac); emulation **skip** (per-element spin locks with no forward-progress guarantee; MPS-friendly narrowing is exact for what it touches); diagnostic **scratch, ~10 lines** in `visit(AtomicOpStmt)` (`spirv_codegen.cpp:1671-1718`), low, patch 0002 already makes the failure loud | — |
| 50 | Port the 1.7.4 fork to LLVM 20 for native sm_90/sm_120 (Q6) | Quadrants did the equivalent with LLVM 22 (row 34) | **skip on 1.7.4**: upstream PR #8735 is an unmerged, bot-flagged rewrite with no prebuilt LLVM, and Hopper/Blackwell already run the sm_86 PTX through driver forward-compatibility (Algan's kernels use no sm_89+ ISA features). Native targets arrive only with Track B | — |
| 51 | CUDA driver JIT cache env; Metal warm-load split; CUDA mempool release threshold (Q1/Q6 measurement items) | Quadrants relies on the driver cache and defeats it deliberately when measuring (`jit_cuda.cpp:449-467`); trims the pool on reset (`cuda_context.cpp:112-118`, #669 `f1841afcf`); its Metal path adds `QD_DUMP_MSL` and a persistent compute encoder but no warm-load timers | driver cache: already handled (`_startup.py:27` sets `CUDA_CACHE_MAXSIZE` before any `cuInit`); optional `CUDA_CACHE_PATH` under `ALGAN_HOME` **scratch, 1 line, low**. Metal split **measurement-first**: add timers to patch 0002's diagnostics around spirv-cross (`metal_device.mm:114-142`), `newLibraryWithSource` (`:1466-1472`) and pipeline creation (`:159`) and read them on the Mac probe before designing any MSL/`MTLBinaryArchive` cache. Mempool threshold **skip** (Algan's arrays are torch-owned; Taichi's pool sees only runtime temporaries); copy #669 only if VRAM held across re-inits shows up in `memory_model.py`'s preflight | — |
| 52 | Toolchain pins and mirrors (Q6) | Quadrants hosts LLVM 22 in an org repo (`Genesis-Embodied-AI/quadrants-sdk-builds`) and pins runner images, but does not persist its sccache across CI runs either | **scratch (Track A)**: mirror the four LLVM 15 archives (two live on personal GitHub accounts, `ti_build/llvm.py:29,44`), the Windows clang-14.0.6 zip (`compiler.py:54`) and the sccache tarballs into Algan's own release assets and point `llvm.py`/`compiler.py`/`sccache.py` at them; `actions/cache` on `~/.cache/ti-build-cache` (a warm sccache makes an edit-to-wheel cycle ~1 min instead of ~12; keep it under the 10 GB per-repo limit, sccache defaults to 40 GB); set `MACOSX_DEPLOYMENT_TARGET`; run `auditwheel show` (upstream forces the `manylinux_2_27` tag by hand); pin `macos-15`/Xcode 16.4, `ubuntu-22.04`, `windows-2022`; re-verify mirrored archives by hash. Under Track B: copy Quadrants' `scripts_new/` layout and its sdk-builds model | `.github/workflows/taichi_build.yaml` |
| 53 | Apache §4(b) modification notices; the `_version_check` phone-home (Q6) | Quadrants deleted `_version_check.py` (no outbound call); carries no per-file notices, only a README attribution | notices **scratch**: one "Modified by the Algan project (algan-taichi); see taichi_patches/README.md" line at the top of each patched file (13 today), optionally a NOTICE listing vendored licences; phone-home: set `TI_SKIP_VERSION_CHECK` from `algan/environment.py` now (Track C step 1) and **copy** Quadrants' deletion of the module in the fork (the reply would be about stock 1.7.4 anyway) | — |

Coverage check: every proposal in the first survey (its 19-row ranked table, its not-recommended list, its
question-by-question sections and its verification list) maps to a row above; rows 37–53 are the ones the
first draft of this file had compressed into other rows or left implicit.

## 5. Other things worth copying that were not on the list

Correctness (all verified in source; ranked):

- `78ed263a9` (#376) and `3d9af7189` (#810/#812): `cache_loop_invariant_global_vars` could serve a
  stale value for a load whose location is also an atomic destination, or when two dynamically
  indexed accesses to one ndarray may alias inside a serialized loop. Algan composites with atomics
  into ndarrays inside loops and the pass runs today. Two files, ~340 lines. **Highest-value copy.**
- `e98b7a91b` (#442) shared-memory offset not reset between CUDA kernels; `9fd62652f` (#391) shared
  array name collision — copy if Algan uses `ti.simt.block.SharedArray` (check with grep first).
- `a7c46c898` (#309) parallel offline-cache write corruption; `9f4cb3577` (#375) lock around compile —
  the daemon plus the test suite can compile concurrently.
- `a747b4eb3` (#451/#453) raise on device-allocation failure instead of proceeding — Algan's memory
  preflight (`render_loop.py`) currently guesses.
- `e3c40d165` (#419) abort kernel on assertion failure instead of segfaulting (`debug=True` only).
- `4331f125a` (#489) SPIR-V ID overflow on large kernels; `41b5086f1` (#513) Vulkan narrow (8/16-bit)
  storage caps — relevant to f16 ndarrays on the SPIR-V path; `ea7d24f14` (#372) `serialize=False`
  on Metal in `demote_atomics`; `7a9b6cb23` (#384) a Metal SPIR-V miscompile.
- `981ef70b9` (#278) ndarray leak; `cd19c293c` (#172) ndarray crash after `ti.reset`.
- `ab9a58ab5` (#847) `DeviceMemoryPool::allocate` ignored alignment — only if Algan allocates through
  Taichi's pool (it does not today; torch owns the arena).

Diagnostics and ergonomics: `QD_DUMP_SIMPLIFY` (item 17); `fe3cd8908` (#853) stop caches from
defeating IR-dump env vars; `8b8f12c72` (#440) named top-level loops for profiler attribution; the
in-kernel clock counter (`python/quadrants/lang/intrinsics.py:7-60`, ~50 lines) for timing inside a
megakernel; `7337f2926` (#324, 2-line early return in `materialize`) and `c670394fb` (#325, memoise
`CompiledKernelData` per specialization) — both Python, both apply.

Design ideas (not code): fail closed on unknown types in a cache key with a one-shot warning
(`args_hasher.py:107-125`); exclude process-local values from cache keys (#850 excluded the Metal
queue address — audit Algan's key inputs the same way); atomic cache writes via `mkstemp`+`os.replace`
(`python_side_cache.py:46-56`); the driver's `~/.nv/ComputeCache` does PTX→SASS for free, so cache
PTX, not cubin, and defeat the driver cache when measuring cold compiles.

Things Quadrants did **not** fix that this plan still needs from scratch: argument-load invariance
(13), `max_reg` (14), readonly hints (15), early return (20), `_inside_class` (8), stale lock (9),
cp314 (32). Nobody has solved the 86 % repeated-inlining cost (7).

## 6. The strategic decision: which base

Two options. **A**: stay on Taichi 1.7.4 and cherry-pick. **B**: rebase Algan's fork onto Quadrants
(`v1.3.0`, `ab9a58ab5`, or a newer tag) and port Algan's patches forward.

What B gives for free: PTX cache, persistent arg buffers, native Metal f32 atomics, the shared
queue, LLVM 22 with native sm_89–sm_121, scikit-build-core, nanobind (lower per-call binding cost),
org-hosted LLVM, the `core.h` fix, the loop-invariant correctness fixes, the compile-time wins, the
MSL cast fix, active maintenance. What B costs: a full pixel re-baseline of unknown magnitude (LLVM
15→22 float codegen, native SM targets), rewriting `taichi_fast_launch.py` against a class that
changes weekly, dropping Python 3.9, macOS 13 minimum, losing every offline cache once, a single
vendor whose roadmap is robotics (Vulkan default off, graphics removed), and the unknown status of
upstream bugs #8744/#8745/#8794 there. Under B Algan carries ~1.5 patches (0001; 0002's
`ContinueStmt` hunk + diagnostics) instead of 2, plus the scratch items above on either base.

**Recommendation: do not decide from this document. Run the fact-finding gate first (about two
days), then choose. Default to B if the gate passes.** *The gate has since been run; §6.1 is what it
measured, and it passes. Read §6.1 before acting on anything in §4-§5, which it corrects in seven
places.*

Fact-finding gate:
1. Now, on the 1.7.4 fork regardless: apply the `operator""_f` whitespace fix (`taichi/common/core.h:170-201`,
   ~30 one-line edits) so `macos-latest` builds.
2. One CI run: build stock Quadrants `v1.3.0` on the macOS runner with Algan's `taichi_build.yaml`
   adapted to `quadrants-src/.github/workflows/scripts_new/macosx/{1_prerequisites,2_build}.sh`
   (`brew install llvm@22`, `./build.py wheel`). This is the highest-information experiment: it
   answers whether the toolchain story is as clean as the scripts imply.
3. One afternoon, no fork, nothing committed: `pip install quadrants==1.3.0` in a scratch venv;
   `sed` `import taichi as ti` → `import quadrants as ti` (84 sites, 63 under `algan/`, `tests/`,
   `benchmarks/`); apply the 13 glue changes in §7.3; set `ALGAN_TAICHI_WARMSTART=0
   ALGAN_TAICHI_FAST_LAUNCH=0 QD_KERNEL_COVERAGE=0`; run `uv run -m pytest -q --fast` on CPU and read
   the pixel deltas of `tests/fast`. That number decides the question.
4. Same session: run the public repros for taichi-dev/taichi #8744 (dead-branch miscompile, CUDA/x64),
   #8745 (Metal result depends on an unrelated field's shape), #8794 (Metal segfault at iteration 512)
   against that install. Quadrants' history cites none of them.
5. If a CUDA box is available: `tests/full_renders` on the same install for the CUDA pixel deltas and
   a warm `save_frame` timing with `ALGAN_LOG_TAICHI_COMPILES=1`.

Pass criteria for B: the macOS build is green; `tests/fast` pixel deltas are ≤2 per channel or are
explainable float-contraction drift that an inspected re-baseline accepts; the three repros do not
reproduce (or have a bounded workaround); no Algan user depends on Python 3.9 or macOS < 13.

### 6.1 Gate results (measured 2026-09-04)

**All four pass criteria are met, three of them by a wider margin than the criterion asked for, and
the gate additionally found that Track A no longer builds on a current macOS runner. The
recommendation is B.** Everything below is measured unless it says otherwise; the CUDA arm was run
by the maintainer in a prior session, and the Kaggle T4 was deliberately not used (a maintainer
session held the GPU, and a concurrent job changes the pixels a determinism reading depends on —
`agent_guidance/gpu_harnesses.md`).

The switching machinery this rests on is `algan/taichi_compat.py`: every kernel module imports its
compiler from there, `ALGAN_TAICHI_BACKEND` picks `taichi` or `quadrants`, and a mixed process is
unrepresentable rather than merely discouraged. Both arms of everything below are that variable.

#### Step 1 — `operator""_f` (done, and it is not sufficient)

`taichi_patches/0003-literal-operator-whitespace.patch`: 20 declarations across
`taichi/common/core.h` and `taichi/common/types.h`, verified applying cleanly onto a pristine v1.7.4
after 0001 and 0002. The diagnostic and clang's suggested spelling were reproduced locally on
clang 18 with `-Wdeprecated-literal-operator` forced on.

**It works and it is not enough.** On the macOS 26 build below, `-Wdeprecated-literal-operator`
never appears — and the build then dies on a *different* `-Werror` diagnostic:

    taichi/math/linalg.h:245:12: error: first argument in call to 'memcpy' is a pointer to
    non-trivially copyable type 'taichi::VectorND<3, float, taichi::InstSetExt::None>'
    [-Werror,-Wnontrivial-memcall]

Four occurrences, one file, at object 15 of 589 — so a `0004` would be small, but the build stopped
too early to say it is the last one. **Taichi 1.7.4 + Algan's patches does not currently produce a
wheel on `macos-latest`.** The `llvm@15` bottle, which was the predicted stopping point (no
`arm64_sequoia`/`arm64_tahoe` bottle in homebrew-core), poured cleanly in 60 s — that risk is real
but has not arrived yet.

#### Step 2 — the macOS build, both bases, one runner image

Run through `run_on_mac.yaml` on `macos-latest` (macOS 26, Apple clang 21.0.0), `mac-cpu`, `-j3`;
`scripts/gate/{quadrants,taichi}_macos_build.sh` are what they ran.

| | Quadrants v1.3.0 ([run 33822498217](https://github.com/algorithmicsimplicity/algan/actions/runs/33822498217)) | Taichi v1.7.4 + patches 0001-0003 ([run 33822503269](https://github.com/algorithmicsimplicity/algan/actions/runs/33822503269)) |
|---|---|---|
| verdict | **PASS** | **FAIL** (`phase=build`) |
| clone / submodules | 2 s / 60 s | 63 s |
| toolchain (`brew llvm@22` / `llvm@15`) | 60 s | 60 s |
| cold build | 601 s | 181 s, to the error |
| total | 748 s | 312 s |
| wheel | `quadrants-1.3.0-cp311-cp311-macosx_13_0_arm64.whl`, 23.3 MB | none |
| smoke test | `cpu ok`, **`metal ok`** | not reached |
| distinct `-W` diagnostics | **none at all** | `-Wnontrivial-memcall` ×4 |

So the toolchain story is as clean as Quadrants' scripts imply, which is what this experiment
existed to answer. Two structural reasons, both verified in their tree rather than inferred: LLVM is
an org-hosted prebuilt 22.1.0 archive (nothing builds LLVM), and `entry.py`'s
`setup_clang(as_compiler=False)` leaves `CMAKE_C/CXX_COMPILER` on Xcode clang while
`CLANG_EXECUTABLE` stays on brew clang-22 — the same split `taichi_build.yaml` had to discover by
hand after `ld: library 'System' not found`, except Quadrants does it itself.

Caveat on precision, not on direction: the two jobs drew slightly different images (26.6.2/cmake
4.4.3 against 26.5.2/cmake 4.4.0) despite both asking for `macos-latest`. Apple clang was identical.

#### Step 3 — CPU pixel parity and what it costs (Linux x64, the number that was to decide it)

Not "≤2 and explainable" — **identical**. `tests/fast` under `ALGAN_TAICHI_BACKEND=quadrants`
produced an mp4 **byte-identical to the committed baseline** (md5 `7d382c56588a3bbb2dc612b609e868e7`,
182,938 bytes): 0 of 37,635,840 channel samples differ across 45 frames of 704×396. A second,
independent `save_frame` of a `Square` in separate processes per backend: 0 of 836,352. The
maintainer's Windows CUDA render was bit-identical too.

> **Corrected 2026-09-04, and the correction is the more important number.** This section first
> concluded from the above that "LLVM 15 → 22 costs no re-baseline on x86-64". **That is false for
> the dense scenes.** Running the *full* suite on both backends on one box — which the gate never
> did, it only ran `--fast` — puts Taichi at **2964 passed, 0 failed**, so those baselines do travel
> here, and puts Quadrants at four `tests/full_renders` scenes over tolerance:
>
> | scene | max channel delta (tolerance 2) |
> | --- | --- |
> | `complex_hierarchy_become` | 3 |
> | `materials_and_lighting` | 12 |
> | `solids_and_camera` | **100** |
> | `text_and_media` | **158** |
>
> The two that pass are exactly the two `tests/full_renders/test_full_renders.py:76-92` names as
> portable, and the four that fail are the ones carrying PN surfaces, shadows, refraction or glTF —
> which is what that comment's own mechanism predicts: `fast_math` flips borderline tessellation
> levels, and *which* levels are borderline is a property of the arithmetic. A different LLVM is a
> different arithmetic in exactly the way a different CPU is, and the magnitudes land in the same
> 29-204 band it measured across machines.
>
> So the honest statement is: **`tests/fast` and one CUDA render are byte-identical; the PN-heavy
> full renders are not, and Track B carries a real `tests/full_renders` re-baseline on CPU**,
> inspected scene by scene, plus the release-asset repackaging `tests/README.md` requires. §7.3
> step 6 already asked for that; what changed is that it is now known to be necessary rather than
> precautionary. Apple Silicon remains untested.

`uv run -m pytest -q --fast`: **526 passed on both backends**, zero failures in `algan/`. The whole
engine ran unmodified. Two test files imported the compiler directly and needed rerouting through
`taichi_compat` (`test_mps_friendly.py`, `test_ux_regressions.py`); one of them genuinely failed
(`taichi_accumulate_dtype() is ti.f64` comparing a Quadrants `DataTypeCxx` against a Taichi
`DataType`) and the other was quietly decorating a `@ti.func` with Taichi inside a Quadrants
process.

**The cost is 2.1× warm wall time, and it is one diagnosed problem.** Single `save_frame`, 22
kernels, `ALGAN_LOG_TAICHI_COMPILES=1`:

| arm | cold | warm | warm frontend | warm backend |
|---|---|---|---|---|
| taichi 1.7.4 + warmstart + fast_launch (shipped default) | 17.7 s | **5.7 s** | 3.66 s | 0.27 s |
| taichi 1.7.4, both off | 24.7 s | 12.7 s | 10.68 s | 0.27 s |
| quadrants 1.3.0 (both no-op there) | 40.2 s | **27.0 s** | 25.32 s | 0.24 s |

The offline cache is a dead heat (0.24 s vs 0.27 s warm); the entire gap is frontend. cProfile puts
`get_pos_info` at 478,598 calls / 25.95 s = **40.5 % of the profiled render**, and **every** frontend
counter is exactly **2.00×** Taichi's (`get_tree_and_ctx` 11,036 vs 5,518; `build_Name` 158,074 vs
79,037) — §7.3 item 3's "builds each kernel AST twice" reproduces on CPU. Warm `--fast` goes
49 s → 101 s.

Both halves are addressable and neither is a correctness or pixel risk: Algan already carries the
`get_pos_info` memo in `taichi_warmstart.py` and version-gates it to Taichi 1.7, and the doubled AST
build is untouched by that memo. See the release finding in the corrections below — **no Quadrants
release contains the upstream memo**, so this is port-or-pin, not "upgrade later".

> **Since measured: the port landed, and it halves this.** The maintainer chose Quadrants pinned to
> public releases, so `taichi_warmstart.py` now patches both compilers rather than version-gating
> itself off on one. Same box, same scene, warm, 22 kernels
> (`benchmarks/_taichi_warmstart_check.py`): Quadrants **29.0 s → 12.5 s** (2.32×), Taichi
> **15.1 s → 7.2 s** (2.11×, unchanged behaviour); `--fast` on Quadrants **106 s → 78 s**. Frames
> are byte-identical across all three arms and across both compilers, with
> `ALGAN_TAICHI_WARMSTART_VERIFY=1` recomputing every memoized value the original way inside a real
> materialization. The residual ~1.7× against Taichi is the doubled AST build, which is untouched
> and is now the only known frontend item left.

New and unexplained, worth a look before adopting: every render prints, twice,
`UserWarning: cannot create weak reference to 'DataTypeCxx' object. Template mapper caching
disabled.` Quadrants is disabling its own template-mapper cache for Algan's kernel arguments.

#### Step 4 — the three upstream repros

`benchmarks/_upstream_repro_874{4,5}.py` and `_upstream_repro_8794.py`; each honours
`REPRO_BACKEND` and `REPRO_ARCH` and prints one verdict line. None of the three was filed by this
project; all three are open, uncommented, with no linked PR, and no Quadrants commit cites any.

| issue | Taichi 1.7.4 | Quadrants 1.3.0 | where |
|---|---|---|---|
| #8744 dead-branch miscompile | REPRODUCES | REPRODUCES | Linux x64 (not CUDA-specific, and not `@ti.func`-specific) |
| #8794 segfault at iteration 512 | REPRODUCES | REPRODUCES | Linux x64 (not Metal at all) |
| #8745 Metal field-shape dependence | **REPRODUCES** | **CLEAN** | Mac runner, real Apple GPU, macOS 26.5.2 |

- **#8744** is bounded, and the bound is already Algan's shipped config: `cfg_optimization=False`
  alone fixes it, `advanced_optimization=False` fixes it, and Algan runs with the latter off. The
  symptom `agent_guidance/taichi.md:11` documents for turning it on — `pbr_neutral_tonemap` losing
  the rescale inside its compression branch — is the same class of failure, so #8744 is very likely
  Algan's own known miscompile with a public repro attached. Row 17's "then re-test `ALGAN_ADV_OPT=1`"
  stays blocked on **both** bases.
- **#8794** is root-caused: `kMaxNumSnodeTreesLlvm` is 512, nothing bounds `tree_id` against it, and
  the field declared after `Ptr roots[512]` / `root_mem_sizes[512]` is `Ptr thread_pool` — so
  materializing tree 512 overwrites the pointer the next parallel launch jumps through. Algan
  declares no `ti.field` and cannot reach it.
- **#8745 is the one result that separates the bases**, and it separates them in B's favour. The
  named candidate is `7a9b6cb23` (#384), which decorates every SPIR-V storage buffer `Volatile` on
  Metal because the Metal compiler hoists storage-buffer loads out of loops and serves stale reads
  when a buffer is written and re-read in one loop. Algan's ndarrays are storage buffers on that
  same SPIR-V→MSL path, so this is a live correctness exposure on the Apple path today: §5's
  "copy if" for #384 becomes a **must-copy under Track A**, and comes free under B.

#### Step 5 — CUDA

Covered by the maintainer's own run in a prior session: Quadrants rendered successfully and
**pixel-identical on Windows CUDA**. The T4 was not used, per above. What remains unmeasured on CUDA
is a warm-timing split (the frontend regression in step 3 should appear there too, since it is
Python-side) and any pixel reading on Apple Silicon.

#### Verdict, and what B costs that §6 did not price

Criterion by criterion: macOS build **green for B, red for A**; pixel deltas **zero on the criterion
the gate names** (`tests/fast`, byte-identical) but **up to 158 channel values on four of the six
dense full-render scenes**, which the gate did not run and which the correction above measures — so
the criterion passes and the re-baseline it was meant to rule out is real after all; the three
repros **do not distinguish the bases except in B's favour**; and
Python 3.9 / macOS < 13 is met on every piece of evidence in the repository — Intel Macs are
*refuted* as a loss (Taichi 1.7.4 has never published a macOS x86_64 wheel for any Python), Quadrants
*adds* manylinux aarch64 which Taichi never shipped, the only genuine loss is cp39 on Linux-x86_64
and Windows, and `algan` is not on PyPI yet (live 404), so no one can have installed it through the
documented channel. **The maintainer should still confirm no early adopter is on 3.9**; that is the
one question the repository cannot answer about itself.

Three costs the gate found that §6's "what B costs" list did not:

1. **The frontend tax is real, and no Quadrants *release* fixes it** (2.1× warm; port Algan's memo
   or pin a commit; the 2× AST build remains after that).
2. **Track A is not free either.** It has stopped building on the current runner image, needs a
   `0004` for `-Wnontrivial-memcall` with no guarantee that is the last, and must copy #384 to be
   correct on Metal. "Stay on 1.7.4" is now itself accruing maintenance against a dormant upstream.
3. **Pre-Volta is unchanged and will not fix itself**: `kernel_atomic_syncscope.h:29-34` is
   byte-identical between v1.3.0 and `main`, nothing in the 34 commits since v1.3.0 touches
   `.sys`-scope atomics or `activemask`, and their GPU CI is a T4 (sm_75) with an `sm70` pytest
   marker, so pre-Volta is never exercised. §7.3's Prerequisite 0 stands exactly as written and
   remains Track B's real blocker on the primary dev box.

#### Corrections this gate makes to §2-§5 and §7.3

- **§7.3 item 2, `gpu_max_reg`**: *wrong for the 1.3.0 wheel.* It raises no `KeyError` —
  `qd.init` builds its accepted-kwarg set from `dir(cfg)` (`quadrants/lang/misc.py:435-443`), so
  every field on `CompileConfig` is accepted and Algan's whole `taichi_init_kwargs()` dict goes
  through verbatim.
- **§7.3 item 2, `get_runtime().prog` → `._prog`**: needs no change in `algan/`. Both sites are
  already inside a bare `try` / `contextlib.suppress` (`taichi_runtime.py:376-379,397-403`). The
  seven unguarded reads are in `tests/unit_tests/test_taichi_runtime_config.py`, outside `--fast`.
- **§7.3 item 3, the memo**: `895dd5ea1` (#858) landed 2026-08-13, two days *after* v1.3.0, and
  `git tag --contains` finds it in **no release tag** (only `archive/po-6-combined-wip`). 1.3.0 is
  still the newest release. "Pin a build that has it" is not available; port it or pin a commit.
- **§7.3 item 5, `QD_WITH_VULKAN=OFF` drops MoltenVK**: true of CMake, false of `build.py`, which
  fetches and runs the LunarG macOS SDK unconditionally on Darwin (`entry.py:65-75`,
  `vulkan.py:41-78`).
- **§2.3, the wheel-size comparison**: their CI turns Vulkan **on** for macOS (`2_build.sh:5`), so
  the shipped 26.7 MB wheel carries MoltenVK. Our own Vulkan-off build is 23.3 MB.
- **Row 30, the pending-launch valve**: keep it on its own merits, but drop the #8794 hypothesis —
  #8794 is the LLVM runtime's snode-tree array, a path the gfx/Metal runtime does not share.
- **§5, `7a9b6cb23` (#384)**: promote from "copy if" to must-copy under Track A (step 4 above).
- Locators: `llvm.py:19-24` → `:22-28`; the `cap >= 60` half2 pattern row 34 says to mirror is in
  `visit(AtomicOpStmt)` at `codegen_cuda.cpp:399`, not inside `optimized_reduction`.

## 7. Implementation plan

### 7.1 Track C — common to both bases (do first)

> **Not taken, 2026-09-04, with one exception.** The base decision and Track B's first steps were
> done instead. The exception is row 47's banner (`ENABLE_QUADRANTS_HEADER_PRINT` beside the Taichi
> one, `algan/__init__.py`), which the flip forced: `import algan` must print nothing to stdout and
> Quadrants prints its version line at import. Everything else below — the six stale cache claims,
> the `gpu_max_reg` docstring, `_inside_class`, `TI_SKIP_VERSION_CHECK`, `cfg_optimization=False`,
> the stale `ticache.lock` rule, `enable_fallback=False`, the eviction comment and the 64-argument
> belief — is untouched and still worth doing on the new base.

Order chosen so that each step is independently mergeable and measurable.

1. **Docs and no-fork wins** (one PR, Algan only): fix the six stale "cache ignores `@ti.func` edits"
   claims and the nested-tuple claim (§2.1, §2.2); retire the `gpu_max_reg` docstring in
   `taichi_runtime.py:18-25` and mark `ALGAN_GPU_MAX_REG` as inert pending item 14; add the
   `_inside_class` monkeypatch to `algan/utils/taichi_warmstart.py` (version-gated, with an
   `ALGAN_TAICHI_WARMSTART_VERIFY`-style check); `print_full_traceback` behind an env flag; set
   `TI_SKIP_VERSION_CHECK` from `algan/environment.py`; try `cfg_optimization=False` (`ti.init`
   kwarg present in 1.7.4) with one process per arm and record compile time and pixels; clear a stale
   `ticache.lock` in `init_taichi` when older than a threshold and no live process holds it; pass the
   concrete arch with `enable_fallback=False` (row 46); fix the eviction comment at
   `taichi_runtime.py:611-612` (row 9) and the 64-argument belief (row 43); optionally pin
   `CUDA_CACHE_PATH` (row 51). Verify: `--fast`, then the full suite.
2. **Source-keyed cache index** (item 1): C++ pybinds (`set_kernel_key_for_cache`,
   `get_cached_kernel_key`, `DeviceCapabilityConfig.hashed_key`, and store the IR hash `ir_key` in
   the artifact metadata); Python `algan/utils/taichi_source_key.py`: L1 = kernel source lines hash ∥
   `taichi` version+commit ∥ config hash ∥ caps hash ∥ file path + start line; L1 value = the list of
   every transitively visited `@ti.func` (file, line range, hash) plus Algan's environment
   fingerprint (a hash over every declared `ALGAN_` variable's value, from `algan/environment.py`) and
   `SETTINGS.raytracing` state; L2 = L1 ∥ per-argument (dtype, ndim, needs_grad, boundary) ∥ template
   values (numbers by value, functions by source hash + closure walk, tuples ordered, dtypes by name,
   `None` by tag, anything else → poison, fail closed with a one-shot warning). On a hit, set the key,
   run the transform with a FunctionDef-only walk (parameters declared from annotations; body
   skipped), let `load_or_compile` find the `.tic`. On a miss, today's path, then persist L1/L2 with
   `mkstemp`+`os.replace`. `ALGAN_TAICHI_SOURCE_KEY_VERIFY=1` runs the full transform on every hit and
   compares with `ir_key`; `ALGAN_TAICHI_SOURCE_KEY=0` kills it. Default off until verify mode is
   clean across `tests/full_renders` on CPU and CUDA. Deletable afterwards: most of `taichi_warmstart.py`.
3. **Codegen from scratch** (items 13, 14, 15, 18), each gated on the CUDA measurements in §8:
   `!invariant.load` + `addrspace(1)`; the `max_reg` one-liner then `loop_config(max_reg=)`;
   readonly loads through the existing intrinsic path; fast `expf`. Clear the cache; run
   `tests/full_renders`; re-baseline only for expected float drift (fast `expf` will move pixels).
4. **Correctness copies** (§5 top group) as small patches, each with its Quadrants test ported.
5. **Metal patches 0003/0004** (items 26, 29) plus the valve (30) and `random_seed` (31); then delete
   the sync pair in `mps_zero_copy.py`. Verify on the Mac probe workflow (`.github/workflows/mps_probe.yaml`),
   including a >600-frame render for #8794.
6. **Shader work** (items 20, 21), Algan-side, after the docs fix.

### 7.2 Track A — if staying on 1.7.4

Add, as numbered patches applied by `taichi_build.yaml` in order, with a README section each:
`core.h` `operator""_f`; PTX cache (`2e5ed0771` + `89987df5e` + `5f926630c`, renaming
`quadrants::`→`taichi::`, `QD_`→`TI_`, and dropping nanobind-only bits); persistent per-handle arg and
result buffers (`4e35bd556`, adapted: 1.7.4 has no `stream_parallel_group_id`, so the
`default_stream_path` predicate reduces to "null stream"); `5a20dbc66`; batched `set_args_*`
pybinds consumed by `taichi_fast_launch.py`; `QD_DUMP_SIMPLIFY`; compile-time wins (`d5ab903d4`,
`b65c5f6df`, `f3ccac394`); the `dump_cache_data_to_disk` pybind; cp314 (`ast.Str` ×3, `pybind11>=3`,
`scikit-build<1`). Build config: `TI_WITH_C_API=OFF`, OpenGL/Vulkan/DX off (Vulkan off also removes
the headless probe crash structurally: `with_vulkan` becomes a constant-false lambda,
`taichi/python/export_misc.cpp:162`); mirror the LLVM 15 archives and the Windows clang zip into
Algan's own release assets; persist `~/.cache/ti-build-cache` with `actions/cache`; add Linux
(`ubuntu-22.04`, manylinux) and Windows (`windows-2022`) legs; Apache §4(b) notice lines on every
patched file. Keep patches 0001 and 0002 unchanged.

### 7.3 Track B — if rebasing onto Quadrants

> **Taken, 2026-09-04. Steps 1, 2 and 4 are done; 3, 5, 6 and 7 are not.** Quadrants is the default
> compiler (`quadrants>=1.3.0,<1.4`, `BACKENDS[0]`), `quadrants_patches/` carries three patches,
> and Prerequisite 0 is written as `0003` — compiled, never run on pre-Volta hardware. The per-step
> markers below say what closed each one and what it cost; `MIGRATION.md` is the whole account,
> including two defects in the ported Metal patch that only a real Apple GPU could have found.
>
> The one step whose shape changed: **step 3 is not "rewrite `taichi_fast_launch.py`"** any more.
> The warm-start memoization was ported instead (`taichi_warmstart.py`, both compilers, 2.3× on a
> warm render) because no Quadrants *release* carries the upstream `get_pos_info` memo. Fast-launch
> remains version-gated to Taichi 1.7 and no-ops on Quadrants, which is a cost this migration has
> not paid down and step 3 still owns.

**Prerequisite 0 — pre-Volta CUDA support. Quadrants 1.3.0 cannot `qd.init(qd.gpu)` on a GPU older
than sm_70, which includes this repository's development machine (GTX 1050, sm_61).** Measured
2026-09-03 against the PyPI wheel 1.3.0 (commit `ab9a58ab`, LLVM 22.1.0), driver 576.52. This is
not a general Pascal gap — Quadrants detects the device and emits correct `.version 5.0 /
.target sm_61` PTX — but two unrelated, small defects, both fixable in ~4 lines:

* **(a) `.sys`-scope atomics.** `qd.init` dies with `CUDA_ERROR_NOT_SUPPORTED ... cuModuleLoadDataEx`
  loading the *runtime* module, because of one instruction: `atom.sys.cas.b64`, emitted for the
  `__atomic_compare_exchange_n` in `runtime_eval_adstack_max_reduce`
  (`quadrants/runtime/llvm/runtime_module/adstack_runtime.cpp`). Its default (system) memory scope is
  lowered to `.sys` by LLVM 22's NVPTX backend; LLVM 15 did not, which is why Taichi 1.7.4 is
  unaffected. Module load is all-or-nothing, so a function Algan never calls takes the whole runtime
  down. **Probed directly against the driver: `.sys` is the only rejected scope — `atom.gpu`,
  `atom.cta` and unscoped `atom` all load on sm_61.** Fix: give that cmpxchg `syncscope("device")`,
  and change `quadrants/runtime/llvm/kernel_atomic_syncscope.h:29` (returns `SyncScope::System` for
  CUDA, `"agent"` only for AMDGPU) to `"device"`, at least for `cap < 70` — kernel-side CAS atomics
  (i64 max, f32 min) hit the same wall. That header is already shared by both emit sites
  (`codegen_llvm.cpp`, `llvm_context.cpp`), so it is one line.
* **(b) `llvm.nvvm.activemask`.** Kernels then fail to compile with `LLVM Fatal Error: Cannot select:
  intrinsic %llvm.nvvm.activemask`, reached from `codegen_cuda.cpp:369 optimized_reduction`
  (warp-aggregated reductions) via `llvm_context.cpp:441`'s
  `patch_intrinsic("cuda_active_mask", Intrinsic::nvvm_activemask)`. The *hardware* is fine —
  `activemask.b32` is sm_30+ — it is LLVM 22's NVPTX instruction selection that is gated to sm_70+.
  Fix: gate `optimized_reduction` on `cap >= 70` and fall back to plain atomics, mirroring the
  `cap >= 60` half2 check 30 lines below it in the same function.

**Evidence these are the only two:** with (a) emulated by an inline hook on `cuModuleLoadDataEx` that
rewrites `.sys` out of the PTX, and (b) emulated by `make_thread_local=False`, Algan renders a
90-frame video and a frame on Quadrants on this GPU, bit-identical to the Taichi arm
(0 of 419,904 pixels differ). Five `.sys` instructions are patched per render: one in the runtime
module, four in the wavefront megakernels.

**That hook (`debug/_qd/hook.py`) is a diagnostic, not a solution.** It rewrites the PTX in flight by
inline-patching `nvcuda.dll`'s `cuModuleLoadDataEx` with a jump to a Python callback: Windows-only,
x86-64-only, unsafe against concurrent module loads, and silently dependent on the exact instruction
text and on Quadrants continuing to load modules through that one entry point. It is the right tool
for answering "is this the only blocker" and the wrong one for shipping. **Fix it in the compiler.**
Prefer upstreaming all three edits over carrying them: each is defensible on its own merits (device
scope is more correct *and* faster than system scope; the cc gate copies a pattern Quadrants already
applies to `match_any_sync` and half2), none touches a non-sm_61 path, and a permanent fork delta is
a rebase tax for nothing. Fork only as a stopgap while an upstream PR lands. If neither happens,
Track B is blocked on any pre-Volta machine, which today includes the primary dev box.

> **Written 2026-09-04 as `quadrants_patches/0003-pre-volta-cuda.patch`, and writing it corrected
> the diagnosis above in two load-bearing ways** (`quadrants_patches/PORTING-NOTES.md` §7 has the
> evidence; the patch is **unbuilt and unrun** — `scripts/gate/quadrants_linux_build.sh` is the
> compile check, since Quadrants forces CUDA off on Apple and the macOS leg never compiles a line of
> it, and only the maintainer's sm_61 box can answer whether it actually works):
>
> * **The source-level fix for (a) does not work.** `__scoped_atomic_compare_exchange_n(...,
>   __MEMORY_SCOPE_DEVICE)` emits IR byte-identical to the unscoped builtin — no `syncscope` at all —
>   because only the AMDGPU, NVPTX and SPIR-V backends override
>   `TargetCodeGenInfo::getLLVMSyncScopeID`, and the runtime bitcode is compiled with no `-target`
>   (`runtime_module/CMakeLists.txt`). The patch therefore re-scopes those atomics **in IR**, in the
>   CUDA branch of `module_from_file` in `llvm_context.cpp` — the file that already exists to fix up
>   host-compiled bitcode for the GPU target. Only one `.sys` survives into the loaded module because
>   `init_runtime_module` runs `eliminate_unused_functions` keeping `runtime_*`/`LLVMRuntime_*`
>   (`llvm_context.cpp:1158`): `runtime_eval_adstack_max_reduce` matches, and `stack_push` — which
>   holds deliberately system-scoped overflow-flag atomics — is dropped, which is what makes the
>   narrow rewrite both sufficient and safe.
> * **The gate for (b) is `cap >= 75`, not 70, and sm_70/sm_72 are broken today too.** LLVM 22's
>   predicate on `activemask` is `Requires<[hasPTX<62>, hasSM<30>]>`: the hardware bound is sm_30 and
>   Pascal meets it — what fails is the *PTX ISA version*. Quadrants passes an empty feature string to
>   `createTargetMachine` (`jit_cuda.cpp:265`), so LLVM defaults `PTXVersion` to
>   `getMinPTXVersionForSM`, which is 5.0 for sm_6x, 6.0 for sm_70, 6.1 for sm_72 and 6.3 for sm_75.
>   The `.version 5.0 / .target sm_61` observed above is direct confirmation. A `cap >= 70` gate would
>   have left a V100 exactly as broken; gating at 75 is still a no-op from Turing up, including
>   Quadrants' own T4 CI.
>
> One caveat on attribution, since it bears on how the fix should be argued upstream: the `.sys`
> refusal was measured on a single sm_61 **Windows** box, so whether the discriminator is compute
> capability, WDDM, or `hostNativeAtomicSupported` is unestablished. `cap < 70` is safe either way.

1. **[done]** Fork `Genesis-Embodied-AI/quadrants` at a tag; new `quadrants_patches/` directory; port 0001
   (all 11 target files exist at renamed paths; `MetalDevice::import_mtl_buffer` still at
   `quadrants/rhi/metal/metal_device.mm:1280`; `MetalShaderResourceSet::rw_buffer` still honours
   `ptr.offset`, `:292-305`; the one `export_lang.cpp` binding becomes nanobind; the gfx bind site
   moved to `runtime/gfx/runtime.cpp:698`); port 0002 minus the cast hunk; add the `ContinueStmt`
   test.
2. **[done, and two of the thirteen did not apply]** Python glue: `import quadrants as ti` (84 sites);
   `get_runtime().prog` → `._prog` (the property now raises when unset, `lang/impl.py:443-447`;
   `taichi_runtime.py:376,400`, seven sites in `tests/unit_tests/test_taichi_runtime_config.py`);
   delete `gpu_max_reg` from `taichi_init_kwargs` (a `KeyError` in Quadrants); `TI_OFFLINE_CACHE_FILE_PATH`
   → `QD_OFFLINE_CACHE_FILE_PATH` (`environment.py:52`, `taichi_runtime.py:623`);
   `kernel.compiled_kernels` → `materialized_kernels`/`compiled_kernel_data_by_key`;
   `taichi.lang.kernel_impl.Kernel` → `quadrants.lang.kernel.Kernel`; `mapper.lookup(args)` →
   `lookup(raise_on_templated_floats, py_args)`; `_get_tree_and_ctx` gone (now inside
   `_func_base.py:390-470`); `ASTTransformerContext` → `ASTTransformerFuncContext`; exception class
   renames (comments only); `pyproject.toml`: `quadrants>=1.3.0`, `requires-python>=3.10,<3.14`;
   **set `QD_KERNEL_COVERAGE=0` in `test.yaml` before `--cov`** (Quadrants' pytest plugin otherwise
   instruments every kernel under pytest-cov, changing memory layout); `prog.compile_kernel` returns
   `CompileResult` (take `.compiled_kernel_data`).
3. **[not done — superseded in part; see the note above]** Rewrite `taichi_fast_launch.py` against `quadrants.lang.kernel.Kernel` (prefer hooking
   `launch_kernel` over `__call__`, which now carries checkpoint/`qd.Tensor`/stream stages); re-run
   `benchmarks/_taichi_fast_launch_check.py` with verify on. Trim `taichi_warmstart.py` to the
   source-retrieval memo (`get_pos_info` is memoised upstream, `ast_transformer_utils.py:408-424`)
   — **but only on a build that has it.** Verified 2026-09-04: that memo is on `main`
   (`get_pos_info` at `:417`, with a comment giving the same reasoning as Algan's patch) and is
   **absent from the released PyPI wheel 1.3.0** (commit `ab9a58ab`), where `get_pos_info` still
   builds a `TextWrapper`-wrapped source excerpt per node. Measured on 1.3.0: **617,152 calls,
   67.3 s, ~41% of the profiled frontend time, versus 0 calls on Taichi** — because
   `taichi_warmstart.apply()` version-gates to `(1, 7)` (`taichi_warmstart.py:70`) and silently
   no-ops elsewhere. Consequence: a Track B build pinned to 1.3.0 pays a frontend cost Algan
   already solved on Taichi (measured whole-process kernel materialize: ~107 s vs ~30 s for the
   same 27 kernels). Either pin a Quadrants build that contains the upstream memo, or port the
   memo rather than trimming it. **Done, by porting** (2026-09-04): pinning was not on offer —
   the memo is in no release tag (§6.1) — so `taichi_warmstart.py` now installs against either
   compiler, memoizing `get_pos_info` and, on Quadrants, `get_source_info_and_src` and the
   per-line `textwrap.fill` behind `get_tree_and_ctx`. It stands down by itself if a future
   release carries upstream's own memo (`_build_pos_info`), and `algan check` now reports the
   version gate refusing to fire rather than leaving it silent. Quadrants additionally builds each kernel AST **twice**
   (`get_tree_and_ctx` 2.00x per kernel — a pruning pass then an enforcing pass; the first is
   skipped only on a fastcache hit), which the memo does not address: node visits measured at
   505,978 vs Taichi's 252,989, `build_Name` 203,288 vs 101,644.
4. **[done — the module needed no change; `taichi_compat` absorbed it]** `mps_zero_copy.py`: replace the per-launch sync pair with
   `qd.init(external_metal_command_queue=quadrants.interop.get_mps_command_queue(), external_metal_command_queue_is_torch_queue=True)`.
5. **[not done — `scripts/gate/*_build.sh` build wheels on demand, but no release workflow ships one]** CI: rewrite `taichi_build.yaml` from `quadrants-src/.github/workflows/scripts_new/`
   (macOS, manylinux x86_64/aarch64, Windows); `QD_WITH_VULKAN=OFF` on macOS drops the MoltenVK
   dependency; enable `QD_WITH_CUDA=ON` on Linux/Windows.
6. **[outstanding, and now known to be necessary — §6.1's correction: four scenes move by up to 158 channel values]** Re-baseline `tests/fast` and `tests/full_renders` with the diffs inspected; record warm
   `save_frame` timings before/after; delete `ALGAN_GPU_MAX_REG`.
7. **[not started]** Then Track C items on the new base: item 1 reuses `Program.load_fast_cache` and the
   FunctionDef-only path instead of new pybinds; items 13–15, 18, 20 unchanged; item 2 and the
   Metal copies are already present.

## 8. Verification and measurements to record

- CUDA box, before Track C step 3: `ti.init(print_kernel_asm=True)` + `ptxas -v` on
  `sheet_resolve_shade`, `wavefront_shade`, `pt_shade`, `wavefront_traverse_events`, with and without
  `ALGAN_GPU_MAX_REG` (expect no change), then with the one-line fix; PTX dump after `!invariant.load`
  to confirm argument `ld.u64/ld.u32` have left the loop body; warm alternating A/B on a
  `benchmarks/_arena_view_real_kernel_ab.py`-class capture.
- CUDA box, before Track A's PTX cache: `TI_LOG_LEVEL=trace` and a timer around
  `compile_module_to_ptx` to split the 7.0 s into parse / O3 / NVPTX / `cuModuleLoadDataEx`, with the
  4 GiB driver cache in effect.
- Every kernel-IR-affecting patch: clear the offline cache, `pytest -q tests/full_renders`, inspect
  diffs, re-baseline only when expected.
- Item 1: `ALGAN_TAICHI_SOURCE_KEY_VERIFY=1` clean across the full render suite on CPU and CUDA;
  count poisons on user-authored shader stages; key cost < 0.1 s per process.
- Metal: the Mac probe workflow after each of 0003/0004; a >600-frame render; pixel parity of the
  compositing atomics against the CPU reference.
- Record in `agent_guidance/taichi.md`: the CUDA backend split, ptxas register counts, warm
  `save_frame` before/after each cache patch, and the base decision with its gate results.

## 9. Open questions

- Pixel-delta magnitude of LLVM 15 → 22 (the number that decides §6).
- Whether #8744 / #8745 / #8794 reproduce on Quadrants (no commit cites them).
- Whether LLVM 15's LICM hoists `!invariant.load` argument loads into the grid-stride preheader or
  only CSEs them, and the register-pressure effect on the 161-register megakernel.
- Static-dependency-walk poison rate on user-authored `@ti.func` stages (item 1).
- Whether the Quadrants per-task artifact cache stabilises (eviction, tests) enough to adopt later;
  it is the strongest idea in the tree for a project that edits megakernels daily.
- Minimum NVIDIA driver implied by LLVM 22's PTX ISA version (unverified; matters for Windows/Linux
  CUDA users on old drivers under Track B).
- Whether any Algan user needs Python 3.9 or macOS < 13.

## Appendix A. Quadrants commits referenced (all verified to exist at `b9e953111`)

| sha | PR | date | subject |
|---|---|---|---|
| `20ce619f0` | #131 | 2025-08-15 | Add SRC-LL caching (fastcache origin) |
| `b9189e5d1` | #283 | 2025-11-24 | Fastcache key reuses the C++ frontend cache key |
| `1fbfe5523` | #286 | 2025-11-26 | file+lineno in the fastcache key |
| `590024293` | #289 | 2025-11-26 | version in the fastcache key |
| `47e9dfb5b` | #809 | 2026-07-23 | pruning fixes (dead static branch, swapped slots) |
| `39f20102a` | #850 | 2026-08-14 | exclude non-codegen config keys; caps hashed_key |
| `fa3b8a944` | #705 | 2026-08-31 | fastcache L1/L2 split, pruning union |
| `2e5ed0771` | #130 | 2025-08-22 | PTX cache |
| `89987df5e` | #361 | 2026-02-11 | SM version in PTX cache key |
| `5f926630c` | #580 | 2026-04-28 | honour `offline_cache=False` end-to-end |
| `81ed162ac` | #880 | 2026-08-27 | split frontend per top-level construct |
| `bdb9b49fe` | #875 | 2026-08-20 | per-task CUDA modules via composite JITModule |
| `42cc5f74a` | #864 | 2026-08-20 | content-keyed global-temp offsets |
| `f6df02c2e` | #893 | 2026-08-31 | cross-process per-task artifact cache (CUDA) |
| `7337f2926` | #324 | 2025-12-19 | early return in materialize |
| `c670394fb` | #325 | 2025-12-19 | memoise CompiledKernelData |
| `895dd5ea1` | #858 | 2026-08-13 | memoise pos-info banner |
| `ec13f405a` | #886 | 2026-08-25 | source capture resilient to linecache misses |
| `5a20dbc66` | #267 | 2025-11-12 | do not copy kernel parameters |
| `4e35bd556` | #619 | 2026-05-04 | persistent launch context/arg buffer, Metal encoder |
| `e8a44de15` | #654 | 2026-05-09 | drop per-launch host syncs |
| `3e45a7a7c` | #890 | 2026-08-26 | remove unused init options incl. `gpu_max_reg` |
| `d5ab903d4` | #577 | 2026-04-27 | whole_kernel_cse 2.5× compile speedup |
| `b65c5f6df` | #579 | 2026-04-28 | skip IR verifier unless debug |
| `f3ccac394` | #729 | 2026-06-16 | cfg pass compile time on multi-task kernels |
| `5f8138df2` | #790 | 2026-08-05 | CSE per offload (do not copy) |
| `78ed263a9` | #376 | 2026-02-20 | loop-invariant caching vs atomic destinations |
| `3d9af7189` | #810/#812 | 2026-07-28 | loop-invariant caching vs may-aliasing buffers |
| `91c590563` | #866 | 2026-08-28 | AMDGPU address-space-at-source (pattern for item 13) |
| `47494eadf` | #702 | 2026-05-18 | `qd.volatile_load` (per-load flag plumbing) |
| `7c704d987` | #884 | 2026-08-25 | `QD_DUMP_CFG` no longer changes behaviour |
| `fe3cd8908` | #853 | 2026-08-14 | caches no longer defeat IR-dump env vars |
| `b44f0e279` | #788 | 2026-08-06 | Metal native float atomics |
| `718bb695e` | #618 | 2026-05-05 | external Metal command queue |
| `64d9ea240` | #694 | 2026-05-13 | FIFO ordering when sharing the queue |
| `ae2d1c0b5` | #551 | 2026-04-24 | pending-launch valve (with the MoltenVK SDK change) |
| `989e4ae4a` | #786 | 2026-07-16 | respect `random_seed` on SPIR-V |
| `eff73c380` | #392 | 2026-03-03 | Metal physical storage buffers (>31 ndarrays) |
| `97add824e` | #846 | 2026-08-13 | partial writes to a torch zero tensor on Metal |
| `9542c0004` | #543 | 2026-04-25 | contains the OpBitcast(T,T) cast fix |
| `13dcddde7` | #794 | 2026-07-22 | macOS minimum 13.0 |
| `685feb23b` | #759 | 2026-06-25 | migrate to nanobind |
| `ce05ab150` | #747 | 2026-06-22 | scikit-build-core |
| `73a9bd7b4` | #394 | 2026-03-13 | LLVM 22 |
| `aada31022` | #405 | 2026-03-16 | CUDA graphs (unconditional) |
| `e3c40d165` | #419 | 2026-04-22 | abort on assertion instead of segfault |
| `a747b4eb3` | #451/#453 | 2026-04-25 | raise on device allocation failure |
| `e98b7a91b` | #442 | 2026-04-01 | shared-memory offset reset between kernels |
| `9fd62652f` | #391 | 2026-03-02 | shared array name collision |
| `41b5086f1` | #513 | 2026-04-24 | Vulkan SPIR-V correctness incl. narrow storage caps |
| `79ec04903` | #432 | 2026-04-09 | SPIR-V atomics parity |
| `4331f125a` | #489 | 2026-04-16 | SPIR-V ID overflow on large kernels |
| `ea7d24f14` | #372 | 2026-02-16 | `serialize=False` on Metal |
| `7a9b6cb23` | #384 | 2026-02-25 | Metal SPIR-V miscompile |
| `8b8f12c72` | #440 | 2026-04-01 | named top-level loops |
| `a7c46c898` | #309 | 2025-12-02 | parallel cache write fix |
| `9f4cb3577` | #375 | 2026-02-20 | lock around kernel compile |
| `46df9384c` | #395 | 2026-03-02 | evict stale id-cache entries on GC |
| `981ef70b9` | #278 | 2025-11-20 | ndarray memory leak |
| `cd19c293c` | #172 | 2025-09-08 | ndarray crash after reset |
| `ab9a58ab5` | #847 | 2026-08-11 | device pool alignment (also tag `v1.3.0`) |
| `5b56bdd02` | #668 | 2026-05-09 | persistent rand-state buffer |
| `f1841afcf` | #669 | 2026-05-09 | trim mempool on reset |
