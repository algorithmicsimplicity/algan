# Taichi fork roadmap: what to take from Quadrants, what to build, and on which base

Status: **plan, not implemented.** Written 2026-09-03 from a read-only survey of Taichi v1.7.4, the
Quadrants fork at `b9e953111` (2026-09-02), and this repository. It is self-contained: a fresh session
can execute it from §0 without any other context. Every claim carries the file, commit or measurement
it rests on; "verified" means read in source, "measured" means run on the stated machine,
"projected" means an estimate.

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
| 8 | `_inside_class` import cost; lazy pos-info | `get_pos_info` memoised (#858 `895dd5ea1` — same fix Algan's warmstart has); `_inside_class` untouched | **scratch**: 5-line monkeypatch in `taichi_warmstart.py` (−1.1 s per process, measured) | `python/taichi/lang/kernel_impl.py:1067-1076` |
| 9 | Cache hit/miss stats API; mid-run cache flush; stale lock; int64 cap | stats: rich `CompileResult` + observation dataclasses (`_kernel_types.py:16-62`); flush: `Program.dump_cache_data_to_disk` pybound (`export_lang.cpp:488`); lock and cap unchanged | **copy** the flush pybind (1 line) and the stats idea (~30 lines); **scratch** the stale-lock rule (Algan-side, ~15 lines in `init_taichi`); cap **skip** | Algan then deletes the fd-2 stderr hijack in `taichi_runtime.py:141-204` |
| 10 | Launch-path cost (`taichi_fast_launch` residue) | extensive Python work, but `LaunchContextBufferCache` marks raw torch tensors non-cacheable (`_func_base.py:858-874`) so it would never engage for Algan | **keep** `taichi_fast_launch.py`; **copy** batched `set_args_*` pybinds (design: `export_lang.cpp:686-699`, `kernel.py:822-834`), `5a20dbc66` (#267, don't copy kernel params, 12 lines), and the persistent per-handle arg/result buffers from `4e35bd556` (#619; `kernel_launcher.cpp:365-377,463-480,533-535,579-592`, ~60 lines) | Consider `e8a44de15` (#654, drop per-launch `stream_synchronize`) after measuring |
| 11 | Two resident Programs (Q2) | absent; singleton hardened | **skip** | run `benchmarks/_taichi_arch_coexistence_probe.py` on a CUDA box only if a second prep kernel ever clears an A/B |
| 12 | AoT Python loader (Q3) | deleted | **skip** | — |
| 13 | `!invariant.load` on argument loads (Q4) | absent (only read-only SNodes, `codegen_cuda.cpp:583-616`) | **scratch** | helper at `codegen_llvm.cpp` sites 1896, 1927, 2925, 2944, 2961 + `!dereferenceable(args_size)`; borrow the `addrspace(1)` tagging pattern from `91c590563` (#866, AMDGPU) for NVPTX `ld.global`; ~35 lines |
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
| 33 | Slim wheel, org-hosted toolchain, `operator""_f`, phone-home, Apache notices (Q6) | Quadrants is the model: C API/OpenGL/DX gone, LLVM in `quadrants-sdk-builds`, `core.h:170` fixed, `_version_check.py` deleted | **copy** the `core.h` fix and the CMake option shape; **scratch** the rest on 1.7.4 | `taichi_build.yaml` |
| 34 | LLVM 20/22 for native sm_89/90/120 | done (`73a9bd7b4` #394; clamp 86→121) | **not portable to LLVM 15**; comes free with a rebase | — |
| 35 | CUDA graphs | done, mature, one graph per `@qd.kernel` (`graph_manager.cpp`) | **skip** (granularity does not fit Algan's per-stage kernels) | — |
| 36 | `BufferView`, `qd.Tensor`, tiles, subgroups, streams, AMDGPU | done | **skip** for Algan (BufferView is the arena convention as a type, over Quadrants-owned memory only) | — |

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
days), then choose. Default to B if the gate passes.**

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

## 7. Implementation plan

### 7.1 Track C — common to both bases (do first)

Order chosen so that each step is independently mergeable and measurable.

1. **Docs and no-fork wins** (one PR, Algan only): fix the six stale "cache ignores `@ti.func` edits"
   claims and the nested-tuple claim (§2.1, §2.2); retire the `gpu_max_reg` docstring in
   `taichi_runtime.py:18-25` and mark `ALGAN_GPU_MAX_REG` as inert pending item 14; add the
   `_inside_class` monkeypatch to `algan/utils/taichi_warmstart.py` (version-gated, with an
   `ALGAN_TAICHI_WARMSTART_VERIFY`-style check); `print_full_traceback` behind an env flag; set
   `TI_SKIP_VERSION_CHECK` from `algan/environment.py`; try `cfg_optimization=False` (`ti.init`
   kwarg present in 1.7.4) with one process per arm and record compile time and pixels; clear a stale
   `ticache.lock` in `init_taichi` when older than a threshold and no live process holds it.
   Verify: `--fast`, then the full suite.
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

1. Fork `Genesis-Embodied-AI/quadrants` at a tag; new `quadrants_patches/` directory; port 0001
   (all 11 target files exist at renamed paths; `MetalDevice::import_mtl_buffer` still at
   `quadrants/rhi/metal/metal_device.mm:1280`; `MetalShaderResourceSet::rw_buffer` still honours
   `ptr.offset`, `:292-305`; the one `export_lang.cpp` binding becomes nanobind; the gfx bind site
   moved to `runtime/gfx/runtime.cpp:698`); port 0002 minus the cast hunk; add the `ContinueStmt`
   test.
2. Python glue, the 13 breaking differences: `import quadrants as ti` (84 sites);
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
3. Rewrite `taichi_fast_launch.py` against `quadrants.lang.kernel.Kernel` (prefer hooking
   `launch_kernel` over `__call__`, which now carries checkpoint/`qd.Tensor`/stream stages); re-run
   `benchmarks/_taichi_fast_launch_check.py` with verify on. Trim `taichi_warmstart.py` to the
   source-retrieval memo (`get_pos_info` is memoised upstream, `ast_transformer_utils.py:408-424`).
4. `mps_zero_copy.py`: replace the per-launch sync pair with
   `qd.init(external_metal_command_queue=quadrants.interop.get_mps_command_queue(), external_metal_command_queue_is_torch_queue=True)`.
5. CI: rewrite `taichi_build.yaml` from `quadrants-src/.github/workflows/scripts_new/`
   (macOS, manylinux x86_64/aarch64, Windows); `QD_WITH_VULKAN=OFF` on macOS drops the MoltenVK
   dependency; enable `QD_WITH_CUDA=ON` on Linux/Windows.
6. Re-baseline `tests/fast` and `tests/full_renders` with the diffs inspected; record warm
   `save_frame` timings before/after; delete `ALGAN_GPU_MAX_REG`.
7. Then Track C items on the new base: item 1 reuses `Program.load_fast_cache` and the
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
