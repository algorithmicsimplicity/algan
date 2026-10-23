# Compiler and kernel guidance

## Import and initialize through Algan

Use `from algan.taichi_compat import ti`, and `submodule("lang.impl")` when a
compiler submodule is needed. The same rule applies to `algan/`, tests and
benchmarks: importing Taichi directly can create an invalid mixed-compiler
process. `ALGAN_TAICHI_BACKEND` selects the compiler module. A normal installation
uses `quadrants` supplied by the pinned `algan-quadrants` distribution; upstream
Taichi is an optional comparison backend. See [api_settings.md](api_settings.md)
and [../quadrants_patches/PYPI.md](../quadrants_patches/PYPI.md).

Call `init_taichi()`, or `taichi_runtime.start_program(**overrides)` when a
harness must initialize with an extra keyword (the kernel profiler passes
`kernel_profiler=True`). Do not call bare `ti.init`: it silently reinstates
compiler defaults for later kernels in the same process, and even
`ti.init(**taichi_init_kwargs(), ...)` skips the stale-lock sweep and the
exit-time `flush_kernel_cache` -- a profiled process used to leave no `.qdc`
behind, so every cold start measured through `profile_scene` was wrong. The
flush is also registered when the first specialization is built, whatever
started the program. There is deliberately no mid-process dump: Quadrants'
`KernelCompilationManager::dump` frees the kernels it writes, so a program that
launches again afterwards reads freed memory. In particular, Algan disables
`advanced_optimization`; historical tonemapping regressions under the enabled
arm motivate this default. `ALGAN_ADV_OPT=1` is a comparison switch, not evidence
that every renderer kernel supports that configuration.
`tests/unit_tests/test_taichi_runtime_config.py` checks initialization discipline.

## Specialization and cache boundaries

A `ti.static` read of a module setting is captured at compilation. Changing that
setting after a kernel has compiled does not change the cached specialization —
nothing keys the specialization on it. In contrast, a `ti.template()`
argument participates in specialization and can select distinct variants in one
process. Clearing the disk cache does not remove an already-live specialization.

Renderer *settings* read that way are named in
`rt_settings.KERNEL_COMPILED_IN_FIELDS`, and for those the runtime closes the
gap: `taichi_runtime` records what each specialization folded in, and
`ensure_taichi_for_render` re-inits the program — dropping every specialization
— when one of them has moved since. So both arms of `linear_color_space` (and
of the other four) render correctly in one process, at the cost of a kernel
preparation pass at the switch, logged at `PERF`. **Add a new `ti.static` gate
over a mutable setting to that tuple**; the defect it prevents is invisible
otherwise, because the first render of the process is always right and only the
second is wrong. `tests/unit_tests/test_compiled_in_settings.py` reads the
gates out of the source and fails on one that is missing.

That covers settings, not everything a kernel folds in. A `ti.static` gate over
anything else with a live value — an environment variable read at the point of
use, a module global some other code assigns — is still one process per arm.

Pass a dtype as a `ti.template()` argument through
`algan.taichi_compat.template_dtype(ti.f64)`, never bare. Quadrants' template
mapper weak-references every non-scalar argument and warns "Template mapper
caching disabled" for a raw `DataTypeCxx`; the handle is weakly referenceable
and otherwise the same dtype (equal, same hash, same source key and precompile
spec), so it compiles nothing new. The fragment-pipeline tuples cannot be made
weakly referenceable, and that one message is filtered in `taichi_compat`
(and re-applied for pytest in `tests/conftest.py`).
`tests/unit_tests/test_quadrants_warning_filter.py` renders a `Cube` with every
launch going through the mapper and fails on any other argument it cannot
weak-reference.

Run Python through the environment's interpreter, not bare `uv run`, when using
a locally patched compiler wheel. Lockfile synchronization can replace it. Check
the installed distribution and required compiler features rather than using the
version banner alone as proof that a patch is present.

Kernel modules retain their `_taichi.py` suffix and must not acquire
`from __future__ import annotations`: compiler annotations need runtime objects,
not strings. They are linted but excluded from formatting. Preserve deliberate
re-exports with `# noqa: F401`; see [../AGENTS.md](../AGENTS.md).

## Measure the current compiler, not an old benchmark

The migration measurements in [../taichi_patches/MIGRATION.md](../taichi_patches/MIGRATION.md)
and [../taichi_patches/PLAN.md](../taichi_patches/PLAN.md) are historical. The
pre-Volta build blocker was addressed by Quadrants patch 0003; argument-load
optimizations and warm-start/source-key work also landed afterward. Those older
startup ratios are not current performance guarantees.

`algan/utils/taichi_warmstart.py` memoizes reusable frontend work, and
`algan/utils/taichi_source_key.py` provides guarded source-key cache reuse. These
are complementary to the compiler's offline cache and a live warm daemon, not a
promise that every edited kernel or process start avoids compilation. Check
`tests/unit_tests/test_taichi_warmstart.py` and the source-key tests when changing
these paths. Time startup, frontend work, launch/synchronization and device work
separately, with both comparison arms on the same compiler/backend.

The runtime removes stale compiler cache lock files older than ten minutes
before initialization, with a warning. This is recovery from an interrupted
process, not a reason to delete active cache locks. `ALGAN_TI_FULL_TRACEBACK=1`
retains compiler frames in error reports. Runtime settings and daemon refusal
rules are documented in [api_settings.md](api_settings.md).

Do not edit a kernel file while a process is preparing to compile it: source
inspection occurs lazily, and mixing old imports with new source text can make a
validation result meaningless.

## Precompiling, progress, and the source memo

Three pieces cut first-run and fresh-process start-up; all three are
performance only and none may change a key or a pixel.

**The source memo** (`taichi_source_key._SourceMemo`) persists what source
retrieval and the bytecode walk produce -- the part of a source key that was
~90 % of its cost -- in `algan_source_key_memo.json` beside the kernel cache,
keyed by `(path, sha256 of the file's bytes, first line, qualname)`. Keyed by
content, not mtime: a wheel install can give two versions of a file the same
timestamp and size. An entry is stored only when re-deriving it from exactly
those bytes gives the same answer (a private `linecache` entry for the source
lines, a recompile of the module for the chains), so a file edited under a
running process never leaves an entry a later process would trust. Keys are
byte-identical with it on and off (`ALGAN_TAICHI_SOURCE_KEY_MEMO=0`);
`tests/unit_tests/test_taichi_source_key_memo.py` holds that over every kernel
body. Classes are not memoized (`inspect` finds them per-version).

**The precompile pool** (`algan/rendering/kernel_precompile.py`) records every
specialization a render materializes as a portable spec -- kernel path, array
dtypes and ranks, template values -- in `algan_kernel_specs.json`, and compiles
specs in worker processes (`python -m algan.rendering._precompile_worker`) with
one-element placeholder tensors, never launching. Facts it rests on, all
measured on Quadrants 1.3:

* `Program.compile_kernel` and `load_fast_cache` **hold the GIL** for their
  whole duration (a 20 s gap in a 50 ms ticker while the CPU backend compiled
  `sheet_resolve_shade_arena`). Threads cannot overlap kernel compilation with
  anything; only processes can. That is also why the progress heartbeat cannot
  speak during a backend compile and a slow one is announced before it starts.
* A program reads the offline cache's index **once, at its first use of the
  cache**, and never again: an artifact another process dumps after that is
  invisible to it, through the source-key index and the offline cache alike,
  even after dumping its own (`test_a_process_that_used_the_cache_does_not_see_later_artifacts`
  pins this). Before that first use, it sees everything on disk; concurrent
  dumps from several processes do not clobber each other.
* `device_memory_GB` is not in the C++ key; it is now excluded from the source
  key too (`_CONFIG_EXCLUDE_NAMES`), so CUDA workers run with a 0.25 GB pool.

The worker's own variables (`ALGAN_PRECOMPILE_WORKER`, `ALGAN_PRECOMPILE_JOBS`)
are in `_ENV_NOT_IN_KEY`; **anything a worker sets differently from the render
must be left out of the key**, or its index entries never match. Because of the
read-once index, a process waits for a running pool at its **first**
materialization (`before_first_materialization`) and never later -- mid-render
handoff was built first and measured useless. The implicit pool is scoped to
the running script's recorded specs, and runs only for specs not confirmed
under the current environment stamp: at the end of `import algan` in a
scene-script process (the daemon handoff's own test) -- under the settings the
script last rendered with, since a script that sets `SETTINGS` itself records
its kernels under those and not the import-time ones -- at the start of a
daemon run while its program is untouched, and at render start and at the
first materialization, where a pool for settings the script has since changed
is replaced by one for the current settings. It compiles the built-in list
only from the import-time decision on a **first run** (an empty manifest: a
new machine or `ALGAN_CACHE_DIR`) with at least six workers, and never other
scripts' kernels: measured, speculative work only slowed the render's own
compiles on a 4-core box, and anywhere with a record the record is the better
list (`ALGAN_FIRST_RUN_POOL=0` turns the first-run list off). It is a bet on
the scene matching a built-in variant: a cold first `explainer` render (a
match) went from 73-77 s to 43 s, while a cold first `graphics` render (shadows
set after import, its own lights: no megakernel matches) that got a
replacement built-in pool at render start waited 40 s for it and still
compiled its megakernels, 103 s against 81 s -- hence never this late (63 s
without it, mostly the torch warm-up below; one run each, noisy). Specs are portable only when everything
in them is `algan.*` or plain data -- a scene's own `@ti.func` stage keeps
compiling at first launch. **A spec's identity must contain only what selects a
specialization**: template values, array dtypes and ranks, scalar *types*. The
first version carried scalar values (counts, frame sizes) too; they differ
between batches of one scene, so the render never recognised the work a worker
held.

`algan warmup` (`kernel_warmup.py`) runs the pool over every recorded spec and
the built-in list, then renders the variant scenes at 4 fps, about one per four
cores (a render already uses every core; three at once on four cores took ~4x
longer each). **When a
kernel's signature or a pipeline's choice of kernels changes, regenerate the
built-in list** with `scripts/generate_kernel_specs.py`;
`test_every_builtin_spec_still_matches_a_live_kernel` (in `--fast`) fails until
you do. A
stale entry only costs parallelism, never correctness -- but the test checks
arity, not template values, and those decide whether a first run's pool
helps at all: the shading megakernels specialize on the light count (rounded
to a power of two), and a list holding only the lit 3-D scene's four slots
left a cold default-rig `explainer` render compiling both megakernels itself
after the pool had done everything else. Hence the `3d-default-light` variant.
(The sheet resolve now takes the light bucket only in its shadowed shading
pass, mode 2: its shadow-free and event-pass launches read no visibility and
pass one slot, so the two variants share their resolve and differ in
`wavefront_shade`.)
The list is generated on the CPU and matched the CUDA render's specs apart
from that (and three small CUDA-only kernels). With it, a cold first
`explainer` PREVIEW render on an RTX 5080 took 43 s against 73-77 s with no
pool, bit-identical.

Workers never launch: a launch with placeholder arguments is not safe
(out-of-range reads, unbounded loops over garbage data). On the published
wheel that leaves a CUDA worker producing the `.qdc` and nothing more, and the
render still generates PTX and pays the driver JIT at each kernel's first
launch (~14 s of a cold `explainer` render on an RTX 5080: first launches
36 s, of which the backend compile was 22 s). Quadrants patch 0011 adds the
entry point that does that work without the launch, `Program.prepare_kernel`,
and makes `dump_cache_data_to_disk` write the PTX cache too; where the wheel
has it, a CUDA worker prepares every spec before its dump
(`kernel_precompile.prepare_for_launch`, off under `ALGAN_PRECOMPILE_PTX=0`).
The render finds that PTX even though its PTX index was read at `init`,
before the pool finished: 0011 lets a content-hash lookup take a file the
index lacks. 0011 does not make the dump safe mid-process: its PTX half keeps
what it wrote, but the `.qdc` half still frees every kernel it writes, so the
rule above stands -- and a worker never prepares a kernel it compiled and
dumped itself (`_DUMPED_HERE`), whose handle that dump freed. At exit the rule
does not bite, and `taichi_runtime.write_ptx_at_exit` dumps there when the
flush will not: a process whose kernels all came from the index builds no
specialization, so the flush skips it, and Quadrants has no exit hook of its
own -- without it, PTX such a process generated (a cached kernel whose PTX was
missing) was never written, and every later process generated it again.
`quadrants_patches/check_ptx_cache.py <cache dir>` shows, without a GPU, which
cached kernels have their content-keyed PTX on disk.

**Loading cached kernels cannot overlap authoring on a thread.** On a warm
cache a fresh process's first render spends ~2.5 s (`explainer`) to ~4.4 s
(`graphics`) in `Program.load_fast_cache` and each kernel's first
`launch_kernel`, and a 0.5 ms ticker thread got no turns at all during either:
both hold the GIL throughout, so a background loader would stop the authoring
thread for as long as it saved the render. What it can do is less: ~90 % of a
cached kernel's load is LLVM parsing the IR text in its `.qdc` (CPU arch,
`sheet_resolve_shade_arena`: 151 of 166 ms), and the first launch then
verifies, clones, prints and hashes the module again to find its PTX.
Quadrants patch 0010 keys the PTX cache by the SHA-256 the `.qdc` already
stores and parses the IR only on a PTX miss, so on CUDA a fully cached kernel
skips all of that. `taichi_init_kwargs` turns it on where the wheel has it;
`ALGAN_PTX_CACHE_BY_CONTENT_HASH=0` is the off arm, and neither arm moves a
key (`quadrants_patches/README.md`, 0010).
The CPU arch had no machine-code tier at all, so every fresh process also
re-ran LLVM's codegen at each kernel's first launch (2-3 s a render). Patch
0012 keeps the CPU JIT's object file under the same content hash plus the host
CPU, links it on a hit without parsing the IR, and lets a CPU precompile
worker's `prepare_kernel` leave it behind; `ALGAN_CPU_OBJECT_CACHE=0` is its
off arm (`quadrants_patches/README.md`, 0012).

**Torch's lazy CUDA kernel loading does overlap**, because torch's bindings
release the GIL inside each op. `rendering/device_warmup.py` calls the ops the
renderer uses once, on tiny tensors, from a thread started during
`import algan` (scene scripts and the daemon, CUDA only), and the first render
job stops and joins it so it never shares the device with a render's memory
measurements. With every cache warm that loading is ~0.7 s of the first
render. With a cold CUDA driver cache (`CUDA_CACHE_PATH`, which lives under
`ALGAN_CACHE_DIR`: a new machine, a driver update, `algan.clear_cache()`) the
driver JIT-compiles torch's kernels and the same op list costs ~15 s, serially
-- four threads took as long as one -- which is the ~14 s a cold render spent
before its first Taichi kernel. Keep the op list in step with what the
renderer calls; a missing op only costs its load again. The thread stops on
its own when the render device moves off the one it warms, and a render that
outwaits its step in flight (30 s) logs at `PERF` and starts beside it.

**Stopping a pool never kills a worker that may hold the cache's metadata
lock** (a bare `O_EXCL` file; a holder killed inside it blocks every process's
loads and saves until the ten-minute stale-lock sweep). Polled on the CPU
arch, a worker holds it for the index read and each dump -- milliseconds,
never during a compile -- so a worker asks the parent (`enter`, answered `go`,
`finish` or `abort`) before bringing a program up (which reads the index, and
on a re-init dumps the old program) and before its index store and dump, and
`PrecompilePool.terminate` decides under the same lock: it waits (bounded) for
granted sections to end, kills only workers outside one, and refuses later
requests. `_kill_tree` takes the venv launcher's child interpreter with it,
and skips a process already reaped (its PID may be reused).

**Progress** (`kernel_progress.py`) replaced the one-shot "several minutes"
notice: a header on the first index miss of a render, a numbered line per real
compile (`CompileResult.cache_hit` says which were served from the offline
cache; the fd-2 capture remains only for the Taichi backend), the first-touch
wait for a pool and the kernels it finishes, a heartbeat, and a summary. Tests run with `ALGAN_PRECOMPILE_JOBS=0`
(`tests/conftest.py`).

### Check disk-cache reuse in separate processes

A first pass in a process is not necessarily a cold disk-cache pass. The scene
profiler labels first/repeat passes and records `source_key_cache` deltas for
both authoring and rendering. A source-key **hit** bypasses the Python AST
transform; a **miss** rebuilds the frontend but may still hit the compiler's
backend cache. Zero lookups on a repeat pass usually mean the specializations
are already resident, not that disk caching is disabled. The `launch` column
also includes cache loading and submission, so it is not a compilation timer.
Use `ALGAN_LOG_TAICHI_COMPILES=1` for per-specialization timings.

`python benchmarks/_taichi_source_key_check.py --arms warm,on,verify` runs
three separate processes against one cache. It requires every warm lookup to
hit and every verify lookup to re-derive the stored C++ key, as well as matching
pixels; identical pixels alone cannot prove caching worked. It forces daemon
use off and isolates verify mode from inherited environment settings. Use a
fresh `ALGAN_CACHE_DIR` to measure an actually empty cache. Changing the scene,
material-pipeline tuple, kernel source, compiler configuration or template
arguments can legitimately need new variants. Test an identical repeat before
calling a new variant a cache failure.
