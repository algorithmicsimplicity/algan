# Taichi gotchas (these cost real debugging time)

> **Importing the compiler.** Never write `import taichi` inside `algan/`. Use
> `from algan.taichi_compat import ti` (and `submodule("lang.impl")` for a submodule):
> `ALGAN_TAICHI_BACKEND` selects **Quadrants 1.3.x (the default, and the only compiler a
> plain install carries)** or taichi 1.7.x (the `taichi` extra), and the layer exists
> so both can never be live in one process. See `agent_guidance/api_settings.md`.
> **The rule covers `tests/` and `benchmarks/` too** — a direct `import taichi` in a test is a
> mixed-compiler process, and one of the two that existed was a real failure under Quadrants
> (a `DataTypeCxx` compared against a `DataType`).
- **Which base the fork should sit on was gated on 2026-09-04; the gate passed for Quadrants.**
  Measured, in `taichi_patches/PLAN.md` §6.1, with `taichi_patches/MIGRATION.md` the record of
  the migration itself (what shipped, what it corrected, what is still unverified): `tests/fast` renders **byte-identical** under
  Quadrants 1.3.0 on Linux x64 (0 of 37.6 M channel samples differ, same md5 as the baseline) and
  pixel-identical on Windows CUDA, so LLVM 15 → 22 needs no re-baseline on x86-64; stock Quadrants
  builds clean on `macos-latest` in 12.5 min with a working Metal wheel, where Taichi 1.7.4 plus
  `taichi_patches/` no longer builds at all (`-Wnontrivial-memcall`, after patch 0003 cleared
  `-Wdeprecated-literal-operator`); and upstream #8745 — a stale Metal read when a buffer is written
  and re-read in one loop — reproduces on Taichi and is clean on Quadrants. Two things argue the
  other way. One is **pre-Volta CUDA**, which their CI never exercises and which still blocks the
  primary dev box (§7.3 Prerequisite 0). The other was a 2.1× warm frontend, since **halved by
  porting `taichi_warmstart.py` to Quadrants** — no Quadrants *release* carries the upstream
  `get_pos_info` memo, so Algan carries its own on both compilers now.
- **Renders and timings taken on one backend are still not comparable to the other's by wall
  clock.** One warm `save_frame` of a `Square` (22 kernels), memoization on:  **7.2 s on Taichi,
  12.5 s on Quadrants**; `--fast` is ~50 s against 78 s. What is left is Quadrants building every
  kernel AST **twice** (a pruning pass and an enforcing pass — every frontend counter is exactly
  2.00× Taichi's), which the memo does not touch. None of the gap is in the offline cache: warm
  backend time is 0.24 s against 0.27 s. A/B anything with both arms on one backend.
- **The memoization is version-gated to internals it patches** (taichi 1.7.x, quadrants 1.3.x) and
  turns itself off on anything else — which has already cost this project ~25 s per render once,
  unnoticed. After a compiler upgrade, run `algan check`: it names the reason when the accelerator
  is off. `benchmarks/_taichi_warmstart_check.py` is the audit (three arms, one of them recomputing
  every memoized value the original way and comparing).
- **The offline kernel cache does invalidate on `@ti.func` edits.** Its key is a SHA-256 over the compile config, the device capabilities and the kernel's *frontend IR* — which already contains every inlined `@ti.func` body and every captured global — so any edit that changes what is compiled misses the cache by itself (probe-verified on both compilers, 2026-09-04: edit only a func body, kernel source untouched, and Quadrants' `CompileResult.cache_hit` goes `False`; a cache hit costs 1 ms of backend time, the miss 40–60 ms for a trivial kernel). Older notes here and in several DESIGN files said the opposite and were wrong. `clear_cached_kernels()` is for a measurement that needs a **cold** compile (a compile-time A/B, a first-run timing), not for correctness. What *does* need one process per arm is a `ti.static` gate, below.
- Never edit `*_taichi.py` while a render **is running**: the JIT reads files at first launch and can compile half-edited code. Between runs you are covered — the daemon fingerprints every Algan source file and refuses to serve a run once any of them changes, shutting down so the script executes in a fresh process (`DESIGN_daemon_lifecycle.md`). You no longer restart it by hand; you do still pay the cold start, and a kernel edit still pays a full recompile.
- Cold kernel compilation takes minutes (the Monte Carlo path tracer is a separate kernel with its own cold compile); compiled kernels are cached.
- In kernels, use `ti.static(bool(x))` rather than `is not None` for template gates. A `ti.template()` argument may be a **nested tuple** — `((1, 2), (3,))`, three levels deep, a tuple of tuples of `@ti.func`s — on both compilers (probe-verified 2026-09-04, walked with nested `ti.static(range(len(...)))` loops); what fails is a `dict` or `list` anywhere in the value (`TypeError: cannot create weak reference`). On Quadrants a tuple argument also disables the launch-time *template mapper cache* for that kernel (a one-shot warning), which is a launch-cost consideration, not a compile error.
- **Kernel arguments.** The "64-argument ceiling" is a Python-side counter — `max_arg_num = 64` in taichi 1.7's `kernel_impl.py`, `MAX_ARG_NUM = 512` in Quadrants' `_func_base.py` — not a codegen or C++ limit: the args struct is sized from the parameter list. It is still not a number to lean on. Every extra ndarray argument costs a `set_arg` per launch and per-use argument loads inside the kernel, and Metal's 31-buffer limit stands, which is why the arena convention packs into few arguments.
- **The offline cache's metadata lock has no staleness rule** (`ticache.lock` on taichi, `kernel_compilation_manager/qdcache.lock` and `ptx_cache_sm_*/ptxcache.lock` on Quadrants — a bare `O_EXCL` file, five 50 ms retries). A process killed while holding it leaves every later run unable to load *or save* a kernel, with only a warning. `init_taichi()` removes one older than ten minutes before `ti.init` and logs a warning saying so; `ALGAN_TI_FULL_TRACEBACK=1` keeps the compiler's own frames in a compile error.
- **Never call `ti.init` yourself — call `init_taichi()` (idempotent), or pass `**taichi_init_kwargs()` and override from there.** `ti.init` is process-global and takes Taichi's *default* for every kwarg it is not given, so a bare call reconfigures Taichi for everything compiled after it, in code that never mentions it. The kwarg that matters is `advanced_optimization`, which Algan runs with **off**: under Taichi's default (on), `pbr_neutral_tonemap` miscompiles — the peak rescale inside its compression branch is dropped, tonemapping an authored white to 244 instead of 222. A bare `ti.init` in a *test* is what made three `test_tonemapping.py` guards fail in CI while every one of them passed when run alone (the file that broke them sorts earlier in the run). `tests/unit_tests/test_taichi_runtime_config.py` enforces the rule across `algan/`, `tests/` and `benchmarks/`. The same hazard applies to `ALGAN_ADV_OPT=1`, which is an A/B switch, not a supported render config — write a kernel so it survives being compiled either way.
- **A `ti.static` gate is resolved when the kernel compiles, so flipping the setting behind it mid-process does nothing.** The second arm silently reuses the first arm's code and reports its numbers as its own — it does not error, and clearing the offline cache does not help because that is not the cause. This bit the linear-colour work twice: an A/B harness whose two arms were both really the first arm, and a probe where an ambient change appeared to do nothing (the shadow floor sat at `encode(0.1)`, the other arm's value). **Run one process per arm for anything a `ti.static` gate controls.** A gate passed as a `ti.template()` *argument* is fine — Taichi specialises on those, which is why `tonemap_to_u8` can be flipped in-process and the shading stages cannot.