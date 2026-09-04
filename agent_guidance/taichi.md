# Taichi gotchas (these cost real debugging time)

> **Importing the compiler.** Never write `import taichi` inside `algan/`. Use
> `from algan.taichi_compat import ti` (and `submodule("lang.impl")` for a submodule):
> `ALGAN_TAICHI_BACKEND` selects taichi 1.7.x or the Quadrants fork, and the layer exists
> so both can never be live in one process. See `agent_guidance/api_settings.md`.
> **The rule covers `tests/` and `benchmarks/` too** — a direct `import taichi` in a test is a
> mixed-compiler process, and one of the two that existed was a real failure under Quadrants
> (a `DataTypeCxx` compared against a `DataType`).
- **Which base the fork should sit on was gated on 2026-09-04; the gate passed for Quadrants.**
  Measured, in `taichi_patches/PLAN.md` §6.1: `tests/fast` renders **byte-identical** under
  Quadrants 1.3.0 on Linux x64 (0 of 37.6 M channel samples differ, same md5 as the baseline) and
  pixel-identical on Windows CUDA, so LLVM 15 → 22 needs no re-baseline on x86-64; stock Quadrants
  builds clean on `macos-latest` in 12.5 min with a working Metal wheel, where Taichi 1.7.4 plus
  `taichi_patches/` no longer builds at all (`-Wnontrivial-memcall`, after patch 0003 cleared
  `-Wdeprecated-literal-operator`); and upstream #8745 — a stale Metal read when a buffer is written
  and re-read in one loop — reproduces on Taichi and is clean on Quadrants. The two things that
  argue the other way are both unfixed there: **a 2.1× warm frontend** on the released wheel (no
  Quadrants *release* carries the `get_pos_info` memo Algan already has, and it builds every kernel
  AST twice), and **pre-Volta CUDA**, which their CI never exercises (§7.3 Prerequisite 0).
- Renders and timings taken on one backend are not comparable to the other's *by wall clock*: the
  Quadrants frontend costs ~25 s per process against Algan's ~4 s on Taichi, all of it Python, none
  of it in the offline cache (warm backend is 0.24 s against 0.27 s). A/B anything with both arms on
  one backend.
- The offline kernel cache does **not** invalidate on `@ti.func` edits — clear it before A/B-benchmarking kernel changes with `clear_cached_kernels()`.
- Never edit `*_taichi.py` while a render **is running**: the JIT reads files at first launch and can compile half-edited code. Between runs you are covered — the daemon fingerprints every Algan source file and refuses to serve a run once any of them changes, shutting down so the script executes in a fresh process (`DESIGN_daemon_lifecycle.md`). You no longer restart it by hand; you do still pay the cold start, and a kernel edit still pays a full recompile.
- Cold kernel compilation takes minutes (the Monte Carlo path tracer is a separate kernel with its own cold compile); compiled kernels are cached.
- In kernels, use `ti.static(bool(x))` rather than `is not None` for template gates, and keep template argument structures **flat** (nested tuples fail).
- **Never call `ti.init` yourself — call `init_taichi()` (idempotent), or pass `**taichi_init_kwargs()` and override from there.** `ti.init` is process-global and takes Taichi's *default* for every kwarg it is not given, so a bare call reconfigures Taichi for everything compiled after it, in code that never mentions it. The kwarg that matters is `advanced_optimization`, which Algan runs with **off**: under Taichi's default (on), `pbr_neutral_tonemap` miscompiles — the peak rescale inside its compression branch is dropped, tonemapping an authored white to 244 instead of 222. A bare `ti.init` in a *test* is what made three `test_tonemapping.py` guards fail in CI while every one of them passed when run alone (the file that broke them sorts earlier in the run). `tests/unit_tests/test_taichi_runtime_config.py` enforces the rule across `algan/`, `tests/` and `benchmarks/`. The same hazard applies to `ALGAN_ADV_OPT=1`, which is an A/B switch, not a supported render config — write a kernel so it survives being compiled either way.
- **A `ti.static` gate is resolved when the kernel compiles, so flipping the setting behind it mid-process does nothing.** The second arm silently reuses the first arm's code and reports its numbers as its own — it does not error, and clearing the offline cache does not help because that is not the cause. This bit the linear-colour work twice: an A/B harness whose two arms were both really the first arm, and a probe where an ambient change appeared to do nothing (the shadow floor sat at `encode(0.1)`, the other arm's value). **Run one process per arm for anything a `ti.static` gate controls.** A gate passed as a `ti.template()` *argument* is fine — Taichi specialises on those, which is why `tonemap_to_u8` can be flipped in-process and the shading stages cannot.