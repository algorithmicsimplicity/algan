# Compiler and kernel guidance

## Import and initialize through Algan

Use `from algan.taichi_compat import ti`, and `submodule("lang.impl")` when a
compiler submodule is needed. The same rule applies to `algan/`, tests and
benchmarks: importing Taichi directly can create an invalid mixed-compiler
process. `ALGAN_TAICHI_BACKEND` selects the compiler module. A normal installation
uses `quadrants` supplied by the pinned `algan-quadrants` distribution; upstream
Taichi is an optional comparison backend. See [api_settings.md](api_settings.md)
and [../quadrants_patches/PYPI.md](../quadrants_patches/PYPI.md).

Call `init_taichi()`, or use `taichi_init_kwargs()` when a harness must override
initialization. Do not call bare `ti.init`: it silently reinstates compiler
defaults for later kernels in the same process. In particular, Algan disables
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
