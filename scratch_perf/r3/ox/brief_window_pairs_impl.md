# Brief: kernelise `_class_pairs_flat` and hoist the `_window_pairs` prologue

Implement options **A + B** of `scratch_perf/r3/ox/REPORT_window_pairs.md` §5
(read it first — it is your own audit and its §1 order contract and §5
acceptance contract govern this work). Read `CLAUDE.md` and obey it,
especially the Taichi gotchas and linting sections. You are in
`/home/user/algan`. Do not commit, do not push. `ALGAN_USE_DAEMON=0` in every
run. This container has no GPU; everything here runs on the Taichi CPU
backend, which is real verification for this change (integer host pass — no
CUDA-specific behavior).

## Scope

1. **A — hoist the kind-independent prologue** of `_window_pairs`
   (`raster_pipeline.py:1006-1018`: frame indices, `lo_p/hi_p`,
   `row_lo/row_hi`, `rl_f/rh_f`) into `prepare_sparse_raster_coverage`,
   computed once and passed to both call sites. The per-kind
   `rows = f_abs % pre_f.shape[0]` stays per kind. Byte-identical by
   construction — identical tensors computed once.
2. **B — replace the body of `_class_pairs_flat` with a Taichi kernel.** The
   host keeps `.nonzero()` (the only allowed sync today) and may keep
   `cumsum` plus ONE scalar readback to size the output (`M`); everything
   else — the gathers, `repeat_interleave`, `arange`, offset arithmetic, the
   `stack`, the int32 cast — moves into the kernel. Thread per candidate
   `k`, writing its `nch_k` rows contiguously at the host-computed base
   offset, ascending strip order. Emission order (ascending candidate,
   ascending strip) is LOAD-BEARING — your audit §1 established fragment
   layout follows pair-row order through two stable sorts.
   Note `repeat_interleave` with tensor repeats implicitly syncs to size its
   output, so your one `.item()` readback replaces a sync, not adds one;
   count syncs before/after with the census probe and report both.

Do NOT attempt option C (fusing the float clamp/reach predicate) — it is a
follow-up with its own proof burden.

## Mechanics and conventions

- Kernel goes in an existing `*_taichi.py` (e.g. `raster_taichi.py`, beside
  the count/write kernels that consume the rows) — the `_taichi` suffix is
  load-bearing for lint/format config. No `from __future__ import
  annotations` there. Remember the Taichi scoping rule that shipped a broken
  fan once: a local assigned first inside an `if` branch does not exist
  outside it — declare before branching.
- Toggle: module global + env-var default + setter, surfaced through
  `SETTINGS.raytracing.experimental`, name declared in
  `algan/environment.py` (two-step rule there), read live at call time.
  Default ON only if every check below passes bit-exact; otherwise default
  OFF and say why. Env kill-switch must restore today's torch path
  byte-identically.
- The int32 rows and every intermediate are integers — the kernel must be
  exact by construction. No fast-math concerns arise for integer ops; do not
  enable anything nonstandard.

## Acceptance contract (from your audit §5 — all required, quote outputs)

1. **Elementwise harness**, extending the `benchmarks/_sheet_kernel_check.py`
   pattern: torch arm vs kernel arm on the four pair tables — equal values,
   equal ROW ORDER, dtype int32, equal shape — at 4K-scale shapes and edge
   cases: empty mask (both arms return `None`), single candidate, bbox area
   exactly divisible by `RASTER_CHUNK`, `x0 == x1`, width-1 and height-1
   boxes, K large enough to exercise multi-block launches. Randomized tables
   with a fixed seed, at least 200 random cases plus the named edges.
2. **A compile-and-run unit test** for the kernel in
   `tests/unit_tests/` that actually launches it on real (small) inputs and
   asserts equality with the torch path — a host-side test cannot see a
   Taichi scoping/compile error, and that class of defect shipped before.
   Confirm the test FAILS if you break the kernel deliberately (say how you
   checked), then restore it.
3. **A/B render, toggle off vs on**: nn scene at PREVIEW on this box, both
   arms in separate processes, pin
   `SETTINGS.computing.available_memory_override` to the same value in both,
   `ffmpeg_params=["-c:v", "libx264rgb", "-qp", "0"]`, compare with
   `benchmarks/_video_diff.py`. 0 differing pixels required. (A new kernel
   pays a cold Taichi compile on its first arm — budget for it.)
4. **Census after**: re-run
   `scratch_perf/r3/probes/count_window_pairs_dispatches.py` against the
   kernel path (adapt a copy under `scratch_perf/r3/probes/` if needed) and
   report dispatched-ops and syncs per call, before vs after. Target is
   roughly 119 -> ~45 or better with both classes live.
5. `uv run -m pytest -q --fast` (report the timing line; it will be inflated
   by the new kernel's cold compile on run 1 — run it twice and report both).
6. `uv run -m pytest -q tests/unit_tests/test_raster*.py` plus any test file
   covering code you touched, and your new test. Do NOT run the whole unit
   suite in one process (RAM).
7. `uv run ruff check --no-fix` and `uv run ruff format --check` on every
   file you touched (the `*_taichi.py` file is excluded from format — do not
   "fix" that).

## Traps

- `tests/full_renders` baselines are per-machine and FAIL here —
  pre-existing, do not chase.
- The Taichi offline cache does not invalidate on `@ti.func` edits — if you
  edit the kernel after a run, clear the cache
  (`clear_cached_kernels()`) before re-verifying.
- Never edit a `*_taichi.py` while one of your own renders is running.
- Do not modify `scratch_perf/r3/probes/count_window_pairs_dispatches.py`
  in place — copy it if it needs adapting.

## Report

`scratch_perf/r3/ox/REPORT_window_pairs_impl.md`: what you changed (files,
line ranges), the kernel's threading/offset design in five lines, every
verification output verbatim, the census before/after table, and —
explicitly — everything you did NOT verify (the T4 timing claim belongs
there: this box cannot measure it; the operator will run the T4 A/B).
