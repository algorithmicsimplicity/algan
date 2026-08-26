# Brief: overlap the arena preflight with the previous batch's render

Read `CLAUDE.md` first and obey it — especially **Memory** (`ManualMemory`,
the runtime chunk-peak model, why the OOM retry is load-bearing) and the
prefetch pipeline. Run Python as `uv run python`. You are in
`/home/user/algan`, branch `claude/algan-t4-optimization-b5a10b`. Do not
commit, do not push. This container has **no GPU** — see "Verification split"
below before designing your test plan. `ALGAN_USE_DAEMON=0` in every run.

## Why (measured on a Kaggle T4, warm)

At PREVIEW the nn scene is preparation-bound: render 3.25 s,
`Scene.get_batch_of_primitives` 2.32 s (hidden by the prefetch worker),
**arena preflight 1.52 s (24%) serial on the render thread by design**. Round
2 measured the two ratios that decide this design
(`DESIGN_T4_optimization.md` §5.3): the un-overlappable floor (batch 1) is
0–6% of the preflight sum, and **the first window probe succeeds 100% of the
time** — the window chosen before the exact preflight was final in every
batch measured. A 467-line WIP patch existed but is lost; you are building
this fresh.

## The design constraint — read these before anything

* `RenderLoopMixin._prewarm_render_batch` (`algan/render_loop.py`):
  projection rides the prefetch worker only when it runs on the CPU; on the
  render device it is deferred to the render thread so its transient device
  peak is measured without a concurrent render polluting the counter. Same
  for the merge.
* `_prepared_batch_fits_render_arena` (`render_loop.py:859`) does not just
  check — it PERFORMS the projection and merge, wraps each in
  `begin_cuda_peak`/`end_cuda_peak`, and feeds `_project_peak_ratio` /
  `_merge_peak_ratio`, the predictors that size the next batch's window.

The serialization buys an uncontended peak measurement and a window decision
made with real numbers. Your overlap must keep both honest.

## What to build

Let the prefetch worker run the projection and merge for batch b+1 while
batch b renders, **once the peak predictors are warm**; the first batch (or
two) stays on the render thread exactly as today. Gate on the predictors'
own have-enough-observations signal if the code supports it, not a bare
batch counter. Points to get right:

1. **Never measure a peak while a render runs.** Skip `observe()` for
   overlapped batches (predictors keep what the un-overlapped ones taught
   them). Say in the report that you chose this, or justify an alternative.
2. **Derate headroom for the concurrent render.**
   `_gpu_merge_headroom_bytes` assumes nothing else is running. Make the
   derate a setting, not a magic number.
3. **Window decision.** The first probe is the final answer ~100% of the
   time, so speculate on it; keep the existing shrink path as fallback, and
   when the render thread rejects the worker's window, throw the worker's
   work away and fall back to today's serial path for that batch. (History:
   speculative prefetch of the whole successor batch measured +15% — the
   overlap must not repeat that by waiting out wrong guesses.)
4. **The OOM retry stays working.** Force it (pin
   `SETTINGS.computing.available_memory_override` low enough that a batch
   shrinks) and show the render completes.

Gate everything behind a new setting + env var declared in
`algan/environment.py`, read live at call time (`rt_settings.X` pattern),
**default OFF**. The default path must stay byte-identical by construction —
with the gate off, no behavior may change.

## Verification split (this box has no GPU)

First determine and report: with the render device on CPU, does
`_prepared_batch_fits_render_arena` still take the same deferral control
flow, so a CPU A/B exercises your overlap path? If yes, run the A/B here: nn
scene at PREVIEW, toggle off vs on, same
`SETTINGS.computing.available_memory_override` pinned in both arms,
`ffmpeg_params=["-c:v", "libx264rgb", "-qp", "0"]`, compare with
`benchmarks/_video_diff.py`; report differing pixels and the chosen windows
of both arms (a pixel diff at identical windows is a bug; a window change
must be explained). If the CPU path cannot exercise the overlap, say so
plainly and compensate with unit tests that drive the new scheduling logic
directly (fake predictor warmth, fake worker, forced rejection) — the T4
A/B and perf measurement will be run by the operator afterward; your job is
to make that run turnkey (state the exact env var and arms it needs).

Also required, quote actual output:
- the forced-OOM-retry check from point 4;
- `uv run -m pytest -q --fast` (report the timing line; run it three times if
  you touched anything a kernel variant depends on);
- `uv run -m pytest -q tests/unit_tests/test_render_loop*.py tests/unit_tests/test_memory*.py`
  plus any test file covering code you touched. Do NOT run the whole unit
  suite in one process (RAM);
- `uv run ruff check --no-fix` and `uv run ruff format --check` on touched files.

## Report

`scratch_perf/r3/ox/REPORT_preflight_overlap_impl.md`: what you changed and
why; the CPU-path answer; every verification output; what fraction of
preflights your gate actually overlaps on the nn scene (measure: it should
be all but the first one or two); and — explicitly — everything you did NOT
verify. Every claim carries a line number or a measured number.
