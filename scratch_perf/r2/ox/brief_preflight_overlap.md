# Task: stop the arena preflight from being serial on the render thread

Read `CLAUDE.md` in this worktree first and obey it — especially the sections on
**Memory** (`ManualMemory`, the runtime chunk-peak model, why the OOM retry is
load-bearing) and on the prefetch pipeline. Run Python as `uv run python` from
this worktree. **Stay inside this worktree**: the box carries several git
worktrees of algan (`D:\algan`, `D:\algan_wt_prep`, `D:\algan_wt_sheet`,
`D:\algan_wt_lab`) and other agents are working in them right now. Do not read
from, write to, copy into, or run anything outside your own tree. Do not commit
and do not push.

Set `ALGAN_USE_DAEMON=0` for every script. You share one GPU (NVIDIA GTX 1050,
4 GB) with other agents, so a single pair of wall-clock numbers is not a
measurement: alternate arms in separate processes, at least 3 each, report
medians — and say how you controlled for contention.

## Why this task exists

At `PREVIEW` quality the nn benchmark scene is preparation-bound. On a Tesla T4,
warm steady state, whole run **6.25 s** for 50 frames in 3 batches:

```
stage                                   calls  incl (s) incl(%)   note
ray traced render total                     5    3.245   52.0%
Scene.get_batch_of_primitives               3    2.322   37.2%   runs on the prefetch worker
arena preflight (batch)                     3    1.519   24.3%   runs on the RENDER thread
  - project_to_screen (prewarm)             3    0.578    9.3%
     logical PN: dice + shade + pack        9    0.390    6.2%
  merge collections + build BVHs            3    0.930   14.9%
     - refit-BVH build (in merge)           6    0.403    6.5%
```

2.32 + 3.25 + 1.52 ≈ 6.25: **preparation, preflight and render are essentially
serial.** The prefetch pipeline already hides `get_batch_of_primitives` behind
the previous batch's render, but the preflight is deliberately *not* hidden, and
it is a quarter of the run.

The "deliberately" matters — read these before designing anything:

* `RenderLoopMixin._prewarm_render_batch` (`algan/render_loop.py`) says
  projection rides the prefetch worker **only when it runs on the CPU**; when it
  runs on the render device (the default, `settings.PROJECT_ON_GPU`) it is
  deferred to the render thread "so its transient device peak is
  measured/bounded without a concurrent render polluting the pool". The same
  sentence appears for the merge (`settings.MERGE_ON_GPU`).
* `RenderLoopMixin._prepared_batch_fits_render_arena` is where that deferral
  lands. It does not just *check* — it **performs** the projection and the merge,
  wraps each in `begin_cuda_peak` / `end_cuda_peak`, and feeds the measured peak
  to `self._project_peak_ratio` / `self._merge_peak_ratio`, which are the
  predictors that size the *next* batch's frame window.

So the serialization buys two things: an uncontended CUDA peak measurement, and
a window decision made with the real numbers in hand. Any overlap has to keep
both honest or replace them with something equally honest.

## What to do

### Part 1 — measure and confirm, before changing anything

Reproduce the split on this box. `scratch_perf/r2/probe_prep_cprofile.py` drives
`get_batch_of_primitives` directly; you will need a companion that drives a real
`save_video` with the profiler installed so you get the `arena preflight (batch)`
stage. Report, for the nn scene at PREVIEW and at HD:

* wall time of the whole render, of the preflight, and of its two halves;
* how many batches the run uses and how many preflights that is;
* what fraction of the preflight is the **first** batch (the first batch cannot
  be overlapped with anything — there is no previous render to hide it behind —
  so it is a floor on what this task can win).

That last number decides the size of the prize. State it before proceeding.

### Part 2 — overlap what can be overlapped

The design I would try first, but convince yourself with Part 1's numbers rather
than taking it as given:

**Let the prefetch worker run the GPU projection and merge for batch b+1 once
the peak predictors are warm, and leave the first batch (or first two) on the
render thread exactly as today.** The predictors already exist and already
report whether they have enough observations to predict — use that as the gate,
not a batch counter, if the code supports it. Points to get right:

1. **The peak measurement must not be taken while a render is running.** Either
   skip `observe()` for an overlapped batch (the predictor keeps the estimate it
   learned from the un-overlapped ones) or measure it in a way that is immune to
   concurrent allocation. Skipping is honest and simple; say which you chose.
2. **Headroom must account for the concurrent render.** `_gpu_merge_headroom_bytes`
   sizes the out-of-arena scratch against pool headroom computed as if nothing
   else were running. When the merge runs beside a render, the real free memory
   is lower. Derate it, and make the derate a setting rather than a magic number.
3. **The window decision.** The render thread's preflight currently *chooses* the
   frame window by probing candidate durations. If the worker has already
   projected and merged a window, a render-thread rejection wastes that work.
   Establish from Part 1 how often the first probe already succeeds
   (`render_loop.py` claims "It is often already the final answer") and design
   for that case, with the existing shrink path as the fallback. Note the
   related negative result in this repo's history: a *speculative* prefetch of
   the successor batch measured +15% because a wrong guess had to be waited out.
4. **The OOM retry stays.** It is the backstop and must keep working; show that
   it still does (force it, e.g. by pinning
   `SETTINGS.computing.available_memory_override` low enough that a batch has to
   shrink, and show the render still completes and is byte-identical).

Gate the whole thing behind a new setting + env var (declared in
`algan/environment.py`, read live at call time), **default OFF** until the
verification below passes, and say in the report what you think the default
should be and why.

If Part 1 shows the prize is smaller than it looks — for instance if the first
batch is most of the 1.5 s — then say so with the numbers and stop. That is a
good outcome; do not build the machinery anyway.

## Verification (all required; quote the actual output)

- **Byte-identical render, toggle off vs on: 0 differing pixels.** Render the nn
  scene at `PREVIEW` in both arms, pass
  `ffmpeg_params=["-c:v", "libx264rgb", "-qp", "0"]` (an H.264 re-encode turns
  single-channel differences into thousands of differing pixels), compare with
  `benchmarks/_video_diff.py`, and pin
  `SETTINGS.computing.available_memory_override` to the same value in both arms.
  **Window changes legitimately move pixels** — the batch window sets batch-wide
  tessellation maxima — so if your change alters the chosen windows, say so and
  show the windows in both arms; a pixel difference caused by a different window
  is expected, a pixel difference at identical windows is a bug.
- Repeat the byte-identity check at `HD`.
- The forced-OOM-retry check from point 4.
- `uv run -m pytest -q tests/unit_tests` — full unit suite.
- `uv run -m pytest -q --fast` — report the timing line. Its pixel-comparison
  test fails on this machine even on unmodified code (the committed CUDA
  baseline came from a different GPU); report it and move on.
- `uv run ruff check --no-fix` and `uv run ruff format --check` on every file you
  touched.

## Report

Write `scratch_perf/r2/ox/REPORT_preflight_overlap.md`: Part 1's measurements
including the first-batch floor; what you changed; measured before/after with
how you controlled for the other agents on this GPU; what the change does to
peak VRAM (measure it, do not reason about it); and — explicitly — everything
you did **not** verify.
