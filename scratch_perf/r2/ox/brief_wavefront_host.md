# Task: attribute (and then cut) the wavefront loop's host time on the sheet route

Read `D:\algan\CLAUDE.md` first and obey it — in particular the rules on
`*_taichi.py` files, on never calling `ti.init` yourself, on declaring every
`ALGAN_` env var in `algan/environment.py`, and on `ruff check --no-fix`.
Run Python as `uv run python` from `D:\algan`. Work on the current branch's
working tree. **Do not commit and do not push.**

You have the GPU (NVIDIA GTX 1050, 4 GB) to yourself. Set `ALGAN_USE_DAEMON=0`
for every script you run.

## Why this task exists

Profiling `benchmarks/performance/nn_scene_UHD.py` on a Tesla T4 (30 frames at
3840x2160, warm steady state, whole run 30.9 s) gives this at the top:

```
stage                                       calls  incl (s) incl(%)  excl (s) excl(%)
ray traced render total                        30    26.099   84.5%     0.122    0.4%
wavefront_loop                                 30    25.123   81.3%    12.022   38.9%
raster: sparse discovery                       30     8.634   28.0%     2.286    7.4%
kernel: wavefront_shade                       240     7.885   25.5%     7.885   25.5%
raster:   - compact_sheets                     30     4.503   14.6%     2.617    8.5%
raster: sparse resolve                         30     4.171   13.5%     4.171   13.5%
kernel: raster_shadow_trace                    30     3.733   12.1%     3.733   12.1%
kernel: wavefront_traverse_events             240     3.383   11.0%     3.383   11.0%
wavefront:   - compact active                 270     0.258    0.8%     0.258    0.8%
wavefront:   - raster tables (batch)            2     0.039    0.1%     0.001    0.0%
```

`wavefront_loop` is `raytrace_render_wavefront` in
`algan/rendering/raytracing/tracer.py`. **Its own exclusive time is 12.0 s — the
single largest item in the profile, 39% of the render, and completely
unattributed.** Every Taichi kernel it launches is separately hooked and
subtracted, so those 12 s are host-side torch work plus whatever GPU work the
stage timer's boundary syncs pull in.

The reason it is unattributed is structural. This scene takes the sheet /
analytic-raster route, so it enters the `if use_raster:` branch at
`tracer.py:2506`, which contains its **own** inline bounce loop
(`while active.numel() > 0 and it < max_iters:` at ~2562) and its own inline
covered-pixel chunk loop (~2830). The module-level helper `_run_wavefront_tiles`
*is* hooked as `wavefront:   - tile loop`, but that helper is only used by the
other (non-raster) routes — which is why that stage never appears in the report
above. Nothing inside the `use_raster` branch is hooked at all.

## What to do

### Part 1 — make the branch measurable (required, do this first)

1. Add an *opt-in, zero-cost-when-off* inline stage helper to
   `algan/utils/profiling_utils.py`. Suggested shape: a module-level flag set to
   True by `install_pipeline_hooks()` (and back to False by whatever uninstalls
   the hooks), plus

   ```python
   def stage(name):
       """Time an inline block, but only while the profiler's hooks are installed."""
       return TIMERS.stage(name) if _HOOKS_INSTALLED else _NULL
   ```

   where `_NULL` is a single shared `contextlib.nullcontext()`. This matters:
   `TIMERS.stage` calls `_sync_devices()` on entry *and* exit, so an
   unconditional stage inside a per-bounce loop would both slow the render and
   change what it measures. Verify the off path allocates nothing per call.

2. Wrap the phases of the `use_raster` branch with that helper, one stage per
   phase, named `wavefront:   - <phase>` so they sort under the existing
   `wavefront:` entries. Choose the phase boundaries by reading the code, not by
   guessing; at minimum separate:
   - setup before the bounce loop (pool/arena allocation, primary ray seeding,
     `pix_accum` setup, anything done once per frame-part);
   - **per-iteration host work inside the bounce `while`** — and split this
     further into the distinct torch passes you find (index/compaction
     bookkeeping, arena `get_tensor` calls, `.item()`/`.numel()` host syncs,
     mask builds, gathers/scatters). Any `.item()`, `.numel()`, `bool(...)` or
     `int(...)` on a CUDA tensor inside the loop is a full device sync and must
     be timed separately — call these out by line number;
   - the covered-pixel chunk loop at ~2830 and its inner `while` at ~2848;
   - composite / finalize / readback after the loop.

   Keep the render byte-identical: this part adds timing only.

3. Prove the instrumentation works and costs nothing when off:
   - render `benchmarks/performance/nn_scene_UHD.py`'s scene at **HD** (not
     UHD — this box has 4 GB of VRAM) *with* and *without* the profiler, and
     show the two outputs are byte-identical. Copy
     `scratch_perf/render_once.py` for the no-profiler arm and pass
     `ffmpeg_params=["-c:v", "libx264rgb", "-qp", "0"]` so the comparison is
     lossless (an H.264 re-encode amplifies single-channel differences into
     thousands of pixels — use `benchmarks/_video_diff.py`);
   - show the unprofiled wall time is unchanged (alternate the two arms at
     least 3 times each in separate processes and give medians, not one pair —
     wall-clock on this box is noisy).

### Part 2 — report the breakdown

Run the profile at **HD** and at **PREVIEW** (both `nn_scene_UHD.py`'s scene
function, with the `video_settings` argument changed; do not try UHD on this
card) and report the new stage table for `wavefront_loop` and its children, with
`wavefront_loop`'s remaining exclusive time. State clearly how much of the 12 s
you have now accounted for **in proportion** — you are measuring a different GPU
and a smaller frame, so absolute seconds will not match; what transfers is the
*share* of `wavefront_loop`'s exclusive time each phase takes, and how that share
moves between PREVIEW and HD (i.e. whether a phase scales with pixel count or is
a fixed per-bounce-iteration cost).

### Part 3 — cut the largest one

Pick the largest phase you found and reduce it, keeping the render
byte-identical. Do not guess at the fix before you have Part 2's numbers. Things
worth checking for, in the order they usually pay:

- **Host syncs in the bounce loop.** `active.numel()` on a CUDA tensor produced
  by a compaction is a device sync per bounce per tile per frame; so is any
  `.item()` used to size the next launch. If a sync only exists to decide
  "should I stop?", consider whether the loop can run a fixed number of
  iterations with a cheap device-side early-out instead, or whether the value
  can be read once per tile instead of once per iteration. Any change here must
  keep the same number of kernel launches with the same arguments, or it is not
  byte-identical — verify, do not assume.
- **Per-iteration arena traffic.** `memory.get_tensor(...)` calls, `.zero_()`
  fills and `torch.empty`/`cat`/`stack` inside the loop that could be hoisted to
  the frame or the tile.
- **Repeated recomputation of loop-invariant tensors** (index ranges, frame-part
  bounds, constant device tensors rebuilt per iteration).

Anything you change must be behind a setting toggle following the existing
convention in `algan/rendering/raytracing/settings.py` (env var declared in
`algan/environment.py`, read live as `rt_settings.X` at call time, default ON
only if you have proved bit-identity). If the honest answer is "the 12 s is
irreducible GPU time that the boundary syncs are attributing here", say that
plainly and stop — that is a valuable result and I would rather have it than a
change that does not help.

## Verification (all required; quote the actual output)

- `uv run -m pytest -q --fast` — report its timing line. The fast suite's
  pixel-comparison test fails on this machine on `master` too (the committed CUDA
  baseline came from a different GPU); confirm that by running the same test on a
  pristine `master` worktree and report both, but do not chase it.
- `uv run -m pytest -q tests/unit_tests` — full unit suite.
- The lossless byte-identity render A/B from Part 1.3, re-run after Part 3's
  change (toggle off vs on): must be **0 differing pixels**.
- `uv run ruff check --no-fix` and `uv run ruff format --check` on every file you
  touched.

## Report

Write `scratch_perf/r2/ox/REPORT_wavefront_host.md` containing: the new stage
table (PREVIEW and HD), the share of `wavefront_loop` exclusive time each phase
holds, a list of every host-sync site you found with its line number, what you
changed and what it bought, and — explicitly — everything you did **not**
verify. Do not overstate: if you did not run something, say so.
