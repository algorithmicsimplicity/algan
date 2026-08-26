# Report: overlap the arena preflight with the previous batch's render

Branch `claude/algan-t4-optimization-b5a10b`, on top of `4e77fde` (plus
`1b01f31`, which another actor landed mid-session and which does not touch the
render loop). Brief:
`scratch_perf/r3/ox/brief_preflight_overlap_impl.md`. The recovered round-2
patch (`scratch_perf/r2/patches/ox_preflight_overlap_WIP.patch`) was applied,
reviewed, corrected, completed, and verified. **Nothing is committed by me**;
an external harness committed my working tree as restart insurance
(`a2d7a69`, "WIP checkpoint"), and my subsequent fixes are uncommitted working-
tree changes on top of it. Note: `/scratch_*` is gitignored (.gitignore:168),
so the harnesses below are scratch files by convention.

## What you get, in one paragraph

With `SETTINGS.computing.prefetch_gpu_prep` (default **OFF**) or env
`ALGAN_PREFETCH_GPU_PREP=1`, once both transient-peak predictors are calibrated
(the first batch of a job always prepares on the render thread exactly as
today), the batch-prep worker runs the successor batch's projection and merge
while the current batch renders, bounded against pool headroom derated to 60%
(`overlap_pool_headroom_fraction`, env `ALGAN_OVERLAP_HEADROOM_FRACTION`),
stamps the batch `_rt_prep_overlapped`, and the render-thread arena preflight
skips only what is moot — the peak observations and the already-spent
estimates — while the exact arena accounting decides as before. A rejected
overlapped window is discarded and that stretch falls back to today's serial
shrink-and-refetch path. Gate off ⇒ no code path changes, by construction and
by measurement (below).

## Changes, by file (line numbers in the current working tree)

* `algan/settings/computing_settings.py:84,91` — new settings
  `prefetch_gpu_prep: bool = False`,
  `overlap_pool_headroom_fraction: float = 0.6`, validated in `__post_init__`
  (bool / finite float in (0, 1], lines 125-132).
* `algan/environment.py:206,208` — `ALGAN_OVERLAP_HEADROOM_FRACTION` and
  `ALGAN_PREFETCH_GPU_PREP` declared in `_LIVE_VARIABLES`; both are read live
  via accessors at their call sites, which is what `test_environment.py`
  enforces.
* `algan/render_loop.py:2477-2502` — `_overlap_headroom_fraction`: setting +
  live env override; an out-of-range override warns and falls back to the
  setting rather than silently dropping the derate (review fix).
* `algan/render_loop.py:2504-2523` — `_overlap_gpu_prep_active`: requires the
  setting (live read), **both** GPU builds active, and the calling thread to be
  the prefetch worker — a synchronous/retry fetch has no render to hide behind.
* `algan/render_loop.py:2525-2634` — `_prepare_batch_on_worker`: declines
  unless both predictors are calibrated (`PeakRatioModel.is_calibrated()`,
  memory_model.py:472 — the "predictors' own have-enough-observations signal",
  not a batch counter); bounds projection and then merge against the derated
  headroom; runs the real builds via `_prewarm_render_batch` /
  `_prepare_merged_host_scene(track_peak=False)`; stamps
  `_rt_prep_overlapped` (2631) only when everything succeeded. Merge OOM clears
  partial state and defers (2606-2628); non-OOM errors re-raise into
  `fetch_batch`'s existing defer-with-warning handler (2910-2918).
* `algan/render_loop.py:904-909,933,998,1038,1054-1061` — the preflight's
  `overlapped` branch: skips `_prewarm_render_batch`, both peak observations
  (project observe previously at old line ~948, merge observe at ~1026), both
  proactive estimates, and the `_note_batch_cost("projection"/"merge")` terms;
  keeps the exact scene bytes, the frame-cost model, `_last_arena_preflight`,
  and the verdict unchanged.
* `algan/render_loop.py:2905-2923` — `fetch_batch` consults the overlap gate
  first, then the legacy CPU-prewarm branch (review reorder; production
  behavior identical because the gate itself requires the GPU builds, but it
  makes the CPU test seam reachable without faking device state).
* `algan/rendering/memory_model.py:355-381` — `PeakRatioModel` gains a lock so
  the worker's `predict` cannot race the render thread's `observe` on the
  `maxlen` deque.
* `algan/rendering/raytracing/scene_builder.py:1245-1276` — `_merge_scene`
  accepts `track_peak=None` (default = today's behavior). Review finding: with
  the patch as written, a worker-side merged build under `MERGE_TRACK_PEAK`
  would call `begin_cuda_peak` → `torch.cuda.reset_peak_memory_stats`
  **under the live render** (scene_builder.py:1280) — exactly what point 1
  forbids — and would write a render-polluted `_gpu_merge_peak_bytes`
  (scene_builder.py:1982). The overlapped path now passes `track_peak=False`
  through `_prepare_merged_host_scene(primitive_batch, *, track_peak=None)`
  (render_loop.py:457-471): no counter reset beside a render, nothing measured
  that cannot be honest.
* `algan/utils/profiling_utils.py:747` — profiler hook for the worker stage.

## The four points to get right

1. **Never measure a peak while a render runs — chose skip, as instructed.**
   Overlapped batches observe nothing (`render_loop.py:998,1038`) and the
   worker-side merge does not even reset the counter (`track_peak=False`,
   above). The predictors keep what batches 1..k taught them; on the T4 those
   are real measurements of the same builds. Alternative rejected: observing
   scaled deltas — pure speculation, no measurement to validate it.
2. **Derated headroom, as a setting.**
   `_gpu_merge_headroom_bytes()` × `overlap_pool_headroom_fraction` (0.6),
   env-overridable live, range-guarded (render_loop.py:2577-2580). Boundary is
   inclusive-above: predicted peak must be ≤ derated headroom; unit-tested at
   600 000/600 001 against a 1 000 000 headroom.
3. **Window decision.** The worker speculatively prepares the whole fetched
   window (the first probe was final in 100% of measured batches — DESIGN_T4
   §5.3); the fetch boundary is still chosen only after the current duration is
   final (`render_loop.py:3177`, unchanged), so the +15% wrong-boundary
   failure mode from the history note cannot recur. When the render thread's
   exact check rejects the stamped batch, the existing shrink path discards the
   work (`primitives[0]._rt_* = None`, memory reset, refetch shorter) — forced
   end-to-end below ("rejected overlapped window" unit test + OOM arm).
4. **OOM retry stays working.** Forced on this box; quoted below. On CPU the
   equivalent pin is `max_cpu_memory_used` because
   `available_memory_override` only stands in for measured devices
   (memory_utils.py:71).

## The CPU-path answer (asked first, as instructed)

**No — a CPU render cannot exercise the overlap path.** Both
`project_on_gpu_active()` and `merge_on_gpu_active()` hard-require
`_RENDER_DEVICE.type == "cuda"` (`raytracing/settings.py:2405,2451`), and
`_overlap_gpu_prep_active()` requires both (render_loop.py:2509-2517); the
preflight's `overlapped` branch is unreachable for the same reason. Two
consequences worth stating plainly:

* On this box a *real* render never reaches `_prepare_batch_on_worker`.
  Compensated with 16 unit tests driving the scheduling logic directly
  (fake predictor warmth, fake worker, forced rejection, forced OOM-defer) plus
  two harnesses that patch **only the CUDA gate** (+ finite headroom stand-in,
  + two predictor seed observations standing in for batch 1's measurements) and
  otherwise run the real machinery end-to-end, real renderer included.
* There is also nothing serial left to overlap on CPU: with projection on CPU,
  today's `ALGAN_PREFETCH_MERGE` branch already runs the CPU builds on the
  worker (render_loop.py:2920), which is why the PREVIEW wall times below
  do not move between arms even though 49.7 s of build work changed threads.

## Verification — actual output

### Unit tests driving the scheduling logic (new file)

`tests/unit_tests/test_preflight_overlap.py`, 16 tests:

```
uv run -m pytest -q tests/unit_tests/test_preflight_overlap.py
16 passed, 3 warnings in 0.79s
```

Coverage map to the brief: gate needs setting + both GPU builds + worker
thread (`test_overlap_gate_requires_setting_cuda_builds_and_worker_thread`);
calibration gate keeps batch 1 on the render thread
(`..._needs_calibrated_predictors`); stamp/order
(`..._runs_both_builds_and_stamps`); derate arithmetic incl. the boundary
(`..._derates_headroom_for_the_render`); merge-OOM defers, real errors re-raise
(`..._merge_oom_defers...`, `..._reraises_real_merge_errors`); preflight of a
stamped batch skips prewarm/observations/estimates but keeps the exact verdict
and its cost terms (`test_preflight_of_overlapped_batch_skips...` vs
`test_preflight_of_unoverlapped_batch_measures_and_estimates_as_today`); the
real batching loop with a live worker thread overlaps every successor and none
of the first (`test_worker_overlaps_every_batch_after_the_first` — flags
[False, True, True]); gate off stamps nothing; a rejected overlapped window is
discarded and finishes serially with every frame rendered exactly once
(`test_rejected_overlapped_window_discards_work_and_finishes_serially`); and a
lock race test for `PeakRatioModel`.

### Forced OOM retry (point 4)

`scratch_perf/r3/ox/oom_retry_check.py`, arena pinned via
`max_cpu_memory_used=1 MB` (arena ≈ 0.4 MB) on a 10-frame 32×32 shadows-on
cube scene, `max_animation_batch_size=2`. Serial arm:

```
[summary] {"arm": "serial", "status": "rendered", ... "preflight_calls": 16,
"rejections": 7, "overlapped_preflights": 0,
"render_windows": [[0, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
[8, 9], [9, 10]], "worker_prep": []}
```

with the engine's own DEBUG trail showing the designed loop, e.g.:

```
Arena preflight rejects: scene 0.1 + frame 0.6 = 0.7 MB vs 0.4 MB remaining ...
Prepared batch does not fit the render arena; binary-searching the largest fitting duration.
Fetching batch 3:4.
```

Overlap arm (same pin, overlap active through the documented seams):

```
[summary] {"arm": "overlap", "status": "rendered", ... "preflight_calls": 16,
"rejections": 7, "overlapped_preflights": 8, "render_windows": [[0, 2], [2, 3],
..., [9, 10]], "worker_prep": [{"dt": 0.0258, "thread": "algan-batch-prep_0",
"stamped": true}, ... x8]}
```

Same windows, same 7 rejections, render completes. Pixel diff of the two
outputs:

```
uv run python benchmarks/_video_diff.py .../oom_base_SMOKE_TEST.mp4 .../oom_ovl_SMOKE_TEST.mp4
frames compared: 10
worst channel diff: 0 (frame -1)
pixels over tol 2: worst frame 0 of 1024 (0.000%, frame -1); mean 0.0/frame; 0 of 10 frames affected
```

On the T4 the operator's version of this pin is
`SETTINGS.computing.set(available_memory_override=<small>)` (cuda/mps-only per
memory_utils.py:70-72).

### A/B, nn scene, identical windows ⇒ pixel diff must be zero (it is)

`scratch_perf/r3/ox/ab_preflight_overlap.py` (adapted from the r2 companion;
adds overlapped-fraction reporting and the `--cpu-seams` mode). nn scene,
shadows on, lossless `libx264rgb -qp 0`, one process per arm, arms flipped via
`ALGAN_PREFETCH_GPU_PREP`.

SMOKE_TEST (10 frames, 4 batches):

```
off: preflight_calls 4, overlapped 0,      windows [[0,3],[3,6],[6,9],[9,10]]
on:  preflight_calls 4, overlapped 3 (75%),windows [[0,3],[3,6],[6,9],[9,10]]
     worker_prep_calls 3, worker_prep_sum_s 3.807
_video_diff: frames compared: 10 | worst channel diff: 0 | 0 pixels over tol 2
```

PREVIEW, nn scene (50 frames, 17 batches):

```
off: wall 63.718s, preflight_calls 17, overlapped 0,       preflight_sum 0.011s
on:  wall 63.557s, preflight_calls 17, overlapped 16 (94.1%), preflight_sum 0.010s
     worker_prep_calls 16, worker_prep_sum_s 49.688
     windows: 17 x 3-frame batches, identical in both arms
_video_diff: frames compared: 50 | worst channel diff: 0 (278,784 px/frame)
             0 pixels over tol 2, 0 of 50 frames affected
```

Identical windows were expected on CPU (both arms bookkeep only the arena term
because the projection/merge estimate terms exist solely under the CUDA
builds) — and a diff at identical windows would have been a bug. It is 0.
The wall times also do not move, for the reason given in the CPU-path answer:
CPU builds were already hidden by `ALGAN_PREFETCH_MERGE`. The 1.48 s/batch
serial-preflight prize this feature exists for exists only where those builds
run on the render device — i.e., on the T4.

### Fraction of preflights actually overlapped

Measured, not assumed: **all successors after the first** —
16/17 (94.1%) on the nn scene at PREVIEW, 3/4 on the SMOKE_TEST variant, 8/8
successor batches under the forced-OOM pin (`overlapped_preflights` fields
above). This matches the design target "all but the first one or two"; the
calibration gate alone decides the count (`is_calibrated` flips after batch 1's
render-thread measurements), and no estimate decline fired in any run here.

### Required suites

```
uv run -m pytest -q --fast
fast suite: 17s of its 75s budget (22%)
1 failed, 276 passed, 1968 deselected
```

The one failure is the fast suite's pixel-compared render, and it is
**pre-existing**: checked out detached at my starting commit `4e77fde` and ran
it there —

```
FAILED tests/fast/test_fast_render.py::test_the_fast_scene_renders_and_matches_its_baseline
AssertionError: fast.mp4 differs from its baseline by up to 5 channel values (worst at frame 27)
```

— byte-for-byte the same signature as with my changes (5 @ frame 27). `4e77fde`
is itself the weight-floor landing whose accepted 1-LSB-class variation moved
baselines (DESIGN_T4 §5.3 item 5); this box's CPU baseline disagrees with it.
I did not re-baseline: not my change, and CLAUDE.md says to look before
re-baselining — the operator should decide on a machine matching how
`expected_outputs_cpu/` was produced. I touched no kernels and no kernel
variants, so per CLAUDE.md the single fast-suite timing is representative
(the three-run rule is for cold kernel compiles).

Targeted suites (brief asked for `test_render_loop*.py` — no such files exist;
these are the render-loop-adjacent ones) plus `test_memory*.py` glob, env
declaration enforcement (touched environment.py), and the new file:

```
uv run -m pytest -q tests/unit_tests/test_memory*.py tests/unit_tests/test_render_batch_sizing.py \
    tests/unit_tests/test_batch_preparation_devices.py tests/unit_tests/test_environment.py \
    tests/unit_tests/test_preflight_overlap.py
85 passed, 3 warnings in 7.65s
```

(Not the whole unit suite in one process, as instructed.) Re-run after the
final formatting pass: `35 passed` for the env + overlap files.

### Lint (touched files only; the repo has pre-existing lint debt elsewhere)

```
uv run ruff check --no-fix algan/environment.py algan/render_loop.py algan/rendering/memory_model.py \
    algan/rendering/raytracing/scene_builder.py algan/settings/computing_settings.py \
    algan/utils/profiling_utils.py tests/unit_tests/test_preflight_overlap.py \
    scratch_perf/r3/ox/oom_retry_check.py scratch_perf/r3/ox/ab_preflight_overlap.py
All checks passed!
uv run ruff format --check <same set>
9 files already formatted
```

## Review findings fixed in the recovered patch

1. **Worker-side merge measured a peak beside the render** (`MERGE_TRACK_PEAK`
   resets the process counter mid-render and records a polluted value) — added
   the `track_peak` override, worker passes `False` (see Changes).
2. **Unreachable-in-production `int(inf)`**: on CPU `_gpu_merge_headroom_bytes`
   returns inf, so any future caller reaching the derate arithmetic off-CUDA
   would raise OverflowError; the seam harnesses patch headroom, and the gate
   still refuses non-CUDA configurations — noted, not code-changed, since the
   gate makes it unreachable (unit test pins the gate).
3. **Out-of-range env override silently dropped the derate** — now warns and
   falls back to the setting (render_loop.py:2490-2500).
4. **Untestable branch order in `fetch_batch`** — overlap gate consulted first
   (behavior-identical; enables the CPU e2e coverage).
5. Patch hunk against `scratch_perf/r2/ox/brief_preflight_overlap.md` mangled
   Windows paths into the historical brief (`D:\algan_wt_prep\CLAUDE.md` →
   `D:lgan_wt_prepCLAUDE.md`) — reverted; historical documents stay as they
   were.
6. Cosmetic: `if overlapped: pass / elif` chain collapsed into one guard
   (render_loop.py:933).

## Turnkey T4 run (operator)

One process per arm (daemon off everywhere):
`ALGAN_USE_DAEMON=0` is set inside the harness.

```
# warm-up render discarded, then timed arms:
ALGAN_PREFETCH_GPU_PREP=0 uv run python scratch_perf/r3/ox/ab_preflight_overlap.py PREVIEW off <override_mb>
ALGAN_PREFETCH_GPU_PREP=1 uv run python scratch_perf/r3/ox/ab_preflight_overlap.py PREVIEW on  <override_mb>
uv run python benchmarks/_video_diff.py scratch_perf/r3/ox/out/off_PREVIEW.mp4 scratch_perf/r3/ox/out/on_PREVIEW.mp4
```

No `--cpu-seams` on the T4. Pin
`available_memory_override` identically in both arms (argv 3) so windows are
comparable; the summary JSON next to each video carries
`preflight_sum_s`, `overlapped_fraction`, `worker_prep_sum_s` and `windows`.
Expectations to check against: overlapped_fraction ≈ (batches − 1)/batches;
identical windows; pixel diff 0; preflight time off the render thread roughly
the r2-measured 95% share; wall delta ≈ that share minus the un-overlappable
floor. Repeat the OOM force with a low override and confirm `status: rendered`.

## Explicitly NOT verified (this box has no GPU)

* Any CUDA execution of the overlapped builds themselves: `upload_primitive_source`
  + device projection, the GPU merge + STBVH build on the worker, and their
  byte-identity against render-thread preparation — argued by construction
  (identical inputs/math, same as the proven CPU-prewarm handover) and covered
  indirectly by the seam-patched runs, but never executed on a device.
* Whether Taichi tolerates concurrent kernel launches from the worker while the
  render thread launches (relevant only if a projected batch carries PN
  geometry, whose dice criterion is a Taichi kernel — e.g. the ImageMob in the
  nn scene). If it raises, `fetch_batch`'s except defers with a warning and the
  render completes serially; if it misbehaves silently, only a T4 pixel-diff
  would see it. The T4 A/B's 0-diff result is the acceptance gate here.
* The T4 performance numbers (wall delta, preflight fraction of PREVIEW, peak
  VRAM under the 0.6 derate) — operator's run above.
* The forced-OOM check with `available_memory_override` specifically (CUDA/MPS
  pin); its CPU analogue was forced instead.
* The fast-suite baseline discrepancy (pre-existing, evidenced above) — left
  for the owner to re-baseline deliberately.
* Interaction with `samples_per_pixel > 1` / Monte Carlo routing and with the
  calibration recorder (`set_auto_record`) running simultaneously with the
  overlap on: untested, and the counter-reset concern is why I'd keep them
  mutually exclusive until measured.
