# T4 render throughput — plan of record

The reference machine for this work is a **Tesla T4** (Turing, sm_75, 15.6 GB),
reached through Kaggle notebooks. `DESIGN_optimization_targets.md` remains the
plan of record for algan's general render performance and its T1–T7 / P1–P13
numbering; **this document owns the T4 line of work only**, because its rankings
come from a different machine, a different scene and a different bottleneck mix.

Read this file and `agent_guidance/claude_memory/t4-round2-findings.md` before
picking anything up.

---

## 0. How to measure

**Scenes.** `benchmarks/performance/nn_scene_UHD.py` (30 frames at 3840x2160)
and `benchmarks/performance/nn_scene_PREVIEW.py` (50 frames). Both run
`profile_scene(..., runs=2)`; **read RUN 2**, the warm one. RUN 1 includes the
Taichi JIT and is a cold-start measurement, not a throughput one.

**Ablations.** `benchmarks/performance/nn_ablation.py <arm> [QUALITY]` renders
the same scene with one lever moved. Arms: `base`, `noshadow`, `sec1`, `b1`,
`b0`. **One process per arm, always** — a `ti.static` gate is baked when the
kernel compiles, so a second arm in the same process silently reuses the first
arm's code and reports its numbers as its own.

**The T4 box.** `scratch_perf/kaggle/make_notebook.py` builds the notebook;
`agent_guidance/claude_memory/kaggle-t4-measurement.md` is the operating manual,
including the two-batch-session limit, the payload-size ceiling and the
snapshot/`--from-tag` mechanism. If the branch can be pushed to GitHub, pass
`--branch` and none of the payload machinery is needed.

**Reading a profile.** Before `9f3fdb90` a stage's `excl` column included every
Taichi kernel that stage launched, because the kernel hooks bypassed the stage
stack. Any older report must have its kernels subtracted by hand. The fix is
`TIMERS.charge_kernel_to_parent`; `tests/unit_tests/test_profiler_stage_attribution.py`
pins it.

**Wall clock is unreliable on the dev box** (a 4 GB GTX 1050 shared with agents).
Lead with counts — rays, launches, calls — which contention cannot distort, and
give medians of alternating separate-process runs when seconds are unavoidable.

---

## 1. Baselines

Kaggle T4, master @ `95271dac`, warm RUN 2:

| scene | warm | cold |
|---|---|---|
| `nn_scene_UHD.py` | **29.90 s** | 85.85 s |
| `nn_scene_PREVIEW.py` | **6.25 s** | 32.72 s |

Round 1 (branch `perf/t4-nn-scene-throughput`, merged at `fc100cd8`) took these
from 50.0 s and 36.5 s on a Colab T4; peak VRAM 8.4 -> 6.5 GB (UHD) and 6.2 GB
(PREVIEW).

Round 2 integrated tree (branch `claude/algan-t4-optimization-b5a10b` @
`8773ae93`, 2026-08-26, log at `scratch_perf/r3/t4_r2t4verify_run.log`):

| scene | warm | vs master | cold |
|---|---|---|---|
| `nn_scene_UHD.py` | **27.81 s** | **-7.0%** | 84.25 s |
| `nn_scene_PREVIEW.py` | **5.86 s** | **-6.2%** | 32.46 s |

As predicted, the dev box's 1.26x did not transfer — the change is prep-side
and the T4 runs 2-3 batches where the 4 GB card runs 10. At PREVIEW the warm
split is now render 3.11 s / prep 1.77 s / preflight 1.48 s (prep down from
2.32 s — P13 at work), still nearly serial.

---

## 2. Where the time goes

### UHD is render-bound

Corrected budget of the 26.0 s render inside `save_video`:

| item | s | share | kind |
|---|---|---|---|
| `wavefront_shade` | 8.79 | 29% | Taichi kernel, 8 bounce iterations/frame |
| sheet route's host-torch passes | ~5.2 | 17% | `compact_sheets` 2.3, `window pairs` 1.14, `lexsort` 0.91, `shade class` 0.52, `fragment sort` 0.41, `prim split` 0.36 |
| `raster_shadow_trace` | 4.47 | 15% | Taichi kernel, 1 launch/frame |
| `wavefront_traverse_events` | 3.74 | 13% | Taichi kernel |
| `raster_tri_count` + `raster_tri_write` | 1.36 | 5% | Taichi kernels |
| prep + arena preflight | ~1.6 | 5% | |
| post-process | 0.71 | 2% | |

A `torch.profiler` capture of 6 frames agrees: 5.278 s of CUDA time, 63% of it
in those three kernels, ~30% in torch ops (`copy_` 213 ms over 8261 calls,
sorts 211 ms, `index_select` 175, `gather` 172, `cat` 167, `stack` 154,
`fill_` 116).

### PREVIEW is preparation-bound

```
ray traced render total            5   3.245 s  (52%)
Scene.get_batch_of_primitives      3   2.322 s  (37%)   -- on the prefetch worker
arena preflight (batch)            3   1.519 s  (24%)   -- on the RENDER thread, by design
AttributeTimeline.get          22702   0.807 s  (13%)
```

2.32 + 3.25 + 1.52 ≈ 6.25: prep, preflight and render are nearly serial. The
prefetch pipeline already hides `get_batch_of_primitives` (turning it off costs
+18%), but `_prepared_batch_fits_render_arena` deliberately runs the GPU
projection and merge on the render thread so their CUDA peak can be measured
without a concurrent render polluting the counter.

### Ablations (T4, UHD, warm, one process per arm)

| arm | warm | vs base | what it removes |
|---|---|---|---|
| base | 27.92 s | — | |
| `shadows=False` | **16.25 s** | **-42%** | every shadow ray, both call sites |
| `max_bounces=0` | 16.40 s | -41% | the whole continuation bounce loop |
| `max_bounces=1` | 22.02 s | -21% | bounces 2..8 |
| `ALGAN_ANALYTIC_AA_SECONDARY=1` | 24.97 s | -11% | 3 of 4 continuations per reflective fragment |

`raster_shadow_trace` alone is ~4.5 s of the 11.67 s that `noshadow` saves, so
roughly **7 s is shadow rays fired from inside the bounce loop** — about 60% of
that loop's own 11.5 s. Shadow work pays twice.

At PREVIEW: base 6.09, `noshadow` 5.07 (-17%), `ALGAN_BVH_REFIT=0` 6.38
(**refit is a win**), `ALGAN_PREFETCH_BATCHES=0` 7.17 (+18%).

### Per-bounce attribution

The sheet route's drain loop is instrumented per iteration (rays in,
traverse/shade time, continuations spawned):

```
nn scene, HD, warm
   bounce  calls     rays in  traverse s   shade s continuations
        0     15     1584902       2.521     4.797        484736
        1     15      484736       0.881     2.176        326470
        2     15      326470       1.159     1.827          37944
        3     15       37944       0.185     0.649            938
        4     15         938       0.185     0.226             21
        5..7  13           21    ~0.44 total ~0.34 total         -
  first iteration holds 47.9% of all wavefront_shade time
```

* **Bounce 0 is ~48% at both PREVIEW and HD; the first three are ~88%.** The
  tail past bounce 3 is 4% of end-to-end, so a throughput cutoff has a small
  ceiling.
* 1.58 M rays for a 2.07 M-pixel frame, and the count is **invariant** to
  `ALGAN_ANALYTIC_AA_SECONDARY` — these are not sharp-reflection taps.
* ~~**21 rays never terminate**, riding to `max_iters`~~ — **that claim was
  wrong, and the cohort is diagnosed** (2026-08-26,
  `scratch_perf/r3/ox/REPORT_immortal_rays.md`). They ride to the **bounce
  cap**, never `max_iters` — the bounce table's last-row `-` cannot show
  continuations and was misread as "still alive". The cohort (30 rays at
  PREVIEW with identical counts on the T4 and on CPU; ~2,855 entering
  bounce 7 at UHD) is sub-`MIN_WEIGHT` transport kept alive by a
  control-flow gap: every in-place reflection branch `break`s past the
  weight-floor exit, and the post-loop exits deliberately exclude bounced
  rays — so a sub-floor ray that reflects gets its full 8 bounces while one
  that pass-throughs retires immediately. Image correct; bounces 5-7 exist
  almost solely for these rays (~1.7% of the UHD render). One-line fix
  proposed in the report §7, not yet implemented.

---

## 3. Closed — do not re-derive

**Receiver-facing shadow cull.** Culling shadow fans whose receiver faces away
from the light looks like the obvious answer to "shadows are 42%". It is
**already implemented** as a per-sample horizon guard (`f142f72d`,
`raster_taichi.py:3047` and `wavefront_kernels_taichi.py:2984`). Adding the
whole-fan cull on top removes 9% of *entered fans* and **0% of marched rays** —
one ray over a whole video — with kernel self-time flat. Patch and the
stage-by-stage soundness audit are kept at `scratch_perf/r2/patches/` and
`scratch_perf/r2/ox/REPORT_shadow_facing_cull.md`; the audit is worth reading
before touching shading (it establishes which stages carry `max(N·L, 0)` on
every vis-multiplied term, that **manim does not**, and that `event_snrm` holds
the already-oriented shading normal).

**The corollary:** the 42% is not waste. Those rays are marched because they
contribute. Cutting shadow cost needs a cheaper occlusion structure, ray
coherence, or fewer events — not a better cull.

**Continuation-ray spawn floor.** The bounce loop is 41% of the UHD render and
bounce 0 is ~48% of `wavefront_shade`, so cutting the 1.58 M continuations it
starts with looks like the biggest single lever. Classified at HD, they are
**65% glossy-prefilter rows and 35% pooled reflections** — zero refraction, zero
in-place bounces — and **every one is born at full surface visibility**, so none
are being shadowed away. An opt-in `ALGAN_SPECULAR_SPAWN_FLOOR` was built and
its curve measured (counts at HD, pixel differences at PREVIEW against a
lossless base with the memory override pinned):

| floor | continuations born | worst channel diff | pixels > tol 2 |
|---|---|---|---|
| 0 (default) | 1,584,902 | — | — |
| 0.03 | 1,115,136 (-30%) | 49 | 1.30% |
| 0.06 | 1,051,374 (-34%) | 52 | 1.47% |
| 0.12 | 141,137 (-91%) | 140 | 2.91% |

**No floor is byte-identical**, and at 0.12 the pooled-reflection class goes to
zero outright rather than being trimmed. So this is a **quality knob, not an
optimization** — the rays are real reflection work. Ships default OFF (0.0) if
it ships at all; the curve above is the thing to keep. Patch:
`scratch_perf/r2/patches/ox_continuation_rays_WIP.patch`.

**The corollary, and it now applies twice.** Both of the two biggest apparent
levers — shadows (42%) and continuations (41%) — turned out to be *real work
that contributes to the image*, not waste awaiting a cull. Neither can be cut
byte-identically. What is left that can be is the **host-side** cost: the sheet
route's torch passes, and the serial arena preflight.

**`ALGAN_ANALYTIC_AA_SECONDARY`.** Measured across 1/2/4/8 at HD. N=8 costs
+129% wall through pool fragmentation (`_split_pool_ratio` reserves N+1 slots
per primary, so tiles shrink N-fold and every per-tile phase repeats); N=1 is
the only value that visibly damages quality (peak channel diff 118, 0.086% of
the frame). **Keep the default at 4.**

**`ALGAN_BVH_REFIT=0` and `ALGAN_PREFETCH_BATCHES=0`** are both losses. The
existing defaults are right.

---

## 4. Shipped

### Round 1 — `perf/t4-nn-scene-throughput`, merged at `fc100cd8`

Wide attributes (textures) materialize on the render device; per-device batch
budgets; NVENC encoder selection; the glossy prefilter's per-tile bounce loop;
reproducible batch windows; window release after primitives are built. PREVIEW
36.5 -> 7.7 s, UHD 50.0 -> ~31 s on a Colab T4.

### Round 2 — `perf/r2-lab` @ `9f3fdb90`

* **Profiler attribution fix.** A stage's `excl` no longer includes the kernels
  it launched. This invalidated round 1's remaining plan, which had been built
  on "13.2 s of unattributed host work in `wavefront_loop`" — 12.5 s of which
  was two kernels already named in the same table.
* **Per-bounce instrumentation** of the sheet route's drain loop, through a
  `profiling_utils.stage()` helper that is a shared `nullcontext` when the
  profiler is down (~70 ns/site, no allocation, byte-identical renders).
* **P13, the batched idle updater** (`ALGAN_BATCHED_IDLE_UPDATER`, default on):
  batch prep **-21.8%**, timeline `get` 2841 -> 1436 and `modify` 258 -> 6 per
  batch.
* **Dead-clone removal** on engine-internal read paths (`shapes_3d`, `surface`,
  `render_loop`). `render_loop` had been cloning every surface's whole grid once
  per batch to read its `.shape`.

Byte-identical: nn PREVIEW rendered losslessly against HEAD, 50/50 frames, worst
channel diff 0; `--fast` 277 passed. Locally 105.4 s -> 83.7 s (1.26x; a bigger
share on a 4 GB card, which uses 10 batches where the T4 uses 3).

### Open verification debt

Two things this round did not finish, and the next session should:

* ~~**The integrated tree has never run on the T4.**~~ **Closed 2026-08-26.**
  Measured via the `--branch` mechanism (now actually implemented in
  `make_notebook.py` — it had accepted the flag and ignored it): UHD
  29.90 -> **27.81 s** warm (-7.0%), PREVIEW 6.25 -> **5.86 s** warm (-6.2%).
  See §1. The dev box's 1.26x was indeed a batch-count artifact.
* ~~`tests/unit_tests` has not passed cleanly on the integrated tree.~~
  **Closed.** It now passes: **2077 passed, 132 skipped, 0 failed**, the same
  count master reports. The earlier single failure in `test_glossy_prefilter.py`
  was contention, exactly as suspected — that file contains a cross-process
  determinism test, and free VRAM at job start decides the arena size and hence
  tile sizes. Run on a quiet box it passes.

  **How to run it here without OOMing the machine:**
  `scratch_perf/r2/run_suite_chunked.sh`. A single
  `pytest -q tests/unit_tests` grows its own interpreter to ~2.2 GB (torch +
  taichi + accumulated fixtures) and then forks a ~1.4 GB render child — 3.6 GB
  on a 16 GB box with ~4 GB free once a browser and an agent are running, which
  took this machine to the edge of OOM twice. Running 8 files per interpreter
  keeps free RAM flat (measured 3.0 -> 6.1 GB across the 14 chunks, no downward
  drift) and costs only the repeated import. Total wall time 45 min, 28 of which
  is the one render-heavy chunk.

**Not yet measured on the T4** — the integrated tree never reached Kaggle
because the payload transport was still being solved. Do this first.

---

## 5. What is left, in order

1. **`wavefront_shade`'s first iteration** — 29% of the UHD render, ~48% of it
   in bounce 0, and roughly 60% of the bounce loop is inline shadow rays. The
   open questions are *what those 1.58 M continuations are* (pass-through behind
   partial coverage? Fresnel sheen?) and whether any of them provably cannot
   contribute. A ray that provably contributes nothing can be dropped
   byte-identically; anything less is opt-in with a measured max channel diff.
2. **The sheet route's host-torch passes** — ~5.2 s (17%) at UHD. A previous
   round kernelised three of them (`scratch_perf/ox/REPORT_sheet_chain.md`) and
   left the solid-shell ceiling block as the largest remaining. `window pairs`
   (1.14 s) has never been looked at. The sorts are cuB-backed and should stay;
   what is fair game is the segment construction and gathers around them, and
   the `cat`/`stack`/`copy_` family, which is the biggest non-sort op group in
   the torch profile.
3. **The arena preflight — the best remaining host-side prize, and now
   quantified.** It is 24% of PREVIEW on the T4 and runs serial on the render
   thread by design, so its CUDA peak can be measured uncontended. Round 2
   measured the two ratios that decide whether overlapping it is worth building
   (dev box, PREVIEW at a 2 GiB override and HD; **ratios transfer, the shares
   do not** — those runs used 25 and 150 batches where the T4 uses 3):

   * **the un-overlappable floor is 0-6% of the preflight.** The first batch of
     a job cannot hide behind anything, and it costs 1-2 s of a 27-50 s
     preflight sum at PREVIEW, 1.2-6.2 s of 415-578 s at HD. So ~95% of it is
     in principle overlappable.
   * **the first probe succeeds 100% of the time.** The frame window chosen
     before the exact preflight is the final answer in every batch measured,
     which kills the main objection — that speculating on the wrong window
     wastes the projection and merge. (Contrast the recorded negative result
     for *speculative* prefetch of the successor batch, +15%.)

   Inside it, `merge collections + build BVHs` is the bulk (295 s of 422 s at
   HD), of which `refit-BVH build` is 199 s; `project_to_screen (prewarm)` is
   125 s. Work in progress at `scratch_perf/r2/patches/ox_preflight_overlap_WIP.patch`
   (467 lines: an `_rt_prep_overlapped` flag, a predictor-calibration gate so
   the first batches still measure their peak on the render thread, and a
   headroom derate for the concurrent render). Unverified — it was stopped
   mid-implementation. The OOM retry must keep working; force it and show the
   render still completes.
4. **Shadow ray cost by another route** — see §3 for why the obvious cull is
   closed.
5. ~~**The 21 immortal rays**~~ **Diagnosed 2026-08-26**
   (`scratch_perf/r3/ox/REPORT_immortal_rays.md`): not immortal, not a
   rendering bug — a missing weight-floor exit for rays whose last hit
   bounced in place (see §2's corrected entry). What remains is the fix:
   the report's §7 one-liner (the floor check in `wavefront_shade`'s
   post-loop block, retiring as completion, not truncation), behind a
   toggle, with an A/B parity render — the report bounds the dropped
   transport at under half a u8 LSB but that is an envelope argument, not
   a measurement.
6. **Cold start** — 85.85 s at UHD against 29.90 s warm, almost all Taichi JIT.
   Amortized for a long video, so it ranks below everything above, but it is the
   whole story for a short one.
