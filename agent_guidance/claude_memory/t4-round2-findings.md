---
name: t4-round2-findings
description: "Round 2 of the T4 optimization work (2026-08-26): the profiler bug that invalidated round 1's plan, what the UHD render actually costs, and what shipped"
metadata:
  node_type: memory
  type: project
  modified: 2026-08-26
---

Round 2 ran on the user's desktop (Windows, GTX 1050 4 GB) with **Kaggle T4
notebooks as the measurement box** (`[[kaggle-t4-measurement]]`) and three
Ox Alpha agents working in parallel git worktrees. Integration commit
`9f3fdb90` on branch `perf/r2-lab`. Supersedes `[[t4-perf-next-steps]]`.

## The finding that matters most: the profiler was lying

`StageTimers.stage` subtracts nested **stages** from a stage's `excl` column,
but the **kernel** hooks (`_make_kernel_wrapper`) write straight into
`TIMERS.times` without ever opening a stage. So every `excl` in every report was
inflated by the kernels that stage launched.

That is what produced round 1's headline conclusion — "`wavefront_loop` has
13.2 s of unattributed host work at 4K, 44% of the render". Of that 13.2 s,
**12.5 s was `wavefront_shade` + `wavefront_traverse_events`**, both listed by
name in the same table. Real host work there: ~0.5 s over 30 frames. The same
correction turns `raster: sparse resolve`'s 4.86 s "own" time into
`raster_shadow_trace` (4.47) + `sheet_resolve_shade` (0.30) + 0.09 s of host.

Fixed by `TIMERS.charge_kernel_to_parent(dt)`, which takes the one bookkeeping
step a nested stage takes on exit.
`tests/unit_tests/test_profiler_stage_attribution.py` pins the contract with no
device involved. After the fix a real profile reads `wavefront_loop` excl
**0.026 s**.

**Do not read an `excl` column from any report written before `9f3fdb90`
without subtracting the kernels underneath it by hand.**

## What the UHD render actually costs

Ablations on the Kaggle T4, `nn_scene_UHD.py`, 30 frames at 3840x2160, warm
RUN 2, **one process per arm** (a `ti.static` gate is baked at compile time):

| arm | warm | vs base |
|---|---|---|
| base | 27.92 s | — |
| `shadows=False` | **16.25 s** | **-42%** |
| `max_bounces=0` | 16.40 s | -41% |
| `max_bounces=1` | 22.02 s | -21% |
| `ALGAN_ANALYTIC_AA_SECONDARY=1` | 24.97 s | -11% |

At PREVIEW on the T4: base 6.09 s, `shadows=False` 5.07 s (-17%),
`ALGAN_BVH_REFIT=0` 6.38 s (**refit is a win — do not disable**),
`ALGAN_PREFETCH_BATCHES=0` 7.17 s (**prefetch is already earning +18%**).

Corrected budget of the 26.0 s render: `wavefront_shade` 8.79 s (29%), the
sheet route's host-torch passes ~5.2 s (17%), `raster_shadow_trace` 4.47 s
(15%), `wavefront_traverse_events` 3.74 s (13%), `raster_tri_count/write`
1.36 s (5%), prep + arena preflight ~1.6 s (5%). A `torch.profiler` capture
agrees: 63% of CUDA time is those three kernels.

**PREVIEW is preparation-bound, UHD is render-bound.** At PREVIEW prep (2.32 s)
+ preflight (1.52 s) + render (3.25 s) ≈ the whole 6.25 s: they are nearly
serial, because the arena preflight deliberately runs on the render thread.

## Per-bounce attribution (new instrumentation, shipped)

The sheet route's bounce loop now reports rays-in, traverse/shade time and
continuations spawned per iteration. On the nn scene the **first bounce holds
~48% of all `wavefront_shade` time at both PREVIEW and HD**, the first three
hold ~88%, and everything past bounce 3 is ~4% of end-to-end. So a throughput
cutoff has a small ceiling and the work is in iteration 0. Bounce 0 takes
1.58 M rays for a 2.07 M-pixel frame, and that count is **invariant** to
`ALGAN_ANALYTIC_AA_SECONDARY` (1 vs 2 vs 4) — they are not sharp-reflection
taps. Also: 21 rays never terminate and ride to `max_iters`, costing 3 extra
launch pairs per frame-part; undiagnosed, possibly a bug.

`ALGAN_ANALYTIC_AA_SECONDARY` curve at HD: N=8 is +129% wall (pool
fragmentation — `_split_pool_ratio` reserves N+1 slots per primary, so tiles
shrink N-fold and every per-tile phase repeats), N=1 is the only value that
visibly damages quality (peak channel diff 118). **Keep the default at 4.**

## The shadow cull: a measured no-op — do not re-derive it

Shadows are 42% of the UHD render, so "cull shadow fans whose receiver faces
away from the light" looks like the obvious win. **It is already there**: commit
`f142f72d` put a per-sample horizon guard into both trace sites
(`raster_taichi.py:3047`, `wavefront_kernels_taichi.py:2984`), so those BVH
marches are already skipped. Implementing the whole-fan cull on top removed 9%
of *entered fans* but **0% of marched rays** — one ray over a whole video — and
kernel self-time was flat.

The corollary is the planning-relevant part: **the 42% is not waste.** Those
rays are marched because they contribute. Cutting shadow cost needs a different
idea (a cheaper occlusion structure, ray coherence, fewer events), not a better
cull. The patch and a full stage-by-stage soundness audit — which establishes
that lambert/phong/standard/physical/toon carry `max(N·L, 0)` on every
vis-multiplied term, that **manim does not**, and that `event_snrm` holds the
already-oriented shading normal — are kept at `scratch_perf/r2/patches/` and
`scratch_perf/r2/ox/REPORT_shadow_facing_cull.md`.

## What shipped in `9f3fdb90`

* the profiler attribution fix + its unit test;
* per-bounce instrumentation of the sheet route's drain loop, via a
  `profiling_utils.stage()` helper that is a shared `nullcontext` while the
  profiler is down (~70 ns/site, no allocation);
* **P13**, the batched `NeuralNetMLP` idle updater (`ALGAN_BATCHED_IDLE_UPDATER`,
  default on): four per-mob Python loops over 15 neurons and 80 synapses became
  three timeline writes. Batch prep **-21.8%**, timeline `get` 2841 -> 1436 and
  `modify` 258 -> 6 per batch. Falls back to the loops on any structure it does
  not recognise, and every read happens before any write so the fallback is
  clean;
* `copy=False` on engine-internal read paths in `shapes_3d` / `surface` /
  `render_loop` that only feed out-of-place arithmetic (the public getters still
  copy — callers may mutate what they get). `render_loop` was cloning **every
  surface's whole grid once per batch just to read its `.shape`**.

Verified byte-identical: nn PREVIEW rendered losslessly against HEAD, 50/50
frames, worst channel diff **0**; `--fast` 277 passed; and the **full unit
suite passes -- 2077 passed, 132 skipped, 0 failed**, the same count master
reports. Run it with `scratch_perf/r2/run_suite_chunked.sh`: a single
`pytest -q tests/unit_tests` grows to ~2.2 GB and forks a ~1.4 GB render
child, which OOMs this 16 GB box once a browser and an agent are up; 8 files
per interpreter keeps free RAM flat. Locally that render went
105.4 s -> 83.7 s (1.26x — a bigger share on a 4 GB card, which uses 10 batches
where the T4 uses 3).

**Not measured on the T4** — the integrated tree never reached Kaggle, because
the payload transport was still being solved. That is the first thing to do next
session.
