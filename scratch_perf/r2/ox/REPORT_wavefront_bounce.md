# Report: attributing the sheet route's bounce loop per iteration

Task: `brief_wavefront_host2.md`. Machine: NVIDIA GeForce GTX 1050 (4 GB),
Windows, CUDA torch + Taichi 1.7.4, shared with other agents' renders (see
"Contamination control"). All timings are `RENDER_WALL`/stage-wall medians of
alternated separate-process runs unless stated otherwise. Everything ran in
`D:\algan` with `ALGAN_USE_DAEMON=0`.

## 1. Is `charge_kernel_to_parent` correct? Yes.

Three legs:

**Code reading.** `StageTimers.stage` on exit decrements `stack_level`, adds
its whole wall to `level_times[parent_level]`, and computes
`excl[name] += wall - level_times[own_level]`. A kernel launched while a stage
at running level L is open executes with `stack_level == L`, so
`charge_kernel_to_parent(dt)` doing `level_times[L] += dt` is *exactly* the
one bookkeeping step a nested child stage performs. A kernel launched outside
any stage charges `level_times[0]`, which no stage ever reads (a top-level
stage exits subtracting `level_times[1]`), and its time is still reported via
`TIMERS.times["kernel: ..."]` — reported, corrupting nothing. A side benefit:
the report's "accounted" sum no longer double-counts kernels (before the fix a
kernel sat inside its stage's excl AND appeared as its own excl row).

**Unit tests.** `tests/unit_tests/test_profiler_stage_attribution.py` (4
tests, shipped by the predecessor) pass; I additionally wrote
`tests/unit_tests/test_profiler_bounce_attribution.py` (7 tests) covering the
new `items=` plumbing and the bounce-table aggregation, including that the
capped bucket sorts last and that a row's continuation count comes from the
next row's rays-in.

```
4 passed, 3 warnings in 0.52s   (test_profiler_stage_attribution.py)
7 passed                        (test_profiler_bounce_attribution.py)
```

**Real-render arithmetic.** nn scene, HD, warm run
(`algan_profile_report_bounce_hd.txt`, RUN 2):

```
raster: sparse resolve     incl 9.285   excl 0.133
  minus kernel: raster_shadow_trace        8.511
  minus kernel: sheet_resolve_shade        0.642
  = 0.132  -- matches the stage's own excl column
wavefront_loop             incl 35.944  excl 0.052
  (all bounce/sheet/tile child stages + their kernels now subtracted;
   what remains is genuinely host-side tile bookkeeping)
```

This is precisely the correction §0 claimed: `sparse resolve`'s former 4.86 s
"own" time is now attributed to `raster_shadow_trace` + `sheet_resolve_shade`,
and `wavefront_loop` no longer reports its children's kernels as host work.

## 2. What I built

* `algan/utils/profiling_utils.py`: `StageTimers.stage(name, items=None)`
  accumulates a per-label work-unit total (`item_totals`); module-level
  `stage()` passes it through (still a shared `nullcontext` when the profiler
  is down); `format_report` gained a per-bounce table assembled from the
  `wavefront:   - bounce <i> <phase>` rows.
* `algan/rendering/raytracing/tracer.py`: the sheet route's drain loop
  (`_drain_sparse_secondary`) wraps `wavefront_traverse_events` and
  `wavefront_shade` in `wavefront:   - bounce <i> traverse|shade` stages,
  `items=na`. Labels are capped at indices 0..7; later iterations share
  `bounce 8+` (`_BOUNCE_STAGE_CAP`). Loop iteration 1 is labelled bounce 0 —
  primary visible-surface work is the sheet resolve, not a loop iteration.
* **No added device sync**: `na = int(active.numel())` reads shape metadata;
  the count itself was already read to the host by the compactor
  (`_ArenaRayCompactor.select` -> `count.item()`) or came from
  `compactor.initial`. An iteration's "continuations" column is simply the
  next row's rays-in, so the spawn count costs nothing extra; the last row
  shows `-` (the loop ended; nothing observed past it).

Byte-identity of all of this is verified in §5.

## 3. The deliverable: per-iteration tables

### nn scene, PREVIEW (704x396), warm run, end-to-end 5.43 s

```
Wavefront bounce iterations (sheet route; wall time incl. launch + sync):
   bounce  calls     rays in  traverse s   shade s continuations
        0      2       75734       0.111     0.247         22603
        1      2       22603       0.046     0.112         15679
        2      2       15679       0.029     0.092          1865
        3      2        1865       0.015     0.025            42
        4      2          42       0.010     0.016             2
        5      2           2       0.008     0.010             2
        6      2           2       0.008     0.011             2
        7      2           2       0.010     0.008             -
  total                          0.235     0.520
  first iteration holds 47.5% of all wavefront_shade time
```

### nn scene, HD (1920x1080), warm run, end-to-end 50.0 s

```
Wavefront bounce iterations (sheet route; wall time incl. launch + sync):
   bounce  calls     rays in  traverse s   shade s continuations
        0     15     1584902       2.521     4.797        484736
        1     15      484736       0.881     2.176        326470
        2     15      326470       1.159     1.827         37944
        3     15       37944       0.185     0.649           938
        4     15         938       0.185     0.226            21
        5     13          21       0.172     0.162            21
        6     13          21       0.090     0.097            21
        7     13          21       0.182     0.084             -
  total                          5.376    10.020
  first iteration holds 47.9% of all wavefront_shade time
```

Every one of the 114 `wavefront_shade` launches landed in a labelled
iteration (15x5 + 13x3 = 114), so the denominators above are the kernels'
whole totals. Caveats: the per-iteration split is stage WALL (includes
launch+sync overhead, which inflates the tiny tail iterations relatively);
cold runs were not used (JIT noise).

### Reading

1. **The first iteration holds ~48% of `wavefront_shade` at BOTH
   resolutions** (47.5% PREVIEW, 47.9% HD). The first three hold ~88%.
   This is the number nobody had measured; it kills any idea that the tail
   dominates, and equally any idea that iteration 0 is "everything".
2. **There IS a long thin tail, and it is pathological in a specific way**:
   21 rays never terminate — they ride every slice to the `max_iters` cap
   (bounces 5..7 process the same 21 rays again and again until the loop is
   cut). At HD that forces 3 extra traverse+shade launch pairs (~0.55 s wall)
   plus bounce 3/4's thin work; the whole tail from bounce 3 on is 2.03 s of
   50.0 s (**4.1% of end-to-end**, 13% of bounce-loop kernel time). At
   PREVIEW the tail from bounce 3 on is 0.12 s of 5.4 s (2.2%).
3. Bounce-0's ray count (1.58 M at HD) is set by the sheet resolve's
   continuation spawn, and — see §6 — is **not** governed by
   `ALGAN_ANALYTIC_AA_SECONDARY`.

## 4. Part B: what the data chose, and the honest result

The data did not cleanly select any candidate:

* **Candidate 1 (deterministic throughput cutoff)** — the tail exists but its
  entire ceiling is ~4% of end-to-end at HD (§3.2), it can change output bytes
  (so opt-in only), and the 21 never-dying rays would only be removed if their
  accumulated throughput is genuinely negligible — undiagnosed. I did not
  implement it: a toggle + cutoff-curve campaign to chase a bounded 4% did not
  survive contact with its own price tag. Numbers recorded here make the
  decision repeatable.
* **Candidate 2 (work inside `wavefront_shade`)** — the premise ("first
  iteration is everything") is half-true: 48%, not 90%+. Real, but it points
  at open-ended kernel surgery whose win caps around half of a kernel that is
  itself 20% of the render. Out of scope for this round; flagged as the
  direction a future kernel-level round should start from.
* **Candidate 3 (continuation multiplier)** — measured, and its premise is
  **false on this scene** (next section). Reported as the brief asks.

### The `ALGAN_ANALYTIC_AA_SECONDARY` (N) curve

Premise check first, using the new table itself (PREVIEW, bounce-0 rays-in):

| N | bounce-0 rays in | first-iter share |
|---|------------------|------------------|
| 1 | 75,723 | 49.4% |
| 2 | 75,727 | 49.9% |
| 4 (default) | 75,734 | 47.5% |

Invariant to N. On the nn scene the reflective pixels' continuations come from
the pass-through/glossy splitting machinery, not from sharp-reflection taps;
the multiplier is not what dominates the first iteration's ray count. That
alone settles the candidate: **do not touch the default expecting a first-
iteration win here.**

Full-curve measurement at HD anyway (lossless libx264rgb qp 0, memory override
pinned to 2.5 GiB in every arm so batch windows match, interleaved separate
processes, medians of 3):

| N | wall median (s) | runs (s) | max channel diff vs N=4 | pixels > tol 2, worst frame |
|---|---|---|---|---|
| 1 | 74.4 | 67.9 / 74.4 / 108.2 | **118** | 1782 of 2.07 M (0.086%) |
| 2 | 91.0 | 114.2 / 86.9 / 91.0 | 57 | 60 (0.003%) |
| 4 (default) | 95.3 | 84.3 / 95.3 / 109.2 | – | – |
| 8 | 218.0 | 239.2 / 147.5 / 218.0 | 46 | 64 (0.003%) |

Reading:

* Cost is dominated not by extra rays but by **pool fragmentation**:
  `_split_pool_ratio` reserves N+1 slots per primary whenever the batch
  splits, so tiles shrink roughly N-fold and every per-tile phase
  (`shade_sparse_raster_coverage`, drain, composite) repeats proportionally —
  hence N=8 catastrophic (+129%) although bounce-0's ray count barely moves.
* Quality does not order cleanly with N here: going UP to 8 moves about as
  many pixels as going DOWN to 2 (≈50–64 px/frame, peak 46–57 channels) —
  this scene's reflections don't visibly converge past 4. N=1 is where real
  damage appears (peak 118, 0.086% of the frame).
* **Recommendation: keep the default at 4** (unchanged, as instructed). It is
  the knee of this curve. N=2 is defensible for scenes whose reflections are
  peripheral (−4.5% wall, sub-tolerance pixel count); N=8 is not viable on
  4 GB cards. Sharp-MIRROR-heavy scenes would shift the quality leg of this
  curve and were not measured.

## 5. Verification (quotes)

**Byte-identity, lossless H.264 (`libx264rgb -qp 0`), HD, 15 frames, diffed
with `benchmarks/_video_diff.py`, memory override pinned identically in all
arms:**

* instrumentation ON vs OFF, three interleaved rounds — all exactly zero:
```
== prof4 vs off4 (round 1) ==   worst channel diff: 0    0 of 15 frames affected
== prof4 vs off4 (round 2) ==   worst channel diff: 0    0 of 15 frames affected
== prof4 vs off4 (round 3) ==   worst channel diff: 0    0 of 15 frames affected
```
* working tree vs HEAD (no pristine master checkout exists — see §7; HEAD was
  reproduced in place by swapping the ONLY two tracked files that differ,
  rendering, and restoring):
```
round 1: worst channel diff: 0    0 of 15 frames affected
round 2: worst channel diff: 0    0 of 15 frames affected
round 3: worst channel diff: 1    0 of 15 frames affected   (<=2: the documented
        re-windowed rate-function rounding; within the suites' tolerance)
```
* cross-run reproducibility of the pinned baseline (off4 r1 vs r2, r1 vs r3):
  worst channel diff 0, both.

**Instrumentation free when off** — alternated separate processes, HEAD-files
vs tree-files arms (profiler down in both), two independent sessions:

```
session 2 (GPU busy, tighter pairing): head4 median 135.8s   off4 median 136.1s  (+0.2%)
session 1:                             head4 median 113.7s   off4 median 130.7s
                                        (one off4 run spiked to 197.6s on tenant load)
```
Session 2 is the controlled comparison: +0.2%, i.e. the nullcontext call sites
are free. Instrumentation ON vs OFF medians were 99.7 s vs 95.3 s (+4.6%,
ranges overlap heavily — treat as an upper bound under unknown tenant load;
syncing stages necessarily cost something while profiling, and profiling runs
are never wall-clock references).

**Test suites:**

```
uv run -m pytest -q tests/unit_tests
2077 passed, 131 skipped, 172 warnings in 3309.41s (0:55:09)

uv run -m pytest -q --fast     (run 1, cold-ish)
fast suite: 304s of its 75s budget (405%) -- over budget
276 passed, 1940 deselected

uv run -m pytest -q --fast     (run 2, steady state)
fast suite: 125s of its 75s budget (166%) -- over budget
276 passed, 1940 deselected
```

Note: the brief expected `tests/fast/test_fast_render.py::
test_the_fast_scene_renders_and_matches_its_baseline` to fail on this machine.
It **passed** (twice). I have no pristine checkout to confirm why against, but
this tree's history contains `df07859d "Rebaselined CUDA"` — newer than the
brief's assumption. The budget overrun (166% steady-state) is consistent with
a shared GPU and was not investigated further.

**Ruff** (`--no-fix`; CI-enforced `format --check` included):

```
algan/utils/profiling_utils.py              check: clean   format: ok
algan/rendering/raytracing/tracer.py        check: I001 at :96 only — pre-existing
                                            (verified present in HEAD via
                                            git show + ruff on a temp copy)
tests/unit_tests/test_profiler_bounce_attribution.py   check+format: clean
scratch_perf/r2/ox/{render_nn_hd,alternate_hd,alternate_head,bounce_profile}.py
                                            format: clean; check: one deliberate
                                            E402 in each render script (Algan
                                            settings must precede `from algan
                                            import *`; the repo's own A/B
                                            scripts share the pattern)
```

**Contamination control** (other agents share this GPU): every comparison in
this report interleaves arms as separate processes (schedules in
`hd_schedule.json` / `alternate_head.py`), takes ≥3 runs per arm, reports
medians, and pins `SETTINGS.computing.available_memory_override` so windows
cannot drift with tenants. Cross-round baseline renders were byte-identical
(above), which is direct evidence window-drift did not contaminate the pixel
comparisons. The one 197.6 s outlier is called out rather than hidden.

## 6. What I changed, and what it bought

Changed (all output-neutral, proven above):

* `profiling_utils.py`: kept and verified the predecessor's
  `charge_kernel_to_parent` fix; added `items=` work-unit accounting, the
  bounce-table section of the report, and guard tests.
* `tracer.py`: per-bounce-iteration stage labels + ray counts in the sheet
  route's drain loop (`_BOUNCE_STAGE_CAP = 8`).

What it bought: **the per-iteration table itself** (§3) — the first measured
answer to "what does each bounce cost", the 48% first-iteration share, the
discovery that a 21-ray set rides every slice to the iteration cap, and the
N-curve that falsifies the cheapest-looking optimization before anyone spends
a day on it. Performance-wise the round's honest yield is the negative result:
no code change in this report makes any render faster, and the two plausible
toggles (throughput cutoff; N retune) are priced here at ≤4% and ≈0%
respectively for this content.

Where a REAL next win probably lives, from the corrected stage table (HD
warm): `raster_shadow_trace` 8.5 s (17%) is the single largest kernel and none
of the three candidates touches it; then bounce-0 `wavefront_shade` 4.8 s
(9.6%), then the sparse discovery/resolve chain. Also worth noting for
readers of the full report: the "unaccounted" line can go NEGATIVE (−32% on
the PREVIEW run) because prefetch-worker stages legitimately overlap render-
thread wall; that is the TLS concurrency model working, not a bug.

## 7. Explicitly not verified

* **No pristine-master render.** The brief forbids leaving `D:\algan` and the
  other worktrees are in use, so the HEAD comparison was made by swapping the
  two differing tracked files in place. That reproduces HEAD *for those two
  files*; any other divergence between HEAD and master is untested here.
* The +4.6% instrumented-vs-not median is an upper bound under unknown tenant
  load; I did not get a quiet-GPU window to tighten it.
* Per-iteration GPU time (Taichi kernel profiler) is not split by iteration —
  the bounce table is wall-based (launch+sync inclusive). A CUPTI/nvprof pass
  could refine the small-iteration overhead figures.
* The ±1-channel difference in HEAD-vs-tree round 3 was not bisected; it sits
  inside the documented ≤2 re-window rounding and within suite tolerance.
* The source of the 21 never-dying rays (which material / ray type) was not
  diagnosed.
* Only the nn scene was measured. Sharp-mirror-heavy scenes will move the
  N-curve's quality leg; the tail-ceiling estimate (≤4%) is scene-specific.
* `ANALYTIC_AA_SECONDARY_MIN_ENERGY` interactions with the curve were not
  measured.
* The `--fast` budget overrun (166% steady state) was not investigated.
* Full-suite (`pytest -q`, ~12 min nominal; 55 min actual for unit_tests alone
  on this busy box) render suites beyond `--fast` were not run; CI-equivalent
  coverage came from `tests/unit_tests` + `tests/fast`.

Artifacts: `algan_profile_report_bounce_{preview,hd}.txt`,
`algan_outputs/profiling/profilingbounce_*_run*.mp4`,
`scratch_perf/r2/ox/ab/hd_*.mp4` (15 + 12 lossless renders),
`scratch_perf/r2/ox/{bounce_profile,render_nn_hd,alternate_hd,alternate_head}.py`,
`scratch_perf/r2/ox/head_medians_{x,y}.txt`, `scratch_perf/r2/ox/hd_medians.txt`.
