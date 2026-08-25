# Task: attribute the sheet route's bounce loop per iteration, then cut the biggest real cost

**This brief replaces `brief_wavefront_host.md`, which was based on a wrong
reading of the profile. Read the correction in §0 before anything else.** Your
predecessor's instrumentation work is already in this working tree and is good;
you are continuing from it, not starting over.

Read `D:\algan\CLAUDE.md` first and obey it — the rules on `*_taichi.py` files,
on never calling `ti.init` yourself, on declaring every `ALGAN_` env var in
`algan/environment.py`, on reading settings live (`rt_settings.X` at call time),
on `ruff check --no-fix`, and on `ti.static` gates being baked at kernel compile
time (so anything a static gate controls needs **one process per arm**).

Run Python as `uv run python` from `D:\algan`. Set `ALGAN_USE_DAEMON=0` for
every script.

**Stay inside `D:\algan`.** This box has other git worktrees of the same repo
(`D:\algan_wt_prep`, `D:\algan_wt_sheet`, `D:\algan_wt_lab`). Other agents are
working in them **right now**. Do not read from, write to, copy into, or run
anything in any directory other than `D:\algan`. There is no pristine `master`
checkout available to you; where a verification step would want one, just say
you could not run it.

You share one GPU (NVIDIA GTX 1050, 4 GB) with those agents, so **a single pair
of wall-clock numbers is not a measurement**. Alternate arms in separate
processes, repeat at least 3 times each, and report medians — or argue from
launch counts and per-iteration ray counts, which contention does not distort.

## §0 The correction

The previous brief said `wavefront_loop` has "12 s of unattributed host-side
torch work" at 4K. **That was wrong.** The stage timers (`StageTimers.stage` in
`algan/utils/profiling_utils.py`) subtract *nested stages* from a stage's
exclusive column, but the **kernel** hooks (`_make_kernel_wrapper`) write
straight into `TIMERS.times[label]` without ever opening a stage — so a stage's
`excl` column silently **included every Taichi kernel it launched**.

Do the arithmetic on the T4 numbers and it falls out exactly:

```
wavefront_loop            incl 26.015   excl 13.218
  minus kernel: wavefront_shade            8.788
  minus kernel: wavefront_traverse_events  3.740
  minus kernel: compact_ray_slots          0.205
  = about 0.5 s of genuine host work over 30 frames
```

The same correction applies elsewhere: `raster: sparse resolve`'s 4.861 s "own"
time is `raster_shadow_trace` (4.470) + `sheet_resolve_shade` (0.302) + ~0.09 s
of host.

I have already fixed this in `profiling_utils.py` in this tree — the kernel
wrapper now calls `TIMERS.charge_kernel_to_parent(dt)`, which takes the one
bookkeeping step a nested stage takes on exit. **Verify my fix does what it
claims** (a stage's excl should now equal its wall time minus its child stages
*and* its kernels; a kernel launched outside any stage must still be reported
and must not corrupt anything) before you trust any number below it. If it is
wrong, fix it and say so.

A `torch.profiler` capture of 6 UHD frames on the T4 confirms the corrected
picture — the render is **GPU-bound**, and three Taichi kernels are 63% of all
CUDA time:

```
                                        self CUDA   share   calls (6 frames)
wavefront_shade                            1.710s   32.4%     48
raster_shadow_trace                        0.890s   16.9%      6
wavefront_traverse_events                  0.736s   14.0%     48
(all torch ops together)                  ~1.6s     ~30%
                                total CUDA 5.278s
```

48 launches over 6 frames is **8 bounce iterations per frame**.

## §1 What is already in this tree

`git diff` shows two modified files:

* `algan/utils/profiling_utils.py` — a `stage(name)` helper that is
  `TIMERS.stage` while the hooks are installed and a shared `nullcontext`
  otherwise (measured ~70 ns over a bare `with`, no allocation), plus my
  `charge_kernel_to_parent` fix;
* `algan/rendering/raytracing/tracer.py` — the `if use_raster:` branch's phases
  wrapped in inline `stage(...)` blocks.

Your predecessor verified the render is byte-identical with the instrumentation
present (15 frames, 0 differing channels, lossless encode). **Re-verify that
yourself** — do not inherit the claim — but you do not need to redo the design.

## §2 What to do

### Part A — per-iteration attribution (the deliverable I most want)

The interesting question is no longer "where does the host time go" but **what
the 8 bounce iterations per frame actually cost, one at a time**. Extend the
instrumentation so a profile reports, per bounce iteration `it`:

* the number of active rays entering it,
* the `wavefront_traverse_events` and `wavefront_shade` time for that iteration,
* the number of continuation rays it spawns.

Iteration index has to reach the stage label for this (e.g.
`wavefront:   - bounce 0 shade`), and the ray count must be read **without
adding a device sync that is not already there** — the compaction already
produces a count on the host each iteration; use that one, and say where it
comes from. Cap the labels at some small number of iterations and bucket the
rest as `bounce 8+` so a pathological scene cannot blow the table up.

Then report, for the nn scene at PREVIEW and at HD (not UHD — this card has
4 GB): the per-iteration table, and what fraction of `wavefront_shade`'s total
the **first** iteration holds. That single number decides what is worth doing
next, and nobody has measured it.

### Part B — cut the biggest real cost

Only after Part A, and let Part A choose. Do not pre-commit to an idea. The
candidates I can see, with what would make each one legitimate:

1. **If iterations 2..N are a long thin tail**, a deterministic throughput
   cutoff (terminate a continuation whose accumulated throughput cannot change
   any output byte) removes them. This is *not* byte-identical by construction,
   so it must be opt-in, and the deliverable is a measured **maximum channel
   difference** across a full render at several cutoff values, plus the wall
   time at each — not an assertion that it "looks the same". Find the largest
   cutoff whose max channel difference is 0, and report the curve past it.
2. **If the first iteration is everything**, the tail is irrelevant and the work
   is inside `wavefront_shade` itself: look at what it does per ray that could
   be hoisted, gated out by a `ti.static` template for scenes that do not need
   it, or skipped for rays whose material cannot reach that branch. The
   compile-time gating machinery already exists (`_frag_pid_mask`,
   `skip_unlit_normal`, the `has_*` templates) — extend it rather than inventing
   a parallel mechanism.
3. **The number of continuation rays.** `_split_pool_ratio` (tracer.py:391)
   shows a strong reflector spawns `ALGAN_ANALYTIC_AA_SECONDARY` (default 4)
   continuations per reflective fragment. If Part A shows the first iteration's
   ray count is dominated by that multiplier, then quantifying the quality/cost
   curve is itself a useful result — measure it, report it, and say what you
   think the default should be. Do not change the default.

Anything that can change output is opt-in behind a setting toggle following the
conventions in `algan/rendering/raytracing/settings.py` (env var declared in
`algan/environment.py`), default OFF, with the measured pixel difference stated.
Anything that cannot change output may default ON once you have proved
byte-identity.

**If Part A's honest conclusion is "this is irreducible kernel time and I found
nothing worth changing", write that down and stop.** The per-iteration table is
worth the task on its own; I would much rather have it plus an honest negative
than a change that does not help.

## §3 Verification (all required; quote the actual output)

- **Lossless render A/B: 0 differing pixels** for every change you claim is
  byte-identical. Render the nn scene at `HD`, pass
  `ffmpeg_params=["-c:v", "libx264rgb", "-qp", "0"]` (an H.264 re-encode turns
  single-channel differences into thousands of differing pixels), compare with
  `benchmarks/_video_diff.py`, and pin
  `SETTINGS.computing.available_memory_override` to the same value in both arms
  so the batch windows match — a window change legitimately moves pixels and
  would mask a regression. One process per arm.
- **The instrumentation must be free when off.** Show the unprofiled wall time
  is unchanged: alternate instrumented/uninstrumented in separate processes, at
  least 3 each, medians.
- `uv run -m pytest -q tests/unit_tests` — full unit suite.
- `uv run -m pytest -q --fast` — report the timing line. Its pixel-comparison
  test fails on this machine even on unmodified code (the committed CUDA
  baseline came from a different GPU), so report the failure and move on; you
  have no pristine checkout to confirm it against, which is fine — say so.
- `uv run ruff check --no-fix` and `uv run ruff format --check` on every file you
  touched.

## §4 Report

Write `scratch_perf/r2/ox/REPORT_wavefront_bounce.md`: whether my
`charge_kernel_to_parent` fix is correct; the corrected stage table for the nn
scene; **the per-iteration bounce table**; the first-iteration share; what you
changed and what it bought (with how you controlled for the other agents on this
GPU); and — explicitly — everything you did **not** verify.

## §5 Addendum (added after you started)

`tests/unit_tests/test_profiler_stage_attribution.py` is now in this tree: four
tests that pin the bookkeeping contract of `charge_kernel_to_parent` directly
against `StageTimers`, with no device involved. They pass. Use them as the
starting point for §0's "verify my fix" step rather than re-deriving it, and
extend them if you find a case they miss.
