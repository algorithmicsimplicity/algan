# Optimization targets

Ranked candidates for further speeding up Algan, measured against the project's
reference workload: `s05_learning_to_program_setup.py` at `LD` (864x486, 15 fps,
3883 frames), profiled with `algan.utils.profiling_utils`.

> **Read the percentages, not the seconds.** Every profile run quoted here
> landed on a thermally throttled machine (`nvidia-smi` reported `SwThermal`
> throughout each). Shares of the run are stable; absolute seconds are not
> comparable between the columns of any table below. Where a claim is an
> estimate rather than a measurement it says so.
>
> **And check what the profiler does not hook before reading an exclusive
> column.** An unhooked callee is not reported as a missing stage -- its time
> lands in its caller's *own* time, where it reads as irreducible overhead. That
> hid the largest item in this document for three rounds (P10).

---

## Status -- start here

| item | state |
| --- | --- |
| **T1** PN subdivision-level criterion | **shipped and confirmed end to end** -- 67.9 s -> 1.40 s |
| **T2** Bezier chord-count search | **shipped and confirmed end to end** -- 18.4 s -> 0.89 s |
| **T4** dice write-out | **shipped, and re-scoped by what it found** -- the dice ignored *temporal coherence*: a mesh that holds still is handed T identical source rows and diced T times. Collapsing them, deduping the patch evaluation and interpolating attributes on the shared vertices is **1.27-1.37x on the dice calls that can use it and 1.05-1.15x on the dice overall, measured inside real renders** -- because only 19-55% of a real scene's dice time has frame-invariant geometry. **Bit-identical**, and all six full-render scenes match their CPU baselines. Read the "how much of a real scene is static" measurement before extrapolating from a synthetic mesh. The `allocate()` zero-fills the last revision nominated turn out to be **4%** of the write-out, not its expensive half. See T4 |
| **P2** contiguous replay time-selector | **shipped** -- 1.62x on replay-time `get`/`modify` |
| **P3** lazily-zeroed remat buffer | **shipped** -- 1.20x on `rematerialize_state_at_times` |
| **P4** whole-scene per-batch scans | **shipped** -- honestly ~3.7 s of a ~500 s render (66x on the actor filter), not the 11.63 s first reported; see the correction under P4 |
| **P5** replay-checkpoint in-place growth | **shipped, near-worthless** -- the clone it removes happens once per render, not once per batch; see the correction under P5 |
| **P1** compact rematerialization buffer | **shipped** -- 1.37x on materialization + replay reads, and 604 MB/batch of buffers -> 204 MB |
| **P6** endpoint row dedup in `_query_row_states` | **shipped, minor** -- byte-identical, but only **1.051x on the query**: the old item-1 premise predated P1; see P6 |
| **P7** updater-trace clone memo + batched `get_orthonormal_vector` | **shipped** -- 1.042x on the whole replay stage and 1.23x on the updater section; **and it re-scoped item 1 again**: the replay loop's cost is the *updater bodies*, not the per-event machinery. See P7 |
| **P8** collating the per-descendant fan-out | **shipped** -- 99.5% of s05's 25 582 recorded events were per-descendant fan-out of a few hundred subtree-wide operations. Spawn/despawn now record **one** animation for the whole set: s05 25 582 -> 12 659 events, **1.64x on `set_state_to_times`**, 1.25x on prep. **Not byte-identical** -- it restores `Text`/`Tex` wave exits an ancestor's fade used to overwrite; two full-render baselines move. See P8 |
| **P8b** the `dim_mobs` family, collated in the video project | **shipped** -- `map_animated_attribute` replaced the per-descendant loops. With P8 this took `save_video` 396.7 s -> **348.75 s** and `set_state_to_times` own time 92.5 s -> **56.4 s (1.64x)**, reproducing the harness prediction end to end. See the 2026-08-16 re-profile |
| **P10** the batched surface build | **measured -- the top item.** `get_batch_of_primitives`' own time was never orchestration: **85.35 s (21.9%)** is `get_render_primitives_batched`, which **had no profiler hook**. With the hook added the stage's own time drops 23.8% -> **4.5%**. Inside it, 59.8% is `compute_grid_vertex_normals` and only 24.2% is the per-surface tail. See P10 |
| **P11** pairwise triangle sides in `compute_grid_vertex_normals` | **shipped and confirmed end to end** -- **2.0-2.2x** on the function, **1.51x on the whole stage** (85.35 s / 21.9% -> 56.62 s / 15.8%), **bit-identical** (asserted on bit patterns, incl. NaN payloads and signed zeros). Removes an 8-copy stack, an 8-wide subtract, two stride-2 gathers and a length-4 reduction. See P11 |
| **P12** packed surfaces (`Surface.from_batches`) | **shipped** -- a collection of like surfaces can now be built as **one** Mob instead of N. Construction stops scaling with the member count (2048 spheres: 2.26 s -> 0.006 s), the per-frame primitive build is **54x** the per-actor path and **13x** the cross-actor batcher it makes unnecessary, and the Scene loses 2(N-1) actors. Byte-identical -- all 6 full-render scenes and `tests/fast` match their committed CPU baselines. See P12 |
| **P9** widening the batched bezier build | **measured, not started** -- it reaches **18.4%** of s05's circuits; **51.5% are reverted by the all-or-nothing group clash** the code calls "rare". ~19 s, ~1.04x. See P9 |
| **T5** sparse-coverage host chain | the render thread's largest non-kernel item |
| **T3**, **T6** | untouched; both shrank in share |

**Read this before picking anything up.** The 2026-08-16 round moved the
ranking's centre of gravity three times over.

* The re-profile after P8 showed that **everything that shrank was on the
  timeline/replay side, and everything on the geometry-build and render-thread
  side was flat in absolute seconds.** `set_state_to_times`, the stage two
  rounds had attacked, fell below `get_batch_of_primitives`' own time.
* Measuring that own time (P10) found it was **not orchestration at all**:
  essentially all of it was one unhooked function, the batched surface geometry
  build, at 85.35 s (21.9%). Hooking it dropped the stage's own time to 4.5%.
  Hence the rule at the top: **an unhooked callee shows up as its caller's own
  time, and "own time" reads like irreducible overhead.**
* Splitting *that* (P10) put 59.8% of it in `compute_grid_vertex_normals`, and
  P11 cut that 2x, taking the whole stage to 56.62 s (15.8%).

What is left on top is **`AttributeTimeline.get`** at 72.58 s (20.3%) -- which
has never been targeted and has risen to first place by attrition. See "What is
left, in order".

The re-profile this document demanded has been done (2026-08-14, warm run 2,
`save_video` = 382.3 s) and it moved three things:

* **T1 and T2 are confirmed.** Their two stages went from a combined 86.3 s
  (10.8%) to **2.3 s (0.6%)**. The paper estimate was ~73 s; the real figure is
  ~84 s. This is the confirmation the previous revision asked for.
* **The two poles are no longer the same size.** Prep is now 78.3% of
  `save_video` against the render thread's 51.3%, so the "cutting only one pole
  buys almost nothing" advice below no longer applies symmetrically -- prep is
  the pole to cut.
* **P1's premise was wrong**, and only a direct measurement showed it. See P1.

**That re-profile has now been done too** (2026-08-15, warm run 2,
`save_video` = 396.7 s; table under "The shape of the problem") and it moved
two things:

* **P1 and P3 are confirmed end to end**: `rematerialize_state_at_times` went
  from 94.1 s (24.6%) to **38.4 s (9.7%)**.
* **`_query_row_states` is no longer the item its ranking claimed.** The 42.7%
  measurement predated P1, which changed the query's calling convention (compact
  live rows, ranks already deduped). Measured directly on s05
  (`AB_OPT=rowdedup _prep_timeslice_ab_s05.py`): the endpoint row dedup that
  collapses its searches ~S-fold buys **1.051x on the query, ~0.02 s/pass** --
  the query is no longer search-bound. Shipped anyway (byte-identical, cost
  bounded by its break-even bail; the T4 precedent), documented under P6.

The largest prep items are now `set_state_to_times` own time (23.3%),
`AttributeTimeline.get` outside replay (21.4%) and `get_batch_of_primitives`
own time (21.3%) -- the "fewer calls, not faster ones" family.

### Reproducing the profile

The reference scene lives **outside this repo**, in the video project:

```text
D:\AlgorithmicSimplicity\videos\rl2\animations\scenes\s05_learning_to_program_setup.py
```

That scene appends its own parent directory to `sys.path` and imports `utils` /
`backgrounds_taichi` from `videos/rl2`, so it only runs from inside that
project, not from this repo. Its entry point is the scene function
`learning_to_program_setup()` -- **an earlier revision of this document said it
authored at module scope with no scene function, which is no longer true**: the
video project is organized around `Project` and scene functions now, and a
wrapper that merely imports the module renders an empty scene.

The drivers are checked in beside the scene, in `videos/rl2/animations/`:

| script | what it does |
| --- | --- |
| `_profile_s05.py` | the reference profile (`profile_scene`, 2 runs, LD) |
| `_p1_probe_s05.py` | splits `rematerialize_state_at_times` into its parts |
| `_prep_timeslice_ab_s05.py` | in-process A/B of the prep optimizations, CPU only |

All three set `SETTINGS.skip_save_frame`: the scene sprinkles ~22
`Scene.save_frame` review stills through authoring, and each is a real render
that would land in the profiler's stage timers, because `TIMERS.reset()` happens
*before* `scene_func()`. Skipping them keeps the stage table a measurement of
`save_video` alone.

**The 2026-08-16 re-profile crashed the first time, and the bug was real.** Two
thirds through run 1 a bloom FFT buffer exhausted the arena -- the ordinary,
designed backstop -- and the split retry then died with `RuntimeError: repeats
can not be negative` inside `_class_pairs_flat`. Cause:
`_build_raster_tables` allocates the batch-wide projection / screen-bounds
tables at the arena's **persistent (reverse)** end from inside the first chunk
and caches them on `merged` for the whole batch, and `render_chunk`'s rewind
restored the reverse pointer unconditionally -- handing that range back to the
allocator while the cache still pointed into it. Ordinarily invisible, because
the render loop re-protects the range the instant `render_batch_raytraced`
returns; but the OOM retry rewinds and then **re-enters** `render_chunk` on each
half, and those halves allocate forward straight over the tables, so
`_window_pairs` reads garbage bounds and a negative bbox width reaches
`repeat_interleave`. Fixed by `rewind_to()` (`tracer.py`), which clamps the
rewind to the published `_raster_tables_reverse_pointer`; guarded by
`benchmarks/_raster_tables_retry_check.py`, whose `--mutate` arm reproduces the
crash message exactly. **Any scene dense enough to OOM a chunk could hit this,
so it is not specific to the reference workload.**

`profile_scene` writes `algan_profile_report[_tag].txt` beside the caller. Run 1
is cold (Taichi JIT, cold clocks); **read run 2**. The knobs are documented in
that module's docstring (`ALGAN_PROFILE_RUNS`, `ALGAN_PROFILE_CPROFILE`,
`ALGAN_PROFILE_TELEMETRY`, `ALGAN_PROFILE_NVPROF`); the many
`algan_profile_report__*.txt` files at this repo's root are earlier runs from
other investigations, kept for comparison.

**Kernel rows are not subtracted from the stage that launches them.** The kernel
hooks write straight to `TIMERS.times`, while `TIMERS.stage` is what computes
the exclusive column, so a stage's `incl`/`excl` both still contain its kernels.
`raster: sparse discovery` reads 95.5 s and its four kernels are 57.8 s of that
-- the host chain T5 targets is the ~37.7 s difference, not the 95.5 s.

**Authoring is slow enough to shape the harness.** s05 takes 90-110 s to author
before a single frame is prepared. Anything that needs repeated measurement
should author once and alternate arms in-process, which is what
`_prep_timeslice_ab_s05.py` does.

If the external project is unavailable, `benchmarks/bezier_rendering.py` and
`benchmarks/neural_net_benchmark.py` are self-contained in-repo drivers that run
as-is. They are smaller and bezier-heavy, so they rank T2/T3/T5 usefully but say
little about the PN targets, and their absolute shares do not match the table
below.

### The measurement trap that invalidated part of this document

**Calling `Scene.get_batch_of_primitives` directly is not a faithful stand-in
for a render, and every prep harness here does it.** Replay executes a recorded
function's *undecorated* body (`f.function(f.caller, **kwargs)` in
`set_state_to_times`), while the `animated_function` decorator normally runs
that body inside `AnimationContext(record_funcs=False, ...)`. Replay does not
reproduce that wrap, so a recorded function whose body calls another animated
function -- `Cylinder.set_start_point` -> `_move_between_points` -> `move_to`
is one -- **records a brand new event every time it is replayed**. Measured:
+6 events per call, 480% growth over four calls on one window
(`benchmarks/_replay_records_check.py`).

A real render suppresses this upstream and records nothing. Measured over a
genuine 58-batch `save_video` of the reference scene
(`videos/rl2/animations/_real_render_scans_s05.py`):

| | direct `get_batch_of_primitives` probe | real `save_video` |
| --- | --- | --- |
| function applications | +31..99 per batch | **0 over 58 batches** |
| `_resolve_replay_windows` | every batch | **once** |
| `_windows` cache | rebuilt every batch | **1 rebuild, 57 hits** |

Consequences, spelled out because two entries below were written from the wrong
column: the per-batch re-resolve and the per-batch event-window rebuild **do not
happen in a render**. Anything measured through a direct-call harness that
depends on recording, resolving or window-cache state is measuring the harness.
Stage costs that are unconditional per batch -- `rematerialize_state_at_times`,
the accessors, the actor filter, the event lookups themselves -- are still real
through such a harness, which is why P1/P2/P3 survive unchanged.

**Both halves are now fixed**, so this section is history rather than a live
hazard -- kept because the numbers it invalidated are quoted throughout:

* `set_state_to_times` **enters the non-recording context itself**, so replay
  records nothing for any caller. Verified: a direct, unwrapped
  `get_batch_of_primitives` loop that previously added +6 events per call now
  adds 0 (`benchmarks/_replay_records_check.py`, which asserts it).
* The context has one definition,
  :meth:`~algan.render_loop.RenderLoopMixin.batch_prep_context`, used by the
  render loop and by every prep harness, so the two cannot drift.

Two things worth knowing from the sweep: `benchmarks/_prep_profile.py` had
already been wrapping correctly all along (it just predated the helper), and
`tests/unit_tests/test_render_batch_sizing.py` only ever defines its *own* stub
`get_batch_of_primitives` on a fake Scene, so it never touched this at all.

### Verifying a change to any of this

* `pytest -q --fast` is the loop, **but it cannot see a PN level flip**:
  `tests/fast/scene.py` deliberately contains no `Surface`/PN geometry, and
  since the fast suite became a curated set the PN behavioural tests
  (`test_logical_pn_tessellation.py`, `test_surface_autotune.py`) are outside it
  too. Run those two files directly for a tessellation change.
* `pytest -q tests/full_renders` (~10 min) is the one that can. Run it for
  anything touching tessellation, projection or the criteria.
* Prep changes are seen by neither on their own -- but both suites do exercise
  them, since every render prepares batches. Run the full renders for a prep
  change too, and A/B the stage in-process (below) for the number.
* A/B parity scripts for the shipped work, the kernel ones CUDA-only:
  `benchmarks/_pn_criterion_kernel_ab.py` (levels + shared-edge agreement +
  timing, static *and* moving meshes), `benchmarks/_bez_chord_kernel_ab.py`
  (chord counts + timing), `benchmarks/_pn_dice_ab.py` (the dice against its
  pre-temporal-coherence self: bit-equality of every diced array, plus
  alternating in-process timing, on static / orbiting / deforming meshes),
  `benchmarks/_logical_pn_crack_check.py` (seam integrity),
  `benchmarks/_p1_zerofill_ab.py` (the lazily-zeroed buffer, CPU),
  `benchmarks/_query_rowdedup_parity.py` (the endpoint row dedup, CPU: dense
  path == dedup path == brute force, with every branch -- fast path, break-even
  bail, S==1 skip -- asserted exercised, not assumed),
  `benchmarks/_orthonormal_bitwise_ab.py` (the batched-axis orthonormal basis:
  bit-pattern equality against the inlined pre-rewrite implementation, incl.
  NaN and signed zero),
  `benchmarks/_updater_clone_memo_parity.py` (the updater clone memo, CPU:
  parity across `detach_history` splits, non-vacuity asserted, and a mutation
  check that removes the invalidation and requires the comparison to fail),
  `benchmarks/_prep_timeslice_ab.py` + `videos/rl2/animations/_prep_timeslice_ab_s05.py`
  (the prep optimizations, CPU, `ALGAN_OPT_DISABLE`-gated),
  `benchmarks/_event_index_parity.py` (the interval index, CPU),
  `benchmarks/_resolve_rollback_check.py` (the replay checkpoint survives a
  render unchanged -- read its note on how it avoids being vacuous),
  `benchmarks/_raster_tables_retry_check.py` (the out-of-memory split retry
  keeps the cached raster tables; asserts the *invariant* as well as the frames,
  because whether the corruption manifests depends on how much arena headroom
  the machine has).
* **Real-render** probes, which are the only ones that can see recording,
  resolve and cache behaviour honestly (see "The measurement trap" above):
  `benchmarks/_replay_records_check.py` (replay re-records; quantifies it),
  `benchmarks/_render_time_growth_check.py` (a real multi-batch render records
  nothing and resolves once), `videos/rl2/animations/_real_render_scans_s05.py`
  (the per-batch scans timed inside a genuine `save_video`).
* **Where the geometry-build time goes**, all in `videos/rl2/animations/`, all
  prep-only on the CPU: `_gbop_section_probe_s05.py` splits
  `get_batch_of_primitives`' own time into its untracked sections (P10),
  `_surface_build_probe_s05.py` splits `get_render_primitives_batched` into its
  shared prefix and per-surface tail (P10), and `_bezier_batchability_s05.py`
  attributes every circuit to batched / group-reverted / gate-rejected, naming
  the clause that rejected it (P9). **All three wrap shared helpers, so they
  scope their timers to the function under test** -- an unscoped wrapper times
  the whole prep pass and produced one wrong reading already (see P10).
* `benchmarks/_grid_normals_ab.py` (P11: bit-pattern equality of the vertex
  normals across 13 grid topologies plus timing; read its `REAL` rows, the small
  cases are dispatch-bound).
* **Where the recorded events come from** (P8), both in
  `videos/rl2/animations/`: `_authored_funcs_s05.py` attributes every recorded
  `FunctionApplicationEvent` to its authoring site, the algan-internal chain
  that reached it, and the fan-out driver that multiplied it (authoring only,
  no render); `_fanout_collate_ab_s05.py` A/Bs stock authoring against a
  collated arm, one arm per process (`AB_ARM=A` / `B`, then `--report`), and
  checks the arms are the same scene before comparing their times.
* Scaling probes, all in `videos/rl2/animations/`, all CPU-only prep with no
  render. **They drive `get_batch_of_primitives` directly, so their recording
  and cache-invalidation behaviour is not a render's** -- read them for the
  stage costs that are unconditional per batch, not for anything else:
  `_remat_scaling_s05.py` (materialization vs total rows N vs active rows R),
  `_quadratic_scans_s05.py` (the whole-scene per-batch scans),
  `_windows_rebuild_probe_s05.py` (splits a lookup into resolve / cache rebuild
  / query -- the one that found P4's real bottleneck),
  `_event_duration_dist_s05.py` (why an interval index does or does not prune),
  `_replay_loop_probe_s05.py` (splits `set_state_to_times` own time into
  pre-loop lookups / remat / event bodies net of accessors / loop machinery --
  the one that scoped item 1 of "What is left").
* Re-baseline, only after looking at the frames:
  `ALGAN_UPDATE_FULL_RENDER_BASELINES=1 <venv-python> -m pytest tests/full_renders -q`

## The shape of the problem

The render is a two-thread pipeline, and **prep is now the larger pole by a
clear margin**:

| pole | stage | original | 2026-08-14 | 2026-08-15 | 08-16 (P8) | 08-16 (P11) |
| --- | --- | --- | --- | --- | --- | --- |
| batch-prep worker | `Scene.get_batch_of_primitives` | 519 s (64.8%) | 299.4 s (78.3%) | 310.2 s (78.2%) | 257.2 s (73.7%) | 263.6 s (**73.6%**) |
| render thread | `ray traced render total` | ~543 s (67.7%) | 196.1 s (51.3%) | 190.3 s (48.0%) | 189.6 s (54.4%) | 203.0 s (56.7%) |

They still overlap, so neither number is the wall clock. But the old advice --
"cutting only one pole buys almost nothing, work the two lists together" -- was
written when the poles were level. They are not any more: T1 and T2 took ~84 s
off the render thread and nothing off prep, and prep is what the total now
tracks. **Prefer a prep item over a render item of the same size.**

P8 then took 53 s off prep and **nothing** off the render thread (189.6 s
against 190.3 s -- the same number twice), which is why the gap narrowed from
78.2/48.0 to 73.7/54.4 without the advice changing. P11 narrowed it again.
**The render thread's share has risen at every re-profile without its work
changing at all**, purely because prep keeps shrinking; the two poles are close
to level, so the next round or two should re-check that advice rather than
inherit it.

A second consequence, unchanged: this scene is *not* GPU-limited. Average GPU
utilisation was 44% on the re-profile. The Taichi candidates below are worth
doing because they are on the render thread's critical path, not because the
device is saturated.

### The re-profile, warm run 2 (`save_video` = 348.75 s, 2026-08-16, post P8 + P8b)

The scene's authoring changed under this measurement: the video project's
`dim_mobs` / `restore_mobs` now call `Mob.map_animated_attribute` instead of
looping `set_non_recursive` over `get_descendants()`, which is the half of P8
that lives outside the engine. That plus P8 itself is everything between this
column and the last one.

**This is the pre-hook, pre-P11 column; the current one is below it.** It
predates the `surfaces: get_render_primitives_batched` hook, so that function's
cost is inside `get_batch_of_primitives`' own time here. The row is left as
measured so the profiles stay comparable.

| item | measured | was (08-15) | |
| --- | --- | --- | --- |
| **`get_batch_of_primitives` own** | **82.98 s (23.8%)** | 84.6 s (21.3%) | flat -- **~85% of it is the batched surface build**; see P10 |
| `AttributeTimeline.get` (542 052 calls) | 57.14 s (16.4%) | 84.9 s (21.4%) | 1.49x, 105 000 fewer calls |
| **P8** `set_state_to_times` own | **56.38 s (16.2%)** | 92.5 s (23.3%) | **1.64x** |
| `bloom fft conv` | 44.12 s (12.7%) | 42.6 s (10.7%) | flat |
| `rematerialize_state_at_times` | 39.74 s (11.4%) | 38.4 s (9.7%) | flat |
| **T5** sparse-discovery *host* chain | ~36.66 s (10.5%) of 91.44 s incl | ~39.9 s (10.1%) | flat |
| `BezierCircuitCubic.get_render_primitives` own | 36.22 s (10.4%) | 39.8 s (10.0%) | ~flat, 11 607 calls |
| `memory reclaim (gc + cuda cache)` | 31.35 s (9.0%) | 39.9 s (10.1%) | |
| **T3** `_build_circuit_geometry` | 19.26 s (5.5%) | 19.2 s (4.8%) | flat |
| **T4** `_dice_logical_pn` own | 10.04 s (2.9%) | 9.5 s (2.4%) | flat |
| **T6** precompute tables | 5.55 s (1.6%) | 5.7 s (1.4%) | flat |
| **T1+T2** criterion kernels | 1.38 s (0.4%) | 2.6 s (0.7%) | |

Read the "was" column as a *ratio* check, not a subtraction: both runs were
thermally throttled (`SwThermal` throughout, avg GPU utilisation 45%).

**What this says, and it is the whole point of the re-profile:** P8 delivered
what its harness predicted -- 1.64x on the replay stage, against the 1.64x
`_fanout_collate_ab_s05.py` measured -- and the two-pole extrapolation of
"~1.23-1.33x on the render" was **too optimistic**: the real figure is
**1.14x** (396.7 s -> 348.75 s). The saving landed entirely on the pole that was
already overlapped, so a third of it disappeared into the render thread's
shadow. Treat future two-pole extrapolations the same way: they bound the win,
they do not predict it.

The other half of the finding is negative and more useful. Nine of the twelve
rows above are **flat in absolute seconds**. Everything the last three rounds
moved was reached through the timeline; nothing that builds geometry or renders
has been touched since T1/T2. That is where the remaining time is.

### The current profile, warm run 2 (`save_video` = 358.05 s, 2026-08-16, post P11)

Hooked and post-P11, so this is the column to rank against.

| item | measured | share |
| --- | --- | --- |
| **`AttributeTimeline.get`** (542 052 calls) | **72.58 s** | **20.3%** |
| **`set_state_to_times` own** | **64.21 s** | **17.9%** |
| **`surfaces: get_render_primitives_batched`** | **56.62 s** | **15.8%** |
| `bloom fft conv` | 45.62 s | 12.7% |
| `BezierCircuitCubic.get_render_primitives` own | 40.97 s | 11.4% |
| **T5** sparse-discovery *host* chain | ~41.3 s of 99.99 s incl | ~11.5% |
| `rematerialize_state_at_times` | 34.97 s | 9.8% |
| `memory reclaim (gc + cuda cache)` | 31.26 s | 8.7% |
| **T3** `_build_circuit_geometry` | 19.96 s | 5.6% |
| `get_batch_of_primitives` own | 18.44 s | 5.2% |
| `AttributeTimeline.modify` (68 020 calls) | 11.46 s | 3.2% |
| **T6** precompute tables | 6.20 s | 1.7% |

Poles: prep 263.60 s (73.6%), render thread 203.04 s (56.7%).

**`AttributeTimeline.get` is now the largest single item**, at 542 052 calls of
~134 us each. P8 cut the replay-side calls and did not touch the geometry-side
ones, and P11 removed work *around* them rather than any of the calls
themselves, so this row has risen to the top by attrition. It is the same
"fewer calls, not faster ones" family as P9 and the per-surface tail of P10 --
and both of those *are* ways of removing these calls.

### The previous re-profile, warm run 2 (`save_video` = 396.7 s, 2026-08-15, post P1-P6)

| item | measured | was (08-14) |
| --- | --- | --- |
| `set_state_to_times` own | 92.5 s (**23.3%**) | 19.7% |
| `AttributeTimeline.get` (647 399 calls) | 84.9 s (**21.4%**) | 14.5% |
| `get_batch_of_primitives` own | 84.6 s (**21.3%**) | 16.6% |
| **T5** sparse-discovery *host* chain | ~39.9 s (~10.1%) of 91.8 s incl | ~9.9% |
| `bloom fft conv` | 42.6 s (10.7%) | 11.7% |
| `memory reclaim (gc + cuda cache)` | 39.9 s (10.1%) | 9.9% |
| **P1** `rematerialize_state_at_times` | **38.4 s (9.7%)** | 24.6% -- P1+P3 confirmed |
| `BezierCircuitCubic.get_render_primitives` own | 39.8 s (10.0%) | 7.5% |
| **T3** `_build_circuit_geometry` | 19.2 s (4.8%) | 5.1% |
| **T4** `_dice_logical_pn` own | 9.5 s (2.4%) | 2.7% |
| **T6** precompute tables | 5.7 s (1.4%) | 1.4% |
| **T1+T2** criterion kernels | 2.6 s (0.7%) | 0.6% |

`arena preflight (batch)` reads 58.7 s inclusive but 0.4 s exclusive -- its
children (`project_to_screen (prewarm)`, the bezier/PN packing stages) are
already listed and should not be double-counted against it.

### The previous re-profile, warm run 2 (`save_video` = 382.3 s, 2026-08-14)

| item | measured | was |
| --- | --- | --- |
| **P1** `rematerialize_state_at_times` | 94.1 s (24.6%) | 24.1% |
| `set_state_to_times` own | 75.2 s (19.7%) | 15.6% |
| `get_batch_of_primitives` own | 63.4 s (16.6%) | 11.7% |
| `AttributeTimeline.get` (660 013 calls) | 55.6 s (14.5%) | 12.0% |
| `bloom fft conv` | 44.7 s (11.7%) | 6.8% -- still not worth a kernel |
| `memory reclaim (gc + cuda cache)` | 37.7 s (9.9%) | 4.9% -- **see below** |
| **T5** sparse-discovery *host* chain | ~37.7 s (~9.9%) | ~51 s of 127.6 s |
| `BezierCircuitCubic.get_render_primitives` own | 28.5 s (7.5%) | 6.3% |
| **T3** `_build_circuit_geometry` | 19.4 s (5.1%) | 47.1 s (5.9%) |
| **T4** `_dice_logical_pn` own | 10.3 s (2.7%) | 2.1% |
| **T6** precompute tables | 5.4 s (1.4%) | 8.6 s (1.1%) |
| **T1** PN subdivision levels | **1.40 s (0.4%)** | 67.9 s (8.5%) |
| **T2** bezier chord search | **0.89 s (0.2%)** | 18.4 s (2.3%) |

### `memory reclaim` doubled in share, and the reason is a gate that never closes

`empty_cache` (`utils/memory_utils.py`) is deliberately gated: it only runs
`gc.collect()` / `torch.cuda.empty_cache()` when `_gpu_memory_pressure()` says
the device is above **80% of total memory**. The reference machine is a 4 GB
GTX 1050 and the render sits at 3.4 GB peak reserved for its whole duration, so
that gate is open essentially always -- 510 calls at ~74 ms each. The gate is
doing nothing on this card.

Before optimizing it, note what it is not: it is not a general win. On a card
with headroom the gate closes and the cost is already near zero, so this is a
*small-VRAM* item, and the project's rule is that optimizations target general
scenes rather than one configuration. Worth measuring on a second device before
spending anything on it.

---

## Part 1 -- Dedicated Taichi kernels

Every entry here is torch code that already runs on the **render device**
(`PROJECT_ON_GPU` is on by default, so primitive source geometry, the camera
snapshot and the scratch arena are all `cuda:0` by the time these run). They
share one failure mode: tens of elementwise passes over large arrays to
produce a handful of scalars, so they are VRAM-bandwidth bound rather than
launch bound. Fusing each into a single kernel that keeps intermediates in
registers is the win.

### T1. PN subdivision-level criterion -- 67.9 s (8.5%) -> **1.40 s (0.4%)** -- **DONE, CONFIRMED**

> **Shipped.** `algan/rendering/raytracing/logical_pn_taichi.py`
> (`pn_patch_flatness_error`, `pn_edge_chord_error`), default on, gated by
> `ALGAN_PN_CRITERION_KERNEL` / `SETTINGS.raytracing.experimental
> .pn_criterion_kernel`. Measured on `benchmarks/_pn_criterion_kernel_ab.py`:
> the **criterion itself is 4-15x faster** (typically ~6x; 16 ms -> 1.0 ms at
> level 4), and the whole level search is 2.6-3.1x on those small meshes -- the
> residual there is the per-level loop machinery, which is a much smaller share
> on the reference scene than on a 24-patch benchmark (see below).
>
> **The re-baseline prediction held; the fast suite is not enough to see it.**
> Every synthetic case tried chose levels *identical* to torch -- sphere and
> torus, fine and coarse grids, tight tolerance, static (stride-0 broadcast) and
> moving (stride-1 per-frame) geometry, ~34 000 level decisions -- and the fast
> suite passes unchanged. But three **full-render** scenes moved:
> `complex_hierarchy_become`, `materials_and_lighting`, `solids_and_camera`
> (all three pass again with `ALGAN_PN_CRITERION_KERNEL=0`, so the attribution
> is exact).
>
> The diffs are the silhouette signature, not a defect. Measured old baseline
> against new, losslessly:
>
> | scene | frames changed | peak channel diff | changed px, whole video |
> | --- | --- | --- | --- |
> | `complex_hierarchy_become` | 46 / 75 | 48 | 1.04% |
> | `materials_and_lighting` | 25 / 179 | 30 | 0.13% |
> | `solids_and_camera` | 102 / 239 | 44 | 0.49% |
>
> Looked at (the step the re-baseline instruction exists for): the worst frame of
> each is visually indistinguishable, and at 8x amplification the difference is
> confined to sphere silhouette rims, mesh edges and glyph outlines. No shape,
> material or lighting change anywhere. Per-sample errors differ by ~5e-5 px
> against a ~540 px threshold, so a flip needs a patch within ~1e-7 relative of
> the boundary -- rare per decision, inevitable across a dense scene's millions
> of them. **Baselines for these three scenes were regenerated.**
>
> **Do not conclude byte-equality from the microbenchmarks or the fast suite;
> only the full render suite can see this.** The fast suite deliberately
> contains no `Surface`/PN geometry, so its one render cannot flip a PN level at
> all.
>
> Crack-freeness holds regardless: `benchmarks/_logical_pn_crack_check.py`
> reports pixel-identical coverage on both paths, and the A/B script groups
> every (frame, patch, edge) by its canonical controls and confirms all 433
> shared curves agree. The reduction being `max` -- exact and order-independent
> -- is what makes the cross-thread `atomic_max` safe.

#### Why it was the top target (original measurement, kept as evidence)

`_dice_logical_pn` is 86.2 s (10.7%), and 67.9 s of that was *choosing* the
levels, not dicing:

| | measured |
| --- | --- |
| `_required_subdivision_levels` | 67.9 s |
| ↳ `_guarded_pixel_error` (4562 calls) | 32.1 s |
| ↳ `evaluate_logical_pn` (3121 calls) | 17.5 s |
| ↳ error-function own time (chords, einsum, masks) | 17.0 s |
| ↳ **per-level loop machinery** | **2.6 s** |

Per call `_guarded_pixel_error` ran ~30 elementwise ops -- **two** full
perspective projections (12 ops each) plus clamp / subtract / norm / guard /
where / amax -- over `[K, N, 3]` arrays of ~2^18 elements, moving ~180 MB to
produce one scalar per patch. 4.4 ms/call is what that traffic costs at this
card's bandwidth. The estimate was 67.9 s -> 8-12 s.

**Still true, and still worth not re-litigating:** the loop bookkeeping
(`levels.clone()` per level, `_triangle_counts` over `[T, P]`, the `.any()` /
`.nonzero()` device syncs) was only 3.9% of the search. Restructuring the climb,
seeding it analytically from the 4^-L error scaling, or compacting the per-level
work to the active set would all buy ~nothing *on this scene*. That was measured
specifically to rule them out; do not spend time there. Note the caveat the
microbenchmarks added: on a small mesh (tens of patches, a handful of frames)
that machinery is instead the *majority* of the search, which is why the A/B
script reports 2.6-3.1x for the whole search against ~6x for the criterion. Do
not read the small-mesh ratio as the scene-level one, in either direction.

### T2. Bezier chord-count search -- 18.4 s (2.3%) -> **0.89 s (0.2%)** -- **DONE, CONFIRMED**

> **Shipped**, folded into T1's kernel family as predicted:
> `bezier_chord_hull_error` reuses `_screen_pixels` / `_cubic_evaluate` and adds
> `_cubic_derivative` + `_segment_distance_squared`, behind the same toggle.
> Measured on `benchmarks/_bez_chord_kernel_ab.py` over text, LaTeX and circle
> geometry: **5.4x** (76.3 ms -> 14.1 ms), with identical chord counts on every
> segment of those scenes -- with the same caveat as T1, since the two share one
> toggle and the full-render diffs were not attributed between them.
> One thread per (segment, frame, subcurve); the
> straddling-subcurve `inf` fallback is a comparison-based magnitude test rather
> than an `isfinite` intrinsic, so `fast_math` cannot fold it away.

#### Why it was a target (original measurement)

`RayTracedBezierCircuitPrimitive._compute_samples_per_segment` was the *same
computation shape* as T1: climb subdivision levels, project each level's control
hull, reduce to a max error per segment, keep the first level under tolerance --
the same iterative structure, the same projection block written out again, the
same per-chunk elementwise passes. The only differences are the evaluator
(uniform cubic subcurves rather than a PN patch) and that the bound is a
control-hull distance rather than a sample deviation. **Estimated** 18.4 s ->
~3 s (~1.9%).

### T3. Bezier circuit geometry build -- 19.4 s (5.1%)

`_build_circuit_geometry` samples world-space polylines into per-circuit plane
coordinates and packs the per-circuit metadata the trace kernel reads. It is
elementwise sampling plus `repeat_interleave` / gather / scatter plumbing over
the whole batch's segments.

Less certain than T1/T2 -- it is a build, not a reduction, so the output is as
large as the input and the bandwidth argument is weaker. Worth a
`memory.scope`-level breakdown before committing. **Estimated** 47.1 s -> ~25 s
(~2.7%), lower confidence.

### T4. The dice write-out -- **temporal coherence, shipped**

> The item was framed as "the write-out moves too many bytes", and both halves
> of that framing were wrong. The dice's real problem was that **it recomputed
> the same answer once per frame**: a patch's diced geometry is a function of
> the patch and its level and *nothing else* -- the camera only picks the level
> -- while materialization hands the dice one source row per frame whether or
> not the mob moved. A still `Sphere` reaches it as T byte-identical copies
> (measured on a two-mob scene: `corners` arrives `[4, 508, 3, 3]` with
> `distinct_frames = 1`) and was diced T times over.
>
> Shipped, all **bit-identical** (`benchmarks/_pn_dice_ab.py` runs the
> pre-change dice beside the current one in one process and compares every
> diced array bit for bit -- corners, normals, colours, the surface and shader
> parameters, uvs, the padding mask, both level arrays and the per-row surface
> ids -- across static, orbiting, multi-level and deforming meshes):
>
> 1. **`_collapse_redundant_frames`.** Detect the identical rows (compare frame
>    1 first, so a genuinely deforming mesh is rejected for a (T-1)th of the
>    cost) and keep one. The three control-net builds --
>    `logical_pn_control_points`, `logical_pn_normal_control_points`,
>    `logical_pn_edge_control_points`, together ~25% of the write-out -- then
>    run once instead of T times, and `_frame_broadcast_base` hands the
>    criterion kernels a stride-0 net instead of T uploaded copies of one
>    answer.
> 2. **Per-patch dedup in the write-out.** The selected pairs are listed
>    PATCH-major (`nonzero` on the transpose), so a patch's frames share a
>    chunk; the patch and normal evaluations then run once per distinct patch
>    and fan out with one `index_select`. Measured **9-20x** fewer evaluations
>    than pairs under a fast orbit, and exactly T when the camera holds still.
>    The boundary snap stays per row -- two frames dicing one patch need not
>    agree on its boundary levels.
> 3. **Attributes interpolate on the shared subdivision vertices**
>    (`interpolate_patch_vertex_attribute`) and gather through
>    `subdivision_triangle_indices` instead of being evaluated at every
>    microtriangle corner: the corners *are* those vertices, so it is the same
>    arithmetic over a sixth as many of them.
>
> The same dedup is applied to the **torch fallback** of the patch-flatness
> criterion (`share_patches`), which re-evaluated a patch once per frame still
> searching. Deliberately off on the kernel path, which keeps its samples in
> registers and has nothing to share.
>
> Verified end to end: all six `tests/full_renders` scenes match their
> committed **CPU** baselines here. That is a real check on this machine and not
> a vacuous one -- re-baselining on it before the change rewrote every file
> byte-identically, so the committed CPU set *is* this machine's output. CUDA is
> unverified: this session has no GPU, and the criterion-kernel path
> (`share_patches` off, stride-0 control nets) is only exercised there.

#### How much it is worth, and why the synthetic number overstated it

On isolated meshes `benchmarks/_pn_dice_ab.py` reports **1.13-1.33x** for a mesh
that holds still while the camera moves. **Do not quote that as the render-level
figure.** Measured *inside* a real render, by running both dice implementations
on every call with the same inputs and alternating which goes first:

| scene | dice calls | frame-invariant | deforming | whole dice |
| --- | --- | --- | --- | --- |
| `solids_and_camera` | 10 | **1.37x** (2 calls, 22% of dice time) | 0.98x | **1.05x** |
| `materials_and_lighting` | 38 | **1.27x** (20 calls, 60%) | 1.01x | **1.15x** |
| `complex_hierarchy_become` | 4 | -- (one 10 ms call) | 1.14x | **1.14x** |

The gap is not a measurement artifact, it is the workload: **a mob's `corners`
are world-space**, so "frame invariant" means the mob does not move *at all*
during the batch, and a scene whose whole point is motion spends most of its
dice time on meshes that do move. `solids_and_camera` has 2 of 10 calls static;
`materials_and_lighting`, which animates the camera around mostly-parked
spheres, has 20 of 38. Batches are large (101 and 138 frames here), so where it
does fire the redundancy removed is correspondingly large.

`_dice_logical_pn` was ~18% of `solids_and_camera`'s render, so 1.05x on the
dice is ~1% of that render. On `materials_and_lighting` the dice is a bigger
share and the saving is 2.9 s of 22.5 s.

**And one thing this measurement caught that the synthetic benchmark hid.**
Listing the work list patch-major is what lets the dedup see its duplicates,
but it is not free: consecutive rows then write a whole frame apart in the
output instead of in one run, and read their control points the same way. On
the deforming half of `materials_and_lighting`'s calls -- which have nothing to
dedup -- that cost **0.965x** while the isolated benchmark's deforming mesh
reported a harmless 1.03x. Patch-major ordering is therefore gated on
`geometry_static`, exactly as `share_patches` is in the search; with the gate
the same calls measure 1.005x. A synthetic mesh was too small to show it.

#### Three things measured here that change what to do next

* **The `allocate()` zero-fills are not the expensive half.** The previous
  revision nominated them as where a fresh attempt should start. Measured:
  `torch.zeros` is **4%** of the write-out on CPU (9.3 ms of 227 ms on a
  32-frame, 1056-patch batch) and proportionally less on a device that memsets
  at hundreds of GB/s. Filling only the padding *is* available -- the unwritten
  rows are each frame's contiguous tail -- it is simply not worth the branch.
  Do not spend the round there.
* **Deduping the attribute interpolation is a net loss.** A barycentric blend
  of three corner values is cheap enough that fanning the result back out costs
  more than recomputing it per row: **0.86x** on a deforming mesh whose colours
  were frame invariant. Only the patch evaluation -- a ten-term polynomial plus
  a snap, twenty-odd passes -- is expensive enough for the trade to pay. That
  asymmetry is the rule to carry: dedup buys nothing unless what it skips costs
  more than a gather.
* **Emitting a `[1, ...]` diced array is worth much less than it looks.** When
  the levels *and* the edge levels come out frame-uniform (a still mesh, which
  is common) the whole diced output is T identical copies, and the flat path
  already accepts `[1, N]` geometry. But `scene_builder`'s
  `_cat_collections` runs `_unify_time` over the primitives it concatenates, so
  a single moving flat mesh anywhere in the scene expands the static one
  straight back out at merge time. The saving is confined to the primitive's
  own allocation and packing -- real, but not the T-fold win the shape
  suggests, and it changes the primitive's output contract (three tests assert
  a per-frame diced array). Left undone deliberately.

#### Where the dice's remaining cost is

On CPU it is **the level search, not the write-out**: 65-85% of
`_dice_logical_pn` across the benchmark's meshes. The criterion is inherently
expensive -- per level tried it evaluates each patch at `V + 13 * 4 ** L`
parameters and perspective-projects *both* the exact and the approximated point
sets, so it does ~26 projections per output triangle where the dice does ~1.5
vertex evaluations. On CUDA that is T1's fused kernel and costs nothing. **The
obvious next move for the CPU path is to let those same kernels run on Taichi's
x64 backend** -- `pn_criterion_kernel_active` gates them on a CUDA *render
device* today, and the "Taichi would stage every argument through VRAM"
objection in that comment is about CUDA-arch Taichi against CPU torch tensors,
not about a CPU-arch runtime. It is not free: `fast_math` flips borderline
levels, so it would move the CPU baselines exactly as T1 moved the CUDA ones,
and it needs the same deliberate re-baseline.

#### Why it was a target (original measurement)

`_dice_logical_pn`'s own time (12.4 s) plus `interpolate_patch_attribute`
(3.1 s), `evaluate_logical_pn_normals` (1.0 s) and `snap_boundary_values`
(0.4 s). Two costs were identified: the `allocate()` zero-fills
(`[T, max_triangles, 3, D]` for corners, normals, colours and every
surface/shader parameter) and the advanced-index scatters. An earlier round
folded (frame, column) into one flattened row index so each output is written
with a single `index_copy_` over `[T * M, ...]`; that was byte-exact and worth
**1.02-1.19x** on the whole dice, well under the ~2x it assumed. Neither
identified cost was the real one.

### T5. Sparse-coverage host chain -- ~37.7 s of 95.5 s (~9.9%) -- **next**

`raster: sparse discovery` is 95.5 s (25.0%), but 57.8 s of that is already
Taichi (`raster_tri_count/write`, `raster_bez_count/write`) -- the stage row
still contains its kernels, because the kernel hooks bypass `TIMERS.stage`. The
remaining ~37.7 s is the host-side ordering and compaction: the count prefix sum,
`_exact_fragment_order` (two `argsort`s), six parallel `index_select` gathers
of the fragment arrays, `unique_consecutive`, the `scatter_reduce_` opaque
truncation and the keep-mask compaction.

The two `argsort`s are library code and already good. The **six gathers are one
kernel** -- they read the same permutation and write six arrays, so they move
six passes of traffic where one would do. They are at
[`raster_pipeline.py:1356-1362`](algan/rendering/raytracing/raster_pipeline.py:1356)
(`key_s`/`ref_s`/`ab_s`/`cov_s`/`msk_s`/`opaque_s`), with a second five-gather
compaction on `keep_idx` a few lines below. Each gather re-reads the whole
permutation, so six of them move ~48 bytes of index traffic per fragment
against ~29 bytes of payload. Do that; leave the sorts alone.
**Estimated** ~10-15 s (~1.5%), moderate confidence.

### T6. Raster precompute tables -- 5.4 s (1.4%)

`precompute_triangle_projection` (1.9 s), `precompute_circuit_screen_bounds`
(2.5 s), `precompute_triangle_screen_bounds` (0.9 s). Already cut ~4x an
earlier round by hoisting them from per-chunk to per-batch. What remains is
`precompute_triangle_projection` building ~8 large `[F, N, 3, 3]` temporaries
(`d`, `hit`, `rel`, two crosses, ...) to fill a `[F, N, 13]` table -- a clean
one-kernel fusion. Small on this scene; grows with triangle count.
**Estimated** 8.6 s -> ~3 s (~0.7%).

### Not worth a kernel

* **Bloom** (`bloom fft conv`, 44.7 s / 11.7%) is cuFFT. A direct separable
  Taichi convolution already exists (`bloom_conv1d_f32`) and is *far* slower
  for the wide blur -- measured 660 ms vs 23 ms for the 293-tap horizontal
  pass. The transform-length rounding shipped this round is the available win.
* **`fill_background_from_func`** (2.6 s / 0.7%) is already a kernel, launched
  once per chunk, evaluating the scene's procedural background per pixel per
  frame. That is real work, not overhead.
* **Anything on the animation device.** `rematerialize_state_at_times` and
  friends run on the CPU by design; launching Taichi against CPU tensors makes
  it stage every argument through VRAM from the prep worker, which is a
  regression that has already shipped once. See `generate_array_states`'s
  docstring.

**Part 1, settled.** T1 and T2 landed and are confirmed against the reference
scene: 86.3 s (10.8%) -> 2.3 s (0.6%), against a paper estimate of ~73 s. What
is left of Part 1 is **T5** (~9.9%, the only sizeable one), then T3 (5.1%),
T4's remainder (2.7%) and T6 (1.4%) -- and all of it now sits behind the prep
items, because the render thread is no longer the binding pole.

---

## Part 2 -- The prep worker (no Taichi; CPU by design)

**The larger pole (78.3%), and the one to cut.** See the re-profile table above
for the current shares; the notes below are what is known about each.

| target | note |
| --- | --- |
| `AttributeTimeline.rematerialize_state_at_times` | measured into its parts -- see P1 |
| `set_state_to_times` own time | the function-replay loop; the first batch replays 1403 recorded functions. **P2 shipped against this** |
| `AttributeTimeline.get` | 660k calls, ~85 us/call of Python + torch dispatch. 37% of them are made *while a function is replaying* -- **P2 shipped against those**; the rest is fewer calls, not faster ones |
| `get_batch_of_primitives` own time | geometry-build orchestration |
| `BezierCircuitCubic.get_render_primitives` | per-actor build. `build_render_primitives_batched` covers the batchable case **and reaches only 18.4% of the scene's circuits** -- see P9 |
| `memory reclaim` | see "a gate that never closes" above -- small-VRAM-specific, not a general win |

### P2 -- contiguous replay time-selector (shipped)

> **Shipped**, `_contiguous_time_selector` in `timeline.py`, gated by
> `ALGAN_OPT_DISABLE=timeslice`. **1.618x** on the calls it touches
> (0.454 s -> 0.281 s per pass of `_prep_timeslice_ab_s05.py`); byte-identical
> -- it selects the same elements in the same order.
>
> `AttributeTimeline.get` / `modify` already branched on `active_time_inds`: a
> `slice` reads a view and writes a slice-assign, a tensor pays an
> advanced-index gather and a scatter. Outside replay it is `slice(None)`, so
> the slow path existed *only* while `set_state_to_times` replays recorded
> functions -- and it built the selector with `.nonzero()`. The predicate is an
> interval over the batch's ascending frame times, so those indices are
> contiguous: on the reference scene 78 of 78 replay selectors converted. The
> helper *tests* contiguity rather than assuming it, so an updater's
> clone-grouped frame set (genuinely scattered) still takes the tensor path.
>
> **The trap this walked into, worth not repeating:** the first A/B ran on the
> in-repo debug scene and reported **1.002x**, because that scene is materials
> heavy and barely replays anything. And whole-prep wall time cannot resolve a
> change this size on this machine -- eight alternating rounds swung 5.9-10.0 s
> on *both* arms, and a first attempt read 1.138x from three rounds of pure
> noise. Time the specific functions the change reaches, on s05, or measure
> nothing.
>
> A latent hazard it exposes: `get(copy=False)` now hands back a real view
> during replay where it used to return a gathered copy. The docstring already
> required that ("pass False only for a read you will not retain"), and every
> caller feeds it into out-of-place arithmetic, but a future mutating caller
> would now corrupt the buffer instead of being silently tolerated.

### P3 -- lazily-zeroed rematerialization buffer (shipped)

> **Shipped**, `_sparsely_written_zeros` in `timeline.py`, gated by
> `ALGAN_OPT_DISABLE=lazyzeros`. **1.201x** on the whole of
> `rematerialize_state_at_times` across windows spanning the entire scene;
> byte-identical -- both paths produce zeros.
>
> `torch.zeros` is `empty` + an explicit `memset` over the whole allocation.
> `np.zeros` is `calloc`, and an allocation this size comes from the OS page
> allocator already zeroed, so only the pages actually touched are charged.
> Measured standalone (`benchmarks/_p1_zerofill_ab.py`): the fill itself goes
> from 8-77 ms to ~0.1 ms (**128-835x**), and fill + scatter together
> 1.14-1.31x. The direct check that the pages are genuinely lazy is in that
> script: untouched 0.0 ms vs fully touched 189.7 ms.
>
> CPU only (on CUDA the fill is a device memset), with a `torch.zeros` fallback
> for any dtype numpy cannot express.
>
> **Measure this kind of change on windows spread across the scene.** Rows
> accumulate as mobs spawn, so the buffer is ~5 500 rows wide at frame 0 and
> ~505 000 at frame 3800. Windows taken from the first 150 frames gave 1.133x;
> the same test spread over the whole scene gave 1.201x.

### P4 -- the whole-scene per-batch scans (shipped)

> The only genuinely **quadratic** thing in prep, and the reason to care about
> it is not its size today but its shape: each of these scanned the *whole
> scene* once per batch, and there are O(scene) batches, so cost grows with the
> square of scene length. On s05 that was 11.63 s (3%); a scene 4x as long would
> have spent ~186 s there.
>
> | per-batch scan | length | before | after |
> | --- | --- | --- | --- |
> | actor lifespan filter (`render_loop.py`) | 12 232 actors | 96.1 ms | **0.91 ms** |
> | `get_functions_for_times` | 25 904 events | 54.8 ms | **3.7 ms** |
> | `get_updaters_for_times` | 1 updater | 0.14 ms | 0.45 ms -> 0.14 ms |
> | **over the render** | | **11.63 s** | **0.39 s** |
>
> Two pieces. `_event_interval_index` sorts events by start and records the
> running maximum end, which bounds a window query from both sides using only
> facts the activity test already implies -- so the survivors are a superset and
> the exact test still decides, which keeps unsorted `times` correct too. The
> actor filter gets the same treatment plus a per-render cache, since its
> timestamps are fixed for a render (the invariant that already justified
> reading them once per batch).
>
> Parity is asserted by `benchmarks/_event_index_parity.py` over four interval
> layouts including `one_long` -- one scene-length event, which defeats the
> lower bound and must therefore still be *correct*, just unpruned -- and
> unsorted times. Same events, same order; order matters because replay
> re-executes in recorded order.
>
> **Three traps, each of which cost a measurement to find:**
>
> * **Do not key these caches on `TIMING_VERSION`.** It looks like the safer
>   key and it is a trap: it is bumped whenever a timespan is configured,
>   including for the transient mobs a render itself creates, so it changes
>   *during* a render. Keying on it turned the actor cache into a per-batch
>   rebuild at **257 ms a batch -- worse than the 96 ms scan it replaced.**
>   Timing is fixed for the duration of a render; length plus explicit
>   invalidation is the correct envelope.
> * **The bottleneck was not the thing that looked like it.** The `[F, T]`
>   matrix was never the cost -- the index cut candidates to 0-5.5% of events
>   (to *zero* at window 2400) and the lookup did not get faster at all. It was
>   `_windows` rebuilding: the resolver un-resolves on every batch (a render
>   records edits of its own), and a full invalidation re-walked every recorded
>   event's span, 58 ms a batch. The fix is that `invalidate_window_cache` now
>   takes the lowest event slot whose window actually moved and keeps the
>   prefix. Measure the parts before optimizing the one you assume.
> * **Below ~64 events an index is a pessimization** -- building and sorting
>   its tensors is fixed-overhead-dominated. Updaters regressed 0.14 -> 0.45 ms
>   before the small-list short circuit went in.
>
> Still O(F log F) per batch in the index rebuild (~5 ms of argsort at 26 000
> events, amortized over a batch's worth of appended events), so this is a very
> large constant-factor cut plus removal of the *Python-side* growth term, not
> a proof of sub-linearity. Merging the sorted tail instead of re-sorting is the
> next step if a scene ever gets big enough to care.
>
> ### CORRECTION -- most of the table above is a harness artifact
>
> Everything above was measured through direct `get_batch_of_primitives` calls,
> which re-record events and therefore re-resolve and rebuild the window cache
> every batch (see "The measurement trap" near the top). **A real render does
> neither.** Re-measured inside a genuine 58-batch `save_video` of s05, *with
> the fix in place*:
>
> | scan | per batch | over the render |
> | --- | --- | --- |
> | `get_functions_for_times` | 5.84 ms | 0.34 s |
> | `get_updaters_for_times` | 0.35 ms | 0.02 s |
> | actor index build | 1.16 ms | 0.07 s |
> | actor window queries (2/batch) | 38.95 ms | 2.26 s |
> | **all three scans** | | **2.69 s of 504.3 s (0.5%)** |
>
> So the "11.63 s -> 0.39 s" figure is not a render-level saving. The harnesses
> have since been fixed (they now enter `Scene.batch_prep_context`, as a render
> does) and the same probe re-run: **the three scans total 2.59 ms/batch,
> 0.20 s over the render**, against 151 ms/batch before the harness fix. Almost
> all of the original "before" was the artifact.
>
> **The honest before/after**, both arms timed inside the render's own context
> (`_actor_filter_before_after_s05.py`):
>
> | | per batch | over 77 batches |
> | --- | --- | --- |
> | pre-P4 actor filter expression | 48.16 ms | 3.71 s |
> | post-P4 actor index | 0.73 ms | 0.06 s |
> | | **66x** | **3.65 s saved** |
>
> So P4 is worth roughly **3.7 s of a ~500 s render (~0.7%)**, essentially all
> of it the actor filter, which had no cache at all and re-walked every actor's
> `TimelineEvent` chain every batch. The event lookups were a much smaller part than
> claimed. Note even the 96 ms/batch originally measured for the filter was ~2x
> inflated: recording during prep bumps the global timing revision, and
> `actor.lifespan.start()` memoizes against it, so the artifact was invalidating
> that memo too.
>
> Note also `actor_query` is now the largest of the four at 38.95 ms/batch,
> against 0.91 ms measured for the whole filter in the harness -- a 40x gap.
> Prep runs on the prefetch worker while the render thread holds the GPU, so
> Python-level work there is far more expensive than the same code measured on
> an idle main thread. **Treat every harness-measured prep number in this
> document as a lower bound on its real cost, and the reverse for anything that
> depends on recording.**

### P5 -- the replay checkpoint grows in place (shipped)

> **CORRECTION FIRST: the premise of this item was wrong.** It was written from
> a direct-`get_batch_of_primitives` harness, where recording during prep
> un-resolves the timeline and makes the resolver re-run every batch. A real
> render records nothing, so **`_resolve_replay_windows` resolves exactly once
> per render** (measured: 1 resolve over 58 batches of s05). The clone this
> removes therefore happened once per render, not once per batch, and the
> change is worth ~7 MB and ~1.5 ms per render -- not the O(n^2) term claimed
> below. It is kept because it is strictly better and already validated, but it
> should not be counted as a win. The original (incorrect) reasoning follows.
>
> `_resolve_replay_windows` runs on **every batch** of a render (a render
> records edits of its own, which un-resolves the timeline), and its prologue
> rebuilt `_resolved_row_ends` by cloning one float64 buffer per attribute
> sized by that attribute's *total* row count. On s05: **7.2 MB allocated per
> batch across 9 attributes**, and O(n^2) over a render since both the row
> count and the batch count grow with the scene.
>
> Now grown in place, with capacity doubling. Measured: **zero reallocations
> and 0.0 MB across every steady-state batch**, and total resolve 8.3 -> ~7 ms
> (the allocation was only ~1.5 ms of *time*; the point of this one is the
> allocation shape, not the clock).
>
> Two things make it sound, and both are load-bearing:
>
> * **The update is a monotone max-assign**, so re-applying a partially applied
>   suffix recomputes the same ends. That is what makes in-place mutation safe
>   without an undo log if a resolve is interrupted.
> * **`preserving_authoring_state` now copies the checkpoint** instead of
>   holding the dict, and drops `_row_ends_capacity` on restore. Every
>   re-renderable render enters that block (`save_frame`, `show_frame`,
>   `save_video(reset=False)`), and it restores the checkpoint on exit -- with
>   in-place growth, holding the dict alone would hand back views the render
>   had already overwritten. This moves one O(rows) copy from *every batch* to
>   *once per render*.
>
> **`benchmarks/_resolve_rollback_check.py` guards that second point, and the
> first version of it was vacuous** -- worth knowing before trusting it. A
> render only grows the checkpoint if it records while preparing batches; s05
> appends 77-304 edits a batch, but a small scene appends none, so the check
> passed with the copy deliberately removed. Adding `Text` and `Sphere` did not
> fix it. It now records inside the block explicitly and **asserts that the
> checkpoint actually changed** before checking that it was restored, so it
> cannot silently go vacuous again. Verified by re-introducing the aliasing
> bug: it fails with "location values changed across the render (19 of 1292
> rows differ)".

### P1 -- the compact rematerialization buffer (shipped)

> **Shipped**, gated by `ALGAN_OPT_DISABLE=compactstate`.
> `rematerialize_state_at_times` materializes only the window's live rows and
> keeps a global-row -> column map (`_set_active_row_map`), instead of building
> `[T, N, D]` and scattering ~30% of it into place.
>
> | | before | after |
> | --- | --- | --- |
> | materialization + replay reads | 2.474 s/pass | **1.801 s (1.37x)** |
> | buffers allocated per batch | 604.4 MB | **203.9 MB (34%)** |
> | widest single buffer | 289.2 MB | **76-133 MB** |
>
> Both halves of the original target at once: the `index_copy_` scatter is gone
> *and* so is the O(N)-per-attribute-per-batch commit. The map is the one thing
> that stays full width, but at 8 bytes a row against `T * D * 4` it is ~150x
> smaller than the buffer it replaces, and it is grown in place and rewritten
> only where it changed, so a batch costs O(active rows).
>
> Three properties carry it:
>
> * **`active_rows` is ascending**, so the map is the rank of a row within it
>   and therefore monotone. That is what lets `_compact_span` decide a whole
>   contiguous mob range from its two endpoints alone -- and what keeps
>   `get(copy=False)` returning a real **view**, the constraint this document
>   flagged as the hard one.
> * **Unmaterialized rows read as zero**, matching the zeroed full-width buffer.
> * **The lifespan mask applies straight to the compact buffer** -- it is
>   already in `active_rows` order, so the gather-modify-scatter through global
>   layout disappears (and with it the reason the opacity-only mask fusion was
>   ever worth considering separately).
>
> **The full-render suite caught a bug parity did not, and the reason
> generalizes.** `text_and_media` failed with `IndexError: index is out of
> bounds for dimension with size 0`: a window whose working set is empty leaves
> a `[T, 0, D]` buffer, and a read of any row then has no column to gather
> from. The full-width buffer always had a real zeroed row for it, so this is
> not a wrong value -- it is a shape that only becomes possible once the buffer
> is compact, and a parity script comparing *values over live rows* cannot see
> it. `get` now builds zeros and fills the live entries rather than gathering
> and masking, and `_compact_index` returns unclamped columns so a `-1` cannot
> quietly become column 0. `benchmarks/_compact_state_parity.py` opens with
> that case now.
>
> The general lesson, and it applies to the next change of this kind:
> **parity over the values a change computes does not cover the shapes it newly
> makes possible.** Run the full renders.
>
> **It costs ~3% on small scenes, and that is the intended trade.** A/B over the
> fast suite: 153.7 s with `compactstate` disabled, 158.3 s enabled. A tiny
> scene has `R` ~ `N`, so the compact buffer saves nothing while still paying
> to maintain the row map and translate every accessor call. The saving is
> proportional to how *dead* the buffer was, which is a property of large
> scenes -- 34% of full width on the reference scene, ~100% on a unit test.
>
> **One trap inside that, which cost 50 s of fast-suite time before it was
> found:** `_compact_span` reads two scalars from the map per accessor call,
> and `int(some_torch_tensor[i])` costs ~10 us against ~100 ns for numpy. With
> the map read through torch the fast suite went 130 s -> 184 s. It now keeps a
> **zero-copy `buffer.numpy()` view** purely for those scalar reads (rebuilt
> whenever the buffer is reallocated; in-place writes stay visible through it).
> Anything else added to this hot path should be priced on a suite of small
> scenes, not on s05, where per-call overhead disappears into the batch size.



#### The measurement that re-scoped it, kept as evidence

An earlier revision called this "the one prep change with a large payoff" and
scoped it as the zero-fill plus the `index_copy_`. Measuring first
(`_p1_probe_s05.py`, 441 calls over a full render) said otherwise:

| part of `rematerialize_state_at_times` | share |
| --- | --- |
| `_query_row_states` -- the compact query itself | **42.7%** |
| residual (lifespan mask, `active_rows` union/convert, misc) | 17.9% |
| the `[T, N, D]` zero-fill | 18.8% |
| the `index_copy_` scatter | 14.9% |
| `rows_for_mob_ids` | 5.6% |
| `prepare_for_queries` | 0.2% |

That is why P3 (the lazily-zeroed fill) went first and was worth most of the
time, and why the compact buffer's real payoff turned out to be the **memory**
(604 MB/batch -> 204 MB) rather than the clock. Two notes worth keeping:

* The old suggested fallback -- "`torch.empty` + explicit zeroing of only the
  *inactive* rows" -- is backwards: inactive rows are ~69% of `N`, so zeroing
  only those costs *more* than one contiguous `memset`.
* **`_query_row_states` is 42.7% and is now the largest single item left in
  prep.** No entry in this document has ever targeted it. It runs
  `searchsorted` over the flat composite key for every (row, distinct-rank)
  pair -- for `location`'s 140 000 live rows and ~50 frames that is ~7 M binary
  searches of ~20 random accesses each. The rank dedup already collapses
  frames; rows are not deduped at all, and most rows have one or two edits.
  Hard constraint before designing: **this runs on the animation device (CPU)
  and must stay there** -- see the "Not worth a kernel" note about Taichi
  staging CPU tensors through VRAM.
  **(Since re-scoped: this paragraph did not survive P1. The row dedup it
  suggests shipped as P6 and bought 1.051x -- the query is write-bound now.)**

**One thing measured and ruled out along the way:** reusing the buffer across
batches instead of reallocating it. Batch prefetch forbids it -- prep for batch
b+1 runs on a worker while b renders, and it is the *reallocation* that keeps
b's handed-out views valid. A shared buffer would have b+1 overwrite what b is
still reading. (The compact buffer is still reallocated per batch; it is only
the row *map* that is grown in place, and nothing hands out views into that.)

### P6 -- endpoint row dedup in `_query_row_states` (shipped, minor)

> **Shipped**, in `_query_row_states` (`timeline.py`), gated by
> `ALGAN_OPT_DISABLE=rowdedup`. Byte-identical: parity against both the dense
> path *and* a brute-force per-(time, row) evaluation straight from the edit
> list (`benchmarks/_query_rowdedup_parity.py`, which asserts every branch was
> exercised rather than assuming it); fast suite x3 (133.5 s on run 3, in
> band) and the full render suite (7/7, pixel-identical) pass unchanged.
>
> The idea: the key position a row's search lands on is monotone in the query
> rank, so searching each row once at the window's lowest and highest rank
> identifies every row whose answer is constant across the whole window --
> those need one search-pair and one value gather instead of S of each, and
> the result is broadcast. Only rows with an edit boundary inside the window
> pay the per-rank search. The engage test is the exact search-count
> break-even (2R endpoint searches against S * n_const saved), so a
> mostly-changing window keeps the dense path and the overhead is bounded.
>
> **Measured on s05: 1.051x on the query, ~0.02 s per 6-window pass**
> (`AB_OPT=rowdedup _prep_timeslice_ab_s05.py`) -- not the win the ranking
> predicted, and the miss is the finding. The "42.7% of remat, ~7M binary
> searches" premise was measured *before* P1 shipped: with the buffer compact
> (R = live rows only) and ranks already deduped, the query's cost is its
> output writes and the rank->frame expansion, not the searches. Kept because
> it is byte-exact and free where it does not help, but **do not count it as
> a win, and do not spend more on search-count reductions here** -- the next
> lever on remat, if it is ever worth one, is write traffic.

### P7 -- the updater section, and what actually costs in replay (shipped)

> **The measurement first, because it overturns the previous revision's item 1.**
> `_replay_loop_probe_s05.py` now times each *section* of the replay loop
> rather than lumping the residual together. Shares of `set_state_to_times`,
> unprofiled run, after P7:
>
> | section | share |
> | --- | --- |
> | **updater section** (bodies + their accessor calls) | **31.2%** (was 43.9%) |
> | `rematerialize` (tracked separately) | 23.4% |
> | rate functions | 6.7% |
> | `ev/interp` (elapsed, `a`, the `where`/clamp) | 3.5% |
> | `ev/window` (mask + `nonzero`) | 3.2% |
> | `ev/lerp` (kwargs + `cast_to_tensor` + `torch.lerp`) | 2.1% |
> | `ev/select` (selector + `set_active_time_inds`) | 2.0% |
> | event bodies net of accessors | 0.9% |
> | pre-loop lookups, final reset | 0.3% |
>
> **The per-event machinery the previous revision proposed batching into one
> `[F, T]` pass is ~11% of the stage in total, not 59%.** The earlier "59.3%
> residual" was a residual, and it was mostly the *updater* loop, which that
> revision's arithmetic silently folded into per-event overhead. A batched
> window-test/interpolant pass is therefore worth at most ~2.5% of
> `save_video`, before any of it is actually recovered -- not the ~14% claimed.
> Two ranked items in a row have now died the same way (see P6); the rule that
> keeps earning its keep is **measure the parts before designing for them**.
>
> Two fixes shipped against what the probe did find:
>
> * **The clone memo.** `_register_known_history_clones` ran on every traced
>   updater Mob access -- 65 640 `register_history_clone` calls per pass -- to
>   re-derive a mapping that changes only when `Mob.detach_history` runs. Now
>   memoized per (event, mob id) behind a version counter that
>   `register_updater_history_split`, the registry's only writer, bumps.
>   **1.042x on the whole replay stage** (3.206 -> 3.076 s/pass,
>   `AB_OPT=clonememo _prep_timeslice_ab_s05.py`), and
>   `trace_updater_mob_access` 1.211 -> 0.262 s cumulative (4.6x).
>   Gated `ALGAN_OPT_DISABLE=clonememo`.
> * **Batched-axis `get_orthonormal_vector`.** It built the perpendicular basis
>   with a Python loop over the three standard-basis seeds (~40 small
>   dispatches) and is called per animated Cylinder point-move -- 1104 calls a
>   pass on this scene, at ~1.08 ms each. All axes now project in one pass over
>   a new axis dim. **Bit-identical** and **1.45x** on the updater's real shape.
>
> Together: updater section 2.276 -> 1.853 s and the updater body 4.970 ->
> 3.927 s (1.27x), both arms measured under the same cProfile attribution.
>
> **Guards.** `benchmarks/_orthonormal_bitwise_ab.py` inlines the pre-rewrite
> implementation and compares bit patterns (not just `==`, so a signed-zero
> flip cannot pass) over 15 cases including axis-aligned seeds, spanning pairs,
> zero vectors, NaN and float64. `benchmarks/_updater_clone_memo_parity.py`
> compares full per-row state across windows of a scene that interleaves an
> updater with `detach_history`, asserts the run actually registers clones,
> bumps the version and hits the memo, **and mutates the invalidation away to
> prove the check can fail** (it does: 78 of 120 location rows differ).
> Fast suite x3 (127 s on run 3, in budget) and full renders 7/7 unchanged.
>
> **Three traps worth keeping:**
>
> * **The obvious mutant does not work.** Restoring the version counter *after*
>   `register_updater_history_split` returns tests nothing: that function
>   re-registers every updater's clones before it returns, while the counter is
>   still bumped, so the damage is already undone. The mutant has to model a
>   memo entry outliving the registry change -- i.e. remove the version check
>   itself.
> * **`cProfile` around the updater bodies inflates them ~20%** and dropped the
>   probe's section coverage from 99.5% to 79.9%. It is now opt-in
>   (`PROBE_CPROFILE=1`). Read a profiled run for *attribution* and an
>   unprofiled one for *time*, never one for both.
> * **`torch.where` chains are not `argmax`.** The batched rewrite must keep the
>   original's zero-initialised, strictly-greater, first-max-wins selection:
>   `argmax` differs on NaN, and starting from axis 0 instead of zeros differs
>   when every residual is NaN. Both were caught by the bitwise A/B, not by the
>   suites.

### P8 -- collating the per-descendant fan-out (shipped)

> **Shipped**, gated by `ALGAN_OPT_DISABLE=collate`.
> `Animatable._create_recursive` / `_destroy_recursive` classify the subtree
> first (`_collatable_members`), record **one** animation for every Mob whose
> entrance/exit is the standard opacity write (`_collated_fade_in` /
> `_collated_fade_out`), and then run the untouched per-Mob walk for whatever is
> left. The collated write is addressed by an explicit `RowRanges` **scope** --
> a third addressing mode threaded through `_get_attr_ranges`,
> `get_animated_attribute` and `_setattr_and_record_modification` as the private
> `_scope` argument, and carried by `Mob._apply_change`'s recorded `scope`
> kwarg so replay lands on the rows the recording wrote. `Mob.map_animated_attribute`
> is the same primitive as public API.
>
> | measured on s05 (prep only, CPU, 5 rounds) | before | after | |
> | --- | --- | --- | --- |
> | recorded events | 25 582 | 12 659 | **2.02x** |
> | `set_state_to_times` | 2.527 s | 1.540 s | **1.64x** |
> | `AttributeTimeline.get` (replay calls) | 6519 | 3803 | 1.29x on time |
> | whole 6-window prep pass | 5.393 s | 4.325 s | 1.25x |
> | fast suite (small scenes) | 166 s | 159 s | no small-scene tax |
>
> Estimated ~1.17x on the reference render (~60 s of 397 s) on the two-pole
> model. **The remaining 4667 events are the `dim_mobs` family**, which is
> video-project code, not engine code: it loops
> `d.set_non_recursive(opacity=d.opacity * f)` over `get_descendants()` and can
> now be one `map_animated_attribute('opacity', lambda o: o * f)` call. That is
> the difference between this and the 3.20x the prototype measured.
>
> ### CONFIRMED END TO END (2026-08-16), with the `dim_mobs` half shipped
>
> `dim_mobs` / `restore_mobs` in the video project now call
> `map_animated_attribute`, so both halves are in. Re-profiled on the reference
> scene:
>
> | | 2026-08-15 | 2026-08-16 | |
> | --- | --- | --- | --- |
> | `set_state_to_times` own | 92.5 s (23.3%) | **56.4 s (16.2%)** | **1.64x** |
> | `AttributeTimeline.get` | 84.9 s / 647 399 calls | 57.1 s / 542 052 calls | 1.49x |
> | `rematerialize_state_at_times` | 38.4 s | 39.7 s | flat |
> | prep pole | 310.2 s (78.2%) | 257.2 s (73.7%) | 53 s |
> | render thread | 190.3 s (48.0%) | 189.6 s (54.4%) | unchanged |
> | **`save_video`** | **396.7 s** | **348.75 s** | **1.14x** |
>
> **The stage prediction held exactly; the render prediction did not.** 1.64x on
> the stage is what the harness said. But "~1.23-1.33x on the render" assumed
> the poles keep overlapping as they did before, and they do not: the saving
> came off the *larger, already-overlapped* pole, so a third of it vanished into
> the render thread's shadow. 1.14x is the honest figure. **The rule this earns:
> a two-pole extrapolation is an upper bound, not a prediction** -- quote it as
> "at most", or measure the render.
>
> **It is not byte-identical, and the reason is a behaviour fix.** An ancestor's
> `Mob.on_destroy` is a *recursive* opacity write, so despawning a group faded a
> `Tex`'s glyphs uniformly on top of the diagonal wave its own `on_destroy`
> records -- the wave was authored and then overwritten, and the text left as a
> plain fade. Collated, a Mob with its own exit is excluded from the ancestor's
> write and its wave is what plays. Verified frame by frame: the difference is
> confined to that Mob's own rows, and both paths converge again once the exits
> finish. `shapes_and_timeline` and `text_and_media` move; the diffs are glyphs
> only (1.1% and 16% of frame area, nothing non-text at any frame).
>
> **Three traps, each of which cost a full render suite to find:**
>
> * **`context.get_end_time()` reads a timespan the exit animation itself
>   extends.** Stamping despawn times before the collated write puts them at the
>   *start* of the fade, which masks every Mob to zero from its first frame
>   instead of fading it. Write, then stamp.
> * **Stamping them after the per-Mob walk is just as wrong, and far subtler.**
>   A subclass exit that runs longer than the fade (Text's wave does) then
>   stretches every sibling's lifespan to match it. Nothing changes visually --
>   they are already transparent -- but they stay in the batch's actor set, which
>   re-windows the render and moved `materials_and_lighting` and
>   `solids_and_camera` by up to 189 channel values of sub-visual speckle across
>   56% of the frame. Both scenes pass once the stamp happens immediately after
>   the collated write. **A lifespan that is longer than it needs to be is a
>   rendering change, not just bookkeeping.**
> * **A subclass hook may claim its own subtree** -- `Text.on_create` calls
>   `_create_recursive(animate=False)` so its glyphs do not also fade in
>   individually underneath its wave. So the classification pass must stop
>   descending at any Mob with its own hook, and the per-Mob walk afterwards
>   picks up anything a hook turns out not to claim.
>
> Guarded by `benchmarks/_collated_fanout_parity.py`: a strict scene that must
> match **bit-exactly** per row and per frame on materialized state (it does),
> plus the custom-exit scene where the difference is *required* to be confined to
> that Mob's rows and to vanish once the exits finish. It asserts branch coverage
> so it cannot go vacuous, and `--mutate` breaks the implementation and requires
> the script to notice.

#### The measurement that motivated it

> **The measurement first.** `videos/rl2/animations/_authored_funcs_s05.py`
> attributes every one of s05's 25 582 recorded events to its authoring site.
> Almost none of them are distinct authored animations -- 99.5% are
> *per-descendant fan-out* of a few hundred subtree-wide operations:
>
> | mechanism | events | share |
> | --- | --- | --- |
> | despawn fade-out -- one `on_destroy` per descendant | 8190 | 32.0% |
> | spawn fade-in -- one `on_create` per descendant | 6495 | 25.4% |
> | `set_non_recursive` in per-descendant loops in the video project | 6036 | 23.6% |
> | wave/pulse colour -- one call per part, lagged | 4502 | 17.6% |
> | everything else (moves, fits, `forward`, `NumericDisplay`) | 359 | 1.4% |
>
> By recorded function: **76.9% `Mob._apply_change`, 22.7%
> `Mob.apply_absolute_change_two`**, 92 events in the whole scene anything
> else. One `dim_mobs(...)` call at `s05:733` is **19.7% of the timeline**;
> a single `spawn()` records up to 1376 events and a single `despawn()` 1246.
>
> The first three families all write **one uniform thing over a contiguous
> subtree**, which the engine can already express as a single recursive event
> (`_apply_change(..., recursive=True)` writes every descendant row and records
> once). `videos/rl2/animations/_fanout_collate_ab_s05.py` A/Bs that:
>
> | | arm A (stock) | arm B (collated) | |
> | --- | --- | --- | --- |
> | recorded events | 25 582 | 7 992 | 3.20x |
> | `opacity` edit records | 26 449 | 3 700 | 7.15x |
> | rows those edits cover | 365 793 | 319 287 | 1.15x |
> | **`set_state_to_times`** | **3.497 s** | **1.757 s** | **1.99x** |
> | `rematerialize_state_at_times` | 0.756 s | 0.613 s | 1.23x |
> | `AttributeTimeline.get` | 0.773 s | 0.527 s | 1.47x |
> | whole 6-window prep pass | 7.177 s | 4.890 s | 1.47x |
> | authoring | 84.1 s | 76.4 s | 1.10x |
>
> **1.99x on the replay stage, reproduced three times** (1.999x / 1.993x /
> 1.991x on independent run pairs; the arms' per-round distributions do not
> overlap). Extrapolated on the two-pole model -- prep 310.2 s of a 396.7 s
> `save_video`, 86.5 s of it not overlapped, render thread 190.3 s -- that is
> **~1.23-1.33x on the render (75-100 s)**, the lower figure counting only the
> replay stage. Estimates, not measurements: they assume the saving scales and
> that the poles overlap as they did on the reference profile.
>
> **Arm B is state-equivalent, and that is checked rather than asserted.**
> Live-row state matches on all nine attributes to four decimals with zero
> NaNs, the actor set selected per window is identical, and `get` calls made by
> the *geometry* build are identical (12 030 both arms) -- all 2706 saved calls
> are in replay. So nothing was deleted; the same rows are written by 3.4x
> fewer records. How each family collates:
>
> * **spawn** -- arm A's `on_create` fades each descendant from 0 to *its own*
>   opacity, which is why it is non-recursive. Arm B captures the subtree's
>   per-row opacity vector, zeroes the subtree in one unrecorded write, and
>   animates one recursive `_apply_change` whose `change` **is that per-row
>   vector**, so each row still lands on its own value.
> * **despawn** -- `Mob.on_destroy` is *already* a recursive write, so arm A's
>   per-descendant calls re-write subsets of what the root already wrote: a row
>   at depth d collects d+1 edits. Arm B calls it once at the root.
> * **`dim_mobs`** -- `d.set_non_recursive(opacity=d.opacity * f)` per
>   descendant becomes one recursive change of `cur * (f - 1)`. Note this half
>   is a *user-code* change (or a new recursive scale-opacity API), not an
>   engine one.
>
> **The three families are not equally safe, and only a per-family run shows
> it.** The digest above compares *authored* state; `_materialized_digest`
> compares `active_state` after each window -- the compact `[T, R, D]` buffer
> the primitive builders actually read. Collating one family at a time:
>
> | family | events saved | materialized state |
> | --- | --- | --- |
> | `dim_mobs` (set family) | 5003 | **48/48 attribute-windows bit-identical** |
> | spawn | 5159 | 43/48 bit-identical; sums equal at rel 0.0e+00 in all six windows, but the window-0 buffer is 28 773 rows against 28 766 -- a working-set (layout) difference, not a value one, as far as buffer-level sums can show |
> | despawn | 7428 | 47/48 bit-identical; **window 1600 differs for real** -- opacity sums 502 504.0 vs 500 901.2 (rel 3.2e-03) |
>
> The event savings are exactly additive (5003 + 5159 + 7428 = 17 590), and the
> saving per pass tracks events removed at ~1.3e-4 s/event: `dim` alone
> measured 6.558 s against arm A's 7.177 s (0.62 s for 5003 events) where the
> proportional prediction is 0.65 s. So allocate the 2.29 s roughly
> despawn 0.97 s / spawn 0.67 s / dim 0.65 s.
>
> **Why despawn moves output**, and it is not a bug in the collated form:
> `Mob.on_destroy` is already recursive, so arm A writes a row at depth d
> *d+1 times* with overlapping edits, and `_resolve_replay_windows` extends
> each edit's window over the earlier-executed ones it overlaps. One clean edit
> resolves differently from a nest of redundant ones, so intermediate frames of
> a fade differ even though the end state is identical. The collated result is
> arguably the more correct one, but it is a **visual change: it needs
> re-baselining and someone looking at the frames** (the T1 precedent).
>
> **What none of this proves.** Arm B is a harness patch, not an
> implementation. It keeps the per-mob path for subclasses that override
> `on_create` / `on_destroy` (251 + 164 calls -- Text/Tex wave in and out) and
> falls back wholesale on partially-spawned subtrees (205 despawns), and both
> have to survive into a real version. And buffer-level sums are not per-mob
> equality: a parity script for this must compare per-row, per-frame.
>
> **Four traps, each of which cost a measurement:**
>
> * **`current_state` is `torch.empty`**, so a fingerprint over the whole
>   buffer compares uninitialised rows -- it reported `glow` as NaN and
>   `border_width` as 8.5e31 in *both* arms and flagged spurious differences on
>   two more attributes. Compare only rows in `mob_id_to_inds`.
> * **The whole-pass number is noise-dominated; the replay stage is not.** Arm
>   A's rounds spread 6.35-10.35 s. The geometry-side residual duly shrank from
>   1.49x to 1.17x as rounds went 3 -> 7, which is what a noise artifact looks
>   like. Quote the stage, not the pass.
> * **Two authored s05 timelines do not fit in RAM** (3.4 GB free of 16), so
>   this is one arm per process rather than the in-process alternation
>   `_prep_timeslice_ab_s05.py` uses. Run the arms alternately if drift is a
>   worry.
> * **Removing the fan-out is not the same experiment as collating it.**
>   Deleting the events would leave mobs at opacity 0, change what is visible,
>   and quietly measure a cheaper *scene*. The row-coverage row in the table
>   above (1.15x, not 26x) is what rules that out.

### P10 -- `get_batch_of_primitives`' "own time" is the batched surface build (measured, not started)

> **The largest single item in the profile was an artifact of the hook list.**
> Every revision of this document has called `get_batch_of_primitives`' own time
> "geometry-build orchestration" and left it unmeasured. Split into sections
> (`videos/rl2/animations/_gbop_section_probe_s05.py`, six windows across the
> scene, prep only, CPU), a 7.48 s pass is:
>
> | section | s/pass | share | tracked by the profiler? |
> | --- | --- | --- | --- |
> | `set_state_to_times` | 2.636 | 35.3% | yes |
> | per-actor bezier build | 2.002 | 26.8% | yes |
> | **deferred surface build** | **1.762** | **23.6%** | **no** |
> | collection construction | 0.442 | 5.9% | no |
> | unattributed orchestration | 0.460 | 6.2% | no |
> | deferred bezier build | 0.091 | 1.2% | yes |
> | dispatch predicates (5052 calls) | 0.064 | 0.9% | no |
> | `actor_query` | 0.015 | 0.2% | no |
> | `_materialize_render_state` | 0.005 | 0.1% | yes |
>
> The untracked rows sum to **36.8%** of the pass against the profiler's
> **32.3%** own-time share of the stage (82.98 s of 257.18 s) -- consistent, so
> the attribution holds, and it points at `get_render_primitives_batched` in
> `mobs/surfaces/surface.py`.
>
> Why it was invisible: its bezier counterpart `build_render_primitives_batched`
> has been hooked all along, but the surface one never was, and on this scene
> *every* surface is batchable -- every batch reports `num_pn=0` and
> `Surface.get_render_primitives` does not appear in the report at all. So the
> whole cost fell into the stage's exclusive column with no row of its own.
>
> #### Confirmed by hooking it (re-profile, warm run 2, `save_video` = 390.47 s)
>
> | | measured | was |
> | --- | --- | --- |
> | **`surfaces: get_render_primitives_batched`** (172 calls) | **85.35 s excl (21.9%)**, 89.45 s incl | no row at all |
> | `get_batch_of_primitives` own | **17.66 s (4.5%)** | 82.98 s (23.8%) |
>
> **The stage's own time collapses to 4.5%.** Essentially all of it was this one
> function, and the harness split above *understated* it: ~53 s inferred against
> ~85 s measured, which is now the **largest single item in the whole render**.
> (This run is thermally slower than the 348.75 s one -- 390.47 s end to end, the
> usual `SwThermal` -- so compare its shares, not its seconds, against the table
> above.)
>
> #### Inside it: 60% is `compute_grid_vertex_normals`
>
> `videos/rl2/animations/_surface_build_probe_s05.py` splits the function by
> nesting, so a callee reached from the per-surface tail is charged separately
> from the same callee run on the whole stack. Per pass (12 calls, 223 surfaces,
> 18.6 per call), the function costs 1.396 s -- **22.7% of the prep pass against
> the profiler's 21.9% of the render, so the attribution transfers**:
>
> | section | s/pass | % of fn |
> | --- | --- | --- |
> | **`compute_grid_vertex_normals`** | **0.836** | **59.8%** |
> | `grid_to_triangle_vertices` (whole stack) | 0.191 | 13.7% |
> | per-surface primitive construction | 0.125 | 8.9% |
> | per-surface `grid_to_triangle_vertices` | 0.114 | 8.1% |
> | per-surface residual + timeline reads | 0.094 | 6.7% |
> | everything else | 0.036 | 2.8% |
> | **shared prefix / per-surface tail** | | **74.8% / 24.2%** |
>
> The obvious hypothesis going in -- that the "batched" build is only batched for
> grids, normals and corners while colours, shader parameters and construction
> stay per-surface -- **is true but second order**: the whole per-surface tail is
> 24.2%. The shared prefix is three quarters of it, and one function is 60%.
>
> **This measurement was wrong the first time, in an instructive way.** The probe
> wraps module-level helpers, and the first version timed them *everywhere in
> prep*, not just inside the target: `get_animated_attribute` duly reported
> 16 670 calls against 223 surfaces and came out as the largest row. The guard
> that fixes it is one flag set by the outer wrapper. **A wrapper on a shared
> helper measures the whole program unless it is scoped.**
>
> Three things follow. First, **item 1 is a real target, not bookkeeping**, and
> P11 below is the first cut into it. Second, the general lesson, which has now
> cost two rounds: **an unhooked callee does not show up as a missing stage, it
> shows up as its caller's own time** -- and "own time" reads like irreducible
> overhead, which is exactly the wrong conclusion. When an exclusive column is
> large and unexplained, check the hook list before designing. Third, the
> per-surface tail (24.2%) is a smaller, separate item: batching the colour and
> shader-parameter gathers the way the grids already are.
>
> `actor_query` deserves a footnote from the section probe: **0.015 s per pass
> there against the 38.95 ms/batch this document measured inside a real
> render.** That gap is the prefetch-worker effect (see the end of this
> document), not a change in the code, and it is the sharpest illustration of it
> yet -- a CPU harness cannot see that item at all.

### P11 -- pairwise triangle sides in `compute_grid_vertex_normals` (shipped)

> **Shipped**, bit-identical, gated by `ALGAN_OPT_DISABLE=gridnormals`.
> **2.0-2.2x on the shapes the batched build actually passes.**
>
> The four triangles around a grid vertex use each rolled neighbour twice --
> once as the second side of one triangle, once as the first side of the next --
> and the code made that literal. It stacked eight copies of the grid into
> `[..., W, H, 8, 3]`, subtracted the grid from all eight, sliced two **stride-2**
> views back out to cross, and then reduced a `[..., W, H, 4, 3]` tensor over its
> triangle axis. Now each neighbour is differenced once, the four pairs are
> crossed directly, and the four results are added.
>
> | case | before | after | |
> | --- | --- | --- | --- |
> | `[19, 50, 24, 12, 3]` (the real shape) | 110.5 ms | 50.3 ms | **2.20x** |
> | `[19, 50, 40, 20, 3]` | 297.5 ms | 146.0 ms | **2.04x** |
> | small grids (a few thousand points) | | | 1.0-1.6x |
>
> **Bit-identical, and that is asserted on bit patterns rather than `==`** so a
> signed-zero flip or a changed NaN payload cannot pass
> (`benchmarks/_grid_normals_ab.py`, 13 cases: open / x-closed / pole grids,
> single-column and two-row degenerates, collapsed columns, float64, NaN, inf,
> signed zeros, and the batched stack). Every operation is elementwise on the
> same values in the same order; the one step that needed checking is dropping
> `stack(...).sum(-2)` in favour of `n0 + n1 + n2 + n3`, since float addition is
> not associative -- verified that a length-4 reduction over a contiguous axis
> *is* that sequential order, bitwise, before relying on it.
>
> **Two traps.** First, **the first A/B said 0.93x on the batched case and it was
> noise**: ten un-alternated iterations of a sub-millisecond op on this machine.
> Alternating the arms per round and taking medians turned the same code into
> 1.46x. This document has warned about wall-clock A/Bs before; it applies at the
> microbenchmark scale too. Second, **the small-grid rows are dispatch-bound and
> say nothing about the win** -- the paired form runs *more* torch calls, so it
> only pays once the tensors are large enough for bandwidth to dominate. Read the
> `REAL` rows.
>
> #### Confirmed end to end (re-profile, warm run 2, `save_video` = 358.05 s)
>
> | | before P11 | after P11 | |
> | --- | --- | --- | --- |
> | `surfaces: get_render_primitives_batched` excl | 85.35 s (21.9%) | **56.62 s (15.8%)** | **1.51x** |
> | prep pole | 302.24 s (77.4%) | 263.60 s (73.6%) | |
> | `save_video` | 390.47 s | 358.05 s | 1.09x |
>
> **The stage prediction was right this time**: the microbenchmark said 2.0-2.2x
> on 59.8% of the stage, which predicts `0.402 + 0.598/2.1 = 0.69` -> 1.45x, and
> the measured figure is 1.51x. Both runs carry the hook, so they are directly
> comparable in structure.
>
> Read the 1.09x on the total with more caution than the stage: these runs sit
> 390.5 s and 358.0 s apart on a thermally throttled machine, so some of that
> gap is not P11. The stage's *share* -- 21.9% -> 15.8% of its own run -- is the
> number that does not depend on thermal state.

### P12 -- packed surfaces: one Mob for a whole collection (shipped)

> **The cheapest primitive build is the one that was never a separate build.**
> P10 and P11 attacked `get_render_primitives_batched`, which re-batches N
> separate surface actors into one tensor pass **every frame batch**. A packed
> surface does that batching **once, at construction**, and then there is only
> one actor and one build to begin with.
>
> The mechanism already existed on the render side and was only reachable the
> expensive way. `Surface._packed_grid_count` / `_reshape_grid_for_render` /
> the per-shell `mesh_ids` stamp have handled a concatenated grid all along,
> but the only way to *get* one was `batch_mobs`, which packs Mobs that already
> exist -- so it removed the per-frame cost and none of the construction cost.
> `Surface.from_batches` builds the packed grid directly, the way
> `BezierCircuitCubic.from_batches` has always built a page of text.
>
> Measured on this machine (CPU, `Sphere(resolution=(8,4))`, 3 warm passes):
>
> | | cost |
> | --- | --- |
> | construct N spheres + `batch_mobs` (N=2048) | 2.258 s |
> | **`Sphere.from_batches` (N=2048)** | **0.006 s** -- **378x**, and flat in N |
> | primitive build/frame, 256 separate actors | 0.2133 s |
> | same 256 via `get_render_primitives_batched` | 0.0527 s (4.0x) |
> | **same 256 as one packed Mob** | **0.0040 s -- 54x / 13x** |
> | Scene actors for 256 spheres | 512 -> **2** |
>
> And the case that motivated it, end to end -- `PMobject` built one `Dot3D`
> per point and packed them afterwards, and now calls `from_batches` once:
>
> | `PMobject(points=rand(N, 3))` | before | after |
> | --- | --- | --- |
> | N = 1000 | 1.507 s | 0.174 s |
> | N = 5000 | 6.742 s | **0.025 s** |
>
> **Read the third row against P10 before budgeting.** The 13x is against the
> cross-actor batcher, so on a scene whose surfaces are already batchable this
> caps what is left of the 56.62 s (15.8%) P11 left behind -- but only for
> collections the author is willing to declare as one Mob. It does nothing for
> surfaces that are genuinely independent, which is most of s05. The actor-count
> drop feeds the same per-batch scans P4 measured at 0.5%, so do not budget on
> that either.
>
> **What it cost to get right, which is the part worth keeping.** Two defects,
> both silent, both in machinery that predates this work:
>
> * `parent_batch_sizes` documents itself as the map by which "the parent's
>   attribute modifications will be expanded for this animatable's attributes",
>   and **that expansion was never wired up** -- `_expand_batch_if_necessary`
>   had no callers. So `pack.move(UP)` raised a shape error on *every* pack,
>   `Text`'s glyph batch included, which is why a `Text` is moved through its
>   unbatched container. `Mob._distribute_over_packed_subtree` supplies it.
> * A subtree is addressed in **buffer order**, not descendant order --
>   `RowRanges.from_runs` sorts and coalesces. Distributing a per-member value
>   by concatenating in descendant order therefore lines up in *count* and
>   hands every member a neighbour's value. A uniform move looks perfect either
>   way; only distinct per-member values catch it, and only on a pack whose two
>   orders differ (`from_batches` builds the grid first, `batch_mobs` does not,
>   so the test parametrizes both).
>
> The general lesson matches P10's: **a caller-free helper is not dead code, it
> is an unimplemented contract.** Grep for callers before trusting a docstring
> that describes a mechanism.

### P9 -- the batched bezier build reaches 18.4% of the circuits (measured, not started)

> **Measured first** (`videos/rl2/animations/_bezier_batchability_s05.py`, six
> windows spread across the scene, prep only, no render). The vectorized
> `build_render_primitives_batched` -- which reads each attribute from the
> timeline **once for a whole group** instead of per actor -- is reached by
> under a fifth of s05's circuits:
>
> | outcome | circuits | share |
> | --- | --- | --- |
> | batched | 308 | **18.4%** |
> | **reverted by the group clash** | **863** | **51.5%** |
> | rejected by `_is_batchable_bezier` | 504 | 30.1% |
> | *of which* `empty` (returns `None` immediately, ~free) | 306 | 18.3% |
> | *of which* `batched-control-points` (real work) | 198 | 11.8% |
>
> **The dominant cause is not the per-actor gate; it is the all-or-nothing
> group revert**, and that is a surprise worth stating plainly. The code says so
> itself:
>
> ```python
> # A non-batchable primitive sharing a group's batch identifier
> # would have been concatenated into the same collection,
> # interleaved by actor order; fall back to the per-actor build
> # for such (rare) groups so the collection layout is unchanged.
> ```
>
> They are **not rare**: 198 circuits with batched control points poison 863
> otherwise-batchable peers, because one raw primitive sharing a group's batch
> identifier reverts the entire group. One window (1600) batches *nothing at
> all*.
>
> Scale, from the same profile: `build_render_primitives_batched` is 2.02 s
> inclusive for its 18.4% share, against 46.80 s inclusive for the per-actor
> path's remainder -- so **batched is ~5x cheaper per circuit**. Moving the 51.5%
> that revert is worth roughly 19 s of the prep pole; after the overlap discount
> P8 just demonstrated, expect **~1.04x on the render**, not more. Real, ranked
> first among geometry-build items, but not a headline -- and the estimate should
> be checked against P8's lesson before anyone budgets on it.
>
> **What makes it non-trivial**, and why this is a design note rather than a
> patch: the revert exists to keep the merged collection's layout identical, and
> the final grouping loop cannot express a mixed key. `grouped_primitives[key]`
> carries one class marker, and the downstream loop branches on it --
> `_PREBUILT_COLLECTION` appends collections directly while
> `BezierCircuitPrimitive` re-batches raw primitives under a size cap -- so a key
> holding both would be misread. The layout-preserving shape is to split each key
> into **maximal runs of consecutive batchable actors** in actor order, merge each
> run, and teach that final loop to walk a heterogeneous list. Byte-identity is
> the standard here, and the merged collection layout is exactly what decides it.

## Part 3 -- Authoring (90-110 s, outside `save_video`)

Diffuse: no single site above ~4%. The top costs are `AttributeTimeline.get`
(422k calls), `traverse` (8.1M calls), `animation_manager_for.collect` (1.8M
calls), `_capture_resolution_boundary_events`, `Animatable.__deepcopy__`
(881k `deepcopy` calls) and manim SVG cache I/O. Several were trimmed this
round. The remaining structural item is
`generate_animatable_attr_set_get_methods`, which installs two closures per
animatable attribute on **every** Mob instance -- ~20 entries per `__dict__`,
which every clone then deep-copies.

---

## What is left, in order

Prep is 73.6% and the render thread 56.7% (2026-08-16 re-profile, post P11), so
prep items still come first at equal size -- but only just; see item 5. Done and
off this list: the re-profile itself, `_query_row_states` (P6 -- the premise was
stale), the per-descendant fan-out (P8, **confirmed at 1.64x on the stage, 1.14x
on the render**) with its `dim_mobs` half in the video project, and the vertex
normals (P11, **confirmed at 1.51x on the surface-build stage**).

**The list is reordered, and the reorder is the point.** Three rounds of work
landed on the timeline, so the replay stage fell below the geometry build; P10
then found the geometry build's largest piece had been hiding in an unhooked
callee, and P11 cut it in half. What has risen to the top through all of it is
`AttributeTimeline.get` -- not because it grew, but because it is the one thing
every round has left alone.

1. **`AttributeTimeline.get`** -- **72.58 s (20.3%)**, 542 052 calls at ~134 us
   each: the largest single item after P11, reached the top by attrition rather
   than by growing. P8 cut the replay-side calls; the geometry-side ones have
   never been touched and are now the majority. The lever is **fewer calls**,
   and items 2 and 3 are both concrete ways of removing them, so measure the
   three together rather than attacking this row directly. Re-measure the
   `get/full` vs `get/replay` split first (`_prep_timeslice_ab_s05.py` reports
   it): the old "two thirds in the geometry build" figure predates P8, which
   changed the denominator.
2. **P9 -- widen the batched bezier build** -- measured: it reaches **18.4%** of
   s05's circuits, and **51.5% are reverted by the all-or-nothing group clash**
   that the code calls "rare". Batched is ~5x cheaper per circuit. The per-actor
   build is 40.97 s (11.4%) of own time and every one of those builds makes its
   own accessor round trips, so this cuts item 1 as well. Read P9 for the layout
   constraint that makes it non-trivial.
3. **The rest of the batched surface build (P10)** -- **56.62 s (15.8%)** after
   P11 took 1.51x off it. What is left, in the proportions measured *before*
   P11 (so re-split it before choosing):

   * **the per-surface tail, 24.2%** -- colours, shader parameters and
     primitive construction are still done one surface at a time, each with its
     own timeline reads and its own gather of the same shape as the corners that
     were just batched. The lever is to stack them exactly as the grids already
     are, and it is the part that also cuts item 1. Smaller than it looks, which
     is itself the finding: the obvious hypothesis was that this *was* the cost.
   * **the rest of `compute_grid_vertex_normals`** -- P11 took the side
     construction; the seam merge, the two pole fans and the final
     `F.normalize` are untouched.
   * **`grid_to_triangle_vertices` on the whole stack, 13.7%** -- an
     advanced-index gather expanding `[..., W*H, C]` to `[..., 3*T, C]`, run
     twice per call (corners and normals) on the same index tensor: the same
     "two gathers sharing one permutation" shape as T5.
4. **`set_state_to_times` own time** -- 64.21 s (17.9%), down from 92.5 s.
   **Measured section by section** (P7), which killed the plan an earlier
   revision put here (a batched `[F, T]` window-test and interpolant pass):
   that machinery is ~11% of the stage. Those shares predate P8, which removed
   half the events, so they need re-measuring too. What was left inside:

   * **the updater section, 31.2%** (pre-P8) -- the bodies themselves, now that
     P7 has taken the trace-registry walk and the orthonormal loop out of them.
     The remaining cost is per-Mob Python in the body: on this scene one updater
     drives ~120 neurons and their synapses, and each `move_to` /
     `set_start_point` / `move_between_points` is a separate animated call with
     its own accessor round trip. The lever is **batching across the mobs an
     updater touches** (one `[T, M, 3]` write instead of M writes), a `Mob`-API
     change rather than a timeline one. Note the general form: the same "fewer
     calls, not faster ones" family as items 2 and 3.
   * **rate functions, 6.7%** -- 2844 calls a pass, each on a small `[T,1,1]`
     tensor. Groupable by shared function object *only* for the elementwise
     ones; `torch.lerp` of the grouped result is not obviously byte-identical
     across a regrouping, so this needs a bitwise A/B before it is worth
     anything.
   * the ~11% of per-event machinery -- and if it is ever done, the caching rule
     still holds: key on length + explicit invalidation, **not**
     `TIMING_VERSION` (P4).
5. **T5** -- the sparse-discovery host chain, ~41.3 s (~11.5%): the six
   `index_select` gathers that share one permutation are one kernel. The largest
   render-thread item left, and the render thread is now **56.7%** -- its share
   has risen across three re-profiles purely because prep keeps shrinking, so
   the two poles are nearly level again and the "prefer prep at equal size"
   advice is close to expiring.
6. **T3** (5.6%), then **T6** (1.7%). **T4 is done** -- and its `allocate()`
   zero-fills, which this list used to nominate, were measured at 4% of the
   write-out and are not worth opening the dice for. What T4 left behind is a
   *CPU-path* item, not a reference-scene one: the level search is 65-85% of
   the dice without the criterion kernels, which are gated on a CUDA render
   device. See T4.

Ruled out this round: further search-count work in `_query_row_states` (P6:
write-bound now), bloom as a Taichi kernel (measured far slower), and the
`memory reclaim` gate (9.0% but small-VRAM-specific -- measure on a card with
headroom before spending anything).

**And one method note the 2026-08-16 round earned.** P8's stage measurement was
right to three significant figures and its *render* estimate was ~15% high,
because it extrapolated across two overlapping poles. The pattern to expect: a
saving on the larger, already-overlapped pole is discounted by roughly the
overlap. Quote such extrapolations as upper bounds and re-profile to get the
number -- which is also why every entry above now carries a measured share
rather than a modelled one.

**On scaling:** P4 removed the per-batch whole-scene event and actor scans, and
P1 the O(N)-per-attribute-per-batch buffer commit. P5's target turned out not to
be per-batch at all. Measured inside a real render, all the per-batch scans
together are now 0.5%, so scaling is not the lever it looked like -- **the
apparent O(n^2) was substantially an artifact of measuring through direct
`get_batch_of_primitives` calls.** Before spending on scaling again, reproduce
the concern inside a real `save_video`.

**And a live one worth fixing on its own terms:** `actor_query` costs
38.95 ms/batch in a real render against 0.91 ms measured for the entire actor
filter in a harness. Prep runs on the prefetch worker while the render thread
holds the GPU, and Python-level work there is far more expensive than the same
code on an idle main thread. That gap, not the algorithmic shape, is where the
prep pole's remaining cost lives -- and no harness in this repo can see it.

Re-profile after each: on a two-pole pipeline the ranking moves as soon as one
pole shrinks. And prefer a per-stage in-process A/B (`_prep_timeslice_ab_s05.py`)
to a whole-render wall-clock comparison -- the noise floor on this machine is
larger than most of the items left on this list.

## Maintaining the shipped kernels

`algan/rendering/raytracing/logical_pn_taichi.py` holds all three kernels
(`pn_patch_flatness_error`, `pn_edge_chord_error`, `bezier_chord_hull_error`)
plus the shared `@ti.func`s (`_screen_pixels`, `_guarded_error`, `_pn_evaluate`,
`_cubic_evaluate`, `_cubic_derivative`, `_segment_distance_squared`). They are
reached only when `SETTINGS.raytracing.experimental.pn_criterion_kernel` is on
**and** projection is already running on a CUDA render device
(`rt_settings.pn_criterion_kernel_active()`). Off, or on CPU tensors, the torch
paths are untouched and still exercised -- that is what the A/B scripts compare
against -- so this is a genuinely reversible change. The dispatch and the input
preparation (`_pn_criterion_inputs`, `_bezier_criterion_inputs`,
`_frame_broadcast_base`) live in `primitives.py`.

Traps, all of which cost real time here:

* **`_expand_frames` hands out stride-0 views** for geometry shared by every
  frame of a batch, and a Taichi ndarray needs real memory. Materializing that
  expansion would allocate the whole batch's control points to repeat one static
  mesh, so `_frame_broadcast_base` passes the single real frame plus a stride
  the kernel multiplies its frame index by. **Anything else plumbed into these
  kernels needs the same treatment**, and both branches need testing -- the A/B
  script labels each case `[broadcast]` or `[per-frame]` for that reason.
* **`nonzero()` returns a transposed view on CUDA**, which Taichi rejects
  outright. Index arrays need an explicit `.contiguous()` at the boundary.
* **A harness that builds a primitive from `mob.get_render_primitives()` leaves
  it on the animation device**, so the kernel gate silently declines and the
  benchmark measures torch against torch (it reported 0.93x before this was
  caught). `_pn_criterion_kernel_ab.py` now asserts the context is reachable;
  keep that assertion in anything new.
* **`fast_math` is a global `ti.init` flag**, so bit-identity with torch is not
  attainable per-kernel. Do not spend time trying to recover it; the reduction
  being `max` is what preserves the properties that actually matter
  (crack-freeness, determinism across launches).
