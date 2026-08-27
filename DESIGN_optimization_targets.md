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
| **P10b** the re-split, after P11 | **measured, and it re-ranks P10.** `compute_grid_vertex_normals` 59.8% -> **44.9%**, of which **76.8% is "sides + crosses"** -- the seam merges and pole fans item 12 named are ~4% of the function and cannot matter. The per-surface tail grew 24.2% -> **31.0%**, its largest row the **primitive construction (13.5%)**, which no plan named. The whole-stack gather fell to 9.5%, so fusing the two buys ~2%, not the 13.7% the queue implies. One candidate tried and **rejected on measurement**: batching the per-surface colour gather is bit-identical and **1.002x**. See P10b |
| **P11b** the sides written without a materialized roll | **shipped**, bit-identical (same `gridnormals` arm, so `_grid_normals_ab.py` covers it across 13 topologies). `roll`-then-subtract wrote the grid twice per side; writing the difference straight into its buffer, plus accumulating the four crosses in place, is **1.33x** on the sides-and-crosses block at `[120, 50, 24, 12, 3]` (1.44x on the sides alone), predicting ~1.09x on the stage. See P11b |
| **P11** pairwise triangle sides in `compute_grid_vertex_normals` | **shipped and confirmed end to end** -- **2.0-2.2x** on the function, **1.51x on the whole stage** (85.35 s / 21.9% -> 56.62 s / 15.8%), **bit-identical** (asserted on bit patterns, incl. NaN payloads and signed zeros). Removes an 8-copy stack, an 8-wide subtract, two stride-2 gathers and a length-4 reduction. See P11 |
| **P12** packed surfaces (`Surface.from_batches`) | **shipped** -- a collection of like surfaces can now be built as **one** Mob instead of N. Construction stops scaling with the member count (2048 spheres: 2.26 s -> 0.006 s), the per-frame primitive build is **54x** the per-actor path and **13x** the cross-actor batcher it makes unnecessary, and the Scene loses 2(N-1) actors. Byte-identical -- all 6 full-render scenes and `tests/fast` match their committed CPU baselines. See P12 |
| **T7** per-dimension dicing + the surface's own accuracy | **shipped, counts measured, downstream wall-clock not** -- the dice was isotropic and measured against a reference it had no right to trust that far. Both fixed: **2.2x fewer microtriangles on a sphere/torus and 8.5-38.9x on developable shapes** (cylinder, cone), same tolerance, silhouette unchanged to a fraction of a pixel. The across search costs **988 ms of a 4151 ms torus dice** on a CPU session and should be cheaper in one round. **Moves rendered output**, so every full-render baseline needs regenerating on the machine that owns it. See T7 |
| **P9** widening the batched bezier build | **shipped**, byte-identical (a lossless two-arm render: 0 differing pixels), gated by `ALGAN_BEZIER_GROUP_RUNS`. A clashing group is split into **maximal runs of consecutive batchable actors** instead of reverted wholesale -- the layout constraint is positional, not group-wide. On a clashing scene, 97.6% of circuits move from the per-actor build to the batched one and `get_batch_of_primitives` runs at **0.43-0.48x**. It also turned up a real defect in the builder it widens: it flattened curves to twice the per-actor path's chord tolerance, masked by the analytic-AA route's clamp. See P9 |
| **P13** batching an updater across its mobs | **shipped for the idle-updater family**, bit-identical at the buffer level (all attribute timelines plus non-timeline `direction`s, two frame windows, three layer sizes) and 0-differing-pixels on the nn scene; gated by `ALGAN_BATCHED_IDLE_UPDATER` (default on). The four per-mob loops of `_update_neural_net_idle` became three timeline writes: warm-batch prep **0.78x** (medians 2380 -> 1860 ms per 17-frame window on a loaded CPU box), timeline `modify` calls 258 -> 6 and `get` calls 2841 -> 1436 per batch. Reads-before-writes makes any unsupported structure fall back cleanly. See P13 |
| **T5** sparse-coverage host chain | **the host loops are done; T5's own item is the one that did NOT pay.** The compaction's per-sample-lane reductions -- which post-date this document -- are kernels now, default on, bit-identical, **1.25-1.33x on `compact_sheets`** (4K: 471 -> 354 ms, 6.5% of the frame), and the conflict-rank scan followed on 2026-08-21 (`SHEET_RANK_KERNEL`, default on, bit-identical, 33 -> 6 ms of a 1080p frame on CPU). The six-array gather T5 proposed is built and bit-identical too, but worth only ~4 ms of a 1.3 s 4K frame while costing 50-160 MB of peak, so it ships **default OFF**. Three measurement traps recorded below. The sorts are untouched and should stay that way. See T5 |
| **P13** the sides-and-crosses block as a CPU Taichi kernel | **shipped, default on** -- **2.3-5.0x on `compute_grid_vertex_normals`** (the block itself is 8-11x; the rest of the function is unchanged, so Amdahl caps it there). Dispatched only when Taichi's arch is the CPU, because on CUDA Taichi would stage every argument through VRAM on the prep worker. **Not bit-identical** -- 1-2 ulp on ~4% of elements, and the cause is `torch.cross`'s rounding on the cross product's cancelling third component, not Taichi (`fast_math=False` changes nothing). Two-arm full renders: 4 of 6 scenes byte-identical, `solids_and_camera` 13 pixels at 1 channel value, `materials_and_lighting` 0.006% of pixels past tolerance -- the documented epsilon/tie machinery. Watertightness holds structurally and is asserted. **Two sibling kernels do not pay and ship off**: the `grid_to_triangle_vertices` gather (0.84-1.20x) and `TrianglePrimitive`'s colour bake (0.89-0.92x), both byte-identical. Both were *slower* at first for a reason worth knowing -- `ti.ndrange` over several dimensions runs a copy at 1.7-4.3 GB/s against a flat 1-D loop's 14-22, same bytes (`_taichi_loop_shapes_taichi.py`). See P13 |
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
* `benchmarks/_grid_normals_ab.py` (P11 **and P11b**: bit-pattern equality of the
  vertex normals across 13 grid topologies plus timing. It A/Bs *whatever the
  paired arm currently is* against the legacy stacked form, so it covers later
  work inside that block without a new harness. Read its `REAL` rows, the small
  cases are dispatch-bound).
* `benchmarks/_bez_batch_parity.py` (the guarantee the whole batched bezier
  build rests on: every attribute of a merged group, bitwise, against the
  per-actor build. It had rotted past running and was repaired for P9; it is
  what caught the chord-tolerance drift).
* `benchmarks/_bezier_batchability.py` (P9: every circuit attributed to
  batched / group-reverted / gate-rejected under **both** arms of
  `ALGAN_BEZIER_GROUP_RUNS`, by watching which builder it reached rather than
  by re-deriving the predicate, plus an alternating-arm wall clock. The
  repo-local replacement for `videos/rl2/animations/_bezier_batchability_s05.py`.
  Note it also constructs a clashing scene, because the repo's own benchmark
  scene has no clash and measures nothing).
* `benchmarks/_bezier_run_split_ab.py` (P9: the clashing scene rendered twice in
  one process, lossless, arms flipped between renders; asserts 0 differing
  pixels, identical batch windows, and non-vacuous arms).
* `benchmarks/_surface_build_split.py` (P10b: the current split of
  `get_render_primitives_batched` and of `compute_grid_vertex_normals`, from
  instrumented copies it verifies bit-identical to the shipped functions before
  timing them. Reports per-pass **shares** with their ranges and names the rows
  it cannot separate. The repo-local replacement for
  `videos/rl2/animations/_surface_build_probe_s05.py`).
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

## The structural round (2026-08-26): texture/geometry time dedup, shadow any-hit, sync fixes

The build-out of `DESIGN_renderer_structural_candidates.md` items 1, 3.1, 5
(content dedup + copy chain) and 8 (avoidable syncs), plus item 2's
qualification, on `claude/structural-redesigns-perf-pmpkv3`. Everything
shipped is **byte-identical** on unchanged batch windows and each piece has
its own kill switch; that file carries per-item status stamps, this section
is the measurement record.

**What shipped.**

* **`TEXTURE_TIME_FLAT`** (default on): every texture map's frames are
  flattened along the texel axis with the map's own time length riding in
  `tri_tex_meta` cols 10-12 (the meta widened 10 → 13; placement quadruples
  `(offset, w, h, t)`), so the assembled `scene["textures"]` always has time
  length 1. The per-map length travels as *data* through the samplers
  (`_sample_texture` / `_sample_tex_vec5`: `texel = offset + (f % t)*w*h +
  local`), so one compiled kernel serves both layouts and the toggle flips
  in-process — no `ti.static` arm to fall into. A static map stores one
  frame whatever else animates; the environment map (which was expanded to
  the bank's T on every append) stores one copy too.
* **`TEXTURE_CONTENT_DEDUP`** (default on): `_append_texture` reuses the
  placement of an already-appended map with byte-identical processed texels
  (shape-bucketed prefilter, exact `torch.equal` match) — every textured
  primitive is a singleton collection, so N mobs sharing one image used to
  store it N times.
* **`TEXTURE_WINDOW_COLLAPSE`** (default on): `Surface.get_render_primitives`
  collapses a colour-texture window whose frames AND opacity are
  byte-identical across the batch to one frame *before* the wrap-pad /
  premultiply / decode / merge copies are made, and records the outcome on
  the mob (`_texture_window_collapsed`); the batch sizers price a collapsed
  texture at the materialized window alone (animation copies 2-3 → 1,
  render-device factor 6 → 2) off that observation — the same
  read-off-the-previous-build contract as `_texture_is_wrap_padded`, with
  the out-of-render-memory retry as the backstop for a texture that starts
  animating.
* **`MERGE_DEDUP_GEOMETRY`** (default on, rides `MERGE_DEDUP_TIME`): the
  merge-time collapse now also covers `tri_pos` / `tri_obj` / `tri_closed`
  and both geometry types' per-frame bounds/opacity/caster tables. Collapsed
  bounds reach `build_stbvh` / `build_refit_bvh` at `Tc == 1`, waking their
  (previously starved) static branches — one instance spanning all frames
  instead of per-frame structure over identical boxes — on the eager and the
  deferred build path both. The raster host tables already read every input
  `f % shape[0]` and the projection table spans the *longest* input, so a
  fully-parked batch's tables shrink too — except that the CAMERA tensors
  are indexed dense by the kernels (`cam_origin[f]`, no modulo), so a parked
  camera still pins the projection tables at T. That is the next slice of
  item 3 and is deliberately not taken here.
* **Sync fixes (item 8)**: `_shadow_identity_epsilons`' scene diagonal — a
  whole-batch `tri_pos` reduction ending in `.item()`, previously run once
  per TILE ATTEMPT — is cached on the merged scene per batch;
  the sheet compaction's two split-group diagnostics stay 0-d device
  tensors (and their group tables over-allocate to `nb`, removing the
  `ngroups` sync as well): three device syncs per compaction gone.

**Parity.** `benchmarks/_texture_dedup_ab.py` renders a scene exercising
every changed path — two ImageMobs of one file, a third whose texture
animates (per-map `t > 1` asserted), a static half-scene and a moving cube
(collapsed AND dense merges asserted), shadows on — under all-toggles-off
(the legacy layout byte for byte) and all-toggles-on with pinned batch
windows: **every frame byte-identical**, non-vacuity asserted per path. The
fast suite (277 tests incl. the pixel-compared render) and the sheet
compaction / texture memory / environment / settings-API unit tests pass.

**Measured, CPU box (4-vCPU cloud container; shares, not wall-clock gospel):**

* The probe scene (`scratch_perf/probe_time_expansion.py`, four static mobs
  incl. an ImageMob, 20 PREVIEW frames): merged upload **127.2 MB → 32.3 MB**
  (textures `[4, 1.57M, 5] → [1, 1.57M, 5]`, `tri_pos` and every bounds/flag
  table at `[1, ...]`); with the moving-cube arm only `tri_pos`/bounds stay
  dense at the window length — the item-3 all-or-nothing rule is broken for
  every other table.
* The parity scene renders **11.9 s → 3.9 s** (3.0x) across the toggle flip
  on this CPU (merge/upload-bound at PREVIEW; treat as an upper bound for
  GPU boxes, per the two-pole caveat).
* **Item 9's first number** (`benchmarks/_resolve_mode_ratio.py`, the
  measurement item 4 is staged behind): mode 1 / mode 2 = **0.78** on CPU —
  the event-building walk costs nearly as much as the shading walk, so the
  shadowed double-resolve is worth attacking. T4: see below.

**Shadow any-hit qualification (item 2).** On this machine,
`benchmarks/_shadow_anyhit_check.py PREVIEW`: both corner-case scenes prove
their case reached (tie separation MATTERS; peel limit REACHED, 130 rays past
`MAX_SURFACES_PER_RAY`) and modes 0 / 1 / gather produce **byte-identical
videos on both** — and on `materials_and_lighting` (the pixel suite's only
shadowed scene), rendered under all three modes in one process. T4
qualification and the flip's measured effect: see below.

**T4 (Kaggle, Tesla T4), A/B on identical code with the four toggles flipped
per arm** (`scratch_perf/kaggle/nb_struct1.py` / `nb_struct2.py`; warm RUN 2
of `profile_scene`, read per the usual rules):

| scene | toggles ON | toggles OFF | ratio |
| --- | --- | --- | --- |
| `static_gallery_PREVIEW` (the item-1 population) | **4.61 s** | 12.53 s | **2.7x** |
| `nn_scene_PREVIEW` (everything animates) | 5.56 s | 5.60 s | 1.0x |
| `nn_scene_UHD` | 24.78 s | 24.67 s | 1.0x |

The gallery's mechanism is exactly the item-1 prediction — the estimator
repricing lengthened the batch windows and every per-batch cost amortized:
**25 batches → 8**, arena preflight 7.79 → 2.29 s, merge 3.69 → 0.91 s,
`_dice_logical_pn` 3.26 → 1.11 s, refit-BVH builds 2.02 → 0.62 s,
projection prewarm 4.00 → 1.35 s, `get_batch_of_primitives` 3.47 → 1.41 s.
This is also item 6's re-measurement baseline: the per-batch families it
lists shrank by lengthening alone, before any cross-batch cache.

The nn scenes — where every texture texel and every triangle moves per
frame, so nothing is collapsible — are the no-regression check, and the
FIRST cut failed it: **+22% end to end on both** (nn UHD merge own time
0.50 → 5.82 s), because the new constancy probes each ended in a device
sync, the merge runs on the prefetch worker while the previous chunk
renders, and every sync waits out the whole queued chunk. Fixed
(`679a232`) by not probing texture maps at the merge at all (a static
window arrives already collapsed from `TEXTURE_WINDOW_COLLAPSE`, whose own
sync sits where the queue is shallow and measured free), gating content
dedup to maps of at least 4096 texels, and folding all per-table collapse
probes into ONE stacked sync per geometry block (`_dedup_time_group`).
Post-fix the nn arms read the table above — neutral — with merge own time
back at 0.51 s. **The lesson to carry: on the prefetch worker, a device
sync costs whatever the render thread has queued, not what the probe
computes.**

**Shadow any-hit (item 2), the flip NOT taken.** Qualification passed
everywhere it was run — all three modes byte-identical on both corner
scenes (cases proven reached) and on `materials_and_lighting`, on this CPU
box and on the T4 — so the modes are safe to select per render. But the
measured default flip is a REGRESSION on the translucent-carrying nn UHD
batch (mode 2): **29.5 → 34.2 s**, `raster_shadow_trace` 3.8 → 6.6 s (the
deferred any-hit pre-pass pays a second full traversal on miss-dominated
rays) and `wavefront_shade` 6.3 → 8.2 s (the wider mode-2 variant), while
the shadowed gallery was neutral (4.59 vs 4.66 s). `SHADOW_ANYHIT` stays
default off; the candidate that survives this measurement is engaging the
any-hit only where mode 3 applies (batch provably translucent-free).

**Item 9's number on the T4** (`benchmarks/_resolve_mode_ratio.py`, MD):
mode 1 / mode 2 = **0.685** (10.5 s vs 15.3 s over 7 launches; 0.78 on the
CPU box) — the shadowed double-resolve nearly doubles resolve cost, which
ranks the ~15-floats-per-sheet memoization (and behind it item 4's
transport/shade split) as a real candidate.

## The sheet-resolve memo (2026-08-27): built, byte-identical, and it does not pay

`RENDERER_WORK_QUEUE.md` item 9's memoization, built as the cheap half of
`DESIGN_renderer_structural_candidates.md` item 4. Shipped as
`SHEET_RESOLVE_MEMO`, **default OFF on measurement**.

**What it does.** A shadowed batch launches `sheet_resolve_shade` twice over
the same sheets -- mode 1 walks the transport and builds the shadow events,
mode 2 shades reading the traced visibility -- and mode 1 already fetches
everything mode 2 re-fetches. Mode 1 now stores each processed triangle
sheet's colour(4), alpha, reflectivity, roughness, IOR, transmission and
surface point (twelve floats) and mode 2 reads them back instead of calling
`_tri_color_g` / `_tri_extra_g` / `_tri_ior_transmission_g` /
`_tri_surface_point` again. Sound because the two walks process exactly the
same sheets: every `mode != 1` gate in that kernel wraps a spawn, a shade, the
truncation counter or a pixel commit, and none touches loop-carried transport
state or a break/continue condition.

**Parity.** `benchmarks/_sheet_memo_parity.py`: byte-identical with the toggle
off and on, on this CPU box AND on a Tesla T4 -- 0 differing pixels. A third
arm poisons the memo between the two launches and must differ, which it does
by 1379657 pixels, so the read is proven live rather than assumed. The fixture
carries an unlit mob (processed sheets that build no event -- the rows the
existing event tables never cover), a translucent one (the IOR/transmission
columns), a textured one (the sampler path) and a `Text` (interleaved circuit
sheets, which the memo skips).

**Measured, Tesla T4, warm RUN 2, UNSYNCED profile:**

| scene | `sheet_resolve_shade` off -> on | share of render | end to end |
| --- | --- | --- | --- |
| `nn_scene_UHD` | 0.306 -> 0.304 s | **1.2%** of 22.9 s | 25.69 -> 25.85 s |
| `static_gallery_PREVIEW` | 0.027 -> 0.027 s | **0.6%** of 4.5 s | 4.73 -> 4.51 s |

The stage is under 1.5% of a render and the memo moves it by less than a
millisecond, while costing 48 B per sheet of arena that the runtime memory
model prices into the next chunk's length. Default off. (The gallery's
end-to-end -4.7% is not this change: a 0.027 s kernel cannot produce it.)

**The measurement lesson, which is the durable part.** The number that ranked
this work -- mode1/mode2 = 0.685, "10.5 s against 15.3 s" -- came from
`benchmarks/_resolve_mode_ratio.py`, which brackets every launch with a device
sync. Each launch therefore absorbs the queue it drains, and it reported ~12 s
and ~16 s per mode on a render whose entire resolve kernel is 0.3 s. The
harness warns about this in its own docstring ("read the two modes' TOTALS
against each other, not against an unsynced profile"); the reading that ranked
the memoization went past it and treated the ratio as if it sized the stage.
**A ratio between two launches does not size the work either of them does.**
Size a stage from the unsynced profile before ranking anything on it -- and
note this also demotes item 4, which was staged behind the same number.

**A CPU render is a different story and is not refuted here.** On the CPU box
the same A/B moved the sync-bracketed mode 2 by 10-13% across two independent
runs; that measurement has the same defect, but the resolve genuinely is a
larger share of a CPU render, so `ALGAN_SHEET_RESOLVE_MEMO=1` may be worth it
there. It should not be flipped on for a GPU render without a fresh unsynced
profile of the target scene.

## The T4 round (2026-08-25): the nn performance scenes

> The T4 line of work has its own plan of record now:
> **`DESIGN_T4_optimization.md`**. It supersedes this section, whose
> rankings were read off profiles whose `excl` columns silently included
> the Taichi kernels each stage launched (fixed at `9f3fdb90`).

A second reference workload, measured on a Google Colab box -- **Tesla T4, 2
vCPUs, 12 GB RAM** -- which is a different machine from everything above (GTX
1050, Windows): `benchmarks/performance/nn_scene_PREVIEW.py` (704x396, 10 fps,
50 frames) and `nn_scene_UHD.py` (3840x2160, 60 fps, 30 frames), both a
`NeuralNetMLPV3` (40 PN spheres with physical materials, 80 unlit cylinders and
an idle updater that repositions all of them every frame), a textured
`ImageMob` whose 1774x887 texture animates, and a `Text` label. Baseline
reports are in `benchmarks/performance/reports/t4_baseline/`; read RUN 2.

| item | state |
| --- | --- |
| **Wide attributes on the render device** | **shipped, byte-identical at PREVIEW.** The animated texture was 83% of batch preparation: a 7.87M-channel window gathered, lerped, written and premultiplied per frame on the CPU (~150 ms/frame), and the ImageMob's grid child carried a second, never-read copy. An `AttributeTimeline` at least 65536 channels wide now materializes its frame window and gathers its edit log on the render device (`materialize_device`, `ALGAN_WIDE_ATTR_RENDER_DEVICE=0` restores); texture writes stopped propagating to the grid. Image-only prep 100 -> 3.6 ms/frame |
| **Per-device batch budgets** | **shipped.** The texture's bytes were charged to the 300 MB animation-device budget and capped every batch at 3 frames. `_get_render_device_memory_used_per_timestep` prices what lives on the render device against a budget of its own. 17 x 3-frame batches -> 3 batches at PREVIEW |
| **Texture maps stay on their device** | **shipped.** `TrianglePrimitive` relocated texture maps to the corners' (CPU) device -- a T x 31 MB copy back per batch for the projection upload to copy up again |
| **Constant material parameters broadcast, not gathered** | **shipped, bit-identical.** The per-surface primitive build expanded every constant parameter to the grid and gathered it per vertex: 808 gathers per 21-frame batch, ~1 ms each on this CPU |
| **Encoder** | **shipped (Ox Alpha).** `libx264 -preset slower` on two cores stalled the frame queue and left a 14-18 s drain at UHD. `save_video` now picks `h264_nvenc` when an ffmpeg that can drive it exists (`ALGAN_VIDEO_ENCODER`, `algan/utils/video_encoding.py`); the benchmark scripts pass a fast x264 preset so the profile reads as it would on a full CPU |
| **Glossy prefilter tiling** | **shipped, byte-identical.** The split-sum glossy route clamped the sparse tile loop to one frame, so every frame paid a whole bounce loop (traverse + shade + compaction launches per iteration): 319 shade launches for 50 PREVIEW frames against 40 with the loop per tile. The tile now spans frames and the per-frame reflection buffers are filled one frame-part at a time (`gloss_scatter` took a `row_base`). `wavefront_loop` 7.8 s -> 3.1 s at PREVIEW; inert at UHD where a chunk is one frame |
| **Wide windows released after the primitive build** | **shipped, byte-identical.** The texture's materialized window (an image per frame of the batch, on the render device) was kept until the next batch's rematerialization replaced it, so two batches' windows sat on the device while one rendered. `AnimationTimeline.release_wide_windows` drops it once the batch's primitives are built. Peak VRAM at UHD 9.3 GB -> 6.5 GB (baseline 8.4), at PREVIEW 9.0 -> 6.2 GB (baseline 6.2) |
| **Batch windows reproducible** | **shipped.** The merge headroom and the render-device budget derive from the device's total memory (or `available_memory_override`), not from what is free at the moment of asking. See the trap below |

End to end, warm: **PREVIEW 36.5 s -> 7.7 s**, **UHD 50.0 s -> 30.9 s** (the
UHD baseline carried 14.3 s of encoder drain; its render thread went 26.2 s ->
26.1 s, which is where the remaining work is). Peak VRAM at UHD 8.4 -> 6.5 GB and at
PREVIEW 6.2 -> 6.2 GB, with the windows released as above.

**Three measurement traps from this round, all of which cost hours:**

* **Batch windows move pixels, and windows moved with free VRAM.** The chord
  count of every bezier segment is a maximum over the batch's frames, so two
  renders whose windows differ draw different glyph edges (~5% of a 4K frame,
  up to 80 channel values; `scratch_perf/ox/REPORT_batchwide_audit.md` lists
  every batch-wide decision). The window was sized from `0.9 x` live free
  VRAM, so a 100 MB tenant on the GPU -- Ox's verification renders, a probe --
  turned an [19, 8, 3] split into [19, 11] and read as nondeterminism. Renders
  with the same windows are byte-identical whatever the chunk plan, tile size
  or arena size (`scratch_perf/tiles_chain.log`), and `ALGAN_ARENA_POISON`
  (new, `ManualMemory`) showed no read-before-write. **Never pixel-compare
  while anything else uses the GPU**, and pin `available_memory_override` for
  a byte-reproducible render.
* **A float32 atomic add saturates at 2^24**, so a Taichi checksum of more
  than 16.7M ones reads 16777216 and looks like a stale read. Taichi kernels
  do see torch's pending default-stream writes (`scratch_perf/probe_stream_race.py`).
* **The profiler's own syncs** land every torch op's GPU time in the enclosing
  stage's own column; `wavefront_loop`'s 12 s "own" at UHD is torch work in the
  sparse tile loop, not Python.

**What is left on this box, in order:** the UHD render thread (kernels 15 s:
`wavefront_shade` 7.9, `raster_shadow_trace` 3.7, `wavefront_traverse_events`
3.4; the sheet chain's torch passes ~8 s -- Ox is kernelising the largest,
`scratch_perf/ox/brief_sheet_chain.md`); at PREVIEW the arena preflight (~0.6 s
per batch: PN dice, merge, refit BVH) and the per-batch primitive build.

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

### T5. Sparse-coverage host chain -- ~37.7 s of 95.5 s (~9.9%) -- **partly shipped**

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

**BUILT 2026-08-20, MEASURED, AND PARKED OFF.**
`sheet_compact_taichi.gather_fragment_arrays` behind `RASTER_FUSED_GATHER`,
**default OFF**. It is bit-identical (a gather copies bits, so `fast_math` has
nothing to act on) and it is faster, but by far less than this section
predicted: **13.7 ms -> 9.6 ms** across both gather sites of a real 3840x2160
frame, i.e. ~4 ms of a 1.3 s frame against the ~1.5% estimated.

And it is not free. Fusing forces all six outputs to exist before the kernel
writes the first, where six sequential `index_select`s let the caching
allocator hand each output the block the previous stage just freed: the frame's
peak CUDA allocation rises **50-160 MB**. Four milliseconds does not buy that,
so the flag exists and the default does not use it. Turn it on for a
bandwidth-bound machine with VRAM to spare.

The traffic argument above is sound as far as it goes; what it omits is that
six sequential gathers each get a coalesced write and one coherent read stream,
while the fused one interleaves six scattered streams per thread -- and that
peak allocation, not traffic, is what binds this stage on a small card.

**Measure this with the REAL permutation or the answer inverts.** On a
`torch.randperm` of the same length the fused gather is **0.82x -- slower**,
at every size from 1 M to 7 M fragments. The permutation this code actually
gathers through is a sort order (pixel, then depth), so consecutive output rows
read nearby source rows and the six streams stay in cache; a random index makes
each of the six miss independently. So a `randperm` microbenchmark gets the
SIGN wrong here, not just the magnitude -- worth knowing before anyone
re-measures this, or measures the same shape somewhere else in the pipeline.

**Shipped, and the half that actually paid: the per-sample-lane reductions.** This
section predates the sheet route, so it does not mention `sheets.compact_sheets`
-- which is now the biggest host item in the stage (31% of a 4K frame, 7% of an
LD one) and grows with resolution. Three of its passes were written one SAMPLE
LANE at a time because torch cannot say "reduce these eight bits at once": the
mask union and the DESIGN_sheet_resolve.md ss6.2 fusion detector cost one
`scatter_add_` per lane, and `_popcount_lanes` cost eight shift/and/add passes
to count at most eight bits. `sheet_compact_taichi.sheet_band_reduce` and
`mask_popcount` (`SHEET_MASK_KERNEL`, default on) do each in one pass.

The exact-area sum went into the same kernel, because it walks the identical
stream and in torch could not share it: `scatter_add_` wants matching dtypes,
so the f64 accumulate needed an f64 copy of the whole fragment array first --
29 MB on a 4K frame, for a value read once. Widening in a register off the f32
read deletes that copy and a launch: ~2 ms and **27 MB of peak**. The f64
accumulator itself stays; only the copy is gone. Narrowing it to f32 was
measured and rejected -- 0.2% of a frame against re-opening an ordering
dependence on a value that feeds thresholds, on the 1.6% of a real frame's
sheets that hold three or more fragments (81% hold one, 17% hold two, and
those are order-independent at any width).

Bit-identical, and the fusion detector is the only part where that needs an
argument rather than an inspection: `atomic_or` returns the value *before* this
fragment's contribution, so for a lane claimed by k fragments of a band exactly
k - 1 of them observe it already set, whatever order the hardware serializes
them in -- which is `lane > 1` on the count the loop used to build. The float
area sum stays in torch float64 (ss6.6.4); no float reduction moved.

Together, on the alternating in-process A/B
(`benchmarks/_sheet_kernel_ab.py`, sphere + cube + glass + text):

| resolution | `compact_sheets` | frame |
| --- | --- | --- |
| 864x486 | 40 -> 32 ms (1.25x) | 0.467 -> 0.450 s |
| 1920x1080 | 130 -> 100 ms (1.30x) | 0.933 -> 0.916 s |
| 3840x2160 | 471 -> 354 ms (1.33x) | 1.381 -> 1.297 s (1.065x) |

(that A/B ran with the gather on as well; on its own the gather is the ~4 ms
above.) The stage figure is the reliable one; the frame column is within noise
below 4K. Peak allocation is **+49 MB** at 4K for the mask kernels alone,
after narrowing the union/dup/sliver arrays to int32 -- they hold eight-bit
masks and every consumer casts explicitly, so the width was pure bandwidth. Parity is `benchmarks/_sheet_kernel_check.py` (the kernels against the
exact torch expressions they replace, plus four rendered frames hashed with
both toggles on and both off). `tests/full_renders` and `tests/fast` pass
unchanged on CUDA.

**Shipped 2026-08-21: the conflict-rank scan, the last of the compaction's
multi-pass loops.** `sheet_compact_taichi.sheet_conflict_rank`
(`SHEET_RANK_KERNEL`, default on) replaces eight `torch.cumsum` passes over the
fragment stream -- plus a per-lane `index_select`, `maximum` and two `where`s,
and five live `[n]` arrays -- with one pass: a thread per band walking its
fragments forward with the eight per-lane counters in registers. It needed no
answer to ss10.4's blocked-scan question after all, because the bands are
already contiguous runs of the sorted stream and there is nothing to segment;
see that section for the shape and for the band-length distribution it rests on
(mean 1.11 fragments, max 15 on a real 1080p frame). Bit-identical **by
construction** rather than by argument -- both arms are integer and walk the
stream in the same order -- and pinned by `benchmarks/_sheet_kernel_check.py`,
whose eleven rank cases include ones the render cannot produce.

Measured on **CPU only** (a 4-vCPU cloud container; this is a CPU number and
nothing else, and the reference machine's is unknown): at 1920x1080, one call
per frame over 976,231 fragments, `_conflict_rank` goes **33 ms -> 6 ms** and
`compact_sheets` **480 ms -> 458 ms**. Do not re-measure it on a
`torch.randperm`: on the same captured tensors that reads 4.0x instead of 7.7x,
because the side a random permutation scatters is the kernel's own
`msk[order[j]]` gather -- the mirror image of the gather trap two paragraphs up.
It understates rather than inverting, but it is still wrong.

**What is left of T5, and what to leave alone.** The sorts (`_lexsort`'s three
stable `argsort`s and two `torch.unique` calls, ~87 ms of a 4K compaction)
are cuB radix sorts; Taichi has no sort primitive and hand-writing one to lose
is not a plan. What remains worth measuring is the per-fragment gathers in
`_shade_class` and `_prim_split_after`.

**Shipped 2026-08-25: the next three host passes, measured on the real stream
first** (`scratch_perf/ox/REPORT_sheet_chain.md`). A per-block replay of
`compact_sheets` on a captured nn-scene 3840x2160 frame re-ranked the
candidates: the compaction's remaining scatter blocks are small once the lane
loops above shipped, and the largest non-sort host items were elsewhere --
the SHEET_SAMPLE_DEPTH lane loop (14.5 ms), the emission's opaque-prefix
truncation (15.3 ms incl shared syncs) and its one-mesh reduction (10.7 ms).
Three kernels, default on, each behind its own toggle:
`sheet_lane_first_owner` (`SHEET_SAMPLE_DEPTH_KERNEL`, integer amin per
(sheet, lane) slot -- exact), `opaque_prefix_keep`
(`RASTER_OPAQUE_TRUNC_KERNEL`, integer flags -- exact by construction) and
`one_mesh_pixel_reduce`/`one_mesh_pixel_apply` (`SHEET_ONE_MESH_KERNEL`; the
f64 coverage sums keep the ss6.6.4 accumulate-and-round contract, bitwise vs
torch by measurement, and now serial per pixel so they are order-reproducible
run to run besides). Alternating in-process A/B at 3840x2160:
frame 1.384 -> 1.139 s (1.215x), `compact_sheets` stage 297 -> 210 ms;
same-input per-pass: truncation 2.9 -> 0.4 ms, one-mesh 8.5 -> 2.1 ms,
lane owners 42.3 -> 7.7 ms. Parity: `_sheet_kernel_check.py` extended to all
six toggles (unit arms + edge cases + four rendered frames hashed both ways)
and a lossless HD nn-scene render A/B whose two mp4s are md5-identical. Still
left alone: the sorts, `_shade_class`/`_prim_split_after`'s gathers, the
solid-shell ceiling's sort core, and everything below ~5 ms.

### T6. Raster precompute tables -- 5.4 s (1.4%)

`precompute_triangle_projection` (1.9 s), `precompute_circuit_screen_bounds`
(2.5 s), `precompute_triangle_screen_bounds` (0.9 s). Already cut ~4x an
earlier round by hoisting them from per-chunk to per-batch. What remains is
`precompute_triangle_projection` building ~8 large `[F, N, 3, 3]` temporaries
(`d`, `hit`, `rel`, two crosses, ...) to fill a `[F, N, 13]` table -- a clean
one-kernel fusion. Small on this scene; grows with triangle count.
**Estimated** 8.6 s -> ~3 s (~0.7%).

### T7. Per-dimension dicing, and what the criterion may stop resolving (shipped)

> **The cheapest microtriangle is the one the criterion never asks for.** T1 and
> T4 made the level search and the dice write-out cheap. T7 is about the number
> they were operating on: the dice was uniform in all three barycentric
> directions, and it was measuring against a reference surface it had no right
> to trust to the precision it was chasing.

Two changes, both of which reduce the *count* rather than the cost per triangle,
and a third that reduces the patch count they start from:

1. **The dice is `(level, across level, apex)`.** `2 ** level` rows fanning from
   one corner, each cut into at most `2 ** across` columns
   (`logical_pn.dice_pattern`, `n * (m + 1) - m` microtriangles). Equal levels
   reproduce the uniform barycentric grid *exactly* -- same vertices, same
   triangle order -- so this can only remove triangles from a patch whose two
   directions want different detail. The across level starts at the coarsest
   boundary curve's own level and is measured by the same criterion as the
   isotropic dice, so an anisotropic patch is one that passed, never one
   inferred from its boundary. Crack-freeness is unchanged and now tested at
   mesh scale (`test_a_whole_diced_mesh_stays_watertight_across_every_seam`,
   which fails if the boundary snap is removed).
2. **Both criteria stop at the logical surface's own accuracy.** They score the
   flat dice against the PN patch, which is itself only `geometry_tolerance`
   away from the analytic surface. Measured on one cylinder patch: a 31-triangle
   strip sits 0.000768 world units from the analytic cylinder where the uniform
   level-4 dice's 256 triangles sit 0.000782 from it -- both against a PN patch
   0.000739 off. The 225 extra triangles resolve the PN patch's own error. The
   searches now subtract that accuracy, projected per sample, from what they
   measure (`PN_GEOMETRY_SLACK`). It is a world-space length, so it does nothing
   for distant geometry and does its work in the close-ups where the counts
   hurt.
3. **`grid_aspect_ratio` is gone.** `Cylinder` and `Cone` tied their two grid
   axes at `1/PI`, which bypassed the per-axis geometry search entirely. For the
   cone it spent the resolution on exactly the wrong axis: 40 divisions along
   the *ruled* slant and 13 around, where the free search picks 4 x 19 and meets
   the same tolerance. `min_grid_resolution` now defaults to 2, since an axis a
   surface is straight along needs one cell and the search measures that.

Diced microtriangles for one frame, 1080p, camera 1.6 units back, default
tolerances (`benchmarks`-style probe, CPU session):

| shape | before | grid only | all three |
| --- | --- | --- | --- |
| cylinder r=0.3 h=8 | 1392 | 448 | **163** |
| cylinder r=1 h=1 | 3898 | 976 | **188** |
| cone | 9792 | 378 | **252** |
| sphere | 8682 | 8682 | **3355** |
| torus | 27075 | 27075 | **12562** |

**What the across search costs.** It is not free: on the torus above (1218
patches, one frame, torch path on a loaded CPU session) it took **988 ms of a
4151 ms dice in 3 rounds**, evaluating 25 665 microtriangles across 26 criterion
calls, to remove 21% of that mesh's triangles. Two things should make that
cheaper and are untried: the prediction ought to land in one round rather than
three, and the rounds launch one criterion per distinct `(along, across, apex)`
group, which is launch-bound on small groups. On a CUDA device the fused
criterion kernel carries this, so the shape of the cost is different -- measure
there before tuning it.

**What is not measured.** Everything else is triangle counts, not time: this
landed on a CPU-only session, so what the smaller meshes are worth downstream
(BVH build, traversal, shading, render memory) is unquantified. That wants a
CUDA machine and the reference scene.

**Output moves**, so every full-render baseline needs regenerating -- CPU and
CUDA are separate committed sets and each belongs to the machine that made it.
`tests/fast` is unaffected (no PN geometry), which is exactly the blind spot
this document already warns about.

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

### P10b -- the re-split, after P11 (measured, and it moves the ranking)

> **Every share in the P10 table above was measured before P11 halved
> `compute_grid_vertex_normals`.** `RENDERER_WORK_QUEUE.md` item 12 asks for a
> re-split before anything inside P10 is chosen; this is it, from
> `benchmarks/_surface_build_split.py` -- a repo-local probe (the
> `videos/rl2/` one is not in this repo), self-checking: it measures
> instrumented **copies** of `get_render_primitives_batched` and
> `compute_grid_vertex_normals` and aborts unless each copy's output is
> bit-identical to the shipped function's, on bit patterns.
>
> 220 independent `Sphere` actors at the reference shape, 40 per batched call,
> `[40, 50, 24, 12, 3]`, CPU, medians of per-pass **shares** over 6 timed passes
> (share-of-medians does not sum to 100% under this much jitter), taken with the
> box otherwise idle:
>
> | section | pre-P11 (P10 table) | now |
> | --- | --- | --- |
> | `compute_grid_vertex_normals` | 59.8% | **44.9%** |
> | per-surface `grid_to_triangle_vertices` | 8.1% | **10.5%** |
> | per-surface primitive construction | 8.9% | **13.5%** |
> | whole-stack `grid_to_triangle_vertices` (both gathers) | 13.7% | **9.5%** |
> | per-surface residual + timeline reads | 6.7% | **6.9%** |
> | grid materialize + stack, weld flags | (in "everything else") | 2.6% |
> | unattributed | 2.8% | 9.0% |
> | **shared prefix / per-surface tail** | **74.8% / 24.2%** | **57.0% / 31.0%** |
>
> **Three things follow, and two of them contradict the work queue's guesses.**
>
> 1. **`compute_grid_vertex_normals` is still the top item**, and inside it
>    **"sides + crosses" is 76.8%** -- the rolls, the four differences, the four
>    cross products, the boundary zeroing and the sum. Everything item 12 named
>    as "the rest of `compute_grid_vertex_normals`" is small: pole fans 5.5%,
>    final normalize 4.3%, the two seam merges 2.6% combined. **Optimizing the
>    seam merges or the pole fans cannot matter** -- together they are ~4% of
>    the function, under 2% of the stage.
> 2. **The whole-stack gather is no longer the second target.** Item 12 names it
>    ("two gathers sharing one permutation"); at 9.5% of the stage, fusing the
>    two into one buys at most half of that, ~2%.
> 3. **The per-surface tail grew to 31%**, and its largest single row is the
>    primitive **construction** (13.5%), which no plan named. That is
>    `TrianglePrimitive.__init__`'s non-collection branch: a full clone of the
>    `[T, M, 5]` colours plus two full-size in-place passes, per surface.
>
> **One candidate was tried and does not pay: batching the per-surface colour
> gather.** Stacking every surface's colour grid and gathering once is exactly
> what the grids already do, it is bit-identical (asserted over 480 tensors),
> and it is **1.002x** -- a dead wash. Measured in isolation over 16 alternated
> rounds at 120 surfaces, because the whole-function A/B could not resolve it
> (three runs read 0.83x, 0.95x, 1.08x on a component worth a tenth of the
> function -- a textbook case of the noise this document keeps warning about).
> The reason is structural and will not change: a `Surface` allocates one colour
> row per grid point whatever the colour is, so the `torch.stack` that feeds the
> single gather copies exactly the bytes the saved dispatches were worth. **Do
> not re-try it on CPU.** On a CUDA animation device, where a dispatch costs far
> more than the copy, the balance may differ -- that is unmeasured here.
>
> **What to do next inside P10**, ranked by these numbers:
>
> * **The remaining "sides + crosses" (~35% of the stage).** The only *large*
>   win available is the identity
>   `cross(xm,ym) + cross(ym,xp) + cross(xp,yp) + cross(yp,xm) = cross(xm - xp, ym - yp)`
>   -- four cross products collapsing to one. It is exact in real arithmetic and
>   **not bit-identical in floating point**, and the boundary zeroing breaks the
>   algebra at the grid's edges, so it is a deliberate decision with baselines to
>   regenerate, not a patch. P11b below took the byte-identical part of this
>   block instead.
> * **The per-surface primitive construction (13.5%).** The colour clone and its
>   two in-place passes are per surface and elementwise; they are the same shape
>   of work the grids batched successfully. Unlike the gather, there is no extra
>   copy to pay for -- the clone already exists.
> * **Fusing the two whole-stack gathers (~2%).** Cheap, byte-identical (a
>   gather is a permutation), and small. Do it when touching that code anyway.

### P13 -- the sides-and-crosses block as a CPU Taichi kernel (shipped, default on)

> **Shipped and default on**, gated by `ALGAN_OPT_DISABLE=cpunormals` (or
> `cpukernels` for all three). `benchmarks/_cpu_prep_kernels_ab.py` is the A/B;
> `tests/unit_tests/test_surface_prep_kernels.py` is the correctness net.
>
> P10b left "sides + crosses" as ~35% of the stage and named only one large win
> for the torch form -- collapsing the four cross products with
> `cross(xm,ym) + ... = cross(xm - xp, ym - yp)` -- which is not bit-identical
> and breaks at the boundaries. A kernel gets a bigger reduction on better
> terms. The torch form is arithmetically cheap and structurally expensive:
> four `_wrapped_difference` buffers, four cross buffers and one accumulator,
> **nine full-size tensors written to produce one** at ~57 flops per grid point,
> every intermediate read exactly once. One stencil pass reads the grid and
> writes the normals.
>
> | | measured |
> | --- | --- |
> | the block alone, `[120, 50, 24, 12, 3]` | **8.4-11.3x** |
> | whole `compute_grid_vertex_normals`, `[19, 50, 24, 12, 3]` | **2.3x** |
> | whole `compute_grid_vertex_normals`, `[19, 50, 40, 20, 3]` | **5.0x** |
> | whole `compute_grid_vertex_normals`, `[120, 50, 24, 12, 3]` | **4.6x** |
>
> The function-level numbers are lower than the block's because the seam merges,
> pole fans and normalize are untouched -- P10b measures them at ~4% each, and
> the block at 76.8%, which is the Amdahl ceiling this lands against.
>
> **CPU arch only.** Every Algan kernel takes torch tensors, and Taichi stages
> any argument not already on its arch's device. Launching this with the CPU
> batch tensors while the arch is CUDA would copy the grid *and* the result
> through VRAM on the prep worker thread that is deliberately kept off the GPU
> -- the trap that made the timeline's own kernels a liability. So this is a
> CPU-render optimization today; the CUDA case needs the arch-coexistence work
> (AOT + the C API), which is a separate subsystem.
>
> **Not bit-identical, and not for the reason expected.** 1-2 ulp on ~4% of
> elements. Rebuilding with `fast_math=False` changes nothing, so Taichi's
> codegen is not responsible. On the cross product's third component
> -- `a0 * b1 - a1 * b0`, the one that catastrophically cancels for a sphere's
> tangential sides -- `torch.cross` matches neither that expression in float32
> nor its products taken exactly in double and rounded once; the other two
> components match both. Taichi 1.7.4 exposes no FMA intrinsic, so no
> formulation of the kernel reproduces it.
>
> **Rendered output moves, slightly.** Two-arm full renders (kernels on vs off,
> CPU): `complex_hierarchy_become`, `manim_compat_and_plots`,
> `shapes_and_timeline` and `text_and_media` byte-identical;
> `solids_and_camera` 13 pixels of 66.6M at 1 channel value (inside tolerance);
> `materials_and_lighting` 0.014% of pixels differ, 0.006% by more than 2, peak
> 14. That is sparse speckle through the epsilon/tie machinery the batchwide
> audit documents (depth-tie bins, shadow seam de-dup, f16 box rounding), not a
> shading change. Baselines were **not** regenerated here -- the committed CPU
> set is already stale on this box (all 6 scenes and `tests/fast` fail
> identically on the base branch), so that is a separate job on a machine that
> owns them.
>
> **Watertightness holds structurally**, which is what let this ship without
> bit-identity. The kernel produces the same `unnormalized_normals` buffer the
> seam-merge and pole-fan code consumes, and that code assigns one shared value
> to both sides of a closed seam and one to a whole pole row -- so grid points
> that must agree still read the same element afterwards. This matters beyond
> shading: logical PN patches build curvature from corner normals, so a seam
> whose sides disagreed would crack the geometry. Asserted bitwise on both
> closed axes and both poles, sphere and cylinder.
>
> **Two sibling kernels were implemented and do not pay.** Both byte-identical,
> both shipped **off** (`ALGAN_OPT_ENABLE=cpugather,cpucolors`):
>
> | kernel | row | measured |
> | --- | --- | --- |
> | `gather_grid_to_triangles` | `grid_to_triangle_vertices`, ~20% of the stage | **0.84-1.20x** |
> | `apply_glow_and_opacity` | `TrianglePrimitive` colour bake, 13.5% | **0.89-0.92x** |
>
> Both first measured **worse** than that -- 0.69-1.03x and 0.79-0.81x -- and
> chasing why produced the more useful finding, recorded in
> `benchmarks/_taichi_loop_shapes_taichi.py`. "Memory-bound" explains why a
> kernel is not *faster*; it does not explain why one is *slower*. Copying the
> same 35 MB four ways:
>
> | form | GB/s |
> | --- | --- |
> | `torch.Tensor.copy_` | 30-59 |
> | ti flat 1-D loop | 14-22 |
> | ti nested plain loops, 3-D indexing | ~7.3 |
> | ti `ndrange(B, L)` + static channel | ~4.3 |
> | ti `ndrange(B, L, C)` | **1.7-1.9** |
>
> **`ti.ndrange` over several dimensions is expensive**: Taichi flattens it into
> one parallel loop and recovers each index per iteration, and that arithmetic
> dominates a copy whose useful work is one load and one store. Multi-dimensional
> ndarray addressing costs again on top, ~2-3x. Rewriting the gather to a flat
> loop with flat offsets took it from 0.68x to parity; the same rewrite took the
> colour bake from 0.79x to 0.90x. **Neither kernel was slow because gathers or
> copies are slow -- they were slow because of how the loops were written.**
>
> What is left is structural: even Taichi's best form streams below torch's
> vectorized `copy_`, and a bake that is one add and one multiply over a
> full-width copy cannot make that back. The colour bake's "three passes to one"
> premise also overcounted -- the two in-place passes touch one channel each, so
> torch moves ~`14N` floats against the kernel's `10N`, a 1.4x traffic saving
> rather than 3x. Launch overhead is ~80 us per call, which only matters for
> small work. `advanced_optimization` is **not** the explanation, which is worth
> recording because it is the obvious suspect: `ALGAN_ADV_OPT=1` changes nothing
> outside noise.
>
> **The flattening does not help the normals kernel** (0.87-1.12x, so it keeps
> its `ndrange(B, W, H)`), and that is the rule rather than an exception: index
> overhead matters in proportion to how little work each element does. ~57 flops
> per grid point swamps it; one load and one store does not.
>
> **The lesson generalizes: in this pipeline a kernel wins where there are
> intermediates to fuse, and loses where there are not -- but before concluding
> a kernel cannot win, check the loop shape.**
>
> **Concurrency note.** Batch prep runs on a `ThreadPoolExecutor` worker while
> the main thread renders, so on a CPU arch both threads now launch Taichi
> kernels into the one `Program`. Twelve minutes of full renders across six
> scenes completed clean, twice, which is the evidence; no lock was needed.

### P11b -- the sides written without a materialized roll (shipped)

> **Shipped**, bit-identical, inside the same `gridnormals` arm P11 introduced,
> so `benchmarks/_grid_normals_ab.py` covers it across all 13 grid topologies
> without a new harness.
>
> P11 left the four sides as `grid.roll(shift, axis) - grid`. `roll` allocates
> and fills a whole copy of the grid that the subtraction then reads once and
> throws away: **two full-size writes where one will do**, four times over.
> `_wrapped_difference` writes the difference straight into one output buffer,
> in the two pieces the wrap-around splits it into. The four crossed pairs are
> then accumulated in place instead of through three temporaries.
>
> Bit-identical, and not by an argument about associativity: every element is
> the same subtraction of the same two elements written to a different place,
> and `t = a + b; t += c; t += d` is the same three additions in the same order
> as `a + b + c + d`. Asserted on bit patterns across open / x-closed / pole
> grids, single-column and two-row degenerates, collapsed columns, float64, NaN,
> inf and signed zeros.
>
> | | measured |
> | --- | --- |
> | the four sides alone, `[120, 50, 24, 12, 3]` | **1.44x** |
> | sides + crosses + accumulate, `[120, 50, 24, 12, 3]` | **1.33x** |
> | the same at `[40, 50, 24, 12, 3]` | 1.065x |
> | `_grid_normals_ab.py` REAL rows, paired arm vs the legacy stack | 2.20x -> **2.99x** and 2.04x -> **2.12x** |
>
> **Read the large rows.** The small-grid cases are dispatch-bound and the
> paired form runs *more* torch calls there, exactly the caveat P11 records; two
> of them read below 1.0x and say nothing about the win.
>
> Predicted effect on the stage from the re-split above: sides+crosses is 76.8%
> of a function that is 44.9% of `get_render_primitives_batched`, so
> `0.232 + 0.768/1.33 = 0.81` on the function and **~1.09x on the stage**. Not
> confirmed end to end -- that needs the reference machine, and a CPU-only box
> cannot speak for it.

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

### P9 -- the batched bezier build reaches 18.4% of the circuits (shipped)

> **Shipped**, byte-identical, gated by `ALGAN_BEZIER_GROUP_RUNS` (default on;
> `=0` restores the all-or-nothing revert so both arms run in one process).
> The measurement that motivated it is kept below, unchanged, because it is the
> evidence for why the revert had to go.
>
> **What was built.** A clashing group is no longer given up wholesale. The
> insight is that the layout constraint is *positional*, not group-wide: within
> one batch identifier, each deferred circuit sits after some number of raw
> primitives of that identifier and before the rest, so splitting the group into
> **maximal runs of consecutive batchable actors** and merging each run on its
> own puts every merged collection on exactly the span its circuits' raw
> primitives would have occupied. `get_batch_of_primitives` stamps each deferred
> entry with that position -- the count of raw primitives of its identifier seen
> so far in one ordered walk -- and `_build_deferred_beziers` groups by
> `(group key, run)`. A `grouped_primitives` bucket became an ordered list that
> may hold merged collections *and* raw primitives; the emission walk flushes
> each maximal raw run through the per-class emission that used to run over the
> whole bucket (factored out unchanged as `_emit_primitive_collections`). With
> no raw primitives of an identifier present, every entry gets run 0 and the
> grouping is what it always was. `_PREBUILT_COLLECTION` is gone: a bucket no
> longer needs a single class marker, which is what forced the revert.
>
> **Measured** (`benchmarks/_bezier_batchability.py`, which replaces the
> `videos/rl2/` probe named below and does not exist in this repo). Two scenes,
> six 8-frame windows each, outcomes measured by watching which builder each
> circuit actually reached rather than by re-deriving the predicate:
>
> | scene | arm | batched | reverted by the clash |
> | --- | --- | --- | --- |
> | `benchmarks/bezier_rendering.py` | either | 99.9% (15 006 / 15 018) | 0 |
> | a packed `Text` sharing an identifier with 40 circles/squares | `=0` | 0 | **97.6% (240)** |
> | the same | `=1` | **97.6% (240)** | 0 |
>
> On the clashing scene `get_batch_of_primitives` goes from ~38 ms to ~17 ms per
> window, **0.43-0.48x, consistently across all six** (medians, arms alternated
> per round). **The benchmark scene in this repo has no clash at all**, which is
> worth knowing before anyone A/Bs on it: the payoff is entirely scene-shaped,
> and an A/B there measures two arms doing identical work.
>
> **Byte-identity** is settled by a render, not by an argument
> (`benchmarks/_bezier_run_split_ab.py`): the clashing scene rendered twice in
> one process, lossless, arms flipped between renders -- **max channel
> difference 0, 0 differing pixels over 10/10 frames**, with both arms asserted
> to have seen the same batch windows and the arms asserted non-vacuous (41
> per-actor circuit builds under `=0` against 1 under `=1`).
> `tests/unit_tests/test_bezier_group_runs.py` is the standing guard.
>
> **What widening the path turned up.** `build_render_primitives_batched`
> documents itself as a byte-identical replacement for the per-actor
> constructor, and it was not: it set `num_pixels_per_sample = 1` where
> `BezierCircuitPrimitive`'s constructor defaults to `0.5`. That value is the
> maximum screen-space curve-to-chord error in pixels, so every batched circuit
> was flattened to twice the per-actor path's tolerance. The **default
> analytic-AA route hides it** -- it clamps the tolerance to
> `ANALYTIC_AA_CHORD_TOLERANCE = 0.25`, and 0.5 and 1 both land on 0.25 -- and
> the classic supersampled route does not. It shipped in the same commit as the
> function (`8e29beb`), and `benchmarks/_bez_batch_parity.py` was written
> expecting the two to be equal, which is how it surfaced. Both now read one
> named constant, `DEFAULT_CHORD_TOLERANCE_PIXELS`. Harmless while the batched
> build reached a fifth of a scene's circuits; not harmless once a clash stops
> sending the rest down the other path.
>
> **`benchmarks/_bez_batch_parity.py` did not run at `HEAD`** and was repaired
> as part of this: `set_render_settings`, `AnimationManager.instance()` /
> `TimelineManager.instance()` (managers are per-Scene now) and
> `scene.actors[-1]` (actors is a flat list) had all been gone for some time,
> and its `ATTRS` named attributes the primitive no longer has. It is the
> harness that guarantees the builder this item widens, so it was not optional.
> It now passes on a 2513-circuit group.
>
> ---
>
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

### P13 -- batching an updater across its mobs (shipped for the idle updater)

> **Shipped**, bit-identical, gated by `ALGAN_BATCHED_IDLE_UPDATER` (default on;
> `=0` restores the per-mob loops in one process). The nn benchmark scene
> (`NeuralNetMLPV3([5,5,5,5])`, the `benchmarks/performance` scene) is
> preparation-bound at `PREVIEW`, and ~32% of one batch's preparation under
> cProfile was `_update_neural_net_idle`: 15 neurons and 80 synapses walked in
> Python, each `move_to`/`set_end_point`/`move_between_points`/`set_start_point`
> several tiny timeline reads and writes on `[T, 1, 3]` tensors -- dispatch and
> Python, not arithmetic.
>
> **What was built.** Option 2 of the two the round considered: the mobs stay
> unpacked, and the updater computes what all four loops would write and lands
> it in **three** timeline writes (one recursive-subtree location write for the
> neurons, one fused location write over every synapse row and tube-grid row,
> one basis write). Option 1 (packing the synapses at construction) was
> rejected: packed members share one lifespan, and while nothing in
> `neural_net.py` spawns or despawns an individual synapse today, packing
> heterogeneous `Cylinder`-in-`Mob`-in-neuron trees through `from_batches` is a
> structural change far past this round's size, and the timeline-level fix
> generalises to any updater that touches many sibling mobs.
>
> **Bit-identity is by replication, not by tolerance.** The batched path
> re-evaluates the *same expressions* the loops do, batched: the setter dance
> (`old + (target - old)`, never `target`) on every shifted row including the
> intermediate loop-1 shift the loops perform before overwriting grids; the two
> offset formulas (raw basis row vs normalized-direction-times-scale); the
> interpolation pass-through at `interpolation = 1.0`; `coord_function` verbatim
> against the new basis. Every read happens before any write, so an unsupported
> structure (capped cylinders, ragged grids, differing `v_range`s) raises
> before state moves and falls back to the loops. Output-layer neurons are not
> idle neurons and never move; their synapses take a padded zero change.
> Dependency tracing is preserved per mob (`trace_updater_mob_access`), so
> future windows keep materializing the same working set.
>
> **Measured** (17-frame window, warm medians of 6, arms in separate processes,
> loaded CPU box shared with another tenant -- read the deltas, not the
> absolutes): prep **2380 -> 1860 ms (0.78x)**; earlier pairs 2792 -> 2128 and
> 2295 -> 1870. Per batch: `AttributeTimeline.get` **2841 -> 1436 calls**,
> `.modify` **258 -> 6**, tensor `.clone()` **2814 -> 1312**; the four wrapped
> loops disappear from the profile entirely (the updater's remaining cost is
> its irreducible waypoint head).
>
> **Verification.** `scratch_perf/r2/parity_idle_updater.py` materializes one
> window per arm on freshly built nets and compares **every attribute buffer
> plus the non-timeline `direction`s bitwise** -- identical across two frame
> windows (0-16, 20-32) and three layer sizes ([5,5,5,5], [3,4,2], [2,3,2]),
> before and after the read-path changes below.
> The nn scene rendered lossless (`libx264rgb qp 0`,
> `available_memory_override` pinned) differs by **0 pixels over the suites'
> tolerance** pre/post change (worst single channel value 1, which an
> identical-code rerun also shows -- that is the documented cross-run noise,
> not the change). `tests/unit_tests/test_neural_net_idle.py::test_batched_
> idle_updater_writes_what_the_loops_write` is the standing guard (marked
> fast).
>
> **Alongside it, the read-path clones the loops were paying went away where
> they were provably dead**: `AttributeTimeline.get(copy=False)` at the
> Cylinder stretch sites (`set_start_point`, `set_end_point`,
> `move_between_points`, `coord_function`, `_cap_ring_offsets`),
> `Surface.set_location_by_function`'s location add, the batched surface build's
> grid stack, `compute_grid_color`'s double clone, and
> `_build_deferred_surfaces`' group key, which cloned every surface's whole
> grid once per batch to read its `.shape`. All feed out-of-place arithmetic
> only; the solids_and_camera full render decodes **byte-identically** against
> a clean checkout of the branch (239/239 frames, worst channel diff 0), which
> is the proof those flips cannot move values. Caching `_get_attr_ranges` /
> `_compact_span` was looked at and **not done**: post-change they are ~1400
> calls of mostly dict lookups and two numpy scalar reads, single-digit ms of a
> ~1.9 s window on this box -- below the noise floor of the machine that would
> have to measure the win.

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
2. ~~**P9 -- widen the batched bezier build**~~ -- **shipped.** The
   all-or-nothing group revert is gone: a clashing group is split into maximal
   runs of consecutive batchable actors, because the layout constraint is
   positional rather than group-wide. Byte-identical (0 differing pixels on a
   lossless two-arm render); 0.43-0.48x on `get_batch_of_primitives` on a
   clashing scene. It removes the per-actor build's accessor round trips for
   those circuits, so it cuts item 1 as well. See P9.
3. **The rest of the batched surface build (P10)** -- **56.62 s (15.8%)** after
   P11 took 1.51x off it. **Re-split (P10b), and the re-split re-ranks it** --
   the proportions this list used to quote were measured before P11 and were
   wrong about which parts matter:

   * **`compute_grid_vertex_normals`, still 44.9% of the stage, and 76.8% of
     that is "sides + crosses".** P11b took the byte-identical part of that
     block (the materialized `roll`s and the accumulation temporaries, 1.33x).
     What is left needs the four-crosses-to-one identity, which is exact in
     real arithmetic and **not bit-identical**, so it is a decision with
     baselines to regenerate rather than a patch. **The seam merges and pole
     fans are ~4% of the function and are not worth touching** -- this list
     used to name them.
   * **the per-surface tail, now 31.0%**, whose largest row is the **primitive
     construction (13.5%)**: a full clone of the `[T, M, 5]` colours plus two
     in-place passes, per surface. No plan named it before the re-split.
     Batching the colour *gather* was tried and is a wash (1.002x, bit-identical
     -- see P10b before repeating it); the construction has no such extra copy
     to pay for.
   * **`grid_to_triangle_vertices` on the whole stack, now 9.5%, not 13.7%** --
     the same "two gathers sharing one permutation" shape as T5, but fusing them
     buys at most ~2% of the stage. Do it when in that code anyway.
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
