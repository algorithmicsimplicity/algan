# Optimization targets

Ranked candidates for further speeding up Algan, measured against the project's
reference workload: `s05_learning_to_program_setup.py` at `LD` (864x486, 15 fps,
3883 frames), profiled with `algan.utils.profiling_utils`.

> **Read the percentages, not the seconds.** Both profile runs quoted here
> landed on a thermally throttled machine (`nvidia-smi` reported `SwThermal`
> throughout each). Shares of the run are stable; absolute seconds are not
> comparable between the two columns of any table below. Where a claim is an
> estimate rather than a measurement it says so.

---

## Status -- start here

| item | state |
| --- | --- |
| **T1** PN subdivision-level criterion | **shipped and confirmed end to end** -- 67.9 s -> 1.40 s |
| **T2** Bezier chord-count search | **shipped and confirmed end to end** -- 18.4 s -> 0.89 s |
| **T4** dice write-out (cheap half only) | **shipped** -- byte-exact, but only ~1.07x |
| **P2** contiguous replay time-selector | **shipped** -- 1.62x on replay-time `get`/`modify` |
| **P3** lazily-zeroed remat buffer | **shipped** -- 1.20x on `rematerialize_state_at_times` |
| **P4** whole-scene per-batch scans | **shipped** -- honestly ~3.7 s of a ~500 s render (66x on the actor filter), not the 11.63 s first reported; see the correction under P4 |
| **P5** replay-checkpoint in-place growth | **shipped, near-worthless** -- the clone it removes happens once per render, not once per batch; see the correction under P5 |
| **P1** compact rematerialization buffer | **shipped** -- 1.37x on materialization + replay reads, and 604 MB/batch of buffers -> 204 MB |
| **T5** sparse-coverage host chain | **next** -- the render thread's largest non-kernel item |
| **T3**, **T6** | untouched; both shrank in share |

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

Since then P1-P5 have all shipped, all in prep, so **the shares in this document
are stale by that much and want a re-profile** before the next ranking decision.
`_query_row_states` (42.7% of `rematerialize_state_at_times`) is the largest
prep item nothing has touched.

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

* `pytest -q --fast` (~112-147 s) is the loop, **but it cannot see a PN level
  flip**: `tests/fast/scene.py` deliberately contains no `Surface`/PN geometry.
* `pytest -q tests/full_renders` (~10 min) is the one that can. Run it for
  anything touching tessellation, projection or the criteria.
* Prep changes are seen by neither on their own -- but both suites do exercise
  them, since every render prepares batches. Run the full renders for a prep
  change too, and A/B the stage in-process (below) for the number.
* A/B parity scripts for the shipped work, the kernel ones CUDA-only:
  `benchmarks/_pn_criterion_kernel_ab.py` (levels + shared-edge agreement +
  timing, static *and* moving meshes), `benchmarks/_bez_chord_kernel_ab.py`
  (chord counts + timing), `benchmarks/_pn_dice_scatter_ab.py` (byte-equality of
  every diced array), `benchmarks/_logical_pn_crack_check.py` (seam integrity),
  `benchmarks/_p1_zerofill_ab.py` (the lazily-zeroed buffer, CPU),
  `benchmarks/_prep_timeslice_ab.py` + `videos/rl2/animations/_prep_timeslice_ab_s05.py`
  (the prep optimizations, CPU, `ALGAN_OPT_DISABLE`-gated),
  `benchmarks/_event_index_parity.py` (the interval index, CPU),
  `benchmarks/_resolve_rollback_check.py` (the replay checkpoint survives a
  render unchanged -- read its note on how it avoids being vacuous).
* **Real-render** probes, which are the only ones that can see recording,
  resolve and cache behaviour honestly (see "The measurement trap" above):
  `benchmarks/_replay_records_check.py` (replay re-records; quantifies it),
  `benchmarks/_render_time_growth_check.py` (a real multi-batch render records
  nothing and resolves once), `videos/rl2/animations/_real_render_scans_s05.py`
  (the per-batch scans timed inside a genuine `save_video`).
* Scaling probes, all in `videos/rl2/animations/`, all CPU-only prep with no
  render. **They drive `get_batch_of_primitives` directly, so their recording
  and cache-invalidation behaviour is not a render's** -- read them for the
  stage costs that are unconditional per batch, not for anything else:
  `_remat_scaling_s05.py` (materialization vs total rows N vs active rows R),
  `_quadratic_scans_s05.py` (the whole-scene per-batch scans),
  `_windows_rebuild_probe_s05.py` (splits a lookup into resolve / cache rebuild
  / query -- the one that found P4's real bottleneck),
  `_event_duration_dist_s05.py` (why an interval index does or does not prune).
* Re-baseline, only after looking at the frames:
  `ALGAN_UPDATE_FULL_RENDER_BASELINES=1 .venv/Scripts/python.exe -m pytest tests/full_renders -q`

## The shape of the problem

The render is a two-thread pipeline, and **prep is now the larger pole by a
clear margin**:

| pole | stage | then | now |
| --- | --- | --- | --- |
| batch-prep worker | `Scene.get_batch_of_primitives` | 519 s (64.8%) | 299.4 s (**78.3%**) |
| render thread | `ray traced render total` | ~543 s (67.7%) | 196.1 s (51.3%) |

They still overlap, so neither number is the wall clock. But the old advice --
"cutting only one pole buys almost nothing, work the two lists together" -- was
written when the poles were level. They are not any more: T1 and T2 took ~84 s
off the render thread and nothing off prep, and prep is what the total now
tracks. **Prefer a prep item over a render item of the same size.**

A second consequence, unchanged: this scene is *not* GPU-limited. Average GPU
utilisation was 44% on the re-profile. The Taichi candidates below are worth
doing because they are on the render thread's critical path, not because the
device is saturated.

### The re-profile, warm run 2 (`save_video` = 382.3 s)

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

### T4. The dice write-out -- ~17 s (2.1%) -- **cheap half done, disappointing**

> `_scatter_diced_rows` now folds (frame, column) into one flattened row index
> and writes each output with a single `index_copy_` over `[T * M, ...]`
> (`padding` with `index_fill_`). **Byte-identical** -- every diced array
> compares equal on `benchmarks/_pn_dice_scatter_ab.py`.
>
> But it is only **1.02-1.19x on the whole dice** (257 ms -> 239 ms across four
> meshes), not the ~2x on the write-out the estimate below assumed. `index_put_`
> was apparently not costing much more than `index_copy_` here. Kept -- it is
> free and byte-exact -- but the remaining T4 estimate should be treated as
> unproven, and the `allocate()` zero-fills (not the scatter) are the likelier
> half of that 17 s.

#### Why it was a target (original measurement)

`_dice_logical_pn`'s own time (12.4 s) plus `interpolate_patch_attribute`
(3.1 s), `evaluate_logical_pn_normals` (1.0 s) and `snap_boundary_values`
(0.4 s). Two costs were identified: the `allocate()` zero-fills
(`[T, max_triangles, 3, D]` for corners, normals, colours and every
surface/shader parameter) and the advanced-index scatters.

**The scatter half is done and was worth little (above), so the remaining
`~17 s -> ~8 s` estimate now rests entirely on the untested half: the
`allocate()` zero-fills.** That is where a fresh attempt should start --
`max_triangles` is the batch's *widest frame*, so every frame's buffer is padded
to it and the surplus is zeroed for nothing. Sizing per frame, or filling only
the padding rows, is the idea to test. Writing the dice as a kernel (the more
expensive option originally sketched here) should wait until the zero-fill has
been measured on its own.

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
| `BezierCircuitCubic.get_render_primitives` | per-actor build; `build_render_primitives_batched` already covers the batchable case |
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

**One thing measured and ruled out along the way:** reusing the buffer across
batches instead of reallocating it. Batch prefetch forbids it -- prep for batch
b+1 runs on a worker while b renders, and it is the *reallocation* that keeps
b's handed-out views valid. A shared buffer would have b+1 overwrite what b is
still reading. (The compact buffer is still reallocated per batch; it is only
the row *map* that is grown in place, and nothing hands out views into that.)

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

Prep is 78.3% and the render thread 51.3%, so prep items come first at equal
size.

0. **Re-profile.** P1-P5 all landed since the 382.3 s run this document is
   ranked against, and prep is where every one of them hit. The shares below
   are stale by that much -- the same trap the previous revision fell into.
1. **`_query_row_states`** -- 42.7% of `rematerialize_state_at_times`, ~10.5% of
   `save_video`, and never previously named as a target. Now the largest single
   item in prep, and untouched: P1 changed where its result is *stored*, not
   how it is computed. CPU-only by constraint. Detailed under P1.
2. **`set_state_to_times` own time** (19.7%) and the ~63% of
   `AttributeTimeline.get` calls P2 did *not* touch (those made outside replay,
   during the geometry build). The lever there is fewer calls, not faster ones.
3. **T5** -- the sparse-discovery host chain, ~37.7 s (~9.9%): the six
   `index_select` gathers that share one permutation are one kernel. The
   largest non-kernel item on the render thread, and the largest item left
   anywhere now that the prep structural work is done.
4. **T3** (5.1%), then **T4's remaining half** (the `allocate()` zero-fills --
   small, its estimate unproven, and P3 suggests the same lazily-zeroed trick
   may apply; fold it in whenever the dice is open anyway), then **T6** (1.4%).

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

Re-profile after each: on a two-pole pipeline the ranking moves as soon as one
pole shrinks.

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
