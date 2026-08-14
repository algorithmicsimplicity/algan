# Optimization targets

Ranked candidates for further speeding up Algan, measured against the project's
reference workload: `s05_learning_to_program_setup.py` at `LD` (864x486, 15 fps,
3883 frames). Every number below comes from one profile run of that scene with
`algan.utils.profiling_utils` on the current tree.

> **Read the percentages, not the seconds.** That run landed on a thermally
> throttled, RAM-contended machine: `save_video` took 802 s where a quiet run
> of the same code takes ~425 s. Shares of the run are stable; absolute
> seconds are inflated roughly 1.9x throughout. Where a claim is an estimate
> rather than a measurement it says so.

---

## Status -- start here

| item | state |
| --- | --- |
| **T1** PN subdivision-level criterion | **shipped** -- fused kernel, ~6x on the criterion |
| **T2** Bezier chord-count search | **shipped** -- 5.4x, same kernel family |
| **T4** dice write-out (cheap half only) | **shipped** -- byte-exact, but only ~1.07x |
| **P1** compact rematerialization buffer | **next**, after a re-profile |
| **T3**, **T5**, **T6** | untouched; re-rank them off the next profile |

The T1/T2 numbers are **microbenchmarks, not a re-profile of the reference
scene.** Nothing has confirmed what they are worth end to end, and the ranking
in this document is stale by exactly that much: ~86 s of measured torch work has
come off the render thread since it was written. **Re-profile before starting
P1** -- on a two-pole pipeline (see below) the ordering moves as soon as one
pole shrinks.

### Reproducing the profile

The reference scene lives **outside this repo**, in the video project:

```text
D:\AlgorithmicSimplicity\videos\rl2\animations\scenes\s05_learning_to_program_setup.py
```

That scene appends its own parent directory to `sys.path` and imports `utils` /
`backgrounds_taichi` from `videos/rl2`, so it only runs from inside that
project, not from this repo. It also authors the scene at module scope rather
than exposing a scene function, so profiling it means driving it from a wrapper
in `videos/rl2` -- the same shape every in-repo benchmark uses:

```python
from algan.utils.profiling_utils import profile_scene

profile_scene(scene_func, LD, tag="s05", runs=2, kernel_profiler=False)
```

`profile_scene` writes `algan_profile_report[__tag].txt` beside the caller. Run 1
is cold (Taichi JIT, cold clocks); **read run 2**. The knobs are documented in
that module's docstring (`ALGAN_PROFILE_RUNS`, `ALGAN_PROFILE_CPROFILE`,
`ALGAN_PROFILE_TELEMETRY`, `ALGAN_PROFILE_NVPROF`); the many
`algan_profile_report__*.txt` files at this repo's root are earlier runs from
other investigations, kept for comparison.

If the external project is unavailable, `benchmarks/bezier_rendering.py` and
`benchmarks/neural_net_benchmark.py` are self-contained in-repo drivers that run
as-is. They are smaller and bezier-heavy, so they rank T2/T3/T5 usefully but say
little about the PN targets, and their absolute shares do not match the table
below.

### Verifying a change to any of this

* `pytest -q --fast` (~112-147 s) is the loop, **but it cannot see a PN level
  flip**: `tests/fast/scene.py` deliberately contains no `Surface`/PN geometry.
* `pytest -q tests/full_renders` (~10 min) is the one that can. Run it for
  anything touching tessellation, projection or the criteria.
* A/B parity scripts for the shipped work, all CUDA-only:
  `benchmarks/_pn_criterion_kernel_ab.py` (levels + shared-edge agreement +
  timing, static *and* moving meshes), `benchmarks/_bez_chord_kernel_ab.py`
  (chord counts + timing), `benchmarks/_pn_dice_scatter_ab.py` (byte-equality of
  every diced array), `benchmarks/_logical_pn_crack_check.py` (seam integrity).
* Re-baseline, only after looking at the frames:
  `ALGAN_UPDATE_FULL_RENDER_BASELINES=1 .venv/Scripts/python.exe -m pytest tests/full_renders -q`

## The shape of the problem

The render is a two-thread pipeline and **both poles are now the same size**:

| pole | stage | of `save_video` |
| --- | --- | --- |
| batch-prep worker | `Scene.get_batch_of_primitives` | 519 s (64.8%) |
| render thread | tracer + arena preflight + reclaim | ~543 s (67.7%) |

`save_video` is 802 s, so the two overlap well. The consequence for planning:
**cutting only one pole buys almost nothing.** A change that halves the render
thread moves the total by a few percent until prep is cut too, and vice versa.
Work the two lists together.

A second consequence: this scene is *not* GPU-limited. Average GPU utilisation
was 32% before the current round of work. The Taichi candidates below are
worth doing because they are on the render thread's critical path, not because
the device is saturated.

---

## Part 1 -- Dedicated Taichi kernels

Every entry here is torch code that already runs on the **render device**
(`PROJECT_ON_GPU` is on by default, so primitive source geometry, the camera
snapshot and the scratch arena are all `cuda:0` by the time these run). They
share one failure mode: tens of elementwise passes over large arrays to
produce a handful of scalars, so they are VRAM-bandwidth bound rather than
launch bound. Fusing each into a single kernel that keeps intermediates in
registers is the win.

### T1. PN subdivision-level criterion -- 67.9 s (8.5%) -- **DONE**

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

### T2. Bezier chord-count search -- 18.4 s (2.3%) -- **DONE**

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

### T3. Bezier circuit geometry build -- 47.1 s (5.9%)

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

### T5. Sparse-coverage host chain -- ~51 s of 127.6 s (6.4%)

`raster: sparse discovery` is 127.6 s (15.9%), but 76.5 s of that is already
Taichi (`raster_tri_count/write`, `raster_bez_count/write`). The remaining
~51 s is the host-side ordering and compaction: the count prefix sum,
`_exact_fragment_order` (two `argsort`s), six parallel `index_select` gathers
of the fragment arrays, `unique_consecutive`, the `scatter_reduce_` opaque
truncation and the keep-mask compaction.

The two `argsort`s are library code and already good. The **six gathers are one
kernel** -- they read the same permutation and write six arrays, so they move
six passes of traffic where one would do. Do that; leave the sorts alone.
**Estimated** ~10-15 s (~1.5%), moderate confidence.

### T6. Raster precompute tables -- 8.6 s (1.1%)

`precompute_triangle_projection` (2.5 s), `precompute_circuit_screen_bounds`
(4.7 s), `precompute_triangle_screen_bounds` (1.4 s). Already cut ~4x this
round by hoisting them from per-chunk to per-batch. What remains is
`precompute_triangle_projection` building ~8 large `[F, N, 3, 3]` temporaries
(`d`, `hit`, `rel`, two crosses, ...) to fill a `[F, N, 13]` table -- a clean
one-kernel fusion. Small on this scene; grows with triangle count.
**Estimated** 8.6 s -> ~3 s (~0.7%).

### Not worth a kernel

* **Bloom** (`bloom fft conv`, 54.2 s / 6.8%) is cuFFT. A direct separable
  Taichi convolution already exists (`bloom_conv1d_f32`) and is *far* slower
  for the wide blur -- measured 660 ms vs 23 ms for the 293-tap horizontal
  pass. The transform-length rounding shipped this round is the available win.
* **`fill_background_from_func`** (19.2 s / 2.4%) is already a kernel, launched
  once per chunk, evaluating the scene's procedural background per pixel per
  frame. That is real work, not overhead.
* **Anything on the animation device.** `rematerialize_state_at_times` and
  friends run on the CPU by design; launching Taichi against CPU tensors makes
  it stage every argument through VRAM from the prep worker, which is a
  regression that has already shipped once. See `generate_array_states`'s
  docstring.

**Part 1 total if all land: ~120 s of 802 s (~15%).** T1 and T2 are in, which
was ~73 s of that ~120 s on paper; the criterion microbenchmarks came in at ~6x
and ~5.4x respectively, so the paper estimate looks about right, but it has not
yet been confirmed against the reference scene (see "Status -- start here").

---

## Part 2 -- The prep worker (no Taichi; CPU by design)

The larger pole, and untouched by any kernel work.

| target | measured | note |
| --- | --- | --- |
| `AttributeTimeline.rematerialize_state_at_times` | 193.3 s (24.1%) | already 3.1x faster this round via the rank dedup; what remains is the `[T, N, D]` zero-fill + `index_copy_` and the per-attribute `prepare_for_queries` / `rows_for_mob_ids` |
| `set_state_to_times` own time | 125.1 s (15.6%) | the function-replay loop; the first batch replays 1403 recorded functions |
| `AttributeTimeline.get` | 96.5 s (12.0%), 658k calls | ~150 us/call of Python + torch dispatch; the fix is fewer calls, not faster ones |
| `get_batch_of_primitives` own time | 93.6 s (11.7%) | geometry-build orchestration |
| `BezierCircuitCubic.get_render_primitives` | 50.8 s (6.3%), 11607 calls | per-actor build; `build_render_primitives_batched` already covers the batchable case |
| `memory reclaim` | 39.3 s (4.9%) | now almost entirely `torch.cuda.empty_cache()` handing blocks back to the driver, ~77 ms a call on WDDM |

### P1 -- the compact rematerialization buffer (next up)

The single structural idea here: `rematerialize_state_at_times` allocates and
zeroes a `[T, N, D]` buffer where `N` is **every row the scene ever allocated**
(505 407 for `location` on this scene) while only ~31% of rows are active in a
given window. Keeping a compact `[T, R, D]` buffer plus a global-row map would
cut both the zero-fill and the `index_copy_`, but every reader of
`active_state` -- including `get(copy=False)`, which hands out *views* -- would
have to go through the map. That is the one prep change with a large payoff and
a real design cost.

Where it actually is:

* [`timeline.py:485-487`](algan/animation_timeline/timeline.py:485) --
  `out = torch.zeros((T, N, D)...)` followed by `out.index_copy_(1, active_rows,
  ...)`. **This is the whole target.** The rows are already computed compactly
  by `_query_row_states(times, prepared, active_rows)`; the full-width buffer is
  built only to scatter them back into global row order.
* The comment immediately above it states the constraint: the global layout is
  kept "for animated-function replay", and rows outside the working set "stay
  zero and are never consumed by primitive preparation for this batch". So the
  question to answer first is **who actually indexes `active_state` by global
  row** — `AttributeTimeline.get` /`__setitem__`
  ([`timeline.py:633-676`](algan/animation_timeline/timeline.py:633)) do, via
  `mob_id_to_inds` ranges, and `get(copy=False)` hands out views into it.
* `rematerialize_state_at_times`
  ([`timeline.py:1066`](algan/animation_timeline/timeline.py:1066)) also applies
  the lifespan endpoint mask with `self.active_state[:, rows] *= mask`, another
  global-row index that would move to the map.

Suggested attack, cheapest first:

1. **Measure before designing.** Split the current cost between the zero-fill
   and the `index_copy_` (`torch.cuda`/CPU timers around those two lines are
   enough). If the fill dominates, a `torch.empty` + explicit zeroing of only
   the *inactive* rows already gets most of it for a fraction of the risk.
2. Only if that is not enough, do the compact buffer + row map, and put the map
   behind the existing accessors rather than at every call site. `get(copy=False)`
   returning views is the hard constraint -- a compact buffer must still hand out
   a view for a contiguous mob row range, which it can, since `mob_id_to_inds`
   ranges stay contiguous under a monotone remap.
3. Gate it. Prep is not covered by any pixel test on its own, so validate with a
   full-render run *and* an A/B script comparing materialized `active_state`
   tensors between the two paths.

## Part 3 -- Authoring (217.9 s, outside `save_video`)

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

1. **Re-profile the reference scene.** Not optional and not a formality: T1/T2
   removed ~86 s of measured torch work from the render thread, and this
   document's ranking predates that. See "Reproducing the profile" at the top.
2. **P1** (compact rematerialization buffer) -- the prep pole's big one; without
   it the Part 1 wins are capped by prep. Detailed above, in Part 2.
3. **T3**, **T5**, **T6** -- in whichever order the new profile ranks them.
4. **T4's remaining half** (the `allocate()` zero-fills) -- small, and its
   estimate is now unproven; fold it in whenever the dice is open anyway.

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
