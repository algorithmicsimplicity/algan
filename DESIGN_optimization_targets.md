# Optimization targets

Ranked candidates for further speeding up Algan, measured against the project's
reference workload: `rl2/animations/scenes/s05_learning_to_program_setup.py` at
`LD` (864x486, 15 fps, 3883 frames). Every number below comes from one profile
run of that scene with `algan.utils.profiling_utils` on the current tree.

> **Read the percentages, not the seconds.** That run landed on a thermally
> throttled, RAM-contended machine: `save_video` took 802 s where a quiet run
> of the same code takes ~425 s. Shares of the run are stable; absolute
> seconds are inflated roughly 1.9x throughout. Where a claim is an estimate
> rather than a measurement it says so.

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

### T1. PN subdivision-level criterion -- 67.9 s (8.5%)

**The single biggest kernel opportunity.** `_dice_logical_pn` is 86.2 s
(10.7%), and 67.9 s of that is *choosing* the levels, not dicing:

| | measured |
| --- | --- |
| `_required_subdivision_levels` | 67.9 s |
| ↳ `_guarded_pixel_error` (4562 calls) | 32.1 s |
| ↳ `evaluate_logical_pn` (3121 calls) | 17.5 s |
| ↳ error-function own time (chords, einsum, masks) | 17.0 s |
| ↳ **per-level loop machinery** | **2.6 s** |

The loop bookkeeping -- `levels.clone()` per level, `_triangle_counts` over
`[T, P]`, the `.any()` / `.nonzero()` device syncs -- is 3.9% of the search.
Restructuring the climb, seeding it analytically from the 4^-L error scaling,
or compacting the per-level work to the active set would all buy ~nothing.
This was measured specifically to rule those out; do not spend time there.

What costs is the criterion. Per call `_guarded_pixel_error` runs ~30
elementwise ops -- **two** full perspective projections (12 ops each) plus
clamp / subtract / norm / guard / where / amax -- over `[K, N, 3]` arrays of
~2^18 elements, moving ~180 MB to produce one scalar per patch. Measured
4.4 ms/call, which is what that traffic costs at this card's bandwidth.

**Kernel design.** One `@ti.func` doing, per (frame, patch): evaluate the
logical-PN patch at the level's vertex and interior-sample parameters, project
both point sets, apply the guard box, and reduce to the patch's max pixel
error -- all in registers, one pass over the sample points, one scalar out.
Two kernels wrap it (`_patch_flatness_error`, `_edge_chord_error`); the edge
one substitutes `evaluate_cubic_curve` and the chord blend.

**Estimated** 67.9 s -> 8-12 s, i.e. ~55 s (6.9% of `save_video`).

Two constraints the implementation must respect:

* **Crack-freeness.** A boundary curve's level must come out bit-identical for
  the two patches sharing it. That holds inside a kernel for the same reason it
  holds now -- both evaluate the same canonically ordered controls through the
  same code -- but it is the property to regression-test
  (`benchmarks/_logical_pn_crack_check.py`).
* **Output changes.** Taichi initialises with `fast_math` on, so the fused
  arithmetic differs from torch's in the last bits and will flip borderline
  patches to a different level. That changes *geometry*, not just pixel
  rounding: silhouettes move, within `render_tolerance` by construction, but
  more visibly than a pure rounding change. Budget a re-baseline that is
  actually looked at. Gate it (`ALGAN_PN_CRITERION_KERNEL=0`) so the torch path
  stays reachable for A/B.

### T2. Bezier chord-count search -- 18.4 s (2.3%)

`RayTracedBezierCircuitPrimitive._compute_samples_per_segment` is the *same
computation shape*: climb subdivision levels, project each level's control
hull, reduce to a max error per segment, keep the first level under tolerance.
It has the same iterative structure, the same projection block written out
again, and the same per-chunk elementwise passes.

Fold it into T1's kernel family rather than writing a third one -- the only
differences are the evaluator (uniform cubic subcurves rather than a PN patch)
and that the bound is a control-hull distance rather than a sample deviation.

**Estimated** 18.4 s -> ~3 s (~1.9%). Do it immediately after T1, while the
projection/guard `@ti.func`s are fresh.

### T3. Bezier circuit geometry build -- 47.1 s (5.9%)

`_build_circuit_geometry` samples world-space polylines into per-circuit plane
coordinates and packs the per-circuit metadata the trace kernel reads. It is
elementwise sampling plus `repeat_interleave` / gather / scatter plumbing over
the whole batch's segments.

Less certain than T1/T2 -- it is a build, not a reduction, so the output is as
large as the input and the bandwidth argument is weaker. Worth a
`memory.scope`-level breakdown before committing. **Estimated** 47.1 s -> ~25 s
(~2.7%), lower confidence.

### T4. The dice write-out -- ~17 s (2.1%)

`_dice_logical_pn`'s own time (12.4 s) plus `interpolate_patch_attribute`
(3.1 s), `evaluate_logical_pn_normals` (1.0 s) and `snap_boundary_values`
(0.4 s). Two costs: the `allocate()` zero-fills (`[T, max_triangles, 3, D]` for
corners, normals, colours and every surface/shader parameter) and the
`diced_X[target_rows, target_columns] = ...` advanced-index scatters, which are
`index_put_` over a `[chunk, num_triangles]` destination.

Each selected (frame, patch) writes a *contiguous run* of columns, which
`index_put_` cannot exploit. A kernel that writes each patch's run directly --
or, more cheaply, a flattened `index_copy_` on `[T*M, 3, D]` -- should recover
most of it. Try the `index_copy_` version first: it is a few lines and needs no
kernel. **Estimated** ~17 s -> ~8 s (~1.1%).

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

**Part 1 total if all land: ~120 s of 802 s (~15%).**

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

The single structural idea here: `rematerialize_state_at_times` allocates and
zeroes a `[T, N, D]` buffer where `N` is **every row the scene ever allocated**
(505 407 for `location` on this scene) while only ~31% of rows are active in a
given window. Keeping a compact `[T, R, D]` buffer plus a global-row map would
cut both the zero-fill and the `index_copy_`, but every reader of
`active_state` -- including `get(copy=False)`, which hands out *views* -- would
have to go through the map. That is the one prep change with a large payoff and
a real design cost.

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

## Suggested order

1. **T1** (PN criterion kernel) -- biggest, and it establishes the projection /
   guard `@ti.func`s the rest reuse.
2. **T2** (bezier chord search) -- same kernel family, near-free once T1 exists.
3. **T4's cheap half** (`index_copy_` instead of `index_put_`) -- a few lines.
4. **P1** (compact rematerialization buffer) -- the prep pole's big one; without
   it the Part 1 wins are capped by prep.
5. **T3**, **T5**, **T6** -- in whichever order the next profile ranks them.

Re-profile after each: on a two-pole pipeline the ranking moves as soon as one
pole shrinks.
