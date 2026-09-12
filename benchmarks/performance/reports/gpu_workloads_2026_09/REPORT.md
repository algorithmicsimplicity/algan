# Where the default renderer's time goes on a GPU, and what would give the most back

Two workloads (`explainer_scene.py`, a Manim-style math explainer;
`graphics_scene.py`, a Three.js-style lit, shadowed, reflective 3-D scene),
two GPUs (Kaggle Tesla T4 on CUDA; GitHub's virtualized M1 on Metal), two
presets (PREVIEW 704x396 @ 10 fps; UHD 3840x2160 @ 60 fps), every number a
warm second run with the stage profiler and (on the T4) Taichi's per-kernel
GPU profiler on. The raw logs are beside this file; `README.md` says how they
were taken. Branch `claude/peaceful-carson-px98kh` at `921997b`.

## 1. The answer in one table

The renderer has **two different bottlenecks for the two workloads**, and
neither is "the ray tracer is slow".

| workload | T4 PREVIEW | T4 UHD | Mac PREVIEW | Mac UHD |
| --- | ---: | ---: | ---: | ---: |
| explainer, warm | **1.75 s** / 60 fr = 29 ms/frame | **4.05 s** / 30 fr = 135 ms/frame | 9.62 s / 60 fr = 160 ms/frame | 18.52 s / 15 fr = 1.23 s/frame |
| explainer, cold | 104 s | 85 s | 94 s | 53 s |
| graphics, warm | **8.21 s** / 60 fr = 137 ms/frame (7.91 s in a second session) | **115.4 s** / 30 fr = 3.85 s/frame (109.3 s in a second session) | 130.5 s / 60 fr = 2.17 s/frame | ~110 s/frame: a 2-frame diagnostic took 290 s cold, ~70 s of it compile (§6.4); the 15-frame job hit its 60-minute timeout |
| graphics, cold | 143 s | 231 s | 415 s | 290 s for 2 frames |
| graphics, **shadows forced off** (`--no-shadows`) | | **25.2 s** warm | | 134 s cold for 2 frames (~42 s/frame) |
| T4 GPU utilization, warm (avg) | 11% / 32% | 17% / 46% | | |

* **The graphics workload is bound by one kernel: shadow rays.**
  `raster_shadow_trace` is **67% of the UHD render (77.5 s of 115.4 s) and
  37% at PREVIEW** on the T4. Its cost is the ray *count*: an area light with
  `samples=3` contributes 9 emitter rows and each row fires a fixed
  `SOFT_SHADOW_SAMPLES = 8` fan, so every shading event pays **75 shadow rays
  (72 for the area light, 3 for the other lights) at every one of up to seven
  bounces**. All of Taichi's other kernels together are 15% of that render.
  The A/B arm confirms the attribution: the same UHD clip with
  `SETTINGS.raytracing.shadows` forced off renders in **25.2 s against
  109.3 s** in the same session -- shadows are 77% of the render, and the
  profiler's 67% for the trace kernel plus the shadow-event build's host
  glue is that number.
* **The explainer workload is bound by everything except kernels.** At
  PREVIEW on the T4 every Taichi kernel together is 0.03 s of a 1.75 s render
  (the GPU idles 89% of the time); the render is scene preparation (52%), a
  `gc.collect()` the render loop runs once per render (24%) and the host side
  of the sheet compaction. At UHD the kernels are 0.3 s of 4.05 s; the render
  is the compaction's torch chain (26%), the device-to-host frame copy (17%)
  and preparation (25%). On the Mac the compaction chain alone is 56% of the
  UHD render.
* **Cold start is the largest number a user actually sees.** A first render
  in a process costs 53-231 s against 1.75-115 s warm; 59-85 s of it is the
  compile of one kernel (`sheet_resolve_shade`) and 10 s is `torch.compile`
  of the PN dice, and each preset and scene variant recompiles.

Ranked by wall time it would give back on the T4 (§5 has the how and the
risk):

| # | target | workload it helps | what it costs today | plausible gain |
| --- | --- | --- | --- | --- |
| 1 | shadow-ray budget: one stratified fan per light per event, bounce-depth budget, cone/hemisphere culling | 3-D scenes with area or soft lights | 77% of graphics UHD (measured by the shadows-off arm), 37% at PREVIEW | **shipped: -19% UHD, -10% PREVIEW (§5.1.1)**; the rest is hard-light traversal, not ray count |
| 2 | the sheet compaction's torch chain, fused into kernels and rid of its per-chunk readbacks | everything; dominant for 2-D at UHD, and on Metal | 26% of explainer UHD (T4), 56% (Mac); 7% of graphics UHD | 1.3x on 2-D UHD (T4), ~2x on Metal |
| 3 | scene preparation off the critical path: split the first batch so prefetch can overlap; GPU-side circuit sampling and PN dice | every short render at PREVIEW | 52% of explainer PREVIEW, 25% of graphics PREVIEW | 1.5-2x at PREVIEW |
| 4 | fixed per-render overheads: the pre-render `gc.collect()`, the pageable frame copy, the encode tail | every render, most visible on short ones | 24% of explainer PREVIEW; 17% of explainer UHD | 0.3-0.7 s per render |
| 5 | reflection-ray budget: roughness-aware use of the glossy prefilter instead of tracing, contribution cutoff | scenes with large glossy surfaces | 89% of UHD pixels spawn a bounce; traverse + shade + compaction of 222 M continuation rays = ~12% of graphics UHD | 1.1x on graphics UHD |
| 6 | cold start: ship or persist compiled variants, fewer specializations | every first render | 53-231 s per process | the single biggest user-facing win; not a "renderer speedup" |

Hardware rasterization (§5.7) is **not** on this list: the exact
coverage emission it would replace (`raster_tri_count`/`raster_tri_write`,
`raster_bez_*`) is 1-2% of either UHD render on the T4.

## 2. Method, briefly

* Scenes: `benchmarks/performance/explainer_scene.py` and
  `graphics_scene.py`, storyboarded as fractions of the clip length so the
  same authoring renders at any preset and length (`_profile_cli.py`). What
  each contains is in `README.md`; both were rendered locally and eyeballed
  before being sent to the GPUs.
* Both GPU runs read RUN 2 (warm). RUN 1 pays Taichi's JIT and
  `torch.compile`; its stage table is what §4.6 uses.
* The T4 numbers carry Taichi's kernel profiler; the Mac's do not (its
  per-launch costs include the runner's virtualization tax, so its wall times
  over-rank launch- and sync-bound stages -- `../mac_2026_09/FINDINGS.md` §0).
* Every clip was one primitive batch, so the prefetch worker never ran and
  **scene preparation was entirely serial** with the render. That is what a
  6-second PREVIEW or a 0.5-second UHD clip does today, so it is the right
  measurement for short renders; long clips (2+ batches) hide part of
  preparation behind the previous batch -- the `nn_scene_*` reference steps
  in `t4_session2_nn_*.log` are that case (§6.2).
* The graphics scene's two warm runs produce different mp4 digests; the
  explainer's are identical. That is the documented split-pixel
  non-determinism of reflective geometry under analytic AA
  (`agent_guidance/memory_perf.md`), not a fixture defect.
* The T4 telemetry reports `SwPowerCap` throttling during both UHD runs (SM
  clock 300-1590 MHz, mean 790-840). UHD numbers on this box move ~3% run to
  run for that reason; the shares do not.

## 3. Warm stage tables, T4

Shares are of the warm end-to-end time. `own` is the stage's exclusive time
(children and kernels subtracted).

### 3.1 graphics, UHD, 30 frames: 115.4 s (109.3 s in the second session)

| stage | s | share |
| --- | ---: | ---: |
| `raster_shadow_trace` kernel (1198 launches, 2396 device records, 32 ms avg, 560 ms max) | 77.9 | 67.5% |
| `wavefront_traverse_events` kernel (1070 launches) | 7.7 | 6.7% |
| sparse discovery (fragment emission + sort + compaction), incl. | 8.1 | 7.0% |
| of which `compact_sheets` own torch time | 2.7 | 2.4% |
| of which fragment sort | 1.1 | 0.9% |
| `sheet_resolve_shade` kernel (350 launches) | 2.2 | 1.9% |
| `wavefront_shade` kernel (1041 launches) | 2.1 | 1.8% |
| tile state allocation (181 tiles) | 2.3 | 2.0% |
| `compact_ray_slots` kernel (1216 launches) | 1.7 | 1.5% |
| `wavefront_shadow_events` kernel | 1.5 | 1.3% |
| bounce loop host glue (shadow-event build own time, bounces 0-6) | 2.9 | 2.5% |
| arena preflight (projection + merge + BVH), preparation | 1.1 + 1.0 | 1.8% |
| post-process incl. device-to-host copy | 0.7 | 0.6% |
| unaccounted on the render thread | 0.5 | 0.4% |

Continuation rays: 222.1 M enter bounce 0 (89% of the 249 M rendered
pixels: the metallic slab, the chrome and glass spheres and the torus all
spawn one), 65 M survive to bounce 1, 42 M to bounce 2, 24 M to bounce 3,
14 M to bounce 4, 5 M to bounce 5, 3 M to bounce 6, 1.8 M to bounce 7. Every
lit hit on every one of those bounces builds shadow events, and the event
count times 75 rays is what `raster_shadow_trace` traces. GPU kernel time is
82% of the wall clock here; the rest is host orchestration of 33 chunks x
5-6 tiles x 8 bounces (about 6,300 kernel launches). Peak allocation
12.5 GB; the 16 GB card ran one frame per chunk.

### 3.2 graphics, PREVIEW, 60 frames: 8.21 s

| stage | s | share |
| --- | ---: | ---: |
| `raster_shadow_trace` kernel (55 launches, 27 ms avg) | 3.03 | 36.9% |
| `Scene._get_batch_of_primitives` (timeline materialization + geometry) | 1.06 | 12.9% |
| arena preflight (projection, PN dice 0.45, merge + BVH 0.37) | 1.02 | 12.5% |
| sparse discovery incl. (compaction 0.46) | 0.77 | 9.4% |
| `wavefront_traverse_events` kernel | 0.37 | 4.6% |
| shadow-event build, bounces 0-7, own | 0.19 | 2.3% |
| `wavefront_shade` + `sheet_resolve_shade` kernels | 0.35 | 4.2% |
| `AttributeTimeline.get` (4,148 calls) | 0.23 | 2.8% |
| unaccounted on the render thread | 0.47 | 5.8% |

Same shape as UHD, with preparation now a quarter of the render because the
frame is 30x smaller and preparation is not.

### 3.3 explainer, UHD, 30 frames: 4.05 s

| stage | s | share |
| --- | ---: | ---: |
| sparse discovery incl. | 1.07 | 26.4% |
| of which `compact_sheets` (torch: lexsorts, `unique_consecutive`, gathers) | 0.65 | 16.1% |
| post-process: device-to-host copy of 30 UHD frames (746 MB) | 0.68 | 16.9% |
| arena preflight (projection + merge + BVH) | 0.50 | 12.4% |
| `Scene._get_batch_of_primitives` | 0.42 | 10.3% |
| tile composite (gloss pyramid + accumulate) | 0.32 | 7.8% |
| `AnimationTimeline.set_state_to_times` own | 0.15 | 3.7% |
| video encode tail | 0.14 | 3.5% |
| tile state allocation | 0.12 | 2.9% |
| every Taichi kernel together (profiler) | ~0.30 | 7% |
| unaccounted on the render thread | 0.37 | 9.1% |

The largest kernels are `gloss_pyramid_level` (46 ms, 522 records -- the
one `MeshStandardMaterial` sphere makes every chunk build a glossy pyramid),
the bezier emission (`raster_bez_count`+`write`, 62 ms), `gloss_composite`
(29 ms), `tonemap_to_u8` (27 ms) and `sheet_resolve_shade` (20 ms). The
render is 7.4 fps at UHD with the GPU 17% busy.

### 3.4 explainer, PREVIEW, 60 frames: 1.75 s

| stage | s | share |
| --- | ---: | ---: |
| `Scene._get_batch_of_primitives` incl. (timeline replay 0.27, geometry 0.19) | 0.47 | 27.0% |
| arena preflight incl. (merge + BVH 0.19, bezier sample+pack 0.14, PN dice 0.08) | 0.44 | 25.3% |
| unaccounted on the render thread -- `gc.collect()` before the render (§4.4) | 0.42 | 24.2% |
| ray traced render total incl. | 0.37 | 21.1% |
| of which sparse discovery (compaction 0.10) | 0.16 | 8.8% |
| of which tile composite (531 `gloss_pyramid_level` launches) | 0.12 | 6.6% |
| every Taichi kernel together (profiler) | 0.03 | 1.7% |

34 fps at PREVIEW, GPU 11% busy. **Preparation plus the collection is
three quarters of the render.**

## 4. What the tables say

### 4.1 Shadow rays are the 3-D workload

`raster_shadow_trace_arena` runs one thread per `(event, light)` cell and
traces that cell's fan in a loop: one ray for a hard light, `SOFT_SHADOW_SAMPLES`
(8) for a row with a non-zero emitter extent (`raster_taichi.py` ~2891-3060;
`settings.py` ~3505-3560 documents the cost rule). The graphics scene has 12
light rows (default point, directional, spot, 9 area cells) so an event
costs 75 rays, and the scene generates events at every bounce of a
reflection chain that reaches seven deep on the slab. The kernel is 82% of
GPU time and its per-launch time varies 32 ms mean / 560 ms max, which is
the fan loop's divergence: a warp whose events straddle the penumbra of the
area light runs 8 traversals per row while its neighbours run 1.

The path tracer already integrates the same emitter with one random point
per sample; the deterministic renderer integrates it with a fixed 8-ray fan
per cell, per event, per bounce, whether or not the pixel is anywhere near a
penumbra. Nothing about analytic AA or the transparency stack depends on
that fan: shadow visibility is a scalar per (event, light), consumed by the
shading stage.

### 4.2 The compaction is the 2-D workload's render

`prepare_sparse_raster_coverage` -> `compact_sheets` is the host chain that
turns the emitted fragment stream into per-pixel sheets: window-pair
expansion, count/write emission, an exact fragment order, `unique_consecutive`
over pixels, opaque-prefix truncation, one-mesh caps, band/rank/class
grouping, sibling weights, depth ownership, CSR offsets. On CUDA several of
its sorts and reductions are already kernels (`pixel_group_order`,
`rank_groups`, `key_run_order`, `sheet_band_reduce`, `sheet_lane_*`,
`sheet_depth_lose`); the remaining torch ops and the readbacks between them
(`counts.sum().item()`, `prefix[_pair_starts].tolist()`, `opaque_s.any().item()`,
`unique_consecutive(..., return_counts=True)` twice) are what the
26%/16% own time is. Each is small; there are about thirty of them per
chunk, and a readback drains the queue.

On the Mac it is 56% of the explainer's UHD render (`compact_sheets` 6.3 s
of 18.5 s, fragment sort 1.5 s, window pairs 1.1 s) with `sheet_resolve_shade`
at 0.24 s: torch's MPS sort and unique are the slow arm there
(`../mac_2026_09/DEVICE_SORT.md`), and every readback waits out the queue.

### 4.3 Preparation is serial, and at PREVIEW it is the render

With one batch per clip there is nothing for the prefetch worker to overlap,
so `_get_batch_of_primitives` (timeline replay, bezier circuit geometry,
`AttributeTimeline.get` at 1,268-4,148 calls a render) and the arena
preflight (bezier sample + pack, PN dice, merge, refit-BVH build) run on the
render thread before the first kernel. On the T4 that is 0.9 s of the
explainer's 1.75 s and 2.1 s of the graphics scene's 8.2 s at PREVIEW, and
0.9-2.1 s at UHD too -- it does not scale with resolution, so it is a fixed
floor under short renders. `refit_bvh._binary_split` is a Python loop
(28 calls, 0.09 s locally); `_circuit_edge_inward_signs` is 0.05 s of Python
per render.

### 4.4 Fixed costs: a collection, a copy, a drain

* **`gc.collect()`** -- `scene_excluded_from_gc` (`memory_utils.py` ~487)
  runs one full collection before freezing the scene, and a full collection
  walks the authored scene: measured **0.28 s** locally with cProfile on
  (`raytracing_cprofilecprof_explainer_run2.txt`, `gc.collect` 0.284 s
  tottime of a 1.21 s render) and it is the bulk of the 0.37-0.47 s
  "unaccounted" on every T4 row. It is the one unhooked item on the render
  thread; 24% of the explainer's PREVIEW render.
* **The device-to-host copy** -- 0.68 s for 30 UHD frames on the T4 is
  1.1 GB/s, a pageable synchronous `.cpu()`; the card sits on PCIe 3.0 x16,
  whose pinned transfers run an order of magnitude faster than that. It sits on the render thread between chunks.
* **The encode tail** is small here (0.03-0.14 s) because `-preset ultrafast`
  keeps up; a default preset would not on these 2-4 vCPU boxes.

### 4.5 Reflection rays

89% of the graphics scene's UHD pixels spawn a bounce-0 continuation -- the
slab is a `MeshStandardMaterial(metalness=0.55, roughness=0.18)` and covers
most of the frame -- and the chain averages 1.7 more bounces per pixel. The
glossy prefilter (`gloss_pyramid_level`/`gloss_scatter`/`gloss_composite`,
0.57 s) exists to blur those reflections after tracing them; the tracing
itself (traverse 7.7 s, shade 2.1 s, compaction 1.7 s, shadow events on the
secondary hits) is the second-largest block of the render.

### 4.6 Cold start

| step | cold | warm | compile and warm-up attributed |
| --- | ---: | ---: | --- |
| explainer PREVIEW | 104.2 s | 1.75 s | `sheet_resolve_shade` launch 59.1 s, `wavefront_shade` 12.5 s, PN dice (`torch.compile`) 10.7 s, tri projection 4.4 s, `raster_tri_*` 6.2 s |
| explainer UHD | 84.9 s | 4.05 s | same variants, second process of the session |
| graphics PREVIEW | 142.6 s | 8.2 s | |
| graphics UHD | 231.0 s | 115.4 s | `sheet_resolve_shade` 84.6 s (two modes), `wavefront_shade` 11.8 s, PN dice 7.5 s; `raster_shadow_trace` itself cold 80.2 s vs warm 77.9 s |
| Mac explainer PREVIEW | 94.1 s | 9.6 s | |
| Mac explainer UHD | 52.9 s | 18.5 s | |

Every step ran in a fresh process with `ALGAN_CACHE_DIR` on the persistent
disk, and every step paid the full compile: the offline cache did not
shorten the second explainer step's `sheet_resolve_shade` (59 s -> 45 s at
UHD; `t4_explainer_uhd.log`). `../t4_2026_09/README.md` recorded the same
and left it open.

## 5. The targets, in order

Each item keeps the two things the renderer must keep: exact analytic
coverage (the sheet emission and resolve are untouched by 1, 3, 4, 5, 6 and
unchanged in output by 2) and correct compositing of arbitrarily stacked
transparent fragments (the sheet pipeline). Each names a kill switch, as
the codebase's optimizations do.

### 5.1 Shadow-ray budget (graphics UHD 115 s -> ~40-55 s)

The kernel is already a lean any-hit traversal; the win is tracing fewer
rays.

1. **One stratified fan per light, not per cell.** Integrate a
   `RectAreaLight`'s visibility with `N` low-discrepancy samples over the
   whole rectangle (Cranley-Patterson rotated per pixel, so the set is
   deterministic per pixel and the fan is the same in both shadow passes),
   with `N` = the light's `samples` squared or a new `shadow_samples`
   default of 8-16 -- instead of `K` cells x 8. At `samples=3` that is
   72 -> 9-16 rays per event, 4.5-8x fewer; the radiance and power
   fractions per row stay as they are (the rows still carry `1/K` each and
   the per-cell visibility becomes the shared integral). This is what the
   path tracer does with its one point per sample. Penumbrae get slightly
   noisier per pixel (dithered, not banded); the R2 fan's structured banding
   goes away with it.
2. **A bounce-depth budget.** Secondary hits (bounce >= 1) are seen through
   a reflection or refraction and carry the surface's Fresnel weight;
   give them 1 ray per light (cell centre) and reserve the fan for bounce 0.
   In this scene bounces 1-7 are 51% of all events. Deterministic, and
   visually below the split-pixel noise floor on glossy surfaces.
3. **Cull before launching.** A spot's cone and a directional light's
   hemisphere are known per event: an event outside the cone (`SpotLight`
   `cone_angle`) or facing away (`n . l <= 0`) needs no ray. The kernel
   already skips zero-radiance rows (the "geometric zero-radiance culling"
   block); extend it to the cone test and to the area light's own
   half-space, and move the test to `wavefront_shadow_events` so the events
   are never emitted.
4. **Penumbra detection for what is left.** Trace the cell-corner rays first
   (the `adaptive_taps` idea the kernel already has for the AA sub-pixel
   fan); if all agree, the row is fully lit or fully shadowed and the fan
   is skipped. Most pixels of a shadowed scene are one or the other. Exact
   where it fires; a false "all agree" on a thin occluder is the same
   sampling error the fixed fan already has.
5. **Thread per ray, not per cell.** A grid of `(event, light, sample)`
   removes the fan loop's divergence (32 ms mean / 560 ms max per launch is
   a warp-imbalance signature) and lets a sort by `(light, direction
   octant)` -- `shadow_primary_sort` exists for the primary pass, apply it to
   the bounce passes -- make traversal coherent. Worth 1.3-1.5x on the
   kernel by itself; second-order next to 1-4.

Kill switches: `SETTINGS.raytracing.experimental.shadow_light_fan`,
`shadow_bounce_budget`, per the existing `area_light_soft_shadows` shape.
Validation: `benchmarks/_area_light_shadow_check.py` (the fan's acceptance
harness) against the path tracer, and the shadows-off arm
(`--no-shadows`, §6) as the floor.

#### 5.1.1 Measured: the budget as shipped (commit `a63a33d`)

Items 1 and 2 are implemented -- `SETTINGS.raytracing.experimental.shadow_ray_budget`
(16 rays per light per primary event, stratified over an area light's
cells and Cranley-Patterson rotated per event) and `shadow_bounce_rays`
(one ray per soft row at a secondary hit); `ALGAN_SHADOW_RAY_BUDGET=0` is
byte-identical to the pre-budget renderer (0 of 129,600 pixels differ
against a pre-budget checkout, `benchmarks/_shadow_budget_check.py`).
One T4 session, both arms of each pair back to back (`t4_budget_*.log`):

| arm | warm end-to-end | `raster_shadow_trace` kernel | primary shadows (in `sparse resolve`) | secondary shadow stages |
| --- | ---: | ---: | ---: | ---: |
| graphics UHD, legacy fan | 117.1 s | 79.3 s | 45.9 s | 54.9 s |
| graphics UHD, budget | **94.9 s (-19%)** | **57.2 s (-28%)** | 33.7 s (-27%) | 45.0 s (-18%) |
| graphics PREVIEW, legacy fan | 8.27 s | 3.11 s | 1.65 s | 2.72 s |
| graphics PREVIEW, budget | **7.43 s (-10%)** | **2.12 s (-32%)** | 1.17 s | 2.19 s |

Picture: 0.44% of the check frame's pixels move, 59% of them by one level,
the tail (up to 66) in the penumbrae and in the glossy sphere's reflection,
as dither. The 4x zoom in the session shows no visible change.

**Why -28% and not the -70% the ray count promised.** Two things the
tables did not show before the arms were run:

1. **The legacy fan was already small at secondary hits.** A deferred bounce
   event carries a single covered sub-pixel position, and under analytic AA
   the legacy fan skips every sample whose position is not covered -- so of
   an area row's 8 rays only 2 were traced at a secondary hit, and a hard
   light's 4 adaptive taps were 1. The budget takes those 2 to 1; half the
   events were never paying 75 rays.
2. **The hard lights dominate what is left.** A primary event now traces
   about 18 area rays plus 6-12 for the three hard lights (2-4 sub-pixel taps
   each), a secondary event 9 plus 3 -- and the shadow kernel lost only 28%
   for a 2.6x cut in rays, which says the hard-light rays (the directional
   at `ldist = 1e7`, and a point light near the camera, both crossing the
   whole scene) cost far more per ray than the short area-light rays, and
   that a warp's time is set by its slowest `(event, light)` thread.

So the remaining shadow time is a **traversal** problem, not a count
problem, and the next lever is item 5 (one thread per ray, light-major
ordering so a warp traces 32 rays toward one light, the primary sort
applied to the bounce passes), with the directional light's unbounded ray
the first thing to measure on its own. Items 3-4 are second order now.

### 5.2 The compaction chain (explainer UHD: -25%; Mac: -50%)

The sorts-plus-scans design (`DESIGN_sheet_resolve.md` §2) was meant to run
in the kernel language; today about half of it does. What remains on the
host, in order of cost on the T4:

1. **Readbacks.** `num_frags = int(counts64.sum().item())`,
   `prefix[_pair_starts].tolist()`, `bool(opaque_s.any().item())`,
   `int(keep.sum().item())`, `_read_tile_alloc`'s 30-int copy: each drains
   the queue before the next torch op can be issued, so the GPU idles
   between them (utilization 17% at UHD). Replace with device-side counts
   consumed by kernels (capacity bounds already exist for every array) and a
   single readback per chunk.
2. **`unique_consecutive(pix_s, return_counts=True)`** twice, plus the
   `torch.cumsum` for `run_offsets` -- a run-boundary kernel over the
   pixel-sorted stream (`key_run_order` already exists) gives both.
3. **`_lexsort` / `_exact_fragment_order`** on CUDA is CUB via torch and
   fast (15.8 ms for 2.9 M keys); on Metal it is 215 ms. The run sort
   (`pixel_group_order`) removed the big one; `sheets lexsort` still runs
   for the band/class keys. Same fix: a per-pixel-run kernel.
4. **Everything per chunk is per frame.** 33 chunks at UHD, each paying the
   full chain; fragment streams for 2-3 frames at a time would amortise the
   fixed part (the tile allocation alone is 2.3 s / 181 tiles at UHD
   graphics). The chunk count is the memory model's; the compaction's
   per-chunk fixed cost is what should be reduced.

Output is byte-identical by construction (same reductions, same fixed
trees); `benchmarks/_sheet_kernel_check.py` and `_sheet_compact_breakdown.py`
are the existing harnesses. On the Mac this is the whole game: the torch
ops are 2.5x faster than its three CPU cores and every readback is a queue
drain (`../mac_2026_09/SHARED_QUEUE.md`).

### 5.3 Preparation off the critical path (PREVIEW: 1.5-2x)

1. **Split the first batch.** Fetch frames `[0, n)` first with `n` small
   (one chunk's worth), start rendering, and let the prefetch worker
   prepare `[n, end)` behind it. Today a 6-second PREVIEW clip is one batch
   and the worker never runs; the memory model already re-fits the window
   after every batch, so nothing about capacity changes. The loop's
   `_arena_fetch_frame_cap` is the natural place to cap the *first* fetch.
2. **Circuit geometry on the device.** `_build_circuit_geometry` (0.10-0.63 s
   per render) and `_compute_samples_per_segment` are torch on the CPU
   animation device; `_circuit_edge_inward_signs` is 24 ms of Python. The
   bezier sample + pack is the explainer's largest preparation item.
3. **PN dice.** 0.45 s per render for the graphics scene's six surfaces at
   PREVIEW (`_dice_logical_pn` 0.20 own; `snap boundary`, `evaluate
   normals`, `interpolate attrs` the rest). The level searches are torch on
   the animation device; the criterion kernels exist on CUDA
   (`pn_criterion_kernel_active`).
4. **`AttributeTimeline.get`**: 1,268-4,148 calls a render at 55-70 us each.
   A batched query for a materialization pass is the shape the timeline
   guide already describes.
5. **`refit_bvh._binary_split`**: a Python recursion (28 calls, 0.09 s) --
   `refit_pack_kernel` exists behind `ALGAN_REFIT_PACK_KERNEL` (off).

### 5.4 Fixed per-render overheads (0.3-0.7 s per render)

1. **The pre-render `gc.collect()`** (`scene_excluded_from_gc`): 0.28-0.42 s.
   `gc.freeze()` without the collection, or a collection of generation 0-1
   only, keeps the freeze's benefit (the render's own collections skip the
   scene) and drops the full walk. Cost of skipping it: pre-existing cycles
   stay uncollectable for the render's duration, which is what `freeze`
   does to them anyway. Equivalent: run the collection on the prefetch
   worker while the first kernels compile or launch.
2. **Pinned, asynchronous frame copies.** Allocate the host frame buffer
   pinned, copy with `non_blocking=True` into a double buffer and hand the
   previous chunk's frames to the writer while the next chunk renders.
   0.68 s -> ~0.06 s at UHD on the T4 and it comes off the render thread
   entirely. On the T4 `h264_nvenc` could also consume device frames
   directly, which removes the copy; that is an encoder-path change and the
   x264 arm is still needed for parity fixtures.
3. **The encode tail** is already overlapped; keep `-preset` fast on boxes
   with few cores (the benchmark scripts do).

### 5.5 Reflection-ray budget

1. **Roughness-aware tracing.** Above a roughness threshold (0.3-0.4) the
   prefiltered glossy pyramid already produces a blurred reflection from the
   traced result; for a rough metal the environment map's prefiltered
   lookup plus the direct-light specular term is what three.js shows, and
   it needs no continuation ray at all. Keep the ray for roughness below
   the threshold and for every dielectric (glass needs the refraction).
2. **Contribution cutoff.** Stop a chain when its accumulated weight falls
   under 1/255 (the u8 write cannot show it) -- the bounce table shows 1.8 M
   rays reaching bounce 7 with Fresnel-attenuated weights.
3. **Depth budget by material**: `max_bounces` is global; per-material
   `reflection_bounces` (1 for the slab, 4 for glass) is what a graphics
   author expects to set.

### 5.6 Cold start

Not a renderer speedup, but for a user who renders once per process it is
the number: 104 s cold against 1.75 s warm for a six-second PREVIEW clip.

1. **Make the offline cache actually hit.** Two processes in one session on
   the same disk both compiled `sheet_resolve_shade` (59 s then 45 s). Find
   out whether the specialization key differs per preset (it should not for
   a resolution change) or the cache is not consulted, then fix that: it is
   worth more than every warm optimization above for short clips.
2. **Ship compiled variants** for the common `(device, features)` keys with
   the wheel or build them at install (`benchmarks/_taichi_aot_build.py`
   exists), and fold the `torch.compile` of the PN dice into the same
   step or replace it with a kernel.
3. **Fewer specializations.** `sheet_resolve_shade` compiles per material
   pipeline tuple, per light-slot bucket, per mode; each variant is 30-45 s.
   A runtime material-pipeline dispatch for the rare pipelines (keep the
   specialization for the common ones) trades a few percent of shade time
   for a bounded number of compiles.

### 5.7 What about hardware rasterization, and Quadrants patches?

The renderer's front end is already a rasterizer: exact conservative
coverage per (pixel, primitive) with the fragment stream compacted into
depth-banded sheets. A hardware pipeline could produce the candidate
`(pixel, primitive)` pairs (conservative raster into a fragment list) and
leave the exact clipping and the sheet resolve to compute kernels, but
`raster_tri_count` + `raster_tri_write` + `raster_bez_*` are **1.3 s of
115 s** at UHD graphics and 0.1 s of 4.05 s for the explainer on the T4.
There is no time to win there, and Quadrants has no graphics-pipeline
surface to patch into; do not spend a patch on it.

Compiler patches that could matter, in order:

* **Per-kernel register cap** (`0005-cuda-max-reg.patch`, `qd.loop_config(max_reg=)`)
  on `raster_shadow_trace` and `wavefront_traverse_events` -- the two
  traversal kernels, which are latency-bound with a large per-thread state
  (`../t4_2026_09/README.md` found the global knob never reached ptxas; the
  per-kernel one has not been measured on these kernels).
* **Read-only cache loads** (`0006-cuda-readonly-ndarray-ldg.patch`) for the
  BVH node arrays and `tri_pos`, which every traversal reads and never
  writes.
* **Invariant argument loads** (`0004`) are on by default; measure them
  off/on on the shadow kernel to confirm they engage through `ArenaView`.
* A **shared-memory traversal stack** or **persistent-threads** shape for
  the traversal kernels is a kernel rewrite, not a patch; it is the
  algorithmic lever `../t4_2026_09/README.md` left ("dependent-load latency
  and divergence in BVH traversal"). After 5.1 it is what remains of the
  graphics workload.

## 6. Reference and A/B steps

The second Kaggle session (`t4_session2_*.log`) re-ran the four profiles
(1.74 / 3.98 / 7.91 / 109.3 s warm against 1.75 / 4.05 / 8.21 / 115.4 s --
a 1-6% spread, UHD graphics the widest under `SwPowerCap`) and added
three steps:

### 6.1 graphics UHD with shadows forced off: 25.2 s

| stage | s | share |
| --- | ---: | ---: |
| sparse discovery incl. (compaction own 2.8, fragment sort 1.1) | 8.0 | 31.6% |
| `wavefront_traverse_events` kernel (593 launches) | 5.7 | 22.8% |
| `wavefront_shade` kernel | 1.5 | 6.1% |
| `raster_tri_write` kernel | 1.1 | 4.5% |
| tile state allocation / `compact_ray_slots` / compact active | 1.1 / 0.9 / 1.0 | 12% |
| preparation (`_get_batch_of_primitives` + preflight) | 2.1 | 8.5% |
| device-to-host copy | 0.8 | 3.3% |
| `sheet_resolve_shade` kernel | 0.8 | 3.1% |

This is what the graphics workload costs once the shadow rays are gone:
the compaction chain and the reflection traversal, in that order -- §5.2
and §5.5. The two runs' digests differ (split pixels), as expected.

### 6.2 The `nn_scene` references, and the worker-thread ledger

`nn_scene_UHD.py` (30 frames, shadows off): **8.39 s** warm, in the
7.8-8.5 s band `../mac_2026_09/DEVICE_SORT.md` recorded for it, so this
session is comparable to the earlier ones. `nn_scene_PREVIEW.py` (50 frames,
shadows on): **4.46 s** warm.

Both are multi-batch clips (2 batches), which is what the profiler fix was
for. The PREVIEW run's budget line now reads: render thread accounted
3.83 s + unaccounted **0.38 s (8.6%)**; worker-thread prep overlapped
1.05 s (23.5%); and the new `wait for prefetched batch` stage shows the
render thread idled **0.25 s (5.6%)** waiting for the worker -- so a
quarter of the second batch's preparation was on the critical path and the
rest was hidden. The same table used to print the worker's 1.05 s inside
the render thread's budget and an unaccounted line of about -0.7 s. At UHD
the wait is 0.000 s: the first batch's 8 chunks render for longer than the
second batch takes to prepare.

### 6.3 Mac, graphics PREVIEW: 130.5 s warm (2.17 s a frame at 704x396)

| stage | s | share |
| --- | ---: | ---: |
| sparse discovery incl. | 57.2 | 43.8% |
| of which `compact_sheets` (own 20.9) | 30.2 | 23.2% |
| of which window pairs / fragment sort / sibling weights | 6.9 / 4.9 / 3.1 | 11.4% |
| bounce drain incl. | 39.7 | 30.4% |
| of which `raster_shadow_trace` kernel (348 launches) | 20.2 | 15.5% |
| of which shadow-event build, bounces 0-7, own | 21.2 | 16.2% |
| sparse resolve incl. | 13.7 | 10.5% |
| PN dice | 4.4 | 3.3% |
| `memory reclaim (gc + cuda cache)` (42 calls) | 2.2 | 1.7% |

The Metal arm inverts the T4's ranking: the torch-op compaction chain is
first and the shadow kernel second, because the box's per-launch and
per-readback costs are two orders of magnitude above the T4's
(`../mac_2026_09/SHARED_QUEUE.md`). The shadow-event build's 21 s of *own*
time is the same host glue that costs 2.9 s on the T4 -- readbacks waiting
out the queue. The UHD arm never completed: 15 cold frames had not finished
50 minutes after the PREVIEW pass, so at UHD this scene is a multi-hour
render on that runner, and §5.2 is the only lever that reaches it.

### 6.4 Why the Mac's UHD graphics pass timed out: throughput, not a hang

A follow-up job (`mac_mps_graphics_uhd_2f_diagnostic.log`) rendered a
**2-frame** UHD clip of the same scene with PERF logging and line-mode
progress on, once with shadows and once with `--no-shadows`, one cold run
each:

| arm | end-to-end (cold) | of which kernel compile (launch column) | per frame, compile removed |
| --- | ---: | ---: | ---: |
| shadows on | 290.4 s | ~68 s (`sheet_resolve_shade` 51.4, `wavefront_shade` 7.6, `raster_tri_*` 3.5, `raster_shadow_trace` 2.4, `wavefront_shadow_events` 2.4) | **~110 s** |
| shadows off | 133.9 s | ~50 s (`sheet_resolve_shade` 36.6, `wavefront_shade` 9.2, `raster_tri_*` 3.6) | **~42 s** |

So the 15-frame pass needed roughly 28 minutes of rendering plus compile
after the 9-minute PREVIEW pass in the same job, which is the 60-minute
limit -- and the PERF log shows the memory model splitting even the 2-frame
batch ("Reducing the frame batch to fit memory: 1:4, 2:4"), so a 15-frame
batch on the 7 GB box pays several rematerialize-and-re-project rounds on
top. Nothing wedged; the log streams to the end and both arms exit 0.

Where the 110 s a frame goes is the Metal profile of §6.3 at UHD scale:
the **shadow-event build's own host time is 92 s of the 220 s** (bounces
0-7, 8.5-19 s each -- torch ops on MPS issued one small dispatch at a time,
each readback draining the queue), the `raster_shadow_trace` kernel itself
18.6 s, `sheet_resolve_shade` 1.9 s of device time under 51 s of compile,
the sparse resolve's own time 13.4 s, `compact_sheets` 26.5 s own and the
fragment sort 11.0 s. Shadows cost ~68 s of the 110 s a frame (62%; the T4
says 77%), and the rest is the compaction chain. Both arms of §5 that
matter on the T4 (5.1, 5.2) are what would reach this box too, with the
host-glue half of 5.1 -- fewer, larger dispatches in the shadow-event
build -- mattering far more here than on CUDA.

## 7. What this round did not measure

* A multi-batch clip of either new scene: the worker-thread ledger and the
  prefetch wait were exercised by the `nn_scene_*` reference steps (§6.2).
* The graphics scene at UHD on the Mac beyond a 2-frame diagnostic (§6.4).
* A physical Mac. The runner's virtualization tax is in every Mac launch
  and readback number; the Mac compaction share (56%) is directionally
  right and numerically inflated.
* Text through LaTeX (`Tex`): the explainer uses Pango `Text` so the GPU
  boxes need no TeX; a glyph is the same circuit either way.
* SPP > 1 (the path tracer), the FXAA/SMAA post-processes, transparent
  backgrounds.
