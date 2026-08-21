# Renderer work queue

**What this is.** The output of a full audit of the render system (design docs,
docs, and code) carried out on 2026-08-21 at `2016a26`. Every claim below was
checked against the source; where a document and the code disagreed, the code
won and the disagreement is listed. The companion deliverable is
`docs/source/advanced_user_tutorials/renderer_limitations.rst`, which states the
same findings as user-facing limitations. This file says what to *do* about
them, ordered by likely impact.

**What it is not.** It is not a replacement for `DESIGN_optimization_targets.md`
(the performance plan of record), `DESIGN_sheet_resolve.md` (the current
renderer's design) or `DESIGN_mesh_identity_open.md` (the identity queue). It
cites all three, folds their open items into one ranking, and adds what the
audit found that none of them record.

**Evidence.** Measurements quoted from those documents are attributed and were
taken on the reference machine (Windows / GTX 1050). Everything attributed to
"this audit" was verified in this container: Ubuntu 24.04, 4 vCPU, **CPU-only**,
torch 2.7.1+cu126, Taichi 1.7.4. A CPU-only box cannot speak for CUDA
behaviour or for wall-clock rankings; where that matters it says so.

---

## Ranking

| # | Item | Kind | Why here |
| --- | --- | --- | --- |
| 1 | [Silent truncations have no instrument](#1-silent-truncations-have-no-instrument) | Correctness | Four ceilings degrade the image with no signal. Cheapest high-value item on the list. |
| 2 | [§I self-shadow rejection by identity](#2-i-self-shadow-rejection-by-identity) | Correctness | Designed, not built. The absolute epsilons are what couple the renderer to scene scale. |
| 3 | [Texture minification has no filter](#3-texture-minification-has-no-filter) | Quality | The largest remaining image-quality gap on the default path, and the one the analytic-AA design explicitly left open. |
| 4 | [§H nested-IOR refraction](#4-h-nested-ior-refraction) | Correctness | Designed, not built. Any nested glass renders with the wrong relative index. |
| 5 | [Decide what to do about unlit Bezier circuits](#5-decide-what-to-do-about-unlit-bezier-circuits) | Capability | Scoped decision, not a bug — but it is the capability gap users meet first. |
| 6 | [Four materials silently ignore most of the lighting rig](#6-four-materials-silently-ignore-most-of-the-lighting-rig) | Correctness | `MeshToonMaterial` and friends drop every extended light and all shadows without a word. |
| 7 | [Two public settings are no-ops; a whole path tracer is unreachable](#7-two-public-settings-are-no-ops-and-a-whole-path-tracer-is-unreachable) | API / dead code | `light_intensity` and `ambient_light` reach nothing. |
| 8 | [The shadowed resolve runs the resolve kernel twice](#8-the-shadowed-resolve-runs-the-resolve-kernel-twice) | Performance | Not in the optimization plan. Plausibly the largest untargeted render-thread item on shadowed scenes. |
| 9 | [`AttributeTimeline.get` — the prep pole](#9-attributetimelineget--the-prep-pole) | Performance | 20.3% of the reference render, never targeted. |
| 10 | [T5 — the sparse-discovery host chain](#10-t5--the-sparse-discovery-host-chain) | Performance | Largest render-thread item in the plan; half shipped. |
| 11 | [P9 / P10 — the batched geometry builds](#11-p9--p10--the-batched-geometry-builds) | Performance | Measured, not started. |
| 12 | [`empty_cache` always collects on a CPU render](#12-empty_cache-always-collects-on-a-cpu-render) | Performance | One-line gate; unconditional cost on the CPU path. |
| 13 | [Delete the dead render paths](#13-delete-the-dead-render-paths) | Maintenance | ~1,600 lines, two references to modules that do not exist. |
| 14 | [Stale docstrings that describe a renderer that no longer exists](#14-stale-docstrings-that-describe-a-renderer-that-no-longer-exists) | Docs | Each has already misled someone reading the code. |
| 15 | [Nine experimental toggles are unreachable from `SETTINGS`](#15-nine-experimental-toggles-are-unreachable-from-settings) | API | Includes a route precondition that cannot be flipped from Python. |
| 16 | [The CPU baseline debt](#16-the-cpu-baseline-debt) | Process | `DESIGN_mesh_identity_open.md` §B, still open, and now further behind. |
| 17 | [An untracked file on the default path reached master](#17-an-untracked-file-on-the-default-path-reached-master) | Process | Fixed at `2016a26`; the gap that allowed it is not. |
| 18 | [Open design-doc items with no owner](#18-open-design-doc-items-with-no-owner) | Various | §J, §L, §G, §4.6, P7 — recorded so they are not rediscovered. |

---

## 1. Silent truncations have no instrument

**Status: not built. Cheapest item here with a real payoff.**

Four ceilings in the render path degrade the image and report nothing. Three of
them are documented in the code as deliberate bounds; none is counted.

| Ceiling | Where | Value | What happens |
| --- | --- | --- | --- |
| Surfaces composited along one primary ray | `sheet_resolve_taichi.py:209, 809`; `wavefront_kernels_taichi.py:2901` | `MAX_SURFACES_PER_RAY = 256` | The walk stops; the ray's leftover weight is handed to the background. |
| Shadowed lights per fragment | `shading_taichi.py:99` | `MAX_SHADOW_LIGHTS = 16` | Lights past the cap are lit but never shadowed. |
| Overlapping layers of one surface in one pixel | `sheets.py:812` (`rank.clamp_(max=15)`) | 16 | Further layers merge into the last sub-band and attenuate once instead of per layer. |
| Continuation-pool reservation at `pool_ratio == 1` | `tracer.py:1760, 2550` | — | `overflow = pool_ratio > 1 and …` — at ratio 1 the pool's own overflow flag is **not read**, so a failed reservation is dropped silently. `_secondary_split_needed`'s docstring records this; nothing detects it. |

`DESIGN_mesh_identity_open.md` §Y already states the rule this violates: *"an
instrument that reports zero may not be looking."* The work is a per-batch
counter for each, reported the way the wavefront pool retries already are
(`logger.log(PERF, …)`), plus a `RenderPlan` field so a script can assert on it.

**Done when** a scene built to exceed each ceiling produces a log line naming
it, and `RenderPlan` carries the counts.

---

## 2. §I self-shadow rejection by identity

**Status: designed down to the argument list in
`DESIGN_mesh_identity_open.md` §I. Not started — verified: `raster_shadow_trace`
(`raster_taichi.py:2570`) still takes no `tri_obj` argument.**

A shadow ray rejects its own surface with `MIN_HIT_DISTANCE = 1e-4` plus a
normal offset of `10 * MIN_HIT_DISTANCE`. Both are absolute world-space
constants applied to *every* hit, so a small object resting on a plane loses its
contact shadow within 1e-3 of the contact, and grazing light on small geometry
produces acne.

This is the item that matters most beyond its own symptom: **it is the main
thing coupling the renderer to scene scale.** The four absolute constants
(`MIN_HIT_DISTANCE` 1e-4, `DEPTH_TIE_EPSILON` 1e-4, `TRIANGLE_EDGE_EPSILON`
2e-4, the 1e-3 shadow offset) all assume roughly unit-scale geometry. §I retires
the first of them for cross-mesh hits.

The design's cost estimate holds: one new ndarray on `raster_shadow_trace`, the
source id packed into the spare 28 bits of `event_msk`, and `(src_sid, tri_obj)`
threaded through five `@ti.func` signatures shared with the megakernel. Note the
warning in the design — reject "same mesh **and** near-zero `t`", never "same
mesh", or a concave solid stops shadowing itself.

---

## 3. Texture minification has no filter

**Status: named as the open residual by `DESIGN_analytic_aa.md` §19 ("what is
still untouched is texture minification (no mip chain)"). Not started.**

`_sample_tex_vec5` (`wavefront_kernels_taichi.py:244`) is a plain bilinear tap
with no level of detail. Combined with the sheet resolve's **one shade per
same-surface region per pixel**, a minified texture is point-sampled: the region
is shaded at its dominant fragment and whatever texel that lands on wins the
pixel. It aliases statically and crawls under camera motion.

Two things make this the top *quality* item rather than a nice-to-have:

* Everything else in the frame is antialiased exactly, so the texture is the
  only aliasing left and reads as a defect rather than as a resolution.
* `DESIGN_sheet_resolve.md` §10.5 anticipated it (the flip's own review named
  "the minified-texture `ImageMob`" as one of the populations that moved) and
  §4.7 records the remedy it *did not* take: area-weighted multi-sample shading
  per material.

Both remedies are open. A mip chain is the cheaper one at render time and costs
build time and texture memory; the sheet record already carries the exact area,
which is a screen-space footprint the LOD could be derived from without
derivatives. Measure before choosing.

---

## 4. §H nested-IOR refraction

**Status: designed in `DESIGN_mesh_identity_open.md` §H. Not started —
verified: `_alloc_wavefront_state` is still called with `sca_width = 7`
(`tracer.py:2476`) and columns 0-6 are fully spoken for, so there is no IOR
stack.**

Every interface assumes air outside. Glass in glass, a sphere in a box, a bubble
in a liquid: all take the wrong relative index at the inner interface. The
design's revision to carry a stack of **IORs** rather than mesh ids is the
cheap one — it costs no kernel argument (the stack rides in `rs_sca` columns 7+,
whose width is already a parameter) and `_refract_ray` does not change at all.

Rank it below §I because the population is smaller: nested transmissive media
are rarer in explanatory animation than contact shadows. Rank it above the
performance items because it is a wrong image rather than a slow one.

Gate the `rs_sca` width on the feature so `test_render_batch_sizing.py` and
`test_memory_model.py` only need re-checking when it is on.

---

## 5. Decide what to do about unlit Bezier circuits

**Status: working as built. A decision, not a defect — but the largest
capability asymmetry in the renderer, and it should be decided deliberately.**

`RayTracedBezierCircuitPrimitive.project_to_screen` (`primitives.py:2478`) never
calls `_shade_vertex_colors`, and the sheet resolve's per-fragment shading and
shadow-event build both sit inside `if not fetched_bez:`
(`sheet_resolve_taichi.py:333-409`). So every 2-D shape, `Text`, `Tex`, and
Manim vector mob:

* is never lit by any light,
* never receives a shadow,
* has no use for a normal map or material-property map,

while it *does* cast shadows, occlude, reflect and transmit. A mixed 2-D/3-D
scene therefore has flat labels sitting in a lit world, which is usually what
you want for a caption and never what you want for a logo lying on a lit table.

Three ways out, in increasing cost:

1. **Document and route users to `TriangulatedBezierCircuit`**, which already
   exists precisely for this ("what a shape needs when its interior has to carry
   per-fragment shading, a texture, or 3-D lighting"). This is done — the
   limitations page says so — and may be the whole answer.
2. **Give circuits a vertex-shaded arm**: call `_shade_vertex_colors` from the
   circuit primitive. Cheap, and gets point lights only. Probably not worth it.
3. **Give circuits a fragment-shaded arm in the sheet resolve**: they already
   arrive with a plane normal (`_bezier_normal`), a colour and material
   transport channels, so the missing pieces are a material block (circuits have
   no `tri_mat`/`tri_mat_id` equivalent) and a shadow-event build. That is a
   real feature, not a fix.

**The decision to record is which of these Algan wants**, because the docs
currently assert (1) as intent and the code offers no path to (3).

---

## 6. Four materials silently ignore most of the lighting rig

**Status: confirmed by reading; no test covers it.**

`_build_core_shader_ids` (`settings.py:2014`) registers exactly seven shaders.
`MeshToonMaterial`, `MeshNormalMaterial`, `MeshMatcapMaterial` and
`MeshDepthMaterial` are not among them, so:

* `_shaded_per_fragment` (`primitives.py:536`) is False → they bake at vertices;
* `_shade_vertex_colors` (`primitives.py:593`) **skips every light with
  `_render_aux`**, i.e. every light that is not a plain `PointLight`;
* `_shader_material_id` maps them to `_MID_UNLIT`, and the sheet resolve's
  shadow-event build refuses `pid_e == _MID_UNLIT`
  (`sheet_resolve_taichi.py:387`) → they never receive shadows.

So `Sphere().set_material(MeshToonMaterial())` under a `DirectionalLight` plus an
`AmbientLight` renders unlit-flat, with shadows on, and says nothing.

Cheapest honest fix: **warn at `set_material` time** when a non-core material is
combined with an extended light or with `shadows=True`, in the same style as the
existing "textures are not sampled" warning. The real fix is in-kernel ports of
the four shaders, which is a bigger job and probably only worth it for toon.

Note the second-order effect while you are there: because they shade at
vertices, their output resolution is the mesh's, so a toon band on a
`render_tolerance`-diced surface is faceted regardless of the dice (the dice
governs the *render* triangles; the vertex shade happens on the construction
grid).

---

## 7. Two public settings are no-ops, and a whole path tracer is unreachable

**Status: confirmed by exhaustive grep.**

`SETTINGS.raytracing.light_intensity` and `SETTINGS.raytracing.ambient_light`
are in `_PUBLIC_FIELDS` (`settings/raytracing_settings.py:116`), have public
setters, and are documented as "physical mode" controls. Their only consumer is
`path_trace_physical_stbvh` (`raytrace_kernels_taichi.py:3091`), a ~370-line
Monte Carlo kernel with next-event estimation and explicit light sampling —
**which no code path launches.** `render_batch_raytraced` dispatches only
`path_trace_scene_stbvh` for `samples > 1`. The only caller of the physical
kernel in the repository is `tests/unit_tests/test_raytracing_unit.py:255`.

So there is a shipped, tested, compiled and unreachable renderer, and two public
settings that cannot affect a frame.

Decide one of:

* **Wire it up** — give `samples_per_pixel > 1` a way to select it (it is the
  better path tracer of the two: NEE instead of naive scattering), and the two
  settings become real; or
* **Delete it** and deprecate the settings, keeping the unit test as the record
  of what was measured.

Either is fine. Leaving a public setting that provably does nothing is not.

`INDIRECT_BOUNCE_STRENGTH` is genuinely read, but only by the Monte Carlo path
(`tracer.py:1440`); that is correct and documented.

---

## 8. The shadowed resolve runs the resolve kernel twice

**Status: new. Not in `DESIGN_optimization_targets.md`. Measurement needed —
this container is CPU-only and cannot rank it.**

`shade_sparse_raster_coverage` (`raster_pipeline.py:1994-2094`) launches
`sheet_resolve_shade` **twice** on any shadowed batch: `mode = 1` to build the
shadow events, then `mode = 2` to shade reading the traced visibility.

That is the right architecture — it is what makes a resolve/shadow desync
structurally impossible, and `DESIGN_sheet_resolve.md` Phase 4a is explicit that
it replaced a hand-maintained lockstep pair. But the two passes are not cheap
and cheap: mode 1 runs the *entire* transport walk, including

* the per-sheet coverage/`corr`/one-mesh/band arithmetic,
* `_tri_color_g`, `_tri_extra_g`, `_tri_ior_transmission_g` — every texture
  fetch,
* `_tri_shadow_normals` and `_pixel_footprint`,

and discards all of it except the event tables. Only `_shade_tri_hit` and the
`_spawn_pool_ray` calls are compiled out.

Two things to try, in order:

1. **Hoist what mode 2 recomputes.** The event pass already writes per-sheet
   tables; the material fetches it performs are exactly the ones mode 2 repeats.
   Widening the event tables to carry them trades bandwidth for arithmetic.
2. **Or cut mode 1 down.** It needs position, two normals, the sub-pixel mask,
   the pipeline id and the footprint — the colour and transport fetches are dead
   in it. `_tri_color_g` is called before the material-id test, so it cannot be
   gated by `ti.static(mode)` as written; restructuring so the `mode == 1` arm
   fetches only what it writes should be a compile-time win with no output
   change.

Both are byte-identical by construction, which makes this measurable with the
project's existing A/B discipline. On the reference profile shadows are named as
cost item 4 ("multiplied by the number of lights"); this doubles the *resolve*
on top of that, and nothing has measured it.

---

## 9. `AttributeTimeline.get` — the prep pole

**`DESIGN_optimization_targets.md`, "What is left, in order", item 1. Unchanged
by this audit; repeated here so the ranking is complete.**

72.58 s (20.3%) of the reference render, 542,052 calls at ~134 us. It reached
the top by attrition: three rounds of work landed elsewhere. The lever is
**fewer calls**, and items 2 and 3 of that document (P9, P10) are both concrete
ways of removing them, so measure the three together. Re-measure the
`get/full` vs `get/replay` split first — the "two thirds in the geometry build"
figure predates P8, which changed the denominator.

---

## 10. T5 — the sparse-discovery host chain

**`DESIGN_optimization_targets.md` T5. Half shipped; the shipped half is the one
that paid.**

The compaction's per-sample-lane reductions are kernels now
(`sheet_compact_taichi.py`, `SHEET_MASK_KERNEL` default on, bit-identical,
measured 1.25-1.33x on `compact_sheets`). The six-array gather T5 originally
proposed is built and bit-identical too but ships **default off**: worth ~4 ms
of a 1.3 s 4K frame against 50-160 MB of peak.

What remains, in `sheets.compact_sheets`:

* **The conflict-rank scan: eight `torch.cumsum` passes over `[n]`**, plus an
  `index_select` and a `maximum` per lane. `DESIGN_sheet_resolve.md` §10.4 names
  this as the genuine remaining scan. It is the natural next kernel — the mask
  kernel beside it already proves the shape works.
* **The sorts.** T5's own advice is to leave them alone: `_lexsort` is three
  stable `argsort`s and there are two `torch.unique` calls after it. Radix sorts
  at these sizes are not the bottleneck the scans are.

Do the cumsum scan; leave the sorts.

---

## 11. P9 / P10 — the batched geometry builds

**`DESIGN_optimization_targets.md` items 2 and 3. Measured, not started.**

* **P9** — the batched bezier build reaches only 18.4% of the reference scene's
  circuits, and **51.5% are reverted by an all-or-nothing group clash** the code
  calls "rare". Batched is ~5x cheaper per circuit; the per-actor build is
  40.97 s (11.4%) of own time, each with its own accessor round trips, so this
  cuts item 9 as well.
* **P10 remainder** — 56.62 s (15.8%) after P11 halved
  `compute_grid_vertex_normals`. What is left: the per-surface tail (colours,
  shader parameters, primitive construction, still one surface at a time), the
  rest of `compute_grid_vertex_normals` (seam merge, pole fans, final
  normalize), and `grid_to_triangle_vertices` on the whole stack — "two gathers
  sharing one permutation", the same shape T5's gather had.

Re-split P10 before choosing inside it: its proportions were measured *before*
P11.

---

## 12. `empty_cache` always collects on a CPU render

**Status: new. One-line change; measure before and after.**

`_gpu_memory_pressure()` (`utils/memory_utils.py:90`) returns **True** when CUDA
is unavailable — "No CUDA telemetry; keep the original (always-gc) behavior."
`empty_cache()` gates its `gc.collect()` on that, so on a CPU (or MPS) render
**every call runs a full collection**, several times per frame batch, on the
scene's whole object graph.

The docstring for `empty_cache` puts `gc.collect()` at ~0.2 s on a large scene
and records it costing ~40% of a small render before it was gated. That gating
is exactly what a CPU render does not get. `scene_excluded_from_gc` softens it,
but the collection still walks everything else.

The CPU path is not the reference workload, but it is what CI runs, what a
cloud session runs, and what a laptop without CUDA runs. A cheap host-side
pressure proxy (or simply "no telemetry → do not force") is worth measuring.

Related, and to be kept honest about: on the reference CUDA machine the same
gate is open *always* for the opposite reason — a 4 GB card sits above the 80%
threshold for the whole render, 510 calls at ~74 ms
(`DESIGN_optimization_targets.md`, "`memory reclaim` doubled in share"). Both
ends of the gate are wrong for their machine. That document's advice stands:
measure on a card with headroom before spending anything on the CUDA end.

---

## 13. Delete the dead render paths

**Status: ~1,600 lines, and two of them import modules that do not exist.**

An import audit over the package (excluding `external_libraries`) found six
unresolvable intra-package imports. One was the missing compaction kernel, fixed
at `2016a26`. The rest are dead:

| Reference | Module | Guarded? |
| --- | --- | --- |
| `tracer.py:2867` | `wavefront_textured_kernels_taichi` | Yes — `set_textured_wavefront(True)` raises, `WF_TEXTURED` is a hard `False`, and `_validate_render_capabilities` raises on direct mutation. |
| `tracer.py:3180` | `wavefront_sorted_kernels_taichi` | Yes — same pattern via `set_material_sorting`. |
| `profiling_utils.py:694` | `ray_trace_taichi` | Yes — wrapped in `try/except Exception: pass`. |

What that costs to keep:

* `_raytrace_render_wavefront_textured` — 232 lines (`tracer.py:2830-3061`).
* `_raytrace_render_wavefront_sorted` — 360 lines (`tracer.py:3130-3489`).
* `_build_textured_scene` and its call site (`scene_builder.py:1072`, ~120
  lines), reachable only when `WF_TEXTURED`, which cannot be set.
* `path_trace_physical_stbvh` — ~370 lines (item 7).
* `WF_TEXTURED`, `WF_TEXTURED_FEATURES`, `WF_TEX_*`, `WAVEFRONT_SORT_MATERIALS`
  and their guarded setters, plus every `merged.get("textured_active")` and
  `WAVEFRONT_SORT_MATERIALS is True` branch scattered through the live route
  decisions — including two inside `analytic_raster_route_active`, the single
  most load-bearing function in the renderer.
* `post_processing/anti_aliasing/smaa.py` — 532 lines, plus `AreaTex.png` and
  `SearchTex.png`. **Never imported by anything**; `post_process_frames` wires
  up FXAA only. The module's own `__init__.py` advertises "anti-aliasing" as a
  built-in pass.
* `bloom.py` — three unused implementations beside the live one:
  `bloom_filter_old` (with a `# TODO fix up this code`, ~90 lines of
  commented-out experiments, and statements whose results are discarded:
  `xb[..., -1:] + k`), `bloom_filter_premultiply`, and `bloom_filter_conv`
  (which has ~20 lines of unreachable code after its `return`).
* `is_ray_tracing_enabled()` (`tracer.py:3490`) — "Vestigial: always False",
  kept only because `post_processing.bloom` probes for it. And that probe is
  itself dead: `_should_bypass_bloom` imports `is_raytraced_glow_enabled`, which
  **does not exist anywhere in the package**, so the function raises
  `ImportError` internally and returns `False` on every call.
* `_scene_has_user_pipeline` (`settings.py:2138`) iterates
  `("tri_mat_id", "pn_mat_id")`. There is no `pn_mat_id` key: PN patches are
  diced to flat triangles before the merge and there are exactly two geometry
  types in `merged`.

This is a cleanup, not a fix, so it ranks below everything that changes an
image. It ranks this high because the live route decisions are threaded through
the dead ones, and that is where the next subtle routing bug will come from.

---

## 14. Stale docstrings that describe a renderer that no longer exists

Each of these is a load-bearing docstring that is now false. Listed with what
the code actually does.

| Location | Says | Actually |
| --- | --- | --- |
| `shaders/materials.py:26-33` (module docstring) | "Algan shades per vertex and has no UV / image-sampling pipeline, so every texture / image-based property … is not sampled" | `FRAGMENT_SHADING` defaults **on**, and there is a full UV/texture pipeline: colour, material-property and normal maps are sampled per fragment. Only the *`Material` class's* map slots are ignored — the maps are set on the geometry instead. The sentence is right about the outcome and wrong about the reason, which is why it reads as "Algan has no textures". |
| `mobs/surfaces/surface.py:741-744` (`normal_texture`) | "under the default vertex-shaded pipeline lighting is baked at the vertices, so a normal map only affects effects evaluated per fragment" | The default pipeline is fragment-shaded. Normal maps affect direct lighting. |
| `CLAUDE.md`, Architecture | "`Camera` (`camera.py`): perspective/orthographic projection" | True orthographic is **not implemented**; `set_to_orthographic()` warns and substitutes a perspective camera at distance 1e5 (`camera.py:173-195`). |
| `settings.py:219` (`FRAG_PID_GATE`) | "built at merge time as `{tri,pn}_material_ids`" | No PN geometry type exists at merge time. |
| `settings.py:648` (`HYBRID_RASTER`) | "PN geometry is preserved and simply falls back to classic primary traversal" | PN patches are diced to flat triangles by `logical_pn.py` before the merge; there is nothing for the raster path to fall back from. The same claim appears at `tracer.py:2082`. |
| `settings.py:1849` (`SHADOWS`) | "Implies per-fragment shading … and forces the general kernel" | Correct for triangles; it should say that Bezier circuits neither receive shadows nor are fragment-shaded, since the same block is the reference for what shadows do. |
| `raster_taichi.py:199-200` (`_AA_SAMPLES`) | "16 matches the sampling density of the anti_alias_level=4 reference" | The live pattern is `_AA_PATTERN_8`; `_AA_NUM_SAMPLES` is 8. The comment reads as though 16 were selected. |
| `CLAUDE.md`, "Cloud sessions" | The full-render baselines are per machine because "`pn_criterion_kernel` runs under Taichi's `fast_math` and which tessellation levels sit on the borderline depends on the CPU evaluating the criterion" | `pn_criterion_kernel_active()` is `PN_CRITERION_KERNEL and project_on_gpu_active()`, and `project_on_gpu_active()` requires `_RENDER_DEVICE.type == "cuda"`. **On a CPU render device the criterion kernels never run** (verified in this container: `pn_criterion_kernel_active()` is `False`), so `fast_math` cannot be the mechanism for the measured CPU-to-CPU divergence. The *observation* stands — 5 of 6 scenes missed by 29-204 channel values on a GitHub runner, and what moved carried PN surfaces — so something else in the torch criterion path is machine-sensitive. Worth attributing properly, because the paragraph is the reason the suite skips itself in CI. |

---

## 15. Nine experimental toggles are unreachable from `SETTINGS`

`_FIELD_TO_LEGACY` (`settings/raytracing_settings.py:30`) enumerates 76 fields.
These renderer globals are **not** among them, so they can only be set through
their environment variables — i.e. before `import algan`, and never A/B'd
in-process:

`ANALYTIC_AA_EXACT`, `ANALYTIC_AA_BEZ_WEDGE`, `ANALYTIC_AA_RUN`,
`ANALYTIC_AA_RUN_RULE`, `SHADOW_ANYHIT`, `OPAQUE_BVH_SKIP_DEAD`,
`MERGE_DEDUP_TIME`, `FRAG_PID_GATE`, `RASTER_STRADDLE_CLIP`.

Two of them matter more than the rest:

* **`ANALYTIC_AA_RUN` is a route precondition.** `analytic_raster_route_active`
  (`tracer.py:409`) vetoes the sheet route on it, so it decides which renderer a
  batch takes — and there is no way to flip it from Python. `set_analytic_aa`
  accepts a `run=` keyword, but `_SETTER_OVERRIDES` calls it as
  `set_analytic_aa(value)`, so only the master toggle is reachable.
* **`SHADOW_ANYHIT`** has three modes (`False` / `True` / `"gather"`) and a
  documented non-byte-identical corner; it is exactly the sort of thing the
  experimental section exists to expose.

The fix is mechanical: add the names to `_FIELD_TO_LEGACY`, and give
`analytic_aa` a setter override that forwards its sub-options.

While there: `CLAUDE.md` says "~55 kernel/perf switches"; the real count is
`76 - 12 = 64`.

---

## 16. The CPU baseline debt

**`DESIGN_mesh_identity_open.md` §B, open, and further behind than when it was
written.**

`tests/*/expected_outputs_cpu/` was last regenerated at `28efe67`. Eight commits
have touched `algan/rendering` since, at least one of which
(`c293da3`, "Fixed bug in AAA which caused speckling artefacts") moved output
and shipped a **CUDA** rebaseline without a CPU one.

This matters more than the design doc's framing suggests, because **CI renders
on CPU**: `tests/fast` compares against `expected_outputs_cpu/`, and
`test_full_render_scene` skips itself when `CI` is set. So the CPU fast baseline
is the only pixel gate CI has, and it is the one drifting.

The two traps §B records are still traps: `CUDA_VISIBLE_DEVICES=` (empty) does
not hide the GPU on Windows — use `-1`; and the render suites pick their
baseline directory from `torch.cuda.is_available()`, not from the render device,
so a CPU render that can still see a GPU compares against the CUDA set and
passes.

---

## 17. An untracked file on the default path reached master

**Fixed at `2016a26`. The gap that allowed it is not.**

`1e90c87` ("Optimized memory usage of sheet resolve…") added references to
`algan.rendering.raytracing.sheet_compact_taichi` from `sheets.py` (twice),
`raster_pipeline.py` and `settings.py`, and turned `SHEET_MASK_KERNEL` on by
default — without committing the module. At `2e3264b` the repository could not
render at all on default settings: every `save_video` raised
`ModuleNotFoundError` inside `compact_sheets`.

Measured in this container at `2e3264b`, before the fix:

* `pytest -q tests/unit_tests` — **29 failed, 1372 passed, 92 skipped**.
* `pytest -q --fast` — **1 failed, 317 passed**; the one failure was
  `test_the_fast_scene_renders_and_matches_its_baseline`, the fast suite's only
  test that renders.
* The same `pytest -q tests/unit_tests` with `ALGAN_SHEET_MASK_KERNEL=0` —
  **1401 passed, 92 skipped, 0 failed**, which is what says all 29 traced to
  that one import and nothing else was wrong.

All of it was invisible locally, because the file was on the author's disk.

`scripts/run_ci.py` (added one commit earlier) would not have caught it either,
for the same reason.

The gap is narrow and the guard is cheap: **fail if `git status --porcelain`
reports an untracked `.py` under `algan/`** — as a pre-push hook, as a step in
`run_ci.py`, or as a unit test in the family of
`tests/unit_tests/test_environment.py` (which already enforces a
whole-package invariant by walking the source). The same test can assert that
every intra-package import resolves; the ~20-line scan used in this audit found
all six dangling modules in under a second.

---

## 18. Open design-doc items with no owner

Recorded so they are not rediscovered from scratch. None is started.

* **`DESIGN_mesh_identity_open.md` §J — the MD-resolution noise floor.** A scene
  of six translucent sheets plus a glass sphere measured a run-to-run floor of
  46 channel values over 212,210 pixels at `--res md`, unexplained and far above
  the |d| = 1 cap of known split-pixel nondeterminism. `DESIGN_sheet_resolve.md`
  Phase 4 later reports the same scene at **zero** on the sheet route at LD and
  MD. Those two readings are compatible (the floor was the fragment walk's) but
  nobody has said so explicitly. **Re-run `_order_window_check.py --res md` on
  the current default route and close or reopen §J on the result** — it is a
  half-hour of machine time and it retires a standing unknown.
* **P7 — deterministic continuation slots.** Deferred by measurement in
  `DESIGN_sheet_resolve.md` Phase 4: the criterion (run-to-run zero) was met
  without it. `rs_alloc`'s atomic slot allocation and the bounce loop's
  `pix_accum` atomic adds remain the resolve's only atomics, and a pixel with
  several continuations sums them in nondeterministic order. If §J's re-run
  shows a floor, P7's shape is recorded (count-pass template + host int-scan +
  exact-slot emit).
* **`DESIGN_sheet_resolve.md` §4.6 — K and overflow.** Never built, and now
  moot: the compaction is a host CSR with no per-pixel sheet cap, so there is no
  K to overflow. The real ceiling is `MAX_SURFACES_PER_RAY`, which is item 1's
  business. **Amend §4.6 to say so** rather than leaving a design describing a
  structure the code does not have.
* **`DESIGN_mesh_identity_open.md` §L — coincident duplicates.** Unbuilt, and
  the symptom has never been demonstrated. Its own instruction stands: render
  two coincident quads of the same mesh and of different meshes and measure the
  darkening before writing any code. It may close with a test.
* **`DESIGN_mesh_identity_open.md` §G — two-level BVH (TLAS/BLAS).**
  Deliberately deferred: the BVH build is ~1% of a shadowed five-solid render
  and no workload in the repo has thousands of repeated meshes. Revisit only
  when one does.
* **`DESIGN_analytic_aa.md` §20 / `GLOSSY_REFLECTION`.** Off by default because
  four taps cannot integrate a wide lobe — with the screen-space rotation on it
  dithers and crawls, with it off it ghosts. Neither `ANALYTIC_AA_SECONDARY = 8`
  nor turning the rotation off fixes it. A real fix needs more taps than the
  coverage budget can pay for, which makes it a path-tracer feature; record it
  as such rather than as a tuning problem.

---

## Two things the audit checked and found sound

Worth stating, because both were candidates for this list and neither earned a
place.

* **The route decision is single-sourced.** `analytic_raster_route_active` is
  computed once, drives AA-level planning, the frame-buffer prefill, the
  wavefront route and the emission's compaction mode, and the three
  disagreement points (`tracer.py:2101`, `raster_pipeline.py:1338`,
  `raster_pipeline.py:1897`) all **raise** rather than fall back. That is the
  discipline `DESIGN_mesh_identity_open.md` §Y.4 asks for, and it is being kept.
* **The memory model measures rather than models.** `memory_model.py` fits
  `peak(n) = a + b*n` to the arena's own high-water mark, so a new primitive or
  a user post-process is accounted for by running it. The OOM retry is retained
  as the backstop for the one case it cannot see (a batch that densifies later).
  Do not add byte formulas back.
