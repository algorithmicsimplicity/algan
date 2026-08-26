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

> **Before anything else: CI is red on `master` right now.**
> `tests/fast`'s pixel comparison — the only pixel gate CI has, because
> `tests/full_renders` skips itself when `CI` is set — fails by **40 channel
> values against a tolerance of 2**, and has been failing since `1e90c87`. It is
> a stale baseline rather than a broken renderer, and the evidence is in
> [item 17](#17-the-cpu-baseline-debt). Fixing it is a rebaseline plus a look at
> the frames, and it should happen before any of the work below, because until
> it does no change to the renderer can be told from the drift already there.

## Ranking

| # | Item | Kind | Why here |
| --- | --- | --- | --- |
| 1 | ~~Silent truncations have no instrument~~ | Correctness | **Done.** All four ceilings are counted, warn once per render, and ride on `RenderPlan.truncations`. |
| 2 | [The verification harnesses the docs and the source name do not exist](#2-the-verification-harnesses-the-docs-and-the-source-name-do-not-exist) | Process | 56 missing, 24 of them cited from inside `algan/`, including the stated gate for eight default-on toggles. Every item below is harder without them. |
| 3 | [§I self-shadow rejection by identity](#3-i-self-shadow-rejection-by-identity) | Correctness | **Built, default on.**       |
| 4 | [Texture minification has no filter](#4-texture-minification-has-no-filter) | Quality | The largest remaining image-quality gap on the default path, and the one the analytic-AA design explicitly left open.                         |
| 5 | [§H nested-IOR refraction](#5-h-nested-ior-refraction) | Correctness | **Done, default on.** Nested media take the correct relative index; only a batch that refracts pays the wider ray state. |
| 6 | [Decide what to do about unlit Bezier circuits](#6-decide-what-to-do-about-unlit-bezier-circuits) | Capability | Scoped decision, not a bug — but it is the capability gap users meet first.                                                                   |
| 7 | [Four materials silently ignore most of the lighting rig](#7-four-materials-silently-ignore-most-of-the-lighting-rig) | Correctness | `MeshToonMaterial` and friends drop every extended light and all shadows. **No longer without a word** — they warn now; the in-kernel ports are still open. |
| 8 | [Two public settings are no-ops; a whole path tracer is unreachable](#8-two-public-settings-are-no-ops-and-a-whole-path-tracer-is-unreachable) | API / dead code | `light_intensity` and `ambient_light` reach nothing.                                                                                          |
| 9 | [The shadowed resolve runs the resolve kernel twice](#9-the-shadowed-resolve-runs-the-resolve-kernel-twice) | Performance | Not in the optimization plan, and never separated out from "shadows are expensive". Measure before building.                                  |
| 10 | [`AttributeTimeline.get` — the prep pole](#10-attributetimelineget--the-prep-pole) | Performance | 20.3% of the reference render, never targeted.                                                                                                |
| 11 | [T5 — the sparse-discovery host chain](#11-t5--the-sparse-discovery-host-chain) | Performance | Largest render-thread item in the plan; the host loops are shipped, the sorts stay. |
| 12 | [P9 / P10 — the batched geometry builds](#12-p9--p10--the-batched-geometry-builds) | Performance | **P9 shipped; P10 re-split and one piece shipped.** The re-split moved the ranking — what this item named as P10's remainder is mostly not where the time is. |
| 13 | [`empty_cache` always collects on a CPU render](#13-empty_cache-always-collects-on-a-cpu-render) | Performance | One-line gate; unconditional cost on the CPU path.                                                                                            |
| 14 | [Delete the dead render paths](#14-delete-the-dead-render-paths) | Maintenance | ~1,600 lines, two references to modules that do not exist.                                                                                    |
| 15 | [Stale docstrings that describe a renderer that no longer exists](#15-stale-docstrings-that-describe-a-renderer-that-no-longer-exists) | Docs | Each has already misled someone reading the code.                                                                                             |
| 16 | [Nine experimental toggles are unreachable from `SETTINGS`](#16-nine-experimental-toggles-are-unreachable-from-settings) | API | Includes a route precondition that cannot be flipped from Python.                                                                             |
| 17 | [The CPU baseline debt](#17-the-cpu-baseline-debt) | Process | **Do this first.** `DESIGN_mesh_identity_open.md` §B, and it is now a red CI on master.                                                       |
| 18 | [An untracked file on the default path reached master](#18-an-untracked-file-on-the-default-path-reached-master) | Process | Fixed at `2016a26`; the gap that allowed it is not.                                                                                           |
| 19 | [Open design-doc items with no owner](#19-open-design-doc-items-with-no-owner) | Various | §J, §L, §G, §4.6, P7 — recorded so they are not rediscovered.                                                                                 |
| 20 | [The shadow terminator on diced surfaces](#20-the-shadow-terminator-on-diced-surfaces) | Correctness | **Built, default on.** Hanika's offset onto the smooth surface; a lit torus goes from 41 speckle pixels to 4. Flat geometry is byte-identical by construction. |

---

## 1. Done

Built in `algan/rendering/raytracing/truncation.py`. All four ceilings are
counted per batch, accumulated over the render job, reported once each at
**`WARNING`** — not `PERF`, which is for the budget events (batch splits, pool
retries) that are the memory model working as designed; a truncation moves the
image — and carried on `RenderPlan.truncations` as a `TruncationCounts`
(exported from `algan`). Later batches escalate a growing total at `PERF`
rather than repeating the warning, and a batch that adds nothing says nothing.

Where each is detected:

| Ceiling | Detected | Verified by |
| --- | --- | --- |
| `MAX_SURFACES_PER_RAY` | In-kernel, `sheet_resolve_taichi.py` and `wavefront_shade`, into a new `rs_alloc` word | 300 stacked quads at opacity 0.002 (thin enough that `MIN_WEIGHT` does not stop the walk first): 16,856 rays truncated, warned. |
| `MAX_SHADOW_LIGHTS` | Host, in `render_batch_raytraced` where `num_lights` and `shadow_flag` are both known | 21 light slots with shadows on: reports 5. |
| Sheet conflict rank (16) | Host, `sheets.compact_sheets`, an `amax` before the clamp with the `[n]` count only in the case being reported | One `Polyhedron` of 24 stacked quads: 3,320 fragments, warned. |
| Continuation-pool reservation at `pool_ratio == 1` | Host, `_read_tile_alloc` now reads the overflow word on **every** tile | **Structurally unreachable — see below.** |

Two things worth carrying forward:

* **The ratio-1 drop is currently closed, and the counter says so.** Every
  kernel branch that reserves a pool slot is compiled in only under
  `ti.static(refraction != 0)` or is reached only by reflective geometry on the
  sheet route; and every condition that sets `refraction_flag` — refractive,
  reflective-transparent, custom scatter, `_secondary_split_needed` — also
  drives `pool_ratio` above 1, where the host discards and retries the tile.
  No scene reaches it. The counter stays because that agreement is between the
  host's `merged` flags and the kernel's runtime tests, and the day they stop
  agreeing is exactly the day nobody would notice. It reads zero, and the zero
  is now a measurement.
* **`save_frame` never filled its `render_plan`.** Documented on
  `RenderResult` from the start, filled only by the video path. Fixed here,
  since half the output surface could not otherwise report its counts.

Guarded by `tests/unit_tests/test_render_truncations.py` (unmarked: it only
breaks when the instrument does).

**Rendered output does not move.** Kernel-side counting is one `ti.atomic_add`
inside a branch that was already there, and the store it sits beside is
unchanged. Measured rather than asserted, by running each render suite on this
branch and on `ecd6947` in the same container:

| Suite | Base `ecd6947` | With the instrument |
| --- | --- | --- |
| `tests/fast` | 32 @ frame 6 | 32 @ frame 6 |
| `full_renders/complex_hierarchy_become` | 189 @ frame 3 | 189 @ frame 3 |
| `full_renders/manim_compat_and_plots` | 190 @ frame 123 | 190 @ frame 123 |
| `full_renders/materials_and_lighting` | 221 @ frame 36 | 221 @ frame 36 |
| `full_renders/shapes_and_timeline` | 198 @ frame 179 | 198 @ frame 179 |
| `full_renders/solids_and_camera` | 205 @ frame 67 | 205 @ frame 67 |
| `full_renders/text_and_media` | 207 @ frame 150 | 207 @ frame 150 |

Every one of those is a *failure* against a tolerance of 2, and every one of
them fails identically without this change — the fast row is item 17's baseline
debt, and the six full-render rows are the per-machine baselines this container
was never going to match (`CLAUDE.md`, "Cloud sessions"). No baseline was
regenerated. What the pairing buys is the parity statement: identical worst
channel *and* identical worst frame on seven scenes covering PN surfaces,
shadows, refraction, glTF and `Text` — the paths `tests/fast` does not reach.

The one added host cost is a single `rs_alloc.tolist()` per accepted tile, where
the ray compactor already forces a device synchronisation per wavefront
iteration *inside* the tile.

---

## 2. Done

## 3. §I self-shadow rejection by identity

**Status: BUILT, behind `SHADOW_IDENTITY_REJECT` (default on). Extended past
§I as designed — see "what shipped" below.**

**What shipped.** The acceptance floor is now chosen per hit from three tiers:
the ray's own triangle keeps `eps_self`, another triangle of the same mesh
keeps `eps_near` (`SHADOW_NEAR_FRACTION`, default 0 — primitive-precise), and
any other mesh gets 0. Both epsilons are proportional to the batch's scene
scale (`SHADOW_EPS_RELATIVE`, default 1e-5), which is what actually retires the
absolute constant rather than merely bypassing it: 1e-4 is only ever right for
a scene about ten units across, and the default reproduces it exactly there.

Two corrections to what this item claimed, both found while building it:

* **§I as written could not do half of it.** The item named both lost contact
  shadows and grazing-light acne. §I relaxes only *cross-mesh* hits, and acne
  is a mesh shadowing itself, which by design keeps its floor. The acne claim
  traces to a bullet in `DESIGN_mesh_identity.md:1994` that is internally
  inconsistent — "reject its own mesh at near-zero `t`" *is* keeping the
  epsilon for exactly the population acne is made of. Acne needs the shadow
  terminator fix (offset onto the smooth surface implied by the vertex
  normals, both normals already on the event), not an identity test. That is
  now [item 20](#20-the-shadow-terminator-on-diced-surfaces).
* **§I's plumbing costing was stale.** It names `raster_shadow_event_build`,
  deleted on 2026-08-19 by the sheet-resolve flip, and claims `event_msk` has
  28 free bits when bits 8+ carry the material pipeline id. The source
  triangle rides its own array instead.

**Verified** on CPU: default path byte-identical (`--fast` 271 passed, the
item 17 baseline failure unchanged at exactly 40 channel values / frame 6);
and, on a torus — one mesh, concave, so it genuinely self-shadows — the
feature moves real pixels and both knobs behave as specified. With
`SHADOW_NEAR_FRACTION=1` the output tracks `SHADOW_EPS_RELATIVE` (2,174 →
3,448 pixels moved as it goes 1e-5 → 5e-2); with it at 0 the output is
*completely insensitive* to it (2,273 pixels at every value), which is exactly
right because the same-mesh floor is then 0 regardless. No acne was introduced
at the default: the change is a coherent band over the ring's self-shadowed
region, not scattered speckle.

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

## 4. Texture minification has no filter

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

## 5. §H nested-IOR refraction

**Status: built, and now DEFAULT ON** (see `DESIGN_mesh_identity_open.md` §H
for the four deliberate deviations from the design text: the overflow rule,
Fresnel left on the material index, custom-scatter scenes excluded, and the
per-corner-IOR approximation). Turn it off with
`SETTINGS.raytracing.experimental.set(nested_ior=False)`.
`benchmarks/_nested_ior_ab.py` is the four-frame check: a nested pair that must
move, a transmissive pane that must be byte-identical, and two single solids
bounded to the edge/silhouette band the design doc accounts for.
`tests/unit_tests/test_nested_ior.py` pins the stack arithmetic itself.

**What the default costs, and who pays it — read `refraction_flag` before
answering.** `tracer`'s `ior_stack_flag` is `nested_ior_mode() != 0 and
refraction_flag`, and the obvious reading of that ("only scenes with glass")
is wrong: `refraction_flag` is also set by a *reflective* primitive under
analytic AA, through `_secondary_split_needed`, which is what gives a mirror
the split pool it needs — and every PBR triangle is reflective. So an ordinary
`MeshStandardMaterial` scene with no transmission anywhere takes the wider
`rs_sca` (5 extra f32 per ray: `IOR_STACK_DEPTH` entries plus the depth
counter — `settings.py`'s own comment said 4 while the gate was off and nobody
was paying it) and compiles the stack's kernel variants.

What such a scene does *not* get is different pixels, and that is measured
rather than argued: with nothing transmissive no transmitted child is ever
spawned, so nothing pushes or pops the stack. `tests/fast` — itself a
`MeshStandardMaterial` scene, hence a widened one — renders byte-identically
with the gate off and on, as do five of the six `tests/full_renders` scenes.
Only `materials_and_lighting`, the one scene carrying transmission, moves: 49
of 179 frames, worst 42 channel values, ~18 pixels a frame. Of the pixel suites, only `tests/full_renders`' `materials_and_lighting`
carries transmission at all.

Worth carrying forward into [item 3](#3-i-self-shadow-rejection-by-identity):
building this surfaced an epsilon artifact that §I would fix at its source. A
ray grazing a shared edge can be classified as ENTERING a convex solid it
never left, which is how a single cube reaches stack depth 2. §I's mesh
identity is exactly what would let the stack refuse that.

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

## 6. Decide what to do about unlit Bezier circuits

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

## 7. Four materials silently ignore most of the lighting rig

**Status: the silence is fixed. The warning ships;
`tests/unit_tests/test_materials.py` covers it. The in-kernel ports are not
done and are still the real fix — see the bottom of this item.**

`set_material` now warns when a material Algan can only bake into vertex
colours meets a rig that asks for more than the bake delivers (any light beyond
a plain `PointLight`, `shadows=True`, an environment map), naming the material
and each thing being dropped. The check keys on the shader rather than on a
list of four class names, so a custom per-vertex `set_shader` — which loses
exactly the same things — is covered by construction. Because the usual
authoring order chooses the material *before* the lights are spawned, every
render re-runs the same check over the whole scene from `_get_frames_impl`
(one attribute read per actor, so both `save_video` and `save_frame` pay it).
It is a warning and not `report_unsupported_features`: that policy defaults to
raising, and these materials render fine — what they drop is part of the
lighting rig, not the render. `docs/.../renderer_limitations.rst` and the
shaders tutorial say the same thing.

The original finding, kept because the in-kernel ports are still open:

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
`AmbientLight` renders unlit-flat, with shadows on — which it now says, and
which is all the warning buys. **The real fix is in-kernel ports of the four
shaders**, a bigger job and probably only worth it for toon. Nothing about the
rendered frame has changed.

Note the second-order effect while you are there: because they shade at
vertices, their output resolution is the mesh's, so a toon band on a
`render_tolerance`-diced surface is faceted regardless of the dice (the dice
governs the *render* triangles; the vertex shade happens on the construction
grid).

---

## 8. Two public settings are no-ops, and a whole path tracer is unreachable

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

**A third public setting reaches only half of what it names.**
`glossy_reflection` is in `_PUBLIC_FIELDS`, but `_glossy_reflect` is called
**only from `sheet_resolve_taichi.py`**, and `glossy_reflection_mode()` is
passed down **only from `raster_pipeline.py`**. So on the supersampled fallback
— which is a deterministic render, reached by anything in the fallback table of
the limitations page — turning glossy reflections on does nothing at all, and
`_mirror_share` is what governs a rough reflector there. Either plumb the mode
into `wavefront_shade` or say so in the setter's docstring; silently applying to
one of two paths is the worst of the three options.

---

## 9. The shadowed resolve runs the resolve kernel twice

**Status: new. Not in `DESIGN_optimization_targets.md`. Measurement needed —
this container is CPU-only and cannot rank it.**

`shade_sparse_raster_coverage` (`raster_pipeline.py:1994-2094`) launches
`sheet_resolve_shade` **twice** on any shadowed batch: `mode = 1` to build the
shadow events, then `mode = 2` to shade reading the traced visibility.

That is the right architecture — it is what makes a resolve/shadow desync
structurally impossible, and `DESIGN_sheet_resolve.md` Phase 4a is explicit that
it replaced a hand-maintained lockstep pair. The cost is that a shadowed batch
walks its sheets twice, and the second walk recomputes everything the first one
did: mode 1 runs the *entire* transport (`corr`, the one-mesh ceiling, the §4.4
band arithmetic, the per-sample `svis` writes) plus `_tri_color_g`,
`_tri_extra_g`, `_tri_ior_transmission_g`, `_tri_shadow_normals` and
`_pixel_footprint`. Only `_shade_tri_hit` and the `_spawn_pool_ray` calls are
compiled out of it.

**The obvious saving is not available, and it is worth writing down why.** It
looks as though mode 1 could skip the colour and transport fetches, since it
shades nothing — but every one of them is load-bearing for the walk itself:

* the colour fetch's **alpha** becomes `mat_alpha` and drives `_run_svis_write`,
  so it decides the visibility later sheets see;
* **transmission** is passed to the same write as `trans_share`;
* **reflectivity** reaches `R`, and `refl_max >= cover_pass` is what `break`s
  the walk — so it decides which sheets are reached at all.

Since the accepted event set depends on all three, cutting them changes the
shadows. The kernel is already as thin as a mode-gate can make it.

What is left is **memoization between the passes**: mode 1 has already computed
each accepted sheet's colour, alpha, reflectivity, roughness, IOR, transmission
and normals, and the event tables it writes are indexed by exactly the sheet
index mode 2 uses. Widening those tables to carry the fetched values and having
mode 2 read them trades ~15 floats per sheet of bandwidth for six texture
fetches and their barycentric interpolations. The values are copied verbatim, so
it is byte-identical, which puts it inside the project's existing A/B
discipline (`benchmarks/_rt2_raster_shadow_parity.py`).

Whether that pays is unmeasured, and this container cannot rank it. The figure
to get first is simply the **ratio of mode 1's device time to mode 2's** from
`utils/profiling_utils.py` on a shadowed scene: if mode 1 is a small fraction of
mode 2, the walk is cheap next to the shading and there is nothing here. On the
reference profile shadows are named only as cost item 4 ("multiplied by the
number of lights") — that this also doubles the *resolve* has never been
separated out.

> **Measured 2026-08-26** (`benchmarks/_resolve_mode_ratio.py`, sync-bracketed
> per-launch wall time attributed by mode): on a shadowed spheres-over-ground
> scene, **mode 1 / mode 2 = 0.78 on a CPU box and 0.685 on a Tesla T4** (at
> MD: 10.5 s of mode 1 against 15.3 s of mode 2 over 7 launches) — the
> event-building walk costs nearly as much as the shading walk, so the double
> resolve close to doubles a shadowed batch's resolve cost and the
> ~15-floats-per-sheet memoization has real headroom. The memoization itself
> remains unbuilt.

---

## 10. `AttributeTimeline.get` — the prep pole

**`DESIGN_optimization_targets.md`, "What is left, in order", item 1. Unchanged
by this audit; repeated here so the ranking is complete.**

72.58 s (20.3%) of the reference render, 542,052 calls at ~134 us. It reached
the top by attrition: three rounds of work landed elsewhere. The lever is
**fewer calls**, and items 2 and 3 of that document (P9, P10) are both concrete
ways of removing them, so measure the three together. Re-measure the
`get/full` vs `get/replay` split first — the "two thirds in the geometry build"
figure predates P8, which changed the denominator.

---

## 11. T5 — the sparse-discovery host chain

**`DESIGN_optimization_targets.md` T5. The host loops are done; only the sorts
are left, and they should stay.**

The compaction's per-sample-lane reductions are kernels now
(`sheet_compact_taichi.py`, `SHEET_MASK_KERNEL` default on, bit-identical,
measured 1.25-1.33x on `compact_sheets`). The six-array gather T5 originally
proposed is built and bit-identical too but ships **default off**: worth ~4 ms
of a 1.3 s 4K frame against 50-160 MB of peak.

**The conflict-rank scan is done** (2026-08-21). It was eight `torch.cumsum`
passes over `[n]` plus a per-lane `index_select`, `maximum` and two `where`s,
with five live `[n]` arrays; `sheet_compact_taichi.sheet_conflict_rank`
(`SHEET_RANK_KERNEL`, default on) is one pass, a thread per band walking its
fragments forward with the eight per-lane counters in registers. It closes the
last of `DESIGN_sheet_resolve.md` §10.4, and it closes it by not needing what
that section was asking about: the bands are already contiguous runs of the
sorted stream, so no blocked segmented scan was required. Bit-identical **by
construction** — both arms are integer and walk the stream in the same order —
rather than by the order-independence argument the mask kernels beside it need.
Measured on CPU only (this container): at 1080p, one call per frame over
976,231 fragments, `_conflict_rank` 33 ms → 6 ms and `compact_sheets`
480 ms → 458 ms. **CUDA is unmeasured**, which is the open item on it.

What remains is **the sorts**, and T5's own advice is to leave them alone:
`_lexsort` is three stable `argsort`s and there are two `torch.unique` calls
after it. Radix sorts at these sizes were never the bottleneck the scans were.
Worth measuring instead, per T5: the per-fragment gathers in `_shade_class` and
`_prim_split_after`.

---

## 12. P9 / P10 — the batched geometry builds

**`DESIGN_optimization_targets.md` items 2 and 3. P9 shipped; P10 re-split and
one piece of it shipped.** What follows is what was found and what is left; the
detail is in that document under P9, P10b and P11b.

* **P9 — done.** The all-or-nothing group revert is gone. The constraint it
  existed to protect is *positional*, not group-wide: within one batch
  identifier a deferred circuit sits after some number of raw primitives of that
  identifier and before the rest, so splitting a group into **maximal runs of
  consecutive batchable actors** and merging each run puts every merged
  collection on exactly the span its circuits would have occupied. Byte-identical
  — a lossless two-arm render of a clashing scene differs by **0 pixels** — and
  on that scene 97.6% of circuits move to the batched build, taking
  `get_batch_of_primitives` to **0.43–0.48x**. Gated by
  `ALGAN_BEZIER_GROUP_RUNS`; guarded by
  `tests/unit_tests/test_bezier_group_runs.py`.

  Two things it turned up on the way. **`benchmarks/_bez_batch_parity.py` — the
  harness that guarantees the builder this widens — had rotted past running**
  (item 2's problem in the flesh, in the one script that certifies
  byte-identity for a path P9 quadruples the reach of: `set_render_settings`,
  `AnimationManager.instance()`, `TimelineManager.instance()` and
  `scene.actors[-1]` had all been gone for some time, and its attribute list
  named fields the primitive no longer has). Repaired, and it now runs on a
  2513-circuit group. And with it
  running again it caught a real defect: the batched builder flattened curves to
  **twice** the per-actor path's chord tolerance, which the default analytic-AA
  route hides by clamping and the classic supersampled route does not. Fixed;
  both builders now read one named constant.

* **P10 — re-split, and the re-split changed the answer.** The proportions this
  item quoted were measured before P11 halved `compute_grid_vertex_normals`, and
  they were wrong about which parts matter.
  `benchmarks/_surface_build_split.py` is the probe (it verifies its
  instrumented copies bit-identical to the shipped functions before timing
  them).

  * The **seam merge, pole fans and final normalize** named above are together
    **~4% of `compute_grid_vertex_normals`** — under 2% of the stage. Not worth
    touching. What is worth touching is the other 76.8%, "sides + crosses".
  * **`grid_to_triangle_vertices` on the whole stack is 9.5%, not 13.7%.**
    Fusing the two gathers buys ~2%, not the item this listed it as.
  * **The per-surface tail grew to 31%**, and its largest row is the primitive
    **construction (13.5%)** — a full colour clone plus two in-place passes per
    surface — which nothing had named.
  * **Shipped from it (P11b):** the four triangle sides are written straight
    into their buffers instead of through a materialized `roll`, and the four
    crossed pairs accumulate in place. Bit-identical, covered by the existing
    `benchmarks/_grid_normals_ab.py` across 13 grid topologies, **1.33x** on the
    sides-and-crosses block at the shape the batched build passes.
  * **Tried and rejected on measurement:** batching the per-surface colour
    gather. Bit-identical and **1.002x** — the `torch.stack` that feeds the
    single gather copies exactly the bytes the saved dispatches were worth. Read
    P10b before repeating it.
  * **What is left is a decision, not a patch.** The remaining large win inside
    the normals is the identity that collapses four cross products into one. It
    is exact in real arithmetic and **not bit-identical**, and the boundary
    zeroing breaks the algebra at the grid's edges — so it moves baselines and
    needs someone to decide that is acceptable.

---

## 13. `empty_cache` always collects on a CPU render

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

## 14. Delete the dead render paths

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

## 15. Stale docstrings that describe a renderer that no longer exists

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

## 16. Nine experimental toggles are unreachable from `SETTINGS`

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

## 17. The CPU baseline debt

**`DESIGN_mesh_identity_open.md` §B, and it is now a red CI on `master`.
Measured here, per commit.**

`tests/*/expected_outputs_cpu/` was last regenerated at `28efe67`. This
container reproduces that lineage exactly — the fast render **passes** there —
so it is a usable instrument for the drift since. Rendering `tests/fast` at each
commit that followed (with `sheet_compact_taichi.py` restored from `2016a26` so
the pre-fix commits can render at all):

| Commit | Worst channel difference vs the committed CPU baseline |
| --- | --- |
| `28efe67` — CPU baselines written here | **pass** |
| `1e90c87` — "Optimized memory usage of sheet resolve…" | **140** |
| `2b17e16` — "Minor UX pass." | 140 |
| `c293da3` — "Fixed bug in AAA which caused speckling artefacts" | 140 |
| `3be97a6` — "Ruff format." | 140 |
| `ead6dde` — merge of `github/master` (which brings in `28efe67` itself) | **40** |
| `b24a0c4`, `2e3264b`, `2016a26` (master) | 40 |

The control is what makes this readable: **`28efe67` passes here**, so this
container is on the same CPU lineage as the committed baselines and the rows
below measure drift rather than a machine difference. (The exact magnitudes may
differ on the GitHub runner; the pass/fail structure will not. Running the suite
writes a diff video to `tests/fast/output_errors/fast.mp4` — gitignored — which
is what to look at before regenerating anything.)

Tolerance is 2. Three things follow, and the third is the one that matters:

1. **The move starts at `1e90c87`**, a commit whose message describes it as a
   memory optimization ("combined some PyTorch ops, freed some variables when
   they are un-needed, implemented some things in Taichi kernels"). No baseline,
   CPU or CUDA, was regenerated with it.
2. **It is not the new compaction kernel.** Re-running `1e90c87` with
   `ALGAN_SHEET_MASK_KERNEL=0` still gives **140**, and HEAD gives **40** with
   the kernel on and 40 with it off. `sheet_compact_taichi.py`'s bit-identity
   claim survives this test; the movement is in the torch passes beside it,
   which is what "combined some PyTorch ops" would be expected to do to float
   reassociation.
3. **The merge that halved it is the merge of `28efe67`.** The local line had
   diverged before the commit whose CPU baselines it is being compared against,
   so part of the 140 was never a regression at all — it was the two lines
   disagreeing. That is exactly the situation a rebaseline exists to end.

This matters more than §B's framing suggests, because **CI renders on CPU**:
`tests/fast` compares against `expected_outputs_cpu/`, and
`test_full_render_scene` skips itself when `CI` is set. The CPU fast baseline is
the only pixel gate CI has, and it is the one that has been red since
`1e90c87`. Until it is green, no renderer change can be distinguished from the
drift already present — which disarms the project's primary safety net for
every item above.

The two traps §B records are still traps: `CUDA_VISIBLE_DEVICES=` (empty) does
not hide the GPU on Windows — use `-1`; and the render suites pick their
baseline directory from `torch.cuda.is_available()`, not from the render device,
so a CPU render that can still see a GPU compares against the CUDA set and
passes.

**Done when** `pytest -q --fast` is green on a CPU machine at `master`, the
regenerated frames have been looked at (`benchmarks/_diff_frame.py`), and the
commit says why output moved — per `CLAUDE.md`'s standing rule that a
rebaseline is never used to turn a red test green.

---

## 18. An untracked file on the default path reached master

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

## 19. Open design-doc items with no owner

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
* **`DESIGN_mesh_identity_open.md` §L — coincident duplicates.** Unbuilt, and
  the symptom has never been demonstrated. Its own instruction stands: render
  two coincident quads of the same mesh and of different meshes and measure the
  darkening before writing any code. It may close with a test.
* **`DESIGN_mesh_identity_open.md` §G — two-level BVH (TLAS/BLAS).**
  Deliberately deferred: the BVH build is ~1% of a shadowed five-solid render
  and no workload in the repo has thousands of repeated meshes. Revisit only
  when one does.

* **~~One `set_material` anywhere in a Scene silences every other mob's
  `color=`.~~ Fixed 2026-08-23 — and the title was wrong.** `set_material`
  never had anything to do with it, and the repro this entry used to carry
  does not reproduce: a `Cube` with a `MeshBasicMaterial` beside a
  `Line3D(color=MAGENTA)` renders the line magenta, measured. **The trigger is
  the lighting rig, and the site is `_stage_default`** in
  `algan/rendering/raytracing/shading_taichi.py` — the fragment stage every
  mob that was never given a material falls to.

  It summed a per-light fade weight and clamped the total at 1, and at 1 the
  albedo term is multiplied by zero, so the fragment renders as pure light
  colour. Two lights reach that alone: an ambient-like row (ambient /
  hemisphere / environment SH) shades along the surface normal, so `n · l` is
  exactly 1 and its share is the maximum 0.5 **whatever its intensity** — the
  fade weight was geometric only, carrying no radiance factor at all, while
  the illumination budget beside it had been radiance-weighted in `1a4c9d2`
  for precisely the reason that geometric counting is wrong. Two head-on
  direct lights do it too. It reads as a material problem because a mob that
  *has* a material shades through one of the other stages, every one of which
  multiplies the albedo rather than displacing it.

  The stage now composes the shares (`keep *= 1 - share`) instead of summing
  them — a product that shrinks but does not vanish, which is what the legacy
  per-light lerps did — and weights each share by the light's own radiance.
  Algan's default rig is one white point light, so a scene that adds no
  lights of its own is byte-identical (verified by md5). Four of the seven
  pixel suites move, measured HEAD against the change on a CPU-only machine:
  `text_and_media` 27, `complex_hierarchy_become` 34, `shapes_and_timeline`
  37, `solids_and_camera` 118 channel values; `tests/fast`,
  `manim_compat_and_plots` and `materials_and_lighting` are byte-identical.
  **No baseline was regenerated**, because both suites are already red on
  that machine at HEAD without the change: `tests/fast` by 5 channel values
  on one pixel (which this change does not touch), and the full-render CPU
  set by 231 at identical worst frames — its background is 24 where the
  renderer and the fresher CUDA set both produce 34, and at HEAD this
  machine agrees with the CUDA set to a mean of 0.02–0.11. The CPU set has
  drifted from changes predating this one; both device sets want a
  deliberate pass from someone who can run both.
  `tests/unit_tests/test_default_shader_albedo.py` is the guard. Two
  follow-ups left open: the torch twin `default_shader`
  (`shaders/pbr_shaders.py`) still lerps sequentially with the geometric-only
  weight, which is only reachable under
  `samples_per_pixel > 1` or `set_fragment_shading(False)`; and the shadow-fan
  culling sites that exclude `_MID_DEFAULT` because its fade used to
  accumulate at zero radiance (`wavefront_kernels_taichi.py:161`,
  `raster_taichi.py:2859`) are now merely conservative rather than
  load-bearing. `OX_DEFAULT_STAGE_AUDIT.md` is the call-site inventory.

* **`DESIGN_sheet_resolve.md` §6.1.1 — interpenetration is not antialiased.**
  Added 2026-08-23. Two opaque surfaces crossing inside one pixel both claim
  exact area 1 and the full sample union, and a sheet carries one scalar
  depth, so the whole pixel goes to whichever sheet sorts first. Which one
  that is has been repaired twice. `SHEET_POSITIONED_DEPTH` (2026-08-23)
  stopped an area donor — which owns no sample — from setting a sheet's
  depth. `SHEET_SAMPLE_DEPTH` (2026-08-24, default on) makes the decision
  PER SAMPLE rather than per pixel: the compaction reduces each sheet's
  nearest fragment *owning each sample lane*, an opaque full-coverage sheet
  publishes that as a per-(pixel, sample) floor, and another surface's sheet
  cedes the samples where the floor is strictly nearer. This is what the
  audit said did not exist — and it did not need a new datum, only a reduction
  the fragments already supported. Measured against a route-off supersampled
  reference on `solids_and_camera`: t=14.3 nine pixels move and all nine land
  closer (169 -> 2, 152 -> 0; summed error 834 -> 61), t=19.1 nine better and
  two worse; 150 pixels move over the 234-frame scene and none lands on the
  background. `benchmarks/_sample_depth_check.py` is the acceptance run.

  **Still open, and now the whole of what is left here:** the seam is a
  per-sample z-buffer, not an antialiased crossing. A ceded sample goes wholly
  to the winner, and because a lane's depth is its fragment's CENTROID depth
  the margin is untrustworthy when it is fine — hence the all-or-nothing rule
  above a `_SAMPLE_DEPTH_CEDE_FRACTION` floor, and hence the residual
  over-correction on pixels a reference blends. A true blend needs a depth plane per sheet and a
  per-sample tie-break in the resolve; `OX_SHEET_INTERPENETRATION_AUDIT.md`
  §6 scopes those call sites. Unowned.

  **What was NOT this, and what it cost to find out** (2026-08-24). Five
  arrow-coloured specks survived `SHEET_SAMPLE_DEPTH` in the triad's Act 3
  (video frames 141–143, 154, 165) and read as more of the same residual. They
  were a different defect entirely: `slice_time_window` deletes every `_rt_*`
  attribute of the shallow copy the arena preflight renders, surface identity
  wore that prefix, and nothing rebuilds it — so a merged collection collapsed
  to ONE `tri_obj` id and the compaction fused the red arrow, the green arrow
  and the Dot3D into a single sheet. Fixed by renaming the three attributes out
  of the prefix (see `RenderPrimitive.slice_time_window`); 65 of 234 frames and
  2687 pixels moved, and both remaining fused-sheet families in this scene (the
  triad, and the saddle leaking through the torus at t≈19.1) went with it.

  **The measurement trap it exposes, which applies to every item here:** the
  preflight only slices a window when there is more than one frame, so a
  `save_frame` of time *t* and the video's own frame at *t* were rendering
  DIFFERENT scenes — same fragments, different sheets. Every harness in
  `benchmarks/` that renders the triad (`_triad_artifact_frame.py`,
  `_sample_depth_check.py`, `_triad_sheet_probe.py`) uses `save_frame` and was
  therefore structurally blind to it. Reproduce a video artifact with a video
  render (`_triad_video_probe.py`), and note that ffmpeg's 1-based frame files
  are one ahead of the `--at` grid: `f0165.png` is `t = 16.4`.


## 20. The shadow terminator on diced surfaces

**Status: built, default on** (`SETTINGS.raytracing.experimental.shadow_terminator`
/ `ALGAN_SHADOW_TERMINATOR`). This is the half of item 3 that item 3 could never
have delivered, split out once that was established rather than left attached to
a mechanism that cannot address it. What the section below diagnosed was right
about the mechanism and wrong about the symptom's visibility; the correction is
under "What building it found" at the end.

Item 3 named two symptoms: contact shadows erased by an absolute epsilon, and
shadow acne at grazing light angles. Only the first is an epsilon problem. Acne
is a mesh shadowing *itself*, and every identity scheme keeps a floor for that
population by construction -- otherwise a concave solid stops shadowing itself.
Worse, the acne hits are not near-zero `t` at all: near the terminator the ray
leaves almost tangentially and travels a long way before striking a
neighbouring facet, so no acceptance floor of any size rejects them.

The claim traces to a bullet in `DESIGN_mesh_identity.md:1994` ("reject its own
mesh at near-zero `t` ... removing shadow acne"), which is internally
inconsistent: rejecting near-zero `t` on your own mesh is precisely *keeping*
the guard on the population acne comes from. §I inherited the sentence.

**What actually causes it here.** `_faces_viewer`'s docstring
(`shading_taichi.py:143`) states the geometry: a PN patch carries "a quadratic
normal field over a quadratic position patch", diced to *flat* triangles. The
shading normal is therefore not perpendicular to the facet the ray starts on,
and the facet is a chord *below* the smooth surface it approximates, so
neighbouring facets rise above the plane the ray was offset from. The current
origin offset is `sorigin = spos + fnrm * (10 * MIN_HIT_DISTANCE)`
(`raster_taichi.py:2641`) -- along the *face* normal, a fixed 1e-3, with no
relation to how far the true surface bulges above the facet.

**The remedy is standard and its inputs are already on the event.** Hanika's
shadow terminator fix (Ray Tracing Gems II, ch. 4) offsets the origin onto the
smooth surface implied by the vertex normals, by an amount derived from the
hit's barycentrics and the per-vertex normal deviation. `event_snrm` (shading)
and `event_fnrm` (geometric) are both already stored per shadow event and both
already read in `raster_shadow_trace`; only `fnrm` is used today.

Note what currently hides the symptom: the shadow trace is entered only where
`fnrm.dot(wis) > 1e-3 and snrm.dot(wis) > 1e-4` (`raster_taichi.py:2756`), so
the terminator band does not trace shadow rays at all. That is why a convex
solid shows no acne today -- and also why a convex solid cannot be used to test
any of this. Use a concave single mesh (a `Torus` is the cheap one).

**Done when** a diced curved surface under a grazing light shows no acne with
the guard angles relaxed, and item 3's feature can be turned on without
introducing speckle at seams.

### What building it found

**The acne was never hidden. It is on the default path today, and the offset
removes it.** A lit `Torus` under a side-on point light at LD carries 41 speckle
pixels (darker than their own 3x3 neighbourhood median by more than 6 levels)
with the feature off and **4** with it on -- and the diagnostic arm that relaxes
the cull *without* moving the origin carries 38, sitting with the off arm rather
than the on arm. That is the attribution: relaxing the cull is not what cleans
the image, the offset is. `benchmarks/_shadow_terminator_ab.py` is the run.

Three corrections to the diagnosis above, all found by measuring it:

* **"The terminator band does not trace shadow rays at all" overstates the
  cull.** `fnrm.dot(wis) > 1e-3` rejects a ray only within 0.06 degrees of the
  facet's own plane. It is a hair, not a band, and it was never what kept acne
  off a convex solid -- adaptive dicing was, by keeping facets near a pixel
  wide. Relaxing it moves 177 pixels of a torus and 24 of a sphere; the offset
  moves the other ~40 speckle pixels that the cull never touched.
* **A convex solid does show the symptom**, just less of it: on a diced sphere
  the relax arm darkens 20 of the 24 pixels it moves, and the on arm darkens
  none of them. So "a convex solid cannot be used to test any of this" is too
  strong -- it is a weaker instrument than a concave mesh, not a blind one.
* **A flat facet is untouched by construction, not by tolerance** — by either
  of two guards, and for Algan's own flat family it is the one you would not
  guess. A `Polyhedron` packs no vertex normals at all (its corner normals are
  literally zero), so the degenerate-normal guard returns the zero vector; a
  mesh that does carry a duplicated face normal per corner trips the
  constant-normal-field test instead. Either way `delta` is the zero vector and
  neither the origin nor the cull moves. `Cube` and a 2-D circuit scene are
  byte-identical in all three arms, which is why every flat-shaded scene cannot
  move — and `tests/fast`, which has no shadows at all, could not have anyway.

The one thing the cull relaxation is for is stated where it is done: with the
origin on the smooth surface, the *face* normal's horizon is no longer the
surface's, so keeping it would refuse rays the corrected origin can now trace
honestly.

**Not covered.** The Monte Carlo megakernel (`samples_per_pixel > 1`) keeps the
old origin; so does every reflection/refraction continuation. `wavefront_shadow`
carries the change but has no caller, so it is compiled by nothing.

---

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

---

## Appendix: where render memory goes

The audit was asked for memory-wasteful sections as well as slow ones. The
honest headline is that **the render arena is measured rather than modelled**
(`memory_model.py` fits `peak(n) = a + b*n` to its own high-water mark), so
there is no stale byte formula to go wrong and no allocation table to
regenerate. What follows is not a list of leaks; it is where the bytes actually
are, so a future change knows what it is trading against.

Ranked by how much of a large frame each accounts for.

**1. The whole render chunk's fragment stream, live at once.**
`prepare_sparse_raster_coverage` runs **once per render chunk, over every frame
in it**, not per frame. Its own reservation arithmetic states the cost:

```
discovery_bytes = discovery_frags * 29        # pre-truncation scratch
                + num_frags      * 28         # the persistent compact result
                + num_covered    * 8
                + num_sheets     * 32
                + (num_covered + 1) * 4
```

So roughly **57 bytes per emitted fragment**, and a 4K frame is ~3.7 M
fragments. This is the reason a dense scene's chunks are short, and it is the
first thing to look at if a scene will not fit. `SPARSE_DISCOVERY_SAFETY` (1.25)
pads the learned figure so the *next* chunk is sized to fit it.

**2. `compact_sheets`' own peak.** Its comments carry the measurements, all at a
3840x2160 frame: the popcount's `zeros_like` held 26 MB of int64 for values
below nine (now int32); `_shade_class`'s `[n, 3]` / `[n, 3, 3]` gathers are 42
and 126 MB and were the function's allocation peak until each was freed the
statement after its last read; `_exact_fragment_order`'s two permutations are
56 MB each and are the discovery peak; the conflict-rank loop's five live `[n]`
arrays cost 70 MB until they were narrowed to int32, and cost nothing at all
now that the kernel arm (item 11) keeps only its output. All of that is already
optimized once.

**3. The frame buffer, and the fallback's multiplier.** Under post-process
tonemapping (the default) the frame buffer is **float32 rather than uint8** —
4x — which the settings comment names as the cost of doing bloom in linear HDR.
`ALGAN_HDR_BUFFER_F16` halves it back but is opt-in because Pascal-class cards
run f16 torch post-processing far slower than the saving is worth. On top of
that, any batch that falls off the analytic path renders at `anti_alias_level`
squared — **another 4x at the default**. The two compound: a fallback batch
under HDR is 16x the byte-buffer baseline.

**4. Shadow event tables.** With shadows on, `shade_sparse_raster_coverage`
allocates eight dense per-sheet arrays sized by the *slice's whole sheet count*
— `sheet_accept`, `event_pos/snrm/fnrm` (3 x 3 floats), `event_frame`,
`event_msk`, `event_dp` (6 floats when `sec_aa > 1`) and `sheet_event_id` —
about **76 bytes per sheet**, plus `shadow_vis` at `num_events * num_lights * 4`.
They are allocated for every sheet and then compacted down to the accepted ones,
so the peak is set by the sheet count and not by the event count.

**5. `wf_finalize_uncovered`'s full-frame mask.** When the tonemap runs
in-kernel (`post_process_tonemap=False`) the sparse route allocates a
`uint8[total_px]` coverage mask over the whole chunk purely to find the pixels
the covered composite will not touch. One byte per pixel-frame, for a boolean.

**6. Transients the model cannot see.** The merge and `project_to_screen` build
out of place in *pool headroom*, not in the arena, so the runtime chunk model
has no visibility into them. They are bounded instead by deliberately generous
multiples of their packed inputs: `MERGE_GPU_PEAK_FACTOR` **6.0** and
`PROJECT_GPU_PEAK_FACTOR` **8.0**. Both are estimates with the OOM retry as the
exact fallback. This is a known, accepted imprecision — do not tighten either
without re-checking `test_render_batch_sizing.py`.

**7. The STBVH's temporal expansion.** At the confirmed-optimal tightness of
1.0, moving geometry segments into near-per-frame instances, so the classic tree
is **~10x the primitive count** (`DESIGN_hybrid_raster.md` §9 — the finding that
motivated `BVH_REFIT`, now default on). `BVH_DEFER` (default on) avoids building
trees at all for a batch that provably never traverses one, which is the common
shadow-free non-reflective case.

**8. `RASTER_FUSED_GATHER`, and why it is off.** The fused six-array gather is
bit-identical and faster, and it ships **default off** because forcing all six
outputs to exist before the first is written raises a 4K frame's peak CUDA
allocation by **50-160 MB** to save 4 ms of a 1.3 s frame. Recorded here because
it is the clearest example in the codebase of the trade this appendix is about,
and because the reasoning is easy to lose: turn it on only on a bandwidth-bound
machine with VRAM to spare.
