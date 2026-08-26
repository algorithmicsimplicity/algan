# Rendering pipeline

The render loop, the sheet route, shading contracts and colour. Read this before touching
`algan/rendering/` or any `*_taichi.py` kernel.

## The render loop

The render loop is implemented in `../algan/render_loop.py` as `RenderLoopMixin`, mixed into `Scene`. It is responsible for:

- choosing frame windows according to animation and render memory budgets;
- materializing the Scene timeline at frame times;
- building and batching actor render primitives;
- snapshotting camera and light state;
- optionally prefetching the next batch on a worker thread;
- projecting, merging, and uploading scene data;
- invoking the configured render kernel;
- applying post-processing and streaming frames to the writer;
- reducing the frame window and retrying on render-memory exhaustion.

`ALGAN_PREFETCH_BATCHES=0` disables next-batch prefetch. Keep Scene render-state snapshots immutable so preparation can run safely while the previous batch renders.

### Scene merge and acceleration structures

`../algan/rendering/raytracing/scene_builder.py` packs projected primitives into contiguous tensor arrays grouped by geometry type and builds the corresponding spatio-temporal acceleration structures. The merged dictionary is the contract consumed by the tracer orchestration and Taichi kernels.

Do not casually change merged-field widths, ordering, dtype, or lifetime. Those changes affect memory estimators, arena preflight, kernel signatures, projection/merge paths, and potentially cached Taichi variants.

### Renderer dispatch

`render_batch_raytraced` is the production render entry point registered in `KERNEL_REGISTRY`.

- `samples_per_pixel == 1` selects the deterministic wavefront renderer. It uses bounded primary-ray tiles, traversal, shading, compaction, compositing, and a shared continuation pool for reflective/refractive splits. Tile overflow is retried with fewer primaries rather than approximated.
- `samples_per_pixel > 1` selects the Monte Carlo path-tracing megakernel (`raytrace_kernels_taichi.py`), which has its own cold compile. Some deterministic-only features are rejected or handled according to the unsupported-feature policy.

The deterministic renderer resolves primary visibility through the sheet route (`../algan/rendering/raytracing/DESIGN_sheet_resolve.md`) when the batch qualifies: exact analytic-coverage fragments are emitted for flat triangles and Bezier circuits, compacted on the host into per-pixel depth-banded sheets (`sheets.py`), and resolved/shaded — shadow events included — by the one kernel body in `sheet_resolve_taichi.py`, while reflection/refraction continuations remain in the ray-based wavefront system. It is the only analytic-coverage resolve; batches the route rejects (analytic AA off, transparent background with an env map, SPP > 1, route toggles off) render through the classic supersampled wavefront. `analytic_raster_route_active` in `tracer.py` is the single host-side route decision shared by allocation planning and rendering. Do not describe the current renderer as either a pure rasterizer or a pure one-primary-ray-per-pixel tracer.

The classes under `../algan/rendering/primitives` are still used for primitive construction and batching. They are not a separate supported legacy raster backend. New renderer work belongs in the active raytracing/hybrid pipeline unless a deliberate new backend is being introduced through the registries.

## Kernels, shaders and materials

Kernels live in `*_taichi.py` files. Material pipelines and custom scatter (ray-continuation) functions are injected as `ti.template()` parameters — compose user `@ti.func`s into one func and pass **flat** tuples. Nested tuples do not work as kernel template arguments.

Shaders (`shaders/`): Three.js-style `Material` objects configure shaders and register animatable shader parameters. Per-vertex shaders run in Python/torch; per-fragment shading and custom fragment pipelines (`fragment_shaders.py`, `FragmentStage`) execute inside the Taichi shade kernel.

A scene with custom fragment shading/scatter may force deterministic fragment-shading paths and alter continuation-pool requirements. Keep capability detection, memory estimation, render planning, and actual kernel dispatch consistent.

Feature toggles live in `raytracing/settings.py` as module globals with env-var defaults plus setter functions, surfaced through `SETTINGS.raytracing`. **Read them live** (`rt_settings.X` at call time) — importing them by value at module import freezes them before user code runs (this bug has shipped before).

Post-processing (`post_processing/`): bloom/glow, FXAA/SMAA, tonemapping. `Camera` (`camera.py`): perspective/orthographic projection, fov/near/far; render code consumes an immutable camera/light snapshot per batch so batch prep for frame batch N+1 can run on a worker thread while N renders (`ALGAN_PREFETCH_BATCHES=0` disables).

## The default for a Mob that sets no material is a *material*

`SETTINGS.style.default_material` is a `Material` instance — `DiffuseMaterial()` at import; `Scene.use_manim_defaults()` swaps in `ManimMaterial()`, which reproduces Manim's `get_shaded_rgb` (offset added in display-referred sRGB, so the stage encodes, adds, clamps and decodes). `TrianglePrimitive` takes both the shader *and* the material's parameter values off it, so a configured default is honoured rather than silently rendering at `_MAT_DEFAULTS`.

**Reaching the triangle primitive is not the same as being a 3-D Mob**: `ImageMob` (a `Surface`, but a picture) and a `TriangulatedBezierCircuit`'s fills (glyph fills, plot curves — circuits that happen to be triangulated) declare themselves unlit with `set_shader(null_shader)`, which is also what `set_shader(None)`'s docstring has always promised. And a bare Mob now carries the same `lambert_shader` object an explicit `MeshLambertMaterial` Mob does, so `get_batch_identifier` separates default-seeded from mob-authored primitives — the collection merge transposes parameter rows column-wise and would otherwise truncate the authored ones.

`tests/unit_tests/test_default_material.py` is the guard.

## Coplanar 2-D geometry draws in author order

Circuits at the same depth tie on distance, and the tie is broken by their position in the merged arrays — which follows the draw order only *within one merge block*, since filled circuits, stroked ones and each distinct texture-grid shape are packed separately and a block lands wherever its first member did.

`RenderLoopMixin._authored_draw_order` resolves the order the author asked for (each tree walked parent-first, roots in creation order — Manim's flattened family, stable-sorted by `z_index`) and spends it as a per-circuit bias toward the camera of one `DEPTH_TIE_EPSILON` per merge-block *alternation* along that walk — tens of bins for a whole scene, not one per Mob, which is what keeps the displacement sub-pixel. `BezierCircuitCubic.z_index` is the author's override and propagates to the sub-hierarchy.

`ALGAN_COPLANAR_DRAW_ORDER=0` restores the previous rule (a global sort by hierarchy depth, which interleaved unrelated trees) for A/B. `tests/unit_tests/test_coplanar_z_index.py` is the guard.

## Shading sidedness is declared by the geometry, not the material

A mob with a meaningful outside — every built-in solid — sets `Mob.two_sided = False`, and a back-facing hit on it is shaded as its *inside* (ambient) rather than being lit as though it faced the camera. Everything else (a parametric `Surface`, 2-D shapes, `Text`, imported meshes, a `Polyhedron` whose faces are not a closed orientable shell) stays two-sided and keeps the viewer-facing flip.

The declaration rides the primitive as `one_sided` and lands in slot `_MAT_ONE_SIDED` of the material block, where `_run_frag_pipeline` reads it once per hit — so every ray type (camera, reflection, the coverage pass-through behind a transparent surface) agrees. Set it before spawning; the primitive reads it once.

It is what stops a half-transparent solid's far shell from rendering as a second lit front shell, and it makes outward normals load-bearing — `tests/unit_tests/test_normal_orientation.py` is the guard.

## Shadows are per mob, not just per scene

`SETTINGS.raytracing.shadows` remains the switch for the feature; on top of it a mob declares `Mob.casts_shadows` and `Mob.receives_shadows` (both default `True`), set before spawning like `two_sided`, and neither is animatable.

Each rides a word the renderer **already loads**, so opting out costs no extra memory traffic and leaving them alone costs nothing at all. Casting is a bit in the BVH leaf word — bit 15 of `stbvh.leaf_tspan`, which was free because the frame-interval halves are clipped to 15 bits, and bit 29 of the refit tree's link word, for which `LINK_PRIM_MASK` narrowed from 30 bits to 29 — tested only where a **shadow** ray accepts a leaf (a `nocast` template that is 1 at the shadow call sites and 0 everywhere else, so a non-caster stays visible to camera, reflection and refraction rays). Receiving is slot `_MAT_NO_SHADOW_RECEIVE` of the material block, read where a shadow ray would be *spawned* — the sheet route's mode-1 event build and `wavefront_shade`'s inline block — so declining is strictly cheaper than the default rather than a query whose answer is discarded. Both are spelled negatively (`no_shadow_cast` / `no_shadow_receive`) because the material block's padding rule and the `_surface_params` fill both require a 0.0 to mean the old behaviour.

**A flag set on an aggregate reaches its geometry**: `resolved_shadow_flags()` walks ancestors, so `cube.casts_shadows = False` covers the faces that actually hold the triangles and `group.casts_shadows = False` covers a subtree — reading the flag off the mob that builds the primitive silently ignored both.

**`receives_shadows` is inert wherever a mob was never shadowed anyway** — 2-D geometry and anything unlit, and a mob carrying a custom fragment pipeline, whose own parameters own the block slot this would ride (the same reason a custom pipeline is never asked about `one_sided`). `casts_shadows` has none of those exceptions.

A **diced** collection additionally splits its merge group by the caster flag: the leaf word carries one bit per merged column for the whole batch, and an adaptively-diced column can otherwise change hands between frames, which ate a bite out of a *casting* sphere's shadow on every frame until `get_batch_identifier` separated them.

`benchmarks/_shadow_flags_check.py` is the acceptance harness (each flag against an independently rendered oracle — the same scene with the mob deleted, and with shadows globally off), `tests/unit_tests/test_shadow_flags.py` is the guard, and `ALGAN_PER_MOB_SHADOW_FLAGS=0` restores the old behaviour byte-identically, host-side, with no kernel variant changed.

## A closed solid at `opacity < 1` composites once, not once per shell

`Mob.opacity` is a property of the Mob, so rendering at `a` must give `a * (the Mob rendered opaque) + (1 - a) * backdrop` — which a flat `Circle` satisfied and no solid did: a camera ray crosses the shell twice, both crossings composited, and an authored 0.55 rendered as an effective 0.679 on a `Sphere` and 0.744 on a `Cube`.

The sheet compaction now caps a shell's cumulative exact coverage per (pixel, surface) at `max(front, back)`, spent in true depth order. It is keyed on a declared **closed shell** (`Mob.closed_shell`), not on `two_sided`, and that distinction is load-bearing: an open `Cone`'s mouth pixels legitimately hold both facings of one surface, and the conflict-rank machinery deliberately double-attenuates a translucent surface a ray crosses twice.

Closedness is computed rather than asserted — a partial-sweep `Sphere` is open, `Cone`/`Cylinder` need caps *and* a full sweep, and `Polyhedron` takes it from the closed-orientable-manifold proof `orient_faces_outward` already performs.

**The rule reaches primary visibility only**: a mirror's image of a translucent solid, and any `samples_per_pixel > 1` render, keep the doubled composite, because the wavefront bounce loop carries no surface identity. `benchmarks/_opacity_alpha_check.py` is the acceptance harness and `ALGAN_SOLID_SHELL_ALPHA=0` restores the old behaviour byte-identically.

## The working colour space is linear

Authored colour is display-referred (`Mob.color` reads back what you set, and colour tweens interpolate there), decoded to linear light where it is packed for the renderer, and encoded with the sRGB OETF once at the byte write — after exposure and after any tonemap curve, which is three.js's order.

This is what makes lights additive: sRGB encoding is concave, so summing encoded values overshoots badly. Unlit flat 2-D is unaffected, because decode-then-encode with no arithmetic between is the identity; what moves is anything the renderer computes — lit surfaces, antialiased edges, alpha compositing, the supersample downsample.

`SETTINGS.raytracing.linear_color_space` / `ALGAN_LINEAR_COLOR=0` restores the display-referred pipeline (and with it the illumination-budget normalisation and the peak-scale bound, which only exist to tame gamma-space light sums). `AMBIENT_STRENGTH` is 0.01 under the linear space and 0.1 under the other — the same fill, different units.

`LINEAR_COLOR_WORK.md` is the reference; `benchmarks/_linear_color_check.py` is the acceptance harness, one process per arm.

## An area light's shadow is an integral over its emitter, not a stack of hard tests

A `RectAreaLight` expands into `K = k*k` point emitter rows carrying `1/K` of the power each, and every one of them used to cast its own *hard* shadow — so the union was a staircase with `K + 1` levels (measured `[0.01, 0.25, 0.52, 0.74]`, a `k/4` grid, at the shipped `samples = 4`) where the reference path tracer ramps continuously.

A row stands for one **cell** of the grid, so its visibility is now the average over that cell: `build_aux` packs the cell's half-extents and the rectangle's `right` axis into columns the area type never used (packed 9/10, 12-14; column 11, the shadow-radius column both fans already read, is the gate and the isotropic fallback), and both deterministic fans place their `SOFT_SHADOW_SAMPLES` rays inside the cell **in the light's own plane** — an R2 low-discrepancy set whose `s = 0` sample is exactly the cell centre, so a one-sample fan degenerates to the old ray.

Radiance, power fractions and `intensity` are untouched, which is why this fixes the shadow half of `REPORT.md` §6.7 and not the falloff half. Cost: `K * SOFT_SHADOW_SAMPLES` shadow rays instead of `K`, the same rule a `PointLight` with a non-zero `shadow_radius` already obeys.

Two things worth knowing: the umbra legitimately *lifts* (a `k x k` centre grid spans only `(1 - 1/k)` of the rectangle, so `samples = 4` shadowed from an emitter half the authored size), and `MAX_SHADOW_LIGHTS` is 16 with one slot per emitter sample, so `samples > 16` silently loses shadow — raising `samples` was never a route past this.

The flag is read **host-side only** (`ALGAN_AREA_LIGHT_SOFT_SHADOWS` / `SETTINGS.raytracing.experimental.area_light_soft_shadows`): off, `build_aux` packs zeros and the kernels take their existing path with no recompile and no per-arm process. `benchmarks/_area_light_shadow_check.py` is the acceptance harness; `tests/unit_tests/test_area_light_soft_shadow.py` is the guard, and its render arms exist to **compile both fans** — a host-side test cannot see a Taichi scoping error, which is how one shipped mid-review.

## The render path's fixed ceilings are counted, not silent

`raytracing/truncation.py` counts surfaces per ray, shadowed lights, overlapping layers of one surface in a pixel, and dropped continuation rays. Each warns **once per render job** at `WARNING` — these degrade the image, unlike the batch splits and pool retries that log at `PERF` because they are the memory model working — and the running totals ride on `RenderPlan.truncations`.

The counters are unconditional, so a zero is a reading rather than a missing instrument; keep them that way when adding a ceiling.
