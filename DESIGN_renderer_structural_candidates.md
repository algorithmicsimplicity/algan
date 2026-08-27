# Renderer structural improvement candidates

**What this is.** A structural audit of the default renderer — the deterministic
hybrid rasterizer/raytracer (sheet route for primary visibility, wavefront loop
for continuations) — carried out 2026-08-26 on `claude/algan-renderer-perf-h76mcs`,
looking for *representation and architecture* changes that would speed it up:
the acceleration structures, texture handling, deduplication, geometry
representation, and the kernel/launch shape. It complements
`DESIGN_optimization_targets.md`, which is the plan of record for measured
optimization work and stays that way: anything picked up from here should land
there with its measurements. `RENDERER_WORK_QUEUE.md` is the correctness/quality
queue; two of its items (9, 4) reappear here because they are structural.

**Evidence basis.** Source reading in a CPU-only cloud container (no wall-clock
rankings from here are meaningful, and none are quoted); one new probe,
`scratch_perf/probe_time_expansion.py`, whose numbers are reproduced below; the
measurement record in the design docs; and four read-only Ox Alpha audits whose
reports are beside this file's evidence trail
(`scratch_perf/ox/REPORT_struct_{accel,textures,merge,launches}.md` — briefs
beside them). Audit claims quoted here were spot-checked against the source.

**How to read the ranking.** Expected impact = how many scenes are affected x
how much of an affected render moves x confidence. Each item states all three
and its cost. The two-pole caveat from `DESIGN_optimization_targets.md` applies
throughout: prep and render overlap, so a render-thread win is discounted by
the overlap — quote per-stage numbers, treat whole-render extrapolations as
upper bounds.

---

> **STATUS 2026-08-26 (`claude/structural-redesigns-perf-pmpkv3`): items 1,
> 3.1, 5 (content dedup + copy chain), and 8's avoidable syncs are BUILT,
> byte-identical, each behind its own kill switch; item 2's flip is
> qualified and measured. Measurements and the full record live in
> `DESIGN_optimization_targets.md`, "The structural round (2026-08-26)" —
> this file keeps only per-item status stamps below.**

## 1. The merged scene stores byte-identical frames — dedup the texture bank, then the geometry

> **BUILT, byte-identical** (`benchmarks/_texture_dedup_ab.py` — both arms,
> pinned windows, non-vacuity asserted per path). (1) is `TEXTURE_TIME_FLAT`:
> each map's frames flatten along the texel axis with a real per-map time
> length in `tri_tex_meta` cols 10-12, so the assembled bank keeps time
> length 1 — the length travels as *data*, which is what lets one compiled
> sampler serve both layouts (and the fix also stops the env map's T-fold
> re-expansion). (2) is `MERGE_DEDUP_GEOMETRY`: `tri_pos`/`tri_obj`/
> `tri_closed` and both geometry types' bounds/opacity/caster tables join
> the collapse list. (3) is shipped as observed-constancy pricing: the
> primitive build records whether the window collapsed
> (`Surface._texture_window_collapsed`, the `_texture_is_wrap_padded`
> pattern) and the two estimators price a collapsed texture at the
> materialized window alone. The probe scene's merged upload: 127.2 MB →
> 32.3 MB. The full timeline-level window stride (materializing `[1, ...]`
> in the first place) remains item 3's larger build.

**The measured fact.** On a 20-frame PREVIEW scene of four static mobs (an
`ImageMob`, a `Sphere`, a `Text`, a `Cube`; nothing animating, camera parked),
the merged upload is **127.2 MB, of which 125.9 MB — 99% — is
`scene["textures"]` at `[4, 1573542, 5] f32`: four byte-identical copies of the
image's texel bank** (verified elementwise; `scratch_perf/probe_time_expansion.py`).
And the 4 is not the scene's length — it is the batch window *the texture bytes
themselves capped*: the control arm with the `ImageMob` swapped for a `Circle`
renders the same scene as **one 20-frame batch** (merged total 3.9 MB), so the
window went 20 → 4 on the texture pricing alone. The control also shows the
geometry side of the same fact at full length: `tri_pos` is `[20, 2469, 9]`
with **all 20 frames byte-identical** (1.78 MB where 0.09 would do), and
`tri_obj` `[20, 2469] int32` likewise — while `tri_mat`/`tri_norm`/`tri_uvs`,
which are in the dedup list, sit collapsed at `[1, ...]` right beside them.

**Why.** `MERGE_DEDUP_TIME` (default on) collapses a temporally-constant time
axis to length 1 — but only for the ten arrays in its two lists
(`scene_builder.py:1639`, `:1829`: the material/normal/colour/uv triangle
tables and the circuit tables). `textures` is not in the list, and `tri_pos` is
excluded deliberately ("rigid motion lives in tri_pos", `scene_builder.py:1638`)
— a rationale about the common moving case that also forfeits the static case,
where the equality check is one pass and the saving is (T-1)/T of the array.
Every consumer already reads time as `f % shape[0]`, so the collapse is
mechanical.

**What to build, in cost order:**

1. Time-collapse the texel bank. The cheap seam is per map, not per scene:
   run `_dedup_time` on each map inside `_append_texture`
   (`scene_builder.py:1327`) — my probe's bank was dense-T with a *single*
   textured collection, so the density arrives from the materialized window,
   upstream of the assembly. Byte-identical by construction (same mechanism,
   same `f % shape[0]` consumers as the ten arrays already collapsed). One
   step remains after that: the assembly `_cat_collections(_texture_tensors,
   1, ...)` (`scene_builder.py:1722`) re-expands every bank to a common T
   when banks disagree — so a scene mixing an animated texture with static
   ones needs a per-texture time length in the texture meta (alongside the
   existing `(offset, w, h)`) for the collapse to survive. That is the
   complete fix and is where this item meets item 3.
2. Reconsider `tri_pos` (and the other per-frame geometry the collapse skips):
   the equality probe is cheap, and a static batch currently pays T copies of
   its diced geometry. T4's closing note ("emitting a `[1, ...]` diced array is
   worth much less than it looks", `DESIGN_optimization_targets.md`) blamed
   exactly this merge-time re-expansion; collapsing at the merge is the half of
   that item that does not change the primitive's contract.
3. **Teach the batch-size estimator the same facts.** The per-timestep pricing
   (`_get_render_device_memory_used_per_timestep`, the render-device budget of
   the T4 round) charges a static texture per frame, so dedup alone shrinks
   VRAM but the windows stay short — the estimator must price a
   temporally-constant wide attribute at one frame for the batches to
   lengthen. Longer batches then amortize *every* per-batch cost (BVH build,
   projection tables, prep passes, launches), which is where most of the win
   is.

**Impact.** Breadth: every scene carrying an image/video texture, plus (via
`tri_pos`) every scene with static diced geometry — and through the batch
window, every per-batch overhead. Depth: on the probe scene the merged bytes
drop ~4x and the window can grow ~5x; on the nn benchmark family the T4 round
already measured texture transport dominating prep before its fixes, and this
is the remaining copy it left in place. Confidence: high — mechanism shipped
ten times already, and the probe is elementwise. Cost: small for (1)+(2);
moderate for the estimator half; the per-texture time axis is a real contract
change (see `CLAUDE.md`'s merged-field warning).

## 2. Shadow rays: the early-out is already built and nobody turned it on; then stop walking one tree per geometry type

> **QUALIFIED — and the flip is deliberately NOT taken, on measurement.**
> Correctness: `benchmarks/_shadow_anyhit_check.py` on this CPU box AND on a
> Tesla T4: both corner-case scenes prove their case reached (peel limit
> hit, tie separation sensitive) and modes 0/1/gather are byte-identical on
> both — and on `materials_and_lighting` (three modes in one process). But
> the measured perf refutes this item's premise for the mixed batch: on nn
> UHD (translucent present → mode 2) the flip cost **29.5 → 34.2 s** — the
> deferred any-hit pre-pass pays a second full traversal on miss-dominated
> rays and the wider mode-2 shade variant loses occupancy — and the shadowed
> static gallery measured neutral. The all-opaque mode-3 case this item's
> depth claim rests on was not reached by either benchmark scene. Default
> stays off; numbers in `DESIGN_optimization_targets.md`. The mixed-type
> any-hit tree is NOT built (unchanged: a real build on the `refit_bvh.py`
> pattern, stealing a leaf type bit).

Two structural facts about the deterministic shadow query
(`_shadow_occluded`, `raytrace_kernels_taichi.py:2743`; mode selection
`tracer.py:1358-1381`; the Ox accel audit,
`scratch_perf/ox/REPORT_struct_accel.md`, traced the exact loop nests):

* **Default behaviour is the most expensive mode.** With `SHADOW_ANYHIT` off
  (the default), every shadow ray runs the ordered transmittance march —
  *realized as repeated nearest-hit restarts*: a peel loop up to
  `MAX_SURFACES_PER_RAY` where every iteration begins a complete fresh
  two-tree nearest-hit traversal (`raytrace_kernels_taichi.py:2915, 2925`).
  A k-surface translucent stack pays k+1 full traversal pairs; an all-opaque
  batch pays the ordered machinery where a single unordered any-hit walk
  answers the same question. The cheap modes exist and are wired: mode 3
  (any-hit only, march compiled out, first-hit early exit) when the batch
  provably has no translucent geometry, mode 2 (march + one deferred any-hit
  walk) for mixed batches, mode 4 ("gather", KBUF-batched peel —
  `ceil((k+1)/KBUF)` traversals instead of `k+1`) for any batch. They ship
  **default off**, "experimental while the pixel suites qualify it"
  (`settings.py:599-624`), byte-identical except two enumerated corner cases
  where the any-hit answer is the physically correct one. The candidate is
  qualification, not engineering: run the suites under each mode (the harness
  exists, `benchmarks/_shadow_anyhit_check.py`), then default
  `SHADOW_ANYHIT=1`.
* **The fan multiplies serial tree walks.** The audit's loop-nest reading of
  `raster_shadow_trace` (`raster_taichi.py:2744`): one thread per
  (event, light); inside it the fan runs serially — `ns = 8`
  (`SOFT_SHADOW_SAMPLES`) for soft lights, and the analytic-AA secondary
  sampling forces `ns >= 4` even for *hard* lights — and each sample's
  `_shadow_occluded` walks the triangle tree then the bezier tree. Total:
  **events x lights x fan samples x (peels+1) x trees-present**, with area
  lights multiplying further (K emitter rows). The single mixed-type any-hit
  tree `DESIGN_hybrid_raster.md` §13.3 proposes removes the trees factor and
  lets one walk terminate on the first opaque hit of either type. The audit
  scoped it concretely: the leaf needs a type discriminator (the refit link
  word's primitive field is already narrowed to 29 bits, so the bit must be
  stolen or ride an aux array), its consumers are `_shadow_anyhit_opaque` and
  `_shadow_occluded`'s callers (the sheet route's shadow trace and both
  wavefront shadow paths), the Monte Carlo megakernels keep per-type trees
  (they need ordered hits) — and `_collect_hits` already interleaves both
  types into one KBUF gather with a packed hit type, so the mixed tree
  "promotes that gather-level merge into the structure". Note also two counts
  the design docs still get wrong (PN deletion predates them): there are two
  geometry types and at most **four** trees per batch, and by default
  `OPAQUE_BVH_SKIP_DEAD` already aliases the opaque-prepass trees to the main
  ones (~40% of the per-batch BVH build already saved).

**Impact.** Breadth: shadows are opt-in (`SHADOWS` defaults off), so this is
the shadowed population only — but every lit-3D production scene turns them
on, and on the UHD nn benchmark `raster_shadow_trace` was the second-largest
kernel (3.7 s of 15 s kernel time, T4 round). Depth: the march→any-hit flip
alone removes the per-surface traversal restart on the common all-opaque
batch; the deferred-shadow experiment already established this stage is
traversal-bound. Confidence: high for the flip (built, harnessed); medium for
the mixed tree (unbuilt, touches every shadow call site). Cost: qualification
+ a default change; the mixed tree is a real build on the `refit_bvh.py`
pattern.

## 3. One moving mob re-expands every static collection: give the merged contract a real time stride

> **Step (1) BUILT** as part of item 1's `MERGE_DEDUP_GEOMETRY` (the collapse
> happens at the merge rather than at `_pack_frame_visibility`, which covers
> the same consumers in one place): collapsed bounds now reach both BVH
> builders at `Tc == 1`, waking their static branches, and the raster host
> tables already read `f % shape[0]`. The projection tables still key their
> length off the CAMERA tensors, which the kernels index dense (`cam_origin[f]`,
> no modulo) — so a parked camera still builds per-frame projection tables;
> collapsing the camera is the next cheapest slice of this item and needs
> the kernels' camera reads made modulo first. Steps (2)/(3) (block split /
> per-row stride) remain unbuilt.

`_cat_collections` (`raytracing/utils.py:51`) broadcasts every collection's
time axis to the batch maximum and then **materializes** it with
`torch.cat(...).contiguous()`. The Ox merge audit
(`scratch_perf/ox/REPORT_struct_merge.md`) confirmed the mechanism and moved
its origin one step earlier: **the T identical rows exist before the merge
ever runs** — timeline materialization hands every attribute back as a dense
`[T, rows, D]` window whether or not the mob moved, and
`_pack_projected_flat_geometry` packs dense — "the merge's contribution is
preventing them from ever going back to 1" (its whole-array `_dedup_time` is
the one recovery point, and one mover voids it per table). Downstream,
everything then does per-frame work on static rows: the projection and
screen-bounds tables (confirmed: no static shortcut anywhere, and the lights
are expanded per frame too), the refit-BVH's per-frame bounds reduction and
link words, the upload, and the arena peak that sizes the chunk. The audit's
field table prices it: `tri_mat` is 136 B/(frame·tri) and survives at full T
when *any* material slot animates anywhere; `tri_colors`/`tri_extra` 60 each
under the same all-or-nothing rule; `tri_pos` (36) and the bounds/flag tables
(~50) are never collapsed at all.

The pipeline already computes static-ness and throws it away — the audit
names five signals and where each dies. The sharpest: **both BVH builders
already implement the static case** (`build_stbvh`: `Tc == 1` → one instance
spanning all frames, `stbvh.py:692-735`; `build_refit_bvh` accepts `Tc == 1`
and "dedupes to one time slice", `refit_bvh.py:284-298`) — and both branches
are **dead code**, because `_pack_frame_visibility` unifies bounds against
the corners' frame count so `frame_lo/hi` always arrive with T rows. (The
two audits disagreed here — the accel audit read the refit docstring's
"static geometry dedupes to T = 1" as shipped, the merge audit called the
branch starved; the probe settles it: the all-static arm's merged `tri_pos`
is `[20, ...]` dense, so the collapse never receives a collapsed input.) The
others: `geometry_static` (dies at the dice's dense `allocate()`),
`_frame_broadcast_base`'s stride-0 detection (feeds only the level-search
kernels — but it is the proven precedent: one real frame plus a stride the
kernel multiplies by), `_dedup_time` (cannot say *which rows* were constant),
and the frame-valid masks (key on visibility, not constancy).

**What to build.** In ascending cost:

1. **Stop expanding at pack where a consumer already handles length-1**:
   let `_pack_frame_visibility` keep a static collection's bounds at
   `[1, N, 3]` and the BVH builders' existing static branches wake up.
2. **Static/dynamic block split**: partition each geometry type's
   collections by staticness before the cat, two merged blocks with their
   own time lengths. Kernels already read each array as `f % shape[0]`;
   per-row `[t, n]` arrays need the block boundary threaded through.
3. **Per-row stride**: `pos[f * stride[n], n]` with a per-primitive stride
   word. One extra load per access, single arrays, no reordering — but no
   consumer supports mixed strides today; the audit's claim-5 inventory
   (four kernel modules, the raster host tables, the sheets host chain, both
   BVH builders, the bezier acceleration) is the change list.

**Impact.** Breadth: every mixed static/moving scene — the *general moving
scene* the project's performance discipline names as the target. Depth:
proportional to the static fraction of merged bytes and of every per-frame
pass over them; the probe's static texture case (item 1) is the extreme end
of the same distribution. Confidence: medium-high for (1) — the consumer
already exists; medium beyond. Cost: (1) is small; (3) is the largest build
in this document. Do item 1 first — it is this item's cheapest slice and pays
for the estimator work both need.

## 4. Split the resolve monolith: the sheet stream was designed for material-sorted shading and never got it

> **NOT BUILT — and DEMOTED 2026-08-27, because the measurement it was
> staged behind turned out not to size the stage.** The item-9 memoization
> was built as the cheap half of this idea and measured on a Tesla T4:
> `sheet_resolve_shade` is **1.2% of an `nn_scene_UHD` render and 0.6% of a
> `static_gallery_PREVIEW` one**, and the memo moved it by under a
> millisecond. The ~12 s / ~16 s per-mode figures that ranked this work came
> from `benchmarks/_resolve_mode_ratio.py`, which brackets each launch with a
> device sync and therefore charges it the queue it drains — on a render
> whose whole resolve kernel is 0.3 s. See `RENDERER_WORK_QUEUE.md` item 9
> for the table and the reasoning.
>
> That does not refute this item's *architecture* argument — the resolve is
> still a 72-parameter megakernel at a 21-25% occupancy ceiling, and that
> ceiling is still real. What it removes is the evidence that splitting it
> would buy much on these scenes: a rework "on the scale of the sheet flip
> itself" cannot be justified by a stage measured at 1.2%. Before this is
> picked up again it needs a scene where the resolve is a large share of the
> render, established from an UNSYNCED profile. If no such scene exists in
> the benchmark set, that is the finding.

`sheet_resolve_shade` (`sheet_resolve_taichi.py:111`) is one kernel per covered
pixel that walks the depth-sorted sheets and does *everything* inline:
transport (per-sample transmittance, bands, caps, cedes), all texture/material
fetches, full shading for every material pipeline composed in as `ti.template`s,
spawns, and the shadow-event build. It is at **72 parameters against Taichi's
64 runtime-argument ceiling** (its own comment, `sheet_resolve_taichi.py:235`),
and it inherits the megakernel family's measured occupancy ceiling (21-25%,
register-capped — `DESIGN_hybrid_raster.md` §1, §13.4). The sheet design
itself specified the alternative and it was never built: P5, "flat kernel over
the sheet stream, **sorted by material pipeline id** for dispatch coherence;
one evaluation per sheet" (`DESIGN_sheet_resolve.md` §3) — the coherence the
old sorted-material experiment wanted and could not afford *per fragment* is
affordable *per sheet* (measured S/F compaction 0.39-1.00 on the six pixel
scenes, i.e. up to a 2.6x shrink there and more on dense diced geometry — and
the stream already exists on the host).

The Ox launches audit (`scratch_perf/ox/REPORT_struct_launches.md`)
inventoried what every thread actually carries: an 8-float per-sample
transmittance vector plus band/cap/bounce scalars at pixel scope, and per
sheet a **48-float `lvis` vector** (3 x `MAX_SHADOW_LIGHTS` = 16, declared
unconditionally because the direct-specular add-back re-reads it), an 8-float
slot vector, and four reflection/refraction continuation blocks with up to
32-tap unrolled jitter fans — all compiled into every variant whose batch
flags allow them, paid by matte threads that never take the branches. It also
confirmed the mode-1/mode-2 double walk re-fetches everything (only
`_shade_tri_hit`, the visibility read and the spawns differ between modes)
and that the item-9 event tables carry no material payload today.

**What to build.** Three stages sharing the sheet tables: (a) transport walk —
per-pixel, small state, writes each sheet's visible weight and per-sample
visibility; (b) shade — flat launch over accepted sheets, sorted by pipeline
id, one material evaluation per sheet writing radiance; (c) composite —
weights x radiance segmented-summed into pixels. This *subsumes*
`RENDERER_WORK_QUEUE.md` item 9 (the shadowed resolve running the whole kernel
twice, mode 1 + mode 2, re-fetching everything): with transport separated and
its per-sheet results stored, the event build is the transport pass and the
re-fetch disappears — the ~15-floats-per-sheet memoization that item scopes
becomes the natural interface rather than a widening.

**Impact.** Breadth: every sheet-route render (the default path). Depth:
unknown until measured — the binding constraint is occupancy, and register
relief has precedent as the fix but no measurement at sheet granularity; item
9's mode-1/mode-2 ratio is the cheap first number to get (its section says
exactly how). Confidence: medium. Cost: large — this is a resolve rework on
the scale of the sheet flip itself; stage it behind the item-9 measurement,
and note the scene-descriptor ndarray (`DESIGN_hybrid_raster.md` §13.9) is the
enabler that keeps three kernels' argument lists sane.

## 5. Texture storage: 20 bytes per texel, one bank, no content dedup, no mip chain

> **Content dedup BUILT** (`TEXTURE_CONTENT_DEDUP`: exact-match reuse at
> `_append_texture`, so N mobs sharing an image store it once — `get_image`
> still re-decodes the file per call; the merge-level dedup makes that a
> per-authoring cost rather than a per-batch one, so the file cache was not
> taken). **Copy chain shortened for the static case** by item 1's window
> collapse (`TEXTURE_WINDOW_COLLAPSE`: premultiply/pad/decode/concat run on
> one frame instead of T). **u8 storage NOT taken**: colour maps are stored
> premultiplied by the (animated) opacity, so exact u8 round-tripping needs
> the premultiply moved in-kernel with a per-(prim, frame) opacity input —
> a real contract change; a 256-entry decode LUT solves the sRGB half but
> not this. **Mip chain NOT taken** (quality-first design, unchanged).

The shared texel bank stores **five f32 channels per texel** for every map
(colour, material, normal — `scene_builder.py:1318`, item 1's probe: 20 B/texel
x T). The Ox textures audit (`scratch_perf/ox/REPORT_struct_textures.md`)
walked the full transport and put numbers on the multipliers; four structural
upgrades, smallest first:

* **Content dedup.** A texture used by N mobs is stored **N times**: every
  textured primitive is deliberately a singleton collection
  (`render_loop.py:2329-2338`), the merge appends one bank per collection
  with no key of any kind (`scene_builder.py:1593-1597`), and `get_image`
  re-reads and re-decodes the *file* on every call with no cache
  (`utils/file_utils.py:44-54`) — even the source tensors are distinct.
  `_split_promotable` already groups promoted 1x1 maps by value
  (`scene_builder.py:584-635`); image maps need the same idea with a content
  hash at `_append_texture`, plus a cache in `get_image`. Cheap.
* **u8 storage for u8 sources.** Sources arrive 8-bit and are stored f32 with
  a padded fifth channel: **x5 bytes** vs u8-RGBA (x4 dtype, x1.25 channels)
  on the largest array in any textured merge. The sampler consumes plain
  lerps of stored values, so nothing requires f32 storage; in-kernel decode
  (the sRGB->linear decode already has an in-kernel twin) can reproduce the
  current f32 values exactly for a byte-identical flip. Do item 1's dedup
  first (bigger factor), then narrow.
* **Shorten the copy chain.** The audit counts ~4 full-size copies of every
  texel per batch beyond the buffer the kernel reads — the materialize write,
  a per-batch opacity-premultiply `clone`, the sRGB decode, the arena upload
  — plus up to two device moves, with 3-4 near-full representations coexisting
  at peak. Fusable pairwise (premultiply-on-upload, decode-in-place) without
  representation changes.
* **A mip chain.** `RENDERER_WORK_QUEUE.md` item 4's quality gap is also a
  bandwidth gap: minified sampling strides the full-resolution bank. The sheet
  record already carries the exact covered area (`sheet_cov`, unclamped f64
  sum — `sheets.py:881-885`) from which an LOD can be derived without
  derivatives — with the audit's caveat that screen area alone needs the
  per-primitive UV scale (`tri_uvs` + meta w/h) to become a texture-space
  footprint. Quality-first item; the perf side rides along.

**Impact.** Breadth: textured scenes. Depth: large there — for the audit's
animated 1774x887 case, ~944 MB per 30-frame batch reaches the GPU at f32-5ch
where the genuine data is ~1/5 of that; for static textures items 1+5 compose
(xB frames, xN mobs, x5 width, all removable). Confidence: high for
dedup/narrowing mechanics (source-verified); the mip chain is a design. Cost:
small / moderate / small / moderate respectively.

## 6. Batch amnesia: nothing frame-invariant survives a batch boundary

> **NOT BUILT, by this item's own staging** ("lengthen batches first, then
> measure what per-batch cost remains"). Item 1's collapse + pricing are the
> lengthening half; the static-gallery profile in the structural round
> (`DESIGN_optimization_targets.md`) is the re-measurement baseline for any
> future cross-batch cache.

Every batch rebuilds from scratch — the Ox merge audit confirmed there is no
cross-batch persistence of geometry *values* anywhere (what does persist is
pure topology: dice patterns, subdivision indices, sample-weight tensors):
the PN dice (T4 collapsed identical frames *within* a batch; across batches a
parked `Sphere` re-searches levels and re-evaluates every patch), the bezier
circuit geometry and edge-acceleration tables, the BVH *topology* (binned SAH
over ever-visible primitives, rebuilt per batch even when the actor set is
unchanged — the refit machinery updates bounds per *frame* but the topology
itself is per-batch), and the projection precomputes. One caution the audit
added: **bezier chord counts cannot be reused verbatim across batches** — the
search reduces its error over all frames of the batch (`amax` at
`primitives.py:3018`), so the count is window-dependent by construction and
caching it across windows changes rendered edges (the same batch-window
sensitivity `scratch_perf/ox/REPORT_batchwide_audit.md` catalogues). Each is
individually modest by the shipped measurements (dice 2.9% post-T4; BVH build
~1% of a shadowed render, which is why §G's TLAS/BLAS deferral stands; T6's
tables 1.4%), and item 1 lengthening batches shrinks all of them — which is
the right order: **lengthen batches first, then measure what per-batch cost
remains** before building any cross-batch cache. If one is built, the daemon's
source-fingerprint pattern and the dice's `geometry_static` detection are the
precedents (key on content + explicit invalidation, not on identity — P4's
caching rule).

**Impact.** Breadth: static-heavy scenes. Depth: bounded by the per-batch
shares above — small-to-moderate, and *smaller after item 1*. Confidence:
high on the sizes, by prior measurement. Cost: caching across batches crosses
the render-state snapshot boundary; not cheap. Rank it after 1 re-measures it.

## 7. The chunk-wide fragment stream sets the memory shape

> **NOT BUILT.** The prescribed first measurement (sort-size sensitivity)
> already exists: `torch.sort` was linear 1M→16M keys with no cliff
> (`DESIGN_hybrid_raster.md` §2.1), which says the smaller sorts cost
> little — the open half is the peak-memory trade of a per-frame emission,
> still unmeasured. Rank it after item 1's longer batches re-shape chunk
> lengths.

`prepare_sparse_raster_coverage` (`raster_pipeline.py:1402`) emits and holds
the **whole chunk's** fragment stream at once — ~57 B per fragment
(`RENDERER_WORK_QUEUE.md` appendix item 1), ~3.7 M fragments per 4K frame —
which is what makes dense scenes' chunks short, and short chunks multiply
per-chunk overheads (item 6's family). The batching exists to feed a few large
radix sorts, which are efficient and should stay (T5's standing advice). The
structural question, unmeasured: whether a per-frame (or half-chunk) emission
+ compaction pipeline, keeping only compacted *sheets* chunk-resident (S/F
0.39-1.00 measured, and sheets are 32 B against the fragment stream's ~57),
trades enough peak memory to lengthen chunks more than the smaller sorts
cost. Measure the sort-size sensitivity first
(`torch.sort` was linear 1M→16M keys with no cliff, `DESIGN_hybrid_raster.md`
§2.1 — which argues the smaller sorts cost little).

**Impact/confidence.** Depth: indirect (through chunk length) and scene-
dependent; confidence low until the trade is measured. Cost: moderate — the
emission is already per-frame inside; the lifetime change is the work.

## 8. Launch mechanics: the 64-argument ceiling and the sync inventory

> **The avoidable syncs are FIXED**: `_shadow_identity_epsilons` caches the
> batch's scene diagonal on the merged scene (was a whole-batch `tri_pos`
> reduction + `.item()` per TILE ATTEMPT), and the two sheet-compaction
> split-group diagnostics stay 0-d device tensors (with the group tables
> over-allocated to `nb` so the `ngroups` sync went too). The
> `gen_meta`/`layer_offsets_t` re-materialization was looked at and left:
> they are <64-byte H2D copies, and `layer_offsets_t[7]` is rewritten per
> tile by the glossy route, so caching them across chunks buys microseconds
> and risks the arena lifetime. The scene-descriptor ndarray stays open
> with item 4.

Two enablers rather than wins:

* The **scene-descriptor ndarray** (`DESIGN_hybrid_raster.md` §13.9, still
  open): the resolve is over the argument ceiling already (item 4), every new
  feature pays argument gymnastics (`layer_offsets` smuggling), and the
  remaining ~0.8 ms/launch pybind floor only a descriptor removes. Do it with
  or before item 4.
* The **per-chunk sync inventory** (Ox launches audit, Question A: seventeen
  host passes between launches, each classified). The bounce loop's
  per-iteration count readback and per-tile pool readback are known and
  accepted (§13.7 ranked the sync-free loop low on measurement); the
  *avoidable* ones the audit found are worth their few lines each:
  `_shadow_identity_epsilons` reduces the **entire batch's `tri_pos`** to a
  scene diagonal and syncs on `.item()` at **every resolve call — once per
  tile attempt** — for an answer fixed per batch
  (`raster_pipeline.py:1379-1388`, called at `:2235`); two diagnostic
  counters sync unconditionally per compaction (`sheets.py:1313-1314`); and
  `gen_meta`/`layer_offsets_t` re-materialize batch-constant scalars into the
  arena per chunk. Fix the first one on sight; the rest opportunistically.

## Anti-candidates — measured or decided; do not re-attempt without new evidence

* **The compaction sorts** (three stable argsorts + two uniques): cuB radix;
  hand-writing a Taichi sort to lose is not a plan (T5).
* **`RASTER_FUSED_GATHER` default-on**: 4 ms for 50-160 MB of peak at 4K —
  the peak binds first on small cards (T5). The flag exists for big ones.
* **TLAS/BLAS**: build is ~1% and no workload has thousands of repeated
  meshes (`DESIGN_mesh_identity_open.md` §G). Revisit only when one does.
* **PN-patch rasterization / force-flat under raster / K-buffer in registers
  / per-frame BVH rebuild / time-interpolated OBB nodes / deferred shadow
  kernel for the wavefront**: `DESIGN_hybrid_raster.md` §14's list stands.
* **Bloom as a Taichi kernel**: measured far slower than cuFFT.
* **Attribute-interpolation dedup and the batched colour gather**: both
  measured at or below break-even (T4, P10b) — dedup pays only when what it
  skips beats a gather.
* **Byte-formula memory accounting**: the runtime model measures; keep it
  that way (`RENDERER_WORK_QUEUE.md`, "found sound").
* **A per-pixel bezier evaluation accelerator**: already built. This audit
  went in expecting a candidate here and found the pruning shipped —
  16-band scanline bins for crossing parity, an 8x8 spatial grid for border
  distance, circuit-level early-outs (`bezier_acceleration.py`), applied at
  emission, traversal and every shadow walker; only the sheet resolve skips
  it, correctly, because it re-uses the emission's stored hit. The worst
  case degenerates on adversarial glyphs, but that is data, not structure
  (Ox launches audit, claim 3 — REFUTED as asked).

## Doc drift the audits surfaced

> **FIXED 2026-08-27.** All of the below, plus two sites the original list
> missed (`settings.py`'s `SHADOW_ANYHIT` note and `scene_builder.py`'s
> `_empty_scene_part` docstring, both saying "three-tree"/"all six"), and
> §13's ranked future-work list, whose items 1 and 2 still asked for the
> default flips that have since happened. §13 item 3 (the mixed-type tree)
> keeps its rank but now carries the measured caveat from item 2 above.

For `RENDERER_WORK_QUEUE.md` item 15's list, found while auditing rather than
hunted: `DESIGN_hybrid_raster.md` §9/§13 still say "three trees"/"six trees
(3 full + 3 opaque-prepass)" — there are two geometry types and at most four
trees since the PN deletion, and by default the opaque trees are aliases; its
§11 default table predates `BVH_REFIT`/`HYBRID_RASTER`/`ANALYTIC_AA` turning
on; and the gather-march docstring in `raytrace_kernels_taichi.py:3066-3068`
also still says "three-tree".

## The boundary this document does not cross

Prep — the timeline/replay/geometry-build pole — is the *larger* pole of the
reference render (73.6% vs 56.7% post-P11) and its top items
(`AttributeTimeline.get`, `set_state_to_times`, the surface build) are ranked
in `DESIGN_optimization_targets.md` "What is left, in order". Nothing here
supersedes that ranking; items 1 and 3 above are the places where a renderer-
side representation change also removes prep work (fewer bytes built, fewer
frames expanded), which is the only kind of renderer item that beats a prep
item of the same size under the current pole balance.
