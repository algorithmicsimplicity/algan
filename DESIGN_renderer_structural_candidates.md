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

## 1. The merged scene stores byte-identical frames — dedup the texture bank, then the geometry

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

Two structural facts about the deterministic shadow query
(`_shadow_occluded`, `raytrace_kernels_taichi.py:2743`; mode selection
`tracer.py:1358-1381`):

* **Default behaviour is the most expensive mode.** With `SHADOW_ANYHIT` off
  (the default), every shadow ray runs an *ordered closest-hit march that
  restarts a full traversal of every geometry type's tree once per peeled
  surface* — on batches that are provably all-opaque, where a single unordered
  any-hit walk answers the same question. The cheap modes exist and are wired:
  mode 3 (any-hit only, march compiled out) engages when the batch provably
  contains no translucent geometry, mode 2 (any-hit pre-pass, march fallback)
  for mixed batches, mode 4 ("gather", KBUF-batched peel — `ceil((k+1)/KBUF)`
  traversals instead of `k+1`) for any batch. They ship **default off**,
  "experimental while the pixel suites qualify it" (`settings.py:599-624`),
  and are byte-identical except two enumerated corner cases where the any-hit
  answer is the physically correct one. The candidate is qualification, not
  engineering: run the suites under each mode (the harness exists,
  `benchmarks/_shadow_anyhit_check.py`), then default `SHADOW_ANYHIT=1`.
* **The fan multiplies serial tree walks.** `raster_shadow_trace`
  (`raster_taichi.py:2744`) launches one thread per (event, light); inside it
  the soft-shadow fan runs `SOFT_SHADOW_SAMPLES` serially, and each sample's
  `_shadow_occluded` walks the triangle tree and the bezier tree one after the
  other. So a soft-shadowed scene with both geometry types pays
  `events x lights x samples x trees` sequential traversals, and an area light
  multiplies further (`K` emitter rows x fan). The single mixed-type any-hit
  tree `DESIGN_hybrid_raster.md` §13.3 proposes — leaves carrying a type bit +
  typed primitive index — removes the `x trees` factor and lets one walk
  terminate on the first opaque hit of either type. Shadow rays need any-hit
  only, so this tree can also drop the classic tree's ordering obligations.

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

`_cat_collections` (`raytracing/utils.py:51`) broadcasts every collection's
time axis to the batch maximum and then **materializes** it with
`torch.cat(...).contiguous()`. `_expand_frames` hands out stride-0 views, so
the expansion is free until the cat; the cat is where a parked mesh becomes T
physical copies because one other mob of its geometry type moved. Downstream,
everything that iterates the merged arrays then does per-frame work on static
rows: the projection table (`[T, N, 13]`), the screen-bounds tables, the
refit-BVH's per-frame bounds reduction and per-(frame, child) link words, the
upload, and the arena peak that sizes the chunk.

The pipeline already knows what is static and throws it away: `_dedup_time`
collapses whole-static arrays (defeated by one mover), the dice computes
`distinct_frames`/`geometry_static` per call (T4), and the criterion kernels
already take **one real frame plus a stride the kernel multiplies its frame
index by** (`_frame_broadcast_base`, `DESIGN_optimization_targets.md`
"Maintaining the shipped kernels") — the exact mechanism this item generalizes.

**What to build.** Two designs, in ascending fidelity:

* **Static/dynamic block split**: partition each geometry type's collections
  by staticness before the cat, keep two merged blocks with their own time
  lengths. Kernels already read each array as `f % shape[0]`, but per-row
  arrays indexed `[t, n]` need the block boundary threaded through — every
  consumer gains one branch or one extra launch per block.
* **Per-row stride**: one `int8`/bitmask array `row_is_static[n]`; consumers
  index `pos[f * stride[n], n]`. One extra load per access, single arrays,
  no reordering — likely the cheaper retrofit given how many consumers there
  are.

**Impact.** Breadth: every mixed static/moving scene — which is the *general
moving scene* the project's own performance discipline names as the target.
Depth: proportional to the static fraction of merged bytes and of every
per-frame pass over them; the probe's static texture case (item 1) is the
extreme end of the same distribution. Confidence: medium — the mechanism is
proven (`_frame_broadcast_base`), but the consumer inventory is wide
(`CLAUDE.md`: do not casually change merged-field shapes; the Ox merge audit's
consumer list is the map). Cost: the largest build in this document; do item 1
first — it is this item's cheapest slice and pays for the estimator work both
need.

## 4. Split the resolve monolith: the sheet stream was designed for material-sorted shading and never got it

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

Every batch rebuilds from scratch: the PN dice (T4 collapsed identical frames
*within* a batch; across batches a parked `Sphere` re-dices every ~4-100
frames), the bezier chord counts and circuit geometry, the BVH *topology*
(binned SAH over ever-visible primitives, rebuilt per batch even when the
actor set is unchanged — the refit machinery updates bounds per *frame* but
the topology itself is per-batch), and the projection precomputes. Each is
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

Two enablers rather than wins:

* The **scene-descriptor ndarray** (`DESIGN_hybrid_raster.md` §13.9, still
  open): the resolve is over the argument ceiling already (item 4), every new
  feature pays argument gymnastics (`layer_offsets` smuggling), and the
  remaining ~0.8 ms/launch pybind floor only a descriptor removes. Do it with
  or before item 4.
* The **per-chunk sync inventory** (Ox launches audit): the bounce loop's
  per-iteration count readback and per-tile pool readback are known and
  accepted (§13.7 ranked the sync-free loop low on measurement); the audit's
  value is the list of *avoidable* syncs — host passes re-deriving per-batch
  facts per chunk. Fix opportunistically.

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

## The boundary this document does not cross

Prep — the timeline/replay/geometry-build pole — is the *larger* pole of the
reference render (73.6% vs 56.7% post-P11) and its top items
(`AttributeTimeline.get`, `set_state_to_times`, the surface build) are ranked
in `DESIGN_optimization_targets.md` "What is left, in order". Nothing here
supersedes that ranking; items 1 and 3 above are the places where a renderer-
side representation change also removes prep work (fewer bytes built, fewer
frames expanded), which is the only kind of renderer item that beats a prep
item of the same size under the current pole balance.
