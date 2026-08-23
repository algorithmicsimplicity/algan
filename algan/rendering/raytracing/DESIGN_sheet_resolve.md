# Algan — the sheet resolve: analytic AA as data, not control flow

**STATUS: BUILT AND SHIPPED (2026-08-19), same day as the decision.** The
sheet resolve is the default renderer for every batch it accepts; both CUDA
baseline sets were regenerated once under it on the box that owns them,
frames reviewed. The CPU sets are deliberately still the old epoch: they are
regenerated on the CI machine (owner's decision, 2026-08-19), and until then
`tests/full_renders` and the fast pixel compare are only meaningful on CUDA. Each phase's record lives inline in §8, and every deviation from
the plan is annotated where the plan states the original intent — the two
that matter: the one-mesh ceiling SURVIVES as sheet data (§7's subsumption
claim was refuted by measurement), and P7's slot rework is DEFERRED because
the determinism criterion was met without it (§8 Phase 4 record). §H/§I
remain live items, now on sheet records. Decided 2026-08-19 with the project
owner: the renderer is being redesigned, output WILL move, and the committed
baselines will be regenerated once at the end — so this document optimizes
for the right system, not for byte-compatibility with the old one.

================================================================================
0. WHAT THIS SUPERSEDES, AND WHAT IT KEEPS
================================================================================

**The following are OUTDATED as designs. Do not work their queues.**

* `DESIGN_analytic_aa.md`, `DESIGN_analytic_aa_v2.md` — the run rule, the
  per-fragment resolve semantics, and everything layered on them.
* `DESIGN_mesh_identity.md` — the run-scan limit, the one-mesh cap, the exact
  run totals (§6.6–§6.8), and the remaining §-queue items about them.
* `DESIGN_mesh_identity_open.md` — its §C (run-scan limit) queue is closed by
  this redesign; §H (nested-IOR refraction) and §I (self-shadow rejection by
  identity) remain live items and port cleanly onto sheets (§9).
* `DESIGN_hybrid_raster.md` — partially: fragment EMISSION (the raster
  front-end, the exact clippers, the CSR) survives; the per-fragment sequential
  resolve it describes is replaced.

**They remain the measurement record.** Every number, refutation and trap in
them still happened; this document cites them for *why* rather than repeating
them. `DESIGN_optimization_targets.md` (render performance) is unaffected.

Two findings from the closing investigation of the old system shape everything
below, and are recorded in `DESIGN_mesh_identity.md` §6.7.3:

1. **A pixel diff read from an MP4 measures the encoder, not the renderer.**
   The moved-pixel figures the old design agonized over were inflated by up to
   ~2,000x by H.264 decode (18 real pixel-frames read as ~37,000; a "42 over
   16% of a frame" that was really 2,063 pixel-frames at worst |d| 4). All
   arbitration of this redesign happens on lossless renders
   (`codec="libx264rgb"`, `ffmpeg_params=["-crf", "0"]`) and on the exact-
   reference harnesses — never on decoded mp4 diffs.
2. **Every measured defect class of the old resolve traces to one fact**: it
   rediscovers surface identity DURING a bounded, stateful, per-pixel
   sequential walk. The 16-fragment scan budget and its notches (§0.5), the
   truncation machinery (§6.7/§6.8), the engagement gate whose interaction
   with zero-alpha despawned fragments produced a season of confusion
   (§6.7.3), the one-mesh cap (§6.6), the host/kernel dual-language bugs
   (§6.7.1 and four prior — §0.1 rule 4), and the requirement that the shadow
   walk replay the shade walk in lockstep — all of it is the cost of
   aggregating too late, inside control flow, with bounded lookahead.

The fix is structural: aggregate FIRST, into an explicit data structure, and
make the resolve consume that.


================================================================================
1. THE DESIGN IN ONE PARAGRAPH
================================================================================

A **sheet** is a maximal same-surface region within one pixel: keyed
`(pixel, mesh id, facing, depth band, region class)`, carrying the exact
covered area (a sum over its fragments, which the fill rule makes exact within
a band), the union of sub-pixel sample masks, a representative depth, and a
shading reference. A batch-wide **compaction** pass turns the sorted fragment
stream into the sheet stream. The **resolve** composites each pixel's few
depth-sorted sheets front to back — per-sample transmittance as a prefix
product, per-sheet magnitude from its own exact area — with the **background
(flat color or environment map) as the final sheet**, so nothing about the
resolve depends on what the background is. **Shading is evaluated per sheet**,
not per fragment. Every reduction in the pipeline is a sort plus a
fixed-order segmented scan; **no stage uses atomics**, so the whole resolve is
deterministic by construction. Analytic AA is then a single boolean: it is on
whenever the raster front-end is, for any scene the front-end accepts —
environment maps and tonemap choices are not inputs to that decision.


================================================================================
2. WHY SCANS, NOT BOUNDED SEQUENTIAL WALKS
================================================================================

The question was asked directly during the design discussion: since this is a
software renderer, can prefix scans replace the limited sequential resolves?
Yes — and not as an optimization but as the correct architecture, for three
independent reasons. The economics point first: a hardware rasterizer cannot
afford global sorts and multi-pass scans per frame, which is why hardware AA
is fixed-sample; a software batch renderer already pays for a global fragment
sort per batch (the emission sorts by `(pixel, depth)` today) and runs at
memory bandwidth on flat arrays. Scans are the primitive this renderer is
GOOD at, and the old design's bounded walk was hardware thinking in a
software renderer.

**2.1 Scans have no lookahead limit.** `_AA_MAX_RUN_SCAN = 16` existed only
because a per-thread sequential walk cannot afford unbounded lookahead — and
that single constant generated the notch defect, the truncation flag, the
frag_cap substitution, the exact-totals lanes, and their bug tails. A
segmented reduction over the fragment stream aggregates a 200-fragment sheet
and a 2-fragment sheet in the same two passes. The entire §6.6–§6.8 apparatus
is what "we could not afford a scan" cost.

**2.2 Fixed-order scans are deterministic; atomics are not.** Every
determinism defect in the record is an atomics defect: `scatter_add_` and
`cumsum` are non-reproducible on CUDA (memory: f64-accumulate-then-round was
the workaround), split pixels differ run to run through `pix_accum`'s
atomic-add order, and the §J MD-resolution noise floor (46 channel values on
translucent stack edges, unexplained) is in the same family. A hand-written
segmented scan with a fixed reduction tree reassociates floats — differently
from left-to-right, but the SAME way every run, on every launch geometry.
Since the re-baseline is planned anyway, the fixed-tree arithmetic becomes
*the* arithmetic. Rule for the whole pipeline: **no atomic reductions
anywhere in the resolve; every reduction is a sort plus a segmented scan or
segmented reduce with a fixed tree.** This includes continuation-pool slot
allocation (a prefix sum over spawn flags — deterministic slot ids), which
the current `rs_alloc` atomic cannot give and which is the prime suspect for
§J's MD floor.

**2.3 One language.** The old system computed the same quantity in two places
four separate times (host reduction vs kernel scan, §0.1 rule 4), because the
sequential walk could not host the aggregation and the host could not host
the walk. A scan pipeline puts every pass in the kernel language, over the
same arrays, in one arithmetic. The torch-side alternative exists
(`torch.cumsum`/`scatter_add_`) but is non-reproducible in f32 and pays f64
bandwidth for the workaround — so the scans are Taichi kernels with fixed
tree order, full stop.

**Where scans do NOT pay, and the design does not force them:**

* **Sheet-level passes.** A pixel has at most `K` sheets (§4.6, K ≈ 16–32).
  Fragment-level passes (compaction inputs) are scan-shaped because fragment
  counts per pixel are unbounded and wildly variable — that variability is
  exactly what made the old walk's worst case bad. Sheet-level passes are
  bounded and small; a per-pixel loop over ≤ K records is coherent, cheap,
  and simpler than a scan. The rule is: **no unbounded per-thread loops**,
  not "no loops".
* **Early termination.** A sequential walk stops at the first opaque
  fragment; a scan computes prefixes past saturation. The waste is bounded
  by K sheet records per pixel (not by fragment count — the emission's
  opaque truncation still discards occluded fragments), and it buys the
  load-balance: no warp ever stalls on one deep pixel.
* **Feedback rules.** The old rule B (redistributing clamped write residue
  onto the run's unowned samples) is a sequential feedback loop. It is not
  ported. Its job — keeping occlusion exact when a sub-pixel sheet's area
  exceeds its sample ownership — is done structurally by §4.4's sibling
  compositing; if a residual rule is ever measured necessary again, it runs
  as a separate second pass, not as walk state.

**The sequential reference stays — as the oracle.** A plain, readable,
per-pixel sequential implementation of §4's semantics (unbounded, no budget)
is kept as the verification arm: the scan pipeline must match it exactly
wherever fixed-tree and sequential rounding agree, and to within one ulp of
reassociation where they do not. That comparison is a harness, not a shipping
path.


================================================================================
3. THE PIPELINE
================================================================================

Per frame batch, after projection and merge, replacing the current per-pixel
fragment walk. Passes marked [scan] are fragment-sized and stream-parallel;
passes marked [pixel] are bounded per-pixel loops over ≤ K sheets.

    P0  emission (kept)      raster COUNT/WRITE -> fragment records
                             (pixel, depth, mesh, facing, region, exact area,
                             sample mask, shading ref), sorted by
                             (pixel, depth). The exact clippers, the analytic
                             bezier coverage, and the opaque truncation all
                             survive unchanged.
    P1  band assignment      re-sort fragment keys by (pixel, mesh, facing,
        [scan]               depth); adjacent-gap test marks band starts;
                             segmented prefix sum numbers the bands. §4.2.
    P2  sheet compaction     segmented reduce keyed (pixel, mesh, facing,
        [scan]               band, region): area += (exact, fixed tree),
                             mask |= , depth = min, shading ref = argmax
                             area. Output: the sheet stream + per-pixel CSR.
    P3  depth order          per-pixel sort of ≤ K sheet records by depth.
        [pixel]
    P4  transmittance        per-sample prefix product over the pixel's
        [pixel]              sheets front to back; per-sheet visible weight;
                             residual background weight. §4.3–§4.5.
    P5  shading              flat kernel over the sheet stream, sorted by
        [scan]               material pipeline id for dispatch coherence;
                             one evaluation per sheet (§4.7). Writes each
                             sheet's radiance.
    P6  composite            weights x radiance summed into pixels by
        [scan]               segmented reduce (no atomics); background weight
                             shades the final sheet — constant color and env
                             map are the same code path (§4.5). Output is
                             LINEAR HDR; tonemapping is post (§4.8).
    P7  continuations        spawn flags -> prefix-sum slot allocation ->
        [scan]               compact continuation stream for the bounce loop;
                             shadow events taken directly from sheet records
                             (§4.9). No second walk exists to desynchronize.

Cost shape: one extra fragment sort (P1) beyond today's one, two segmented
fragment passes (P1, P2), and everything after P2 runs on sheet-sized streams
— which on diced geometry are 3–10x smaller than fragment streams, and it is
the SHADING that shrinks by that factor (P5 replaces per-fragment shading).
The bet, to be measured in Phase 1 (§8): the sort + scans cost less than the
per-fragment shading and divergent walk they delete. Radix sorts of
million-scale keys are milliseconds on this class of GPU, and the memory
model measures whatever the truth is (`memory_model.py` needs no annotation).


================================================================================
4. SEMANTICS
================================================================================

4.1 The sheet key
------------------
`(pixel, mesh id, facing, depth band, region class)`.

* **mesh id** is the render-level surface identity — one id for one authored
  surface (the `ALGAN_MESH_ID=1` semantics of the old design, §2.2 there),
  not one per triangle. Prerequisite carried over: faces must wind
  consistently so the facing bit names a sheet (`DESIGN_mesh_identity.md`
  §6.5 — `ALGAN_POLYHEDRON_WINDING` becomes default-on and non-optional).
* **facing** separates a closed surface's near and far sheets.
* **depth band** (§4.2) separates a concave surface's multiple same-facing
  sheets — the case that broke the "areas partition the pixel" assumption in
  the old lane reduction (two adjacent front sheets, measured: 82k corrupted
  run starts).
* **region class** distinguishes shading regions of one primitive that tile
  it: a circuit's fill vs its inward border. They compact into SIBLING sheets
  of one band (§4.4), which is what deletes the seam-grouping machinery: the
  fill/border seam cannot double-blend when the two regions' exact areas are
  disjoint by construction and composite additively.

4.2 Depth bands, without a world-space epsilon
-----------------------------------------------
Within `(pixel, mesh, facing)`, fragments sorted by depth split into a new
band where the gap to the previous fragment exceeds a RELATIVE threshold: the
gap is compared against the running band's own depth extent and the
fragment's own primitive scale (both available from the record), never
against an absolute constant. The old system's absolute epsilons
(`MIN_HIT_DISTANCE`, `TRIANGLE_EDGE_EPSILON`) are exactly what the mesh
identity project existed to retire; this design does not add one back. The
precise rule is the design's one open parameter (§10.1) and Phase 1 measures
candidate rules against the harness scenes — the fallback (band = facing
alternation only, i.e. at most one band per facing) is the old system's
behavior and is already acceptable there.

4.3 Coverage and occlusion across bands
----------------------------------------
Front to back over a pixel's depth-sorted sheets, per-sample transmittance
`T[s]` starts at 1 and each sheet multiplies in a factor — a prefix product,
which is what makes it scannable. A sheet's claim is

    Q      = popcount(mask) / N            (its owned-sample fraction)
    corr   = min(area, 1) / max(Q, 1/N)    (exact area vs sampled ownership)
    a[s]   = alpha * slot[s] * corr        (per owned sample)
    weight = sum_s T[s] * a[s] / N         (what it paints)
    T[s]  *= (1 - a[s]) + a[s] * ts        (ts = transmission share)

This is the old system's corrected write with the one difference that matters:
`area` is the sheet's WHOLE exact area, always, because compaction had no
budget. There is no truncated-sum arm, no cap substitution, no dust-band
special case for "the sum should have been 1 but the scan stopped" — a full
interior tiling produces `area == 1`, `Q == 1`, `corr == 1` natively.
Intra-sheet coverage is exact; inter-sheet occlusion is resolved at N sample
positions — that split is deliberate (§6.1).

4.4 Within a band: siblings composite additively
-------------------------------------------------
Sheets sharing `(pixel, mesh, facing, band)` but different region class
(fill/border), and any same-band tiles the compaction chose not to merge,
have DISJOINT exact areas that sum to the band's coverage. They composite
additively against the same incoming `T` (no mutual occlusion), and the band
occludes deeper sheets once, by its summed claim. This replaces both the seam
grouping (aa_grp 1) and rule B's redistribution: a sub-pixel rod whose area
exceeds its sample ownership no longer needs a residue pushed onto unowned
samples, because the band's total claim is bounded by its total area, which
the additive rule respects.

**BUILT 2026-08-20, after the shading-class split (§10.5) made the first real
siblings and shipped without this.** Until then no band held more than one
sheet, so §4.4 was unexercised prose and the resolve walked every sheet as an
independent occluder. Split crease faces walked that way occlude each other:
the near sibling's write dims the samples the far one reads, and a DONOR
sibling (exact area, no samples of its own -- the ordinary shape of a crease
pixel where one face is a corner sliver) is treated as a uniform veil and
claims a few percent of its area instead of all of it. Measured on
`solids_and_camera`'s Icosahedron: a crease pixel's two front faces claimed
0.9346 + 0.0043 of it instead of 0.9346 + 0.0654, and the 6% deficit was
filled by the solid's own BACK faces, whose Phong highlight is over twice as
bright as the front -- the bright seam along every interior edge.

The implementation keeps the walk and gives it the band. The compaction bands
and ranks CLASS-BLIND (a band is the sheet the split-off compaction would
build) and subdivides it afterwards, so each sibling can be handed the band's
sample union and `p_i = corr * share_i`, its share of the band's per-sample
coverage factor. Every sibling but the one that closes the band in walk order
carries `p_i` NEGATED: the sign is the flag that tells the resolve the band
continues, so it claims against the undimmed visibility and defers the write.
The resolve sums `p_i` in a register and writes once, at the closing sibling,
with `corr` and the material alpha -- exactly the write the unsplit band
made. Committed coverage is therefore independent of the split at any alpha,
which is what the compaction suite pins (donor, corr > 1, partitioned and
mismatched samples, alpha 1/0.6/0.25). Areal bands (empty union, or a
fragment sliver) are position-less and stay whole. A band another surface
interleaves at a coincident depth closes early and its remainder composites
sheet by sheet, as it did before the split; nothing reorders the walk for a
band, because pulling a sibling forward past a nearer sheet of another
surface flips which face paints the pixel (measured, 90 channel values on
`tests/fast`).

4.5 The background is the final sheet
--------------------------------------
Area 1, depth infinity, claim = residual `T`. The resolve emits, per pixel,
a background weight (and the primary ray direction it already has); a flat
background stage shades `weight x background(ray)` where `background` is the
constant color, the image plate, or the environment map — the resolve neither
knows nor cares. This deletes the `env_active` gate: an env-mapped scene runs
the identical resolve, and the empty-pixel fast path degrades gracefully
(weight 1, one env sample) instead of forcing a different renderer.

4.6 K and overflow
-------------------
`K` sheets per pixel (start at 24; measure). Compaction overflow merges the
farthest sheets into the last slot (conservative: summed area clamped,
nearest depth kept) and COUNTS it; the count is reported per batch like the
pool retries are today. Never a silent truncation — §Y's "an instrument that
reports zero may not be looking" applies to overflow policies too.

4.7 Shading per sheet
----------------------
One material evaluation per sheet at its dominant fragment's geometry
(barycentrics, interpolated normal, UV), area-weighted variants available
per material where one sample is measurably insufficient (high-frequency
textures). Sub-pixel shading variation WITHIN a sheet is precisely what AA is
licensed to average; the win is that a diced sphere's 13-fragment pixel
shades once per visible sheet instead of 13 times. P5 sorts the sheet stream
by material pipeline id, which is the dispatch coherence the old
sorted-material experiment wanted and could not afford per fragment
(memory: sorted dispatch was 1.5–2.2x SLOWER at fragment granularity).

**MEASURED GAP (2026-08-19, found post-flip): the license does not extend
across a hard crease.** Two flat-shaded faces of one solid meeting inside a
pixel share every component of the sheet key (same mesh, same facing, no
depth gap, masks partition), so they fuse and the pixel takes the DOMINANT
face's color outright — a winner-take-all staircase along every interior
(non-silhouette) edge of a lit flat-shaded mesh, where the fragment walk
blended per fragment by exact area (visible as jagged interior edges on
`solids_and_camera`'s Platonic solids against the pre-sheet CPU baselines).
The fix is `SHEET_SHADE_SPLIT` (§10.5): a SHADING CLASS in the compaction's
group key — a flat-shaded triangle's quantized unit face normal (declared,
or the geometric fallback `_triangle_normal` substitutes for all-zero vertex
normals, the Polyhedron family), class 0 for smooth-shaded triangles — so
crease faces become §4.4 siblings and each shades with its own normal.
Smooth diced geometry keeps class 0: the shading-shrink win is untouched.

4.8 Tonemap out of the resolve
-------------------------------
P6 writes linear HDR. Tonemapping (and bloom, FXAA, anything spatial) is
post-processing, where the shipped pipeline already prefers it
(tonemap-last). The `_get_tonemap_t_val() == 3` condition disappears from
path selection; a non-default tonemap changes post, not the resolve.

4.9 Continuations, shadows, and the identity items that stay live
------------------------------------------------------------------
Reflective/refractive sheets spawn continuations weighted by their visible
claim, into slots assigned by prefix sum (deterministic). Shadow queries are
built FROM sheet records — position from the dominant fragment, the sheet's
mask for sub-pixel sampling — so the "second walk in lockstep" hazard is
structurally gone. The two live items from the old queue port directly:

* **Nested-IOR refraction** (`DESIGN_mesh_identity_open.md` §H): the IOR
  stack rides the continuation exactly as designed there; nothing about it
  was run-rule-dependent.
* **Self-shadow rejection by identity** (§I): the shadow ray carries its
  source SHEET's mesh id, and the acceptance rule "reject same-mesh hits
  only at near-zero t" replaces the absolute `MIN_HIT_DISTANCE` epsilon —
  now easier, because the source mesh id is already on the sheet record the
  event is built from.


================================================================================
5. PATH UNIFICATION
================================================================================

There is ONE resolve. The current dense/sparse fork survives only below the
waterline, as two STORAGE strategies for P0's output (per-tile z + CSR when
coverage is dense; compact covered-pixel CSR when sparse) — both feed the
same P1–P7. The aa_grp ladder (0–6), `_aa_group_dense`, the pinned-level
plumbing, and the settings that select among them collapse to one boolean:
analytic AA is on when the raster front-end is. `use_raster` itself shrinks
to the structural facts (SPP == 1, no custom per-ray scatter; textured
batches stay — the per-sheet shading calls the same material pipeline the
wavefront textured shader uses). A batch the front-end rejects falls back to
the classic wavefront path WITHOUT analytic AA, and that is the only
configuration in which AA quality may differ — not env maps, not tonemaps,
not empty-pixel optimizations.


================================================================================
6. HONEST LIMITS
================================================================================

6.1 Inter-sheet overlap is sampled at N positions. Intra-sheet coverage is
    exact — and the record says intra-sheet is where every measured defect
    lived (notches, truncation, wobble). Silhouette-against-silhouette error
    is bounded by contrast/N and was never the measured problem; N is cheap
    to raise per-sheet (8 -> 16) if it ever is.

    6.1.1 INTERPENETRATION is the one place that bound does not hold, and it
    is a DEPTH limit rather than a coverage one. Two opaque surfaces crossing
    inside one pixel both claim exact area 1 and the full sample union; a
    sheet carries one scalar depth and the walk order is fixed on the host,
    so the whole pixel goes to whichever sheet sorts first. There is no
    per-sample depth anywhere downstream of emission — not even per fragment,
    whose one depth is evaluated at the centroid of the samples it owns — so
    nothing can blend the crossing. Blending it needs a depth plane per sheet
    (a min/max or a gradient) and a per-sample tie-break in the resolve;
    `OX_SHEET_INTERPENETRATION_AUDIT.md` scopes that, and it is NOT built.

    What IS repaired (2026-08-23, `SHEET_POSITIONED_DEPTH`, default on) is
    which surface the whole-pixel decision picks. The depth was the nearest
    fragment of any kind, AREA DONORS included — fragments that carry exact
    area but own no sample, which §4.4 elsewhere calls position-less. A donor
    at the leading corner of a pixel therefore carried its whole surface in
    front of one that was nearer at every sample the resolve compares them
    at. Measured on `solids_and_camera`'s axis triad (an `Arrow3D` shaft
    buried in a `Dot3D`): a 0.017-area donor at t=7.49997 beat the sphere's
    0.54-area fragment at t=7.50552 and took the pixel, where a supersampled
    reference gives it ~100% sphere. Ordering on the nearest POSITIONED
    fragment leaves the disjoint-range case untouched and makes the crossing
    pixel land on the majority surface — a z-buffer's answer, not an
    antialiased one.
6.2 The band rule is a new heuristic (relative, but a heuristic). Its failure
    mode is a concave surface's two sheets fusing (over-claims coverage) or
    one sheet splitting (composites additively — benign by §4.4). Phase 1
    measures it on the icosahedron/torus cases that broke the old
    assumptions.
6.3 Scan waste past saturation is bounded by K, paid for load balance.
6.4 One additional fragment sort per batch, plus scan passes. To be measured
    against the deleted per-fragment shading before Phase 2 ships anything.
6.5 Mesh ids must be stable across a batch's frames for banding and identity
    to mean anything; the row-mapping subtleties (`_tri_obj_row`) get simpler
    — the sheet key carries the id explicitly instead of re-deriving it per
    read — but the merge must keep ids frame-stable, which it already does
    for everything but adaptive re-dicing (level changes keep the SURFACE id
    stable, which is all the key uses).
6.6 SPP > 1 (the Monte Carlo megakernel) is out of scope: it supersamples
    everything and needs no analytic AA.


================================================================================
7. WHAT GETS DELETED
================================================================================

For the avoidance of nostalgia, the machinery this design retires outright:
`_aa_run_scan` and `_AA_MAX_RUN_SCAN`; the truncation probe; the §6.7 lanes,
their host segment reduction, `--verify-lanes` and the f32/f64 dual-language
question; §6.8's `frag_cap` substitution; ~~the one-mesh bit, cap and ink
accounting (§6.6, subsumed by per-sheet claims)~~ — **CORRECTED by Phase 2's
measurement: the subsumption is FALSE.** With the cap gone the coarse
Cylinder's far sheet re-claims the corr residue (`corr < 1` scales the
occlusion write, the residual transmittance on owned samples has no
position, the far sheet of the same solid claims it) and ink wobble
regresses 2–4x, including 0.000 → 0.032 at the exact-fit angles. The
one-mesh ceiling therefore SURVIVES — as data on the sheet record
(`sheet_cap`, the host's f64-reduced `max(front, back)`), clamped in the
resolve's bounded per-pixel loop with occlusion scaled alongside the claim
(§6.6.2's completion). What §7 does retire of §6.6 is its *walk* costume:
the per-fragment clamp interleaved with run state. Continuing the list: run
rules A and B **as walk state** and the redistribution residue (rule B's
arithmetic survives per-record, sheet-local, no pending flag); the run
engagement gate (`uni_v`) and every behavior that flowed from
`svis`-uniformity (including the fade-out interaction §6.7.3 diagnosed);
seam grouping as a special mode (aa_grp 1 -> §4.4 siblings); the aa_grp
ladder and `_aa_group_dense`; the emission/resolve level-pinning contract;
the `env_active` and tonemap conditions in path selection; the lockstep
shadow-event walk; and every atomic reduction in the resolve, `rs_alloc`
and `pix_accum` splat order included.


================================================================================
8. MIGRATION AND ARBITRATION
================================================================================

Instruments first, then phases, each phase shippable and measured. The
arbiters are the exact-reference harnesses and LOSSLESS diffs; committed
mp4 baselines are regenerated once, at the end, frames reviewed.

**Phase 0 — instruments.**
* Extend `_order_window_check.py` (§J) with an env-mapped scene and a
  non-default-tonemap scene — combinations the old gates made untestable.
  Its levers (KBUF, order, tiles, windows) must stay byte-inert through
  every later phase, and its MD-resolution run-to-run floor is the number
  §2.2's no-atomics rule predicts will go to zero — that prediction is
  falsifiable and should be checked explicitly.
* Adopt lossless A/B as the only moved-pixel instrument
  (`DESIGN_mesh_identity.md` §6.7.3's recipe).

  **BUILT (2026-08-19).** `_order_window_check.py` now renders every arm
  lossless (`libx264rgb`, crf 0), and gained `env_*` arms (the base scene
  under a deterministic float-tensor gradient environment map — no file
  dependency) and `tm_*` arms (the base scene with `post_process_tonemap`
  off, which is the toggle the route gate actually reads; the public
  `tonemapping` flag does not change path selection). Each scene family
  gets its own noise floor. First readings on CUDA at LD: env noise floor
  **1** channel value (the known `pix_accum` split-pixel atomic cap — the
  population §2.2 predicts goes to 0), tonemap-in-kernel floor **0**.

**Phase 1 — compaction beside the old walk.** Build P1–P2 and a harness that
feeds sheets back through the EXISTING walk as synthesized one-fragment-per-
sheet lists. Score against the current system on `_aa_run_gate_check`
(coverage error must strictly improve — truncation is gone),
`_aa_line_check` (ink wobble must not regress), and the six scenes lossless.
This phase decides the band rule (§4.2) and measures the sort/scan cost
(§6.4) before anything is deleted.

**BUILT AND MEASURED (2026-08-19).** `sheets.compact_sheets` (host torch:
three stable argsorts + segmented reductions, areas f64-accumulated and
rounded per §6.6.4's measured pattern; the §10.4 kernel-scan question stays
open for the shipping path). `_aa_run_gate_check --sheets` feeds one
synthesized fragment per sheet through the verified replay of the existing
walk; a torus (concave fold) case was added. Results at md on CUDA
(`benchmarks/_sheet_phase1_md.log`):

* **Coverage error ties or improves on every case**: the line-check rod
  0.0050 → 0.0011 (−78%), sphere 0.0057 → 0.0040, torus 0.0065 → 0.0048,
  every flat case ties exactly. `on-lattice` collapses to ~0 everywhere,
  and shipped `split`/`capped` populations cease to exist.
* **Interior notch depth drops 7.5x** on the rod (253 @ mean 0.0090 →
  339 @ 0.0012). The higher *count* is the notch counter's 1e-3 threshold
  on pixels whose TRUE coverage is 0.999x: the sheets paint the exact area
  where the walk rounded to 1.0 — verified per pixel (truth 0.99920,
  sheet 0.99875, walk 0.99966).
* **The band rule (§4.2/§10.1) is DECIDED: `prim`, c = 2** — split where
  the depth gap exceeds 2x the two fragments' own scales (triangle
  camera-distance extent + one pixel's world size at that depth; no
  absolute constant). All four candidate rules are output-
  indistinguishable on every case (identical painted deviation even on
  the reference-dropped fold pixels), so the decision rests on §6.2's
  asymmetry: fusing over-claims, splitting is benign, and c = 2 fuses
  least (torus fold: 686 vs 1773 for facing-only) with zero false splits
  on every non-fold case. Fold-tangency fusion is IRREDUCIBLE for any
  gap rule — projection stops being injective at a fold — so the fusion
  detector's baseline is nonzero on curved geometry by nature.
* **The six real scenes** (`_sheet_scene_stats.py`, streams certified by
  a byte-identical lossless render against the committed CUDA baseline):
  S/F compaction ratio 0.39 (text_and_media, the glTF scene — a 2.6x
  shading shrink) to 1.00 (manim_compat, pure circuits); **max sheets
  per pixel is 6** across all six scenes against the design's K = 24,
  overflow population zero. Host compaction cost ~50 ms per ~300k-
  fragment batch (~2–3% of these renders' wall) — indicative only; §6.4's
  real gate is Phase 2's A/B.
* `_aa_line_check`'s ink-wobble gate needs a real render path and moves
  to Phase 2's checklist; the line-check cases' coverage columns above
  are its Phase-1 proxy.

**Phase 2 — the sheet resolve behind a flag.** P3–P6 as the scan pipeline,
with the sequential reference implementation (§2's oracle) beside it and a
parity harness between them. The old walk remains the default.

**BUILT AND MEASURED (2026-08-19).** `ALGAN_SHEET_RESOLVE` (default off;
`SETTINGS.raytracing.experimental.set(sheet_resolve=...)`) routes the sparse
covered-pixel path through `sheet_resolve_taichi.sheet_resolve_shade`: one
thread per covered pixel over its depth-sorted sheets, shading once per
sheet at the dominant fragment, per-sheet `corr` where the run rule's state
machine used to be, rule-B redistribution collapsed to sheet-local
arithmetic, and the walk's material split / continuations / ray-state
contract unchanged (Phase 4 rebuilds those from sheet records). Compaction
runs in `prepare_sparse_raster_coverage` (prim, c = 2); the §6.7 run lanes
are skipped as genuinely subsumed. Shadowed batches keep the walk until
Phase 4. Results:

* **Determinism**: both arms byte-identical across repeated runs (A/A
  max|d| 0), engagement asserted (kernel launch counters).
* **Oracle parity**: `sheets.resolve_pixel_reference` (the §2 oracle) vs
  the kernel's `ALGAN_AA_DUMP` rows — worst |claim| diff **8.94e-08** over
  56 committed sheet rows at 24 probed pixels.
* **Ink wobble** (`_aa_line_check`, md, daemon-free A/B): bezier and flat
  quad identical to four decimals; the coarse Cylinder's mean over
  non-degenerate angles **0.0159 → 0.0050 (−69%)** and the fine rod
  **0.0442 → 0.0130 (−71%)**, worst angle 0.0878 → 0.0165. The two
  degenerate axis-aligned rod angles move +0.0006/+0.0010 (values ~0.01).
* **The §7 one-mesh correction was found here**: the first build deleted
  the cap and the coarse Cylinder regressed 2–4x; restored as sheet data
  the regression became the −69% above. §7 carries the full note.
* **Off-vs-on output** (LD parity scene, lossless): max|d| 65 over ~0.4%
  of pixels, reviewed as a worst-frame panel — movement confined to diced
  silhouettes/rims plus faint interior dusting on lit curved meshes
  (per-sheet shading), no notches, seams or structural change.
* **Two A/B traps hit and recorded**: a warm auto-daemon served both arms
  of an env-var A/B from its own environment (rows identical to the
  digit — always disable the daemon in BOTH arms), and an edit to a
  `*_taichi.py` while its first run was live invalidated that run (killed,
  re-run clean).
* Flag off is byte-identical by construction (new kernel, untouched walk);
  `pytest -q --fast` 213 passed.

**Phase 3 — unification.** Dense storage feeds the same compaction;
background-as-sheet lands; tonemap un-fuses; the env/tonemap gates and the
ladder die. §J's extended harness gates this phase.

**BUILT AND MEASURED (2026-08-19), for the sheet route.** The `env_active`
and tonemap conditions are gone from the sparse gate when the route is on:

* **Background-as-final-sheet (§4.5)**: `env_background_prefill` writes
  `env(ray)` per (frame, pixel) into the frame buffer, the resolve hands
  its leftover weight to the composite (`env_in_composite`, pinned at
  emission), and empty pixels are final with no resolve launch. One bug
  found by the parity panel: frame-buffer column 3 is the GLOW lane, and
  prefilling it with 255 bloomed every pixel white (max|d| 222 over the
  whole frame) — the sky emits glow 0, matching the dense retire.
* **Tonemap out of the resolve (§4.8)**: the resolve stays linear;
  `wf_composite_accum_sparse` gained the tonemapping template (3 = the
  historical exact path), and `wf_finalize_uncovered` pays `finalize(bg)`
  for untouched pixels under an in-kernel tonemap — including whole empty
  frames (it runs even when the emission found nothing).
* The route decision is computed ONCE (`sheet_route` in
  `render_batch_raytraced`) and passed down; the emission raises on a
  compaction refusal when the relaxation was load-bearing.
* **Measured** (parity harness, env and tm scene variants, LD lossless):
  engagement asserted both ways; **the env scene's run-to-run noise floor
  drops from 1 (dense route's `pix_accum` atomic) to exactly 0 on the
  sheet route** — the first §2.2 determinism prediction to land. The
  off-vs-on diffs are the resolve change's own signature (rims and
  reflection interiors), reviewed as panels. The §J env/tm lever arms
  under the flag: tiles and windows byte-inert at the floor (env batches
  7→4 and 7→23 — the levers demonstrably moved); `KBUF 8` moves **one
  pixel in 9.2M pixel-frames by |d| 4**, sheet-route-specific (the old
  dense route is KBUF-inert on the same scene, and the arm's own A/A is
  clean). The mechanism sits in the classic bounce loop's KBUF-refill
  batching, reached with different (valid) continuation inputs; it is an
  OPEN ITEM for Phase 4's §J gate, where P7 reworks that hand-off anyway.
* Dense storage feeding the compaction is NOT built: the sparse route now
  serves every configuration the sheet resolve accepts, so the dense path
  survives only as the non-sheet fallback until Phase 4 settles its fate.

**Phase 4 — identity features and the flip.** Shadow events and
continuations from sheet records (P7); §H and §I built on top; the old walk
and its machinery (§7's list) deleted; ONE re-baseline, lossless-reviewed,
CUDA on the reference box and the CPU set with it (which also settles the
standing CPU-baseline debt, `DESIGN_mesh_identity_open.md` §B).

**Phase 4 RECORD (2026-08-19), with three deviations stated plainly:**

* **P7 (deterministic continuation slots) is DEFERRED, by measurement.**
  The criterion it exists for — run-to-run determinism exactly zero — was
  measured MET without it: the §J scene's noise floor is 0 on the sheet
  route at LD and MD (the fragment walk's was 1), shadowed scenes
  included. `rs_alloc` and the bounce iterations' `pix_accum` adds remain
  the resolve's only atomics, both OUTSIDE iteration 0's per-pixel
  single-writer structure; if a scene ever measures a floor again, P7's
  shape is recorded in the worklog history (count-pass template + host
  int-scan + exact-slot emit).
* **§H (nested IOR) and §I (self-shadow by identity) are built AFTER the
  flip**, not before it, as separate reviewed features: each moves output
  in a small population of its own (nested-glass interfaces; shadow-acne
  pixels), each now has `sheet_sid`-carrying records to build on, and
  bundling them into the flip's re-baseline would have hidden their
  movement inside a 5-scene diff. They remain the top of the queue.
* **Two defects the flip's own review caught before the baselines were
  written** — the ones the phased discipline exists for:
  1. Self-overlapping same-mesh translucent geometry (a mid-morph
     tetrahedron) FUSED into one sheet and attenuated once where a ray
     crosses the surface twice. Depth banding cannot split a zero-gap
     overlap; the FILL RULE can — a band holding a sample bit twice holds
     two sheets by definition. Each fragment's mask-CONFLICT RANK (integer
     cumsums, deterministic) is now part of the sheet key; overlapping
     layers attenuate per layer, tilings are untouched, and fold fusions
     vanish with it, making `sheet_fused` a true invariant.
  2. Two generic triangle mobs shared one merge-level surface id
     (`TriangleTriangulated` + `QuadTriangulated`); they now declare
     one surface per mob, the Polyhedron pattern.
* **The re-baseline**: five of six full-render scenes and the fast scene
  moved; `manim_compat_and_plots` (pure circuits) is byte-identical, the
  mechanism confirming itself. Every scene's worst frame was reviewed as
  a panel: movement is mesh silhouettes/interior edges, ~1px shadow
  boundaries, reflection rims, and the minified-texture ImageMob (§10.5's
  anticipated case). Both CUDA sets were regenerated on this box under the
  render-twice discipline. The CPU sets are intentionally left stale: the
  project owner regenerates them on the CI machine (decided mid-flip,
  2026-08-19), which also keeps the CPU baselines the property of the
  lineage that reproduces them (`DESIGN_mesh_identity.md` §3.5's standing
  rule).

**Phase 4a BUILT AND MEASURED (2026-08-19): shadow events from sheet
records.** The resolve kernel gained a `mode` template — 1 walks the
transport and writes one candidate event per accepted lit triangle sheet
(no shading, no spawns), 2 shades reading the traced visibility — so the
event pass and the shading pass are the SAME kernel body and a
resolve/shadow desync is structurally impossible (the machinery the old
lockstep pair needed, `_aa_dump_check`'s lockstep leg included, has
nothing left to check on this route). Event identity is the SHEET INDEX
into dense per-sheet tables: no counter, no atomic reserve; the host
compacts accepted rows (deterministic ascending order) and the unchanged
`raster_shadow_trace` runs over the compact queue. `sheet_route` no
longer requires shadows off.

* **A shadowed scene's A/A is now byte-identical in BOTH arms** (the
  fragment walk's was too on this scene; the sheet route keeps it while
  deleting the lockstep).
* The remaining off-vs-on movement is the walk's dark same-id seam leak
  REMOVED (the walk under-tiles a quad diagonal whose two triangles it
  treats as separate engagement contexts — sheets paint the exact
  tiling), shadow-boundary shifts of ~a pixel (per-sheet dominant-
  fragment event positions), and mirror-rim reflection shifts. Reviewed
  as panels and per-pixel dumps, not just measured.
* **One band-rule defect found by the shadow panel and fixed with a
  test**: the `prim` scale first used the triangle's RAW camera-distance
  extent, and a large wall's extent (~4 units) swamped a 1.04 gap in
  front of it — a same-sid quad+backdrop pair FUSED into one sheet and
  shaded with the backdrop's color (a bright line where they overlapped
  on screen). The scale is now the per-PIXEL depth slope (extent /
  projected pixel size from `tri_screen`, straddlers keep the
  conservative extent). The ink-wobble table is unchanged by the fix.
* **Identity finding for the §4.1 sweep**: two DIFFERENT generic
  triangle mobs (`TriangleTriangulated`, `QuadTriangulated`) shared one
  surface id in the merge. The old run rule had the same exposure
  (bounded by per-fragment shading); sheets amplify shared ids into
  shared shading, so generic triangle mobs deserve a declared
  `mesh_key` before the flip.

**Success criteria**, stated before the work: coverage error on the harness
cases strictly better than shipped; ink wobble no worse; §J extended
byte-inert on every lever including env/tonemap scenes; run-to-run
determinism at LD, MD and PREVIEW exactly zero; one compiled resolve per
AA on/off; and the fade-out/despawn scenario that closed the old design
(`shapes_and_timeline`'s dot clouds) rendering with no engagement-dependent
behavior — its truncated-run population simply ceases to exist as a concept.

**Phase 4 DELETION DONE (2026-08-19).** The fragment walk and its machinery
are gone: `raster_first_shade`, `raster_shadow_event_build`, the z-prepass
kernels, the run scan (`_aa_run_scan`, `_AA_MAX_RUN_SCAN`), the run lanes and
their host reduction, `raster_iteration_zero` and the dense tile plumbing in
`_run_wavefront_tiles` (retired-empty pix_accum prefill, tile-empty/covered
composite modes), and the settings `ANALYTIC_AA_RUN_EXACT` /
`ANALYTIC_AA_RUN_CAP` / `ANALYTIC_AA_ONE_MESH_DENS` (their semantics are
inherent in per-sheet claim arithmetic). The route collapsed with it:
`analytic_raster_route_active` now carries every sheet-route precondition
(SHEET_RESOLVE, ANALYTIC_AA_RUN, the three sparse toggles, and the
transparent-background relaxation — allowed except combined with an env map,
§5's fallback), so `sheet_route == analytic_raster`, `sparse_coverage ==
use_raster`, and the raise pair (tracer's route check, prepare's emission
check, the resolve's sheets check) guards drift. KEPT, per the inventory: the
emission kernels and their exact-area lane, `_aa_group`/`_aa_run_full`
(emission truncation), `ANALYTIC_AA_ONE_MESH` + the host cap reduction
(feeds `sheet_cap`), `ANALYTIC_AA_RUN_FULL`, and `raster_shadow_trace`.
Byte-identical at defaults (deleted code was unreachable post-flip),
validated against the fresh CUDA baselines.


================================================================================
9. RELATION TO THE OLD QUEUES
================================================================================

`DESIGN_mesh_identity_open.md`: §C (run-scan limit) — closed by this design;
its C.4 "flip work" is superseded by Phase 4's single re-baseline. §H, §I —
live, port per §4.9. §B (CPU baselines) — absorbed into Phase 4. §E/§F/§J —
instruments, kept; §J gains Phase 0's scenes. §G (TLAS/BLAS), §K, §L —
untouched by this design; revisit after Phase 4 on their own merits.


================================================================================
10. OPEN QUESTIONS
================================================================================

10.1 The band-gap rule's exact form (relative to band extent vs primitive
     scale vs both; hysteresis or not). Phase 1 owns it.
10.2 N (samples) and K (sheets): 8 and 24 to start; both are compile-time
     and cheap to sweep once the harnesses exist.
10.3 Whether P3/P4 as per-pixel bounded loops ever need the scan treatment
     (only if K grows or profiling says the sheet passes are the bottleneck).
10.4 Segmented-scan implementation shape in Taichi (two-pass blocked scan vs
     sort-order exploitation) — an implementation question, not a semantic
     one, provided the tree order is fixed.

     **PARTLY ANSWERED (2026-08-20) for the reductions that are not scans.**
     The compaction's per-sample-lane passes moved to kernels
     (``sheet_compact_taichi.sheet_band_reduce`` / ``mask_popcount``,
     ``SHEET_MASK_KERNEL``, default on): the mask union, the §6.2 fusion
     detector, the popcount and the exact-area sum are one pass each instead
     of eight -- the area rode into the same kernel because it walks the
     identical stream -- worth 1.26x on ``compact_sheets``, 445 -> 352 ms on a
     4K frame. See DESIGN_optimization_targets.md T5 for the numbers and for
     the measurement trap in the gather beside them.

     The mask passes are REDUCTIONS, not scans, so §2.2's "no atomic
     reductions" rule is not what governs them: it exists because a float sum
     reassociates, and these are integer OR/max, exact under any order. The
     fusion detector stays order-independent through ``atomic_or``'s return
     value rather than through a fixed tree -- a lane claimed k times is
     observed already-set by exactly k-1 fragments however the atomics
     serialize.

     The area sum is the one float reduction and it does NOT get that
     guarantee. It keeps §6.6.4's float64 accumulator, now widened in a
     register off an f32 read rather than from an f64 copy of the fragment
     array. Two measurements stand behind it: the kernel agrees with the torch
     ``scatter_add_`` bitwise at a 4K frame's shapes and with 4096 addends in
     one bin, and repeated runs agree with each other. Narrowing it to f32 was
     considered and rejected on measurement -- 81% of a real frame's sheets
     hold one fragment and 17% hold two, both order-independent at any width,
     but the remaining 1.6% run to eleven and ``sheet_cov`` feeds thresholds,
     for 0.2% of a frame.

     **ANSWERED (2026-08-21) for the scan too, and the answer is that it
     needed neither shape this question offered.** The conflict-rank
     exclusive-prefix-sum per lane -- eight ``torch.cumsum`` passes plus a
     per-lane ``index_select``, ``maximum`` and two ``where``s, five live
     ``[n]`` arrays -- is ``sheet_compact_taichi.sheet_conflict_rank`` now
     (``SHEET_RANK_KERNEL``, default on). No two-pass blocked scan and no
     sort-order exploitation: the premise does not hold here, because the
     bands are already CONTIGUOUS runs of the sorted stream and there is
     nothing left to segment. One thread per band walks its own fragments
     forward with the eight per-lane counters in registers, reading each
     before that fragment's own increment -- which is the exclusive prefix --
     and no lane's counter is touched by another band's walk.

     Bit-identity here is by construction rather than by argument: both arms
     are integer and visit the stream in the same order, so §2.2's fixed-tree
     rule has nothing to bind. That is firmer footing than the mask
     reductions above, which need the ``atomic_or``-return argument.

     **What the shape assumes -- measured, not asserted.** The walk is serial
     WITHIN a band, so load balance is bounded by the longest band. On a real
     1920x1080 frame of the ``benchmarks/_sheet_kernel_ab.py`` scene (976,231
     fragments) there are 877,047 bands: mean 1.11 fragments, median 1, p99 2,
     **maximum 15**, 90.0% holding one and 9.2% two. A band cannot be long for
     the same reason a sheet cannot -- it is the fragments of one surface, one
     facing, in ONE pixel. A future band rule that produced a band holding a
     large fraction of the stream would make this the wrong shape; the parity
     harness covers that case for correctness (one band, whole stream) and
     nothing covers it for speed.

     Measured in a real render, on CPU (a 4-vCPU cloud container, so a CPU
     number and nothing else -- CUDA is unmeasured): at 1920x1080 the scan is
     one call per frame and goes **33 ms -> 6 ms**, taking ``compact_sheets``
     from 480 ms to 458 ms. A note for whoever re-measures it: substituting a
     ``torch.randperm`` for the real ``order`` reads 4.0x instead of 7.7x on
     the same captured tensors, because the side that goes scattered is the
     KERNEL's own ``msk[order[j]]`` gather (the real order's mean
     ``|order[j+1] - order[j]|`` is 1.9). That is the mirror image of the
     gather trap DESIGN_optimization_targets.md T5 records -- it understates
     instead of inverting -- but a synthetic permutation still gets the
     number wrong.
10.5 Whether any material class measurably needs multi-sample sheet shading
     (§4.7) beyond the dominant fragment.

     **ANSWERED (2026-08-19): it is not a material-class question but a
     normal-discontinuity one, and the remedy is a finer sheet key, not
     multi-sample shading.** Interior crease edges of lit flat-shaded meshes
     lost their AA to single-point sheet shading (§4.7's measured-gap note).
     Built as `ALGAN_SHEET_SHADE_SPLIT` /
     `SETTINGS.raytracing.experimental.sheet_shade_split`, DEFAULT OFF:
     `sheets._shade_class` adds a flat-face shading class to the group key
     (quantized unit face normal, 64 bins/component; class 0 for smooth
     triangles; the zero-normal geometric fallback mirrored from
     `_triangle_normal`), splitting crease pixels into §4.4 siblings that
     shade separately and blend by exact area. Measured
     (`benchmarks/_sheet_shade_split_check.py`, CUDA, LD, daemon-free,
     linear output):

     * Lit Icosahedron: +1131 sheets (27,503 fragments), 983 px moved,
       worst |d| 98, movement confined to the solid — the OFF panel shows
       the facet-edge staircase, the ON panel clean blends, silhouettes
       identical. A/A byte-identical in both arms.
     * Tent (one mob, two planar faces, 3° screen-slope ridge): edge-position
       wobble 0.149 → 0.086 px RMS. Instructive detail: 197 of 242 crease
       pixels were ALREADY two sheets in the OFF arm — the emission dilates
       coverage across shared edges, so fold fragments' masks conflict and
       the rank key splits them (their areas sum past 1: 0.529 + 0.578 on a
       probed pixel). The class split fixes the 45 genuinely fused ones and
       makes the split independent of where a sample happens to land in the
       overlap band.
     * `_aa_line_check --res md`, both arms daemon-free: identical to four
       decimals on every case (bezier, flat quad, both cylinders) — the
       line scenes carry no hard crease, and smooth geometry is class 0 by
       construction. No ink-wobble regression.
     * Flag off is bit-identical by construction (the class multiplies into
       the group key only inside the `shade_split` branch); `pytest -q
       --fast` 213 passed including the pixel-compared render against the
       committed CUDA baseline, and the compaction unit suite pins the
       crease/smooth/coplanar/zero-normal class semantics.

     Flipping the default moves every lit crease edge (toward the pre-sheet
     fragment-walk appearance) and is therefore a re-baseline of the scenes
     carrying flat-shaded solids; it should ride the next planned
     re-baseline. Residual after the split, quantified by the tent's
     0.086 px (not 0.02): the seam-overlap dilation means sibling areas are
     not an exact partition, and a `corr > 1` sibling's rule-B residue
     lands on its co-sibling's samples — bounded, sub-pixel, and a property
     of the emission's seam handling rather than of this split.

     **CORRECTED 2026-08-20.** That last sentence read the symptom as a
     bound. The residue landing on a co-sibling is not sub-pixel noise: it
     is the band claiming less coverage than it has, and the same
     mechanism -- siblings walked as independent occluders -- costs a donor
     sibling nearly its whole claim. What the deficit admits is whatever
     lies behind the crease, which on a closed solid is its own back faces.
     `solids_and_camera`'s Icosahedron rendered a BRIGHT SEAM along its
     interior edges from exactly that. Fixed by building §4.4 (see there):
     the band's sheets now claim additively and occlude once. Measured
     after: of the 9,895 pixels the fix moves across the scene, the ones
     brighter than both their neighbours -- the seam signature -- fall from
     522 to 88, the tent's wobble win is unchanged (0.1487 → 0.0836 px RMS,
     +45 sheets), and the split-off path stays byte-identical.
