# Algan — the sheet resolve: analytic AA as data, not control flow

**STATUS: design, plan of record for the resolve redesign. Nothing here is
built.** Decided 2026-08-19 with the project owner: the renderer is being
redesigned, output WILL move, and the committed baselines will be regenerated
once at the end — so this document optimizes for the right system, not for
byte-compatibility with the current one.

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
question; §6.8's `frag_cap` substitution; the one-mesh bit, cap and ink
accounting (§6.6, subsumed by per-sheet claims); run rules A and B and the
redistribution residue; the run engagement gate (`uni_v`) and every behavior
that flowed from `svis`-uniformity (including the fade-out interaction
§6.7.3 diagnosed); seam grouping as a special mode (aa_grp 1 -> §4.4
siblings); the aa_grp ladder and `_aa_group_dense`; the emission/resolve
level-pinning contract; the `env_active` and tonemap conditions in path
selection; the lockstep shadow-event walk; and every atomic reduction in the
resolve, `rs_alloc` and `pix_accum` splat order included.


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

**Phase 1 — compaction beside the old walk.** Build P1–P2 and a harness that
feeds sheets back through the EXISTING walk as synthesized one-fragment-per-
sheet lists. Score against the current system on `_aa_run_gate_check`
(coverage error must strictly improve — truncation is gone),
`_aa_line_check` (ink wobble must not regress), and the six scenes lossless.
This phase decides the band rule (§4.2) and measures the sort/scan cost
(§6.4) before anything is deleted.

**Phase 2 — the sheet resolve behind a flag.** P3–P6 as the scan pipeline,
with the sequential reference implementation (§2's oracle) beside it and a
parity harness between them. The old walk remains the default.

**Phase 3 — unification.** Dense storage feeds the same compaction;
background-as-sheet lands; tonemap un-fuses; the env/tonemap gates and the
ladder die. §J's extended harness gates this phase.

**Phase 4 — identity features and the flip.** Shadow events and
continuations from sheet records (P7); §H and §I built on top; the old walk
and its machinery (§7's list) deleted; ONE re-baseline, lossless-reviewed,
CUDA on the reference box and the CPU set with it (which also settles the
standing CPU-baseline debt, `DESIGN_mesh_identity_open.md` §B).

**Success criteria**, stated before the work: coverage error on the harness
cases strictly better than shipped; ink wobble no worse; §J extended
byte-inert on every lever including env/tonemap scenes; run-to-run
determinism at LD, MD and PREVIEW exactly zero; one compiled resolve per
AA on/off; and the fade-out/despawn scenario that closed the old design
(`shapes_and_timeline`'s dot clouds) rendering with no engagement-dependent
behavior — its truncated-run population simply ceases to exist as a concept.


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
10.5 Whether any material class measurably needs multi-sample sheet shading
     (§4.7) beyond the dominant fragment.
