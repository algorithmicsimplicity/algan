# Algan — Analytic Anti-Aliasing v2: Exact Uncontended Coverage

Status: DESIGN, nothing built. This is the plan of record for the next round of
analytic-AA work. It supersedes the ss21 exploration in `DESIGN_analytic_aa.md`
as a plan, but not as a record: every negative result this document leans on is
measured there, and section references of the form "ss__" point into that file.

The one-sentence version: keep the shipped 8-sample fill-rule masks as the
ATOMIC OWNERSHIP substrate for everything contended, and layer EXACT AREAS on
top of them in the two places where nothing is contended — per-RUN magnitude
correction for triangle meshes, and a correctly ORIENTED two-wall model for
bezier circuit corners and stems. Ownership stays sampled; magnitude becomes
exact exactly where "exact" is well-defined.

Reading order: §1–§3 are the requirements and the argument; §4 is the triangle
design; §5 is the circuit design; §6–§8 gating, validation, phasing; §9 risks.
§4 and §5 are independent workstreams touching disjoint code.


================================================================================
1. REQUIREMENTS, AND THE RELAXATION THAT CHANGES THE PROBLEM
================================================================================

Set by the project owner, 2026-08-12, in priority order:

  R1  Surface (triangle-mesh) boundaries have EXACT analytic coverage when
      uncontended — one surface's edge over whatever is behind it, no other
      partial geometry in the pixel. Continuous gradation, not 1/8 steps.
  R2  Bezier circuit boundaries have exact analytic coverage when uncontended —
      INCLUDING corners and sub-pixel stems, which the shipped model gets wrong
      (§5.1). Long straight boundaries are already exact (ss21.2).
  R3  Interior edges of a mesh never seam. Non-negotiable; not an "overlap"
      case even though two fragments share the pixel, because they tile it.
  R4  STACKS of partially covering fragments — crossing silhouettes,
      interpenetration, translucent self-overlap — may be APPROXIMATED. But
      gracefully: ss18's measured 3x-worse-than-aliased interpenetration
      failure is not an acceptable approximation, it is a bug. "Approximate"
      means "at least as good as what ships today".
  R5  As fast as possible. The fully-covered hot path must not pay; the shipped
      speed wins over supersampling (1.27x on meshes, parity on reflective,
      ss16.5/ss19.4) must survive.
  R6  Kill switch: everything off is byte-identical, proven by the ss21.5 hash
      discipline (SHA256 of the fast render vs baseline AND vs a stashed-HEAD
      render).

The relaxation in R4 is what makes this solvable. The rule that killed three
exact-coverage attempts — a fragment's CLAIM and its OCCLUSION must be the same
quantity, and only atomic sub-pixel ownership guarantees it (ss21.10) — is a
constraint about CONTENDED pixels. It only has teeth when several fragments
fight over sub-pixel real estate, which is exactly the case R4 just released.
On an uncontended boundary there is no ownership question at all: within one
sheet of one surface, exact clipped areas sum exactly over a tiling (property 3
of `benchmarks/_aa_clip_area_check.py`, verified to 1.19e-07), so the surface's
coverage is a well-defined scalar and the only defect in the shipped render is
that the scalar is quantized to eighths. A scalar deficiency gets a scalar fix.

Why the three prior attempts do not satisfy these requirements, in one line
each (details in ss21.3/21.8/21.9):

  * Per-fragment reconciliation (exact area vs sample mask, ss21.3): breaks
    claim==occlusion at every boundary pixel; 5920 notches. The clamp lives on
    in `_coverage_density` and is INERT while cov is mask-derived; it must
    never become load-bearing.
  * Cells (ss21.8): a cell is not atomic; two triangles splitting one cell
    composite to 1-a*b where the answer is 1. 6942 notches.
  * Scalar surface accounting (ss21.9): removes sampled ownership EVERYWHERE,
    including the contended cases R4 still wants handled gracefully — it
    re-inherits ss18's interpenetration failure by construction, strands
    `_sec_positions` and the shadow walk (both think in masks), and its
    accounting bug was never isolated. Parked, not fixed (§6.3).

ss21.8's closing paragraph already named the shape of the fix: "a
representation that is atomic AND finite-area ... it would need the fill rule
kept". This document is that fourth attempt, realized at RUN level — the level
at which areas provably sum — rather than at cell level, where they do not.


================================================================================
2. INVENTORY: WHERE EXACTNESS STANDS TODAY
================================================================================

    boundary                              today                       verdict
    --------------------------------------------------------------------------
    triangle mesh silhouette              8-sample mask, 9 levels     §4 fixes
    triangle interior shared edge         exact partition (ss15)      keep
    triangle thinner than sample spacing  dropped; thin ink 0.855     §4.5 fixes
    circuit long straight boundary        exact half-plane (ss21.2)   keep
    circuit corner                        vertex-gradient rounding    §5 fixes
    circuit stem (two walls in a pixel)   half-plane over-coverage,   §5 fixes
                                          text -6.8% vs box filter
    circuit border inner edge             own box filter (ss13.4),    keep (§5.4)
                                          single-boundary model
    stacks / interpenetration             per-sample svis (ss18)      keep — this
                                                                      IS the R4
                                                                      approximation
    camera-plane straddlers               per-sample raycast masks    keep
                                          (ss19.1)
    reflected/refracted image content     N continuations (ss17)      out of scope

The per-sample transmittance resolve (ss18) is the best-measured mechanism this
codebase has for contended pixels — it beat the group rule, beat aliased, and
beats aa=2 on translucent configs. Nothing in this plan weakens it; §4 only
rescales what it already computes.


================================================================================
3. THE DECOMPOSITION
================================================================================

A fragment's coverage answers two questions (ss21.1):

    WHERE in the pixel is it    -> set question   -> answered by the mask,
                                                     atomic under the fill rule
    HOW MUCH of the pixel       -> measure question -> answered by a real number

The shipped system uses the mask for both, which quantizes HOW MUCH to eighths.
ss21.3 tried to give HOW MUCH its own exact number per fragment and broke WHERE.
The v2 rule is that the exact number exists per fragment but is only ever READ
at run level:

    Ownership, occlusion, ordering, continuations, shadows:  the mask. Always.
    Magnitude:  the sum of exact areas over a RUN — a maximal consecutive
                group of fragments from one surface and one sheet — applied as
                a single scalar correction to that run, and ONLY when the run
                began uncontended.

"Began uncontended" has a precise, cheap definition: every `svis` entry equal
(§4.2). That predicate is R1/R2's "no overlap" — and where it is false, the
correction silently does not fire and the shipped machinery runs bit-for-bit.
Failure of the gate degrades to today's render, never below it.


================================================================================
4. TRIANGLES: RUN-CORRECTED COVERAGE
================================================================================

4.1 Fragment payload and emission
---------------------------------
`_ss_pixel` (raster_taichi.py) keeps the fixed-point edge functions, the
top-left fill rule, the owned-sample centroid, and the backface bit exactly as
shipped. Changes, all under a new representation value (§6.1):

  * `frag_cov` carries the EXACT clipped area from `_pixel_clip_area` instead
    of `popcount(mask)/N`. The two lanes already exist and are already read
    together by the resolve (`frag_cov[idx]` / `frag_msk[idx]`); no new
    per-fragment storage, no packing games — the ss21.9 trap of packing areas
    into the mask lane (bit 16 collides with `_AA_BACKFACE_BIT`) stays dead.
  * SNAP TO 1.0: when the mask is full and every snapped vertex distance
    clears the pixel's half-diagonal, emit exactly 1.0 (and in any case snap
    `cov >= 0.9999` to 1.0 on a full mask). Interior fragments must stay
    bit-clean: the resolve's `cov < AA_FULL_COVERAGE` gates (centroid ray
    rebuild, `_tri_surface_point`) and the z-prepass acceptance all key on
    full coverage, and a 0.99999997 from a boundary integral would flip them.
  * The sampled claim needs no lane: the resolve derives it as
    `popcount(msk)/N`.
  * Acceptance widens from `mask != 0` to `mask != 0 or cov > 0` — a sample-less
    triangle (sliver) is now EMITTED, with an empty mask, its exact area, and
    its depth/barycentrics evaluated at `_pixel_clip_centroid` (both primitives
    exist and are validated standalone; the centroid of triangle∩pixel is
    inside the triangle, so the ss15 mis-sort argument is unchanged). The
    lattice-degenerate case (`area2 == 0` on the 1/4096 grid) may also emit if
    its float clip area is positive; it is an area donor like any sliver.
  * The `ANALYTIC_AA_SLIVER` policy knob is INERT in this representation:
    sliver behavior is fixed by §4.5, not configurable. The knob keeps meaning
    what it means for the point representation.

4.2 The run rule in the resolve
-------------------------------
One rule, added to both walk kernels (`raster_first_shade` and
`raster_shadow_event_build` — §4.6), gated so the hot path never sees it.

Definitions, all per pixel, inside the existing `while q < total` walk:

  * sid(fragment) = `tri_obj[ref]` for a triangle, `-1 - circuit` for a bezier
    fragment. `tri_obj` is built unconditionally at merge
    (scene_builder.py: `repeat_interleave` over the per-primitive blocks) and
    threaded to both kernels; ss21.9 rebuilt it and this design is why it stays.
  * A RUN is a maximal consecutive subsequence of the depth-sorted fragment
    list sharing (sid, facing bit). The z-winner and every circuit fragment
    have a full mask and terminate runs; they are never corrected themselves
    (a circuit's scalar coverage is already exact for its own boundary).
  * UNIFORM(v): all `_AA_NUM_SAMPLES` entries of `svis` equal v. Eight
    compares at run start. v > 0 required.

At the first fragment of a run, if the fragment's mask is partial AND svis is
uniform at v, LOOKAHEAD over the run (fragments are CSR-indexed; `idx =
start + q` random-accesses them — the lookahead is index arithmetic plus loads
of `frag_ref/frag_cov/frag_msk` and `tri_obj[ref]`):

    E     = sum of frag_cov over the run          (exact, includes slivers)
    U     = OR of masks over the run              (within a sheet the fill rule
                                                   makes these disjoint, but OR
                                                   is robust to mis-sorts)
    Q     = popcount(U) / N
    corr  = 1                     if U == _AA_MASK_ALL   (tiling; also kills
                                                          float dust in E)
          = E / Q  clamped to [0.5, 2]  otherwise        (|E-Q| is bounded by
                                                          the union's
                                                          quantization, so the
                                                          clamp is paranoia,
                                                          not policy)
    stop the scan at a (sid, facing) change, at the z-winner, or at a cap of
    MAX_RUN_SCAN fragments (beyond the cap: corr = 1 for the remainder —
    shipped behavior, graceful).

Then commit the run's fragments as today, with two multiplications by corr:

  * the claim: `eff` (formed at the single site that feeds color, glow, tint
    weight, shadow events, and continuation energy) is scaled by corr;
  * the occlusion: the per-sample attenuation the fragment writes into `svis`
    is scaled by corr, subject to §4.4.

Invariants the implementation must hold, and the dump harness (§7.1) checks:

    committed run total  = v * E * (run's alpha profile)
    sum(svis)/N after    = v * (1 - E * abar)     for a uniform-alpha run
    every claim >= 0, every svis in [0, 1], pixel totals conserve energy

The background needs no separate correction: the walk already finishes with
`vis_all += svis[s]`, so once the run's svis writes are corrected, the
leftover the composite hands to the background (or to the environment map) is
the corrected quantity for free. This is the same single-choke-point argument
ss6 item 4 made for coverage-as-alpha, and it is why the correction is two
multiplies and not a re-plumbing.

Why the gate makes the failure modes structurally unreachable:

  * A FULLY COVERED pixel: first fragment has a full mask -> no lookahead, no
    correction, bit-identical. The ss5 lattice cannot come back through a code
    path that full pixels never execute.
  * An INTERIOR-EDGE pixel (two fragments tiling it): lookahead runs, U is
    full, corr = 1 exactly. R3 holds by construction, not by tuning.
  * The BACK SHEET at a silhouette: it is a different run (facing differs).
    Its samples are occluded and svis is no longer uniform -> uncorrected ->
    shipped ~0 contribution. The exact area of the back sheet — which covers
    the SAME sub-region as the front sheet, the trap ss21.9's max() rule
    existed for — is never consulted.
  * A SECOND OBJECT behind a partial edge: svis non-uniform -> uncorrected ->
    shipped quantized claim of the remaining samples. That is R4's accepted
    approximation, and it is bounded by the quantization we already ship.
  * A stack of FULL-COVERage translucent fragments (every fade transition):
    full masks with uniform per-sample writes PRESERVE uniformity, so each
    surface's own silhouette pixels inside a fade still get corrected. The
    chromatic tint path also preserves uniformity: `trans_share` is a
    per-fragment scalar applied identically to every owned sample, and the
    owned set is all of them.
  * INTERPENETRATING translucent meshes: their sheets interleave, runs break,
    the gate closes, and the resolve is bit-for-bit ss18's — the configuration
    that mechanism was built for and measured on (0.112 vs aliased 0.187).
  * A run SPLIT by the ss16.6 depth mis-sort (near-tangent triangles): the
    second partial run of the same surface starts non-uniform -> uncorrected.
    No new failure; the 8-17 residual notches neither grow nor shrink.

4.3 A worked example
--------------------
Opaque sphere edge over background, true coverage 0.370. Today: the front
sheet's fragments own 3 of 8 samples, the pixel reads 0.375, and neighboring
pixels along the edge step by eighths. With the run rule: E = 0.370 (the
front-sheet areas tile up to the silhouette line), Q = 0.375, corr = 0.9867;
the run claims 0.370, svis ends at mean 0.630, the background gets 0.630, and
the gradation along the edge is continuous. The back sheet's run follows,
finds its three samples at zero, contributes nothing — same as today. If a
second sphere sits behind the edge, its run starts non-uniform and claims the
sampled remainder, exactly as today: the FRONT edge is still exact, and only
the sliver of error the quantization already made moves to the occludee.

4.4 The corr > 1 case — the one open empirical question
-------------------------------------------------------
corr < 1 (surface covers less than its samples suggest) is clean: owned
samples keep a positive remainder `v*(1 - corr)` under scaled writes, which
reads as "the over-quantized part of the edge lets the background through at
the owned samples". Position approximate, magnitude exact, everything in
range.

corr > 1 (surface covers MORE than its sample share; the mirror case of the
same quantization) cannot be expressed on the owned samples alone once they
hit zero. Two candidate rules, both bounded by |E - Q| <= ~1/(2N) per
boundary:

    RULE A (scale-and-clamp): scale the svis writes by corr, clamp at zero.
    The claim is exact; the leftover under-shrinks by up to (E - Q)*v, i.e.
    the background keeps a residual of the error the claim shed. Simplest,
    strictly monotone per sample, never exceeds pre-run values.

    RULE B (redistribute): after the run, multiply the run's NON-owned
    samples by (1 - E*abar) / (1 - Q*abar), pushing the excess onto them.
    Leftover exact; individual unowned samples may exceed their pre-run value
    (never 1 when v = 1 fails only transiently in mixed-alpha runs — cap and
    fold the cap residue back into owned samples). Exact totals, slightly
    weirder per-sample semantics.

This is deliberately left as a measured decision, not a designed one: build
both behind the flag, read the dump (§7.1) on a hand-checked pixel, and let
`_aa_iter.py`'s notch/ink columns pick. Every prior round that guessed at a
resolve accounting detail lost a render cycle to it; the harness costs 12
seconds.

4.5 Slivers donate area
-----------------------
The ss16 finding stands: a sample-less triangle must never OCCLUDE, and giving
it a positional claim is noise. Under the run rule it does not need either:

  * In a corrected run, a sliver's `frag_cov` joins E and its eff is ZERO.
    Its area is real (it is part of the sheet's tiling — this is where the
    rim of a sub-pixel-diced PN surface lives), its color folds into its
    same-surface neighbors via corr, and nothing downstream can tell the
    difference because the neighbors are the same surface.
  * In an uncorrected context (non-uniform svis), a sliver contributes
    nothing — exactly today's `drop`.
  * The one new claiming case: a PRISTINE run with Q == 0 — every fragment a
    sliver, i.e. a rod thinner than the sample spacing crossing the pixel
    between samples. Claim `min(E, 1) * v`, distributed over the run's
    fragments by area, no svis writes (nothing is occluded that the samples
    can express; the leftover invariant carries the removal). This is the
    cells experiment's stranded gain (`thin` ink 0.855 -> 0.999, ss21.8)
    harvested without the halo, because the claim exists only when nothing
    else in the pixel has claimed and the sliver still never takes a sample
    from a neighbor.

Rods crossing OTHER geometry keep today's behavior (they vanish into it) —
accepted under R4.

4.6 The shadow walk
-------------------
`raster_shadow_event_build` replays the resolve's transport walk fragment for
fragment; ss6 step 5 and ss13.2 both flag it as the easiest place to introduce
a silent desync, and nothing has changed about that. The run scan and the corr
application must land in both kernels in the same change, ideally as one
shared `@ti.func` so they cannot drift (with the standing discipline: editing
a `@ti.func` does NOT invalidate the offline kernel cache — clear it before
every A/B, never edit `*_taichi.py` under a live render process).

4.7 Continuations, glossy, and the ray-cast fallback
----------------------------------------------------
Untouched, and that is a design feature, not an omission:

  * `_sec_positions` keeps reading the mask, so reflective/refractive
    continuation counts, the per-position gating (ss19.2), the pool sizing
    (`base * N`, ss19.2a) and the glossy stratification (ss20) are all
    unchanged. corr scales the ENERGY a continuation carries (it rides in the
    fragment's committed weights), not the ray count.
  * A Q == 0 claiming sliver run (§4.5) has no positions; if its material
    reflects, spawn ONE continuation at the clipped centroid via the existing
    single-ray path (which deliberately never perturbs a lone ray, ss20.3).
    Sub-pixel reflective rods are dim by construction (energy <= E*v); one
    ray is proportionate.
  * `_raycast_pixel` (camera-plane straddlers) already answers the mask
    question by definition — one ray per sample (ss19.1). It emits
    `cov = popcount/N`, so straddler fragments are self-consistent (E == Q,
    corr == 1) and correction is a no-op there. Carrying exact areas through
    the fallback is possible later but out of scope; a horizon-crossing
    ground plane keeps its current (sampled, already-shipping) quality.

4.8 Cost, and the 4-sample follow-up
------------------------------------
Where the cycles go:

  * Fully covered pixels: zero change (no lookahead, no clip area — the
    emission computes `_pixel_clip_area` only when the mask is partial; a
    full-mask fragment snaps to 1.0 via the distance test it already does).
  * Boundary fragments: one `_pixel_clip_area` in COUNT and WRITE (both
    recompute the intersection today; the area is a five-comparator sort
    network per edge — this is the cost the ss21.3/21.8 experiments already
    paid without complaint).
  * Boundary pixels: one lookahead re-reading the run's (ref, cov, msk) plus
    one `tri_obj` load per fragment, then the walk proceeds as before. Runs
    are short at silhouettes (a few fragments); the worst case is a densely
    tessellated or sub-pixel-diced mesh where MOST covered pixels contain an
    interior edge — there the lookahead prices every such pixel, and the
    `meshes` scene of `_analytic_aa_bez_ab.py` (two spheres + torus, 720p) is
    the gate: in-process alternating A/B, budget +-2% wall, with the
    occupancy counter checked (the resolve already sits at a 21-25% ceiling).
    If register pressure notches occupancy down, the ss5(c) two-kernel split
    (full-coverage pixels vs boundary pixels, via the existing covered_idx
    compaction) is the prepared fallback — do not fatten the monolith.
  * Register budget for the run state: sid (i32), face+uniform flags, v,
    E, Q/U, corr, scan-end — about seven registers. Same order as what ss18
    removed (`grp_absorb`, `grp_cov`, `grp_obj`, `occ_msk`).

The follow-up this unlocks, deliberately NOT in scope for the first landing:
once magnitude is exact, the mask's only job is arbitrating contended pixels,
and 4 samples is aa=2-grade arbitration. Halving `_AA_NUM_SAMPLES` halves the
svis loops in the occupancy-bound resolve and the per-sample loops in
count/write, and `_sec_positions` already thinks in 4 positions. Measure the
notch and interpenetration configs before believing it — and remember the
sample pattern is a compile-time constant with no template argument (the
documented cache trap): switching counts means editing the line and clearing
the cache, or promoting the count into the `aa` template value first.

4.9 What stays approximate, exhaustively
----------------------------------------
Crossing silhouettes of different objects in one pixel; interpenetration
curves; translucent back sheets at rims (the front sheet is corrected, the
back is sampled); same-surface runs split by depth mis-sorts; runs past
MAX_RUN_SCAN; straddler-fallback pixels; everything ss7/ss19.5 lists for
secondary content. All bounded by the eighth-quantization the shipped render
already has, all covered by existing configs, none regressed by the gate
construction.


================================================================================
5. CIRCUITS: THE ORIENTED WEDGE
================================================================================

5.1 The two failures, precisely
-------------------------------
The shipped exact filter (ss21.2, `_boundary_coverage` -> `_halfplane_clip_area`,
default ON) models the boundary near the pixel as ONE line: the level set of
the SDF, oriented by the closest-point vector `_bezier_point_metrics` returns.
Two situations put TWO boundary features in one pixel, and one line cannot say
so:

  * A CORNER. The closest point on the outline is the shared vertex, so the
    "normal" points from the query at the vertex — the true SDF gradient, but
    the level set there is the vertex's distance CIRCLE, so corners render
    rounded and over-covered. Sub-pixel error, but R2 names corners.
  * A STEM. A ~1px glyph stem has both walls inside the pixel; linearizing to
    the nearer wall reads the stem as solid past it. Measured: text L1 0.1081
    (box) -> 0.1155 (exact) at matched dilation — the exact FORMULA lost to
    the crude one because the MODEL was wrong (ss21.2).

The built fix, `ALGAN_ANALYTIC_AA_BEZ_WEDGE` (ss21.6, default OFF), has the
right area primitive (`_two_halfplane_area`, validated standalone to 0.0115)
and the wrong orientation: the crossing parity is a property of the QUERY, so
it orients only the NEAREST wall, and the second wall's inward side was
inferred by a handedness calibration `sign(cross(dir1, n1))` that needs n1 to
be a true wall normal. At a corner n1 points at the vertex, the sign is
arbitrary, and the second half-plane flips — a plain square measured 0.1093 ->
0.2467. Corners are the pixels the model exists for, so gating the calibration
to interior closest points is circular. The conclusion stands: orientation
must be a property of the EDGE, computed where the contour is known, not
recovered from the query.

5.2 Flatten-time inward sides
-----------------------------
`_build_circuit_geometry` (rendering/raytracing/primitives.py) packs
`_rt_edges = [x0, y0, x1, y1, border_visible]`, per frame, per flattened edge.
Add a sixth column, the INWARD SIGN sigma_e ∈ {+1, -1, 0}:

    sigma_e = +1  if the drawn (odd-parity) side of edge e's line is the side
                  its leftward perpendicular points to, -1 the other way,
                  0 for degenerate edges (edge_degenerate already exists).

Computed in batched torch alongside `border_visible`, per frame (a morph can
flip a contour's winding mid-animation, and `border_visible` is already
per-frame for the same reason):

  * probe = midpoint(e) + eps * leftward_normal(e), eps = 0.05 * |e|;
  * parity(probe) by even-odd crossing count of a +x ray against ALL edges of
    the same circuit — [E_c x E_c] comparisons per circuit per frame, batched;
    a glyph is a few hundred edges, so this is thousands of comparisons per
    glyph, trivial at torch batch granularity but MEASURED on a text-heavy
    scene before shipping (it lands in animate/prep, the pipeline stage the
    batching work just spent effort shrinking);
  * the definitional invariant doubles as the validity check: the two sides of
    an edge must have opposite parity. Where they do not (eps crossed another
    feature — sub-pixel stems make this reachable), halve eps and retry,
    bounded; a still-inconsistent edge gets sigma = 0 and the kernel falls
    back to the single half-plane for it.

Even-odd holes are handled by construction — parity IS the fill rule, so a
hole's contour gets inward signs pointing out of the hole regardless of its
winding, which is the property ss21.6 said "left of the direction" cannot
give. `bezier_acceleration.py` validates `edges_2d.shape[-1] >= 5`, so a sixth
column passes untouched; only `_bezier_point_metrics` learns to read it.

5.3 Kernel changes
------------------
`_bezier_point_metrics` already tracks the nearest and second-nearest
segments and returns both closest-point vectors and directions (11 values); it
additionally returns sigma1 and sigma2, read from column 5 at the two argmin
updates. Callers in the wavefront kernels take the widened tuple (an arity
change of this exact shape was part of ss21's byte-identical refactor; same
discipline, same hash proof).

In `_bez_pixel_hit`'s wedge branch (`int(aa) == 3`):

  * Wall normals come from storage: `n_i = sigma_i * perp_hat(e_i)`. The
    handedness calibration `hh` is DELETED — it was the bug.
  * Wall 1 also uses its stored normal IN WEDGE MODE (at a corner the stored
    normal is the true wall where the gradient points at the vertex). The
    single-half-plane path (`aa == 2`) keeps the parity-oriented gradient
    unchanged: for a lone plane at a vertex, tangent-to-the-distance-circle is
    the best one plane can do, and that path's numbers are the shipped
    baseline.
  * Signed distances to the wall LINES as ss21.6 built them:
    `sd_i = (outline_w - n_i . cp_i) * inv_px` — valid whether the closest
    point is interior or an endpoint, because the endpoint lies on the line.
  * INTERSECTION vs UNION is selected by parity agreement, which is what makes
    CONCAVE corners work (ss21.6 had intersection forms only): evaluate both
    predicted memberships at the pixel centre — inside(∩) = (sd1>0 and sd2>0),
    inside(∪) = (sd1>0 or sd2>0) — and pick the boolean form that matches the
    KNOWN crossing parity at that point. The parity is ground truth at one
    point; the model must interpolate through it. If both forms agree with it
    they locally coincide; if neither does (degenerate geometry), fall back to
    the single half-plane. Coverage is then `_two_halfplane_area(...)` for the
    intersection or `a1 + a2 - a∩` for the union.
  * Kept from ss21.6, they were correct: the trivial-containment short-circuit
    BEFORE any wedge math (an unrelated far segment's apex sits tens of pixels
    away and truncated rays miss the pixel — short-circuiting also bounds the
    apex to ~7px, which keeps the f32 shoelace accurate), and the `nd < 0.9`
    same-wall gate (the second-nearest segment being the next flattening chord
    of the SAME wall must fold into wall 1, not form a wedge with it).

5.4 Scope
---------
The wedge applies to the FILLED outer boundary — silhouettes, corners, stems —
which is where R2's failures live. Explicitly deferred, each with today's
behavior:

  * the border's inner edge (ss13.4's own box filter): single-boundary model;
    a border corner is one more refinement of the same shape once the outer
    wedge is proven;
  * unfilled circuits (bands): the near/far-wall band logic already models two
    walls of the SAME stroke; a bent stroke's corner keeps its current
    rounding;
  * anisotropy of `pixel_size` on tilted planes (ss4's caveat): orthogonal,
    unchanged.

Flattening tolerance (`ANALYTIC_AA_CHORD_TOLERANCE`) is unchanged: the wedge
consumes the same chords the SDF does.

5.5 Measurement discipline
--------------------------
In order, because the first item inverts the sign of everything after it:

  1. MATCH THE DILATION. The aa=4 reference dilates filled circuits by 0.15
     output px, the analytic arms by `ANALYTIC_AA_BEZ_MIN_HALF_WIDTH = 0.3`; a
     sharper filter amplifies that fixed offset in proportion to its slope, so
     the exact form scores WORSE for being MORE faithful unless the arms run
     at 0.15 (ss21.2: slant reads -0.6% at 0.3 and +8.0% at 0.15). Every wedge
     number is taken at matched dilation. (Whether to SHIP 0.15 is ss21.10
     item 3 — a stroke-weight appearance decision for a human with rendered
     Tex, out of scope here.)
  2. Standalone first: extend `benchmarks/_aa_wedge_check.py` with corner and
     CONCAVE-corner configurations and the ∩/∪ selection rule, and add the
     sigma invariant (probe parity flips across every non-degenerate edge) as
     a torch-level property test on real glyph contours — a Tex "o8B" line
     exercises holes and both winding conventions. ss21.10's lesson is
     verbatim: skipping the standalone harness for `_two_halfplane_area` cost
     two render cycles.
  3. `benchmarks/_aa_iter.py` gates (slant / stem / corner / glyph): the wedge
     must beat BOTH the box and the lone-exact arms on stem, corner and glyph,
     and must not regress slant (the ss21.6 failure signature) or border.
  4. Default flip of `ALGAN_ANALYTIC_AA_BEZ_WEDGE` on those gates plus the
     ss19 matrix unchanged-or-better, with the usual hash proof that
     wedge-off is byte-identical.


================================================================================
6. SETTINGS, TEMPLATES, CACHE DISCIPLINE
================================================================================

6.1 New surface
---------------
    ALGAN_ANALYTIC_AA_RUN (default 0)   Triangle run-corrected coverage: exact
                                        area emission + run rule + sliver
                                        donation. Subordinate to
                                        ALGAN_ANALYTIC_AA / _TRI.
    ALGAN_ANALYTIC_AA_BEZ_WEDGE         Existing flag, re-scoped to the
                                        ORIENTED wedge; flips default per §5.5.

Both live in `raytracing/settings.py` with setters beside `set_analytic_aa`,
read LIVE at call time (`rt_settings.X` — importing by value freezes them
before user code runs; that bug has shipped before).

Template encoding, because the offline cache serves stale kernels to anything
it cannot distinguish: the geometry kernels' `aa` value becomes
`1 + sliver_mode + 4 * repr` with repr 0 = points, 1 = the ss21.3 exact
emission (kept only as a measured negative), 2 = run-corrected; the resolve's
`aa_tri` gains value 3 = run (1 points, 2 cells). Every representation gets
its own compiled variant and cache entry, exactly the sliver-mode precedent.

6.2 Byte-identity matrix
------------------------
    ALGAN_ANALYTIC_AA_RUN=0, WEDGE=0     byte-identical to HEAD (SHA256, fast
                                         render, vs baseline AND stashed HEAD)
    ALGAN_ANALYTIC_AA=0                  byte-identical, as today
    RUN=1                                changes silhouette pixels only; an
                                         all-full-coverage frame is provably
                                         byte-identical (the gate never opens)

No test re-baseline while the new flags default OFF. The flip, when earned, is
a deliberate re-baseline commit with eyes on the diffs (standing practice).

6.3 Parked and deleted
----------------------
  * ss21.9's scalar surface accounting (`rem/sheet_cov/obj_cov/obj_absorb`,
    the `ALGAN_ANALYTIC_AA_EXACT_TRI` resolve branch) is PARKED: no further
    debugging. Once run mode passes its gates, delete it together with the
    cell machinery (`_unpack_cells`, `_AA_CELL_*`, `_coverage_slots`' cells
    branch, `_cell_clip_area`) in one commit. `tri_obj`, the facing bit, and
    the clip primitives (`_pixel_clip_area`, `_pixel_clip_centroid`,
    `_halfplane_clip_area`, `_two_halfplane_area`) all stay — they are this
    design's load-bearing parts.
  * `_coverage_density`'s reconciliation clamp stays inert forever; a comment
    should point here. Making it load-bearing is ss21.3 (5920 notches).


================================================================================
7. VALIDATION
================================================================================

7.1 The dump harness comes FIRST
--------------------------------
The ss21.9 postmortem, verbatim: six inference rounds found three real bugs
and did not find the one that mattered; THE NEXT STEP IS INSTRUMENTATION, NOT
ANOTHER GUESS. Before either feature lands:

  * `ALGAN_AA_DUMP="px,py,frame"` (debug-only, env-gated, compiled out
    otherwise) makes the resolve print, per fragment at that pixel: q, sid,
    facing, mask (hex), cov, popcount, corr, eff, and svis after commit; plus
    the pixel's final vis_all and committed color.
  * A host-side golden walk in the harness recomputes the same pixel from the
    dumped fragment inputs in numpy and diffs every column.
  * The same dump serves the shadow walk, which is how the §4.6 desync gets
    caught on day one instead of in a flickering render.

7.2 Gates, in run order
-----------------------
  1. Standalone primitives (`_aa_clip_area_check.py` already covers area +
     centroid; `_aa_wedge_check.py` extended per §5.5.2).
  2. Byte-identity matrix (§6.2).
  3. `benchmarks/_aa_iter.py` (12s/arm): mesh silhouette L1 must beat the
     8-sample floor (0.124 on the ss16.2 tri config; 16 samples' 0.114 is the
     number to beat from below — the run rule should land near the exact
     reference's own noise); thin ink 0.855 -> >= 0.99; interior notches <=
     the shipped 8-17 (target: no change — the mis-sort mechanism is
     explicitly out of scope); seam config unchanged; stem/corner/glyph per
     §5.5.3.
  4. `benchmarks/_aa_match_aa2.py`: the 8/11 must not shrink; mesh, text and
     thin should improve; spec/flat/glass are expected UNCHANGED (their
     shortfall is minified secondary content, ss19.5 — not coverage).
  5. `_analytic_aa_bez_ab.py meshes` in-process A/B: wall within +-2% of
     shipped analytic; `tracer._WAVEFRONT_POOL_RETRIES` unchanged (nothing
     here spawns rays, so a retry delta means a bug, ss19.2a).
  6. The seam scan (`_aa_seam_check` config), then the repo pixel suite, one
     process at a time on Windows.

7.3 Traps already known to invalidate measurements here
-------------------------------------------------------
Dilation matching (§5.5.1); the ti.func offline-cache staleness (clear before
every A/B); lockstep drift of a whole config = reference bug, diff the
reference image first (ss19.3); `shapes_and_timeline` in tests/full_renders
was already stale at HEAD (max dev 170, deterministic) and stays un-rebaselined
— do not attribute it to this work.


================================================================================
8. PHASING
================================================================================

  Phase A — Instrumentation. The §7.1 dump + golden walk; verify `tri_obj`
  maps diced-PN logical triangles to their source Surface (believed true from
  merge order; a one-line assert in the dump proves it). Read-only, no
  behavior change.

  BUILT (2026-08-13), and the assert DISPROVED the belief, twice over: the
  scene batcher merges every same-identifier mob into ONE collection primitive
  (two plain spheres shared sid 0), and a diced logical-PN row's patch moves
  from frame to frame with the adaptive levels, so a time-invariant per-part
  id cannot express the mapping at all. `tri_obj` is now built per MEMBER at
  pack time ([1, N] for flat collections, [T, N] for diced PN via the dice's
  own counts/offsets) and offset per primitive at merge — a primitive's kept
  and promoted slices share its offset, so constant-property promotion cannot
  split a surface in two. Proven by `benchmarks/_aa_dump_check.py` (golden
  walk to 4e-9, resolve/shadow lockstep, per-sphere sids), byte-identical by
  stashed-HEAD hash. `benchmarks/_aa_iter.py` was rebuilt (the ss21 original
  and its cached refs were lost to a truncated write); shipped-arm baselines
  at 320x180, refs aa=4: slant .1828/ink 1.014, stem .4011/1.315,
  corner .2623/1.038/12n, glyph .3008/1.366, mesh .0355/1.000, thin
  .0617/0.857 — and at the §5.5.1 matched dilation (0.15), exact vs box:
  slant .0347/.0386, stem .1255/.1264, corner .0680/.0750, glyph .1269/.1259.

  Phase B — The circuit wedge (§5). Self-contained: one host column, one
  metrics arity change, one kernel branch rewrite, no resolve involvement.
  Ships on its own gates; earliest visible win (R2).

  Phase C — Triangle emission (§4.1). Exact areas + sliver donation behind
  ALGAN_ANALYTIC_AA_RUN, resolve still reading masks only (corr hard-wired to
  1): proves the payload and the acceptance change in isolation,
  byte-identical by construction except sliver-donor fragments existing
  (accept-rule parity checked in the dump).

  Phase D — The run rule (§4.2) in BOTH walks, rules A and B for §4.4 behind
  a sub-toggle. The dump decides A/B; `_aa_iter` and the §7.2 ladder gate it.

  BUILT (2026-08-13). One shared scan (`_aa_run_scan`) and one shared
  corrected write (`_run_svis_write`) serve both walks; rules A/B ride in
  `aa_tri` as 3/4 (`ALGAN_ANALYTIC_AA_RUN_RULE`). Findings, all by harness:

  * §4.4 is DECIDED: rule B (redistribute). tri video L1 0.119 → 0.107 with
    edge levels 620 against the aa=4 reference's own 621 (R1's continuous
    gradation, measured); rule A reads 0.110/609. seam notches 6 → 9 (B)
    vs 12 (A), inside the documented 8–17 mis-sort band; trans 0.058 → 0.056;
    static mesh L1 0.0355 → 0.0292. The golden walk reproduces the corrected
    kernels to 2.5e-8 and the two walks stay in lockstep.
  * The designed corr clamp [0.5, 2] was WRONG and is replaced by the tiling
    bound `corr = min(E, 1) / Q`: a sub-pixel rod that owns one sample but
    covers several samples' worth of area needs corr well above 2 (the §4.2
    bound argument covers one silhouette boundary, not a strip), E above 1 is
    a mis-scan and is capped, and rule B keeps the occlusion side exact under
    large corr where rule A would leak (E - Q) · v as double-counted light.
  * The §7.2 thin target (ink ≥ 0.99) is UNREACHABLE within this design's own
    safety rules, and the dump shows why precisely: at a 0.22 px closed tube
    the front-sheet run works perfectly (six donors + one owner, claim = E
    exactly), but HALF the band's ink is carried by the tube's back-facing
    wall — genuinely visible geometry at sub-sample scale, not the redundant
    back sheet §4.2 assumes — whose run correctly starts non-uniform because
    the sheets do overlap at the owned sample. Its signature is identical to
    a thick surface's occluded back sheet, so any positional claim is the
    measured halo catastrophe (ss16.2). Rods ≳ 0.45 px sit at or above
    parity; the 0.22 px tube keeps its front half. thin ink lands at 0.884
    (from 0.857); the video thin config is at parity (L1 0.097 → 0.098). The
    0.99 number was calibrated on the cells experiment, whose non-atomic
    accounting is exactly what v2 rejects; a future two-sided-visibility
    rule is the only sound route past it.
  * The `_analytic_aa_bez_check` triangle "shipped" arms had been silently
    measuring the PARKED cells mode (`exact_tri` coupled to the circuit
    exact arm — trans read 1.449, the documented cells breakage). Decoupled;
    every number above is against the true points baseline.
  * COST, on the named worst case (the sub-pixel-diced `meshes` A/B, 720p):
    raster kernels +27% device (count 0.088→0.094, write 0.082→0.121,
    resolve 0.056→0.070 warm seconds over 8 frames) ≈ +6.6% of frame device;
    fragments +8.7% (donors). The ±2% sub-budget is MISSED there and met
    nowhere near it on ordinary content (the clip work prices only stored
    boundary fragments); R5's actual bar — the 1.27x win over aa=2 supersampling
    must survive — holds at ~1.19x worst-case. §4.8's costing was wrong in one
    place: the point representation never clipped, so "the cost the ss21.3/21.8
    experiments already paid" was not in the shipped baseline. What was tried
    and measured: per-candidate clipping (count +36%/write +42%), post-cull
    recompute in WRITE (worse — divergence + duplicated setup, write +91%),
    and the shipped compromise — COUNT/Z never clip (sampled keep decisions,
    identical branches in WRITE), WRITE clips stored fragments cheapest-first
    (fully-inside shoelace / one-cutting-edge closed form / full clip), donors
    behind a one-sided oriented reject with a moment-free clip in COUNT. The
    residual is register pressure from the clip code's presence, not executed
    instructions; the prepared follow-ups are a dedicated exact-lane post-pass
    over the stored stream and §4.8's 4-sample arbitration.

  Phase E — Flip decisions: BEZ_WEDGE default per §5.5.4; ANALYTIC_AA_RUN
  default on its gates; delete the parked code (§6.3); re-baseline with eyes
  on diffs.

  Phase F (optional, separately justified) — the 4-sample arbitration
  experiment (§4.8), only after E has soaked.


================================================================================
9. RISKS AND OPEN QUESTIONS
================================================================================

  * §4.4 (corr > 1) is the one place this design admits it does not know the
    answer. Bounded either way; decided by harness, not argument.
  * Resolve occupancy: ~7 registers of run state on a kernel at a 21-25%
    ceiling. Mitigation prepared (two-kernel split, ss5(c)); measured, not
    assumed.
  * Lookahead bandwidth on densely tessellated meshes — the one place R5 is
    genuinely at risk; the `meshes` A/B is the gate, and MAX_RUN_SCAN bounds
    the worst case.
  * f32 area sums over long runs: error ~1e-6 per fragment, three orders below
    the 1/255 output quantum at any plausible run length; the U == ALL
    short-circuit removes the only case where "exactly 1" matters.
  * The sigma parity probe on text-heavy scenes lands in CPU prep; measure on
    a Tex-dense scene and cache per (circuit, frame) — it depends on nothing
    else.
  * Both features change rendered output when ON; nothing here is an
    optimization with a parity proof. The kill-switch matrix (§6.2) is the
    contract, and the flip is a reviewed re-baseline like ss13's.
  * The dense route shares `raster_first_shade` (both call sites), so the run
    rule covers it automatically — but its z-prepass feeds the walk a
    full-coverage winner with prefilled mask, which the gate must keep
    treating as run-terminating (the prefill is 0xFF in the low bits and does
    NOT set `_AA_BACKFACE_BIT`; the ss21.9 all-ones trap applied to the
    32-bit cell packing, not to this).


================================================================================
10. ANTI-GOALS
================================================================================

Inherited unchanged from v1 ss12 (no wavefront coverage, no post-process-only
AA, no temporal AA, no per-sample shading, `anti_alias_level` keeps working),
plus, hard-won here:

  * NO per-fragment reconciliation of exact area against the mask. The clamp
    in `_coverage_density` stays inert. (ss21.3)
  * NO fractional sub-pixel ownership — cells or otherwise. Atomic samples
    own; scalars only ever rescale a run. (ss21.8)
  * NO mask-free resolve. The mask is what keeps R4 graceful and what
    continuations and shadows consume. (ss21.9, ss18)
  * NO mesh silhouette extraction / adjacency subsystem. The tiling property
    of exact areas already reassembles a surface without one.
  * NO query-derived contour orientation. Inward sides are edge data, written
    at flatten time where the contour is known. (ss21.6)
