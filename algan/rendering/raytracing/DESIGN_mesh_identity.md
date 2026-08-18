# Algan — Mesh Identity in the Triangle Renderer

**Status: PARTLY LANDED. This file is the handoff document — start here.**
**One limitation ships knowingly: §0.5.**

Plan of record for replacing the renderer's epsilon-based seam heuristics with
declared mesh identity. Written to be self-contained: a fresh session with only
this file and the repo should be able to continue without reconstructing any of
the reasoning.

Reading order. §0 is the state of the branch and what to do next. **§0.5 is a
known limitation that ships in the default renderer** — diagnosed, costed and
deliberately unfixed, so read it before treating a diced mesh's interior pixel as
a bug. §1–§2 are the problem and what shipped. §3 is the unstarted work with the anchors to do it. §4
is what needs a CUDA device and the experiment for each claim. §5 is what the
system enables. §6 is **what has actually been measured about the AA gap** — the
result that closed it is §6.6, and everything before it is a door that closed;
read the whole of §6 before building anything in this area, it will save you a
day. §7 is methodology that cost real debugging time.

Everything measured here was measured on the **CPU** render device on a machine
with no GPU, unless it says otherwise. That is why §4 exists.


================================================================================
0. STATE OF THE BRANCH, AND WHAT TO DO NEXT
================================================================================

Branch `claude/renderer-mesh-id-rework-n5ezw5`, on top of `efb3a95`:

    4891ffd  Cap the mesh's coverage claim instead of suppressing its far sheet (§6.6)
    3f1cca2  Add the one-mesh coverage rule (§6.6): a Cylinder now beats the bezier Line
    4827e35  Bring DESIGN_mesh_identity.md §0 up to date with the whole §3 sweep
    2d1432a  Turn mesh identity and Polyhedron winding on by default (§3.5, §3.7)
    86ff500  Add the watertight ray/triangle test (§3.2), gated off
    d878f76  Weld closed surface seams and collapsed poles (§3.1), gated off
    517c842  Build §6.3.2's relaxed AA run gate and §3.4's bezier split, both gated off
    32bdc9d  Make a packed grid's declared mesh_ids reach the renderer, and measure it
    89b81a1  Correct three stale drop counts and the last refuted winding prediction
    59d6782  Sound the sheet reference, correct the MESH_ID verdict, land the winding fix
    e067702  Qualify ALGAN_MESH_ID on coverage, and find the gate that costs the AA error
    e851ee6  Replay the resolve's svis walk: the diced-mesh AA gap is ownership
    c8e9b9b  Make DESIGN_mesh_identity.md a self-contained handoff; drop orphaned PN comments
    a90b2ff  Apply ruff format to the files this branch touched
    6d02488  Delete TriangleVertices2 and correct stale renderer comments
    568b5ae  Add DESIGN_mesh_identity.md: CUDA verification plan and negative results
    690009a  Mob-declared surface identity for tri_obj, gated off pending the run-rule fix
    c87c26b  Add _aa_run_gate_check: attribute the diced-mesh AA gap
    b49b01b  Delete the unreachable curved PN-patch renderer

Since merged to `master` (`9a23b46`), and the whole of §4 has now been run on
**CUDA** (GTX 1050, driver 576.52, Taichi 1.7.4, torch 2.7.1+cu128, Taichi cache
cleared first). Both CUDA baseline sets are regenerated against the shipped
defaults, and everything portable is green:

    pytest -q tests/unit_tests tests/fast      1046 passed, 89 skipped
    pytest -q tests/full_renders                  7 passed
    ruff check --no-fix / ruff format --check   clean

The two **CPU** baseline sets are the one piece of debt left, and it cannot be paid
from this machine — see §3.5, which measures why.

**Four defects were found by qualifying the gates rather than by using them.**
The first two are fixed here; the last two are diagnosed and scoped, and each
keeps its gate off. They are the most useful thing in this round, so they are
listed before the wins:

1. **`ONE_MESH` did not actually enable the relaxed run gate it "implies"** — the
   implication was wired on the kernel side and not on the host side, so the
   relaxed scan ran over fragment lists whose area donors had already been
   discarded. Worth most of §6.6's win on flat geometry (-8% of ink wobble against
   -63%). §6.6.1; now one predicate, with an AST audit.
2. **The cap's per-pixel ceiling was built with a float atomic**, so a render was
   not reproducible: two consecutive renders of `materials_and_lighting` differed
   by 28 channel values over 9.6% of a frame, because the ceiling feeds a
   *threshold*. §6.6.4; now accumulated in float64 and rounded, verified bitwise
   stable.
3. **§3.3's scope was wrong** — "delete the epsilons" is two deletions with
   different owners, because `BARYCENTRIC_EPSILON` has two ungated consumers in
   the raster front-end that `_tri_hit` never touches. §3.3.
4. **Only the render path is weld-aware** — with §3.1 on, `convert_to_pn_soup`
   and `get_render_primitives` disagree about a `Sphere`'s triangulation, so the
   mesh renders one way and morphs another. Found by flipping the gate and
   running the whole suite, not by rendering. §3.1; the gate stays off.

**THE GATES, and what each is worth.** Nine switches now, all declared in
`algan/environment.py` and surfaced on `SETTINGS.raytracing.experimental`.

    setting                          default  what it buys
    ---------------------------------------------------------------------------
    ALGAN_MESH_ID                    ON       per-mesh tri_obj (§2.2, §3.5)
    ALGAN_POLYHEDRON_WINDING         ON       consistent face winding (§3.7)
    ALGAN_ANALYTIC_AA_ONE_MESH       ON       THE AA RESULT (§6.6), implies ↓
    ALGAN_ANALYTIC_AA_ONE_MESH_DENS  ON       the capped write's other half (§6.6.2)
    ALGAN_ANALYTIC_AA_RUN_FULL       off      the relaxed run gate ALONE (§6.3.2)
    ALGAN_WELD_SURFACE_SEAMS         off      shared seam/pole vertices (§3.1)
    ALGAN_WATERTIGHT_TRI             off      Woop-Benthin-Wald (§3.2)
    ALGAN_BEZ_BVH_SPLIT              off      median-split bezier BVH (§3.4)
    ALGAN_ANALYTIC_AA_RUN_RULE       redist.  pre-existing (v2 §4.4)

Four of the nine are on, and the four still off are off for stated reasons
rather than for want of attention:

* `ALGAN_ANALYTIC_AA_RUN_FULL` is **subsumed**, not pending: `ONE_MESH` implies it
  (`aa_grp` 3 or 4, and `_aa_run_full` accepts anything from 2 up), so it only selects
  the relaxed gate *without* the cap — a configuration kept for the harness.
* `ALGAN_WATERTIGHT_TRI` is correctness-qualified and **cost-unqualified**: its cost
  cannot be measured on a thermally throttled machine, and because the flag is read
  at import an in-process alternating A/B is impossible. §3.2 says what would settle
  it.
* `ALGAN_BEZ_BVH_SPLIT` is byte-identical and shows **no** measurable speed-up, so
  there is no evidence either way; §3.4 recommends leaving it off until something
  counts traversal steps.
* `ALGAN_WELD_SURFACE_SEAMS` has its *stated* risk closed (byte-identical on a
  static frame, textures and normal maps included) and picked up two new ones on
  the way: only the render path is weld-aware, so with it on a `Sphere` morphs
  from a different triangulation than it renders; and it does move a moving PN
  scene, so it needs baselines after all. §3.1.

**THE HEADLINE: the diced-mesh AA gap is largely closed, and one earlier claim
about it was too strong.** `_aa_line_check` opened this whole line of work by
measuring a tessellated `Cylinder` at 0.057 px of ink wobble against 0.014 for a
flat quad and 0.004 for a bezier `Line`. With `ALGAN_ANALYTIC_AA_ONE_MESH=1`,
measured on **CUDA** (see §6.6.1 for why the CPU column is not the shipping one):

    kind           shipped   CUDA now         earlier CPU claim
    bezier Line     0.0042    0.0042          0.0042  (never entered this path)
    flat quad       0.0138    0.0052  -63%    0.0051
    Cylinder        0.0568    0.0124  -78%    0.0039   <- DOES NOT REPRODUCE
    Cylinder fine   0.0772    0.0429  -44%    0.0411

`on-lattice` — the share of silhouette pixels landing on a multiple of 1/8 —
falls from 8–91% to 0–1.6%, and coverage error against an exact reference falls
70–100% on all eleven harness cases. **The coverage is no longer sample-based**,
which is the result. §6.6 is the rule, §6.3 the diagnosis it rests on.

**A previous revision of this section said "a `Cylinder` now anti-aliases better
than a bezier `Line`". It does not.** On CUDA the best available is 0.0124
against the Line's 0.0042. The win is real and large; the ordering claim was
wrong. §6.6.1 has the reconciliation, including the gate bug that accounted for
the rest of the gap and is now fixed.

**What the rule is.** Where every fragment in a pixel is an opaque triangle of
ONE surface, the mesh may claim at most `max(front_area, back_area)` in total —
a per-pixel ceiling the host computes from the exact clipped areas and carries
per fragment in `frag_cap`. That removes the far-sheet re-claim: a run's
`corr < 1` scales the occlusion write as well as the claim, so the near sheet
leaves a residual transmittance standing for area OUTSIDE the mesh, and the
solid's own far sheet was claiming it as though it were background. **This is
what §2.2's declared identity was built to enable and what nothing read until
now** — "these two sheets are one mesh" is not a geometric question and no
epsilon can answer it.

**FOUR THINGS THIS DOCUMENT PREDICTED THAT TURNED OUT WRONG.** Each is corrected
where it belongs; none was quietly dropped.

* **§6.3.2's −88% does not exist.** Its premise was false: the emission
  truncates a pixel's fragment list at the first full-mask fragment, so the run
  scan can never reach that sheet's area donors. As specified it *notched*
  interior tilings. It is −63% on flat geometry and inert on a diced mesh —
  which is what §6.6 then fixed by a different mechanism.
* **§3.1 neither moves the pixels nor retires its two epsilons.** A
  Sphere/Cylinder/Torus/Cone scene is byte-identical across the weld; the normal
  accumulation runs on the grid, not the welded topology, so both fixups stay.
* **Suppressing the far sheet regresses sub-pixel dicing** (+114% on a
  0.045-radius rod diced 256×). Two follow-up hypotheses — scrambled facing bits,
  and the u-seam — were both tested and refuted. §6.6 has the refutations and the
  cap that replaced suppression.
* **`tests/full_renders` cannot arbitrate from a cloud container.** All six
  scenes fail here at shipped defaults with every gate off; those baselines are
  another machine's. §3.5 lists the debt.

**WHERE EVERY SECTION STANDS.**

    §3.1  weld surface seams/poles     STAYS OFF — pixel case proved, morph/render
                                       agreement FIXED; only baselines left (§3.1)
    §3.2  watertight tri intersection  QUALIFIED on correctness, cost UNMEASURABLE
                                       here; stays off, and §3.3 stays blocked
    §3.3  delete the epsilons          BLOCKED, and it is TWO deletions not one —
                                       the raster path has its own consumers (§3.3)
    §3.4  median-split bezier BVH      MEASURED: byte-identical, no speed-up;
                                       recommendation is to leave it off
    §3.5  mesh identity                FLIPPED ON, both devices re-baselined
    §3.6  two-level BVH                NOT STARTED, and the perf case is MEASURED
                                       not to justify starting (§3.6)
    §3.7  Polyhedron winding           FLIPPED ON, same re-baseline
    §6.3.2 relaxed AA run gate         SUBSUMED by §6.6; the switch alone stays off.
                                       It OWNS the residual interior notches (~92%),
                                       which ship KNOWINGLY UNFIXED — see §0.5
    §6.6  one-mesh coverage cap        FLIPPED ON — the AA result, plus a gate bug
                                       found and fixed while qualifying it
    §6.6.2 capped occlusion write      FLIPPED ON — closes the claim-vs-occlusion
                                       desync; the CLAIM-side shortfall stays open

**WHAT IS LEFT, in priority order.** Every item in the previous revision of this
list has been run; these are what running them produced.

0. **The interior-notch limitation is CLOSED as a decision, not as a fix — see
   §0.5.** Diagnosed to the line (`_AA_MAX_RUN_SCAN = 16`), costed, and
   deliberately left in place: ~2 channel values typically, ~13 at the worst pixel
   of deliberately pathological geometry, against a fix that is either a hot-path
   loop bound or a withdrawal of the gate's silhouette win. **Do not reopen it
   without first measuring the six `tests/full_renders` scenes**, which nobody has
   done; §0.5 says how, and what result would change the decision. This entry
   exists so nobody spends a day rediscovering §0.5.
1. **Build a traversal-step (or instruction) counter.** It settles §3.4's
   inherited "~20-25% fewer traversal steps" and most of §3.6's case, and it is
   the right instrument for THIS machine: a step count is deterministic, so
   unlike wall-clock it does not dissolve into thermal drift (§7.15).

   **It does not settle §3.2, though this list said it did.** §3.2 changes the
   ray/triangle INTERSECTION TEST, not which nodes get visited — the same
   traversal reaches the same leaves either way, so a step count is identical
   across the arms by construction and prices nothing. §3.2's cost is a
   time question and stays one.

   Two ways to build it, and the cheap one is probably right: a `ti.static`-gated
   counter in the kernels compiles out when off but costs a ~40 minute cold
   recompile per iteration on this box, while a HOST replay of the walk over the
   same STBVH arrays costs no recompile and is the pattern that worked three
   times in §6.6.2 — replay the same thing with one input changed. A host walk
   has to be validated against the kernel (the §6.6.2 replay had `--verify` for
   exactly this) or it measures itself.
2. **§3.2's cost, on hardware that is not throttling.** Correctness is done and
   clean (§3.2/§4.7: zero cracks, no double blend, byte-identical on opaque
   geometry). Only the cost is open, and this machine cannot resolve it — the
   controls drift as much as the target kernels, and the flag is read at import so
   an in-process alternating A/B is impossible. Then §3.3's ray-path half.
3. **§3.3, as two deletions with different owners.** Not one deletion gated on
   §3.2: `BARYCENTRIC_EPSILON` also has two ungated consumers in the raster
   front-end's own candidate acceptance, which `_tri_hit` never touches. §3.3 has
   the consumer table. Do not promise the per-ray `f32` until both have landed.
4. **§3.1 now needs only baselines.** Both blockers are gone: the stated pixel
   risk was closed by measurement (byte-identical on a static frame, textures and
   normal maps included), and the topological one is fixed — the morph path asks
   `surface_weld_flags` for the same grid the render path asks about, so a
   `Sphere` no longer morphs from a different triangulation than it renders. The
   whole unit suite is green with `ALGAN_WELD_SURFACE_SEAMS=1`. What is left is
   that it *does* move a moving PN scene, so flipping it regenerates both
   devices' baselines — and the CPU set cannot be regenerated on the machine that
   owns the CUDA one (§3.5). That is the only thing standing between this gate
   and its default.
5. **§4.6's case 1 still has no reach check.** The purpose-built scene exists
   (`benchmarks/_shadow_anyhit_check.py`) and settles case 2: the 304-sheet stack
   demonstrably reaches the peel limit and all three `SHADOW_ANYHIT` modes are
   still byte-identical, so that documented disagreement does not appear from the
   public API — plausibly because an opaque blocker is found through the
   opaque-only BVH prepass, which is a hypothesis nobody has checked. Case 1's
   scene renders and agrees, but nothing demonstrates it puts an opaque edge hit
   within `DEPTH_TIE_EPSILON` of a translucent one, so its agreement means
   nothing yet. Build that check before reading that column.
6. **§3.6 only if something changes.** Measured not to justify starting: the BVH
   build it would amortize is ~1% of a shadowed render, and the instancing win it
   would unlock needs a workload with thousands of repeated meshes, which no
   scene in the repo has.

Do **not** start by regrouping the run rule, by consulting `E` only inside the
existing gate, by buying more samples, or by suppressing the far sheet. All four
were built or measured here and none is the lever — §6.

And do **not** repeat these five, each of which this document asserted and
measurement refuted: that scaling `dens` would also close the interior notches
and the `--verify` failures (§6.6.2 — it closed neither, and could not have:
both are claim-side and it changes the occlusion write); that a `Cylinder` now
beats the bezier `Line` (§6.6.1); that `ONE_MESH` alone gives the relaxed gate
(§6.6.1 — it did not, and that was a bug); that welding moves pixel baselines
(§3.1 — byte-identical, textures included); and that the PN deletion shrank the
compile surface (§4.4 — the deleted variant was never compiled).


================================================================================
0.5 KNOWN LIMITATION, SHIPPED AND DELIBERATELY NOT FIXED
================================================================================

**A diced mesh can lose up to ~5% of one interior pixel's coverage, and the
default renderer ships that way.** This was diagnosed to the line, costed, and
then left alone as a considered decision rather than an oversight — the fix is
not worth its price at the sizes measured. Read this before "fixing" it.

**WHAT IS WRONG.** The run scan sums one sheet's exact clipped areas to get `E`,
the area that sheet covers in the pixel, and stops after `_AA_MAX_RUN_SCAN = 16`
fragments (`raster_taichi.py`). If it stops early, `E` is a **partial sum — a
lower bound on the sheet's area**. §6.3.2's relaxed gate then does

    if rU == _AA_MASK_ALL:        # the scanned fragments own every sub-sample
        run_corr = min(rE, 1.0)   # ... so scale the pixel's coverage by E

On a SILHOUETTE pixel that is the intended fix: the sheet really does cover only
`E` of the pixel. On an INTERIOR pixel the sheet covers all of it, and `E < 1`
only because the scan quit early — so the pixel is scaled down by exactly the
area the scan never summed.

**WHEN IT FIRES.** Three conditions, all required:

1. **Triangle geometry** — `Sphere`, `Cylinder`, `Cone`, `Torus`, `Surface`,
   `Polyhedron`, imported glTF. Bezier circuits never enter this path, so `Text`,
   `Tex` and the 2-D shapes are structurally immune. That is why
   `manim_compat_and_plots` moves zero pixels through every flip in this file.
2. **The pixel is INTERIOR** — wholly inside the surface, not on its outline.
3. **More than 16 fragments of the SAME SHEET** (one surface, one facing) land in
   that one pixel.

Condition 3 is a facets-per-pixel question, not a tessellation question, and the
harness holds a matched pair that says so: `Cylinder(radius=0.9,
resolution=(256, 2))` notches **zero** pixels while `Cylinder(radius=0.045,
resolution=(256, 2))` — identical tessellation, 20x thinner, ~9 px wide on
screen — notches 253. Two ways a scene gets there: a finely tessellated object
drawn small, or **the limb of any curved surface**, where facets foreshorten and
crowd. The limb is why a large `Sphere` notches at all.

**HOW BAD, measured `--res md` on CUDA** (`--notch-probe`):

    case                              notched interior px    mean     worst
    0.045 rod, resolution=(256, 2)      253 / 3546  (7.1%)   0.0090   0.0515
    Sphere(192, 96)                      24 / 26480 (0.09%)  0.0018   0.0036
    line-check cylinder                   4 / 10195          0.0010   0.0010
    packed 4x4 (overlap)                  3 / 30531          0.0014   0.0017
    the other seven harness cases         0                     -         -

Seven of eleven cases are clean, and the one bad case was **built to break the
coverage rule** rather than because anyone renders rods that way.

The shortfall is a coverage error, so what shows is that fraction of the CONTRAST
between object and background. Since §6.6.2 the pixel is energy-conserving, so it
appears as background bleeding through the solid's interior rather than as
darkening. On 8-bit against a high-contrast background that is **~2 channel
values typically and ~13 at the worst pixel of the worst case** — the typical
figure sits at the render suites' tolerance of 2, which is why no suite catches
it.

**IT IS NOT ENTIRELY THE GATE'S.** With no gate at all the rod already had 50
notched pixels; the relaxed gate took it to 239 and the one-mesh cap added 14.
The gate quadrupled a pre-existing effect rather than creating one.

**WHAT WOULD FIX IT, and why neither was done.** Replaying each notched pixel
with the scan limit lifted recovers 231 of 253 on the rod and 13 of 24 on the
Sphere — so the limit is the mechanism, and there are two levers:

* **Raise `_AA_MAX_RUN_SCAN`.** One constant. But it is a loop bound in the
  megakernel's hot path, paid by every pixel that scans, and the cap exists
  deliberately.
* **Refuse to consult `E` when the scan hit its limit**, falling back to the
  shipped `corr = 1` short-circuit. Cheap and principled — a truncated sum is not
  an area — but it withdraws the gate's win from every long-run SILHOUETTE pixel,
  and on the rod those are most of the frame (`capped` is 3011 of 3546 clean
  interior pixels).

Either needs a kernel recompile and a cost number, and cost is exactly what the
machine this was measured on cannot resolve (§7.15). Against a worst case of ~13
channel values on deliberately pathological geometry, that is not a good trade.

**WHAT IS NOT MEASURED, said plainly.** Everything above is the synthetic
harness. **Nobody has counted notched pixels in the six `tests/full_renders`
scenes**, which are the only realistic scenes here. What is known about them is
weaker: the worst-differing frames of `solids_and_camera` and
`materials_and_lighting` were reviewed side by side at 12x amplification and show
no notches, rims or interior speckle — but "looked and did not see it" is not
"there are none" for a 2-channel effect. Those scenes carry `Sphere`, `Cylinder`,
`Torus` and `Surface` at auto-chosen grid resolutions, so they satisfy conditions
1 and 2, and every one of those shapes has a limb.

**If you are picking this up, measure that first.** Point `--notch-probe` at the
full-render scenes rather than the harness cases; roughly an hour, mostly render
time. If real scenes show single-digit pixel counts, the standing decision —
document it and leave it — is confirmed and no kernel change is warranted.


================================================================================
1. THE PROBLEM
================================================================================

The renderer resolves visibility over a flat pool of independent triangles.
Nothing in it knew that the triangles of a `Sphere` are one surface, so "is this
second hit the same surface point I already shaded?" was guessed geometrically,
by a two-part epsilon heuristic with a mutable running state:

    edge_hit = smallest barycentric coordinate < TRIANGLE_EDGE_EPSILON (2e-4)
    skip if  edge_hit and (t_hit - seam_t <= DEPTH_TIE_EPSILON)   # 1e-4

The duplicate it drops is manufactured on purpose: `BARYCENTRIC_EPSILON` (1e-4)
dilates every triangle so a ray on a shared edge cannot miss *both* neighbours
and leave a crack (`raytrace_kernels_taichi.py:107`). Dilation and dedup are
a matched pair; neither means anything alone.

What it costs:

* **Replicated in 8 kernels**, with 8 initialisations, 5 bounce resets and a
  dedicated per-ray state slot (`rs_sca[r, 3]`, laid out at `tracer.py:186`).
* **Not an equivalence relation.** The depth *order* was already fixed by
  binning (`_comes_after`), but the dedup is a greedy window against a mutable
  `seam_t`, so it chains and is asymmetric.
* **It makes output depend on discovery order, and that blocks real work.**
  `stbvh.py` keeps bezier circuits on the slower Morton builder — triangles get
  the ~20-25% faster median split — purely "to preserve baselines".
* The epsilons are absolute world-space constants with no scene-scale
  adaptation.


================================================================================
2. WHAT LANDED
================================================================================

2.1 The curved PN-patch renderer is deleted (`b49b01b`)
-------------------------------------------------------
`RENDERER_REGISTRY.triangle_primitive` is never rebound from
`RayTracedTrianglePrimitive` (`settings/renderer_settings.py:17`), so
`RayTracedPNTrianglePrimitive` was unreachable through the public API and
`merged["num_pn"]` was always 0. `Surface` — and therefore
`Sphere`/`Cylinder`/`Cone`/`Torus` — reaches the renderer as *logical PN*
patches diced to flat triangles before the tracer or the STBVH sees them
(`algan/rendering/logical_pn.py`), crack-free by construction because adjacent
patches derive **bit-identical** shared boundary vertices.

Removed: the class; `pn_patch.py` in full; `_pn_intersect` and its cubic solver;
`_nearest_pn_hit`; `_obb_misses`; `_pn_normal`; `_shade_pn_hit`;
`_anyhit_opaque_pn`; the wavefront's five `_pn_hit_*` helpers; every
`htype == 2` branch; the `pn_*` merged keys and their two STBVH builds per batch
(six trees → four); and four epsilons (`PN_BARYCENTRIC_EPSILON`,
`PN_EDGE_EPSILON`, `PN_DEDUP_UV_EPSILON`, `PN_SEAM_DEPTH_EPSILON`). `seam_eps`
collapses to `DEPTH_TIE_EPSILON` at all five sites that selected it. ~2800 lines
net. `num_pn == 0` also disappears as an always-true clause from
`analytic_raster_route_active`, `use_raster`, `_projection_anti_alias_level`,
`_bvh_deferral_eligible` and the `WF_TEXTURED` gate.

**Output is byte-identical** on CPU, verified per-frame and per-video, with the
merged scene tensors and derived render flags hashed equal and the batch windows
unchanged.

2.2 Mob-declared surface identity — now shipped ON (`690009a`, flipped by `2d1432a`)
--------------------------------------------------------
`tri_obj` is what the analytic-AA resolve groups fragments by, and its
granularity was one id per merged **collection member** — right only when one
member is one surface, which is wrong at both ends:

* `Polyhedron` hands the batcher one member per **triangle**, so a `Cube` was
  twelve surfaces and no run could span a face diagonal.
* A packed-grid `Surface` hands it one member covering **every** packed sphere,
  so distinct spheres were unioned and their coverage summed across objects that
  merely overlap.

Mobs now declare identity on the primitive they build, resolved by
`primitives._mesh_ids_from_collection`:

    mesh_key   merge with the consecutive neighbours sharing it (matched against
               the preceding member only, so identity cannot leak across an
               unrelated mob that happens to sit between two halves of one)
    mesh_ids   subdivide one member into per-triangle shells; needs no contiguity

Declared by `Polyhedron` (one solid), packed-grid `Surface` (one shell per
grid), and `TriangleMesh`, whose `corner_index` already carried the loader's
topology and was only kept for smooth normals — `triangle_shell_ids()` walks it
for **edge**-connected components via scipy, so an imported file's disconnected
parts stop being one surface. Edge- not vertex-connectivity, which would fuse
two cones meeting at an apex. Deliberately **not** `Arrow3D`: its children are
separate interpenetrating solids, not one mesh.

Verified: a `Cube` + `Icosahedron` go from 32 surfaces to 2.

`ALGAN_MESH_ID` **now defaults ON** (§3.5). It was introduced default-off, which is
why §4.5 reads as a case for flipping it; that case is closed.

2.3 A measurement harness (`c87c26b`)
-------------------------------------
`benchmarks/_aa_run_gate_check.py` intercepts the sparse-raster fragment build
and replays the analytic-AA run rule's grouping and magnitude decisions on the
host for **every covered pixel**, so questions about it get a population
statistic instead of one dumped pixel. It produced §6. Its own docstring carries
the measured tables.

2.4 First tests for `tri_obj` (`690009a`)
-----------------------------------------
`tests/unit_tests/test_mesh_identity.py`. Nothing tested `tri_obj` in any suite
before. Pure tensor assertions, no render, no Taichi, so the end-to-end cases
are marked `fast`.

2.5 The resolve, replayed on the host (`e851ee6`)
--------------------------------------------------
The same harness now also replays `raster_first_shade`'s per-sample
transmittance walk for every covered pixel and scores the coverage each pixel
ends up with against an **exact** analytic reference, verified against the
kernel's own `ALGAN_AA_DUMP` rows. That is what §4.5 asked for and could not
have, and it is what answered §6.3. No engine code changed; output is untouched.


================================================================================
3. WHAT HAS NOT LANDED
================================================================================

3.1 Weld the `Sphere` u-seam and the pole fans  [LANDED, STAYS OFF — the morph
    path is not weld-aware]
-------------------------------------------------------------------------------
`get_grid_to_triangle_indices` (`surface.py:211`) builds two triangles per
grid cell and **never bridges column `W-1` back to column 0**, so a closed
surface's wraparound is a genuine two-copy seam. Measured on a `Sphere`, float32:

    col0 vs col(W-1) max abs diff: 1.7484555314695172e-07
    bitwise equal: False

The poles are collapsed degenerate fans — every point of grid row 0 maps to
`(≈0, -1, ≈0)`, x jitter 4.37e-08 — and `surface.py:392` documents at length
the bright sliding blob that costs.

Interior shared edges, by contrast, are **bit-identical** duplicates: the same
gather from the same `flat_grid` row (`surface.py:250`). That asymmetry is
the whole point — a watertight intersection test fixes numerical ambiguity, and
would *open* a crack at the u-seam rather than close one, because that gap is
real geometry.

What to do. The closed-seam predicate already exists (`is_closed_x`,
`surface.py:363`, currently used with a 1e-4 tolerance to merge normals):
when it holds, index the wrap cell against column 0 instead of emitting a
duplicate column, and emit a single shared pole vertex instead of a fan. This
retires two authoring-side epsilon special-cases (the 1e-4 normal merge and the
pole-normal salvage) and slightly reduces triangle count.

**LANDED**, gated `ALGAN_WELD_SURFACE_SEAMS`, now default **ON**, surfaced as
`SETTINGS.raytracing.experimental.set(weld_surface_seams=...)`. Implemented as
described: `surface_weld_flags(grid)` reads `(wrap_x, pole_lo, pole_hi)` off the
grid once per primitive build, `get_grid_to_triangle_indices` takes it (and keys
its cache on it), the wrap cell indexes column 0, a pole row collapses to one
vertex, and the `W-1` degenerate triangles each pole contributed are dropped.
`tests/unit_tests/test_surface_welding.py` pins all of it, including that the
unwelded path is exactly what it always was.

Two things §3.1 asserted above are **wrong**, both measured:

* **"Geometry moves, so all pixel baselines move."** They do not. A scene of a
  `Sphere(48, 24)`, a `Cylinder`, a `Torus` and a `Cone` renders
  **byte-identical** across the gate at `--res md`, despite the sphere going
  from 2304 triangles to 2208. That is the expected result once stated plainly:
  the welded vertices were coincident to 1.7e-07 and the dropped triangles had
  zero area, so nothing the rasterizer can see changes. `pytest -q --fast`
  passes with the gate on. The remaining risk is a texture-mapped or
  normal-mapped closed surface. That risk has since been measured and is not
  real (see the qualification below). Read that qualification before quoting
  this bullet: "byte-identical" here is a claim about a STATIC frame, and the
  full renders do move (§7.18). The gate stays off, for a reason that turned out
  not to be about pixels at all.
* **"Retires two authoring-side epsilon special-cases."** It does not.
  `compute_grid_vertex_normals` accumulates over the **grid**, not over the
  welded triangle list, so column 0 still misses the wrap-around neighbourhood
  and a pole row still accumulates from sub-epsilon differences. The 1e-4 normal
  merge and the pole-normal salvage both stay necessary and stay in place.
  Retiring them needs the normal accumulation itself to run on welded topology,
  which is a separate change. The weld also still needs a tolerance of its own
  to decide whether a parametrization closes -- that is a property of the
  coordinates and no topology change can remove it.

Note the UV subtlety, which cost a shape mismatch before it was handled: the
**pole** welds apply to the uv gather (they change the triangle list, so every
per-vertex attribute must go through the same indices), but the **u-seam** wrap
deliberately does not. Wrapping it would give the last cell column `u = 0` where
the texture needs `u = 1`, running the map backwards across that column. The
duplicate uv column exists precisely to carry that discontinuity.

**QUALIFIED ON CUDA, AND THE ONE STATED RISK IS CLEARED.** The reason this stayed
off was "a texture-mapped or normal-mapped closed surface". Measured,
`benchmarks/_weld_check.py`, `--res md`, CUDA:

    shape                        tris off   tris on   max|d|   px>2
    plain (Sphere/Cyl/Cone/Torus)    6668      6572        0      0
    checker (colour texture)         4096      3968        0      0
    normals (normal map)             4096      3968        0      0

The weld demonstrably engaged — 128 triangles fewer on the sphere, which is
exactly the two poles' `W-1` degenerate triangles at `W = 64` — and output is
byte-identical on all three, textured and normal-mapped included. The
checkerboard is the instrument on purpose: a one-column uv error would mirror or
shift a hard edge, which a smooth photo would hide. **§3.1's stated risk — a
textured or normal-mapped closed surface — is closed.**

**But "byte-identical" does not generalize, and this harness is too narrow to have
shown that.** Running the full suites across the gate contradicts it:

    scene                    max|d|   worst-frame px    frames
    shapes_and_timeline           0     0 (0.000%)       0/301
    text_and_media                0     0 (0.000%)       0/182
    materials_and_lighting       31 28501(10.223%)      92/179
    solids_and_camera            54 20159 (7.231%)     222/239

The split is exactly the geometry families: the two scenes built from circuits and
flat meshes do not move, and **both scenes carrying `Surface`/PN geometry do**.
`_weld_check` renders a *single static frame*; the full renders move a camera over
adaptively diced PN surfaces, and the dice level is chosen per patch per frame from
projected size, so a different triangle list can land on a different level. That is
precisely the class `CLAUDE.md` warns is "invisible to `--fast`".

So the honest statement is: the weld is byte-identical on a static frame, including
textured and normal-mapped closed surfaces, and it **does** move a moving PN scene.
It needs baselines after all — which, combined with the morph-path inconsistency
below, is why it stays off.

**THE BLOCKER BELOW IS NOW CLEARED; what remains is baselines.** The morph path
asks `surface_weld_flags` for the same grid the render path asks about, so both
build the same triangulation, and the `DotCloud` test derives its expected count
from the builder instead of restating the unwelded formula. With
`ALGAN_WELD_SURFACE_SEAMS=1` the whole unit suite is green on CUDA (1050 passed,
89 skipped), including the two tests named below. The gate still ships OFF for
the one reason left: it moves a moving PN scene, so flipping it needs both
devices' baselines regenerated, and the CPU set cannot be regenerated here (§3.5).

*The original diagnosis, kept because it names the class of defect:*

**Flipping it on and running the whole suite found the real blocker, which was
not about pixels.** Two tests failed, and they were not stale expectations:

* `test_pn_mesh.test_surface_conversion_reproduces_its_logical_pn_primitive` —
  `convert_to_pn_soup(Sphere)` and `Sphere.get_render_primitives()` return
  *different triangles*.
* `test_point_cloud_rendering.test_dot_cloud_spheres_have_disconnected_triangle_topology`
  — 400 triangles against a hard-coded `2*(W-1)*(H-1)` = 480, which is exactly the
  two pole fans the weld drops.

The first is the one that matters. The weld lives in
`get_grid_to_triangle_indices`, and **only the render path calls it**. The morph
path builds its triangles with `grid_to_triangle_vertices`
(`morph_conversions._grid_to_pn_soup`), which knows nothing about the gate. So with
the weld on, a `Sphere` renders with one triangulation and morphs from another —
a mesh that disagrees with itself.

That is the same class of defect as §6.6.1's half-wired gate and §7.11's lesson:
one question, two answerers.

**And the fix was smaller than this section predicted.** It called for routing
every consumer behind "one weld-aware builder", the shape of §3.2's `_tri_hit`.
There was no such refactor to do: `grid_to_triangle_vertices` already takes the
weld flags, and the render path already passes them — the morph path simply
called it without the argument and silently took the unwelded default. So the
change is that `_grid_to_pn_soup` computes `surface_weld_flags(grid)` and passes
it to all three gathers, which is what makes the two paths ask the same question
rather than two questions. Worth carrying to the next item that looks like a
refactor: check whether the shared function already has the parameter before
designing one.

The `DotCloud` test is the one-line consequence, done the same way: its expected
count comes from the builder now, not from the unwelded `2*(W-1)*(H-1)`.

3.2 Watertight ray/triangle intersection  [LANDED, correctness QUALIFIED,
    cost UNMEASURABLE here, stays off]
------------------------------------------------------------------
With seams welded and interior edges bit-identical, a watertight test
(Woop–Benthin–Wald: ray-space transform, consistent edge-function signs, a
deterministic tie-break) returns exactly one hit per shared edge with no
dilation. The deterministic tie-break it needs already exists as `layer`
(`= layer_offset + prim`, `raytrace_kernels_taichi.py:807`).

Note the raster path already has a watertight rule to imitate: an exact
fixed-point rasterizer on a 1/4096-pixel lattice with int64 edge functions and a
top-left fill rule that *partitions* sub-pixel samples (`_ss_pixel`,
`raster_taichi.py`). Its long comment explains why exact integer arithmetic makes
two triangles' shared-edge functions exact negatives — that argument is what a
ray-path version has to reproduce.

**LANDED**, gated `ALGAN_WATERTIGHT_TRI`, default off. Implemented as `_tri_hit`
in `raytrace_kernels_taichi.py`, one `@ti.func` that both arms go through, so the
three intersection sites (`_nearest_triangle_hit`, `_collect_hits`,
`_anyhit_opaque_tri`) can no longer drift apart. Read at **import**, not live: it
changes the compiled kernel body, so a runtime toggle would silently reuse a
cached kernel (the `_AA_SAMPLES` cache-trap rule). Clear the Taichi cache when
flipping it.

The permutation is written as three explicit cases rather than dynamic vector
indexing, which Taichi supports only under a global flag and codegens poorly in
the hottest loop in the renderer. The exact-zero edge case gets a
canonical-endpoint tie-break (`_edge_is_canonical`) — consistently wound
neighbours traverse a shared edge in opposite directions, so a strict total order
on the projected endpoints picks exactly one owner. That is the ray-side analogue
of the raster path's top-left fill rule, and it is the part the sign test alone
does not give you: exact negation makes a zero edge function zero in *both*
neighbours.

**Verified on CPU**, `tests/unit_tests/test_watertight_triangle.py`, which asserts
whichever arm the environment selected:

* Watertight arm — a ray exactly on a shared edge hits **exactly one**
  neighbour, at every one of 37 positions along it.
* Shipped arm — the same rays hit **both**, which is precisely the duplicate
  `TRIANGLE_EDGE_EPSILON` exists to discard, and is the clearest statement of
  why the two epsilons are a matched pair.

End to end, with the hybrid raster disabled so all visibility goes through the
ray path, a Sphere/Cube/plane scene moves **11 of 419904 pixels by at most 1
channel value** across the flag — edge-localized and sub-quantization, which is
the expected signature of removing a 1e-4 barycentric dilation. With the hybrid
raster on (the default), the same scene is byte-identical, because primary
visibility never touches this code.

**§4.7 IS NOW RUN ON CUDA, AND THE CORRECTNESS HALF PASSES CLEANLY.**
`benchmarks/_watertight_check.py`, `--res ld`, hybrid raster **off** so all
primary visibility goes through `_tri_hit`:

    scene / metric                   shipped arm      watertight arm
    grazing quads   drawn px             18514              18514
                    CRACKS                   0                  0
    diced Sphere    drawn px             59272              59272
    (192x96)        CRACKS                   0                  0
    translucent     ridges @ a=0.35    114/35736          114/35736
                    ridges @ a=0.60          0                  0
                    ridges @ a=0.85          0                  0

* **No cracks in f32, and this is the result that mattered.** Removing the
  `BARYCENTRIC_EPSILON` dilation is exactly the change that could open a seam, and
  on the two scenes built to provoke it — quads at 84/87/89 degrees of tilt, and a
  192x96 sphere filling the frame — the watertight arm leaks **zero** enclosed
  background pixels. Counted by filling the silhouette's holes, not eyeballed.
* **No double blend introduced.** The ridge counts are identical, so the
  watertight arm is not blending a shared edge twice where the epsilon used to
  clean up after the dilation.

And the two arms differ almost nowhere. Per-scene image diff across the flag:

    grazing         0 pixels of 419904 differ  (byte-identical)
    diced Sphere    0 pixels of 419904 differ  (byte-identical)
    translucent     2 pixels differ, max 15 / 25 / 35 at a = 0.35 / 0.60 / 0.85

That distribution is the mechanism stated in pixels: on **opaque** geometry a
duplicate edge hit is absorbed by the nearer one and the two arms cannot differ,
so they do not; on **translucent** geometry the duplicate is what would blend
twice, and that is where the arms disagree — on two pixels of a 419904-pixel
frame. Both arms are already correct there (the epsilon discards the duplicate,
the watertight test never makes one); what differs is which neighbour owns the
edge and its undilated barycentrics, hence a shading difference on the seam pixel.

Note this is the **ray-path** measurement. In the shipped configuration the hybrid
raster front-end owns primary visibility, so the flag reaches only secondary rays
(reflection, refraction, shadow) — which is why the same scene is byte-identical
with the front-end on, and why a default flip is a smaller change than it sounds.

**THE PERF HALF IS NOT MEASURABLE ON THIS MACHINE, AND THE DEFAULT THEREFORE STAYS
OFF.** Three interleaved runs per arm on the shadowed PN scene, warm rows, taking
the minimum across reps (the usual drift-robust statistic):

    kernel / stage            min(off)   min(on)   delta   flag can reach it?
    raster_shadow_trace        35.247    38.248   +8.5%    yes (secondary rays)
    raster_first_shade          1.025     1.130  +10.2%    yes
    raster_shadow_event_build    0.450     0.498  +10.7%    yes
    raster_tri_count            0.430     0.467   +8.6%    NO  (rasterizer)
    raster_tri_write            0.504     0.574  +13.9%    NO  (rasterizer)
    raster_bez_count            0.503     0.549   +9.1%    NO  (rasterizer)
    raster_bez_write            0.504     0.557  +10.5%    NO  (rasterizer)
    merge + build BVHs          0.419     0.485  +15.8%    NO  (host, no kernel)

**The last five rows are the control, and they move as much as the first three.**
`WATERTIGHT_TRI` changes exactly one `@ti.func` in `raytrace_kernels_taichi`; it
cannot alter a bezier count kernel or a host-side BVH build. So the +8-16% is
drift, not cost, and it has a specific cause: the runs went off, on, off, on, off,
on, so the `off` arm always occupied the cooler slot of each pair while the machine
heated (every kernel rises monotonically across reps in BOTH arms). Interleaving
that never varies the ORDER is not interleaving.

And the honest limit is structural, not a matter of running more reps:
`WATERTIGHT_TRI` is read at **import**, because it changes the compiled kernel
body. So the in-process alternating A/B this project mandates for exactly this
problem is **impossible for this flag** — one process can only hold one arm.

What would settle it, for whoever picks this up:

* a machine that is not thermally throttling (this one reported SW thermal
  slowdown at 85 C throughout), or
* order-balanced repetition (off,on,on,off,... ) with enough reps to average the
  ordering bias out, or
* an instruction/traversal-step counter, which would also settle §3.4 and §3.6 and
  is the single highest-leverage instrument missing from this area.

**So §3.2's correctness is qualified and its cost is not.** The default stays off,
and §3.3 stays blocked — which is the conclusion §3.3 already reaches from the
other direction: flipping now would promote an intersection routine whose cost
nobody has measured to being the only one available.

3.3 Delete the epsilon apparatus  [BLOCKED — not startable, see below]
------------------------------------------------------------------------
`BARYCENTRIC_EPSILON`, `TRIANGLE_EDGE_EPSILON`, the `edge_hit` flag bit
(packing documented at `raytrace_kernels_taichi.py:1708`, frees a bit), `seam_t` (`rs_sca[r, 3]`,
frees a per-ray f32) and the 8 call sites with their initialisations and bounce
resets. `rs_sca` shrinking moves the arena fit — re-check `memory_model` (§4.7).

**Deliberately not attempted, and the dependency is structural rather than a
matter of effort.** Deleting the epsilons removes the *shipped* arm of
`_tri_hit`, which makes the watertight path mandatory. That path is default off
because it is unqualified: §4.7's CUDA runs have not happened, and nothing has
measured what its extra branches cost in the innermost loop of three traversal
kernels. So the chain is: qualify §3.2 on CUDA → flip its default → *then* this
becomes a deletion rather than a behaviour change. Doing it now would silently
promote an unmeasured intersection routine to the only one, which is the
opposite of what a gated rollout is for.

What *can* be done ahead of that, and was: §3.2 now routes both arms through one
`_tri_hit`, so the deletion is a single function body plus the constants, rather
than eight independently drifting call sites.

**THE SCOPE ABOVE IS WRONG, AND THE CORRECTION MATTERS BEFORE ANYONE STARTS.**
"Deleting the epsilons" is not one deletion gated on §3.2, because the two
constants have consumers **outside the ray path entirely**, which `_tri_hit`
never touches. Grep before planning:

    constant / state              consumer                              gated by
    -------------------------------------------------------------------------------
    BARYCENTRIC_EPSILON           _tri_hit's shipped arm                WATERTIGHT_TRI
    BARYCENTRIC_EPSILON           raster_taichi projected acceptance    UNGATED
    BARYCENTRIC_EPSILON           raster_taichi per-sample MT fallback  UNGATED
    TRIANGLE_EDGE_EPSILON+seam_t  the three ray-path sites              shipped arm
    TRIANGLE_EDGE_EPSILON+seam_t  raster_first_shade / shadow_event     ti.static(not aa_tri)
    seam_t (rs_sca[r, 3])         both paths                            with the above

So flipping §3.2 unblocks the **ray-path** half and nothing else. The two
ungated `BARYCENTRIC_EPSILON` reads in `raster_taichi` are the raster
front-end's own candidate-acceptance test, live in the default primary-visibility
path, and they are a *different mechanism*: the exact fixed-point sample
partition downstream is what makes coverage watertight there, while this dilation
decides which triangles are candidates at all. Removing it needs its own
argument and its own measurement — the raster path has no `_tri_hit` and gets no
benefit from Woop-Benthin-Wald.

The raster path's `TRIANGLE_EDGE_EPSILON`/`seam_t` pair is easier: it compiles in
only under `ti.static(not aa_tri)`, i.e. the non-analytic-AA fallback, so it
disappears with that fallback rather than with §3.2.

Net: §3.3 is **two** deletions with different owners, and `rs_sca` only shrinks
by its f32 when both have landed. Plan it that way, or the "one f32 per ray"
saving will be claimed and not delivered.

3.4 Median-split STBVH for bezier circuits  [BUILT, MEASURED, stays off]
--------------------------------------------------------------------
Once resolution is order-independent, `stbvh.py:302`'s reason for pinning
bezier to Morton is gone (PN, the other pinned type, no longer exists).

**LANDED**, gated `ALGAN_BEZ_BVH_SPLIT`, default off, flipping the bezier
default to `"split"` (both the main tree and the opaque one, or the two disagree
about instance order). ~20-25% fewer traversal steps is the claim inherited from
the triangle tree; **nothing here measured it**, because a traversal-step count
needs the kernel profiler and this container has no GPU. Default off because a
circuit's seam de-dup is discovery-order sensitive, so the reorder moves output
at the epsilon level — it is a performance change with a pixel cost, and both
halves need a CUDA machine to judge.

Note there is **no remaining slot-order freeze to undo**: the "every patch keeps
its slot" constraint was PN-specific and went with the PN merge block in
`b49b01b`. The only value-order reorder left in the merge is `_split_promotable`
grouping promoted triangles by material value (`scene_builder.py:572`), which is
unrelated to BVH build order.

**MEASURED ON CUDA: byte-identical, and the speed-up is not there to find on this
scene.** `benchmarks/_bez_bvh_ab.py` renders 35 independent circuits plus `Text`
and `Tex`, moving, at `--res md`, and compares the reorder against the scene's own
**run-to-run noise floor** (§4.8's point: byte-identity is the wrong gate because
split pixels are not reproducible in general):

    noise floor   off vs off   max 0,  0 px over tolerance, 60 frames
    A/B           off vs on    max 0,  0 px over tolerance, 60 frames
    wall          off 2.78s    on 2.76s    ratio 0.993x

So two things, and they pull in opposite directions:

* **The feared cost does not appear.** §3.4 was held off because "a circuit's seam
  de-dup is discovery-order sensitive, so the reorder moves output at the epsilon
  level". On this scene it moves output by **nothing at all** — and the noise floor
  is also zero, so that is a real byte-identity rather than a diff hidden under
  jitter. Flipping it would need no baselines.
* **The claimed benefit does not appear either.** 0.993x on a 2.8 s render is
  noise. The "~20-25% fewer traversal steps" is inherited from the triangle tree
  and is *still* unmeasured, because nothing counts traversal steps and this
  scene's traversal is not where its time goes.

One instrument note, because it produced a confident wrong number first: the
build-time column read `0.000s` in both arms, since `TIMERS` only records stages
something has wrapped and nothing wraps them outside `profile_scene`. The harness
now calls `install_pipeline_hooks()` itself. A profiler that reports zero because
it was never installed is worse than one that reports nothing.

**Recommendation: leave it off.** Not because of risk — it is byte-identical here
— but because the case for it was a perf claim, and after measurement there is no
perf evidence either way. Flip it when something measures traversal steps, or
when §5.2's order-independence work needs it; do not flip it on the strength of an
inherited number.

3.5 Flip `ALGAN_MESH_ID` on  [DONE, default ON]
-------------------------------------------------------------------------
§4.5 is measured. On the scored `|actual-E|` metric it is **neutral**: nothing
regresses, nothing gains, and the fill-rule and dump checks pass with it on.
Reference-free it is **slightly positive**: the packed-grid case gains in the
predicted direction (18 of 36224 pixels, `off − on` positive), its
non-overlapping control moves zero pixels, and the `Icosahedron` movement that
looked like a cost is mostly §3.7's winding defect (235 pixels down to 11 with
the winding gate on). Every remaining CPU question here is answered.

So this is blocked only on someone deciding the correctness argument is worth a
re-baseline. Note the packed-grid gain **depends on the dice fix in §4.5** — a
`Surface`'s `mesh_ids` reached nothing before it, so a flip on an older tree
would have bought the `Polyhedron` half of §2.2 and none of the packed half.

**FLIPPED**, together with §3.7. `tests/fast`'s CPU baseline was regenerated
here and the diff was looked at first: the change is confined to thin silhouette
outlines, **435 pixels of 278784 at the worst frame** (0.16%), peak deviation 53
channel values, mean 243 pixels per frame over 32 frames. That is the "up to 49
channel values at solid edges" the settings comment predicted, and it is the
intended effect — coarser runs resolve a solid's edge differently.

**THE CUDA BASELINE DEBT IS PAID.** Both CUDA sets were regenerated on a GTX 1050
(driver 576.52), and the machine established its right to own them first: with
`ALGAN_MESH_ID=0 ALGAN_POLYHEDRON_WINDING=0` it reproduces the pre-branch CUDA
baseline with the **same sha256** (§4.9), and four of the six full-render scenes
pass unchanged. The two that did not were attributed before anything was
overwritten — one to a master-side commit, one to a 2-pixel bloom epsilon (§4.1).

Both sets now correspond to the shipped defaults, which since this round include
§6.6 and §3.1. The movement was reviewed frame by frame before regenerating, not
just measured: side-by-side panels of the worst-differing frame of the fast scene,
`solids_and_camera` and `materials_and_lighting` are visually indistinguishable,
with the difference confined to silhouette outlines and interior mesh edges and
**no notches, rims or other artifacts** (`benchmarks/_diff_frame.py` writes the
panel). On `materials_and_lighting` the difference is a broad low-amplitude field
over the bloom halo rather than a localized error — bloom spreading a small
coverage change, which is why its pixel count (13.6%) is the largest of the six
while its peak (53) is not.

What §6.6 + §3.1 move, isolated against the previous defaults:

    scene                       max|d|   worst-frame px      frames
    manim_compat_and_plots           0        0 (0.000%)      0/171
    shapes_and_timeline             55     7928 (2.844%)     68/301
    complex_hierarchy_become        54    10554 (3.786%)     71/75
    solids_and_camera               88    22793 (8.176%)    228/239
    text_and_media                  47    30013(10.766%)    163/182
    materials_and_lighting          53    37964(13.618%)    159/179

**`manim_compat_and_plots` moving by exactly zero is the mechanism confirming
itself**: it is built from bezier circuits, and the one-mesh rule requires every
fragment in the pixel to be an opaque *triangle* of one surface. The only scene the
rule cannot touch is the only scene that did not move.

**Remaining debt: the two CPU sets, and this machine may NOT pay it.**

* `tests/fast/expected_outputs_cuda/` — **regenerated (this round).**
* `tests/full_renders/expected_outputs_cuda/` — **regenerated (this round).**
* `tests/fast/expected_outputs_cpu/` — **stale for §6.6/§3.1, and must be
  regenerated elsewhere.**
* `tests/full_renders/expected_outputs_cpu/` — same.

The reason is measured, not assumed. Rendering `tests/fast` on **CPU** here, at
exactly the settings the committed CPU baseline was written with (`2d1432a`:
MESH_ID and winding on, ONE_MESH and weld off), misses that baseline by **30
channel values over 0.86% of pixels on 43 of 45 frames** — *before* any change in
this round. So this machine's CPU output is simply not the portable one, and
regenerating from here would replace a baseline that CI reproduces with one it does
not.

Two checks make that reading sound rather than a guess:

* The difference is **not** feature-shaped. The diff panel puts it on text glyph
  edges, circuit outlines and mesh edges alike — everything in the frame, faintly
  — which is what host float math differing between machines looks like, not what
  a stale feature looks like.
* It is **not** `35fe6ec` staleness either, which was the obvious suspect. That
  commit verified bit-identity **on CPU** and only moved CUDA, and the measurement
  agrees: the committed CPU baseline sits 53 channel values from a CUDA render of
  the same code, and this machine's CPU render sits 53 from it too. Both CPU
  renders are the same distance from CUDA; they differ from *each other*.

**So the practical consequence, said out loud: CI runs `tests/unit_tests tests/fast`
on a CPU-only runner, and `tests/fast` will fail there until the CPU set is
regenerated on a machine of that lineage.** Note this is not a regression this round
introduced on *this* machine — the CPU baseline already failed here beforehand —
but §6.6 does move CPU output, so the set is genuinely stale for the runner. Whoever
has a CPU-only box of the CI lineage should run:

    ALGAN_UPDATE_FAST_BASELINE=1 <venv-python> -m pytest -q tests/fast
    ALGAN_UPDATE_FULL_RENDER_BASELINES=1 <venv-python> -m pytest -q tests/full_renders

and render twice, baselining the second (§4.10). §7.17 has the two traps that make
a CPU baseline check on a CUDA machine silently lie.

3.6 Two-level BVH (TLAS/BLAS)  [design only — NOT attempted, scoped below]
---------------------------------------------------------------------------
See §5.2. Blocker to clear first: `_split_promotable` (`scene_builder.py:572`)
reorders promoted triangles by material value, so a partly-promoted surface
already lands in two disjoint spans; per-mesh contiguity has to exist before a
BLAS is meaningful.

**Scope, so the next person can decide rather than discover.** This is the one
item in §3 that is not a gated switch. It needs, at minimum:

* a per-mesh contiguity guarantee in the merge, which means either reversing
  `_split_promotable`'s material grouping or making promotion mesh-aware —
  `scene_builder.py` is ~2100 lines and the merge's field layout is load-bearing
  for every kernel ("do not casually change merged-field widths, ordering, dtype
  or lifetime");
* a two-level build in `stbvh.py` (~840 lines), which today builds one flat
  instance tree per geometry type;
* two-level traversal in `raytrace_kernels_taichi.py` (~3340 lines), in the
  megakernel *and* the wavefront path, plus the raster path's own gather.

That is a project measured in days with a CUDA machine for the perf case that
justifies it, not a session's work, and a half-landed version is worse than
none: a TLAS that does not actually reduce traversal steps costs a build per
batch for nothing. Left unstarted on purpose.

**THE PERF CASE HAS NOW BEEN MEASURED, AND IT DOES NOT JUSTIFY STARTING.**
`benchmarks/_pn_deletion_profile.py` at `--res md` on CUDA — five solids covering
every curved family, shadows on, everything moving — puts the render's device time
here:

    stage / kernel                     warm time      share
    raster_shadow_trace                  41.33 s     80.2%
    raster_first_shade                    1.23 s      2.4%
    raster_tri_write                      0.62 s      1.2%
    raster_bez_write                      0.61 s      1.2%
    raster_bez_count                      0.60 s      1.2%
    raster_shadow_event_build             0.55 s      1.1%
    raster_tri_count                      0.53 s      1.0%
    merge collections + build BVHs        0.49 s      0.9%
      - of which the refit-BVH build      0.28 s      0.5%

(Warm/RUN 2, which is the column to read; the cold rows in the same report put
`raster_first_shade` at 38% because it is paying its own JIT there. See §7.14.)

§5.2 offers two motivations for a two-level BVH, and the measurement bounds them
separately:

* **"Per-mesh BLAS reusable across a batch's frames, since the STBVH rebuilds per
  batch."** That is an amortization argument, and what it can amortize is the
  build: **0.4% of the render.** Even a perfect BLAS cache is worth at most that,
  against a multi-day project with a structural blocker. This half is dead.
* **"True instancing — a point cloud of 10k spheres becomes one BLAS plus 10k
  transforms."** Not bounded by the above, because it attacks traversal, which
  *is* large (`raster_shadow_trace` is 35.4% and is pure traversal). But this
  scene has **five** instances, so it cannot show the win, and no workload in the
  repo has thousands of repeated meshes. The win is real in principle and
  unmeasurable in practice until such a workload exists.

Two further reasons not to start, from outside this document:

* `DESIGN_optimization_targets.md` is the plan of record for render performance,
  and **BVH build and traversal appear nowhere in its rankings.** Its measured
  poles are batch prep at 73.6% of `save_video` against the render thread's 56.7%,
  and its named top items — `AttributeTimeline.get`, the batched surface build,
  `set_state_to_times` — are all CPU prep. A two-level BVH targets neither pole's
  top item.
* **Nothing counts traversal steps.** §3.4 ran into the same wall: its inherited
  "~20-25% fewer traversal steps" could not be confirmed or refuted. Any TLAS work
  would be flying blind in exactly the way the doc warns against.

**Recommendation: leave §3.6 unstarted, and if anyone wants to revisit it, build
the traversal-step counter first.** That is a day's work rather than a week's, it
is useful on its own (it would also settle §3.4), and it converts this item from a
guess into a decision. Starting the TLAS without it risks precisely the outcome
this section already names — a build per batch for nothing.

3.7 Orient `Polyhedron` faces outward  [DONE, default ON]
------------------------------------------------------------
§6.5 is the measurement, including the part where its predicted interaction with
§3.5 was measured and refuted. It was not fixed by hand-reversing the four hardcoded
index lists, because the same broken lists reach Algan through user data and
through every Manim script and `Polyhedron` is public API. It orients at
construction —
flood-fill winding consistency across shared edges (a consistently oriented pair
of faces traverses their shared edge in opposite directions), then flips the
whole shell if the signed volume comes out negative — and **no-ops** when the
input is not a closed orientable manifold (any undirected edge not shared by
exactly two faces, a flood fill that contradicts itself, a shell in more than
one piece, or zero volume). That fixes any closed polyhedron, convex or not, and
leaves open and non-manifold input alone.

**LANDED**, gated `ALGAN_POLYHEDRON_WINDING`, now default **ON** (§3.7), surfaced as
`SETTINGS.raytracing.experimental.set(polyhedron_winding=...)`. Implemented as
described above in `shapes_3d.orient_faces_outward`, called from
`Polyhedron.__init__`. `tests/unit_tests/test_mesh_identity.py` pins the defect
itself (the per-solid inward counts, so a face-list edit cannot change them
quietly), that the pass fixes all five solids without changing which vertices a
face uses, that it declines on open / non-manifold / degenerate input, and that
it repairs a deliberately mis-wound and a wholly inverted tetrahedron.

**It moves a `become` morph, and nothing above covers that.** Reversing an
inward face reverses the vertex order *within* it, and `become` pairs primitives
corner by corner, so the interpolation path changes. Measured:
`Tetrahedron.become(Cube)` differs by up to **227** channel values across the
gate, while a *static* `Tetrahedron` is byte-identical and
`Tetrahedron.become(Tetrahedron)` is byte-identical too — there the reordering
cancels on both sides, which is why the first probe of this looked clean and the
mechanism took a full-render investigation to find. The endpoints are the correct
solids either way; only the in-between path moves. This is the whole of
`complex_hierarchy_become`'s 197-channel movement in `tests/full_renders`
(attributed: MESH_ID alone passes that scene, winding alone reproduces the 197).

Measured, and **not** what §6.5 first predicted: with `ALGAN_MESH_ID` off the
fast-suite render is **byte-identical** across this gate (same sha256, and that
scene draws a `Cube`, an `Icosahedron` and an `Octahedron`). A per-triangle
surface id makes every run one fragment, so the facing bit groups nothing and
flipping it changes nothing downstream. With `ALGAN_MESH_ID=1` the render does
change — which is the mechanism stated plainly: one id per solid leaves facing
as the only thing separating the two sheets.

**FLIPPED**, together with §3.5, and the two were re-baselined as one change
because their effects overlap (a per-solid `sid` is what makes the facing bit
load-bearing at all). `tests/full_renders` could not be used as the gate it was
meant to be: those baselines are not this machine's, and all six scenes fail
here at the shipped defaults with every gate off. See §3.5 for the exact
baseline debt this leaves.


================================================================================
4. WHAT MUST BE VERIFIED ON A CUDA DEVICE
================================================================================

Clear the Taichi cache (`clear_cache(taichi_kernels=True)`) before any A/B — it
does not invalidate on `@ti.func` edits. Never edit `*_taichi.py` while a render
or a warm daemon is running.

**ALL OF §4 HAS NOW BEEN RUN ON CUDA** (GTX 1050, driver 576.52, Taichi 1.7.4,
torch 2.7.1+cu128), with the Taichi cache cleared first. Each item below carries
its result. `1035 passed, 89 skipped` on `pytest -q tests/unit_tests`.

4.1 **Confirm the committed CUDA baselines still pass.** — **DONE.** The premise
    was wrong: §3.5/§3.7 *did* move output, deliberately, so at shipped defaults
    `tests/fast` fails (49 channel values, 6847 px of 278784 at the worst frame,
    27 of 45 frames) and four of six `full_renders` scenes fail. The useful run
    is therefore §4.9's, which pins the gates off; see there. What matters is
    that every difference is attributed, and all six are:

        scene                     gates off      the two flips move it
        ------------------------------------------------------------------
        complex_hierarchy_become  PASSES         206 values, 4.6% px
        solids_and_camera         PASSES          99 values, 8.3% px
        shapes_and_timeline       PASSES           0
        text_and_media            PASSES           0
        manim_compat_and_plots    fails, 220       0    <- NOT this branch
        materials_and_lighting    fails,   3       0    <- epsilon, see below

    Note this differs from the CPU attribution in §3.5, where `materials_and_lighting`
    (47) and `shapes_and_timeline` (96) also moved under MESH_ID. On CUDA they do
    not move at all: the flips change which runs form, and whether that changes a
    *pixel* turns on borderline comparisons that differ by device. Two scenes
    move here, four on CPU.

    `manim_compat_and_plots` is `35fe6ec` from master, and it is an improvement —
    §7.13 has the two-diff test that establishes it. `materials_and_lighting` is
    **2 pixels of 278784 in 1 frame of 179, by 3 channel values**, on near-black
    pixels ([2,12,22] against [5,15,25]) in a scene carrying glow + bloom +
    tonemapping: the bloom-amplified epsilon pattern. Deterministic — two
    independent passes differ from the baseline identically and from each other
    by zero.

4.2 **Confirm the PN deletion is byte-identical on CUDA** — **DONE, and the
    proposed method was unnecessary.** No stash and no pre-deletion tree is
    needed, because the committed CUDA baselines *are* the pre-deletion tree's
    output: they were written by `efb3a95`, which is this branch's base and
    therefore sits before `b49b01b`. So §4.9's gates-off run is already the
    comparison, and it comes back with the **same sha256** as
    `tests/fast/expected_outputs_cuda/fast.mp4`. The PN deletion, the watertight
    refactor, the weld, the bezier split and the harness work are byte-identical
    on CUDA with their gates off.

    (Superseded method, kept for the reasoning:

        git stash && pytest -q tests/fast && sha256sum tests/fast/algan_outputs/fast.mp4
        git stash pop && pytest -q tests/fast && sha256sum tests/fast/algan_outputs/fast.mp4

    It would also have been *weaker* than what was available: with both gates
    flipped since, a stash of the working tree no longer isolates the deletion,
    while the baseline does. Prefer "which committed artifact predates the
    change" over "can I reconstruct the old tree".)

4.3 **Confirm the kernels did not get slower.** — **DONE: neutral.**
    `benchmarks/_pn_deletion_profile.py`, run once per tree (a `git worktree` at
    `efb3a95` for the pre arm) with a **separate `ALGAN_CACHE_DIR` per arm**,
    because the offline cache does not invalidate on `@ti.func` edits and both
    trees compile identically-named kernels. Gates pinned off in both — and note
    the pre tree warns that `ALGAN_MESH_ID` / `ALGAN_POLYHEDRON_WINDING` are
    unknown variables, which is the right answer: they did not exist yet, so the
    pre arm *is* the gates-off configuration.

    Device times, `--res md`, five solids covering every curved family with
    shadows on and everything moving. **These are the WARM (RUN 2) numbers** —
    `profile_scene` renders twice and writes both, and the cold rows come first in
    the file, which is a trap worth knowing (§7.14):

        kernel / stage             pre        post     delta
        raster_shadow_trace     41.454 s   41.330 s    -0.3%    (80.2% of the run)
        raster_first_shade       1.228 s    1.232 s    +0.3%
        raster_shadow_event      0.527 s    0.552 s    +4.7%
        raster_tri_count         0.521 s    0.528 s    +1.3%
        raster_tri_write         0.617 s    0.623 s    +1.0%
        raster_bez_count         0.595 s    0.600 s    +0.8%
        raster_bez_write         0.605 s    0.611 s    +1.0%
        merge + build BVHs       0.587 s    0.489 s   -16.7%    (0.9% of the run)
        end-to-end              51.31 s    51.55 s    +0.5%

    **Neutral, with the one predicted win where it was predicted.** The kernel that
    dominates (80% of the warm render) moves -0.3%, everything else moves under 5%
    on sub-second absolute numbers, and the **BVH build drops 16.7%** — which is
    exactly the two-fewer-trees-per-batch saving §2.1 removed, showing up in the
    stage that owns it. End to end it is +0.5%, i.e. nothing.

    So "neutral to faster" was right in shape and small in size: the freed builds
    are real, and they were ~1% of the render to begin with (§3.6). The deletion's
    case is ~2800 lines and byte-identical output, not speed.

    Two caveats on the instrument, because they bound what this can claim. The
    machine throttles (`nvidia-smi` reported SW thermal slowdown at 85 C
    throughout), and a cross-tree comparison cannot be in-process, which is what
    §4.3 asked for. The cold rows in these same reports moved by +5% to +15% in
    every kernel *including ones neither tree changed*, which is the size of the
    drift this setup carries. Trust the warm rows and the direction, not a few
    percent.

4.4 **Confirm the compile surface shrank.** — **DONE, and it did not.** Both arms
    compile **13 offline-cache entries** for the same scene, in fresh per-arm
    caches. Not a contradiction once stated plainly, and §2.1 already contains the
    reason: `has_pn` was a template dimension only one value of which was ever
    instantiated, because `RayTracedPNTrianglePrimitive` was unreachable and
    `merged["num_pn"]` was always 0. Removing a variant nothing compiled removes no
    cache entries and no compile time.

    So §4.4's expectation was inconsistent with §2.1's own premise. The compile
    surface shrank in *source* (four kernels lost a template parameter), which is a
    maintainability win, not a build-time one. Do not quote a compile-time saving
    for this deletion.

4.5 **`ALGAN_MESH_ID=1` — measured NEUTRAL on coverage.**
    The arbiter this asked for now exists and is CPU-runnable:
    `_aa_run_gate_check.py`'s `|actual-E|` column (§6.3) compares the *rendered*
    coverage — replayed per-sample transmittance and all — against an EXACT
    analytic reference, which is precisely what the old per-fragment error
    metric could not do. Run both ways:

        for m in 0 1; do ALGAN_MESH_ID=$m <venv-python> \
            benchmarks/_aa_run_gate_check.py --res md --verify 4; done

    Measured, `--res md`, CPU, mean |actual-E| over silhouette pixels:

        case               MESH_ID=0   MESH_ID=1
        quad (control)        0.0020      0.0020   (declares no identity)
        cube                  0.0250      0.0248
        icosahedron           0.0258      0.0256   (0.0264 -> 0.0262 with
                                                    §3.7's winding gate on)
        cylinder              0.0260      0.0260   (a Surface is already
        cylinder (256x2)      0.0211      0.0211    one merged member, so
        sphere (192x96)       0.0383      0.0383    its sid does not move)

    **Nothing regresses, and nothing gains beyond noise.** So the coverage
    evidence neither blocks the flip nor argues for it, and the case for
    MESH_ID rests where §2.2 put it — a `Cube`'s face diagonal ought to be an
    interior edge, a packed grid's distinct spheres ought not to be unioned —
    plus §5.2's unlocks, not on a measured quality win.

    **Read this before quoting an earlier number.** A previous revision of this
    section reported the icosahedron going 0.0492 → 0.0231 and called MESH_ID
    qualified. That was wrong, and it was the *reference* that was wrong, not
    the walk: `_exact_coverage` then accepted a mis-wound pixel whose two sheets
    had landed in one facing group, reporting double its true coverage, which
    both inflated the icosahedron's error and made MESH_ID look like it halved
    it. The gate is now the fill rule's own property — within one sheet the
    masks partition the samples, so a facing group whose masks overlap is
    holding two sheets and the pixel is dropped — and every row prints its drop
    count. The other five rows were unaffected and did not move.

    **The packed-grid experiment — RUN, and it found a defect first.** The six
    cases above are all one solid, so none of them was the end §2.2 fixes in the
    *other* direction: a packed-grid `Surface`, one merged member covering every
    packed sphere. Two cases now cover it, both a 4×4 grid of `Sphere`s flattened
    by `batch_mobs` into one packed grid:

        packed 4x4 (apart)     centres 0.75 apart, radii summing to 0.56, so no
                               two footprints can touch — the CONTROL
        packed 4x4 (overlap)   centres 0.45 apart, alternating depth, so adjacent
                               footprints genuinely overlap

    The first run came back **byte-identical** between `ALGAN_MESH_ID=0` and `=1`,
    and the reason was not that identity does not matter — it was that
    **`Surface`'s declared `mesh_ids` were never read by anything.** A packed
    grid is diced logical PN, `_pack_projected_flat_geometry` gives the dice's
    `_logical_pn_tri_obj` priority over `_rt_obj_ids`, and `_dice_logical_pn`
    built its patch→surface map from the per-member `_rt_obj_counts` alone. For a
    lone packed primitive — one member covering every sphere — that is a single
    id, so the whole pack diced to one surface and the `mesh_ids`
    `Surface.get_render_primitives` stamps (`surface.py:2618`, added by §2.2)
    were resolved correctly at construction and then discarded. **Fixed**: the
    dice now consults the declaration first, in the same order as the flat path.
    Gated behind `MESH_ID`, so the default path is untouched, and
    `test_declared_shells_survive_the_logical_pn_dice` renders a frame and reads
    the merge's own `tri_obj` to pin it (it fails without the fix — checked, not
    assumed).

    **What the fixed measurement says.** The scored `|actual-E|` barely moves
    (overlap 0.0340 → 0.0340), but that column cannot settle this case: on a
    packed grid the pixels `_exact_coverage` must **drop** are exactly the
    overlapping ones, which is the population at issue. So the harness grew a
    reference-free A/B (`--mesh-ab`) that differences painted coverage per pixel
    between the two settings — no reference, so it sees the dropped pixels too:

        case                 covered px   moved   max |d|   mean off−on
        quad (control)            33438       0    0.0000       +0.0000
        cube                      39914      17    0.0885       +0.0001
        icosahedron               46220     235    0.4968       −0.2098
        cylinder / (256x2)      43124/43228     0    0.0000       +0.0000
        sphere (192x96)           27734       0    0.0000       +0.0000
        packed 4x4 (apart)        43560       0    0.0000       +0.0000
        packed 4x4 (overlap)      36224      18    0.2002       +0.0539

    The packed prediction is **confirmed in sign and mechanism, and small in
    population**: 18 of 36224 pixels, and `off − on` is *positive*, meaning
    MESH_ID=0 paints more — the over-claim §2.2 predicts, where one id for the
    whole pack lets a run carry across two spheres until their masks OR to a full
    union and `corr` short-circuits to 1. The `apart` control moves **zero**
    pixels, which is what makes that reading sound: the effect is the packing,
    not the batching. The scored rows agree at the margin (overlap `split`
    48 → 34 pixels as runs stop at the sphere boundary).

    **This also re-reads the icosahedron.** Its 235 moved pixels at mean |d|
    0.21 were the strongest evidence *against* MESH_ID. With §3.7's winding gate
    on they collapse to **11 pixels at mean |d| 0.024** — so nearly all of it was
    the winding defect, not MESH_ID. That does not resurrect the refuted
    prediction in §6.5 (the *scored* metric is still neutral either way, and
    MESH_ID still does not "pay"); the two instruments simply see different
    populations, because a mis-wound pixel is one `_exact_coverage` drops. Quote
    them together or neither.

    **Net.** On coverage the flip is neutral-to-slightly-positive: a small
    genuine gain on packed grids, no measurable cost anywhere once winding is
    fixed. The case for it still rests on §2.2's correctness argument and §5.2's
    unlocks rather than on a quality win — but the one case that was supposed to
    show a win does show one, in the predicted direction.

    Corroborated, both with `ALGAN_MESH_ID=1`: `_analytic_aa_fillrule_check.py`
    reports `FILL_RULE_OK: True` over 256000 pixel tests with 0 samples claimed
    by both or neither, and `_aa_dump_check.py` passes all nine checks including
    resolve/shadow lockstep (worst golden-walk error 2.75e-08).

    Whenever it is flipped, it moves the fast-suite render by up to 49 channel
    values at solid edges, so **both** device baseline sets have to be
    regenerated and `expected_outputs_cuda/` needs a GPU.

4.6 **Shadow-mode agreement — a testable prediction.** Three `SHADOW_ANYHIT`
    modes disagree today in corner cases documented as seam-merge artifacts
    (`raytrace_kernels_taichi.py:2337` and 2535). Once identity replaces
    the epsilon (§3.3) those disagreements should vanish:

        for m in 0 1 gather; do ALGAN_SHADOW_ANYHIT=$m pytest -q tests/full_renders; done

    Diff the three outputs; they should become identical. If not, there is a
    second cause worth isolating before §3.3 ships.

    **RUN, AND THE PREDICTION IS UNTESTABLE BY THIS INSTRUMENT.** All three modes
    already produce the **identical sha256** on CUDA — 0 channel difference over
    179 frames, and `materials_and_lighting` is the only scene in the suite that
    turns shadows on, so it *is* the suite's shadow coverage. There is nothing
    here to vanish.

    That is not the same as the corner cases being gone, and the difference
    matters. Read the kernel's own docstring
    (`raytrace_kernels_taichi.py`, `_shadow_occlusion`): the early-out is "not
    strictly byte-identical to the plain march in two corner cases the early-out
    deliberately overrules":

    1. an opaque edge hit the seam merge would have folded into an earlier
       translucent edge hit within `DEPTH_TIE_EPSILON` — **identity-related**;
    2. an opaque blocker past `MAX_SURFACES_PER_RAY` (= 256) peeled surfaces —
       **not identity-related at all**.

    So §3.3 could only ever remove the first, and the second-cause hunt §4.6 asks
    for is already answered by reading: it is the peel depth. Note also the
    docstring's last clause — "in both the any-hit's full occlusion is the
    physically correct answer" — so the disagreement is a deliberate improvement,
    not a defect, and "they should become identical" was the wrong goal for case 2.

    What the prediction actually needs is a **purpose-built scene**, because the
    suite does not reach either case: a translucent stack whose edge hits sit
    within `DEPTH_TIE_EPSILON` of an opaque hit (case 1), and a >256-surface
    translucent stack (case 2). Until such a scene exists, "the three modes agree"
    is a statement about the scene, not about the renderer.

    **BUILT: `benchmarks/_shadow_anyhit_check.py`. Case 2 is reached, and the
    three modes agree anyway.** The scene stacks 304 translucent sheets between an
    off-axis light and an opaque blocker, and it demonstrably reaches the peel
    limit — rendering the same scene 8 sheets deep instead of 304 moves `max|d|`
    to 255, so the depth is doing something. With the case reached, all three
    modes still produce the **identical sha256**. So the disagreement case 2
    documents does not appear in the shipped configuration, and the docstring's
    corner case is narrower than it reads, or unreachable from the public API.

    The likely reason, **stated as a hypothesis and not measured**: an opaque
    blocker is found through the opaque-only BVH prepass, which does not peel
    translucent surfaces at all, so the peel depth never gates whether the
    blocker is seen. Anyone continuing here should check that before building a
    third scene.

    **Case 1 is still untested, and its green result should not be read.** The
    tie scene renders and all three modes agree on it, but nothing yet
    demonstrates it puts an opaque edge hit within `DEPTH_TIE_EPSILON` of a
    translucent one — there is no reach check for case 1 the way there is for
    case 2. An agreement column from a scene that may reach nothing is exactly
    the false negative this section already fell into once, so it is recorded as
    unproven rather than as agreement.

    **The first run of this harness was itself that false negative**, which is
    why the reach check exists: the stack scene's light was directly overhead, so
    its shadow fell on ground the camera sees edge-on at the horizon. Three modes
    agreed, on a frame with no shadow in it. `_shadow_stack.png` in the first run
    showed a fully lit floor and nobody would have noticed from the numbers.

4.7 **Watertight test (§3.2), once built.** — **RUN; see §3.2 for the verdict.**
    `benchmarks/_watertight_check.py` covers the first two items. Note one setup
    trap it hit first: forcing the ray path (`hybrid_raster=False`) allocates
    per-ray state for every pixel, so at `--res md` with the usual 1.4 GB pin it
    raises `OutOfRenderMemory` on "a single frame". `--res ld` with a 2.2 GB pin
    fits on a 4 GB card. The third item is measured as device time rather than
    occupancy, because Nsight does not support this machine's Pascal GPU.

    Original plan, for reference:
    * **No cracks in f32.** Large adjacent triangles at grazing incidence plus a
      finely diced welded `Sphere` at extreme silhouette; assert zero background
      pixels interior to the mesh. Extend `_analytic_aa_fillrule_check`'s
      partition property to the ray path.
    * **No double blend.** Translucent `Sphere`/`Cylinder` at several alphas;
      interior edges must show no brightness ridge. `_aa_dump_check` on rim
      pixels.
    * **Register pressure.** The ray-space transform is per-ray hoistable but
      adds live state; check occupancy against the 21–25% resolve ceiling
      (`DESIGN_hybrid_raster.md` §13).
    * **`rs_sca` shrinks by one f32 per ray** when `seam_t` goes, which moves
      the arena fit: `test_render_batch_sizing.py`, `test_memory_model.py`, and
      a long multi-batch render checking OOM-retry counts do not regress.

4.8 **Median-split bezier BVH (§3.4), once built.** — **DONE.**
    `benchmarks/_bez_bvh_ab.py` was written for it, and it compares against the
    scene's own noise floor rather than against byte-identity, which is §4.8's
    point. Result in §3.4: byte-identical, 0.993x wall, and the traversal-step
    claim is still unmeasured because nothing counts traversal steps.
    Recommendation there is to leave it off.

    Note byte-identity turned out to be *available* on this scene — the noise
    floor measured zero too — so the caution above ("split pixels are not
    byte-reproducible") is about scenes with PBR/coverage-miss branches, not about
    circuits. Check the noise floor before concluding a diff is a change.

4.9 **Every gate off is byte-identical** — **DONE, on CUDA, byte-exact.**
    `ALGAN_MESH_ID=0 ALGAN_POLYHEDRON_WINDING=0` with the four opt-in gates at
    their defaults reproduces `tests/fast/expected_outputs_cuda/fast.mp4` with the
    **same sha256**, and four of the six `full_renders` scenes pass unchanged (the
    two that do not are §4.1's, neither from this branch). This is the load-bearing
    result of §4: it says the whole branch is inert until a gate is flipped, on the
    device where the kernel variants actually live.

4.10 **Render twice, baseline the second.** The first render on a fresh machine
    populates the Manim Tex geometry cache and its `MathTex` glyph antialiasing
    differs from every run after it — 18 channel values over 100 frames of
    `text_and_media`, against a tolerance of 2.

    **Measured here, and this machine does not show it on the fast scene.** With
    the whole cache wiped (`clear_cache(taichi_kernels=True)` takes the Manim
    caches with it), run 2 and run 3 of `tests/fast` are byte-identical to each
    other and to the committed baseline. So the rule is still the right default —
    it costs one render and the failure mode is a baseline nobody can reproduce —
    but the effect is scene-specific, and `text_and_media` is where to look for
    it, not `fast`.


================================================================================
5. WHAT THE SYSTEM ENABLES
================================================================================

5.1 Delivered
-------------
* **A much smaller renderer** — ~2800 lines, `pn_patch.py`, 12 merged keys, two
  BVH builds per batch, four epsilons and a template dimension, output
  byte-identical.
* **One fewer route rejection** (`num_pn > 0`, as an always-true clause in five
  places).
* **Correct identity for polyhedra and packed grids**, and topological shells
  for imported meshes — **shipped on** (§3.5).
* **A way to ask questions about the run rule** and get population answers
  instead of anecdotes — which is what turned two plausible theories into §6.
* **`tri_obj` is under test.**
* **Coverage that is no longer sample-quantized** (§6.6, shipped on): error
  against an exact analytic reference down 70-100% on eleven cases, and the share
  of silhouette pixels landing on a multiple of 1/8 down from 8-91% to 0-1.6%.
  Ink wobble on a diced `Cylinder` down 78%, on a flat quad 63%. Costs ~4%.
* **A host/kernel gate that cannot drift again** — `aa_grp` has one definition and
  one predicate, with an AST audit that fails if anything else reads the raw
  setting. This was a live bug costing most of §6.6's win (§6.6.1).
* **Six new measurement harnesses**, each answering a question §4 asked and could
  not: `_one_mesh_ab` (what §6.6 costs), `_weld_check` (the textured-surface risk),
  `_bez_bvh_ab` (a reorder against its own noise floor), `_watertight_check`
  (cracks and double blends, counted), `_pn_deletion_profile` (cross-tree device
  times), `_diff_frame` (looking at a re-baseline rather than measuring it).

5.2 Unlocked by the identity, worth building next
-------------------------------------------------
* **Order- and window-independent output.** Once the greedy `seam_t` rule is
  gone, resolution is a function of the canonically sorted hit list alone —
  independent of KBUF width, BVH builder, tile size and batch window. This is
  the property the rework was asked for, and the precondition for §3.4 and for
  ever reordering primitives in the merge.
* **Nested-IOR refraction.** A stable mesh id at every hit lets a ray carry an
  "inside which mesh" stack, so glass-inside-glass and a sphere inside a box get
  the correct *relative* IOR at each interface instead of assuming air outside.
  `wavefront_kernels_taichi.py` currently special-cases thin panes because it
  cannot reliably tell an entry from an exit.
* **Robust self-shadow rejection.** A shadow ray can reject its own mesh at
  near-zero `t` by identity rather than by `MIN_HIT_DISTANCE = 1e-4` plus a
  normal offset — removing shadow acne at grazing light angles and on
  small-scale geometry, and removing another scale-dependent epsilon.
* **Material dispatch coherence.** Sorting hits by mesh id groups identical
  material evaluation, which is what `WAVEFRONT_SORT_MATERIALS` wants.
* **Exact absorption of coincident duplicates.** A union of sample masks is
  idempotent, so two genuinely coplanar stacked quads stop double-darkening.
* **Geometry that is actually watertight** (§3.1).
* **Two-level BVH (TLAS/BLAS).** Per-mesh BLAS reusable across a batch's frames
  for rigid meshes (the STBVH rebuilds per batch today), true instancing (a
  point cloud of 10k spheres becomes one BLAS plus 10k transforms instead of
  10k copies of the geometry), and per-mesh culling.


================================================================================
6. MEASURED NEGATIVE RESULTS — READ BEFORE BUILDING HERE
================================================================================

`benchmarks/_aa_line_check.py` reports the symptom this work was partly aimed
at: a tessellated `Cylinder` scores **0.0568 px** of ink wobble against
**0.0138** for a flat two-triangle quad, and **0.0773** when diced to
`resolution=(256, 2)` — worse the finer it gets. Two plausible causes were built
and measured. **Neither is it.**

6.1 The consecutive-run requirement is not the problem
------------------------------------------------------
The obvious theory: `_aa_run_scan` takes a maximal *consecutive* run of
`(sid, facing)`, so a sheet whose fragments interleave with another's gets
corrected against a partial `Q`. Replaying the grouping for every covered pixel
puts `split` at **0.00–0.02%** on every case and `capped` under 1%. The grouping
is sound.

Regrouping it into an order-independent equivalence class is still worth doing
for §5.2's order-independence — which is what unblocks §3.4 — but **it will not
move any AA metric.** Do not motivate it as a quality fix.

6.2 The union-full short-circuit is real but too small to matter
----------------------------------------------------------------
What *does* scale with tessellation density is v2 §4.2's
`U == _AA_MASK_ALL → corr = 1`, as a fraction of covered pixels:

    flat quad          1.0%
    Cylinder default  25.2%
    Cylinder (256,2)  72.4%
    Sphere (192,96)   87.6%

Almost all of it is the benign interior tiling the short-circuit exists for
(`1 - E` is float dust: 343 / 10770 / 31096 / 23282 pixels). The residual is a
genuinely dilated silhouette tail of **1 / 105 / 181 / 1004** pixels with
`1 - E` up to 0.15 (0.30 on the sphere).

Consulting `E` there was implemented — on that path `Q == 1`, so `corr = E/Q` is
just `E`, with a 1e-3 dust band keeping genuine tilings bit-identical — and
measured:

    default Cylinder  wobble 0.0568 -> 0.0566   rms 0.0094 -> 0.0099
    fine Cylinder     wobble 0.0773 -> 0.0781   rms 0.0164 -> 0.0166

Neutral at best, marginally worse on rms. A few hundred pixels cannot move a
frame-wide metric, which the dust bucket dominating every histogram already
implied. **Not shipped; the code was reverted rather than left as a dead gated
path.** If you want it back, the shape was: widen `aa_grp` from 0/1 to 0/1/2
(every existing `ti.static(aa_grp)` test is truthiness, so 2 is safe and costs
no new kernel argument), and branch on `ti.static(aa_grp == 2)` inside the
`rU == _AA_MASK_ALL` arm at **both** lockstep sites in `raster_taichi.py`
(`raster_first_shade` and `raster_shadow_event_build`).

6.3 ANSWERED — the pixel lands on the ownership answer, but the magnitude
    that would move it off is being *discarded*, not missing
--------------------------------------------------------------------------
The hypothesis was that this is a *representation* limit rather than a bug in
the run rule: eight sample positions cannot resolve a silhouette crossed by a
dozen sub-pixel triangles however exactly each area is known. **The symptom is
exactly that. The diagnosis is not** — see §6.3.2, which was measured after
§6.3.1 and supersedes its prescription. Read all three parts before acting.

`_aa_run_gate_check.py` now replays `raster_first_shade`'s per-sample
transmittance walk in Python, for every covered pixel, and compares the coverage
the pixel actually ends up with against the **exact** area of (footprint ∩
pixel) — summed from one sheet's exact clipped areas, with the other sheet
required to agree or the pixel dropped. No supersampled reference, no fitted
model. `--verify` proves the replay against the kernel's own `ALGAN_AA_DUMP`
rows rather than asserting it: worst per-fragment `eff` difference 5e-8 over six
cases. Mean over silhouette pixels, `--res md`, CPU:

    case               silh  |actual-E|  |own-E|  |actual-own|  on-lattice
    quad (control)      827      0.0020   0.0390        0.0370        7.9%
    cube                947      0.0250   0.0405        0.0241       51.0%
    icosahedron         898      0.0258   0.0407        0.0174       59.5%
    cylinder           2307      0.0260   0.0367        0.0116       72.5%
    cylinder (256x2)   2139      0.0211   0.0329        0.0128       70.6%
    sphere (192x96)    2628      0.0383   0.0408        0.0047       90.8%

`own` is `popcount(union of every fragment mask)/N` — the pixel's coverage with
all magnitude information discarded. `on-lattice` is the share of silhouette
pixels whose painted coverage is an exact multiple of 1/N.

Read it as: **on the flat control the magnitude machinery works** (error 0.0020
against 0.0390 for ownership alone, so 95% of the sample quantization is
removed, and only 7.9% of pixels land on the sample lattice). **On a diced
closed mesh it is neutralized** — the sphere's painted coverage sits 0.0047 from
the pure-ownership answer and 91% of its silhouette pixels land exactly on
eighths. The signed error is positive in every case: dilation, which is what
`_aa_line_check` reads as ink wobble. The control is what makes this reading
sound; without it "the error is near the ownership answer" could just mean that
answer happens to be good.

`own` is **not** a floor for this architecture, and §6.3.2 is where that
matters: it is the floor for a scheme carrying no magnitude at all, and the run
correction produces off-lattice coverage wherever it is allowed to run.

Two mechanisms produce it, and the by-verdict line separates them:

* **`full`** — 52% of the sphere's silhouette pixels, mean error 0.042. ONE
  fragment owns all N samples while covering less than the whole pixel, so the
  run scan never starts (v2 §4.2 gates on a partial mask) and the pixel is
  painted at 1.0. Its exact area sits unread in `frag_cov`.
* **The far-sheet re-claim.** A run's `corr < 1` scales the occlusion write as
  well as the claim, so the samples the near sheet owns keep a residual
  transmittance — standing for the part of the pixel the sheet does not cover,
  which at a silhouette lies OUTSIDE the mesh entirely. The residue has no
  position, so the far sheet of the same solid claims it, uncorrected (`svis` is
  no longer uniform, so its own run cannot engage). Measured on one cylinder
  pixel: near sheet claims 0.2396 (exact, `corr` 0.9583), far sheet adds 0.0104,
  pixel lands on 0.2500 = 2/8 against a true 0.2394. The harness's `1sheet`
  column suppresses it: **0.0250 → 0.0041 on the cube** (84% of the error), but
  only 0.0383 → 0.0346 on the sphere, where `full` dominates.

  This is the *opaque* face of something `DESIGN_analytic_aa.md` §16.6 already
  recorded for translucency — "scalar transmittance treats a mesh's two sheets
  as independently overlapping rather than as one sub-area seen twice".

Both are magnitude thrown away rather than magnitude unavailable, but neither is
reachable by the run rule as scoped: the first never enters it, and the second
needs to know that two sheets belong to ONE mesh — which is what §2.2 declares
and no consumer yet reads.

6.3.1 The sample count is the live lever — measured
----------------------------------------------------
`_AA_SAMPLES` is a compile-time constant rather than a setting
(`raster_taichi.py:213`), so the experiment is: edit that line to
`_AA_PATTERN_16`, clear the Taichi cache, re-run. Done, same machine, `--res md`
(`_AA_DUMP_COLS` must become `16 + _AA_NUM_SAMPLES` or the dump writes off the
end of its buffer):

    ink wobble (px)        8 samples   16 samples
    bezier Line               0.0042       0.0042   (SDF coverage, no masks)
    flat quad                 0.0138       0.0141
    Cylinder                  0.0568       0.0391    -31%
    Cylinder (256, 2)         0.0773       0.0543    -30%

    |actual-E| (harness)   8 samples   16 samples
    quad (control)            0.0020       0.0028
    cylinder                  0.0260       0.0126
    sphere (192x96)           0.0383       0.0236

**The flat control does not move and the diced meshes improve ~30%.** That is
the signature of an ownership-limited error, and it is the first thing measured
in this area that moves the metric §6 is about — §6.2's `consult E` moved it by
0.4%.

This is NOT a recommendation to ship 16. `DESIGN_analytic_aa.md` §16.4 measured
8-vs-16 on a different metric (L1 against an aa=4 reference over four configs)
and found a wash bought at ~30% more device time in `raster_tri_count` /
`raster_tri_write` / `raster_first_shade`, plus a regression on the `thin`
config, and concluded 8 ships. Nothing here overturns the cost side; what it adds
is that the *benefit* is concentrated exactly on the case §6 is chasing, which
§16.4's aggregate metrics could not see. A sample-count change is a
`DESIGN_analytic_aa.md` decision, not a mesh-identity one.

**And §6.3.2 makes it the wrong lever to pull first anyway.**

6.3.2 THE ACTUAL FIX — let the run rule see full-mask pixels
--------------------------------------------------------------
The `full` verdict is the largest single contributor (52% of a fine `Sphere`'s
silhouette pixels) and it is excluded from the run rule *by the run rule's own
gate*: v2 §4.2 starts the lookahead only when the first fragment's mask is
partial, so a pixel whose first fragment owns all N samples never scans, never
computes `E`, and is painted at 1.0 however little of the pixel its sheet
covers. That gate exists for the hot path — an interior pixel is one full-mask
fragment and must not pay for a lookahead — and an interior full-mask fragment
has `cov` within float dust of 1. So the gate can be relaxed to

    partial mask  OR  (full mask AND cov < 1 - 1e-3)

which leaves the interior hot path untouched and admits exactly the silhouette
pixels. The scan's `rU == _AA_MASK_ALL` arm then takes `corr = E` (`Q == 1`
there), which is §6.2's rule finally reaching the pixels that needed it.

Replayed in the harness as the `|cF-E|` column:

    |actual-E|         shipped   16 samples   relaxed gate
    quad (control)      0.0020       0.0028         0.0000
    cube                0.0250            -         0.0214
    icosahedron         0.0258            -         0.0120
    cylinder            0.0260       0.0126         0.0030    -88%
    cylinder (256x2)    0.0211            -         0.0030    -86%
    sphere (192x96)     0.0383       0.0236         0.0060    -84%

The flat control becomes **exact**. This is worth far more than doubling the
sample count and costs no samples and no interior work.

**BUILT, AND THE PREDICTION ABOVE DOES NOT SURVIVE CONTACT.** Implemented as
specified (`ALGAN_ANALYTIC_AA_RUN_FULL`, default off) and measured on CPU. Read
this before quoting the `relaxed gate` column: two things were wrong with it.

*The premise is false: the donors are not there to be summed.* The emission
truncates a pixel's fragment list at its first **full-mask** opaque fragment
(`raster_pipeline.py`, "a full-mask fragment occludes every sample whatever its
exact area says"). So the run scan the relaxed gate starts on a full-mask
fragment can never reach that sheet's empty-mask area donors — they were
discarded before the resolve ran. `E` comes back as the one fragment's area, and
the pixel is darkened by `1 - E`. At a silhouette that is the intended fix; in an
**interior tiling** it is a notch, and the two are indistinguishable after the
cut. Measured before the mitigation: **531 interior pixels of a flat quad and
920 of a `Cylinder` darkened by a mean 0.027**, with `_aa_line_check` getting
uniformly worse (default `Cylinder` 0.0568 → 0.0639 at 33°, flat quad
0.0060 → 0.0134 at 26.6°).

That makes the shipped `corr = 1` short-circuit **load-bearing rather than
lazy**: after truncation, a full sample mask is the renderer's only remaining
evidence that the sheet tiles the pixel. §6.3.2 read it as an approximation to
improve, and it is a compensation for information the emission already threw
away.

*The mitigation works but shrinks the win to nothing on the target case.*
Requiring a fragment to own every sample **and** cover the pixel before it
truncates (gated behind the same flag) lets the donors survive. Notches drop to
0 on every `_aa_run_gate_check` case **that existed when this was written** —
and that qualifier is load-bearing, because it does not hold for the cases added
since. Re-measured on CUDA with `--notch-probe`, the relaxed gate WITH its
mitigation still notches four of them (§6.6.2 has the table): a fine `Sphere`
2 -> 22, `line-check cyl` 0 -> 4, `line-check cylfine` 50 -> 239, `packed 4x4
(overlap)` 0 -> 3. The flat quad and both plain `Cylinder`s are still zero in
every arm, so the original measurement was right about the geometry it covered.
**This is the open item**, and §6.6 inherited it by implying this gate — for a
while the residue was recorded as the one-mesh cap's, which it is not.

*DIAGNOSED: the residue is `_AA_MAX_RUN_SCAN`.* The scan sums at most 16
consecutive fragments of a sheet. When it stops early, `E` is a **lower bound on
the sheet's area, not the sheet's area** — and the relaxed gate's `rU ==
_AA_MASK_ALL` arm then takes `run_corr = min(rE, 1.0)`, so an interior pixel
whose sheet genuinely tiles it is darkened by exactly the unscanned remainder.
Measured by replaying each notched pixel's own fragments with the scan limit
lifted and nothing else changed (`--notch-probe`):

    case                  notched   paints full unbounded   mean paint
    line-check cylfine      253            231              0.99102 -> 0.99967
    sphere (192x96)          24             13              0.99823 -> 0.99940
    line-check cyl            4              0              0.99898 (unchanged)
    packed 4x4 (overlap)      3              0              0.99865 (unchanged)

So **244 of the 277 notches on the two cases that carry them are the scan
limit**, and the remaining seven pixels across the two small cases are something
else, still unattributed.

*Do not read the verdict column to find them.* `_classify` returns the FIRST
matching label and tests `union-full` before `capped`, so a pixel that is both
reports as `union-full` — which is why the notched pixels look like a union-full
population (189 of 253) while their mean run length is 24.22, well past the limit
of 16. The verdict histogram sent this diagnosis down a wrong path once.

*Two fixes, and the choice is not obvious.* (a) Raise the limit: it is one
constant, but it is a loop bound in the megakernel's hot path and the cap exists
deliberately. (b) Refuse to consult `E` when the scan hit its limit, falling back
to the shipped `corr = 1` short-circuit — cheap and principled, since a truncated
sum is not an area, but it also withdraws the gate's win from every long-run
SILHOUETTE pixel, and on `cylfine` those are most of the frame (`capped` is 3011
of 3546 clean interior pixels). Neither is free; (b) needs the silhouette
population measured before it is chosen, and both need a kernel recompile and a
cost measurement this box cannot resolve (§7.15). But with real donors in `E`, the coverage
win shrinks — `Cylinder` 0.0260 → 0.0080 rather than → 0.0030 — and the metric
§6 is actually about barely moves. Mean ink wobble over the nine non-degenerate
angles, `--res md` CPU:

    kind        shipped   relaxed    delta
    bezier Line  0.0042    0.0042   +0.0000   (circuits never enter the run rule)
    flat quad    0.0138    0.0051   -63%
    Cylinder     0.0568    0.0563    -1%
    Cylinder fine 0.0772   0.0781    +1%

**So it is a real win on FLAT triangle geometry and does nothing for a diced
mesh** — the opposite of what it was built for. A flat quad has no far sheet and
loses no donors, so the relaxed gate is clean there; a diced closed mesh is
dominated by the far-sheet re-claim, which this does not touch. The `|cF-E|`
column that predicted −88% was computed with the shipped truncation in place,
i.e. against fragment lists whose donors were already gone, and
`_aa_run_gate_check` scores **silhouette pixels only**, so it could not see the
notches either. Both instruments have since been fixed: the harness now counts
interior notches beside the win, and the replay follows `aa_grp == 2` so
`--verify` compares like with like (8 cases pass, worst `eff` diff 3e-8).

One open question, deliberately not chased further: with the mitigation on, a
crude LUT-based image diff still finds ~355 interior pixels of the
`_aa_line_check` quad strip darkened, while the harness's exact-area notch
counter finds **zero** on its own cases. The two disagree because they are
different geometry and different instruments; the quad's wobble improves 63%
regardless. Resolve that before flipping the default.

Scope it to the **run**, not the fragment. A full-mask fragment owns every
sample, so by the fill rule the rest of its sheet in that pixel owns none — they
are empty-mask area donors whose area is real, and only the run's `E` counts
them. Both were measured; on the sphere fragment scope reaches 0.0255 and run
scope 0.0060, and on the two flat solids (no donors) they coincide.

Why it is measured rather than built here: it moves output, so it needs a gate
plus regenerated baselines on both devices, and it is a `DESIGN_analytic_aa_v2`
change rather than a mesh-identity one. The implementation shape is the one §6.2
already sketched — widen `aa_grp` from 0/1 to 0/1/2 (every existing
`ti.static(aa_grp)` test is a truthiness test, so 2 is safe and costs no new
kernel argument) and change the scan gate plus the `rU == _AA_MASK_ALL` arm at
**both** lockstep sites in `raster_taichi.py` (`raster_first_shade` and
`raster_shadow_event_build`; any divergence desynchronizes every shadow id).
Qualify it with `_analytic_aa_fillrule_check.py`, `_aa_dump_check.py`,
`_aa_line_check.py` and this harness, and look at the diff videos.

One caution, from §21.3: reconciling EVERY fragment's magnitude against its
exact area put 5920 notches into a mesh. A full mask is exactly the case where
that argument does not apply — the fragment owning all N samples is alone in its
sheet's sample partition, so there is no neighbour to disagree with — but "the
argument does not apply" is not a proof, and `_analytic_aa_fillrule_check` is.

Note also what this does **not** fix: the two flat solids barely move
(`cube` 0.0250 → 0.0214), because their error is the far-sheet re-claim, not the
`full` gate. That one still wants the mesh-level union rule, and therefore §2.2's
identity. The two halves of §6.3 have different owners.

6.6 THE ONE-MESH RULE — Line-quality on a Cylinder, and where it breaks
------------------------------------------------------------------------
This is what §2.2's identity was built to enable and what no consumer read
until now. `ALGAN_ANALYTIC_AA_ONE_MESH`, now default **ON**, implies §6.3.2's
relaxed gate — see §6.6.1, where that implication turned out to be only half
wired.

**The rule.** Where every fragment in a pixel is an OPAQUE triangle of ONE
surface, the pixel's coverage is that mesh's NEAR SHEET's exact area and nothing
else. So once a facing has committed ink, the other facing commits none. The
host marks those pixels (a segment reduction over the CSR it already has) and
carries the flag in a spare `frag_msk` bit, so no kernel argument changes; it
rides as `aa_grp = 3`.

**What it fixes.** The run rule's `corr < 1` scales the OCCLUSION write as well
as the claim, so the samples the near sheet owns keep a residual transmittance
standing for the part of the pixel the sheet does not cover. That residue lies
OUTSIDE the mesh, but it carries no position, so the far sheet of the same solid
claims it as though it were background — uncorrected, because `svis` is no
longer uniform and its own run cannot engage.

**Measured, `--res md` CPU.** Coverage error against the exact reference goes to
zero almost everywhere, and `on-lattice` — the share of pixels landing on a
multiple of 1/8 — collapses with it. That second number is the one that answers
"is it still sample-based":

    case                 |actual-E| shipped -> one-mesh   on-lattice
    quad (flat control)      0.0020 -> 0.0000              7.9% -> 0.0%
    cube                     0.0248 -> 0.0000             51.4% -> 0.1%
    icosahedron              0.0262 -> 0.0000             57.6% -> 0.0%
    cylinder                 0.0260 -> 0.0000             72.5% -> 0.0%
    cylinder (256x2)         0.0211 -> 0.0000             70.6% -> 0.1%
    sphere (192x96)          0.0383 -> 0.0072             90.8% -> 3.7%
    line-check cyl (33deg)   0.0299 -> 0.0000             57.6% -> 0.0%
    packed 4x4 (overlap)     0.0340 -> 0.0017             80.8% -> 1.4%

And on the metric §6 is actually about, mean ink wobble over nine
non-degenerate angles:

    bezier Line   0.0042 -> 0.0042    (unchanged; circuits never enter this)
    flat quad     0.0138 -> 0.0051    -63%
    Cylinder      0.0568 -> 0.0039    -93%   <- below the bezier Line
    Cylinder fine 0.0772 -> 0.1650   +114%   <- REGRESSION, see below

**A default `Cylinder` now beats the bezier `Line`** on the metric the Line was
winning by an order of magnitude. That is the goal met.

**Where SUPPRESSION broke, and what replaced it.** The first form of this rule
suppressed the far sheet outright, and `cyl_fine` — `resolution=(256, 2)` on a
0.045-radius rod, 256 facets around a shape ~9 px wide — regressed **+114%**:
signed error flipped to −0.0344 (under-covering), 1676 of 3508 interior pixels
notched by up to 0.41.

Two hypotheses were tested and **both refuted**, so nobody spends the effort
twice:

* *"The facing bit is noise on sub-pixel facets."* The fill rule's partition
  test — within one sheet no sample may be claimed twice — was implemented as a
  host gate and fires on **zero** pixels: 100% of both the coarse and the fine
  rod's pixels pass it. The bit is not scrambled. Removed rather than left as
  dead code.
* *"It is the u-seam."* `ALGAN_WELD_SURFACE_SEAMS=1` changes the sphere's
  residual by nothing at all, to the last digit.

The **premise** was what failed. `|1sheet-E|` on the fine rod is 0.0392 against
an `|actual-E|` of 0.0192 *at shipped settings* — suppressing the far sheet is
already worse there before any of this code runs. "Both sheets project to the
same silhouette, so coverage is the near sheet's area" holds strictly INSIDE a
silhouette; at the boundary the near sheet's projected area shrinks toward zero
while the footprint does not, and on a rod that thin diced that finely nearly
every pixel is boundary (`capped` 59.5%, `split` 12.5%).

**So the shipped rule CAPS rather than suppresses.** The mesh may claim at most
`max(front_area, back_area)` in total, a per-pixel ceiling the host computes from
the same exact areas and carries per fragment in `frag_cap`. Well inside a
silhouette the two sheets tile to the same area, the near sheet fills the
ceiling, and the far sheet gets no room — suppression recovered exactly. At the
boundary the ceiling leaves precisely the room the near sheet does not fill.
One exclusion, and it is not a fudge: `run_mode == 2` (the pristine all-sliver
claim) maintains its own `run_claimed` renormalization against the run-start
transmittance, and clipping its `eff` without adjusting `run_pd` desynchronizes
that bookkeeping — measured, it was the whole of the sphere's `--verify`
divergence (6.3e-4 → 2.2e-8).

Ink wobble, mean over nine non-degenerate angles, `--res md` CPU:

    kind          shipped   suppress      cap
    bezier Line    0.0042     0.0042    0.0042   (circuits never enter this)
    flat quad      0.0138     0.0051    0.0051   -63%
    Cylinder       0.0568     0.0039    0.0039   -93%, below the bezier Line
    Cylinder fine  0.0772     0.1650    0.0411   -47%  (was +114%)

The cap keeps every win suppression had and turns its one regression into an
improvement. Interior notches follow: `cyl_fine` 1676/3508 at mean 0.0978 under
suppression, **234/3508 at mean 0.0092** under the cap, and zero on six of the
eleven cases.

**Read the `|cap-E|` column with care — it is partly circular.** The ceiling is
`max(front, back)` over the exact areas and `_exact_coverage`'s truth is
essentially the same formula on the same numbers, so a small `|cap-E|` shows the
walk CAN land on the exact-area answer, not that that answer is right at a
grazing boundary. The independent evidence is the ink-wobble table above, which
does not consult `_exact_coverage` at all.

**Open, and not papered over.** `--verify` passes on 9 of 11 cases (worst
6e-7) and still fails two — `line-check cyl` at 1.1e-4 and `packed 4x4 (apart)`
at 5.6e-4. Both diverge on a single **sliver** fragment (`msk` empty, the areal
donor path) whose `eff` is below `MIN_ALPHA`, so neither the kernel nor the
replay commits it and no rendered pixel differs. The mechanism is unexplained;
the likely shape is the same bookkeeping mismatch the `run_mode == 2` exclusion
fixed, since a sliver is the other branch that writes areally rather than by
sample. It was left failing rather than excluded, because adding exclusions
until a check passes is how an integrity check stops being one.

6.6.1 MEASURED ON CUDA — and the "implies" was only half wired
---------------------------------------------------------------
Everything above was CPU. Reproducing it on CUDA found a **bug in the gate**, so
read this before quoting any number in §6.6.

*The bug.* §6.6 says the rule "implies §6.3.2's relaxed gate", and on the kernel
side it does: `aa_grp = 3` and `_aa_run_full` returns true for 2 **or** 3. But
§6.3.2's other half is a HOST change — the emission must stop truncating a
pixel's prefix at a full-mask fragment, or the run scan cannot see its sheet's
empty-mask area donors — and that test read `ANALYTIC_AA_RUN_FULL` **alone**. It
was written by `517c842` (the RUN_FULL commit) and neither one-mesh commit
updated it. So `ALGAN_ANALYTIC_AA_ONE_MESH=1` by itself ran the relaxed scan over
fragment lists whose donors had already been discarded, which is exactly the
configuration §6.3.2 measured as an interior notch. Ink wobble, `--res md`, CUDA:

    kind           shipped   ONE_MESH alone   ONE_MESH + RUN_FULL
    bezier Line     0.0042           0.0042                0.0042
    flat quad       0.0139           0.0128  (-8%)          0.0052  (-63%)
    Cylinder        0.0568           0.0301 (-47%)          0.0124  (-78%)
    Cylinder fine   0.0765           0.0427 (-44%)          0.0429  (-44%)

The two flat/coarse cases are the ones the relaxed gate carries, and they lose
most of the win; `cyl_fine`, which the CAP carries (`capped` 59.5%), is
unaffected. That split is what identified the bug.

*The fix.* `aa_grp` is now computed once by `raster_pipeline._aa_group`, and the
truncation tests `_aa_run_full(aa_grp)` — the same predicate the kernels test —
so the host and the kernel can no longer disagree about whether the relaxed gate
is active. With it, `ONE_MESH=1` alone reproduces the `+ RUN_FULL` column
exactly. `tests/unit_tests/test_analytic_aa_gates.py` pins it, including an AST
audit that only `_aa_group` may read the raw setting; the audit was checked to
FAIL with the bug reintroduced, because an audit nobody has seen fail is not one.

*What reproduces, and the one claim that does not.* Coverage error against the
exact reference, `--res md`, CUDA, with the fix:

    case                 |actual-E| off -> on     on-lattice off -> on
    quad (flat control)      0.0020 -> 0.0000       7.9% -> 0.0%
    cube (flat)              0.0248 -> 0.0041      51.3% -> 0.0%
    icosahedron (flat)       0.0262 -> 0.0022      57.6% -> 0.0%
    cylinder (default)       0.0260 -> 0.0005      72.5% -> 0.0%
    cylinder (256x2)         0.0211 -> 0.0005      70.6% -> 0.0%
    sphere (192x96)          0.0382 -> 0.0012      90.8% -> 0.4%
    line-check cyl (33deg)   0.0298 -> 0.0020      57.6% -> 0.0%
    line-check cylfine       0.0168 -> 0.0050      79.2% -> 1.6%
    line-check quad (33deg)  0.0035 -> 0.0000      15.3% -> 0.1%
    packed 4x4 (apart)       0.0313 -> 0.0024      72.5% -> 0.2%
    packed 4x4 (overlap)     0.0340 -> 0.0020      80.8% -> 0.6%

The off column matches the CPU numbers above to the last digit on every case, so
the harness is device-consistent and the coverage win is real on both devices.

**But §0's headline does not survive CUDA.** "A `Cylinder` now anti-aliases
better than a bezier `Line`" rested on 0.0039 against the Line's 0.0042. On CUDA
the best available is **0.0124**, three times the Line, with the gate correctly
wired and both flags set. The improvement is large (-78%) and the ordering claim
is wrong; do not repeat it. Nothing here explains the CPU/CUDA gap on that one
figure, and the two flat cases and `cyl_fine` all reproduce, so it is a single
unexplained outlier rather than a systematic device difference.

6.6.2 THE DESYNC IS FIXED — and it was ONE symptom, not three
---------------------------------------------------------------
**Shipped ON as `ALGAN_ANALYTIC_AA_ONE_MESH_DENS`.** The cap clipped a
fragment's CLAIM and left its OCCLUSION write alone: in `raster_first_shade`,
`alpha = mat_alpha * eff` uses the capped `eff` while `a_s = mat_alpha * dens` --
the per-sample transmittance write -- used the **uncapped** `dens`. So a capped
fragment hid more background than it painted, and the pixel lost that energy.

The fix is one line at each of the two clamp sites: scale `dens` by the same
ratio the cap applied to `eff`. It rides as `aa_grp = 4`.

The obvious objection is that the far sheet is really there and really does
occlude, so its write should stand. It should not. The near sheet's own `dens`
already occludes everything the mesh covers, and the residue the far sheet was
consuming stands for area OUTSIDE the mesh — occluding it twice is the same
double-count on the occlusion side that §6.6 removes on the claim side.

**MEASURED, CUDA, `_aa_run_gate_check --res md --verify 40`.** The desync is
gone, completely and on every case:

    arm        claim-vs-occlusion, over the 11 cases
    shipped    7.8e-06 .. 2.2e-01      (up to 22% of a pixel)
    with fix   1.1e-16 .. 5.4e-16      (float dust — where NO cap sits)

**AND THE OTHER TWO SYMPTOMS DID NOT MOVE, which refutes what this section used
to say.** The previous revision claimed one mechanism behind three symptoms and
that one fix would close all three. Measured:

    symptom                     shipped              with fix
    claim-vs-occlusion          7.8e-06 .. 2.2e-01   1.1e-16 .. 5.4e-16   FIXED
    interior notches            24 / 4 / 253 / 3     24 / 4 / 253 / 3     unchanged
    --verify failures           5, worst 9.6e-04     5, worst 9.6e-04     unchanged
    ink wobble (9 angles)       .0042/.0052/.0124/.0429   identical       unchanged

The refutation is **structural, not bad luck**, which is why it should never have
been predicted: `notched` is counted from `actual`, `--verify` diffs the `effs`
sequence, and ink wobble reads rendered ink. All three are the CLAIM. This fix
changes only the occlusion write, so it could not have moved any of them, and a
minute spent reading the harness's own accumulators would have said so.

**AND THE NOTCHES ARE NOT THE CAP'S AT ALL — they are §6.3.2's relaxed run
gate.** This section called them "the cap's claim-side shortfall" and that was
also wrong; `--notch-probe` settled it two ways that agree.

*Attribution by gate*, notches on INTERIOR pixels, `--res md`, CUDA (the seven
cases not listed are zero in every arm):

    case                  neither gate   relaxed gate ALONE   shipped (gate+cap)
    sphere (192x96)         2/23629          22/26480             24/26480
    line-check cyl          0/9050            4/10195              4/10195
    line-check cylfine     50/3546          239/3546            253/3546
    packed 4x4 (overlap)    0/28610           3/30531              3/30531

The relaxed gate carries **~92%** of the increase and the cap ~8%. That is not
new behaviour discovered here — §6.3.2 already recorded that the relaxed gate
notches interior tilings — but it *is* a correction to who owns the residue,
because `ONE_MESH` implies the gate and so inherited the blame for it.

*Attribution per pixel*, holding the fragment list fixed: replay a notched
pixel's own fragments with the clip disabled and nothing else changed. It
recovers **14 of 253** on `cylfine`, **2 of 24** on the sphere, and **0 of 4** and
**0 of 3** on the other two — the same 8%, arrived at independently. The mean
barely moves (`cylfine` 0.99102 -> 0.99109, which is 0.8% of a 0.00898
shortfall). The clip is a bystander.

The two instruments matter separately: the gate table changes the EMISSION, so it
cannot isolate the clip; the per-pixel replay holds emission fixed, so it can.
Neither alone would have been conclusive.

**So the open item is the relaxed gate's interior notches, not the cap's.** The
ceiling is not the lever — on `cylfine` it is *identical* on notched and clean
pixels (0.99972 both), which is the single cleanest statement of the negative
result. Anyone picking this up starts at §6.3.2, on a diced mesh's interior
tiling, and should not spend time on `frag_cap`.

What the fix DID change is what the residue looks like: an over-bitten interior
pixel used to render too DARK (paint 0.95, hide 1.00) and now shows that much
background instead. Both are wrong; the second is at least energy-conserving.
That was checked rather than argued — `benchmarks/_one_mesh_dens_ab.py` renders
every arm over `DARKER_GRAY` **and** over `WHITE`, because a bright background is
where bleed-through would be ugly, and the worst frames are visually
indistinguishable with the difference confined to silhouettes and shadow edges
(`max|d|` 43-66 over 1.4-3.1% of pixel-frames).

**Determinism holds.** A/A byte-identical on all four arms, twice. That is not a
formality here — §6.6.4 is a reproducibility bug in this same ceiling, found by a
freshly written baseline failing on the next render.

**Cost: not resolvable on this machine, and the first number was wrong.** A fixed
off,on,off,on ordering gave 1.022-1.054x. Alternating the ORDER on the same 40 s
shadowed scene gave **0.878x** — the ON arm apparently faster, which added work
cannot be. The two orderings straddle 1.0, so the measurement is thermal drift
(§7.15, and the same trap that produced a uniform 8-16% bias once before) and the
honest statement is "below this box's noise floor". Do not quote a percentage
until something measures it on hardware that is not throttling.


6.6.3 WHAT IT COSTS — measured, which nothing had done
-------------------------------------------------------
Three things are new when the rule is on: a host **segment reduction** over the
fragment CSR (two `scatter_reduce_`, two `scatter_add_`, a `repeat_interleave`),
a **per-fragment f32** (`frag_cap`), and a **running clamp** in the inner loop of
`raster_first_shade` and `raster_shadow_event_build`.

The f32 costs nothing: `frag_cap` is allocated unconditionally in both raster
paths already, so the arena footprint is identical in both arms and the memory
model does not move. The other two, `benchmarks/_one_mesh_ab.py`, alternating in
one process at `--res md` on CUDA:

    shape                      off       on     ratio
    diced (Sphere/Cyl/Torus)  2.06s    2.17s    1.052x
    flat (Cube/Icosa/Octa)    2.31s    2.36s    1.021x
    mixed + shadows          34.94s   36.27s    1.038x

**~2-5% slower, and the honest figure is the 1.038x.** The first two scenes
render in ~2 s, where fixed per-render overhead dominates and a few percent is
barely above noise; `mixed` is a 35-second shadowed render and exercises the
second resolve kernel, so it is the only row that measures the clamp rather than
the harness. Nothing here is free, and the trade is explicit: ~4% of render time
for coverage that stops being sample-quantized.

Output moves, so byte-identity is the wrong gate and was not sought:
`max|d|` 42 / 67 / 63 on the three shapes, over 1.7-3.5% of pixel-frames.

6.6.4 THE CEILING MUST NOT COME FROM A FLOAT ATOMIC — found by re-baselining
-----------------------------------------------------------------------------
Flipping §6.6 on made a render **non-reproducible**, and the pixel suites are what
caught it: after re-baselining `materials_and_lighting` from one render, the very
next render of the same configuration missed its own fresh baseline by **28 channel
values over 9.6% of a frame, on 28 of 179 frames**. Not a baseline error — the
scene simply did not render the same way twice.

Attributed, not guessed. Two renders with the rule OFF are **bit-identical (same
sha256)**; two with it on are not. So the rule introduced it.

*The mechanism, and why it is so much larger than it looks.* The host builds the
per-pixel ceiling with `scatter_add_`, which is a float atomic add, so its
summation order is not reproducible on CUDA — measured directly, a 400k-into-5k
reduction of this shape spreads **1.5e-05** across six runs and is never bitwise
equal. A 1e-05 wobble in a colour would be invisible. But this feeds a
**threshold**: the kernel clips only when `eff > frag_cap - mesh_ink`, so a ceiling
that moves in its low bits flips borderline fragments in and out of being clipped,
which is a *finite* coverage change — and this scene carries bloom, which spreads
each flipped pixel over a halo. That is the whole path from 1e-05 to 28.

*The fix.* Accumulate the two sheet areas in **float64** and round the ceiling back
to float32. Verified: the reduction is then bitwise stable over six runs (spread
0.0), and two full renders of `materials_and_lighting` come out with the same
sha256. The cast is what makes it robust rather than merely better — float64
reassociation error lands about nine orders below a float32 ulp, so it cannot
survive the round.

*Two dead ends, so nobody re-walks them.* "Use a scan instead of atomics" does not
work here: `torch.cumsum` on CUDA is **also** not bitwise reproducible on this build
(spread 0.0625 over 400k elements), while `torch.sum` is. And
`torch._segment_reduce` *was* bitwise stable, but it is a private torch API and not
worth depending on when a dtype change does the job.

*What it implies for the rest of the renderer.* Any host-side float reduction whose
result reaches a comparison is a latent nondeterminism bug of this shape. The
existing `_split_determinism_check` findings are the benign version — float atomic
adds into `pix_accum`, bounded at `|d| = 1` because they only ever perturb a colour
that is then truncated to `u8`. The dangerous version is a reduction that decides a
branch, and this was one.

**The fix that follows, not attempted here.** Scale `dens` by the same ratio the
cap scales `eff`: `k = room / eff` before clipping, then `eff = room` and
`dens *= k`. Then a capped fragment occludes exactly what it paints. Worth
reasoning through before building it, because the naive worry is wrong: well
inside a silhouette the near sheet's masks partition all N samples and `corr = 1`,
so `svis` is already 0 and the far sheet's `k = 0` costs nothing; at a boundary
the residual `1 - cap` is background OUTSIDE the mesh, which is what should show
through. That makes scaling `dens` the completion of the cap rather than a fudge,
and it should close the `--verify` failures too. It moves output, so it needs its
own gate, its own baselines and a re-run of this harness plus `_aa_line_check`.

Shipped ON; both CUDA baseline sets regenerated (§3.5).

6.4 This interacts with §4.5
-----------------------------
`ALGAN_MESH_ID=1` makes runs coarser, which puts *more* pixels through the
union-full branch. The two changes are coupled, and the arbiter has to be
rendered coverage against an exact reference, not a per-fragment error metric.
**That arbiter now exists** — the §6.3 harness is it, and `settings.py:485`'s
"not this harness" caveat is out of date. See §4.5.

6.5 `Polyhedron` does not wind its faces consistently
------------------------------------------------------
Found while building §6.3's exact reference, and load-bearing for §3.5.

`Polyhedron` builds each face from a hardcoded index list (Manim's, verbatim),
and those lists are not consistently oriented. Measured — outward test is
`dot(cross(p1-p0, p2-p0), face_centroid - solid_centroid) > 0`:

    Tetrahedron    2 of  4 faces wound inward
    Cube           0 of  6
    Octahedron     2 of  8
    Icosahedron   12 of 20
    Dodecahedron   3 of 12

The projected winding sign **is** `_AA_BACKFACE_BIT` (`raster_taichi.py:152`),
so on those solids the facing bit does not name a sheet. Measured on the
icosahedron: 960 of 46220 covered pixels have one facing group holding *both*
sheets — one such pixel sums that group to 1.98 while the true sheets tile to
1.0000 and 1.0000. The §6.3 harness drops those pixels rather than referencing
them wrongly, and reports the count.

Why this matters here rather than in the AA docs: the run rule groups by
`(sid, facing)`. Today `sid` is per-triangle for a `Polyhedron`, so a run is one
triangle and a broken facing bit is nearly harmless. Under `ALGAN_MESH_ID=1`
(§2.2) the whole solid becomes ONE `sid`, and then `facing` is the *only* thing
separating the near sheet from the far one — so a run can span both sheets and
sum their exact areas into one `E`.

**That predicted MESH_ID=1 would hurt the icosahedron and not the cube. It is
wrong**, and so was the follow-up guess that fixing the winding would make
MESH_ID pay. Both were measured, in the full 2x2, mean |actual-E| on the
icosahedron:

                       MESH_ID=0   MESH_ID=1
    winding as shipped    0.0258      0.0256
    winding fixed         0.0264      0.0262

The winding does not interact with MESH_ID at all on this metric. What it *does*
do is decide whether the pixel is measurable: the harness's own
sheet-decomposition check drops **960** of the icosahedron's 46220 covered
pixels as shipped and **4** with `ALGAN_POLYHEDRON_WINDING=1`. That number is
the evidence the orientation pass works, and it is the concrete cost of the
defect — on four of the five solids the facing bit names nothing, so anything
downstream that wants a mesh's near sheet cannot have it.

Keep both refutations. The first cost a plausible-sounding paragraph in this
file; the second nearly cost a second one.

The third prediction, that fixing it would move output, is wrong too: the
fast-suite render is byte-identical across `ALGAN_POLYHEDRON_WINDING` while
`ALGAN_MESH_ID` is off (§3.7). It only moves with MESH_ID on — which is the same
mechanism stated once more, since that is the only configuration where facing
groups anything.

The harness still gained an opaque `Cube` case (0 of 6 inward) as the
*referenced* polyhedron, so that the polyhedron family is measurable with the
gate off as well as on.


================================================================================
7. METHODOLOGY THAT COST REAL DEBUGGING TIME
================================================================================

7.1 `layer_offsets` has THREE consumers, not two
-------------------------------------------------
The packed `layer_offsets` array lost its PN slot and renumbered 8 → 7 entries.
Renumbering only the two obvious reads in `wavefront_kernels_taichi` left
`raster_first_shade` (`raster_taichi.py`, which reads env-map placement, far clip
and `max_bounces` from the same array) reading `max_bounces` **off the end**.
That silently changed every PBR reflection in the fast-suite render by up to 162
channel values. Always `grep -rn 'layer_offsets\['` before touching it.

7.2 A matching single-frame render is a FALSE NEGATIVE
-------------------------------------------------------
While hunting 7.1, `save_frame` at two different times was byte-identical
between the two trees while the *video* differed by 162. That is not a batching
artifact: the fast scene's solids only expose such a bug at certain animation
orientations, so a single frame can easily miss it. Later, the same trap
appeared in reverse for `MESH_ID` — single frames matched at t=1.0…2.0 and
differed at t=2.4.

**Always A/B the video.** If single frames match, conclude nothing.

7.3 The arena changes the slicing, and that looks like a semantic difference
----------------------------------------------------------------------------
Freeing arena bytes (7 fewer stub tensors, 2 fewer BVH builds) changed how the
sparse resolve slices covered pixels: the pre-deletion tree *attempted* a
655,532-pixel slice, hit `InsufficientMemoryException` and retried as two halves,
while the post-deletion tree fits it in one. Instrumenting
`rp.shade_sparse_raster_coverage` shows `[0,15009] [0,655532] [0,327766]
[327766,655532]` versus `[0,15009] [0,655532]`.

Good news, established by this branch: **the sparse resolve IS slice-invariant.**
After 7.1 was fixed, the one-slice and two-slice renders agree byte-for-byte.
So slicing differences are a red herring — but they will mislead you for an hour
if you do not know that, because the slice counts differ in a diff you are
trying to prove is a no-op.

7.4 How to bisect an ABI-coupled kernel refactor
-------------------------------------------------
Positional Taichi kernel arguments mean a half-reverted tree does not run, so
`git checkout` of one file is useless. What worked:

    git worktree add /tmp/algan-head HEAD     # A/B reference that stays runnable
    git diff > /tmp/phase.patch               # save everything first
    for f in <non-kernel files>; do git show "HEAD:$f" > "$f"; done
    # ... test ... then: git apply /tmp/phase.patch to restore

Reverting the *non-kernel* files while keeping the kernel+tracer changes is
ABI-consistent (the kernels simply stop reading data the merge still produces),
which splits the search space in half in one step. That is what localized 7.1.

Complementary: hash the renderer's inputs rather than guessing. Wrapping
`scene_builder._merge_scene` to print SHA-256 of every tensor, and
`KERNEL_REGISTRY.render_kernel` to print `(time_start, time_end)`, proved the
merged tensors, the derived flags and the batch windows were all identical — so
the difference had to be inside a kernel, not upstream of it.

7.5 Large mechanical deletions: assert the counts
--------------------------------------------------
Deleting ~1000 lines of positional parameters across four 3000-line kernel files
by hand is where silent argument-shift bugs come from. What worked: a script
that deletes/rewrites whole lines matched **exactly** (stripped), with an
**expected occurrence count per rule**, aborting without writing on any
mismatch. Two of my hand-counted expectations were wrong and the assertion
caught both before anything was written. For whole blocks, delete by structural
boundary (decorator → next top-level `def`), not by pasted text.

7.6 The fast suite is curated and enforced
-------------------------------------------
`tests/unit_tests/test_fast_suite_curation.py` fails if a new `fast` marker is
not documented in `tests/README.md`'s membership table, with a reason naming
which change *elsewhere* would break it. Add the row in the same commit.

7.7 CI runs two ruff gates
---------------------------
`ruff check` and — separately — `ruff format --check` (pinned `ruff@0.12.4`,
`.github/workflows/code_quality.yaml`). Running only the former will pass
locally and fail CI. `*_taichi.py` is excluded from both by `extend-exclude`,
which matters: the formatter's `from __future__ import annotations` breaks
Taichi kernel compilation.

7.8 What this GPU-less container can and cannot prove
------------------------------------------------------
Validates fully here: `tests/unit_tests`, `tests/fast` (including its
pixel comparison — it genuinely sees a renderer regression on the CPU path), the
`benchmarks/_aa_*` harnesses, and any A/B of gate-off byte-identity.

Does not: `tests/full_renders` (skips under `CI`; its baselines are per-machine,
not merely per-device, because `pn_criterion_kernel` runs under `fast_math`),
CUDA/CPU divergence, kernel timings, and register pressure. `ALGAN_UPDATE_*`
writes only `expected_outputs_cpu/` — a change that moves output is not complete
until the CUDA set is regenerated too.

7.9 Two instruments that disagree are ONE instrument
------------------------------------------------------
`_aa_run_gate_check` said §6.3.2's relaxed gate cut the `Cylinder`'s coverage
error 70%; `_aa_line_check` said its ink wobble did not move. Both were run
correctly. They were measuring **different geometry** — the coverage harness's
fat `Cylinder` against the line check's 0.045-radius rod — and the mechanisms
that dominate the two are different (far-sheet re-claim 79% on the rod, the
`full` gate 19%).

The fix was to put the line check's own scene into the coverage harness as a
case (`line-check cyl (33deg)`), so both instruments describe the same pixels.
That single change explained a contradiction that had survived two sessions, and
it is why §6.6 exists. **Reconcile the instruments before theorising about the
renderer.** The cost is one case; the alternative is chasing a mechanism that is
real but irrelevant to the metric you are being judged on.

A second, sharper version of the same trap: `_aa_run_gate_check` scored
**silhouette pixels only**, so it was structurally blind to interior notches.
§6.3.2's first build looked like an 84% win in that harness while putting 531
notches into a flat quad. The harness now counts interior notches beside the
win. An instrument that cannot see a regression will report one as a triumph.

7.10 A reference derived from the same formula proves nothing
--------------------------------------------------------------
§6.6's cap is `max(front_area, back_area)` over the exact clipped areas.
`_exact_coverage`'s truth is essentially that same formula over those same
numbers. So the `|cap-E|` column came back **0.0000 on every case** — which
looks like a triumph and is very nearly a tautology. It shows the walk *can*
land on the exact-area answer, not that the exact-area answer is right where the
question is hard (a grazing boundary, which is exactly where it is not).

The independent evidence for §6.6 is the ink-wobble table, which never consults
`_exact_coverage`. Before quoting a number from this harness, ask what would
have to be true for it to come out wrong — and if the answer is "nothing",
it is measuring itself.

7.11 A gate that "implies" another must be wired in exactly one place
----------------------------------------------------------------------
§6.6 said `ONE_MESH` implies §6.3.2's relaxed gate, and it was true on the kernel
side (`aa_grp = 3`, and `_aa_run_full` accepts 2 or 3) and false on the host side
(the emission truncation tested `ANALYTIC_AA_RUN_FULL` alone). One question, two
readers, in two languages, and the answers differed for two commits without any
test noticing — because the failure is silent: output is produced, looks
plausible, and carries interior notches the coverage harness is structurally
blind to (§7.9).

The rule that would have prevented it is the one §3.2 already applied to
`_tri_hit`: when N sites must agree, give them one function to ask. `aa_grp` is
now computed by `_aa_group` and interrogated only through `_aa_run_full` /
`_aa_one_mesh`, and an AST audit fails the build if anything else reads the raw
setting.

Generalize it: **an implication between feature flags is a fact about one
derived value, not a convention two call sites are trusted to remember.** If you
find yourself writing the implication twice, the second one is already wrong.

7.12 Record the environment with the measurement, or the number is not about
     the default you are about to ship
----------------------------------------------------------------------------
Every §6.6 CPU figure in this document is consistent with having been measured
with **both** `ALGAN_ANALYTIC_AA_ONE_MESH=1` and `ALGAN_ANALYTIC_AA_RUN_FULL=1`
exported, which is not the configuration a single default flip produces — and
because of §7.11 the difference was most of the win (a flat quad's ink wobble
-63% against -8%). The numbers were right about the *rule*; they were not right
about the *flip*.

Cheap habit that closes it: have the harness print the gate values it actually
ran under, and quote that line beside the table. `_aa_run_gate_check` prints its
`aa_tri/aa_grp modes` per case, which is exactly this and is what made the split
diagnosable after the fact.

7.13 When output moves, ask which device it moved TOWARD
---------------------------------------------------------
`manim_compat_and_plots` failed its CUDA baseline by 220 channel values with
every gate on this branch turned off, which looks like an unattributable
regression. It was not this branch: `35fe6ec` (from master) pinned an `argmax`
tie-break that torch does not specify for equal maxima, and verified bit-identity
on CPU.

The test that settled it needed no bisect and no worktree. The CPU baselines were
the *fresher* set, so compare both CUDA renders against them:

    my CUDA render (gates off) vs fresh CPU baseline   peak  52
    committed CUDA baseline    vs fresh CPU baseline   peak 218

The render moved **toward** the other device, which is the signature of a fix
removing device-dependent behaviour rather than of a regression. §3.5 states the
same reasoning in the other direction ("a correct CPU render moves toward the
CUDA baseline"); it is worth naming as a general instrument, because it costs two
video diffs and replaces an afternoon of bisecting.

7.14 `profile_scene` writes TWO runs, and the cold one comes first in the file
-----------------------------------------------------------------------------
`profile_scene` renders twice by design — RUN 1 cold (Taichi JIT, cold GPU
clocks), RUN 2 warm — and its own docstring says to use the warm numbers. Both are
written to the same report, cold first, so `grep -m1 'kernel: raster_first_shade'`
silently reads the **cold** row. The gap is not subtle: 17.270 s cold against
1.167 s warm for that kernel on one scene, because cold it is paying its own
compile.

This nearly put cold numbers into §4.3 and §3.6 as measurements. Cold rows also
make the profile look like a different renderer: cold puts `raster_first_shade` at
38% and `raster_shadow_trace` at 35%, warm puts `raster_shadow_trace` at **80%**
and `raster_first_shade` at 2.4%. Any conclusion about where render time goes
inverts depending on which table you read.

Parse the last `RUN n` section, not the first match in the file.

7.15 Interleaving that never varies the ORDER is not interleaving
------------------------------------------------------------------
The A/B for §3.2 ran off, on, off, on, off, on and took per-kernel minima, which
is the standard drift-robust recipe. It still produced a uniform +8-16% for the
`on` arm — including in kernels the flag cannot reach — because the `off` arm
occupied the cooler slot of *every* pair while the machine heated monotonically.
Minima do not remove a bias that is systematic within each pair.

Two habits fix it, and the second is worth more: balance the order
(off,on,on,off,...), and **always include a control** — a kernel or stage the
change provably cannot affect. The control is what turned an apparent 8.5%
regression into a measurement of the room temperature. Without one, that number
would have gone into this document.

7.16 A nondeterministic reduction is invisible until it feeds a threshold
-------------------------------------------------------------------------
`scatter_add_` on CUDA floats is not reproducible, which everyone knows and nobody
worries about, because a 1e-05 error in a colour is not a bug. §6.6 put one behind
a **comparison** — the cap clips when `eff > frag_cap - mesh_ink` — and the same
1e-05 became 28 channel values over 9.6% of a frame, because a threshold turns an
epsilon into a branch and bloom turns a branch into a region (§6.6.4).

The rule to carry forward: **classify every host-side float reduction by what
consumes it.** Feeding a colour, an atomic add is fine. Feeding a comparison, a
sort key, a count, or an index, it is a correctness bug waiting for the right
scene. The cheap defence is to accumulate in float64 and round to the consumer's
dtype, which costs one pass and removes the class.

And the cheap *detection* is an A/A render: run the identical configuration twice
and diff. It costs one render, it caught this, and no amount of comparing against a
baseline can distinguish "the baseline is stale" from "the renderer is not
reproducible" — which is exactly how this presented.

7.17 Two traps when checking a CPU baseline from a CUDA machine
----------------------------------------------------------------
Both cost a wasted run here, and both fail *silently* — they produce a green or red
result that looks like an answer.

**`CUDA_VISIBLE_DEVICES=` (empty) does not hide the GPU on Windows.**
`torch.cuda.is_available()` still returns True. Use `CUDA_VISIBLE_DEVICES=-1`,
which does. Verified both ways rather than assumed.

**The render suites pick their baseline directory from `torch.cuda.is_available()`,
not from the render device.** So a run that renders on CPU but still sees the GPU
compares against `expected_outputs_cuda/`. Combined with the first trap, a
"CPU baseline check" silently became a CUDA run compared against the CUDA baseline
— and it *passed*, which is the worst possible outcome. What caught it was hashing
the output: identical to the CUDA baseline, byte for byte.

**And check which settings the baseline you are comparing against was written
with.** The CUDA and CPU sets are not from the same point in history: the CUDA set
came from `efb3a95` (pre-branch, every gate off) while the CPU set came from
`2d1432a` (MESH_ID and winding already on). "Gates off" is therefore the right
reproduction check for one and the wrong one for the other. `git log -1 -- <baseline>`
before choosing the arm.

7.18 "Byte-identical" is a claim about the scenes you rendered
---------------------------------------------------------------
`_weld_check` reported the weld byte-identical on three scenes, textured and
normal-mapped included, and §3.1 was written up on that basis. Running the full
suites across the same gate then moved **two** scenes, by 31 and 54 channel values
over 7-10% of a frame.

Both instruments were right. The harness renders a *single static frame*; the
full-render scenes move a camera over adaptively diced PN surfaces, and the dice
level is chosen per patch per frame from projected size — so a changed triangle
list can land on a different level, which a static frame cannot expose. The split
in the results says so exactly: the two scenes made of circuits and flat meshes
moved zero pixels, and both scenes carrying `Surface` geometry moved.

The lesson is not "write a better harness". It is that **byte-identity has a
scope, and the scope is the scenes rendered** — so state it that way ("byte-
identical on a static frame including textures") rather than as a property of the
change. And for anything touching geometry, the confirming run is
`tests/full_renders`, because `--fast` deliberately contains no PN surface and
therefore cannot see tessellation move at all.
