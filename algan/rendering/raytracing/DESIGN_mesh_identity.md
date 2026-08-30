# Algan — Mesh Identity in the Triangle Renderer

**Status: PARTLY LANDED. This file is the RECORD — what was built, measured and
settled, and why every number is what it is.**
**One limitation ships knowingly: §0.5 — but read §6.8 first, which removes its
full-mask half and now ships on by default.**

**If you are here to do work rather than to look something up, start with
`DESIGN_mesh_identity_open.md`** — the queue of what is left, self-contained,
with each item's how-to and what would settle it. It cites this file by section
for the measurements rather than repeating them.

Plan of record for replacing the renderer's epsilon-based seam heuristics with
declared mesh identity. Written to be self-contained: a fresh session with only
this file and the repo should be able to continue without reconstructing any of
the reasoning.

Reading order. §0 is the state of the branch and what to do next, and **§0.1 is
how to run everything and what this machine may and may not measure** — read it
first if you are resuming from this file alone. **§0.5 is a
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

    pytest -q tests/unit_tests tests/fast      1051 passed, 89 skipped
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

**THE GATES, and what each is worth.** Ten switches now, all declared in
`algan/environment.py` and surfaced on `SETTINGS.raytracing.experimental`.

    setting                          default  what it buys
    ---------------------------------------------------------------------------
    ALGAN_MESH_ID                    ON       per-mesh tri_obj (§2.2, §3.5)
    ALGAN_POLYHEDRON_WINDING         ON       consistent face winding (§3.7)
    ALGAN_ANALYTIC_AA_ONE_MESH       ON       THE AA RESULT (§6.6), implies ↓
    ALGAN_ANALYTIC_AA_ONE_MESH_DENS  ON       the capped write's other half (§6.6.2)
    ALGAN_ANALYTIC_AA_RUN_CAP        ON       frag_cap on a truncated run (§6.8)
    ALGAN_ANALYTIC_AA_RUN_EXACT      off      exact run totals (§6.7, §6.7.2)
    ALGAN_ANALYTIC_AA_RUN_FULL       off      the relaxed run gate ALONE (§6.3.2)
    ALGAN_WELD_SURFACE_SEAMS         ON       shared seam/pole vertices (§3.1)
    ALGAN_WATERTIGHT_TRI             ON       Woop-Benthin-Wald (§3.2)
    ALGAN_BEZ_BVH_SPLIT              ON       median-split bezier BVH (§3.4)
    ALGAN_ANALYTIC_AA_RUN_RULE       redist.  pre-existing (v2 §4.4)

Seven of the ten are on, and the two still off are off for stated reasons rather
than for want of attention:

* `ALGAN_ANALYTIC_AA_RUN_FULL` is **subsumed**, not pending: `ONE_MESH` implies it
  (`aa_grp` 3 or higher, and `_aa_run_full` accepts anything from 2 up), so it only
  selects the relaxed gate *without* the cap — a configuration kept for the harness.
* `ALGAN_ANALYTIC_AA_RUN_EXACT` is confined, verified, ATTRIBUTED and still
  off: it makes every truncated run exact and leaves complete runs
  bit-identical (§6.7.2), and `shapes_and_timeline`'s once-unattributed move
  is measured at **18 lossless pixel-frames** — the engaged truncated runs it
  exists to fix — inflated ~2,000x by the H.264 decode the comparisons read
  (§6.7.3). What holds it off now is only the flip work: cost and baselines
  (`DESIGN_mesh_identity_open.md` §C.4).

And three that were off in the previous revision are now on:

* `ALGAN_WATERTIGHT_TRI` is correctness-qualified and remains **cost-unqualified**
  — its cost cannot be measured on a thermally throttled machine, and because the
  flag is read at import an in-process alternating A/B is impossible. What is new
  is that `_watertight_check.py` run on BOTH arms gives identical quality (0 cracks
  each, identical ridge counts), so the dilated arm has nothing left to offer as a
  default and its remaining value is as an A/B control.
* `ALGAN_BEZ_BVH_SPLIT`: the inherited "~20-25% fewer traversal steps" is now
  measured at **30%** (3.300 → 2.302 sibling-block tests per ray), byte-identical.
  But read §3.4's note first: `BVH_REFIT` defaults ON and its build ignores
  `builder`, so at shipped defaults **no STBVH is built at all** and this flag —
  like `ALGAN_BVH_BUILD` — changes nothing. That is why `_bez_bvh_ab.py` measured
  nothing: it was A/B-ing one render against itself.
* `ALGAN_WELD_SURFACE_SEAMS`: both stated risks are closed (byte-identical on a
  static frame with textures and normal maps; the morph path now asks
  `surface_weld_flags` for the same grid the render path does), and the moving-PN
  baselines it needed are regenerated on CUDA and reviewed. §3.1.

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
    §3.2  watertight tri intersection  FLIPPED ON — cost is under the noise floor
                                       here, and the dilation was a real defect
    §3.3  delete the epsilons          NO LIVE EPSILON READ LEFT at shipped
                                       defaults; what remains are two ti.static
                                       arms that are also the A/B levers (§3.3)
    §3.4  median-split bezier BVH      FLIPPED ON. 30% fewer traversal steps,
                                       byte-identical -- but INERT at defaults
                                       (BVH_REFIT builds no STBVH), see §3.4
    §3.5  mesh identity                FLIPPED ON, both devices re-baselined
    §3.6  two-level BVH                NOT STARTED, and the perf case is MEASURED
                                       not to justify starting (§3.6)
    §3.7  Polyhedron winding           FLIPPED ON, same re-baseline
    §6.3.2 relaxed AA run gate         SUBSUMED by §6.6; the switch alone stays off.
                                       It OWNS the residual interior notches (~92%),
                                       which ship KNOWINGLY UNFIXED — see §0.5,
                                       now MEASURED on the six real scenes
    §6.6  one-mesh coverage cap        FLIPPED ON — the AA result, plus a gate bug
                                       found and fixed while qualifying it
    §6.6.2 capped occlusion write      FLIPPED ON — closes the claim-vs-occlusion
                                       desync; the CLAIM-side shortfall stays open
    §6.7  exact run totals (no scan)   BUILT, OFF — host half verified on 49.6M
                                       run starts; now CONFINED to truncated runs
                                       (§6.7.2), its dense-path OOB read fixed
                                       (§6.7.1), its precision question decided;
                                       one unattributed scene move blocks it
    §6.8  frag_cap on a truncated run  FLIPPED ON — exact on the full-mask arm
                                       wherever a cap exists, no new lane (§6.8)

**WHAT IS LEFT, in priority order.** Every item in the previous revision of this
list has been run; these are what running them produced.

0. **The interior-notch limitation is now MEASURED on the six real scenes, and
   the decision needs restating rather than defending — see §0.5.** The previous
   revision of this entry said the six `tests/full_renders` scenes had never been
   counted and that single-digit counts would confirm the standing decision.
   They are counted now (`benchmarks/_notch_scene_check.py`, CUDA, every frame,
   each render byte-identical to its committed baseline first) and they are not
   single-digit: **`text_and_media` notches ~900 interior pixels on each of the
   112 of its 182 frames that carry any**, `solids_and_camera` an order of magnitude fewer, and the
   other four are clean. The per-pixel size is unchanged and small — ~1.7 channel
   values typically, ~2.5 at the worst pixel, cross-calibrated at 8% of the
   claim-side shortfall against `--notch-probe` on two harness cases that agree
   to 0.1%. So it is **not rare, still small**: a low-amplitude coverage error
   over most of an imported mesh's interior, sitting just under the suites'
   tolerance, rather than the handful of pathological pixels §0.5 assumed.

   Two consequences, both in §0.5. The measurement moves §6.3.2's open choice —
   fix (b) was blocked on the silhouette population it would give up, and on the
   real scene that population is **5%** of the pixels it would fix, against more
   than 100% on the harness case that blocked it. And **fix (a) has now been
   A/B'd** (`_AA_MAX_RUN_SCAN` 16 → 128, cold-compiled): it moves four of the six
   scenes by 31-50 channel values over up to 10% of a frame, so it is a
   both-devices re-baseline rather than a patch — **and most of that movement is
   not the notch**, because the limit bounds the run's EXTENT as well as its area
   sum. `shapes_and_timeline` has zero notched pixels and still moves 31.

   **UPDATE — the full-mask half is now fixed and shipped (§6.8), and
   `shapes_and_timeline`'s 31 is EXPLAINED (§6.7.3).** §6.8 takes the mesh's
   own coverage ceiling instead of a truncated sum on the full-mask arm, which
   is exact wherever a cap exists and costs no lane. What is left of §0.5 is the
   PARTIAL-mask arm, which is the larger half in both population and magnitude
   (mean 0.2762 over 314,072 truncated pixels in `text_and_media`) and needs the
   sample union fixed, not just the area — only §6.7 does that. The "197
   pixel-frames of truncated runs against ~37,000 pixel-frames that move" that
   held it off is resolved: the renderer's change is 18 lossless pixel-frames
   inside the truncated population, and the other ~37,000 are the H.264 decode
   spreading them (§6.7.3). Both counts were right; they measure different
   instruments.

   **A third fix is now built and gated off: §6.7** takes the run's totals from
   a host segment reduction and deletes the kernel's scan entirely, which is the
   only one of the three that fixes the sample UNION as well as the area. Its
   host half is verified exactly (49.6M run starts, zero mismatches) and it
   reproduces the raised-limit render byte-for-byte on the scene carrying the
   largest truncated population. It is not shippable yet — see §6.7's four
   remaining items, starting with looking at the frames.

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
2. **§3.2 is DONE and ON by default.** The cost question was retired rather than
   answered: the control kernel the flag cannot reach moved as much as the ones it
   can (+8.6% against +8.5–10.7%), which is thermal drift by §7.15's own
   criterion, so the cost sits under this machine's noise floor. Weighed against a
   real defect — the dilation tests every triangle wider than it is — the flag was
   flipped. Both CUDA baseline sets regenerated; the fast scene and three of six
   full renders moved, all through SECONDARY rays.
3. **§3.3 is now a fallback-retirement decision, not an epsilon problem.**
   `_raycast_pixel` asks `_tri_hit`, and with `WATERTIGHT_TRI` on there is **no
   live `BARYCENTRIC_EPSILON` read left at shipped defaults** — all three
   survivors sit in `ti.static` arms that do not compile in (§3.3 has the table).
   What remains is a call about two `ti.static` branches — and note before you
   make it that **both are A/B levers, not dead weight**: `aa == 0` is the control
   arm for every analytic-AA measurement in §6, and `_tri_hit`'s dilated arm is
   the control for §3.2, whose cost is still unmeasured on non-throttling
   hardware. Neither is a per-pixel fallback and neither is needed for
   correctness (§3.3 says why, including for the overlapping-transparency case
   that looks like it would need one). Deleting them buys the constants, the
   `edge_hit` bit and `seam_t`; it costs the ability to measure. Do not promise
   the per-ray `f32` until both have gone, and re-check `memory_model` when
   `rs_sca` shrinks.
4. **§3.1 is DONE and on by default.** Both blockers were already gone (the
   stated pixel risk closed by measurement, byte-identical on a static frame with
   textures and normal maps; the topological one fixed by routing the morph path
   through `surface_weld_flags`), and the baselines it still needed have been
   regenerated on CUDA and reviewed: `materials_and_lighting` moves 19 channel
   values, concentrated in the bloom halo around its two glowing spheres — the
   amplified-epsilon pattern, visually identical — and `solids_and_camera` moves
   33, as interior speckle on the diced solids. `tests/fast` does not move at
   all. The CPU set cannot be regenerated on the machine that owns the CUDA one
   (§3.5), so it is now additionally stale; the open queue's §B carries that debt.
5. **§4.6 is answered as far as an outside instrument can answer it.** Both
   purpose-built scenes now have live shadow paths and reach as far as pixels can
   show, and all three `SHADOW_ANYHIT` modes are byte-identical on both — so the
   documented disagreements do not reproduce from the public API. Going further
   needs in-kernel instrumentation (item 1's territory): from outside, nothing
   can show that a shadow ray peeled 256 surfaces and stopped, or that its two
   hits landed inside the merge band. Read §4.6's list of the four ways this
   check produced non-evidence before extending it.
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
0.1 STARTING A SESSION FROM THIS FILE ALONE
================================================================================

Everything needed to resume is here. `CLAUDE.md` is the repo-wide contract and
still governs; this section is what is specific to *this* work.

**Run everything through the venv.** `<venv-python>` below is
`.venv\Scripts\python.exe` on Windows, `.venv/bin/python` elsewhere, or just
`uv run python`. The system Python has no taichi.

    <venv-python> -m pytest -q tests/unit_tests tests/fast   # ~5-10 min, CI's paths
    <venv-python> -m pytest -q tests/full_renders            # ~6-9 min, 6 dense scenes
    <venv-python> -m ruff check --no-fix algan tests         # NEVER without --no-fix
    <venv-python> -m ruff format --check algan tests

Expected green at the tip of this work: **1056 passed, 89 skipped** and
**7 passed**. If either is red before you change anything, find out why before
building on it.

**THE MACHINE THIS WAS MEASURED ON, and what that forbids.** A Windows box with a
GTX 1050 (4 GB), driver 576.52, Taichi 1.7.4, torch 2.7.1+cu128.

* **It owns the CUDA baselines and must never write the CPU ones.** Its CPU
  render of `tests/fast` misses the committed CPU baseline by 30 channel values
  on 43 of 45 frames *before any change in this work*, so regenerating from here
  would replace a baseline CI reproduces with one it does not. §3.5 and §7.17.
* **It throttles, so wall-clock A/B is not available.** A control kernel the
  change cannot touch drifts as much as the target; §7.15 and §3.2 both record a
  measurement destroyed this way. Prefer counts, byte-diffs and in-process
  alternating A/B with an explicit control. If a number straddles 1.0 across two
  orderings, it is room temperature.
* **Cold Taichi compiles run 35-45 minutes** after `clear_cached_kernels(
  True)`, and a new `ti.static` template VALUE (a new `aa_grp`, flipping
  `WATERTIGHT_TRI`) is a new variant with its own cold compile. Budget for it.
  Clear the cache before any kernel A/B: the offline cache does **not** invalidate
  on `@ti.func` edits.
* **One render process at a time.** Killed background renders orphan children
  that keep output mp4s locked.
* **4 GB is a real constraint.** 304 translucent screen-filling cubes exhaust it
  on a single LD frame; shrink geometry on screen rather than reducing the count
  when a scene needs depth.

**THE HARNESSES, and the question each answers.** All take `--res md` or a
quality name; all live in `benchmarks/`.

    _aa_run_gate_check.py    coverage error, ink wobble, notches, --verify replay,
                             --notch-probe (the ss6.3.2/ss6.6 instrument of record)
    _notch_scene_check.py    ss0.5's mechanism in the SIX FULL-RENDER SCENES, which
                             --notch-probe cannot reach; diffs its own render
                             against the committed baseline before reporting
    _aa_line_check.py        ink wobble along an edge over nine angles
    _one_mesh_ab.py          cost of the one-mesh cap (ss6.6.3)
    _one_mesh_dens_ab.py     cost + look of the capped occlusion write (ss6.6.2)
    _watertight_check.py     cracks, double blend, pixel diff across ss3.2
    _weld_check.py           ss3.1's weld, textured and normal-mapped arms
    _bez_bvh_ab.py           ss3.4's bezier split ordering against a noise floor
    _shadow_anyhit_check.py  ss4.6's three SHADOW_ANYHIT modes, with reach checks
    _diff_frame.py           side-by-side worst frame of two videos (LOOK at this)
    _video_diff.py           how far two videos moved, and over how many pixels

**SIX RULES THIS WORK KEEPS LEARNING THE HARD WAY.** Every one of them cost a
wrong result recorded in this file:

1. **A check must show it REACHES its case.** §4.6 produced "all three modes
   agree" four times from scenes that had no shadow in them at all. Prove the
   mechanism is live (render with the feature off and diff) before reading any
   agreement.
2. **Replay the same inputs with exactly ONE thing changed.** That is what
   attributed the notches (§6.6.2) three times running; anything that changes the
   emission cannot isolate the resolve.
3. **Read which accumulator a metric scores.** §6.6.2 predicted an occlusion-side
   fix would close claim-side symptoms. It could not have, and the prediction was
   published before anyone checked.
4. **A gate that "implies" another must be wired in one place.** §6.6.1: the
   kernel and the host disagreed about the same question and it cost most of a
   win.
5. **Look at the frames before re-baselining.** `_diff_frame.py` exists for this.
   "Measured 42 channel values" is not the same as knowing what moved.
6. **A harness that renders the full-render scenes must diff its own render
   against the committed baseline.** Reproducing the suite's settings is not
   enough — the vendored fonts are registered by `tests/conftest.py`, and a
   script that skips it measures a scene 205-232 channel values away from the
   one it names. §7.19.

**WHERE TO START.** §0's priority list, top entry first. §0.5 is a shipped
limitation you should read before treating a diced mesh's interior pixel as a
bug. §6 is measured negative results — read it before building anything in the
AA area, it will save a day.


================================================================================
0.5 KNOWN LIMITATION, SHIPPED AND DELIBERATELY NOT FIXED
================================================================================

**A diced mesh can lose up to ~5% of one interior pixel's coverage, and the
default renderer ships that way.** This was diagnosed to the line, costed, and
then left alone as a considered decision rather than an oversight — the fix is
not worth its price at the sizes measured. Read this before "fixing" it.

**The six real scenes have since been counted** (the last part of this section),
and the size holds while "rare" does not: one of them carries it on most of an
imported mesh's interior in most of its frames.

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
  deliberately. **Now A/B'd — see the end of this section: it moves four of the
  six scenes by 31-50 channel values, and most of that is not the notch.**
* **Refuse to consult `E` when the scan hit its limit**, falling back to the
  shipped `corr = 1` short-circuit. Cheap and principled — a truncated sum is not
  an area — but it withdraws the gate's win from every long-run SILHOUETTE pixel,
  and on the rod those are most of the frame (`capped` is 3011 of 3546 clean
  interior pixels).

Either needs a kernel recompile and a cost number, and cost is exactly what the
machine this was measured on cannot resolve (§7.15). Against a worst case of ~13
channel values on deliberately pathological geometry, that is not a good trade.
(The pixel half of that recompile has since been paid for (a); the perf half has
not, for either.)

**NOW MEASURED IN THE SIX REAL SCENES, AND THE PREMISE OF THE DECISION ABOVE IS
HALF WRONG.** The paragraph this replaces said that if real scenes showed
single-digit pixel counts the standing decision was confirmed. They do not, and
by four orders of magnitude in one scene. What survives is the *per-pixel*
figure; what does not is "rare".

`benchmarks/_notch_scene_check.py`, CUDA, `PREVIEW`, **every frame of every
scene**, each render verified byte-identical to its own committed CUDA baseline
before a single number was read (that check is in the harness because the first
run of it was not: it had not registered the vendored fonts and came back
205-232 channel values from the baselines, which is a different scene, not a
drift):

    scene                     covered px   truncated   NOTCHED   interior   mean   worst
    complex_hierarchy_become   2,342,971          49         3          2  0.0131  0.0229
    manim_compat_and_plots     3,270,906           0         0          0       -       -
    materials_and_lighting    10,477,207           0         0          0       -       -
    shapes_and_timeline       11,356,119         107         0          0       -       -
    solids_and_camera          7,366,872       7,008     2,180      1,363  0.0302  0.1232
    text_and_media             5,922,322     420,552   106,283    100,618  0.0849  0.1249

`mean`/`worst` are over the INTERIOR pixels, in coverage. Frames carrying at
least one: 3 of 75, 0, 0, 0, **70 of 239**, **112 of 182**. Longest same-sheet
run: 56, 0, 16, 20, 76, 66.

**Three scenes are clean and one of them says why.** `materials_and_lighting`
scans 2.5M pixels and its longest run is **exactly 16** — it comes up to the cap
across a whole dense shadowed scene and never past it. `shapes_and_timeline` and
`manim_compat_and_plots` are circuits, which never enter this path at all.

**The two that are not clean are the two with dense triangle meshes drawn
small.** `text_and_media` carries an imported glTF model — condition 3 exactly,
a finely tessellated object at a fraction of the screen — and it notches ~900
interior pixels on each of the 112 of its 182 frames that carry any. `solids_and_camera` is the
predicted limb case, an order of magnitude smaller.

**WHAT THE PROBE SCORES, and why it is not the same number as the table above
it.** It replays the run scan with exactly ONE input changed — the scan limit —
and reports `corr(limit=inf) - corr(limit=16)`: the coverage the sheet's first
run loses purely because the scan stopped. That needs no material model and no
exact reference, which is what lets it run on a real scene at all;
`--notch-probe` cannot, because its reference drops any pixel holding two
objects and its walk assumes matte opaque geometry (the harness's docstring says
so at length). The price is that it measures the CLAIM, not the paint: what the
pixel finally shows depends on how much of the shortfall the far sheet refills
under the cap.

**Both instruments on the same cases put that factor at 8%.** Run on the four
harness cases §6.3.2 attributes, the new probe reproduces the old one's
attribution exactly — nonzero on the two the unbounded-scan replay recovered,
**zero** on the two it did not:

    case                 --notch-probe (painted)   _notch_scene_check (claim)   ratio
    line-check cylfine     253 px, mean 0.0090       402 px, mean 0.1113         8.1%
    sphere (192x96)         24 px, mean 0.0018        29 px, mean 0.0226         8.0%
    line-check cyl           4 px  (0 recovered)       0 px                         -
    packed 4x4 (overlap)     3 px  (0 recovered)       0 px                         -

Two independent points landing on 8.0% and 8.1% is a usable conversion, and it
says the far-sheet refill takes back ~92% of the first run's shortfall. Applied
to the real scenes: `text_and_media` ~0.007 of painted coverage typically and
~0.010 at its worst pixel, i.e. **~1.7 channel values typically and ~2.5 at the
worst pixel** against a maximally contrasting background. `solids_and_camera` is
~0.6 and ~2.5.

**One reason the 8% could be optimistic, stated because the A/B below is what
settles it.** Both calibration points are MATTE. The refill that takes back 92%
is the far sheet claiming the residue under the cap, and on a lit PBR material
the walk can break before it gets there: `raster_first_shade` breaks out of the
fragment loop when `refl_max >= cover_pass`, and on an interior pixel
`cover_pass = 1 - alpha` is near zero, so any reflectivity at grazing incidence
can end the walk with the shortfall still standing. `text_and_media`'s glTF model
is exactly that material. So the conversion is a floor for a matte mesh and not
necessarily one for a reflective one.

**So the honest restatement is: not rare, still small.** §0.5's per-pixel
estimate ("~2 channel values typically") is confirmed on real scenes and was
never the doubtful part. What was wrong is the picture of a handful of
pathological pixels: on a scene with an imported mesh this is a low-amplitude
coverage error spread over most of the mesh's interior in most frames, sitting
just under the suites' tolerance of 2 — which is why no suite catches it and why
looking at an amplified diff did not find it either.

**AND IT MOVES §6.3.2's OPEN CHOICE.** Fix (b) — refuse to consult `E` when the
scan hit its limit — was left unchosen because it withdraws the gate's win from
long-run SILHOUETTE pixels, and on `cylfine` those are most of the frame. §6.3.2
says that population had to be measured before (b) could be chosen. Measured, as
the interior share of the truncated full-mask pixels:

    cylfine (the harness case that blocked it)   402 interior of  849   47%
    solids_and_camera                          1,363 interior of 2,180  63%
    text_and_media                           100,618 interior of 106,283 **95%**

On the scene that actually carries this, the silhouette population (b) would
give up is **5% of the pixels it would fix**. The harness case was not
representative of it.

**THE A/B IS RUN, AND IT SAYS FIX (a) IS NOT A SMALL CHANGE.**
`_AA_MAX_RUN_SCAN` raised 16 → 128, Taichi kernel cache wiped and cold-rebuilt,
the six scenes re-rendered and diffed against the committed CUDA baselines. The
arm reached its case, which is the first thing to check: `truncated` goes
420,552 / 7,008 / 107 / 49 → **0 on every scene** (the longest run anywhere is
76).

    scene                     max|d|   worst frame        frames    mean px/frame
    manim_compat_and_plots         0   byte-identical    0 of 171               0
    materials_and_lighting         0   byte-identical    0 of 179               0
    shapes_and_timeline           31    4,514 (1.6%)    12 of 301             119
    complex_hierarchy_become      49    8,119 (2.9%)    61 of  75           4,826
    text_and_media                49   28,172 (10.1%)  158 of 182           6,046
    solids_and_camera             50   20,276 (7.3%)   224 of 239          12,113

The two that do not move are exactly the two the probe said could not: pure
circuits, and the scene whose longest run is exactly 16.

*(Caveat added later: these moved-pixel counts are decoded-H.264 measurements
and overstate the renderer's own change — §6.7.3 measured the equivalent
`shapes_and_timeline` population at 18 lossless pixel-frames against the
~37,000 the decode reports. WHICH scenes move, and the byte-identical rows,
are unaffected.)*

**But this is NOT the notch's price, and reading it as one would be the fourth
mistake of this kind in this document.** `shapes_and_timeline` has **zero**
notched pixels and still moves 31 channel values over 4,514 of them. The limit
bounds the run's EXTENT as well as its area sum: `run_end = rj` comes back from
the same scan, so fragments past the 16th of a run sit outside the corrected run
— and their own rescan cannot engage, because by then `svis` is no longer
uniform and `uni_v` fails. They are painted uncorrected. Raising the limit
brings them under the run's correction, which is a second, larger change riding
on the same constant.

So what the A/B settles is the shape of fix (a), not the size of the defect:

* **(a) is a re-baseline, not a patch.** Four of six scenes move by 31-50
  channel values over up to 10% of a frame — both device baseline sets, on a
  CUDA machine, with the CPU set already in debt (§3.5).
* **Most of that movement is not the notch.** The notch population is 106,283
  pixels across `text_and_media`'s whole 182 frames; the diff touches 6,046
  pixels per frame on average there. The rest is the extent effect.
* **Nobody has shown the new frames are better.** More fragments receiving the
  exact-area correction *should* be more accurate, and "should be" is precisely
  what §6.3.2 and §6.6.2 each got wrong once. Deciding (a) needs
  `_aa_line_check` and `_aa_run_gate_check` re-run in the raised arm, which
  costs a second cold compile and was not done here.
* **It does not validate the 8%.** The two arms differ by more than the notch,
  so the claim-to-paint conversion above is still a conversion.

**If you take this further, isolate the two effects first.** A variant that
keeps `run_end` at 16 while letting `E` come from an unbounded sum measures the
notch alone; the difference between that and this arm is the extent effect. Both
need their own cold compile. And fix (b) — refuse to consult `E` on a truncated
scan — is a third arm again, and the one whose case the interior/silhouette
split above has just improved.

*Method note, because it cost the shipped cache once already:*
`clear_cached_kernels()` deletes the **whole** cache directory, Manim
Tex geometry included, and the first render after that differs from every later
one (§4.10) — which would land in the diff as if it were the change. Back up
`~/.algan/cache/taichi`, remove only that, and restore it afterwards; the
restore was verified here by re-rendering `solids_and_camera` and getting the
committed baseline back byte for byte.


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
    checker (color texture)         4096      3968        0      0
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

3.2 Watertight ray/triangle intersection  [LANDED, DEFAULT ON]
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

The *texture* is what has to be continuous across it, and that is a separate
mechanism with a separate predicate. `surface_closed_axes` (`surface.py`) runs
the same coincidence test on both axes, ungated — a closed surface's map wraps
whether or not its seam vertices are shared — and `wrap_pad_texture` repeats
column 0 at column `W`, so the sampler's `u * (W - 1)` clamp addresses a padded
`W + 1` columns and interpolates the wrap cell exactly as it does an interior
one. Both are needed: the uv column carries `u = 1` to the seam, the pad gives
`u = 1` the same texels as `u = 0`.

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

**FLIPPED ON.** Correctness was qualified (above) and the cost turned out not to
be a cost. The measured deltas were +8.5% to +10.7% on the kernels this flag can
reach — but `raster_tri_count`, which it CANNOT reach (the rasterizer has no
`_tri_hit`), moved **+8.6% in the same runs**. A control that moves with the
target is §7.15's definition of thermal drift rather than a cost, so the honest
reading is that the flag sits under this machine's noise floor, and no amount of
re-running it here will say otherwise. Against an unmeasurable cost stands a real
if small defect: the dilation tests every triangle slightly WIDER than it is, so a
ray that should miss can hit. That is the trade taken.

**Rendered output moved, and both CUDA baseline sets were regenerated.** The
movement is exactly where the mechanism predicts and nowhere else:

    suite / scene            max|d|   worst-frame px>2      note
    tests/fast                  35     1931 (0.46%)         metallic icosahedron
    materials_and_lighting      42     5641 (2.02%)         shadow boundaries
    solids_and_camera           47    18383 (6.59%)         diced PN surfaces
    text_and_media              49     6153 (2.21%)
    complex_hierarchy_become     0            0             unchanged
    manim_compat_and_plots       0            0             unchanged (circuits)
    shapes_and_timeline          0            0             unchanged

Reviewed frame by frame before regenerating, not merely measured: the panels are
visually indistinguishable, and the amplified difference lands on shadow
boundaries and sphere silhouettes in `materials_and_lighting`, and on a fine
dusting across the PN solids' interiors in `solids_and_camera` — which is what a
finely diced surface looks like when edge ownership changes, because nearly every
pixel there is close to some triangle edge. In `tests/fast` the split is cleanest:
the title and the 2-D circuit row are **exactly zero** and every moved pixel is on
the 3-D mesh row, 27 of them above 10. Circuits never call `_tri_hit`.

Note what does the reaching: the raster front-end owns primary visibility, so this
flag arrives through SECONDARY rays — the metallic icosahedron's reflection bounce
in the fast scene, shadow rays in `materials_and_lighting`. A scene with neither
does not move at all, which is three of the six full renders.

**The CPU baseline sets are now stale for this as well as for §6.6**, and still
cannot be regenerated here (§3.5).

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

**THE TABLE ABOVE WAS WRONG ABOUT BOTH RASTER ROWS.** Read the code before
planning from it:

* **`_ss_pixel` (`raster_taichi.py:1124`) is DEAD at shipped defaults.** The
  `accept` it builds from `BARYCENTRIC_EPSILON` is *immediately overwritten*
  inside `if ti.static(aa)` by a conservative half-pixel-diagonal reject, and
  never read in between. Coverage then comes from the exact int64 edge functions
  and the top-left fill rule, whose own comment says the masks "partition the
  pixel with no epsilon anywhere". It is live only when `aa == 0`.
* **`_raycast_pixel` was not "candidate acceptance".** It is the fallback for
  triangles that STRADDLE the camera plane, where screen-space projection is
  invalid and the exact fill rule is therefore unavailable. It casts one ray per
  sub-pixel sample — it *is* the ray path's test, reused because projection is
  unavailable, inheriting the epsilon by construction rather than to match the
  ray path's output. Its argument FOR the dilation was real: with a float test
  and no exact tie-break, a sample on a shared edge must be erred one way, and
  double-claiming is harmless (per-sample transmittance gives it to the nearer
  fragment) where dropping it is a crack.

**DONE: `_raycast_pixel` now asks `_tri_hit`.** It was inlining what `_tri_hit`
does, so routing its per-sample membership test through the shared function makes
the straddler path inherit whichever intersection the ray path ships — watertight
included, which means no dilation, no duplicate and no crack rather than the
"err toward double-claiming" trade. With `WATERTIGHT_TRI` off it is the same
arithmetic in the same order as the code it replaced.

**AND THAT LEAVES NO LIVE `BARYCENTRIC_EPSILON` READ AT SHIPPED DEFAULTS.** All
three survivors sit in arms that do not compile in:

    site                              live when              status
    _tri_hit's Moller-Trumbore arm    WATERTIGHT_TRI off     off by default now
    _ss_pixel (raster_taichi:1124)    aa == 0                overwritten under aa
    _raycast_pixel (:1593)            aa == 0                read only in the else

So §3.3's remaining deletion is not an epsilon question any more — it is a
question about two `ti.static` branches, and **"retire them" is the wrong framing
for at least one of them.** Get the following right before planning that work,
because the obvious objection to it is wrong and the real one is different.

*`aa == 0` is NOT a per-pixel fallback for pixels analytic coverage cannot
resolve.* It is a whole-batch compile-time value: `raster_pipeline` sets
`aa_tri` from `rt_settings.analytic_aa_tri_active()` and the `tri_screen` column
count, once, and every kernel in the batch is specialised on it. No pixel ever
falls back to it at runtime.

The runtime per-candidate fallback is a *different* mechanism and it is not going
anywhere: `_ss_setup` returns `use_ss`, choosing `_ss_pixel` (screen-space, exact
fixed-point fill rule) or `_raycast_pixel` (one ray per sub-pixel sample) for
triangles that straddle the camera plane. **Both supply analytic coverage**, and
`_raycast_pixel`'s docstring is explicit that under analytic coverage it does not
report full coverage but answers the membership question directly.

*And the hard case does not need `aa == 0` either.* Several partially transparent
fragments from different meshes overlapping one pixel is exactly what the
per-sample transmittance walk is for: each fragment attenuates only the samples
its own mask owns, and `svis` carries the result front to back. Nothing in that
requires the fragments to share a surface — the ONE-MESH rule (§6.6) is a special
case layered on top precisely because the general case is already general, and it
restricts itself to opaque single-surface pixels for that reason. What degrades
on such a pixel is the run rule's *area* correction, not correctness: interleaved
meshes give runs of length one, so `run_mode` stays 0 and the pixel is resolved by
its exact sample masks alone — sample-quantized, but partitioned exactly and
composited in order.

*The real argument for keeping `aa == 0` is that it is the A/B lever.* Half the
measurements in §6 exist because analytic coverage can be switched off and the
same scene rendered both ways. Delete the branch and you delete the control arm
of every future analytic-AA experiment, on a machine where controls are already
the scarce thing (§7.15). That is a much better reason to keep it than
correctness, and it points at a different decision: keep the toggle, delete only
what is genuinely unreachable under it.

*The non-watertight arm of `_tri_hit` is the same shape of question* and has the
same answer: it is `WATERTIGHT_TRI`'s off arm, so deleting it deletes the ability
to A/B §3.2 ever again. §3.2's cost is still unmeasured on hardware that does not
throttle; deleting the control before anyone has measured it would close that
door permanently.

**What the refactor did NOT do, said plainly: it moved zero pixels.**
`tests/unit_tests tests/fast` (1051 passed, 89 skipped) and `tests/full_renders`
(7 passed) are unchanged with it in. That is consistent with the fix being real
but rare — it takes a sub-pixel sample landing on an edge shared by two triangles
that BOTH straddle the camera plane — but consistency is not evidence. Nothing
here demonstrates the double-claim occurring in an existing scene, so the
end-to-end case is reasoned, not observed. The underlying property (exactly one
hit per shared edge) is tested separately in `test_watertight_triangle.py`; the
new test there pins the WIRING, since an inlined second copy of an intersection
test is precisely what drifts.

The raster path's `TRIANGLE_EDGE_EPSILON`/`seam_t` pair is easier: it compiles in
only under `ti.static(not aa_tri)`, i.e. the non-analytic-AA fallback, so it
disappears with that fallback rather than with §3.2.

Net: §3.3 is **two** deletions with different owners, and `rs_sca` only shrinks
by its f32 when both have landed. Plan it that way, or the "one f32 per ray"
saving will be claimed and not delivered.

3.4 Median-split STBVH for bezier circuits  [BUILT, MEASURED, default ON — and
    inert at shipped defaults]
--------------------------------------------------------------------
Once resolution is order-independent, `stbvh.py:302`'s reason for pinning
bezier to Morton is gone (PN, the other pinned type, no longer exists).

**LANDED**, gated `ALGAN_BEZ_BVH_SPLIT`, flipping the bezier default to
`"split"` (both the main tree and the opaque one, or the two disagree about
instance order).

**The inherited claim is now measured, and it holds.** `benchmarks/_bvh_steps.py`
(see the open queue's §E) counts sibling-block tests per ray on 35 circuits plus
`Text` and `Tex`, with `ALGAN_BVH_REFIT=0`:

    ordering        groups/ray (primary)   groups/ray (incoherent)
    morton                 3.300                  3.159
    median split           2.302                  2.219

30%, slightly better than the inherited 20-25%, and the same on incoherent rays
so it is not a coherence artifact. Leaf slots and primitive tests are identical
to four decimals, and every one of 51,200 rays returns the same primitive — the
reorder changes cost, not answers. The feared epsilon-level output cost does not
appear either: `_order_window_check.py` renders it byte-identical against a
noise floor of zero.

**READ THIS BEFORE CONCLUDING ANYTHING FROM THE FLAG.** `BVH_REFIT` defaults ON,
and `_build_accel`'s refit branch ignores `builder` outright — its own docstring
says so — so at shipped defaults **no STBVH is built for any geometry type** and
this flag, like `ALGAN_BVH_BUILD`, changes nothing at all. It governs the tree
you get with `ALGAN_BVH_REFIT=0`, and that is the only configuration in which
either the win above or any A/B of it exists. `benchmarks/_bez_bvh_ab.py` did
not know that, which is exactly why it reported byte-identity at wall 0.993x: it
was comparing one render with itself. The default is flipped anyway — strictly
better on the tree it governs, byte-identical, free today.

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

Clear the Taichi cache (`clear_cached_kernels()`) before any A/B — it
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
    `_logical_pn_tri_obj` priority over `_obj_ids`, and `_dice_logical_pn`
    built its patch→surface map from the per-member `_obj_counts` alone. For a
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

    **BUILT: `benchmarks/_shadow_anyhit_check.py`. Both cases now have a live
    shadow path, both scenes reach as far as this instrument can show, and all
    three modes are byte-identical anyway.**

        scene   shadows on vs off   reach check              modes
        tie     max|d| 30 (LIVE)    separation matters (43)  all 3 IDENTICAL
        stack   max|d| 24 (LIVE)    peel limit reached (141) all 3 IDENTICAL

    So with the peel limit genuinely exceeded (304 translucent sheets against a
    blocker past them) and with an opaque and a translucent slab a tenth of
    `DEPTH_TIE_EPSILON` apart, the march, the any-hit early-out and the
    gather-march still produce the identical sha256. The disagreements the kernel
    docstring documents do not reproduce from the public API.

    **THE FIRST THREE VERSIONS OF THIS RESULT WERE WRONG, and the way they were
    wrong is the point.** Each reported "all three modes agree", and none of the
    scenes had any shadow in it:

    1. *Shadow off-frame.* Both scenes lit from directly overhead, so the shadow
       fell on ground the camera sees edge-on at the horizon. Caught by looking
       at a frame.
    2. *A reach check that measured the wrong thing.* Rendering 304 sheets versus
       8 does change the image -- because the sheets are VISIBLE -- so "the peel
       limit is reached" was read off a difference that visibility alone
       explains. Caught by noticing the light could be moved 2.5 units with the
       output staying byte-identical, which no lit scene does.
    3. *The scenes could not cast shadows at all.* Everything was built from
       `Square`, and **a `Square` is a bezier circuit, which does not enter the
       shadow path**. Rendering with `shadows=True` and `shadows=False` gave the
       identical sha256. Rebuilt on `Cube` geometry, the same A/B differs, which
       is what makes the numbers above mean anything.

    The harness therefore asks **shadows on versus off, of every scene, before it
    compares anything**, and skips the mode comparison outright when that comes
    back identical. A corner-case check that cannot show it reaches its corner
    case is not evidence, and this section has now produced that non-evidence
    four times: once from the suite, three times from purpose-built scenes.

    **What is still not proven**, stated so the next reader does not over-read
    the table: the stack's reach check shows the stack DEPTH changes the image
    with shadows live, not specifically that a shadow ray peeled 256 surfaces and
    stopped; and the tie scene's check shows the slabs' SEPARATION matters, which
    is necessary for case 1 and not sufficient -- whether a shadow ray's two hits
    land inside the merge band is kernel state that nothing outside the kernel
    can see. Closing either properly needs in-kernel instrumentation, which is
    the same missing instrument §0's item 1 describes.

    One incidental, worth knowing before building here: 304 translucent `Cube`s
    at screen-filling size exhaust a 4 GB card on a single LD frame
    (`OutOfRenderMemory`). Peel memory scales with pixels times surfaces per ray,
    so the stack is SHRUNK on screen rather than shortened -- shortening it below
    256 would un-reach the case.

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
    the whole cache wiped (`clear_cached_kernels()` takes the Manim
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
deliberately — **and §0.5's A/B now shows it is also not a small output change,
because the limit bounds the run's EXTENT as well as its area sum**. (b) Refuse to consult `E` when the scan hit its limit, falling back
to the shipped `corr = 1` short-circuit — cheap and principled, since a truncated
sum is not an area, but it also withdraws the gate's win from every long-run
SILHOUETTE pixel, and on `cylfine` those are most of the frame (`capped` is 3011
of 3546 clean interior pixels). Neither is free; (b) needs the silhouette
population measured before it is chosen — **and it now is, on the scenes that
carry this rather than on the harness case that blocked it: the interior share
of the truncated full-mask pixels is 95% in `text_and_media` and 63% in
`solids_and_camera`, against 47% on `cylfine` (§0.5). The population (b) would
give up is 5% of the one it would fix on the real scene**, which is close to the
opposite of what this paragraph assumed. Both still need a kernel recompile and a
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
equal. A 1e-05 wobble in a color would be invisible. But this feeds a
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
adds into `pix_accum`, bounded at `|d| = 1` because they only ever perturb a color
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

6.6.5 THE FLAG READ A DIFFERENT FRAME'S SURFACE MAP THAN THE KERNEL — latent
----------------------------------------------------------------------------
Found while writing `_notch_scene_check.py`, which has to compute a fragment's
surface id the way the kernel does and so had to read both derivations.

`tri_obj` is `[T, N]` for a diced logical-PN primitive — row → source surface,
per frame, because the adaptive levels re-lay the rows every frame. A fragment's
compact pixel index is CHUNK-relative (`_pair_pixel` writes `lp = (f -
time_start) * ppf + p - tile_start`), and every kernel converts it back with
`f = time_start + g // ppf` before indexing. The ONE-MESH reduction in
`prepare_sparse_raster_coverage` did not: it used `pix_s // ppf` directly, so on
**any chunk that does not start at frame 0** it grouped fragments by a different
frame's surface map than the resolve it feeds. Every other frame derivation in
that same file adds `time_start` (`_window_pairs`'s `f_abs`, both per-frame pair
loops), which is what makes this a slip rather than a design.

**Measured reach: zero.** Over all six `tests/full_renders` scenes — 41 to 78
offset chunks each, five of the six carrying per-frame `tri_obj` — **not one
fragment's surface id moves** between the two rows. The reason is worth writing
down, because it is what bounds the defect: `_logical_pn_tri_obj` maps a row to
the SOURCE SURFACE of the patch that owns it, and that map is frame-invariant
whenever all of a primitive's patches belong to one surface. Every PN primitive
in those scenes is one mesh. It takes a primitive carrying SEVERAL surfaces — a
packed-grid `Surface`, or several meshes batched into one primitive — for the
rows to differ at all. A purpose-built packed grid under a 512 MB override
(`--row-split-demo`) chunked as intended and still moved nothing, so this is
recorded as latent rather than as a bug anyone has seen bite.

Fixed anyway, because the two derivations must not be allowed to disagree while
"the ids happen to be equal" is the only thing between them and a wrong answer:
`_tri_obj_row` is now the single place that answers the question, with
`test_the_tri_obj_row_does_not_depend_on_where_the_chunk_starts` pinning the
invariant (the row is a property of the ABSOLUTE frame, so it cannot depend on
where the render loop split the batch). Output is unchanged, which the measured
reach predicts and `tests/full_renders` confirms.

§7.11's lesson generalizes past a gate: the same question asked in two languages
needs one answer, and the host/kernel boundary is where this codebase keeps
finding second ones.

6.7 EXACT RUN TOTALS FROM A HOST SEGMENT REDUCTION  [BUILT, default OFF]
-------------------------------------------------------------------------
`ALGAN_ANALYTIC_AA_RUN_EXACT`, `aa_grp = 5`. The third answer to §0.5's
limitation, and the one that subsumes the other two: instead of raising the scan
budget (a) or refusing to use a truncated sum (b), take the run's totals from a
segment reduction on the host and delete the kernel's forward scan.

**Why it is a reduction at all.** `_aa_run_scan` accumulates three things over
consecutive fragments sharing `(pixel, surface, facing)`, stopping at a circuit:
the exact-area sum `E`, the sample union `U`, and the run's end. Those three
terminators make a run a SEGMENT of the CSR the host already owns — the same
structure it reduces to build `frag_cap`. So the host can compute all three
exactly, at any run length, and the kernel reads them in O(1). This does not
raise the limit, it removes the loop.

Two lanes, because the argument count is not free (`raster_first_shade` already
takes 47 ndarrays): `frag_run_e` (f32) and `frag_run_uw`, which packs the union
in its low 8 bits and the fragments REMAINING in the run above them. Remaining,
not an absolute index, so the per-slice views `shade_sparse_raster_coverage`
takes stay valid.

**THE HOST HALF IS VERIFIED, and the check found two bugs.**
`_notch_scene_check.py --verify-lanes` turns the reduction on while pinning
`_aa_group` below 5, so the host fills the lanes, the kernel still compiles the
shipped variant, and the render stays byte-identical to its baseline — then it
diffs the lanes against its own independently derived reduction. Over the six
full-render scenes: **49,644,625 run starts, 0 bad E (worst 1.19e-07, an f32
ulp), 0 bad U, 0 bad end.**

It did not start there. Both bugs were in the union, and both were invisible
until measured:

* **Summing masks instead of OR-ing them is wrong here, and the design note that
  proposed it said why without believing it.** Within one SHEET the fill rule
  partitions the sub-samples, so a sum would be exact. But a run is consecutive
  fragments sharing `(surface, facing)`, and a concave mesh can lay two
  front-facing sheets next to each other in depth order; their masks overlap,
  the sum passes 255, and it carried straight into the packed extent field.
  Measured before the fix: 82,061 of `complex_hierarchy_become`'s 2.8M run
  starts and 15,909 of `shapes_and_timeline`'s 11.8M. `_aa_run_scan` ORs, so
  this must too.
* **Then the PROBE was the one still summing**, and the disagreement went UP
  (82,061 → 101,440) rather than away. Two instruments, one of them wrong, and
  the second reading is what said which (§7.9).

Both sides now OR one bit lane at a time. Eight scatter-adds against a reduction
that already runs at ~1% of a render (§6.6.3).

**THE KERNEL HALF REPRODUCES THE RAISED-LIMIT ANSWER ON THE SCENES THAT CARRY
THE DEFECT, AND NOT EVERYWHERE.** The oracle was supposed to be exact: with
`_AA_MAX_RUN_SCAN` above a scene's longest run, the budget cannot bind, so the
two arms should agree. Rendered against each other:

    scene                     max|d| vs the 128-limit arm   pixels/frame
    complex_hierarchy_become        0  byte-identical                 0
    manim_compat_and_plots          0  byte-identical                 0
    text_and_media                  0  byte-identical                 0
    shapes_and_timeline             6            <= 10 px, 1 frame of 301
    solids_and_camera              18         <= 1,540 px, 65 of 239
    materials_and_lighting         42        <= 28,854 px, 79 of 179

`text_and_media` is the load-bearing row: it carries 420,552 truncated pixels,
by far the largest population in this file, and the two arms are byte-identical
there. Whatever the other three rows are, they are not the run reduction being
wrong about truncation.

**The oracle was a bad prediction, and the reason was written down before it was
used.** The two arms accumulate `E` differently — the host sums in float64 and
rounds to f32, the kernel sums f32 sequentially — so they differ in the last
bits BY CONSTRUCTION. §6.6.4 is the same lesson: a float reduction that feeds a
threshold is not a color. Here `E` feeds `|1 - E| > _AA_FULL_DUST`, a division
by `Q`, and `eff > MIN_ALPHA`, and on `materials_and_lighting` it also decides
whether a fragment emits a SHADOW EVENT, which is discrete — a surface point
gains or loses a shadow ray rather than shifting by an ulp. That scene is the
only one with shadows and it is the largest disagreement; `solids_and_camera`
has neither shadows nor glow and disagrees ten times less. Consistent, but not
demonstrated — nobody has traced one of those pixels.

So the honest statement: the reduction is exact and verified, the kernel reads
it correctly where truncation is what matters, and the arm is **not** a
byte-reproduction of raising the limit. It has to be judged on its own frames.

**WHAT IS LEFT BEFORE IT COULD SHIP.**

1. **Look at the frames.** Nothing in this section has done that (§0.1 rule 5).
   `materials_and_lighting` moving 42 channel values over 10% of a frame needs
   `_diff_frame.py` and an explanation, not a hypothesis.
2. **Decide the `E` precision question deliberately.** Matching the kernel's f32
   sequential sum would make the arms comparable but reintroduces the
   non-reproducibility §6.6.4 removed; keeping float64 is more accurate and
   permanently un-oracle-able. It is a real choice and this section does not
   make it.
3. **Cost.** Two f32/i32 lanes per fragment (8 B), accounted in
   `discovery_bytes`, against a bounded loop deleted from two megakernels' hot
   paths. It could be a speed-up; nothing here measured it, and this machine
   cannot (§7.15).
4. **Baselines.** It moves four of six scenes, so both device sets — and the CPU
   set is the standing debt (§3.5).

**Where it leaves (a), (b) and (c).** (c) — reuse `frag_cap` on the full-mask
arm — is still the cheapest correct thing available and is exact there (§0.5).
This supersedes it in scope rather than in kind: it is the same idea taken to
both arms and to `U` as well as `E`, which is what (c) provably cannot reach.

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


6.7.1 THE DENSE RESOLVE WAS READING §6.7's LANES OUT OF BOUNDS — latent, shipped
--------------------------------------------------------------------------------
`raster_iteration_zero` — the DENSE resolve — passes one-element dummy arrays
for `frag_run_e` / `frag_run_uw`, because the segment reduction that fills them
is part of the SPARSE emission and does not exist there. The comment beside them
said `_aa_run_exact` compiles the reads out.

It does not. `_aa_run_exact` is a function of `aa_grp`, `aa_grp` came from
`_aa_group`, and `_aa_group` answers a question about the *settings*, not about
which path is launching. So with `ALGAN_ANALYTIC_AA_RUN_EXACT=1` the dense
kernel compiled the reads in and indexed a one-element array by fragment index.

**What it cost — CORRECTED.** This section originally attributed
`shapes_and_timeline`'s last-twelve-frames move to the OOB read, on the
inference that its fade-out segment took the dense path. Measured at HEAD
(§6.7.3): **every batch of that scene takes the SPARSE path** — both arms,
whole render, `raster_iteration_zero` never launches — no raster-path default
changed in between, and the move survives this fix byte-for-byte identical. So
the OOB read never fired there, the attribution is retracted, and the real
mechanism is §6.7.3's. The fix itself stands on its own feet: a dense launch
at `aa_grp == 6` would index a one-element array by fragment index, the cap is
the correct defense in the path's own language, and
`test_analytic_aa_gates.py` pins it.

**The fix is `_aa_group_dense`**, which caps the level at 5, and the point of it
is where it lives: the level a path can express is a property of the PATH, so
the cap belongs beside the launch and not inside the kernel. Capped at 5 rather
than 4 because §6.8 needs only `frag_cap`, which the dense path does pass at
full length — as the "no ceiling" sentinel, so the rule is inert there rather
than wrong. `tests/unit_tests/test_analytic_aa_gates.py` pins it; the failure it
guards against is silent wrong output, not a crash.

This is §0.1 rule 4 again, and the fourth time: a question asked in two
languages needs one answer.


6.7.2 THE ARM IS NOW CONFINED TO TRUNCATED RUNS, WHICH IS WHAT MADE IT LEGIBLE
--------------------------------------------------------------------------------
As first built, the kernel read the host lanes at EVERY scanned run start. That
replaces `E` even on runs the scan finished, where the host's float64 sum and
the kernel's sequential f32 one differ in the last bits by construction. The
cost of that was measured, and it is not small:

**`materials_and_lighting` moved by 42 channel values over 28,854 pixels — 16%
of a frame — on a video with ZERO truncated runs.** 0 of 4,740,970 scanned run
starts over 179 frames; its longest run is exactly 16. Nothing there is the
defect §6.7 exists to fix.

Reading the lanes only where `_aa_run_scan` reports budget truncation makes that
scene **byte-identical** and leaves every truncated run exact. The bounded loop
stays (the confinement needs its truncation flag), so §6.7 no longer claims to
delete work from the hot path — it adds two lanes and keeps the loop.

**Three hypotheses were tested along the way and all three are wrong.** They are
recorded because each is the kind of explanation that sounds sufficient:

* *Bloom amplification.* `shapes_and_timeline` has no glow.
* *Re-windowing.* The two lanes grow `discovery_bytes`, and that scene really
  does go from 29 to 32 frame batches. But `--verify-lanes` — host reduction on,
  kernel pinned to the shipped variant, so the ONLY thing changed is whether the
  kernel reads them — renders byte-identical at 32 batches. Windowing is not it.
* *The shadow-event decision.* §6.7's own note guessed that `E` deciding whether
  a fragment emits a shadow event turned an ulp into a whole shadow ray, and
  `materials_and_lighting` is the only scene with shadows. Rendered with
  `shadows=False`, the unconditional arm still moves it by 42.

**The `E` precision question is decided: keep float64.** `--precision` measures
what the lane check could not — that check compares the host lane against the
probe's own host reduction, so it is host-versus-host and blind to this. Against
the kernel's actual sequential f32 sum, on `materials_and_lighting`: worst
|E_host − E_kernel| **1.79e-07**, one ulp, over 4,740,970 scanned run starts,
with **0** dust-band verdict flips and **0** runs where `corr` moves by more
than 1e-4. Matching f32 sequential summation on the host would need a segmented
f32 scan — `cumsum` — which is not reproducible on CUDA, so it would reintroduce
precisely the non-determinism §6.6.4 paid to remove, on the quantity that feeds
the discrete decisions. Accuracy and reproducibility point the same way.


6.7.3 THE `shapes_and_timeline` MOVE IS THE CODEC'S RENDERING OF 18 PIXELS
--------------------------------------------------------------------------------
The last open question of §6.7: with the arm confined and §6.7.1 capped,
`shapes_and_timeline` still moved by 31 channel values over 4,514 pixels —
~37,000 pixel-frames across its last twelve frames — while the whole render
holds 198 truncated runs over 197 pixel-frames. Those numbers looked
irreconcilable, and the elimination that reconciled them is worth keeping
whole (`benchmarks/_c3_*.py`; every arm below diffed against an arm-OFF
render that is itself byte-identical to the committed CUDA baseline):

* **The move is real at HEAD and perfectly reproducible** — the twelve frames
  differ, every other frame is byte-identical at zero tolerance, and a second
  render of each arm reproduces its video byte-for-byte.
* **Every batch takes the sparse path** (both arms — `raster_iteration_zero`
  never launches), so §6.7.1's dense-path story could not be the mechanism;
  and all 198 truncated runs sit in the fade-out's own sparse batches.
* **The moved DECODED pixels are not the truncated population**: on the worst
  frame, 4,514 moved pixels held 7 truncated runs, and most moved pixels'
  longest run is 1–8 fragments.
* **The lanes' existence is innocent**: reduction on, arena grown, windows
  re-split, kernel pinned to the shipped level — byte-identical. Halving
  `WAVEFRONT_TILE_RAYS` — byte-identical. Post-processing off — the diff is
  unchanged (bloom is a no-op without glow, so §6.7.2's dismissal was right
  for the wrong-sounding reason).
* **A one-pixel `ALGAN_AA_DUMP` golden walk of the worst pixel is IDENTICAL
  under both arms** — same fragments, same engaged run (complete, length 3),
  same corr, same final accumulation — while its decoded value differs by 31.

That contradiction has exactly one resolution, and re-rendering both arms
losslessly (`codec="libx264rgb"`, `ffmpeg_params=["-crf", "0"]`) confirms it:
**the renderer's change is 18 pixel-frames — 1–3 pixels per frame, worst
|d| 21 — on the `PointCloudDot` ring's midline, where overlapping dots of one
packed cloud share a surface id and fuse into the scene's only long runs.**
Those are engaged truncated runs taking the exact totals, which is the arm
doing precisely what it ships to do. The other ~37,000 pixel-frames are the
suite comparison's own instrument: the videos are H.264 at yuv420p with
I-frames 250 apart, so 1–3 changed pixels ride inter-prediction, 16x16 DCT
blocks and 2x2 chroma subsampling across the dot footprints for the rest of
the GOP. The window (frames 286–297 of 301) is also mechanism, not noise:
`animate_fade_out` despawns mobs staggered, a despawned dot's fragments stay
in the CSR at zero alpha, a zero-alpha write leaves `svis` exactly uniform, so
the walk's run gate only reaches a deep truncated run between the first
despawn and the last — the census counts 2–12 truncated runs per frame across
the WHOLE fade-out, but only the engaged 1–3 per frame inside that window can
read the lanes.

Two standing lessons. First, §0.1 rule 1 in a new costume: a pixel diff read
from an MP4 measures the encoder's output, not the renderer's. Byte-identity
claims survive (identical raw frames encode identically — every byte-identical
verdict in this file stands), but every "moved by N over M pixels" figure in
these documents is measured through the codec, including §0.5's raised-limit
table, whose `shapes_and_timeline` row is this same population. Before
attributing a move's size or shape, re-render both arms losslessly. Second,
rule 4 held again: the 198-vs-37,000 "contradiction" was two instruments
answering different questions, and both were right.

**The same instrument, pointed back at §6.7.2's own case** (a worktree at
fc0f93f, the unconditional-read revision, both arms lossless —
`_c3_uncond_lossless_ab.py`): the "42 channel values over 28,854 pixels — 16%
of a frame" that motivated the confinement decodes to **2,063 pixel-frames
over 35 of 179 frames, worst |d| 4**, mostly single |d| = 1 pixels. The ulp
flips are real and the confinement still removes them exactly, but their true
cost was ~10x smaller in magnitude and ~two orders smaller in extent than the
number the decision was made on. That reopens §0.5's option (a) — raising the
scan bound — whose case was closed on the same codec-inflated evidence; the
open queue's §C.4 carries the reopened choice.
--------------------------------------------------------------------------
`ALGAN_ANALYTIC_AA_RUN_CAP`, `aa_grp = 5`. The cheap half of what §6.7 does,
needing no new lane, no new host reduction and no new argument.

On the FULL-MASK arm every sample in the pixel is owned by the run, so `Q == 1`
and `corr = min(E, 1)` — the mesh's total claim over the pixel. The one-mesh
reduction (§6.6) has already computed exactly that quantity as the larger of the
two sheets' exact areas, and the walk already loads it for the cap clamp. Where
the scan ran out of budget and a cap exists, take it.

**Scored per truncated full-arm pixel against `corr(unbounded run sum)`**, with
the gate ON so the probe scores the rule the walk actually runs:

    scene                cap available   err on capped px   corr>1 introduced
    text_and_media       89,117/106,283       0.0000             0
    solids_and_camera     2,259/  2,661       0.0001             0
    complex_hierarchy         2/      3       0.0005             0
    shapes_and_timeline      45/     67       0.0000             0

Exact on the scene carrying most of the defect, and `corr > 1` is impossible on
this arm since `Q` is 1 — so it cannot push the write into the clamp-and-
redistribute path. It does not reach the ~15% of truncated pixels holding more
than one mesh (no cap exists), and it does not touch the partial-mask arm, which
needs the sample union fixed and not merely the area.

**THE TRUNCATION TEST IS THE SUBTLE PART.** `_aa_run_scan` had to report whether
it stopped for BUDGET, and "the loop stopped with fragments left in the pixel"
is the wrong test: a run of exactly `_AA_MAX_RUN_SCAN` that ends of its own
accord leaves the loop in the same state as one that was cut short. It probes
one fragment further with the loop's own three terminators and accumulates
nothing. That is not a corner case — `materials_and_lighting`'s longest run is
exactly 16, so the naive test would have replaced an EXACT `E` with an estimate
on every one of its 4.7M scanned runs. The probe is paid only by runs that reach
the budget, and compiles out entirely (`want_trunc`) when the rule is off, which
is what keeps the shipped kernel byte-identical with the gate off.

**Interaction with §6.7.** §6.7 fixes `U` and the extent as well, so it
supersedes this; the kernel gates §6.8 on `_aa_run_cap and not _aa_run_exact` so
the two can never both write `rE`. In the ladder §6.8 sits at 5 and §6.7 at 6.

**Output.** Moves `solids_and_camera` by 54 channel values and `text_and_media`
by 49. Reviewed frame by frame: fine interior speckle on diced and textured
surfaces, no silhouette movement and no structural change. What it is doing is
letting the near sheet's prefix claim the mesh's whole footprint, after which
the one-mesh ceiling starves the uncorrected tail that used to paint on top of
it — so the pixel's total ink is unchanged and its COLOUR mix is corrected. That
is why the signed change is near-symmetric rather than a brightening, which is
worth knowing before reading a mean-delta and concluding nothing happened.


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
worries about, because a 1e-05 error in a color is not a bug. §6.6 put one behind
a **comparison** — the cap clips when `eff > frag_cap - mesh_ink` — and the same
1e-05 became 28 channel values over 9.6% of a frame, because a threshold turns an
epsilon into a branch and bloom turns a branch into a region (§6.6.4).

The rule to carry forward: **classify every host-side float reduction by what
consumes it.** Feeding a color, an atomic add is fine. Feeding a comparison, a
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

7.19 A harness that renders the full-render scenes must diff its own render
------------------------------------------------------------------------------
`_notch_scene_check.py` reproduced `tests/full_renders`' render environment
carefully — `PREVIEW`, the same `available_memory_override`, the same working
directory, `animate_fade_out=True` — and still rendered a different scene than
the suite does, because the fixture that registers the vendored fonts lives in
`tests/conftest.py` and a script does not run it. `Text` then resolves through
whatever Pango offers on the host. Measured: **205-232 channel values from the
committed baselines over 8-11% of every frame of all six scenes.** Nothing in
the run looked wrong; the notch counts it produced were within a few percent of
the correct ones, which is worse, because they were plausible.

Two things follow, and the second is the general one:

* Load `tests/conftest.py` by path rather than copying its ten lines, so the two
  cannot drift.
* **Diff the harness's own render against the committed baseline before reading
  any number off it**, and print the result beside the numbers. It costs one
  video decode. Every claim of the form "measured on the six full-render scenes"
  is a claim about *those* scenes, and this is the only thing that checks it —
  the same discipline as §0.1's rule 1, applied to the harness rather than to
  the feature.

Corollary for anyone A/B-ing a kernel constant on these scenes:
`clear_cached_kernels()` takes the **Manim Tex geometry cache** with it, and the
first render after that differs from every later one (§4.10). Back up and
restore `~/.algan/cache/taichi` instead, or render twice per arm and keep the
second.
