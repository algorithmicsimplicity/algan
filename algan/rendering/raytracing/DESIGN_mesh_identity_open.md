# Algan — Mesh Identity: the work that is left

**This file is a queue, not a history.** Everything in it is unstarted,
half-finished, or blocked. What was built, measured and settled lives in
`DESIGN_mesh_identity.md`, which is the record and the place to look up *why* a
number is what it is; this file cites it by section rather than repeating it.

It is written to be startable cold. §A is what you need to run anything, §B is
the two debts that gate several items, §C onward is the queue in priority order,
§Y is the handful of rules this subsystem keeps re-learning, and §Z is the
list of things that are CLOSED so nobody rebuilds them.

**If you read nothing else:** the largest measured defect in this subsystem is
§C — the analytic-AA run scan's 16-fragment budget costs real coverage on real
scenes, ~0.28 of a pixel on 314,072 pixels of `text_and_media` alone — and three
candidate fixes exist, one of them already built and gated off. Start there.


================================================================================
A. RUNNING ANYTHING
================================================================================

`CLAUDE.md` is the repo-wide contract and still governs. This is the part
specific to this subsystem.

**The venv.** `<venv-python>` is `.venv\Scripts\python.exe` on Windows,
`.venv/bin/python` elsewhere, or `uv run python`. The system Python has no
taichi.

    <venv-python> -m pytest -q tests/unit_tests tests/fast   # ~9 min, CI's paths
    <venv-python> -m pytest -q tests/full_renders            # ~8 min, 6 dense scenes
    <venv-python> -m ruff check --no-fix algan tests         # NEVER without --no-fix
    <venv-python> -m ruff format --check algan tests

Green at the tip of this work: **1056 passed, 89 skipped** and **7 passed**. If
either is red before you change anything, find out why first.

**Never name a `*_taichi.py` file on a ruff command line.** They are excluded in
the config; naming one explicitly overrides the exclusion, and the
auto-inserted `from __future__ import annotations` breaks kernel compilation.

**The harnesses, and the question each answers.** All live in `benchmarks/`.

    _notch_scene_check.py    the run-scan limit scored per pixel over the SIX
                             full-render scenes; --verify-lanes checks §C's host
                             reduction; --cases runs the synthetic cases instead
    _aa_run_gate_check.py    coverage error, ink wobble, --verify replay,
                             --notch-probe (the §6.3.2/§6.6 instrument of record)
    _aa_line_check.py        ink wobble along an edge over nine angles
    _one_mesh_ab.py          cost of the one-mesh cap
    _watertight_check.py     cracks, double blend, pixel diff across §3.2
    _weld_check.py           §3.1's weld, textured and normal-mapped arms
    _bez_bvh_ab.py           §3.4's bezier split against a noise floor
    _diff_frame.py           side-by-side worst frame of two videos (LOOK at this)
    _video_diff.py           how far two videos moved, and over how many pixels

**The machine that produced every number here** is a Windows box with a GTX 1050
(4 GB), driver 576.52, Taichi 1.7.4, torch 2.7.1+cu128. Four consequences:

* **It owns the CUDA baselines and must never write the CPU ones** (§B.1).
* **It throttles, so wall-clock A/B is not available.** A control kernel the
  change cannot touch drifts as much as the target. Prefer counts, byte-diffs
  and in-process alternating A/B with an explicit control. A number that
  straddles 1.0 across two orderings is room temperature.
* **A cold Taichi rebuild is ~10 minutes** for this scene set after wiping the
  kernel cache (the older 35-45 minute figure was measured on a larger variant
  set). A new `ti.static` template VALUE is a new variant with its own compile.
* **4 GB is a real constraint.** Shrink geometry on screen rather than reducing
  object count when a scene needs depth.

**Wiping the kernel cache for a kernel-constant A/B: do NOT use
`clear_cache(taichi_kernels=True)`.** It deletes the whole `~/.algan/cache`,
Manim Tex geometry included, and the first render after that differs from every
later one — which lands in the diff looking like the change. Back up
`~/.algan/cache/taichi`, `rm -rf` only that, restore afterwards, and verify the
restore by re-rendering one scene against its baseline.

**Any harness that renders the six full-render scenes must diff its own render
against the committed baseline before reporting a number.** Reproducing the
suite's settings is not enough: the scenes name `Algan Test Sans` and
`tests/conftest.py` registers the vendored faces, so a script that skips it
measures a scene 205-232 channel values away from the one it names.
`_notch_scene_check.py` loads `tests/conftest.py` by path for this reason and
prints the baseline diff beside every table.


================================================================================
B. THE TWO DEBTS THAT GATE OTHER WORK
================================================================================

B.1 The CPU baseline sets are stale, and this machine cannot write them
-----------------------------------------------------------------------
**STATUS: open, and it blocks §C.4 and §D.**

`tests/fast/expected_outputs_cpu/` and `tests/full_renders/expected_outputs_cpu/`
were written at `2d1432a`, before several gates flipped. This machine's own CPU
render of `tests/fast` misses the committed CPU baseline by 30 channel values on
43 of 45 frames *before any change*, so regenerating from here would replace a
baseline CI reproduces with one it does not.

**How to pay it.** On a machine whose CPU renders reproduce the committed CPU
set — or by accepting a new reference machine and saying so in the commit:

    ALGAN_UPDATE_FAST_BASELINE=1 <venv-python> -m pytest -q tests/fast
    ALGAN_UPDATE_FULL_RENDER_BASELINES=1 <venv-python> -m pytest -q tests/full_renders

and **look at the frames** (`_diff_frame.py`) before committing. Two traps, both
of which have produced a silently wrong answer here: `CUDA_VISIBLE_DEVICES=`
(empty) does not hide the GPU on Windows — use `-1`; and the render suites pick
their baseline directory from `torch.cuda.is_available()`, not from the render
device, so a CPU render that can still see the GPU compares against the CUDA
set and passes.

**Why it gates.** Every remaining output-moving change (§C.4, §D) needs both
device sets regenerated. The efficient path is to batch them: land the code
changes gated off, then pay the baselines once for all of them.

B.2 `shapes_and_timeline` moves for a reason nobody has found
--------------------------------------------------------------
**STATUS: open, unexplained, and it undermines confidence in any re-baseline of
that scene.**

Raising `_AA_MAX_RUN_SCAN` from 16 to 128 moves that scene by **31 channel
values over 4,514 pixels** on its worst frame, 12 of 301 frames affected. But
the whole 301-frame video contains only **107 truncated runs**, 7 of which
change their coverage by a mean of 0.0113. Those 7 pixels cannot produce 4,514.

What is ruled out: the scene has no `glow`, so bloom amplification is not
available as an explanation; the shipped arm is byte-reproducible (verified
twice against its baseline); split-pixel nondeterminism is capped at |d| = 1.

**How to chase it.** Dump the moved pixels' coordinates from `_video_diff.py`'s
worst frame, then point `_notch_scene_check.py` at that scene and ask whether
those pixels appear in ANY of its populations (truncated, scanned, notched). If
they do not, the scan limit is reaching them through something this file has not
modelled — the most likely candidate is a run start the probe does not score,
since it scores only the first non-circuit fragment per pixel and stops there.

**Done when** the 4,514 pixels are attributed to a mechanism, or the scene is
shown to move for a reason unrelated to the constant.


================================================================================
C. THE RUN-SCAN LIMIT — the largest measured defect here
================================================================================

**The mechanism.** `_aa_run_scan` (`raster_taichi.py`) walks at most
`_AA_MAX_RUN_SCAN = 16` consecutive fragments of one sheet, returning the exact-
area sum `E`, the sample union `U`, and the run's end. On a longer run all three
are short, and the rule then uses them as though they were complete:
`corr = min(E, 1) / Q` where `Q = popcount(U)/N`. A run's fragments partition
the sheet's samples, so the run's total claim is `min(E, 1)` on either arm —
which means a truncated `E` is a coverage error, full stop.

**Measured on the six full-render scenes** (`_notch_scene_check.py`, CUDA,
`PREVIEW`, every frame, each render verified byte-identical to its baseline
first):

    scene                    truncated   full-mask arm      partial-mask arm
                                          n     mean          n        mean
    text_and_media             420,552  106,283  0.0849   314,072    0.2762
    solids_and_camera            7,008    2,180  0.0302     4,335    0.2088
    complex_hierarchy_become        49        3  0.0275        42    0.0825
    shapes_and_timeline            107        0       -        40    0.0647
    materials_and_lighting           0        -       -         -         -
    manim_compat_and_plots           0        -       -         -         -

Means are coverage lost per pixel. The **partial arm is the larger half in both
population and magnitude** and was unscored until recently — §0.5's "~2 channel
values" describes the full-mask arm only. `materials_and_lighting` is clean
because its longest run is exactly 16; `manim_compat_and_plots` is all circuits,
which never enter this path.

Three fixes exist. They are not alternatives in the usual sense — C.1 is a
subset of C.3, and C.2 is a different trade.

C.1 Reuse `frag_cap` on the full-mask arm  [SCORED EXACT, NOT BUILT]
---------------------------------------------------------------------
**The cheapest correct thing available, and it is not built.**

The host already reduces every covered pixel's exact areas by facing and packs
`max(front, back)` into `frag_cap` for one-mesh pixels; the walk already loads it
for the cap clamp (`raster_taichi.py`, the `_AA_ONE_MESH_BIT` branch). On the
full-mask arm `Q == 1`, so `corr = min(E, 1)` and `frag_cap` is exactly the
quantity wanted.

**Scored against `corr(unbounded run sum)` as the ideal**, per truncated pixel:

    scene                cap available   shipped err   residual   over-covers
    text_and_media       89,117/106,283      0.0832     0.0000      0 px
    solids_and_camera     2,259/  2,661      0.0253     0.0001     35 px (0.0555)
    cylfine (harness)       849/    849      0.0791     0.0003    125 px (0.0039)
    sphere  (harness)        61/     61      0.0208     0.0002      4 px (0.0016)

On the scene carrying most of the defect it is **exact on every pixel where the
cap exists**, and it introduces **zero** `corr > 1` pixels (impossible on this
arm, since `Q == 1`). Carrying the fragment's own-facing sum in a new lane
instead was also scored and matches to the fourth decimal — **it does not earn
its lane**.

**How to build it.** One predicate at each of the two run-scan sites
(`raster_first_shade` and `raster_shadow_event_build`, which must stay in
lockstep or every shadow id desynchronizes from its fragment): when the scan hit
its budget and `frag_cap[idx] <= 1.0` (the availability test — 2.0 is the "no
ceiling" sentinel), take `rE = frag_cap[idx]`. `_aa_run_scan` must return
whether it stopped for budget rather than because the run ended; a run of
exactly 16 that ends naturally is otherwise indistinguishable.

**Limit, stated plainly.** 15% of truncated pixels hold more than one mesh, have
no cap, and keep the shipped behaviour. And the arm it fixes is the *smaller*
half of the defect.

**Done when** it renders, and `_notch_scene_check.py` reports the full-arm
shipped error at zero.

C.2 Refuse to consult `E` on a truncated scan  [DECIDED AGAINST, but re-costed]
-------------------------------------------------------------------------------
Fall back to the `corr = 1` short-circuit when the scan hit its budget. Cheap
and principled — a truncated sum is not an area — but it hands back the relaxed
gate's win on long-run SILHOUETTE pixels.

That objection came from the harness rod, where the silhouette population is
most of the frame. **On the real scenes it is not**: the interior share of
truncated full-mask pixels is **95%** in `text_and_media` and 63% in
`solids_and_camera`, against 47% on the rod. The population C.2 would give up is
5% of the one it would fix.

**Recommendation: do not build it.** C.1 fixes the same arm exactly and gives up
nothing. C.2 is recorded here only because §6.3.2 left it open and the number
that was missing now exists.

C.3 Exact run totals from a host segment reduction  [BUILT, DEFAULT OFF]
-------------------------------------------------------------------------
`ALGAN_ANALYTIC_AA_RUN_EXACT`, `aa_grp = 5`. See `DESIGN_mesh_identity.md` §6.7
for the full write-up; this is what is left.

It takes `E`, `U` and the run's extent from a host segment reduction over the
CSR and deletes the kernel's forward scan. It is the only one of the three that
fixes `U` as well as `E`, which is what the partial arm needs and what no area
alone can supply.

**Verified**: 49,644,625 run starts across the six scenes, 0 bad `E` (worst
1.19e-07, one f32 ulp), 0 bad `U`, 0 bad extent, with every render still
byte-identical to its baseline (`--verify-lanes` pins `_aa_group` below 5 so the
host fills the lanes while the kernel compiles the shipped variant).

**Four things stand between it and a default.**

1. **Look at the frames.** Nothing has. Use `_diff_frame.py` on the worst frames
   of `materials_and_lighting` and `solids_and_camera` against the shipped arm.
2. **Decide the `E` precision question.** The host sums in float64 and rounds to
   f32; the kernel sums f32 sequentially. They differ in the last bits by
   construction, and `E` feeds thresholds (`_AA_FULL_DUST`, `MIN_ALPHA`, a
   division by `Q`) rather than a colour. That is why the arm reproduces the
   raised-limit render byte-for-byte on three of six scenes — including
   `text_and_media`, which carries 420,552 truncated pixels — and differs by 6,
   18 and 42 channel values on the other three. Matching f32 sequential
   summation would make the arms comparable but reintroduces the
   non-reproducibility §6.6.4 removed. Keeping float64 is more accurate and
   permanently un-oracle-able. **Choose deliberately and write down which.**
3. **Cost.** Two lanes per fragment (8 B, accounted in `discovery_bytes`)
   against a bounded loop deleted from two megakernels' hot paths. It could be a
   speed-up. Nothing measured it, and this machine cannot (§A).
4. **Baselines** (§B.1).

**One hypothesis worth testing while doing (1):** `materials_and_lighting` is
the only scene with shadows and the largest disagreement, and on it `E` also
decides whether a fragment emits a *shadow event* — a discrete outcome, so an
ulp becomes a whole shadow ray. `solids_and_camera` has neither shadows nor glow
and disagrees ten times less. Consistent, untraced.

C.4 Which to ship
------------------
C.1 and C.3 are not in competition: C.1 is a two-line special case of what C.3
does generally. The efficient order is **C.1 first** (exact, free, no new
argument, no output risk beyond its own arm), then C.3 once (2) is decided —
and both re-baseline together with §D under §B.1.


================================================================================
D. §3.1's WELD — needs baselines and nothing else
================================================================================
**STATUS: landed, gated off (`ALGAN_WELD_SURFACE_SEAMS`), both blockers cleared.**

`get_grid_to_triangle_indices` bridges a closed surface's wrap column back to
column 0 and collapses a pole fan to one vertex. Both original objections are
measured away: the textured/normal-mapped risk is closed (byte-identical on a
static frame, `_weld_check.py`), and the morph path now asks
`surface_weld_flags` for the same grid the render path does, so a `Sphere` no
longer morphs from a different triangulation than it renders. The whole unit
suite is green with it on.

**What is left is only that it moves a moving PN scene** —
`materials_and_lighting` by 31 channel values over 10.2% of a frame and
`solids_and_camera` by 54 over 7.2% — because the dice level is chosen per patch
per frame from projected size, so a different triangle list can land on a
different level. That needs both device baseline sets (§B.1).

**Done when** the baselines are regenerated with the frames reviewed, and the
default flips.


================================================================================
E. A TRAVERSAL-STEP COUNTER — the instrument two items are waiting on
================================================================================
**STATUS: not started. It settles §F and most of §G.**

Nothing in this repo counts traversal steps, so two standing claims cannot be
confirmed or refuted, and one of them is the whole case for a multi-day project.
A step count is **deterministic**, which makes it the right instrument for a
machine that cannot do wall-clock (§A).

**Two ways, and the cheap one is probably right.**

* A `ti.static`-gated counter in the traversal kernels compiles out when off,
  but costs a cold recompile per iteration.
* A **host replay** of the walk over the same STBVH arrays costs no recompile
  and is the pattern that worked three times in §6.6.2 — replay the same thing
  with one input changed. It must be validated against the kernel (the §6.6.2
  replay had `--verify` for exactly this) or it measures itself.

**It does NOT settle §3.2's cost**, though an earlier revision of the old doc
said it did: §3.2 changes the ray/triangle intersection TEST, not which nodes
are visited, so a step count is identical across its arms by construction.

**Done when** it reports steps per ray for a scene, agrees with the kernel on a
spot check, and can be pointed at two BVH builders.


================================================================================
F. MEDIAN-SPLIT BEZIER BVH — built, measured neutral, waiting on E
================================================================================
**STATUS: landed, gated off (`ALGAN_BEZ_BVH_SPLIT`), no evidence either way.**

`benchmarks/_bez_bvh_ab.py` on 35 circuits plus `Text` and `Tex`, moving, at
`--res md`: **byte-identical** (and the scene's own noise floor is also zero, so
that is real byte-identity, not a diff hidden under jitter), wall 0.993x — noise.
So the feared cost does not appear *and neither does the claimed benefit*. The
inherited "~20-25% fewer traversal steps" is still unmeasured.

**Do not flip it on the strength of the inherited number.** Flip it when §E can
count steps, or when order-independence work (§J) needs it.


================================================================================
G. TWO-LEVEL BVH (TLAS/BLAS) — scoped, and the perf case is measured NOT to justify starting
================================================================================
**STATUS: not started, deliberately.**

**What it would need**, at minimum: a per-mesh contiguity guarantee in the merge
(`_split_promotable` reorders promoted triangles by material value, so a partly
promoted surface already lands in two disjoint spans); a two-level build in
`stbvh.py`, which today builds one flat instance tree per geometry type; and
two-level traversal in `raytrace_kernels_taichi.py` — in the megakernel *and*
the wavefront path, plus the raster path's own gather. Days with a CUDA machine,
not a session, and a half-landed version is worse than none.

**The measurement that says wait.** On a five-solid shadowed scene at `--res md`
the BVH build is **~1% of the render** (`raster_shadow_trace` is 80%). So the
amortization half of the argument — a BLAS reusable across a batch's frames — is
worth at most that, and is dead. The instancing half attacks traversal, which
*is* large, but that scene has five instances and **no workload in the repo has
thousands of repeated meshes**, so it cannot be shown.

**Start it only if** §E exists (so the win is measurable rather than assumed)
AND a workload with thousands of repeated meshes exists to measure it on.


================================================================================
H. NESTED-IOR REFRACTION — unlocked by identity, not built
================================================================================
**STATUS: not started. The identity it needs now exists.**

`wavefront_kernels_taichi.py` treats a circuit as a thin pane (`is_pane`) and a
triangle mesh as a solid (`is_glass`), and assumes air outside every interface,
because it cannot reliably tell an entry from an exit. With a stable mesh id at
every hit, a ray can carry an "inside which mesh" stack and take the correct
*relative* IOR at each interface — glass inside glass, a sphere inside a box.

**How.** Per-ray state, so it lands in `rs_sca`/`rs_int` (`tracer.py:186`
documents the layout; `rs_sca[r, 3]` is `seam_t`, which §I may free). A depth-N
stack of mesh ids costs N slots per ray; N = 4 covers everything a scene here
builds, and overflow must degrade to today's behaviour rather than corrupt.
The entry/exit test is the facing bit, which the merge already carries.

**Watch for**: the refraction path forces the wavefront tracer, and pool sizing
there is measured, not modelled — raise the POOL, not `pool_ratio`, if a first
attempt overflows (`pool_ratio` scales tile count).

**Done when** a sphere inside a box renders with the correct relative IOR at the
inner interface, and the existing glass scenes are unchanged.


================================================================================
I. SELF-SHADOW REJECTION BY IDENTITY — unlocked, not built
================================================================================
**STATUS: not started.**

A shadow ray currently rejects its own surface with `MIN_HIT_DISTANCE = 1e-4`
(`raytrace_kernels_taichi.py:95`) plus a normal offset. Both are absolute
world-space constants with no scene-scale adaptation, and they are what produces
shadow acne at grazing light angles and on small-scale geometry.

With mesh identity the test becomes "reject a hit on the mesh this ray started
from, at near-zero `t`", which is scale-free and removes one more epsilon.

**Care needed:** rejecting the whole *mesh* is wrong for a concave solid that
legitimately shadows itself. The rejection has to be "the same mesh AND
near-zero `t`", not "the same mesh".

**Done when** a grazing-light scene shows no acne, `tests/full_renders`'
shadowed scene is unchanged or reviewed, and the constant is gone from the
shadow path.


================================================================================
J. ORDER- AND WINDOW-INDEPENDENT OUTPUT — mostly delivered, not finished
================================================================================
**STATUS: achieved at shipped defaults, not structurally guaranteed.**

The property asked for was that resolution be a function of the canonically
sorted hit list alone — independent of KBUF width, BVH builder, tile size and
batch window. The greedy `seam_t` dedup that broke it is now compiled out at
shipped defaults: `TRIANGLE_EDGE_EPSILON`/`seam_t` survive only under
`ti.static(not aa_tri)`, and `_tri_hit`'s dilated arm only under
`WATERTIGHT_TRI` off, which is no longer the default.

**What is left is two things.** Deleting the dead arms (§K) so the property is
structural rather than a consequence of the current defaults; and then actually
*using* it — reordering primitives in the merge, and §F.

**Do not claim it as delivered** until something demonstrates it: render one
scene at two KBUF widths and two batch-window sizes and diff. That check does
not exist.


================================================================================
K. THE TWO `ti.static` ARMS — a decision, not an epsilon problem
================================================================================
**STATUS: open. It is a call about what to keep, and the obvious framing is wrong.**

There is **no live `BARYCENTRIC_EPSILON` read at shipped defaults**. All three
survivors sit in arms that do not compile in: `_tri_hit`'s Möller-Trumbore arm
(live only with `WATERTIGHT_TRI` off), and two `raster_taichi` sites live only
at `aa == 0`. So this is not "delete the epsilons" — it is whether to delete two
`ti.static` branches.

**Both are A/B levers, and that is the real argument for keeping them.**
`aa == 0` is the control arm for every analytic-AA measurement in the old doc's
§6; `_tri_hit`'s dilated arm is the control for §3.2, whose cost is still
unmeasured on hardware that does not throttle. Deleting them deletes the ability
to measure, on a machine where controls are already the scarce thing.

**Two things that sound like arguments to delete and are not.** `aa == 0` is not
a per-pixel fallback — it is a whole-batch compile-time value, and no pixel ever
falls back to it at runtime. And the hard case (several partially transparent
fragments from different meshes in one pixel) does not need it: the per-sample
transmittance walk handles it exactly, since each fragment attenuates only the
samples its own mask owns.

**What deleting buys:** the constants, the `edge_hit` bit, and `seam_t` —
`rs_sca[r, 3]`, one f32 per ray. That shrinks `rs_sca`, which **moves the arena
fit**, so `test_render_batch_sizing.py`, `test_memory_model.py` and a long
multi-batch render's OOM-retry count all have to be re-checked. Do not promise
the per-ray f32 until BOTH arms have gone.

**Recommendation: keep them until §3.2's cost is measured on non-throttling
hardware**, then revisit.


================================================================================
L. EXACT ABSORPTION OF COINCIDENT DUPLICATES — unlocked, unbuilt, unmeasured
================================================================================
**STATUS: not started, and nobody has shown the symptom.**

A union of sample masks is idempotent, so two genuinely coplanar stacked quads
should stop double-darkening once coverage is taken per mesh rather than per
fragment. The machinery for that is §C.3's reduction plus the one-mesh rule.

**Before building anything, show the defect.** Render two coincident quads of
the same mesh and of different meshes and measure the darkening. If the
per-sample transmittance walk already handles it — which is plausible, since
masks partition within a sheet — this item closes with a test rather than a
change.


================================================================================
Y. FIVE RULES THIS SUBSYSTEM KEEPS RE-LEARNING
================================================================================
Each cost a wrong result that is recorded in `DESIGN_mesh_identity.md`.

1. **A check must show it REACHES its case.** "All three modes agree" was
   produced four times from scenes with no shadow in them.
2. **Replay the same inputs with exactly ONE thing changed.** That is what
   attributed the notches three times running, and what verified §C.3's host
   reduction without a kernel compile.
3. **Read which accumulator a metric scores.** An occlusion-side fix was
   predicted to close claim-side symptoms. It could not have.
4. **A question asked in two languages needs one answer.** The host/kernel
   boundary is where this codebase keeps finding second ones — `aa_grp` drifted
   once and cost most of a win; the one-mesh flag read a different frame's
   surface map than the kernel did.
5. **Look at the frames before re-baselining.** `_diff_frame.py` exists for
   this. "Measured 42 channel values" is not knowing what moved.

And one specific to this file's own instruments: **a metric that reports zero
may be an instrument that is not looking.** `_notch_scene_check.py` scored only
a pixel's first run and reported zero for `shapes_and_timeline`, which is 96%
circuit-led; and it scored only the full-mask arm, which turned out to be the
smaller half of the defect.


================================================================================
Z. CLOSED — do not rebuild these
================================================================================

* **Material dispatch coherence by sorting hits.** Built and measured **1.5-2.2x
  SLOWER** than the monolithic scatter, which drains up to KBUF hits per launch
  where sorting pays per-event kernel round trips and host syncs
  (`_wf_sorted_ab.py`, `_wf_monolith_scatter_ab.py`). `WAVEFRONT_SORT_MATERIALS`
  defaults off and the sorted route is unsupported, kept for reference.
* **Suppressing the far sheet** instead of capping it (+114% on a sub-pixel
  diced rod). Two follow-up hypotheses — scrambled facing bits, and the u-seam —
  were both tested and refuted.
* **Regrouping the run into an order-independent equivalence class**: `split` is
  ~0.02% everywhere, so there is nothing there.
* **Consulting `E` only inside the existing gate**, and **buying more samples**:
  both measured, neither is the lever.
* **Raising `_AA_MAX_RUN_SCAN` as the fix** (§C's option (a)). It works, and it
  moves four of six scenes by 31-50 channel values — most of which is not the
  notch, because the limit bounds the run's EXTENT as well as its area sum. §C.3
  does the same thing correctly and cheaper.
* **Carrying the fragment's own-facing sum in a new lane** rather than reusing
  `frag_cap`: scored, matches to the fourth decimal, does not earn the lane.
* **A traversal-step counter to settle §3.2**: it cannot, by construction.
