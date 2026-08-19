# Algan — Mesh Identity: the work that is left

**This file is a queue, not a history.** Everything in it is unstarted,
half-finished, or blocked. What was built, measured and settled lives in
`DESIGN_mesh_identity.md`, which is the record and the place to look up *why* a
number is what it is; this file cites it by section rather than repeating it.

It is written to be startable cold. §A is what you need to run anything, §B is
the debt that gates other work, §C onward is the queue in priority order, §Y is
the handful of rules this subsystem keeps re-learning, and §Z is the list of
things that are CLOSED so nobody rebuilds them.

**If you read nothing else:** two items remain unbuilt — §H (nested-IOR
refraction) and §I (self-shadow rejection by identity). Both are designed down
to the argument list in this file; neither has been started. Everything else in
the previous revision of this queue is now measured, decided, or shipped, and
the entries below say which.


================================================================================
A. RUNNING ANYTHING
================================================================================

`CLAUDE.md` is the repo-wide contract and still governs. This is the part
specific to this subsystem.

**The venv.** `<venv-python>` is `.venv\Scripts\python.exe` on Windows,
`.venv/bin/python` elsewhere, or `uv run python`. The system Python has no
taichi.

    <venv-python> -m pytest -q tests/unit_tests tests/fast   # ~9 min, CI's paths
    <venv-python> -m pytest -q tests/full_renders            # ~6 min, 6 dense scenes
    <venv-python> -m ruff check --no-fix algan tests         # NEVER without --no-fix
    <venv-python> -m ruff format --check algan tests

Green at the tip of this work: **1058 passed, 89 skipped** and **7 passed**. If
either is red before you change anything, find out why first.

**Never name a `*_taichi.py` file on a ruff command line.** They are excluded in
the config; naming one explicitly overrides the exclusion, and the
auto-inserted `from __future__ import annotations` breaks kernel compilation.

**Never edit a `*_taichi.py` file while a render or the suite is running.** The
JIT reads sources at first launch and will compile half an edit. Done here by
accident: 13 tests failed with a green tree and nothing wrong with the change.

**The harnesses, and the question each answers.** All live in `benchmarks/`.

    _notch_scene_check.py    the run-scan limit scored per pixel over the SIX
                             full-render scenes. --all-runs scores every run
                             start (upper bound) beside the per-pixel first-run
                             columns (lower bound); --precision diffs the host's
                             float64 run sum against the kernel's sequential
                             f32 one; --cap scores §6.8; --verify-lanes checks
                             §6.7's host reduction; --batch-frames prints the
                             time_start/frame mapping any pixel-to-frame claim
                             rests on; --cases runs the synthetic cases
    _c3_*.py                 the §C.3 attribution set: fadeout_ab (A/B + batch
                             path log), run_census + census_join (truncated-run
                             population vs moved pixels), dump_pixel (one-pixel
                             golden walk, both arms), tile_ab (slice partition),
                             nopost_ab (bloom), lossless_ab (the codec — §Y.7)
    _bvh_steps.py            traversal STEPS per ray, transcribed from the
                             production walk and verified against it (§E)
    _order_window_check.py   is the render a function of the sorted hit list
                             alone? KBUF / BVH order / tile size / batch
                             window, each against the scene's own noise floor
    _glyph_cache_cold_warm.py does the glyph geometry cache move a render?
    _aa_run_gate_check.py    coverage error, ink wobble, --verify replay
    _aa_line_check.py        ink wobble along an edge over nine angles
    _one_mesh_ab.py          cost of the one-mesh cap
    _watertight_check.py     cracks and double blend across §3.2 (NOT cost)
    _weld_check.py           §3.1's weld, textured and normal-mapped arms
    _bez_bvh_ab.py           §3.4's bezier split — **vacuous at defaults**, see §F
    _diff_frame.py           side-by-side worst frame of two videos (LOOK at this)
    _video_diff.py           how far two videos moved, and over how many pixels

**The machine that produced every number here** is a Windows box with a GTX 1050
(4 GB), driver 576.52, Taichi 1.7.4, torch 2.7.1+cu128. Four consequences:

* **It owns the CUDA baselines and must never write the CPU ones** (§B).
* **It throttles, so wall-clock A/B is not available.** A control kernel the
  change cannot touch drifts as much as the target. Prefer counts, byte-diffs
  and in-process alternating A/B with an explicit control. A number that
  straddles 1.0 across two orderings is room temperature. `_bvh_steps.py` now
  exists precisely because a step count is deterministic where a clock is not.
* **A cold Taichi rebuild is ~10 minutes** for this scene set after wiping the
  kernel cache. A new `ti.static` template VALUE is a new variant with its own
  compile.
* **4 GB is a real constraint.** Shrink geometry on screen rather than reducing
  object count when a scene needs depth.

**Wiping the kernel cache for a kernel-constant A/B: do NOT use
`clear_cache(taichi_kernels=True)`.** It deletes the whole `~/.algan/cache`,
Manim Tex geometry included. On this machine that is worse than the doc used to
say: `_glyph_cache_cold_warm.py` cannot even produce a cold arm here, because a
cold Tex cache needs a working LaTeX run and `latex` fails ("did not produce a
log file") from a redirected cache directory. Back up `~/.algan/cache/taichi`,
`rm -rf` only that, restore afterwards — or, better, point
`TI_OFFLINE_CACHE_FILE_PATH` at a scratch directory and leave the real cache
alone entirely.

**Any harness that renders the six full-render scenes must diff its own render
against the committed baseline before reporting a number.** Reproducing the
suite's settings is not enough: the scenes name `Algan Test Sans` and
`tests/conftest.py` registers the vendored faces, so a script that skips it
measures a scene 205-232 channel values away from the one it names.
`_notch_scene_check.py` loads `tests/conftest.py` by path for this reason and
prints the baseline diff beside every table.

**A probe pixel is (batch-relative pixel, absolute time_start) — and
`time_start` RESTARTS per render segment.** `animate_fade_out=True` renders
extra segments that number from 0 again, so `time_start + pix // ppf` is not a
video frame index and a frame-range comparison built on it is meaningless. This
produced a confident "the moved pixels and the truncated pixels are disjoint"
that had to be withdrawn. `--batch-frames` prints the mapping; prefer COUNTS,
which are segment-independent, over frame identities.


================================================================================
B. THE CPU BASELINE SETS ARE STALE, AND THIS MACHINE CANNOT WRITE THEM
================================================================================
**STATUS: open, and deliberately left open.**

`tests/fast/expected_outputs_cpu/` and `tests/full_renders/expected_outputs_cpu/`
were written at `2d1432a`, before several gates flipped. This machine's own CPU
render of `tests/fast` misses the committed CPU baseline by 30 channel values on
43 of 45 frames *before any change*, so regenerating from here would replace a
baseline CI reproduces with one it does not.

**What this session's changes did to the debt.** §C.4 and §D flipped two
defaults that move rendered output. `tests/fast` **still passes unchanged**, so
the fast CPU baseline is not affected. `tests/full_renders`' three moved scenes
(`materials_and_lighting`, `solids_and_camera`, `text_and_media`) had their
**CUDA** baselines regenerated and reviewed; their **CPU** baselines are now
additionally stale. `test_full_render_scene` skips itself when `CI` is set, so
this does not turn CI red, but the CPU set no longer describes the renderer.

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


================================================================================
C. THE RUN-SCAN LIMIT — largely closed
================================================================================

**The mechanism.** `_aa_run_scan` (`raster_taichi.py`) walks at most
`_AA_MAX_RUN_SCAN = 16` consecutive fragments of one sheet, returning the exact-
area sum `E`, the sample union `U`, the run's end, and (new) whether it stopped
for BUDGET rather than because the run ended. On a longer run the first three
are short, and the rule then uses them as though they were complete:
`corr = min(E, 1) / Q` where `Q = popcount(U)/N`.

**Measured on the six full-render scenes** (`_notch_scene_check.py`, CUDA,
`PREVIEW`, every frame, each render verified byte-identical to its baseline
first). The first-run columns are a LOWER bound; `--all-runs` is the matching
upper one, and the two together are what bound a population:

    scene                    truncated   full-mask arm      partial-mask arm   all-runs
                                          n     mean          n        mean    trunc px
    text_and_media             420,552  106,283  0.0849   314,072    0.2762    420,572
    solids_and_camera            7,008    2,180  0.0302     4,335    0.2088     11,174
    complex_hierarchy_become        49        3  0.0275        42    0.0825         62
    shapes_and_timeline            107        0       -        40    0.0647        197
    materials_and_lighting           0        -       -         -         -          0
    manim_compat_and_plots           0        -       -         -         -          0

Means are coverage lost per pixel. The **partial arm is the larger half in both
population and magnitude**. `materials_and_lighting` is clean because its
longest run is exactly 16 — 0 truncated runs of 4,740,970 scanned, in the whole
video, which makes it the control scene for anything claiming to fix truncation.

C.1 Reuse `frag_cap` on the full-mask arm  [SHIPPED, default ON]
------------------------------------------------------------------
`ALGAN_ANALYTIC_AA_RUN_CAP`, `aa_grp = 5`. See `DESIGN_mesh_identity.md` §6.8.

On the full-mask arm `Q == 1`, so `corr = min(E, 1)` — the mesh's total claim
over the pixel, which the one-mesh reduction has already packed into `frag_cap`
and the walk already loads. Where the scan ran out of budget and a cap exists,
take it. No new lane, no new argument, no new host work.

**Scored against `corr(unbounded run sum)` as the ideal, with the gate ON so the
probe scores the rule the walk now runs**, per truncated full-arm pixel:

    scene                cap available   err on capped px   corr>1 introduced
    text_and_media       89,117/106,283       0.0000             0
    solids_and_camera     2,259/  2,661       0.0001             0
    complex_hierarchy         2/      3       0.0005             0
    shapes_and_timeline      45/     67       0.0000             0

Exact on the scene carrying most of the defect, and `corr > 1` is impossible on
this arm since `Q` is 1. It cannot reach the ~15% of truncated pixels holding
more than one mesh, and it does not touch the partial arm.

**Two things it needed that were not obvious.** `_aa_run_scan` had to report
budget truncation, and "the loop stopped with fragments left" is the WRONG test
— a run of exactly 16 that ends of its own accord looks identical. It probes one
fragment further with the loop's own three terminators and accumulates nothing.
That matters: `materials_and_lighting`'s longest run is exactly 16, so the naive
test would have replaced an EXACT `E` with an estimate on every one of them.

**Output.** Moves `solids_and_camera` (54 channel values) and `text_and_media`
(49). Reviewed with `_diff_frame.py`: fine interior speckle on diced and
textured surfaces, no silhouette or structural change. CUDA baselines
regenerated.

C.2 Refuse to consult `E` on a truncated scan  [DECIDED AGAINST — do not build]
-------------------------------------------------------------------------------
Falls back to `corr = 1` when the scan hit its budget. It hands back the relaxed
gate's win on long-run silhouette pixels, and the interior share of truncated
full-mask pixels is **95%** in `text_and_media` and 63% in `solids_and_camera`.
C.1 fixes the same arm exactly and gives up nothing.

C.3 Exact run totals from a host segment reduction  [BUILT, DEFAULT OFF]
-------------------------------------------------------------------------
`ALGAN_ANALYTIC_AA_RUN_EXACT`, `aa_grp = 6`. `DESIGN_mesh_identity.md` §6.7.

It takes `E`, `U` and the run's extent from a host segment reduction over the
CSR. It is the only one of the three that fixes `U` as well as `E`, which is
what the partial arm needs.

**Three of the four things standing between it and a default are now done.**

1. **The frames were looked at.** The moved pixels are the dense point-cloud of
   small spheres in `shapes_and_timeline` and the diced interiors elsewhere.
2. **The `E` precision question is DECIDED: keep float64.** The host sums in
   float64 and rounds to f32; the kernel sums f32 sequentially. `--precision`
   now measures the gap the lane check could not see (that check compares host
   to host): on `materials_and_lighting`, where nothing is truncated, the two
   differ by **1.79e-07** — one ulp — over 4,740,970 scanned run starts, with
   **0** dust-band verdict flips and **0** pixels where `corr` moves by more
   than 1e-4. Matching f32 sequential summation on the host would need a
   segmented f32 scan, i.e. `cumsum`, which is not reproducible on CUDA — it
   would reintroduce exactly the non-determinism §6.6.4 paid to remove, and on
   the quantity that feeds the discrete decisions. Accuracy and reproducibility
   both point the same way.
3. **The arm is now CONFINED to truncated runs.** It reads the lanes only where
   `_aa_run_scan` reported budget truncation, and keeps the kernel's own sum on
   every complete run. Unconditional reads moved `materials_and_lighting` by 42
   channel values over 28,854 pixels — 16% of a frame — on a video with **no
   truncated run at all**; confined, that scene is **byte-identical**. This is
   the change that makes the arm's output move describable as "the population it
   fixes" rather than "everything".
4. **Cost** is still unmeasured. Two lanes per fragment (8 B) against a bounded
   loop that is now KEPT (the confinement needs the scan's truncation flag), so
   the "deletes the loop" speed argument no longer applies.

**THE BLOCKER IS ATTRIBUTED, AND THE ARM IS INNOCENT** (measured 2026-08-19,
`DESIGN_mesh_identity.md` §6.7.3). The "31 channel values over 4,514 pixels,
~37,000 pixel-frames over 12 frames" reproduces byte-for-byte at HEAD — and
re-rendering BOTH arms under a lossless codec (`codec="libx264rgb"`,
`-crf 0`) collapses it to **18 pixel-frames: 1–3 pixels per frame, worst
|d| 21**, a tight cluster on the `PointCloudDot` ring's midline, where
overlapping same-sid dots fuse into the scene's only long runs. Those are
exactly the engaged truncated runs the arm exists to fix, changed the way the
design intends. Everything else was the MP4: the six-scene comparisons decode
H.264 at yuv420p with I-frames 250 apart, so 1–3 real pixels per frame ride
inter-prediction, 16x16 DCT blocks and 2x2 chroma subsampling across the dot
footprints for the rest of the GOP — a ~2,000x inflation of the moved-pixel
count. The "two numbers that cannot both describe the same population" were
BOTH right: 198 truncated runs counts the renderer's change, ~37,000
pixel-frames counts the codec's rendering of it.

The move runs frames 286–297 because ENGAGEMENT, not run length, gates the
population: the fade-out is a staggered despawn, a despawned dot's fragments
stay in the CSR with zero alpha, and a zero-alpha write leaves `svis` exactly
uniform — so only once the first dot dies can the walk's run gate reach a deep
truncated run at all, and once every dot is dead there is nothing left to
change. The truncated-run census is near-uniform over the whole fade-out
(2–12 per frame); the ENGAGED subset is the 1–3 per frame the lossless diff
shows.

**Both of the previous revision's suspects are dead, measured:**

* *The dense path.* Every batch of this scene takes the SPARSE path — both
  arms, whole render, `raster_iteration_zero` never launches
  (`_c3_fadeout_ab.py` logs every batch). §6.7.1's "its fade-out segment
  takes the dense path" was an inference and is retracted in the record; the
  `_aa_group_dense` cap stays, as defense for any batch that IS dense at
  level 6, with `test_analytic_aa_gates.py` still pinning it.
* *A run crossing a `shade_sparse_raster_coverage` slice boundary.*
  Impossible by construction — the host reduction breaks runs at pixel
  changes (`raster_pipeline.py`, `starts` includes `pix_s` changes) and the
  slices are pixel-aligned views of `run_offsets` — and the slicing levers
  are byte-inert: halving `WAVEFRONT_TILE_RAYS` is byte-identical
  (`_c3_tile_ab.py`), as is building the lanes with the kernel pinned to the
  shipped variant.

The elimination chain, for the next investigation of this shape
(`benchmarks/_c3_*.py`, every arm diffed against an arm-OFF render that is
itself byte-identical to the committed CUDA baseline): per-batch path log +
truncated-run census (moved DECODED pixels are overwhelmingly NOT the
truncated population), lanes-built-but-kernel-pinned (byte-identical → the
lanes and their arena footprint are innocent), a one-pixel `ALGAN_AA_DUMP`
golden walk under both arms (the worst decoded pixel's walk is IDENTICAL in
both), post-processing off (identical diff — bloom is a no-op without glow),
and only then the codec.

C.4 What ships
---------------
**C.1 ships on. C.3 is attributed and stays off only for the flip work.** They
are not in competition — C.1 is a special case of what C.3 does generally, and
when C.3 flips, C.1 becomes inert under it (the kernel already compiles it out:
`_aa_run_cap and not _aa_run_exact`). §Y.1's "show the mechanism before
flipping" is now satisfied; what the flip still needs:

1. **Cost** (item 4 above): 8 B/fragment of lanes plus the reduction, against
   a bounded loop that stays. Wall-clock is not measurable on this machine
   (§A); the lanes are already accounted in `discovery_bytes`.
2. **Baselines.** The scenes carrying truncated runs (`text_and_media` above
   all — 420,552 truncated pixels, and the partial-mask arm only this fixes)
   move in exactly those populations. Regenerate CUDA baselines on this box,
   diff each scene under the LOSSLESS codec first so what is reviewed is the
   renderer's change and not the encoder's spread of it, look at the frames,
   and say in the commit that the CPU debt (§B) grows by the same scenes.

**AND THE MECHANISM CHOICE ITSELF IS REOPENED** by two of §6.7.3's
measurements, so decide it before doing the flip work:

* §0.5's **option (a) — raise `_AA_MAX_RUN_SCAN`** (to
  `MAX_SURFACES_PER_RAY`, which bounds any run the walk can see) — was closed
  on "(a) is a re-baseline, not a patch". That case was codec-inflated, and
  (a) has properties §6.7 cannot have: complete runs keep byte-for-byte the
  SHIPPED arithmetic (no confinement machinery needed, no host/kernel second
  language — rule Y.4 by construction), it reaches the DENSE path (the lanes
  never can: they are a torch-side reduction over the sparse emission's CSR,
  and the dense path's fragment lists exist only inside the per-tile
  kernels), it deletes the lanes, the reduction and 8 B/fragment, and its
  oracle is itself. Cost: each fragment of a >16 run is read by the scan once
  more than today — unmeasured, and this box cannot measure it (§A). §0.5's
  open obligation stands: re-run `_aa_line_check` / `_aa_run_gate_check` in
  the raised arm before believing the new frames are better.
* The **unconditional-lanes** variant ("read the lanes everywhere, drop the
  confinement, re-baseline") is also cheaper than §6.7.2 believed: its
  collateral on `materials_and_lighting` is really 2,063 pixel-frames at
  worst |d| 4, not 42-over-16%-of-a-frame. But it is dominated by (a): same
  intended fix, plus ulp-flip sprinkles (a) does not have, and it keeps the
  lanes while still missing the dense path.

Before spending anything on the dense path, show it matters: the six-scene
suite renders sparse throughout (§6.7.3), the dense resolve is the fallback
(env maps, non-default tonemap, `use_raster` rejections), and no instrument
has ever counted truncated runs there (§Y's "an instrument that reports zero
may not be looking").


================================================================================
D. §3.1's WELD — SHIPPED
================================================================================
**STATUS: default ON (`ALGAN_WELD_SURFACE_SEAMS`).**

`get_grid_to_triangle_indices` bridges a closed surface's wrap column back to
column 0 and collapses a pole fan to one vertex. Both original objections were
already measured away (textured/normal-mapped byte-identical on a static frame;
the morph path asks `surface_weld_flags` for the same grid the render path
does). The setting's own comment still described the morph-path blocker as open
— it was stale, and is corrected.

What remained was that it moves a moving PN scene, because the dice level is
chosen per patch per frame from projected size, so a different triangle list can
land on a different level. Done: `materials_and_lighting` (19 channel values on
its worst frame, concentrated in the bloom halo around the two glowing spheres —
the amplified-epsilon pattern, visually identical) and `solids_and_camera` (33,
interior speckle on the diced solids). CUDA baselines regenerated, frames
reviewed, unit suite green.


================================================================================
E. A TRAVERSAL-STEP COUNTER — BUILT
================================================================================
**STATUS: done. `benchmarks/_bvh_steps.py`.**

Counts, per ray, sibling-block tests (`_group_test` — the traversal step), leaf
slots reached, and primitive intersections actually performed, for one geometry
type's STBVH. Deterministic, which is what a machine that cannot do wall-clock
needs.

**How it is kept honest.** The walk is a transcription of
`_nearest_triangle_hit` / `_nearest_bezier_hit` with counters added — same
`_group_test`, same `_nearest_pending_child`, same `best_t` pruning, and the
leaf bodies call the same production `@ti.func`s. `--verify` runs the production
function over the identical rays and arrays and compares the hit: same
`(t, prim)` for every ray means the same nodes were visited, because the only
things deciding which node a ray enters are the block data, the ray, and the
sequence of `best_t` updates. Measured: **51,200/51,200 rays agree, worst
|dt| 0**, in every arm run so far.

**And it reproduces a number nobody gave it.** The triangle tree's median split
was independently known to be ~25% faster to traverse; the counter says 7.944 →
5.960 groups per ray, 25%. That is what says it is measuring the tree.

**Two traps it walked into first, both worth knowing.**

* **A deferred BVH is a placeholder** — 1 block, 1 leaf slot, no live primitive,
  an all-zero root — when the batch provably never traverses one. The counter
  reported 1 group per ray and no hits, and AGREED with the production walk,
  which finds nothing either. Both harness scenes now carry a metallic sphere
  purely so a secondary ray exists and the tree is really built.
* **`BVH_REFIT` defaults ON**, and a `RefitBVH` stores its topology as
  per-(frame, child) link words with the leaf-slot arrays unused. The counter
  transcribes the `refit == 0` walk, so pointed at one it reports the same
  nothing. It now refuses to run and tells you to pass `ALGAN_BVH_REFIT=0`.

**It does NOT settle §3.2's cost**: §3.2 changes the ray/triangle intersection
TEST, not which nodes are visited, so a step count is identical across its arms
by construction.


================================================================================
F. MEDIAN-SPLIT BEZIER BVH — measured, default flipped, and mostly moot
================================================================================
**STATUS: `ALGAN_BEZ_BVH_SPLIT` default ON. The inherited claim is confirmed.
The flag is INERT at shipped defaults, which is the more important finding.**

With `ALGAN_BVH_REFIT=0`, on 35 circuits plus `Text` and `Tex`:

    ordering        groups/ray (primary)   groups/ray (incoherent)
    morton                 3.300                  3.159
    median split           2.302                  2.219

A **30% reduction** in traversal steps, slightly better than the inherited
"~20-25%", and the same on incoherent rays — so it is not an artifact of
primary-ray coherence. Leaf slots and primitive tests are identical to four
decimals in both arms, and every one of 51,200 rays returns the same primitive,
so the reorder changes cost and not answers. `_order_window_check.py` confirms
byte-identical rendered output against a zero noise floor.

**Why `_bez_bvh_ab.py` found nothing.** `BVH_REFIT` defaults ON and
`_build_accel`'s refit branch ignores `builder` outright, so at shipped defaults
**no STBVH is built for any geometry type** and `ALGAN_BEZ_BVH_SPLIT` — like
`ALGAN_BVH_BUILD` — changes nothing. That harness was A/B-ing one render against
itself, which is exactly why it measured byte-identity at wall 0.993x. Any
future comparison of instance ORDER must set `ALGAN_BVH_REFIT=0`, and
`_order_window_check.py`'s order arms now do.

The default is flipped anyway: it is strictly better on the tree it governs,
byte-identical, and costs nothing today.


================================================================================
G. TWO-LEVEL BVH (TLAS/BLAS) — out of scope for this queue
================================================================================
**STATUS: not started, deliberately. Untouched this round; see the previous
revision's reasoning, which still holds** — the BVH build is ~1% of a shadowed
five-solid render, so the amortization half of the argument is dead, and no
workload in the repo has thousands of repeated meshes to show the instancing
half on. §E now exists, which removes one of the two preconditions.


================================================================================
H. NESTED-IOR REFRACTION — designed to the argument list, NOT BUILT
================================================================================
**STATUS: not started. The design below is new work; the implementation is not.**

`wavefront_kernels_taichi.py` treats a circuit as a thin pane (`is_pane`) and a
triangle mesh as a solid (`is_glass`), and assumes air outside every interface.
A ray should instead carry what it is *inside* and take the correct RELATIVE
index at each interface — glass inside glass, a sphere inside a box.

**The design changed in one important way, and it makes the item much cheaper.**
The previous revision proposed a depth-N stack of MESH IDS. That needs
`tri_obj` inside the shade kernel, and `wavefront_shade` already takes **37
ndarray arguments** against a ceiling this project has only tested to 36 (memory
note "Taichi kernel arg headroom"; the CUDA cap is 64 but the practical limit
here is lower). Carry a stack of **IORs** instead:

* It costs **no new kernel argument**. `rs_sca`'s width is already the parameter
  `sca_width` of `_alloc_wavefront_state`, so the stack rides in columns 7+ and
  the gate is a `ti.template()`, not an ndarray.
* It needs no lookup: the id alone cannot give you an index of refraction
  without `tri_obj` *and* a per-mesh IOR table.
* **`_refract_ray` does not change at all.** It reads its `ior` argument as
  n_inside/n_outside and picks the side from `sign(rd · n)`. Passing the
  RELATIVE index is therefore exactly right in both directions — entering gives
  `eta = n_outside/n_inside`, exiting gives `eta = n_inside/n_outside`.

**The shape of it.**

    rs_sca[r, 7]        stack depth, as a float (exact well past any depth)
    rs_sca[r, 8 .. 8+N-1]   the IOR of each medium the ray is inside, outermost
                            first. N = 4 covers everything a scene here builds.

At an `is_glass` hit, with `entering = (rd · face_n) < 0`:

    entering:  n_out = depth ? stack[depth-1] : 1.0;  rel = ior / n_out;  push ior
    exiting:   n_in  = depth ? stack[depth-1] : ior;  pop
               n_out = depth ? stack[depth-1] : 1.0;  rel = n_in / n_out

then `_refract_ray(rd, normal, rel)` unchanged, and copy the updated stack into
the continuation's `rs_sca[c, 7..]` beside the six values already written there.
Overflow at depth N must fall back to today's behaviour (treat the outside as
air) rather than corrupt.

**What does NOT need touching, and why.** A reflection stays in the same medium
and a coverage-miss never crossed an interface, so the primary ray's stack is
correct by doing nothing — it keeps its own slot. Only the refracted
continuation carries a modified stack.

**Watch for**: the refraction path forces the wavefront tracer, and pool sizing
there is measured, not modelled — raise the POOL, not `pool_ratio`, if a first
attempt overflows (`pool_ratio` scales tile count). Widening `rs_sca` also moves
the arena fit, so `test_render_batch_sizing.py` and `test_memory_model.py` want
re-checking; gate the width so it only grows when the feature is on.

**Done when** a sphere inside a box renders with the correct relative IOR at the
inner interface, and the existing glass scenes are unchanged with the gate off.


================================================================================
I. SELF-SHADOW REJECTION BY IDENTITY — costed, NOT BUILT
================================================================================
**STATUS: not started. The plumbing cost below is new; the implementation is
not.**

A shadow ray currently rejects its own surface with `MIN_HIT_DISTANCE = 1e-4`
(`raytrace_kernels_taichi.py:95`) plus a normal offset of `10 *
MIN_HIT_DISTANCE`. Both are absolute world-space constants with no scene-scale
adaptation, and they are what produces shadow acne at grazing light angles and
on small-scale geometry. Worse, the acceptance epsilon applies to EVERY hit, so
a small object resting on a plane loses its contact shadow within 1e-4 of the
contact.

With mesh identity the test becomes "reject a hit on the mesh this ray started
from, at near-zero `t`" — which lets the cross-mesh threshold go to zero and
keeps the self-rejection exactly as safe as today:

    accept = (t < max_t) and (hit_mesh != src_mesh ? t > 0 : t > MIN_HIT_DISTANCE)

**Care needed:** rejecting the whole *mesh* is wrong for a concave solid that
legitimately shadows itself. The rejection has to be "the same mesh AND
near-zero `t`", not "the same mesh".

**What it costs, measured rather than guessed.** The source mesh id is available
where the shadow event is BUILT (`raster_shadow_event_build` reads `tri_obj`)
and not where it is TRACED. So:

* One new ndarray on `raster_shadow_trace` for `tri_obj` (it currently takes
  ~28, so this is within headroom). The source id itself needs **no** new array:
  `event_msk` uses only its low 4 bits for sub-pixel masks, leaving 28 bits.
* Threading `(src_sid, tri_obj)` through five `@ti.func` signatures —
  `_shadow_occluded` → `_shadow_anyhit_opaque` / `_shadow_march_occluded` /
  `_shadow_gather_occluded` → `_anyhit_opaque_tri` / `_nearest_surface_g` →
  `_nearest_triangle_hit` — which are shared with the megakernel, so its call
  sites move too.

That is the whole of the work, and it is why this is a session of its own rather
than a corner of one.

**Done when** a grazing-light scene shows no acne, `tests/full_renders`'
shadowed scene is unchanged or reviewed, and the constant is gone from the
cross-mesh shadow path.


================================================================================
J. ORDER- AND WINDOW-INDEPENDENT OUTPUT — DEMONSTRATED
================================================================================
**STATUS: done. `benchmarks/_order_window_check.py`.**

The property asked for was that resolution be a function of the canonically
sorted hit list alone — independent of KBUF width, BVH builder, tile size and
batch window. It was claimed but never shown. On a scene built for it (six
translucent sheets in front of a glass sphere, a mirror and three solids), at
`--res ld`:

    lever                            batches      worst |d|
    run-to-run noise floor            7 -> 7          0
    KBUF 1                            7 -> 6          0
    KBUF 8                            7 -> 7          0
    Morton order (STBVH)              7 -> 7          0
    median-split order (STBVH)        7 -> 7          0
    16x more wavefront tiles          7 -> 3          0
    a third of the batch memory       7 -> 22         0
    the same, on a static scene       1 -> 1          0

**Byte-identical on every lever, against a noise floor of zero** — so that is
real byte-identity, not a diff hidden under jitter. The batch counts are printed
because they are what says the lever was reached: the window arm really did
re-window, all the way to one frame per batch, and still did not move a pixel.
That is stronger than the `<= 2` the repo's re-windowing note would allow.

**Two things the check itself found.**

* The BVH-order leg is **vacuous unless `ALGAN_BVH_REFIT=0`** (§F). It now sets
  it, and compares against a `refit_off` reference rather than the default one.
* **At `--res md` the same scene is not reproducible run to run at all**: a
  noise floor of 46 channel values over 212,210 pixels, confined to the edges of
  the six-deep translucent stack and the glass sphere. That is far above the
  |d| = 1 cap of known split-pixel nondeterminism, and it is a separate defect
  from anything §J is about — it is not the analytic-AA resolve, and nothing in
  this queue covers it. Worth its own item. The harness prints a warning and
  points at `--res ld` when the floor is non-zero.


================================================================================
K. THE TWO `ti.static` ARMS — DECIDED: KEEP
================================================================================
**STATUS: closed as a decision. Revisit only if the per-ray f32 is needed.**

There is no live `BARYCENTRIC_EPSILON` read at shipped defaults. All three
survivors sit in arms that do not compile in: `_tri_hit`'s Möller-Trumbore arm
(live only with `WATERTIGHT_TRI` off, and it now defaults ON), and two
`raster_taichi` sites live only at `aa == 0`.

**Keep both.** The reasoning, with what this round added:

* **They cost nothing at defaults.** `aa == 0` is a whole-batch compile-time
  value — no pixel ever falls back to it at runtime — and at `aa != 0` both
  raster sites compute their epsilon verdict into a variable the very next
  `ti.static(aa)` block overwrites. `WATERTIGHT_TRI` compiles the Möller arm out
  entirely. Deleting them buys no instructions.
* **They are the only A/B controls for their measurements**, on a machine where
  controls are the scarce thing (§A).
* **The Möller arm has no remaining future as a candidate default**, which is
  new: `_watertight_check.py` run on both arms gives **identical** results — 0
  cracks on grazing quads and on a diced Sphere in each, and the same ridge
  counts (114 / 0 / 0 at alpha 0.35 / 0.6 / 0.85, max deviation 9.0). The
  dilation buys nothing. That settles what the arm is FOR: it is a control, and
  a control is exactly the thing you keep as a compiled-out branch.
* **What deleting would buy is not free either.** The constants, the `edge_hit`
  bit, and `seam_t` — `rs_sca[r, 3]`, one f32 per ray — and only once BOTH arms
  have gone. That shrinks `rs_sca`, which moves the arena fit, so
  `test_render_batch_sizing.py`, `test_memory_model.py` and a long multi-batch
  render's OOM-retry count all have to be re-checked.

§3.2's *cost* remains unmeasured, and `_watertight_check.py` never measured it
despite its docstring saying so (corrected). The decision above is written not
to depend on it.


================================================================================
L. EXACT ABSORPTION OF COINCIDENT DUPLICATES — unlocked, unbuilt, unmeasured
================================================================================
**STATUS: not started, and nobody has shown the symptom.** Untouched this round.

A union of sample masks is idempotent, so two genuinely coplanar stacked quads
should stop double-darkening once coverage is taken per mesh rather than per
fragment. **Before building anything, show the defect.** Render two coincident
quads of the same mesh and of different meshes and measure the darkening. If the
per-sample transmittance walk already handles it — which is plausible, since
masks partition within a sheet — this item closes with a test rather than a
change.


================================================================================
Y. SEVEN RULES THIS SUBSYSTEM KEEPS RE-LEARNING
================================================================================
Each cost a wrong result that is recorded in `DESIGN_mesh_identity.md`.

1. **A check must show it REACHES its case.** "All three modes agree" was
   produced four times from scenes with no shadow in them. This round: a step
   counter that agreed with the production walk on all 51,200 rays, on a
   placeholder BVH where both found nothing; and an order-independence check
   whose BVH leg compared one render against itself.
2. **Replay the same inputs with exactly ONE thing changed.** That is what
   attributed the notches three times running, what verified §6.7's host
   reduction without a kernel compile, and what showed re-windowing was NOT
   §6.7's mechanism (`--verify-lanes`: same 32 batches, shipped rule,
   byte-identical).
3. **Read which accumulator a metric scores.** An occlusion-side fix was
   predicted to close claim-side symptoms. It could not have.
4. **A question asked in two languages needs one answer.** The host/kernel
   boundary is where this codebase keeps finding second ones — `aa_grp` drifted
   once and cost most of a win; the one-mesh flag read a different frame's
   surface map than the kernel did; and §6.7.1's dense path read run lanes it was
   handed as one-element dummies, because the gate was a group value that did not
   know which path had launched it.
5. **Look at the frames before re-baselining.** `_diff_frame.py` exists for
   this. "Measured 42 channel values" is not knowing what moved.
6. **A COUNT is segment-independent; a frame INDEX is not.** `time_start`
   restarts per render segment, so anything mapping a probe pixel to a video
   frame is wrong across a fade-out. Argue from populations.
7. **A pixel diff read from an MP4 measures the ENCODER's output, not the
   renderer's.** The suites decode H.264 at yuv420p with I-frames 250 apart,
   so inter-prediction, DCT blocks and chroma subsampling spread a real change
   into pixels the renderer never touched: §C.3's 18 lossless pixel-frames
   decoded as ~37,000 (§6.7.3). Byte-identity claims survive (identical raw
   frames encode identically); every "moved by N over M pixels" claim is
   codec-inflated. Before attributing a move's SIZE or SHAPE, re-render both
   arms with `codec="libx264rgb"`, `ffmpeg_params=["-crf", "0"]`.

And one specific to this file's own instruments: **a metric that reports zero
may be an instrument that is not looking.** `_notch_scene_check.py` scored only
a pixel's first run and reported zero for `shapes_and_timeline`; it scored only
the full-mask arm, which is the smaller half of the defect; its lane check
compares host to host and so could not see the host/kernel gap at all
(`--precision` now does); and it hooks only the sparse emission, so **every
number it prints is blind to dense-path batches** — which is where §6.7.1's bug
lived and where §C.3's unexplained residual may yet live.


================================================================================
Z. CLOSED — do not rebuild these
================================================================================

* **Material dispatch coherence by sorting hits.** Built and measured **1.5-2.2x
  SLOWER** than the monolithic scatter (`_wf_sorted_ab.py`,
  `_wf_monolith_scatter_ab.py`). `WAVEFRONT_SORT_MATERIALS` defaults off and the
  sorted route is unsupported, kept for reference.
* **Suppressing the far sheet** instead of capping it (+114% on a sub-pixel
  diced rod). Two follow-up hypotheses — scrambled facing bits, and the u-seam —
  were both tested and refuted.
* **Regrouping the run into an order-independent equivalence class**: `split` is
  ~0.02% everywhere, so there is nothing there.
* **Consulting `E` only inside the existing gate**, and **buying more samples**:
  both measured, neither is the lever.
* **Raising `_AA_MAX_RUN_SCAN` as the fix** (§C's option (a)). §6.7 does the same
  thing correctly and cheaper.
* **Carrying the fragment's own-facing sum in a new lane** rather than reusing
  `frag_cap`: scored, matches to the fourth decimal, does not earn the lane.
* **A traversal-step counter to settle §3.2**: it cannot, by construction. §E
  built the counter anyway, for §F and §G, and says so in its own docstring.
* **The glyph geometry cache as the explanation for §B.2's moved pixels.** Tested
  and refuted — but note that a genuinely cold arm cannot be produced on this
  machine at all (§A), so the refutation is by population counts, not by a
  cold/warm diff.
* **The shadow-event path as the amplifier of §6.7's ulp.** `materials_and_lighting`
  rendered with `shadows=False` still moves by 42 channel values under the
  unconditional arm. Refuted; the confinement in §C.3 removed the move instead.
