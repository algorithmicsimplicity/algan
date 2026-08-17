# Algan — Mesh Identity in the Triangle Renderer

Status: PARTLY LANDED. This is the plan of record for replacing the renderer's
epsilon-based seam heuristics with declared mesh identity. §1–§3 are the problem
and what shipped; §4 is what must be verified on a CUDA device and the experiment
for each claim; §5 is what the new system enables; §6 is the measured negative
results, kept so nobody spends the effort twice.

Written on a machine with no GPU. Everything claimed here was measured on the CPU
render device unless it says otherwise, and §4 exists because that is not enough.


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
and leave a crack (`raytrace_kernels_taichi.py:121-128`). Dilation and dedup are
a matched pair; neither means anything alone.

What it costs:

* **Replicated in 8 kernels**, with 8 initialisations, 5 bounce resets and a
  dedicated per-ray state slot (`rs_sca[r, 3]`).
* **Not an equivalence relation.** The depth *order* was already fixed by binning
  (`_comes_after`), but the dedup is a greedy window against a mutable `seam_t`,
  so it chains and is asymmetric.
* **It makes output depend on discovery order, and that blocks real work.**
  `stbvh.py` keeps bezier circuits on the slower Morton builder — triangles get
  the ~20-25% faster median split — purely "to preserve baselines".
* The epsilons are absolute world-space constants with no scene-scale adaptation.


================================================================================
2. WHAT LANDED
================================================================================

2.1 The curved PN-patch renderer is deleted
-------------------------------------------
`RENDERER_REGISTRY.triangle_primitive` is never rebound from
`RayTracedTrianglePrimitive` (`settings/renderer_settings.py:17`), so
`RayTracedPNTrianglePrimitive` was unreachable through the public API and
`merged["num_pn"]` was always 0. `Surface` — and therefore
`Sphere`/`Cylinder`/`Cone`/`Torus` — reaches the renderer as *logical PN* patches
diced to flat triangles before the tracer or the STBVH sees them
(`algan/rendering/logical_pn.py`), crack-free by construction because adjacent
patches derive **bit-identical** shared boundary vertices.

Removed: the class; `pn_patch.py` in full; `_pn_intersect` and its cubic solver;
`_nearest_pn_hit`; `_obb_misses`; `_pn_normal`; `_shade_pn_hit`;
`_anyhit_opaque_pn`; the wavefront's five `_pn_hit_*` helpers; every `htype == 2`
branch; the `pn_*` merged keys and their two STBVH builds per batch (six trees →
four); and four epsilons (`PN_BARYCENTRIC_EPSILON`, `PN_EDGE_EPSILON`,
`PN_DEDUP_UV_EPSILON`, `PN_SEAM_DEPTH_EPSILON`). `seam_eps` collapses to
`DEPTH_TIE_EPSILON` at all five sites that selected it. ~2800 lines net.

**Output is byte-identical**, verified per-frame and per-video, with the merged
scene tensors and derived render flags hashed equal and the batch windows
unchanged.

One trap worth recording. The packed `layer_offsets` array loses its PN slot, so
it renumbers from eight entries to seven — and it has **three** consumers, not
two. Renumbering only the two obvious reads in `wavefront_kernels_taichi` left
`raster_first_shade` (`raster_taichi.py`) reading `max_bounces` off the end of the
array, which silently changed every PBR reflection in the fast-suite render by up
to 162 channel values while single-frame renders stayed identical. Grep
`layer_offsets\[` before touching that array.

2.2 Mob-declared surface identity, gated off
--------------------------------------------
`tri_obj` is what the analytic-AA resolve groups fragments by, and its
granularity was one id per merged **collection member** — right only when one
member is one surface, which is wrong at both ends:

* `Polyhedron` hands the batcher one member per **triangle**, so a `Cube` was
  twelve surfaces and no run could span a face diagonal.
* A packed-grid `Surface` hands it one member covering **every** packed sphere, so
  distinct spheres were unioned and their coverage summed across objects that
  merely overlap.

Mobs now declare identity on the primitive they build, resolved by
`primitives._mesh_ids_from_collection`:

    mesh_key   merge with the consecutive neighbours sharing it (matched against
               the preceding member only, so identity cannot leak across an
               unrelated mob that happens to sit between two halves of one)
    mesh_ids   subdivide one member into per-triangle shells; needs no contiguity

Declared by `Polyhedron` (one solid), packed-grid `Surface` (one shell per grid),
and `TriangleMesh`, whose `corner_index` already carried the loader's topology and
was only kept for smooth normals — `triangle_shell_ids()` walks it for
**edge**-connected components via scipy, so an imported file's disconnected parts
stop being one surface. Edge- not vertex-connectivity, which would fuse two cones
meeting at an apex. Deliberately **not** `Arrow3D`: its children are separate
interpenetrating solids, not one mesh.

`ALGAN_MESH_ID` **defaults off**, so this is byte-identical. See §6.2 for why.

2.3 A measurement harness
-------------------------
`benchmarks/_aa_run_gate_check.py` replays the analytic-AA run rule's grouping and
magnitude decisions on the host for every covered pixel, so questions about it
get a population statistic instead of one dumped pixel. It is what produced §6.

2.4 First tests for `tri_obj`
-----------------------------
`tests/unit_tests/test_mesh_identity.py`. Nothing tested `tri_obj` in any suite
before. Pure tensor assertions, no render, so the end-to-end cases are in the fast
suite.


================================================================================
3. WHAT HAS NOT LANDED
================================================================================

In the order the plan plans them, with the state of each:

| # | Item | State |
| --- | --- | --- |
| 3a | Weld the `Sphere` u-seam and collapse the pole fans | not started |
| 3b | Watertight ray/triangle test (Woop–Benthin–Wald), gated | not started |
| 3c | Median-split STBVH for bezier circuits, gated | not started |
| 3d | Delete `BARYCENTRIC_EPSILON` / `edge_hit` / `seam_t` and the 8 seam sites | blocked on 3b |
| 3e | Flip `ALGAN_MESH_ID` on | blocked on §6.3 |

3a is the prerequisite for 3b. A `Sphere`'s u-wraparound is a genuine two-copy
seam: `get_grid_to_triangle_indices` (`surface.py:213-249`) never bridges column
`W-1` back to column 0, so the two copies sit ~1.75e-7 world units apart
(measured), and the poles are collapsed degenerate fans. Interior shared edges,
by contrast, are bit-identical duplicates — the same gather. Until the wrap seam
is welded, a watertight intersection test would open a crack there rather than
close one, because the gap is real geometry rather than a numerical artifact. The
closed-seam predicate already exists (`is_closed_x`, `surface.py:364-391`), and
welding also retires two authoring-side epsilon special-cases: the 1e-4
normal-merge tolerance and the pole-normal salvage.


================================================================================
4. WHAT MUST BE VERIFIED ON A CUDA DEVICE
================================================================================

Everything below was either measured on CPU only or not measured at all. Clear
the Taichi cache (`clear_cache(taichi_kernels=True)`) before any A/B — it does not
invalidate on `@ti.func` edits. Never edit `*_taichi.py` while a render or a warm
daemon is running.

4.1 Regenerate the CUDA baselines. **Required before anything else.**
    `ALGAN_UPDATE_FAST_BASELINE=1` / `ALGAN_UPDATE_FULL_RENDER_BASELINES=1` write
    only `expected_outputs_cpu/`. Nothing here has moved output yet, so the
    committed CUDA baselines should still pass as-is — confirm that first, since
    it also validates that the PN deletion really was byte-identical on the
    device that has the other kernel variants.

        pytest -q tests/fast
        ALGAN_RUN_FULL_RENDERS=1 pytest -q tests/full_renders

    Expected: pass unchanged. A failure here is the PN deletion, not a baseline
    staleness, and the first thing to check is `layer_offsets` (§2.1).

4.2 Confirm the PN deletion is byte-identical on CUDA.
    Hash the renders against the pre-deletion commit rather than trusting the
    baselines, which is stronger:

        git stash && pytest -q tests/fast && sha256sum tests/fast/algan_outputs/fast.mp4
        git stash pop && pytest -q tests/fast && sha256sum tests/fast/algan_outputs/fast.mp4

    The six `tests/full_renders` scenes are the ones that matter: they carry the
    PN surfaces, shadows, refraction and glTF that the fast scene deliberately
    omits.

4.3 Confirm the kernels did not get slower.
    The deletion removes 12 merged keys, two BVH builds per batch and ~10
    parameters from every traverse/shade signature, so it should be neutral to
    faster. Kernel-profiler **device** times, not wall clock — thermal throttling
    swings cross-process throughput ~2x, so use in-process alternating A/B.
    `utils/profiling_utils.py` auto-hooks the kernels and pipeline stages.
    Watch `wavefront_shade`, `wavefront_traverse`, both MC megakernels and
    `raster_first_shade`, plus the per-batch BVH build time (six trees → four).

4.4 Confirm the compile surface shrank.
    Count offline-cache entries and cold-compile wall time before/after. The
    deletion removes the `has_pn` template dimension from four kernels, so the
    variant count should drop.

4.5 Qualify `ALGAN_MESH_ID=1`, which needs an arbiter this session did not have.
    Turning it on moves the fast-suite render by up to 49 channel values at solid
    edges, and `_aa_run_gate_check` reports an Icosahedron's per-fragment error
    *rising* (14.35 → 146.84) because coarser identity exposes the union-full
    short-circuit. That harness cannot say which render is better: it does not
    model the per-sample transmittance that the fine-grained ids get wrong. The
    experiment that can:

        Render a Polyhedron silhouette at anti_alias_level=8 as ground truth,
        then compare MESH_ID=0 and MESH_ID=1 at aa=1 against it. Extend
        benchmarks/_aa_line_check.py with a Polyhedron case -- it already
        compares against exact analytic coverage rather than a supersampled
        reference, which is the right standard.

    Also re-run `benchmarks/_analytic_aa_fillrule_check.py` (the sample-partition
    property) and `benchmarks/_aa_dump_check.py` (the golden host-side walk vs
    the kernel's own dump — this is what catches a resolve/shadow-event desync).
    If MESH_ID=1 wins, re-baseline **both** device sets and flip the default.

4.6 Shadow-mode agreement — a testable prediction.
    Three `SHADOW_ANYHIT` modes disagree today in corner cases documented as
    seam-merge artifacts (`raytrace_kernels_taichi.py:3346-3351`, 3570-3574).
    Once identity replaces the epsilon (3d), those disagreements should vanish:

        for m in 0 1 gather; do ALGAN_SHADOW_ANYHIT=$m pytest -q tests/full_renders; done

    Diff the three outputs. They should become identical. If they do not, the
    remaining difference is a second cause and worth isolating before 3d ships.

4.7 Watertight test (3b), once built.
    * **No cracks in f32.** Large adjacent triangles at grazing incidence plus a
      finely diced welded `Sphere` at extreme silhouette; assert zero background
      pixels interior to the mesh. Extend `_analytic_aa_fillrule_check`'s
      partition property to the ray path.
    * **No double blend.** Translucent `Sphere`/`Cylinder` at several alphas;
      interior edges must show no brightness ridge. `_aa_dump_check` on rim
      pixels.
    * **Register pressure.** The ray-space transform is per-ray hoistable but adds
      live state; check occupancy against the 21–25% resolve ceiling
      (`DESIGN_hybrid_raster.md` §13).
    * **`rs_sca` shrinks by one f32 per ray** when `seam_t` goes, which moves the
      arena fit: `test_render_batch_sizing.py`, `test_memory_model.py`, and a long
      multi-batch render checking that OOM-retry counts do not regress.

4.8 Median-split bezier BVH (3c), once built.
    `benchmarks/_split_determinism_check.py` distribution before/after, plus
    traversal-step counts and build times for the bezier tree. Byte-identity is
    the **wrong** gate: that script already documents that split pixels are not
    byte-reproducible (|d| = 1 from non-associative float `atomic_add` on
    `pix_accum`), so compare distributions.

4.9 Every gate off is byte-identical.
    The standing discipline for each new switch — `ALGAN_MESH_ID=0`,
    `ALGAN_WELD_SURFACE_SEAMS=0`, `ALGAN_WATERTIGHT_TRI=0`,
    `ALGAN_BEZ_BVH_SPLIT=0`. Verified on CPU for `ALGAN_MESH_ID`; redo on CUDA,
    where the other kernel variants live.

4.10 Render twice, baseline the second.
    The first render on a fresh machine populates the Manim Tex geometry cache and
    its `MathTex` glyph antialiasing differs from every run after it — 18 channel
    values over 100 frames of `text_and_media`, against a tolerance of 2.


================================================================================
5. WHAT THE SYSTEM ENABLES
================================================================================

5.1 Delivered
-------------
* **A much smaller renderer.** ~2800 lines, `pn_patch.py`, 12 merged keys, two
  BVH builds per batch, four epsilons, and the `has_pn` template dimension —
  gone, with output byte-identical.
* **One fewer route rejection.** `num_pn > 0` disappears from
  `analytic_raster_route_active`, `use_raster`, `_projection_anti_alias_level`,
  `_bvh_deferral_eligible` and the `WF_TEXTURED` gate as an always-true clause.
* **Correct identity for polyhedra and packed grids**, and topological shells for
  imported meshes so a glTF file's disconnected parts stop being one surface
  (gated off pending §4.5).
* **A way to ask questions about the run rule** and get population answers
  instead of anecdotes, which is what turned two plausible theories into §6.
* **`tri_obj` is now under test.**

5.2 Unlocked by the identity, worth building next
-------------------------------------------------
* **Order- and window-independent output.** Once the greedy `seam_t` rule is gone,
  resolution is a function of the canonically sorted hit list alone — independent
  of KBUF width, BVH builder, tile size and batch window. This is the property the
  rework was asked for, and the precondition for 3c and for ever reordering
  primitives in the merge.
* **Nested-IOR refraction.** A stable mesh id at every hit lets a ray carry an
  "inside which mesh" stack, so glass-inside-glass and a sphere inside a box get
  the correct *relative* IOR at each interface instead of assuming air outside.
  `wavefront_kernels_taichi.py` currently special-cases thin panes because it
  cannot reliably tell an entry from an exit.
* **Robust self-shadow rejection.** A shadow ray can reject its own mesh at
  near-zero `t` by identity rather than by `MIN_HIT_DISTANCE = 1e-4` plus a normal
  offset — removing shadow acne at grazing light angles and on small-scale
  geometry, and removing another scale-dependent epsilon.
* **Material dispatch coherence.** Sorting hits by mesh id groups identical
  material evaluation, which is what `WAVEFRONT_SORT_MATERIALS` wants.
* **Exact absorption of coincident duplicates.** A union of sample masks is
  idempotent, so two genuinely coplanar stacked quads stop double-darkening.
* **Geometry that is actually watertight** (3a), retiring the u-seam gap, the
  degenerate pole fans, and two authoring-side epsilon special-cases in normal
  computation.
* **Two-level BVH (TLAS/BLAS).** Per-mesh BLAS reusable across a batch's frames
  for rigid meshes (the STBVH rebuilds per batch today), true instancing (a point
  cloud of 10k spheres becomes one BLAS plus 10k transforms instead of 10k copies
  of the geometry), and per-mesh culling. Blocker to clear first:
  `_split_promotable` (`scene_builder.py`) reorders promoted triangles by material
  value, so a partly-promoted surface already lands in two disjoint spans;
  per-mesh contiguity has to be established before a BLAS is meaningful.


================================================================================
6. MEASURED NEGATIVE RESULTS
================================================================================

`benchmarks/_aa_line_check.py` reports the symptom this work was partly aimed at:
a tessellated `Cylinder` scores 0.0568 px of ink wobble against 0.0138 for a flat
two-triangle quad, and 0.0773 when diced to `resolution=(256, 2)` — worse the
finer it gets. Two plausible causes were built and measured. **Neither is it.**

6.1 The consecutive-run requirement is not the problem
------------------------------------------------------
The obvious theory: `_aa_run_scan` takes a maximal *consecutive* run of
`(sid, facing)`, so a sheet whose fragments interleave with another's gets
corrected against a partial `Q`. Replaying the grouping for every covered pixel
puts `split` at **0.00–0.02%** on every case and `capped` under 1%. The grouping
is sound. Regrouping it into an order-independent equivalence class remains worth
doing for §5.2's order-independence, but it will not move any AA metric — do not
expect it to.

6.2 The union-full short-circuit is real but too small to matter
----------------------------------------------------------------
What *does* scale with tessellation density is v2 §4.2's
`U == _AA_MASK_ALL → corr = 1`: 1.0% of covered pixels on the flat quad, 25.2% on
a default `Cylinder`, 72.4% at `(256, 2)`, 87.6% on a fine `Sphere`. Almost all of
it is the benign interior tiling the short-circuit exists for (`1 - E` is float
dust: 343 / 10770 / 31096 / 23282 pixels). The residual is a genuinely dilated
silhouette tail of 1 / 105 / 181 / 1004 pixels with `1 - E` up to 0.15 (0.30 on
the sphere).

Consulting `E` there was implemented — on that path `Q == 1`, so `corr = E/Q` is
just `E`, with a 1e-3 dust band keeping genuine tilings bit-identical — and
measured: default Cylinder wobble 0.0568 → **0.0566** px with coverage rms 0.0094
→ **0.0099**; fine Cylinder 0.0773 → **0.0781** / 0.0164 → **0.0166**. Neutral at
best, marginally worse on rms. A few hundred pixels cannot move a frame-wide
metric, which the dust bucket dominating every histogram already implied. Not
shipped; the code was reverted rather than left as a dead gated path.

6.3 What that leaves
--------------------
Ruled out: the grouping, and the magnitude correction. Not examined: the one
thing the harness deliberately skips — the per-sample transmittance, and the
sampled **ownership** underneath it. A partial fragment's claim is positioned on
eight fixed sample points whatever its magnitude, and on a sub-pixel-diced mesh
many fragments compete for those eight positions; `_aa_line_check`'s own
docstring already blames "silhouette pixels contended by several triangles". That
is a *representation* limit, not a bug in the run rule, and the honest reading is
that eight sample positions cannot resolve a silhouette crossed by a dozen
sub-pixel triangles no matter how exactly each one's area is known.

Before building anything else here, extend `_aa_run_gate_check.py` to replay
`svis` and report how much of a diced silhouette pixel's final coverage is
decided by ownership rather than magnitude. If ownership dominates, the lever is
the sample count or an ownership representation, not the run rule — and note that
`_AA_SAMPLES` is deliberately a compile-time constant rather than a setting
(`raster_taichi.py`), so changing it means editing that line and clearing the
cache.

This also bears on §4.5: `ALGAN_MESH_ID=1` makes runs coarser, which puts *more*
pixels through the union-full branch. The two interact, and the arbiter has to be
rendered coverage against an exact reference, not a per-fragment error metric.
