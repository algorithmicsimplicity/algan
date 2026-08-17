# Algan — Mesh Identity in the Triangle Renderer

**Status: PARTLY LANDED. This file is the handoff document — start here.**

Plan of record for replacing the renderer's epsilon-based seam heuristics with
declared mesh identity. Written to be self-contained: a fresh session with only
this file and the repo should be able to continue without reconstructing any of
the reasoning.

Reading order. §0 is the state of the branch and what to do next. §1–§2 are the
problem and what shipped. §3 is the unstarted work with the anchors to do it. §4
is what needs a CUDA device and the experiment for each claim. §5 is what the
system enables. §6 is the **measured negative results** — read it before
building anything in this area, it will save you a day. §7 is methodology that
cost real debugging time.

Everything measured here was measured on the **CPU** render device on a machine
with no GPU, unless it says otherwise. That is why §4 exists.


================================================================================
0. STATE OF THE BRANCH, AND WHAT TO DO NEXT
================================================================================

Branch `claude/triangle-rendering-rework-73e2jv`, six commits on top of
`efb3a95`:

    b49b01b  Delete the unreachable curved PN-patch renderer
    c87c26b  Add _aa_run_gate_check: attribute the diced-mesh AA gap
    690009a  Mob-declared surface identity for tri_obj, gated off
    568b5ae  Add DESIGN_mesh_identity.md
    6d02488  Delete TriangleVertices2 and correct stale renderer comments
    a90b2ff  Apply ruff format to the files this branch touched

`981 passed, 87 skipped` on `pytest -q tests/unit_tests tests/fast`; `ruff check
--no-fix` and `ruff format --check` both clean; pixel baselines untouched (every
behaviour change is gated off). PR #34 was opened as a draft and closed
deliberately — the next session owns the PR.

**Recommended next step, in priority order.**

1. **§6.3's `svis` measurement.** Cheapest, and it decides whether any further
   work in this area is worth doing. Extend `benchmarks/_aa_run_gate_check.py`
   to replay the resolve's per-sample transmittance and report how much of a
   diced silhouette pixel's final coverage is decided by *ownership* (which of
   the eight fixed sample points a fragment claims) rather than *magnitude*
   (how exactly its area is known). §6 has ruled out grouping and magnitude;
   ownership is the untested hypothesis, and if it dominates then the lever is
   the sample count or the ownership representation, not the run rule.
2. **§3.1 seam welding.** Self-contained, CPU-verifiable, valuable on its own
   (it retires two authoring-side epsilons), and the prerequisite for §3.2.
3. **§3.2 the watertight test**, then §3.3 and §3.4.

Do **not** start by regrouping the run rule or by making the run rule consult
`E`. Both were built and measured on this branch and neither works — §6.


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

2.2 Mob-declared surface identity, gated off (`690009a`)
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

`ALGAN_MESH_ID` **defaults off**, so this is byte-identical. Why, and what would
justify flipping it: §4.5.

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


================================================================================
3. WHAT HAS NOT LANDED
================================================================================

3.1 Weld the `Sphere` u-seam and the pole fans  [CPU-verifiable]
----------------------------------------------------------------
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

Gate `ALGAN_WELD_SURFACE_SEAMS`, default off. Geometry moves, so **all** pixel
baselines move on both devices. Validation: extend
`test_logical_pn_tessellation.py`'s watertightness style (assert wrapped column
indices resolve to column 0, pole rows contribute no degenerate triangles);
`benchmarks/_grid_normals_ab.py` already covers 13 cases including pole and
closed-seam grids.

3.2 Watertight ray/triangle intersection  [needs CUDA to qualify]
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

Gate `ALGAN_WATERTIGHT_TRI`, default off. With it on, `BARYCENTRIC_EPSILON`'s
dilation goes to zero and all 8 seam-rule sites compile out.

3.3 Delete the epsilon apparatus  [blocked on 3.2 qualifying]
--------------------------------------------------------------
`BARYCENTRIC_EPSILON`, `TRIANGLE_EDGE_EPSILON`, the `edge_hit` flag bit
(packing documented at `raytrace_kernels_taichi.py:1708`, frees a bit), `seam_t` (`rs_sca[r, 3]`,
frees a per-ray f32) and the 8 call sites with their initialisations and bounce
resets. Cannot land while a gate must still be able to select the old
behaviour, so this is a follow-up commit conditioned on §4.7 coming back clean.
`rs_sca` shrinking moves the arena fit — re-check `memory_model` (§4.7).

3.4 Median-split STBVH for bezier circuits  [needs CUDA to qualify]
--------------------------------------------------------------------
Once resolution is order-independent, `stbvh.py:302`'s reason for pinning
bezier to Morton is gone (PN, the other pinned type, no longer exists). Add
`ALGAN_BEZ_BVH_SPLIT`, default off, flipping the bezier default to `"split"`.
~20-25% fewer traversal steps.

Note there is **no remaining slot-order freeze to undo**: the "every patch keeps
its slot" constraint was PN-specific and went with the PN merge block in
`b49b01b`. The only value-order reorder left in the merge is `_split_promotable`
grouping promoted triangles by material value (`scene_builder.py:572`), which is
unrelated to BVH build order.

3.5 Flip `ALGAN_MESH_ID` on  [blocked on §4.5]
-----------------------------------------------

3.6 Two-level BVH (TLAS/BLAS)  [design only]
---------------------------------------------
See §5.2. Blocker to clear first: `_split_promotable` (`scene_builder.py:572`)
reorders promoted triangles by material value, so a partly-promoted surface
already lands in two disjoint spans; per-mesh contiguity has to exist before a
BLAS is meaningful.


================================================================================
4. WHAT MUST BE VERIFIED ON A CUDA DEVICE
================================================================================

Clear the Taichi cache (`clear_cache(taichi_kernels=True)`) before any A/B — it
does not invalidate on `@ti.func` edits. Never edit `*_taichi.py` while a render
or a warm daemon is running.

4.1 **Confirm the committed CUDA baselines still pass.** Nothing on this branch
has moved output, so they should:

        pytest -q tests/fast
        ALGAN_RUN_FULL_RENDERS=1 pytest -q tests/full_renders

    A failure here is the PN deletion, not baseline staleness, and the first
    thing to check is `layer_offsets` (§7.1). The six `full_renders` scenes are
    the ones that matter — they carry the PN surfaces, shadows, refraction and
    glTF that the fast scene deliberately omits.

4.2 **Confirm the PN deletion is byte-identical on CUDA**, which is stronger
    than the baselines because it compares against the pre-deletion tree:

        git stash && pytest -q tests/fast && sha256sum tests/fast/algan_outputs/fast.mp4
        git stash pop && pytest -q tests/fast && sha256sum tests/fast/algan_outputs/fast.mp4

4.3 **Confirm the kernels did not get slower.** The deletion removes 12 merged
    keys, two BVH builds per batch and ~10 parameters from every traverse/shade
    signature, so expect neutral to faster. Kernel-profiler **device** times,
    not wall clock — thermal throttling swings cross-process throughput ~2x, so
    use in-process alternating A/B. `utils/profiling_utils.py` auto-hooks the
    kernels and pipeline stages. Watch `wavefront_shade`,
    `wavefront_traverse`, both MC megakernels, `raster_first_shade`, and the
    per-batch BVH build time.

4.4 **Confirm the compile surface shrank.** Count offline-cache entries and
    cold-compile wall time; the deletion removes the `has_pn` template
    dimension from four kernels.

4.5 **Qualify `ALGAN_MESH_ID=1`.** This needs an arbiter this branch did not
    have. Turning it on moves the fast-suite render by up to 49 channel values
    at solid edges (edge-confined; no interior shading change), and
    `_aa_run_gate_check` reports an Icosahedron's per-fragment error *rising*
    (14.35 → 146.84) because coarser identity puts more pixels through the
    union-full short-circuit. That harness **cannot** say which render is
    better: it does not model the per-sample transmittance that the
    fine-grained ids get wrong. The experiment that can:

        Render a Polyhedron silhouette at anti_alias_level=8 as ground truth,
        then compare MESH_ID=0 and MESH_ID=1 at aa=1 against it. Better: add a
        Polyhedron case to benchmarks/_aa_line_check.py, which already compares
        against EXACT analytic coverage rather than a supersampled reference.

    Also re-run `_analytic_aa_fillrule_check.py` (the sample-partition
    property) and `_aa_dump_check.py` (golden host-side walk vs the kernel's own
    dump — this is what catches a resolve/shadow-event desync). If MESH_ID=1
    wins, re-baseline **both** device sets and flip the default.

4.6 **Shadow-mode agreement — a testable prediction.** Three `SHADOW_ANYHIT`
    modes disagree today in corner cases documented as seam-merge artifacts
    (`raytrace_kernels_taichi.py:2337` and 2535). Once identity replaces
    the epsilon (§3.3) those disagreements should vanish:

        for m in 0 1 gather; do ALGAN_SHADOW_ANYHIT=$m pytest -q tests/full_renders; done

    Diff the three outputs; they should become identical. If not, there is a
    second cause worth isolating before §3.3 ships.

4.7 **Watertight test (§3.2), once built.**
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

4.8 **Median-split bezier BVH (§3.4), once built.**
    `benchmarks/_split_determinism_check.py` distribution before/after, plus
    traversal-step counts and build times. Byte-identity is the **wrong** gate:
    that script documents that split pixels are not byte-reproducible (|d| = 1
    from non-associative float `atomic_add` on `pix_accum`), so compare
    distributions.

4.9 **Every gate off is byte-identical** — `ALGAN_MESH_ID=0`,
    `ALGAN_WELD_SURFACE_SEAMS=0`, `ALGAN_WATERTIGHT_TRI=0`,
    `ALGAN_BEZ_BVH_SPLIT=0`. Verified on CPU for `ALGAN_MESH_ID`; redo on CUDA,
    where the other kernel variants live.

4.10 **Render twice, baseline the second.** The first render on a fresh machine
    populates the Manim Tex geometry cache and its `MathTex` glyph antialiasing
    differs from every run after it — 18 channel values over 100 frames of
    `text_and_media`, against a tolerance of 2.


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
  for imported meshes (gated off pending §4.5).
* **A way to ask questions about the run rule** and get population answers
  instead of anecdotes — which is what turned two plausible theories into §6.
* **`tri_obj` is under test.**

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

6.3 What that leaves — the untested hypothesis
-----------------------------------------------
Ruled out: the grouping, and the magnitude correction. **Not examined:** the one
thing the harness deliberately skips — the per-sample transmittance, and the
sampled **ownership** underneath it. A partial fragment's claim is positioned on
eight fixed sample points whatever its magnitude, and on a sub-pixel-diced mesh
many fragments compete for those eight positions; `_aa_line_check`'s own
docstring already blames "silhouette pixels contended by several triangles".

The honest reading is that this is a *representation* limit rather than a bug in
the run rule: eight sample positions cannot resolve a silhouette crossed by a
dozen sub-pixel triangles however exactly each area is known. If that is right,
the levers are the sample count or the ownership representation — and note
`_AA_SAMPLES` is deliberately a compile-time constant rather than a setting
(`raster_taichi.py:213`), so changing it means editing that line and
clearing the cache.

Measure before building: extend `_aa_run_gate_check.py` to replay `svis`.

6.4 This interacts with §4.5
-----------------------------
`ALGAN_MESH_ID=1` makes runs coarser, which puts *more* pixels through the
union-full branch. The two changes are coupled, and the arbiter has to be
rendered coverage against an exact reference, not a per-fragment error metric.


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
