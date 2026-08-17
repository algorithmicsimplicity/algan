# Algan — Mesh Identity in the Triangle Renderer

**Status: PARTLY LANDED. This file is the handoff document — start here.**

Plan of record for replacing the renderer's epsilon-based seam heuristics with
declared mesh identity. Written to be self-contained: a fresh session with only
this file and the repo should be able to continue without reconstructing any of
the reasoning.

Reading order. §0 is the state of the branch and what to do next. §1–§2 are the
problem and what shipped. §3 is the unstarted work with the anchors to do it. §4
is what needs a CUDA device and the experiment for each claim. §5 is what the
system enables. §6 is **what has actually been measured about the AA gap**,
mostly results that closed a door — read it before building anything in this
area, it will save you a day. §7 is methodology that cost real debugging time.

Everything measured here was measured on the **CPU** render device on a machine
with no GPU, unless it says otherwise. That is why §4 exists.


================================================================================
0. STATE OF THE BRANCH, AND WHAT TO DO NEXT
================================================================================

Branch `claude/renderer-mesh-id-rework-n5ezw5`, on top of `efb3a95`:

    b49b01b  Delete the unreachable curved PN-patch renderer
    c87c26b  Add _aa_run_gate_check: attribute the diced-mesh AA gap
    690009a  Mob-declared surface identity for tri_obj, gated off
    568b5ae  Add DESIGN_mesh_identity.md
    6d02488  Delete TriangleVertices2 and correct stale renderer comments
    a90b2ff  Apply ruff format to the files this branch touched
    c8e9b9b  Make DESIGN_mesh_identity.md a self-contained handoff
    e851ee6  Replay the resolve's svis walk: the AA gap is ownership
    e067702  Qualify ALGAN_MESH_ID on coverage; find the gate that costs the
             AA error
    (this)   Sound sheet reference; ALGAN_POLYHEDRON_WINDING

`985 passed, 87 skipped` on `pytest -q tests/unit_tests tests/fast`; `ruff check
--no-fix` and `ruff format --check` clean; pixel baselines untouched (every
behaviour change is still gated off, and the fast-suite render is byte-identical
across `ALGAN_POLYHEDRON_WINDING` with `ALGAN_MESH_ID` off).

**What the last session settled.** §6.3 was the gating question and it is
answered, in three steps that must be read in order because the third supersedes
the second's prescription:

* The diced-mesh AA error lands exactly on the **ownership** answer — 91% of a
  fine `Sphere`'s silhouette pixels on the 1/8 lattice (§6.3).
* Doubling the sample count behaves like a sampling limit: −30% ink wobble on a
  `Cylinder`, flat control unmoved (§6.3.1).
* But it is **not** a sampling limit. The magnitude is available and is being
  discarded by one gate: v2 §4.2 starts the run lookahead only on a partial
  mask, which excludes 52% of a `Sphere`'s silhouette pixels. Relaxing that gate
  to "partial mask, or a full mask whose exact area is not within dust of 1"
  takes the `Cylinder` from 0.0260 to 0.0030 and makes the flat control exact —
  no extra samples, no interior cost (§6.3.2).

Building that measurement also produced the arbiter §4.5 had been waiting for.
`ALGAN_MESH_ID=1` measures **neutral** on coverage — nothing regresses, nothing
gains beyond noise — which neither blocks the flip nor argues for it, and leaves
the case for it resting on §2.2's correctness argument rather than on a measured
win (§4.5). The one case that could still show a win, a packed-grid `Surface`,
is the one the arbiter has not been pointed at; §4.5 says how.

And it turned up a `Polyhedron` winding defect, now fixed behind
`ALGAN_POLYHEDRON_WINDING`, default off (§3.7, §6.5). Three predictions about it
were made and measured, and **all three were wrong**: it is not why MESH_ID
regressed under the old metric, fixing it does not make MESH_ID pay, and it does
not move the fast-suite render at all with MESH_ID off (byte-identical). What it
does do is make the sheets of a solid nameable — the arbiter's drop count on an
`Icosahedron` goes 960 to 4. Read §6.5 before reusing any of them.

**Recommended next step, in priority order.**

1. **§6.3.2's gate relaxation.** The biggest measured quality win in this area
   by a wide margin, and the only one that costs nothing. It is a
   `DESIGN_analytic_aa_v2` change rather than a mesh-identity one, but it is
   fully specified in §6.3.2 including the implementation shape and the four
   harnesses that qualify it. Needs a CUDA machine only for the baselines.
2. **Point the arbiter at a packed-grid `Surface`** (§4.5). One new case in
   `_aa_run_gate_check.py`, CPU-runnable, and it is the missing evidence for or
   against §3.5. Do this before spending a CUDA re-baseline on MESH_ID.
3. **On a CUDA machine: §4.1–§4.4**, the deferred verification for the PN
   deletion — one run each. If 1 and §3.5 are both going in, land them together
   so it is one re-baseline instead of two.
4. **§3.1 seam welding.** Self-contained, CPU-verifiable, valuable on its own
   (it retires two authoring-side epsilons), and the prerequisite for §3.2.
   Note §6.3 has downgraded the *AA* case for §3.1/§3.2: neither addresses what
   the error turned out to be. They are worth doing for watertightness and for
   the epsilon retirement, not as a quality fix.
5. **§3.2 the watertight test**, then §3.3, §3.4.

Do **not** start by regrouping the run rule, by consulting `E` only inside the
existing gate, or by buying more samples. All three were built or measured here
and none is the lever — §6.


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

3.5 Flip `ALGAN_MESH_ID` on  [qualified; blocked only on CUDA baselines]
-------------------------------------------------------------------------
§4.5 is measured and comes back **neutral**: nothing regresses, nothing gains,
and the fill-rule and dump checks pass with it on. So this is no longer blocked
on a quality question — it is blocked on someone deciding the correctness
argument is worth a re-baseline, and on the one piece of evidence that could
still make it a win: the packed-grid `Surface` case §4.5 asks for. Point the
arbiter there first.

If it goes in: regenerate `expected_outputs_cpu/` **and**
`expected_outputs_cuda/` for `tests/fast` and `tests/full_renders`, look at the
diff videos, then change the default in `settings.py` and rewrite the
"DEFAULT OFF" comment there.

3.6 Two-level BVH (TLAS/BLAS)  [design only]
---------------------------------------------
See §5.2. Blocker to clear first: `_split_promotable` (`scene_builder.py:572`)
reorders promoted triangles by material value, so a partly-promoted surface
already lands in two disjoint spans; per-mesh contiguity has to exist before a
BLAS is meaningful.

3.7 Orient `Polyhedron` faces outward  [LANDED, gated off]
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

**LANDED**, gated `ALGAN_POLYHEDRON_WINDING`, default off, surfaced as
`SETTINGS.raytracing.experimental.set(polyhedron_winding=...)`. Implemented as
described above in `shapes_3d.orient_faces_outward`, called from
`Polyhedron.__init__`. `tests/unit_tests/test_mesh_identity.py` pins the defect
itself (the per-solid inward counts, so a face-list edit cannot change them
quietly), that the pass fixes all five solids without changing which vertices a
face uses, that it declines on open / non-manifold / degenerate input, and that
it repairs a deliberately mis-wound and a wholly inverted tetrahedron.

Measured, and **not** what §6.5 first predicted: with `ALGAN_MESH_ID` off the
fast-suite render is **byte-identical** across this gate (same sha256, and that
scene draws a `Cube`, an `Icosahedron` and an `Octahedron`). A per-triangle
surface id makes every run one fragment, so the facing bit groups nothing and
flipping it changes nothing downstream. With `ALGAN_MESH_ID=1` the render does
change — which is the mechanism stated plainly: one id per solid leaves facing
as the only thing separating the two sheets.

Left to do before the default flips: `tests/full_renders` on a machine whose
baselines those are, plus the CUDA fast suite. If those are byte-identical too,
this can flip with no re-baseline at all.


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

    **The gap in the evidence, and the experiment that closes it.** None of
    these six cases is a **packed-grid `Surface`**, which is the end §2.2 fixes
    in the other direction: one merged member covering every packed sphere, so
    distinct spheres are unioned into one surface and their coverage summed
    across objects that merely overlap. That is where a measurable win should
    be, and it is the one case the arbiter has not been pointed at. Add it
    before deciding.

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
