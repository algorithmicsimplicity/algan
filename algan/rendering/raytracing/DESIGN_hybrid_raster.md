# Algan Raytracer v2 — Hybrid Rasterization Front-End: Design Document

Status: in progress. Phases 1–3 built and validated; the front-end ships
behind an opt-in toggle (`ALGAN_HYBRID_RASTER`, default OFF) while the classic
deterministic wavefront remains the default renderer. This document explains
the current implementation, the reasoning behind each design decision, the
benchmark results that drove those decisions, and the planned future work.

Author's note on scope: this is a *ground-up redesign* of the deterministic
(samples-per-pixel == 1) render path. Byte-for-byte identity with the old
renderer is explicitly NOT a goal — all render tests are being re-validated,
and the new path is free to change hit-ordering semantics where that makes the
output more accurate or the renderer faster. The Monte Carlo path tracer
(samples > 1) is out of scope and untouched.


================================================================================
1. MOTIVATION
================================================================================

The existing deterministic renderer is a *wavefront ray tracer*
(`wavefront_kernels_taichi.py`, orchestrated by `tracer.py`). For each pixel it
casts a primary ray, traverses a spatio-temporal BVH (STBVH) per geometry type
(triangles, PN patches, bezier circuits), gathers up to KBUF=4 nearest hits,
shades them front-to-back, and follows reflection/refraction continuations.

Profiling (universal profiler, `profiling_utils.py`, kernel device times) on a
dense scene shows where the time goes:

    wavefront_traverse   84.8 %   (BVH traversal)
    wavefront_shade      11.8 %
    wavefront_generate    1.5 %
    wf_composite_accum    1.4 %
    compact_ray_slots     0.3 %

BVH traversal dominates. It is *latency-bound pointer chasing*: each node visit
is a dependent global-memory read, and the kernel is occupancy-starved
(measured 21–25 % achieved occupancy, register-capped, with heavy local-memory
spilling — see the RT occupancy diagnosis). Two structural problems compound:

  (a) Every one of ~2M pixels/frame (at HD) independently descends the tree
      from the root. The traversal cost scales with pixels × tree-depth.

  (b) The K-buffer only holds 4 hits, so a ray peeling through a deep
      transparency stack (a fade, stacked translucent panes, dense text over
      shapes) re-traverses all three BVHs from the root once per refill. The
      worst case is exactly Algan's most common transition — a fade, where
      everything is semi-transparent at once and the opaque-hit pruning that
      normally caps gathering stops working entirely.

The redesign attacks both by replacing the *primary* ray trace (one ray per
pixel) with a rasterization step. Instead of gathering candidates from the ray
side by walking a tree, we enumerate them from the primitive side by scatter:
each primitive computes its screen-space bounding box, we test the covered
pixels, and we composite the resulting fragment lists. This is streaming,
coalesced, embarrassingly-parallel work — the right trade on a bandwidth-limited
GPU — and it removes the K-buffer refill loop entirely (all hits along a primary
ray are found in one pass, unbounded in depth).

Secondary rays (shadows, reflections, refractions) still ray-trace, so the BVH
is not going away; but for many real scenes (no shadows, reflections, or
refraction) the primary pass needs no BVH at all.


================================================================================
2. MEASUREMENT GATES (Phase 1) — why we believed it would work
================================================================================

Before writing a kernel, three cheap measurements were run to de-risk the
design. All three passed. Scripts: `benchmarks/_rt2_sort_bench.py`,
`benchmarks/_rt2_capture.py` (+ `_rt2_overdraw.py`, `_rt2_refit_sah.py`).

2.1 Sort throughput (GTX 1050)
------------------------------
The design sorts per-tile transparent-fragment records by a 64-bit
(pixel | depth) key. `torch.sort` (cub radix on CUDA):

    raw int64 keys:                        ~230 Mkeys/s
    end-to-end incl. 16 B payload gather:  ~140 Mkeys/s

Perfectly linear from 1M to 16M keys; no size cliff. Verdict: the sort is not a
bottleneck. At realistic fragment volumes it costs single-digit ms/frame.

2.2 Candidate / overdraw statistics
-----------------------------------
Captured real merged scenes and projected every primitive's per-frame AABB
through the exact camera mapping to a screen bbox, comparing candidate-test
count against true perspective-projected coverage:

    scene (resolution)          cand/px   depth-complexity/px   overdraw   sort/frame
    basic_mixed (PREVIEW)         0.74           0.55            tri 3.8x    0.09M
    text_fade (PREVIEW)           1.36           1.36            ~1x         0.33M
    neural_net (HD AA2, PN)       5.49           0.55            PN 10.3x    0.73M

Candidates/pixel of 0.7–5.5 is ~20x fewer tests than the classic walk's ~99
node + ~14 leaf tests per sub-ray, and it is streamed rather than
pointer-chased. The predicted pathology (thin primitives whose bbox area is
quadratically larger than their coverage — arrows, underlines, synapse
cylinders) showed up exactly where expected: 10.3x overdraw on the neural net's
thin cylinders. Mitigations: cheap inside-tests (screen-space edge functions)
and, for PN, projecting OBB corners rather than AABB corners.

2.3 Refit-topology staleness (for the acceleration structure)
-------------------------------------------------------------
Compared SAH expected-visit cost of a per-frame-rebuilt tree, a
union-centroid topology refit per frame, and the current STBVH:

    union-refit vs per-frame rebuild:  1.000–1.04 (mean, all scenes/types)
    current STBVH vs union-refit:      tri 1.37x, PN 1.67x, bez 1.78–2.33x worse

Refitting one topology across a 7–15-frame batch is essentially free of quality
loss, and beats the current spatio-temporal instance tree. This validates the
planned Phase-4 acceleration structure (§7).


================================================================================
3. THE HYBRID RASTER FRONT-END — architecture
================================================================================

Files:
  algan/rendering/raytracing/raster_taichi.py     — GPU kernels (Taichi)
  algan/rendering/raytracing/raster_pipeline.py   — host orchestration (torch)
  algan/rendering/raytracing/scene_builder.py     — per-frame visibility masks
  algan/rendering/raytracing/tracer.py            — gate + iteration-0 wiring
  algan/settings/renderer_settings.py             — force-flat helper

The front-end replaces the wavefront's *first* iteration (generate + first
traverse + first shade). Bounced continuations (reflection/refraction) it
spawns are handed to the unchanged classic wavefront loop for iterations >= 1.
Thus the front-end is a drop-in replacement for primary visibility only; all
the material, shadow, environment-map, and multi-bounce machinery is reused.

Per screen tile, per frame-group (all torch/Taichi, `memory.temp()`-scoped):

  1. BIN      Project each primitive's per-frame world AABB (or the 3 triangle
              verts) to a conservative screen bbox; emit fixed-size
              (primitive, pixel-chunk) pair rows.
  2. Z-PREPASS (triangles only) Exact ray/triangle test per covered pixel;
              keep the nearest *opaque* hit per pixel with a packed
              (depth_bits << 32 | prim) int64 atomicMin.
  3. COUNT    Per pair, count surviving transparent fragments (strictly nearer
              than the pixel's opaque z-winner). Sizes the fragment list
              exactly — no atomic append, deterministic layout.
  4. WRITE    Emit (key, payload) records at each pair's exact offset.
  5. SORT     torch.argsort (stable) the fragment keys → per-pixel depth-ordered
              runs.
  6. RESOLVE  One thread per tile pixel walks its sorted run front-to-back (the
              opaque z-winner appended as the terminal hit), alpha-composites,
              runs material/fragment shading, and spawns reflect/refract
              continuations into the shared wavefront pool.

The remaining wavefront iterations then trace only the spawned continuations.


================================================================================
4. DESIGN DECISIONS AND RATIONALE
================================================================================

4.1 Cull before the sort, not after
------------------------------------
The original sketch was: rasterize bbox → sort all fragments → null occluded.
But the sort is the bandwidth-heavy step, so misses must never reach it. The
exact intersection test (and the cheap alpha fetch) are fused into fragment
*emission*: a fragment is appended only if the pixel's ray actually hits the
primitive and clears the opaque z-buffer. Bbox overdraw then costs only
intersection tests, not sort traffic.

4.2 Opaque z-prepass via packed atomicMin
-----------------------------------------
Fragments from proven-opaque primitives skip the fragment list entirely: an
`atomicMin` of `(depth_bits << 32 | prim)` per pixel keeps the nearest. `min`
is commutative, so the winner is deterministic regardless of thread order — no
sort needed for opaque geometry, and a fully-opaque scene degenerates to a
plain z-buffer. Transparent emission then culls against this z-buffer, so the
sort only ever sees translucent survivors. A fully-opaque scene does zero
sorting; a fade (everything translucent) routes everything to the sorted path,
which is correct and still one sort.

Per-frame opacity flags already exist (the STBVH marks interval-opaque
instances); the front-end reuses them as a `[T, N]` mask.

4.3 Two-level (primitive, chunk) pairs bound load imbalance
-----------------------------------------------------------
One thread per primitive iterating its whole bbox has terrible load imbalance
(a full-screen fade rectangle = millions of pixels in one thread). Instead each
primitive's clipped bbox is split into chunks of RASTER_CHUNK (=256) pixels;
one thread processes one (primitive, chunk) pair. This bounds per-thread work
and gives a natural OOM-retry granularity.

4.4 Per-pixel SERIAL resolve, not a parallel prefix scan
--------------------------------------------------------
The compositing rules are not a clean associative scan: seam de-duplication
(edge-flagged triangle hits within a depth tolerance of the previous accepted
edge hit are dropped), and termination at the first *path-bending* surface
(refraction / metal reflection / custom scatter make everything behind them
along the straight ray invalid). A per-pixel thread walking its sorted run
serially expresses all of this directly — it is the classic shade drain loop
minus the traversal, with unlimited K. There are ~2M pixels/frame, so
parallelism is not the constraint, and each run is a genuine sequential
dependence chain anyway.

4.5 Raw-depth ordering — the deliberate semantics change
--------------------------------------------------------
The classic renderer bins hit distances into DEPTH_TIE_EPSILON-wide bins and
falls back to a layer index within a bin. That epsilon-bin order is
*non-transitive*, which made K-buffer eviction order-sensitive and pixels
perturb under any BVH change. The new path sorts by *raw f32 depth bits*, a
true total order — compositing is deterministic and independent of tile size,
chunking, and build order. This is strictly better; it is also why the new path
is not byte-identical to the old one (see §6).

4.6 Fragment tagging and coplanar layering
------------------------------------------
Payload arrays are SoA: `frag_key` (sort key), `frag_t`, `frag_prim`,
`frag_ab` (barycentrics or plane u,v), `frag_flags`. A triangle fragment has
`frag_prim >= 0`; a bezier fragment has `frag_prim = -(circuit + 1)` and its
in-border flag in `frag_flags`. The resolve decodes the sign. Bezier fragments
are emitted at *lower indices* than triangle fragments so the stable sort places
a coplanar circuit ahead of a triangle at equal depth (circuits-over-triangles).

4.7 In-place bounce continuations
---------------------------------
The resolve thread for pixel r owns ray slot r. A reflected/pass-through
continuation is written back into slot r (status ACTIVE) exactly like the
existing fused-generation path; split branches (glass refraction, semi-
transparent reflectors) atomically append to the shared continuation pool via
`_reserve_continuation_slot`. The pool's overflow-retry (halve the primaries,
retry) is unchanged. So the front-end's postcondition matches the classic first
traverse+shade exactly: `pix_accum` holds every retired pixel's colour +
leftover background weight, bounced pixels are ACTIVE, and the classic loop
takes it from there.

4.8 Memory model
----------------
The wavefront's per-ray state stays arena-backed (bump allocator, deterministic
release). The raster's transient buffers (z-buffer, pair rows, fragment arrays,
run tables) are ordinary torch CUDA allocations *outside* the arena. An
allocation failure raises OutOfMemoryError, which the render loop's existing
halve-the-window-and-retry already handles. This is a known wart (the transient
buffers compete with arena headroom and can trigger extra splits); arena-backing
them is a planned cleanup (§7).


================================================================================
5. GEOMETRY PATHS
================================================================================

5.1 Flat triangles
------------------
Both the z-prepass and the transparent path test candidate pixels against the
triangle. Two intersection modes, selected by `ALGAN_RASTER_SS`:

  Ray-cast (`_raycast_pixel`): generate the pixel's world ray and run
  Möller-Trumbore. Straightforward, exact.

  Screen-space (`_ss_setup` + `_ss_pixel`, default ON): project the triangle's
  3 vertices ONCE per pair (the exact forward of the camera's `_generate_ray`
  mapping: screen-plane normal n = pbx × pby, perspective divisor
  d_i = dot(V_i − cam_o, n)), then per pixel evaluate three edge functions and
  perspective-correct barycentric weights w_i = (E_i / d_i) / Σ(E_j / d_j). The
  3D hit point H = Σ w_i V_i gives the exact distance t = |H − cam_o| and
  barycentrics. A triangle straddling the camera plane (any vertex at/behind it)
  falls back to ray-cast, where the projective map would be non-finite.

  Why screen-space: for high-overdraw scenes the cheap edge-function sign test
  rejects a *miss* far more cheaply than a full ray-gen + Möller-Trumbore, and
  the projection setup is hoisted out of the per-pixel loop. It is numerically
  equivalent to ray-cast (verified worst |Δt| ~5e-5, |Δbary| ~6e-5 over ~1900
  random hits, `benchmarks/_rt2_ss_math_check.py`), so parity is unchanged.

  Screen-space math (perspective correctness): the screen coordinate of a world
  point is a projective function of world position with perspective divisor
  d_i; interpolating a per-vertex attribute perspective-correctly weights by
  1/d_i. The barycentric coordinate itself is such an attribute, giving the
  formula above. A screen-plane point has d = D so it projects to itself
  (round-trip identity).

5.2 Bezier circuits (2D shapes, text glyphs)
--------------------------------------------
A circuit is a planar cubic-bezier outline embedded in 3D via a plane
(center + normal + basis u/v). Bezier is routed ENTIRELY through the transparent
sorted path — never the z-prepass — which was a deliberate low-risk choice: the
validated triangle z-prepass and triangle kernels are completely untouched.

  `raster_bez_count` / `raster_bez_write` (`_bez_pair_pixel`): project the
  circuit's per-frame world AABB (8 corners) to a screen bbox; per candidate
  pixel, cast the ray, intersect the plane, project to (u, v), and run the
  existing `_bezier_point_metrics` (crossing count for fill + nearest-edge
  distance for border/outline) to test inside/border. This ports the bezier
  branch of the classic `_collect_hits` unchanged; base_dist is 0 for
  primaries, so the screen-constant border width uses t directly.

  Correctness of always-transparent: an opaque bezier simply appears as a
  transparent fragment with alpha = 1, which terminates the straight-line
  accumulation (weight → 0). Transparent bezier fragments are still culled
  against the opaque *triangle* z-buffer, so geometry behind an opaque triangle
  is correctly removed. The cost is that opaque bezier does not itself cull
  geometry behind it in the z-prepass, but bezier coverage is typically small
  (text/thin shapes), so the extra fragments are few.

  Resolve: circuits keep their sampled colour (they are never material-shaded —
  a deliberate deviation matching the classic renderer), take reflectivity /
  IOR / transmission from `circuit_meta`, use `_bezier_normal`, and their
  continuation is the "thin pane" case (reflect into a split slot, transmit
  unbent into the pass-through). The pane branch shares the triangle
  `split_refl` body; the glass (refracting) branch stays triangle-only.

Why bezier and not PN: bezier (text, 2D shapes) is the bulk of real explanatory-
math content, so without it the front-end helps almost no real scene. PN patches
(curved surfaces) are the hard case (quadratic patch intersection) — instead of
rasterizing them, the front-end forces PN surfaces to render as *flat*
triangles when raster is on.

5.3 Force-flat under raster
---------------------------
`renderer_settings.effective_triangle_primitive()` returns the flat triangle
class whenever `HYBRID_RASTER` is on; the three surface/mesh/2D-shape build
sites go through it, and `Surface._uses_pn_triangles()` returns False under
raster so surfaces auto-tessellate against the flat error metric. Consequently a
qualifying raster scene always has `num_pn == 0`; the gate keeps a `num_pn == 0`
check purely as a safety that routes any stray PN geometry to the classic path.


================================================================================
6. SEMANTICS DELTAS FROM THE CLASSIC RENDERER
================================================================================

The front-end is intentionally not byte-identical. Known differences, all
accepted (the test suite is being re-baselined):

  * Raw-depth hit ordering replaces the epsilon-bin + layer order (§4.5). This
    perturbs scattered pixels wherever surface depths tie — silhouettes,
    coincident triangle edges. Measured a few tenths of a percent of pixels,
    visually identical.

  * Strict opaque z-cull: a transparent fragment exactly coincident in depth
    with an opaque surface is culled (the comparison is strict). If a coplanar
    decal exactly on an opaque surface ever needs to survive, revisit with a
    <= comparison plus resolve-side ordering.

  * Coplanar-overlap order for circuits: two overlapping circuits at (near-)
    equal depth may swap which is on top versus the classic layer order. This is
    an explicitly accepted difference (user rule: swapping the order of
    overlapping beziers is fine as long as there are no rendering artifacts).


================================================================================
7. ACCELERATION STRUCTURE
================================================================================

Current (`stbvh.py`): a spatio-temporal BVH per geometry type. Time is a fourth
dimension; primitives are adaptively segmented into (frame-interval, union-
bound) instances, ordered along a 4D Morton curve, and packed into an implicit
4-ary tree with sibling-block nodes (one aligned fetch tests a whole sibling
group). Triangles use a median-split build; PN/bezier use Morton. This is used
by the wavefront (secondary rays) and, until the front-end lands fully, primary
rays.

Its weakness (measured, §2.3): at the confirmed-optimal tightness=1.0, moving
geometry segments to near-per-frame instances, so the tree is ~10x larger than
the primitive count and every ray wades through mostly other-frames' instances
gated out by frame-interval tests.

Planned (Phase 4): shared-topology SAH tree with per-frame refit. Build ONE
binned-SAH topology per batch over the N primitives (not instances), with an
explicit child-base index per sibling block (unlocking unbalanced trees), and
refit node bounds per frame as a vectorized `[T, blocks, 8, ARITY]` reduction
(static geometry deduped to T=1). Measured benefits: eliminates the instance
blowup (nodes ∝ primitives), gives exactly-tight per-frame boxes (the thing the
tightness A/B proved dominates), removes the frame-interval gates, and makes the
~14% SAH win affordable because topology is built once per batch. Refit
staleness over a batch is negligible (§2.3). Also planned: a single mixed-type
tree for shadow rays (they currently walk three trees serially and only need
any-hit).


================================================================================
8. BENCHMARK RESULTS
================================================================================

All perf numbers are kernel-isolated (sync-fenced timing of only the ray-traced
render call), warm, alternating in one process. Wall-clock at low resolution is
useless here — the render is sub-0.5s and prep+video-encode dominate — so all
comparisons are at MD/HD where the GPU render dominates. Hardware: GTX 1050
(Pascal, 4 GB). Scripts: `benchmarks/_rt2_raster_kp.py` (spheres),
`_rt2_raster_nn_kp.py` (neural net), `_rt2_raster_bez_parity.py`,
`_rt2_raster_parity.py`, `_rt2_raster_refract_parity.py`.

IMPORTANT measurement caveat (cost real hours): the raster gate requires
num_pn == 0 AND num_circuits == 0-eligible geometry. A `Text(...)` label or any
2D shape is a bezier circuit; before bezier support, its presence silently
routed the whole batch to classic. Early neural_net numbers were therefore
classic-vs-classic and bogus. All benchmarks now assert engagement (a counter
on `raster_iteration_zero`), and the profiler cross-checks (raster kernels
present, wavefront_generate absent).

8.1 Speed — raster vs classic (engagement-verified)
---------------------------------------------------
    scene                                    resolution   raster vs classic
    20 flat-tri spheres                      HD 1920x1080      2.26x
    dense flat-tri neural_net (no text)      MD 1280x720       3.27x
    full neural_net (net + Text label)       MD 1280x720       2.75x   *
      * previously ZERO benefit — the text silently forced classic.
        classic 8.70s  ->  raster 3.16s.

The win scales with both pixel count and BVH depth/primitive count (the raster
eliminates per-pixel traversal; classic wavefront_traverse is ~85% of classic
GPU time). Raster kernel times are also far tighter run-to-run than classic
(less thermal sensitivity).

8.2 Screen-space vs ray-cast (the intersection-mode A/B)
--------------------------------------------------------
    scene                                overdraw   SS vs ray-cast
    20 spheres HD                          low          0.94x  (SS ~6% slower)
    dense flat neural_net MD               ~10x         1.36x  (SS 36% faster)
    full neural_net (net + text) MD        mixed        1.11x

Screen-space wins in proportion to bbox overdraw: its edge-function inside-test
rejects a miss far more cheaply than ray-gen + Möller-Trumbore, and dense/thin
triangles are almost all misses. Low-overdraw spheres fill their bbox, so
there are no misses to save on and the per-triangle setup is pure overhead.
Default ON: the high-overdraw case is both the expensive one and the realistic
one. (An earlier "SS is a miss" reading was the bezier-text artifact above.)

8.3 Parity (engagement-verified)
--------------------------------
    test                                          result
    triangle, vertex-shaded                       0.005% px differ (max 8)
    triangle, fragment-shaded                     0.331% px differ (max 38)
    glass refraction + refl-transparent splits    BYTE-IDENTICAL (max 0)
    bezier: text + shapes + sphere                2.88% px (max 112)

The triangle diffs are the accepted raw-depth tie noise (scattered silhouette /
coincident-edge pixels; visually identical). The bezier 2.88% is dominated by a
translucent circle and an opaque square swapping which is on top where they
overlap — an accepted coplanar-order swap, not an artifact — plus raw-depth edge
noise on the sphere and text. Adding bezier did not regress the triangle path
(pure-triangle parity unchanged). The default renderer (raster off) is
unaffected.

8.4 Phase-1 gate measurements
-----------------------------
Sort ~140 Mkeys/s end-to-end; candidates/pixel 0.7–5.5 (≈20x fewer tests than
the classic walk); refit-topology staleness 1.00–1.04 vs per-frame rebuild and
1.37–2.33x better than the current STBVH. See §2.


================================================================================
9. SETTINGS / TOGGLES
================================================================================

  ALGAN_HYBRID_RASTER (default 0)   Enable the raster front-end. Also forces
                                    surfaces to flat triangles.
  ALGAN_RASTER_SS     (default 1)   Screen-space rasterization vs per-pixel
                                    ray-cast (both correct; SS wins on high
                                    overdraw, ~6% slower on low overdraw).

  set_hybrid_raster(bool), set_raster_screen_space(bool) — programmatic setters.

Gate for engaging the front-end (`use_raster` in tracer.py): HYBRID_RASTER on,
num_pn == 0, (num_triangles > 0 or num_circuits > 0), not textured/sorted-legacy,
no mem-trim, no custom scatter, no shadows, near_clip <= 0, aa_level <= 1.
Fragment shading and refraction / refl-transparent splits ARE supported.


================================================================================
10. PLANNED FUTURE IMPROVEMENTS
================================================================================

Near-term (front-end):
  * Shadows through the raster path via a post-visibility deferred pass (group
    survivors, one shadow-ray batch), rather than the gate's current
    shadows-route-to-classic.
  * Arena-back the transient raster buffers (z-buffer, fragment arrays) so they
    stop competing with arena headroom and cannot trigger extra OOM splits.
  * Material/geometry-type-grouped shading of survivors: one partition of the
    survivor list, a handful of launches (distinct from the failed sorted-
    material pipeline, which paid per-event round trips inside a loop). Cuts the
    resolve kernel's register footprint.
  * Adaptive per-pair intersection mode: the host already knows each pair's
    bbox area vs coverage, so it could pick screen-space for high-overdraw pairs
    and ray-cast for low-overdraw ones, capturing both wins.
  * Escape the Taichi 64-arg ceiling structurally: pass one scene-descriptor
    ndarray instead of the current per-type array smuggling.

Acceleration structure (Phase 4):
  * Shared-topology binned-SAH tree with per-frame refit (§7). Explicit child-
    base index per sibling block; static geometry deduped to T=1; a single
    mixed-type any-hit tree for shadow rays.

Loop / scheduling (Phase 5):
  * Sync-free bounce loop: over-launch the compacted kernels at pool capacity
    with a device-side active-count read, eliminating the per-iteration
    count.item() host sync so the whole batch becomes one async stream.

Explicitly OUT of scope:
  * PN-patch rasterization. Surfaces render as flat triangles under raster; the
    curved-patch intersection is not worth a rasterizer.
  * The Monte Carlo path tracer (samples > 1) is untouched by this redesign.

Anti-goals / dead ends already measured (do not re-attempt):
  * Time-interpolated / OBB-in-node BVH — tightness dominates node count;
    interpolation nets ~wash and doubles node size.
  * Forcing the K-buffer into registers — adds register pressure, fatal to the
    already occupancy-starved kernel.
  * Byte-identical parity — the raw-depth ordering is a deliberate improvement.


================================================================================
11. FILE MAP
================================================================================

  raster_taichi.py        raster kernels: _ss_setup/_ss_pixel/_raycast_pixel,
                          _pair_pixel, _bez_pair_pixel, raster_tri_z,
                          raster_tri_count/write, raster_bez_count/write,
                          raster_first_shade (the resolve).
  raster_pipeline.py      host: _project_verts, _screen_bbox, _frame_pairs,
                          _frame_bez_pairs, raster_iteration_zero.
  tracer.py               use_raster gate; _run_wavefront_tiles(raster=...);
                          iteration-0 call into raster_iteration_zero; compact
                          secondary surface-event allocation.
  scene_builder.py        tri/bez_frame_valid, tri_frame_opaque,
                          bez_frame_lo/hi masks next to the BVH build.
  settings.py             HYBRID_RASTER, RASTER_SS + setters.
  renderer_settings.py    effective_triangle_primitive() (force-flat).
  wavefront_kernels_taichi.py
                          transient traversal event emission and immediate
                          general-wavefront event shading.
  stbvh.py                current spatio-temporal BVH (secondary rays).

  benchmarks/_rt2_*.py    all measurement, parity, and A/B scripts referenced
                          above; captures in benchmarks/_rt2_out/.


================================================================================
12. SECONDARY RADIANCE QUEUE REDESIGN
================================================================================

12.5 Remove the global K-buffer from secondary radiance rays — IMPLEMENTED
----------------------------------------------------------------------------

The maintained general deterministic wavefront no longer allocates six
`[continuation_pool, KBUF]` arrays beside the lifetime state of every ray.
At `KBUF = 4`, this removes 24 four-byte words, or 96 persistent bytes, from
all continuation-pool slots. The persistent slot falls from approximately
196 bytes to 100 bytes once pixel ownership and the two compaction-index banks
are included. Persistent secondary-radiance state now contains only:

  * origin and direction;
  * accumulated radiance and RGB throughput;
  * previous-hit, seam, base-distance, bounce and status state;
  * pixel ownership and continuation-pool bookkeeping.

For each host bounce iteration, the compacted active queue has length `A`.
The host opens a temporary arena scope and allocates two packed surface-event
arrays:

    hit_f [A, KBUF, 4]  = (distance, layer, barycentric/plane a, b)
    hit_i [A, KBUF, 2]  = (typed primitive index, hit flags)

`wavefront_traverse_events` retains the existing register-local KBUF gather and
writes events by active-queue ordinal rather than sparse continuation-pool slot.
`wavefront_shade` consumes the event batch immediately. The temporary arena
scope is then released before ray compaction, so the same bytes can be reused by
the next phase or bounce.

This preserves the established four-hit batching and exact ordering semantics.
A ray with more than KBUF same-direction transparent hits still re-enters
traversal after draining the batch, exactly as before; the redesign removes the
*permanent global storage*, not the existing depth-peeling policy. The packed batch costs the same 96 bytes per *active* ray, but is proportional
to the compact queue rather than continuation-pool capacity. In the hybrid
renderer the primary pass never allocates it, and unused continuation reserve
no longer carries six K-buffer arrays.

The unsupported legacy textured and material-sorted orchestrators continue to
use `wavefront_traverse` and their old persistent hit arrays. They are isolated
from the maintained `wavefront_traverse_events` / `wavefront_shade` path.
Automatic tile sizing still charges the conservative worst case where every
continuation slot is active, ensuring the exact-size temporary allocation cannot
overrun the render arena.
