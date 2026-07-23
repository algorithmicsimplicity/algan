# Algan Raytracer v2 — Hybrid Rasterization Front-End: Design Document

Status: in progress. The front-end (raster primary visibility, exact sparse
shadow queue, classic-exact fragment ordering) is built and validated behind
an opt-in toggle (`ALGAN_HYBRID_RASTER`, default OFF) while the classic
deterministic wavefront remains the default renderer. The secondary-ray
K-buffer removal (§7) is live on the DEFAULT path. This document explains the
current implementation, the reasoning behind each design decision, the
benchmark results that drove those decisions, and the planned future work
ranked by expected improvement (§13).

History note: the front-end was built in three pushes on 2026-07-18
(`fa7afd4` prototype → `f40cf76` feature-complete front-end → `61d177f`
secondary K-buffer removal). `61d177f` accidentally reverted most of
`f40cf76`'s front-end work (sparse shadow queue, PN preservation, opaque
bezier z-prepass, per-primitive alpha masks, the Taichi compile logger) while
landing the K-buffer change; that revert was identified and undone on
2026-07-19, merging both lines. If a source file disagrees with this
document, suspect a repeat of that failure mode first.

Author's note on scope: this is a *ground-up redesign* of the deterministic
(samples-per-pixel == 1) render path. Byte-for-byte identity with the old
renderer is explicitly NOT a goal — render tests are being re-validated, and
the new path may change semantics where that makes the output more accurate
or the renderer faster (the deltas are enumerated in §8; they are small
because the ordering relation is deliberately classic-exact, §4.5). The
Monte Carlo path tracer (samples > 1) is out of scope and untouched.


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
GPU — and it removes the K-buffer refill loop: all hits along a primary ray are
found in one pass, with depth capped only by the straight-ray safety limit
`MAX_SURFACES_PER_RAY` (256), not by KBUF.

Secondary rays (shadows, reflections, refractions) still ray-trace, so the BVH
is not going away; but for scenes without them the primary pass needs no BVH
at all.


================================================================================
2. MEASUREMENT GATES (Phase 1) — why we believed it would work
================================================================================

Before writing a kernel, three cheap measurements were run to de-risk the
design. All three passed. Scripts: `benchmarks/_rt2_sort_bench.py`,
`benchmarks/_rt2_capture.py` (+ `_rt2_overdraw.py`, `_rt2_refit_sah.py`).

2.1 Sort throughput (GTX 1050)
------------------------------
The design sorts transparent-fragment records by a 64-bit (pixel | depth) key.
`torch.sort` (cub radix on CUDA):

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
thin cylinders. Mitigations shipped: screen-space edge-function inside tests
(§6.1) and the behind-camera cull (§4.9); still open: per-pair adaptive
intersection mode and square-tile binning (§13).

2.3 Refit-topology staleness (for the acceleration structure)
-------------------------------------------------------------
Compared SAH expected-visit cost of a per-frame-rebuilt tree, a
union-centroid topology refit per frame, and the current STBVH:

    union-refit vs per-frame rebuild:  1.000–1.04 (mean, all scenes/types)
    current STBVH vs union-refit:      tri 1.37x, PN 1.67x, bez 1.78–2.33x worse

Refitting one topology across a 7–15-frame batch is essentially free of quality
loss, and beats the current spatio-temporal instance tree. This validates the
planned acceleration structure (§9), which is also the top-ranked future work
item (§13).


================================================================================
3. THE HYBRID RASTER FRONT-END — architecture
================================================================================

Files:
  algan/rendering/raytracing/raster_taichi.py     — GPU kernels (Taichi)
  algan/rendering/raytracing/raster_pipeline.py   — host orchestration (torch)
  algan/rendering/raytracing/scene_builder.py     — per-frame visibility masks
  algan/rendering/raytracing/tracer.py            — gate + iteration-0 wiring
  algan/rendering/taichi_runtime.py               — single ti.init entry point
                                                    + compile-timing logger

The front-end replaces the wavefront's *first* iteration (generate + first
traverse + first shade). Bounced continuations (reflection/refraction) it
spawns are handed to the unchanged classic wavefront loop for iterations >= 1.
Thus the front-end is a drop-in replacement for primary visibility only; all
the material, environment-map, and multi-bounce machinery is reused. Primary
shadows are handled by the front-end itself via a sparse event queue (§5).

The host tile is the wavefront's *linear ray tile* (a contiguous pixel range,
normally one or more row bands per frame), not a square screen tile. Per tile
(all torch/Taichi, scratch in one `memory.temp()` arena scope):

  0. PROJECT  (once per batch, not per tile) Build the triangle projection
              table `tri_screen [T, N, 10]`: continuous screen x/y of the 3
              vertices, reciprocal perspective divisors, and a 3-state
              validity flag (§4.6, §4.9).
  1. BIN      Per frame covered by the tile: emit fixed-size (primitive,
              pixel-chunk) pair rows from each primitive's clipped screen
              bbox — triangles from the projection table, circuits from their
              per-frame world AABB (8 corners projected). Opaque and
              transparent primitives bin separately; per-primitive
              texture-alpha certainty (§4.7) decides which side a textured
              triangle joins. Provably-behind primitives are culled (§4.9).
  2. Z-PREPASS Typed opaque visibility buffer, shared by proven-opaque
              triangles AND proven-opaque bezier circuits: an int64 atomicMin
              per pixel of the packed key
              `(depth_bin << 32) | (0xFFFFFFFF - layer)` (§4.2). The key is
              layer-aware, so coplanar layer semantics survive the atomic.
  3. COUNT    Per transparent pair, count surviving fragments (alpha-fetched:
              alpha-zero texels are dropped at emission, §4.8; strictly nearer
              than the pixel's opaque winner). Sizes the fragment list
              exactly — no atomic append, deterministic layout.
  4. WRITE    Emit (key, ref, ab) records at each pair's exact offset.
  5. ORDER    `_exact_fragment_order`: stable argsort by descending layer,
              then stable argsort by (pixel << 32 | depth_bin) — the classic
              transitive (depth-bin, descending-layer) relation (§4.5) —
              then build per-pixel CSR run offsets.
  6. SHADOWS  (only when shadows are on) Build the exact sparse shadow-event
              queue and trace it: one visibility value per accepted shading
              point per light, hard AND soft (§5).
  7. RESOLVE  One thread per tile pixel walks its ordered run front-to-back
              (the z-prepass winner appended as the terminal hit),
              alpha-composites with material/fragment shading and per-light
              visibility lookups, and spawns reflect/refract continuations
              into the shared wavefront pool (§4.10).

The remaining wavefront iterations then trace only the spawned continuations,
using the compact surface-event batches of §7.


================================================================================
4. DESIGN DECISIONS AND RATIONALE
================================================================================

4.1 Cull before the sort, not after
------------------------------------
The original sketch was: rasterize bbox → sort all fragments → null occluded.
But the sort is the bandwidth-heavy step, so misses must never reach it. The
exact intersection test and the alpha fetch are fused into fragment
*emission*: a fragment is appended only if the pixel's ray actually hits the
primitive, has non-zero sampled alpha, and clears the opaque z-buffer. Bbox
overdraw then costs only intersection tests, not sort traffic.

4.2 Typed layer-aware opaque z-prepass via packed atomicMin
-----------------------------------------------------------
Fragments from proven-opaque primitives skip the fragment list entirely: an
`atomicMin` per pixel keeps the nearest. The packed key is

    (floor(t / DEPTH_TIE_EPSILON) << 32) | (0xFFFFFFFF - layer)

`min` is commutative, so the winner is deterministic regardless of thread
order — and because the key bins depth exactly like the classic renderer and
breaks ties by *descending layer*, an opaque circuit coplanar with an opaque
triangle wins by the same rule the classic compositor uses. Triangle layers
are `layer_offset_triangles + prim`; circuit layers are the circuit index; the
resolve recovers the winning geometry type from the layer alone, so no second
winner buffer is needed. Exact t/barycentrics are recomputed only for the
terminal winner (`_terminal_z_hit`).

Both triangles (`raster_tri_z`) and proven-opaque bezier circuits
(`raster_bez_z`) feed this buffer, so a large filled 2D shape culls everything
behind it. Transparent emission culls against the buffer, so the sort only
ever sees translucent survivors: a fully-opaque scene does zero sorting; a
fade (everything translucent) routes everything to the sorted path, which is
correct and still one sort.

4.3 Two-level (primitive, chunk) pairs bound load imbalance
-----------------------------------------------------------
One thread per primitive iterating its whole bbox has terrible load imbalance
(a full-screen fade rectangle = millions of pixels in one thread). Instead each
primitive's clipped bbox is split into chunks of RASTER_CHUNK (=256) pixels;
one thread processes one (primitive, chunk) pair. This bounds per-thread work
and gives a natural OOM-retry granularity (§4.12).

4.4 Per-pixel SERIAL resolve, not a parallel prefix scan
--------------------------------------------------------
The compositing rules are not a clean associative scan: seam de-duplication
(edge-flagged triangle hits within a depth tolerance of the previous accepted
edge hit are dropped), and termination at the first *path-bending* surface
(refraction / metal reflection make everything behind them along the straight
ray invalid). A per-pixel thread walking its ordered run serially expresses
all of this directly — it is the classic shade drain loop minus the traversal,
with K limited only by MAX_SURFACES_PER_RAY. There are ~2M pixels/frame, so
parallelism is not the constraint, and each run is a genuine sequential
dependence chain anyway.

4.5 Ordering: the classic transitive (depth-bin, descending-layer) relation
---------------------------------------------------------------------------
An earlier prototype ordered fragments by raw f32 depth bits. That is a true
total order, but it diverges from the classic renderer wherever depths tie or
near-tie (coplanar circuits, text on panels, silhouettes), which showed up as
a 2.88% pixel diff dominated by coplanar-overlap swaps. The shipped ordering
instead reproduces the classic relation *exactly*:

    primary key:   floor(t / DEPTH_TIE_EPSILON)   (the classic depth bin)
    tie-break:     descending layer index

Note this is NOT the same thing as the classic K-buffer's pairwise
epsilon-comparison (which is non-transitive and order-sensitive): binning by
`floor(t/eps)` is a genuine total preorder, so the sorted result is
deterministic and independent of tile size, chunking, and build order — while
agreeing with the classic compositor everywhere the classic compositor's own
answer is well-defined. Host-side this is two stable argsorts
(`_exact_fragment_order`): by descending layer, then by (pixel, depth-bin).

4.6 Per-batch triangle projection table
---------------------------------------
`precompute_triangle_projection` builds one compact record per (frame,
triangle) — screen x/y of the three vertices, reciprocal perspective
divisors, validity flag — ONCE per render batch, vectorized in torch,
arena-backed. Every raster kernel phase (z, count, write, shadow-event build,
resolve) previously recomputed the same camera projection per (pair, phase);
now `_ss_setup` is a table load. Screen-space math: the screen coordinate of
a world point is a projective function with perspective divisor
d_i = dot(V_i − cam_o, n), n = pbx × pby; perspective-correct barycentrics
weight edge functions by 1/d_i; the 3D hit point H = Σ w_i V_i gives exact
distance and barycentrics. Numerically equivalent to ray-cast (verified worst
|Δt| ~5e-5, |Δbary| ~6e-5, `benchmarks/_rt2_ss_math_check.py`). A triangle
straddling the camera plane falls back to exact per-pixel Möller-Trumbore
(`_raycast_pixel`).

Both geometries also have a once-per-window *candidate bounds* table
(`precompute_circuit_screen_bounds`, kill-switch
`ALGAN_RASTER_BEZ_PRECOMPUTE=0`; `precompute_triangle_screen_bounds`,
kill-switch `ALGAN_RASTER_TRI_PRECOMPUTE=0`): screen bbox rows/columns and
the front/reach/class masks, batched over all frames — circuits from their
projected AABB corners, triangles from the projection table above. Candidate
emission then runs once per tile for all covered frames (`_window_pairs` +
`_class_pairs_flat`, shared table schema) instead of per (tile, frame) — the
per-frame `_frame_bez_pairs` path cost ~130 small tensor dispatches per call
and dominated host time on circuit-only scenes (tiny-scene render floor: ~8s
of a ~19s 300-frame MD render). Only the row-band clamp of the bbox is
tile-dependent; flattening (frame, primitive) row-major preserves the exact
pair-row order the per-frame loop produced, so the fragment sort's
tie-breaking (and thus output) is byte-identical
(`benchmarks/_raster_bez_pre_parity.py`).

4.7 Per-primitive texture-alpha certainty
-----------------------------------------
A color texture with an alpha channel can cut a surface, making an
"interval-opaque" primitive effectively translucent. The merge proves alpha
opacity per texture (`_texture_alpha_is_opaque`: no texture, or no alpha
channel, or all alpha >= 1-1e-6 ⇒ cannot cut) and stores an exact
per-primitive mask in merged triangle order (`tri_alpha_uncertain`). Only the
genuinely uncertain primitives are demoted to the transparent path; everything
else keeps opaque z-culling. (An aggregate flag would let a single cutout
texture disable z-culling scene-wide; the per-primitive mask was restored
after `61d177f` regressed exactly that.)

4.8 Fragment tagging
--------------------
Fragment records are three SoA arrays: `frag_key` (pixel << 32 | f32 depth
bits, for run grouping and exact t recovery), `frag_ref` (triangle primitive
index >= 0, or `-(circuit << 1 | in_border) - 1` for a bezier fragment — the
border flag rides in the packed ref, no separate flags array), and `frag_ab`
(barycentrics or plane u,v). The resolve decodes the sign.

4.9 Behind-camera cull
----------------------
A primitive whose vertices (or AABB corners) are ALL behind the camera-origin
plane is provably unhittable by a forward primary ray: any point of the
primitive is a convex combination of its vertices, so its plane projection is
<= 0, while every ray point at t > 0 projects > 0. Without the cull, such a
primitive fell back to the conservative full-row-band bbox and became a
full-screen candidate scan (~8k pairs/frame at HD, per primitive — a real
pathology when the camera moves past geometry). The projection table's flag
is 3-state (1 = all-front, SS valid; 0 = straddling/degenerate, full-window
ray-cast fallback; -1 = provably behind, culled at binning), and the bezier
path culls on `front.any() == False` over the 8 AABB corners.

4.10 In-place bounce continuations
----------------------------------
The resolve thread for pixel r owns ray slot r. A reflected/pass-through
continuation is written back into slot r (status ACTIVE) exactly like the
existing fused-generation path; split branches (glass refraction,
semi-transparent reflectors) atomically append to the shared continuation pool
via `_reserve_continuation_slot`. The pool's overflow-retry (halve the
primaries, retry the tile) is unchanged. The front-end's postcondition matches
the classic first traverse+shade exactly: `pix_accum` holds every retired
pixel's colour + leftover background weight, bounced pixels are ACTIVE, and
the classic loop takes it from there. Free pool slots are pre-marked DONE by
the host so a full-pool compaction finds exactly the spawned continuations.

4.11 Empty-pixel fast path (retired-empty pre-fill + host pair flags)
---------------------------------------------------------------------
On sparse screens most resolve threads have nothing to shade, yet each one
paid ray generation plus 8 strided state writes (background weight + DONE
status) -- ~15 ms/tile of `raster_first_shade` GPU time on an EMPTY screen,
the dominant kernel of the tiny-scene render floor. Two paired fixes
(both default ON, byte-identical, `benchmarks/_raster_empty_skip_parity.py`):

  ALGAN_RASTER_EMPTY_SKIP: the host pre-fills every primary's `pix_accum`
  row with the retired-empty result `[0,0,0,0, 1,1,1]` (one broadcast copy
  replacing the zero-fill; the pool is already pre-marked DONE), so the
  committed state of an empty pixel exists before the kernel runs. A
  `prefill` compile-time template makes empty pixels (no fragments, no
  z-winner, no environment map) exit before ray generation with ZERO
  writes; retiring pixels *store* their leftover weight into cols 4-6
  instead of accumulating onto a zero base, and bouncing pixels store the
  columns back to zero (each pixel has exactly one writer in iteration 0,
  so stores are race-free and value-identical). A tile with no candidate
  pairs at all skips the resolve and shadow-event launches entirely. The
  toggle is read once per batch so the host fill and kernel template can
  never disagree. Measured (size-0 circuit, MD 300 frames, warm paired
  A/B together with PAIR_FLAGS): raster_first_shade 4.97 s -> 1.75 s
  (GPU sync 3.66 s -> 0.46 s), wall 11.9 s -> 8.8 s (-26%).

  ALGAN_RASTER_PAIR_FLAGS: the candidate-bounds precomputes additionally
  reduce one conservative host-side (opaque, translucent) any-candidates
  bool per frame (`_class_any_flags`, one `.cpu()` per window). Per tile,
  `_window_pairs` skips its ~20 tensor dispatches -- and the synchronizing
  `.nonzero()` inside `_class_pairs_flat` -- for every (tile, class) whose
  covered frames provably have no candidates (the per-tile reach mask is
  contained in the per-frame reach base, so a False flag is exact). This
  replaces up to 4 per-tile GPU->host syncs with one per-window transfer.

4.12 Memory model
-----------------
Persistent per-ray state is the K-buffer-free ~100-byte pool slot of §7,
arena-backed. Raster transient scratch (projection table aside, which lives
for the whole batch) — z-buffer, pair rows, fragment arrays, CSR runs, the
shadow-event queue — is ALSO arena-backed, allocated inside one
`memory.temp()` scope that is released before the bounce loop runs, so raster
scratch and the per-iteration surface-event batches of §7 reuse the same
bytes. The one exception is torch's radix-sort/index scratch, which PyTorch
allocates internally and cannot write into an arena view.

Raster scratch scales with the tile's fragment volume, which up-front tile
sizing cannot know. A failed attempt (arena `InsufficientMemoryException` or
torch `OutOfMemoryError`) is therefore caught at the TILE level: restore the
arena pointer, halve the primaries, retry — doubling continuation headroom
and halving scratch per attempt, without discarding the whole frame window.
Only a single-pixel failure escalates to `OutOfRenderMemory` (window halving).


================================================================================
5. SHADOWS — THE EXACT SPARSE SHADOW-EVENT QUEUE
================================================================================

Primary shadows run inside the front-end; they do NOT route the batch back to
classic, and they support everything the classic inline path supports (all
MAX_SHADOW_LIGHTS lights; hard shadows; soft shadows from point/spot
`shadow_radius` and directional `shadow_angle` via the same deterministic
golden-angle fan of SOFT_SHADOW_SAMPLES rays).

Two-kernel design:

  raster_shadow_event_build — replays the resolve's exact walk order,
      seam-de-duplication and transport/termination decisions per pixel, and
      *accepts* only the shading points the resolve will actually light:
      lit triangle fragments on the straight ray plus the terminal z-winner.
      Each accepted point reserves one event row (position, shading normal,
      face normal, frame) via an atomic counter; `frag_shadow_id[fragment]`
      and `z_shadow_id[pixel]` record the event id (-1 = never lit, e.g.
      bezier fragments, which keep their sampled colour and receive no
      shadows — but bezier geometry still occludes shadow rays).

  raster_shadow_trace — one thread per (event), tracing per light: any-hit
      against the full three STBVHs, one visibility float per (event, light)
      row; soft lights average a golden-angle fan. No packed-bit budget, no
      fragment-slot cap, no light cap: `shadow_vis` is a dense
      [events, lights] f32 table.

The resolve then looks up `shadow_vis[event, light]` for each shaded point.
Because the event build mirrors the resolve's acceptance decisions, no
visibility is computed for fragments that seam-dedup or termination will
discard, and — unlike a fixed per-pixel slot budget — arbitrarily deep
translucent stacks receive correct shadows on every lit surface.

Why sparse events instead of packed per-pixel bits: an earlier design packed
(slot × light) occlusion bits into one int32 per pixel — 4 shading points × 8
lights. It was simple but capped: deep stacks went unshadowed past slot 3,
soft lights and >8-light scenes had to route the whole batch back to classic.
The sparse queue costs one indirection and one compact trace launch, sized by
*accepted lit points* (≈ depth-complexity × lit fraction, far below raw
fragment count), and removes every cap. The packed-bit design is retired
(§14).


================================================================================
6. GEOMETRY PATHS
================================================================================

6.1 Flat triangles
------------------
Both the z-prepass and the transparent path test candidate pixels against the
triangle. Two intersection modes, selected by `ALGAN_RASTER_SS`:

  Screen-space (default ON): per-pair table load (§4.6), per pixel three edge
  functions + perspective-correct barycentrics. Wins in proportion to bbox
  overdraw — the edge-function sign test rejects a miss far more cheaply than
  ray-gen + Möller-Trumbore. Kernel-isolated A/B: dense thin-triangle mesh
  (~10x overdraw) 1.36x faster; low-overdraw spheres ~6% slower (no misses to
  save on). Default ON because the high-overdraw case is both the expensive
  one and the realistic one.

  Ray-cast (`ALGAN_RASTER_SS=0`, and always for camera-plane straddlers):
  generate the pixel's world ray, Möller-Trumbore.

6.2 Bezier circuits (2D shapes, text glyphs)
--------------------------------------------
A circuit is a planar cubic-bezier outline embedded in 3D via a plane
(center + normal + basis u/v). Candidate pixels cast the ray, intersect the
plane, project to (u, v), and run the existing `_bezier_point_metrics`
(crossing count for fill + nearest-edge distance for border/outline) — the
classic `_collect_hits` bezier branch, ported unchanged; base_dist is 0 for
primaries so the screen-constant border width uses t directly.

Proven-opaque circuits participate in the typed z-prepass (§4.2) and cull
geometry behind them; translucent/bordered circuits ride the ordered fragment
stream. Circuits keep their sampled colour (never material-shaded — a
deliberate deviation matching the classic renderer), take reflectivity / IOR /
transmission from `circuit_meta`, use `_bezier_normal`, and their continuation
is the "thin pane" case (reflect into a split slot, transmit unbent into the
pass-through).

Why bezier and not PN: bezier (text, 2D shapes) is the bulk of real
explanatory-math content, so without it the front-end helps almost no real
scene (a single Text label silently routed early benchmarks to classic and
produced bogus numbers — see §10's measurement caveat).

6.3 PN patches: preserved, classic fallback — NOT flattened
-----------------------------------------------------------
PN-patch rasterization is out of scope (the curved-patch intersection is not
worth a rasterizer). The policy is: geometry construction must NOT depend on
whether a later render batch happens to qualify for the raster front-end.
`effective_triangle_primitive()` returns the configured primitive class
unchanged; `Surface._uses_pn_triangles()` is raster-agnostic. A batch that
contains PN geometry simply fails the `num_pn == 0` gate and renders through
the classic wavefront, with its PN quality intact.

The rejected alternative (force-flat: tessellate every surface flat whenever
`HYBRID_RASTER` is on) was shipped briefly and reverted: any batch that falls
back to classic for an unrelated reason (custom scatter, mem-trim, near clip,
in-place AA) would silently render *flat* geometry through the *slow* path —
quality lost with zero raster benefit, controlled by a global toggle acting at
mob-construction time. Since flat triangles are already Algan's default
primitive (PN is opt-in), the practical cost of preserving PN is only that
deliberately-PN scenes don't engage the raster front-end — which is exactly
what the author of such a scene asked for.


================================================================================
7. SECONDARY-RAY STATE: THE K-BUFFER REMOVAL
================================================================================

The general deterministic wavefront no longer allocates six
`[continuation_pool, KBUF]` arrays beside the lifetime state of every ray.
At KBUF = 4 this removes 24 four-byte words — 96 bytes — from every
continuation-pool slot: the persistent slot falls from ~196 to ~100 bytes
(ro 12 + rd 12 + acc 16 + sca 28 + int 20 + pixel 4 + two compaction-index
banks 8). Persistent secondary-radiance state is now only: origin/direction;
accumulated radiance + RGB throughput; previous-hit, seam, base-distance,
bounce, status and hit-count words; pixel ownership and pool bookkeeping.

Per host bounce iteration with compacted active-queue length A, the tracer
opens a temporary arena scope and allocates two packed surface-event arrays:

    hit_f [A, KBUF, 4]  = (distance, layer, barycentric/plane a, b)
    hit_i [A, KBUF, 2]  = (typed primitive index, hit flags)

`wavefront_traverse_events` keeps the register-local KBUF gather (unchanged
`_collect_hits`, Matrix-Pencil PN solver included) and writes events by
active-queue ordinal instead of sparse pool slot; `wavefront_shade` consumes
the batch immediately; the scope is released before compaction, so the same
bytes are reused by the next phase or bounce (and by the raster scratch scope,
which closes before the loop starts). The four-hit batching and exact ordering
semantics are preserved — a ray with more than KBUF same-direction transparent
hits still re-enters traversal after draining the batch. Validated: repo
pixel tests pass on the default path post-change.

Sizing: automatic tiling still charges the conservative worst case (every
pool slot active ⇒ the same 96 B/slot as before, `(25 + 6*KBUF) * 4` in
`primitives._set_raytrace_memory_estimates` and
`tracer._wavefront_state_bytes_per_primary`), so the exact-size transient
batch can never overrun the arena. In the hybrid renderer the primary pass
never allocates a hit batch at all, and unused continuation reserve no longer
carries six K-buffer arrays.

The unsupported legacy textured and material-sorted orchestrators keep
`wavefront_traverse` and pool-wide K-buffers (`_alloc_wavefront_state
(global_hits=True)`); they are isolated from the maintained path. The
deferred `wavefront_shadow` kernel (never launched; measured slower than
inline) was retargeted to the event-batch ABI — its docstring carries the
host contract (ordinal-indexed `rs_vis[num_active]`) should it ever be
revived.


================================================================================
8. SEMANTICS DELTAS FROM THE CLASSIC RENDERER
================================================================================

The front-end is intentionally not byte-identical, but the deltas are small
and enumerable (the ordering relation itself is classic-exact, §4.5):

  * Strict opaque z-cull: a transparent fragment whose (depth-bin, layer) key
    exactly equals the opaque winner's is culled (the comparison is strict).
    If a coplanar decal exactly on an opaque surface ever needs to survive,
    revisit with <= plus resolve-side ordering.

  * Shadow shading points are reconstructed from raster-side hit math
    (projection-table barycentrics, ~5e-5 numeric agreement with the classic
    traverse) and soft-fan origins differ by the same epsilon, so shadow
    *boundary* pixels can move by a pixel. Measured (480x270, 8 frames,
    engagement-asserted): hard shadows max|diff| 28, 0.666% of pixels > 2;
    soft shadows + opaque/translucent circuits + text: max|diff| 29, 0.689%.
    (`benchmarks/_rt2_raster_shadow_parity.py`,
    `benchmarks/_rt2_raster_soft_bez_parity.py`.)

  * Alpha-zero texels are dropped at emission (§4.8); the classic path
    composites them with zero contribution. Output-identical, but
    MAX_SURFACES_PER_RAY counts differently in pathological all-invisible
    stacks.

  * Refraction / refl-transparent split parity was BYTE-IDENTICAL on the
    earlier prototype and the continuation code is unchanged; re-verify with
    `_rt2_raster_refract_parity.py` after any resolve edit.


================================================================================
9. ACCELERATION STRUCTURE (secondary rays)
================================================================================

Classic (`stbvh.py`, the default): a spatio-temporal BVH per geometry type.
Time is a fourth dimension; primitives are adaptively segmented into
(frame-interval, union-bound) instances, ordered along a 4D Morton curve, and
packed into an implicit 4-ary tree with sibling-block nodes (one aligned
fetch tests a whole sibling group). Triangles use a median-split build;
PN/bezier use Morton. Used by the wavefront (secondary rays), the raster
shadow trace, and classic-path primary rays.

Its weakness (measured, §2.3): at the confirmed-optimal tightness=1.0, moving
geometry segments to near-per-frame instances, so the tree is ~10x larger than
the primitive count and every ray wades through mostly other-frames' instances
gated out by frame-interval tests.

BUILT (2026-07-19, opt-in `ALGAN_BVH_REFIT`, default OFF while soaking):
shared-topology binned-SAH tree with per-frame refit (`refit_bvh.py`). ONE
binned-SAH topology per batch over the N ever-visible primitives (not
instances), built level-synchronously in vectorized torch (one batched
16-bin SAH split pass per binary level across every node of that level;
log2(ARITY) binary passes collapse directly into ARITY-wide sibling blocks;
depth is budgeted against the kernels' 16-deep sibling stack by forced median
splits). The tree is *unbalanced*, which the implicit heap cannot express, so
each block stores **explicit per-(frame, child) int32 link words** in the
lanes the classic tspan occupied: -1 = absent child OR subtree invisible
this frame (an empty box cannot gate -- the slab test min/max-normalizes each
axis, so inverted bounds still pass); sign bit = leaf (bits 0-29 the
primitive, bit 30 its *per-frame* full-opacity flag -- exact, vs the classic
per-interval flag); >= 0 = the child's block index. Bounds are refit per
frame as one vectorized `[T, blocks-in-level, ARITY, 3]` reduction per tree
level (static geometry dedupes to T = 1); blocks flatten to
`[Tb * num_blocks, 8, ARITY]`, same dtype rules (f16 conservative rounding)
as classic.

ABI: `RefitBVH` quacks like `STBVH` -- the same five tensor fields, with
`first_leaf` carrying `num_blocks` and the leaf-slot arrays as 1-element
stubs -- so every `(blocks, node_miss, leaf_prim, leaf_tspan, first_leaf)`
launch quintuple, the arena upload and the memory accounting are unchanged.
The kernels select the walk with ONE compile-time `refit` template threaded
through `_test_children`/`_collect_hits`/`_shadow_occluded`/`_transmittance`/
`_nearest_surface(_g)` and every kernel that launches them (wavefront
traverse/shade, raster shadow trace, both Monte Carlo megakernels), so both
modes coexist in one process (in-process A/B). All six trees of a batch
(3 full + 3 opaque-prepass) are built the same kind by
`scene_builder._build_accel`; the tree object's *type* selects the template
at launch, never the live toggle. The unsupported legacy textured/sorted
orchestrators stay classic (`refit_bvh_active` gates them out).

Validated: `benchmarks/_rt2_refit_build_check.py` (structural + conservative-
walk superset checks on randomized moving scenes with visibility holes);
`benchmarks/_rt2_refit_parity.py` -- animated multi-frame renders, classic vs
refit, BYTE-IDENTICAL (max|d| = 0) on all five configs: tri (opaque +
translucent + mid-batch spawn), bez, hard shadows, soft shadows, glass
refraction.

Still planned: a single mixed-type any-hit tree for shadow rays (they
currently walk three trees serially).


================================================================================
10. BENCHMARK RESULTS
================================================================================

All perf numbers are kernel-isolated (sync-fenced timing of only the ray-traced
render call), warm, alternating in one process, GTX 1050 (Pascal, 4 GB).
Wall-clock at low resolution is useless here — prep+video-encode dominate — so
comparisons are at MD/HD where the GPU render dominates.

ERA CAVEAT: the speed table below was measured on the `fa7afd4`-era prototype
kernels. The shipped front-end since gained the projection table (faster),
alpha-filtered emission (faster on cutout content), classic-exact ordering
(one extra stable sort, host-side), and the sparse shadow queue (new
capability). Directionally the numbers hold — the win is the removal of
per-pixel BVH traversal, which is unchanged — but re-measure before quoting
them (`_rt2_raster_kp.py`, `_rt2_raster_nn_kp.py`).

IMPORTANT measurement caveat (cost real hours): the raster gate requires
eligible geometry. Before bezier support, a single `Text(...)` label silently
routed the whole batch to classic and early neural_net numbers were
classic-vs-classic and bogus. All benchmarks now assert engagement (a counter
wrapped around `raster_iteration_zero`) and the profiler cross-checks (raster
kernels present, wavefront_generate absent).

10.1 Speed — raster vs classic (engagement-verified, prototype era)
-------------------------------------------------------------------
    scene                                    resolution   raster vs classic
    20 flat-tri spheres                      HD 1920x1080      2.26x
    dense flat-tri neural_net (no text)      MD 1280x720       3.27x
    full neural_net (net + Text label)       MD 1280x720       2.75x
      (previously ZERO benefit — the text silently forced classic:
       classic 8.70s -> raster 3.16s.)

The win scales with both pixel count and BVH depth/primitive count. Raster
kernel times are also far tighter run-to-run than classic (less thermal
sensitivity).

10.2 Screen-space vs ray-cast
-----------------------------
    scene                                overdraw   SS vs ray-cast
    20 spheres HD                          low          0.94x  (SS ~6% slower)
    dense flat neural_net MD               ~10x         1.36x  (SS 36% faster)
    full neural_net (net + text) MD        mixed        1.11x

10.3 Parity (engagement-verified, current front-end)
----------------------------------------------------
    test                                          result
    hard shadows (sphere over ground, 1 light)    0.666% px > 2 (max 28)
    soft shadows + opaque sq + transl circle
      + Text, shadow_radius=0.6                   0.689% px > 2 (max 29)
    PN batch under HYBRID_RASTER                  raster did NOT engage;
                                                  classic render correct
    default path (raster OFF) repo pixel tests    basic / text / shapes PASS
    glass refraction + refl-transparent splits    BYTE-IDENTICAL (max 0,
                                                  prototype era; §8)

10.4 Phase-1 gate measurements
------------------------------
Sort ~140 Mkeys/s end-to-end; candidates/pixel 0.7–5.5 (≈20x fewer tests than
the classic walk); refit-topology staleness 1.00–1.04 vs per-frame rebuild and
1.37–2.33x better than the current STBVH. See §2.


================================================================================
11. SETTINGS / GATE
================================================================================

  ALGAN_HYBRID_RASTER (default 0)   Enable the raster front-end.
  ALGAN_RASTER_SS     (default 1)   Screen-space intersection from the
                                    projection table vs per-pixel ray-cast
                                    (both correct; SS wins on high overdraw).
  ALGAN_BVH_REFIT     (default 0)   Build the shared-topology binned-SAH
                                    refit BVH instead of the classic STBVH
                                    (§9; all render paths honor it via the
                                    compile-time ``refit`` template).
  ALGAN_RASTER_EMPTY_SKIP (default 1)  Retired-empty pre-fill + resolve
                                    early-out / launch skip (§4.11;
                                    byte-identical kill-switch).
  ALGAN_RASTER_PAIR_FLAGS (default 1)  Host per-frame candidate-class flags
                                    skipping empty per-tile pair emission
                                    and its `.nonzero()` syncs (§4.11).

  set_hybrid_raster(bool), set_raster_screen_space(bool),
  set_refit_bvh(bool) — programmatic.

Gate for engaging the front-end (`use_raster` in tracer.py): HYBRID_RASTER on,
merged visibility masks present, (num_triangles > 0 or num_circuits > 0),
num_pn == 0 (PN batches keep classic, §6.3), not textured/sorted-legacy, no
mem-trim, no custom scatter, near_clip <= 0, aa_level <= 1.

SUPPORTED through the front-end: fragment shading, refraction and
refl-transparent splits, environment maps, far clip, hard AND soft shadows
with all MAX_SHADOW_LIGHTS lights (§5). NOT gated on shadows or light count.


================================================================================
12. FILE MAP
================================================================================

  raster_taichi.py        kernels + funcs: _order_key/_frag_t (packed keys),
                          _pack_bez_ref/_decode_bez_ref/_decode_z_layer,
                          _ss_setup/_ss_pixel (projection-table SS),
                          _raycast_pixel, _pair_pixel, _bez_pixel_hit,
                          raster_tri_z / raster_bez_z (typed z-prepass),
                          raster_tri_count/write, raster_bez_count/write,
                          _terminal_z_hit, _tri_shadow_normals,
                          raster_shadow_event_build, raster_shadow_trace,
                          raster_first_shade (the resolve).
  raster_pipeline.py      host: precompute_triangle_projection, _screen_bbox,
                          _class_pairs, _frame_pairs, _frame_bez_pairs,
                          _exact_fragment_order, raster_iteration_zero.
  tracer.py               use_raster gate; tri_screen precompute;
                          _run_wavefront_tiles(raster=..., global_hits=...)
                          incl. the raster tile OOM retry; iteration-0 call
                          in a memory.temp() scope; per-iteration compact
                          surface-event batches (§7).
  wavefront_kernels_taichi.py
                          wavefront_traverse_events + event-batch
                          wavefront_shade (§7); legacy wavefront_traverse
                          retained for the unsupported orchestrators.
  scene_builder.py        tri/bez_frame_valid, tri/bez_frame_opaque,
                          tri_alpha_uncertain (per-primitive), bez_frame_lo/hi
                          masks next to the BVH build.
  refit_bvh.py            RefitBVH + build_refit_bvh: level-synchronous
                          binned-SAH topology (batch-union boxes, ARITY-wide
                          blocks with explicit per-(frame, child) links) +
                          the vectorized per-frame bound refit (§9).
  settings.py             HYBRID_RASTER, RASTER_SS, BVH_REFIT,
                          RASTER_EMPTY_SKIP, RASTER_PAIR_FLAGS + setters
                          (refit_bvh_active gates the legacy orchestrators
                          back to classic).
  renderer_settings.py    effective_triangle_primitive() (raster-agnostic;
                          returns the configured class, §6.3).
  taichi_runtime.py       single ti.init entry point (init_taichi), tuned
                          runtime kwargs, sync_devices, and the Taichi
                          compile-timing logger (ALGAN_LOG_TAICHI_COMPILES,
                          ALGAN_TAICHI_COMPILE_LOG).

  benchmarks/_rt2_*.py    measurement, parity and A/B scripts referenced
                          above; captures in benchmarks/_rt2_out/.
                          _rt2_raster_shadow_parity.py (hard shadows),
                          _rt2_raster_soft_bez_parity.py (soft + bezier + PN
                          fallback) are the engagement-asserted parity gates.
                          _raster_empty_skip_parity.py (byte-identity of the
                          §4.11 empty-pixel fast paths, incl. bounce/split,
                          shadow and env-map interplay).


================================================================================
13. FUTURE WORK — ordered by expected improvement
================================================================================

Ranked by expected wall-clock improvement on real (moving, mixed-content)
scenes, with the evidence basis for each rank. Measured numbers where they
exist; ranks without numbers are marked (est.).

  1. Shared-topology binned-SAH refit BVH (§9). BUILT 2026-07-19, opt-in
     `ALGAN_BVH_REFIT`, byte-identical on all parity configs (see §9).
     Motivation (measured): secondary rays — shadows, reflections,
     refractions — and every classic-fallback batch still pay traversal,
     which is ~85% of classic GPU time; the classic STBVH is 1.37–2.33x
     worse in SAH expected-visit cost than a refit topology; refit staleness
     across a batch is <= 1.04. Benefits ALL paths, including the raster
     front-end's shadow/bounce stages. Remaining work: measure the traversal
     win on real scenes (`benchmarks/_rt2_refit_ab.py`), then flip the
     default ON.

  2. Re-baseline and default-ON the raster front-end. Measured 2.26–3.27x on
     qualifying scenes — but only scenes that opt in get it. Work: re-run the
     render-test suite under HYBRID_RASTER=1, accept/re-baseline the §8
     deltas, re-measure §10.1 on the shipped kernels, then flip the default.
     No new engineering; highest value-per-effort after item 1.

  3. Single mixed-type any-hit tree for shadow rays (part of the §9 rebuild).
     Shadow rays currently walk three trees serially and need only any-hit;
     shadow-heavy scenes are traversal-bound (the deferred-shadow experiment
     confirmed traversal dominates that stage). (est.: large on shadowed
     scenes, nil elsewhere.)

  4. Material/geometry-type-grouped shading of raster survivors: partition
     the survivor list once, launch a handful of lean shade kernels instead
     of one register-heavy resolve monolith (distinct from the failed
     sorted-material pipeline, which paid per-event round trips inside the
     bounce loop). The resolve inherits the megakernel's occupancy ceiling
     (21–25%), so register relief has measured precedent as the binding
     constraint. (est.: moderate-to-large on shade-heavy scenes.)

  5. Adaptive per-pair intersection mode: the host already knows each pair's
     bbox area vs projected coverage, so it can pick screen-space for
     high-overdraw pairs and ray-cast for low-overdraw ones. Captures both
     measured wins (+36% and +6% on their respective scene classes) instead
     of trading them. Small, cheap, well-understood.

  6. Square screen-tile binning (replace the linear row-band tile): better
     projection reuse and cache locality for tall/thin coverage; also makes
     chunk bboxes compact. (est.: moderate on high-overdraw scenes; medium
     effort — touches binning, chunk iteration and run layout.)

  7. Sync-free bounce loop: over-launch the compacted kernels at pool
     capacity with a device-side active count, eliminating the per-iteration
     count.item() host sync so a whole batch becomes one async stream.
     Ranked low because the tile-auto study showed per-launch host cost is
     already hidden by async overlap in unprofiled runs; the win is latency
     on many-bounce scenes. (est.: small-moderate.)

  8. Near-plane clipping of camera-plane-straddling primitives (replace the
     full-row-band bbox fallback that remains after the §4.9 behind-cull).
     Rare case; bounded cost today. (est.: small.)

  9. Scene-descriptor ndarray to escape the Taichi 64-arg ceiling: pass one
     descriptor array instead of per-type array smuggling. Maintainability
     enabler (unblocks items 4 and 6); ~zero direct perf.
     The *perf* half of this item -- per-launch Python argument marshalling
     (~0.15-0.3 ms per ndarray arg) -- is DONE without kernel changes:
     utils/taichi_fast_launch.py caches a launch plan per template
     instantiation and replays only the C++ set_arg calls (measured
     2.85x per launch on raster_bez_z, benchmarks/_taichi_fast_launch_kp.py;
     byte-identical + per-launch instantiation verification,
     benchmarks/_taichi_fast_launch_check.py; kill-switch
     ALGAN_TAICHI_FAST_LAUNCH=0). What remains of item 9 is the
     maintainability/64-arg-ceiling part, plus the C++ pybind per-arg floor
     (~0.8 ms/launch) that only a real descriptor array can remove.


================================================================================
14. ANTI-GOALS / DEAD ENDS (measured or decided — do not re-attempt)
================================================================================

  * PN-patch rasterization — the curved-patch intersection is not worth a
    rasterizer; PN batches take the classic path with geometry intact (§6.3).
  * Force-flat under raster — shipped briefly, reverted: silently degrades
    PN scenes that fall back to classic for unrelated reasons (§6.3).
  * Packed per-pixel shadow bits (4 slots × 8 lights in an int32) —
    superseded by the exact sparse event queue (§5): the budget capped stack
    depth and light count and forced soft-shadow scenes back to classic.
  * Raw-f32-depth fragment ordering — replaced by the classic-exact
    transitive (depth-bin, descending-layer) relation (§4.5); the raw order
    swapped coplanar layers (2.88% px on the bezier parity scene).
  * Time-interpolated / OBB-in-node BVH — tightness dominates node count;
    interpolation nets ~wash and doubles node size.
  * Forcing the K-buffer into registers — adds register pressure, fatal to
    the occupancy-starved kernel. (The K-buffer is now gone from persistent
    state entirely, §7 — via compact transient batches, not registers.)
  * Deferred shadow kernel for the general wavefront — measured ~7% slower
    (traversal-bound); kernel retained unused with its host contract
    documented.
  * Byte-identical parity as a front-end goal — the semantics deltas of §8
    are deliberate; parity scripts bound them instead.
  * The Monte Carlo path tracer (samples > 1) is untouched by this redesign.
