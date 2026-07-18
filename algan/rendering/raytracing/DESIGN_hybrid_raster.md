# Algan Raytracer v2: Hybrid Raster Primary Frontend

Status: implemented behind `ALGAN_HYBRID_RASTER` (default off). The frontend
replaces iteration zero of the deterministic one-sample-per-pixel wavefront
renderer. The Monte Carlo renderer is unchanged.

This document describes the code as it exists after the first redesign pass,
including its exact feature gates, memory contract, known limitations, and the
next architectural steps. It is intentionally implementation-focused; old
benchmark numbers are omitted because they are not part of the correctness
contract.

## 1. Goals and scope

The classic deterministic renderer generates one camera ray per pixel, traverses
separate triangle, PN-patch, and Bezier STBVHs, gathers a four-entry K-buffer,
shades the entries, and repeats traversal when a straight ray has more than four
surfaces. Primary traversal is expensive and particularly poor during fades,
where many surfaces become translucent simultaneously.

The hybrid frontend exploits the coherence of camera rays. Primitives are
projected to screen bounds and scatter candidate pixel work. Exact intersections
are evaluated only inside those bounds. Proven-opaque surfaces populate a typed
visibility buffer; all remaining accepted fragments are ordered per pixel and
resolved front-to-back. Reflected and refracted branches then enter the classic
secondary-ray wavefront.

The frontend therefore removes:

- Primary BVH traversal for eligible batches.
- Primary K-buffer refills through transparency stacks.
- Permanent K-buffer allocation for primary rays that retire during raster
  resolve.

It does not yet replace secondary radiance traversal or its global K-buffer.

## 2. Eligibility and fallback

`tracer.py` engages raster only when all of the following hold:

- `HYBRID_RASTER` is enabled.
- At least one flat triangle or Bezier circuit is present.
- No PN patches are present.
- The legacy textured/material-sorted routes are inactive.
- Memory trimming and custom scatter are inactive.
- Near clipping is disabled.
- In-place AA is at one sample.
- Hard shadows, emitter-radius soft shadows, and packed area-light samples are
  supported by the sparse primary shadow queue.

Fallback is behavior-preserving. Enabling the raster setting no longer changes
surface construction or tessellation. In particular, PN surfaces remain PN
surfaces; a PN batch simply fails the frontend gate and uses classic primary
traversal.

## 3. Current data flow

The frontend processes the same linear ray tile used by the wavefront host. A
linear tile is normally a contiguous row band, not a conventional fixed square
screen tile.

For each tile:

1. **Pair construction**
   - Per-frame primitive screen bounds are clipped to the current row band.
   - Each bound is split into `RASTER_CHUNK` candidate groups.
   - A pair row stores primitive, frame, clipped rectangle, and flattened chunk
     offset.

2. **Typed opaque visibility pass**
   - Proven-opaque flat triangles call `raster_tri_z`.
   - Proven-opaque Bezier circuits call `raster_bez_z`.
   - Exact hits atomically minimize one packed key per pixel.

3. **Transparent/uncertain count pass**
   - Exact triangle or circuit intersection is evaluated.
   - Texture/vertex/circuit alpha is sampled.
   - Alpha-zero hits and hits ordered behind the opaque winner are rejected.
   - One count per pair sizes the fragment arrays exactly.

4. **Transparent/uncertain write pass**
   - The exact test and alpha fetch are repeated.
   - Accepted records are emitted at prefix-summed pair offsets.

5. **Deterministic ordering**
   - Host sorting produces runs ordered by pixel, depth bin, then descending
     layer, matching the classic transitive order.
   - One CSR-style `run_offsets[pixel:p+2]` table identifies each run.

6. **Optional exact sparse shadow queue**
   - A serial transport walk performs seam rejection and termination decisions.
   - Only accepted triangle local-shading events are emitted.
   - A separate any-hit kernel traces one compact event/light visibility row.

7. **Serial per-pixel resolve**
   - `raster_first_shade` walks the transparent run and then the opaque terminal
     winner.
   - It evaluates color, material transport, local lighting, alpha blending,
     reflection, thin-pane transmission, and solid refraction.
   - Retired pixels commit radiance and residual background throughput.
   - Surviving branches are written to a compact primary continuation pool.

8. **Secondary handoff**
   - Raster scratch is released as one arena temporary scope.
   - Only active compact continuations are copied into a full classic wavefront
     state.
   - Full K-buffer arrays are allocated for active rays plus continuation
     reserve, not for every original primary pixel.

## 4. Ordering semantics

The classic deterministic order is a strict total relation:

```text
(depth_bin = floor(t / DEPTH_TIE_EPSILON), descending layer)
```

The raster path uses the same relation.

The opaque visibility key packs:

```text
high 32 bits: depth bin
low  32 bits: inverted layer
```

Atomic minimum therefore resolves depth-bin ties by choosing the higher layer.
The terminal winner's geometry type and primitive index are recovered from the
layer range: Bezier layers precede triangle layers, and PN layers are absent
from eligible raster batches.

Transparent fragment records retain exact positive `t` bits for hit-point and
material calculations. Host sorting derives the depth bin from exact `t` and
uses stable sorting to preserve descending-layer order within a bin.

This avoids the former raw-depth/coplanar decal problem and prevents a lower
layer from culling a higher-layer transparent fragment at nearly equal depth.

## 5. Primitive paths

### 5.1 Flat triangles

Triangle screen projection is precomputed once per `(frame, primitive)` rather
than once per chunk and pass. Camera and geometry timelines may each be
independently deduplicated to one frame; projection expands to the longest
dynamic input and indexes every source modulo its own time dimension. The
compact record contains:

- Three continuous screen x coordinates.
- Three continuous screen y coordinates.
- Three reciprocal perspective divisors.
- A validity flag.

Valid projections use edge functions and perspective-correct barycentric
weights. Invalid or camera-plane-straddling projections use the exact per-pixel
ray-cast fallback.

Per-primitive texture alpha certainty is preserved by `scene_builder.py`.
Unrelated opaque triangles continue to populate the visibility buffer when one
cutout texture exists elsewhere in the scene. During count/write, sampled
`alpha <= MIN_ALPHA` hits never enter the sort.

### 5.2 Bezier circuits

Bezier bounds are projected from per-frame world AABBs. Each candidate casts
the exact camera ray to the circuit plane, maps the hit to plane coordinates,
and reuses `_bezier_point_metrics` for fill/border coverage.

Proven-opaque circuits now participate in the typed visibility buffer. This is
important for large filled panels, circles, and backgrounds, which can cull
substantial geometry behind them. Translucent, reflective, or transmissive
circuits enter the ordered fragment stream. Their packed negative primitive
reference includes the border flag.

Circuits use thin-pane transport: reflected energy bends, while the
transmitted/coverage continuation remains on the original straight ray.

### 5.3 PN patches

PN patches are not rasterized and are not flattened. Their presence routes the
batch to the classic deterministic frontend. A future PN candidate path may use
projected OBB bounds plus the existing exact solver, but it is not required for
the current frontend.

## 6. Fragment and run memory layout

Each emitted fragment stores:

```text
frag_key : int64   (local pixel in high bits, exact positive f32 t bits low)
frag_ref : int32   (triangle id, or packed negative Bezier id/border flag)
frag_ab  : float2  (triangle barycentrics or Bezier plane coordinates)
```

There is no separate `frag_t` or flags array. `t` is recovered by bit-casting the
low key bits. One `run_offsets` array of length `tile_pixels + 1` replaces
separate run-start and run-length arrays.

The fragment depth is not literally unlimited. Both the transport/shadow event
walk and final resolve stop at `MAX_SURFACES_PER_RAY`, currently 256 confirmed
hit positions. Shared-edge duplicates count toward the limit before seam rejection,
matching both walks. This is a safety bound far above the old
four-entry K-buffer but must remain documented.

## 7. Memory allocation and retry

Large raster transients are arena-backed:

- Typed z-buffer.
- Pair counts and offsets.
- Unsorted/sorted fragment arrays.
- CSR run offsets.
- Sparse shadow event data and visibility.

PyTorch sort/index scratch remains allocator-owned because `torch.argsort` and
`index_select` cannot target a supplied arena view.

The host wraps the complete raster phase in `memory.temp()`, so all transients
are released before secondary state allocation. Allocation failures inside a
raster tile are caught by the wavefront tile loop. The current primary count is
halved and retried. This provides a useful retry path even for a single-frame
batch; it no longer relies only on halving the outer frame window.

The compact primary queue lives outside the arena during raster resolve. It
contains only:

- Origin and direction.
- Accumulated radiance.
- RGB throughput and scalar continuation state.
- Integer bounce/status state.
- Pixel id.

It has inert one-element placeholders instead of six `[pool, KBUF]` arrays.
After resolve, active records are compacted and copied into a full secondary
wavefront allocation. The old K-buffer is therefore paid only by rays that need
secondary traversal.

## 8. Shadow queues

Primary hard and soft shadows use a dedicated sparse queue rather than
position-limited packed bits.

The event build pass mirrors the ordered primary transport decisions sufficiently
to identify accepted triangle local-shading events. Each event stores:

- World hit point.
- Viewer-oriented shading normal.
- Consistently oriented geometric face normal.
- Frame index.

A separate any-hit kernel traces every event against every supported shadow
light and writes a float visibility matrix. Zero-radius point/spot/directional
lights emit one hard-shadow ray. Non-zero point/spot radii and directional
angular radii emit the same fixed golden-angle fan used by the classic
wavefront shader. Rect-area lights are already represented as multiple packed
sample rows, so averaging their per-row visibility naturally produces a soft
penumbra. Fragment and terminal-winner event IDs map the visibility rows back
into resolve.

There is no additional raster-specific limit of three transparent positions or
eight lights. The renderer-wide compile-time `MAX_SHADOW_LIGHTS` limit still
applies equally to raster and classic material shading. The fan size is the
compile-time `SOFT_SHADOW_SAMPLES` setting, exactly as on the classic path.
Secondary-bounce shadows remain inside the classic wavefront shade kernel;
moving all secondary shadows to a specialized queue is future work.

## 9. Host synchronization

Avoidable camera-projection scalar conversions were removed. Projection math
remains tensorized on the render device, and pair expansion uses output shapes
rather than explicit `sum().item()` reads where practical.

Some synchronization remains structurally necessary with dynamic PyTorch
allocations:

- Total emitted fragment count before allocating exact arrays.
- Total sparse shadow event count before allocating visibility.
- Active continuation count before allocating full secondary state.
- Wavefront continuation overflow flag.

Removing these requires capacity-based persistent buffers, device-side queue
scheduling, or CUDA-graph-compatible allocator changes rather than local source
edits.

## 10. Taichi compilation timing

Algan instruments Taichi 1.7.4 at its two compilation boundaries:

- `Kernel.materialize`: Python AST inspection/transformation and creation of the
  C++ kernel object (reported as `frontend`).
- `Program.compile_kernel`: backend lowering, or loading the equivalent offline
  cache artifact (reported as `backend`).

For every newly materialized specialization, the runtime logs a start timestamp
and a completion timestamp with frontend, backend, and total seconds. Ordinary
launches of an already-ready specialization are not logged. Logging is enabled
by default and can be disabled with `ALGAN_LOG_TAICHI_COMPILES=0`. Setting
`ALGAN_TAICHI_COMPILE_LOG=/path/to/log.jsonl` also writes machine-readable JSON
records. With an empty offline cache the reported backend duration is cold
compilation time; with a populated cache it is cache lookup/load time.

## 11. Current limitations

- The frontend uses linear row-band wavefront tiles, not square block tiles.
- Triangle/circuit exact intersections are repeated in COUNT and WRITE.
- Camera-plane straddlers conservatively expand to the complete current row
  band; all-behind bounds are not specially rejected by the host.
- `RASTER_CHUNK` is fixed at 256 and has not been retuned across scene classes.
- PN patches, near clipping, custom scatter, and multi-sample AA use the
  classic frontend.
- Secondary radiance rays still carry the classic global K-buffer.
- Secondary shadow queries are still performed by the classic shade path.
- Raster engagement is feature-based rather than cost-model-based.

## 12. Planned work

### 12.1 Avoid the second exact intersection

COUNT and WRITE currently repeat the expensive geometry test. The following
alternatives should be benchmarked rather than chosen by intuition:

1. **Candidate records followed by stream compaction**
   - Emit one provisional record for every bbox candidate.
   - Store a validity bit and exact hit payload once.
   - Prefix-scan validity and compact accepted records.
   - Advantage: one exact test.
   - Cost: provisional memory scales with bbox overdraw, which is worst for thin
     cylinders/arrows and can exceed the accepted-fragment volume by an order of
     magnitude.

2. **One cooperative block per pair with a block-local scan**
   - Threads test candidate pixels in parallel.
   - Accepted counts are scanned in shared memory.
   - One block reservation allocates a contiguous output interval.
   - Payload is written directly without a second test.
   - Advantage: projection/setup reuse, fewer global atomics, bounded temporary
     state.
   - Cost: Taichi control over block shape/shared storage is less flexible, and
     very small pairs waste lanes.

3. **Block-local append plus one global atomic reservation**
   - Threads retain hit payload in registers/local state.
   - A block count reserves one global range with a single atomic add.
   - A local prefix determines final positions.
   - Advantage: no global count array or host prefix allocation.
   - Cost: output order becomes block-dependent, so the complete deterministic
     tie-break must be encoded in sort keys. Capacity must be provisioned or an
     overflow/retry protocol added.

4. **Cache first-pass hit payload per candidate chunk**
   - COUNT stores accepted hit data in a bounded chunk-local scratch region.
   - WRITE consumes the cached region after host offsets are known.
   - Advantage: preserves exact sizing and deterministic pair layout.
   - Cost: scratch approaches provisional-candidate memory and must survive the
     host prefix step, reducing the value of the two-pass scheme.

5. **Persistent accepted-fragment pool with device-side offsets**
   - Each pair or block appends directly into a capacity-sized pool.
   - A final device-side sort/CSR build consumes the used prefix.
   - Advantage: one test and fewer host synchronizations.
   - Cost: requires robust pool sizing, overflow retry, and tighter integration
     with the manual arena.

### 12.2 Square block tiles

Investigate 8x8 or 16x16 screen tiles with primitive-to-tile binning. Potential
benefits include projection reuse, tile-local sorting, inline short fragment
lists, and better cache locality. Costs include another binning hierarchy and
load imbalance for very large or very sparse primitives. The current linear
wavefront tile remains the baseline until measured.

### 12.3 Camera/near-plane clipping

Replace full-row fallback for straddlers with one of:

- Reject bounds whose complete control hull is behind the camera plane.
- Clip flat triangles against camera/near planes before projection.
- Clip projected conservative polygons for AABB/OBB bounds.
- Split bounds at the plane and project only the front portion.

Near-clip support must use the same clipped ray origin/base-distance convention
as the classic renderer.

### 12.4 Chunk-size tuning

Benchmark `RASTER_CHUNK` values 32, 64, 128, and 256. Also store an explicit
valid count for the final chunk so tiny bboxes do not execute 256 failed loop
iterations. A cooperative block-per-pair implementation may make the optimal
value a launch geometry rather than a serial loop length.

### 12.5 Remove the global K-buffer from secondary radiance rays

The current secondary engine permanently attaches six K-buffer arrays to every
allocated ray. A redesigned traversal should keep a small closest/multi-hit set
in registers, emit compact surface events, and allocate spill storage only for
rays with deep same-direction transparency. Separate record types should remain:

- Radiance continuation queue.
- Any-hit shadow queue.
- Same-direction transparency/surface-event queue.

### 12.6 BLAS/TLAS instancing

The current STBVH flattens world-space primitives across a frame batch. A future
renderer should store local-space geometry in reusable BLASes and represent
rigid animation with per-frame instance transforms and bounds. A small TLAS can
then be rebuilt/refit per frame or frame group. Deforming geometry can refit or
rebuild its BLAS according to a quality metric.

### 12.7 Unified typed top-level hierarchy

Secondary radiance currently walks separate triangle, PN, and Bezier trees.
Introduce one typed TLAS over homogeneous BLAS roots/chunks. Query masks should
allow rejection of subtrees that cannot satisfy opaque, shadow, geometry-type,
or visibility requirements. Keep low-level leaves homogeneous to avoid severe
warp divergence.

### 12.8 Transport events before grouped shading

For material-grouped shading, do not partition raw fragments directly. First run
an ordered transport stage that performs seam rejection, alpha/material fetch,
throughput calculation, path-bending decisions, and accepted local-event
emission. Then group accepted events by material/geometry, shade them, and reduce
weighted contributions by pixel. The same event stream can feed exact shadows.

### 12.9 Dynamic frontend routing

Add a cheap per-batch cost estimator based on projected candidate count,
primitive count, opaque fraction, expected accepted fragment volume, resolution,
and unsupported features. Very sparse scenes with shallow BVHs may remain faster
through classic primary tracing; dense/faded scenes should strongly prefer
raster.

### 12.10 Fully asynchronous scheduling

The remaining dynamic-count host reads prevent a single graph-like asynchronous
batch. Future capacity-sized persistent queues can overlaunch kernels against
device-side counts, use block-level queue reservations, and swap queue roles
without `item()` synchronization. This should be attempted only after the queue
and acceleration-structure redesigns stabilize.

## 13. File map

- `raster_pipeline.py`: host projection, pair construction, exact sizing,
  sorting, CSR runs, shadow queue orchestration, and primary resolve.
- `raster_taichi.py`: geometry tests, typed z passes, count/write, shadow event
  and any-hit kernels, final resolve.
- `scene_builder.py`: per-frame visibility/AABB masks and per-primitive texture
  alpha certainty.
- `tracer.py`: frontend gate, compact primary state, tile retry, projection
  precompute, secondary handoff.
- `renderer_settings.py` and `mobs/surfaces/surface.py`: preserve configured PN
  geometry independently of raster eligibility.
- `stbvh.py`: unchanged secondary-ray acceleration structure.
- `rendering/taichi_runtime.py`: runtime initialization and per-specialization
  frontend/backend compilation timing.
