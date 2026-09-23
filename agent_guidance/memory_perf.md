# Memory and performance

The arena, runtime batch sizing, and the validation standard for optimizations.

## Manual memory

`ManualMemory` (`algan/utils/memory_utils.py`) is the render-time arena: a bump allocator for render-time GPU tensors, with deterministic forward allocations and pointer snapshot/restore so callers free deterministically. Render out-of-memory retries by shrinking the frame window (`OutOfRenderMemory`).

The chunk model observes arena peaks rather than duplicating every allocation
formula. Scene preflight and per-slot budgeting still have explicit bounds, so
check those when changing geometry or ray-state storage. What always applies:

- account for dtype alignment and fixed versus per-frame/per-ray scaling;
- restore arena pointers at the same lifetime boundary at which data becomes dead;
- test one-frame and multi-frame windows;
- test retry behavior rather than relying on host OOM exceptions.

## Batch sizing fits a model to observed arena peaks

`rendering/memory_model.py` fits `peak(n) = a + b*n` to the arena's own high-water mark over rendered chunks, and sizes the next chunk from it. New allocations made **through the arena** contribute to that measurement.
External PyTorch tensors, compiler workspace and driver allocations do not; they
need separate headroom and telemetry. An affine estimate is not an exact memory
formula for every scene.

Texture preparation has its own retry before arena preflight. A PyTorch OOM
while materializing or building primitives clears partial timeline buffers and
halves the frame window on the same preparation worker, including prefetched
batches. One-frame OOMs and unrelated errors still propagate. Custom animation
callbacks disable compact working sets, so batch sizing also prices every
allocated texture row (including inactive panels), with assignment scratch,
against the device where those rows materialize.

Bezier projection reuses unchanged circuit polylines (including Text) across
frames and preparation windows. Edges are sampled from the controls and centers
*before* the author-order/z-index depth bias, which slides each circuit along its
own eye ray and so moves every circuit whenever the camera moves; sampling after
it defeated reuse in every moving-camera shot. A still circuit is reused
exactly. A rigidly translated one (relative controls within 16 float epsilons of
the coordinate scale, compared in 8-frame chunks) reuses its first frame's
edges, which differ from per-frame edges by rounding only. Rotating, scaling or
morphing circuits are partitioned out without changing draw order.
Camera-dependent chord selection still runs, and exact pre-bias controls, plane
frames, connectivity and chord counts key the cross-batch cache; materials and
bounds remain live. A/B this with batching pinned
(`available_memory_override`, `ALGAN_PREFETCH_BATCHES=0`): the cache changes
batch sizes, and re-windowing moving text alone produces 255-level pixel diffs. Each render context retains at most 16 MiB
of keys and device edges outside the arena, visible to pool-headroom telemetry.
The cache is cleared after joining the prep worker, including errors and closed
generators. `SETTINGS.raytracing.experimental.bezier_geometry_cache = False`
restores the uncached build for A/B checks.

Consequences worth knowing when changing render code:

- First-chunk peaks can understate later workspace (historical probes measured
  roughly 30%). The model grows chunks geometrically (`memory_probe_growth`)
  and fits from the two largest observations rather than trusting that first
  sample. The percentage is not a fixed property of every render.
- Batches land on different lines when the frame buffer or geometry scale changes; `chunk_signature` keys that, with geometry bucketed logarithmically so ordinary scene drift keeps a usable fit.
- The **OOM retry is the backstop and must stay** — the model measures the batch's first frames and cannot see a scene that densifies later.
- Auto-sized ray tiles mark observations that were limited by available arena capacity. Their high-water marks still bound render chunks, but `predict_preflight` leaves scene preparation in probe mode: an elastic workspace's chosen capacity is not its minimum requirement. Otherwise the safety margin can price one frame above the whole arena and force every later scene batch to one frame. The unmeasured-batch guard, scene/merge/projection checks, and OOM retries remain active. This applies to both shadowed and unshadowed scenes. Changed batch windows can refine adaptive curve subdivision and move a few shadow-edge samples; review and update affected render baselines when enabling it. `SETTINGS.raytracing.experimental.elastic_preflight = False` restores the old prediction for all scenes for A/B checks.

The merge and projection build *outside* the arena in pool headroom, so the model cannot see them; they keep the deliberately generous `MERGE_GPU_PEAK_FACTOR` / `PROJECT_GPU_PEAK_FACTOR` bounds on their packed inputs.

`ManualMemory.scope()` / the allocation recorder are **diagnostics only** — they do not participate in batch sizing. Use them to attribute arena usage per stage when investigating; do not add scopes expecting them to affect a render's memory budget.

## Host pressure and warm CUDA programs

Render teardown drops the arena and unfreezes the scene before the outermost
arch scope handles a deferred host-pressure reset. That scope first repeats
the ordinary pressure-gated reclamation, so newly freed CUDA cache blocks can
be returned before deciding whether to discard Quadrants. The prep worker is
joined before any of this, including on errors and generator close.
Video jobs additionally hold the outer arch scope through encoder draining
and timeline restoration, so their last CPU frame buffers have been released
before the reset decision.

Native heap reclamation uses glibc `malloc_trim(0)` on Linux and Windows
`HeapSetInformation(HeapOptimizeResources)` on unused process heap caches.
Neither discards live compiler state. A successful reset is followed by
another trim to return the allocations the compiler just freed.

On Windows CUDA, a Program restored entirely from the source-key cache has
much less reclaimable compiler IR than one that ran the compiler frontend.
Ordinary host pressure (15% available physical RAM) still triggers garbage
collection and CUDA/native cache reclamation. A cache-only Program is retained
while `GlobalMemoryStatusEx` reports available commit above 15% of the commit
limit (and 1 GiB). Low available physical RAM alone must not discard this
cache-only Program: repeated screenshots otherwise reload the same kernels
after every frame, without relieving the underlying pressure.
New compilation, lower headroom, or unavailable telemetry retains the reset
fallback. This exception does not apply to Linux/cgroups or CPU rendering.
It changes the destructive reset decision, not arena sizing or OOM retries.

`benchmarks/performance/pressure_reset_probe.py` records reset counts, process
memory, host/commit headroom, and teardown ordering on the real graphics scene.
Its `--sequence legacy,legacy,current,current,current,legacy,legacy` option
loads only the old cleanup functions from the specified `--baseline-ref` for
an in-process comparison with identical rendering settings.

## Performance and renderer validation

For performance changes:

- compare warm in-process alternating A/B runs when possible;
- use device-side kernel-profiler timings to separate launch/synchronization from kernel execution;
- avoid drawing conclusions from a single cross-process wall-clock run;
- verify the intended optimization gate actually engaged;
- record render route and relevant live settings;
- validate output parity before accepting a speedup.

The CUDA bounce drain uses 32-bit spatial sort keys for render windows of at
most 16 frames (`wf_ray_sort_compact`). It retains frame and direction-octant
priority and drops the bottom six Morton position bits. This only reorders
rays; their state and intersection arithmetic stay the same. Wider windows
and other devices retain the original 64-bit keys. The bound is the full
rendered frame window, **not** the sparse ray pool's slot count. Set
`SETTINGS.raytracing.experimental.wf_ray_sort_compact = False` for A/B checks.

Use focused parity/benchmark scripts under `../benchmarks` when present. The default path should remain output-compatible unless the change intentionally modifies rendering. If adding an experimental optimization, provide a kill switch and keep capability checks, memory estimation, and fallback behavior coherent.

Primary shadow queues with at least 8,192 events are sorted by source primitive
on CUDA (`shadow_primary_sort`). This happens before the existing payload
gathers; the sheet event-ID scatter uses the same permutation, preserving the
visibility lookup. Source refs, including -1 for Beziers, are reused for
identity rejection. Small queues and other devices retain their original order.
The shadowed UHD benchmark improved 7.6% in warm instrumented A/B means, with
all 30 raw frames byte-identical and essentially unchanged peak GPU allocation.
Set `SETTINGS.raytracing.experimental.shadow_primary_sort = False` for A/B checks.

CUDA shadow fans use light-major lane scheduling for queues with at least
16,384 events and multiple lights (`shadow_light_major`). Each event/light
cell retains its exact serial sample and reduction order; small queues and
other devices keep event-major scheduling. The graphics UHD workload on T4
improved from 88.36 to 69.59 s in warm alternating render medians, with decoded
channel differences within two. `benchmarks/performance/shadow_schedule_ab.py`
checks identical live primary/secondary queues and records device times.

Completed CUDA frames use owned pinned host storage (`pinned_frame_readback`)
for transfers up to 256 MiB per batch. The current CUDA stream is synchronized
before the tensor is returned, so this does not change the writer's readiness
or ownership contract. Larger batches, other devices, and pinning allocation
failures use pageable storage. T4 explainer UHD render medians improved from
3.40 to 2.84 s with identical pixels. Both switches have `ALGAN_` environment
equivalents and live controls under `SETTINGS.raytracing.experimental`.
Full measurements and compiler/device details are in
`benchmarks/performance/reports/cuda_workloads_2026_09/`.

CUDA sheet compaction assigns conflict-rank groups with prefix counts
(`sheet_rank_groups`) instead of globally sorting `(parent * 16 + rank)`.
This relies on dense, ordered parents and ranks containing every integer from
zero to the parent's maximum: a fragment increments each claimed sample lane
by one. Ranks can decrease, so consecutive unique is not valid here. Outputs
retain exactly the original dense IDs. The captured UHD operation used 41.45
instead of 120.15 MiB temporary GPU memory and ran 79% faster; whole-render
warm A/B means improved a modest 1.7%, with unchanged overall peak allocation.
Other devices keep the original path. Set
`SETTINGS.raytracing.experimental.sheet_rank_groups = False` for A/B checks.

CUDA sheet sorting uses exact packed keys (`sheet_packed_sort`) when the
observed pixel/group/depth ranges fit signed int64. Float32 depth bits are
retained exactly; no quantization is used. Negative/nonfinite depths, negative
zero, oversized combined ranges and other devices retain the stable reference
sort. Packing starts at 32,768 rows for pixel/group/depth and 262,144 for shell
key/depth, based on crossover measurements. Captured UHD sorts took 37-53%
less time. Whole-render A/B sets disagreed (-4.5% then +10.9% improvement),
while both reduced compaction time; pooled total time improved 1.7%, with
substantial variability. Set
`SETTINGS.raytracing.experimental.sheet_packed_sort = False` for A/B checks.

### Split pixels are not byte-reproducible: pick A/B fixtures accordingly

Some scenes render slightly differently every run, with no change to the code
or the settings, so they cannot serve as byte-identical A/B fixtures.

Every branch of a pixel commits its premultiplied colour and its leftover
background throughput into the shared per-pixel accumulator `pix_accum` with
`ti.atomic_add` (`wavefront_kernels_taichi.py` ~3063-3095, `raster_taichi.py`
~4655-4665). Float atomic add is *commutative but not associative*, and the
order in which branches of one pixel reach the accumulator is GPU scheduling
order, which varies run to run. A pixel carrying one or two branches is
therefore still bit-exact; a pixel carrying **three or more** is not.

Multi-branch pixels come from the shared continuation pool, which is only used
when `_split_pool_ratio` exceeds 1 — reflective/refractive geometry *and*
analytic AA (or `ANALYTIC_AA_SECONDARY_SAMPLES > 1`, which puts N sub-pixel
reflection taps on one pixel). Measured on 3 cubes + 2 spheres + a reflective
ground at MD over 60 frames (`benchmarks/_split_determinism_check.py`): every
tile's `pix_accum` digest differs between two runs in one process, while the
merged scene tensors feeding the kernels hash identically. Turning off any one
ingredient — `max_bounces=0`, `analytic_aa=False`, or all-unlit materials —
makes the digests match and the frames byte-identical.

The output effect is bounded and small. `wf_composite_accum` truncates to `u8`
(~line 1324), so a reassociation difference can move a channel by at most 1: the
measured spread is `|d| = 1` on tens of channel samples out of 165M, and the
encoded mp4 came out bit-identical. This is a parity-fixture constraint, not a
rendering defect.

Practical rules:

- Do not use a scene with reflective/refractive geometry under analytic AA as a
  byte-identical parity fixture. Establish the arm's own run-to-run floor first,
  or pin one of the ingredients off.
- A *larger* difference than `|d| = 1` is NOT this mechanism. Branches are never
  silently dropped: a continuation that does not fit raises the pool's overflow
  flag, and the host discards and retries the tile with fewer primaries
  (deterministically — verified by starving the pool to force 13 retries, which
  changed nothing). Suspect the change under test instead. That "never" is now
  instrumented rather than asserted: the host reads the flag on *every* tile,
  including the split-free ones at `pool_ratio == 1` that used to short-circuit
  past it, and counts any reservation past the capacity as
  `RenderPlan.truncations.dropped_continuations` — which reads zero on the
  shipped renderer and is there to catch the change that makes it stop.
- The fix, if byte-identical A/B on reflective scenes is ever needed, is to
  accumulate in fixed point: integer atomic add *is* associative and
  commutative, so the sum stops depending on arrival order. That trades a scale
  factor and a conversion for reproducibility; it has not been needed so far.

For source-only correctness checks, at minimum run import/compile checks on modified non-Taichi modules. For visual renderer changes, render a minimal `SMOKE_TEST` scene or a single diagnostic frame. Do not run a long benchmark merely to prove that code imports.


## Performance discipline

- The standard for optimizations is **visually imperceptible** on the test suite.
- Wall-clock kernel timing is noisy (thermal throttling swings cross-process throughput ~2x); use in-process alternating A/B runs or kernel-profiler device times. `utils/profiling_utils.py` auto-hooks all Taichi kernels and pipeline stages.

## Warm screenshot preparation (September 13)

Windows host-pressure checks use GlobalMemoryStatusEx directly for physical
memory, falling back to psutil if telemetry is unavailable or invalid. The
threshold and freshness are unchanged; psutil's broader Windows query had cost
0.47–0.59 seconds across 62 calls in a small screenshot.

`_prewarm_render_batch` coalesces small reclamation requests within its bounded
preparation scope. State is thread-local and nested scopes drain once, including
on exceptions. Forced cleanup, GPU pressure, a reclaimable CUDA cache of at least
128 MiB, MPS cleanup, and host pressure without safe Windows commit headroom
remain immediate. Arena accounting and allocation/OOM retry behavior are unchanged.

Unfilled Bezier strokes omit inward-edge parity preparation: only the filled
wedge coverage branch reads that data. They still emit the same six-column edge
layout, with zero in the unused sign channel.

The `1_one_weight` PREVIEW screenshot comparison (GTX 1050, same-process
legacy/current/legacy/current after warm-up) found the two slowest frames improved
from 2.25/2.12 seconds to 1.69/1.50 seconds in two-run means. All six screenshots
were byte-identical after decoding. These are loaded-machine wall timings;
per-stage profiling confirms cleanup dropped from 0.70 to 0.20 seconds on f4,
with nine collections instead of 32, independently of whole-frame variability.
