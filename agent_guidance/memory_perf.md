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
