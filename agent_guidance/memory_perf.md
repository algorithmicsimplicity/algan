# Memory and performance

The arena, runtime batch sizing, and the validation standard for optimizations.

## Manual memory

`ManualMemory` (`algan/utils/memory_utils.py`) is the render-time arena: a bump allocator for render-time GPU tensors, with deterministic forward allocations and pointer snapshot/restore so callers free deterministically. Render out-of-memory retries by shrinking the frame window (`OutOfRenderMemory`).

There are **no byte estimators to update** when you add a buffer — see the next section. What still applies:

- account for dtype alignment and fixed versus per-frame/per-ray scaling;
- restore arena pointers at the same lifetime boundary at which data becomes dead;
- test one-frame and multi-frame windows;
- test retry behavior rather than relying on host OOM exceptions.

The whole package enters process-global `torch.inference_mode()` during import. Importing Algan therefore disables autograd for the importing process. Do not use the same process for Algan rendering and Torch model training.

## Batch sizing is measured at runtime, not modelled

`rendering/memory_model.py` fits `peak(n) = a + b*n` to the arena's own high-water mark over rendered chunks, and sizes the next chunk from it. **Nothing describes what gets allocated**, so a new primitive, a new tracer path or a user's own post-process is accounted for the moment it runs — there is nothing to annotate, register or regenerate. This replaced a set of hand-written byte formulas *and* a generated calibration table; do not add either back.

Consequences worth knowing when changing render code:

- The **first chunk of a job is ~30% cheaper per frame** than steady state (kernel/allocator warm-up), so the model grows chunks geometrically (`PROBE_GROWTH`) and fits from the two *largest* samples rather than extrapolating off the first.
- Batches land on different lines when the frame buffer or geometry scale changes; `chunk_signature` keys that, with geometry bucketed logarithmically so ordinary scene drift keeps a usable fit.
- The **OOM retry is the backstop and must stay** — the model measures the batch's first frames and cannot see a scene that densifies later.

The merge and projection build *outside* the arena in pool headroom, so the model cannot see them; they keep the deliberately generous `MERGE_GPU_PEAK_FACTOR` / `PROJECT_GPU_PEAK_FACTOR` bounds on their packed inputs.

`ManualMemory.scope()` / the allocation recorder are **diagnostics only** — they do not participate in batch sizing. Use them to attribute arena usage per stage when investigating; do not add scopes expecting them to affect a render's memory budget.

## Performance and renderer validation

Optimize general moving and animated scenes, not only static-scene fast paths.

For performance changes:

- compare warm in-process alternating A/B runs when possible;
- use device-side kernel-profiler timings to separate launch/synchronization from kernel execution;
- avoid drawing conclusions from a single cross-process wall-clock run;
- verify the intended optimization gate actually engaged;
- record render route and relevant live settings;
- validate output parity before accepting a speedup.

Use focused parity/benchmark scripts under `../benchmarks` when present. The default path should remain output-compatible unless the change intentionally modifies rendering. If adding an experimental optimization, provide a kill switch and keep capability checks, memory estimation, and fallback behavior coherent.

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

- **`DESIGN_optimization_targets.md` is the plan of record for render performance.** It opens with a status table, how to reproduce the reference profile, and what to verify. Read it before starting (or resuming) optimization work, and update it when something lands.
- Optimizations must target general moving scenes, not static-only fast paths.
- The standard for optimizations is **byte-identical output** validated by an A/B parity script (see the `benchmarks/_*_check.py` / `_*_ab.py` conventions); features are gated behind settings toggles so the default path stays byte-identical.
- One shipped exception, deliberately taken: the subdivision-level criterion kernels (`pn_criterion_kernel`, default on) run under Taichi's `fast_math`, so they flip a handful of borderline tessellation levels and **moved three full-render baselines**. Bit-identity is not recoverable per-kernel there. Note what this implies generally: a change to tessellation, projection or a level criterion is **invisible to `--fast`** — `tests/fast/scene.py` has no PN geometry — so it needs `pytest -q tests/full_renders`.
- Wall-clock kernel timing is noisy (thermal throttling swings cross-process throughput ~2x); use in-process alternating A/B runs or kernel-profiler device times. `utils/profiling_utils.py` auto-hooks all Taichi kernels and pipeline stages.
