# Warm UHD optimization, shadows disabled

Workload: the supplied `nn_scene_UHD.py`, unchanged: 0.3 seconds, 18 frames at
3840×2160, shadows off, the existing ultrafast x264 arguments. GTX 1050 4 GiB,
Windows WDDM, patched Quadrants 1.3.1 (invariant argument loads enabled).

## Results

Six alternating in-process renders, discarding the first pass of each arm.
Timings include encoding and exclude authoring. No profiling hooks or
per-kernel synchronizations run inside the measured renders; the preflight
wrapper only increments a counter. The reference restores the old exhausted-
budget handling and selects the original tensor depth reduction.

| Measurement | Reference | Optimized |
| --- | ---: | ---: |
| Warm render pair 1 | 95.573 s | 34.223 s |
| Warm render pair 2 | 77.197 s | 35.167 s |
| Mean warm render | 86.385 s | 34.695 s |
| Exact preflight calls per render | 46 | 14 |
| Mean peak allocated CUDA memory | 2256.27 MiB | 2254.98 MiB |
| Mean peak reserved CUDA memory | 2453 MiB | 2337 MiB |

The mean is **2.49× faster / 59.8% less time**. Individual paired speedups are
2.20–2.79×; absolute wall time varies considerably on this host. Whole-render
peak allocation is essentially unchanged: a different stage determines it.
Reserved allocator memory falls by 116 MiB (4.7%).

On the first frame's actual captured sheet inputs, six alternating reduction
measurements give median wall times of **69.17 → 6.79 ms**. Temporary CUDA
allocation is **451.013 → 4.839 MiB**, a **98.9% reduction**. The device profiler
attributes about 6.21 ms to the new kernel's two launches. Every output word
matches the original tensor implementation exactly. These stage-local memory
savings must not be presented as a 98.9% reduction of whole-render memory.

Raw measurements: `warm_combined_0906/results.json`, `depth_micro_0906.log`.
The initial instrumented baseline is `warm_baseline_0906.log`; its wall times
are not mixed into the clean A/B numbers above.

## Changes

1. **Retain an exhausted batch budget.** `_note_batch_cost` first receives
   the exact scene-upload budget and then the budget after reserving forward
   workspace. It discarded the latter observation when zero or negative,
   retaining the earlier optimistic capacity. Subsequent batches repeated
   expensive projection, tessellation and BVH builds for the same rejected
   candidate sizes. Recording a zero capacity carries the actual verdict into
   the next fetch. Memory margins, single-frame fallback, and OOM retries stay
   active. No safety multiplier was reduced.

2. **Reduce competing depths inside each pixel.** The original tensor path
   expands enforcers into eight sample-lane rows, globally sorts them, and
   gathers the best other-surface depth back into another expanded table.
   `sheet_depth_lose` performs two unbounded walks of each pixel's sheets,
   maintaining the nearest and second-nearest depths from distinct surfaces
   for each lane. It writes only the final loss word per sheet. Repeated
   crossings of the same surface remain excluded. Threshold and tie semantics
   are preserved; there is no new overlap limit.

3. **Fuse the lane-owner depth gather.** `sheet_lane_depths` copies valid
   owner depths and writes infinity for missing owners, without expanded
   clamp, gather, and masking temporaries.

The tensor implementation remains available via
`SETTINGS.raytracing.experimental.sheet_depth_reduce_kernel = False`, or
`ALGAN_SHEET_DEPTH_REDUCE_KERNEL=0` before import. This controls both depth
optimizations and takes effect live.

## Validation

- Focused sheet/depth tests: 49 passed.
- Required fast regression run: passed, including its pixel-compared render.
- Full UHD raw-pixel comparison for the depth changes: 4 of 447,897,600 color
  channels differ, all by exactly one level. This is within the documented
  split-ray float-atomic variation. No channel differs by more than one.
- All truncation counters are zero in every A/B render.
- An output frame was visually inspected. Ruff and `git diff --check` pass.
- Full regression suite: stopped at the user's request before completion;
  **no full-suite pass is claimed**. The existing
  `full_renders/cuda` files do not match `tests/baselines.json`; the independent
  `scripts/package_baselines.py --verify` reproduces that metadata failure.
  No baseline file or manifest was changed. The CLI test that inherited
  `ALGAN_USE_DAEMON=0` from the full-suite command passes on a clean rerun.

The changes are left in the working tree. The supplied benchmark's existing
edits were preserved; no scene-quality setting was reduced.

## Reproduce

Run from the repository root, one GPU process at a time:

```powershell
.venv/Scripts/python.exe benchmarks/performance/nn_depth_experiment.py --runs 6 --capture-depth --tag warm_ab
.venv/Scripts/python.exe benchmarks/performance/sheet_depth_benchmark.py benchmarks/performance/warm_ab/depth_inputs.pt
.venv/Scripts/python.exe benchmarks/performance/nn_depth_experiment.py --runs 2 --verify --tag parity
.venv/Scripts/python.exe -m pytest -q --fast
.venv/Scripts/python.exe -m pytest -q
```

`--depth-only` keeps the fixed batching behavior in both arms to isolate the
depth change. `--verify` compares raw channels before encoding; run it
separately because reading/writing the reference stream affects timing.
The scripts consume the original benchmark's scene function and settings,
rather than maintaining a copy that can drift from the representative workload.
