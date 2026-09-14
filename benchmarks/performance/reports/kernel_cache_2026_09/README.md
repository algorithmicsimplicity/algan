# Cross-process kernel-cache investigation (Mac GPU, 2026-09-12)

## Result

The suspected general failure of the persistent kernel cache was **not
reproduced on the Mac GPU**. Both current master and the original optimization
branch reuse their on-disk kernels in fresh processes. PREVIEW-to-UHD changes
also reuse the source keys shared by the two workloads. This change fixes
misleading profiler output and a cache acceptance test which previously only
warned about misses; it does **not** claim a renderer speedup or change the
cache's key/invalidation rules.

The original GPU workload report's first-pass and launch timings did not by
themselves establish whether the disk cache missed. The profiler's
`cold (includes Taichi JIT compile)` label was unconditional. A first pass
includes framework initialization, cache loads
and submission as well as any real compilation; it is not evidence of a cold
disk cache. A source-key miss also is not sufficient evidence of backend
compilation: it rebuilds frontend IR and can still hit the native IR-keyed cache.

## Environment and evidence

All GPU measurements used GitHub's `mac-mps` arm, real Metal on its virtualized
Apple-silicon runner, CPU animation, software video encoding, and the normal
locked `algan-quadrants==1.3.0.post2` dependency. The request selected
`taichi_wheel_run_id: "none"`, not the harness's historical Taichi override.
`ALGAN_AUTO_DAEMON=0` and `ALGAN_USE_DAEMON=0` prevented process reuse. Each
experiment used a separate fresh `ALGAN_CACHE_DIR`, shared only by its child
processes. No Kaggle/CUDA measurement was performed.

Revisions:

- Current-master control: `40aab2d7b110d2988e3b8095ebdb745a0208e854`.
- Original `claude/peaceful-carson-px98kh` control:
  `325ac0f2aeafd852fed932b10faf0bf93ecefba0`.

Completed GPU jobs and their `run-on-mac-mac-mps` artifacts:

- [Master and short explainer control](https://github.com/algorithmicsimplicity/algan/actions/runs/34692677048)
  (artifact `10297431245`).
- [Original branch, cross-preset and verification](https://github.com/algorithmicsimplicity/algan/actions/runs/34693040045)
  (artifact `10297736272`).
- [Original full clips and patch acceptance](https://github.com/algorithmicsimplicity/algan/actions/runs/34693700867)
  (artifact `10298442728`).

Artifacts contain `run-output.txt`, rendered videos and, for the cross-preset
experiment, `algan_outputs/cache_probe/*.keys.jsonl` with the exact source-key
inputs and hashes. The capture wrapper only observed the hasher; it returned
its original result. Artifact retention is finite.

## Current-master control

The existing square fixture was run in three **separate processes** against
one cache, with the same settings and frame:

| Process | Source-key result | Render seconds |
| --- | --- | ---: |
| Empty cache, fill index | 26 misses | 72.31 |
| New process, populated cache | 26 hits, 0 misses | 11.38 |
| New process, verify mode | 26 verified, 0 misses | 30.48 |

All three frames had the same PNG digest (`5394b41335b82452`). Verify mode
intentionally runs the full transform and compares the resulting C++ keys; it
is a correctness control, not a fast path.

For the original explainer authoring on master, at PREVIEW with two frames:

| Process/pass | Source-key result | End-to-end seconds |
| --- | --- | ---: |
| First process, first pass after square warm-up | 24 hits, 10 misses | 80.78 |
| Same process, repeat | already resident | 1.64 |
| Second process, first pass | 34 hits, 0 misses | 16.01 |
| Same second process, repeat | already resident | 1.31 |

The second process's `sheet_resolve_shade_arena` materialization fell from
29.606 seconds at its first sighting to 0.306 seconds, marked `fast-cache hit:
AST transform skipped`, with no call to the backend compilation boundary.
Nevertheless, the old profiler labelled that pass as including JIT compilation.
All four decoded RGB videos were byte-identical.

The remaining 16.01-versus-1.31-second fresh-process penalty is real, but it is
not evidence of source-cache failure. The materialization counters do not
separately time every driver/pipeline/framework first-use operation, so these
runs do not establish the precise cause of all that residual overhead.

### A remaining first-use layer

Source inspection gives a concrete boundary to profile next. In the pinned
[Quadrants Metal backend](https://github.com/Genesis-Embodied-AI/quadrants/blob/ab9a58ab5/quadrants/rhi/metal/metal_device.mm),
`MetalPipeline::create_compute_pipeline` translates SPIR-V to MSL and calls
`newComputePipelineStateWithFunction`; `MetalDevice::get_mtl_library` calls
`newLibraryWithSource`. Creating these driver objects is distinct from serving
a hit in Algan's source-key/compiled-intermediate-code cache. Apple's own
internal caching may also affect those calls. This is a source-derived
explanation of why a hit does not imply free first launch, not a measurement
assigning the entire residual delay to the Metal compiler.

## Original branch: short cross-preset matrix

Each row below is a new interpreter running the original explainer script with
`--frames 2 --runs 1`. PREVIEW is 704x396; UHD is 3840x2160.

| Process | Hits | Misses | Verified | End-to-end seconds |
| --- | ---: | ---: | ---: | ---: |
| PREVIEW, empty cache | 0 | 34 | 0 | 73.16 |
| UHD, same disk cache | 34 | 2 | 0 | 20.46 |
| UHD repeat | 36 | 0 | 0 | 20.33 |
| PREVIEW repeat | 34 | 0 | 0 | 24.88 |
| UHD, verify mode | 0 | 0 | 36 | 41.25 |

There were no poisoned keys or verification mismatches. Comparing the captured
key inputs confirmed that **all 34 shared source keys were identical across
PREVIEW and UHD**. The two UHD misses were newly exercised functions,
`wavefront_ray_sort_keys` and `reorder_ray_slots`, not recompiles of the shared
shading kernels. `sheet_resolve_shade_arena` took 22.382 seconds cold and
0.238/0.215 seconds on the two UHD cached processes.

Decoded RGB SHA-256 checks, including the verify-mode output:

- Both PREVIEW videos:
  `ce429368d21c20d1d3c1f6d6de49009b1aa6cfe4b917074bca29d47f04c0f102`.
- All three UHD videos:
  `c83b55818c1b912693c0f89525a9d81025b5161f68b7fd11c33534496a32755b`.

## Original branch: full-length cross-preset matrix

The third job repeated the original Mac workload sizes, still in separate
processes: **60 PREVIEW frames** (6 seconds at 10 fps) and **15 UHD frames**
(0.25 seconds at 60 fps). It used a fresh directory before the first row.

| Process | Hits | Misses | Verified | End-to-end seconds |
| --- | ---: | ---: | ---: | ---: |
| PREVIEW, empty cache | 0 | 36 | 0 | 133.96 |
| UHD, same disk cache | 36 | 0 | 0 | 47.87 |
| UHD repeat | 36 | 0 | 0 | 44.82 |
| PREVIEW repeat | 36 | 0 | 0 | 22.80 |
| UHD, verify mode | 0 | 0 | 36 | 45.89 |

Every row used exactly the same set of 36 captured source keys. There were
no poisoned keys or verify mismatches. Unlike the two-frame PREVIEW fixture,
the longer PREVIEW clip already exercised both ray-sorting kernels, so the
first UHD process required **no additional specializations**.

All 60 decoded PREVIEW frames matched their repeat; all 15 decoded UHD frames
matched both the repeat and the verify-mode render. Their decoded RGB SHA-256s:

- PREVIEW: `4ed6b92f9d9c5e3f4c07fea4fb9bfd487beef9b45c10599dd09eadc98829ac56`.
- UHD: `8298a05bfe26728e550e878fdeb08871e00017fe315f1f3f18e78a6fdcfc6c42`.

Thus the large first-process-versus-resident-process difference must not be
interpreted as a failed source/IR cache: even the 47.87-second first UHD pass
hit every entry. These are control measurements of the existing renderer,
not speedups delivered by the diagnostic patch.

## Corrections shipped

`profile_scene` now reports first/repeat **process passes**, without presuming
the first pass JIT-compiles. Its result includes `source_key_cache` with
separate per-pass authoring/render deltas for hits, misses, poisoned keys,
verified keys and key-computation time. The process-global counters are not
reset. Disabled/unavailable or unrecorded cache telemetry is distinguished from
zero lookups on an already-resident repeat pass. Reports identify the selected
render device instead of incorrectly reporting CPU for an MPS render.

`_taichi_source_key_check.py` now fails if the warm arm misses, keys cannot be
built, no kernels were keyed, or verification did not cover every keyed
kernel. Pixel equality alone no longer produces PASS when caching was not
exercised. Children explicitly disable the daemon and isolate inherited verify
mode. Unit tests cover these failure paths and counter/report behavior.

## Mac acceptance of the patch

After the full-clip control, the third job checked out the exact master base,
applied the five-file diagnostic patch, verified SHA-256 for each complete
file, and ran the tests and actual GPU renders. The artifact includes
`algan_outputs/cache_full/validated.patch` and the validated file hashes.

- **105 focused tests passed on macOS**, matching the local selection.
- The newly strict square harness passed: 26 misses on fill, 26/26 hits in
  the fresh cached process, 26/26 verified in the final process, identical
  PNGs across all three arms.
- Two fresh interpreter invocations of the profiled two-frame explainer
  showed the intended per-pass counters: first process, 34 misses then
  zero new lookups; second process, 34 hits and zero misses, then zero
  new lookups. All four decoded videos matched the unmodified control's
  `ce429368...c0f102` hash above.
- Reports correctly identified `device: mps` and used first/repeat labels,
  rather than asserting that the all-hit first pass compiled kernels.

## Local validation

The focused profiler, cache-key, runtime-configuration and harness tests passed:
**105 passed**. Python compilation checks and `git diff --check` passed.

The canonical `pytest -q --fast` run gave **605 passed, 1 failed, 3765
deselected**. Its sole failure was the fast render's existing container
LaTeX/SVG baseline mismatch: maximum channel difference 221 at frame 4. Running
that render again with the original, unmodified profiler reproduced exactly
the same failure. The environment uses the compatibility `dvisvgm` wrapper
described in `LOCAL_INSTALL.md`; this is not a new renderer regression. Ruff
was unavailable in the local environment.

## Reproduction

Use the installed environment's Python directly. For a genuine empty-cache
control choose a **new directory**, and keep the same directory for all arms:

```sh
ALGAN_RENDER_DEVICE=mps ALGAN_ANIMATION_DEVICE=cpu \
ALGAN_AUTO_DAEMON=0 ALGAN_USE_DAEMON=0 \
ALGAN_CACHE_DIR=/tmp/algan-cache-check-new-directory \
.venv/bin/python benchmarks/_taichi_source_key_check.py --arms warm,on,verify
```

For a particular scene, run the identical command in another interpreter with
`ALGAN_LOG_TAICHI_COMPILES=1` and inspect the source-key counts and per-kernel
records. Then change the preset, record which kernels are genuinely new, repeat
that preset, and run once with `ALGAN_TAICHI_SOURCE_KEY_VERIFY=1`. Keep compiler,
source paths and other kernel-affecting settings fixed. A persistent daemon or
a second pass inside the same process is not a disk-cache reuse test.

## Limits

These results do not establish that the historical T4 jobs were cache hits or
explain their exact first-pass costs. Different compiler builds, paths,
configuration and exercised template variants can legitimately change cache
behavior. The Mac measurements are not physical-Mac performance claims. No
cache-key dependencies were removed, no invalidation was weakened, and no
precompiled kernels were introduced in this change.
