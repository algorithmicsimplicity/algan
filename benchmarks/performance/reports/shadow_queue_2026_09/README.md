# Light-major shadow-ray queue: implementation and Mac measurements

Base: `325ac0f2aeafd852fed932b10faf0bf93ecefba0` on
`claude/peaceful-carson-px98kh`. Measured 2026-09-12 on the repository's Mac GPU
runner, using the published `algan-quadrants==1.3.0.post2` distribution. The
runtime reported `mps` and Quadrants `arch=metal`; this was not a CPU fallback.
No Kaggle/CUDA measurement was performed.

## Status and switches

The implementation is complete as an **opt-in experiment**, not a demonstrated
whole-render speedup on Metal. The primary shadow pass improved, but the
secondary queues and the measured full-frame median regressed. The serial
path therefore remains the default on all devices.

```python
SETTINGS.raytracing.experimental.shadow_ray_parallel = True
SETTINGS.raytracing.experimental.shadow_secondary_sort = True
```

`ALGAN_SHADOW_RAY_PARALLEL=1` enables the same path before import.
`shadow_secondary_sort` / `ALGAN_SHADOW_SECONDARY_SORT` defaults to true within
that path; turn it off to isolate light-major traversal from secondary sorting.
Both settings can change between renders without resetting compiled settings.
This changes scheduling, **not** the branch's existing shadow sample budgets.

## Execution and memory contract

The serial reference and queue emitter share `_shadow_fan_cell`: light-type
handling, sample positions, jitter, coverage masks, terminator displacement,
horizon rejection and geometric zero-radiance culling use the same source.
Queue entries are light-major sample planes, and each traversal worker handles
one ray. Rays retain original sample-slot IDs regardless of execution order.
Secondary queues of at least 8,192 padded slots are stably ordered by light,
actual ray-direction octant, and source triangle. Scheduling source IDs do not
change intersection identity or epsilon rules: the deferred-secondary
acceptance identity remains -1.

Adaptive hard-light taps remain two-phase: trace the diagonal pair first, then
trace off-diagonal taps only where the pair did not terminate the reference
fan. Reduction reads accepted samples in their original order, retaining RGB
transparency and adaptive doubling. No float atomics are used for this sum.
Splitting generation and traversal into different kernels nevertheless need not
produce bit-identical floating-point results on every backend; see below.

All large queue payloads and sort indices are temporary arena storage. Fan
bounds include all packed frame rows and custom budgets. The dispatcher chunks
events to fit, caps payload storage at 64 MiB, and restores arena pointers on
success or failure. Insufficient scratch for one event uses the serial path;
no rays are dropped. Backend sort workspace and tiny cached layout tables use
the renderer's existing external headroom.

The light-layout cache uses the CPU table retained during light packing. Arena
slices share a tensor version counter, so keying only on a GPU light slice's
version invalidated the cache on unrelated scratch writes and caused avoidable
readbacks. The retained host metadata fixes that; weak ownership and a bounded
cache prevent keeping entire old arenas alive. Standalone callers without host
metadata still have a version-aware fallback. This case has a regression test.

## Mac GPU results

### Graphics workload, final renderer implementation

[Workflow run 34696768088](https://github.com/algorithmicsimplicity/algan/actions/runs/34696768088)
succeeded. Raw output: [mac_graphics.txt](mac_graphics.txt); machine-readable
readings: [mac_graphics.json](mac_graphics.json).

This is `graphics_scene.py` at time 0.5 s, **one 704 x 396 frame**, not the
original report's 60-frame video or its UHD benchmark. Each full-frame timing
includes rebuilding the authored scene, rendering and saving a PNG. Every arm
was warmed first, followed by mirrored serial/unsorted/sorted and reverse
orders, yielding four warm readings per arm.

| Arm | Warm full-frame median |
| --- | ---: |
| Original serial fan | 4.7904 s |
| Ray-parallel, secondary sorting off | 6.5865 s |
| Ray-parallel, secondary sorting on | 5.5454 s |

The sorted path's median was **15.8% longer** than serial in this run. Full-frame
readings were noisy (serial ranged from 4.08 to 6.06 s), so this is not a precise
estimate of a universal regression. It is certainly not evidence for enabling
the new path by default. Sorting was better than the unsorted parallel arm in
this set, but that does not make either faster than the serial renderer.

The matched-input queue measurements separate shadow dispatch from scene
construction and the other renderer stages. They synchronize before and after
each call and mirror serial/parallel order:

| Queue | Serial | Ray-parallel with secondary sorting |
| --- | ---: | ---: |
| Primary, 87,907 events | 38.98 ms | 23.12 ms (**40.7% less**) |
| First secondary, 66,337 events | 20.92 ms | 56.94 ms |
| Remaining secondary queues | See raw readings | All slower in this run |

These are wall times for the complete queue operation, including generation,
sorting when enabled, traversal and reduction, **not device-only kernel times**.
The parity/isolated-queue pass forces sorting even below the normal threshold
to exercise it; full-frame timings use the shipped 8,192-slot threshold.

The useful result is that primary ray parallelism can pay on this workload,
whereas this implementation's secondary queue overhead does not. The measurement
does not isolate sorting, payload traffic and extra dispatches well enough to
assign each a precise share. More direct traversal or less secondary dispatch
work would need a new measurement; no CUDA conclusion follows from Metal.

### Correctness

The final graphics run compared **6,120,216 scalar visibility values** across
one primary and eight secondary queues on identical inputs. 5,808 scalar values
were not bit-identical; the maximum absolute difference was
**2.980232238769531e-7**. The initial strict graphics check stopped on a
1.1920928955078125e-7 difference. The later run explicitly requested an absolute
visibility tolerance of `2e-6`; the harness still defaults to exact comparison.
This tolerance and the observed difference are recorded, not hidden.

Most importantly, the saved graphics frames for serial, parallel-unsorted and
parallel-sorted had **zero differing channel values**. The frame assertion
remained the repository's independent maximum tolerance of 2, irrespective of
the requested visibility tolerance. Queue pointer restoration also passed.
These fixtures are evidence for the tested cases, not a claim of universal
byte identity or a complete full-render baseline sweep.

The earlier soft-light fixture in
[run 34695900098](https://github.com/algorithmicsimplicity/algan/actions/runs/34695900098)
passed strict equality across **515,160** scalar visibility values, one primary
and one secondary queue. Saved serial and parallel frames were identical.
Its warm medians were 0.7708 s serial and 0.7765 s parallel (no meaningful gain);
[mac_soft_initial.json](mac_soft_initial.json) preserves the readings. That run
predated the retained-host-metadata cache fix.

## Test coverage and limitations

The new tests cover live switches, custom/multi-frame fan bounds, masks,
RGB/adaptive reduction, actual direction sorting keys, bounded chunking,
fallback, arena restoration including exceptions, and host-metadata cache
invalidation. The existing arena-binding tests cover the new traversal kernel.
The deferred-sample tests now instrument the dispatcher rather than a removed
module-local import. Their masked-fan fixture explicitly selects the legacy
sample budget; the assertion that sub-pixel collapse changes that legacy soft
fan is retained.

Validation record:

* Final renderer: 36 queue/arena/budget tests passed on the Mac GPU; the same
  group passed locally on CPU. The earlier Mac run also passed the broader
  47-test group including area-light and transparent-shadow tests.
* 61 CLI/environment checks passed after restoring alphabetical registry order.
  The eight compiled-setting checks passed in the preceding combined run.
* A CPU control render from the untouched base and both new paths matched
  byte-for-byte at 64 x 36. The matched-input CPU fixture also matched exactly
  across 7,185 scalar visibility values.
* The initial opt-in fast run had 605 passes and one rendered-video baseline
  failure, max channel delta 221 at frame 4. The untouched base reproduced the
  same delta at the same frame in this container; no baseline was replaced.
* A full-suite attempt was stopped after 15.5 minutes, with 541 passes and
  three failures: an externally set daemon variable conflicting with a CLI
  assertion, plus two tests using the removed instrumentation point. CLI passed
  with that variable unset; the instrumentation was updated as described above.
  **The complete full suite was not finished.**

The final opt-in fast rerun again had **605 passes and the same sole baseline
failure** (221 at frame 4, 159.52 s). After updating the instrumentation and
pinning the legacy masked-fan fixture, **all 10 deferred-sample tests passed**
with ray parallelism enabled (112.52 s).

Ruff checks, non-kernel formatting checks, Python compilation and
`git diff --check` were also run. No kernel source was run through Ruff format.

## Reproduction

```sh
ALGAN_RENDER_DEVICE=mps ALGAN_ANIMATION_DEVICE=cpu ALGAN_USE_DAEMON=0 \
  .venv/bin/python benchmarks/_shadow_queue_check.py \
  --scene graphics --width 704 --height 396 --pairs 2 --queue-pairs 1 \
  --sort-ab --atol 0.000002 --require-mps
```

Omit `--atol` for strict visibility equality. `--require-mps` rejects a CPU
fallback. The harness prints results after each queue/render and writes JSON
and frames under `algan_outputs/shadow_queue/<scene>/`.
