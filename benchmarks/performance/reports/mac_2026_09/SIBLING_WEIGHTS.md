# Fused sibling coverage on the Mac runner

## Candidate

`_sibling_weights()` previously built member counts and disjoint-run counts,
gathered band properties, counted sample bits and produced continuation flags
through a sequence of PyTorch operations. It also read `multi.any()` back to
the host to decide whether to continue. Prior Mac profiles showed several
seconds in this helper per UHD render.

The candidate replaces that sequence with two kernels:

1. Count each band's members and disjoint runs using int32 atomic additions.
2. Calculate the coverage and mask for each sheet. Only a band with multiple
   sheets in one uninterrupted run receives shared coverage. Interleaved bands
   keep each sheet's own coverage and mask. A continuing sibling has negative
   weight; the last sibling closes the band. Partial unions retain the original
   sample-count factor, and zero areas retain the `1e-12` divide floor.

The integer reductions are order-independent. Arithmetic uses the existing
accumulator policy: float32 in MPS-friendly mode, the wide accumulator where
supported. The renderer still selects the same sheet order and coverage rules.
There is no new arena, persistent scratch cache or change to chunk sizing.

The live A/B gate is
`SETTINGS.raytracing.experimental.sheet_sibling_weights_kernel`; its environment
seed is `ALGAN_SHEET_SIBLING_WEIGHTS_KERNEL`. The old Torch path remains the
reference. The earlier depth-buffer-reuse experiment stays off in both arms.

## Measurement

- Candidate commit: `9643f0a6e546f047efa1cbf68e114305ecb21c36`.
- Measurement commit: `57cb6d34a03c8777e2c95045044c50432485b1cd`. Its only extra
  change is the Mac request file.
- [Run 85](https://github.com/algorithmicsimplicity/algan/actions/runs/34206388060),
  using the patched Quadrants wheel from build `33850787142`.
- Direct venv interpreter:
  `benchmarks/_mac_depth_buffer_ab.py --target sibling --matched --sequence ABABBA`.
- One VM, same neural-network/globe/text fixture and seed 20260908. UHD,
  shadows off, CPU animation, three Torch threads, Torch compilation disabled,
  software x264 encoding, and fixed 1720 MiB render arenas.
- CPU-before: candidate cold/warm. GPU: warm A and B, then measured A/B/B/A.
  CPU-after: candidate cold/warm. A is the original Torch helper; B is the two
  kernels. The GPU order balances a linear drift across the four warm runs.
- Wall time spans `Scene.save_video()` through encoding. Scene authoring and
  memory snapshots are outside it. Scope timers are nested host timers, not
  exclusive GPU kernel durations. No extra device synchronization or heavy
  profiler is inserted into timing.
- Per-render assertions verify which kernels ran. Events retain chunk/batch
  counts, arenas and import-cache memory at chunk boundaries. The parent
  streams logs live and separately bounds teardown after the last render.

## Local validation

Linux CPU, Quadrants 1.3.0, Torch 2.8.0+cpu and Python 3.12.13:

- Full suite with the candidate enabled: **3,446 passed, 150 skipped**, 2454.76 s.
  Large-scene renders completed, but their unavailable reference baselines
  prevent their pixel comparisons. The fast render's pixel baseline passed.
- All **14 compiler-setting tests** passed separately with
  `ALGAN_TORCH_COMPILE` unset; the broad run used `ALGAN_TORCH_COMPILE=0`.
- Fast suite: **539 passed** with the candidate enabled.
- Final focused kernel and harness checks: **29 passed**. Cases include empty
  and single-sheet streams, uninterrupted and interleaved bands, long runs,
  exact masks and signs, tiny area floors, partial and empty unions, storage
  offsets and canaries, repeated toggles and preservation of earlier outputs.
- Four local A/B/B/A preview videos are identical in all decoded RGB frames.
  Each contains three 704x396 frames. Kernel counters confirm that only B
  engaged the replacement kernels.
- Ruff and whitespace checks passed. CUDA was not available for measurement.

## Mac result

**Keep this candidate disabled.** Run 85 completed successfully, but the warm
candidate mean was **14.9% slower** than the original. This is a failed
whole-render optimization despite a smaller host time inside the target helper.

All times below are seconds for a complete 18-frame 3840x2160 video, including
software encoding, on this one VM. Cold A and cold B are excluded from the warm
comparison; results from other runs or differently sized arenas are not mixed
into this table.

| Variant | Warm render times | Mean |
| --- | --- | --- |
| Original GPU A, runs 3 and 6 | 134.357, 150.244 | 142.300 |
| Candidate GPU B, runs 4 and 5 | 146.002, 181.095 | 163.549 |
| Candidate CPU B, before and after | 93.547, 78.532 | 86.039 |

The adjacent A3/B4 comparison is 8.7% slower for B; the reversed A6/B5
comparison is 20.5% slower. These are two warm observations per GPU variant,
with substantial variability, not a precise estimate of performance on every
Mac runner. They provide no evidence for enabling B. Candidate GPU B is 90.1%
slower than the mean of the CPU B brackets in this run; both use the same
candidate algorithm. The CPU brackets themselves differ by 15.0 seconds.

For completeness, cold GPU A1 was 217.532 seconds and cold GPU B2 was 154.923
seconds. These pay different initialization costs and are not an A/B speedup.
CPU-before's first render took 139.369 seconds and CPU-after's first took
76.881 seconds; the latter can benefit from the compiler's on-disk cache.

### What the scope and memory diagnostics show

| Nested host scope | Warm GPU A mean | Warm GPU B mean |
| --- | --- | --- |
| `_sibling_weights` | 17.678 | 13.652 |
| `compact_sheets`, including sibling weights | 85.888 | 93.911 |
| `raytrace_render_wavefront`, including compaction | 120.290 | 136.496 |

Sibling weights' host time fell 22.8%, but compaction and full rendering did
not improve. These scopes overlap and must not be added together. Within B,
the first count-kernel call accounts for 12.431 seconds and the second
coverage-kernel call for 0.824 seconds per warm render, averaged over 18 calls
to each. This is **not** evidence that the integer atomics themselves take
12.431 GPU seconds: `mps_zero_copy.zero_copy_call()` synchronizes Torch before
the launch and Taichi after it. The first kernel can therefore inherit the
wait for preceding Torch operations. Replacing arithmetic in a host-timed
hotspot can move where queued work is waited for without removing that work.
Device execution and each side of this queue handoff need separate measurement
before selecting another kernel from these host timings.

All MPS launches used the zero-copy bridge, with zero staged MPS or host
arguments. Warm A used 1125 converted launches per render and B used 1143,
a net increase of 18. B's replacement comprises 36 new-kernel calls over the
18 chunks. This confirms engagement; it does not demonstrate fewer total
driver submissions, since the Torch operations have their own submissions.

The maximum bytes held by the import cache at the recorded warm chunk ends
were 2243.7 MiB for A and 2273.8 MiB for B, including the same 1720 MiB arena.
B's recorded maximum was 30.1 MiB larger. These are sampled retained storage
sizes, not peak GPU allocation or a measurement of how much retained storage
was dead. Both arms cleared the import cache by each render end. Driver
allocation stayed around 4.71 GB against a 5.01 GB recommended maximum; the VM
went from zero swap at GPU initialization to about 1.3 GiB used during GPU
rendering. Memory pressure is observable, but these snapshots do not measure
paging time or establish that it caused the regression. This run also does not
separate virtualization cost from framework, driver or synchronization cost.

### Completion and correctness

- The Mac focused suite passed **15 tests**; 10 float64-policy cases were
  skipped because Metal does not support that policy. The actual render
  metadata reports `Arch.metal`, Torch 2.7.1 and patched Quadrants 1.3.1.
- All ten renders finished with two batches, 18 chunks and the exact same
  1803550720-byte arena. B's two kernels ran 18 times each; A ran neither.
  Every render retained the original depth-allocation path.
- All three process blocks exited with code 0. The GPU process exited about
  16.7 seconds after its `renders_complete` marker. Neither the overall timeout
  nor the separate teardown deadline fired; no render or unfinished process
  was omitted from the result.
- All ten videos decode to 18 UHD frames. All four CPU videos match exactly.
  The six GPU videos fall into two decoded-RGB sequences:
  A1/B4/B5 match exactly, and B2/A3/A6 match exactly. Every candidate video
  therefore matches an original video over every decoded channel; there is no
  candidate-only output sequence.
- Comparing the two original sequences, A1 versus A3, changes 0.000331% of
  decoded RGB channels, with mean absolute difference 0.00000635 on the 0-255
  scale and maximum 6. These tiny differences already occur within A. The
  candidate introduces no new decoded output beyond that observed variation.
  Representative frames from both sequences were also inspected visually.

The implementation and its tests remain available behind the opt-in gate for
further experiments. Both `sheet_sibling_weights_kernel` and
`sheet_depth_buffer_reuse` remain **false by default**. The useful retained
change is the benchmark's live progress and bounded teardown diagnostics;
neither experimental renderer change has earned a default-on switch.

### Reproducible evidence

[Run 85 artifact](https://github.com/algorithmicsimplicity/algan/actions/runs/34206388060/artifacts/10048848862)
contains the videos, timestamped events, process logs and `outcomes.json`.
The artifact ZIP SHA-256 is
`cd248e0b78e7a3ea6c7ff583c911609d848d4c15801638615691991504609df5`.
The separate evidence bundle retains that original archive, complete job log,
analysis and decoded-frame hashes, difference statistics and local test logs.
