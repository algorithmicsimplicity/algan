# Reusing the sheet depth-table buffer

## Change

The previous [matched profile](https://github.com/algorithmicsimplicity/algan/actions/runs/34190829021) identified `_lane_first_owners()` as the next
target: 19–21 seconds on MPS versus 0.834 seconds on CPU, with most MPS time
outside its two custom-kernel calls. The [detailed allocation trace](https://github.com/algorithmicsimplicity/algan/actions/runs/34189505030) attributed
17.516 seconds to `[N, 8]` allocations. Allocation timing includes any allocator
reclamation, driver work and waits; it is not a measurement of raw allocation
instructions alone.

The function already allocates an int32 owner-index table of `nb * 8` elements.
After the integer owner reduction completes, those indices are consumed only
once to gather float32 depths. The new `sheet_lane_depths_inplace` kernel reads
each index and replaces that same slot with the depth's float32 bit pattern.
The caller returns a float32 view of the storage. Missing owners still produce
positive infinity.

No thread reads another thread's output slot. There are no aliased kernel
arguments, persistent caches, extra queue fences, new arena allocations or
changes to batching policy. Every call owns its returned storage. The saved
logical allocation is 32 bytes per sheet; physical allocator savings can differ
because buffers and heaps are rounded or cached.

The original path remains the default: the experiment below did not establish
an end-to-end improvement. Opt into the candidate with
`SETTINGS.raytracing.experimental.sheet_depth_buffer_reuse = True`, or
`ALGAN_SHEET_DEPTH_BUFFER_REUSE=1` before import. This is a live host-side gate,
so both variants can be warmed and alternated in one process.

## Experiment

- Measured candidate: `6cb71cefb90221e2a72bfc2f6f4609cc05c7706c`, based on the
  previously merged fixes at `272d6965880927141d92d7794a1b7239415b6669`.
- Measurement commit: `33e302233ff103b22fb780dc0a8ea268399cbf20`; its only extra
  change is the Mac request file. [Run 81](https://github.com/algorithmicsimplicity/algan/actions/runs/34196201543).
- Harness: `benchmarks/_mac_depth_buffer_ab.py --matched`, run directly with
  `.venv/bin/python` to preserve the patched Quadrants wheel from run
  `33850787142`.
- Same neural-network/globe/text fixture, seed 20260908, UHD, 18 frames,
  shadows off, CPU animation, three Torch threads, Torch compilation disabled,
  software x264 encoding and a fixed 1720 MiB render arena.
- One VM: CPU-before (cold/warm), MPS (warm A and B separately, then measured
  A/B/B/A/A/B), CPU-after (cold/warm). A is the original separate allocation;
  B reuses the owner buffer. CPU controls use the candidate.
- Wall time surrounds `Scene.save_video()` through encoding completion.
  Scene authoring and memory snapshots are outside it. All arms share the same
  coarse scope timers; no ATen dispatch trace, cProfile, stack sampling, native
  method hooks, extra synchronization or extra cache clearing is enabled.
- The harness asserts the requested render backend, the bloom kernel gate,
  and exclusive engagement of the selected depth-gather kernel. It records
  chunk/batch counts and actual table sizes so a speedup cannot silently be
  credited to a smaller workload.

## Validation

Local Linux/Quadrants CPU validation: 3,416 tests passed, 150 skipped; all 14
compiler-setting tests passed separately with `ALGAN_TORCH_COMPILE` unset.
The broad run otherwise uses `ALGAN_TORCH_COMPILE=0`. Its fast render passes the
pixel baseline. Large-scene renders complete, but their unavailable reference
baselines prevent a visual comparison and are included among the skips.

New tests cover exact float bits (including signed zero and missing owners),
empty tables, nonzero storage offsets with canaries, repeated table sizes and
live A/B toggles, and retention of earlier results across later calls. An
independent owner-selection oracle checks the complete helper. Four local
PREVIEW videos, including both warmed variants, decode identically.
The candidate's local preview also matches the previous profiling harness's
warm preview in every decoded RGB frame.

On the Mac, all seven new focused tests passed on Metal with the patched
Quadrants 1.3.1 compiler, LLVM 22.1.0, Torch 2.7.1, Python 3.11.9 and macOS
26.6.2 arm64. CUDA was not available for validation.

All twelve UHD videos contain the expected eighteen 3840x2160 frames. All four
candidate MPS videos are decoded-RGB identical, including the last slow run;
original-path runs 3 and 7 match those videos exactly. Original runs 1 and 6
match each other but differ slightly from that shared result: 0.000331% of
channels change, maximum 6/255, mean absolute difference 0.00000635/255. This
same difference occurs between two original-path runs, so it is not specific
to buffer reuse. All four CPU controls match each other exactly. Cross-device
CPU/MPS pixel identity is not asserted.

## Measured result: do not enable by default

Times are seconds, excluding each arm's first render. The same arena, two
batches and eighteen chunks were verified in every render.

| Measurement | Original MPS A | Reused MPS B | CPU B |
| --- | ---: | ---: | ---: |
| Warm wall times | 114.1, 205.8, 200.7 | 126.3, 180.1, 376.7 | 104.6 before, 74.1 after |
| Mean wall time | 173.54 | 227.70 | 89.36 |
| Mean `_lane_first_owners` | 30.006 | 2.915 | 0.704 |
| Mean depth-gather call | 7.045 | 0.088 | 0.204 |
| Mean `compact_sheets` | 103.145 | 106.157 | 35.503 |
| Mean wavefront rendering | 149.093 | 181.363 | 75.423 |

These are nested host scope timers, not disjoint GPU kernel durations; do not
add them together. The helper's roughly 90% lower host time does not establish
that the complete render saved that time. Allocation can force deferred work
to finish, so avoiding one allocation can move waiting into a later operation.
The rest of sheet compaction became slower despite the helper improvement.

End-to-end, B's mean was 31.2% slower than A. Adjacent warm A/B comparisons
ranged from 12.5% faster to 87.7% slower for B. Both variants were slower than
the bracketing CPU controls. The CPU controls themselves changed by 29.1%, and
the GPU's baseline rose from 114 to about 201 seconds. These data show substantial
variation and do not isolate a causal whole-render regression or improvement.
The last B render is retained in the result; it is not silently discarded as
an outlier.

Each MPS render calls the helper eighteen times. Reuse avoids eighteen logical
depth-table allocations totaling 639,730,176 bytes (about 610.1 MiB over the
whole render), with a largest table of 36,166,720 bytes. This is not a reduction
of 610 MiB in peak memory. The earlier trace's 72 `[N,8]` allocations also
included other tables; the candidate does not remove all 72.

The process retained about 4.7 GB of MPS driver allocation near its 5.01 GB
recommended limit, with around 1 GiB of system swap and little live tensor
storage between renders. This is consistent with allocator/driver pressure,
but is not proof of its contribution or of virtualization overhead. This run
does not contain the synchronized device timing or physical-Mac control needed
to distinguish deferred waits, memory pressure and changing host conditions.

## Why the runner appeared stalled

The parent harness sent each child's output to a file and printed it only after
the entire child process exited. Thus the Actions log stayed at the completed
CPU-before block while all eight MPS renders were running. The uploaded
[artifact](https://github.com/algorithmicsimplicity/algan/actions/runs/34196201543/artifacts/10045444737)
contains all eight MPS `render_end` records and complete videos.

The GPU process nonetheless exceeded its 1,600-second process deadline and was
killed with recorded exit code 124. CPU-before and CPU-after exited normally.
The GPU's measured render walls sum to about 1,518 seconds; the remaining time
includes process initialization, scene authoring, snapshots and teardown.
Because the last `render_end` was emitted before the timeout, the timeout was
after the rendering loop's useful work. There is no stack from that old
process, so the exact teardown operation cannot be identified retrospectively.
The workflow failed because of that timeout, not a failed pixel/kernel test.

The harness now forwards each child's log live while preserving the file,
emits timestamped chunk start/end and explicit block start/end records, and
marks completion of all renders. A timed-out Mac child receives a bounded
native `sample` capture before its process group is killed. A Python traceback
timer is armed after the final render to diagnose slow teardown without
sampling the measured renders. Local regression tests cover live visibility,
nonzero exit codes/stderr and bounded timeout with retained output. These
harness changes have not themselves been rerun on the Mac.

Decision: retain the buffer-reuse implementation as a disabled experimental
candidate in the draft PR. Pixel correctness is encouraging, but this measured
result does not justify shipping it as an enabled performance improvement.
