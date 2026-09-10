# Metal sheet grouping: recovered evidence and opt-in candidate

Report date: 2026-09-10. The source patch is based on master
`d055e77f10dc28e3908329158b2b41d80826e9da`. Its source archive was downloaded
from Actions run `34416430692`, artifact `10129243743`, with SHA-256
`d570c7847230325eab6debcd7fd9f148be51bb63defd698e150f91099ae3f594`.

**Decision: leave `sheet_mps_grouping` disabled by default.** Local grouping
and metadata kernels are much faster in isolated operation tests, but the
whole-video comparisons disagree. In particular, the last zero-copy comparison
was slower. This patch does not claim a verified warm wall-time improvement.

## What the patch contains

The independent correctness fixes retain integer precision when gathering
shading classes and grouping sorted IDs on the locked Metal implementation.
Adjacent values above `2**24` must remain distinct. The fallback compares integer
boundaries and constructs inverse IDs with an integer prefix scan instead of
routing wide keys through the problematic unique operation. CPU and CUDA retain
their existing grouping choices.

The opt-in implementation sorts `(band, class, original position)` within existing
surface/facing runs. Those runs own disjoint, increasing band ranges, so local
sorting preserves global lexicographic group IDs. Short runs use insertion sort;
long runs use heapsort, without a fixed fragment-count limit or truncated scan.
A prefix scan and scatter construct the inverse map and the group's band ID.
A separate boundary/scatter pair handles already sorted pixel IDs.

A fused metadata kernel unpacks pixel/frame/primitive identifiers, facing and
group keys. Depth is a bit reinterpretation, not a numerical conversion. It does
not change coverage, shading, tessellation, sampling, resolution, memory budgets
or retry policy. Boolean flags are passed as an unsigned-byte view of the same
storage, avoiding host staging through the Metal buffer bridge.

The kernels require a live, local Metal backend, contiguous integer inputs and
fewer than `2**31` entries. Small streams retain the reference path. Wrapper
checks reject unsupported shapes, dtypes and devices before launching. Empty
streams do not import null Metal buffers. The final local hardening also changes
the heapsort internal-node test to avoid overflowing a 32-bit child-index
expression for a large leaf. These last validation changes have CPU coverage,
not a new Mac run.

Temporary arrays are ordinary PyTorch allocations, not arena allocations. The
class operation briefly holds three 32-bit work arrays while creating the
prefix scan, then two during scatter, plus its 64-bit inverse and group-band
outputs. These buffers still require external headroom; a faster
operation alone does not prove that the complete render uses less memory.

Enable only for measurement:

```python
from algan import SETTINGS

SETTINGS.raytracing.experimental.sheet_mps_grouping = True
```

`ALGAN_SHEET_MPS_GROUPING=1` seeds the same setting at startup. Setting it to
false restores the non-kernel route, but deliberately retains the independent
integer-correctness fixes. The benchmark's optional reference-commit argument
can load the two old grouping functions for a comparison with their original
behavior. It is not a second complete source checkout.

## Workload and measurement method

The experiments load the factory, video settings and encoder arguments directly
from `benchmarks/performance/nn_scene_UHD.py`: 3840 by 2160, 60 frames per second,
30 frames, SSAA 2, and software libx264 with CRF 17 and the ultrafast preset.
The runner reports three CPU threads, 7168 MiB RAM, macOS 26.6.2, Python 3.11.9,
PyTorch 2.7.1, `algan-quadrants` 1.3.0.post2, and actual `Arch.metal` execution.
Animation stays on CPU; the daemon and Torch compilation are off for the
sheet-grouping comparisons. No memory override was installed.

Both arms were warmed in the same process before alternating measured renders.
The timer covers the complete `Scene.save_video()` operation and the final
device synchronization: preparation, rendering, output copies and encoder drain
are included. Scene authoring and interpreter teardown are outside the timer.
Separate unmeasured renders compare all pre-encoder frames with a maximum
allowed channel deviation of two. Absolute times from different runner jobs
are not comparable controls.

The numerical summaries, raw timing arrays, run URLs, source bases and artifact
hashes are recorded in [`sheet_grouping_results.json`](sheet_grouping_results.json).

## Recovered whole-video results

Positive reduction means less time. Means and medians are both shown because
they can disagree materially on these small, drifting samples.

| Experiment | Warm samples per arm | A median | B median | Median reduction | Mean reduction | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CPU PN compilation confirmation | 3 | 142.07 s | 143.81 s | -1.23% | +5.12% | Earlier 4.7% estimate is not confirmed |
| Bounded speculative batch growth | 2 | 129.06 s | 100.22 s | +22.35% | +22.35% | Reject: raw parity fails |
| Local class and sorted-ID grouping | 3 | 157.30 s | 139.56 s | +11.28% | -1.54% | Inconclusive |
| Grouping plus metadata fusion | 3 | 122.89 s | 113.58 s | +7.58% | +6.44% | Promising single comparison, not conclusive |
| Combined candidate with byte-view zero-copy flags | 2 | 100.50 s | 115.01 s | -14.44% | -14.44% | Final retest is slower; keep opt-in |

CPU compilation and batch growth used base
`c2075083a49cb21bf8c7d6f46139eeb69d3e80ee`. The grouping work used
`795044725a540687b1a34380b8a6046358e7cdbe`. Current master has since changed its
CPU garbage-collection pressure handling and packaging. No new Mac measurement
on current master was possible in this continuation, so these measurements
must not be relabeled as results for the final patch.

The batch-growth comparison failed on eight frames, with 12 pixel positions
in total exceeding the channel tolerance and a maximum deviation of 36. It is
not accepted as a performance improvement with equivalent output. No tolerance
was relaxed, and the patch does not contain the batch-growth change.

All three grouping comparisons passed 30 raw frames, with maximum channel
deviation one and zero pixels above the threshold. The final zero-copy run also
reported zero staged arguments, zero host arguments and no staging reasons.
Correct pixels and successful zero-copy engagement do not establish a wall-time
improvement.

The first grouping job failed during interpreter teardown after writing all
its completed render timings and passing parity result. Its measurements are
retained as completed-render evidence, not described as a passing workflow.
Later supervision excluded that vendor destructor from the save-video metric;
render and parity exceptions still failed before the success-only process exit.

## Isolated operation results

These probes captured the scene's real 2,565,039-fragment stream, rather than a
synthetic replacement workload. They synchronized around each operation, so
their timings must not be substituted for the whole-render metric.

| Operation | Reference median | Candidate median | Operation speedup |
| --- | ---: | ---: | ---: |
| Class grouping | 146.53 ms | 11.49 ms | 12.75 times |
| Unique sorted IDs | 7.22 ms | 4.92 ms | 1.47 times |
| Fragment metadata | 16.96 ms | 4.77 ms | 3.55 times |

Grouping probe: [run 34413681479](https://github.com/algorithmicsimplicity/algan/actions/runs/34413681479).
Metadata probe: [run 34415376015](https://github.com/algorithmicsimplicity/algan/actions/runs/34415376015).
The grouping and metadata results were checked against integer CPU oracles.

A separate dependency-upgrade attempt,
[run 34417422774](https://github.com/algorithmicsimplicity/algan/actions/runs/34417422774),
finished its PyTorch 2.7.1 control but failed while installing the requested
`torchaudio==2.12.0`. It never produced a newer-Torch comparison. No dependency
upgrade or performance claim is included here.

## More diagnostic benchmark

`benchmarks/performance/nn_sheet_grouping_ab.py` now defaults to `ABBABAAB`.
Every adjacent pair contains A and B, and both arms have the same average
position in the sequence. It reports all samples, both mean and median
reductions, and adjacent-pair ratios. Warmup and parity renders cannot enter the
reported warm samples; unit tests enforce that accounting.

The harness counts actual nonempty helper calls, not just a preliminary gate
condition. It records source hashes, live rendering settings, memory overrides,
and per-render zero-copy counter deltas. Coarse host operation timers add no
stage synchronization, and can include earlier queued work. They are attribution
hints, not device execution timings. The parity consumer also checks that both
arms supply the authored frame count.

Run from the repository root on the Mac harness with the locked compiler:

```bash
.venv/bin/python -u benchmarks/performance/nn_sheet_grouping_ab.py \
  --component all --sequence ABBABAAB \
  --out algan_outputs/sheet_grouping_confirm
```

For original pre-fix grouping behavior, first fetch the immutable reference and
pass `--reference-commit 795044725a540687b1a34380b8a6046358e7cdbe`. Without that
argument, A uses the current integer-correct reference path. `--component class`,
`unique` and `metadata` isolate individual candidates. Use a fresh output directory.

The benchmark's final instrumentation changes have unit coverage but have not
been executed on Metal. A controlled repeat is still required before enabling
any optimization by default. Attribute the remaining complete-render variability
with the new settings, counter and operation records rather than selecting the
most favorable median or comparing different runners' absolute times.

## Validation and publication

The exact recovered byte-view prototype passed 134 targeted tests, with three
skips, on the Mac in
[run 34417861733](https://github.com/algorithmicsimplicity/algan/actions/runs/34417861733).
It also passed the 30-frame UHD raw comparison described above. Those results
precede the final local hardening and disabled-by-default setting.

Local validation results for the final source are recorded below.
The local environment follows `LOCAL_INSTALL.md`: editable source, CPU rendering,
Python 3.13.5, PyTorch 2.10.0+cpu and the supplied patched compiler build. These
are correctness tests, not Mac speed measurements. The upstream test environment
and compiler versions differ from the locked Mac environment.

The current chat exposes GitHub read operations but no file-write, Git-data-write,
workflow-dispatch or PR-creation actions. Therefore no new Mac job was submitted
and no remote branch or PR was published in this continuation. The deliverable
is a source patch on the recorded current-master base, without the temporary
experiment workflows or staging payloads from the earlier branches.


### Final local results

- Focused grouping, metadata, benchmark-accounting, reuse, environment and
  MPS-compatibility tests: **160 passed, 1 skipped**, in 95.79 seconds. The skip
  requires the actual Metal buffer bridge.
- Fast suite: **561 passed, 1 failed**, with 3527 deselected. The failure is
  `tests/fast/test_fast_render.py`, maximum channel difference 221 at frame 4.
  Running that render test on the unmodified `d055e77` source reproduces exactly
  the same 221-value failure at frame 4. This matches the installation runbook's
  documented `dvisvgm` compatibility-wrapper limitation. No baseline was changed.
- Full configured suite (`pytest -q --maxfail=3`): did not finish within the
  explicit 300-second validation limit; exit code 124. This is not a passing
  full-suite result.
- All nine changed/new Python files pass AST compilation. Kernel modules retain
  runtime annotations without `from __future__ import annotations`.
- `git diff --check` and applying the complete patch with `git apply --check`
  against the clean, recorded master source pass.
- Ruff was unavailable in the offline local overlay, and an installation attempt
  could not obtain its wheel. The final edits do not have a new Ruff result.
  Earlier Mac lint results apply only to the tested prototype.

The review bundle contains the raw local logs, exact environment versions,
source hashes, recovered Mac JSON records and the previous tested prototype
patch. The final local hardening and changed benchmark have not been represented
as Mac-tested source or as a measured performance improvement.
