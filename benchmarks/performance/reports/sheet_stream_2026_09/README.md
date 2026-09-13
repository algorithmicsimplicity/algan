# Fused sheet streams and exact pixel runs

Base: `b116338e09956448f581db1819ba1f04cadc651d` on the existing
`claude/peaceful-carson-px98kh` optimization branch. The implementation landed
as `687ab5e5` followed by the persistent-output follow-up `b212895a`.

## What changed

`sheet_fused_stream` replaces the sorted fragment gathers and group-boundary
comparisons with one kernel. It does not materialize a sorted group-key array.
The final record kernel composes the final-sheet, nearest-fragment and
dominant-fragment permutations directly. Packed 64-bit keys are copied as
integers, not converted through floats, and the final barycentrics/reference
still come from the dominant fragment while depth comes from the nearest
fragment. Sorted scratch uses the arena; returned sheet records own their
storage before that scratch is released.

`sheet_device_runs` detects pixel boundaries from packed keys, uses an integer
prefix scan, and scatters the covered-pixel list and CSR offsets. An exact
shape still requires one scalar readback; this is not a zero-synchronization
pipeline. Opaque-prefix truncation computes each run's retained length and
writes its gather indices directly. Because every run retains at least its
first fragment, the covered-pixel list is reused and a second
`unique_consecutive` is unnecessary. There is no full-stream keep mask or
`nonzero` result in this path. Sequential truncation gathers are deliberately
retained: replacing them with a fused gather would keep both complete input
and output streams alive simultaneously.

`sheet_fragment_run_sort` performs one global stable primary-key sort instead
of global layer and primary-key sorts. It sorts descending geometry layer only
inside equal primary-key runs using the existing stable run sorter. Original
positions break ties. Depth-bin construction remains the reference PyTorch
expression; no quantization beyond the existing depth rule is introduced.
Long runs use heapsort rather than a bounded local array, so arbitrary
transparent-stack depth does not create a new fragment ceiling.

Coverage arithmetic, sample ownership, shell ceilings, representative selection,
sibling weights and resolve rules are unchanged. The new paths require a local,
initialized compiler and fewer than 2**31 fragment slots; otherwise dispatch
retains the reference. Scan and sorted-stream scratch have separate arena
lifetimes, and the explicit discovery reserve includes their conservative
bounds. Existing out-of-memory retries remain in place.

## Switches and reproduction

```python
SETTINGS.raytracing.experimental.sheet_fused_stream = True
SETTINGS.raytracing.experimental.sheet_device_runs = True
SETTINGS.raytracing.experimental.sheet_fragment_run_sort = True
```

All three are off by default. Each has an `ALGAN_SHEET_...` environment counterpart. They are live host-side
switches, not values captured by a `ti.static` setting. The benchmark disables
the previous experimental parallel-shadow path to isolate this work.

```sh
ALGAN_RENDER_DEVICE=mps ALGAN_ANIMATION_DEVICE=cpu \
  .venv/bin/python benchmarks/_sheet_stream_check.py \
  --scene explainer --quality PREVIEW --frames 60 --pairs 2 \
  --arms reference,fused,runs,sort,all --require-mps
```

The harness warms each arm, mirrors its execution order, emits a line after each
complete video and checks decoded-frame parity. Lossless H.264 is requested so
lossy encoding does not magnify small renderer differences. Timings include
scene authoring, rendering and video output; they are not kernel-only timings.
The recorded clips preserve the workload's complete normalized storyboard.
Different frame counts/presets must not be compared as equivalent workloads.

## Mac GPU complete-video measurement

[Run 34724812631](https://github.com/algorithmicsimplicity/algan/actions/runs/34724812631)
validated the published implementation commit
`687ab5e5ef029dc7df4fb70f6d4ba2f4100e9885`. The environment reported `mps`,
Quadrants started on `arch=metal`, and all 19 final stream tests passed
(18.78 seconds). No CUDA measurement was performed.

The benchmark rendered the complete 60-frame, 704 x 396 explainer clip,
including authoring and lossless video output. Each arm was warmed first;
then mirrored A/B ordering produced four warm readings per arm:

| Arm | Warm readings (s) | Median (s) | Mean (s) |
| --- | --- | ---: | ---: |
| Reference | 11.087, 9.153, 8.433, 8.327 | 8.793 | 9.250 |
| All three changes | 11.069, 8.542, 9.658, 8.056 | 9.100 | 9.331 |

The new path's median was 3.5% longer; its mean was 0.9% longer. The readings
vary enough that neither percentage is a precise general regression estimate,
but this is **not a demonstrated whole-render speedup**. All three gates stay
off by default. The direct old/new videos have zero differing decoded channel
values over all 60 frames, and all five helper operations engaged on two chunks
per optimized render. The complete machine-readable summary and raw transcript are preserved on the linked workflow run.

A follow-up Metal run measured the three switches independently rather than
only as a bundle ([run 34726009100](https://github.com/algorithmicsimplicity/algan/actions/runs/34726009100)).
Each arm again rendered the complete 60-frame PREVIEW explainer and matched the
reference video exactly. Four mirrored warm readings gave:

| Arm | Warm readings (s) | Median (s) | Median vs reference |
| --- | --- | ---: | ---: |
| Reference | 7.686, 6.336, 8.976, 6.448 | 7.067 | -- |
| Fused sorted/final stream | 8.547, 5.821, 12.509, 7.155 | 7.851 | +11.1% |
| Device pixel runs | 7.503, 5.543, 11.427, 7.000 | 7.252 | +2.6% |
| Primary-key + run-local layer sort | 9.064, 8.098, 10.070, 8.115 | 8.590 | +21.5% |

The run is noisy, but none of the original individual changes demonstrated a
whole-video speedup on this Metal runner. In particular, eliminating a pass in
isolation is not sufficient when the replacement adds extra kernel dispatch or
temporary traffic.

### Persistent fused production records

The fused path still copied seven final production arrays into persistent arena
storage after its final gather. A follow-up removes those launches: the final
gather writes `sheet_key`, `sheet_ref`, `sheet_ab`, and `sheet_cap` directly to
reverse-arena storage, while the post-compositing weight/mask/CSR results are
persisted once before the temporary compaction scope is released.

[Run 34726462126](https://github.com/algorithmicsimplicity/algan/actions/runs/34726462126)
measured this revised fused path on the same complete 60-frame PREVIEW explainer.
All 20 stream tests passed on Metal first. Six mirrored warm readings per arm
were:

| Arm | Warm readings (s) | Median (s) | Mean (s) |
| --- | --- | ---: | ---: |
| Reference | 7.627, 6.551, 5.529, 6.277, 5.671, 6.900 | 6.414 | 6.426 |
| Persistent fused stream | 6.249, 5.904, 6.638, 5.516, 6.646, 5.582 | 6.077 | 6.089 |

The revised fusion is **5.26% faster by median and 5.24% by mean end to end**
on this run. All 60 decoded frames are pixel-identical. `gather_group_stream`
and `gather_sheet_records` engaged twice per render. This is the first sheet
stream variant in this round that demonstrates a complete-video speedup rather
than only reducing an attributed substage. The complete machine-readable summary and raw transcript are preserved on the linked workflow run.

## Validation

The two-frame CPU smoke clip and a 12-frame 704 x 396 explainer video were
pixel-identical with all three paths enabled. The latter exercised two chunks
per render. Its 2.981/2.934-second reference/all warm medians are only a two-reading smoke
comparison, not a robust CPU speedup claim.

The final focused local CPU group passed 130 tests (stream operations, existing
compaction/group-reuse/depth/pixel-sort checks, and persistent arena lifetime).
The persistent-output test overwrites reclaimed forward scratch before checking
returned records. They include empty streams, long runs (8,193 fragments), gapped pixels,
packed-key low bits, mixed geometry layers within depth bins, stable ties,
opaque-prefix edge cases, and cleanup after an injected kernel failure.

The opt-in fast suite passed 606 tests and failed its video baseline check:
maximum channel difference 221 at frame 4. The untouched base reproduced the
same failure in this container. No baseline was replaced.

The full-suite attempt was interrupted near its 20-minute execution limit:
2,203 passed, 142 skipped and 19 failed, in 1,165.82 seconds. Eleven failures
were TeX/SVG-group issues (three documentation examples and eight numeric-display
cases), reproduced on the untouched base. The other eight involved the
pre-existing area-light auxiliary-table reshape in `area_light_quads.py`, which
still expects a 13-value row. All eight failing area-light tests reproduced
the same shape failures on the untouched base (37.90 seconds). Thus all
19 observed failures reproduce on the base, but this is not a completed
full-suite validation.

The first Mac attempt exposed a zero-byte optional-buffer binding in the new
final-record kernel: 77 passed and three tests failed. The corrected wrapper
reuses an existing output as the compile-time-unused argument instead of passing
an empty Metal buffer. The failed attempt is recorded rather than counted as
passing validation.

### Short graphics signal for the persistent fused path

[Run 34727616476](https://github.com/algorithmicsimplicity/algan/actions/runs/34727616476)
measured the persistent fused path on one PREVIEW graphics frame, using three
mirrored warm pairs. This is deliberately a short diagnostic rather than a
replacement for the branch's longer graphics workload profiles:

| Arm | Warm readings (s) | Median (s) | Median vs reference |
| --- | --- | ---: | ---: |
| Reference | 4.005, 3.727, 3.151, 2.932, 2.853, 3.108 | 3.129 | -- |
| Persistent fused stream | 4.808, 4.393, 3.218, 3.250, 3.312, 3.460 | 3.386 | +8.2% |

The decoded outputs themselves were identical: maximum channel difference 0
and zero differing channels. The harness exited non-zero only because ffmpeg
reported two decoded frames for this authored one-frame MP4, while its parity
assertion also requires the decoded frame count to equal the requested authored
frame count. The pixel comparison passed; the run is therefore useful as a
timing signal but should not be called a passing acceptance run.

This signal is enough to keep `sheet_fused_stream` **off by default** despite
its 5.3% complete-video explainer win. The optimization is workload-sensitive
on this Metal runner. `sheet_device_runs` and `sheet_fragment_run_sort` also
remain opt-in because their independent full-explainer measurements regressed.
No CUDA default is inferred without a CUDA measurement. The complete
machine-readable summary and raw transcript are preserved on the linked workflow run.
