# REPORT: cut the host-torch passes of the sheet compaction at 4K, byte-identically

Brief: `scratch_perf/ox/brief_sheet_chain.md`. Branch `perf/t4-nn-scene-throughput`
(at b4b8e60; none of the files this task touched moved under me while another
session committed). All GPU measurements waited for `scratch_perf/gpu_gate.txt`,
ran with `ALGAN_USE_DAEMON=0`, one process per arm where a gate could be baked
in (these three toggles are host-side per-call gates, so the in-process A/Bs
are honest; renders were run as separate processes anyway). Nothing is
committed.

## 1. Measure first

### `benchmarks/_sheet_compact_breakdown.py` (standalone pass replay, mean of 5)

```
=== compact_sheets pass costs, n=3,661,824 sheets=3,290,404, mean of 5 ===
     29.8 ms   31.7%  bit-lane rank scan (8 x cumsum/gather over [n])
     18.0 ms   19.1%  bit-lane union + fusion detector (8 x scatter_add, x2/frame)
     17.5 ms   18.6%  _lexsort (3 stable argsorts)
     10.3 ms   11.0%  6 x index_select on one permutation
      8.7 ms    9.2%  torch.unique(sorted, return_inverse)
      6.6 ms    7.0%  bit-lane popcount over [sheets] (x2 per frame)
      3.2 ms    3.4%  segmented reductions (area/first/max/count)
     94.1 ms  100.0%  (sum of the passes benchmarked)
```

The top two rows are already kernels (`SHEET_RANK_KERNEL` / `SHEET_MASK_KERNEL`);
this script models only a subset of the compaction's passes, so I built two
probes that time the ACTUAL statement groups on a captured real frame
(`scratch_perf/ox/probe_sheet_chain_breakdown.py` and `probe_prepare_blocks.py`,
inputs captured from one real nn-scene UHD frame by monkeypatching
`compact_sheets`; n=3,128,845 fragments, 755,877 covered pixels).

### Per-block cost of `compact_sheets`, real stream, mean of 3

```
captured: n=3,128,845 covered=755,877 shade_split=True positioned_depth=True sample_depth=True band_rule=prim
solid-shell block triggers: True

=== compact_sheets block costs on the real stream (n=3,128,845, nb=1,441,601), mean of 3 ===
     24.0 ms   43.4%  S: solid-shell ceiling block
     14.5 ms   26.3%  G: sample-depth lane loop (8 lanes)
      4.6 ms    8.4%  A: per-band order stats (1,441,601 bands, positioned_depth)
      4.2 ms    7.7%  E: final argsort + sibling weights + CSR
      2.5 ms    4.6%  D: dominant fragment + nfrag
      1.9 ms    3.5%  H: group diagnostics
      1.5 ms    2.8%  C: shade-split band composite (_band_composite)
      1.2 ms    2.1%  B: new_group/band_start/band_id construction
      0.7 ms    1.3%  R: main _band_reduce (kernel, shipped)
```

### Per-block cost of `prepare_sparse_raster_coverage`'s host blocks

```
captured: n=3,128,845 covered=755,877 mean frags/pixel 4.14
=== prepare_sparse_raster_coverage block costs, mean of 3 ===
     15.3 ms   58.8%  T: opaque-prefix truncation (first_opaque + keep)
     10.7 ms   41.2%  M: one-mesh reduction block (lo/hi + front/back f64 + apply)
```

### What the ranking did to the brief's candidate list

* Candidate 1 (segmented `new_group`/`band_start`/`band_id` construction +
  per-band reductions) measured **small** on the real stream -- B = 1.2 ms,
  A = 4.6 ms, D = 2.5 ms -- because the shipped lane-loop kernels already
  removed what made those scans expensive and because the real post-unique
  band count here is 1.44 M, not the 3.29 M the synthetic breakdown models.
  The one member of that family that IS big is **G**, the SHEET_SAMPLE_DEPTH
  eight-lane amin loop (14.5 ms) -- literally "per-band reductions", just
  per (band, lane).
* Candidate 2 (**M**, one-mesh block) = 10.7 ms. Candidate 3 (**T**, opaque
  truncation) = 15.3 ms.
* S (solid-shell ceiling, 24 ms) was left alone: half of it is the lexsort
  the brief says to leave (cuB), and the rest is entangled with that sort's
  segment construction.

So I kernelised the three non-sort passes above ~10 ms: **T, G, M** --
covering every named candidate's viable part.

## 2. What was implemented

All in `sheet_compact_taichi.py`, each behind its own toggle following the
`SHEET_RANK_KERNEL` pattern (module global + env default + setter, registered
in `SETTINGS.raytracing.experimental`, declared in `algan/environment.py`;
all default ON):

| kernel(s) | toggle | replaces | exactness argument |
| --- | --- | --- | --- |
| `opaque_prefix_keep` | `RASTER_OPAQUE_TRUNC_KERNEL` | the truncation chain: `[n]` arange + whole-stream `repeat_interleave` segment map + nonzero/index_select/scatter_reduce amin + two full-length compares -> one thread per covered pixel walking its CSR run twice (find first opaque, write flags) | integer flag comparisons over identical ranges -- identical by construction |
| `sheet_lane_first_owner` | `SHEET_SAMPLE_DEPTH_KERNEL` | the 8-lane loop in `sheets._lane_first_owners` (was inline in `compact_sheets`): per lane a full-length masked `where` + amin `scatter_reduce_` -> one pass doing all lanes' atomic mins into one pre-filled table | integer amin per disjoint (band, lane) slot -- order-independent |
| `one_mesh_pixel_reduce` + `one_mesh_pixel_apply` | `SHEET_ONE_MESH_KERNEL` | the emission's one-mesh block behind `raster_pipeline._one_mesh_pixel_caps`: id-spread amin/amax + two f64 facing-split `scatter_add_`s routed through a whole-stream segment map, plus the mask/cap fold -> thread per pixel keeping all four aggregates in registers, then an in-place fold | id spread integer min/max (exact); f64 coverage sums keep the §6.6.4 accumulate-f64/round-f32 contract -- bitwise vs torch **by measurement** (below); serial per pixel so now order-reproducible run to run besides |

Two details worth knowing:

* `one_mesh_pixel_reduce`'s i32 fills differ from torch's (i32 max vs `1<<40`)
  only where a pixel would have zero fragments, which cannot happen (every
  covered pixel has >= 1), so observed values are identical -- argued in the
  kernel docstring.
* Taichi infers `fr = 0.0` as f32 and silently narrows the accumulation (it
  warns "atomic add may lose precision"). The registers are declared
  `ti.f64(0.0)` explicitly; the check harness would not have caught a silent
  narrowing at small shapes, so this was fixed on the warning, not on a test.

Left alone, deliberately: all sorts/uniques/cumsums (cuB, T5's own advice);
the solid-shell ceiling block (sort-dominated); `_shade_class` /
`_prim_split_after`'s gathers (~19 + ~13 ms/frame but gather-shaped, next in
line if wanted); per-band order stats / dominant fragment / group diagnostics
(< 5 ms each on the real stream); the six-array gather (`RASTER_FUSED_GATHER`,
parked off upstream).

## 3. Verification (all required steps)

### Unit checks -- `benchmarks/_sheet_kernel_check.py` extended

New sections compare each helper both arms AND against the verbatim torch
statements, at 4K shapes plus edge cases:

```
_opaque_prefix_keep vs the torch first_opaque/keep chain it replaced
  ok    4K shapes, ~20% opaque (kept 2230680 of 3661824)
  ok    no opaque at all (kept 3661824 of 3661824)
  ok    every fragment opaque (kept 755877 of 3661824)
  ok    first fragment of each pixel opaque (kept 755877 of 3661824)
  ok    last fragment of each pixel opaque (kept 3661824 of 3661824)
  ok    single pixel holding the whole stream (kept 5 of 3661824)
  ok    n == 4, two pixels (kept 4 of 4)

_one_mesh_pixel_caps vs the torch one-mesh block it replaced
  ok    4K shapes, mixed surfaces/facings/opacity (flagged fragments: 1614201)
  ok    one surface everywhere (flagged fragments: 3128845)
  ok    circuits only (no flags, sentinel caps) (flagged fragments: 3128845)
  ok    every pixel a single fragment (flagged fragments: 3128845)
  ok    all fragments back-facing (flagged fragments: 3128845)
  (+ each case repeats the kernel arm 4x and requires bitwise-stable caps)

_lane_first_owners vs the eight-lane amin loop it replaced
  ok    4K shapes, random bands/masks (inf entries: 3893811)
  ok    half the band table unused (inf entries: 6425010)          <- empty bands
  ok    every fragment its own sheet (inf entries: 12512880)       <- single-fragment bands
  ok    one sheet, every lane claimed by everything (inf entries: 0)
  ok    donors only (no lane owned) (inf entries: 8)
```

(The "flagged fragments" labels include bits the random INPUT masks already
carried -- e.g. "circuits only" flags nothing new; the assertion compared is
bitwise equality between arms.) The pre-existing sections (gather, band
reduce, conflict rank) still pass unchanged. End-to-end, four rendered frames
hashed with ALL SIX toggles off vs on:

```
rendered frames, all six toggles ON vs all six OFF
  ok    frame 0  822e3a311bc34185
  ok    frame 1  2409707818368ef0
  ok    frame 2  72c3ad5478727d8c
  ok    frame 3  9d117ca7d3a389c5
FAILURES: none -- bit-identical
```

Float-contract statement: the two float reductions (`_band_reduce`'s area sum
-- pre-existing -- and `one_mesh_pixel_reduce`'s front/back sums) are verified
BITWISE equal to their torch arms at 4K shapes and bitwise-stable across 4-6
kernel runs, in the harness above. Integer passes are exact by construction
and additionally checked elementwise.

### Alternating in-process A/B -- `benchmarks/_sheet_kernel_ab.py 3840 2160 3`

My three toggles added to both arms:

```
=== 3840x2160, 3 alternating rounds ===
  torch   frame  1.384s   compact_sheets  0.297s   gather  0.002s
  kernel  frame  1.139s   compact_sheets  0.210s   gather  0.002s
  speedup 1.215x on the frame  (245 ms saved)
```

### Same-input per-pass A/B (real captured stream; `probe_after_timings.py`)

```
n=3,128,845 covered=755,877
      2.9 ms  _opaque_prefix_keep (torch )     0.4 ms  kernel
      8.5 ms  _one_mesh_pixel_caps (torch )    2.1 ms  kernel
     42.3 ms  _lane_first_owners (torch )      7.7 ms  kernel
```

(The lane-owner pair is on synthetic uniform bands; on the stream's own
skewed bands the loop measured 14.5 ms before -- either way most of the pass
is gone.)

### Paired stage timing, same scene/script/box (`_sheet_stage_timing.py 3840 2160`)

Before (torch arms):
```
frame      wall   prepare  emission sort/oth  compact   |sorts  |elemwise
    1    2.856s    0.373s    0.023s   0.021s   0.280s    0.067s     0.213s
    2    2.857s    0.368s    0.020s   0.030s   0.271s    0.071s     0.200s
```
After:
```
    1    1.435s    0.266s    0.014s   0.020s   0.212s    0.053s     0.159s
    2    1.077s    0.268s    0.014s   0.020s   0.215s    0.054s     0.161s
```

`compact_sheets` 275 -> 214 ms; `prepare` total 370 -> 267 ms while its
measured children (emission kernels, torch sorts, `compact_sheets`) stayed at
the same or smaller values -- i.e. roughly 100 ms of per-frame host work
removed, consistent with the three same-input pass pairs above plus
run-to-run variance. Wall columns swing with the shared GPU's thermals;
trust the alternating A/B for frame-level claims.

### Lossless render A/B (nn scene at HD)

`scratch_perf/ox/render_once_lossless.py` (copy of `render_once.py` at HD,
libx264rgb qp 0), torch arm = my three toggles off via env, kernel arm =
defaults:

```
frames compared: 15
worst channel diff: 0 (frame -1)
pixels over tol 2: worst frame 0 of 2073600 (0.000%, frame -1); mean 0.0/frame; 0 of 15 frames affected

ec968b0a3992992d79fe669c0d224a4b  sheetchain_torch_arm.mp4
ec968b0a3992992d79fe669c0d224a4b  sheetchain_kernel_arm.mp4
```

Zero differing pixels; the two lossless files are md5-identical.

### Test suite

`uv run -m pytest -q --fast`:

```
fast suite: 39s of its 75s budget (52%)
1 failed, 275 passed, 1926 deselected in 38.64s
FAILED tests/fast/test_fast_render.py::test_the_fast_scene_renders_and_matches_its_baseline
```

Pre-existing, as the brief said: verified by checking out the branch tip in a
clean worktree (/tmp/opencode/base_check, no changes from this task) and
running the same test -- it fails there too, and the produced error videos are
byte-identical to my tree's (md5 449e01af5e4768b2ed4aa71bcfcfe978 on both),
i.e. the same deviation, not mine.

Targeted suites around the changed code:

```
uv run -m pytest -q tests/unit_tests/test_sheet_compaction.py \
    tests/unit_tests/test_environment.py tests/unit_tests/test_taichi_runtime_config.py \
    tests/unit_tests/test_default_material.py
67 passed in 17.38s
```

### Lint

```
uv run ruff check --no-fix <touched files>
  -> only algan/rendering/raytracing/settings.py:2739 I001 remains, which is
     pre-existing on HEAD (reproduced in the clean worktree; inside
     _build_core_shader_ids, added by another session's commit)
uv run ruff format --check <touched files>   -> all formatted
```

(`sheet_compact_taichi.py` is linted, never formatted, per CLAUDE.md.)
Everything I created under `scratch_perf/ox/` also passes both.

## 4. Everything I did NOT verify

* CUDA-only box: nothing was verified on CPU beyond that the unit tests
  (which run CPU tensors through these helpers in CI) pass here on CUDA;
  CI will exercise the CPU path.
* `tests/full_renders` (the six dense pixel-compared scenes) was NOT run --
  the brief did not require it and each is expensive; the fast-suite render,
  the four hashed check frames and the lossless HD nn render are the render
  evidence. Given the change is proven byte-identical on real streams, a
  full-render baseline move is not expected, but it is unverified.
* The Monte Carlo path (SPP > 1) never enters the sparse route; nothing about
  it was tested.
* Wall-clock frame numbers on this shared T4 swing with the other tenant's
  thermal state (observed 1.08-2.86 s wall for identical work); the speedups
  quoted are alternating medians or same-input pairs, not cross-process walls.
* Peak-memory effect: the kernels delete several [n] intermediates (segment
  maps, arange, f64 copies) and add one [nb*8] i32 table; I did not measure
  the net peak-allocation delta.
* I did not re-profile end-to-end with `profile_scene` to produce a fresh
  RUN-style report table; the stage/pair numbers above stand in for it.
* The solid-shell ceiling block (24 ms, biggest single compact-side item) is
  untouched -- its sort core is off-limits per the brief, and I did not probe
  whether its post-sort half alone could pay for a kernel.
