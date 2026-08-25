# Task: cut the host-torch passes of the sheet compaction at 4K, byte-identically

Read `/content/algan/CLAUDE.md` first (rules on `*_taichi.py` files, `ti.init`,
env-var declarations, `ruff check --no-fix`). Set `ALGAN_USE_DAEMON=0` for every
script. Work on the current branch's tree; do not commit. Another session is using
the GPU intermittently: **before every GPU measurement, run**
`until grep -q "GPU-FREE" /content/algan/scratch_perf/gpu_gate.txt; do sleep 20; done`
(the file appears when the GPU is yours; it is there already if nothing is running).

## Why
Profiling `benchmarks/performance/nn_scene_UHD.py` on this Tesla T4 (report:
`scratch_perf/report_UHD_r2.txt`, RUN 2) shows the sparse route's host chain is
~9 s of a 30 s render: `raster:   - compact_sheets` 5.0 s incl (own 3.0 s,
`sheets lexsort` 0.94 s, `shade class` 0.54 s, `prim split` 0.38 s), `raster: sparse
discovery` own 2.7 s, `window pairs` 1.46 s, `fragment sort` 0.45 s, for 30 frames
(~3.7 M fragments / 3.3 M sheets per frame). The kernels that already replaced the
per-sample-lane reductions (`SHEET_MASK_KERNEL`, `SHEET_RANK_KERNEL`,
`RASTER_FUSED_GATHER`) are the precedent: same work, one pass, bit-identical.

## What to do
1. Measure first, on this box: `uv run python benchmarks/_sheet_compact_breakdown.py`
   and `uv run python benchmarks/_sheet_stage_timing.py 3840 2160` (read their
   docstrings). Report the per-pass table. Identify the largest remaining host-torch
   passes in `algan/rendering/raytracing/sheets.py::compact_sheets` and
   `raster_pipeline.py::prepare_sparse_raster_coverage` that are NOT sorts/unique/cumsum
   (those are cuB-backed; DESIGN_optimization_targets.md T5 says leave them).
2. Implement the biggest one or two as Taichi kernels (a new `*_taichi.py` module or
   `sheet_compact_taichi.py`), each behind a toggle in
   `algan/rendering/raytracing/settings.py` following the `SHEET_RANK_KERNEL` pattern
   (env var declared in `algan/environment.py`, default ON only if bit-identical).
   Candidates you should evaluate, in order: the segmented `new_group`/`band_start`
   /`band_id` construction and the per-band reductions in `compact_sheets`; the
   one-mesh reduction block in `prepare_sparse_raster_coverage` (`scatter_reduce_` amin/amax
   + f64 `scatter_add_` pairs -- keep the f64 accumulate + f32 round contract); the
   opaque-prefix truncation block (`first_opaque` scatter_reduce + `keep`). Integer
   passes must be exactly identical; float passes must reproduce the torch result bit
   for bit (state how you verified).
3. Verification (all required; report outputs verbatim):
   - extend `benchmarks/_sheet_kernel_check.py`'s unit half to cover each new kernel
     against the exact torch expression it replaces at 4K shapes, including edge cases
     (empty bands, single-fragment bands, all-opaque pixel, no opaque at all);
   - `uv run python benchmarks/_sheet_kernel_ab.py 3840 2160 3` (alternating in-process
     A/B, medians) with your toggles added to the arms it alternates;
   - a lossless render A/B of `benchmarks/performance/nn_scene_UHD.py`'s scene at
     `HD` with the toggle off vs on (copy `scratch_perf/render_once.py`, pass
     `ffmpeg_params=["-c:v", "libx264rgb", "-qp", "0"]` or use `_video_diff.py` on
     the two outputs) -- must be 0 differing pixels;
   - `uv run -m pytest -q --fast` (report the timing line; the fast-suite pixel test's
     baseline failure on this machine is pre-existing -- master fails it identically --
     so report it but do not chase it);
   - `uv run ruff check --no-fix` and `uv run ruff format --check` on touched files.
4. Report to `scratch_perf/ox/REPORT_sheet_chain.md`: per-pass table before/after,
   which passes you kernelised and which you left and why, measured speedups, and
   everything you did NOT verify.
