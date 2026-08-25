# Task: cut the sheet route's remaining *host-torch* passes at 4K, byte-identically

Read `D:\algan_wt_sheet\CLAUDE.md` first and obey it — especially the rules on
`*_taichi.py` files (never formatted, keep the `_taichi` suffix, `SIM`/`I002`
are off there for reasons), on never calling `ti.init` yourself, on declaring
every `ALGAN_` env var in `algan/environment.py`, on reading settings live
(`rt_settings.X` at call time), and on `ruff check --no-fix`.

**A `ti.static` gate is baked when the kernel compiles**, so anything a static
gate controls needs **one process per arm** — otherwise arm 2 silently reuses
arm 1's code and reports its numbers as its own.

Work **only** inside `D:\algan_wt_sheet` (a git worktree on branch
`perf/r2-sheet` with its own `.venv`); run Python as `uv run python` from that
directory. Another agent is working in `D:\algan` and another in
`D:\algan_wt_prep` — **never read from, write to, or run anything in those
trees.** Do not commit and do not push.

Set `ALGAN_USE_DAEMON=0` for every script. You share one GPU (NVIDIA GTX 1050,
4 GB) with another agent, so wall-clock GPU timing is noisy: use in-process
alternating A/B with medians over many repetitions and say so, or argue from
bytes moved and launch counts. Never present a single pair of wall-clock numbers
as a result.

## Why this task exists

On a Tesla T4, `benchmarks/performance/nn_scene_UHD.py` (30 frames at
3840x2160, warm steady state, whole run 29.9 s) spends **~5.2 s of the render in
host-side torch passes of the sheet route**:

```
stage                                    calls   own time (s)   note
raster:   - compact_sheets                  30       2.31       excl, essentially all host torch
raster:     - sheets lexsort                90       0.91       3 stable argsorts per call
raster:   - window pairs                    60       1.14
raster:     - sheets shade class            30       0.52
raster:   - fragment sort                   30       0.41
raster:     - sheets prim split             30       0.36
raster:   - fragment gather                 30       0.08
```

For scale, the whole render is 26.0 s and its three dominant Taichi kernels
(`wavefront_shade` 8.8 s, `raster_shadow_trace` 4.5 s,
`wavefront_traverse_events` 3.7 s) are 17 s of it. So this chain is the largest
*non-kernel* cost left, and it is the one a host-side change can actually move.

An independent `torch.profiler` capture of 6 UHD frames on the same T4 agrees,
and names the ops (self CUDA time, 6 frames, 5.28 s total):

```
aten::copy_          213 ms   8261 calls
aten::sort/argsort   211 ms    206 / 148 calls
aten::index_select   175 ms    703 calls
aten::gather         172 ms    712 calls
aten::cat            167 ms    820 calls
aten::stack          154 ms    324 calls
aten::to/_to_copy    152 ms   5535 / 2374 calls
aten::fill_          116 ms   2945 calls
aten::index           84 ms   2115 calls
aten::zero_           66 ms   1197 calls
aten::_unique2        59 ms     72 calls
```

Note the **call counts**: 8261 `copy_` and 820 `cat` for six frames is a lot of
small launches, and the `cat`/`stack`/`copy_` family together is bigger than the
sorts.

## What has already been done — read this first, do not redo it

A previous round kernelised three passes in this chain. Its report is in this
tree at `scratch_perf/ox/REPORT_sheet_chain.md`. **Read it before doing
anything.** It covers the opaque-prefix truncation (`T`), the sample-depth lane
loop (`G`) and the one-mesh reduction (`M`), and it left the largest remaining
block in `compact_sheets` — **`S`, the solid-shell ceiling block, measured at
24 ms/frame** — on the grounds that "half of it is the lexsort the brief says to
leave (cuB), and the rest is entangled with that sort's segment construction".
That reasoning is worth revisiting now that `S` is the top of the list.

Also **out of scope, and here is why**: I originally aimed this task at
`shade_sparse_raster_coverage`'s event-compaction block (the `nonzero` + six
`index_select`s). That was a misreading of the profile: the stage timers do
**not** subtract a kernel's time from the enclosing stage's exclusive column, so
`raster: sparse resolve`'s 4.86 s "own" time is really `raster_shadow_trace`
(4.47 s) plus `sheet_resolve_shade` (0.30 s) plus about **0.09 s** of host work.
Do not spend time there. (Mentioned so you can apply the same correction if you
read any other stage table: a stage's `excl` column includes every Taichi kernel
it launched.)

## What to do

### Part 1 — measure, on the real stream

Copy the previous round's technique: monkeypatch the function, capture one real
frame's inputs from the nn scene, and time the statement groups individually
with the inputs replayed. Its probes are in this tree
(`scratch_perf/ox/probe_sheet_chain_breakdown.py`) as a worked example. This card
has 4 GB, so capture at the largest frame size that fits and report the fragment
count `n`, the covered-pixel count, and the band count, so the numbers can be
scaled to 4K.

Produce a per-block table for **all** of these, not just `compact_sheets`:

- `compact_sheets` (`algan/rendering/raytracing/sheets.py`) — re-measure every
  block including `S`, since the shipped kernels have moved the ranking since
  the last report;
- `_window_pairs`, `_exact_fragment_order`, `_gather_fragment_arrays`
  (`raster_pipeline.py`) — `window pairs` alone is 1.14 s at UHD and has never
  been looked at;
- `_shade_class` and `_prim_split_after` (`sheets.py`).

### Part 2 — cut the top two or three

Rules:

- **Sorts stay.** `DESIGN_optimization_targets.md` T5 says leave the
  cuB-backed sorts/unique/cumsum alone, and the previous round agreed. What is
  fair game is the *segment construction and gathers around* them — including
  replacing "sort then gather six arrays" with "sort once, gather once in a
  fused kernel".
- The `cat`/`stack`/`copy_` family is the biggest op group in the torch profile
  and is not a sort. Chasing where 8261 `copy_` calls per six frames come from
  is a legitimate line of attack; so is removing a `cat` that only exists to
  build an argsort key that a kernel could compute in place.
- Integer passes must be **exactly** identical. Float passes must reproduce the
  torch result **bit for bit** — state how you verified each one. Note that a
  float reduction on CUDA is not order-reproducible: where the existing code
  accumulates in f64 and rounds to f32, keep that contract exactly.
- Every change goes behind a toggle in
  `algan/rendering/raytracing/settings.py` following the `SHEET_RANK_KERNEL` /
  `SHEET_MASK_KERNEL` pattern (env var declared in `algan/environment.py`),
  defaulting ON only once bit-identity is proved.

## Verification (all required; quote the actual output)

- **Lossless render A/B, toggle off vs on: 0 differing pixels.** Use
  `scratch_perf/render_once.py` as the template, render the nn scene at `HD`,
  pass `ffmpeg_params=["-c:v", "libx264rgb", "-qp", "0"]` (an H.264 re-encode
  turns single-channel differences into thousands of differing pixels), compare
  with `benchmarks/_video_diff.py`, and pin
  `SETTINGS.computing.available_memory_override` to the same value in both arms
  so the batch windows match — a window change legitimately moves pixels and
  would mask a regression. One process per arm.
- A **unit-level** check of each new kernel against the exact torch expression it
  replaces, at 4K-scale shapes, with the edge cases spelled out: empty input,
  a single fragment, a single band, all fragments in one band, and a pixel whose
  fragments are all opaque. Extend `benchmarks/_sheet_kernel_check.py`.
- `uv run -m pytest -q tests/unit_tests` — full unit suite.
- `uv run -m pytest -q --fast` — report the timing line. Its pixel-comparison
  test fails on this machine even on unmodified code (the committed CUDA
  baseline came from a different GPU); confirm that against a clean checkout,
  report both, and do not chase it.
- `uv run ruff check --no-fix` and `uv run ruff format --check` on every file you
  touched.

## Report

Write `scratch_perf/r2/ox/REPORT_sheet_host_chain.md`: the Part 1 table with the
stream's `n` / covered / band counts; what you changed and what each change
bought (say whether the numbers are wall-clock — and how you controlled for the
other agent on this GPU — or byte/launch counts); which candidates you rejected
and why; and — explicitly — everything you did **not** verify.
