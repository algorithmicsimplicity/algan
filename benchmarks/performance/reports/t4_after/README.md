# T4 round: after

> **Superseded, and its `excl` column is not comparable to a report taken
> today.** These profiles predate `charge_kernel_to_parent`
> (`utils/profiling_utils.py`), so a stage's "exclusive" time still contained
> every Taichi kernel it launched: `wavefront_loop` reads 12.0 s of apparent
> host work here, of which 11.4 s is the two kernels listed by name in the same
> table. They also predate the inline per-bounce stages. See
> `../t4_2026_09/` for a baseline taken with the current profiler.

Warm (RUN 2) profiles of the two nn scenes on the Tesla T4 box after the
branch's changes, both benchmark scripts encoding with `libx264 -preset
ultrafast` (see the scripts for why). Two pairs because the last change of the
round (releasing a wide attribute's window after the primitive build) only
moves memory:

* `*_timing.txt` -- taken with the GPU otherwise idle, before the window
  release: **PREVIEW 7.66 s** (baseline 36.47 s), **UHD 30.88 s** (baseline
  50.00 s, of which 14.3 s was the x264 `slower` drain).
* `*_memory.txt` -- taken after the window release, with Ox Alpha's paused
  render still holding 4.8 GB of the card, so the times are pessimistic
  (8.5 s / 30.4 s) but the peaks are what the branch ships: **PREVIEW 6.2 GB**
  (baseline 6.2 GB), **UHD 6.5 GB** (baseline 8.4 GB).

Rendered output: PREVIEW is byte-identical to the baseline video; UHD differs
through the batch window only (19/11 frames instead of 3), as documented in
`DESIGN_optimization_targets.md` ("The T4 round").
