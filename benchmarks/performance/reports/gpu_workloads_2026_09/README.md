# GPU workload profiles, 2026-09: where the default renderer's time goes

Two workload scenes, two GPUs, two quality presets, one question: what would
make the default (deterministic, analytic-AA) renderer faster, and by how
much. The measurements are in this directory; the ranked answer is
[`REPORT.md`](REPORT.md).

## The workloads

| scene | stands for | what it exercises |
| --- | --- | --- |
| `benchmarks/performance/explainer_scene.py` | a Manim-style math explainer | ~180 glyph circuits, `Axes` + plots, a `NumberPlane`, a 9-node graph, a 6x6 heat map, one `Sphere` and one `Cube`; staggered spawns, colour tweens, `Indicate`/`Circumscribe`, a glyph-morph `become`; static camera, shadows off |
| `benchmarks/performance/graphics_scene.py` | a Three.js-style 3-D scene | environment map, directional + spot + rect-area lights with shadows, glass, chrome and brushed metal on a reflective slab, PN solids, a textured parametric `Surface`, a 12-prism skyline, an imported textured glTF model; camera turntable, everything moving |

Both take `--quality` and `--seconds`/`--frames` (`_profile_cli.py`), so one
authoring is measured at every preset and clip length. Two profiled runs per
measurement; **RUN 2 (warm) is the reading**, RUN 1 is the cold JIT cost.

## The boxes

| harness | hardware | frames measured |
| --- | --- | --- |
| Kaggle T4 (`scripts/kaggle/`) | Tesla T4 (Turing, 16 GB), 4 vCPUs, CUDA | PREVIEW 60 frames (6 s @ 10 fps), UHD 30 frames (0.5 s @ 60 fps) |
| GitHub Mac runner (`run_on_mac.yaml`) | virtualized M1, real Metal GPU, 3 CPUs, 7 GB | PREVIEW 60 frames, UHD 15 frames |

The Mac's per-launch and per-readback costs include the runner's
virtualization tax (`agent_guidance/gpu_harnesses.md`, and
`../mac_2026_09/FINDINGS.md` §0): its wall times rank launch- and
sync-bound stages higher than a physical Mac would. Its compute numbers are
sound.

## Files

* `mac_mps_explainer.log`, `mac_mps_graphics.log` -- the Mac runs' command
  output (both profiled runs, per stage, per kernel launch). The graphics
  job's UHD pass was cut by the 60-minute timeout before its cold run
  finished; `mac_mps_graphics_uhd_2f_diagnostic.log` is the 2-frame
  follow-up (shadows on and off, PERF logging) that priced it.
* `t4_<step>.log` -- the first Kaggle session's per-step logs, with Taichi's
  per-kernel GPU profiler on; `t4_session2_<step>.log` -- the second
  session, which repeats the four profiles and adds the two `nn_scene`
  references and the shadows-off arm of the graphics UHD scene.
* `t4_budget_<step>.log` -- the third session: the shadow-ray budget A/B
  (`ALGAN_SHADOW_RAY_BUDGET=0` against the default) at UHD and PREVIEW, and
  the check frame comparison.
* `REPORT.md` -- the analysis and the ranked optimization targets.

## The profiler was fixed first

The "(unaccounted ...)" line of `profile_scene` used to go negative on
prefetching renders (down to -107% in `../t4_after/`) because the prefetch
worker's stages were summed into the same ledger as the render thread's and
subtracted from one thread's wall clock. `StageTimers` now keeps a
per-thread ledger; the budget line subtracts only the render thread's
exclusive time, the worker's overlapped prep is printed on its own line, and
the render thread's wait on the prefetched batch is a stage of its own
(`wait for prefetched batch`) -- the one number that says whether scene
preparation is on the critical path. `tests/unit_tests/test_profiler_thread_budget.py`
pins it. Every run's section is also printed the moment the run finishes,
so a reclaimed remote job keeps what it measured.
