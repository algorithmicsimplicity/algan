---
name: t4-perf-branch
description: "Ongoing T4/Colab optimization of the nn performance scenes on branch perf/t4-nn-scene-throughput; what was found, what shipped, what is next"
metadata: 
  node_type: memory
  type: project
  originSessionId: 12bc6cf5-d2f5-42dd-93f0-f55a01321657
  modified: 2026-08-25T05:12:35.863Z
---

Work started 2026-08-25 on a Google Colab box (Tesla T4, 2 vCPUs, 12 GB RAM) on
branch `perf/t4-nn-scene-throughput` (pushed to GitHub). Targets:
`benchmarks/performance/nn_scene_PREVIEW.py` (50 frames) and `nn_scene_UHD.py`
(30 frames). Baseline reports committed under
`benchmarks/performance/reports/t4_baseline/`.

**Why:** the user asked for maximum warm steady-state throughput/memory on the T4;
the optimization plan of record (`DESIGN_optimization_targets.md`) was written on a
GTX 1050 box, so its rankings were treated as hints only.

**Findings:** the animated `color_texture` (7.87M channels) dominated prep -- the
timeline materialized/lerped/copied it per frame on the CPU (~150 ms/frame, 83% of
prep) and the ImageMob's grid child carried a dead second texture row. Batches were 3
frames because the 300 MB CPU budget was charged 63 MB/frame for the texture. At UHD
the CPU x264 `-preset slower` encoder drain was 29% of the run.

**Shipped (byte-identical on PREVIEW):** wide attributes (>=65536 channels)
materialize their frame window and edit-log gather on the render device
(`AttributeTimeline.materialize_device`, `ALGAN_WIDE_ATTR_RENDER_DEVICE`); texture
writes are non-recursive; `TrianglePrimitive` no longer relocates texture maps;
per-device batch memory budgets (`_get_render_device_memory_used_per_timestep`,
`_render_device_prep_budget`); constant material params broadcast instead of
gathered. PREVIEW 36.5 s -> 17.05 s.

**How to apply:** measure with `profile_scene` reports (read RUN 2), diff videos with
`benchmarks/_video_diff.py` against `scratch_perf/baseline_videos/`. Ox Alpha
(`opencode run --auto --variant max --model opencode/x-preview-f-free`) is flaky
("Endpoint is unavailable") -- drive it via `scratch_perf/ox/run_ox.sh` and never
`pkill -f` a pattern that appears in your own command line. Ox built the NVENC encoder selection
(`algan/utils/video_encoding.py`, committed; `ALGAN_VIDEO_ENCODER=auto|software|nvenc`).
When comparing videos against x264 baselines, pin `ALGAN_VIDEO_ENCODER=software`. See [[t4-perf-next-steps]].

**Open (2026-08-25):** UHD output is not run-to-run deterministic on the branch
(worst 74 channel values at frame 22 in every pair: profiled, unprofiled, and with
ALGAN_PREFETCH_BATCHES=0), while PREVIEW is byte-identical. Differences sit on every
edge (glyphs, sphere silhouettes) and as speckle over the textured quad -- a
frame-wide sub-pixel/coverage effect, not one object. Being bisected with
`scratch_perf/determinism.py` (arms: CPU texture + big batches; 3-render sequences on
the branch and on a pristine `master` worktree at /content/algan_master).
Results: master itself differs on its FIRST render only (67 levels, 3/30 frames;
renders 1 and 2 identical); the branch alternates X,Y,X between renders (78 levels,
9/30 frames). Not the prefetch worker, not the GPU texture path (both ruled out by
arms). `tests/fast`'s pixel test fails on this T4 on master too (CUDA baseline is
from the user's other GPU) and the branch's fast-scene output is byte-identical to
master's, so that failure is not the branch's.
Later: two renders with DIFFERENT chunk plans were byte-identical, and x264 `slower`
re-encodes identical frames identically -- so the earlier run-to-run diffs were most
likely environmental: Ox's own GPU verification renders overlapped those runs, changing
free VRAM at job start (arena size -> tile sizes). Confirmed: with the GPU quiet, auto/pinned tile sizes, different chunk plans and
different arena sizes all render byte-identically (`scratch_perf/tiles_chain.log`).
Rule: never run determinism/pixel checks while another process is using the GPU.

**Render-thread findings (2026-08-25, later):** at PREVIEW the split-sum glossy
prefilter (`glossy_reflection_mode()==3`, the default) clamped the sparse tile loop to
one frame, so every frame paid a full bounce loop: PREVIEW 17 s -> 9 s with the
prefilter off. Fix in progress: one bounce loop per tile, then per-frame-part
scatter/composite/finish (`gloss_scatter` gained a `row_base` arg). `ALGAN_GPU_MAX_REG=64`
changed nothing at UHD. UHD with a fast encoder: 32 s (baseline 50 s); x264 `slower` on
this box was the 18 s tail. UHD output differs from the 3-frame-batch baseline
(edges/silhouettes, ~5% px) -- suspected batch-wide tessellation/chord decisions;
being checked by rendering the branch with `max_animation_batch_size=3`.

**Determinism root-cause work:** merged-scene arrays are bit-identical between the GPU
and CPU texture paths (hash probe), batch windows are the dominant *legitimate*
mover (chord counts are batch-wide maxima -- Ox audit `scratch_perf/ox/REPORT_batchwide_audit.md`),
and the merge headroom / render-device budget are now total-memory based so windows
are reproducible. Taichi kernels DO see torch's pending default-stream writes (no
stream race; `scratch_perf/probe_stream_race.py`, mind the float32 atomic-add
saturation trap at 2^24). Remaining suspect for the wide-attr-only, process-dependent
pixel drift: an arena read-before-write; `ALGAN_ARENA_POISON=<byte>` was added to
`ManualMemory` to test it.
Poison test (ALGAN_ARENA_POISON 0/255/none): byte-identical -> no uninitialised arena
reads. Conclusion: UHD pixels move only with the batch WINDOW (chord counts are
batch-wide maxima); windows moved with live free VRAM (preflight headroom, fetch cap),
now total-memory based. `available_memory_override` remains the knob for full
byte-reproducibility (arena size is still 40% of free VRAM at job start).
Committed 3rd round (gloss per-tile loop, total-memory headroom/budget, ALGAN_ARENA_POISON,
profiler hooks, DESIGN T4 section). Warm: PREVIEW 7.66 s, UHD 30.9 s. UHD GPU time
(torch.profiler, 6 frames): wavefront_shade 32%, raster_shadow_trace 16%,
traverse_events 13.5%, torch sorts/copies/cats ~35%. Ox is on the sheet-chain torch
passes (`scratch_perf/ox/brief_sheet_chain.md`, GPU gated by scratch_perf/gpu_gate.txt).
