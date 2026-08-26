---
name: t4-perf-next-steps
description: "What is left after the T4 optimization round on perf/t4-nn-scene-throughput, ranked, with the traps that cost time"
metadata: 
  node_type: memory
  type: project
  originSessionId: 12bc6cf5-d2f5-42dd-93f0-f55a01321657
  modified: 2026-08-25T07:16:50.441Z
---

> **SUPERSEDED (2026-08-26).** The ranking below was built on a profiler bug:
> a stage's `excl` column included every Taichi kernel that stage launched, so
> item 1's "render thread is GPU-bound, `wavefront_shade` 32%, torch passes of
> the sheet chain ~35%" mixed real kernel time with a phantom. In particular
> `wavefront_loop`'s 13.2 s of "unattributed host work" was 12.5 s of two
> kernels already named in the same table. Fixed at `9f3fdb90`
> (`TIMERS.charge_kernel_to_parent`).
>
> Use `DESIGN_T4_optimization.md` and `[[t4-round2-findings]]` instead. What is
> still worth reading here is the **"How to apply / traps"** section at the
> bottom, which is unaffected.

State on 2026-08-25 (branch `perf/t4-nn-scene-throughput`, see [[t4-perf-branch]]):
warm PREVIEW 36.5 s -> 7.7 s, UHD 50.0 s -> 30.9 s; peak VRAM PREVIEW 6.2 GB, UHD
8.4 -> 6.5 GB. The plan of record's "The T4 round" section in
`DESIGN_optimization_targets.md` has the status table.

**Left, in order:**
1. UHD render thread is GPU-bound: `wavefront_shade` 32% of GPU time (the first
   bounce iteration, ~420k rays/frame at ~260 ns/ray incl. inline shadow rays),
   `raster_shadow_trace` 16%, `wavefront_traverse_events` 13.5%; torch passes of the
   sheet chain ~35% (Ox was kernelising these -- `scratch_perf/ox/brief_sheet_chain.md`,
   report expected at `scratch_perf/ox/REPORT_sheet_chain.md`; if it landed, verify
   with `benchmarks/_sheet_kernel_check.py` and a lossless render A/B before trusting).
2. PREVIEW: the arena preflight is serial on the render thread (~0.6 s/batch: PN dice
   0.22, merge 0.33 incl. refit-BVH 0.17) and the first batch's preparation (~1.2 s)
   precedes any rendering. Shrinking the first window would overlap it but moves
   pixels (windows decide chord counts); decide deliberately.
3. The `NeuralNetMLP` idle updater costs ~7.5k `AttributeTimeline.get` calls per batch
   (80 `move_between_points`); packing synapses would remove it. Prep is overlapped
   now, so this only matters for the first batch.
4. `ALGAN_GPU_MAX_REG=64` did nothing; fast launch is already on.

**How to apply / traps:**
- Never time or pixel-compare while another process uses the GPU (Ox's verification
  renders, a stuck pytest holding 4 GB). Check `nvidia-smi` first; `SIGSTOP` Ox.
- Windows move pixels; compare same-window renders. Pin
  `SETTINGS.computing.available_memory_override` for byte-reproducibility.
- The fast suite's pixel test fails on this machine on master too (CUDA baseline from
  another GPU); the branch's fast frames are byte-identical to master's.
- The `--fast` suite's second pytest in a chain once hung at interpreter exit while
  holding VRAM; run test batches in the foreground when the GPU matters.
- NVENC refuses frames under ~145x49 and odd-sided 4:2:0 frames and ffmpeg's refusal
  never reaches Python (an empty mp4 results); `select_video_encoder` keeps outputs
  under 256x128 or odd-sided on libx264. Full `tests/unit_tests`: 2058 passed after
  this (the 3 failures were exactly the 32x32 SMOKE_TEST renders).
