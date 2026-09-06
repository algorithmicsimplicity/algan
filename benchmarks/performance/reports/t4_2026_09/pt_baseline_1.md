# Path tracer baseline, Kaggle T4, 2026-09-04

The first measurement of the `samples_per_pixel > 1` renderer
(`benchmarks/performance/pt_baseline.py`, nine arms in one session, branch
`claude/algan-path-tracer-design-8uj1vl` at `b457281`; the full transcript is
`pt-baseline-1.txt`). It exists to rank `DESIGN_path_tracer_roadmap.md` §0 by
measured cost. Device verified `cuda` / `arch=cuda` on every arm; GPU
otherwise idle; `libx264 -preset ultrafast`; warm RUN 2 throughout.

Every arm renders **5 frames** (one second at 5 fps) at 16 spp, 4 bounces,
denoiser on, so the per-frame numbers below are these divided by five.

## The switches (roadmap §0.2), lit scene, all-opaque batch

`ALGAN_PT_SHADOW_ANYHIT=0 ALGAN_PT_OPAQUE_CLOSEST=0` is the "off" arm. Both
arms of each pair rendered **byte-identical video** (`a9e9a8ff7115fcf9` at
320x180, `7f3e33f640312ad6` at 1280x720), which is the acceptance criterion:
on an opaque batch the two modes answer the same question.

| arm | end-to-end | `pt_shade` | traverse | compact | PT kernels total |
| --- | --- | --- | --- | --- | --- |
| 320x180 on | 0.827 s | 18.2 ms | 10.9 ms | 1.1 ms | 32 ms |
| 320x180 off | 0.815 s | 21.1 ms | 13.6 ms | 1.1 ms | 38 ms |
| 1280x720 on | 1.758 s | 248 ms | 152 ms | 15.5 ms | 453 ms |
| 1280x720 off | 1.836 s | 273 ms | 195 ms | 15.5 ms | 508 ms |

So the two switches take 9-14% off the shade kernel and 20-22% off traversal,
**11% of the path tracer's device time at 720p**, and 4% end to end. Modest,
as predicted: the ordered march already exits at the first opaque hit, so
mode 3 saves the ordered part of the walk and not the walk.

## Where a lit 720p frame goes (5 frames, 1.758 s)

| item | s | % |
| --- | --- | --- |
| host: prep, merge, encode | 0.879 | 50.0 |
| denoiser (torch U-Net, fp32) | 0.383 | 21.8 |
| `pt_shade` (NEE and its shadow rays inside it) | 0.248 | 14.1 |
| `wavefront_traverse_events` | 0.152 | 8.7 |
| `pt_generate` + `pt_reduce` + `compact_ray_slots` | 0.051 | 2.9 |

Three readings from this table, the ones roadmap §0.1 asked for:

1. **The host sync per iteration does not pay to remove** (§0.2-bis). There
   are 5 iterations per wave (4 bounces + 1) and 25 launches of each PT
   kernel per 5 frames; `compact_ray_slots` is 1.3% of wall and its wall
   time equals its device time, so the round trips are hidden. The rewrite
   would matter only at tiny tiles, which the arena budget does not
   produce here (one wave covers a whole frame's pixels at 16 samples in
   flight; peak arena 6.2 GB). Deferred until a profile at a small memory
   budget says otherwise.
2. **The denoiser is the largest device-side item**, at 77 ms per 720p
   frame, and it is a fixed per-frame cost independent of spp. It runs in
   fp32 NCHW, per frame, per 512-pixel tile, through plain `F.conv2d`. That
   is the next §0 target: fp16 on CUDA (OIDN's own GPU path runs half) and
   channels-last, and it needs a tolerance rather than byte identity.
3. **The path tracer's kernels are 0.09 s per 720p frame** at 16 spp and 4
   bounces: ~7.4 M paths per 5 frames through 0.45 s, or ~16 M paths/s
   including next-event estimation and shadows. The deterministic renderer's
   kernels on the same scene are 38 ms per 5 frames (its arm below), so the
   path tracer is ~12x the deterministic renderer in device time and
   **1.7x end to end** (1.758 s against 1.033 s) because host prep and the
   denoiser, not transport, set the wall clock at this size.

## Many lights, 320x180

| arm | end-to-end | `pt_shade` | traverse |
| --- | --- | --- | --- |
| lit, 3 lights | 0.827 s | 18.2 ms | 10.9 ms |
| many_lights, 64 lights | 0.879 s | 20.1 ms | 10.5 ms |

Flat in light count, as the estimator promises: next-event estimation draws
`pt_light_samples` emitters per vertex whatever the rig holds. (The
deterministic renderer would shadow 16 of the 64 and shade all of them
per fragment.) The remaining light-count term, the authored-appearance
branch, is not in this scene — roadmap §6a-bis.

## Text and transparency, 2-D (the §2 case)

| arm | end-to-end | traverse | `pt_shade` | denoiser | host |
| --- | --- | --- | --- | --- | --- |
| 320x180 | 0.383 s | 12.7 ms | 13.9 ms | 43 ms | 301 ms |
| 1280x720 | 1.165 s | 153 ms | 147 ms | 394 ms | 406 ms |

1.4 iterations per wave: the camera-segment peel ends almost every path at
its first batch. At 720p the peel is 0.30 s of kernel time per 5 frames for
content that is zero-variance by construction — that is the adaptive-sampling
win (§2) sized: with a floor of 1-2 samples it would be a 16x reduction of
that 0.30 s, or ~0.28 s of the 1.165 s. The denoiser is 34% of this arm and
does nothing useful to zero-variance pixels.

## Deterministic reference, same lit scene

| arm | end-to-end | all kernels |
| --- | --- | --- |
| 320x180 | 0.840 s | 8.4 ms |
| 1280x720 | 1.033 s | 38 ms |

## Cold compiles

The path tracer's cold RUN 1 was 16-31 s per arm (kernel cache warm across
arms of one session after the first); the deterministic arms' 74-86 s is the
sheet route's many kernels. Both are paid once per cache.
