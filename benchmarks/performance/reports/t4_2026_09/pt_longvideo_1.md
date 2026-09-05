# Per-frame cost on a longer video, Kaggle T4, 2026-09-05

Every earlier path-tracer session rendered five frames, and the host column
("prep, merge, encode") was about a second of each — enough to ask whether
the fallback is bounded by the host rather than the kernels. Session
`pt-longvideo-1` (`pt-longvideo-1.txt`, at `dca3567`): `pt_baseline.py` at
`--fps 30` (30 frames of the same one-second scene) against `--fps 5`, 16 spp
ceiling with adaptive sampling (5.75–5.77 mean spp in every arm), 4 bounces,
denoiser on, warm RUN 2, device `cuda`, one batch per render.

## Numbers

| arm | frames | end-to-end | per frame | `pt_shade` /frame | traverse /frame | denoise /frame | host total | host /frame |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lit 1280x720 | 5 | 1.573 s | 315 ms | 42 ms | 21 ms | 37 ms | 980 ms | 196 ms |
| lit 1280x720 | 30 | 5.785 s | **193 ms** | 43 ms | 21 ms | 38 ms | 2154 ms | 72 ms |
| many_lights (64) 1280x720 | 30 | 6.281 s | **209 ms** | 53 ms | 21 ms | 39 ms | 2348 ms | 78 ms |
| lit 1920x1080 | 30 | 10.319 s | **344 ms** | 108 ms | 51 ms | 72 ms | 2642 ms | 88 ms |

## Reading

* **The fixed host cost amortises.** Host time is ~0.75 s per render plus
  ~47 ms per frame at 720p (63 ms at 1080p): the batch prep, arena
  preflight, the one `gc.collect()` and the encoder drain are paid once,
  and a 30-frame render spends 37% of its wall on the host against 62% for
  five frames. The fallback at 720p runs at **~5 frames per second** on a
  T4 at a 16-spp ceiling, 4 bounces, with the denoiser, and 64 lights cost
  8% more than three.
* **What is left per frame** is roughly 100 ms of device work at 720p
  (shade 43, denoise 38, traverse 21) and ~40 ms of host inside the render
  call (`ray traced render total`'s exclusive time, 1.15 s over 30 frames):
  the next-event setup per window, the adaptive sampler's per-wave
  `.item()` syncs and pixel-list builds, the compaction's host side and 15
  launches per frame. That 40 ms is §0.2-bis's item (the host sync every
  iteration), deferred earlier because it did not dominate a five-frame
  render; at 20% of a long render's per-frame cost it is now the largest
  host item and the natural next §0 candidate, ahead of anything in §8.
* **The denoiser is the second-largest device item** at 720p (38 ms, torch
  U-Net in half precision) and is proportional to pixels, not samples; at
  1080p it is 72 ms of a 344 ms frame.
