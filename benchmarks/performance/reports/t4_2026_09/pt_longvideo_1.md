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
  launches per frame. The next section attributes it: most of it is the
  profiling harness measuring itself, and the real in-call host cost is
  nearer 25 ms, led by the per-launch arena argument packing.
* **The denoiser is the second-largest device item** at 720p (38 ms, torch
  U-Net in half precision) and is proportional to pixels, not samples; at
  1080p it is 72 ms of a 344 ms frame.

## Attribution of the per-frame host time (`pt-longhost-1.txt`)

Same 30-frame lit arm at 720p, twice: without Taichi's kernel profiler
(`--no-kernel-profiler`) the render is **5.521 s** against 5.785 s with it,
so the profiler's per-launch accounting costs 9 ms per frame. Under cProfile
(`ALGAN_PROFILE_CPROFILE=1`, 6.586 s) the 4.06 s inside `path_trace_render`
splits as:

| where | total | per frame | what it is |
| --- | --- | --- | --- |
| kernel launch calls (`_fast_call`) | 2.29 s | 76 ms | device time: launches block under the profiler |
| `_sync_devices` (7,037 calls) | 0.87 s | 29 ms | **the profiling harness's own** enter/exit syncs around every stage |
| `arena_args_taichi.pack` (900 calls) | 0.40 s | 13 ms | packing the arena offset/shape tables per launch — host, cacheable |
| `_build_nee_tables` | 0.18 s | 6 ms | per-window next-event setup (light tree 3 ms of it) |
| `_pt_active_pixels` | 0.11 s | 3.5 ms | the adaptive sampler's pixel list |
| compactor `select` (450 calls) | 0.25 s | 8 ms | the per-iteration count read-back, sync included |

So of the ~40 ms per frame the stage table showed inside the render call,
~29 ms is the harness measuring itself, and the true in-call host cost is
nearer 25 ms of a ~150 ms production frame. The §0.2-bis sync rewrite is
**not** the item: the compactor's read-back is 8 ms per frame including the
wait for the kernels it is behind. The one cheap win is `pack`: it rebuilds
the arena argument tables for every launch (900 per 30 frames, 0.44 ms
each) although the tensor set is the same for every iteration of a window,
so a per-window cache would return ~13 ms per frame, 8% of the production
frame. The next-event setup could memoise on a static rig for another 6 ms.
The denoiser (35 ms per frame, torch U-Net) is now the largest single item
after `pt_shade`.
