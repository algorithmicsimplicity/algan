# Denoiser precision A/B, Kaggle T4, 2026-09-04

Follow-up to `pt_baseline_1.md`, which found the fp32 torch U-Net to be the
largest device-side item of a path-traced frame. `denoise_precision` now
defaults to half precision with channels-last activations on CUDA; this
session measures what that bought and what it cost (`pt-denoise-1.txt` is
the transcript; branch `claude/algan-path-tracer-design-8uj1vl` at
`163291e`, device verified `cuda` on every arm).

## The filter alone (`benchmarks/_denoise_precision_check.py`)

Same process, same scene, both precisions, device-synchronised timing of
`Denoiser.__call__`, raw frames compared in 8-bit counts:

| resolution | fp32 per frame | fp16 per frame | speed-up | max diff | mean diff | channel samples over 2 |
| --- | --- | --- | --- | --- | --- | --- |
| 1280x720 | 75.2 ms | 44.4 ms | 1.69x | 1 count | 0.0025 | 0 of 5.5 M |
| 1920x1080 | 193.6 ms | 114.0 ms | 1.70x | 1 count | 0.0025 | 0 of 12.4 M |

So the cost is at most one count on a handful of channel samples, and the
default is the half-precision path. A render that must match an fp32
baseline pins `denoise_precision = "fp32"`.

## End to end (`pt_baseline.py`, 5 frames, 16 spp, 4 bounces, warm RUN 2)

| arm | fp32 | fp16 (default) | change |
| --- | --- | --- | --- |
| lit 1280x720 | 1.784 s (denoise 0.385 s) | 1.630 s (denoise 0.224 s) | -8.6% |
| text_2d 1280x720 | 1.165 s (denoise 0.394 s, from `pt_baseline_1`) | 1.007 s (denoise 0.228 s) | -13.6% |

The transport kernels are unchanged (`pt_shade` 248 ms, traverse 153 ms on
the lit arm, as before), which is the point: the switch touches nothing
the sampler or the kernels compute.

1.7x is well short of what the T4's tensor cores can do for a convolution
stack; the remaining time is in the per-tile loop (six 512-pixel tiles at
720p, each a separate forward pass), the NHWC/NCHW conversions at the
boundary, and the upsample/concat glue between convolutions. Batching the
tiles into one forward pass and a `torch.compile` of the U-Net are the next
two things to try, in that order, when the denoiser is again the largest
item on the profile.
