# Adaptive sampling A/B, Kaggle T4, 2026-09-05

Roadmap §2 landed at `39b6c08` (branch `claude/algan-path-tracer-design-8uj1vl`):
`samples_per_pixel` is a ceiling, and a pixel stops early only when every
one of its samples was deterministic given the sub-pixel jitter and its
even/odd half-sums agree. This session measures it against the uniform arm
(`--pt-error-target 0`) with `benchmarks/performance/pt_baseline.py`; the
transcript is `pt-adaptive-1.txt`. Five frames, 16 spp, 4 bounces, denoiser
on (half precision), warm RUN 2, device `cuda` on every arm.

## End to end

| arm | uniform | adaptive | mean spp taken | change |
| --- | --- | --- | --- | --- |
| text_2d 1280x720 | 1.011 s | 0.750 s | 4.37 of 16 | **-26%** (1.35x) |
| text_2d 1920x1080 | 1.976 s | 1.301 s | 4.27 of 16 | **-34%** (1.52x) |
| lit 1280x720 | 1.646 s | 1.597 s | 5.76 of 16 | -3% |
| many_lights 1280x720 | — | 1.651 s | 5.76 of 16 | — |

## Where it went, text_2d 1280x720

| item | uniform | adaptive |
| --- | --- | --- |
| `wavefront_traverse_events` | 152 ms | 45 ms |
| `pt_shade` | 150 ms | 44 ms |
| launches of each | 7 | 21 |
| denoiser | 223 ms | 220 ms |
| host: prep, merge, encode | 418 ms | 383 ms |

The transport kernels drop **3.4x** (3.6x at 1080p: 668 ms to 185 ms) for
three times as many launches, because the waves after the floor run over
the shrinking pixel list — the launches are small. What stays is the
denoiser and the host, which is why the end-to-end gain is 1.35-1.5x
rather than 3x, and why the denoiser is the next item on the 2-D arm: it
is 29% of an adaptive text frame at 720p and 44% at 1080p, filtering
pixels that are exact.

## The lit scene

Every lit pixel is stochastic and runs to the ceiling, so what the adaptive
arm saved (5.76 mean spp) is the background — one traversal and out — and
the kernels drop 28% (410 ms to 294 ms) for a 3% end-to-end gain. The two
arms' videos have different digests (`b1ce54abbd243261` against
`4bb792d1cf78159c`) while the CPU harness finds the raw frames
byte-identical. `benchmarks/_pt_adaptive_check.py` on the T4 at 720p
(session `pt-adaptive-check-1`, raw frames, `d15c6a5`) settles it:

| scene | denoiser | max diff | pixels differing | of 1.84 M | interior |
| --- | --- | --- | --- | --- | --- |
| lit | off | 1 count | 2 | | 0 |
| lit | on (fp16) | 1 count | 4540 | | 14 |
| text_2d | off | 152 counts | 4022 | | **0** |
| text_2d | on (fp16) | 115 counts | 72475 | | 9706 |

The lit difference is float summation order on two exact background
pixels landing on a rounding boundary, one count each, and the fp16
network spreads that to 4540 pixels at one count. The text differences
with the denoiser off all sit on an edge of the uniform reference (the
floor-count jittered anti-aliasing); with the denoiser on they had spread
into 9706 interior pixels, which is the filter softening exact content —
the reason the denoiser now passes exact pixels through untouched
(`5651292`).

## What the numbers say about the roadmap

* §2 is landed and pays where predicted: the 2-D arm, at the resolution a
  user renders at, is 1.35-1.5x faster end to end with the same interiors
  and floor-count jittered edges.
* The remaining 2-D cost is the denoiser (fixed per frame) and host prep.
  A denoiser that skips tiles whose every pixel is deterministic would
  take most of the former off the text arm; that is the next §0.2 item.
* On lit scenes adaptive sampling is neutral by design; §6 (the light
  tree) and §5 are where those pixels' cost per sample lives.

## The denoiser pass-through (`5651292`), same box, `pt-denoise-skip-1.txt`

With the stochastic mask the filter passes exact pixels through and skips
tiles whose core holds none:

| arm | before | after | denoiser before | denoiser after |
| --- | --- | --- | --- | --- |
| text_2d 1280x720 | 0.750 s | **0.625 s** | 220 ms | 4 ms |
| text_2d 1920x1080 | 1.301 s | **0.784 s** | 567 ms | 7 ms |
| lit 1280x720 | 1.597 s | 1.588 s | 222 ms | 184 ms |

Against the uniform arm at 16 spp with the whole-frame filter (1.011 s and
1.976 s), the text frame is now 1.6x faster at 720p and 2.5x at 1080p end
to end; against the first baseline of the day (fp32 filter, 1.165 s at
720p) it is 1.9x.

`_pt_adaptive_check.py` at 720p, denoiser on, now compares "filtered
everywhere" (the uniform arm has no mask) with "filtered where estimated":
the text scene differs on 408 k pixels, 265 k of them interior, by up to 138
counts. That is what the whole-frame filter had been doing to exact text —
softening it — and it is the reason contract 4 wanted the pass-through,
not a regression: with the denoiser off the same comparison is 0 interior
pixels. The lit scene's background differs by up to 101 counts along
object silhouettes, where the filter used to bleed lit pixels outward into
exact background; that bleed now stops at the coverage boundary.
