# Area-light quads A/B, Kaggle T4, 2026-09-05

Roadmap §6a-ter landed at `81f63c4`: the path tracer sees a `RectAreaLight`
as two emissive triangles (`area_light_quads.py`) instead of its `K = k*k`
packed cell rows. `ALGAN_PT_AREA_LIGHT_QUADS=0` is the rows arm, byte-identical
to the branch before it. Session `pt-arealight-1` (`pt-arealight-1.txt`, at
`df25526`): `benchmarks/performance/pt_area_lights.py --scene area_lights`,
the `lit` solids under four `samples = 16` area lights — 64 packed rows
against 8 triangles — five frames, 16 spp ceiling with adaptive sampling
(both arms took 5.76 mean spp at 720p, 5.74 at 1080p), 4 bounces, denoiser
on, warm RUN 2, device `cuda`.

## Cost

| arm | end-to-end | `pt_shade` | traverse | host | peak alloc |
| --- | --- | --- | --- | --- | --- |
| 1280x720, quads | **1.931 s** | 272 ms | 142 ms | 1.159 s | 6118 MB |
| 1280x720, rows | 2.034 s | 255 ms | 111 ms | 1.324 s | 6018 MB |
| 1920x1080, quads | **2.752 s** | 666 ms | 323 ms | 1.236 s | 6300 MB |
| 1920x1080, rows | 2.802 s | 615 ms | 246 ms | 1.424 s | 6201 MB |

End to end the quads arm is **5% faster at 720p and 2% at 1080p**, and the
two halves of that pull in opposite directions:

* **Device time is up 8%** (720p: +17 ms shade, +31 ms traverse; 1080p:
  +51 ms shade, +77 ms traverse). The traverse half is the price of the
  quads being packed non-opaque: a batch holding one turns
  `all_visible_opaque` off, which takes `pt_opaque_closest` (nearest-hit
  traversal) and the any-hit shadow query off for the whole batch, so every
  secondary ray gathers the four-deep k-buffer again. The shade half is the
  two-strategy MIS at emitter hits plus the falloff multiplier. The fix for
  the traverse half is the one §6a-ter names as the right end state: the
  quads entering the merge with a camera-invisible leaf bit the traversal
  tests, which keeps the batch provably opaque.
* **Host time is down 165–190 ms**: the next-event table drops from 64 light
  rows to 8 triangles, so the per-chunk setup (the light tree over 8 leaves
  instead of 64, the CDF, the uploads) is a fraction of what it was. The
  per-batch triangle-tree rebuild the quads need is in this column too and
  is evidently smaller than what the rows cost.
* Peak allocation is +100 MB, the second copy of the triangle tables that
  the widened merge keeps in the arena's persistent end for the batch.

## Variance on the T4

`benchmarks/_pt_area_light_quad_variance.py --resolution 128 --trials 4`
(one `samples = 16` area light over a Lambert floor with a smooth metal
sphere, adaptive sampling off, MSE against a 1024-spp reference):

```
reference 1024 spp at 128x128: mean abs difference between the two arms 0.644 counts
 rows arm, 16 spp: MSE 252.78 (per trial: 264.18, 254.89, 251.65, 240.39)
quads arm, 16 spp: MSE 138.33 (per trial: 139.05, 141.11, 136.08, 137.09)
quad arm is 1.83x better in MSE
```

The CPU figure at 64x64 was 2.09x; the 0.64-count gap between the two
references bounds the bias at well under a channel count, so this is
variance. At equal error that is roughly 1.8x fewer samples for the same
frame, on top of the wall-clock gain above.
