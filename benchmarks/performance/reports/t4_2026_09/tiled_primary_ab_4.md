# Tiled primary: replacing the per-pixel CSR and sort

The ordering half of the tiled frontend is now **cheaper than the reference
frontend's global sort**, and the residual on the anchor scene is down to
+1.03%. What is left is the binning pass itself, and on that scene it
certifies nothing, so there is nothing left to make cheaper — only a policy
question about when to run it at all.

Follow-up to `tiled_primary_ab_3.md`, which ruled out the candidate count.

**Run:** Kaggle T4, one session (`tiled-ab-8`), commit `16ad89d`, warm RUN 2,
arms strictly alternating. Transcript `tiled-primary-ab-8.txt`. All six videos
byte-identical, as in every session of this series.

## What replaced what

The old path built a per-tile pixel-bucket CSR: a zeroed
`active_tiles x FINE_TILE**2` table, two passes over every fragment each doing
a binary search over the sorted active-tile list plus an atomic, a `nonzero`,
a second sort over the buckets, three gathers and a scatter — all to hand
`_sort_run` a contiguous run per pixel.

The new path sorts the int32 pixel ids once. `torch.sort` returns the
permutation and the sorted keys together, so run boundaries are a positional
comparison on an immutable array and cost no gather — exactly the shape
`sheet_sort_taichi.key_run_order` already consumes. Five kernels, two device
funcs and the active-tile list went with it; the list existed only to be
binary-searched, and building it was a `nonzero`, a gather, a sort and a host
drain in the *binning* half.

It is also a narrower global key than the reference's: 32 bits of pixel where
`_exact_fragment_order` sorts `(pixel << 32) | depth_bin` and then composes a
second permutation.

## nn at UHD, six alternating pairs

| arm | median | mean | sd |
| --- | ---: | ---: | ---: |
| base | 8.275 s | 8.288 s | 0.063 |
| tile | 8.380 s | 8.373 s | 0.055 |

Paired: **+1.03%, t = 2.88** (ab-7, same protocol: +2.40%, t = 6.1).

## Where it went, stage by stage

This round is the first with the frontend's two halves hooked as profiler
stages, which is why the ledger finally closes. Exclusive-time medians:

| stage | base | tile | delta |
| --- | ---: | ---: | ---: |
| `raster:   - window pairs` | 0.046 | — | |
| `raster:   - tile binning` | — | 0.135 | **+0.089** |
| `raster:   - fragment sort` | 0.270 | — | |
| `raster:   - tile fragment order` | — | 0.181 | **-0.089** |
| `kernel: raster_tri_count` | 0.123 | 0.157 | +0.034 |
| `kernel: raster_tri_write` | 0.109 | 0.121 | +0.012 |
| `raster: sparse discovery` (excl) | 0.285 | 0.290 | +0.005 |

Two things to read off that.

**The ordering replacement worked, and then some.** 0.181 s against the
reference's 0.270 s: the tiled frontend now orders the stream 33% faster than
the frontend it is competing with, rather than paying a premium for the
privilege. That is the whole of the -0.089 s.

**The discovery's unattributed time is gone.** It was 0.285 vs 0.591 in ab-7
and is 0.285 vs 0.290 now. That 0.30 s was never launch overhead or host
round-trips — it was the bucket CSR and the active-tile construction, which
had no stage of their own, which is precisely why three earlier rounds of
tuning aimed at the wrong thing.

## What is left

+0.089 s of binning and +0.046 s of count/write, against -0.089 s of ordering.
The count/write term is the residual packing gap (tile-clipped rows still hand
COUNT about 16% more candidate pixels and 21% more chunks than row spans over
whole bboxes). The binning term is the two-level bin construction itself.

On `nn` that construction certifies **zero** occlusions, so it is pure cost;
on `overdraw` the same construction is what turns 6.49 s into 1.78 s. The
frontend is not paying for a slow implementation any more — it is paying for a
proof that this particular scene's geometry cannot satisfy. Making it cheaper
is not the lever; deciding whether to run it is.

## overdraw at HD, and the series

| arm | ab-1 | ab-4 | ab-5 | ab-7 | ab-8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 6.54 | 6.44 | 6.60 | 6.46 | 6.49 |
| tile | 1.86 (3.5x) | 1.80 (3.6x) | 1.95 (3.4x) | 1.85 (3.5x) | **1.78 (3.6x)** |
| both | 1.44 (4.6x) | 1.46 (4.4x) | 1.54 (4.3x) | — | **1.40 (4.6x)** |

Best of the series on both arms.

And the anchor, across every change:

| commit | change | nn paired delta |
| --- | --- | ---: |
| 2774686 | as implemented | +2.8% |
| 70a048b | row spans inside the tile | +1.58% |
| 67f139d | one class readback | (no effect) |
| d7a9084 | lazy distance proof | (no effect) |
| 2af53f7 | 32x32 fine tiles | (no effect) |
| 16ad89d | pixel sort instead of bucket CSR | **+1.03%** |
