# Tiled primary: does a 32x32 fine tile close the rest?

**No.** It is neutral, on both scenes, while cutting the candidate count by a
factor of three. That is a more useful answer than a win would have been,
because it rules out the remaining obvious suspect.

Follow-up to `tiled_primary_ab_2.md`, which left `raster_tile_binning` at about
+1.5% on the anchor scene after the chunk-packing fix.

**Run:** Kaggle T4, one session (`tiled-ab-7`), commit `2af53f7`, warm RUN 2,
arms strictly rotating base / f16 / f32. Transcript
`tiled-primary-ab-7.txt`. All six videos byte-identical
(`5d3b3e7f35f420f7` for nn, `00706585c125bec5` for overdraw), so the tile size
is confirmed to be a work-partitioning knob and nothing else.

## nn at UHD, five rounds

| arm | median | mean | sd | vs base | t |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 8.350 s | 8.340 s | 0.085 | — | — |
| tile, 16x16 | 8.530 s | 8.540 s | 0.030 | **+2.40%** | 6.1 |
| tile, 32x32 | 8.520 s | 8.510 s | 0.048 | **+2.04%** | 3.6 |

f32 against f16, paired: **-0.35%, t = -1.11** — not a reading, and two of the
five paired deltas are not even negative.

What the bigger tile did do is large and unambiguous:

| counter (nn, whole render) | 16x16 | 32x32 |
| --- | ---: | ---: |
| `tile_candidates` | 47,582,524 | **15,897,582** (-67%) |
| `tile_bbox_rejected` | 35,913,338 | 9,007,570 |
| candidate pixels handed to COUNT (1 HD frame) | 3.39 M | **3.17 M** |
| chunks (1 HD frame) | 418,666 | **380,516** |

So a 3x reduction in binning work, a 9% reduction in chunks and a 6% reduction
in candidate pixels bought nothing measurable end to end.

## overdraw at HD

| arm | warm | `tile_candidates` | `tile_occluded` | fragments |
| --- | ---: | ---: | ---: | ---: |
| base | 6.46 s | — | — | 62,456,400 |
| tile, 16x16 | 1.85 s | 7,833,600 | 3,673,938 | 62,456,400 |
| tile, 32x32 | 1.84 s | 1,958,400 | 905,466 | 62,456,400 |

Identical. The occlusion count falls with the tile count because there are 4x
fewer tiles to certify, not because fewer fragments are rejected — the emitted
fragment count is the same to the unit. A CPU measurement had suggested 32x32
would *lose* here (its candidate pixels go 2.71 M -> 3.35 M); on the GPU that
does not show up at all.

## What this rules out

Four things have now been tried against the ~2% residual on `nn`, three of
them negative, and together they narrow it a lot:

| change | effect on the counters | effect on warm time |
| --- | --- | --- |
| row spans inside the tile | candidate pixels 11.4 M -> 3.39 M | **+2.8% -> ~+1.5%** |
| one class readback instead of four | 5 host round-trips -> 1 | none measurable |
| `_rect_proof` only where it can certify | second proof on ~61k candidates -> 0 | none measurable |
| 32x32 fine tiles | tile candidates 47.6 M -> 15.9 M | none measurable |

The residual is therefore not the chunk packing (fixed), not host round-trips,
not the proof kernels, and not the number of tile candidates. What is left is
the per-fragment and per-covered-pixel machinery that replaces the reference's
one global radix sort, and the stage table says so directly: the two largest
tiled-only kernel rows are `primary_pixel_order` (0.082 s) and
`scatter_fragment_order` (0.027 s), and **both are identical at f16 and f32**
(0.082/0.083 and 0.027/0.029) because they scale with covered pixels and
fragments, not with tiles. `fragment_bucket_counts` and
`scatter_fragment_order` each walk all 154 M fragments doing a binary search
over the active-tile list plus an atomic; `primary_pixel_order` runs an
insertion/heap sort per covered pixel over 34.9 M of them.

Against that the tiled path removes `raster: - fragment sort` (0.267 s) and
`pair_expand_count`/`pair_expand_write` (0.048 s). It is close, and on this
scene it comes out behind.

## Recommendation

Leave the default at 16x16. Nothing measured prefers 32x32, it costs 24% more
pixel-bucket memory on a sparse scene (2.8 MB -> 3.5 MB per HD frame window),
and a bigger tile makes `_rect_proof`'s intervals looser, which is the
mechanism the occlusion proof depends on — the overdraw fixture still
certifies at 32x32, but nothing measured says that holds for less extreme
geometry.

The knob is worth keeping: it is two frozen settings fields, it made this
experiment a one-line change, and the next scene may not be `nn`.
