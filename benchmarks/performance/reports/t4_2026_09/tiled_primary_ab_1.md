# Tiled primary discovery and simple interiors: T4 A/B

> **Superseded in part by `tiled_primary_ab_2.md`.** The chunk-packing defect
> this report identifies as most of the `nn` regression was fixed; the
> regression fell to about +1.5% and the rest turned out not to be packing.
> The counter readings, the overdraw numbers and the parity results below
> still stand.

`raster_tile_binning` (speed-audit Rank 3) and `raster_simple_interiors`
(Rank 4), both opt-in and both off by default, measured against the branch's
own `base` configuration.

**Run:** Kaggle notebook `algan-tiled-ab-1`, one session, `device: cuda`,
Tesla T4 (the box has two; the render used one, the other idle).
Branch `codex/renderer-ranks-3-4` at `2774686`, `ALGAN_VIDEO_ENCODER=software`,
`libx264 -preset ultrafast`. Raw transcript: `tiled-primary-ab-1.txt`.

All fifteen arms ran in **one session**, interleaved ABBA per scene, and every
number below is `profile_scene`'s **warm RUN 2**. Per-step `seconds` in
`results.json` is whole-script wall clock including each arm's cold JIT and is
not a measurement (`gpu_harnesses.md`, "Reading the numbers").

Each arm also prints a `DIAG` line with the coverage counters, so a timing is
never read without knowing whether the feature did anything.

## Output parity

Every arm of a scene produced a **byte-identical** video:
`5d3b3e7f35f420f7` for all four `nn` arms, `00706585c125bec5` for all four
`overdraw` arms. `num_fragments` and `num_sheets` are also identical across
arms of a scene, so the pre-emission occlusion rejection removes exactly what
the late opaque-prefix truncation would have removed and nothing else.

## `nn` — the standing anchor (`nn_scene_UHD`'s scene, 3840x2160, shadows off)

| arm | run 1 | run 2 | median | vs base | GPU peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 8.23 s | 8.30 s | **8.27 s** | — | 7998 MB |
| tile | 8.50 s | 8.50 s | **8.50 s** | **+2.8%** | 7998 MB |
| simple | 8.31 s | 8.29 s | **8.30 s** | +0.4% | 7998 MB |
| both | 8.63 s | 8.61 s | **8.62 s** | **+4.3%** | 7998 MB |

Within-arm spread is at most 0.07 s (0.8%), so +2.8% and +4.3% are readings,
not noise.

**Both mechanisms are inert on this scene.** Summed over 18 coverage windows:

```
tile_candidates 47,582,524   tile_bbox_rejected 35,913,338   tile_occluded 0
num_covered     34,878,832   num_simple_pixels           0
```

Nothing is occlusion-rejected and no pixel is certified simple, on 25,082
triangles and 34.9 M covered pixels. The reasons are structural, not a tuning
accident (attributed on CPU at LD with the same counters):

* 22,360 of 25,082 triangles are **closed shells**, which
  `interior_fragment_proofs` rejects outright (`closed == 0`), and only 8,201
  of 419,012 fragments even reach the geometry test (the rest fail
  `cov == 1.0` or the full sample mask). A pixel is simple only if *every*
  one of its ~12 layers passes, so the joint probability is nil.
* `_rect_proof`'s distance interval is too loose to close on geometry this
  size: it returned a finite `far` for **58 of 93,070** tile candidates at LD,
  versus 10,416 of 20,088 on the large-quad scene below. Without a finite
  `near` the occlusion test cannot fire, whatever the material proof says —
  and the material proof itself is fine here (13,640 of 25,082 certified).

So the anchor arm measures overhead with the feature switched on in name only.

**The overhead is not the proof kernels.** Exclusive times, both repeats:

| stage | base | tile |
| --- | ---: | ---: |
| `kernel: raster_tri_count` | 0.115 / 0.133 | 0.225 / 0.250 |
| `kernel: raster_tri_write` | 0.088 / 0.109 | 0.150 / 0.158 |
| `raster:   - fragment sort` | 0.269 / 0.280 | — (replaced) |
| `raster: sparse discovery` (host) | 0.264 / 0.279 | 0.512 / 0.523 |
| new tile kernels, all rows | — | ~0.19 |

Tile binning removes a 0.27 s global sort and pays for it three times over:

* **The count and write passes get ~90% and ~55% more expensive.** Clipping a
  primitive's bbox at every 16x16 tile boundary produces more partial
  `raster_chunk`s than the reference's `raster_span_candidates` row-span
  candidates do — the same candidate pixels in worse-packed chunks. This is
  the largest single item and it is a consequence of the tiling geometry, not
  of the proofs.
* **Host time in the discovery doubles**, +0.25 s, matching the ~18 extra host
  round-trips per coverage window the tiled frontend adds (eight `_prefix`
  `.item()`s, five `.nonzero()`s, a `.cpu().tolist()` of the stats counters, a
  `unique_consecutive` and two `argsort`s) — the direction Rank **2** of the
  same audit asks the renderer to move away from.
* The proof kernels themselves are the cheapest part: `primary_pixel_order`
  0.084 s, `fine_write` 0.030 s, `candidate_proofs` 0.016 s for 47.6 M
  candidates, `opaque_material_proofs` 0.004 s.

Treat the per-stage `incl`/`excl` split as indicative only: launches are async,
so a stage's inclusive time can contain the queue drained by a later sync.
The end-to-end warm times are the measurement.

## `overdraw` — 16 frame-filling flat opaque quads (1920x1080)

The audit's "heavy opaque overlap behind a front surface" workload, and the
only shape found so far on which either switch certifies anything. Built from
open `TriangleMesh` quads, and the obvious alternatives all fail a gate: a
`Square` is a circuit (never an occluder, never occluded), and a `Cube`
defaults to opacity 0.75 so `tri_frame_opaque` is false for it (measured: 0 of
24 faces) *and* it is a closed shell, which `interior_fragment_proofs` rejects
outright. Getting this wrong is silent — the arm runs, the counters read zero,
and the timing looks like a neutral result.

| arm | run 1 | run 2 | median | vs base | GPU peak | windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 6.48 s | 6.59 s | **6.54 s** | — | 14403 MB | 10 |
| tile | 1.92 s | 1.80 s | **1.86 s** | **3.5x faster** | 9957 MB | 8 |
| simple | 6.42 s | — | **6.42 s** | 1.02x | 14403 MB | 10 |
| both | 1.44 s | 1.43 s | **1.44 s** | **4.6x faster** | 7531 MB | 8 |

```
tile_candidates 7,833,600   tile_bbox_rejected 3,851,442   tile_occluded 3,673,938
num_covered    62,208,000   num_simple_pixels 62,141,040   (99.9%)
```

Rank 3 is the win here, and it is a large one: 3.5x end-to-end, and a **31%
lower GPU peak** (14.4 GB → 10.0 GB) because the fragments it never emits were
the discovery peak. With both switches the peak halves to 7.5 GB and the batch
fits in 8 windows instead of 10.

The whole of it lands in two stages, exclusive time:

| stage | base | tile |
| --- | ---: | ---: |
| `kernel: raster_tri_write` | 2.26 s | 0.15 s |
| `raster:   - fragment sort` | 1.83 s | — (0.003 s in `primary_pixel_order`) |

4.1 s of a 6.5 s render was writing fragments the opaque-prefix truncation was
about to delete and globally sorting them on the way. That is the audit's
thesis stated in one table — and note that the same tile geometry that costs
90% more count time on `nn` costs 91% *less* here, because the candidates it
rejects are whole tiles rather than slivers.

Rank 4 is the surprise. It certifies **99.9% of pixels** and cuts
`compact_sheets` from 0.50 s to 0.16 s — and buys 2% end-to-end, because
compaction was 8% of this render. It is worth 23% (1.86 s → 1.44 s) only once
Rank 3 has already removed the occluded stream and compaction is a large share
of what is left.

## Reading

* **Rank 3 works, on the geometry it was designed for.** Large opaque
  occluders, few layers of them, and a big enough triangle for the interval
  arithmetic to close: 3.5x and half the VRAM. Nothing else measured gets near
  that.
* **Rank 3 costs 2.8% where it certifies nothing**, and none of that cost is
  the proofs. Two thirds is chunk packing (tile-clipped boxes vs. row spans)
  and one third is host round-trips, so both are addressable without touching
  the proof arithmetic: keep counts and offsets on the device, and let a
  candidate that owns a whole tile emit one well-packed run rather than
  ceil(area/32) partial chunks per tile.
* **Rank 4 is not, on this evidence, worth promoting on its own.** Its best
  measured standalone result is +2% on a scene where it certifies 99.9% of
  pixels; on the anchor it certifies nothing and costs 0.4%. What it is
  really measuring is that general sheet compaction is not where the warm time
  goes.
* **Neither should be defaulted on.** They are correctly shipped off.
* **The counters are the deliverable.** Any future promotion argument has to
  quote `tile_occluded` and `num_simple_pixels` for the scene it is arguing
  about, because a neutral timing and an inert feature look identical without
  them.

## Gap in this round

`DIAG` records tile candidates but not the **emitted `raster_chunk` pair
count**, and the reference frontend records neither. So "35.9 M of 47.6 M tile
candidates rejected" cannot be compared against the row-span candidates
`raster_span_candidates` (on by default) would have produced for the same
frame — the tiled arm is partly rejecting work that tiling itself created.
Both frontends should report pairs and candidate pixels before the next round.
