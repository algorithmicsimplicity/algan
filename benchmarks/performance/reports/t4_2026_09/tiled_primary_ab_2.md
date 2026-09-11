# Tiled primary: fixing the chunk packing

Follow-up to `tiled_primary_ab_1.md`, which measured the opt-in
`raster_tile_binning` at **+2.8%** on the anchor scene and attributed most of
that to chunk packing rather than to the proofs. This is what fixing the
packing did, and what it did not.

**Runs:** Kaggle T4, one session each, warm RUN 2 only, arms strictly
alternating base/tile within the session. Transcripts
`tiled-primary-ab-{2..6}.txt`. Every arm of every session rendered a
byte-identical video (`5d3b3e7f35f420f7` for nn, `00706585c125bec5` for
overdraw), across five sessions and four commits.

## What the packing defect was

Both frontends hand the COUNT pass `raster_chunk`-sized chunks of a
`bw x bh` box. The reference has `raster_span_candidates`: a large box is
walked row by row and each row clipped to the projection's own x-extent. The
tiled frontend had nothing equivalent — it clipped the primitive's bbox to the
tile and emitted that rectangle — so a thin diagonal that genuinely crosses a
tile handed COUNT the whole tile.

Candidate pixels the COUNT pass tests, one HD frame (scene-deterministic, so
this is a measurement and not a timing):

| scene | reference | tiled, before | tiled, after |
| --- | ---: | ---: | ---: |
| nn | 2.93 M | **11.42 M** | **3.39 M** |
| overdraw | 33.25 M | 33.25 M (boxes are already full) | **2.71 M** |

Always taking rows is wrong in the other direction: a tile a large triangle
covers outright packs perfectly at 32 pixels a chunk, and row-splitting it
doubles the chunk count for nothing (the overdraw scene went 3.08s -> 3.99s
warm on CPU that way). The COUNT pass now walks the rows, compares their
pixels against the box's and takes rows only where they at least halve them.

## What it bought, in the stage table

`nn` at UHD, exclusive-time medians over 8 alternating pairs (ab-6):

| stage | base | tiled, before (ab-1) | tiled, after (ab-6) |
| --- | ---: | ---: | ---: |
| `kernel: raster_tri_count` | 0.129 s | 0.238 s (+92%) | 0.160 s (+24%) |
| `kernel: raster_tri_write` | 0.104 s | 0.154 s (+56%) | 0.122 s (+17%) |
| `raster:   - fragment sort` | 0.267 s | — (replaced) | — (replaced) |
| `raster: sparse discovery` (excl) | 0.294 s | 0.512 s | 0.590 s |

The packing gap closed from +0.16 s to +0.05 s, which is what it was supposed
to do. The other line did not move, and that is the answer.

## What it did not buy

End-to-end on `nn` at UHD, paired within each session:

| session | commit | n pairs | base spread | paired delta | t |
| --- | --- | ---: | ---: | ---: | ---: |
| ab-1 (before) | 2774686 | 2 | 0.8% | **+2.8%** | — |
| ab-3 (spans) | 70a048b | 6 | 0.5% | **+1.58%** | 7.9 |
| ab-4 (+ one readback) | 67f139d | 6 | 1.6% | **+2.67%** | 5.5 |
| ab-5 (+ lazy near) | d7a9084 | 6 | 0.8% | **-0.46%** | -0.3 |
| ab-6 (tie-break) | d7a9084 | 8 | 0.5% | **+2.05%** | 3.3 |

**Tile binning is still slower than the reference frontend on this scene** —
around +1.5% averaged over the four post-fix sessions, against a session-to-
session spread of about the same size. Three of four sessions put it clearly
above zero. It is no longer the clean, repeatable +2.8% it was, and it is not
parity either.

Two later attempts to close the rest failed to move anything, and both are
worth recording because they rule out the obvious explanations:

* **Host round-trips are not it.** Grouping the emitted pairs by class with
  one transfer instead of four `(classes == cls).nonzero()` drains (ab-4) left
  the discovery's exclusive time at 0.592 s against 0.569 s the session
  before. Removing four of roughly fifteen round-trips per coverage window was
  not measurable.
* **`candidate_proofs` is not it either.** It was running `_rect_proof` twice
  for every triangle candidate; the second pass certified **0 of 93 070**
  candidates on this scene. Restricting it to candidates that fill their tile,
  and deferring everything else to a `candidate_near` pass that only runs on
  tiles with a certified occluder (ab-5), takes the second proof to zero
  candidates on `nn` — and the discovery excl went 0.592 -> 0.633 s in a
  session whose base arm was itself 9% slower. No effect.

So the residual is spread across the binning pipeline as a whole — bbox
records, the coarse sort and `unique_consecutive`, the fine count/write scans,
the per-pixel bucket CSR and the per-pixel sort — and it is about what the
0.267 s global fragment sort it replaces costs. On a scene where the binning
certifies no occlusion at all, that is a wash plus the extra count/write, and
there is no single term left to remove.

`nn` is a hard case for this architecture rather than a badly tuned one: its
triangles are long thin diagonals whose bboxes span some 63 fine tiles each,
of which 75% are then rejected as `outside`. The tiled frontend spends its
time rejecting work that tiling itself created, and the reference's row spans
had already solved the same problem without the bins.

## The overdraw win is intact

| arm | ab-1 (before) | ab-4 | ab-5 |
| --- | ---: | ---: | ---: |
| base | 6.54 s | 6.44 s | 6.60 s |
| tile | 1.86 s (3.5x) | 1.80 s (3.6x) | 1.95 s (3.4x) |
| both | 1.44 s (4.6x) | 1.46 s (4.4x) | 1.54 s (4.3x) |

The row-span choice is per candidate precisely so this does not regress: on
this scene the tiles a quad covers keep their boxes, and the candidate pixels
fall 33.25 M -> 2.71 M because of the occlusion rejection rather than the
packing.

## Where that leaves the switch

Still off by default, and still right to be. The packing defect is fixed and
was real; what remains is that a two-level bin costs about as much to build as
the global sort it removes, so the switch pays only where its proofs certify
something. `tile_occluded` and `num_simple_pixels` remain the numbers to quote
before arguing otherwise.
